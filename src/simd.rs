//! Explicit SIMD kernels (docs/plan.md Phase 2), **default-on** via runtime dispatch.
//!
//! Discipline (the scalar-twin rule):
//! - Every kernel here has a scalar twin in this module; the twin is the
//!   reference and is never deleted.
//! - A unit test drives both over randomized inputs and asserts **bit equality**
//!   — a fast wrong answer is a regression, not a win.
//! - x86-64 uses AVX2 when the CPU has it (checked once, cached); aarch64 uses
//!   NEON unconditionally (baseline for the architecture). Everything else gets
//!   the scalar twin. No nightly features; `core::arch` intrinsics only.
//!
//! Not served by these kernels, by design: `warp8x8` (a different filter per
//! pixel), `ext_warp` (4x4 cells, about 1% of 8-tap samples on real streams) and
//! `put_bilin_win` (a 2-tap bilinear filter).
//!
//! Shapes covered (the Phase 2 map, cheapest-first), each over a block of rows:
//! - 8-tap horizontal FIR (`fir8_h_u16`): the H pass of motion compensation.
//! - 8-tap vertical FIR (`fir8_v` over the row-major mid buffer, `fir8_v_u16`
//!   over a reference plane): the V pass. Summing `mid[(y+k)*w + x]` across `k`
//!   is 8 contiguous-row loads, so lanewise it is one multiply-add per output
//!   instead of eight.
//! - Elementwise compound blends (`avg`, `w_avg`, `mask`): two prep-precision
//!   buffers in, one clamped block out.
//!
//! Samples are `i32` today (Phase 1 narrowing to u8/u16 is future work), so
//! AVX2 processes 8 lanes (two four-wide rows per register) and NEON 4. The
//! kernels work on blocks of rows, so dispatch is paid per block.

#![allow(clippy::missing_safety_doc)]
// The kernels are one screen of intrinsics each; a per-call unsafe block per
// intrinsic would be pure noise. Each dispatcher documents the actual safety
// argument (feature verified + bounds asserted) at its single unsafe call.
#![allow(unsafe_op_in_unsafe_fn)]

/// Cached CPU capability: 0 = scalar, 1 = AVX2 (x86-64) / NEON (aarch64).
#[inline]
pub fn simd_level() -> u32 {
    use std::sync::OnceLock;
    static LEVEL: OnceLock<u32> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        if crate::env_is_set!("RUSTY_AV2D_NOSIMD") {
            return 0;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return 1;
            }
            0
        }
        #[cfg(target_arch = "aarch64")]
        {
            1 // NEON is baseline on aarch64
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            0
        }
    })
}

/// `(rows - 1) * stride + len`: the reach of a block of `rows` rows, `stride` apart, each `len`
/// long. Checked arithmetic: a wrapped sum would let the kernels read out of bounds.
#[inline]
fn reach(rows: usize, stride: usize, len: usize) -> Option<usize> {
    rows.checked_sub(1)?.checked_mul(stride)?.checked_add(len)
}

#[inline]
fn fits(rows: usize, stride: usize, len: usize, have: usize) -> bool {
    reach(rows, stride, len).is_some_and(|need| need <= have)
}

// ---------------------------------------------------------------------------
// Every kernel works on a BLOCK of rows, so the `#[target_feature]` boundary, its register saves
// and the tap broadcasts are paid once per block rather than once per row. Measured per row, that
// fixed cost made a four-wide row slower through the kernel than through the scalar twin.
// Four-wide blocks also pack two rows into each 256-bit register.
// ---------------------------------------------------------------------------

/// Horizontal 8-tap: `out[r*os + x] = (sum_k taps[k]*src[r*ss + x + k] + rnd) >> sh`, `x < w`,
/// `r < rows`. A row reads `w + 7` samples; `ss` may be 0 to filter one source row repeatedly.
/// `clamp_max < 0` = no clamp.
#[allow(clippy::too_many_arguments)]
pub fn fir8_h_u16(src: &[u16], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    if rows == 0 || w == 0 {
        return;
    }
    assert!(fits(rows, ss, w + 7, src.len()), "fir8 source too short");
    assert!(fits(rows, os, w, out.len()), "fir8 output too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified by simd_level(); bounds asserted above.
        unsafe { fir8_h_u16_avx2(src, ss, taps, out, os, w, rows, rnd, sh, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        for r in 0..rows {
            // SAFETY: NEON is baseline on aarch64; the row slices are bounds-checked.
            unsafe { fir8_row_u16_neon(&src[r * ss..r * ss + w + 7], taps, &mut out[r * os..r * os + w], rnd, sh, clamp_max) };
        }
        return;
    }
    fir8_h_u16_scalar(src, ss, taps, out, os, w, rows, rnd, sh, clamp_max);
}

/// Scalar twin of [`fir8_h_u16`].
#[allow(clippy::too_many_arguments)]
pub fn fir8_h_u16_scalar(src: &[u16], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    for r in 0..rows {
        fir8_row_u16_scalar(&src[r * ss..r * ss + w + 7], taps, &mut out[r * os..r * os + w], rnd, sh, clamp_max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fir8_h_u16_avx2(src: &[u16], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let vr = _mm256_set1_epi32(rnd);
    let sh_v = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let t: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi32(taps[k] as i32));
    let (sp, op) = (src.as_ptr(), out.as_mut_ptr());
    let mut r = 0usize;
    if w == 4 {
        // Two rows per register: row r in the low four lanes, row r + 1 in the high four.
        while r + 2 <= rows {
            let (p0, p1) = (sp.add(r * ss), sp.add((r + 1) * ss));
            let mut acc = vr;
            for (k, tk) in t.iter().enumerate() {
                let s = _mm_unpacklo_epi64(_mm_loadl_epi64(p0.add(k) as *const __m128i), _mm_loadl_epi64(p1.add(k) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_cvtepu16_epi32(s), *tk));
            }
            let mut v = _mm256_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
            }
            _mm_storeu_si128(op.add(r * os) as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeu_si128(op.add((r + 1) * os) as *mut __m128i, _mm256_extracti128_si256::<1>(v));
            r += 2;
        }
    }
    while r < rows {
        let (p, o) = (sp.add(r * ss), op.add(r * os));
        let mut x = 0usize;
        while x + 8 <= w {
            let mut acc = vr;
            for (k, tk) in t.iter().enumerate() {
                let s = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.add(x + k) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tk));
            }
            let mut v = _mm256_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
            }
            _mm256_storeu_si256(o.add(x) as *mut __m256i, v);
            x += 8;
        }
        if x + 4 <= w {
            let mut acc = _mm256_castsi256_si128(vr);
            for (k, tk) in t.iter().enumerate() {
                let s = _mm_cvtepu16_epi32(_mm_loadl_epi64(p.add(x + k) as *const __m128i));
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s, _mm256_castsi256_si128(*tk)));
            }
            let mut v = _mm_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm_min_epi32(_mm_max_epi32(v, _mm256_castsi256_si128(zero)), _mm256_castsi256_si128(vmax));
            }
            _mm_storeu_si128(o.add(x) as *mut __m128i, v);
            x += 4;
        }
        if x < w {
            let s = core::slice::from_raw_parts(p.add(x), w - x + 7);
            let d = core::slice::from_raw_parts_mut(o.add(x), w - x);
            fir8_row_u16_scalar(s, taps, d, rnd, sh, clamp_max);
        }
        r += 1;
    }
}

/// Vertical 8-tap over an `i32` buffer: `out[r*os + x] = (sum_k taps[k]*src[(r+k)*ss + x] + rnd)
/// >> sh`, `x < w`, `r < rows`. `clamp_max < 0` = no clamp.
#[allow(clippy::too_many_arguments)]
pub fn fir8_v(src: &[i32], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    if rows == 0 || w == 0 {
        return;
    }
    assert!(rows.checked_add(7).is_some_and(|r| fits(r, ss, w, src.len())), "fir8 source too short");
    assert!(fits(rows, os, w, out.len()), "fir8 output too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified by simd_level(); bounds asserted above.
        unsafe { fir8_v_avx2(src, ss, taps, out, os, w, rows, rnd, sh, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        for r in 0..rows {
            // SAFETY: NEON is baseline on aarch64; the row slices are bounds-checked.
            unsafe { fir8_col_neon(&src[r * ss..(r + 7) * ss + w], ss, taps, &mut out[r * os..r * os + w], rnd, sh, clamp_max) };
        }
        return;
    }
    fir8_v_scalar(src, ss, taps, out, os, w, rows, rnd, sh, clamp_max);
}

/// Scalar twin of [`fir8_v`].
#[allow(clippy::too_many_arguments)]
pub fn fir8_v_scalar(src: &[i32], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    for r in 0..rows {
        fir8_col_scalar(&src[r * ss..(r + 7) * ss + w], ss, taps, &mut out[r * os..r * os + w], rnd, sh, clamp_max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fir8_v_avx2(src: &[i32], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let vr = _mm256_set1_epi32(rnd);
    let sh_v = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let t: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi32(taps[k] as i32));
    let (sp, op) = (src.as_ptr(), out.as_mut_ptr());
    let mut r = 0usize;
    if w == 4 {
        while r + 2 <= rows {
            let mut acc = vr;
            for (k, tk) in t.iter().enumerate() {
                let lo = _mm_loadu_si128(sp.add((r + k) * ss) as *const __m128i);
                let hi = _mm_loadu_si128(sp.add((r + 1 + k) * ss) as *const __m128i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set_m128i(hi, lo), *tk));
            }
            let mut v = _mm256_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
            }
            _mm_storeu_si128(op.add(r * os) as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeu_si128(op.add((r + 1) * os) as *mut __m128i, _mm256_extracti128_si256::<1>(v));
            r += 2;
        }
    }
    while r < rows {
        let (p, o) = (sp.add(r * ss), op.add(r * os));
        let mut x = 0usize;
        while x + 8 <= w {
            let mut acc = vr;
            for (k, tk) in t.iter().enumerate() {
                let s = _mm256_loadu_si256(p.add(k * ss + x) as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tk));
            }
            let mut v = _mm256_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
            }
            _mm256_storeu_si256(o.add(x) as *mut __m256i, v);
            x += 8;
        }
        if x + 4 <= w {
            let mut acc = _mm256_castsi256_si128(vr);
            for (k, tk) in t.iter().enumerate() {
                let s = _mm_loadu_si128(p.add(k * ss + x) as *const __m128i);
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s, _mm256_castsi256_si128(*tk)));
            }
            let mut v = _mm_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm_min_epi32(_mm_max_epi32(v, _mm256_castsi256_si128(zero)), _mm256_castsi256_si128(vmax));
            }
            _mm_storeu_si128(o.add(x) as *mut __m128i, v);
            x += 4;
        }
        if x < w {
            let s = core::slice::from_raw_parts(p.add(x), 7 * ss + (w - x));
            let d = core::slice::from_raw_parts_mut(o.add(x), w - x);
            fir8_col_scalar(s, ss, taps, d, rnd, sh, clamp_max);
        }
        r += 1;
    }
}

/// Vertical 8-tap over a `u16` plane; otherwise [`fir8_v`].
#[allow(clippy::too_many_arguments)]
pub fn fir8_v_u16(src: &[u16], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    if rows == 0 || w == 0 {
        return;
    }
    assert!(rows.checked_add(7).is_some_and(|r| fits(r, ss, w, src.len())), "fir8 source too short");
    assert!(fits(rows, os, w, out.len()), "fir8 output too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified by simd_level(); bounds asserted above.
        unsafe { fir8_v_u16_avx2(src, ss, taps, out, os, w, rows, rnd, sh, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        for r in 0..rows {
            // SAFETY: NEON is baseline on aarch64; the row slices are bounds-checked.
            unsafe { fir8_col_u16_neon(&src[r * ss..(r + 7) * ss + w], ss, taps, &mut out[r * os..r * os + w], rnd, sh, clamp_max) };
        }
        return;
    }
    fir8_v_u16_scalar(src, ss, taps, out, os, w, rows, rnd, sh, clamp_max);
}

/// Scalar twin of [`fir8_v_u16`].
#[allow(clippy::too_many_arguments)]
pub fn fir8_v_u16_scalar(src: &[u16], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    for r in 0..rows {
        fir8_col_u16_scalar(&src[r * ss..(r + 7) * ss + w], ss, taps, &mut out[r * os..r * os + w], rnd, sh, clamp_max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fir8_v_u16_avx2(src: &[u16], ss: usize, taps: &[i8; 8], out: &mut [i32], os: usize, w: usize, rows: usize, rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let vr = _mm256_set1_epi32(rnd);
    let sh_v = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let t: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi32(taps[k] as i32));
    let (sp, op) = (src.as_ptr(), out.as_mut_ptr());
    let mut r = 0usize;
    if w == 4 {
        while r + 2 <= rows {
            let mut acc = vr;
            for (k, tk) in t.iter().enumerate() {
                let lo = _mm_cvtepu16_epi32(_mm_loadl_epi64(sp.add((r + k) * ss) as *const __m128i));
                let hi = _mm_cvtepu16_epi32(_mm_loadl_epi64(sp.add((r + 1 + k) * ss) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set_m128i(hi, lo), *tk));
            }
            let mut v = _mm256_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
            }
            _mm_storeu_si128(op.add(r * os) as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeu_si128(op.add((r + 1) * os) as *mut __m128i, _mm256_extracti128_si256::<1>(v));
            r += 2;
        }
    }
    while r < rows {
        let (p, o) = (sp.add(r * ss), op.add(r * os));
        let mut x = 0usize;
        while x + 8 <= w {
            let mut acc = vr;
            for (k, tk) in t.iter().enumerate() {
                let s = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.add(k * ss + x) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tk));
            }
            let mut v = _mm256_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
            }
            _mm256_storeu_si256(o.add(x) as *mut __m256i, v);
            x += 8;
        }
        if x + 4 <= w {
            let mut acc = _mm256_castsi256_si128(vr);
            for (k, tk) in t.iter().enumerate() {
                let s = _mm_cvtepu16_epi32(_mm_loadl_epi64(p.add(k * ss + x) as *const __m128i));
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s, _mm256_castsi256_si128(*tk)));
            }
            let mut v = _mm_sra_epi32(acc, sh_v);
            if clamp_max >= 0 {
                v = _mm_min_epi32(_mm_max_epi32(v, _mm256_castsi256_si128(zero)), _mm256_castsi256_si128(vmax));
            }
            _mm_storeu_si128(o.add(x) as *mut __m128i, v);
            x += 4;
        }
        if x < w {
            let s = core::slice::from_raw_parts(p.add(x), 7 * ss + (w - x));
            let d = core::slice::from_raw_parts_mut(o.add(x), w - x);
            fir8_col_u16_scalar(s, ss, taps, d, rnd, sh, clamp_max);
        }
        r += 1;
    }
}

// ---------------------------------------------------------------------------
// Compound blends over a block: `a`, `b` (and the mask) rows are `ps` apart, output rows `os`.
// ---------------------------------------------------------------------------

/// The blend a block kernel applies, one variant per compound mode.
#[derive(Clone, Copy)]
enum Blend<'a> {
    /// `(a + b + 16) >> 5`
    Avg,
    /// `(a*wt + b*(16-wt) + 128) >> 8`
    WAvg(i32),
    /// `(a*m + b*(64-m) + 512) >> 10`, `m` rows `ps` apart like `a` and `b`
    Mask(&'a [u8]),
}

/// Compound average of a block, clamped to `0..=clamp_max`.
#[allow(clippy::too_many_arguments)]
pub fn avg(a: &[i32], b: &[i32], ps: usize, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    blend(a, b, ps, Blend::Avg, out, os, w, rows, clamp_max);
}

/// Weighted compound average of a block, clamped to `0..=clamp_max`.
#[allow(clippy::too_many_arguments)]
pub fn w_avg(a: &[i32], b: &[i32], ps: usize, out: &mut [i32], os: usize, w: usize, rows: usize, wt: i32, clamp_max: i32) {
    blend(a, b, ps, Blend::WAvg(wt), out, os, w, rows, clamp_max);
}

/// Per-pixel mask blend of a block, clamped to `0..=clamp_max`. `m` has the layout of `a`.
#[allow(clippy::too_many_arguments)]
pub fn mask(a: &[i32], b: &[i32], m: &[u8], ps: usize, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    assert!(rows == 0 || w == 0 || fits(rows, ps, w, m.len()), "blend input too short");
    blend(a, b, ps, Blend::Mask(m), out, os, w, rows, clamp_max);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn blend(a: &[i32], b: &[i32], ps: usize, kind: Blend, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    if rows == 0 || w == 0 {
        return;
    }
    assert!(fits(rows, ps, w, a.len()) && fits(rows, ps, w, b.len()), "blend input too short");
    assert!(fits(rows, os, w, out.len()), "blend output too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified by simd_level(); bounds asserted above and, for the mask, in
        // `mask`.
        unsafe { blend_avx2(a, b, ps, kind, out, os, w, rows, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        for r in 0..rows {
            let (ra, rb, ro) = (&a[r * ps..r * ps + w], &b[r * ps..r * ps + w], &mut out[r * os..r * os + w]);
            // SAFETY: NEON is baseline on aarch64; the row slices are bounds-checked.
            unsafe {
                match kind {
                    Blend::Avg => avg_row_neon(ra, rb, ro, clamp_max),
                    Blend::WAvg(wt) => w_avg_row_neon(ra, rb, ro, wt, clamp_max),
                    Blend::Mask(m) => mask_row_neon(ra, rb, &m[r * ps..r * ps + w], ro, clamp_max),
                }
            }
        }
        return;
    }
    blend_scalar(a, b, ps, kind, out, os, w, rows, clamp_max);
}

/// Scalar twin of the blend kernels.
#[allow(clippy::too_many_arguments)]
fn blend_scalar(a: &[i32], b: &[i32], ps: usize, kind: Blend, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    for r in 0..rows {
        let (ra, rb, ro) = (&a[r * ps..r * ps + w], &b[r * ps..r * ps + w], &mut out[r * os..r * os + w]);
        match kind {
            Blend::Avg => avg_row_scalar(ra, rb, ro, clamp_max),
            Blend::WAvg(wt) => w_avg_row_scalar(ra, rb, ro, wt, clamp_max),
            Blend::Mask(m) => mask_row_scalar(ra, rb, &m[r * ps..r * ps + w], ro, clamp_max),
        }
    }
}

/// Scalar twin of [`avg`].
#[allow(clippy::too_many_arguments)]
pub fn avg_scalar(a: &[i32], b: &[i32], ps: usize, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    blend_scalar(a, b, ps, Blend::Avg, out, os, w, rows, clamp_max);
}

/// Scalar twin of [`w_avg`].
#[allow(clippy::too_many_arguments)]
pub fn w_avg_scalar(a: &[i32], b: &[i32], ps: usize, out: &mut [i32], os: usize, w: usize, rows: usize, wt: i32, clamp_max: i32) {
    blend_scalar(a, b, ps, Blend::WAvg(wt), out, os, w, rows, clamp_max);
}

/// Scalar twin of [`mask`].
#[allow(clippy::too_many_arguments)]
pub fn mask_scalar(a: &[i32], b: &[i32], m: &[u8], ps: usize, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    blend_scalar(a, b, ps, Blend::Mask(m), out, os, w, rows, clamp_max);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn blend_avx2(a: &[i32], b: &[i32], ps: usize, kind: Blend, out: &mut [i32], os: usize, w: usize, rows: usize, clamp_max: i32) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let (ap, bp, op) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
    // The blend of one register of `a`, `b` and (for a mask) the matching weights.
    let (shift, vr, vw, vw2) = match kind {
        Blend::Avg => (5, _mm256_set1_epi32(16), zero, zero),
        Blend::WAvg(wt) => (8, _mm256_set1_epi32(128), _mm256_set1_epi32(wt), _mm256_set1_epi32(16 - wt)),
        Blend::Mask(_) => (10, _mm256_set1_epi32(512), zero, _mm256_set1_epi32(64)),
    };
    let sh_v = _mm_cvtsi32_si128(shift);
    let mp = match kind {
        Blend::Mask(m) => m.as_ptr(),
        _ => core::ptr::null(),
    };
    let mix = |va: __m256i, vb: __m256i, vm: __m256i| -> __m256i {
        let s = match kind {
            Blend::Avg => _mm256_add_epi32(_mm256_add_epi32(va, vb), vr),
            Blend::WAvg(_) => _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(va, vw), _mm256_mullo_epi32(vb, vw2)), vr),
            Blend::Mask(_) => _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(va, vm), _mm256_mullo_epi32(vb, _mm256_sub_epi32(vw2, vm))),
                vr,
            ),
        };
        _mm256_min_epi32(_mm256_max_epi32(_mm256_sra_epi32(s, sh_v), zero), vmax)
    };
    let is_mask = matches!(kind, Blend::Mask(_));
    let mut r = 0usize;
    if w == 4 {
        while r + 2 <= rows {
            let (i0, i1) = (r * ps, (r + 1) * ps);
            let va = _mm256_set_m128i(_mm_loadu_si128(ap.add(i1) as *const __m128i), _mm_loadu_si128(ap.add(i0) as *const __m128i));
            let vb = _mm256_set_m128i(_mm_loadu_si128(bp.add(i1) as *const __m128i), _mm_loadu_si128(bp.add(i0) as *const __m128i));
            let vm = if is_mask {
                let m0 = _mm_cvtsi32_si128((mp.add(i0) as *const i32).read_unaligned());
                let m1 = _mm_cvtsi32_si128((mp.add(i1) as *const i32).read_unaligned());
                _mm256_cvtepu8_epi32(_mm_unpacklo_epi32(m0, m1))
            } else {
                zero
            };
            let v = mix(va, vb, vm);
            _mm_storeu_si128(op.add(r * os) as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeu_si128(op.add((r + 1) * os) as *mut __m128i, _mm256_extracti128_si256::<1>(v));
            r += 2;
        }
    }
    while r < rows {
        let (i, o) = (r * ps, op.add(r * os));
        let mut x = 0usize;
        while x + 8 <= w {
            let va = _mm256_loadu_si256(ap.add(i + x) as *const __m256i);
            let vb = _mm256_loadu_si256(bp.add(i + x) as *const __m256i);
            let vm = if is_mask { _mm256_cvtepu8_epi32(_mm_loadl_epi64(mp.add(i + x) as *const __m128i)) } else { zero };
            _mm256_storeu_si256(o.add(x) as *mut __m256i, mix(va, vb, vm));
            x += 8;
        }
        if x + 4 <= w {
            let va = _mm256_castsi128_si256(_mm_loadu_si128(ap.add(i + x) as *const __m128i));
            let vb = _mm256_castsi128_si256(_mm_loadu_si128(bp.add(i + x) as *const __m128i));
            let vm = if is_mask {
                _mm256_cvtepu8_epi32(_mm_cvtsi32_si128((mp.add(i + x) as *const i32).read_unaligned()))
            } else {
                zero
            };
            _mm_storeu_si128(o.add(x) as *mut __m128i, _mm256_castsi256_si128(mix(va, vb, vm)));
            x += 4;
        }
        if x < w {
            let n = w - x;
            let ra = core::slice::from_raw_parts(ap.add(i + x), n);
            let rb = core::slice::from_raw_parts(bp.add(i + x), n);
            let ro = core::slice::from_raw_parts_mut(o.add(x), n);
            match kind {
                Blend::Avg => avg_row_scalar(ra, rb, ro, clamp_max),
                Blend::WAvg(wt) => w_avg_row_scalar(ra, rb, ro, wt, clamp_max),
                Blend::Mask(_) => mask_row_scalar(ra, rb, core::slice::from_raw_parts(mp.add(i + x), n), ro, clamp_max),
            }
        }
        r += 1;
    }
}

// ---------------------------------------------------------------------------
// Row bodies: the scalar twins (the oracle, and the tails of the vector kernels) and the NEON
// rows the block dispatchers loop over. NEON needs no target feature, so they inline.
// ---------------------------------------------------------------------------

/// Vertical 8-tap over a row-major buffer: out[x] = (sum_k taps[k]*src[k*stride+x] + rnd) >> sh.
/// `src` must have `7*stride + out.len()` readable elements from its start.
pub fn fir8_col_scalar(src: &[i32], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let s: i32 = (0..8).map(|k| taps[k] as i32 * src[k * stride + x]).sum();
        let v = (s + rnd) >> sh;
        *o = if clamp_max >= 0 { v.clamp(0, clamp_max) } else { v };
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fir8_col_neon(src: &[i32], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::aarch64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = vdupq_n_s32(rnd);
    let vsh = vdupq_n_s32(-sh);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(clamp_max);
    let t: [int32x4_t; 8] = core::array::from_fn(|k| vdupq_n_s32(taps[k] as i32));
    while x + 4 <= n {
        let mut acc = vr;
        for (k, tk) in t.iter().enumerate() {
            let s = vld1q_s32(src.as_ptr().add(k * stride + x));
            acc = vmlaq_s32(acc, s, *tk);
        }
        let mut v = vshlq_s32(acc, vsh);
        if clamp_max >= 0 {
            v = vminq_s32(vmaxq_s32(v, zero), vmax);
        }
        vst1q_s32(out.as_mut_ptr().add(x), v);
        x += 4;
    }
    if x < n {
        fir8_col_scalar(&src[x..], stride, taps, &mut out[x..], rnd, sh, clamp_max);
    }
}

pub fn fir8_row_u16_scalar(src: &[u16], taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let s0 = &src[x..x + 8];
        let s: i32 = (0..8).map(|k| taps[k] as i32 * s0[k] as i32).sum();
        let v = (s + rnd) >> sh;
        *o = if clamp_max >= 0 { v.clamp(0, clamp_max) } else { v };
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fir8_row_u16_neon(src: &[u16], taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::aarch64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = vdupq_n_s32(rnd);
    let vsh = vdupq_n_s32(-sh);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(clamp_max);
    let t: [int32x4_t; 8] = core::array::from_fn(|k| vdupq_n_s32(taps[k] as i32));
    while x + 4 <= n {
        let mut acc = vr;
        for (k, tk) in t.iter().enumerate() {
            let s16 = vld1_u16(src.as_ptr().add(x + k));
            let s = vreinterpretq_s32_u32(vmovl_u16(s16));
            acc = vmlaq_s32(acc, s, *tk);
        }
        let mut v = vshlq_s32(acc, vsh);
        if clamp_max >= 0 {
            v = vminq_s32(vmaxq_s32(v, zero), vmax);
        }
        vst1q_s32(out.as_mut_ptr().add(x), v);
        x += 4;
    }
    if x < n {
        fir8_row_u16_scalar(&src[x..], taps, &mut out[x..], rnd, sh, clamp_max);
    }
}

pub fn fir8_col_u16_scalar(src: &[u16], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let s: i32 = (0..8).map(|k| taps[k] as i32 * src[k * stride + x] as i32).sum();
        let v = (s + rnd) >> sh;
        *o = if clamp_max >= 0 { v.clamp(0, clamp_max) } else { v };
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fir8_col_u16_neon(src: &[u16], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::aarch64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = vdupq_n_s32(rnd);
    let vsh = vdupq_n_s32(-sh);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(clamp_max);
    let t: [int32x4_t; 8] = core::array::from_fn(|k| vdupq_n_s32(taps[k] as i32));
    while x + 4 <= n {
        let mut acc = vr;
        for (k, tk) in t.iter().enumerate() {
            let s16 = vld1_u16(src.as_ptr().add(k * stride + x));
            let s = vreinterpretq_s32_u32(vmovl_u16(s16));
            acc = vmlaq_s32(acc, s, *tk);
        }
        let mut v = vshlq_s32(acc, vsh);
        if clamp_max >= 0 {
            v = vminq_s32(vmaxq_s32(v, zero), vmax);
        }
        vst1q_s32(out.as_mut_ptr().add(x), v);
        x += 4;
    }
    if x < n {
        fir8_col_u16_scalar(&src[x..], stride, taps, &mut out[x..], rnd, sh, clamp_max);
    }
}

/// avg: (a + b + 16) >> 5, clamped to [0, max].
pub fn avg_row_scalar(a: &[i32], b: &[i32], out: &mut [i32], clamp_max: i32) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = ((x + y + 16) >> 5).clamp(0, clamp_max);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn avg_row_neon(a: &[i32], b: &[i32], out: &mut [i32], clamp_max: i32) {
    use core::arch::aarch64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = vdupq_n_s32(16);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(clamp_max);
    while x + 4 <= n {
        let va = vld1q_s32(a.as_ptr().add(x));
        let vb = vld1q_s32(b.as_ptr().add(x));
        let s = vshrq_n_s32::<5>(vaddq_s32(vaddq_s32(va, vb), vr));
        let v = vminq_s32(vmaxq_s32(s, zero), vmax);
        vst1q_s32(out.as_mut_ptr().add(x), v);
        x += 4;
    }
    if x < n {
        avg_row_scalar(&a[x..], &b[x..], &mut out[x..], clamp_max);
    }
}

/// w_avg: (a*wt + b*(16-wt) + 128) >> 8, clamped.
pub fn w_avg_row_scalar(a: &[i32], b: &[i32], out: &mut [i32], wt: i32, clamp_max: i32) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = ((x * wt + y * (16 - wt) + 128) >> 8).clamp(0, clamp_max);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn w_avg_row_neon(a: &[i32], b: &[i32], out: &mut [i32], wt: i32, clamp_max: i32) {
    use core::arch::aarch64::*;
    let n = out.len();
    let mut x = 0usize;
    let vw = vdupq_n_s32(wt);
    let vw2 = vdupq_n_s32(16 - wt);
    let vr = vdupq_n_s32(128);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(clamp_max);
    while x + 4 <= n {
        let va = vld1q_s32(a.as_ptr().add(x));
        let vb = vld1q_s32(b.as_ptr().add(x));
        let s = vaddq_s32(vaddq_s32(vmulq_s32(va, vw), vmulq_s32(vb, vw2)), vr);
        let s = vshrq_n_s32::<8>(s);
        let v = vminq_s32(vmaxq_s32(s, zero), vmax);
        vst1q_s32(out.as_mut_ptr().add(x), v);
        x += 4;
    }
    if x < n {
        w_avg_row_scalar(&a[x..], &b[x..], &mut out[x..], wt, clamp_max);
    }
}

/// mask blend: (a*m + b*(64-m) + 32*rnd_add) >> shift, per-pixel mask.
pub fn mask_row_scalar(a: &[i32], b: &[i32], m: &[u8], out: &mut [i32], clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let mm = m[x] as i32;
        *o = ((a[x] * mm + b[x] * (64 - mm) + 512) >> 10).clamp(0, clamp_max);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn mask_row_neon(a: &[i32], b: &[i32], m: &[u8], out: &mut [i32], clamp_max: i32) {
    use core::arch::aarch64::*;
    let n = out.len();
    let mut x = 0usize;
    let v64 = vdupq_n_s32(64);
    let vr = vdupq_n_s32(512);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(clamp_max);
    while x + 4 <= n {
        let va = vld1q_s32(a.as_ptr().add(x));
        let vb = vld1q_s32(b.as_ptr().add(x));
        // widen 4 mask bytes to 4 x i32
        let mb: [i32; 4] = [m[x] as i32, m[x + 1] as i32, m[x + 2] as i32, m[x + 3] as i32];
        let vm = vld1q_s32(mb.as_ptr());
        let vm2 = vsubq_s32(v64, vm);
        let s = vaddq_s32(vaddq_s32(vmulq_s32(va, vm), vmulq_s32(vb, vm2)), vr);
        let s = vshrq_n_s32::<10>(s);
        let v = vminq_s32(vmaxq_s32(s, zero), vmax);
        vst1q_s32(out.as_mut_ptr().add(x), v);
        x += 4;
    }
    if x < n {
        mask_row_scalar(&a[x..], &b[x..], &m[x..], &mut out[x..], clamp_max);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// The vector arm must be live on any host that has it, or every comparison below is the
    /// scalar twin checked against itself and passes without testing anything.
    #[test]
    fn vector_arm_is_live_on_this_host() {
        if crate::env_is_set!("RUSTY_AV2D_NOSIMD") {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        assert_eq!(simd_level(), std::arch::is_x86_feature_detected!("avx2") as u32);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(simd_level(), 1);
    }

    /// Every kernel against its twin over whole blocks: widths that hit the paired four-wide path,
    /// the 256- and 128-bit steps and the scalar tail; odd row counts; strides wider than the
    /// block (and 0 for a repeated source row); every clamp mode; full-range taps; sources up to
    /// 14 bits. Outputs start as a sentinel and are compared in full, so a write outside the
    /// block fails too.
    #[test]
    fn kernels_match_scalar_across_the_full_matrix() {
        let mut st = 0x9e3779b97f4a7c15u64;
        let clamps = [-1i32, 255, 1023, 4095];
        let shifts = [(2i32, 2i32), (32, 6), (34, 6), (512, 10), (1024, 11)];
        let peaks = [255u64, 1023, 4095, 16383];
        let widths = [4usize, 4, 4, 8, 8, 2, 12, 16, 1, 3, 5, 7, 9, 20, 33, 64, 70];
        for trial in 0..4000usize {
            let w = widths[trial % widths.len()];
            let rows = 1 + (xorshift(&mut st) % 9) as usize;
            let pad = (xorshift(&mut st) % 6) as usize;
            let peak = peaks[(xorshift(&mut st) % 4) as usize];
            let extreme = trial % 3 == 0;
            let n = 16 * (rows + 8) * (w + 16);
            let u: Vec<u16> = (0..n)
                .map(|_| {
                    let r = xorshift(&mut st);
                    (if extreme { (r & 1) * peak } else { r % (peak + 1) }) as u16
                })
                .collect();
            let i: Vec<i32> = u.iter().map(|&x| x as i32 - peak as i32 / 3).collect();
            let taps: [i8; 8] = core::array::from_fn(|_| {
                let r = xorshift(&mut st);
                if extreme {
                    [127, -128][(r & 1) as usize]
                } else {
                    r as i8
                }
            });
            let (rnd, sh) = shifts[trial % 5];
            let clamp = clamps[(trial / 5) % 4];
            let os = w + (xorshift(&mut st) % 5) as usize;
            let (mut a, mut b) = (vec![7i32; rows * os + 8], vec![7i32; rows * os + 8]);
            let tag = format!("trial {trial} w={w} rows={rows} os={os} clamp={clamp}");

            let hs = if trial % 7 == 0 { 0 } else { w + 7 + pad };
            fir8_h_u16_scalar(&u, hs, &taps, &mut a, os, w, rows, rnd, sh, clamp);
            fir8_h_u16(&u, hs, &taps, &mut b, os, w, rows, rnd, sh, clamp);
            assert_eq!(a, b, "fir8_h_u16 {tag} ss={hs}");
            let vs = w + pad;
            fir8_v_scalar(&i, vs, &taps, &mut a, os, w, rows, rnd, sh, clamp);
            fir8_v(&i, vs, &taps, &mut b, os, w, rows, rnd, sh, clamp);
            assert_eq!(a, b, "fir8_v {tag} ss={vs}");
            fir8_v_u16_scalar(&u, vs, &taps, &mut a, os, w, rows, rnd, sh, clamp);
            fir8_v_u16(&u, vs, &taps, &mut b, os, w, rows, rnd, sh, clamp);
            assert_eq!(a, b, "fir8_v_u16 {tag} ss={vs}");

            // Blends always clamp; prediction buffers are `w` apart in production.
            let bd = if clamp < 0 { 255 } else { clamp };
            let ps = w + if trial % 2 == 0 { 0 } else { pad };
            let (s1, s2) = (&i[..], &i[n / 2..]);
            let wt = (xorshift(&mut st) % 17) as i32;
            let m: Vec<u8> = (0..rows * ps + w)
                .map(|_| {
                    let r = xorshift(&mut st);
                    if extreme {
                        [0, 64][(r & 1) as usize]
                    } else {
                        (r % 65) as u8
                    }
                })
                .collect();
            avg_scalar(s1, s2, ps, &mut a, os, w, rows, bd);
            avg(s1, s2, ps, &mut b, os, w, rows, bd);
            assert_eq!(a, b, "avg {tag} ps={ps}");
            w_avg_scalar(s1, s2, ps, &mut a, os, w, rows, wt, bd);
            w_avg(s1, s2, ps, &mut b, os, w, rows, wt, bd);
            assert_eq!(a, b, "w_avg {tag} ps={ps}");
            mask_scalar(s1, s2, &m, ps, &mut a, os, w, rows, bd);
            mask(s1, s2, &m, ps, &mut b, os, w, rows, bd);
            assert_eq!(a, b, "mask {tag} ps={ps}");
        }
    }

    #[test]
    #[should_panic(expected = "fir8 source too short")]
    fn short_horizontal_source_is_refused() {
        fir8_h_u16(&[0; 22], 12, &[0; 8], &mut [0; 16], 4, 4, 2, 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "fir8 source too short")]
    fn short_vertical_source_is_refused() {
        fir8_v(&[0; 35], 4, &[0; 8], &mut [0; 8], 4, 4, 2, 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "fir8 output too short")]
    fn short_output_is_refused() {
        fir8_v_u16(&[0; 64], 4, &[0; 8], &mut [0; 7], 4, 4, 2, 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "fir8 source too short")]
    fn wrapping_stride_is_refused() {
        fir8_v_u16(&[0; 64], usize::MAX / 7 + 1, &[0; 8], &mut [0; 8], 4, 4, 1, 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "blend input too short")]
    fn short_blend_input_is_refused() {
        mask(&[0; 8], &[0; 8], &[0; 7], 4, &mut [0; 8], 4, 4, 2, 255);
    }
}
