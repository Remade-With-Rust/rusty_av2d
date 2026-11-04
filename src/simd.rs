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
//! Shapes covered (the Phase 2 map, cheapest-first):
//! - 8-tap FIR over a row of samples (`fir8_row_u16`): the H pass of motion
//!   compensation.
//! - 8-tap FIR down columns (`fir8_col` over the row-major mid buffer,
//!   `fir8_col_u16` over a reference plane): the V pass. Summing
//!   `mid[(y+k)*w + x]` across `k` is 8 contiguous-row loads, so lanewise it is
//!   one multiply-add per output instead of eight.
//! - Elementwise compound blends (`avg_row`, `w_avg_row`, `mask_row`): two
//!   prep-precision buffers in, one clamped row out.
//!
//! Samples are `i32` today (Phase 1 narrowing to u8/u16 is future work), so
//! AVX2 processes 8 lanes and NEON 4. The kernels are written against rows so
//! the dispatch cost is per-row, not per-sample.

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
        if std::env::var("RUSTY_AV2D_NOSIMD").is_ok() {
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

/// Whether a source of `len` holds `7 * stride + n` samples, the reach of a vertical 8-tap kernel.
/// Checked arithmetic: a wrapped sum would let the kernels read out of bounds.
#[inline]
fn col_fits(len: usize, stride: usize, n: usize) -> bool {
    stride.checked_mul(7).and_then(|reach| reach.checked_add(n)).is_some_and(|need| need <= len)
}

// ---------------------------------------------------------------------------
// 8-tap FIR over a row of samples: out[x] = clamp/((sum_k taps[k]*src[x+k]) + rnd) >> sh
// `src` must have `out.len() + 7` readable elements. `clamp_max < 0` = no clamp
// (prep precision).
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fir8_col_avx2(src: &[i32], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = _mm256_set1_epi32(rnd);
    let sh_v = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let t: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi32(taps[k] as i32));
    while x + 8 <= n {
        let mut acc = vr;
        for (k, tk) in t.iter().enumerate() {
            let s = _mm256_loadu_si256(src.as_ptr().add(k * stride + x) as *const __m256i);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tk));
        }
        let mut v = _mm256_sra_epi32(acc, sh_v);
        if clamp_max >= 0 {
            v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, v);
        x += 8;
    }
    if x < n {
        fir8_col_scalar(&src[x..], stride, taps, &mut out[x..], rnd, sh, clamp_max);
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

/// Dispatching vertical 8-tap. `src.len() >= 7*stride + out.len()`.
#[inline]
pub fn fir8_col(src: &[i32], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    assert!(col_fits(src.len(), stride, out.len()), "fir8 col source too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified; bounds asserted.
        unsafe { fir8_col_avx2(src, stride, taps, out, rnd, sh, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        // SAFETY: NEON baseline; bounds asserted.
        unsafe { fir8_col_neon(src, stride, taps, out, rnd, sh, clamp_max) };
        return;
    }
    fir8_col_scalar(src, stride, taps, out, rnd, sh, clamp_max);
}

// ---------------------------------------------------------------------------
// u16-source variants (Phase 1 item 1): the PLANES are u16; widening happens on
// load, inside the vector, halving the reference-read memory traffic vs i32.
// The mid/prep buffers stay i32 (intermediates go negative and exceed 16 bits).
// ---------------------------------------------------------------------------

pub fn fir8_row_u16_scalar(src: &[u16], taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let s0 = &src[x..x + 8];
        let s: i32 = (0..8).map(|k| taps[k] as i32 * s0[k] as i32).sum();
        let v = (s + rnd) >> sh;
        *o = if clamp_max >= 0 { v.clamp(0, clamp_max) } else { v };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fir8_row_u16_avx2(src: &[u16], taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = _mm256_set1_epi32(rnd);
    let sh_v = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let t: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi32(taps[k] as i32));
    while x + 8 <= n {
        let mut acc = vr;
        for (k, tk) in t.iter().enumerate() {
            let s16 = _mm_loadu_si128(src.as_ptr().add(x + k) as *const __m128i);
            let s = _mm256_cvtepu16_epi32(s16);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tk));
        }
        let mut v = _mm256_sra_epi32(acc, sh_v);
        if clamp_max >= 0 {
            v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, v);
        x += 8;
    }
    if x < n {
        fir8_row_u16_scalar(&src[x..], taps, &mut out[x..], rnd, sh, clamp_max);
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

/// Dispatching u16-source 8-tap row FIR. `src.len() >= out.len() + 7`.
#[inline]
pub fn fir8_row_u16(src: &[u16], taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    assert!(src.len() >= out.len() + 7, "fir8 row source too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified; bounds asserted.
        unsafe { fir8_row_u16_avx2(src, taps, out, rnd, sh, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        // SAFETY: NEON baseline; bounds asserted.
        unsafe { fir8_row_u16_neon(src, taps, out, rnd, sh, clamp_max) };
        return;
    }
    fir8_row_u16_scalar(src, taps, out, rnd, sh, clamp_max);
}

pub fn fir8_col_u16_scalar(src: &[u16], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let s: i32 = (0..8).map(|k| taps[k] as i32 * src[k * stride + x] as i32).sum();
        let v = (s + rnd) >> sh;
        *o = if clamp_max >= 0 { v.clamp(0, clamp_max) } else { v };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fir8_col_u16_avx2(src: &[u16], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = _mm256_set1_epi32(rnd);
    let sh_v = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    let t: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi32(taps[k] as i32));
    while x + 8 <= n {
        let mut acc = vr;
        for (k, tk) in t.iter().enumerate() {
            let s16 = _mm_loadu_si128(src.as_ptr().add(k * stride + x) as *const __m128i);
            let s = _mm256_cvtepu16_epi32(s16);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tk));
        }
        let mut v = _mm256_sra_epi32(acc, sh_v);
        if clamp_max >= 0 {
            v = _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, v);
        x += 8;
    }
    if x < n {
        fir8_col_u16_scalar(&src[x..], stride, taps, &mut out[x..], rnd, sh, clamp_max);
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

/// Dispatching u16-source vertical 8-tap. `src.len() >= 7*stride + out.len()`.
#[inline]
pub fn fir8_col_u16(src: &[u16], stride: usize, taps: &[i8; 8], out: &mut [i32], rnd: i32, sh: i32, clamp_max: i32) {
    assert!(col_fits(src.len(), stride, out.len()), "fir8 col source too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified; bounds asserted.
        unsafe { fir8_col_u16_avx2(src, stride, taps, out, rnd, sh, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        // SAFETY: NEON baseline; bounds asserted.
        unsafe { fir8_col_u16_neon(src, stride, taps, out, rnd, sh, clamp_max) };
        return;
    }
    fir8_col_u16_scalar(src, stride, taps, out, rnd, sh, clamp_max);
}

// ---------------------------------------------------------------------------
// Compound blends (Phase 2 map #3): elementwise over prep-precision rows.
// ---------------------------------------------------------------------------

/// avg: (a + b + 16) >> 5, clamped to [0, max].
pub fn avg_row_scalar(a: &[i32], b: &[i32], out: &mut [i32], clamp_max: i32) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = ((x + y + 16) >> 5).clamp(0, clamp_max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avg_row_avx2(a: &[i32], b: &[i32], out: &mut [i32], clamp_max: i32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let mut x = 0usize;
    let vr = _mm256_set1_epi32(16);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    while x + 8 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(x) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(x) as *const __m256i);
        let s = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(va, vb), vr), 5);
        let v = _mm256_min_epi32(_mm256_max_epi32(s, zero), vmax);
        _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, v);
        x += 8;
    }
    if x < n {
        avg_row_scalar(&a[x..], &b[x..], &mut out[x..], clamp_max);
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

/// Dispatching compound average over a row (`a.len() == b.len() == out.len()`).
#[inline]
pub fn avg_row(a: &[i32], b: &[i32], out: &mut [i32], clamp_max: i32) {
    assert!(a.len() >= out.len() && b.len() >= out.len(), "blend input too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified; bounds asserted.
        unsafe { avg_row_avx2(a, b, out, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        // SAFETY: NEON baseline; bounds asserted.
        unsafe { avg_row_neon(a, b, out, clamp_max) };
        return;
    }
    avg_row_scalar(a, b, out, clamp_max);
}

/// w_avg: (a*wt + b*(16-wt) + 128) >> 8, clamped.
pub fn w_avg_row_scalar(a: &[i32], b: &[i32], out: &mut [i32], wt: i32, clamp_max: i32) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = ((x * wt + y * (16 - wt) + 128) >> 8).clamp(0, clamp_max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn w_avg_row_avx2(a: &[i32], b: &[i32], out: &mut [i32], wt: i32, clamp_max: i32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let mut x = 0usize;
    let vw = _mm256_set1_epi32(wt);
    let vw2 = _mm256_set1_epi32(16 - wt);
    let vr = _mm256_set1_epi32(128);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    while x + 8 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(x) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(x) as *const __m256i);
        let s = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(va, vw), _mm256_mullo_epi32(vb, vw2)),
            vr,
        );
        let s = _mm256_srai_epi32(s, 8);
        let v = _mm256_min_epi32(_mm256_max_epi32(s, zero), vmax);
        _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, v);
        x += 8;
    }
    if x < n {
        w_avg_row_scalar(&a[x..], &b[x..], &mut out[x..], wt, clamp_max);
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

/// Dispatching weighted average.
#[inline]
pub fn w_avg_row(a: &[i32], b: &[i32], out: &mut [i32], wt: i32, clamp_max: i32) {
    assert!(a.len() >= out.len() && b.len() >= out.len(), "blend input too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified; bounds asserted.
        unsafe { w_avg_row_avx2(a, b, out, wt, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        // SAFETY: NEON baseline; bounds asserted.
        unsafe { w_avg_row_neon(a, b, out, wt, clamp_max) };
        return;
    }
    w_avg_row_scalar(a, b, out, wt, clamp_max);
}

/// mask blend: (a*m + b*(64-m) + 32*rnd_add) >> shift, per-pixel mask.
pub fn mask_row_scalar(a: &[i32], b: &[i32], m: &[u8], out: &mut [i32], clamp_max: i32) {
    for (x, o) in out.iter_mut().enumerate() {
        let mm = m[x] as i32;
        *o = ((a[x] * mm + b[x] * (64 - mm) + 512) >> 10).clamp(0, clamp_max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mask_row_avx2(a: &[i32], b: &[i32], m: &[u8], out: &mut [i32], clamp_max: i32) {
    use core::arch::x86_64::*;
    let n = out.len();
    let mut x = 0usize;
    let v64 = _mm256_set1_epi32(64);
    let vr = _mm256_set1_epi32(512);
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi32(clamp_max);
    while x + 8 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(x) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(x) as *const __m256i);
        // widen 8 mask bytes to 8 x i32
        let mb = _mm_loadl_epi64(m.as_ptr().add(x) as *const __m128i);
        let vm = _mm256_cvtepu8_epi32(mb);
        let vm2 = _mm256_sub_epi32(v64, vm);
        let s = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(va, vm), _mm256_mullo_epi32(vb, vm2)),
            vr,
        );
        let s = _mm256_srai_epi32(s, 10);
        let v = _mm256_min_epi32(_mm256_max_epi32(s, zero), vmax);
        _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, v);
        x += 8;
    }
    if x < n {
        mask_row_scalar(&a[x..], &b[x..], &m[x..], &mut out[x..], clamp_max);
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

/// Dispatching per-pixel mask blend ((a*m + b*(64-m) + 512) >> 10).
#[inline]
pub fn mask_row(a: &[i32], b: &[i32], m: &[u8], out: &mut [i32], clamp_max: i32) {
    assert!(a.len() >= out.len() && b.len() >= out.len() && m.len() >= out.len(), "blend input too short");
    #[cfg(target_arch = "x86_64")]
    if simd_level() >= 1 {
        // SAFETY: avx2 verified; bounds asserted.
        unsafe { mask_row_avx2(a, b, m, out, clamp_max) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if simd_level() >= 1 {
        // SAFETY: NEON baseline; bounds asserted.
        unsafe { mask_row_neon(a, b, m, out, clamp_max) };
        return;
    }
    mask_row_scalar(a, b, m, out, clamp_max);
}

// ---------------------------------------------------------------------------
// Scalar-twin equality tests: randomized inputs, bit-exact agreement.
// ---------------------------------------------------------------------------

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
        if std::env::var("RUSTY_AV2D_NOSIMD").is_ok() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        assert_eq!(simd_level(), std::arch::is_x86_feature_detected!("avx2") as u32);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(simd_level(), 1);
    }

    /// Every kernel against its twin, in every clamp mode production uses and beyond: full-range
    /// taps, sources up to 14 bits, extreme values, and every tail length.
    #[test]
    fn kernels_match_scalar_across_the_full_matrix() {
        let mut st = 0x9e3779b97f4a7c15u64;
        let clamps = [-1i32, 255, 1023, 4095];
        let shifts = [(2i32, 2i32), (32, 6), (34, 6), (512, 10), (1024, 11)];
        let peaks = [255u64, 1023, 4095, 16383];
        for trial in 0..6000usize {
            let n = 1 + trial % 140;
            let stride = n + (xorshift(&mut st) % 9) as usize;
            let peak = peaks[(xorshift(&mut st) % 4) as usize];
            let extreme = trial % 3 == 0;
            let u: Vec<u16> = (0..7 * stride + n + 8)
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
            let (mut a, mut b) = (vec![7i32; n], vec![7i32; n]);

            fir8_col_scalar(&i, stride, &taps, &mut a, rnd, sh, clamp);
            fir8_col(&i, stride, &taps, &mut b, rnd, sh, clamp);
            assert_eq!(a, b, "fir8_col trial {trial} n={n} clamp={clamp}");
            fir8_row_u16_scalar(&u, &taps, &mut a, rnd, sh, clamp);
            fir8_row_u16(&u, &taps, &mut b, rnd, sh, clamp);
            assert_eq!(a, b, "fir8_row_u16 trial {trial} n={n} clamp={clamp}");
            fir8_col_u16_scalar(&u, stride, &taps, &mut a, rnd, sh, clamp);
            fir8_col_u16(&u, stride, &taps, &mut b, rnd, sh, clamp);
            assert_eq!(a, b, "fir8_col_u16 trial {trial} n={n} clamp={clamp}");

            // Blends always clamp.
            let bd = if clamp < 0 { 255 } else { clamp };
            let (s1, s2) = (&i[..n], &i[stride..stride + n]);
            let wt = (xorshift(&mut st) % 17) as i32;
            let m: Vec<u8> = (0..n)
                .map(|_| {
                    let r = xorshift(&mut st);
                    if extreme {
                        [0, 64][(r & 1) as usize]
                    } else {
                        (r % 65) as u8
                    }
                })
                .collect();
            avg_row_scalar(s1, s2, &mut a, bd);
            avg_row(s1, s2, &mut b, bd);
            assert_eq!(a, b, "avg trial {trial} n={n} clamp={bd}");
            w_avg_row_scalar(s1, s2, &mut a, wt, bd);
            w_avg_row(s1, s2, &mut b, wt, bd);
            assert_eq!(a, b, "w_avg trial {trial} n={n} clamp={bd}");
            mask_row_scalar(s1, s2, &m, &mut a, bd);
            mask_row(s1, s2, &m, &mut b, bd);
            assert_eq!(a, b, "mask trial {trial} n={n} clamp={bd}");
        }
    }

    #[test]
    #[should_panic(expected = "fir8 row source too short")]
    fn short_row_source_is_refused() {
        fir8_row_u16(&[0; 14], &[0; 8], &mut [0; 8], 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "fir8 col source too short")]
    fn short_col_source_is_refused() {
        fir8_col(&[0; 8], 16, &[0; 8], &mut [0; 8], 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "fir8 col source too short")]
    fn wrapping_stride_is_refused() {
        fir8_col_u16(&[0; 64], usize::MAX / 7 + 1, &[0; 8], &mut [0; 8], 0, 0, -1);
    }

    #[test]
    #[should_panic(expected = "blend input too short")]
    fn short_blend_input_is_refused() {
        mask_row(&[0; 8], &[0; 8], &[0; 4], &mut [0; 8], 255);
    }

    #[test]
    fn fir8_col_matches_scalar() {
        let mut st = 0xabcdefu64;
        for trial in 0..200 {
            let n = 1 + (xorshift(&mut st) % 130) as usize;
            let stride = n + (xorshift(&mut st) % 8) as usize;
            let src: Vec<i32> = (0..7 * stride + n).map(|_| (xorshift(&mut st) % 4096) as i32 - 1024).collect();
            let taps: [i8; 8] = core::array::from_fn(|_| (xorshift(&mut st) as i8) / 2);
            let mut a = vec![0i32; n];
            let mut b = vec![0i32; n];
            fir8_col_scalar(&src, stride, &taps, &mut a, 512, 10, 255);
            fir8_col(&src, stride, &taps, &mut b, 512, 10, 255);
            assert_eq!(a, b, "fir8_col trial {trial} n={n}");
        }
    }

    #[test]
    fn fir8_u16_matches_scalar() {
        let mut st = 0x5eedu64;
        for trial in 0..200 {
            let n = 1 + (xorshift(&mut st) % 130) as usize;
            let stride = n + (xorshift(&mut st) % 8) as usize;
            let src: Vec<u16> = (0..7 * stride + n + 7).map(|_| (xorshift(&mut st) % 1024) as u16).collect();
            let taps: [i8; 8] = core::array::from_fn(|_| (xorshift(&mut st) as i8) / 2);
            let mut a = vec![0i32; n];
            let mut b = vec![0i32; n];
            fir8_row_u16_scalar(&src, &taps, &mut a, 34, 6, 255);
            fir8_row_u16(&src, &taps, &mut b, 34, 6, 255);
            assert_eq!(a, b, "fir8_row_u16 trial {trial}");
            fir8_col_u16_scalar(&src, stride, &taps, &mut a, 2, 2, -1);
            fir8_col_u16(&src, stride, &taps, &mut b, 2, 2, -1);
            assert_eq!(a, b, "fir8_col_u16 trial {trial}");
        }
    }

    #[test]
    fn blends_match_scalar() {
        let mut st = 0x777u64;
        for trial in 0..200 {
            let n = 1 + (xorshift(&mut st) % 130) as usize;
            let a: Vec<i32> = (0..n).map(|_| (xorshift(&mut st) % 16384) as i32 - 4096).collect();
            let b: Vec<i32> = (0..n).map(|_| (xorshift(&mut st) % 16384) as i32 - 4096).collect();
            let m: Vec<u8> = (0..n).map(|_| (xorshift(&mut st) % 65) as u8).collect();
            let (mut s1, mut s2) = (vec![0i32; n], vec![0i32; n]);
            avg_row_scalar(&a, &b, &mut s1, 255);
            avg_row(&a, &b, &mut s2, 255);
            assert_eq!(s1, s2, "avg trial {trial}");
            let wt = (xorshift(&mut st) % 17) as i32;
            w_avg_row_scalar(&a, &b, &mut s1, wt, 255);
            w_avg_row(&a, &b, &mut s2, wt, 255);
            assert_eq!(s1, s2, "w_avg trial {trial}");
            mask_row_scalar(&a, &b, &m, &mut s1, 255);
            mask_row(&a, &b, &m, &mut s2, 255);
            assert_eq!(s1, s2, "mask trial {trial}");
        }
    }
}
