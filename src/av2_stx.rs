//! AV2 secondary transform (STX) — dav2d `stx_tmpl.c`. STX is applied to the
//! leading coefficients of a block before the primary inverse transform: a small
//! matrix-vector product (`stxfm`) followed by a 4×4 or 8×8 scatter.
//!
//! The basis kernels (`stx_tables.c`, ~355 KB) are a generator job (like the CDFs);
//! this module is the *algorithm*, unit-tested with synthetic kernels.

/// Scatter mappings for the 8×8 STX write-out (dav2d `dav2d_coeff8x8_mapping`).
static COEFF8X8_MAPPING: [[u8; 48]; 3] = [
    [
        0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, //
        33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 42, 35, 28, 21, 14, //
        7, 15, 22, 29, 36, 43, 44, 37, 30, 23, 31, 38, 45, 46, 39, 47,
    ],
    [
        0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, //
        33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35, //
        28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 51, 44, 37, 30, 45,
    ],
    [
        0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, //
        33, 26, 19, 12, 5, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35, 28, //
        21, 29, 36, 43, 50, 57, 58, 51, 44, 37, 45, 52, 59, 60, 53, 61,
    ],
];

#[inline]
fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 {
        -v
    } else {
        v
    }
}

/// STX matrix-vector core (dav2d `stxfm_c`): `out[x] = clip(sign·((|Σ cf[y]·k[y·sz+x]| + 64) >> 7))`,
/// summing over the `eob+1` leading coefficients. `sz` is 16 (4×4) or 48 (8×8).
fn stxfm(out: &mut [i32], cf: &[i32], kernel: &[i8], sz: usize, eob: usize, bitdepth_max: i32) {
    let min = -128 * (1 + bitdepth_max);
    let max = 128 * (1 + bitdepth_max) - 1;
    let h = eob + 1;
    for x in 0..sz {
        if !crate::av2_recon::work_tick("av2_stx:42") { break; }
        let mut sum = 0i32;
        for y in 0..h {
            if !crate::av2_recon::work_tick("av2_stx:44") { break; }
            sum += cf[y] * kernel[y * sz + x] as i32;
        }
        out[x] = apply_sign((sum.abs() + 64) >> 7, sum).clamp(min, max);
    }
}

/// 4×4 secondary inverse transform (dav2d `stxfm4_c`). Reads the `eob+1` leading
/// coeffs of `cf`, zeroes `cf[4..8]`, and writes the 16 results into the 4×4 block
/// (row stride `stride`), transposed or not.
pub fn stxfm4(cf: &mut [i32], stride: usize, transpose: bool, kernel: &[i8], eob: usize, bitdepth_max: i32) {
    let mut sums = [0i32; 16];
    stxfm(&mut sums, cf, kernel, 16, eob, bitdepth_max);
    cf[4..8].fill(0);
    for y in 0..4 {
        for x in 0..4 {
            let s = sums[y * 4 + x];
            if transpose {
                cf[y * stride + x] = s;
            } else {
                cf[x * stride + y] = s;
            }
        }
    }
}

/// 8×8 secondary inverse transform (dav2d `stxfm8_c`). 48 results scattered into the
/// 8×8 block via `COEFF8X8_MAPPING[map_idx]`.
pub fn stxfm8(cf: &mut [i32], stride: usize, transpose: bool, map_idx: usize, kernel: &[i8], eob: usize, bitdepth_max: i32) {
    let mut sums = [0i32; 48];
    stxfm(&mut sums, cf, kernel, 48, eob, bitdepth_max);
    cf[..32].fill(0);
    let mapping = &COEFF8X8_MAPPING[map_idx];
    if !transpose {
        for n in 0..48 {
            let rc = mapping[n] as usize;
            cf[(rc & 7) * stride + (rc >> 3)] = sums[n];
        }
    } else if stride > 8 {
        for n in 0..48 {
            let rc = mapping[n] as usize;
            cf[(rc >> 3) * stride + (rc & 7)] = sums[n];
        }
    } else {
        for n in 0..48 {
            cf[mapping[n] as usize] = sums[n];
        }
    }
}

/// Whether an intra block's STX write-out is transposed, by wide-angle-remapped mode: everything
/// except H / D157 / D67 / SMOOTH_H (dav2d recon_tmpl.c:2705). Inter and intrabc always transpose.
pub fn intra_transpose(y_mode: u8) -> bool {
    const NOT_TRANSPOSED: u32 = (1 << 2) | (1 << 6) | (1 << 8) | (1 << 11);
    (NOT_TRANSPOSED >> y_mode) & 1 == 0
}

/// Inverse secondary transform of a dequantized `tw`x`th` luma block, ahead of the primary itx.
/// `coeff` holds coefficients at their raster positions, while the kernels consume them in SCAN
/// order, so this gathers them first (dav2d stores them scan-indexed at parse: `cf[is_stx ? i : rc]`,
/// recon_tmpl.c:905). `stx_type` is the coded 1..=3; every coefficient past the signalled `eob`
/// is zero, so summing over the kernel's full height matches summing over `eob + 1`.
pub fn apply_stx(coeff: &mut [i32], scan: &[u16], tw: usize, th: usize, transpose: bool, stx_type: u8, set: u8, bitdepth_max: i32) {
    use crate::av2_tables_gen::{STX_4X4_KERNEL, STX_8X8_KERNEL};
    let (ty, set) = ((stx_type - 1) as usize, set as usize);
    let large = tw >= 8 && th >= 8;
    let height = if large { 32 } else { 8 };
    let mut lead = [0i32; 32];
    for (l, &rc) in lead[..height].iter_mut().zip(scan) {
        *l = core::mem::take(&mut coeff[rc as usize]);
    }
    coeff[..height].copy_from_slice(&lead[..height]);
    let stride = th.min(32);
    if large {
        let map_idx = crate::av2_stx_tables::COEFF8X8_MAPPING_IDX[set * 3 + ty] as usize;
        stxfm8(coeff, stride, transpose, map_idx, &STX_8X8_KERNEL[set][ty], height - 1, bitdepth_max);
    } else {
        stxfm4(coeff, stride, transpose, &STX_4X4_KERNEL[set][ty], height - 1, bitdepth_max);
    }
}

/// `(sin, cos)` of the cross-chroma rotation, Q8, indexed by `cctx_type - 1` (dav2d `dav2d_cctx_angle`):
/// 45, 30, 60, -45, -30, -60 degrees.
const CCTX_ANGLE: [(i64, i64); 6] = [(181, 181), (128, 222), (222, 128), (-181, 181), (-128, 222), (-222, 128)];

/// Inverse cross-chroma transform over dequantized coefficients (dav2d itx_tmpl.c `cctx_c`). V is
/// often not coded at all: its residual comes entirely from this rotation of U.
pub fn cctx_rotate(u: &mut [i32], v: &mut [i32], cctx_type: u8, bitdepth_max: i32) {
    let (sin, cos) = CCTX_ANGLE[(cctx_type - 1) as usize];
    let bd = (bitdepth_max + 1).trailing_zeros();
    let (min, max) = (-(1i64 << (bd + 7)), (1i64 << (bd + 7)) - 1);
    let round = |x: i64| ((x + 128 - (x < 0) as i64) >> 8).clamp(min, max) as i32;
    for (cu, cv) in u.iter_mut().zip(v.iter_mut()) {
        let (a, b) = (*cu as i64, *cv as i64);
        *cu = round(a * cos - b * sin);
        *cv = round(a * sin + b * cos);
    }
}

/// The dequantized coefficients of chroma plane `pl` (0 = U, 1 = V). `dequant` is per-site because
/// each one derives its quantizers differently; a `cctx_type` block needs both planes dequantized
/// before either can go through its inverse transform.
pub fn chroma_coeffs<C: core::ops::DerefMut<Target = [i32]>>(pl: usize, cctx_type: u8, bitdepth_max: i32, mut dequant: impl FnMut(usize) -> C) -> C {
    if cctx_type == 0 {
        return dequant(pl);
    }
    let (mut u, mut v) = (dequant(0), dequant(1));
    cctx_rotate(&mut u, &mut v, cctx_type, bitdepth_max);
    if pl == 0 { u } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stxfm_single_row_matches_formula() {
        // eob=0 (h=1): out[x] = ((|cf[0]·k[x]| + 64) >> 7), sign of cf[0].
        let kernel: Vec<i8> = (0..16).map(|x| (x as i8) * 8).collect(); // k[x] = 8x
        let cf = [4i32; 16];
        let mut out = [0i32; 16];
        stxfm(&mut out, &cf, &kernel, 16, 0, 255);
        for x in 0..16 {
            assert_eq!(out[x], ((4 * 8 * x as i32).abs() + 64) >> 7, "x={x}");
        }
    }

    #[test]
    fn stxfm_applies_sign() {
        let kernel: Vec<i8> = (1..=16).map(|x: i32| x as i8).collect(); // all positive
        let cf = [-3i32; 16];
        let mut out = [0i32; 16];
        stxfm(&mut out, &cf, &kernel, 16, 0, 255);
        // sum = -3*k[x] < 0 → negative magnitude
        for x in 0..16 {
            let mag = ((3 * (x as i32 + 1)).abs() + 64) >> 7;
            assert_eq!(out[x], -mag, "x={x}");
        }
    }

    #[test]
    fn stxfm_clips_to_range() {
        // huge coeffs saturate to ±128*(1+bd).
        let kernel = [127i8; 16];
        let cf = [100000i32; 16];
        let mut out = [0i32; 16];
        stxfm(&mut out, &cf, &kernel, 16, 0, 255);
        assert!(out.iter().all(|&v| v == 128 * (1 + 255) - 1));
    }

    #[test]
    fn stxfm4_layout_transpose_vs_not() {
        let kernel: Vec<i8> = (0..4 * 16).map(|i| ((i % 7) as i8) - 3).collect();
        let coeffs = [5i32, -2, 3, 1, 0, 0, 0, 0];
        let eob = 3;
        let mut sums = [0i32; 16];
        stxfm(&mut sums, &coeffs, &kernel, 16, eob, 255);

        let mut cf_n = vec![0i32; 8 * 8];
        cf_n[..8].copy_from_slice(&coeffs);
        stxfm4(&mut cf_n, 8, false, &kernel, eob, 255);
        let mut cf_t = vec![0i32; 8 * 8];
        cf_t[..8].copy_from_slice(&coeffs);
        stxfm4(&mut cf_t, 8, true, &kernel, eob, 255);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(cf_n[x * 8 + y], sums[y * 4 + x], "non-T y={y} x={x}");
                assert_eq!(cf_t[y * 8 + x], sums[y * 4 + x], "T y={y} x={x}");
            }
        }
    }

    #[test]
    fn real_stx_kernel_runs() {
        // The generated STX kernels drive the actual secondary transform.
        use crate::av2_tables_gen::{STX_4X4_KERNEL, STX_8X8_KERNEL};
        // first kernel, hand-verified against dav2d stx_tables.c
        assert_eq!(&STX_4X4_KERNEL[0][0][..4], &[102, -53, -5, -2]);
        assert_eq!((STX_4X4_KERNEL.len(), STX_4X4_KERNEL[0].len(), STX_4X4_KERNEL[0][0].len()), (14, 3, 128));
        assert_eq!((STX_8X8_KERNEL.len(), STX_8X8_KERNEL[0].len(), STX_8X8_KERNEL[0][0].len()), (11, 3, 1536));
        let kernel = &STX_4X4_KERNEL[0][0];
        let mut cf = vec![0i32; 64];
        cf[..8].copy_from_slice(&[12, -4, 7, 0, 3, -1, 2, 0]);
        stxfm4(&mut cf, 8, false, kernel, 3, 255); // real kernel, end-to-end, no panic
    }

    #[test]
    fn apply_stx_gathers_in_scan_order() {
        // A lone coefficient at scan[1] must select kernel row 1, wherever it sits in raster order.
        use crate::av2_tables_gen::STX_4X4_KERNEL;
        let scan: Vec<u16> = vec![0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15];
        let mut coeff = vec![0i32; 16];
        coeff[scan[1] as usize] = 1024;
        apply_stx(&mut coeff, &scan, 4, 4, true, 1, 0, 255);
        let row1 = &STX_4X4_KERNEL[0][0][16..32];
        for y in 0..4 {
            for x in 0..4 {
                let sum = 1024 * row1[y * 4 + x] as i32;
                let want = if sum < 0 { -((-sum + 64) >> 7) } else { (sum + 64) >> 7 };
                assert_eq!(coeff[y * 4 + x], want, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn cctx_rotation_derives_v_from_u() {
        // 45 degrees, V not coded: both planes come out at U * 181/256, rounding away from zero
        // at exact halves on neither side of the sign.
        let (mut u, mut v) = ([256, -256, 1], [0, 0, 0]);
        cctx_rotate(&mut u, &mut v, 1, 255);
        assert_eq!(u, [181, -181, 1]);
        assert_eq!(v, [181, -181, 1]);
        // -45 degrees flips the sign of the derived plane.
        let (mut u, mut v) = ([256], [0]);
        cctx_rotate(&mut u, &mut v, 4, 255);
        assert_eq!((u[0], v[0]), (181, -181));
        // The clamp follows the bit depth.
        let (mut u, mut v) = ([i32::MAX / 2], [0]);
        cctx_rotate(&mut u, &mut v, 2, 1023);
        assert_eq!(u[0], (1 << 17) - 1);
    }

    #[test]
    fn chroma_coeffs_only_touches_both_planes_for_cctx() {
        let mut calls = vec![];
        let c = chroma_coeffs(1, 0, 255, |p| { calls.push(p); vec![p as i32 + 7] });
        assert_eq!((c, calls), (vec![8], vec![1]));
        let mut calls = vec![];
        let c = chroma_coeffs(1, 1, 255, |p| { calls.push(p); vec![if p == 0 { 256 } else { 0 }] });
        assert_eq!((c, calls), (vec![181], vec![0, 1]));
    }

    #[test]
    fn intra_transpose_matches_mode_mask() {
        for mode in 0u8..12 {
            assert_eq!(intra_transpose(mode), ![2, 6, 8, 11].contains(&mode), "mode={mode}");
        }
    }

    #[test]
    fn stxfm8_scatters_via_mapping() {
        let kernel: Vec<i8> = (0..48).map(|i| ((i % 11) as i8) - 5).collect();
        let coeffs = [7i32; 64];
        let eob = 0;
        let mut sums = [0i32; 48];
        stxfm(&mut sums, &coeffs, &kernel, 48, eob, 255);

        let mut cf = vec![0i32; 8 * 8];
        cf[..64].copy_from_slice(&coeffs);
        stxfm8(&mut cf, 8, false, 0, &kernel, eob, 255);
        for n in 0..48 {
            let rc = COEFF8X8_MAPPING[0][n] as usize;
            assert_eq!(cf[(rc & 7) * 8 + (rc >> 3)], sums[n], "n={n}");
        }
    }
}
