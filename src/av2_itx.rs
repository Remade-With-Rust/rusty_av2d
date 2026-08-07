//! AV2 1-D inverse transforms — pure kernels, bit-exact transcription from dav2d
//! `itx_1d.c`. AV2 restructured AV1's per-size transforms into a compact form:
//! DCT via a recursive matrix-kernel (`inv_dct4` butterfly + generic odd-part
//! `inv_dct_1d`), DST/ADST/DDT via direct matrix-multiply with an optional flip,
//! identity via scaling. All operate in place on an `i32` coefficient buffer with
//! a positive stride.
//!
//! These are unit-tested in isolation (DC-flat, impulse/column responses) and feed
//! the 2-D transform (`itx_tmpl`) once that is wired — see docs/decode-core.md §3.

// ---- kernel matrices (dav2d itx_1d.c) ------------------------------------------------

static DCT8_KERNEL: [i8; 4 * 4] = [
    89, 75, 50, 18, //
    75, -18, -89, -50, //
    50, -89, 18, 75, //
    18, -50, 75, -89,
];

static DCT16_KERNEL: [i8; 8 * 8] = [
    90, 87, 80, 70, 57, 43, 26, 9, //
    87, 57, 9, -43, -80, -90, -70, -26, //
    80, 9, -70, -87, -26, 57, 90, 43, //
    70, -43, -87, 9, 90, 26, -80, -57, //
    57, -80, -26, 90, -9, -87, 43, 70, //
    43, -90, 57, 26, -87, 70, 9, -80, //
    26, -70, 90, -80, 43, 9, -57, 87, //
    9, -26, 43, -57, 70, -80, 87, -90,
];

static DCT32_KERNEL: [i8; 16 * 16] = [
    90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4, //
    90, 82, 67, 47, 22, -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13, //
    88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39, 73, 90, 85, 61, 22, //
    85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30, //
    82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67, 4, 73, 88, 39, //
    78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 30, 90, 54, -39, -90, -47, //
    73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61, -47, -88, -4, 85, 54, //
    67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47, -73, -61, //
    61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67, //
    54, -85, -4, 88, -47, -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73, //
    47, -90, 39, 54, -90, 30, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, //
    39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82, //
    30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85, //
    22, -61, 85, -90, 73, -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88, //
    13, -39, 61, -78, 88, -90, 85, -73, 54, -30, 4, 22, -47, 67, -82, 90, //
    4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
];

static ADST4_KERNEL: [i8; 4 * 4] = [
    18, 50, 75, 89, //
    50, 89, 18, -75, //
    75, 18, -89, 50, //
    89, -75, 50, -18,
];

static ADST8_KERNEL: [i8; 8 * 8] = [
    11, 34, 54, 71, 84, 88, 79, 50, //
    28, 74, 89, 68, 17, -44, -83, -69, //
    44, 89, 48, -41, -89, -44, 50, 81, //
    58, 76, -34, -86, 10, 88, 6, -84, //
    70, 39, -87, 1, 86, -44, -59, 78, //
    79, -12, -66, 87, -35, -44, 86, -62, //
    86, -58, 12, 38, -75, 88, -74, 40, //
    89, -86, 79, -70, 58, -44, 29, -14,
];

static ADST16_KERNEL: [i8; 16 * 16] = [
    8, 25, 41, 55, 67, 77, 84, 88, 89, 87, 81, 73, 62, 48, 33, 17, //
    17, 48, 73, 87, 88, 77, 55, 25, -8, -41, -67, -84, -89, -81, -62, -33, //
    25, 67, 88, 81, 48, 0, -48, -81, -88, -67, -25, 25, 67, 88, 81, 48, //
    33, 81, 84, 41, -25, -77, -87, -48, 17, 73, 88, 55, -8, -67, -89, -62, //
    41, 88, 62, -17, -81, -77, -8, 67, 87, 33, -48, -89, -55, 25, 84, 73, //
    48, 88, 25, -67, -81, 0, 81, 67, -25, -88, -48, 48, 88, 25, -67, -81, //
    55, 81, -17, -89, -25, 77, 62, -48, -84, 8, 88, 33, -73, -67, 41, 87, //
    62, 67, -55, -73, 48, 77, -41, -81, 33, 84, -25, -87, 17, 88, -8, -89, //
    67, 48, -81, -25, 88, 0, -88, 25, 81, -48, -67, 67, 48, -81, -25, 88, //
    73, 25, -89, 33, 67, -77, -17, 88, -41, -62, 81, 8, -87, 48, 55, -84, //
    77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, //
    81, -25, -48, 88, -67, 0, 67, -88, 48, 25, -81, 81, -25, -48, 88, -67, //
    84, -48, -8, 62, -88, 77, -33, -25, 73, -89, 67, -17, -41, 81, -87, 55, //
    87, -67, 33, 8, -48, 77, -89, 81, -55, 17, 25, -62, 84, -88, 73, -41, //
    88, -81, 67, -48, 25, 0, -25, 48, -67, 81, -88, 88, -81, 67, -48, 25, //
    89, -88, 87, -84, 81, -77, 73, -67, 62, -55, 48, -41, 33, -25, 17, -8,
];

static FLIPADST4_KERNEL: [i8; 4 * 4] = [
    89, 75, 50, 18, //
    75, -18, -89, -50, //
    50, -89, 18, 75, //
    18, -50, 75, -89,
];

static FLIPADST16_KERNEL: [i8; 16 * 16] = [
    89, 88, 87, 84, 81, 77, 73, 67, 62, 55, 48, 41, 33, 25, 17, 8, //
    88, 81, 67, 48, 25, 0, -25, -48, -67, -81, -88, -88, -81, -67, -48, -25, //
    87, 67, 33, -8, -48, -77, -89, -81, -55, -17, 25, 62, 84, 88, 73, 41, //
    84, 48, -8, -62, -88, -77, -33, 25, 73, 89, 67, 17, -41, -81, -87, -55, //
    81, 25, -48, -88, -67, 0, 67, 88, 48, -25, -81, -81, -25, 48, 88, 67, //
    77, 0, -77, -77, 0, 77, 77, 0, -77, -77, 0, 77, 77, 0, -77, -77, //
    73, -25, -89, -33, 67, 77, -17, -88, -41, 62, 81, -8, -87, -48, 55, 84, //
    67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67, -67, 48, 81, -25, -88, //
    62, -67, -55, 73, 48, -77, -41, 81, 33, -84, -25, 87, 17, -88, -8, 89, //
    55, -81, -17, 89, -25, -77, 62, 48, -84, -8, 88, -33, -73, 67, 41, -87, //
    48, -88, 25, 67, -81, 0, 81, -67, -25, 88, -48, -48, 88, -25, -67, 81, //
    41, -88, 62, 17, -81, 77, -8, -67, 87, -33, -48, 89, -55, -25, 84, -73, //
    33, -81, 84, -41, -25, 77, -87, 48, 17, -73, 88, -55, -8, 67, -89, 62, //
    25, -67, 88, -81, 48, 0, -48, 81, -88, 67, -25, -25, 67, -88, 81, -48, //
    17, -48, 73, -87, 88, -77, 55, -25, -8, 41, -67, 84, -89, 81, -62, 33, //
    8, -25, 41, -55, 67, -77, 84, -88, 89, -87, 81, -73, 62, -48, 33, -17,
];

static DDT8_KERNEL: [i8; 8 * 8] = [
    4, 6, 22, 57, 96, 103, 78, 56, //
    7, 14, 48, 94, 73, -17, -79, -96, //
    15, 36, 85, 76, -43, -80, 7, 98, //
    33, 77, 88, -26, -69, 56, 56, -77, //
    65, 100, 0, -73, 55, 15, -82, 54, //
    98, 45, -86, 34, 20, -66, 79, -33, //
    106, -57, -23, 54, -71, 75, -56, 19, //
    80, -98, 82, -66, 53, -41, 26, -6,
];

static DDT16_KERNEL: [i8; 16 * 16] = [
    12, 17, 37, 45, 47, 60, 64, 82, 89, 100, 92, 84, 69, 50, 51, 44, //
    15, 23, 49, 60, 60, 74, 70, 73, 48, 9, -35, -71, -83, -79, -89, -95, //
    19, 30, 60, 69, 61, 64, 40, 3, -53, -99, -91, -46, 2, 47, 73, 124, //
    23, 38, 69, 73, 49, 28, -19, -80, -96, -45, 42, 88, 75, 14, -17, -126, //
    30, 48, 75, 66, 19, -31, -79, -91, -5, 84, 71, -16, -78, -60, -45, 108, //
    39, 61, 75, 40, -29, -87, -78, 10, 89, 36, -69, -67, 18, 67, 89, -81, //
    51, 76, 61, -8, -77, -82, 11, 94, 16, -81, -22, 79, 50, -37, -103, 54, //
    66, 87, 29, -65, -83, 4, 92, 18, -83, 4, 85, -22, -85, -6, 97, -30, //
    78, 83, -18, -91, -16, 88, 28, -84, 12, 73, -60, -46, 81, 49, -83, 16, //
    88, 59, -67, -57, 75, 54, -85, -5, 75, -60, -17, 84, -43, -80, 71, -6, //
    94, 19, -96, 21, 93, -55, -41, 80, -51, -17, 77, -68, -6, 98, -56, 1, //
    97, -30, -83, 86, 3, -77, 82, -17, -43, 76, -70, 15, 53, -99, 44, 3, //
    93, -73, -28, 81, -92, 29, 39, -70, 81, -55, 11, 46, -81, 90, -31, -4, //
    83, -99, 40, 8, -74, 88, -83, 47, -14, -21, 56, -83, 88, -71, 22, 5, //
    68, -99, 84, -69, 32, 3, -37, 55, -75, 81, -83, 82, -69, 48, -11, -3, //
    50, -76, 83, -90, 97, -86, 83, -68, 67, -56, 49, -40, 32, -19, 5, 2,
];

// ---- DCT (recursive matrix-kernel) ---------------------------------------------------

/// Generic odd-part step shared by `inv_dct{8,16,32}` (dav2d `inv_dct_1d_c`).
/// Even samples (`c[i*2*stride]`, already transformed by the smaller DCT) form `a`;
/// the odd samples are combined through `mat` (n×n) into `b`; outputs butterfly.
fn inv_dct_1d(c: &mut [i32], stride: usize, mat: &[i8], n: usize) {
    let mut a = [0i32; 16];
    let mut b = [0i32; 16];
    let k = n * 2 - 1;
    let mut m = 0;
    for i in 0..n {
        if !crate::av2_recon::work_tick("av2_itx:153") { break; }
        let mut sum = 0i32;
        let mut j = 1;
        while j <= k {
            sum += mat[m] as i32 * c[j * stride];
            m += 1;
            j += 2;
        }
        a[i] = c[i * 2 * stride];
        b[i] = sum;
    }
    for i in 0..n {
        if !crate::av2_recon::work_tick("av2_itx:164") { break; }
        c[i * stride] = a[i] + b[i];
        c[(k - i) * stride] = a[i] - b[i];
    }
}

/// 4-point inverse DCT butterfly (dav2d `inv_dct4_1d_c`), constants 64/83/35.
fn inv_dct4_1d(c: &mut [i32], stride: usize) {
    let a0 = c[0] * 64 + c[2 * stride] * 64;
    let a1 = c[0] * 64 - c[2 * stride] * 64;
    let b0 = c[stride] * 83 + c[3 * stride] * 35;
    let b1 = c[stride] * 35 - c[3 * stride] * 83;
    c[0] = a0 + b0;
    c[stride] = a1 + b1;
    c[2 * stride] = a1 - b1;
    c[3 * stride] = a0 - b0;
}

fn inv_dct8_1d(c: &mut [i32], stride: usize) {
    inv_dct4_1d(c, 2 * stride);
    inv_dct_1d(c, stride, &DCT8_KERNEL, 4);
}

fn inv_dct16_1d(c: &mut [i32], stride: usize) {
    inv_dct8_1d(c, 2 * stride);
    inv_dct_1d(c, stride, &DCT16_KERNEL, 8);
}

fn inv_dct32_1d(c: &mut [i32], stride: usize) {
    inv_dct16_1d(c, 2 * stride);
    inv_dct_1d(c, stride, &DCT32_KERNEL, 16);
}

// ---- DST / ADST / DDT (direct matrix-multiply, optional flip) ------------------------

/// Generic n×n matrix multiply (dav2d `inv_dst_1d_c`). `flip` reverses the output
/// order (the C trick `c += f*stride; stride = -stride`, here written directly).
fn inv_dst_1d(c: &mut [i32], stride: usize, mat: &[i8], n: usize, flip: bool) {
    let mut sums = [0i32; 16];
    let mut m = 0;
    for sum_i in sums.iter_mut().take(n) {
        if !crate::av2_recon::work_tick("av2_itx:204") { break; }
        let mut sum = 0i32;
        for j in 0..n {
            if !crate::av2_recon::work_tick("av2_itx:206") { break; }
            sum += mat[m] as i32 * c[j * stride];
            m += 1;
        }
        *sum_i = sum;
    }
    for i in 0..n {
        if !crate::av2_recon::work_tick("av2_itx:212") { break; }
        let dst = if flip { (n - 1 - i) * stride } else { i * stride };
        c[dst] = sums[i];
    }
}

fn inv_adst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST4_KERNEL, 4, false);
}
fn inv_adst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST8_KERNEL, 8, false);
}
fn inv_adst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST16_KERNEL, 16, false);
}
fn inv_flipadst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &FLIPADST4_KERNEL, 4, false);
}
fn inv_flipadst8_1d(c: &mut [i32], stride: usize) {
    // dav2d: flipadst8 reuses the adst8 kernel with flip=1.
    inv_dst_1d(c, stride, &ADST8_KERNEL, 8, true);
}
fn inv_flipadst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &FLIPADST16_KERNEL, 16, false);
}
fn inv_ddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT8_KERNEL, 8, false);
}
fn inv_ddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT16_KERNEL, 16, false);
}
fn inv_flipddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT8_KERNEL, 8, true);
}
fn inv_flipddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT16_KERNEL, 16, true);
}

// ---- identity (scaling) --------------------------------------------------------------

fn inv_identity4_1d(c: &mut [i32], stride: usize) {
    for i in 0..4 {
        c[i * stride] *= 128;
    }
}
fn inv_identity8_1d(c: &mut [i32], stride: usize) {
    for i in 0..8 {
        c[i * stride] *= 181;
    }
}
fn inv_identity16_1d(c: &mut [i32], stride: usize) {
    for i in 0..16 {
        c[i * stride] *= 256;
    }
}
fn inv_identity32_1d(c: &mut [i32], stride: usize) {
    for i in 0..32 {
        c[i * stride] *= 362;
    }
}

// ---- 2-D inverse transform (square sizes) --------------------------------------------

type Itx1dFn = fn(&mut [i32], usize);

/// `dav2d_tx1d_fns[log2(side/4)][type]` — the row/col 1-D transform selector.
/// types: 0=DCT 1=ADST 2=FLIPADST 3=IDENTITY 4=DDT 5=FLIPDDT.
fn tx1d_fn(lsize: usize, ty: usize) -> Itx1dFn {
    match (lsize, ty) {
        (0, 0) => inv_dct4_1d,
        (0, 1) => inv_adst4_1d,
        (0, 2) => inv_flipadst4_1d,
        (0, 3) => inv_identity4_1d,
        (1, 0) => inv_dct8_1d,
        (1, 1) => inv_adst8_1d,
        (1, 2) => inv_flipadst8_1d,
        (1, 3) => inv_identity8_1d,
        (1, 4) => inv_ddt8_1d,
        (1, 5) => inv_flipddt8_1d,
        (2, 0) => inv_dct16_1d,
        (2, 1) => inv_adst16_1d,
        (2, 2) => inv_flipadst16_1d,
        (2, 3) => inv_identity16_1d,
        (2, 4) => inv_ddt16_1d,
        (2, 5) => inv_flipddt16_1d,
        (3, 0) => inv_dct32_1d,
        (3, 3) => inv_identity32_1d,
        (4, 0) => inv_dct32_1d, // 64x64 reuses dct32 (upsampled)
        _ => panic!("invalid (lsize {lsize}, type {ty}) tx1d combination"),
    }
}

#[inline]
fn iclip(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Per-`(lw, lh)` `{row_shift, col_shift}` (dav2d_tx_shift), sizes 4..64
/// (`lw`,`lh` ∈ 0..4). Verified against the dav2d table entry for each size.
const TX_SHIFT: [[[u32; 2]; 5]; 5] = [
    //  lh=0      lh=1      lh=2      lh=3      lh=4
    [[7, 10], [7, 10], [6, 12], [7, 11], [6, 13]], // lw=0 (4-wide)
    [[7, 10], [7, 11], [7, 11], [6, 13], [6, 12]], // lw=1 (8-wide)
    [[6, 12], [7, 11], [6, 13], [6, 12], [6, 13]], // lw=2 (16-wide)
    [[7, 11], [6, 13], [6, 12], [6, 13], [6, 12]], // lw=3 (32-wide)
    [[6, 13], [6, 12], [6, 13], [6, 12], [6, 13]], // lw=4 (64-wide)
];

/// 2-D inverse transform for TX sizes 4..64 (square or rectangular), `TX_CLASS_2D`,
/// DPCM-flag 0. `lw`/`lh` = log2(width/4)/log2(height/4). Rect blocks (`(lw+lh)`
/// odd) apply the `*181>>8` read scaling. 64-wide dimensions transform only the
/// leading 32 coefficients then **upsample 2×** (idct64 path). Writes the per-pixel
/// `i32` residual (row-major, pre clip/add) to `out`. Bit-exact core of dav2d
/// `inv_txfm_add_c`.
pub fn inv_txfm_2d(coeff: &[i32], lw: usize, lh: usize, row_ty: usize, col_ty: usize, out: &mut [i32]) {
    let w = 4usize << lw;
    let h = 4usize << lh;
    let sw = w.min(32);
    let sh = h.min(32);
    let is_rect2 = (lw + lh) & 1 == 1;
    // HARDENING: a corrupt stream can present an out-of-range tx-size log2 — clamp to the
    // largest defined transform rather than indexing past TX_SHIFT.
    let (lw, lh) = (lw.min(TX_SHIFT.len() - 1), lh.min(TX_SHIFT[0].len() - 1));
    let [shift0, shift1] = TX_SHIFT[lw][lh];
    let row_fn = tx1d_fn(lw, row_ty); // width transform (lsize 4 → dct32)
    let col_fn = tx1d_fn(lh, col_ty); // height transform

    let mut tmp = [0i32; 32 * 32];
    // row pass: gather a row (stride sh), rect-scale, transform in place
    for col in 0..sh {
        if !crate::av2_recon::work_tick("av2_itx:341") { break; }
        let base = col * sw;
        for x in 0..sw {
            if !crate::av2_recon::work_tick("av2_itx:343") { break; }
            let v = coeff[col + x * sh];
            tmp[base + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
        row_fn(&mut tmp[base..base + sw], 1);
    }
    // intermediate rounding + clip: dav2d itx_tmpl.c:150-154 — i16 range at 8-bit,
    // ±(1 << (bitdepth+7)) at high bit depth (`~bitdepth_max << 7`).
    let bdmax = crate::av2_frame::BDMAX.with(|c| c.get());
    let (row_min, row_max) = if bdmax == 255 {
        (i16::MIN as i32, i16::MAX as i32)
    } else {
        (!bdmax << 7, !(!bdmax << 7))
    };
    let rnd0 = (1i32 << shift0) >> 1;
    for v in tmp.iter_mut().take(sw * sh) {
        if !crate::av2_recon::work_tick("av2_itx:358") { break; }
        *v = iclip((*v + rnd0) >> shift0, row_min, row_max);
    }
    // column pass (stride sw)
    for x in 0..sw {
        if !crate::av2_recon::work_tick("av2_itx:362") { break; }
        col_fn(&mut tmp[x..], sw);
    }
    // final rounding → residual, with idct64 2× upsampling where w>sw / h>sh
    let rnd1 = (1i32 << shift1) >> 1;
    let xr = (w > sw) as usize; // replicate each sample 2× horizontally
    let yr = (h > sh) as usize; // ...and/or vertically
    for sy in 0..sh {
        if !crate::av2_recon::work_tick("av2_itx:369") { break; }
        for sx in 0..sw {
            if !crate::av2_recon::work_tick("av2_itx:370") { break; }
            let cf = (tmp[sy * sw + sx] + rnd1) >> shift1;
            for dy in 0..=yr {
                if !crate::av2_recon::work_tick("av2_itx:372") { break; }
                for dx in 0..=xr {
                    if !crate::av2_recon::work_tick("av2_itx:373") { break; }
                    out[((sy << yr) + dy) * w + (sx << xr) + dx] = cf;
                }
            }
        }
    }
}

/// 4-point inverse Walsh-Hadamard transform (dav2d `dav2d_inv_wht4_1d_c`). Lossless.
fn inv_wht4_1d(c: &mut [i32], stride: usize) {
    let (in0, in1, in2, in3) = (c[0], c[stride], c[2 * stride], c[3 * stride]);
    let t0 = in0 + in1;
    let t2 = in2 - in3;
    let t4 = (t0 - t2) >> 1;
    let t3 = t4 - in3;
    let t1 = t4 - in1;
    c[0] = t0 - t3;
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = t2 + t1;
}

/// 2-D inverse WHT 4×4 (dav2d `inv_txfm_add_wht_wht_4x4_c`). The `>>3` input
/// prescale is the only scaling (lossless path; no rounding shift). Writes the
/// 16 `i32` residuals (row-major) to `out`.
pub fn inv_txfm_wht4x4(coeff: &[i32], out: &mut [i32]) {
    let mut tmp = [0i32; 16];
    for y in 0..4 {
        let base = y * 4;
        for x in 0..4 {
            tmp[base + x] = coeff[y + x * 4] >> 3;
        }
        inv_wht4_1d(&mut tmp[base..base + 4], 1);
    }
    for x in 0..4 {
        inv_wht4_1d(&mut tmp[x..], 4);
    }
    out[..16].copy_from_slice(&tmp);
}

/// DC-only fast path (dav2d `inv_txfm_add_c` DC branch). Returns the single
/// residual value added to every pixel of the block.
pub fn inv_txfm_dc(dc_in: i32, lw: usize, lh: usize) -> i32 {
    let is_rect2 = (lw + lh) & 1 == 1;
    // HARDENING: a corrupt stream can present an out-of-range tx-size log2 — clamp to the
    // largest defined transform rather than indexing past TX_SHIFT.
    let (lw, lh) = (lw.min(TX_SHIFT.len() - 1), lh.min(TX_SHIFT[0].len() - 1));
    let [shift0, shift1] = TX_SHIFT[lw][lh];
    let shift_p1 = shift0 as i32;
    let shift = shift_p1 + shift1 as i32 - 12;
    let rnd = (1 << (shift - 1)) + shift_p1 - 6;
    let mut dc = dc_in;
    if is_rect2 {
        dc = (dc * 181 + 128) >> 8;
    }
    (dc + rnd) >> shift
}

/// Cross-component transform (dav2d `cctx_c`): rotate each `(u, v)` coefficient
/// pair by the CCTX angle `[sina, cosa, -sina]`. `bd` is the bit depth (8/10/12).
pub fn cctx(u: &mut [i32], v: &mut [i32], sina: i32, cosa: i32, bd: u32) {
    let min = -(1i32 << (bd + 7));
    let max = (1i32 << (bd + 7)) - 1;
    for i in 0..u.len() {
        if !crate::av2_recon::work_tick("av2_itx:436") { break; }
        let a = u[i] * cosa - v[i] * sina;
        let b = u[i] * sina + v[i] * cosa;
        u[i] = iclip((a + 128 - (a < 0) as i32) >> 8, min, max);
        v[i] = iclip((b + 128 - (b < 0) as i32) >> 8, min, max);
    }
}

/// Add the transformed residual `c` to the prediction in `dst`, rounding by
/// `(+rnd) >> shift` and clipping to `[0, bitdepth_max]` (dav2d `residual_add`).
/// `dpcm`: 0 = direct add; 1 = horizontal accumulate; 2 = vertical accumulate.
/// This is the final reconstruction step — residual + prediction → pixels.
pub fn residual_add(dst: &mut [i32], stride: usize, c: &[i32], w: usize, h: usize, rnd: i32, shift: u32, dpcm: u32, bitdepth_max: i32) {
    if c.len() < w * h || dst.len() < (h - 1) * stride + w {
        crate::dlog!("[RESADD-BAD] c.len={} dst.len={} w={w} h={h} stride={stride} dpcm={dpcm}", c.len(), dst.len());
    }
    let clip = |p: i32| p.clamp(0, bitdepth_max);
    match dpcm {
        0 => {
            for y in 0..h {
                if !crate::av2_recon::work_tick("av2_itx:455") { break; }
                for x in 0..w {
                    if !crate::av2_recon::work_tick("av2_itx:456") { break; }
                    let o = y * stride + x;
                    dst[o] = clip(dst[o] + ((c[y * w + x] + rnd) >> shift));
                }
            }
        }
        1 => {
            for y in 0..h {
                if !crate::av2_recon::work_tick("av2_itx:463") { break; }
                let mut acc = 0;
                for x in 0..w {
                    if !crate::av2_recon::work_tick("av2_itx:465") { break; }
                    acc += (c[y * w + x] + rnd) >> shift;
                    let o = y * stride + x;
                    dst[o] = clip(dst[o] + acc);
                }
            }
        }
        2 => {
            for x in 0..w {
                if !crate::av2_recon::work_tick("av2_itx:473") { break; }
                let mut acc = 0;
                for y in 0..h {
                    if !crate::av2_recon::work_tick("av2_itx:475") { break; }
                    acc += (c[y * w + x] + rnd) >> shift;
                    let o = y * stride + x;
                    dst[o] = clip(dst[o] + acc);
                }
            }
        }
        _ => unreachable!("dpcm flag {dpcm}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn itx32_block_8_0_repro() {
        let nz: &[(usize, i32)] = &[(0,-11514), (1,-3505), (3,-399), (5,-142), (7,-85), (8,-28), (9,-28), (10,-28), (11,-28), (13,-28), (15,-28), (32,-2337), (65,28), (79,28), (96,-285), (100,-28), (136,28), (160,-85), (177,-28), (189,28), (224,-57), (279,28), (288,-28), (298,-28), (354,28), (384,-28), (416,-28), (420,28), (444,-28), (457,28), (463,28), (470,-28), (529,28), (587,28), (606,-28), (608,-28), (609,-28), (615,28), (654,-28), (679,-28), (682,28), (685,28), (687,28), (705,-28), (722,28), (723,28), (736,-28), (770,28), (814,-28), (854,28), (896,-28), (943,28), (945,28), (972,28), (1009,28)];
        let mut coeff = vec![0i32; 1024];
        for &(i, v) in nz { coeff[i] = v; }
        let mut res = vec![0i32; 1024];
        super::inv_txfm_2d(&coeff, 3, 3, 0, 0, &mut res);
        for r in 16..26 {
            eprintln!("ITXR r{} c30={} c31={}", r, res[r*32+30], res[r*32+31]);
        }
    }

    use super::*;

    #[test]
    fn dct4_dc_is_flat() {
        // DC-only input → constant output of dc*64 (b-part is 0).
        let mut c = [5, 0, 0, 0];
        inv_dct4_1d(&mut c, 1);
        assert_eq!(c, [320, 320, 320, 320]);
    }

    #[test]
    fn dct4_impulse_matches_butterfly() {
        // c1=1 → a=0, b0=83, b1=35 → [83,35,-35,-83].
        let mut c = [0, 1, 0, 0];
        inv_dct4_1d(&mut c, 1);
        assert_eq!(c, [83, 35, -35, -83]);
    }

    #[test]
    fn dct8_odd_impulse_exercises_kernel() {
        // c1=1 (odd) → even part 0; b[i] = DCT8_KERNEL[i*4+0] = first column;
        // c[i]=b[i], c[7-i]=-b[i].
        let mut c = [0, 1, 0, 0, 0, 0, 0, 0];
        inv_dct8_1d(&mut c, 1);
        assert_eq!(c, [89, 75, 50, 18, -18, -50, -75, -89]);
    }

    #[test]
    fn dct8_dc_is_flat() {
        let mut c = [1, 0, 0, 0, 0, 0, 0, 0];
        inv_dct8_1d(&mut c, 1);
        assert_eq!(c, [64; 8]); // even recursion: dct4 DC → 64, kernel sees zeros
    }

    #[test]
    fn adst4_columns() {
        // c0=1 → first column [18,50,75,89]; c1=1 → second column [50,89,18,-75].
        let mut c = [1, 0, 0, 0];
        inv_adst4_1d(&mut c, 1);
        assert_eq!(c, [18, 50, 75, 89]);
        let mut c = [0, 1, 0, 0];
        inv_adst4_1d(&mut c, 1);
        assert_eq!(c, [50, 89, 18, -75]);
    }

    #[test]
    fn flipadst8_is_adst8_reversed() {
        // flipadst8 uses the adst8 kernel with output reversed.
        let mut a = [3, -2, 5, 0, 1, 7, -4, 2];
        let mut f = a;
        inv_adst8_1d(&mut a, 1);
        inv_flipadst8_1d(&mut f, 1);
        a.reverse();
        assert_eq!(a, f);
    }

    #[test]
    fn identity_scales() {
        let mut c = [1, 2, 3, 4];
        inv_identity4_1d(&mut c, 1);
        assert_eq!(c, [128, 256, 384, 512]);
        let mut c = [1i32; 8];
        inv_identity8_1d(&mut c, 1);
        assert_eq!(c, [181; 8]);
    }

    #[test]
    fn dc_only_general_matches_fast_path() {
        // The DC-only DCT_DCT fast path is an independent shortcut; the full 2-D
        // pipeline (row → scale → col → scale) must produce a flat block equal to it.
        // This cross-validates the whole 2-D transform against an independent oracle.
        for lsize in 0..5 {
            // lsize 4 (64x64) exercises the idct64 2x upsampling: the flat residual
            // must stay flat through replication and equal the DC fast path.
            let n = 4usize << lsize;
            let coeff_n = n.min(32); // 64x64 has only a 32x32 coeff block
            for &dc in &[1, 16, 64, 100, 255, -50, -200, 511] {
                let mut coeff = vec![0i32; coeff_n * coeff_n];
                coeff[0] = dc;
                let mut out = vec![0i32; n * n];
                inv_txfm_2d(&coeff, lsize, lsize, 0, 0, &mut out); // DCT_DCT
                let fast = inv_txfm_dc(dc, lsize, lsize);
                assert!(
                    out.iter().all(|&r| r == fast),
                    "lsize={lsize} dc={dc}: general path {:?} != flat fast {fast}",
                    &out[..n.min(4)]
                );
            }
        }
    }

    #[test]
    fn spine_composes_dc_reconstruction() {
        // The full reconstruction spine, in isolation: coefficient token → dequant →
        // DC coeff → 2-D DCT inverse → flat residual → add to prediction → pixels.
        use crate::av2_dequant::{cf_max, dequant_coeff, dq_lookup, dq_shift};
        let dq = dq_lookup(80);
        let cf0 = dequant_coeff(3, dq, dq_shift(0, 1), cf_max(8), 0, false) as i32;
        let mut coeff = vec![0i32; 64]; // 8x8
        coeff[0] = cf0;
        let mut residual = vec![0i32; 64];
        inv_txfm_2d(&coeff, 1, 1, 0, 0, &mut residual);
        // the DCT of a DC coeff is flat and equals the independent DC fast path
        assert!(residual.iter().all(|&r| r == inv_txfm_dc(cf0, 1, 1)));
        // add to a flat prediction → uniform, in-range pixels
        let mut pixels = vec![128i32; 64];
        residual_add(&mut pixels, 8, &residual, 8, 8, 0, 0, 0, 255);
        let expect = (128 + residual[0]).clamp(0, 255);
        assert!(pixels.iter().all(|&p| p == expect));
    }

    #[test]
    fn residual_add_direct() {
        // dpcm 0: dst = clip(pred + residual). rnd=0, shift=0.
        let mut dst = [100; 4]; // prediction
        let c = [8, -30, 200, 5];
        residual_add(&mut dst, 4, &c, 4, 1, 0, 0, 0, 255);
        assert_eq!(dst, [108, 70, 255, 105]); // 100+200 clips to 255
        // lower clamp
        let mut dst = [10; 2];
        residual_add(&mut dst, 2, &[-50, 3], 2, 1, 0, 0, 0, 255);
        assert_eq!(dst, [0, 13]);
    }

    #[test]
    fn residual_add_rounding_shift() {
        // (residual + rnd) >> shift, then add to pred.
        let mut dst = [0; 2];
        residual_add(&mut dst, 2, &[12, 13], 2, 1, 4, 3, 0, 255);
        assert_eq!(dst, [(12 + 4) >> 3, (13 + 4) >> 3]); // [2, 2]
    }

    #[test]
    fn residual_add_dpcm_accumulate() {
        // dpcm 1: horizontal running sum across each row.
        let mut dst = [0; 4];
        residual_add(&mut dst, 4, &[4, 4, 4, 4], 4, 1, 0, 0, 1, 255);
        assert_eq!(dst, [4, 8, 12, 16]);
        // dpcm 2: vertical running sum down each column.
        let mut dst = [0; 4]; // 1 wide, 4 tall, stride 1
        residual_add(&mut dst, 1, &[4, 4, 4, 4], 1, 4, 0, 0, 2, 255);
        assert_eq!(dst, [4, 8, 12, 16]);
    }

    #[test]
    fn wht4_1d_known_vectors() {
        let mut c = [8, 0, 0, 0];
        inv_wht4_1d(&mut c, 1);
        assert_eq!(c, [4, 4, 4, 4]);
        let mut c = [0, 0, 0, 8];
        inv_wht4_1d(&mut c, 1);
        assert_eq!(c, [4, -4, 4, -4]);
    }

    #[test]
    fn wht4x4_dc_is_flat() {
        // coeff[0]=64 → row WHT of [8,0,0,0]=[4,4,4,4], col WHT of [4,0,0,0]=[2,2,2,2].
        let mut coeff = vec![0i32; 16];
        coeff[0] = 64;
        let mut out = [0i32; 16];
        inv_txfm_wht4x4(&coeff, &mut out);
        assert_eq!(out, [2i32; 16]);
    }

    #[test]
    fn rect_dc_general_matches_fast_path() {
        // Extends the DC oracle to rectangular TX: the *181 read-scaling general
        // path must still produce a flat block equal to the rect DC fast path.
        for &(lw, lh) in &[
            (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
            (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2),
        ] {
            let (w, h) = (4usize << lw, 4usize << lh);
            for &dc in &[1, 16, 64, 100, -50, 255] {
                let mut coeff = vec![0i32; w * h];
                coeff[0] = dc;
                let mut out = vec![0i32; w * h];
                inv_txfm_2d(&coeff, lw, lh, 0, 0, &mut out);
                let fast = inv_txfm_dc(dc, lw, lh);
                assert!(
                    out.iter().all(|&r| r == fast),
                    "lw={lw} lh={lh} dc={dc}: {:?} != {fast}",
                    &out[..4]
                );
            }
        }
    }

    #[test]
    fn dc_zero_is_zero_residual() {
        let coeff = vec![0i32; 8 * 8];
        let mut out = vec![0i32; 8 * 8];
        inv_txfm_2d(&coeff, 1, 1, 0, 0, &mut out);
        assert!(out.iter().all(|&r| r == 0));
    }

    #[test]
    fn cctx_identity_and_rotation() {
        // (sina=0, cosa=256) → identity; (sina=256, cosa=0) → 90° rotation (u,v)->(-v,u).
        let mut u = [3, -7, 100];
        let mut v = [5, 11, -40];
        let (u0, v0) = (u, v);
        cctx(&mut u, &mut v, 0, 256, 8);
        assert_eq!((u, v), (u0, v0));
        let mut u = [3, -7, 100];
        let mut v = [5, 11, -40];
        cctx(&mut u, &mut v, 256, 0, 8);
        for i in 0..3 {
            assert_eq!(u[i], -v0[i], "u[{i}]");
            assert_eq!(v[i], u0[i], "v[{i}]");
        }
    }

    #[test]
    fn strided_matches_contiguous() {
        // A 1-D transform on a strided column must equal the contiguous result.
        let input = [7, -3, 4, 9];
        let mut contig = input;
        inv_dct4_1d(&mut contig, 1);
        let mut strided = [0i32; 4 * 3];
        for i in 0..4 {
            strided[i * 3] = input[i];
        }
        inv_dct4_1d(&mut strided, 3);
        for i in 0..4 {
            assert_eq!(strided[i * 3], contig[i]);
        }
    }
}

#[cfg(test)]
mod rect_itx_test {
    use super::*;
    /// Regression vs dav2d ground truth: a wide-angle-remapped 4×8 chroma block (dav clip
    /// corr_av2_2f, chroma (140,8) U) with dequantized coeffs at {0,1,8,9,10}. dav2d's txtp for
    /// this block is DCT_ADST (row=ADST, col=DCT) via `wide_angle_remap` (D45→HOR_UP), NOT DCT_DCT
    /// — so this exercises the rectangular ADST-row/DCT-col path against a captured reference.
    #[test]
    fn rect_4x8_dct_adst_matches_dav2d() {
        let mut coeff = [0i32; 32];
        coeff[0] = 624; coeff[1] = -312; coeff[8] = -156; coeff[9] = -156; coeff[10] = 156;
        let mut out = [0i32; 32];
        // DCT_ADST (byte 2) → row_ty = T1D[2] = ADST(1), col_ty = T1D[0] = DCT(0).
        inv_txfm_2d(&coeff, 0, 1, 1, 0, &mut out); // 4×8, ADST row / DCT col
        #[rustfmt::skip]
        let dav: [i32; 32] = [
            -2, -2, 4, 10,  -3, -3, 5, 14,  -4, -5, 8, 21,  -4, -3, 11, 27,
            -1, 3, 17, 30,   3, 11, 22, 30,  8, 21, 26, 28,  11, 26, 29, 26,
        ];
        assert_eq!(out, dav, "4×8 DCT_ADST residual must match dav2d ground truth");
    }
}
