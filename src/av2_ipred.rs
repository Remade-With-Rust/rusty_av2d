//! AV2 intra predictors — pure pixel-domain kernels, bit-exact from dav2d
//! `ipred_tmpl.c` (base case: no MRL / IBP / multi-reference extensions yet).
//!
//! Neighbour model (matching dav2d's `topleft` pointer): `top[x]` is `topleft[1+x]`
//! (the above row, `top[width]` = the "right" extension); `left[y]` is
//! `topleft[-(y+1)]` (the left column, `left[height]` = the "bottom" extension);
//! `corner` is `topleft[0]`. Predictors write a `w`×`h` block into `dst` (row
//! stride `stride`). Arithmetic in `i32`; values stay in pixel range by construction.
//!
//! Unit-tested in isolation (flat→flat for every mode, exact copy for V/H, known
//! Paeth/DC values). Wired into reconstruction with the stateful pipeline.

/// AV2 smooth weights (dav2d `dav2d_sm_weights[scale][i]`), zero-padded to 64.
static SM_WEIGHTS: [[u8; 64]; 3] = [
    pad(&[32, 8, 2]),
    pad(&[32, 16, 8, 4, 2, 1]),
    pad(&[32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1]),
];

const fn pad(prefix: &[u8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut i = 0;
    while i < prefix.len() {
        out[i] = prefix[i];
        i += 1;
    }
    out
}

#[inline]
fn ulog2(x: usize) -> u32 {
    (usize::BITS - 1) - (x as usize).leading_zeros()
}

fn splat(dst: &mut [i32], stride: usize, w: usize, h: usize, v: i32) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:36") { break; }
        for x in 0..w {
            if !crate::av2_recon::work_tick("ipred:37") { break; }
            dst[y * stride + x] = v;
        }
    }
}

/// Reciprocal table for `fast_div32_dc` (dav2d `dav2d_div_recip`, 129 entries). Also used by
/// the warp least-squares solve (`resolve_divisor_64`, brick B / av2_refmvs).
pub(crate) static DIV_RECIP: [u16; 129] = [
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462, //
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417, //
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381, //
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350, //
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324, //
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302, //
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282, //
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265, //
    264, 263, 262, 261, 260, 259, 258, 257, 256,
];

/// Fixed-point `num / den` for `den ∈ 1..=255` (dav2d `fast_div32_dc`).
pub(crate) fn fast_div32_dc(num: u32, den: u32) -> i32 {
    let mut shift = ulog2(den as usize);
    let rem = den - (1 << shift);
    let idx = (rem << (7 - shift)) as usize;
    shift += 9;
    (((num * DIV_RECIP[idx] as u32) + ((1u32 << shift) >> 1)) >> shift) as i32
}

/// DC prediction — average of top + left (dav2d `dc_gen`, base case). Power-of-two
/// `width+height` uses an exact shift; other (rectangular) sizes use the
/// reciprocal-table division, clamped to the pixel range.
pub fn ipred_dc(dst: &mut [i32], stride: usize, top: &[i32], left: &[i32], w: usize, h: usize, bitdepth_max: i32) {
    let n_pel = w + h;
    let mut dc: u32 = 0;
    for &t in &top[..w] {
        dc += t as u32;
    }
    for &l in &left[..h] {
        dc += l as u32;
    }
    let dc = if n_pel & (n_pel - 1) == 0 {
        ((dc + w as u32) >> n_pel.trailing_zeros()) as i32
    } else {
        fast_div32_dc(dc, n_pel as u32).clamp(0, bitdepth_max)
    };
    splat(dst, stride, w, h, dc);
}

/// DC with only the left column available (dav2d/avm `dc_left`) — average of `left[..h]`.
pub fn ipred_dc_left(dst: &mut [i32], stride: usize, left: &[i32], w: usize, h: usize) {
    let mut sum: u32 = 0;
    for &l in &left[..h] {
        sum += l as u32;
    }
    let dc = ((sum + (h as u32 >> 1)) / h as u32) as i32;
    splat(dst, stride, w, h, dc);
}

/// DC with only the top row available (avm `dc_top`) — average of `top[..w]`.
pub fn ipred_dc_top(dst: &mut [i32], stride: usize, top: &[i32], w: usize, h: usize) {
    let mut sum: u32 = 0;
    for &t in &top[..w] {
        sum += t as u32;
    }
    let dc = ((sum + (w as u32 >> 1)) / w as u32) as i32;
    splat(dst, stride, w, h, dc);
}

/// DC with neither neighbour available (avm `dc_128`) — `1 << (bitdepth-1)`.
pub fn ipred_dc_128(dst: &mut [i32], stride: usize, w: usize, h: usize, bitdepth_max: i32) {
    splat(dst, stride, w, h, (bitdepth_max + 1) >> 1);
}

/// IBP (Intra Boundary Prediction) weight rows (avm `ibp_weights[5][16]`), indexed by
/// `IBP_SIZE_TO_WIDX[dim>>3]`. Blend shift is `IBP_WEIGHT_SHIFT = DIV_LUT_BITS = 7`
/// (`IBP_WEIGHT_REF = 128`).
pub const IBP_WEIGHTS: [[i32; 16]; 5] = [
    [96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [86, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [77, 90, 102, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [71, 78, 86, 92, 100, 107, 114, 121, 0, 0, 0, 0, 0, 0, 0, 0],
    [68, 72, 76, 79, 83, 87, 90, 94, 98, 102, 106, 109, 113, 117, 121, 124],
];
pub const IBP_SIZE_TO_WIDX: [usize; 9] = [0, 1, 2, 0, 3, 0, 0, 0, 4];
const IBP_SHIFT: i32 = 7;
const IBP_REF: i32 = 128;
#[inline]
fn ibp_blend(reference: i32, dc: i32, weight: i32) -> i32 {
    (reference * (IBP_REF - weight) + dc * weight + (1 << (IBP_SHIFT - 1))) >> IBP_SHIFT
}

/// Apply the IBP boundary gradient to an already-DC-filled `dst` (avm `ibp_dc*_predictor`).
/// `have_top`/`have_left` select which of the three avm variants (dc_top / dc_left / dc) runs.
/// Precondition: `dst` holds the flat DC value; `top`/`left` are the neighbour edges.
pub fn ipred_ibp_dc(
    dst: &mut [i32], stride: usize, top: &[i32], left: &[i32], w: usize, h: usize,
    have_top: bool, have_left: bool,
) {
    // HARDENING: bound the blend extents by the actual edge/dst buffers (a corrupt stream
    // can present dims larger than the gathered edges).
    let w = w.min(top.len()).min(if stride > 0 { stride } else { w });
    let h = h.min(left.len()).min(if stride > 0 { dst.len() / stride.max(1) } else { h });
    if have_left && !have_top {
        // ibp_dc_left: blend first bw/4 columns toward left[r].
        let wts = &IBP_WEIGHTS[IBP_SIZE_TO_WIDX[(w >> 3).min(IBP_SIZE_TO_WIDX.len() - 1)]];
        let len = w >> 2;
        let len = len.min(wts.len()); // HARDENING: corrupt dims vs the weight row
        for r in 0..h {
            if !crate::av2_recon::work_tick("ipred:145") { break; }
            for c in 0..len {
                if !crate::av2_recon::work_tick("ipred:146") { break; }
                dst[r * stride + c] = ibp_blend(left[r], dst[r * stride + c], wts[c]);
            }
        }
    } else if have_top && !have_left {
        // ibp_dc_top: blend first bh/4 rows toward top[c].
        let wts = &IBP_WEIGHTS[IBP_SIZE_TO_WIDX[(h >> 3).min(IBP_SIZE_TO_WIDX.len() - 1)]];
        let len = (h >> 2).min(wts.len()); // HARDENING
        for r in 0..len {
            if !crate::av2_recon::work_tick("ipred:154") { break; }
            for c in 0..w {
                if !crate::av2_recon::work_tick("ipred:155") { break; }
                dst[r * stride + c] = ibp_blend(top[c], dst[r * stride + c], wts[r]);
            }
        }
    } else if have_top && have_left {
        // ibp_dc: top rows (0..bh/4) toward top, then left cols (0..bw/4) toward left,
        // with the shorter dimension's band suppressed on the overlap (row_start/col_start).
        let (len_h, len_w) = (h >> 2, w >> 2);
        let (row_start, col_start) = if w >= h { (len_h, 0) } else { (0, len_w) };
        let wts_t = &IBP_WEIGHTS[IBP_SIZE_TO_WIDX[(h >> 3).min(IBP_SIZE_TO_WIDX.len() - 1)]];
        for r in 0..len_h {
            if !crate::av2_recon::work_tick("ipred:165") { break; }
            for c in col_start..w.min(top.len()) {
                if !crate::av2_recon::work_tick("ipred:166") { break; }
                dst[r * stride + c] = ibp_blend(top[c], dst[r * stride + c], wts_t[r.min(wts_t.len() - 1)]);
            }
        }
        let wts_l = &IBP_WEIGHTS[IBP_SIZE_TO_WIDX[(w >> 3).min(IBP_SIZE_TO_WIDX.len() - 1)]];
        let len_w = len_w.min(wts_l.len()); // HARDENING
        for r in row_start..h.min(left.len()) {
            if !crate::av2_recon::work_tick("ipred:172") { break; }
            for c in 0..len_w {
                if !crate::av2_recon::work_tick("ipred:173") { break; }
                dst[r * stride + c] = ibp_blend(left[r], dst[r * stride + c], wts_l[c]);
            }
        }
    }
}

// ===== AV2 IDIF directional prediction (avm `reconintra.c`) =====
// The IDIF projection core is bit-identical to dav2d's z1/z2/z3, but AV2 wraps it with its own
// edge-buffer prep + edge filtering. These take OFFSET buffers: `above[ao-1]` is the corner,
// `above[ao+i]` the i-th top sample; `left[lo-1]` the corner, `left[lo+i]` the i-th left sample.

/// avm `intra_edge_filter_strength(bs0, bs1, delta, type)` — DIFFERENT from dav2d's
/// `get_filter_strength` (extra `≤12` bucket, distinct thresholds). `type` = neighbour-smooth.
thread_local! {
    pub static DIR_DBG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub fn intra_edge_filter_strength(bs0: i32, bs1: i32, delta: i32, ty: i32) -> usize {
    let d = delta.abs();
    let blk_wh = bs0 + bs1;
    let mut s = 0;
    if ty == 0 {
        if blk_wh <= 8 {
            if d >= 56 { s = 1; }
        } else if blk_wh <= 12 {
            if d >= 40 { s = 1; }
        } else if blk_wh <= 16 {
            if d >= 40 { s = 1; }
        } else if blk_wh <= 24 {
            if d >= 8 { s = 1; }
            if d >= 16 { s = 2; }
            if d >= 32 { s = 3; }
        } else if blk_wh <= 32 {
            if d >= 1 { s = 1; }
            if d >= 4 { s = 2; }
            if d >= 32 { s = 3; }
        } else if d >= 1 {
            s = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 { s = 1; }
        if d >= 64 { s = 2; }
    } else if blk_wh <= 16 {
        if d >= 20 { s = 1; }
        if d >= 48 { s = 2; }
    } else if blk_wh <= 24 {
        if d >= 4 { s = 3; }
    } else if d >= 1 {
        s = 3;
    }
    s
}

/// avm `av2_filter_intra_edge_high(p, sz, strength)` applied to `buf[start .. start+sz]`,
/// filtering positions `1..sz` in-place (position 0 unchanged), edge-clamped.
pub fn av2_filter_intra_edge(buf: &mut [i32], start: usize, sz: usize, strength: usize) {
    if strength == 0 {
        return;
    }
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let k = &KERNEL[strength - 1];
    let edge: Vec<i32> = buf[start..start + sz].to_vec();
    for i in 1..sz {
        if !crate::av2_recon::work_tick("ipred:235") { break; }
        let mut s = 0i32;
        for (j, &kj) in k.iter().enumerate() {
            let idx = (i as i32 - 2 + j as i32).clamp(0, sz as i32 - 1) as usize;
            s += edge[idx] * kj;
        }
        buf[start + i] = (s + 8) >> 4;
    }
}

/// avm `filter_intra_edge_corner_high` (kernel {5,6,5}) — blends left[0], corner, top[0] into
/// the shared corner. `ao`/`lo` are the offsets of the first top / first left sample.
pub fn filter_intra_edge_corner(above: &mut [i32], ao: usize, left: &mut [i32], lo: usize) {
    let s = left[lo] * 5 + above[ao - 1] * 6 + above[ao] * 5;
    let s = (s + 8) >> 4;
    above[ao - 1] = s;
    left[lo - 1] = s;
}

/// Raw IDIF zone-1 (`0<angle<90`) projection into `dst` from offset buffer `above`
/// (`above[ao-1]`=corner). avm `av2_highbd_dr_prediction_z1_idif_c`. `dx = dr_intra_derivative`.
#[allow(clippy::too_many_arguments)]
pub fn dr_z1_idif(dst: &mut [i32], stride: usize, w: usize, h: usize, above: &[i32], ao: usize, dx: i32, mrl: i32, chroma: bool, bdmax: i32) {
    let ai = |i: i32| above[(ao as i32 + i) as usize];
    let max_base_x = (w + h) as i32 - 1 + (mrl << 1);
    let mut x = dx * (1 + mrl);
    for r in 0..h {
        if !crate::av2_recon::work_tick("ipred:261") { break; }
        let base = x >> 6;
        let shift = ((x & 0x3F) >> 1) as usize;
        if base > max_base_x {
            for rr in r..h {
                if !crate::av2_recon::work_tick("ipred:265") { break; }
                for c in 0..w {
                    if !crate::av2_recon::work_tick("ipred:266") { break; }
                    dst[rr * stride + c] = ai(max_base_x);
                }
            }
            return;
        }
        let f = &DR_INTERP_FILTER[shift];
        let mut b = base;
        for c in 0..w {
            if !crate::av2_recon::work_tick("ipred:274") { break; }
            if b <= max_base_x {
                // Chroma uses the standard 2-tap linear interp (avm ipred_z1 !is_luma path);
                // luma uses the 4-tap IDIF. Same base index; the 2-tap = the f[1]/f[2] centre taps.
                dst[r * stride + c] = if chroma {
                    let v = (32 - shift as i32) * ai(b) + shift as i32 * ai(b + 1);
                    ((v + 16) >> 5).clamp(0, bdmax)
                } else {
                    let v = f[0] * ai(b - 1) + f[1] * ai(b) + f[2] * ai(b + 1) + f[3] * ai(b + 2);
                    ((v + 64) >> 7).clamp(0, bdmax)
                };
            } else {
                dst[r * stride + c] = ai(max_base_x);
            }
            b += 1;
        }
        x += dx;
    }
}

/// Raw IDIF zone-3 (`180<angle<270`) projection. avm `av2_highbd_dr_prediction_z3_idif_c`.
#[allow(clippy::too_many_arguments)]
pub fn dr_z3_idif(dst: &mut [i32], stride: usize, w: usize, h: usize, left: &[i32], lo: usize, dy: i32, mrl: i32, chroma: bool, bdmax: i32) {
    let li = |i: i32| left[(lo as i32 + i) as usize];
    let max_base_y = (w + h) as i32 - 1 + (mrl << 1);
    let mut y = dy * (1 + mrl);
    for c in 0..w {
        if !crate::av2_recon::work_tick("ipred:300") { break; }
        let base0 = y >> 6;
        let shift = ((y & 0x3F) >> 1) as usize;
        let f = &DR_INTERP_FILTER[shift];
        let mut base = base0;
        for r in 0..h {
            if !crate::av2_recon::work_tick("ipred:305") { break; }
            if base <= max_base_y {
                dst[r * stride + c] = if chroma {
                    let v = (32 - shift as i32) * li(base) + shift as i32 * li(base + 1);
                    ((v + 16) >> 5).clamp(0, bdmax)
                } else {
                    let v = f[0] * li(base - 1) + f[1] * li(base) + f[2] * li(base + 1) + f[3] * li(base + 2);
                    ((v + 64) >> 7).clamp(0, bdmax)
                };
                base += 1;
            } else {
                for rr in r..h {
                    if !crate::av2_recon::work_tick("ipred:316") { break; }
                    dst[rr * stride + c] = li(max_base_y);
                }
                break;
            }
        }
        y += dy;
    }
}

/// Raw IDIF zone-2 (`90<angle<180`) projection reading from both edges. avm
/// `av2_highbd_dr_prediction_z2_idif_c`.
#[allow(clippy::too_many_arguments)]
pub fn dr_z2_idif(dst: &mut [i32], stride: usize, w: usize, h: usize, above: &[i32], ao: usize, left: &[i32], lo: usize, dx: i32, dy: i32, mrl: i32, chroma: bool, bdmax: i32) {
    let ai = |i: i32| above[(ao as i32 + i) as usize];
    let li = |i: i32| left[(lo as i32 + i) as usize];
    let min_base_x = -1 - mrl;
    for r in 0..h {
        if !crate::av2_recon::work_tick("ipred:333") { break; }
        for c in 0..w {
            if !crate::av2_recon::work_tick("ipred:334") { break; }
            let yv = (r as i32) + 1;
            let x = ((c as i32) << 6) - (yv + mrl) * dx;
            let base_x = x >> 6;
            let val = if base_x >= min_base_x {
                let shift = ((x & 0x3F) >> 1) as usize;
                let f = &DR_INTERP_FILTER[shift];
                if chroma {
                    let v = (32 - shift as i32) * ai(base_x) + shift as i32 * ai(base_x + 1);
                    ((v + 16) >> 5).clamp(0, bdmax)
                } else {
                    let v = f[0] * ai(base_x - 1) + f[1] * ai(base_x) + f[2] * ai(base_x + 1) + f[3] * ai(base_x + 2);
                    ((v + 64) >> 7).clamp(0, bdmax)
                }
            } else {
                let xv = (c as i32) + 1;
                let yy = ((r as i32) << 6) - (xv + mrl) * dy;
                let base_y = yy >> 6;
                let shift = ((yy & 0x3F) >> 1) as usize;
                let f = &DR_INTERP_FILTER[shift];
                if chroma {
                    let v = (32 - shift as i32) * li(base_y) + shift as i32 * li(base_y + 1);
                    ((v + 16) >> 5).clamp(0, bdmax)
                } else {
                    let v = f[0] * li(base_y - 1) + f[1] * li(base_y) + f[2] * li(base_y + 1) + f[3] * li(base_y + 2);
                    ((v + 64) >> 7).clamp(0, bdmax)
                }
            };
            dst[r * stride + c] = val;
        }
    }
}

// ===== Directional IBP (avm): blend the z1/z3 prediction with a secondary opposite-edge
// projection using generated per-mode weights. Applies for tx != 4x4, angle_delta%2==0, and the
// `is_ibp_enabled` directional modes.

/// avm `angle_to_mode_index[90]` — maps an angle (0..89) to the IBP weight mode index (0..15;
/// 15 = "no dedicated weights").
static ANGLE_TO_MODE_INDEX: [u8; 90] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, //
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, //
    15, 0, 0, 15, 0, 0, 14, 0, 0, 13, 0, 0, 12, 0, 0, 11, 0, 0, //
    10, 0, 0, 0, 9, 0, 0, 8, 0, 0, 7, 0, 0, 6, 0, 0, 5, 0, //
    0, 4, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0,
];
/// avm `is_ibp_enabled[16]`.
static IS_IBP_ENABLED: [bool; 16] = [
    false, true, false, false, true, false, true, false, true, false, false, true, false, true,
    false, true,
];

thread_local! {
    /// Lazily-generated IBP directional weight table `[row][col][mode_idx]` (avm `init_ibp_info`).
    static IBP_DIR_WEIGHTS: std::cell::OnceCell<Box<[[[i32; 17]; 16]; 16]>> =
        const { std::cell::OnceCell::new() };
}

/// avm `div_lut[DIV_LUT_NUM+1]` (129 entries, precision `DIV_LUT_PREC_BITS=9`, `DIV_LUT_BITS=7`).
/// NOTE: distinct from dav2d's warp DIV_LUT (257 entries / prec 14) — the IBP weight generation
/// needs THIS table + convention, else weights are off by 1 ULP.
#[rustfmt::skip]
static DIV_LUT_AVM: [i32; 129] = [
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256,
];

/// avm `resolve_divisor_32` (warped_motion.h): `1/D ≈ div / 2^shift` at precision 9. Returns
/// `(shift, div)`. Distinct from dav2d's (8-bit/prec-14) version in warpmv.rs.
fn avm_resolve_divisor_32(d: u32) -> (i32, i32) {
    let msb = ulog2(d as usize) as i32; // get_msb
    let e = (d - (1u32 << msb)) as i32;
    let f = if msb > 7 {
        (e + (1 << (msb - 8))) >> (msb - 7) // ROUND_POWER_OF_TWO(e, msb-7)
    } else {
        e << (7 - msb)
    };
    (msb + 9, DIV_LUT_AVM[f as usize]) // shift += DIV_LUT_PREC_BITS(9)
}

/// Generate one mode's weights (avm `av2_dr_prediction_z1_info`) into `w[r][c][mode_idx]`.
fn ibp_gen_mode(w: &mut [[[i32; 17]; 16]; 16], dy: i32, mode_idx: usize) {
    for (r, wr) in w.iter_mut().enumerate() {
        let mut y = dy;
        for wc in wr.iter_mut() {
            if !crate::av2_recon::work_tick("ipred:425") { break; }
            let dist = (((r as i32) + 1) << 6) + y;
            let (shift0, div) = avm_resolve_divisor_32(dist as u32);
            let shift = shift0 - 7; // shift -= DIV_LUT_BITS(7)
            let weight0 = (y * div + (1 << (shift - 1))) >> shift;
            wc[mode_idx] = weight0;
            y += dy;
        }
    }
}

/// Run `f` with the (lazily built) IBP directional weight table.
fn with_ibp_weights<R>(f: impl FnOnce(&[[[i32; 17]; 16]; 16]) -> R) -> R {
    IBP_DIR_WEIGHTS.with(|cell| {
        let w = cell.get_or_init(|| {
            let mut w = Box::new([[[128i32; 17]; 16]; 16]); // IBP_WEIGHT_MAX default
            // avm init_ibp_info: (V,-2),(D67,-2),(D45,-2),(D67,0),(D45,0),(D67,2),(D45,2).
            // mode_to_angle: V=90, D67=67, D45=45. dy = dr_intra_derivative[90-angle].
            for &(base, delta) in &[
                (90, -2), (67, -2), (45, -2), (67, 0), (45, 0), (67, 2), (45, 2),
            ] {
                let angle = base + delta * 3;
                let mode_idx = ANGLE_TO_MODE_INDEX[angle as usize] as usize;
                let dy = dr_intra_deriv(90 - angle);
                ibp_gen_mode(&mut w, dy, mode_idx);
            }
            w
        });
        f(w)
    })
}

/// Apply the directional IBP blend to a z1 (`0<angle<90`) or z3 (`180<angle<270`) prediction in
/// `dst`. `above`/`left` are the post-filter offset edge buffers (`ao`/`lo` = first-sample index).
/// No-op unless `angle_delta` is even and the mode has dedicated weights. avm
/// `av2_build_intra_predictors_high_default` IBP tail.
#[allow(clippy::too_many_arguments)]
pub fn apply_dir_ibp(
    dst: &mut [i32], w: usize, h: usize, p_angle: i32, angle_delta: i32, above: &mut [i32],
    ao: usize, left: &mut [i32], lo: usize, bdmax: i32,
) {
    if angle_delta % 2 != 0 {
        return;
    }
    let n = (w + h) as i32;
    let (col_shift, row_shift) = ((w >> 5) as u32, (h >> 5) as u32);
    let mut second = vec![0i32; w * h];
    if p_angle > 0 && p_angle < 90 {
        let mode_idx = ANGLE_TO_MODE_INDEX[p_angle as usize] as usize;
        if mode_idx >= 16 || !IS_IBP_ENABLED[mode_idx] {
            return;
        }
        // second predictor: z3 projection from the left edge, dy = dr_deriv[90-angle], dx=1.
        let dy = dr_intra_deriv(90 - p_angle);
        left[(lo as i32 + n) as usize] = left[(lo as i32 + n - 1) as usize];
        left[(lo as i32 + n + 1) as usize] = left[(lo as i32 + n - 1) as usize];
        dr_z3_idif(&mut second, w, w, h, left, lo, dy, 0, false, bdmax);
        with_ibp_weights(|wt| {
            for r in 0..h {
                if !crate::av2_recon::work_tick("ipred:483") { break; }
                let ri = r >> row_shift;
                for c in 0..w {
                    if !crate::av2_recon::work_tick("ipred:485") { break; }
                    let ci = c >> col_shift;
                    let weight = wt[ri][ci][mode_idx];
                    dst[r * w + c] =
                        (dst[r * w + c] * weight + second[r * w + c] * (128 - weight) + 64) >> 7;
                }
            }
        });
    } else if p_angle > 180 && p_angle < 270 {
        let mode_idx = ANGLE_TO_MODE_INDEX[(270 - p_angle) as usize] as usize;
        if mode_idx >= 16 || !IS_IBP_ENABLED[mode_idx] {
            return;
        }
        // second predictor: z1 projection from the above edge, dx = dr_deriv[angle-180], dy=1.
        let dx = dr_intra_deriv(p_angle - 180);
        above[(ao as i32 + n) as usize] = above[(ao as i32 + n - 1) as usize];
        above[(ao as i32 + n + 1) as usize] = above[(ao as i32 + n - 1) as usize];
        dr_z1_idif(&mut second, w, w, h, above, ao, dx, 0, false, bdmax);
        with_ibp_weights(|wt| {
            for c in 0..w {
                if !crate::av2_recon::work_tick("ipred:504") { break; }
                let ci = c >> col_shift;
                for r in 0..h {
                    if !crate::av2_recon::work_tick("ipred:506") { break; }
                    let ri = r >> row_shift;
                    let weight = wt[ci][ri][mode_idx];
                    dst[r * w + c] =
                        (dst[r * w + c] * weight + second[r * w + c] * (128 - weight) + 64) >> 7;
                }
            }
        });
    }
}

/// Intra-edge smoothing filter (dav2d `filter_edge`). `strength ∈ 1..=3` selects
/// one of three 5-tap kernels (each summing to 16). Endpoints outside
/// `[lim_from, lim_to)` are copied (edge-clamped).
pub fn filter_edge(out: &mut [i32], sz: usize, lim_from: usize, lim_to: usize, inp: &[i32], from: i32, to: i32, strength: usize) {
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let k = &KERNEL[strength - 1];
    let clampi = |i: i32| inp[i.clamp(from, to - 1) as usize];
    let mut i = 0;
    while i < sz.min(lim_from) {
        out[i] = clampi(i as i32);
        i += 1;
    }
    while i < lim_to.min(sz) {
        let mut s = 0i32;
        for j in 0..5 {
            s += clampi(i as i32 - 2 + j as i32) * k[j];
        }
        out[i] = (s + 8) >> 4;
        i += 1;
    }
    while i < sz {
        out[i] = clampi(i as i32);
        i += 1;
    }
}

/// Directional-edge filter strength selector (dav2d `get_filter_strength`).
pub fn get_filter_strength(wh: i32, angle: i32, is_sm: bool) -> i32 {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else if wh <= 8 {
        if angle >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if angle >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 16 {
            return 2;
        }
        if angle >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 4 {
            return 2;
        }
        return 1;
    } else {
        return 3;
    }
    0
}

/// Vertical prediction — copy the above row down every row (dav2d `ipred_v_c`).
pub fn ipred_v(dst: &mut [i32], stride: usize, top: &[i32], w: usize, h: usize) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:601") { break; }
        dst[y * stride..y * stride + w].copy_from_slice(&top[..w]);
    }
}

/// Horizontal prediction — splat each left sample across its row (dav2d `ipred_h_c`).
pub fn ipred_h(dst: &mut [i32], stride: usize, left: &[i32], w: usize, h: usize) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:608") { break; }
        splat_row(&mut dst[y * stride..], w, left[y]);
    }
}

#[inline]
fn splat_row(row: &mut [i32], w: usize, v: i32) {
    for x in 0..w {
        if !crate::av2_recon::work_tick("ipred:615") { break; }
        row[x] = v;
    }
}

/// Paeth prediction — per pixel pick top/left/corner closest to `left+top-corner`
/// (dav2d `ipred_paeth_c`).
pub fn ipred_paeth(dst: &mut [i32], stride: usize, top: &[i32], left: &[i32], corner: i32, w: usize, h: usize) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:623") { break; }
        let l = left[y];
        for x in 0..w {
            if !crate::av2_recon::work_tick("ipred:625") { break; }
            let t = top[x];
            let base = l + t - corner;
            let ldiff = (l - base).abs();
            let tdiff = (t - base).abs();
            let tldiff = (corner - base).abs();
            dst[y * stride + x] = if ldiff <= tdiff && ldiff <= tldiff {
                l
            } else if tdiff <= tldiff {
                t
            } else {
                corner
            };
        }
    }
}

#[inline]
fn smooth_scale(n_pel: usize) -> usize {
    (n_pel >= 64) as usize + (n_pel > 512) as usize
}

/// Smooth prediction — blend of vertical + horizontal interpolation (dav2d `ipred_smooth_c`).
pub fn ipred_smooth(dst: &mut [i32], stride: usize, top: &[i32], left: &[i32], w: usize, h: usize) {
    // HARDENING: bound the predictor extents by the gathered edge buffers (a desynced
    // stream can present block dims larger than the edges/dst that were allocated).
    let w = w.min(top.len());
    let h = h.min(left.len());
    let bwl2 = ulog2(w);
    let bhl2 = ulog2(h);
    let rnd_ver = (h >> 1) as i32;
    let rnd_hor = (w >> 1) as i32;
    let weights = &SM_WEIGHTS[smooth_scale(w * h)];
    let right = top[w];
    let bottom = left[h];
    let h = h.min(weights.len()); // HARDENING: weight row bounds the smooth extent
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:661") { break; }
        let l = left[y];
        let diff_hor = l - right;
        let off_ver = (h - 1 - y) as i32;
        let w_ver = weights[y] as i32;
        for x in 0..w {
            if !crate::av2_recon::work_tick("ipred:666") { break; }
            let above = top[x];
            let mul_ver = (above - bottom) * off_ver;
            let mul_hor = diff_hor * (w - 1 - x) as i32;
            let mut pred_ver = bottom + ((mul_ver + rnd_ver) >> bhl2);
            let mut pred_hor = right + ((mul_hor + rnd_hor) >> bwl2);
            pred_ver += ((above - pred_ver) * w_ver + 32) >> 6;
            pred_hor += ((l - pred_hor) * weights[x.min(weights.len() - 1)] as i32 + 32) >> 6;
            dst[y * stride + x] = (pred_ver + pred_hor + 1) >> 1;
        }
    }
}

/// Vertical smooth — interpolate top→bottom only (dav2d `ipred_smooth_v_c`).
pub fn ipred_smooth_v(dst: &mut [i32], stride: usize, top: &[i32], left: &[i32], w: usize, h: usize) {
    // HARDENING: bound the predictor extents by the gathered edge buffers (a desynced
    // stream can present block dims larger than the edges/dst that were allocated).
    let w = w.min(top.len());
    let h = h.min(left.len());
    let bhl2 = ulog2(h);
    let rnd = (h >> 1) as i32;
    let weights = &SM_WEIGHTS[smooth_scale(w * h)];
    // HARDENING: the weight row + the bottom edge sample bound the extents.
    let h = h.min(weights.len()).min(left.len().saturating_sub(1));
    let bottom = left[h];
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:691") { break; }
        let off = (h - 1 - y) as i32;
        let w_ver = weights[y] as i32;
        for x in 0..w {
            if !crate::av2_recon::work_tick("ipred:694") { break; }
            let above = top[x];
            let mul = (above - bottom) * off;
            let pred = bottom + ((mul + rnd) >> bhl2);
            dst[y * stride + x] = pred + (((above - pred) * w_ver + 32) >> 6);
        }
    }
}

/// Horizontal smooth — interpolate left→right only (dav2d `ipred_smooth_h_c`).
pub fn ipred_smooth_h(dst: &mut [i32], stride: usize, top: &[i32], left: &[i32], w: usize, h: usize) {
    // HARDENING: bound the predictor extents by the gathered edge buffers (a desynced
    // stream can present block dims larger than the edges/dst that were allocated).
    let w = w.min(top.len());
    let h = h.min(left.len());
    let bwl2 = ulog2(w);
    let rnd = (w >> 1) as i32;
    let weights = &SM_WEIGHTS[smooth_scale(w * h)];
    let right = top[w];
    for y in 0..h {
        if !crate::av2_recon::work_tick("ipred:713") { break; }
        let l = left[y];
        let diff = l - right;
        for x in 0..w {
            if !crate::av2_recon::work_tick("ipred:716") { break; }
            let mul = diff * (w - 1 - x) as i32;
            let pred = right + ((mul + rnd) >> bwl2);
            dst[y * stride + x] = pred + (((l - pred) * weights[x.min(weights.len() - 1)] as i32 + 32) >> 6);
        }
    }
}

/// Angle index (0..89) → derivative (dav2d `dav2d_dr_intra_derivative`).
static DR_INTRA_DERIVATIVE: [u16; 90] = [
    0, 4096, 2048, //
    1365, 1024, 819, //
    682, 585, 512, //
    455, 409, 409, 409, 372, //
    341, 292, 273, //
    256, 227, 215, //
    204, 186, 178, //
    170, 157, 151, //
    146, 136, 132, //
    128, 117, 110, //
    107, 99, 97, 97, //
    93, 87, 83, //
    81, 77, 74, //
    73, 69, 66, //
    64, 62, 59, //
    56, 55, 53, //
    50, 49, 47, //
    44, 42, 42, 41, //
    38, 37, 35, //
    32, 31, 30, //
    28, 27, 26, //
    24, 23, 22, //
    20, 19, 18, //
    16, 15, 14, //
    12, 11, 10, 10, 10, //
    9, 8, 7, //
    6, 5, 4, //
    3, 2, 1,
];

/// Raw `dr_intra_derivative[idx]` accessor (0..=89).
pub fn dr_intra_deriv(idx: i32) -> i32 {
    DR_INTRA_DERIVATIVE[idx as usize] as i32
}

/// avm `av2_get_dx(angle)` — derivative for the X projection (z1: `[angle]`, z2: `[180-angle]`).
pub fn av2_get_dx(angle: i32) -> i32 {
    if angle > 0 && angle < 90 {
        DR_INTRA_DERIVATIVE[angle as usize] as i32
    } else if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32
    } else {
        1
    }
}

/// avm `av2_get_dy(angle)` — derivative for the Y projection (z2: `[angle-90]`, z3: `[270-angle]`).
pub fn av2_get_dy(angle: i32) -> i32 {
    if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32
    } else if angle > 180 && angle < 270 {
        DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32
    } else {
        1
    }
}

/// 4-tap fractional interpolation filters (dav2d `dr_interp_filter`), indexed by
/// `shift` 0..31; each `{a,b,c,d}` row sums to 128.
static DR_INTERP_FILTER: [[i32; 4]; 32] = [
    [0, 128, 0, 0], [-2, 127, 4, -1], [-3, 125, 8, -2], [-5, 123, 13, -3],
    [-6, 121, 17, -4], [-7, 118, 22, -5], [-9, 116, 27, -6], [-9, 112, 32, -7],
    [-10, 109, 37, -8], [-11, 106, 41, -8], [-11, 102, 46, -9], [-12, 98, 52, -10],
    [-12, 94, 56, -10], [-12, 90, 61, -11], [-12, 85, 66, -11], [-12, 81, 71, -12],
    [-12, 76, 76, -12], [-12, 71, 81, -12], [-11, 66, 85, -12], [-11, 61, 90, -12],
    [-10, 56, 94, -12], [-10, 52, 98, -12], [-9, 46, 102, -11], [-8, 41, 106, -11],
    [-8, 37, 109, -10], [-7, 32, 112, -9], [-6, 27, 116, -9], [-5, 22, 118, -7],
    [-4, 17, 121, -6], [-3, 13, 123, -5], [-2, 8, 125, -3], [-1, 4, 127, -2],
];

/// Zone-1 directional prediction (above-right, `angle < 90°`), luma base path
/// (no MRL / IBP yet) — dav2d `ipred_z1_c`. `topleft_in[0]` is the corner,
/// `topleft_in[1..]` the top edge (≥ `1+width+height` samples); the predictor
/// walks the angle through `top` with the 4-tap fractional filter.
#[allow(clippy::too_many_arguments)]
pub fn ipred_z1(dst: &mut [i32], stride: usize, topleft_in: &[i32], width: usize, height: usize, angle: usize,
                enable_edge_filter: bool, have_top: bool, is_sm_t: bool, max_width: usize, bitdepth_max: i32) {
    assert!(angle < 90);
    let dx = DR_INTRA_DERIVATIVE[angle] as i32;
    let max_base_x = (width + height) as i32 - 1;
    let sz = 1 + width + height;
    let mut filt = [0i32; 141];
    let strength = if enable_edge_filter && have_top {
        get_filter_strength((width + height) as i32, 90 - angle as i32, is_sm_t)
    } else {
        0
    };
    if strength > 0 {
        filter_edge(&mut filt[1..], sz, 1, sz + max_width - width, topleft_in, 0, sz as i32, strength as usize);
    } else {
        filt[1..1 + sz].copy_from_slice(&topleft_in[..sz]);
    }
    filt[0] = filt[1];
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz];
    // top[i] == filt[2 + i]; the 4-tap reads top[base-1..=base+2].
    let edge = filt[(2 + max_base_x) as usize];
    let mut xpos = dx;
    for y in 0..height {
        if !crate::av2_recon::work_tick("ipred:824") { break; }
        if (xpos >> 6) > max_base_x {
            for yy in y..height {
                if !crate::av2_recon::work_tick("ipred:826") { break; }
                splat_row(&mut dst[yy * stride..], width, edge);
            }
            break;
        }
        let f = &DR_INTERP_FILTER[((xpos & 0x3F) >> 1) as usize];
        let mut base = xpos >> 6;
        for x in 0..width {
            if !crate::av2_recon::work_tick("ipred:833") { break; }
            if base > max_base_x {
                for xx in x..width {
                    if !crate::av2_recon::work_tick("ipred:835") { break; }
                    dst[y * stride + xx] = edge;
                }
                break;
            }
            let b = (base + 2) as usize;
            let v = f[0] * filt[b - 1] + f[1] * filt[b] + f[2] * filt[b + 1] + f[3] * filt[b + 2];
            dst[y * stride + x] = ((v + 64) >> 7).clamp(0, bitdepth_max);
            base += 1;
        }
        xpos += dx;
    }
}

/// Zone-3 directional prediction (below-left, `angle > 180°`), luma base path
/// (no MRL / IBP yet) — dav2d `ipred_z3_c`. `left_in[0]` is the corner,
/// `left_in[k]` is the k-th left sample going down (`topleft_in[-k]`), needing
/// `≥ width+height+1` samples. Mirrors `ipred_z1` across the diagonal.
#[allow(clippy::too_many_arguments)]
pub fn ipred_z3(dst: &mut [i32], stride: usize, left_in: &[i32], width: usize, height: usize, angle: usize,
                enable_edge_filter: bool, have_left: bool, is_sm_l: bool, max_height: usize, bitdepth_max: i32) {
    assert!(angle > 180);
    let dy = DR_INTRA_DERIVATIVE[270 - angle] as i32;
    let max_base_y = (width + height) as i32 - 1;
    let n_px = width + height;
    let sz = 1 + width + height;
    // dav2d stores the left edge reversed in `filt`; build that exactly.
    let mut filt = [0i32; 141];
    let mut rev = [0i32; 141];
    for j in 0..sz {
        if !crate::av2_recon::work_tick("ipred:864") { break; }
        rev[j] = left_in[n_px - j];
    }
    let strength = if enable_edge_filter && have_left {
        get_filter_strength(n_px as i32, angle as i32 - 180, is_sm_l)
    } else {
        0
    };
    if strength > 0 {
        filter_edge(&mut filt[2..], sz, height.saturating_sub(max_height), sz - 1, &rev, 0, sz as i32, strength as usize);
    } else {
        filt[2..2 + sz].copy_from_slice(&rev[..sz]);
    }
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];
    let edge = filt[2]; // left[-max_base_y]
    let mut ypos = dy;
    for x in 0..width {
        if !crate::av2_recon::work_tick("ipred:882") { break; }
        let f = &DR_INTERP_FILTER[((ypos & 0x3F) >> 1) as usize];
        let mut base = ypos >> 6;
        for y in 0..height {
            if !crate::av2_recon::work_tick("ipred:885") { break; }
            if base > max_base_y {
                for yy in y..height {
                    if !crate::av2_recon::work_tick("ipred:887") { break; }
                    dst[yy * stride + x] = edge;
                }
                break;
            }
            let m = (sz as i32 - base) as usize;
            let v = f[0] * filt[m + 1] + f[1] * filt[m] + f[2] * filt[m - 1] + f[3] * filt[m - 2];
            dst[y * stride + x] = ((v + 64) >> 7).clamp(0, bitdepth_max);
            base += 1;
        }
        ypos += dy;
    }
}

/// Zone-2 directional prediction (both edges, `90° < angle < 180°`), luma base
/// path (no MRL) — dav2d `ipred_z2_c`. Each pixel projects onto the top edge
/// (`top_in[0]`=corner, `top_in[1..]`=above) or the left edge (`left_in[0]`=corner,
/// `left_in[k]`=k-th left). Reduces to the down-right diagonal at 135°.
#[allow(clippy::too_many_arguments)]
pub fn ipred_z2(dst: &mut [i32], stride: usize, top_in: &[i32], left_in: &[i32], width: usize, height: usize,
                angle: usize, enable_edge_filter: bool, have_top: bool, have_left: bool, is_sm_t: bool,
                is_sm_l: bool, max_width: usize, max_height: usize, bitdepth_max: i32) {
    assert!(angle > 90 && angle < 180);
    let dy = DR_INTRA_DERIVATIVE[angle - 90] as i32;
    let dx = DR_INTRA_DERIVATIVE[180 - angle] as i32;

    // top edge buffer: top = &filt[0]
    let mut filt = [0i32; 70];
    let sz_t = 1 + width;
    let str_t = if enable_edge_filter && have_top {
        get_filter_strength((width + height) as i32, angle as i32 - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        filter_edge(&mut filt[1..], sz_t, 1, sz_t + max_width - width, top_in, 0, sz_t as i32, str_t as usize);
    } else {
        filt[1..1 + sz_t].copy_from_slice(&top_in[..sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // left edge buffer (stored reversed): left = &filt2[height+2]
    let mut filt2 = [0i32; 70];
    let sz_l = 1 + height;
    let mut rev_l = [0i32; 70];
    for k in 0..sz_l {
        if !crate::av2_recon::work_tick("ipred:933") { break; }
        rev_l[k] = left_in[height - k];
    }
    let str_l = if enable_edge_filter && have_left {
        get_filter_strength((width + height) as i32, 180 - angle as i32, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        filter_edge(&mut filt2[1..], sz_l, height.saturating_sub(max_height), sz_l - 1, &rev_l, 0, sz_l as i32, str_l as usize);
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&rev_l[..sz_l]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    let h = height as i32;
    for y in 0..height {
        if !crate::av2_recon::work_tick("ipred:950") { break; }
        let mut xpos = -((y + 1) as i32) * dx;
        let mut x = 0;
        // left-edge projection
        while x < width && xpos < -64 {
            let ypos_l = ((y as i32) << 6) - (x as i32 + 1) * dy;
            let base_y = ypos_l >> 6;
            let f = &DR_INTERP_FILTER[((ypos_l & 0x3F) >> 1) as usize];
            let v = f[0] * filt2[(h + 1 - base_y) as usize]
                + f[1] * filt2[(h - base_y) as usize]
                + f[2] * filt2[(h - 1 - base_y) as usize]
                + f[3] * filt2[(h - 2 - base_y) as usize];
            dst[y * stride + x] = ((v + 64) >> 7).clamp(0, bitdepth_max);
            x += 1;
            xpos += 64;
        }
        // top-edge projection
        while x < width {
            let base_x = xpos >> 6;
            let f = &DR_INTERP_FILTER[((xpos & 0x3F) >> 1) as usize];
            let v = f[0] * filt[(base_x + 1) as usize]
                + f[1] * filt[(base_x + 2) as usize]
                + f[2] * filt[(base_x + 3) as usize]
                + f[3] * filt[(base_x + 4) as usize];
            dst[y * stride + x] = ((v + 64) >> 7).clamp(0, bitdepth_max);
            x += 1;
            xpos += 64;
        }
    }
}

/// DIP (decoder-side intra prediction) neighbour-summary inputs (dav2d
/// `ipred_dip_c`, the `in[]`/`in_sum` computation). Reduces the block's neighbours
/// to 11 averages — corner, 4 top quarters, 4 left quarters, top-right, bottom-left
/// — that drive the data-driven weighted prediction grid (`dav2d_dip_weights`,
/// itself a generator job like the STX kernels). `top` needs `width + width/4`
/// samples, `left` needs `height + height/4`. Returns the inputs and their sum.
pub fn dip_input_summary(corner: i32, top: &[i32], left: &[i32], width: usize, height: usize, trans: bool) -> ([i32; 11], i32) {
    let wd = width >> 2;
    let hd = height >> 2;
    let wl2 = ulog2(wd);
    let hl2 = ulog2(hd);
    let wrnd = (width >> 3) as i32;
    let hrnd = (height >> 3) as i32;
    let i_t = 1 + 4 * trans as usize;
    let i_l = 5 - 4 * trans as usize;
    let mut inp = [0i32; 11];
    inp[0] = corner;
    let mut in_sum = corner;
    // 4 top quarter-averages
    for i in 0..4 {
        let sum: i32 = top[i * wd..(i + 1) * wd].iter().sum();
        inp[i_t + i] = (sum + wrnd) >> wl2;
        in_sum += inp[i_t + i];
    }
    // 4 left quarter-averages
    for i in 0..4 {
        let sum: i32 = left[i * hd..(i + 1) * hd].iter().sum();
        inp[i_l + i] = (sum + hrnd) >> hl2;
        in_sum += inp[i_l + i];
    }
    // top-right average
    let sum: i32 = top[width..width + wd].iter().sum();
    inp[9 + trans as usize] = (sum + wrnd) >> wl2;
    in_sum += inp[9 + trans as usize];
    // bottom-left average
    let sum: i32 = left[height..height + hd].iter().sum();
    inp[10 - trans as usize] = (sum + hrnd) >> hl2;
    in_sum += inp[10 - trans as usize];
    (inp, in_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All neighbours equal to V → every predictor must output a flat V block.
    fn assert_flat(pred: impl Fn(&mut [i32], usize, &[i32], &[i32], usize, usize), w: usize, h: usize) {
        let v = 137;
        let top = vec![v; w + 1];
        let left = vec![v; h + 1];
        let mut dst = vec![0i32; w * h];
        pred(&mut dst, w, &top, &left, w, h);
        assert!(dst.iter().all(|&p| p == v), "non-flat output for {w}x{h}");
    }

    #[test]
    fn flat_neighbours_flat_output() {
        // Now covers square AND rectangular DC (rect via fast_div32_dc).
        for &(w, h) in &[(4, 4), (8, 8), (16, 16), (4, 8), (8, 4), (32, 32), (16, 8), (4, 16)] {
            assert_flat(|d, s, t, l, w, h| ipred_dc(d, s, t, l, w, h, 1023), w, h);
            assert_flat(|d, s, t, _l, w, h| ipred_v(d, s, t, w, h), w, h);
            assert_flat(|d, s, _t, l, w, h| ipred_h(d, s, l, w, h), w, h);
            assert_flat(|d, s, t, l, w, h| ipred_paeth(d, s, t, l, 137, w, h), w, h);
            assert_flat(|d, s, t, l, w, h| ipred_smooth(d, s, t, l, w, h), w, h);
            assert_flat(|d, s, t, l, w, h| ipred_smooth_v(d, s, t, l, w, h), w, h);
            assert_flat(|d, s, t, l, w, h| ipred_smooth_h(d, s, t, l, w, h), w, h);
        }
    }

    #[test]
    fn rect_dc_divides() {
        // 4x8: top all 10, left all 20 → sum=200, n_pel=12 → fast_div32_dc(200,12).
        let top = [10, 10, 10, 10, 0];
        let left = [20, 20, 20, 20, 20, 20, 20, 20, 0];
        let mut dst = [0i32; 32];
        ipred_dc(&mut dst, 4, &top, &left, 4, 8, 1023);
        assert!(dst.iter().all(|&p| p == fast_div32_dc(200, 12)));
        // exact divisions: fast_div32_dc must equal integer division on clean cases.
        assert_eq!(fast_div32_dc(1200, 12), 100);
        assert_eq!(fast_div32_dc(48, 12), 4);
    }

    #[test]
    fn edge_filter_flat_is_identity() {
        // every kernel sums to 16 → (16*V + 8) >> 4 == V on a flat edge.
        for strength in 1..=3 {
            if !crate::av2_recon::work_tick("ipred:1066") { break; }
            let inp = [50i32; 16];
            let mut out = [0i32; 16];
            filter_edge(&mut out, 16, 1, 15, &inp, 0, 16, strength);
            assert!(out.iter().all(|&p| p == 50), "strength {strength}");
        }
    }

    #[test]
    fn z1_flat_edge_is_flat() {
        // Flat top edge → flat output (the 4-tap filter sums to 128).
        let tl = [200i32; 1 + 16 + 16 + 2];
        let mut dst = [0i32; 8 * 8];
        ipred_z1(&mut dst, 8, &tl, 8, 8, 30, false, true, false, 8, 1023);
        assert!(dst.iter().all(|&p| p == 200));
    }

    #[test]
    fn z1_45deg_is_diagonal_copy() {
        // angle index 45 → dx=64 (slope 1.0, shift 0 → filter {0,128,0,0}); the
        // predictor becomes a pure diagonal copy: dst[y][x] = topleft_in[y+x+2].
        assert_eq!(DR_INTRA_DERIVATIVE[45], 64);
        let tl: Vec<i32> = (0..13).collect(); // corner + top edge = 0,1,2,...
        let mut dst = [0i32; 16];
        ipred_z1(&mut dst, 4, &tl, 4, 4, 45, false, true, false, 4, 1023);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], (y + x + 2) as i32, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn z3_flat_edge_is_flat() {
        let li = [77i32; 1 + 16 + 16 + 2];
        let mut dst = [0i32; 8 * 8];
        ipred_z3(&mut dst, 8, &li, 8, 8, 225, false, true, false, 8, 1023);
        assert!(dst.iter().all(|&p| p == 77));
    }

    #[test]
    fn z3_225deg_is_diagonal_copy() {
        // angle 225 → dy=64 (slope 1.0, filter {0,128,0,0}); left-edge analogue of
        // z1's 45°: dst[y][x] = left_in[x+y+2].
        assert_eq!(DR_INTRA_DERIVATIVE[270 - 225], 64);
        let li: Vec<i32> = (0..13).collect();
        let mut dst = [0i32; 16];
        ipred_z3(&mut dst, 4, &li, 4, 4, 225, false, true, false, 4, 1023);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], (x + y + 2) as i32, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn z2_flat_edges_is_flat() {
        let ti = [88i32; 1 + 16 + 2];
        let li = [88i32; 1 + 16 + 2];
        let mut dst = [0i32; 8 * 8];
        ipred_z2(&mut dst, 8, &ti, &li, 8, 8, 135, false, true, true, false, false, 8, 8, 1023);
        assert!(dst.iter().all(|&p| p == 88));
    }

    #[test]
    fn z2_135deg_is_down_right_diagonal() {
        // dx=dy=64; each pixel is corner on the diagonal, top edge above it, left
        // edge below it: dst[y][x] = x>y ? top_in[x-y] : x<y ? left_in[y-x] : corner.
        assert_eq!(DR_INTRA_DERIVATIVE[135 - 90], 64);
        assert_eq!(DR_INTRA_DERIVATIVE[180 - 135], 64);
        let top_in = [100, 101, 102, 103, 104, 105, 106, 107, 108];
        let left_in = [100, 200, 201, 202, 203, 204, 205, 206, 207];
        let mut dst = [0i32; 16];
        ipred_z2(&mut dst, 4, &top_in, &left_in, 4, 4, 135, false, true, true, false, false, 4, 4, 1023);
        let expect = [
            100, 101, 102, 103, //
            200, 100, 101, 102, //
            201, 200, 100, 101, //
            202, 201, 200, 100,
        ];
        assert_eq!(dst, expect);
    }

    #[test]
    fn dip_summary_flat_neighbours() {
        // flat V neighbours → every summary input is V, in_sum = 11V.
        let v = 90;
        let top = vec![v; 16 + 4];
        let left = vec![v; 16 + 4];
        let (inp, in_sum) = dip_input_summary(v, &top, &left, 16, 16, false);
        assert_eq!(inp, [v; 11]);
        assert_eq!(in_sum, 11 * v);
    }

    #[test]
    fn dip_prediction_composes_to_flat() {
        // summary + generated weights + the prediction formula compose: flat input → V.
        use crate::av2_tables_gen::DIP_WEIGHTS;
        // first weight row, hand-verified against dav2d dip_tables.c
        assert_eq!(DIP_WEIGHTS[0][0], [3104, 6856, 4308, 3992, 4172, 5748, 4628, 4108, 4108, 4044, 4092]);
        let v = 100;
        let top = vec![v; 16 + 4];
        let left = vec![v; 16 + 4];
        let (inp, in_sum) = dip_input_summary(v, &top, &left, 16, 16, false);
        let (m, idx) = (0, 0);
        let sum: i32 = (0..11).map(|i| DIP_WEIGHTS[m][idx][i] as i32 * inp[i]).sum();
        let pred = (((sum + 2048) >> 12) - in_sum).clamp(0, 255);
        assert_eq!(pred, v); // weights sum ≈ 12·4096 → flat reconstructs V
    }

    #[test]
    fn dip_summary_quarter_averages() {
        // top split into 4 quarters of distinct values → 4 quarter averages.
        let top = [
            10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, // 16-wide top
            50, 50, 50, 50, // top-right extension (wd=4)
        ];
        let left = vec![7; 16 + 4];
        let (inp, _) = dip_input_summary(0, &top, &left, 16, 16, false);
        // i_t = 1 (trans=false): inp[1..5] = the 4 top quarter averages
        assert_eq!(&inp[1..5], &[10, 20, 30, 40]);
        // inp[9] = top-right average
        assert_eq!(inp[9], 50);
        // inp[5..9] = left quarters (all 7)
        assert_eq!(&inp[5..9], &[7, 7, 7, 7]);
    }

    #[test]
    fn filter_strength_lookup() {
        assert_eq!(get_filter_strength(8, 64, true), 2);
        assert_eq!(get_filter_strength(8, 40, true), 1);
        assert_eq!(get_filter_strength(8, 10, true), 0);
        assert_eq!(get_filter_strength(40, 0, false), 3);
        assert_eq!(get_filter_strength(32, 32, false), 3);
        assert_eq!(get_filter_strength(32, 4, false), 2);
        assert_eq!(get_filter_strength(32, 0, false), 1);
    }

    #[test]
    fn v_copies_top_row() {
        let top = [10, 20, 30, 40, 99];
        let mut dst = [0i32; 4 * 3];
        ipred_v(&mut dst, 4, &top, 4, 3);
        for y in 0..3 {
            assert_eq!(&dst[y * 4..y * 4 + 4], &[10, 20, 30, 40]);
        }
    }

    #[test]
    fn h_splats_left_column() {
        let left = [10, 20, 30, 99];
        let mut dst = [0i32; 4 * 3];
        ipred_h(&mut dst, 4, &left, 4, 3);
        assert_eq!(&dst[0..4], &[10, 10, 10, 10]);
        assert_eq!(&dst[4..8], &[20, 20, 20, 20]);
        assert_eq!(&dst[8..12], &[30, 30, 30, 30]);
    }

    #[test]
    fn dc_averages_neighbours() {
        // 4x4: top all 10, left all 20 → (40+80+4)>>3 = 15.
        let top = [10, 10, 10, 10, 0];
        let left = [20, 20, 20, 20, 0];
        let mut dst = [0i32; 16];
        ipred_dc(&mut dst, 4, &top, &left, 4, 4, 1023);
        assert!(dst.iter().all(|&p| p == 15));
    }

    #[test]
    fn paeth_picks_nearest() {
        // corner=10, top=20, left=5 → base=15; tdiff=5=tldiff, ldiff=10 → picks top.
        let top = [20, 0];
        let left = [5, 0];
        let mut dst = [0i32; 1];
        ipred_paeth(&mut dst, 1, &top, &left, 10, 1, 1);
        assert_eq!(dst[0], 20);
    }
}
