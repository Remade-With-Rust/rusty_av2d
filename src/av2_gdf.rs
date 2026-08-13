//! AV2 GDF (Guided Deblocking Filter) — avm `gdf.c` / `gdf_block.c`. A learned,
//! table-driven directional filter with no AV1/dav2d analog. Luma-only, applied last in
//! Stage C (after CCSO). For each 64×64 unit it (1) computes 4 directional Laplacian
//! activity features + a 4-way class per 2×2 block, (2) runs a small per-pixel inference
//! over reconstruction-difference + gradient features → an expected-coding-error index
//! into a LUT, (3) adds `scale·err` to the reconstruction. Reads a pristine guided copy
//! of the post-CCSO luma (with post-deblock stripe boundaries); writes a separate output.
//!
//! Frame 1 is intra, so only the intra tables are used (`gdf_tables`).

use crate::gdf_tables::{GDF_INTRA_ALPHA, GDF_INTRA_BIAS, GDF_INTRA_ERROR, GDF_INTRA_WEIGHT};

const GDF_TEST_INP_PREC: i32 = 10;
const HOR_BORDER: usize = 6; // GDF_TEST_EXTRA_HOR_BORDER
const VER_BORDER: usize = 6; // GDF_TEST_EXTRA_VER_BORDER
const STRIPE_OFF: i32 = 8; // GDF_TEST_STRIPE_OFF
const STRIPE_SIZE: i32 = 64; // GDF_TEST_STRIPE_SIZE = unit size
const BLK_SIZE_DEFAULT: i32 = 128; // GDF_TEST_BLK_SIZE
const FRAME_BOUNDARY: i32 = 0; // GDF_TEST_FRAME_BOUNDARY_SIZE
const ERR_MARGIN: usize = 16; // GDF_ERR_STRIDE_MARGIN

const INP_REC_NUM: usize = 18; // GDF_NET_INP_REC_NUM
const INP_GRD_NUM: usize = 4; // GDF_NET_INP_GRD_NUM
const INP_TOT: usize = INP_REC_NUM + INP_GRD_NUM; // 22
const CLS_NUM: usize = 4; // GDF_TRAIN_CLS_NUM
const GRD_SHIFT: i32 = 4; // GDF_TRAIN_GRD_SHIFT
const PAR_SCALE_LOG2: i32 = 5; // GDF_TRAIN_PAR_SCALE_LOG2
const TRAIN_INP_PREC: i32 = 0; // GDF_TRAIN_INP_PREC
const LUT_IDX_NUM: usize = 3; // GDF_NET_LUT_IDX_NUM
const LUT_IDX_INTRA_MAX: i32 = 16; // GDF_NET_LUT_IDX_INTRA_MAX
const LUT_IDX_INTER_MAX: i32 = 10; // GDF_NET_LUT_IDX_INTER_MAX

// direction indices
const VER: usize = 0;
const HOR: usize = 1;
const DIAG0: usize = 2;
const DIAG1: usize = 3;

/// Guided reconstruction sample offsets [dy, dx] (avm gdf_guided_sample_coordinates_fwd/bwd).
const FWD: [[i32; 2]; INP_REC_NUM] = [
    [-6, 0], [-5, 0], [-4, 0], [-3, 0], [-2, -1], [-2, 0], [-2, 1], [-1, -2], [-1, -1], [-1, 0],
    [-1, 1], [-1, 2], [0, -6], [0, -5], [0, -4], [0, -3], [0, -2], [0, -1],
];
const BWD: [[i32; 2]; INP_REC_NUM] = [
    [6, 0], [5, 0], [4, 0], [3, 0], [2, 1], [2, 0], [2, -1], [1, 2], [1, 1], [1, 0], [1, -1],
    [1, -2], [0, 6], [0, 5], [0, 4], [0, 3], [0, 2], [0, 1],
];

#[inline]
fn clip(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdf_tables_and_derivation() {
        // Table shapes + spot values (auto-extracted from avm gdf_block.c).
        assert_eq!(GDF_INTRA_ALPHA.len(), 6);
        assert_eq!(GDF_INTRA_ALPHA[0].len(), 88);
        assert_eq!(GDF_INTRA_ALPHA[0][0], 511);
        assert_eq!(GDF_INTRA_WEIGHT[0].len(), INP_TOT * CLS_NUM * LUT_IDX_NUM); // 264
        assert_eq!(GDF_INTRA_BIAS[0].len(), CLS_NUM * LUT_IDX_NUM); // 12
        assert_eq!(GDF_INTRA_ERROR[0].len(), 16 * 16 * 16); // 4096
        // FWD/BWD are point-symmetric (bwd = -fwd).
        for k in 0..INP_REC_NUM {
            if !crate::av2_recon::work_tick("gdf:68") { break; }
            assert_eq!([FWD[k][0] + BWD[k][0], FWD[k][1] + BWD[k][1]], [0, 0]);
        }
        // Dev clip: intra, base_qindex=120 → qp_idx_avg=1 → qp_idx_base=0. err_shift=4, pxl_shift=2.
        let (qp_base, qp) = (85, 120);
        let avg = if qp < qp_base + 12 { 0 } else if qp < qp_base + 37 { 1 } else { 2 };
        assert_eq!(avg, 1);
        assert_eq!(clip(avg - 2, 0, 2), 0);
    }
}

/// avm `gdf_set_lap_and_cls_unit_c`: fill the per-2×2 directional Laplacian activity (`lap`,
/// 4 directions, u32 packed low-16) and 4-way class (`cls`) for the unit. `rec` is the guided
/// buffer, `base` = index of `inp[(i_min-1)][(j_min-1)]`, `stride` = inp_stride.
#[allow(clippy::too_many_arguments)]
fn set_lap_and_cls(
    blk_height: usize,
    blk_width: usize,
    rec: &[i32],
    base: usize,
    stride: usize,
    bit_depth: i32,
    lap: &mut [Vec<u32>; INP_GRD_NUM],
    lap_y_stride: usize,
    cls: &mut [u32],
    cls_stride: usize,
) {
    let s = stride as isize;
    let off_ver = s;
    let off_dia0 = s + 1;
    let off_dia1 = s - 1;
    let clip_mask: u32 = ((1i64 << (16 - (GDF_TEST_INP_PREC - bit_depth.min(GDF_TEST_INP_PREC)))) - 1) as u32;
    let lap_cls_height = blk_height >> 1;
    let rd = |o: isize| rec[(base as isize + o) as usize];

    // gradient of the 2×2 pair at column j (reads rows y00/y10 = std..std+ver, cols j and j+1)
    // for all four directions, from a std_pos base offset `sp` (relative to `base`).
    let grad = |sp: isize, j: i32, dir: usize| -> u32 {
        let j = j as isize;
        let (y00, y10, y_10, y20) = (sp, sp + off_ver, sp - off_ver, sp + 2 * off_ver);
        let (y0_1, y01, y1_1, y11) = (sp - 1, sp + 1, sp + off_ver - 1, sp + off_ver + 1);
        let (y_1_1, y21) = (sp - off_dia0, sp + off_ver + off_dia0);
        let (y_11, y2_1) = (sp - off_dia1, sp + off_ver + off_dia1);
        let a = |o: isize, k: isize| rd(o + k);
        match dir {
            VER => (((a(y00, j) << 1) - a(y_10, j) - a(y10, j)).abs()
                + ((a(y10, j) << 1) - a(y00, j) - a(y20, j)).abs()
                + ((a(y00, j + 1) << 1) - a(y_10, j + 1) - a(y10, j + 1)).abs()
                + ((a(y10, j + 1) << 1) - a(y00, j + 1) - a(y20, j + 1)).abs()) as u32,
            HOR => (((a(y00, j) << 1) - a(y0_1, j) - a(y01, j)).abs()
                + ((a(y10, j) << 1) - a(y1_1, j) - a(y11, j)).abs()
                + ((a(y00, j + 1) << 1) - a(y0_1, j + 1) - a(y01, j + 1)).abs()
                + ((a(y10, j + 1) << 1) - a(y1_1, j + 1) - a(y11, j + 1)).abs()) as u32,
            DIAG0 => (((a(y00, j) << 1) - a(y_1_1, j) - a(y11, j)).abs()
                + ((a(y10, j) << 1) - a(y0_1, j) - a(y21, j)).abs()
                + ((a(y00, j + 1) << 1) - a(y_1_1, j + 1) - a(y11, j + 1)).abs()
                + ((a(y10, j + 1) << 1) - a(y0_1, j + 1) - a(y21, j + 1)).abs()) as u32,
            _ => (((a(y00, j) << 1) - a(y_11, j) - a(y1_1, j)).abs()
                + ((a(y10, j) << 1) - a(y01, j) - a(y2_1, j)).abs()
                + ((a(y00, j + 1) << 1) - a(y_11, j + 1) - a(y1_1, j + 1)).abs()
                + ((a(y10, j + 1) << 1) - a(y01, j + 1) - a(y2_1, j + 1)).abs()) as u32,
        }
    };

    // std_pos starts at inp[(i_max-1)][(j_min-1)] = base + blk_height*stride.
    // `above`/`cur` = row indices into lap (u32 stride = lap_y_stride).
    let mut sp = (blk_height as isize) * s; // relative to base
    // above_line row = lap_cls_height-1, cur_line row = lap_cls_height (may be scratch)
    let mut above_row: isize = (lap_cls_height as isize) - 1;
    let mut cur_row: isize = lap_cls_height as isize;
    let ncols = blk_width / 2 + 2; // 2×2 cols + slack
    // seed above_line from the bottom 2×2 row (std_pos as-is, sp base)
    let seed = |lap: &mut [Vec<u32>; INP_GRD_NUM], row: isize, sp: isize| {
        for d in 0..INP_GRD_NUM {
            if !crate::av2_recon::work_tick("gdf:141") { break; }
            let r = (row as usize) * lap_y_stride;
            lap[d][r] = grad(sp, 0, d);
        }
        let mut j0 = 2i32;
        while j0 <= blk_width as i32 {
            let j00 = ((j0 - 2) >> 1) as usize;
            let j01 = j00 + 1;
            let r = (row as usize) * lap_y_stride;
            for d in 0..INP_GRD_NUM {
                let g = grad(sp, j0, d);
                lap[d][r + j01] = g;
                lap[d][r + j00] = lap[d][r + j00].wrapping_add(g);
            }
            j0 += 2;
        }
    };
    let _ = ncols;
    seed(lap, above_row, sp);

    let mut cls_row = lap_cls_height as isize; // gdf_cls advanced then decremented
    for i in (0..lap_cls_height as isize).rev() {
        if !crate::av2_recon::work_tick("gdf:162") { break; }
        sp -= 2 * s;
        cls_row -= 1;
        above_row -= 1;
        cur_row -= 1;
        let cr = (cur_row as usize) * lap_y_stride;
        let clr = (cls_row.max(0) as usize) * cls_stride;
        if i == 0 {
            for d in 0..INP_GRD_NUM {
                lap[d][cr] = lap[d][cr].wrapping_add(grad(sp, 0, d));
            }
            let mut j0 = 2i32;
            while j0 <= blk_width as i32 {
                let j00 = ((j0 - 2) >> 1) as usize;
                let j01 = j00 + 1;
                let g: [u32; 4] = [grad(sp, j0, VER), grad(sp, j0, HOR), grad(sp, j0, DIAG0), grad(sp, j0, DIAG1)];
                for d in 0..INP_GRD_NUM {
                    lap[d][cr + j00] = lap[d][cr + j00].wrapping_add(g[d]);
                    lap[d][cr + j01] = lap[d][cr + j01].wrapping_add(g[d]);
                    lap[d][cr + j00] &= clip_mask;
                }
                cls[clr + j00] = (if lap[VER][cr + j00] > lap[HOR][cr + j00] { 0 } else { 1 })
                    | (if lap[DIAG0][cr + j00] > lap[DIAG1][cr + j00] { 0 } else { 2 });
                for d in 0..INP_GRD_NUM {
                    if !crate::av2_recon::work_tick("gdf:185") { break; }
                    lap[d][cr + j00] |= lap[d][cr + j00] << 16;
                }
                j0 += 2;
            }
        } else {
            let ar = (above_row as usize) * lap_y_stride;
            for d in 0..INP_GRD_NUM {
                if !crate::av2_recon::work_tick("gdf:192") { break; }
                lap[d][ar] = grad(sp, 0, d);
            }
            let mut j0 = 2i32;
            while j0 <= blk_width as i32 {
                let j00 = ((j0 - 2) >> 1) as usize;
                let j01 = j00 + 1;
                for d in 0..INP_GRD_NUM {
                    let g = grad(sp, j0, d);
                    lap[d][ar + j01] = g;
                    lap[d][ar + j00] = lap[d][ar + j00].wrapping_add(g);
                    lap[d][cr + j00] = lap[d][cr + j00].wrapping_add(lap[d][ar + j00]);
                    lap[d][cr + j00] &= clip_mask;
                }
                cls[clr + j00] = (if lap[VER][cr + j00] > lap[HOR][cr + j00] { 0 } else { 1 })
                    | (if lap[DIAG0][cr + j00] > lap[DIAG1][cr + j00] { 0 } else { 2 });
                for d in 0..INP_GRD_NUM {
                    if !crate::av2_recon::work_tick("gdf:208") { break; }
                    lap[d][cr + j00] |= lap[d][cr + j00] << 16;
                }
                j0 += 2;
            }
        }
    }
}

/// avm `gdf_inference_unit_c` (intra path): per-pixel feature extraction + weighted LUT
/// index → expected-coding-error, written to `err`. `rec` guided buffer, `recbase` = index
/// of `inp[i_min][j_min]`.
#[allow(clippy::too_many_arguments)]
fn inference(
    blk_height: usize,
    blk_width: usize,
    qp_idx: usize,
    rec: &[i32],
    recbase: usize,
    stride: usize,
    lap: &[Vec<u32>; INP_GRD_NUM],
    lap_y_stride: usize,
    lap_cls_height: usize,
    cls: &[u32],
    cls_stride: usize,
    err: &mut [i32],
    err_stride: usize,
    pxl_shift: i32,
    // 0 = intra (INTRA tables, LUT_IDX_INTRA_MAX); 1..=5 = inter, indexing GDF_INTER_*[ref_dst_idx-1].
    ref_dst_idx: usize,
) {
    let is_intra = ref_dst_idx == 0;
    let gdf_frm_max = if is_intra { LUT_IDX_INTRA_MAX } else { LUT_IDX_INTER_MAX };
    let gdf_idx_min = -(gdf_frm_max >> 1);
    let gdf_idx_max = gdf_frm_max - 1 + gdf_idx_min;
    let gdf_idx_scale = (-gdf_idx_min).max(gdf_idx_max);
    let gdf_shift = GDF_TEST_INP_PREC - TRAIN_INP_PREC + PAR_SCALE_LOG2; // 10-0+5=15
    let gdf_shift_half = 1i32 << (gdf_shift - 1);
    let norm = |va: i32| -> i32 {
        let t = if va > 0 {
            (gdf_idx_scale * va + gdf_shift_half) >> gdf_shift
        } else {
            -((gdf_idx_scale * (-va) + gdf_shift_half) >> gdf_shift)
        };
        t - gdf_idx_min
    };
    // Same inference algorithm for intra/inter; only the learned tables + LUT_IDX_MAX differ.
    let (alpha, weight, bias, gdftable): (&[i16], &[i16], &[i32], &[i8]) = if is_intra {
        (&GDF_INTRA_ALPHA[qp_idx], &GDF_INTRA_WEIGHT[qp_idx], &GDF_INTRA_BIAS[qp_idx], &GDF_INTRA_ERROR[qp_idx])
    } else {
        let r = ref_dst_idx - 1;
        (&crate::gdf_inter_tables::GDF_INTER_ALPHA[r][qp_idx],
         &crate::gdf_inter_tables::GDF_INTER_WEIGHT[r][qp_idx],
         &crate::gdf_inter_tables::GDF_INTER_BIAS[r][qp_idx],
         &crate::gdf_inter_tables::GDF_INTER_ERROR[r][qp_idx])
    };
    let mut idx_offset = [1i32; LUT_IDX_NUM];
    for idx in 0..LUT_IDX_NUM {
        if !crate::av2_recon::work_tick("gdf:265") { break; }
        for _ in 0..(LUT_IDX_NUM - 1 - idx) {
            idx_offset[idx] *= gdf_frm_max;
        }
    }
    let s = stride as isize;
    // per-2-rows lap/cls row pointer
    let mut lap_row = 0usize; // in u32 units, advances by lap_y_stride per 2 output rows
    let mut cls_col_row = 0usize; // cls advances by cls_stride per 2 output rows
    let _ = lap_cls_height;
    for i in 0..blk_height {
        if !crate::av2_recon::work_tick("gdf:275") { break; }
        let rec_ptr = recbase as isize + (i as isize) * s;
        // gdf_idx[j][idx]
        let mut gdf_idx = vec![[0i32; LUT_IDX_NUM]; blk_width];
        for k in 0..INP_TOT {
            let (fy, fx, by, bx) = if k < INP_REC_NUM {
                (FWD[k][0], FWD[k][1], BWD[k][0], BWD[k][1])
            } else {
                (0, 0, 0, 0)
            };
            for j in 0..blk_width {
                let cls_idx = cls[cls_col_row + (j >> 1)] as usize;
                let cls_offset = k * CLS_NUM + cls_idx;
                let center = rec[(rec_ptr + j as isize) as usize];
                let inp_value = if k < INP_REC_NUM {
                    let fwd = rec[(rec_ptr + (fy as isize * s + fx as isize) + j as isize) as usize];
                    (fwd - center) * (1 << pxl_shift)
                } else {
                    let grd = (lap[k - INP_REC_NUM][lap_row + (j >> 1)] & 0xffff) as i32;
                    (grd << pxl_shift) >> GRD_SHIFT
                };
                let a = alpha[cls_offset] as i32;
                let mut gdf_inp = clip(inp_value, -a, a);
                if k < INP_REC_NUM {
                    let bwd = rec[(rec_ptr + (by as isize * s + bx as isize) + j as isize) as usize];
                    let inp2 = (bwd - center) * (1 << pxl_shift);
                    gdf_inp += clip(inp2, -a, a);
                }
                gdf_inp = clip(gdf_inp, -(1 << (GDF_TEST_INP_PREC - 1)), (1 << (GDF_TEST_INP_PREC - 1)) - 1);
                for idx in 0..LUT_IDX_NUM {
                    if !crate::av2_recon::work_tick("gdf:304") { break; }
                    let w = weight[cls_offset + (INP_TOT * CLS_NUM) * idx] as i32;
                    gdf_idx[j][idx] += gdf_inp * w;
                }
                if k == INP_TOT - 1 {
                    let mut tb = 0i32; // offset into gdftable
                    for idx in 0..LUT_IDX_NUM {
                        gdf_idx[j][idx] += bias[cls_idx + CLS_NUM * idx];
                        let ti = norm(gdf_idx[j][idx]);
                        tb += clip(ti, 0, gdf_frm_max - 1) * idx_offset[idx];
                    }
                    err[i * err_stride + j] = gdftable[tb as usize] as i32;
                }
            }
        }
        if i & 1 == 1 {
            cls_col_row += cls_stride;
            lap_row += lap_y_stride;
        }
    }
}

/// avm `gdf_compensation_unit_c`: `out = clip(out + round(scale·err >> err_shift))`.
#[allow(clippy::too_many_arguments)]
fn compensation(
    out: &mut [i32],
    obase: usize,
    ostride: usize,
    err: &[i32],
    err_stride: usize,
    err_shift: i32,
    scale: i32,
    pxl_max: i32,
    blk_height: usize,
    blk_width: usize,
) {
    let half = if err_shift > 0 { 1 << (err_shift - 1) } else { 0 };
    let gdbg: Option<(usize, usize)> = std::env::var("GDBG").ok().and_then(|v| {
        let mut it = v.split(',');
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    });
    for i in 0..blk_height {
        if !crate::av2_recon::work_tick("gdf:345") { break; }
        for j in 0..blk_width {
            let mut res = (scale * err[i * err_stride + j]) as i16 as i32; // C truncates to int16_t
            res = if res > 0 { (res + half) >> err_shift } else { -(((-res) + half) >> err_shift) };
            let o = obase + i * ostride + j;
            if let Some((tx, ty)) = gdbg {
                if o == ty * ostride + tx {
                    crate::dlog!("[MGDBG] px=({tx},{ty}) err={} res={res} pre={} scale={scale}", err[i * err_stride + j], out[o]);
                }
            }
            out[o] = clip(res + out[o], 0, pxl_max);
        }
    }
}

/// Build the padded guided frame from `luma` (post-CCSO, `w`×`h`) with replicated 6px borders.
/// Returns (inp buffer, inp_stride, inp_origin index of pixel (0,0)). For 8-bit no shift.
fn build_guided(luma: &[i32], w: usize, h: usize) -> (Vec<i32>, usize, usize) {
    let inp_stride = (((w + STRIPE_SIZE as usize) >> 4) << 4) + 16;
    let rows = VER_BORDER + h + VER_BORDER + 4;
    let mut inp = vec![0i32; rows * inp_stride];
    let origin = VER_BORDER * inp_stride + HOR_BORDER;
    for y in 0..h {
        if !crate::av2_recon::work_tick("gdf:367") { break; }
        for x in 0..w {
            inp[origin + y * inp_stride + x] = luma[y * w + x];
        }
    }
    // extend: left/right replicate, then top/bottom replicate full rows (gdf_extend_frame_highbd)
    for y in 0..h {
        if !crate::av2_recon::work_tick("gdf:373") { break; }
        let r = origin + y * inp_stride;
        let left = inp[r];
        let right = inp[r + w - 1];
        for b in 1..=HOR_BORDER {
            inp[r - b] = left;
            inp[r + w - 1 + b] = right;
        }
    }
    for b in 1..=VER_BORDER {
        if !crate::av2_recon::work_tick("gdf:382") { break; }
        // top: copy row 0 (incl borders) up
        let src = origin - HOR_BORDER;
        let dstt = origin - b * inp_stride - HOR_BORDER;
        let dstb = origin + (h - 1 + b) * inp_stride - HOR_BORDER;
        let srcb = origin + (h - 1) * inp_stride - HOR_BORDER;
        for x in 0..(w + 2 * HOR_BORDER) {
            inp[dstt + x] = inp[src + x];
            inp[dstb + x] = inp[srcb + x];
        }
    }
    (inp, inp_stride, origin)
}

/// Apply GDF to the whole (single-tile) frame. `guided` is the post-CCSO luma; `out` is the
/// mutable output (starts = post-CCSO). `block_on(blk_idx)` gives the per-block flag; `gdf_mode`
/// 1 = all-on. Returns nothing (writes `out`).
#[allow(clippy::too_many_arguments)]
pub fn gdf_filter_frame(
    guided: &[i32],
    dblk: &[i32],
    out: &mut [i32],
    w: usize,
    h: usize,
    bit_depth: i32,
    base_qindex: i32,
    gdf_mode: i32,
    gdf_pic_qp_idx: i32,
    gdf_pic_scale_idx: i32,
    gdf_block_size: i32,
    block_flags: &[bool],
    // 0 = intra frame; 1..=5 = inter (avm gdf_get_ref_dst_idx: frame-2 single-ref dist-1 → 1).
    ref_dst_idx: usize,
) {
    crate::prof_scope!(9);
    let pxl_max = (1 << bit_depth) - 1;
    let pxl_shift = GDF_TEST_INP_PREC - bit_depth.min(GDF_TEST_INP_PREC);
    let err_shift = 2 /*GDF_RDO_SCALE_NUM_LOG2*/ + GDF_TEST_INP_PREC - bit_depth;
    // gdf_get_qp_idx_base: qp_base = intra ? 85 : 110 (the ONLY qp derivation difference).
    let qp_base = if ref_dst_idx == 0 { 85 } else { 110 };
    let qp_offset = 24 * (bit_depth - 8);
    let qp = base_qindex;
    let qp_idx_avg = if qp < qp_base + 12 + qp_offset { 0 }
        else if qp < qp_base + 37 + qp_offset { 1 }
        else if qp < qp_base + 62 + qp_offset { 2 }
        else if qp < qp_base + 87 + qp_offset { 3 }
        else if qp < qp_base + 112 + qp_offset { 4 } else { 5 };
    let qp_idx_base = clip(qp_idx_avg - 2, 0, 6 - 4);
    let qp_idx = (qp_idx_base + gdf_pic_qp_idx) as usize;
    let scale_val = gdf_pic_scale_idx + 1;

    let (mut inp, inp_stride, origin) = build_guided(guided, w, h);
    let unit_size = STRIPE_SIZE;
    let block_size = gdf_block_size;
    let block_num_w = 1 + (w as i32 - 1) / block_size;

    // Inject a post-deblock reference row `drow` into guided-copy row `dst_row` (with the 6px
    // horizontal replicate), saving the old contents (incl. borders) into `save`.
    let inject_row = |inp: &mut [i32], dst_row: i32, drow: i32, save: &mut Vec<i32>| {
        let base = (origin as isize + dst_row as isize * inp_stride as isize) as usize - HOR_BORDER;
        for x in 0..(w + 2 * HOR_BORDER) {
            if !crate::av2_recon::work_tick("gdf:441") { break; }
            save.push(inp[base + x]);
        }
        let dr = drow.clamp(0, h as i32 - 1) as usize;
        for x in 0..w {
            if !crate::av2_recon::work_tick("gdf:445") { break; }
            inp[base + HOR_BORDER + x] = dblk[dr * w + x];
        }
        let left = inp[base + HOR_BORDER];
        let right = inp[base + HOR_BORDER + w - 1];
        for b in 1..=HOR_BORDER {
            if !crate::av2_recon::work_tick("gdf:450") { break; }
            inp[base + HOR_BORDER - b] = left;
            inp[base + HOR_BORDER + w - 1 + b] = right;
        }
    };
    let restore_rows = |inp: &mut [i32], rows: &[i32], save: &[i32]| {
        let mut k = 0usize;
        for &dst_row in rows {
            let base = (origin as isize + dst_row as isize * inp_stride as isize) as usize - HOR_BORDER;
            for x in 0..(w + 2 * HOR_BORDER) {
                inp[base + x] = save[k];
                k += 1;
            }
        }
    };

    let lap_y_stride = ((unit_size as usize + ERR_MARGIN) + 1) >> 1; // u32 units (gdf_lap_stride>>1)
    let cls_stride = (unit_size as usize >> 1) + ERR_MARGIN;
    let err_stride = unit_size as usize + ERR_MARGIN;
    let lap_rows = (unit_size as usize >> 1) + 2;
    let mut lap: [Vec<u32>; INP_GRD_NUM] = [
        vec![0u32; lap_y_stride * lap_rows],
        vec![0u32; lap_y_stride * lap_rows],
        vec![0u32; lap_y_stride * lap_rows],
        vec![0u32; lap_y_stride * lap_rows],
    ];
    let mut cls = vec![0u32; cls_stride * lap_rows];
    let mut err = vec![0i32; err_stride * unit_size as usize];

    let tile_h = h as i32;
    let tile_w = w as i32;
    // Stripe-major iteration (v_pos step 64 from -STRIPE_OFF). Per stripe, inject the post-deblock
    // reference lines into the guided-copy borders (avm gdf_setup_reference_lines, copy_above/below
    // = 1 for a single tile), process all units, then restore.
    let mut v_pos = -STRIPE_OFF;
    while v_pos < tile_h {
        let i_min = ((v_pos.max(FRAME_BOUNDARY)) + 1) & !1;
        let i_max = ((v_pos + unit_size).min(tile_h - FRAME_BOUNDARY)) & !1;
        if i_max <= i_min {
            v_pos += unit_size;
            continue;
        }
        // reference lines: above rows i_min-6..i_min-1, below rows i_max..i_max+5. Only at
        // INTERNAL stripe boundaries — the frame top/bottom keep the post-CCSO extend (the saved
        // boundary at the picture edge is the guided copy's own extension, not a cross-stripe line).
        let mut save: Vec<i32> = Vec::new();
        let mut rows: Vec<i32> = Vec::new();
        if i_min > 0 {
            for i in (-(VER_BORDER as i32))..0 {
                if !crate::av2_recon::work_tick("gdf:498") { break; }
                let dst = i_min + i;
                let drow = if i == -1 { i_min - 1 } else { i_min - 2 };
                inject_row(&mut inp, dst, drow, &mut save);
                rows.push(dst);
            }
        }
        if i_max < tile_h {
            for i in 0..(VER_BORDER as i32) {
                if !crate::av2_recon::work_tick("gdf:506") { break; }
                let dst = i_max + i;
                let drow = if i == 0 { i_max } else { i_max + 1 };
                inject_row(&mut inp, dst, drow, &mut save);
                rows.push(dst);
            }
        }
        let br = ((v_pos + STRIPE_OFF) / block_size) as usize;
        let mut u_pos = 0i32;
        while u_pos < tile_w {
            let j_min = ((u_pos.max(FRAME_BOUNDARY)) + 1) & !1;
            let j_max = ((u_pos + unit_size).min(tile_w - FRAME_BOUNDARY)) & !1;
            let bc = (u_pos / block_size) as usize;
            let blk_idx = br * block_num_w as usize + bc;
            if j_max > j_min && (gdf_mode == 1 || block_flags.get(blk_idx).copied().unwrap_or(false)) {
                let (i_min, i_max, j_min, j_max) = (i_min as usize, i_max as usize, j_min as usize, j_max as usize);
                let bh = i_max - i_min;
                let bw = j_max - j_min;
                let lap_cls_height = bh >> 1;
                for v in lap.iter_mut() {
                    if !crate::av2_recon::work_tick("gdf:525") { break; }
                    v.iter_mut().for_each(|x| *x = 0);
                }
                cls.iter_mut().for_each(|x| *x = 0);
                let base = origin + (i_min - 1) * inp_stride + (j_min - 1);
                set_lap_and_cls(bh, bw, &inp, base, inp_stride, bit_depth, &mut lap, lap_y_stride, &mut cls, cls_stride);
                let recbase = origin + i_min * inp_stride + j_min;
                inference(bh, bw, qp_idx, &inp, recbase, inp_stride, &lap, lap_y_stride, lap_cls_height, &cls, cls_stride, &mut err, err_stride, pxl_shift, ref_dst_idx);
                let obase = i_min * w + j_min;
                compensation(out, obase, w, &err, err_stride, err_shift, scale_val, pxl_max, bh, bw);
            }
            u_pos += unit_size;
        }
        restore_rows(&mut inp, &rows, &save);
        v_pos += unit_size;
    }
}
