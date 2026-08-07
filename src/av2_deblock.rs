//! AV2 deblocking filter — dav2d `deblock_tmpl.c`. AV2 replaced AV1's fixed
//! 4/8/14-tap boundary filters with a derivative-driven design: `filter_choice`
//! picks a filter width from the local second-derivatives, and `deblock` applies a
//! width-tapered delta across the edge. Operates on one boundary (4 lines) at a time.
//!
//! `dst` indexing is offset-based: `stridea` steps along the edge (the 4 lines),
//! `strideb` steps across it (the filter direction); both may be negative.

const MAX_WIDTH_Y: [i32; 4] = [1, 3, 6, 8];
#[allow(dead_code)]
const MAX_WIDTH_UV: [i32; 3] = [1, 3, 4];
const Q_FIRST: [i32; 5] = [45, 40, 32, 0, 0];
const Q_THRESH_MULTS: [i32; 8] = [32, 25, 19, 19, 0, 18, 0, 17];
const W_MULT: [i32; 8] = [85, 51, 37, 28, 0, 20, 0, 15];

/// Pick the deblock filter width from the boundary's local derivatives (dav2d
/// `filter_choice`). Examines the first (`s_off`) and last (`t_off`) of the 4
/// lines; returns 0 (skip) up to `max_width_pos`.
#[allow(clippy::too_many_arguments)]
fn filter_choice(dst: &[i32], s_off: isize, t_off: isize, stride: isize, max_width_neg: i32, max_width_pos: i32, q_thr: i32, side_thr: i32) -> i32 {
    // HARDENING: corrupt block geometry can push the filter taps outside the plane slice.
    let g = |o: isize| -> i32 { if o < 0 { 0 } else { *dst.get(o as usize).unwrap_or(&0) } };
    let s = |d: isize| g(s_off + d * stride);
    let t = |d: isize| g(t_off + d * stride);
    let mut sd = [0i32; 4]; // sd[dist+2] for dist -2..=1
    for dist in -2..2i32 {
        if !crate::av2_recon::work_tick("deblock:26") { break; }
        let d = dist as isize;
        let deriv_s = (s(d - 1) - (s(d) << 1) + s(d + 1)).abs();
        let deriv_t = (t(d - 1) - (t(d) << 1) + t(d + 1)).abs();
        sd[(dist + 2) as usize] = (deriv_s + deriv_t + 1) >> 1;
    }
    let sec = |dist: i32| sd[(dist + 2) as usize];
    let high_deriv = sec(-2).max(sec(1));

    if high_deriv > side_thr {
        return 0;
    }
    if max_width_pos == 1 {
        return 1;
    }
    let side_thr2 = side_thr >> 2;
    let mut transition = sec(-1) + sec(0);
    if high_deriv > side_thr2 {
        return 1;
    }
    if transition > q_thr * 4 {
        return 1;
    }
    let side_thr3 = side_thr >> 3;
    if high_deriv > side_thr3 {
        return 2;
    }
    if transition > q_thr * 3 {
        return 2;
    }
    let end_thr = (side_thr * 3) >> 4;
    if max_width_neg >= 3 {
        let ds = (s(-1) - s(-4) - 3 * (s(-1) - s(-2))).abs();
        let dt = (t(-1) - t(-4) - 3 * (t(-1) - t(-2))).abs();
        if ((ds + dt + 1) >> 1) > end_thr {
            return 2;
        }
    }
    let ds = (s(0) - s(3) - 3 * (s(0) - s(1))).abs();
    let dt = (t(0) - t(3) - 3 * (t(0) - t(1))).abs();
    if ((ds + dt + 1) >> 1) > end_thr {
        return 2;
    }
    if max_width_pos == 3 {
        return 3;
    }
    transition <<= 4;
    let mut prev_dist = 3;
    let mut dist = 4;
    while dist <= max_width_pos {
        let q_thr4 = q_thr * Q_FIRST[((dist - 4) >> 1) as usize];
        let end_thr4 = (side_thr * dist) >> 4;
        if transition > q_thr4 {
            return prev_dist;
        }
        let dist2 = 7.min(dist);
        if max_width_neg >= dist2 {
            let ds = (s(-1) - s(-(dist2 as isize) - 1) - dist2 * (s(-1) - s(-2))).abs();
            let dt = (t(-1) - t(-(dist2 as isize) - 1) - dist2 * (t(-1) - t(-2))).abs();
            if ((ds + dt + 1) >> 1) > end_thr4 {
                return prev_dist;
            }
        }
        let ds = (s(0) - s(dist2 as isize) - dist2 * (s(0) - s(1))).abs();
        let dt = (t(0) - t(dist2 as isize) - dist2 * (t(0) - t(1))).abs();
        if ((ds + dt + 1) >> 1) > end_thr4 {
            return prev_dist;
        }
        prev_dist = dist;
        dist += 2;
    }
    max_width_pos
}

/// Deblock one edge — 4 lines (dav2d `deblock`). Smooths a `width`-tapered delta
/// across the boundary at `center` (between `center-strideb` and `center`).
#[allow(clippy::too_many_arguments)]
pub fn deblock(dst: &mut [i32], center: usize, q_thr: i32, side_thr: i32, stridea: isize, strideb: isize, max_width_pos: i32, max_width_neg: i32, pos_lossless: bool, neg_lossless: bool, bitdepth_max: i32) {
    let c = center as isize;
    let width = filter_choice(dst, c, c + 3 * stridea, strideb, max_width_neg, max_width_pos, q_thr, side_thr);
    if width < 1 {
        return;
    }
    let width_neg = width.min(max_width_neg);
    let width_pos = width;
    let q_thr_clamp = q_thr * Q_THRESH_MULTS[(width - 1) as usize];

    let mut row = c;
    for _ in 0..4 {
        let p0 = dst[row as usize];
        let pm1 = dst[(row - strideb) as usize];
        let p1 = dst[(row + strideb) as usize];
        let pm2 = dst[(row - 2 * strideb) as usize];
        let delta_m2 = (4 * (3 * (p0 - pm1) - (p1 - pm2))).clamp(-q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let delta = delta_m2 * W_MULT[(width_neg - 1) as usize];
            for j in 0..width_neg {
                if !crate::av2_recon::work_tick("deblock:123") { break; }
                let idx = (row + (-j as isize - 1) * strideb) as usize;
                let diff = (delta * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] + diff).clamp(0, bitdepth_max);
            }
        }
        if !pos_lossless {
            let delta = delta_m2 * W_MULT[(width_pos - 1) as usize];
            for j in 0..width_pos {
                if !crate::av2_recon::work_tick("deblock:131") { break; }
                let idx = (row + j as isize * strideb) as usize;
                let diff = (delta * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] - diff).clamp(0, bitdepth_max);
            }
        }
        row += stridea;
    }
}

/// Luma max filter width by boundary-strength index (dav2d `max_width_y`).
pub fn max_width_y(idx: usize) -> i32 {
    MAX_WIDTH_Y[idx]
}

// ===== threshold derivation (dav2d db_apply_tmpl.c + quantizer.c) =====

const DQ_LOOKUP_TBL: [u8; 24] = [
    40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78,
];

/// dav2d `dav2d_dq_lookup` — the AC dequant magnitude used by the deblock quant threshold.
fn dq_lookup(qidx: i32) -> i32 {
    if qidx == 0 {
        return 64;
    }
    let q = qidx - 1;
    let shift = q / 24;
    (DQ_LOOKUP_TBL[(q % 24) as usize] as i32) << shift
}

/// dav2d `deblock_quant_thr`: `(dq_lookup(clip(qidx,0,qmax)) + 4) >> 9`, `qmax = 255 + 48*hbd`
/// (`hbd` = seq header high-bit-depth code: 0 = 8-bit, 1 = 10-bit, 2 = 12-bit).
pub fn deblock_quant_thr(qidx: i32, hbd: i32) -> i32 {
    let qmax = 255 + 48 * hbd;
    (dq_lookup(qidx.clamp(0, qmax)) + 4) >> 9
}

/// dav2d `deblock_side_thr`: table index shifts down by `24·(2·hbd)` and the rounding/shift
/// scale with `bitdepth_min_8 = 2·hbd`.
pub fn deblock_side_thr(qidx: i32, hbd: i32) -> i32 {
    let bd_min8 = 2 * hbd;
    let side = DEBLOCK_SIDE_THRESHOLDS[(qidx - 24 * bd_min8).clamp(0, 295) as usize] as i32;
    (side + (16 >> bd_min8)).max(0) >> (5 - bd_min8)
}

/// dav2d `dav2d_deblock_side_thresholds[296]` (tables.c). 8-bit high-derivative
/// threshold indexed by the (delta-adjusted) qindex.
#[rustfmt::skip]
const DEBLOCK_SIDE_THRESHOLDS: [i16; 296] = [
    -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,
    -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,
    -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,
    -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,  -16,
    -16,  -16,  -16,  -16,  -16,  -14,  -13,  -11,  -10,  -9,   -7,   -6,   -4,
    -3,   -2,   0,    0,    2,    3,    5,    6,    7,    9,    10,   12,   13,
    15,   16,   17,   19,   20,   22,   23,   24,   26,   27,   29,   30,   32,
    33,   34,   36,   37,   39,   40,   42,   43,   44,   46,   47,   49,   50,
    51,   53,   54,   56,   57,   59,   60,   61,   63,   64,   66,   67,   69,
    70,   71,   73,   74,   76,   77,   78,   80,   81,   83,   84,   86,   87,
    88,   90,   91,   93,   94,   96,   101,  111,  120,  130,  140,  150,  160,
    170,  180,  190,  200,  210,  220,  230,  240,  249,  259,  269,  279,  289,
    299,  309,  319,  329,  339,  349,  359,  368,  378,  388,  398,  408,  418,
    428,  438,  448,  458,  468,  478,  488,  497,  507,  517,  527,  537,  547,
    557,  567,  577,  587,  597,  607,  616,  626,  636,  646,  656,  666,  676,
    686,  696,  706,  716,  726,  736,  745,  755,  765,  775,  785,  795,  805,
    815,  825,  835,  845,  855,  864,  874,  884,  894,  904,  914,  924,  934,
    944,  954,  964,  974,  984,  993,  1003, 1013, 1023, 1033, 1043, 1053, 1063,
    1073, 1083, 1093, 1103, 1112, 1122, 1132, 1142, 1152, 1162, 1172, 1182, 1192,
    1202, 1212, 1222, 1232, 1241, 1251, 1261, 1271, 1281, 1291, 1301, 1311, 1321,
    1331, 1341, 1351, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450,
    1460, 1470, 1480, 1489, 1499, 1509, 1519, 1529, 1539, 1549, 1559, 1569, 1579,
    1589, 1599, 1608, 1618, 1628, 1638, 1648, 1658, 1668, 1678,
];

/// Full-frame AV2 deblock of one plane (dav2d `db_apply_tmpl.c` order, simplified for a
/// single-tile / no-segmentation / no-lossless keyframe). `buf` is a `stride`-major
/// `i32` plane modified in place. Edges come from the per-4×4 grids: `db_left[cell]`
/// marks a block-left (vertical) edge, `db_top` a block-top (horizontal) edge;
/// `db_lw`/`db_lh` are per-cell tx width/height levels. The edge strength index is `min`
/// of the two adjacent cells' levels, mapped through `max_width`. Iterates per band
/// (`band_cells` 4px rows = 64 luma px), vertical edges (cols) then horizontal (rows);
/// the band's top row caps max_width_neg at `band_neg_cap` (luma 6, chroma 2).
#[allow(clippy::too_many_arguments)]
thread_local! {
    /// Probe (env MDBW): per-edge intended max_width_pos, luma only — [0]=V edges, [1]=H.
    /// Fixed 108x60 grid (432x240 probe clips). Dumped per frame by filter_frame_chain.
    pub static WMAP: std::cell::RefCell<Vec<i8>> = std::cell::RefCell::new(Vec::new());
    /// Probe (env MDBW): [0..6480)=V mwp, [6480..12960)=H mwp, then V q_thr, H q_thr... i16 grid.
    pub static QMAP: std::cell::RefCell<Vec<i16>> = std::cell::RefCell::new(Vec::new());
    /// Probe tag: set by the frame filter driver around the U-plane call when DBLK444 is set.
    pub static DBG_TAG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn deblock_plane(
    buf: &mut [i32],
    iw4: usize,
    ih4: usize,
    stride: usize,
    db_lw: &[u8],
    db_lh: &[u8],
    db_left: &[bool],
    db_top: &[bool],
    q_thr_v: i32,
    side_thr_v: i32,
    q_thr_h: i32,
    side_thr_h: i32,
    // Per-64px-SB threshold override (delta-q frames; dav db_apply init_deblock_thr_lut per
    // lflvl->qidx): per-SB (qv, sv, qh, sh) tuples, SB grid width, and this plane's subsampling
    // (to map plane 4px cells onto luma SBs). None = the frame-level scalars above.
    sb_thr: Option<(&[(i32, i32, i32, i32)], usize, u32, u32)>,
    // Sub-PU WEAK-edge grids (dav masks[..][4] → deblock_tmpl setup_thr `>> 3*subpu`): a
    // marked cell's V (spv) / H (sph) edge filters with thresholds >> 3. Empty = no layer.
    spv: &[bool],
    sph: &[bool],
    bdmax: i32,
    max_width: &[i32],
    band_cells: usize,
    band_neg_cap: i32,
    apply_v: bool,
    apply_h: bool,
    tile_v_x4: &[usize],
) {
    let nbands = ih4.div_ceil(band_cells);
    for band in 0..nbands {
        if !crate::av2_recon::work_tick("deblock:255") { break; }
        let y0 = band * band_cells;
        let bh = (ih4 - y0).min(band_cells);
        // COLS pass — vertical edges, columns left-to-right (edges overlap horizontally,
        // so column order is normative). `edge`=1 only at tile-column boundaries
        // (`tile_v_x4`, plane 4px units): dav db_apply passes `tile_end == x64*16` into
        // deblock_sb, capping max_width_neg (the already-decoded left tile's side).
        // Gated per direction (dav db_apply_tmpl.c:442/587: cols on level_y[0], rows on level_y[1]).
        for x4 in (1..iw4).take_while(|_| apply_v) {
            let tile_edge = tile_v_x4.contains(&x4);
            for dy in 0..bh {
                if !crate::av2_recon::work_tick("deblock:265") { break; }
                let y4 = y0 + dy;
                let cell = y4 * iw4 + x4;
                if !db_left[cell] {
                    continue;
                }
                let level = db_lw[cell].min(db_lw[cell - 1]) as usize;
                let mwp = max_width[level];

                let mwn = if tile_edge { band_neg_cap.min(mwp) } else { mwp };
                let (q_thr_v, side_thr_v) = match sb_thr {
                    Some((g, gw, ssh, ssv)) => {
                        let sbx = ((x4 * 4) << ssh) >> 6;
                        let sby = ((y4 * 4) << ssv) >> 6;
                        let t = g[(sby * gw + sbx).min(g.len() - 1)];
                        // SB-boundary V edge: avg of the left SB's and this SB's thresholds
                        // (dav setup_thr_rows_sb64_dq_c: (cur+prev+1)>>1, 0 → the other).
                        if ((x4 * 4) << ssh) % 64 == 0 && sbx > 0 {
                            let p = g[(sby * gw + sbx - 1).min(g.len() - 1)];
                            let qq = if t.0 != 0 && p.0 != 0 { (t.0 + p.0 + 1) >> 1 } else { t.0 | p.0 };
                            let ss = if t.1 != 0 && p.1 != 0 { (t.1 + p.1 + 1) >> 1 } else { t.1 | p.1 };
                            (qq, ss)
                        } else {
                            (t.0, t.1)
                        }
                    }
                    None => (q_thr_v, side_thr_v),
                };
                let (q_thr_v, side_thr_v) = if spv.get(cell).copied().unwrap_or(false) {
                    (q_thr_v >> 3, side_thr_v >> 3)
                } else {
                    (q_thr_v, side_thr_v)
                };
                let center = (y4 * 4) * stride + x4 * 4;
                if DBG_TAG.with(|c| c.get()) && (204..=214).contains(&(x4 * 4)) && (84..=96).contains(&(y4 * 4)) {
                    crate::dlog!("[MDBV] xy=({},{}) lvl={} neg={} pos={} q={} side={}", x4 * 4, y4 * 4, level, mwn, mwp, q_thr_v, side_thr_v);
                }
                if max_width.len() > 3 && x4 < 108 && y4 < 60 {
                    QMAP.with(|w| { let mut w = w.borrow_mut(); if w.len() == 6 * 6480 { let i = y4 * 108 + x4; w[i] = mwp as i16; w[6480 * 2 + i] = q_thr_v as i16; w[6480 * 4 + i] = side_thr_v as i16; } });
                }
                deblock(buf, center, q_thr_v, side_thr_v, stride as isize, 1, mwp, mwn, false, false, bdmax);
            }
        }
        if !apply_h {
            continue;
        }
        // ROWS pass — horizontal edges, rows top-to-bottom (edges overlap vertically).
        // The band's top row (`dy==0`, global y4>0) caps max_width_neg (dav2d `!y`).
        for dy in 0..bh {
            if !crate::av2_recon::work_tick("deblock:313") { break; }
            let y4 = y0 + dy;
            if y4 == 0 {
                continue;
            }
            let band_top = dy == 0;
            for x4 in 0..iw4 {
                if !crate::av2_recon::work_tick("deblock:319") { break; }
                let cell = y4 * iw4 + x4;
                if std::env::var("DBQ13").is_ok() && x4 == 13 && y4 == 16 {
                    crate::dlog!("[DBQ13] rows pass reached (13,16): db_top={} lh={} lh_above={} apply_h={apply_h} mwlen={}", db_top[cell], db_lh[cell], db_lh[cell - iw4], max_width.len());
                }
                if !db_top[cell] {
                    continue;
                }
                let level = db_lh[cell].min(db_lh[cell - iw4]) as usize;
                let mwp = max_width[level];

                let mwn = if band_top { band_neg_cap.min(mwp) } else { mwp };
                let (q_thr_h, side_thr_h) = match sb_thr {
                    Some((g, gw, ssh, ssv)) => {
                        let sbx = ((x4 * 4) << ssh) >> 6;
                        let sby = ((y4 * 4) << ssv) >> 6;
                        let t = g[(sby * gw + sbx).min(g.len() - 1)];
                        // SB-boundary H edge: avg of the above SB's and this SB's thresholds.
                        if ((y4 * 4) << ssv) % 64 == 0 && sby > 0 {
                            let p = g[((sby - 1) * gw + sbx).min(g.len() - 1)];
                            let qq = if t.2 != 0 && p.2 != 0 { (t.2 + p.2 + 1) >> 1 } else { t.2 | p.2 };
                            let ss = if t.3 != 0 && p.3 != 0 { (t.3 + p.3 + 1) >> 1 } else { t.3 | p.3 };
                            (qq, ss)
                        } else {
                            (t.2, t.3)
                        }
                    }
                    None => (q_thr_h, side_thr_h),
                };
                let (q_thr_h, side_thr_h) = if sph.get(cell).copied().unwrap_or(false) {
                    (q_thr_h >> 3, side_thr_h >> 3)
                } else {
                    (q_thr_h, side_thr_h)
                };
                let center = (y4 * 4) * stride + x4 * 4;
                if DBG_TAG.with(|c| c.get()) && (204..=214).contains(&(x4 * 4)) && (84..=96).contains(&(y4 * 4)) {
                    crate::dlog!("[MDBH] xy=({},{}) lvl={} neg={} pos={} q={} side={}", x4 * 4, y4 * 4, level, mwn, mwp, q_thr_h, side_thr_h);
                }
                if max_width.len() > 3 && x4 < 108 && y4 < 60 {
                    QMAP.with(|w| { let mut w = w.borrow_mut(); if w.len() == 6 * 6480 { let i = y4 * 108 + x4; w[6480 + i] = mwp as i16; w[6480 * 3 + i] = q_thr_h as i16; w[6480 * 5 + i] = side_thr_h as i16; } });
                }
                deblock(buf, center, q_thr_h, side_thr_h, 1, stride as isize, mwp, mwn, false, false, bdmax);
            }
        }
    }
}

/// Luma / chroma boundary-strength → max filter width tables (dav2d `max_width_y/uv`).
pub const MAX_WIDTH_Y_TBL: [i32; 4] = MAX_WIDTH_Y;
pub const MAX_WIDTH_UV_TBL: [i32; 3] = MAX_WIDTH_UV;

#[cfg(test)]
mod tests {
    use super::*;

    // 4 lines × 24 columns, boundary at column 12; stridea = 24 (down a row),
    // strideb = 1 (across columns).
    const COLS: usize = 24;
    const BND: usize = 12;

    fn buf(neg: i32, pos: i32) -> Vec<i32> {
        let mut v = vec![0i32; 4 * COLS];
        for row in 0..4 {
            for x in 0..COLS {
                if !crate::av2_recon::work_tick("deblock:382") { break; }
                v[row * COLS + x] = if x < BND { neg } else { pos };
            }
        }
        v
    }

    #[test]
    fn flat_boundary_unchanged() {
        // No step → all derivatives 0 → delta 0 → no change.
        let mut v = buf(120, 120);
        let orig = v.clone();
        deblock(&mut v, BND, 40, 50, COLS as isize, 1, 8, 8, false, false, 255);
        assert_eq!(v, orig);
    }

    #[test]
    fn step_edge_is_smoothed() {
        // A clean 100|150 step at the boundary. filter_choice → width 6; the
        // boundary pixels move toward each other, tapered by distance.
        let mut v = buf(100, 150);
        deblock(&mut v, BND, 40, 50, COLS as isize, 1, 8, 8, false, false, 255);
        for row in 0..4 {
            let base = row * COLS;
            // boundary pixels smoothed toward the midpoint (hand-computed)
            assert_eq!(v[base + 11], 123, "neg boundary, row {row}"); // 100 + 23
            assert_eq!(v[base + 12], 127, "pos boundary, row {row}"); // 150 - 23
            // width-6 reach: col 6..11 (neg) and 12..17 (pos) touched, beyond not
            assert_eq!(v[base + 5], 100, "untouched neg");
            assert_eq!(v[base + 18], 150, "untouched pos");
            // monotone taper: closer to the edge moves more
            assert!(v[base + 11] > v[base + 10] && v[base + 10] > v[base + 9]);
        }
    }

    #[test]
    fn threshold_derivation_matches_oracle() {
        // Locks the dev clip's verified C1 thresholds (qidx=120, dq_y1=-2 → qidx 104):
        // luma V (dir 0) q=2/side=2, luma H (dir 1) q=1/side=2 — the values that produced
        // a byte-exact deblock on all three planes vs dav2d.
        assert_eq!(deblock_quant_thr(120, 0), 2);
        assert_eq!(deblock_side_thr(120, 0), 2);
        assert_eq!(deblock_quant_thr(104, 0), 1); // 120 + 8*(-2)
        assert_eq!(deblock_side_thr(104, 0), 2);
        // qidx 0 → dq_lookup base 64 → (64+4)>>9 = 0.
        assert_eq!(deblock_quant_thr(0, 0), 0);
        // clamp: qidx above 255/295 saturates the tables (no OOB).
        assert_eq!(deblock_quant_thr(1000, 0), deblock_quant_thr(255, 0));
        assert_eq!(deblock_side_thr(1000, 0), deblock_side_thr(295, 0));
    }

    #[test]
    fn lossless_sides_skip() {
        // neg_lossless leaves the negative side untouched; pos still filters.
        let mut v = buf(100, 150);
        deblock(&mut v, BND, 40, 50, COLS as isize, 1, 8, 8, false, true, 255);
        assert_eq!(v[11], 100, "neg side untouched when lossless");
        assert_eq!(v[12], 127, "pos side still filtered");
    }
}
