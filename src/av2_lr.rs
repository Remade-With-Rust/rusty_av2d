//! AV2 loop restoration (NS-Wiener / PC-Wiener) — dav2d obu.c restoration parse state +
//! decode.c per-SB unit syntax + lr_apply_tmpl.c / looprestoration_tmpl.c collapsed to a
//! whole-frame pass. The dav stripe/row-backup machinery exists because dav filters in
//! place; the filter only ever READS pre-LR pixels (rows below the cursor are unfiltered,
//! rows above come from backups, left/right strips from pre-LR borders), so with a full
//! pre-wiener snapshot the pass is a pure src→dst transform with two boundary rules:
//!   - frame edges replicate (clamp x/y into the frame);
//!   - STRIPE boundaries (y = 56 + 64k, single tile): the 2 rows beyond the stripe come
//!     from the saved POST-DEBLOCK lines, rows further out replicate those.
//! GDF (already ported, avm semantics) runs after, with the PRE-wiener snapshot as its
//! guided input — mirroring dav's gdf_prep(pre-wiener)/gdf_add(post-wiener) split.

use crate::av2_lr_tables::*;
use std::cell::{Cell, RefCell};

pub const REST_NONE: u8 = 0;
pub const REST_PC: u8 = 1;
pub const REST_NS: u8 = 2;
pub const REST_SWITCHABLE: u8 = 3;

//         C
//   E     A     F      (dav looprestoration_tmpl.c wiener_ns_config_y)
pub const WIENER_NS_CONFIG_Y: [[i8; 2]; 16] = [
    [1, 0], [0, 1], [2, 0], [0, 2],
    [1, 1], [-1, 1], [2, 1], [2, -1],
    [1, 2], [1, -2], [3, 0], [0, 3],
    [4, 0], [0, 4], [3, 3], [3, -3],
];
pub const PC_WIENER_CONFIG: [[i8; 2]; 12] = [
    [1, 0], [0, 1], [2, 0], [0, 2], [1, 1], [-1, 1],
    [2, 1], [2, -1], [1, 2], [1, -2], [3, 0], [0, 3],
];
pub const PC_WIENER_NORMALIZER: [u16; 3] = [3739, 3273, 3074];

#[derive(Clone)]
pub struct LrPlane {
    pub r_type: u8,
    pub ffon: bool,
    pub temporal: bool,
    pub refidx: u8,
    pub num_classes_idx: u8,
    pub num_classes: u8,
    pub filter: [[i8; 18]; 16],
}
impl Default for LrPlane {
    fn default() -> Self {
        LrPlane { r_type: 0, ffon: false, temporal: false, refidx: 0, num_classes_idx: 0, num_classes: 0, filter: [[0; 18]; 16] }
    }
}
#[derive(Clone, Default)]
pub struct LrFrameCfg {
    pub p: [LrPlane; 3],
    pub unit_size: [u8; 2],
}
impl LrFrameCfg {
    pub fn enabled(&self) -> bool {
        self.p.iter().any(|p| p.r_type != REST_NONE)
    }
}

thread_local! {
    /// Current frame's parsed restoration config (frame header).
    pub static LR_CFG: RefCell<LrFrameCfg> = RefCell::new(LrFrameCfg::default());
    /// Per-ref-slot restoration configs (temporal filter inheritance reads these).
    pub static LR_SLOT: RefCell<[Option<LrFrameCfg>; 8]> = RefCell::new(std::array::from_fn(|_| None));
    /// Seq rst_disable_mask[0..1] (luma, chroma).
    pub static SEQ_RST_MASK: Cell<(u8, u8)> = const { Cell::new((0, 0)) };
    /// Per-frame parsed LR units: (plane, aligned_px_x, aligned_px_y) -> unit type.
    pub static LR_UNITS: RefCell<std::collections::HashMap<(u8, usize, usize), u8>> =
        RefCell::new(std::collections::HashMap::new());
}

/// dav decode.c:86 `init_wiener` q-bucket: 0..3 from the frame's yac qindex.
pub fn lr_q_idx(yac: u32, hbd: bool) -> usize {
    let qmax = 255 + 24 * hbd as u32;
    let qidx = yac.min(qmax);
    if qidx < 130 { 0 } else if qidx < 190 { 1 } else if qidx < 220 { 2 } else { 3 }
}

/// dav looprestoration.h WienerQValApproxData + setup (per-frame, from the dq step).
#[derive(Clone)]
pub struct QvalApprox {
    pub error_lut: [i8; 3 * 64],
    pub offsets: [i32; 3],
    pub slopes: [i16; 3],
    pub idx0_changeover: i8,
    pub class_lut_offset: usize,
}
impl Default for QvalApprox {
    fn default() -> Self {
        QvalApprox { error_lut: [0; 3 * 64], offsets: [0; 3], slopes: [0; 3], idx0_changeover: 0, class_lut_offset: 0 }
    }
}

pub fn setup_qval_approx(qstep: i32) -> QvalApprox {
    const MODE_WEIGHTS: [[i32; 3]; 4] = [
        [-527, 15325, 321],
        [26436, -17705, 17905],
        [366, -147, -194],
        [202, -267, -179],
    ];
    const MODE_OFFSETS: [i32; 4] = [-547, -21565, -573, -680];
    let apply_sign = |v: i32, s: i32| if s < 0 { -v } else { v };
    let mut data = QvalApprox::default();
    let mut init_idx0 = 0i32;
    let mut idx0_low = 0i32;
    let mut idx0_changeover = -1i32;
    for s in 0..=36i32 {
        let tskip = s * 7;
        let prod = (tskip * qstep + 128) >> 8;
        let mut qval = MODE_WEIGHTS[0][0] * (tskip << 5) + MODE_WEIGHTS[0][1] * qstep + MODE_WEIGHTS[0][2] * prod;
        let abs_qval = qval.abs();
        qval = apply_sign((abs_qval + 4096) >> 13, qval);
        qval = 255 * (MODE_OFFSETS[0] + qval);
        qval = (qval.max(0) + 8192) >> 14;
        qval = qval.min(255) >> 5;
        if s == 0 {
            init_idx0 = qval;
            idx0_low = qval;
        } else if qval != init_idx0 {
            idx0_low = qval;
            idx0_changeover = s;
            break;
        }
    }
    data.idx0_changeover = idx0_changeover as i8;
    data.class_lut_offset = (idx0_low as usize) << 9;
    for i in 0..3usize {
        let mut slope = 7 * 32 * MODE_WEIGHTS[i + 1][0]
            + ((7 * MODE_WEIGHTS[i + 1][2] as i64 * qstep as i64) >> 8) as i32;
        slope = (slope + 4096) >> 13;
        let mut tmp = [0i32; 37];
        for s in 0..=36i32 {
            let tskip = s * 7;
            let prod = (tskip * qstep + 128) >> 8;
            let mut qval = MODE_WEIGHTS[i + 1][0] * (tskip << 5) + MODE_WEIGHTS[i + 1][1] * qstep + MODE_WEIGHTS[i + 1][2] * prod;
            let abs_qval = qval.abs();
            qval = apply_sign((abs_qval + 4096) >> 13, qval);
            qval = MODE_OFFSETS[i + 1] + qval;
            tmp[s as usize] = qval - s * slope;
        }
        data.slopes[i] = slope as i16;
        data.offsets[i] = tmp[0];
        for s in 0..=36usize {
            data.error_lut[i * 64 + s] = (tmp[s] - tmp[0]) as i8;
        }
    }
    data
}

/// dav wiener_tables.c dav2d_init_wiener_classes: full 4096-entry class LUTs built from
/// lut_to_class + the sub-classify tables (lazy, once).
static WIENER_CLASSES: std::sync::OnceLock<(Vec<[u8; 4096]>, Vec<[[u8; 4096]; 7]>)> = std::sync::OnceLock::new();
pub fn wiener_classes() -> &'static (Vec<[u8; 4096]>, Vec<[[u8; 4096]; 7]>) {
    WIENER_CLASSES.get_or_init(|| {
        let mut pre = vec![[0u8; 4096]; 4];
        let mut user = vec![[[0u8; 4096]; 7]; 4];
        for i in 0..4 {
            for k in 0..4096 {
                let cls = LUT_TO_CLASS[k] as usize;
                pre[i][k] = PRETRAINED_SUB_CLASSIFY[i][cls];
                for j in 0..7 {
                    user[i][j][k] = USER_SUB_CLASSIFY[i][j][cls];
                }
            }
        }
        (pre, user)
    })
}

/// Whole-frame LUMA restoration pass. `src` = pre-wiener (post-CCSO) snapshot; `dblk` =
/// post-deblock plane (stripe-boundary lines); `dst` filtered in place (starts == src
/// content). Unit types come from LR_UNITS; classification reads `lr_noskip` (per-4px
/// luma cells with coded coefficients, dav lr_noskip_mask).
#[allow(clippy::too_many_arguments)]
pub fn lr_filter_luma(
    dst: &mut [i32],
    src: &[i32],
    dblk: &[i32],
    w: usize,
    h: usize,
    yac: u32,
    bdmax: i32,
    lr_noskip: &[bool],
    iw4: usize,
) {
    let cfg = LR_CFG.with(|c| c.borrow().clone());
    let pd = &cfg.p[0];
    if pd.r_type == REST_NONE {
        return;
    }
    let bit_depth = (bdmax + 1).trailing_zeros() as i32;
    let hbd = bit_depth > 8;
    let bitdepth_min_8 = bit_depth - 8;
    // dav init_wiener: qstep from the frame's dq step.
    let base_q = crate::av2_dequant::dq_lookup(yac) as i32;
    let qstep = (base_q + ((1 << bitdepth_min_8) >> 1)) >> bitdepth_min_8;
    let qa = setup_qval_approx(qstep);
    let qi = lr_q_idx(yac, hbd);
    let (pre_cls, user_cls) = wiener_classes();
    let pc_filters: &[[i16; 13]; 64] = &PC_WIENER_FILTERS[qi];
    let unit_sz_log2 = cfg.unit_size[0] as usize;
    let unit_sz = 1usize << unit_sz_log2;
    let units = LR_UNITS.with(|u| u.borrow().clone());

    // Source accessor: stripe-aware row resolution + frame-edge clamp. Stripes (single
    // tile): [0,56), then 64-row bands. Rows outside the current stripe: the 2 adjacent
    // POST-DEBLOCK rows, then replicate; frame top/bottom replicate src.
    let stripe_of = |y: usize| -> (usize, usize) {
        if y < 56 { (0, 56.min(h)) } else { (56 + (y - 56) / 64 * 64, (56 + (y - 56) / 64 * 64 + 64).min(h)) }
    };

    let mut out_rows: Vec<i32> = vec![0; w]; // scratch per row

    // Iterate stripes × 64-col strips (the strip only matters for classification's
    // noskip-window clamp — pixel access is global thanks to the snapshot).
    let mut y0 = 0usize;
    while y0 < h {
        if !crate::av2_recon::work_tick("av2_lr:218") { break; }
        let (sy0, sy1) = stripe_of(y0);
        debug_assert_eq!(sy0, y0);
        let px = |x: i32, y: i32| -> i32 {
            let xx = x.clamp(0, w as i32 - 1) as usize;
            if y < sy0 as i32 {
                if sy0 == 0 {
                    src[xx] // frame top: replicate row 0
                } else {
                    let ry = y.max(sy0 as i32 - 2) as usize; // post-deblock rows sy0-2, sy0-1
                    dblk[ry * w + xx]
                }
            } else if y >= sy1 as i32 {
                if sy1 >= h {
                    src[(h - 1) * w + xx] // frame bottom: replicate last row
                } else {
                    let ry = y.min(sy1 as i32 + 1) as usize; // post-deblock rows sy1, sy1+1
                    dblk[ry * w + xx]
                }
            } else {
                src[y as usize * w + xx]
            }
        };
        let mut x0 = 0usize;
        while x0 < w {
            if !crate::av2_recon::work_tick("av2_lr:242") { break; }
            let sw = (w - x0).min(64);
            // Resolve this (strip, stripe)'s restoration unit (lr_sbrow placement rules).
            // The sbrow convention offsets by +8: stripe [56,120) belongs to unit row 64
            // (dav lr_sbrow row_y = y + 8*!first_sb_in_tile_row).
            let row_y = if y0 == 0 { 0 } else { y0 + 8 };
            let mut ay = row_y & !(unit_sz - 1);
            if ay != 0 && ay + unit_sz / 2 > h {
                ay -= unit_sz;
            }
            let mut ax = x0 & !(unit_sz - 1);
            if ax != 0 && w - ax < unit_sz / 2 {
                ax -= unit_sz;
            }
            let u_type = units.get(&(0u8, ax, ay)).copied().unwrap_or(REST_NONE);
            if u_type == REST_NONE {
                x0 += 64;
                continue;
            }
            // classification (PC-wiener or multi-class NS): per 4px block of the strip.
            let multi = u_type == REST_PC || (u_type == REST_NS && pd.num_classes > 1);
            let bh_blk = (sy1 - sy0).div_ceil(4);
            let mut classes = vec![0u8; sw.div_ceil(4)];
            for y in y0..sy1 {
                if multi && (y - sy0) % 4 == 0 {
                    let by = (y - sy0) / 4;
                    for (bxi, cls) in classes.iter_mut().enumerate() {
                        // gradient features over the 6x6 window (dav get_class_lut_idx)
                        let mut f2 = [0i32; 3];
                        for dy in -1i32..=4 {
                            for dx in -1i32..=4 {
                                let x = (x0 + bxi * 4) as i32 + dx;
                                let yy = y as i32 + dy;
                                let m = px(x, yy);
                                f2[0] += (px(x, yy - 1) - 2 * m + px(x, yy + 1)).abs();
                                f2[1] += (px(x + 1, yy - 1) - 2 * m + px(x - 1, yy + 1)).abs();
                                f2[2] += (px(x - 1, yy - 1) - 2 * m + px(x + 1, yy + 1)).abs();
                            }
                        }
                        // skip-count over the 3x3 4px-block neighbourhood, clamped to the
                        // 64px strip window + the stripe rows (dav noskip_mask semantics).
                        let mut s = 0i32;
                        const NUM_PIXELS: [i32; 3] = [16, 4, 1];
                        let strip_c0 = x0 / 4;
                        let strip_cmax = ((x0 + 63) / 4).min(iw4 - 1); // last valid col in strip
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let edge = (dy != 0) as usize + (dx != 0) as usize;
                                let fx = ((bxi as i32 + dx).clamp(0, 15) as usize + strip_c0).min(strip_cmax);
                                let fy_blk = (by as i32 + dy).clamp(0, bh_blk as i32 - 1) as usize;
                                let cell_y = (sy0 / 4 + fy_blk).min(h.div_ceil(4) - 1);
                                let ns = lr_noskip.get(cell_y * iw4 + fx).copied().unwrap_or(false);
                                s += NUM_PIXELS[edge] * !ns as i32;
                            }
                        }
                        let s = s as usize;
                        let rnd = (1i32 << bitdepth_min_8) >> 1;
                        for i in 0..3 {
                            f2[i] = (f2[i] * PC_WIENER_NORMALIZER[i] as i32 + rnd) >> bitdepth_min_8;
                        }
                        let mut lut_idx = (((s as i32) < qa.idx0_changeover as i32) as usize) << 9;
                        for i in 0..3 {
                            let qval = qa.slopes[i] as i32 * s as i32 + qa.error_lut[i * 64 + s] as i32 + qa.offsets[i];
                            let mut sub = (0.max(f2[i] + 255 * qval) + 8192) >> 14;
                            sub = sub.min(255) >> 5;
                            lut_idx |= (sub as usize) << (3 * (2 - i));
                        }
                        let cls_lut: &[u8; 4096] = if u_type == REST_PC {
                            &pre_cls[qi]
                        } else {
                            &user_cls[qi][pd.num_classes_idx as usize - 1]
                        };
                        *cls = cls_lut[lut_idx + qa.class_lut_offset];
                    }
                }
                // filter the row's strip pixels
                for bxi in 0..sw.div_ceil(4) {
                    let xs = x0 + bxi * 4;
                    let xe = (xs + 4).min(w);
                    for x in xs..xe {
                        let v = if u_type == REST_NS {
                            let filter: &[i8; 18] = if pd.num_classes > 1 {
                                &pd.filter[classes[bxi] as usize]
                            } else {
                                &pd.filter[0]
                            };
                            let m = px(x as i32, y as i32);
                            let mut s = m << 7;
                            for (i, cfgi) in WIENER_NS_CONFIG_Y.iter().enumerate() {
                                let (dy, dx) = (cfgi[0] as i32, cfgi[1] as i32);
                                let diff = px(x as i32 + dx, y as i32 + dy) + px(x as i32 - dx, y as i32 - dy) - 2 * m;
                                s += diff * filter[i] as i32;
                            }
                            (s + 64) >> 7
                        } else {
                            // PC-Wiener: pretrained 13-tap
                            let filter = &pc_filters[classes[bxi] as usize];
                            let mut s = px(x as i32, y as i32) * filter[12] as i32;
                            for (i, cfgi) in PC_WIENER_CONFIG.iter().enumerate() {
                                let (dy, dx) = (cfgi[0] as i32, cfgi[1] as i32);
                                s += filter[i] as i32 * (px(x as i32 + dx, y as i32 + dy) + px(x as i32 - dx, y as i32 - dy));
                            }
                            (s + 64) >> 7
                        };
                        out_rows[x] = v.clamp(0, bdmax);
                    }
                }
                for x in x0..(x0 + sw).min(w) {
                    dst[y * w + x] = out_rows[x];
                }
            }
            x0 += 64;
        }
        y0 = sy1;
    }
}

// ===== Per-SB restoration-unit syntax (dav decode.c:4590 loop + read_restoration_info) =====

/// Per-tile NS-Wiener filter bank (dav Dav2dTileState.ns_wiener_bank), luma+chroma.
#[derive(Clone)]
pub struct NsWienerBank {
    pub filter: [[[i8; 18]; 16]; 5],
    pub bank_size: [u8; 16],
    pub bank_idx: [u8; 16],
}
impl Default for NsWienerBank {
    fn default() -> Self {
        NsWienerBank { filter: [[[0; 18]; 16]; 5], bank_size: [0; 16], bank_idx: [0; 16] }
    }
}
thread_local! {
    pub static NS_BANK: RefCell<[NsWienerBank; 3]> = RefCell::new(std::array::from_fn(|_| NsWienerBank::default()));
}

/// dav decode.c:4283 tile-init: reset the banks; slot-0 filters = mid-range defaults.
pub fn lr_tile_init() {
    let cfg = LR_CFG.with(|c| c.borrow().clone());
    NS_BANK.with(|b| {
        let mut banks = b.borrow_mut();
        for pl in 0..3 {
            banks[pl] = NsWienerBank::default();
            if cfg.p[pl].r_type == REST_NS || cfg.p[pl].r_type == REST_SWITCHABLE {
                let n_classes = cfg.p[pl].num_classes as usize;
                for n in 0..n_classes.min(16) {
                    for m in 0..(16 + 2 * (pl != 0) as usize) {
                        let (nbits, lo) = if pl != 0 {
                            (crate::av2_lr_tables::NS_WIENER_COEF_RANGE_UV[m][0], crate::av2_lr_tables::NS_WIENER_COEF_RANGE_UV[m][1])
                        } else {
                            (crate::av2_lr_tables::NS_WIENER_COEF_RANGE_Y[m][0], crate::av2_lr_tables::NS_WIENER_COEF_RANGE_Y[m][1])
                        };
                        banks[pl].filter[0][n][m] = (lo as i32 + ((1i32 << nbits) >> 1)) as i8;
                    }
                }
            }
        }
    });
    LR_UNITS.with(|u| u.borrow_mut().clear());
}

/// dav decode.c:4306 decode_4way: 2-bit adaptive bin + bypass remainder, recentered.
fn decode_4way(msac: &mut crate::msac::MsacContext, refv: i32, cdf: &mut [u16; 4], n_bits: i32) -> i32 {
    fn inv_recenter(r: i32, v: i32) -> i32 {
        if v > 2 * r { v } else if v & 1 == 0 { r + (v >> 1) } else { r - ((v + 1) >> 1) }
    }
    let bin = crate::msac::rav1d_msac_decode_symbol_adapt4(msac, cdf, 3) as i32;
    let nb = n_bits + bin + (bin == 0) as i32 - 4;
    let mut rem = 0i32;
    for _ in 0..nb {
        rem = (rem << 1) | crate::msac::rav1d_msac_decode_bool_bypass(msac) as i32;
    }
    let v = if bin != 0 { 1 << (n_bits + bin - 4) } else { 0 } + rem;
    let n = 1 << n_bits;
    if refv * 2 <= n { inv_recenter(refv, v) } else { n - 1 - inv_recenter(n - 1 - refv, v) }
}

/// dav decode.c:4320 read_restoration_info: one unit's type (+ per-unit NS filters when
/// the frame banks are off). Returns the unit type.
pub fn read_restoration_info(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    p: usize,
    frame_type: u8,
) -> u8 {
    use crate::msac::{rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bool_bypass};
    let u_type;
    if frame_type == REST_SWITCHABLE {
        if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.rst_switchable[0]) {
            u_type = REST_NONE;
        } else {
            let t = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.rst_switchable[1]);
            u_type = if t { REST_PC } else { REST_NS };
        }
    } else {
        let c = if frame_type == REST_NS { &mut cdf.m.rst_ns_wiener } else { &mut cdf.m.rst_pc_wiener };
        let t = rav1d_msac_decode_bool_adapt(msac, c);
        u_type = if t { frame_type } else { REST_NONE };
    }
    let (ffon, n_classes) = LR_CFG.with(|c| {
        let cfg = c.borrow();
        (cfg.p[p].ffon, cfg.p[p].num_classes as usize)
    });
    if u_type == REST_NS && !ffon {
        // per-unit filters coded against the tile bank (dav decode.c:4340)
        NS_BANK.with(|b| {
            let mut banks = b.borrow_mut();
            let bank = &mut banks[p];
            let n_feat = 16 + 2 * (p != 0) as usize;
            let mut exact_match_mask = 0u32;
            let mut bank_refs = [0usize; 16];
            for n in 0..n_classes {
                let exact = rav1d_msac_decode_bool_bypass(msac);
                let bank_size = bank.bank_size[n] as i32;
                let mut r = 0i32;
                while r < bank_size - 1 {
                    if !crate::av2_recon::work_tick("av2_lr:456") { break; }
                    if rav1d_msac_decode_bool_bypass(msac) {
                        break;
                    }
                    r += 1;
                }
                let r = ((bank.bank_idx[n] as i32 - r) & 3) as usize;
                exact_match_mask |= (exact as u32) << n;
                bank_refs[n] = r;
            }
            for n in 0..n_classes {
                let r = bank_refs[n];
                let ref_filter = bank.filter[r][n];
                if exact_match_mask & (1 << n) != 0 {
                    // stash the unit filter in slot 4 (the apply reads it)
                    bank.filter[4][n] = ref_filter;
                    if bank.bank_size[n] == 0 {
                        bank.bank_size[n] = 1;
                    }
                    continue;
                }
                let mut filter = [0i8; 18];
                let mut s = 0usize;
                while s < 3 - (p != 0) as usize {
                    if !crate::av2_recon::work_tick("av2_lr:479") { break; }
                    if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.wiener_ns_len[(p != 0) as usize]) {
                        break;
                    }
                    s += 1;
                }
                let mask: u32 = if p != 0 {
                    crate::av2_lr_tables::SUBSET_MASKS_UV[s]
                } else {
                    crate::av2_lr_tables::SUBSET_MASKS_Y[s]
                };
                let asym = p != 0 && s != 0 && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.wiener_ns_sym);
                let mut i = 0usize;
                let mut m = mask;
                while i < n_feat {
                    if !crate::av2_recon::work_tick("av2_lr:493") { break; }
                    if m & 1 != 0 {
                        let (nbits, lo) = if p != 0 {
                            (crate::av2_lr_tables::NS_WIENER_COEF_RANGE_UV[i][0], crate::av2_lr_tables::NS_WIENER_COEF_RANGE_UV[i][1])
                        } else {
                            (crate::av2_lr_tables::NS_WIENER_COEF_RANGE_Y[i][0], crate::av2_lr_tables::NS_WIENER_COEF_RANGE_Y[i][1])
                        };
                        filter[i] = (decode_4way(msac, ref_filter[i] as i32 - lo as i32, &mut cdf.m.wiener_ns_cf, nbits as i32) + lo as i32) as i8;
                        if asym && i >= 6 {
                            filter[i + 1] = filter[i];
                            i += 1;
                            m >>= 1;
                        }
                    }
                    i += 1;
                    m >>= 1;
                }
                bank.filter[4][n] = filter;
                let bidx = ((1 + bank.bank_idx[n]) & 3) as usize;
                bank.bank_idx[n] = bidx as u8;
                bank.filter[bidx][n] = filter;
                if bank.bank_size[n] < 4 {
                    bank.bank_size[n] += 1;
                }
            }
        });
    }
    u_type
}

/// Per-SB restoration-unit loop (dav decode.c:4590), single tile. `bx_sb`/`by_sb` in 4px
/// cells; `iw4`/`ih4` frame dims in cells; `ssh`/`ssv` chroma subsampling.
#[allow(clippy::too_many_arguments)]
pub fn read_lr_units_sb(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    bx_sb: usize,
    by_sb: usize,
    iw4: usize,
    ih4: usize,
    ssh: usize,
    ssv: usize,
) {
    let cfg = LR_CFG.with(|c| c.borrow().clone());
    if !cfg.enabled() {
        return;
    }
    let sbsz = crate::av2_recon::sb_step4() * 4; // SB size in px
    for p in 0..3usize {
        let frame_type = cfg.p[p].r_type;
        if frame_type == REST_NONE {
            continue;
        }
        let (ss_hor, ss_ver) = if p == 0 { (0usize, 0usize) } else { (ssh, ssv) };
        let tx = (4 * bx_sb) >> ss_hor;
        let ty = (4 * by_sb) >> ss_ver;
        let usz_log2 = cfg.unit_size[(p != 0) as usize] as usize;
        let unit_sz = 1usize << usz_log2;
        if (tx | ty) & (unit_sz - 1) != 0 {
            continue;
        }
        let tw = (iw4 * 4) >> ss_hor;
        let th = (ih4 * 4) >> ss_ver;
        let half = unit_sz >> 1;
        let (fx, fy) = (tx, ty);
        if (ty != 0 && fy + half > th) || (tx != 0 && fx + half > tw) {
            continue;
        }
        let sbw = sbsz >> ss_hor;
        let sbh = sbsz >> ss_ver;
        let lruw = 1usize.max((tw - fx + half).min(sbw) >> usz_log2);
        let lruh = 1usize.max((th - fy + half).min(sbh) >> usz_log2);
        for y in 0..lruh {
            for x in 0..lruw {
                let t = read_restoration_info(msac, cdf, p, frame_type);
                if std::env::var("MLRU").is_ok() {
                    crate::dlog!("[MLRU] p={p} unit=({},{}) type={t} rng={}", fx + (x << usz_log2), fy + (y << usz_log2), msac.rng);
                }
                LR_UNITS.with(|u| {
                    u.borrow_mut().insert((p as u8, fx + (x << usz_log2), fy + (y << usz_log2)), t);
                });
            }
        }
    }
}
