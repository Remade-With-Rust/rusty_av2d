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
    /// Per-unit NS filters for planes whose frame filters are OFF (`!ffon`): the unit
    /// codes its own filters against the tile bank; the apply must use THIS unit's
    /// decoded set, not the (empty) frame-filter slot. Keyed like LR_UNITS; value =
    /// the bank's slot-4 stash right after the unit's read, one row per class.
    pub static LR_UNIT_FILTERS: RefCell<std::collections::HashMap<(u8, usize, usize), [[i8; 18]; 16]>> =
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

/// Chroma NS-Wiener symmetric UV tap pairs, one representative `[dr,dc]` per pair
/// (avm `wienerns_simd_config_uv_from_uv`, features 0..5; the `{0,0,18}` center entry
/// is excluded by `num_pixels = len-1`, its effect carried by the subtract-center form).
const WIENER_NS_CONFIG_UV: [[i8; 2]; 6] = [[1, 0], [0, 1], [1, 1], [-1, 1], [2, 0], [0, 2]];
/// Cross-component taps into the downsampled luma (avm `wienerns_simd_config_uv_from_y`,
/// asymmetric singles; filter positions 6..17 in config order).
const WIENER_NS_CONFIG_UV_Y: [[i8; 2]; 12] = [
    [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
    [-1, 1], [1, -1], [2, 0], [-2, 0], [0, 2], [0, -2],
];

/// Chroma NS-Wiener loop restoration, whole-frame (avm `apply_wienerns_class_id_highbd`,
/// dual-input branch → `av2_convolve_nonsep_dual_highbd`).
///
/// The cross-component taps read a LUMA image downsampled to chroma resolution with the
/// seq's CfL downsample filter (`WIENERNS_CROSS_FILT_LUMA_TYPE == 2`:
/// `calc_wienerns_ds_luma_420`; 422/444 point-sample). avm builds that copy BEFORE any
/// LR applies (`wienerns_copy_luma_with_virtual_lines`), so the luma input here is the
/// PRE-LR luma (`src_l`), with the stripe-boundary virtual lines: the ±2 luma rows at
/// each stripe edge are the POST-DEBLOCK rows (`dblk_l`), and rows beyond those two are
/// plain pre-LR interior — unlike the chroma plane itself, whose out-of-stripe rows use
/// the standard replicate-the-saved-rows rule exactly like the luma apply.
///
/// Chroma is always single-class (`NUM_WIENERNS_CLASS_INIT_CHROMA == 1`), so there is no
/// classifier; `clip_base` in the reference is the identity, so tap differences are used
/// raw. PC-Wiener chroma units have no oracle stream yet and are refused loudly.
#[allow(clippy::too_many_arguments)]
pub fn lr_filter_chroma(
    dst: &mut [i32],
    src: &[i32],
    dblk: &[i32],
    p: usize,
    cw: usize,
    ch: usize,
    src_l: &[i32],
    dblk_l: &[i32],
    lw: usize,
    lh: usize,
    ssh: usize,
    ssv: usize,
    ds_type: u8,
    bdmax: i32,
) {
    let cfg = LR_CFG.with(|c| c.borrow().clone());
    let pd = &cfg.p[p];
    if pd.r_type == REST_NONE {
        return;
    }
    if std::env::var("MLRU").is_ok() {
        crate::dlog!("[MLRC] p={p} ds_type={ds_type} ffon={} ncls={} unit={} cw={cw} ch={ch}", pd.ffon as u8, pd.num_classes, 1usize << cfg.unit_size[1]);
    }
    let unit_sz_log2 = cfg.unit_size[1] as usize;
    let unit_sz = 1usize << unit_sz_log2;
    let units = LR_UNITS.with(|u| u.borrow().clone());
    // Stripe geometry in chroma rows: first stripe 64-8 luma rows, then 64-row bands,
    // all >> ss_v (avm limits: RESTORATION_PROC_UNIT_SIZE / RESTORATION_UNIT_OFFSET >> ss_y).
    let band = 64usize >> ssv;
    let first = band - (8 >> ssv);
    let stripe_of = |y: usize| -> (usize, usize) {
        if y < first { (0, first.min(ch)) } else {
            let s0 = first + (y - first) / band * band;
            (s0, (s0 + band).min(ch))
        }
    };
    let mut y0 = 0usize;
    while y0 < ch {
        if !crate::av2_recon::work_tick("av2_lr:chroma_stripe") { break; }
        let (sy0, sy1) = stripe_of(y0);
        debug_assert_eq!(sy0, y0);
        // Chroma-plane accessor: standard LR stripe rule (interior = pre-LR src,
        // out-of-stripe = the two saved post-deblock rows, replicated beyond).
        let px = |x: i32, y: i32| -> i32 {
            let xx = x.clamp(0, cw as i32 - 1) as usize;
            if y < sy0 as i32 {
                if sy0 == 0 { src[xx] } else { dblk[(y.max(sy0 as i32 - 2) as usize) * cw + xx] }
            } else if y >= sy1 as i32 {
                if sy1 >= ch { src[(ch - 1) * cw + xx] } else { dblk[(y.min(sy1 as i32 + 1) as usize) * cw + xx] }
            } else {
                src[y as usize * cw + xx]
            }
        };
        // Luma accessor for the ds build, with this stripe's LUMA bounds and the
        // virtual-lines rule (±2 rows deblock, then pre-LR interior, frame-edge clamp).
        let sy0l = (sy0 << ssv).min(lh);
        let sy1l = (sy1 << ssv).min(lh);
        // Same out-of-stripe rule as the luma filter's own accessor: avm's
        // setup_processing_stripe_boundary replaces RESTORATION_BORDER_VERT=4 rows
        // on each side, built from the 2 saved post-deblock rows with the OUTERMOST
        // saved row duplicated outward (src_row = max(i+2,0) above / min below). A
        // clamp into the two dblk rows reproduces that exactly; falling back to
        // pre-LR src beyond +-2 does not (the outer ds border rows were the last
        // 125 differing bytes of the cpu3 repro).
        let pxl = |x: i32, y: i32| -> i32 {
            let xx = x.clamp(0, lw as i32 - 1) as usize;
            let yy = y.clamp(0, lh as i32 - 1);
            if yy < sy0l as i32 {
                if sy0l == 0 { src_l[xx] } else { dblk_l[(yy.max(sy0l as i32 - 2) as usize) * lw + xx] }
            } else if yy >= sy1l as i32 && sy1l < lh {
                dblk_l[(yy.min(sy1l as i32 + 1) as usize) * lw + xx]
            } else {
                src_l[yy as usize * lw + xx]
            }
        };
        // Downsampled-luma accessor at chroma coords (avm calc_wienerns_ds_luma_420 /
        // make_wienerns_ds_luma; ds_type = seq cfl_ds_filter_index, plain shifts, no rounding).
        let dsl = |x: i32, y: i32| -> i32 {
            // Clamp in CHROMA space: avm extends the ds buffer by replicating its
            // outermost ROWS/cols, so an off-frame tap reads the replicated ds row
            // (built from the last TWO luma rows) — clamping the luma rows instead
            // would average the final luma row with itself.
            let x = x.clamp(0, cw as i32 - 1);
            let y = y.clamp(0, ch as i32 - 1);
            let (lx, ly) = (x << ssh, y << ssv);
            if ssh == 1 && ssv == 1 {
                match ds_type {
                    1 => (pxl(lx, ly) + pxl(lx, ly + 1)) >> 1,
                    2 => pxl(lx, ly),
                    _ => (pxl(lx, ly) + pxl(lx + 1, ly) + pxl(lx, ly + 1) + pxl(lx + 1, ly + 1)) >> 2,
                }
            } else {
                pxl(lx, ly)
            }
        };
        if p == 1 {
            if let Ok(path) = std::env::var("MLRDS") {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                let mut rows: Vec<i32> = Vec::new();
                if sy0 > 0 { rows.push(sy0 as i32 - 2); rows.push(sy0 as i32 - 1); }
                for y in sy0..sy1 { rows.push(y as i32); }
                if sy1 < ch { rows.push(sy1 as i32); rows.push(sy1 as i32 + 1); }
                for y in rows {
                    for x in 0..cw {
                        let v = dsl(x as i32, y) as u16;
                        f.write_all(&v.to_le_bytes()).unwrap();
                    }
                }
            }
        }
        // Restoration-unit row for this stripe (dav lr_sbrow +8 convention, subsampled),
        // with the last-unit absorb rule on both axes.
        let row_y = if y0 == 0 { 0 } else { y0 + (8 >> ssv) };
        let mut ay = row_y & !(unit_sz - 1);
        if ay != 0 && ay + unit_sz / 2 > ch {
            ay -= unit_sz;
        }
        let n_cols = 1usize.max((cw + unit_sz / 2) >> unit_sz_log2);
        for k in 0..n_cols {
            if !crate::av2_recon::work_tick("av2_lr:chroma_unit") { break; }
            let x0 = k << unit_sz_log2;
            let x1 = if k + 1 == n_cols { cw } else { (k + 1) << unit_sz_log2 };
            let u_type = units.get(&(p as u8, x0, ay)).copied().unwrap_or(REST_NONE);
            if u_type == REST_NONE {
                continue;
            }
            if u_type != REST_NS {
                crate::dlog!("[rav2d] WARNING: PC-Wiener chroma restoration unit is unverified (no oracle stream)");
                continue;
            }
            let unit_f = if pd.ffon { None } else {
                LR_UNIT_FILTERS.with(|u| u.borrow().get(&(p as u8, x0, ay)).map(|f| f[0]))
            };
            let filter: &[i8; 18] = match (&unit_f, pd.ffon) {
                (Some(f), _) => f,
                (None, true) => &pd.filter[0], // frame filters, chroma single-class
                (None, false) => {
                    // An NS unit whose banked filters are missing = the parse/apply unit
                    // keying disagrees (the bug class the cpu3 campaign was made of).
                    // Skipping silently would ship an unfiltered unit as success — say so.
                    crate::dlog!("[rav2d] WARNING: NS restoration unit (p={p} x={x0} y={ay}) has no banked filters — unit left unfiltered (keying bug?)");
                    debug_assert!(false, "NS unit without banked filters: p={p} x0={x0} ay={ay}");
                    continue;
                }
            };
            for y in sy0..sy1 {
                for x in x0..x1 {
                    let m = px(x as i32, y as i32);
                    let mut s = m << 7;
                    for (i, c) in WIENER_NS_CONFIG_UV.iter().enumerate() {
                        let (dr, dc) = (c[0] as i32, c[1] as i32);
                        let diff = px(x as i32 + dc, y as i32 + dr) + px(x as i32 - dc, y as i32 - dr) - 2 * m;
                        s += diff * filter[i] as i32;
                    }
                    let lm = dsl(x as i32, y as i32);
                    for (j, c) in WIENER_NS_CONFIG_UV_Y.iter().enumerate() {
                        let (dr, dc) = (c[0] as i32, c[1] as i32);
                        s += (dsl(x as i32 + dc, y as i32 + dr) - lm) * filter[6 + j] as i32;
                    }
                    dst[y * cw + x] = ((s + 64) >> 7).clamp(0, bdmax);
                }
            }
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
    LR_UNIT_FILTERS.with(|u| u.borrow_mut().clear());
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
/// Read the restoration-unit headers this SB owns, for planes `p_start..p_end`.
///
/// The plane range mirrors avm's per-tree schedule (decodeframe.c:2081,
/// `get_partition_plane_start/end(xd->tree_type)`): an inter/SHARED tree reads
/// ALL planes' units at the SB root (0..3), but an SDP key frame runs two tree
/// passes per SB — the LUMA pass reads plane 0's units before the luma tree,
/// then the CHROMA pass reads planes 1..3 before the chroma tree, AFTER the
/// whole luma subtree has decoded. Reading all three up front desyncs the
/// entropy stream on any key frame with chroma LR (the cpu3 real-photo repro).
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
    p_start: usize,
    p_end: usize,
) {
    let cfg = LR_CFG.with(|c| c.borrow().clone());
    if !cfg.enabled() {
        return;
    }
    let sbsz = crate::av2_recon::sb_step4() * 4; // SB size in px
    for p in p_start..p_end {
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
                if std::env::var("MLRU").is_ok() {
                    crate::dlog!("[MLRUPRE] p={p} unit=({},{}) rng={} nscdf={}", fx + (x << usz_log2), fy + (y << usz_log2), msac.rng, cdf.m.rst_ns_wiener[0]);
                }
                let t = read_restoration_info(msac, cdf, p, frame_type);
                if std::env::var("MLRU").is_ok() {
                    crate::dlog!("[MLRU] p={p} unit=({},{}) type={t} rng={}", fx + (x << usz_log2), fy + (y << usz_log2), msac.rng);
                }
                LR_UNITS.with(|u| {
                    u.borrow_mut().insert((p as u8, fx + (x << usz_log2), fy + (y << usz_log2)), t);
                });
                if t == REST_NS && !cfg.p[p].ffon {
                    let snap = NS_BANK.with(|b| b.borrow()[p].filter[4]);
                    if p > 0 && std::env::var("MLRU").is_ok() {
                        crate::dlog!("[MLRF] p={p} unit=({},{}) taps={:?}", fx + (x << usz_log2), fy + (y << usz_log2), &snap[0][..18]);
                    }
                    LR_UNIT_FILTERS.with(|u| {
                        u.borrow_mut().insert((p as u8, fx + (x << usz_log2), fy + (y << usz_log2)), snap);
                    });
                }
            }
        }
    }
}
