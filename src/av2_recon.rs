//! AV2 real `decode_b` neighbour-context machinery (dav2d `decode_b`, intra-luma
//! path). The hand-wired obu.rs scaffold proved every symbol + context bit-exact
//! against the oracle across three block types; this module turns that into the
//! *structural* decoder: live above/left neighbour arrays whose splats make every
//! context COMPUTED rather than hand-analysed.
//!
//! The core primitive is `gather_nb`: dav2d collects up to two spatial neighbours
//! (the `nb[0..2]`/`boff[0..2]` logic). When a block has only a left neighbour (top
//! of SB), BOTH slots collapse onto the left edge — which is exactly why a 4x4 next
//! to an `fsc=1` block reads `ctx = 2*1 = 2`.

/// Scaffold plumbing of parsed-header tool flags into the block decode. The
/// defaults are the dev clip's values (gdf/cdef/ccso enabled, intrabc allowed),
/// so existing verified runs are byte-identical; streams that DISABLE a tool in
/// their headers (e.g. rusty_av2e's minimal E3 streams) set these from the parse
/// so the per-SB filter symbols / intrabc flag are not read — matching the
/// normative gating (dav2d obu.c: cdef/gdf/ccso params only exist when enabled).

/// Decode-order per-leaf luma recon scorer (env MSCORE=<path to avm pre-filter yuv>): after a
/// leaf's luma write, compare the block region of FRAME pl0 against the reference dump and
/// print the first divergent leaves in DECODE order (intrabc frames propagate raster-late
/// errors to raster-early positions, so raster-order first-diff is misleading).
pub fn mscore_luma(tag: &str, px0: usize, py0: usize, w: usize, h: usize, blk: &[i32], bstride: usize) {
    thread_local! {
        static MSCORE_REF: std::cell::RefCell<Option<Vec<u8>>> = const { std::cell::RefCell::new(None) };
        static MSCORE_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let Ok(path) = std::env::var("MSCORE") else { return };
    if let Ok(fsel) = std::env::var("MSCOREF") {
        let cur = crate::av2_frame::DECODE_FRAME_N.with(|c| c.get());
        if fsel.parse::<u32>().ok() != Some(cur) { return; }
    }
    if MSCORE_N.with(|c| c.get()) > 12 { return; }
    MSCORE_REF.with(|r| {
        let mut rr = r.borrow_mut();
        if rr.is_none() {
            *rr = std::fs::read(&path).ok();
        }
        let Some(refbuf) = rr.as_ref() else { return };
        let fw = 432usize; // key clip dims (probe-only)
        let fh = 240usize;
        let mut nd = 0usize;
        let mut first = None;
        for yy in 0..h {
            let y = py0 + yy;
            if y >= fh { break; }
            for xx in 0..w {
                let x = px0 + xx;
                if x >= fw { break; }
                let mv = blk[yy * bstride + xx].clamp(0, 255) as u8;
                let rv = refbuf[y * fw + x];
                if mv != rv {
                    nd += 1;
                    if first.is_none() { first = Some((x, y, mv, rv)); }
                }
            }
        }
        if nd > 0 {
            MSCORE_N.with(|c| c.set(c.get() + 1));
            crate::dlog!("[MSCORE] {tag} px=({px0},{py0}) {w}x{h} nd={nd} first={first:?}");
        }
    });
}

/// Inter-intra blend (dav2d recon_tmpl.c:2889 `iiblend`): build the II intra predictor
/// (DC / V / H / SMOOTH) over the block from the frame-recon edges and blend it over the
/// inter/warp prediction with the II mask (`(inter*(64-m) + intra*m + 32) >> 6`).
/// `lx4/ly4/lw4/lh4` = the block in LUMA 4px cells (drives availability + the SMOOTH
/// n_tr/n_bl rules); `px0/py0/w/h` = this PLANE's pixel region; `pl` = plane index.
#[allow(clippy::too_many_arguments)]
pub fn ii_blend(
    pred: &mut [i32],
    pl: usize,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    lx4: usize,
    ly4: usize,
    lw4: usize,
    lh4: usize,
    mode: i8,
) {
    const WTS: [u8; 64] = [
        60, 56, 52, 48, 45, 42, 39, 37, 34, 32, 30, 28, 26, 24, 22, 21,
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 10, 9, 8, 8, 7, 7,
        6, 6, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2,
        2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    if mode < 0 {
        return;
    }
    let bdmax = bdmax_g();
    let base = (bdmax + 1) >> 1;
    let tb = TILE_B.with(|t| t.get());
    let (cs4, ce4, rs4, re4) = (tb.0, tb.1, tb.2, tb.3);
    let have_top = ly4 > rs4;
    let have_left = lx4 > cs4;
    let (ssh, ssv) = if pl == 0 { (0usize, 0usize) } else { let sg = ss_g(); (sg.0, sg.1) };
    crate::av2_frame::FRAME.with(|fr| {
        let f = fr.borrow();
        let plane = &f.pl[pl];
        if plane.w == 0 {
            return;
        }
        // SMOOTH n_tr / n_bl (dav iiblend: top-SB-boundary → full width; else 1px max from the
        // decode-order availability grid). Luma cells; other II modes take 0.
        let (mut n_tr, mut n_bl) = (0usize, 0usize);
        if mode == 3 {
            let (ce4c, re4c) = (ce4.min(f.iw4), re4.min(f.ih4));
            if have_top {
                let wa = (lw4 as i32).min(ce4c as i32 - (lx4 + lw4) as i32);
                let sbs = crate::av2_recon::sb_step4();
                if ly4 & (sbs - 1) == 0 {
                    n_tr = wa.max(0) as usize;
                } else {
                    let end = ((lx4 + sbs) & !(sbs - 1)).min(ce4c);
                    let w2 = wa.min(end as i32 - (lx4 + lw4) as i32);
                    if w2 > 0 {
                        let idx = pl.min(1); // chroma availability from the chroma grid
                        let grid = if idx == 0 { &f.mi_coded } else { &f.mi_coded_c };
                        let (gr, gc) = if idx == 0 {
                            ((ly4.wrapping_sub(1)) & (sbs - 1), (lx4 + lw4) & (sbs - 1))
                        } else {
                            (((ly4 >> ssv).wrapping_sub(1)) & (sbs - 1), ((lx4 + lw4) >> ssh) & (sbs - 1))
                        };
                        n_tr = grid[gr * 32 + gc] as usize;
                    }
                }
            }
            if have_left {
                let sbs = crate::av2_recon::sb_step4();
                let end = ((ly4 + sbs) & !(sbs - 1)).min(re4c);
                let h2 = (lh4 as i32).min(end as i32 - (ly4 + lh4) as i32);
                if h2 > 0 {
                    if lx4 & (sbs - 1) == 0 {
                        n_bl = h2 as usize;
                    } else {
                        let idx = pl.min(1);
                        let grid = if idx == 0 { &f.mi_coded } else { &f.mi_coded_c };
                        let (gr, gc) = if idx == 0 {
                            ((ly4 + lh4) & (sbs - 1), (lx4.wrapping_sub(1)) & (sbs - 1))
                        } else {
                            (((ly4 + lh4) >> ssv) & (sbs - 1), ((lx4.wrapping_sub(1)) >> ssh) & (sbs - 1))
                        };
                        n_bl = grid[gr * 32 + gc] as usize;
                    }
                }
            }
        }
        let n_tr_px = ((n_tr << 2) >> ssh).min(w);
        let n_bl_px = ((n_bl << 2) >> ssv).min(h);
        let (top, left, _corner) = crate::av2_frame::gather_edges(plane, px0, py0, w, h, have_top, have_left, n_tr_px, n_bl_px, base);
        let mut intra = vec![0i32; w * h];
        match mode {
            0 => {
                use crate::av2_ipred::*;
                match (have_top, have_left) {
                    (true, true) => ipred_dc(&mut intra, w, &top, &left, w, h, bdmax),
                    (false, true) => ipred_dc_left(&mut intra, w, &left, w, h),
                    (true, false) => ipred_dc_top(&mut intra, w, &top, w, h),
                    (false, false) => ipred_dc_128(&mut intra, w, w, h, bdmax),
                }
                // dav iiblend: apply_ibp = seq ibp && max(ssbw4, ssbh4) > 1 (plane cells).
                let apply_ibp = f.ibp && (w >> 2).max(h >> 2) > 1;
                if apply_ibp && (have_top || have_left) {
                    ipred_ibp_dc(&mut intra, w, &top, &left, w, h, have_top, have_left);
                }
            }
            1 => crate::av2_ipred::ipred_v(&mut intra, w, &top, w, h),
            2 => crate::av2_ipred::ipred_h(&mut intra, w, &left, w, h),
            _ => crate::av2_ipred::ipred_smooth(&mut intra, w, &top, &left, w, h),
        }
        // II mask (dav wedge.c build_nondc_ii_masks; step = 64/max(w,h); DC = flat 32).
        let step = 64 / w.max(h);
        for y in 0..h {
            for x in 0..w {
                let m = match mode {
                    0 => 32i32,
                    1 => WTS[y * step] as i32,
                    2 => WTS[x * step] as i32,
                    _ => WTS[x.min(y) * step] as i32,
                };
                let i = y * w + x;
                pred[i] = (pred[i] * (64 - m) + intra[i] * m + 32) >> 6;
            }
        }
    });
}

/// Chroma twin of [`mscore_luma`] (env MSCORE + MSCOREF): compare a chroma leaf's recon at
/// write time vs the reference dump (Y+U+V layout, 4:2:0 8-bit assumed for the probe).
pub fn mscore_chroma(pl: usize, cpx: usize, cpy: usize, cw: usize, ch: usize, blk: &[i32]) {
    thread_local! {
        static MSC_REF: std::cell::RefCell<Option<Vec<u8>>> = const { std::cell::RefCell::new(None) };
        static MSC_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let Ok(path) = std::env::var("MSCORE") else { return };
    if let Ok(fsel) = std::env::var("MSCOREF") {
        let cur = crate::av2_frame::DECODE_FRAME_N.with(|c| c.get());
        if fsel.parse::<u32>().ok() != Some(cur) { return; }
    }
    if MSC_N.with(|c| c.get()) > 12 { return; }
    MSC_REF.with(|r| {
        let mut rr = r.borrow_mut();
        if rr.is_none() {
            *rr = std::fs::read(&path).ok();
        }
        let Some(refbuf) = rr.as_ref() else { return };
        let (fw, fh) = (216usize, 120usize);
        let off = 432 * 240 + (pl - 1) * fw * fh;
        let mut nd = 0usize;
        let mut first = None;
        for yy in 0..ch {
            let y = cpy + yy;
            if y >= fh { break; }
            for xx in 0..cw {
                let x = cpx + xx;
                if x >= fw { break; }
                let mv = blk[yy * cw + xx].clamp(0, 255) as u8;
                let rv = refbuf[off + y * fw + x];
                if mv != rv {
                    nd += 1;
                    if first.is_none() { first = Some((x, y, mv, rv)); }
                }
            }
        }
        if nd > 0 {
            MSC_N.with(|c| c.set(c.get() + 1));
            crate::dlog!("[MSCOREC] pl={pl} cpx=({cpx},{cpy}) {cw}x{ch} nd={nd} first={first:?}");
        }
    });
}

/// TX-partition parse (dav2d decode.c:1378 `read_tx_part`, non-lossless arm): for a non-skip
/// block <=64px under frame `txfm_mode == SWITCHABLE`, decode the tx-split flag and, when set,
/// the partition kind. Returns the TxPartition (0=NONE, 1=SPLIT, 2=H, 3=V, 4=H4, 5=V4,
/// 6=H5, 7=V5). Context groups are keyed by the block DIMS (mine's bs codes differ from dav's).
pub fn read_tx_part(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    bw4: usize,
    bh4: usize,
    fsc: bool,
    inter: bool,
    skip_txfm: bool,
) -> u8 {
    use crate::msac::{rav1d_msac_decode_bool_adapt, rav1d_msac_decode_symbol_adapt8};
    let cfg = HDR_TOOL_CFG.with(|c| c.get());
    if skip_txfm || !cfg.tx_switchable || (bw4 == 1 && bh4 == 1) || bw4.max(bh4) > 16 {
        return 0;
    }
    // size_to_tx_part_group_lookup (decode.c:1404), by dims (4px cells):
    let szctx: usize = match (bw4, bh4) {
        (16, 16) => 7,
        (16, 8) | (8, 16) => 6,
        (8, 8) => 5,
        (8, 4) | (4, 8) => 4,
        (4, 4) => 3,
        (4, 2) | (2, 4) => 2,
        (2, 2) => 1,
        (2, 1) | (1, 2) | (1, 1) => 0,
        _ => 8, // 1:4+ shapes
    };
    let f = fsc as usize;
    let i = inter as usize;
    let is_split = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.tx_split[f][i][szctx]);
    if std::env::var("MTXP").is_ok() {
        crate::dlog!("[MTXP] split={} szctx={szctx} fsc={f} inter={i} rng={}", is_split as u8, msac.rng);
    }
    if !is_split {
        return 0;
    }
    if bw4.min(bh4) >= 2 {
        // 2D arm: size_to_tx_type_group_vh_lookup (decode.c:1439), by dims:
        let vhctx: usize = match (bw4, bh4) {
            (16, 16) => 9,
            (16, 8) => 8,
            (8, 16) => 7,
            (8, 8) => 6,
            (8, 4) => 5,
            (4, 8) => 4,
            (4, 4) => 3,
            (4, 2) => 2,
            (2, 4) => 1,
            (2, 2) => 0,
            (16, 4) => 13,
            (4, 16) => 12,
            (16, 2) | (8, 2) => 11,
            (2, 16) | (2, 8) => 10,
            _ => 0,
        };
        let part = 1 + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.tx2d.tx_part_2d[f][i][vhctx], 6);
        if std::env::var("MTXP").is_ok() { crate::dlog!("[MTXP] part2d={part} vhctx={vhctx} rng={}", msac.rng); }
        part
    } else if bw4.max(bh4) >= 4 {
        // 1D arm (thin 1:4+ blocks): ctx = bw4 >= 4 cells (16px wide).
        let ctx = (bw4 >= 4) as usize;
        let four = if SEQ_REDUCED_TX_PART.with(|c| c.get()) {
            false
        } else {
            rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.tx_part_1d[f][i][ctx])
        };
        // TX_PARTITION_H(2) + ctx + 4way*2
        (2 + ctx + four as usize * 2) as u8
    } else {
        // 4x8 -> H, 8x4 -> V
        if bh4 > bw4 { 2 } else { 3 }
    }
}

/// TX-unit layout for a block under `tx_part`: avm `partition_shift_bits` +
/// `get_tx_partition_sizes` mirrored exactly. Returns (x4, y4, w4, h4) per unit in
/// DECODE order — offsets in 4px cells. NONE=block, SPLIT=quad, HORZ/VERT=halves,
/// HORZ4/VERT4=quarter strips, and HORZ5/VERT5 are 5-unit NON-uniform layouts
/// (HORZ5 = two (w/2,h/4) top, one (w,h/2) middle, two (w/2,h/4) bottom).
pub fn tx_part_layout(bw4: usize, bh4: usize, part: u8) -> Vec<(usize, usize, usize, usize)> {
    let (rs, cs, ro, co): (&[usize], &[usize], &[usize], &[usize]) = match part {
        1 => (&[1, 1, 1, 1], &[1, 1, 1, 1], &[0, 0, 1, 1], &[0, 1, 0, 1]),
        2 => (&[1, 1], &[0, 0], &[0, 1], &[0, 0]),
        3 => (&[0, 0], &[1, 1], &[0, 0], &[0, 1]),
        4 => (&[2, 2, 2, 2], &[0, 0, 0, 0], &[0, 1, 2, 3], &[0, 0, 0, 0]),
        5 => (&[0, 0, 0, 0], &[2, 2, 2, 2], &[0, 0, 0, 0], &[0, 1, 2, 3]),
        6 => (&[2, 2, 1, 2, 2], &[1, 1, 0, 1, 1], &[0, 0, 1, 3, 3], &[0, 1, 0, 0, 1]),
        7 => (&[1, 1, 0, 1, 1], &[2, 2, 1, 2, 2], &[0, 1, 0, 0, 1], &[0, 0, 1, 3, 3]),
        _ => (&[0], &[0], &[0], &[0]),
    };
    // 128px blocks disallow TX partitioning (avm coding_block_disallows_tx_partitioning):
    // part is always NONE and the block tiles into 64x64 max-TX units, row-major.
    if part == 0 && (bw4 > 16 || bh4 > 16) {
        let (uw, uh) = (bw4.min(16), bh4.min(16));
        let mut v = Vec::new();
        for uy in (0..bh4).step_by(uh) {
            for ux in (0..bw4).step_by(uw) {
                v.push((ux, uy, uw, uh));
            }
        }
        return v;
    }
    let mut wstep = bw4 / 2;
    let mut hstep = bh4 / 2;
    if part == 6 { hstep /= 2; }
    if part == 7 { wstep /= 2; }
    if part == 4 || part == 5 { wstep /= 2; hstep /= 2; }
    rs.iter()
        .zip(cs)
        .zip(ro.iter().zip(co))
        .map(|((&r, &c), (&rof, &cof))| {
            (cof * wstep, rof * hstep, (bw4 >> c).max(1), (bh4 >> r).max(1))
        })
        .collect()
}

/// Sub-PU deblock filter level (dav2d lf_mask.c:241 `subpu_flt_lvl`): a sub-PU-refined inter
/// block (TIP ref / compound-opfl / refinemv+AVG) gets weak inner deblock edges at a
/// `1<<subpu_l2` cell cadence. Returns 3 (= no sub-PU layer) for everything else.
/// `bw4`/`bh4` are the FULL block dims (b_dim, unclamped).
#[allow(clippy::too_many_arguments)]
pub fn subpu_flt_lvl(
    intra: bool,
    lf_sub_pu: bool,
    is_tip_block: bool,
    is_comp: bool,
    inter_mode: u8,
    refine_mv: bool,
    comp_is_avg: bool,
    bw4: usize,
    bh4: usize,
) -> usize {
    if intra || !lf_sub_pu {
        return 3;
    }
    if is_tip_block {
        let cfg = HDR_TOOL_CFG.with(|c| c.get());
        let seq_tip_refine = SEQ_TIP.with(|c| c.get()).4;
        let opfl = seq_tip_refine && (cfg.tip_frame_mode == 1 || cfg.tip_subpel_filter == 2);
        // dav: 1 + (fm==2 ? !opfl : ((!opfl && min>=4) || bs==BS_256x256))
        let big = bw4 == 64 && bh4 == 64 && false; // BS_256x256 needs 256px SBs (unsupported)
        return 1 + if cfg.tip_frame_mode == 2 {
            !opfl as usize
        } else {
            ((!opfl && bw4.min(bh4) >= 4) || big) as usize
        };
    }
    if is_comp {
        if inter_mode >= 24 {
            // OPFL_* compound modes: level 1, except 8x8 blocks → 0 (4px cadence).
            return 1 - (bw4 == 2 && bh4 == 2) as usize;
        }
        if refine_mv && comp_is_avg {
            return 2;
        }
    }
    3
}

/// Per-SB delta-q parse (dav2d decode.c:1941): at the FIRST has_luma leaf of every 64px SB
/// (`!((bx|by) & 15)`), when the frame header set `delta.q.present`, decode the delta-q symbol
/// chain and update the running `LAST_QIDX`. `bs`/`skip_txfm` implement dav's
/// `(bs != root_bs || !b->skip_txfm)` read gate (a whole-SB skip block codes no delta-q).
pub fn read_delta_q(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    bs: usize,
    skip_txfm: bool,
    bx4: usize,
    by4: usize,
) {
    if (bx4 | by4) & (sb_step4() - 1) != 0 {
        return;
    }
    let cfg = HDR_TOOL_CFG.with(|c| c.get());
    if !cfg.delta_q_present {
        return;
    }
    if !(bs == 6 && skip_txfm) {
        use crate::msac::{rav1d_msac_decode_bool_bypass, rav1d_msac_decode_bools_bypass, rav1d_msac_decode_symbol_adapt8};
        let mut delta_q = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.delta_q, 7) as i32;
        if delta_q == 7 {
            let n_bits = 1 + rav1d_msac_decode_bools_bypass(msac, 3) as i32;
            delta_q = rav1d_msac_decode_bools_bypass(msac, n_bits as u8) as i32 + 1 + (1 << n_bits);
        }
        if delta_q != 0 {
            if rav1d_msac_decode_bool_bypass(msac) {
                delta_q = -delta_q;
            }
            delta_q *= 1 << cfg.delta_q_res_log2;
        }
        let qmax = 255i32; // 8-bit (dav: 255 + 48*hbd — plumb hbd when a 10-bit delta-q clip exists)
        let new_qidx = (LAST_QIDX.with(|c| c.get()) as i32 + delta_q).clamp(1, qmax) as u32;
        LAST_QIDX.with(|c| c.set(new_qidx));
        // dav init_quant_tables(new_qidx): the per-plane DC/AC quantizers follow the NEW qindex
        // (frame deltas unchanged). The AC luma path reads dq_lookup(LAST_QIDX) directly; the
        // DC + chroma paths read F2_DCQ/F2_ACQ, so recompute them here.
        let d = crate::av2_frame::F2_QDELTAS.with(|c| c.get());
        let clipq = |dd: i32| (new_qidx as i32 + dd).clamp(0, qmax) as u32;
        crate::av2_frame::F2_DCQ.with(|c| c.set([
            crate::av2_dequant::dq_lookup(clipq(d[0])),
            crate::av2_dequant::dq_lookup(clipq(d[1])),
            crate::av2_dequant::dq_lookup(clipq(d[2])),
        ]));
        crate::av2_frame::F2_ACQ.with(|c| c.set([
            crate::av2_dequant::dq_lookup(clipq(0)),
            crate::av2_dequant::dq_lookup(clipq(d[3])),
            crate::av2_dequant::dq_lookup(clipq(d[4])),
        ]));
        if std::env::var("MDQ").is_ok() {
            crate::dlog!("[MDQ] mi=({bx4},{by4}) dq={delta_q} qidx={new_qidx} rng={}", msac.rng);
        }
    }
    // Per-SB qidx for the deblock lut (dav lf_mask->qidx): stored for EVERY SB, read or not.
    if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
        let q = LAST_QIDX.with(|c| c.get());
        crate::av2_frame::FRAME.with(|f| f.borrow_mut().set_sb_qidx(bx4, by4, q as u16));
    }
}

/// Sequence-header tool enables (see SEQ_TOOLS).
#[derive(Clone, Copy)]
pub struct SeqTools {
    pub bawp: bool,
    pub adaptive_mvd: bool,
    pub mvd_sign_derive: bool,
    /// enable_mrls — gates the per-block mrl_index symbol (directional intra).
    pub mrls: bool,
    /// enable_ist / enable_inter_ist — gate the intra / inter secondary-transform symbols.
    pub ist: bool,
    pub ist_inter: bool,
    /// enable_cctx — gates the cross-chroma-transform symbol in the chroma coef decode.
    pub cctx: bool,
    /// enable_ibp + enable_intra_edge_filter — feed reset_frame's recon config.
    pub ibp: bool,
    pub edge_filter: bool,
    /// enable_drl_reorder in THIS decoder's numbering: 0=disabled, 2=constraint, 1=always
    /// (avm reads bit1=1 -> disabled; else bit2=1 -> constraint, 0 -> always; the corpus'
    /// default-constraint streams are byte-exact with 2 — the historical hardcode).
    pub drl_reorder: u8,
    /// enable_refmvbank — gates the ref-MV bank as a DRL candidate source.
    pub refmvbank: bool,
    /// Seq enabled_motion_modes mask (bit n = mode n enabled; 1=INTERINTRA, 2=WARP_CAUSAL,
    /// 3=WARP_DELTA, 4=WARP_EXTEND; bit 0 TRANSLATION always set). Gates the per-block
    /// warp/interintra mode symbols (was hardcoded 0x1d).
    pub motion_modes: u32,
    /// Seq six_param_warp_delta — WARP_DELTA wri==1 blocks code 4 params instead of 2.
    pub six_param_warp: bool,
}
impl SeqTools {
    pub const DEFAULT: SeqTools = SeqTools {
        bawp: true, adaptive_mvd: true, mvd_sign_derive: true, mrls: true,
        ist: true, ist_inter: true, cctx: true, ibp: true, edge_filter: true,
        drl_reorder: 2, refmvbank: true, motion_modes: 0x1d, six_param_warp: true,
    };
}

#[derive(Clone, Copy)]
pub struct HdrToolCfg {
    pub gdf: bool,
    pub cdef: bool,
    pub ccso: bool,
    pub allow_intrabc: bool,
    /// seq `cfl` enable: when false, `cfl_allowed` is force-false and the per-leaf
    /// `is_cfl` bool is not coded (dav2d decode.c:2159 gates on the seq flag).
    pub cfl: bool,
    /// seq `cfl_ds_filter_index` — the CfL luma downsampling filter: 0=UNIFORM,
    /// 1=VSTRIP, 2=GAUSS (dav2d `CFL_FLT_TYPE_*`).
    pub cfl_ds_filter: u8,
    /// frame `subpel_filter_mode` (Rav1dFilterMode repr: 0=Regular,1=Smooth,2=Sharp,
    /// 3=Bilinear,4=Switchable). An inter block codes its interp `filter` symbol ONLY
    /// when this == Switchable(4) (dav2d decode.c:3272); otherwise `filter = this value`
    /// with no symbol. Default 4 keeps switchable-clip behaviour unchanged.
    pub subpel_filter_mode: u8,
    /// frame `mv_precision` (dav2d obu.c:1392, ∈ {0,1,2,3}). Seeds the per-block inter MV
    /// precision `mv_prec = 3 + this` + the mvprec_rem table index. Default 2 (v432 f2).
    pub mv_precision: u8,
    /// frame `warp_motion` (dav2d obu.c:1943): the per-block `allow_warp` symbol is coded ONLY
    /// when this is set (decode.c:2965). Default true (2-frame clips); v432_8f f1 = false.
    pub warp_motion: bool,
    /// frame `tip.frame_mode` (dav2d obu.c:1246): 0=disabled, 1=TIP-as-reference, 2=TIP-as-output.
    /// When !=0 the frame's blocks can reference the synthesized TIP_FRAME. Default 0.
    pub tip_frame_mode: u8,
    /// frame `tip.subpel_filter` (dav2d obu.c:1276, coded only for frame_mode==2):
    /// 0=REGULAR, 1=SMOOTH, 2=SHARP (default SHARP).
    pub tip_subpel_filter: u8,
    /// frame `tip.gmv` (dav2d obu.c:1268, frame_mode==2 only): the whole-frame TIP block MV.
    pub tip_gmv: (i32, i32),
    /// frame `tip.apply_filter` (dav2d obu.c:1300, frame_mode==2 + seq db_sub_pu): deblock the
    /// synthesized TIP frame (levels forced 1). Default false.
    pub tip_apply_filter: bool,
    /// frame `skip_mode_enabled` (dav2d obu.c): when set, a block with `bw4*bh4 > 2` (and not an
    /// intra region) codes a `skip_mode` flag BEFORE is_inter (decode.c:1658). Default false.
    pub skip_mode_enabled: bool,
    /// frame `enable_bawp` (avm decodeframe.c:9537, coded for non-intra frames when seq bawp):
    /// gates the per-block bawp flag. Default true (all prior inter clips coded bawp=1).
    pub bawp: bool,
    /// frame `allow_screen_content_tools` — gates av2_allow_palette (blockd.h:3362): a luma
    /// DC_PRED leaf of 8x8..64x64 codes the palette_y_mode symbol. Default false.
    pub allow_scc: bool,
    /// frame RESOLVED `force_integer_mv` (avm read_screen_content_params): when set, an
    /// intrabc BV codes NO precision bit (is_qpel forced false). Default 0.
    pub force_integer_mv: u8,
    /// frame `switchable_comp_refs` (reference_select): when set, a non-tip block with `bw4*bh4>=4`
    /// codes an `is_comp` (comp[ctx]) flag between is_tip and the ref decode (decode.c:2461). When
    /// off, every block is single-reference (no is_comp symbol). Default false.
    pub switchable_comp_refs: bool,
    /// frame `opfl_refine_type` (dav2d obu.c:1250): 0=none, 1=switchable (per-block opfl symbol on
    /// compound modes), 2=always. Gates the compound `opfl` symbol (decode.c:2591). Default 0.
    pub opfl_refine_type: u8,
    /// frame `delta.q.present` (dav2d obu.c:1534): when set, the first has_luma leaf of every
    /// 64px SB codes a delta-q symbol chain (decode.c:1941) that updates the running qindex.
    pub delta_q_present: bool,
    /// frame `delta.q.res_log2` — the coded delta is scaled by `1 << res_log2`.
    pub delta_q_res_log2: u8,
    /// frame `txfm_mode == TX_SWITCHABLE` (dav2d obu.c:1935): non-skip blocks <=64px code a
    /// TX-PARTITION symbol chain (dav decode.c:1378 read_tx_part) and split into TX units.
    /// LARGEST mode (all prior corpus streams) codes nothing — TX == block.
    pub tx_switchable: bool,
}
thread_local! {
    pub static HDR_TOOL_CFG: std::cell::Cell<HdrToolCfg> =
        std::cell::Cell::new(HdrToolCfg {
            gdf: true, cdef: true, ccso: true, allow_intrabc: true, cfl: true,
            cfl_ds_filter: 0, subpel_filter_mode: 4, mv_precision: 2, warp_motion: true,
            tip_frame_mode: 0, tip_subpel_filter: 2, tip_gmv: (0, 0), tip_apply_filter: false,
            skip_mode_enabled: false, bawp: true, allow_scc: false, force_integer_mv: 0,
            switchable_comp_refs: false,
            opfl_refine_type: 0, delta_q_present: false, delta_q_res_log2: 0,
            tx_switchable: false,
        });
    /// dav2d `ts->last_qidx`: the running per-SB qindex under delta-q. Reset to the frame's
    /// base yac at each frame/tile start; updated by the per-SB delta-q symbol; read by every
    /// dequant site in place of the frame yac.
    pub static LAST_QIDX: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Per-frame WORK BUDGET (hardening): every unbounded-ish decode/recon walker ticks this.
    /// A corrupt stream can otherwise spin a step-loop forever (fuzz HANG class). The budget is
    /// generous vs. real work (a 512x512 frame is ~16k 4px blocks; each does bounded per-block
    /// work), so no valid stream can reach it — it is a safety net, not a policy.
    pub static WORK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub static WORK_TRIPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// The current frame is an S_FRAME (frame_type==3). Consumed by update_ref_slots so the
    /// slot records is_sframe (excluded from primary/secondary CDF-source derivation).
    pub static CUR_IS_SFRAME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// SEQ-header tool enables that gate per-frame/per-block parse, captured at the seq parse
    /// (previously read-and-discarded scaffold; consumers hardcoded the mints' defaults —
    /// the 2026-08-06 tool-off mint battery proved each a real desync/recon bug).
    pub static SEQ_TOOLS: std::cell::Cell<SeqTools> = const { std::cell::Cell::new(SeqTools::DEFAULT) };
    /// seq `reduced_tx_part_set` (dav2d obu.c:559): drops the 4-way arm of the 1D tx-part read.
    pub static SEQ_REDUCED_TX_PART: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Per-ref-list-index distance arrays (dav2d decode.c:5501): `.0` = refdist[i] (signed wrapped
    /// poc delta ref−cur), `.1` = absrefdist[i], `.2` = furthest_future_refidx (-2 = none). Set per
    /// inter frame alongside CUR_REFDIR; feeds the compound joint_ctx / refine_mv / comp_type gates.
    pub static CUR_REFDIST: std::cell::Cell<([i32; 7], [i32; 7], i32)> =
        const { std::cell::Cell::new(([0; 7], [0; 7], -2)) };
    /// (order_hint_n_bits, frame poc, per-LIST-INDEX ref pocs) — feeds ref_flip_pair (t_swap).
    pub static CUR_REF_POC: std::cell::Cell<(u32, u32, [u32; 7])> = const { std::cell::Cell::new((0, 0, [0; 7])) };
    /// dav2d t->scratch.seg_mask emulation: a PERSISTENT per-tile scratch the SEG w_mask and
    /// bacp get_mask write into and the chroma blends read from — including dav's stride
    /// mismatches and stale bytes (recon 3288/3305/3760).
    pub static SEG_SCRATCH: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0u8; 128 * 128]);
    /// Seq-header compound flags: `.0`=masked_compound, `.1`=num_same_ref_comp, `.2`=cwp,
    /// `.3`=avg_cdf (pri/sec 7:1 CDF average gate), `.4`=mv_traj.
    pub static SEQ_COMP: std::cell::Cell<(bool, u8, bool, bool, bool)> = const { std::cell::Cell::new((false, 0, false, false, false)) };
    /// Seq-header TIP flags (dav2d seqhdr), needed by the FRAME tip block (obu.c:1246-1296) which
    /// mine's AV2 front-header must parse for `tip.frame_mode=1` frames (else the header desyncs).
    /// `.0`=tip enable, `.1`=tip_hole_fill, `.2`=opfl_refine(2b), `.3`=refine_mv, `.4`=tip_refine_mv.
    pub static SEQ_TIP: std::cell::Cell<(u8, bool, u8, bool, bool)> = const { std::cell::Cell::new((0, false, 0, false, false)) };
    /// seq `tip_explicit_qp` (dav2d obu.c:490).
    pub static SEQ_TIP_QP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Number of frames DECODED so far (incremented after each frame). For a P-chain this equals
    /// the implicit `n_ref_frames` for the next frame (frame 1→1 ref, frame 2→2 refs, …) — a
    /// stand-in for dav2d's implicit ref-buffer scoring (get_ref_frames) which mine defers.
    pub static FRAME_DECODE_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// The 8-slot reference buffer's per-slot metadata (dav2d `c->refs[8]`), for the implicit
    /// `get_ref_frames` scoring + `has_bothside_refs`. A slot is None until a frame refreshes it.
    pub static REF_SLOTS: std::cell::RefCell<[Option<RefSlot>; 8]> =
        const { std::cell::RefCell::new([None, None, None, None, None, None, None, None]) };
    /// The current frame's (order_hint, refresh_frame_flags, qidx/yac, width, height) — set during
    /// the header parse, applied to REF_SLOTS after the frame decodes (per refresh_frame_flags).
    pub static CUR_FRAME_REF: std::cell::Cell<(u32, u32, u16, u32, u32)> = const { std::cell::Cell::new((0, 0, 0, 0, 0)) };
    /// The current inter frame's (n_ref_frames, refidx[0..7]) — set during the header parse so the
    /// SB-loop can point the primary reference at REF_PICS[refidx[0]] (the correct primary ref slot).
    pub static CUR_FRAME_REFIDX: std::cell::Cell<(u32, [u8; 7])> = const { std::cell::Cell::new((0, [0; 7])) };
    /// The current frame's (show_immediate, show_implicit) header bits (dav2d obu.c:1118-1123).
    pub static AV2_SHOW: std::cell::Cell<(bool, bool)> = const { std::cell::Cell::new((true, false)) };
    /// dav2d `c->dpb_poc` — the poc of the most recently queued output picture.
    pub static AV2_DPB_POC: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// The current frame's primary_ref_frame (7 = PRIMARY_REF_NONE → default CDF). When != NONE the
    /// SB-loop inits its CDF from the stashed cdf of ref slot `refidx[primary_ref_frame]`.
    pub static CUR_PRIMARY_REF: std::cell::Cell<u8> = const { std::cell::Cell::new(7) };
    /// The current frame's tile grid (dav2d `hdr->tiling`): tile column/row starts in 4px units
    /// (entry `[cols]`/`[rows]` = the rounded-UP frame extent), the log2 counts, and the tile-size
    /// field width `n_bytes`. Default = one frame-sized tile.
    pub static TILE_INFO: std::cell::Cell<TileInfo> = const { std::cell::Cell::new(TileInfo::single()) };
    /// The tile bounds of the tile CURRENTLY being decoded, in 4px units:
    /// (col_start, col_end, row_start, row_end). Availability (`have_left`/`have_top`) is
    /// gated against the tile origin, not the frame origin (dav2d `ts->tiling`).
    pub static TILE_B: std::cell::Cell<(usize, usize, usize, usize)> =
        const { std::cell::Cell::new((0, 1 << 30, 0, 1 << 30)) };
    /// Per-frame TILE-ADAPTIVE filter-unit sizes in 4px units (avm ccso.c:25 + gdf.c
    /// init_gdf): `.0` = CCSO unit (64 = the 256px default; shrinks when non-last tile
    /// spans aren't 4/2-SB-divisible), `.1` = GDF block (32 = 128px; 16 when any span is
    /// odd-SB with 64px superblocks). Set at frame-header parse, read by reset_frame and
    /// the per-SB filter-symbol reads.
    pub static FILTER_UNITS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((64, 32)) };
    /// secondary_ref_frame (dav2d obu.c:1461): with primary != NONE, the 2nd-best derived ref.
    /// When != 7 (and seq avg_cdf && !avg_cdf_type && inter && tip.frame_mode != 2), the frame's
    /// CDF init is the 7:1 average of primary+secondary saved CDFs (decode.c:5401/5013).
    pub static CUR_SECONDARY_REF: std::cell::Cell<u8> = const { std::cell::Cell::new(7) };
    /// dav2d `refdir_with_intra`: index 0 = intra (ref -1), index 1+i = ref i's direction
    /// (1 = future / order_hint > current, 0 = past). Feeds `get_comp_ctx`. Set during parse.
    /// dav2d `refdir_with_intra` (internal.h:258, 1 intra + 7 refs + 1 tip): index 0 = the INTRA
    /// slot = -1 (lib.c:274, truthy in get_comp_ctx's `&& refdir[ref]` arms but NOT `== 1`);
    /// index 1+i = ref i's direction (1 = future); index 8 = the TIP slot = 1 (lib.c:275).
    /// i8 to preserve the -1 truthiness semantics.
    pub static CUR_REFDIR: std::cell::Cell<[i8; 9]> = const { std::cell::Cell::new([-1, 0, 0, 0, 0, 0, 0, 0, 1]) };
    /// ext-SDP recursion limit (dav2d decode.c:3883 `child_dir` bit 24): true when the PARENT
    /// partition has already split to a size where a child may NOT decode a `region_type`
    /// symbol. Threaded through the inter partition tree via save-at-entry / restore-at-exit
    /// (a returning sibling leaves it at the shared parent value). Reset false per SB.
    pub static EXT_SDP_LIMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Process-global pixel max for kernels called without a threaded bd param.
#[inline]
fn bdmax_g() -> i32 {
    crate::av2_frame::BDMAX.with(|c| c.get())
}

/// Process-global chroma subsampling `(ss_hor, ss_ver)` as usize (420=(1,1), 422=(1,0), 444=(0,0)).
#[inline]
fn ss_g() -> (usize, usize) {
    crate::av2_frame::SS.with(|c| {
        let s = c.get();
        (s.0 as usize, s.1 as usize)
    })
}

/// The frame's tile grid (dav2d `Dav2dTileInfo`, 4px-unit starts). `col_start4[cols]` /
/// `row_start4[rows]` hold the rounded-UP frame extent (may exceed the visible 4px dims —
/// clamp with `min(iw4/ih4)` when forming a tile's end bound).
#[derive(Clone, Copy)]
pub struct TileInfo {
    pub cols: u8,
    pub rows: u8,
    pub log2_cols: u8,
    pub log2_rows: u8,
    /// Tile-size field width in bytes (frame header `tiling.n_bytes`, 1..4).
    pub n_bytes: u8,
    pub col_start4: [u16; 17],
    pub row_start4: [u16; 17],
}

impl TileInfo {
    pub const fn single() -> Self {
        let mut col_start4 = [0u16; 17];
        let mut row_start4 = [0u16; 17];
        col_start4[1] = u16::MAX;
        row_start4[1] = u16::MAX;
        Self { cols: 1, rows: 1, log2_cols: 0, log2_rows: 0, n_bytes: 1, col_start4, row_start4 }
    }
}

/// Per-slot reference-buffer metadata for the implicit ref scoring (dav2d get_ref_frames).
#[derive(Clone, Copy)]
pub struct RefSlot {
    pub order_hint: u32,
    pub qidx: u16,
    pub width: u32,
    pub height: u32,
    /// dav2d `IS_KEY_OR_INTRA(refhdr)` — a key/intra ref is skipped as a primary-ref-cdf source
    /// (derive_pri_sec_ref, obu.c:929) and never contributes an adapted CDF worth inheriting.
    pub is_key_or_intra: bool,
    /// dav2d `hdr->show_implicit` — a not-immediately-shown frame that the POC-ordered output
    /// queue (dav2d lib.c `dav2d_queue_output`) may emit later from its ref slot.
    pub show_implicit: bool,
    /// avm `is_restricted` (restricted_prediction_switch s-frame): the slot is excluded from
    /// implicit ref scoring and its display order hint is void (epoch reset).
    pub restricted: bool,
    /// The slot holds an S_FRAME. avm choose_primary_secondary_ref_frame requires candidates
    /// with `frame_type == INTER_FRAME` (pred_common.c:507) — an s-frame slot is never the
    /// derived primary/secondary CDF source (the switch point severs entropy inheritance).
    pub is_sframe: bool,
}

/// dav2d `get_poc_diff` — the wrapped (mod 2^n_bits) signed order-hint difference `a - b`.
/// dav2d ref_flip bit (refmvs.c:2069): t_swap for a compound temporal store. sign[i] =
/// (ref_poc[i] older than cur); flip = same-sign ? poc_i < poc_j : sign_j.
pub fn ref_flip_pair(r0: i8, r1: i8) -> bool {
    let (nbits, poc, refpoc) = CUR_REF_POC.with(|c| c.get());
    let (p0, p1) = (refpoc[r0 as usize], refpoc[r1 as usize]);
    let s0 = get_poc_diff(nbits, p0, poc) < 0;
    let s1 = get_poc_diff(nbits, p1, poc) < 0;
    if s0 == s1 {
        get_poc_diff(nbits, p0, p1) < 0
    } else {
        s1
    }
}

pub fn get_poc_diff(n_bits: u32, a: u32, b: u32) -> i32 {
    if n_bits == 0 {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1i32 << (n_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// dav2d `get_ref_frames` (obu.c:776) — the IMPLICIT reference-frame scoring. Scores each of the
/// 8 ref slots by |pocdiff| (+ qidx + resolution ratio; mlayer/tlayer deps assumed absent here),
/// sorts by score, and fills `refidx[0..7]`. Returns `n_ref_frames` (≤7). This clip corpus is
/// single-layer + constant-resolution, so the layer-dependency + resolution-clamp branches are
/// no-ops; ported faithfully so a hierarchical GOP (v320/v432_8f) picks the same refs as dav2d.
/// Superblock step in 4px cells (16 = 64px SBs, 32 = 128px). Set per frame from the
/// sequence header's sb128; every SB-relative gate/grid keys off this.
pub fn sb_step4() -> usize {
    SB_STEP4.load(std::sync::atomic::Ordering::Relaxed)
}
/// PROCESS-global: the harness can run parse and recon on different pool threads.
pub static SB_STEP4: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(16);

pub fn get_ref_frames(n_bits: u32, poc: u32) -> (u32, [u8; 7]) {
    let slots = REF_SLOTS.with(|s| *s.borrow());
    // have_fwd_refs: any slot with a FUTURE order hint (pocdiff < 0).
    let mut have_fwd_refs = false;
    for n in 0..8 {
        if let Some(r) = slots[n] {
            if r.restricted {
                continue; // avm is_restricted: excluded from implicit ref scoring
            }
            if get_poc_diff(n_bits, poc, r.order_hint) < 0 {
                have_fwd_refs = true;
                break;
            }
        }
    }
    #[derive(Clone, Copy, Default)]
    struct Score { score: i32, poc: u32, pocdiff: i32, qidx: u16 }
    let mut ref_info = [Score::default(); 8];
    let mut sort_idx = [0u8; 8];
    let mut n_refs = 0usize;
    let mut last: Option<(u32, u16, u32, u32)> = None;
    let (mut minq, mut maxq) = (512i32, -1i32);
    for n in 0..8 {
        let refslot = match slots[n] {
            Some(r) => r,
            None => continue,
        };
        if refslot.restricted {
            continue; // avm is_restricted (s-frame epoch): never implicitly referenced
        }
        // Skip duplicate consecutive slot (dav `refhdr == last_refhdr`, pointer identity → the
        // keyframe splatted into all 8 slots reads as ONE distinct ref).
        if last == Some((refslot.order_hint, refslot.qidx, refslot.width, refslot.height)) {
            continue;
        }
        let poc_v = refslot.order_hint;
        let pocdiff = get_poc_diff(n_bits, poc, refslot.order_hint);
        let qidx = refslot.qidx;
        let res_ratio_log2 = -(ulog2_u32(refslot.width * refslot.height) as i32);
        let tdist = pocdiff.abs(); // + mlayer - r->mlayer (both 0 here)
        let mut score = if have_fwd_refs {
            tdist << 6
        } else {
            128 - (128 >> tdist.min(6)) + (tdist - 6).max(0)
        };
        score += res_ratio_log2 * (1 << 5) + qidx as i32;
        // Dedup by (score, poc): if an equal ref already sorted, skip.
        let mut m = 0usize;
        while m < n_refs {
            if !crate::av2_recon::work_tick("av2_recon:811") { break; }
            let r2 = &ref_info[sort_idx[m] as usize];
            if score == r2.score && poc_v == r2.poc {
                break;
            }
            m += 1;
        }
        if m < n_refs {
            continue;
        }
        {
            let r = &mut ref_info[n];
            r.poc = poc_v;
            r.pocdiff = pocdiff;
            r.qidx = qidx;
            r.score = score;
        }
        maxq = maxq.max(qidx as i32);
        minq = minq.min(qidx as i32);
        while m > 0 {
            if !crate::av2_recon::work_tick("av2_recon:830") { break; }
            let idx = sort_idx[m - 1] as usize;
            if ref_info[idx].score <= score {
                break;
            }
            sort_idx[m] = sort_idx[m - 1];
            m -= 1;
        }
        sort_idx[m] = n as u8;
        n_refs += 1;
        last = Some((refslot.order_hint, refslot.qidx, refslot.width, refslot.height));
    }
    // Full-buffer (n_refs==8) ALTREF reassignment (dav obu.c:853-880).
    if n_refs == 8 {
        let q_thr = (maxq + minq + 1) >> 1;
        let (mut maxpocdiff, mut num, mut furthest) = ([0i32; 2], [0i32; 2], [0usize; 2]);
        for n in 0..8 {
            let r = &ref_info[sort_idx[n] as usize];
            if (r.qidx as i32) < q_thr {
                continue;
            }
            if r.pocdiff > 0 {
                if r.pocdiff > maxpocdiff[0] { maxpocdiff[0] = r.pocdiff; furthest[0] = n; }
                num[0] += 1;
            } else if r.pocdiff < 0 {
                if r.pocdiff < maxpocdiff[1] { maxpocdiff[1] = r.pocdiff; furthest[1] = n; }
                num[1] += 1;
            }
        }
        let idx = if num[0] > num[1] { furthest[0] }
            else if num[0] < num[1] { furthest[1] }
            else { furthest[(maxpocdiff[0] < -maxpocdiff[1]) as usize] };
        if idx < 7 {
            let saved = sort_idx[idx];
            sort_idx.copy_within(idx + 1..8, idx);
            sort_idx[7] = saved;
        }
    }
    let mut refidx = [0u8; 7];
    for n in 0..7 {
        refidx[n] = sort_idx[if n < n_refs { n } else { 0 }];
    }
    if std::env::var("PREFDBG").is_ok() {
        let mut line = format!("[MREFL] poc={poc} n={} list:", n_refs.min(7));
        for n in 0..n_refs.min(7) {
            let sl = sort_idx[n] as usize;
            let oh = slots[sl].map(|r| r.order_hint as i32).unwrap_or(-1);
            let q = slots[sl].map(|r| r.qidx as i32).unwrap_or(-1);
            line += &format!(" [{n}]=oh{oh}(slot{sl},q{q},s{})", ref_info[sl].score);
        }
        crate::dlog!("{line}");
    }
    (n_refs.min(7) as u32, refidx)
}

fn ulog2_u32(v: u32) -> u32 {
    31 - v.max(1).leading_zeros()
}

/// Neighbour-context array length in 4px units, sized from the sequence's max frame dims
/// (was a fixed 128 = 512px scaffold cap). Set at the seq-header parse; the +32 margin covers
/// the off-frame spill an edge superblock's splats can address.
pub static NB_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);

#[inline]
pub fn nb_len() -> usize {
    NB_LEN.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_nb_len(iw4: usize, ih4: usize) {
    NB_LEN.store((iw4.max(ih4) + 32).max(128), std::sync::atomic::Ordering::Relaxed);
}

/// Capture-oracle file path (debug only, DAVCAP). Base dir from `DAVCAP_DIR`, else the
/// system temp dir — no developer-specific paths in source.
pub fn cap_path(name: &str) -> String {
    let dir = std::env::var("DAVCAP_DIR").unwrap_or_else(|_| {
        std::env::temp_dir().to_string_lossy().into_owned()
    });
    let d: &str = dir.trim_end_matches(|c: char| c == 0x2f as char || c == 0x5c as char);
    format!("{d}/{name}")
}

/// Master debug switch. ALL decoder tracing is behind this — a normal decode must be SILENT.
/// Set `RUSTY_AV2D_DEBUG=1` to re-enable the in-tree probes (they additionally honour their own
/// env vars, e.g. `RUSTY_AV2D_DEBUG=1 MSBT=1`). `RAV2D_DEBUG` is accepted as a legacy alias.
#[inline]
pub fn dbg_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("RUSTY_AV2D_DEBUG").is_ok() || std::env::var("RAV2D_DEBUG").is_ok()
    })
}

/// Debug print: compiles to a cheap boolean test when the switch is off.
#[macro_export]
macro_rules! dlog {
    ($($arg:tt)*) => { if $crate::av2_recon::dbg_on() { eprintln!($($arg)*); } };
}

/// Per-frame work budget: reset at frame start.
pub fn work_reset() {
    WORK.with(|c| c.set(0));
    WORK_TRIPPED.with(|c| c.set(false));
}

/// Tick the work budget. Returns false once the frame's budget is exhausted — callers must
/// then stop iterating (a corrupt stream is spinning). Logs once, on the site that tripped it.
#[inline]
pub fn work_tick(site: &str) -> bool {
    let n = WORK.with(|c| { let v = c.get() + 1; c.set(v); v });
    // ~64M ticks: orders of magnitude above any legal frame's walker iterations.
    // WORKBUDGET=<n> overrides it (diagnostics: set it low to prove the budget is live and
    // to name whichever walker a corrupt stream is spinning in).
    // The cap is read ONCE (an env lookup per tick was a severe slowdown — the tick is on
    // the hottest inner loops).
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| {
        std::env::var("WORKBUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(64_000_000)
    });
    if n <= cap {
        return true;
    }
    if !WORK_TRIPPED.with(|c| c.replace(true)) {
        crate::dlog!("[rav2d AV2] work budget exhausted at `{site}` — corrupt stream, aborting frame");
    }
    false
}

/// True once this frame's budget tripped (recursion/loops check to unwind fast).
#[inline]
pub fn work_dead() -> bool {
    WORK_TRIPPED.with(|c| c.get())
}

/// Reset ALL cross-frame AV2 decoder state (called on a NEW video sequence, so a prior
/// stream's ref slots / CDF stashes / motion fields / grain / caches cannot leak into the
/// next one — the thread-local equivalent of dav1d's per-sequence state free).
pub fn reset_av2_stream_state() {
    REF_SLOTS.with(|s| *s.borrow_mut() = std::array::from_fn(|_| None));
    CUR_IS_SFRAME.with(|c| c.set(false));
    HDR_TOOL_CFG.with(|c| c.set(HdrToolCfg {
        gdf: true, cdef: true, ccso: true, allow_intrabc: true, cfl: true,
        cfl_ds_filter: 0, subpel_filter_mode: 4, mv_precision: 2, warp_motion: true,
        tip_frame_mode: 0, tip_subpel_filter: 2, tip_gmv: (0, 0), tip_apply_filter: false,
        skip_mode_enabled: false, bawp: true, allow_scc: false, force_integer_mv: 0,
        switchable_comp_refs: false, opfl_refine_type: 0, delta_q_present: false,
        delta_q_res_log2: 0, tx_switchable: false,
    }));
    SEQ_TOOLS.with(|c| c.set(SeqTools::DEFAULT));
    SEQ_REDUCED_TX_PART.with(|c| c.set(false));
    crate::cdf_av2::reset_ref_cdf();
    crate::av2_frame::reset_stream_state();
    crate::av2_refmvs::reset_stream_state();
    crate::av2_grain::reset_stream_state();
    crate::av2_qm::set_frame_qm(false, 15, 15, 15);
}

/// Commit the just-decoded frame into the reference buffer: write its RefSlot into every slot
/// selected by refresh_frame_flags (dav2d: the keyframe refreshes 0xff → all 8 = this frame).
pub fn update_ref_slots(is_key_or_intra: bool) {
    let (order_hint, refresh, qidx, width, height) = CUR_FRAME_REF.with(|c| c.get());
    let show_implicit = AV2_SHOW.with(|c| c.get()).1;
    let is_sframe = CUR_IS_SFRAME.with(|c| c.get());
    let slot = RefSlot { order_hint, qidx, width, height, is_key_or_intra, show_implicit, restricted: false, is_sframe };
    REF_SLOTS.with(|s| {
        let mut slots = s.borrow_mut();
        for i in 0..8 {
            if refresh & (1 << i) != 0 {
                slots[i] = Some(slot);
            }
        }
    });
    // Commit the decoded pixels into the matching reference-picture slots (parallel buffer).
    crate::av2_frame::update_ref_pics(refresh);
    // Save this frame's CCSO per-SB map + resolved ccso config into the refreshed slots
    // (dav2d c->refs[].ccsomap / per-slot frame header) — consumed by later frames' reuse/sb_reuse.
    {
        let map = crate::av2_frame::FRAME.with(|f| f.borrow().ccso_blk.clone());
        let cfg = crate::av2_frame::CCSO_CFG.with(|c| c.borrow().clone());
        crate::av2_frame::CCSO_SLOT_MAP.with(|m| {
            let mut slots = m.borrow_mut();
            for i in 0..8 {
                if refresh & (1 << i) != 0 {
                    slots[i] = Some(map.clone());
                }
            }
        });
        crate::av2_frame::CCSO_SLOT_CFG.with(|c| {
            let mut slots = c.borrow_mut();
            for i in 0..8 {
                if refresh & (1 << i) != 0 {
                    slots[i] = Some(cfg.clone());
                }
            }
        });
        // Loop-restoration frame config per slot (dav refhdr->restoration): temporal filter
        // inheritance + the ref-filter bank collection read these.
        let lr = crate::av2_lr::LR_CFG.with(|c| c.borrow().clone());
        crate::av2_lr::LR_SLOT.with(|s| {
            let mut slots = s.borrow_mut();
            for i in 0..8 {
                if refresh & (1 << i) != 0 {
                    slots[i] = Some(lr.clone());
                }
            }
        });
    }
    // Commit the frame's temporal motion field (dav rp -> rp_ref[slot]; empty/INVALID for keys),
    // with the frame's own ref metadata for the mfmv projection setup.
    {
        let (_nb, poc, refpoc) = CUR_REF_POC.with(|c| c.get());
        let n_ref = CUR_FRAME_REFIDX.with(|c| c.get()).0;
        crate::av2_refmvs::rp_save(refresh, poc, refpoc, n_ref);
    }
}

/// dav2d `tip_frame_recon_sb` (decode.c:4424): synthesize one 64x64 SB of a TIP-as-output
/// (frame_mode==2) frame — a fabricated whole-SB TIP block (mv = tip.gmv, filter =
/// tip.subpel_filter, skip_txfm=1) through the normal TIP recon path. No symbols, no residual;
/// per-cell temporal writes happen inside tip_pred_luma.
pub fn tip_recon_sb(bx4: usize, by4: usize, iw4: usize, ih4: usize) {
    if std::env::var("TIPDBG").is_ok() {
        crate::dlog!("[MTIPSB] bx4={bx4} by4={by4}");
    }
    let cfg = HDR_TOOL_CFG.with(|c| c.get());
    let fmv = crate::av2_refmvs::Mv { y: cfg.tip_gmv.0, x: cfg.tip_gmv.1 };
    let filter = cfg.tip_subpel_filter;
    let (bw4, bh4) = (sb_step4(), sb_step4());
    let (w4c, h4c) = (bw4.min(iw4.saturating_sub(bx4)), bh4.min(ih4.saturating_sub(by4)));
    let (w, h) = (bw4 * 4, bh4 * 4);
    let mut luma_pred = vec![0i32; w * h];
    let _ = tip_pred_luma(&mut luma_pred, bx4, by4, bw4, bh4, w4c, h4c, fmv, filter, iw4, ih4);
    crate::av2_frame::FRAME.with(|fr| {
        let mut f = fr.borrow_mut();
        if f.pl[0].w == 0 {
            return;
        }
        f.ensure_sb(bx4, by4);
        f.mark_coded_avail(bx4, by4, bw4, bh4);
        let stride = f.pl[0].stride;
        let wc = w.min(f.pl[0].w.saturating_sub(bx4 * 4));
        let hc = h.min(f.pl[0].h.saturating_sub(by4 * 4));
        for yy in 0..hc {
            let dst = (by4 * 4 + yy) * stride + bx4 * 4;
            for xx in 0..wc {
                f.pl[0].px[dst + xx] = luma_pred[yy * w + xx].clamp(0, bdmax_g());
            }
        }
        crate::av2_frame::write_recon_pad(0, bx4 * 4, by4 * 4, &luma_pred, w, h);
        mscore_luma("inter", bx4 * 4, by4 * 4, w, h, &luma_pred, w);
        f.mark_coded(bx4, by4, bw4, bh4, 0);
    });
    // Chroma: the TIP dual-MC per-cell path (the recon_inter_chroma tip arm; step per the
    // dav fm2 arm — 2<<(!opfl)).
    let (ssh, ssv) = ss_g();
    let (cpx, cpy) = ((bx4 * 4) >> ssh, (by4 * 4) >> ssv);
    let (cw, ch) = ((bw4 * 4) >> ssh, (bh4 * 4) >> ssv);
    let seq_tip_refine = SEQ_TIP.with(|c| c.get()).4;
    let opfl = seq_tip_refine && (cfg.tip_frame_mode == 1 || cfg.tip_subpel_filter == 2);
    let step = 2usize << (!opfl as usize);
    let tip_arg = Some((step, step, true, iw4, ih4));
    recon_inter_chroma(bx4, by4, cpx, cpy, cw, ch, fmv, None, filter, false, None, None, None, tip_arg, -1);
}

/// dav2d `derive_pri_sec_ref` (obu.c:929): when primary_ref_frame is NOT signaled, it is derived
/// as the best (lowest qdiff, then pocdiff, then poc) NON-key/intra reference. Returns
/// (primary, secondary) ref INDICES into refidx[0..n_ref] (7 = PRIMARY_REF_NONE). The secondary
/// drives the 7:1 pri/sec CDF average (decode.c:5401/5013).
pub fn derive_pri_sec_ref(n_bits: u32, poc: u32, qidx: u16, n_ref: u32, refidx: &[u8; 7]) -> (u8, u8) {
    // Two-slot running best, mirroring dav's refs[2] + best toggle. We track only refs[0].
    let (mut refs, mut best) = ([255u8; 2], 0usize);
    let (mut best_qdiff, mut best_pocdiff, mut best_poc) = ([0i32; 2], [0i32; 2], [0u32; 2]);
    REF_SLOTS.with(|s| {
        let slots = s.borrow();
        for i in 0..n_ref as usize {
            let rs = match slots[refidx[i] as usize] {
                // avm pred_common.c:505-509: candidate must be a non-restricted INTER frame —
                // key/intra AND s-frame slots are excluded (the switch severs CDF inheritance).
                Some(r) if !r.is_key_or_intra && !r.is_sframe && !r.restricted => r,
                _ => continue,
            };
            let qdiff = (rs.qidx as i32 - qidx as i32).abs();
            let ref_poc = rs.order_hint;
            let pocdiff = get_poc_diff(n_bits, poc, ref_poc).abs();
            let mut m = best;
            for n in 0..2 {
                let take = refs[m] == 255
                    || qdiff < best_qdiff[m]
                    || (qdiff == best_qdiff[m]
                        && (pocdiff < best_pocdiff[m]
                            || (pocdiff == best_pocdiff[m]
                                && get_poc_diff(n_bits, best_poc[m], ref_poc) < 0)));
                if take {
                    refs[best ^ 1] = i as u8;
                    best_pocdiff[best ^ 1] = pocdiff;
                    best_qdiff[best ^ 1] = qdiff;
                    best_poc[best ^ 1] = ref_poc;
                    if n == 0 {
                        best ^= 1;
                    }
                    break;
                }
                m ^= 1;
            }
        }
    });
    if best != 0 {
        refs.swap(0, 1);
    }
    (
        if refs[0] == 255 { 7 } else { refs[0] },
        if refs[1] == 255 { 7 } else { refs[1] },
    )
}

/// dav2d decode.c:3870 `lim[bp]` — the child split-factor limits (in 4px units) per partition
/// type. A child is ext-SDP-limited when `parent_bw4 <= lim[0] || parent_bh4 <= lim[1]`.
pub fn ext_sdp_child_limited(bp: crate::av2_decode::BlockPartition, bw4: usize, bh4: usize) -> bool {
    use crate::av2_decode::BlockPartition::*;
    let (lw, lh) = match bp {
        None => (1, 1),
        H => (1, 2),
        V => (2, 1),
        H3 => (2, 4),
        V3 => (4, 2),
        H4a | H4b => (1, 8),
        V4a | V4b => (8, 1),
        Split => (2, 2),
        Invalid => (1, 1),
    };
    bw4 <= lw || bh4 <= lh
}

/// Per-column (above) or per-row (left) neighbour context arrays for the intra-luma
/// decode. Indexed in 4-pixel units across the tile/SB. `midx` defaults to `0xff`
/// ("no directional mode"); the rest default to 0.
#[derive(Clone)]
pub struct BlockNbCtx {
    pub partition: Vec<u8>,
    pub intrabc: Vec<u8>,
    /// intrabc `morph_pred` flag (1 only for intrabc blocks with morph=1), base 0. Feeds the
    /// SCC morph_pred context (avm get_morph_pred_ctx: count of intrabc-with-morph neighbours).
    pub morph: Vec<u8>,
    pub midx: Vec<u8>,
    pub fsc: Vec<u8>,
    pub mrl: Vec<u8>,
    pub multi_mrl: Vec<u8>,
    /// txfm-skip flag per column/row (dav2d `skip_txfm`), base 0. Drives the intrabc/
    /// inter skip context (intra blocks store 0).
    pub skip_txfm: Vec<u8>,
    /// Coefficient cumulative level per column/row (dav2d `lcoef`/`ccoef`), base 0.
    pub lcoef: Vec<u8>,
    // --- inter neighbour state (frame 2+), drives the inter mode/MV contexts ---
    /// `intra` flag (1=intra, 0=inter), base 1 (the off-tile/keyframe default). Feeds
    /// the is_inter (`get_intra_ctx`) context — counts intra neighbours.
    pub intra: Vec<u8>,
    /// motion_mode per col/row (dav2d MotionMode: 0=TRANSLATION..3=WARP_DELTA..), base 0.
    /// Feeds get_warp_ctx + the warp_extend/causal (has_cs_ext) ext/cs contexts.
    pub motion_mode: Vec<u8>,
    /// primary ref index (+1 so 0 = "no/unavailable"), base 0. Feeds has_cs_ext (ref match)
    /// + the DRL single-ref context.
    pub ref0: Vec<u8>,
    /// second ref index (dav2d `ref[1]`), raw (-1 = single-ref, base). A compound/skip_mode block
    /// stores a non-negative marker here. Feeds get_comp_ctx (is_comp).
    pub ref1: Vec<i8>,
    /// mv precision per col/row, base 0. Feeds the mvprec ctx1/ctx2.
    pub mvprec: Vec<u8>,
    /// inter_mode per col/row (dav2d `mode`, e.g. NEWMV=15..WARPNEWMV=17..), base 0. Feeds the
    /// single/compound-ref DRL contexts (the NEWMV*_MODE_MASK newmv-family check).
    pub mode: Vec<u8>,
    /// Subpel filter per col/row (dav2d `filter`), base 0 (REGULAR). Feeds get_filter_ctx.
    pub filter: Vec<u8>,
    /// Adaptive-MVD flag per col/row (dav2d `amvd`), base 0. Feeds the NEWMV amvd context.
    pub amvd: Vec<u8>,
    /// skip_mode flag per col/row (dav2d `skip_mode`), base 0. Feeds the skip_mode context (sum of
    /// the two nx neighbours) and the skip_txfm context (skip_mode*3).
    pub skip_mode: Vec<u8>,
    /// compound type per col/row (dav2d `comp_type`: 0=NONE, 1=AVG, 2=WEDGE, 3=SEG), base 0.
    /// Feeds the comp_type_masked context (neighbour comp_type > AVG when compound).
    pub comp_type: Vec<u8>,
}

impl BlockNbCtx {
    pub fn new(n: usize) -> Self {
        Self {
            partition: vec![0; n],
            intrabc: vec![0; n],
            morph: vec![0; n],
            midx: vec![0xff; n],
            fsc: vec![0; n],
            mrl: vec![0; n],
            multi_mrl: vec![0; n],
            skip_txfm: vec![0; n],
            // dav2d base: bit-6 set (0x40) → `>>6 == 1` ("no DC sign") for cleared cols.
            lcoef: vec![0x40; n],
            intra: vec![1; n],
            motion_mode: vec![0; n],
            ref0: vec![0; n],
            ref1: vec![-1; n],
            mvprec: vec![0; n],
            mode: vec![0; n],
            filter: vec![0; n],
            amvd: vec![0; n],
            skip_mode: vec![0; n],
            comp_type: vec![0; n],
        }
    }
}

/// Reverse of `BLOCK_DIMENSIONS`: the bs index whose luma dims are `(w4, h4)` 4px units.
pub fn bs_from_dims(w4: usize, h4: usize) -> usize {
    for (i, d) in crate::av2_decode::BLOCK_DIMENSIONS.iter().enumerate() {
        if d[0] as usize == w4 && d[1] as usize == h4 {
            return i;
        }
    }
    0
}

/// dav2d `size_group_lookup` (decode.c:1344), keyed by the block's (bw4, bh4). Drives the
/// warp-interintra / interintra size context.
pub fn size_group(bw4: usize, bh4: usize) -> usize {
    match (bw4.min(bh4), bw4.max(bh4)) {
        (1, 1) | (1, 2) | (1, 4) => 0,
        (1, 8) | (2, 2) | (2, 4) | (2, 8) => 1,
        (1, 16) | (2, 16) | (4, 4) | (4, 8) | (4, 16) => 2,
        _ => 3,
    }
}

/// dav2d `get_filter_ctx` (env.h:120): subpel-filter context from the two nb/boff neighbours'
/// filter values (using `N_SWITCHABLE_FILTERS` = 3 when the neighbour is unavailable or its ref
/// doesn't match). Single-ref only (comp=0). Returns 0..=7.
pub fn get_filter_ctx(a: &BlockNbCtx, l: &BlockNbCtx, nb: [(bool, i32); 2], ref_: u8, comp: bool) -> usize {
    const N: u8 = 3; // N_SWITCHABLE_FILTERS
    let flt = |sel: (bool, i32)| -> u8 {
        if sel.1 < 0 { return N; }
        let (d, o) = (if sel.0 { l } else { a }, sel.1 as usize);
        // dav2d env.h:124 — match on ref[0]==ref OR ref[1]==ref (a compound neighbour matches on
        // either slot). ref0 stored +1 (0=unavailable); ref1 raw (-1=none).
        let m0 = d.ref0[o] != 0 && d.ref0[o] - 1 == ref_;
        let m1 = d.ref1[o] == ref_ as i8;
        if m0 || m1 { d.filter[o] } else { N }
    };
    let flt0 = flt(nb[0]);
    let flt1 = flt(nb[1]);
    // A compound block (ref[1] != -1) offsets the ctx by 4 (env.h:123 `comp*4 + …`).
    let base = if comp { 4 } else { 0 };
    base + if flt0 == flt1 || flt1 == N {
        flt0 as usize
    } else if flt0 == N {
        flt1 as usize
    } else {
        // Both neighbours carry a valid but DIFFERENT filter → the "mixed" ctx is
        // N_SWITCHABLE_FILTERS (dav2d env.h:136 `comp*4 + N`), NOT N+1.
        N as usize
    }
}

/// dav2d `read_wedge_idx` (decode.c:1477): wedge quad + angle + distance → wedge index.
pub fn read_wedge_idx(msac: &mut crate::msac::MsacContext, m: &mut crate::cdf_av2::CdfModeContext) -> i32 {
    use crate::msac::{rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8};
    const WEDGE_ANGLE_DIST2IDX: [[i8; 4]; 20] = [
        [-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18],
        [-1, 19, 20, 21], [22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33], [34, 35, 36, 37],
        [-1, 38, 39, 40], [-1, 41, 42, 43], [-1, 44, 45, 46], [-1, 47, 48, 49], [-1, 50, 51, 52],
        [-1, 53, 54, 55], [-1, 56, 57, 58], [-1, 59, 60, 61], [-1, 62, 63, 64], [-1, 65, 66, 67],
    ];
    let quad = rav1d_msac_decode_symbol_adapt4(msac, &mut m.wedge_quad, 3) as usize;
    let angle = 5 * quad + rav1d_msac_decode_symbol_adapt8(msac, &mut m.wedge_angle[quad], 4) as usize;
    let dist = if (angle as u32).wrapping_sub(1) >= 9 || angle == 5 {
        1 + rav1d_msac_decode_symbol_adapt4(msac, &mut m.wedge_dist2, 2) as usize
    } else {
        rav1d_msac_decode_symbol_adapt4(msac, &mut m.wedge_dist, 3) as usize
    };
    WEDGE_ANGLE_DIST2IDX[angle][dist] as i32
}

/// Splat an inter block's neighbour state into the above (`a`) row + left (`l`) column,
/// across its width/height (dav2d `set_ctx`). `ref0` is the raw ref index (stored +1 so 0
/// means "unavailable"). Mirrors splat_partition but for the inter mode/MV context fields.
pub fn splat_inter_nb(
    a: &mut BlockNbCtx, l: &mut BlockNbCtx, bx4: usize, by4: usize, bw4: usize, bh4: usize,
    intra: u8, motion_mode: u8, ref0: u8, mvprec: u8, mode: u8, skip_mode: u8, ref1: i8,
) {
    for x in bx4..bx4 + bw4 {
        a.intra[x] = intra;
        a.motion_mode[x] = motion_mode;
        a.ref0[x] = ref0;
        a.ref1[x] = ref1;
        a.mvprec[x] = mvprec;
        a.mode[x] = mode;
        a.skip_mode[x] = skip_mode;
    }
    for y in by4..by4 + bh4 {
        l.intra[y] = intra;
        l.motion_mode[y] = motion_mode;
        l.ref0[y] = ref0;
        l.ref1[y] = ref1;
        l.mvprec[y] = mvprec;
        l.mode[y] = mode;
        l.skip_mode[y] = skip_mode;
    }
}

/// dav2d `get_snglref_ctx` (env.h:207): single-ref DRL context. Counts ref-matching top (`col`)
/// and left (`row`) neighbours and whether any is a NEWMV-family mode → `!!row + !!col + 2*!!newmv`.
pub fn get_snglref_ctx(
    a: &BlockNbCtx, l: &BlockNbCtx, yb4: usize, xb4: usize,
    have_top: bool, have_left: bool, have_top_right: bool, have_bottom_left: bool,
    bw4: usize, bh4: usize, ref_: u8,
) -> usize {
    // NEWMV0_MODE_MASK: NEWMV(15), NEWMV_NEARMV(20), NEWMV_NEWMV(22), JOINT_NEWMV(23),
    // OPFL_NEWMV_NEARMV(26), OPFL_NEWMV_NEWMV(27), OPFL_JOINT_NEWMV(28). (WARPNEWMV=17 NOT in it.)
    const MASK0: u32 = (1 << 15) | (1 << 20) | (1 << 22) | (1 << 23) | (1 << 26) | (1 << 27) | (1 << 28);
    // NEWMV1_MODE_MASK (env.h:226): ref[1]-side newmv modes — NEARMV_NEWMV(19), NEWMV_NEWMV(22),
    // OPFL_NEARMV_NEWMV(25), OPFL_NEWMV_NEWMV(27). Joint modes deliberately absent.
    const MASK1: u32 = (1 << 19) | (1 << 22) | (1 << 25) | (1 << 27);
    // dav2d add_matching (env.h:230): a compound neighbour matches on ref[0] (NEWMV0 mask) ELSE
    // on ref[1] (NEWMV1 mask). Returns (matched, newmv-hit).
    let add_matching = |d: &BlockNbCtx, i: usize| -> (bool, bool) {
        if d.ref0[i] != 0 && d.ref0[i] - 1 == ref_ {
            (true, ((1u32 << d.mode[i]) & MASK0) != 0)
        } else if d.ref1[i] == ref_ as i8 {
            (true, ((1u32 << d.mode[i]) & MASK1) != 0)
        } else {
            (false, false)
        }
    };
    let (mut row, mut col, mut newmv) = (0i32, 0i32, 0i32);
    if have_top {
        let (m, n) = add_matching(a, xb4);
        col += m as i32; newmv += n as i32;
        if have_top_right {
            let (m, n) = add_matching(a, xb4 + bw4 - 1);
            col += m as i32; newmv += n as i32;
        }
    }
    if have_left {
        let (m, n) = add_matching(l, yb4);
        row += m as i32; newmv += n as i32;
        if have_bottom_left {
            let (m, n) = add_matching(l, yb4 + bh4 - 1);
            row += m as i32; newmv += n as i32;
        }
    }
    (row != 0) as usize + (col != 0) as usize + 2 * (newmv != 0) as usize
}

/// dav2d `get_compref_ctx` (env.h:256): compound-DRL context. Counts top (`col`) / left (`row`)
/// neighbours whose FULL ref pair matches the block's, and whether any is a NEWMV-family mode →
/// `!!row + !!col + 2*!!newmv`. (The TIP-neighbour arm — neighbour ref[0]==TIP_FRAME matching the
/// frame's tip ref pair — is omitted: no TIP block is tracked yet.)
pub fn get_compref_ctx(
    a: &BlockNbCtx, l: &BlockNbCtx, yb4: usize, xb4: usize,
    have_top: bool, have_left: bool, have_top_right: bool, have_bottom_left: bool,
    bw4: usize, bh4: usize, ref0: u8, ref1: i8,
    tip_pair: (i8, i8),
) -> usize {
    // NEWMV_MODE_MASK (env.h:268): NEWMV(15) + the compound newmv-family modes.
    const MASK: u32 = (1 << 15) | (1 << 19) | (1 << 20) | (1 << 22) | (1 << 23)
        | (1 << 25) | (1 << 26) | (1 << 27) | (1 << 28);
    // (matched, is_newmv): a TIP neighbour (stored ref0 = 8) matches when the FRAME's tip
    // ref pair equals the block's pair, and its newmv test is plain NEWMV only (env.h:278).
    let probe = |d: &BlockNbCtx, i: usize| -> (bool, bool) {
        if d.ref0[i] == 8 {
            (tip_pair.0 == ref0 as i8 && tip_pair.1 == ref1, d.mode[i] == 15)
        } else if d.ref0[i] != 0 && d.ref0[i] - 1 == ref0 && d.ref1[i] == ref1 {
            (true, ((1u32 << d.mode[i]) & MASK) != 0)
        } else {
            (false, false)
        }
    };
    let (mut row, mut col, mut newmv) = (0i32, 0i32, 0i32);
    if have_top {
        let (m, n) = probe(a, xb4);
        if m { col += 1; newmv += n as i32; }
        if have_top_right {
            let (m, n) = probe(a, xb4 + bw4 - 1);
            if m { col += 1; newmv += n as i32; }
        }
    }
    if have_left {
        let (m, n) = probe(l, yb4);
        if m { row += 1; newmv += n as i32; }
        if have_bottom_left {
            let (m, n) = probe(l, yb4 + bh4 - 1);
            if m { row += 1; newmv += n as i32; }
        }
    }
    (row != 0) as usize + (col != 0) as usize + 2 * (newmv != 0) as usize
}

/// dav2d `dav2d_mv_projection` (refmvs.c:431): scale an MV by num/den (poc-distance ratio).
pub fn mv_projection(mv: (i32, i32), num: i32, den: i32, min: i32, max: i32) -> (i32, i32) {
    const DIV_MULT: [i32; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340,
        2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
        1024, 963, 910, 862, 819, 780, 744, 712,
        682, 655, 630, 606, 585, 564, 546, 528,
    ];
    debug_assert!(den > 0 && den < 32 && num > -32 && num < 32);
    let frac = num * DIV_MULT[den as usize];
    let y = mv.0 * frac;
    let x = mv.1 * frac;
    (
        ((y + 8192 + (y >> 31)) >> 14).clamp(min, max),
        ((x + 8192 + (x >> 31)) >> 14).clamp(min, max),
    )
}

/// dav2d `get_warp_ctx` (env.h:171). Counts top/left neighbours that both match the block's
/// `ref` AND have motion_mode >= MM_WARP_CAUSAL(2). (The SB-boundary `a_sb_cache` path — used
/// only when the top neighbour crosses an SB row — is omitted here; add when a block needs it.)
pub fn get_warp_ctx(
    a: &BlockNbCtx, l: &BlockNbCtx, yb4: usize, xb4: usize,
    have_top: bool, have_left: bool, have_top_right: bool, have_bottom_left: bool,
    bw4: usize, bh4: usize, ref_: u8,
    // At a SB-row boundary (`by & (sb_step-1) == 0` with a valid top), the top neighbour is read
    // from the committed row-above edge at 8px (even) granularity, and the top-right is only
    // consulted for blocks ≥16px wide (dav2d env.h:186-195 + decode.c:3041-3047). `col_end` is
    // the tile's 4px column end for the boundary top-right availability check.
    is_sb_boundary: bool, col_end: usize,
) -> usize {
    let m = |dir: &BlockNbCtx, idx: usize| -> usize {
        // env.h:183 — ref match on EITHER slot (ref0 stored +1; ref1 raw compound second ref).
        (((dir.ref0[idx] != 0 && dir.ref0[idx] - 1 == ref_) || dir.ref1[idx] == ref_ as i8)
            && dir.motion_mode[idx] >= 2) as usize
    };
    let mut ctx = 0;
    if have_top {
        if is_sb_boundary {
            ctx += m(a, xb4 & !1);
            if bw4 >= 4 && ((xb4 + bw4 - 2) & !1) < col_end {
                ctx += m(a, (xb4 + bw4 - 2) & !1);
            }
        } else {
            ctx += m(a, xb4);
            if have_top_right { ctx += m(a, xb4 + bw4 - 1); }
        }
    }
    if have_left {
        ctx += m(l, yb4);
        if have_bottom_left { ctx += m(l, yb4 + bh4 - 1); }
    }
    ctx
}

/// dav2d nb/boff spatial-neighbour setup (decode.c:1696-1730): fills two neighbour slots in
/// priority order (bottom-left, above-right, top-left, above-left) for the warp-extend/causal +
/// mv-precision contexts. Returns [(is_left, off); 2]; off = -1 marks an unavailable slot.
/// `w4`/`h4` are the frame-clamped block dims (bw4==w4 / bh4==h4 ⇒ not clipped at the edge).
pub fn nb_setup(
    have_left: bool, have_top_in_sb: bool,
    bx4: usize, by4: usize, bw4: usize, bh4: usize, w4: usize, h4: usize,
) -> [(bool, i32); 2] {
    let mut nb = [(true, -1i32), (true, -1i32)];
    let mut idx = 0usize;
    if have_left && bh4 == h4 { nb[0] = (true, (by4 + bh4 - 1) as i32); idx += 1; }
    if have_top_in_sb && bw4 == w4 { nb[idx] = (false, (bx4 + bw4 - 1) as i32); idx += 1; }
    if have_left && idx < 2 { nb[idx] = (true, by4 as i32); idx += 1; }
    if have_top_in_sb && idx < 2 { nb[idx] = (false, bx4 as i32); idx += 1; }
    nb
}

#[inline]
fn nb_pick<'a>(a: &'a BlockNbCtx, l: &'a BlockNbCtx, sel: (bool, i32)) -> Option<(&'a BlockNbCtx, usize)> {
    if sel.1 < 0 { None } else { Some((if sel.0 { l } else { a }, sel.1 as usize)) }
}
/// Neighbour motion_mode at a slot (MM_TRANSLATION=0 if unavailable).
pub fn nb_motion_mode(a: &BlockNbCtx, l: &BlockNbCtx, sel: (bool, i32)) -> u8 {
    nb_pick(a, l, sel).map_or(0, |(d, o)| d.motion_mode[o])
}
/// Neighbour mvprec at a slot (0 if unavailable).
pub fn nb_mvprec(a: &BlockNbCtx, l: &BlockNbCtx, sel: (bool, i32)) -> u8 {
    nb_pick(a, l, sel).map_or(0, |(d, o)| d.mvprec[o])
}

/// dav2d `match_ref`: does the neighbour at (dir, idx) reference the block's primary ref on
/// EITHER of its ref slots? (ref0 stored +1, 0 = unavailable; ref1 raw, -1 = single.)
#[inline]
fn match_ref(dir: &BlockNbCtx, idx: usize, ref_: u8) -> bool {
    (dir.ref0[idx] != 0 && dir.ref0[idx] - 1 == ref_) || dir.ref1[idx] == ref_ as i8
}

/// dav2d `has_cs_ext` (decode.c:3092): any of {left@by4, left@bottom, above@bx4, above@right}
/// references the same ref. Cleared neighbours (ref0=0) never match, so tile edges are safe.
/// Non-SB-boundary case only (the SB-boundary above uses a_sb_cache — add when a block needs it).
pub fn has_cs_ext(
    a: &BlockNbCtx, l: &BlockNbCtx, bx4: usize, by4: usize, bw4: usize, bh4: usize,
    row_end: usize, col_end: usize, ref_: u8,
) -> bool {
    match_ref(l, by4, ref_)
        || (by4 + bh4 <= row_end && match_ref(l, by4 + bh4 - 1, ref_))
        || match_ref(a, bx4, ref_)
        || (bx4 + bw4 <= col_end && match_ref(a, bx4 + bw4 - 1, ref_))
}

/// dav2d `nx`/`xoff` setup (decode.c:1643-1660) for the is_inter + skip contexts — a DIFFERENT
/// priority order than nb/boff (uses `have_top`, not have_top_in_sb, and duplicates the above slot
/// when nothing else fills it). Returns ([(is_left, off); 2], n_ctx).
pub fn nx_setup(
    have_left: bool, have_top: bool, bx4: usize, by4: usize, bw4: usize, bh4: usize,
    row_end: usize, col_end: usize,
) -> ([(bool, i32); 2], usize) {
    let mut nx = [(false, -1i32), (false, -1i32)];
    let mut idx = 0usize;
    if have_left && by4 + bh4 <= row_end { nx[0] = (true, (by4 + bh4 - 1) as i32); idx += 1; }
    if have_top && bx4 + bw4 <= col_end { nx[idx] = (false, (bx4 + bw4 - 1) as i32); idx += 1; }
    if idx < 2 && have_left { nx[idx] = (true, by4 as i32); idx += 1; }
    if idx < 2 {
        nx[idx] = (false, bx4 as i32);
        if idx == 0 { nx[1] = (false, bx4 as i32); }
        idx += have_top as usize;
    }
    (nx, idx)
}

/// dav2d `get_comp_ctx` (env.h:140): the `is_comp` context. `refdir_with_intra[stored]` is the
/// direction (0=past/1=future) of a neighbour's ref, where `stored` = raw ref+1 (0=intra, so index
/// 0 = the intra slot). Single-reference frame ⇒ every neighbour's ref[1] is -1, so only the two
/// single-ref sub-cases fire (both reduce to a refdir XOR / a single refdir bit).
pub fn get_comp_ctx(a: &BlockNbCtx, l: &BlockNbCtx, nx: [(bool, i32); 2], n_ctx: usize, refdir_with_intra: &[i8; 9]) -> usize {
    // Dispatch on the TRUE availability count `n_ctx` (dav's idx) — the nx fallback slots hold
    // valid-looking offsets even when nothing is available (decode.c:1650), so off>=0 lies.
    // dir(ref0_stored) = refdir_with_intra[stored] where stored = raw ref+1 (0 = intra slot = -1).
    // dav's single-neighbour arms test TRUTHINESS (`&& refdir[ref]` — intra's -1 counts as true);
    // only the both-single XOR arm compares `== 1`.
    let dir = |d: &BlockNbCtx, off: i32| -> i8 { refdir_with_intra[d.ref0[off as usize] as usize] };
    let ref1 = |d: &BlockNbCtx, off: i32| -> i8 { d.ref1[off as usize] };
    let intrabc = |d: &BlockNbCtx, off: i32| -> u8 { d.intrabc[off as usize] };
    match n_ctx {
        0 => 1,
        1 => {
            let (sel, off) = nx[0];
            let d = if sel { l } else { a };
            if ref1(d, off) == -1 {
                ((intrabc(d, off) == 0) && dir(d, off) != 0) as usize
            } else {
                3
            }
        }
        _ => {
            let (sa, oa) = nx[0];
            let (sb, ob) = nx[1];
            let (da, db) = (if sa { l } else { a }, if sb { l } else { a });
            let refa2 = ref1(da, oa);
            let refb2 = ref1(db, ob);
            if refa2 == -1 {
                if refb2 == -1 {
                    ((dir(da, oa) == 1) ^ (dir(db, ob) == 1)) as usize
                } else {
                    2 + ((intrabc(da, oa) == 0) && dir(da, oa) != 0) as usize
                }
            } else if refb2 == -1 {
                2 + ((intrabc(db, ob) == 0) && dir(db, ob) != 0) as usize
            } else {
                4
            }
        }
    }
}

/// dav2d `get_intra_ctx` (env.h): is_inter/is_intra context — counts non-intrabc intra neighbours.
pub fn get_intra_ctx(a: &BlockNbCtx, l: &BlockNbCtx, nx: [(bool, i32); 2], n_ctx: usize) -> usize {
    if n_ctx == 0 { return 0; }
    let rd = |sel: (bool, i32)| -> usize {
        if sel.1 < 0 { return 0; }
        let (d, o) = (if sel.0 { l } else { a }, sel.1 as usize);
        (d.intra[o] != 0 && d.intrabc[o] == 0) as usize
    };
    let sum = rd(nx[0]) + rd(nx[n_ctx - 1]);
    sum + (sum == n_ctx) as usize
}

/// skip_txfm context (decode.c:1751): sum of the two nx neighbours' skip_txfm + skip_mode*3.
pub fn get_skip_txfm_ctx(a: &BlockNbCtx, l: &BlockNbCtx, nx: [(bool, i32); 2], skip_mode: u8) -> usize {
    let rd = |sel: (bool, i32)| -> usize {
        if sel.1 < 0 { return 0; }
        (if sel.0 { l } else { a }).skip_txfm[sel.1 as usize] as usize
    };
    rd(nx[0]) + rd(nx[1]) + skip_mode as usize * 3
}

/// skip_mode context (decode.c:1664): sum of the two nx neighbours' skip_mode.
pub fn get_skip_mode_ctx(a: &BlockNbCtx, l: &BlockNbCtx, nx: [(bool, i32); 2]) -> usize {
    let rd = |sel: (bool, i32)| -> usize {
        if sel.1 < 0 { return 0; }
        (if sel.0 { l } else { a }).skip_mode[sel.1 as usize] as usize
    };
    rd(nx[0]) + rd(nx[1])
}

/// Live above/left chroma coefficient-context (`ccoef`) arrays for the SDP chroma tree,
/// one pair per chroma plane (`[0] = U`, `[1] = V`), indexed in **chroma** 4-pixel units.
/// Base `0x40` (the "no coefficients / no DC sign" cleared state); each decoded chroma TX
/// splats its `cf_ctx` so the next block's `skip_ctx_chroma` sees the right state.
pub struct ChromaNb {
    pub a: [Vec<u8>; 2],
    pub l: [Vec<u8>; 2],
    /// Above/left chroma prediction mode per chroma-4px unit (dav2d `a->uvmode`/`l.uvmode`),
    /// for the CfL flag's context (`+1` per neighbour that is `CFL_PRED`). Base `0` (DC).
    pub a_uvmode: Vec<u8>,
    pub l_uvmode: Vec<u8>,
    /// F157 "limit SDP-imposed CfL delay" — set per-SB at the chroma 64x64 root when the
    /// chroma split direction differs from luma's; disables CfL for that SB's small blocks
    /// (dav2d `t->sdp_cfl_disallowed`, set decode.c:3839, consumed in the `cfl_allowed` test).
    pub sdp_cfl_disallowed: bool,
}

impl ChromaNb {
    pub fn new(n: usize) -> Self {
        Self {
            a: [vec![0x40; n], vec![0x40; n]],
            l: [vec![0x40; n], vec![0x40; n]],
            a_uvmode: vec![0; n],
            l_uvmode: vec![0; n],
            sdp_cfl_disallowed: false,
        }
    }
}

/// One gathered neighbour slot: `(is_left, offset)` into the corresponding array.
pub type NbSlot = Option<(bool, usize)>;

/// Gather the (up to two) spatial neighbours for a block (dav2d `decode_b` nb/boff
/// setup). `have_top` already accounts for "do not cross SB boundaries vertically".
/// `(w4,h4)` are the block's coded width/height (== `bw4`/`bh4` when not edge-clipped).
pub fn gather_nb(
    have_left: bool,
    have_top: bool,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    w4: usize,
    h4: usize,
) -> [NbSlot; 2] {
    let mut slots: [NbSlot; 2] = [None, None];
    let mut idx = 0;
    if have_left && bh4 == h4 {
        slots[0] = Some((true, by4 + bh4 - 1));
        idx += 1;
    }
    if have_top && bw4 == w4 {
        slots[idx] = Some((false, bx4 + bw4 - 1));
        idx += 1;
    }
    if have_left && idx < 2 {
        slots[idx] = Some((true, by4));
        idx += 1;
    }
    if have_top && idx < 2 {
        slots[idx] = Some((false, bx4));
        idx += 1;
    }
    slots
}

/// Sum a neighbour context value over the two gathered slots (`boff == -1` → 0),
/// reading from the above (`a`) or left (`l`) array per slot.
pub fn nb_sum(slots: &[NbSlot; 2], a: &[u8], l: &[u8]) -> u32 {
    slots
        .iter()
        .map(|s| match s {
            Some((is_left, off)) => (if *is_left { l[*off] } else { a[*off] }) as u32,
            None => 0,
        })
        .sum()
}

/// Recursive superblock-luma decode state (dav2d `decode_sb`): the MSAC, CDFs, the
/// partition-context arrays, and the per-field neighbour arrays, threaded through the
/// recursion. `filters_done` gates the once-per-SB filter params to the first leaf.
pub struct SbState<'a> {
    pub msac: &'a mut crate::msac::MsacContext,
    pub cdf: &'a mut crate::cdf_av2::CdfContext,
    pub a_part: Vec<u8>,
    pub l_part: Vec<u8>,
    pub a_nb: BlockNbCtx,
    pub l_nb: BlockNbCtx,
    pub filters_done: bool,
    pub force_integer_mv: u8,
    pub max_bvp_drl_bits: u8,
    /// 16x16 per-SB map of each luma leaf's directional mode index (`midx`, base `0xff`),
    /// splatted over the leaf's region. The SDP chroma tree reads this for `intra_uv_mode`
    /// (dav2d `luma_intra_dir_mode_map`).
    pub luma_dir_map: [u8; 256],
    /// The previous SB's decoded cdef index (the left neighbour for the next SB's cdef
    /// context); `-1` = no left neighbour. Reset to `-1` at the start of each SB row.
    pub left_cdef: i8,
    /// The last decoded ccso flag per plane — the left-neighbour ccso context for the next
    /// 256px-aligned ccso block (dav2d `t->lf_mask[-1].ccso[p]`). Reset to 0 at the row start.
    pub left_ccso: [u8; 3],
    /// Frame dimensions in 4px units (`iw4 = (w+3)/4`, `ih4 = (h+3)/4`) — for the partition
    /// frame-boundary forcing + out-of-frame child-skip on partial edge SBs.
    pub iw4: usize,
    pub ih4: usize,
    /// Per-SB-column cdef index (the TOP-neighbour cdef context for the SB below); `-1` = no
    /// top. Indexed by SB column `bx4 >> 4`; persists across rows (this frame is 1 256px tall).
    pub top_cdef: [i8; 8],
}

/// dav2d decode.c:1841 — the per-64px-CELL `cdef_idx` read: fires at the FIRST leaf that
/// touches an unset (-1) 64-cell and is "coded" (`!skip_txfm || cdef.on_skiptx`). ctx comes
/// from the LEFT cell (tile-clamped, crossing SBs within the row) and the TOP cell (never
/// crossing the SB's top 64-row: `!((by4&~15) & (sb_step-1)) -> -1`). The decoded index
/// splats over the block's full 64-cell span (a 128-wide/tall block covers 2 cells/axis).
/// At 64px SBs this reduces EXACTLY to the old once-per-SB read (one cell per SB, top
/// always -1, left = the previous SB's value).
pub fn read_cdef_per64(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    bx4: usize,
    by4: usize,
    w4: usize,
    h4: usize,
    skip_txfm: bool,
) {
    if !HDR_TOOL_CFG.with(|c| c.get().cdef) {
        return;
    }
    let (n_str, on_skip) = crate::av2_frame::CDEF_CFG.with(|c| {
        let g = c.get();
        (g.n_strengths, g.on_skiptx)
    });
    if skip_txfm && !on_skip {
        return;
    }
    let (cell_state, w, cx, cy) = crate::av2_frame::FRAME.with(|f| {
        let f = f.borrow();
        if f.cdef_idx.is_empty() {
            return (None, 0usize, 0usize, 0usize);
        }
        let w = f.cdef_sbw;
        let (cx, cy) = (bx4 >> 4, by4 >> 4);
        let cell = cy * w + cx;
        if cell >= f.cdef_idx.len() || f.cdef_idx[cell] != -1 {
            return (None, w, cx, cy);
        }
        let tb = TILE_B.with(|t| t.get());
        let left = if (bx4 as i32) - 16 < tb.0 as i32 { -1i8 } else { f.cdef_idx[cell - 1] };
        let top = if (by4 & !15) & (sb_step4() - 1) == 0 { -1i8 } else { f.cdef_idx[cell - w] };
        (Some((left, top)), w, cx, cy)
    });
    let Some((left, top)) = cell_state else { return };
    let v = if n_str == 1 {
        0
    } else {
        let ctx = if (left | top) != -1 {
            let c = (left == 0) as usize + (top == 0) as usize;
            c + (c == 2) as usize
        } else {
            ((left & top) == 0) as usize * 2
        };
        read_cdef_v(msac, &mut cdf.m, ctx, n_str)
    };
    crate::av2_frame::FRAME.with(|f| {
        let mut f = f.borrow_mut();
        let len = f.cdef_idx.len();
        for yy in 0..(h4 / 16).max(1) {
            for xx in 0..(w4 / 16).max(1) {
                if cx + xx < w {
                    let c2 = (cy + yy) * w + cx + xx;
                    if c2 < len {
                        f.cdef_idx[c2] = v as i8;
                    }
                }
            }
        }
    });
}

/// dav2d decode.c:1875-1918 — the once-per-SB `cdef_idx` value, mirroring the reference:
/// `n_strengths==1` → 0 (no symbol at all); else a `cdef_idx0[ctx]` bool (true→0); else for
/// `n_strengths==2` the value is 1 with **NO further symbol**; else `1 + adapt8(cdef_idx[rem])`
/// with `rem = n_strengths-3`. The `n_strengths==2` no-symbol case is per-frame (frame 1 of our
/// clip has 3 strengths, frame 2 has 2), so it must be threaded, not hardcoded to the ≥3 path.
pub fn read_cdef_v(
    msac: &mut crate::msac::MsacContext,
    m: &mut crate::cdf_av2::CdfModeContext,
    ctx: usize,
    n_strengths: usize,
) -> i32 {
    use crate::msac::{rav1d_msac_decode_bool_adapt, rav1d_msac_decode_symbol_adapt8};
    if n_strengths == 1 {
        return 0;
    }
    if rav1d_msac_decode_bool_adapt(msac, &mut m.cdef_idx0[ctx]) {
        0
    } else if n_strengths == 2 {
        1
    } else {
        let rem = n_strengths - 3;
        1 + rav1d_msac_decode_symbol_adapt8(msac, &mut m.cdef_idx[rem], (rem + 1) as u8) as i32
    }
}

/// Decode the once-per-SB filter params at the first leaf (dav2d `decode_b`: gdf,
/// cdef_idx, ccso×3). Cleared-neighbour contexts (the SB corner). Frame-1 (keyframe) path,
/// which has `n_strengths=3`.
fn decode_sb_filters(s: &mut SbState) {
    use crate::msac::rav1d_msac_decode_bool_adapt;
    let _gdf = rav1d_msac_decode_bool_adapt(s.msac, &mut s.cdf.m.gdf);
    let _cdef = read_cdef_v(s.msac, &mut s.cdf.m, 0, 3);
    for p in 0..3 {
        let _ccso = rav1d_msac_decode_bool_adapt(s.msac, &mut s.cdf.m.ccso[p][0]);
    }
}

/// Recursively decode a superblock's luma (dav2d `decode_sb`): decode the partition,
/// then either decode the leaf (`decode_b_luma`, with filters at the SB corner) or
/// recurse into the sub-blocks. V/H use the half-split sizes; H3/V3 the 4-sub-block
/// three-band geometry. `bx4`/`by4` are 4-pixel block coords within the frame.
pub fn decode_sb_luma(s: &mut SbState, bs: usize, bx4: usize, by4: usize) -> u32 {
    use crate::av2_decode::{decode_partition, splat_partition, BlockPartition, BLOCK_DIMENSIONS};
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    // Frame-boundary split availability (dav2d): a half-split is available only if the frame
    // extends past the block's mid-line. `iw4`/`ih4` are the frame's 4px dimensions.
    let (iw4, ih4) = (s.iw4, s.ih4);
    let have_h_split = iw4 > bx4 + bw4 / 2;
    let have_v_split = ih4 > by4 + bh4 / 2;
    // Key-frame twin of the inter path's [SBI] probe, aligned with avm's PARTPROBE `[PARTIN]`
    // (decodeframe.c:1843) so the two traces diff node-for-node. `rng` is printed at partition-node
    // ENTRY, as avm's is; `dif` comes too, because rng alone hides bypass-only divergences.
    if std::env::var("PARTIN").is_ok() {
        crate::dlog!("[PARTIN] mi=({bx4},{by4}) bs={bs} rng={} dif={:x}", s.msac.rng, s.msac.dif);
    }
    let (bp, _half) = decode_partition(
        s.msac, &mut s.cdf.m, bs, &s.a_part, &s.l_part, bx4, by4, have_h_split, have_v_split, 3,
        true, 0, 0, 0, iw4, ih4, false,
    );
    // Accumulate the dav2d `dir_ptr` for SDP F164 chroma inference: byte0 = partition code
    // (NONE → 0xff, else the BlockPartition value), bits 16-23 = OR of the *direct*
    // children's byte0. The chroma 64x64 root reads it to infer (or decode) its partition.
    let mut children_or = 0u32;
    match bp {
        BlockPartition::None => {
            let do_filters = !s.filters_done;
            s.filters_done = true;
            let tb = TILE_B.with(|t| t.get());
            let have_left = bx4 > tb.0;
            let have_top = by4 > tb.2;
            // Generic symbol-level oracle: entry rng at each luma leaf, aligned to dav2d's
            // `decode_b[y=,x=,plane=y]: r=` (y/x in 4px units). Diff to find the first divergence.
            if std::env::var("MTRACE").is_ok() {
                crate::dlog!("[ML] y={by4} x={bx4} r={} d={:x}", s.msac.rng, s.msac.dif);
            }
            // cdef top-neighbour: for 64px SBs each SB is a single cdef unit, so the top
            // neighbour is always the SB *above* — which doesn't cross the SB-row boundary
            // vertically (and isn't committed during the parse pass). So it is always -1,
            // exactly like the block-context have_top_in_sb. (The left neighbour DOES cross
            // the SB-column boundary — that is `left_cdef`, threaded across SBs in the row.)
            let top_cdef = -1i8;
            let info = decode_b_luma(
                s.msac, s.cdf, &mut s.a_nb, &mut s.l_nb, bs, bx4, by4, have_left, have_top, do_filters,
                s.force_integer_mv, s.max_bvp_drl_bits, &mut s.left_cdef, &mut s.left_ccso, top_cdef,
                true, false, false, 3,
            );
            if std::env::var("SBTRACE").is_ok() {
                crate::dlog!(
                    "[rav2d SB] leaf bs={bs} ({bx4},{by4}) fsc={} intrabc={} eob={} rng={}",
                    info.fsc as u8, info.intrabc as u8, info.eob, s.msac.rng
                );
            }
            if std::env::var("F1BLK").is_ok() && info.intrabc {
                crate::dlog!("F1BLK bx={bx4} by={by4} bv=({},{})", info.ibc_bv.0, info.ibc_bv.1);
            }
            if std::env::var("AIDBG").map_or(false, |v| v == "all" || v == format!("{bx4},{by4}")) {
                crate::dlog!("AIDBG ({bx4},{by4}) bs={bs} y_mode_idx={} midx={} mrl={} fsc={} eob={} rng={}",
                    info.y_mode_idx, info.midx, info.mrl_index, info.fsc as u8, info.eob, s.msac.rng);
            }
            // intrabc leaves parse via decode_b_luma but it returns WITHOUT reconstructing (the
            // intrabc branch has no recon_intra_luma). Reconstruct the LUMA here (chroma is done by
            // the separate SDP chroma tree) — else the block is a 0 gap. dav2d recon_tmpl.c intrabc.
            if info.intrabc {
                recon_intrabc(bx4, by4, bw4, bh4, bd[2] as usize, bd[3] as usize, info.ibc_bv, info.ibc_morph, &info.cf, info.txtp, info.all_zero, None, info.stx, info.eob, true, &info.units);
                // Splat this intrabc block's BV into the refmvs grid + bank so a LATER intrabc
                // block's `refmvs_find(ref=-1)` (spatial scan + bank re-seed) can collect it. The
                // keyframe uses the SAME DRL machinery as frame 2 (brick B) — without this every
                // frame-1 intrabc BV fell to the defaults.
                use crate::av2_refmvs::{Mv, BANK, GRID};
                let mv = Mv { y: info.ibc_bv.0, x: info.ibc_bv.1 };
                GRID.with(|g| g.borrow_mut().splat_intrabc(bx4, by4, bw4, bh4, mv, bs as u8));
                BANK.with(|bk| bk.borrow_mut().add_block(bw4, bh4, by4, bx4, sb_step4(), sb_step4() >> 5, -1, mv));
            } else {
                // Intra luma block: splat ref=-1/INVALID into the grid (dav `splat_intraref`) +
                // refresh the bank avail/hits so a later intrabc block's spatial scan / bank works.
                use crate::av2_refmvs::{BANK, GRID};
                GRID.with(|g| g.borrow_mut().splat_intra(bx4, by4, bw4, bh4, bs as u8));
                BANK.with(|bk| bk.borrow_mut().bank_update_intra(bw4, bh4, by4, bx4, sb_step4(), sb_step4() >> 5));
            }
            // SDP: record this leaf's directional mode index into the 16x16 SB map, read
            // by the chroma tree's intra_uv_mode (dav2d luma_intra_dir_mode_map set_ctx).
            let off = (by4 & 15) * 16 + (bx4 & 15);
            for y in 0..bh4.min(16) {
                for x in 0..bw4.min(16) {
                    s.luma_dir_map[off + y * 16 + x] = info.midx;
                }
            }
            splat_partition(&mut s.a_part, &mut s.l_part, bs, bx4, by4);
        }
        BlockPartition::V => {
            let hw4 = bw4 / 2;
            let h = crate::av2_decode::PART_HALF[bs][1] as usize;
            children_or |= decode_sb_luma(s, h, bx4, by4) & 0xff;
            if bx4 + hw4 < iw4 {
                children_or |= decode_sb_luma(s, h, bx4 + hw4, by4) & 0xff;
            }
        }
        BlockPartition::H => {
            let hh4 = bh4 / 2;
            let h = crate::av2_decode::PART_HALF[bs][0] as usize;
            children_or |= decode_sb_luma(s, h, bx4, by4) & 0xff;
            if by4 + hh4 < ih4 {
                children_or |= decode_sb_luma(s, h, bx4, by4 + hh4) & 0xff;
            }
        }
        BlockPartition::H3 => {
            let (qh4, hw4, hh4) = (bh4 / 4, bw4 / 2, bh4 / 2);
            let (strip, mid) = h3_sub_sizes(bs);
            children_or |= decode_sb_luma(s, strip, bx4, by4) & 0xff;
            if by4 + qh4 < ih4 {
                children_or |= decode_sb_luma(s, mid, bx4, by4 + qh4) & 0xff;
            }
            if bx4 + hw4 < iw4 && by4 + qh4 < ih4 {
                children_or |= decode_sb_luma(s, mid, bx4 + hw4, by4 + qh4) & 0xff;
            }
            if by4 + qh4 + hh4 < ih4 {
                children_or |= decode_sb_luma(s, strip, bx4, by4 + qh4 + hh4) & 0xff;
            }
        }
        BlockPartition::V3 => {
            let (qw4, hw4, hh4) = (bw4 / 4, bw4 / 2, bh4 / 2);
            let (strip, mid) = v3_sub_sizes(bs);
            children_or |= decode_sb_luma(s, strip, bx4, by4) & 0xff;
            if bx4 + qw4 < iw4 {
                children_or |= decode_sb_luma(s, mid, bx4 + qw4, by4) & 0xff;
            }
            if bx4 + qw4 < iw4 && by4 + hh4 < ih4 {
                children_or |= decode_sb_luma(s, mid, bx4 + qw4, by4 + hh4) & 0xff;
            }
            if bx4 + qw4 + hw4 < iw4 {
                children_or |= decode_sb_luma(s, strip, bx4 + qw4 + hw4, by4) & 0xff;
            }
        }
        BlockPartition::V4a | BlockPartition::V4b => {
            // Uneven vertical 4-way (dav2d decode.c:4052): 4 full-height strips of widths
            // (in bw4/8 = `ew4` units) 1,2,4,1 (V4a) or 1,4,2,1 (V4b), using the
            // eighth/quarter/half V sub-sizes. (Frame-boundary child-skip omitted — interior.)
            let ew4 = bw4 / 8;
            let var = matches!(bp, BlockPartition::V4b) as usize;
            let (w8, h8) = (bw4 as u8, bh4 as u8);
            let eighth = bs_for_dims(w8 / 8, h8);
            let (c2, c3) = if var == 0 {
                (bs_for_dims(w8 / 4, h8), bs_for_dims(w8 / 2, h8)) // quarter, half
            } else {
                (bs_for_dims(w8 / 2, h8), bs_for_dims(w8 / 4, h8)) // half, quarter
            };
            let (w4a, w4b) = ((bw4 / 4) << var, (bw4 / 2) >> var);
            children_or |= decode_sb_luma(s, eighth, bx4, by4) & 0xff;
            if bx4 + ew4 < iw4 {
                children_or |= decode_sb_luma(s, c2, bx4 + ew4, by4) & 0xff;
            }
            if bx4 + ew4 + w4a < iw4 {
                children_or |= decode_sb_luma(s, c3, bx4 + ew4 + w4a, by4) & 0xff;
            }
            if bx4 + ew4 + w4a + w4b < iw4 {
                children_or |= decode_sb_luma(s, eighth, bx4 + ew4 + w4a + w4b, by4) & 0xff;
            }
        }
        BlockPartition::H4a | BlockPartition::H4b => {
            // Uneven horizontal 4-way (dav2d decode.c:4093): 4 full-width strips of heights
            // 1,2,4,1 (H4a) or 1,4,2,1 (H4b) in bh4/8 = `eh4` units.
            let eh4 = bh4 / 8;
            let var = matches!(bp, BlockPartition::H4b) as usize;
            let (w8, h8) = (bw4 as u8, bh4 as u8);
            let eighth = bs_for_dims(w8, h8 / 8);
            let (c2, c3) = if var == 0 {
                (bs_for_dims(w8, h8 / 4), bs_for_dims(w8, h8 / 2))
            } else {
                (bs_for_dims(w8, h8 / 2), bs_for_dims(w8, h8 / 4))
            };
            let (h4a, h4b) = ((bh4 / 4) << var, (bh4 / 2) >> var);
            children_or |= decode_sb_luma(s, eighth, bx4, by4) & 0xff;
            if by4 + eh4 < ih4 {
                children_or |= decode_sb_luma(s, c2, bx4, by4 + eh4) & 0xff;
            }
            if by4 + eh4 + h4a < ih4 {
                children_or |= decode_sb_luma(s, c3, bx4, by4 + eh4 + h4a) & 0xff;
            }
            if by4 + eh4 + h4a + h4b < ih4 {
                children_or |= decode_sb_luma(s, eighth, bx4, by4 + eh4 + h4a + h4b) & 0xff;
            }
        }
        other => {
            crate::dlog!("[rav2d SB] UNHANDLED partition {other:?} at bs={bs} ({bx4},{by4})");
        }
    }
    // dir_ptr: byte0 = the DIRECTION code (dav decode.c:3872-3876 `(uint8_t)dir`):
    // H-family → 1, V-family → 2, NONE **and SPLIT** → 0xff (dir stays -1, cast to u8).
    // NOT the raw partition value — H3/H4x enum values leaked into the F157 lumadir
    // compare and falsely disallowed CfL for whole SBs (tilecols2 frame 3 (16,48)).
    // bits 8-15 = the partition value (read by F164 as the chroma bp); bits 16-23 = the
    // OR of the direct children's byte0.
    let byte0: u32 = match bp {
        BlockPartition::None | BlockPartition::Split => 0xff,
        BlockPartition::H | BlockPartition::H3 | BlockPartition::H4a | BlockPartition::H4b => 1,
        _ => 2, // V / V3 / V4a / V4b
    };
    byte0 | ((bp as u32) << 8) | (children_or << 16)
}

/// dav2d `size_group_lookup` (decode.c:1344) — NOT plain `min(log2w,log2h)`: the thin
/// 1:8/1:16 shapes deviate (64x8/8x64/64x4/4x64 -> 2, 32x4/4x32 -> 1).
pub static SIZE_GROUP: [u8; 31] = [
    3, 3, 3, 3, 3, 3, // 256x256..64x128
    3, 3, 2, 2, 2,    // 64x64..64x4
    3, 3, 2, 1, 1,    // 32x64..32x4
    2, 2, 2, 1, 0,    // 16x64..16x4
    2, 1, 1, 1, 0,    // 8x64..8x4
    2, 1, 0, 0, 0,    // 4x64..4x4
];

/// KEY-frame superblock walk for 128px SBs (dav2d `decode_sb` with `lbs == cbs`): the
/// upper tree (128x128 / 128x64 / 64x128) is decoded JOINTLY on both planes; when the
/// descent reaches 64x64 the SDP fork applies — the luma 64-subtree decodes, then the
/// chroma 64-subtree, per 64-region, interleaved (dav decode.c:3557). For a 64px-SB root
/// this degenerates immediately to the fork == the previous two-pass structure.
#[allow(clippy::too_many_arguments)]
pub fn decode_sb_key(
    s: &mut SbState,
    uv_a: &mut [u8],
    uv_l: &mut [u8],
    uv_nb: &mut ChromaNb,
    bs: usize,
    bx4: usize,
    by4: usize,
) {
    use crate::av2_decode::{decode_partition, splat_partition, BlockPartition, BLOCK_DIMENSIONS, PART_HALF};
    if bs >= 6 {
        // SDP fork at the 64x64 boundary: fresh dir accumulator, luma then chroma.
        let dirptr = decode_sb_luma(s, bs, bx4, by4);
        // CHROMA restoration units are read at the CHROMA pass root — after the whole
        // luma subtree, before the chroma tree (avm decodeframe.c:2081 via
        // get_partition_plane_start(CHROMA_PART)=1; corners gate on bsize == sb_size).
        // Only when this 64-region IS the SB root; for 128px SBs the chroma pass root
        // is the 128 block, a schedule no current stream exercises with chroma LR.
        if (bx4 | by4) & (sb_step4() - 1) == 0 {
            if sb_step4() == 16 {
                let css = ss_g();
                crate::av2_lr::read_lr_units_sb(
                    s.msac, s.cdf, bx4, by4, s.iw4, s.ih4, css.0, css.1, 1, 3,
                );
            } else {
                crate::av2_lr::LR_CFG.with(|c| {
                    let cfg = c.borrow();
                    if cfg.p[1].r_type != 0 || cfg.p[2].r_type != 0 {
                        crate::dlog!("[rav2d] WARNING: chroma LR with 128px SBs is unverified (no oracle stream)");
                    }
                });
            }
        }
        let map = s.luma_dir_map;
        decode_sb_chroma(s.msac, s.cdf, uv_a, uv_l, bs, bx4, by4, dirptr, &map, uv_nb, s.iw4, s.ih4);
        return;
    }
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    let (hw4, hh4) = (bw4 / 2, bh4 / 2);
    let (iw4, ih4) = (s.iw4, s.ih4);
    let have_h_split = iw4 > bx4 + hw4;
    let have_v_split = ih4 > by4 + hh4;
    // The JOINT (>64) key tree carries the real subsampling so the frame-boundary forced
    // arm can reject chroma-INVALID children (4:2:2 64x128) — pl stays 0 (shared tree).
    let (jssh, jssv) = { let ss = ss_g(); (ss.0 as u32, ss.1 as u32) };
    let (bp, _half) = decode_partition(
        s.msac, &mut s.cdf.m, bs, &s.a_part, &s.l_part, bx4, by4, have_h_split, have_v_split, 3,
        true, 0, jssh, jssv, iw4, ih4, false,
    );
    if std::env::var("MPARTK").is_ok() { crate::dlog!("[MPARTK] mi=({bx4},{by4}) bs={bs} bp={bp:?} rng={}", s.msac.rng); }
    match bp {
        BlockPartition::None => {
            // Joint >=64px yuv leaf (dav decode_b with both planes valid).
            let tb = TILE_B.with(|t| t.get());
            let have_left = bx4 > tb.0;
            let have_top = by4 > tb.2;
            let do_filters = !s.filters_done;
            s.filters_done = true;
            decode_b_key_yuv(s, uv_nb, bs, bx4, by4, have_left, have_top, do_filters);
            // A joint leaf splats BOTH planes' partition-context arrays (dav decode.c:3934).
            splat_partition(&mut s.a_part, &mut s.l_part, bs, bx4, by4);
            splat_partition(uv_a, uv_l, bs, bx4, by4);
        }
        BlockPartition::H => {
            let h = PART_HALF[bs][0] as usize;
            decode_sb_key(s, uv_a, uv_l, uv_nb, h, bx4, by4);
            if by4 + hh4 < ih4 {
                decode_sb_key(s, uv_a, uv_l, uv_nb, h, bx4, by4 + hh4);
            }
        }
        BlockPartition::V => {
            let h = PART_HALF[bs][1] as usize;
            decode_sb_key(s, uv_a, uv_l, uv_nb, h, bx4, by4);
            if bx4 + hw4 < iw4 {
                decode_sb_key(s, uv_a, uv_l, uv_nb, h, bx4 + hw4, by4);
            }
        }
        BlockPartition::Split => {
            // z-order quadrants at 64x64. A fully off-frame quadrant decodes NOTHING (the
            // 4:2:2 edge-SB forced Split arrives here with the right half out of frame).
            for &(px, py) in &[(bx4, by4), (bx4 + hw4, by4), (bx4, by4 + hh4), (bx4 + hw4, by4 + hh4)] {
                if px < iw4 && py < ih4 {
                    decode_sb_key(s, uv_a, uv_l, uv_nb, 6, px, py);
                }
            }
        }
        other => panic!("decode_sb_key: partition {other:?} at bs={bs} ({bx4},{by4})"),
    }
}

/// A KEY-frame joint (yuv) leaf at >=64px (dav2d `decode_b`, both planes): luma mode
/// (deferred coefs) -> chroma mode -> tx-part + per-unit luma coefs -> chroma coefs ->
/// recon. Mirrors the inter tree's intra-yuv leaf with keyframe parameters
/// (inter_frame=false, cdef n_strengths=3).
fn decode_b_key_yuv(
    s: &mut SbState,
    cnb: &mut ChromaNb,
    bs: usize,
    bx4: usize,
    by4: usize,
    have_left: bool,
    have_top: bool,
    decode_filters: bool,
) {
    use crate::av2_decode::BLOCK_DIMENSIONS;
    use crate::msac::rav1d_msac_decode_bool_adapt;
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    let info = decode_b_luma(
        s.msac, s.cdf, &mut s.a_nb, &mut s.l_nb, bs, bx4, by4, have_left, have_top, decode_filters,
        s.force_integer_mv, s.max_bvp_drl_bits, &mut s.left_cdef, &mut s.left_ccso, -1,
        true, false, true, 3,
    );
    assert!(!info.intrabc, "key joint intrabc leaf at bs={bs} ({bx4},{by4}) unimplemented");
    let cm = decode_chroma_mode(s.msac, s.cdf, cnb, bs, bx4, by4, info.midx);
    let msac = &mut *s.msac;
    let cdf = &mut *s.cdf;
    let (a_nb, l_nb) = (&mut s.a_nb, &mut s.l_nb);
    let tx_part = read_tx_part(msac, cdf, bw4, bh4, info.fsc, false, false);
    // Coef CHUNK walk (dav2d read_coef_blocks, imax(bw4,bh4) > 16): the >64px block is
    // read in 64px luma chunks row-major, and the CHROMA block is read with the FIRST
    // chunk of each 2x2 (420) chunk group — i.e. interleaved luma(0), U, V, luma(1)...,
    // NOT all-luma-then-chroma. tx_part is NONE for >64px blocks (no partition symbols),
    // so each chunk is a single 64x64 TX unit.
    let (sshcw, ssvcw) = ss_g();
    let mut tx_layout: Vec<(usize, usize, usize, usize, bool)> = Vec::new();
    {
        let (uw, uh) = (bw4.min(16), bh4.min(16));
        for cy in 0..bh4.div_ceil(uh) {
            for cx in 0..bw4.div_ceil(uw) {
                let chroma_here = (cx & sshcw) == 0 && (cy & ssvcw) == 0 && (cx | cy) == 0;
                tx_layout.push((cx * uw, cy * uh, uw, uh, chroma_here));
            }
        }
    }
    let skip_set = info.fsc as usize;
    let (fw4, fh4) = crate::av2_frame::FRAME.with(|f| { let f = f.borrow(); (f.iw4, f.ih4) });
    let mut iunits: Vec<(usize, usize, usize, usize, Vec<i32>, u8, u8, u8, bool)> = Vec::new();
    let mut cc_opt: Option<_> = None;
    for &(ux, uy, utw4, uth4, chroma_here) in &tx_layout {
        let (ubx4, uby4) = (bx4 + ux, by4 + uy);
        if ubx4 >= fw4 || uby4 >= fh4 { continue; }
        let (uslw, uslh) = (utw4.trailing_zeros() as usize, uth4.trailing_zeros() as usize);
        let (uclw, uclh) = (uslw.min(3), uslh.min(3));
        let u_tdc = (uslw + uslh + 1) >> 1;
        let u2d = uclw + uclh;
        let ubw4 = if fw4 > ubx4 { utw4.min(fw4 - ubx4) } else { utw4 };
        let ubh4 = if fh4 > uby4 { uth4.min(fh4 - uby4) } else { uth4 };
        let sctx = if info.fsc {
            9
        } else {
            crate::av2_coef::skip_ctx_luma(&a_nb.lcoef[ubx4..], &l_nb.lcoef[uby4..], uslw, uslh, &bd) as usize
        };
        if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({ubx4},{uby4}) pl=0k txs={u_tdc} skipctx={sctx} rng={}", msac.rng); }
        let az = rav1d_msac_decode_bool_adapt(msac, &mut cdf.coef.skip[skip_set][u_tdc][sctx]);
        let mut cf = vec![0i32; 1usize << (uslw + uslh + 4)];
        let (cf_ctx, u_txtp, u_stxt, u_stxs) = if az {
            (0x40u8, DCT_DCT, 0u8, 0u8)
        } else {
            let e = crate::av2_coef::decode_eob(msac, &mut cdf.coef, u2d, 0);
            let scan = crate::av2_tables_gen::SCANS[scan_idx_square(uclw, uclh)];
            let y_mode_raw = y_mode_from_idx(info.y_mode_idx, info.midx);
            let y_mode = if info.y_mode_idx >= 5 {
                wide_angle_remap_mode(y_mode_raw, info.midx as i32 % 7 - 3, info.mrl_index, 4 << uslw, 4 << uslh)
            } else {
                y_mode_raw
            };
            let dc_sign_ctx = crate::av2_coef::get_dc_sign_ctx(&a_nb.lcoef[ubx4..], &l_nb.lcoef[uby4..], uslw, uslh, ubw4 as i32, ubh4 as i32);
            let (r, txtp, stxt, stxs) = decode_coefs_y(msac, cdf, &mut cf, e, info.fsc, y_mode, uslw.min(uslh), u_tdc, uslw, uslh, u2d, scan, false, dc_sign_ctx);
            (r, txtp, stxt, stxs)
        };
        for x in ubx4..(ubx4 + ubw4).min(fw4).min(a_nb.lcoef.len()) { a_nb.lcoef[x] = cf_ctx; }
        for y in uby4..(uby4 + ubh4).min(fh4).min(l_nb.lcoef.len()) { l_nb.lcoef[y] = cf_ctx; }
        iunits.push((ux, uy, uslw, uslh, cf, u_txtp, u_stxt, u_stxs, az));
        if chroma_here {
            cc_opt = Some(decode_chroma_coefs(msac, cdf, cnb, bs, bx4, by4, info.fsc as usize, true, DCT_DCT));
        }
    }
    let cc = cc_opt.expect("joint leaf chroma chunk never read");
    // RECON: per-unit chained luma prediction, then chroma.
    for (ux, uy, uslw, uslh, cf, u_txtp, u_stxt, u_stxs, u_az) in &iunits {
        let (ubx4, uby4) = (bx4 + ux, by4 + uy);
        let ubw4 = if fw4 > ubx4 { (1usize << uslw).min(fw4 - ubx4) } else { 1usize << uslw };
        let ubh4 = if fh4 > uby4 { (1usize << uslh).min(fh4 - uby4) } else { 1usize << uslh };
        recon_intra_luma(
            ubx4, uby4, *uslw, *uslh, ubw4, ubh4, info.y_mode_idx, info.midx, info.mrl_index,
            info.multi_mrl != 0, cf, *u_txtp, *u_stxt, *u_stxs, *u_az, info.fsc,
            have_left || *ux > 0, have_top || *uy > 0,
            tx_part >= 6 && (*ux > 0 || *uy > 0),
            None,
        );
    }
    crate::av2_frame::mark_btype(bx4, by4, bw4, bh4, 2);
    {
        let (sshc, ssvc) = ss_g();
        let (ccbx, ccby) = (bx4 >> sshc, by4 >> ssvc);
        recon_intra_chroma(
            ccbx, ccby, bx4, by4, bw4, bh4, cc.slw, cc.slh, cm.uv_mode, cm.uv_angle, cm.cfl_mode,
            cm.cfl_alpha_u, cm.cfl_alpha_v, cm.mh_dir, &cc.cf_u, &cc.cf_v, cc.u_eob == -1, cc.v_eob == -1,
            ccbx > TILE_B.with(|t| t.get().0 >> sshc), ccby > TILE_B.with(|t| t.get().2 >> ssvc),
        );
        crate::av2_frame::FRAME.with(|f| {
            f.borrow_mut().mark_db_chroma(ccbx, ccby, (bw4 >> sshc).max(1), (bh4 >> ssvc).max(1));
        });
    }
    // refmvs bookkeeping: mirror the key luma leaf (intra grid splat + bank refresh).
    crate::av2_refmvs::GRID.with(|g| g.borrow_mut().splat_intra(bx4, by4, bw4, bh4, bs as u8));
    crate::av2_refmvs::BANK.with(|bk| bk.borrow_mut().bank_update_intra(bw4, bh4, by4, bx4, sb_step4(), sb_step4() >> 5));
}

/// Decode a chroma leaf's `intra_uv_mode` (dav2d `decode_b` `has_chroma` path, the
/// non-lossless / non-dpcm branch). Decodes the CfL flag, then — when not CfL — the
/// `intra_uv_mode` symbol whose context keys off whether the co-located luma block had a
/// directional mode (`midx != 0xff`, read from the SDP luma dir map). Returns
/// `(uv_mode, uv_angle)`. (LAYER 2: CfL is assumed allowed here; `cfl_alpha`/mhccp and the
/// `cfl_allowed` gating via `seq->cfl`/`sdp_cfl_disallowed` land with the coefficient layer.)
pub fn decode_b_chroma(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    nb: &mut ChromaNb,
    luma_dir_map: &[u8; 256],
    bs: usize,
    bx4: usize,
    by4: usize,
) -> (u8, i32) {
    use crate::av2_decode::BLOCK_DIMENSIONS;
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bools_bypass,
        rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8,
    };
    // --- intra_uv_mode --- CfL flag, context = (#neighbours that are CFL_PRED), then UV mode.
    const CFL_PRED: u8 = 13;
    if std::env::var("MTRACE").is_ok() {
        crate::dlog!("[CML] y={by4} x={bx4} r={} d={:x}", msac.rng, msac.dif);
    }
    let bdc0 = BLOCK_DIMENSIONS[bs];
    let (ssh, ssv) = ss_g(); let (cbx4, cby4) = (bx4 >> ssh, by4 >> ssv); // chroma position (ss-general)
    // cfl_allowed (dav2d decode.c:2159): seq cfl (true, plumb from header) AND the block is
    // big (>64px) OR CfL isn't SDP-disallowed for this SB, AND the chroma block is <=16 units.
    // When not allowed, the `is_cfl` flag is NOT coded (the block is plain intra-UV).
    let (bw4c, bh4c) = (bdc0[0] as usize, bdc0[1] as usize);
    let (cbw4c, cbh4c) = ((bw4c >> ssh).max(1), (bh4c >> ssv).max(1));
    let cfl_allowed = HDR_TOOL_CFG.with(|c| c.get().cfl)
        && (bw4c.max(bh4c) > 16 || !nb.sdp_cfl_disallowed)
        && cbw4c.max(cbh4c) <= 16;
    let cfl_ctx = (nb.a_uvmode[cbx4] == CFL_PRED) as usize + (nb.l_uvmode[cby4] == CFL_PRED) as usize;
    let is_cfl = cfl_allowed && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.cfl[cfl_ctx]);
    if std::env::var("MUVCK").is_ok() {
        crate::dlog!("[MUVCK] mi=({bx4},{by4}) allowed={} dis={} ctx={cfl_ctx} cfl={} r={}", cfl_allowed as u8, nb.sdp_cfl_disallowed as u8, is_cfl as u8, msac.rng);
    }
    // C-chroma recon: capture the CfL sub-mode (1=explicit, 2=derived, 3=mhccp) + explicit alphas.
    let mut cfl_mode = 0u8;
    let mut cfl_alpha_u = 0i32;
    let mut cfl_alpha_v = 0i32;
    let mut mh_dir = 0u8;
    let (uv_mode, uv_angle): (u8, i32) = if is_cfl {
        // CfL parameters (dav2d `decode_b` CFL_PRED branch). MHCCP is allowed for small
        // chroma blocks (seq->mhccp=1 this stream); when not taken, the explicit path codes
        // `cfl_type`, then a joint U/V sign symbol, then a per-nonzero-sign alpha magnitude.
        let bdc = BLOCK_DIMENSIONS[bs];
        // MHCCP max chroma TX is 32x32 (avm av2_get_max_uv_txsize_adjusted) — CHROMA dims,
        // per-axis subsampled.
        let (cbw4, cbh4) = ((bdc[0] as usize >> ssh).max(1), (bdc[1] as usize >> ssv).max(1));
        let mhccp_allowed = cbw4.max(cbh4) <= 8 && cbw4 * cbh4 >= 2;
        let seq_cfl = true; // plumb from the seq header
        if std::env::var("MCFL").is_ok() { crate::dlog!("[MCFLSW] mi=({bx4},{by4}) allowed={} cell={:?} rng={}", mhccp_allowed as u8, cdf.m.mhccp, msac.rng); }
        if mhccp_allowed && (!seq_cfl || rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.mhccp)) {
            let sz_ctx = SIZE_GROUP[bs] as usize;
            mh_dir = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.mhccp_filter_dir[sz_ctx], 2) as u8;
            cfl_mode = 3; // MHCCP (multi-param)
        } else if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.cfl_type) {
            // CFL_EXPLICIT: joint sign s ∈ [1,8]; (sign_u, sign_v) = (s/3, s%3); each nonzero
            // sign codes an alpha magnitude from a sign-derived context.
            cfl_mode = 1;
            let sign = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.cfl_sign, 7) as i32 + 1;
            let sign_u = (sign * 0x56) >> 8;
            let sign_v = sign - sign_u * 3;
            // cfl_idx_to_alpha: sign 0=ZERO/1=NEG/2=POS; alpha = ±(mag+1) (·CFL_ADD_BITS_ALPHA=0).
            if sign_u != 0 {
                let ctx = (sign_u == 2) as usize * 3 + sign_v as usize;
                let mag = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.cfl_alpha[ctx], 7) as i32;
                cfl_alpha_u = if sign_u == 2 { mag + 1 } else { -(mag + 1) };
            }
            if sign_v != 0 {
                let ctx = (sign_v == 2) as usize * 3 + sign_u as usize;
                let mag = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.cfl_alpha[ctx], 7) as i32;
                cfl_alpha_v = if sign_v == 2 { mag + 1 } else { -(mag + 1) };
            }
        } else {
            cfl_mode = 2; // CFL_DERIVED (implicit alpha)
        }
        if std::env::var("MUVCK").is_ok() {
            crate::dlog!("[MUVCK2] mi=({bx4},{by4}) cfl_mode={cfl_mode} mh_dir={mh_dir} r={}", msac.rng);
        }
        (CFL_PRED, 0)
    } else {
        let midx = luma_dir_map[(by4 & 15) * 16 + (bx4 & 15)];
        let uv_mode_ctx = (midx != 0xff) as usize;
        let mut uv_mode_idx =
            rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.intra_uv_mode[uv_mode_ctx], 7) as usize;
        if uv_mode_idx == 7 {
            uv_mode_idx += rav1d_msac_decode_bools_bypass(msac, 3) as usize;
        }
        if uv_mode_idx < uv_mode_ctx {
            (REORDERED_DIR_Y_MODE[(midx / 7) as usize], (midx % 7) as i32 - 3)
        } else if uv_mode_idx - uv_mode_ctx < 5 {
            (REORDERED_NONDIR_Y_MODE[uv_mode_idx - uv_mode_ctx], 0)
        } else {
            const DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
            const Y_TO_UV: [usize; 8] = [2, 4, 0, 5, 3, 6, 1, 7];
            let mut idx = uv_mode_idx - 5 - uv_mode_ctx;
            idx += (uv_mode_ctx != 0 && idx >= Y_TO_UV[(midx / 7) as usize]) as usize;
            // `uv_mode_idx > 12` is an invalid-bitstream error in dav2d; here it only occurs
            // once a chroma block has desynced — clamp so the build doesn't panic mid-tree.
            (DEFAULT_MODE_LIST_UV[idx.min(7)], 0)
        }
    };
    if std::env::var("MUVCK").is_ok() {
        crate::dlog!("[MUVCK3] mi=({bx4},{by4}) uv_mode={uv_mode} angle={uv_angle} r={}", msac.rng);
    }
    // Splat this block's uvmode for the next block's CfL context (dav2d uvmode set_ctx).
    let (cbw4_m, cbh4_m) = ((bdc0[0] as usize >> ssh).max(1), (bdc0[1] as usize >> ssv).max(1));
    nb.a_uvmode[cbx4..cbx4 + cbw4_m].fill(uv_mode);
    nb.l_uvmode[cby4..cby4 + cbh4_m].fill(uv_mode);

    // --- chroma coefficient decode (U then V) --- the chroma TX is the luma block
    // subsampled 4:2:0 (each log2 dim minus one), one TX per leaf (chroma <= 32x32 in a
    // 64x64 SB). The skip ctx reads live chroma neighbours; each plane splats its cf_ctx.
    let bd = BLOCK_DIMENSIONS[bs];
    let slw = (bd[2] as usize).saturating_sub(ssh);
    let slh = (bd[3] as usize).saturating_sub(ssv);
    // eob/coef size class uses the 32-clamped CORE dims (64-dim TXs code a 32-core;
    // same convention as the luma path's clw/clh).
    let tx2dszctx = slw.min(3) + slh.min(3);
    let t_dim_ctx = (slw + slh + 1) >> 1;
    let scan = crate::av2_tables_gen::SCANS[scan_idx_square(slw, slh)];
    let (cbw4, cbh4) = (1usize << slw, 1usize << slh); // one-TX chroma block dims
    // C1: record this chroma leaf's deblock edges (RECON_ACTIVE = the scored frame-1 pass).
    if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
        crate::av2_frame::FRAME.with(|f| f.borrow_mut().mark_db_chroma(cbx4, cby4, cbw4, cbh4));
    }
    // U plane.
    let ct48 = std::env::var("CT48").is_ok() && bx4 == 16 && by4 == 48;
    if ct48 { crate::dlog!("[CT48] post-uvmode uv_mode={uv_mode} cfl={is_cfl} cfl_ctx={cfl_ctx} cfl_allowed={cfl_allowed} a_uv[{cbx4}]={} l_uv[{cby4}]={} r={} d={:x}", nb.a_uvmode[cbx4], nb.l_uvmode[cby4], msac.rng, msac.dif); }
    let sctx_u = crate::av2_coef::skip_ctx_chroma(
        &nb.a[0], &nb.l[0], cbx4, cby4, cbw4, cbh4, 1, false, false,
    );
    if ct48 { crate::dlog!("[CT48] sctx_u={sctx_u} t_dim_ctx={t_dim_ctx} a[{cbx4}..+{cbw4}]={:?} l[{cby4}..+{cbh4}]={:?}", &nb.a[0][cbx4..cbx4+cbw4], &nb.l[0][cby4..cby4+cbh4]); }
    let cc96 = std::env::var("CC96").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == bx4 && p[1] == by4 });
    if cc96 {
        crate::dlog!("[CC96] leaf bx4={bx4} by4={by4} uv_mode={uv_mode} cfl={cfl_mode} slw={slw} slh={slh} tx2dszctx={tx2dszctx} t_dim_ctx={t_dim_ctx} sctx_u={sctx_u} scan_len={} rng={} dif={:x}", scan.len(), msac.rng, msac.dif);
        crate::dlog!("[CC96] aU[{cbx4}..+{cbw4}]={:?} lU[{cby4}..+{cbh4}]={:?}", &nb.a[0][cbx4..cbx4 + cbw4], &nb.l[0][cby4..cby4 + cbh4]);
        crate::av2_coef::COEF_DBG.with(|c| c.set(true));
    }
    let mut cf_u = vec![0i32; 1usize << (slw + slh + 4)];
    if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({bx4},{by4}) pl=1 txs={t_dim_ctx} skipctx={sctx_u} rng={}", msac.rng); }
    let (u_eob, cf_ctx_u) = crate::av2_coef::decode_coefs_uv(
        msac, &mut cdf.coef, &mut cdf.m.cctx, &mut cf_u, 1, t_dim_ctx, slw, slh, tx2dszctx, scan, sctx_u, 0, true, 0,
    );
    if cc96 { crate::av2_coef::COEF_DBG.with(|c| c.set(false)); crate::dlog!("[CC96] post-U eob={u_eob} rng={} dif={:x}", msac.rng, msac.dif); }
    if ct48 { crate::dlog!("[CT48] post-U u_eob={u_eob} r={}", msac.rng); }
    if std::env::var("UVCFDIF").is_ok() { crate::dlog!("[UVCFDIF] y={by4} x={bx4} pl=0 eob={u_eob} dif={:x} rng={} cnt={}", msac.dif, msac.rng, msac.cnt); }
    // Off-frame clamp for the cf_ctx splat (dav2d recon_tmpl.c:1284): cells past the chroma
    // plane edge stay 0x40 or a later leaf's skip-ctx fold reads a stale level.
    let (csw, csh) = crate::av2_frame::FRAME.with(|f| {
        let f = f.borrow();
        (((f.iw4 + ssh) >> ssh), ((f.ih4 + ssv) >> ssv))
    });
    let cbw4s = cbw4.min(csw.saturating_sub(cbx4));
    let cbh4s = cbh4.min(csh.saturating_sub(cby4));
    nb.a[0][cbx4..cbx4 + cbw4s].fill(cf_ctx_u);
    nb.l[0][cby4..cby4 + cbh4s].fill(cf_ctx_u);
    // V plane — the offset folds in whether this block's U plane carried coefficients.
    let sctx_v = crate::av2_coef::skip_ctx_chroma(
        &nb.a[1], &nb.l[1], cbx4, cby4, cbw4, cbh4, 2, u_eob != -1, false,
    );
    let mut cf_v = vec![0i32; 1usize << (slw + slh + 4)];
    if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({bx4},{by4}) pl=2 txs={t_dim_ctx} skipctx={sctx_v} rng={}", msac.rng); }
    let (v_eob, cf_ctx_v) = crate::av2_coef::decode_coefs_uv(
        msac, &mut cdf.coef, &mut cdf.m.cctx, &mut cf_v, 2, t_dim_ctx, slw, slh, tx2dszctx, scan, sctx_v, 0, true, 0,
    );
    if ct48 { crate::dlog!("[CT48] sctx_v={sctx_v} post-V v_eob={v_eob} r={}", msac.rng); }
    if std::env::var("UVCFDIF").is_ok() { crate::dlog!("[UVCFDIF] y={by4} x={bx4} pl=1 eob={v_eob} dif={:x} rng={} cnt={}", msac.dif, msac.rng, msac.cnt); }
    nb.a[1][cbx4..cbx4 + cbw4s].fill(cf_ctx_v);
    nb.l[1][cby4..cby4 + cbh4s].fill(cf_ctx_v);
    // C-chroma recon (frame-1 intra): predict + residual into the FRAME chroma planes, score vs REF.
    let (lbw4, lbh4) = (bdc0[0] as usize, bdc0[1] as usize); // luma block dims (for availability)
    recon_intra_chroma(
        cbx4, cby4, bx4, by4, lbw4, lbh4, slw, slh, uv_mode, uv_angle, cfl_mode, cfl_alpha_u, cfl_alpha_v, mh_dir,
        &cf_u, &cf_v, u_eob == -1, v_eob == -1,
        cbx4 > TILE_B.with(|t| t.get().0 >> 1), cby4 > TILE_B.with(|t| t.get().2 >> 1),
    );
    (uv_mode, uv_angle)
}

/// Decode a yuv block's CHROMA MODE only (CFL flag + params, or `intra_uv_mode`) — no coefs.
/// For a yuv (has_luma && has_chroma) inter-frame block, dav2d decodes the chroma mode between
/// the luma mode and `read_coef_blocks`; `midx` is the luma block's directional-mode index.
pub fn decode_chroma_mode(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    cnb: &mut ChromaNb,
    bs: usize,
    bx4: usize,
    by4: usize,
    midx: u8,
) -> ChromaMode {
    use crate::av2_decode::BLOCK_DIMENSIONS;
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bools_bypass,
        rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8,
    };
    const CFL_PRED: u8 = 13;
    let bd = BLOCK_DIMENSIONS[bs];
    let (ssh, ssv) = ss_g();
    let (cbx4, cby4) = (bx4 >> ssh, by4 >> ssv);
    let (bw4c, bh4c) = (bd[0] as usize, bd[1] as usize);
    let (cbw4c, cbh4c) = ((bw4c >> ssh).max(1), (bh4c >> ssv).max(1));
    let cfl_allowed = (bw4c.max(bh4c) > 16 || !cnb.sdp_cfl_disallowed) && cbw4c.max(cbh4c) <= 16;
    let cfl_ctx = (cnb.a_uvmode[cbx4] == CFL_PRED) as usize + (cnb.l_uvmode[cby4] == CFL_PRED) as usize;
    let is_cfl = cfl_allowed && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.cfl[cfl_ctx]);
    let (mut cfl_mode, mut cfl_alpha_u, mut cfl_alpha_v, mut mh_dir) = (0u8, 0i32, 0i32, 0u8);
    let (uv_mode, uv_angle): (u8, i32) = if is_cfl {
        let (cbw4, cbh4) = ((bd[0] as usize >> ssh).max(1), (bd[1] as usize >> ssv).max(1));
        let mhccp_allowed = cbw4.max(cbh4) <= 8 && cbw4 * cbh4 >= 2;
        if std::env::var("MCFL").is_ok() { crate::dlog!("[MCFLSW2] mi=({bx4},{by4}) allowed={} cell={:?} rng={}", mhccp_allowed as u8, cdf.m.mhccp, msac.rng); }
        if mhccp_allowed && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.mhccp) {
            let sz_ctx = SIZE_GROUP[bs] as usize;
            mh_dir = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.mhccp_filter_dir[sz_ctx], 2) as u8;
            cfl_mode = 3;
        } else if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.cfl_type) {
            cfl_mode = 1;
            let sign = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.cfl_sign, 7) as i32 + 1;
            let sign_u = (sign * 0x56) >> 8;
            let sign_v = sign - sign_u * 3;
            if sign_u != 0 {
                let ctx = (sign_u == 2) as usize * 3 + sign_v as usize;
                let mag = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.cfl_alpha[ctx], 7) as i32;
                cfl_alpha_u = if sign_u == 2 { mag + 1 } else { -(mag + 1) };
            }
            if sign_v != 0 {
                let ctx = (sign_v == 2) as usize * 3 + sign_u as usize;
                let mag = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.cfl_alpha[ctx], 7) as i32;
                cfl_alpha_v = if sign_v == 2 { mag + 1 } else { -(mag + 1) };
            }
        } else {
            cfl_mode = 2;
        }
        (CFL_PRED, 0)
    } else {
        let uv_mode_ctx = (midx != 0xff) as usize;
        let mut idx = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.intra_uv_mode[uv_mode_ctx], 7) as usize;
        if idx == 7 {
            idx += rav1d_msac_decode_bools_bypass(msac, 3) as usize;
        }
        if idx < uv_mode_ctx {
            (REORDERED_DIR_Y_MODE[(midx / 7) as usize], (midx % 7) as i32 - 3)
        } else if idx - uv_mode_ctx < 5 {
            (REORDERED_NONDIR_Y_MODE[idx - uv_mode_ctx], 0)
        } else {
            const DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
            const Y_TO_UV: [usize; 8] = [2, 4, 0, 5, 3, 6, 1, 7];
            let mut i = idx - 5 - uv_mode_ctx;
            i += (uv_mode_ctx != 0 && i >= Y_TO_UV[(midx / 7) as usize]) as usize;
            (DEFAULT_MODE_LIST_UV[i.min(7)], 0)
        }
    };
    let (cbw4m, cbh4m) = ((bd[0] as usize >> 1).max(1), (bd[1] as usize >> 1).max(1));
    cnb.a_uvmode[cbx4..cbx4 + cbw4m].fill(uv_mode);
    cnb.l_uvmode[cby4..cby4 + cbh4m].fill(uv_mode);
    ChromaMode { uv_mode, uv_angle, cfl_mode, cfl_alpha_u, cfl_alpha_v, mh_dir }
}

/// The chroma prediction mode + CfL/MHCCP params for a frame-2 intra leaf, from
/// [`decode_chroma_mode`], fed to `recon_intra_chroma`. cfl_mode: 0=directional/DC, 1=CfL-explicit,
/// 2=CfL-derived, 3=MHCCP.
struct ChromaMode {
    uv_mode: u8,
    uv_angle: i32,
    cfl_mode: u8,
    cfl_alpha_u: i32,
    cfl_alpha_v: i32,
    mh_dir: u8,
}

/// Decode the two chroma planes' coefficients (U then V) for a yuv block via `decode_coefs_uv`,
/// updating the chroma coef-neighbour context. Chroma TX = luma TX subsampled (`slw-1, slh-1`).
fn decode_chroma_coefs(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    cnb: &mut ChromaNb,
    bs: usize,
    bx4: usize,
    by4: usize,
    u_skip_set: usize,
    // Block intra flag — gates the U cctx symbol (`eob >= intra`); false for inter leaves.
    intra: bool,
    // Luma txtp for this block. Intra chroma ignores it (txtp comes from a uvmode LUT, always
    // 2D); INTER chroma INHERITS it (dav2d recon 3643/3657), clamped, so the class may be H/V.
    luma_txtp: u8,
) -> ChromaCoefs {
    if std::env::var("HANGCP").is_ok() { crate::dlog!("[CP]   chroma-coefs enter"); }
    let bd = crate::av2_decode::BLOCK_DIMENSIONS[bs];
    let (ssh, ssv) = ss_g();
    let (slw, slh) = ((bd[2] as usize).saturating_sub(ssh), (bd[3] as usize).saturating_sub(ssv));
    let t_dim_ctx = (slw + slh + 1) >> 1;
    // eob/coef size class uses the 32-clamped CORE dims (64-dim TXs code a 32-core;
    // same convention as the luma path's clw/clh).
    let tx2dszctx = slw.min(3) + slh.min(3);
    let scan = crate::av2_tables_gen::SCANS[scan_idx_square(slw, slh)];
    // Chroma txtp: intra → 2D (uvmode LUT); inter → inherit luma txtp with the dav2d clamp
    // (recon_tmpl.c:481-488). For a chroma TX < 32px (cw/ch < 8) the (flip)adst→DCT clamp is a
    // no-op, so small inter chroma keeps the luma's H/V class. `tx_class = (txtp>>3)&3`.
    let uv_txtp = if intra {
        DCT_DCT
    } else {
        let (cw, ch) = (1usize << slw, 1usize << slh); // chroma tx dims in 4px units
        let is_16x16 = slw == 2 && slh == 2;
        if (cw >= 8 && luma_txtp & 0x02 != 0)
            || (ch >= 8 && luma_txtp & 0x40 != 0)
            || (is_16x16 && ((luma_txtp & 0x47) == 0x41 || (luma_txtp & 0xe2) == 0x22))
        {
            DCT_DCT
        } else if luma_txtp == 41 {
            IDTX_TT // IDTX_INV → IDTX
        } else {
            luma_txtp
        }
    };
    let tx_class = ((uv_txtp >> 3) & 3) as usize;
    let (cbx4, cby4) = (bx4 >> ssh, by4 >> ssv);
    let (cbw4, cbh4) = (1usize << slw, 1usize << slh);
    // Off-frame clamp for the cf_ctx SPLAT (dav2d recon_tmpl.c:1284 `imin(x_start+bw4, f->bw)`):
    // a chroma TX at the right/bottom edge extends past the plane; the neighbour-context cells
    // beyond the frame must stay at their 0x40 base so a later block's skip-ctx scan (which reads
    // the FULL tx width, get_skip_ctx recon_tmpl.c:102) doesn't pick up a stale off-frame level.
    let (csw, csh) = crate::av2_frame::FRAME.with(|f| {
        let f = f.borrow();
        (((f.iw4 + ssh) >> ssh), ((f.ih4 + ssv) >> ssv))
    });
    let cbw4s = cbw4.min(csw.saturating_sub(cbx4));
    let cbh4s = cbh4.min(csh.saturating_sub(cby4));
    // C1: record this chroma leaf's deblock edges (RECON_ACTIVE = the scored frame-1 pass).
    if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
        crate::av2_frame::FRAME.with(|f| f.borrow_mut().mark_db_chroma(cbx4, cby4, cbw4, cbh4));
    }
    let sctx_u = crate::av2_coef::skip_ctx_chroma(&cnb.a[0], &cnb.l[0], cbx4, cby4, cbw4, cbh4, 1, false, false);
    let d240 = std::env::var("DBG00").is_ok() && bx4 == 96 && by4 == 32;
    let dcc = std::env::var("CCDBG").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == bx4 && p[1] == by4 });
    if dcc { crate::dlog!("[CC] pre-u bx4={bx4} by4={by4} slw={slw} slh={slh} tx2dszctx={tx2dszctx} sctx_u={sctx_u} tx_class={tx_class} scan[0..8]={:?} rng={} dif={:x}", &scan[..8.min(scan.len())], msac.rng, msac.dif); }
    if d240 { crate::dlog!("CC240 pl=U sctx={sctx_u} tctx={t_dim_ctx} cbw4={cbw4} cbh4={cbh4} cbx4={cbx4} cby4={cby4} a[{cbx4}..{}]={:?} l[{cby4}..{}]={:?} (dav sctx=6)", cbx4+cbw4, &cnb.a[0][cbx4..cbx4+cbw4], cby4+cbh4, &cnb.l[0][cby4..cby4+cbh4]); }
    let mut cf_u = vec![0i32; 1usize << (slw + slh + 4)];
    if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({bx4},{by4}) pl=1 txs={t_dim_ctx} skipctx={sctx_u} rng={}", msac.rng); }
    let (u_eob, cfc_u) = crate::av2_coef::decode_coefs_uv(msac, &mut cdf.coef, &mut cdf.m.cctx, &mut cf_u, 1, t_dim_ctx, slw, slh, tx2dszctx, scan, sctx_u, u_skip_set, intra, tx_class);
    if d240 { crate::dlog!("CC240 pl=U DONE u_eob={u_eob} rng={} dif={:x}", msac.rng, msac.dif); }
    if dcc {
        let nz: Vec<(usize, i32)> = cf_u.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v)).collect();
        crate::dlog!("[CC] post-u u_eob={u_eob} rng={} dif={:x} nz={:?}", msac.rng, msac.dif, nz);
    }
    cnb.a[0][cbx4..cbx4 + cbw4s].fill(cfc_u);
    cnb.l[0][cby4..cby4 + cbh4s].fill(cfc_u);
    let sctx_v = crate::av2_coef::skip_ctx_chroma(&cnb.a[1], &cnb.l[1], cbx4, cby4, cbw4, cbh4, 2, u_eob != -1, false);
    let mut cf_v = vec![0i32; 1usize << (slw + slh + 4)];
    if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({bx4},{by4}) pl=2 txs={t_dim_ctx} skipctx={sctx_v} rng={}", msac.rng); }
    let (_v_eob, cfc_v) = crate::av2_coef::decode_coefs_uv(msac, &mut cdf.coef, &mut cdf.m.cctx, &mut cf_v, 2, t_dim_ctx, slw, slh, tx2dszctx, scan, sctx_v, 0, intra, tx_class);
    cnb.a[1][cbx4..cbx4 + cbw4s].fill(cfc_v);
    cnb.l[1][cby4..cby4 + cbh4s].fill(cfc_v);
    if dcc { crate::dlog!("[CC] post-v v_eob={_v_eob} rng={} dif={:x} cf_v[0..6]={:?}", msac.rng, msac.dif, &cf_v[..6]); }
    if d240 { crate::av2_coef::COEF_DBG.with(|c| c.set(false)); }
    ChromaCoefs { cf_u, cf_v, u_eob, v_eob: _v_eob, uv_txtp, slw, slh }
}

/// The decoded chroma coefficients + inherited txtp for a yuv leaf, returned by
/// [`decode_chroma_coefs`] so the caller (which holds the MC prediction context) can run the
/// Stage-E chroma RECON. `uv_txtp` is dav's PACKED chroma TxfmType (already size/class-clamped).
struct ChromaCoefs {
    cf_u: Vec<i32>,
    cf_v: Vec<i32>,
    u_eob: i32,
    v_eob: i32,
    uv_txtp: u8,
    slw: usize,
    slh: usize,
}

thread_local! {
    /// TIP chroma: plane-U bacp decision reused by plane V (dav `bacpu`).
    pub static TIP_BACP_U: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// dav `t->rmv`: per-8x8 [refined mv (chroma-scaled), pre-refine mv][2 arms] over the
    /// 128px-aligned area (16x16 cells). Written by the TIP luma pred; read by the TIP chroma.
    pub static RMV: std::cell::RefCell<Vec<[[crate::av2_refmvs::Mv; 2]; 2]>> =
        std::cell::RefCell::new(vec![[[crate::av2_refmvs::Mv { y: 0, x: 0 }; 2]; 2]; 256]);
}

/// dav `get_mask` (recon_tmpl.c:1920): OOB check on both arms at `subpel_bits` precision; on OOB
/// lazily 0x20-init the deferred mask then fill the (x4,y4) sub-rect with the bacp boundary mask.
#[allow(clippy::too_many_arguments)]
fn tip_get_mask(
    scr: &mut [u8], stride: usize, bx4: usize, x4: usize, by4: usize, y4: usize,
    mv: [crate::av2_refmvs::Mv; 2], hb: i32, vb: i32, bw4: usize, bh4: usize, iw: i32, ih: i32,
    deferred_sz: usize,
) -> bool {
    let x0 = ((bx4 + x4) * 4) as i32 + (mv[0].x >> hb);
    let y0 = ((by4 + y4) * 4) as i32 + (mv[0].y >> vb);
    let x1 = ((bx4 + x4) * 4) as i32 + (mv[1].x >> hb);
    let y1 = ((by4 + y4) * 4) as i32 + (mv[1].y >> vb);
    if x0 < 0
        || x1 < 0
        || y0 < 0
        || y1 < 0
        || x0 + (bw4 * 4) as i32 >= iw
        || x1 + (bw4 * 4) as i32 >= iw
        || y0 + (bh4 * 4) as i32 >= ih
        || y1 + (bh4 * 4) as i32 >= ih
    {
        if deferred_sz > 0 && scr[0] == 0xff {
            scr[..deferred_sz].fill(0x20);
        }
        crate::av2_inter::bacp_mask(&mut scr[(y4 * stride + x4) * 4..], stride, bw4 * 4, bh4 * 4, x0, y0, x1, y1, iw, ih);
        return true;
    }
    false
}

/// Luma→chroma refined-mv conversion for the RMV grid (the chroma MC consumes 1/16-pel
/// CHROMA-plane units): from a 1/16-pel LUMA mv, subsampled axes halve with dav's
/// scaledown_16pel rounding; non-subsampled axes pass through.
fn rmv_c16(m: crate::av2_refmvs::Mv) -> crate::av2_refmvs::Mv {
    let (ssh, ssv) = ss_g();
    let sd = |v: i32| (v + (v > 0) as i32) >> 1;
    crate::av2_refmvs::Mv {
        y: if ssv != 0 { sd(m.y) } else { m.y },
        x: if ssh != 0 { sd(m.x) } else { m.x },
    }
}
/// Same from a 1/8-pel LUMA mv (dav scaleup_8pel generalized): subsampled axes pass through
/// (1/8 luma == 1/16 half-res chroma), non-subsampled axes double.
fn rmv_c8(m: crate::av2_refmvs::Mv) -> crate::av2_refmvs::Mv {
    let (ssh, ssv) = ss_g();
    crate::av2_refmvs::Mv {
        y: if ssv != 0 { m.y } else { m.y << 1 },
        x: if ssh != 0 { m.x } else { m.x << 1 },
    }
}

/// dav `update_temporal` (recon_tmpl.c:1950): write a refined per-cell compound mv pair into the
/// current frame's temporal field at (bx4,by4) covering bw4mi x bh4mi (t_swap ordering + the
/// INVALID-arm fallbacks). Used by tip_pred / opfl_pred cell writes.
fn update_temporal_cells(bx4: usize, by4: usize, bw4mi: usize, bh4mi: usize, r0: i8, r1: i8, mv: [crate::av2_refmvs::Mv; 2]) {
    use crate::av2_refmvs::{quantize_mv, rp_write, TemporalBlock, INVALID_TRAJ};
    let t_swap = ref_flip_pair(r0, r1) as usize;
    let mut tb = TemporalBlock::default();
    tb.qmv[t_swap] = quantize_mv(mv[0]);
    tb.qmv[1 - t_swap] = quantize_mv(mv[1]);
    let mut r = [0i8; 2];
    r[t_swap] = r0;
    r[1 - t_swap] = r1;
    tb.ref_ = (r[0], r[1]);
    if tb.qmv[0] == INVALID_TRAJ {
        if tb.qmv[1] == INVALID_TRAJ {
            tb.ref_ = (-1, -1);
        } else {
            tb.qmv[0] = tb.qmv[1];
            tb.ref_.0 = tb.ref_.1;
        }
    } else if tb.qmv[1] == INVALID_TRAJ {
        tb.qmv[1] = tb.qmv[0];
        tb.ref_.1 = tb.ref_.0;
    }
    rp_write(bx4, by4, bw4mi, bh4mi, tb);
}

/// dav2d `opfl_pred` (recon_tmpl.c:2222), luma: DMVR (`sad_refine_mv`) and/or optical-flow
/// (`opfl_derive_mv`/`opfl_mv_adj`) refined compound prediction into `tmp[0..2]` (PREP domain),
/// with per-cell RMV stores (chroma), per-cell temporal-field rewrites, and the bacp mask.
/// Returns the final bacp flag for the AVG blend.
#[allow(clippy::too_many_arguments)]
fn opfl_pred_luma(
    tmp: &mut [Vec<i32>; 2], bx4: usize, by4: usize, bw4: usize, bh4: usize, w4: usize, h4: usize,
    mvp: [crate::av2_refmvs::Mv; 2], refs: [usize; 2], filter: u8, cwp: i8, inter_mode: u8,
    refine_mv: u8, comp_type: u8, iw4: usize, ih4: usize,
) -> bool {
    use crate::av2_inter::{opfl_derive_mv, opfl_mv_adj, prep_opfl, put_bilin_win, sad_refine_mv};
    use crate::av2_refmvs::Mv;
    let refine = comp_type == 1 && refine_mv != 0;
    let opfl = inter_mode >= 24;
    let (w_px, h_px) = ((iw4 * 4) as i32, (ih4 * 4) as i32);
    let w = bw4 * 4;
    let (refdist, absd, _f) = CUR_REFDIST.with(|c| c.get());
    let refidx = CUR_FRAME_REFIDX.with(|c| c.get()).1;
    let asign = |v: i32, sg: i32| if sg < 0 { -v } else { v };
    let (d0, d1) = (absd[refs[0]], absd[refs[1]]);
    let di = (asign(1 + (d0 > d1) as i32, -refdist[refs[0]]), asign(1 + (d1 > d0) as i32, refdist[refs[1]]));
    // bs: opfl cell size in mi (dav: 2 - (b->bs == BS_8x8)); 8x8 blocks refine per 4px cell.
    let cbs: usize = if bw4 == 2 && bh4 == 2 { 1 } else { 2 };
    let bacp = cwp == 8; // seq imp_msk_bld=1, no ref scaling
    if bacp {
        SEG_SCRATCH.with(|sc| sc.borrow_mut()[0] = 0xff);
    }
    let mut have_bacp = false;
    let p_stride = (((bw4 + refine as usize * 2) * 4) + 63) & !63;
    // HARDENING: these are LOOP STEPS — a zero step (corrupt/degenerate block dims) spins
    // forever inside the OPFL/refine windows (fuzz timeout class).
    let (sh4, sw4) = (bh4.min(4).max(1), bw4.min(4).max(1));
    let rmv_base = ((by4 & 31) >> 1) * 16 + ((bx4 & 31) >> 1);
    crate::av2_frame::REF_PICS.with(|rp| {
        let pics = rp.borrow();
        let (rp0, rp1) = match (pics[refidx[refs[0]] as usize].as_ref(), pics[refidx[refs[1]] as usize].as_ref()) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };
        let planes = [&rp0[0], &rp1[0]];
        let mut top = [
            (by4 * 4) as i32 + (mvp[0].y >> 3) - 3,
            (by4 * 4) as i32 + (mvp[1].y >> 3) - 3,
        ];
        let mut y = 0usize;
        while y < h4 {
            if !work_tick("opfl.y") { break; }
            let rmv_row = rmv_base + (y >> 1) * 16;
            let mut left = [
                (bx4 * 4) as i32 + (mvp[0].x >> 3) - 3,
                (bx4 * 4) as i32 + (mvp[1].x >> 3) - 3,
            ];
            if refine {
                let mut x = 0usize;
                while x < w4 {
                    if !work_tick("opfl.x") { break; }
                    let mut p = [vec![0i32; p_stride * (sh4 + 2) * 4], vec![0i32; p_stride * (sh4 + 2) * 4]];
                    let wins: [(i32, i32, i32, i32); 2] = std::array::from_fn(|n| (
                        left[n].clamp(0, w_px - 1), (left[n] + (4 * sw4) as i32 + 7).clamp(1, w_px),
                        top[n].clamp(0, h_px - 1), (top[n] + (4 * sh4) as i32 + 7).clamp(1, h_px),
                    ));
                    for n in 0..2 {
                        put_bilin_win(planes[n], &mut p[n], p_stride, ((bx4 + x) * 4) as i32, ((by4 + y) * 4) as i32,
                                      (sw4 + 2) * 4, (sh4 + 2) * 4, mvp[n].y - 32, mvp[n].x - 32, wins[n]);
                    }
                    let (dy, dx) = sad_refine_mv(&p[0], p_stride, &p[1], p_stride, sw4 * 4, sh4 * 4, refine_mv == 2);
                    if std::env::var("RMVREF").is_ok() {
                        crate::dlog!("[DREF] b=({bx4},{by4}) sub=({x},{y}) mvp0=({},{}) mvp1=({},{}) d=({dy},{dx}) p00={} p11={}",
                            mvp[0].y, mvp[0].x, mvp[1].y, mvp[1].x, p[0][2 * p_stride + 2], p[1][2 * p_stride + 2]);
                    }
                    if opfl {
                        // refine+opfl: per 8px cell optical-flow delta on top of the DMVR offset
                        let res = opfl_derive_mv(&p[0][((4 + dy) * p_stride as i32 + 4 + dx) as usize..], p_stride,
                                                 &p[1][((4 - dy) * p_stride as i32 + 4 - dx) as usize..], p_stride,
                                                 sw4 * 4, sh4 * 4, cbs * 4, di);
                        let mut ri = 0usize;
                        let mut by = 0usize;
                        while by < sh4 {
                            if !work_tick("walk") { break; }
                            let mut bx = 0usize;
                            while bx < sw4 {
                                if !work_tick("walk") { break; }
                                let dd = opfl_mv_adj(&res[ri.min(res.len() - 1)], di);
                                ri += 1;
                                let mut mv = [
                                    Mv { y: mvp[0].y * 2 + dd[0].1 + dy * 16, x: mvp[0].x * 2 + dd[0].0 + dx * 16 },
                                    Mv { y: mvp[1].y * 2 + dd[1].1 - dy * 16, x: mvp[1].x * 2 + dd[1].0 - dx * 16 },
                                ];
                                for i in 0..2 {
                                    prep_opfl(planes[i], &mut tmp[i][((y + by) * w + x + bx) * 4..], w,
                                              ((bx4 + x + bx) * 4) as i32, ((by4 + y + by) * 4) as i32,
                                              cbs * 4, cbs * 4, mv[i].y, mv[i].x, filter as usize,
                                              (left[i].clamp(0, w_px - 1), (left[i] + (sw4 * 4) as i32 + 7).clamp(1, w_px),
                                               top[i].clamp(0, h_px - 1), (top[i] + (sh4 * 4) as i32 + 7).clamp(1, h_px)));
                                }
                                if std::env::var("OPFLDBG").is_ok() && ((bx4 == 83 && by4 == 10) || (bx4 == 81 && by4 == 10) || (bx4 == 12 && by4 == 0)) {
                                    crate::dlog!("[MOPFL] cell=({},{}) dydx=({dy},{dx}) mv0=({},{}) mv1=({},{})",
                                        (bx4 + x + bx) * 4, (by4 + y + by) * 4, mv[0].y, mv[0].x, mv[1].y, mv[1].x);
                                }
                                let dmv = [
                                    Mv { y: (mv[0].y + (dd[0].1 > 0) as i32) >> 1, x: (mv[0].x + (dd[0].0 > 0) as i32) >> 1 },
                                    Mv { y: (mv[1].y + (dd[1].1 > 0) as i32) >> 1, x: (mv[1].x + (dd[1].0 > 0) as i32) >> 1 },
                                ];
                                update_temporal_cells(bx4 + x + bx, by4 + y + by, 2, 2, refs[0] as i8, refs[1] as i8, dmv);
                                if bacp {
                                    SEG_SCRATCH.with(|sc| {
                                        let mut scr = sc.borrow_mut();
                                        have_bacp |= tip_get_mask(&mut scr, w, bx4, x + bx, by4, y + by, mv, 4, 4, 2, 2, w_px, h_px, bw4 * bh4 * 16);
                                    });
                                }
                                // scaledown_16pel (per-axis, ss-general) -> RMV[cell][0]
                                for m in mv.iter_mut() {
                                    *m = rmv_c16(*m);
                                }
                                RMV.with(|rm| rm.borrow_mut()[rmv_row + ((by > 0) as usize) * 16 + ((x + bx) >> 1)][0] = mv);
                                bx += 2;
                            }
                            by += 2;
                        }
                    } else {
                        // refine only: block mv +- the DMVR offset (1/8-pel), regular windowed prep
                        let mut mv = [
                            Mv { y: mvp[0].y + dy * 8, x: mvp[0].x + dx * 8 },
                            Mv { y: mvp[1].y + dy * -8, x: mvp[1].x + dx * -8 },
                        ];
                        for i in 0..2 {
                            prep_opfl(planes[i], &mut tmp[i][(y * w + x) * 4..], w,
                                      ((bx4 + x) * 4) as i32, ((by4 + y) * 4) as i32, sw4 * 4, sh4 * 4,
                                      mv[i].y * 2, mv[i].x * 2, filter as usize,
                                      (left[i].clamp(0, w_px - 1), (left[i] + (sw4 * 4) as i32 + 7).clamp(1, w_px),
                                       top[i].clamp(0, h_px - 1), (top[i] + (sh4 * 4) as i32 + 7).clamp(1, h_px)));
                        }
                        update_temporal_cells(bx4 + x, by4 + y, sw4, sh4, refs[0] as i8, refs[1] as i8, mv);
                        if bacp {
                            SEG_SCRATCH.with(|sc| {
                                let mut scr = sc.borrow_mut();
                                have_bacp |= tip_get_mask(&mut scr, w, bx4, x, by4, y, mv, 3, 3, sw4, sh4, w_px, h_px, bw4 * bh4 * 16);
                            });
                        }
                        // scaleup_8pel (per-axis, ss-general) -> RMV[cell][0]
                        RMV.with(|rm| rm.borrow_mut()[rmv_row + (x >> 1)][0] = [rmv_c8(mv[0]), rmv_c8(mv[1])]);
                        let _ = &mut mv;
                    }
                    for l in left.iter_mut() {
                        *l += 16;
                    }
                    x += sw4;
                }
            } else {
                // opfl only: whole-row bilinear pre-MC (full-frame window), per-cbs-cell flow
                let mut p = [vec![0i32; p_stride * (sh4 * 4 + 8)], vec![0i32; p_stride * (sh4 * 4 + 8)]];
                for n in 0..2 {
                    put_bilin_win(planes[n], &mut p[n], p_stride, (bx4 * 4) as i32, ((by4 + y) * 4) as i32,
                                  bw4 * 4, sh4 * 4, mvp[n].y, mvp[n].x, (0, w_px, 0, h_px));
                }
                let res = opfl_derive_mv(&p[0], p_stride, &p[1], p_stride, bw4 * 4, sh4 * 4, cbs * 4, di);
                // res regions: rows of (bw4*4)/(cbs*4) regions each (opfl_derive_mv returns max 4;
                // recompute per row like dav's r_line stepping)
                let mut ddsum = [[0i32; 2]; 2]; // [arm][x,y] accumulated for cbs==1
                let mut first_mv: [Mv; 2] = [Mv { y: 0, x: 0 }; 2];
                let mut ri_row = 0usize;
                let mut by = 0usize;
                while by < sh4 {
                    if !work_tick("walk") { break; }
                    let mut ri = ri_row;
                    let mut bx = 0usize;
                    let mut xx = 0usize;
                    while bx < w4 {
                        if !work_tick("walk") { break; }
                        let dd = opfl_mv_adj(&res[ri.min(res.len() - 1)], di);
                        ri += 1;
                        let mut mv = [
                            Mv { y: mvp[0].y * 2 + dd[0].1, x: mvp[0].x * 2 + dd[0].0 },
                            Mv { y: mvp[1].y * 2 + dd[1].1, x: mvp[1].x * 2 + dd[1].0 },
                        ];
                        for i in 0..2 {
                            prep_opfl(planes[i], &mut tmp[i][((y + by) * w + bx) * 4..], w,
                                      ((bx4 + bx) * 4) as i32, ((by4 + y + by) * 4) as i32,
                                      cbs * 4, cbs * 4, mv[i].y, mv[i].x, filter as usize,
                                      ((left[i] + (bx * 4) as i32).clamp(0, w_px - 1), (left[i] + (bx * 4) as i32 + 7 + 8).clamp(1, w_px),
                                       (top[i] + (by * 4) as i32).clamp(0, h_px - 1), (top[i] + (by * 4) as i32 + 7 + 8).clamp(1, h_px)));
                        }
                        if cbs > 1 {
                            let dmv = [
                                Mv { y: (mv[0].y + (dd[0].1 > 0) as i32) >> 1, x: (mv[0].x + (dd[0].0 > 0) as i32) >> 1 },
                                Mv { y: (mv[1].y + (dd[1].1 > 0) as i32) >> 1, x: (mv[1].x + (dd[1].0 > 0) as i32) >> 1 },
                            ];
                            update_temporal_cells(bx4 + bx, by4 + y + by, cbs, cbs, refs[0] as i8, refs[1] as i8, dmv);
                        } else {
                            ddsum[0][0] += dd[0].0;
                            ddsum[0][1] += dd[0].1;
                            ddsum[1][0] += dd[1].0;
                            ddsum[1][1] += dd[1].1;
                        }
                        if bacp {
                            SEG_SCRATCH.with(|sc| {
                                let mut scr = sc.borrow_mut();
                                have_bacp |= tip_get_mask(&mut scr, w, bx4, bx, by4, y + by, mv, 4, 4, cbs, cbs, w_px, h_px, bw4 * bh4 * 16);
                            });
                        }
                        for m in mv.iter_mut() {
                            *m = rmv_c16(*m);
                        }
                        // for 8x8 (cbs==1) only the first (top-left) cell is stored
                        if cbs > 1 || (bx == 0 && by == 0) {
                            RMV.with(|rm| rm.borrow_mut()[rmv_row + ((by > 0) as usize) * 16 + xx][0] = mv);
                        }
                        if cbs == 1 && bx == 0 && by == 0 {
                            first_mv = mv;
                        }
                        let _ = first_mv;
                        bx += cbs;
                        xx += 1;
                    }
                    ri_row += if cbs == 2 { bw4 >> 1 } else { bw4 };
                    by += cbs;
                }
                if cbs == 1 {
                    // 8x8: one averaged temporal write (dav recon 2404)
                    let avg = |base: i32, sum: i32| (base * 8 + sum + 3 + (sum > 0) as i32) >> 3;
                    let dmv = [
                        Mv { y: avg(mvp[0].y, ddsum[0][1]), x: avg(mvp[0].x, ddsum[0][0]) },
                        Mv { y: avg(mvp[1].y, ddsum[1][1]), x: avg(mvp[1].x, ddsum[1][0]) },
                    ];
                    update_temporal_cells(bx4, by4 + y, 2, 2, refs[0] as i8, refs[1] as i8, dmv);
                }
            }
            for t in top.iter_mut() {
                *t += (4 * sh4) as i32;
            }
            y += sh4;
        }
    });
    bacp && have_bacp
}

/// dav2d `tip_pred` (recon_tmpl.c:2059), luma: per-8x8 rp_proj TIP MVs; on the opfl path a
/// bilinear pre-MC + integer DMVR (`sad_refine_mv`) + optical-flow delta (`opfl_derive_mv` /
/// `opfl_mv_adj`); dual 1/16-pel windowed prep MC (`mc_opfl`); avg (or bacp-mask) blend.
/// Stores the per-cell mv pair into RMV (chroma-scaled slot 0, pre-refine slot 1).
/// Returns the TIP step used (for the chroma path).
#[allow(clippy::too_many_arguments)]
fn tip_pred_luma(
    luma_pred: &mut [i32], bx4: usize, by4: usize, bw4: usize, bh4: usize, w4: usize, h4: usize,
    block_mv: crate::av2_refmvs::Mv, filter: u8, iw4: usize, ih4: usize,
) -> usize {
    use crate::av2_inter::{comp_avg, comp_mask, opfl_derive_mv, opfl_mv_adj, prep_opfl, put_bilin_win, sad8x8, sad_refine_mv};
    use crate::av2_refmvs::{scale_mv, Mv, RP_PROJ, TMVS};
    let t = TMVS.with(|c| c.borrow().clone());
    let (_seq_tip, _hole, seq_opfl, _seq_refine, seq_tip_refine) = SEQ_TIP.with(|c| c.get());
    let frame_mode = HDR_TOOL_CFG.with(|c| c.get()).tip_frame_mode;
    let (refdist, absd, _ffr) = CUR_REFDIST.with(|c| c.get());
    let (n_ref, refidx) = CUR_FRAME_REFIDX.with(|c| c.get());
    let tr = [t.tip_ref.0 as usize, t.tip_ref.1 as usize];
    // opfl / refine / step (dav 848-857 incl. the frame_mode==2 TIP-as-output arms;
    // BS_256 not reachable — 64px SBs).
    let tip_subpel = HDR_TOOL_CFG.with(|c| c.get().tip_subpel_filter);
    let mut opfl = seq_tip_refine && (frame_mode == 1 || tip_subpel == 2);
    let refine = opfl && frame_mode == 1 && refdist[tr[0]] == -refdist[tr[1]];
    let step: usize = if frame_mode == 2 {
        2 << (!opfl as usize)
    } else {
        2 << ((!opfl && bw4.min(bh4) >= 4) as usize)
    };
    let (mut fut, mut past) = (false, false);
    for i in 0..n_ref as usize {
        fut |= refdist[i] > 0;
        past |= refdist[i] < 0;
    }
    opfl &= seq_opfl != 0 && fut && past;
    let (w_px, h_px) = ((iw4 * 4) as i32, (ih4 * 4) as i32);
    let (w, h) = (bw4 * 4, bh4 * 4);
    let mut tmp = [vec![0i32; w * h], vec![0i32; w * h]];
    // bacp deferred marker (seq imp_msk_bld=1, TIP cwp==8, no scaling)
    let bacp = true;
    SEG_SCRATCH.with(|sc| sc.borrow_mut()[0] = 0xff);
    let mut have_bacp = false;
    // signed distance weights d (dav 2090-2092)
    let (d0, d1) = (absd[tr[0]], absd[tr[1]]);
    let asign = |v: i32, s: i32| if s < 0 { -v } else { v };
    let di = (asign(1 + (d0 > d1) as i32, -refdist[tr[0]]), asign(1 + (d1 > d0) as i32, refdist[tr[1]]));
    let stride = t.stride;
    let sad8x8_thr: u32 = if frame_mode == 1 { 6 } else { 15 };
    let rmv_base = ((by4 & 31) >> 1) * 16 + ((bx4 & 31) >> 1);
    crate::av2_frame::REF_PICS.with(|rp| {
        let pics = rp.borrow();
        let refp = [
            pics[refidx[tr[0]] as usize].as_ref(),
            pics[refidx[tr[1]] as usize].as_ref(),
        ];
        let (rp0, rp1) = match (refp[0], refp[1]) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };
        let planes = [&rp0[0], &rp1[0]];
        let mut y = 0usize;
        while y < h4 {
            if !crate::av2_recon::work_tick("av2_recon:2955") { break; }
            let mut x = 0usize;
            while x < w4 {
                if !crate::av2_recon::work_tick("av2_recon:2957") { break; }
                let off_8x8 = 2 * stride + ((((by4 + y) & (sb_step4() - 1)) >> 1) * stride) + ((bx4 + x) >> 1);
                let mut tmv = RP_PROJ.with(|c| c.borrow()[off_8x8].0);
                if tmv.y == crate::av2_refmvs::INVALID_MV_I32 {
                    tmv = Mv { y: 0, x: 0 };
                }
                let mut cmv = [Mv { y: 0, x: 0 }; 2];
                let mut top = [0i32; 2];
                let mut left = [0i32; 2];
                for i in 0..2 {
                    let tipmv = scale_mv(tmv, t.tip_sf[i]);
                    cmv[i] = Mv {
                        y: (tipmv.y + block_mv.y).clamp(-0xffff, 0xffff),
                        x: (tipmv.x + block_mv.x).clamp(-0xffff, 0xffff),
                    };
                    top[i] = ((by4 + y) * 4) as i32 + (cmv[i].y >> 3) - 3;
                    left[i] = ((bx4 + x) * 4) as i32 + (cmv[i].x >> 3) - 3;
                }
                let cell = rmv_base + (y >> 1) * 16 + (x >> 1);
                if std::env::var("TIPDBG").is_ok() && bx4 == 96 && by4 == 32 {
                    crate::dlog!("[MTIP] cell=({},{}) tmv=({},{}) blkmv=({},{}) cmv0=({},{}) cmv1=({},{}) sf={:?}",
                        (bx4 + x) * 4, (by4 + y) * 4, tmv.y, tmv.x, block_mv.y, block_mv.x,
                        cmv[0].y, cmv[0].x, cmv[1].y, cmv[1].x, t.tip_sf);
                }
                RMV.with(|rm| {
                    let mut rm = rm.borrow_mut();
                    rm[cell][1] = [rmv_c8(cmv[0]), rmv_c8(cmv[1])]; // scaleup_8pel per-axis
                });
                let wins: [(i32, i32, i32, i32); 2] = [
                    (left[0].clamp(0, w_px - 1), (left[0] + 7 + (step * 4) as i32).clamp(1, w_px),
                     top[0].clamp(0, h_px - 1), (top[0] + 7 + (step * 4) as i32).clamp(1, h_px)),
                    (left[1].clamp(0, w_px - 1), (left[1] + 7 + (step * 4) as i32).clamp(1, w_px),
                     top[1].clamp(0, h_px - 1), (top[1] + 7 + (step * 4) as i32).clamp(1, h_px)),
                ];
                if opfl {
                    // bilinear pre-MC: (step+2)*4 square at 1/8-pel, mv - 32 (4px up-left)
                    let pw = (step + 2) * 4;
                    let mut p = [vec![0i32; 64 * pw], vec![0i32; 64 * pw]];
                    for i in 0..2 {
                        put_bilin_win(planes[i], &mut p[i], 64, ((bx4 + x) * 4) as i32, ((by4 + y) * 4) as i32,
                                      pw, pw, cmv[i].y - 32, cmv[i].x - 32, wins[i]);
                    }
                    let (dy, dx) = if refine {
                        let (dy, dx) = sad_refine_mv(&p[0], 64, &p[1], 64, step * 4, step * 4, true);
                        cmv[0].y += 8 * dy;
                        cmv[0].x += 8 * dx;
                        cmv[1].y -= 8 * dy;
                        cmv[1].x -= 8 * dx;
                        (dy, dx)
                    } else {
                        (0, 0)
                    };
                    let sad = sad8x8(&p[0][((4 + dy) * 64 + 4 + dx) as usize..], 64,
                                     &p[1][((4 - dy) * 64 + 4 - dx) as usize..], 64);
                    let dd = if sad >= sad8x8_thr {
                        let res = opfl_derive_mv(&p[0][((4 + dy) * 64 + 4 + dx) as usize..], 64,
                                                 &p[1][((4 - dy) * 64 + 4 - dx) as usize..], 64,
                                                 step * 4, step * 4, 8, di);
                        opfl_mv_adj(&res[0], di)
                    } else {
                        [(0, 0); 2]
                    };
                    cmv[0] = Mv { y: cmv[0].y * 2 + dd[0].1, x: cmv[0].x * 2 + dd[0].0 };
                    cmv[1] = Mv { y: cmv[1].y * 2 + dd[1].1, x: cmv[1].x * 2 + dd[1].0 };
                    if std::env::var("TIPDBG").is_ok() && bx4 == 96 && by4 == 16 {
                        crate::dlog!("[MTIP2] cell=({},{}) sad={sad} thr={sad8x8_thr} dydx=({dy},{dx}) dd={dd:?} cmvR0=({},{}) cmvR1=({},{})",
                            (bx4 + x) * 4, (by4 + y) * 4, cmv[0].y, cmv[0].x, cmv[1].y, cmv[1].x);
                    }
                    for i in 0..2 {
                        prep_opfl(planes[i], &mut tmp[i][y * 4 * w + x * 4..], w,
                                  ((bx4 + x) * 4) as i32, ((by4 + y) * 4) as i32, step * 4, step * 4,
                                  cmv[i].y, cmv[i].x, filter as usize, wins[i]);
                    }
                    // update_temporal (dav tip_pred): refined per-cell dmv, tip source ref pair.
                    {
                        let dmv = [
                            Mv { y: (cmv[0].y + (dd[0].1 > 0) as i32) >> 1, x: (cmv[0].x + (dd[0].0 > 0) as i32) >> 1 },
                            Mv { y: (cmv[1].y + (dd[1].1 > 0) as i32) >> 1, x: (cmv[1].x + (dd[1].0 > 0) as i32) >> 1 },
                        ];
                        update_temporal_cells(bx4 + x, by4 + y, step, step, tr[0] as i8, tr[1] as i8, dmv);
                    }
                    if bacp {
                        SEG_SCRATCH.with(|sc| {
                            let mut scr = sc.borrow_mut();
                            have_bacp |= tip_get_mask(&mut scr, w, bx4, x, by4, y, cmv, 4, 4, step, step, w_px, h_px, w * h);
                        });
                    }
                    // scaledown_16pel per-axis (ss-general)
                    RMV.with(|rm| {
                        let mut rm = rm.borrow_mut();
                        rm[cell][0] = [rmv_c16(cmv[0]), rmv_c16(cmv[1])];
                    });
                } else {
                    // non-opfl: regular 1/8-pel prep, full-frame window (mv doubled -> same fracs)
                    for i in 0..2 {
                        prep_opfl(planes[i], &mut tmp[i][y * 4 * w + x * 4..], w,
                                  ((bx4 + x) * 4) as i32, ((by4 + y) * 4) as i32, step * 4, step * 4,
                                  cmv[i].y * 2, cmv[i].x * 2, filter as usize, (0, w_px, 0, h_px));
                    }
                    if bacp {
                        SEG_SCRATCH.with(|sc| {
                            let mut scr = sc.borrow_mut();
                            have_bacp |= tip_get_mask(&mut scr, w, bx4, x, by4, y, cmv, 3, 3, step, step, w_px, h_px, w * h);
                        });
                    }
                    // update_temporal (dav tip_pred non-opfl arm) + the mode-1/step-4 per-8x8 extras
                    update_temporal_cells(bx4 + x, by4 + y, step, step, tr[0] as i8, tr[1] as i8, cmv);
                    if step == 4 && frame_mode == 1 {
                        for pcell in 1..4usize {
                            let off2 = off_8x8 + (pcell & 1) + ((pcell & 2) >> 1) * stride;
                            let mut tmv2 = RP_PROJ.with(|c| c.borrow()[off2].0);
                            if tmv2.y == crate::av2_refmvs::INVALID_MV_I32 {
                                tmv2 = Mv { y: 0, x: 0 };
                            }
                            let mut dmv = [Mv { y: 0, x: 0 }; 2];
                            for i in 0..2 {
                                let tipmv = scale_mv(tmv2, t.tip_sf[i]);
                                dmv[i] = Mv {
                                    y: (tipmv.y + block_mv.y).clamp(-0xffff, 0xffff),
                                    x: (tipmv.x + block_mv.x).clamp(-0xffff, 0xffff),
                                };
                            }
                            update_temporal_cells(bx4 + x + (pcell & 1) * 2, by4 + y + ((pcell & 2) >> 1) * 2, 2, 2, tr[0] as i8, tr[1] as i8, dmv);
                        }
                    }
                    // scaleup_8pel per-axis; rmv[0] = the mv pair in 1/16 chroma units
                    RMV.with(|rm| rm.borrow_mut()[cell][0] = [rmv_c8(cmv[0]), rmv_c8(cmv[1])]);
                }
                x += step;
            }
            y += step;
        }
    });
    if bacp && have_bacp {
        SEG_SCRATCH.with(|sc| {
            let scr = sc.borrow();
            comp_mask(luma_pred, w, &tmp[0], &tmp[1], w, h, &scr, bdmax_g());
        });
    } else {
        comp_avg(luma_pred, w, &tmp[0], &tmp[1], w, h, bdmax_g());
    }
    step
}

/// Stage-E inter chroma RECON for a grid-aligned yuv leaf: build the chroma MC prediction (U then
/// V) with the same dispatch as luma (translational / warp8x8 / ext_warp, ss=1), add the
/// dequantized inverse-transformed residual (if the block is not skipped), and score each plane vs
/// dav's `dav_f2reconc` oracle. Chroma qindex = luma yac (uac/vac deltas 0 this clip, as intra).
#[allow(clippy::too_many_arguments)]
fn recon_inter_chroma(
    bx4: usize, by4: usize, cpx: usize, cpy: usize, cw: usize, ch: usize, fmv: crate::av2_refmvs::Mv,
    warp_pred: Option<[i32; 6]>, filter: u8, forced: bool, cc: Option<&ChromaCoefs>,
    // BAWP: when Some(luma_alpha), morph the chroma MC prediction (reusing the luma alpha) before
    // the residual (dav recon 3724). None = a normal (non-BAWP) inter chroma block.
    bawp_alpha: Option<i32>,
    // Compound: (mv pair, list-index refs, comp_type, cwp, mask_sign, 2x2 seg mask + its stride).
    comp: Option<([crate::av2_refmvs::Mv; 2], [usize; 2], u8, i8, bool, Vec<u8>, usize, i32, bool)>,
    // rmv-driven chroma (dav rmv_uvpred, recon 2423): (r_step, o_step, is_tip, frame iw4, ih4).
    // TIP blocks and opfl/refine compound blocks both take this path (reads RMV).
    tip: Option<(usize, usize, bool, usize, usize)>,
    // Inter-intra blend mode (-1 = off): dav chroma iiblend (recon 3731), after MC, else-of-bawp.
    ii_mode: i8,
) {
    // `bx4`/`by4` = LUMA block pos (for the warp matrix eval); `cpx`/`cpy`/`cw`/`ch` = the chroma
    // block's pixel origin + dims (for a sub-8×8 shared chroma these come from the 8×8 carrier).
    use crate::av2_frame::{INTER_SCORE_RC, REF_F2RECONC, REF_FRAME1};
    if std::env::var("ICHR").is_ok() {
        let path = if tip.is_some() { "tip" } else if comp.is_some() { "comp" } else if forced { "forced" } else if warp_pred.is_some() { "warp" } else { "single" };
        let extra = comp.as_ref().map(|c| format!(" mv0=({},{}) mv1=({},{}) refs={:?} ct={} cwp={}", c.0[0].y, c.0[0].x, c.0[1].y, c.0[1].x, c.1, c.2, c.3)).unwrap_or_default();
        crate::dlog!("[ICHR] cpx={cpx} cpy={cpy} cw={cw} ch={ch} path={path} fmv=({},{}){extra}", fmv.y, fmv.x);
    }
    let yac = LAST_QIDX.with(|c| c.get()); // delta-q: the current SB's effective qindex (== frame yac when delta-q is off)
    let inter_ddt = crate::av2_frame::INTER_DDT.with(|c| c.get());
    let (mut oku, mut okv) = (true, true);
    let mut scored = false;
    REF_FRAME1.with(|rf| {
        REF_F2RECONC.with(|rr| {
            let (rfb, rrb) = (rf.borrow(), rr.borrow());
            for pl in 0..2 {
                // The MC reference is REQUIRED; the dav pre-filter capture (`rec`) is a
                // PROBE-ONLY scoring input — recon must run STANDALONE without it. (This gate
                // once required both, silently skipping ALL inter chroma when the capture file
                // was absent/stale — the decoder was never file-free until this decoupling.)
                let r = match rfb[pl + 1].as_ref() {
                    Some(r) => r,
                    None => continue,
                };
                let rec = rrb[pl].as_ref();
                // Prediction (chroma ss=1). For a `forced` (cbs≠lbs) sub-8×8 shared chroma, dav
                // (recon_tmpl.c:3670) predicts PER LUMA SUB-BLOCK — each 4px cell's chroma region
                // uses that sub-block's own mv/bs (read from the refmvs GRID), translational. The
                // non-forced case mirrors the luma dispatch (translational / warp8x8 / ext_warp).
                let mut buf = vec![0i32; cw * ch];
                if let Some(&(r_step, o_step, is_tip, _iw4, ih4)) = tip.as_ref() {
                    // ===== rmv-driven chroma (dav rmv_uvpred, recon_tmpl.c:2423): per-cell refined
                    // mv pairs from RMV; dual windowed 1/16-pel prep MC per ref; avg or bacp-mask
                    // blend (mask computed on plane U, reused on V). Serves TIP blocks (refs = the
                    // tip source pair, windows from rmv[1]) and opfl/refine compound blocks
                    // (refs/window-mvs = the block's pair). =====
                    use crate::av2_inter::{comp_avg, comp_mask, prep_opfl};
                    let refidx = CUR_FRAME_REFIDX.with(|c| c.get()).1;
                    let (trf, wmv, bcwp): ([usize; 2], [crate::av2_refmvs::Mv; 2], i8) = if is_tip {
                        let t = crate::av2_refmvs::TMVS.with(|c| c.borrow().clone());
                        ([t.tip_ref.0 as usize, t.tip_ref.1 as usize], [crate::av2_refmvs::Mv { y: 0, x: 0 }; 2], 8)
                    } else {
                        let &(mvp, refs, _ct, cwp, _ms, ref _sg, _mstr, _wi, _ws) = comp.as_ref().unwrap();
                        // block mvs are 1/8 LUMA; the window calc reads 1/16 CHROMA units.
                        (refs, [rmv_c8(mvp[0]), rmv_c8(mvp[1])], cwp)
                    };
                    // Chroma px per luma-mi, per axis (420=(2,2), 422=(2,4)).
                    let (pxw, pxh) = { let ss = ss_g(); (4usize >> ss.0, 4usize >> ss.1) };
                    let (cbw4, cbh4) = (cw / pxw, ch / pxh);
                    let (cbx, cby) = (cpx / pxw, cpy / pxh);
                    let (fw_c, fh_c) = (r.w as i32, r.h as i32);
                    // HARDENING: these are LOOP STEPS (bx += ow4, x += rw4). A degenerate chroma
                    // block (cw/ch == 0 from corrupt geometry) makes them 0 and the walk never
                    // advances — the fuzz HANG class. A zero-size block has no work anyway.
                    let (rw4, rh4) = (cbw4.min(r_step).max(1), cbh4.min(r_step).max(1));
                    let (ow4, oh4) = (cbw4.min(o_step).max(1), cbh4.min(o_step).max(1));
                    // taps threshold is on the PIXEL extent (mc d>4 ⇒ 8-tap): rw4 is in luma-mi
                    // units, so scale by the per-axis chroma px (420: rw4>2 ⇔ >4px, same as before).
                    let hhtaps = 2 + 2 * ((rw4 * pxw) > 4) as i32;
                    let hvtaps = 2 + 2 * ((rh4 * pxh) > 4) as i32;
                    let h4c = cbh4.min(ih4.saturating_sub(cby));
                    let w4c = cbw4.min(_iw4.saturating_sub(cbx));
                    let bacp = pl == 0 && bcwp == 8; // seq imp_msk_bld=1
                    if bacp {
                        SEG_SCRATCH.with(|sc| sc.borrow_mut()[0] = 0xff);
                    }
                    let mut have_bacp = false;
                    let mut tc = [vec![0i32; cw * ch], vec![0i32; cw * ch]];
                    crate::av2_frame::REF_PICS.with(|rp| {
                        let pics = rp.borrow();
                        let (p0, p1) = match (pics[refidx[trf[0]] as usize].as_ref(), pics[refidx[trf[1]] as usize].as_ref()) {
                            (Some(a), Some(b)) => (a, b),
                            _ => return,
                        };
                        let planes = [&p0[pl + 1], &p1[pl + 1]];
                        let rmv_base = ((cby & 31) >> 1) * 16 + ((cbx & 31) >> 1);
                        let mut uvoff = 0usize;
                        let mut y = 0usize;
                        while y < h4c {
                            if !crate::av2_recon::work_tick("av2_recon:3198") { break; }
                            let mut x = 0usize;
                            while x < w4c {
                                if !crate::av2_recon::work_tick("av2_recon:3200") { break; }
                                // window per r-tile: from rmv[1] (tip) or the block mv pair (opfl)
                                let rmv1 = if is_tip {
                                    RMV.with(|rm| rm.borrow()[rmv_base + (y >> 1) * 16 + (x >> 1)][1])
                                } else {
                                    wmv
                                };
                                let mut win = [(0i32, 0i32, 0i32, 0i32); 2];
                                for i in 0..2 {
                                    let mut topv = ((cby + y) * pxh) as i32 + (rmv1[i].y >> 4);
                                    let mut leftv = ((cbx + x) * pxw) as i32 + (rmv1[i].x >> 4);
                                    let bottomv = (topv + (rh4 * pxh) as i32 + hvtaps).clamp(1, fh_c);
                                    let rightv = (leftv + (rw4 * pxw) as i32 + hhtaps).clamp(1, fw_c);
                                    topv = (topv + 1 - hvtaps).clamp(0, fh_c - 1);
                                    leftv = (leftv + 1 - hhtaps).clamp(0, fw_c - 1);
                                    win[i] = (leftv, rightv, topv, bottomv);
                                }
                                let mut uvoffi = uvoff;
                                let mut by = 0usize;
                                while by < rh4 {
                                    if !work_tick("walk") { break; }
                                    let mut bx = 0usize;
                                    while bx < rw4 {
                                        if !crate::av2_recon::work_tick("av2_recon:3222") { break; }
                                        let cell = rmv_base + (((y + by) >> 1) * 16) + ((x + bx) >> 1);
                                        let rmv0 = RMV.with(|rm| rm.borrow()[cell][0]);
                                        if std::env::var("MRMVC").is_ok() && cby == 0 && pl == 0 {
                                            crate::dlog!("[MRMVC] cell=({},{}) rmv0=[({},{}),({},{})] win0={:?} win1={:?} steps r={r_step} o={o_step}",
                                                x + bx, y + by, rmv0[0].y, rmv0[0].x, rmv0[1].y, rmv0[1].x, win[0], win[1]);
                                        }
                                        for i in 0..2 {
                                            prep_opfl(planes[i], &mut tc[i][uvoffi + (x + bx) * pxw..], cw,
                                                      ((cbx + x + bx) * pxw) as i32, ((cby + y + by) * pxh) as i32,
                                                      ow4 * pxw, oh4 * pxh,
                                                      rmv0[i].y, rmv0[i].x, filter as usize, win[i]);
                                        }
                                        if std::env::var("MRMVC").is_ok() && cby == 0 && pl == 0 {
                                            let o = uvoffi + (x + bx) * pxw;
                                            crate::dlog!("[MRMVC2] cell=({},{}) tmp0={},{} tmp1={},{}", x + bx, y + by, tc[0][o], tc[0][o + 1], tc[1][o], tc[1][o + 1]);
                                        }
                                        if bacp {
                                            SEG_SCRATCH.with(|sc| {
                                                let mut scr = sc.borrow_mut();
                                                have_bacp |= tip_get_mask(&mut scr, cw, (cbx * pxw) / 4, ((x + bx) * pxw) / 4, (cby * pxh) / 4, ((y + by) * pxh) / 4,
                                                                          rmv0, 4, 4, (ow4 * pxw) / 4, (oh4 * pxh) / 4, fw_c, fh_c, cw * ch);
                                            });
                                        }
                                        bx += ow4;
                                    }
                                    uvoffi += oh4 * pxh * cw;
                                    by += oh4;
                                }
                                x += rw4;
                            }
                            uvoff += rh4 * pxh * cw;
                            y += rh4;
                        }
                    });
                    let bacp_fin = if pl == 0 {
                        let v = bacp && have_bacp;
                        TIP_BACP_U.with(|c| c.set(v));
                        v
                    } else {
                        TIP_BACP_U.with(|c| c.get())
                    };
                    if bacp_fin {
                        SEG_SCRATCH.with(|sc| {
                            let scr = sc.borrow();
                            comp_mask(&mut buf, cw, &tc[0], &tc[1], cw, ch, &scr, bdmax_g());
                        });
                    } else if bcwp == 8 {
                        comp_avg(&mut buf, cw, &tc[0], &tc[1], cw, ch, bdmax_g());
                    } else {
                        crate::av2_inter::comp_w_avg(&mut buf, cw, &tc[0], &tc[1], cw, ch, bcwp as i32, bdmax_g());
                    }
                } else if let Some(&(mvp, refs, ct, cwp, msign, ref segm, mstride, widx, wsign)) = comp.as_ref().filter(|_| !forced) {
                    let icdbg = std::env::var("ICDBG").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == cpx && p[1] == cpy });
                    // Compound carrier: dav's compound-inter chroma branch (recon 3716) blends the
                    // CARRIER's mv pair for the whole chroma region — NO per-sub-block loop even
                    // when forced (sub-8 carriers).
                    use crate::av2_inter::{bacp_mask, comp_avg, comp_mask, comp_w_avg, mc_translate_prep};
                    let refidx = CUR_FRAME_REFIDX.with(|c| c.get()).1;
                    let mut t = [vec![0i32; cw * ch], vec![0i32; cw * ch]];
                    crate::av2_frame::REF_PICS.with(|rp| {
                        let pics = rp.borrow();
                        for i in 0..2 {
                            if let Some(p) = pics[refidx[refs[i]] as usize].as_ref() {
                                mc_translate_prep(&p[pl + 1], &mut t[i], cw, cpx, cpy, cw, ch, mvp[i].y, mvp[i].x, filter as usize, ss_g().0 as u32, ss_g().1 as u32);
                            }
                        }
                        if icdbg && pl == 0 {
                            crate::dlog!("[ICDBG] refs={refs:?} refidx={:?} ct={ct} cwp={cwp} msign={msign}", refidx);
                            crate::dlog!("[ICDBG] t0_r0={:?}", &t[0][..8]);
                            crate::dlog!("[ICDBG] t1_r0={:?}", &t[1][..8]);
                        }
                    });
                    if ct == 2 {
                        // WEDGE chroma (dav recon 3834): the subsampled codebook mask of the
                        // SAME luma-size block (ssidx = ss_hor+ss_ver), dims cw x ch.
                        let (sshw, ssvw) = ss_g();
                        let bw4l = (cw << sshw) / 4;
                        let bh4l = (ch << ssvw) / 4;
                        let mask = crate::av2_wedge::wedge_mask(bw4l, bh4l, widx as usize, sshw + ssvw);
                        let (ta, tb) = if wsign { (&t[1], &t[0]) } else { (&t[0], &t[1]) };
                        comp_mask(&mut buf, cw, ta, tb, cw, ch, &mask, bdmax_g());
                    } else if ct == 3 {
                        // dav chroma SEG (recon 3760): read the scratch at stride cw; <16 chroma
                        // blocks at the base, else offset ((cpy>>2)&15)... dav: ((ssby&15)*4*cw +
                        // (ssbx&15)*4) with ssbx/ssby = chroma 4px block coords = cpx/4, cpy/4.
                        let _ = (segm, mstride);
                        // dav recon 3762: min(cbw4,cbh4) < 16 (luma 4px units = cw/2) -> base.
                        let moff = if ((cw << ss_g().0) / 4).min((ch << ss_g().1) / 4) < 16 {
                            0
                        } else {
                            (((cpy / 4) & 15) * 4) * cw + ((cpx / 4) & 15) * 4
                        };
                        let m: Vec<u8> = SEG_SCRATCH.with(|sc| {
                            let scr = sc.borrow();
                            (0..cw * ch)
                                .map(|i| scr[(moff + (i / cw) * cw + (i % cw)).min(64 * 64 - 1)])
                                .collect()
                        });
                        let (ta, tb) = if msign { (&t[1], &t[0]) } else { (&t[0], &t[1]) };
                        comp_mask(&mut buf, cw, ta, tb, cw, ch, &m, bdmax_g());
                    } else {
                        let wt = cwp as i32;
                        if wt == 8 {
                            let (fw_px, fh_px) = ((r.w) as i32, (r.h) as i32);
                            let ssb = ss_g();
                            let x0 = cpx as i32 + (mvp[0].x >> (3 + ssb.0 as i32));
                            let y0 = cpy as i32 + (mvp[0].y >> (3 + ssb.1 as i32));
                            let x1 = cpx as i32 + (mvp[1].x >> (3 + ssb.0 as i32));
                            let y1 = cpy as i32 + (mvp[1].y >> (3 + ssb.1 as i32));
                            let oof = x0 < 0 || x1 < 0 || y0 < 0 || y1 < 0
                                || x0 + cw as i32 >= fw_px || x1 + cw as i32 >= fw_px
                                || y0 + ch as i32 >= fh_px || y1 + ch as i32 >= fh_px;
                            if oof {
                                SEG_SCRATCH.with(|sc| {
                                    let mut scr = sc.borrow_mut();
                                    if pl == 0 {
                                        bacp_mask(&mut scr, cw, cw, ch, x0, y0, x1, y1, fw_px, fh_px);
                                    }
                                    comp_mask(&mut buf, cw, &t[0], &t[1], cw, ch, &scr, bdmax_g());
                                });
                            } else {
                                comp_avg(&mut buf, cw, &t[0], &t[1], cw, ch, bdmax_g());
                            }
                        } else {
                            comp_w_avg(&mut buf, cw, &t[0], &t[1], cw, ch, wt, bdmax_g());
                        }
                    }
                } else if forced {
                    // Chroma px per luma-mi cell, per axis: (4>>ssh, 4>>ssv) — 420=(2,2), 422=(2,4).
                    let (pxw, pxh) = { let ss = ss_g(); (4usize >> ss.0, 4usize >> ss.1) };
                    let (ccbx4, ccby4) = (cpx / pxw, cpy / pxh);
                    let (cbw4, cbh4) = (cw / pxw, ch / pxh);
                    let mut covered = [0u32; 16];
                    for y in 0..cbh4 {
                        if !work_tick("chroma:3381") { break; }
                        for x in 0..cbw4 {
                            if !work_tick("chroma:3382") { break; }
                            if covered[y] & (1 << x) != 0 {
                                continue;
                            }
                            // Use lmv[0] = the BASE decoded block mv (== dav's frame_thread.b mv[0]
                            // that the sub-8×8 chroma MCs with). For a WARP block, grid mv[0] is the
                            // warp-PROJECTED per-cell mv (for refmvs); lmv[0] keeps the base. For a
                            // non-warp block lmv[0] == mv[0].
                            let (mv2, sbs, sref) = crate::av2_refmvs::GRID.with(|g| {
                                let b2 = *g.borrow().at(ccby4 + y, ccbx4 + x);
                                (b2.lmv[0], b2.bs, b2.ref_[0])
                            });
                            let sdim = crate::av2_decode::BLOCK_DIMENSIONS[sbs as usize];
                            let (sw, sh) = (sdim[0] as usize, sdim[1] as usize); // luma 4px
                            // Each sub-block MCs with its OWN filter AND its OWN reference
                            // (dav recon 3667: ref = b2->ref.ref[0], refp = f->refp[ref]).
                            let gs = crate::av2_frame::FILTER_GRID_STRIDE.with(|c| c.get());
                            let sfilt = crate::av2_frame::FILTER_GRID.with(|fg| fg.borrow().get((ccby4 + y) * gs + (ccbx4 + x)).copied().unwrap_or(0));
                            let off = (y * pxh) * cw + x * pxw;
                            let refidx2 = CUR_FRAME_REFIDX.with(|c| c.get()).1;
                            let done = crate::av2_frame::REF_PICS.with(|rp| {
                                let pics = rp.borrow();
                                if sref >= 0 {
                                    if let Some(p2) = pics[refidx2[sref as usize] as usize].as_ref() {
                                        crate::av2_inter::mc_translate(&p2[pl + 1], &mut buf[off..], cw, (ccbx4 + x) * pxw, (ccby4 + y) * pxh, sw * pxw, sh * pxh, mv2.y, mv2.x, sfilt as usize, ss_g().0 as u32, ss_g().1 as u32, bdmax_g());
                                        return true;
                                    }
                                }
                                false
                            });
                            if !done {
                                crate::av2_inter::mc_translate(r, &mut buf[off..], cw, (ccbx4 + x) * pxw, (ccby4 + y) * pxh, sw * pxw, sh * pxh, mv2.y, mv2.x, sfilt as usize, ss_g().0 as u32, ss_g().1 as u32, bdmax_g());
                            }
                            let m2 = (((1u32 << sw) - 1) << x) & 0xffff;
                            for yy in y..(y + sh).min(16) {
                                if !work_tick("chroma:3415") { break; }
                                covered[yy] |= m2;
                            }
                        }
                    }
                } else {
                    match warp_pred {
                        Some(m) => match crate::av2_warp::get_shear_params(&m).filter(|_| cw.min(ch) >= 8) {
                            Some(abcd) => crate::av2_warp::warp_affine(r, &mut buf, cw, &m, &abcd, bx4, by4, cw, ch, ss_g().0 as u32, ss_g().1 as u32, bdmax_g()),
                            None => crate::av2_warp::ext_warp(r, &mut buf, cw, &m, bx4, by4, cw, ch, ss_g().0 as u32, ss_g().1 as u32, bdmax_g()),
                        },
                        None => crate::av2_inter::mc_translate(r, &mut buf, cw, cpx, cpy, cw, ch, fmv.y, fmv.x, filter as usize, ss_g().0 as u32, ss_g().1 as u32, bdmax_g()),
                    }
                }
                // BAWP: morph the chroma MC prediction (reuse luma alpha) BEFORE the residual.
                if let Some(la) = bawp_alpha {
                    bawp_morph_chroma(&mut buf, pl + 1, cpx, cpy, cw, ch, fmv, la, r);
                } else if ii_mode >= 0 {
                    // Inter-intra blend (dav recon 3731 chroma iiblend — else-of-bawp).
                    let (sshc, ssvc) = ss_g();
                    let (lx4, ly4) = ((cpx << sshc) >> 2, (cpy << ssvc) >> 2);
                    let (lw4, lh4) = ((cw << sshc) >> 2, (ch << ssvc) >> 2);
                    ii_blend(&mut buf, pl + 1, cpx, cpy, cw, ch, lx4, ly4, lw4, lh4, ii_mode);
                }
                // Residual (dequant chroma levels → inv transform → add). uv_txtp is dav's packed
                // chroma TxfmType; apply the inter_ddt DDT mask then the DAV2MINE 1d-index remap.
                if let Some(cc) = cc {
                    let cf = if pl == 0 { &cc.cf_u } else { &cc.cf_v };
                    let (slw, slh) = (cc.slw, cc.slh);
                    let (tw, th) = (4usize << slw, 4usize << slh);
                    let n = tw * th;
                    use crate::av2_dequant::{cf_max, dequant_coeff, dq_lookup};
                    let dq = dq_lookup(yac);
                    let tx_scale = (n > 256) as u32 + (n > 1024) as u32;
                    let cfmax = cf_max((bdmax_g() + 1).trailing_zeros());
                    // itx types hoisted ABOVE the dequant (QM applies to 2D transforms only).
                    let mut txtp = cc.uv_txtp;
                    if inter_ddt {
                        let mask = (if slw == 1 || slw == 2 { 0x02u8 } else { 0 })
                            | (if slh == 1 || slh == 2 { 0x40u8 } else { 0 });
                        txtp += txtp & mask;
                    }
                    const DAV2MINE: [usize; 8] = [0, 3, 1, 2, 4, 5, 6, 7];
                    let (row_ty, col_ty) = (DAV2MINE[(txtp & 7) as usize], DAV2MINE[((txtp >> 5) & 7) as usize]);
                    let iqm = crate::av2_qm::iqm_slice(pl + 1, tw, th, row_ty != 3 && col_ty != 3);
                    let mut coeff = vec![0i32; n];
                    for i in 0..n.min(cf.len()) {
                        let lvl = cf[i];
                        if lvl != 0 {
                            let s = (lvl < 0) as u32;
                            let q = crate::av2_qm::qm_apply(iqm, i, th.min(32), dq);
                            let mag0 = dequant_coeff(lvl.unsigned_abs(), q, 3, cfmax, s, false) as i32;
                            let mag = (mag0 >> tx_scale).min(cfmax);
                            coeff[i] = if lvl < 0 { -mag } else { mag };
                        }
                    }
                    let mut residual = vec![0i32; n];
                    crate::av2_itx::inv_txfm_2d(&coeff, slw, slh, row_ty, col_ty, &mut residual);
                    if std::env::var("CQD").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == cpx && p[1] == cpy }) {
                        crate::dlog!("[CQD] pl={pl} tw={tw} th={th} coeff_nz={:?} resid_r0={:?} resid_r1={:?} pred_r0={:?}",
                            coeff.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v)).collect::<Vec<_>>(),
                            &residual[..tw.min(32)], &residual[tw..tw + tw.min(32)],
                            &buf[..tw.min(32)]);
                    }
                    crate::av2_itx::residual_add(&mut buf, cw, &residual, tw, th, 0, 0, 0, bdmax_g());
                }
                // Score vs dav's pre-filter chroma recon (probe-only, capture optional).
                let mut ok = true;
                if let Some(rec) = rec {
                    scored = true;
                    'cmp: for yy in 0..ch {
                        if !work_tick("chroma:3485") { break; }
                        for xx in 0..cw {
                            if !work_tick("chroma:3486") { break; }
                            let (px, py) = (cpx + xx, cpy + yy);
                            if px >= rec.w || py >= rec.h {
                                continue;
                            }
                            if buf[yy * cw + xx].clamp(0, 255) != rec.at(px, py) {
                                ok = false;
                                crate::dlog!("IRECMISSC pl={pl} cpx={cpx} cpy={cpy} cw={cw} ch={ch} fmv=({},{}) filt={filter} warp={} at({xx},{yy}) mine={} dav={} hascf={}", fmv.y, fmv.x, warp_pred.is_some() as u8, buf[yy * cw + xx].clamp(0, 255), rec.at(px, py), cc.is_some() as u8);
                                break 'cmp;
                            }
                        }
                    }
                }
                if pl == 0 { oku = ok } else { okv = ok }
                // Persist the inter chroma recon into the frame-2 FRAME buffer (for chroma intra
                // neighbours). Same indexing as luma; chroma plane pl+1.
                crate::av2_frame::FRAME.with(|fr| {
                    let mut f = fr.borrow_mut();
                    if f.pl[pl + 1].w != 0 {
                        // Write the in-frame (clipped) portion — edge blocks that spill still recon.
                        let cstride = f.pl[pl + 1].stride;
                        let wc = cw.min(f.pl[pl + 1].w.saturating_sub(cpx));
                        let hc = ch.min(f.pl[pl + 1].h.saturating_sub(cpy));
                        for yy in 0..hc {
                            if !work_tick("chroma:3509") { break; }
                            let d = (cpy + yy) * cstride + cpx;
                            for xx in 0..wc {
                                f.pl[pl + 1].px[d + xx] = buf[yy * cw + xx].clamp(0, bdmax_g());
                            }
                        }
                        crate::av2_frame::write_recon_pad(pl + 1, cpx, cpy, &buf, cw, ch);
                        mscore_chroma(pl + 1, cpx, cpy, cw, ch, &buf);
                    }
                });
            }
        })
    });
    // Mark the chroma decode-order availability grid (mi_coded_c) for this chroma block's luma-
    // equivalent region so later chroma intra blocks read the right top-right/bottom-left avail.
    crate::av2_frame::FRAME.with(|fr| {
        let mut f = fr.borrow_mut();
        if f.pl[0].w != 0 {
            let ss = ss_g();
            f.mark_coded_c_avail((cpx << ss.0) / 4, (cpy << ss.1) / 4, (cw << ss.0) / 4, (ch << ss.1) / 4);
        }
    });
    if scored {
        INTER_SCORE_RC.with(|s| {
            let (u, v, t) = s.get();
            s.set((u + oku as u32, v + okv as u32, t + 1));
        });
    }
    crate::av2_frame::dbg_block_miss_c(cpx, cpy, cw, ch, "inter");
}

/// dav2d BAWP (block-adaptive weighted prediction) morph for LUMA (recon_tmpl.c:2731 `bawp` +
/// mc_tmpl.c:958 `morph` + derivation.h:35 `derive_alpha`). Fits a linear model `(alpha,beta)`
/// between the REFERENCE template (REF_FRAME1 at the rounded mv-offset) and the CURRENT template
/// (the assembled FRAME's row-above + left-col), then applies `p = clip((alpha*p + beta) >> 8)` to
/// the MC prediction `pred` in place. No-op (leaves `pred`) when the ref template goes off-frame.
/// Returns the derived luma alpha (256 when the ref template is off-frame / no morph) so the
/// chroma BAWP morph can reuse it (dav recon 2852: chroma `alpha = have_left||have_above ? bawp[0].alpha : 256`).
/// avm `av2_build_morph_pred` (reconinter.c:4106): intrabc `morph_pred=1` refines the block-copy
/// prediction with a BAWP-style linear model `(alpha,beta)` fitted between the current block's
/// L-shaped template (1 row above + 1 col left, `BAWP_REF_LINES=1`) and the SAME template around
/// the intrabc SOURCE position — both in the current frame's recon. Luma only; applied after the
/// copy, before the residual.
fn intrabc_morph_luma(pl: &crate::av2_frame::Plane, pred: &mut [i32], bx4: usize, by4: usize, w: usize, h: usize, bv: (i32, i32)) {
    let (cur_x, cur_y) = (bx4 as i32 * 4, by4 as i32 * 4);
    if cur_x >= pl.w as i32 || cur_y >= pl.h as i32 {
        return;
    }
    // get_fullmv_from_mv: arithmetic >>3 (intrabc BVs are whole-pel, so exact).
    let (dvy, dvx) = (bv.0 >> 3, bv.1 >> 3);
    // ref_w/ref_h: the template fit uses the VISIBLE (edge-clamped) block dims.
    let ref_w = (w as i32).min(pl.w as i32 - cur_x) as usize;
    let ref_h = (h as i32).min(pl.h as i32 - cur_y) as usize;
    // avm xd->up_available / left_available (tile-relative; single tile == frame edges).
    let t = TILE_B.with(|c| c.get());
    let up_avail = by4 > t.2;
    let left_avail = bx4 > t.0;
    // derive_bawp_parameters (reconinter.c:1795), plane=0 → max 16 samples per side.
    let bw = ref_w.min(16);
    let bh = ref_h.min(16);
    const LOG2_BAWP: [u8; 17] = [0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4];
    const LOG_TO_SIZE: [usize; 5] = [0, 2, 4, 8, 16];
    let width = LOG_TO_SIZE[LOG2_BAWP[bw] as usize];
    let height = LOG_TO_SIZE[LOG2_BAWP[bh] as usize];
    // derive_number_ref_samples_bawp
    let above_ok = width != 0 && up_avail;
    let left_ok = height != 0 && left_avail;
    let (numb_up, numb_left) = if above_ok && left_ok {
        if width == 16 && height == 16 { (16, 16) }
        else if width > 4 && height > 4 { (8, 8) }
        else if width < 16 && height < 16 { (4, 4) }
        else if width == 16 { (16, 0) }
        else { (0, 16) }
    } else if above_ok {
        (width, 0)
    } else if left_ok {
        (0, height)
    } else {
        (0, 0)
    };
    let at = |x: i32, y: i32| -> i32 { pl.at(x.max(0) as usize, y.max(0) as usize) };
    let (mut sum_x, mut sum_y, mut sum_xy, mut sum_xx, mut count) = (0i64, 0i64, 0i64, 0i64, 0i32);
    let mut side = |n: usize, len: usize, real: usize, gets: &dyn Fn(usize) -> (i32, i32)| {
        if n == 0 { return; }
        let mut rp = [0i32; 16];
        let mut cp = [0i32; 16];
        for (i, (r, c)) in rp.iter_mut().zip(cp.iter_mut()).enumerate().take(real) {
            let (rv, cv) = gets(i);
            *r = rv;
            *c = cv;
        }
        for i in 0..len.saturating_sub(real) {
            rp[i + real] = rp[i];
            cp[i + real] = cp[i];
        }
        let step = len / n;
        let start = if step == 1 { 0 } else { step >> 1 };
        let mut i = start;
        while i < len {
            if !crate::av2_recon::work_tick("recon:morph") { break; }
            sum_x += rp[i] as i64;
            sum_y += cp[i] as i64;
            sum_xy += (rp[i] * cp[i]) as i64;
            sum_xx += (rp[i] * rp[i]) as i64;
            i += step;
        }
        count += n as i32;
    };
    side(numb_up, width, bw, &|i| {
        (at(cur_x + dvx + i as i32, cur_y + dvy - 1), at(cur_x + i as i32, cur_y - 1))
    });
    side(numb_left, height, bh, &|i| {
        (at(cur_x + dvx - 1, cur_y + dvy + i as i32), at(cur_x - 1, cur_y + i as i32))
    });
    let (alpha, beta);
    if count > 0 {
        // derive_linear_parameters_alpha (cfl.h:36): true /count divisions.
        let der = (sum_xx - sum_x * sum_x / count as i64) as i32;
        let nor = (sum_xy - sum_x * sum_y / count as i64) as i32;
        alpha = bawp_derive_alpha(nor, der, 256);
        beta = (((sum_y as i32) << 8) - sum_x as i32 * alpha) / count;
    } else {
        alpha = 256;
        beta = -128;
    }
    let bdmax = bdmax_g();
    for p in pred.iter_mut().take(w * h) {
        *p = ((*p * alpha + beta) >> 8).clamp(0, bdmax);
    }
}

/// dav2d/avm shared BAWP linear-model alpha (dav derivation.h:35 `derive_alpha`, ==
/// avm `resolve_divisor_32_CfL(nor, der, 8)` with the 0→256 fallback).
fn bawp_derive_alpha(num: i32, den: i32, mut alpha: i32) -> i32 {
    #[inline]
    fn ulog2(d: u32) -> i32 { 31 - d.leading_zeros() as i32 }
    #[inline]
    fn apply_sign(a: i32, b: i32) -> i32 { if b < 0 { -a } else { a } }
    let max = (2 << 8) - 1;
    if num != 0 && den > 0 {
        let num_abs = num.abs();
        let shift_n = ulog2(num_abs as u32);
        let shift_d = ulog2(den as u32);
        let e_d = den - (1 << shift_d);
        let f_d = if shift_d > 7 { (e_d + (1 << (shift_d - 8))) >> (shift_d - 7) } else { e_d << (7 - shift_d) };
        let f_n = if shift_n > 7 { (num_abs + (1 << (shift_n - 8))) >> (shift_n - 7) } else { num_abs << (7 - shift_n) };
        let shift_add = shift_d - shift_n - 8;
        if shift_add <= 1 {
            let shift0 = 9 + 7 + shift_add;
            let tmp = if shift0 < 0 { max } else { ((crate::av2_ipred::DIV_RECIP[f_d as usize] as i64 * f_n as i64) >> shift0).min(max as i64) as i32 };
            if tmp != 0 { alpha = apply_sign(tmp, num); }
        }
    }
    alpha
}

fn bawp_morph(pred: &mut [i32], bx4: usize, by4: usize, bw4: usize, bh4: usize, mv: crate::av2_refmvs::Mv, bawp_idx: u8, ref0: usize) -> i32 {
    #[inline]
    fn ulog2(d: u32) -> i32 { 31 - d.leading_zeros() as i32 }
    #[inline]
    fn apply_sign(a: i32, b: i32) -> i32 { if b < 0 { -a } else { a } }
    fn derive_alpha(num: i32, den: i32, mut alpha: i32) -> i32 {
        let max = (2 << 8) - 1;
        if num != 0 && den != 0 {
            let num_abs = num.abs();
            let shift_n = ulog2(num_abs as u32);
            let shift_d = ulog2(den as u32);
            let e_d = den - (1 << shift_d);
            let f_d = if shift_d > 7 { (e_d + (1 << (shift_d - 8))) >> (shift_d - 7) } else { e_d << (7 - shift_d) };
            let f_n = if shift_n > 7 { (num_abs + (1 << (shift_n - 8))) >> (shift_n - 7) } else { num_abs << (7 - shift_n) };
            let shift_add = shift_d - shift_n - 8;
            if shift_add <= 1 {
                let shift0 = 9 + 7 + shift_add;
                let tmp = if shift0 < 0 { max } else { ((crate::av2_ipred::DIV_RECIP[f_d as usize] as i64 * f_n as i64) >> shift0).min(max as i64) as i32 };
                if tmp != 0 { alpha = apply_sign(tmp, num); }
            }
        }
        alpha
    }
    // n_edge_samples[have_above && have_left][lh4][lw4][above, left] (recon_tmpl.c:2789).
    const NES: [[[[u8; 2]; 3]; 3]; 2] = [
        [[[2, 2], [3, 2], [4, 2]], [[2, 3], [3, 3], [4, 3]], [[2, 4], [3, 4], [4, 4]]],
        [[[2, 2], [2, 2], [4, 0]], [[2, 2], [3, 3], [3, 3]], [[0, 4], [3, 3], [4, 4]]],
    ];
    let (w, h) = (bw4 * 4, bh4 * 4);
    // BAWP template availability is TILE-relative (dav recon_tmpl.c:2839 `bx > tiling.col_start`).
    let tbawp = TILE_B.with(|t| t.get());
    let have_left = bx4 > tbawp.0;
    let have_above = by4 > tbawp.2;
    // ref template position (luma ss=0). Round-to-integer mv (dav recon 2769).
    let mvx = (mv.x + 3 + (mv.x >= 0) as i32) >> 3;
    let mvy = (mv.y + 3 + (mv.y >= 0) as i32) >> 3;
    let ref_x = bx4 as i32 * 4 + mvx;
    let ref_y = by4 as i32 * 4 + mvy;
    // can_morph: the ref template must lie inside the frame (inter ref ⇒ tile edges = frame edges).
    // Frame dims from the current frame's stored size (NOT hardcoded — v320 is 320×176).
    let (cfw, cfh) = CUR_FRAME_REF.with(|c| { let v = c.get(); (v.3 as i32, v.4 as i32) });
    let (fw, fh, fbw4, fbh4) = (cfw, cfh, cfw / 4, cfh / 4);
    let sb_w4 = (bw4 as i32).min(fbw4 - bx4 as i32);
    let sb_h4 = (bh4 as i32).min(fbh4 - by4 as i32);
    if !(ref_y + sb_h4 * 4 <= fh && ref_x + sb_w4 * 4 <= fw && ref_y - 1 >= 0 && ref_x - 1 >= 0) {
        return 256;
    }
    let lw4 = (ulog2(bw4 as u32).min(2)) as usize;
    let lh4 = (ulog2(bh4 as u32).min(2)) as usize;
    let idx = (have_above && have_left) as usize;
    let n_above = if have_above { NES[idx][lh4][lw4][0] as i32 } else { 0 };
    let n_left = if have_left { NES[idx][lh4][lw4][1] as i32 } else { 0 };
    let count_l2 = n_above + if n_above == n_left { (n_above != 0) as i32 } else { n_left };
    let (mut sx, mut sy, mut sxy, mut sx2) = (0i64, 0i64, 0i64, 0i64);
    crate::av2_frame::FRAME.with(|fr| {
        crate::av2_frame::REF_FRAME1.with(|rf| {
            let f = fr.borrow();
            let rfb = rf.borrow();
            let rp = match rfb[0].as_ref() {
                Some(r) => r,
                None => return,
            };
            let fstride = f.pl[0].stride;
            // HARDENING: corrupt geometry can push the BAWP template outside the plane.
            let getf = |x: i32, y: i32| -> i64 {
                f.pl[0].px.get(y.max(0) as usize * fstride + x.max(0) as usize).copied().unwrap_or(0) as i64
            };
            let getr = |x: i32, y: i32| -> i64 { rp.at(x as usize, y as usize) as i64 };
            if n_above != 0 {
                let bw = 4i32 << lw4;
                let step = bw >> n_above;
                let mut i = step >> 1;
                while i < bw {
                    if !crate::av2_recon::work_tick("av2_recon:3594") { break; }
                    let (x, y) = (getr(ref_x + i, ref_y - 1), getf(bx4 as i32 * 4 + i, by4 as i32 * 4 - 1));
                    sx += x; sy += y; sxy += x * y; sx2 += x * x;
                    i += step;
                }
            }
            if n_left != 0 {
                let bh = 4i32 << lh4;
                let step = bh >> n_left;
                let mut i = step >> 1;
                while i < bh {
                    if !crate::av2_recon::work_tick("av2_recon:3604") { break; }
                    let (x, y) = (getr(ref_x - 1, ref_y + i), getf(bx4 as i32 * 4 - 1, by4 as i32 * 4 + i));
                    sx += x; sy += y; sxy += x * y; sx2 += x * x;
                    i += step;
                }
            }
        })
    });
    let (sx, sy, sxy, sx2) = (sx as i32, sy as i32, sxy as i32, sx2 as i32);
    let alpha = if bawp_idx != 1 {
        // explicit alpha (dav recon 2854): idx = (1 + (bawp>>2) + (absrefdist[ref] > 4)) * ±1.
        let absd = CUR_REFDIST.with(|c| c.get()).1[ref0];
        let idx2 = (1 + (bawp_idx as i32 >> 2) + (absd > 4) as i32) * if bawp_idx & 1 != 0 { 1 } else { -1 };
        256 + 16 * idx2
    } else if count_l2 != 0 {
        let num = sxy - (((sx as i64 * sy as i64) >> count_l2) as i32);
        let den = sx2 - (((sx as i64 * sx as i64) >> count_l2) as i32);
        derive_alpha(num, den, 256)
    } else {
        256
    };
    let beta = if count_l2 != 0 {
        let diff = (sy << 8) - sx * alpha;
        apply_sign(diff.abs() >> count_l2, diff)
    } else {
        -128
    };
    for p in pred.iter_mut().take(w * h) {
        *p = ((alpha * *p + beta) >> 8).clamp(0, bdmax_g());
    }
    alpha
}

/// dav2d BAWP morph for a CHROMA plane (recon_tmpl.c:2731 with `plane != 0`, ss=1 for 4:2:0):
/// reuses the LUMA alpha (`luma_alpha`) and fits only `beta` from the chroma ref/current templates,
/// then morphs the chroma MC prediction `pred` (in `cw`×`ch` chroma pixels at chroma-pixel origin
/// `(cpx, cpy)`) in place. No-op when the ref template goes off the chroma frame.
#[allow(clippy::too_many_arguments)]
fn bawp_morph_chroma(pred: &mut [i32], plane: usize, cpx: usize, cpy: usize, cw: usize, ch: usize, mv: crate::av2_refmvs::Mv, luma_alpha: i32, r: &crate::av2_frame::Plane) {
    // avm reconinter.c derive_bawp_parameters / av2_build_one_bawp_inter_predictor, ss-general
    // (dav2d only decodes 420, so the 422/444 arms mirror avm directly).
    let (ssh, ssv) = { let s = crate::av2_frame::SS.with(|c| c.get()); (s.0 as i32, s.1 as i32) };
    let (fw, fh) = CUR_FRAME_REF.with(|c| {
        let v = c.get();
        (((v.3 as i32) + ((1 << ssh) - 1)) >> ssh, ((v.4 as i32) + ((1 << ssv) - 1)) >> ssv)
    });
    // BAWP template availability is TILE-relative (dav recon_tmpl.c:2839), in chroma pixels.
    let tbawp = TILE_B.with(|t| t.get());
    let have_left = cpx > (tbawp.0 * 4) >> ssh;
    let have_above = cpy > (tbawp.2 * 4) >> ssv;
    // GET_MV_RAWPEL then >> ss per axis (nested floor-shifts == one combined shift).
    let mvx = (mv.x + 3 + (mv.x >= 0) as i32) >> (3 + ssh);
    let mvy = (mv.y + 3 + (mv.y >= 0) as i32) >> (3 + ssv);
    let ref_x = cpx as i32 + mvx;
    let ref_y = cpy as i32 + mvy;
    let sb_w = (cw as i32).min(fw - cpx as i32);
    let sb_h = (ch as i32).min(fh - cpy as i32);
    if !(ref_y + sb_h <= fh && ref_x + sb_w <= fw && ref_y - 1 >= 0 && ref_x - 1 >= 0) {
        return;
    }
    // Chroma template sides use at most BAWP_MAX_REF_NUMB/2 = 8 samples, padded up to a
    // power-of-2 width/height per blk_size_log2_bawp.
    const BLK_LOG2: [usize; 17] = [0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4];
    const LOG_TO_SIZE: [usize; 5] = [0, 2, 4, 8, 16];
    let bw = (sb_w.max(0) as usize).min(8);
    let bh = (sb_h.max(0) as usize).min(8);
    let width = LOG_TO_SIZE[BLK_LOG2[bw]];
    let height = LOG_TO_SIZE[BLK_LOG2[bh]];
    let above_avail = have_above && width != 0;
    let left_avail = have_left && height != 0;
    // derive_number_ref_samples_bawp
    let (numb_up, numb_left) = if above_avail && left_avail {
        if width == 16 && height == 16 { (16, 16) }
        else if width > 4 && height > 4 { (8, 8) }
        else if width < 16 && height < 16 { (4, 4) }
        else if width == 16 { (16, 0) }
        else { (0, 16) }
    } else if above_avail { (width, 0) }
    else if left_avail { (0, height) }
    else { (0, 0) };
    let mut count = 0usize;
    let (mut sx, mut sy) = (0i64, 0i64);
    // Current template = the assembled FRAME's chroma row-above/left-col; ref template = the caller's
    // already-borrowed ref chroma plane `r`.
    crate::av2_frame::FRAME.with(|fr| {
        let f = fr.borrow();
        let fstride = f.pl[plane].stride;
        let getf = |x: i32, y: i32| -> i64 { f.pl[plane].px[y as usize * fstride + x as usize] as i64 };
        let getr = |x: i32, y: i32| -> i64 { r.at(x as usize, y as usize) as i64 };
        let mut side = |n: usize, len: usize, pad: usize, rf: &dyn Fn(usize) -> (i64, i64)| {
            if n == 0 { return; }
            let mut refp = [0i64; 16];
            let mut recp = [0i64; 16];
            for i in 0..len { let (x, y) = rf(i); refp[i] = x; recp[i] = y; }
            for i in 0..pad.saturating_sub(len) { refp[len + i] = refp[i]; recp[len + i] = recp[i]; }
            let step = pad / n;
            let mut i = if step == 1 { 0 } else { step >> 1 };
            let step = step.max(1); // HARDENING: zero step would spin
            while i < pad { sx += refp[i]; sy += recp[i]; i += step; }
        };
        side(numb_up, bw, width, &|i| (getr(ref_x + i as i32, ref_y - 1), getf(cpx as i32 + i as i32, cpy as i32 - 1)));
        count += numb_up;
        side(numb_left, bh, height, &|i| (getr(ref_x - 1, ref_y + i as i32), getf(cpx as i32 - 1, cpy as i32 + i as i32)));
        count += numb_left;
    });
    // chroma alpha REUSES the luma alpha; only beta is refit from the chroma templates.
    let (alpha, beta) = if count > 0 {
        let alpha = luma_alpha;
        (alpha, (((sy as i32) << 8) - sx as i32 * alpha) / count as i32)
    } else {
        (256, -128)
    };
    for p in pred.iter_mut().take(cw * ch) {
        *p = ((alpha * *p + beta) >> 8).clamp(0, bdmax_g());
    }
}

/// dav2d intrabc RECON (recon_tmpl.c:3203): block-copy the LUMA + CHROMA from the assembled FRAME
/// (the current frame) at the block vector `bv` (1/8-pel; integer ⇒ pure copy) + inter-style
/// residual, and write back into FRAME. Marks the decode-order availability grids.
fn recon_intrabc(bx4: usize, by4: usize, bw4: usize, bh4: usize, slw: usize, slh: usize, bv: (i32, i32), morph: bool, luma_cf: &[i32], luma_txtp: u8, all_zero: bool, cc: Option<&ChromaCoefs>, luma_stx: u8, luma_eob: i32, luma_only: bool, units: &[TxUnitCf]) {
    use crate::av2_dequant::{cf_max, dequant_coeff, dq_lookup};
    const DAV2MINE: [usize; 8] = [0, 3, 1, 2, 4, 5, 6, 7];
    let (w, h) = (bw4 * 4, bh4 * 4);
    let yac = LAST_QIDX.with(|c| c.get()); // delta-q: the current SB's effective qindex (== frame yac when delta-q is off)
    // Dequant + inverse-transform a coef block into `residual` (intrabc = intra, so NO DDT remap).
    let build_residual = |cf: &[i32], tw: usize, th: usize, txtp0: u8, slw: usize, slh: usize, dcq: u32, stx: u8, eob: i32, plane: usize| -> Vec<i32> {
        let n = tw * th;
        let dq = dq_lookup(yac);
        let tx_scale = (n > 256) as u32 + (n > 1024) as u32;
        let cfmax = cf_max((bdmax_g() + 1).trailing_zeros());
        // QM weighting (2D transforms only; txtp0 is mine-packed, per-axis IDENTITY == 1).
        let is_2d = ((txtp0 & 7) & 3) != 1 && (((txtp0 >> 5) & 7) & 3) != 1;
        let iqm = crate::av2_qm::iqm_slice(plane, tw, th, is_2d);
        let mut coeff = vec![0i32; n];
        for i in 0..n.min(cf.len()) {
            let lvl = cf[i];
            if lvl != 0 {
                let s = (lvl < 0) as u32;
                // The DC coefficient (index 0) uses the plane's DC quantizer (dq_lookup(yac+dc_delta));
                // AC uses dq_lookup(yac). `dcq==0` means "no DC delta, fall back to the AC step".
                let q = if i == 0 && dcq != 0 { dcq } else { dq };
                let q = crate::av2_qm::qm_apply(iqm, i, th.min(32), q);
                let mag = (dequant_coeff(lvl.unsigned_abs(), q, 3, cfmax, s, false) as i32 >> tx_scale).min(cfmax);
                coeff[i] = if lvl < 0 { -mag } else { mag };
            }
        }
        // Secondary transform (STX/IST) on the dequantized coeffs before the primary itx — same as
        // the inter path (dav2d stx_tmpl.c stxfm8, set=0, transpose). intrabc (≥16×16 DCT_DCT) too.
        if stx > 0 {
            let ty = (stx - 1) as usize;
            let kernel = &crate::av2_stx_tables::STX_8X8_KERNEL_SET0[ty];
            let map_idx = crate::av2_stx_tables::COEFF8X8_MAPPING_IDX[ty] as usize;
            let mapping = &crate::av2_stx_tables::COEFF8X8_MAPPING[map_idx];
            let stride = th.min(32);
            let hh = (eob + 1) as usize;
            let (cmin, cmax) = (-128 * 256, 128 * 256 - 1);
            let mut sums = [0i32; 48];
            for (x, sx) in sums.iter_mut().enumerate() {
                let mut sum = 0i32;
                for y in 0..hh { sum += coeff[y] * kernel[y][x] as i32; }
                let m = (sum.abs() + 64) >> 7;
                *sx = (if sum < 0 { -m } else { m }).clamp(cmin, cmax);
            }
            for c in coeff.iter_mut().take(32) { *c = 0; }
            for (n_i, &rc) in mapping.iter().enumerate() {
                let (x, y) = ((rc & 7) as usize, (rc >> 3) as usize);
                if stride > 8 { coeff[y * stride + x] = sums[n_i]; } else { coeff[rc as usize] = sums[n_i]; }
            }
        }
        // intrabc is an INTRA block (dav2d `b->intra == 1`), so it does NOT apply the inter DDT
        // remap (`txtp += txtp & tx_ddt_mask` is gated on `!b->intra`, recon_tmpl.c:2713). Its
        // txtp stays as decoded (e.g. ADST×IDENTITY), same as a normal intra block.
        let txtp = txtp0;
        let (rt, ct) = (DAV2MINE[(txtp & 7) as usize], DAV2MINE[((txtp >> 5) & 7) as usize]);
        let mut residual = vec![0i32; n];
        // dav2d DC-only fast path (itx_tmpl.c:129 `eob + txtp == 0`): a DC-only DCT_DCT block adds a
        // single flat `dc` to every pixel — this differs by ±1 from the two-pass inverse transform
        // at some rectangular sizes (e.g. 64×16), so it must be used to match dav's arithmetic.
        if eob == 0 && txtp == DCT_DCT && stx == 0 {
            let dc = crate::av2_itx::inv_txfm_dc(coeff[0], slw, slh);
            residual.iter_mut().for_each(|r| *r = dc);
        } else {
            crate::av2_itx::inv_txfm_2d(&coeff, slw, slh, rt, ct, &mut residual);
        }
        residual
    };
    crate::av2_frame::FRAME.with(|fr| {
        let mut f = fr.borrow_mut();
        if f.pl[0].w == 0 {
            return;
        }
        f.ensure_sb(bx4, by4);
        f.mark_coded_avail(bx4, by4, bw4, bh4);
        // The frame-1 SDP luma tree marks only the LUMA grid; the chroma grid is the chroma tree's job.
        if !luma_only { f.mark_coded_c_avail(bx4, by4, bw4, bh4); }
        // (edge blocks spilling past the frame are still reconstructed — the writes below clip.)
        // LUMA: block-copy from FRAME at bv + residual.
        let mut pred = vec![0i32; w * h];
        // BILINEAR (5) — dav2d recon_tmpl.c:3212 uses DAV2D_FILTER_BILINEAR for the intrabc
        // block-copy subpel (a qpel BV interpolates with the 2-tap filter, not the 8-tap).
        crate::av2_inter::mc_translate_luma(&f.pl[0], &mut pred, w, bx4 * 4, by4 * 4, w, h, bv.0, bv.1, 5, bdmax_g());
        if morph {
            intrabc_morph_luma(&f.pl[0], &mut pred, bx4, by4, w, h, bv);
        }
        if std::env::var("MPROBE2").is_ok() && bx4 * 4 == 192 && by4 * 4 == 48 {
            crate::dlog!("[MPROBE2] px=(192,48) w={w} h={h} bv={bv:?} az={all_zero} eob={luma_eob} stx={luma_stx} txtp={luma_txtp} slw={slw} slh={slh} qidx={} dcq={} nz={:?} pred_row0={:?}",
                LAST_QIDX.with(|c| c.get()), crate::av2_frame::F2_DCQ.with(|c| c.get()[0]),
                luma_cf.iter().take(96).enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v)).collect::<Vec<_>>(),
                &pred[..8]);
        }
        if !units.is_empty() {
            // TX-partitioned block: one MC copy for the whole block, then per-unit residuals
            // at the unit offsets (avm get_tx_partition_sizes layout).
            let dcq = crate::av2_frame::F2_DCQ.with(|c| c.get()[0]);
            for u in units {
                if !u.all_zero {
                    f.mark_lr_noskip(bx4 + u.ux4, by4 + u.uy4, 1 << u.slw, 1 << u.slh);
                }
                if u.all_zero { continue; }
                let (uw, uh) = (4usize << u.slw, 4usize << u.slh);
                let residual = build_residual(&u.cf, uw, uh, u.txtp, u.slw, u.slh, dcq, u.stx, u.eob, 0);
                let off = u.uy4 * 4 * w + u.ux4 * 4;
                crate::av2_itx::residual_add(&mut pred[off..], w, &residual, uw.min(w - u.ux4 * 4), uh.min(h - u.uy4 * 4), 0, 0, 0, bdmax_g());
            }
        } else if !all_zero && !luma_cf.is_empty() {
            f.mark_lr_noskip(bx4, by4, bw4, bh4);
            let dcq = crate::av2_frame::F2_DCQ.with(|c| c.get()[0]);
            let residual = build_residual(luma_cf, w, h, luma_txtp, slw, slh, dcq, luma_stx, luma_eob, 0);
            crate::av2_itx::residual_add(&mut pred, w, &residual, w, h, 0, 0, 0, bdmax_g());
            if std::env::var("MPROBE2").is_ok() && bx4 * 4 == 192 && by4 * 4 == 48 {
                crate::dlog!("[MPROBE2] res_row0={:?} recon_row0={:?}", &residual[..8], &pred[..8]);
            }
        }
        let ls = f.pl[0].stride;
        let (wc, hc) = (w.min(f.pl[0].w.saturating_sub(bx4 * 4)), h.min(f.pl[0].h.saturating_sub(by4 * 4)));
        for yy in 0..hc {
            let d = (by4 * 4 + yy) * ls + bx4 * 4;
            for xx in 0..wc {
                f.pl[0].px[d + xx] = pred[yy * w + xx].clamp(0, bdmax_g());
            }
        }
        crate::av2_frame::write_recon_pad(0, bx4 * 4, by4 * 4, &pred, w, h);
        mscore_luma("intrabc", bx4 * 4, by4 * 4, w, h, &pred, w);
        f.mark_coded(bx4, by4, bw4, bh4, 0);
        // TX-partitioned block: per-UNIT deblock tx levels + unit-boundary edges (avm
        // av2_loopfilter.c get_tx_partition_sizes drives the filter level per unit).
        for u in units {
            f.mark_db(bx4 + u.ux4, by4 + u.uy4, 1usize << u.slw, 1usize << u.slh);
        }
        if units.is_empty() {
            // Un-TX-partitioned (incl. skip_txfm) intrabc block: dav's create_db_mask marks
            // EVERY `b->intra` block's edges — a skip intrabc block still deblocks its
            // boundaries with the block dims as the TX dims (tipfm2 y=64 edge, rows 62-64).
            f.mark_db(bx4, by4, bw4, bh4);
        }
        crate::av2_frame::mark_btype(bx4, by4, bw4, bh4, 3);
        // CHROMA: block-copy from FRAME chroma at bv (ss=1) + residual. Skipped for the frame-1 SDP
        // LUMA tree (`luma_only`) — its chroma is reconstructed later by the separate chroma tree.
        if luma_only { return; }
        let (sshc, ssvc) = ss_g();
        let (cw, ch) = ((bw4 * 4) >> sshc, (bh4 * 4) >> ssvc);
        let (cpx, cpy) = ((bx4 * 4) >> sshc, (by4 * 4) >> ssvc);
        for pl in 0..2 {
            if f.pl[pl + 1].w == 0 {
                continue;
            }
            let mut cpred = vec![0i32; cw * ch];
            // dav2d intrabc chroma block-copy uses the BILINEAR subpel filter (recon_tmpl.c:3649
            // `DAV2D_FILTER_BILINEAR`), NOT the 8-tap. Chroma BVs are half-pel when the luma BV is an
            // odd pixel count; the wrong filter gave a ±2 error on every subpel intrabc chroma block.
            crate::av2_inter::mc_translate(&f.pl[pl + 1], &mut cpred, cw, cpx, cpy, cw, ch, bv.0, bv.1, 5, ss_g().0 as u32, ss_g().1 as u32, bdmax_g());
            if let Some(cc) = cc {
                let cf = if pl == 0 { &cc.cf_u } else { &cc.cf_v };
                let (tw, th) = (4usize << cc.slw, 4usize << cc.slh);
                let dcq = crate::av2_frame::F2_DCQ.with(|c| c.get()[pl + 1]);
                let residual = build_residual(cf, tw, th, cc.uv_txtp, cc.slw, cc.slh, dcq, 0, if pl == 0 { cc.u_eob } else { cc.v_eob }, pl + 1);
                crate::av2_itx::residual_add(&mut cpred, cw, &residual, tw, th, 0, 0, 0, bdmax_g());
            }
            let cs = f.pl[pl + 1].stride;
            let (wc, hc) = (cw.min(f.pl[pl + 1].w.saturating_sub(cpx)), ch.min(f.pl[pl + 1].h.saturating_sub(cpy)));
            for yy in 0..hc {
                let d = (cpy + yy) * cs + cpx;
                for xx in 0..wc {
                    f.pl[pl + 1].px[d + xx] = cpred[yy * cw + xx].clamp(0, bdmax_g());
                }
            }
            crate::av2_frame::write_recon_pad(pl + 1, cpx, cpy, &cpred, cw, ch);
        }
        // Record the intrabc block's CHROMA deblock edges. dav's create_db_mask marks EVERY
        // block with `b->intra` (intrabc has intra=1), but recon_intrabc's chroma block-copy never
        // called mark_db_chroma — so intrabc chroma edges were left unmarked and under-deblocked.
        // Chroma block = the whole intrabc block subsampled (one "TX" — a block copy, no TX split).
        if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
            let (sshm, ssvm) = ss_g();
            f.mark_db_chroma(bx4 >> sshm, by4 >> ssvm, (bw4 >> sshm).max(1), (bh4 >> ssvm).max(1));
        }
    });
    if !luma_only {
        crate::av2_frame::dbg_block_miss_c(bx4 * 2, by4 * 2, bw4 * 2, bh4 * 2, "intrabc");
    }
    mscref_check(bx4, by4, w, h, "ibc");
}

/// Decode one **mixed-region** yuv leaf (dav2d `decode_b`, `!intra_region`): `is_inter` then
/// either the inter path (`decode_b_inter` + inter luma/chroma coefs) or the intra-yuv path
/// (`decode_b_luma` defer + chroma mode + intra luma/chroma coefs). All contexts computed.
#[allow(clippy::too_many_arguments)]
pub fn decode_leaf(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    a_nb: &mut BlockNbCtx,
    l_nb: &mut BlockNbCtx,
    cnb: &mut ChromaNb,
    bs: usize,
    bx4: usize,
    by4: usize,
    have_left: bool,
    have_top: bool,
    have_top_in_sb: bool,
    is_sb_boundary: bool,
    row_end: usize,
    col_end: usize,
    decode_filters: bool,
    left_cdef: &mut i8,
    left_ccso: &mut [u8; 3],
    // SDP intra region: leaf is FORCED intra + luma-only (no is_inter, no inline chroma).
    intra_region: bool,
    luma_dir_map: &mut [u8; 256],
    // Chroma block size (bs, -1 = luma-only) + origin (luma 4px coords). For a sub-8x8 leaf
    // (`cbs != bs`) the block is FORCED inter, luma-only unless it carries the shared chroma.
    cbs: i32,
    ccbx4: usize,
    ccby4: usize,
) {
    use crate::msac::rav1d_msac_decode_bool_adapt;
    let bd = crate::av2_decode::BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    let (slw, slh) = (bd[2] as usize, bd[3] as usize);
    let t_dim_ctx = (slw + slh + 1) >> 1;
    let (clw, clh) = (slw.min(3), slh.min(3));
    let tx2dszctx = clw + clh;
    // has_chroma: this leaf carries the (possibly shared) chroma. forced_inter: `lbs != cbs`
    // (incl. cbs=-1) ⇒ dav2d forces b->intra=0 with NO is_inter symbol (decode.c:1679).
    let has_chroma = cbs != -1;
    let forced_inter = cbs != bs as i32;
    if std::env::var("MTRACE2").is_ok() { crate::dlog!("[IL] y={by4} x={bx4} bs={bs} intra_region={intra_region} forced_inter={forced_inter} cbs={cbs} r={} d={:x}", msac.rng, msac.dif); }

    // SDP intra-region leaf: forced intra, luma-only, NO is_inter symbol (dav2d 1695). fsc uses
    // the neighbour-sum ctx (inter_frame=false) since `IS_INTER && !intra_region` is false here.
    if intra_region {
        // allow_intrabc=FALSE: the intrabc gate is `... && !intra_region` (dav2d 1732), so an
        // intra-region leaf does NOT code an intrabc flag. When this leaf is the SB's first, it
        // still decodes the once-per-SB filters (gdf/cdef) via the threaded left_cdef/left_ccso.
        let info = decode_b_luma(
            msac, cdf, a_nb, l_nb, bs, bx4, by4, have_left, have_top, decode_filters, 0, 3, left_cdef,
            left_ccso, -1, false, false, false, 2,
        );
        for yy in by4..by4 + bh4 {
            for xx in bx4..bx4 + bw4 {
                luma_dir_map[(yy & 15) * 16 + (xx & 15)] = info.midx;
            }
        }
        if std::env::var("SBTRACE").is_ok() { crate::dlog!("SBLEAF-LUMA ({bx4},{by4}) bs={bs} midx={} rng={} dif={:x}", info.midx, msac.rng, msac.dif); }
        // Intra block splats intra=1 + base inter state (ref0=0 unavailable, motion_mode=0),
        // else a later block reads a stale ref/warp neighbour (its is_inter/warp_ctx go wrong).
        splat_inter_nb(a_nb, l_nb, bx4, by4, bw4, bh4, 1, 0, 0, 0, 0, 0, -1);
        splat_nb(&mut a_nb.comp_type, &mut l_nb.comp_type, bx4, by4, bw4, bh4, 0);
        // Intra-region leaf clears its temporal-field footprint (avm copy_frame_mvs NONE cells).
        crate::av2_refmvs::rp_write(bx4, by4, bw4, bh4, crate::av2_refmvs::TemporalBlock::default());
        // Splat this intra region into the refmvs grid (ref=-1) so the warp neighbour walk steps
        // over it (brick B). The chroma-only region root (cbs=-1 subtree) still covers luma cells.
        crate::av2_refmvs::GRID.with(|g| g.borrow_mut().splat_intra(bx4, by4, bw4, bh4, bs as u8));
        // dav2d `splat_intraref` bank_update: this luma-only SDP intra leaf also refreshes avail.
        crate::av2_refmvs::BANK.with(|bk| bk.borrow_mut().bank_update_intra(bw4, bh4, by4, bx4, sb_step4(), sb_step4() >> 5));
        return;
    }

    // skip_mode (dav2d decode.c:1658): coded BEFORE is_inter for a block with bw4*bh4 > 2 (not an
    // intra region) when the frame enables it. ctx = sum of the two nx neighbours' skip_mode. A
    // skip_mode=1 block is compound-implied and codes only a skip_mode_drl_idx DRL loop (handled in
    // decode_b_inter).
    let (nx_sm, nctx_sm) = nx_setup(have_left, have_top, bx4, by4, bw4, bh4, row_end, col_end);
    let skip_mode = if !intra_region
        && (bw4 * bh4) > 2
        && HDR_TOOL_CFG.with(|c| c.get().skip_mode_enabled)
    {
        let ctx = get_skip_mode_ctx(a_nb, l_nb, nx_sm);
        let v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.skip_mode[ctx]);
        if std::env::var("MLEAF").is_ok() { crate::dlog!("[MLEAF] mi=({bx4},{by4}) skip_mode={} ctx={ctx} rng={}", v as u8, msac.rng); }
        v
    } else {
        if std::env::var("MLEAF").is_ok() { crate::dlog!("[MLEAF] mi=({bx4},{by4}) skip_mode NOT-CODED (area {} intra_region={intra_region})", bw4 * bh4); }
        false
    };

    // is_inter — NOT coded for skip_mode (forced inter) or a forced-inter (lbs!=cbs) block.
    let intra = if skip_mode || forced_inter {
        false
    } else {
        let ictx = get_intra_ctx(a_nb, l_nb, nx_sm, nctx_sm);
        let vi = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.intra[ictx]);
        if std::env::var("MLEAF").is_ok() { crate::dlog!("[MLEAF] mi=({bx4},{by4}) is_inter={} ictx={ictx} rng={} cell={:?} cnt={} dif={:x}", vi as u8, msac.rng, cdf.m.intra[ictx], msac.cnt, msac.dif); }
        !vi
    };

    if !intra {
        // Stage E: the MC luma prediction, hoisted out of the refmvs block so the residual (decoded
        // later in the entropy stream) can be added to it → the full pre-filter luma recon. `fmv`
        // (final MV) and `warp_pred` (warp matrix / None=INVALID) are hoisted for the chroma recon.
        let mut luma_pred: Vec<i32> = Vec::new();
        let mut fmv = crate::av2_refmvs::Mv { y: 0, x: 0 };
        let mut warp_pred: Option<[i32; 6]> = None;
        // Compound recon state threaded from the luma blend to the chroma blend:
        // the 2x2 SEG mask, whether SEG produced it, and the (mv pair, list-index refs).
        let mut comp_seg_mask: Vec<u8> = Vec::new();
        let mut comp_has_seg = false;
        let mut comp_preds: Option<([crate::av2_refmvs::Mv; 2], [usize; 2], i8)> = None;
        let leaf_cfg = HDR_TOOL_CFG.with(|c| c.get());
        let info = decode_b_inter(
            msac, cdf, a_nb, l_nb, bs, bx4, by4, have_left, have_top, have_top_in_sb,
            // mvd_sign_derive=true: this stream's seq header sets it (obu.rs:862), so a
            // translation NEWMV block derives its last MV sign from sum_mvd&1 (dav2d 3254).
            // scc / force_integer_mv come from the FRAME header (avm is_mvd_sign_derive_allowed
            // reconinter.h:1317: allow_screen_content_tools kills sign-derive — SCC blocks read
            // explicit sign bypass bits).
            is_sb_boundary, row_end, col_end, bw4, bh4, 2, leaf_cfg.force_integer_mv != 0, 3,
            SEQ_TOOLS.with(|c| c.get().motion_modes),
            SEQ_TOOLS.with(|c| c.get().mvd_sign_derive), leaf_cfg.allow_scc, !forced_inter,
            false, SEQ_TOOLS.with(|c| c.get().adaptive_mvd), decode_filters, left_cdef, left_ccso, true, has_chroma, 2,
            SEQ_TOOLS.with(|c| c.get().six_param_warp), skip_mode,
        );
        // Per-block reference selection: point REF_FRAME1 at the picture this block's single_ref
        // picked (`REF_PICS[refidx[ref0]]`). A B-frame block may use ref 0 OR ref 1; single-ref
        // frames stay on the primary (no reload). Do this BEFORE the MC below reads REF_FRAME1.
        {
            let refidx = CUR_FRAME_REFIDX.with(|c| c.get()).1;
            crate::av2_frame::ensure_ref1_slot(refidx[info.ref0 as usize] as usize);
        }
        // Inter leaves carry no palette — refresh the neighbour palette caches with none.
        crate::av2_palette::pal_splat(bx4, by4, bw4, bh4, 0, &[0u16; 8]);
        // Splat this inter block's interpolation filter into the per-4px FILTER_GRID so a later
        // sub-8×8 shared-chroma carrier can MC each sub-block's region with its OWN filter.
        crate::av2_frame::FILTER_GRID.with(|fg| {
            let mut g = fg.borrow_mut();
            let gs = crate::av2_frame::FILTER_GRID_STRIDE.with(|c| c.get());
            for j in 0..bh4 {
                let row = (by4 + j) * gs;
                for i in 0..bw4 {
                    if let Some(d) = g.get_mut(row + bx4 + i) { *d = info.filter; }
                }
            }
        });
        // ===== brick B: MV predictor (refmvs_find) + final-MV verification (thread-local grid). =====
        // The DRL predictor is mvstack[drl_idx]; NEWMV(15)/WARPNEWMV(17) add the decoded residual.
        // Splat this block + bank its MV so later blocks read it. (Warp splat + WARPMV/GLOBALMV
        // predictor paths are follow-ups — see [[rav2d-refmvs-find]].)
        {
            use crate::av2_refmvs::{derive_warpmv, get_warpmv_2d, mv_reduce_prec, reconstruct_warp_delta_matrix, refmvs_find, refmvs_find_warp, Mv, BANK, GRID, IDENTITY_WARP, WARPBANK};
            let bdw = crate::av2_decode::BLOCK_DIMENSIONS[bs];
            let bd4 = [bdw[0], bdw[1], bdw[2], bdw[3]];
            let (w4c, h4c) = (bw4.min(col_end - bx4), bh4.min(row_end - by4));
            // Stack strategy by mode (dav decode.c:1203-1251): single-ref → single find;
            // skip_mode + compound modes > NEARMV_NEWMV → ONE compound pair find; compound
            // NEARMV_NEARMV/NEARMV_NEWMV → per-ref single finds zipped (sameref duplicates).
            let comp = info.ref1 >= 0;
            // TIP block: every refmvs-facing ref is TIP_FRAME(7) — the stack search runs the TIP
            // arms (gmv=0, empty-stack fallback), the grid/bank store ref 7 (dav b->ref).
            let eref0: i8 = if info.is_tip { 7 } else { info.ref0 as i8 };
            let z = Mv { y: 0, x: 0 };
            let find1 = |r0: i8, r1: i8| {
                GRID.with(|g| {
                    BANK.with(|bk| { let st = SEQ_TOOLS.with(|c| c.get()); let b = bk.borrow(); refmvs_find(&g.borrow(), bx4, by4, bw4, bh4, r0, r1, z, z, sb_step4(), col_end, row_end, st.drl_reorder, if st.refmvbank { Some(&b) } else { None }, 3, skip_mode) })
                })
            };
            let dump_stk = std::env::var("MSTK2").map_or(false, |v| v == "all" || v == format!("{bx4},{by4}"));
            let (stack, _cnt) = if !comp {
                let r = find1(eref0, -1);
                if dump_stk {
                    let s: Vec<String> = r.0.iter().take(6).map(|c| format!("({},{};w{})", c.mv[0].y, c.mv[0].x, c.weight)).collect();
                    crate::dlog!("[MSTK2] mi=({bx4},{by4}) rf={eref0} n={} mode={} drl={} mvd=({},{}) prec={} {}", r.1, info.inter_mode, info.drl_idx, info.mv_y, info.mv_x, info.mv_prec, s.join(" "));
                }
                r
            } else if skip_mode || info.inter_mode > 19 {
                let (s, c) = find1(info.ref0 as i8, info.ref1);
                if std::env::var("RMVC").is_ok() {
                    for (n, e) in s.iter().take(c).enumerate() {
                        crate::dlog!("RMVC {bx4} {by4} {n}/{c} y={} x={} y2={} x2={} w={} cwp={}", e.mv[0].y, e.mv[0].x, e.mv[1].y, e.mv[1].x, e.weight, e.cwp);
                    }
                }
                (s, c)
            } else if info.ref0 as i8 == info.ref1 {
                let (mut s, c) = find1(info.ref0 as i8, -1);
                for e in s.iter_mut() {
                    e.mv[1] = e.mv[0];
                    e.weight *= 0x101;
                }
                if std::env::var("RMVC").is_ok() {
                    for (n, e) in s.iter().take(c).enumerate() {
                        crate::dlog!("RMVZ {bx4} {by4} 0 {n}/{c} y={} x={} w={}", e.mv[0].y, e.mv[0].x, e.weight & 0xff);
                        crate::dlog!("RMVZ {bx4} {by4} 1 {n}/{c} y={} x={} w={}", e.mv[1].y, e.mv[1].x, (e.weight >> 8) & 0xff);
                    }
                }
                (s, c)
            } else {
                let (mut s, c) = find1(info.ref0 as i8, -1);
                let (s2, c2) = find1(info.ref1, -1);
                for n in 0..6 {
                    s[n].mv[1] = s2[n].mv[0];
                    s[n].weight = (s[n].weight & 0xff) | (s2[n].weight << 8);
                }
                if std::env::var("RMVC").is_ok() {
                    for n in 0..c.max(c2) {
                        if n < c {
                            crate::dlog!("RMVZ {bx4} {by4} 0 {n}/{c} y={} x={} w={}", s[n].mv[0].y, s[n].mv[0].x, s[n].weight & 0xff);
                        }
                        if n < c2 {
                            crate::dlog!("RMVZ {bx4} {by4} 1 {n}/{c2} y={} x={} w={}", s[n].mv[1].y, s[n].mv[1].x, (s[n].weight >> 8) & 0xff);
                        }
                    }
                }
                (s, c)
            };
            let is_warp = info.motion_mode >= 2; // MM_WARP_CAUSAL(2) or MM_WARP_DELTA(3)
            // warp[] (corner + neighbour models + fill) is for the WARPMV predictor + the DELTA
            // matrix. MM_WARP_CAUSAL uses derive_warpmv (a least-squares fit) instead.
            let warp = if info.motion_mode == 3 || info.inter_mode == 16 {
                GRID.with(|g| WARPBANK.with(|wb| refmvs_find_warp(&g.borrow(), &wb.borrow(), bx4, by4, bd4, info.ref0 as i8, IDENTITY_WARP, sb_step4(), col_end, row_end)))
            } else {
                [IDENTITY_WARP; 4]
            };
            // Predictor: WARPMV(16) uses get_warpmv_2d(warp[wri]) at mv_prec if it carries an MVD
            // (else 6); others use mvstack[drl_idx].
            let mut fmv1 = z;
            let mut cwp: i8 = info.cwp;
            fmv = if info.inter_mode == 16 {
                let prec = if info.warpmv_with_mvd { info.mv_prec } else { 6 };
                get_warpmv_2d(&warp[info.warp_ref_idx.min(3)], bx4 as i32, by4 as i32, bw4 as i32, bh4 as i32, col_end as i32, row_end as i32, prec)
            } else {
                stack[info.drl_idx].mv[0]
            };
            let pred = fmv;
            if comp {
                if skip_mode {
                    // skip_mode: the DRL candidate pair verbatim + its cwp (dav decode.c:1220).
                    fmv = stack[info.drl_idx].mv[0];
                    fmv1 = stack[info.drl_idx].mv[1];
                    cwp = stack[info.drl_idx].cwp;
                } else if info.inter_mode == 21 {
                    // GLOBALMV_GLOBALMV: both arms = gmv (identity → 0 for this stream).
                    fmv = z;
                    fmv1 = z;
                } else {
                    // per-arm: mv[n] = stack[drl_idx[n]].mv[n]; NEWMV arms add the residual after
                    // reduce_prec with the packed per-arm precision nibble (dav decode.c:1274).
                    const PRED_MODES_R: [[u8; 2]; 11] = [
                        [13, 13], [13, 15], [15, 13], [14, 14], [15, 15], [15, 15],
                        [13, 13], [13, 15], [15, 13], [15, 15], [15, 15],
                    ];
                    let drls = [info.drl_idx, info.drl_idx1];
                    let diffs = [(info.mv_y, info.mv_x), (info.mv_y1, info.mv_x1)];
                    let midx = (info.inter_mode - 18) as usize;
                    let mut out = [z; 2];
                    for n in 0..2 {
                        let mut m = stack[drls[n]].mv[n];
                        if PRED_MODES_R[midx][n] == 15 {
                            let prec = (info.mv_prec >> (n * 4)) & 0xf;
                            if !info.amvd && prec <= 3 {
                                mv_reduce_prec(&mut m, prec);
                            }
                            m.y += diffs[n].0;
                            m.x += diffs[n].1;
                        }
                        out[n] = m;
                    }
                    fmv = out[0];
                    fmv1 = out[1];
                }
            } else if info.inter_mode == 15 || info.inter_mode == 17 || (info.inter_mode == 16 && info.warpmv_with_mvd) {
                // NEWMV(15)/WARPNEWMV(17)/WARPMV-with-mvd(16) add the decoded residual after reduce_prec.
                if !info.amvd && info.mv_prec <= 3 {
                    mv_reduce_prec(&mut fmv, info.mv_prec);
                }
                fmv.y += info.mv_y;
                fmv.x += info.mv_x;
            }
            let _ = &fmv1;
            let _ = cwp;
            // RMVFIN harness: mine's final MV per inter block (vs oracle DMV). Broaden the
            // `by4 < 60` gate to all blocks to re-measure the full-frame MV match.
            if std::env::var("RMVFIN").is_ok() {
                let oh = crate::av2_recon::CUR_FRAME_REF.with(|c| c.get().0);
                if comp {
                    crate::dlog!("RMVFIN oh{oh} {bx4} {by4} {} {} | {} {}", fmv.y, fmv.x, fmv1.y, fmv1.x);
                } else {
                    crate::dlog!("RMVFIN oh{oh} {bx4} {by4} {} {}", fmv.y, fmv.x);
                }
            }
            // ===== Stage D MC harness: single-ref MM_SIMPLE translational luma prediction, scored
            // vs dav2d's frame-2 prediction plane. Gated to the first SB (refmvs MVs bit-exact). =====
            if info.motion_mode == 0 && info.bawp == 0 {
                use crate::av2_frame::{INTER_SCORE, REF_F2PRED, REF_FRAME1};
                REF_FRAME1.with(|rf| {
                    REF_F2PRED.with(|rp| {
                        if let (Some(r0), Some(pr)) = (rf.borrow()[0].as_ref(), rp.borrow().as_ref()) {
                            let (w, h) = (bw4 * 4, bh4 * 4);
                            let mut pred = vec![0i32; w * h];
                            crate::av2_inter::mc_translate_luma(r0, &mut pred, w, bx4 * 4, by4 * 4, w, h, fmv.y, fmv.x, info.filter as usize, bdmax_g());
                            let mut ok = true;
                            'sc: for yy in 0..h {
                                for xx in 0..w {
                                    let (px, py) = (bx4 * 4 + xx, by4 * 4 + yy);
                                    if px >= pr.w || py >= pr.h {
                                        continue;
                                    }
                                    if pred[yy * w + xx] != pr.at(px, py) {
                                        ok = false;
                                        crate::dlog!("IMCMISS ({bx4},{by4}) w={w} h={h} fmv={},{} filt={} at({xx},{yy}) mine={} dav={}", fmv.y, fmv.x, info.filter, pred[yy * w + xx], pr.at(px, py));
                                        break 'sc;
                                    }
                                }
                            }
                            INTER_SCORE.with(|s| {
                                let (o, t) = s.get();
                                s.set((o + ok as u32, t + 1));
                            });
                        }
                    })
                });
            }
            // Chroma MC harness (same luma MV, 4:2:0 subsample). Gated to ≥8×8 blocks aligned to
            // the chroma grid (bx4/by4 even) with their own chroma; odd-aligned blocks share chroma
            // with an even neighbour (dav captures at (bx4>>1)*4,(by4>>1)*4) — excluded for now.
            if info.motion_mode == 0 && info.bawp == 0 && bw4 >= 2 && bh4 >= 2 && bx4 % 2 == 0 && by4 % 2 == 0 {
                use crate::av2_frame::{INTER_SCORE_C, REF_F2PREDC, REF_FRAME1};
                REF_FRAME1.with(|rf| {
                    REF_F2PREDC.with(|rp| {
                        let (rfb, rpb) = (rf.borrow(), rp.borrow());
                        let (cw, ch, cpx, cpy) = (bw4 * 2, bh4 * 2, bx4 * 2, by4 * 2);
                        let (mut oku, mut okv) = (false, false);
                        for pl in 0..2 {
                            if let (Some(r), Some(pr)) = (rfb[pl + 1].as_ref(), rpb[pl].as_ref()) {
                                let mut pred = vec![0i32; cw * ch];
                                crate::av2_inter::mc_translate(r, &mut pred, cw, cpx, cpy, cw, ch, fmv.y, fmv.x, info.filter as usize, ss_g().0 as u32, ss_g().1 as u32, bdmax_g());
                                let mut ok = true;
                                'cc: for yy in 0..ch {
                                    for xx in 0..cw {
                                        if cpx + xx >= pr.w || cpy + yy >= pr.h {
                                            continue;
                                        }
                                        if pred[yy * cw + xx] != pr.at(cpx + xx, cpy + yy) {
                                            ok = false;
                                            break 'cc;
                                        }
                                    }
                                }
                                if pl == 0 { oku = ok } else { okv = ok }
                                if !ok {
                                    'f: for yy in 0..ch { for xx in 0..cw {
                                        if cpx + xx < pr.w && cpy + yy < pr.h && pred[yy*cw+xx] != pr.at(cpx+xx, cpy+yy) {
                                            crate::dlog!("IMCMISSC ({bx4},{by4}) pl={pl} cw={cw} ch={ch} fmv={},{} filt={} at({xx},{yy}) mine={} dav={} ref1={}",
                                                fmv.y, fmv.x, info.filter, pred[yy*cw+xx], pr.at(cpx+xx, cpy+yy), r.at(cpx+xx, cpy+yy));
                                            break 'f;
                                        }
                                    }}
                                }
                            }
                        }
                        INTER_SCORE_C.with(|s| {
                            let (u, v, t) = s.get();
                            s.set((u + oku as u32, v + okv as u32, t + 1));
                        });
                    })
                });
            }
            // Warp splat: MM_WARP_CAUSAL(2) fits derive_warpmv; MM_WARP_DELTA(3) reconstructs off
            // warp[wri]+delta; MM_WARP_EXTEND(4) extends a neighbour's warp (warp_extend, may be
            // INVALID → uniform). Push valid warps to the warp bank; non-warp/INVALID splat uniform.
            // `warp_pred` keeps the raw Option: None = dav's warpmv.type == INVALID (derive/extend
            // failed) → dav uses translational mc(), NOT warp_affine/ext_warp (recon_tmpl.c:3220).
            warp_pred = if is_warp {
                if info.motion_mode == 2 {
                    GRID.with(|g| derive_warpmv(&g.borrow(), bx4, by4, bw4, bh4, w4c, h4c, info.ref0 as i8, fmv, sb_step4(), col_end))
                } else if info.motion_mode == 4 {
                    let drl_off = (stack[info.drl_idx].x_off, stack[info.drl_idx].y_off);
                    let bd = crate::av2_decode::BLOCK_DIMENSIONS[bs];
                    GRID.with(|g| crate::av2_refmvs::warp_extend(&g.borrow(), bx4, by4, bw4, bh4, bd[2] as u32, bd[3] as u32, info.ref0 as i8, fmv, drl_off, sb_step4(), col_end, row_end, IDENTITY_WARP))
                } else {
                    Some(reconstruct_warp_delta_matrix(warp[info.warp_ref_idx.min(3)], info.warp_delta, fmv, bw4, bh4, bx4, by4))
                }
            } else {
                None
            };
            // Refmvs/grid-facing matrix: MM_WARP_CAUSAL with a FAILED derive still splats IDENTITY to
            // the grid (bit-exact refmvs, brick B); do NOT collapse this to `warp_pred`.
            let warp_m: Option<[i32; 6]> = if is_warp && info.motion_mode == 2 {
                Some(warp_pred.unwrap_or(IDENTITY_WARP))
            } else {
                warp_pred
            };
            let (mf, matrix) = match warp_m {
                Some(m) => {
                    WARPBANK.with(|wb| wb.borrow_mut().add(info.ref0 as i8, m));
                    (2u8, m)
                }
                None => (0u8, [0i32; 6]),
            };
            // ===== Stage D warp-MC harness: warp_affine luma prediction vs dav's frame-2 plane.
            // Gated to first-SB warp blocks (matrix from refmvs is bit-exact there). =====
            if is_warp && info.bawp == 0 {
                use crate::av2_frame::{INTER_SCORE_W, REF_F2PRED, REF_FRAME1};
                let (w, h) = (bw4 * 4, bh4 * 4);
                if w >= 8 && h >= 8 {
                    if let Some(abcd) = crate::av2_warp::get_shear_params(&matrix) {
                        REF_FRAME1.with(|rf| {
                            REF_F2PRED.with(|rp| {
                                if let (Some(r0), Some(pr)) = (rf.borrow()[0].as_ref(), rp.borrow().as_ref()) {
                                    let mut pred = vec![0i32; w * h];
                                    crate::av2_warp::warp_affine(r0, &mut pred, w, &matrix, &abcd, bx4, by4, w, h, 0, 0, bdmax_g());
                                    let mut ok = true;
                                    'w: for yy in 0..h {
                                        for xx in 0..w {
                                            let (px, py) = (bx4 * 4 + xx, by4 * 4 + yy);
                                            if px >= pr.w || py >= pr.h {
                                                continue;
                                            }
                                            if pred[yy * w + xx] != pr.at(px, py) {
                                                ok = false;
                                                crate::dlog!("IWMISS ({bx4},{by4}) im={} mm={} w={w} h={h} at({xx},{yy}) mine={} dav={} abcd={:?}", info.inter_mode, info.motion_mode, pred[yy * w + xx], pr.at(px, py), abcd);
                                                break 'w;
                                            }
                                        }
                                    }
                                    INTER_SCORE_W.with(|s| {
                                        let (o, t) = s.get();
                                        s.set((o + ok as u32, t + 1));
                                    });
                                }
                            })
                        });
                        // Warp CHROMA (luma ≥16×16 → chroma ≥8px 8x8-warp path; dav uses ext_warp
                        // below that). Same matrix, ss=1, grid-aligned.
                        if bw4 >= 4 && bh4 >= 4 && bx4 % 2 == 0 && by4 % 2 == 0 {
                            use crate::av2_frame::{INTER_SCORE_WC, REF_F2PREDC};
                            REF_FRAME1.with(|rf| {
                                REF_F2PREDC.with(|rp| {
                                    let (rfb, rpb) = (rf.borrow(), rp.borrow());
                                    let (cw, ch, cpx, cpy) = (bw4 * 2, bh4 * 2, bx4 * 2, by4 * 2);
                                    let (mut oku, mut okv) = (false, false);
                                    for pl in 0..2 {
                                        if let (Some(r), Some(pr)) = (rfb[pl + 1].as_ref(), rpb[pl].as_ref()) {
                                            let mut pred = vec![0i32; cw * ch];
                                            crate::av2_warp::warp_affine(r, &mut pred, cw, &matrix, &abcd, bx4, by4, cw, ch, ss_g().0 as u32, ss_g().1 as u32, bdmax_g());
                                            let mut ok = true;
                                            'wc: for yy in 0..ch {
                                                for xx in 0..cw {
                                                    if cpx + xx < pr.w && cpy + yy < pr.h && pred[yy * cw + xx] != pr.at(cpx + xx, cpy + yy) {
                                                        ok = false;
                                                        break 'wc;
                                                    }
                                                }
                                            }
                                            if pl == 0 { oku = ok } else { okv = ok }
                                        }
                                    }
                                    INTER_SCORE_WC.with(|s| {
                                        let (u, v, t) = s.get();
                                        s.set((u + oku as u32, v + okv as u32, t + 1));
                                    });
                                })
                            });
                        }
                    }
                }
            }
            // Splat + bank the FULL pair (mf carries cwp<<2 — feeds neighbours' comp candidates).
            // dav packs cwp into mf ONLY for compound blocks (splat_tworef_mv: mf = cwp<<2 | …;
            // splat_oneref_mv leaves mf = gmv|warp bits only).
            let mf_pair = mf | if comp { (cwp as u8) << 2 } else { 0 };
            let mv1_cell = if comp { fmv1 } else { Mv { y: -0x8000, x: -0x8000 } };
            GRID.with(|g| g.borrow_mut().splat_pair(bx4, by4, bw4, bh4, [fmv, mv1_cell], (eref0, info.ref1), bs as u8, mf_pair, matrix));
            BANK.with(|bk| bk.borrow_mut().add_block_pair(bw4, bh4, by4, bx4, sb_step4(), sb_step4() >> 5, eref0, info.ref1, [fmv, fmv1], cwp));
            // Temporal motion-field write (dav splat_oneref/tworef_mv t arm, decode.c:574/655):
            // quantized MV pair + ref pair. Single: ref=(r0,r0), both arms = q(mv0), INVALID →
            // pair (-1,-1). Compound: t_swap orders the pair by ref_flip (poc/sign ordering);
            // INVALID arms fall back to the other. (Wedge per-cell masks + opfl/refine-skip +
            // warp per-cell mvs are follow-ups — those blocks currently store the block MV.)
            {
                use crate::av2_refmvs::{quantize_mv, rp_write, TemporalBlock, INVALID_TRAJ};
                let mut tb = TemporalBlock::default();
                if !comp && (mf & 2) != 0 {
                    // WARP block: per-8x8-cell warp-projected temporal MVs (dav splat_warpmv_c,
                    // refmvs.c:2312): mv sampled at (bx+1+2k, by+1+2j) in the warp model, >>11.
                    let m = matrix;
                    let mut mvy_row = (m[4] as i64) * (bx4 as i64 + 1) + ((m[1] as i64) >> 2)
                        + ((m[5] as i64 - 0x10000) * (by4 as i64 + 1));
                    let mut mvx_row = ((m[2] as i64 - 0x10000) * (bx4 as i64 + 1))
                        + (m[3] as i64) * (by4 as i64 + 1) + ((m[0] as i64) >> 2);
                    let mut j = 0usize;
                    while j < bh4 {
                        if !crate::av2_recon::work_tick("av2_recon:4392") { break; }
                        let mut mvxi = mvx_row;
                        let mut mvyi = mvy_row;
                        let mut k = 0usize;
                        while k < bw4 {
                            if !crate::av2_recon::work_tick("av2_recon:4396") { break; }
                            let wy = {
                                let a = ((mvyi.abs() + 1024) >> 11) as i32;
                                (if mvyi < 0 { -a } else { a }).clamp(-0xffff, 0xffff)
                            };
                            let wx = {
                                let a = ((mvxi.abs() + 1024) >> 11) as i32;
                                (if mvxi < 0 { -a } else { a }).clamp(-0xffff, 0xffff)
                            };
                            let q = quantize_mv(Mv { y: wy, x: wx });
                            let ctb = if q == INVALID_TRAJ {
                                TemporalBlock::default()
                            } else {
                                TemporalBlock { ref_: (info.ref0 as i8, info.ref0 as i8), qmv: [q, q] }
                            };
                            rp_write(bx4 + k, by4 + j, 2, 2, ctb);
                            mvxi += (m[2] as i64 - 0x10000) * 2;
                            mvyi += m[4] as i64 * 2;
                            k += 2;
                        }
                        mvx_row += m[3] as i64 * 2;
                        mvy_row += (m[5] as i64 - 0x10000) * 2;
                        j += 2;
                    }
                    // per-cell write done; skip the block-level write below.
                    tb.ref_ = (-2, -2); // sentinel: handled
                } else if info.is_tip {
                    // TIP block: NO generic temporal write (dav decode.c:571 t dst = NULL);
                    // tip_pred writes per-8x8 cells (ref pair = tip source pair) itself.
                    tb.ref_ = (-2, -2); // sentinel: handled
                } else if comp && info.inter_mode >= 24 && info.refine_mv != 0 && info.comp_type == 1 {
                    // opfl && refinemv (dav decode.c:653): parse writes nothing; opfl_pred writes
                    // the refined per-cell mvs itself.
                    tb.ref_ = (-2, -2); // sentinel: handled
                } else if !comp {
                    let q = quantize_mv(fmv);
                    if q == INVALID_TRAJ {
                        tb.ref_ = (-1, -1);
                    } else {
                        tb.ref_ = (info.ref0 as i8, info.ref0 as i8);
                        tb.qmv = [q, q];
                    }
                } else {
                    let t_swap = ref_flip_pair(info.ref0 as i8, info.ref1) as usize;
                    let mvp = [fmv, fmv1];
                    let refp = [info.ref0 as i8, info.ref1];
                    // raw (pre-fallback) t-swapped pair — the wedge per-cell store reads it.
                    let mut raw_q = [0u16; 2];
                    raw_q[t_swap] = quantize_mv(mvp[0]);
                    raw_q[1 - t_swap] = quantize_mv(mvp[1]);
                    let mut r = [0i8; 2];
                    r[t_swap] = refp[0];
                    r[1 - t_swap] = refp[1];
                    if info.comp_type == 2 && info.wedge_idx >= 0 {
                        // WEDGE temporal store (dav refmvs.c splat_comp_wedgemv): per 2x2 cell the
                        // TMVP winner map picks a single side (or both, with the plain fallback).
                        let tm = crate::av2_wedge::wedge_tmvp(bw4, bh4, info.wedge_idx as usize);
                        let w_swap = (info.wedge_sign as usize) ^ t_swap;
                        for cy in 0..bh4 / 2 {
                            for cx in 0..bw4 / 2 {
                                let d = tm[cy * (bw4 / 2) + cx] as usize;
                                let mut cell = TemporalBlock::default();
                                if d != 2 {
                                    let idx = 1 - (d ^ w_swap);
                                    let m = raw_q[idx];
                                    cell.qmv = [m, m];
                                    cell.ref_ = if m == INVALID_TRAJ { (-1, -1) } else { (r[idx], r[idx]) };
                                } else {
                                    cell.qmv = raw_q;
                                    cell.ref_ = (r[0], r[1]);
                                    if cell.qmv[0] == INVALID_TRAJ {
                                        if cell.qmv[1] == INVALID_TRAJ {
                                            cell.ref_ = (-1, -1);
                                        } else {
                                            cell.qmv[0] = cell.qmv[1];
                                            cell.ref_.0 = cell.ref_.1;
                                        }
                                    } else if cell.qmv[1] == INVALID_TRAJ {
                                        cell.qmv[1] = cell.qmv[0];
                                        cell.ref_.1 = cell.ref_.0;
                                    }
                                }
                                rp_write(bx4 + cx * 2, by4 + cy * 2, 2, 2, cell);
                            }
                        }
                        tb.ref_ = (-2, -2); // handled per cell
                    } else {
                        tb.qmv = raw_q;
                        tb.ref_ = (r[0], r[1]);
                        if tb.qmv[0] == INVALID_TRAJ {
                            if tb.qmv[1] == INVALID_TRAJ {
                                tb.ref_ = (-1, -1);
                            } else {
                                tb.qmv[0] = tb.qmv[1];
                                tb.ref_.0 = tb.ref_.1;
                            }
                        } else if tb.qmv[1] == INVALID_TRAJ {
                            tb.qmv[1] = tb.qmv[0];
                            tb.ref_.1 = tb.ref_.0;
                        }
                    }
                }
                if tb.ref_ != (-2, -2) {
                    rp_write(bx4, by4, bw4, bh4, tb);
                }
            }
            // Stage E: build the MC luma prediction into `luma_pred` (residual added after the coefs
            // decode below). Mirrors dav's dispatch (recon_tmpl.c:3220 → warp_affine → ext_warp):
            //   warp_pred None (type INVALID) → translational mc();
            //   warp_pred Some + valid shear + min(w,h)≥8 → warp8x8 (warp_affine);
            //   warp_pred Some + (!affine || min<8) → ext_warp (per-4x4 affine MC).
            {
                let (w, h) = (bw4 * 4, bh4 * 4);
                luma_pred = vec![0i32; w * h];
                if std::env::var("OPFLDBG").is_ok() && ((bx4 == 83 && by4 == 10) || (bx4 == 81 && by4 == 10) || (bx4 == 12 && by4 == 0)) {
                    crate::dlog!("[MBLK] mi=({bx4},{by4}) bw4={bw4} bh4={bh4} tip={} comp={comp} im={} rmv={} ct={} r0={} r1={} fmv=({},{}) fmv1=({},{})",
                        info.is_tip as u8, info.inter_mode, info.refine_mv, info.comp_type, info.ref0, info.ref1, fmv.y, fmv.x, fmv1.y, fmv1.x);
                }
                if info.is_tip {
                    // ===== TIP block (dav recon 3241): tip_pred fills the dual preds from the
                    // per-8x8 projected TIP field + refinement, blends avg/bacp. =====
                    let _ = tip_pred_luma(&mut luma_pred, bx4, by4, bw4, bh4, w4c, h4c, fmv, info.filter, col_end, row_end);
                } else if comp {
                    // ===== COMPOUND dual prep-MC + blend (dav recon_tmpl.c:3237-3321). Both refs
                    // fetched straight from REF_PICS[refidx[ref_i]]; blend per comp_type. The
                    // opfl/refine-mv refinement + wedge masks + compound warp are follow-ups
                    // (wedge falls back to avg; refine/opfl blocks blend unrefined preds). =====
                    use crate::av2_inter::{bacp_mask, comp_avg, comp_mask, comp_w_avg, comp_w_mask_ss, mc_translate_prep};
                    let refidx = CUR_FRAME_REFIDX.with(|c| c.get()).1;
                    let mvp = [fmv, fmv1];
                    let refs = [info.ref0 as usize, info.ref1 as usize];
                    let mut t = [vec![0i32; w * h], vec![0i32; w * h]];
                    // opfl/refine blocks take the refined dual prediction (dav recon 3243-3246);
                    // bacp_state: -1 = normal path decides (oof check), else the opfl bacp result.
                    let opfl_block = info.inter_mode >= 24 || (info.refine_mv != 0 && info.comp_type == 1);
                    let mut bacp_state: i32 = -1;
                    if opfl_block {
                        bacp_state = opfl_pred_luma(&mut t, bx4, by4, bw4, bh4, w4c, h4c, mvp, refs,
                                                    info.filter, cwp, info.inter_mode, info.refine_mv,
                                                    info.comp_type, col_end, row_end) as i32;
                    } else {
                        crate::av2_frame::REF_PICS.with(|rp| {
                            let pics = rp.borrow();
                            for i in 0..2 {
                                if let Some(p) = pics[refidx[refs[i]] as usize].as_ref() {
                                    mc_translate_prep(&p[0], &mut t[i], w, bx4 * 4, by4 * 4, w, h, mvp[i].y, mvp[i].x, info.filter as usize, 0, 0);
                                }
                            }
                        });
                    }
                    let (fw_px, fh_px) = ((col_end * 4) as i32, (row_end * 4) as i32);
                    comp_seg_mask = vec![0u8; (w / 2).max(1) * (h / 2).max(1)];
                    match info.comp_type {
                        2 => {
                            // WEDGE (dav recon_tmpl.c:3329): blend with the codebook mask;
                            // first operand = tmp[wedge_sign].
                            let mask = crate::av2_wedge::wedge_mask(bw4, bh4, info.wedge_idx as usize, 0);
                            let (ta, tb) = if info.wedge_sign { (&t[1], &t[0]) } else { (&t[0], &t[1]) };
                            comp_mask(&mut luma_pred, w, ta, tb, w, h, &mask, bdmax_g());
                        }
                        3 => {
                            // SEG: difference mask; blend order by mask_sign. The 2x2 mask is
                            // written into the persistent SEG_SCRATCH at dav's position/stride
                            // (recon 3286: stride = min(bw4*2,64); <16 blocks at the base, else
                            // offset ((by4>>1)&15)*4*stride + ((bx4>>1)&15)*4).
                            let ms = info.mask_sign as usize;
                            let (ta, tb) = if ms == 1 { (&t[1], &t[0]) } else { (&t[0], &t[1]) };
                            let (sssh, sssv) = ss_g();
                            let (mw, mh) = (w >> sssh, h >> sssv);
                            let mstride = mw.min(64);
                            // dav recon 3288: blocks with min(bw4,bh4) < 16 (4px units, i.e.
                            // < 64px) write at the scratch BASE; only >=64px blocks offset.
                            let moff = if bw4.min(bh4) < 16 {
                                0
                            } else {
                                ((((by4 << 2) >> sssv) & 63 & !3) * mstride) + (((bx4 << 2) >> sssh) & 63 & !3)
                            };
                            SEG_SCRATCH.with(|sc| {
                                let mut scr = sc.borrow_mut();
                                let mut tmp_mask = vec![0u8; mw * mh];
                                // seed from the scratch (the odd-row fold reads the current cell)
                                for y in 0..mh {
                                    for x in 0..mw {
                                        tmp_mask[y * mw + x] = scr[(moff + y * mstride + x).min(64 * 64 - 1)];
                                    }
                                }
                                comp_w_mask_ss(&mut luma_pred, w, ta, tb, w, h, &mut tmp_mask, mw, info.mask_sign as i32, sssh, sssv, bdmax_g());
                                for y in 0..mh {
                                    for x in 0..mw {
                                        let d = moff + y * mstride + x;
                                        if d < 64 * 64 {
                                            scr[d] = tmp_mask[y * mw + x];
                                        }
                                    }
                                }
                            });
                            comp_seg_mask = Vec::new();
                            comp_has_seg = true;
                        }
                        _ => {
                            let wt = cwp as i32;
                            if wt == 8 {
                                let bacp_on = if bacp_state >= 0 {
                                    // opfl/refine path: the mask (if any) is already in SEG_SCRATCH
                                    bacp_state == 1
                                } else {
                                    // bacp (boundary-aware) applies only when a pred reads off-frame.
                                    let x0 = bx4 as i32 * 4 + (mvp[0].x >> 3);
                                    let y0 = by4 as i32 * 4 + (mvp[0].y >> 3);
                                    let x1 = bx4 as i32 * 4 + (mvp[1].x >> 3);
                                    let y1 = by4 as i32 * 4 + (mvp[1].y >> 3);
                                    let oof = x0 < 0 || x1 < 0 || y0 < 0 || y1 < 0
                                        || x0 + w as i32 >= fw_px || x1 + w as i32 >= fw_px
                                        || y0 + h as i32 >= fh_px || y1 + h as i32 >= fh_px;
                                    let on = oof && info.motion_mode != 2 && info.inter_mode != 21;
                                    if on {
                                        SEG_SCRATCH.with(|sc| {
                                            let mut scr = sc.borrow_mut();
                                            bacp_mask(&mut scr, w, w, h, x0, y0, x1, y1, fw_px, fh_px);
                                        });
                                    }
                                    on
                                };
                                if bacp_on {
                                    SEG_SCRATCH.with(|sc| {
                                        let scr = sc.borrow();
                                        comp_mask(&mut luma_pred, w, &t[0], &t[1], w, h, &scr, bdmax_g());
                                    });
                                } else {
                                    comp_avg(&mut luma_pred, w, &t[0], &t[1], w, h, bdmax_g());
                                }
                            } else {
                                comp_w_avg(&mut luma_pred, w, &t[0], &t[1], w, h, wt, bdmax_g());
                            }
                        }
                    }
                    if std::env::var("MPB").map_or(false, |v| v == format!("{bx4},{by4},{}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()))) {
                        crate::dlog!("[MPB] mi=({bx4},{by4}) mvp={:?} refs={:?} cwp={cwp} bacp={bacp_state} comp_type={} t0row0={:?} t1row0={:?} out={:?}", mvp, refs, info.comp_type, &t[0][..8], &t[1][..8], &luma_pred[..8]);
                    }
                    comp_preds = Some((mvp, refs, cwp));
                } else {
                    crate::av2_frame::REF_FRAME1.with(|rf| {
                        if let Some(r0) = rf.borrow()[0].as_ref() {
                            match warp_pred {
                                Some(m) => match crate::av2_warp::get_shear_params(&m).filter(|_| w.min(h) >= 8) {
                                    Some(abcd) => crate::av2_warp::warp_affine(r0, &mut luma_pred, w, &m, &abcd, bx4, by4, w, h, 0, 0, bdmax_g()),
                                    None => crate::av2_warp::ext_warp(r0, &mut luma_pred, w, &m, bx4, by4, w, h, 0, 0, bdmax_g()),
                                },
                                None => crate::av2_inter::mc_translate_luma(r0, &mut luma_pred, w, bx4 * 4, by4 * 4, w, h, fmv.y, fmv.x, info.filter as usize, bdmax_g()),
                            }
                        }
                    });
                    // Inter-intra blend (dav recon_tmpl.c:3253: `else if (mm == MM_INTERINTRA ||
                    // b->warp_ii) iiblend(...)` — after MC/warp, else-of-bawp, single-ref non-TIP).
                    if info.ii_mode >= 0 && info.bawp == 0 && !info.is_tip {
                        ii_blend(&mut luma_pred, 0, bx4 * 4, by4 * 4, w, h, bx4, by4, bw4, bh4, info.ii_mode);
                    }
                }
            }
        }
        if std::env::var("MMVP").map_or(false, |v| v.parse::<u32>().ok() == Some(crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()))) {
            crate::dlog!("[MMVP] mi=({bx4},{by4}) fmv=({},{}) fmv1=({},{}) drl=({},{}) filt={} warp={:?} refinemv={} mode={} comp={} ii={}", fmv.y, fmv.x, info.mv_y1, info.mv_x1, info.drl_idx, info.drl_idx1, info.filter, warp_pred, info.refine_mv as u8, info.inter_mode, comp_preds.is_some() as u8, info.ii_mode);
        }
        let d240 = std::env::var("DBG00").is_ok() && ((bx4 == 96 && by4 == 32) || (bx4 == 0 && by4 == 1) || (bx4 == 2 && by4 == 2) || (bx4 == 96 && by4 == 44));
        // Also capture the luma txtp: an inter block's chroma INHERITS it (dav2d recon 3643/3657).
        // ===== TX PARTITION (dav decode.c:3324: read_tx_part AFTER the subpel filter) +
        // per-unit inter luma coefs/itx. Under LARGEST mode part==NONE and the single unit ==
        // the old whole-block path. Layout mirrors avm partition_shift_bits (HORZ5/VERT5 are
        // 5-unit non-uniform).
        let tx_part = read_tx_part(msac, cdf, bw4, bh4, false, true, info.skip);
        let tx_layout = tx_part_layout(bw4, bh4, tx_part);
        let (fw4i, fh4i) = crate::av2_frame::FRAME.with(|fr| { let f = fr.borrow(); (f.iw4, f.ih4) });
        // (ux4, uy4, uslw, uslh, cf, txtp, eob, stx) per CODED unit.
        let mut unit_cfs: Vec<(usize, usize, usize, usize, Vec<i32>, u8, i32, u8)> = Vec::new();
        let mut cf_ctx = 0x40u8; // last unit's (drives the block-level splat when part==NONE)
        let mut luma_txtp = DCT_DCT; // last coded unit's (chroma inherits it)
        // >64px blocks read coefs in 64px CHUNKS with the chroma block riding after the FIRST
        // chunk (dav2d read_coef_blocks) — luma(0), U, V, luma(1)... — not all-luma-then-chroma.
        let mut chroma_cc = None;
        if !info.skip {
            for &(ux, uy, utw4, uth4) in &tx_layout {
                let (ubx4, uby4) = (bx4 + ux, by4 + uy);
                // avm skips units whose ORIGIN is off-frame (read_tx_partition TX_INVALID /
                // clamped unit loops) — they code NO symbols.
                if ubx4 >= fw4i || uby4 >= fh4i { continue; }
                let (uslw, uslh) = (utw4.trailing_zeros() as usize, uth4.trailing_zeros() as usize);
                let (uclw, uclh) = (uslw.min(3), uslh.min(3));
                let u_tdc = (uslw + uslh + 1) >> 1;
                let u2d = uclw + uclh;
                let ubw4 = if fw4i > ubx4 { utw4.min(fw4i - ubx4) } else { utw4 };
                let ubh4 = if fh4i > uby4 { uth4.min(fh4i - uby4) } else { uth4 };
                let sctx = crate::av2_coef::skip_ctx_luma(&a_nb.lcoef[ubx4..], &l_nb.lcoef[uby4..], uslw, uslh, &bd) as usize;
                if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({ubx4},{uby4}) pl=0 txs={u_tdc} skipctx={sctx} rng={}", msac.rng); }
                let az = rav1d_msac_decode_bool_adapt(msac, &mut cdf.coef.skip[1][u_tdc][sctx]);
                if d240 { crate::dlog!("C240 luma all_zero={} sctx={sctx} tctx={u_tdc} rng={} dif={:x}", az as u8, msac.rng, msac.dif); }
                let ucfc = if az {
                    0x40u8
                } else {
                    if d240 { crate::av2_coef::COEF_DBG.with(|c| c.set(true)); }
                    let e = crate::av2_coef::decode_eob(msac, &mut cdf.coef, u2d, 1);
                    if d240 { crate::dlog!("C240 luma eob={e} rng={} dif={:x}", msac.rng, msac.dif); }
                    let mut cf = vec![0i32; 1usize << (uslw + uslh + 4)];
                    let (r, txtp, stx) = decode_luma_tx_level(
                        msac, cdf, &mut cf, e, uslw, uslh, uclw, uclh, u_tdc, u2d, ubw4, ubh4,
                        &a_nb.lcoef[ubx4..], &l_nb.lcoef[uby4..],
                    );
                    if d240 { crate::av2_coef::COEF_DBG.with(|c| c.set(false)); crate::dlog!("C240 luma DONE rng={} dif={:x}", msac.rng, msac.dif); }
                    unit_cfs.push((ux, uy, uslw, uslh, cf, txtp, e, stx));
                    // chroma inherits the FIRST unit's txtp (dav recon_tmpl.c:3650 reads
                    // txtp_map at the block's TOP-LEFT cell, splatted by that unit).
                    if ux == 0 && uy == 0 { luma_txtp = txtp; }
                    r
                };
                cf_ctx = ucfc;
                if std::env::var("ILSP").is_ok() && bx4 == 48 && by4 == 0 { crate::dlog!("[ILSP] mi=({bx4},{by4}) unit=({ux},{uy}) {utw4}x{uth4} ucfc={ucfc:x} nlay={} rng={}", tx_layout.len(), msac.rng); }
                if tx_layout.len() > 1 {
                    // per-UNIT lcoef ctx splat (the next unit's skip/dc-sign ctx reads it)
                    for x in ubx4..(ubx4 + ubw4).min(fw4i) { a_nb.lcoef[x] = ucfc; }
                    for y in uby4..(uby4 + ubh4).min(fh4i) { l_nb.lcoef[y] = ucfc; }
                }
                if bw4.max(bh4) > 16 && ux == 0 && uy == 0 && has_chroma {
                    chroma_cc = Some(decode_chroma_coefs(msac, cdf, cnb, cbs as usize, ccbx4, ccby4, 1, false, luma_txtp));
                }
            }
        }
        // Stage E: frame-2 luma RECON = MC prediction (+ BAWP morph) + per-unit residual
        // (dequant -> STX -> inverse transform -> add at the unit offset).
        if !luma_pred.is_empty() {
            let (w, h) = (bw4 * 4, bh4 * 4);
            if info.bawp != 0 {
                let a = bawp_morph(&mut luma_pred, bx4, by4, bw4, bh4, fmv, info.bawp, info.ref0 as usize);
                crate::av2_frame::BAWP_ALPHA.with(|c| c.set(a));
            }
            for (ux, uy, uslw, uslh, cf, utxtp, ueob, ustx) in &unit_cfs {
                use crate::av2_dequant::{cf_max, dequant_coeff, dq_lookup};
                let (uw, uh) = (4usize << uslw, 4usize << uslh);
                let yac = LAST_QIDX.with(|c| c.get());
                let dq = dq_lookup(yac);
                let pels = uw * uh;
                let tx_scale = (pels > 256) as u32 + (pels > 1024) as u32;
                let cfmax = cf_max((bdmax_g() + 1).trailing_zeros());
                let n = uw * uh;
                let dcq = crate::av2_frame::F2_DCQ.with(|c| c.get()[0]);
                let iqm = crate::av2_qm::iqm_slice(0, uw, uh,
                    ((utxtp & 7) & 3) != 1 && (((utxtp >> 5) & 7) & 3) != 1);
                let mut coeff = vec![0i32; n];
                for i in 0..n.min(cf.len()) {
                    let lvl = cf[i];
                    if lvl != 0 {
                        let s = (lvl < 0) as u32;
                        let q = if i == 0 && dcq != 0 { dcq } else { dq };
                        let q = crate::av2_qm::qm_apply(iqm, i, uh.min(32), q);
                        let mag0 = dequant_coeff(lvl.unsigned_abs(), q, 3, cfmax, s, false) as i32;
                        let mag = (mag0 >> tx_scale).min(cfmax);
                        coeff[i] = if lvl < 0 { -mag } else { mag };
                    }
                }
                // Inter secondary transform (STX/IST) on the dequantized unit coeffs.
                if *ustx > 0 {
                    let ty = (*ustx - 1) as usize;
                    let kernel = &crate::av2_stx_tables::STX_8X8_KERNEL_SET0[ty];
                    let map_idx = crate::av2_stx_tables::COEFF8X8_MAPPING_IDX[ty] as usize; // set=0
                    let mapping = &crate::av2_stx_tables::COEFF8X8_MAPPING[map_idx];
                    let stride = uh.min(32);
                    let hh = (*ueob + 1) as usize;
                    let (cmin, cmax) = (-128 * 256, 128 * 256 - 1);
                    let mut sums = [0i32; 48];
                    for (x, sx) in sums.iter_mut().enumerate() {
                        let mut sum = 0i32;
                        for y in 0..hh { sum += coeff[y] * kernel[y][x] as i32; }
                        let m = (sum.abs() + 64) >> 7;
                        *sx = (if sum < 0 { -m } else { m }).clamp(cmin, cmax);
                    }
                    for c in coeff.iter_mut().take(32) { *c = 0; }
                    for (n_i, &rc) in mapping.iter().enumerate() {
                        let (x, y) = ((rc & 7) as usize, (rc >> 3) as usize);
                        if stride > 8 { coeff[y * stride + x] = sums[n_i]; } else { coeff[rc as usize] = sums[n_i]; }
                    }
                }
                // Packed TxfmType -> per-axis 1d types (+ inter DDT remap), per unit dims.
                let mut txtp = *utxtp;
                if crate::av2_frame::INTER_DDT.with(|c| c.get()) {
                    let mask = (if *uslw == 1 || *uslw == 2 { 0x02u8 } else { 0 })
                        | (if *uslh == 1 || *uslh == 2 { 0x40u8 } else { 0 });
                    txtp += txtp & mask;
                }
                const DAV2MINE: [usize; 8] = [0, 3, 1, 2, 4, 5, 6, 7];
                let (row_ty, col_ty) = (DAV2MINE[(txtp & 7) as usize], DAV2MINE[((txtp >> 5) & 7) as usize]);
                let mut residual = vec![0i32; n];
                crate::av2_itx::inv_txfm_2d(&coeff, *uslw, *uslh, row_ty, col_ty, &mut residual);
                let off = uy * 4 * w + ux * 4;
                crate::av2_itx::residual_add(&mut luma_pred[off..], w, &residual, uw.min(w - ux * 4), uh.min(h - uy * 4), 0, 0, 0, bdmax_g());
            }
            crate::av2_frame::REF_F2RECON.with(|r| {
                if let Some(rp) = r.borrow().as_ref() {
                    let mut ok = true;
                    let (mut mx, mut my, mut mmine, mut mdav) = (0, 0, 0, 0);
                    'rc: for yy in 0..h {
                        for xx in 0..w {
                            let (px, py) = (bx4 * 4 + xx, by4 * 4 + yy);
                            if px < rp.w && py < rp.h && luma_pred[yy * w + xx].clamp(0, 255) != rp.at(px, py) {
                                ok = false;
                                mx = xx; my = yy; mmine = luma_pred[yy * w + xx].clamp(0, 255); mdav = rp.at(px, py);
                                break 'rc;
                            }
                        }
                    }
                    if !ok {
                        crate::dlog!("IRECMISS ({bx4},{by4}) w={w} h={h} mm={} im={} skip={} bawp={} txtp={luma_txtp} hascf={} at({mx},{my}) mine={mmine} dav={mdav}", info.motion_mode, info.inter_mode, info.skip as u8, info.bawp, !unit_cfs.is_empty() as u8);
                    }
                    crate::av2_frame::INTER_SCORE_R.with(|s| { let (o, t) = s.get(); s.set((o + ok as u32, t + 1)); });
                }
            });
        }
        // Persist the inter luma recon into the frame-2 FRAME buffer + maintain the decode-order
        // availability grid (`mi_coded`) so later intra blocks predict from these neighbours.
        // Every inter block marks availability (incl. bawp/off-frame); pixels are written for
        // in-frame non-bawp blocks (bawp not yet recon'd — its `luma_pred` lacks the post-scale).
        crate::av2_frame::FRAME.with(|fr| {
            let mut f = fr.borrow_mut();
            if f.pl[0].w != 0 {
                f.ensure_sb(bx4, by4);
                f.mark_coded_avail(bx4, by4, bw4, bh4);
                let (w, h) = (bw4 * 4, bh4 * 4);
                if !luma_pred.is_empty() {
                    // Write the IN-FRAME portion (clip w/h to the frame) — an edge block that spills
                    // past the frame (partial SB) is still reconstructed for its visible pixels
                    // (dav writes the full block into a padded buffer; only the visible part matters).
                    let stride = f.pl[0].stride;
                    let wc = w.min(f.pl[0].w.saturating_sub(bx4 * 4));
                    let hc = h.min(f.pl[0].h.saturating_sub(by4 * 4));
                    for yy in 0..hc {
                        let dst = (by4 * 4 + yy) * stride + bx4 * 4;
                        for xx in 0..wc {
                            f.pl[0].px[dst + xx] = luma_pred[yy * w + xx].clamp(0, bdmax_g());
                        }
                    }
                    crate::av2_frame::write_recon_pad(0, bx4 * 4, by4 * 4, &luma_pred, w, h);
        mscore_luma("inter", bx4 * 4, by4 * 4, w, h, &luma_pred, w);
                }
                // Mark the luma deblock edge grids (joint=0 — inter blocks aren't smooth) so the
                // frame-2 deblock filters this block's boundaries (the intra path does this via
                // recon_intra_luma; the inter path must do it too).
                f.mark_coded(bx4, by4, bw4, bh4, 0);
                if std::env::var("RECONDBG").is_ok() {
                    crate::av2_frame::REF_F2RECON.with(|r| {
                        if let Some(rp) = r.borrow().as_ref() {
                            'chk: for yy in 0..(bh4 * 4) {
                                for xx in 0..(bw4 * 4) {
                                    let (px, py) = (bx4 * 4 + xx, by4 * 4 + yy);
                                    if px < rp.w && py < rp.h && px < f.pl[0].w && py < f.pl[0].h {
                                        let m = f.pl[0].px[py * f.pl[0].stride + px];
                                        if m != rp.at(px, py) {
                                            crate::dlog!("BLKMISS[inter] fn={} ({bx4},{by4}) w={} h={} at({xx},{yy}) mine={m} dav={}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()), bw4 * 4, bh4 * 4, rp.at(px, py));
                                            break 'chk;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                // LR noskip: every coded luma TX unit (dav recon_tmpl.c:2694 eob != -1)
                for (ux, uy, uslw, uslh, _, _, _, _) in &unit_cfs {
                    f.mark_lr_noskip(bx4 + ux, by4 + uy, 1 << uslw, 1 << uslh);
                }
                // TX-partitioned block: per-UNIT deblock tx levels + unit-boundary edges.
                // A SKIP block has NO TX edges (dav create_db_mask skip arm: outer only) — a
                // >64px NONE-skip block must not mark its 64px chunk boundaries.
                if !info.skip && tx_layout.len() > 1 {
                    for &(ux, uy, uw4, uh4) in &tx_layout {
                        if bx4 + ux >= f.iw4 || by4 + uy >= f.ih4 { continue; }
                        f.mark_db(bx4 + ux, by4 + uy, uw4, uh4);
                    }
                }
                // Sub-PU deblock layer (dav lf_mask.c create_db_mask): TIP / compound-opfl /
                // refinemv+AVG blocks cap their edge levels and add weak inner edges.
                let dcfg = crate::av2_frame::DEBLOCK_CFG.with(|c| c.get());
                if dcfg.sub_pu {
                    let spl = subpu_flt_lvl(false, true, info.is_tip, info.ref1 >= 0, info.inter_mode, info.refine_mv != 0, info.comp_type == 1, bw4, bh4);
                    if spl < 3 {
                        let fm2 = HDR_TOOL_CFG.with(|c| c.get().tip_frame_mode) == 2;
                        f.mark_db_subpu(bx4, by4, bw4, bh4, spl, fm2);
                        let (sshm, ssvm) = ss_g();

                    }
                }
            }
        });
        crate::av2_frame::mark_btype(bx4, by4, bw4, bh4, if info.bawp != 0 { 4 } else { 1 });
        // dav clamps the cf_ctx splat to the frame edge (recon_tmpl.c:1226 imin(t_dim->w,
        // f->bw - t->bx)) — out-of-frame cells stay 0x40 and DO feed a later block's
        // dc_sign_ctx sum (hm_q80 (96,44): 4 poisoned cells at bx4=108..112 → ctx 2 vs 0).
        let (fw4, fh4) = crate::av2_frame::FRAME.with(|f| { let f = f.borrow(); (f.iw4, f.ih4) });
        if tx_layout.len() == 1 || info.skip {
            // partitioned blocks already splatted per unit inside the coef loop — EXCEPT a
            // SKIP block (unit loop never runs): its whole extent splats the 0x40 base
            // (dav2d read_coef_blocks skip memset), incl. multi-unit >64px blocks.
            for x in bx4..(bx4 + bw4).min(fw4) { a_nb.lcoef[x] = cf_ctx; }
            for y in by4..(by4 + bh4).min(fh4) { l_nb.lcoef[y] = cf_ctx; }
        }
        // Oracle emits LEAFDIF from the luma tx cf read (not called for block-level skip); mirror
        // so the frame-2 inter luma verification stream aligns with dav2d's.
        if !info.skip {
            if std::env::var("SBTRACE").is_ok() { crate::dlog!("LEAFDIF ({bx4},{by4}) bs={bs} dif={:x} rng={}", msac.dif, msac.rng); }
        }
        if bx4 == 0 && by4 == 0 {
            crate::dlog!("SBL00 post-inter-luma skip={} rng={} dif={:x}", info.skip as u8, msac.rng, msac.dif);
        }
        if std::env::var("HANGCP").is_ok() {
            crate::dlog!("[CP] leaf ({bx4},{by4}) post-luma");
        }
        // Chroma coefs at the chroma block size/origin (== bs/bx4/by4 for a normal yuv leaf;
        // the shared 8x8-unit chroma for a sub-8x8 leaf that carries it). skip_set=1 (inter).
        if std::env::var("HANGCP").is_ok() { crate::dlog!("[CP]   arm-select cc={} skip={} hc={}", chroma_cc.is_some(), info.skip as u8, has_chroma as u8); }
        let chroma_coefs = if chroma_cc.is_some() {
            chroma_cc
        } else if !info.skip && has_chroma {
            Some(decode_chroma_coefs(msac, cdf, cnb, cbs as usize, ccbx4, ccby4, 1, false, luma_txtp))
        } else if has_chroma {
            // Skipped block still updates the chroma neighbour context to base (dav2d set_ctx
            // memsets ccoef=0x40 even with no coded coefficients) — else a later block reads a
            // stale non-base value and picks the wrong skip context.
            let cbd = crate::av2_decode::BLOCK_DIMENSIONS[cbs as usize];
            let (sshc, ssvc) = ss_g();
            let (cslw, cslh) = ((cbd[2] as usize).saturating_sub(sshc), (cbd[3] as usize).saturating_sub(ssvc));
            let (ccbx, ccby) = (ccbx4 >> sshc, ccby4 >> ssvc);
            let (ccw, cch) = (1usize << cslw, 1usize << cslh);
            for p in 0..2 {
                cnb.a[p][ccbx..ccbx + ccw].fill(0x40);
                cnb.l[p][ccby..ccby + cch].fill(0x40);
            }
            // A SKIP inter block still deblocks its OUTER chroma edges (dav mask_outer_edge for
            // the skip branch of create_db_mask). The non-skip path marks these inside
            // decode_chroma_coefs; the skip path must do it explicitly (else chroma under-filters).
            if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
                crate::av2_frame::FRAME.with(|f| f.borrow_mut().mark_db_chroma(ccbx, ccby, ccw, cch));
            }
            None
        } else {
            None
        };
        if std::env::var("HANGCP").is_ok() { crate::dlog!("[CP]   post-arm"); }
        // Stage E chroma RECON (MC prediction + optional residual). Grid-aligned yuv leaves (chroma
        // block == luma block) are bit-exact; sub-8×8 SHARED chroma (`cbs != bs`) needs dav's
        // shared-chroma MV/position rule (the carrier fmv gives small pred deltas) — deferred.
        // Grid-aligned yuv leaves (chroma block == luma block) recon bit-exact. Sub-8×8 SHARED
        // chroma (`cbs != bs`) is DEFERRED: mine's inter sub-8×8 chroma PARTITION diverges from
        // dav (mine predicts an 8×8-unit chroma at positions dav has no chroma block at — a
        // chroma-tree granularity issue, not just an MV rule). See [[rav2d-stage-d-inter]].
        if has_chroma {
            let cbd_c = crate::av2_decode::BLOCK_DIMENSIONS[cbs as usize];
            let (sshc, ssvc) = ss_g();
            let (cw, ch) = ((cbd_c[0] as usize * 4) >> sshc, (cbd_c[1] as usize * 4) >> ssvc);
            if std::env::var("HANGCP").is_ok() { crate::dlog!("[CP]   chroma-recon enter cw={cw} ch={ch} cbs={cbs}"); }
            let (cpx, cpy) = ((ccbx4 * 4) >> sshc, (ccby4 * 4) >> ssvc);
            // Chroma BAWP (dav `b->bawp[1]`, a SEPARATE bit from luma bawp): morph the chroma with
            // the reused luma alpha. Gated on `bawp_chroma`, NOT `bawp` (luma) — they can differ.
            let bawp_alpha = if info.bawp_chroma != 0 {
                Some(crate::av2_frame::BAWP_ALPHA.with(|c| c.get()))
            } else {
                None
            };
            // forced_inter (cbs≠bs) sub-8×8 uses per-luma-sub-block chroma MC (dav recon 3670).
            // avm singleref_for_compound (reconinter.c:3259): thin 4xN/Nx4 blocks (min dim 4px,
            // max >= 16px) predict CHROMA from ref0 only — no compound blend, no opfl/refine.
            let thin_4xn = bw4.min(bh4) == 1 && bw4.max(bh4) >= 4;
            let comp_arg = if thin_4xn { None } else {
                comp_preds.map(|(mvp, refs, rcwp)| {
                    (mvp, refs, info.comp_type, rcwp, info.mask_sign, comp_seg_mask.clone(), bw4 * 2, info.wedge_idx, info.wedge_sign)
                })
            };
            let _ = comp_has_seg;
            let tip_arg = if info.is_tip {
                // TIP chroma step (dav recon 3722): recomputed from the frame/seq tip flags.
                let seq_tip_refine = SEQ_TIP.with(|c| c.get()).4;
                let opfl = seq_tip_refine && HDR_TOOL_CFG.with(|c| c.get()).tip_frame_mode == 1;
                let step = 2usize << ((!opfl && bw4.min(bh4) >= 4) as usize);
                Some((step, step, true, col_end, row_end))
            } else if !thin_4xn && info.ref1 >= 0 && (info.inter_mode >= 24 || (info.refine_mv != 0 && info.comp_type == 1)) {
                // opfl/refine compound chroma (dav recon 3737): r_step = 2<<refine, o_step = 4>>opfl.
                let refine = (info.comp_type == 1 && info.refine_mv != 0) as usize;
                let opfl = (info.inter_mode >= 24) as usize;
                Some((2usize << refine, 4usize >> opfl, false, col_end, row_end))
            } else { None };
            // Inter-intra chroma blend: single-ref non-TIP only (mirrors the luma gate).
            let cii = if info.ref1 < 0 && !info.is_tip && info.bawp == 0 { info.ii_mode } else { -1 };
            recon_inter_chroma(bx4, by4, cpx, cpy, cw, ch, fmv, warp_pred, info.filter, cbs != bs as i32, chroma_coefs.as_ref(), bawp_alpha, comp_arg, tip_arg, cii);
            // Sub-PU deblock CHROMA layer — AFTER the chroma recon (recon_inter_chroma's
            // mark_db_chroma writes the chroma edge grids; the caps must land on top of them).
            let dcfg2 = crate::av2_frame::DEBLOCK_CFG.with(|c| c.get());
            if dcfg2.sub_pu && std::env::var("NOSPC").is_err() {
                let spl2 = subpu_flt_lvl(false, true, info.is_tip, info.ref1 >= 0, info.inter_mode, info.refine_mv != 0, info.comp_type == 1, bw4, bh4);
                if spl2 < 3 {
                    let fm2 = HDR_TOOL_CFG.with(|c| c.get().tip_frame_mode) == 2;
                    let (sshm, ssvm) = ss_g();
                    crate::av2_frame::FRAME.with(|f| {
                        // chroma cells are 4 CHROMA px (cpx/cw are chroma px)
                        f.borrow_mut().mark_db_subpu_chroma(cpx / 4, cpy / 4, (cw / 4).max(1), (ch / 4).max(1), spl2, sshm, ssvm, fm2);
                    });
                }
            }
        }
        // Inter blocks reset the uvmode neighbour context to DC_PRED (dav2d decode.c:3403) so a
        // later CfL block's `cfl_ctx` doesn't read a stale CFL_PRED from an earlier intra chroma.
        if has_chroma {
            let cbd = crate::av2_decode::BLOCK_DIMENSIONS[cbs as usize];
            let (sshc, ssvc) = ss_g();
            let (ccbx, ccby) = (ccbx4 >> sshc, ccby4 >> ssvc);
            let (cuw, cuh) = (((cbd[0] as usize) >> sshc).max(1), ((cbd[1] as usize) >> ssvc).max(1));
            cnb.a_uvmode[ccbx..ccbx + cuw].fill(0); // DC_PRED
            cnb.l_uvmode[ccby..ccby + cuh].fill(0);
        }
    } else {
        // intra-yuv leaf: luma mode (defer coefs) → chroma mode → luma coefs → chroma coefs.
        // If this leaf is the SB's first, decode_b_luma decodes the once-per-SB filters using the
        // threaded left_cdef/left_ccso (top=-1 for 64px SBs) — same state the inter path uses.
        // avm copy_frame_mvs writes NONE_FRAME cells for intra/intrabc blocks — an intra leaf
        // CLEARS its temporal-field footprint (else an earlier inter block's mv leaks into the
        // next frame's TMVP/skip-mode list; dav2d gets this free by saving from the grid).
        crate::av2_refmvs::rp_write(bx4, by4, bw4, bh4, crate::av2_refmvs::TemporalBlock::default());
        let frame_ibc = HDR_TOOL_CFG.with(|c| c.get().allow_intrabc);
        let info = decode_b_luma(
            msac, cdf, a_nb, l_nb, bs, bx4, by4, have_left, have_top, decode_filters, 0, 3, left_cdef,
            left_ccso, -1, frame_ibc, true, true, 2,
        );
        if bx4 == 8 && by4 == 8 { crate::dlog!("Y88 post-luma-mode midx={} fsc={} y_mode_idx={} rng={} dif={:x}", info.midx, info.fsc as u8, info.y_mode_idx, msac.rng, msac.dif); }
        if info.intrabc {
            // intrabc is intra=0 (inter-like): decode_b_luma ALREADY decoded the luma coefs (+
            // splatted lcoef), and there is NO chroma mode/CfL. Decode only the chroma coefs with
            // inter semantics — skip_set=1 (!intra), intra=false, inheriting the luma txtp. When
            // the block is skip_txfm (block-level skip), dav2d reads NO coefs at all (luma OR
            // chroma), so the chroma decode is gated on `!info.skip`.
            let bdc = crate::av2_decode::BLOCK_DIMENSIONS[bs];
            let (sshc, ssvc) = ss_g();
            let (ccbx, ccby) = (bx4 >> sshc, by4 >> ssvc);
            let ibc_cc = if !info.skip {
                Some(decode_chroma_coefs(msac, cdf, cnb, bs, bx4, by4, 1, false, info.txtp))
            } else {
                // A block-level SKIP reads NO chroma coefs, but dav2d's set_ctx still writes the
                // "no coefs" marker (0x40) to the chroma coef context (a/l, both U & V planes) so
                // a later chroma block's skip context isn't computed from a stale coef neighbour.
                let (cbw4, cbh4) = (
                    1usize << (bdc[2] as usize).saturating_sub(sshc),
                    1usize << (bdc[3] as usize).saturating_sub(ssvc),
                );
                for pl in 0..2 {
                    cnb.a[pl][ccbx..ccbx + cbw4].fill(0x40);
                    cnb.l[pl][ccby..ccby + cbh4].fill(0x40);
                }
                None
            };
            // Frame-2 intrabc RECON: block-copy from the assembled FRAME (current frame) at the
            // block vector + residual (dav2d recon_tmpl.c:3203). Integer bv ⇒ pure copy.
            recon_intrabc(bx4, by4, bw4, bh4, slw, slh, info.ibc_bv, info.ibc_morph, &info.cf, info.txtp, info.all_zero, ibc_cc.as_ref(), info.stx, info.eob, false, &info.units);
            crate::av2_frame::dbg_block_miss(bx4, by4, bw4, bh4, "intrabc");
            // intrabc has NO uv mode (chroma is block-copied) → reset the uvmode neighbour ctx to
            // DC_PRED so a later CfL block's cfl_ctx doesn't read a stale CFL_PRED (dav2d set_ctx).
            let (cuw, cuh) = (((bdc[0] as usize) >> sshc).max(1), ((bdc[1] as usize) >> ssvc).max(1));
            cnb.a_uvmode[ccbx..ccbx + cuw].fill(0); // DC_PRED
            cnb.l_uvmode[ccby..ccby + cuh].fill(0);
            // intrabc refmvs GRID splat (dav2d `splat_intrabc_mv`, decode.c:674): stores the block
            // vector in mv[0] with ref=-1 so a LATER intrabc block's refmvs_find(ref=-1) spatial
            // scan can collect it (the spatial-DRL follow-up; inter reads skip ref=-1 cells either way).
            crate::av2_refmvs::GRID.with(|g| g.borrow_mut().splat_intrabc(bx4, by4, bdc[0] as usize, bdc[1] as usize, crate::av2_refmvs::Mv { y: info.ibc_bv.0, x: info.ibc_bv.1 }, bs as u8));
            {
                use crate::av2_refmvs::{Mv, BANK};
                // dav2d splat_intrabc_mv → dav2d_refmvs_bank_add stores the block's actual BV (not 0)
                // so a later intrabc block's refmvs_find bank read can supply it as a candidate.
                BANK.with(|bk| bk.borrow_mut().add_block(bdc[0] as usize, bdc[1] as usize, by4, bx4, sb_step4(), sb_step4() >> 5, -1, Mv { y: info.ibc_bv.0, x: info.ibc_bv.1 }));
            }
            // NOTE the intrabc BANK push (`BANK.add_block(ref0=-1)`: decrements the global bank
            // avail/hits so later inter class-0 pushes gate as dav) is dav-correct and improves the
            // least-cascaded SB0 (30→33), but it REGRESSES the cascaded SB32/48 (total 88→80) — those
            // SBs already diverge upstream (the warp-matrix cascade) so their matches are accidental
            // and flip when the bank state changes. Re-add it once the SB32/48 warp-cascade root is
            // fixed so the total metric is judgeable. See [[rav2d-refmvs-find]].
        } else {
        if std::env::var("MLEAF").is_ok() { crate::dlog!("[MLEAF] mi=({bx4},{by4}) post-luma-mode rng={}", msac.rng); }
        let cm = decode_chroma_mode(msac, cdf, cnb, bs, bx4, by4, info.midx);
        if std::env::var("MLEAF").is_ok() { crate::dlog!("[MLEAF] mi=({bx4},{by4}) post-chroma-mode uv={} cfl={} mh={} rng={}", cm.uv_mode, cm.cfl_mode, cm.mh_dir, msac.rng); }
        if bx4 == 0 && by4 == 6 { crate::dlog!("SBL06 post-chroma-mode fsc={} rng={} dif={:x} (oracle pre_coef rng=36272)", info.fsc as u8, msac.rng, msac.dif); }
        // --- palette (defer path): the JOINT intra leaf reads palette AFTER the chroma mode
        // (avm read_intra_frame_mode_info order y→uv→palette), then the color-index map
        // before the tx partition (av2_visit_palette). Same gates as the SDP hook.
        let mut palette: Option<crate::av2_palette::PaletteBlock> = None;
        {
            let tool_cfg2 = HDR_TOOL_CFG.with(|c| c.get());
            if tool_cfg2.allow_scc && info.y_mode_idx == 0 && !info.intrabc
                && bw4 * bh4 >= 4 && bw4 <= 16 && bh4 <= 16
                && crate::msac::rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.pal_y)
            {
                let n = crate::msac::rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.pal_sz, 6) as usize + 2;
                let bd_bits = if bdmax_g() > 255 { 10u8 } else { 8 };
                let colors = crate::av2_palette::read_palette_colors_y(msac, bd_bits, n, bx4, by4, have_left, have_top);
                if std::env::var("MPAL").is_ok() { crate::dlog!("[MPAL] mi=({bx4},{by4}) n={n} colors={:?} rng={} (defer)", &colors[..n], msac.rng); }
                palette = Some(crate::av2_palette::PaletteBlock { n, colors, map: Vec::new(), w: bw4 * 4, h: bh4 * 4 });
            }
            let bw4cl = bw4.min(col_end.saturating_sub(bx4));
            let bh4cl = bh4.min(row_end.saturating_sub(by4));
            let (pn, pc) = palette.as_ref().map_or((0, [0u16; 8]), |p| (p.n, p.colors));
            crate::av2_palette::pal_splat(bx4, by4, bw4cl, bh4cl, pn, &pc);
            if let Some(p) = palette.as_mut() {
                let rows = bh4cl * 4;
                let cols = bw4cl * 4;
                p.map = crate::av2_palette::decode_color_map(
                    msac, &mut cdf.m.pal_idx_identity, &mut cdf.m.pal_idx, p.n, p.w, p.h, rows, cols,
                ).expect("palette color map (defer)");
                if std::env::var("MPAL").is_ok() { crate::dlog!("[MPAL] mi=({bx4},{by4}) map done rng={} (defer)", msac.rng); }
            }
        }
        // ===== TX PARTITION (dav decode.c:2328: read AFTER the full intra mode info incl. the
        // chroma mode) + per-unit luma coefs. Parse order: tx_part → luma units → chroma coefs;
        // recon runs after the parse, per unit in decode order (chained prediction).
        let tx_part = read_tx_part(msac, cdf, bw4, bh4, info.fsc, false, false);
        let tx_layout = tx_part_layout(bw4, bh4, tx_part);
        let skip_set = info.fsc as usize;
        let (fw4, fh4) = crate::av2_frame::FRAME.with(|f| { let f = f.borrow(); (f.iw4, f.ih4) });
        // (ux, uy, uslw, uslh, cf, txtp, stxt, stxs, az)
        let mut iunits: Vec<(usize, usize, usize, usize, Vec<i32>, u8, u8, u8, bool)> = Vec::new();
        let mut cc_opt2 = None;
        for &(ux, uy, utw4, uth4) in &tx_layout {
            let (ubx4, uby4) = (bx4 + ux, by4 + uy);
            if ubx4 >= fw4 || uby4 >= fh4 { continue; }
            let (uslw, uslh) = (utw4.trailing_zeros() as usize, uth4.trailing_zeros() as usize);
            let (uclw, uclh) = (uslw.min(3), uslh.min(3));
            let u_tdc = (uslw + uslh + 1) >> 1;
            let u2d = uclw + uclh;
            let ubw4 = if fw4 > ubx4 { utw4.min(fw4 - ubx4) } else { utw4 };
            let ubh4 = if fh4 > uby4 { uth4.min(fh4 - uby4) } else { uth4 };
            let sctx = if info.fsc {
                9
            } else {
                crate::av2_coef::skip_ctx_luma(&a_nb.lcoef[ubx4..], &l_nb.lcoef[uby4..], uslw, uslh, &bd) as usize
            };
            if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({ubx4},{uby4}) pl=0y txs={u_tdc} skipctx={sctx} rng={}", msac.rng); }
            let az = rav1d_msac_decode_bool_adapt(msac, &mut cdf.coef.skip[skip_set][u_tdc][sctx]);
            if bx4 == 0 && by4 == 6 { crate::dlog!("SBL06 luma all_zero={} skip_set={skip_set} sctx={sctx} tctx={u_tdc} rng={}", az as u8, msac.rng); }
            let mut cf = vec![0i32; 1usize << (uslw + uslh + 4)];
            let (cf_ctx, u_txtp, u_stxt, u_stxs) = if az {
                (0x40u8, DCT_DCT, 0u8, 0u8)
            } else {
                let e = crate::av2_coef::decode_eob(msac, &mut cdf.coef, u2d, 0);
                let scan = crate::av2_tables_gen::SCANS[scan_idx_square(uclw, uclh)];
                // dav2d applies wide_angle_remap to y_mode BEFORE md_idx2type (recon_tmpl.c:2562)
                // with the UNIT tx dims — a wide-angle rectangular unit's luma txtp follows the
                // remapped mode.
                let y_mode_raw = y_mode_from_idx(info.y_mode_idx, info.midx);
                let y_mode = if info.y_mode_idx >= 5 {
                    wide_angle_remap_mode(y_mode_raw, info.midx as i32 % 7 - 3, info.mrl_index, 4 << uslw, 4 << uslh)
                } else {
                    y_mode_raw
                };
                let dc_sign_ctx = crate::av2_coef::get_dc_sign_ctx(&a_nb.lcoef[ubx4..], &l_nb.lcoef[uby4..], uslw, uslh, ubw4 as i32, ubh4 as i32);
                if std::env::var("CYAT").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == bx4 && p[1] == by4 }) {
                    crate::dlog!("[CYAT] mi=({bx4},{by4}) e={e} dcsctx={dc_sign_ctx} uslw={uslw} uslh={uslh} u2d={u2d} rng={} dif={:x} a={:x?} l={:x?}", msac.rng, msac.dif, &a_nb.lcoef[ubx4..ubx4+16.min(a_nb.lcoef.len()-ubx4)], &l_nb.lcoef[uby4..uby4+8.min(l_nb.lcoef.len()-uby4)]);
                }
                let (r, txtp, stxt, stxs) = decode_coefs_y(msac, cdf, &mut cf, e, info.fsc, y_mode, uslw.min(uslh), u_tdc, uslw, uslh, u2d, scan, false, dc_sign_ctx);
                if std::env::var("CYAT").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == bx4 && p[1] == by4 }) {
                    crate::dlog!("[CYAT] mi=({bx4},{by4}) post-y txtp={txtp} rng={} dif={:x}", msac.rng, msac.dif);
                }
                (r, txtp, stxt, stxs)
            };
            // frame-edge clamp: see the inter-path splat above (dav recon_tmpl.c:1226).
            for x in ubx4..(ubx4 + ubw4).min(fw4).min(a_nb.lcoef.len()) { a_nb.lcoef[x] = cf_ctx; }
            for y in uby4..(uby4 + ubh4).min(fh4).min(l_nb.lcoef.len()) { l_nb.lcoef[y] = cf_ctx; }
            iunits.push((ux, uy, uslw, uslh, cf, u_txtp, u_stxt, u_stxs, az));
            // >64px chunk order: the chroma block rides with the FIRST 64px luma chunk.
            if bw4.max(bh4) > 16 && ux == 0 && uy == 0 {
                cc_opt2 = Some(decode_chroma_coefs(msac, cdf, cnb, bs, bx4, by4, info.fsc as usize, true, DCT_DCT));
            }
        }
        // chroma U all_zero skip set = (!intra || fsc) → for an intra block, `fsc` (dav2d
        // recon_tmpl.c:387). An fsc/IDTX intra block's chroma uses skip[1], not skip[0].
        let cc = match cc_opt2 {
            Some(c) => c,
            None => decode_chroma_coefs(msac, cdf, cnb, bs, bx4, by4, info.fsc as usize, true, DCT_DCT),
        };
        // Frame-2 intra LUMA RECON: per-unit chained prediction from the assembled FRAME.
        if std::env::var("BRDBG").is_ok() && bx4 >= 94 && by4 >= 48 {
            crate::dlog!("LEAF-RECON ({bx4},{by4}) bw4={bw4} bh4={bh4} midx={} ymode={}", info.midx, info.y_mode_idx);
        }
        for (ux, uy, uslw, uslh, cf, u_txtp, u_stxt, u_stxs, u_az) in &iunits {
            let (ubx4, uby4) = (bx4 + ux, by4 + uy);
            let ubw4 = if fw4 > ubx4 { (1usize << uslw).min(fw4 - ubx4) } else { 1usize << uslw };
            let ubh4 = if fh4 > uby4 { (1usize << uslh).min(fh4 - uby4) } else { 1usize << uslh };
            recon_intra_luma(
                ubx4, uby4, *uslw, *uslh, ubw4, ubh4, info.y_mode_idx, info.midx, info.mrl_index,
                info.multi_mrl != 0, cf, *u_txtp, *u_stxt, *u_stxs, *u_az, info.fsc,
                have_left || *ux > 0, have_top || *uy > 0,
                tx_part >= 6 && (*ux > 0 || *uy > 0),
                palette.as_ref().map(|p| (p, *ux * 4, *uy * 4)),
            );
        }
        crate::av2_frame::mark_btype(bx4, by4, bw4, bh4, 2);
        crate::av2_frame::dbg_block_miss(bx4, by4, bw4, bh4, "intra");
        // Frame-2 intra CHROMA RECON (predict from FRAME chroma neighbours + CfL/MHCCP + residual).
        if has_chroma {
            let (sshc, ssvc) = ss_g();
            let (ccbx, ccby) = (bx4 >> sshc, by4 >> ssvc);
            recon_intra_chroma(
                ccbx, ccby, bx4, by4, bw4, bh4, cc.slw, cc.slh, cm.uv_mode, cm.uv_angle, cm.cfl_mode,
                cm.cfl_alpha_u, cm.cfl_alpha_v, cm.mh_dir, &cc.cf_u, &cc.cf_v, cc.u_eob == -1, cc.v_eob == -1,
                ccbx > TILE_B.with(|t| t.get().0 >> sshc), ccby > TILE_B.with(|t| t.get().2 >> ssvc),
            );
            // Mark the CHROMA deblock edge grids for an INTRA leaf inside an INTER frame — the
            // key path marks via its own chroma tree (decode_sb_chroma), the inter leaf via
            // recon_inter_chroma; this path marked NEITHER, so its chroma block edges were
            // never deblocked (visible at high QP where the thresholds are large).
            crate::av2_frame::FRAME.with(|f| {
                f.borrow_mut().mark_db_chroma(ccbx, ccby, (bw4 >> sshc).max(1), (bh4 >> ssvc).max(1));
            });
        }
        // dav2d `splat_intraref` (decode.c:1378): a luma intra block refreshes the bank avail/hits
        // counters (no MV added). Skipping this drained `avail` and blocked later inter bank pushes.
        crate::av2_refmvs::BANK.with(|bk| bk.borrow_mut().bank_update_intra(bw4, bh4, by4, bx4, sb_step4(), sb_step4() >> 5));
        }
        splat_inter_nb(a_nb, l_nb, bx4, by4, bw4, bh4, 1, 0, 0, 0, 0, 0, -1);
        splat_nb(&mut a_nb.comp_type, &mut l_nb.comp_type, bx4, by4, bw4, bh4, 0);
        // Splat the intra-yuv block into the refmvs grid (ref=-1) for the warp neighbour walk.
        // NOT for intrabc — it already splatted its BV via splat_intrabc above; overwriting with an
        // INVALID_MV intra cell here erases the BV that a later block's bank re-seed / spatial scan
        // reads (dav keeps the intrabc grid cell). This was corrupting the intrabc DRL cascade.
        if !info.intrabc {
            crate::av2_refmvs::GRID.with(|g| g.borrow_mut().splat_intra(bx4, by4, bw4, bh4, bs as u8));
        }
    }
    if std::env::var("SBTRACE").is_ok() { crate::dlog!("SBLEAF ({bx4},{by4}) bs={bs} intra={} rng={} dif={:x}", intra as u8, msac.rng, msac.dif); }
}

/// Recursively decode a **mixed-region** superblock partition sub-tree (dav2d `decode_sb`,
/// inter frame, no SDP decoupling). Walks the partition tree (NONE/H/V for now), decoding the
/// ext-SDP `region_type` at qualifying splits, and dispatches each leaf to `decode_leaf`.
/// `sb_by4` is the superblock's top row (for `have_top_in_sb`/`is_sb_boundary`).
#[allow(clippy::too_many_arguments)]
pub fn decode_sb_inter(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    a_part: &mut [u8],
    l_part: &mut [u8],
    a_nb: &mut BlockNbCtx,
    l_nb: &mut BlockNbCtx,
    cnb: &mut ChromaNb,
    bs: usize,
    bx4: usize,
    by4: usize,
    iw4: usize,
    ih4: usize,
    // SDP ext-region state: true when inside an intra region (leaves forced intra, luma-only).
    intra_region: bool,
    // Luma directional-mode map (16x16 SB grid), populated by luma leaves for the chroma tree.
    luma_dir_map: &mut [u8; 256],
    // Chroma block size (bs index, -1 = BS_INVALID/luma-only) + its origin in luma 4px coords.
    // Below 8x8 the chroma is shared: at an 8x8 split `sub4` is false → first child cbs=-1
    // (luma-only), last child carries the shared chroma (dav2d decode.c:3989/4010).
    cbs: i32,
    cbx4: usize,
    cby4: usize,
    // Cross-SB filter neighbour state (left SB's cdef index + ccso flags), threaded through the
    // recursion and consumed once per SB at the first leaf. Reset per SB-row in the harness.
    left_cdef: &mut i8,
    left_ccso: &mut [u8; 3],
) {
    use crate::av2_decode::{decode_partition, splat_partition, BlockPartition, BLOCK_DIMENSIONS, PART_HALF};
    use crate::msac::rav1d_msac_decode_bool_adapt;
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    let (hw4, hh4) = (bw4 / 2, bh4 / 2);
    let have_h = iw4 > bx4 + hw4;
    let have_v = ih4 > by4 + hh4;
    // dav2d 3947: when the luma block matches the chroma block size, the chroma origin is here.
    let (cbx4, cby4) = if bs as i32 == cbs { (bx4, by4) } else { (cbx4, cby4) };
    let sbdbg = std::env::var("SBI").is_ok();
    let r_in = msac.rng;
    let iss = ss_g();
    let (bp, _half) = decode_partition(msac, &mut cdf.m, bs, a_part, l_part, bx4, by4, have_h, have_v, 3, true, 0, iss.0 as u32, iss.1 as u32, iw4, ih4, !intra_region);
    if sbdbg { crate::dlog!("[SBI] ({bx4},{by4}) bs={bs} bp={bp:?} r_in={r_in} r={} dif={:x} intra_region={intra_region} have_h={have_h} have_v={have_v}", msac.rng, msac.dif); }
    // ext-SDP recursion limit set by the PARENT (dav2d `dir_ptr & (1<<24)`): read at entry so
    // the region_type decode below can honour it, then re-establish it for children before the
    // match. A returning sibling leaves the cell at the shared parent value (this block's entry
    // value), so it is restored at function exit.
    let ext_sdp_limited = EXT_SDP_LIMIT.with(|c| c.get());

    // ext-SDP region_type (mixed vs intra region). When intra: this block becomes the region
    // root — its subtree is luma-only (children forced intra), then ONE chroma block is decoded
    // here (dav2d decode.c:3937 + 4228).
    use BlockPartition::*;
    let (bw4i, bh4i) = (bw4 as i32, bh4 as i32);
    let mut child_intra = intra_region;
    let mut is_region_root = false;
    // dav gate is `bs != f->root_bs`: with 128px SBs a 64x64 node is NOT the root and DOES
    // code region_type.
    let root_bs_now = if sb_step4() == 32 { 3 } else { 6 };
    if !intra_region && !ext_sdp_limited && bs != root_bs_now && matches!(bp, H | V | H3 | V3) && bw4i.max(bh4i) <= 16 && bw4i.min(bh4i) >= 2 {
        let sz = (bd[2] + bd[3]) as i32;
        let ctx = (sz - 4).clamp(0, 3) as usize + (sz == 4) as usize;
        let rt = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.region_type[ctx]);
        if sbdbg { crate::dlog!("[SBI] ({bx4},{by4}) region_type ctx={ctx} mixed={} r={} dif={:x}", rt as u8, msac.rng, msac.dif); }
        if !rt {
            child_intra = true;
            is_region_root = true;
        }
    }

    let hbs = PART_HALF[bs][match bp { V | V3 | V4a | V4b => 1, _ => 0 }] as usize;
    // Child chroma block size. `sub4` = the chroma can subdivide (half-sample > 0) at a block
    // whose luma matches the chroma size. When false, the first child is luma-only (-1) and the
    // last child keeps the parent chroma. The intra region forces all children luma-only.
    // avm have_nz_chroma_ref_offset (blockd.h:1222), per-axis ss: children carry their own
    // chroma iff every child chroma dim stays >= 4px (1 unit).
    let (sshc, ssvc) = ss_g();
    let sub4_v = bs as i32 == cbs && (hw4 >> sshc) > 0 && (bh4 >> ssvc) > 0;
    let sub4_h = bs as i32 == cbs && (bw4 >> sshc) > 0 && (hh4 >> ssvc) > 0;
    let (cbs0_h, cbs1_h) = if child_intra { (-1, -1) } else if sub4_h { (hbs as i32, hbs as i32) } else { (-1, cbs) };
    let (cbs0_v, cbs1_v) = if child_intra { (-1, -1) } else if sub4_v { (hbs as i32, hbs as i32) } else { (-1, cbs) };
    // Establish the ext-SDP limit for the children (dav2d `child_dir`, computed FRESH from this
    // block's dims + partition — NOT inherited from our own ext_sdp_limited). Each recursive
    // child reads this at its entry; a returning child restores it to this same value for its
    // siblings (its own entry value == our child_limited).
    EXT_SDP_LIMIT.with(|c| c.set(ext_sdp_child_limited(bp, bw4, bh4)));
    match bp {
        None => {
            // Availability is TILE-relative (dav ts->tiling col/row_start), not frame-relative.
            let tbl = TILE_B.with(|t| t.get());
            let have_left = bx4 > tbl.0;
            let have_top = by4 > tbl.2;
            let sbm = sb_step4() - 1;
            let have_top_in_sb = (by4 & sbm) != 0;
            let is_sb_boundary = (by4 & sbm) == 0;
            let decode_filters = (bx4 & sbm) == 0 && (by4 & sbm) == 0;
            decode_leaf(msac, cdf, a_nb, l_nb, cnb, bs, bx4, by4, have_left, have_top, have_top_in_sb, is_sb_boundary, ih4, iw4, decode_filters, left_cdef, left_ccso, child_intra, luma_dir_map, cbs, cbx4, cby4);
            splat_partition(a_part, l_part, bs, bx4, by4);
        }
        H => {
            decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, hbs, bx4, by4, iw4, ih4, child_intra, luma_dir_map, cbs0_h, cbx4, cby4, left_cdef, left_ccso);
            if by4 + hh4 < ih4 {
                decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, hbs, bx4, by4 + hh4, iw4, ih4, child_intra, luma_dir_map, cbs1_h, cbx4, cby4, left_cdef, left_ccso);
            }
        }
        V => {
            decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, hbs, bx4, by4, iw4, ih4, child_intra, luma_dir_map, cbs0_v, cbx4, cby4, left_cdef, left_ccso);
            if bx4 + hw4 < iw4 {
                decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, hbs, bx4 + hw4, by4, iw4, ih4, child_intra, luma_dir_map, cbs1_v, cbx4, cby4, left_cdef, left_ccso);
            }
        }
        Split => {
            // 4 quadrants at the split size; SPLIT requires bs==cbs (dav2d 4044), so children yuv.
            let sbs = bs_from_dims(hw4, hh4);
            let cc = if child_intra { -1 } else { sbs as i32 };
            for &(px, py) in &[(bx4, by4), (bx4 + hw4, by4), (bx4, by4 + hh4), (bx4 + hw4, by4 + hh4)] {
                if px < iw4 && py < ih4 {
                    decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, sbs, px, py, iw4, ih4, child_intra, luma_dir_map, cc, cbx4, cby4, left_cdef, left_ccso);
                }
            }
        }
        H3 => {
            // top 1/4 (quarter) + middle 1/2 split into two half-width (split) + bottom 1/4.
            let qh4 = bh4 / 4;
            let quarter = bs_from_dims(bw4, qh4);
            let split = bs_from_dims(hw4, hh4);
            let half = bs_from_dims(bw4, hh4);
            let sub4 = bs as i32 == cbs && (qh4 >> ssvc) > 0 && (hw4 >> sshc) > 0;
            let i3only = cbs == -1 || (!sub4 && bs != bs_from_dims(2, 8));
            let ci = child_intra;
            let c_top = if ci { -1 } else if i3only { -1 } else { quarter as i32 };
            let c_ml = if ci { -1 } else if sub4 { split as i32 } else { -1 };
            let c_mr = if ci { -1 } else if i3only { -1 } else if sub4 { split as i32 } else { half as i32 };
            let c_bot = if ci { -1 } else if i3only { cbs } else { quarter as i32 };
            decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, quarter, bx4, by4, iw4, ih4, ci, luma_dir_map, c_top, cbx4, cby4, left_cdef, left_ccso);
            if by4 + qh4 < ih4 {
                decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, split, bx4, by4 + qh4, iw4, ih4, ci, luma_dir_map, c_ml, bx4, by4 + qh4, left_cdef, left_ccso);
                if bx4 + hw4 < iw4 {
                    decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, split, bx4 + hw4, by4 + qh4, iw4, ih4, ci, luma_dir_map, c_mr, bx4, by4 + qh4, left_cdef, left_ccso);
                }
                if by4 + qh4 + hh4 < ih4 {
                    decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, quarter, bx4, by4 + qh4 + hh4, iw4, ih4, ci, luma_dir_map, c_bot, cbx4, cby4, left_cdef, left_ccso);
                }
            }
        }
        V3 => {
            let qw4 = bw4 / 4;
            let quarter = bs_from_dims(qw4, bh4);
            let split = bs_from_dims(hw4, hh4);
            let half = bs_from_dims(hw4, bh4);
            let sub4 = bs as i32 == cbs && (qw4 >> sshc) > 0 && (hh4 >> ssvc) > 0;
            let i3only = cbs == -1 || (!sub4 && bs != bs_from_dims(8, 2));
            let ci = child_intra;
            let c_left = if ci { -1 } else if i3only { -1 } else { quarter as i32 };
            let c_mt = if ci { -1 } else if sub4 { split as i32 } else { -1 };
            let c_mb = if ci { -1 } else if i3only { -1 } else if sub4 { split as i32 } else { half as i32 };
            let c_right = if ci { -1 } else if i3only { cbs } else { quarter as i32 };
            decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, quarter, bx4, by4, iw4, ih4, ci, luma_dir_map, c_left, cbx4, cby4, left_cdef, left_ccso);
            if bx4 + qw4 < iw4 {
                decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, split, bx4 + qw4, by4, iw4, ih4, ci, luma_dir_map, c_mt, bx4 + qw4, by4, left_cdef, left_ccso);
                if by4 + hh4 < ih4 {
                    decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, split, bx4 + qw4, by4 + hh4, iw4, ih4, ci, luma_dir_map, c_mb, bx4 + qw4, by4, left_cdef, left_ccso);
                }
                if bx4 + qw4 + hw4 < iw4 {
                    decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, quarter, bx4 + qw4 + hw4, by4, iw4, ih4, ci, luma_dir_map, c_right, cbx4, cby4, left_cdef, left_ccso);
                }
            }
        }
        V4a | V4b => {
            // 4-way vertical (dav2d decode.c:4172): widths eighth | quarter<<var | half>>var |
            // eighth (var=1 for V4b mirrors the middle two). Only the LAST child carries the
            // parent chroma unless the eighth still subdivides in chroma (sub4).
            let var = matches!(bp, V4b) as usize;
            let qw4 = bw4 / 4;
            let ew4 = qw4 >> 1; // eighth width
            let (w4a, w4b) = (qw4 << var, hw4 >> var);
            let (b1, b4) = (bs_from_dims(ew4, bh4), bs_from_dims(ew4, bh4));
            let (b2, b3) = (bs_from_dims(w4a, bh4), bs_from_dims(w4b, bh4));
            let sub4 = bs as i32 == cbs && (ew4 >> sshc) > 0 && (bh4 >> ssvc) > 0;
            let ci = child_intra;
            let cc = |sz: usize| if ci { -1 } else if sub4 { sz as i32 } else { -1 };
            let c4 = if ci { -1 } else if sub4 { b4 as i32 } else { cbs };
            let (x1, x2, x3, x4) = (bx4, bx4 + ew4, bx4 + ew4 + w4a, bx4 + ew4 + w4a + w4b);
            decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b1, x1, by4, iw4, ih4, ci, luma_dir_map, cc(b1), cbx4, cby4, left_cdef, left_ccso);
            if x2 < iw4 { decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b2, x2, by4, iw4, ih4, ci, luma_dir_map, cc(b2), cbx4, cby4, left_cdef, left_ccso); }
            if x3 < iw4 { decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b3, x3, by4, iw4, ih4, ci, luma_dir_map, cc(b3), cbx4, cby4, left_cdef, left_ccso); }
            if x4 < iw4 { decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b4, x4, by4, iw4, ih4, ci, luma_dir_map, c4, cbx4, cby4, left_cdef, left_ccso); }
        }
        H4a | H4b => {
            // 4-way horizontal (dav2d decode.c:4213): heights eighth | quarter<<var |
            // half>>var | eighth. Transpose of V4a/V4b.
            let var = matches!(bp, H4b) as usize;
            let qh4 = bh4 / 4;
            let eh4 = qh4 >> 1; // eighth height
            let (h4a, h4b) = (qh4 << var, hh4 >> var);
            let (b1, b4) = (bs_from_dims(bw4, eh4), bs_from_dims(bw4, eh4));
            let (b2, b3) = (bs_from_dims(bw4, h4a), bs_from_dims(bw4, h4b));
            let sub4 = bs as i32 == cbs && (bw4 >> sshc) > 0 && (eh4 >> ssvc) > 0;
            let ci = child_intra;
            let cc = |sz: usize| if ci { -1 } else if sub4 { sz as i32 } else { -1 };
            let c4 = if ci { -1 } else if sub4 { b4 as i32 } else { cbs };
            let (y1, y2, y3, y4) = (by4, by4 + eh4, by4 + eh4 + h4a, by4 + eh4 + h4a + h4b);
            decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b1, bx4, y1, iw4, ih4, ci, luma_dir_map, cc(b1), cbx4, cby4, left_cdef, left_ccso);
            if y2 < ih4 { decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b2, bx4, y2, iw4, ih4, ci, luma_dir_map, cc(b2), cbx4, cby4, left_cdef, left_ccso); }
            if y3 < ih4 { decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b3, bx4, y3, iw4, ih4, ci, luma_dir_map, cc(b3), cbx4, cby4, left_cdef, left_ccso); }
            if y4 < ih4 { decode_sb_inter(msac, cdf, a_part, l_part, a_nb, l_nb, cnb, b4, bx4, y4, iw4, ih4, ci, luma_dir_map, c4, cbx4, cby4, left_cdef, left_ccso); }
        }
        BlockPartition::Invalid => {
            crate::dlog!("decode_sb_inter: Invalid partition at ({bx4},{by4}) bs={bs}");
        }
    }
    // Restore the ext-SDP limit to THIS block's entry value so our parent's next sibling reads
    // the shared parent-level value (the recursion's save/restore invariant).
    EXT_SDP_LIMIT.with(|c| c.set(ext_sdp_limited));

    // Region root: after the luma subtree, decode ONE chroma block for the region (dav2d 4231).
    if is_region_root {
        let (uv, _ang) = decode_b_chroma(msac, cdf, cnb, luma_dir_map, bs, bx4, by4);
        if std::env::var("SBTRACE").is_ok() { crate::dlog!("SBRGN-CHROMA ({bx4},{by4}) bs={bs} uv_mode={uv} rng={} dif={:x}", msac.rng, msac.dif); }
    }
}

/// Recursively decode the SDP **chroma** partition tree of a superblock (dav2d
/// `decode_sb` with `lbs == BS_INVALID`). This is the *second* tree decoded on the same
/// MSAC stream, after the luma tree completes. It operates on luma-sized `bs` over the
/// luma coordinate grid, but the plane is 4:2:0-subsampled (`ss=1`), so `decode_partition`
/// forces NONE once the chroma sample block hits 4x4. The 64x64 root partition is
/// *inferred* (F164), not coded.
///
/// LAYER 1 (this commit): verify the partition descent only — `decode_b_chroma` is a stub
/// that records the first leaf's entry rng and aborts further recursion via `done`, so the
/// emitted partition rngs can be diffed against the oracle's chroma `read_partition` lines.
#[allow(clippy::too_many_arguments)]
/// SDP: map a *shared* luma partition onto the chroma tree's partition — avm
/// `sdp_chroma_part_from_luma` (common_data.h:331), keyed on the CHROMA dimensions.
///
/// The extended types are the whole point: a luma `HORZ_3` shares as chroma `HORZ_3`
/// (three sub-blocks) whenever the chroma height still reaches 16. Collapsing anything
/// that is not H/V to `None` decodes ONE chroma leaf where the bitstream holds three,
/// which desyncs the entropy decoder for the rest of the frame.
///
/// `luma_part` is the raw partition code (this crate's `BlockPartition` discriminants,
/// which match avm's `PARTITION_TYPE` values one-for-one); `bw`/`bh` are the LUMA
/// dimensions in pixels, shifted here by the subsampling.
fn sdp_chroma_part_from_luma(
    luma_part: u32,
    bw: usize,
    bh: usize,
    ss_hor: usize,
    ss_ver: usize,
) -> crate::av2_decode::BlockPartition {
    use crate::av2_decode::BlockPartition as P;
    let bh_chr = bh >> ss_ver;
    let bw_chr = bw >> ss_hor;
    match luma_part {
        0 => P::None,
        1 => if bh_chr < 8 { P::None } else { P::H },
        2 => if bw_chr < 8 { P::None } else { P::V },
        3 => if bh_chr >= 16 { P::H3 } else if bh_chr < 8 { P::None } else { P::H },
        4 => if bw_chr >= 16 { P::V3 } else if bw_chr < 8 { P::None } else { P::V },
        5 => if bh_chr >= 32 { P::H4a } else if bh_chr >= 8 { P::H } else { P::None },
        6 => if bh_chr >= 32 { P::H4b } else if bh_chr >= 8 { P::H } else { P::None },
        7 => if bw_chr >= 32 { P::V4a } else if bw_chr >= 8 { P::V } else { P::None },
        8 => if bw_chr >= 32 { P::V4b } else if bw_chr >= 8 { P::V } else { P::None },
        9 => if bh_chr < 8 || bw_chr < 8 { P::None } else { P::Split },
        _ => P::None,
    }
}

pub fn decode_sb_chroma(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    a_part: &mut [u8],
    l_part: &mut [u8],
    bs: usize,
    bx4: usize,
    by4: usize,
    luma_dirptr: u32,
    luma_dir_map: &[u8; 256],
    nb: &mut ChromaNb,
    iw4: usize,
    ih4: usize,
) {
    use crate::av2_decode::{decode_partition, splat_partition, BlockPartition, BLOCK_DIMENSIONS};
    let css = ss_g();
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    // Frame-boundary split availability (luma coords, same as luma tree).
    let have_h_split = iw4 > bx4 + bw4 / 2;
    let have_v_split = ih4 > by4 + bh4 / 2;
    // Chroma-tree half of the [PARTIN] probe. avm's PARTPROBE emits BOTH trees into one stream
    // (luma nodes, then the SDP chroma tree restarting at the SB origin), so this has to fire too
    // or the two traces cannot be aligned past the first SB.
    if std::env::var("PARTIN").is_ok() {
        crate::dlog!("[PARTINC] mi=({bx4},{by4}) bs={bs} rng={} dif={:x}", msac.rng, msac.dif);
    }
    // SDP chroma root (64x64): F164 inference from the luma `dir_ptr`. If luma did not
    // split (byte0 == 0xff) the chroma is NONE; if luma split one way and *all* its direct
    // children split the other (the 0x10002 / 0x20001 patterns over bits 0-1 | 16-17), the
    // chroma copies the luma partition (bits 8-15); otherwise it is coded.
    let bp = if bs == 6 {
        let byte0 = luma_dirptr & 0xff;
        let mask3 = luma_dirptr & 0x30003;
        if byte0 == 0xff {
            BlockPartition::None
        } else if mask3 == 0x10002 || mask3 == 0x20001 {
            sdp_chroma_part_from_luma((luma_dirptr >> 8) & 0xff, bw4 * 4, bh4 * 4, css.0, css.1)
        } else {
            decode_partition(msac, &mut cdf.m, bs, a_part, l_part, bx4, by4, have_h_split, have_v_split, 3, true, 1, css.0 as u32, css.1 as u32, iw4, ih4, false).0
        }
    } else {
        let (p, _half) =
            decode_partition(msac, &mut cdf.m, bs, a_part, l_part, bx4, by4, have_h_split, have_v_split, 3, true, 1, css.0 as u32, css.1 as u32, iw4, ih4, false);
        p
    };
    if std::env::var("DBLK444").is_ok() && bx4 * 4 >= 176 && bx4 * 4 <= 216 && by4 * 4 >= 104 && by4 * 4 <= 128 {
        crate::dlog!("[MSBC] ({},{}) bs={bs} bw={} bh={} bp={bp:?}", bx4 * 4, by4 * 4, bw4 * 4, bh4 * 4);
    }
    // F157: at the chroma 64x64 root, CfL is disallowed for the SB when the chroma split
    // direction is non-NONE and differs from luma's root dir (dav2d decode.c:3839). dirs:
    // 1=H, 2=V; luma's lives in `luma_dirptr` bits 0-1.
    if bs == 6 {
        let chroma_dir: i32 = match bp {
            // dav's `dir` stays -1 for NONE **and SPLIT** (no direction) — Split must not
            // count as a direction mismatch.
            BlockPartition::None | BlockPartition::Split => -1,
            BlockPartition::H | BlockPartition::H3 | BlockPartition::H4a | BlockPartition::H4b => 1,
            _ => 2, // V / V3 / V4a / V4b
        };
        let luma_dir = (luma_dirptr & 0x3) as i32;
        nb.sdp_cfl_disallowed = chroma_dir != -1 && chroma_dir != luma_dir;
        if std::env::var("F157DBG").is_ok() {
            crate::dlog!("[F157] ({bx4},{by4}) bp={bp:?} cdir={chroma_dir} ldir={luma_dir} dirptr={luma_dirptr:x} disallow={}", nb.sdp_cfl_disallowed);
        }
    }
    match bp {
        BlockPartition::None => {
            // Decode the chroma leaf (intra_uv_mode + U/V coefficients), threading the live
            // chroma neighbour context so each leaf's skip context is computed.
            let _ = decode_b_chroma(msac, cdf, nb, luma_dir_map, bs, bx4, by4);
            splat_partition(a_part, l_part, bs, bx4, by4);
        }
        BlockPartition::V => {
            let hw4 = bw4 / 2;
            let h = crate::av2_decode::PART_HALF[bs][1] as usize;
            decode_sb_chroma(msac, cdf, a_part, l_part, h, bx4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            if bx4 + hw4 < iw4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, h, bx4 + hw4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
        }
        BlockPartition::H => {
            let hh4 = bh4 / 2;
            let h = crate::av2_decode::PART_HALF[bs][0] as usize;
            decode_sb_chroma(msac, cdf, a_part, l_part, h, bx4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            if by4 + hh4 < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, h, bx4, by4 + hh4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
        }
        BlockPartition::H3 => {
            let (qh4, hw4, hh4) = (bh4 / 4, bw4 / 2, bh4 / 2);
            let (strip, mid) = h3_sub_sizes(bs);
            decode_sb_chroma(msac, cdf, a_part, l_part, strip, bx4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            if by4 + qh4 < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, mid, bx4, by4 + qh4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if bx4 + hw4 < iw4 && by4 + qh4 < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, mid, bx4 + hw4, by4 + qh4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if by4 + qh4 + hh4 < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, strip, bx4, by4 + qh4 + hh4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
        }
        BlockPartition::V3 => {
            let (qw4, hw4, hh4) = (bw4 / 4, bw4 / 2, bh4 / 2);
            let (strip, mid) = v3_sub_sizes(bs);
            decode_sb_chroma(msac, cdf, a_part, l_part, strip, bx4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            if bx4 + qw4 < iw4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, mid, bx4 + qw4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if bx4 + qw4 < iw4 && by4 + hh4 < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, mid, bx4 + qw4, by4 + hh4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if bx4 + qw4 + hw4 < iw4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, strip, bx4 + qw4 + hw4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
        }
        BlockPartition::V4a | BlockPartition::V4b => {
            let ew4 = bw4 / 8;
            let var = matches!(bp, BlockPartition::V4b) as usize;
            let (w8, h8) = (bw4 as u8, bh4 as u8);
            let eighth = bs_for_dims(w8 / 8, h8);
            let (c2, c3) = if var == 0 {
                (bs_for_dims(w8 / 4, h8), bs_for_dims(w8 / 2, h8))
            } else {
                (bs_for_dims(w8 / 2, h8), bs_for_dims(w8 / 4, h8))
            };
            let (w4a, w4b) = ((bw4 / 4) << var, (bw4 / 2) >> var);
            decode_sb_chroma(msac, cdf, a_part, l_part, eighth, bx4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            if bx4 + ew4 < iw4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, c2, bx4 + ew4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if bx4 + ew4 + w4a < iw4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, c3, bx4 + ew4 + w4a, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if bx4 + ew4 + w4a + w4b < iw4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, eighth, bx4 + ew4 + w4a + w4b, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
        }
        BlockPartition::H4a | BlockPartition::H4b => {
            let eh4 = bh4 / 8;
            let var = matches!(bp, BlockPartition::H4b) as usize;
            let (w8, h8) = (bw4 as u8, bh4 as u8);
            let eighth = bs_for_dims(w8, h8 / 8);
            let (c2, c3) = if var == 0 {
                (bs_for_dims(w8, h8 / 4), bs_for_dims(w8, h8 / 2))
            } else {
                (bs_for_dims(w8, h8 / 2), bs_for_dims(w8, h8 / 4))
            };
            let (h4a, h4b) = ((bh4 / 4) << var, (bh4 / 2) >> var);
            decode_sb_chroma(msac, cdf, a_part, l_part, eighth, bx4, by4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            if by4 + eh4 < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, c2, bx4, by4 + eh4, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if by4 + eh4 + h4a < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, c3, bx4, by4 + eh4 + h4a, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
            if by4 + eh4 + h4a + h4b < ih4 {
                decode_sb_chroma(msac, cdf, a_part, l_part, eighth, bx4, by4 + eh4 + h4a + h4b, luma_dirptr, luma_dir_map, nb, iw4, ih4);
            }
        }
        other => {
            crate::dlog!("[rav2d UV] UNHANDLED partition {other:?} at bs={bs} ({bx4},{by4})");
        }
    }
}

/// The `BlockSize` index for a `[w4, h4]` dimension pair (reverse `BLOCK_DIMENSIONS`).
fn bs_for_dims(w4: u8, h4: u8) -> usize {
    crate::av2_decode::BLOCK_DIMENSIONS
        .iter()
        .position(|d| d[0] == w4 && d[1] == h4)
        .expect("valid block dimensions")
}

/// H3 (horizontal three-band) sub-block sizes `(strip, mid)`: the strip is full-width ×
/// ¼-height, the (split) middle band is ½-width × ½-height.
fn h3_sub_sizes(bs: usize) -> (usize, usize) {
    let bd = crate::av2_decode::BLOCK_DIMENSIONS[bs];
    (bs_for_dims(bd[0], bd[1] / 4), bs_for_dims(bd[0] / 2, bd[1] / 2))
}

/// V3 (vertical three-band) sub-block sizes `(strip, mid)`: the strip is ¼-width ×
/// full-height, the (split) middle band is ½-width × ½-height.
fn v3_sub_sizes(bs: usize) -> (usize, usize) {
    let bd = crate::av2_decode::BLOCK_DIMENSIONS[bs];
    (bs_for_dims(bd[0] / 4, bd[1]), bs_for_dims(bd[0] / 2, bd[1] / 2))
}

// Transform-type enum values (dav2d levels.h: `HOR_1D | (tx_class<<3) | (VER_1D<<5)`;
// 1D types DCT=0, IDENTITY=1, ADST=2, FLIPADST=3). `tx_class = (txtp>>3)&3` → 2D=0,H=2,V=3.
const DCT_DCT: u8 = 0;
const DCT_ADST: u8 = 2;
const DCT_FLIPADST: u8 = 3;
const ADST_DCT: u8 = 64;
const ADST_ADST: u8 = 66;
const ADST_FLIPADST: u8 = 67;
const FLIPADST_DCT: u8 = 96;
const FLIPADST_ADST: u8 = 98;
const FLIPADST_FLIPADST: u8 = 99;
const V_DCT: u8 = 25;
const V_ADST: u8 = 89;
const V_FLIPADST: u8 = 121;
const H_DCT: u8 = 48;
const H_ADST: u8 = 50;
const H_FLIPADST: u8 = 51;
const IDTX_TT: u8 = 33;
const PAETH_PRED: u8 = 12;

/// Directional prediction angle (avm `wide_angle_mapping`): base `mode_to_angle_map[mode]` +
/// scaled angle-delta + `mrl_index_to_delta`, then the wide-angle remap (±180) for rectangular
/// blocks whose aspect ratio pushes the angle past the WAIP thresholds.
fn av2_p_angle(y_mode: u8, angle_delta: i32, mrl_index: u8, w: usize, h: usize) -> i32 {
    const MODE_TO_ANGLE: [i32; 9] = [0, 90, 180, 45, 135, 113, 157, 203, 67];
    const MRL_DELTA: [i32; 4] = [0, 1, -1, 0];
    let mut p = MODE_TO_ANGLE[y_mode as usize] + angle_delta + MRL_DELTA[(mrl_index & 3) as usize];
    let (tw, th) = (w as i32, h as i32);
    // WAIP thresholds (blockd.h): ratio 2→61, 4→73, 8→82, 16→86.
    if (th == 2 * tw && p < 61)
        || (th == 4 * tw && p < 73)
        || (th == 8 * tw && p < 82)
        || (th == 16 * tw && p < 86)
    {
        p += 180;
    } else if (tw == 2 * th && p > 270 - 61)
        || (tw == 4 * th && p > 270 - 73)
        || (tw == 8 * th && p > 270 - 82)
        || (tw == 16 * th && p > 270 - 86)
    {
        p -= 180;
    }
    p
}

/// Wide-angle MODE remap (dav2d `wide_angle_remap`, recon_tmpl.c:1200): a rectangular block whose
/// (delta-adjusted) angle crosses the WAIP threshold has its directional MODE reassigned — tall →
/// `HOR_UP` (7), wide → `DIAG_DOWN_LEFT` (3). dav2d applies this to `b->uv_mode` BEFORE the intra
/// chroma txtp LUT, so the txtp of a wide-angle block follows the REMAPPED mode (e.g. a tall D45
/// block → HOR_UP → DCT_ADST, not DCT_DCT). Mirrors `av2_p_angle`'s branch conditions exactly.
fn wide_angle_remap_mode(mode: u8, angle_delta: i32, mrl_index: u8, w: usize, h: usize) -> u8 {
    if mode < 1 || mode > 8 {
        return mode; // only the 8 directional modes remap
    }
    const MODE_TO_ANGLE: [i32; 9] = [0, 90, 180, 45, 135, 113, 157, 203, 67];
    const MRL_DELTA: [i32; 4] = [0, 1, -1, 0];
    // `angle_delta` is the RAW delta (dav2d `wide_angle_remap`: base + delta*3 + mrl_adj).
    let p = MODE_TO_ANGLE[mode as usize] + 3 * angle_delta + MRL_DELTA[(mrl_index & 3) as usize];
    let (tw, th) = (w as i32, h as i32);
    if (th == 2 * tw && p < 61) || (th == 4 * tw && p < 73) || (th == 8 * tw && p < 82) || (th == 16 * tw && p < 86) {
        7 // HOR_UP
    } else if (tw == 2 * th && p > 209) || (tw == 4 * th && p > 197) || (tw == 8 * th && p > 188) || (tw == 16 * th && p > 184) {
        3 // DIAG_DOWN_LEFT
    } else {
        mode
    }
}

/// avm `get_y_intra_mode_set` — the neighbour-reordered directional joint-mode list. Returns the
/// joint mode (5..60) selected by directional `y_mode_idx` (>=5). `bl`/`ar` are the bottom-left /
/// above-right neighbours' joint modes (avm `av2_get_joint_mode`; 0 = DC when unavailable/inter).
/// Small blocks (< 8x8) use the fixed default list; ≥8x8 insert directional neighbour modes first
/// (and, for >64-sample blocks, their ±1..4 angle-delta derivatives) before the default fill.
fn reorder_dir_joint(y_mode_idx: i32, bl: u8, ar: u8, w: usize, h: usize) -> u8 {
    let is_small = (w == 4 && h == 4) || (w == 4 && h == 8) || (w == 8 && h == 4);
    let is_large = w * h > 64;
    let mut nj = [bl as i32, ar as i32]; // [0]=left(bottom-left), [1]=above(above-right)
    let is_left_dir = nj[0] >= 5;
    let is_above_dir = nj[1] >= 5;
    let mut selected = [false; 61];
    for s in selected.iter_mut().take(5) {
        *s = true; // non-directional joint 0..4 always present
    }
    let mut list: Vec<u8> = Vec::with_capacity(56); // directional part (positions 5+)
    if !is_small {
        let mut cnt = is_above_dir as i32 + is_left_dir as i32;
        if cnt == 2 && nj[0] == nj[1] {
            cnt = 1;
        }
        if cnt == 1 && !is_left_dir {
            nj[0] = nj[1];
        }
        for &j in nj.iter().take(cnt as usize) {
            list.push(j as u8);
            selected[j as usize] = true;
        }
        if is_large {
            for i in 0..4 {
                for &jm in nj.iter().take(cnt as usize) {
                    let ld = ((jm - i + 50) % 56 + 5) as usize;
                    let rd = ((jm + i - 4) % 56 + 5) as usize;
                    if !selected[ld] {
                        list.push(ld as u8);
                        selected[ld] = true;
                    }
                    if !selected[rd] {
                        list.push(rd as u8);
                        selected[rd] = true;
                    }
                }
            }
        }
    }
    for &d in DEFAULT_MODE_LIST_Y.iter() {
        if list.len() >= 56 {
            break;
        }
        let jm = d as usize + 5;
        if !selected[jm] {
            list.push(jm as u8);
            selected[jm] = true;
        }
    }
    list[(y_mode_idx - 5) as usize]
}

/// avm `has_top_right` for the luma single-TX case (row_off=col_off=0, no >64 blocks, no TX
/// partition). Returns `(available, px_top_right)`. `coded(sb_row, sb_col)` reads the SB-relative
/// decode-order grid.
fn has_top_right(
    bx4: usize, by4: usize, bw4: usize, w: i32, xr: i32, have_top: bool, right_avail: bool,
    coded: impl Fn(i32, i32) -> bool,
) -> (bool, i32) {
    if !have_top || !right_avail {
        return (false, 0);
    }
    let px_tr_common = w.min(xr);
    if px_tr_common <= 0 {
        return (false, 0);
    }
    let mut px = px_tr_common;
    // row_off=0: `col_off + tx_wide_unit < plane_bw_unit` → `0 + bw4 < bw4` = false → general case.
    let st = crate::av2_recon::sb_step4() as i32;
    let sbm = st - 1;
    let tr_mask_row = (by4 as i32 & sbm) - 1;
    let tr_mask_col = (bx4 as i32 & sbm) + bw4 as i32;
    if tr_mask_row < 0 {
        return (true, px); // top-right is in the SB row above (already decoded)
    }
    if tr_mask_col >= st {
        return (false, px); // in the next SB to the right (not yet decoded)
    }
    let has_tr = coded(tr_mask_row, tr_mask_col);
    if has_tr {
        let mut mi_tr = 0;
        for i in 0..bw4 as i32 {
            if tr_mask_col + i >= st || !coded(tr_mask_row, tr_mask_col + i) {
                break;
            }
            mi_tr += 1;
        }
        px = (mi_tr * 4).min(px_tr_common);
    }
    (has_tr, px)
}

/// avm `has_bottom_left` for the luma single-TX case. Returns `(available, px_bottom_left)`.
fn has_bottom_left(
    bx4: usize, by4: usize, bh4: usize, h: i32, yd: i32, bottom_avail: bool, have_left: bool,
    coded: impl Fn(i32, i32) -> bool,
) -> (bool, i32) {
    if !bottom_avail || !have_left {
        return (false, 0);
    }
    let px_bl_common = h.min(yd);
    if px_bl_common <= 0 {
        return (false, 0);
    }
    let mut px = px_bl_common;
    // col_off=0: `row_off + tx_high_unit < plane_bh_unit` → `0 + bh4 < bh4` = false → general.
    let st = crate::av2_recon::sb_step4() as i32;
    let sbm = st - 1;
    let bl_mask_row = (by4 as i32 & sbm) + bh4 as i32;
    let bl_mask_col = (bx4 as i32 & sbm) - 1;
    if bl_mask_col < 0 {
        // block at SB left edge: bottom-left is in the (already-decoded) left neighbour SB,
        // available down to the SB bottom (SB size in px, not 64).
        let plane_bottom_row = ((by4 as i32 & sbm) << 2) + h;
        px = (st * 4 - plane_bottom_row).min(px_bl_common);
        return (px > 0, px);
    }
    if bl_mask_row >= st {
        return (false, px);
    }
    let has_bl = coded(bl_mask_row, bl_mask_col);
    if has_bl {
        let mut mi_bl = 0;
        for i in 0..bh4 as i32 {
            if bl_mask_row + i >= st || !coded(bl_mask_row + i, bl_mask_col) {
                break;
            }
            mi_bl += 1;
        }
        px = (mi_bl * 4).min(px_bl_common);
    }
    (has_bl, px)
}

/// Faithful AV2 IDIF directional prediction (avm `av2_build_intra_predictors_high_default`, mrl=0
/// directional path + `highbd_dr_predictor_idif`). `top`/`left` are gathered with the correct
/// top-right / bottom-left extension already applied. NOTE: the directional-IBP second-predictor
/// blend (z1/z3, angle_delta%2==0 & is_ibp_enabled) is NOT yet applied — Phase 2.
#[allow(clippy::too_many_arguments)]
fn recon_dir_idif(
    pred: &mut [i32], w: usize, h: usize, top: &[i32], left: &[i32], corner: i32, p: i32,
    angle_delta: i32, mrl: i32, apply_ibp: bool, sm_separate: bool, above_sm: bool, left_sm: bool,
    ef: bool, chroma: bool, bdmax: i32,
) {
    use crate::av2_ipred::*;
    const AO: usize = 16;
    // needs (avm lines 1458-1471; apply_ibp forces all three).
    let (need_above, need_left, need_above_left) = if apply_ibp {
        (true, true, true)
    } else if p < 90 {
        (true, false, true)
    } else if p == 90 {
        (true, false, false)
    } else if p < 180 {
        (true, true, true)
    } else if p == 180 {
        (false, true, false)
    } else {
        (false, true, true)
    };
    let need_right = if apply_ibp { p < 90 || p > 180 } else { p < 90 };
    let need_bottom = if apply_ibp { p < 90 || p > 180 } else { p > 180 };

    // Offset edge buffers: buf[AO-1]=corner, buf[AO+i]=sample i (already extended by gather).
    let n = w + h;
    let mut abuf = vec![0i32; AO + n + 16];
    let mut lbuf = vec![0i32; AO + n + 16];
    abuf[AO - 1] = corner;
    lbuf[AO - 1] = corner;
    for i in 0..n {
        abuf[AO + i] = top[i];
        lbuf[AO + i] = left[i];
    }
    for i in AO + n..abuf.len() {
        abuf[i] = top[n - 1];
        lbuf[i] = left[n - 1];
    }

    // avm edge filtering (skipped for p==90/180; only if seq edge filter enabled).
    if ef && p != 90 && p != 180 {
        let ab_le = usize::from(need_above_left);
        // avm computes the edge-filter smooth flags with `apply_ibp = seq_ibp && tx!=4x4`
        // (recon_tmpl.c:3874) — the value BEFORE the mode-specific restriction (`&= DC_PRED`
        // for chroma / directional-IBP-enabled for luma). So the SEPARATE per-edge smooth
        // flags (ft_above=sm_top, ft_left=sm_left) apply whenever ibp is enabled and tx!=4x4,
        // even for chroma directional (which never actually blends IBP). Using the combined
        // (sm_top||sm_left) flag there wrongly filters an edge whose own neighbour isn't smooth.
        let (ft_above, ft_left) = if sm_separate {
            (above_sm as i32, left_sm as i32)
        } else {
            let t = (above_sm || left_sm) as i32;
            (t, t)
        };
        let angle_above = if apply_ibp && p > 180 { p - 180 - 90 } else { p - 90 };
        let angle_left = if apply_ibp && p < 90 { p } else { p - 180 };
        if need_above && need_left && (w + h >= 24) {
            filter_intra_edge_corner(&mut abuf, AO, &mut lbuf, AO);
        }
        if need_above {
            let s = intra_edge_filter_strength(w as i32, h as i32, angle_above, ft_above);
            let n_px = w + ab_le + if need_right { h } else { 0 };
            av2_filter_intra_edge(&mut abuf, AO - ab_le, n_px, s);
        }
        if need_left {
            let s = intra_edge_filter_strength(h as i32, w as i32, angle_left, ft_left);
            let n_px = h + ab_le + if need_bottom { w } else { 0 };
            av2_filter_intra_edge(&mut lbuf, AO - ab_le, n_px, s);
        }
    }

    if crate::av2_ipred::DIR_DBG.with(|c| c.get()) {
        crate::dlog!("[MDIRD] p={p} ef={ef} chroma={chroma} apply_ibp={apply_ibp} sm_sep={sm_separate} lbuf_postfilt={:?}", &lbuf[AO - 1..(AO + (w + h)).min(lbuf.len())]);
    }
    project_idif(pred, w, h, &mut abuf, &mut lbuf, AO, p, mrl, chroma, bdmax);
    // Directional IBP blend (z1/z3, tx != 4x4, even angle-delta, mrl=0, LUMA ONLY —
    // avm reconintra.c gates the blend on plane == PLANE_TYPE_Y; chroma still uses the
    // unrestricted apply_ibp for the NEED flags / corner filter / edge-filter angles).
    if apply_ibp && mrl == 0 && !chroma {
        apply_dir_ibp(pred, w, h, p, angle_delta, &mut abuf, AO, &mut lbuf, AO, bdmax);
    }
}

/// IDIF projection dispatch (avm `highbd_dr_predictor_idif`) into `pred` from the offset edge
/// buffers `abuf`/`lbuf` (offset `ao`), including the pre-projection edge replications.
#[allow(clippy::too_many_arguments)]
fn project_idif(
    pred: &mut [i32], w: usize, h: usize, abuf: &mut [i32], lbuf: &mut [i32], ao: usize, p: i32,
    mrl: i32, chroma: bool, bdmax: i32,
) {
    use crate::av2_ipred::*;
    let (dx, dy) = (av2_get_dx(p), av2_get_dy(p));
    let n = (w + h) as i32;
    let idx = |i: i32| (ao as i32 + i) as usize;
    if p > 0 && p < 90 {
        let mbx = n - 1 + (mrl << 1);
        abuf[idx(mbx + 1)] = abuf[idx(mbx)];
        abuf[idx(mbx + 2)] = abuf[idx(mbx)];
        dr_z1_idif(pred, w, w, h, abuf, ao, dx, mrl, chroma, bdmax);
    } else if p > 90 && p < 180 {
        let minb = -1 - mrl;
        abuf[idx(minb - 1)] = abuf[idx(minb)];
        lbuf[idx(minb - 1)] = lbuf[idx(minb)];
        if mrl == 0 {
            abuf[ao + w] = abuf[ao + w - 1];
            lbuf[ao + h] = lbuf[ao + h - 1];
        }
        dr_z2_idif(pred, w, w, h, abuf, ao, lbuf, ao, dx, dy, mrl, chroma, bdmax);
    } else if p > 180 && p < 270 {
        let mby = n - 1 + (mrl << 1);
        lbuf[idx(mby + 1)] = lbuf[idx(mby)];
        lbuf[idx(mby + 2)] = lbuf[idx(mby)];
        dr_z3_idif(pred, w, w, h, lbuf, ao, dy, mrl, chroma, bdmax);
    } else if p == 90 {
        for r in 0..h {
            for c in 0..w {
                pred[r * w + c] = abuf[ao + c];
            }
        }
    } else if p == 180 {
        for r in 0..h {
            for c in 0..w {
                pred[r * w + c] = lbuf[ao + r];
            }
        }
    }
}

/// Build the offset edge buffers for one MRL reference-line config (avm
/// `av2_build_intra_predictors_high`, no edge filter for mrl>0). `above_y`/`left_x` are the
/// reference row/col; `corner_cnt` = mrl+1 diagonal corner samples. Reads clamp to the plane.
#[allow(clippy::too_many_arguments)]
fn build_mrl_buf(
    src: &crate::av2_frame::Plane, px0: usize, py0: usize, w: usize, h: usize, above_y: i32,
    left_x: i32, corner_cnt: usize, sb_bnd: bool, have_top: bool, have_left: bool, n_tr: usize,
    n_bl: usize, base: i32,
) -> (Vec<i32>, Vec<i32>) {
    const AO: usize = 16;
    let n = w + h;
    let mut abuf = vec![base; AO + n + 16];
    let mut lbuf = vec![base; AO + n + 16];
    let (pw, ph) = (src.w as i32, src.h as i32);
    let spx =
        |x: i32, y: i32| src.at(x.clamp(0, pw - 1) as usize, y.clamp(0, ph - 1) as usize);
    let (px, py) = (px0 as i32, py0 as i32);
    // MRL projection reads `2*mrl` further into the top-right / bottom-left (max_base grows by
    // `mrl<<1`); mrl = corner_cnt-1. Cap the real fill accordingly (was `h`/`w` = the mrl=0 amount).
    let mrl2 = 2 * (corner_cnt - 1);
    if have_top {
        for j in 0..w {
            abuf[AO + j] = spx(px + j as i32, above_y);
        }
        let tr = n_tr.min(h + mrl2);
        for j in 0..tr {
            abuf[AO + w + j] = spx(px + (w + j) as i32, above_y);
        }
        let last = abuf[AO + w + tr - 1];
        for a in abuf.iter_mut().skip(AO + w + tr) {
            *a = last;
        }
    } else if have_left {
        let v = spx(left_x, py);
        for a in abuf.iter_mut().skip(AO) {
            *a = v;
        }
    }
    if have_left {
        for j in 0..h {
            lbuf[AO + j] = spx(left_x, py + j as i32);
        }
        let bl = n_bl.min(w + mrl2);
        for j in 0..bl {
            lbuf[AO + h + j] = spx(left_x, py + (h + j) as i32);
        }
        let last = lbuf[AO + h + bl - 1];
        for l in lbuf.iter_mut().skip(AO + h + bl) {
            *l = last;
        }
    } else if have_top {
        let v = spx(px, above_y);
        for l in lbuf.iter_mut().skip(AO) {
            *l = v;
        }
    }
    for i in 1..=corner_cnt {
        let (a, l) = if have_top && have_left {
            let lc = if sb_bnd { spx(left_x, py - 1) } else { spx(left_x, py - i as i32) };
            (spx(px - i as i32, above_y), lc)
        } else if have_top {
            let v = spx(px, above_y);
            (v, v)
        } else if have_left {
            let v = spx(left_x, py);
            (v, v)
        } else {
            (base, base)
        };
        abuf[AO - i] = a;
        lbuf[AO - i] = l;
    }
    (abuf, lbuf)
}

/// MRL (mrl_index>0) directional prediction (avm `av2_build_intra_predictors_high` path): predict
/// from the mrl-offset reference line, optionally 2-line-averaged with line 0. No edge filter, no
/// directional IBP.
#[allow(clippy::too_many_arguments)]
fn recon_dir_mrl(
    pred: &mut [i32], w: usize, h: usize, src: &crate::av2_frame::Plane, px0: usize, py0: usize,
    p: i32, mrl: i32, sb_bnd: bool, multi_mrl: bool, have_top: bool, have_left: bool, n_tr: usize,
    n_bl: usize, base: i32, bdmax: i32,
) {
    const AO: usize = 16;
    let above_mrl_idx = if sb_bnd { 0 } else { mrl };
    let corner_cnt = (mrl + 1) as usize;
    let (mut a1, mut l1) = build_mrl_buf(
        src, px0, py0, w, h, py0 as i32 - 1 - above_mrl_idx, px0 as i32 - 1 - mrl, corner_cnt,
        sb_bnd, have_top, have_left, n_tr, n_bl, base,
    );
    let mrldbg = std::env::var("MRLDBG").is_ok_and(|v| {
        let mut it = v.split(',');
        it.next() == Some(&px0.to_string()) && it.next() == Some(&py0.to_string())
    });
    if mrldbg {
        crate::dlog!("MRLDBG l1={:02x?}", &l1[AO - corner_cnt..AO + w + h]);
        crate::dlog!("MRLDBG a1={:02x?}", &a1[AO - corner_cnt..AO + w + h]);
    }
    project_idif(pred, w, h, &mut a1, &mut l1, AO, p, mrl, false, bdmax);
    if mrldbg {
        for y in 0..h {
            crate::dlog!("MRLDBG p1 {:02x?}", &pred[y * w..y * w + w]);
        }
    }
    // multi_line_mrl: average with the line-0 prediction (tx != 4x4).
    if multi_mrl && !(w == 4 && h == 4) {
        let (mut a0, mut l0) = build_mrl_buf(
            src, px0, py0, w, h, py0 as i32 - 1, px0 as i32 - 1, corner_cnt, sb_bnd, have_top,
            have_left, n_tr, n_bl, base,
        );
        let mut pred0 = vec![0i32; w * h];
        if mrldbg {
            crate::dlog!("MRLDBG l0={:02x?}", &l0[AO - 1..AO + w + h]);
        }
        project_idif(&mut pred0, w, h, &mut a0, &mut l0, AO, p, 0, false, bdmax);
        if mrldbg {
            for y in 0..h {
                crate::dlog!("MRLDBG p0 {:02x?}", &pred0[y * w..y * w + w]);
            }
        }
        for (d, &s) in pred.iter_mut().zip(pred0.iter()) {
            *d = (*d + s + 1) >> 1;
        }
    }
}

/// AV2 neighbour-reordered intra mode: resolves `y_mode_idx` through the joint-mode list
/// (reordered for ≥8x8 directional blocks by the bottom-left / above-right neighbour joint modes,
/// read from the frame `joint` grid). Returns `(y_mode, midx, joint)`.
#[allow(clippy::too_many_arguments)]
pub fn reordered_mode(
    f: &crate::av2_frame::Frame, y_mode_idx: i32, bx4: usize, by4: usize, bw4: usize, bh4: usize,
    w: usize, h: usize, have_left: bool, have_top: bool,
) -> (u8, u8, u8) {
    if y_mode_idx < 5 {
        (REORDERED_NONDIR_Y_MODE[y_mode_idx as usize], 0u8, y_mode_idx as u8)
    } else {
        let bl = if have_left { f.joint_at(bx4 as i32 - 1, (by4 + bh4 - 1) as i32) } else { 0 };
        let ar = if have_top { f.joint_at((bx4 + bw4 - 1) as i32, by4 as i32 - 1) } else { 0 };
        let joint = reorder_dir_joint(y_mode_idx, bl, ar, w, h);
        let midx = joint - 5;
        (REORDERED_DIR_Y_MODE[(midx / 7) as usize], midx, joint)
    }
}

/// Reconstruct one intra luma block (single TX = block, frame-1 keyframe path) into the recon
/// FRAME buffer. ADDITIVE to the entropy parse — consumes the parsed `cf` (raster signed levels)
/// + intra mode + txtp. No-op when the FRAME buffer is unallocated (i.e. not the frame-1 pass).
#[allow(clippy::too_many_arguments)]
thread_local! {
    /// MSCREF=path: lazy-loaded luma plane for the decode-order intra block-miss probe.
    static MSCREF_PLANE: std::cell::RefCell<Option<crate::av2_frame::Plane>> = const { std::cell::RefCell::new(None) };
}
fn mscref_check(bx4: usize, by4: usize, w: usize, h: usize, tag: &str) {
    let Ok(path) = std::env::var("MSCREF") else { return };
    MSCREF_PLANE.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            if let Ok(b) = std::fs::read(&path) {
                let mut p = crate::av2_frame::Plane::alloc(432, 240);
                for i in 0..(432 * 240).min(b.len()) { p.px[i] = b[i] as i32; }
                *c = Some(p);
            }
        }
        let Some(rp) = c.as_ref() else { return };
        crate::av2_frame::FRAME.with(|fr| {
            let f = fr.borrow();
            if f.pl[0].w == 0 { return; }
            for yy in 0..h {
                for xx in 0..w {
                    let (px, py) = (bx4 * 4 + xx, by4 * 4 + yy);
                    if px < rp.w && py < rp.h && px < f.pl[0].w && py < f.pl[0].h {
                        let m = f.pl[0].px[py * f.pl[0].stride + px];
                        if m != rp.at(px, py) {
                            crate::dlog!("MSCREF[{tag}] ({bx4},{by4}) w={w} h={h} at({xx},{yy}) mine={m} dav={}", rp.at(px, py));
                            return;
                        }
                    }
                }
            }
        });
    });
}

fn recon_intra_luma(
    bx4: usize,
    by4: usize,
    slw: usize,
    slh: usize,
    bw4: usize,
    bh4: usize,
    y_mode_idx: i32,
    parse_midx: u8,
    mrl_index: u8,
    multi_mrl: bool,
    cf: &[i32],
    txtp: u8,
    _stx_type: u8,
    _stx_set: u8,
    all_zero: bool,
    fsc: bool,
    have_left: bool,
    have_top: bool,
    // dav recon_tmpl.c:2602 `is_hv5`: a unit of an H5/V5-partitioned block that is NOT at
    // the block origin forces n_tr = n_bl = 0 (no top-right / bottom-left extension).
    hv5_off: bool,
    // Palette block (avm reconintra.c:1699): the color-index map IS the predictor —
    // (block, unit x offset px, unit y offset px). The intra pred is skipped entirely.
    palette: Option<(&crate::av2_palette::PaletteBlock, usize, usize)>,
) {
    // 1D-type map: txtp encodes DCT=0/IDENTITY=1/ADST=2/FLIPADST=3; av2_itx wants DCT=0/ADST=1/
    // FLIPADST=2/IDENTITY=3.
    const T1D: [usize; 4] = [0, 3, 1, 2];
    // Only the scored SB-loop pass writes the frame buffer; the later scaffold "full recursion"
    // re-invokes this and must no-op (else it double-fires probes / overwrites the dump).
    if !crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
        return;
    }
    if std::env::var("BRDBG").is_ok() && bx4 >= 94 && by4 >= 48 {
        crate::dlog!("BRINTRA ({bx4},{by4}) bw4={bw4} bh4={bh4} hl={have_left} ht={have_top} all_zero={all_zero} ymode={y_mode_idx} midx={parse_midx}");
    }
    if std::env::var("RILTRACE").is_ok() {
        crate::dlog!("[RIL] ({bx4},{by4}) {bw4}x{bh4} ymode={y_mode_idx} midx={parse_midx} mrl={mrl_index} az={all_zero}");
    }
    let (w, h) = (4usize << slw, 4usize << slh);
    let (px0, py0) = (bx4 * 4, by4 * 4);
    crate::av2_frame::FRAME.with(|fr| {
        let mut f = fr.borrow_mut();
        if f.pl[0].w == 0 {
            return; // FRAME not allocated (non-frame-1)
        }
        f.ensure_sb(bx4, by4); // reset the SB decode-order grid on SB crossing
        // Mark the decode-order availability grid for EVERY decoded leaf — including blocks
        // that extend past the frame's right/bottom edge. dav2d marks `is_coded` for the
        // full block extent (only pixel writes are clipped); skipping it understates an
        // interior neighbour's top-right / bottom-left availability. Must precede the
        // off-frame bounds check below.
        f.mark_coded_avail(bx4, by4, bw4, bh4);
        if !all_zero {
            f.mark_lr_noskip(bx4, by4, bw4, bh4); // LR noskip: coded luma TX unit
        }
        if px0 >= f.pl[0].w || py0 >= f.pl[0].h {
            return; // ENTIRELY off-frame: grid marked, but skip pixel recon.
        }
        // A block whose top-left is in-frame but that spills past the right/bottom edge (a partial
        // SB, e.g. a 64-wide block at x=384 in a 432-wide frame) still reconstructs its VISIBLE
        // pixels — dav writes the full block into a padded buffer, so a later in-frame neighbour
        // reads real recon (not 0). The pred is computed full-size; the write loop clips to w/h.
        let (yac, bdmax, stride, ef, ibp_on) =
            (f.yac, f.bitdepth_max, f.pl[0].stride, f.edge_filter, f.ibp);
        let apply_ibp = ibp_on && !(w == 4 && h == 4); // avm: is_ibp_allowed_blk_sz = tx != TX_4X4
        let base = (bdmax + 1) >> 1; // 1 << (bitdepth - 1)
        let (iw4, ih4) = (f.iw4, f.ih4);
        // AV2 neighbour-reordered intra mode: y_mode_idx indexes the joint-mode list. For ≥8x8
        // directional blocks the list is reordered by the bottom-left / above-right neighbour joint
        // modes, so the SAME decoded index maps to a DIFFERENT mode+angle-delta than the fixed list.
        let is_dir = y_mode_idx >= 5;
        // Use the PARSE-computed midx (from a.midx/l.midx, dav-matching) instead of re-deriving from
        // the recon joint grid (which diverges on partition gaps → wrong luma directional mode).
        let (y_mode, recon_midx, joint) = if is_dir {
            (REORDERED_DIR_Y_MODE[(parse_midx / 7) as usize], parse_midx, parse_midx + 5)
        } else {
            (REORDERED_NONDIR_Y_MODE[y_mode_idx as usize], 0u8, y_mode_idx as u8)
        };
        // Directional angle + avm top-right / bottom-left edge availability (has_top_right /
        // has_bottom_left over the is_mi_coded grid) — drives the edge extension for z1/z3.
        let p_angle = if is_dir {
            av2_p_angle(y_mode, 3 * ((recon_midx % 7) as i32 - 3), mrl_index, w, h)
        } else {
            0
        };
        // SMOOTH/SMOOTH_V/SMOOTH_H consume the +1 EXTENSION anchors (top[w], left[h]) — dav
        // computes n_tr/n_bl for EVERY intra block (recon_tmpl.c:2570-2616) and the anchors
        // come from the REAL top-right/bottom-left pixels when the availability scan says so
        // (allintra 4x4 SMOOTH at (14,10): anchors ef/f3, not the replicated ed/f0).
        let is_sm = !is_dir && (y_mode == 9 || y_mode == 10 || y_mode == 11);
        let (n_tr, n_bl) = if hv5_off {
            (0, 0)
        } else if is_dir || is_sm {
            let need_tr = if is_sm { true } else if apply_ibp { p_angle < 90 || p_angle > 180 } else { p_angle < 90 };
            let need_bl = if is_sm { true } else if apply_ibp { p_angle < 90 || p_angle > 180 } else { p_angle > 180 };
            // TILE-clamped right/bottom availability (dav ts->tiling.col_end/row_end): a
            // tile-edge block must not extend from across the boundary (those pixels are
            // not even decoded yet in tile-sequential order).
            let tbl = TILE_B.with(|t| t.get());
            let (ce4, re4) = (tbl.1.min(iw4), tbl.3.min(ih4));
            let xr = (ce4 as i32 - (bx4 + bw4) as i32) * 4;
            let yd = (re4 as i32 - (by4 + bh4) as i32) * 4;
            let right_avail = bx4 + bw4 < ce4;
            let bottom_avail = yd > 0 && by4 + bh4 < re4;
            let n_tr = if need_tr {
                let (av, px) =
                    has_top_right(bx4, by4, bw4, w as i32, xr, have_top, right_avail, |r, c| {
                        f.mi_coded_at(r, c)
                    });
                if av { px } else { 0 }
            } else {
                0
            };
            let n_bl = if need_bl {
                let (av, px) =
                    has_bottom_left(bx4, by4, bh4, h as i32, yd, bottom_avail, have_left, |r, c| {
                        f.mi_coded_at(r, c)
                    });
                if av { px } else { 0 }
            } else {
                0
            };
            if std::env::var("MINTRA").is_ok() {
                crate::dlog!("[MINTRA] fn={} y={by4} x={bx4} bw4={bw4} bh4={bh4} ntr={} nbl={} mode={y_mode} pang={p_angle} mrl={mrl_index} midx={recon_midx}",
                    crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()), n_tr / 4, n_bl / 4);
            }
            (n_tr as usize, n_bl as usize)
        } else {
            (0, 0)
        };
        let above_sm = f.smooth_at(bx4 as i32, by4 as i32 - 1);
        let left_sm = f.smooth_at(bx4 as i32 - 1, by4 as i32);
        if std::env::var("MDCP").map_or(false, |v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); p[0] == bx4 && p[1] == by4 }) {
            let tr: Vec<i32> = (0..8).map(|i| f.pl[0].at(bx4 * 4 + i, by4 * 4 - 1)).collect();
            let lc: Vec<i32> = (0..8).map(|i| f.pl[0].at(bx4 * 4 - 1, by4 * 4 + i)).collect();
            crate::dlog!("[MDCP] fn={} ({bx4},{by4}) w={w} h={h} ymode={y_mode} asm={} lsm={} ibp={} jointa={} jointl={} ntr={n_tr} nbl={n_bl} havet={have_top} havel={have_left} top={tr:?} left={lc:?}",
                crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()), above_sm as u8, left_sm as u8, apply_ibp as u8,
                f.joint_at(bx4 as i32, by4 as i32 - 1), f.joint_at(bx4 as i32 - 1, by4 as i32));
        }
        // Self-ref intra gather reads the assembled recon f.pl, CLAMPED at the frame edge by
        // gather_edges' bounds — dav edge-replicates off-frame neighbours, it does NOT use the
        // real off-frame recon that RECON_PAD holds. Byte-exact for BOTH the keyframe and inter
        // frames. (REF_LUMA crutch retained purely as an isolation-harness override when loaded.)
        let selfref = std::env::var("SELFREF").is_ok();
        let (top, left, corner) = crate::av2_frame::REF_LUMA.with(|r| {
            let rb = r.borrow();
            if let Some(rp) = rb.as_ref().filter(|_| !selfref) {
                crate::av2_frame::gather_edges(rp, px0, py0, w, h, have_top, have_left, n_tr, n_bl, base)
            } else {
                crate::av2_frame::gather_edges(&f.pl[0], px0, py0, w, h, have_top, have_left, n_tr, n_bl, base)
            }
        });
        let mut pred = vec![0i32; w * h];
        match y_mode {
            0 => {
                // DC: avm dispatches by neighbour availability, then applies the IBP gradient.
                use crate::av2_ipred::*;
                match (have_top, have_left) {
                    (true, true) => ipred_dc(&mut pred, w, &top, &left, w, h, bdmax),
                    (false, true) => ipred_dc_left(&mut pred, w, &left, w, h),
                    (true, false) => ipred_dc_top(&mut pred, w, &top, w, h),
                    (false, false) => ipred_dc_128(&mut pred, w, w, h, bdmax),
                }
                if apply_ibp && (have_top || have_left) {
                    ipred_ibp_dc(&mut pred, w, &top, &left, w, h, have_top, have_left);
                }
            }
            9 => crate::av2_ipred::ipred_smooth(&mut pred, w, &top, &left, w, h),
            10 => crate::av2_ipred::ipred_smooth_v(&mut pred, w, &top, &left, w, h),
            11 => crate::av2_ipred::ipred_smooth_h(&mut pred, w, &top, &left, w, h),
            PAETH_PRED => {
                // avm prepare_intra_edges: PAETH degrades by availability.
                match (have_top, have_left) {
                    (true, true) => {
                        crate::av2_ipred::ipred_paeth(&mut pred, w, &top, &left, corner, w, h)
                    }
                    (true, false) => crate::av2_ipred::ipred_v(&mut pred, w, &top, w, h),
                    (false, true) => crate::av2_ipred::ipred_h(&mut pred, w, &left, w, h),
                    (false, false) => crate::av2_ipred::ipred_dc_128(&mut pred, w, w, h, bdmax),
                }
            }
            1..=8 if mrl_index == 0 => {
                if std::env::var("MRLDBG").is_ok_and(|v| v == format!("{px0},{py0}")) {
                    crate::dlog!("MRLDBG arm=idif mode={y_mode} pang={p_angle} mrl={mrl_index}");
                }
                // Faithful AV2 IDIF directional prediction (avm build_intra_predictors_default +
                // highbd_dr_predictor_idif). p_angle + edge availability computed above.
                recon_dir_idif(
                    &mut pred, w, h, &top, &left, corner, p_angle, (recon_midx % 7) as i32 - 3,
                    mrl_index as i32, apply_ibp, apply_ibp, above_sm, left_sm, ef, false, bdmax,
                );
            }
            1..=8 => {
                // MRL (mrl_index>0): predict from the offset reference line(s). avm
                // `av2_build_intra_predictors_high` path (no edge filter, no dir-IBP).
                let sb_bnd = by4 % sb_step4() == 0; // MRL top-line clamp at the TRUE SB top row
                crate::av2_frame::REF_LUMA.with(|r| {
                    let rb = r.borrow();
                    if let Some(rp) = rb.as_ref() {
                        recon_dir_mrl(&mut pred, w, h, rp, px0, py0, p_angle, mrl_index as i32, sb_bnd, multi_mrl, have_top, have_left, n_tr, n_bl, base, bdmax);
                    } else {
                        crate::av2_frame::RECON_PAD.with(|p| {
                            let pad = p.borrow();
                            let src = pad.first().filter(|pl| pl.w != 0).unwrap_or(&f.pl[0]);
                            recon_dir_mrl(&mut pred, w, h, src, px0, py0, p_angle, mrl_index as i32, sb_bnd, multi_mrl, have_top, have_left, n_tr, n_bl, base, bdmax);
                        });
                    }
                });
            }
            _ => {
                // DIP / filter-intra etc. not wired yet — DC fallback (will diverge; tracked).
                crate::av2_ipred::ipred_dc(&mut pred, w, &top, &left, w, h, bdmax);
            }
        }
        // Palette predictor (avm reconintra.c:1699): overwrite the (discarded) intra pred with
        // palette_colors[map] for this TX unit's window into the block-sized color map.
        if let Some((p, offx, offy)) = palette {
            for r in 0..h {
                for c in 0..w {
                    pred[r * w + c] = p.colors[p.map[(offy + r) * p.w + offx + c] as usize] as i32;
                }
            }
        }
        // residual: dequant the parsed levels → inverse transform → add.
        if !all_zero {
            use crate::av2_dequant::{cf_max, dequant_coeff, dq_lookup};
            let dq = dq_lookup(LAST_QIDX.with(|c| c.get())); // delta-q: current SB's qindex (== frame yac when off)
            if std::env::var("MPROBE").is_ok() && px0 <= 349 && 349 < px0 + w && py0 <= 29 && 29 < py0 + h {
                crate::dlog!("[MPROBE] px0={px0} py0={py0} w={w} h={h} mode={y_mode_idx} qidx={} dq={dq} dcq={} stx={_stx_type} txtp={txtp} nz={:?}",
                    LAST_QIDX.with(|c| c.get()), crate::av2_frame::F2_DCQ.with(|c| c.get()[0]),
                    cf.iter().take(64).enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v)).collect::<Vec<_>>());
            }
            // avm dequant: ROUND(level·dq, QUANT_TABLE_BITS=3) >> av2_get_tx_scale(pels), where
            // tx_scale = (pels>256)+(pels>1024) — a pels-based (NOT max-dimension) second shift.
            let pels = w * h;
            let tx_scale = (pels > 256) as u32 + (pels > 1024) as u32;
            let cfmax = cf_max((bdmax_g() + 1).trailing_zeros());
            let n = w * h;
            let dcq = crate::av2_frame::F2_DCQ.with(|c| c.get()[0]);
            let iqm = crate::av2_qm::iqm_slice(0, w, h,
                !fsc && ((txtp & 7) & 3) != 1 && (((txtp >> 5) & 7) & 3) != 1);
            let mut coeff = vec![0i32; n];
            for i in 0..n.min(cf.len()) {
                let lvl = cf[i];
                if lvl != 0 {
                    let s = (lvl < 0) as u32;
                    let q = if i == 0 && dcq != 0 { dcq } else { dq };
                    let q = crate::av2_qm::qm_apply(iqm, i, h.min(32), q);
                    if std::env::var("MDQ").is_ok() && bx4 == 16 && by4 == 0 {
                        crate::dlog!("[MDQ] i={i} lvl={lvl} q={q} w={w} h={h} iqm={:?} txtp={txtp} fsc={}", iqm.map(|(m, mw)| (m[i], mw)), fsc as u8);
                    }
                    let mag0 = dequant_coeff(lvl.unsigned_abs(), q, 3, cfmax, s, false) as i32;
                    let mag = (mag0 >> tx_scale).min(cfmax);
                    coeff[i] = if lvl < 0 { -mag } else { mag };
                }
            }
            if std::env::var("MDQDUMP").map_or(false, |v| v == format!("{bx4},{by4}")) {
                let nz: Vec<(usize, i32)> = coeff.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v)).collect();
                crate::dlog!("[MDQDUMP] ({bx4},{by4}) w={w} h={h} txtp={txtp} dq={dq} dcq={dcq} tx_scale={tx_scale} stx={_stx_type} slw={slw} slh={slh} nz={nz:?}");
            }
            let (row_ty, col_ty) = if fsc {
                (3, 3) // IDTX: identity/identity
            } else {
                (T1D[(txtp & 7) as usize & 3], T1D[((txtp >> 5) & 7) as usize & 3])
            };
            let mut residual = vec![0i32; n];
            crate::av2_itx::inv_txfm_2d(&coeff, slw, slh, row_ty, col_ty, &mut residual);
            let dump_px = std::env::var("MDQDUMP").map_or(false, |v| v == format!("{bx4},{by4}"));
            if dump_px {
                for r in 16..26.min(h) {
                    crate::dlog!("[MPX] r{r} pred=({},{}) res=({},{})", pred[r * w + 30], pred[r * w + 31], residual[r * w + 30], residual[r * w + 31]);
                }
            }
            crate::av2_itx::residual_add(&mut pred, w, &residual, w, h, 0, 0, 0, bdmax);
            if dump_px {
                for r in 16..26.min(h) {
                    crate::dlog!("[MPX] r{r} recon=({},{})", pred[r * w + 30], pred[r * w + 31]);
                }
            }
        }
        // Isolation scoreboard: compare this block's recon to the dav2d reference.
        crate::av2_frame::REF_LUMA.with(|r| {
            if let Some(rp) = r.borrow().as_ref() {
                let mut ok = true;
                'cmp: for y in 0..h {
                    for x in 0..w {
                        if pred[y * w + x].clamp(0, 255) != rp.at(px0 + x, py0 + y) {
                            ok = false;
                            break 'cmp;
                        }
                    }
                }
                crate::av2_frame::REF_SCORE.with(|s| {
                    let (o, t) = s.get();
                    s.set((o + ok as u32, t + 1));
                });
                if !ok {
                    let ang = if is_dir { p_angle } else { -1 };
                    crate::dlog!(
                        "REFMISS px=({},{}) w={w} h={h} mode={y_mode} ang={ang} az={} fsc={} mrl={mrl_index} txtp={txtp} stx={_stx_type} | mine[0]={} dav[0]={}",
                        px0, py0, all_zero as u8, fsc as u8,
                        pred[0].clamp(0, 255), rp.at(px0, py0)
                    );
                }
            }
        });
        // write reconstructed pixels back (clip to the frame — an edge block spilling past the
        // boundary still reconstructs its visible pixels).
        let wc = w.min(f.pl[0].w.saturating_sub(px0));
        let hc = h.min(f.pl[0].h.saturating_sub(py0));
        for y in 0..hc {
            let dst = (py0 + y) * stride + px0;
            // checked row write (Plane::set_row) — total, no bounds panic reachable
            f.pl[0].set_row(px0, py0 + y, &pred[y * w..y * w + wc]);
        }
        // Mirror the FULL block (incl. off-frame spill) into the padded gather buffer.
        crate::av2_frame::write_recon_pad(0, px0, py0, &pred, w, h);
        mscore_luma("intra", px0, py0, w, h, &pred, w);
        // Debug (env BRDBG): print the bottom-right block's gathered neighbours + pred so the
        // DC-from-0 cascade ROOT is visible (the first block whose top/left is wrongly 0).
        if std::env::var("BRDBG").is_ok() && px0 >= 384 && py0 >= 192 {
            let t0 = *top.get(0).unwrap_or(&-1);
            let t1 = *top.get(1).unwrap_or(&-1);
            let l0 = *left.get(0).unwrap_or(&-1);
            let l1 = *left.get(1).unwrap_or(&-1);
            let fno = crate::av2_frame::FRAME_NO.with(|c| c.get());
            crate::dlog!("BRGATHER f{fno} px=({px0},{py0}) w={w} h={h} mode={y_mode} ht={have_top} hl={have_left} top=[{t0},{t1}] left=[{l0},{l1}] pred0={} predlast={}", pred[0].clamp(0,255), pred[w*h-1].clamp(0,255));
        }
        // Record this block in the decode-order grid (for later blocks' edge availability) + its
        // joint mode (for neighbour mode-list reordering + the edge-filter smooth `type`).
        f.mark_coded(bx4, by4, bw4, bh4, joint);
        // Deblock levels come from the TX dims (dav t_dim), NOT the frame-clamped extent —
        // an edge unit with 48px visible of a 64px tx still filters at the 64px level.
        f.mark_db_lvl(bx4, by4, bw4, bh4, (slw as u8).min(3), (slh as u8).min(3));
    });
    mscref_check(bx4, by4, bw4 * 4, bh4 * 4, "intraY");
}

// ===== MHCCP (multi-hypothesis cross-component prediction, dav2d cfl_gen_y/gen_mat/calc_alphas/
// mhccp_pred). A 3-tap linear model `pred = a0·luma + a1·SQRND(luma) + a2·mid` whose alphas come
// from a per-plane 3×3 least-squares solve over the neighbour edge. All fixed-point, bit-exact.
// COMPLETE: `recon_mhccp` (below) ports gen_y + gen_mat + calc_alphas + mhccp_pred for dir ∈
// {CENTER,TOP,LEFT}, UNIFORM 4:2:0; verified byte-exact vs dav2d ground truth (all 11 cfl=3 pass).
const DIV_SCALE_SH_OFFSET: [u16; 8] = [4822, 5952, 6624, 6792, 6408, 5424, 3792, 1466];
const DIV_SCALE_SH_BIAS: [u16; 8] = [12784, 12054, 11670, 11583, 11764, 12195, 12870, 13782];
const DIV_SCALE_SH_COEFW: [u8; 8] = [214, 153, 113, 86, 67, 53, 43, 35];

/// dav2d `get_div_scale_sh`: normalize `d` to [1,2) in Q14, return (scale, shift) approximating 1/d.
fn get_div_scale_sh(d0: i32) -> (i32, i32) {
    let mut d = d0.abs().max(1);
    let sh = (d as u32).ilog2() as i32;
    let nsh = sh - 14;
    if nsh >= 0 {
        let rnd = if nsh > 0 { 1 << (nsh - 1) } else { 0 };
        d = (d + rnd) >> nsh;
    } else {
        d <<= -nsh;
    }
    d = d.clamp(1, 0x7fff) & ((1 << 14) - 1);
    let idx = (d >> 11) as usize;
    d -= DIV_SCALE_SH_OFFSET[idx] as i32;
    let scale = (((DIV_SCALE_SH_COEFW[idx] as i32 * ((d * d) >> 14)) >> 8) - (d >> 1)
        + DIV_SCALE_SH_BIAS[idx] as i32) << 2;
    (scale, sh)
}

/// dav2d `mul32`: `(a·b + round) >> sh` kept in 32-bit intermediates (drop bits, symmetric round).
fn mul32(a: i32, b: i32, sh: i32) -> i32 {
    let a2 = (a.unsigned_abs() | 1).ilog2() as i32 + 1;
    let b2 = (b.unsigned_abs() | 1).ilog2() as i32 + 1;
    let drop = if a2 + b2 > 29 { a2 + b2 - 29 } else { 0 };
    let (ash, bsh) = (drop >> 1, drop - (drop >> 1));
    let adj = sh - (ash + bsh);
    let mul = (a >> ash) * (b >> bsh);
    if adj <= 0 {
        return mul;
    }
    let bias = 1i64 << (adj - 1);
    if mul >= 0 {
        ((mul as i64 + bias) >> adj) as i32
    } else {
        -(((-(mul as i64) + bias) >> adj) as i32)
    }
}

/// Least-squares alpha derivation (dav2d `derive_alpha`, derivation.h): `num/den` in a fixed-point
/// reciprocal form, clamped to ±511, carrying `num`'s sign. Returns `alpha` unchanged if degenerate.
fn derive_alpha(num: i32, den: i32, mut alpha: i32) -> i32 {
    let max = (2 << 8) - 1; // 511
    if num != 0 && den > 0 {
        let num_abs = num.abs();
        let shift_n = (num_abs as u32).ilog2() as i32;
        let shift_d = (den as u32).ilog2() as i32;
        let e_d = den - (1 << shift_d);
        let f_d = if shift_d > 7 { (e_d + (1 << (shift_d - 8))) >> (shift_d - 7) } else { e_d << (7 - shift_d) };
        let f_n = if shift_n > 7 { (num_abs + (1 << (shift_n - 8))) >> (shift_n - 7) } else { num_abs << (7 - shift_n) };
        let shift_add = shift_d - shift_n - 8;
        if shift_add <= 1 {
            let shift0 = 9 + 7 + shift_add;
            let tmp = if shift0 < 0 {
                max
            } else {
                ((crate::av2_ipred::DIV_RECIP[f_d as usize] as i32 * f_n) >> shift0).min(max)
            };
            if tmp != 0 {
                alpha = if num < 0 { -tmp } else { tmp };
            }
        }
    }
    alpha
}
use crate::av2_ipred::fast_div32_dc;

/// CfL EXPLICIT / IMPLICIT chroma prediction (dav2d `cfl_pred`, ipred_tmpl.c:824), 4:2:0 only.
/// Reads the current block's reconstructed luma + neighbour luma/chroma (from the reference planes
/// in the isolation harness) → downsamples luma (UNIFORM/VSTRIP/GAUSS) → DC(luma,U,V) → alpha
/// (explicit: `cfl_alpha·32`; implicit: least-squares over edge samples) → per-pixel
/// `chroma_dc + apply_sign((|alpha·(ds_luma - luma_dc)| + 1024) >> 11)`. Writes into `pu`/`pv`.
#[allow(clippy::too_many_arguments)]
fn recon_cfl(
    yl: &crate::av2_frame::Plane, cu: &crate::av2_frame::Plane, cv: &crate::av2_frame::Plane,
    px0: usize, py0: usize, by4: usize, w: usize, h: usize, has_t: bool, has_l: bool,
    implicit: bool, alpha_u_raw: i32, alpha_v_raw: i32, ds: u8, bdmax: i32,
    pu: &mut [i32], pv: &mut [i32],
) {
    let (sshc, ssvc) = ss_g();
    let (lx0, ly0) = (px0 << sshc, py0 << ssvc); // luma pixel origin (ss-general)
    let is_top_sb_edge = (by4 & (sb_step4() - 1)) == 0;
    // Downsample a 2×2 luma block whose top-left is luma-relative (rx, ry); `brow` = the second
    // row offset (0 collapses the vertical pair, used at the top SB edge). Result is <<3-scaled.
    // Clamp off-frame luma reads to the plane edge (edge block's CfL luma spills past the frame).
    let (lyw, lyh) = (yl.w as i32, yl.h as i32);
    let ly = |x: i32, y: i32| yl.at((lx0 as i32 + x).clamp(0, lyw - 1) as usize, (ly0 as i32 + y).clamp(0, lyh - 1) as usize);
    // `tclamp`: the GAUSS top tap reads the CENTER instead of the row above — dav clamps it at
    // the first row of the left-neighbour column (`yleft[y ? -ystride : 0]`, ipred_tmpl.c:875)
    // and at (chroma_y & 31) == 0 rows of the in-block AC loop (`(y & 31) == 0 ? xl : xl-ystride`).
    let ds_luma = |rx: i32, ry: i32, brow: i32, tclamp: bool| -> i32 {
        // Per-format arms (dav2d cfl_pred, ipred_tmpl.c:870-951): 4:4:4 = copy<<3;
        // 4:2:2 = horizontal-only taps (GAUSS degenerates to a copy<<3); 4:2:0 = the 2-row forms.
        if sshc == 0 && ssvc == 0 {
            return ly(rx, ry) << 3;
        }
        if ssvc == 0 {
            return match ds {
                1 => {
                    let left = (rx - 1).max(rx & -64);
                    (ly(left, ry) + 2 * ly(rx, ry) + ly(rx + 1, ry)) << 1
                }
                2 => ly(rx, ry) << 3,
                _ => (ly(rx, ry) + ly(rx + 1, ry)) << 2,
            };
        }
        match ds {
            1 => {
                // VSTRIP: [1 2 1; 1 2 1]
                let left = (rx - 1).max(rx & -64);
                (ly(left, ry) + 2 * ly(rx, ry) + ly(rx + 1, ry)
                    + ly(left, ry + brow) + 2 * ly(rx, ry + brow) + ly(rx + 1, ry + brow))
            }
            2 => {
                // GAUSS: [.. 1 4 1 ..] + top/bottom
                let left = (rx - 1).max(rx & -64);
                let tb = if tclamp { 0 } else { brow };
                ly(left, ry) + 4 * ly(rx, ry) + ly(rx + 1, ry) + ly(rx, ry - tb) + ly(rx, ry + brow)
            }
            _ => (ly(rx, ry) + ly(rx + 1, ry) + ly(rx, ry + brow) + ly(rx + 1, ry + brow)) << 1, // UNIFORM
        }
    };
    let (mut dc0, mut dc1, mut dc2) = (0i64, 0i64, 0i64);
    let (skiph, skipv) = (w == 64, h == 64);
    // dav edge CfL (ipred_tmpl.c:963-975): a block spilling past the frame REPLICATES the last
    // VISIBLE column/row for the padded region — the luma downsample value AND the chroma neighbour
    // are both taken at `xlim-1`/`ylim-1`, NOT recomputed at the clamped-luma position. `xlim`/`ylim`
    // = the block's visible chroma extent.
    let xlim = w.min(cu.w.saturating_sub(px0)).max(1);
    let ylim = h.min(cu.h.saturating_sub(py0)).max(1);
    // implicit edge-sample collection
    let mut edge: Vec<(i32, i32, i32)> = Vec::with_capacity(16);
    let (mut n_top, mut n_left) = (0usize, 0usize);
    if implicit {
        if has_t && has_l {
            if w > h * 2 { n_top = 8; } else if h > w * 2 { n_left = 8; } else { n_top = 4; n_left = 4; }
        } else {
            n_top = if has_t { 8.min(w) } else { 0 };
            n_left = if has_l { 8.min(h) } else { 0 };
        }
    }
    if has_l {
        let step = if n_left != 0 { h >> n_left.trailing_zeros() } else { 0 };
        for y in 0..h {
            let yc = y.min(ylim - 1);
            let l = ds_luma(-(1 + sshc as i32), (yc << ssvc) as i32, 1, y == 0);
            if !skipv || (y & 1) == 0 {
                dc0 += l as i64;
                dc1 += cu.at(px0 - 1, py0 + yc) as i64;
                dc2 += cv.at(px0 - 1, py0 + yc) as i64;
                if implicit && w == 32 && h == 8 && std::env::var("CFLA").is_ok() {
                    crate::dlog!("[MCFLD] L y={y} l={l} u={} v={}", cu.at(px0 - 1, py0 + yc), cv.at(px0 - 1, py0 + yc));
                }
            }
            if n_left != 0 && (y & (step - 1)) == (step >> 1) {
                edge.push((l >> 3, cu.at(px0 - 1, py0 + yc), cv.at(px0 - 1, py0 + yc)));
            }
        }
    }
    if has_t {
        let brow = if ssvc == 0 || is_top_sb_edge { 0 } else { 1 };
        let tr = if ssvc == 0 { -1 } else if is_top_sb_edge { -1 } else { -2 };
        let step = if n_top != 0 { w >> n_top.trailing_zeros() } else { 0 };
        for x in 0..w {
            let xc = x.min(xlim - 1);
            let l = ds_luma((xc << sshc) as i32, tr, brow, false);
            if !skiph || (x & 1) == 0 {
                dc0 += l as i64;
                dc1 += cu.at(px0 + xc, py0 - 1) as i64;
                dc2 += cv.at(px0 + xc, py0 - 1) as i64;
                if implicit && w == 32 && h == 8 && std::env::var("CFLA").is_ok() {
                    crate::dlog!("[MCFLD] T x={x} l={l} u={} v={}", cu.at(px0 + xc, py0 - 1), cv.at(px0 + xc, py0 - 1));
                }
            }
            if n_top != 0 && (x & (step - 1)) == (step >> 1) {
                edge.push((l >> 3, cu.at(px0 + xc, py0 - 1), cv.at(px0 + xc, py0 - 1)));
            }
        }
    }
    // finalize DCs
    let (d0, d1, d2);
    if !has_t && !has_l {
        d0 = 4 * (bdmax + 1); // dav `4 << bitdepth` (mid-gray in the <<3 luma-AC scale); 8-bit = 1024
        d1 = (bdmax + 1) >> 1;
        d2 = (bdmax + 1) >> 1;
    } else {
        let npx = (if has_t { w >> skiph as usize } else { 0 }) + (if has_l { h >> skipv as usize } else { 0 });
        let fin = |acc: i64| -> i32 {
            if npx & (npx - 1) == 0 {
                ((acc + (npx as i64 >> 1)) >> npx.trailing_zeros()) as i32
            } else {
                fast_div32_dc(acc as u32, npx as u32)
            }
        };
        d0 = fin(dc0);
        d1 = fin(dc1);
        d2 = fin(dc2);
    }
    // alpha
    let (a_u, a_v);
    if implicit {
        let n = n_top + n_left;
        let count_l2 = if n > 0 { n.trailing_zeros() } else { 0 };
        let (mut sx, mut sxx, mut syu, mut syv, mut sxyu, mut sxyv) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
        for &(x, yu, yv) in &edge {
            sx += x as i64; syu += yu as i64; syv += yv as i64;
            sxx += (x * x) as i64; sxyu += (x * yu) as i64; sxyv += (x * yv) as i64;
        }
        let den = (sxx - ((sx * sx) >> count_l2)) as i32;
        a_u = derive_alpha((sxyu - ((sx * syu) >> count_l2)) as i32, den, 0);
        a_v = derive_alpha((sxyv - ((sx * syv) >> count_l2)) as i32, den, 0);
        if std::env::var("CFLA").is_ok() {
            crate::dlog!("[MCFLA] w={w} h={h} nt={n_top} nl={n_left} dc={},{},{} au={a_u} av={a_v} den={den} edge={:?}", d0, d1, d2, edge);
        }
    } else {
        a_u = alpha_u_raw * 32; // CFL_ALPHA_LOG2 = 5
        a_v = alpha_v_raw * 32;
    }
    if std::env::var("CFLT").is_ok_and(|v| v == format!("{px0},{py0}")) {
        crate::dlog!("[CFLT] px=({px0},{py0}) {w}x{h} has_t={has_t} has_l={has_l} impl={implicit} d0={d0} d1={d1} d2={d2} au={a_u} av={a_v} topsb={is_top_sb_edge}");
    }
    // combine
    for pl in 0..2 {
        let (alpha, dc_c, dst) = if pl == 0 { (a_u, d1, &mut *pu) } else { (a_v, d2, &mut *pv) };
        for y in 0..h {
            for x in 0..w {
                if alpha == 0 {
                    dst[y * w + x] = dc_c;
                } else {
                    let ac = ds_luma((x << sshc) as i32, (y << ssvc) as i32, 1, (y & 31) == 0) - d0;
                    let diff = alpha * ac;
                    let mag = (diff.abs() + 1024) >> 11;
                    let val = dc_c + if diff < 0 { -mag } else { mag };
                    dst[y * w + x] = val.clamp(0, bdmax);
                }
            }
        }
    }
}

/// MHCCP prediction (dav2d `cfl_gen_y`/`cfl_gen_mat`/`cfl_calc_alphas`/`cfl_mhccp_pred`), 4:2:0,
/// UNIFORM filter, dir ∈ {CENTER=0, TOP=1}. Reads reconstructed luma (REF_LUMA) + neighbour chroma
/// (REF_CHROMA). `n_tr`/`n_bl` are the top-right / bottom-left extension in chroma-4-units.
#[allow(clippy::too_many_arguments)]
fn recon_mhccp(
    yl: &crate::av2_frame::Plane, cu: &crate::av2_frame::Plane, cv: &crate::av2_frame::Plane,
    px0: usize, py0: usize, w: usize, h: usize, has_t: bool, has_l: bool, is_top_sb_edge: bool,
    dir: u8, n_tr: usize, n_bl: usize, bdmax: i32, pu: &mut [i32], pv: &mut [i32],
) {
    let (mssh, mssv) = ss_g();
    let (lx0, ly0) = ((px0 << mssh) as i32, (py0 << mssv) as i32);
    // Clamp off-frame luma reads to the plane edge (a bottom/right edge block's MHCCP samples
    // spill past the visible frame). TODO verify vs dav's padded-recon read for byte-exactness.
    let (lyw, lyh) = (yl.w as i32, yl.h as i32);
    let ly = |x: i32, y: i32| yl.at((lx0 + x).clamp(0, lyw - 1) as usize, (ly0 + y).clamp(0, lyh - 1) as usize);
    // dav ipred_tmpl.c:1284-1285: bd = bitdepth, mid = 1 << (bd-1).
    let bd = 32 - (bdmax as u32).leading_zeros() as i32;
    let mid = 1 << (bd - 1);
    let sqrnd = |v: i32| (v * v + mid) >> bd;
    let (dir_t, dir_l) = (dir == 1, dir == 2);
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    // reference dims (recon_tmpl.c:3029-3084). The STRIP dims use the FRAME-CLAMPED block
    // extent (dav ctw4/cth4 = imin(uv_t_dim, in-frame cells)): a bottom/right-edge block's
    // model samples cover only the in-frame rows/cols (16f_tip2 (32,96) 32x32: 24 in-frame
    // chroma rows -> refh 25 with top, NOT 33 — the off-frame rows drift the alphas).
    // The PREDICTION below still covers the full w×h (writes clip at commit).
    let w_ref = w.min(cu.w.saturating_sub(px0)).max(1);
    let h_ref = h.min(cu.h.saturating_sub(py0)).max(1);
    let mut refw = w_ref + n_tr * 4 + if has_l { 2 } else { 0 };
    let mut subleft = has_l && !dir_l;
    if refw > 64 { refw = 64; subleft = false; }
    let refh_geny = h_ref + n_bl * 4;
    let refh = (refh_geny + has_t as usize).min(64 - 2 * has_t as usize);
    let refw_geny = refw - subleft as usize;
    let st = (refw + 63) & !63; // luma-buffer top stride (8-bit)
    // --- gen_y: build the downsampled-luma reference buffer (dual-stride) ---
    // dav cfl_gen_y_420_c (ipred_tmpl.c:1113): three seq-selected filters with per-REGION
    // l-tap clamps (left-strip/block: `(n_left&1) ? c-1 : max(c-1,0)`; top region:
    // `n_left ? c-1 : max(c-1,0)`) and a GAUSS vertical top tap whose row is region-specific
    // (`tsy` below). `brow`=0 collapses the bottom tap onto the center row (SB-edge top rows).
    let mut luma = vec![0i32; 2 * 64 * 64 + 256];
    let ds = HDR_TOOL_CFG.with(|c| c.get().cfl_ds_filter);
    let filt = |sx0: i32, sy: i32, c: i32, brow: i32, lclamp: bool, tsy: i32| -> i32 {
        let cc = sx0 + c;
        let r = cc + 1;
        let l = if lclamp { sx0 + (c - 1).max(0) } else { cc - 1 };
        // Per-format: 4:4:4 = identity; 4:2:2 = horizontal-only (GAUSS degenerates to copy,
        // mirroring cfl_pred's 422 arms); 4:2:0 = the 2-row forms.
        if mssh == 0 && mssv == 0 {
            return ly(cc, sy);
        }
        if mssv == 0 {
            return match ds {
                1 => (ly(l, sy) + 2 * ly(cc, sy) + ly(r, sy)) >> 2,
                2 => ly(cc, sy),
                _ => (ly(cc, sy) + ly(r, sy)) >> 1,
            };
        }
        match ds {
            1 => (ly(l, sy) + 2 * ly(cc, sy) + ly(r, sy)
                + ly(l, sy + brow) + 2 * ly(cc, sy + brow) + ly(r, sy + brow)) >> 3,
            2 => (ly(l, sy) + 4 * ly(cc, sy) + ly(r, sy) + ly(cc, tsy) + ly(cc, sy + brow)) >> 3,
            _ => (ly(cc, sy) + ly(r, sy) + ly(cc, sy + brow) + ly(r, sy + brow)) >> 2,
        }
    };
    let src_sx0 = -((n_left as i32) << mssh); // src -= n_left chroma cols (luma-relative)
    let mut dst_off = 0usize;
    let mut dst_left_off = n_top * st + 64 * 64;
    if has_t {
        // top rows: base row = ly0 - n_top*2 (non-edge) or -1 (sb edge); brow collapses at edge.
        // GAUSS top-tap offset t: -b for n_top==1 (0 at the edge since b=0), else 0; after each
        // non-edge row it becomes -1 row (dav: t = -src_stride). At the edge top never advances.
        let mut top_sy = if is_top_sb_edge { -1 } else { -((n_top as i32) << mssv) };
        let brow = if is_top_sb_edge || mssv == 0 { 0 } else { 1 };
        let mut t_off: i32 = if n_top == 1 && !is_top_sb_edge { -1 } else { 0 };
        for _y in 0..n_top {
            for x in 0..n_left {
                luma[dst_left_off + x] =
                    filt(src_sx0, top_sy, ((x << mssh)) as i32, brow, (n_left & 1) == 0, top_sy + t_off);
            }
            for x in n_left..refw_geny {
                luma[dst_off + x - n_left] =
                    filt(src_sx0, top_sy, ((x << mssh)) as i32, brow, n_left == 0, top_sy + t_off);
            }
            if !is_top_sb_edge { top_sy += 1 << mssv; t_off = -1; }
            dst_left_off += n_left;
            dst_off += st;
        }
    }
    // block rows: GAUSS top tap = row above (first row: SB-edge/row-above when has_t, else self)
    let lclamp_blk = (n_left & 1) == 0;
    let ssv_one = mssv as i32; // second-row offset of the vertical pair (0 when no vertical ss)
    let mut src_sy = 0i32;
    for yy in 0..h {
        let tsy = if src_sy == 0 { if has_t { -1 } else { 0 } } else { src_sy - 1 };
        // LEFT-strip samples only over the FRAME-CLAMPED rows (dav's strip height = clamped
        // cth4); the BLOCK region still fills all h rows (the pred consumes them).
        if yy < h_ref {
            for x in 0..n_left {
                luma[dst_left_off + x] = filt(src_sx0, src_sy, ((x << mssh)) as i32, ssv_one, lclamp_blk, tsy);
            }
            dst_left_off += n_left;
        }
        for x in n_left..n_left + w {
            luma[dst_off + x - n_left] = filt(src_sx0, src_sy, ((x << mssh)) as i32, ssv_one, lclamp_blk, tsy);
        }
        src_sy += 1 << mssv;
        dst_off += w;
    }
    // bottom-left rows (only dst_left) — continue from the CLAMPED strip row
    let mut src_sy = (h_ref as i32) << mssv;
    for _y in 0..(refh_geny - h_ref) {
        for x in 0..n_left {
            luma[dst_left_off + x] = filt(src_sx0, src_sy, ((x << mssh)) as i32, ssv_one, lclamp_blk, src_sy - 1);
        }
        src_sy += 1 << mssv;
        dst_left_off += n_left;
    }
    // --- gen_mat + per-plane alpha accumulation over the edge samples (same n-order) ---
    let left_base = n_top * st + 64 * 64;
    let lbuf = |i: i32| luma[(left_base as i32 + i) as usize];
    let ybuf = |i: i32| luma[i as usize];
    let dtl = (dir_t as i32) | (dir_l as i32);
    // gen_mat: iterate the edge samples in dav2d order (corner, top row, left col), build imat.
    let (mut imat0, mut imat1): (Vec<i32>, Vec<i32>) = (Vec::new(), Vec::new());
    if has_t {
        for i in 0..n_left as i32 {
            imat0.push(lbuf(i));
            imat1.push(sqrnd(if i == 0 { lbuf(i + dtl) } else { ybuf(0) }));
        }
        let start = (!dir_l && !has_l) as i32;
        let bound = refw as i32 - n_left as i32 - 1 - (start == 0) as i32; // - !start
        for i in start..bound {
            imat0.push(ybuf(i));
            imat1.push(sqrnd(ybuf(dir_t as i32 * st as i32 + i + dir_l as i32)));
        }
    }
    if has_l {
        let start = (dir_t && !has_t) as i32;
        for i in (1 - start)..(refh as i32 - start - 1) {
            imat0.push(lbuf(i * n_left as i32));
            imat1.push(sqrnd(lbuf((i + dir_t as i32) * n_left as i32 + dir_l as i32)));
        }
    }
    let n = imat0.len() as i64;
    let mut mat = [[0i64; 3]; 3];
    for k in 0..imat0.len() {
        let (v0, v1) = (imat0[k], imat1[k]);
        mat[0][0] += (v0 * v0) as i64;
        mat[0][1] += (v0 * v1) as i64;
        mat[0][2] += (v0 << (bd - 1)) as i64;
        mat[1][1] += (v1 * v1) as i64;
        mat[1][2] += (v1 << (bd - 1)) as i64;
    }
    // calc_alphas: accumulate the per-plane rhs over the SAME edge order (imat index aligns).
    let calc_a = |cc: &crate::av2_frame::Plane| -> [i64; 3] {
        let mut a = [0i64; 3];
        let mut nn = 0usize;
        // Clamp off-frame chroma reference reads to the plane edge (edge block's MHCCP refs spill
        // past the visible frame). TODO verify vs dav's padded-recon read for byte-exactness.
        let (ccw, cch) = (cc.w as i32, cc.h as i32);
        let cat = |x: i32, y: i32| cc.at(x.clamp(0, ccw - 1) as usize, y.clamp(0, cch - 1) as usize);
        if has_t {
            let start = !has_l as i32;
            for i in start..(refw as i32 - 1 - (start == 0) as i32) {
                let ch = cat(px0 as i32 - has_l as i32 + i, py0 as i32 - 1);
                a[0] += (imat0[nn] * ch) as i64;
                a[1] += (imat1[nn] * ch) as i64;
                a[2] += (ch << (bd - 1)) as i64;
                nn += 1;
            }
        }
        if has_l {
            for i in (!has_t as i32)..(refh as i32 - 1 - has_t as i32) {
                let ch = cat(px0 as i32 - 1, py0 as i32 + i);
                a[0] += (imat0[nn] * ch) as i64;
                a[1] += (imat1[nn] * ch) as i64;
                a[2] += (ch << (bd - 1)) as i64;
                nn += 1;
            }
        }
        a
    };
    let mut au = calc_a(cu);
    let mut av = calc_a(cv);
    mat[2][2] = n << ((bd - 1) * 2);
    let nl2 = 63 - (n as u64).leading_zeros() as i32;
    let mat_sh = 22 - 2 * bd - nl2 - (n & ((1 << nl2) - 1) != 0) as i32;
    let shift_row = |v: &mut [i64; 3]| {
        for e in v.iter_mut() {
            if mat_sh > 0 { *e <<= mat_sh; } else if mat_sh < 0 { *e >>= -mat_sh; }
        }
    };
    for r in 0..3 { shift_row(&mut mat[r]); }
    shift_row(&mut au);
    shift_row(&mut av);
    mat[0][0] += (2 << (bd - 8)) as i64;
    mat[1][1] += (2 << (bd - 8)) as i64;
    mat[2][2] += (2 << (bd - 8)) as i64;
    mat[1][0] = mat[0][1]; mat[2][0] = mat[0][2]; mat[2][1] = mat[1][2];
    let mati = [
        [mat[0][0] as i32, mat[0][1] as i32, mat[0][2] as i32],
        [mat[1][0] as i32, mat[1][1] as i32, mat[1][2] as i32],
        [mat[2][0] as i32, mat[2][1] as i32, mat[2][2] as i32],
    ];
    // --- Gaussian elimination (dav2d cfl_calc_alphas) per plane ---
    let solve = |a_in: [i64; 3], m: [[i32; 3]; 3]| -> [i32; 3] {
        let mut a = [a_in[0] as i32, a_in[1] as i32, a_in[2] as i32];
        let mut t = [[0i32; 2]; 3];
        let (s, sh) = get_div_scale_sh(m[0][0]);
        t[0][0] = mul32(m[0][1], s, sh);
        t[0][1] = mul32(m[0][2], s, sh);
        a[0] = mul32(a[0], s, sh);
        t[1][0] = m[1][1] - mul32(m[1][0], t[0][0], 16);
        t[1][1] = m[1][2] - mul32(m[1][0], t[0][1], 16);
        a[1] -= mul32(m[1][0], a[0], 16);
        t[2][0] = m[2][1] - mul32(m[2][0], t[0][0], 16);
        t[2][1] = m[2][2] - mul32(m[2][0], t[0][1], 16);
        a[2] -= mul32(m[2][0], a[0], 16);
        let (s, sh) = get_div_scale_sh(t[1][0]);
        t[1][1] = mul32(t[1][1], s, sh);
        a[1] = mul32(a[1], s, sh);
        t[2][1] -= mul32(t[2][0], t[1][1], 16);
        a[2] -= mul32(t[2][0], a[1], 16);
        let (s, sh) = get_div_scale_sh(t[2][1]);
        a[2] = mul32(a[2], s, sh);
        a[1] -= mul32(t[1][1], a[2], 16);
        a[0] -= mul32(t[0][0], a[1], 16) + mul32(t[0][1], a[2], 16);
        a
    };
    let alpha_u = solve(au, mati);
    let alpha_v = solve(av, mati);
    if std::env::var("CMH").is_ok() {
        crate::dlog!("[CMH] px0={px0} py0={py0} w={w} h={h} dir={dir} bd={bd} n={} mat0={:?} mat1={:?} mat2={:?} au={au:?} av={av:?} alpha_u={alpha_u:?} alpha_v={alpha_v:?}",
            imat0.len(), mati[0], mati[1], mati[2]);
    }
    // --- mhccp_pred (dir ∈ {CENTER, TOP, LEFT}) ---
    let blk_base = n_top * st; // block region row 0
    let mh_left_base = n_top * st + 64 * 64 + n_left * n_top; // block-region left samples
    let a2v_u = mul32(alpha_u[2], mid, 16);
    let a2v_v = mul32(alpha_v[2], mid, 16);
    for (alpha, a2v, dst) in [(alpha_u, a2v_u, &mut *pu), (alpha_v, a2v_v, &mut *pv)] {
        for y in 0..h {
            for x in 0..w {
                let cur = luma[blk_base + y * w + x];
                // CENTER: v0 = current. TOP: v0 = row above. LEFT: v0 = column left (x=0 → left ref).
                // dav2d cfl_mhccp_pred (ipred_tmpl.c:1510): dir_t reads the row above via
                // offset `((!!y)|has_t)*w` — at y==0 with NO top (has_t=false) that offset is 0,
                // so v0 = the current pixel (edge-replicated), NOT the absent top row. dir_l is
                // symmetric (`imax(x-1,0)` → cur at x==0 when the left is unavailable).
                let v0 = if dir_t {
                    if y == 0 {
                        if has_t { luma[blk_base - st + x] } else { cur }
                    } else {
                        luma[blk_base + (y - 1) * w + x]
                    }
                } else if dir_l {
                    if x == 0 {
                        if has_l { luma[mh_left_base + y * n_left + 1] } else { cur }
                    } else {
                        luma[blk_base + y * w + (x - 1)]
                    }
                } else {
                    cur
                };
                let v1 = sqrnd(cur);
                let val = mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v;
                dst[y * w + x] = val.clamp(0, bdmax);
                if y == 0 && x < 6 && std::env::var("CMH").is_ok() {
                    crate::dlog!("[MPX] y={y} x={x} v0={v0} v1={v1} cur={cur} out={}", dst[y * w + x]);
                }
            }
        }
    }
}

/// Frame-1 intra CHROMA reconstruction (U + V). Mirrors `recon_intra_luma`: gather chroma edges
/// (from REF_CHROMA when scoring), dispatch by `uv_mode`, dequant the parsed chroma levels →
/// inverse transform → residual-add, score vs the dav2d chroma reference. CfL/MHCCP predict a
/// DC base for now (the cross-component AC is a follow-up); non-CfL reuses the luma predictors.
#[allow(clippy::too_many_arguments)]
fn recon_intra_chroma(
    cbx4: usize,
    cby4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    slw: usize,
    slh: usize,
    uv_mode: u8,
    uv_angle: i32,
    cfl_mode: u8,
    cfl_alpha_u: i32,
    cfl_alpha_v: i32,
    mh_dir: u8,
    cf_u: &[i32],
    cf_v: &[i32],
    az_u: bool,
    az_v: bool,
    have_left: bool,
    have_top: bool,
) {
    use crate::av2_frame::{FRAME, RECON_ACTIVE, REF_CHROMA, REF_CHROMA_SCORE};
    if !RECON_ACTIVE.with(|a| a.get()) {
        return;
    }
    let (w, h) = (4usize << slw, 4usize << slh);
    let (px0, py0) = (cbx4 * 4, cby4 * 4);
    if std::env::var("CMODE").is_ok() {
        crate::dlog!("[CMODE] cpx={px0} cpy={py0} w={w} h={h} uv_mode={uv_mode} cfl={cfl_mode} mh={mh_dir} au={cfl_alpha_u} av={cfl_alpha_v}");
    }
    FRAME.with(|fr| {
        let mut f = fr.borrow_mut();
        if f.pl[1].w == 0 {
            return;
        }
        // Mark the chroma decode-order availability grid for EVERY decoded leaf, including
        // frame-edge blocks (dav2d marks `is_coded[CHROMA]` for the full extent). Must precede
        // the off-frame bounds check, or an interior chroma block's top-right / bottom-left
        // availability is understated. Mirrors the luma fix in recon_intra_luma.
        f.mark_coded_c_avail(bx4, by4, bw4, bh4);
        if px0 >= f.pl[1].w || py0 >= f.pl[1].h {
            return; // ENTIRELY off-frame. A partial (edge-spilling) block still recons its visible
                    // chroma (the write below clips) so a later in-frame neighbour reads real recon.
        }
        let (yac, bdmax, stride, ef, ibp_on) =
            (f.yac, f.bitdepth_max, f.pl[1].stride, f.edge_filter, f.ibp);
        let base = (bdmax + 1) >> 1; // 1 << (bitdepth - 1)
        let is_dir = (1..=8).contains(&uv_mode) && cfl_mode == 0;
        if std::env::var("MUVM").map_or(false, |v| { let pp: Vec<usize> = v.split(",").filter_map(|x| x.parse().ok()).collect(); pp.len() == 2 && pp[0] == px0 && pp[1] == py0 }) {
            crate::dlog!("[MUVM] f={} px=({px0},{py0}) {w}x{h} uv_mode={uv_mode} cfl={cfl_mode} mh={mh_dir} uv_angle={uv_angle} az_u={az_u} bx4={bx4} by4={by4} bw4={bw4} bh4={bh4}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()));
        }
        let p_angle = if is_dir { av2_p_angle(uv_mode, 3 * uv_angle, 0, w, h) } else { 0 };
        let apply_ibp = ibp_on && !(w == 4 && h == 4);
        // Chroma top-right / bottom-left availability from the CHROMA decode-order grid (avm
        // `is_mi_coded[CHROMA tree]`), in luma-mi units, px subsampled 4:2:0. SMOOTH's `top[w]`/
        // `left[h]` anchors + directional edge extension read these. Once per block (same U/V).
        let (iw4, ih4) = (f.iw4, f.ih4);
        let (ssh_a, ssv_a) = ss_g();
        let luma_w = (w << ssh_a) as i32; // luma-px width of the chroma block region
        let luma_h = (h << ssv_a) as i32;
        // Right/bottom availability clamps to the TILE, not the frame (dav ts->tiling.col_end/
        // row_end): a tile-edge block must not extend its edges from across the boundary —
        // in tile-sequential decode those pixels don't even exist yet.
        let tb = TILE_B.with(|t| t.get());
        let (ce4, re4) = (tb.1.min(iw4), tb.3.min(ih4));
        let xr = (ce4 as i32 - (bx4 + bw4) as i32) * 4;
        let yd = (re4 as i32 - (by4 + bh4) as i32) * 4;
        let right_avail = bx4 + bw4 < ce4;
        let bottom_avail = yd > 0 && by4 + bh4 < re4;
        let n_tr = if have_top {
            let (av, px) = has_top_right(bx4, by4, bw4, luma_w, xr, have_top, right_avail, |r, c| f.mi_coded_c_at(r, c));
            if av { (px as usize) >> ssh_a } else { 0 }
        } else {
            0
        };
        let n_bl = if have_left {
            let (av, px) = has_bottom_left(bx4, by4, bh4, luma_h, yd, bottom_avail, have_left, |r, c| f.mi_coded_c_at(r, c));
            if av { (px as usize) >> ssv_a } else { 0 }
        } else {
            0
        };
        // Neighbour SMOOTH flags for the intra edge filter (dav2d `b->is_sm[1].a/.l`, chroma tree).
        let ciw4 = f.ciw4;
        let sm_at = |cx: i32, cy: i32| -> bool {
            cx >= 0 && cy >= 0 && (cx as usize) < ciw4 && (cy as usize) < f.cih4
                && f.sm_c[cy as usize * ciw4 + cx as usize] != 0
        };
        let above_sm = have_top && sm_at(cbx4 as i32, cby4 as i32 - 1);
        let left_sm = have_left && sm_at(cbx4 as i32 - 1, cby4 as i32);
        // Chroma directional does NOT get IBP (dav2d `apply_ibp &= uv_mode == DC_PRED`).
        let dir_ibp = false;
        // CfL EXPLICIT(1)/IMPLICIT(2): full U/V from the reconstructed luma. MHCCP(3): 3-param solve.
        let cfl_pred: Option<(Vec<i32>, Vec<i32>)> = if cfl_mode != 0 {
            let ds = HDR_TOOL_CFG.with(|c| c.get().cfl_ds_filter);
            let is_top_sb = (by4 & (sb_step4() - 1)) == 0;
            REF_CHROMA.with(|rc| {
                crate::av2_frame::REF_LUMA.with(|rl| {
                    let (rlb, rcb) = (rl.borrow(), rc.borrow());
                    // Isolation harness reads dav's REF planes when loaded (frame 1); for frame 2
                    // they are cleared → read the assembled FRAME planes. dav's edge CfL
                    // (ipred_tmpl.c:963-975 + 1062-1072) REPLICATES the last VISIBLE column
                    // (`utop[xlim-1]`) into the padded region for the luma DC/AC, the chroma DC, the
                    // implicit-alpha edge samples, AND the output — so an edge block spilling past
                    // the frame just clamps every read to the visible edge. `Plane::at` already
                    // clamps to `w-1`/`h-1` (== that replication), so NO guard is needed: edge
                    // blocks run CfL with clamped reads instead of wrongly falling back to DC.
                    let yl = rlb.as_ref().unwrap_or(&f.pl[0]);
                    let cu = rcb[0].as_ref().unwrap_or(&f.pl[1]);
                    let cv = rcb[1].as_ref().unwrap_or(&f.pl[2]);
                    if yl.w != 0 {
                        let (mut pu, mut pv) = (vec![0i32; w * h], vec![0i32; w * h]);
                        if cfl_mode == 3 {
                            recon_mhccp(yl, cu, cv, px0, py0, w, h, have_top, have_left,
                                is_top_sb, mh_dir, n_tr >> 2, n_bl >> 2, bdmax, &mut pu, &mut pv);
                        } else {
                            recon_cfl(yl, cu, cv, px0, py0, by4, w, h, have_top, have_left,
                                cfl_mode == 2, cfl_alpha_u, cfl_alpha_v, ds, bdmax, &mut pu, &mut pv);
                        }
                        Some((pu, pv))
                    } else {
                        None
                    }
                })
            })
        } else {
            None
        };
        let (mut ok_u, mut ok_v) = (false, false);
        for pl in 0..2usize {
            let cf = if pl == 0 { cf_u } else { cf_v };
            let az = if pl == 0 { az_u } else { az_v };
            let (mut top, mut left, corner) = REF_CHROMA.with(|r| {
                let rb = r.borrow();
                if let Some(rp) = rb[pl].as_ref() {
                    crate::av2_frame::gather_edges(rp, px0, py0, w, h, have_top, have_left, n_tr, n_bl, base)
                } else {
                    crate::av2_frame::RECON_PAD.with(|p| {
                        let pad = p.borrow();
                        let src = pad.get(pl + 1).filter(|pl| pl.w != 0).unwrap_or(&f.pl[pl + 1]);
                        crate::av2_frame::gather_edges(src, px0, py0, w, h, have_top, have_left, n_tr, n_bl, base)
                    })
                }
            });
            // TILE/FRAME-BOTTOM left-column clamp (dav prepare_intra_edges max_height =
            // tiling.row_end): a block spilling past the frame bottom reads its below-bottom
            // left samples as a REPLICATE of the last in-frame row — NOT the recon-pad content
            // (mine's pad holds the neighbour's off-frame extension, which dav ignores here).
            // 16f_tip2 key leaf (208,96) 8x32: chroma rows 120..127 are off-frame; avm
            // replicates left[23]=173 (DC 156), mine read pad garbage (DC 132).
            {
                let re4 = TILE_B.with(|t| t.get().3).min(f.ih4); // luma cells
                let bot_px = ((re4 * 4) >> ssv_a).min(f.pl[pl + 1].h);
                if have_left && py0 + h > bot_px && bot_px > py0 {
                    let lim = bot_px - py0;
                    let last = left[lim - 1];
                    for v in left[lim..h].iter_mut() {
                        *v = last;
                    }
                }
                if std::env::var("MUVP").map_or(false, |v| v == format!("{cbx4},{cby4},{}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()))) {
                    crate::dlog!("[MUVP] pl={pl} cpx=({px0},{py0}) {w}x{h} mode={uv_mode} ang={uv_angle} ntr={n_tr} nbl={n_bl} top={:?} left={:?}", &top[..(w + 4).min(top.len())], &left[..(h + 4).min(left.len())]);
                }
                // symmetric top clamp at the tile/frame right edge
                let ce4 = TILE_B.with(|t| t.get().1).min(f.iw4);
                let right_px = ((ce4 * 4) >> ssh_a).min(f.pl[pl + 1].w);
                if have_top && px0 + w > right_px && right_px > px0 {
                    let lim = right_px - px0;
                    let last = top[lim - 1];
                    for v in top[lim..w].iter_mut() {
                        *v = last;
                    }
                }
            }
            let mut pred = vec![0i32; w * h];
            use crate::av2_ipred::*;
            if let Some((ref pu, ref pv)) = cfl_pred {
                // CfL EXPLICIT/IMPLICIT full prediction (luma-AC applied).
                pred.copy_from_slice(if pl == 0 { pu } else { pv });
            } else if cfl_mode != 0 || uv_mode == 0 {
                if std::env::var("MUVM").map_or(false, |v| { let pp: Vec<usize> = v.split(",").filter_map(|x| x.parse().ok()).collect(); pp.len() == 2 && pp[0] == px0 && pp[1] == py0 }) {
                    crate::dlog!("[MUVME] pl={pl} ht={have_top} hl={have_left} top={:?} leftALL={:?}", &top[..8.min(top.len())], &left[..32.min(left.len())]);
                }
                // DC base (MHCCP → DC for now; DC uv_mode; CfL when luma ref unavailable).
                match (have_top, have_left) {
                    (true, true) => ipred_dc(&mut pred, w, &top, &left, w, h, bdmax),
                    (false, true) => ipred_dc_left(&mut pred, w, &left, w, h),
                    (true, false) => ipred_dc_top(&mut pred, w, &top, w, h),
                    (false, false) => ipred_dc_128(&mut pred, w, w, h, bdmax),
                }
                // Chroma DC uses the IBP gradient blend too (avm applies IBP to the DC base).
                if std::env::var("MUVM").map_or(false, |v| { let pp: Vec<usize> = v.split(",").filter_map(|x| x.parse().ok()).collect(); pp.len() == 2 && pp[0] == px0 && pp[1] == py0 }) {
                    crate::dlog!("[MUVMD] pl={pl} dc_pred0={} apply_ibp={apply_ibp}", pred[0]);
                }
                if apply_ibp && (have_top || have_left) {
                    ipred_ibp_dc(&mut pred, w, &top, &left, w, h, have_top, have_left);
                }
                if std::env::var("MUVM").map_or(false, |v| { let pp: Vec<usize> = v.split(",").filter_map(|x| x.parse().ok()).collect(); pp.len() == 2 && pp[0] == px0 && pp[1] == py0 }) {
                    crate::dlog!("[MUVMD] pl={pl} post_ibp pred0={} pred_r14={:?}", pred[0], &pred[14 * w..14 * w + 4]);
                }
            } else {
                match uv_mode {
                    9 => ipred_smooth(&mut pred, w, &top, &left, w, h),
                    10 => ipred_smooth_v(&mut pred, w, &top, &left, w, h),
                    11 => ipred_smooth_h(&mut pred, w, &top, &left, w, h),
                    PAETH_PRED => match (have_top, have_left) {
                        (true, true) => ipred_paeth(&mut pred, w, &top, &left, corner, w, h),
                        (true, false) => ipred_v(&mut pred, w, &top, w, h),
                        (false, true) => ipred_h(&mut pred, w, &left, w, h),
                        (false, false) => ipred_dc_128(&mut pred, w, w, h, bdmax),
                    },
                    1..=8 => {
                        let dbg_on = std::env::var("MUVP").map_or(false, |v| v == format!("{cbx4},{cby4},{}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get())));
                        crate::av2_ipred::DIR_DBG.with(|c| c.set(dbg_on));
                        recon_dir_idif(
                            &mut pred, w, h, &top, &left, corner, p_angle, uv_angle, 0,
                            apply_ibp, apply_ibp, above_sm, left_sm, ef, true, bdmax,
                        );
                        let _ = dir_ibp;
                        crate::av2_ipred::DIR_DBG.with(|c| c.set(false));
                        if std::env::var("MUVP").map_or(false, |v| v == format!("{cbx4},{cby4},{}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get()))) {
                            crate::dlog!("[MUVPP] pl={pl} p={p_angle} ibp={apply_ibp} sm=({above_sm},{left_sm}) ef={ef} pred_r0={:?}", &pred[..w.min(32)]);
                            crate::dlog!("[MUVPP] pl={pl} pred_r1={:?}", &pred[w..w + w.min(32)]);
                        }
                    }
                    _ => ipred_dc(&mut pred, w, &top, &left, w, h, bdmax),
                }
            }
            // residual (dequant chroma levels → inverse transform → add). Chroma qindex = yac
            // (uac/vac deltas are 0 for this clip).
            if !az {
                use crate::av2_dequant::{cf_max, dequant_coeff, dq_lookup};
                let dq = dq_lookup(LAST_QIDX.with(|c| c.get())); // delta-q: current SB's qindex (== frame yac when off)
                let pels = w * h;
                let tx_scale = (pels > 256) as u32 + (pels > 1024) as u32;
                let cfmax = cf_max((bdmax_g() + 1).trailing_zeros());
                let n = w * h;
                // Intra chroma txtp is mode-derived and ALWAYS a 2D (DCT/ADST) combo — QM applies.
                let iqm = crate::av2_qm::iqm_slice(1 + pl, w, h, true);
                let mut coeff = vec![0i32; n];
                for i in 0..n.min(cf.len()) {
                    let lvl = cf[i];
                    if lvl != 0 {
                        let s = (lvl < 0) as u32;
                        let q = crate::av2_qm::qm_apply(iqm, i, h.min(32), dq);
                        let mag0 = dequant_coeff(lvl.unsigned_abs(), q, 3, cfmax, s, false) as i32;
                        let mag = (mag0 >> tx_scale).min(cfmax);
                        coeff[i] = if lvl < 0 { -mag } else { mag };
                    }
                }
                let mut residual = vec![0i32; n];
                // Intra chroma txtp is MODE-DERIVED (dav2d recon_tmpl.c:480 + dav2d_txtp_from_uvmode).
                // Parse-safe (all these are TX_CLASS_2D, so the coef context is unaffected) but the
                // inverse transform differs: DC/D45 → DCT_DCT, V/D113/D67/SMOOTH_V → ADST_DCT,
                // H/D157/D203/SMOOTH_H → DCT_ADST, D135/SMOOTH/PAETH → ADST_ADST. CfL → DCT_DCT.
                const TXTP_FROM_UVMODE: [u8; 13] = [
                    DCT_DCT, ADST_DCT, DCT_ADST, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_ADST,
                    ADST_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_ADST,
                ];
                const T1D: [usize; 4] = [0, 3, 1, 2];
                // dav2d applies wide_angle_remap to uv_mode BEFORE this LUT (recon_tmpl.c:1592), so
                // a wide-angle-remapped directional block's txtp follows the remapped mode.
                let remapped_mode = wide_angle_remap_mode(uv_mode, uv_angle, 0, w, h);
                let uvtt_raw = if cfl_mode != 0 { DCT_DCT } else { TXTP_FROM_UVMODE[remapped_mode as usize] };
                // Size clamp (recon_tmpl.c:481-488): (flip)adst is only valid for tx dims ≤ 16px, so
                // a ≥32px dim downgrades that axis' adst to DCT (whole tx → DCT_DCT here).
                let is_16 = slw == 2 && slh == 2;
                let uvtt = if ((1usize << slw) >= 8 && uvtt_raw & 0x02 != 0)
                    || ((1usize << slh) >= 8 && uvtt_raw & 0x40 != 0)
                    || (is_16 && ((uvtt_raw & 0x47) == 0x41 || (uvtt_raw & 0xe2) == 0x22))
                {
                    DCT_DCT
                } else {
                    uvtt_raw
                };
                let row_ty = T1D[(uvtt & 7) as usize & 3];
                let col_ty = T1D[((uvtt >> 5) & 7) as usize & 3];
                crate::av2_itx::inv_txfm_2d(&coeff, slw, slh, row_ty, col_ty, &mut residual);
                if px0 == 96 && py0 == 72 && std::env::var("CMH").is_ok() {
                    crate::dlog!("[MRES] pl={pl} dq={dq} cfmax={cfmax} txs={tx_scale} lvl0..4={:?} coef0..4={:?} res_r0={:?} pred_r0={:?}",
                        &cf[..4.min(cf.len())], &coeff[..4], &residual[..4], &pred[..4]);
                }
                crate::av2_itx::residual_add(&mut pred, w, &residual, w, h, 0, 0, 0, bdmax);
            }
            // score vs the dav2d chroma reference
            REF_CHROMA.with(|r| {
                if let Some(rp) = r.borrow()[pl].as_ref() {
                    let mut ok = true;
                    let az = if pl == 0 { az_u } else { az_v };
                    // Isolation-scoring probe: only compare the VISIBLE region (an edge block spills
                    // past the reference plane — the recon write below already clips to in-frame).
                    let rhc = h.min(rp.h.saturating_sub(py0));
                    let rwc = w.min(rp.w.saturating_sub(px0));
                    'cmp: for y in 0..rhc {
                        for x in 0..rwc {
                            if pred[y * w + x].clamp(0, 255) != rp.at(px0 + x, py0 + y) {
                                ok = false;
                                if az {
                                    let maxd = (0..rhc).flat_map(|yy| (0..rwc).map(move |xx| (xx, yy)))
                                        .map(|(xx, yy)| (pred[yy*w+xx].clamp(0,255) - rp.at(px0+xx,py0+yy) as i32).abs())
                                        .max().unwrap_or(0);
                                    crate::dlog!("CWMISS px=({px0},{py0}) pl={pl} uvmode={uv_mode} ang={} (x={x},y={y}) mine={} dav={} maxd={maxd}",
                                        if is_dir { p_angle } else { -1 }, pred[y*w+x].clamp(0,255), rp.at(px0+x,py0+y));
                                }
                                break 'cmp;
                            }
                        }
                    }
                    if pl == 0 { ok_u = ok } else { ok_v = ok };
                }
            });
            let wc = w.min(f.pl[pl + 1].w.saturating_sub(px0));
            let hc = h.min(f.pl[pl + 1].h.saturating_sub(py0));
            for y in 0..hc {
                f.pl[pl + 1].set_row(px0, py0 + y, &pred[y * w..y * w + wc]);
            }
            crate::av2_frame::write_recon_pad(pl + 1, px0, py0, &pred, w, h);
            mscore_chroma(pl + 1, px0, py0, w, h, &pred);
        }
        REF_CHROMA_SCORE.with(|s| {
            let (u, v, t) = s.get();
            s.set((u + ok_u as u32, v + ok_v as u32, t + 1));
        });
        if !(ok_u && ok_v) && std::env::var("CREFDBG").is_ok() {
            crate::dlog!(
                "CREFMISS px=({px0},{py0}) w={w} h={h} uvmode={uv_mode} cfl={cfl_mode} ang={} azU={} azV={} okU={} okV={}",
                if is_dir { p_angle } else { -1 }, az_u as u8, az_v as u8, ok_u as u8, ok_v as u8
            );
        }
        // Mark this chroma block's luma-equivalent region into the chroma decode grid (for later
        // chroma blocks' top-right / bottom-left availability). Done AFTER this block's own check.
        f.mark_coded_c(bx4, by4, bw4, bh4);
        // Mark the chroma SMOOTH flag (uv_mode 9/10/11) for later neighbours' edge filter.
        let is_smooth = (9..=11).contains(&uv_mode) as u8;
        let (cw4, ch4) = (w / 4, h / 4);
        let ciw4 = f.ciw4;
        for r in cby4..(cby4 + ch4).min(f.cih4) {
            for c in cbx4..(cbx4 + cw4).min(ciw4) {
                f.sm_c[r * ciw4 + c] = is_smooth;
            }
        }
    });
    crate::av2_frame::dbg_block_miss_c(px0, py0, w, h, "intra");
}

/// Intra-luma transform-type mapping (dav2d `md_idx2type[sz_ctx][y_mode][tx_idx]`).
#[rustfmt::skip]
static MD_IDX2TYPE: [[[u8; 7]; 13]; 3] = [
    [ // sz_ctx = 0
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, V_DCT, V_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_ADST, H_DCT, H_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, V_ADST, V_FLIPADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_ADST, H_DCT, H_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, V_DCT, V_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, FLIPADST_ADST, V_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
        [DCT_DCT, ADST_ADST, DCT_ADST, V_DCT, H_DCT, V_ADST, H_ADST],
    ],
    [ // sz_ctx = 1
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, ADST_FLIPADST, FLIPADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_ADST, FLIPADST_DCT, ADST_FLIPADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, FLIPADST_ADST, ADST_FLIPADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, FLIPADST_FLIPADST, ADST_FLIPADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, V_DCT, H_DCT, H_ADST],
    ],
    [ // sz_ctx = 2
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, FLIPADST_ADST, ADST_FLIPADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
        [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, V_DCT, H_DCT, V_ADST],
    ],
];

/// LONG-path transform-type table (dav2d `txtp_long_tbl[long_dct][w<h][short_idx]`).
#[rustfmt::skip]
static TXTP_LONG_TBL: [[[u8; 4]; 2]; 2] = [
    [[V_DCT, V_ADST, V_FLIPADST, IDTX_TT], [H_DCT, H_ADST, H_FLIPADST, IDTX_TT]],
    [[DCT_DCT, ADST_DCT, FLIPADST_DCT, H_DCT], [DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT]],
];

/// Inter transform-type inverse table (dav2d `txtp_inv_tbl[setidx][idx]`): maps the
/// decoded full-set index to a transform type. Row 1 (setidx=1) uses only 12 entries.
#[rustfmt::skip]
static TXTP_INV_TBL: [[u8; 16]; 2] = [
    [IDTX_TT, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST,
     DCT_DCT, ADST_DCT, DCT_ADST, FLIPADST_DCT, DCT_FLIPADST,
     ADST_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
    [IDTX_TT, V_DCT, H_DCT,
     DCT_DCT, ADST_DCT, DCT_ADST, FLIPADST_DCT, DCT_FLIPADST,
     ADST_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST, 0, 0, 0, 0],
];

/// y-mode index for `md_idx2type` from `y_mode_idx`: non-directional reorder (idx<5),
/// else the directional reorder via `midx`. (DC=0, SMOOTH=9, SMOOTH_V=10, SMOOTH_H=11,
/// PAETH=12; dir: D45=3, D67=8, V=1, D113=5, D135=4, D157=6, H=2, D203=7.)
const REORDERED_NONDIR_Y_MODE: [u8; 5] = [0, 9, 10, 11, 12];
const REORDERED_DIR_Y_MODE: [u8; 8] = [3, 8, 1, 5, 4, 6, 2, 7];

fn y_mode_from_idx(y_mode_idx: i32, midx: u8) -> u8 {
    if y_mode_idx < 5 {
        REORDERED_NONDIR_Y_MODE[y_mode_idx as usize]
    } else {
        REORDERED_DIR_Y_MODE[(midx / 7) as usize]
    }
}

/// Decode an intra-luma coefficient block's transform type + secondary transform,
/// then the coefficient tokens (dav2d `recon_tmpl.c` `decode_coefs` tail). Dispatches:
/// FSC → IDTX level loop; else txtp (`txtp_ext` → `DCT_DCT` for `tx_idx=0`) + stx
/// eligibility/decode, then `decode_coefs_dct_y`. `t_dim_min` is the TX min-dim log2
/// (TX_16X16=2). The non-`DCT_DCT` `md_idx2type` table and `stx_set` (when `stx>0`)
/// are asserted-out until a block exercises them — the verified path is DCT_DCT/no-stx.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs_y(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    cf: &mut [i32],
    eob: i32,
    fsc: bool,
    y_mode: u8,
    t_dim_min: usize,
    t_dim_ctx: usize,
    slw: usize,
    slh: usize,
    tx2dszctx: usize,
    scan: &[u16],
    tcq_enabled: bool,
    dc_sign_ctx: usize,
) -> (u8, u8, u8, u8) {
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_symbol_adapt4,
        rav1d_msac_decode_symbol_adapt8,
    };
    let sz_ctx_coef = t_dim_ctx.min(2);
    if fsc {
        // fsc/IDTX intra: identity transform in both dims, no STX.
        return (
            crate::av2_coef::decode_coefs_idtx_y(
                msac, &mut cdf.coef, cf, eob, tx2dszctx, sz_ctx_coef, slw, slh, scan,
            ),
            IDTX_TT,
            0,
            0,
        );
    }
    // txtp (dav2d order): a 64-core TX (sub == TX_32X32: 64x64/64x32/32x64) or a square
    // 32x32, or a dc-only block, all force DCT_DCT with no coded symbol. Else `t_dim->max >=
    // TX_32X32` takes the LONG path (txtp_long_tbl, long_dct forced when max == TX_64X64);
    // otherwise the ext_new_tx_set symbol → md_idx2type.
    let txtp = if slw.min(slh) >= 3 && slw.max(slh) == 4 {
        DCT_DCT // sub == TX_32X32
    } else if eob == 0 || (slw == 3 && slh == 3) {
        DCT_DCT // dc-only or square 32x32
    } else {
        let tmax = slw.max(slh);
        if tmax >= 3 {
            let long_dct =
                tmax >= 4 || rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.txtp_long32_dct[0]);
            let short_idx =
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.txtp_intra_short_1d[t_dim_min], 3)
                    as usize;
            TXTP_LONG_TBL[long_dct as usize][(slw < slh) as usize][short_idx]
        } else {
            let tx_idx =
                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.txtp_ext[t_dim_min], 6) as usize;
            let sz_ctx = (slw + slh) >> 1;
            MD_IDX2TYPE[sz_ctx][y_mode as usize][tx_idx]
        }
    };
    let tx_class = (txtp >> 3) & 3;
    // stx eligibility: DCT_DCT || ADST_ADST, eob>=1, mode != PAETH, eob < lim.
    let stx_eligible = eob >= 1
        && y_mode != PAETH_PRED
        && (txtp == DCT_DCT || txtp == ADST_ADST)
        && {
            let lim = if slw == 1 && slh == 1 && txtp == DCT_DCT {
                20 // 8x8 DCT_DCT
            } else if t_dim_min >= 1 {
                if txtp == DCT_DCT { 32 } else { 20 }
            } else {
                8
            };
            eob < lim
        };
    let mut stx_type = 0u8;
    let mut stx_set = 0u8;
    if stx_eligible && SEQ_TOOLS.with(|c| c.get().ist) {
        stx_type = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.stx[0][t_dim_min], 3) as u8;
        if stx_type > 0 {
            // Secondary-transform set. The set symbol selects which STX basis; the recon applies
            // the inverse secondary transform to the top-left coefficients BEFORE the primary itx.
            stx_set = if t_dim_min >= 1 && txtp == ADST_ADST {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.stx_set_adst, 3) as u8
            } else {
                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.stx_set, 6) as u8
            };
        }
    }
    // Coefficient decode — 2D class (tx_class 0) reuses the DCT/dc-only paths; the H/V
    // (identity-based) transform classes (2/3) use the transposed coefficient layout.
    // Coefficient region clamps to the 32x32 core for 64-core TXs (scan/levels/get_lo_ctx);
    // t_dim_ctx keeps the full TX context.
    let (clw, clh) = (slw.min(3), slh.min(3));
    let cf_ctx = if tx_class != 0 {
        crate::av2_coef::decode_coefs_hv_y(
            msac, &mut cdf.coef, cf, eob, tx_class as usize, tx2dszctx, t_dim_ctx, clw, clh, tcq_enabled, dc_sign_ctx,
        )
    } else if eob == 0 {
        crate::av2_coef::decode_coefs_dc_only_y(msac, &mut cdf.coef, cf, t_dim_ctx, dc_sign_ctx)
    } else {
        crate::av2_coef::decode_coefs_dct_y(
            msac, &mut cdf.coef, cf, eob, tx2dszctx, t_dim_ctx, clw, clh, scan, tcq_enabled, dc_sign_ctx,
        )
    };
    (cf_ctx, txtp, stx_type, stx_set)
}

/// FSC block-size group (dav2d `fsc_bsize_groups`), indexed by `BlockSize` in this
/// crate's `BLOCK_DIMENSIONS` order. Selects the `sz_ctx` second index of `fsc[][]`.
#[rustfmt::skip]
pub static FSC_BSIZE_GROUPS: [u8; 31] = [
    0,0,0,0,0,0, // 256x256..64x128
    0,0,0,0,0,    // 64x64..64x4
    0,5,5,4,4,    // 32x64, 32x32, 32x16, 32x8, 32x4
    0,5,4,3,3,    // 16x64, 16x32, 16x16, 16x8, 16x4
    0,4,3,2,1,    // 8x64, 8x32, 8x16, 8x8, 8x4
    0,4,3,1,0,    // 4x64, 4x32, 4x16, 4x8, 4x4
];

/// Default directional-mode reorder list (dav2d `default_mode_list_y`) — maps the
/// directional `y_mode_idx-5` to a `midx`. Used both to splat `midx` (so neighbours'
/// `mode_ctx` see a directional mode) and to derive the actual y-mode.
#[rustfmt::skip]
pub static DEFAULT_MODE_LIST_Y: [u8; 56] = [
    17, 45, 3, 10, 24, 31, 38, 52,
    15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54,
    16, 18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53,
    14, 20, 42, 48, 0, 6, 7, 13, 21, 27, 28, 34, 35, 41, 49, 55,
];

/// What `decode_b_luma` produces for the caller (coefficient decode + reconstruction).
/// One TX unit's parsed coefficients under a TX partition (dav per-tx read_luma_tx_cf).
pub struct TxUnitCf {
    pub ux4: usize,
    pub uy4: usize,
    pub slw: usize,
    pub slh: usize,
    pub cf: Vec<i32>,
    pub txtp: u8,
    pub eob: i32,
    pub stx: u8,
    pub all_zero: bool,
}

pub struct LeafInfo {
    pub intrabc: bool,
    pub y_mode_idx: i32,
    pub midx: u8,
    pub fsc: bool,
    pub mrl_index: u8,
    /// Multi-reference-line flag (dav2d `multi_mrl`) — needed by the frame-2 intra luma recon.
    pub multi_mrl: u8,
    /// intrabc block vector (y, x) in 1/8-pel = predictor(DRL) + delta (dav2d decode.c:1051).
    pub ibc_bv: (i32, i32),
    /// intrabc `morph_pred` flag (avm decodemv.c:1484): refine the copy with a BAWP linear model.
    pub ibc_morph: bool,
    pub all_zero: bool,
    /// End-of-block (`-1` when `all_zero`), and the decoded coefficients (raster).
    pub eob: i32,
    pub cf: Vec<i32>,
    /// Luma txtp — only meaningful for an intrabc leaf, whose chroma inherits it (intra=0).
    pub txtp: u8,
    /// Block-level `skip_txfm` — when set, dav2d skips read_coef_blocks entirely (NO luma or
    /// chroma coefs). The intrabc chroma decode in decode_leaf must be gated on `!skip`.
    pub skip: bool,
    /// intrabc luma secondary transform (STX/IST) type (0 = none). Decoded for a ≥16×16 DCT_DCT
    /// intrabc block with eob∈[3,32); applied to the dequantized luma coeffs before the primary itx.
    pub stx: u8,
    /// TX-partition units (empty = single TX == the block; recon loops these).
    pub units: Vec<TxUnitCf>,
}

/// Result of the general single-ref inter block decode (mode + MV syntax, pre-coefs).
/// dav2d `read_amvd` (decode.c:121): the adaptive-MVD residual — a joint symbol then a per-axis
/// magnitude index. Returns (mv_y, mv_x) magnitudes (signs handled by the caller).
pub fn read_amvd(msac: &mut crate::msac::MsacContext, m: &mut crate::cdf_av2::CdfModeContext) -> (i32, i32) {
    use crate::msac::{rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8};
    let joint = rav1d_msac_decode_symbol_adapt4(msac, &mut m.amvd_joint, 3);
    if joint == 0 {
        return (0, 0);
    }
    let axis = |msac: &mut crate::msac::MsacContext, cdf: &mut [u16; 8]| -> i32 {
        let s = rav1d_msac_decode_symbol_adapt8(msac, cdf, 7) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    };
    let my = if joint & 2 != 0 { axis(msac, &mut m.amvd_index[0]) } else { 0 };
    let mx = if joint & 1 != 0 { axis(msac, &mut m.amvd_index[1]) } else { 0 };
    (my, mx)
}

pub struct InterInfo {
    pub skip: bool,
    pub inter_mode: u8,
    pub motion_mode: u8,
    /// The value dav2d stores in the `mvprec` neighbour edge (`mvprec_def`: 1 default, 2 if the
    /// mvprec_rem symbol was decoded). Feeds later blocks' mvprec context.
    pub mvprec_def: u8,
    pub mv_y: i32,
    pub mv_x: i32,
    pub warp_ref_idx: usize,
    /// DRL index (the `n` the DRL loop stopped at) — selects the MV predictor from the mvstack.
    pub drl_idx: usize,
    /// Resolved MV precision (3..6) — feeds `mv_reduce_prec` in the MV finalization (brick B).
    pub mv_prec: i32,
    /// Adaptive-MVD flag — the finalization skips `reduce_prec` when set (dav2d decode.c:1112).
    pub amvd: bool,
    /// Warp-delta params (`b->matrix`) for MM_WARP_DELTA: `[n]` = signed delta × step, `[2]=-0x80`
    /// marks np==2. Used to reconstruct the warp matrix for the grid splat (brick B).
    pub warp_delta: [i32; 4],
    /// WARPMV blocks with a coded MV residual: the predictor is `get_warpmv_2d` at `mv_prec` (not
    /// 6) plus the residual (dav2d decode.c:1104-1115).
    pub warpmv_with_mvd: bool,
    /// Interpolation filter (0=REGULAR/1=SMOOTH/2=SHARP), applied to both H and V. Stage-D MC.
    pub filter: u8,
    /// BAWP flag (block adaptive weighted prediction): 0=off, 1=implicit, 2=explicit, 3=explicit
    /// +scale. dav2d's f2pred capture skips BAWP blocks (`!bawp0`), and the MC prediction for them
    /// is post-scaled/offset from neighbouring recon samples. Non-zero ⇒ not plain MC.
    pub bawp: u8,
    /// Chroma BAWP flag (dav2d `b->bawp[1]`, a SEPARATE bit from `bawp[0]`): 1 ⇒ also morph the
    /// chroma prediction (reusing the luma alpha). Can be 0 even when `bawp` (luma) is set.
    pub bawp_chroma: u8,
    /// The block's single reference index (`single_ref` selection, 0..n_ref). The MC fetches the
    /// picture from `REF_PICS[refidx[ref0]]` — a B-frame block may pick ref 0 OR ref 1.
    pub ref0: u8,
    /// The block's second reference (compound/skip_mode pair), raw list index or -1 = single-ref.
    /// A compound block's MC must blend REF_PICS[refidx[ref0]] and REF_PICS[refidx[ref1]].
    pub ref1: i8,
    /// Compound second-arm DRL index (dav b->drl_idx[1]); == drl_idx for single-ref.
    pub drl_idx1: usize,
    /// Compound second-arm MV residual (mvs[1] from the compound parse).
    pub mv_y1: i32,
    pub mv_x1: i32,
    /// Compound weighted-prediction index (8 = equal); skip_mode takes it from the DRL candidate.
    pub cwp: i8,
    /// Compound type (0=NONE/single, 1=AVG, 2=WEDGE, 3=SEG).
    pub comp_type: u8,
    /// TIP block (dav `b->ref.ref[0] == TIP_FRAME`): refmvs/splat/bank ref = 7, no temporal write.
    pub is_tip: bool,
    /// Compound refine_mv (0=off, 1=explicit, 2=implicit): with comp_type AVG the recon runs the
    /// DMVR/opfl refinement (`opfl_pred`).
    pub refine_mv: u8,
    /// WEDGE parameters (comp_type 2).
    pub wedge_idx: i32,
    pub wedge_sign: bool,
    /// Inter-intra blend mode (dav2d `b->interintra_mode` when MM_INTERINTRA or `warp_ii`):
    /// -1 = off; 0=DC, 1=V, 2=H, 3=SMOOTH. Recon blends an intra predictor over the inter/warp
    /// prediction with the II mask (dav recon iiblend).
    pub ii_mode: i8,
    /// SEG mask sign (comp_type 3).
    pub mask_sign: bool,
}

/// Decode a single **single-ref, non-compound** inter block's mode + MV syntax (dav2d
/// `decode_b`, the `!b->intra` path from skip_txfm through the warp-delta params — i.e.
/// everything before `read_coef_blocks`). All contexts are COMPUTED from the live neighbour
/// arrays. Splats the inter neighbour state (intra=0, motion_mode, ref0, mvprec, mode, skip_txfm).
/// The caller decodes `is_inter` before this and the coefficients after (using the returned skip).
/// Covers this clip: 1 ref frame (ref forced 0), no TIP/compound; extend for those as needed.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_inter(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    a: &mut BlockNbCtx,
    l: &mut BlockNbCtx,
    bs: usize,
    bx4: usize,
    by4: usize,
    have_left: bool,
    have_top: bool,
    have_top_in_sb: bool,
    is_sb_boundary: bool,
    row_end: usize,
    col_end: usize,
    w4: usize,
    h4: usize,
    frame_mv_precision: i32,
    force_integer_mv: bool,
    max_drl_bits: usize,
    motion_modes: u32,
    mvd_sign_derive: bool,
    scc: bool,
    // cbs == lbs (the block carries its own chroma, i.e. `cbs == bs` / `!forced_inter`). Gates the
    // per-block is_tip flag (dav2d decode.c:2446) — a sub-8 / chroma-shared leaf codes no is_tip.
    cbs_eq_lbs: bool,
    seg_globalmv_skip: bool,
    // Seq `adaptive_mvd`: a NEWMV block codes an `amvd` flag that switches its MV to read_amvd.
    adaptive_mvd: bool,
    // When true (block is the SB's first leaf), decode the once-per-SB filter params (gdf,
    // cdef, ccso) after skip_txfm (dav2d decode.c:1810/1842/1905). `left_cdef`/`left_ccso` are
    // the cross-SB filter neighbour state (reset per SB-row); top cdef is always -1 for 64px SBs.
    decode_filters: bool,
    left_cdef: &mut i8,
    left_ccso: &mut [u8; 3],
    // Frame `bawp` (block-adaptive weighted prediction) enabled + block has chroma. For a
    // non-warp mode (inter_mode <= NEWMV, !GLOBALMV) dav2d decodes bawp before the DRL.
    bawp_enabled: bool,
    has_chroma: bool,
    // Frame cdef `n_strengths` (obu.c:1635): controls the once-per-SB cdef_idx read. ==2 → the
    // "not v=0" branch is a bare v=1 with no symbol; ≥3 reads a cdef_idx symbol. Per-frame.
    cdef_n_strengths: usize,
    // Seq `six_param_warp_delta` (obu.c): when set, a WARPNEWMV/MM_WARP_DELTA block with
    // warp_ref_idx==1 codes 4 warp-delta params (np=4) instead of the wri==0 np=2 set.
    six_param_warp_delta: bool,
    // skip_mode (decoded by the caller before is_inter). A skip_mode block is compound-implied:
    // it codes NO is_tip/is_comp/ref/inter_mode/mvd — only a `skip_mode_drl_idx` DRL loop — then
    // derives its refs (skip_mode_refs / neighbour override), inter_mode=NEARMV_NEARMV, no subpel.
    skip_mode: bool,
) -> InterInfo {
    if std::env::var("MPREDRL").is_ok() {
        crate::dlog!("[MBEG] mi=({bx4},{by4}) rng={}", msac.rng);
    }
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bool_bypass, rav1d_msac_decode_symbol_adapt4,
        rav1d_msac_decode_symbol_adapt8,
    };
    const NEARMV: u8 = 13;
    const GLOBALMV: u8 = 14;
    const NEWMV: u8 = 15;
    const WARPMV: u8 = 16;
    const WARPNEWMV: u8 = 17;
    const MM_TRANSLATION: u8 = 0;
    const MM_WARP_CAUSAL: u8 = 2;
    const MM_WARP_DELTA: u8 = 3;
    const MM_WARP_EXTEND: u8 = 4;
    let bd = crate::av2_decode::BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    // Edge clamp for the nb neighbour scheme (dav decode.c:1696 convention, same as the
    // decode_b_luma fix): the CLAMPED dims gate the above-right / bottom-left slots
    // (`bw4c == bw4` fails when the block spills off-frame → slot skipped, matching avm's
    // NULL above_right_mbmi), while the FULL dims stay the reference for the gate.
    let (bw4c, bh4c) = (bw4.min(col_end.saturating_sub(bx4)), bh4.min(row_end.saturating_sub(by4)));

    // --- skip_txfm (nx neighbour scheme) ---
    let (nx, nctx) = nx_setup(have_left, have_top, bx4, by4, bw4, bh4, row_end, col_end);
    // dav's `idx` from the NX setup (decode.c:1640) — the has_luma-scoped nb idx (1701) closes
    // before the inter branch, so is_comp ctx, the single/comp ref-histogram gates + cnt_rem, and
    // comptype_ctx are all gated by THIS count. (The fallback slots hold valid-looking offsets even
    // when idx==0 — a frame-corner block — so `off >= 0` is NOT a validity test.)
    let nb_idx = nctx;
    // skip_txfm ctx adds skip_mode*3 (dav2d decode.c:1747) — a skip_mode block picks the +3 CDF cell.
    let sctx = get_skip_txfm_ctx(a, l, nx, skip_mode as u8);
    let skip = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.skip_txfm[sctx]);
    let dbg00 = std::env::var("DBG00").is_ok() && ((bx4 == 0 && by4 == 0) || (bx4 == 0 && by4 == 1) || (bx4 == 2 && by4 == 2) || (bx4 == 2 && by4 == 1) || (bx4 == 1 && by4 == 2) || (bx4 == 96 && by4 == 32) || (bx4 == 12 && by4 == 14) || (bx4 == 14 && by4 == 12) || (bx4 == 20 && by4 == 0) || (bx4 == 24 && by4 == 2) || (bx4 == 30 && by4 == 14) || (bx4 == 30 && by4 == 12) || (bx4 == 76 && by4 == 4) || (bx4 == 0 && by4 == 2));
    if dbg00 { crate::dlog!("[D0] === block ({bx4},{by4}) bs={bs} skip_mode={skip_mode} r_enter={} ==", msac.rng); }
    let dv2 = std::env::var("DV2").is_ok() && ((bx4 == 4 && by4 == 0) || (bx4 == 0 && by4 == 24) || (bx4 == 32 && by4 == 4) || (bx4 == 96 && by4 == 44) || (bx4 == 2 && by4 == 4) || (bx4 == 96 && by4 == 20) || (bx4 == 64 && by4 == 0));
    // The frame's mv_precision is per-FRAME (obu.c:1392) — read from HDR_TOOL_CFG, not the
    // hardcoded caller constant (2). v432 f2=2 (unchanged), v320 f2=3. Seeds mv_prec + rem index.
    let frame_mv_precision = HDR_TOOL_CFG.with(|c| c.get().mv_precision) as i32;
    if dbg00 || dv2 { crate::dlog!("[D0] skip ctx={sctx} v={} r={} (dav skip r=36352, then ccso u r=41152 v r=48960)", skip as u8, msac.rng); }
    let dbg04 = false;

    // --- once-per-SB filter params: gdf + cdef (dav2d decode.c:1810/1842) ---
    // gdf: once per 128px (`(bx|by)&31==0`); cdef: once per 64px SB, ctx from the left SB's cdef
    // index (top is always -1 for 64px SBs — verified against the oracle for all 28 SBs). This
    // inter frame does NOT code ccso (the pre-ccso build was bit-exact over the whole first SB).
    // Once-per-SB filter params (gdf, cdef_idx, ccso×3) — mirrors decode_b_luma's block exactly
    // (dav2d order + gates). Each tool is gated on its frame-header enable (HDR_TOOL_CFG): golden's
    // frame-2 enabled gdf/cdef but not ccso, so the old inter path hardcoded "no ccso + always gdf";
    // a stream like v432 disables gdf (post-gdf[0]) and ENABLES per-plane ccso, so all three must
    // be gated the same way as the keyframe path. top cdef is -1 for 64px SBs (SB-row 0).
    let tool_cfg = HDR_TOOL_CFG.with(|c| c.get());
    if !decode_filters {
        // Non-SB-first leaves can still trigger a per-64-cell cdef_idx read (128px SBs).
        read_cdef_per64(msac, cdf, bx4, by4, bw4, bh4, skip);
    }
    if decode_filters {
        // TILE-ADAPTIVE units (avm): gdf block 128px→64px, ccso unit 256px→SB.
        let (ccso_u4, gdf_bs4) = FILTER_UNITS.with(|c| c.get());
        if dbg00 || dv2 { crate::dlog!("[D0] gdf-gate tool_cfg.gdf={} (dav: gdf disabled, NO symbol for frame 2)", tool_cfg.gdf); }
        if tool_cfg.gdf && (bx4 | by4) & (gdf_bs4 - 1) == 0 {
            let gdf = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.gdf);
            if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
                crate::av2_frame::FRAME.with(|f| f.borrow_mut().set_gdf_blk(bx4, by4, gdf));
            }
        }
        // Per-64-cell cdef_idx (dav order: gdf → cdef → ccso within the leaf).
        read_cdef_per64(msac, cdf, bx4, by4, bw4, bh4, skip);
        let _ = cdef_n_strengths;
        if tool_cfg.ccso && (bx4 | by4) & (ccso_u4 - 1) == 0 {
            // dav2d decode.c:1888 — per-SB ccso flag coded ONLY for planes that are enabled AND
            // NOT sb_reuse. An inter frame's sb_reuse plane INHERITS its per-SB flags from the
            // previous frame's ccsomap (no symbol); the filter pass reads PREV_CCSO_MAP for those.
            let (ccso_en, ccso_reuse): ([bool; 3], [bool; 3]) = crate::av2_frame::CCSO_CFG.with(|c| {
                let cfg = c.borrow();
                (
                    std::array::from_fn(|p| cfg.p.get(p).map_or(false, |pc| pc.enabled)),
                    std::array::from_fn(|p| cfg.p.get(p).map_or(false, |pc| pc.sb_reuse)),
                )
            });
            for p in 0..3 {
                if !ccso_en[p] || ccso_reuse[p] {
                    continue;
                }
                let ctx = if bx4 as i32 - ccso_u4 as i32 >= TILE_B.with(|t| t.get().0) as i32 {
                    left_ccso[p] as usize * 2
                } else {
                    0
                };
                let (pre_r, pre_d, pre_cdf) = (msac.rng, msac.dif, cdf.m.ccso[p][ctx][0]);
                left_ccso[p] = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.ccso[p][ctx]) as u8;
                if dbg00 || dv2 { crate::dlog!("[D0] ccso p={p} ctx={ctx} v={} r={} | pre_r={pre_r} pre_d={pre_d:x} pre_cdf={pre_cdf}", left_ccso[p], msac.rng); }
                if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
                    let on = left_ccso[p] != 0;
                    crate::av2_frame::FRAME.with(|f| f.borrow_mut().set_ccso_blk(bx4, by4, p, on));
                }
            }
        }
        // --- delta-q (dav2d decode.c:1941): per-SB running qindex, after ccso, before the mode.
        read_delta_q(msac, cdf, bs, skip, bx4, by4);
    }

    // --- is_tip: per-block TIP flag (dav2d decode.c:2446). Coded when the frame is a TIP frame,
    // the block carries its own chroma (cbs==lbs), and imax(bw4,bh4)>=2. ctx from neighbours'
    // ref[0]==TIP_FRAME (0 here — no TIP block tracked). Always 0 for v320 but consumes msac state.
    let is_tip = {
        let tip_frame_mode = HDR_TOOL_CFG.with(|c| c.get().tip_frame_mode);
        // avm is_tip_allowed_bsize (blockd.h:3296): chroma-ref + bsize==base (cbs_eq_lbs)
        // AND min luma dim >= 8px — the min-dim term is what the 420 cbs==lbs proxy hid
        // (at 4:2:2 an own-chroma 16x4 exists and still codes NO tip flag).
        if !skip_mode && tip_frame_mode != 0 && cbs_eq_lbs && bw4.min(bh4) >= 2 {
            // ctx (decode.c:2449): count of NX neighbours whose ref[0]==TIP_FRAME (stored 7+1=8),
            // gated by the NX idx like is_comp.
            let tip_of = |k: usize| -> usize {
                if nb_idx <= k { return 0; }
                let (sel, off) = nx[k];
                let d = if sel { &*l } else { &*a };
                (d.ref0[off as usize] == 8) as usize
            };
            let ctx = tip_of(0) + tip_of(1);
            let v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.tip[ctx]);
            if dbg00 || dv2 { crate::dlog!("[D0] tip ctx={ctx} v={} r={}", v as u8, msac.rng); }
            v
        } else {
            false
        }
    };

    // --- skip_mode branch (dav2d decode.c:2456/2479). A skip_mode block is compound-implied
    // (is_comp=1, no is_tip/is_comp/ref/inter_mode/mvd symbols): it codes ONLY a `skip_mode_drl_idx`
    // DRL loop, then derives its refs (skip_mode_refs / neighbour override) with inter_mode
    // NEARMV_NEARMV and has_subpel_filter=0. No more symbols follow, so return here. ---
    if skip_mode {
        let mut drl = 0usize;
        let mut ctx = 0usize;
        while drl < max_drl_bits {
            if !crate::av2_recon::work_tick("av2_recon:7893") { break; }
            if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.skip_mode_drl_idx[ctx]) {
                break;
            }
            drl += 1;
            if ctx < 2 {
                ctx += 1;
            }
        }
        if dbg00 || dv2 { crate::dlog!("[D0] skip_mode drl={drl} r={}", msac.rng); }
        if std::env::var("MSKM").is_ok() { crate::dlog!("[MSKM] mi=({bx4},{by4}) drl={drl} fn={}", crate::av2_frame::DECODE_FRAME_N.with(|c| c.get())); }
        // Ref pair derivation (dav2d decode.c:2494): default = f->skip_mode_refs (frame-level,
        // decode.c:5643: ref[0]=0, ref[1]=(skip_mode_enabled && n_ref>1 && |absd[0]-absd[1]|<=1)),
        // then the FIRST informative NX neighbour overrides: TIP → the tip source pair (not yet
        // tracked); compound → its pair; single inter → keep default (break).
        let (mut sm_ref0, mut sm_ref1): (u8, i8) = {
            let (_rd, absd, _ffr) = CUR_REFDIST.with(|c| c.get());
            let n_ref = CUR_FRAME_REFIDX.with(|c| c.get()).0;
            (0, (n_ref > 1 && (absd[0] - absd[1]).abs() <= 1) as i8)
        };
        for k in 0..2usize {
            if nb_idx <= k { break; }
            let (sel, off) = nx[k];
            let d = if sel { &*l } else { &*a };
            let o = off as usize;
            if d.ref0[o] == 8 {
                // TIP neighbour (dav decode.c:2498): pair = (min, max) of the TIP source refs.
                let tr = crate::av2_refmvs::TMVS.with(|c| c.borrow().tip_ref);
                sm_ref0 = tr.0.min(tr.1);
                sm_ref1 = tr.0.max(tr.1) as i8;
                break;
            } else if d.ref1[o] != -1 {
                sm_ref0 = d.ref0[o] - 1;
                sm_ref1 = d.ref1[o];
                break;
            } else if d.ref0[o] != 0 {
                break;
            }
        }
        // Splat this skip_mode block's neighbour state (dav2d set_ctx). skip_mode=1 is the key one
        // (feeds neighbours' skip_mode + skip_txfm ctx); ref pair from the derivation above.
        splat_inter_nb(a, l, bx4, by4, bw4, bh4, 0, MM_TRANSLATION, sm_ref0 + 1, 1, NEARMV, 1, sm_ref1);
        splat_nb(&mut a.skip_txfm, &mut l.skip_txfm, bx4, by4, bw4, bh4, skip as u8);
        // skip_mode has has_subpel_filter=0 → dav2d sets b->filter = FILTER_8TAP_SHARP (2) (decode.c:3274).
        splat_nb(&mut a.filter, &mut l.filter, bx4, by4, bw4, bh4, 2);
        // skip_mode is compound with comp_type = COMP_INTER_AVG (decode.c:2508).
        splat_nb(&mut a.comp_type, &mut l.comp_type, bx4, by4, bw4, bh4, 1);
        splat_nb(&mut a.amvd, &mut l.amvd, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.midx, &mut l.midx, bx4, by4, bw4, bh4, 0xff);
        splat_nb(&mut a.fsc, &mut l.fsc, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.mrl, &mut l.mrl, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.multi_mrl, &mut l.multi_mrl, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.intrabc, &mut l.intrabc, bx4, by4, bw4, bh4, 0);
        return InterInfo {
            skip, inter_mode: NEARMV, motion_mode: MM_TRANSLATION, mvprec_def: 1,
            mv_y: 0, mv_x: 0, warp_ref_idx: 0, drl_idx: drl, mv_prec: frame_mv_precision,
            amvd: false, warp_delta: [0, 0, 0, 0], warpmv_with_mvd: false, filter: 2,
            bawp: 0, bawp_chroma: 0, ref0: sm_ref0, ref1: sm_ref1,
            drl_idx1: drl, mv_y1: 0, mv_x1: 0, cwp: 8, comp_type: 1, wedge_idx: -1, wedge_sign: false, mask_sign: false, is_tip: false, refine_mv: 0, ii_mode: -1,
        };
    }

    // --- is_comp: compound-prediction flag (dav2d decode.c:2461). Coded between is_tip and the
    // ref decode when switchable_comp_refs is on, the block is not TIP/globalmv-skip, and
    // bw4*bh4>=4 (sub-8 blocks are single-ref only). ctx from neighbour ref directions
    // (get_comp_ctx / refdir_with_intra). This clip is single-ref, so we assert !is_comp. ---
    let (n_ref_frames, _cur_refidx) = CUR_FRAME_REFIDX.with(|c| c.get());
    let is_comp = {
        let scr = HDR_TOOL_CFG.with(|c| c.get().switchable_comp_refs);
        if scr && !is_tip && !seg_globalmv_skip && bw4 * bh4 >= 4 {
            let ctx = get_comp_ctx(a, l, nx, nb_idx, &CUR_REFDIR.with(|c| c.get()));
            let v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.comp[ctx]);
            if dbg00 || dv2 { crate::dlog!("[D0] is_comp ctx={ctx} v={} r={}", v as u8, msac.rng); }
            v
        } else {
            false
        }
    };
    // ============ COMPOUND branch (dav2d decode.c:2513-2904) ============
    // A compound block decodes: ref pair (comp0_ref/comp1_ref iterative bits) → compound
    // inter_mode (sameref / joint / comp_mode + opfl) → amvd → jmvd_scale → NEWMV_NEWMV
    // warp_causal → compound DRL (drl_idx[ctx][compref_ctx]) → mv precision → per-ref MV
    // residuals + sign-derive (+ JOINT mv projection) → refine_mv → comp_type (avg/wedge/seg)
    // → cwp_idx → subpel filter (common tail, comp ctx +4). Returns early like skip_mode.
    if is_comp {
        const C_NEARMV_NEARMV: u8 = 18;
        const C_NEARMV_NEWMV: u8 = 19;
        const C_GLOBALMV_GLOBALMV: u8 = 21;
        const C_NEWMV_NEWMV: u8 = 22;
        const C_JOINT_NEWMV: u8 = 23;
        const C_OPFL_NEARMV_NEARMV: u8 = 24;
        const C_OPFL_JOINT_NEWMV: u8 = 28;
        let (masked_compound, num_same_ref_comp, seq_cwp, _seq_avg_cdf, _seq_mv_traj) = SEQ_COMP.with(|c| c.get());
        let seq_refine_mv = SEQ_TIP.with(|c| c.get()).3;
        let opfl_refine_type = HDR_TOOL_CFG.with(|c| c.get().opfl_refine_type);
        let (refdist, absrefdist, ffr) = CUR_REFDIST.with(|c| c.get());
        let refdir_wi = CUR_REFDIR.with(|c| c.get());
        // f->refdir[i] (list-indexed, 0/1) = refdir_with_intra[1+i].
        let f_refdir = |i: i32| refdir_wi[(i + 1) as usize] as u8;

        // --- ref pair (decode.c:2514) ---
        let n_refs = n_ref_frames as i32;
        let mut refp = [0i32; 2];
        if n_refs > 1 {
            let same_refs = num_same_ref_comp as i32;
            let mut n = 0usize;
            let mut cnt = [0i32; 9];
            // NB-idx-gated histogram (decode.c:2519), reading the NX slots.
            for (k, &(sel, off)) in [nx[0], nx[1]].iter().enumerate() {
                if nb_idx <= k { break; }
                let d = if sel { &*l } else { &*a };
                cnt[d.ref0[off as usize] as usize] += 1;
                cnt[(d.ref1[off as usize] + 1) as usize] += 1;
            }
            let mut cnt_rem = nb_idx as i32 * 2 - cnt[0] - cnt[8];
            let mut maybe_same_ref = (same_refs != 0) as i32;
            let mut dir = 0u8;
            let mut i = 0i32;
            while i < n_refs + n as i32 - 2 + maybe_same_ref {
                if !crate::av2_recon::work_tick("av2_recon:8011") { break; }
                let cnt_cur = cnt[(i + 1) as usize];
                cnt_rem -= cnt_cur;
                let bit = if n == 0 && (i == 2 || (i >= n_refs - 2 && i + 1 >= same_refs)) {
                    true
                } else {
                    let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
                    if n == 0 {
                        rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.comp0_ref[ctx][i as usize])
                    } else {
                        rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.comp1_ref[ctx][(dir ^ f_refdir(i)) as usize][i as usize])
                    }
                };
                if bit {
                    refp[n] = i;
                    n += 1;
                    if n == 2 { break; }
                    dir = f_refdir(i);
                }
                if maybe_same_ref != 0 {
                    maybe_same_ref = (!bit && i + 1 < same_refs) as i32;
                    if bit { i -= 1; cnt_rem += cnt_cur; }
                }
                i += 1;
            }
            if n < 2 {
                refp[1] = n_refs - 1;
                if n == 0 { refp[0] = n_refs - 1 - ((same_refs < n_refs) as i32); }
            }
        }
        let (refp0, refp1) = (refp[0], refp[1]);
        if dbg00 || dv2 { crate::dlog!("[D0] comp ref=({refp0},{refp1}) r={}", msac.rng); }

        // --- compound DRL context (decode.c:2569, computed before the mode) ---
        let have_top_right = bx4 + bw4 <= col_end;
        let have_bottom_left = by4 + bh4 <= row_end;
        let tip_pair = crate::av2_refmvs::TMVS.with(|c| {
            let t = c.borrow();
            (t.tip_ref.0 as i8, t.tip_ref.1 as i8)
        });
        let comp_ctx = get_compref_ctx(a, l, by4, bx4, have_top, have_left, have_top_right,
                                       have_bottom_left, bw4, bh4, refp0 as u8, refp1 as i8,
                                       tip_pair);

        // --- compound inter mode (decode.c:2573) ---
        if dbg00 || dv2 { crate::dlog!("[D0] pre-joint jcell={:?} rng={} dif={:x}", cdf.m.comp_mode_joint[0], msac.rng, msac.dif); }
        let mut inter_mode: u8;
        if refp0 == refp1 {
            inter_mode = C_NEARMV_NEARMV
                + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.comp_mode_sameref[comp_ctx], 3) as u8;
            inter_mode += (inter_mode > C_NEARMV_NEWMV) as u8; // skip NEWMV_NEARMV
        } else {
            let joint_ctx = (refdist[refp0 as usize] != -refdist[refp1 as usize]) as usize;
            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.comp_mode_joint[joint_ctx]) {
                inter_mode = C_JOINT_NEWMV;
            } else {
                inter_mode = C_NEARMV_NEARMV
                    + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.comp_mode[comp_ctx], 4) as u8;
            }
        }
        if opfl_refine_type == 1 && inter_mode != C_GLOBALMV_GLOBALMV && bw4.min(bh4) >= 2
            && f_refdir(refp0) != f_refdir(refp1)
        {
            let ctx = (inter_mode > C_NEARMV_NEARMV) as usize;
            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.opfl[ctx]) {
                inter_mode += 6 - (inter_mode >= C_GLOBALMV_GLOBALMV) as u8;
            }
        }
        if dbg00 || dv2 { crate::dlog!("[D0] comp_inter_mode cctx={comp_ctx} mode={inter_mode} r={} jcell={:?}", msac.rng, cdf.m.comp_mode_joint[0]); }

        // --- amvd (decode.c:2612) ---
        const C_NEWMV_MASK: u32 = (1 << 19) | (1 << 20) | (1 << 22) | (1 << 23)
            | (1 << 25) | (1 << 26) | (1 << 27) | (1 << 28);
        let is_newmv_mode = ((1u32 << inter_mode) & C_NEWMV_MASK) != 0;
        let mut amvd = false;
        if adaptive_mvd && is_newmv_mode {
            // amvd_mode_context, indexed mode - NEARMV_NEWMV (decode.c:2613).
            const AMVD_MODE_CTX: [u8; 10] = [0, 1, 0, 7, 5, 0, 2, 3, 8, 6];
            let mode_ctx = AMVD_MODE_CTX[(inter_mode - C_NEARMV_NEWMV) as usize] as usize;
            let mut ctx = 0usize;
            for &(sel, off) in &[nx[0], nx[1]] {
                if off < 0 { continue; }
                let d: &BlockNbCtx = if sel { l } else { a };
                let o = off as usize;
                ctx += ((d.ref0[o] as i32 - 1) == refp0 && d.amvd[o] != 0) as usize;
            }
            amvd = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.amvd[mode_ctx][ctx]);
        }

        // --- jmvd scale mode (decode.c:2633) ---
        let mut jmvd_scale_mode = 0u32;
        if inter_mode == C_JOINT_NEWMV || inter_mode == C_OPFL_JOINT_NEWMV {
            jmvd_scale_mode = if amvd {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.jmvd_amvd_scale_mode, 2) as u32
            } else {
                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.jmvd_scale_mode, 4) as u32
            };
        }

        // --- NEWMV_NEWMV compound warp_causal (decode.c:2644) ---
        let mut motion_mode = MM_TRANSLATION;
        if inter_mode == C_NEWMV_NEWMV && bw4.min(bh4) > 1 && !force_integer_mv
            && refp0 != refp1 && opfl_refine_type != 2 && (motion_modes & (1 << 2)) != 0
        {
            // match_refs (decode.c:2653): any left/above edge cell whose ref[0] or ref[1] == r.
            // Cleared cells (ref0 stored 0, ref1 -1) never match a real r >= 0.
            fn m_ref(d: &BlockNbCtx, i: usize, r: i32) -> bool {
                (d.ref0[i] != 0 && (d.ref0[i] as i32 - 1) == r) || (d.ref1[i] as i32) == r
            }
            let match_refs = |r: i32, a: &BlockNbCtx, l: &BlockNbCtx| -> bool {
                m_ref(l, by4, r)
                    || (by4 + bh4 <= row_end && m_ref(l, by4 + bh4 - 1, r))
                    || (if is_sb_boundary {
                        m_ref(a, bx4 & !1, r)
                            || (((bx4 + bw4 - 2) & !1) < col_end && m_ref(a, (bx4 + bw4 - 2) & !1, r))
                    } else {
                        m_ref(a, bx4, r)
                            || (bx4 + bw4 <= col_end && m_ref(a, bx4 + bw4 - 1, r))
                    })
            };
            if match_refs(refp0, a, l) && match_refs(refp1, a, l) {
                let nbw = nb_setup(have_left, have_top_in_sb, bx4, by4, bw4c, bh4c, bw4, bh4);
                let mm_of = |sel: (bool, i32), a: &BlockNbCtx, l: &BlockNbCtx| -> u8 {
                    if sel.1 < 0 { return MM_TRANSLATION; }
                    (if sel.0 { l } else { a }).motion_mode[sel.1 as usize]
                };
                let x1 = mm_of(nbw[0], a, l);
                let x2 = mm_of(nbw[1], a, l);
                let cs_ctx = ((x1 >= MM_WARP_CAUSAL || x2 >= MM_WARP_CAUSAL) as usize)
                    + (x1 == MM_WARP_CAUSAL) as usize + (x2 == MM_WARP_CAUSAL) as usize;
                if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_causal[cs_ctx]) {
                    motion_mode = MM_WARP_CAUSAL;
                }
            }
        }

        // --- compound DRL (decode.c:2684) ---
        let mut drl_idx = [0usize; 2];
        if inter_mode != C_GLOBALMV_GLOBALMV {
            let n_drls = 1 + (inter_mode <= C_NEARMV_NEWMV) as usize;
            let mut n = 0usize;
            let mut ctx = 0usize;
            for r in 0..n_drls {
                while n < max_drl_bits {
                    if !crate::av2_recon::work_tick("av2_recon:8154") { break; }
                    if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.drl_idx[ctx][comp_ctx]) {
                        break;
                    }
                    n += 1;
                    if ctx < 2 { ctx += 1; }
                }
                drl_idx[r] = n;
                n = if inter_mode == C_NEARMV_NEARMV && refp0 == refp1 {
                    drl_idx[0] + (drl_idx[0] < max_drl_bits) as usize
                } else {
                    0
                };
                ctx = n.min(2);
            }
            if n_drls == 1 { drl_idx[1] = drl_idx[0]; }
            if dbg00 || dv2 { crate::dlog!("[D0] comp drl=({},{}) r={}", drl_idx[0], drl_idx[1], msac.rng); }
        }

        // --- mv precision (decode.c:2708) ---
        let mv_prec_tbl: [[i32; 3]; 2] = [[3, 1, 0], [4, 3, 1]];
        let mut mv_prec = 3 + frame_mv_precision;
        let mut mvprec_def: u8 = 1;
        if mv_prec > 3 && !amvd && is_newmv_mode {
            let nbp = nb_setup(have_left, have_top_in_sb, bx4, by4, bw4c, bh4c, bw4, bh4);
            let mvp1 = nb_mvprec(a, l, nbp[0]);
            let mvp2 = nb_mvprec(a, l, nbp[1]);
            let ctx1 = (mvp1 & 1) as usize + (mvp2 & 1) as usize;
            if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.mvprec_def[ctx1]) {
                let ctx2 = ((mvp1 | mvp2) >> 1) as usize;
                let idx = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.mvprec_rem[ctx2][(mv_prec - 4) as usize], 2) as usize;
                mv_prec = mv_prec_tbl[(mv_prec == 6) as usize][idx];
                mvprec_def = 2;
            }
        }

        // --- per-ref MV residuals + sign-derive + JOINT projection (decode.c:2731) ---
        // PRED_MODES[mode-18][n]: which of the two MVs is NEWMV(15)-coded (tables.c:347).
        const PRED_MODES: [[u8; 2]; 11] = [
            [13, 13], [13, 15], [15, 13], [14, 14], [15, 15], [15, 15],
            [13, 13], [13, 15], [15, 13], [15, 15], [15, 15],
        ];
        let mut mvs = [(0i32, 0i32); 2];
        let mut mv_prec_out = mv_prec;
        if inter_mode != C_GLOBALMV_GLOBALMV {
            let mut start = 0usize;
            let mut end = 2usize;
            let mut rdist = [0i32; 2];
            if inter_mode == C_JOINT_NEWMV || inter_mode == C_OPFL_JOINT_NEWMV {
                rdist[0] = absrefdist[refp0 as usize];
                rdist[1] = absrefdist[refp1 as usize];
                start = (rdist[0] < rdist[1]) as usize;
                if f_refdir(refp0) ^ f_refdir(refp1) != 0 { rdist[1] = -rdist[1]; }
                end = start + 1;
            }
            let midx = (inter_mode - C_NEARMV_NEARMV) as usize;
            let mut sum_mvd = 0i32;
            let mut nnzc = 0i32;
            for n in start..end {
                if PRED_MODES[midx][n] != 15 { continue; }
                if amvd {
                    let (my, mx) = read_amvd(msac, &mut cdf.m);
                    mvs[n] = (my, mx);
                } else {
                    let mut st = cdf.mv.shell_tip;
                    let (my, mx) = read_mv_residual(msac, &mut cdf.mv, &mut st, mv_prec);
                    cdf.mv.shell_tip = st;
                    mvs[n] = (my, mx);
                    sum_mvd += my + mx;
                    nnzc += (my != 0) as i32 + (mx != 0) as i32;
                }
            }
            if inter_mode != C_NEARMV_NEARMV && inter_mode != C_OPFL_NEARMV_NEARMV {
                const BIDIR_NEWMV_MASK: u32 = (1 << 22) | (1 << 27) | (1 << 23) | (1 << 28);
                if !mvd_sign_derive || drl_idx[0] != 0 || drl_idx[1] != 0
                    || nnzc < 3 * (end - start) as i32 - 2
                    || scc || frame_mv_precision == 3 || mv_prec >= 5
                    || ((1u32 << inter_mode) & BIDIR_NEWMV_MASK) == 0
                    || motion_mode != MM_TRANSLATION
                {
                    nnzc = 5; // sign-derive never triggers
                }
                sum_mvd >>= 6 - mv_prec;
                let mut nnzc2 = 0i32;
                for n in start..end {
                    if PRED_MODES[midx][n] != 15 { continue; }
                    if mvs[n].0 != 0 {
                        nnzc2 += 1;
                        let s = if nnzc2 == nnzc { sum_mvd & 1 } else { rav1d_msac_decode_bool_bypass(msac) as i32 };
                        if s != 0 { mvs[n].0 = -mvs[n].0; }
                    }
                    if mvs[n].1 != 0 {
                        nnzc2 += 1;
                        let s = if nnzc2 == nnzc { sum_mvd & 1 } else { rav1d_msac_decode_bool_bypass(msac) as i32 };
                        if s != 0 { mvs[n].1 = -mvs[n].1; }
                    }
                }
                if inter_mode == C_JOINT_NEWMV || inter_mode == C_OPFL_JOINT_NEWMV {
                    let np = end & 1; // "the one not handled above" (decode.c:2802)
                    mv_prec_out = (6 << (np * 4)) | (mv_prec << ((np == 0) as usize * 4));
                    let (py, px) = mv_projection(mvs[1 - np], rdist[1], rdist[0], -0xffff, 0xffff);
                    mvs[np] = (py, px);
                    // jmvd_scale (decode.c:1510)
                    if amvd {
                        match jmvd_scale_mode {
                            1 => { mvs[np].0 *= 2; mvs[np].1 *= 2; }
                            2 => { mvs[np].0 /= 2; mvs[np].1 /= 2; }
                            _ => {}
                        }
                    } else {
                        match jmvd_scale_mode {
                            1 => mvs[np].0 *= 2,
                            2 => mvs[np].1 *= 2,
                            3 => mvs[np].0 /= 2,
                            4 => mvs[np].1 /= 2,
                            _ => {}
                        }
                    }
                } else {
                    mv_prec_out = mv_prec * 0x11;
                }
            }
        }

        // --- refine_mv (decode.c:2814). SVC ref scaling unsupported → assumed unscaled. ---
        let mut refine_mv_v = 0u8;
        if seq_refine_mv && bw4.min(bh4) >= 2 && bw4 * bh4 > 4
            && inter_mode != C_GLOBALMV_GLOBALMV
            && refdist[refp0 as usize] == -refdist[refp1 as usize]
            && (opfl_refine_type != 1
                || ((1u32 << inter_mode) & ((1 << 19) | (1 << 20) | (1 << 22) | (1 << 23))) == 0)
        {
            if ((1u32 << inter_mode) & ((1 << 18) | (1 << 24) | (1 << 28))) != 0 {
                refine_mv_v = 2; // implicitly enabled
            } else {
                let ctx = (inter_mode - C_NEARMV_NEARMV) as usize;
                refine_mv_v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.refine_mv[ctx]) as u8;
                if dbg00 || dv2 { crate::dlog!("[D0] refine_mv ctx={ctx} v={refine_mv_v} r={}", msac.rng); }
            }
        }

        let has_subpel = inter_mode <= C_JOINT_NEWMV && refine_mv_v == 0
            && motion_mode == MM_TRANSLATION
            && (inter_mode != C_GLOBALMV_GLOBALMV || bw4.min(bh4) == 1);

        // --- comp_type: avg / wedge / seg (decode.c:2843) ---
        let mut comp_type = 1u8; // COMP_INTER_AVG
        let mut _wedge_idx = -1i32;
        let mut _wedge_sign = false;
        let mut _mask_sign = false;
        if inter_mode <= C_JOINT_NEWMV && refine_mv_v != 1
            && !(inter_mode == C_JOINT_NEWMV && amvd)
            && masked_compound && bw4.min(bh4) >= 2
        {
            // comptype_ctx (decode.c:2851): `num >= idx ? 0 : …` — gated by the NB idx.
            let ct_of = |sel: (bool, i32), a: &BlockNbCtx, l: &BlockNbCtx| -> usize {
                let (d, o) = (if sel.0 { l } else { a }, sel.1 as usize);
                if d.ref1[o] != -1 {
                    (d.comp_type[o] > 1) as usize
                } else {
                    ((d.ref0[o] != 0 && (d.ref0[o] as i32 - 1) == ffr) as usize) * 2
                }
            };
            let cctx0 = if nb_idx < 1 { 0 } else { ct_of(nx[0], a, l) };
            let cctx1 = if nb_idx < 2 { 0 } else { ct_of(nx[1], a, l) };
            let ctx = cctx0 + cctx1 + ((cctx0 != 0 && cctx1 != 0) as usize)
                + ((absrefdist[refp0 as usize] == absrefdist[refp1 as usize]) as usize) * 6;
            let has_mask = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.comp_type_masked[ctx]);
            if has_mask {
                if bw4.max(bh4) <= 16
                    && !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.comp_type_weighted)
                {
                    comp_type = 2; // COMP_INTER_WEDGE
                    _wedge_idx = read_wedge_idx(msac, &mut cdf.m);
                    _wedge_sign = rav1d_msac_decode_bool_bypass(msac);
                } else {
                    comp_type = 3; // COMP_INTER_SEG
                    _mask_sign = rav1d_msac_decode_bool_bypass(msac);
                }
            }
            if dbg00 || dv2 { crate::dlog!("[D0] comp_type ctx={ctx} ct={comp_type} r={}", msac.rng); }
        }

        // --- cwp_idx (decode.c:2883) ---
        let mut cwp_out: i8 = 8;
        if refine_mv_v == 0 && jmvd_scale_mode == 0 && seq_cwp && comp_type == 1
            && (inter_mode == C_NEARMV_NEARMV || inter_mode == C_JOINT_NEWMV)
        {
            let mut n = 0usize;
            while n < 4 {
                if !crate::av2_recon::work_tick("av2_recon:8343") { break; }
                if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.cwp_idx[n]) { break; }
                n += 1;
            }
            const CWP_W: [[i32; 5]; 2] = [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
            cwp_out = CWP_W[((f_refdir(refp0) ^ f_refdir(refp1)) == 0) as usize][n] as i8;
        }

        // --- subpel filter (common tail, decode.c:3268) ---
        let filter: u8 = if refine_mv_v != 0 || inter_mode >= C_OPFL_NEARMV_NEARMV {
            2 // FILTER_8TAP_SHARP
        } else {
            let subpel_mode = HDR_TOOL_CFG.with(|c| c.get().subpel_filter_mode);
            if subpel_mode == 4 {
                if has_subpel {
                    let nbf = nb_setup(have_left, have_top_in_sb, bx4, by4, bw4c, bh4c, bw4, bh4);
                    let ctx = get_filter_ctx(a, l, nbf, refp0 as u8, true);
                    let f = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.filter[ctx], 2) as u8;
                    if dbg00 || dv2 { crate::dlog!("[D0] comp subpelfilter ctx={ctx} f={f} r={}", msac.rng); }
                    f
                } else {
                    0 // REGULAR
                }
            } else {
                subpel_mode
            }
        };

        // --- neighbour splat + return ---
        splat_inter_nb(a, l, bx4, by4, bw4, bh4, 0, motion_mode, refp0 as u8 + 1, mvprec_def,
                       inter_mode, 0, refp1 as i8);
        splat_nb(&mut a.skip_txfm, &mut l.skip_txfm, bx4, by4, bw4, bh4, skip as u8);
        splat_nb(&mut a.filter, &mut l.filter, bx4, by4, bw4, bh4, filter);
        splat_nb(&mut a.amvd, &mut l.amvd, bx4, by4, bw4, bh4, amvd as u8);
        splat_nb(&mut a.comp_type, &mut l.comp_type, bx4, by4, bw4, bh4, comp_type);
        splat_nb(&mut a.midx, &mut l.midx, bx4, by4, bw4, bh4, 0xff);
        splat_nb(&mut a.fsc, &mut l.fsc, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.mrl, &mut l.mrl, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.multi_mrl, &mut l.multi_mrl, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.intrabc, &mut l.intrabc, bx4, by4, bw4, bh4, 0);
        return InterInfo {
            skip, inter_mode, motion_mode, mvprec_def,
            mv_y: mvs[0].0, mv_x: mvs[0].1, warp_ref_idx: 0, drl_idx: drl_idx[0],
            mv_prec: mv_prec_out, amvd, warp_delta: [0, 0, 0, 0], warpmv_with_mvd: false,
            filter, bawp: 0, bawp_chroma: 0, ref0: refp0 as u8, ref1: refp1 as i8,
            drl_idx1: drl_idx[1], mv_y1: mvs[1].0, mv_x1: mvs[1].1, cwp: cwp_out,
            comp_type, wedge_idx: _wedge_idx, wedge_sign: _wedge_sign, mask_sign: _mask_sign, is_tip: false, refine_mv: refine_mv_v, ii_mode: -1,
        };
    }
    let ref0: u8 = if !is_tip && !seg_globalmv_skip && n_ref_frames > 1 {
        // cnt[k] = #neighbour ref-slots using ref (k-1). dav2d (decode.c:2925) gates the two adds
        // by the NB idx (`if (idx > 0)…if (idx > 1)`), reading the NX slots, and counts BOTH
        // ref[0]+1 and ref[1]+1 per neighbour. Mine's ref0 is raw+1 (0=unavailable) so cnt[ref0]++
        // == cnt[raw+1]++; ref1 is raw (-1=single) so cnt[ref1+1]++.
        let mut cnt = [0i32; 9];
        for (k, &(sel, off)) in [nx[0], nx[1]].iter().enumerate() {
            if nb_idx <= k {
                break;
            }
            let d = if sel { &*l } else { &*a };
            if dbg00 || dv2 { crate::dlog!("[D0] single_ref nb sel={sel} off={off} ref0_stored={} ref1={}", d.ref0[off as usize], d.ref1[off as usize]); }
            cnt[d.ref0[off as usize] as usize] += 1;
            cnt[(d.ref1[off as usize] + 1) as usize] += 1;
        }
        let mut cnt_rem = nb_idx as i32 * 2 - cnt[0] - cnt[8];
        let idx = nb_idx as i32;
        let mut i = 0usize;
        let mut ctx0 = 0usize;
        loop {
            if !work_tick("recon_loop:8457") { break; }
            let cnt_cur = cnt[i + 1];
            cnt_rem -= cnt_cur;
            let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
            if i == 0 { ctx0 = ctx; }
            if (dbg00 || dv2) && i == 0 { crate::dlog!("[D0] single_ref cell[{ctx}][0]={:?} rng={} dif={:x}", cdf.m.single_ref[ctx][i], msac.rng, msac.dif); }
            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.single_ref[ctx][i]) {
                break;
            }
            i += 1;
            if i >= n_ref_frames as usize - 1 {
                break;
            }
        }
        if dbg00 || dv2 { crate::dlog!("[D0] single_ref ref0={i} ctx0={ctx0} idx={idx} cnt={cnt:?} r={}", msac.rng); }
        i as u8
    } else {
        0
    };
    let have_top_right = bx4 + bw4 <= col_end;
    let have_bottom_left = by4 + bh4 <= row_end;
    let sngl_ctx = get_snglref_ctx(a, l, by4, bx4, have_top, have_left, have_top_right, have_bottom_left, bw4, bh4, ref0);

    // --- inter mode ---
    let mut inter_mode;
    if seg_globalmv_skip {
        inter_mode = GLOBALMV;
    } else if is_tip {
        inter_mode = NEARMV + 2 * rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.tip_mode) as u8;
    } else {
        let mut allow_warp = false;
        // dav2d decode.c:2965 gates allow_warp on the FRAME `warp_motion` flag, not seq motion_modes.
        if bw4.min(bh4) >= 2 && HDR_TOOL_CFG.with(|c| c.get().warp_motion) {
            if std::env::var("MPREDRL").is_ok() {
                crate::dlog!("[MWARP] mi=({bx4},{by4}) rng={}", msac.rng);
            }
            let ctx = get_warp_ctx(a, l, by4, bx4, have_top, have_left, have_top_right, have_bottom_left, bw4, bh4, ref0, is_sb_boundary, col_end);
            allow_warp = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp[ctx]);
            if dbg00 || dv2 { crate::dlog!("[D0] warp ctx={ctx} allow={} r={} (dav (0,0) no warp shown, mode NEARMV r=42464)", allow_warp as u8, msac.rng); }
            if dbg04 { crate::dlog!("DBI04 warp_ctx={ctx} allow={} rng={} sngl_ctx={sngl_ctx}", allow_warp as u8, msac.rng); }
            if bx4 == 24 && by4 == 4 { crate::dlog!("B244 warp_ctx={ctx} allow={} rng={} dif={:x} (oracle warp_ctx=0 allow=0 rng=34896)", allow_warp as u8, msac.rng, msac.dif); }
        }
        if allow_warp {
            inter_mode = if !force_integer_mv && !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_newmv) {
                WARPNEWMV
            } else {
                WARPMV
            };
        } else {
            inter_mode = NEARMV + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.inter_mode[sngl_ctx], 2) as u8;
        }
        if bx4 == 24 && by4 == 0 { crate::dlog!("W240 after_warp allow={} inter_mode={inter_mode} rng={} dif={:x}", allow_warp as u8, msac.rng, msac.dif); }
        if bx4 == 24 && by4 == 4 { crate::dlog!("B244 inter_mode={inter_mode} sngl_ctx={sngl_ctx} rng={} dif={:x} (oracle NEARMV=13 sngl_ctx=0 rng=48960)", msac.rng, msac.dif); }
    }
    if dbg00 || dv2 { crate::dlog!("[D0] ref sngl_ctx={sngl_ctx} inter_mode={inter_mode} r={} (dav ref r=48960 sim[ctx=0,15] r=64672)", msac.rng); }

    // --- amvd (adaptive MVD): only NEWMV, when the seq enables it (dav2d decode.c:3043). ---
    let mut amvd = false;
    if adaptive_mvd && inter_mode == NEWMV {
        // dav decode.c:3010: the neighbour's ref[0] must equal THE BLOCK's ref[0] — for a TIP
        // block that is TIP_FRAME (stored marker 8), not the default single ref.
        let block_stored: u8 = if is_tip { 8 } else { ref0 + 1 };
        let rd = |sel: (bool, i32)| -> usize {
            if sel.1 < 0 { return 0; }
            let (d, o) = (if sel.0 { &*l } else { &*a }, sel.1 as usize);
            (d.ref0[o] == block_stored && d.amvd[o] != 0) as usize
        };
        let ctx = rd(nx[0]) + rd(nx[1]);
        amvd = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.amvd[4][ctx]);
        if dbg00 || dv2 { crate::dlog!("[D0] amvd ctx={ctx} v={} r={} (dav amvd[4|0,0] r=64080)", amvd as u8, msac.rng); }
    }

    // --- motion mode (warp modes) ---
    let mut motion_mode = MM_TRANSLATION;
    let mut warp_ref_idx = 0usize;
    let mut warpmv_with_mvd = false;
    let mut bawp_out = 0u8;
    let mut mm_ii_mode: i8 = -1;
    let mut bawp1_out = 0u8;
    // warp_delta params (brick B warp splat): b->matrix[n] = signed decoded delta × step; [2]=-0x80
    // marks np==2 (only set for wri==0 WARPNEWMV). Default [0,0,0,0] = no delta (dav2d b->matrix),
    // so the matrix reconstruction reduces to `warp[wri]` unchanged for WARPMV / wri>0 blocks.
    let mut warp_delta = [0i32; 4];
    if inter_mode == WARPNEWMV || inter_mode == WARPMV {
        motion_mode = MM_WARP_DELTA;
        if inter_mode == WARPNEWMV && has_cs_ext(a, l, bx4, by4, bw4, bh4, row_end, col_end, ref0) {
            let nb = nb_setup(have_left, have_top_in_sb, bx4, by4, bw4c, bh4c, bw4, bh4);
            let x1 = nb_motion_mode(a, l, nb[0]);
            let x2 = nb_motion_mode(a, l, nb[1]);
            let ext_ctx = (x1 >= MM_WARP_CAUSAL) as usize + (x2 >= MM_WARP_CAUSAL) as usize;
            if motion_modes & (1 << MM_WARP_EXTEND) != 0
                && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_extend[ext_ctx])
            {
                motion_mode = MM_WARP_EXTEND;
            } else if motion_modes & (3 << MM_WARP_CAUSAL) == (3 << MM_WARP_CAUSAL) {
                let cs_ctx = (ext_ctx > 0) as usize + (x1 == MM_WARP_CAUSAL) as usize + (x2 == MM_WARP_CAUSAL) as usize;
                motion_mode = if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_causal[cs_ctx]) {
                    MM_WARP_CAUSAL
                } else {
                    MM_WARP_DELTA
                };
            } else {
                motion_mode = if motion_modes & (1 << MM_WARP_CAUSAL) != 0 { MM_WARP_CAUSAL } else { MM_WARP_DELTA };
            }
        }
        if dbg04 { crate::dlog!("DBI04 warp: inter_mode={inter_mode} motion_mode={motion_mode} rng={} (oracle mm=3)", msac.rng); }
        if motion_mode == MM_WARP_DELTA {
            while warp_ref_idx < 3 {
                if !crate::av2_recon::work_tick("av2_recon:8515") { break; }
                if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_ref_idx[warp_ref_idx]) {
                    break;
                }
                warp_ref_idx += 1;
            }
            if dbg04 { crate::dlog!("DBI04 warp_ref_idx={warp_ref_idx} rng={} (oracle wri=0 rng=42560)", msac.rng); }
        }
        if bx4 == 24 && by4 == 0 { crate::dlog!("W240 wri={warp_ref_idx} mm={motion_mode} rng={} dif={:x} (oracle wri=2 mm=3 rng=44544)", msac.rng, msac.dif); }
        if inter_mode == WARPMV && warp_ref_idx < 2 {
            warpmv_with_mvd = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warpmv_with_mvd);
        }
    } else if !is_tip && inter_mode <= NEWMV {
        // Non-warp mode (NEARMV/GLOBALMV/NEWMV): bawp, then interintra (dav2d decode.c:3058).
        // A TIP block codes NO bawp (dav decode.c:3008 `if (is_tip) { /* do nothing */ }`).
        // (amvd for NEWMV is a follow-up — needs neighbour amvd state; NEARMV/GLOBALMV skip it.)
        let mut bawp0 = 0u8;
        if bawp_enabled && tool_cfg.bawp && inter_mode != GLOBALMV && bw4.min(bh4) >= 2 {
            bawp0 = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.bawp[0]) as u8;
            if bawp0 != 0 {
                let ctx = if inter_mode == NEWMV { 2 - amvd as usize } else { 0 };
                bawp0 += rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.bawp_explicit[ctx]) as u8;
                if bawp0 == 2 {
                    bawp0 += rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.bawp_explicit_scale) as u8;
                    // Encode the mode-context into bits 2+ (dav2d decode.c:3079): the explicit-BAWP
                    // alpha (bawp_morph) reads `bawp_idx>>2` + `bawp_idx&1` as its magnitude+sign.
                    bawp0 |= (ctx as u8) << 2;
                }
                if has_chroma {
                    bawp1_out = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.bawp[1]) as u8;
                }
            }
        }
        bawp_out = bawp0;
        let _ = &mm_ii_mode;
        if bx4 == 24 && by4 == 4 { crate::dlog!("B244 bawp0={bawp0} inter_mode={inter_mode} rng={} dif={:x} (oracle bawp0=1 rng=44288)", msac.rng, msac.dif); }
        if dv2 { crate::dlog!("DV2 bawp={bawp0} bawp1={bawp1_out} bawp_enabled={bawp_enabled} r={} (dav bawp[1,1] r=37064)", msac.rng); }
        // interintra (MM_INTERINTRA=1): off for motion_modes=0x1d, but code the gate for generality.
        if motion_modes & (1 << 1) != 0 && bawp0 == 0 && bw4 * bh4 > 2 && bw4.max(bh4) <= 16 && inter_mode >= NEARMV {
            // size_group_lookup — interintra is OFF for this clip (dead code); plumb the real
            // table when a stream enables MM_INTERINTRA.
            let ctx = 0usize;
            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.interintra[ctx]) {
                motion_mode = 1; // MM_INTERINTRA
                mm_ii_mode = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.interintra_mode[ctx], 3) as i8;
                if bw4.min(bh4) > 1 && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.interintra_wedge) {
                    // read_wedge_idx — deferred (not reached for this clip)
                }
            }
        }
    }

    // --- DRL ---
    let mut drl_idx = 0usize;
    if inter_mode != WARPMV && inter_mode != GLOBALMV {
        let mut n = 0usize;
        let mut ctx = 0usize;
        if dbg00 { crate::dlog!("[D0] pre-drl sngl_ctx={sngl_ctx} cell={:?} rng={} dif={:x}", cdf.m.drl_idx[0][sngl_ctx], msac.rng, msac.dif); }
        if std::env::var("MPREDRL").is_ok() {
            crate::dlog!("[MPREDRL] mi=({bx4},{by4}) mode={inter_mode} mctx={sngl_ctx} rng={} cell={:?}", msac.rng, cdf.m.drl_idx[0][sngl_ctx]);
        }
        while n < max_drl_bits {
            if !crate::av2_recon::work_tick("av2_recon:8573") { break; }
            let cdf_cell = if is_tip { &mut cdf.m.tip_drl_idx[ctx] } else { &mut cdf.m.drl_idx[ctx][sngl_ctx] };
            if dbg04 { crate::dlog!("DBI04 drl iter n={n} ctx={ctx} cell={cdf_cell:?} rng_pre={}", msac.rng); }
            let bit = rav1d_msac_decode_bool_adapt(msac, cdf_cell);
            if dbg04 { crate::dlog!("DBI04 drl bit={} rng_post={}", bit as u8, msac.rng); }
            if !bit {
                break;
            }
            n += 1;
            if ctx < 2 {
                ctx += 1;
            }
        }
        drl_idx = n;
        if dbg00 || dv2 { crate::dlog!("[D0] drl={n} r={} (dav drl[0,-1] r=61424)", msac.rng); }
        if dbg04 { crate::dlog!("DBI04 drl={n} rng={}", msac.rng); }
        if bx4 == 24 && by4 == 0 { crate::dlog!("W240 drl={n} sngl_ctx={sngl_ctx} rng={} dif={:x} (oracle drl=2 rng=53936)", msac.rng, msac.dif); }
        if bx4 == 24 && by4 == 4 { crate::dlog!("B244 drl={n} sngl_ctx={sngl_ctx} rng={} dif={:x} (oracle drl=2 rng=61328)", msac.rng, msac.dif); }
    }

    // --- mv precision ---
    let mv_prec_tbl: [[i32; 3]; 2] = [[3, 1, 0], [4, 3, 1]];
    let mut mv_prec = 3 + frame_mv_precision;
    let mut mvprec_def: u8 = 1;
    if mv_prec > 3 && !amvd && (inter_mode == NEWMV || inter_mode == WARPNEWMV) {
        let nb = nb_setup(have_left, have_top_in_sb, bx4, by4, bw4c, bh4c, bw4, bh4);
        let mvp1 = nb_mvprec(a, l, nb[0]);
        let mvp2 = nb_mvprec(a, l, nb[1]);
        let ctx1 = (mvp1 & 1) as usize + (mvp2 & 1) as usize;
        if !rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.mvprec_def[ctx1]) {
            let ctx2 = ((mvp1 | mvp2) >> 1) as usize;
            let idx = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.mvprec_rem[ctx2][(mv_prec - 4) as usize], 2) as usize;
            mv_prec = mv_prec_tbl[(mv_prec == 6) as usize][idx];
            mvprec_def = 2;
        }
    }
    if dbg00 || dv2 { crate::dlog!("[D0] mvprec mv_prec={mv_prec} mvprec_def={mvprec_def} r={} (dav mvprec[..3] r=37088)", msac.rng); }
    if bx4 == 24 && by4 == 0 { crate::dlog!("W240 post-mvprec mv_prec={mv_prec} mvprec_def={mvprec_def} rng={} dif={:x} (oracle before_mv mv_prec=3 rng=39088)", msac.rng, msac.dif); }

    // --- MV residual + signs ---
    let (mut mv_y, mut mv_x) = (0i32, 0i32);
    if inter_mode == NEWMV || inter_mode == WARPNEWMV || (inter_mode == WARPMV && warpmv_with_mvd) {
        let mut nnzc;
        let mut sum_mvd = 0i32;
        if amvd {
            // Adaptive-MVD block: joint + per-axis index (magnitudes); signs stay bypass (nnzc=3).
            let (my, mx) = read_amvd(msac, &mut cdf.m);
            mv_y = my;
            mv_x = mx;
            nnzc = 3;
        } else {
            let mut st = cdf.mv.shell_tip;
            let dbgmv = (bx4 == 0 && by4 == 15) || (bx4 == 24 && by4 == 0);
            if dbgmv { crate::dlog!("MVDBG ({bx4},{by4}) before_mv mv_prec={mv_prec} rng={} dif={:x}", msac.rng, msac.dif); }
            let (my, mx) = read_mv_residual(msac, &mut cdf.mv, &mut st, mv_prec);
            if dbgmv { crate::dlog!("MVDBG ({bx4},{by4}) after_mv mv=({mx},{my}) rng={} dif={:x} (oracle 24,0: mv=24,32 rng=42048 dif=9ae13fffffffffff)", msac.rng, msac.dif); }
            cdf.mv.shell_tip = st;
            mv_y = my;
            mv_x = mx;
            if dbg00 || dv2 { crate::dlog!("[D0] mvdiff mv=(y:{my},x:{mx}) r={} (dav mvdiff y:528 x:240 r=38112)", msac.rng); }
            nnzc = (mv_x != 0) as i32 + (mv_y != 0) as i32;
            sum_mvd = (mv_x + mv_y) >> (6 - mv_prec);
            if inter_mode == WARPMV
                || nnzc == 0
                || !mvd_sign_derive
                || motion_mode != MM_TRANSLATION
                || scc
                || frame_mv_precision == 3
                || mv_prec >= 5
            {
                nnzc = 3;
            }
        }
        let mut nnzc2 = 0i32;
        if mv_y != 0 {
            nnzc2 += 1;
            let s = if nnzc2 == nnzc { sum_mvd & 1 } else { rav1d_msac_decode_bool_bypass(msac) as i32 };
            if s != 0 {
                mv_y = -mv_y;
            }
        }
        if mv_x != 0 {
            nnzc2 += 1;
            let s = if nnzc2 == nnzc { sum_mvd & 1 } else { rav1d_msac_decode_bool_bypass(msac) as i32 };
            if s != 0 {
                mv_x = -mv_x;
            }
        }
        if bx4 == 0 && by4 == 15 { crate::dlog!("M015 mv=({mv_y},{mv_x}) nnzc={nnzc} sum_mvd={sum_mvd} mvd_sign_derive={mvd_sign_derive} mm={motion_mode} fmp={frame_mv_precision} mv_prec={mv_prec} amvd={} rng={} dif={:x}", amvd as u8, msac.rng, msac.dif); }
    }

    // --- warp-delta params (dav2d decode.c:3312) — WARPNEWMV + MM_WARP_DELTA, coded when
    // warp_ref_idx==0 (np=2) OR (six_param_warp_delta seq flag && warp_ref_idx==1) (np=4). ---
    if inter_mode == WARPNEWMV
        && motion_mode == MM_WARP_DELTA
        && ((six_param_warp_delta && warp_ref_idx == 1) || warp_ref_idx == 0)
    {
        let prec = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_delta_prec[bs]);
        let step = 2 >> prec as i32;
        let np = if six_param_warp_delta && warp_ref_idx == 1 { 4 } else { 2 };
        // b->matrix[n] = the signed decoded delta × step; the reconstruction adds `matrix[n] << 10`
        // to warp[wri][n+2]. np=4 uses all four matrix slots (the six-param affine deltas).
        for n in 0..np {
            let ctx = (n as u32).wrapping_sub(1) > 1;
            let nc = (!ctx) as usize;
            let mut m = crate::msac::rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.warp_delta_param[0][nc], 7) as i32;
            if m == 7 && prec {
                m += crate::msac::rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.warp_delta_param[1][nc], 7) as i32;
            }
            if m != 0 {
                if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_delta_sign) {
                    m = -m;
                }
                m *= step;
            }
            warp_delta[n] = m;
        }
        if np == 2 {
            warp_delta[2] = -0x80; // np==2 marker (dav2d decode.c:3337)
        }
    }

    // --- warp inter-intra (dav2d decode.c:3314): a WARPMV block decodes warp_interintra
    // regardless of the MM_INTERINTRA motion-mode flag. WARPNEWMV blocks skip it.
    let mut ii_mode: i8 = -1;
    if inter_mode == WARPMV && bw4.min(bh4) >= 2 && bw4.max(bh4) <= 16 {
        let ctx = size_group(bw4, bh4);
        if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.warp_interintra[ctx]) {
            ii_mode = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.interintra_mode[ctx], 3) as i8;
            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.interintra_wedge) {
                crate::dlog!("[rav2d] warp_ii WEDGE=1 — wedge_idx parse NOT implemented (desync ahead)");
            }
        }
    }

    // --- subpel filter (dav2d decode.c:3272): the interp filter symbol is coded ONLY in
    // SWITCHABLE mode, and only for non-warp modes (has_subpel: inter_mode<=NEWMV, not GLOBALMV
    // on big blocks). When the frame's subpel filter is a FIXED mode, `filter` = that mode with
    // NO symbol (golden frame-2 IS switchable → old unconditional decode worked; v432 is Regular).
    let has_subpel = inter_mode <= NEWMV && (inter_mode != GLOBALMV || bw4.min(bh4) == 1);
    let mut filter = 0u8;
    if dbg00 || dv2 { crate::dlog!("[D0] filter-gate subpel_mode={} has_subpel={has_subpel} r_pre={}", tool_cfg.subpel_filter_mode, msac.rng); }
    if is_tip {
        // TIP block: b->ref[0]==TIP_FRAME forces FILTER_8TAP_SHARP with NO symbol (decode.c:3270).
        filter = 2;
    } else if tool_cfg.subpel_filter_mode == 4 {
        if has_subpel {
            let nbf = nb_setup(have_left, have_top_in_sb, bx4, by4, bw4c, bh4c, bw4, bh4);
            let ctx = get_filter_ctx(a, l, nbf, ref0, false);
            if dbg00 || dv2 {
                for (i, sel) in nbf.iter().enumerate() {
                    let (d, tag) = (if sel.0 { &*l } else { &*a }, if sel.0 { "l" } else { "a" });
                    if sel.1 >= 0 {
                        let o = sel.1 as usize;
                        crate::dlog!("[D0] fltrnb[{i}]={tag}[{o}] ref0st={} ref1={} fltr={} (blk ref0={ref0})", d.ref0[o], d.ref1[o], d.filter[o]);
                    } else {
                        crate::dlog!("[D0] fltrnb[{i}] NONE");
                    }
                }
            }
            filter = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.filter[ctx], 2) as u8;
            if dbg00 || dv2 { crate::dlog!("[D0] subpelfilter ctx={ctx} filter={filter} r={}", msac.rng); }
        }
    } else {
        filter = tool_cfg.subpel_filter_mode;
    }

    if std::env::var("MPREDRL").is_ok() {
        crate::dlog!("[MEND] mi=({bx4},{by4}) rng={} skip={} skip_mode={}", msac.rng, skip as u8, skip_mode as u8);
    }
    // --- splat inter neighbour state ---
    // A TIP block stores ref[0]=TIP_FRAME(7) → mine's +1 convention = 8 (feeds the tip ctx +
    // the single_ref histogram's cnt[8]).
    let ref0_stored = if is_tip { 8 } else { ref0 + 1 };
    splat_inter_nb(a, l, bx4, by4, bw4, bh4, 0, motion_mode, ref0_stored, mvprec_def, inter_mode, skip_mode as u8, -1);
    splat_nb(&mut a.skip_txfm, &mut l.skip_txfm, bx4, by4, bw4, bh4, skip as u8);
    splat_nb(&mut a.comp_type, &mut l.comp_type, bx4, by4, bw4, bh4, 0);
    splat_nb(&mut a.filter, &mut l.filter, bx4, by4, bw4, bh4, filter);
    splat_nb(&mut a.amvd, &mut l.amvd, bx4, by4, bw4, bh4, amvd as u8);
    // Inter blocks reset midx/fsc to their base (dav2d decode.c:3385/3386) so a later intra
    // block's directional-mode reordering (mode_ctx + custom_mode_list) doesn't read a stale
    // directional midx from an earlier intra block.
    splat_nb(&mut a.midx, &mut l.midx, bx4, by4, bw4, bh4, 0xff);
    splat_nb(&mut a.fsc, &mut l.fsc, bx4, by4, bw4, bh4, 0);
    // Inter blocks also reset mrl/multi_mrl/intrabc to 0 (dav2d decode.c:3413/3423/3424) so a
    // later intra block's mrl/intrabc neighbour ctx doesn't read a stale 1 from an earlier intra.
    splat_nb(&mut a.mrl, &mut l.mrl, bx4, by4, bw4, bh4, 0);
    splat_nb(&mut a.multi_mrl, &mut l.multi_mrl, bx4, by4, bw4, bh4, 0);
    splat_nb(&mut a.intrabc, &mut l.intrabc, bx4, by4, bw4, bh4, 0);

    if bx4 == 24 && by4 == 0 { crate::dlog!("W240 END-mode inter_mode={inter_mode} mm={motion_mode} filter={filter} rng={} dif={:x}", msac.rng, msac.dif); }
    if bx4 == 24 && by4 == 4 { crate::dlog!("B244 END-mode inter_mode={inter_mode} mm={motion_mode} filter={filter} rng={} dif={:x} (oracle SUBPEL filter=1 rng=33136)", msac.rng, msac.dif); }
    InterInfo { skip, inter_mode, motion_mode, mvprec_def, mv_y, mv_x, warp_ref_idx, drl_idx, mv_prec, amvd, warp_delta, warpmv_with_mvd, filter, bawp: bawp_out, bawp_chroma: bawp1_out, ref0, ref1: -1, drl_idx1: drl_idx, mv_y1: 0, mv_x1: 0, cwp: 8, comp_type: 0, wedge_idx: -1, wedge_sign: false, mask_sign: false, is_tip, refine_mv: 0, ii_mode: if ii_mode >= 0 { ii_mode } else { mm_ii_mode } }
}

/// TX scan index into `SCANS` from the TX log2 dims `(slw, slh)` — the dav2d
/// `RectTxfmSize` enum order (square on the diagonal, rectangular off it).
#[rustfmt::skip]
pub fn scan_idx_square(slw: usize, slh: usize) -> usize {
    const TX_SCAN_IDX: [[u8; 5]; 5] = [
        // slh:  0   1   2   3   4
        [        0,  5, 13, 19, 23], // slw=0: 4x{4,8,16,32,64}
        [        6,  1,  7, 15, 21], // slw=1: 8x{4,8,16,32,64}
        [       14,  8,  2,  9, 17], // slw=2: 16x{4,8,16,32,64}
        [       20, 16, 10,  3, 11], // slw=3: 32x{4,8,16,32,64}
        [       24, 22, 18, 12,  4], // slw=4: 64x{4,8,16,32,64}
    ];
    TX_SCAN_IDX[slw.min(4)][slh.min(4)] as usize
}

/// Decode one intra-luma leaf block's mode + skip info (dav2d `decode_b`, intra path,
/// pre-coefficient) with every context COMPUTED from the live neighbour arrays, then
/// splat each field. The filter params (gdf/cdef/ccso) are decoded once per SB by the
/// caller; coefficient decode is dispatched by the caller from `LeafInfo`. Interior
/// blocks only (w4==bw4, h4==bh4). Returns the decoded info.
#[allow(clippy::too_many_arguments)]
/// Decode an MV difference (dav2d `read_mv_residual`) into `(y, x)` via shell/pair
/// coding. `cdf_mv` is the dmv (intrabc) or mv (inter) component context; `shell_tip` is
/// the global tip CDF; `mv_prec ∈ {3,5}` sets the shell layout (`n_syms = 9 + mv_prec`).
/// Returns `(0, 0)` when the shell index is zero. Magnitudes are unsigned here — the
/// caller applies the per-component sign bits.
pub fn read_mv_residual(
    msac: &mut crate::msac::MsacContext,
    cdf_mv: &mut crate::cdf_av2::CdfMvContext,
    shell_tip: &mut [u16; 2],
    mv_prec: i32,
) -> (i32, i32) {
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bool_bypass,
        rav1d_msac_decode_symbol_adapt8, rav1d_msac_decode_uniform,
    };
    let n_syms = 9 + mv_prec;
    let h_syms = n_syms >> 1;
    let mut sh_class: i32;
    if rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.shell_set) {
        let h_syms2 = n_syms - h_syms;
        sh_class = h_syms
            + 1
            + rav1d_msac_decode_symbol_adapt8(
                msac,
                &mut cdf_mv.shell_upper[mv_prec as usize],
                h_syms2.min(7) as u8,
            ) as i32;
        if mv_prec + sh_class == 21 {
            sh_class += rav1d_msac_decode_bool_adapt(msac, shell_tip) as i32;
        }
    } else {
        sh_class = rav1d_msac_decode_symbol_adapt8(
            msac,
            &mut cdf_mv.shell_lower[mv_prec as usize],
            h_syms as u8,
        ) as i32;
    }

    let mut sh_index: i32;
    if sh_class < 2 {
        sh_index =
            rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.shell_offset_low[sh_class as usize])
                as i32;
    } else if sh_class == 2 {
        sh_index = rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.shell_offset_cl2) as i32;
        if sh_index != 0 {
            sh_index += rav1d_msac_decode_bool_bypass(msac) as i32;
            if sh_index == 2 {
                sh_index += rav1d_msac_decode_bool_bypass(msac) as i32;
            }
        }
    } else {
        sh_index = 0;
        let mut m = 1i32;
        for i in 0..sh_class {
            sh_index |= m
                * rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.shell_offset_hi[i as usize])
                    as i32;
            m <<= 1;
        }
    }

    if sh_class != 0 {
        sh_index += 1 << sh_class;
    }
    if sh_index == 0 {
        return (0, 0);
    }

    let mut pair_index = 0i32;
    if sh_index >= 2 {
        pair_index = rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.col_component[0]) as i32;
        if pair_index != 0 && sh_index >= 4 {
            pair_index += rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.col_component[1]) as i32;
            if pair_index == 2 && sh_index >= 6 {
                pair_index +=
                    rav1d_msac_decode_uniform(msac, ((sh_index >> 1) - 1) as std::ffi::c_uint) as i32;
            }
        }
    }

    let sh = 6 - mv_prec;
    if pair_index * 2 == sh_index {
        let v = (sh_index >> 1) << sh;
        (v, v)
    } else {
        let b = rav1d_msac_decode_bool_adapt(msac, &mut cdf_mv.col_index[sh_class.min(3) as usize]);
        if b {
            (pair_index << sh, (sh_index - pair_index) << sh)
        } else {
            ((sh_index - pair_index) << sh, pair_index << sh)
        }
    }
}

/// Inter (and intrabc) luma TX coefficient decode AFTER the eob: the txtp symbol(s) then the
/// level decode, returning the cf neighbour-context byte. Shared by decode_b_luma's intrabc path
/// and the frame-2 inter descent. `a_lcoef`/`l_lcoef` are the already-sliced neighbour lcoef
/// spans (cleared `[0u8; ..]` for the first block).
pub fn decode_luma_tx_level(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    cf: &mut [i32],
    e: i32,
    slw: usize,
    slh: usize,
    clw: usize,
    clh: usize,
    t_dim_ctx: usize,
    tx2dszctx: usize,
    bw4: usize,
    bh4: usize,
    a_lcoef: &[u8],
    l_lcoef: &[u8],
) -> (u8, u8, u8) {
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_symbol_adapt4,
        rav1d_msac_decode_symbol_adapt8,
    };
    let t_dim_min = slw.min(slh);
    let tmax = slw.max(slh);
    let txtp = if slw.min(slh) >= 3 && tmax == 4 {
        DCT_DCT
    } else {
        // eob→(x,y) uses the COEF-REGION-clamped log-width (clw), not the full slw (dav2d
        // recon_tmpl.c:597) — a 64-wide/tall tx codes only the 32-core, so slw=4→clw=3. Only
        // 64-wide/tall inter blocks (v320 f2 (24,0) 64×16) exposed it; clw==slw for ≤32.
        let ey = e >> (2 + clw);
        let ex = e & ((4 << clw) - 1);
        let xy = ex + ey;
        let (tw, th) = (bw4.min(8), bh4.min(8));
        let txtp_ctx = if xy < 2 {
            1
        } else if xy > 4 * (tw as i32 + th as i32) - 4 {
            2
        } else {
            0
        } as usize;
        if slw == 3 && slh == 3 {
            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.txtp_inter_dct_idtx[txtp_ctx][3]) {
                DCT_DCT
            } else {
                IDTX_TT
            }
        } else if tmax >= 3 {
            let long_dct =
                tmax >= 4 || rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.txtp_long32_dct[1]);
            let short_idx = rav1d_msac_decode_symbol_adapt4(
                msac,
                &mut cdf.m.txtp_inter_short_1d[txtp_ctx][t_dim_min],
                3,
            ) as usize;
            TXTP_LONG_TBL[long_dct as usize][(slw < slh) as usize][short_idx]
        } else {
            let setidx = (slw == 2 && slh == 2) as usize;
            let set = rav1d_msac_decode_bool_adapt(
                msac,
                &mut cdf.m.txtp_inter_tx_set[setidx][txtp_ctx][t_dim_min],
            );
            let txtp_idx = if !set {
                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.txtp_inter_set0[setidx][txtp_ctx], 7)
                    as usize
            } else if setidx == 1 {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.txtp_inter_set2[txtp_ctx], 3)
                    as usize
                    + 8
            } else {
                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.txtp_inter_set1[txtp_ctx], 7)
                    as usize
                    + 8
            };
            TXTP_INV_TBL[setidx][txtp_idx]
        }
    };
    let tx_class = (txtp >> 3) & 3;
    if crate::av2_coef::COEF_DBG.with(|c| c.get()) {
        crate::dlog!("TXDBG post-txtp txtp={txtp} tmin={t_dim_min} rng={} dif={:x}", msac.rng, msac.dif);
    }
    let scan = crate::av2_tables_gen::SCANS[scan_idx_square(clw, clh)];
    let mut stx_type = 0u8; // inter secondary transform (STX/IST), 0 = none
    let cf_ctx = if txtp == IDTX_TT {
        crate::av2_coef::decode_coefs_idtx_y(
            msac, &mut cdf.coef, cf, e, tx2dszctx, t_dim_ctx.min(2), clw, clh, scan,
        )
    } else {
        let dc_sign_ctx =
            crate::av2_coef::get_dc_sign_ctx(a_lcoef, l_lcoef, slw, slh, bw4 as i32, bh4 as i32);
        if crate::av2_coef::COEF_DBG.with(|c| c.get()) {
            crate::dlog!(
                "TXDBG dcsign-nb a={:02x?} l={:02x?} ctx={dc_sign_ctx}",
                &a_lcoef[..(1 << slw).min(a_lcoef.len())],
                &l_lcoef[..(1 << slh).min(l_lcoef.len())]
            );
        }
        if tx_class != 0 {
            crate::av2_coef::decode_coefs_hv_y(
                msac, &mut cdf.coef, cf, e, tx_class as usize, tx2dszctx, t_dim_ctx, clw, clh,
                false, dc_sign_ctx,
            )
        } else {
            if t_dim_min >= 2 && txtp == DCT_DCT && e >= 3 && e < 32 && SEQ_TOOLS.with(|c| c.get().ist_inter) {
                stx_type = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.stx[1][t_dim_min], 3) as u8;
                if crate::av2_coef::COEF_DBG.with(|c| c.get()) {
                    crate::dlog!("TXDBG post-stx stx={stx_type} rng={} dif={:x}", msac.rng, msac.dif);
                }
            }
            if e == 0 {
                crate::av2_coef::decode_coefs_dc_only_y(
                    msac, &mut cdf.coef, cf, t_dim_ctx, dc_sign_ctx,
                )
            } else {
                crate::av2_coef::decode_coefs_dct_y(
                    msac, &mut cdf.coef, cf, e, tx2dszctx, t_dim_ctx, clw, clh, scan, false,
                    dc_sign_ctx,
                )
            }
        }
    };
    (cf_ctx, txtp, stx_type)
}

pub fn decode_b_luma(
    msac: &mut crate::msac::MsacContext,
    cdf: &mut crate::cdf_av2::CdfContext,
    a: &mut BlockNbCtx,
    l: &mut BlockNbCtx,
    bs: usize,
    bx4: usize,
    by4: usize,
    have_left: bool,
    have_top: bool,
    decode_filters: bool,
    force_integer_mv: u8,
    max_bvp_drl_bits: u8,
    left_cdef: &mut i8,
    left_ccso: &mut [u8; 3],
    top_cdef: i8,
    allow_intrabc: bool,
    // True for an intra block in an INTER frame that is NOT in an SDP intra_region: such a
    // block's fsc uses the special CDF context 3 (dav2d decode.c:2124), not the neighbour sum.
    inter_frame: bool,
    // True for a YUV block (has_luma && has_chroma) in an inter frame: dav2d decodes chroma
    // MODE between the luma mode and read_coef_blocks, so we must stop after the luma mode
    // (splatting mode ctx) and let the caller decode chroma mode + all coefs. Returns eob=-1.
    defer_coefs: bool,
    // Frame cdef `n_strengths` (obu.c:1635): keyframe(frame 1)=3, inter frame(frame 2)=2. Drives
    // the once-per-SB cdef_idx read (==2 → no cdef_idx symbol after the cdef_idx0 bool).
    cdef_n_strengths: usize,
) -> LeafInfo {
    use crate::av2_decode::BLOCK_DIMENSIONS;
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bool_bypass,
        rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8,
    };
    let bd = BLOCK_DIMENSIONS[bs];
    // w4/h4 = FULL block dims; bw4/bh4 = CLAMPED to the frame (dav2d: an edge block spilling past
    // the frame has bw4 < w4, so the `bw4 == w4` neighbour-context checks skip the off-frame
    // above-right / bottom-left slot and fall through to the in-frame top-left / above neighbour).
    let (w4, h4) = (bd[0] as usize, bd[1] as usize);
    let (iw4c, ih4c) = crate::av2_frame::FRAME.with(|fr| {
        let f = fr.borrow();
        (f.iw4, f.ih4)
    });
    let bw4 = if iw4c > bx4 { w4.min(iw4c - bx4) } else { w4 };
    let bh4 = if ih4c > by4 { h4.min(ih4c - by4) } else { h4 };
    // The block-context TOP neighbour (intrabc/skip_txfm/fsc/mrl) must NOT cross the SB
    // boundary vertically (dav2d `have_top_in_sb = by4 & (sb_step-1)`; 64px SB = 16 4px). The
    // tile-level `have_top` (used for intra-prediction edges later) still crosses it.
    let have_top_in_sb = (by4 & (sb_step4() - 1)) != 0;
    let slots = gather_nb(have_left, have_top_in_sb, bx4, by4, bw4, bh4, w4, h4);

    // Parsed-header tool gating (see HdrToolCfg): defaults keep the dev clip exact.
    let tool_cfg = HDR_TOOL_CFG.with(|c| c.get());
    let allow_intrabc = allow_intrabc && tool_cfg.allow_intrabc;
    if std::env::var("MKEYL").is_ok() { crate::dlog!("[MKEYL] mi=({bx4},{by4}) bs={bs} rng={}", msac.rng); }
    let p60 = ((bx4 == 6 && by4 == 0) || (bx4 == 0 && by4 == 6)) && allow_intrabc;
    // intrabc — only coded when allowed (off in inter frames) AND min(bw4,bh4) < 16.
    // dav2d decode.c:1727: the size gate uses the FULL block dims (dav bw4/bh4 = b_dim,
    // ours are w4/h4) — an edge-clamped 64x64 leaf (visible 48px) codes NO intrabc bool.
    let intrabc = if allow_intrabc && w4.min(h4) < 16 {
        let ctx = nb_sum(&slots, &a.intrabc, &l.intrabc) as usize;
        let v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.intrabc[ctx]);
        if std::env::var("MIBC").is_ok() { crate::dlog!("[MIBC] mi=({bx4},{by4}) ctx={ctx} ibc={} rng={}", v as u8, msac.rng); }
        if p60 { crate::dlog!("R60L intrabc ctx={ctx} DECODED={} rng={} dif={:x}", v as u8, msac.rng, msac.dif); }
        if std::env::var("V320").is_ok() && bx4 == 0 && by4 == 0 { crate::dlog!("V300 intrabc ctx={ctx} v={} r={} (dav intrabc r=53240)", v as u8, msac.rng); }
        if std::env::var("V64").is_ok() && bx4 == 64 && by4 == 24 { crate::dlog!("V64 intrabc ctx={ctx} v={} r={} (dav intrabc[ctx=2,1] r=34328)", v as u8, msac.rng); }
        v
    } else {
        if std::env::var("V320").is_ok() && bx4 == 0 && by4 == 0 { crate::dlog!("V300 intrabc NOT-DECODED (allow={allow_intrabc} mindim={})", bw4.min(bh4)); }
        false
    };

    // skip_txfm — decoded for intrabc (intra blocks store 0). Its ctx uses the CROSS-SB
    // neighbour set (dav2d `nx`, `have_top` crossing the SB-row boundary via the committed above
    // context; decode.c:1670-1687 + 1783) — NOT the within-SB spatial set (`gather_nb`/
    // `have_top_in_sb`) used for the intrabc flag / mode / mv contexts. At a SB-row boundary the
    // above skip_txfm neighbour is the row above (differs from the within-SB left fallback).
    let skip_txfm = if intrabc {
        let mut nx = [(false, -1i32), (false, -1i32)];
        let mut ii = 0usize;
        if have_left && bh4 == h4 { nx[0] = (true, (by4 + bh4 - 1) as i32); ii += 1; }
        if have_top && bw4 == w4 { nx[ii] = (false, (bx4 + bw4 - 1) as i32); ii += 1; }
        if ii < 2 && have_left { nx[ii] = (true, by4 as i32); ii += 1; }
        if ii < 2 { nx[ii] = (false, bx4 as i32); if ii == 0 { nx[1] = (false, bx4 as i32); } }
        let ctx = get_skip_txfm_ctx(a, l, nx, 0);
        let v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.skip_txfm[ctx]);
        if std::env::var("MIBC").is_ok() && bx4 == 64 && by4 == 0 { crate::dlog!("[MSEQ] post-skiptx skip={} ctx={ctx} rng={}", v as u8, msac.rng); }
        v
    } else {
        false
    };

    // Once-per-SB filter params (gdf, cdef_idx, ccso×3) — decoded at the SB's first leaf,
    // AFTER intrabc and BEFORE the mode (dav2d order). Each has its own alignment/context:
    //  - gdf  : once per 128px block (KEY-frame gdf_bs = 32 4px units) → (bx|by)&31==0;
    //  - cdef : once per 64px SB, context from the left/top neighbour cdef value (top is
    //           absent in SB-row 0, so ctx = left==0 ? 2 : 0; -1 = no neighbour → 0);
    //  - ccso : once per 256px block → (bx|by)&63==0.
    if !decode_filters {
        // Non-SB-first leaves can still trigger a per-64-cell cdef_idx read (128px SBs).
        read_cdef_per64(msac, cdf, bx4, by4, w4, h4, skip_txfm);
    }
    if decode_filters {
        // TILE-ADAPTIVE units (avm): gdf block 128px→64px, ccso unit 256px→SB.
        let (ccso_u4, gdf_bs4) = FILTER_UNITS.with(|c| c.get());
        if std::env::var("V320").is_ok() && bx4 == 0 && by4 == 0 { crate::dlog!("V300 filter-block gdf_gate={} cdef_gate={} r={} (dav: NO gdf/cdef for v320 f1 (0,0))", tool_cfg.gdf, tool_cfg.cdef, msac.rng); }
        if tool_cfg.gdf && (bx4 | by4) & (gdf_bs4 - 1) == 0 {
            let gdf = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.gdf);
            // C4: store this 128px GDF block's on/off flag for the filter pass.
            if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
                crate::av2_frame::FRAME.with(|f| f.borrow_mut().set_gdf_blk(bx4, by4, gdf));
            }
        }
        // cdef ctx from the left + top 64px-SB cdef indices (dav2d decode.c:1853): both -1 =
        // no neighbour. Both present → `(left==0)+(top==0)`, bumped 2→3; else `(left&top==0)*2`
        // (which, with one side -1, reduces to `(present_side==0)*2` — and the row-0 `left==0?2:0`).
        // Per-64-cell cdef_idx (dav order: gdf → cdef → ccso within the leaf).
        read_cdef_per64(msac, cdf, bx4, by4, w4, h4, skip_txfm);
        let _ = (top_cdef, cdef_n_strengths);
        if tool_cfg.ccso && (bx4 | by4) & (ccso_u4 - 1) == 0 {
            // dav2d decode.c:1892 — a per-SB ccso flag is coded ONLY for planes that are ENABLED
            // AND NOT sb_reuse (avm read_ccso: an sb_reuse plane inherits the ref slot's per-SB
            // flags with NO coded symbol — same rule as the inter path's block above).
            let (ccso_en, ccso_reuse): ([bool; 3], [bool; 3]) = crate::av2_frame::CCSO_CFG.with(|c| {
                let cfg = c.borrow();
                (
                    std::array::from_fn(|p| cfg.p.get(p).map_or(false, |pc| pc.enabled)),
                    std::array::from_fn(|p| cfg.p.get(p).map_or(false, |pc| pc.sb_reuse)),
                )
            });
            for p in 0..3 {
                if !ccso_en[p] || ccso_reuse[p] {
                    continue;
                }
                // ccso left-neighbour context (dav2d decode.c:1905): `left_ccso[p]*2` when a
                // left ccso UNIT exists inside this tile (`bx4 - unit >= tile col_start`),
                // else 0. The value threads across units (non-aligned SBs don't recode it).
                let ctx = if bx4 as i32 - ccso_u4 as i32 >= TILE_B.with(|t| t.get().0) as i32 {
                    left_ccso[p] as usize * 2
                } else {
                    0
                };
                let v300_precdf = cdf.m.ccso[p][ctx][0];
                left_ccso[p] = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.ccso[p][ctx]) as u8;
                if std::env::var("V320").is_ok() && bx4 == 0 && by4 == 0 { crate::dlog!("V300 ccso p={p} ctx={ctx} v={} r={} pre_cdf={v300_precdf} (dav ccso pl=y v=1 r=46160)", left_ccso[p], msac.rng); }
                // C3: store this 256px SB's CCSO enable flag for the filter pass.
                if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
                    let on = left_ccso[p] != 0;
                    crate::av2_frame::FRAME.with(|f| f.borrow_mut().set_ccso_blk(bx4, by4, p, on));
                }
            }
        }
        // --- delta-q (dav2d decode.c:1941): per-SB running qindex, after ccso, before the mode.
        read_delta_q(msac, cdf, bs, skip_txfm, bx4, by4);
    }

    // Intra-block-copy fork — the block-copy MV decode replaces the intra mode path.
    // Verified bit-exact against the oracle: the mv-diff lands at the oracle's rng.
    if intrabc {
        // intrabc leaves carry no palette — refresh the neighbour palette caches with none.
        crate::av2_palette::pal_splat(bx4, by4, bw4, bh4, 0, &[0u16; 8]);
        if std::env::var("MIBC").is_ok() && bx4 == 64 && by4 == 0 { crate::dlog!("[MSEQ] pre-bvinfo rng={}", msac.rng); }
        // is_refmv → DRL bypass run → (precision when force_integer_mv==0) → mv residual
        // + per-component signs.
        let is_refmv = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.intrabc_mode);
        let mut drl = 0u8;
        while (drl as usize) < max_bvp_drl_bits as usize {
            if !rav1d_msac_decode_bool_bypass(msac) {
                break;
            }
            drl += 1;
        }
        let mut is_qpel = force_integer_mv == 0; // !force_integer_mv
        if !is_refmv && force_integer_mv == 0 {
            is_qpel = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.intrabc_precision);
        }
        let mut mv = (0i32, 0i32);
        if !is_refmv {
            let mv_prec = 3 + 2 * is_qpel as i32;
            let (mut my, mut mx) =
                read_mv_residual(msac, &mut cdf.dmv, &mut cdf.mv.shell_tip, mv_prec);
            if my != 0 && rav1d_msac_decode_bool_bypass(msac) {
                my = -my;
            }
            if mx != 0 && rav1d_msac_decode_bool_bypass(msac) {
                mx = -mx;
            }
            mv = (my, mx);
        }
        // intrabc block vector = mvstack[drl] + delta (dav2d decode.c:1039-1067). The predictor is
        // the FULL refmvs spatial scan with ref=-1 — the intrabc candidates are neighbouring intrabc
        // blocks' BVs splatted into the grid (splat_intrabc, ref=-1) + the refmv bank + defaults,
        // via the same `refmvs_find` machinery as inter (brick B).
        let (iw4v, ih4v) = crate::av2_frame::FRAME.with(|fr| { let f = fr.borrow(); (f.iw4, f.ih4) });
        let (stack, _cnt) = crate::av2_refmvs::GRID.with(|g| {
            crate::av2_refmvs::BANK.with(|bk| {
                {
                let st = SEQ_TOOLS.with(|c| c.get());
                let bkb = bk.borrow();
                crate::av2_refmvs::refmvs_find(
                    &g.borrow(), bx4, by4, w4, h4, -1, -1, crate::av2_refmvs::Mv { y: 0, x: 0 }, crate::av2_refmvs::Mv { y: 0, x: 0 },
                    sb_step4(), iw4v, ih4v, st.drl_reorder,
                    if st.refmvbank { Some(&bkb) } else { None },
                    max_bvp_drl_bits as usize, false,
                )
                }
            })
        });
        let pred = stack.get(drl as usize).map(|c| c.mv[0]).unwrap_or(crate::av2_refmvs::Mv { y: 0, x: 0 });
        let (mut bvy, mut bvx) = (pred.y, pred.x);
        if std::env::var("V64").is_ok() && bx4 == 64 && by4 == 24 { crate::dlog!("V64 is_refmv={is_refmv} drl={drl} is_qpel={is_qpel} mvdelta=({},{}) pred=({},{}) BV=({bvy},{bvx}) r={} (dav mode1 drl1 prec1 r=58912)", mv.0, mv.1, pred.y, pred.x, msac.rng); }
        // (0,0) predictor fallback (dav decode.c:995-1004): force a default direction.
        // The "first SB row" test is TILE-relative (t->by - sb_step < tiling.row_start).
        if bvy == 0 && bvx == 0 {
            let sbsz = (sb_step4() * 4) as i32; // dav decode.c:998 `64 << sb128`
            let row_start = TILE_B.with(|t| t.get().2) as i32;
            if (by4 as i32 - sb_step4() as i32) < row_start { bvx = -(8 * (sbsz + 256)); } else { bvy = -(8 * sbsz); }
        }
        if std::env::var("IBCDBG").is_ok() {
            crate::dlog!("[IBC] ({bx4},{by4}) is_refmv={is_refmv} drl={drl} qpel={is_qpel} mvd=({},{}) pred=({},{}) r={}", mv.0, mv.1, pred.y, pred.x, msac.rng);
        }
        if !is_refmv {
            if !is_qpel {
                bvx = (bvx - (bvx >> 15) + 3) & !7;
                bvy = (bvy - (bvy >> 15) + 3) & !7;
            }
            bvx += mv.1;
            bvy += mv.0;
        }
        let ibc_bv = (bvy, bvx);
        if std::env::var("MIBC").is_ok() && bx4 == 64 && by4 == 0 { crate::dlog!("[MSEQ] post-bvinfo bv=({bvy},{bvx}) rng={}", msac.rng); }
        // morph_pred (avm read_intrabc_info tail, decodemv.c:1479): SCC INTRA frames code a
        // per-BV morph flag (av2_allow_intrabc_morph_pred reconintra.h:81 = seq intra_bawp
        // [== seq bawp, =1 in all mints] && frame allow_scc && frame_is_intra_only). ctx =
        // count of NX neighbours that are intrabc WITH morph (get_morph_pred_ctx) — the
        // `morph` nb array stores intrabc&&morph so nb_sum IS the ctx.
        let mut morph = false;
        if !inter_frame && tool_cfg.allow_scc && SEQ_TOOLS.with(|c| c.get().bawp) {
            let mctx = nb_sum(&slots, &a.morph, &l.morph) as usize;
            morph = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.morph_pred[mctx]);
            if std::env::var("MIBC").is_ok() { crate::dlog!("[MIBC] mi=({bx4},{by4}) morph ctx={mctx} v={} rng={}", morph as u8, msac.rng); }
        }

        // Neighbour-context splats for an intrabc block (dav2d set_ctx, intrabc path):
        // fsc=0, midx=0xff, mrl/multi_mrl=0, intrabc=1 (+ skip_txfm decoded above).
        splat_nb(&mut a.skip_txfm, &mut l.skip_txfm, bx4, by4, bw4, bh4, skip_txfm as u8);
        splat_nb(&mut a.intrabc, &mut l.intrabc, bx4, by4, bw4, bh4, 1);
        splat_nb(&mut a.morph, &mut l.morph, bx4, by4, bw4, bh4, morph as u8);
        splat_nb(&mut a.fsc, &mut l.fsc, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.midx, &mut l.midx, bx4, by4, bw4, bh4, 0xff);
        splat_nb(&mut a.mrl, &mut l.mrl, bx4, by4, bw4, bh4, 0);
        splat_nb(&mut a.multi_mrl, &mut l.multi_mrl, bx4, by4, bw4, bh4, 0);
        // Mark the intrabc block in the decode-order grid (avm marks ALL decoded blocks; its
        // joint mode is DC=0). Else neighbours get wrong top-right / bottom-left availability.
        if crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
            crate::av2_frame::FRAME.with(|fr| {
                let mut f = fr.borrow_mut();
                if f.pl[0].w != 0 {
                    f.ensure_sb(bx4, by4);
                    f.mark_coded(bx4, by4, bw4, bh4, 0);
                }
            });
        }

        // ===== intrabc coefficient decode (inter path) =====
        // dav decode.c:2428 reads tx_part AFTER the BV/morph section (inter ctx: intrabc
        // counts as inter). Coefs then decode PER TX UNIT; recon_intrabc loops the units.
        let tx_part = read_tx_part(msac, cdf, bd[0] as usize, bd[1] as usize, false, true, skip_txfm);
        let layout = tx_part_layout(bd[0] as usize, bd[1] as usize, tx_part);
        let v64 = std::env::var("V64").is_ok() && bx4 == 64 && by4 == 24;
        let (iw4g, ih4g) = crate::av2_frame::FRAME.with(|fr| { let f = fr.borrow(); (f.iw4, f.ih4) });
        let mut units: Vec<TxUnitCf> = Vec::new();
        let mut last = (true, -1i32, Vec::new(), DCT_DCT, 0u8);
        let mut first_txtp = DCT_DCT;
        {
            for &(ux, uy, tw4, th4) in &layout {
                let (ubx4, uby4) = (bx4 + ux, by4 + uy);
                if ubx4 >= iw4g || uby4 >= ih4g { continue; }
                let ubw4 = if iw4g > ubx4 { tw4.min(iw4g - ubx4) } else { tw4 };
                let ubh4 = if ih4g > uby4 { th4.min(ih4g - uby4) } else { th4 };
                let (slw, slh) = (tw4.trailing_zeros() as usize, th4.trailing_zeros() as usize);
                let (clw, clh) = (slw.min(3), slh.min(3));
                let t_dim_ctx = (slw + slh + 1) >> 1;
                let tx2dszctx = clw + clh;
                let all_zero = skip_txfm || {
                    let sctx = crate::av2_coef::skip_ctx_luma(&a.lcoef[ubx4..], &l.lcoef[uby4..], slw, slh, &bd) as usize;
                    if std::env::var("MTXB").is_ok() { crate::dlog!("[MTXB] mi=({ubx4},{uby4}) pl=0i txs={t_dim_ctx} skipctx={sctx} rng={}", msac.rng); }
                    let az = rav1d_msac_decode_bool_adapt(msac, &mut cdf.coef.skip[1][t_dim_ctx][sctx]);
                    if v64 { crate::dlog!("V64 all_zero={} sctx={sctx} tctx={t_dim_ctx} r={}", az as u8, msac.rng); }
                    az
                };
                let mut cf = vec![0i32; 1usize << (slw + slh + 4)];
                let mut ibc_stx = 0u8;
                let (eob, cf_ctx, luma_txtp) = if all_zero {
                    (-1, 0x40u8, DCT_DCT)
                } else {
                    let e = crate::av2_coef::decode_eob(msac, &mut cdf.coef, tx2dszctx, 1);
                    let t_dim_min = slw.min(slh);
                    let tmax = slw.max(slh);
                    let txtp = if slw.min(slh) >= 3 && tmax == 4 {
                        DCT_DCT
                    } else {
                        let ey = e >> (2 + clw);
                        let ex = e & ((4 << clw) - 1);
                        let xy = ex + ey;
                        let (tw, th) = (ubw4.min(8), ubh4.min(8));
                        let txtp_ctx = if xy < 2 {
                            1
                        } else if xy > 4 * (tw as i32 + th as i32) - 4 {
                            2
                        } else {
                            0
                        } as usize;
                        if slw == 3 && slh == 3 {
                            if rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.txtp_inter_dct_idtx[txtp_ctx][3]) {
                                DCT_DCT
                            } else {
                                IDTX_TT
                            }
                        } else if tmax >= 3 {
                            let long_dct = tmax >= 4
                                || rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.txtp_long32_dct[1]);
                            let short_idx = rav1d_msac_decode_symbol_adapt4(
                                msac, &mut cdf.m.txtp_inter_short_1d[txtp_ctx][t_dim_min], 3,
                            ) as usize;
                            TXTP_LONG_TBL[long_dct as usize][(slw < slh) as usize][short_idx]
                        } else {
                            let setidx = (slw == 2 && slh == 2) as usize;
                            let set = rav1d_msac_decode_bool_adapt(
                                msac, &mut cdf.m.txtp_inter_tx_set[setidx][txtp_ctx][t_dim_min],
                            );
                            let txtp_idx = if !set {
                                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.txtp_inter_set0[setidx][txtp_ctx], 7) as usize
                            } else if setidx == 1 {
                                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.txtp_inter_set2[txtp_ctx], 3) as usize + 8
                            } else {
                                rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.txtp_inter_set1[txtp_ctx], 7) as usize + 8
                            };
                            TXTP_INV_TBL[setidx][txtp_idx]
                        }
                    };
                    let tx_class = (txtp >> 3) & 3;
                    let scan = crate::av2_tables_gen::SCANS[scan_idx_square(clw, clh)];
                    let ctx = if txtp == IDTX_TT {
                        crate::av2_coef::decode_coefs_idtx_y(
                            msac, &mut cdf.coef, &mut cf, e, tx2dszctx, t_dim_ctx.min(2), clw, clh, scan,
                        )
                    } else {
                        let dc_sign_ctx = crate::av2_coef::get_dc_sign_ctx(
                            &a.lcoef[ubx4..], &l.lcoef[uby4..], slw, slh, ubw4 as i32, ubh4 as i32,
                        );
                        if tx_class != 0 {
                            crate::av2_coef::decode_coefs_hv_y(
                                msac, &mut cdf.coef, &mut cf, e, tx_class as usize, tx2dszctx, t_dim_ctx,
                                clw, clh, false, dc_sign_ctx,
                            )
                        } else {
                            if t_dim_min >= 2 && txtp == DCT_DCT && e >= 3 && e < 32 && SEQ_TOOLS.with(|c| c.get().ist_inter) {
                                ibc_stx = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.stx[1][t_dim_min], 3) as u8;
                            }
                            if e == 0 {
                                crate::av2_coef::decode_coefs_dc_only_y(
                                    msac, &mut cdf.coef, &mut cf, t_dim_ctx, dc_sign_ctx,
                                )
                            } else {
                                crate::av2_coef::decode_coefs_dct_y(
                                    msac, &mut cdf.coef, &mut cf, e, tx2dszctx, t_dim_ctx, clw, clh, scan,
                                    false, dc_sign_ctx,
                                )
                            }
                        }
                    };
                    (e, ctx, txtp)
                };
                splat_nb(&mut a.lcoef, &mut l.lcoef, ubx4, uby4, ubw4, ubh4, cf_ctx);
                if layout.len() > 1 {
                    units.push(TxUnitCf { ux4: ux, uy4: uy, slw, slh, cf: cf.clone(), txtp: luma_txtp, eob, stx: ibc_stx, all_zero });
                }
                if ux == 0 && uy == 0 { first_txtp = luma_txtp; }
                last = (all_zero, eob, cf, luma_txtp, ibc_stx);
            }
        }
        let (all_zero, eob, cf, luma_txtp, ibc_stx) = last;
        // For a partitioned block, LeafInfo.txtp feeds the CHROMA inherit — dav reads the
        // txtp_map at the block's TOP-LEFT cell, i.e. the FIRST unit's txtp.
        let out_txtp = if units.is_empty() { luma_txtp } else { first_txtp };
        if !skip_txfm {
            if std::env::var("SBTRACE").is_ok() { crate::dlog!("LEAFDIF ({bx4},{by4}) bs={bs} dif={:x} rng={}", msac.dif, msac.rng); }
        }
        return LeafInfo {
            intrabc: true,
            y_mode_idx: 0,
            midx: 0xff,
            fsc: false,
            mrl_index: 0,
            multi_mrl: 0,
            ibc_bv,
            ibc_morph: morph,
            all_zero,
            eob,
            cf,
            txtp: out_txtp,
            skip: skip_txfm,
            stx: ibc_stx,
            units,
        };
    }

    // intra_y_mode — set, then idx (two-level for set 0), then derive midx.
    if bx4 == 8 && by4 == 8 { crate::dlog!("Y88L pre-y_set rng={} dif={:x} intrabc={} skip={}", msac.rng, msac.dif, intrabc as u8, skip_txfm as u8); }
    let v300 = std::env::var("V320").is_ok() && bx4 == 0 && by4 == 0;
    if v300 { crate::dlog!("V300 pre-y_set r={} (dav: ccso done r=46160)", msac.rng); }
    if v300 { crate::dlog!("V300 y_set_cdf={:?} dif={:x} cnt={}", cdf.m.intra_y_set, msac.dif, msac.cnt); }
    if std::env::var("MYSET").is_ok() { crate::dlog!("[MYSET] mi=({bx4},{by4}) cell={:?} r={}", cdf.m.intra_y_set, msac.rng); }
    let y_set = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.intra_y_set, 3);
    if v300 { crate::dlog!("V300 y_set={y_set} r={}", msac.rng); }
    let y_mode_idx = if y_set == 0 {
        let mctx = mode_ctx(&a.midx, &l.midx, bx4, by4, bw4, bh4, w4, h4);
        if std::env::var("MYCTX").is_ok() { crate::dlog!("[MYCTX] mi=({bx4},{by4}) set={y_set} mctx={mctx} rng={}", msac.rng); }
        if bx4 == 8 && by4 == 8 { crate::dlog!("Y88L y_set={y_set} mctx={mctx} rng={} dif={:x}", msac.rng, msac.dif); }
        let mut idx = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.intra_y_idx0[mctx], 7) as i32;
        if idx == 7 {
            idx += rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.intra_y_idx1[mctx], 5) as i32;
        }
        idx
    } else {
        // set>=1: 4 bypass bits (dav2d `y_set*16 - 3 + bools_bypass(4)`).
        y_set as i32 * 16 - 3 + crate::msac::rav1d_msac_decode_bools_bypass(msac, 4) as i32
    };
    // midx: non-directional modes (idx<5) → 0xff; else the NEIGHBOUR-REORDERED directional index
    // (dav2d decode.c:2072 `midx = reorder[dir_y_mode_idx]`). Both the CHROMA inherit AND the luma
    // recon read this for their mode+angle, so it MUST match dav's parse-time reorder exactly.
    // Parse-safe: the midx value doesn't feed the entropy (neighbour ctx uses `!= 0xff`; chroma
    // cfl-ctx uses `== CFL_PRED`).
    let midx = if y_mode_idx < 5 {
        0xff
    } else {
        // dav2d reorder (decode.c:2027-2072): the neighbour midx comes from the PARSE context
        // arrays `l.midx` / `a.midx` (persistent per-row/col, gated on h4==bh4 / w4==bw4), NOT a 2D
        // recon-populated grid (which diverges on partition gaps / recon timing → a WRONG inherited
        // midx for chroma AND a wrong luma directional mode). Parse-safe: every consumer of this
        // midx (mode_ctx, mrl gate, chroma uv_mode_ctx) tests `!= 0xff`, never the value.
        let lmidx = if h4 == bh4 && by4 + bh4 - 1 < l.midx.len() { l.midx[by4 + bh4 - 1] } else { 0xff };
        let amidx = if w4 == bw4 && bx4 + bw4 - 1 < a.midx.len() { a.midx[bx4 + bw4 - 1] } else { 0xff };
        let bl = if lmidx != 0xff { lmidx + 5 } else { 0 };
        let ar = if amidx != 0xff { amidx + 5 } else { 0 };
        reorder_dir_joint(y_mode_idx, bl, ar, bw4 * 4, bh4 * 4) - 5
    };

    let dbg = bx4 == 0 && by4 == 4;
    if dbg {
        crate::dlog!("[dbg] mode rng={} mode_idx={y_mode_idx} (oracle mode 61504)", msac.rng);
    }
    if p60 { crate::dlog!("R60L y_set={y_set} idx={y_mode_idx} rng={} dif={:x} cnt={}", msac.rng, msac.dif, msac.cnt); }

    // fsc (forward-skip / intra IDTX) — only coded for blocks <=32px in both dims and when
    // the seq enables idtx_intra (dav2d decode.c:2088 `imax(bw4,bh4) <= 8 && idtx_intra`).
    // Larger blocks keep the default fsc=0 with no coded symbol. ctx = neighbour fsc sum,
    // sz_ctx = block-size group.
    let idtx_intra = true; // plumb from the seq header
    let fsc = if bw4.max(bh4) <= 8 && idtx_intra {
        // Inter-frame intra block (not intra_region) forces ctx=3 (dav2d decode.c:2124).
        let ctx = if inter_frame { 3 } else { nb_sum(&slots, &a.fsc, &l.fsc) as usize };
        let sz_ctx = FSC_BSIZE_GROUPS[bs] as usize;
        let v = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.fsc[ctx][sz_ctx]);
        if p60 { crate::dlog!("R60L fsc ctx={ctx} sz={sz_ctx} fsc={} rng={} dif={:x}", v as u8, msac.rng, msac.dif); }
        if v300 { crate::dlog!("V300 y_mode={y_mode_idx} fsc={} r={} (dav mode DC r=37696, fsc0 r=37624)", v as u8, msac.rng); }
        v
    } else {
        false
    };
    if dbg {
        crate::dlog!("[4x8dbg] FSCDIF dif={:x} rng={} (fsc={})", msac.dif, msac.rng, fsc as u8);
    }

    // mrl_index — only for directional modes; ctx = neighbour mrl sum. When >0, a
    // multi_line_mrl flag follows (ctx = neighbour multi_mrl sum).
    let mut multi_mrl = 0u8;
    // seq enable_mrls gates the mrl_index symbol entirely (tool-off mint: crash without).
    let mrl_index = if midx != 0xff && SEQ_TOOLS.with(|c| c.get().mrls) {
        let ctx = nb_sum(&slots, &a.mrl, &l.mrl) as usize;
        if dbg {
            crate::dlog!(
                "[4x8dbg] mrl ctx={ctx} cdf={:?} dif={:x} cnt={} rng={}",
                cdf.m.mrl_index[ctx], msac.dif, msac.cnt, msac.rng
            );
        }
        let mrl = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.m.mrl_index[ctx], 3);
        if mrl > 0 {
            let ctx2 = nb_sum(&slots, &a.multi_mrl, &l.multi_mrl) as usize;
            multi_mrl = rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.multi_mrl[ctx2]) as u8;
        }
        mrl
    } else {
        0
    };
    if dbg {
        crate::dlog!("[4x8dbg] mrl={mrl_index} multi={multi_mrl} rng={} (oracle mrl 48512, multi 60928)", msac.rng);
    }
    if std::env::var("MKEYL").is_ok() { crate::dlog!("[MKEYL2] mi=({bx4},{by4}) yidx={y_mode_idx} midx={midx} fsc={} mrl={mrl_index} rng={}", fsc as u8, msac.rng); }

    // all_zero/skip — fsc forces sctx=9; else get_skip_ctx (0 when TX covers block).
    // TX-size context (dav2d `t_dim->ctx`): average of the min-square and max-square TX
    // log2 sizes = `(slw+slh+1)>>1`. 4x4=0, 8x4=1, 8x8=1, 16x4=1, 16x16=2 (verified:
    // 16x4 → ctx 1, not max(2,0)=2).
    // Splat the per-field neighbour context for subsequent blocks (dav2d splats these mode
    // fields regardless of the coefficient decode — they feed neighbour mode contexts).
    splat_nb(&mut a.intrabc, &mut l.intrabc, bx4, by4, bw4, bh4, intrabc as u8);
    splat_nb(&mut a.morph, &mut l.morph, bx4, by4, bw4, bh4, 0);
    // dav2d always sets b->skip_txfm (0 for intra-non-intrabc) and splats it; the intra path
    // must too, else a prior intrabc block's skip_txfm=1 leaks into a later block's ctx.
    splat_nb(&mut a.skip_txfm, &mut l.skip_txfm, bx4, by4, bw4, bh4, skip_txfm as u8);
    // C2 CDEF: mark this block into the noskip mask when it carries coded residual
    // (dav2d decode.c:3517 `has_luma && !skip_txfm`). Intra non-intrabc always has skip_txfm=0.
    if !skip_txfm && crate::av2_frame::RECON_ACTIVE.with(|a| a.get()) {
        crate::av2_frame::FRAME.with(|f| f.borrow_mut().mark_noskip(bx4, by4, bw4, bh4));
    }
    splat_nb(&mut a.midx, &mut l.midx, bx4, by4, bw4, bh4, midx);
    splat_nb(&mut a.fsc, &mut l.fsc, bx4, by4, bw4, bh4, fsc as u8);
    // mrl neighbour context stores a boolean (mrl_index>0), not the raw index (dav2d `!!`).
    splat_nb(&mut a.mrl, &mut l.mrl, bx4, by4, bw4, bh4, (mrl_index != 0) as u8);
    splat_nb(&mut a.multi_mrl, &mut l.multi_mrl, bx4, by4, bw4, bh4, multi_mrl);

    // --- palette (avm read_palette_mode_info, decodemv.c:1043): a LUMA DC_PRED leaf of
    // 8x8..64x64 (FULL dims, av2_allow_palette blockd.h:3362) codes palette_y_mode when the
    // frame allows screen content. A JOINT (defer_coefs) leaf reads palette in the CALLER
    // after the chroma mode (avm order y→uv→palette); the SDP luma-tree leaf reads it here.
    let mut palette: Option<crate::av2_palette::PaletteBlock> = None;
    // Size gate = avm `bs >= BLOCK_8X8 && wide<=64 && high<=64` (blockd.h:3362). In the AV2
    // enum only 4x4/4x8/8x4 sit below BLOCK_8X8 — every extended rect (4x16..4x32..64x8)
    // qualifies — which in 4px cells is exactly w4*h4 >= 4.
    if !defer_coefs && !intrabc && tool_cfg.allow_scc && y_mode_idx == 0
        && w4 * h4 >= 4 && w4 <= 16 && h4 <= 16
        && rav1d_msac_decode_bool_adapt(msac, &mut cdf.m.pal_y)
    {
        let n = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.m.pal_sz, 6) as usize + 2;
        let bd_bits = if bdmax_g() > 255 { 10u8 } else { 8 };
        let colors = crate::av2_palette::read_palette_colors_y(msac, bd_bits, n, bx4, by4, have_left, have_top);
        if std::env::var("MPAL").is_ok() { crate::dlog!("[MPAL] mi=({bx4},{by4}) n={n} colors={:?} rng={} cnt={}", &colors[..n], msac.rng, msac.cnt); }
        palette = Some(crate::av2_palette::PaletteBlock { n, colors, map: Vec::new(), w: w4 * 4, h: h4 * 4 });
    }
    if !defer_coefs {
        // Every luma leaf refreshes the neighbour palette caches (n=0 for non-palette).
        let (pn, pc) = palette.as_ref().map_or((0, [0u16; 8]), |p| (p.n, p.colors));
        crate::av2_palette::pal_splat(bx4, by4, bw4, bh4, pn, &pc);
    }

    // YUV block: the luma coefs are decoded AFTER the chroma mode (dav2d read_coef_blocks),
    // so stop here with the luma mode. The caller decodes chroma mode then all coefs.
    if defer_coefs {
        return LeafInfo { intrabc, y_mode_idx, midx, fsc, mrl_index, multi_mrl, ibc_bv: (0, 0), ibc_morph: false, all_zero: false, eob: -1, cf: Vec::new(), txtp: DCT_DCT, skip: skip_txfm, stx: 0, units: Vec::new() };
    }
    // avm av2_visit_palette (decodeframe.c:1572): the color-index map is decoded after the
    // whole mbmi, BEFORE the tx partition + coefficients. rows/cols = within-frame-bounds
    // dims (av2_get_block_dimensions block_rows/cols) = the clamped bw4/bh4 in px.
    if let Some(p) = palette.as_mut() {
        p.map = crate::av2_palette::decode_color_map(
            msac, &mut cdf.m.pal_idx_identity, &mut cdf.m.pal_idx, p.n, p.w, p.h, bh4 * 4, bw4 * 4,
        ).expect("palette color map");
        if std::env::var("MPAL").is_ok() { crate::dlog!("[MPAL] mi=({bx4},{by4}) map done rng={} cnt={}", msac.rng, msac.cnt); }
    }

    // ===== TX PARTITION + per-unit coef/recon (dav decode.c:2328 read_tx_part + per-TX
    // recon_b_luma_tx). Under frame LARGEST mode tx_part==NONE and this is one unit ==
    // the old single-TX path (byte-identical by construction). =====
    let tx_part = read_tx_part(msac, cdf, bd[0] as usize, bd[1] as usize, fsc, false, false);
    let layout = tx_part_layout(bd[0] as usize, bd[1] as usize, tx_part);
    let mt00 = std::env::var("MTRACE").is_ok() && bx4 == 0 && by4 == 0;
    let (iw4g, ih4g) = crate::av2_frame::FRAME.with(|fr| { let f = fr.borrow(); (f.iw4, f.ih4) });
    let mut last_ret = (false, -1i32, Vec::new(), DCT_DCT, 0x40u8);
    {
        for &(ux, uy, tw4, th4) in &layout {
            let (uslw, uslh) = (tw4.trailing_zeros() as usize, th4.trailing_zeros() as usize);
            let (ubx4, uby4) = (bx4 + ux, by4 + uy);
            if ubx4 >= iw4g || uby4 >= ih4g { continue; }
            // Clamped visible unit dims (frame-edge units splat/ctx over visible cells only).
            let ubw4 = if iw4g > ubx4 { tw4.min(iw4g - ubx4) } else { tw4 };
            let ubh4 = if ih4g > uby4 { th4.min(ih4g - uby4) } else { th4 };
            let t_dim_ctx = (uslw + uslh + 1) >> 1;
            let sctx = if fsc {
                9
            } else {
                crate::av2_coef::skip_ctx_luma(&a.lcoef[ubx4..], &l.lcoef[uby4..], uslw, uslh, &bd) as usize
            };
            let skip_set = (!true || fsc) as usize; // intra=true -> !intra=false; fsc-dependent
            if mt00 { crate::dlog!("[MT00] pre-all_zero r={} skip_set={skip_set} t_dim_ctx={t_dim_ctx} sctx={sctx}", msac.rng); }
            let all_zero = rav1d_msac_decode_bool_adapt(msac, &mut cdf.coef.skip[skip_set][t_dim_ctx][sctx]);
            if mt00 { crate::dlog!("[MT00] after all_zero={} r={}", all_zero as u8, msac.rng); }
            if dbg {
                crate::dlog!("[4x8dbg] all_zero={} (sctx={sctx},tctx={t_dim_ctx}) rng={} (oracle 0/52480)", all_zero as u8, msac.rng);
            }
            let (slw, slh) = (uslw, uslh);
            let (clw, clh) = (slw.min(3), slh.min(3));
            let tx2dszctx = clw + clh;
            let mut cf = vec![0i32; 1usize << (slw + slh + 4)];
            let (eob, cf_ctx, blk_txtp, blk_stxt, blk_stxs) = if all_zero {
                (-1, 0x40u8, DCT_DCT, 0u8, 0u8)
            } else {
                let e = crate::av2_coef::decode_eob(msac, &mut cdf.coef, tx2dszctx, 0);
                if e >= (1 << (clw + clh + 4)) {
                    crate::dlog!("EOBOV bx={ubx4} by={uby4} slw={slw} slh={slh} e={e} r={}", msac.rng);
                }
                if mt00 { crate::dlog!("[MT00] after eob={e} tx2dszctx={tx2dszctx} r={}", msac.rng); }
                let scan = crate::av2_tables_gen::SCANS[scan_idx_square(clw, clh)];
                // wide_angle_remap uses the UNIT tx dims (dav remaps per tx: recon_b_luma_tx).
                let y_mode = if y_mode_idx < 5 {
                    REORDERED_NONDIR_Y_MODE[y_mode_idx as usize]
                } else {
                    wide_angle_remap_mode(
                        REORDERED_DIR_Y_MODE[(midx / 7) as usize], midx as i32 % 7 - 3, mrl_index, 4 << slw, 4 << slh,
                    )
                };
                let t_dim_min = slw.min(slh);
                let dc_sign_ctx = crate::av2_coef::get_dc_sign_ctx(
                    &a.lcoef[ubx4..], &l.lcoef[uby4..], slw, slh, ubw4 as i32, ubh4 as i32,
                );
                let (ctx, rtxtp, rstxt, rstxs) = decode_coefs_y(
                    msac, cdf, &mut cf, e, fsc, y_mode, t_dim_min, t_dim_ctx, slw, slh, tx2dszctx,
                    scan, false, dc_sign_ctx,
                );
                if mt00 { crate::dlog!("[MT00] after coefs txtp={rtxtp} r={}", msac.rng); }
                (e, ctx, rtxtp, rstxt, rstxs)
            };
            // Per-UNIT cf_ctx splat (the next unit's skip/dc-sign contexts read it).
            splat_nb(&mut a.lcoef, &mut l.lcoef, ubx4, uby4, ubw4, ubh4, cf_ctx);
            // Per-UNIT chained intra recon: this unit's prediction reads the previous units'
            // recon through the frame buffer (dav predicts per TX block).
            let (uhl, uht) = (have_left || ux > 0, have_top || uy > 0);
            recon_intra_luma(
                ubx4, uby4, slw, slh, ubw4, ubh4, y_mode_idx, midx, mrl_index, multi_mrl != 0,
                &cf, blk_txtp, blk_stxt, blk_stxs, all_zero, fsc, uhl, uht,
                tx_part >= 6 && (ux > 0 || uy > 0),
                palette.as_ref().map(|p| (p, ux * 4, uy * 4)),
            );
            last_ret = (all_zero, eob, cf, blk_txtp, cf_ctx);
        }
    }
    if std::env::var("SBTRACE").is_ok() { crate::dlog!("LEAFDIF ({bx4},{by4}) bs={bs} dif={:x} rng={}", msac.dif, msac.rng); }
    if std::env::var("BRDBG").is_ok() && bx4 >= 94 && by4 >= 48 {
        crate::dlog!("DBLUMA-RECON ({bx4},{by4}) bw4={bw4} bh4={bh4} midx={midx} ymode={y_mode_idx} intrabc={intrabc}");
    }
    let (all_zero, eob, cf, blk_txtp, _cfc) = last_ret;
    LeafInfo { intrabc, y_mode_idx, midx, fsc, mrl_index, multi_mrl, ibc_bv: (0, 0), ibc_morph: false, all_zero, eob, cf, txtp: blk_txtp, skip: skip_txfm, stx: 0, units: Vec::new() }
}

/// Splat a decoded block's neighbour value across its above (`a`) columns and left
/// (`l`) rows (dav2d `set_ctx` per-field). This is what makes the *next* block's
/// `gather_nb`/`nb_sum` non-zero — applied per context field after each `decode_b`.
pub fn splat_nb(a: &mut [u8], l: &mut [u8], bx4: usize, by4: usize, bw4: usize, bh4: usize, val: u8) {
    // HARDENING (shared writer): corrupt block geometry can run a splat past the neighbour
    // arrays — clamp the span rather than panicking (in-range spans are unchanged).
    let ae = (bx4 + bw4).min(a.len());
    let le = (by4 + bh4).min(l.len());
    if bx4 < ae { a[bx4..ae].fill(val); }
    if by4 < le { l[by4..le].fill(val); }
}

/// Intra y-mode neighbour context (dav2d): counts whether the above-right and
/// left-bottom neighbours carry a *directional* mode (`midx != 0xff`), each gated on
/// the block spanning the full coded width/height. Unavailable neighbours read the
/// cleared `0xff` and contribute 0, so no explicit edge check is needed.
pub fn mode_ctx(
    a_midx: &[u8],
    l_midx: &[u8],
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    w4: usize,
    h4: usize,
) -> usize {
    // HARDENING (shared reader): an absent/out-of-range neighbour cell reads as "unset".
    let above = (w4 == bw4 && a_midx.get(bx4 + bw4 - 1).copied().unwrap_or(0xff) != 0xff) as usize;
    let left = (h4 == bh4 && l_midx.get(by4 + bh4 - 1).copied().unwrap_or(0xff) != 0xff) as usize;
    above + left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_ctx_mixed_neighbours_is_n_not_n_plus_1() {
        // dav2d get_filter_ctx (env.h:136): when BOTH above+left neighbours match the ref but
        // carry DIFFERENT filters, the single-ref ctx is N_SWITCHABLE_FILTERS (=3), NOT N+1 (=4).
        // Regression for the (58,4) frame-2 divergence.
        let mut a = BlockNbCtx::new(128);
        let mut l = BlockNbCtx::new(128);
        // above (slot 0, xb4=10) filter=0, left (slot 1, yb4=4) filter=2, both ref0=1 (ref_=0).
        a.ref0[10] = 1;
        a.filter[10] = 0;
        l.ref0[4] = 1;
        l.filter[4] = 2;
        let ctx = get_filter_ctx(&a, &l, [(false, 10), (true, 4)], 0, false);
        assert_eq!(ctx, 3, "mixed valid filters → ctx = N (3), not N+1 (4)");
        // Matching filters → that filter's ctx.
        l.filter[4] = 0;
        assert_eq!(get_filter_ctx(&a, &l, [(false, 10), (true, 4)], 0, false), 0);
        // No matching neighbours (ref mismatch) → both flt=N → ctx = N (3).
        a.ref0[10] = 0;
        l.ref0[4] = 0;
        assert_eq!(get_filter_ctx(&a, &l, [(false, 10), (true, 4)], 0, false), 3);
    }

    #[test]
    fn doubled_left_neighbour_gives_ctx2() {
        // 4x4 at (bx4=5, by4=0): have_left, NO top (SB top). Both nb slots collapse
        // onto the left edge → fsc ctx = 2 * left_fsc. This reproduces the oracle's
        // `fsc[ctx=2]` for block #2's second leaf (left = fsc=1 block).
        let slots = gather_nb(true, false, 5, 0, 1, 1, 1, 1);
        assert!(matches!(slots[0], Some((true, 0))));
        assert!(matches!(slots[1], Some((true, 0))));
        let a = vec![0u8; 16];
        let mut l = vec![0u8; 16];
        l[0] = 1; // left neighbour fsc = 1
        assert_eq!(nb_sum(&slots, &a, &l), 2);
    }

    #[test]
    fn left_and_top_gather_distinct_slots() {
        // Interior block with both neighbours → slot0 = left-bottom, slot1 = top-right.
        let slots = gather_nb(true, true, 4, 4, 1, 1, 1, 1);
        assert!(matches!(slots[0], Some((true, 4)))); // left at by4+bh4-1 = 4
        assert!(matches!(slots[1], Some((false, 4)))); // top at bx4+bw4-1 = 4
    }

    #[test]
    fn mode_ctx_from_left_directional() {
        // 8x4 at (bx4=6, by4=0): no top (midx 0xff), left carries a directional mode
        // (the 4x4 at (5,0), midx=4) → ctx=1, matching the oracle's intra_y_mode ctx=1.
        let a_midx = vec![0xffu8; 16];
        let mut l_midx = vec![0xffu8; 16];
        l_midx[0] = 4;
        assert_eq!(mode_ctx(&a_midx, &l_midx, 6, 0, 2, 1, 2, 1), 1);
        // both cleared → ctx 0 (block #1 / leaf #1).
        assert_eq!(mode_ctx(&vec![0xff; 16], &vec![0xff; 16], 0, 0, 4, 4, 4, 4), 0);
    }

    #[test]
    fn splat_writes_both_edges() {
        let mut a = vec![0u8; 16];
        let mut l = vec![0u8; 16];
        splat_nb(&mut a, &mut l, 4, 0, 2, 1, 7);
        assert_eq!(&a[4..6], &[7, 7]);
        assert_eq!(l[0], 7);
        assert_eq!(a[6], 0); // beyond the block stays cleared
    }

    #[test]
    fn no_neighbours_gives_ctx0() {
        // Top-left of SB: no left, no top → both slots None → ctx 0 (the first block).
        let slots = gather_nb(false, false, 0, 0, 4, 4, 4, 4);
        assert!(slots[0].is_none() && slots[1].is_none());
        assert_eq!(nb_sum(&slots, &[0u8; 16], &[0u8; 16]), 0);
    }
}
