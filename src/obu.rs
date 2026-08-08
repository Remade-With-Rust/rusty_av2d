#![deny(unsafe_code)]

use std::ffi::{c_int, c_uint};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::{array, cmp, fmt, mem};

use crate::c_arc::CArc;
use crate::decode::rav1d_submit_frame;
use crate::env::get_poc_diff;
use crate::error::{Rav1dError, Rav1dResult};
use crate::getbits::GetBits;
use crate::include::common::intops::{clip_u8, ulog2};
use crate::include::dav1d::common::Rav1dDataProps;
use crate::include::dav1d::data::Rav1dData;
use crate::include::dav1d::dav1d::Rav1dDecodeFrameType;
use crate::include::dav1d::headers::{
    DRav1d, Dav1dSequenceHeader, Rav1dAdaptiveBoolean, Rav1dChromaSamplePosition,
    Rav1dColorPrimaries, Rav1dColorRange, Rav1dContentLightLevel, Rav1dFilmGrainData,
    Rav1dFilterMode, Rav1dFrameHeader, Rav1dFrameHeaderCdef, Rav1dFrameHeaderDelta,
    Rav1dFrameHeaderDeltaLF, Rav1dFrameHeaderDeltaQ, Rav1dFrameHeaderFilmGrain,
    Rav1dFrameHeaderLoopFilter, Rav1dFrameHeaderOperatingPoint, Rav1dFrameHeaderQuant,
    Rav1dFrameHeaderRestoration, Rav1dFrameHeaderSegmentation, Rav1dFrameHeaderSuperRes,
    Rav1dFrameHeaderTiling, Rav1dFrameSize, Rav1dFrameSkipMode, Rav1dFrameType, Rav1dITUTT35,
    Rav1dLoopfilterModeRefDeltas, Rav1dMasteringDisplay, Rav1dMatrixCoefficients, Rav1dObuType,
    Rav1dPixelLayout, Rav1dProfile, Rav1dRestorationType, Rav1dSegmentationData,
    Av2SeqHdr, Rav1dSegmentationDataSet, Rav1dSequenceHeader,
    Rav1dSequenceHeaderOperatingParameterInfo,
    Rav1dSequenceHeaderOperatingPoint, Rav1dTransferCharacteristics, Rav1dTxfmMode,
    Rav1dWarpedMotionParams, Rav1dWarpedMotionType, RAV1D_MAX_CDEF_STRENGTHS,
    RAV1D_MAX_OPERATING_POINTS, RAV1D_MAX_TILE_COLS, RAV1D_MAX_TILE_ROWS, RAV1D_PRIMARY_REF_NONE,
    RAV1D_REFS_PER_FRAME,
};
use crate::internal::{Rav1dContext, Rav1dState, Rav1dTileGroup, Rav1dTileGroupHeader};
use crate::levels::ObuMetaType;
use crate::msac::{
    rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bool_equi, rav1d_msac_decode_symbol_adapt4,
    rav1d_msac_decode_symbol_adapt8, MsacContext,
};
use crate::log::Rav1dLog as _;
use crate::picture::{rav1d_picture_copy_props, PictureFlags, Rav1dThreadPicture};
use crate::thread_task::FRAME_ERROR;

struct Debug {
    enabled: bool,
    name: &'static str,
    start: usize,
}

impl Debug {
    pub const fn new(enabled: bool, name: &'static str, gb: &GetBits) -> Self {
        Self {
            enabled,
            name,
            start: gb.pos(),
        }
    }

    const fn named(&self, name: &'static str) -> Self {
        let &Self {
            enabled,
            name: _,
            start,
        } = self;
        Self {
            enabled,
            name,
            start,
        }
    }

    pub fn log(&self, gb: &GetBits, msg: fmt::Arguments) {
        let &Self {
            enabled,
            name,
            start,
        } = self;
        if !enabled {
            return;
        }
        let offset = gb.pos() - start;
        println!("{name}: {msg} [off={offset}]");
    }

    pub fn post(&self, gb: &GetBits, post: &str) {
        self.log(gb, format_args!("post-{post}"));
    }
}

/// dav2d getbits.c dav2d_get_uniform: output in [0, max-1]; reads nothing when max <= 1.
fn av2_get_uniform(gb: &mut GetBits, max: u32) -> u32 {
    if max <= 1 {
        return 0;
    }
    let l = (31 - max.leading_zeros()) + 1; // ulog2(max) + 1
    let m = (1u32 << l) - max;
    let v = gb.get_bits(l as c_int - 1);
    if v < m { v } else { (v << 1) - m + gb.get_bit() as u32 }
}

/// dav2d getbits.c dav2d_get_bits_subexp_u (the AV2 variant: parameterized k, n-exclusive).
fn av2_subexp_u(gb: &mut GetBits, refv: u32, n: u32, k: i32) -> u32 {
    fn inv_recenter(r: u32, v: u32) -> u32 {
        // dav2d intops.h: even → r + v/2; odd → r - (v+1)/2
        if v > 2 * r {
            v
        } else if v & 1 == 0 {
            r + (v >> 1)
        } else {
            r - ((v + 1) >> 1)
        }
    }
    let mut v = 0u32;
    let mut i = 0i32;
    loop {
        let b = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b;
        if n <= v + 3 * a {
            v += av2_get_uniform(gb, n - v);
            break;
        }
        if !gb.get_bit() {
            v += gb.get_bits(b as c_int);
            break;
        }
        v += a;
        i += 1;
    }
    if refv * 2 <= n {
        inv_recenter(refv, v)
    } else {
        n - 1 - inv_recenter(n - 1 - refv, v)
    }
}

fn check_trailing_bits(gb: &mut GetBits, strict_std_compliance: bool) -> Rav1dResult {
    let trailing_one_bit = gb.get_bit();

    if gb.has_error() != 0 {
        return Err(Rav1dError::InvalidArgument);
    }

    if !strict_std_compliance {
        return Ok(());
    }

    if !trailing_one_bit || gb.pending_bits() != 0 {
        return Err(Rav1dError::InvalidArgument);
    }

    gb.bytealign();

    if gb.get_bytes(gb.remaining_len()).iter().any(|&b| b != 0) {
        return Err(Rav1dError::InvalidArgument);
    }

    Ok(())
}

#[inline(never)]
/// AV2 sequence-header tile info (dav2d obu.c `parse_tile_info`). For now this only
/// consumes the correct bits; the tile geometry is recomputed later when stored.
fn parse_seq_tile_info(
    gb: &mut GetBits,
    sb128: u32,
    _seq_sb128: u32,
    w: c_uint,
    h: c_uint,
    level: c_uint,
    tier: bool,
) -> (u8, u8) {
    let uniform = gb.get_bit();
    let sb128 = sb128 as c_int;
    let (w, h, level, tier) = (w as c_int, h as c_int, level as c_int, tier as c_int);
    let sbsz_min1 = (64 << sb128) - 1;
    let sbsz_log2 = 6 + sb128;
    let sbw = (w + sbsz_min1) >> sbsz_log2;
    let sbh = (h + sbsz_min1) >> sbsz_log2;
    let w_adj = (level >= 18) as c_int + (level >= 14 && tier != 0) as c_int;
    let max_tile_width_sb = 4096 >> (sbsz_log2 - w_adj);
    let sz_adj =
        (level >= 14) as c_int + (level >= 18) as c_int + (level >= 14 && tier != 0) as c_int;
    let max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2 - sz_adj);
    let min_log2_cols = tile_log2(max_tile_width_sb, sbw);
    let max_log2_cols = tile_log2(1, cmp::min(sbw, RAV1D_MAX_TILE_COLS as c_int));
    let max_log2_rows = tile_log2(1, cmp::min(sbh, RAV1D_MAX_TILE_ROWS as c_int));
    let min_log2_tiles = cmp::max(tile_log2(max_tile_area_sb, sbw * sbh), min_log2_cols);

    if uniform {
        let mut log2_cols = min_log2_cols;
        while log2_cols < max_log2_cols && gb.get_bit() {
            log2_cols += 1;
        }
        let min_log2_rows = cmp::max(min_log2_tiles as c_int - log2_cols as c_int, 0) as u8;
        let mut log2_rows = min_log2_rows;
        while log2_rows < max_log2_rows && gb.get_bit() {
            log2_rows += 1;
        }
        (log2_cols, log2_rows)
    } else {
        let (mut sbx, mut cols, mut widest_tile) = (0, 0, 0);
        while sbx < sbw && cols < RAV1D_MAX_TILE_COLS as c_int {
            let tile_width_sb = cmp::min(sbw - sbx, max_tile_width_sb);
            let tile_w = if tile_width_sb > 1 {
                1 + gb.get_uniform(tile_width_sb as c_uint) as c_int
            } else {
                1
            };
            sbx += tile_w;
            cols += 1;
            widest_tile = cmp::max(widest_tile, tile_w);
        }
        let mut max_tile_area_sb = sbw * sbh;
        if min_log2_tiles != 0 {
            max_tile_area_sb >>= min_log2_tiles as c_int + 1;
        }
        let max_tile_height_sb = cmp::max(max_tile_area_sb / widest_tile, 1);
        let (mut sby, mut rows) = (0, 0);
        while sby < sbh && rows < RAV1D_MAX_TILE_ROWS as c_int {
            let tile_height_sb = cmp::min(sbh - sby, max_tile_height_sb);
            let tile_h = if tile_height_sb > 1 {
                1 + gb.get_uniform(tile_height_sb as c_uint) as c_int
            } else {
                1
            };
            sby += tile_h;
            rows += 1;
        }
        (tile_log2(1, cols), tile_log2(1, rows))
    }
}

/// dav2d_ccso_offset[scale_idx][idx] (tables.c) — maps a parsed truncated-unary index (0..7)
/// to the signed CCSO sample offset.
const CCSO_OFFSET: [[i8; 8]; 4] = [
    [0, 1, -1, 3, -3, 7, -7, -10],
    [0, 2, -2, 6, -6, 14, -14, -20],
    [0, 3, -3, 9, -9, 21, -21, -30],
    [0, 4, -4, 12, -12, 28, -28, -40],
];

/// dav2d_ccso_quant_sz[scale_idx][quant_idx] (tables.c) — gates CCSO `edge_clf`.
const CCSO_QUANT_SZ: [[u16; 4]; 4] = [
    [16, 8, 32, 0],
    [56, 40, 64, 128],
    [48, 24, 96, 192],
    [80, 112, 160, 256],
];

/// AV2 frame header — FRONT (dav2d obu.c:1052..1130): ids, frame type from the OBU
/// type, long-term-ref id, and show flags. The rest (obu.c:1130..2045) is the next
/// pass; for now this parses the front, reports, and returns Err.
fn parse_av2_frame_hdr_front(
    seq_hdr: &Rav1dSequenceHeader,
    obu_type: Rav1dObuType,
    gb: &mut GetBits,
) -> Rav1dResult<(u32, u8)> {
    use Rav1dObuType::*;
    let id = gb.get_vlc();
    if id != 0 {
        return Err(Rav1dError::InvalidArgument);
    }
    let seqhdr_idx = gb.get_vlc();
    if seqhdr_idx != seq_hdr.av2.id {
        return Err(Rav1dError::InvalidArgument);
    }
    if obu_type == Sef {
        let _existing = gb.get_bits(seq_hdr.av2.ref_frames_log2 as c_int);
        gb.get_bit(); // FIXME poc
        crate::dlog!("[rav2d AV2 framehdr] show_existing_frame");
        return Err(Rav1dError::InvalidArgument);
    }
    // frame_type: 0=KEY, 1=INTER, 2=INTRA, 3=SWITCH
    //
    // A single-picture-header stream (AV2 `single_picture_header_flag`, carried here under
    // dav1d's inherited `reduced_still_picture_header` name) omits the frame_type symbol and
    // is a KEY frame by definition. It does NOT skip the rest of the header: base_q_idx, the
    // filter params and the tile info are all still coded. The body below is already
    // reduced-aware (see the `reduced_still_picture_header == 0` guards through it), so the
    // reduced case must FALL THROUGH into it rather than be branched around — branching
    // around it returned (yac=0, frame_type=0) and decoded every still picture to flat grey,
    // silently.
    // KNOWN LIMITATION: a single-picture-header stream parses through the header now (the
    // outer branch used to skip it entirely, silently decoding every still picture to flat
    // grey), but the parse is not yet bit-exact -- the tile-info/seq-header interaction still
    // diverges, and avm gates ~22 fields on this flag of which we mirror only some. Refuse the
    // stream rather than emit wrong pixels silently; that silence was the actual defect.
    // RUSTY_AV2D_ALLOW_SINGLE_PICTURE_HEADER=1 opts in for development on the remaining work.
    if seq_hdr.reduced_still_picture_header != 0
        && std::env::var("RUSTY_AV2D_ALLOW_SINGLE_PICTURE_HEADER").is_err()
    {
        crate::dlog!("[rav2d AV2 framehdr] single-picture-header stream: parse incomplete, refusing");
        return Err(Rav1dError::UnsupportedBitstream);
    }
    let frame_type = if seq_hdr.reduced_still_picture_header != 0 {
        0
    } else {
        match obu_type {
            ClosedLoopKf | OpenLoopKf => 0,
            Switch => 3,
            LeadingTip | Tip | Bridge => 1,
            _ => {
                if !gb.get_bit() {
                    2 // INTRA
                } else {
                    1 // INTER
                }
            }
        }
    };
    {
        // S-frames (SWITCH/RAS): restricted_prediction_switch (avm decodeframe.c:8275; dav2d
        // LACKS this bit — its s-frame segfault). When set, every current ref slot becomes
        // RESTRICTED: pending implicit outputs flush, and the slot is excluded from all later
        // implicit ref scoring (display order restarts an epoch).
        let mut restricted_prediction_switch = false;
        crate::av2_recon::CUR_IS_SFRAME.with(|c| c.set(frame_type == 3));
        if frame_type == 3 {
            restricted_prediction_switch = gb.get_bit();
            if restricted_prediction_switch {
                crate::av2_recon::REF_SLOTS.with(|s| {
                    let mut slots = s.borrow_mut();
                    for sl in slots.iter_mut().flatten() {
                        sl.restricted = true;
                    }
                });
            }
            crate::dlog!("F2HDR sframe restricted={}", restricted_prediction_switch as u8);
        }
        let _ = restricted_prediction_switch;
        let nbits_lt = seq_hdr.av2.number_of_bits_for_lt_frame_id as c_int;
        let mut ltr_id: i32 = -1;
        if frame_type == 0 {
            if nbits_lt != 0 {
                ltr_id = gb.get_bits(nbits_lt) as i32 - 1;
            }
        } else if obu_type == OpenLoopKf && nbits_lt != 0 {
            let n_ref = gb.get_bits(3);
            for _ in 0..n_ref {
                gb.get_bits(nbits_lt);
            }
        }
        let mut show_immediate = 0u32;
        let mut show_implicit = 0u32;
        if obu_type != Bridge {
            if obu_type != OpenLoopKf {
                show_immediate = gb.get_bit() as u32;
            }
            if show_immediate == 0 && seq_hdr.av2.monotonic == 0 {
                show_implicit = gb.get_bit() as u32;
            }
        }
        crate::av2_recon::AV2_SHOW.with(|c| c.set((show_immediate != 0, show_implicit != 0)));
        let _ = ltr_id;

        let is_inter_or_switch = frame_type == 1 || frame_type == 3;
        let is_tip = matches!(obu_type, LeadingTip | Tip);
        let off_base = gb.pos();
        // --- frame size override + offset + (INTER) primary ref (dav2d obu.c:1133) ---
        let mut frame_size_override = false;
        let mut frame_order_hint = 0u32;
        let mut primary_ref_frame = 7u32; // PRIMARY_REF_NONE
        let mut primary_ref_signaled = false;
        if seq_hdr.reduced_still_picture_header == 0 {
            frame_size_override = if frame_type == 3 { true } else { gb.get_bit() };
            frame_order_hint = gb.get_bits(seq_hdr.av2.order_hint_n_bits as c_int);
            if frame_type == 1 {
                primary_ref_signaled = gb.get_bit();
                if !is_tip {
                    gb.get_bit(); // cross_frame_context
                }
                if primary_ref_signaled {
                    primary_ref_frame = gb.get_bits(3);
                }
            }
        }
        crate::dlog!("F2HDR override off={} order={} ft={} tip={}", gb.pos() - off_base, frame_order_hint, frame_type, is_tip as u8);
        // --- refresh_frame_flags (dav2d obu.c:1153) ---
        let refresh_frame_flags = if obu_type == ClosedLoopKf && seq_hdr.av2.max_mlayer_id == 0 {
            (1u32 << seq_hdr.av2.ref_frames) - 1
        } else if obu_type == OpenLoopKf || seq_hdr.av2.max_mlayer_id != 0 {
            if seq_hdr.av2.short_refresh_frame_flags != 0 {
                1 << gb.get_bits(seq_hdr.av2.ref_frames_log2 as c_int)
            } else {
                gb.get_bits(seq_hdr.av2.ref_frames as c_int)
            }
        } else if frame_type != 3 && seq_hdr.av2.short_refresh_frame_flags != 0 {
            if gb.get_bit() {
                1 << gb.get_bits(seq_hdr.av2.ref_frames_log2 as c_int)
            } else {
                0
            }
        } else {
            gb.get_bits(seq_hdr.av2.ref_frames as c_int)
        };
        crate::dlog!("F2HDR refresh off={}", gb.pos() - off_base);
        // --- refs (dav2d obu.c:1179): inter/switch. Explicit map signals n_ref + refidx[]; the
        // implicit path (this clip) derives them by ref-buffer scoring with NO coded bits. The
        // scoring (get_ref_frames) is deferred; this clip is single-reference, so n_ref_frames=1
        // (verified: dav2d F2NREF n_ref_frames=1). n_ref_frames feeds the ccso refidx n_bits and
        // the later per-ref loops (brick B). ---
        let mut n_ref_frames = 0u32;
        let mut refidx = [0u8; 7];
        let mut has_bothside_refs = false;
        if is_inter_or_switch {
            if frame_type == 3 || seq_hdr.av2.explicit_ref_frame_map != 0 {
                n_ref_frames = gb.get_bits(3);
                for n in 0..n_ref_frames as usize {
                    refidx[n] = gb.get_bits(seq_hdr.av2.ref_frames_log2 as c_int) as u8;
                }
            } else {
                // Implicit ref-buffer scoring (dav2d get_ref_frames) → n_ref_frames + refidx[].
                let ohb = seq_hdr.av2.order_hint_n_bits as u32;
                let (n, ri) = crate::av2_recon::get_ref_frames(ohb, frame_order_hint);
                n_ref_frames = n;
                refidx = ri;
            }
            // has_bothside_refs (dav2d obu.c:1194-1202): any selected ref with a FUTURE order hint
            // AND any with a PAST one. Needed by the TIP block's global_wtd_idx gate.
            let ohb = seq_hdr.av2.order_hint_n_bits as u32;
            let (mut fut, mut past) = (false, false);
            crate::av2_recon::REF_SLOTS.with(|s| {
                let slots = s.borrow();
                for n in 0..n_ref_frames as usize {
                    if let Some(r) = slots[refidx[n] as usize] {
                        let d = crate::av2_recon::get_poc_diff(ohb, frame_order_hint, r.order_hint);
                        if d < 0 { fut = true; }
                        if d > 0 { past = true; }
                    }
                }
            });
            has_bothside_refs = fut && past;
            // Publish (n_ref, refidx) so the inter SB-loop can select the primary ref picture slot.
            crate::av2_recon::CUR_FRAME_REFIDX.with(|c| c.set((n_ref_frames, refidx)));
            // refdir_with_intra (dav2d decode.c:5508): index 0 = intra (0), index 1+i = ref i's
            // direction (1 = future / order_hint > current). Feeds get_comp_ctx (is_comp).
            // Alongside: refdist[i] (signed wrapped poc delta), absrefdist[i], and
            // furthest_future_refidx (the future ref with the largest delta; -2 = none) — the
            // compound joint_ctx / refine_mv / comp_type gates read these (decode.c:5501).
            let ohb = seq_hdr.av2.order_hint_n_bits as u32;
            // Index 0 = the intra slot = -1 (dav2d refdir_intra, lib.c:274).
            let mut refdir = [0i8; 9];
            refdir[0] = -1;
            refdir[8] = 1; // TIP slot (lib.c:275)
            let mut refdist = [0i32; 7];
            let mut absrefdist = [0i32; 7];
            let mut ffr: i32 = -2;
            crate::av2_recon::REF_SLOTS.with(|s| {
                let slots = s.borrow();
                for i in 0..n_ref_frames as usize {
                    if let Some(r) = slots[refidx[i] as usize] {
                        let d = crate::av2_recon::get_poc_diff(ohb, r.order_hint, frame_order_hint);
                        refdist[i] = d;
                        absrefdist[i] = d.abs();
                        refdir[i + 1] = (d > 0) as i8;
                        if d > 0 && (ffr < 0 || refdist[ffr as usize] < d) {
                            ffr = i as i32;
                        }
                    }
                }
            });
            crate::av2_recon::CUR_REFDIR.with(|c| c.set(refdir));
            crate::av2_recon::CUR_REFDIST.with(|c| c.set((refdist, absrefdist, ffr)));
            // Per-list-index ref pocs (feeds ref_flip_pair / the temporal engine).
            let mut refpoc = [0u32; 7];
            crate::av2_recon::REF_SLOTS.with(|sl| {
                let slots = sl.borrow();
                for i in 0..n_ref_frames as usize {
                    if let Some(r) = slots[refidx[i] as usize] {
                        refpoc[i] = r.order_hint;
                    }
                }
            });
            crate::av2_recon::CUR_REF_POC.with(|c| c.set((ohb, frame_order_hint, refpoc)));
        }
        if std::env::var("OHDBG").is_ok() {
            crate::dlog!("[MINE-OH] type={} order_hint={} n_ref={} bothside={} refidx={},{},{},{},{},{},{}",
                frame_type, frame_order_hint, n_ref_frames, has_bothside_refs as u8,
                refidx[0], refidx[1], refidx[2], refidx[3], refidx[4], refidx[5], refidx[6]);
            let rff = crate::av2_recon::CUR_FRAME_REF.with(|c| c.get()).1;
            crate::dlog!("[MINE-RFF] oh={frame_order_hint} refresh={rff:02x}");
        }
        crate::dlog!("F2HDR refs off={}", gb.pos() - off_base);
        // --- read_frame_size: override → explicit; else inferred from ref (= max here, 0 bits) ---
        let (width, height) = if frame_size_override {
            (
                gb.get_bits(seq_hdr.width_n_bits as c_int) + 1,
                gb.get_bits(seq_hdr.height_n_bits as c_int) + 1,
            )
        } else {
            (seq_hdr.max_width as u32, seq_hdr.max_height as u32)
        };
        crate::dlog!("F2HDR framesize off={}", gb.pos() - off_base);
        // --- refs2 (dav2d obu.c:1219): inter && !explicit → implicit scoring, 0 bits ---
        // --- refmvbits (dav2d obu.c:1235) ---
        if is_inter_or_switch {
            let mut use_ref_frame_mvs = false;
            // avm frame_might_allow_ref_frame_mvs: S-frames NEVER code the bit (temporal MVs
            // are impossible at a switch point) — forced 0.
            if seq_hdr.ref_frame_mvs != 0 && frame_type != 3 {
                use_ref_frame_mvs = gb.get_bit();
            }
            // tmvp_sample_step (dav2d obu.c:1237): bit read only if use && n_ref>1 && sb128.
            // step=2 subsamples the temporal-MV projection grid (load_tmvs sample_step).
            let tmvp_sample_step = if use_ref_frame_mvs && n_ref_frames > 1 && seq_hdr.sb128 != 0 && gb.get_bit() {
                2i32
            } else {
                1i32
            };
            // --- TIP frame block (dav2d obu.c:1246-1296). For a non-OBU_TIP inter frame with
            // seq tip enabled + n_ref>1 + use_ref_frame_mvs, read tip.frame_mode + opfl_refine_type
            // + (if frame_mode) tip_hole_fill/global_wtd_idx. Missing this desynced the tip=1
            // frames. `has_bothside_refs` is approximated false (P-chain: all refs precede) — a
            // stream with bidirectional refs needs the real ref order-hint analysis. ---
            let (seq_tip, seq_hole_fill, seq_opfl, seq_refine_mv, seq_tip_refine) =
                crate::av2_recon::SEQ_TIP.with(|c| c.get());
            let mut tip_frame_mode = 0u8;
            let mut frame_tip_hole_fill = false;
            let mut tip_gmv = (0i32, 0i32);
            let mut tip_subpel_filter = 2u8; // SHARP default
            let mut tip_global_wtd_idx = 0u32;
            // opfl_refine_type (dav2d obu.c:1254): seq opfl_refine<3 → the seq value with 0 bits;
            // else coded: 1 bit → switchable(1), else another bit → all(2)/none(0).
            let mut opfl_refine_type = if seq_opfl < 3 { seq_opfl } else { 0 };
            if seq_tip != 0 && n_ref_frames > 1 && use_ref_frame_mvs {
                if is_tip {
                    // OBU_TIP / OBU_LEADING_TIP: frame_mode FORCED 2 with no bit; opfl derived
                    // (dav2d obu.c:1249-1251).
                    tip_frame_mode = 2;
                    opfl_refine_type = 2 * ((seq_opfl != 0 && seq_tip_refine) as u8);
                } else {
                    tip_frame_mode = gb.get_bit() as u8; // 1: TIP-as-ref, 0: disabled
                    if seq_opfl >= 3 {
                        opfl_refine_type = if gb.get_bit() { 1 } else { 2 * gb.get_bit() as u8 };
                    }
                }
                if tip_frame_mode != 0 {
                    if seq_hole_fill {
                        frame_tip_hole_fill = gb.get_bit(); // tip.hole_fill
                    }
                    if !has_bothside_refs || !seq_tip_refine || (seq_opfl == 0 && !seq_refine_mv) {
                        tip_global_wtd_idx = gb.get_bits(3); // tip.global_wtd_idx
                    }
                    if tip_frame_mode == 2 {
                        // gmv + subpel filter (dav2d obu.c:1267-1279).
                        if !gb.get_bit() {
                            let mut gy = gb.get_bits(4) as i32;
                            let mut gx = gb.get_bits(4) as i32;
                            if gy != 0 && gb.get_bit() { gy = -gy; }
                            if gx != 0 && gb.get_bit() { gx = -gx; }
                            tip_gmv = (gy, gx);
                        }
                        tip_subpel_filter = if gb.get_bit() { 2 } else if gb.get_bit() { 0 } else { 1 };
                    }
                }
                // find_tip_ref_frames: ref selection only, no coded bits.
            } else if seq_opfl >= 3 {
                opfl_refine_type = if gb.get_bit() { 1 } else { 2 * gb.get_bit() as u8 };
            }
            let _ = tip_global_wtd_idx; // cwp table idx (dav2d_tip_wts) — 0 (=8, EQUAL) in corpus
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.tip_frame_mode = tip_frame_mode;
                cfg.opfl_refine_type = opfl_refine_type;
                cfg.tip_subpel_filter = tip_subpel_filter;
                cfg.tip_gmv = tip_gmv;
                c.set(cfg);
            });
            // Temporal MV projection setup (dav refmvs_init_frame temporal part): mfmv list +
            // scale arrays + tip refs/sf + window allocation. Per-SB-row projection runs in the
            // SB loop (load_tmvs).
            {
                let (nb2, poc2, refpoc2) = crate::av2_recon::CUR_REF_POC.with(|c| c.get());
                let seq_mv_traj2 = crate::av2_recon::SEQ_COMP.with(|c| c.get()).4;
                crate::av2_refmvs::tmvs_setup(
                    nb2, poc2, n_ref_frames as usize, &refidx, &refpoc2,
                    use_ref_frame_mvs, seq_mv_traj2, seq_tip != 0,
                    tip_frame_mode, frame_tip_hole_fill,
                    ((width + 3) / 4) as usize, ((height + 3) / 4) as usize,
                    tmvp_sample_step,
                );
            }
            // --- TIP-as-output (frame_mode==2, dav2d obu.c:1298-1351): the header ENDS here.
            // Parse the fm2 tail (deblock sub_pu/apply_filter + the derived qp), install the
            // skipped sections' defaults (single tile, all filters off, disable_cdf_update
            // semantics via pri/sec derive), and return early. ---
            if tip_frame_mode == 2 {
                let mut apply_filter = false;
                if seq_hdr.av2.db_sub_pu != 0 {
                    let sub_pu = gb.get_bit();
                    if sub_pu {
                        apply_filter = gb.get_bit();
                    }
                }
                let yac2: u16 = if crate::av2_recon::SEQ_TIP_QP.with(|c| c.get()) {
                    crate::dlog!("[rav2d AV2 framehdr] fm2 tip_explicit_qp — unimplemented");
                    return Err(Rav1dError::InvalidArgument);
                } else {
                    let tr = crate::av2_refmvs::TMVS.with(|c| c.borrow().tip_ref);
                    crate::av2_recon::REF_SLOTS.with(|sl| {
                        let sl = sl.borrow();
                        let q0 = sl[refidx[tr.0 as usize] as usize].as_ref().map_or(0, |r| r.qidx);
                        let q1 = sl[refidx[tr.1 as usize] as usize].as_ref().map_or(0, |r| r.qidx);
                        (q0 + q1 + 1) >> 1
                    })
                };
                if apply_filter {
                    crate::dlog!("[rav2d AV2 framehdr] fm2 tip.apply_filter — sub-PU deblock port pending");
                    return Err(Rav1dError::InvalidArgument);
                }
                crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                    let mut cfg = c.get();
                    cfg.tip_apply_filter = apply_filter;
                    c.set(cfg);
                });
                crate::av2_recon::CUR_FRAME_REF.with(|c| c.set((frame_order_hint, refresh_frame_flags, yac2, width, height)));
                {
                    let ohb = seq_hdr.av2.order_hint_n_bits as u32;
                    let (r0, r1) = crate::av2_recon::derive_pri_sec_ref(ohb, frame_order_hint, yac2, n_ref_frames, &refidx);
                    crate::av2_recon::CUR_PRIMARY_REF.with(|c| c.set(r0 as u8));
                    crate::av2_recon::CUR_SECONDARY_REF.with(|c| c.set(if r1 != r0 { r1 as u8 } else { r0 as u8 }));
                }
                crate::av2_recon::TILE_INFO.with(|c| {
                    let mut t = c.get();
                    t.cols = 1;
                    t.rows = 1;
                    t.log2_cols = 0;
                    t.log2_rows = 0;
                    c.set(t);
                });
                crate::av2_frame::DEBLOCK_CFG.with(|c| c.set(Default::default()));
                crate::av2_frame::CDEF_CFG.with(|c| { let mut f = c.get(); f.enabled = false; c.set(f); });
                crate::av2_frame::CCSO_CFG.with(|c| { let mut f = c.borrow().clone(); f.enabled = false; *c.borrow_mut() = f; });
                crate::av2_frame::GDF_CFG.with(|c| { let mut f = c.get(); f.enabled = false; c.set(f); });
                crate::dlog!("[rav2d AV2 framehdr] *** TIP fm2 FRAME HEADER COMPLETE *** yac={yac2}");
                return Ok((yac2 as u32, frame_type as u8));
            }
        }
        crate::dlog!("F2HDR refmvbits off={}", gb.pos() - off_base);
        let _ = (width, height);

        // --- screen content tools (dav2d obu.c:1354) ---
        let allow_scc = if seq_hdr.av2.screen_content_tools == 2 {
            gb.get_bit()
        } else {
            seq_hdr.av2.screen_content_tools != 0
        };
        // frame force_integer_mv (avm read_screen_content_params): coded only when scc is on
        // and the seq flag is adaptive(2); else it IS the seq value when scc is on.
        let force_integer_mv_frame = if allow_scc {
            if seq_hdr.av2.force_integer_mv == 2 {
                gb.get_bit()
            } else {
                seq_hdr.av2.force_integer_mv != 0
            }
        } else {
            false
        };
        crate::av2_recon::HDR_TOOL_CFG.with(|c| {
            let mut cfg = c.get();
            cfg.allow_scc = allow_scc;
            cfg.force_integer_mv = force_integer_mv_frame as u8;
            c.set(cfg);
        });
        crate::dlog!("F2HDR scc off={}", gb.pos() - off_base);
        // --- intrabc (obu.c:1367) ---
        // dav2d: allow_global read only if IS_KEY_OR_INTRA; then
        //   allow_local = !allow_global || get_bit()  — the `||` short-circuits, so the
        //   local bit is read exactly when allow_global == 1 (NOT when !allow_global).
        let allow_intrabc = gb.get_bit();
        // Plumb the parsed flag to the block-decode scaffold (HdrToolCfg): streams
        // that disable intrabc must not read the per-block intrabc symbol.
        crate::av2_recon::HDR_TOOL_CFG.with(|c| {
            let mut cfg = c.get();
            cfg.allow_intrabc = allow_intrabc;
            c.set(cfg);
        });
        if allow_intrabc {
            let is_key_or_intra = frame_type == 0 || frame_type == 2;
            let allow_global_intrabc = is_key_or_intra && gb.get_bit();
            let _allow_local_intrabc = !allow_global_intrabc || gb.get_bit();
            if seq_hdr.av2.allow_max_bvp_drl_bits != 0 {
                // max_bvp_drl_bits = get_ref_uniform(3, def_max_bvp_drl_bits) + 1
                crate::dlog!("[rav2d AV2 framehdr] intrabc+max_bvp (get_ref_uniform) — deferred");
                return Err(Rav1dError::InvalidArgument);
            }
        }
        crate::dlog!("F2HDR ibc off={}", gb.pos() - off_base);
        // --- frametype-specific bits (dav2d obu.c:1387): inter/switch only ---
        if is_inter_or_switch {
            // max_drl_bits: allow_frame_max_drl_bits ? get_ref_uniform(3,def)+1 : def
            //   (fdrlbits:0 here → 0 bits; plumb allow_frame_max_drl_bits from seq).
            // mv_precision (dav2d obu.c:1392): !force_integer_mv → (get_bit ? 2 : 1+2*get_bit) ∈
            // {1,2,3}; force_integer_mv → 0. The value seeds the per-block `mv_prec = 3+this` +
            // the mvprec_rem table index — MUST be stored, not discarded (v432 f2=2, v320 f2=3).
            let mv_precision: u8 = if !force_integer_mv_frame {
                if gb.get_bit() { 2 } else { 1 + 2 * gb.get_bit() as u8 }
            } else {
                0
            };
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.mv_precision = mv_precision;
                c.set(cfg);
            });
            // subpel_filter_mode (dav2d obu.c:1393): get_bit ? SWITCHABLE(4) : get_bits(2). The
            // value gates the per-block interp-filter symbol — MUST be stored, not discarded
            // (default 4 = switchable; a fixed-filter stream like v432 frame-2 codes 0=Regular).
            let subpel_filter_mode: u8 = if gb.get_bit() { 4 } else { gb.get_bits(2) as u8 };
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.subpel_filter_mode = subpel_filter_mode;
                c.set(cfg);
            });
            // motion_modes: frame_motion_modes_present ? per-mode bits : seq (fmm:0 here → 0 bits).
        }
        let disable_cdf_update = gb.get_bit();
        crate::dlog!("F2HDR cdfupd off={}", gb.pos() - off_base);
        let _ = (allow_scc, allow_intrabc, disable_cdf_update);

        // --- parse_tile_info_frmhdr (obu.c:964): keyframe at full res => reuse_allowed,
        // so frame tiling == seq tiling. Size-mismatch fresh-parse handled too. ---
        let (t_log2_cols, t_log2_rows);
        if seq_hdr.av2.tiling_present != 0
            && (seq_hdr.av2.tiling_present == 1 || gb.get_bit())
        {
            t_log2_cols = seq_hdr.av2.tiling_log2_cols;
            t_log2_rows = seq_hdr.av2.tiling_log2_rows;
        } else {
            let frame_sb128 = (seq_hdr.sb128 != 0) as u32;
            (t_log2_cols, t_log2_rows) = parse_seq_tile_info(
                gb,
                frame_sb128,
                seq_hdr.sb128 as u32,
                width,
                height,
                seq_hdr.av2.level as c_uint,
                seq_hdr.av2.tier != 0,
            );
        }
        crate::dlog!("F2HDR tiling0 off={} TILEGEO log2c={} log2r={} w={} h={}", gb.pos() - off_base, t_log2_cols, t_log2_rows, width, height);
        let mut tiling_n_bytes = 1u8;
        if t_log2_cols != 0 || t_log2_rows != 0 {
            if seq_hdr.av2.avg_cdf_type == 0 {
                gb.get_bits((t_log2_cols + t_log2_rows) as c_int); // tiling.update
            }
            tiling_n_bytes = gb.get_bits(2) as u8 + 1;
        }
        // Tile grid geometry (dav2d parse_tile_info UNIFORM branch, obu.c:127-160): the tile
        // distribution runs over FULL SBs only (`fsbw = max(1,(w+7)>>seq_sbsz_log2)`), tile_w =
        // max(1, fsbw>>log2), remainder spread +1 over the FIRST tiles; the terminator entry is
        // the rounded-UP extent. Non-uniform seq tiling isn't minted in the corpus yet — this
        // recompute assumes uniform (the seq bit consumption was already handled at parse time).
        {
            let sbl2 = 6 + seq_hdr.sb128 as usize;
            let sbw = ((width as usize) + (1 << sbl2) - 1) >> sbl2;
            let sbh = ((height as usize) + (1 << sbl2) - 1) >> sbl2;
            let fsbw = (((width as usize) + 7) >> sbl2).max(1);
            let fsbh = (((height as usize) + 7) >> sbl2).max(1);
            let sb4 = 1usize << (sbl2 - 2); // SB size in 4px units
            let mut ti = crate::av2_recon::TileInfo::single();
            ti.log2_cols = t_log2_cols;
            ti.log2_rows = t_log2_rows;
            ti.n_bytes = tiling_n_bytes;
            let tile_w = (fsbw >> t_log2_cols).max(1);
            let mut extra = fsbw.saturating_sub(tile_w << t_log2_cols) as i32;
            let (mut cols, mut sbx) = (0usize, 0usize);
            while sbx < fsbw {
                ti.col_start4[cols] = (sbx * sb4) as u16;
                sbx += tile_w + (extra > 0) as usize;
                extra -= 1;
                cols += 1;
            }
            ti.cols = cols as u8;
            ti.col_start4[cols] = (sbw * sb4) as u16;
            let tile_h = (fsbh >> t_log2_rows).max(1);
            let mut extra = fsbh.saturating_sub(tile_h << t_log2_rows) as i32;
            let (mut rows, mut sby) = (0usize, 0usize);
            while sby < fsbh {
                ti.row_start4[rows] = (sby * sb4) as u16;
                sby += tile_h + (extra > 0) as usize;
                extra -= 1;
                rows += 1;
            }
            ti.rows = rows as u8;
            ti.row_start4[rows] = (sbh * sb4) as u16;
            crate::av2_recon::TILE_INFO.with(|c| c.set(ti));
            // Tile-adaptive filter units (avm get_ccso_unit_size_log2_adaptive_tile +
            // init_gdf): non-last tile spans not 4/2-SB-divisible shrink the CCSO unit
            // (256px → SB..128px); any odd-SB span with 64px SBs shrinks GDF (128→64px).
            let (mut e2, mut e4) = (0usize, 0usize);
            for i in 0..cols - 1 {
                let s = ((ti.col_start4[i + 1] - ti.col_start4[i]) as usize) / sb4;
                e2 += s & 1;
                e4 += s & 3;
            }
            for i in 0..rows - 1 {
                let s = ((ti.row_start4[i + 1] - ti.row_start4[i]) as usize) / sb4;
                e2 += s & 1;
                e4 += s & 3;
            }
            let multi = cols > 1 || rows > 1;
            let ccso_log2 = if !multi {
                8
            } else if sbl2 < 8 {
                if e4 == 0 { (sbl2 + 2).min(8) } else if e2 == 0 { (sbl2 + 1).min(8) } else { sbl2 }
            } else {
                sbl2
            };
            let gdf_bs4 = if multi && seq_hdr.sb128 == 0 && e2 > 0 { 16 } else { 32 };
            crate::av2_recon::FILTER_UNITS.with(|c| c.set((1usize << (ccso_log2 - 2), gdf_bs4)));
        }

        crate::dlog!("F2HDR tiling off={}", gb.pos() - off_base);
        // --- quant: yac (obu.c:1434) ---
        let yac = gb.get_bits(8 + (seq_hdr.hbd != 0) as c_int);
        // Record this frame's reference metadata (applied to REF_SLOTS after it decodes). qidx=yac
        // (the ref scoring reads each ref's yac). refresh_frame_flags selects which slots to update.
        crate::av2_recon::CUR_FRAME_REF.with(|c| c.set((frame_order_hint, refresh_frame_flags, yac as u16, width, height)));
        // primary/secondary_ref_frame (dav2d obu.c:1461): derive_pri_sec_ref yields the two best
        // non-key/intra refs; primary = refs[0] when not signaled; secondary = refs[refs[1] !=
        // primary] whenever primary != NONE. The secondary drives the 7:1 pri/sec CDF average.
        let mut secondary_ref_frame = 7u32;
        if is_inter_or_switch {
            let ohb = seq_hdr.av2.order_hint_n_bits as u32;
            let (r0, r1) = crate::av2_recon::derive_pri_sec_ref(
                ohb, frame_order_hint, yac as u16, n_ref_frames, &refidx,
            );
            if !primary_ref_signaled {
                primary_ref_frame = r0 as u32;
            }
            if primary_ref_frame != 7 {
                secondary_ref_frame =
                    if r1 as u32 != primary_ref_frame { r1 as u32 } else { r0 as u32 };
            }
        }
        if std::env::var("PREFDBG").is_ok() {
            crate::dlog!("[PREF] poc={frame_order_hint} signaled={primary_ref_signaled} resolved={primary_ref_frame} sec={secondary_ref_frame}");
        }
        // Publish primary+secondary so the SB-loop can inherit/average the CDF (dav2d 5394-5413).
        crate::av2_recon::CUR_PRIMARY_REF.with(|c| c.set(primary_ref_frame as u8));
        crate::av2_recon::CUR_SECONDARY_REF.with(|c| c.set(secondary_ref_frame as u8));
        let _ = (t_log2_cols, t_log2_rows);
        let is_i400 = seq_hdr.layout == Rav1dPixelLayout::I400;
        // quant DC/AC deltas (obu.c:1435). uac/vac feed the chroma deblock thresholds.
        // The DC deltas (ydc/udc/vdc) select a SEPARATE DC quantizer per plane (dav2d decode.c:128
        // `ydc = clip(yac + ydc_delta)`, `dq[.][.][0] = dq_lookup(ydc)`): the DC coefficient (index
        // 0) of every block dequantizes with dq_lookup(yac+dc_delta), NOT dq_lookup(yac).
        let mut uac_delta = 0i32;
        let mut vac_delta = 0i32;
        let (mut ydc_delta, mut udc_delta, mut vdc_delta) = (0i32, 0i32, 0i32);
        if seq_hdr.av2.ydc_dq_enabled != 0 && gb.get_bit() {
            ydc_delta = gb.get_sbits(7) as i32; // ydc_delta
        }
        if !is_i400 && (seq_hdr.av2.uvdc_dq_enabled != 0 || seq_hdr.av2.uvac_dq_enabled != 0) {
            let diff_uv_delta = seq_hdr.av2.separate_uv_delta_q != 0 && gb.get_bit();
            if seq_hdr.av2.uvdc_dq_enabled != 0 && gb.get_bit() {
                udc_delta = gb.get_sbits(7) as i32; // udc
            }
            if seq_hdr.av2.uvac_dq_enabled != 0 && gb.get_bit() {
                uac_delta = gb.get_sbits(7) as i32; // uac
            }
            if diff_uv_delta {
                if seq_hdr.av2.uvdc_dq_enabled != 0 && gb.get_bit() {
                    vdc_delta = gb.get_sbits(7) as i32; // vdc
                }
                if seq_hdr.av2.uvac_dq_enabled != 0 && gb.get_bit() {
                    vac_delta = gb.get_sbits(7) as i32; // vac
                }
            } else {
                vac_delta = uac_delta; // v shares u when not separately signalled
                vdc_delta = udc_delta;
            }
        }
        // DC quantizer per plane = dq_lookup(clip(yac + dc_delta, 0, qmax)). qmax = 255 (8-bit;
        // hbd adds 48). Stored for the coefficient dequant (DC uses this, AC uses dq_lookup(yac)).
        let qmax = 255 + 48 * (seq_hdr.hbd != 0) as i32;
        let clipq = |d: i32| (yac as i32 + d).clamp(0, qmax) as u32;
        crate::av2_frame::F2_DCQ.with(|c| c.set([
            crate::av2_dequant::dq_lookup(clipq(ydc_delta)),
            crate::av2_dequant::dq_lookup(clipq(udc_delta)),
            crate::av2_dequant::dq_lookup(clipq(vdc_delta)),
        ]));
        crate::av2_frame::F2_ACQ.with(|c| c.set([
            crate::av2_dequant::dq_lookup(clipq(0)),
            crate::av2_dequant::dq_lookup(clipq(uac_delta)),
            crate::av2_dequant::dq_lookup(clipq(vac_delta)),
        ]));
        crate::av2_frame::F2_QDELTAS.with(|c| c.set([ydc_delta, udc_delta, vdc_delta, uac_delta, vac_delta]));
        if std::env::var("QDBG").is_ok() {
            crate::dlog!("[QDBG] yac={yac} ydc_d={ydc_delta} udc_d={udc_delta} vdc_d={vdc_delta} uac_d={uac_delta} vac_d={vac_delta}");
        }
        // [IS_INTER_OR_SWITCH secondary_ref derive: keyframe skips]
        // segmentation (obu.c:1477)
        let seg_enabled = gb.get_bit();
        if seg_enabled {
            let reuse = seq_hdr.av2.seg_info_present != 0
                && (seq_hdr.av2.seg_adaptive == 0 || gb.get_bit());
            if !reuse {
                let n_seg = 8u32 << seq_hdr.av2.seg_ext as u32; // parse_seg_info
                for _ in 0..n_seg {
                    if gb.get_bit() {
                        gb.get_sbits(10);
                    }
                    gb.get_bit(); // skip
                    gb.get_bit(); // globalmv
                }
            }
            // keyframe: primary_ref==NONE => update_map=1, no bits (inter reads update_map/temporal)
        }
        // quantizer matrix (obu.c:1510)
        let qm_enabled = gb.get_bit();
        if std::env::var("MHDRQ").is_ok() { crate::dlog!("[MHDRQ] yac={yac} qm={}", qm_enabled as u8); }
        if qm_enabled {
            // avm setup_qm_params (decodeframe.c:3697): per-set qm levels; predefined matrices
            // (the decoder-default qm_list; no QM OBU minted so far — user matrices are a loud
            // unsupported error at the OBU parse).
            let qm_num = if seg_enabled { gb.get_bits(2) + 1 } else { 1 };
            if qm_num > 1 {
                crate::dlog!("[rav2d AV2] pic_qm_num > 1 (segmented QM) unsupported");
                return Err(Rav1dError::InvalidArgument);
            }
            for _ in 0..qm_num {
                let qy = gb.get_bits(4) as u8; // qm.y
                let (mut qu, mut qv) = (qy, qy);
                if !is_i400 && !gb.get_bit() {
                    qu = gb.get_bits(4) as u8; // qm.u
                    qv = if seq_hdr.av2.separate_uv_delta_q != 0 {
                        gb.get_bits(4) as u8 // qm.v
                    } else {
                        qu
                    };
                }
                crate::av2_qm::set_frame_qm(true, qy, qu, qv);
            }
        } else {
            crate::av2_qm::set_frame_qm(false, 15, 15, 15);
        }
        // delta q (obu.c:1534) — capture present + res_log2 for the per-SB delta-q parse.
        let (dq_present, dq_res_log2) = if yac != 0 && gb.get_bit() {
            (true, gb.get_bits(2) as u8)
        } else {
            (false, 0)
        };
        crate::av2_recon::HDR_TOOL_CFG.with(|c| {
            let mut cfg = c.get();
            cfg.delta_q_present = dq_present;
            cfg.delta_q_res_log2 = dq_res_log2;
            c.set(cfg);
        });
        let _ = (seg_enabled, qm_enabled);

        // lossless: no-segmentation => qidx==yac, so all_lossless iff yac==0 (general delta TODO)
        let all_lossless = yac == 0;
        // tcq / parity (obu.c:1564)
        if !all_lossless {
            let tcq = if seq_hdr.av2.tcq == 2 {
                gb.get_bit() as u32
            } else {
                seq_hdr.av2.tcq as u32
            };
            if tcq == 0 && seq_hdr.av2.parity_hiding != 0 {
                gb.get_bit(); // parity_hiding
            }
        }
        // deblock (obu.c:1575); INTER reads the lf_sub_pu bit first when seq db_sub_pu set.
        let mut lf_sub_pu = false;
        if !all_lossless {
            if frame_type == 1 && seq_hdr.av2.db_sub_pu != 0 {
                lf_sub_pu = gb.get_bit();
                if std::env::var("MHDRQ").is_ok() { crate::dlog!("[MHDRQ] lf_sub_pu={}", lf_sub_pu as u8); }
            }
            let level_y0 = gb.get_bit();
            let level_y1 = gb.get_bit();
            let mut level_u = false;
            let mut level_v = false;
            if !is_i400 && (level_y0 || level_y1) {
                level_u = gb.get_bit();
                level_v = gb.get_bit();
            }
            let bits = seq_hdr.av2.df_par_bits as c_int;
            // Offset-binary q-index deltas (dav2d obu.c:1587): value = get_bits(bits) - (1<<(bits-1)).
            // Bit consumption is unchanged from the prior discard path (parse stays bit-exact);
            // y1 inherits y0 when its present-bit is 0.
            let off = 1i32 << (bits - 1);
            let dq_y0 = if level_y0 && gb.get_bit() { gb.get_bits(bits) as i32 - off } else { 0 };
            let dq_y1 = if level_y1 {
                if gb.get_bit() { gb.get_bits(bits) as i32 - off } else { dq_y0 }
            } else {
                0
            };
            let dq_u = if level_u && gb.get_bit() { gb.get_bits(bits) as i32 - off } else { 0 };
            let dq_v = if level_v && gb.get_bit() { gb.get_bits(bits) as i32 - off } else { 0 };
            crate::av2_frame::DEBLOCK_CFG.with(|c| {
                c.set(crate::av2_frame::DeblockCfg {
                    level_y0, level_y1, level_u, level_v, dq_y0, dq_y1, dq_u, dq_v,
                    uac_delta, vac_delta, sub_pu: lf_sub_pu,
                });
            });
        }
        if is_inter_or_switch {
            crate::dlog!("F2HDR deblock off={}", gb.pos() - off_base);
        }
        // gdf (obu.c:1609)
        if !all_lossless && seq_hdr.av2.gdf != 0 {
            let gdf_bs = 128i32 << (seq_hdr.sb128 == 2) as i32;
            let gdf_enabled = seq_hdr.reduced_still_picture_header != 0 || gb.get_bit();
            let mut gdf_mode = 0i32;
            // A gdf-DISABLED frame must CLEAR the cfg — GDF_CFG is a thread-local that otherwise
            // keeps the previous frame's enabled state (v432_8f oh=7 has post-gdf[0]).
            crate::av2_frame::GDF_CFG.with(|c| {
                let mut g = c.get();
                g.enabled = false;
                c.set(g);
            });
            if gdf_enabled {
                // gdf_mode = 1 (ON) + block-control bit (only when frame > gdf_bs) → 2 = ADAPTIVE.
                gdf_mode = 1;
                if cmp::max(width as i32, height as i32) > gdf_bs {
                    gdf_mode += gb.get_bit() as i32;
                }
                let qp_idx = gb.get_bits(2) as i32; // gdf_pic_qp_idx (qp_idx_offset)
                let scale_idx = gb.get_bits(2) as i32; // gdf_pic_scale_idx (scale)
                // gdf_block_size: SB-matched or max(SB, 128). This clip = 128 (default).
                crate::av2_frame::GDF_CFG.with(|c| {
                    c.set(crate::av2_frame::GdfCfg { enabled: true, mode: gdf_mode, qp_idx, scale_idx, block_size: 128 });
                });
            }
            // The per-block gdf symbol is coded ONLY in ADAPTIVE mode (mode==2; dav2d decode.c:1808
            // `gdf.enabled == DAV2D_ADAPTIVE`). ON(1)/OFF(0) code no symbol. v432_8f f1 = ON(1) →
            // mine used to decode a spurious gdf; golden/v432 keyframe = ADAPTIVE(2) → unchanged.
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.gdf = gdf_mode == 2;
                c.set(cfg);
            });
        }
        // cdef (obu.c:1630)
        let mut cdef_enabled = false;
        if !all_lossless && seq_hdr.cdef != 0 {
            cdef_enabled = seq_hdr.reduced_still_picture_header != 0 || gb.get_bit();
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.cdef = cdef_enabled;
                c.set(cfg);
            });
            if cdef_enabled {
                let damping = gb.get_bits(2) as i32 + 3;
                let n_strengths = gb.get_bits(3) as usize + 1;
                let on_skiptx = if seq_hdr.av2.cdef_on_skiptx == 2 {
                    gb.get_bit()
                } else {
                    seq_hdr.av2.cdef_on_skiptx != 0
                };
                let mut y_strength = [0i32; 8];
                let mut uv_strength = [0i32; 8];
                for i in 0..n_strengths {
                    let yb = 6 - 4 * gb.get_bit() as c_int;
                    y_strength[i] = gb.get_bits(yb) as i32;
                    if !is_i400 {
                        let uvb = 6 - 4 * gb.get_bit() as c_int;
                        uv_strength[i] = gb.get_bits(uvb) as i32;
                    }
                }
                crate::av2_frame::CDEF_CFG.with(|c| {
                    c.set(crate::av2_frame::CdefCfg {
                        enabled: true, damping, n_strengths, on_skiptx, y_strength, uv_strength,
                    });
                });
            }
        }
        // A cdef-DISABLED frame must CLEAR the cfg (same stale-thread-local class; the driver
        // was saved only by the empty cdef_idx guard).
        if !cdef_enabled {
            crate::av2_frame::CDEF_CFG.with(|c| { let mut f = c.get(); f.enabled = false; c.set(f); });
        }

        // ===== loop restoration (dav2d obu.c:1657): per-plane types, NS-Wiener class
        // params (with temporal inheritance from a ref slot), unit sizes, and the frame
        // filter banks (grouped refs + subexp-coded coefficients). =====
        {
            let lr_off0 = gb.pos();
            let mut lr = crate::av2_lr::LrFrameCfg::default();
            if !all_lossless && seq_hdr.restoration != 0 {
                use crate::av2_lr::{LrPlane, REST_NONE, REST_NS};
                let (mask0, mask1) = crate::av2_lr::SEQ_RST_MASK.with(|c| c.get());
                let n_bits_ref = if n_ref_frames <= 2 {
                    n_ref_frames.saturating_sub(1)
                } else {
                    1 + (31 - (n_ref_frames - 1).leading_zeros())
                };
                for p in 0..3usize {
                    let disable_mask = if p == 0 { mask0 } else { mask1 };
                    let mut pd = LrPlane::default();
                    pd.r_type = if disable_mask == 0 {
                        gb.get_bits(2) as u8
                    } else if disable_mask == 3 {
                        REST_NONE
                    } else {
                        gb.get_bit() as u8 * (3 - disable_mask)
                    };
                    if pd.r_type >= REST_NS {
                        pd.ffon = gb.get_bit();
                        if pd.ffon {
                            if is_inter_or_switch {
                                pd.temporal = gb.get_bit();
                            }
                            if pd.temporal {
                                let mut r = 0u32;
                                if n_bits_ref > 0 {
                                    r = gb.get_bits(n_bits_ref as c_int);
                                    if r >= n_ref_frames {
                                        return Err(Rav1dError::InvalidArgument);
                                    }
                                }
                                pd.refidx = r as u8;
                                let slot = refidx[(r as usize).min(6)] as usize;
                                let inh = crate::av2_lr::LR_SLOT.with(|s| s.borrow()[slot].clone());
                                let refcfg = inh.ok_or(Rav1dError::InvalidArgument)?;
                                let rpd = if !refcfg.p[p].ffon && p != 0 { &refcfg.p[3 - p] } else { &refcfg.p[p] };
                                if !rpd.ffon {
                                    return Err(Rav1dError::InvalidArgument);
                                }
                                pd.num_classes_idx = rpd.num_classes_idx;
                                pd.num_classes = rpd.num_classes;
                            } else {
                                let val = gb.get_bits(3) as i32;
                                pd.num_classes_idx = val as u8;
                                pd.num_classes = (1 + val + (val - 3).max(0) + (val - 5).max(0) * 2) as u8;
                            }
                        } else {
                            pd.num_classes_idx = 0;
                            pd.num_classes = 1;
                        }
                    }
                    lr.p[p] = pd;
                }
                let fsb128 = seq_hdr.sb128 as u8;
                lr.unit_size[0] = 9;
                if lr.p[0].r_type != 0 {
                    if gb.get_bit() {
                        lr.unit_size[0] -= 1;
                    } else if fsb128 < 2 && !gb.get_bit() {
                        lr.unit_size[0] -= 2 + (fsb128 == 0 && !gb.get_bit()) as u8;
                    }
                }
                let ss = (seq_hdr.layout != Rav1dPixelLayout::I444) as u8;
                lr.unit_size[1] = 9 - ss;
                if lr.p[1].r_type != 0 || lr.p[2].r_type != 0 {
                    if gb.get_bit() {
                        lr.unit_size[1] -= 1;
                    } else if fsb128 < 2 && !gb.get_bit() {
                        lr.unit_size[1] -= 2 + (fsb128 == 0 && !gb.get_bit()) as u8;
                    }
                }
                // ---- frame filter banks (dav obu.c:1741) ----
                for p in 0..3usize {
                    if !lr.p[p].ffon {
                        continue;
                    }
                    let n_feat = 16 + 2 * (p != 0) as usize;
                    let plane_mask = if p == 0 { mask0 } else { mask1 };
                    let n_classes = lr.p[p].num_classes as usize;
                    let n_ref_filters: usize = if plane_mask & 1 != 0 { 16 } else { 48 - n_classes };
                    if lr.p[p].temporal {
                        let slot = refidx[(lr.p[p].refidx as usize).min(6)] as usize;
                        let refcfg = crate::av2_lr::LR_SLOT.with(|s| s.borrow()[slot].clone())
                            .ok_or(Rav1dError::InvalidArgument)?;
                        let rpd = if !refcfg.p[p].ffon && p != 0 { refcfg.p[3 - p].clone() } else { refcfg.p[p].clone() };
                        for n in 0..n_classes {
                            lr.p[p].filter[n] = rpd.filter[n];
                        }
                        continue;
                    }
                    // collect candidate ref filters from the ref slots' frame banks
                    let mut ref_filters: Vec<[i8; 18]> = Vec::new();
                    for r in 0..n_ref_frames as usize {
                        let slot = refidx[r.min(6)] as usize;
                        let refcfg = match crate::av2_lr::LR_SLOT.with(|s| s.borrow()[slot].clone()) {
                            Some(c) => c,
                            None => continue,
                        };
                        // dir walk: p=0 -> [0]; p=1 -> [1, 2]; p=2 -> [2, 1]
                        let mut p2 = p as i32;
                        let mut dir = [0i32, 1, -1][p];
                        loop {
                            let rpd = &refcfg.p[p2 as usize];
                            if rpd.ffon {
                                let take = (n_ref_filters - ref_filters.len()).min(rpd.num_classes as usize);
                                for n in 0..take {
                                    ref_filters.push(rpd.filter[n]);
                                }
                            }
                            if dir == 0 {
                                break;
                            }
                            p2 += dir;
                            dir = 0;
                        }
                    }
                    let n_filters: usize = if plane_mask & 1 != 0 { 16 } else { 64 };
                    let grp_cnt = [n_classes, ref_filters.len(), n_filters - n_classes - ref_filters.len()];
                    let mut grp_ref_cnt = [0usize; 3];
                    let mut pred_grp: usize = 2 - (grp_cnt[1] > 2) as usize;
                    let nnz_grps = 1 + (grp_cnt[1] != 0) as usize + (grp_cnt[2] != 0) as usize;
                    let mut filter_refs = [0usize; 64];
                    for n in 0..n_classes {
                        let group: usize = if nnz_grps == 1 || !gb.get_bit() {
                            pred_grp
                        } else if nnz_grps == 2 {
                            2 - (grp_cnt[2] == 0) as usize - pred_grp
                        } else if gb.get_bit() {
                            2 - (pred_grp == 2) as usize
                        } else {
                            (pred_grp == 0) as usize
                        };
                        grp_ref_cnt[group] += 1;
                        if grp_ref_cnt[group] + (group < pred_grp) as usize > grp_ref_cnt[pred_grp] {
                            pred_grp = group;
                        }
                        let base = grp_cnt[0] * (group != 0) as usize + grp_cnt[1] * (group == 2) as usize;
                        let range = if group != 0 { grp_cnt[group] } else { n + 1 };
                        filter_refs[n] = base + if range == 1 {
                            0
                        } else {
                            av2_subexp_u(gb, (range >> 1) as u32, range as u32, 4) as usize
                        };
                        if std::env::var("MLRH").is_ok() {
                            crate::dlog!("[MLRP] n={n} group={group} pred={pred_grp} base={base} range={range} fref={} off={}", filter_refs[n], gb.pos() - lr_off0);
                        }
                    }
                    let mut exact_match_mask = 0u32;
                    for n in 0..n_classes {
                        exact_match_mask |= (gb.get_bit() as u32) << n;
                    }
                    const SHUFFLED_INDEX: [u8; 64] = [
                        16, 7, 58, 21, 12, 61, 26, 38, 18, 30, 50, 45, 23, 49, 43, 62,
                        42, 54, 27, 36, 17, 44, 32, 34, 4, 24, 52, 31, 37, 11, 33, 19,
                        35, 6, 22, 53, 63, 25, 41, 47, 1, 59, 0, 28, 40, 55, 48, 8,
                        5, 51, 9, 46, 56, 60, 15, 2, 13, 14, 57, 29, 3, 20, 39, 10,
                    ];
                    for n in 0..n_classes {
                        let r = filter_refs[n];
                        let ref_filter: [i8; 18] = if r == 0 {
                            [0; 18]
                        } else if r < n_classes {
                            lr.p[p].filter[r - 1]
                        } else if r < n_classes + grp_cnt[1] {
                            ref_filters[r - n_classes]
                        } else {
                            let mut f = [0i8; 18];
                            let src = &crate::av2_lr_tables::WIENER_NS_FILTERS
                                [SHUFFLED_INDEX[r - n_classes - grp_cnt[1]] as usize];
                            f[..16].copy_from_slice(src);
                            f
                        };
                        if std::env::var("MLRH").is_ok() {
                            crate::dlog!("[MLRR] n={n} r={r} exact={} ref_filter={:?}", (exact_match_mask >> n) & 1, &ref_filter[..6]);
                        }
                        if exact_match_mask & (1 << n) != 0 {
                            lr.p[p].filter[n] = ref_filter;
                            continue;
                        }
                        lr.p[p].filter[n] = [0; 18];
                        let mut s = 0usize;
                        while s < 3 - (p != 0) as usize {
                            if !gb.get_bit() {
                                break;
                            }
                            s += 1;
                        }
                        if std::env::var("MLRH").is_ok() {
                            crate::dlog!("[MLRS] n={n} s={s} off={}", gb.pos() - lr_off0);
                        }
                        let mask: u32 = if p != 0 {
                            crate::av2_lr_tables::SUBSET_MASKS_UV[s]
                        } else {
                            crate::av2_lr_tables::SUBSET_MASKS_Y[s]
                        };
                        for i in 0..n_feat {
                            if mask & (1 << i) == 0 {
                                continue;
                            }
                            let (nbits, lo) = if p != 0 {
                                (crate::av2_lr_tables::NS_WIENER_COEF_RANGE_UV[i][0], crate::av2_lr_tables::NS_WIENER_COEF_RANGE_UV[i][1])
                            } else {
                                (crate::av2_lr_tables::NS_WIENER_COEF_RANGE_Y[i][0], crate::av2_lr_tables::NS_WIENER_COEF_RANGE_Y[i][1])
                            };
                            let refv = (ref_filter[i] as i32 - lo as i32) as u32;
                            lr.p[p].filter[n][i] = (av2_subexp_u(gb, refv, 1u32 << nbits, nbits as i32 - 3) as i32
                                + lo as i32) as i8;
                        }
                    }
                }
                if std::env::var("MLRH").is_ok() {
                    crate::dlog!("[MLRH] restoration y={} u={} v={} ffon={} ncls={} usz={} nbits={}", lr.p[0].r_type, lr.p[1].r_type, lr.p[2].r_type, lr.p[0].ffon as u8, lr.p[0].num_classes, lr.unit_size[0], gb.pos() - lr_off0);
                    if lr.p[0].ffon {
                        for n in 0..lr.p[0].num_classes as usize {
                            crate::dlog!("[MLRF] p=0 n={n}: {:?}", &lr.p[0].filter[n][..16]);
                        }
                    }
                }
            }
            crate::av2_lr::LR_CFG.with(|c| *c.borrow_mut() = lr);
        }
        if is_inter_or_switch {
            crate::dlog!("F2HDR cdef off={}", gb.pos() - off_base);
        }
        // CCSO (obu.c:1852); keyframe skips the IS_INTER reuse block (reuse stays 0)
        if !all_lossless && seq_hdr.av2.ccso != 0 {
            let ccso_enabled = seq_hdr.reduced_still_picture_header != 0 || gb.get_bit();
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.ccso = ccso_enabled;
                c.set(cfg);
            });
            // A ccso-DISABLED frame must CLEAR the cfg — CCSO_CFG is a thread-local that
            // otherwise keeps the previous frame's enabled state + offsets (same latent-state
            // class as the GDF fix above; exposed by the tipfm2 clip's disabled deep-B frames).
            if !ccso_enabled {
                crate::av2_frame::CCSO_CFG.with(|c| *c.borrow_mut() = Default::default());
            }
            if ccso_enabled {
                let n_planes = if is_i400 { 1 } else { 3 };
                let mut ccfg = crate::av2_frame::CcsoCfg {
                    enabled: true,
                    p: vec![crate::av2_frame::CcsoPlaneCfg::default(); 3],
                };
                for p in 0..n_planes {
                    if !gb.get_bit() {
                        continue; // ccso.p[p].enabled == 0
                    }
                    // INTER/SWITCH: per-plane CCSO reuse of a ref frame's filter (dav2d obu.c:1860):
                    // reuse + sb_reuse flags, then refidx (n_bits = 0 here as n_ref_frames=1; in
                    // general 1+ulog2(n_ref_frames-1)). When `reuse`, the filter is inherited from
                    // the ref — no further bits are coded. When `sb_reuse`, the per-SB on/off flags
                    // are inherited from the ref frame's ccso map (not decoded per-SB).
                    let mut sb_reuse_flag = false;
                    let mut reuse_slot = 0u8;
                    // avm read_ccso: the reuse/sb_reuse bits are gated `!intra_only && !sframe`
                    // — an S-frame (random access) always codes the FULL LUTs, no bits.
                    if is_inter_or_switch && frame_type != 3 {
                        let reuse = gb.get_bit();
                        let sb_reuse = gb.get_bit();
                        sb_reuse_flag = sb_reuse;
                        if reuse || sb_reuse {
                            // refidx (dav2d obu.c:1864): n_bits = n_ref_frames<=2 ? n_ref_frames-1
                            // : 1+ulog2(n_ref_frames-1). The coded value is a LIST index; the
                            // consuming slot = refidx[list index].
                            let n_bits = if n_ref_frames <= 2 {
                                n_ref_frames.saturating_sub(1)
                            } else {
                                1 + (31 - (n_ref_frames - 1).leading_zeros())
                            };
                            let r = if n_bits > 0 { gb.get_bits(n_bits as c_int) } else { 0 };
                            reuse_slot = refidx[(r as usize).min(6)];
                        }
                        if reuse {
                            // dav2d obu.c ~1912: on `reuse` the plane's filter config is INHERITED
                            // from the REF SLOT's saved ccso config (memcpy of bo_only..filter_off).
                            let inherited = crate::av2_frame::CCSO_SLOT_CFG.with(|c| {
                                c.borrow()[reuse_slot as usize].as_ref().and_then(|cfg| cfg.p.get(p).cloned())
                            });
                            if let Some(mut pc) = inherited {
                                pc.enabled = true;
                                pc.sb_reuse = sb_reuse_flag;
                                pc.reuse_slot = reuse_slot;
                                ccfg.p[p] = pc;
                            }
                            continue;
                        }
                    }
                    let bo_only = gb.get_bit();
                    let si = gb.get_bits(2) as usize; // scale_idx
                    let mut edge_clf = false;
                    let mut ext_filter = 0usize;
                    let mut quant_step = 0i32;
                    let max_band_log2;
                    if bo_only {
                        max_band_log2 = gb.get_bits(3);
                    } else {
                        let qi = gb.get_bits(2) as usize; // quant_idx
                        ext_filter = gb.get_bits(3) as usize;
                        if ext_filter == 7 {
                            return Err(Rav1dError::InvalidArgument);
                        }
                        quant_step = CCSO_QUANT_SZ[si][qi] as i32;
                        if quant_step != 0 {
                            edge_clf = gb.get_bit();
                        }
                        max_band_log2 = gb.get_bits(2);
                    }
                    let n_edge_off_intervals = if bo_only { 1 } else { 3 - edge_clf as u32 };
                    let max_band = 1u32 << max_band_log2;
                    // filter_offset LUT, indexed (band<<4)|(cls0<<2)|cls1; parse order (cls0,cls1,band).
                    let mut filter_offset = vec![0i8; ((max_band as usize) << 4) + 16];
                    for cls0 in 0..n_edge_off_intervals {
                        for cls1 in 0..n_edge_off_intervals {
                            for band in 0..max_band {
                                // truncated-unary index (0..7) into CCSO_OFFSET[scale_idx]
                                let mut idx = 0usize;
                                while idx < 7 {
                                    if !gb.get_bit() {
                                        break;
                                    }
                                    idx += 1;
                                }
                                let lut = ((band as usize) << 4) | ((cls0 as usize) << 2) | (cls1 as usize);
                                filter_offset[lut] = CCSO_OFFSET[si][idx];
                            }
                        }
                    }
                    ccfg.p[p] = crate::av2_frame::CcsoPlaneCfg {
                        enabled: true,
                        bo_only,
                        quant_step,
                        ext_filter,
                        edge_clf,
                        max_band_log2,
                        filter_offset,
                        sb_reuse: sb_reuse_flag,
                        reuse_slot,
                    };
                }
                if std::env::var("CCSODBG2").is_ok() {
                    for pp in 0..3 {
                        let pc = &ccfg.p[pp];
                        crate::dlog!("[MCCSO] poc={frame_order_hint} p={pp} en={} sbreuse={} slot={} boonly={} quant={} eclf={} maxband={} off=[{} {} {} {} {} {} {} {}]",
                            pc.enabled as u8, pc.sb_reuse as u8, pc.reuse_slot, pc.bo_only as u8,
                            pc.quant_step, pc.edge_clf as u8, pc.max_band_log2,
                            pc.filter_offset.first().copied().unwrap_or(0), pc.filter_offset.get(1).copied().unwrap_or(0),
                            pc.filter_offset.get(2).copied().unwrap_or(0), pc.filter_offset.get(3).copied().unwrap_or(0),
                            pc.filter_offset.get(16).copied().unwrap_or(0), pc.filter_offset.get(17).copied().unwrap_or(0),
                            pc.filter_offset.get(18).copied().unwrap_or(0), pc.filter_offset.get(19).copied().unwrap_or(0));
                    }
                }
                crate::av2_frame::CCSO_CFG.with(|c| *c.borrow_mut() = ccfg);
            }
        }
        if is_inter_or_switch {
            crate::dlog!("F2HDR ccso off={}", gb.pos() - off_base);
        }
        // tx mode (obu.c:1930)
        if !all_lossless {
            let sw = gb.get_bit(); // txfm_mode (switchable / largest)
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.tx_switchable = sw;
                c.set(cfg);
            });
        } else {
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.tx_switchable = false;
                c.set(cfg);
            });
        }
        // modebits inter block (obu.c:1935): comp_refs + skip_mode (always), bawp (seq bawp=1),
        // warp_motion (seq motion_modes & (1<<MM_WARP_DELTA), set here). (Both gates true for this
        // clip — plumb seq bawp / motion_modes if a stream disables them.)
        if is_inter_or_switch {
            let switchable_comp_refs = gb.get_bit(); // reference_select — gates the per-block is_comp
            // avm av2_setup_skip_mode_allowed: frame_is_sframe → allowed=0 → the header bit is
            // NOT read and no per-block skip_mode symbols exist (mvref_common.c:4605).
            let skip_mode_enabled = if frame_type != 3 { gb.get_bit() } else { false };
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.skip_mode_enabled = skip_mode_enabled;
                cfg.switchable_comp_refs = switchable_comp_refs;
                c.set(cfg);
            });
            // frame enable_bawp (avm decodeframe.c:9537): the bit exists ONLY when the SEQ
            // enables bawp (else no bit — a header desync if read unconditionally). Coded for
            // s-frames too, but av2_allow_bawp (reconinter.h:1191) returns 0 for every s-frame
            // block — so the per-block gate is (seq && bit && !sframe).
            let bawp = crate::av2_recon::SEQ_TOOLS.with(|c| c.get().bawp) && gb.get_bit() && frame_type != 3;
            // warp_motion (dav2d obu.c:1943): the per-block `allow_warp` symbol is gated on THIS
            // frame flag (decode.c:2965), not the seq motion_modes. v432_8f f1 = 0 (no warp); the
            // 2-frame clips = 1 → unchanged. MUST store, not discard.
            let warp_motion = gb.get_bit();
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.bawp = bawp;
                cfg.warp_motion = warp_motion;
                c.set(cfg);
            });
        }
        gb.get_bits(2); // reduced_txtp_set (obu.c:1944)
        if is_inter_or_switch {
            crate::dlog!("F2HDR modebits off={}", gb.pos() - off_base);
        }
        // global motion (obu.c:1960): `seqhdr.global_motion && get_bit` → gmv params. seq
        // global_motion=0 for this clip → 0 bits (short-circuit). Plumb the seq flag + the
        // per-ref warp params (needs n_ref_frames) when a stream enables global motion.
        // film grain (dav obu.c:2073): seq-gated + only for SHOWN frames (immediate or
        // implicit); present bit → FGM table id (3b) + per-frame seed (16b). The tables
        // themselves arrive in the FGM OBU (type 23).
        crate::av2_grain::CUR_GRAIN.with(|c| c.set(None));
        {
            let (show_imm, show_impl) = crate::av2_recon::AV2_SHOW.with(|c| c.get());
            if seq_hdr.film_grain_present != 0 && (show_imm || show_impl) {
                let present = seq_hdr.reduced_still_picture_header != 0 || gb.get_bit();
                if present {
                    let id = gb.get_bits(3) as u8;
                    let seed = gb.get_bits(16) as u16;
                    crate::av2_grain::CUR_GRAIN.with(|c| c.set(Some((id, seed))));
                }
            }
        }
        crate::dlog!(
            "[rav2d AV2 framehdr] *** FRAME HEADER COMPLETE *** bit_error={} hdr_bits={}",
            gb.has_error() != 0,
            gb.pos()
        );
        return Ok((yac, frame_type as u8)); // base Y-AC q-index + frame_type (1=INTER)
    }
    Ok((0, frame_type as u8))
}

#[allow(unreachable_code)] // AV2 port in progress: the AV1 body below is being replaced
fn parse_seq_hdr(
    gb: &mut GetBits,
    strict_std_compliance: bool,
) -> Rav1dResult<Rav1dSequenceHeader> {
    let debug = Debug::new(false, "SEQHDR", gb);

    // ===================== AV2 sequence header (porting dav2d obu.c:201) ======================
    // MILESTONE: parse through max_width/max_height and report, to prove the AV2 OBU framing +
    // early seq-header bit-parse is bit-exact. The tool-flag tail (dav2d obu.c:283..681), the
    // seg/tiling sub-parsers, and full `Rav1dSequenceHeader` construction are the next pass.
    {
        let id = gb.get_vlc();
        let profile = gb.get_bits(5); // AV2: 5 bits (AV1 was 3)
        // AV2 codes 4:2:2 as profile_idc 3 and 4:4:4 as 4 (dav2d obu.c:214 rejects >2 — an
        // oracle LIMITATION, not a spec rule; avmdec decodes them. Accept ≤4 here).
        if profile > 4 {
            return Err(Rav1dError::InvalidArgument);
        }
        let reduced_still_picture_header = gb.get_bit();
        let level = gb.get_bits(5);
        let tier = if level >= 4 && !reduced_still_picture_header {
            gb.get_bit()
        } else {
            false
        };
        // layout: VLC index → dav2d_layouts[] = {I420, I400, I444, I422}
        let layout_idx = gb.get_vlc();
        if layout_idx > 3 {
            return Err(Rav1dError::InvalidArgument);
        }
        let layout = [1u32, 0, 3, 2][layout_idx as usize];
        // Publish the chroma subsampling for the AV2 decode path (I400 keeps the 420 default —
        // its chroma planes are simply absent).
        crate::av2_frame::SS.with(|c| c.set(match layout {
            3 => (0, 0),      // I444
            2 => (1, 0),      // I422
            _ => (1, 1),      // I420 / I400
        }));
        // bit depth: VLC; for hbd<2 the spec xors with 1 (8→0,10→1 swap), hbd==2 ⇒ 12-bit
        let mut hbd = gb.get_vlc();
        if hbd > 2 {
            return Err(Rav1dError::InvalidArgument);
        }
        if hbd < 2 {
            hbd ^= 1;
        }
        let mut max_tlayer_id = 0u32;
        let mut max_mlayer_id = 0u32;
        let mut monotonic = 1u32; // reduced_still_picture_header => 1
        if !reduced_still_picture_header {
            let _lcr_id = gb.get_bits(3);
            let _still_picture = gb.get_bit();
            max_tlayer_id = gb.get_bits(2);
            max_mlayer_id = gb.get_bits(3);
            monotonic = gb.get_bit() as u32;
        }
        let width_n_bits = gb.get_bits(4) + 1;
        let height_n_bits = gb.get_bits(4) + 1;
        let max_width = gb.get_bits(width_n_bits as c_int) + 1;
        // (The former blanket >512px reject is gone: 640x320 is byte-identical. The real
        // limitation is dimension ALIGNMENT, enforced per-frame by `dims_supported` below.)
        if max_width > 32768 {
            return Err(Rav1dError::InvalidArgument);
        }
        let max_height = gb.get_bits(height_n_bits as c_int) + 1;
        if max_height > 32768 {
            return Err(Rav1dError::InvalidArgument);
        }

        // crop window (dav2d obu.c:283)
        if gb.get_bit() {
            gb.get_vlc();
            gb.get_vlc();
            gb.get_vlc();
            gb.get_vlc(); // left, right, top, bottom
        }
        // decoder model (298)
        if !reduced_still_picture_header {
            if gb.get_bit() {
                gb.get_bits(4); // max_initial_display_delay
            }
            if gb.get_bit() {
                // decoder_model_info_present
                gb.get_bits(32); // num_units_in_decoding_tick
                gb.get_vlc(); // max_decoder_buffer_delay
                gb.get_vlc(); // max_encoder_buffer_delay
            }
        }
        // temporal/material layer dependencies (319)
        if max_tlayer_id != 0 && gb.get_bit() {
            for n in 1..max_tlayer_id {
                gb.get_bits(n as c_int);
            }
        }
        if max_mlayer_id != 0 && gb.get_bit() {
            for n in 1..max_mlayer_id {
                gb.get_bits(n as c_int);
            }
        }
        // superblock size (355): 256 / 128 / 64
        let sb128 = if gb.get_bit() { 2 } else { gb.get_bit() as u32 };
        // partition flags (362)
        let is_i400 = layout == 0;
        if !is_i400 && gb.get_bit() && !reduced_still_picture_header {
            gb.get_bit(); // ext_sdp
        }
        if gb.get_bit() {
            gb.get_bit(); // uneven_4way_partitions
        }
        let _max_pb_aspect_ratio_log2 = if gb.get_bit() {
            1 + gb.get_bit() as u32
        } else {
            3
        };
        // segmentation (380)
        let seg_ext = gb.get_bit();
        let mut seg_info_present = false;
        let mut seg_adaptive = false;
        if gb.get_bit() {
            seg_info_present = true;
            seg_adaptive = gb.get_bit();
            let n_seg = 8u32 << seg_ext as u32; // parse_seg_info(8 << ext)
            for _ in 0..n_seg {
                if gb.get_bit() {
                    gb.get_sbits(10); // delta_q (clip is non-syntactic)
                }
                gb.get_bit(); // skip
                gb.get_bit(); // globalmv
            }
        }
        // intra tools (393)
        gb.get_bit(); // intra_dip
        let seq_edge_filter = gb.get_bit(); // intra_edge_filter
        let seq_mrls = gb.get_bit(); // mrls
        crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.edge_filter = seq_edge_filter; t.mrls = seq_mrls; c.set(t); });
        let seq_cfl = gb.get_bit(); // cfl
        // Plumb to the block decode: seq cfl=0 must suppress the per-leaf is_cfl bool.
        crate::av2_recon::HDR_TOOL_CFG.with(|c| {
            let mut cfg = c.get();
            cfg.cfl = seq_cfl;
            c.set(cfg);
        });
        if !is_i400 {
            let cfl_ds = gb.get_bits(2) as u8; // cfl_ds_filter_index
            crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                let mut cfg = c.get();
                cfg.cfl_ds_filter = cfl_ds;
                c.set(cfg);
            });
        }
        gb.get_bit(); // mhccp
        let seq_ibp = gb.get_bit(); // ibp
        crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.ibp = seq_ibp; c.set(t); });
        // inter tools (414)
        let mut order_hint_n_bits = 0u32;
        let mut ref_frame_mvs = 0u8;
        let mut db_sub_pu = 0u8;
        // Compound seq flags for the per-block compound parse (SEQ_COMP).
        let mut seq_masked_compound = false;
        let mut seq_num_same_ref_comp = 0u8;
        let mut seq_cwp = false;
        let mut motion_modes = 1u32; // 1 << MM_TRANSLATION
        if !reduced_still_picture_header {
            let mut n = 2u32;
            while n <= 16 {
                if gb.get_bit() {
                    motion_modes |= n;
                }
                n <<= 1;
            }
            if motion_modes & !1 != 0 {
                let fmm = gb.get_bit(); // frame_motion_modes_present
                if fmm {
                    // Frame headers would then code per-mode enable bits (unimplemented — no
                    // avmenc path mints this). Fail loudly instead of desyncing silently.
                    crate::dlog!("[rav2d AV2] frame_motion_modes_present=1 unsupported");
                    return Err(Rav1dError::InvalidArgument);
                }
            }
            let mut six_param_warp = false;
            if motion_modes & (1 << 3) != 0 {
                six_param_warp = gb.get_bit(); // six_param_warp_delta (MM_WARP_DELTA=3)
            }
            crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.motion_modes = motion_modes; t.six_param_warp = six_param_warp; c.set(t); });
            seq_masked_compound = gb.get_bit();
            ref_frame_mvs = gb.get_bit() as u8;
            if ref_frame_mvs != 0 {
                gb.get_bit(); // reduced_ref_frame_mvs_mode
            }
            order_hint_n_bits = gb.get_bits(4) + 1;
        }
        // references (443)
        let seq_refmvbank = gb.get_bit(); // refmv_bank
        // drl_reorder (avm decodeframe.c:6290): bit1=1 -> DISABLED(my 0); else bit2=1 ->
        // CONSTRAINT (my 1: reorder at nearest>=4), bit2=0 -> ALWAYS (my 2: nearest>=2 —
        // what every corpus stream codes; SEQDBG-verified).
        let seq_drl_reorder: u8 = if gb.get_bit() { 0 } else if gb.get_bit() { 1 } else { 2 };
        crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.refmvbank = seq_refmvbank; t.drl_reorder = seq_drl_reorder; c.set(t); });
        let mut number_of_bits_for_lt_frame_id = 0u32;
        let mut explicit_ref_frame_map = 0u8;
        // dav2d obu.c:446-452: default 2; in !reduced, signalled (bit ? bits(4)+1 : 8).
        let mut ref_frames = 2u32;
        if !reduced_still_picture_header {
            explicit_ref_frame_map = gb.get_bit() as u8;
            ref_frames = if gb.get_bit() { gb.get_bits(4) + 1 } else { 8 };
            number_of_bits_for_lt_frame_id = gb.get_bits(3);
            gb.get_uniform(5); // def_max_drl_bits (+1)
            gb.get_bit(); // allow_frame_max_drl_bits
        }
        let def_max_bvp_drl_bits = gb.get_uniform(3) + 1;
        let allow_max_bvp_drl_bits = gb.get_bit();
        if !reduced_still_picture_header {
            seq_num_same_ref_comp = gb.get_bits(2) as u8;
        }
        // TIP + inter1 (478)
        let mut tip = 0u32;
        let mut tip_hole_fill = false;
        let mut seq_mv_traj = false;
        if !reduced_still_picture_header {
            tip = if gb.get_bit() { 1 + gb.get_bit() as u32 } else { 0 };
            if tip != 0 {
                tip_hole_fill = gb.get_bit(); // tip_hole_fill
            }
            seq_mv_traj = gb.get_bit(); // mv_traj
        }
        let seq_bawp = gb.get_bit(); // bawp (gates the FRAME bawp bit + intra-bawp/morph)
        if !reduced_still_picture_header {
            seq_cwp = gb.get_bit();
            gb.get_bit(); // imp_msk_bld
            db_sub_pu = gb.get_bit() as u8;
            if tip == 1 && db_sub_pu != 0 {
                crate::av2_recon::SEQ_TIP_QP.with(|c| c.set(gb.get_bit())); // tip_explicit_qp
            }
        }
        // inter2 (506)
        let mut short_refresh_frame_flags = 0u8;
        if !reduced_still_picture_header {
            let opfl_refine = gb.get_bits(2);
            let refine_mv = gb.get_bit();
            let mut tip_refine_mv = false;
            if tip != 0 && (opfl_refine != 0 || refine_mv) {
                tip_refine_mv = gb.get_bit(); // tip_refine_mv
            }
            // Stash the seq TIP flags for the frame tip block (dav2d obu.c:1246-1296).
            crate::av2_recon::SEQ_TIP.with(|c| c.set((tip as u8, tip_hole_fill, opfl_refine as u8, refine_mv, tip_refine_mv)));
            gb.get_bit(); // bru
            let seq_amvd = gb.get_bit(); // adaptive_mvd (gates the per-block amvd symbol)
            let seq_signd = gb.get_bit(); // mvd_sign_derive (gates MVD sign derivation)
            crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.adaptive_mvd = seq_amvd; t.mvd_sign_derive = seq_signd; c.set(t); });
            gb.get_bit(); // flex_mvres
            gb.get_bit(); // global_motion
            short_refresh_frame_flags = gb.get_bit() as u8;
        }
        // screen content tools (535)
        let mut screen_content_tools = 2u32; // ADAPTIVE (reduced_still_picture default)
        let mut force_integer_mv = 2u32; // ADAPTIVE
        if !reduced_still_picture_header {
            screen_content_tools = if gb.get_bit() { 2 } else { gb.get_bit() as u32 };
            force_integer_mv = if screen_content_tools != 0 {
                if gb.get_bit() {
                    2
                } else {
                    gb.get_bit() as u32
                }
            } else {
                2
            };
        }
        crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.bawp = seq_bawp; c.set(t); });
        // Size the neighbour contexts from THIS sequence's max frame dims (lifts the old
        // fixed-128 / 512px scaffold cap).
        crate::av2_recon::set_nb_len((max_width as usize + 3) / 4, (max_height as usize + 3) / 4);
        // transform-group tools (551)
        let fsc = gb.get_bit();
        if !fsc {
            gb.get_bit(); // idtx_intra = fsc || bit
        }
        let seq_ist0 = gb.get_bit(); // ist[0] (intra IST)
        let seq_ist1 = gb.get_bit(); // ist[1] (inter IST)
        crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.ist = seq_ist0; t.ist_inter = seq_ist1; c.set(t); });
        if !is_i400 {
            gb.get_bit(); // chroma_dctonly
        }
        let inter_ddt = if !reduced_still_picture_header { gb.get_bit() } else { false };
        crate::av2_frame::INTER_DDT.with(|c| c.set(inter_ddt));
        crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.set(gb.get_bit())); // reduced_tx_part_set
        let seq_cctx = if !is_i400 { gb.get_bit() } else { false };
        if std::env::var("SEQDBG").is_ok() {
            let t = crate::av2_recon::SEQ_TOOLS.with(|c| c.get());
            crate::dlog!("[SEQDBG] bawp={} amvd={} signd={} mrls={} ist={} ist_inter={} ibp={} edge={} drlro={} rmvb={} cctx_pre={}",
                t.bawp, t.adaptive_mvd, t.mvd_sign_derive, t.mrls, t.ist, t.ist_inter, t.ibp, t.edge_filter, t.drl_reorder, t.refmvbank, seq_cctx);
        }
        crate::av2_recon::SEQ_TOOLS.with(|c| { let mut t = c.get(); t.cctx = seq_cctx; c.set(t); });
        // coefficient tools (576)
        let mut tcq = gb.get_bit() as u32;
        if tcq != 0 && !reduced_still_picture_header {
            tcq += gb.get_bit() as u32;
        }
        let mut parity_hiding = false;
        if tcq != 1 {
            parity_hiding = gb.get_bit();
        }
        // averaged CDF (588)
        let avg_cdf = reduced_still_picture_header || gb.get_bit();
        // Stash the compound seq flags (masked_compound / num_same_ref_comp / cwp / avg_cdf) for
        // the per-block compound parse + the pri/sec CDF-average gate (decode.c:5403).
        crate::av2_recon::SEQ_COMP.with(|c| c.set((seq_masked_compound, seq_num_same_ref_comp, seq_cwp, avg_cdf, seq_mv_traj)));
        let mut avg_cdf_type = 0u32;
        if avg_cdf {
            avg_cdf_type = if reduced_still_picture_header {
                1
            } else {
                gb.get_bit() as u32
            };
        }
        // quantizer flags (597)
        let mut separate_uv_delta_q = false;
        if !is_i400 {
            separate_uv_delta_q = gb.get_bit();
        }
        let equal_ac_dc_q = gb.get_bit();
        let mut ydc_dq_enabled = false;
        let mut uvdc_dq_enabled = false;
        let mut uvac_dq_enabled = false;
        if !equal_ac_dc_q {
            gb.get_bits(5); // base_ydc_dq
            ydc_dq_enabled = gb.get_bit();
        }
        if !is_i400 {
            if !equal_ac_dc_q {
                gb.get_bits(5); // base_uvdc_dq
                uvdc_dq_enabled = gb.get_bit();
            }
            gb.get_bits(5); // base_uvac_dq
            uvac_dq_enabled = gb.get_bit();
        }
        // in-loop filters (628)
        gb.get_bit(); // disable_loopfilters_across_tiles
        let cdef = gb.get_bit();
        let gdf = gb.get_bit();
        if gdf && sb128 == 0 {
            gb.get_bit(); // gdf_unit_matches_sbsz
        }
        let restoration = gb.get_bit();
        if restoration {
            // dav2d obu.c:635 rst_disable_mask: [0] = (no_ns_wiener_y<<1)|no_pc_wiener;
            // [1] = explicit ? (bit<<1)|1 : mask[0]|1 (chroma never has PC-wiener).
            let no_pc_wiener = gb.get_bit() as u8;
            let no_ns_wiener_y = gb.get_bit() as u8;
            let mask0 = (no_ns_wiener_y << 1) | no_pc_wiener;
            let mask1 = if gb.get_bit() {
                ((gb.get_bit() as u8) << 1) | 1
            } else {
                mask0 | 1
            };
            crate::av2_lr::SEQ_RST_MASK.with(|c| c.set((mask0, mask1)));
        }
        let ccso = gb.get_bit();
        if ccso {
            gb.get_bit(); // ccso_unit_matches_sbsz
        }
        let ccso_flag = ccso;
        let mut cdef_on_skiptx = 2u32; // reduced_still_picture => ADAPTIVE
        if !reduced_still_picture_header {
            cdef_on_skiptx = if gb.get_bit() {
                1
            } else if gb.get_bit() {
                0
            } else {
                2
            };
        }
        let df_par_bits = 2 + gb.get_bits(2);
        // tiling (667)
        let mut tiling_present = 0u32;
        let mut tiling_log2_cols = 0u8;
        let mut tiling_log2_rows = 0u8;
        if gb.get_bit() {
            tiling_present = 1 + gb.get_bit() as u32;
            (tiling_log2_cols, tiling_log2_rows) =
                parse_seq_tile_info(gb, sb128, sb128, max_width, max_height, level, tier);
        }
        // film grain (681)
        let film_grain_present = gb.get_bit();

        let bit_error = gb.has_error() != 0;
        crate::dlog!(
            "[rav2d AV2 seqhdr] profile={profile} level={level} tier={tier} layout={layout} \
             bitdepth={} size={max_width}x{max_height} sb128={sb128} motion_modes={motion_modes:#x} \
             tip={tip} gdf={gdf} restoration={restoration} ccso={ccso} \
             filmgrain={film_grain_present} bit_error={bit_error}",
            8 + 2 * hbd
        );
        if bit_error {
            return Err(Rav1dError::InvalidArgument);
        }
        let (ss_hor, ss_ver): (u8, u8) = match layout {
            2 => (1, 0), // I422
            3 => (0, 0), // I444
            _ => (1, 1), // I400 / I420
        };
        // Construct the sequence header from the parsed common fields; AV2-only tool
        // flags get a dedicated sub-struct once the decode path consumes them. This is
        // enough for the pipeline to advance to the frame header.
        return Ok(Rav1dSequenceHeader {
            profile: match profile {
                1 => Rav1dProfile::High,
                2 => Rav1dProfile::Professional,
                _ => Rav1dProfile::Main,
            },
            max_width: max_width as c_int,
            max_height: max_height as c_int,
            layout: match layout {
                1 => Rav1dPixelLayout::I420,
                2 => Rav1dPixelLayout::I422,
                3 => Rav1dPixelLayout::I444,
                _ => Rav1dPixelLayout::I400,
            },
            hbd: hbd as u8,
            reduced_still_picture_header: reduced_still_picture_header as u8,
            still_picture: reduced_still_picture_header as u8,
            width_n_bits: width_n_bits as u8,
            height_n_bits: height_n_bits as u8,
            sb128: sb128 as u8,
            ref_frame_mvs,
            cdef: cdef as u8,
            restoration: restoration as u8,
            film_grain_present: film_grain_present as u8,
            ss_hor,
            ss_ver,
            num_operating_points: 1,
            av2: Av2SeqHdr {
                id,
                number_of_bits_for_lt_frame_id: number_of_bits_for_lt_frame_id as u8,
                ref_frames: ref_frames as u8,
                ref_frames_log2: if ref_frames <= 2 {
                    (ref_frames - 1) as u8
                } else {
                    (1 + (31 - (ref_frames - 1).leading_zeros())) as u8
                },
                explicit_ref_frame_map,
                short_refresh_frame_flags,
                db_sub_pu,
                monotonic: monotonic as u8,
                max_mlayer_id: max_mlayer_id as u8,
                order_hint_n_bits: order_hint_n_bits as u8,
                tip: tip as u8,
                screen_content_tools: screen_content_tools as u8,
                force_integer_mv: force_integer_mv as u8,
                def_max_bvp_drl_bits: def_max_bvp_drl_bits as u8,
                allow_max_bvp_drl_bits: allow_max_bvp_drl_bits as u8,
                tiling_present: tiling_present as u8,
                tiling_log2_cols,
                tiling_log2_rows,
                avg_cdf_type: avg_cdf_type as u8,
                level: level as u8,
                tier: tier as u8,
                ydc_dq_enabled: ydc_dq_enabled as u8,
                uvdc_dq_enabled: uvdc_dq_enabled as u8,
                uvac_dq_enabled: uvac_dq_enabled as u8,
                separate_uv_delta_q: separate_uv_delta_q as u8,
                seg_info_present: seg_info_present as u8,
                seg_adaptive: seg_adaptive as u8,
                seg_ext: seg_ext as u8,
                tcq: tcq as u8,
                parity_hiding: parity_hiding as u8,
                df_par_bits: df_par_bits as u8,
                gdf: gdf as u8,
                cdef_on_skiptx: cdef_on_skiptx as u8,
                ccso: ccso_flag as u8,
                ..Default::default()
            },
            ..Default::default()
        });
    }

    let profile =
        Rav1dProfile::from_repr(gb.get_bits(3) as usize).ok_or(Rav1dError::InvalidArgument)?;
    debug.post(gb, "post-profile");

    let still_picture = gb.get_bit() as u8;
    let reduced_still_picture_header = gb.get_bit() as u8;
    if reduced_still_picture_header != 0 && still_picture == 0 {
        return Err(Rav1dError::InvalidArgument);
    }
    debug.post(gb, "post-stillpicture_flags");

    let num_operating_points;
    let mut operating_points =
        [Rav1dSequenceHeaderOperatingPoint::default(); RAV1D_MAX_OPERATING_POINTS];
    let timing_info_present;
    let num_units_in_tick;
    let time_scale;
    let equal_picture_interval;
    let num_ticks_per_picture;
    let decoder_model_info_present;
    let encoder_decoder_buffer_delay_length;
    let num_units_in_decoding_tick;
    let buffer_removal_delay_length;
    let frame_presentation_delay_length;
    let display_model_info_present;
    let mut operating_parameter_info =
        [Rav1dSequenceHeaderOperatingParameterInfo::default(); RAV1D_MAX_OPERATING_POINTS];
    if reduced_still_picture_header != 0 {
        num_operating_points = 1;
        operating_points[0].major_level = gb.get_bits(3) as u8;
        operating_points[0].minor_level = gb.get_bits(2) as u8;
        operating_points[0].initial_display_delay = 10;

        // Default initialization.
        timing_info_present = Default::default();
        num_units_in_tick = Default::default();
        time_scale = Default::default();
        equal_picture_interval = Default::default();
        num_ticks_per_picture = Default::default();
        decoder_model_info_present = Default::default();
        encoder_decoder_buffer_delay_length = Default::default();
        num_units_in_decoding_tick = Default::default();
        buffer_removal_delay_length = Default::default();
        frame_presentation_delay_length = Default::default();
        display_model_info_present = Default::default();
    } else {
        timing_info_present = gb.get_bit() as u8;
        if timing_info_present != 0 {
            num_units_in_tick = gb.get_bits(32) as u32;
            time_scale = gb.get_bits(32) as u32;
            if strict_std_compliance && (num_units_in_tick == 0 || time_scale == 0) {
                return Err(Rav1dError::InvalidArgument);
            }
            equal_picture_interval = gb.get_bit() as u8;
            if equal_picture_interval != 0 {
                let num_ticks_per_picture_ = gb.get_vlc();
                if num_ticks_per_picture_ == 0xffffffff {
                    return Err(Rav1dError::InvalidArgument);
                }
                num_ticks_per_picture = num_ticks_per_picture_ + 1;
            } else {
                // Default initialization.
                num_ticks_per_picture = Default::default();
            }

            decoder_model_info_present = gb.get_bit() as u8;
            if decoder_model_info_present != 0 {
                encoder_decoder_buffer_delay_length = gb.get_bits(5) as u8 + 1;
                num_units_in_decoding_tick = gb.get_bits(32) as u32;
                if strict_std_compliance && num_units_in_decoding_tick == 0 {
                    return Err(Rav1dError::InvalidArgument);
                }
                buffer_removal_delay_length = gb.get_bits(5) as u8 + 1;
                frame_presentation_delay_length = gb.get_bits(5) as u8 + 1;
            } else {
                // Default initialization.
                encoder_decoder_buffer_delay_length = Default::default();
                num_units_in_decoding_tick = Default::default();
                buffer_removal_delay_length = Default::default();
                frame_presentation_delay_length = Default::default();
            }
        } else {
            // Default initialization.
            num_units_in_tick = Default::default();
            time_scale = Default::default();
            equal_picture_interval = Default::default();
            num_ticks_per_picture = Default::default();
            decoder_model_info_present = Default::default();
            encoder_decoder_buffer_delay_length = Default::default();
            num_units_in_decoding_tick = Default::default();
            buffer_removal_delay_length = Default::default();
            frame_presentation_delay_length = Default::default();
        }
        debug.post(gb, "post-timinginfo");

        display_model_info_present = gb.get_bit() as u8;
        num_operating_points = gb.get_bits(5) as u8 + 1;
        for i in 0..num_operating_points {
            let op = &mut operating_points[i as usize];
            op.idc = gb.get_bits(12) as u16;
            if op.idc != 0 && (op.idc & 0xff == 0 || op.idc & 0xf00 == 0) {
                return Err(Rav1dError::InvalidArgument);
            }
            op.major_level = 2 + gb.get_bits(3) as u8;
            op.minor_level = gb.get_bits(2) as u8;
            if op.major_level > 3 {
                op.tier = gb.get_bit() as u8;
            }
            if decoder_model_info_present != 0 {
                op.decoder_model_param_present = gb.get_bit() as u8;
                if op.decoder_model_param_present != 0 {
                    let opi = &mut operating_parameter_info[i as usize];
                    opi.decoder_buffer_delay =
                        gb.get_bits(encoder_decoder_buffer_delay_length.into()) as u32;
                    opi.encoder_buffer_delay =
                        gb.get_bits(encoder_decoder_buffer_delay_length.into()) as u32;
                    opi.low_delay_mode = gb.get_bit() as u8;
                }
            }
            if display_model_info_present != 0 {
                op.display_model_param_present = gb.get_bit() as u8;
            }
            op.initial_display_delay = if op.display_model_param_present != 0 {
                gb.get_bits(4) as u8 + 1
            } else {
                10
            };
        }
        debug.post(gb, "operating-points");
    }

    let width_n_bits = gb.get_bits(4) as u8 + 1;
    let height_n_bits = gb.get_bits(4) as u8 + 1;
    let max_width = gb.get_bits(width_n_bits.into()) as c_int + 1;
    let max_height = gb.get_bits(height_n_bits.into()) as c_int + 1;
    debug.post(gb, "size");
    let frame_id_numbers_present;
    let delta_frame_id_n_bits;
    let frame_id_n_bits;
    if reduced_still_picture_header == 0 {
        frame_id_numbers_present = gb.get_bit() as u8;
        if frame_id_numbers_present != 0 {
            delta_frame_id_n_bits = gb.get_bits(4) as u8 + 2;
            frame_id_n_bits = gb.get_bits(3) as u8 + delta_frame_id_n_bits + 1;
        } else {
            // Default initialization.
            delta_frame_id_n_bits = Default::default();
            frame_id_n_bits = Default::default();
        }
    } else {
        // Default initialization.
        frame_id_numbers_present = Default::default();
        delta_frame_id_n_bits = Default::default();
        frame_id_n_bits = Default::default();
    }
    debug.post(gb, "frame-id-numbers-present");

    let sb128 = gb.get_bit() as u8;
    let filter_intra = gb.get_bit() as u8;
    let intra_edge_filter = gb.get_bit() as u8;
    let screen_content_tools;
    let force_integer_mv;
    let inter_intra;
    let masked_compound;
    let warped_motion;
    let dual_filter;
    let order_hint;
    let jnt_comp;
    let ref_frame_mvs;
    let order_hint_n_bits;
    if reduced_still_picture_header != 0 {
        screen_content_tools = Rav1dAdaptiveBoolean::Adaptive;
        force_integer_mv = Rav1dAdaptiveBoolean::Adaptive;

        // Default initialization.
        inter_intra = Default::default();
        masked_compound = Default::default();
        warped_motion = Default::default();
        dual_filter = Default::default();
        order_hint = Default::default();
        jnt_comp = Default::default();
        ref_frame_mvs = Default::default();
        order_hint_n_bits = Default::default();
    } else {
        inter_intra = gb.get_bit() as u8;
        masked_compound = gb.get_bit() as u8;
        warped_motion = gb.get_bit() as u8;
        dual_filter = gb.get_bit() as u8;
        order_hint = gb.get_bit() as u8;
        if order_hint != 0 {
            jnt_comp = gb.get_bit() as u8;
            ref_frame_mvs = gb.get_bit() as u8;
        } else {
            // Default initialization.
            jnt_comp = Default::default();
            ref_frame_mvs = Default::default();
        }
        screen_content_tools = if gb.get_bit() {
            Rav1dAdaptiveBoolean::Adaptive
        } else {
            gb.get_bit().into()
        };
        debug.post(gb, "screentools");
        force_integer_mv = if screen_content_tools != Rav1dAdaptiveBoolean::Off {
            if gb.get_bit() {
                Rav1dAdaptiveBoolean::Adaptive
            } else {
                gb.get_bit().into()
            }
        } else {
            Rav1dAdaptiveBoolean::Adaptive
        };
        if order_hint != 0 {
            order_hint_n_bits = gb.get_bits(3) as u8 + 1;
        } else {
            // Default initialization.
            order_hint_n_bits = Default::default();
        }
    }
    let super_res = gb.get_bit() as u8;
    let cdef = gb.get_bit() as u8;
    let restoration = gb.get_bit() as u8;
    debug.post(gb, "featurebits");

    let hbd = {
        let mut hbd = gb.get_bit() as u8;
        if profile == Rav1dProfile::Professional && hbd != 0 {
            hbd += gb.get_bit() as u8;
        }
        hbd
    };
    let monochrome;
    if profile != Rav1dProfile::High {
        monochrome = gb.get_bit() as u8;
    } else {
        // Default initialization.
        monochrome = Default::default();
    }
    let color_description_present = gb.get_bit();
    let pri;
    let trc;
    let mtrx;
    if color_description_present {
        pri = gb.get_bits(8).try_into().unwrap_or_default();
        trc = gb.get_bits(8).try_into().unwrap_or_default();
        mtrx = gb.get_bits(8).try_into().unwrap_or_default();
    } else {
        pri = Default::default();
        trc = Default::default();
        mtrx = Default::default();
    }
    let full_color_range;
    let layout;
    let ss_ver;
    let ss_hor;
    let chr;
    if monochrome != 0 {
        full_color_range = gb.get_bit();
        layout = Rav1dPixelLayout::I400;
        ss_ver = 1;
        ss_hor = ss_ver;
        chr = Rav1dChromaSamplePosition::Unknown;
    } else if pri == Rav1dColorPrimaries::BT709
        && trc == Rav1dTransferCharacteristics::SRGB
        && mtrx == Rav1dMatrixCoefficients::Identity
    {
        layout = Rav1dPixelLayout::I444;
        full_color_range = true;
        if profile != Rav1dProfile::High && !(profile == Rav1dProfile::Professional && hbd == 2) {
            return Err(Rav1dError::InvalidArgument);
        }

        // Default initialization.
        ss_hor = Default::default();
        ss_ver = Default::default();
        chr = Rav1dChromaSamplePosition::Unknown;
    } else {
        full_color_range = gb.get_bit();
        match profile {
            Rav1dProfile::Main => {
                layout = Rav1dPixelLayout::I420;
                ss_ver = 1;
                ss_hor = ss_ver;
            }
            Rav1dProfile::High => {
                layout = Rav1dPixelLayout::I444;

                // Default initialization.
                ss_hor = Default::default();
                ss_ver = Default::default();
            }
            Rav1dProfile::Professional => {
                if hbd == 2 {
                    ss_hor = gb.get_bit() as u8;
                    if ss_hor != 0 {
                        ss_ver = gb.get_bit() as u8;
                    } else {
                        // Default initialization.
                        ss_ver = Default::default();
                    }
                } else {
                    ss_hor = 1;

                    // Default initialization.
                    ss_ver = Default::default();
                }
                layout = if ss_hor != 0 {
                    if ss_ver != 0 {
                        Rav1dPixelLayout::I420
                    } else {
                        Rav1dPixelLayout::I422
                    }
                } else {
                    Rav1dPixelLayout::I444
                };
            }
        }
        chr = if ss_hor & ss_ver != 0 {
            // HARDENING: the 2-bit field has 4 encodings but the enum defines 3 — value 3 is
            // reserved. from_repr(..).unwrap() panicked on it (found by static audit, NOT by
            // ~3000 fuzz cases: the seq-header colour-config path is hard to reach by mutation).
            match Rav1dChromaSamplePosition::from_repr(gb.get_bits(2) as usize) {
                Some(v) => v,
                None => return Err(Rav1dError::InvalidArgument),
            }
        } else {
            Rav1dChromaSamplePosition::Unknown
        };
    }
    let color_range = Rav1dColorRange::from_is_full(full_color_range);
    if strict_std_compliance
        && mtrx == Rav1dMatrixCoefficients::Identity
        && layout != Rav1dPixelLayout::I444
    {
        return Err(Rav1dError::InvalidArgument);
    }
    let separate_uv_delta_q;
    if monochrome == 0 {
        separate_uv_delta_q = gb.get_bit() as u8;
    } else {
        // Default initialization.
        separate_uv_delta_q = Default::default();
    }
    debug.post(gb, "colorinfo");

    let film_grain_present = gb.get_bit() as u8;
    debug.post(gb, "filmgrain");

    // We needn't bother flushing the OBU here: we'll check we didn't
    // overrun in the caller and will then discard gb, so there's no
    // point in setting its position properly.

    check_trailing_bits(gb, strict_std_compliance)?;
    Ok(Rav1dSequenceHeader {
        av2: Default::default(),
        profile,
        max_width,
        max_height,
        layout,
        pri,
        trc,
        mtrx,
        chr,
        hbd,
        color_range,
        num_operating_points,
        operating_points,
        still_picture,
        reduced_still_picture_header,
        timing_info_present,
        num_units_in_tick,
        time_scale,
        equal_picture_interval,
        num_ticks_per_picture,
        decoder_model_info_present,
        encoder_decoder_buffer_delay_length,
        num_units_in_decoding_tick,
        buffer_removal_delay_length,
        frame_presentation_delay_length,
        display_model_info_present,
        width_n_bits,
        height_n_bits,
        frame_id_numbers_present,
        delta_frame_id_n_bits,
        frame_id_n_bits,
        sb128,
        filter_intra,
        intra_edge_filter,
        inter_intra,
        masked_compound,
        warped_motion,
        dual_filter,
        order_hint,
        jnt_comp,
        ref_frame_mvs,
        screen_content_tools,
        force_integer_mv,
        order_hint_n_bits,
        super_res,
        cdef,
        restoration,
        ss_hor,
        ss_ver,
        monochrome,
        color_description_present,
        separate_uv_delta_q,
        film_grain_present,
        operating_parameter_info,
    })
}

pub(crate) fn rav1d_parse_sequence_header(
    mut data: &[u8],
) -> Rav1dResult<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>> {
    let mut res = Err(Rav1dError::NoEntity);

    while !data.is_empty() {
        let gb = &mut GetBits::new(data);

        gb.get_bit(); // obu_forbidden_bit
        let r#type = Rav1dObuType::from_repr(gb.get_bits(4) as usize);
        let has_extension = gb.get_bit();
        let has_length_field = gb.get_bit();
        gb.get_bits(1 + has_extension as i32 * 8); // reserved

        // obu length field
        let obu_end = if has_length_field {
            let len = gb.get_uleb128() as usize;
            let len = gb.byte_pos() + len;
            if len > data.len() {
                return Err(Rav1dError::InvalidArgument);
            }
            len
        } else {
            data.len()
        };

        if r#type == Some(Rav1dObuType::SeqHdr) {
            res = Ok(parse_seq_hdr(gb, false)?);
            if gb.byte_pos() > obu_end {
                return Err(Rav1dError::InvalidArgument);
            }
            gb.bytealign();
        }

        if gb.has_error() != 0 {
            return Err(Rav1dError::InvalidArgument);
        }
        assert!(!gb.has_pending_bits());

        data = &data[obu_end..]
    }

    res.map(DRav1d::from_rav1d)
}

fn parse_frame_size(
    state: &Rav1dState,
    seqhdr: &Rav1dSequenceHeader,
    refidx: Option<&[i8; RAV1D_REFS_PER_FRAME]>,
    frame_size_override: bool,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameSize> {
    if let Some(refidx) = refidx {
        for i in 0..7 {
            if gb.get_bit() {
                let r#ref = &state.refs[refidx[i as usize] as usize].p;
                let ref_size = &r#ref
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(Rav1dError::InvalidArgument)?
                    .size;
                let width1 = ref_size.width[1];
                let height = ref_size.height;
                let render_width = ref_size.render_width;
                let render_height = ref_size.render_height;
                let enabled = seqhdr.super_res != 0 && gb.get_bit();
                let width_scale_denominator;
                let width0;
                if enabled {
                    width_scale_denominator = 9 + gb.get_bits(3) as u8;
                    let d = width_scale_denominator as c_int;
                    width0 = cmp::max((width1 * 8 + (d >> 1)) / d, cmp::min(16, width1));
                } else {
                    width_scale_denominator = 8;
                    width0 = width1;
                }
                let width = [width0, width1];
                return Ok(Rav1dFrameSize {
                    width,
                    height,
                    render_width,
                    render_height,
                    super_res: Rav1dFrameHeaderSuperRes {
                        enabled,
                        width_scale_denominator,
                    },
                    have_render_size: 0,
                });
            }
        }
    }

    let width1;
    let height;
    if frame_size_override {
        width1 = gb.get_bits(seqhdr.width_n_bits.into()) as c_int + 1;
        height = gb.get_bits(seqhdr.height_n_bits.into()) as c_int + 1;
    } else {
        width1 = seqhdr.max_width;
        height = seqhdr.max_height;
    }
    let enabled = seqhdr.super_res != 0 && gb.get_bit();
    let width_scale_denominator;
    let width0;
    if enabled {
        width_scale_denominator = 9 + gb.get_bits(3) as u8;
        let d = width_scale_denominator as c_int;
        width0 = cmp::max((width1 * 8 + (d >> 1)) / d, cmp::min(16, width1));
    } else {
        width_scale_denominator = 8;
        width0 = width1;
    }
    let have_render_size = gb.get_bit() as u8;
    let render_width;
    let render_height;
    if have_render_size != 0 {
        render_width = gb.get_bits(16) as c_int + 1;
        render_height = gb.get_bits(16) as c_int + 1;
    } else {
        render_width = width1;
        render_height = height;
    }
    let width = [width0, width1];
    Ok(Rav1dFrameSize {
        width,
        height,
        render_width,
        render_height,
        super_res: Rav1dFrameHeaderSuperRes {
            enabled,
            width_scale_denominator,
        },
        have_render_size,
    })
}

#[inline]
fn tile_log2(sz: c_int, tgt: c_int) -> u8 {
    let mut k = 0;
    while sz << k < tgt {
        k += 1;
    }
    k
}

static DEFAULT_MODE_REF_DELTAS: Rav1dLoopfilterModeRefDeltas = Rav1dLoopfilterModeRefDeltas {
    mode_delta: [0, 0],
    ref_delta: [1, 0, 0, 0, -1, 0, -1, -1],
};

fn parse_refidx(
    state: &Rav1dState,
    seqhdr: &Rav1dSequenceHeader,
    frame_ref_short_signaling: u8,
    frame_offset: u8,
    frame_id: u32,
    gb: &mut GetBits,
) -> Rav1dResult<[i8; RAV1D_REFS_PER_FRAME]> {
    let mut refidx = [-1; RAV1D_REFS_PER_FRAME];
    if frame_ref_short_signaling != 0 {
        // FIXME: Nearly verbatim copy from section 7.8
        refidx[0] = gb.get_bits(3) as i8;
        refidx[3] = gb.get_bits(3) as i8;

        let mut shifted_frame_offset = [0; 8];
        let current_frame_offset = 1 << seqhdr.order_hint_n_bits - 1;
        for i in 0..8 {
            shifted_frame_offset[i as usize] = current_frame_offset
                + get_poc_diff(
                    seqhdr.order_hint_n_bits,
                    state.refs[i as usize]
                        .p
                        .p
                        .frame_hdr
                        .as_ref()
                        .ok_or(Rav1dError::InvalidArgument)?
                        .frame_offset as c_int,
                    frame_offset as c_int,
                );
        }

        let mut used_frame = [0, 0, 0, 0, 0, 0, 0, 0];
        used_frame[refidx[0] as usize] = 1;
        used_frame[refidx[3] as usize] = 1;

        let mut latest_frame_offset = -1;
        for i in 0..8 {
            let hint = shifted_frame_offset[i as usize];
            if used_frame[i as usize] == 0
                && hint >= current_frame_offset
                && hint >= latest_frame_offset
            {
                refidx[6] = i;
                latest_frame_offset = hint;
            }
        }
        if latest_frame_offset != -1 {
            used_frame[refidx[6] as usize] = 1;
        }

        let mut earliest_frame_offset = i32::MAX;
        for i in 0..8 {
            let hint = shifted_frame_offset[i as usize];
            if used_frame[i as usize] == 0
                && hint >= current_frame_offset
                && hint < earliest_frame_offset
            {
                refidx[4] = i;
                earliest_frame_offset = hint;
            }
        }
        if earliest_frame_offset != i32::MAX {
            used_frame[refidx[4] as usize] = 1;
        }

        earliest_frame_offset = i32::MAX;
        for i in 0..8 {
            let hint = shifted_frame_offset[i as usize];
            if used_frame[i as usize] == 0
                && hint >= current_frame_offset
                && hint < earliest_frame_offset
            {
                refidx[5] = i;
                earliest_frame_offset = hint;
            }
        }
        if earliest_frame_offset != i32::MAX {
            used_frame[refidx[5] as usize] = 1;
        }

        for i in 1..7 {
            if refidx[i as usize] < 0 {
                latest_frame_offset = -1;
                for j in 0..8 {
                    let hint = shifted_frame_offset[j as usize];
                    if used_frame[j as usize] == 0
                        && hint < current_frame_offset
                        && hint >= latest_frame_offset
                    {
                        refidx[i as usize] = j;
                        latest_frame_offset = hint;
                    }
                }
                if latest_frame_offset != -1 {
                    used_frame[refidx[i as usize] as usize] = 1;
                }
            }
        }

        earliest_frame_offset = i32::MAX;
        let mut r#ref = -1;
        for i in 0..8 {
            let hint = shifted_frame_offset[i as usize];
            if hint < earliest_frame_offset {
                r#ref = i;
                earliest_frame_offset = hint;
            }
        }
        for i in 0..7 {
            if refidx[i as usize] < 0 {
                refidx[i as usize] = r#ref;
            }
        }
    }
    for i in 0..7 {
        if frame_ref_short_signaling == 0 {
            refidx[i as usize] = gb.get_bits(3) as i8;
        }
        if seqhdr.frame_id_numbers_present != 0 {
            let delta_ref_frame_id = gb.get_bits(seqhdr.delta_frame_id_n_bits.into()) as u32 + 1;
            let ref_frame_id = frame_id + (1 << seqhdr.frame_id_n_bits) - delta_ref_frame_id
                & (1 << seqhdr.frame_id_n_bits) - 1;
            state.refs[refidx[i as usize] as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .filter(|ref_frame_hdr| ref_frame_hdr.frame_id == ref_frame_id)
                .ok_or(Rav1dError::InvalidArgument)?;
        }
    }
    Ok(refidx)
}

fn parse_tiling(
    seqhdr: &Rav1dSequenceHeader,
    size: &Rav1dFrameSize,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameHeaderTiling> {
    let uniform = gb.get_bit() as u8;
    let sbsz_min1 = ((64) << seqhdr.sb128) - 1;
    let sbsz_log2 = 6 + seqhdr.sb128;
    let sbw = size.width[0] + sbsz_min1 >> sbsz_log2;
    let sbh = size.height + sbsz_min1 >> sbsz_log2;
    let max_tile_width_sb = 4096 >> sbsz_log2;
    let max_tile_area_sb = 4096 * 2304 >> 2 * sbsz_log2;
    let min_log2_cols = tile_log2(max_tile_width_sb, sbw);
    let max_log2_cols = tile_log2(1, cmp::min(sbw, RAV1D_MAX_TILE_COLS as c_int));
    let max_log2_rows = tile_log2(1, cmp::min(sbh, RAV1D_MAX_TILE_ROWS as c_int));
    let min_log2_tiles = cmp::max(tile_log2(max_tile_area_sb, sbw * sbh), min_log2_cols);
    let mut log2_cols;
    let mut cols;
    let mut log2_rows;
    let mut rows;
    let mut col_start_sb = [0; RAV1D_MAX_TILE_COLS + 1];
    let mut row_start_sb = [0; RAV1D_MAX_TILE_ROWS + 1];
    if uniform != 0 {
        log2_cols = min_log2_cols;
        while log2_cols < max_log2_cols && gb.get_bit() {
            log2_cols += 1;
        }
        let tile_w = 1 + (sbw - 1 >> log2_cols);
        cols = 0;
        let mut sbx = 0;
        while sbx < sbw {
            col_start_sb[cols as usize] = sbx as u16;
            sbx += tile_w;
            cols += 1;
        }
        let min_log2_rows = min_log2_tiles.saturating_sub(log2_cols);

        log2_rows = min_log2_rows;
        while log2_rows < max_log2_rows && gb.get_bit() {
            log2_rows += 1;
        }
        let tile_h = 1 + (sbh - 1 >> log2_rows);
        rows = 0;
        let mut sby = 0;
        while sby < sbh {
            row_start_sb[rows as usize] = sby as u16;
            sby += tile_h;
            rows += 1;
        }
    } else {
        cols = 0;
        let mut widest_tile = 0;
        let mut max_tile_area_sb = sbw * sbh;
        let mut sbx = 0;
        while sbx < sbw && cols < RAV1D_MAX_TILE_COLS as u8 {
            let tile_width_sb = cmp::min(sbw - sbx, max_tile_width_sb);
            let tile_w = if tile_width_sb > 1 {
                1 + gb.get_uniform(tile_width_sb as c_uint) as c_int
            } else {
                1
            };
            col_start_sb[cols as usize] = sbx as u16;
            sbx += tile_w;
            widest_tile = cmp::max(widest_tile, tile_w);
            cols += 1;
        }
        log2_cols = tile_log2(1, cols.into());
        if min_log2_tiles != 0 {
            max_tile_area_sb >>= min_log2_tiles + 1;
        }
        let max_tile_height_sb = cmp::max(max_tile_area_sb / widest_tile, 1);

        rows = 0;
        let mut sby = 0;
        while sby < sbh && rows < RAV1D_MAX_TILE_ROWS as u8 {
            let tile_height_sb = cmp::min(sbh - sby, max_tile_height_sb);
            let tile_h = if tile_height_sb > 1 {
                1 + gb.get_uniform(tile_height_sb as c_uint) as c_int
            } else {
                1
            };
            row_start_sb[rows as usize] = sby as u16;
            sby += tile_h;
            rows += 1;
        }
        log2_rows = tile_log2(1, rows.into());
    }
    col_start_sb[cols as usize] = sbw as u16;
    row_start_sb[rows as usize] = sbh as u16;
    let update;
    let n_bytes;
    if log2_cols != 0 || log2_rows != 0 {
        update = gb.get_bits((log2_cols + log2_rows).into()) as u16;
        if update >= cols as u16 * rows as u16 {
            return Err(Rav1dError::InvalidArgument);
        }
        n_bytes = gb.get_bits(2) as u8 + 1;
    } else {
        update = 0;
        n_bytes = update as u8;
    }
    debug.post(gb, "tiling");
    Ok(Rav1dFrameHeaderTiling {
        uniform,
        n_bytes,
        min_log2_cols,
        max_log2_cols,
        log2_cols,
        cols,
        // TODO(kkysen) Never written or read in C; is this correct?
        min_log2_rows: 0,
        max_log2_rows,
        log2_rows,
        rows,
        col_start_sb,
        row_start_sb,
        update,
    })
}

fn parse_quant(
    seqhdr: &Rav1dSequenceHeader,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dFrameHeaderQuant {
    let yac = gb.get_bits(8) as u8;
    let ydc_delta = if gb.get_bit() {
        gb.get_sbits(7) as i8
    } else {
        0
    };
    let udc_delta;
    let uac_delta;
    let vdc_delta;
    let vac_delta;
    if seqhdr.monochrome == 0 {
        // If the sequence header says that delta_q might be different
        // for U, V, we must check whether it actually is for this
        // frame.
        let diff_uv_delta = if seqhdr.separate_uv_delta_q != 0 {
            gb.get_bit() as c_int
        } else {
            0
        };
        udc_delta = if gb.get_bit() {
            gb.get_sbits(7) as i8
        } else {
            0
        };
        uac_delta = if gb.get_bit() {
            gb.get_sbits(7) as i8
        } else {
            0
        };
        if diff_uv_delta != 0 {
            vdc_delta = if gb.get_bit() {
                gb.get_sbits(7) as i8
            } else {
                0
            };
            vac_delta = if gb.get_bit() {
                gb.get_sbits(7) as i8
            } else {
                0
            };
        } else {
            vdc_delta = udc_delta;
            vac_delta = uac_delta;
        }
    } else {
        // Default initialization.
        udc_delta = Default::default();
        uac_delta = Default::default();
        vdc_delta = Default::default();
        vac_delta = Default::default();
    }
    debug.post(gb, "quant");
    let qm = gb.get_bit() as u8;
    let qm_y;
    let qm_u;
    let qm_v;
    if qm != 0 {
        qm_y = gb.get_bits(4) as u8;
        qm_u = gb.get_bits(4) as u8;
        qm_v = if seqhdr.separate_uv_delta_q != 0 {
            gb.get_bits(4) as u8
        } else {
            qm_u
        };
    } else {
        // Default initialization.
        qm_y = Default::default();
        qm_u = Default::default();
        qm_v = Default::default();
    }
    debug.post(gb, "qm");
    Rav1dFrameHeaderQuant {
        yac,
        ydc_delta,
        udc_delta,
        uac_delta,
        vdc_delta,
        vac_delta,
        qm,
        qm_y,
        qm_u,
        qm_v,
    }
}

fn parse_seg_data(gb: &mut GetBits) -> Rav1dSegmentationDataSet {
    let mut preskip = 0;
    let mut last_active_segid = -1 as i8;
    let d = array::from_fn(|i| {
        let i = i as i8;
        let delta_q;
        if gb.get_bit() {
            delta_q = gb.get_sbits(9) as i16;
            last_active_segid = i;
        } else {
            delta_q = 0;
        }
        let delta_lf_y_v;
        if gb.get_bit() {
            delta_lf_y_v = gb.get_sbits(7) as i8;
            last_active_segid = i;
        } else {
            delta_lf_y_v = 0;
        }
        let delta_lf_y_h;
        if gb.get_bit() {
            delta_lf_y_h = gb.get_sbits(7) as i8;
            last_active_segid = i;
        } else {
            delta_lf_y_h = 0;
        }
        let delta_lf_u;
        if gb.get_bit() {
            delta_lf_u = gb.get_sbits(7) as i8;
            last_active_segid = i;
        } else {
            delta_lf_u = 0;
        }
        let delta_lf_v;
        if gb.get_bit() {
            delta_lf_v = gb.get_sbits(7) as i8;
            last_active_segid = i;
        } else {
            delta_lf_v = 0;
        }
        let r#ref;
        if gb.get_bit() {
            r#ref = gb.get_bits(3) as i8;
            last_active_segid = i;
            preskip = 1;
        } else {
            r#ref = -1;
        }
        let skip = gb.get_bit() as u8;
        if skip != 0 {
            last_active_segid = i;
            preskip = 1;
        }
        let globalmv = gb.get_bit() as u8;
        if globalmv != 0 {
            last_active_segid = i;
            preskip = 1;
        }
        Rav1dSegmentationData {
            delta_q,
            delta_lf_y_v,
            delta_lf_y_h,
            delta_lf_u,
            delta_lf_v,
            r#ref,
            skip,
            globalmv,
        }
    });
    Rav1dSegmentationDataSet {
        d,
        preskip,
        last_active_segid,
    }
}

fn parse_segmentation(
    state: &Rav1dState,
    primary_ref_frame: u8,
    refidx: &[i8; RAV1D_REFS_PER_FRAME],
    quant: &Rav1dFrameHeaderQuant,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameHeaderSegmentation> {
    let enabled = gb.get_bit() as u8;
    let update_map;
    let temporal;
    let update_data;
    let seg_data = if enabled != 0 {
        if primary_ref_frame == RAV1D_PRIMARY_REF_NONE {
            update_map = 1;
            temporal = 0;
            update_data = 1;
        } else {
            update_map = gb.get_bit() as u8;
            temporal = if update_map != 0 {
                gb.get_bit() as u8
            } else {
                0
            };
            update_data = gb.get_bit() as u8;
        }

        if update_data != 0 {
            parse_seg_data(gb)
        } else {
            // segmentation.update_data was false so we should copy
            // segmentation data from the reference frame.
            assert!(primary_ref_frame != RAV1D_PRIMARY_REF_NONE);
            let pri_ref = refidx[primary_ref_frame as usize];
            state.refs[pri_ref as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .ok_or(Rav1dError::InvalidArgument)?
                .segmentation
                .seg_data
                .clone()
        }
    } else {
        // Default initialization.
        update_map = Default::default();
        temporal = Default::default();
        update_data = Default::default();

        let mut seg_data = Rav1dSegmentationDataSet::default();
        for data in &mut seg_data.d {
            data.r#ref = -1;
        }
        seg_data
    };
    debug.post(gb, "segmentation");

    // derive lossless flags
    let delta_lossless = quant.ydc_delta == 0
        && quant.udc_delta == 0
        && quant.uac_delta == 0
        && quant.vdc_delta == 0
        && quant.vac_delta == 0;
    let qidx = array::from_fn(|i| {
        if enabled != 0 {
            clip_u8(quant.yac as c_int + seg_data.d[i].delta_q as c_int)
        } else {
            quant.yac
        }
    });
    let lossless = array::from_fn(|i| qidx[i] == 0 && delta_lossless);
    Ok(Rav1dFrameHeaderSegmentation {
        enabled,
        update_map,
        temporal,
        update_data,
        seg_data,
        lossless,
        qidx,
    })
}

fn parse_delta(
    quant: &Rav1dFrameHeaderQuant,
    allow_intrabc: bool,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dFrameHeaderDelta {
    let q = {
        let present = if quant.yac != 0 {
            gb.get_bit() as u8
        } else {
            0
        };
        let res_log2 = if present != 0 {
            gb.get_bits(2) as u8
        } else {
            0
        };
        Rav1dFrameHeaderDeltaQ { present, res_log2 }
    };
    let lf = {
        let present = (q.present != 0 && !allow_intrabc && gb.get_bit()) as u8;
        let res_log2 = if present != 0 {
            gb.get_bits(2) as u8
        } else {
            0
        };
        let multi = if present != 0 { gb.get_bit() as u8 } else { 0 };
        Rav1dFrameHeaderDeltaLF {
            present,
            res_log2,
            multi,
        }
    };
    debug.post(gb, "delta_q_lf_flags");
    Rav1dFrameHeaderDelta { q, lf }
}

fn parse_loopfilter(
    state: &Rav1dState,
    seqhdr: &Rav1dSequenceHeader,
    all_lossless: bool,
    allow_intrabc: bool,
    primary_ref_frame: u8,
    refidx: &[i8; RAV1D_REFS_PER_FRAME],
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameHeaderLoopFilter> {
    let level_y;
    let level_u;
    let level_v;
    let mode_ref_delta_enabled;
    let mode_ref_delta_update;
    let mut mode_ref_deltas;
    let sharpness;
    if all_lossless || allow_intrabc {
        level_y = [0; 2];
        level_v = 0;
        level_u = level_v;
        sharpness = 0;
        mode_ref_delta_enabled = 1;
        mode_ref_delta_update = 1;
        mode_ref_deltas = DEFAULT_MODE_REF_DELTAS.clone();
    } else {
        level_y = [gb.get_bits(6) as u8, gb.get_bits(6) as u8];
        if seqhdr.monochrome == 0 && (level_y[0] != 0 || level_y[1] != 0) {
            level_u = gb.get_bits(6) as u8;
            level_v = gb.get_bits(6) as u8;
        } else {
            // Default initialization.
            level_u = Default::default();
            level_v = Default::default();
        }
        sharpness = gb.get_bits(3) as u8;

        if primary_ref_frame == RAV1D_PRIMARY_REF_NONE {
            mode_ref_deltas = DEFAULT_MODE_REF_DELTAS.clone();
        } else {
            let r#ref = refidx[primary_ref_frame as usize];
            mode_ref_deltas = state.refs[r#ref as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .ok_or(Rav1dError::InvalidArgument)?
                .loopfilter
                .mode_ref_deltas
                .clone();
        }
        mode_ref_delta_enabled = gb.get_bit() as u8;
        if mode_ref_delta_enabled != 0 {
            mode_ref_delta_update = gb.get_bit() as u8;
            if mode_ref_delta_update != 0 {
                for i in 0..8 {
                    if gb.get_bit() {
                        mode_ref_deltas.ref_delta[i as usize] = gb.get_sbits(7) as i8;
                    }
                }
                for i in 0..2 {
                    if gb.get_bit() {
                        mode_ref_deltas.mode_delta[i as usize] = gb.get_sbits(7) as i8;
                    }
                }
            }
        } else {
            // Default initialization.
            mode_ref_delta_update = Default::default();
        }
    }
    debug.post(gb, "lpf");
    Ok(Rav1dFrameHeaderLoopFilter {
        level_y,
        level_u,
        level_v,
        mode_ref_delta_enabled,
        mode_ref_delta_update,
        mode_ref_deltas,
        sharpness,
    })
}

fn parse_cdef(
    seqhdr: &Rav1dSequenceHeader,
    all_lossless: bool,
    allow_intrabc: bool,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dFrameHeaderCdef {
    let damping;
    let n_bits;
    let mut y_strength = [0; RAV1D_MAX_CDEF_STRENGTHS];
    let mut uv_strength = [0; RAV1D_MAX_CDEF_STRENGTHS];
    if !all_lossless && seqhdr.cdef != 0 && !allow_intrabc {
        damping = gb.get_bits(2) as u8 + 3;
        n_bits = gb.get_bits(2) as u8;
        for i in 0..1 << n_bits {
            y_strength[i as usize] = gb.get_bits(6) as u8;
            if seqhdr.monochrome == 0 {
                uv_strength[i as usize] = gb.get_bits(6) as u8;
            }
        }
    } else {
        // Default initialization.
        damping = Default::default();

        n_bits = 0;
        y_strength[0] = 0;
        uv_strength[0] = 0;
    }
    debug.post(gb, "cdef");
    Rav1dFrameHeaderCdef {
        damping,
        n_bits,
        y_strength,
        uv_strength,
    }
}

fn parse_restoration(
    seqhdr: &Rav1dSequenceHeader,
    all_lossless: bool,
    super_res_enabled: bool,
    allow_intrabc: bool,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dFrameHeaderRestoration {
    let r#type;
    let unit_size;
    if (!all_lossless || super_res_enabled) && seqhdr.restoration != 0 && !allow_intrabc {
        let type_0 = Rav1dRestorationType::from_repr(gb.get_bits(2) as usize).unwrap();
        r#type = if seqhdr.monochrome == 0 {
            [
                type_0,
                Rav1dRestorationType::from_repr(gb.get_bits(2) as usize).unwrap(),
                Rav1dRestorationType::from_repr(gb.get_bits(2) as usize).unwrap(),
            ]
        } else {
            [
                type_0,
                Rav1dRestorationType::None,
                Rav1dRestorationType::None,
            ]
        };

        unit_size = match r#type {
            [Rav1dRestorationType::None, Rav1dRestorationType::None, Rav1dRestorationType::None] => {
                [8, 0]
            }
            _ => {
                // Log2 of the restoration unit size.
                let mut unit_size_0 = 6 + seqhdr.sb128;
                if gb.get_bit() {
                    unit_size_0 += 1;
                    if seqhdr.sb128 == 0 {
                        unit_size_0 += gb.get_bit() as u8;
                    }
                }

                let unit_size_1 = if (r#type[1] != Rav1dRestorationType::None
                    || r#type[2] != Rav1dRestorationType::None)
                    && seqhdr.ss_hor == 1
                    && seqhdr.ss_ver == 1
                {
                    unit_size_0 - gb.get_bit() as u8
                } else {
                    unit_size_0
                };

                [unit_size_0, unit_size_1]
            }
        };
    } else {
        r#type = [Rav1dRestorationType::None; 3];

        // Default initialization.
        unit_size = Default::default();
    }
    debug.post(gb, "restoration");
    Rav1dFrameHeaderRestoration { r#type, unit_size }
}

fn parse_skip_mode(
    state: &Rav1dState,
    seqhdr: &Rav1dSequenceHeader,
    switchable_comp_refs: u8,
    frame_type: Rav1dFrameType,
    frame_offset: u8,
    refidx: &[i8; RAV1D_REFS_PER_FRAME],
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameSkipMode> {
    let mut allowed = 0;
    let mut refs = Default::default();
    if switchable_comp_refs != 0 && frame_type.is_inter_or_switch() && seqhdr.order_hint != 0 {
        let poc = frame_offset as c_uint;
        let mut off_before = 0xffffffff;
        let mut off_after = -1;
        let mut off_before_idx = 0;
        let mut off_after_idx = 0;
        for i in 0..7 {
            let refpoc = state.refs[refidx[i as usize] as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .ok_or(Rav1dError::InvalidArgument)?
                .frame_offset as c_uint;

            let diff = get_poc_diff(seqhdr.order_hint_n_bits, refpoc as c_int, poc as c_int);
            if diff > 0 {
                if off_after == -1
                    || get_poc_diff(seqhdr.order_hint_n_bits, off_after, refpoc as c_int) > 0
                {
                    off_after = refpoc as c_int;
                    off_after_idx = i;
                }
            } else if diff < 0
                && (off_before == 0xffffffff
                    || get_poc_diff(
                        seqhdr.order_hint_n_bits,
                        refpoc as c_int,
                        off_before as c_int,
                    ) > 0)
            {
                off_before = refpoc;
                off_before_idx = i;
            }
        }

        if off_before != 0xffffffff && off_after != -1 {
            refs = [
                cmp::min(off_before_idx, off_after_idx),
                cmp::max(off_before_idx, off_after_idx),
            ];
            allowed = 1;
        } else if off_before != 0xffffffff {
            let mut off_before2 = 0xffffffff;
            let mut off_before2_idx = 0;
            for i in 0..7 {
                let refpoc = state.refs[refidx[i as usize] as usize]
                    .p
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(Rav1dError::InvalidArgument)?
                    .frame_offset as c_uint;
                if get_poc_diff(
                    seqhdr.order_hint_n_bits,
                    refpoc as c_int,
                    off_before as c_int,
                ) < 0
                {
                    if off_before2 == 0xffffffff
                        || get_poc_diff(
                            seqhdr.order_hint_n_bits,
                            refpoc as c_int,
                            off_before2 as c_int,
                        ) > 0
                    {
                        off_before2 = refpoc;
                        off_before2_idx = i;
                    }
                }
            }

            if off_before2 != 0xffffffff {
                refs = [
                    cmp::min(off_before_idx, off_before2_idx),
                    cmp::max(off_before_idx, off_before2_idx),
                ];
                allowed = 1;
            }
        }
    }
    let enabled = if allowed != 0 { gb.get_bit() as u8 } else { 0 };
    debug.post(gb, "extskip");
    Ok(Rav1dFrameSkipMode {
        allowed,
        enabled,
        refs,
    })
}

fn parse_gmv(
    state: &Rav1dState,
    frame_type: Rav1dFrameType,
    primary_ref_frame: u8,
    refidx: &[i8; RAV1D_REFS_PER_FRAME],
    hp: bool,
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dResult<[Rav1dWarpedMotionParams; RAV1D_REFS_PER_FRAME]> {
    let mut gmv = array::from_fn(|_| Rav1dWarpedMotionParams::default());

    if frame_type.is_inter_or_switch() {
        for (i, gmv) in gmv.iter_mut().enumerate() {
            gmv.r#type = if !gb.get_bit() {
                Rav1dWarpedMotionType::Identity
            } else if gb.get_bit() {
                Rav1dWarpedMotionType::RotZoom
            } else if gb.get_bit() {
                Rav1dWarpedMotionType::Translation
            } else {
                Rav1dWarpedMotionType::Affine
            };
            if gmv.r#type == Rav1dWarpedMotionType::Identity {
                continue;
            }

            let default_gmv = Default::default();
            let ref_gmv = if primary_ref_frame == RAV1D_PRIMARY_REF_NONE {
                &default_gmv
            } else {
                let pri_ref = refidx[primary_ref_frame as usize];
                &state.refs[pri_ref as usize]
                    .p
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(Rav1dError::InvalidArgument)?
                    .gmv[i]
            };
            let mat = &mut gmv.matrix;
            let ref_mat = &ref_gmv.matrix;
            let bits;
            let shift;

            if gmv.r#type >= Rav1dWarpedMotionType::RotZoom {
                mat[2] = (1 << 16) + 2 * gb.get_bits_subexp(ref_mat[2] - (1 << 16) >> 1, 12);
                mat[3] = 2 * gb.get_bits_subexp(ref_mat[3] >> 1, 12);

                bits = 12;
                shift = 10;
            } else {
                bits = 9 - !hp as c_int;
                shift = 13 + !hp as c_int;
            }

            if gmv.r#type == Rav1dWarpedMotionType::Affine {
                mat[4] = 2 * gb.get_bits_subexp(ref_mat[4] >> 1, 12);
                mat[5] = (1 << 16) + 2 * gb.get_bits_subexp(ref_mat[5] - (1 << 16) >> 1, 12);
            } else {
                mat[4] = -mat[3];
                mat[5] = mat[2];
            }

            mat[0] = gb.get_bits_subexp(ref_mat[0] >> shift, bits as c_uint) * (1 << shift);
            mat[1] = gb.get_bits_subexp(ref_mat[1] >> shift, bits as c_uint) * (1 << shift);
        }
    }
    debug.post(gb, "gmv");
    Ok(gmv)
}

fn parse_film_grain_data(
    seqhdr: &Rav1dSequenceHeader,
    seed: c_uint,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFilmGrainData> {
    let num_y_points = gb.get_bits(4) as c_int;
    if num_y_points > 14 {
        return Err(Rav1dError::InvalidArgument);
    }

    let mut y_points = [[0; 2]; 14];
    for i in 0..num_y_points {
        y_points[i as usize][0] = gb.get_bits(8) as u8;
        if i != 0 && y_points[(i - 1) as usize][0] as c_int >= y_points[i as usize][0] as c_int {
            return Err(Rav1dError::InvalidArgument);
        }
        y_points[i as usize][1] = gb.get_bits(8) as u8;
    }

    let chroma_scaling_from_luma = seqhdr.monochrome == 0 && gb.get_bit();
    let mut num_uv_points = [0; 2];
    let mut uv_points = [[[0; 2]; 10]; 2];
    if seqhdr.monochrome != 0
        || chroma_scaling_from_luma
        || seqhdr.ss_ver == 1 && seqhdr.ss_hor == 1 && num_y_points == 0
    {
        num_uv_points = [0; 2];
    } else {
        for pl in 0..2 {
            num_uv_points[pl as usize] = gb.get_bits(4) as c_int;
            if num_uv_points[pl as usize] > 10 {
                return Err(Rav1dError::InvalidArgument);
            }
            for i in 0..num_uv_points[pl as usize] {
                uv_points[pl as usize][i as usize][0] = gb.get_bits(8) as u8;
                if i != 0
                    && uv_points[pl as usize][(i - 1) as usize][0] as c_int
                        >= uv_points[pl as usize][i as usize][0] as c_int
                {
                    return Err(Rav1dError::InvalidArgument);
                }
                uv_points[pl as usize][i as usize][1] = gb.get_bits(8) as u8;
            }
        }
    }

    if seqhdr.ss_hor == 1
        && seqhdr.ss_ver == 1
        && (num_uv_points[0] != 0) != (num_uv_points[1] != 0)
    {
        return Err(Rav1dError::InvalidArgument);
    }

    let scaling_shift = gb.get_bits(2) as u8 + 8;
    let ar_coeff_lag = gb.get_bits(2) as c_int;
    let num_y_pos = 2 * ar_coeff_lag * (ar_coeff_lag + 1);
    let mut ar_coeffs_y = [0; 24];
    if num_y_points != 0 {
        for i in 0..num_y_pos {
            ar_coeffs_y[i as usize] = gb.get_bits(8).wrapping_sub(128) as i8;
        }
    }
    let mut ar_coeffs_uv = [[0; 28]; 2];
    for pl in 0..2 {
        if num_uv_points[pl as usize] != 0 || chroma_scaling_from_luma {
            let num_uv_pos = num_y_pos + (num_y_points != 0) as c_int;
            for i in 0..num_uv_pos {
                ar_coeffs_uv[pl as usize][i as usize] = gb.get_bits(8).wrapping_sub(128) as i8;
            }
            if num_y_points == 0 {
                ar_coeffs_uv[pl as usize][num_uv_pos as usize] = 0;
            }
        }
    }
    let ar_coeff_shift = gb.get_bits(2) as u8 + 6;
    let grain_scale_shift = gb.get_bits(2) as u8;
    let mut uv_mult = [0; 2];
    let mut uv_luma_mult = [0; 2];
    let mut uv_offset = [0; 2];
    for pl in 0..2 {
        if num_uv_points[pl as usize] != 0 {
            uv_mult[pl as usize] = gb.get_bits(8) as c_int - 128;
            uv_luma_mult[pl as usize] = gb.get_bits(8) as c_int - 128;
            uv_offset[pl as usize] = gb.get_bits(9) as c_int - 256;
        }
    }
    let overlap_flag = gb.get_bit();
    let clip_to_restricted_range = gb.get_bit();
    Ok(Rav1dFilmGrainData {
        seed,
        num_y_points,
        y_points,
        chroma_scaling_from_luma,
        num_uv_points,
        uv_points,
        scaling_shift,
        ar_coeff_lag,
        ar_coeffs_y,
        ar_coeffs_uv,
        ar_coeff_shift,
        grain_scale_shift,
        uv_mult,
        uv_luma_mult,
        uv_offset,
        overlap_flag,
        clip_to_restricted_range,
    })
}

fn parse_film_grain(
    state: &Rav1dState,
    seqhdr: &Rav1dSequenceHeader,
    show_frame: u8,
    showable_frame: u8,
    frame_type: Rav1dFrameType,
    ref_indices: &[i8; RAV1D_REFS_PER_FRAME],
    debug: &Debug,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameHeaderFilmGrain> {
    let present = (seqhdr.film_grain_present != 0
        && (show_frame != 0 || showable_frame != 0)
        && gb.get_bit()) as u8;
    let update;
    let data = if present != 0 {
        let seed = gb.get_bits(16);
        update = (frame_type != Rav1dFrameType::Inter || gb.get_bit()) as u8;
        if update == 0 {
            let refidx = gb.get_bits(3) as i8;
            let mut found = false;
            for i in 0..7 {
                if ref_indices[i as usize] == refidx {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Rav1dError::InvalidArgument);
            }
            Rav1dFilmGrainData {
                seed,
                ..state.refs[refidx as usize]
                    .p
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(Rav1dError::InvalidArgument)?
                    .film_grain
                    .data
                    .clone()
            }
        } else {
            parse_film_grain_data(seqhdr, seed, gb)?
        }
    } else {
        // Default initialization.
        update = Default::default();

        Default::default()
    };
    debug.post(gb, "filmgrain");
    Ok(Rav1dFrameHeaderFilmGrain {
        data,
        present,
        update,
    })
}

fn parse_frame_hdr(
    c: &Rav1dContext,
    state: &Rav1dState,
    seqhdr: &Rav1dSequenceHeader,
    temporal_id: u8,
    spatial_id: u8,
    gb: &mut GetBits,
) -> Rav1dResult<Rav1dFrameHeader> {
    let debug = Debug::new(false, "HDR", gb);

    debug.post(gb, "show_existing_frame");
    let show_existing_frame = (seqhdr.reduced_still_picture_header == 0 && gb.get_bit()) as u8;
    let existing_frame_idx;
    let mut frame_presentation_delay;
    if show_existing_frame != 0 {
        existing_frame_idx = gb.get_bits(3) as u8;
        if seqhdr.decoder_model_info_present != 0 && seqhdr.equal_picture_interval == 0 {
            frame_presentation_delay =
                gb.get_bits(seqhdr.frame_presentation_delay_length.into()) as u32;
        } else {
            // Default initialization.
            frame_presentation_delay = Default::default();
        }
        let frame_id;
        if seqhdr.frame_id_numbers_present != 0 {
            frame_id = gb.get_bits(seqhdr.frame_id_n_bits.into()) as u32;
            state.refs[existing_frame_idx as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .filter(|ref_frame_hdr| ref_frame_hdr.frame_id == frame_id)
                .ok_or(Rav1dError::InvalidArgument)?;
        } else {
            // Default initialization.
            frame_id = Default::default();
        }
        return Ok(Rav1dFrameHeader {
            spatial_id,
            temporal_id,
            show_existing_frame,
            existing_frame_idx,
            frame_presentation_delay,
            frame_id,
            // TODO(kkysen) I think an [`Option`] somewhere could avoid having to `#[derive(Default)]` everything.
            // There are also `enum`s that don't have a clear default other than being 0.
            ..Default::default()
        });
    } else {
        // Default initialization.
        existing_frame_idx = Default::default();
        frame_presentation_delay = Default::default();
    }

    let frame_type = if seqhdr.reduced_still_picture_header != 0 {
        Rav1dFrameType::Key
    } else {
        Rav1dFrameType::from_repr(gb.get_bits(2) as usize).unwrap()
    };
    let show_frame = (seqhdr.reduced_still_picture_header != 0 || gb.get_bit()) as u8;
    let showable_frame;
    if show_frame != 0 {
        if seqhdr.decoder_model_info_present != 0 && seqhdr.equal_picture_interval == 0 {
            frame_presentation_delay =
                gb.get_bits(seqhdr.frame_presentation_delay_length.into()) as u32;
        }
        showable_frame = (frame_type != Rav1dFrameType::Key) as u8;
    } else {
        showable_frame = gb.get_bit() as u8;
    }
    let error_resilient_mode = (frame_type == Rav1dFrameType::Key && show_frame != 0
        || frame_type == Rav1dFrameType::Switch
        || seqhdr.reduced_still_picture_header != 0
        || gb.get_bit()) as u8;
    debug.post(gb, "frametype_bits");
    let disable_cdf_update = gb.get_bit() as u8;
    let allow_screen_content_tools = match seqhdr.screen_content_tools {
        Rav1dAdaptiveBoolean::Adaptive => gb.get_bit(),
        Rav1dAdaptiveBoolean::On => true,
        Rav1dAdaptiveBoolean::Off => false,
    };
    let mut force_integer_mv = if allow_screen_content_tools {
        match seqhdr.force_integer_mv {
            Rav1dAdaptiveBoolean::Adaptive => gb.get_bit(),
            Rav1dAdaptiveBoolean::On => true,
            Rav1dAdaptiveBoolean::Off => false,
        }
    } else {
        false
    };

    if frame_type.is_key_or_intra() {
        force_integer_mv = true;
    }

    let frame_id;
    if seqhdr.frame_id_numbers_present != 0 {
        frame_id = gb.get_bits(seqhdr.frame_id_n_bits.into()) as u32;
    } else {
        // Default initialization.
        frame_id = Default::default();
    }

    let frame_size_override = if seqhdr.reduced_still_picture_header != 0 {
        false
    } else if frame_type == Rav1dFrameType::Switch {
        true
    } else {
        gb.get_bit()
    };
    debug.post(gb, "frame_size_override_flag");
    let frame_offset = if seqhdr.order_hint != 0 {
        gb.get_bits(seqhdr.order_hint_n_bits.into()) as u8
    } else {
        0
    };
    let primary_ref_frame = if error_resilient_mode == 0 && frame_type.is_inter_or_switch() {
        gb.get_bits(3) as u8
    } else {
        RAV1D_PRIMARY_REF_NONE
    };

    let buffer_removal_time_present;
    let mut operating_points =
        [Rav1dFrameHeaderOperatingPoint::default(); RAV1D_MAX_OPERATING_POINTS];
    if seqhdr.decoder_model_info_present != 0 {
        buffer_removal_time_present = gb.get_bit() as u8;
        if buffer_removal_time_present != 0 {
            for i in 0..seqhdr.num_operating_points {
                let seqop = &seqhdr.operating_points[i as usize];
                let op = &mut operating_points[i as usize];
                if seqop.decoder_model_param_present != 0 {
                    let in_temporal_layer = seqop.idc >> temporal_id & 1;
                    let in_spatial_layer = seqop.idc >> spatial_id + 8 & 1;
                    if seqop.idc == 0 || in_temporal_layer != 0 && in_spatial_layer != 0 {
                        op.buffer_removal_time =
                            gb.get_bits(seqhdr.buffer_removal_delay_length.into()) as u32;
                    }
                }
            }
        }
    } else {
        // Default initialization.
        buffer_removal_time_present = Default::default();
    }

    let refresh_frame_flags;
    let size;
    let refidx;
    let allow_intrabc;
    let use_ref_frame_mvs;
    let frame_ref_short_signaling;
    let hp;
    let subpel_filter_mode;
    let switchable_motion_mode;
    if frame_type.is_key_or_intra() {
        refresh_frame_flags = if frame_type == Rav1dFrameType::Key && show_frame != 0 {
            0xff
        } else {
            gb.get_bits(8) as u8
        };
        if refresh_frame_flags != 0xff && error_resilient_mode != 0 && seqhdr.order_hint != 0 {
            for _ in 0..8 {
                gb.get_bits(seqhdr.order_hint_n_bits.into());
            }
        }
        if c.strict_std_compliance
            && frame_type == Rav1dFrameType::Intra
            && refresh_frame_flags == 0xff
        {
            return Err(Rav1dError::InvalidArgument);
        }
        size = parse_frame_size(state, seqhdr, None, frame_size_override, gb)?;
        allow_intrabc = allow_screen_content_tools && !size.super_res.enabled && gb.get_bit();
        use_ref_frame_mvs = 0;

        // Default initialization.
        refidx = Default::default();
        frame_ref_short_signaling = Default::default();
        hp = Default::default();
        subpel_filter_mode = Rav1dFilterMode::Regular8Tap;
        switchable_motion_mode = Default::default();
    } else {
        allow_intrabc = false;
        refresh_frame_flags = if frame_type == Rav1dFrameType::Switch {
            0xff
        } else {
            gb.get_bits(8) as u8
        };
        if error_resilient_mode != 0 && seqhdr.order_hint != 0 {
            for _ in 0..8 {
                gb.get_bits(seqhdr.order_hint_n_bits.into());
            }
        }
        frame_ref_short_signaling = (seqhdr.order_hint != 0 && gb.get_bit()) as u8;
        refidx = parse_refidx(
            state,
            seqhdr,
            frame_ref_short_signaling,
            frame_offset,
            frame_id,
            gb,
        )?;
        let use_ref = error_resilient_mode == 0 && frame_size_override;
        size = parse_frame_size(
            state,
            seqhdr,
            Some(&refidx).filter(|_| use_ref),
            frame_size_override,
            gb,
        )?;
        hp = !force_integer_mv && gb.get_bit();
        subpel_filter_mode = if gb.get_bit() {
            Rav1dFilterMode::Switchable
        } else {
            Rav1dFilterMode::from_repr(gb.get_bits(2) as usize).unwrap()
        };
        // Plumb into the block-decode gate: an inter block codes its interp filter symbol
        // ONLY when the frame's subpel filter mode is Switchable (dav2d decode.c:3272).
        crate::av2_recon::HDR_TOOL_CFG.with(|c| {
            let mut cfg = c.get();
            cfg.subpel_filter_mode = subpel_filter_mode as u8;
            c.set(cfg);
        });
        switchable_motion_mode = gb.get_bit() as u8;
        use_ref_frame_mvs = (error_resilient_mode == 0
            && seqhdr.ref_frame_mvs != 0
            && seqhdr.order_hint != 0
            && frame_type.is_inter_or_switch()
            && gb.get_bit()) as u8;
    }
    debug.post(gb, "frametype-specific-bits");

    let refresh_context = (seqhdr.reduced_still_picture_header == 0
        && disable_cdf_update == 0
        && !gb.get_bit()) as u8;
    debug.post(gb, "refresh_context");

    let tiling = parse_tiling(seqhdr, &size, &debug, gb)?;
    let quant = parse_quant(seqhdr, &debug, gb);
    let segmentation = parse_segmentation(state, primary_ref_frame, &refidx, &quant, &debug, gb)?;
    let all_lossless = segmentation.lossless.iter().all(|&it| it);
    let delta = parse_delta(&quant, allow_intrabc, &debug, gb);
    let loopfilter = parse_loopfilter(
        state,
        seqhdr,
        all_lossless,
        allow_intrabc,
        primary_ref_frame,
        &refidx,
        &debug,
        gb,
    )?;
    let cdef = parse_cdef(seqhdr, all_lossless, allow_intrabc, &debug, gb);
    let restoration = parse_restoration(
        seqhdr,
        all_lossless,
        size.super_res.enabled,
        allow_intrabc,
        &debug,
        gb,
    );

    let txfm_mode = if all_lossless {
        Rav1dTxfmMode::Only4x4
    } else if gb.get_bit() {
        Rav1dTxfmMode::Switchable
    } else {
        Rav1dTxfmMode::Largest
    };
    debug.post(gb, "txfmmode");
    let switchable_comp_refs = if frame_type.is_inter_or_switch() {
        gb.get_bit() as u8
    } else {
        0
    };
    debug.post(gb, "refmode");
    let skip_mode = parse_skip_mode(
        state,
        seqhdr,
        switchable_comp_refs,
        frame_type,
        frame_offset,
        &refidx,
        &debug,
        gb,
    )?;
    let warp_motion = (error_resilient_mode == 0
        && frame_type.is_inter_or_switch()
        && seqhdr.warped_motion != 0
        && gb.get_bit()) as u8;
    debug.post(gb, "warpmotionbit");
    let reduced_txtp_set = gb.get_bit() as u8;
    debug.post(gb, "reducedtxtpset");

    let gmv = parse_gmv(
        state,
        frame_type,
        primary_ref_frame,
        &refidx,
        hp,
        &debug,
        gb,
    )?;
    let film_grain = parse_film_grain(
        state,
        seqhdr,
        show_frame,
        showable_frame,
        frame_type,
        &refidx,
        &debug,
        gb,
    )?;

    Ok(Rav1dFrameHeader {
        size,
        film_grain,
        frame_type,
        frame_offset,
        temporal_id,
        spatial_id,
        show_existing_frame,
        existing_frame_idx,
        frame_id,
        frame_presentation_delay,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        frame_size_override,
        primary_ref_frame,
        buffer_removal_time_present,
        operating_points,
        refresh_frame_flags,
        allow_intrabc,
        frame_ref_short_signaling,
        refidx,
        hp,
        subpel_filter_mode,
        switchable_motion_mode,
        use_ref_frame_mvs,
        refresh_context,
        tiling,
        quant,
        segmentation,
        delta,
        all_lossless,
        loopfilter,
        cdef,
        restoration,
        txfm_mode,
        switchable_comp_refs,
        skip_mode,
        warp_motion,
        reduced_txtp_set,
        gmv,
    })
}

fn parse_tile_hdr(tiling: &Rav1dFrameHeaderTiling, gb: &mut GetBits) -> Rav1dTileGroupHeader {
    let n_tiles = tiling.cols as c_int * tiling.rows as c_int;
    let have_tile_pos = if n_tiles > 1 {
        gb.get_bit() as c_int
    } else {
        0
    };

    if have_tile_pos != 0 {
        let n_bits = tiling.log2_cols + tiling.log2_rows;
        let start = gb.get_bits(n_bits.into()) as c_int;
        let end = gb.get_bits(n_bits.into()) as c_int;
        Rav1dTileGroupHeader { start, end }
    } else {
        Rav1dTileGroupHeader {
            start: 0,
            end: n_tiles - 1,
        }
    }
}

// Frame 1's fully-adapted entropy CDF, captured after its decode so the next (inter) frame's
// descent can INHERIT it — dav2d's primary_ref_frame CDF load (decode.c:5442). Proven necessary:
// frame 2's m.intrabc CDF is frame 1's ADAPTED [564,1283], not the default [683,1280]. The coef
// CDFs, however, re-init to the current frame's qcat default (block 0,0's coefs matched default).
thread_local! {
    static FRAME1_ADAPTED_CDF: std::cell::Cell<Option<crate::cdf_av2::CdfContext>> =
        const { std::cell::Cell::new(None) };
}

/// Emit the byte-exact filtered FRAME buffer (av2_frame::FRAME) as a real output picture.
/// Allocates a `Rav1dPicture` via the context allocator, copies the 3 planes (i32→u8, clamped,
/// respecting the picture stride), and sets `state.out` so `rav1d_get_picture` hands it to the
/// CLI muxer. Called once per decoded AV2 frame after its in-loop filter chain completes.
thread_local! {
    /// POC-ordered output pictures beyond the first of a temporal unit (dav2d's dpb queue):
    /// drained one per `gen_picture` call via `av2_pop_pending`.
    static AV2_PENDING: std::cell::RefCell<std::collections::VecDeque<[crate::av2_frame::Plane; 3]>> =
        const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

/// Pop one pending POC-ordered output picture into `state.out` (called from `gen_picture`
/// before parsing more input). Returns true if a picture was emitted.
pub(crate) fn av2_pop_pending(c: &Rav1dContext, state: &mut Rav1dState) -> Rav1dResult<bool> {
    if state.out.p.data.is_some() {
        return Ok(false);
    }
    let front = AV2_PENDING.with(|q| q.borrow_mut().pop_front());
    match front {
        Some(planes) => {
            emit_av2_planes(c, state, &planes)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// dav2d `dav2d_queue_output` (lib.c:353): on a SHOWN frame, first flush implicit-showable ref
/// pictures with dpb_poc < poc < cur (ascending), then the trigger frame, then chase the
/// `dpb_poc + 1` chain of implicit refs. First picture -> `state.out`, rest -> AV2_PENDING.
fn emit_av2_output(c: &Rav1dContext, state: &mut Rav1dState) -> Rav1dResult<()> {
    use crate::av2_recon::{get_poc_diff, AV2_DPB_POC, AV2_SHOW, CUR_REF_POC, REF_SLOTS};
    let (show, _implicit) = AV2_SHOW.with(|x| x.get());
    if !show {
        return Ok(()); // not shown now; its ref slot (show_implicit) feeds a later queue flush
    }
    let (nb, cur_poc, _) = CUR_REF_POC.with(|x| x.get());
    let mut dpb = AV2_DPB_POC.with(|x| x.get());
    let mut emits: Vec<[crate::av2_frame::Plane; 3]> = Vec::new();
    let mut picked = [false; 8];
    let slots = REF_SLOTS.with(|s| *s.borrow());
    // pre-scan: implicit refs strictly between the last output and the trigger, ascending poc
    loop {
        let mut cand = usize::MAX;
        let mut cand_poc = cur_poc;
        for (n, sl) in slots.iter().enumerate() {
            if picked[n] {
                continue;
            }
            let Some(sl) = sl else { continue };
            if !sl.show_implicit {
                continue;
            }
            let ipoc = sl.order_hint;
            if get_poc_diff(nb, ipoc, dpb) > 0 && get_poc_diff(nb, ipoc, cand_poc) < 0 {
                cand = n;
                cand_poc = ipoc;
            }
        }
        if cand == usize::MAX {
            break;
        }
        picked[cand] = true;
        if let Some(mut p) = crate::av2_frame::REF_PICS.with(|rp| rp.borrow()[cand].clone()) {
            if let Some((id, seed)) = crate::av2_grain::GRAIN_SLOTS.with(|g| g.borrow()[cand]) {
                crate::av2_grain::apply_grain_to_planes(&mut p, id, seed);
            }
            emits.push(p);
            dpb = cand_poc;
        }
    }
    // the trigger frame itself (current FRAME buffer)
    let mut cur_planes = crate::av2_frame::FRAME.with(|fr| {
        let f = fr.borrow();
        [f.pl[0].clone(), f.pl[1].clone(), f.pl[2].clone()]
    });
    if cur_planes[0].w != 0 {
        if let Some((id, seed)) = crate::av2_grain::CUR_GRAIN.with(|c| c.get()) {
            crate::av2_grain::apply_grain_to_planes(&mut cur_planes, id, seed);
        }
        emits.push(cur_planes);
        dpb = cur_poc;
    }
    // immediately-adjacent future implicit refs (dav lib.c:389): poc == dpb_poc + 1 chain
    loop {
        let mut found = usize::MAX;
        for (n, sl) in slots.iter().enumerate() {
            if picked[n] {
                continue;
            }
            let Some(sl) = sl else { continue };
            if !sl.show_implicit {
                continue;
            }
            if get_poc_diff(nb, sl.order_hint, dpb) == 1 {
                found = n;
                break;
            }
        }
        if found == usize::MAX {
            break;
        }
        picked[found] = true;
        if let Some(mut p) = crate::av2_frame::REF_PICS.with(|rp| rp.borrow()[found].clone()) {
            dpb = slots[found].as_ref().unwrap().order_hint;
            if let Some((id, seed)) = crate::av2_grain::GRAIN_SLOTS.with(|g| g.borrow()[found]) {
                crate::av2_grain::apply_grain_to_planes(&mut p, id, seed);
            }
            emits.push(p);
        }
    }
    AV2_DPB_POC.with(|x| x.set(dpb));
    let mut it = emits.into_iter();
    if let Some(first) = it.next() {
        emit_av2_planes(c, state, &first)?;
    }
    AV2_PENDING.with(|q| q.borrow_mut().extend(it));
    Ok(())
}

fn emit_av2_planes(c: &Rav1dContext, state: &mut Rav1dState, planes: &[crate::av2_frame::Plane; 3]) -> Rav1dResult<()> {
    use crate::include::common::bitdepth::BitDepth8;
    let seq_hdr = state.seq_hdr.clone().ok_or(Rav1dError::InvalidArgument)?;
    let (w, h) = (planes[0].w, planes[0].h);
    if w == 0 {
        return Ok(());
    }
    // Minimal frame header so the output path (has_grain → film_grain) doesn't unwrap a None;
    // film_grain defaults to no points, and the muxer takes dimensions from p.w/p.h, not this.
    let frame_hdr = Rav1dFrameHeader {
        size: Rav1dFrameSize {
            width: [w as c_int, w as c_int],
            height: h as c_int,
            render_width: w as c_int,
            render_height: h as c_int,
            ..Default::default()
        },
        ..Default::default()
    };
    let frame_hdr = Some(Arc::new(DRav1d::from_rav1d(frame_hdr)));
    let pic = c
        .allocator
        .alloc_picture_data(w as c_int, h as c_int, seq_hdr, frame_hdr)?;
    {
        let pd = pic.data.as_ref().unwrap();
        let hbd_code = pic.p.bpc > 8;
        let bdmax = (1i32 << pic.p.bpc) - 1;
        let hbd = hbd_code;
        for (pl, src) in planes.iter().enumerate() {
            if src.w == 0 {
                continue;
            }
            let stride_bytes = pic.stride[(pl != 0) as usize] as usize;
            let comp = &pd.data[pl];
            if hbd {
                use crate::include::common::bitdepth::BitDepth16;
                let stride_px = stride_bytes / 2;
                for y in 0..src.h {
                    let base = y * stride_px;
                    let mut row = comp.slice_mut::<BitDepth16, _>(base..base + src.w);
                    for x in 0..src.w {
                        row[x] = src.px[y * src.stride + x].clamp(0, bdmax) as u16;
                    }
                }
            } else {
                let stride_px = stride_bytes; // 8-bit: bytes == pixels
                for y in 0..src.h {
                    let base = y * stride_px;
                    let mut row = comp.slice_mut::<BitDepth8, _>(base..base + src.w);
                    for x in 0..src.w {
                        row[x] = src.px[y * src.stride + x].clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
    state.out = Rav1dThreadPicture {
        p: pic,
        visible: true,
        showable: false,
        flags: state.frame_flags,
        progress: None,
    };
    state.frame_flags = PictureFlags::empty(); // flags belong to the frame just emitted
    Ok(())
}

fn parse_obus(
    c: &Rav1dContext,
    state: &mut Rav1dState,
    r#in: &CArc<[u8]>,
    props: &Rav1dDataProps,
    gb: &mut GetBits,
) -> Rav1dResult<()> {
    fn skip(state: &mut Rav1dState) {
        // update refs with only the headers in case we skip the frame
        for i in 0..8 {
            if state.frame_hdr.as_ref().unwrap().refresh_frame_flags & (1 << i) != 0 {
                let _ = mem::take(&mut state.refs[i as usize].p);
                state.refs[i as usize].p.p.frame_hdr = state.frame_hdr.clone();
                state.refs[i as usize].p.p.seq_hdr = state.seq_hdr.clone();
            }
        }

        let _ = mem::take(&mut state.frame_hdr);
        state.n_tiles = 0;
    }

    // obu header — AV2 framing (dav2d obu.c): the leb128 length comes FIRST and is
    // mandatory; then a byte-aligned 1-byte header `has_extension(1) | type(5) |
    // tlayer_id(2)`, with an optional `mlayer_id(3) | xlayer_id(5)` extension byte.
    // (AV1 was: forbidden(1) | type(4) | ext_flag(1) | has_size(1) | reserved(1),
    //  with the size leb128 trailing — both layout and order changed in AV2.)
    let len = gb.get_uleb128() as usize;
    gb.bytealign();
    gb.set_remaining_len(len)
        .ok_or(Rav1dError::InvalidArgument)?;

    let has_extension = gb.get_bit();
    let raw_type = gb.get_bits(5);
    let r#type = Rav1dObuType::from_repr(raw_type as usize);
    let temporal_id = gb.get_bits(2) as u8; // AV2 tlayer_id

    let mut spatial_id = 0;
    let mut mlayer_id = 0u8;
    if has_extension {
        mlayer_id = gb.get_bits(3) as u8; // AV2 mlayer_id
        spatial_id = gb.get_bits(5) as u8; // AV2 xlayer_id
    }
    let _ = mlayer_id;
    if gb.has_error() != 0 {
        return Err(Rav1dError::InvalidArgument);
    }

    // We must have read a whole number of bytes at this point
    // (1 byte for the header and whole bytes at a time
    // when reading the leb128 length field).

    assert!(gb.is_byte_aligned());

    // skip obu not belonging to the selected temporal/spatial layer
    if !matches!(r#type, Some(Rav1dObuType::SeqHdr | Rav1dObuType::Td))
        && has_extension
        && state.operating_point_idc != 0
    {
        let in_temporal_layer = (state.operating_point_idc >> temporal_id & 1) as c_int;
        let in_spatial_layer = (state.operating_point_idc >> spatial_id + 8 & 1) as c_int;
        if in_temporal_layer == 0 || in_spatial_layer == 0 {
            return Ok(());
        }
    }

    fn parse_tile_grp(
        state: &mut Rav1dState,
        r#in: &CArc<[u8]>,
        props: &Rav1dDataProps,
        gb: &mut GetBits,
    ) -> Rav1dResult {
        let hdr = parse_tile_hdr(
            &state
                .frame_hdr
                .as_ref()
                .ok_or(Rav1dError::InvalidArgument)?
                .tiling,
            gb,
        );
        // Align to the next byte boundary and check for overrun.
        gb.bytealign();
        if gb.has_error() != 0 {
            return Err(Rav1dError::InvalidArgument);
        }

        let mut data = r#in.clone();
        {
            // HARDENING: a corrupt tile-group header can leave byte_pos past the payload or
            // remaining_len larger than what is left — either slice would panic in c_arc.
            let raw: &[u8] = r#in;
            if gb.byte_pos() > raw.len()
                || gb.remaining_len() > raw.len().saturating_sub(gb.byte_pos())
            {
                return Err(Rav1dError::InvalidArgument);
            }
        }
        data.slice_in_place(gb.byte_pos()..);
        data.slice_in_place(..gb.remaining_len());
        // Ensure tile groups are in order and sane; see 6.10.1.
        if hdr.start > hdr.end || hdr.start != state.n_tiles {
            state.tiles.clear();
            state.n_tiles = 0;
            return Err(Rav1dError::InvalidArgument);
        }
        if let Err(_) = state.tiles.try_reserve_exact(1) {
            return Err(Rav1dError::InvalidArgument);
        }
        state.n_tiles += 1 + hdr.end - hdr.start;
        state.tiles.push(Rav1dTileGroup {
            data: Rav1dData {
                data: Some(data),
                // TODO(kkysen) Are props needed here?
                // Also, if it's not needed, we don't need the `Option` for `CArc<[u8]>` either.
                m: props.clone(),
            },
            hdr,
        });

        Ok(())
    }

    match r#type {
        Some(Rav1dObuType::SeqHdr) => {
            let seq_hdr = parse_seq_hdr(gb, c.strict_std_compliance).inspect_err(|_| {
                writeln!(c.logger, "Error parsing sequence header");
            })?;
            if gb.has_error() != 0 {
                return Err(Rav1dError::InvalidArgument);
            }

            let op_idx = if c.operating_point < seq_hdr.num_operating_points {
                c.operating_point
            } else {
                0
            };
            state.operating_point_idc = seq_hdr.operating_points[op_idx as usize].idc as c_uint;
            let spatial_mask = state.operating_point_idc >> 8;
            state.max_spatial_id = if spatial_mask != 0 {
                ulog2(spatial_mask) as u8
            } else {
                0
            };

            // If we have read a sequence header which is different from the old one,
            // this is a new video sequence and can't use any previous state.
            // Free that state.

            match &state.seq_hdr {
                None => {
                    state.frame_hdr = None;
                    state.frame_flags |= PictureFlags::NEW_SEQUENCE;
                }
                Some(c_seq_hdr) if !seq_hdr.eq_without_operating_parameter_info(&c_seq_hdr) => {
                    // See 7.5, `operating_parameter_info` is allowed to change in
                    // sequence headers of a single sequence.
                    // A DIFFERENT sequence: clear ALL cross-frame AV2 state (ref slots,
                    // CDF stashes, motion fields, grain, caches) so nothing leaks.
                    crate::av2_recon::reset_av2_stream_state();
                    state.frame_hdr = None;
                    let _ = mem::take(&mut state.content_light);
                    let _ = mem::take(&mut state.mastering_display);
                    for i in 0..8 {
                        if state.refs[i as usize].p.p.frame_hdr.is_some() {
                            let _ = mem::take(&mut state.refs[i as usize].p);
                        }
                        let _ = mem::take(&mut state.refs[i as usize].segmap);
                        let _ = mem::take(&mut state.refs[i as usize].refmvs);
                        let _ = mem::take(&mut state.cdf[i]);
                    }
                    state.frame_flags |= PictureFlags::NEW_SEQUENCE;
                }
                Some(c_seq_hdr)
                    if seq_hdr.operating_parameter_info != c_seq_hdr.operating_parameter_info =>
                {
                    // If operating_parameter_info changed, signal it
                    state.frame_flags |= PictureFlags::NEW_OP_PARAMS_INFO;
                }
                _ => {}
            }
            state.seq_hdr = Some(Arc::new(DRav1d::from_rav1d(seq_hdr))); // TODO(kkysen) fallible allocation
        }
        Some(t) if t.is_frame() => {
            // AV2 frame OBU (dav2d obu.c:2436): all frame-carrying types (KF/tile-grp/
            // SEF/TIP/bridge/switch) route here. `first_tile`/`has_hdr` gate whether a
            // frame header is (re)parsed before the tile data.
            state.frame_hdr = None;
            let first_tile = t.is_sef_like() || gb.get_bit();
            let has_hdr = first_tile || gb.get_bit();
            // Clone the Arc (not borrow) so `state` stays free for the per-frame
            // `emit_av2_picture(state)` output hook; `.field` access derefs transparently.
            let seq_hdr = state.seq_hdr.clone().ok_or(Rav1dError::InvalidArgument)?;
            crate::dlog!(
                "[rav2d AV2] frame OBU type={raw_type} first_tile={first_tile} has_hdr={has_hdr}"
            );
            let (yac, frame_type) = if has_hdr {
                parse_av2_frame_hdr_front(&seq_hdr, t, gb)?
            } else {
                (0, 0)
            };
            // Tile-group framing (dav2d parse_tile_hdr, obu.c:2325): multi-tile frames read a
            // have_tile_pos bit (+ start/end when set) BEFORE the byte-align; single-tile
            // frames read 0 bits. The remaining OBU bytes are the tile data.
            let tinfo = crate::av2_recon::TILE_INFO.with(|c| c.get());
            let n_tiles = tinfo.cols as usize * tinfo.rows as usize;
            if n_tiles > 1 && gb.get_bit() {
                let nb = (tinfo.log2_cols + tinfo.log2_rows) as c_int;
                gb.get_bits(nb); // tile start
                gb.get_bits(nb); // tile end
            }
            gb.bytealign();
            let tile_data_len = gb.remaining_len();
            let tile_off = gb.byte_pos();
            // HARDENING: a corrupt header can leave the bit reader past the OBU payload — a
            // slice past the buffer would panic in c_arc::slice_in_place. Bail gracefully.
            {
                let raw: &[u8] = r#in;
                if tile_off > raw.len() || tile_off + tile_data_len > raw.len() {
                    return Err(Rav1dError::InvalidArgument);
                }
            }
            // Split the tile data into per-tile (offset, len) slices: each tile except the
            // last is prefixed by an n_bytes little-endian (size-1) field (dav2d decode.c:5043).
            let tile_slices: Vec<(usize, usize)> = {
                let raw: &[u8] = r#in;
                let mut v = Vec::with_capacity(n_tiles);
                let (mut off, mut rem) = (tile_off, tile_data_len);
                for j in 0..n_tiles {
                    if j + 1 == n_tiles {
                        v.push((off, rem));
                    } else {
                        let mut sz = 0usize;
                        for k in 0..tinfo.n_bytes as usize {
                            // HARDENING: a corrupt tile-count/size prefix can run past the
                            // OBU. `off + k` is checked with checked_add — a plain `off + k`
                            // can itself wrap and silently pass the very bound it guards.
                            if off.checked_add(k).is_none_or(|i| i >= raw.len()) {
                                return Err(Rav1dError::InvalidArgument);
                            }
                            sz |= (raw[off + k] as usize) << (k * 8);
                        }
                        sz += 1;
                        // HARDENING: `sz` is attacker-controlled (assembled from up to
                        // n_bytes of stream data). Both subtractions below underflow — and
                        // panic — when the declared tile size exceeds the tile data actually
                        // left, so check against `rem` instead of assuming it fits.
                        let nb = tinfo.n_bytes as usize;
                        if rem < nb || rem - nb < sz {
                            return Err(Rav1dError::InvalidArgument);
                        }
                        off += nb;
                        rem -= nb;
                        v.push((off, sz));
                        off += sz;
                        rem -= sz;
                    }
                }
                v
            };
            if std::env::var("TDBG").is_ok() {
                crate::dlog!("[TDBG] n_tiles={n_tiles} n_bytes={} slices={:?} total={tile_data_len}", tinfo.n_bytes, tile_slices.iter().map(|&(o, l)| (o - tile_off, l)).collect::<Vec<_>>());
            }

            // R1 foundation: initialize the MSAC arithmetic decoder on the real tile data.
            // (NOTE: decoding bypass bits below only proves the entropy core is live + the
            // tile is framed; the first MEANINGFUL symbol is CDF-coded — bit-exact decode
            // needs the CDF tables + partition logic + YUV-vs-avmdec, the ~30k-LOC R1 body.)
            // === TIP-as-output frame (dav2d decode.c:4424 tip_frame_recon_sb + the
            // frame_without_data flow, obu.c:2729): no tile data, no symbols — synthesize the
            // whole TIP frame per SB, stash the UNADAPTED init CDF (disable_cdf_update), skip
            // all filters (apply_filter=0), and emit. ===
            if crate::av2_recon::HDR_TOOL_CFG.with(|cc| cc.get().tip_frame_mode) == 2 {
                let refidx2 = crate::av2_recon::CUR_FRAME_REFIDX.with(|cc| cc.get()).1;
                let p_ref = crate::av2_recon::CUR_PRIMARY_REF.with(|cc| cc.get());
                // use_pri_sec_cdf excludes tip.frame_mode==2 (dav decode.c:5394) → plain primary.
                let cdf_fm2 = if p_ref != 7 {
                    crate::cdf_av2::load_cdf(refidx2[p_ref as usize] as usize)
                        .unwrap_or_else(|| crate::cdf_av2::default_cdf_context(yac, crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.get()) as usize))
                } else {
                    crate::cdf_av2::default_cdf_context(yac, crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.get()) as usize)
                };
                let f2w = seq_hdr.max_width as usize;
                let f2h = seq_hdr.max_height as usize;
                let f2iw4 = (f2w + 3) / 4;
                let f2ih4 = (f2h + 3) / 4;
                crate::av2_recon::work_reset();
                crate::av2_refmvs::reset_refmvs();
                crate::av2_frame::F2_YAC.with(|cc| cc.set(yac));
                crate::av2_recon::LAST_QIDX.with(|cc| cc.set(yac));
                crate::av2_frame::DECODE_FRAME_N.with(|cc| cc.set(cc.get() + 1));
                crate::av2_refmvs::rp_reset(f2iw4, f2ih4);
                crate::av2_frame::reset_frame(f2w, f2h, yac as u8, crate::av2_recon::SEQ_TOOLS.with(|c| c.get().edge_filter), crate::av2_recon::SEQ_TOOLS.with(|c| c.get().ibp), (1i32 << (8 + 2 * seq_hdr.hbd as i32)) - 1);
                crate::av2_frame::RECON_ACTIVE.with(|a| a.set(true));
                let sbst_fm2 = if seq_hdr.sb128 != 0 { 32usize } else { 16 };
                crate::av2_recon::SB_STEP4.store(sbst_fm2, std::sync::atomic::Ordering::Relaxed);
                for by_sb in (0..f2ih4).step_by(sbst_fm2) {
                    crate::av2_refmvs::load_tmvs((by_sb >> 1) as i32, ((by_sb + sbst_fm2) >> 1) as i32);
                    for bx_sb in (0..f2iw4).step_by(sbst_fm2) {
                        crate::av2_recon::tip_recon_sb(bx_sb, by_sb, f2iw4, f2ih4);
                    }
                }
                let refresh = crate::av2_recon::CUR_FRAME_REF.with(|cc| cc.get()).1;
                // fm2 frames decode NO symbols: dav's update_set stays 0, so with seq
                // avg_cdf_type==0 the stash is the inherited cdf VERBATIM (no reset walk);
                // avg_cdf_type!=0 takes dav's always-save branch (reset applied).
                if seq_hdr.av2.avg_cdf_type != 0 {
                    crate::cdf_av2::stash_cdf(&cdf_fm2, refresh, false);
                } else {
                    crate::cdf_av2::stash_cdf_verbatim(&cdf_fm2, refresh);
                }
                crate::av2_frame::filter_frame_chain(0); // all filter cfgs off → identity
                crate::av2_frame::stash_decoded_frame1();
                crate::av2_recon::FRAME_DECODE_COUNT.with(|cc| cc.set(cc.get() + 1));
                crate::av2_recon::update_ref_slots(false);
                crate::av2_grain::stash_grain_slots(crate::av2_recon::CUR_FRAME_REF.with(|cc| cc.get()).1);
                emit_av2_output(c, state)?;
                return Ok(());
            }
            let mut tile_data = r#in.clone();
            {
                let raw: &[u8] = r#in;
                if tile_off.checked_add(tile_data_len).map_or(true, |e| e > raw.len()) {
                    return Err(Rav1dError::InvalidArgument);
                }
            }
            tile_data.slice_in_place(tile_off..tile_off + tile_data_len);
            let mut msac = MsacContext::new(tile_data, false, &c.dsp.msac);
            // INTER (1) and SWITCH/S-frame (3) both decode through the inter tree (an s-frame
            // is an inter frame with explicit refs, no temporal MVs, and a default-init CDF).
            if frame_type == 1 || frame_type == 3 {
                crate::dlog!("F2TILE init dif={:x} rng={} len={}", msac.dif, msac.rng, tile_data_len);
                // D: inter partition descent into the first child to the first leaf — the shared
                // decode_partition should reproduce the oracle's PART2 path (64x64 H → 64x32 V →
                // 32x32 H → 32x16 V → 16x16 NONE, rng → 59456). Verifies the inter partition decode.
                let mut tile_d = r#in.clone();
                { let raw: &[u8] = r#in; if tile_off.checked_add(tile_data_len).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); } }
                tile_d.slice_in_place(tile_off..tile_off + tile_data_len);
                let mut msac_d = MsacContext::new(tile_d, false, &c.dsp.msac);
                // EXPERIMENTAL: run the GENERAL decode_sb_inter recursion on a FRESH msac+cdf to
                // verify it reproduces the oracle's per-leaf block-ends (SBLEAF vs BLKDIF), without
                // disturbing the verified hand-unrolled descent below.
                {
                    let mut tile_sb = r#in.clone();
                    { let raw: &[u8] = r#in; if tile_off.checked_add(tile_data_len).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); } }
                tile_sb.slice_in_place(tile_off..tile_off + tile_data_len);
                    let mut msac_sb = MsacContext::new(tile_sb, false, &c.dsp.msac);
                    // CDF init (dav2d decode.c:5388-5413): primary_ref==NONE → default(qcat);
                    // else inherit refidx[primary_ref]'s stashed CDF — or, when use_pri_sec_cdf
                    // (secondary != NONE && inter && seq avg_cdf && !avg_cdf_type && tip.mode != 2),
                    // the 7:1 per-u16 average of primary+secondary (cdf.c pri_sec_average).
                    let mut cdf_sb = {
                        let p_ref = crate::av2_recon::CUR_PRIMARY_REF.with(|c| c.get());
                        let s_ref = crate::av2_recon::CUR_SECONDARY_REF.with(|c| c.get());
                        let seq_avg_cdf = crate::av2_recon::SEQ_COMP.with(|c| c.get()).3;
                        let tip_mode = crate::av2_recon::HDR_TOOL_CFG.with(|c| c.get().tip_frame_mode);
                        let use_pri_sec = p_ref != 7 && s_ref != 7 && seq_avg_cdf
                            && seq_hdr.av2.avg_cdf_type == 0 && tip_mode != 2;
                        let refidx = crate::av2_recon::CUR_FRAME_REFIDX.with(|c| c.get()).1;
                        let inherited = if p_ref != 7 {
                            let pri = crate::cdf_av2::load_cdf(refidx[p_ref as usize] as usize);
                            if use_pri_sec {
                                let sec = crate::cdf_av2::load_cdf(refidx[s_ref as usize] as usize);
                                match (pri, sec) {
                                    (Some(p), Some(s)) => Some(crate::cdf_av2::CdfContext::avg_7_1(&p, &s)),
                                    (p, _) => p,
                                }
                            } else {
                                pri
                            }
                        } else {
                            None
                        };
                        let inh_flag = inherited.is_some();
                        let c = inherited.unwrap_or_else(|| crate::cdf_av2::default_cdf_context(yac, crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.get()) as usize));
                        if std::env::var("MCDF").is_ok() {
                            crate::dlog!("[MCDF] part_split[0][12]={:?} rt[0][24]={:?} ext[0][12]={:?} pal_y={:?}", c.m.part_split[0][12], c.m.part_dir[0][24], c.m.part_ext[0][12], c.m.pal_y);
                            crate::dlog!("[MCDF] yac={yac} p_ref={p_ref} s_ref={s_ref} use_pri_sec={use_pri_sec} inherited={inh_flag} eob_bin_32[1]={:?} skip[1][1][0]={:?} tip_drl0={:?}",
                                c.coef.eob_bin_32[1], c.coef.skip[1][1][0], c.m.tip_drl_idx[0]);
                        }
                        c
                    };
                    // Tile-wide neighbour context (above row + left column), indexed by absolute
                    // 4px coords so distinct SB rows use distinct ranges (no per-row reset needed,
                    // mirroring the frame-1 SbState loop). msac+cdf persist across ALL superblocks.
                    let mut ap = vec![0u8; crate::av2_recon::nb_len()];
                    let mut lp = vec![0u8; crate::av2_recon::nb_len()];
                    let mut anb = crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len());
                    let mut lnb = crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len());
                    let mut cnb = crate::av2_recon::ChromaNb::new(crate::av2_recon::nb_len());
                    // 2D SB loop over the whole frame (432x240 → 7x4 64px SBs). luma_dir_map resets
                    // per SB (it's the intra-dir map for the SB's chroma tree). The first SB is
                    // fully bit-exact (42/42 leaves); later SBs drive out the rest of the 1060.
                    crate::av2_recon::work_reset();
                crate::av2_refmvs::reset_refmvs(); // brick B: fresh MV grid + bank for frame 2
                    crate::av2_frame::F2_YAC.with(|c| c.set(yac as u32)); // Stage E: residual dequant
                    crate::av2_recon::LAST_QIDX.with(|c| c.set(yac as u32)); // delta-q running qindex reset
                    crate::av2_frame::DECODE_FRAME_N.with(|c| c.set(c.get() + 1));
                    // Stage E: assemble the frame-2 RECON buffer. Allocate FRAME + enable recon so
                    // every decoded block (inter + intra) writes its recon into FRAME in decode
                    // order → frame-2 intra blocks can predict from reconstructed neighbours.
                    // Arbitrary dims: drive the inter frame from the real frame size (was hardcoded
                    // 432×240 → decoded off-frame SBs at bx4=80/96 for a 320-wide clip).
                    let f2w = seq_hdr.max_width as usize;
                    let f2h = seq_hdr.max_height as usize;
                    let f2iw4 = (f2w + 3) / 4;
                    let f2ih4 = (f2h + 3) / 4;
                    // Fresh temporal motion field for this frame (saved to ref slots at emit).
                    crate::av2_refmvs::rp_reset(f2iw4 as usize, f2ih4 as usize);
                    crate::av2_palette::pal_reset(f2iw4 as usize, f2ih4 as usize);
                    crate::av2_frame::reset_frame(f2w, f2h, yac as u8, crate::av2_recon::SEQ_TOOLS.with(|c| c.get().edge_filter), crate::av2_recon::SEQ_TOOLS.with(|c| c.get().ibp), (1i32 << (8 + 2 * seq_hdr.hbd as i32)) - 1);
                    crate::av2_frame::RECON_ACTIVE.with(|a| a.set(true));
                    // The intra recon gathers neighbours from REF_LUMA/REF_CHROMA when loaded (the
                    // frame-1 isolation harness). For frame 2 those hold STALE frame-1 pixels — clear
                    // them so the gather falls back to the assembled frame-2 FRAME (== REF for frame
                    // 1 since bit-exact; correct here). Frame-2 uses IFRAMEY/IFRAMEC for scoring.
                    crate::av2_frame::REF_LUMA.with(|r| *r.borrow_mut() = None);
                    crate::av2_frame::REF_CHROMA.with(|r| *r.borrow_mut() = [None, None]);
                    // Stage D: load the inter-prediction reference (frame-1 FILTERED output) +
                    // the dav2d frame-2 luma prediction plane for the per-block MC harness.
                    // Prefer mine's own stashed frame 1 (standalone decode); fall back to the dav
                    // reference file only if no frame was stashed (e.g. isolated frame-2 testing).
                    // Primary reference = REF_PICS[refidx[0]] (the correct implicit-scored primary
                    // ref slot). Fall back to the P-chain stash, then the external dav file.
                    let primary_slot = crate::av2_recon::CUR_FRAME_REFIDX.with(|c| c.get().1[0]) as usize;
                    if !crate::av2_frame::load_ref_frame1_from_slot(primary_slot)
                        && !crate::av2_frame::load_ref_frame1_from_stash()
                        && std::env::var("DAVCAP").is_ok()
                    {
                        crate::av2_frame::load_ref_frame1(&crate::av2_recon::cap_path("dav_filtered.yuv"), 432, 240);
                    }
                    if std::env::var("DAVCAP").is_ok() {
                    if let Ok(b) = std::fs::read(&crate::av2_recon::cap_path("dav_f2pred.yuv")) {
                        if b.len() >= 432 * 240 {
                            let mut p = crate::av2_frame::Plane::alloc(432, 240);
                            for i in 0..432 * 240 { p.px[i] = b[i] as i32; }
                            crate::av2_frame::REF_F2PRED.with(|r| *r.borrow_mut() = Some(p));
                            crate::av2_frame::INTER_SCORE.with(|s| s.set((0, 0)));
                        }
                    }
                    if let Ok(b) = std::fs::read(&crate::av2_recon::cap_path("dav_f2recon.yuv")) {
                        if b.len() >= 432 * 240 {
                            let mut p = crate::av2_frame::Plane::alloc(432, 240);
                            for i in 0..432 * 240 { p.px[i] = b[i] as i32; }
                            crate::av2_frame::REF_F2RECON.with(|r| *r.borrow_mut() = Some(p));
                            crate::av2_frame::INTER_SCORE_R.with(|s| s.set((0, 0)));
                        }
                    }
                    if let Ok(b) = std::fs::read(&crate::av2_recon::cap_path("dav_f2predc.yuv")) {
                        let csz = 216 * 120;
                        if b.len() >= 2 * csz {
                            let mut pu = crate::av2_frame::Plane::alloc(216, 120);
                            let mut pv = crate::av2_frame::Plane::alloc(216, 120);
                            for i in 0..csz { pu.px[i] = b[i] as i32; pv.px[i] = b[csz + i] as i32; }
                            crate::av2_frame::REF_F2PREDC.with(|r| *r.borrow_mut() = [Some(pu), Some(pv)]);
                            crate::av2_frame::INTER_SCORE_C.with(|s| s.set((0, 0, 0)));
                        }
                    }
                    if let Ok(b) = std::fs::read(&crate::av2_recon::cap_path("dav_f2reconc.yuv")) {
                        let csz = 216 * 120;
                        if b.len() >= 2 * csz {
                            let mut pu = crate::av2_frame::Plane::alloc(216, 120);
                            let mut pv = crate::av2_frame::Plane::alloc(216, 120);
                            for i in 0..csz { pu.px[i] = b[i] as i32; pv.px[i] = b[csz + i] as i32; }
                            crate::av2_frame::REF_F2RECONC.with(|r| *r.borrow_mut() = [Some(pu), Some(pv)]);
                            crate::av2_frame::INTER_SCORE_RC.with(|s| s.set((0, 0, 0)));
                        }
                    }
                    }
                    let sbst = if seq_hdr.sb128 != 0 { 32usize } else { 16 };
                    crate::av2_recon::SB_STEP4.store(sbst, std::sync::atomic::Ordering::Relaxed);
                    let root_bs = if seq_hdr.sb128 != 0 { 3usize } else { 6 };
                    if n_tiles > 1 {
                        // Multi-tile INTER frame (mirrors the key multi-tile branch): each tile is
                        // an independent entropy stream — fresh MSAC on its slice, a CLONE of the
                        // inherited frame-init CDF, fresh neighbour/partition/chroma ctx, fresh
                        // refmvs grid+bank (tile isolation), TILE_B = the tile bounds. Decoded
                        // tile-sequentially; the load_tmvs window carry across a tile switch only
                        // pollutes rows -2/-1 of the tile's FIRST SB row, which no in-frame block
                        // reads (single-tile-ROW configs; multi-tile-ROW would need a window reset).
                        let tinfo_i = crate::av2_recon::TILE_INFO.with(|t| t.get());
                        let cdf_init = cdf_sb;
                        let mut tile_cdfs_i: Vec<crate::cdf_av2::CdfContext> = Vec::new();
                        for trow in 0..tinfo_i.rows as usize {
                            for tcol in 0..tinfo_i.cols as usize {
                                let j = trow * tinfo_i.cols as usize + tcol;
                                let (toff, tlen) = tile_slices[j];
                            {
                                let raw: &[u8] = r#in;
                                if toff.checked_add(tlen).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); }
                            }
                                {
                                    let raw: &[u8] = r#in;
                                    if toff.checked_add(tlen).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); }
                                }
                                let mut td = r#in.clone();
                                { let raw: &[u8] = r#in; if toff.checked_add(tlen).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); } }
                                td.slice_in_place(toff..toff + tlen);
                                let mut msac_t = MsacContext::new(td, false, &c.dsp.msac);
                                let mut cdf_t = cdf_init;
                                let cs4 = tinfo_i.col_start4[tcol] as usize;
                                let ce4 = (tinfo_i.col_start4[tcol + 1] as usize).min(f2iw4);
                                let rs4 = tinfo_i.row_start4[trow] as usize;
                                let re4 = (tinfo_i.row_start4[trow + 1] as usize).min(f2ih4);
                                crate::av2_recon::TILE_B.with(|t| t.set((cs4, ce4, rs4, re4)));
                                let mut ap_t = vec![0u8; crate::av2_recon::nb_len()];
                                let mut lp_t = vec![0u8; crate::av2_recon::nb_len()];
                                let mut anb_t = crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len());
                                let mut lnb_t = crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len());
                                let mut cnb_t = crate::av2_recon::ChromaNb::new(crate::av2_recon::nb_len());
                                crate::av2_recon::work_reset();
                crate::av2_refmvs::reset_refmvs();
                                crate::av2_recon::LAST_QIDX.with(|c| c.set(yac as u32));
                                crate::av2_lr::lr_tile_init();
                                for by_sb in (rs4..re4).step_by(sbst) {
                                    crate::av2_refmvs::load_tmvs((by_sb >> 1) as i32, ((by_sb + sbst) >> 1) as i32);
                                    crate::av2_refmvs::reset_sbrow();
                                    let mut left_cdef = -1i8;
                                    let mut left_ccso = [0u8; 3];
                                    for bx_sb in (cs4..ce4).step_by(sbst) {
                                        crate::av2_refmvs::reset_sb(bx_sb, by_sb, sbst, f2iw4, by_sb == rs4);
                                        crate::av2_lr::read_lr_units_sb(
                                            &mut msac_t, &mut cdf_t, bx_sb, by_sb, f2iw4, f2ih4,
                                            (seq_hdr.layout != Rav1dPixelLayout::I444) as usize,
                                            (seq_hdr.layout == Rav1dPixelLayout::I420) as usize,
                                        );
                                        let mut ldm = [0xffu8; 256];
                                        crate::av2_recon::decode_sb_inter(
                                            &mut msac_t, &mut cdf_t, &mut ap_t, &mut lp_t, &mut anb_t, &mut lnb_t, &mut cnb_t,
                                            root_bs, bx_sb, by_sb, f2iw4, f2ih4, false, &mut ldm, root_bs as i32, bx_sb, by_sb,
                                            &mut left_cdef, &mut left_ccso,
                                        );
                                        if std::env::var("MSBT").is_ok() {
                                            crate::dlog!("[MSBT] mi=({bx_sb},{by_sb}) rng={} cnt={}", msac_t.rng, msac_t.cnt);
                                        }
                                    }
                                }
                                tile_cdfs_i.push(cdf_t);
                            }
                        }
                        crate::av2_recon::TILE_B.with(|t| t.set((0, 1 << 30, 0, 1 << 30)));
                        // Stash = the tiles' shift-average (seq avg_cdf_type, dav decode.c:5283).
                        let log2 = (tinfo_i.log2_cols + tinfo_i.log2_rows) as u32;
                        cdf_sb = crate::cdf_av2::tile_avg_cdf(&tile_cdfs_i, log2);
                    } else {
                    crate::av2_lr::lr_tile_init();
                    for by_sb in (0..f2ih4).step_by(sbst) {
                        // Temporal projection for this SB row (dav load_tmvs per sbrow).
                        crate::av2_refmvs::load_tmvs((by_sb >> 1) as i32, ((by_sb + sbst) >> 1) as i32);
                        crate::av2_refmvs::tpl_dump_window((by_sb >> 1) as i32, ((by_sb + sbst) >> 1) as i32);
                        if std::env::var("TMVSDBG").map_or(false, |v| v == format!("{}", crate::av2_recon::CUR_FRAME_REF.with(|c| c.get().0))) {
                            crate::av2_refmvs::tmvs_dump((by_sb >> 1) as i32, ((by_sb + 16) >> 1) as i32);
                        }
                        // dav2d tile_sbrow_init: reset refmv+warp bank size/idx to 0 at each SB-row
                        // start (the row-above entries become logically empty). Before reset_sb.
                        crate::av2_refmvs::reset_sbrow();
                        // Filter neighbour state resets at the leftmost SB of each SB-row (no left
                        // neighbour): left_cdef=-1, left_ccso=0. Threaded across the row's SBs.
                        let mut left_cdef = -1i8;
                        let mut left_ccso = [0u8; 3];
                        for bx_sb in (0..f2iw4).step_by(sbst) {
                            // dav2d reset_sb: reset bank hits + seed banks from the above-SB-row
                            // (skipped on the first SB row). Must fire before the SB's blocks.
                            crate::av2_refmvs::reset_sb(bx_sb, by_sb, sbst, f2iw4, by_sb == 0);
                            // Restoration units (dav decode.c:4590: read BEFORE the SB's tree).
                            crate::av2_lr::read_lr_units_sb(
                                &mut msac_sb, &mut cdf_sb, bx_sb, by_sb, f2iw4, f2ih4,
                                (seq_hdr.layout != Rav1dPixelLayout::I444) as usize,
                                (seq_hdr.layout == Rav1dPixelLayout::I420) as usize,
                            );
                            let mut ldm = [0xffu8; 256];
                            crate::av2_recon::decode_sb_inter(
                                &mut msac_sb, &mut cdf_sb, &mut ap, &mut lp, &mut anb, &mut lnb, &mut cnb,
                                root_bs, bx_sb, by_sb, f2iw4, f2ih4, false, &mut ldm, root_bs as i32, bx_sb, by_sb,
                                &mut left_cdef, &mut left_ccso,
                            );
                            if std::env::var("MSBT").is_ok() {
                                crate::dlog!("[MSBT] mi=({bx_sb},{by_sb}) rng={} cnt={}", msac_sb.rng, msac_sb.cnt);
                            }
                        }
                    }
                    }
                    // Stash this frame's adapted CDF into its refresh slots for the NEXT frame's
                    // primary_ref inheritance (dav2d out_cdf → c->cdf[refresh]).
                    let refresh = crate::av2_recon::CUR_FRAME_REF.with(|c| c.get()).1;
                    crate::cdf_av2::stash_cdf(&cdf_sb, refresh, false);
                }
                crate::av2_frame::INTER_SCORE.with(|s| {
                    let (o, t) = s.get();
                    if t > 0 {
                        crate::dlog!("[IMCSCORE] Stage-D translational MC (first SB, single-ref MM_SIMPLE): {o}/{t} luma blocks bit-exact vs dav2d ({:.1}%)", 100.0 * o as f64 / t as f64);
                    }
                });
                crate::av2_frame::INTER_SCORE_R.with(|s| {
                    let (o, t) = s.get();
                    if t > 0 {
                        crate::dlog!("[IRECON] Stage-E luma RECON (MC prediction + residual, full-frame non-BAWP): {o}/{t} luma blocks bit-exact vs dav2d pre-filter recon ({:.1}%)", 100.0 * o as f64 / t as f64);
                    }
                });
                // Whole-frame LUMA byte-exact: the assembled frame-2 FRAME luma vs dav's f2recon.
                crate::av2_frame::FRAME.with(|fr| {
                    crate::av2_frame::REF_F2RECON.with(|rr| {
                        let f = fr.borrow();
                        if let Some(rp) = rr.borrow().as_ref() {
                            if f.pl[0].w != 0 {
                                let (mut ok, mut tot, mut unwritten, mut wrong, mut first_wrong) = (0usize, 0usize, 0usize, 0usize, None);
                                let btype = crate::av2_frame::BTYPE.with(|b| b.borrow().clone());
                                // misses per block type: [unwritten, inter, intra, intrabc, bawp]
                                let mut miss_by_type = [0usize; 5];
                                let mut first_by_type: [Option<(usize, usize, i32, i32)>; 5] = [None; 5];
                                let mut first_unwritten: Option<(usize, usize, usize)> = None;
                                let mut unwritten_by_type = [0usize; 5];
                                let mut first_untag_unwr: Option<(usize, usize)> = None;
                                let mut first_intra_unwr: Option<(usize, usize)> = None;
                                for y in 0..f.pl[0].h.min(rp.h) {
                                    for x in 0..f.pl[0].w.min(rp.w) {
                                        tot += 1;
                                        let m = f.pl[0].px[y * f.pl[0].stride + x];
                                        if m == rp.at(x, y) { ok += 1; }
                                        else {
                                            let t = (btype.get(y * f.pl[0].w + x).copied().unwrap_or(0) as usize).min(4);
                                            miss_by_type[t] += 1;
                                            if first_by_type[t].is_none() { first_by_type[t] = Some((x, y, m, rp.at(x, y))); }
                                            if m == 0 { unwritten += 1; unwritten_by_type[t] += 1; if first_unwritten.is_none() { first_unwritten = Some((x, y, t)); } if t == 0 && first_untag_unwr.is_none() { first_untag_unwr = Some((x, y)); } if t == 2 && first_intra_unwr.is_none() { first_intra_unwr = Some((x, y)); } }
                                            else { wrong += 1; if first_wrong.is_none() { first_wrong = Some((x, y, m, rp.at(x, y))); } }
                                        }
                                    }
                                }
                                crate::dlog!("[IFRAMEY] first-wrong by type: untagged={:?} inter={:?} intra={:?} intrabc={:?} bawp={:?}", first_by_type[0], first_by_type[1], first_by_type[2], first_by_type[3], first_by_type[4]);
                                crate::dlog!("[IFRAMEY] frame-2 assembled LUMA vs dav2d pre-filter recon: {ok}/{tot} px bit-exact ({:.2}%); miss: {unwritten} unwritten(mine=0) + {wrong} wrong; first-wrong={first_wrong:?}", 100.0 * ok as f64 / tot as f64);
                                crate::dlog!("[IFRAMEY] miss by block-type: untagged={} inter={} intra={} intrabc={} bawp={}; first-unwritten(x,y,btype)={first_unwritten:?}", miss_by_type[0], miss_by_type[1], miss_by_type[2], miss_by_type[3], miss_by_type[4]);
                                crate::dlog!("[IFRAMEY] UNWRITTEN by block-type: untagged={} inter={} intra={} intrabc={} bawp={}; first-untagged-unwr={first_untag_unwr:?} first-intra-unwr={first_intra_unwr:?}", unwritten_by_type[0], unwritten_by_type[1], unwritten_by_type[2], unwritten_by_type[3], unwritten_by_type[4]);
                                if std::env::var("BRDBG").is_ok() {
                                    // Dump the bottom-right SB region: mine vs dav, every 4th pixel, rows 200..240 cols 380..432.
                                    let st = f.pl[0].stride;
                                    for y in (200..240).step_by(4) {
                                        let mine_row: Vec<i32> = (380..432).step_by(4).map(|x| f.pl[0].px[y * st + x]).collect();
                                        let dav_row: Vec<i32> = (380..432).step_by(4).map(|x| rp.at(x, y)).collect();
                                        crate::dlog!("BRDUMP y={y} mine={mine_row:?}");
                                        crate::dlog!("BRDUMP y={y} dav ={dav_row:?}");
                                    }
                                }
                            }
                        }
                    });
                });
                // Whole-frame CHROMA byte-exact: the assembled frame-2 FRAME U/V vs dav's f2reconc.
                crate::av2_frame::FRAME.with(|fr| {
                    crate::av2_frame::REF_F2RECONC.with(|rr| {
                        let f = fr.borrow();
                        let rb = rr.borrow();
                        for pl in 0..2 {
                            if let Some(rp) = rb[pl].as_ref() {
                                if f.pl[pl + 1].w != 0 {
                                    let (mut ok, mut tot, mut unwritten, mut wrong, mut first_wrong) = (0usize, 0usize, 0usize, 0usize, None);
                                    for y in 0..f.pl[pl + 1].h.min(rp.h) {
                                        for x in 0..f.pl[pl + 1].w.min(rp.w) {
                                            tot += 1;
                                            let m = f.pl[pl + 1].px[y * f.pl[pl + 1].stride + x];
                                            if m == rp.at(x, y) { ok += 1; }
                                            else if m == 0 { unwritten += 1; }
                                            else { wrong += 1; if first_wrong.is_none() { first_wrong = Some((x, y, m, rp.at(x, y))); } }
                                        }
                                    }
                                    crate::dlog!("[IFRAMEC] frame-2 assembled CHROMA pl={pl} vs dav2d pre-filter recon: {ok}/{tot} px ({:.2}%); miss: {unwritten} unwritten + {wrong} wrong; first-wrong={first_wrong:?}", 100.0 * ok as f64 / tot as f64);
                                }
                            }
                        }
                    });
                });
                // ===== STAGE C per-stage ISOLATION on FRAME 2 (like frame 1): apply each filter to
                // dav's CORRECT frame-2 previous-stage input with mine's frame-2 grids/params, score
                // vs dav's frame-2 this-stage. Isolates which filter's params are wrong. =====
                // DAVCAP: dav2d capture-file oracles (filter stage isolation). Plain runs
                // skip them — stale captures poisoned live debugging twice (see memory).
                if std::env::var("DAVCAP").is_ok() {
                crate::dlog!("--- FRAME-2 filter stage isolation ---");
                crate::av2_frame::dump_frame2_luma_wmap(&crate::av2_recon::cap_path("mine_f2wmap.bin"));
                // Same-run oracle: pre-deblock recon → post-deblock (both from dav's normal decode,
                // so the frame-1 reference is consistent). The --inloopfilters-token files are INVALID
                // for inter frames (each run filters frame-1 differently → different frame-2 recon).
                crate::av2_frame::run_deblock_verify(
                    &crate::av2_recon::cap_path("dav_f2predeblk.yuv"),
                    &crate::av2_recon::cap_path("dav_f2postdeblk.yuv"),
                );
                crate::av2_frame::run_cdef_verify(
                    &crate::av2_recon::cap_path("dav_f2lf_deblk.yuv"),
                    &crate::av2_recon::cap_path("dav_f2lf_cdef.yuv"),
                );
                crate::av2_frame::run_ccso_verify(
                    &crate::av2_recon::cap_path("dav_f2lf_deblk.yuv"),
                    &crate::av2_recon::cap_path("dav_f2lf_cdef.yuv"),
                    &crate::av2_recon::cap_path("dav_f2lf_ccso.yuv"),
                );
                crate::av2_frame::run_gdf_verify(
                    &crate::av2_recon::cap_path("dav_f2lf_ccso.yuv"),
                    &crate::av2_recon::cap_path("dav_f2lf_all.yuv"),
                    &crate::av2_recon::cap_path("dav_f2lf_deblk.yuv"),
                );
                }
                // ===== STAGE C: chain the 4 in-loop filters (deblock→CDEF→CCSO→GDF) on the
                // assembled FRAME buffer → the real frame-2 filtered output. Compare vs dav2d's
                // frame-2 decoded (post-filter) output = the 2nd frame in dav_out.yuv. =====
                // ISOLATION TEST: overwrite FRAME with dav's CORRECT pre-filter recon so [IFILT]
                // measures the FILTERS alone (independent of mine's 91% recon). Remove to measure
                // the real end-to-end output.
                if std::env::var("FILT_ISO").is_ok() {
                    if let (Ok(rl), Ok(rc)) = (std::fs::read(&crate::av2_recon::cap_path("dav_f2recon.yuv")), std::fs::read(&crate::av2_recon::cap_path("dav_f2reconc.yuv"))) {
                        crate::av2_frame::FRAME.with(|fr| {
                            let mut f = fr.borrow_mut();
                            for i in 0..432 * 240 { f.pl[0].px[i] = rl[i] as i32; }
                            for i in 0..216 * 120 { f.pl[1].px[i] = rc[i] as i32; f.pl[2].px[i] = rc[216 * 120 + i] as i32; }
                        });
                    }
                }
                // env OH1PRE: dump this frame's PRE-FILTER recon when it's the oh=1 frame.
                if std::env::var("OH1PRE").is_ok() {
                    crate::av2_frame::FRAME.with(|fr| {
                        let f = fr.borrow();
                        let mut out = Vec::new();
                        for pl in 0..3 {
                            let p = &f.pl[pl];
                            for y in 0..p.h {
                                for x in 0..p.w {
                                    out.push(p.at(x, y) as u8);
                                }
                            }
                        }
                        let oh = crate::av2_recon::CUR_FRAME_REF.with(|c| c.get().0);
                        let path = (crate::av2_recon::cap_path(&format!("mine_oh{oh}_prefilter.yuv")));
                        let _ = std::fs::write(path, &out);
                    });
                }
                // GDF ref_dst_idx (dav decode.c:4899): tbl[min(max_dist,11)] over the first
                // <=2 refs' absrefdist (v432-f2 dist-1 -> 1; oh=1 dists {1,2} -> 2).
                let gdf_rd = {
                    const TBL: [usize; 12] = [5, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5];
                    let (_rd, absd, _ffr) = crate::av2_recon::CUR_REFDIST.with(|c| c.get());
                    let n_ref = crate::av2_recon::CUR_FRAME_REFIDX.with(|c| c.get()).0 as usize;
                    let mut max_dist = 0i32;
                    for i in 0..n_ref.min(2) {
                        max_dist = max_dist.max(absd[i]);
                    }
                    TBL[(max_dist as usize).min(11)]
                };
                crate::av2_frame::filter_frame_chain(gdf_rd); // frame-2 inter single-ref dist-1 → inter GDF tables
                if std::env::var("DAVCAP").is_ok() {
                if let Ok(b) = std::fs::read(&crate::av2_recon::cap_path("dav_out.yuv")) {
                    let fsz = 432 * 240 + 2 * 216 * 120; // 155520 bytes/frame
                    if b.len() >= 2 * fsz {
                        let f2 = &b[fsz..2 * fsz]; // frame-2 filtered
                        crate::av2_frame::FRAME.with(|fr| {
                            let f = fr.borrow();
                            for (pl, (pw, ph, off)) in [(432usize, 240usize, 0usize), (216, 120, 432 * 240), (216, 120, 432 * 240 + 216 * 120)].into_iter().enumerate() {
                                if f.pl[pl].w == 0 { continue; }
                                let (mut ok, mut tot, mut first) = (0usize, 0usize, None);
                                for y in 0..ph.min(f.pl[pl].h) {
                                    for x in 0..pw.min(f.pl[pl].w) {
                                        tot += 1;
                                        let m = f.pl[pl].px[y * f.pl[pl].stride + x];
                                        let d = f2[off + y * pw + x] as i32;
                                        if m == d { ok += 1; } else if first.is_none() { first = Some((x, y, m, d)); }
                                    }
                                }
                                crate::dlog!("[IFILT] frame-2 FILTERED pl={pl} vs dav2d final output: {ok}/{tot} px ({:.2}%) first-miss={first:?}", 100.0 * ok as f64 / tot as f64);
                            }
                        });
                    }
                }
                }
                // Multi-frame P-chain: stash THIS decoded+filtered inter frame so the NEXT inter
                // frame's `load_ref_frame1_from_stash` references it (was only the keyframe stashing
                // → frame 3 wrongly referenced frame 1). Also re-stash the adapted CDF for the next
                // frame's primary_ref inheritance (default-CDF clips are unaffected).
                crate::av2_frame::stash_decoded_frame1();
                crate::av2_recon::FRAME_DECODE_COUNT.with(|c| c.set(c.get() + 1));
                crate::av2_recon::update_ref_slots(false); // inter frame
                crate::av2_grain::stash_grain_slots(crate::av2_recon::CUR_FRAME_REF.with(|cc| cc.get()).1);
                // Hand the byte-exact filtered frame to the output pipeline, then RETURN — the
                // real inter decode (this msac_sb block) is complete and emitted. The block below
                // (`msac_d` hand-decode-trace scaffold, F2(x,y) prints) is DEAD diagnostic code that
                // panics on generic streams (decode_partition bs underflow); skipping it lets the
                // CLI muxer write the frame (and finish the prior flush) → standalone byte-identical.
                emit_av2_output(c, state)?;
                return Ok(());
            }
            // R1 BLOCK DECODE — first symbol: the top-level superblock partition, decoded
            // through the live MSAC using the generated + spec-validated CDF context.
            // (Full partition recursion needs the neighbour-context infra; here we decode
            // the first part_split symbol at the tile's top-left SB, ctx [0][0].)
            let mut cdf = crate::cdf_av2::default_cdf_context(yac, crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.get()) as usize);
            // M1 — real partition recursion (decode_sb), top-left descent, each
            // read_partition's rng diffed against the dav2d oracle. seq: aspect_log2=3,
            // ext_partitions=1. BS_64x64 = index 6.
            crate::dlog!("[rav2d TRACE] tile_bytes={tile_data_len} qidx={yac}");
            // ===== FULL RECURSIVE decode_sb (fresh MSAC) — sweep the whole first SB's
            // luma through the real recursion + decode_b_luma, verified at the oracle. =====
            {
                let mut tile_data2 = r#in.clone();
                { let raw: &[u8] = r#in; if tile_off.checked_add(tile_data_len).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); } }
                tile_data2.slice_in_place(tile_off..tile_off + tile_data_len);
                crate::msac::MSAC_SUM_D.with(|p| p.set(0)); // isolate the sweep
                let mut msac2 = MsacContext::new(tile_data2, false, &c.dsp.msac);
                let mut cdf2 = crate::cdf_av2::default_cdf_context(yac, crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.get()) as usize);
                // Tile-wide neighbour context (sized for the frame); the SB loop threads it
                // across SB boundaries. The above context persists down the column and the
                // left context across a row — both indexed by absolute 4px coords, so distinct
                // SB-rows use distinct index ranges and need no per-row reset.
                let mut sb = crate::av2_recon::SbState {
                    msac: &mut msac2,
                    cdf: &mut cdf2,
                    a_part: vec![0u8; crate::av2_recon::nb_len()],
                    l_part: vec![0u8; crate::av2_recon::nb_len()],
                    a_nb: crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len()),
                    l_nb: crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len()),
                    filters_done: false,
                    // Frame RESOLVED force_integer_mv (SCC frames force integer BVs — no
                    // intrabc precision bit); def_max_bvp_drl_bits=3.
                    force_integer_mv: crate::av2_recon::HDR_TOOL_CFG.with(|c| c.get().force_integer_mv),
                    max_bvp_drl_bits: 3,
                    luma_dir_map: [0xff; 256],
                    left_cdef: -1,
                    left_ccso: [0; 3],
                    // Frame dims in 4px units (432x240 → 108x60). Plumb from the real header.
                    iw4: 108,
                    ih4: 60,
                    top_cdef: [-1; 8],
                };
                // Chroma partition (`partition[1]`) + coefficient (`ccoef`) context, tile-wide.
                let mut uv_a = vec![0u8; crate::av2_recon::nb_len()];
                let mut uv_l = vec![0u8; crate::av2_recon::nb_len()];
                let mut uv_nb = crate::av2_recon::ChromaNb::new(crate::av2_recon::nb_len());
                // Multi-SB loop over the first SB row (per-SB filter params — gdf/cdef/ccso
                // with their alignment + neighbour context — and the luma dir-map reset per
                // SB; `left_cdef` resets to -1 at the row's leftmost SB). SB #0+#1 are fully
                // bit-exact; SB #2 starts with a 64x64-TX block (AV2 codes it as a 32x32
                // core — needs the TX-size clamp, the next brick), so the range stops there.
                // Frame-1 SB#0+#1+#2 (319 luma leaves) are FULLY bit-exact. Widening to
                // [0,16,32,48,64,80] reaches 349 matching leaves, then SB#3 diverges at the
                // bottom-16x4 of the (56-59,3) region: rav2d's partition splits the LEFT 8x4
                // and keeps the RIGHT whole, while the oracle does the opposite (a phantom
                // (57,3) leaf, absent from the oracle, with (56,3) still matching — an
                // 8x4-partition-context subtlety to chase next). Kept at the verified range.
                // Frame-1 SB#0+#1+#2 = 319 luma leaves, fully bit-exact. The skip_txfm
                // block-skip fix carries SB#3 to 350 matching leaves; it then hits a 4x4
                // intrabc block-vector divergence at (59,3) and an unwired intrabc inter-stx
                // (16x16+ DCT_DCT) panic — the next two intrabc bricks. Verified range here.
                // 2D SB loop over the whole frame (432x240 → 7x4 64px SBs). Per SB ROW the
                // left context resets (`left_cdef=-1`, `left_ccso=0` — the row's leftmost SB
                // has no left neighbour); `top_cdef` persists down columns. The SDP order is
                // per-SB: luma tree then chroma tree.
                // Gate to frame 1 only: this is the frame-1 intra verification path. Running it
                // for frame 2 (inter data through the intra decoder) just produces garbage that
                // contaminates the shared dct_y/decode_coefs probes.
                if frame_type == 0 {
                // TODO: thread real seq `enable_ibp` (parsed+discarded at the AV2 seq path ~786);
                // this clip has ibp=1 (confirmed by the DC/z1/z3 boundary gradient in the oracle).
                // Header plumbing (rusty_av2e E3): dims + filter-tool gating come from the
                // PARSED headers instead of the dev clip's hardcoded values. The dev clip
                // (432x240, gdf/cdef/ccso on) takes the exact same path as before.
                let fw = seq_hdr.max_width as usize;
                let fh = seq_hdr.max_height as usize;
                let iw4 = (fw + 3) / 4;
                let ih4 = (fh + 3) / 4;
                sb.iw4 = iw4;
                sb.ih4 = ih4;
                crate::av2_recon::HDR_TOOL_CFG.with(|c| {
                    // Frame-level enables (set during parse_av2_frame_hdr_front above)
                    // are the normative per-SB symbol gates; AND with the seq flags so
                    // seq-disabled tools stay off even when the frame parse skipped them.
                    let mut cfg = c.get();
                    cfg.gdf = cfg.gdf && seq_hdr.av2.gdf != 0;
                    cfg.cdef = cfg.cdef && seq_hdr.cdef != 0;
                    cfg.ccso = cfg.ccso && seq_hdr.av2.ccso != 0;
                    c.set(cfg);
                });
                // TODO: thread real seq `enable_intra_edge_filter` + `enable_ibp` — both are parsed
                // then DISCARDED at the AV2 seq path (~779/786), so seq_hdr defaults them to 0. This
                // clip has both = 1 (z1/z3 edges are smoothed + DC/z1/z3 IBP gradient in the oracle).
                crate::av2_frame::reset_frame(fw, fh, yac as u8, crate::av2_recon::SEQ_TOOLS.with(|c| c.get().edge_filter), crate::av2_recon::SEQ_TOOLS.with(|c| c.get().ibp), (1i32 << (8 + 2 * seq_hdr.hbd as i32)) - 1); // RECON: allocate the frame buffer
                crate::av2_frame::RECON_ACTIVE.with(|a| a.set(true)); // recon writes FRAME (was set by load_ref_luma)
                crate::av2_frame::F2_YAC.with(|c| c.set(yac as u32)); // frame-1 intrabc residual AC dequant (was only set for frame 2)
                crate::av2_recon::LAST_QIDX.with(|c| c.set(yac as u32)); // delta-q running qindex reset
                    crate::av2_frame::DECODE_FRAME_N.with(|c| c.set(c.get() + 1));
                // The per-block isolation refs (REF_LUMA/REF_CHROMA) are a GOLDEN-SPECIFIC crutch:
                // gathers read them instead of mine's own recon. Correct only when the loaded file IS
                // this stream's recon (golden). For any other stream it feeds STALE golden pixels →
                // wrong prediction. DEFAULT = self-referential (read mine's own FRAME); env ISOLATE
                // re-enables the crutch for golden-only per-block bring-up scoring.
                if std::env::var("ISOLATE").is_ok() {
                    crate::av2_frame::load_ref_luma(&crate::av2_recon::cap_path("dav_f1luma.bin"), fw, fh);
                    crate::av2_frame::load_ref_chroma(
                        &crate::av2_recon::cap_path("dav_lf_none.yuv"),
                        (fw + 1) >> 1, (fh + 1) >> 1, fw * fh,
                    );
                }
                // Frame 1's intrabc blocks derive their block-vector via `refmvs_find(ref=-1)` over
                // the SAME grid/bank machinery as frame 2 (brick B). The keyframe path previously
                // never built it → every intrabc block fell back to the (0,-2560)/… defaults. Reset
                // + per-SB-row/per-SB seed it here; the leaves splat their BVs (av2_recon.rs).
                crate::av2_recon::work_reset();
                crate::av2_refmvs::reset_refmvs();
                crate::av2_palette::pal_reset(
                    (seq_hdr.max_width as usize + 3) / 4,
                    (seq_hdr.max_height as usize + 3) / 4,
                );
                let tinfo2 = crate::av2_recon::TILE_INFO.with(|t| t.get());
                let n_tiles2 = tinfo2.cols as usize * tinfo2.rows as usize;
                let mut tile_cdfs: Vec<crate::cdf_av2::CdfContext> = Vec::new();
                if n_tiles2 > 1 {
                    // Multi-tile keyframe: each tile is an independent entropy stream — fresh
                    // MSAC over its slice, fresh CDF (keys init from qcat default), fresh
                    // neighbour/partition/chroma ctx, fresh refmvs bank; TILE_B publishes the
                    // tile bounds so availability gates against the tile origin (dav ts->tiling).
                    for trow in 0..tinfo2.rows as usize {
                        for tcol in 0..tinfo2.cols as usize {
                            let j = trow * tinfo2.cols as usize + tcol;
                            let (toff, tlen) = tile_slices[j];
                            {
                                let raw: &[u8] = r#in;
                                if toff.checked_add(tlen).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); }
                            }
                            let mut td = r#in.clone();
                            { let raw: &[u8] = r#in; if toff.checked_add(tlen).map_or(true, |e| e > raw.len()) { return Err(Rav1dError::InvalidArgument); } }
                                td.slice_in_place(toff..toff + tlen);
                            let mut msac_t = MsacContext::new(td, false, &c.dsp.msac);
                            let mut cdf_t = crate::cdf_av2::default_cdf_context(yac, crate::av2_recon::SEQ_REDUCED_TX_PART.with(|c| c.get()) as usize);
                            let cs4 = tinfo2.col_start4[tcol] as usize;
                            let ce4 = (tinfo2.col_start4[tcol + 1] as usize).min(iw4);
                            let rs4 = tinfo2.row_start4[trow] as usize;
                            let re4 = (tinfo2.row_start4[trow + 1] as usize).min(ih4);
                            crate::av2_recon::TILE_B.with(|t| t.set((cs4, ce4, rs4, re4)));
                            let mut sbt = crate::av2_recon::SbState {
                                msac: &mut msac_t,
                                cdf: &mut cdf_t,
                                a_part: vec![0u8; crate::av2_recon::nb_len()],
                                l_part: vec![0u8; crate::av2_recon::nb_len()],
                                a_nb: crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len()),
                                l_nb: crate::av2_recon::BlockNbCtx::new(crate::av2_recon::nb_len()),
                                filters_done: false,
                                force_integer_mv: crate::av2_recon::HDR_TOOL_CFG.with(|c| c.get().force_integer_mv),
                                max_bvp_drl_bits: 3,
                                luma_dir_map: [0xff; 256],
                                left_cdef: -1,
                                left_ccso: [0; 3],
                                iw4,
                                ih4,
                                top_cdef: [-1; 8],
                            };
                            let mut uv_a_t = vec![0u8; crate::av2_recon::nb_len()];
                            let mut uv_l_t = vec![0u8; crate::av2_recon::nb_len()];
                            let mut uv_nb_t = crate::av2_recon::ChromaNb::new(crate::av2_recon::nb_len());
                            crate::av2_recon::work_reset();
                crate::av2_refmvs::reset_refmvs();
                            let sbst_t = if seq_hdr.sb128 != 0 { 32usize } else { 16 };
                            crate::av2_recon::SB_STEP4.store(sbst_t, std::sync::atomic::Ordering::Relaxed);
                            let root_bs_t = if seq_hdr.sb128 != 0 { 3usize } else { 6 };
                            for by_sb in (rs4..re4).step_by(sbst_t) {
                                sbt.left_cdef = -1;
                                sbt.left_ccso = [0; 3];
                                crate::av2_refmvs::reset_sbrow();
                                for bx_sb in (cs4..ce4).step_by(sbst_t) {
                                    crate::av2_refmvs::reset_sb(bx_sb, by_sb, sbst_t, iw4, true);
                                    sbt.luma_dir_map = [0xff; 256];
                                    sbt.filters_done = false;
                                    crate::av2_recon::decode_sb_key(
                                        &mut sbt, &mut uv_a_t, &mut uv_l_t, &mut uv_nb_t, root_bs_t, bx_sb, by_sb,
                                    );
                                    if std::env::var("MSBT").is_ok() {
                                        crate::dlog!("[MSBT] mi=({bx_sb},{by_sb}) rng={} cnt={}", sbt.msac.rng, sbt.msac.cnt);
                                    }
                                }
                            }
                            tile_cdfs.push(cdf_t);
                        }
                    }
                    crate::av2_recon::TILE_B.with(|t| t.set((0, 1 << 30, 0, 1 << 30)));
                } else {
                let sbst = if seq_hdr.sb128 != 0 { 32usize } else { 16 };
                crate::av2_recon::SB_STEP4.store(sbst, std::sync::atomic::Ordering::Relaxed);
                let root_bs = if seq_hdr.sb128 != 0 { 3usize } else { 6 };
                crate::av2_lr::lr_tile_init();
                for by_sb in (0..ih4).step_by(sbst) {
                    sb.left_cdef = -1;
                    sb.left_ccso = [0; 3];
                    crate::av2_refmvs::reset_sbrow();
                    for bx_sb in (0..iw4).step_by(sbst) {
                        // Keyframe: dav's reset_sb SKIPS the above-SB-row bank re-seed
                        // (`IS_KEY_OR_INTRA` early-return, refmvs.c:1330). Pass first_sb_row=true for
                        // EVERY frame-1 SB so only the bank hits reset (not the re-seed) runs.
                        crate::av2_refmvs::reset_sb(bx_sb, by_sb, sbst, iw4, true);
                        // Restoration units (dav decode.c:4590: read BEFORE the SB's tree).
                        crate::av2_lr::read_lr_units_sb(
                            sb.msac, sb.cdf, bx_sb, by_sb, sb.iw4, sb.ih4,
                            (seq_hdr.layout != Rav1dPixelLayout::I444) as usize,
                            (seq_hdr.layout == Rav1dPixelLayout::I420) as usize,
                        );
                        sb.luma_dir_map = [0xff; 256];
                        sb.filters_done = false;
                        crate::av2_recon::decode_sb_key(
                            &mut sb, &mut uv_a, &mut uv_l, &mut uv_nb, root_bs, bx_sb, by_sb,
                        );
                        if std::env::var("MSBT").is_ok() {
                            crate::dlog!("[MSBT] mi=({bx_sb},{by_sb}) rng={} cnt={}", sb.msac.rng, sb.msac.cnt);
                        }
                    }
                }
                }
                crate::dlog!(
                    "[R-TELL] sum_d={} pulled={} cnt={} tell={}",
                    crate::msac::MSAC_SUM_D.with(|p| p.get()),
                    crate::msac::MSAC_PULLED.with(|p| p.get()),
                    sb.msac.cnt,
                    crate::msac::MSAC_PULLED.with(|p| p.get()) as i64 * 8
                        - sb.msac.cnt as i64
                        - 14
                );
                // Stash frame 1's fully-adapted CDF for the next (inter) frame to inherit.
                // Multi-tile + seq avg_cdf_type: the stash is the tiles' SHIFT-AVERAGE
                // (dav2d decode.c:5283 cdf_shift + accumulate), not any single tile's cdf.
                let adapted = if tile_cdfs.len() > 1 {
                    let log2 = (tinfo2.log2_cols + tinfo2.log2_rows) as u32;
                    crate::cdf_av2::tile_avg_cdf(&tile_cdfs, log2)
                } else {
                    *sb.cdf
                };
                FRAME1_ADAPTED_CDF.with(|cell| cell.set(Some(adapted)));
                // RECON: dump the reconstructed luma plane (pre-loop-filter) for verification.
                // (DAVCAP-gated: capture-oracle scaffolding, off in plain runs.)
                if std::env::var("DAVCAP").is_ok() {
                crate::av2_frame::FRAME.with(|fr| {
                    let f = fr.borrow();
                    let p = &f.pl[0];
                    let mut buf = Vec::with_capacity(p.w * p.h);
                    for y in 0..p.h {
                        for x in 0..p.w {
                            buf.push(p.px[y * p.stride + x].clamp(0, 255) as u8);
                        }
                    }
                    let _ = std::fs::write(&crate::av2_recon::cap_path("rav2d_f1luma.bin"), &buf);
                    crate::dlog!("[RECONDUMP] wrote frame-1 luma {}x{} ({} bytes)", p.w, p.h, buf.len());
                    // Whole-frame ASSEMBLED f.pl vs dav's ref (REF_LUMA) — catches recon GAPS that the
                    // per-block REFSCORE misses (a gap = no block runs there = never scored).
                    crate::av2_frame::REF_LUMA.with(|r| {
                        if let Some(rp) = r.borrow().as_ref() {
                            let (mut ok, mut wrong, mut zero, mut first) = (0usize, 0usize, 0usize, None);
                            let (mut first_gap, mut first_wrong): (Option<(usize,usize)>, Option<(usize,usize,i32,i32)>) = (None, None);
                            for y in 0..p.h { for x in 0..p.w {
                                let m = p.px[y * p.stride + x].clamp(0, 255);
                                let d = rp.at(x, y);
                                if m == d { ok += 1; } else { wrong += 1; if m == 0 { zero += 1; if first_gap.is_none() { first_gap = Some((x,y)); } } else if first_wrong.is_none() { first_wrong = Some((x,y,m,d)); } if first.is_none() { first = Some((x, y, m, d)); } }
                            }}
                            crate::dlog!("[F1ASSEMBLED] f.pl vs dav ref: {ok}/{} ok; {wrong} wrong ({zero} are mine=0=GAP); first_gap={first_gap:?} first_wrong={first_wrong:?}", p.w * p.h);
                            if std::env::var("F1DUMP").is_ok() {
                                for (tag, pos) in [("GAP", first_gap.map(|(x,y)|(x,y))), ("WRONG", first_wrong.map(|(x,y,_,_)|(x,y)))] {
                                    if let Some((x0,y0)) = pos {
                                        crate::dlog!("F1DUMP {tag} around ({x0},{y0}):");
                                        for y in y0.saturating_sub(1)..(y0+4).min(p.h) {
                                            let md: Vec<i32> = (x0.saturating_sub(2)..(x0+10).min(p.w)).map(|x| p.px[y*p.stride+x].clamp(0,255) - rp.at(x,y)).collect();
                                            crate::dlog!("  y={y} x{}.. diff={md:?}", x0.saturating_sub(2));
                                        }
                                    }
                                }
                            }
                        }
                    });
                });
                crate::av2_frame::end_recon_pass(); // scored pass done; scaffold re-runs no-op
                // C1: verify the luma deblock in isolation — deblock dav2d's pre-filter luma
                // with our edge grids + derived thresholds, score vs dav2d's post-deblock luma.
                crate::av2_frame::run_deblock_verify(
                    &crate::av2_recon::cap_path("dav_lf_none.yuv"),
                    &crate::av2_recon::cap_path("dav_lf_deblk.yuv"),
                );
                // C2: verify CDEF in isolation — input = dav post-deblock, oracle = dav post-CDEF.
                crate::av2_frame::run_cdef_verify(
                    &crate::av2_recon::cap_path("dav_lf_deblk.yuv"),
                    &crate::av2_recon::cap_path("dav_lf_cdef.yuv"),
                );
                // C3: verify CCSO — classify from post-deblock luma, add to post-CDEF, vs post-CCSO.
                crate::av2_frame::run_ccso_verify(
                    &crate::av2_recon::cap_path("dav_lf_deblk.yuv"),
                    &crate::av2_recon::cap_path("dav_lf_cdef.yuv"),
                    &crate::av2_recon::cap_path("dav_lf_ccso.yuv"),
                );
                // C4: verify GDF (luma-only) — input post-CCSO, oracle post-GDF (== all, LR off);
                // stripe reference lines come from post-deblock (saved before CDEF).
                crate::av2_frame::run_gdf_verify(
                    &crate::av2_recon::cap_path("dav_lf_ccso.yuv"),
                    &crate::av2_recon::cap_path("dav_lf_all.yuv"),
                    &crate::av2_recon::cap_path("dav_lf_deblk.yuv"),
                );
                }
                crate::av2_frame::REF_SCORE.with(|s| {
                    let (ok, tot) = s.get();
                    if tot > 0 {
                        crate::dlog!("[REFSCORE] per-block isolated: {ok}/{tot} luma blocks bit-exact vs dav2d ({:.1}%)", 100.0 * ok as f64 / tot as f64);
                    }
                });
                crate::av2_frame::REF_CHROMA_SCORE.with(|s| {
                    let (u, v, t) = s.get();
                    if t > 0 {
                        crate::dlog!(
                            "[CREFSCORE] chroma per-block isolated: U {u}/{t} ({:.1}%), V {v}/{t} ({:.1}%)",
                            100.0 * u as f64 / t as f64, 100.0 * v as f64 / t as f64
                        );
                    }
                });
                // PRE-FILTER recon oracle: compare mine's f.pl (before any loop filter) to dav's
                // keyframe pre-filter recon (f->cur.p.data, what intrabc/intra prediction reads).
                // Isolates a RECON ±1 from a FILTER ±1 and cuts through the intrabc copy-chain.
                if let Some(b) = std::fs::read(&crate::av2_recon::cap_path("dav_f1prefilter.yuv")).ok().filter(|_| std::env::var("DAVCAP").is_ok()) {
                    crate::av2_frame::FRAME.with(|fr| {
                        let f = fr.borrow();
                        for (pl, (pw, ph, off)) in [(432usize, 240usize, 0usize), (216, 120, 432 * 240), (216, 120, 432 * 240 + 216 * 120)].into_iter().enumerate() {
                            if f.pl[pl].w == 0 { continue; }
                            let (mut ok, mut tot, mut first) = (0usize, 0usize, None);
                            for y in 0..ph.min(f.pl[pl].h) {
                                for x in 0..pw.min(f.pl[pl].w) {
                                    tot += 1;
                                    let m = f.pl[pl].px[y * f.pl[pl].stride + x].clamp(0, 255);
                                    let d = b[off + y * pw + x] as i32;
                                    if m == d { ok += 1; } else if first.is_none() { first = Some((x, y, m, d)); }
                                }
                            }
                            crate::dlog!("[F1PRE] frame-1 PRE-FILTER recon pl={pl} vs dav pre-filter: {ok}/{tot} ({:.2}%) first-miss={first:?}", 100.0 * ok as f64 / tot as f64);
                        }
                    });
                }
                // ===== Frame-1 END-TO-END filter chain: chain deblock→CDEF→CCSO→GDF on the
                // assembled (byte-exact) frame-1 FRAME buffer, compare vs dav2d's frame-1 decoded
                // (post-filter) output = the FIRST frame in dav_out.yuv. =====
                // env OH1PRE: dump the KEY's pre-filter recon (file oh 99, matching dav's PF1).
                if std::env::var("OH1PRE").is_ok() {
                    crate::av2_frame::FRAME.with(|fr| {
                        let f = fr.borrow();
                        let mut out = Vec::new();
                        for pl in 0..3 {
                            let p = &f.pl[pl];
                            for y in 0..p.h {
                                for x in 0..p.w {
                                    out.push(p.at(x, y) as u8);
                                }
                            }
                        }
                        let _ = std::fs::write(crate::av2_recon::cap_path("mine_oh99_prefilter.yuv"), &out);
                    });
                }
                crate::av2_frame::filter_frame_chain(0); // frame-1 keyframe → intra GDF tables
                if let Some(b) = std::fs::read(&crate::av2_recon::cap_path("dav_out.yuv")).ok().filter(|_| std::env::var("DAVCAP").is_ok()) {
                    let fsz = 432 * 240 + 2 * 216 * 120;
                    if b.len() >= fsz {
                        let f1 = &b[0..fsz];
                        crate::av2_frame::FRAME.with(|fr| {
                            let f = fr.borrow();
                            for (pl, (pw, ph, off)) in [(432usize, 240usize, 0usize), (216, 120, 432 * 240), (216, 120, 432 * 240 + 216 * 120)].into_iter().enumerate() {
                                if f.pl[pl].w == 0 { continue; }
                                let (mut ok, mut tot, mut first) = (0usize, 0usize, None);
                                for y in 0..ph.min(f.pl[pl].h) {
                                    for x in 0..pw.min(f.pl[pl].w) {
                                        tot += 1;
                                        let m = f.pl[pl].px[y * f.pl[pl].stride + x];
                                        let d = f1[off + y * pw + x] as i32;
                                        if m == d { ok += 1; } else if first.is_none() { first = Some((x, y, m, d)); }
                                    }
                                }
                                crate::dlog!("[F1FILT] frame-1 FILTERED pl={pl} vs dav2d final output: {ok}/{tot} px ({:.2}%) first-miss={first:?}", 100.0 * ok as f64 / tot as f64);
                            }
                        });
                    }
                }
                // Stash mine's own filtered frame 1 as the inter frame's reference (standalone
                // decode — no external file), then hand it to the output pipeline.
                crate::av2_frame::stash_decoded_frame1();
                crate::av2_recon::FRAME_DECODE_COUNT.with(|c| c.set(c.get() + 1));
                crate::av2_recon::update_ref_slots(true); // keyframe
                crate::av2_grain::stash_grain_slots(crate::av2_recon::CUR_FRAME_REF.with(|cc| cc.get()).1);
                emit_av2_output(c, state)?;
                }
                // Frame fully decoded, filtered, and emitted — done with this frame OBU.
                // (The block below is dead R1 hand-decode-trace scaffolding kept for reference.)
                return Ok(());
            }
            // Fall-through: a frame_type this decoder does not handle yet (2 = INTRA_ONLY
            // never minted by avmenc so far). Fail loudly instead of running stale scaffold.
            crate::dlog!("[rav2d AV2] UNHANDLED frame_type {frame_type} — no decode path");
            return Err(Rav1dError::InvalidArgument);
        }
        Some(Rav1dObuType::Metadata) => {
            let debug = Debug::new(false, "OBU", &gb);

            // obu metadata type field
            let meta_type = gb.get_uleb128();
            if gb.has_error() != 0 {
                return Err(Rav1dError::InvalidArgument);
            }

            match ObuMetaType::from_repr(meta_type as usize) {
                Some(ObuMetaType::HdrCll) => {
                    let debug = debug.named("CLLOBU");
                    let max_content_light_level = gb.get_bits(16) as u16;
                    debug.log(
                        &gb,
                        format_args!("max-content-light-level: {max_content_light_level}"),
                    );
                    let max_frame_average_light_level = gb.get_bits(16) as u16;
                    debug.log(
                        &gb,
                        format_args!(
                            "max-frame-average-light-level: {max_frame_average_light_level}"
                        ),
                    );

                    check_trailing_bits(gb, c.strict_std_compliance)?;

                    state.content_light = Some(Arc::new(Rav1dContentLightLevel {
                        max_content_light_level,
                        max_frame_average_light_level,
                    })); // TODO(kkysen) fallible allocation
                }
                Some(ObuMetaType::HdrMdcv) => {
                    let debug = debug.named("MDCVOBU");
                    let primaries = array::from_fn(|i| {
                        let primary = [gb.get_bits(16) as u16, gb.get_bits(16) as u16];
                        debug.log(&gb, format_args!("primaries[{i}]: {primary:?}"));
                        primary
                    });
                    let white_point_x = gb.get_bits(16) as u16;
                    debug.log(&gb, format_args!("white-point-x: {white_point_x}"));
                    let white_point_y = gb.get_bits(16) as u16;
                    debug.log(&gb, format_args!("white-point-y: {white_point_y}"));
                    let white_point = [white_point_x, white_point_y];
                    let max_luminance = gb.get_bits(32);
                    debug.log(&gb, format_args!("max-luminance: {max_luminance}"));
                    let min_luminance = gb.get_bits(32);
                    debug.log(&gb, format_args!("min-luminance: {min_luminance}"));
                    check_trailing_bits(gb, c.strict_std_compliance)?;

                    state.mastering_display = Some(Arc::new(Rav1dMasteringDisplay {
                        primaries,
                        white_point,
                        max_luminance,
                        min_luminance,
                    })); // TODO(kkysen) fallible allocation
                }
                Some(ObuMetaType::ItutT35) => {
                    let mut payload_size = gb.remaining_len() as isize;
                    // Don't take into account all the trailing bits for `payload_size`.
                    while payload_size > 0 && gb[payload_size as usize - 1] == 0 {
                        payload_size -= 1; // trailing_zero_bit x 8
                    }
                    payload_size -= 1; // trailing_one_bit + trailing_zero_bit x 7

                    let mut country_code_extension_byte = 0;
                    let country_code = gb.get_bits(8) as c_int;
                    payload_size -= 1;
                    if country_code == 0xff {
                        country_code_extension_byte = gb.get_bits(8) as c_int;
                        payload_size -= 1;
                    }

                    if payload_size <= 0 || gb[payload_size as usize] != 0x80 {
                        writeln!(c.logger, "Malformed ITU-T T.35 metadata message format");
                    } else {
                        let country_code = country_code as u8;
                        let country_code_extension_byte = country_code_extension_byte as u8;
                        let payload = gb.get_bytes(payload_size as usize).into(); // TODO fallible allocation
                        let itut_t35 = Rav1dITUTT35 {
                            country_code,
                            country_code_extension_byte,
                            payload,
                        };
                        state.itut_t35.push(itut_t35); // TODO fallible allocation
                    }
                }
                Some(ObuMetaType::Scalability | ObuMetaType::Timecode) => {} // Ignore metadata OBUs we don't care about.
                None => {
                    // Print a warning, but don't fail for unknown types.
                    writeln!(c.logger, "Unknown Metadata OBU type {meta_type}");
                }
            }
        }
        Some(Rav1dObuType::Td) => state.frame_flags |= PictureFlags::NEW_TEMPORAL_UNIT,
        Some(Rav1dObuType::Qm) => {
            // avm read_qm_obu (obu_qm.c:160): 15-bit qm_id bitmap + chroma-present flag,
            // then per-set data. A PREDEFINED set (qm_is_predefined_flag=1) matches the
            // decoder-default qm_list — nothing to store. USER-DEFINED matrices are
            // unported: fail loudly rather than silently mis-weighting the dequant.
            let qm_bit_map = gb.get_bits(15);
            let _chroma_present = gb.get_bit();
            if qm_bit_map != 0 {
                for j in 0..15 {
                    if qm_bit_map & (1 << j) != 0 {
                        let predefined = gb.get_bit();
                        if !predefined {
                            crate::dlog!("[rav2d AV2] QM OBU with USER-DEFINED matrices (qm_id {j}) unsupported");
                            return Err(Rav1dError::InvalidArgument);
                        }
                    }
                }
            }
        }
        Some(Rav1dObuType::Fgm) => {
            // FGM OBU (dav2d parse_fgm_hdr, obu.c:2094): up to 8 film-grain parameter tables,
            // mask-selected. Raw uv fields are stored UNOFFSET (avm convention; the synthesis
            // applies -128/-256 itself); ar_coeffs are stored signed (-128 applied).
            let mask = gb.get_bits(8) as u32;
            let _layout = gb.get_vlc(); // must match seq layout; unchecked here
            for idx in 0..8usize {
                if mask & (1 << idx) == 0 {
                    continue;
                }
                let mut fgd = crate::av2_grain::FilmGrainData::default();
                let seq_hdr = state.seq_hdr.as_ref();
                let monochrome = false;
                let _ = seq_hdr;
                let mut num_pl = 1usize;
                if !monochrome {
                    fgd.chroma_scaling_from_luma = gb.get_bit();
                    if !fgd.chroma_scaling_from_luma {
                        num_pl = 3;
                    }
                }
                for pl in 0..num_pl {
                    fgd.num_points[pl] = gb.get_bits(4) as usize;
                    if fgd.num_points[pl] > 14 {
                        return Err(Rav1dError::InvalidArgument);
                    }
                    if fgd.num_points[pl] == 0 {
                        continue;
                    }
                    let index_bits = 1 + gb.get_bits(3);
                    let scaling_bits = 5 + gb.get_bits(2);
                    let mut base = 0i32;
                    let mut prev_x = -1i32;
                    for i in 0..fgd.num_points[pl] {
                        base += gb.get_bits(index_bits as c_int) as i32;
                        if base > 255 {
                            return Err(Rav1dError::InvalidArgument);
                        }
                        // SEMANTIC VALIDATION (not a bounds check): the scaling points must be
                        // STRICTLY INCREASING in x. The spec guarantees it; a corrupt stream can
                        // violate it, and the consumer then divides by a zero delta_x
                        // (av2_grain init_scaling_function). Reject the field at the parse — the
                        // class of bug that bounds discipline structurally cannot catch.
                        if base <= prev_x {
                            return Err(Rav1dError::InvalidArgument);
                        }
                        prev_x = base;
                        fgd.points[pl][i] = (base, gb.get_bits(scaling_bits as c_int) as i32);
                    }
                }
                fgd.scaling_shift = gb.get_bits(2) as i32 + 8;
                fgd.ar_coeff_lag = gb.get_bits(2) as i32;
                let num_pos = (2 * fgd.ar_coeff_lag * (fgd.ar_coeff_lag + 1)) as usize;
                for pl in 0..3usize {
                    if fgd.num_points[pl] == 0 && (pl == 0 || !fgd.chroma_scaling_from_luma) {
                        continue;
                    }
                    let num_pl_pos = num_pos + (pl != 0 && fgd.num_points[0] != 0) as usize;
                    let coef_bits = 5 + gb.get_bits(2);
                    // avm obu_fgm.c: recenter by the HALF-RANGE midpoint 1<<(bits-1) — NOT a
                    // flat -128 (dav2d's flat -128 is its known grain divergence from avm).
                    let mid = 1i32 << (coef_bits - 1);
                    for i in 0..num_pl_pos {
                        fgd.ar_coeffs[pl][i] = gb.get_bits(coef_bits as c_int) as i32 - mid;
                    }
                }
                fgd.ar_coeff_shift = gb.get_bits(2) as i32 + 6;
                fgd.grain_scale_shift = gb.get_bits(2) as i32;
                for pl in 0..2usize {
                    if fgd.num_points[1 + pl] == 0 {
                        continue;
                    }
                    fgd.uv_mult[pl] = gb.get_bits(8) as i32;
                    fgd.uv_luma_mult[pl] = gb.get_bits(8) as i32;
                    fgd.uv_offset[pl] = gb.get_bits(9) as i32;
                }
                fgd.overlap_flag = gb.get_bit();
                fgd.clip_to_restricted_range = gb.get_bit();
                if fgd.clip_to_restricted_range {
                    fgd.mc_identity = gb.get_bit();
                }
                fgd.block_size = gb.get_bit() as i32;
                if std::env::var("FGMDBG").is_ok() {
                    crate::dlog!("[FGM] id={idx} csfl={} np={:?} pts0={:?} pts1={:?} pts2={:?} ssh={} lag={} arsh={} gss={} uvm={:?} uvlm={:?} uvo={:?} ovl={} clip={} mcid={} bs={}",
                        fgd.chroma_scaling_from_luma as u8, fgd.num_points,
                        &fgd.points[0][..fgd.num_points[0]], &fgd.points[1][..fgd.num_points[1]],
                        &fgd.points[2][..fgd.num_points[2]],
                        fgd.scaling_shift, fgd.ar_coeff_lag, fgd.ar_coeff_shift, fgd.grain_scale_shift,
                        fgd.uv_mult, fgd.uv_luma_mult, fgd.uv_offset,
                        fgd.overlap_flag as u8, fgd.clip_to_restricted_range as u8,
                        fgd.mc_identity as u8, fgd.block_size);
                }
                crate::av2_grain::FGM_TABLE.with(|t| t.borrow_mut()[idx] = Some(fgd));
            }
        }
        Some(_) => {} // Ignore other AV2 OBU types (multi-frame-hdr, metadata-grp, op-pt-set, …).
        None => {
            // Print a warning, but don't fail for unknown types.
            let len = gb.remaining_len();
            writeln!(c.logger, "Unknown OBU type {raw_type} of size {len}");
        }
    }

    if let (Some(_), Some(frame_hdr)) = (state.seq_hdr.as_ref(), state.frame_hdr.as_ref()) {
        let frame_hdr = &***frame_hdr;
        if frame_hdr.show_existing_frame != 0 {
            match state.refs[frame_hdr.existing_frame_idx as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .ok_or(Rav1dError::InvalidArgument)?
                .frame_type
            {
                Rav1dFrameType::Inter | Rav1dFrameType::Switch
                    if c.decode_frame_type > Rav1dDecodeFrameType::Reference =>
                {
                    return Ok(skip(state));
                }
                Rav1dFrameType::Intra if c.decode_frame_type > Rav1dDecodeFrameType::Intra => {
                    return Ok(skip(state));
                }
                _ => {}
            }
            if state.refs[frame_hdr.existing_frame_idx as usize]
                .p
                .p
                .data
                .is_none()
            {
                return Err(Rav1dError::InvalidArgument);
            }
            if c.strict_std_compliance
                && !state.refs[frame_hdr.existing_frame_idx as usize].p.showable
            {
                return Err(Rav1dError::InvalidArgument);
            }
            if c.fc.len() == 1 {
                state.out = state.refs[frame_hdr.existing_frame_idx as usize].p.clone();
                rav1d_picture_copy_props(
                    &mut state.out.p,
                    state.content_light.clone(),
                    state.mastering_display.clone(),
                    // Must be moved from the context to the frame.
                    Rav1dITUTT35::to_immut(mem::take(&mut state.itut_t35)),
                    props.clone(),
                );
                state.event_flags |= state.refs[frame_hdr.existing_frame_idx as usize]
                    .p
                    .flags
                    .into();
            } else {
                let mut task_thread_lock = c.task_thread.lock.lock();
                // Need to append this to the frame output queue.
                let next = state.frame_thread.next;
                state.frame_thread.next = (state.frame_thread.next + 1) % c.fc.len() as u32;

                let fc = &c.fc[next as usize];
                while !fc.task_thread.finished.load(Ordering::SeqCst) {
                    fc.task_thread.cond.wait(&mut task_thread_lock);
                }
                let out_delayed = &mut state.frame_thread.out_delayed[next as usize];
                if out_delayed.p.data.is_some() || fc.task_thread.error.load(Ordering::SeqCst) != 0
                {
                    let first = c.task_thread.first.load(Ordering::SeqCst);
                    if first as usize + 1 < c.fc.len() {
                        c.task_thread.first.fetch_add(1, Ordering::SeqCst);
                    } else {
                        c.task_thread.first.store(0, Ordering::SeqCst);
                    }
                    let _ = c.task_thread.reset_task_cur.compare_exchange(
                        first,
                        u32::MAX,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    );
                    if c.task_thread.cur.get() != 0
                        && (c.task_thread.cur.get() as usize) < c.fc.len()
                    {
                        c.task_thread.cur.update(|cur| cur - 1);
                    }
                }
                let error = &mut *fc.task_thread.retval.try_lock().unwrap();
                if error.is_some() {
                    state.cached_error = mem::take(error);
                    state.cached_error_props = out_delayed.p.m.clone();
                    let _ = mem::take(out_delayed);
                } else if out_delayed.p.data.is_some() {
                    let progress =
                        out_delayed.progress.as_ref().unwrap()[1].load(Ordering::Relaxed);
                    if (out_delayed.visible || c.output_invisible_frames) && progress != FRAME_ERROR
                    {
                        state.out = out_delayed.clone();
                        state.event_flags |= out_delayed.flags.into();
                    }
                    let _ = mem::take(out_delayed);
                }
                *out_delayed = state.refs[frame_hdr.existing_frame_idx as usize].p.clone();
                out_delayed.visible = true;
                rav1d_picture_copy_props(
                    &mut out_delayed.p,
                    state.content_light.clone(),
                    state.mastering_display.clone(),
                    // Must be moved from the context to the frame.
                    Rav1dITUTT35::to_immut(mem::take(&mut state.itut_t35)),
                    props.clone(),
                );
            }
            if state.refs[frame_hdr.existing_frame_idx as usize]
                .p
                .p
                .frame_hdr
                .as_ref()
                .unwrap()
                .frame_type
                == Rav1dFrameType::Key
            {
                let r = frame_hdr.existing_frame_idx;
                state.refs[r as usize].p.showable = false;
                for i in 0..8 {
                    if i == r {
                        continue;
                    }

                    if state.refs[i as usize].p.p.frame_hdr.is_some() {
                        let _ = mem::take(&mut state.refs[i as usize].p);
                    }
                    state.refs[i as usize].p = state.refs[r as usize].p.clone();

                    state.cdf[i as usize] = state.cdf[r as usize].clone();

                    state.refs[i as usize].segmap = state.refs[r as usize].segmap.clone();
                    let _ = mem::take(&mut state.refs[i as usize].refmvs);
                }
            }
            state.frame_hdr = None;
        } else if state.n_tiles == frame_hdr.tiling.cols as c_int * frame_hdr.tiling.rows as c_int {
            match frame_hdr.frame_type {
                Rav1dFrameType::Inter | Rav1dFrameType::Switch
                    if c.decode_frame_type > Rav1dDecodeFrameType::Reference
                        || c.decode_frame_type == Rav1dDecodeFrameType::Reference
                            && frame_hdr.refresh_frame_flags == 0 =>
                {
                    return Ok(skip(state));
                }
                Rav1dFrameType::Intra
                    if c.decode_frame_type > Rav1dDecodeFrameType::Intra
                        || c.decode_frame_type == Rav1dDecodeFrameType::Reference
                            && frame_hdr.refresh_frame_flags == 0 =>
                {
                    return Ok(skip(state));
                }
                _ => {}
            }
            if state.tiles.is_empty() {
                return Err(Rav1dError::InvalidArgument);
            }
            rav1d_submit_frame(c, state)?;
            assert!(state.tiles.is_empty());
            state.frame_hdr = None;
            state.n_tiles = 0;
        }
    }

    Ok(())
}

pub(crate) fn rav1d_parse_obus(
    c: &Rav1dContext,
    state: &mut Rav1dState,
    r#in: &CArc<[u8]>,
    props: &Rav1dDataProps,
) -> Rav1dResult<usize> {
    let gb = &mut GetBits::new(r#in);

    parse_obus(c, state, r#in, props, gb)
        .inspect_err(|_| {
            state.cached_error_props = props.clone();
            writeln!(
                c.logger,
                "{}",
                if gb.has_error() != 0 {
                    "Overrun in OBU bit buffer"
                } else {
                    "Error parsing OBU data"
                }
            );
        })
        .map(|_| gb.len())
}
