//! AV2 coefficient decode (dav2d `decode_coefs`) — the stateful bridge from the
//! entropy decoder to the reconstruction spine. Reads coefficient tokens from the
//! live MSAC using the generated coef CDF context.
//!
//! This is the ENTRY of `decode_coefs` (the end-of-block position). The full
//! base/BR level decode + per-coefficient context derivation lands toward M1,
//! where the whole path is verified at pixels against `avmdec` — it is stateful
//! and not isolation-verifiable, so the compile-check (correct MSAC + CDF wiring)
//! is the verification at this stage.

// Temporary verification flag: set by the frame-2 inter descent around its (single)
// decode_luma_tx_level call so the shared coef-level probes fire ONLY for that block, not for
// the frame-1 verification blocks that share dct_y. Remove with the rest of the F2 scaffold.
thread_local! {
    pub static COEF_DBG: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

use crate::cdf_av2::CdfCoefContext;
use crate::msac::{
    rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bools, rav1d_msac_decode_bools_bypass,
    rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8,
    rav1d_msac_decode_unary_bypass, MsacContext,
};

/// Exp-golomb residual (dav2d `decode_exp_golomb`): order-`k`, length-prefixed via a
/// 21-bit-capped unary prefix.
fn decode_exp_golomb(msac: &mut MsacContext, k: u32) -> u32 {
    let length = rav1d_msac_decode_unary_bypass(msac, 21) + k;
    let x = (1u32 << length) + rav1d_msac_decode_bools_bypass(msac, length as u8);
    x - (1 << k)
}

/// High-range coefficient residual (dav2d `decode_hr`): adaptive golomb appended when a
/// base/BR token saturates (tok >= 6). `hr_avg` is the running average of prior residuals
/// in this block (clamped to [2,64] for the golomb order `m = ulog2(..) ∈ 1..6`).
fn decode_hr(msac: &mut MsacContext, hr_avg: i32) -> i32 {
    let m = 31 - (hr_avg.clamp(2, 64) as u32).leading_zeros(); // ulog2 → 1..6
    let cmax = (m + 4).min(6); // 5 or 6
    let q = rav1d_msac_decode_unary_bypass(msac, cmax);
    let rem = if q == cmax {
        decode_exp_golomb(msac, m + 1)
    } else {
        rav1d_msac_decode_bools_bypass(msac, m as u8)
    };
    (rem + (q << m)) as i32
}

/// IDTX base-level context (dav2d `get_lo_ctx_idtx`): magnitude of the left + above
/// neighbours in the bordered level array. Returns `(lo_ctx, hi_ctx)` — `lo` indexes
/// the base-token CDF, `hi` the base-range CDF (capped at 6).
#[inline]
fn idtx_lo_ctx(levels: &[i8], idx: usize, stride: usize) -> (usize, usize) {
    let mut lo_mag = 0u32;
    let mut hi_mag = 0u32;
    for off in [idx - 1, idx - stride] {
        if !crate::av2_recon::work_tick("coef:55") { break; }
        // C `unsigned val = (int8_t)v;` — sign-extend then reinterpret unsigned.
        let val = levels[off] as i32 as u32;
        lo_mag += val.min(3);
        hi_mag += val.min(5);
    }
    (lo_mag as usize, hi_mag.min(6) as usize)
}

/// IDTX sign context (dav2d `get_sign_ctx_idtx`): from the signed sum of the left,
/// above and above-left neighbours plus an offset when the current level > 3.
#[inline]
fn idtx_sign_ctx(levels: &[i8], idx: usize, stride: usize) -> usize {
    let sum = levels[idx - 1] as i32 + levels[idx - stride] as i32 + levels[idx - stride - 1] as i32;
    let offset = if levels[idx] > 3 { 2 } else { 0 };
    match sum {
        -3 => offset + 6,
        -2 | -1 => offset + 2,
        0 => 0,
        1 | 2 => offset + 1,
        3 => offset + 5,
        _ => unreachable!("idtx sign sum out of range: {sum}"),
    }
}

/// DCT (TX_CLASS_2D) luma base-level context (dav2d `get_lo_ctx`, 2D luma path).
/// Reads the right/below/below-right neighbour magnitudes (these feed both lo+hi
/// accumulators) plus right-2/below-2 (lo only), then maps to `(lo_ctx, hi_ctx)` via
/// the frequency-dependent offset/limit. `xy = x + y` of the current coefficient.
/// Returns `lo_ctx` (indexes the base-token CDF) and `hi_ctx` (the base-range CDF).
#[inline]
pub fn get_lo_ctx_2d_luma(levels: &[i8], idx: usize, stride: usize, xy: usize) -> (usize, usize) {
    let lo_freq = xy < 4; // luma 2D: lo-frequency region is x+y < 4
    let mut lim: i32 = if lo_freq { 5 } else { 3 };
    let mut lo_mag = 0i32;
    let mut hi_mag = 0i32;
    // right, below, below-right: contribute to both lo (capped at `lim`) and hi (at 5).
    for off in [idx + 1, idx + stride, idx + stride + 1] {
        if !crate::av2_recon::work_tick("coef:92") { break; }
        let v = levels[off] as i32;
        lo_mag += v.min(lim);
        hi_mag += v.min(5);
    }
    // right-2 and below-2: lo only.
    lo_mag += (levels[idx + 2] as i32).min(lim) + (levels[idx + 2 * stride] as i32).min(lim);
    let offset = if lo_freq {
        lim = if xy == 0 { 8 } else if xy < 2 { 6 } else { 4 };
        if xy == 0 { 0 } else if xy < 2 { 9 } else { 16 }
    } else {
        lim = 4;
        if xy < 6 { 0 } else if xy < 8 { 5 } else { 10 }
    };
    // hi offset: luma, lo-freq, and (xy>0 since this is the 2D path) → +7.
    let hi_off = if lo_freq && xy > 0 { 7 } else { 0 };
    let lo_ctx = offset + ((lo_mag + 1) >> 1).min(lim);
    let hi_ctx = hi_off + ((hi_mag + 1) >> 1).min(6);
    (lo_ctx as usize, hi_ctx as usize)
}

/// Decode an FSC-IDTX luma coefficient block (dav2d `recon_tmpl.c` IDTX path):
/// `bob` token, the base-token loop (from `bob+1` up), then the sign loop. Fills
/// `cf` (raster order) with signed token magnitudes. Stateful — verified bit-exact
/// at the rng oracle, symbol by symbol. `slw`/`slh` are the TX log2 dims (0 for 4x4),
/// `sz_ctx = min(t_dim.ctx, 2)`, `scan` the IDTX scan for this TX.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs_idtx_y(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cf: &mut [i32],
    eob: i32,
    tx2dszctx: usize,
    sz_ctx: usize,
    slw: usize,
    slh: usize,
    scan: &[u16],
) -> u8 {
    let stride = 1 + (4 << slh);
    let mut levels = vec![0i8; stride * ((4 << slw) + 1)];
    let sz = (16i32 << tx2dszctx) - 1;
    let bob = sz - eob;
    let shift = slh + 2;
    let mask = (4usize << slh) - 1;

    // base-of-block token: ctx from how far `bob` sits into the block.
    let bctx = (bob > (2 << tx2dszctx)) as usize + (bob > (4 << tx2dszctx)) as usize;
    let mut tok = 1 + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.bob_base_y_tok[sz_ctx][bctx], 2) as i32;
    if tok == 3 {
        tok += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_idtx[sz_ctx][0], 3) as i32;
    }
    // HARDENING: clamp a corrupt bob/position into the scan + coef/level buffers.
    let bob = bob.max(0);
    let rc = scan[(bob as usize).min(scan.len() - 1)] as usize;
    let rc = rc.min(cf.len() - 1);
    cf[rc] = tok;
    let li = ((1 + (rc >> shift)) * stride + (1 + (rc & mask))).min(levels.len() - 1);
    levels[li] = tok as i8;

    // base-token loop, from bob+1 to the end of the scan.
    for i in (bob.max(0) + 1)..=sz.min(scan.len() as i32 - 1) {
        if !crate::av2_recon::work_tick("coef:152") { break; }
        let rc = (scan[(i.max(0) as usize).min(scan.len() - 1)] as usize).min(cf.len() - 1);
        let lidx = ((1 + (rc >> shift)) * stride + (1 + (rc & mask))).min(levels.len() - 1);
        let (lo, hr) = idtx_lo_ctx(&levels, lidx, stride);
        let mut tok = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.base_y_tok_idtx[sz_ctx][lo], 3) as i32;
        if tok == 3 {
            tok += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_idtx[sz_ctx][hr], 3) as i32;
        }
        cf[rc] = tok;
        levels[lidx] = tok as i8;
    }

    // sign loop over the coded coefficients (bob..sz): sign, then (for saturated tokens)
    // the high-range residual. `hr_avg` is the running residual average for the block.
    let mut dc_sign_level: u8 = 0x40; // base: DC (scan[0]) coded only when bob==0
    let mut hr_avg = 0i32;
    for i in bob..=sz {
        if !crate::av2_recon::work_tick("coef:168") { break; }
        let rc = scan[i as usize] as usize;
        let mut tok = cf[rc];
        if tok == 0 {
            continue;
        }
        let lidx = (1 + (rc >> shift)) * stride + (1 + (rc & mask));
        let sctx = idtx_sign_ctx(&levels, lidx, stride);
        let sign = rav1d_msac_decode_bool_adapt(msac, &mut cdf.sign_idtx[sz_ctx][sctx]);
        // dav2d `*level = 1 - 2*sign`: the level becomes a ±1 sign indicator (NOT the
        // magnitude), which bounds the neighbour sum read by get_sign_ctx_idtx to ±3.
        levels[lidx] = 1 - 2 * sign as i8;
        if i == 0 {
            dc_sign_level = if sign { 0x00 } else { 0x80 };
        }
        // High-range residual when the base/BR token saturated (tok >= 6).
        if tok >= 6 {
            let hr = decode_hr(msac, hr_avg);
            tok += hr;
            hr_avg = (hr_avg + hr) >> 1;
            tok &= 0xfffff;
        }
        cf[rc] = if sign { -tok } else { tok };
    }
    // cf_ctx = min(cumulative level, 63) | dc_sign_level (DC at scan[0] only coded when
    // bob==0, i.e. a full block; otherwise the base 0x40 stands).
    let cul_level: u32 = (bob..=sz).map(|i| cf[scan[i as usize] as usize].unsigned_abs()).sum();
    (cul_level.min(63) as u8) | dc_sign_level
}

/// `tx2dszctx` = `min(lw, TX_32X32) + min(lh, TX_32X32)`, the 2-D-size class
/// indexing the EOB-bin CDFs.
pub fn tx2dsz_ctx(lw: usize, lh: usize) -> usize {
    lw.min(3) + lh.min(3)
}

/// OR-fold the first `1<<lsize` bytes of a neighbour context array (dav2d
/// `MERGE_CTX` — the wide-read + shift-fold is just an OR over the TX's columns).
fn fold_ctx(arr: &[u8], lsize: usize) -> u32 {
    // HARDENING (shared context reader): a corrupt TX size can exceed the neighbour slice —
    // fold what exists (absent cells are cleared, matching the frame/tile-edge convention).
    let n = (1usize << lsize).min(arr.len());
    arr[..n].iter().fold(0u32, |acc, &b| acc | b as u32)
}

/// All-zero (txb_skip) context for luma (dav2d `get_skip_ctx`, `plane == 0`). When
/// the TX covers the whole block the context is 0; otherwise it combines the
/// above/left neighbour coefficient levels (capped at 4). `b_dim` is the block's
/// `[w4, h4, wl2, hl2]`; `a`/`l` are the per-column context arrays.
pub fn skip_ctx_luma(a: &[u8], l: &[u8], lw: usize, lh: usize, b_dim: &[u8]) -> u32 {
    if b_dim[2] as usize == lw && b_dim[3] as usize == lh {
        return 0;
    }
    let la = fold_ctx(a, lw) & 0x3F;
    let ll = fold_ctx(l, lh) & 0x3F;
    (la.min(4) + ll.min(4) + 3) >> 1
}

/// Decode the end-of-block position (dav2d `decode_coefs`, EOB section). Selects
/// the size-specific `eob_bin` CDF, decodes the coarse bin (+ bypass extension for
/// large sizes), then refines with `eob_hi_bit` + bypass bits. `eob_ctx` is
/// `chroma ? 2 : !intra`. Returns the exact EOB (number of coded coefficients).
pub fn decode_eob(msac: &mut MsacContext, cdf: &mut CdfCoefContext, tx2dszctx: usize, eob_ctx: usize) -> i32 {
    // coarse bin: size-specific CDF + symbol count; large sizes add bypass bits
    let mut eob = match tx2dszctx {
        0 => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_16[eob_ctx], 4),
        1 => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_32[eob_ctx], 5),
        2 => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_64[eob_ctx], 6),
        3 => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_128[eob_ctx], 7),
        4 => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_256[eob_ctx], 7),
        5 => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_512[eob_ctx], 7),
        _ => rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_bin_1024[eob_ctx], 7),
    } as i32;
    // extra bypass bits for the largest bins (eb = 1 for 256, 2 for 512/1024)
    let eb = match tx2dszctx {
        4 => 1,
        5 | 6 => 2,
        _ => 0,
    };
    if eb != 0 && eob == 7 {
        eob += rav1d_msac_decode_bools(msac, eb) as i32;
        // (dav2d returns INT_MIN on the invalid 512/eob==10 case — error, omitted)
    }

    if COEF_DBG.with(|c| c.get()) { crate::dlog!("EOBDBG coarse_bin={eob} tx2dszctx={tx2dszctx} eob_ctx={eob_ctx} rng={} dif={:x}", msac.rng, msac.dif); }
    // refine to the exact position via the high bit + remaining bypass bits
    if eob > 1 {
        let eob_hi_bit = rav1d_msac_decode_bool_adapt(msac, &mut cdf.eob_hi_bit) as i32;
        if COEF_DBG.with(|c| c.get()) { crate::dlog!("EOBDBG hi_bit={eob_hi_bit} rng={} dif={:x}", msac.rng, msac.dif); }
        let eob_bin = eob - 2;
        eob = eob_hi_bit | 2;
        if eob_bin != 0 {
            eob = (eob << eob_bin) | rav1d_msac_decode_bools(msac, eob_bin as u8) as i32;
        }
    }
    eob
}

/// DC-sign context (dav2d `get_dc_sign_ctx`): sums the high 2 bits (`>>6`) of the
/// neighbour cumulative-level bytes over the TX width (above `a`) + height (left `l`).
/// Each byte's bits 6-7 encode the neighbour's DC sign — `0x80`→2 (positive), `0x40`→1
/// (no DC), `0x00`→0 (negative). `tw`/`th` are the TX dims in 4-px units.
pub fn get_dc_sign_ctx(a: &[u8], l: &[u8], lw: usize, lh: usize, _tw: i32, _th: i32) -> usize {
    let mut t = 0i32;
    for &b in &a[..1 << lw] {
        t += (b as i32 & 0xC0) >> 6;
    }
    for &b in &l[..1 << lh] {
        t += (b as i32 & 0xC0) >> 6;
    }
    // dav2d recon_tmpl.c:194 subtracts the TRANSFORM dims (t_dim->w/h = 1<<lw / 1<<lh),
    // NOT the frame-clamped block dims — at frame edges bw4 < 1<<lw would skew the ctx.
    let s = t - (1i32 << lw) - (1i32 << lh);
    (s != 0) as usize + (s > 0) as usize
}

/// Decode a DC-only luma coefficient block (dav2d `recon_tmpl.c`, `eob==0` branch):
/// a single DC token from `eob_base_y_tok_lf[ctx][0]` (+ base-range if it saturates),
/// then the DC sign. Returns the `cf_ctx` splat byte.
pub fn decode_coefs_dc_only_y(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cf: &mut [i32],
    t_dim_ctx: usize,
    dc_sign_ctx: usize,
) -> u8 {
    let mut dc_tok = 1 + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_base_y_tok_lf[t_dim_ctx][0], 4) as i32;
    if dc_tok == 5 {
        // tx_class is 2D for the luma DC-only path → br index 0.
        dc_tok += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[0], 3) as i32;
    }
    let neg = rav1d_msac_decode_bool_adapt(msac, &mut cdf.dc_sign[0][0][dc_sign_ctx]);
    let dc_sign_level: u8 = if neg { 0x00 } else { 0x80 };
    // High-range golomb residual when the token saturates (max_br = 8 for the 2D luma DC);
    // hr_avg starts at 0 since a DC-only block has no prior (AC) residuals. Bypass-only — so
    // its omission diverges `dif` while leaving `rng` matched.
    if dc_tok >= 8 {
        let hr = decode_hr(msac, 0);
        dc_tok = (dc_tok + hr) & 0xfffff;
    }
    cf[0] = if neg { -dc_tok } else { dc_tok };
    (dc_tok.min(63) as u8) | dc_sign_level
}

/// AV2 TCQ state transition (dav2d `tcq_next_state`). When TCQ is disabled the state
/// is 0 and this returns 0 (the `& (state>>31)` mask), so `tcq` is always 0.
#[inline]
fn tcq_next_state(state: i32, abs_level: i32) -> i32 {
    (((state & 0x4) ^ (((abs_level & 1) ^ (state & 0x1)) << 2))
        | ((state & 0x6) >> 1)
        | i32::MIN)
        & (state >> 31)
}

/// Decode a DCT (TX_CLASS_2D) luma coefficient block (dav2d `recon_tmpl.c`
/// `DECODE_COEFS_CLASS`, 2D luma, lf path): eob_tok → AC base-token loop (driven by
/// the verified `get_lo_ctx_2d_luma`) → dc_tok → signs (AC bypass, DC `dc_sign` CDF).
/// Fills `cf` (raster) with signed token magnitudes. The lf path covers `eob < 10`
/// (the common small-eob blocks); the hf path + tok≥6 residual/dequant land as richer
/// blocks exercise them. Verified bit-exact at the oracle.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs_dct_y(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cf: &mut [i32],
    eob: i32,
    tx2dszctx: usize,
    t_dim_ctx: usize,
    slw: usize,
    slh: usize,
    scan: &[u16],
    tcq_enabled: bool,
    dc_sign_ctx: usize,
) -> u8 {
    use crate::msac::{rav1d_msac_decode_bool_bypass, rav1d_msac_decode_symbol_adapt8};
    // HARDENING: a corrupt stream's eob extra bits can exceed the scan — clamp (valid
    // streams are unaffected; the desynced frame yields garbage instead of a panic).
    let eob = eob.min(scan.len() as i32 - 1);
    let stride = 4usize << slh;
    let mut levels = vec![0i8; stride * ((4 << slw) + 2)];
    let shift = slh + 2;
    let mask = (4usize << slh) - 1;
    let hi_to_low_tx = 10i32; // 2D luma
    // `lim` selects the frequency regime: 3 = hf (adapt4 base, br_hf), 5 = lf (adapt8
    // base, br_lf). Starts hf when eob >= hi_to_low_tx, transitions to lf at i==9.
    let mut lim = if eob >= hi_to_low_tx { 3 } else { 5 };
    let eob_ctx = 1 + (eob > (2 << tx2dszctx)) as usize + (eob > (4 << tx2dszctx)) as usize;

    // eob token (the highest coded coefficient). The lf path is n_symbols=4 → range
    // [0,4], so it MUST use adapt8 (adapt4 clamps `% 4`, mapping a base of 4 → 0).
    let mut tok = if lim == 5 {
        1 + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_base_y_tok_lf[t_dim_ctx][eob_ctx], 4) as i32
    } else {
        1 + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.eob_base_y_tok_hf[t_dim_ctx][eob_ctx], 2) as i32
    };
    if tok == lim {
        tok += if lim == 5 {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[7], 3) as i32
        } else {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_hf[0], 3) as i32
        };
    }
    // TEMP guard (oh=2 pri_sec-average desync): clamp the eob into the scan so a desynced frame
    // produces garbage instead of panicking — lets earlier byte-exact frames still emit.
    let rc0 = scan[(eob as usize).min(scan.len() - 1)] as usize;
    cf[rc0] = tok;
    levels[rc0] = tok.min(127) as i8;
    let mut tcq_state: i32 = if tcq_enabled { i32::MIN } else { 0 };
    tcq_state = tcq_next_state(tcq_state, tok);

    // AC base-token loop, eob-1 down to 1. (TEMP: clamp the loop to the scan length so a desynced
    // frame — oh=2's unimplemented pri_sec CDF average — degrades to garbage instead of panicking.)
    for i in (1..eob.min(scan.len() as i32)).rev() {
        if !crate::av2_recon::work_tick("coef:380") { break; }
        if i == hi_to_low_tx - 1 {
            lim = 5; // hf → lf transition
        }
        let rc = scan[i as usize] as usize;
        let xy = (rc >> shift) + (rc & mask);
        let (lo, hr) = get_lo_ctx_2d_luma(&levels, rc, stride, xy);
        let tcq = ((tcq_state & 2) >> 1) as usize;
        let mut t = if lim == 5 {
            rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_y_tok_lf[t_dim_ctx][lo][tcq], 5) as i32
        } else {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.base_y_tok_hf[t_dim_ctx][lo][tcq], 3) as i32
        };
        if t == lim {
            t += if lim == 5 {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[hr], 3) as i32
            } else {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_hf[hr], 3) as i32
            };
        }
        cf[rc] = t;
        levels[rc] = t.min(127) as i8;
        tcq_state = tcq_next_state(tcq_state, t);
    }

    // DC token (position 0) — only when eob>0; always lf (the loop has transitioned
    // before reaching 0). For eob==0 the eob_tok already IS the DC.
    if eob > 0 {
        let (lo, hr) = get_lo_ctx_2d_luma(&levels, 0, stride, 0);
        let tcq = ((tcq_state & 2) >> 1) as usize;
        let mut t = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_y_tok_lf[t_dim_ctx][lo][tcq], 5) as i32;
        if t == 5 {
            t += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[hr], 3) as i32;
        }
        cf[0] = t;
        levels[0] = t.min(127) as i8;
    }

    // signs + high-range residual: AC (i=eob..1) sign via bypass, then the golomb
    // residual when the token saturates (tok >= max_br; max_br = 8 for i<hi_to_low_tx,
    // else 6 — luma). DC (i=0) sign via the dc_sign CDF, then the same residual.
    // `hr_avg` is the running residual average shared across the AC + DC residuals.
    let mut hr_avg = 0i32;
    for i in (1..=eob).rev() {
        if !crate::av2_recon::work_tick("coef:423") { break; }
        let rc = scan[i as usize] as usize;
        let mut tok = cf[rc];
        if tok == 0 {
            continue;
        }
        let sign = rav1d_msac_decode_bool_bypass(msac);
        let max_br = if i < hi_to_low_tx { 8 } else { 6 };
        if tok >= max_br {
            let hr = decode_hr(msac, hr_avg);
            tok += hr;
            hr_avg = (hr_avg + hr) >> 1;
            tok &= 0xfffff;
        }
        cf[rc] = if sign { -tok } else { tok };
    }
    let mut dc_sign_level: u8 = 0x40; // base: no DC sign coded
    if COEF_DBG.with(|c| c.get()) {
        crate::dlog!(
            "DCSIGN cf0={} eob={eob} ctx={dc_sign_ctx} pre rng={} dif={:x} cnt={}",
            cf[0], msac.rng, msac.dif, msac.cnt
        );
    }
    if cf[0] != 0 {
        let mut tok = cf[0];
        let neg = rav1d_msac_decode_bool_adapt(msac, &mut cdf.dc_sign[0][0][dc_sign_ctx]);
        dc_sign_level = if neg { 0x00 } else { 0x80 };
        if tok >= 8 {
            let hr = decode_hr(msac, hr_avg);
            tok += hr;
            hr_avg = (hr_avg + hr) >> 1;
            tok &= 0xfffff;
        }
        cf[0] = if neg { -tok } else { tok };
    }
    // cf_ctx = min(cumulative level, 63) | dc_sign_level — splatted for neighbour context.
    let cul_level: u32 = (0..=eob).map(|i| cf[scan[i as usize] as usize].unsigned_abs()).sum();
    (cul_level.min(63) as u8) | dc_sign_level
}

/// H/V transform-class luma base-level context (dav2d `get_lo_ctx`, H/V luma path).
/// Reads (x,y+1)+(x+1,y) (lo capped at `lim0`, hi at 5), (x,y+2) (lo 3, hi 5) and
/// (x,y+3)/(x,y+4) (lo 3 only) along the 1D-transform axis. `xy` is the cross-axis coord
/// (`y`); `idx` = `x*stride + y` (the transposed level offset).
#[inline]
pub fn get_lo_ctx_hv_luma(levels: &[i8], idx: usize, stride: usize, xy: usize) -> (usize, usize) {
    let lo_freq = xy < 2;
    let lim0 = if lo_freq { 5 } else { 3 };
    let n1 = levels[idx + 1] as i32; // (x, y+1)
    let n2 = levels[idx + stride] as i32; // (x+1, y)
    let mut lo_mag = n1.min(lim0) + n2.min(lim0);
    let mut hi_mag = n1.min(5) + n2.min(5);
    let n3 = levels[idx + 2] as i32; // (x, y+2): lo capped at 3, hi at 5
    lo_mag += n3.min(3);
    hi_mag += n3.min(5);
    lo_mag += (levels[idx + 3] as i32).min(3) + (levels[idx + 4] as i32).min(3); // (x,y+3),(x,y+4)
    let (offset, lim_final) = if lo_freq {
        if xy == 0 { (21i32, 6i32) } else { (28, 4) }
    } else {
        (15, 4)
    };
    let hi_off = if lo_freq { 7 } else { 0 }; // H/V (tx_class != 2D) → lo_freq always gets +7
    let lo_ctx = offset + ((lo_mag + 1) >> 1).min(lim_final);
    let hi_ctx = hi_off + ((hi_mag + 1) >> 1).min(6);
    (lo_ctx as usize, hi_ctx as usize)
}

/// Decode an H/V transform-class luma coefficient block (dav2d `DECODE_COEFS_CLASS`,
/// TX_CLASS_H=2 / TX_CLASS_V=3). The identity-based 1D transforms use a transposed level
/// layout (`levels[x*32 + y]`) and an *implicit* (no-scan) coefficient order: position
/// `i` maps directly to `(x = i & mask, y = i >> shift)`; the raster `rc` is `i` for H,
/// `(x << shift2) | y` for V. Signs in the DC row (`y == 0`) use the `dc_sign` CDF (ctx 0
/// for AC coeffs, the real `dc_sign_ctx` for the true DC). Mirrors the verified 2D path.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs_hv_y(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cf: &mut [i32],
    eob: i32,
    tx_class: usize, // 2 = H, 3 = V
    tx2dszctx: usize,
    t_dim_ctx: usize,
    slw: usize,
    slh: usize,
    tcq_enabled: bool,
    dc_sign_ctx: usize,
) -> u8 {
    use crate::msac::{
        rav1d_msac_decode_bool_adapt, rav1d_msac_decode_bool_bypass, rav1d_msac_decode_symbol_adapt8,
    };
    let stride = 32usize;
    let is_v = tx_class == 3;
    // H keys off slh (the transform is horizontal-identity), V off slw.
    let (shift, shift2, mask, axis) = if is_v {
        (slw + 2, slh + 2, (4usize << slw) - 1, 4usize << slw)
    } else {
        (slh + 2, 0usize, (4usize << slh) - 1, 4usize << slh)
    };
    let hi_to_low_tx = 8i32 << if is_v { slw } else { slh };
    let mut levels = vec![0i8; stride * (axis + 2)];

    // position i → (x, y, rc, level_idx).
    let pos = |i: i32| -> (usize, usize, usize, usize) {
        let x = (i as usize) & mask;
        let y = (i as usize) >> shift;
        let rc = if is_v { (x << shift2) | y } else { i as usize };
        (x, y, rc, x * stride + y)
    };

    let mut lim = if eob >= hi_to_low_tx { 3 } else { 5 };
    // eob==0 (DC-only) uses eob_ctx 0 (dav2d, same as decode_coefs_dc_only_y); the band ctx
    // otherwise. Missing this only bites an H/V-class DC-only TX — first hit at frame-2 (50,8).
    let eob_ctx = if eob == 0 {
        0
    } else {
        1 + (eob > (2 << tx2dszctx)) as usize + (eob > (4 << tx2dszctx)) as usize
    };

    // eob coefficient (highest position) — eob_tok br is fixed [7] (lf) / [0] (hf).
    let mut tok = if lim == 5 {
        1 + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_base_y_tok_lf[t_dim_ctx][eob_ctx], 4) as i32
    } else {
        1 + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.eob_base_y_tok_hf[t_dim_ctx][eob_ctx], 2) as i32
    };
    if tok == lim {
        tok += if lim == 5 {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[7], 3) as i32
        } else {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_hf[0], 3) as i32
        };
    }
    let (_, _, rc0, lidx0) = pos(eob);
    // HARDENING: clamp the corrupt-stream position into the coef/level buffers.
    let (rc0, lidx0) = (rc0.min(cf.len() - 1), lidx0.min(levels.len() - 1));
    cf[rc0] = tok;
    levels[lidx0] = tok.min(127) as i8;
    let mut tcq_state: i32 = if tcq_enabled { i32::MIN } else { 0 };
    tcq_state = tcq_next_state(tcq_state, tok);

    // AC base-token loop, eob-1 down to 1 — driven by get_lo_ctx_hv_luma (xy = y).
    for i in (1..eob).rev() {
        if !crate::av2_recon::work_tick("coef:563") { break; }
        if i == hi_to_low_tx - 1 {
            lim = 5; // hf → lf transition
        }
        let (_x, y, rc, lidx) = pos(i);
        let (lo, hr) = get_lo_ctx_hv_luma(&levels, lidx, stride, y);
        let tcq = ((tcq_state & 2) >> 1) as usize;
        let mut t = if lim == 5 {
            rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_y_tok_lf[t_dim_ctx][lo][tcq], 5) as i32
        } else {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.base_y_tok_hf[t_dim_ctx][lo][tcq], 3) as i32
        };
        if t == lim {
            t += if lim == 5 {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[hr], 3) as i32
            } else {
                rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_hf[hr], 3) as i32
            };
        }
        // HARDENING: clamp corrupt-stream positions into the coef/level buffers.
        let (rc, lidx) = (rc.min(cf.len() - 1), lidx.min(levels.len() - 1));
        cf[rc] = t;
        levels[lidx] = t.min(127) as i8;
        tcq_state = tcq_next_state(tcq_state, t);
    }

    // DC token (position 0) — always lf (loop has transitioned).
    if eob > 0 {
        let (lo, hr) = get_lo_ctx_hv_luma(&levels, 0, stride, 0);
        let tcq = ((tcq_state & 2) >> 1) as usize;
        let mut t = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_y_tok_lf[t_dim_ctx][lo][tcq], 5) as i32;
        if t == 5 {
            t += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_y_tok_lf[hr], 3) as i32;
        }
        cf[0] = t;
        levels[0] = t.min(127) as i8;
    }

    // signs + high-range residual: AC (i=eob..1) — DC row (y==0) via dc_sign[0], else
    // bypass; then the golomb residual when tok >= max_br (8 for i<hi_to_low_tx, else 6).
    let mut hr_avg = 0i32;
    for i in (1..=eob).rev() {
        if !crate::av2_recon::work_tick("coef:604") { break; }
        let (_x, y, rc, _lidx) = pos(i);
        // HARDENING: clamp a corrupt-stream position into the coef buffer.
        let rc = rc.min(cf.len() - 1);
        let mut t = cf[rc];
        if t == 0 {
            continue;
        }
        let neg = if y == 0 {
            rav1d_msac_decode_bool_adapt(msac, &mut cdf.dc_sign[0][0][0])
        } else {
            rav1d_msac_decode_bool_bypass(msac)
        };
        let max_br = if i < hi_to_low_tx { 8 } else { 6 };
        if t >= max_br {
            let hr = decode_hr(msac, hr_avg);
            t += hr;
            hr_avg = (hr_avg + hr) >> 1;
            t &= 0xfffff;
        }
        cf[rc] = if neg { -t } else { t };
    }
    // DC (position 0): dc_sign via the real ctx; same residual.
    let mut dc_sign_level: u8 = 0x40;
    if cf[0] != 0 {
        let mut t = cf[0];
        let neg = rav1d_msac_decode_bool_adapt(msac, &mut cdf.dc_sign[0][0][dc_sign_ctx]);
        dc_sign_level = if neg { 0x00 } else { 0x80 };
        if t >= 8 {
            let hr = decode_hr(msac, hr_avg);
            t += hr;
            hr_avg = (hr_avg + hr) >> 1;
            t &= 0xfffff;
        }
        cf[0] = if neg { -t } else { t };
    }
    // HARDENING: clamp corrupt positions in the cul_level sum.
    let cul_level: u32 = (0..=eob).map(|i| { let (_, _, rc, _) = pos(i); cf[rc.min(cf.len() - 1)].unsigned_abs() }).sum();
    (cul_level.min(63) as u8) | dc_sign_level
}

/// Chroma skip (txb_skip) context (dav2d `get_skip_ctx`, `plane != 0`): a binary "any
/// non-cleared neighbour" over the TX, plus a per-plane offset. `ca`/`cl` test whether
/// any above/left chroma `ccoef` entry over the TX (`tx_w4`/`tx_h4` in chroma-4px units)
/// differs from the cleared base `0x40`. The offset is `6` for U; for V it folds in
/// whether *this block's* U plane carried coefficients (`u_has_cf`) and whether the chroma
/// block spans more than one TX (`not_one_blk`).
#[allow(clippy::too_many_arguments)]
pub fn skip_ctx_chroma(
    a: &[u8],
    l: &[u8],
    cbx4: usize,
    cby4: usize,
    tx_w4: usize,
    tx_h4: usize,
    plane: usize,
    u_has_cf: bool,
    not_one_blk: bool,
) -> usize {
    let ca = a[cbx4..cbx4 + tx_w4].iter().any(|&v| v != 0x40) as usize;
    let cl = l[cby4..cby4 + tx_h4].iter().any(|&v| v != 0x40) as usize;
    let offset = if plane == 1 { 6 } else { 6 * u_has_cf as usize + not_one_blk as usize * 3 };
    offset + ca + cl
}

/// 2D chroma base-level context (dav2d `get_lo_ctx`, chroma 2D path). Unlike luma it
/// reads only the right/below/below-right neighbours (no right-2/below-2), the lo-freq
/// region is `xy < 1`, the lo clamp is 3, the per-plane offset is `0` (U) / `4` (V), and
/// the hi context has no `+7` offset and caps at 3.
#[inline]
pub fn get_lo_ctx_2d_chroma(levels: &[i8], idx: usize, stride: usize, xy: usize, plane: usize) -> (usize, usize) {
    let lo_freq = xy < 1;
    let lim0 = if lo_freq { 5 } else { 3 };
    let n1 = levels[idx + 1] as i32; // right
    let n2 = levels[idx + stride] as i32; // below
    let n3 = levels[idx + stride + 1] as i32; // below-right
    let lo_mag = n1.min(lim0) + n2.min(lim0) + n3.min(lim0);
    let hi_mag = n1.min(5) + n2.min(5) + n3.min(5);
    let offset = if plane == 1 { 0 } else { 4 };
    let lo_ctx = offset + ((lo_mag + 1) >> 1).min(3);
    let hi_ctx = ((hi_mag + 1) >> 1).min(3);
    (lo_ctx as usize, hi_ctx as usize)
}

/// Decode one chroma plane's coefficients (dav2d `decode_coefs`, `plane != 0`): the
/// `all_zero` (txb_skip) flag, the eob (chroma `eob_ctx = 2`), the inferred `DCT_DCT`
/// txtp (no coded symbol), then the 2D coefficient block. Returns `(eob, cf_ctx)`.
/// `plane` is 1 (U) / 2 (V); `sctx` is the precomputed skip context.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs_uv(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cctx_cdf: &mut [u16],
    cf: &mut [i32],
    plane: usize,
    t_dim_ctx: usize,
    slw: usize,
    slh: usize,
    tx2dszctx: usize,
    scan: &[u16],
    sctx: usize,
    // U-plane all_zero skip set: 0 for intra blocks, 1 for inter blocks (dav2d skip[set]).
    // The V plane uses `skip_v` (no set index) regardless.
    u_skip_set: usize,
    // Block intra flag: gates the U-plane cctx symbol on `eob >= intra` (dav2d recon 754) —
    // intra blocks code cctx only with AC coefs (eob>=1), inter blocks even DC-only (eob>=0).
    intra: bool,
    // Chroma transform class (0 = 2D, 2 = H, 3 = V). Intra chroma is always 2D; inter chroma
    // INHERITS the luma txtp's class (may be H/V), which selects a different scan + coef path.
    tx_class: usize,
) -> (i32, u8) {
    // 64-dim TXs code only a 32-core: the coefficient grid, level contexts, and scan all
    // work in CORE dims (same clw/clh convention as the luma path). Clamping here also
    // leaves the `< 3` cctx size gate unchanged.
    let (slw, slh) = (slw.min(3), slh.min(3));
    // HARDENING: a desynced stream can produce out-of-range contexts — clamp into the CDF dims
    // (valid streams always land in range, so this is a no-op for them).
    let sctx_v = sctx.min(cdf.skip_v.len() - 1);
    let u_skip_set = u_skip_set.min(cdf.skip.len() - 1);
    let t_dim_ctx = t_dim_ctx.min(cdf.skip[u_skip_set].len() - 1);
    let sctx_u = sctx.min(cdf.skip[u_skip_set][t_dim_ctx].len() - 1);
    let all_zero = if plane == 2 {
        rav1d_msac_decode_bool_adapt(msac, &mut cdf.skip_v[sctx_v])
    } else {
        rav1d_msac_decode_bool_adapt(msac, &mut cdf.skip[u_skip_set][t_dim_ctx][sctx_u])
    };
    if COEF_DBG.with(|c| c.get()) { crate::dlog!("UVDBG pl={plane} all_zero={} sctx={sctx} tctx={t_dim_ctx} uset={u_skip_set} rng={} dif={:x}", all_zero as u8, msac.rng, msac.dif); }
    if all_zero {
        return (-1, 0x40);
    }
    let eob = decode_eob(msac, cdf, tx2dszctx, 2);
    if COEF_DBG.with(|c| c.get()) { crate::dlog!("UVDBG pl={plane} eob={eob} tx2dszctx={tx2dszctx} rng={} dif={:x}", msac.rng, msac.dif); }
    // chroma intra txtp is inferred (ADST_ADST/DCT_DCT — both tx_class 2D, so it doesn't
    // change the coefficient decode). The U plane carries a cross-component transform
    // context symbol (dav2d `cctx`) when `eob >= intra` under I420: intra needs an AC coef
    // (eob>=1), inter codes it even for a DC-only block (eob>=0).
    // is_cctx_allowed (avm pred_common.h:504): always at 4:2:0; at 4:2:2/4:4:4 only when the
    // chroma block is < 32px in at least one dimension (slw < 3 || slh < 3).
    let cctx_allowed = {
        let ss = crate::av2_frame::SS.with(|c| c.get());
        (ss.0 == 1 && ss.1 == 1) || slw < 3 || slh < 3
    };
    // seq enable_cctx gates the whole symbol (tool-off mint: crash without).
    if plane == 1 && eob >= intra as i32 && cctx_allowed && crate::av2_recon::SEQ_TOOLS.with(|c| c.get().cctx) {
        let cctx = rav1d_msac_decode_symbol_adapt8(msac, cctx_cdf, 6);
        if COEF_DBG.with(|c| c.get()) { crate::dlog!("UVDBG pl={plane} cctx={cctx} rng={} dif={:x}", msac.rng, msac.dif); }
    }
    // 2D (intra, or inter with a 2D luma txtp) uses the square scan; an inherited H/V luma
    // txtp (inter only) uses the transposed implicit-order path.
    let cf_ctx = if tx_class != 0 {
        decode_coefs_hv_uv(msac, cdf, cf, eob, tx_class, tx2dszctx, slw, slh)
    } else {
        decode_coefs_dct_uv(msac, cdf, cf, plane, eob, tx2dszctx, slw, slh, scan)
    };
    (eob, cf_ctx)
}

/// Decode a chroma 2D (DCT_DCT) coefficient block (dav2d `DECODE_COEFS_CLASS`, chroma).
/// Mirrors `decode_coefs_dct_y` but with the chroma specifics: `hi_to_low_tx = 1` (so the
/// only lf coefficient is the DC), the `*_uv_*` CDFs (the chroma DC/lf path has NO base-
/// range CDF), the chroma `get_lo_ctx`, all signs (incl. DC) decoded by **bypass**, and
/// `max_br = 5` for the DC / `6` for AC.
#[allow(clippy::too_many_arguments)]
fn decode_coefs_dct_uv(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cf: &mut [i32],
    plane: usize,
    eob: i32,
    tx2dszctx: usize,
    slw: usize,
    slh: usize,
    scan: &[u16],
) -> u8 {
    use crate::msac::{rav1d_msac_decode_bool_bypass, rav1d_msac_decode_symbol_adapt8};
    // HARDENING: clamp a corrupt eob into the scan (see decode_coefs_dct_y).
    let eob = eob.min(scan.len() as i32 - 1);
    let stride = 4usize << slh;
    let mut levels = vec![0i8; stride * ((4 << slw) + 2)];
    let shift = slh + 2;
    let mask = (4usize << slh) - 1;
    let hi_to_low_tx = 1i32; // chroma
    let lim = if eob >= hi_to_low_tx { 3 } else { 5 };
    // dc-only (eob==0) uses eob_ctx 0; otherwise the standard frequency-band context.
    let eob_ctx = if eob == 0 {
        0
    } else {
        1 + (eob > (2 << tx2dszctx)) as usize + (eob > (4 << tx2dszctx)) as usize
    };

    // eob coefficient. The chroma lf path (DC) has no base-range CDF; only the hf eob
    // coeff can extend via br_uv_tok_hf[0].
    let tok = if lim == 5 {
        1 + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_base_uv_tok_lf[eob_ctx], 4) as i32
    } else {
        let mut t = 1 + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.eob_base_uv_tok_hf[eob_ctx], 2) as i32;
        if t == 3 {
            t += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_uv_tok_hf[0], 3) as i32;
        }
        t
    };
    if COEF_DBG.with(|c| c.get()) { crate::dlog!("UVDBG pl={plane} eob_base tok={tok} lim={lim} rng={} dif={:x}", msac.rng, msac.dif); }
    // TEMP guard (oh=2 pri_sec-average desync): clamp the eob into the scan so a desynced frame
    // produces garbage instead of panicking — lets earlier byte-exact frames still emit.
    let rc0 = scan[(eob as usize).min(scan.len() - 1)] as usize;
    cf[rc0] = tok;
    levels[rc0] = tok.min(127) as i8;

    // AC base-token loop (eob-1..1): all hf for chroma (the lf transition is at i==0).
    for i in (1..eob).rev() {
        if !crate::av2_recon::work_tick("coef:813") { break; }
        let rc = scan[i as usize] as usize;
        let xy = (rc >> shift) + (rc & mask);
        let (lo, hr) = get_lo_ctx_2d_chroma(&levels, rc, stride, xy, plane);
        let mut t = rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.base_uv_tok_hf[lo], 3) as i32;
        if t == 3 {
            t += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_uv_tok_hf[hr], 3) as i32;
        }
        cf[rc] = t;
        levels[rc] = t.min(127) as i8;
        if COEF_DBG.with(|c| c.get()) && plane == 1 { crate::dlog!("CHRAC2m i={i} rc={rc} lo={lo} tok={t} rng={} dif={:x}", msac.rng, msac.dif); }
    }

    if COEF_DBG.with(|c| c.get()) { crate::dlog!("UVDBG pl={plane} post-AC rng={} dif={:x}", msac.rng, msac.dif); }
    // DC token (i=0): lf, no base-range CDF.
    if eob > 0 {
        let (lo, _hr) = get_lo_ctx_2d_chroma(&levels, 0, stride, 0, plane);
        let t = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_uv_tok_lf[lo], 5) as i32;
        cf[0] = t;
        levels[0] = t.min(127) as i8;
    }
    if COEF_DBG.with(|c| c.get()) { crate::dlog!("UVDBG pl={plane} post-DC cf0={} rng={} dif={:x}", cf[0], msac.rng, msac.dif); }

    // signs (all bypass for chroma) + high-range residual. max_br = 5 (DC) / 6 (AC).
    let mut hr_avg = 0i32;
    for i in (1..=eob).rev() {
        if !crate::av2_recon::work_tick("coef:838") { break; }
        let rc = scan[i as usize] as usize;
        let mut t = cf[rc];
        if t == 0 {
            continue;
        }
        let neg = rav1d_msac_decode_bool_bypass(msac);
        let max_br = if i < hi_to_low_tx { 5 } else { 6 };
        if t >= max_br {
            let hr = decode_hr(msac, hr_avg);
            t += hr;
            hr_avg = (hr_avg + hr) >> 1;
            t &= 0xfffff;
        }
        cf[rc] = if neg { -t } else { t };
    }
    // DC (i=0): bypass sign + residual (max_br = 5).
    let mut dc_sign_level: u8 = 0x40;
    if cf[0] != 0 {
        let mut t = cf[0];
        let neg = rav1d_msac_decode_bool_bypass(msac);
        dc_sign_level = if neg { 0x00 } else { 0x80 };
        if t >= 5 {
            let hr = decode_hr(msac, hr_avg);
            t += hr;
            hr_avg = (hr_avg + hr) >> 1;
            t &= 0xfffff;
        }
        cf[0] = if neg { -t } else { t };
    }
    let cul_level: u32 = (0..=eob).map(|i| cf[scan[i as usize] as usize].unsigned_abs()).sum();
    (cul_level.min(63) as u8) | dc_sign_level
}

/// H/V transform-class CHROMA base-level context (dav2d `get_lo_ctx`, chroma non-2D path).
/// Like the luma H/V helper it reads only the right/below neighbours (transposed layout),
/// but with the chroma specifics: lo-freq is `xy < 1`, a fixed offset of 8, lo clamp 3, and
/// a hi context that caps at 3 with no `+7` (chroma never gets the luma lo-freq bump).
#[inline]
pub fn get_lo_ctx_hv_chroma(levels: &[i8], idx: usize, stride: usize, xy: usize) -> (usize, usize) {
    let lo_freq = xy < 1;
    let lim0 = if lo_freq { 5 } else { 3 };
    let n1 = levels[idx + 1] as i32; // (x, y+1)
    let n2 = levels[idx + stride] as i32; // (x+1, y)
    let lo_mag = n1.min(lim0) + n2.min(lim0);
    let hi_mag = n1.min(5) + n2.min(5);
    let lo_ctx = 8 + ((lo_mag + 1) >> 1).min(3);
    let hi_ctx = ((hi_mag + 1) >> 1).min(3);
    (lo_ctx as usize, hi_ctx as usize)
}

/// Decode an H/V transform-class CHROMA coefficient block (dav2d `DECODE_COEFS_CLASS`,
/// chroma, TX_CLASS_H=2 / TX_CLASS_V=3). Combines the H/V structure of `decode_coefs_hv_y`
/// (transposed `levels`, implicit no-scan order, lf/hf transition at `hi_to_low_tx`) with the
/// chroma specifics of `decode_coefs_dct_uv`: uv CDFs, no tcq (`lo_cdf_idx == ctx`), NO
/// base-range CDF in the lf region (`hi_cdf` is NULL — only hf extends), `get_lo_ctx_hv_chroma`,
/// all-bypass signs, and `max_br` 5/6.
#[allow(clippy::too_many_arguments)]
fn decode_coefs_hv_uv(
    msac: &mut MsacContext,
    cdf: &mut CdfCoefContext,
    cf: &mut [i32],
    eob: i32,
    tx_class: usize, // 2 = H, 3 = V
    tx2dszctx: usize,
    slw: usize,
    slh: usize,
) -> u8 {
    use crate::msac::{
        rav1d_msac_decode_bool_bypass, rav1d_msac_decode_symbol_adapt4, rav1d_msac_decode_symbol_adapt8,
    };
    let stride = 32usize;
    let is_v = tx_class == 3;
    let (shift, shift2, mask, axis) = if is_v {
        (slw + 2, slh + 2, (4usize << slw) - 1, 4usize << slw)
    } else {
        (slh + 2, 0usize, (4usize << slh) - 1, 4usize << slh)
    };
    // dav2d recon_tmpl.c:1087/1095 — for CHROMA the lf/hf boundary is halved (`>> chroma`),
    // so a 4x4 chroma H/V TX transitions at i==3, not i==7 (the luma value).
    let hi_to_low_tx = (8i32 << if is_v { slw } else { slh }) >> 1;
    let mut levels = vec![0i8; stride * (axis + 2)];
    let pos = |i: i32| -> (usize, usize, usize, usize) {
        let x = (i as usize) & mask;
        let y = (i as usize) >> shift;
        let rc = if is_v { (x << shift2) | y } else { i as usize };
        (x, y, rc, x * stride + y)
    };

    let mut lim = if eob >= hi_to_low_tx { 3 } else { 5 };
    let eob_ctx = if eob == 0 {
        0
    } else {
        1 + (eob > (2 << tx2dszctx)) as usize + (eob > (4 << tx2dszctx)) as usize
    };

    // eob coefficient (highest position). lf has no base-range CDF (chroma); hf extends via
    // br_uv_tok_hf[0] when the base token saturates.
    let mut tok = if lim == 5 {
        1 + rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.eob_base_uv_tok_lf[eob_ctx], 4) as i32
    } else {
        let mut t = 1 + rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.eob_base_uv_tok_hf[eob_ctx], 2) as i32;
        if t == 3 {
            t += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_uv_tok_hf[0], 3) as i32;
        }
        t
    };
    let (_, _, rc0, lidx0) = pos(eob);
    // HARDENING: clamp the corrupt-stream position into the coef/level buffers.
    let (rc0, lidx0) = (rc0.min(cf.len() - 1), lidx0.min(levels.len() - 1));
    cf[rc0] = tok;
    levels[lidx0] = tok.min(127) as i8;

    // AC base-token loop (eob-1 down to 1), driven by get_lo_ctx_hv_chroma(xy = y).
    for i in (1..eob).rev() {
        if !crate::av2_recon::work_tick("coef:952") { break; }
        if i == hi_to_low_tx - 1 {
            lim = 5; // hf → lf transition (chroma: hi_cdf becomes NULL)
        }
        let (_x, y, rc, lidx) = pos(i);
        let (lo, hr) = get_lo_ctx_hv_chroma(&levels, lidx, stride, y);
        let mut t = if lim == 5 {
            rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_uv_tok_lf[lo], 5) as i32
        } else {
            rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.base_uv_tok_hf[lo], 3) as i32
        };
        if lim == 3 && t == 3 {
            t += rav1d_msac_decode_symbol_adapt4(msac, &mut cdf.br_uv_tok_hf[hr], 3) as i32;
        }
        cf[rc] = t;
        levels[lidx] = t.min(127) as i8;
    }

    // DC token (position 0) — always lf by this point (hi_to_low_tx >= 8 > 0), no base-range.
    if eob > 0 {
        let (lo, _hr) = get_lo_ctx_hv_chroma(&levels, 0, stride, 0);
        let t = rav1d_msac_decode_symbol_adapt8(msac, &mut cdf.base_uv_tok_lf[lo], 5) as i32;
        cf[0] = t;
        levels[0] = t.min(127) as i8;
    }

    // signs (all bypass for chroma) + high-range residual. max_br = 5 (i < hi_to_low_tx) / 6.
    let mut hr_avg = 0i32;
    for i in (1..=eob).rev() {
        if !crate::av2_recon::work_tick("coef:980") { break; }
        let (_x, _y, rc, _lidx) = pos(i);
        let mut t = cf[rc];
        if t == 0 {
            continue;
        }
        let neg = rav1d_msac_decode_bool_bypass(msac);
        let max_br = if i < hi_to_low_tx { 5 } else { 6 };
        if t >= max_br {
            let hr = decode_hr(msac, hr_avg);
            t += hr;
            hr_avg = (hr_avg + hr) >> 1;
            t &= 0xfffff;
        }
        cf[rc] = if neg { -t } else { t };
    }
    let mut dc_sign_level: u8 = 0x40;
    if cf[0] != 0 {
        let mut t = cf[0];
        let neg = rav1d_msac_decode_bool_bypass(msac);
        dc_sign_level = if neg { 0x00 } else { 0x80 };
        if t >= 5 {
            let hr = decode_hr(msac, hr_avg);
            t += hr;
            hr_avg = (hr_avg + hr) >> 1;
            t &= 0xfffff;
        }
        cf[0] = if neg { -t } else { t };
    }
    // HARDENING: clamp corrupt positions in the cul_level sum.
    let cul_level: u32 = (0..=eob).map(|i| { let (_, _, rc, _) = pos(i); cf[rc.min(cf.len() - 1)].unsigned_abs() }).sum();
    (cul_level.min(63) as u8) | dc_sign_level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_lo_ctx_matches_block1_dc() {
        // Block #1 (16x16 DCT): the eob coeff lands at scan[1]=(0,1) → levels[1]=1, all
        // else 0. The DC's get_lo_ctx (xy=0) must return ctx=1 — the value dav2d's oracle
        // printed for "dc_tok[ctx=2|1|0]". stride = 4<<slh = 16 for a 16x16 TX.
        let mut levels = vec![0i8; 16 * 18];
        levels[1] = 1; // eob coefficient magnitude
        let (lo, hi) = get_lo_ctx_2d_luma(&levels, 0, 16, 0);
        assert_eq!(lo, 1, "DC lo_ctx must match oracle ctx=1");
        assert_eq!(hi, 1);
        // A cleared neighbourhood → ctx 0.
        let z = vec![0i8; 16 * 18];
        assert_eq!(get_lo_ctx_2d_luma(&z, 0, 16, 0), (0, 0));
        // Two unit neighbours at the DC → lo_mag=2 → (2+1)>>1 = 1.
        let mut t = vec![0i8; 16 * 18];
        t[1] = 1;
        t[16] = 1; // right + below
        assert_eq!(get_lo_ctx_2d_luma(&t, 0, 16, 0).0, 1);
    }

    #[test]
    fn skip_ctx_luma_cases() {
        // TX covers the whole block → context 0.
        let z = [0u8; 16];
        let b_dim_8x8 = [2, 2, 1, 1]; // BS_8x8: w4=2,h4=2,wl2=1,hl2=1
        assert_eq!(skip_ctx_luma(&z, &z, 1, 1, &b_dim_8x8), 0); // tx 8x8 == block
        // TX smaller than block, cleared neighbours → (0+0+3)>>1 = 1.
        let b_dim_16x16 = [4, 4, 2, 2];
        assert_eq!(skip_ctx_luma(&z, &z, 1, 1, &b_dim_16x16), 1); // tx 8x8 in 16x16 block
        // neighbours carry coefficient levels → capped contribution.
        let mut a = [0u8; 16];
        a[0] = 0x05;
        a[1] = 0x03; // OR over 2 columns (tx 8x8, lw=1) = 0x07 → min(7,4)=4
        assert_eq!(skip_ctx_luma(&a, &z, 1, 1, &b_dim_16x16), (4 + 0 + 3) >> 1);
    }
}

// Runtime verification of `decode_eob` is M1-gated: it requires a live MSAC built
// on real tile data (the `CArc<[u8]>` construction goes through the decoder's
// CRef/Pin plumbing) and is checked bit-exact at pixels vs `avmdec`, since the
// decoded EOB only has meaning in the correct bitstream position. The compile-check
// (this module type-checks against the real `MsacContext` + generated
// `CdfCoefContext`) verifies the entropy↔CDF↔coeff wiring is correct here.
