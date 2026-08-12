#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_uint};
use std::ops::{Deref, DerefMut, Range};
use std::{mem, ptr, slice};

use cfg_if::cfg_if;

use crate::c_arc::CArc;
use crate::cpu::CpuFlags;
use crate::include::common::attributes::clz;
use crate::include::common::intops::{inv_recenter, ulog2};







pub struct Rav1dMsacDSPContext {
    symbol_adapt16: unsafe extern "C" fn(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint,
}

impl Rav1dMsacDSPContext {
    pub const fn default() -> Self {
        Self {
            symbol_adapt16: rav1d_msac_decode_symbol_adapt_c,
        }
    }





    #[inline(always)]
    const fn init(self, flags: CpuFlags) -> Self {


        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            self
        }
    }

    pub const fn new(flags: CpuFlags) -> Self {
        Self::default().init(flags)
    }
}

impl Default for Rav1dMsacDSPContext {
    fn default() -> Self {
        Self::default()
    }
}

pub type EcWin = usize;

/// # Safety
///
/// [`Self`] must be the first field of [`MsacAsmContext`] for asm layout purposes,
/// and that [`MsacAsmContext`] must be a field of [`MsacContext`].
/// And [`Self::pos`] and [`Self::end`] must be either [`ptr::null`],
/// or [`Self::pos`] must point into (or the end of) [`MsacContext::data`],
/// and [`Self::end`] must point to the end of [`MsacContext::data`],
/// where [`MsacContext::data`] is part of the [`MsacContext`]
/// containing [`MsacAsmContext`] and thus also [`Self`].
#[repr(C)]
struct MsacAsmContextBuf {
    pos: *const u8,
    end: *const u8,
}

/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Send`], [`MsacAsmContextBuf`] is also [`Send`].
unsafe impl Send for MsacAsmContextBuf {}

/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Sync`], [`MsacAsmContextBuf`] is also [`Sync`].
unsafe impl Sync for MsacAsmContextBuf {}

impl Default for MsacAsmContextBuf {
    fn default() -> Self {
        Self {
            pos: ptr::null(),
            end: ptr::null(),
        }
    }
}

impl From<&[u8]> for MsacAsmContextBuf {
    fn from(value: &[u8]) -> Self {
        let Range { start, end } = value.as_ptr_range();
        Self { pos: start, end }
    }
}

#[repr(C)]
pub struct MsacAsmContext {
    buf: MsacAsmContextBuf,
    pub dif: EcWin,
    pub rng: c_uint,
    pub cnt: c_int,
    allow_update_cdf: c_int,

}

impl Default for MsacAsmContext {
    fn default() -> Self {
        Self {
            buf: Default::default(),
            dif: Default::default(),
            rng: Default::default(),
            cnt: Default::default(),
            allow_update_cdf: Default::default(),


        }
    }
}

impl MsacAsmContext {
    fn allow_update_cdf(&self) -> bool {
        self.allow_update_cdf != 0
    }
}

#[derive(Default)]
pub struct MsacContext {
    asm: MsacAsmContext,
    data: Option<CArc<[u8]>>,
}

impl Deref for MsacContext {
    type Target = MsacAsmContext;

    fn deref(&self) -> &Self::Target {
        &self.asm
    }
}

impl DerefMut for MsacContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.asm
    }
}

impl MsacContext {
    pub fn data(&self) -> &[u8] {
        &**self.data.as_ref().unwrap()
    }

    pub fn buf_index(&self) -> usize {
        // We safely subtract instead of unsafely use `ptr::offset_from`
        // as asm sets `buf_pos`, so we don't need to rely on its safety,
        // and because codegen is no less optimal this way.
        self.buf.pos as usize - self.data().as_ptr() as usize
    }

    fn with_buf(&mut self, mut f: impl FnMut(&[u8]) -> &[u8]) {
        let data = &**self.data.as_ref().unwrap();
        let buf = &data[self.buf_index()..];
        let buf = f(buf);
        self.buf.pos = buf.as_ptr();
        // We don't actually need to set `self.buf_end` since it has not changed.
    }
}

/// Return value uses `n` bits.
#[inline]
pub fn rav1d_msac_decode_bools(s: &mut MsacContext, n: u8) -> c_uint {
    // AV2: a single n-bit bypass read (dav2d `decode_bools_bypass`), NOT n separate
    // equiprobable bools — the AV2 bypass decodes the bits together off one `dif`.
    rav1d_msac_decode_bools_bypass(s, n)
}

#[inline]
pub fn rav1d_msac_decode_uniform(s: &mut MsacContext, n: c_uint) -> c_int {
    assert!(n > 0);
    let l = ulog2(n) as u8 + 1;
    assert!(l > 1);
    let m = (1 << l) - n;
    let v = rav1d_msac_decode_bools(s, l - 1);
    (if v < m {
        v
    } else {
        (v << 1) - m + rav1d_msac_decode_bool_equi(s) as c_uint
    }) as c_int
}

const EC_PROB_SHIFT: c_uint = 6;
const EC_MIN_PROB: c_uint = 4;

/// AV2 CDF-adaptation rate (dav2d `dav2d_msac_rate[125][3]`), indexed
/// `[pc>>8][count>>4]` where `pc` is the CDF count slot. Replaces AV1's
/// `4 + (count>>4)` formula.
#[rustfmt::skip]
static AV2_MSAC_RATE: [[u8; 3]; 125] = [
    [4,5,6],[4,5,5],[4,5,4],[4,5,7],[4,5,7],[4,4,6],[4,4,5],[4,4,4],[4,4,7],[4,4,7],
    [4,3,6],[4,3,5],[4,3,4],[4,3,7],[4,3,7],[4,6,6],[4,6,5],[4,6,4],[4,6,7],[4,6,7],
    [4,6,6],[4,6,5],[4,6,4],[4,6,7],[4,6,7],[3,5,6],[3,5,5],[3,5,4],[3,5,7],[3,5,7],
    [3,4,6],[3,4,5],[3,4,4],[3,4,7],[3,4,7],[3,3,6],[3,3,5],[3,3,4],[3,3,7],[3,3,7],
    [3,6,6],[3,6,5],[3,6,4],[3,6,7],[3,6,7],[3,6,6],[3,6,5],[3,6,4],[3,6,7],[3,6,7],
    [2,5,6],[2,5,5],[2,5,4],[2,5,7],[2,5,7],[2,4,6],[2,4,5],[2,4,4],[2,4,7],[2,4,7],
    [2,3,6],[2,3,5],[2,3,4],[2,3,7],[2,3,7],[2,6,6],[2,6,5],[2,6,4],[2,6,7],[2,6,7],
    [2,6,6],[2,6,5],[2,6,4],[2,6,7],[2,6,7],[5,5,6],[5,5,5],[5,5,4],[5,5,7],[5,5,7],
    [5,4,6],[5,4,5],[5,4,4],[5,4,7],[5,4,7],[5,3,6],[5,3,5],[5,3,4],[5,3,7],[5,3,7],
    [5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],[5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
    [5,5,6],[5,5,5],[5,5,4],[5,5,7],[5,5,7],[5,4,6],[5,4,5],[5,4,4],[5,4,7],[5,4,7],
    [5,3,6],[5,3,5],[5,3,4],[5,3,7],[5,3,7],[5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
    [5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
];

/// AV2 symbol-decode minimum probabilities (dav2d `dav2d_msac_min_prob[7][8]`),
/// indexed `[n_symbols-1][val]`.
#[rustfmt::skip]
static AV2_MSAC_MIN_PROB: [[u16; 8]; 7] = [
    [   63, 65535, 65535, 65535, 65535, 65535, 65535, 65535],
    [   47,    87, 65535, 65535, 65535, 65535, 65535, 65535],
    [   31,    63,    95, 65535, 65535, 65535, 65535, 65535],
    [   31,    55,    79,   103, 65535, 65535, 65535, 65535],
    [   23,    47,    63,    87,   111, 65535, 65535, 65535],
    [   23,    39,    55,    79,    95,   111, 65535, 65535],
    [   15,    31,    47,    63,    79,    95,   111, 65535],
];
const _: () = assert!(EC_MIN_PROB <= (1 << EC_PROB_SHIFT) / 16);

const EC_WIN_SIZE: usize = mem::size_of::<EcWin>() << 3;

thread_local! {
    /// E3b tell probe: bytes pulled into the entropy window (avm bptr-buf twin).
    pub static MSAC_PULLED: std::cell::Cell<u32> = std::cell::Cell::new(0);
    /// E3b tell probe: sum of renormalization shifts (reader tell = 1 + this).
    pub static MSAC_SUM_D: std::cell::Cell<u32> = std::cell::Cell::new(0);
}

#[inline]
fn ctx_refill(s: &mut MsacContext) {
    let mut c = (EC_WIN_SIZE as c_int) - 24 - s.cnt;
    let mut dif = s.dif;
    s.with_buf(|mut buf| {
        loop {
            if buf.is_empty() {
                // dav2d convention: no 1s-fill — the unfilled low bits are already 1s
                // (from the all-1s init and the `((dif+1)<<d)-1` norm). This matters for
                // the AV2 bypass's `dif+1` carry.
                break;
            }
            // XOR the raw byte (not OR-inverted): 1 ^ buf == ~buf for the filled bits.
            dif ^= (buf[0] as EcWin) << c;
            MSAC_PULLED.with(|p| p.set(p.get() + 1));
            buf = &buf[1..];
            c -= 8;
            if c < 0 {
                break;
            }
        }
        buf
    });
    s.dif = dif;
    s.cnt = (EC_WIN_SIZE as c_int) - 24 - c;
}

#[inline]
fn ctx_norm(s: &mut MsacContext, dif: EcWin, rng: c_uint) {
    let d = 15 ^ (31 ^ clz(rng));
    MSAC_SUM_D.with(|p| p.set(p.get() + d as u32));
    let cnt = s.cnt;
    assert!(rng <= 65535);
    // dav2d convention: `((dif+1)<<d)-1` fills the new low d bits with 1s (vs rav1d's
    // plain `dif<<d` → 0s). The 1s are required for the AV2 bypass's `dif+1` carry.
    s.dif = (dif.wrapping_add(1) << d).wrapping_sub(1);
    s.rng = rng << d;
    s.cnt = cnt - d;
    // unsigned compare avoids redundant refills at eob
    if (cnt as u32) < (d as u32) {
        ctx_refill(s);
    }
}


fn rav1d_msac_decode_bool_rust(s: &mut MsacContext, f: c_uint) -> bool {
    let r = s.rng;
    let mut dif = s.dif;
    assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
    // AV2 probability quantization (dav2d msac.c decode_bool): differs from AV1's
    // `(r>>8)*(f>>6)>>1 + 4`. AV2 remaps f to p then scales.
    let p = ((f >> 7) << 4) + 8;
    let mut v = ((r >> 8) * p >> 7) << 3;
    let vw = (v as EcWin) << (EC_WIN_SIZE - 16);
    let ret = dif >= vw;
    dif -= (ret as EcWin) * vw;
    v = v.wrapping_add((ret as c_uint) * (r.wrapping_sub(2 * v)));
    ctx_norm(s, dif, v);
    !ret
}

/// AV2 multi-bit bypass (dav2d `dav2d_msac_decode_bools_bypass_c`). Fundamentally
/// different from AV1's equiprobable bool: it leaves `rng` UNCHANGED and consumes
/// `n_bits` directly from the 64-bit `dif` window (`vw = r << 47`, compare/subtract
/// per bit, then `dif = ((dif+1) << n_bits) - 1`). This is why the oracle's `rng`
/// is constant across a sign/bypass read.
pub fn rav1d_msac_decode_bools_bypass(s: &mut MsacContext, n_bits: u8) -> c_uint {
    debug_assert!(n_bits > 0 && n_bits <= 32);
    if (s.cnt as c_uint) < n_bits as c_uint {
        ctx_refill(s);
    }
    let r = s.rng as u64;
    let mut dif = s.dif as u64;
    let mut vw = r << 47;
    let mut ret = 0u32;
    for _ in 0..n_bits {
        ret <<= 1;
        if dif >= vw {
            dif -= vw;
        } else {
            ret |= 1;
        }
        vw >>= 1;
    }
    s.dif = (((dif + 1) << n_bits) - 1) as EcWin;
    s.cnt -= n_bits as c_int;
    ret
}

/// AV2 single-bit bypass (dav2d `decode_bool_bypass_c` = `decode_bools_bypass(s, 1)`).
#[inline]
pub fn rav1d_msac_decode_bool_bypass(s: &mut MsacContext) -> bool {
    rav1d_msac_decode_bools_bypass(s, 1) != 0
}

/// AV2 bypass unary decode (dav2d `decode_unary_bypass_c`): reads up to `max_bits`
/// bypass bits, returning the count of leading "continue" bits (the unary value).
/// Used by the high-range residual (`decode_hr`) and exp-golomb paths.
pub fn rav1d_msac_decode_unary_bypass(s: &mut MsacContext, max_bits: u32) -> u32 {
    if (s.cnt as i64) < max_bits as i64 {
        ctx_refill(s);
    }
    let r = s.rng as u64;
    let mut dif = s.dif as u64;
    let mut vw = r << 47;
    let mut ret = 0u32;
    let mut bit = 0u32;
    // Mirrors the dav2d for-loop: each `dif >= vw` consumes a bit and increments the
    // count; the first `dif < vw` consumes one terminating bit and stops.
    while bit < max_bits {
        if dif >= vw {
            dif -= vw;
            vw >>= 1;
            ret += 1;
            bit += 1;
        } else {
            bit += 1;
            break;
        }
    }
    s.dif = (((dif + 1) << bit) - 1) as EcWin;
    s.cnt -= bit as c_int;
    ret
}

pub fn rav1d_msac_decode_subexp(s: &mut MsacContext, r#ref: c_uint, n: c_uint, mut k: u8) -> c_int {
    assert!(n >> k == 8);
    let mut a = 0;
    if rav1d_msac_decode_bool_equi(s) {
        if rav1d_msac_decode_bool_equi(s) {
            k += rav1d_msac_decode_bool_equi(s) as u8 + 1;
        }
        a = 1 << k;
    }
    let v = rav1d_msac_decode_bools(s, k) + a;
    (if r#ref * 2 <= n {
        inv_recenter(r#ref, v)
    } else {
        n - 1 - inv_recenter(n - 1 - r#ref, v)
    }) as c_int
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..16`, so it is really a `u4`.
fn rav1d_msac_decode_symbol_adapt_rust(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as c_uint;
    let r = s.rng >> 8;
    let mut u;
    let mut v = s.rng;
    let mut val = 0;
    assert!(n_symbols < 16);
    // AV2 symbol probability (dav2d): p = (cdf|127) - min_prob[val], scaled —
    // replaces AV1's (cdf>>6) + EC_MIN_PROB*(n-val).
    let min_prob = &AV2_MSAC_MIN_PROB[(n_symbols - 1) as usize];
    loop {
        u = v;
        let p = ((cdf[val as usize] | 127) as c_uint).saturating_sub(min_prob[val as usize] as c_uint);
        v = (r * p >> 10) << 3;
        if !(c < v) {
            break;
        }
        val += 1;
    }
    assert!(u <= s.rng);
    ctx_norm(
        s,
        s.dif.wrapping_sub((v as EcWin) << (EC_WIN_SIZE - 16)),
        u - v,
    );
    if s.allow_update_cdf() {
        // AV2 table-driven rate (dav2d) + (n>2) bump; count slot holds rate-ctx<<8|count.
        let pc = cdf[n_symbols as usize];
        let count = (pc & 0xff) as usize;
        // HARDENING: clamp the rate-context lookup (corrupt CDF state).
        let rate = AV2_MSAC_RATE[((pc >> 8) as usize).min(AV2_MSAC_RATE.len() - 1)]
            [(count >> 4).min(AV2_MSAC_RATE[0].len() - 1)] as u16 + (n_symbols > 2) as u16;
        let val = val as usize;
        for cdf in &mut cdf[..val] {
            *cdf += ((1 << 15) - *cdf) >> rate;
        }
        for cdf in &mut cdf[val..n_symbols as usize] {
            *cdf -= *cdf >> rate;
        }
        cdf[n_symbols as usize] = pc + (count < 32) as u16;
    }
    debug_assert!(val <= n_symbols as _);
    val as u8
}

/// # Safety
///
/// Must be called through [`Rav1dMsacDSPContext::symbol_adapt16`]
/// in [`rav1d_msac_decode_symbol_adapt16`].
#[allow(dead_code)] // reachable only through the (now removed) asm dispatch
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn rav1d_msac_decode_symbol_adapt_c(
    s: &mut MsacAsmContext,
    cdf: *mut u16,
    n_symbols: usize,
    cdf_len: usize,
) -> c_uint {
    // SAFETY: In the `rav1d_msac_decode_symbol_adapt16` caller,
    // `&mut s.asm` is passed, so we can reverse this to get back `s`.
    // The `.sub` is safe since were are subtracting the offset of `asm` within `s`,
    // so that will stay in bounds of the `s: MsacContext` allocated object.
    let s = unsafe {
        &mut *ptr::from_mut(s)
            .sub(mem::offset_of!(MsacContext, asm))
            .cast::<MsacContext>()
    };

    // SAFETY: This is only called from [`dav1d_msac_decode_symbol_adapt16`],
    // where it comes from `cdf.len()`.
    let cdf = unsafe { slice::from_raw_parts_mut(cdf, cdf_len) };

    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols as u8) as c_uint
}

fn rav1d_msac_decode_bool_adapt_rust(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    // AV2: must use the Rust bool decode (AV2 formula); the asm path is AV1.
    let bit = rav1d_msac_decode_bool_rust(s, cdf[0] as c_uint);
    if s.allow_update_cdf() {
        // AV2 adaptation rate (dav2d): table-driven by the rate-context in the high
        // byte of the count slot — NOT AV1's `4 + (count>>4)`.
        let pc = cdf[1];
        let count = (pc & 0xff) as usize;
        // HARDENING: a corrupt-stream CDF cell can carry an out-of-range rate context.
        let rate = AV2_MSAC_RATE[((pc >> 8) as usize).min(AV2_MSAC_RATE.len() - 1)]
            [(count >> 4).min(AV2_MSAC_RATE[0].len() - 1)] as u16;
        if bit {
            cdf[0] += ((1 << 15) - cdf[0]) >> rate;
        } else {
            cdf[0] -= cdf[0] >> rate;
        }
        cdf[1] = pc + (count < 32) as u16;
    }
    bit
}

/// Return value is in the range `0..=15`.
fn rav1d_msac_decode_hi_tok_rust(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let mut tok_br = rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
    let mut tok = 3 + tok_br;
    if tok_br == 3 {
        tok_br = rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
        tok = 6 + tok_br;
        if tok_br == 3 {
            tok_br = rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
            tok = 9 + tok_br;
            if tok_br == 3 {
                tok = 12 + rav1d_msac_decode_symbol_adapt4(s, cdf, 3);
            }
        }
    }
    tok
}

impl MsacContext {
    pub fn new(data: CArc<[u8]>, disable_cdf_update_flag: bool, dsp: &Rav1dMsacDSPContext) -> Self {
        let asm = MsacAsmContext {
            buf: data.as_ref().into(),
            // dav2d init: all 1s except the top bit, so XOR-refill yields ~buf in the
            // filled bits and leaves 1s in the unfilled low bits.
            dif: (1 << (EC_WIN_SIZE - 1)) - 1,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: (!disable_cdf_update_flag).into(),

        };
        let mut s = Self {
            asm,
            data: Some(data),
        };
        let _ = dsp.symbol_adapt16; // Silence unused warnings.
        ctx_refill(&mut s);
        s
    }
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..4`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt4(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 4);
    // AV2 MSAC: Rust path only (the dav1d asm implements the AV1 formula).
    let ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
    debug_assert!(ret < 4);
    ret as u8 % 4
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..8`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt8(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 8);
    // AV2 MSAC: Rust path only (the dav1d asm implements the AV1 formula).
    let ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
    debug_assert!(ret < 8);
    ret as u8 % 8
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..16`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt16(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 16);
    // AV2 MSAC: Rust path only (the dav1d asm implements the AV1 formula).
    let ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
    debug_assert!(ret < 16);
    ret as u8 % 16
}

pub fn rav1d_msac_decode_bool_adapt(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    // AV2 reworked the MSAC probability + adaptation; the dav1d asm is AV1-only, so
    // always take the (AV2-corrected) Rust path. SIMD for the AV2 MSAC is a later opt.
    rav1d_msac_decode_bool_adapt_rust(s, cdf)
}

pub fn rav1d_msac_decode_bool_equi(s: &mut MsacContext) -> bool {
    // AV2 reworked bypass: a single bit consumed from `dif` with `rng` unchanged
    // (NOT AV1's range-halving equi). The AV1 asm/_rust path is wrong for AV2.
    rav1d_msac_decode_bool_bypass(s)
}

pub fn rav1d_msac_decode_bool(s: &mut MsacContext, f: c_uint) -> bool {
    rav1d_msac_decode_bool_rust(s, f)
}

/// Return value is in the range `0..16`.
#[inline(always)]
pub fn rav1d_msac_decode_hi_tok(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let ret = rav1d_msac_decode_hi_tok_rust(s, cdf);
    debug_assert!(ret < 16);
    ret % 16
}
