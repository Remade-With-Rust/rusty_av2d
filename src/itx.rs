use std::num::NonZeroUsize;
use std::{cmp, slice};

use strum::EnumCount;

use crate::cpu::CpuFlags;
use crate::enum_map::DefaultValue;
use crate::ffi_safe::FFISafe;


use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynCoef, DynPixel};
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
};
use crate::itx_1d::rav1d_inv_wht4_1d_c;
use crate::itx_1d::{
    rav1d_inv_adst16_1d_c, rav1d_inv_adst4_1d_c, rav1d_inv_adst8_1d_c, rav1d_inv_dct16_1d_c,
    rav1d_inv_dct32_1d_c, rav1d_inv_dct4_1d_c, rav1d_inv_dct64_1d_c, rav1d_inv_dct8_1d_c,
    rav1d_inv_flipadst16_1d_c, rav1d_inv_flipadst4_1d_c, rav1d_inv_flipadst8_1d_c,
    rav1d_inv_identity16_1d_c, rav1d_inv_identity32_1d_c, rav1d_inv_identity4_1d_c,
    rav1d_inv_identity8_1d_c,
};
use crate::levels::{
    TxfmSize, TxfmType, ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST,
    FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX,
    N_TX_TYPES_PLUS_LL, V_ADST, V_DCT, V_FLIPADST, WHT_WHT,
};
use crate::strided::Strided as _;
use crate::wrap_fn_ptr::wrap_fn_ptr;

pub type Itx1dFn = fn(c: &mut [i32], stride: NonZeroUsize, min: i32, max: i32);

#[inline(never)]
fn inv_txfm_add<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    w: usize,
    h: usize,
    shift: u8,
    first_1d_fn: Itx1dFn,
    second_1d_fn: Itx1dFn,
    has_dc_only: bool,
    bd: BD,
) {
    let bitdepth_max = bd.bitdepth_max().as_::<i32>();

    assert!(w >= 4 && w <= 64);
    assert!(h >= 4 && h <= 64);
    assert!(eob >= 0);

    let is_rect2 = w * 2 == h || h * 2 == w;
    let rnd = 1 << shift >> 1;

    if eob < has_dc_only as i32 {
        let mut dc = coeff[0].as_::<i32>();
        coeff[0] = 0.as_();
        if is_rect2 {
            dc = dc * 181 + 128 >> 8;
        }
        dc = dc * 181 + 128 >> 8;
        dc = dc + rnd >> shift;
        dc = dc * 181 + 128 + 2048 >> 12;
        for y in 0..h {
            let dst = dst + (y as isize * dst.pixel_stride::<BD>());
            let dst = &mut *dst.slice_mut::<BD>(w);
            for x in 0..w {
                dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + dc);
            }
        }
        return;
    }

    let sh = cmp::min(h, 32);
    let sw = cmp::min(w, 32);

    let coeff = &mut coeff[..sh * sw];

    let row_clip_min;
    let col_clip_min;
    if BD::BITDEPTH == 8 {
        row_clip_min = i16::MIN as i32;
        col_clip_min = i16::MIN as i32;
    } else {
        row_clip_min = (!bitdepth_max) << 7;
        col_clip_min = (!bitdepth_max) << 5;
    }
    let row_clip_max = !row_clip_min;
    let col_clip_max = !col_clip_min;

    let mut tmp = [0; 64 * 64];
    let mut c = &mut tmp[..];
    for y in 0..sh {
        if is_rect2 {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_::<i32>() * 181 + 128 >> 8;
            }
        } else {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_();
            }
        }
        first_1d_fn(c, 1.try_into().unwrap(), row_clip_min, row_clip_max);
        c = &mut c[w..];
    }

    coeff.fill(0.into());
    for i in 0..w * sh {
        tmp[i] = iclip(tmp[i] + rnd >> shift, col_clip_min, col_clip_max);
    }

    for x in 0..w {
        second_1d_fn(
            &mut tmp[x..],
            w.try_into().unwrap(),
            col_clip_min,
            col_clip_max,
        );
    }

    for y in 0..h {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(w);
        for x in 0..w {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + (tmp[y * w + x] + 8 >> 4));
        }
    }
}

fn inv_txfm_add_rust<const W: usize, const H: usize, const TYPE: TxfmType, BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) {
    let shift = match (W, H) {
        (4, 4) => 0,
        (4, 8) => 0,
        (4, 16) => 1,
        (8, 4) => 0,
        (8, 8) => 1,
        (8, 16) => 1,
        (8, 32) => 2,
        (16, 4) => 1,
        (16, 8) => 1,
        (16, 16) => 2,
        (16, 32) => 1,
        (16, 64) => 2,
        (32, 8) => 2,
        (32, 16) => 1,
        (32, 32) => 2,
        (32, 64) => 1,
        (64, 16) => 2,
        (64, 32) => 1,
        (64, 64) => 2,
        _ => unreachable!(),
    };
    let has_dc_only = TYPE == DCT_DCT;

    enum Type {
        Identity,
        Dct,
        Adst,
        FlipAdst,
    }
    use Type::*;
    // For some reason, this is flipped.
    let (second, first) = match TYPE {
        IDTX => (Identity, Identity),
        DCT_DCT => (Dct, Dct),
        ADST_DCT => (Adst, Dct),
        FLIPADST_DCT => (FlipAdst, Dct),
        H_DCT => (Identity, Dct),
        DCT_ADST => (Dct, Adst),
        ADST_ADST => (Adst, Adst),
        FLIPADST_ADST => (FlipAdst, Adst),
        DCT_FLIPADST => (Dct, FlipAdst),
        ADST_FLIPADST => (Adst, FlipAdst),
        FLIPADST_FLIPADST => (FlipAdst, FlipAdst),
        V_DCT => (Dct, Identity),
        H_ADST => (Identity, Adst),
        H_FLIPADST => (Identity, FlipAdst),
        V_ADST => (Adst, Identity),
        V_FLIPADST => (FlipAdst, Identity),

WHT_WHT if (W, H) == (4, 4) => return inv_txfm_add_wht_wht_4x4_rust(dst, coeff, bd),

        _ => unreachable!(),
    };

    fn resolve_1d_fn(r#type: Type, n: usize) -> Itx1dFn {
        match (r#type, n) {
            (Identity, 4) => rav1d_inv_identity4_1d_c,
            (Identity, 8) => rav1d_inv_identity8_1d_c,
            (Identity, 16) => rav1d_inv_identity16_1d_c,
            (Identity, 32) => rav1d_inv_identity32_1d_c,
            (Dct, 4) => rav1d_inv_dct4_1d_c,
            (Dct, 8) => rav1d_inv_dct8_1d_c,
            (Dct, 16) => rav1d_inv_dct16_1d_c,
            (Dct, 32) => rav1d_inv_dct32_1d_c,
            (Dct, 64) => rav1d_inv_dct64_1d_c,
            (Adst, 4) => rav1d_inv_adst4_1d_c,
            (Adst, 8) => rav1d_inv_adst8_1d_c,
            (Adst, 16) => rav1d_inv_adst16_1d_c,
            (FlipAdst, 4) => rav1d_inv_flipadst4_1d_c,
            (FlipAdst, 8) => rav1d_inv_flipadst8_1d_c,
            (FlipAdst, 16) => rav1d_inv_flipadst16_1d_c,
            _ => unreachable!(),
        }
    }

    let first_1d_fn = resolve_1d_fn(first, W);
    let second_1d_fn = resolve_1d_fn(second, H);

    inv_txfm_add(
        dst,
        coeff,
        eob,
        W,
        H,
        shift,
        first_1d_fn,
        second_1d_fn,
        has_dc_only,
        bd,
    )
}

/// # Safety
///
/// Must be called by [`itxfm::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn inv_txfm_add_c_erased<
    const W: usize,
    const H: usize,
    const TYPE: TxfmType,
    BD: BitDepth,
>(
    _dst_ptr: *mut DynPixel,
    _stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    coeff_len: u16,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `itxfm::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn itxfm::Fn::call` passes `coeff.len()` as `coeff_len`.
    let coeff = unsafe { slice::from_raw_parts_mut(coeff.cast(), coeff_len.into()) };
    let bd = BD::from_c(bitdepth_max);
    inv_txfm_add_rust::<W, H, TYPE, BD>(dst, coeff, eob, bd)
}

wrap_fn_ptr!(unsafe extern "C" fn itxfm(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl itxfm::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        coeff: &mut [BD::Coef],
        eob: i32,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let dst_stride = dst.stride();
        let coeff_len = coeff.len() as u16;
        let coeff = coeff.as_mut_ptr().cast();
        let bd = bd.into_c();
        let dst = dst.into_ffi_safe();
        // SAFETY: Fallback `fn inv_txfm_add_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(dst_ptr, dst_stride, coeff, eob, bd, coeff_len, dst) }
    }
}

pub struct Rav1dInvTxfmDSPContext {
    pub itxfm_add: [[itxfm::Fn; N_TX_TYPES_PLUS_LL]; TxfmSize::COUNT],
}

fn inv_txfm_add_wht_wht_4x4_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [BD::Coef],
    bd: BD,
) {
    const H: usize = 4;
    const W: usize = 4;

    let coeff = &mut coeff[..W * H];

    let mut tmp = [0; W * H];
    let mut c = &mut tmp[..];
    for y in 0..H {
        for x in 0..W {
            c[x] = coeff[y + x * H].as_::<i32>() >> 2;
        }
        rav1d_inv_wht4_1d_c(c, 1.try_into().unwrap());
        c = &mut c[W..];
    }
    coeff.fill(0.into());

    for x in 0..W {
        rav1d_inv_wht4_1d_c(&mut tmp[x..], H.try_into().unwrap());
    }

    for y in 0..H {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(W);
        for x in 0..W {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + tmp[y * W + x]);
        }
    }
}





















impl Rav1dInvTxfmDSPContext {
    const fn assign<const W: usize, const H: usize, BD: BitDepth>(mut self) -> Self {
        let tx = TxfmSize::from_wh(W, H) as usize;

        macro_rules! assign {
            ($type:expr) => {{
                self.itxfm_add[tx][$type as usize] =
                    itxfm::Fn::new(inv_txfm_add_c_erased::<W, H, $type, BD>);
            }};
        }

        let max_wh = if W > H { W } else { H };

        let assign84 = W * H <= 8 * 16;
        let assign16 = assign84 || (W == 16 && H == 16);
        let assign32 = assign16 || max_wh == 32;
        let assign64 = assign32 || max_wh == 64;

        if assign84 {
            assign!(H_FLIPADST);
            assign!(V_FLIPADST);
            assign!(H_ADST);
            assign!(V_ADST);
        }
        if assign16 {
            assign!(DCT_ADST);
            assign!(ADST_DCT);
            assign!(ADST_ADST);
            assign!(ADST_FLIPADST);
            assign!(FLIPADST_ADST);
            assign!(DCT_FLIPADST);
            assign!(FLIPADST_DCT);
            assign!(FLIPADST_FLIPADST);
            assign!(H_DCT);
            assign!(V_DCT);
        }
        if assign32 {
            assign!(IDTX);
        }
        if assign64 {
            assign!(DCT_DCT);
        }

        if W == 4 && H == 4 {
            assign!(WHT_WHT);
        }

        self
    }

    pub const fn default<BD: BitDepth>() -> Self {
        let mut c = Self {
            itxfm_add: [[itxfm::Fn::DEFAULT; N_TX_TYPES_PLUS_LL]; TxfmSize::COUNT],
        };

        c = c.assign::<4, 4, BD>();
        c = c.assign::<4, 8, BD>();
        c = c.assign::<4, 16, BD>();
        c = c.assign::<8, 4, BD>();
        c = c.assign::<8, 8, BD>();
        c = c.assign::<8, 16, BD>();
        c = c.assign::<8, 32, BD>();
        c = c.assign::<16, 4, BD>();
        c = c.assign::<16, 8, BD>();
        c = c.assign::<16, 16, BD>();
        c = c.assign::<16, 32, BD>();
        c = c.assign::<16, 64, BD>();
        c = c.assign::<32, 8, BD>();
        c = c.assign::<32, 16, BD>();
        c = c.assign::<32, 32, BD>();
        c = c.assign::<32, 64, BD>();
        c = c.assign::<64, 16, BD>();
        c = c.assign::<64, 32, BD>();
        c = c.assign::<64, 64, BD>();

        c
    }





    #[inline(always)]
    const fn init<BD: BitDepth>(self, flags: CpuFlags, bpc: u8) -> Self {


        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            let _ = bpc;
            self
        }
    }

    pub const fn new<BD: BitDepth>(flags: CpuFlags, bpc: u8) -> Self {
        Self::default::<BD>().init::<BD>(flags, bpc)
    }
}
