//! AV2 warped-motion compensation (Stage D). Self-contained i32 port of dav2d `warp_affine`
//! (recon_tmpl.c:1853) + `warp_affine_8x8_c` (mc_tmpl.c) + `get_shear_params` (warpmv.c:52).
//! Reads a reference `Plane`, writes the current FRAME buffer, in the intra-recon style.
//!
//! The warp matrix `mat[6]` comes from the parse/refmvs (reconstruct_warp_delta_matrix /
//! derive_warpmv, already verified). `get_shear_params` derives the per-8x8 shear `[α,β,γ,δ]`.
//! 8-bit: intermediate_bits=4 ⇒ warp h-shift=3, v-shift=11. Warp filter = 449-row 7×64+1 table.

use crate::av2_frame::Plane;
use crate::av2_ipred::DIV_RECIP;

include!("av2_warp_filter.rs"); // pub static MC_WARP_FILTER: [[i8; 8]; 449]
include!("av2_ext_warp_table.rs"); // pub static EXT_WARP_FILTER: [[i8; 8]; 63]

#[inline]
fn ulog2(d: u32) -> i32 {
    31 - d.leading_zeros() as i32
}
#[inline]
fn apply_sign(a: i32, b: i32) -> i32 {
    if b < 0 { -a } else { a }
}
#[inline]
fn apply_sign64(a: i32, b: i64) -> i32 {
    if b < 0 { -a } else { a }
}
/// dav2d `iclip_wmp`: clip to i16 range then round to a multiple of 64 (6-bit warp granularity).
#[inline]
fn iclip_wmp(v: i32) -> i32 {
    // dav2d warpmv.c:37 — round to a multiple of 64 then clamp to [-0x8000, 0x7fc0]. The upper
    // clamp is 0x7fc0 (511*64), NOT 0x8000: a large shear (e.g. gamma from a 0.5× scale) must land
    // on 32704, not 32768 (which overflows the i16 abcd field → -32768 and breaks warp8x8).
    ((v + 0x20 - (v < 0) as i32) & !0x3f).clamp(-0x8000, 0x7fc0)
}
/// dav2d `dav2d_resolve_divisor_32` (prec 9): `1/d ≈ recip / 2^shift`. Returns `(recip, shift)`.
#[inline]
pub fn resolve_divisor_32(d: u32) -> (i32, i32) {
    let shift = ulog2(d);
    let e = (d - (1u32 << shift)) as i32;
    let f = if shift > 7 { (e + (1 << (shift - 8))) >> (shift - 7) } else { e << (7 - shift) };
    (DIV_RECIP[f as usize] as i32, shift + 9)
}

/// dav2d `get_shear_params` (warpmv.c:52): derive the per-8x8 shear params `[α,β,γ,δ]` from the
/// warp matrix. Returns `None` if the warp is not a valid affine (dav falls back to `ext_warp`).
pub fn get_shear_params(mat: &[i32; 6]) -> Option<[i16; 4]> {
    if mat[2] <= 0 {
        return None;
    }
    let alpha = iclip_wmp(mat[2] - 0x10000);
    let beta = iclip_wmp(mat[3]);
    let (y0, shift) = resolve_divisor_32(mat[2].unsigned_abs());
    let y = apply_sign(y0, mat[2]);
    let v1 = mat[4] as i64 * 0x10000 * y as i64;
    let rnd = (1i64 << shift) >> 1;
    let gamma = iclip_wmp(apply_sign64(((v1.abs() + rnd) >> shift) as i32, v1));
    let v2 = mat[3] as i64 * mat[4] as i64 * y as i64;
    let delta = iclip_wmp(mat[5] - apply_sign64(((v2.abs() + rnd) >> shift) as i32, v2) - 0x10000);
    let affine =
        4 * alpha.abs() + 7 * beta.abs() < 0x30000 && 4 * gamma.abs() + 4 * delta.abs() < 0x30000;
    affine.then_some([alpha as i16, beta as i16, gamma as i16, delta as i16])
}

/// Warp one 8x8 sub-block into `dst` at `(ox,oy)` (stride `dstw`), reading `rf` around integer
/// offset `(dx,dy)` with per-pixel shear subpel `(mx,my)` stepping by `abcd`. dav2d `warp_affine_8x8_c`.
#[allow(clippy::too_many_arguments)]
fn warp8x8(
    rf: &Plane, dst: &mut [i32], dstw: usize, ox: usize, oy: usize, dx: i32, dy: i32, abcd: &[i16; 4],
    mx: i32, my: i32, bdmax: i32,
) {
    let (rw, rh) = (rf.w as i32, rf.h as i32);
    let get = |x: i32, y: i32| -> i32 { rf.at(x.clamp(0, rw - 1) as usize, y.clamp(0, rh - 1) as usize) };
    const H_SH: i32 = 7 - 4; // 3
    const V_SH: i32 = 7 + 4; // 11
    let mut mid = [[0i32; 8]; 15];
    let mut mxr = mx;
    for yy in 0..15usize {
        if !crate::av2_recon::work_tick("av2_warp:77") { break; }
        let sy = dy - 3 + yy as i32;
        let mut tmx = mxr;
        for xx in 0..8usize {
            if !crate::av2_recon::work_tick("av2_warp:80") { break; }
            let f = &MC_WARP_FILTER[(3 * 64 + ((tmx + 512) >> 10)) as usize];
            let s: i32 = (0..8).map(|k| f[k] as i32 * get(dx + xx as i32 + k as i32 - 3, sy)).sum();
            mid[yy][xx] = (s + ((1 << H_SH) >> 1)) >> H_SH;
            tmx += abcd[0] as i32; // += alpha per column
        }
        mxr += abcd[1] as i32; // += beta per row
    }
    let mut myr = my;
    for yy in 0..8usize {
        if !crate::av2_recon::work_tick("av2_warp:89") { break; }
        let mut tmy = myr;
        for xx in 0..8usize {
            if !crate::av2_recon::work_tick("av2_warp:91") { break; }
            let f = &MC_WARP_FILTER[(3 * 64 + ((tmy + 512) >> 10)) as usize];
            let s: i32 = (0..8).map(|k| f[k] as i32 * mid[yy + k][xx]).sum();
            dst[(oy + yy) * dstw + ox + xx] = ((s + ((1 << V_SH) >> 1)) >> V_SH).clamp(0, bdmax);
            tmy += abcd[2] as i32; // += gamma per column
        }
        myr += abcd[3] as i32; // += delta per row
    }
}

/// Warp-affine motion compensation of a `bw`×`bh` PLANE block into `dst` (stride `dstw`) from `rf`.
/// `mat` = warp matrix, `abcd` = shear params (from [`get_shear_params`]); `(bx4,by4)` = block
/// top-left in luma-mi; `ss_hor`/`ss_ver` for chroma. dav2d `warp_affine` (the ≥8px affine path).
#[allow(clippy::too_many_arguments)]
pub fn warp_affine(
    rf: &Plane, dst: &mut [i32], dstw: usize, mat: &[i32; 6], abcd: &[i16; 4], bx4: usize, by4: usize,
    bw: usize, bh: usize, ss_hor: u32, ss_ver: u32, bdmax: i32,
) {
    for y in (0..bh).step_by(8) {
        if !crate::av2_recon::work_tick("av2_warp:109") { break; }
        let src_y = by4 as i32 * 4 + ((y as i32 + 4) << ss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        for x in (0..bw).step_by(8) {
            if !crate::av2_recon::work_tick("av2_warp:113") { break; }
            let src_x = bx4 as i32 * 4 + ((x as i32 + 4) << ss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> ss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> ss_ver;
            let dx = (mvx >> 16) as i32 - 4;
            let mx = (((mvx as i32) & 0xffff) - abcd[0] as i32 * 4 - abcd[1] as i32 * 7) & !0x3f;
            let dy = (mvy >> 16) as i32 - 4;
            let my = (((mvy as i32) & 0xffff) - abcd[2] as i32 * 4 - abcd[3] as i32 * 4) & !0x3f;
            warp8x8(rf, dst, dstw, x, y, dx, dy, abcd, mx, my, bdmax);
            if std::env::var("MWARP2").is_ok() && ss_hor == 0 && bx4 == 120 / 4 && by4 == 80 / 4 {
                crate::dlog!("[MWARP2] sub=({x},{y}) dxy=({dx},{dy}) mxy=({mx},{my}) abcd={abcd:?} out0={} out57={}",
                    dst[y * dstw + x], dst[(y + 7) * dstw + x + 5]);
            }
        }
    }
}

/// dav2d `ext_warp` (recon_tmpl.c:1768) — the affine-MC path used when the shear params are not a
/// valid affine (`get_shear_params` → None) or a plane dim < 8. Each 4x4 sub-block does a
/// translational MC whose MV is sampled from the affine matrix at that 4x4's centre; the subpel
/// filter is `EXT_WARP_FILTER` (6-bit subpel, filter_type=-1 ⇒ bits=7). dav's emu_edge cache window
/// is exactly bounded to the reads and only clamps at frame edges, so clamp-reading the frame
/// (as in warp8x8) yields identical pixels. Luma pass = ss 0/0; chroma = ss 1/1.
pub fn ext_warp(
    rf: &Plane, dst: &mut [i32], dstw: usize, mat: &[i32; 6], bx4: usize, by4: usize,
    bw: usize, bh: usize, ss_hor: u32, ss_ver: u32, bdmax: i32,
) {
    let (rw, rh) = (rf.w as i32, rf.h as i32);
    // dav2d ext_warp (recon_tmpl.c:1768): 8x8 (or block-size, if smaller) GROUPS each compute an
    // emu-edge WINDOW from the GROUP-CENTER MV; every 4x4 sub-block's 8-tap reads clamp into that
    // window — NOT the frame. With extreme warp matrices a 4x4's own MV can stray outside its
    // group's window, and the window-edge replicate differs from a frame clamp (±1..3 px).
    let sw = (bw as i32).min(8);
    let sh = (bh as i32).min(8);
    let (hsw, hsh) = (sw >> 1, sh >> 1);
    let mut gy = 0i32;
    while gy < bh as i32 {
        if !crate::av2_recon::work_tick("av2_warp:149") { break; }
        let src_y = by4 as i32 * 4 + ((gy + hsh) << ss_ver);
        let g_mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let g_mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut gx = 0i32;
        while gx < bw as i32 {
            if !crate::av2_recon::work_tick("av2_warp:154") { break; }
            let src_x = bx4 as i32 * 4 + ((gx + hsw) << ss_hor);
            let g_mvx = (mat[2] as i64 * src_x as i64 + g_mat3_y) >> ss_hor;
            let g_mvy = (mat[4] as i64 * src_x as i64 + g_mat5_y) >> ss_ver;
            let left_window = (g_mvx >> 16) as i32 - hsw - 3;
            let top_window = (g_mvy >> 16) as i32 - hsh - 3;
            let left = left_window.clamp(0, rw - 1);
            let right = (left_window + sw + 7).clamp(1, rw);
            let top = top_window.clamp(0, rh - 1);
            let bottom = (top_window + sh + 7).clamp(1, rh);
            let get = |x: i32, y: i32| -> i32 {
                rf.at(x.clamp(left, right - 1) as usize, y.clamp(top, bottom - 1) as usize)
            };
            let mut yy = gy;
            while yy < gy + sh {
                if !crate::av2_recon::work_tick("av2_warp:168") { break; }
                let src_y = by4 as i32 * 4 + ((yy + 2) << ss_ver);
                let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
                let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
                let mut xx = gx;
                while xx < gx + sw {
                    if !crate::av2_recon::work_tick("av2_warp:173") { break; }
                    let src_x = bx4 as i32 * 4 + ((xx + 2) << ss_hor);
                    let mvx = ((mat[2] as i64 * src_x as i64 + mat3_y) >> ss_hor) + 0x200;
                    let mvy = ((mat[4] as i64 * src_x as i64 + mat5_y) >> ss_ver) + 0x200;
                    let dx = (mvx >> 16) as i32 - 2;
                    let mx = ((mvx >> 10) & 63) as i32;
                    let dy = (mvy >> 16) as i32 - 2;
                    let my = ((mvy >> 10) & 63) as i32;
                    let (ox, oy) = (xx as usize, yy as usize);
                    // ext_warp4x4 = put_8tap 4x4, filter_type=-1 (bits=7, intermediate_bits=4).
                    if mx != 0 && my != 0 {
                        let fh = &EXT_WARP_FILTER[(mx - 1) as usize];
                        let fv = &EXT_WARP_FILTER[(my - 1) as usize];
                        let mut mid = [[0i32; 4]; 11]; // 4 + 7 tap rows
                        for (j, row) in mid.iter_mut().enumerate() {
                            let sy = dy - 3 + j as i32;
                            for (i, m) in row.iter_mut().enumerate() {
                                let s: i32 = (0..8).map(|k| fh[k] as i32 * get(dx + i as i32 + k as i32 - 3, sy)).sum();
                                *m = (s + 4) >> 3; // sh = bits - intermediate_bits = 3
                            }
                        }
                        for j in 0..4usize {
                            if !crate::av2_recon::work_tick("av2_warp:198") { break; }
                            for i in 0..4usize {
                                if !crate::av2_recon::work_tick("av2_warp:199") { break; }
                                let s: i32 = (0..8).map(|k| fv[k] as i32 * mid[j + k][i]).sum();
                                dst[(oy + j) * dstw + ox + i] = ((s + 1024) >> 11).clamp(0, bdmax); // sh = bits + interm = 11
                            }
                        }
                    } else if mx != 0 {
                        let fh = &EXT_WARP_FILTER[(mx - 1) as usize];
                        for j in 0..4usize {
                            if !crate::av2_recon::work_tick("av2_warp:206") { break; }
                            let sy = dy + j as i32;
                            for i in 0..4usize {
                                if !crate::av2_recon::work_tick("av2_warp:208") { break; }
                                let s: i32 = (0..8).map(|k| fh[k] as i32 * get(dx + i as i32 + k as i32 - 3, sy)).sum();
                                dst[(oy + j) * dstw + ox + i] = ((s + 68) >> 7).clamp(0, bdmax); // intermediate_rnd=68, bits=7
                            }
                        }
                    } else if my != 0 {
                        let fv = &EXT_WARP_FILTER[(my - 1) as usize];
                        for j in 0..4usize {
                            if !crate::av2_recon::work_tick("av2_warp:215") { break; }
                            for i in 0..4usize {
                                if !crate::av2_recon::work_tick("av2_warp:216") { break; }
                                let s: i32 = (0..8).map(|k| fv[k] as i32 * get(dx + i as i32, dy + j as i32 + k as i32 - 3)).sum();
                                dst[(oy + j) * dstw + ox + i] = ((s + 64) >> 7).clamp(0, bdmax); // sh = bits = 7
                            }
                        }
                    } else {
                        for j in 0..4usize {
                            if !crate::av2_recon::work_tick("av2_warp:222") { break; }
                            for i in 0..4usize {
                                if !crate::av2_recon::work_tick("av2_warp:223") { break; }
                                dst[(oy + j) * dstw + ox + i] = get(dx + i as i32, dy + j as i32);
                            }
                        }
                    }
                    xx += 4;
                }
                yy += 4;
            }
            gx += sw;
        }
        gy += sh;
    }
}
