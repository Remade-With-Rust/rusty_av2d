//! AV2 inter prediction (Stage D) — motion compensation.
//!
//! Self-contained i32 MC mirroring the intra recon style: reads a reference `Plane`
//! (frame-1 reconstructed+filtered output) and writes the current FRAME buffer. Ports
//! dav2d `mc()` (recon_tmpl.c:1600) + `put_8tap_c` (mc_tmpl.c) for the non-scaled,
//! single-reference translational path. Subpel via `dav2d_mc_subpel_filters[6][15][8]`.
//!
//! 8-bit constants: intermediate_bits=4, bits=6 (filter_type>=0) ⇒ 2D h-shift=2 / v-shift=10;
//! h-only rnd=34 shift=6; v-only rnd=32 shift=6.

use crate::av2_frame::Plane;

/// AV2 subpel interpolation filters (dav2d `dav2d_mc_subpel_filters[6][15][8]`, tables.c:763).
/// Index 0..2 = REGULAR/SMOOTH/SHARP (width>4); 3..4 = narrow REGULAR/SMOOTH (width<=4);
/// 5 = bilinear (scaled). Row = subpel position (1..15) minus 1; 8 taps centred on tap 3.
#[rustfmt::skip]
pub static MC_SUBPEL_FILTERS: [[[i8; 8]; 15]; 6] = [
    // [0] 8TAP_REGULAR
    [[0,1,-3,63,4,-1,0,0],[0,1,-5,61,9,-2,0,0],[0,1,-6,58,14,-4,1,0],[0,1,-7,55,19,-5,1,0],
     [0,1,-7,51,24,-6,1,0],[0,1,-8,47,29,-6,1,0],[0,1,-7,42,33,-6,1,0],[0,1,-7,38,38,-7,1,0],
     [0,1,-6,33,42,-7,1,0],[0,1,-6,29,47,-8,1,0],[0,1,-6,24,51,-7,1,0],[0,1,-5,19,55,-7,1,0],
     [0,1,-4,14,58,-6,1,0],[0,0,-2,9,61,-5,1,0],[0,0,-1,4,63,-3,1,0]],
    // [1] 8TAP_SMOOTH
    [[0,1,14,31,17,1,0,0],[0,0,13,31,18,2,0,0],[0,0,11,31,20,2,0,0],[0,0,10,30,21,3,0,0],
     [0,0,9,29,22,4,0,0],[0,0,8,28,23,5,0,0],[0,-1,8,27,24,6,0,0],[0,-1,7,26,26,7,-1,0],
     [0,0,6,24,27,8,-1,0],[0,0,5,23,28,8,0,0],[0,0,4,22,29,9,0,0],[0,0,3,21,30,10,0,0],
     [0,0,2,20,31,11,0,0],[0,0,2,18,31,13,0,0],[0,0,1,17,31,14,1,0]],
    // [2] 8TAP_SHARP
    [[-1,1,-3,63,4,-1,1,0],[-1,3,-6,62,8,-3,2,-1],[-1,4,-9,60,13,-5,3,-1],[-2,5,-11,58,19,-7,3,-1],
     [-2,5,-11,54,24,-9,4,-1],[-2,5,-12,50,30,-10,4,-1],[-2,5,-12,45,35,-11,5,-1],[-2,6,-12,40,40,-12,6,-2],
     [-1,5,-11,35,45,-12,5,-2],[-1,4,-10,30,50,-12,5,-2],[-1,4,-9,24,54,-11,5,-2],[-1,3,-7,19,58,-11,5,-2],
     [-1,3,-5,13,60,-9,4,-1],[-1,2,-3,8,62,-6,3,-1],[0,1,-1,4,63,-3,1,-1]],
    // [3] narrow REGULAR (width<=4)
    [[0,0,-2,63,4,-1,0,0],[0,0,-4,61,9,-2,0,0],[0,0,-5,58,14,-3,0,0],[0,0,-6,55,19,-4,0,0],
     [0,0,-6,51,24,-5,0,0],[0,0,-7,47,29,-5,0,0],[0,0,-6,42,33,-5,0,0],[0,0,-6,38,38,-6,0,0],
     [0,0,-5,33,42,-6,0,0],[0,0,-5,29,47,-7,0,0],[0,0,-5,24,51,-6,0,0],[0,0,-4,19,55,-6,0,0],
     [0,0,-3,14,58,-5,0,0],[0,0,-2,9,61,-4,0,0],[0,0,-1,4,63,-2,0,0]],
    // [4] narrow SMOOTH (width<=4)
    [[0,0,15,31,17,1,0,0],[0,0,13,31,18,2,0,0],[0,0,11,31,20,2,0,0],[0,0,10,30,21,3,0,0],
     [0,0,9,29,22,4,0,0],[0,0,8,28,23,5,0,0],[0,0,7,27,24,6,0,0],[0,0,6,26,26,6,0,0],
     [0,0,6,24,27,7,0,0],[0,0,5,23,28,8,0,0],[0,0,4,22,29,9,0,0],[0,0,3,21,30,10,0,0],
     [0,0,2,20,31,11,0,0],[0,0,2,18,31,13,0,0],[0,0,1,17,31,15,0,0]],
    // [5] bilinear (scaled)
    [[0,0,0,60,4,0,0,0],[0,0,0,56,8,0,0,0],[0,0,0,52,12,0,0,0],[0,0,0,48,16,0,0,0],
     [0,0,0,44,20,0,0,0],[0,0,0,40,24,0,0,0],[0,0,0,36,28,0,0,0],[0,0,0,32,32,0,0,0],
     [0,0,0,28,36,0,0,0],[0,0,0,24,40,0,0,0],[0,0,0,20,44,0,0,0],[0,0,0,16,48,0,0,0],
     [0,0,0,12,52,0,0,0],[0,0,0,8,56,0,0,0],[0,0,0,4,60,0,0,0]],
];

/// Select the 8-tap filter row for subpel position `pos` (1..15) given block dimension `d`
/// (width for the H filter, height for the V filter) and `filter_type` (0=REGULAR/1=SMOOTH/
/// 2=SHARP). dav2d `GET_H_FILTER`/`GET_V_FILTER`: d<=4 uses the narrow table (3 + type&1).
#[inline]
fn mc_filter(pos: i32, d: usize, filter_type: usize) -> &'static [i8; 8] {
    // filter_type 5 = BILINEAR (dav2d DAV2D_FILTER_BILINEAR) — used unconditionally (regardless of
    // block size) for the intrabc block-copy subpel; the size-narrowing only applies to 8-tap.
    let i = if filter_type == 5 { 5 } else if d > 4 { filter_type } else { 3 + (filter_type & 1) };
    &MC_SUBPEL_FILTERS[i][(pos - 1) as usize]
}

/// Translational single-reference luma motion compensation. Thin wrapper over [`mc_translate`]
/// with no subsampling (`ss_hor=ss_ver=0`); `(px0,py0)`/`w`/`h` in luma pixels.
#[allow(clippy::too_many_arguments)]
pub fn mc_translate_luma(
    rf: &Plane, dst: &mut [i32], dst_w: usize, px0: usize, py0: usize, w: usize, h: usize,
    mvy: i32, mvx: i32, filter_type: usize, bdmax: i32,
) {
    mc_translate(rf, dst, dst_w, px0, py0, w, h, mvy, mvx, filter_type, 0, 0, bdmax);
}

/// Translational single-reference motion compensation into `dst` (`w`×`h` PLANE pixels, row
/// stride `dst_w`) from reference plane `rf`. `(px0,py0)` = block top-left in PLANE pixels;
/// `mv=(mvy,mvx)` in 1/8-LUMA-pel; `ss_hor`/`ss_ver` ∈ {0,1} for 4:2:0 chroma. Border reads clamp
/// to the frame edge (== dav2d `emu_edge` replication). dav2d `mc()` non-scaled path + `put_8tap_c`.
/// Subpel (dav2d recon_tmpl.c:1621): `m = mv & (15 >> !ss)`, filter pos `= m << !ss` (luma even
/// 0..14, chroma all 0..15); integer offset `= mv >> (3 + ss)`.
#[allow(clippy::too_many_arguments)]
pub fn mc_translate(
    rf: &Plane, dst: &mut [i32], dst_w: usize, px0: usize, py0: usize, w: usize, h: usize,
    mvy: i32, mvx: i32, filter_type: usize, ss_hor: u32, ss_ver: u32, bdmax: i32,
) {
    if dst.len() < (h.max(1) - 1) * dst_w + w {
        crate::dlog!("[MCT-BAD] dst.len={} dst_w={dst_w} w={w} h={h} px0={px0} py0={py0} rf={}x{}", dst.len(), rf.w, rf.h);
    }
    let (rw, rh) = (rf.w as i32, rf.h as i32);
    let get = |x: i32, y: i32| -> i32 { rf.at(x.clamp(0, rw - 1) as usize, y.clamp(0, rh - 1) as usize) };
    let mx = (mvx & (15 >> (1 - ss_hor))) << (1 - ss_hor);
    let my = (mvy & (15 >> (1 - ss_ver))) << (1 - ss_ver);
    let dx = px0 as i32 + (mvx >> (3 + ss_hor));
    let dy = py0 as i32 + (mvy >> (3 + ss_ver));
    const IB: i32 = 4; // intermediate_bits (8-bit)
    const BITS: i32 = 6; // 6 + (filter_type < 0); filter_type >= 0 here
    let fh = (mx != 0).then(|| mc_filter(mx, w, filter_type));
    let fv = (my != 0).then(|| mc_filter(my, h, filter_type));
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let (h_sh, v_sh) = (BITS - IB, BITS + IB); // 2, 10
            let midh = h + 7;
            let mut mid = vec![0i32; midh * w];
            for yy in 0..midh {
                if !crate::av2_recon::work_tick("av2_inter:100") { break; }
                let sy = dy - 3 + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:102") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fh[k] as i32 * get(sx + k as i32 - 3, sy)).sum();
                    mid[yy * w + xx] = (s + ((1 << h_sh) >> 1)) >> h_sh;
                }
            }
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:108") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:109") { break; }
                    let s: i32 = (0..8).map(|k| fv[k] as i32 * mid[(yy + k) * w + xx]).sum();
                    dst[yy * dst_w + xx] = ((s + ((1 << v_sh) >> 1)) >> v_sh).clamp(0, bdmax);
                }
            }
        }
        (Some(fh), None) => {
            let rnd = ((1 << BITS) >> 1) + ((1 << (BITS - IB)) >> 1); // 34
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:117") { break; }
                let sy = dy + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:119") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fh[k] as i32 * get(sx + k as i32 - 3, sy)).sum();
                    dst[yy * dst_w + xx] = ((s + rnd) >> BITS).clamp(0, bdmax);
                }
            }
        }
        (None, Some(fv)) => {
            let rnd = (1 << BITS) >> 1; // 32
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:128") { break; }
                let sy = dy + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:130") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fv[k] as i32 * get(sx, sy + k as i32 - 3)).sum();
                    dst[yy * dst_w + xx] = ((s + rnd) >> BITS).clamp(0, bdmax);
                }
            }
        }
        (None, None) => {
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:138") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:139") { break; }
                    let o = yy * dst_w + xx;
                    if o >= dst.len() {
                        continue; // HARDENING: corrupt block dims vs the caller's buffer
                    }
                    dst[o] = get(dx + xx as i32, dy + yy as i32);
                }
            }
        }
    }
}

/// PREP-precision translational MC (dav2d `prep_8tap_c` + `prep_c`, mc_tmpl.c:266/64, 8-bit:
/// intermediate_bits=4, PREP_BIAS=0): output stays at +4-bit precision, NO final clamp — the
/// compound blend kernels (avg/w_avg/mask/w_mask) consume these intermediates.
#[allow(clippy::too_many_arguments)]
pub fn mc_translate_prep(
    rf: &Plane, dst: &mut [i32], dst_w: usize, px0: usize, py0: usize, w: usize, h: usize,
    mvy: i32, mvx: i32, filter_type: usize, ss_hor: u32, ss_ver: u32,
) {
    let (rw, rh) = (rf.w as i32, rf.h as i32);
    let get = |x: i32, y: i32| -> i32 { rf.at(x.clamp(0, rw - 1) as usize, y.clamp(0, rh - 1) as usize) };
    let mx = (mvx & (15 >> (1 - ss_hor))) << (1 - ss_hor);
    let my = (mvy & (15 >> (1 - ss_ver))) << (1 - ss_ver);
    let dx = px0 as i32 + (mvx >> (3 + ss_hor));
    let dy = py0 as i32 + (mvy >> (3 + ss_ver));
    const IB: i32 = 4; // intermediate_bits (8-bit); PREP_BIAS = 0
    const BITS: i32 = 6;
    let fh = (mx != 0).then(|| mc_filter(mx, w, filter_type));
    let fv = (my != 0).then(|| mc_filter(my, h, filter_type));
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let h_sh = BITS - IB; // 2
            let midh = h + 7;
            let mut mid = vec![0i32; midh * w];
            for yy in 0..midh {
                if !crate::av2_recon::work_tick("av2_inter:174") { break; }
                let sy = dy - 3 + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:176") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fh[k] as i32 * get(sx + k as i32 - 3, sy)).sum();
                    mid[yy * w + xx] = (s + ((1 << h_sh) >> 1)) >> h_sh;
                }
            }
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:182") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:183") { break; }
                    let s: i32 = (0..8).map(|k| fv[k] as i32 * mid[(yy + k) * w + xx]).sum();
                    dst[yy * dst_w + xx] = (s + ((1 << BITS) >> 1)) >> BITS;
                }
            }
        }
        (Some(fh), None) => {
            let sh = BITS - IB; // 2
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:191") { break; }
                let sy = dy + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:193") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fh[k] as i32 * get(sx + k as i32 - 3, sy)).sum();
                    dst[yy * dst_w + xx] = (s + ((1 << sh) >> 1)) >> sh;
                }
            }
        }
        (None, Some(fv)) => {
            let sh = BITS - IB; // 2
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:202") { break; }
                let sy = dy + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:204") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fv[k] as i32 * get(sx, sy + k as i32 - 3)).sum();
                    dst[yy * dst_w + xx] = (s + ((1 << sh) >> 1)) >> sh;
                }
            }
        }
        (None, None) => {
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:212") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:213") { break; }
                    dst[yy * dst_w + xx] = get(dx + xx as i32, dy + yy as i32) << IB;
                }
            }
        }
    }
}

/// dav2d `avg_c` (mc_tmpl.c:673): (t1 + t2 + 16) >> 5, clamped.
pub fn comp_avg(dst: &mut [i32], dst_w: usize, t1: &[i32], t2: &[i32], w: usize, h: usize, bdmax: i32) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("av2_inter:223") { break; }
        for x in 0..w {
            if !crate::av2_recon::work_tick("av2_inter:224") { break; }
            dst[y * dst_w + x] = ((t1[y * w + x] + t2[y * w + x] + 16) >> 5).clamp(0, bdmax);
        }
    }
}

/// dav2d `w_avg_c` (mc_tmpl.c:694): (t1*wt + t2*(16-wt) + 128) >> 8, clamped.
pub fn comp_w_avg(dst: &mut [i32], dst_w: usize, t1: &[i32], t2: &[i32], w: usize, h: usize, wt: i32, bdmax: i32) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("av2_inter:232") { break; }
        for x in 0..w {
            if !crate::av2_recon::work_tick("av2_inter:233") { break; }
            dst[y * dst_w + x] = ((t1[y * w + x] * wt + t2[y * w + x] * (16 - wt) + 128) >> 8).clamp(0, bdmax);
        }
    }
}

/// dav2d `mask_c` (mc_tmpl.c:716): (t1*m + t2*(64-m) + 512) >> 10, clamped. `mask` stride = w.
pub fn comp_mask(dst: &mut [i32], dst_w: usize, t1: &[i32], t2: &[i32], w: usize, h: usize, mask: &[u8], bdmax: i32) {
    for y in 0..h {
        if !crate::av2_recon::work_tick("av2_inter:241") { break; }
        for x in 0..w {
            if !crate::av2_recon::work_tick("av2_inter:242") { break; }
            let m = mask[y * w + x] as i32;
            dst[y * dst_w + x] = ((t1[y * w + x] * m + t2[y * w + x] * (64 - m) + 512) >> 10).clamp(0, bdmax);
        }
    }
}

/// dav2d `w_mask_c` 420 variant (mc_tmpl.c:754): SEG difference-mask blend. Computes the per-pixel
/// mask from |t1-t2| (8-bit: mask_sh=8, mask_rnd=8), blends, and stores the 2x2-subsampled mask
/// into `mask_out` (chroma resolution, stride `mask_stride`) for the chroma blend.
#[allow(clippy::too_many_arguments)]
pub fn comp_w_mask_ss(
    dst: &mut [i32], dst_w: usize, t1: &[i32], t2: &[i32], w: usize, h: usize,
    mask_out: &mut [u8], mask_stride: usize, sign: i32, ss_hor: usize, ss_ver: usize, bdmax: i32,
) {
    // dav w_mask (mc_tmpl.c, dav1d heritage): blend by the derived 6-bit mask and store the
    // CHROMA-resolution mask: 444 = m; 422 = (m+n+1-sign)>>1; 420 = two-row fold
    // (m+n [+row0] + 2 - sign) >> 2.
    const MASK_SH: i32 = 8; // bitdepth(8) + intermediate_bits(4) - 4
    const MASK_RND: i32 = 1 << (MASK_SH - 5);
    let mut mrow = 0usize;
    for y in 0..h {
        if !crate::av2_recon::work_tick("av2_inter:263") { break; }
        let mut x = 0usize;
        while x < w {
            if !crate::av2_recon::work_tick("av2_inter:265") { break; }
            let d = t1[y * w + x] - t2[y * w + x];
            let m = (38 + ((d.abs() + MASK_RND) >> MASK_SH)).min(64);
            dst[y * dst_w + x] = ((d * m + t2[y * w + x] * 64 + 512) >> 10).clamp(0, bdmax);
            if ss_hor == 1 {
                x += 1;
                let d2 = t1[y * w + x] - t2[y * w + x];
                let n = (38 + ((d2.abs() + MASK_RND) >> MASK_SH)).min(64);
                dst[y * dst_w + x] = ((d2 * n + t2[y * w + x] * 64 + 512) >> 10).clamp(0, bdmax);
                let cell = &mut mask_out[mrow * mask_stride + (x >> 1)];
                if ss_ver == 1 {
                    if (h - y) & 1 == 1 {
                        *cell = ((m + n + *cell as i32 + 2 - sign) >> 2) as u8;
                    } else {
                        *cell = (m + n) as u8;
                    }
                } else {
                    *cell = ((m + n + 1 - sign) >> 1) as u8;
                }
            } else {
                debug_assert!(ss_ver == 0);
                mask_out[mrow * mask_stride + x] = m as u8;
            }
            x += 1;
        }
        if ss_ver == 0 || (h - y) & 1 == 1 {
            mrow += 1;
        }
    }
}

pub fn bacp_mask(
    mask: &mut [u8], stride: usize, bw: usize, bh: usize,
    x0: i32, y0: i32, x1: i32, y1: i32, fw: i32, fh: i32,
) {
    for y in 0..bh {
        if !crate::av2_recon::work_tick("av2_inter:301") { break; }
        for x in 0..bw {
            if !crate::av2_recon::work_tick("av2_inter:302") { break; }
            let p0 = (x0 + x as i32) >= 0 && (x0 + (x as i32)) < fw && (y0 + y as i32) >= 0 && (y0 + y as i32) < fh;
            let p1 = (x1 + x as i32) >= 0 && (x1 + (x as i32)) < fw && (y1 + y as i32) >= 0 && (y1 + y as i32) < fh;
            mask[y * stride + x] = (32 * (p0 as i32 - p1 as i32 + 1)) as u8;
        }
    }
}

// ===================== TIP / OPFL prediction kernels (dav2d mc_tmpl.c + recon_tmpl.c) =====================

/// Windowed sample fetch (dav `mc`/`mc_opfl` emu_edge semantics): replicate-pad from the
/// rectangle [l..r) x [t..b) of the plane.
#[inline]
fn wget(rf: &Plane, x: i32, y: i32, l: i32, r: i32, t: i32, b: i32) -> i32 {
    // HARDENING: a corrupt warp/ref rectangle can invert or exceed the plane — clamp into it.
    let (rx, by) = (r.max(l + 1).min(rf.w as i32), b.max(t + 1).min(rf.h as i32));
    let (lx, ty) = (l.max(0).min(rx - 1), t.max(0).min(by - 1));
    rf.at(x.clamp(lx, rx - 1) as usize, y.clamp(ty, by - 1) as usize)
}

/// dav2d `put_bilin_c` via the windowed `mc()` (8-bit PUT, mv at 1/8-pel, bilinear 2-tap).
/// FILTER_BILIN = 16*v0 + m*(v1-v0), m = (mv&7)<<1; H sh=0 then (+8)>>4; HV mid sh=0 then (+128)>>8.
pub fn put_bilin_win(
    rf: &Plane, dst: &mut [i32], dstride: usize, bx_px: i32, by_px: i32, w: usize, h: usize,
    mvy: i32, mvx: i32, win: (i32, i32, i32, i32),
) {
    let (l, r, t, b) = win;
    let bdmax = crate::av2_frame::BDMAX.with(|c| c.get());
    let mx = (mvx & 7) << 1;
    let my = (mvy & 7) << 1;
    let dx = bx_px + (mvx >> 3);
    let dy = by_px + (mvy >> 3);
    let bil = |v0: i32, v1: i32, m: i32| 16 * v0 + m * (v1 - v0);
    if mx != 0 && my != 0 {
        let mut mid = vec![0i32; (h + 1) * w];
        for yy in 0..h + 1 {
            if !crate::av2_recon::work_tick("av2_inter:337") { break; }
            for xx in 0..w {
                if !crate::av2_recon::work_tick("av2_inter:338") { break; }
                let (sx, sy) = (dx + xx as i32, dy + yy as i32);
                mid[yy * w + xx] = bil(wget(rf, sx, sy, l, r, t, b), wget(rf, sx + 1, sy, l, r, t, b), mx);
            }
        }
        for yy in 0..h {
            if !crate::av2_recon::work_tick("av2_inter:343") { break; }
            for xx in 0..w {
                if !crate::av2_recon::work_tick("av2_inter:344") { break; }
                let v = bil(mid[yy * w + xx], mid[(yy + 1) * w + xx], my);
                dst[yy * dstride + xx] = ((v + 128) >> 8).clamp(0, bdmax);
            }
        }
    } else if mx != 0 {
        for yy in 0..h {
            if !crate::av2_recon::work_tick("av2_inter:350") { break; }
            for xx in 0..w {
                if !crate::av2_recon::work_tick("av2_inter:351") { break; }
                let (sx, sy) = (dx + xx as i32, dy + yy as i32);
                let px = bil(wget(rf, sx, sy, l, r, t, b), wget(rf, sx + 1, sy, l, r, t, b), mx);
                dst[yy * dstride + xx] = ((px + 8) >> 4).clamp(0, bdmax);
            }
        }
    } else if my != 0 {
        for yy in 0..h {
            if !crate::av2_recon::work_tick("av2_inter:358") { break; }
            for xx in 0..w {
                if !crate::av2_recon::work_tick("av2_inter:359") { break; }
                let (sx, sy) = (dx + xx as i32, dy + yy as i32);
                let v = bil(wget(rf, sx, sy, l, r, t, b), wget(rf, sx, sy + 1, l, r, t, b), my);
                dst[yy * dstride + xx] = ((v + 8) >> 4).clamp(0, bdmax);
            }
        }
    } else {
        for yy in 0..h {
            if !crate::av2_recon::work_tick("av2_inter:366") { break; }
            for xx in 0..w {
                if !crate::av2_recon::work_tick("av2_inter:367") { break; }
                dst[yy * dstride + xx] = wget(rf, dx + xx as i32, dy + yy as i32, l, r, t, b);
            }
        }
    }
}

/// dav2d `mc_opfl` (recon_tmpl.c:1726): 8-tap PREP at **1/16-pel** with a windowed fetch.
/// `px_base`/`py_base` = block origin in THIS plane's px; mv already in 1/16-pel plane units.
pub fn prep_opfl(
    rf: &Plane, dst: &mut [i32], dstride: usize, px_base: i32, py_base: i32, w: usize, h: usize,
    mvy16: i32, mvx16: i32, filter_type: usize, win: (i32, i32, i32, i32),
) {
    if dst.len() < (h.max(1) - 1) * dstride + w {
        crate::dlog!("[PREP-BAD] dst.len={} dstride={dstride} w={w} h={h} px={px_base} py={py_base}", dst.len());
    }
    let (l, r, t, b) = win;
    let mx = mvx16 & 15;
    let my = mvy16 & 15;
    let dx = px_base + (mvx16 >> 4);
    let dy = py_base + (mvy16 >> 4);
    const IB: i32 = 4; // intermediate_bits (8-bit); PREP_BIAS = 0
    const BITS: i32 = 6;
    let fh = (mx != 0).then(|| mc_filter(mx, w, filter_type));
    let fv = (my != 0).then(|| mc_filter(my, h, filter_type));
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let h_sh = BITS - IB; // 2
            let midh = h + 7;
            let mut mid = vec![0i32; midh * w];
            for yy in 0..midh {
                if !crate::av2_recon::work_tick("av2_inter:397") { break; }
                let sy = dy - 3 + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:399") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fh[k] as i32 * wget(rf, sx + k as i32 - 3, sy, l, r, t, b)).sum();
                    mid[yy * w + xx] = (s + ((1 << h_sh) >> 1)) >> h_sh;
                }
            }
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:405") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:406") { break; }
                    let s: i32 = (0..8).map(|k| fv[k] as i32 * mid[(yy + k) * w + xx]).sum();
                    dst[yy * dstride + xx] = (s + ((1 << BITS) >> 1)) >> BITS;
                }
            }
        }
        (Some(fh), None) => {
            let sh = BITS - IB;
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:414") { break; }
                let sy = dy + yy as i32;
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:416") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fh[k] as i32 * wget(rf, sx + k as i32 - 3, sy, l, r, t, b)).sum();
                    dst[yy * dstride + xx] = (s + ((1 << sh) >> 1)) >> sh;
                }
            }
        }
        (None, Some(fv)) => {
            let sh = BITS - IB;
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:425") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:426") { break; }
                    let sx = dx + xx as i32;
                    let s: i32 = (0..8).map(|k| fv[k] as i32 * wget(rf, sx, dy + yy as i32 + k as i32 - 3, l, r, t, b)).sum();
                    dst[yy * dstride + xx] = (s + ((1 << sh) >> 1)) >> sh;
                }
            }
        }
        (None, None) => {
            for yy in 0..h {
                if !crate::av2_recon::work_tick("av2_inter:434") { break; }
                for xx in 0..w {
                    if !crate::av2_recon::work_tick("av2_inter:435") { break; }
                    dst[yy * dstride + xx] = wget(rf, dx + xx as i32, dy + yy as i32, l, r, t, b) << IB;
                }
            }
        }
    }
}

/// dav2d `sad_nxn` (mc_tmpl.c:974): SAD over rows stepped by 2.
fn sad_nxn(p0: &[i32], s0: usize, p1: &[i32], s1: usize, w: usize, h: usize) -> u32 {
    let mut sad = 0u32;
    let mut y = 0;
    while y < h {
        if !crate::av2_recon::work_tick("av2_inter:446") { break; }
        for x in 0..w {
            if !crate::av2_recon::work_tick("av2_inter:449") { break; }
            sad += (p0[y * s0 + x] - p1[y * s1 + x]).unsigned_abs();
        }
        y += 2;
    }
    // dav2d sad_nxn: the SAD is normalized back to 8-bit scale (>> bd-8) — matters for the
    // implicit early-exit threshold AND argmin ties at high bitdepth.
    sad >> bd_min8()
}

/// bd-8 for the current frame (0 at 8-bit).
#[inline]
fn bd_min8() -> u32 {
    (crate::av2_frame::BDMAX.with(|c| c.get()) as u32 + 1).trailing_zeros() - 8
}

/// dav2d `sad_refine_mv_c` (mc_tmpl.c:989): +-2 integer DMVR search over the pre-MC'd bilinear
/// windows (buffer origin = search origin, (w+4)x(h+4) SAD footprint). Returns (dy, dx).
pub fn sad_refine_mv(p0: &[i32], s0: usize, p1: &[i32], s1: usize, w: usize, h: usize, is_implicit: bool) -> (i32, i32) {
    let (sadw, sadh) = (w + 4, h + 4);
    let sad_thr = (sadw * sadh * 2) as u32;
    let mut best_sad = u32::MAX;
    let (mut best_dy, mut best_dx) = (0i32, 0i32);
    if is_implicit {
        best_sad = sad_nxn(&p0[2 * s0 + 2..], s0, &p1[2 * s1 + 2..], s1, sadw, sadh);
        best_sad = (best_sad * 7 + 7) >> 3;
        if best_sad < sad_thr {
            return (0, 0);
        }
    }
    for y_off in -2i32..=2 {
        if !crate::av2_recon::work_tick("av2_inter:479") { break; }
        for x_off in -2i32..=2 {
            if !crate::av2_recon::work_tick("av2_inter:480") { break; }
            if x_off == 0 && y_off == 0 {
                continue;
            }
            let o0 = ((2 + y_off) * s0 as i32 + (2 + x_off)) as usize;
            let o1 = ((2 - y_off) * s1 as i32 + (2 - x_off)) as usize;
            let sad = sad_nxn(&p0[o0..], s0, &p1[o1..], s1, sadw, sadh);
            if sad < best_sad {
                best_sad = sad;
                best_dx = x_off;
                best_dy = y_off;
            }
        }
    }
    (best_dy, best_dx)
}

/// dav2d `sad8x8_c` (mc_tmpl.c:1116).
pub fn sad8x8(p0: &[i32], s0: usize, p1: &[i32], s1: usize) -> u32 {
    let mut sad = 0u32;
    for y in 0..8 {
        for x in 0..8 {
            sad += (p0[y * s0 + x] - p1[y * s1 + x]).unsigned_abs();
        }
    }
    sad >> bd_min8() // dav2d sad8x8_c: normalized to 8-bit scale
}

/// One opfl regression accumulator (dav `struct OpflRegressionData`).
#[derive(Clone, Copy, Default)]
pub struct OpflReg {
    pub su2: i32,
    pub suv: i32,
    pub sv2: i32,
    pub suw: i32,
    pub svw: i32,
}

/// dav2d `opfl_derive_mv_c` (mc_tmpl.c:1030): distance-weighted diff + subpel gradients +
/// per-`bs` regression sums. `d` = the signed distance weights (d.i8[0], d.i8[1]).
pub fn opfl_derive_mv(
    p0: &[i32], s0: usize, p1: &[i32], s1: usize, w: usize, h: usize, bs: usize, d: (i32, i32),
) -> Vec<OpflReg> {
    let mut tmp0 = [0i32; 64 * 16];
    let mut tmp1 = [0i32; 64 * 16];
    // HARDENING: the fixed 64x16 scratch bounds the OPFL unit; a desynced stream can present a
    // larger nominal block — clamp instead of overrunning (valid units are <= 64x16).
    let (w, h) = (w.min(64), h.min(16));
    // dav2d hbd arm (mc_tmpl.c:1036): the difference buffers are normalized back to 8-bit
    // scale with signed rounding before the gradient/regression (avm ROUND_POWER_OF_TWO_SIGNED
    // by bd-8 in compute_pred_using_interp_grad_highbd).
    let bd_min8 = (crate::av2_frame::BDMAX.with(|c| c.get()) as u32 + 1).trailing_zeros() as i32 - 8;
    let rnd = (1i32 << bd_min8) >> 1;
    for y in 0..h {
        if !crate::av2_recon::work_tick("av2_inter:533") { break; }
        for x in 0..w {
            if !crate::av2_recon::work_tick("av2_inter:534") { break; }
            let p0p = p0[y * s0 + x];
            let p1p = p1[y * s1 + x];
            let v = d.0 * p0p + d.1 * p1p;
            if bd_min8 == 0 {
                tmp0[y * 64 + x] = v;
                tmp1[y * 64 + x] = p0p - p1p;
            } else {
                tmp0[y * 64 + x] = (v + rnd - (v < 0) as i32) >> bd_min8;
                tmp1[y * 64 + x] = (p0p - p1p + rnd - (p1p > p0p) as i32) >> bd_min8;
            }
        }
    }
    let mut gx0 = [0i32; 64 * 16];
    let mut gy0 = [0i32; 64 * 16];
    let mut bx = 0usize;
    while bx < w {
        if !crate::av2_recon::work_tick("av2_inter:548") { break; }
        let x_end = (bx + 16).min(w);
        let min_x = (bx & !15) as i32;
        let max_x = x_end as i32 - 1;
        let (min_y, max_y) = (0i32, h as i32 - 1);
        for y in 0..h as i32 {
            if !crate::av2_recon::work_tick("av2_inter:556") { break; }
            for x in bx as i32..x_end as i32 {
                if !crate::av2_recon::work_tick("av2_inter:557") { break; }
                let p0v = tmp0[(y * 64 + (x - 2).max(min_x)) as usize];
                let p1v = tmp0[(y * 64 + (x - 1).max(min_x)) as usize];
                let p2v = tmp0[(y * 64 + (x + 1).min(max_x)) as usize];
                let p3v = tmp0[(y * 64 + (x + 2).min(max_x)) as usize];
                let e1 = (x + 1 > max_x || x - 1 < min_x) as i32;
                let x0 = ((p2v - p1v) * 42 + (p3v - p0v) * -5) * (1 + e1);
                gx0[(y * 64 + x) as usize] = (x0 + 63 + (x0 > 0) as i32) >> 7;
                let q0 = tmp0[((y - 2).max(min_y) * 64 + x) as usize];
                let q1 = tmp0[((y - 1).max(min_y) * 64 + x) as usize];
                let q2 = tmp0[((y + 1).min(max_y) * 64 + x) as usize];
                let q3 = tmp0[((y + 2).min(max_y) * 64 + x) as usize];
                let e2 = (y + 1 > max_y || y - 1 < min_y) as i32;
                let y0 = ((q2 - q1) * 42 + (q3 - q0) * -5) * (1 + e2);
                gy0[(y * 64 + x) as usize] = (y0 + 63 + (y0 > 0) as i32) >> 7;
            }
        }
        bx += 16;
    }
    let mut out = Vec::with_capacity(16);
    let mut y = 0usize;
    while y < h {
        if !crate::av2_recon::work_tick("av2_inter:575") { break; }
        let mut x = 0usize;
        while x < w {
            if !crate::av2_recon::work_tick("av2_inter:577") { break; }
            let mut r = OpflReg { su2: (bs * bs) as i32, sv2: (bs * bs) as i32, ..Default::default() };
            for py in y..y + bs {
                if !crate::av2_recon::work_tick("av2_inter:584") { break; }
                for px in x..x + bs {
                    if !crate::av2_recon::work_tick("av2_inter:585") { break; }
                    let u = gx0[py * 64 + px];
                    let v = gy0[py * 64 + px];
                    let wv = tmp1[py * 64 + px];
                    r.su2 += u * u;
                    r.suv += u * v;
                    r.sv2 += v * v;
                    r.suw += u * wv;
                    r.svw += v * wv;
                }
            }
            out.push(r);
            x += bs;
        }
        y += bs;
    }
    out
}

#[inline]
fn ulog2_i(v: i32) -> i32 {
    31 - (v as u32).leading_zeros() as i32
}

/// dav2d `opfl_mv_adj` (recon_tmpl.c:1986): solve the 2x2 flow system -> per-arm 1/16-pel
/// deltas `dd = [(x0,y0),(x1,y1)]` (arm 0 negated), each clipped +-16. Zero on det<=0.
pub fn opfl_mv_adj(r: &OpflReg, d: (i32, i32)) -> [(i32, i32); 2] {
    let (mut su2, mut suv, mut sv2, mut suw, mut svw) = (r.su2, r.suv, r.sv2, r.suw, r.svw);
    let nb = |v: i32| 1 + ulog2_i(v.abs() + (v == 0) as i32);
    let nbits_max = (nb(su2) + nb(sv2))
        .max((nb(sv2) + nb(suw)).max(nb(suv) + nb(svw)))
        .max((nb(su2) + nb(svw)).max(nb(suv) + nb(suw)));
    let rbits = (nbits_max - 23).max(0) >> 1;
    if rbits > 0 {
        let rnd = (1 << rbits) >> 1;
        su2 = (su2 + rnd) >> rbits;
        sv2 = (sv2 + rnd) >> rbits;
        suv = (suv + rnd - (suv < 0) as i32) >> rbits;
        suw = (suw + rnd - (suw < 0) as i32) >> rbits;
        svw = (svw + rnd - (svw < 0) as i32) >> rbits;
    }
    let det = su2 * sv2 - suv * suv;
    if det <= 0 {
        return [(0, 0); 2];
    }
    let mut s = [sv2 * suw - suv * svw, su2 * svw - suv * suw];
    let (idet, shift) = crate::av2_warp::resolve_divisor_32(det as u32);
    let idet_bits = ulog2_i(idet);
    for si in s.iter_mut() {
        if !crate::av2_recon::work_tick("av2_inter:633") { break; }
        if *si == 0 {
            continue;
        }
        let mut abss = si.abs();
        let rb = (ulog2_i(abss) + idet_bits - 22).max(0);
        if rb > 0 {
            abss = (abss + ((1 << rb) >> 1)) >> rb;
        }
        let ibits = 3 + rb - shift;
        if ibits >= 0 {
            abss = abss.wrapping_mul(idet).wrapping_mul(1 << ibits);
        } else {
            abss = (abss.wrapping_mul(idet) + ((1 << -ibits) >> 1)) >> -ibits;
        }
        *si = if *si < 0 { -abss } else { abss };
    }
    [
        (-(d.0 * s[0]).clamp(-16, 16), -(d.0 * s[1]).clamp(-16, 16)),
        ((d.1 * s[0]).clamp(-16, 16), (d.1 * s[1]).clamp(-16, 16)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2_frame::Plane;

    // Ground-truth per-block translational-MC first rows dumped from dav2d (DINTER probe) for
    // frame-2 first-SB single-ref MM_SIMPLE blocks; reference = dav2d's frame-1 filtered output.
    #[test]
    fn mc_translate_matches_dav2d() {
        let path = &crate::av2_recon::cap_path("dav_filtered.yuv");
        let (w, h) = (432usize, 240usize);
        let bytes = match std::fs::read(path) {
            Ok(b) if b.len() >= w * h => b,
            _ => return, // reference not generated in this environment; skip
        };
        let mut rf = Plane::alloc(w, h);
        for i in 0..w * h {
            if !crate::av2_recon::work_tick("av2_inter:672") { break; }
            rf.px[i] = bytes[i] as i32;
        }
        // (bx, by, w, h, mvy, mvx, filter, expected first 8 pred pixels)
        let cases: &[(usize, usize, usize, usize, i32, i32, usize, [i32; 8])] = &[
            (8, 12, 32, 8, 0, 0, 0, [113, 113, 113, 113, 113, 113, 113, 113]), // zero MV = copy
            (8, 14, 16, 8, 0, 112, 0, [107, 106, 107, 107, 107, 107, 107, 107]), // integer mvx
            (12, 14, 16, 8, -32, 0, 0, [111, 111, 111, 111, 111, 111, 111, 111]), // integer mvy
            (2, 6, 8, 8, -8, 104, 0, [124, 124, 124, 124, 124, 124, 124, 124]), // subpel
            (0, 4, 8, 8, 17, 165, 0, [125, 125, 125, 125, 125, 124, 124, 123]), // 2D subpel
        ];
        for &(bx, by, bw, bh, mvy, mvx, filt, exp) in cases {
            let mut dst = vec![0i32; bw * bh];
            mc_translate_luma(&rf, &mut dst, bw, bx * 4, by * 4, bw, bh, mvy, mvx, filt, 255);
            assert_eq!(&dst[0..8], &exp, "block ({bx},{by}) mv=({mvy},{mvx})");
        }
        // Chroma (4:2:0, ss=1): frame-1 U plane at bytes [103680..], 216×120. Ground truth from
        // dav2d's real-time chroma prediction (DCPRED probe): (0,4) mv=(17,165) → 73; (8,12)
        // mv=(0,0) copy → [79,78,77,77].
        let (cw_, ch_) = (216usize, 120usize);
        if bytes.len() >= w * h + cw_ * ch_ {
            let mut cu = Plane::alloc(cw_, ch_);
            for i in 0..cw_ * ch_ {
                if !crate::av2_recon::work_tick("av2_inter:694") { break; }
                cu.px[i] = bytes[w * h + i] as i32;
            }
            // (cpx, cpy, cw, ch, mvy, mvx, expected first row)
            let ccases: &[(usize, usize, usize, usize, i32, i32, [i32; 4])] = &[
                (16, 24, 16, 4, 0, 0, [79, 78, 77, 77]),  // zero-MV chroma copy
                (0, 8, 4, 4, 17, 165, [73, 73, 73, 73]),  // 2D subpel chroma
            ];
            for &(cpx, cpy, cw, ch, mvy, mvx, exp) in ccases {
                let mut dst = vec![0i32; cw * ch];
                mc_translate(&cu, &mut dst, cw, cpx, cpy, cw, ch, mvy, mvx, 0, 1, 1, 255);
                assert_eq!(&dst[0..4], &exp, "chroma ({cpx},{cpy}) mv=({mvy},{mvx})");
            }
        }
    }
}
