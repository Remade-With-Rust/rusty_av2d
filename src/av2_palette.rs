//! AV2 palette / screen-content: neighbour palette caches, the luma-palette color parse,
//! and the color-index-map token decode. Exact ports of:
//!  - avm decodemv.c:1004 read_palette_colors_y / 1043 read_palette_mode_info (caller does symbols)
//!  - avm pred_common.c:611 palette_add_to_cache + 617 av2_get_palette_cache
//!  - avm entropymode.c:30 swap_color_order + 38 derive_color_index_ctx
//!  - avm detokenize.c:29 decode_color_map_tokens (+ decoder.h:799 av2_read_uniform)
//! Palette is LUMA-ONLY in AV2 (read_palette_mode_info reads no UV arm).

#![deny(clippy::indexing_slicing)]

use crate::msac::{
    rav1d_msac_decode_bool_bypass, rav1d_msac_decode_bools, rav1d_msac_decode_symbol_adapt4,
    rav1d_msac_decode_symbol_adapt8, MsacContext,
};

pub const PALETTE_MAX_SIZE: usize = 8;

#[derive(Clone, Default)]
pub struct PaletteBlock {
    pub n: usize,
    pub colors: [u16; PALETTE_MAX_SIZE],
    /// Color-index map at full plane-block dims (w*h, row-major), post extension.
    pub map: Vec<u8>,
    /// Plane block width/height in px (full block size, not edge-clipped).
    pub w: usize,
    pub h: usize,
}

thread_local! {
    /// Above-row / left-col palette caches (the row-cache equivalent of avm's
    /// xd->above_mbmi / left_mbmi palette_mode_info): per 4px column/row, the palette
    /// (n, colors) of the block whose bottom/right edge borders that cell. n==0 = none.
    static PAL_ABOVE: std::cell::RefCell<Vec<(u8, [u16; PALETTE_MAX_SIZE])>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static PAL_LEFT: std::cell::RefCell<Vec<(u8, [u16; PALETTE_MAX_SIZE])>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Reset both caches at frame (or tile) start.
pub fn pal_reset(iw4: usize, ih4: usize) {
    PAL_ABOVE.with(|c| *c.borrow_mut() = vec![(0u8, [0u16; PALETTE_MAX_SIZE]); iw4 + 32]);
    PAL_LEFT.with(|c| *c.borrow_mut() = vec![(0u8, [0u16; PALETTE_MAX_SIZE]); ih4 + 32]);
}

/// Splat a decoded LUMA leaf's palette (n=0 for non-palette leaves) over its footprint.
/// Call for EVERY luma leaf so stale entries can't leak sideways.
pub fn pal_splat(bx4: usize, by4: usize, bw4: usize, bh4: usize, n: usize, colors: &[u16; PALETTE_MAX_SIZE]) {
    PAL_ABOVE.with(|c| {
        let mut v = c.borrow_mut();
        for x in bx4..(bx4 + bw4).min(v.len()) {
            if let Some(c) = v.get_mut(x) { *c = (n as u8, *colors); }
        }
    });
    PAL_LEFT.with(|c| {
        let mut v = c.borrow_mut();
        for y in by4..(by4 + bh4).min(v.len()) {
            if let Some(c) = v.get_mut(y) { *c = (n as u8, *colors); }
        }
    });
}

/// avm av2_get_palette_cache (plane 0): merge the above/left neighbour palettes into a
/// sorted-merge cache. The ABOVE neighbour only participates when the block is NOT at a
/// 64px SB-row boundary (`row % (1<<MIN_SB_SIZE_LOG2)` — 64px even at sb128), i.e.
/// `by4 & 15 != 0`. avm's palette_add_to_cache does NOT dedup (despite its comment).
pub fn palette_cache(bx4: usize, by4: usize, have_left: bool, have_top: bool) -> ([u16; 2 * PALETTE_MAX_SIZE], usize) {
    let mut cache = [0u16; 2 * PALETTE_MAX_SIZE];
    let above = if have_top && (by4 & 15) != 0 {
        PAL_ABOVE.with(|c| c.borrow().get(bx4).copied().unwrap_or((0, [0; 8])))
    } else {
        (0, [0; 8])
    };
    let left = if have_left {
        PAL_LEFT.with(|c| c.borrow().get(by4).copied().unwrap_or((0, [0; 8])))
    } else {
        (0, [0; 8])
    };
    let (mut an, acol) = (above.0 as usize, above.1);
    let (mut ln, lcol) = (left.0 as usize, left.1);
    let mut n = 0usize;
    let (mut ai, mut li) = (0usize, 0usize);
    while an > 0 && ln > 0 {
        if let (Some(d), Some(&v)) = (cache.get_mut(n), acol.get(ai)) { *d = v; }
        n += 1; ai += 1; an -= 1;
        if let (Some(d), Some(&v)) = (cache.get_mut(n), lcol.get(li)) { *d = v; }
        n += 1; li += 1; ln -= 1;
    }
    while an > 0 {
        if let (Some(d), Some(&v)) = (cache.get_mut(n), acol.get(ai)) { *d = v; }
        n += 1; ai += 1; an -= 1;
    }
    while ln > 0 {
        if let (Some(d), Some(&v)) = (cache.get_mut(n), lcol.get(li)) { *d = v; }
        n += 1; li += 1; ln -= 1;
    }
    (cache, n)
}

/// avm decoder.h:799 av2_read_uniform: l = msb(n)+1 bits total, short codes first.
pub fn read_uniform(msac: &mut MsacContext, n: usize) -> usize {
    let l = (usize::BITS - n.leading_zeros()) as u8; // get_unsigned_bits(n)
    let m = (1usize << l) - n;
    let v = rav1d_msac_decode_bools(msac, l - 1) as usize;
    if v < m {
        v
    } else {
        (v << 1) - m + rav1d_msac_decode_bool_bypass(msac) as usize
    }
}

fn ceil_log2(x: i32) -> u8 {
    if x < 2 {
        0
    } else {
        (32 - (x - 1).leading_zeros()) as u8
    }
}

/// avm read_palette_colors_y (decodemv.c:1004), called AFTER palette_y_mode + size symbols.
/// `bd_bits` = bit depth (8/10). Returns the SORTED color list.
pub fn read_palette_colors_y(
    msac: &mut MsacContext,
    bd_bits: u8,
    n: usize,
    bx4: usize,
    by4: usize,
    have_left: bool,
    have_top: bool,
) -> [u16; PALETTE_MAX_SIZE] {
    let (cache, n_cache) = palette_cache(bx4, by4, have_left, have_top);
    let mut colors = [0u16; PALETTE_MAX_SIZE];
    let mut idx = 0usize;
    let mut i = 0usize;
    while i < n_cache && idx < n {
        if rav1d_msac_decode_bool_bypass(msac) {
            if let (Some(d), Some(&v)) = (colors.get_mut(idx), cache.get(i)) { *d = v; }
            idx += 1;
        }
        i += 1;
    }
    if idx < n {
        if let Some(d) = colors.get_mut(idx) { *d = rav1d_msac_decode_bools(msac, bd_bits) as u16; }
        idx += 1;
        if idx < n {
            let min_bits = bd_bits - 3;
            let mut bits = min_bits + rav1d_msac_decode_bools(msac, 2) as u8;
            let mut range = (1i32 << bd_bits) - colors.get(idx - 1).copied().unwrap_or(0) as i32 - 1;
            while idx < n {
                let delta = rav1d_msac_decode_bools(msac, bits) as i32 + 1;
                let prev = colors.get(idx - 1).copied().unwrap_or(0) as i32;
                let c = (prev + delta).clamp(0, (1 << bd_bits) - 1);
                range -= c - prev;
                if let Some(d) = colors.get_mut(idx) { *d = c as u16; }
                bits = bits.min(ceil_log2(range));
                idx += 1;
            }
        }
    }
    if let Some(sl) = colors.get_mut(..n) { sl.sort_unstable(); }
    colors
}

/// avm entropymode.c:38 derive_color_index_ctx: the (ctx, color_order) pair for map cell
/// (r,c) from its left / top-left / top neighbours. Returns (ctx, order[8]).
pub fn color_index_ctx(map: &[u8], stride: usize, r: usize, c: usize) -> (usize, [u8; PALETTE_MAX_SIZE]) {
    let mut order: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut status = [false; PALETTE_MAX_SIZE];
    let mut cnt = 0usize;
    let mut swap = |order: &mut [u8; 8], status: &mut [bool; 8], cnt: &mut usize, switch_idx: usize, max_idx: u8| {
        if let Some(d) = order.get_mut(switch_idx) { *d = max_idx; }
        if let Some(d) = status.get_mut(max_idx as usize) { *d = true; }
        *cnt += 1;
    };
    let ctx;
    if r > 0 && c > 0 {
        let g = |i: usize| map.get(i).copied().unwrap_or(0);
        let n0 = g(r * stride + c - 1); // left
        let n1 = g((r - 1) * stride + c - 1); // top-left
        let n2 = g((r - 1) * stride + c); // top
        if n0 == n1 && n0 == n2 {
            ctx = 4;
            swap(&mut order, &mut status, &mut cnt, 0, n0);
        } else if n0 == n2 {
            ctx = 3;
            swap(&mut order, &mut status, &mut cnt, 0, n0);
            swap(&mut order, &mut status, &mut cnt, 1, n1);
        } else if n0 == n1 {
            ctx = 2;
            swap(&mut order, &mut status, &mut cnt, 0, n0);
            swap(&mut order, &mut status, &mut cnt, 1, n2);
        } else if n1 == n2 {
            ctx = 2;
            swap(&mut order, &mut status, &mut cnt, 0, n2);
            swap(&mut order, &mut status, &mut cnt, 1, n0);
        } else {
            ctx = 1;
            swap(&mut order, &mut status, &mut cnt, 0, n0);
            swap(&mut order, &mut status, &mut cnt, 1, n2);
            swap(&mut order, &mut status, &mut cnt, 2, n1);
        }
    } else if c == 0 && r > 0 {
        ctx = 0;
        swap(&mut order, &mut status, &mut cnt, 0, map.get((r - 1) * stride + c).copied().unwrap_or(0));
    } else {
        // c > 0 && r == 0 (the (0,0) cell never reaches here)
        ctx = 0;
        swap(&mut order, &mut status, &mut cnt, 0, map.get(r * stride + c - 1).copied().unwrap_or(0));
    }
    let mut write_idx = cnt;
    for read_idx in 0..PALETTE_MAX_SIZE {
        if !status.get(read_idx).copied().unwrap_or(true) {
            if let Some(d) = order.get_mut(write_idx) { *d = read_idx as u8; }
            write_idx += 1;
        }
    }
    (ctx, order)
}

/// avm detokenize.c:29 decode_color_map_tokens. `pw`/`ph` = plane block dims (px, full);
/// `rows`/`cols` = within-frame-bounds dims. Reads the transverse direction bit (iff
/// pw<64 && ph<64), per-line identity flags, and the color symbols; extends the map to
/// the full pw×ph. `cdf_identity` = identity_row_cdf_y [4 ctx][3 sym];
/// `cdf_map` = palette_y_color_index_cdf [n-2][5 ctx][n sym].
#[allow(clippy::too_many_arguments)]
pub fn decode_color_map(
    msac: &mut MsacContext,
    cdf_identity: &mut [[u16; 4]; 4],
    cdf_map: &mut [[[u16; 8]; 5]; 7],
    n_colors: usize,
    pw: usize,
    ph: usize,
    rows: usize,
    cols: usize,
) -> Option<Vec<u8>> {
    let mut map = vec![0u8; pw * ph];
    let transverse_allowed = pw < 64 && ph < 64;
    let direction = if transverse_allowed { rav1d_msac_decode_bool_bypass(msac) as usize } else { 0 };
    let axis1_limit = if direction != 0 { rows } else { cols };
    let axis2_limit = if direction != 0 { cols } else { rows };
    let mut prev_identity = 0usize;
    for ax2 in 0..axis2_limit {
        let ctx = if ax2 == 0 { 3 } else { prev_identity };
        let ctx = ctx.min(cdf_identity.len() - 1);
        let flag = match cdf_identity.get_mut(ctx) {
            Some(c) => rav1d_msac_decode_symbol_adapt4(msac, c, 2) as usize,
            None => return None,
        };
        if flag == 2 && ax2 == 0 {
            return None; // copy-prev-line illegal on the first line
        }
        for ax1 in 0..axis1_limit {
            let (y, x) = if direction != 0 { (ax1, ax2) } else { (ax2, ax1) };
            let dst = y * pw + x;
            let g = |m: &Vec<u8>, i: usize| m.get(i).copied().unwrap_or(0);
            let v = if flag == 2 {
                if direction != 0 { g(&map, y * pw + x.wrapping_sub(1)) } else { g(&map, (y.wrapping_sub(1)) * pw + x) }
            } else if flag == 1 && ax1 > 0 {
                if direction != 0 { g(&map, (y.wrapping_sub(1)) * pw + x) } else { g(&map, y * pw + x.wrapping_sub(1)) }
            } else if ax2 == 0 && ax1 == 0 {
                read_uniform(msac, n_colors) as u8
            } else {
                let (cctx, order) = color_index_ctx(&map, pw, y, x);
                let row = match cdf_map.get_mut(n_colors.wrapping_sub(2)).and_then(|r| r.get_mut(cctx)) {
                    Some(r) => r,
                    None => return None,
                };
                let idx = rav1d_msac_decode_symbol_adapt8(msac, row, (n_colors - 1) as u8) as usize;
                order.get(idx).copied().unwrap_or(0)
            };
            if let Some(d) = map.get_mut(dst) { *d = v; }
        }
        prev_identity = flag;
    }
    // Extend the last coded column / row over the off-frame remainder.
    if cols < pw {
        for r in 0..rows {
            let v = map.get(r * pw + cols - 1).copied().unwrap_or(0);
            for c in cols..pw {
                if let Some(d) = map.get_mut(r * pw + c) { *d = v; }
            }
        }
    }
    for r in rows..ph {
        let (a, b) = map.split_at_mut(r * pw);
        b[..pw].copy_from_slice(&a[(rows - 1) * pw..(rows - 1) * pw + pw]);
    }
    Some(map)
}
