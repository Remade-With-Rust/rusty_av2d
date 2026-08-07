//! AV2 MV reference-list scan (`refmvs_find`) — brick **B**.
//!
//! Produces the DRL candidate list (`mvstack`) whose `[drl_idx]` entry is the MV *predictor*
//! that the decoded residual is added to (NEWMV) or that is used directly (NEARMV). In AV2 the
//! DRL/inter-mode *parse* contexts are neighbour + loop-index based (NOT mvstack-based — an AV2
//! simplification vs AV1), so this is a RECON-side quantity: the final MV, not any entropy ctx.
//!
//! Verification target (dav2d decode.c:1103-1116): `final = reduce_prec(mvstack[drl].mv, prec)
//! + residual` (NEWMV) or `= mvstack[drl].mv` (NEARMV). The oracle dumps the stack + final MV via
//! the `BMVSTK`/`BMVCAND` probes (decode.c ~1118, gated to the first SB).
//!
//! This module currently carries the VERIFIED primitives (`mv_reduce_prec`, `get_gmv_2d`) and the
//! grid types; the full spatial scan (`add_spatial_candidate` with warp projection, weight
//! accumulation, dedup, sort, extended scan) is the next brick — see [[rav2d-frame2-inter]].

use std::cell::RefCell;

thread_local! {
    /// The scratch frame-2 refmvs grid + bank, populated as the inter recursion decodes blocks.
    /// Reset via `reset_refmvs()` at the start of the frame-2 SB loop.
    pub static GRID: RefCell<RefmvsGrid> = RefCell::new(RefmvsGrid::default());
    pub static BANK: RefCell<RefmvBank> = RefCell::new(RefmvBank::default());
    pub static WARPBANK: RefCell<RefmvWarpBank> = RefCell::new(RefmvWarpBank::default());
}

/// Reset the thread-local refmvs grid + banks (call before the frame-2 SB loop).
pub fn reset_refmvs() {
    GRID.with(|g| *g.borrow_mut() = RefmvsGrid::default());
    BANK.with(|b| *b.borrow_mut() = RefmvBank::default());
    WARPBANK.with(|w| *w.borrow_mut() = RefmvWarpBank::default());
}

/// dav2d `dav2d_refmvs_reset_sb` (refmvs.c:1316): call ONCE at each SB start, before decoding any
/// of its blocks. Zeroes the bank/warp hit counters, then — except on the first SB row (or a
/// key/intra/TIP frame) — SEEDS the refmv + warp banks from up to 4 blocks of the committed
/// above-SB-row. The above-row is `ra[]` in dav (an 8x8-resolution snapshot); mine reads it
/// straight from the grid ring at `(by4-1)` at even columns (`ra[k] == grid[by4-1][2k]`), which
/// is intact single-threaded (the new SB row writes rows `by4..`, never `by4-1`).
pub fn reset_sb(bx4: usize, by4: usize, sbsz: usize, iw4: usize, first_sb_row: bool) {
    BANK.with(|bk| {
        let mut b = bk.borrow_mut();
        b.hits0 = 0;
        b.hits1 = 0;
        b.avail = 0;
    });
    WARPBANK.with(|wb| wb.borrow_mut().hits = 0);
    if first_sb_row {
        return;
    }
    let end_x4 = (bx4 + sbsz).min(iw4);
    GRID.with(|g| {
        let grid = g.borrow();
        BANK.with(|bk| {
            let mut bank = bk.borrow_mut();
            WARPBANK.with(|wb| {
                let mut warp = wb.borrow_mut();
                let mut hits = 0;
                let mut x = bx4;
                while x < end_x4 {
                    // ra[x>>1] == grid[above][2*(x>>1)] == grid[above][x & !1]
                    let r = *grid.at(by4 - 1, x & !1);
                    let sz4 = (crate::av2_decode::BLOCK_DIMENSIONS[r.bs as usize][0] as usize).max(1);
                    if r.mv[0].y != -0x8000 {
                        // warp blocks seed with the base 2D MV (r.lmv), else the stored MV.
                        // Compound blocks seed their PAIR into the compound bank classes.
                        let seed = if r.mf & 2 != 0 { r.lmv } else { r.mv };
                        bank.add_raw_pair(r.ref_[0], r.ref_[1], seed, (r.mf >> 2) as i8);
                        if r.mf & 2 != 0 {
                            warp.add(r.ref_[0], r.matrix);
                        }
                        hits += 1;
                        if hits == 4 {
                            break;
                        }
                    }
                    x += sz4;
                }
            });
        });
    });
}

/// dav2d `dav2d_refmvs_tile_sbrow_init` (refmvs.c:1106-1109): at each SB-ROW start, reset the
/// refmv + warp bank SIZE/IDX to 0 (the mv/matrix slots persist but become logically empty, since
/// reads are bounded by size). Mine previously kept size/idx across SB rows, so rows 1+ read a
/// bank full of stale row-above entries (warp bank size=4 vs dav's freshly-0). Call ONCE per SB
/// row, before that row's per-SB `reset_sb` seeding.
pub fn reset_sbrow() {
    BANK.with(|bk| {
        let mut b = bk.borrow_mut();
        b.size = [0; 9];
        b.idx = [0; 9];
    });
    WARPBANK.with(|wb| {
        let mut w = wb.borrow_mut();
        w.size = [0; 7];
        w.idx = [0; 7];
    });
}

/// A motion vector in 1/8-pel units (dav2d `union mv`), y then x.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Mv {
    pub y: i32,
    pub x: i32,
}

/// One 4px block's stored MV/ref/size/flags in the refmvs grid (dav2d `refmvs_block`).
/// `mf` = motion flags: bit0 = uses global MV, bit1 = warp (mv[] is a projected sample), bits2+ =
/// compound weight. `bs` is the block-size index (for warp projection extents).
#[derive(Clone, Copy, Debug)]
pub struct RefmvsBlock {
    pub mv: [Mv; 2],
    pub ref_: [i8; 2], // ref frame indices (-1 = none)
    pub bs: u8,
    pub mf: u8, // bit0: globalmv, bit1: warp (matrix valid), bits2-7: cwp_idx
    pub bx4: u16,
    pub by4: u16,
    pub lmv: [Mv; 2], // 2dmv for warp blocks (the block MV; `derive_warpmv`'s add_sample reads it)
    pub matrix: [i32; 6], // warp model (valid when mf & 2)
}

impl Default for RefmvsBlock {
    fn default() -> Self {
        // INVALID_MV sentinel (dav2d): mv.y == -0x8000 marks "not yet coded".
        RefmvsBlock {
            mv: [Mv { y: -0x8000, x: -0x8000 }; 2],
            ref_: [-1, -1],
            bs: 0,
            mf: 0,
            bx4: 0,
            by4: 0,
            lmv: [Mv { y: -0x8000, x: -0x8000 }; 2],
            matrix: [0; 6],
        }
    }
}

/// dav2d `get_warpmv_proj` (refmvs.c:526): project a warp neighbour's model to a sample point
/// `(x, y)` (in 1px units), yielding the MV that block would place there. This is why a warp
/// neighbour contributes a *projected* MV to the candidate stack, not its stored block MV.
#[allow(clippy::too_many_arguments)]
pub fn get_warpmv_proj(m: &[i32; 6], x: i32, y: i32, minx: i32, maxx: i32, miny: i32, maxy: i32) -> Mv {
    let xc = (m[2] - (1 << 16)) * x + m[3] * y + m[0];
    let yc = (m[5] - (1 << 16)) * y + m[4] * x + m[1];
    let ry = (((yc + 0x1000 - (yc < 0) as i32) >> 13).clamp(-0xffff, 0xffff)).clamp(miny, maxy);
    let rx = (((xc + 0x1000 - (xc < 0) as i32) >> 13).clamp(-0xffff, 0xffff)).clamp(minx, maxx);
    Mv { y: ry, x: rx }
}

/// dav2d `mv_reduce_prec` (env.h:353): round an MV to a coarser precision grid. `mv_prec` 6 =
/// no-op (1/8-pel); lower precisions round toward zero to the `32 >> mv_prec` grid.
pub fn mv_reduce_prec(m: &mut Mv, mv_prec: i32) {
    if mv_prec == 6 {
        return;
    }
    let rnd = 32 >> mv_prec;
    m.x = m.x + rnd - (m.x > 0) as i32;
    m.y = m.y + rnd - (m.y > 0) as i32;
    let mask = !(rnd * 2 - 1);
    m.x &= mask;
    m.y &= mask;
}

/// dav2d warp-motion type (env.h `warp_type` / `get_gmv_2d`). This clip's global motion is
/// IDENTITY for all refs (verified: block (0,0)'s only candidate is (0,0)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WarpType {
    Identity,
    Translation,
    RotZoom,
    Affine,
}

/// The frame's global-motion model for one ref (dav2d `Dav2dWarpedMotionParams`).
#[derive(Clone, Copy, Debug)]
pub struct GmvModel {
    pub ty: WarpType,
    pub matrix: [i32; 6],
}

impl Default for GmvModel {
    fn default() -> Self {
        GmvModel { ty: WarpType::Identity, matrix: [0; 6] }
    }
}

/// dav2d `get_gmv_2d` (env.h:388), TRANSLATION + IDENTITY paths (the ROT_ZOOM/AFFINE warp path
/// is deferred — this clip is IDENTITY). Returns the frame's global MV at this block.
pub fn get_gmv_2d(g: &GmvModel, bx4: i32, by4: i32, bw4: i32, bh4: i32, iw4: i32, ih4: i32) -> Mv {
    match g.ty {
        WarpType::Identity => Mv { y: 0, x: 0 },
        WarpType::Translation => {
            let mut y = g.matrix[0] >> 13;
            let mut x = g.matrix[1] >> 13;
            y = y.clamp(-(by4 + bh4 + 4) * 32, (ih4 - by4 + 4) * 32);
            x = x.clamp(-(bx4 + bw4 + 4) * 32, (iw4 - bx4 + 4) * 32);
            Mv { y, x }
        }
        // ROT_ZOOM/AFFINE (get_warpmv_2d) not needed for this clip; add when a stream uses it.
        _ => Mv { y: 0, x: 0 },
    }
}

/// One DRL candidate (dav2d `refmvs_candidate`): the MV pair, its accumulated weight, and the
/// spatial offset it came from (used later for the OBMC/warp sub-block predictor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub mv: [Mv; 2],
    pub weight: i32,
    pub y_off: i32,
    pub x_off: i32,
    /// Compound-weighted-prediction index carried per candidate (dav2d refmvs_candidate.cwp_idx);
    /// 8 = equal weights. Only meaningful for compound stacks.
    pub cwp: i8,
}

impl Default for Candidate {
    fn default() -> Self {
        Candidate { mv: [Mv::default(); 2], weight: 0, y_off: 0, x_off: 0, cwp: 8 }
    }
}

/// dav2d `add_candidate_comp` (refmvs.c:136): dedup a COMPOUND candidate on BOTH mvs
/// (accumulating weight) or append it with its cwp. Returns true iff appended.
#[allow(clippy::too_many_arguments)]
pub fn add_candidate_comp(
    mvstack: &mut [Candidate],
    cnt: &mut usize,
    max_cnt: usize,
    weight: i32,
    cwp: i8,
    cand_mv: [Mv; 2],
    iter_cntr: &mut i32,
    max_iter: i32,
) -> bool {
    let last = *cnt;
    if *iter_cntr < max_iter {
        for n in 0..last {
            if !crate::av2_recon::work_tick("refmvs:239") { break; }
            if mvstack[n].mv[0] == cand_mv[0] && mvstack[n].mv[1] == cand_mv[1] {
                *iter_cntr += n as i32 + 1;
                mvstack[n].weight += weight;
                return false;
            }
        }
        *iter_cntr += last as i32;
    }
    if last >= max_cnt {
        return false;
    }
    mvstack[last].mv = cand_mv;
    mvstack[last].weight = weight;
    mvstack[last].cwp = cwp;
    mvstack[last].y_off = 0;
    mvstack[last].x_off = 0;
    *cnt = last + 1;
    true
}

/// dav2d `add_candidate_sngl` (refmvs.c): dedup a single-ref candidate into the stack by MV
/// equality (accumulating its weight) or append it (up to `max_cnt`). `iter_cntr` bounds the
/// dedup scan (dav2d caps the total comparisons at `max_iter` to keep the scan O(1)-amortised).
/// Returns true iff a new entry was appended.
#[allow(clippy::too_many_arguments)]
pub fn add_candidate_sngl(
    mvstack: &mut [Candidate],
    cnt: &mut usize,
    max_cnt: usize,
    weight: i32,
    cand_mv: Mv,
    y_off_s: i32,
    x_off_s: i32,
    iter_cntr: &mut i32,
    max_iter: i32,
) -> bool {
    let last = *cnt;
    if *iter_cntr < max_iter {
        for m in 0..last {
            if !crate::av2_recon::work_tick("refmvs:278") { break; }
            if mvstack[m].mv[0] == cand_mv {
                *iter_cntr += m as i32 + 1;
                mvstack[m].weight += weight;
                return false;
            }
        }
        *iter_cntr += last as i32;
    }
    if last >= max_cnt {
        return false;
    }
    mvstack[last].mv[0] = cand_mv;
    mvstack[last].weight = weight;
    mvstack[last].y_off = y_off_s;
    mvstack[last].x_off = x_off_s;
    *cnt = last + 1;
    true
}

/// dav2d `dav2d_set_affine_mv2d` (warpmv.c:104): compute the warp matrix translation part
/// (`m[0]`,`m[1]`) from the block's final MV so the warp model is anchored at the block. Verified
/// by hand vs block (0,0): mv (24,168) → m[0]=1476608, m[1]=182272.
pub fn set_affine_mv2d(bw4: usize, bh4: usize, mv: Mv, m: &mut [i32; 6], bx4: usize, by4: usize) {
    let rsuy = 2 * bh4 as i64 - 1;
    let rsux = 2 * bw4 as i64 - 1;
    let isuy = by4 as i64 * 4 + rsuy;
    let isux = bx4 as i64 * 4 + rsux;
    m[0] = (mv.x as i64 * 0x2000 - isux * (m[2] as i64 - 0x10000) - isuy * m[3] as i64).clamp(-0x8000000, 0x7ffffc0) as i32;
    m[1] = (mv.y as i64 * 0x2000 - isux * m[4] as i64 - isuy * (m[5] as i64 - 0x10000)).clamp(-0x8000000, 0x7ffffc0) as i32;
}

/// Reconstruct the MM_WARP_DELTA warp matrix (dav2d decode.c:1118-1142): apply the decoded
/// `delta` (`b->matrix`, signed×step; `[2]==-0x80` marks np==2) to the base warp model
/// `warp_base` (= warp[warp_ref_idx][2..6]; identity/gmv for a no-warp-neighbour wri==0 block),
/// then anchor the translation via `set_affine_mv2d` with the block's final `mv`.
pub fn reconstruct_warp_delta_matrix(warp_base: [i32; 6], delta: [i32; 4], mv: Mv, bw4: usize, bh4: usize, bx4: usize, by4: usize) -> [i32; 6] {
    let mut m = warp_base;
    let mut n = 0usize;
    while n < 4 && delta[n] != -0x80 {
        if delta[n] != 0 {
            let base: i32 = if (n as u32).wrapping_sub(1) >= 2 { 0x10000 } else { 0 };
            m[2 + n] = (warp_base[2 + n] + delta[n] * (1 << 10)).clamp(base - 0x7fc0, base + 0x7fc0);
        } else {
            m[2 + n] = warp_base[2 + n];
        }
        n += 1;
    }
    if delta[2] == -0x80 {
        m[5] = m[2];
        m[4] = -m[3];
    }
    set_affine_mv2d(bw4, bh4, mv, &mut m, bx4, by4);
    m
}

/// dav2d `resolve_divisor_64` (warpmv.c): fixed-point reciprocal of `d` via the DIV_RECIP table.
fn resolve_divisor_64(d: u64) -> (i32, i32) {
    let mut shift = 63 - d.leading_zeros() as i32; // u64log2
    let e = d as i64 - (1i64 << shift);
    let f = if shift > 7 { (e + (1i64 << (shift - 8))) >> (shift - 7) } else { e << (7 - shift) };
    shift += 9;
    (crate::av2_ipred::DIV_RECIP[f as usize] as i32, shift)
}

fn get_mult_shift_ndiag(px: i64, idet: i64, rnd: i64, sh: i32) -> i32 {
    let v1 = px * idet;
    let v2 = ((v1 + rnd - (v1 < 0) as i64) >> sh) as i32;
    let v3 = (v2 + 0x20 - (v2 < 0) as i32) & !0x3f;
    v3.clamp(-0x7fc0, 0x7fc0)
}

fn get_mult_shift_diag(px: i64, idet: i64, rnd: i64, sh: i32) -> i32 {
    let v1 = px * idet;
    let v2 = ((v1 + rnd - (v1 < 0) as i64) >> sh) as i32;
    let v3 = (v2 + 0x20 - (v2 < 0x10000) as i32) & !0x3f;
    v3.clamp(0x8040, 0x17fc0)
}

/// dav2d `dav2d_find_affine_int` (warpmv.c:120): least-squares affine warp fit from `np` point
/// correspondences `pts[i] = [[in_x,in_y],[out_x,out_y]]`, anchored by the block MV via
/// set_affine. This is the MM_WARP_CAUSAL model. Verified vs (6,4)'s oracle pts.
pub fn find_affine_int(pts: &[[[i32; 2]; 2]], bw4: usize, bh4: usize, mv: Mv, bx4: usize, by4: usize) -> [i32; 6] {
    let (mut a00, mut a01, mut a11) = (0i64, 0i64, 0i64);
    let (mut bx0, mut bx1, mut by0, mut by1) = (0i64, 0i64, 0i64, 0i64);
    let rsuy = 2 * bh4 as i32 - 1;
    let rsux = 2 * bw4 as i32 - 1;
    let (suy, sux) = (rsuy * 8, rsux * 8);
    let (duy, dux) = (suy + mv.y, sux + mv.x);
    for p in pts {
        if !crate::av2_recon::work_tick("refmvs:367") { break; }
        let dx = (p[1][0] - dux) as i64;
        let dy = (p[1][1] - duy) as i64;
        let sx = (p[0][0] - sux) as i64;
        let sy = (p[0][1] - suy) as i64;
        if (sx - dx).abs() < 256 && (sy - dy).abs() < 256 {
            a00 += ((sx * sx) >> 2) + sx * 2 + 8;
            a01 += ((sx * sy) >> 2) + sx + sy + 4;
            a11 += ((sy * sy) >> 2) + sy * 2 + 8;
            bx0 += ((sx * dx) >> 2) + sx + dx + 8;
            bx1 += ((sy * dx) >> 2) + sy + dx + 4;
            by0 += ((sx * dy) >> 2) + sx + dy + 4;
            by1 += ((sy * dy) >> 2) + sy + dy + 8;
        }
    }
    let mut m = [0i32; 6];
    let det = a00 * a11 - a01 * a01;
    if det == 0 {
        m[2] = 0x10000;
        m[5] = 0x10000;
        set_affine_mv2d(bw4, bh4, mv, &mut m, bx4, by4);
        return m;
    }
    let (recip, mut shift) = resolve_divisor_64(det.unsigned_abs());
    let mut idet = (if det < 0 { -recip } else { recip }) as i64;
    shift -= 16;
    if shift < 0 {
        idet <<= -shift;
        shift = 0;
    }
    let r = (1i64 << shift) >> 1;
    m[2] = get_mult_shift_diag(a11 * bx0 - a01 * bx1, idet, r, shift);
    m[3] = get_mult_shift_ndiag(a00 * bx1 - a01 * bx0, idet, r, shift);
    m[4] = get_mult_shift_ndiag(a11 * by0 - a01 * by1, idet, r, shift);
    m[5] = get_mult_shift_diag(a00 * by1 - a01 * by0, idet, r, shift);
    set_affine_mv2d(bw4, bh4, mv, &mut m, bx4, by4);
    m
}

/// dav2d `derive_warpmv` (decode.c:238): the MM_WARP_CAUSAL warp model — gather neighbour point
/// correspondences (top walk by block-width, left walk by block-height, top-left/top-right) and
/// fit them with `find_affine_int`. Returns the matrix (caller validates via get_shear_params).
/// SB-boundary above-row (`ra[]`) top samples are deferred (not-sb-boundary path only for now).
#[allow(clippy::too_many_arguments)]
pub fn derive_warpmv(grid: &RefmvsGrid, bx4: usize, by4: usize, bw4: usize, bh4: usize, w4: usize, h4: usize, ref0: i8, mv: Mv, sbsz: usize, col_end: usize) -> Option<[i32; 6]> {
    use crate::av2_decode::BLOCK_DIMENSIONS;
    let mut pts = [[[0i32; 2]; 2]; 8];
    let mut np = 0usize;
    let is_not_sb_boundary = (by4 & (sbsz - 1)) != 0;
    let have_top = by4 > 0;
    let have_left = bx4 > 0;
    let mut have_topleft = false;
    let mut have_topright = false;

    // dav add_sample (decode.c:249): scan BOTH ref slots — a compound neighbour matching on
    // slot n contributes THAT slot's mv (lmv for warp neighbours). Up to 2 samples per neighbour.
    let mut add_sample = |pts: &mut [[[i32; 2]; 2]; 8], np: &mut usize, dx: i32, dy: i32, sx: i32, sy: i32, rp: &RefmvsBlock| {
        let bd = BLOCK_DIMENSIONS[rp.bs as usize];
        let rmv = if rp.mf & 2 != 0 { rp.lmv } else { rp.mv };
        for n in 0..2usize {
            if !crate::av2_recon::work_tick("refmvs:426") { break; }
            if *np >= 8 {
                return;
            }
            let matches = if n == 0 { rp.ref_[0] == ref0 } else { rp.ref_[1] == ref0 };
            if !matches {
                continue;
            }
            let ix = 16 * (2 * dx + sx * bd[0] as i32) - 8;
            let iy = 16 * (2 * dy + sy * bd[1] as i32) - 8;
            pts[*np] = [[ix, iy], [ix + rmv[n].x, iy + rmv[n].y]];
            *np += 1;
        }
    };

    if have_top {
        if is_not_sb_boundary {
            // top walk: step across the top edge block-by-block (full-res current-SB row).
            let first = *grid.at(by4 - 1, bx4);
            let mut off = first.bx4 as i32 - bx4 as i32;
            have_topleft = off == 0;
            loop {
                if !crate::av2_recon::work_tick("refmvs_loop:451") { break; }
                let cell = *grid.at(by4 - 1, (bx4 as i32 + off) as usize);
                add_sample(&mut pts, &mut np, off, 0, 1, -1, &cell);
                // HARDENING: the scan advances by the NEIGHBOUR's block dimension read from the
                // grid. A corrupt/degenerate cell can report 0 — the walk then never
                // advances (the fuzz HANG). Advance at least one 4px unit.
                off += (BLOCK_DIMENSIONS[cell.bs as usize][0] as i32).max(1);
                if !(off < w4 as i32 && np < 8) {
                    break;
                }
            }
            have_topright = off <= bw4 as i32;
        } else {
            // SB-boundary top walk over the committed above-row `ra[]` (dav2d decode.c:316-341):
            // ra[k] == grid[by4-1][2k]; index at even columns, `ioff` re-derives the block origin.
            have_topleft = true;
            let r2 = *grid.at(by4 - 1, (bx4 >> 1) * 2);
            let mut off = (r2.bx4 as i32 + BLOCK_DIMENSIONS[r2.bs as usize][0] as i32 <= bx4 as i32) as i32;
            let tr_ext = ((bx4 + bw4) & (sbsz - 1) != 0
                && bx4 + bw4 < col_end
                && (grid.at(by4 - 1, ((bx4 + bw4) >> 1) * 2).bx4 as i32) < (bx4 + bw4) as i32) as i32;
            loop {
                if !crate::av2_recon::work_tick("refmvs_loop:469") { break; }
                let off8 = (bx4 as i32 + off) >> 1;
                let cell = *grid.at(by4 - 1, (off8 * 2) as usize);
                let ioff = cell.bx4 as i32 - bx4 as i32;
                add_sample(&mut pts, &mut np, ioff, 0, 1, -1, &cell);
                off = ioff + BLOCK_DIMENSIONS[cell.bs as usize][0] as i32 + 1;
                if !(off < w4 as i32 + tr_ext && np < 8) {
                    break;
                }
            }
            have_topright = true;
        }
        have_topright &= bw4 <= 16
            && bx4 + bw4 + ((!is_not_sb_boundary) as usize) < col_end
            && ((by4 & (sbsz - 1)) == 0 || ((bx4 + bw4) & (sbsz - 1) != 0 && grid.at(by4 - 1, bx4 + bw4).mv[0].y != -0x8000));
    }
    if np < 8 && have_left {
        // left walk: step down the left edge block-by-block (their stored by4 gives the origin).
        let first = *grid.at(by4, bx4 - 1);
        let mut off = first.by4 as i32 - by4 as i32;
        have_topleft &= off == 0;
        loop {
            if !crate::av2_recon::work_tick("refmvs_loop:490") { break; }
            let cell = *grid.at((by4 as i32 + off) as usize, bx4 - 1);
            add_sample(&mut pts, &mut np, 0, off, -1, 1, &cell);
            // HARDENING: the scan advances by the NEIGHBOUR's block dimension read from the
                // grid. A corrupt/degenerate cell can report 0 — the walk then never
                // advances (the fuzz HANG). Advance at least one 4px unit.
                off += (BLOCK_DIMENSIONS[cell.bs as usize][1] as i32).max(1);
            if !(off < h4 as i32 && np < 8) {
                break;
            }
        }
    } else {
        have_topleft = false;
    }
    if is_not_sb_boundary {
        if np < 8 && have_topleft {
            let c = *grid.at(by4 - 1, bx4 - 1);
            add_sample(&mut pts, &mut np, 0, 0, -1, -1, &c);
        }
        if np < 8 && have_topright {
            let c = *grid.at(by4 - 1, bx4 + bw4);
            add_sample(&mut pts, &mut np, bw4 as i32, 0, 1, -1, &c);
        }
    } else {
        // SB-boundary top/left + top/right corner samples from the above-row `ra[]` (dav2d
        // decode.c:366-376), each gated on the ra[] block ending/starting exactly at our edge.
        // ra[(bx-1)>>1] (== ra_tl at a SB-column boundary) coincides with this grid cell this row.
        if np < 8 && have_topleft {
            let c = *grid.at(by4 - 1, (((bx4 as i32 - 1) >> 1) * 2) as usize);
            if BLOCK_DIMENSIONS[c.bs as usize][0] as i32 + c.bx4 as i32 == bx4 as i32 {
                add_sample(&mut pts, &mut np, 0, 0, -1, -1, &c);
            }
        }
        if np < 8 && have_topright {
            let c = *grid.at(by4 - 1, ((bx4 + bw4 + 1) >> 1) * 2);
            if c.bx4 as i32 == (bx4 + bw4) as i32 {
                add_sample(&mut pts, &mut np, bw4 as i32, 0, 1, -1, &c);
            }
        }
    }
    if np == 0 {
        return None;
    }
    Some(find_affine_int(&pts[..np], bw4, bh4, mv, bx4, by4))
}

/// dav2d `get_warpmv_2d` (env.h:363): the WARPMV block predictor — evaluate the warp `matrix`
/// at the block CENTRE (`mv_precision` 6 = 1/8-pel; <6 rounds coarser). Distinct from the
/// WARPNEWMV predictor (which uses the mvstack). Verified vs (2,4)'s oracle final MV (21,158).
#[allow(clippy::too_many_arguments)]
pub fn get_warpmv_2d(m: &[i32; 6], bx4: i32, by4: i32, bw4: i32, bh4: i32, iw4: i32, ih4: i32, mv_precision: i32) -> Mv {
    let x = (bx4 * 4 + bw4 * 2 - 1) as i64;
    let y = (by4 * 4 + bh4 * 2 - 1) as i64;
    let xc = (m[2] as i64 - (1 << 16)) * x + m[3] as i64 * y + m[0] as i64;
    let yc = (m[5] as i64 - (1 << 16)) * y + m[4] as i64 * x + m[1] as i64;
    let not_epel = (mv_precision < 6) as i64;
    let shift = 13 + not_epel;
    let rnd = (1i64 << shift) >> 1;
    let max = 0xffff - not_epel as i32;
    let sgn = |v: i64, s: i64| if s < 0 { -v } else { v };
    let mut ry = (sgn(((yc.abs() + rnd) >> shift) << not_epel, yc)).clamp(-max as i64, max as i64) as i32;
    let mut rx = (sgn(((xc.abs() + rnd) >> shift) << not_epel, xc)).clamp(-max as i64, max as i64) as i32;
    ry = ry.clamp(-(by4 + bh4 + 4) * 32, (ih4 - by4 + 4) * 32);
    rx = rx.clamp(-(bx4 + bw4 + 4) * 32, (iw4 - bx4 + 4) * 32);
    Mv { y: ry, x: rx }
}

/// dav2d `model_from_corners` (refmvs.c:479): build an affine warp model directly from three
/// corner MVs (top-left, top-right, bottom-left). Returns `Some(matrix)` or `None` if degenerate /
/// off-frame. `b_dim` = block_dimensions[bs] (bw4, bh4, w_log2, h_log2); `(xpos,ypos)` = block
/// origin in 1px units. This is the FIRST warp candidate (warp[0]) for a warp block with neighbours.
pub fn model_from_corners(tl: Mv, tr: Mv, bl: Mv, xpos: i32, ypos: i32, b_dim: [u8; 4]) -> Option<[i32; 6]> {
    if tr == tl && bl == tl {
        return None;
    }
    if tl.x.min(bl.x).min(tr.x + b_dim[0] as i32 * 32) < -xpos * 8 {
        return None;
    }
    if tl.y.min(tr.y).min(bl.y + b_dim[1] as i32 * 32) < -ypos * 8 {
        return None;
    }
    let clip32 = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let clip_m01 = |v: i64| v.clamp(-0x8000000, 0x7ffffc0) as i32;
    let mut m = [0i32; 6];
    m[2] = clip32(((tr.x - tl.x) as i64 * (1 << 11)) >> b_dim[2]);
    m[4] = clip32(((tr.y - tl.y) as i64 * (1 << 11)) >> b_dim[2]);
    m[3] = clip32(((bl.x - tl.x) as i64 * (1 << 11)) >> b_dim[3]);
    m[5] = clip32(((bl.y - tl.y) as i64 * (1 << 11)) >> b_dim[3]);
    m[0] = clip_m01(tl.x as i64 * (1 << 13) - xpos as i64 * m[2] as i64 - ypos as i64 * m[3] as i64);
    m[1] = clip_m01(tl.y as i64 * (1 << 13) - xpos as i64 * m[4] as i64 - ypos as i64 * m[5] as i64);
    for i in 2..6 {
        if !crate::av2_recon::work_tick("refmvs:573") { break; }
        m[i] = m[i].clamp(-0x7fc0, 0x7fc0);
        m[i] += 0x20 - (m[i] < 0) as i32;
        m[i] &= !0x3f;
    }
    m[2] += 0x10000;
    m[5] += 0x10000;
    Some(m)
}

/// The identity / default warp model (dav2d `dav2d_default_wm_params.matrix`): the base warp[]
/// candidate when there are no warp neighbours (e.g. block (0,0), or any wri==0 with an empty bank).
pub const IDENTITY_WARP: [i32; 6] = [0, 0, 0x10000, 0, 0, 0x10000];

/// dav2d MM_WARP_EXTEND (`extend_warpmv` decode.c:389 + neighbour selection decode.c:1188): a
/// warp block derives its matrix by EXTENDING a neighbour's warp/gmv/translation model. Returns
/// `Some(matrix)` if the result is a valid warp (`mat[2] > 0`), else `None` (WM_TYPE_INVALID →
/// caller splats uniform). `drl_off = (x_off, y_off)` from the DRL-selected mvstack candidate.
/// NOTE: the `ra[]` above-SB-row neighbour cases (an SB-row-boundary block with have_top reading a
/// top neighbour) are NOT supported — they return None. `(gx,gy)_log2` = block dim log2s (b_dim[2/3]).
#[allow(clippy::too_many_arguments)]
pub fn warp_extend(
    grid: &RefmvsGrid, bx4: usize, by4: usize, bw4: usize, bh4: usize, gx_log2: u32, gy_log2: u32,
    ref0: i8, fmv: Mv, drl_off: (i32, i32), sbsz: usize, iw4: usize, ih4: usize, gmv: [i32; 6],
) -> Option<[i32; 6]> {
    let have_left = bx4 > 0;
    let have_top = by4 > 0;
    let is_sb = ((by4 & (sbsz - 1)) == 0) as i32;
    let (mut x_off, mut y_off) = (0i32, 0i32);
    // 1. DRL-candidate offset (only when it points at a real neighbour, i.e. -1 in either axis).
    if drl_off.0 == -1 || drl_off.1 == -1 {
        x_off = drl_off.0;
        y_off = drl_off.1;
        // TIP-frame reset (dav decode.c:1143-1158): if the pointed-at refmvs block is a TIP
        // block (grid ref marker 7), discard the DRL offset and fall through to the
        // bml/rmt/tml/lmt probes. SB-boundary above reads the committed ra[] row
        // (ra[k] == grid[by4-1][2k]; ra_tl == ra[(bx-1)>>1]).
        let r = if is_sb == 1 && y_off == -1 {
            if (bx4 & (sbsz - 1)) != 0 || x_off >= 0 {
                *grid.at(by4 - 1, ((bx4 as i32 + x_off) & !1) as usize)
            } else {
                *grid.at(by4 - 1, (bx4 - 1) & !1)
            }
        } else {
            *grid.at((by4 as i32 + y_off) as usize, (bx4 as i32 + x_off) as usize)
        };
        if r.ref_[0] == 7 {
            x_off = 0;
            y_off = 0;
        }
        if std::env::var("WEDBG").is_ok() && bx4 == 22 && by4 == 54 {
            crate::dlog!("WEDBG drl_off={drl_off:?} r.ref={:?} r.mf={} -> off=({x_off},{y_off})", r.ref_, r.mf);
        }
    }
    // 2. Neighbour scan (bml → rmt → tml → lmt), first ref-match wins.
    let refm = |b: &RefmvsBlock| b.ref_[0] == ref0 || b.ref_[1] == ref0;
    if x_off == 0 && y_off == 0 {
        let bml = (have_left && by4 + bh4 <= ih4).then(|| *grid.at(by4 + bh4 - 1, bx4 - 1));
        // rmt/lmt read the committed above-row `ra[]` (even cols) for SB-boundary blocks (dav2d
        // decode.c:1205-1211): lmt = ra[bx&~1], rmt = ra[((bx&~1)+bw4-2)&~1].
        let rmt = (have_top && bx4 + bw4 <= iw4).then(|| if is_sb == 1 {
            *grid.at(by4 - 1, ((bx4 & !1) + bw4 - 2) & !1)
        } else {
            *grid.at(by4 - 1, bx4 + bw4 - 1)
        });
        let tml = have_left.then(|| *grid.at(by4, bx4 - 1));
        let lmt = have_top.then(|| if is_sb == 1 {
            *grid.at(by4 - 1, bx4 & !1)
        } else {
            *grid.at(by4 - 1, bx4)
        });
        if bml.as_ref().is_some_and(refm) {
            y_off = bh4 as i32 - 1;
            x_off = -1;
        } else if rmt.as_ref().is_some_and(refm) {
            y_off = -1;
            x_off = -(bx4 as i32 & is_sb) + bw4 as i32 - (1 + is_sb);
        } else if tml.as_ref().is_some_and(refm) {
            y_off = 0;
            x_off = -1;
        } else if lmt.as_ref().is_some_and(refm) {
            y_off = -1;
            x_off = -(bx4 as i32 & is_sb);
        }
    }
    if x_off == 0 && y_off == 0 {
        return None; // no extendable neighbour → WM_TYPE_INVALID
    }
    // extend_warpmv: fetch the neighbour (SB-boundary above reads the committed ra[] at even cols,
    // dav2d decode.c:1189-1195; == ra_tl when bx at a SB-column boundary), seed + extend one axis.
    let r = if y_off == -1 && is_sb == 1 {
        *grid.at(by4 - 1, ((bx4 as i32 + x_off) & !1) as usize)
    } else {
        *grid.at((by4 as i32 + y_off) as usize, (bx4 as i32 + x_off) as usize)
    };
    let mut m = if r.mf & 2 != 0 {
        r.matrix
    } else if r.mf & 1 != 0 {
        gmv
    } else {
        let ri = (r.ref_[0] != ref0) as usize;
        [r.mv[ri].x * (1 << 13), r.mv[ri].y * (1 << 13), 0x10000, 0, 0, 0x10000]
    };
    let sx = bx4 as i64 * 4 + 2 * bw4 as i64 - 1;
    let sy = by4 as i64 * 4 + 2 * bh4 as i64 - 1;
    let px = (sx << 16) + fmv.x as i64 * (1 << 13);
    let py = (sy << 16) + fmv.y as i64 * (1 << 13);
    if x_off >= 0 {
        // top neighbour: derive m[3], m[5]
        let ay = by4 as i64 * 4 - 1;
        let sh = 1 + gy_log2;
        let apx = m[2] as i64 * sx + m[3] as i64 * ay + m[0] as i64;
        let apy = m[4] as i64 * sx + m[5] as i64 * ay + m[1] as i64;
        let m3 = ((px - apx + bh4 as i64 - (px < apx) as i64) >> sh) as i32;
        let m5 = ((py - apy + bh4 as i64 - (py < apy) as i64) >> sh) as i32;
        m[3] = ((m3 + 0x20 - (m3 < 0) as i32) & !0x3f).clamp(-0x7fc0, 0x7fc0);
        m[5] = ((m5 + 0x20 - (m5 < 0x10000) as i32) & !0x3f).clamp(0x8040, 0x17fc0);
    } else {
        // left neighbour: derive m[2], m[4]
        let ax = bx4 as i64 * 4 - 1;
        let sh = 1 + gx_log2;
        let lpx = m[2] as i64 * ax + m[3] as i64 * sy + m[0] as i64;
        let lpy = m[4] as i64 * ax + m[5] as i64 * sy + m[1] as i64;
        let m2 = ((px - lpx + bw4 as i64 - (px < lpx) as i64) >> sh) as i32;
        let m4 = ((py - lpy + bw4 as i64 - (py < lpy) as i64) >> sh) as i32;
        m[2] = ((m2 + 0x20 - (m2 < 0x10000) as i32) & !0x3f).clamp(0x8040, 0x17fc0);
        m[4] = ((m4 + 0x20 - (m4 < 0) as i32) & !0x3f).clamp(-0x7fc0, 0x7fc0);
    }
    set_affine_mv2d(bw4, bh4, fmv, &mut m, bx4, by4);
    if std::env::var("WEDBG").is_ok() && bx4 == 22 && by4 == 54 {
        crate::dlog!("WEDBG final off=({x_off},{y_off}) fmv=({},{}) m={m:x?}", fmv.y, fmv.x);
    }
    (m[2] > 0).then_some(m) // WM_TYPE_INVALID iff mat[2] <= 0 (dav get_shear_params early return)
}

/// dav2d `apply_sign` — give `v` the sign of `s`.
#[inline]
fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 {
        -v
    } else {
        v
    }
}

/// The per-cell warp MV that `splat_warpmv_c` (refmvs.c:2308) writes into the grid for a warp
/// block: evaluate the warp `matrix` at the 2×2-cell block offset `(cx, cy)` (4px units) from the
/// block origin `(bx, by)`. This is why a warp neighbour hands DIFFERENT MVs to different readers
/// (block (0,0)'s rightmost column stores (25,159) at the top and (31,167) lower down).
pub fn warp_cell_mv(m: &[i32; 6], bx: i32, by: i32, cx: i32, cy: i32) -> Mv {
    let mvx = (m[2] - 0x10000) * (bx + 1 + cx) + m[3] * (by + 1 + cy) + (m[0] >> 2);
    let mvy = m[4] * (bx + 1 + cx) + (m[1] >> 2) + (m[5] - 0x10000) * (by + 1 + cy);
    Mv {
        y: apply_sign((mvy.abs() + 1024) >> 11, mvy).clamp(-0xffff, 0xffff),
        x: apply_sign((mvx.abs() + 1024) >> 11, mvx).clamp(-0xffff, 0xffff),
    }
}

/// State threaded through one block's candidate scan (dav2d `struct refmvs_state`, single-ref
/// subset): the stack, its count, and the dedup `iter_cntr`.
pub struct ScanState {
    pub mvstack: [Candidate; 6],
    pub cnt: usize,
    pub iter_cntr: i32,
    /// Compound cross-pair single-arm list (dav2d st->sngl, refmvs.c add_candidate_c2s): (ref, mv)
    /// entries harvested from neighbours carrying only ONE of the block's refs.
    pub sngl: [(i8, Mv); 4],
    pub sngl_cnt: usize,
    pub sngl_iter: i32,
    /// Derived candidates (dav2d st->dr): full pairs synthesized from cross-pairing / mv-traj,
    /// appended to the stack after the sort (add_derived).
    pub dr: [[Mv; 2]; 6],
    pub drvd_cnt: usize,
    pub drvd_iter: i32,
    /// Temporal window cell for THIS block (dav st->b8x8 = (bx4>>1) + ((by4&(sbsz-1))>>1)*stride).
    pub b8x8: usize,
}

impl Default for ScanState {
    fn default() -> Self {
        ScanState {
            mvstack: [Candidate::default(); 6], cnt: 0, iter_cntr: 0,
            sngl: [(-1, Mv::default()); 4], sngl_cnt: 0, sngl_iter: 0,
            dr: [[Mv::default(); 2]; 6], drvd_cnt: 0, drvd_iter: 0,
            b8x8: 0,
        }
    }
}

/// dav2d `add_candidate_c2s` (refmvs.c:101): dedup on (mv, ref) into the sngl list or append.
fn add_candidate_c2s(st: &mut ScanState, ref_: i8, cand_mv: Mv) {
    let last = st.sngl_cnt;
    if st.sngl_iter < 2 {
        for m in 0..last {
            if !crate::av2_recon::work_tick("refmvs:766") { break; }
            if st.sngl[m].1 == cand_mv && st.sngl[m].0 == ref_ {
                st.sngl_iter += m as i32 + 1;
                return;
            }
        }
        st.sngl_iter += last as i32;
    }
    if last >= 4 {
        return;
    }
    st.sngl[last] = (ref_, cand_mv);
    st.sngl_cnt = last + 1;
}

/// dav2d `add_derived` (refmvs.c:405), compound variant: append the derived pairs (weight 0,
/// cwp 8) into the stack up to `lim`.
fn add_derived_comp(st: &mut ScanState, lim: usize) {
    for n in 0..st.drvd_cnt {
        if !crate::av2_recon::work_tick("refmvs:784") { break; }
        if st.cnt >= 6 {
            break;
        }
        let pair = st.dr[n];
        let mut cnt = st.cnt;
        let mut iter = st.iter_cntr;
        add_candidate_comp(&mut st.mvstack, &mut cnt, lim, 0, 8, pair, &mut iter, 16);
        st.cnt = cnt;
        st.iter_cntr = iter;
    }
}

/// Single-ref derived entry into st.dr[..][0] (dav add_candidate_sngl on st->dr, budget 2, cap 4).
fn push_derived_sngl(st: &mut ScanState, cand: Mv) {
    let last = st.drvd_cnt;
    if st.drvd_iter < 2 {
        for n in 0..last {
            if !crate::av2_recon::work_tick("refmvs:801") { break; }
            if st.dr[n][0] == cand {
                st.drvd_iter += n as i32 + 1;
                return;
            }
        }
        st.drvd_iter += last as i32;
    }
    if last >= 4 {
        return;
    }
    st.dr[last][0] = cand;
    st.drvd_cnt = last + 1;
}

/// dav2d `add_derived_comp` entry into st.dr (dedup on both mvs, budget `drvd_iter` ≤ 2, cap 4).
fn push_derived_pair(st: &mut ScanState, pair: [Mv; 2]) {
    let last = st.drvd_cnt;
    if st.drvd_iter < 2 {
        for n in 0..last {
            if !crate::av2_recon::work_tick("refmvs:820") { break; }
            if st.dr[n][0] == pair[0] && st.dr[n][1] == pair[1] {
                st.drvd_iter += n as i32 + 1;
                return;
            }
        }
        st.drvd_iter += last as i32;
    }
    if last >= 4 {
        return;
    }
    st.dr[last] = pair;
    st.drvd_cnt = last + 1;
}

/// dav2d `add_spatial_candidate` (refmvs.c:189), the **single-ref, non-TIP** path: a spatial
/// neighbour `b` contributes its `mv[n]` (or the frame global MV if it's a globalmv block) to
/// the stack for each of its refs that matches the current block's ref. `ref0` is the current
/// block's ref frame; `gmv0` the frame global MV.
pub fn add_spatial_candidate_sngl(
    st: &mut ScanState,
    weight: i32,
    b: &RefmvsBlock,
    y_off: i32,
    x_off: i32,
    oy8: isize,
    ox8: isize,
    // sampled grid-cell mi coords (the grid.at position) — anchor for the avm TIP cell rule.
    cand_bx4: i32,
    cand_by4: i32,
    ref0: i8,
    ref1: i8,
    gmv: [Mv; 2],
) {
    if st.cnt >= 6 {
        return;
    }
    if b.mv[0].y == -0x8000 {
        return; // intra block, no intrabc → INVALID_MV
    }
    // TIP neighbour: rp_proj is read at the NEIGHBOUR's temporal 8x8 cell (oy8,ox8) — dav
    // refmvs.c:199-213. (The tip16 cell-pair quantization `off &= ~1` is a NO-OP in v320's
    // config: frame_mode=1 + seq tip_refine_mv=1 → tip16=0; revisit for other streams.)
    // INVALID → (0,0) per the dav arms (`if (tmv.y == INVALID_MV) tmv.n = 0`).
    let rp_tmv = || -> Mv {
        // avm derive_non_tip_mode_smvp_from_tip (mvref_common.h:1293): the rp_proj cell for a
        // TIP neighbour is the NEIGHBOUR BLOCK's ORIGIN cell + the quantized within-block
        // offset (is16 = TIP coded as 16x16 units). dav2d reads the sampled cell and carries a
        // literal FIXME for it (refmvs.c:208) — avm is normative here.
        let (aoy8, aox8) = if b.ref_[0] == 7 {
            let bd = crate::av2_decode::BLOCK_DIMENSIONS[b.bs as usize];
            let (bw4n, bh4n) = (bd[0] as i32, bd[1] as i32);
            let seq_refine = crate::av2_recon::SEQ_TIP.with(|c| c.get()).4;
            let fm = TMVS.with(|c| c.borrow().tip_frame_mode);
            // dav tip16 formula (refmvs.c:201); frame_mode==2 assumes SHARP subpel (the
            // subpel term only matters for whole-TIP frames, not in the corpus yet).
            let is16 = if fm == 2 { !seq_refine } else { (!seq_refine && bw4n.min(bh4n) >= 4) || bw4n.max(bh4n) >= 64 } as i32;
            let ax = ((b.bx4 as i32) >> 1) + ((((cand_bx4 - b.bx4 as i32) >> 1) >> is16) << is16);
            let ay = ((b.by4 as i32) >> 1) + ((((cand_by4 - b.by4 as i32) >> 1) >> is16) << is16);
            (oy8 - ((cand_by4 >> 1) as isize - ay as isize), ax as isize)
        } else {
            (oy8, ox8)
        };
        let stride = TMVS.with(|c| c.borrow().stride) as isize;
        let idx = 2 * stride + aoy8 * stride + aox8;
        let m = RP_PROJ.with(|c| c.borrow().get(idx.max(0) as usize).map(|p| p.0))
            .unwrap_or(Mv { y: INVALID_MV_I32, x: INVALID_MV_I32 });
        if m.y == INVALID_MV_I32 { Mv { y: 0, x: 0 } } else { m }
    };
    if ref1 < 0 {
        // single-ref: scan both of the neighbour's ref slots (num = 1 + (ref0>=0)).
        let t = TMVS.with(|c| c.borrow().clone());
        let num = 1 + (ref0 >= 0) as usize;
        let tip_pair = [t.tip_ref.0 as i8, t.tip_ref.1 as i8];
        for n in 0..num {
            if !crate::av2_recon::work_tick("refmvs:894") { break; }
            // arm-n effective ref for the mvtj/lnr arms: a TIP neighbour projects through the
            // tip source pair (dav `b->ref.ref[0] == TIP_FRAME ? rf->tip.ref.ref[n] : b->ref.ref[n]`).
            let bref_n = if b.ref_[0] == 7 { tip_pair[n] } else { b.ref_[n] };
            if b.ref_[n] == ref0 {
                let cand_mv = if (b.mf & 1) != 0 && gmv[0].y != -0x8000 { gmv[0] } else { b.mv[n] };
                add_candidate_sngl(&mut st.mvstack, &mut st.cnt, 6, weight, cand_mv, y_off, x_off, &mut st.iter_cntr, 16);
            } else if b.ref_[0] == 7 && t.valid && tip_pair[n] == ref0 {
                // tip-spc (refmvs.c:224): TIP neighbour whose tip source arm n IS the searched
                // ref → rp_proj at the neighbour cell, scaled to arm n, + the block's tip MV.
                let tipmv = scale_mv(rp_tmv(), t.tip_sf[n]);
                let cand_mv = Mv {
                    y: (tipmv.y + b.mv[0].y).clamp(-0xffff, 0xffff),
                    x: (tipmv.x + b.mv[0].x).clamp(-0xffff, 0xffff),
                };
                add_candidate_sngl(&mut st.mvstack, &mut st.cnt, 6, weight, cand_mv, y_off, x_off, &mut st.iter_cntr, 16);
            } else if ref0 == 7 && t.valid && (b.ref_[0], b.ref_[1]) == (tip_pair[0], tip_pair[1]) {
                // tip2-spc (refmvs.c:238): searching TIP, compound neighbour carrying the tip
                // pair → derived from the inter-arm delta scaled by sf[0].
                let in_delta = Mv { y: b.mv[0].y - b.mv[1].y, x: b.mv[0].x - b.mv[1].x };
                let out_delta = scale_mv(in_delta, t.tip_sf[0]);
                let cand = Mv {
                    y: (b.mv[0].y - out_delta.y).clamp(-0xffff, 0xffff),
                    x: (b.mv[0].x - out_delta.x).clamp(-0xffff, 0xffff),
                };
                push_derived_sngl(st, cand);
                break;
            } else if t.valid && t.mv_traj && t.use_ref_frame_mvs
                && (0..7).contains(&ref0)
                && (b.ref_[0] == 7 || (0..7).contains(&b.ref_[n]))
                && RP_TRAJ.with(|c| c.borrow()[ref0 as usize][st.b8x8].y) != INVALID_MV_I32
                && RP_TRAJ.with(|c| c.borrow()[bref_n as usize][st.b8x8].y) != INVALID_MV_I32
            {
                // mv-traj derived (refmvs.c:255): b_mv + traj[ref0] - traj[bref_n] -> drvd.
                // TIP neighbour: b_mv = rp_proj-scaled + tip block MV (refmvs.c:265).
                let a_mv = RP_TRAJ.with(|c| c.borrow()[bref_n as usize][st.b8x8]);
                let c_mv = RP_TRAJ.with(|c| c.borrow()[ref0 as usize][st.b8x8]);
                let b_mv = if b.ref_[0] == 7 {
                    let tipmv = scale_mv(rp_tmv(), t.tip_sf[n]);
                    Mv {
                        y: (tipmv.y + b.mv[0].y).clamp(-0xffff, 0xffff),
                        x: (tipmv.x + b.mv[0].x).clamp(-0xffff, 0xffff),
                    }
                } else {
                    b.mv[n]
                };
                let cand = Mv {
                    y: (b_mv.y + c_mv.y - a_mv.y).clamp(-0xffff, 0xffff),
                    x: (b_mv.x + c_mv.x - a_mv.x).clamp(-0xffff, 0xffff),
                };
                push_derived_sngl(st, cand);
            } else if t.valid && (0..7).contains(&ref0) && b.ref_[0] >= 0 && (0..7).contains(&bref_n)
                && t.ref_sign[ref0 as usize] == t.ref_sign[bref_n as usize]
            {
                // same-sign linear projection (refmvs.c:287) -> drvd. TIP neighbour: numerator
                // MV = rp_proj-scaled + tip block MV, den = abspocdiff[tip source arm].
                let (num_mv, den) = if b.ref_[0] == 7 {
                    let tipmv = scale_mv(rp_tmv(), t.tip_sf[n]);
                    (Mv {
                        y: (tipmv.y + b.mv[0].y).clamp(-0xffff, 0xffff),
                        x: (tipmv.x + b.mv[0].x).clamp(-0xffff, 0xffff),
                    }, t.abspocdiff[bref_n as usize])
                } else {
                    (b.mv[n], t.abspocdiff[b.ref_[n] as usize])
                };
                let cand = mv_projection_t(num_mv, t.abspocdiff[ref0 as usize], den, -0xffff, 0xffff);
                push_derived_sngl(st, cand);
            }
            if b.ref_[1] < 0 && b.ref_[0] != 7 {
                break;
            }
        }
        return;
    }
    // tip-spc COMPOUND (refmvs.c:317): TIP neighbour + searched pair == the tip source pair →
    // both arms from the SAME rp_proj cell mv scaled per-arm, + the tip block MV; cwp=8.
    {
        let t = TMVS.with(|c| c.borrow().clone());
        if b.ref_[0] == 7 && t.valid && (ref0, ref1) == (t.tip_ref.0 as i8, t.tip_ref.1 as i8) {
            let tmv = rp_tmv();
            let t0 = scale_mv(tmv, t.tip_sf[0]);
            let t1 = scale_mv(tmv, t.tip_sf[1]);
            let cand = [
                Mv { y: (t0.y + b.mv[0].y).clamp(-0xffff, 0xffff), x: (t0.x + b.mv[0].x).clamp(-0xffff, 0xffff) },
                Mv { y: (t1.y + b.mv[0].y).clamp(-0xffff, 0xffff), x: (t1.x + b.mv[0].x).clamp(-0xffff, 0xffff) },
            ];
            add_candidate_comp(&mut st.mvstack, &mut st.cnt, 6, weight, 8, cand, &mut st.iter_cntr, 16);
            return;
        }
    }
    // compound traj-derived arm (refmvs.c:346): when the pair does NOT match and traj is live.
    if !(b.ref_[0] == ref0 && b.ref_[1] == ref1) {
        let t = TMVS.with(|c| c.borrow().clone());
        if t.valid && t.mv_traj && t.use_ref_frame_mvs && b.ref_[0] != 7 && ref0 != ref1
            && (0..7).contains(&ref0) && (0..7).contains(&ref1)
            && RP_TRAJ.with(|c| c.borrow()[ref0 as usize][st.b8x8].y) != INVALID_MV_I32
            && RP_TRAJ.with(|c| c.borrow()[ref1 as usize][st.b8x8].y) != INVALID_MV_I32
        {
            let b1 = RP_TRAJ.with(|c| c.borrow()[ref0 as usize][st.b8x8]);
            let b2 = RP_TRAJ.with(|c| c.borrow()[ref1 as usize][st.b8x8]);
            for n in 0..2usize {
                if !crate::av2_recon::work_tick("refmvs:994") { break; }
                if b.ref_[n] < 0 {
                    break;
                }
                if !(0..7).contains(&b.ref_[n]) {
                    continue;
                }
                let a_mv = RP_TRAJ.with(|c| c.borrow()[b.ref_[n] as usize][st.b8x8]);
                if a_mv.y == INVALID_MV_I32 {
                    continue;
                }
                let pair = [
                    Mv { y: (b.mv[n].y + b1.y - a_mv.y).clamp(-0xffff, 0xffff),
                         x: (b.mv[n].x + b1.x - a_mv.x).clamp(-0xffff, 0xffff) },
                    Mv { y: (b.mv[n].y + b2.y - a_mv.y).clamp(-0xffff, 0xffff),
                         x: (b.mv[n].x + b2.x - a_mv.x).clamp(-0xffff, 0xffff) },
                ];
                push_derived_pair(st, pair);
            }
        }
    }
    // ===== COMPOUND (dav2d refmvs.c:337-402; the mv-traj arm 346-375 needs rp_traj — omitted) =====
    if b.ref_[0] == ref0 && b.ref_[1] == ref1 {
        // full pair match → weighted compound candidate carrying the neighbour's cwp (mf>>2).
        let cand = [
            if (b.mf & 1) != 0 && gmv[0].y != -0x8000 { gmv[0] } else { b.mv[0] },
            if (b.mf & 1) != 0 && gmv[1].y != -0x8000 { gmv[1] } else { b.mv[1] },
        ];
        add_candidate_comp(&mut st.mvstack, &mut st.cnt, 6, weight, (b.mf >> 2) as i8, cand, &mut st.iter_cntr, 16);
    } else {
        // cross-pair (refmvs.c:377-401): the neighbour carries only ONE of the block's refs.
        let refp = [ref0, ref1];
        let ns: usize = if ref0 == b.ref_[0] || ref0 == b.ref_[1] {
            0
        } else if ref1 != b.ref_[0] && ref1 != b.ref_[1] {
            return;
        } else {
            1
        };
        let nc = (refp[ns] != b.ref_[0]) as usize;
        // mvxp: pair the neighbour's arm with a previously-harvested sngl arm of the OTHER ref.
        let mut oidx = st.sngl_cnt;
        for i in 0..st.sngl_cnt {
            if !crate::av2_recon::work_tick("refmvs:1036") { break; }
            if refp[1 - ns] == st.sngl[i].0 {
                oidx = i;
                break;
            }
        }
        if oidx < st.sngl_cnt {
            let mut cand = [Mv::default(); 2];
            cand[ns] = b.mv[nc];
            cand[1 - ns] = st.sngl[oidx].1;
            push_derived_pair(st, cand);
        }
        let cand_mv = if (b.mf & 1) != 0 && gmv[nc].y != -0x8000 { gmv[ns] } else { b.mv[nc] };
        add_candidate_c2s(st, b.ref_[nc], cand_mv);
    }
}

/// The frame global MV as a stack candidate (dav2d `refmvs_find` GMV fill, refmvs.c:942): after
/// the spatial scan, if the stack has room and isn't already carrying `gmv0`, append it (weight
/// 0). This alone produces block (0,0)'s degenerate stack `[(0,0)]`.
pub fn add_global_candidate(st: &mut ScanState, gmv0: Mv) {
    if st.cnt >= 6 {
        return;
    }
    let last = st.cnt;
    if st.iter_cntr < 16 {
        for n in 0..last {
            if !crate::av2_recon::work_tick("refmvs:1062") { break; }
            if st.mvstack[n].mv[0] == gmv0 {
                st.iter_cntr += n as i32 + 1;
                return;
            }
        }
        st.iter_cntr += last as i32;
    }
    st.mvstack[last].mv[0] = gmv0;
    st.mvstack[last].weight = 0;
    st.mvstack[last].y_off = 0;
    st.mvstack[last].x_off = 0;
    st.cnt = last + 1;
}

/// The refmv bank (dav2d `rt->bank`, single-ref subset): a per-ref-class rolling ring of ≤4 recent
/// block MVs, `avail`-gated per SB. `refmvs_find` reads it (after the spatial sort) to add recent
/// MVs as weight-0 candidates — the source of e.g. (4,0)'s (24,168) ((0,0)'s block MV, banked).
/// dav2d qmv INVALID_TRAJ sentinel (refmvs.h:43).
pub const INVALID_TRAJ: u16 = 0x8080;

/// One 8x8 cell of a frame's saved temporal motion field (dav2d refmvs_temporal_block):
/// the ref pair (t_swap-ordered) + the QUANTIZED mv pair (packed qmv: y<<8|x as u8 lanes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TemporalBlock {
    pub ref_: (i8, i8),
    pub qmv: [u16; 2],
}

impl Default for TemporalBlock {
    fn default() -> Self {
        TemporalBlock { ref_: (-1, -1), qmv: [INVALID_TRAJ; 2] }
    }
}

/// dav2d `quantize_mv_comp` (refmvs.h:224): log-companded magnitude, |v| < 2048 -> 7 bits.
#[inline]
fn quantize_mv_comp(absv: u32) -> u32 {
    if absv == 0 {
        return 0;
    }
    let nbits = (31 - absv.leading_zeros()).saturating_sub(4).min(6);
    (absv >> nbits) + 16 * nbits
}

/// dav2d `quantize_mv` (refmvs.h:232): pack an MV into a qmv u16 (INVALID_TRAJ when >= 2048).
pub fn quantize_mv(mv: Mv) -> u16 {
    let (ay, ax) = (mv.y.unsigned_abs(), mv.x.unsigned_abs());
    if ay.max(ax) >= 2048 {
        return INVALID_TRAJ;
    }
    let qy = if mv.y < 0 { -(quantize_mv_comp(ay) as i32) } else { quantize_mv_comp(ay) as i32 } as i8;
    let qx = if mv.x < 0 { -(quantize_mv_comp(ax) as i32) } else { quantize_mv_comp(ax) as i32 } as i8;
    ((qy as u8 as u16) << 8) | (qx as u8 as u16)
}

/// dav2d `dequantize_mv_comp` (refmvs.c:1349).
#[inline]
fn dequantize_mv_comp(v: i32) -> i32 {
    let absv = v.unsigned_abs();
    let nbits = (absv >> 4) as i32 - (absv >= 16) as i32;
    let res = ((absv as i32) - nbits * 16) << nbits;
    if v < 0 { -res } else { res }
}

/// dav2d `dequantize_mv` (refmvs.c:1357): qmv -> MV, INVALID_TRAJ -> y = INVALID_MV.
pub fn dequantize_mv(q: u16) -> Mv {
    if q == INVALID_TRAJ {
        return Mv { y: -0x8000, x: 0 };
    }
    let qy = (q >> 8) as u8 as i8 as i32;
    let qx = (q & 0xff) as u8 as i8 as i32;
    Mv { y: dequantize_mv_comp(qy), x: dequantize_mv_comp(qx) }
}

/// A saved per-ref-slot motion field + the metadata the mfmv projection setup needs
/// (dav2d rp_ref[slot] + c->refs[].p.frame_hdr): the field, the frame's poc, its OWN
/// per-list-index ref pocs, and its n_ref (dav refcnt/ref_ref_poc).
#[derive(Clone, Default)]
pub struct SavedMotionField {
    pub w8: usize,
    pub h8: usize,
    pub cells: Vec<TemporalBlock>,
    pub poc: u32,
    pub refpoc: [u32; 7],
    pub n_ref: u32,
}

thread_local! {
    /// The CURRENT frame's temporal motion field (dav2d rf->rp): (stride8, ih8, cells). Reset per
    /// inter frame; written per block (splat_*_mv t arm); saved to RP_REF on frame end.
    pub static RP_CUR: std::cell::RefCell<(usize, usize, Vec<TemporalBlock>)> =
        const { std::cell::RefCell::new((0, 0, Vec::new())) };
    /// Per-ref-slot SAVED motion fields (dav2d rf->rp_ref[slot]), parallel to REF_PICS.
    pub static RP_REF: std::cell::RefCell<[Option<SavedMotionField>; 8]> =
        const { std::cell::RefCell::new([None, None, None, None, None, None, None, None]) };
}

/// Reset RP_CUR for a new frame of `iw4` x `ih4` 4px units (cells are 8x8 -> dims (i+1)>>1).
pub fn rp_reset(iw4: usize, ih4: usize) {
    let (w8, h8) = ((iw4 + 1) >> 1, (ih4 + 1) >> 1);
    RP_CUR.with(|c| {
        let mut b = c.borrow_mut();
        *b = (w8, h8, vec![TemporalBlock::default(); w8 * h8]);
    });
}

/// Write a block's temporal cells (dav2d splat_mv_c t arm): ceil(bw4/2) x ceil(bh4/2) cells
/// at (bx4>>1, by4>>1).
pub fn rp_write(bx4: usize, by4: usize, bw4: usize, bh4: usize, tb: TemporalBlock) {
    RP_CUR.with(|c| {
        let mut b = c.borrow_mut();
        let (w8, h8, ref mut cells) = *b;
        if cells.is_empty() {
            return;
        }
        let (x0, y0) = (bx4 >> 1, by4 >> 1);
        for y in 0..bh4.div_ceil(2) {
            if !crate::av2_recon::work_tick("refmvs:1179") { break; }
            if y0 + y >= h8 {
                break;
            }
            for x in 0..bw4.div_ceil(2) {
                if !crate::av2_recon::work_tick("refmvs:1183") { break; }
                if x0 + x >= w8 {
                    break;
                }
                cells[(y0 + y) * w8 + (x0 + x)] = tb;
            }
        }
    });
}

/// Save RP_CUR into every RP_REF slot in `refresh` (call at frame end, like update_ref_pics).
/// `poc`/`refpoc`/`n_ref` = the just-decoded frame's own reference metadata (feeds the mfmv
/// projection setup: ref_ref_poc / refref2curref_idx).
/// Clear all saved per-slot motion fields + grids (new-sequence reset).
pub fn reset_stream_state() {
    RP_REF.with(|r| *r.borrow_mut() = std::array::from_fn(|_| None));
    GRID.with(|g| *g.borrow_mut() = RefmvsGrid::default());
    BANK.with(|b| *b.borrow_mut() = RefmvBank::default());
}

pub fn rp_save(refresh: u32, poc: u32, refpoc: [u32; 7], n_ref: u32) {
    RP_CUR.with(|c| {
        let (w8, h8, ref cells) = *c.borrow();
        if std::env::var("RPDBG").map_or(false, |v| v == format!("{poc}")) {
            for r in 0..3usize {
                if !crate::av2_recon::work_tick("refmvs:1207") { break; }
                let row: Vec<String> = (0..16.min(w8)).map(|x| {
                    let tb = &cells[r * w8 + x];
                    let m0 = dequantize_mv(tb.qmv[0]);
                    let m1 = dequantize_mv(tb.qmv[1]);
                    format!("({},{};{}|{},{};{})", m0.y, m0.x, tb.ref_.0, m1.y, m1.x, tb.ref_.1)
                }).collect();
                crate::dlog!("[MRMVS] poc={poc} row{r}: {}", row.join(" "));
            }
        }
        let saved = SavedMotionField { w8, h8, cells: cells.clone(), poc, refpoc, n_ref };
        RP_REF.with(|r| {
            let mut slots = r.borrow_mut();
            for (i, slot) in slots.iter_mut().enumerate() {
                if refresh & (1 << i) != 0 {
                    *slot = Some(saved.clone());
                }
            }
        });
    });
}

// ===================== TEMPORAL MV PROJECTION ENGINE (dav2d load_tmvs) =====================
// Per-frame setup (dav refmvs.c:2040-2280) + per-SB-row projection (1760-1928). 64px-SB,
// sample_step=1 geometry: sbsz8=8, mfmv_sbsz8=8, mfmv_edge=4, k_shift=3.

pub const INVALID_MV_I32: i32 = -0x8000;
const INVALID_REF2CUR: i32 = -32;

/// dav2d `scale_mv` (refmvs.h:216).
#[inline]
pub fn scale_mv(m: Mv, sf: i32) -> Mv {
    let y = m.y as i64 * sf as i64;
    let x = m.x as i64 * sf as i64;
    Mv {
        y: (((y + 0x2000 - (y < 0) as i64) >> 14) as i32).clamp(-0xffff, 0xffff),
        x: (((x + 0x2000 - (x < 0) as i64) >> 14) as i32).clamp(-0xffff, 0xffff),
    }
}

const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340,
    2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712,
    682, 655, 630, 606, 585, 564, 546, 528,
];

/// dav2d `dav2d_mv_projection` (refmvs.c:431).
pub fn mv_projection_t(m: Mv, num: i32, den: i32, min: i32, max: i32) -> Mv {
    let frac = num * DIV_MULT[den.unsigned_abs().min(31) as usize];
    let y = m.y * frac;
    let x = m.x * frac;
    Mv {
        y: ((y + 8192 + (y >> 31)) >> 14).clamp(min, max),
        x: ((x + 8192 + (x >> 31)) >> 14).clamp(min, max),
    }
}

/// The per-frame temporal-projection context (subset of dav2d refmvs_frame).
#[derive(Clone, Default)]
pub struct TmvsFrame {
    pub valid: bool,
    pub mv_traj: bool,
    pub n_ref: usize,
    pub pocdiff: [i32; 7],
    pub abspocdiff: [i32; 7],
    pub ref_sign: [bool; 7],
    pub n_mfmvs: usize,
    pub mfmv: [(i8, i8, u8); 4], // (ref, tgt, dir)
    pub mfmv_ref2cur: [i32; 4],
    pub mfmv_ref2ref: [[i32; 7]; 4],
    pub mfmv_ref2idx: [[i8; 7]; 4],
    pub mfmv_ref2sf: [[[i32; 2]; 7]; 4],
    pub use_ref_frame_mvs: bool,
    pub tip_ref: (u8, u8),
    pub tip_delta: i32,
    pub tip_sf: [i32; 2],
    pub tip_frame_mode: u8,
    pub tip_hole_fill: bool,
    pub stride: usize,
    pub iw8: i32,
    pub ih8: i32,
    /// frame hdr `tmvp_sample_step` (1 or 2): the temporal projection samples every step-th
    /// 8x8 cell; step=2 (sb128 only) also doubles the mfmv window (mfmv_sbsz8 16, edge 16,
    /// k_shift 4) and gap-fills the skipped cells after projection (dav refmvs.c:1921).
    pub sample_step: i32,
}

thread_local! {
    pub static TMVS: std::cell::RefCell<TmvsFrame> = std::cell::RefCell::new(TmvsFrame::default());
    /// rp_proj window: (sbsz8+2)=10 rows x stride of (mv, ref), base offset 2 rows (dav poffset).
    pub static RP_PROJ: std::cell::RefCell<Vec<(Mv, i32)>> = const { std::cell::RefCell::new(Vec::new()) };
    /// rp_traj[ref]: sbsz8=8 rows x stride.
    pub static RP_TRAJ: std::cell::RefCell<[Vec<Mv>; 7]> = std::cell::RefCell::new(Default::default());
    /// rp_map[k*7+ref]: sbsz8 rows x stride of (dy, dx) traj deltas ((-128,-128) = INVALID_TRAJ).
    pub static RP_MAP: std::cell::RefCell<Vec<Vec<(i8, i8)>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// dav2d `add_temporal_candidate` (refmvs.c:444), non-TIP refs: read the traj (or projected
/// rp_proj rescaled by pocdiff) at `off_8x8` and add as a weighted candidate.
#[allow(clippy::too_many_arguments)]
fn add_temporal_candidate_t(
    st: &mut ScanState, ref0: i8, ref1: i8, off_8x8: usize,
) -> bool {
    let t = TMVS.with(|c| c.borrow().clone());
    if !t.valid || ref0 < 0 || ref0 >= 7 {
        return false;
    }
    let stride = t.stride;
    let read_arm = |r: i8| -> Option<Mv> {
        let mut m = RP_TRAJ.with(|c| c.borrow()[r as usize][off_8x8]);
        if !t.mv_traj || m.y == INVALID_MV_I32 {
            let (pm, pref) = RP_PROJ.with(|c| c.borrow()[2 * stride + off_8x8]);
            if pm.y == INVALID_MV_I32 {
                return None;
            }
            m = mv_projection_t(pm, t.pocdiff[r as usize], pref, -0xffff, 0xffff);
        }
        Some(m)
    };
    let mv0 = match read_arm(ref0) {
        Some(m) => m,
        None => return false,
    };
    if ref1 == -1 {
        let weight = 1 + (t.abspocdiff[ref0 as usize] <= 2) as i32;
        return add_candidate_sngl(&mut st.mvstack, &mut st.cnt, 6, weight, mv0, 0, 0, &mut st.iter_cntr, 16);
    }
    let mv1 = match read_arm(ref1) {
        Some(m) => m,
        None => return false,
    };
    add_candidate_comp(&mut st.mvstack, &mut st.cnt, 6, 1, 8, [mv0, mv1], &mut st.iter_cntr, 16)
}

fn topo_insert(cnt: usize, idx: usize, order: &mut [i8; 7], rev: &mut [i8; 7],
               cnv: &[[i8; 7]; 7], walk_deps: &[bool; 7]) -> usize {
    if rev[idx] != -1 {
        return cnt;
    }
    rev[idx] = 0; // dummy
    let mut cnt = cnt;
    if walk_deps[idx] {
        for n in 0..7 {
            let r = cnv[idx][n];
            if r == -1 {
                continue;
            }
            cnt = topo_insert(cnt, r as usize, order, rev, cnv, walk_deps);
        }
    }
    order[cnt] = idx as i8;
    rev[idx] = cnt as i8;
    cnt + 1
}

fn abs_closest_ref(ref2ref: &[i32; 7], cur2ref: &[i32; 7], dir: bool) -> i32 {
    let mut b = 0xff;
    for n in 0..7 {
        let a = ref2ref[n].abs();
        if ((cur2ref[n] > 0 && ref2ref[n] > 0 && dir) || (cur2ref[n] < 0 && ref2ref[n] < 0 && !dir)) && a < b {
            b = a;
        }
    }
    b
}

/// Per-frame temporal setup (dav refmvs.c:2040-2280 + find_tip_ref_frames obu.c:888).
/// Reads the SAVED per-slot motion-field metadata from RP_REF via `refidx`.
#[allow(clippy::too_many_arguments)]
pub fn tmvs_setup(
    nbits: u32, poc: u32, n_ref: usize, refidx: &[u8; 7], refpoc: &[u32; 7],
    use_ref_frame_mvs_hdr: bool, mv_traj: bool, seq_tip: bool,
    tip_frame_mode: u8, tip_hole_fill: bool, iw4: usize, ih4: usize,
    sample_step: i32,
) {
    let gpd = |a: u32, b: u32| crate::av2_recon::get_poc_diff(nbits, a, b);
    let mut t = TmvsFrame {
        valid: true,
        mv_traj,
        n_ref,
        use_ref_frame_mvs: false,
        tip_frame_mode,
        tip_hole_fill,
        sample_step: sample_step.max(1),
        stride: (iw4 + 1) >> 1,
        iw8: ((iw4 + 1) >> 1) as i32,
        ih8: ((ih4 + 1) >> 1) as i32,
        ..Default::default()
    };
    // per-ref metadata from the saved fields
    let mut refcnt = [0u32; 7];
    let mut ref_ref_poc = [[0u32; 7]; 7];
    let mut have_field = [false; 7];
    RP_REF.with(|r| {
        let slots = r.borrow();
        for i in 0..n_ref {
            if !crate::av2_recon::work_tick("refmvs:1403") { break; }
            if let Some(f) = slots[refidx[i] as usize].as_ref() {
                if !f.cells.is_empty() {
                    have_field[i] = true;
                    refcnt[i] = f.n_ref;
                    ref_ref_poc[i] = f.refpoc;
                }
            }
        }
    });
    let mut ref2ref = [[0i32; 7]; 7];
    let mut ref2cur = [[0i32; 7]; 7];
    let mut have_ref_sign = [[false; 2]; 7];
    let mut refref2curref_idx = [[-1i8; 7]; 7];
    for i in 0..n_ref {
        if !crate::av2_recon::work_tick("refmvs:1417") { break; }
        t.ref_sign[i] = gpd(refpoc[i], poc) < 0;
        t.pocdiff[i] = gpd(poc, refpoc[i]).clamp(-31, 31);
        t.abspocdiff[i] = t.pocdiff[i].abs();
        if refcnt[i] != 0 {
            // Only the ref's FIRST refcnt entries are valid (avm ref_display_order_hint == -1
            // beyond) — junk tail entries polluted have_ref_sign/interp-dist/topo on the
            // post-s-frame topology (poc3's single future ref faked a past one).
            for nn in 0..(refcnt[i] as usize).min(7) {
                if !crate::av2_recon::work_tick("refmvs:1425") { break; }
                ref2ref[i][nn] = gpd(refpoc[i], ref_ref_poc[i][nn]);
                if ref2ref[i][nn] > 0 { have_ref_sign[i][0] = true; }
                if ref2ref[i][nn] < 0 { have_ref_sign[i][1] = true; }
                ref2cur[i][nn] = gpd(poc, ref_ref_poc[i][nn]);
                let mut m = n_ref;
                for mm in 0..n_ref {
                    if !crate::av2_recon::work_tick("refmvs:1431") { break; }
                    if ref_ref_poc[i][nn] == refpoc[mm] { m = mm; break; }
                }
                refref2curref_idx[i][nn] = if m == n_ref { -1 } else { m as i8 };
            }
        }
    }
    // tip setup (find_tip_ref_frames + sf)
    if seq_tip && n_ref > 1 {
        let mut order = [0u8; 7];
        let mut refdist = [0i32; 7];
        let mut n_past = 0usize;
        for n in 0..n_ref {
            if !crate::av2_recon::work_tick("refmvs:1443") { break; }
            let dist = gpd(refpoc[n], poc);
            refdist[n] = dist;
            let mut m = n;
            while m > 0 && refdist[order[m - 1] as usize] > dist {
                order[m] = order[m - 1];
                m -= 1;
            }
            order[m] = n as u8;
            n_past += (dist < 0) as usize;
        }
        t.tip_ref = if n_past == n_ref {
            (order[n_ref - 1], order[n_ref - 2])
        } else if n_past == 0 {
            (order[0], order[1])
        } else {
            (order[n_past - 1], order[n_past])
        };
        if tip_frame_mode != 0 {
            let d2 = gpd(refpoc[t.tip_ref.1 as usize], refpoc[t.tip_ref.0 as usize]);
            t.tip_delta = d2.abs();
            let d1 = t.pocdiff[t.tip_ref.0 as usize];
            let dv = DIV_MULT[(d2.abs().min(31)) as usize];
            t.tip_sf[0] = d1.abs().min(31) * dv;
            if (d1 < 0) != (d2 < 0) { t.tip_sf[0] = -t.tip_sf[0]; }
            let d3 = t.pocdiff[t.tip_ref.1 as usize];
            t.tip_sf[1] = d3.abs().min(31) * dv;
            if (d3 < 0) != (d2 < 0) { t.tip_sf[1] = -t.tip_sf[1]; }
        }
    }
    // mfmv list (refmvs.c:2102-2245)
    if use_ref_frame_mvs_hdr && nbits != 0 {
        let mut order = [0u8; 7];
        for n in 0..n_ref {
            if !crate::av2_recon::work_tick("refmvs:1476") { break; }
            let pd = t.pocdiff[n];
            let mut m = n;
            while m > 0 && pd > t.pocdiff[order[m - 1] as usize] {
                order[m] = order[m - 1];
                m -= 1;
            }
            order[m] = n as u8;
        }
        let mut first_fut = 0usize;
        while first_fut < n_ref && t.ref_sign[order[first_fut] as usize] {
            first_fut += 1;
        }
        let mut topo_order = [0i8; 7];
        let mut rev_topo = [-1i8; 7];
        let mut topo_cnt = 0usize;
        // avm recur_topo_sort_refs (mvref_common.c:3074): a ref's DEPENDENCIES are walked only
        // when its slot holds a plain INTER_FRAME — key/intra (refcnt==0 naturally) AND S-frame
        // slots are visited as nodes but their stored ref lists are NOT followed.
        let walk_deps: [bool; 7] = {
            let mut w = [false; 7];
            crate::av2_recon::REF_SLOTS.with(|s| {
                let slots = s.borrow();
                for (i, wi) in w.iter_mut().enumerate().take(n_ref) {
                    let plain_inter = matches!(slots[refidx[i] as usize],
                        Some(r) if !r.is_key_or_intra && !r.is_sframe);
                    *wi = refcnt[i] != 0 && plain_inter;
                }
            });
            w
        };
        for n in 0..n_ref {
            if !crate::av2_recon::work_tick("refmvs:1507") { break; }
            topo_cnt = topo_insert(topo_cnt, n, &mut topo_order, &mut rev_topo, &refref2curref_idx, &walk_deps);
        }
        if topo_cnt > 1 {
            // === avm av2_setup_motion_field process-ref selection (mvref_common.c:4211-4340),
            // ported EXACTLY (the previous dav2d-mirror fill agreed on plain GOPs but diverged
            // on the post-s-frame ref topology). No restricted/overlay refs in current streams
            // (restricted slots would additionally be skipped from sort/topo). ===
            // Slot eligibility (is_ref_motion_field_eligible): has a saved field and is not a
            // key/intra slot (an S-FRAME slot IS an eligible projection source; frame sizes
            // are equal for all current streams).
            let elig: [bool; 7] = {
                let mut e = [false; 7];
                crate::av2_recon::REF_SLOTS.with(|s| {
                    let slots = s.borrow();
                    for (i, ei) in e.iter_mut().enumerate().take(n_ref) {
                        let not_key = matches!(slots[refidx[i] as usize], Some(r) if !r.is_key_or_intra);
                        *ei = have_field[i] && not_key;
                    }
                });
                e
            };
            // sort_ref ascending display order == `order`; cur_frame_sort_idx = last past ref.
            let cur_idx: i32 = first_fut as i32 - 1;
            let has_both_sides = first_fut > 0 && first_fut < n_ref;
            let n_past = first_fut;
            let mut checked = [[false; 2]; 7];
            let mut checked_count = 0usize;
            // check_and_add_process_ref (mvref_common.c:3154): eligibility + MAX_FRAME_DISTANCE
            // + the checked[start][side] / checked_count<max_check bookkeeping + process cap 4.
            macro_rules! add_proc {
                ($start:expr, $tgt:expr, $side:expr, $max_check:expr) => {{
                    let sti: i32 = $start;
                    if sti >= 0 {
                        let s = sti as usize;
                        if elig[s] && t.abspocdiff[s] <= 31 && t.n_mfmvs < 4 && !checked[s][$side] {
                            if checked_count < $max_check {
                                checked[s][$side] = true;
                                checked_count += 1;
                                t.mfmv[t.n_mfmvs] = (s as i8, $tgt, $side as u8);
                                t.n_mfmvs += 1;
                            }
                        }
                    }
                }};
            }
            // (a) TIP pair: start = higher topo-stack idx of the two tip refs; target = other;
            // side = dist(start,target) < 0. max_check = TIP_MFMV_STACK_SIZE (3).
            if seq_tip && cur_idx >= 0 && (has_both_sides || n_past >= 2) {
                let tip0 = order[cur_idx as usize] as usize;
                let tip1 = if has_both_sides { order[cur_idx as usize + 1] as usize } else { order[cur_idx as usize - 1] as usize };
                let (start, tgt) = if rev_topo[tip0] > rev_topo[tip1] { (tip0, tip1) } else { (tip1, tip0) };
                let side = (gpd(refpoc[start], refpoc[tgt]) < 0) as usize;
                add_proc!(start as i32, tgt as i8, side, 3);
            }
            // (b) group loop: nearest/2nd-nearest past+future, ordered by interp-ref distance.
            for g in 0..2i32 {
                if !crate::av2_recon::work_tick("refmvs:1563") { break; }
                let mut past_idx: i32 = if cur_idx >= g { cur_idx - g } else { -1 };
                // has_future_ref(ref): the ref's own list holds a frame AFTER the ref.
                if past_idx >= 0 && !have_ref_sign[order[past_idx as usize] as usize][1] { past_idx = -1; }
                let mut fut_idx: i32 = if cur_idx < n_ref as i32 - g - 1 { cur_idx + 1 + g } else { -1 };
                if fut_idx >= 0 && !have_ref_sign[order[fut_idx as usize] as usize][0] { fut_idx = -1; }
                // get_dist_to_closest_interp_ref: -1 when the slot is missing (avm passes -1
                // through); abs_closest_ref returns 0xff (== avm INT_MAX ordering) when none.
                let pd: i32 = if past_idx >= 0 {
                    let r = order[past_idx as usize] as usize;
                    if elig[r] { abs_closest_ref(&ref2ref[r], &ref2cur[r], false) } else { 0xff }
                } else { -1 };
                let fd: i32 = if fut_idx >= 0 {
                    let r = order[fut_idx as usize] as usize;
                    if elig[r] { abs_closest_ref(&ref2ref[r], &ref2cur[r], true) } else { 0xff }
                } else { -1 };
                let pi: i32 = if past_idx >= 0 { order[past_idx as usize] as i32 } else { -1 };
                let fi: i32 = if fut_idx >= 0 { order[fut_idx as usize] as i32 } else { -1 };
                if fd < pd {
                    add_proc!(fi, -1, 0, 3);
                    add_proc!(pi, -1, 1, 3);
                } else {
                    add_proc!(pi, -1, 1, 3);
                    add_proc!(fi, -1, 0, 3);
                }
            }
            // (c) the two nearest past refs again, side 0.
            if cur_idx >= 0 { add_proc!(order[cur_idx as usize] as i32, -1, 0, 3); }
            if cur_idx >= 1 { add_proc!(order[cur_idx as usize - 1] as i32, -1, 0, 3); }
            // (d) reverse topo-stack leftovers (ri > 0), natural side then the opposite;
            // max_check = MFMV_STACK_SIZE (4).
            for ri in (1..topo_cnt).rev() {
                if !crate::av2_recon::work_tick("refmvs:1594") { break; }
                let r = topo_order[ri] as usize;
                let side = (gpd(refpoc[r], poc) < 0) as usize;
                add_proc!(r as i32, -1, side, 4);
                add_proc!(r as i32, -1, 1 - side, 4);
            }
            // per-mfmv scale arrays (refmvs.c:2247)
            for n in 0..t.n_mfmvs {
                if !crate::av2_recon::work_tick("refmvs:1601") { break; }
                let rref = t.mfmv[n].0 as usize;
                let rpoc = refpoc[rref];
                let diff1 = gpd(rpoc, poc);
                if diff1.abs() > 31 {
                    t.mfmv_ref2cur[n] = INVALID_REF2CUR;
                } else {
                    t.mfmv_ref2cur[n] = diff1;
                    for m in 0..7 {
                        let rrpoc = ref_ref_poc[rref][m];
                        let diff2 = gpd(rpoc, rrpoc);
                        t.mfmv_ref2ref[n][m] = if (diff2 + 31) as u32 <= 62 { diff2 } else { 0 };
                        let mut l = 7;
                        for ll in 0..7 {
                            if rrpoc == refpoc[ll] { l = ll; break; }
                        }
                        t.mfmv_ref2idx[n][m] = if l == 7 { -1 } else { l as i8 };
                        let d1 = t.mfmv_ref2cur[n];
                        let d2 = t.mfmv_ref2ref[n][m];
                        let dv = DIV_MULT[(d2.abs().min(31)) as usize];
                        t.mfmv_ref2sf[n][m][0] = d1.abs().min(31) * dv;
                        if (d1 < 0) != (d2 < 0) { t.mfmv_ref2sf[n][m][0] = -t.mfmv_ref2sf[n][m][0]; }
                        let d3 = d1 - d2;
                        t.mfmv_ref2sf[n][m][1] = d3.abs().min(31) * dv;
                        if (d3 < 0) != (d2 > 0) { t.mfmv_ref2sf[n][m][1] = -t.mfmv_ref2sf[n][m][1]; }
                    }
                }
            }
        }
    }
    t.use_ref_frame_mvs = t.n_mfmvs > 0;
    // env MFMVDBG=<poc>: dump the mfmv source list in dav's [MFMV] probe format (setup A/B).
    if std::env::var("MFMVDBG").map_or(false, |v| v == format!("{}", gpd(poc, 0))) {
        crate::dlog!("[MFMV] poc={} n={}", gpd(poc, 0), t.n_mfmvs);
        for n in 0..t.n_mfmvs {
            if !crate::av2_recon::work_tick("refmvs:1635") { break; }
            crate::dlog!("[MFMV] src{n} ref={} side=0 ref2cur={} rpoc={}", t.mfmv[n].0,
                t.mfmv_ref2cur[n], gpd(refpoc[t.mfmv[n].0 as usize], 0));
            for m in 0..3 {
                crate::dlog!("[MFMV]  r2r[{n}][{m}]={} idx={} sf={},{}", t.mfmv_ref2ref[n][m],
                    t.mfmv_ref2idx[n][m], t.mfmv_ref2sf[n][m][0], t.mfmv_ref2sf[n][m][1]);
            }
        }
        crate::dlog!("[MFMV] have_field={:?} refcnt={:?} tip_ref={:?} n_ref={n_ref} refidx={:?} refpoc={:?}",
            &have_field[..n_ref], &refcnt[..n_ref], t.tip_ref, &refidx[..n_ref], &refpoc[..n_ref]);
        crate::dlog!("[MFMV] tip_mode={} delta={} sf={:?} hole_fill={} mv_traj={} use_rfm={}",
            t.tip_frame_mode, t.tip_delta, t.tip_sf, t.tip_hole_fill, t.mv_traj, t.use_ref_frame_mvs);
    }
    // window allocations: rp_proj (sbsz8+2)=10 rows (base offset 2), rp_traj/rp_map sbsz8=8 rows.
    let stride = t.stride;
    // Window rows scale with the SB size (dav sbsz8 = rf->sbsz>>1 = 8<<sb128); the mfmv_*
    // source constants stay at 8/4 while tmvp_sample_step == 1.
    let sbsz8w = crate::av2_recon::sb_step4() / 2;
    RP_PROJ.with(|c| {
        let mut b = c.borrow_mut();
        *b = vec![(Mv { y: INVALID_MV_I32, x: 0 }, 0); (sbsz8w + 2) * stride];
    });
    RP_TRAJ.with(|c| {
        let mut b = c.borrow_mut();
        for v in b.iter_mut() {
            if !crate::av2_recon::work_tick("refmvs:1659") { break; }
            *v = vec![Mv { y: INVALID_MV_I32, x: 0 }; sbsz8w * stride];
        }
    });
    RP_MAP.with(|c| {
        let mut b = c.borrow_mut();
        *b = vec![vec![(-128i8, -128i8); sbsz8w * stride]; 21]; // [k(3)][ref(7)] flattened k*7+ref
    });
    TMVS.with(|c| *c.borrow_mut() = t);
}

#[inline]
fn apply_sign_i32(v: i32, s: i32) -> i32 { if s < 0 { -v } else { v } }

/// dav2d `dav2d_refmvs_load_tmvs` (refmvs.c:1760), single-tile/sample_step=1/64px-SB shape:
/// project the mfmv sources' saved fields into RP_PROJ (+ RP_TRAJ/RP_MAP when mv_traj) for
/// SB row `row_start8..row_end8` (8px units). Call once per SB row before its blocks decode.
pub fn load_tmvs(row_start8: i32, row_end8: i32) {
    let t = TMVS.with(|c| c.borrow().clone());
    if !t.valid {
        return;
    }
    let stride = t.stride as i32;
    let sbsz8 = (crate::av2_recon::sb_step4() / 2) as i32;
    // sample_step-dependent mfmv geometry (dav refmvs_init_frame, refmvs.c:2001-2004):
    // mfmv_sb128 = sb128 && step>1; k_shift = 3+mfmv_sb128; mfmv_sbsz8 = 8<<mfmv_sb128;
    // mfmv_edge = mfmv_sbsz8 >> (step==1).
    let step = t.sample_step.max(1);
    let mfmv_sb128 = (crate::av2_recon::sb_step4() == 32 && step > 1) as i32;
    let mfmv_sbsz8 = 8i32 << mfmv_sb128;
    let mfmv_edge = mfmv_sbsz8 >> (step == 1) as i32;
    let shift = 3i32 + mfmv_sb128;
    let smask = !(step - 1);
    let row_end8 = row_end8.min(t.ih8);
    let col_start8 = 0i32;
    let col_end8 = t.iw8;
    let col_start8i = (col_start8 - mfmv_edge).max(0);
    let col_end8i = (col_end8 + mfmv_edge).min(t.iw8);
    RP_PROJ.with(|proj_c| {
        RP_TRAJ.with(|traj_c| {
            RP_MAP.with(|map_c| {
                let mut proj = proj_c.borrow_mut();
                let mut traj = traj_c.borrow_mut();
                let mut map = map_c.borrow_mut();
                let base = 2 * stride as usize; // dav poffset = 2*stride (non-frame-threading)
                // carry the last 2 rows of the previous window to rows -2/-1
                for r in 0..2usize {
                    if !crate::av2_recon::work_tick("refmvs:1705") { break; }
                    for x in col_start8 as usize..(col_end8 as usize).min(stride as usize) {
                        if !crate::av2_recon::work_tick("refmvs:1706") { break; }
                        // HARDENING: corrupt dims vs the projection window
                        let (di, si) = (r * stride as usize + x, base + ((sbsz8 as usize - 2 + r) * stride as usize) + x);
                        if di >= proj.len() || si >= proj.len() { continue; }
                        proj[di] = proj[si];
                    }
                }
                // INVALID-init the window rows
                for y in row_start8..row_end8 {
                    if !crate::av2_recon::work_tick("refmvs:1714") { break; }
                    let py = base + ((y & (sbsz8 - 1)) * stride) as usize;
                    for x in col_start8 as usize..col_end8 as usize {
                        if !crate::av2_recon::work_tick("refmvs:1716") { break; }
                        if py + x >= proj.len() { continue; } // HARDENING
                        proj[py + x].0 = Mv { y: INVALID_MV_I32, x: 0 };
                    }
                }
                if t.mv_traj {
                    for n in 0..7usize {
                        if !crate::av2_recon::work_tick("refmvs:1722") { break; }
                        for y in row_start8..row_end8 {
                            if !crate::av2_recon::work_tick("refmvs:1723") { break; }
                            let py = ((y & (sbsz8 - 1)) * stride) as usize;
                            for x in col_start8 as usize..col_end8 as usize {
                                if !crate::av2_recon::work_tick("refmvs:1725") { break; }
                                if py + x < traj[n].len() { traj[n][py + x] = Mv { y: INVALID_MV_I32, x: 0 }; }
                            }
                        }
                        let mask = mfmv_sbsz8 - 1;
                        for k in -1i32..=1 {
                            if !crate::av2_recon::work_tick("refmvs:1730") { break; }
                            let x_start = (col_start8 - k * mfmv_sbsz8).max(0);
                            let x_end = (((col_end8 + mask) & !mask) - k * mfmv_sbsz8).min(t.iw8);
                            let mi = ((k + 1) * 7) as usize + n;
                            for y in row_start8..row_end8 {
                                if !crate::av2_recon::work_tick("refmvs:1734") { break; }
                                let py = ((y & (sbsz8 - 1)) * stride) as usize;
                                for x in x_start..x_end {
                                    if !crate::av2_recon::work_tick("refmvs:1736") { break; }
                                    let o = py + x as usize;
                                    if o >= map[mi].len() { continue; } // HARDENING
                                    map[mi][o] = (-128, -128);
                                }
                            }
                        }
                    }
                }
                // main projection loop
                let refidx = crate::av2_recon::CUR_FRAME_REFIDX.with(|c| c.get()).1;
                RP_REF.with(|rr| {
                    let slots = rr.borrow();
                    let col_start8_shifted = col_start8 >> shift;
                    let col_end8_shifted = (col_end8 - 1) >> shift;
                    for n in 0..t.n_mfmvs {
                        if !crate::av2_recon::work_tick("refmvs:1751") { break; }
                        if t.mfmv_ref2cur[n] == INVALID_REF2CUR {
                            continue;
                        }
                        let (rref, rtgt, ref_sign) = t.mfmv[n];
                        let field = match slots[refidx[rref as usize] as usize].as_ref() {
                            Some(f) if !f.cells.is_empty() => f,
                            _ => continue,
                        };
                        for y in (row_start8..row_end8).step_by(step as usize) {
                            if !crate::av2_recon::work_tick("refmvs:1760") { break; }
                            for x in (col_start8i..col_end8i).step_by(step as usize) {
                                if !crate::av2_recon::work_tick("refmvs:1761") { break; }
                                // saved field cell (whole-frame layout)
                                if y >= field.h8 as i32 || x >= field.w8 as i32 {
                                    continue;
                                }
                                let rb = field.cells[(y * field.w8 as i32 + x) as usize];
                                let b_ref = if ref_sign == 1 { rb.ref_.1 } else { rb.ref_.0 };
                                if b_ref == -1 {
                                    continue;
                                }
                                let ref2idx = t.mfmv_ref2idx[n][b_ref as usize];
                                let b_mv = dequantize_mv(if ref_sign == 1 { rb.qmv[1] } else { rb.qmv[0] });
                                if b_mv.y == INVALID_MV_I32 {
                                    continue;
                                }
                                if t.mv_traj && ref2idx != -1 {
                                    check_traj_intersect(&t, &mut traj, &mut map, rref as usize, ref2idx as usize,
                                                         y, x, b_mv, col_start8_shifted, col_end8_shifted,
                                                         stride, sbsz8, mfmv_sbsz8, mfmv_edge, shift, smask);
                                }
                                let ref2ref = t.mfmv_ref2ref[n][b_ref as usize];
                                if ref2ref == 0 || ((ref2ref < 0) as u8) != ref_sign {
                                    continue;
                                }
                                let mv1 = scale_mv(b_mv, -t.mfmv_ref2sf[n][b_ref as usize][0]);
                                let mut y1 = y - apply_sign_i32(mv1.y.abs() >> 6, mv1.y);
                                if y1 < 0 || y1 >= t.ih8 {
                                    continue;
                                }
                                y1 &= smask;
                                let mut x1 = x - apply_sign_i32(mv1.x.abs() >> 6, mv1.x);
                                if x1 < col_start8 || x1 >= col_end8 {
                                    continue;
                                }
                                x1 &= smask;
                                let y_proj_start = y1 & !(mfmv_sbsz8 - 1);
                                let y_proj_end = (y_proj_start + mfmv_sbsz8).min(row_end8);
                                if y < y_proj_start || y >= y_proj_end {
                                    continue;
                                }
                                let x_sb_align = x1 & !(mfmv_sbsz8 - 1);
                                let x_proj_start = (x_sb_align - mfmv_edge).max(0);
                                let x_proj_end = (x_sb_align + mfmv_sbsz8 + mfmv_edge).min(t.iw8);
                                if x < x_proj_start || x >= x_proj_end {
                                    continue;
                                }
                                let pos1 = (base as i32 + (y1 & (sbsz8 - 1)) * stride + x1) as usize;
                                if proj[pos1].0.y != INVALID_MV_I32
                                    && (rtgt == -1 || ref2idx != rtgt || proj[pos1].1 == ref2ref.abs())
                                {
                                    continue;
                                }
                                if t.mv_traj {
                                    let k1 = (x1 >> shift) - (x >> shift);
                                    let pos = ((y & (sbsz8 - 1)) * stride + x) as usize;
                                    let tpos1 = ((y1 & (sbsz8 - 1)) * stride + x1) as usize;
                                    traj[rref as usize][tpos1] = Mv { y: mv1.y.clamp(-2047, 2047), x: mv1.x.clamp(-2047, 2047) };
                                    if (-1..=1).contains(&k1) {
                                        map[(((k1 + 1) * 7) + rref as i32) as usize][pos] = ((y1 - y) as i8, (x1 - x) as i8);
                                    }
                                    if ref2idx >= 0 {
                                        let mv2 = scale_mv(b_mv, t.mfmv_ref2sf[n][b_ref as usize][1]);
                                        traj[ref2idx as usize][tpos1] = Mv { y: mv2.y.clamp(-2047, 2047), x: mv2.x.clamp(-2047, 2047) };
                                        let y2 = y + apply_sign_i32(b_mv.y.abs() >> 6, b_mv.y);
                                        if y2 >= y_proj_start && y2 < y_proj_end {
                                            let y2 = y2 & smask;
                                            let x2 = x + apply_sign_i32(b_mv.x.abs() >> 6, b_mv.x);
                                            if x2 >= x_proj_start && x2 < x_proj_end {
                                                let x2 = x2 & smask;
                                                let pos2 = ((y2 & (sbsz8 - 1)) * stride + x2) as usize;
                                                let k2 = (x1 >> shift) - (x2 >> shift);
                                                if (-1..=1).contains(&k2) {
                                                    map[(((k2 + 1) * 7) + ref2idx as i32) as usize][pos2] = ((y1 - y2) as i8, (x1 - x2) as i8);
                                                }
                                            }
                                        }
                                    }
                                }
                                let mut bm = b_mv;
                                if ref2ref < 0 {
                                    bm.y = -bm.y;
                                    bm.x = -bm.x;
                                }
                                proj[pos1] = (bm, ref2ref.abs());
                            }
                        }
                    }
                });
                // TIP frames: convert rp_proj into TIP-relative MVs (+ hole fill / smoothen).
                let stepu = step as usize;
                if t.tip_frame_mode != 0 {
                    for y in (row_start8..row_end8).step_by(stepu) {
                        if !crate::av2_recon::work_tick("refmvs:1852") { break; }
                        let py = base + ((y & (sbsz8 - 1)) * stride) as usize;
                        for x in (col_start8 as usize..col_end8 as usize).step_by(stepu) {
                            if !crate::av2_recon::work_tick("refmvs:1854") { break; }
                            let (m, r) = proj[py + x];
                            if m.y == INVALID_MV_I32 {
                                continue;
                            }
                            proj[py + x] = (mv_projection_t(m, t.tip_delta, r, -2047, 2047), t.tip_delta);
                        }
                    }
                    if t.tip_hole_fill {
                        // fill_holes (refmvs.c:1388), offsets/steps in sample_step units.
                        for sx in (col_start8..col_end8).step_by(mfmv_sbsz8 as usize) {
                            if !crate::av2_recon::work_tick("refmvs:1864") { break; }
                            let xend = col_end8.min(sx + mfmv_sbsz8);
                            for y in (row_start8..row_end8).step_by(stepu) {
                                if !crate::av2_recon::work_tick("refmvs:1866") { break; }
                                let ystart = y & !(mfmv_sbsz8 - 1);
                                let yend = (ystart + mfmv_sbsz8).min(row_end8);
                                let pb = base as i32 + (y & (sbsz8 - 1)) * stride;
                                for x in (sx..xend).step_by(stepu) {
                                    if !crate::av2_recon::work_tick("refmvs:1870") { break; }
                                    let pos = (pb + x) as usize;
                                    let m = proj[pos].0;
                                    if m.y == INVALID_MV_I32 {
                                        continue;
                                    }
                                    let so = stepu;
                                    let ss = stepu * stride as usize;
                                    if x - step >= sx && proj[pos - so].0.y == INVALID_MV_I32 {
                                        proj[pos - so] = (m, t.tip_delta);
                                    }
                                    if x + step < xend && proj[pos + so].0.y == INVALID_MV_I32 {
                                        proj[pos + so] = (m, t.tip_delta);
                                    }
                                    if y - step >= ystart && proj[pos - ss].0.y == INVALID_MV_I32 {
                                        proj[pos - ss] = (m, t.tip_delta);
                                    }
                                    if y + step < yend && proj[pos + ss].0.y == INVALID_MV_I32 {
                                        proj[pos + ss] = (m, t.tip_delta);
                                    }
                                }
                            }
                        }
                        // smoothen (refmvs.c:1425): 5-tap average, written one sampled row LATE.
                        const IDIV: [i64; 5] = [65536, 32768, 21845, 16384, 13107];
                        for sx in (col_start8..col_end8).step_by(mfmv_sbsz8 as usize) {
                            if !crate::av2_recon::work_tick("refmvs:1895") { break; }
                            let xend = col_end8.min(sx + mfmv_sbsz8);
                            let mut mv_line = [Mv { y: INVALID_MV_I32, x: 0 }; 32];
                            let mut first_line = true;
                            let mut last_y = row_start8;
                            for y in (row_start8..row_end8).step_by(stepu) {
                                if !crate::av2_recon::work_tick("refmvs:1900") { break; }
                                let ystart = y & !(mfmv_sbsz8 - 1);
                                let yend = (ystart + mfmv_sbsz8).min(row_end8);
                                let pb = base as i32 + (y & (sbsz8 - 1)) * stride;
                                for x in (sx..xend).step_by(stepu) {
                                    if !crate::av2_recon::work_tick("refmvs:1904") { break; }
                                    let pos = (pb + x) as usize;
                                    let mut sum_x = 0i64;
                                    let mut sum_y = 0i64;
                                    let mut sum_n = 0usize;
                                    {
                                        let so = stepu;
                                        let ss = stepu * stride as usize;
                                        let mut add = |p: usize| {
                                            if proj[p].0.y != INVALID_MV_I32 {
                                                sum_x += proj[p].0.x as i64;
                                                sum_y += proj[p].0.y as i64;
                                                sum_n += 1;
                                            }
                                        };
                                        add(pos);
                                        if x - step >= sx { add(pos - so); }
                                        if x + step < xend { add(pos + so); }
                                        if y - step >= ystart { add(pos - ss); }
                                        if y + step < yend { add(pos + ss); }
                                    }
                                    if !first_line {
                                        proj[pos - stepu * stride as usize] = (mv_line[(x - sx) as usize], t.tip_delta);
                                    }
                                    if sum_n > 0 {
                                        mv_line[(x - sx) as usize] = Mv {
                                            y: ((sum_y * IDIV[sum_n - 1] + 0x8000 - (sum_y < 0) as i64) >> 16) as i32,
                                            x: ((sum_x * IDIV[sum_n - 1] + 0x8000 - (sum_x < 0) as i64) >> 16) as i32,
                                        };
                                    } else {
                                        mv_line[(x - sx) as usize].y = INVALID_MV_I32;
                                    }
                                }
                                first_line = false;
                                last_y = y;
                            }
                            if !first_line {
                                let pb = base as i32 + (last_y & (sbsz8 - 1)) * stride;
                                for x in (sx..xend).step_by(stepu) {
                                    if !crate::av2_recon::work_tick("refmvs:1942") { break; }
                                    proj[(pb + x) as usize] = (mv_line[(x - sx) as usize], t.tip_delta);
                                }
                            }
                        }
                    }
                }
                // sample_step=2: fill the skipped odd rows/cols (dav refmvs.c:1921
                // fill_gap_traj per ref + fill_gap_proj). The projected even cells seed
                // right/bottom/bottom-right averages into the +1 neighbours.
                if step > 1 {
                    if t.mv_traj {
                        for n in 0..7usize {
                            if !crate::av2_recon::work_tick("refmvs:1954") { break; }
                            let tj = &mut traj[n];
                            for sx in (col_start8..col_end8).step_by(mfmv_sbsz8 as usize) {
                                if !crate::av2_recon::work_tick("refmvs:1956") { break; }
                                let xend = col_end8.min(sx + mfmv_sbsz8);
                                for y in (row_start8..row_end8).step_by(2) {
                                    if !crate::av2_recon::work_tick("refmvs:1958") { break; }
                                    let ystart = y & !(mfmv_sbsz8 - 1);
                                    let yend = (ystart + mfmv_sbsz8).min(row_end8);
                                    let pb = (y & (sbsz8 - 1)) * stride;
                                    for x in (sx..xend).step_by(2) {
                                        if !crate::av2_recon::work_tick("refmvs:1962") { break; }
                                        let pos = (pb + x) as usize;
                                        let s = stride as usize;
                                        let m = tj[pos];
                                        if m.y == INVALID_MV_I32 {
                                            continue;
                                        }
                                        let (mut sum_y, mut sum_x, mut sum_n) = (m.y, m.x, 1i32);
                                        let have_bottom = y + 2 < yend;
                                        if have_bottom && tj[pos + 2 * s].y != INVALID_MV_I32 {
                                            let bm = tj[pos + 2 * s];
                                            sum_x += bm.x;
                                            sum_y += bm.y;
                                            tj[pos + s] = Mv { y: (sum_y + (sum_y > 0) as i32) >> 1, x: (sum_x + (sum_x > 0) as i32) >> 1 };
                                            sum_n += 1;
                                        } else {
                                            tj[pos + s] = m;
                                        }
                                        let have_right = x + 2 < xend;
                                        if have_right && tj[pos + 2].y != INVALID_MV_I32 {
                                            let rm = tj[pos + 2];
                                            sum_x += rm.x;
                                            let mx = m.x + rm.x;
                                            sum_y += rm.y;
                                            let my = m.y + rm.y;
                                            tj[pos + 1] = Mv { y: (my + (my > 0) as i32) >> 1, x: (mx + (mx > 0) as i32) >> 1 };
                                            sum_n += 1;
                                        } else {
                                            tj[pos + 1] = m;
                                        }
                                        if have_right && have_bottom && tj[pos + 2 * (1 + s)].y != INVALID_MV_I32 {
                                            let brm = tj[pos + 2 * (1 + s)];
                                            sum_x += brm.x;
                                            sum_y += brm.y;
                                            sum_n += 1;
                                        }
                                        tj[pos + 1 + s] = match sum_n {
                                            1 => m,
                                            2 => Mv { y: (sum_y + (sum_y > 0) as i32) >> 1, x: (sum_x + (sum_x > 0) as i32) >> 1 },
                                            3 => Mv { y: (sum_y * 85 + 128 - (sum_y < 0) as i32) >> 8, x: (sum_x * 85 + 128 - (sum_x < 0) as i32) >> 8 },
                                            _ => Mv { y: (sum_y + 1 + (sum_y > 0) as i32) >> 2, x: (sum_x + 1 + (sum_x > 0) as i32) >> 2 },
                                        };
                                    }
                                }
                            }
                        }
                    }
                    // fill_gap_proj (refmvs.c:1487): like traj, but neighbours are first
                    // re-projected to THIS cell's ref delta before averaging.
                    for sx in (col_start8..col_end8).step_by(mfmv_sbsz8 as usize) {
                        if !crate::av2_recon::work_tick("refmvs:2011") { break; }
                        let xend = col_end8.min(sx + mfmv_sbsz8);
                        for y in (row_start8..row_end8).step_by(2) {
                            if !crate::av2_recon::work_tick("refmvs:2013") { break; }
                            let ystart = y & !(mfmv_sbsz8 - 1);
                            let yend = (ystart + mfmv_sbsz8).min(row_end8);
                            let pb = base as i32 + (y & (sbsz8 - 1)) * stride;
                            for x in (sx..xend).step_by(2) {
                                if !crate::av2_recon::work_tick("refmvs:2017") { break; }
                                let pos = (pb + x) as usize;
                                let s = stride as usize;
                                let (m, ref_off) = proj[pos];
                                if m.y == INVALID_MV_I32 {
                                    continue;
                                }
                                let (mut sum_y, mut sum_x, mut sum_n) = (m.y, m.x, 1i32);
                                let have_right = x + 2 < xend;
                                if have_right && proj[pos + 2].0.y != INVALID_MV_I32 {
                                    let (rm0, rref) = proj[pos + 2];
                                    let rm = mv_projection_t(rm0, ref_off, rref, -2047, 2047);
                                    sum_x += rm.x;
                                    sum_y += rm.y;
                                    proj[pos + 1] = (Mv { y: (sum_y + (sum_y > 0) as i32) >> 1, x: (sum_x + (sum_x > 0) as i32) >> 1 }, ref_off);
                                    sum_n += 1;
                                } else {
                                    proj[pos + 1] = proj[pos];
                                }
                                let have_bottom = y + 2 < yend;
                                if have_bottom && proj[pos + 2 * s].0.y != INVALID_MV_I32 {
                                    let (bm0, bref) = proj[pos + 2 * s];
                                    let bm = mv_projection_t(bm0, ref_off, bref, -2047, 2047);
                                    sum_x += bm.x;
                                    let mx = m.x + bm.x;
                                    sum_y += bm.y;
                                    let my = m.y + bm.y;
                                    proj[pos + s] = (Mv { y: (my + (my > 0) as i32) >> 1, x: (mx + (mx > 0) as i32) >> 1 }, ref_off);
                                    sum_n += 1;
                                } else {
                                    proj[pos + s] = proj[pos];
                                }
                                if have_right && have_bottom && proj[pos + 2 * (1 + s)].0.y != INVALID_MV_I32 {
                                    let (brm0, brref) = proj[pos + 2 * (1 + s)];
                                    let brm = mv_projection_t(brm0, ref_off, brref, -2047, 2047);
                                    sum_x += brm.x;
                                    sum_y += brm.y;
                                    sum_n += 1;
                                }
                                let dm = match sum_n {
                                    1 => m,
                                    2 => Mv { y: (sum_y + (sum_y > 0) as i32) >> 1, x: (sum_x + (sum_x > 0) as i32) >> 1 },
                                    3 => Mv { y: (sum_y * 85 + 128 - (sum_y < 0) as i32) >> 8, x: (sum_x * 85 + 128 - (sum_x < 0) as i32) >> 8 },
                                    _ => Mv { y: (sum_y + 1 + (sum_y > 0) as i32) >> 2, x: (sum_x + 1 + (sum_x > 0) as i32) >> 2 },
                                };
                                proj[pos + 1 + s] = (dm, ref_off);
                            }
                        }
                    }
                }
            });
        });
    });
}

/// Dump the just-projected window rows in avm's `[TPLALL] r c row,col,offset` coordinates so the
/// whole-frame temporal grid can be diffed line-for-line against avmdec (env TPLALL).
pub fn tpl_dump_window(row_start8: i32, row_end8: i32) {
    if std::env::var("TPLALL").is_err() {
        return;
    }
    let t = TMVS.with(|c| c.borrow().clone());
    if !t.valid {
        return;
    }
    let (stride, sbsz8) = (t.stride as i32, crate::av2_recon::sb_step4() as i32 / 2);
    let row_end8 = row_end8.min(t.ih8);
    RP_PROJ.with(|pc| {
        let p = pc.borrow();
        let base = 2 * stride;
        for y in row_start8..row_end8 {
            for x in 0..stride {
                let i = (base + (y & (sbsz8 - 1)) * stride + x) as usize;
                if let Some((mv, rf)) = p.get(i) {
                    eprintln!("[TPLALL] r={y} c={x} {},{},{}", mv.y, mv.x, rf);
                }
            }
        }
    });
}

/// Debug dump of the projected window (env TMVSDBG), mirrors dav's [TPROJ]/[TTRAJn] probe.
pub fn tmvs_dump(row_start8: i32, row_end8: i32) {
    let t = TMVS.with(|c| c.borrow().clone());
    if !t.valid {
        return;
    }
    let stride = t.stride;
    let row_end8 = row_end8.min(t.ih8);
    RP_PROJ.with(|pc| {
        RP_TRAJ.with(|tc| {
            let proj = pc.borrow();
            let traj = tc.borrow();
            for y in row_start8..row_end8 {
                if !crate::av2_recon::work_tick("refmvs:2084") { break; }
                for x in 0..t.iw8 {
                    if !crate::av2_recon::work_tick("refmvs:2085") { break; }
                    let pp = ((y & 7) as usize) * stride + x as usize;
                    let (m, r) = proj[2 * stride + pp];
                    crate::dlog!("[TPROJ] y={y} x={x} mv={},{} ref={r}", if m.y == INVALID_MV_I32 { -32768 } else { m.y }, m.x);
                    if t.mv_traj {
                        for n in 0..4usize {
                            if !crate::av2_recon::work_tick("refmvs:2090") { break; }
                            let tm = traj[n][pp];
                            crate::dlog!("[TTRAJ{n}] y={y} x={x} mv={},{}", if tm.y == INVALID_MV_I32 { -32768 } else { tm.y }, tm.x);
                        }
                    }
                }
            }
        })
    });
}

/// dav2d `check_traj_intersect` (refmvs.c:1656), sample_step=1.
#[allow(clippy::too_many_arguments)]
fn check_traj_intersect(
    t: &TmvsFrame, traj: &mut [Vec<Mv>; 7], map: &mut [Vec<(i8, i8)>],
    ref1: usize, ref2: usize, y: i32, x: i32, mv_in: Mv,
    col_start8_shifted: i32, col_end8_shifted: i32,
    stride: i32, sbsz8: i32, mfmv_sbsz8: i32, mfmv_edge: i32, shift: i32, smask: i32,
) {
    let pos = ((y & (sbsz8 - 1)) * stride + x) as usize;
    let min_k = (-1).max(col_start8_shifted - (x >> shift));
    let max_k = 1.min(col_end8_shifted - (x >> shift));
    for k in (min_k + 1)..=(max_k + 1) {
        if !crate::av2_recon::work_tick("refmvs:2112") { break; }
        let m1 = map[((k * 7) + ref1 as i32) as usize][pos];
        if m1 == (-128, -128) {
            continue;
        }
        let x1 = x + m1.1 as i32;
        let k1 = (x1 >> shift) - (x >> shift);
        if k1 + 1 != k {
            continue;
        }
        let x_sb_align = x1 & !(mfmv_sbsz8 - 1);
        let x_proj_start = (x_sb_align - mfmv_edge).max(0);
        let x_proj_end = (x_sb_align + mfmv_sbsz8 + mfmv_edge).min(t.iw8);
        if x < x_proj_start || x >= x_proj_end {
            continue;
        }
        let y1 = y + m1.0 as i32;
        let y_proj_start = y1 & !(mfmv_sbsz8 - 1);
        let y_proj_end = (y_proj_start + mfmv_sbsz8).min(t.ih8);
        if y < y_proj_start || y >= y_proj_end {
            continue;
        }
        let pos1 = ((y1 & (sbsz8 - 1)) * stride + x1) as usize;
        if traj[ref2][pos1].y != INVALID_MV_I32 {
            continue;
        }
        let src = traj[ref1][pos1];
        let py = (src.y + mv_in.y).clamp(-2047, 2047);
        let px = (src.x + mv_in.x).clamp(-2047, 2047);
        traj[ref2][pos1] = Mv { y: py, x: px };
        let y2 = y1 + apply_sign_i32(py.abs() >> 6, py);
        let x2 = x1 + apply_sign_i32(px.abs() >> 6, px);
        if x2 < x_proj_start || x2 >= x_proj_end || y2 < y_proj_start || y2 >= y_proj_end {
            continue;
        }
        let (y2, x2) = (y2 & smask, x2 & smask);
        let pos2 = ((y2 & (sbsz8 - 1)) * stride + x2) as usize;
        let k2 = (x1 >> shift) - (x2 >> shift);
        if !(-1..=1).contains(&k2) {
            continue;
        }
        map[((k2 + 1) * 7 + ref2 as i32) as usize][pos2] = ((y1 - y2) as i8, (x1 - x2) as i8);
    }

    // PHASE 2 (dav refmvs.c check_traj_intersect second loop): follow mv_in FORWARD to (y1,x1)
    // and intersect the DST ref's map there, writing back into the SRC ref's traj (mv_src − mv_in).
    let y1 = y + apply_sign_i32(mv_in.y.abs() >> 6, mv_in.y);
    let x1 = x + apply_sign_i32(mv_in.x.abs() >> 6, mv_in.x);
    if y1.min(x1) < 0 || y1 >= t.ih8 || x1 >= t.iw8 {
        return;
    }
    let (y1, x1) = (y1 & smask, x1 & smask);
    let min_k1 = (-1).max(col_start8_shifted - (x1 >> shift));
    let max_k1 = 1.min(col_end8_shifted - (x1 >> shift));
    for k in (min_k1 + 1)..=(max_k1 + 1) {
        if !crate::av2_recon::work_tick("refmvs:2166") { break; }
        let pos1 = ((y1 & (sbsz8 - 1)) * stride + x1) as usize;
        let m1 = map[((k * 7) + ref2 as i32) as usize][pos1];
        if m1 == (-128, -128) {
            continue;
        }
        let x2 = x1 + m1.1 as i32;
        let k2 = (x2 >> shift) - (x1 >> shift);
        if k2 + 1 != k {
            continue;
        }
        let x_sb_align = x2 & !(mfmv_sbsz8 - 1);
        let x_proj_start = (x_sb_align - mfmv_edge).max(0);
        let x_proj_end = (x_sb_align + mfmv_sbsz8 + mfmv_edge).min(t.iw8);
        if x < x_proj_start || x >= x_proj_end || x1 < x_proj_start || x1 >= x_proj_end {
            continue;
        }
        let y2 = y1 + m1.0 as i32;
        let y_proj_start = y2 & !(mfmv_sbsz8 - 1);
        let y_proj_end = (y_proj_start + mfmv_sbsz8).min(t.ih8);
        if y < y_proj_start || y >= y_proj_end || y1 < y_proj_start || y1 >= y_proj_end {
            continue;
        }
        let pos2 = ((y2 & (sbsz8 - 1)) * stride + x2) as usize;
        if traj[ref1][pos2].y != INVALID_MV_I32 {
            continue;
        }
        let src = traj[ref2][pos2];
        let py = (src.y - mv_in.y).clamp(-0xffff, 0xffff);
        let px = (src.x - mv_in.x).clamp(-0xffff, 0xffff);
        traj[ref1][pos2] = Mv { y: py, x: px };
        let y3 = y2 + apply_sign_i32(py.abs() >> 6, py);
        let x3 = x2 + apply_sign_i32(px.abs() >> 6, px);
        if x3 < x_proj_start || x3 >= x_proj_end || y3 < y_proj_start || y3 >= y_proj_end {
            continue;
        }
        let (y3, x3) = (y3 & smask, x3 & smask);
        let pos3 = ((y3 & (sbsz8 - 1)) * stride + x3) as usize;
        let k3 = (x2 >> shift) - (x3 >> shift);
        if !(-1..=1).contains(&k3) {
            continue;
        }
        map[((k3 + 1) * 7 + ref1 as i32) as usize][pos3] = ((y2 - y3) as i8, (x2 - x3) as i8);
    }
}

#[derive(Clone)]
pub struct RefmvBank {
    pub mv: [[Mv; 4]; 9], // [ref class 0..8][ring slot]; class 8 = intrabc (ref[0]=-1) / other
    /// Second MV of a compound bank entry (classes 6/7/8; dav bank.mv[c][i][1]).
    pub mv2: [[Mv; 4]; 9],
    /// The ref pair per slot (shared [4] array, maintained for class-8 pair checks — dav bank.ref[i]).
    pub ref_pair: [(i8, i8); 4],
    /// cwp per compound slot (dav bank.cwp_idx[c-6][i], classes 6/7/8).
    pub cwp: [[i8; 4]; 3],
    pub size: [u8; 9],
    pub idx: [u8; 9],
    pub avail: i32,
    pub hits0: i32,
    pub hits1: i32,
}

impl Default for RefmvBank {
    fn default() -> Self {
        RefmvBank {
            mv: [[Mv::default(); 4]; 9], mv2: [[Mv::default(); 4]; 9],
            ref_pair: [(-1, -1); 4], cwp: [[8; 4]; 3],
            size: [0; 9], idx: [0; 9], avail: 0, hits0: 0, hits1: 0,
        }
    }
}

/// dav2d refmv-bank class for a single-ref/intrabc block (refmvs.c:1229): 0..5 for a normal
/// single-ref, else 8 (intrabc `ref[0]=-1` and other out-of-range). Compound (6/7) not handled.
#[inline]
fn bank_class(ref0: i8) -> usize {
    if (0..=5).contains(&ref0) {
        ref0 as usize
    } else {
        8
    }
}

impl RefmvBank {
    /// dav2d `dav2d_refmvs_bank_update` (refmvs.c:1107): at an SB boundary reset `avail` to
    /// `max(w, 4)`; at a bank-size (`bsz`) boundary add `w`. `sb128` is 0 for 64px SBs (bsz = 2).
    fn update(&mut self, bw4: usize, bh4: usize, by4: usize, bx4: usize, sbsz: usize, sb128: usize) {
        let bsh = 1 + sb128;
        let bsz = 1usize << bsh;
        let w = (1.max(bw4 >> bsh)) * (1.max(bh4 >> bsh));
        if (by4 | bx4) & (sbsz - 1) == 0 {
            // dav2d `dav2d_refmvs_bank_update` (refmvs.c:1112) at the SB corner: hits1=0,
            // avail=max(w,4). hits0 is NOT reset here — it is zeroed (and re-seeded from the
            // above-SB-row) once per SB in `reset_sb`, which fires before the first block.
            self.hits1 = 0;
            self.avail = (w as i32).max(4);
        } else if (by4 | bx4) & (bsz - 1) == 0 {
            self.hits1 = 0;
            self.avail += w as i32;
        }
    }

    /// dav2d `dav2d_refmvs_bank_add` + `refmvs_bank_add` (single-ref): push a decoded block's MV,
    /// gated by `avail`/`hits`, with an LRU move-to-tail if already present.
    pub fn add_block(&mut self, bw4: usize, bh4: usize, by4: usize, bx4: usize, sbsz: usize, sb128: usize, ref0: i8, mv: Mv) {
        self.update(bw4, bh4, by4, bx4, sbsz, sb128);
        if self.hits0 >= 64 || self.hits1 >= 16 || self.avail == 0 {
            return;
        }
        self.hits0 += 1;
        self.hits1 += 1;
        self.avail -= 1;
        self.insert_mv(ref0, mv);
    }

    /// dav2d `dav2d_refmvs_bank_update` called from `splat_intraref` (decode.c:777): an INTRA
    /// block (with luma) refreshes the bank `avail`/`hits1` counters at SB/bsz boundaries WITHOUT
    /// adding any MV. Mine previously skipped this for intra blocks, so `avail` never refilled at
    /// them and drained to 0 — blocking legitimate later inter bank adds (e.g. (52,1)=(216,520)).
    pub fn bank_update_intra(&mut self, bw4: usize, bh4: usize, by4: usize, bx4: usize, sbsz: usize, sb128: usize) {
        self.update(bw4, bh4, by4, bx4, sbsz, sb128);
    }

    /// dav2d `refmvs_bank_add` called from `reset_sb` seeding (refmvs.c:1340): bump hits0 and
    /// insert the above-row block's MV, bypassing the per-block avail/hits1 gate entirely.
    fn add_raw(&mut self, ref0: i8, mv: Mv) {
        self.hits0 += 1;
        self.insert_mv(ref0, mv);
    }

    /// Compound `add_block`: per-block gates then a pair insert (classes 6/7/8).
    #[allow(clippy::too_many_arguments)]
    pub fn add_block_pair(&mut self, bw4: usize, bh4: usize, by4: usize, bx4: usize, sbsz: usize, sb128: usize, ref0: i8, ref1: i8, mv: [Mv; 2], cwp: i8) {
        if ref1 < 0 {
            self.add_block(bw4, bh4, by4, bx4, sbsz, sb128, ref0, mv[0]);
            return;
        }
        self.update(bw4, bh4, by4, bx4, sbsz, sb128);
        if self.hits0 >= 64 || self.hits1 >= 16 || self.avail == 0 {
            return;
        }
        self.hits0 += 1;
        self.hits1 += 1;
        self.avail -= 1;
        self.insert_pair(ref0, ref1, mv, cwp);
    }

    /// Compound `add_raw` (reset_sb seeding path).
    pub fn add_raw_pair(&mut self, ref0: i8, ref1: i8, mv: [Mv; 2], cwp: i8) {
        if ref1 < 0 {
            self.add_raw(ref0, mv[0]);
            return;
        }
        self.hits0 += 1;
        self.insert_pair(ref0, ref1, mv, cwp);
    }

    /// dav2d `refmvs_bank_add` core, COMPOUND classes (refmvs.c:1229-1282): class
    /// `(!ref0 && ref1<=1) ? 6+ref1 : 8`; dedup on BOTH mvs (+ the pair for class 8);
    /// LRU move-to-tail shuffles mv/mv2/cwp (+ ref_pair for class 8).
    fn insert_pair(&mut self, ref0: i8, ref1: i8, mv: [Mv; 2], cwp: i8) {
        let c: usize = if ref0 == 0 && (0..=1).contains(&ref1) { 6 + ref1 as usize } else { 8 };
        let sz = self.size[c] as usize;
        let idx = self.idx[c] as usize;
        let mut n = 0;
        while n < sz {
            let i = (idx + n) & 3;
            if (c < 8 || self.ref_pair[i] == (ref0, ref1))
                && self.mv[c][i] == mv[0]
                && self.mv2[c][i] == mv[1]
            {
                break;
            }
            n += 1;
        }
        if n < sz {
            let to = if sz == 4 { (idx + 3) & 3 } else { sz - 1 };
            let from = (idx + n) & 3;
            if from != to {
                let bak = (self.mv[c][from], self.mv2[c][from], self.ref_pair[from], self.cwp[c - 6][from]);
                let mut n1 = from;
                let mut n2 = (n1 + 1) & 3;
                while n1 != to {
                    self.mv[c][n1] = self.mv[c][n2];
                    self.mv2[c][n1] = self.mv2[c][n2];
                    self.cwp[c - 6][n1] = self.cwp[c - 6][n2];
                    if c == 8 {
                        self.ref_pair[n1] = self.ref_pair[n2];
                    }
                    n1 = n2;
                    n2 = (n2 + 1) & 3;
                }
                self.mv[c][to] = bak.0;
                self.mv2[c][to] = bak.1;
                self.cwp[c - 6][to] = bak.3;
                if c == 8 {
                    self.ref_pair[to] = bak.2;
                }
            }
            return;
        }
        let tgt = if sz == 4 {
            let t = self.idx[c] as usize & 3;
            self.idx[c] = ((self.idx[c] as usize + 1) & 3) as u8;
            t
        } else {
            let t = self.size[c] as usize;
            self.size[c] += 1;
            t
        };
        self.mv[c][tgt] = mv[0];
        self.mv2[c][tgt] = mv[1];
        self.cwp[c - 6][tgt] = cwp;
        if c == 8 {
            self.ref_pair[tgt] = (ref0, ref1);
        }
    }

    /// dav2d `refmvs_bank_add` core (refmvs.c:1229-1282, single-ref subset): dedup-with-move-to-
    /// tail or append `mv` into ref-class `ref0`'s ≤4-entry ring. Shared by `add_block`/`add_raw`.
    fn insert_mv(&mut self, ref0: i8, mv: Mv) {
        // intrabc (ref0 = -1), ref 6 and TIP (7) singles store in class 8 with their ref pair
        // (dav2d refmvs_bank_add: `(unsigned)ref[0] <= 5 ? ref[0] : 8`).
        let c = bank_class(ref0);
        let sz = self.size[c] as usize;
        let idx = self.idx[c] as usize;
        // find an existing equal entry (class 8 also matches the ref pair — dav refmvs.c:1236)
        let mut n = 0;
        while n < sz {
            let i = (idx + n) & 3;
            if (c < 8 || self.ref_pair[i] == (ref0, -1)) && self.mv[c][i] == mv {
                break;
            }
            n += 1;
        }
        if n < sz {
            // move-to-tail (LRU)
            let to = if sz == 4 { (idx + 3) & 3 } else { sz - 1 };
            let from = (idx + n) & 3;
            if from != to {
                let bak = (self.mv[c][from], self.ref_pair[from]);
                let mut n1 = from;
                let mut n2 = (n1 + 1) & 3;
                while n1 != to {
                    self.mv[c][n1] = self.mv[c][n2];
                    if c == 8 {
                        self.ref_pair[n1] = self.ref_pair[n2];
                    }
                    n1 = n2;
                    n2 = (n2 + 1) & 3;
                }
                self.mv[c][to] = bak.0;
                if c == 8 {
                    self.ref_pair[to] = bak.1;
                }
            }
            return;
        }
        // append a new entry
        let tgt = if sz == 4 {
            let t = self.idx[c] as usize & 3;
            self.idx[c] = ((self.idx[c] as usize + 1) & 3) as u8;
            t
        } else {
            let t = self.size[c] as usize;
            self.size[c] += 1;
            t
        };
        self.mv[c][tgt] = mv;
        if c == 8 {
            self.ref_pair[tgt] = (ref0, -1);
        }
    }
}

/// dav2d bank READ (refmvs.c:878-927, single-ref): after the spatial sort, add bank entries
/// (most-recent-first) as weight-0 candidates, deduping vs the stack, until `cnt` reaches `lim`.
#[allow(clippy::too_many_arguments)]
pub fn add_bank_candidates(st: &mut ScanState, bank: &RefmvBank, ref0: i8, ref1: i8, lim: usize, bx4: usize, by4: usize, bw4: usize, bh4: usize, iw8: usize, ih8: usize) {
    // class (dav refmvs.c:879): single 0..5 / 8 (intrabc, out-of-range); compound
    // (!ref0 && ref1<2) → 6+ref1, else 8 (with a pair check per slot).
    let comp = ref1 >= 0;
    let c: usize = if !comp {
        bank_class(ref0)
    } else if ref0 == 0 && (0..=1).contains(&ref1) {
        6 + ref1 as usize
    } else {
        8
    };
    let sz = bank.size[c] as usize;
    let idx = bank.idx[c] as usize;
    let start = sz + idx;
    let mut n = 0;
    while n < sz && st.cnt < lim {
        let bank_idx = (start.wrapping_sub(1).wrapping_sub(n)) & 3;
        if c == 8 && bank.ref_pair[bank_idx] != (ref0, ref1) {
            n += 1;
            continue;
        }
        let mv = bank.mv[c][bank_idx];
        let mv1 = bank.mv2[c][bank_idx];
        // dedup vs the stack (both mvs when compound — dav refmvs.c:891)
        let mut dup = false;
        if st.iter_cntr < 16 {
            for m in 0..st.cnt {
                if !crate::av2_recon::work_tick("refmvs:2470") { break; }
                if st.mvstack[m].mv[0] == mv && (!comp || st.mvstack[m].mv[1] == mv1) {
                    st.iter_cntr += m as i32 + 1;
                    dup = true;
                    break;
                }
            }
            if !dup {
                st.iter_cntr += st.cnt as i32;
            }
        }
        if !dup {
            // range check on EACH mv of the pair (dav refmvs.c:905)
            let mut ok = true;
            for m in [mv, mv1].iter().take(1 + comp as usize) {
                if !crate::av2_recon::work_tick("refmvs:2484") { break; }
                let rx = bx4 as i32 * 4 + apply_sign((m.x.abs()) >> 3, m.x);
                let ry = by4 as i32 * 4 + apply_sign((m.y.abs()) >> 3, m.y);
                if rx <= -(bw4 as i32) * 4 || ry <= -(bh4 as i32) * 4 || rx >= iw8 as i32 * 8 || ry >= ih8 as i32 * 8 {
                    ok = false;
                    break;
                }
            }
            if ok {
                let last = st.cnt;
                st.mvstack[last].mv[0] = mv;
                st.mvstack[last].mv[1] = mv1;
                st.mvstack[last].weight = 0;
                st.mvstack[last].cwp = if comp { bank.cwp[c.max(6) - 6][bank_idx] } else { 8 };
                st.mvstack[last].y_off = 0;
                st.mvstack[last].x_off = 0;
                st.cnt = last + 1;
            }
        }
        n += 1;
    }
}

/// The warp-model bank (dav2d `rt->warp`, single-ref): a per-ref ring of ≤4 recent warp matrices,
/// read in `refmvs_find_warp`'s fill. Each warp block pushes its reconstructed matrix.
#[derive(Clone)]
pub struct RefmvWarpBank {
    pub mat: [[[i32; 6]; 4]; 7],
    pub size: [u8; 7],
    pub idx: [u8; 7],
    pub hits: i32,
}

impl Default for RefmvWarpBank {
    fn default() -> Self {
        RefmvWarpBank { mat: [[[0; 6]; 4]; 7], size: [0; 7], idx: [0; 7], hits: 0 }
    }
}

impl RefmvWarpBank {
    /// dav2d `dav2d_refmvs_warp_add` (refmvs.c:1146): LRU push of a warp matrix for `ref`.
    pub fn add(&mut self, ref0: i8, m: [i32; 6]) {
        if self.hits >= 64 || !(0..7).contains(&(ref0 as i32)) {
            return;
        }
        self.hits += 1;
        let r = ref0 as usize;
        let sz = self.size[r] as usize;
        let idx = self.idx[r] as usize;
        // dav2d dedups on the SHEAR only (matrix[2..6], 4 ints) — not the full matrix. Two blocks
        // with the same warp shear but different translation (m[0]/m[1]) collapse to one bank entry.
        let mut n = 0;
        while n < sz {
            if self.mat[r][(idx + n) & 3][2..6] == m[2..6] {
                break;
            }
            n += 1;
        }
        if n < sz {
            let to = if sz == 4 { (idx + 3) & 3 } else { sz - 1 };
            let from = (idx + n) & 3;
            if from != to {
                let bak = self.mat[r][from];
                let mut n1 = from;
                let mut n2 = (n1 + 1) & 3;
                while n1 != to {
                    self.mat[r][n1] = self.mat[r][n2];
                    n1 = n2;
                    n2 = (n2 + 1) & 3;
                }
                self.mat[r][to] = bak;
            }
            return;
        }
        let tgt = if sz == 4 {
            let t = self.idx[r] as usize & 3;
            self.idx[r] = ((self.idx[r] as usize + 1) & 3) as u8;
            t
        } else {
            let t = self.size[r] as usize;
            self.size[r] += 1;
            t
        };
        self.mat[r][tgt] = m;
    }
}

/// The refmvs grid: a 64-row × 128-col ring buffer of `RefmvsBlock` (dav2d `rt->r`), indexed
/// `by4.min(crate::av2_recon::nb_len() - 1) * crate::av2_recon::nb_len() + bx4.min(crate::av2_recon::nb_len() - 1)`. Each decoded block splats its per-cell MV/ref/flags here so
/// later blocks' `refmvs_find` reads them. Reset to INVALID_MV at the tile start.
pub struct RefmvsGrid {
    pub r: Vec<RefmvsBlock>,
}

impl Default for RefmvsGrid {
    fn default() -> Self {
        RefmvsGrid { r: vec![RefmvsBlock::default(); crate::av2_recon::nb_len() * crate::av2_recon::nb_len()] }
    }
}

impl RefmvsGrid {
    #[inline]
    pub(crate) fn at(&self, by4: usize, bx4: usize) -> &RefmvsBlock {
        &self.r[by4.min(crate::av2_recon::nb_len() - 1) * crate::av2_recon::nb_len() + bx4.min(crate::av2_recon::nb_len() - 1)]
    }

    /// Splat an INTRA block into the grid: INVALID_MV + ref=-1, but the correct `bx4/by4/bs` so
    /// the warp-sample neighbour walk (`derive_warpmv`) steps over it correctly (it's ref-skipped).
    pub fn splat_intra(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, bs: u8) {
        for r in 0..bh4 {
            if !crate::av2_recon::work_tick("refmvs:2593") { break; }
            for c in 0..bw4 {
                if !crate::av2_recon::work_tick("refmvs:2594") { break; }
                let cell = &mut self.r[(by4 + r).min(crate::av2_recon::nb_len() - 1) * crate::av2_recon::nb_len() + (bx4 + c).min(crate::av2_recon::nb_len() - 1)];
                *cell = RefmvsBlock { bs, bx4: bx4 as u16, by4: by4 as u16, ..RefmvsBlock::default() };
            }
        }
    }

    /// Splat an INTRABC block into the grid (dav2d `splat_intrabc_mv`, decode.c:674): stores the
    /// block vector in `mv[0]` with `ref=[-1,-1]` so a later intrabc block's `refmvs_find(ref=-1)`
    /// spatial scan collects it (the scan matches `ref_[0]==-1` and skips INVALID_MV intra blocks).
    pub fn splat_intrabc(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, bv: Mv, bs: u8) {
        for r in 0..bh4 {
            if !crate::av2_recon::work_tick("refmvs:2605") { break; }
            for c in 0..bw4 {
                if !crate::av2_recon::work_tick("refmvs:2606") { break; }
                let cell = &mut self.r[(by4 + r).min(crate::av2_recon::nb_len() - 1) * crate::av2_recon::nb_len() + (bx4 + c).min(crate::av2_recon::nb_len() - 1)];
                *cell = RefmvsBlock {
                    mv: [bv, Mv { y: -0x8000, x: -0x8000 }],
                    ref_: [-1, -1],
                    bs,
                    bx4: bx4 as u16,
                    by4: by4 as u16,
                    lmv: [bv, Mv { y: -0x8000, x: -0x8000 }],
                    ..RefmvsBlock::default()
                };
            }
        }
    }

    /// Splat a decoded block into the grid (dav2d `splat_oneref_mv`, single-ref). A warp block
    /// (`mf & 2`) stores the PER-CELL projected MV (`warp_cell_mv`); others store a uniform MV.
    #[allow(clippy::too_many_arguments)]
    pub fn splat(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, mv0: Mv, ref0: i8, bs: u8, mf: u8, matrix: [i32; 6]) {
        self.splat_pair(bx4, by4, bw4, bh4, [mv0, Mv { y: -0x8000, x: -0x8000 }], (ref0, -1), bs, mf, matrix);
    }

    /// Compound-aware splat: stores the FULL ref pair + MV pair per cell (the pair feeds later
    /// blocks' compound spatial scan + bank seeding).
    #[allow(clippy::too_many_arguments)]
    pub fn splat_pair(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, mvp: [Mv; 2], refp: (i8, i8), bs: u8, mf: u8, matrix: [i32; 6]) {
        for r in 0..bh4 {
            if !crate::av2_recon::work_tick("refmvs:2632") { break; }
            for c in 0..bw4 {
                if !crate::av2_recon::work_tick("refmvs:2633") { break; }
                let mv = if mf & 2 != 0 {
                    warp_cell_mv(&matrix, bx4 as i32, by4 as i32, (c as i32) & !1, (r as i32) & !1)
                } else {
                    mvp[0]
                };
                let cell = &mut self.r[(by4 + r).min(crate::av2_recon::nb_len() - 1) * crate::av2_recon::nb_len() + (bx4 + c).min(crate::av2_recon::nb_len() - 1)];
                *cell = RefmvsBlock {
                    mv: [mv, mvp[1]],
                    ref_: [refp.0, refp.1],
                    bs,
                    mf,
                    bx4: bx4 as u16,
                    by4: by4 as u16,
                    // warp blocks keep the block MV in lmv (add_sample reads lmv when mf&2).
                    lmv: mvp,
                    matrix,
                };
            }
        }
    }
}

/// dav2d `refmvs_find` (refmvs.c:561), the **single-ref** spatial scan + global fill (the subset
/// this keyframe-referenced, single-ref clip exercises — TMVP is empty, no compound/TIP, no
/// refmv_bank unless the seq enables it). Returns the DRL candidate stack + count.
///
/// `sbsz` = SB size in 4px units (16 for 64px SBs); `iw4`/`ih4` = frame dims (4px). `gmv0` = the
/// frame global MV for this ref. Verified against the oracle BMVSTK/BMVCAND harness.
#[allow(clippy::too_many_arguments)]
pub fn refmvs_find(grid: &RefmvsGrid, bx4: usize, by4: usize, bw4: usize, bh4: usize, ref0: i8, ref1: i8, gmv0: Mv, gmv1: Mv, sbsz: usize, iw4: usize, ih4: usize, drl_reorder: u8, bank: Option<&RefmvBank>, max_drl_bits: usize, skip_mode: bool) -> (Vec<Candidate>, usize) {
    let mut st = ScanState::default();
    // TIP search: the gmv is forced to 0 (dav refmvs.c:581 `ref >= TIP_FRAME ? {0}`).
    let gmv = if ref0 >= 7 { [Mv { y: 0, x: 0 }; 2] } else { [gmv0, gmv1] };
    let comp = ref1 >= 0;
    let tm_stride = TMVS.with(|c| c.borrow().stride);
    st.b8x8 = (bx4 >> 1) + (((by4 & (sbsz - 1)) >> 1) * tm_stride.max(1));
    // temporal 8x8 cell coords for the spatial neighbours (dav refmvs.c:674-813): rp_proj reads
    // for TIP neighbours happen at the NEIGHBOUR's cell, not the block's.
    let tms_8x8y = ((by4 & (sbsz - 1)) >> 1) as isize;
    let bms_8x8y = (((by4 + bh4 - 1) & (sbsz - 1)) >> 1) as isize;
    let left_8x8x = (bx4 as isize - 1) >> 1;
    let top_8x8y: isize = if by4 & (sbsz - 1) != 0 { (((by4 - 1) & (sbsz - 1)) >> 1) as isize } else { -1 };
    let w4 = bw4.min(iw4 - bx4);
    let h4 = bh4.min(ih4 - by4);
    let is_sb_boundary = (by4 & (sbsz - 1)) == 0;
    let have_left = bx4 > 0;
    let have_top = by4 > 0;

    // bottom-most left
    let mut bml_bs = None;
    if have_left && bh4 == h4 {
        let b = *grid.at(by4 + bh4 - 1, bx4 - 1);
        bml_bs = Some(b.bs);
        add_spatial_candidate_sngl(&mut st, 1, &b, bh4 as i32 - 1, -1, bms_8x8y, left_8x8x, bx4 as i32 - 1, (by4 + bh4) as i32 - 1, ref0, ref1, gmv);
    }
    // top neighbours. Non-SB-boundary reads the current grid row (by4-1). The SB-boundary case
    // reads the committed above-SB-row: dav2d `ra[]` (refmvs.c:607-615), an 8x8-resolution
    // snapshot where `ra[k] == grid[by4-1][2k]` (even columns). Single-threaded, grid row
    // (by4-1)&63 still holds the previous SB row's bottom edge, so mine reads it straight.
    let (x_off, abw4): (i32, usize) = if is_sb_boundary {
        ((bx4 & 1) as i32, (bw4 + 1) & !1)
    } else {
        (0, bw4)
    };
    let (mut rmt, mut lmt, mut tr, mut tl) = (None, None, None, None);
    if have_top {
        if is_sb_boundary {
            // tl reads ra[(bx4>>1)-1] == grid[by4-1][(bx4&~1)-2]. At a SB-column boundary dav uses
            // `ra_tl`, but during THIS SB row `ra_tl` == that same cell (ra[] isn't overwritten
            // until the row's save), so the direct grid read matches for SB row 1. (SB rows 2+ the
            // lagged ra_tl may differ — a follow-up if a column-boundary tl block diverges.)
            if bx4 as i32 - x_off - 2 >= 0 {
                tl = Some(*grid.at(by4 - 1, ((bx4 & !1) as i32 - 2) as usize));
            }
            if bw4 > 2 {
                lmt = Some(*grid.at(by4 - 1, bx4 & !1));
            }
            if bw4 == w4 {
                rmt = Some(*grid.at(by4 - 1, 2 * ((bx4 >> 1) + (abw4 >> 1) - 1)));
            }
            if (bx4 as i32 - x_off + abw4 as i32) < iw4 as i32 && bw4 <= 16 {
                tr = Some(*grid.at(by4 - 1, 2 * ((bx4 >> 1) + (abw4 >> 1))));
            }
        } else {
            if bw4 == w4 {
                rmt = Some(*grid.at(by4 - 1, bx4 + bw4 - 1));
            }
            if bw4 > 1 {
                lmt = Some(*grid.at(by4 - 1, bx4));
            }
            if (bx4 + bw4) & (sbsz - 1) != 0 && bx4 + bw4 < iw4 && bw4 <= 16 {
                let t = *grid.at(by4 - 1, bx4 + bw4);
                if t.mv[0].y != -0x8000 {
                    tr = Some(t);
                }
            }
            if have_left {
                tl = Some(*grid.at(by4 - 1, bx4 - 1));
            }
        }
    }
    // right-most top. Generalized xpos (dav2d refmvs.c:719): `abw4 - (1<<is_sb_boundary) - x_off`
    // (= bw4-1 for the non-boundary case where x_off=0, abw4=bw4, is_sb_boundary=0).
    if let Some(b) = rmt {
        let xpos = abw4 as i32 - (1 << is_sb_boundary as i32) - x_off;
        add_spatial_candidate_sngl(&mut st, (xpos >= 0) as i32, &b, -1, xpos, top_8x8y, (bx4 as isize + xpos as isize) >> 1, bx4 as i32 + xpos, by4 as i32 - 1, ref0, ref1, gmv);
    }
    // top-most left
    let mut tml_bs = None;
    if have_left && bh4 > 1 {
        let b = *grid.at(by4, bx4 - 1);
        tml_bs = Some(b.bs);
        add_spatial_candidate_sngl(&mut st, 1, &b, 0, -1, tms_8x8y, left_8x8x, bx4 as i32 - 1, by4 as i32, ref0, ref1, gmv);
    }
    // left-most top (dav2d refmvs.c:741): xpos = -x_off, weight = !x_off.
    if let Some(b) = lmt {
        let xpos = -x_off;
        add_spatial_candidate_sngl(&mut st, (x_off == 0) as i32, &b, -1, xpos, top_8x8y, (bx4 as isize + xpos as isize) >> 1, bx4 as i32 + xpos, by4 as i32 - 1, ref0, ref1, gmv);
    }
    // bottom-left
    if have_left && bh4 <= 16 && (by4 + bh4) & (sbsz - 1) != 0 && by4 + bh4 < ih4 {
        let b = *grid.at(by4 + bh4, bx4 - 1);
        add_spatial_candidate_sngl(&mut st, 1, &b, bh4 as i32, -1, (((by4 + bh4) & (sbsz - 1)) >> 1) as isize, left_8x8x, bx4 as i32 - 1, (by4 + bh4) as i32, ref0, ref1, gmv);
    }
    // top-right (dav2d refmvs.c:765): xpos = abw4 - x_off (= bw4 non-boundary).
    if let Some(b) = tr {
        let xpos = abw4 as i32 - x_off;
        add_spatial_candidate_sngl(&mut st, 1, &b, -1, xpos, top_8x8y, (bx4 as isize + xpos as isize) >> 1, bx4 as i32 + xpos, by4 as i32 - 1, ref0, ref1, gmv);
    }
    // normal-priority TMVP (dav2d refmvs.c:772): first at (x_off,y_off)=2*bw8-2*step (clamped
    // in-block), fallback at the block centre for >4px blocks.
    {
        let t_ok = TMVS.with(|c| {
            let t = c.borrow();
            t.valid && t.use_ref_frame_mvs
        });
        if t_ok && (ref0 != ref1 || skip_mode) && st.cnt < 6 {
            let bw8 = (bw4 >> 1).min(8);
            let bh8 = (bh4 >> 1).min(8);
            let step_h = if bw4 >= 16 { 2usize } else { 1 };
            let step_v = if bh4 >= 16 { 2usize } else { 1 };
            let x_off = 2 * bw8 as i32 - 2 * step_h as i32;
            let y_off = 2 * bh8 as i32 - 2 * step_v as i32;
            let mut first = false;
            if (x_off as usize) < w4 && (y_off as usize) < h4 && x_off >= 0 && y_off >= 0 {
                let off = ((((by4 as i32 + y_off) as usize & (sbsz - 1)) >> 1) * tm_stride)
                    + ((bx4 as i32 + x_off) as usize >> 1);
                first = add_temporal_candidate_t(&mut st, ref0, ref1, off);
            }
            if !first && (bw4 > 4 || bh4 > 4) {
                let off = ((((by4 + bh8) & (sbsz - 1)) >> 1) * tm_stride) + ((bx4 + bw8) >> 1);
                add_temporal_candidate_t(&mut st, ref0, ref1, off);
            }
        }
    }
    // top-left (dav2d refmvs.c:799): xpos = -(1<<is_sb_boundary) - x_off (= -1 non-boundary).
    if let Some(b) = tl {
        let xpos = -(1 << is_sb_boundary as i32) - x_off;
        add_spatial_candidate_sngl(&mut st, 0, &b, -1, xpos, top_8x8y, (bx4 as isize + xpos as isize) >> 1, bx4 as i32 + xpos, by4 as i32 - 1, ref0, ref1, gmv);
    }
    let nearest_refmv_count = st.cnt;

    // Extended-left spatial scan (dav2d refmvs.c:806-849): a SECOND left column `adj` cells
    // further left, at the bottom-most-left (bh4-1) and top-most-left (0) rows, weight 0. Only
    // added when the extended block DIFFERS from the immediate neighbour (narrower than `adj`, or
    // a different block size). This is where a WARP neighbour's projected cell MV (e.g. (18,6)'s
    // (12,90) at offset (1,-3)) enters the stack. `adj = 3` (2 for 4px-wide odd-x blocks).
    if have_left {
        let adj = 3 - (bx4 & (bw4 == 1) as usize) as i32;
        if bx4 as i32 - adj >= 0 {
            let exbx = (bx4 as i32 - adj) as usize;
            let dims = |bs: u8| crate::av2_decode::BLOCK_DIMENSIONS[bs as usize][0] as i32;
            if bh4 == h4 {
                let b = *grid.at(by4 + bh4 - 1, exbx);
                if bml_bs.map_or(true, |bs| dims(b.bs) < adj || b.bs != bs) {
                    add_spatial_candidate_sngl(&mut st, 0, &b, bh4 as i32 - 1, -adj, bms_8x8y, (bx4 as isize - adj as isize) >> 1, bx4 as i32 - adj, (by4 + bh4) as i32 - 1, ref0, ref1, gmv);
                }
            }
            if bh4 > 1 {
                let b = *grid.at(by4, exbx);
                if tml_bs.map_or(true, |bs| dims(b.bs) < adj || b.bs != bs) {
                    add_spatial_candidate_sngl(&mut st, 0, &b, 0, -adj, tms_8x8y, (bx4 as isize - adj as isize) >> 1, bx4 as i32 - adj, by4 as i32, ref0, ref1, gmv);
                }
            }
        }
    }

    // sort: drl_reorder swaps the max-weight of the nearest candidates to the front.
    let reorder = (drl_reorder == 2 && nearest_refmv_count >= 2) || (drl_reorder == 1 && nearest_refmv_count >= 4);
    if reorder {
        let mut maxwidx = 0;
        let mut maxw = st.mvstack[0].weight;
        for n in 1..nearest_refmv_count {
            if !crate::av2_recon::work_tick("refmvs:2826") { break; }
            if st.mvstack[n].weight > maxw {
                maxw = st.mvstack[n].weight;
                maxwidx = n;
            }
        }
        if maxwidx != 0 {
            st.mvstack.swap(0, maxwidx);
        }
    }
    if ref0 == -1 {
        // INTRABC: the refmv bank IS read for ref=-1 (dav2d splat_intrabc_mv → refmvs_bank_add, and
        // refmvs_find reads the bank before the intrabc defaults) — the weight-0 recent-BV bank
        // candidates come first, then the 4 default block vectors, up to `max_drl_bits+1`. No
        // global-mv and NO frame-relative clip (BVs like -2560 legitimately exceed the inter range).
        if let Some(b) = bank {
            let lim = 1 + max_drl_bits;
            add_bank_candidates(&mut st, b, ref0, -1, lim, bx4, by4, bw4, bh4, iw4.div_ceil(2), ih4.div_ceil(2));
        }
        // INTRABC default candidates (dav2d refmvs.c:1037-1067): after the bank, append the
        // 4 default block vectors (weight 0) up to `max_drl_bits+1`.
        let sbsz_px = (sbsz * 4) as i32; // dav 64 << sb128 (sbsz param in 4px cells)
        let lim = 1 + max_drl_bits;
        let defaults = [
            (-(sbsz_px * 8), 0i32),
            (0i32, -(8 * (sbsz_px + 256))),
            (-(bh4 as i32 * 32), 0i32),
            (0i32, -(bw4 as i32 * 32)),
        ];
        for (dy, dx) in defaults {
            if st.cnt >= lim {
                break;
            }
            st.mvstack[st.cnt].mv[0] = Mv { y: dy, x: dx };
            st.mvstack[st.cnt].weight = 0;
            st.cnt += 1;
        }
        return (st.mvstack.to_vec(), st.cnt);
    }
    // Tail (dav refmvs.c:872-994): COMPOUND appends the derived pairs BEFORE the bank; single
    // appends its derived AFTER (mine's single derived list is empty — traj arms unbuilt).
    let lim = 1 + max_drl_bits;
    if comp && st.cnt < lim {
        add_derived_comp(&mut st, lim);
    }
    if let Some(b) = bank {
        add_bank_candidates(&mut st, b, ref0, ref1, lim, bx4, by4, bw4, bh4, iw4.div_ceil(2), ih4.div_ceil(2));
    }
    // single-ref derived append AFTER the bank (dav refmvs.c:928).
    if !comp && st.cnt < lim {
        for m in 0..st.drvd_cnt {
            if !crate::av2_recon::work_tick("refmvs:2876") { break; }
            if st.cnt >= 6 {
                break;
            }
            let cand = st.dr[m][0];
            let mut cnt2 = st.cnt;
            let mut iter2 = st.iter_cntr;
            add_candidate_sngl(&mut st.mvstack, &mut cnt2, lim, 0, cand, 0, 0, &mut iter2, 16);
            st.cnt = cnt2;
            st.iter_cntr = iter2;
        }
        st.drvd_cnt = 0;
    }
    // clip candidates to the frame-relative range (BOTH mvs when compound) — dav clamps BEFORE
    // the gmv fill (refmvs.c:931).
    let (minx, maxx) = (-((bx4 + bw4 + 4) as i32) * 32, (iw4 - bx4 + 4) as i32 * 32);
    let (miny, maxy) = (-((by4 + bh4 + 4) as i32) * 32, (ih4 - by4 + 4) as i32 * 32);
    for n in 0..st.cnt {
        if !crate::av2_recon::work_tick("refmvs:2893") { break; }
        st.mvstack[n].mv[0].x = st.mvstack[n].mv[0].x.clamp(minx, maxx);
        st.mvstack[n].mv[0].y = st.mvstack[n].mv[0].y.clamp(miny, maxy);
        if comp {
            st.mvstack[n].mv[1].x = st.mvstack[n].mv[1].x.clamp(minx, maxx);
            st.mvstack[n].mv[1].y = st.mvstack[n].mv[1].y.clamp(miny, maxy);
        }
    }
    // gmv fill (dav refmvs.c:942): dedup on mv[0] AND mv[comp], append pair with cwp 8.
    if st.cnt < 6 && ref0 >= 0 {
        let mut dup = false;
        if st.iter_cntr < 16 {
            for n in 0..st.cnt {
                if !crate::av2_recon::work_tick("refmvs:2905") { break; }
                if st.mvstack[n].mv[0] == gmv[0] && (!comp || st.mvstack[n].mv[1] == gmv[1]) {
                    st.iter_cntr += n as i32 + 1;
                    dup = true;
                    break;
                }
            }
            if !dup {
                st.iter_cntr += st.cnt as i32;
            }
        }
        if !dup {
            let last = st.cnt;
            st.mvstack[last].mv[0] = gmv[0];
            st.mvstack[last].mv[1] = gmv[1];
            st.mvstack[last].weight = 0;
            st.mvstack[last].cwp = 8;
            st.mvstack[last].y_off = 0;
            st.mvstack[last].x_off = 0;
            st.cnt = last + 1;
        }
        // ext_mvp cross-combinations (dav refmvs.c:970): only for min(bw4,bh4) > 8 blocks.
        // Fill dr[0..2) (from the top-2 candidates); with cnt > 2 also dr[2..6); then ONE
        // add pass over all entries (weight 0).
        if bw4.min(bh4) > 8 && st.cnt >= 2 && st.cnt < 6 {
            const EXT_MVP: [(usize, usize); 6] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];
            for c in 0..2usize {
                if !crate::av2_recon::work_tick("refmvs:2931") { break; }
                let mut n = c * 2;
                while n < c * 4 + 2 {
                    let (yidx, xidx) = EXT_MVP[n];
                    st.dr[n][0] = Mv { y: st.mvstack[yidx].mv[0].y, x: st.mvstack[xidx].mv[0].x };
                    if comp {
                        st.dr[n][1] = Mv { y: st.mvstack[yidx].mv[1].y, x: st.mvstack[xidx].mv[1].x };
                    }
                    n += 1;
                }
                st.drvd_cnt = n;
                if st.cnt == 2 {
                    break;
                }
            }
            for m in 0..st.drvd_cnt {
                if !crate::av2_recon::work_tick("refmvs:2946") { break; }
                if st.cnt >= 6 { break; }
                let pair = st.dr[m];
                let mut cnt2 = st.cnt;
                let mut iter2 = st.iter_cntr;
                if comp {
                    add_candidate_comp(&mut st.mvstack, &mut cnt2, 6, 0, 8, pair, &mut iter2, 16);
                } else {
                    add_candidate_sngl(&mut st.mvstack, &mut cnt2, 6, 0, pair[0], 0, 0, &mut iter2, 16);
                }
                st.cnt = cnt2;
                st.iter_cntr = iter2;
            }
        }
    }
    (st.mvstack.to_vec(), st.cnt)
}

/// dav2d `refmvs_find` warp-candidate path (refmvs.c:628-669, 697-802, 999-1035), single-ref:
/// build `warp[4][6]` = corner model (from bml/tl/rmt or bml/lmt/tr) + neighbour warp models
/// (add_matrix) + fill (warp bank → gmv matrix → identity). Returns the models + count. `b_dim` =
/// [bw4,bh4,w_log2,h_log2]. Only warp neighbours (mf&2) contribute models; this clip is single-ref.
#[allow(clippy::too_many_arguments)]
pub fn refmvs_find_warp(grid: &RefmvsGrid, warp_bank: &RefmvWarpBank, bx4: usize, by4: usize, b_dim: [u8; 4], ref0: i8, gmv_matrix: [i32; 6], sbsz: usize, iw4: usize, ih4: usize) -> [[i32; 6]; 4] {
    let (bw4, bh4) = (b_dim[0] as usize, b_dim[1] as usize);
    let w4 = bw4.min(iw4 - bx4);
    let h4 = bh4.min(ih4 - by4);
    let is_sb_boundary = (by4 & (sbsz - 1)) == 0;
    let have_left = bx4 > 0;
    let have_top = by4 > 0;
    let (minx, maxx) = (-((bx4 + bw4 + 4) as i32) * 32, (iw4 - bx4 + 4) as i32 * 32);
    let (miny, maxy) = (-((by4 + bh4 + 4) as i32) * 32, (ih4 - by4 + 4) as i32 * 32);
    let mut warp = [[0i32; 6]; 4];
    let mut cnt1 = 0usize;

    // corner-neighbour match (dav refmvs.c:630): ref[0] match (slot 0, warp-projectable) OR a
    // COMPOUND neighbour matching on ref[1] (slot-1 mv, only when not a warp block). Returns the
    // matched mv slot.
    let matches = |b: &RefmvsBlock| -> Option<usize> {
        if b.mv[0].y == -0x8000 {
            return None;
        }
        if b.ref_[0] == ref0 {
            Some(0)
        } else if b.ref_[1] == ref0 && b.mf & 2 == 0 {
            Some(1)
        } else {
            None
        }
    };
    let proj = |b: &RefmvsBlock, slot: usize, x: i32, y: i32| -> Mv {
        if b.mf & 2 != 0 { get_warpmv_proj(&b.matrix, x, y, minx, maxx, miny, maxy) } else { b.mv[slot] }
    };

    // neighbour blocks
    let bml = if have_left && bh4 == h4 { Some(*grid.at(by4 + bh4 - 1, bx4 - 1)) } else { None };
    let (mut tl, mut lmt, mut rmt, mut tr) = (None, None, None, None);
    if have_top {
        if is_sb_boundary {
            // SB-boundary above-row reads (dav2d refmvs.c:607-615), shared with the spatial scan:
            // ra[k] == grid[by4-1][2k]. tl at a column boundary uses ra_tl (unmaintained) — skipped.
            let x_off = (bx4 & 1) as i32;
            let abw4 = (bw4 + 1) & !1;
            if bx4 as i32 - x_off - 2 >= 0 {
                tl = Some(*grid.at(by4 - 1, ((bx4 & !1) as i32 - 2) as usize));
            }
            if bw4 > 2 { lmt = Some(*grid.at(by4 - 1, bx4 & !1)); }
            if bw4 == w4 { rmt = Some(*grid.at(by4 - 1, 2 * ((bx4 >> 1) + (abw4 >> 1) - 1))); }
            if (bx4 as i32 - x_off + abw4 as i32) < iw4 as i32 && bw4 <= 16 {
                tr = Some(*grid.at(by4 - 1, 2 * ((bx4 >> 1) + (abw4 >> 1))));
            }
        } else {
            if have_left { tl = Some(*grid.at(by4 - 1, bx4 - 1)); }
            if bw4 > 1 { lmt = Some(*grid.at(by4 - 1, bx4)); }
            if bw4 == w4 { rmt = Some(*grid.at(by4 - 1, bx4 + bw4 - 1)); }
            if (bx4 + bw4) & (sbsz - 1) != 0 && bx4 + bw4 < iw4 && bw4 <= 16 {
                let t = *grid.at(by4 - 1, bx4 + bw4);
                if t.mv[0].y != -0x8000 { tr = Some(t); }
            }
        }
    }
    let tml = if have_left && bh4 > 1 { Some(*grid.at(by4, bx4 - 1)) } else { None };
    let bl = if have_left && bh4 <= 16 && (by4 + bh4) & (sbsz - 1) != 0 && by4 + bh4 < ih4 { Some(*grid.at(by4 + bh4, bx4 - 1)) } else { None };

    // corner model → warp[0]
    if let Some((bmlb, bslot)) = bml.and_then(|b| matches(&b).map(|sl| (b, sl))) {
        let bl_mv = proj(&bmlb, bslot, bx4 as i32 * 4, (by4 + bh4) as i32 * 4);
        let build = |a: &RefmvsBlock, asl: usize, c: &RefmvsBlock, csl: usize| -> Option<[i32; 6]> {
            let tl_mv = proj(a, asl, bx4 as i32 * 4, by4 as i32 * 4);
            let tr_mv = proj(c, csl, (bx4 + bw4) as i32 * 4, by4 as i32 * 4);
            model_from_corners(tl_mv, tr_mv, bl_mv, bx4 as i32 * 4, by4 as i32 * 4, b_dim)
        };
        let m1 = (
            tl.and_then(|b| matches(&b).map(|sl| (b, sl))),
            rmt.and_then(|b| matches(&b).map(|sl| (b, sl))),
        );
        if let (Some((a, asl)), Some((c, csl))) = m1 {
            if let Some(m) = build(&a, asl, &c, csl) { warp[0] = m; cnt1 = 1; }
        }
        if cnt1 == 0 {
            let m2 = (
                lmt.and_then(|b| matches(&b).map(|sl| (b, sl))),
                tr.and_then(|b| matches(&b).map(|sl| (b, sl))),
            );
            if let (Some((a, asl)), Some((c, csl))) = m2 {
                if let Some(m) = build(&a, asl, &c, csl) { warp[0] = m; cnt1 = 1; }
            }
        }
    }

    // add_matrix: neighbour warp models, in dav2d scan order (bml, rmt, tml, lmt, bl, tr, tl).
    for nb in [bml, rmt, tml, lmt, bl, tr, tl].into_iter().flatten() {
        if !crate::av2_recon::work_tick("refmvs:3057") { break; }
        if cnt1 < 4 && nb.mf & 2 != 0 && nb.ref_[0] == ref0 {
            warp[cnt1] = nb.matrix;
            cnt1 += 1;
        }
    }
    // extended-scan add_matrix (dav2d refmvs.c:818/837): ext_bml/ext_tml, a column `adj` further
    // left, at the bottom-most-left (bh4-1) and top-most-left (0) rows — only when the extended
    // block DIFFERS from the immediate neighbour (narrower than `adj`, or a different bs). This is
    // where an SB-row-1 warp neighbour's matrix (e.g. (6,16)'s ext_bml/ext_tml = M) enters warp[].
    if have_left {
        let adj = 3 - (bx4 & (bw4 == 1) as usize) as i32;
        if bx4 as i32 - adj >= 0 {
            let exbx = (bx4 as i32 - adj) as usize;
            let dims = |bs: u8| crate::av2_decode::BLOCK_DIMENSIONS[bs as usize][0] as i32;
            if bh4 == h4 {
                let b = *grid.at(by4 + bh4 - 1, exbx);
                if bml.map_or(true, |bm| dims(b.bs) < adj || b.bs != bm.bs) && cnt1 < 4 && b.mf & 2 != 0 && b.ref_[0] == ref0 {
                    warp[cnt1] = b.matrix;
                    cnt1 += 1;
                }
            }
            if bh4 > 1 {
                let b = *grid.at(by4, exbx);
                if tml.map_or(true, |tm| dims(b.bs) < adj || b.bs != tm.bs) && cnt1 < 4 && b.mf & 2 != 0 && b.ref_[0] == ref0 {
                    warp[cnt1] = b.matrix;
                    cnt1 += 1;
                }
            }
        }
    }

    // fill: warp bank (recent-first) → gmv matrix → identity defaults.
    if cnt1 < 4 && (0..7).contains(&(ref0 as i32)) {
        let r = ref0 as usize;
        let sz = warp_bank.size[r] as usize;
        let idx = warp_bank.idx[r] as usize;
        let start = sz + idx;
        let mut n = 0;
        while n < sz && cnt1 < 4 {
            warp[cnt1] = warp_bank.mat[r][(start.wrapping_sub(1).wrapping_sub(n)) & 3];
            cnt1 += 1;
            n += 1;
        }
    }
    if cnt1 < 4 {
        warp[cnt1] = gmv_matrix;
        cnt1 += 1;
    }
    while cnt1 < 4 {
        warp[cnt1] = IDENTITY_WARP;
        cnt1 += 1;
    }
    warp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_bank_dedups_on_shear_not_full_matrix() {
        // dav2d `dav2d_refmvs_warp_add` dedups on the SHEAR (matrix[2..6]) only. Three blocks with
        // the same shear [90112,-8192,32704,59392] but different translation must collapse to ONE
        // bank entry (verified vs the oracle at block (0,10): warp[] = [286720,-889152],(6,4),(0,0),
        // [1225728,...], all from the bank — a full-matrix compare wrongly kept all three).
        let mut wb = RefmvWarpBank::default();
        let shear = [90112, -8192, 32704, 59392];
        wb.add(0, [-507904, -1689920, shear[0], shear[1], shear[2], shear[3]]);
        wb.add(0, [286720, -887616, shear[0], shear[1], shear[2], shear[3]]);
        wb.add(0, [286720, -889152, shear[0], shear[1], shear[2], shear[3]]);
        assert_eq!(wb.size[0], 1, "same-shear adds must collapse to one entry");
        // The retained entry keeps the FIRST-added translation (dedup preserves the existing entry).
        assert_eq!(&wb.mat[0][0][..2], &[-507904, -1689920]);
        // A distinct shear adds a second entry.
        wb.add(0, [665984, -542464, 78144, -8128, 26752, 59520]);
        assert_eq!(wb.size[0], 2);
    }

    #[test]
    fn reduce_prec_matches_oracle_block_4_0() {
        // dav2d verified: block (4,0) predictor (25,159) @ mv_prec=3 → (24,160), then +diff
        // (-32,-56) = final (-8,104) == oracle fmv.
        let mut m = Mv { y: 25, x: 159 };
        mv_reduce_prec(&mut m, 3);
        assert_eq!(m, Mv { y: 24, x: 160 });
        let fmv = Mv { y: m.y + -32, x: m.x + -56 };
        assert_eq!(fmv, Mv { y: -8, x: 104 });
    }

    #[test]
    fn reduce_prec_6_is_noop() {
        let mut m = Mv { y: 25, x: 159 };
        mv_reduce_prec(&mut m, 6);
        assert_eq!(m, Mv { y: 25, x: 159 });
    }

    #[test]
    fn gmv_identity_is_zero() {
        // This clip's global motion is IDENTITY → predictor base (0,0), matching block (0,0).
        let g = GmvModel::default();
        assert_eq!(get_gmv_2d(&g, 0, 0, 4, 4, 108, 60), Mv { y: 0, x: 0 });
    }

    #[test]
    fn warpmv_proj_identity_is_zero() {
        // Identity warp (m[2]=m[5]=1<<16, rest 0) projects to (0,0) at any sample point.
        let m = [0, 0, 1 << 16, 0, 0, 1 << 16];
        assert_eq!(get_warpmv_proj(&m, 16, 0, -0xffff, 0xffff, -0xffff, 0xffff), Mv { y: 0, x: 0 });
    }

    #[test]
    fn add_candidate_dedups_and_appends() {
        let mut stack = [Candidate::default(); 6];
        let mut cnt = 0usize;
        let mut iter = 0i32;
        // append two distinct MVs
        assert!(add_candidate_sngl(&mut stack, &mut cnt, 6, 2, Mv { y: 17, x: 165 }, 0, -1, &mut iter, 16));
        assert!(add_candidate_sngl(&mut stack, &mut cnt, 6, 1, Mv { y: 25, x: 159 }, -1, 0, &mut iter, 16));
        assert_eq!(cnt, 2);
        // re-adding the first MV dedups (accumulates weight), no new entry
        assert!(!add_candidate_sngl(&mut stack, &mut cnt, 6, 3, Mv { y: 17, x: 165 }, 0, -1, &mut iter, 16));
        assert_eq!(cnt, 2);
        assert_eq!(stack[0].weight, 5); // 2 + 3
        assert_eq!(stack[1].weight, 1);
    }

    #[test]
    fn refmvs_find_block_4_0_spatial_candidates() {
        // Grid with warp block (0,0) splatted; scanning (4,0) must produce the oracle's spatial
        // warp candidates (25,159 w1) [bml] and (31,167 w1) [tml], then the global (0,0).
        // (The oracle's extra (24,168) w0 comes from the refmv_bank — a separate deferred brick.)
        let mut grid = RefmvsGrid::default();
        let m = [1476608, 182272, 59392, -8192, 8192, 59392];
        grid.splat(0, 0, 4, 4, Mv { y: 24, x: 168 }, 0, 18, 2, m); // (0,0) warp block
        let (stack, cnt) = refmvs_find(&grid, 4, 0, 4, 4, 0, -1, Mv { y: 0, x: 0 }, Mv { y: 0, x: 0 }, 16, 108, 60, 2, None, 3, false);
        assert_eq!(cnt, 3);
        assert_eq!(stack[0].mv[0], Mv { y: 25, x: 159 });
        assert_eq!(stack[0].weight, 1);
        assert_eq!(stack[1].mv[0], Mv { y: 31, x: 167 });
        assert_eq!(stack[1].weight, 1);
        assert_eq!(stack[2].mv[0], Mv { y: 0, x: 0 }); // global fill
        assert_eq!(stack[2].weight, 0);
    }

    #[test]
    fn refmvs_find_block_4_0_full_stack_with_bank() {
        // Full (4,0) stack incl. the refmv_bank: (0,0) splats AND banks its block MV (24,168);
        // (4,0) then gets spatial (25,159)(31,167) + bank (24,168) + global (0,0) = oracle BMVSTK.
        let mut grid = RefmvsGrid::default();
        let m = [1476608, 182272, 59392, -8192, 8192, 59392];
        grid.splat(0, 0, 4, 4, Mv { y: 24, x: 168 }, 0, 18, 2, m);
        let mut bank = RefmvBank::default();
        bank.add_block(4, 4, 0, 0, 16, 0, 0, Mv { y: 24, x: 168 }); // (0,0) pushes its block MV
        let (stack, cnt) = refmvs_find(&grid, 4, 0, 4, 4, 0, -1, Mv { y: 0, x: 0 }, Mv { y: 0, x: 0 }, 16, 108, 60, 2, Some(&bank), 3, false);
        assert_eq!(cnt, 4);
        assert_eq!(stack[0].mv[0], Mv { y: 25, x: 159 });
        assert_eq!(stack[1].mv[0], Mv { y: 31, x: 167 });
        assert_eq!(stack[2].mv[0], Mv { y: 24, x: 168 }); // from the refmv_bank, weight 0
        assert_eq!(stack[2].weight, 0);
        assert_eq!(stack[3].mv[0], Mv { y: 0, x: 0 }); // global
    }

    #[test]
    fn refmvs_degenerate_block_0_0_is_global_zero() {
        // Block (0,0): no spatial neighbours → global fill → stack = [(0,0), w0], cnt=1.
        // Matches oracle BMVSTK (0,0) ncand=1, candidate (0,0) w=0.
        let grid = RefmvsGrid::default();
        let (stack, cnt) = refmvs_find(&grid, 0, 0, 4, 4, 0, -1, Mv { y: 0, x: 0 }, Mv { y: 0, x: 0 }, 16, 108, 60, 2, None, 3, false);
        assert_eq!(cnt, 1);
        assert_eq!(stack[0].mv[0], Mv { y: 0, x: 0 });
        assert_eq!(stack[0].weight, 0);
        // (also directly:) the global fill alone yields the same degenerate stack.
        let mut st = ScanState::default();
        add_global_candidate(&mut st, Mv { y: 0, x: 0 });
        assert_eq!(st.cnt, 1);
        assert_eq!(st.mvstack[0].mv[0], Mv { y: 0, x: 0 });
        assert_eq!(st.mvstack[0].weight, 0);
        // And the (0,0) finalization: predictor (0,0) + reduce_prec + residual (24,168) = (24,168).
        let mut pred = st.mvstack[0].mv[0];
        mv_reduce_prec(&mut pred, 3);
        assert_eq!(Mv { y: pred.y + 24, x: pred.x + 168 }, Mv { y: 24, x: 168 });
    }

    #[test]
    fn spatial_candidate_matches_ref_and_dedups() {
        // A single-ref neighbour with ref[0]==0 contributes mv[0]; a global fill tails (0,0).
        let mut st = ScanState::default();
        let nb = RefmvsBlock { mv: [Mv { y: 24, x: 168 }, Mv::default()], ref_: [0, -1], ..Default::default() };
        add_spatial_candidate_sngl(&mut st, 2, &nb, -1, 0, 0, 0, nb.bx4 as i32, nb.by4 as i32, 0, -1, [Mv { y: 0, x: 0 }; 2]);
        assert_eq!(st.cnt, 1);
        assert_eq!(st.mvstack[0].mv[0], Mv { y: 24, x: 168 });
        assert_eq!(st.mvstack[0].weight, 2);
        // an intra neighbour (INVALID_MV) contributes nothing
        let intra = RefmvsBlock::default();
        add_spatial_candidate_sngl(&mut st, 2, &intra, 0, -1, 0, 0, 0, 0, 0, -1, [Mv { y: 0, x: 0 }; 2]);
        assert_eq!(st.cnt, 1);
        // a non-matching ref contributes nothing
        let other = RefmvsBlock { mv: [Mv { y: 9, x: 9 }, Mv::default()], ref_: [1, -1], ..Default::default() };
        add_spatial_candidate_sngl(&mut st, 2, &other, 0, -1, 0, 0, 0, 0, 0, -1, [Mv { y: 0, x: 0 }; 2]);
        assert_eq!(st.cnt, 1);
        // global fill appends (0,0) w0 (distinct from the (24,168) already present)
        add_global_candidate(&mut st, Mv { y: 0, x: 0 });
        assert_eq!(st.cnt, 2);
        assert_eq!(st.mvstack[1].mv[0], Mv { y: 0, x: 0 });
    }

    #[test]
    fn add_candidate_respects_max_cnt() {
        let mut stack = [Candidate::default(); 6];
        let mut cnt = 0usize;
        let mut iter = 0i32;
        for i in 0..8 {
            add_candidate_sngl(&mut stack, &mut cnt, 6, 1, Mv { y: i, x: 100 + i }, 0, 0, &mut iter, 16);
        }
        assert_eq!(cnt, 6); // capped at max_cnt=6
    }

    #[test]
    fn find_affine_int_matches_oracle_block_6_4() {
        // (6,4) MM_WARP_CAUSAL: oracle pts + mv (8,104) → matrix [665984,-542464,78144,-8128,26752,59520].
        let pts = [[[-40, 56], [48, 36]], [[-40, -72], [64, -80]]];
        let m = find_affine_int(&pts, 2, 2, Mv { y: 8, x: 104 }, 6, 4);
        assert_eq!(m, [665984, -542464, 78144, -8128, 26752, 59520]);
    }

    #[test]
    fn get_warpmv_2d_matches_oracle_block_2_4() {
        // (2,4) WARPMV predictor: get_warpmv_2d(warp[0], bx4=2,by4=4,bw4=2,bh4=2, prec=6) = (21,158)
        // = the oracle's block (2,4) final MV.
        let warp0 = [1228800, 98304, 59392, 7168, 8192, 64512];
        assert_eq!(get_warpmv_2d(&warp0, 2, 4, 2, 2, 108, 60, 6), Mv { y: 21, x: 158 });
    }

    #[test]
    fn model_from_corners_matches_oracle_block_2_4() {
        // (2,4)'s warp[0] from the oracle MFC probe: corners tl=(18,158) tr=(26,152) bl=(17,165),
        // xpos=8 ypos=16, b_dim=[2,2,1,1] → [1228800,98304,59392,7168,8192,64512].
        let m = model_from_corners(Mv { y: 18, x: 158 }, Mv { y: 26, x: 152 }, Mv { y: 17, x: 165 }, 8, 16, [2, 2, 1, 1]);
        assert_eq!(m, Some([1228800, 98304, 59392, 7168, 8192, 64512]));
    }

    #[test]
    fn reconstruct_warp_matrix_matches_oracle_block_0_0() {
        // (0,0): wri=0 → warp_base = identity; warp_delta [-6,-8] (np==2); final MV (24,168).
        // Must reconstruct the oracle WARPMAT [1476608,182272,59392,-8192,8192,59392].
        let m = reconstruct_warp_delta_matrix(IDENTITY_WARP, [-6, -8, -0x80, -0x8000000], Mv { y: 24, x: 168 }, 4, 4, 0, 0);
        assert_eq!(m, [1476608, 182272, 59392, -8192, 8192, 59392]);
    }

    #[test]
    fn warp_cell_mv_matches_oracle_block_0_0() {
        // Block (0,0)'s warp matrix (from the oracle WARPMAT probe). Its rightmost-column cells,
        // which neighbour (4,0) reads, must project to the oracle's BMVCAND (4,0) candidates:
        //   cell (cx=2, cy=0) = (31,167)  [(4,0) top-most-left]
        //   cell (cx=2, cy=2) = (25,159)  [(4,0) bottom-most-left]
        let m = [1476608, 182272, 59392, -8192, 8192, 59392];
        assert_eq!(warp_cell_mv(&m, 0, 0, 2, 0), Mv { y: 31, x: 167 });
        assert_eq!(warp_cell_mv(&m, 0, 0, 2, 2), Mv { y: 25, x: 159 });
    }

    #[test]
    fn warp_cell_mv_identity_is_zero() {
        let m = [0, 0, 0x10000, 0, 0, 0x10000];
        assert_eq!(warp_cell_mv(&m, 0, 0, 2, 2), Mv { y: 0, x: 0 });
    }

    #[test]
    fn warpmv_proj_matches_hand_computed() {
        // m = [0x2000, 0x1000, 0x10100, 0x80, 0x40, 0x10200], sample (x=16,y=0):
        //   xc = (0x10100-0x10000)*16 + 0x80*0 + 0x2000 = 0x1000 + 0x2000 = 0x3000
        //   yc = (0x10200-0x10000)*0  + 0x40*16 + 0x1000 = 0x400 + 0x1000 = 0x1400
        //   x = (0x3000 + 0x1000) >> 13 = 0x4000>>13 = 2 ; y = (0x1400 + 0x1000) >> 13 = 0x2400>>13 = 1
        let m = [0x2000, 0x1000, 0x10100, 0x80, 0x40, 0x10200];
        assert_eq!(get_warpmv_proj(&m, 16, 0, -0xffff, 0xffff, -0xffff, 0xffff), Mv { y: 1, x: 2 });
    }
}
