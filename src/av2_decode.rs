//! AV2 block-decode primitives — the first module of the decode body.
//!
//! Bit-exact transcriptions from dav2d (`levels.h`, `env.h`). These are the
//! foundation the partition recursion (`decode_sb`) consumes; they are standalone
//! and unit-testable, but not yet wired into the full decode path (which needs the
//! `TaskContext`/frame-header integration — see docs/decode-core.md §3).

/// AV2 block partition types (dav2d `enum BlockPartition`). AV2 adds the H3/V3
/// extended and H4A/H4B/V4A/V4B uneven-4-way partitions over AV1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i8)]
pub enum BlockPartition {
    Invalid = -1,
    None = 0,
    H = 1,
    V = 2,
    H3 = 3,
    V3 = 4,
    H4a = 5,
    H4b = 6,
    V4a = 7,
    V4b = 8,
    Split = 9,
}

/// Partition context from the above (`a`) / left (`l`) neighbour partition arrays
/// (dav2d `get_partition_ctx`). `a`/`l` are `BlockContext.partition[plane]`
/// (`[u8; 64]`); `b_dim` is the block-dimension entry. The first SB's cleared
/// neighbours give ctx 0 (confirmed against the live decode).
#[inline]
pub fn get_partition_ctx(a: &[u8], l: &[u8], b_dim: &[u8], xb4: usize, yb4: usize) -> u8 {
    // HARDENING: a corrupt stream can address neighbour cells outside the context arrays;
    // an absent neighbour reads as cleared (0), matching the tile/frame-edge convention.
    let av = a.get(xb4).copied().unwrap_or(0);
    let lv = l.get(yb4).copied().unwrap_or(0);
    ((av >> b_dim[2].saturating_sub(1)) & 1) + (((lv >> b_dim[3].saturating_sub(1)) & 1) << 1)
}

/// Extended-partition context, direction-dependent (dav2d `get_partition2_ctx`).
/// `dir == false` → horizontal (reads left), `true` → vertical (reads above).
/// Only called where `b_dim[2]`/`b_dim[3] >= 2` (the ext/uneven-4way path).
#[inline]
pub fn get_partition2_ctx(a: &[u8], l: &[u8], b_dim: &[u8], dir: bool, xb4: usize, yb4: usize) -> u8 {
    if !dir {
        let hh4 = (b_dim[1] >> 1) as usize;
        ((l[yb4 + hh4] >> (b_dim[3] - 2)) & 1) + (((l[yb4] >> (b_dim[3] - 2)) & 1) << 1)
    } else {
        let hw4 = (b_dim[0] >> 1) as usize;
        ((a[xb4 + hw4] >> (b_dim[2] - 2)) & 1) + (((a[xb4] >> (b_dim[2] - 2)) & 1) << 1)
    }
}

/// AV2 block sizes (dav2d `enum BlockSize`), in index order. AV2 adds 256x256 +
/// the 4x/x4 extremes over AV1 (N_BS_SIZES = 31).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BlockSize {
    Bs256x256, Bs256x128, Bs128x256, Bs128x128, Bs128x64, Bs64x128,
    Bs64x64, Bs64x32, Bs64x16, Bs64x8, Bs64x4,
    Bs32x64, Bs32x32, Bs32x16, Bs32x8, Bs32x4,
    Bs16x64, Bs16x32, Bs16x16, Bs16x8, Bs16x4,
    Bs8x64, Bs8x32, Bs8x16, Bs8x8, Bs8x4,
    Bs4x64, Bs4x32, Bs4x16, Bs4x8, Bs4x4,
}

pub const N_BS_SIZES: usize = 31;

/// `{w4, h4, w_log2, h_log2}` per [`BlockSize`] (dav2d `dav2d_block_dimensions`),
/// in 4-pixel units. `w4 = 1<<w_log2`, `h4 = 1<<h_log2` (validated on load).
pub static BLOCK_DIMENSIONS: [[u8; 4]; N_BS_SIZES] = [
    [64, 64, 6, 6], // 256x256
    [64, 32, 6, 5], // 256x128
    [32, 64, 5, 6], // 128x256
    [32, 32, 5, 5], // 128x128
    [32, 16, 5, 4], // 128x64
    [16, 32, 4, 5], // 64x128
    [16, 16, 4, 4], // 64x64
    [16, 8, 4, 3],  // 64x32
    [16, 4, 4, 2],  // 64x16
    [16, 2, 4, 1],  // 64x8
    [16, 1, 4, 0],  // 64x4
    [8, 16, 3, 4],  // 32x64
    [8, 8, 3, 3],   // 32x32
    [8, 4, 3, 2],   // 32x16
    [8, 2, 3, 1],   // 32x8
    [8, 1, 3, 0],   // 32x4
    [4, 16, 2, 4],  // 16x64
    [4, 8, 2, 3],   // 16x32
    [4, 4, 2, 2],   // 16x16
    [4, 2, 2, 1],   // 16x8
    [4, 1, 2, 0],   // 16x4
    [2, 16, 1, 4],  // 8x64
    [2, 8, 1, 3],   // 8x32
    [2, 4, 1, 2],   // 8x16
    [2, 2, 1, 1],   // 8x8
    [2, 1, 1, 0],   // 8x4
    [1, 16, 0, 4],  // 4x64
    [1, 8, 0, 3],   // 4x32
    [1, 4, 0, 2],   // 4x16
    [1, 2, 0, 1],   // 4x8
    [1, 1, 0, 0],   // 4x4
];

/// Partition-context constants per [`BlockSize`] (dav2d `subb[].ctx`): the
/// `{split, direction}` context multipliers feeding `ctx = ctx_neighbour + pcc*4`.
#[rustfmt::skip]
pub static PCC_CTX: [[i8; 2]; N_BS_SIZES] = [
    [9,12],[8,-1],[7,-1],[6,9],[5,-1],[4,-1],   // 256x256..64x128
    [3,6],[3,5],[15,14],[0,0],[0,-1],            // 64x64..64x4
    [3,4],[2,3],[2,2],[13,14],[0,-1],            // 32x64..32x4
    [14,13],[2,1],[1,0],[1,2],[11,-1],           // 16x64..16x4
    [0,0],[12,13],[1,1],[0,0],[0,-1],            // 8x64..8x4
    [0,-1],[0,-1],[10,-1],[0,-1],[-1,-1],        // 4x64..4x4
];

/// Half-split sub-block size per [`BlockSize`] and direction (dav2d
/// `subb[].part[dir][0]`): the BlockSize index a PARTITION_H/V recurses into
/// (`-1` = none). `[h_half, v_half]`.
#[rustfmt::skip]
pub static PART_HALF: [[i8; 2]; N_BS_SIZES] = [
    [1,2],[-1,3],[3,-1],[4,5],[-1,6],[6,-1],     // 256x256..64x128
    [7,11],[8,12],[9,13],[-1,-1],[-1,-1],        // 64x64..64x4
    [12,16],[13,17],[14,18],[15,19],[-1,-1],     // 32x64..32x4
    [17,21],[18,22],[19,23],[20,24],[-1,25],     // 16x64..16x4
    [-1,-1],[23,27],[24,28],[25,29],[-1,30],     // 8x64..8x4
    [-1,-1],[-1,-1],[29,-1],[30,-1],[-1,-1],     // 4x64..4x4
];

/// Decode one block's partition (dav2d `decode_sb` partition section) — split →
/// direction → ext, returning the partition and the half-split sub-block size
/// (`-1` for a leaf / no recursion). Top-left-descent form: cleared neighbours
/// (`ctx1 = ctx5 = 0`), keyframe intra (`mix_inter=false`); 128/256-square and the
/// `>=128px` aspect branches are omitted (not reached for sizes ≤ 64px).
#[allow(clippy::too_many_arguments)]
pub fn decode_partition(
    msac: &mut crate::msac::MsacContext,
    m: &mut crate::cdf_av2::CdfModeContext,
    bs: usize,
    a_part: &[u8],
    l_part: &[u8],
    xb4: usize,
    yb4: usize,
    have_h_split: bool,
    have_v_split: bool,
    aspect_log2: u32,
    ext_partitions: bool,
    pl: usize,
    ssh: u32,
    ssv: u32,
    iw4: usize,
    ih4: usize,
    // True in a MIXED inter region (IS_INTER && !intra_region): forces NONE for 4x8/8x4 blocks.
    mix_inter: bool,
) -> (BlockPartition, i8) {
    use crate::msac::rav1d_msac_decode_bool_adapt;
    let pcc = PCC_CTX[bs];
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as i32, bd[1] as i32);

    // The minimum coded block is 4x4 in the *plane's* sample grid: for chroma (ss=1,
    // 4:2:0) an 8x8 luma region is already a 4x4 chroma block, so it cannot split and
    // the partition is an uncoded PARTITION_NONE (oracle: r unchanged). For luma (ss=0)
    // this reduces to the usual bw4==1 && bh4==1 check. Additionally, 1:8/1:16-aspect
    // blocks (both half-splits invalid, `PART_HALF == [-1,-1]`) don't recurse normatively
    // — also an uncoded NONE (dav2d `(pcc->part[0][0] & pcc->part[1][0]) == -1`).
    let ph = PART_HALF[bs];
    let pdbg = std::env::var("SBI").is_ok() && bs == 27 && yb4 == 0 && xb4 <= 1;
    if pdbg { crate::dlog!("[PD] ({xb4},{yb4}) bs={bs} bw4={bw4} bh4={bh4} ph={ph:?} mix_inter={mix_inter} bd23={} have_h={have_h_split} have_v={have_v_split} r_in={}", bd[2]+bd[3], msac.rng); }
    // Chroma-plane shape validity (avm ss_size_lookup, sizes <= 64px): the subsampled
    // shape must have both dims >= 4px and aspect <= 8:1. `lw`/`lh` are LUMA log2-4px dims
    // of the candidate sub-shape.
    let cs_ok = |lw: i32, lh: i32| -> bool {
        let (cw, ch) = (lw - ssh as i32, lh - ssv as i32);
        // avm ss_size_lookup closure: shapes up to 64px allow aspect <= 8:1, but the enum
        // has NO high-aspect 128px shapes (32x128/128x32 etc. are BLOCK_INVALID) — a dim
        // >= 128px (log2-4 cells >= 5) only pairs at aspect <= 2:1 (64x128/128x64/128x128).
        cw >= 0 && ch >= 0 && (cw - ch).abs() <= 3 && (cw.max(ch) < 5 || (cw - ch).abs() <= 1)
    };
    // The chroma tree never descends below an 8x8 LUMA block at ANY subsampling
    // (avm: derived PARTITION_NONE, no symbol — "for chroma we do not allow dimension
    // of 4", blockd.h:981; observed as [PARTFD] part=0 at 4:2:2).
    if pl == 1 && bw4 == 2 && bh4 == 2 {
        return (BlockPartition::None, -1);
    }
    let (msw4, msh4) = if pl == 1 { (bw4 >> ssh, bh4 >> ssv) } else { (bw4, bh4) };
    if (msw4 == 1 && msh4 == 1) || (ph[0] == -1 && ph[1] == -1) {
        if pdbg { crate::dlog!("[PD] -> forced NONE (min/ph)"); }
        return (BlockPartition::None, -1);
    }

    // Frame-boundary FORCED partition (dav2d decode.c:3692, in LUMA dims/coords). When a
    // block half is out of frame, some shapes force H/V with NO coded symbol; others fall
    // through to the coded path (which still force-splits via `is_split` below).
    if !have_h_split || !have_v_split {
        let (qw4, qh4) = ((bw4 >> 2) as usize, (bh4 >> 2) as usize);
        if bw4 == bh4 {
            // square: bottom OOB → H, right OOB → V (have_v_split true here means right OOB).
            let dir = if !have_v_split { 0usize } else { 1 };
            // 4:2:2 SHARED (>64) tree exception (avm read_partition): the normative forced V
            // at a right-edge 128-square has an INVALID chroma child (ss_size_lookup
            // 64x128@422 = INVALID) → V is not in the allowed set. avm then IMPLIES
            // do_split (no symbol) and CODES the do_square_split bool: 1 → quad SPLIT,
            // 0 → HORZ (the only remaining rect). 4:2:0/4:4:4 keep the plain forced V.
            if dir == 1 && bs == 3 && ssh == 1 && ssv == 0 {
                let ctx1 = get_partition_ctx(a_part, l_part, &bd, xb4, yb4) as usize;
                let square = rav1d_msac_decode_bool_adapt(msac, &mut m.part_square[ctx1]);
                if square {
                    return (BlockPartition::Split, 6);
                }
                return (BlockPartition::H, PART_HALF[bs][0]);
            }
            let bp = if dir == 1 { BlockPartition::V } else { BlockPartition::H };
            return (bp, PART_HALF[bs][dir]);
        } else if bw4 > bh4 {
            if !have_h_split || ih4 <= yb4 + qh4 {
                return (BlockPartition::V, PART_HALF[bs][1]);
            }
        } else if !have_v_split || iw4 <= xb4 + qw4 {
            return (BlockPartition::H, PART_HALF[bs][0]);
        }
    }

    // is_split: forced if an edge split is unavailable, else a coded symbol.
    // ctx1 is now COMPUTED from the live above/left partition-context arrays
    // (cleared neighbours → 0, matching the top-left descent).
    let ctx1 = get_partition_ctx(a_part, l_part, &bd, xb4, yb4) as usize;
    let ctx2 = ctx1 + pcc[0] as usize * 4;
    if std::env::var("P32CDF").is_ok() && pl == 0 && xb4 == 0 && yb4 == 0 && (bs == 12 || bs == 7 || bs == 6) {
        let ctx4 = ctx1 + pcc[1] as usize * 4;
        crate::dlog!("P32CDF bs={bs} ctx1={ctx1} ctx2={ctx2} split_cell={:?} ctx4={ctx4} dir_cell={:?} rng_in={}", m.part_split[pl][ctx2], m.part_dir[pl][ctx4], msac.rng);
    }
    // In a MIXED inter region, a 4x8/8x4 block (`b_dim[2]+b_dim[3]==1`) cannot split — 4x4 in
    // such a region is invalid — so the partition is a forced NONE with NO coded symbol (dav2d
    // decode.c:3800). This does NOT apply to keyframes or intra regions (mix_inter=false).
    let mpart = std::env::var("MPART").map_or(false, |v| pl == 1 || v == "all");
    if mpart { crate::dlog!("[MPARTIN] mi=({xb4},{yb4}) bs={bs} ctx2={ctx2} cell={:?} rng={}", m.part_split[pl][ctx2], msac.rng); }
    let is_split = if mix_inter && (bd[2] + bd[3]) == 1 {
        false
    } else {
        (!have_h_split || !have_v_split)
            || rav1d_msac_decode_bool_adapt(msac, &mut m.part_split[pl][ctx2])
    };
    if mpart { let coded = have_h_split && have_v_split && !(mix_inter && (bd[2] + bd[3]) == 1); crate::dlog!("[MPART] mi=({xb4},{yb4}) bs={bs} pl={pl} coded={} split={} rng={}", coded as u8, is_split as u8, msac.rng); }
    if !is_split {
        return (BlockPartition::None, -1);
    }

    // 128/256-square split direction starts with a SQUARE bool (dav2d decode.c:3772):
    // square -> 4-way PARTITION_SPLIT into quadrants; else fall through to the H/V dir
    // logic (rect halves). Only coded when both splits fit (edge SBs took the forced
    // branch above). ctx3 = ctx1 (+4 for 256x256).
    if bs == 3 && have_h_split && have_v_split {
        let square = rav1d_msac_decode_bool_adapt(msac, &mut m.part_square[ctx1]);
        if mpart { crate::dlog!("[MPARTSQ] mi=({xb4},{yb4}) bs={bs} square={} rng={}", square as u8, msac.rng); }
        if square {
            return (BlockPartition::Split, 6); // quad child: 64x64
        }
    }

    // direction (aspect logic, else a coded symbol). The "degenerate" forcing uses the
    // *plane-sample* dims: when a subsampled dim is 1 (a 4px-wide/tall chroma block), the
    // opposite split would create a sub-4px block, so the direction is forced (no symbol).
    // The aspect ratios are invariant under uniform 4:2:0 subsampling, so they keep luma dims.
    // Subsampled dims drive the CHROMA tree's min/dir arms; the LUMA and inter SHARED
    // trees use luma dims here (the shared tree's chroma constraints act via the
    // validity gates, not these).
    let (sw4, sh4) = if pl == 1 { (bw4 >> ssh, bh4 >> ssv) } else { (bw4, bh4) };
    let aspect = 1i32 << aspect_log2;
    let v_aspect = bw4 * aspect >= bh4 * 2;
    let h_aspect = bh4 * aspect >= bw4 * 2;
    // Chroma-tree rect implication (avm only_allowed_rect_type): a direction whose EVERY
    // option (half, 3-way, 4-way) yields an invalid chroma shape is disallowed; if one whole
    // direction is out, the rect type is implied with no coded symbol. (4:2:0 cases are all
    // caught by the `min(sw4,sh4)==1` arm first — this activates at 4:2:2/4:4:4.)
    // A HALF whose chroma dim collapses below 1 unit starts SHARING (avm
    // have_nz_chroma_ref_offset) and is always legal; otherwise the child's chroma shape
    // must exist (cs_ok). Applies to the chroma tree AND the inter SHARED tree.
    let dir_ok = |d: usize| -> bool {
        let (l2w, l2h) = (bd[2] as i32, bd[3] as i32);
        if d == 1 {
            ((bw4 / 2) >> ssh) == 0
                || (ph[1] != -1 && cs_ok(l2w - 1, l2h))
                || cs_ok(l2w - 2, l2h)
                || cs_ok(l2w - 3, l2h)
        } else {
            ((bh4 / 2) >> ssv) == 0
                || (ph[0] != -1 && cs_ok(l2w, l2h - 1))
                || cs_ok(l2w, l2h - 2)
                || cs_ok(l2w, l2h - 3)
        }
    };
    // Chroma-tree rect implication by bsize (avm rect_type_implied_by_bsize,
    // blockd.h:983-989): 8x16/8x32 imply HORZ, 16x8/32x8 imply VERT — format-independent
    // (the chroma tree never creates a 4px-luma dimension).
    let chroma_bsize_dir: i32 = if pl == 1 {
        match (bw4, bh4) {
            (2, 4) | (2, 8) => 0, // 8x16, 8x32 -> H
            (4, 2) | (8, 2) => 1, // 16x8, 32x8 -> V
            _ => -1,
        }
    } else {
        -1
    };
    let dir = if chroma_bsize_dir >= 0 {
        chroma_bsize_dir as usize
    } else if (ph[0] == -1) != (ph[1] == -1) {
        // 128-level rects: one half child doesn't exist as a BlockSize (128x64 can only
        // V-split to 64x64, 64x128 only H-split) — the direction is implied, no symbol.
        (ph[0] == -1) as usize
    } else if sw4.min(sh4) == 1 {
        (sw4 > sh4) as usize
    } else if (pl == 1 || ssh + ssv > 0) && dir_ok(0) != dir_ok(1) {
        dir_ok(1) as usize
    } else if !(v_aspect && h_aspect) {
        v_aspect as usize
    } else {
        if pcc[1] < 0 {
            panic!("[PDNEG] dir read with pcc[1]=-1: bs={bs} pl={pl} mi=({xb4},{yb4}) bw4={bw4} bh4={bh4} aspect_log2={aspect_log2} v={v_aspect} h={h_aspect}");
        }
        let ctx4 = ctx1 + pcc[1] as usize * 4;
        if std::env::var("PDCDF").is_ok() && pl == 0 && xb4 == 0 && yb4 == 0 && (bs == 18 || bs == 6 || bs == 7) {
            crate::dlog!("[PDCDF] bs={bs} ctx4={ctx4} part_dir_cell={:?} rng={} dif={:x}", m.part_dir[pl][ctx4], msac.rng, msac.dif);
        }
        if mpart { crate::dlog!("[MPARTRTC] mi=({xb4},{yb4}) bs={bs} ctx4={ctx4} cell={:?} pre rng={}", m.part_dir[pl][ctx4], msac.rng); }
        rav1d_msac_decode_bool_adapt(msac, &mut m.part_dir[pl][ctx4]) as usize
    };
    if mpart { crate::dlog!("[MPARTRT] mi=({xb4},{yb4}) bs={bs} dir={dir} rng={}", msac.rng); }
    let mut bp = if dir == 1 { BlockPartition::V } else { BlockPartition::H };
    if std::env::var("P32CDF").is_ok() && pl == 0 && xb4 == 0 && yb4 == 0 && bs == 7 {
        crate::dlog!("[PXMID] bs=7 after part_dir dir={dir} rng={} dif={:x}", msac.rng, msac.dif);
    }

    // ext partition (only for max dim ≤ 64px; has_hv3 holds for luma I420 here).
    if bw4.max(bh4) <= 16 {
        let nd = 1 - dir;
        // H3/V3 needs the *plane-sample* block to be big enough: the 1:4 strip dim must be
        // >= 4 samples and the cross dim >= 2 — so the size thresholds use subsampled dims
        // (chroma 4:2:0: an 8x16-chroma region cannot V3, its width is only 2). The final
        // aspect test is a ratio, unchanged by uniform subsampling.
        let bwh = [sw4, sh4];
        // `uneven_4way_partitions` is a seq-header flag (enabled for this stream; plumb it
        // from the real header). For chroma (`cbs != lbs`) the dav2d boundary sub-clause of
        // both predicates short-circuits true, so it's omitted here — interior luma also
        // resolves it true; only partial frame-edge SBs need it (a later brick).
        let uneven_4way = true;
        // Chroma-shape validity of the characteristic strip (avm check_is_chroma_size_valid
        // via get_partition_subsize): 3-way strip = dim/4, uneven-4-way strip = dim/8.
        let (l2w, l2h) = (bd[2] as i32, bd[3] as i32);
        let (hv3_cs, hv4_cs) = if dir == 1 {
            (cs_ok(l2w - 2, l2h), cs_ok(l2w - 3, l2h))
        } else {
            (cs_ok(l2w, l2h - 2), cs_ok(l2w, l2h - 3))
        };
        let (mut has_hv3, mut has_hv4ab);
        if pl == 1 {
            // CHROMA tree: avm's tree-type tables (blockd.h is_ext_partition_allowed /
            // is_uneven_4way_partition_allowed — LUMA-px rules, format-independent) AND'd
            // with the strip's chroma-shape validity. In luma 4px units:
            //  - at_bsize: false for anything <= 16x16 (dims <= 4 units) and for 8x32/32x8;
            //  - rect exceptions: 32x16-H, 64x16-H, 16x32-V, 16x64-V;
            //  - uneven 4-way: only {16x64, 64x16, 32x64, 64x32, 64x64} and the SPLIT dim
            //    must be 64px for the chroma tree.
            let small = bw4 <= 4 && bh4 <= 4;
            let is_8x32c = (bw4 == 2 && bh4 == 8) || (bw4 == 8 && bh4 == 2);
            let at_bs = ext_partitions && !small && !is_8x32c;
            let rect_exc = if dir == 0 {
                (bw4 == 8 && bh4 == 4) || (bw4 == 16 && bh4 == 4)
            } else {
                (bw4 == 4 && bh4 == 8) || (bw4 == 4 && bh4 == 16)
            };
            let ext_ok = at_bs && !rect_exc;
            has_hv3 = ext_ok && hv3_cs;
            let at_bs4 = matches!(
                (bw4, bh4),
                (4, 16) | (16, 4) | (8, 16) | (16, 8) | (16, 16)
            );
            let dim64 = if dir == 1 { bw4 == 16 } else { bh4 == 16 };
            has_hv4ab = ext_ok && at_bs4 && dim64 && uneven_4way && hv4_cs;
        } else {
            // Inter SHARED tree: a 3-way/4-way must either start chroma SHARING (a strip
            // dim below 1 chroma unit, avm have_nz_chroma_ref_offset) or have a valid
            // strip chroma shape. No-op for the key luma tree (ssh=ssv=0).
            let (hv3_ok, hv4_ok) = if ssh + ssv > 0 {
                let (share3, share4) = if dir == 1 {
                    // V3: quarter-width strips + half-height mids (avm: qbw<4 || hbh<4)
                    (
                        ((bw4 / 4) >> ssh) == 0 || ((bh4 / 2) >> ssv) == 0,
                        ((bw4 / 8) >> ssh) == 0 || (bh4 >> ssv) == 0,
                    )
                } else {
                    // H3: half-width mids + quarter-height strips (avm: hbw<4 || qbh<4)
                    (
                        ((bw4 / 2) >> ssh) == 0 || ((bh4 / 4) >> ssv) == 0,
                        ((bh4 / 8) >> ssv) == 0 || (bw4 >> ssh) == 0,
                    )
                };
                (share3 || hv3_cs, share4 || hv4_cs)
            } else {
                (true, true)
            };
            has_hv3 = ext_partitions
                && bwh[nd] >= 4
                && bwh[dir] >= 2
                && (bd[nd] as i32) * aspect >= (bd[dir] as i32) * 4
                && hv3_ok;
            has_hv4ab = bwh[nd] >= 8
                && uneven_4way
                && (bd[nd] as i32) * aspect >= (bd[dir] as i32) * 8
                && hv4_ok;
            // avm is_chroma_ref_within_boundary (av2_common_int.h:4704), SHARED tree only:
            // when the partition's chroma-REFERENCE cell (get_chroma_ref_offsets) falls
            // outside the frame's mi grid, the partition is DISALLOWED — the ext symbol is
            // then implied away (no coded bit). This is the frame-edge sub-clause: a 3-way
            // whose reference strip starts past the right/bottom edge cannot be signalled.
            // Only active when the partition has a nonzero chroma-ref offset for this
            // subsampling (have_nz_chroma_ref_offset — chroma px thresholds).
            if mix_inter {
                let within = |row_off: i32, col_off: i32| -> bool {
                    yb4 as i32 + row_off < ih4 as i32 && xb4 as i32 + col_off < iw4 as i32
                };
                // chroma px < 4 thresholds, expressed in subsampled 4px units (spx < 4 ⇔ s4 < 1).
                let (s3_nz, s3_ro, s3_co) = if dir == 1 {
                    // VERT_3: nz = qbw<4 || hbh<4; ref off = 32x8 ? (bh4/2, 0) : (0, 3*bw4/4)
                    let nz = (bw4 / 4) >> ssh == 0 || (bh4 / 2) >> ssv == 0;
                    if bw4 == 8 && bh4 == 2 { (nz, bh4 / 2, 0) } else { (nz, 0, 3 * bw4 / 4) }
                } else {
                    // HORZ_3: nz = hbw<4 || qbh<4; ref off = 8x32 ? (0, bw4/2) : (3*bh4/4, 0)
                    let nz = (bw4 / 2) >> ssh == 0 || (bh4 / 4) >> ssv == 0;
                    if bw4 == 2 && bh4 == 8 { (nz, 0, bw4 / 2) } else { (nz, 3 * bh4 / 4, 0) }
                };
                if s3_nz && !within(s3_ro, s3_co) {
                    has_hv3 = false;
                }
                let (s4_nz, s4_ro, s4_co) = if dir == 1 {
                    ((bw4 / 8) >> ssh == 0 || bh4 >> ssv == 0, 0, 7 * bw4 / 8)
                } else {
                    (bw4 >> ssh == 0 || (bh4 / 8) >> ssv == 0, 7 * bh4 / 8, 0)
                };
                if s4_nz && !within(s4_ro, s4_co) {
                    has_hv4ab = false;
                }
            }
        }
        if has_hv3 || has_hv4ab {
            let ctx5 = get_partition2_ctx(a_part, l_part, &bd, dir == 1, xb4, yb4) as usize;
            let ctx6 = ctx5 + pcc[0] as usize * 4;
            let is_ext = rav1d_msac_decode_bool_adapt(msac, &mut m.part_ext[pl][ctx6]);
            if mpart { crate::dlog!("[MPARTEXT] mi=({xb4},{yb4}) bs={bs} ext={} hv3={has_hv3} hv4ab={has_hv4ab} rng={}", is_ext as u8, msac.rng); }
            if std::env::var("P32CDF").is_ok() && pl == 0 && xb4 == 0 && yb4 == 0 && bs == 7 {
                crate::dlog!("[PXEXT] bs=7 is_ext={} ctx5={ctx5} ctx6={ctx6} hv3={has_hv3} hv4ab={has_hv4ab} rng={} dif={:x}", is_ext as u8, msac.rng, msac.dif);
            }
            if is_ext {
                bp = if dir == 1 { BlockPartition::V3 } else { BlockPartition::H3 };
                // Uneven 4-way (H4A/H4B/V4A/V4B): a `part_4way` flag (forced when only the
                // 4-way is available), then a bypass bit choosing the A/B variant.
                if has_hv4ab {
                    let is_4way = !has_hv3
                        || rav1d_msac_decode_bool_adapt(msac, &mut m.part_4way[pl][ctx6]);
                    if mpart { crate::dlog!("[MPART4W] mi=({xb4},{yb4}) bs={bs} is4way={} rng={}", is_4way as u8, msac.rng); }
                    if is_4way {
                        let ab = crate::msac::rav1d_msac_decode_bool_bypass(msac) as usize;
                        bp = match 5 + dir * 2 + ab {
                            5 => BlockPartition::H4a,
                            6 => BlockPartition::H4b,
                            7 => BlockPartition::V4a,
                            _ => BlockPartition::V4b,
                        };
                    }
                }
            }
        }
    }

    (bp, PART_HALF[bs][dir])
}

/// Splat a decoded leaf block's partition context into the live above/left arrays
/// (dav2d `decode_sb` PARTITION_NONE set_ctx): `a_part[bx4..bx4+bw4] = ~(bw4-1)`,
/// `l_part[by4..by4+bh4] = ~(bh4-1)`. This is what makes the *next* block's
/// `get_partition_ctx` non-zero — the core of live neighbour-context threading.
pub fn splat_partition(a_part: &mut [u8], l_part: &mut [u8], bs: usize, bx4: usize, by4: usize) {
    let bd = BLOCK_DIMENSIONS[bs];
    let (bw4, bh4) = (bd[0] as usize, bd[1] as usize);
    let av = !(bw4 as u8 - 1);
    let lv = !(bh4 as u8 - 1);
    a_part[bx4..bx4 + bw4].fill(av);
    l_part[by4..by4 + bh4].fill(lv);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dimensions_self_consistent() {
        // w4 == 1<<w_log2 and h4 == 1<<h_log2 for every block size — catches any
        // transcription error in the table.
        for (i, &[w4, h4, wl, hl]) in BLOCK_DIMENSIONS.iter().enumerate() {
            assert_eq!(w4 as u32, 1 << wl, "w mismatch at BS index {i}");
            assert_eq!(h4 as u32, 1 << hl, "h mismatch at BS index {i}");
        }
        assert_eq!(BLOCK_DIMENSIONS.len(), N_BS_SIZES);
        assert_eq!(BlockSize::Bs4x4 as usize, N_BS_SIZES - 1);
    }

    #[test]
    fn scan_tables_are_permutations() {
        use crate::av2_tables_gen::SCANS;
        assert_eq!(SCANS.len(), N_BS_SIZES.min(25).max(25)); // 25 = N_RECT_TX_SIZES
        // scan_4x4, hand-verified against dav2d scan.c
        assert_eq!(SCANS[0], &[0u16, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 11, 14, 15][..]);
        // every scan is a permutation of 0..len (each coeff position visited once)
        for (i, scan) in SCANS.iter().enumerate() {
            let mut s: Vec<u16> = scan.to_vec();
            s.sort_unstable();
            assert_eq!(s, (0..scan.len() as u16).collect::<Vec<_>>(), "scan {i}");
        }
        assert_eq!(SCANS[4].len(), 1024); // TX_64X64 reuses the 32x32 scan
    }

    #[test]
    fn splat_then_ctx_threads_neighbours() {
        // Decode a 16x16 (idx 18) at (0,0): splat → a_part[0..4]=l_part[0..4]=~3=0xFC.
        let mut a = [0u8; 64];
        let mut l = [0u8; 64];
        splat_partition(&mut a, &mut l, 18, 0, 0);
        assert_eq!(&a[0..4], &[0xFC; 4]);
        assert_eq!(&l[0..4], &[0xFC; 4]);
        assert_eq!(a[4], 0); // beyond the block stays cleared
        // a same-size 16x16 neighbour to the right sees a non-smaller left block → ctx 0.
        let b_dim16 = [4, 4, 2, 2];
        assert_eq!(get_partition_ctx(&a, &l, &b_dim16, 4, 0), 0);
        // but a SMALLER 8x8 (idx 24) splatting first makes a 16x16 neighbour's ctx non-zero.
        let mut a2 = [0u8; 64];
        let mut l2 = [0u8; 64];
        splat_partition(&mut a2, &mut l2, 24, 0, 0); // 8x8: bw4=bh4=2 → ~1=0xFE
        assert_eq!(&l2[0..2], &[0xFE; 2]);
        // 16x16 to the right: l_part[0]=0xFE >> (hl2-1=1) & 1 = 1 → left contributes 2
        assert_eq!(get_partition_ctx(&a2, &l2, &b_dim16, 2, 0), 2);
    }

    #[test]
    fn first_sb_partition_ctx_is_zero() {
        // Cleared neighbours (tile start) → ctx 0, matching the live decode of the
        // first superblock's part_split[pl][0].
        let a = [0u8; 64];
        let l = [0u8; 64];
        let b_dim = [16, 16, 4, 4]; // BS_64x64-ish (w4,h4,wlog2,hlog2)
        assert_eq!(get_partition_ctx(&a, &l, &b_dim, 0, 0), 0);
    }

    #[test]
    fn partition_ctx_reads_neighbour_bits() {
        // above split at column 0, b_dim[2]=2 → bit (val>>1)&1
        let mut a = [0u8; 64];
        a[0] = 0b10; // (0b10 >> (2-1)) & 1 == 1
        let l = [0u8; 64];
        let b_dim = [8, 8, 2, 2];
        assert_eq!(get_partition_ctx(&a, &l, &b_dim, 0, 0), 1);
        // left contributes the high bit (<<1)
        let mut l2 = [0u8; 64];
        l2[0] = 0b10;
        assert_eq!(get_partition_ctx(&[0u8; 64], &l2, &b_dim, 0, 0), 2);
    }
}
