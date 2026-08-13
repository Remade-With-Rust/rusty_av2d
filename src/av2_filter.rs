//! AV2 in-loop filter primitives — CDEF (constrained directional enhancement,
//! shared with AV1) and CCSO (cross-component sample offset, AV2-new). dav2d
//! `cdef_tmpl.c` / `ccso_tmpl.c`. These are the per-sample cores; the full block
//! filters (padding + directional accumulation) compose on top of them.

#[inline]
fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 {
        -v
    } else {
        v
    }
}

/// CDEF constrained difference (dav2d `constrain`): passes small differences
/// through, but tapers a difference toward 0 once its magnitude grows past
/// `threshold` (edge-preserving). `threshold == 0` yields 0. Sign is preserved.
pub fn constrain(diff: i32, threshold: i32, shift: u32) -> i32 {
    let adiff = diff.abs();
    apply_sign(adiff.min(0.max(threshold - (adiff >> shift))), diff)
}

/// CDEF directional tap offsets (dav2d `dav2d_cdef_directions`) into a stride-12
/// working buffer. Indexed `[dir+k]`: the filter reads `dir+2` (primary), `dir+4`
/// and `dir+0` (secondary); the 2-row pre/post padding makes those in-range.
pub static CDEF_DIRECTIONS: [[i32; 2]; 12] = [
    [1 * 12, 2 * 12],         // 6
    [1 * 12, 2 * 12 - 1],     // 7
    [-1 * 12 + 1, -2 * 12 + 2], // 0
    [1, -1 * 12 + 2],         // 1
    [1, 2],                   // 2
    [1, 1 * 12 + 2],          // 3
    [1 * 12 + 1, 2 * 12 + 2], // 4
    [1 * 12, 2 * 12 + 1],     // 5
    [1 * 12, 2 * 12],         // 6
    [1 * 12, 2 * 12 - 1],     // 7
    [-1 * 12 + 1, -2 * 12 + 2], // 0
    [1, -1 * 12 + 2],         // 1
];

/// CCSO filter position offsets (dav2d `ccso_pos`), `[ext_filter] = [dy, dx]`.
pub static CCSO_POS: [[i8; 2]; 7] = [
    [-1, 0],
    [0, -1],
    [-1, -1],
    [-1, 1],
    [-1, -2],
    [1, -2],
    [0, 2],
];

/// CCSO 3-way edge classifier (dav2d `ccso_score`, AV2-new): `2` when `diff`
/// exceeds `+quant_step` and the sample is not an edge, `0` when below
/// `-quant_step`, otherwise `1`. Drives the cross-component offset lookup.
pub fn ccso_score(diff: i32, quant_step: i32, edge_classifier: bool) -> u32 {
    if diff > quant_step && !edge_classifier {
        2
    } else if diff < -quant_step {
        0
    } else {
        1
    }
}

/// Working-buffer stride for the CDEF block filter (matches `CDEF_DIRECTIONS`).
pub const CDEF_TMP_STRIDE: usize = 12;

#[inline]
fn ulog2(x: i32) -> i32 {
    31 - (x as u32).leading_zeros() as i32
}

/// Unsigned min (dav2d `umin`) — the CDEF local-range clamp compares as unsigned so the
/// `INT16_MIN` out-of-frame sentinel (huge unsigned) never lowers the block minimum.
#[inline]
fn umin(a: i32, b: i32) -> i32 {
    ((a as u32).min(b as u32)) as i32
}

/// The CDEF out-of-frame padding sentinel (dav2d `fill` uses `INT16_MIN`): `constrain`
/// tapers it to 0 (huge |diff|), and `umin`/`max` ignore it in the local-range clamp.
pub const CDEF_VERY_LARGE: i32 = i16::MIN as i32;

#[inline]
fn bitdepth_from_max(m: i32) -> i32 {
    32 - (m as u32).leading_zeros() as i32
}

/// CDEF block filter (dav2d `cdef_filter_block_c`). `tmp` is the padded input
/// (stride [`CDEF_TMP_STRIDE`], the `w`×`h` block's top-left at index `2*12+2`);
/// writes the filtered block to `dst` (row stride `dst_stride`). `dir` is the
/// detected direction 0..7. Primary taps run along `dir`, secondary along `dir±2`,
/// each passed through [`constrain`]; with both strengths the result is clamped to
/// the local tap min/max. (Interior case — the edge fill-sentinel padding composes
/// on top by zeroing out-of-frame taps.)
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_block(dst: &mut [i32], dst_stride: usize, tmp: &[i32], pri_strength: i32, sec_strength: i32,
                         dir: usize, damping: i32, w: usize, h: usize, bitdepth_max: i32) {
    let bd_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    let origin = (2 * CDEF_TMP_STRIDE + 2) as i32;
    let s = CDEF_TMP_STRIDE as i32;
    let pri_tap = 4 - ((pri_strength >> bd_min_8) & 1);
    let pri_shift = 0.max(damping - if pri_strength > 0 { ulog2(pri_strength) } else { 0 }) as u32;
    let sec_shift = if sec_strength > 0 { (damping - ulog2(sec_strength)) as u32 } else { 0 };
    let both = pri_strength > 0 && sec_strength > 0;

    for y in 0..h {
        if !crate::av2_recon::work_tick("filter:107") { break; }
        for x in 0..w {
            let c = origin + y as i32 * s + x as i32;
            let px = tmp[c as usize];
            let mut sum = 0i32;
            let (mut mn, mut mx) = (px, px);
            if pri_strength > 0 {
                let mut ptk = pri_tap;
                for k in 0..2 {
                    let off = CDEF_DIRECTIONS[dir + 2][k];
                    let p0 = tmp[(c + off) as usize];
                    let p1 = tmp[(c - off) as usize];
                    sum += ptk * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptk * constrain(p1 - px, pri_strength, pri_shift);
                    ptk = (ptk & 3) | 2;
                    mn = umin(umin(mn, p0), p1);
                    mx = mx.max(p0).max(p1);
                }
            }
            if sec_strength > 0 {
                for k in 0..2 {
                    let off2 = CDEF_DIRECTIONS[dir + 4][k];
                    let off3 = CDEF_DIRECTIONS[dir][k];
                    let s0 = tmp[(c + off2) as usize];
                    let s1 = tmp[(c - off2) as usize];
                    let s2 = tmp[(c + off3) as usize];
                    let s3 = tmp[(c - off3) as usize];
                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                    mn = umin(umin(umin(umin(mn, s0), s1), s2), s3);
                    mx = mx.max(s0).max(s1).max(s2).max(s3);
                }
            }
            let filtered = px + ((sum - (sum < 0) as i32 + 8) >> 4);
            // HARDENING: corrupt block geometry can exceed the destination plane slice.
            let o = y * dst_stride + x;
            if o >= dst.len() { continue; }
            dst[o] = if both { filtered.clamp(mn, mx) } else { filtered };
        }
    }
}

/// CDEF direction search (dav2d `cdef_find_dir_c`): finds the dominant edge direction
/// (0..7) of the 8×8 luma block at `img[origin..]` (row stride `stride`) and its variance.
/// Accumulators wrap as `u32` (matching C's unsigned cost arithmetic).
pub fn cdef_find_dir(img: &[i32], origin: usize, stride: usize, bitdepth_max: i32) -> (usize, u32) {
    let bd_min8 = bitdepth_from_max(bitdepth_max) - 8;
    let mut ps_hv = [[0i32; 8]; 2];
    let mut ps_diag = [[0i32; 15]; 2];
    let mut ps_alt = [[0i32; 11]; 4];
    for y in 0..8usize {
        if !crate::av2_recon::work_tick("filter:160") { break; }
        for x in 0..8usize {
            // HARDENING: a corrupt stream can place the CDEF direction window past the plane
            // (the 8x8 block origin is derived from block geometry) — read 0 outside it.
            let px = (img.get(origin + y * stride + x).copied().unwrap_or(0) >> bd_min8) - 128;
            ps_diag[0][y + x] += px;
            ps_alt[0][y + (x >> 1)] += px;
            ps_hv[0][y] += px;
            ps_alt[1][3 + y - (x >> 1)] += px;
            ps_diag[1][7 + y - x] += px;
            ps_alt[2][3 - (y >> 1) + x] += px;
            ps_hv[1][x] += px;
            ps_alt[3][(y >> 1) + x] += px;
        }
    }
    let sq = |v: i32| (v * v) as u32;
    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] = cost[2].wrapping_add(sq(ps_hv[0][n]));
        cost[6] = cost[6].wrapping_add(sq(ps_hv[1][n]));
    }
    cost[2] = cost[2].wrapping_mul(105);
    cost[6] = cost[6].wrapping_mul(105);
    const DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n];
        cost[0] = cost[0].wrapping_add((sq(ps_diag[0][n]).wrapping_add(sq(ps_diag[0][14 - n]))).wrapping_mul(d));
        cost[4] = cost[4].wrapping_add((sq(ps_diag[1][n]).wrapping_add(sq(ps_diag[1][14 - n]))).wrapping_mul(d));
    }
    cost[0] = cost[0].wrapping_add(sq(ps_diag[0][7]).wrapping_mul(105));
    cost[4] = cost[4].wrapping_add(sq(ps_diag[1][7]).wrapping_mul(105));
    for n in 0..4 {
        let cp = n * 2 + 1;
        let mut c = 0u32;
        for m in 0..5 {
            c = c.wrapping_add(sq(ps_alt[n][3 + m]));
        }
        c = c.wrapping_mul(105);
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1];
            c = c.wrapping_add((sq(ps_alt[n][m]).wrapping_add(sq(ps_alt[n][10 - m]))).wrapping_mul(d));
        }
        cost[cp] = c;
    }
    let mut best_dir = 0usize;
    let mut best_cost = cost[0];
    for n in 1..8 {
        if !crate::av2_recon::work_tick("filter:206") { break; }
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }
    let var = best_cost.wrapping_sub(cost[best_dir ^ 4]) >> 10;
    (best_dir, var)
}

/// CDEF variance-adaptive primary-strength scaling (dav2d `adjust_strength`).
pub fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 { ulog2((var >> 6) as i32).min(12) } else { 0 };
    (strength * (4 + i) + 8) >> 4
}

/// Filter one CDEF block. Builds the padded `tmp` (dav2d `padding`) from the pristine
/// `input` (post-deblock, read-only) around the block at `in_off` — out-of-frame taps get
/// [`CDEF_VERY_LARGE`] per the `have_*` edge flags — then applies [`cdef_filter_block`],
/// writing the filtered block to `output` at `out_off`. Because `input` is untouched, no
/// line-buffer backups are needed: every neighbour tap reads the pre-CDEF frame directly.
#[allow(clippy::too_many_arguments)]
pub fn cdef_block(
    output: &mut [i32],
    out_off: usize,
    out_stride: usize,
    input: &[i32],
    in_off: usize,
    in_stride: usize,
    w: usize,
    h: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
    have_top: bool,
    have_bottom: bool,
    have_left: bool,
    have_right: bool,
    bitdepth_max: i32,
) {
    crate::prof_scope!(7);
    const TS: isize = CDEF_TMP_STRIDE as isize;
    let mut tmp = [CDEF_VERY_LARGE; CDEF_TMP_STRIDE * 12];
    let y_start: isize = if have_top { -2 } else { 0 };
    let y_end: isize = if have_bottom { h as isize + 2 } else { h as isize };
    let x_start: isize = if have_left { -2 } else { 0 };
    let x_end: isize = if have_right { w as isize + 2 } else { w as isize };
    let base = 2 * TS + 2;
    let ins = in_stride as isize;
    for dy in y_start..y_end {
        if !crate::av2_recon::work_tick("filter:258") { break; }
        for dx in x_start..x_end {
            let src = (in_off as isize + dy * ins + dx) as usize;
            let ti = (base + dy * TS + dx) as usize;
            // HARDENING: corrupt geometry can push the CDEF source window past the plane
            // (and the tmp index past the scratch) — skip those taps instead of panicking.
            if ti >= tmp.len() { continue; }
            tmp[ti] = input.get(src).copied().unwrap_or(0);
        }
    }
    cdef_filter_block(&mut output[out_off..], out_stride, &tmp, pri, sec, dir, damping, w, h, bitdepth_max);
}

/// CCSO band index (dav2d `ccso_prep_bo`): the high `max_band_log2` bits of a
/// sample (`sample >> shift`, `shift = bitdepth - max_band_log2`).
pub fn ccso_band(sample: i32, shift: u32) -> u8 {
    (sample >> shift) as u8
}

/// CCSO per-pixel classifier index (dav2d `ccso_prep_clf` core): packs the two
/// edge classes (luma at `±offset` vs the centre `c`) with the band into one byte:
/// `(cls0 << 5) | (cls1 << 3) | band`. Composes on [`ccso_score`].
pub fn ccso_classify(c: i32, luma_plus: i32, luma_minus: i32, shift: u32, quant_step: i32, edge_clf: bool) -> u8 {
    let band = (c >> shift) as u8;
    let cls0 = ccso_score(luma_plus - c, quant_step, edge_clf) as u8;
    let cls1 = ccso_score(luma_minus - c, quant_step, edge_clf) as u8;
    (cls0 << 5) | (cls1 << 3) | band
}

/// CCSO offset application (dav2d `ccso_add` core): unpack a per-pixel classifier
/// index into an `offset_lut` entry (via the packed `offset_idxs`) and add it to
/// the sample, clipped to `[0, bitdepth_max]`.
pub fn ccso_apply(sample: i32, idx: u8, offset_idxs: &[u8], offset_lut: &[i8], bitdepth_max: i32) -> i32 {
    let byte_idx = (idx >> 1) as usize;
    let half_idx = (idx & 1) as u32;
    let offset_idx = (7 & (offset_idxs[byte_idx] >> (4 * half_idx))) as usize;
    (sample + offset_lut[offset_idx] as i32).clamp(0, bitdepth_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrain_zero_and_passthrough() {
        assert_eq!(constrain(0, 10, 1), 0);
        // small diff, generous threshold → passes through unchanged
        assert_eq!(constrain(2, 10, 1), 2);
        assert_eq!(constrain(-2, 10, 1), -2);
    }

    #[test]
    fn constrain_tapers_large_diff() {
        // |diff|=20, threshold 4, shift 1 → max(0, 4-10)=0 → tapered to 0
        assert_eq!(constrain(20, 4, 1), 0);
        // |diff|=8, threshold 10, shift 1 → min(8, 10-4=6)=6, sign preserved
        assert_eq!(constrain(8, 10, 1), 6);
        assert_eq!(constrain(-8, 10, 1), -6);
    }

    #[test]
    fn constrain_zero_threshold_is_zero() {
        assert_eq!(constrain(50, 0, 3), 0);
    }

    #[test]
    fn cdef_directions_wraparound() {
        // the table repeats: [0..2) == [8..10), [2..4) == [10..12) (the dir wrap).
        assert_eq!(CDEF_DIRECTIONS[0], CDEF_DIRECTIONS[8]);
        assert_eq!(CDEF_DIRECTIONS[1], CDEF_DIRECTIONS[9]);
        assert_eq!(CDEF_DIRECTIONS[2], CDEF_DIRECTIONS[10]);
        assert_eq!(CDEF_DIRECTIONS[3], CDEF_DIRECTIONS[11]);
        // direction 2 (horizontal): primary offset is +1 column, secondary +2.
        assert_eq!(CDEF_DIRECTIONS[4], [1, 2]);
    }

    #[test]
    fn cdef_flat_block_is_unchanged() {
        // Flat input → every difference is 0 → constrain 0 → sum 0 → px unchanged,
        // for every strength combination and direction.
        for &(pri, sec) in &[(4, 4), (4, 0), (0, 4)] {
            for dir in 0..8 {
                let tmp = [137i32; 144];
                let mut dst = [0i32; 8 * 8];
                cdef_filter_block(&mut dst, 8, &tmp, pri, sec, dir, 3, 8, 8, 255);
                assert!(dst.iter().all(|&p| p == 137), "pri={pri} sec={sec} dir={dir}");
            }
        }
    }

    #[test]
    fn cdef_stays_within_local_range() {
        // pri+sec clamps each output to its local tap min/max, so a textured input
        // is never pushed outside its own value range.
        let mut tmp = [0i32; 144];
        for (i, t) in tmp.iter_mut().enumerate() {
            *t = 100 + (i as i32 * 7) % 11; // values in [100, 110]
        }
        let mut dst = [0i32; 8 * 8];
        cdef_filter_block(&mut dst, 8, &tmp, 8, 8, 2, 3, 8, 8, 255);
        assert!(dst.iter().all(|&p| (100..=110).contains(&p)));
        // and it actually filters (some pixel changed vs the raw input)
        let origin = 2 * CDEF_TMP_STRIDE + 2;
        let changed = (0..8).any(|y| (0..8).any(|x| dst[y * 8 + x] != tmp[origin + y * CDEF_TMP_STRIDE + x]));
        assert!(changed, "filter had no effect on textured input");
    }

    #[test]
    fn cdef_find_dir_detects_diagonal() {
        // pixel = (x+y)*8 is constant along the anti-diagonals (x+y=const), so the dominant
        // edge is that diagonal → dir 0, and its perpendicular (dir 4) has near-equal cost is
        // false here (the ramp is strongly diagonal), giving a large variance.
        let mut img = vec![0i32; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                img[y * 8 + x] = (x + y) as i32 * 8;
            }
        }
        let (dir, var) = cdef_find_dir(&img, 0, 8, 255);
        assert_eq!(dir, 0, "diagonal ramp → dir 0");
        assert!(var > 0, "strong edge → nonzero variance");
    }

    #[test]
    fn cdef_find_dir_flat_zero_variance() {
        // Flat block: every direction's cost equals its perpendicular's → variance 0.
        let img = vec![137i32; 8 * 8];
        let (_dir, var) = cdef_find_dir(&img, 0, 8, 255);
        assert_eq!(var, 0);
    }

    #[test]
    fn ccso_score_three_way() {
        assert_eq!(ccso_score(10, 5, false), 2); // above +quant, not edge
        assert_eq!(ccso_score(10, 5, true), 1); // edge suppresses the +2 class
        assert_eq!(ccso_score(-10, 5, false), 0); // below -quant
        assert_eq!(ccso_score(3, 5, false), 1); // within band
        assert_eq!(ccso_score(-3, 5, true), 1);
    }

    #[test]
    fn ccso_avm_lut_index() {
        // C3 CCSO (avm encoding): lut = (band<<4)|(cls0<<2)|cls1, band = center>>(8-max_band_log2),
        // cls = ccso_score(neighbour-center, quant, edge_clf). Locks the dev clip's pl=0 params
        // (q=8, edge_clf=false, max_band_log2=3) that produced a byte-exact CCSO on all planes.
        let center = 100i32;
        let band = center >> (8 - 3);
        let cls0 = ccso_score(120 - center, 8, false); // +20 > 8 → 2
        let cls1 = ccso_score(100 - center, 8, false); //   0     → 1
        assert_eq!((band, cls0, cls1), (3, 2, 1));
        let lut = ((band as usize) << 4) | ((cls0 as usize) << 2) | (cls1 as usize);
        assert_eq!(lut, 57);
    }

    #[test]
    fn ccso_band_high_bits() {
        assert_eq!(ccso_band(200, 5), 6); // 200 >> 5
        assert_eq!(ccso_band(255, 5), 7);
    }

    #[test]
    fn ccso_classify_packs_classes_and_band() {
        // c=100, +nbr=120 (diff +20 > quant 5 → cls0=2), -nbr=80 (diff -20 → cls1=0),
        // band = 100>>5 = 3 → (2<<5)|(0<<3)|3 = 67.
        assert_eq!(ccso_classify(100, 120, 80, 5, 5, false), (2 << 5) | 3);
        // edge_clf suppresses the +2 class → cls0=1: (1<<5)|(0<<3)|3
        assert_eq!(ccso_classify(100, 120, 80, 5, 5, true), (1 << 5) | (0 << 3) | 3);
    }

    #[test]
    fn ccso_apply_unpacks_and_offsets() {
        let offset_idxs = [0, 0, 0x35];
        let offset_lut = [0i8, 1, 2, 3, 4, 5, 6, 7];
        // idx=4 → byte 2, half 0 → 0x35 & 7 = 5 → lut[5]=5
        assert_eq!(ccso_apply(100, 4, &offset_idxs, &offset_lut, 255), 105);
        // idx=5 → byte 2, half 1 → (0x35>>4)&7 = 3 → lut[3]=3
        assert_eq!(ccso_apply(100, 5, &offset_idxs, &offset_lut, 255), 103);
        // clamp at the top
        assert_eq!(ccso_apply(254, 4, &offset_idxs, &offset_lut, 255), 255);
    }
}
