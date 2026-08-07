//! AV2 compound wedge masks — dav2d wedge.c ported. Masks are generated per block size
//! (68 codebook entries): a 128x128 "master" directional ramp (sharp mul=2 for sizes
//! <=16x16 px, soft mul=1 above), cropped per codebook offset, plus 422/420 subsampled
//! variants and the per-8x8 TMVP winner map (0/1 = single side, 2 = both).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const N_DIR: usize = 20;
const COS_LUT: [i32; N_DIR] = [4, 4, 4, 2, 2, 0, -2, -2, -4, -4, -4, -4, -4, -2, -2, 0, 2, 2, 4, 4];
const SIN_LUT: [i32; N_DIR] = [0, -1, -2, -2, -4, -4, -4, -2, -2, -1, 0, 1, 2, 2, 4, 4, 4, 2, 2, 1];
const WEIGHT: [i32; 29] = [
    8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2,
    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
];

/// (direction, x_offset, y_offset) — dav wedge_codebook_16.
#[rustfmt::skip]
const CODEBOOK: [(u8, u8, u8); 68] = [
    (0,5,4), (0,6,4), (0,7,4),
    (1,4,4), (1,5,4), (1,6,4), (1,7,4),
    (2,4,4), (2,5,4), (2,6,4), (2,7,4),
    (3,4,4), (3,5,4), (3,6,4), (3,7,4),
    (4,4,4), (4,4,3), (4,4,2), (4,4,1),
    (5,4,3), (5,4,2), (5,4,1),
    (6,4,4), (6,4,3), (6,4,2), (6,4,1),
    (7,4,4), (7,3,4), (7,2,4), (7,1,4),
    (8,4,4), (8,3,4), (8,2,4), (8,1,4),
    (9,4,4), (9,3,4), (9,2,4), (9,1,4),
    (10,3,4), (10,2,4), (10,1,4),
    (11,3,4), (11,2,4), (11,1,4),
    (12,3,4), (12,2,4), (12,1,4),
    (13,3,4), (13,2,4), (13,1,4),
    (14,4,5), (14,4,6), (14,4,7),
    (15,4,5), (15,4,6), (15,4,7),
    (16,4,5), (16,4,6), (16,4,7),
    (17,5,4), (17,6,4), (17,7,4),
    (18,5,4), (18,6,4), (18,7,4),
    (19,5,4), (19,6,4), (19,7,4),
];

fn gen_master(mul: i32, wd: usize) -> Vec<u8> {
    let mut master = vec![0u8; 128 * 128];
    let s = SIN_LUT[wd] * mul;
    let c = COS_LUT[wd] * mul;
    for y in 0..128i32 {
        if !crate::av2_recon::work_tick("wedge:46") { break; }
        let dy = (2 * y - 127) * s;
        for x in 0..128i32 {
            if !crate::av2_recon::work_tick("wedge:48") { break; }
            let d = ((2 * x - 127) * c + dy).clamp(-28, 28);
            master[(y * 128 + x) as usize] =
                (4 * if d >= 0 { 16 - WEIGHT[d as usize] } else { WEIGHT[(-d) as usize] }) as u8;
        }
    }
    master
}

fn copy2d(master: &[u8], w8: usize, h8: usize, x_off: usize, y_off: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w8 * 8 * h8 * 8];
    let mut src = (64 - y_off * h8) * 128 + (64 - x_off * w8);
    for y in 0..h8 * 8 {
        if !crate::av2_recon::work_tick("wedge:60") { break; }
        dst[y * w8 * 8..(y + 1) * w8 * 8].copy_from_slice(&master[src..src + w8 * 8]);
        src += 128;
    }
    dst
}

fn subsample_420(src: &[u8], w8: usize, h8: usize) -> Vec<u8> {
    let (w, hw) = (w8 * 8, w8 * 4);
    let mut dst = vec![0u8; hw * h8 * 4];
    for y in 0..h8 * 4 {
        if !crate::av2_recon::work_tick("wedge:70") { break; }
        for x in 0..hw {
            if !crate::av2_recon::work_tick("wedge:71") { break; }
            let s = (y * 2) * w + x * 2;
            dst[y * hw + x] =
                ((src[s] as u32 + src[s + 1] as u32 + src[s + w] as u32 + src[s + w + 1] as u32 + 2) >> 2) as u8;
        }
    }
    dst
}

fn subsample_422(src: &[u8], w8: usize, h8: usize) -> Vec<u8> {
    let (w, hw) = (w8 * 8, w8 * 4);
    let mut dst = vec![0u8; hw * h8 * 8];
    for y in 0..h8 * 8 {
        if !crate::av2_recon::work_tick("wedge:83") { break; }
        for x in 0..hw {
            if !crate::av2_recon::work_tick("wedge:84") { break; }
            let s = y * w + x * 2;
            dst[y * hw + x] = ((src[s] as u32 + src[s + 1] as u32 + 1) >> 1) as u8;
        }
    }
    dst
}

fn fill_tmvp(src: &[u8], w8: usize, h8: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w8 * h8];
    for y in 0..h8 {
        if !crate::av2_recon::work_tick("wedge:94") { break; }
        for x in 0..w8 {
            if !crate::av2_recon::work_tick("wedge:95") { break; }
            let (mut s0, mut s1) = (0u32, 0u32);
            for yy in y * 8..y * 8 + 8 {
                if !crate::av2_recon::work_tick("wedge:97") { break; }
                for xx in x * 8..x * 8 + 8 {
                    if !crate::av2_recon::work_tick("wedge:98") { break; }
                    let v = src[yy * w8 * 8 + xx];
                    s0 += (v < 4) as u32;
                    s1 += (v > 60) as u32;
                }
            }
            dst[y * w8 + x] = if s0 >= 60 { 0 } else if s1 >= 60 { 1 } else { 2 };
        }
    }
    dst
}

/// Per-(bw4,bh4) mask set: [ssidx 0/1/2][widx] pixel masks + [widx] tmvp maps.
struct WedgeSet {
    masks: [Vec<Arc<Vec<u8>>>; 3],
    tmvp: Vec<Arc<Vec<u8>>>,
}

fn build_set(bw4: usize, bh4: usize) -> WedgeSet {
    let (w8, h8) = (bw4 / 2, bh4 / 2);
    // dav init: sizes up to 16x16 px use the sharp master (mul=2), larger the soft (mul=1).
    let mul = if bw4 <= 4 && bh4 <= 4 { 2 } else { 1 };
    let mut masks: [Vec<Arc<Vec<u8>>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut tmvp = Vec::new();
    let mut wd = usize::MAX;
    let mut master = Vec::new();
    for &(dir, xo, yo) in CODEBOOK.iter() {
        if dir as usize != wd {
            master = gen_master(mul, dir as usize);
            wd = dir as usize;
        }
        let m444 = copy2d(&master, w8, h8, xo as usize, yo as usize);
        masks[1].push(Arc::new(subsample_422(&m444, w8, h8)));
        masks[2].push(Arc::new(subsample_420(&m444, w8, h8)));
        tmvp.push(Arc::new(fill_tmvp(&m444, w8, h8)));
        masks[0].push(Arc::new(m444));
    }
    WedgeSet { masks, tmvp }
}

static CACHE: OnceLock<Mutex<HashMap<(usize, usize), Arc<WedgeSet>>>> = OnceLock::new();

fn set_for(bw4: usize, bh4: usize) -> Arc<WedgeSet> {
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    map.entry((bw4, bh4)).or_insert_with(|| Arc::new(build_set(bw4, bh4))).clone()
}

/// The blend mask for a (bw4 x bh4 cells) block, wedge index `widx`, at subsampling
/// `ssidx` (0=444/luma, 1=422, 2=420). Row-major, stride = (bw4*4)>>ss_hor.
pub fn wedge_mask(bw4: usize, bh4: usize, widx: usize, ssidx: usize) -> Arc<Vec<u8>> {
    set_for(bw4, bh4).masks[ssidx][widx].clone()
}

/// The per-8x8px TMVP winner map (0/1 = that single side, 2 = both), (bw4/2 x bh4/2) cells.
pub fn wedge_tmvp(bw4: usize, bh4: usize, widx: usize) -> Arc<Vec<u8>> {
    set_for(bw4, bh4).tmvp[widx].clone()
}
