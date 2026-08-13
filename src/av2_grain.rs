//! AV2 film-grain synthesis — a verbatim port of avm `avm_dsp/grain_synthesis.c`
//! (`av2_add_film_grain_run`). The FGM OBU (type 23) carries up to 8 parameter tables;
//! each shown frame's header selects a table id + a 16-bit seed. Grain is applied to the
//! OUTPUT copy only (reference pictures stay grain-free). avmdec is the oracle for this
//! path (dav2d's grain output disagrees with avm on ~12% of pixels).

use crate::av2_grain_tables::GAUSSIAN_SEQUENCE;

const GAUSS_BITS: i32 = 11;
const MAX_LUMA_SUBBLOCK: usize = 32;

/// One FGM table slot (avm `avm_film_grain_t`, raw header values — offsets applied at use,
/// mirroring avm's synthesis which does `cb_mult - 128` etc. itself).
#[derive(Clone, Default)]
pub struct FilmGrainData {
    pub chroma_scaling_from_luma: bool,
    pub num_points: [usize; 3],
    /// scaling points \[plane\]\[i\] = (x 0..255, scaling)
    pub points: [[(i32, i32); 14]; 3],
    pub scaling_shift: i32,   // 8..11 (final)
    pub ar_coeff_lag: i32,    // 0..3
    /// signed AR coeffs (−128 applied at parse), luma/cb/cr; max 24+1 positions
    pub ar_coeffs: [[i32; 25]; 3],
    pub ar_coeff_shift: i32,  // 6..9 (final)
    pub grain_scale_shift: i32,
    /// RAW header values (avm convention): cb/cr_mult, cb/cr_luma_mult 0..255, cb/cr_offset 0..511
    pub uv_mult: [i32; 2],
    pub uv_luma_mult: [i32; 2],
    pub uv_offset: [i32; 2],
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
    pub mc_identity: bool,
    pub block_size: i32, // 0: 16x16, 1: 32x32
}

thread_local! {
    /// The 8 FGM table slots (OBU type 23).
    pub static FGM_TABLE: std::cell::RefCell<[Option<FilmGrainData>; 8]> =
        const { std::cell::RefCell::new([None, None, None, None, None, None, None, None]) };
    /// The CURRENT frame's grain selection (present → (id, seed)), from the frame header.
    pub static CUR_GRAIN: std::cell::Cell<Option<(u8, u16)>> = const { std::cell::Cell::new(None) };
    /// Per-ref-slot grain selection (parallel to REF_PICS): a queued show_implicit frame
    /// carries its own grain params to its later emit.
    pub static GRAIN_SLOTS: std::cell::RefCell<[Option<(u8, u16)>; 8]> =
        const { std::cell::RefCell::new([None; 8]) };
}

/// Store the CURRENT frame's grain selection into every refresh slot (call next to
/// update_ref_slots so queued implicit frames keep their grain).
/// Clear the FGM tables + per-slot grain refs (new-sequence reset).
pub fn reset_stream_state() {
    FGM_TABLE.with(|t| *t.borrow_mut() = std::array::from_fn(|_| None));
    GRAIN_SLOTS.with(|s| *s.borrow_mut() = std::array::from_fn(|_| None));
    CUR_GRAIN.with(|c| c.set(None));
}

pub fn stash_grain_slots(refresh: u32) {
    let g = CUR_GRAIN.with(|c| c.get());
    GRAIN_SLOTS.with(|s| {
        let mut s = s.borrow_mut();
        for i in 0..8 {
            if refresh & (1 << i) != 0 {
                s[i] = g;
            }
        }
    });
}

struct GrainRng {
    reg: u16,
}
impl GrainRng {
    fn get(&mut self, bits: i32) -> i32 {
        let r = self.reg;
        let bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        self.reg = (r >> 1) | (bit << 15);
        ((self.reg >> (16 - bits)) & ((1 << bits) - 1) as u16) as i32
    }
    fn init(&mut self, luma_line: i32, seed: u16) {
        let msb = (seed >> 8) & 255;
        let lsb = seed & 255;
        self.reg = (msb << 8) + lsb;
        let luma_num = luma_line >> 5;
        self.reg ^= (((luma_num * 37 + 178) & 255) as u16) << 8;
        self.reg ^= ((luma_num * 173 + 105) & 255) as u16;
    }
}

fn init_scaling_function(points: &[(i32, i32)], num: usize, lut: &mut [i32; 256]) {
    if num == 0 {
        return;
    }
    for i in 0..points[0].0 as usize {
        lut[i] = points[0].1;
    }
    for p in 0..num - 1 {
        let delta_y = points[p + 1].1 - points[p].1;
        let delta_x = points[p + 1].0 - points[p].0;
        // HARDENING: the scaling points must be strictly increasing in x; a corrupt FGM table
        // can repeat (or invert) an x, which divides by zero here. Skip the degenerate segment.
        if delta_x <= 0 {
            continue;
        }
        let delta = delta_y as i64 * ((65536 + (delta_x >> 1)) / delta_x) as i64;
        for x in 0..delta_x {
            lut[(points[p].0 + x) as usize] =
                points[p].1 + ((x as i64 * delta + 32768) >> 16) as i32;
        }
    }
    for i in points[num - 1].0 as usize..256 {
        lut[i] = points[num - 1].1;
    }
}

fn scale_lut(lut: &[i32; 256], index: i32, bit_depth: i32) -> i32 {
    let x = (index >> (bit_depth - 8)) as usize;
    if bit_depth == 8 || x == 255 {
        lut[x]
    } else {
        lut[x]
            + (((lut[x + 1] - lut[x]) * (index & ((1 << (bit_depth - 8)) - 1))
                + (1 << (bit_depth - 9)))
                >> (bit_depth - 8))
    }
}

fn ver_boundary_overlap(
    left: &[i32], left_stride: usize, right: &[i32], right_stride: usize,
    dst: &mut [i32], dst_stride: usize, width: usize, height: usize, gmin: i32, gmax: i32,
) {
    if width == 1 {
        for r in 0..height {
            dst[r * dst_stride] = ((left[r * left_stride] * 23 + right[r * right_stride] * 22 + 16)
                >> 5)
                .clamp(gmin, gmax);
        }
    } else if width == 2 {
        for r in 0..height {
            dst[r * dst_stride] = ((27 * left[r * left_stride] + 17 * right[r * right_stride] + 16)
                >> 5)
                .clamp(gmin, gmax);
            dst[r * dst_stride + 1] = ((17 * left[r * left_stride + 1]
                + 27 * right[r * right_stride + 1]
                + 16)
                >> 5)
                .clamp(gmin, gmax);
        }
    }
}

fn hor_boundary_overlap(
    top: &[i32], top_stride: usize, bottom: &[i32], bottom_stride: usize,
    dst: &mut [i32], dst_stride: usize, width: usize, height: usize, gmin: i32, gmax: i32,
) {
    if height == 1 {
        for c in 0..width {
            dst[c] = ((top[c] * 23 + bottom[c] * 22 + 16) >> 5).clamp(gmin, gmax);
        }
    } else if height == 2 {
        for c in 0..width {
            dst[c] = ((27 * top[c] + 17 * bottom[c] + 16) >> 5).clamp(gmin, gmax);
            dst[dst_stride + c] =
                ((17 * top[top_stride + c] + 27 * bottom[bottom_stride + c] + 16) >> 5)
                    .clamp(gmin, gmax);
        }
    }
}

fn copy_area(
    src: &[i32], src_off: usize, src_stride: usize, dst: &mut [i32], dst_off: usize,
    dst_stride: usize, width: usize, height: usize,
) {
    for r in 0..height {
        for c in 0..width {
            dst[dst_off + r * dst_stride + c] = src[src_off + r * src_stride + c];
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn add_noise_to_block(
    p: &FilmGrainData, lut_y: &[i32; 256], lut_cb: &[i32; 256], lut_cr: &[i32; 256],
    luma: &mut [i32], loff: usize, cb: &mut [i32], cboff: usize, cr: &mut [i32], croff: usize,
    luma_stride: usize, chroma_stride: usize,
    lgrain: &[i32], lgoff: usize, cbgrain: &[i32], cbgoff: usize, crgrain: &[i32], crgoff: usize,
    lg_stride: usize, cg_stride: usize, half_h: usize, half_w: usize, bit_depth: i32,
    ssy: usize, ssx: usize,
) {
    let (mut cb_mult, mut cb_luma_mult, mut cb_offset) =
        (p.uv_mult[0] - 128, p.uv_luma_mult[0] - 128, p.uv_offset[0] - 256);
    let (mut cr_mult, mut cr_luma_mult, mut cr_offset) =
        (p.uv_mult[1] - 128, p.uv_luma_mult[1] - 128, p.uv_offset[1] - 256);
    let rounding = 1 << (p.scaling_shift - 1);
    let apply_y = p.num_points[0] > 0;
    let apply_cb = p.num_points[1] > 0 || p.chroma_scaling_from_luma;
    let apply_cr = p.num_points[2] > 0 || p.chroma_scaling_from_luma;
    if p.chroma_scaling_from_luma {
        cb_mult = 0;
        cb_luma_mult = 64;
        cb_offset = 0;
        cr_mult = 0;
        cr_luma_mult = 64;
        cr_offset = 0;
    }
    let (min_luma, max_luma, min_chroma, max_chroma) = if p.clip_to_restricted_range {
        if p.mc_identity {
            (16, 235, 16, 235)
        } else {
            (16, 235, 16, 240)
        }
    } else {
        (0, 255, 0, 255)
    };
    for i in 0..(half_h << (1 - ssy)) {
        for j in 0..(half_w << (1 - ssx)) {
            let average_luma = if ssx == 1 {
                (luma[loff + (i << ssy) * luma_stride + (j << ssx)]
                    + luma[loff + (i << ssy) * luma_stride + (j << ssx) + 1]
                    + 1)
                    >> 1
            } else {
                luma[loff + (i << ssy) * luma_stride + j]
            };
            if apply_cb {
                let cbv = cb[cboff + i * chroma_stride + j];
                let idx = (((average_luma * cb_luma_mult + cb_mult * cbv) >> 6) + cb_offset)
                    .clamp(0, (256 << (bit_depth - 8)) - 1);
                cb[cboff + i * chroma_stride + j] = (cbv
                    + ((scale_lut(lut_cb, idx, 8) * cbgrain[cbgoff + i * cg_stride + j] + rounding)
                        >> p.scaling_shift))
                    .clamp(min_chroma, max_chroma);
            }
            if apply_cr {
                let crv = cr[croff + i * chroma_stride + j];
                let idx = (((average_luma * cr_luma_mult + cr_mult * crv) >> 6) + cr_offset)
                    .clamp(0, (256 << (bit_depth - 8)) - 1);
                cr[croff + i * chroma_stride + j] = (crv
                    + ((scale_lut(lut_cr, idx, 8) * crgrain[crgoff + i * cg_stride + j] + rounding)
                        >> p.scaling_shift))
                    .clamp(min_chroma, max_chroma);
            }
        }
    }
    if apply_y {
        for i in 0..(half_h << 1) {
            for j in 0..(half_w << 1) {
                let yv = luma[loff + i * luma_stride + j];
                luma[loff + i * luma_stride + j] = (yv
                    + ((scale_lut(lut_y, yv, 8) * lgrain[lgoff + i * lg_stride + j] + rounding)
                        >> p.scaling_shift))
                    .clamp(min_luma, max_luma);
            }
        }
    }
}

/// avm `av2_add_film_grain_run`, 8-bit path over the decoder's i32 planes (values 0..255).
/// `width`/`height` are the (even-extended) frame dims; planes must already be even-sized
/// (432×240 is). ssx/ssy = chroma subsampling.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn add_film_grain(
    p: &FilmGrainData, seed: u16, luma: &mut [i32], cb: &mut [i32], cr: &mut [i32],
    width: usize, height: usize, luma_stride: usize, chroma_stride: usize, ssy: usize, ssx: usize,
) {
    let bit_depth = 8i32;
    let mut rng = GrainRng { reg: seed };

    let left_pad = 3usize;
    let right_pad = 3usize;
    let top_pad = 3usize;
    let bottom_pad = 0usize;
    let ar_padding = 3usize;

    let luma_subblock_y = 16usize << p.block_size;
    let luma_subblock_x = 16usize << p.block_size;
    let chroma_subblock_y = luma_subblock_y >> ssy;
    let chroma_subblock_x = luma_subblock_x >> ssx;
    let max_chroma_sub_y = MAX_LUMA_SUBBLOCK >> ssy;
    let max_chroma_sub_x = MAX_LUMA_SUBBLOCK >> ssx;

    let luma_block_size_y = top_pad + 2 * ar_padding + MAX_LUMA_SUBBLOCK * 2 + bottom_pad;
    let luma_block_size_x =
        left_pad + 2 * ar_padding + MAX_LUMA_SUBBLOCK * 2 + 2 * ar_padding + right_pad;
    let chroma_block_size_y = top_pad + (2 >> ssy) * ar_padding + max_chroma_sub_y * 2 + bottom_pad;
    let chroma_block_size_x = left_pad
        + (2 >> ssx) * ar_padding
        + max_chroma_sub_x * 2
        + (2 >> ssx) * ar_padding
        + right_pad;
    let lg_stride = luma_block_size_x;
    let cg_stride = chroma_block_size_x;

    let overlap = p.overlap_flag;
    let grain_min = -(1 << (bit_depth - 1));
    let grain_max = (1 << (bit_depth - 1)) - 1;

    // pred_pos tables (avm init_arrays)
    let num_pos_luma = (2 * p.ar_coeff_lag * (p.ar_coeff_lag + 1)) as usize;
    let mut pred_pos_luma: Vec<(i32, i32)> = Vec::with_capacity(num_pos_luma);
    for row in -p.ar_coeff_lag..0 {
        for col in -p.ar_coeff_lag..p.ar_coeff_lag + 1 {
            pred_pos_luma.push((row, col));
        }
    }
    for col in -p.ar_coeff_lag..0 {
        pred_pos_luma.push((0, col));
    }
    // chroma adds the co-located-luma tap when luma has points
    let chroma_has_luma_tap = p.num_points[0] > 0;

    // grain blocks
    let mut lgrain = vec![0i32; luma_block_size_y * lg_stride];
    let mut cbgrain = vec![0i32; chroma_block_size_y * cg_stride];
    let mut crgrain = vec![0i32; chroma_block_size_y * cg_stride];
    let gauss_sec_shift = 12 - bit_depth + p.grain_scale_shift;

    // luma grain
    if p.num_points[0] != 0 {
        let rounding_offset = 1 << (p.ar_coeff_shift - 1);
        for i in 0..luma_block_size_y {
            for j in 0..luma_block_size_x {
                lgrain[i * lg_stride + j] = (GAUSSIAN_SEQUENCE
                    [rng.get(GAUSS_BITS) as usize]
                    + ((1 << gauss_sec_shift) >> 1))
                    >> gauss_sec_shift;
            }
        }
        for i in top_pad..luma_block_size_y - bottom_pad {
            for j in left_pad..luma_block_size_x - right_pad {
                let mut wsum = 0i32;
                for (pos, &(dy, dx)) in pred_pos_luma.iter().enumerate() {
                    wsum += p.ar_coeffs[0][pos]
                        * lgrain[((i as i32 + dy) as usize) * lg_stride + (j as i32 + dx) as usize];
                }
                lgrain[i * lg_stride + j] = (lgrain[i * lg_stride + j]
                    + ((wsum + rounding_offset) >> p.ar_coeff_shift))
                    .clamp(grain_min, grain_max);
            }
        }
    }

    // chroma grain
    {
        let num_pos_chroma = num_pos_luma + chroma_has_luma_tap as usize;
        let rounding_offset = 1 << (p.ar_coeff_shift - 1);
        if p.num_points[1] != 0 || p.chroma_scaling_from_luma {
            rng.init(7 << 5, seed);
            for i in 0..chroma_block_size_y {
                for j in 0..chroma_block_size_x {
                    cbgrain[i * cg_stride + j] = (GAUSSIAN_SEQUENCE
                        [rng.get(GAUSS_BITS) as usize]
                        + ((1 << gauss_sec_shift) >> 1))
                        >> gauss_sec_shift;
                }
            }
            if std::env::var("AGRAIN").is_ok() {
                crate::dlog!("[MGRAIN] cb PRE row3: {:?} | coeffs: {:?}",
                    &cbgrain[3 * cg_stride + 3..3 * cg_stride + 13], &p.ar_coeffs[1][..13]);
            }
        }
        if p.num_points[2] != 0 || p.chroma_scaling_from_luma {
            rng.init(11 << 5, seed);
            for i in 0..chroma_block_size_y {
                for j in 0..chroma_block_size_x {
                    crgrain[i * cg_stride + j] = (GAUSSIAN_SEQUENCE
                        [rng.get(GAUSS_BITS) as usize]
                        + ((1 << gauss_sec_shift) >> 1))
                        >> gauss_sec_shift;
                }
            }
        }
        for i in top_pad..chroma_block_size_y - bottom_pad {
            for j in left_pad..chroma_block_size_x - right_pad {
                let mut wsum_cb = 0i32;
                let mut wsum_cr = 0i32;
                for pos in 0..num_pos_chroma {
                    if pos < num_pos_luma {
                        let (dy, dx) = pred_pos_luma[pos];
                        let idx =
                            ((i as i32 + dy) as usize) * cg_stride + (j as i32 + dx) as usize;
                        wsum_cb += p.ar_coeffs[1][pos] * cbgrain[idx];
                        wsum_cr += p.ar_coeffs[2][pos] * crgrain[idx];
                    } else {
                        // co-located averaged luma tap
                        let ly = ((i - top_pad) << ssy) + top_pad;
                        let lx = ((j - left_pad) << ssx) + left_pad;
                        let mut av_luma = 0i32;
                        for k in ly..ly + ssy + 1 {
                            for l in lx..lx + ssx + 1 {
                                av_luma += lgrain[k * lg_stride + l];
                            }
                        }
                        av_luma = (av_luma + ((1 << (ssy + ssx)) >> 1)) >> (ssy + ssx);
                        wsum_cb += p.ar_coeffs[1][pos] * av_luma;
                        wsum_cr += p.ar_coeffs[2][pos] * av_luma;
                    }
                }
                if p.num_points[1] != 0 || p.chroma_scaling_from_luma {
                    cbgrain[i * cg_stride + j] = (cbgrain[i * cg_stride + j]
                        + ((wsum_cb + rounding_offset) >> p.ar_coeff_shift))
                        .clamp(grain_min, grain_max);
                }
                if p.num_points[2] != 0 || p.chroma_scaling_from_luma {
                    crgrain[i * cg_stride + j] = (crgrain[i * cg_stride + j]
                        + ((wsum_cr + rounding_offset) >> p.ar_coeff_shift))
                        .clamp(grain_min, grain_max);
                }
            }
        }
    }

    if std::env::var("AGRAIN").is_ok() {
        crate::dlog!("[MGRAIN] seed={seed} dims y={luma_block_size_x}x{luma_block_size_y} c={chroma_block_size_x}x{chroma_block_size_y} lgs={lg_stride} cgs={cg_stride}");
        crate::dlog!("[MGRAIN] lg row3: {:?}", &lgrain[3 * lg_stride + 3..3 * lg_stride + 13]);
        crate::dlog!("[MGRAIN] cb row3: {:?}", &cbgrain[3 * cg_stride + 3..3 * cg_stride + 13]);
        crate::dlog!("[MGRAIN] cr row3: {:?}", &crgrain[3 * cg_stride + 3..3 * cg_stride + 13]);
    }

    // scaling LUTs
    let mut lut_y = [0i32; 256];
    let mut lut_cb = [0i32; 256];
    let mut lut_cr = [0i32; 256];
    init_scaling_function(&p.points[0], p.num_points[0], &mut lut_y);
    if p.chroma_scaling_from_luma {
        lut_cb = lut_y;
        lut_cr = lut_y;
    } else {
        init_scaling_function(&p.points[1], p.num_points[1], &mut lut_cb);
        init_scaling_function(&p.points[2], p.num_points[2], &mut lut_cr);
    }

    // line/col overlap buffers
    let mut y_line = vec![0i32; luma_stride * 2];
    let mut cb_line = vec![0i32; chroma_stride * (2 >> ssy)];
    let mut cr_line = vec![0i32; chroma_stride * (2 >> ssy)];
    let mut y_col = vec![0i32; (luma_subblock_y + 2) * 2];
    let mut cb_col = vec![0i32; (chroma_subblock_y + (2 >> ssy)) * (2 >> ssx)];
    let mut cr_col = vec![0i32; (chroma_subblock_y + (2 >> ssy)) * (2 >> ssx)];

    let mut y = 0usize;
    while y < height / 2 {
        rng.init((y as i32) << 2, seed);
        let mut x = 0usize;
        while x < width / 2 {
            let offset_y = (rng.get(9) * (3 - p.block_size)) >> 6;
            rng.get(16);
            rng.get(16);
            rng.get(16);
            let offset_x = (rng.get(9) * (3 - p.block_size)) >> 6;
            rng.get(16);
            rng.get(16);
            rng.get(16);

            // (avm swaps the pad names here; both pads are 3 so the arithmetic is identical)
            let luma_offset_y = (left_pad + 2 * ar_padding) as i32 + (offset_y << 1);
            let luma_offset_x = (top_pad + 2 * ar_padding) as i32 + (offset_x << 1);
            let chroma_offset_y =
                (top_pad + (2 >> ssy) * ar_padding) as i32 + offset_y * (2 >> ssy) as i32;
            let chroma_offset_x =
                (left_pad + (2 >> ssx) * ar_padding) as i32 + offset_x * (2 >> ssx) as i32;
            let (luma_offset_y, luma_offset_x) = (luma_offset_y as usize, luma_offset_x as usize);
            let (chroma_offset_y, chroma_offset_x) =
                (chroma_offset_y as usize, chroma_offset_x as usize);

            if overlap && x != 0 {
                let ycb = y_col.clone();
                ver_boundary_overlap(
                    &ycb, 2, &lgrain[luma_offset_y * lg_stride + luma_offset_x..], lg_stride,
                    &mut y_col, 2, 2,
                    (luma_subblock_y + 2).min(height - (y << 1)), grain_min, grain_max,
                );
                let cbc = cb_col.clone();
                ver_boundary_overlap(
                    &cbc, 2 >> ssx,
                    &cbgrain[chroma_offset_y * cg_stride + chroma_offset_x..], cg_stride,
                    &mut cb_col, 2 >> ssx, 2 >> ssx,
                    (chroma_subblock_y + (2 >> ssy)).min((height - (y << 1)) >> ssy),
                    grain_min, grain_max,
                );
                let crc = cr_col.clone();
                ver_boundary_overlap(
                    &crc, 2 >> ssx,
                    &crgrain[chroma_offset_y * cg_stride + chroma_offset_x..], cg_stride,
                    &mut cr_col, 2 >> ssx, 2 >> ssx,
                    (chroma_subblock_y + (2 >> ssy)).min((height - (y << 1)) >> ssy),
                    grain_min, grain_max,
                );
                let i = if y != 0 { 1usize } else { 0 };
                let (lg2, cb2, cr2) = (y_col.clone(), cb_col.clone(), cr_col.clone());
                add_noise_to_block(
                    p, &lut_y, &lut_cb, &lut_cr,
                    luma, ((y + i) << 1) * luma_stride + (x << 1),
                    cb, ((y + i) << (1 - ssy)) * chroma_stride + (x << (1 - ssx)),
                    cr, ((y + i) << (1 - ssy)) * chroma_stride + (x << (1 - ssx)),
                    luma_stride, chroma_stride,
                    &lg2, i * 4,
                    &cb2, i * (2 - ssy) * (2 - ssx),
                    &cr2, i * (2 - ssy) * (2 - ssx),
                    2, 2 - ssx,
                    ((luma_subblock_y >> 1).min(height / 2 - y)) - i, 1, bit_depth, ssy, ssx,
                );
            }

            if overlap && y != 0 {
                if x != 0 {
                    let yl = y_line.clone();
                    hor_boundary_overlap(
                        &yl[(x << 1)..], luma_stride, &y_col, 2,
                        &mut y_line[(x << 1)..], luma_stride, 2, 2, grain_min, grain_max,
                    );
                    let cbl = cb_line.clone();
                    hor_boundary_overlap(
                        &cbl[x * (2 >> ssx)..], chroma_stride, &cb_col, 2 >> ssx,
                        &mut cb_line[x * (2 >> ssx)..], chroma_stride,
                        2 >> ssx, 2 >> ssy, grain_min, grain_max,
                    );
                    let crl = cr_line.clone();
                    hor_boundary_overlap(
                        &crl[x * (2 >> ssx)..], chroma_stride, &cr_col, 2 >> ssx,
                        &mut cr_line[x * (2 >> ssx)..], chroma_stride,
                        2 >> ssx, 2 >> ssy, grain_min, grain_max,
                    );
                }
                let xoff = if x != 0 { x + 1 } else { 0 };
                let jx = if x != 0 { 1usize } else { 0 };
                let yl = y_line.clone();
                hor_boundary_overlap(
                    &yl[(xoff << 1)..], luma_stride,
                    &lgrain[luma_offset_y * lg_stride + luma_offset_x + jx * 2..],
                    lg_stride,
                    &mut y_line[(xoff << 1)..], luma_stride,
                    (luma_subblock_x - jx * 2).min(width - (xoff << 1)),
                    2, grain_min, grain_max,
                );
                let cbl = cb_line.clone();
                hor_boundary_overlap(
                    &cbl[xoff << (1 - ssx)..], chroma_stride,
                    &cbgrain[chroma_offset_y * cg_stride + chroma_offset_x + (jx << (1 - ssx))..],
                    cg_stride,
                    &mut cb_line[xoff << (1 - ssx)..], chroma_stride,
                    (chroma_subblock_x - (jx << (1 - ssx))).min((width - (xoff << 1)) >> ssx),
                    2 >> ssy, grain_min, grain_max,
                );
                let crl = cr_line.clone();
                hor_boundary_overlap(
                    &crl[xoff << (1 - ssx)..], chroma_stride,
                    &crgrain[chroma_offset_y * cg_stride + chroma_offset_x + (jx << (1 - ssx))..],
                    cg_stride,
                    &mut cr_line[xoff << (1 - ssx)..], chroma_stride,
                    (chroma_subblock_x - (jx << (1 - ssx))).min((width - (xoff << 1)) >> ssx),
                    2 >> ssy, grain_min, grain_max,
                );
                let (yl2, cbl2, crl2) = (y_line.clone(), cb_line.clone(), cr_line.clone());
                add_noise_to_block(
                    p, &lut_y, &lut_cb, &lut_cr,
                    luma, (y << 1) * luma_stride + (x << 1),
                    cb, (y << (1 - ssy)) * chroma_stride + (x << (1 - ssx)),
                    cr, (y << (1 - ssy)) * chroma_stride + (x << (1 - ssx)),
                    luma_stride, chroma_stride,
                    &yl2, x << 1,
                    &cbl2, x << (1 - ssx),
                    &crl2, x << (1 - ssx),
                    luma_stride, chroma_stride,
                    1, (luma_subblock_x >> 1).min(width / 2 - x), bit_depth, ssy, ssx,
                );
            }

            let i = if overlap && y != 0 { 1usize } else { 0 };
            let j = if overlap && x != 0 { 1usize } else { 0 };
            add_noise_to_block(
                p, &lut_y, &lut_cb, &lut_cr,
                luma, ((y + i) << 1) * luma_stride + ((x + j) << 1),
                cb, ((y + i) << (1 - ssy)) * chroma_stride + ((x + j) << (1 - ssx)),
                cr, ((y + i) << (1 - ssy)) * chroma_stride + ((x + j) << (1 - ssx)),
                luma_stride, chroma_stride,
                &lgrain, (luma_offset_y + (i << 1)) * lg_stride + luma_offset_x + (j << 1),
                &cbgrain,
                (chroma_offset_y + (i << (1 - ssy))) * cg_stride + chroma_offset_x + (j << (1 - ssx)),
                &crgrain,
                (chroma_offset_y + (i << (1 - ssy))) * cg_stride + chroma_offset_x + (j << (1 - ssx)),
                lg_stride, cg_stride,
                ((luma_subblock_y >> 1).min(height / 2 - y)) - i,
                ((luma_subblock_x >> 1).min(width / 2 - x)) - j,
                bit_depth, ssy, ssx,
            );

            if overlap {
                if x != 0 {
                    copy_area(
                        &y_col.clone(), luma_subblock_y << 1, 2, &mut y_line, x << 1,
                        luma_stride, 2, 2,
                    );
                    copy_area(
                        &cb_col.clone(), chroma_subblock_y << (1 - ssx), 2 >> ssx,
                        &mut cb_line, x << (1 - ssx), chroma_stride, 2 >> ssx, 2 >> ssy,
                    );
                    copy_area(
                        &cr_col.clone(), chroma_subblock_y << (1 - ssx), 2 >> ssx,
                        &mut cr_line, x << (1 - ssx), chroma_stride, 2 >> ssx, 2 >> ssy,
                    );
                }
                let xoff = if x != 0 { x + 1 } else { 0 };
                copy_area(
                    &lgrain,
                    (luma_offset_y + luma_subblock_y) * lg_stride + luma_offset_x
                        + if x != 0 { 2 } else { 0 },
                    lg_stride,
                    &mut y_line, xoff << 1, luma_stride,
                    luma_subblock_x.min(width - (x << 1)) - if x != 0 { 2 } else { 0 }, 2,
                );
                copy_area(
                    &cbgrain,
                    (chroma_offset_y + chroma_subblock_y) * cg_stride + chroma_offset_x
                        + if x != 0 { 2 >> ssx } else { 0 },
                    cg_stride,
                    &mut cb_line, xoff << (1 - ssx), chroma_stride,
                    chroma_subblock_x.min((width - (x << 1)) >> ssx)
                        - if x != 0 { 2 >> ssx } else { 0 },
                    2 >> ssy,
                );
                copy_area(
                    &crgrain,
                    (chroma_offset_y + chroma_subblock_y) * cg_stride + chroma_offset_x
                        + if x != 0 { 2 >> ssx } else { 0 },
                    cg_stride,
                    &mut cr_line, xoff << (1 - ssx), chroma_stride,
                    chroma_subblock_x.min((width - (x << 1)) >> ssx)
                        - if x != 0 { 2 >> ssx } else { 0 },
                    2 >> ssy,
                );
                copy_area(
                    &lgrain, luma_offset_y * lg_stride + luma_offset_x + luma_subblock_x,
                    lg_stride, &mut y_col, 0, 2, 2,
                    (luma_subblock_y + 2).min(height - (y << 1)),
                );
                copy_area(
                    &cbgrain, chroma_offset_y * cg_stride + chroma_offset_x + chroma_subblock_x,
                    cg_stride, &mut cb_col, 0, 2 >> ssx, 2 >> ssx,
                    (chroma_subblock_y + (2 >> ssy)).min((height - (y << 1)) >> ssy),
                );
                copy_area(
                    &crgrain, chroma_offset_y * cg_stride + chroma_offset_x + chroma_subblock_x,
                    cg_stride, &mut cr_col, 0, 2 >> ssx, 2 >> ssx,
                    (chroma_subblock_y + (2 >> ssy)).min((height - (y << 1)) >> ssy),
                );
            }

            x += luma_subblock_x >> 1;
        }
        y += luma_subblock_y >> 1;
    }
}

/// Resolve (id, seed) via FGM_TABLE and apply grain to the three planes in place.
pub fn apply_grain_to_planes(planes: &mut [crate::av2_frame::Plane; 3], id: u8, seed: u16) {
    crate::prof_scope!(10);
    let params = FGM_TABLE.with(|t| t.borrow()[id as usize].clone());
    let Some(params) = params else { return };
    let (w, h) = (planes[0].w, planes[0].h);
    if w == 0 || w % 2 != 0 || h % 2 != 0 {
        return; // odd dims need the even-extension path; none minted yet
    }
    let (ssx, ssy) = {
        let s = crate::av2_frame::SS.with(|c| c.get());
        (s.0 as usize, s.1 as usize)
    };
    let ls = planes[0].stride;
    let cs = planes[1].stride.max(1);
    // Split-borrow the three planes.
    let [p0, p1, p2] = planes;
    let (cw, ch) = (p1.w, p1.h);
    let _ = (cw, ch);
    // Grain synthesis works in i32 (it adds signed noise before the clamp); widen the
    // u16 planes around the call. One copy each way per grain frame — grain streams
    // only, and the synthesis itself dominates.
    let mut y32: Vec<i32> = p0.px.iter().map(|&v| v as i32).collect();
    let mut u32b: Vec<i32> = p1.px.iter().map(|&v| v as i32).collect();
    let mut v32: Vec<i32> = p2.px.iter().map(|&v| v as i32).collect();
    add_film_grain(
        &params, seed, &mut y32, &mut u32b, &mut v32, w, h, ls, cs, ssy, ssx,
    );
    for (d, &v) in p0.px.iter_mut().zip(&y32) { *d = v as u16; }
    for (d, &v) in p1.px.iter_mut().zip(&u32b) { *d = v as u16; }
    for (d, &v) in p2.px.iter_mut().zip(&v32) { *d = v as u16; }
}
