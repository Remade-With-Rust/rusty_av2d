//! AV2 quantization matrices (QM): predefined-matrix lookup + the per-position dequant
//! weighting. Ports of avm quant_common.c (av2_qm_init layout / av2_iqmatrix /
//! av2_get_adjusted_tx_size) and decodetxb.c get_dqv:
//!   dqv' = (iqm[pos] * dqv + (1 << (AVM_QM_BITS-1))) >> AVM_QM_BITS,  AVM_QM_BITS = 5.
//! QM applies to 2D transforms only (avm av2_get_iqmatrix: 1D/identity transforms use the
//! flat matrix = no weighting). 64-dim transforms reuse the 32-dim matrix
//! (av2_get_adjusted_tx_size) — their coded coefficient grid is the 32-clamped corner.
//! PANIC-FREEDOM: this module is compiled under `#![deny(clippy::indexing_slicing)]` (see the
//! attribute below) — every element access is a checked accessor, so no bounds panic is
//! reachable here BY CONSTRUCTION rather than by review or fuzzing. This is the pattern the
//! rest of the decode path should adopt module by module.
//! User-defined matrices (QM OBU payload) are NOT ported (loud error at the parse; every
//! avmenc mint so far signals no QM OBU and the decoder-default predefined list is used).

#![deny(clippy::indexing_slicing)]

use crate::av2_qm_tables::IWT_MATRIX;

/// (level, plane) -> slice offset walk, mirroring av2_qm_init: TX_SIZES_ALL order,
/// accumulating only the canonical (t == adjusted_tx_size) sizes.
/// Order: 4x4, 8x8, 16x16, 32x32, 4x8, 8x4, 8x16, 16x8, 16x32, 32x16, 4x16, 16x4,
/// 8x32, 32x8, 4x32, 32x4  (64-dim sizes reuse their 32-clamped partner).
const QM_SIZES: [(usize, usize); 16] = [
    (4, 4), (8, 8), (16, 16), (32, 32), (4, 8), (8, 4), (8, 16), (16, 8),
    (16, 32), (32, 16), (4, 16), (16, 4), (8, 32), (32, 8), (4, 32), (32, 4),
];

/// Offset of the (w, h) canonical matrix inside a (level, plane) run.
fn qm_offset(w: usize, h: usize) -> Option<usize> {
    let mut off = 0usize;
    for &(sw, sh) in &QM_SIZES {
        if sw == w && sh == h {
            return Some(off);
        }
        off += sw * sh;
    }
    None
}

thread_local! {
    /// Frame QM state: (using_qmatrix, qm_y, qm_u, qm_v). Level 15 == flat (no matrix).
    pub static QM_FRAME: std::cell::Cell<(bool, u8, u8, u8)> =
        const { std::cell::Cell::new((false, 15, 15, 15)) };
}

pub fn set_frame_qm(enabled: bool, qy: u8, qu: u8, qv: u8) {
    if std::env::var("MQM").is_ok() {
        crate::dlog!("[MQM] enabled={enabled} y={qy} u={qu} v={qv}");
    }
    QM_FRAME.with(|c| c.set((enabled, qy, qu, qv)));
}

/// The iwt matrix slice for a coded TX block, or None when QM is off / level 15 /
/// the transform is not 2D. `w_px`/`h_px` are the TX dims in pixels (64-dim allowed —
/// the caller indexes positions inside the 32-clamped coded grid, matching avm where a
/// 64-dim TX's coded coefficients live in the adjusted 32-dim corner).
/// Returns (matrix, matrix_width) — index as m[row * mw + col].
pub fn iqm_slice(plane: usize, w_px: usize, h_px: usize, is_2d: bool) -> Option<(&'static [u8], usize)> {
    let (enabled, qy, qu, qv) = QM_FRAME.with(|c| c.get());
    if !enabled || !is_2d {
        return None;
    }
    let level = match plane {
        0 => qy,
        1 => qu,
        _ => qv,
    } as usize;
    if level >= 15 {
        return None; // NUM_QM_LEVELS-1 == flat
    }
    let (w, h) = (w_px.min(32), h_px.min(32)); // av2_get_adjusted_tx_size
    let off = qm_offset(w, h)?;
    let tbl = IWT_MATRIX.get(level)?.get((plane >= 1) as usize)?;
    Some((tbl.get(off..off + w * h)?, w))
}

/// avm decodetxb.c get_dqv: weight a dequant value by the matrix entry at the coef's
/// position. avm's `pos = scan[c]` is ROW-major in the (adjusted) matrix raster; THIS
/// decoder's coefficient arrays are COLUMN-major with stride = coded height
/// (`i = col * hc + row`, hc = min(h,32) — the stx code's `stride = th.min(32)`
/// convention). Map i -> (row * mw + col) for the matrix lookup.
#[inline]
pub fn qm_apply(iqm: Option<(&[u8], usize)>, i: usize, hc: usize, dqv: u32) -> u32 {
    match iqm {
        Some((m, mw)) => {
            let (col, row) = (i / hc, i % hc);
            match m.get(row * mw + col) {
                Some(&w) => ((w as u32) * dqv + 16) >> 5,
                None => dqv,
            }
        }
        None => dqv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_cover_total() {
        let mut off = 0usize;
        for &(w, h) in &QM_SIZES {
            assert_eq!(qm_offset(w, h), Some(off));
            off += w * h;
        }
        assert_eq!(off, crate::av2_qm_tables::QM_TOTAL_SIZE);
    }

    #[test]
    fn adjusted_dims() {
        set_frame_qm(true, 7, 7, 7);
        let (m64, w) = iqm_slice(0, 64, 64, true).unwrap();
        let (m32, w32) = iqm_slice(0, 32, 32, true).unwrap();
        assert_eq!(w, 32);
        assert_eq!(w32, 32);
        assert_eq!(m64.as_ptr(), m32.as_ptr());
        assert!(iqm_slice(0, 8, 8, false).is_none());
        set_frame_qm(true, 15, 15, 15);
        assert!(iqm_slice(0, 8, 8, true).is_none());
        set_frame_qm(false, 7, 7, 7);
        assert!(iqm_slice(0, 8, 8, true).is_none());
    }
}
