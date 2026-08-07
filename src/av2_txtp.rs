//! AV2 transform-type (txtp) encoding — dav2d `levels.h`. A txtp packs the
//! horizontal (row) 1-D type, the tx class, and the vertical (col) 1-D type:
//! `txtp = hor_1d | (class << 3) | (ver_1d << 5)`. The `(VER)_(HOR)` naming means
//! e.g. `ADST_DCT` = vertical ADST, horizontal DCT. Includes the UV-mode → txtp
//! inference LUT (`dav2d_txtp_from_uvmode`) used by the chroma coeff decode.

// 1-D transform types (Txfm1dType) — match `av2_itx::tx1d_fn` indices.
pub const DCT: u32 = 0;
pub const ADST: u32 = 1;
pub const FLIPADST: u32 = 2;
pub const IDENTITY: u32 = 3;
pub const DDT: u32 = 4;
pub const FLIPDDT: u32 = 5;

// TxClass.
pub const TX_CLASS_2D: u32 = 0;
pub const TX_CLASS_2D_INV: u32 = 1;
pub const TX_CLASS_H: u32 = 2;
pub const TX_CLASS_V: u32 = 3;

/// Encode a txtp from horizontal/vertical 1-D types + class (dav2d `TX_TYPE_ENUM`).
pub const fn txtp(hor_1d: u32, ver_1d: u32, class: u32) -> u32 {
    hor_1d | (class << 3) | (ver_1d << 5)
}

// Common 2-D txtps (values: DCT_DCT=0, DCT_ADST=1, ADST_DCT=32, ADST_ADST=33).
pub const DCT_DCT: u32 = txtp(DCT, DCT, TX_CLASS_2D);
pub const DCT_ADST: u32 = txtp(ADST, DCT, TX_CLASS_2D);
pub const ADST_DCT: u32 = txtp(DCT, ADST, TX_CLASS_2D);
pub const ADST_ADST: u32 = txtp(ADST, ADST, TX_CLASS_2D);
pub const IDTX: u32 = txtp(IDENTITY, IDENTITY, TX_CLASS_2D);
pub const V_DCT: u32 = txtp(IDENTITY, DCT, TX_CLASS_V);
pub const H_DCT: u32 = txtp(DCT, IDENTITY, TX_CLASS_H);

/// Row (horizontal) 1-D type of a txtp — selects the row transform.
pub fn txtp_row(t: u32) -> usize {
    (t & 7) as usize
}
/// Column (vertical) 1-D type of a txtp — selects the column transform.
pub fn txtp_col(t: u32) -> usize {
    ((t >> 5) & 7) as usize
}
/// Tx class of a txtp (2D / 2D_INV / H / V) — drives the coeff-scan context.
pub fn txtp_class(t: u32) -> u32 {
    (t >> 3) & 3
}

/// UV intra mode → inferred txtp (dav2d `dav2d_txtp_from_uvmode`), in UV-mode
/// order: DC, VERT, HOR, D45, D135, VERT_RIGHT, HOR_DOWN, HOR_UP, VERT_LEFT,
/// SMOOTH, SMOOTH_V, SMOOTH_H, PAETH.
pub static TXTP_FROM_UVMODE: [u32; 13] = [
    DCT_DCT,   // DC_PRED
    ADST_DCT,  // VERT_PRED
    DCT_ADST,  // HOR_PRED
    DCT_DCT,   // DIAG_DOWN_LEFT_PRED
    ADST_ADST, // DIAG_DOWN_RIGHT_PRED
    ADST_DCT,  // VERT_RIGHT_PRED
    DCT_ADST,  // HOR_DOWN_PRED
    DCT_ADST,  // HOR_UP_PRED
    ADST_DCT,  // VERT_LEFT_PRED
    ADST_ADST, // SMOOTH_PRED
    ADST_DCT,  // SMOOTH_V_PRED
    DCT_ADST,  // SMOOTH_H_PRED
    ADST_ADST, // PAETH_PRED
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn txtp_values() {
        assert_eq!(DCT_DCT, 0);
        assert_eq!(DCT_ADST, 1);
        assert_eq!(ADST_DCT, 32);
        assert_eq!(ADST_ADST, 33);
    }

    #[test]
    fn txtp_unpacks_to_1d_types() {
        // ADST_DCT = vertical ADST, horizontal DCT → row=DCT, col=ADST, class 2D
        assert_eq!(txtp_row(ADST_DCT), DCT as usize);
        assert_eq!(txtp_col(ADST_DCT), ADST as usize);
        assert_eq!(txtp_class(ADST_DCT), TX_CLASS_2D);
        // V_DCT carries the V class
        assert_eq!(txtp_class(V_DCT), TX_CLASS_V);
        assert_eq!(txtp_row(V_DCT), IDENTITY as usize);
        assert_eq!(txtp_col(V_DCT), DCT as usize);
    }

    #[test]
    fn uvmode_lut() {
        assert_eq!(TXTP_FROM_UVMODE[0], DCT_DCT); // DC_PRED
        assert_eq!(TXTP_FROM_UVMODE[1], ADST_DCT); // VERT_PRED
        assert_eq!(TXTP_FROM_UVMODE[2], DCT_ADST); // HOR_PRED
        assert_eq!(TXTP_FROM_UVMODE[12], ADST_ADST); // PAETH_PRED
        assert_eq!(TXTP_FROM_UVMODE.len(), 13);
    }

    #[test]
    fn txtp_drives_transform_dispatch() {
        // Unpacking a txtp into (row, col) types must drive the same 2-D inverse
        // transform as calling with those types explicitly.
        let t = ADST_DCT; // → row DCT, col ADST
        let mut coeff = vec![0i32; 64];
        coeff[0] = 100;
        coeff[5] = 20;
        let mut via_txtp = vec![0i32; 64];
        crate::av2_itx::inv_txfm_2d(&coeff, 1, 1, txtp_row(t), txtp_col(t), &mut via_txtp);
        let mut direct = vec![0i32; 64];
        crate::av2_itx::inv_txfm_2d(&coeff, 1, 1, DCT as usize, ADST as usize, &mut direct);
        assert_eq!(via_txtp, direct);
    }
}
