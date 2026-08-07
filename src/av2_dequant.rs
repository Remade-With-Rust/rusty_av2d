//! AV2 dequantizer — dav2d `quantizer.c` (`dav2d_dq_lookup`) + the coefficient
//! dequant in `recon_tmpl.c`. Bridges decoded coefficient tokens to transform
//! input: `cf = (tok · dq + 4) >> dq_shift`, with the quant step `dq` from a
//! per-q-index lookup (optionally QM-modulated) and `dq_shift` from the TX size.
//!
//! AV2 replaced AV1's dc/ac quant tables with a compact 24-entry mantissa scaled
//! by `1 << (qidx/24)`.

/// 24-entry quant mantissa (dav2d `dq_lookup_tbl`).
const DQ_LOOKUP_TBL: [u8; 24] = [
    40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, //
    57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78,
];

/// Quant step for a q-index (dav2d `dav2d_dq_lookup`): exponential lookup,
/// mantissa `<< (qidx/24)`. `qidx == 0` is the lossless step (64).
pub fn dq_lookup(qidx: u32) -> u32 {
    if qidx == 0 {
        return 64;
    }
    let q = qidx - 1;
    let shift = q / 24;
    (DQ_LOOKUP_TBL[(q % 24) as usize] as u32) << shift
}

/// Dequant right-shift (dav2d `recon_tmpl.c`): grows with the TX size. `tx_ctx` is
/// the TX-size context (0=4×4, 1=8×8, 2=16×16, 3=32×32, 4=64×64); `tcq_enabled`
/// adds one for trellis-coded quantization.
pub fn dq_shift(tcq_enabled: u32, tx_ctx: i32) -> u32 {
    (tcq_enabled as i32 + 3 + 0.max(tx_ctx - 2)) as u32
}

/// Maximum dequantized coefficient magnitude, `(1 << (bpc + 7)) - 1` — equal to
/// dav2d's `~(~127U << bpc)`. (8-bit → 32767.)
pub fn cf_max(bpc: u32) -> i32 {
    (1i32 << (bpc + 7)) - 1
}

/// QM (quantization-matrix) modulation of the quant step (dav2d
/// `dc_dq = (dc_dq * qm + 16) >> 5`). `qm == 32` is the identity (flat matrix).
pub fn apply_qm(dq: u32, qm: u32) -> u32 {
    (dq * qm + 16) >> 5
}

/// Dequantize one coefficient token (dav2d `recon_tmpl.c`). `large` selects the
/// 24-bit-masked + clamped path for tokens that carried a high-range extension;
/// the common small-token path is a plain `(tok·dq + 4) >> dq_shift`. Returns the
/// unsigned magnitude (the caller applies the decoded sign).
pub fn dequant_coeff(tok: u32, dq: u32, dq_shift: u32, cf_max: i32, sign: u32, large: bool) -> u32 {
    if large {
        // The 24-bit mask must be applied to the FULL product, so widen to u64 first —
        // a large token (up to ~20 bits) times a shifted dq overflows u32 before masking.
        let val = (((tok as u64 * dq as u64) & 0xffffff) as u32 + 4) >> dq_shift;
        val.min(cf_max as u32 + sign)
    } else {
        (tok * dq + 4) >> dq_shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dq_lookup_known_values() {
        assert_eq!(dq_lookup(0), 64); // lossless
        assert_eq!(dq_lookup(1), 40); // q=0, shift 0, tbl[0]
        assert_eq!(dq_lookup(24), 78); // q=23, shift 0, tbl[23]
        assert_eq!(dq_lookup(25), 80); // q=24, shift 1, tbl[0]<<1
        assert_eq!(dq_lookup(49), 160); // q=48, shift 2, tbl[0]<<2
        // monotonic non-decreasing across the range.
        let mut prev = 0;
        for qidx in 1..256 {
            if !crate::av2_recon::work_tick("av2_dequant:73") { break; }
            let v = dq_lookup(qidx);
            assert!(v >= prev, "non-monotonic at {qidx}");
            prev = v;
        }
    }

    #[test]
    fn dq_shift_grows_with_tx_size() {
        assert_eq!(dq_shift(0, 0), 3); // 4x4
        assert_eq!(dq_shift(0, 1), 3); // 8x8
        assert_eq!(dq_shift(0, 2), 3); // 16x16
        assert_eq!(dq_shift(0, 3), 4); // 32x32
        assert_eq!(dq_shift(0, 4), 5); // 64x64
        assert_eq!(dq_shift(1, 3), 5); // 32x32 + tcq
    }

    #[test]
    fn cf_max_per_depth() {
        assert_eq!(cf_max(8), 32767);
        assert_eq!(cf_max(10), 131071);
        assert_eq!(cf_max(12), 524287);
    }

    #[test]
    fn apply_qm_identity_and_half() {
        assert_eq!(apply_qm(80, 32), 80); // qm=32 → identity
        assert_eq!(apply_qm(80, 16), 40); // qm=16 → half step
    }

    #[test]
    fn dequant_small_and_large() {
        // small: (tok*dq + 4) >> shift
        assert_eq!(dequant_coeff(5, 40, 3, 32767, 0, false), (5 * 40 + 4) >> 3);
        assert_eq!(dequant_coeff(5, 40, 3, 32767, 0, false), 25);
        // large: 24-bit masked, clamped to cf_max + sign
        assert_eq!(dequant_coeff(10, 40, 3, 32767, 0, true), ((400 & 0xffffff) + 4) >> 3);
        // saturation
        let big = dequant_coeff(0xfffff, 78 << 8, 3, 32767, 1, true);
        assert_eq!(big, 32767 + 1);
    }
}
