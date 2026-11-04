//! AV2 secondary-transform (STX/IST) `[set * 3 + type]` -> write-out mapping index, from dav2d
//! stx_tables.c. The kernels are `STX_{4X4,8X8}_KERNEL` in `av2_tables_gen`.

pub const COEFF8X8_MAPPING_IDX: [u8; 33] = [0,1,1,2,2,1,2,2,1,0,2,1,2,2,2,0,0,0,1,1,1,0,1,1,1,1,1,2,2,2,1,1,1];
