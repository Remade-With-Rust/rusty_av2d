# Debug probes — the env-gated instrument surface

Generated from `git grep 'env::var("…")' src/` (regenerate with
`python tools_gen_probes.py` or the grep in wiring.md §4.2). **126 distinct
variables.** Everything routes through `dlog!`, so `RUSTY_AV2D_DEBUG=1` must be
set for any probe to print (the `=path` file-dump probes write regardless).

These probes are the decoder's regression instrument — the bring-up discipline
keeps them until full conformance. The paired avm probes live in the AVM tree
this repo is gated against (committed there; rebuild `avmdec` with `ninja`).

| env var | sites | what it does | avm pair |
|---|---|---|---|
| `AGRAIN` | av2_grain.rs:356, av2_grain.rs:412 |  |  |
| `AIDBG` | av2_recon.rs:1902 | =all or =bx,by: per-leaf intra info (mode idx, mrl, fsc, eob, rng) |  |
| `ATDBG` | av2_frame.rs:37 |  |  |
| `BRDBG` | av2_recon.rs:5398, av2_recon.rs:6471, av2_recon.rs:6759 … |  |  |
| `CC96` | av2_recon.rs:2413 |  |  |
| `CCDBG` | av2_recon.rs:2613 |  |  |
| `CCSODBG` | av2_frame.rs:2027 |  |  |
| `CCSODBG2` | obu.rs:1435 |  |  |
| `CFGDBG` | av2_frame.rs:1929, av2_frame.rs:1971 |  |  |
| `CFLA` | av2_recon.rs:6935, av2_recon.rs:6955, av2_recon.rs:6996 |  |  |
| `CFLT` | av2_recon.rs:7003 |  |  |
| `CMH` | av2_recon.rs:7258, av2_recon.rs:7294, av2_recon.rs:7585 |  |  |
| `CMODE` | av2_recon.rs:7335 |  |  |
| `CQD` | av2_recon.rs:3535 |  |  |
| `CRECONDBG` | av2_frame.rs:1048 |  |  |
| `CREFDBG` | av2_recon.rs:7630 |  |  |
| `CT48` | av2_recon.rs:2407 |  |  |
| `CYAT` | av2_recon.rs:5373, av2_recon.rs:5377 |  |  |
| `DAVCAP` | obu.rs:3056, obu.rs:3060, obu.rs:3306 … | enable dav2d capture-oracle comparisons (with DAVCAP_DIR) | dav2d instrumented build |
| `DAVCAP_DIR` | av2_recon.rs:908 | directory for dav capture files |  |
| `DBG00` | av2_recon.rs:2612, av2_recon.rs:4882, av2_recon.rs:8081 |  |  |
| `DBLK444` | av2_frame.rs:517, av2_frame.rs:1779, av2_recon.rs:5768 | deblock probe window (444 bring-up) | avm DBLKDBG/DBLK444 |
| `DBQ13` | av2_deblock.rs:328 |  |  |
| `DQGRID` | av2_frame.rs:1763 | per-SB delta-q deblock threshold grid |  |
| `DV2` | av2_recon.rs:8083 |  |  |
| `F157DBG` | av2_recon.rs:5784 | SDP chroma-root partition inference (bp, luma dirptr, CfL disallow) |  |
| `F1BLK` | av2_recon.rs:1899 |  |  |
| `F1DUMP` | obu.rs:3685 |  |  |
| `FGMDBG` | obu.rs:4016 |  |  |
| `FILT_ISO` | obu.rs:3337 |  |  |
| `FORCE_PD` | av2_frame.rs:1802 | replace our post-deblock with the dav capture (cascade isolation) |  |
| `GDBG` | av2_gdf.rs:358 |  |  |
| `HANGCP` | av2_recon.rs:2566, av2_recon.rs:5128, av2_recon.rs:5133 … |  |  |
| `IBCDBG` | av2_recon.rs:9530 |  |  |
| `ICDBG` | av2_recon.rs:3358 |  |  |
| `ICHR` | av2_recon.rs:3203 |  |  |
| `ILSP` | av2_recon.rs:4933 |  |  |
| `ISOLATE` | obu.rs:3520 | per-stage isolation mode with dav captures |  |
| `MCDF` | cdf_av2.rs:477, obu.rs:3004 |  |  |
| `MCFL` | av2_recon.rs:2330, av2_recon.rs:2490 |  |  |
| `MDBGRID` | av2_frame.rs:1665 |  |  |
| `MDBW` | av2_frame.rs:1707, av2_frame.rs:1817 | deblock wmap/qmap per-edge instrument | avm DBLKDBG; the deblock divergence killer |
| `MDCP` | av2_recon.rs:6571 |  |  |
| `MDQ` | av2_recon.rs:447, av2_recon.rs:6687 |  |  |
| `MDQDUMP` | av2_recon.rs:6695, av2_recon.rs:6706 |  |  |
| `MDUMP_CDEF` | av2_frame.rs:2037 | =path: dump post-CDEF luma | avm ADUMP_CDEF=path |
| `MDUMP_PD` | av2_frame.rs:1795 | =path: dump post-deblock luma | avm ADUMP_PD=path |
| `MDUMP_RECON` | av2_frame.rs:1702 | =path: dump pre-deblock recon luma (u16le) | avm ADUMP_RECON=path |
| `MFIN` | av2_frame.rs:2084 |  |  |
| `MFMVDBG` | av2_refmvs.rs:1668 |  |  |
| `MHDRQ` | obu.rs:913, obu.rs:971 |  |  |
| `MIBC` | av2_recon.rs:9374, av2_recon.rs:9398, av2_recon.rs:9473 … |  |  |
| `MINTRA` | av2_recon.rs:6561 |  |  |
| `MKEYL` | av2_recon.rs:9366, av2_recon.rs:9811 |  |  |
| `MLEAF` | av2_recon.rs:4203, av2_recon.rs:4206, av2_recon.rs:4216 … |  |  |
| `MLRDS` | av2_lr.rs:479 | =path: dump our ds-luma (chroma LR cross input) in avm layout | avm ALRDS=path (instrumented build) |
| `MLRH` | obu.rs:1238, obu.rs:1267, obu.rs:1282 … | parsed LR frame header: types/ffon/classes/unit sizes + frame filters | avm ALRU -> [ALRH] |
| `MLRPRE` | av2_frame.rs:2052 | =path: dump pre-LR (post-CCSO) luma | avm ALRPRE=path |
| `MLRU` | av2_lr.rs:415, av2_lr.rs:768, av2_lr.rs:772 … | loop-restoration unit reads: [MLRUPRE] entry rng+cdf, [MLRU] result, [MLRF] chroma unit taps, [MLRC] chroma apply params | avm ALRU -> [ALRUPRE]/[ALRU]/[ALRF]/[ALRH] |
| `MMVP` | av2_recon.rs:4879 |  |  |
| `MPAL` | av2_recon.rs:5314, av2_recon.rs:5327, av2_recon.rs:9850 … |  |  |
| `MPART` | av2_decode.rs:235 |  |  |
| `MPARTK` | av2_recon.rs:2114 |  |  |
| `MPB` | av2_recon.rs:4855 |  |  |
| `MPREDRL` | av2_recon.rs:8047, av2_recon.rs:8739, av2_recon.rs:8872 … |  |  |
| `MPREF` | av2_frame.rs:1617 |  |  |
| `MPROBE` | av2_recon.rs:6666 |  |  |
| `MPROBE2` | av2_recon.rs:4019, av2_recon.rs:4044 |  |  |
| `MQM` | av2_qm.rs:47 |  |  |
| `MREFDUMP` | av2_frame.rs:731 |  |  |
| `MRLDBG` | av2_recon.rs:6343, av2_recon.rs:6621 |  |  |
| `MRMVC` | av2_recon.rs:3308, av2_recon.rs:3318 |  |  |
| `MSBT` | obu.rs:3162, obu.rs:3207, obu.rs:3599 … | per-superblock entropy state (rng/cnt) after each SB | avm SBTELL -> [SBTELL]; diff on rng only, cnt is a register-convention delta |
| `MSCORE` | av2_recon.rs:28, av2_recon.rs:198 |  |  |
| `MSCOREF` | av2_recon.rs:29, av2_recon.rs:199 |  |  |
| `MSCREF` | av2_recon.rs:6407 |  |  |
| `MSKM` | av2_recon.rs:8194 |  |  |
| `MSTK2` | av2_recon.rs:4289 |  |  |
| `MTRACE` | av2_recon.rs:1879, av2_recon.rs:2297, av2_recon.rs:9879 |  |  |
| `MTRACE2` | av2_recon.rs:4160 |  |  |
| `MTXB` | av2_recon.rs:2218, av2_recon.rs:2420, av2_recon.rs:2442 … |  |  |
| `MTXP` | av2_recon.rs:268, av2_recon.rs:294 |  |  |
| `MUVCK` | av2_recon.rs:2312, av2_recon.rs:2356, av2_recon.rs:2382 |  |  |
| `MUVM` | av2_recon.rs:7356, av2_recon.rs:7487, av2_recon.rs:7498 … |  |  |
| `MUVP` | av2_recon.rs:7467, av2_recon.rs:7519, av2_recon.rs:7527 |  |  |
| `MWARP2` | av2_warp.rs:128 |  |  |
| `MYCTX` | av2_recon.rs:9730 |  |  |
| `MYSET` | av2_recon.rs:9725 |  |  |
| `NOCCSO` | av2_frame.rs:1980 |  |  |
| `NOCDEF` | av2_frame.rs:1859 |  |  |
| `NODEBLOCK` | av2_frame.rs:1711 | disable deblocking (isolation) |  |
| `NOGDF` | av2_frame.rs:2072 | disable GDF (isolation) |  |
| `NOLR` | av2_frame.rs:2046 | disable loop restoration (isolation) |  |
| `NOSPC` | av2_recon.rs:5210 |  |  |
| `OH1PD` | av2_frame.rs:1830 |  |  |
| `OH1PRE` | obu.rs:3347, obu.rs:3765 |  |  |
| `OHDBG` | obu.rs:476 |  |  |
| `OPFLDBG` | av2_recon.rs:2832, av2_recon.rs:4733 |  |  |
| `P32CDF` | av2_decode.rs:228, av2_decode.rs:329, av2_decode.rs:449 |  |  |
| `PARTIN` | av2_recon.rs:1859, av2_recon.rs:5746 | partition-node entry rng+dif: [PARTIN] luma tree, [PARTINC] SDP chroma tree | avm PARTPROBE -> [PARTIN]/[PARTFD]/[PARTFO]; avm suppresses derived/forced nodes, align as subsequence |
| `PDCDF` | av2_decode.rs:321 |  |  |
| `PREFDBG` | av2_recon.rs:874, obu.rs:837 | prediction reference debug | avm PREFDBG |
| `PREFDUMP` | av2_frame.rs:1640 |  |  |
| `QDBG` | obu.rs:890 |  |  |
| `RAV2D_DEBUG` | av2_recon.rs:922 |  |  |
| `RECONDBG` | av2_frame.rs:1016, av2_recon.rs:5063 |  |  |
| `RILTRACE` | av2_recon.rs:6474 |  |  |
| `RMVC` | av2_recon.rs:4299, av2_recon.rs:4311, av2_recon.rs:4325 |  |  |
| `RMVFIN` | av2_recon.rs:4404 |  |  |
| `RMVREF` | av2_recon.rs:2803 |  |  |
| `RPDBG` | av2_refmvs.rs:1230 |  |  |
| `RUSTY_AV2D_ALLOW_SINGLE_PICTURE_HEADER` | obu.rs:290 | opt into the incomplete single-picture-header parse instead of refusing |  |
| `RUSTY_AV2D_DEBUG` | av2_recon.rs:922 | master switch: all dlog! output (every probe below also needs it unless noted) |  |
| `SBI` | av2_decode.rs:167, av2_recon.rs:5491 | inter-path partition probe (rng/dif per node) |  |
| `SBTRACE` | av2_recon.rs:1893, av2_recon.rs:4177, av2_recon.rs:5123 … |  |  |
| `SELFREF` | av2_recon.rs:6582 |  |  |
| `SEQDBG` | obu.rs:1818 |  |  |
| `TDBG` | obu.rs:2889 |  |  |
| `TIPDBG` | av2_recon.rs:1054, av2_recon.rs:3056, av2_recon.rs:3101 |  |  |
| `TMVSDBG` | obu.rs:3180 |  |  |
| `TPLALL` | av2_refmvs.rs:2141 |  |  |
| `UVCFDIF` | av2_recon.rs:2426, av2_recon.rs:2447 |  |  |
| `V320` | av2_recon.rs:9376, av2_recon.rs:9380, av2_recon.rs:9417 … |  |  |
| `V64` | av2_recon.rs:9377, av2_recon.rs:9522, av2_recon.rs:9581 |  |  |
| `WEDBG` | av2_refmvs.rs:637, av2_refmvs.rs:716 |  |  |
| `WORKBUDGET` | av2_recon.rs:950 |  |  |

Probes without a description are point instruments from specific bring-up
campaigns (their name matches the feature: `CFL*` CfL, `IBCDBG` intrabc,
`TXB*` transform blocks, `CC*` CCSO, …). When adding a probe, follow the
house pattern: env-gated, `[TAG]` prefix, print `rng` AND `dif` for anything
entropy-adjacent — `rng` alone hides bypass-only divergences.
