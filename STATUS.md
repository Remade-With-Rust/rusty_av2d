# rusty_av2d — status, scope, and known limitations

**Research preview.** `rusty_av2d` is a pure-Rust AV2 decoder, forked from
[rav1d](https://github.com/memorysafety/rav1d) (see `NOTICE.md` for provenance,
`COPYING` for the BSD-2-Clause license).

## What is verified

Every clip in a **45-clip conformance corpus** decodes **byte-identical** to the
reference decoders (AOM's `avmdec`, and `dav2d` where it supports the stream).
The suite also runs 111 unit/integration tests. Both gate every change.

Coverage includes: all intra modes, compound prediction and B-pyramids, TIP
(including TIP-as-output), the warp family, all four in-loop filters, delta-Q,
loop restoration, wedge compound, TX partitioning, 128px superblocks and their
cross-products, the full {8,10-bit} x {4:2:0, 4:2:2, 4:4:4} matrix, film grain,
S-frames, palette / screen content, and quantization matrices. Eleven of the
clips are single-tool-disabled controls that verify each sequence-header tool
gate in both directions. Two clips (640x360, 424x240) cover frame dimensions
that are not superblock-aligned on either axis.

## Known limitations — read before using

1. **AV2 is not a finalized standard.** There are no official conformance
   vectors. "Correct" here means bit-exact against AVM/dav2d as of this commit;
   the bitstream may still change.
2. **Performance is unoptimized and unmeasured.** Correctness was the only goal.
   The decode path also carries ~425 bounds/liveness guard calls (`work_tick`)
   used for corrupt-input hardening.
3. **Research-grade internals.** The AV2 path uses thread-local state rather
   than the upstream pipeline structure, and retains in-tree diagnostic probes
   (all silent unless `RUSTY_AV2D_DEBUG=1`).
4. **Threading is under-tested.** Correctness is gated single-threaded;
   multi-threaded output has only been spot-checked.

### Previously-known, now resolved

Earlier revisions rejected frames whose dimensions were not a multiple of 16
(and before that, anything above 512px). Both root causes are fixed: the
intrabc `morph_pred` refinement was parsed but never applied at recon
(key-frame corruption), and the frame-edge chroma-reference boundary clause was
missing from the extended-partition gates (inter-frame entropy desync at
partial superblocks). No dimension restriction remains.

## Robustness

Corrupt input is fuzz-tested two ways: by mutating real streams (bit-flips,
truncation, header mutation), and by feeding uniformly random bytes straight to
the decode entry point. Both currently produce zero panics and zero hangs, and
malformed streams return errors rather than crashing.

The two distributions find different bugs. A 1000-case corpus-mutation campaign
was clean, but the first random-byte run found an unsigned underflow on a
corrupt tile-size prefix (fixed in 0.1.3) — mutation rarely produces a size
field wildly larger than the data around it, and random bytes do so constantly.
If you fuzz this decoder, run both.

This is sampling, not proof — two
modules (`av2_qm`, `av2_palette`) are additionally panic-free *by construction*
under `#![deny(clippy::indexing_slicing)]`; the rest are not yet.

## Debugging

All tracing is behind `RUSTY_AV2D_DEBUG=1` (`RAV2D_DEBUG` is a legacy alias). Capture-oracle comparisons additionally
need `DAVCAP=1` and write to `DAVCAP_DIR` (default: system temp).
