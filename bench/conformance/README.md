# rav2d conformance harness (Phase 0)

The regression spine for growing rav2d into a full AV2 decoder. Mints AV2 test streams
with the **avm reference encoder** and gates rav2d's decoded output, byte-for-byte,
against a reference decoder.

```
avmenc  ──encode──▶  corpus/<name>.ivf  ──decode──▶  {dav2d | avmdec}  ──▶ ref.yuv
                                         ──decode──▶  rav2d (mine)      ──▶ mine.yuv
                                                                   cmp ref.yuv mine.yuv
```

The oracle is **dav2d** (the decoder rav2d is built to match); `ORACLE=avmdec` cross-checks
against the AVM **reference** decoder. On the golden clip all three agree byte-for-byte.

## Tools
- `mksrc.py <out.yuv> <w> <h> <nframes> [seed]` — synthesize a raw I420 source (moving box +
  gradients + light noise → real signal for intra **and** inter tools).
- `gen.sh <name> <src.yuv> <w> <h> <nframes> [avmenc flags…]` — encode a source into
  `corpus/<name>.ivf` and append to `manifest.txt`. Env: `QP` (0–255, default 128),
  `CPU` (avmenc speed), `KFMAX` (keyframe interval).
- `run.sh [name-glob]` — decode every matching corpus stream with the oracle and with rav2d,
  `cmp` the two, print PASS/FAIL. Exit non-zero if any FAIL (CI gate). Env: `ORACLE=avmdec`,
  `KEEP=1` (keep the per-stream YUVs under `$TMPDIR/rav2d_conf`).

## Layout
- `corpus/*.ivf` — the permanent test vectors (small; kept in-repo). `manifest.txt` describes them.
- `sources/*.yuv` — transient raw sources used only to mint streams (regenerate with `mksrc.py`).

## Typical use
```bash
# mint a stream that isolates a feature, then gate it
python bench/conformance/mksrc.py bench/conformance/sources/s720.yuv 1280 720 12
bash   bench/conformance/gen.sh cif_12f_q100 bench/conformance/sources/s720.yuv 1280 720 12 --qp=100
bash   bench/conformance/run.sh                 # gate the whole corpus
bash   bench/conformance/run.sh cif_12f_q100    # just one
```

## Status (Phase 0 complete)
`golden_2f` (the original byte-exact 2-frame clip) **PASSes**; the fresh streams **FAIL**
(rav2d emits no output) — the honest signal that the decode path is still a clip-specific
scaffold. Making those fresh streams pass is **Phase 1** (see the `rav2d-full-decoder-plan`
memory): kill the hardcoded 432×240, unify the per-frame loop, real reference management,
multi-tile, retire the `dav_*.yuv` crutches.
