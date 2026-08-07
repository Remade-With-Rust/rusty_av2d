# Contributing to rusty_av2d

Thanks for looking. This project has one unusual property that shapes every
contribution: **correctness is defined by byte-equality with the AVM reference
decoder**, not by tests we wrote ourselves.

## The gate

Two things must pass before any change lands. Both are non-negotiable.

```sh
# 1. The conformance corpus — every clip must be byte-identical to the reference.
bash bench/conformance/run.sh

# 2. Unit and integration tests.
cargo test --release
```

`run.sh` decodes each clip in `bench/conformance/corpus/` with this decoder and
with the reference, then compares the raw output byte-for-byte. A single
differing byte in a single frame fails the run. There is no tolerance, no PSNR
threshold, and no "close enough".

### You need the reference decoders

The corpus harness shells out to AOM's `avmdec` (and `dav2d` for streams it
supports). Neither is vendored here — you have to build them yourself from
their upstream repositories, and point the harness at them. Where the two
oracles disagree, **AVM wins**: it is the normative implementation.

## Adding a clip

New coverage is the most valuable contribution. `bench/conformance/gen.sh`
mints a stream with the reference *encoder* and registers it:

```sh
python3 bench/conformance/mksrc.py sources/mysrc.yuv 432 240 8 7 420 8
bash bench/conformance/gen.sh myclip sources/mysrc.yuv 432 240 8 --enable-cdef=0
```

A clip earns its place if it exercises a tool, a format, or a geometry that no
existing clip reaches. A clip that only re-covers ground is churn — the corpus
runs on every change, so each addition is a permanent cost.

## Debugging a divergence

When output diverges, do not start from the pixels. Pixel diffs tell you
*that* you diverged, never *where*. The workflow that actually works:

1. **Compare pre-filter frames first.** In-loop filters smear a single bad
   block across a wide area and will send you to the wrong place.
2. **Find the first divergent block in decode order**, not raster order.
3. **Bisect the entropy stream.** All tracing lives behind
   `RUSTY_AV2D_DEBUG=1`; the in-tree probes print the arithmetic-decoder state
   (`rng`, and `dif` where it matters) at partition, mode, and coefficient
   boundaries. Add the matching `fprintf` to AVM and diff the two traces
   line-for-line.
4. **Compare the full entropy state, never `rng` alone.** Bypass and literal
   bits change `dif` and the bit counter while leaving `rng` untouched, so a
   real divergence can hide behind a matching `rng` for many symbols.

A geometry-correlated failure is almost never geometry math. It usually means
the encoder selected a *tool* at that geometry which the corpus had never
reached before — chase the first divergent symbol, not the spatial pattern.

## Style

- `cargo fmt` before committing; `rustfmt.toml` is checked in.
- Match the surrounding code. The AV2 modules (`src/av2_*.rs`) carry dense
  reference citations in comments (`avm reconinter.c:4106`, `dav recon_tmpl.c:2731`).
  Keep that up — those citations are how the next person finds the ground truth.
- Comments should state constraints the code cannot show. Don't narrate what
  the next line does.

## Licensing

BSD-2-Clause, inherited from dav1d/rav1d and retained in full. Contributions
are accepted under the same license. Do not paste code from AVM or any other
reference implementation into this tree — it is used strictly as an external
oracle. See [`NOTICE.md`](NOTICE.md).
