# rusty_av2d

**A pure-Rust AV2 video decoder. No C, no FFI, no `unsafe` shelling out to a reference library.**

`rusty_av2d` decodes [AV2](https://aomedia.org/) — the successor to AV1 currently
being developed by the Alliance for Open Media — and produces output that is
**byte-identical to the AVM reference decoder** across a 47-clip conformance
corpus covering every major tool in the format.

[![crates.io](https://img.shields.io/crates/v/rusty_av2d.svg)](https://crates.io/crates/rusty_av2d)
[![docs.rs](https://docs.rs/rusty_av2d/badge.svg)](https://docs.rs/rusty_av2d)
[![license](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](COPYING)

> **Research preview.** AV2 is not a finalized standard. Read
> [Limitations](#limitations) before depending on this — especially if you care
> about long-term bitstream stability or decode speed.

---

## Why this exists

Every AV2 decoder available today is the C reference (`libavm`). This is the
first independent implementation, and it is written in a memory-safe language.

The goal was never "fast" — it was **provably correct**. The project is
verified the way a codec should be: not by eyeballing output, but by decoding
real streams and comparing every byte against the reference implementation.
When our output and AVM's output differ by a single bit in a single pixel, the
build is red.

## What "byte-identical" means here

47 clips, each decoded and compared byte-for-byte against AOM's `avmdec`
(and against `dav2d` where it supports the stream). Any mismatch fails the
gate. Alongside them run 112 unit and integration tests. Both must pass before
any change lands.

The corpus exercises:

| Area | Coverage |
|---|---|
| Intra | All prediction modes, palette / screen content, IntraBC + `morph_pred` |
| Inter | Compound prediction, B-pyramids, the warp family, wedge compound, motion modes |
| TIP | Temporal interpolated prediction, including TIP-as-output frames |
| Transforms | TX partitioning, secondary transforms, quantization matrices |
| In-loop filters | Deblocking, CDEF, loop restoration, CCSO, GDF |
| Frame types | Key, inter, S-frames, film grain, delta-Q |
| Geometry | 64px and 128px superblocks, arbitrary (non-aligned) frame dimensions |
| Formats | Full {8-bit, 10-bit} × {4:2:0, 4:2:2, 4:4:4} matrix |
| Tiling | Multi-tile streams and their cross-products |

Eleven of the clips are single-tool-disabled controls: each one turns off
exactly one sequence-header tool, verifying that the gate for that tool is
honored in *both* directions.

## Install

```toml
[dependencies]
rusty_av2d = "0.1"
```

Requires Rust 1.79+. No `nasm`, no C toolchain, no build-time codegen — the
crate is pure Rust and builds with cargo alone.

### Pick an allocator

The decoder allocates heavily — roughly 190 sites across the decode path, many
of them per transform unit. On a system heap that cost **dominates the profile**.
Swapping in [`rusty_alloc`](https://crates.io/crates/rusty_alloc) (a pure-Rust
mimalloc remake) measured **1.38× end-to-end**, byte-identical output, for one
line:

```rust
#[global_allocator]
static ALLOC: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;
```

The bundled CLI does this. The **library deliberately does not** — installing a
global allocator from a library hijacks every dependent's choice. If you embed
this crate, set one in your binary; it is the single largest speedup available
here, larger than fully vectorizing every kernel could deliver.

## Usage

The decoder exposes a safe Rust API that mirrors
[`dav1d-rs`](https://crates.io/crates/dav1d), so it drops into existing code
with minimal changes:

```rust
use rusty_av2d::{Decoder, PlanarImageComponent};

let mut dec = Decoder::new()?;

// Feed one temporal unit at a time (e.g. an IVF frame payload).
dec.send_data(packet.into_boxed_slice(), None, Some(pts), None)?;

// Drain everything that unit produced.
loop {
    match dec.get_picture() {
        Ok(pic) => {
            let y = pic.plane(PlanarImageComponent::Y);
            let stride = pic.stride(PlanarImageComponent::Y) as usize;
            println!(
                "{}x{}, {}-bit, {} bytes of luma",
                pic.width(), pic.height(), pic.bit_depth(), y.len()
            );
            let _ = stride;
        }
        // Needs more input before another picture is available.
        Err(rusty_av2d::Rav1dError::TryAgain) => break,
        Err(e) => return Err(e.into()),
    }
}
```

A `Settings` builder controls thread count, frame delay, film-grain
application, operating point, and which in-loop filters run:

```rust
use rusty_av2d::{Decoder, Settings};

let mut settings = Settings::new();
settings.set_n_threads(4);
settings.set_apply_grain(true);
let mut dec = Decoder::with_settings(&settings)?;
```

### The C ABI

A C ABI is also exported (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`,
…) so the crate can be linked as a `staticlib` by non-Rust callers. It is on by
default via the `capi` feature.

Those symbol names are inherited from the dav1d lineage, which means they
**collide at link time with any other decoder from that lineage** — notably
`rav1d`. If you link both decoders into one binary, build with default
features off:

```toml
rusty_av2d = { version = "0.1", default-features = false, features = ["bitdepth_8", "bitdepth_16"] }
```

That drops the C ABI, leaving no unmangled symbols. The safe Rust API is unaffected either way — this is exactly how
[`remade_ffmpeg_rs`](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)
links AV1 and AV2 side by side.

## Limitations

These are real, and you should read them before building on this.

1. **AV2 is not a finalized standard.** There are no official conformance
   vectors. "Correct" here means bit-exact against AVM as of this commit — the
   bitstream itself may still change under us.
2. **Performance is unoptimized.** Correctness was the only goal. The decoder
   is **fully scalar** — no SIMD and no assembly. Samples are stored
   one-per-`i32` with a bounds check on every read, which costs motion
   compensation roughly 20× what the arithmetic alone would. Do not benchmark
   this against `libavm` and expect a fair fight.
   [`docs/plan.md`](docs/plan.md) has the profile and the plan to fix it.
3. **Research-grade internals.** The AV2 path uses thread-local state rather
   than the upstream pipeline structure, and retains in-tree diagnostic probes
   (silent unless `RUSTY_AV2D_DEBUG=1`).
4. **Threading is under-tested.** Correctness is gated single-threaded;
   multi-threaded output has only been spot-checked.
5. **Decoder only.** There is no encoder in this crate.

See [`STATUS.md`](STATUS.md) for the full scope statement.

## Robustness

Corrupt input is fuzz-tested with bit-flips, truncation, and header mutation.
The most recent 1000-case campaign produced zero panics and zero hangs;
malformed streams return errors rather than crashing.

That is sampling, not proof. Two modules (`av2_qm`, `av2_palette`) are
additionally panic-free *by construction* under
`#![deny(clippy::indexing_slicing)]`; the rest of the decode path is not yet.

## Provenance

`rusty_av2d` is a fork of [rav1d](https://github.com/memorysafety/rav1d), the
pure-Rust AV1 decoder, which is itself a port of
[dav1d](https://code.videolan.org/videolan/dav1d). The AV1 scaffold was
retargeted to AV2 by inverting the AVM reference decoder's parse, brick by
brick, against a symbol-level oracle.

Licensed **BSD-2-Clause**, retained in full from upstream — see
[`COPYING`](COPYING) and [`NOTICE.md`](NOTICE.md). Copyright is held by the
dav1d authors, VideoLAN, Two Orioles LLC, the Rav1d Developers, and Prossimo,
alongside subsequent AV2 work.

## Related

Part of [Remade With Rust](https://github.com/Remade-With-Rust) — a family of
pure-Rust media codecs. `rusty_av2d` plugs into
[`remade_ffmpeg_rs`](https://github.com/Remade-With-Rust/remade_ffmpeg_rs) as
the `rff-codec-av2` backend.
