# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.3.0...rusty_av2d-v0.3.1) - 2026-11-10

### Fixed

- *(envflag)* compare variable names case-insensitively on Windows

### Other

- 2,589 active installs

## [0.3.0](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.2.13...rusty_av2d-v0.3.0) - 2026-11-08

Real `avmenc` streams now decode correctly, and decoding is 2.7–3.8× faster.

### Fixed

- Wrong output on real `avmenc` streams ([#14](https://github.com/Remade-With-Rust/rusty_av2d/issues/14)): frames now decode on the 8px-aligned coded area and are cropped only at output; IST, CCTX, luma per-unit Wiener and inter-intra are applied; compound warp-causal prediction is implemented; TIP-as-output frames are deblocked; plus delta-Q, BAWP, GDF tile-row, superblock-row context and CDF fixes. The nine streams from the issue match `avmdec` byte for byte.
- 10-bit: the difference-weighted compound mask used an 8-bit shift.
- 4:2:2: the inter transform-type context used the frame-clipped unit size.
- 4:4:4: CfL context, MHCCP corner counts and BAWP edge templates.
- Pending frames are flushed at end of stream, and the CLI exits non-zero on a decode error.

### Changed

- Streams using tools that aren't implemented now return an error instead of wrong pixels: segmentation, non-identity global motion, loop filtering disabled across tiles with several tiles, `tip_explicit_qp`, segmented quantization matrices, and monochrome (which used to panic).
- The corpus gains 12 clips with dimensions that aren't multiples of 8 (69 in all).

### Performance

- 2.7–3.8× faster than 0.2.13 on the 432×240 corpus clips (2.1–2.3× fewer instructions), output unchanged. See the README's Performance section for the method.
- The AVX2/NEON motion-compensation kernels now handle every fixed-filter 8-tap sample (12.8% before) and run per block; their safe entry points check bounds in release builds.
- Reference planes are shared instead of copied; loop restoration reads prefetched windows; GDF reads each sample's class once per row; scratch buffers are reused instead of zeroed per call; diagnostic switches are cached per call site; the corrupt-stream work budget is a thread-local decrement.

## [0.2.13](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.2.12...rusty_av2d-v0.2.13) - 2026-10-17

### Other

- bump rusty_alloc-api to =2.2.6 ([#12](https://github.com/Remade-With-Rust/rusty_av2d/pull/12))
- Merge pull request #7 from Remade-With-Rust/chore/release-plz-0-5-139

## [0.2.12](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.2.11...rusty_av2d-v0.2.12) - 2026-10-16

### Fixed

- compute the GDF stripe base without a usize underflow — debug builds panicked ("attempt to subtract with overflow", `av2_gdf.rs:549`) on the first stripe of every GDF frame, valid streams included; release output was already correct and is unchanged

## [0.2.11](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.2.10...rusty_av2d-v0.2.11) - 2026-10-08

### Other

- bump rusty_alloc-api to =2.2.5 ([#9](https://github.com/Remade-With-Rust/rusty_av2d/pull/9))
- 1,558 active installs
- 617 active installs
- update rusty_av2d version to 0.2 in README
- 532 active installs

## [0.2.10](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.2.9...rusty_av2d-v0.2.10) - 2026-08-28

### Other

- bump rusty_alloc-api to =1.1.6 ([#4](https://github.com/Remade-With-Rust/rusty_av2d/pull/4))

## [0.2.9](https://github.com/Remade-With-Rust/rusty_av2d/compare/rusty_av2d-v0.2.8...rusty_av2d-v0.2.9) - 2026-08-28

### Other

- add release-plz so merged dependency bumps actually reach crates.io ([#2](https://github.com/Remade-With-Rust/rusty_av2d/pull/2))
- bump rusty_alloc-api to =1.1.4 ([#1](https://github.com/Remade-With-Rust/rusty_av2d/pull/1))
