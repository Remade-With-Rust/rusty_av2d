# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
