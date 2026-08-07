# Provenance & attribution

**`rusty_av2d`** is a **fork of [rav1d](https://github.com/memorysafety/rav1d)**
(the pure-Rust AV1 decoder, itself a port of
[dav1d](https://code.videolan.org/videolan/dav1d)), used as the starting
scaffold and then retargeted from AV1 to AV2.

- **Upstream:** rav1d v1.1.0 — © Rav1d Developers / Prossimo, and the dav1d
  authors / Two Orioles, LLC / VideoLAN.
- **License:** BSD-2-Clause (see [`COPYING`](COPYING)) — retained in full. This
  fork remains BSD-2-Clause.

BSD-2-Clause requires retaining the above copyright notice and the `COPYING`
text in source redistributions; both are preserved here.

## Relationship to upstream

The import began as a verbatim copy of rav1d. The tree has since been
retargeted to **AV2**: the bitstream parse, reconstruction, and in-loop filter
paths were rewritten against the AVM reference decoder, and the AV2-specific
modules (`src/av2_*.rs`) are new work. Substantial machinery inherited from
rav1d/dav1d remains in use — the entropy decoder, the picture/data plumbing,
the threading and allocation layers, the assembly bindings, and the public API
shape.

The public Rust API deliberately mirrors
[`dav1d-rs`](https://crates.io/crates/dav1d) so that existing AV1 integrations
can adopt AV2 with minimal change. The exported C ABI likewise keeps the
`dav1d_*` symbol names, which is why those identifiers appear throughout this
tree.

## Reference implementations used as oracles

Verification compares output against the **AVM** reference decoder
(`avmdec`, © Alliance for Open Media) and, where it supports the stream,
**dav2d**. Neither is vendored into this repository, and no code from AVM is
included here — they are used only as external oracles during testing.
