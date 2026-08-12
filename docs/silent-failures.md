# Silent-failure audit — what comes out the back of a function when it fails?

**Date:** 2026-08-10, against 0.2.5. **Question:** does any function paper over a
failure by returning zeros / defaults / empty output while the caller reports
success? This is the decoder's defining risk — both real silent-output bugs this
cycle (the flat-grey still, the qp128 photo) were exactly this shape.

**Method:** exhaustive grep of the fallback vocabulary (`unwrap_or*` 94 sites,
`let _ =` 119, `.ok()` 18, `else { 0 }` 81, `=> 0,` / `return 0` 58), each site
classified by reading it in context; plus a control-flow audit of the two
mechanisms that can *manufacture* partial output: the work-budget hang guard and
the output-emit paths.

---

## Findings fixed (both were real)

### 1. A tripped work budget delivered the partial frame as a success

`work_tick()` (the ~64M-iteration hang guard from the hardening campaign) makes
every walker `break` when the budget trips — but **nothing downstream ever read
the tripped flag**. A corrupt stream that spun a walker would unwind mid-frame
and the half-decoded plane buffers were emitted as a normal picture, exit 0.
That is the silent-wrong-output failure mode in its purest form: the guard that
prevented the hang *created* undetectable garbage.

**Fix:** both emit paths (`emit_av2_output`, `emit_av2_planes`) now refuse a
frame whose budget tripped → `Err(InvalidArgument)`. Proven:
`WORKBUDGET=100000` on a valid clip now gives "No data decoded", exit 1;
normal decode unaffected. The flag resets at the next frame's `work_reset`, so
a later clean sequence recovers.

### 2. Empty planes at emit were silently dropped

`emit_av2_planes` returned `Ok(())` when `planes[0].w == 0` — a fabricated
frame with no content would simply *vanish from the output* with a success
status. A caller counting frames (or diffing against a reference) sees a short
file and no error. Now an error: empty planes at emit time means an upstream
bug, and it says so.

### 3. (Hardened, not user-visible) NS restoration unit with missing banked filters

The chroma-LR apply looked up per-unit filters and silently `continue`d on a
miss — shipping the unit unfiltered. A keying disagreement between the parse
and the apply (the exact bug class the cpu3 campaign was made of) would be
invisible. Now: a `dlog!` warning + `debug_assert!`. Not an `Err` because the
by-design case (frame filters on) never takes this path — the miss can only be
an internal invariant break.

---

## Classified as correct (why each fallback is not a silent failure)

| Class | Examples | Why it's sound |
|---|---|---|
| **Spec-mapped defaults** | neighbour ctx `a.get(x).unwrap_or(0)`; `nb_sum` absent-slot 0; `skip_ctx_luma` whole-block → 0; eob extra-bit width `_ => 0` | The spec defines the absent/edge case as zero. Corpus-proven byte-identical — a wrong default here cannot hide, the gate fails. |
| **Fuzz-hardening clamps** | `Plane::at` OOB → 0; palette map reads; deblock `dst.get(o).unwrap_or(&0)` | Added by the corrupt-stream campaign to convert panics into bounded garbage. On *valid* streams the index is in range (57/57 gate). On corrupt streams garbage output is acceptable **because the stream errors elsewhere or is undetectably corrupt anyway** — and now, if the corruption spins a walker, the budget guard errors loudly (finding 1). |
| **By-design absences** | `LR_UNITS.get(...).unwrap_or(REST_NONE)` (an SB simply owns no unit); `warp_pred.unwrap_or(IDENTITY_WARP)` (None = translational); `lr_src.unwrap_or(&pl0)` (LR off → GDF guides on the pre-LR plane) | The `None` case is a legitimate state with defined semantics, not an error being eaten. |
| **Debug/probe plumbing** | `let _ = fs::write(...)` capture dumps; `.ok()` on env parses | Failure of a diagnostic must never fail a decode. |
| **Discarded value-tuples** | `let _ = decode_b_chroma(...)`, `let _ = tip_pred_luma(...)` | Returns are values (mode/count), not `Result`s; the side effects (recon, context splats) are the point at those call sites. |
| **Invariant-backed unwraps** | `Picture` accessors' `seq_hdr.unwrap()` | A `Picture` exists only after a successful decode, which requires a sequence header (the dav1d-rs API shape). Panic-on-broken-invariant is loud, not silent. |

## The standing rule

A fallback value is acceptable only when one of these holds:
1. the spec defines the absent case (and the corpus gate would catch a wrong
   choice),
2. it converts a corrupt-stream panic into bounded garbage **and** the frame
   cannot complete as a false success (the budget guard is what closes that
   loop), or
3. it is diagnostic plumbing.

Anything else must return `Err`, warn via `dlog!`, or `debug_assert!`. When in
doubt: the failure the user cannot see is worse than the crash they can.

## Known remaining softness (documented, deliberate)

- The ~1,750 clamped index ops from the hardening campaign remain
  sampling-verified (fuzz), not structurally proven; the roadmap is checked
  accessors + `deny(clippy::indexing_slicing)` module by module (done:
  `av2_qm`, `av2_palette`).
- `Settings` setters use `InRange::new(...).unwrap()` — out-of-range input
  panics at the API boundary (loud, and upstream rav1d's shape), rather than
  returning `Result`. A candidate for a breaking API pass, not a silent risk.
