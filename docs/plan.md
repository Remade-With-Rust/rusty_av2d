# Plan: SIMD/NEON, and removing the assembly

**Status:** Phases 0–3 are **executed** (0.2.7, 2026-08-10). Phase 3 (assembly
removal) was done first; Phase 0 (harness), Phase 1 (structural), and the first
Phase 2 kernel families landed together. Results below; the honest headline is
the plan's own prediction confirmed: **the data-layout work was the win, the
intrinsics measured ~neutral on x86-64 against the restructured scalar.**

## Execution results (2026-08-10, 0.2.7)

**Phase 0 — done.** `profiling` cargo feature (12 stage timers, zero cost when
off; report at CLI exit), `bench/bench.py` (Python driver — the Git Bash sweep
wedge is a fork-storm problem, Python spawning avoids it), committed baseline
`bench/baselines/p1_start.json`. Incidental find: enabling the feature exposed
that the CLI's MSVC `getopt.c` shim had gone with the C tree and only build
caching hid it — replaced with a pure-Rust `getopt_long`
(`tools/compat/getopt.rs`), so the tools are now C-free too.

**Phase 1 — the structural bricks landed and dominated, as predicted.**
- `work_tick` hoisted out of 77 innermost per-pixel loops (the bounded
  `for x in 0..w` loops cannot spin by construction; walkers and row loops keep
  their ticks, and the budget-trip → error guard from 0.2.6 still fires —
  verified).
- MC: per-call `mid` allocation → thread-local scratch; per-sample
  clamped `at()` → an interior/edge split (one containment test per call,
  direct row-slice access inside; the clamped path only for genuine boundary
  blocks). `mc_translate`/`mc_translate_prep` now share one `mc_core`.
- Measured (idle machine, median of 5, byte-identical): **tipout −47%,
  film-grain clip −36%, 640×360 −19.5%**, sb128+CDEF −56% (later run).
- Post-change profile (tipout): `inter_mc` 14.7% → **8.5%**; coef 0.1%,
  itx 0.2%, recon_pad 0.2%. The remaining ~85% of the SB loop is the untimed
  per-block recon glue — **sample-type narrowing (item 1) is where the next
  factor lives**, and it remains open.

**Phase 2 — first kernel families in, default-on, honestly neutral on x86.**
- `src/simd.rs`: 8-tap FIR row/column kernels (the MC H/V passes) and the
  compound blends (avg / w_avg / mask), each as AVX2 + NEON + scalar twin,
  runtime-dispatched (`simd_level()`, cached; `RUSTY_AV2D_NOSIMD=1` opts out).
  Twin tests assert bit-equality over randomized inputs; corpus 57/57 with the
  kernels live; aarch64 cross-compiles clean (real-ARM verification still due).
- ABBA-alternated A/B vs the restructured scalar: **0.84–1.09× by clip,
  medians equal** — LLVM already auto-vectorizes the cleaned slice loops, and
  on `i32` lanes the intrinsics have no width advantage. Kept default-on
  because they never lose beyond noise, they carry the NEON path (autovec
  quality there unverified), and they are the skeleton the `u8`/`u16`
  narrowing will widen into 16/32-lane wins.

**What remains of the original plan:** Phase 1 item 1 (sample-type narrowing to
`u8`/`u16` — the single biggest remaining lever, and what makes the SIMD lanes
real), item 4 (recon-pad audit), the rest of the Phase 2 map (#4 Wiener,
#5 CDEF, #7 deblock, #8/#9 intra), and threading.

---

*(Original plan follows, unchanged, for the record.)*

Written 2026-08-07 against `rusty_av2d` 0.1.3.

**2026-08-10 — the inherited C tree is gone too.** Separate from the assembly:
rav1d carried dav1d's original C sources alongside its Rust port, and this fork
inherited 124 `.c`/`.h` files (~37,600 lines) plus the meson build that drove
them. Dead in every sense — nothing compiled it, nothing referenced it, and it
was already excluded from the published package — but it made a pure-Rust
decoder read as roughly 30% C. Deleted; corpus 47/47 and 112 tests unchanged,
and the published package is byte-for-byte identical, so no release was needed.

The goal is a decoder that is fast because of portable Rust SIMD, and that carries
no hand-written assembly at all. This document says what we measured, why the
obvious ordering is wrong, and what to do in what order.

---

## 1. Where we actually are

### The assembly was dead code (now removed — see Phase 3)

All 47 x86 `.asm` files plus the ARM `.S` sources (8.6 MB total, ~1.1 MB linked)
compiled under the `asm` feature, but **nothing in the AV2 decode path reached them.** The assembly
is dispatched through dav1d's DSP function-pointer tables, and *zero* `av2_*`
modules reference `Rav1dDSPContext`. Every DSP family got a fresh pure-Rust AV2
implementation during bring-up:

| dav1d/AV1 asm family | AV2 replacement |
|---|---|
| `mc` | `av2_inter.rs` |
| `itx` | `av2_itx.rs` |
| `ipred` | `av2_ipred.rs` |
| `loopfilter` | `av2_deblock.rs` |
| `cdef` | `av2_filter.rs` (CDEF + CCSO) |
| `looprestoration` | `av2_lr.rs` |
| `filmgrain` | `av2_grain.rs` |

That is the complete set, so there is nothing left for the assembly to serve.

Verified empirically, not just by reading: built with and without `asm`,
ABBA-alternated three timing pairs on the largest corpus clip. The `asm` build ran
762–1102 ms, the no-`asm` build 756–854 ms — pure noise, no separation — and the
outputs are byte-identical. A live vector path would not be invisible.

### There is no SIMD either

Zero uses of `std::arch`, `core::arch`, `_mm*`, NEON intrinsics, `target_feature`,
or portable SIMD anywhere in the AV2 modules. The only file in the crate touching
`core::arch` is `cpu.rs`, and that is CPU feature *detection* — retained because
runtime SIMD dispatch will need it. The decoder is fully scalar; the only vectorization is
whatever LLVM auto-vectorizes out of scalar loops, which on this branchy,
table-driven code is very little.

### The profile — and the surprise

Measured on 640×360, 30 frames (1 key + 29 inter), release build, stage timers
around each stage. `samply` needs Administrator on Windows and was unavailable, so
this is coarse instrumentation rather than a sampling profile — treat the split as
first-order, not exact.

| Stage | ms | % of total |
|---|---:|---:|
| SB decode loop | 6451 | 87.8% |
| In-loop filters (deblock, CDEF, LR, CCSO, GDF) | 896 | 12.2% |
| — inter MC | 960 | 13.1% |
| — inverse transforms | 29 | 0.4% |
| — intra prediction | 7 | 0.1% |
| — inter block parse (incl. `refmvs_find`) | 42 | 0.6% |
| — **residual** (recon write-back, dequant, blending, padded-buffer mirroring) | **5413** | **73.7%** |

Three things fall out of this, and they reshape the whole plan:

**The classic SIMD targets are a minority of runtime.** MC + transforms + intra +
filters together are ~26%. Even *infinitely fast* kernels cap the whole-decoder
speedup at about **1.35×** by Amdahl. Porting dav1d's assembly one-for-one, or
writing intrinsics for those kernels first, buys far less than it looks like.

**Entropy decode is not the bottleneck.** The entire inter block parse, including
the DRL stack construction, is 0.6%. This is the opposite of `rav1d`, where
coefficient decode is ~53% serial MSAC and the decoder is at its floor. We are
nowhere near that floor — which is good news, because the remaining cost is the
kind that responds to work.

**The `work_tick` hardening guards are not a meaningful cost.** 4.66 M calls over
the run, well under 1% of runtime. `STATUS.md` implied otherwise; corrected.

### Why the kernels are slow — the root cause

Inter MC costs **~139 ns per output pixel (~500 cycles)**. An 8-tap separable 2D
filter is roughly 16 multiply-accumulates per pixel; this is about **20× off** what
that should cost. The reason is structural, not algorithmic:

```rust
pub struct Plane { pub px: Vec<i32>, .. }               // i32 per sample

pub fn at(&self, x: usize, y: usize) -> i32 {
    self.px.get(y * self.stride + x).copied().unwrap_or(0)   // bounds check per sample
}

let get = |x, y| rf.at(x.clamp(0, rw - 1) as usize,          // two clamps per tap
                       y.clamp(0, rh - 1) as usize);
```

Every sample read in the innermost tap loop pays a bounds check, two clamps, and a
4-byte load for what is 8-bit data. That is 4× the memory traffic it needs, and it
puts **4 samples in a 128-bit vector where 16 belong.**

This is the crux: **you cannot vectorize this code as written.** A per-sample
`Vec::get` with per-coordinate clamping has no vector form. The structural work is
not a nice-to-have before SIMD — it is the thing that makes SIMD expressible at
all, and on these numbers it is also where most of the win lives.

---

## 2. Why the stated order is wrong

The natural instinct is "add SIMD/NEON, then delete the assembly." Two corrections
from the measurements:

**Removing the assembly does not need to wait.** It is provably dead. Nothing
depends on it, deleting it cannot regress output, and it costs a `nasm` build
dependency, ~1.1 MB of binary, 8.6 MB of source, and a documented-but-false claim
in the README. Do it first, in isolation, so the diff is unambiguous. *(Done.)*

**SIMD should not come first either.** Writing intrinsics against `Vec<i32>`
planes with clamped per-sample gets would mean writing the SIMD *twice* — once
against today's data layout, and again after the layout is fixed. Do the
structural work first; it is worth more on its own, and it is the precondition.

Revised order: **remove asm → fix the data layout → then SIMD.**

---

## 3. The plan

### Phase 0 — Measurement harness — **partly done**

Nothing here is worth doing blind, and the stage numbers above came from
throwaway instrumentation that no longer exists.

- **DONE — repeatable benchmark.** `bench/bench.sh` times the corpus and reports
  a median plus min–max spread per clip, with `SAVE=`/`BASELINE=` for recording
  and comparing. Warm-run spread is ~2%, so that is the floor below which a
  change is not a result.
- **TODO — a committed baseline.** Not recorded yet: under Git Bash/MSYS a full
  sweep wedges partway through, leaving unkillable decoder processes, and every
  later spawn stalls. De-forking the timing path (`EPOCHREALTIME` instead of two
  `date` spawns per decode) helped but did not fix it. Individual clips and small
  batches are reliable. Record the baseline under WSL/Linux, or in batches.
- **TODO — stage profiler.** A `profiling` cargo feature (default **off**, zero
  cost when disabled) with timers around: SB loop, each in-loop filter, MC,
  transforms, intra, coef decode, recon write-back. Needed to re-derive the
  Phase 2 ordering after Phase 1 shifts the balance.

**Exit:** one command prints the stage table, and re-running it twice agrees
inside noise.

### Phase 1 — Data layout (the actual bottleneck)

This is where the 73.7% lives. Each item is separately measurable and separately
revertable.

1. **Narrow the sample type.** `Vec<i32>` → `u8` for 8-bit and `u16` for high
   bit depth, behind the existing `bitdepth_8` / `bitdepth_16` split (dav1d's
   `BitDepth` generic pattern is already vendored here and can be reused).
   Expect a large win from memory traffic alone, before any vectorization.
2. **Get bounds checks out of the inner loops.** Replace per-sample `at()` in hot
   kernels with slice-per-row access acquired once, so the check is amortized per
   row rather than paid per sample.
3. **Split edge handling from the interior.** Today every tap clamps. Instead,
   detect the common case where a block plus its filter margin is entirely inside
   the frame and run an unclamped fast path; keep the clamped path only for blocks
   that genuinely touch the boundary. This is what makes the inner loop
   vectorizable.
4. **Audit the recon write-back and the padded-buffer mirror.** `write_recon_pad`
   copies every block a second time. Confirm whether that buffer is still load-
   bearing now that recon is complete, and if so whether it can be written once
   rather than mirrored.
5. **Hoist `work_tick` out of the innermost loops.** It is an opaque call with a
   side effect and an early `break` on every pixel, which forbids vectorization
   outright — 66 sites in `av2_inter`, 47 in `av2_ipred`, 20 in `av2_itx`. It
   costs under 1% of runtime, so this is not a performance fix; it is the
   precondition for Phase 2 existing at all. Move the budget check to per-row or
   per-block, where it still bounds a corrupt stream.
6. **Set a global allocator — DONE, measured 1.38x on its own, for one line.**
   The decoder currently uses the system allocator, and it is allocation-bound
   to a degree nothing else in this document predicted. Measured on the 30-frame
   640x360 clip, median of 7 warmed runs, corpus 45/45 byte-identical throughout:

   | Config | Median | Speedup |
   |---|---:|---:|
   | baseline (system heap, `vec![]` per call) | 4808 ms | 1.00x |
   | + MC scratch buffer only | 4380 ms | 1.10x |
   | + [`rusty_alloc`](https://crates.io/crates/rusty_alloc) only | **3473 ms** | **1.38x** |
   | + both | 3405 ms | 1.41x |

   This was the single cheapest win available — a bigger measured speedup than
   the entire kernel-SIMD phase can deliver (Amdahl ceiling ~1.35x), for one
   `#[global_allocator]` line. **Shipped in 0.2.1**: `rusty_alloc` is installed in
   both `[[bin]]` roots, per the Remade With Rust convention that the allocator
   belongs in binaries and bench roots and never in a library (a library that
   installs one hijacks every dependent's choice). The README documents it as a
   recommendation for embedders.

   It is also now the **benchmark baseline** (`bench/bench.sh`,
   `bench/baselines/`). Later phases quote deltas against the shipped
   configuration; measuring against the system heap would flatter every
   subsequent change by the 1.38x already banked.

7. **Then stop allocating inside kernels.** With a fast allocator in place the
   marginal win shrinks (3473 -> 3405, ~2%), so this is now a cleanup rather than
   the headline — but it is still worth doing, and `vec![0; n]` additionally pays
   for a memset that the next loop immediately overwrites. ~190 allocation sites
   across the decode path, many per-block: `pred`, `coeff`, `residual`, and `cf`
   are allocated per transform unit. Replace the hot ones with thread-local
   scratch (an RAII guard that returns the buffer on drop keeps the call sites
   unchanged).

**Gate for every step:** conformance corpus 45/45 byte-identical, plus the test
suite. This work must not change a single output byte — that is exactly what makes
it safe to do aggressively.

**Exit:** re-profile. Phase 2 is planned against the *new* split, not this one,
because the balance will have shifted.

### Phase 2 — SIMD and NEON

The kernel-by-kernel map. Every entry names the function, the file, what shape it
is, and what has to be true before it can be vectorized at all.

#### Blockers that apply to every kernel

Two things currently make *all* of these unvectorizable regardless of ISA. Both
must be cleared first; neither is SIMD work.

**`work_tick` sits in the innermost per-pixel loops.** It is an opaque call with a
side effect and an early `break` on every iteration, which forbids vectorization
outright. Counts: `av2_inter` 66, `av2_ipred` 47, `av2_itx` 20, `av2_deblock` 8,
`av2_filter` 7, `av2_lr` 5. It costs under 1% of runtime, so this is not a
performance fix — it is a *correctness-of-shape* fix. Hoist the budget check to
per-row or per-block granularity, where it still bounds a corrupt stream but no
longer sits between the compiler and the loop.

**Per-call heap allocation inside kernels.** `av2_itx` 12 sites, `av2_ipred` 9,
`av2_inter` 6, `av2_lr` 4. Motion compensation allocates its intermediate buffer
on *every one of 40,466 calls*. Replace with reusable thread-local scratch sized
once per frame.

#### The map

| # | Kernel | File | Shape | Vectorizes? | Blockers beyond the two above |
|---|---|---|---|---|---|
| 1 | `mc_translate` 2D path | `av2_inter.rs:78` | separable 8-tap H then V, `i32` mid buffer | **Ideal.** Textbook dot-product; 8 taps × 16 lanes | `get()` closure clamps per tap; heap `mid` per call |
| 2 | `mc_translate` H-only / V-only | `av2_inter.rs` | single 8-tap pass | **Ideal**, same shape | same |
| 3 | `comp_avg`, `comp_w_avg`, `comp_mask`, `comp_w_mask_ss`, `bacp_mask` | `av2_inter.rs:242–343` | elementwise blend of two buffers | **Trivial.** Pure elementwise, no gather | none — do these first, they are the cheapest wins |
| 4 | `lr_filter_luma` (Wiener) | `av2_lr.rs:176` | separable 7-tap + NS/PC classes | **Ideal**, same family as MC | per-class branch should hoist out of the pixel loop |
| 5 | `cdef_filter_block` | `av2_filter.rs:97` | per-pixel, 2 pri + 4 sec taps at direction-dependent offsets, running min/max | **Good.** Offsets are constant per call; min/max are `vmin`/`vmax` | branchy `pri_strength>0` / `sec_strength>0`; specialize into 3 variants |
| 6 | `cdef_find_dir` | `av2_filter.rs:157` | 8×8 directional cost search | **Moderate.** Reduction-heavy | called once per 8×8, lower value than the filter |
| 7 | `deblock` | `av2_deblock.rs:104` | 4 rows × short filter, strided both ways | **Moderate.** Vertical edges are contiguous; horizontal edges need a transpose | `filter_choice` is a data-dependent early-out per edge |
| 8 | `dr_z1_idif` / `dr_z3_idif` / `dr_z2_idif` | `av2_ipred.rs:268–347` | directional intra, per-pixel interpolation from an edge array | **Moderate.** Gather-ish but the index step is affine per row | blocks are small; win is per-block-size specialization |
| 9 | `ipred_dc*` / `ipred_ibp_dc` | `av2_ipred.rs:71–134` | sum-then-splat, then a weighted blend | **Easy** (horizontal reduction + broadcast) | small blocks; modest absolute win |
| 10 | `inv_dct*_1d` / `inv_adst*_1d` / `inv_ddt*_1d` | `av2_itx.rs:148–245` | 1D butterflies over **strided** access `c[j*stride]` | **Hostile as written.** Strided loads kill it | needs the dav1d approach: operate on N rows at once and transpose between passes, not one strided vector |
| 11 | `av2_grain` synthesis | `av2_grain.rs` | AR filter + per-pixel blend | **Good**, and already tick-free | only runs on film-grain streams |
| 12 | `sad_nxn`, `prep_opfl` | `av2_inter.rs:493,415` | absolute-difference reduction | **Trivial** (`vabsd` + horizontal add) | small share of runtime |

#### Ordering

Do them in this order, re-deriving after Phase 1 rather than trusting today's
split:

1. **#3 compound blends** — elementwise, no edge cases, immediate win, and they
   establish the scalar-twin test harness on the easiest possible case.
2. **#1/#2 motion compensation** — the largest single kernel share (13.1%).
3. **#4 Wiener** and **#5 CDEF** — the bulk of the 12.2% filter stage.
4. **#7 deblock**, then **#8/#9 intra**.
5. **#10 transforms last.** They are 0.4% of runtime *and* the hardest shape.
   Porting dav1d's transform assembly first would be the single worst use of
   effort available here.

#### How, not just where

- **Portable SIMD first.** `std::simd` where it is expressive enough: one source
  covering x86 and aarch64, no per-architecture duplication. Nightly-only today,
  so gate it behind a feature and keep scalar as the stable default.
- **Intrinsics only where portable SIMD leaves measurable performance behind**, and
  then per-kernel rather than wholesale: `core::arch::x86_64` (SSE4.1/AVX2) and
  `core::arch::aarch64` (NEON), dispatched at runtime through the `cpu.rs` feature
  detection that was deliberately kept when the assembly was deleted.
- **Every kernel keeps its scalar twin**, with a test asserting the two agree
  bit-exactly over randomized inputs. The scalar version is the reference and must
  never be deleted. Correctness here is byte-equality; a fast wrong answer is a
  regression.
- ARM kernels must be verified on real ARM hardware, not only cross-compiled.

**Exit:** each kernel is bit-exact against its twin, corpus is 45/45, and the
benchmark shows a measured per-kernel delta.

### Phase 3 — Delete the assembly — **DONE**

Done first, as this section recommended, because it was free.

**Result:** 115 files and ~244k lines removed; 8.6 MB of assembly source and
5.9 MB of `src/` gone; binary 4,455,936 → 3,347,968 bytes (−1.11 MB, −25%); the
`nasm` build dependency and the `asm*` features are gone, so the crate now
builds with cargo alone. Corpus 45/45 byte-identical and the test suite green at
every step of the removal.

One thing worth recording, because it is stronger than "unused": the MSAC
bindings were not merely unreachable, they were **semantically wrong for AV2**.
AV2 reworked the arithmetic coder's probability, adaptation, and bypass, so the
dav1d assembly implements the AV1 formula and would have produced incorrect
output. The bring-up comments said so explicitly at each dispatch site, and the
AV2 path calls zero of the entry points that still had a live asm branch.

Original scope, for reference:

- Delete `src/x86/`, `src/arm/`, the `.asm`/`.S` sources, and the `nasm-rs` build
  path from `build.rs`.
- Remove the `asm`, `asm_arm64_dotprod`, `asm_arm64_i8mm`, `asm_arm64_sve2`
  features. Keep `cpu.rs` feature detection — SIMD dispatch still needs it.
- Reduce `cpu.rs` to what SIMD dispatch actually consumes.
- Drop the `nasm` requirement from README and CI.
- Correct the README's claim that `nasm` enables "the assembly paths", and
  `STATUS.md`'s implication that the hardening guards are a significant cost.

**Exit:** corpus 45/45, no `nasm` anywhere in the build, binary ~1.1 MB smaller.

---

## 4. What we are *not* claiming

- **No speedup target.** We have a baseline and an Amdahl ceiling for the kernel
  work (~1.35× if kernels went to zero); we do not yet have an estimate for Phase 1,
  which is the larger and less predictable piece. Targets get set after Phase 0.
- **No comparison to `libavm`** until the benchmark is honest: matched clips,
  matched thread counts, CPU time, interleaved runs.
- **Phase 1 dominates.** If effort has to be cut, cut Phase 2 — not Phase 1. The
  measurements say the data layout is worth more than the vectorization, and it is
  the only part that is a prerequisite for anything else. The allocator result
  sharpens this: **1.38x from one line**, versus an Amdahl ceiling of ~1.35x for
  perfect SIMD on every kernel. Allocation and data layout are where this decoder
  actually is.

## 5. Open questions

- Does the padded recon-mirror buffer still need to exist post-bring-up, or is it
  scaffolding? Worth answering early — it may be free performance.
- AV2's subpel filter taps are **identical to AV1's** (verified by diffing the
  tables). So dav1d's `mc` kernels are algorithmically reusable even though we are
  discarding their assembly — useful as a reference for what the vectorized form
  should look like.
- Threading is under-tested and un-profiled (`STATUS.md` limitation 4). Frame- or
  tile-parallel decode may be worth more than any single-core work here, and should
  be scoped before Phase 2 rather than after.
