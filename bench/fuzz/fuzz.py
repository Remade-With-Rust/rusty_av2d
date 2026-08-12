#!/usr/bin/env python3
"""Corrupt-stream fuzz harness for rusty_av2d.

Recreation of the harness that drove the corrupt-stream crash rate 39% -> ~1%
(the original lived in a session scratchpad and was lost; docs/wiring.md 4.1).

Two input distributions, BOTH required -- they historically found DISJOINT bugs:
  * mutate : take a real corpus clip, flip/overwrite a few random bytes.
             Exercises deep decode paths with mostly-valid framing.
  * random : fully random bytes (sometimes behind a valid IVF header).
             Exercises OBU/header validation. The tile-size-prefix underflow
             was found by THIS distribution after `mutate` ran clean.

A case FAILS only on a crash signature: an abnormal Windows/POSIX exit
(access violation etc.) or a hang past the per-case timeout. Ordinary nonzero
exits are the decoder correctly refusing garbage. Failing inputs are saved to
bench/fuzz/crashes/ for replay:

    target/release/dav1d -i bench/fuzz/crashes/<case>.ivf -o out.yuv --threads 1

Usage:
    python bench/fuzz/fuzz.py                 # smoke tier: 200 cases
    python bench/fuzz/fuzz.py -n 2000        # a campaign
    python bench/fuzz/fuzz.py --seed 7       # reproducible
    FUZZ=1 bash bench/conformance/run.sh     # (run.sh may invoke the smoke tier)

Note: this SAMPLES the tail, it does not prove it empty -- a clean 200-round has
been followed by 5 crashes in a 750-case campaign before. Structural closure is
checked accessors + deny(indexing_slicing) (README "Robustness").
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bench" / "conformance" / "corpus"
CRASHES = Path(__file__).resolve().parent / "crashes"

# Abnormal-exit floor: on Windows a crash surfaces as an NTSTATUS-style code
# (0xC0000005 access violation, 0xC0000409 stack overflow / fail-fast, ...);
# on POSIX subprocess reports signals as negative returncodes.
WIN_CRASH_FLOOR = 0x80000000


def is_crash(returncode: int) -> bool:
    if returncode is None:
        return False
    if returncode < 0:  # POSIX signal
        return True
    return returncode >= WIN_CRASH_FLOOR


def ivf_header(width=432, height=240, n_frames=1) -> bytes:
    h = bytearray()
    h += b"DKIF"
    h += (0).to_bytes(2, "little")          # version
    h += (32).to_bytes(2, "little")         # header size
    h += b"AV02"
    h += width.to_bytes(2, "little")
    h += height.to_bytes(2, "little")
    h += (25).to_bytes(4, "little")         # timebase den
    h += (1).to_bytes(4, "little")          # timebase num
    h += n_frames.to_bytes(4, "little")
    h += (0).to_bytes(4, "little")
    return bytes(h)


def gen_mutate(rng: random.Random, clips) -> bytes:
    data = bytearray(rng.choice(clips).read_bytes())
    n = rng.choice([1, 1, 2, 3, 8, 32])
    for _ in range(n):
        i = rng.randrange(len(data))
        op = rng.randrange(3)
        if op == 0:
            data[i] ^= 1 << rng.randrange(8)          # bit flip
        elif op == 1:
            data[i] = rng.randrange(256)              # byte overwrite
        else:                                          # truncate tail
            del data[i:]
            break
    return bytes(data)


def gen_random(rng: random.Random) -> bytes:
    body = bytes(rng.randrange(256) for _ in range(rng.randrange(16, 4096)))
    if rng.random() < 0.5:
        # valid IVF wrapper, garbage payload (reaches the OBU parser)
        return ivf_header() + len(body).to_bytes(4, "little") + (0).to_bytes(8, "little") + body
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds per case")
    ap.add_argument("--decoder", default=str(ROOT / "target" / "release" / ("dav1d.exe" if os.name == "nt" else "dav1d")))
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1 << 32)
    rng = random.Random(seed)
    clips = sorted(CORPUS.glob("*.ivf"))
    if not clips:
        print(f"error: no corpus clips under {CORPUS}", file=sys.stderr)
        return 2
    if not Path(args.decoder).exists():
        print(f"error: decoder not found at {args.decoder} -- cargo build --release first", file=sys.stderr)
        return 2

    CRASHES.mkdir(exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="av2d_fuzz_"))
    crashes = hangs = 0
    t0 = time.time()
    print(f"fuzz: {args.cases} cases, seed={seed}, decoder={args.decoder}")

    for i in range(args.cases):
        # Alternate distributions strictly so a small -n still runs both.
        kind = "mutate" if i % 2 == 0 else "random"
        payload = gen_mutate(rng, clips) if kind == "mutate" else gen_random(rng)
        case = tmpdir / f"case{i}.ivf"
        case.write_bytes(payload)
        out = tmpdir / f"case{i}.yuv"   # real file: `-o /dev/null` kills the CLI on Windows
        try:
            r = subprocess.run(
                [args.decoder, "-i", str(case), "-o", str(out), "--threads", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=args.timeout,
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = None
            hangs += 1
            kept = CRASHES / f"hang_seed{seed}_case{i}_{kind}.ivf"
            case.replace(kept)
            print(f"  HANG  case {i} ({kind}) -> {kept.name}")
            continue
        if is_crash(rc):
            crashes += 1
            kept = CRASHES / f"crash_seed{seed}_case{i}_{kind}_rc{rc & 0xFFFFFFFF:08x}.ivf"
            case.replace(kept)
            print(f"  CRASH case {i} ({kind}) rc={rc & 0xFFFFFFFF:#010x} -> {kept.name}")
        else:
            case.unlink(missing_ok=True)
        out.unlink(missing_ok=True)

    dt = time.time() - t0
    print(f"done: {args.cases} cases in {dt:.0f}s -- {crashes} crashes, {hangs} hangs (repros in {CRASHES})")
    return 1 if (crashes or hangs) else 0


if __name__ == "__main__":
    sys.exit(main())
