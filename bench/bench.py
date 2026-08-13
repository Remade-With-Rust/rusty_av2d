#!/usr/bin/env python3
"""Repeatable decode benchmark (docs/plan.md Phase 0).

The bash sweep (`bench.sh`) wedges under Git Bash/MSYS on long runs (fork storms
leave unkillable decoder processes). Python spawning does not, so this is the
driver of record on Windows; `bench.sh` remains for Unix shells.

    python bench/bench.py                      # time the default clip set
    python bench/bench.py --save baseline      # record bench/baselines/baseline.json
    python bench/bench.py --baseline baseline  # compare against a recording
    python bench/bench.py --clips v640x360 v432_16f_tipout --reps 9

Reports the median of N reps (default 7) with min-max spread per clip. Warm-run
spread is ~2%; below that a delta is noise, not a result. Times are wall-clock
around the whole process (spawn overhead is identical across arms and the
decode dominates).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "bench" / "conformance" / "corpus"
BASELINES = ROOT / "bench" / "baselines"
DECODER = ROOT / "target" / "release" / ("dav1d.exe" if os.name == "nt" else "dav1d")

# The default set: the largest / most work-heavy clips, one per major shape.
DEFAULT_CLIPS = [
    "v640x360",            # largest frame size
    "v432_16f_tipout",     # longest GOP (16f, TIP)
    "v432_8f_sb128cdef",   # 128px SBs + CDEF
    "v320x480_still_cpu3", # real photo, chroma LR + everything intra
    "v432_8f_lr",          # loop restoration
    "v432_4f_grain",       # film grain synthesis
]


def time_once(clip: Path, out: Path) -> float:
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(DECODER), "-i", str(clip), "-o", str(out), "--threads", "1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
    )
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"error: decode failed on {clip.name} (rc={r.returncode})", file=sys.stderr)
        sys.exit(2)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--save", help="record results to bench/baselines/<name>.json")
    ap.add_argument("--baseline", help="compare against bench/baselines/<name>.json")
    args = ap.parse_args()

    if not DECODER.exists():
        print("error: build first: cargo build --release", file=sys.stderr)
        return 2

    base = None
    if args.baseline:
        base = json.loads((BASELINES / f"{args.baseline}.json").read_text())

    tmp = Path(tempfile.mkdtemp(prefix="av2d_bench_"))
    results = {}
    print(f"{'clip':<22} {'median ms':>10} {'min':>8} {'max':>8}"
          + (f" {'baseline':>10} {'delta':>8}" if base else ""))
    for name in args.clips:
        clip = CORPUS / f"{name}.ivf"
        if not clip.exists():
            print(f"skip {name} (not in corpus)")
            continue
        out = tmp / f"{name}.yuv"
        time_once(clip, out)  # warmup (page cache, JIT-ish effects)
        times = [time_once(clip, out) * 1000 for _ in range(args.reps)]
        med = statistics.median(times)
        results[name] = {"median_ms": round(med, 1),
                         "min_ms": round(min(times), 1),
                         "max_ms": round(max(times), 1),
                         "reps": args.reps}
        line = f"{name:<22} {med:>10.1f} {min(times):>8.1f} {max(times):>8.1f}"
        if base and name in base.get("clips", {}):
            b = base["clips"][name]["median_ms"]
            line += f" {b:>10.1f} {100*(med-b)/b:>+7.1f}%"
        print(line, flush=True)
        out.unlink(missing_ok=True)

    if args.save:
        BASELINES.mkdir(exist_ok=True)
        payload = {
            "note": "wall ms per whole-process decode, --threads 1, median of reps (1 warmup)",
            "clips": results,
        }
        path = BASELINES / f"{args.save}.json"
        path.write_text(json.dumps(payload, indent=1))
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
