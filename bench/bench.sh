#!/usr/bin/env bash
# Decode-speed benchmark. This is the number Phase 1/2 of docs/plan.md quote
# deltas against, so it is deliberately boring and repeatable.
#
#   bench.sh [clip.ivf ...]     # default: every clip in bench/conformance/corpus
#
# Env:
#   RUSTY_AV2D=<path>   decoder binary (default: target/release/dav1d)
#   RUNS=<n>            timed runs per clip after warm-up (default 7)
#   WARMUP=<n>          untimed warm-up runs (default 3)
#   BASELINE=<file>     compare against a previous --save file
#   SAVE=<file>         write results for a later --baseline comparison
#
# Reports the MEDIAN and the min-max spread per clip, never a single number:
# on a warm machine the spread here is ~2%, and a change smaller than that is
# not a result.
#
# NOTE: the binary is built with `rusty_alloc` as its global allocator (see
# tools/dav1d.rs). That is the shipped configuration and therefore the baseline
# -- measuring against the system heap would flatter every later change by the
# 1.38x the allocator already provides.
#
# Do NOT `cargo build` while this is running: on Windows the build cannot replace
# a binary that is executing, the run is left with zombie decoder processes, and
# the results are silently truncated rather than failing loudly.
#
# KNOWN ISSUE (Git Bash / MSYS on Windows): a long sweep can wedge partway
# through. Decoder processes are left in an unkillable state and every later
# spawn stalls; the same clips run fine individually and in small batches. The
# timing path was already de-forked (EPOCHREALTIME instead of `date`, which
# removed two spawns per decode) and that helped but did not eliminate it. If
# you hit it, run in batches of a few clips and merge the SAVE files, or run
# under WSL/Linux where process creation is cheap. `timeout` cannot reliably
# kill a native Windows process from MSYS bash, so the per-decode timeout is a
# partial guard only.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
ext=""; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ext=".exe";; esac
dec="${RUSTY_AV2D:-$root/target/release/dav1d$ext}"
runs="${RUNS:-7}"
warmup="${WARMUP:-3}"

if [ ! -x "$dec" ]; then
  echo "error: decoder not found at '$dec' -- run: cargo build --release" >&2
  exit 2
fi

if [ $# -gt 0 ]; then
  clips=("$@")
else
  clips=("$here"/conformance/corpus/*.ivf)
fi

work="${TMPDIR:-/tmp}/rusty_av2d_bench"
mkdir -p "$work"

# Milliseconds for one decode, or "" if it failed/hung.
#
# The decode is wrapped in `timeout`: a decoder that hangs must fail this run
# loudly rather than stall it forever. (Observed in Git Bash on Windows: an
# occasional spawned decode never returns even though the same command run
# directly completes in well under a second. Without the timeout the sweep just
# stops, and a truncated result set looks identical to a short one.)
# Milliseconds now, WITHOUT forking: `$(date ...)` costs two extra process
# spawns per timed decode, and Git Bash's fork emulation wedges under the
# hundreds of spawns a full sweep needs. EPOCHREALTIME is a bash 5 builtin.
now_ms() {
  local t=${EPOCHREALTIME}
  t=${t/,/.}          # comma decimal separator in some locales
  t=${t/./}           # -> microseconds
  echo $(( 10#$t / 1000 ))
}

one() {
  local s e rc
  s=$(now_ms)
  timeout "${DECODE_TIMEOUT:-60}" "$dec" -i "$1" -o "$work/out.yuv" >/dev/null 2>&1
  rc=$?
  e=$(now_ms)
  if [ "$rc" -ne 0 ]; then
    echo "  ! decode failed (rc=$rc) on $(basename "$1")" >&2
    echo ""
    return
  fi
  echo $(( e - s ))
}

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }

printf '%-28s %8s %8s %8s\n' clip median min max
printf '%-28s %8s %8s %8s\n' ---- ------ --- ---
total=0
failed=0
declare -A results
for clip in "${clips[@]}"; do
  [ -f "$clip" ] || continue
  name="$(basename "$clip" .ivf)"
  for _ in $(seq "$warmup"); do one "$clip" >/dev/null; done
  vals=()
  bad=0
  for _ in $(seq "$runs"); do
    v="$(one "$clip")"
    if [ -z "$v" ]; then bad=1; break; fi
    vals+=("$v")
  done
  if [ "$bad" -ne 0 ]; then
    printf '%-28s %8s
' "$name" "FAILED"
    failed=$(( failed + 1 ))
    continue
  fi
  med=$(median "${vals[@]}")
  mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
  mx=$(printf '%s\n' "${vals[@]}" | sort -n | tail -1)
  printf '%-28s %8s %8s %8s\n' "$name" "$med" "$mn" "$mx"
  results[$name]=$med
  total=$(( total + med ))
done
printf '%-28s %8s\n' TOTAL "$total"

if [ -n "${SAVE:-}" ]; then
  : > "$SAVE"
  for k in "${!results[@]}"; do echo "$k ${results[$k]}" >> "$SAVE"; done
  sort -o "$SAVE" "$SAVE"
  echo "saved -> $SAVE"
fi

if [ -n "${BASELINE:-}" ] && [ -f "$BASELINE" ]; then
  echo
  printf '%-28s %8s %8s %8s\n' clip base now delta
  printf '%-28s %8s %8s %8s\n' ---- ---- --- -----
  bt=0; nt=0
  while read -r k v; do
    now="${results[$k]:-}"
    [ -n "$now" ] || continue
    printf '%-28s %8s %8s %7s%%\n' "$k" "$v" "$now" \
      "$(awk -v b="$v" -v n="$now" 'BEGIN{printf "%+.1f", (n-b)*100.0/b}')"
    bt=$(( bt + v )); nt=$(( nt + now ))
  done < "$BASELINE"
  echo
  awk -v b="$bt" -v n="$nt" 'BEGIN{printf "TOTAL %d -> %d  (%+.1f%%, %.3fx)\n", b, n, (n-b)*100.0/b, b/n}'
fi
