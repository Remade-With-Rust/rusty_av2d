#!/usr/bin/env bash
# Mint an AV2 test stream into the corpus by encoding a raw I420 source with avmenc.
#
#   gen.sh <name> <source.yuv> <w> <h> <nframes> [extra avmenc flags...]
#
# Env: QP (default 128, range 0..255), CPU (avmenc --cpu-used, default 8),
#      KFMAX (--kf-max-dist, default = nframes so only frame 0 is a keyframe).
# The source may be generated with mksrc.py. Each stream is a permanent test vector;
# the raw source is transient (only needed to mint the stream).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
avmenc="$root/av2_literature/repos/avm/build_rs/avmenc.exe"
corpus="$here/corpus"
mkdir -p "$corpus"

[ $# -ge 5 ] || { echo "usage: gen.sh <name> <source.yuv> <w> <h> <nframes> [avmenc flags...]" >&2; exit 2; }
name="$1"; src="$2"; w="$3"; h="$4"; n="$5"; shift 5
qp="${QP:-128}"; cpu="${CPU:-8}"; kfmax="${KFMAX:-$n}"

"$avmenc" --codec=av2 -w "$w" -h "$h" --limit="$n" --ivf \
  --cpu-used="$cpu" --end-usage=q --qp="$qp" --kf-max-dist="$kfmax" "$@" \
  -o "$corpus/$name.ivf" "$src" 2>&1 | tail -1

sz=$(wc -c < "$corpus/$name.ivf")
echo "minted corpus/$name.ivf ($sz bytes; ${n}f ${w}x${h} qp=$qp $*)"
# Append to the manifest (one line per stream: name | dims | frames | flags).
printf '%-24s %5sx%-4s %3sf  qp=%-4s %s\n' "$name" "$w" "$h" "$n" "$qp" "$*" >> "$corpus/manifest.txt"
