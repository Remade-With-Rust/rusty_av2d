#!/usr/bin/env bash
# Gate the decoder against the corpus: for each stream, decode with the reference
# decoder (the oracle) and with rusty_av2d (mine), and cmp the two output YUVs.
#
#   run.sh [name-glob]      # no arg = every corpus stream
#
# The reference decoders are NOT vendored — build them from upstream and point
# this script at them:
#   AVMDEC=/path/to/avmdec   (AOM AVM reference decoder; the normative oracle)
#   DAV2D=/path/to/dav2d     (optional second oracle, for streams it supports)
#   RUSTY_AV2D=/path/to/our/decoder binary (default: ./target/release/dav1d)
#
# Env: ORACLE=avmdec  -> force the avm reference decoder as the oracle.
#      KEEP=1         -> keep the per-stream ref/mine YUVs for inspection.
# Exit status is non-zero if any stream FAILs (usable as a CI gate).
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
ext=""; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ext=".exe";; esac
dav2d="${DAV2D:-$root/av2_literature/repos/dav2d/build_debug/tools/dav2d$ext}"
avmdec="${AVMDEC:-$root/av2_literature/repos/avm/build_rs/avmdec$ext}"
rav2d="${RUSTY_AV2D:-$root/target/release/dav1d$ext}"
if [ ! -x "$avmdec" ]; then
  echo "error: AVM reference decoder not found at '$avmdec'" >&2
  echo "       set AVMDEC=/path/to/avmdec (see CONTRIBUTING.md)" >&2
  exit 2
fi
if [ ! -x "$rav2d" ]; then
  echo "error: rusty_av2d binary not found at '$rav2d' -- run: cargo build --release" >&2
  exit 2
fi
corpus="$here/corpus"
work="${TMPDIR:-/tmp}/rav2d_conf"; mkdir -p "$work"

decode_oracle() { # <ivf> <out.yuv>
  case "$(basename "$1")" in
    *_422*|*_444*|*_grain*|*_sframe*|*_scc*|*_off_*|*_qm*|*640x360*|*424x240*)
      # dav2d rejects profile 2/4 streams — avmdec is the only oracle for 4:2:2/4:4:4.
      "$avmdec" --rawvideo -o "$2" "$1" >/dev/null 2>&1
      return
      ;;
  esac
  if [ "${ORACLE:-dav2d}" = avmdec ]; then
    "$avmdec" --rawvideo --i420 -o "$2" "$1" >/dev/null 2>&1
  else
    "$dav2d" --demuxer ivf --threads 1 -i "$1" -o "$2" >/dev/null 2>&1
  fi
}

pat="${1:-*}"; pass=0; fail=0; skip=0
for ivf in "$corpus"/$pat.ivf; do
  [ -e "$ivf" ] || continue
  name="$(basename "$ivf" .ivf)"
  ref="$work/$name.ref.yuv"; out="$work/$name.mine.yuv"
  rm -f "$ref" "$out"
  decode_oracle "$ivf" "$ref"
  "$rav2d" -i "$ivf" -o "$out" --threads 1 >/dev/null 2>&1
  if [ ! -s "$ref" ]; then echo "SKIP $name (oracle produced no output)"; skip=$((skip+1)); continue; fi
  if [ -s "$out" ] && cmp -s "$ref" "$out"; then
    echo "PASS $name  ($(wc -c < "$ref") bytes)"; pass=$((pass+1))
  else
    rs=$(wc -c < "$ref" 2>/dev/null || echo 0)
    os=$([ -f "$out" ] && wc -c < "$out" || echo 0)
    diff=$(cmp "$ref" "$out" 2>&1 | head -1)
    echo "FAIL $name  (oracle=$rs mine=$os)  ${diff:-<no mine output>}"; fail=$((fail+1))
  fi
  [ "${KEEP:-0}" = 1 ] || rm -f "$ref" "$out"
done
echo "---- ${ORACLE:-dav2d} oracle: $pass passed, $fail failed, $skip skipped ----"
[ "$fail" -eq 0 ]
