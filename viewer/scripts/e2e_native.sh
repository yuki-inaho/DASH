#!/usr/bin/env bash
# L3 native E2E: render the DASH model headlessly (sidecar -> dash-runtime ->
# GPU pipeline -> PNG) and assert the frames are non-blank and differ.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

OUT="e2e-out/native"
mkdir -p "$OUT"
rm -f "$OUT"/frame_*.png
export DASH_MASK_MODE=all

echo "== baking 8 frames (orbit) =="
if ! cargo run --release -p wgpu-gs-viewer --bin dash_bake -- \
      --frames 8 --out "$OUT" --orbit 1.0; then
  echo "E2E_NATIVE_FAIL: bake errored"
  exit 1
fi

COUNT=$(ls "$OUT"/frame_*.png 2>/dev/null | wc -l)
if [ "$COUNT" -lt 2 ]; then echo "E2E_NATIVE_FAIL: only $COUNT frames"; exit 1; fi

# Non-blank: a blank 1280x720 PNG compresses to a few KB; real renders are >20KB.
SMALL=$(find "$OUT" -name 'frame_*.png' -size -20k | wc -l)
if [ "$SMALL" -ne 0 ]; then echo "E2E_NATIVE_FAIL: $SMALL near-blank frames"; exit 1; fi

# Differ: distinct content hashes (camera orbit guarantees this).
DISTINCT=$(md5sum "$OUT"/frame_*.png | awk '{print $1}' | sort -u | wc -l)
if [ "$DISTINCT" -lt 2 ]; then echo "E2E_NATIVE_FAIL: frames identical"; exit 1; fi

echo "E2E_NATIVE_PASS frames=$COUNT distinct=$DISTINCT dir=$OUT"
