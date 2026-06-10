#!/usr/bin/env bash
# Build the web demo content: bake the DASH model to a PNG sequence the web
# player (web/index.html) flips through. The frames are real GPU renders from
# the native pipeline (offscreen), so the web demo is WebGPU-free and portable.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

FRAMES="${1:-24}"
# Evaluate the 4D deform path for all points; the smoke model is near-static in
# time, so the visible variation comes from the camera orbit.
export DASH_MASK_MODE="${DASH_MASK_MODE:-all}"

echo "baking $FRAMES frames into web/baked ..."
cargo run --release -p wgpu-gs-viewer --bin dash_bake -- \
  --frames "$FRAMES" --out web/baked --orbit 1.0
echo "web content ready: web/index.html + web/baked/"
