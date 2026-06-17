#!/usr/bin/env bash
#
# Optimize a glTF/GLB asset to reduce its Git LFS footprint while keeping it
# loadable by the Elodin editor's Bevy 0.18 glTF loader.
#
# Bevy 0.18 does NOT support KHR_draco_mesh_compression, EXT_meshopt_compression,
# KHR_mesh_quantization, KHR_texture_basisu, EXT_texture_webp, or
# EXT_mesh_gpu_instancing. So this script intentionally produces PLAIN glTF 2.0:
# it drops unused vertex attributes, welds/joins geometry, prunes unused data,
# and decimates triangle count (the only meaningful lever for uncompressed,
# texture-light CAD models). The output is verified to require no extensions.
#
# Usage:
#   scripts/optimize-glb.sh <input.glb> [keep-ratio] [output.glb]
#
#   keep-ratio  Fraction of triangles to KEEP (default 0.25).
#               0.5 = conservative, 0.25 = aggressive, 0.1 = very aggressive.
#   output.glb  Defaults to overwriting <input.glb> (the original is recoverable
#               via `git checkout -- <input.glb>` since assets are tracked in LFS).
#
# Requires: node + npx (uses @gltf-transform/cli on demand; no install needed).
set -euo pipefail

IN="${1:?usage: scripts/optimize-glb.sh <input.glb> [keep-ratio] [output.glb]}"
RATIO="${2:-0.25}"
OUT="${3:-$IN}"
ERROR="${GLB_SIMPLIFY_ERROR:-0.01}"   # max simplification error (fraction of size)

gt() { npx --yes @gltf-transform/cli@latest "$@"; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/glb-opt.XXXXXXXXXX")"
LOSSLESS="$WORK/lossless.glb"
RESULT="$WORK/result.glb"
trap 'rm -rf "$WORK"' EXIT

echo "==> Optimizing $IN  (keep ${RATIO} of triangles)"

echo "[1/3] lossless cleanup: drop unused UVs, weld, join, prune (no compression)"
gt optimize "$IN" "$LOSSLESS" \
  --compress false --texture-compress false \
  --instance false --palette false --simplify false

echo "[2/3] simplify to ${RATIO} of triangles (error<=${ERROR})"
gt simplify "$LOSSLESS" "$RESULT" --ratio "$RATIO" --error "$ERROR"

echo "[3/3] verify Bevy-safe (no required glTF extensions)"
REQ=$(node -e 'const fs=require("fs");const b=fs.readFileSync(process.argv[1]);const l=b.readUInt32LE(12);const j=JSON.parse(b.subarray(20,20+l).toString("utf8"));const u=j.extensionsUsed||[];const r=j.extensionsRequired||[];if(r.length){console.error("required: "+r.join(", "));process.exit(1)}console.error("used: "+(u.length?u.join(", "):"(none)"));process.stdout.write(String(r.length))' "$RESULT") || {
  echo "ABORT: output requires glTF extensions Bevy 0.18 cannot load." >&2; exit 1; }

before=$(wc -c < "$IN"); after=$(wc -c < "$RESULT")
mv "$RESULT" "$OUT"
awk -v b="$before" -v a="$after" -v o="$OUT" \
  'BEGIN{printf "==> Done: %.1f MB -> %.1f MB (%.0f%% smaller)  %s\n", b/1048576, a/1048576, (1-a/b)*100, o}'
echo "    Open the result in the editor or https://gltf.report to check visual quality before committing."
