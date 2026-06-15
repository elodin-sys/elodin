---
name: gltf-asset-optimization
description: Reduce the size of glTF/GLB 3D assets to cut Git LFS bandwidth/storage while keeping them loadable by the editor's Bevy 0.18 glTF loader. Use when shrinking .glb/.gltf files in assets/, addressing LFS quota/bandwidth, or when a model is too large. Produces plain glTF 2.0 (no Draco/meshopt/quantization, which Bevy cannot read).
---

# glTF/GLB Asset Optimization

Large `.glb`/`.ktx2` assets in `assets/` are tracked by Git LFS. Every clone pulls
the LFS objects at `HEAD`, and GitHub bills that bandwidth to the repo owner
(including forks and anonymous clones). Shrinking the assets that ship at `HEAD`
is the most direct way to lower the recurring bill.

## Hard constraint: stay Bevy-0.18-safe

The editor loads models through Bevy 0.18's glTF loader, which does **not** support
these extensions (see https://docs.rs/bevy_gltf/0.18.0/bevy_gltf/):

- `KHR_draco_mesh_compression`, `EXT_meshopt_compression`, `KHR_mesh_quantization`
- `KHR_texture_basisu`, `EXT_texture_webp`, `EXT_mesh_gpu_instancing`

So the usual 5–10× geometry-compression tools (`gltfpack`, `gltf-transform optimize`
defaults, Draco) produce files that **fail to render in the editor**. Output must be
**plain glTF 2.0 core**. Always verify `extensionsRequired` is empty before committing.

## Step 1: Diagnose where the bytes are

```bash
npx @gltf-transform/cli inspect path/to/model.glb
```

- **Geometry-dominated** (many meshes, large vertex/index counts, few/no textures):
  the only Bevy-safe lever is reducing triangle count + dropping unused attributes.
  Common in CAD exports (over-tessellated, `f32` positions, unused `TEXCOORD_0`).
- **Texture-dominated** (large embedded PNG/JPEG): resize/re-encode textures, but keep
  them as PNG/JPEG (Bevy can't read basisu/webp here).

## Step 2: Optimize (use the bundled script)

```bash
# keep-ratio = fraction of triangles to KEEP (default 0.25)
scripts/optimize-glb.sh assets/<model>.glb 0.25
```

The script runs a lossless cleanup (drop unused UVs, weld, join, prune; no
compression/instancing) then decimates triangles, and verifies the result needs no
extensions. By default it overwrites the input; the original is recoverable with
`git checkout -- assets/<model>.glb` (LFS-tracked).

Equivalent manual pipeline:

```bash
npx @gltf-transform/cli optimize in.glb tmp.glb \
  --compress false --texture-compress false --instance false --palette false --simplify false
npx @gltf-transform/cli simplify tmp.glb out.glb --ratio 0.25 --error 0.01
```

## Choosing a ratio

Size scales ~linearly with triangle ratio for uncompressed geometry. Reference
numbers from a 119 MB CAD drone (`edu-450-v2-drone.glb`, geometry-only):

| keep-ratio | result | reduction | use when |
| --- | --- | --- | --- |
| lossless only | 105 MB | 12% | never decimate (hero asset, close-ups) |
| 0.5 | 57 MB | 52% | curved surfaces, mid-distance |
| 0.25 | 32 MB | 73% | good default for viewport props |
| 0.1 | 15 MB | 87% | small-in-viewport, aggressive |

Tune with: lower `--ratio` for more reduction; if curved surfaces look faceted, raise
it. If gaps open between separate parts, add `--lock-border true` to the simplify step
(reduces achievable savings on multi-shell models). `GLB_SIMPLIFY_ERROR` env var caps
the simplification error (default `0.01`).

## Step 3: Verify before committing

1. Confirm Bevy-safe: the script aborts unless `extensionsRequired` is empty.
2. Visual check: open the output in the editor (drag into a schematic) or
   https://gltf.report. Watch for faceting on curved surfaces and gaps between parts.
3. Do **not** commit (the developer commits). Note the asset is LFS-tracked.

## Prefer not to ship it at all

If an asset is optional or only used by one example, consider removing it from LFS and
loading it on demand (the editor supports remote `glb path="https://…"` via the
`web_asset` plugin). Eliminating a `HEAD` asset saves its full size on every clone.
Note: a dead/remote URL must point at a reachable host — otherwise use a local path.
