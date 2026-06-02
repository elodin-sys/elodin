# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — single-crate refactor

- Collapsed the `crates/bevy_terrain/` + `crates/app_plugin/` workspace
  into a single `world_mesh` crate. The former workspace members now live
  under `src/terrain/` and `src/scenes/` respectively.
- Feature-gated the publishable API surface:
    - `terrain` (default) — core renderer (`TerrainPlugin`,
      `TerrainBundle`, `TerrainMaterialPlugin<M>`, atlas/tile-tree
      internals, `DebugTerrainMaterial`, `EnvScreenshotPlugin`, TIFF
      loader).
    - `high_precision` — `big_space` integration (required for
      Earth-radius scenes).
    - `fetch` — HTTP slippy-tile fetcher + on-disk tile cache + AWS
      Terrain Tiles (terrarium) + EOX Sentinel-2 Cloudless decoders.
      Pulls in `ureq`.
    - `regions` — `Region` presets (Brienz, Death Valley) + the
      `RegionManifest` on-disk format. Pulls in `toml` + `serde`.
    - `scenes` — end-to-end scene plugins (`PlanarScenePlugin`,
      `GlobeScenePlugin`) with the app scaffolding. Depends on
      `regions` + `high_precision`.
    - `synth` — FBM procedural noise generators used by the
      `synthesize_*` binaries. Pulls in `noise`.
- Renamed the scene plugins: `AppPlugin` -> `PlanarScenePlugin`,
  `GlobeAppPlugin` -> `GlobeScenePlugin`.
- Moved `RegionManifest` from `scenes::planar` into the `regions` module
  so the dependency direction matches the feature-flag hierarchy
  (`scenes` depends on `regions`, not vice versa).
- Reduced `sample_height` visibility from `pub` to `pub(crate)` (it has
  no external users).
- Consolidated `assets/shaders/` — the upstream example shaders
  (`planar.wgsl`, `spherical.wgsl`) now live alongside
  `world_mesh.wgsl` under the top-level `assets/shaders/`.
- Embedded shader URLs moved from `embedded://bevy_terrain/shaders/...`
  to `embedded://world_mesh/terrain/shaders/...` to match the new crate
  name + module layout.

### Added

- Workspace-root `LICENSE` file (Apache-2.0 only, single file matching
  the [elodin-sys](https://raw.githubusercontent.com/elodin-sys/elodin/main/LICENSE)
  layout). The vendored `bevy_terrain` MIT terms are preserved via a
  new `NOTICE` file at the workspace root.
- `ARCHITECTURE.md` — a walkthrough of the rendering pipeline, data
  flow, and a pointer tour of the codebase.
- `.cursor/rules/no-git-commits.mdc` — prevents the AI agent from
  running `git commit`/`push`/`tag`/`amend` operations, leaving history
  management to the human.

### Removed

- `LICENSE-MIT` + `LICENSE-APACHE` at the workspace root (replaced by
  the single `LICENSE` + `NOTICE` above).
- `crates/bevy_terrain/LICENSE-*` + `crates/bevy_terrain/README.md`
  (collapsed into the top-level `LICENSE` + `NOTICE` + `README.md`).
- `scripts/phase_run.sh` (superseded by `render_region.sh`).

## Pre-release history

The project started as a revival of
[`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain)
(Bevy 0.9) and was incrementally ported up through Bevy 0.14, 0.15,
0.16, 0.17, and 0.18 along with adjoining dependency bumps
(`big_space` 0.10 → 0.11 → 0.12, `wgpu` 23 → 24 → 25, `encase` 0.11 →
0.12). The full migration record lives in the un-tracked
`context/MIGRATION_NOTES_0.16_AND_BEYOND.md`. Highlights:

- **Bevy 0.16**: `MaterialBindGroupAllocator`/`ErasedRenderAssets`
  overhaul; custom lighting in `fragment.wgsl` to avoid the
  `bevy_pbr::pbr_bindings` `@group(2)` collision; `wgpu` 24 `TexelCopy*`
  renames.
- **Bevy 0.17**: Composable Specialization (replaces
  `SpecializedRenderPipeline<P>`); `PreparedMaterial` / `RenderMaterialInstances`
  de-genericized; `MessageWriter`/`MessageReader` renames; `RenderSet`
  -> `RenderSystems`; `big_space::GridCell` -> `CellCoord`.
- **Bevy 0.18**: `RenderPipelineDescriptor::layout` switched from
  `Vec<BindGroupLayout>` to `Vec<BindGroupLayoutDescriptor>` (PR#21205);
  Material trait methods `enable_prepass()` / `enable_shadows()`
  replace `MaterialPlugin` fields (PR#20999); `GlobalAmbientLight`
  replaces `AmbientLight` resource (PR#21585); `AssetLoader` supertrait
  `TypePath` (PR#21339); encase 0.12 `#[shader(size(N))]`; `big_space`
  0.12; `vertex.wgsl` migrated off deprecated
  `bevy_pbr::view_transformations` to `bevy_render::view`.
