# World Mesh

A large-scale real-world terrain renderer built on [Bevy](https://bevy.org) and
a vendored fork of [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain).
Render a 22 km patch of Death Valley or the entire Earth from public elevation and
satellite imagery using the same pipeline.

![Death Valley](screenshots/world_mesh_death_valley.png)
![Earth from orbit](screenshots/world_mesh_globe.png)

The underlying renderer uses Uniform Distance-Dependent Level of Detail
(UDLOD) geometry and a Chunked Clipmap tile atlas, so it stays seamless whether
you are hugging the ground at 2 m/px or parked at three Earth radii.
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full walkthrough.

## Prerequisites

- **Recommended**: [Nix](https://nixos.org/download.html) with flakes enabled
  (+ optionally [`direnv`](https://direnv.net) for auto-activation).
  The dev shell pins stable Rust 1.91 from nixpkgs 25.11 and pulls in `clang`,
  `cmake`, `pkg-config`, and the platform GPU stack.
- **Manual alternative**: stable Rust 1.90+ (via `rustup`), plus `clang`,
  `cmake`, and `pkg-config`. On Linux you will also need Vulkan, `alsa-lib`,
  `udev`, `libxkbcommon`, Wayland, and the X11 libs; on macOS, the system
  toolchain is enough.
- ~10 GB free disk for the globe scene (7 GB tile cache + 3 GB atlas).
  The planar scenes need ~1 GB each.

## Quickstart

```bash
git clone <this-repo> world_mesh
cd world_mesh
```

### Enter the dev shell

```bash
direnv allow        # if you use direnv — picks up .envrc automatically
# or
nix develop         # drops you into the pinned dev shell manually
```

If you are going the manual Rust route, skip this step and just be sure
`rustc`, `cargo`, `clang`, `cmake`, and `pkg-config` are on your `PATH`.

### Render a scene

All three scenes fetch their source data from public DEM + satellite imagery
providers, build a `bevy_terrain` atlas, render the scene, and write a
screenshot under `screenshots/`. The tile cache at `target/tile_cache/` is
shared across runs, so subsequent renders of any region are near-instant.

| Scene                         | Command                                                          | Cold run  | Re-run |
| ----------------------------- | ---------------------------------------------------------------- | --------- | ------ |
| Death Valley (planar, 22 km)  | `./scripts/render_region.sh death_valley`                        | 5-10 min  | ~30 s  |
| Earth (spherical, fast)       | `./scripts/render_globe.sh --zoom 7 --face-size 2048`            | 15-30 min | ~1 min |
| Earth (spherical, hero)       | `./scripts/render_globe.sh`                                       | ~3 h      | ~5 min |

The hero globe defaults to `--zoom 8 --face-size 8192` and pulls ~4 GB of
source tiles; use the "fast" variant for a quick first look. Other planar
presets live in [`src/regions.rs`](src/regions.rs) — `brienz` is ready, and
adding another is a handful of lines in that file.

### Fly around interactively

Both render scripts exit after capturing the screenshot. To explore
interactively, run the binary directly:

```bash
cargo run --release                              # planar
cargo run --release --bin world_mesh_globe       # globe
```

Fly camera:

| Key                | Effect                                       |
| ------------------ | -------------------------------------------- |
| `W` / `S`          | forward / back                               |
| `A` / `D`          | strafe left / right                          |
| `Space` / `Ctrl`   | up / down                                    |
| `Q` / `E`          | roll left / right                            |
| mouse              | yaw / pitch                                  |
| `Shift`            | boost                                        |

Debug overlays (the `TerrainDebugPlugin` hotkeys, deconflicted off WASD/QE):

| Key       | Effect                               |
| --------- | ------------------------------------ |
| `R`       | wireframe                            |
| `T`       | lighting                             |
| `V`       | tile-tree LOD overlay                |
| `L` / `Y` | data-LOD / geometry-LOD overlay      |
| `M` / `K` | morphing / blending                  |
| `G`       | sample-grad texture filtering        |
| `B`       | world-normal visualisation           |
| `Z`       | tile-tree LOD colouring              |
| `P` / `U` | pixel / UV overlay                   |
| `F`       | freeze / unfreeze the view frustum   |
| `H`       | high-precision coordinates           |
| `N` / `C` | blend distance − / +                 |
| `I` / `O` | morph distance − / +                 |
| `X` / `J` | grid size − / +                      |
| `1`/`2`/`3` | debug flags 1 / 2 / 3              |

## How it works

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a walkthrough of the rendering
pipeline, data flow, and a pointer tour of the codebase.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). Portions
derived from [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain)
may alternatively be used under the upstream MIT terms; see [NOTICE](NOTICE)
for attribution and the full provenance chain.
