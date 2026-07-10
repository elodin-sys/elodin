+++
title = "DB Asset Server"
description = "Persist schematic assets (GLB, icons, skyboxes) inside an Elodin DB and serve them over HTTP for portable replay and follow"
draft = false
weight = 103
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
lead = "Copy schematic files into the database at record time, rewrite KDL paths to db:…, and serve them on port N+1 while elodin-db runs."
+++

Record a simulation once, then replay or share the database directory without shipping a separate `assets/` tree. Meshes, custom `.png` icons, and skyboxes referenced by the schematic are copied into the DB, rewritten in stored KDL as `db:…` paths, and served over HTTP while the database runs.

This complements [Replays](/reference/replays) and [Schematic KDL](/reference/schematic): telemetry lives in the DB layout you already use; this page covers **files** the schematic needs at visualization time.

## Architecture

```mermaid
flowchart LR
  subgraph record_phase ["Record"]
    Sim["Simulation"]
    Parse["Parse schematic KDL"]
    Copy["Copy bytes to db/assets"]
    Rewrite["Rewrite paths to db scheme"]
    Meta["Set schematic.active"]
  end
  subgraph serve_phase ["Serve"]
    TCP["Impeller TCP port N"]
    HTTP["DB Asset Server port N+1"]
  end
  subgraph consume_phase ["Consume"]
    Editor["Elodin Editor"]
    Follow["Follower asset sync"]
  end
  Sim --> Parse --> Copy --> Rewrite --> Meta
  Meta --> TCP
  Copy --> HTTP
  HTTP --> Editor
  HTTP --> Follow
```

### Layers

| Layer | Role |
|-------|------|
| **Blob store** | `{db_path}/assets/{relative_key}` — opaque bytes, any file type |
| **DB Asset Server** | `GET http://host:(tcp_port+1)/{relative_key}` while `elodin-db run` is active |
| **KDL rewrite** | Local paths at record time become `db:{relative_key}` in the stored active schematic asset |
| **Consumers** | Editor loads the active schematic and all assets over HTTP |

### Ports

- **Impeller TCP** on port `N` (default `2240`)
- **DB Asset Server** on port `N + 1` (default `2241`) — `ASSETS_HTTP_PORT_OFFSET = 1`

Do not run a follower Impeller server on `N+1`; that port is reserved for the DB Asset Server on the source.

### The `db:` scheme

At record time, a local schematic path such as `models/jet.glb` is stored as `db:models/jet.glb`. The file lands at `{db}/assets/models/jet.glb`.

Paths are **not** rewritten when they are already:

- `db:…`
- `http://…` or `https://…`
- `icon builtin=…` (no file; see below)

Relative keys must not contain `..` (path traversal is rejected).

## When assets are persisted

Persistence runs during simulation DB initialization (`init_db`), when the world has a schematic:

- **`db_path`** argument to `world.run(…, db_path=…)`, or
- **`ELODIN_DB_PATH`** environment variable (Python SDK)

The active schematic is stored as an asset (default `schematics/main.kdl`) and pointed to by `schematic.active` metadata. Consumers fetch it over the DB Asset Server; there is no inline KDL mirror in DB metadata.

## Simulation source snapshot

When a Python simulation records to an explicit database path, Elodin also writes a source snapshot to `{db}/simulation_source/`:

```text
{db}/simulation_source/manifest.json
{db}/simulation_source/files/...
```

This snapshot is for analysis and provenance. It contains the resolved project-local `.py` files that were imported after the simulation starts; it is not a compiled simulation, Python environment, dependency lockfile, or asset bundle.

The first-pass selection rule is:

- inspect `sys.modules`
- keep modules whose `__file__` resolves to a `.py` file under the simulation entrypoint directory
- exclude virtualenvs, `site-packages` / `dist-packages`, stdlib paths, `__pycache__`, cache/generated files, and non-Python assets

For example, the `examples/rc-jet` snapshot would typically include:

```text
examples/rc-jet/main.py
examples/rc-jet/config.py
examples/rc-jet/sim.py
examples/rc-jet/aero.py
examples/rc-jet/propulsion.py
examples/rc-jet/actuators.py
examples/rc-jet/ground.py
```

Visual assets remain separate under `{db}/assets/`.

## Consumption workflows

| Mode | Typical commands | Assets |
|------|------------------|--------|
| **Live** | `elodin editor examples/foo/main.py` | HTTP from the sim DB during the session |
| **Recorded DB** | `elodin-db run 127.0.0.1:2240 ./my-db` then `elodin editor 127.0.0.1:2240` | HTTP from `./my-db/assets/` |
| **Replay presentation** | Same DB + `elodin editor 127.0.0.1:2240 --replay` | HTTP; editor timeline reveals progressively |
| **Follow** | `elodin-db run … --follows SOURCE:2240` | Follower copies `db:` assets from source HTTP into its own `{db}/assets/` |

See [Elodin CLI](/reference/elodin-cli) for `editor --replay` and `elodin-db --follows`.

### Aleph / HITL

On the Aleph flight computer, the `elodin-db` NixOS service passes
`--assets /var/lib/elodin/assets` (the shared `ELODIN_ASSETS` root,
configurable via `services.elodin-db.assetsDir`) so each fresh boot database
ingests the asset tree on creation. A HITL recording copied off the vehicle is
therefore a complete, portable record: serve it anywhere with `elodin-db run`
and the editor loads schematic and meshes from the DB itself.

### Follow mode and assets

When a follower connects to a source `elodin-db` on port `N`, it replicates telemetry over Impeller TCP. Schematic assets are **not** streamed on that socket — they are fetched separately from the source **DB Asset Server** on port `N+1`.

On connect (and when DB config updates), the follower:

1. `GET`s `http://source:(N+1)/__index__` to list the source asset tree
2. Copies each key that is missing or size-different into `{follower_db}/assets/`
3. Serves the mirrored tree from its own DB Asset Server on `(follower_port + 1)`

The follower can then serve those files from its own DB Asset Server on `(follower_port + 1)` to local editors, without copying the full source database directory.

```sh
# Source (sim or recorded DB)
elodin-db run 127.0.0.1:2240 ./source-db

# Follower — Impeller on 2242, DB Asset Server on 2243
elodin-db run 127.0.0.1:2242 ./follower-db --follows 127.0.0.1:2240

# Editor attaches to the follower
elodin editor 127.0.0.1:2242
```

Point `--follows` at the source **Impeller** port (`N`), not the asset port (`N+1`).

## Supported asset types

### GLB meshes (`object_3d` → `glb path=…`)

| Stage | Behavior |
|-------|----------|
| **Local path** | Resolved via `$ELODIN_ASSETS` (default `./assets`), simulation entry directory, or cwd |
| **On disk in DB** | `assets/{key}.glb` (key preserves subdirectories, e.g. `models/rocket.glb`) |
| **Stored KDL** | `path="db:models/rocket.glb"` |
| **Editor** | `db:…` → `http://127.0.0.1:2241/…` via Bevy `AssetServer` + `WebAssetPlugin` (non-blocking) |

### `.png` icons (`object_3d` → `icon path=…`)

Same pipeline as GLB: persist, rewrite, HTTP load.

**`icon builtin=…`** (Material Icons) is **not** copied into the DB. The editor rasterizes built-in glyphs locally; only custom `path=` `.png` files are persisted in the DB.

### Skybox (`skybox name="…"`)

Skyboxes are **indirect**: the KDL node names a manifest entry, not a single file path.

| Stage | Behavior |
|-------|----------|
| **Local files** | `assets/skyboxes/manifest.ron` plus the active entry's `*.cubemap.ktx2` |
| **On disk in DB** | `assets/skyboxes/manifest.ron` + `assets/skyboxes/{name}.cubemap.ktx2` |
| **Stored KDL** | `skybox name="desert_night"` unchanged (name is logical) |
| **Editor** | Async download from DB HTTP into the local skybox cache, then `SetActiveSkybox` |
| **Clear** | Empty `skybox.active` metadata or schematic without a `skybox` node → skybox cleared in the editor |

The skybox plugin today reads from a **local cache directory**; the editor mirrors DB assets there before activation. GLB meshes and `.png` icons use HTTP directly through Bevy.

### Carried in the DB without extra asset files

- Active schematic KDL at `schematics/*.kdl` (see `schematic.active`)
- Procedural meshes (`sphere`, `box`, `cylinder`, …)
- Built-in [color scheme](/reference/color-schemes) names in `theme scheme=…`
- Telemetry components (separate from this asset pipeline)

### Not supported today

| Item | Why |
|------|-----|
| Custom `color_schemes/*.json` on disk | Only the scheme **name** is in KDL; JSON must exist locally |
| `window path="other.kdl"` | Window sub-schematics are stored as separate assets under `schematics/` |
| `video_stream` panels | H.264 lives in message logs, not `assets/` |
| Arbitrary external URLs in KDL | Not copied into the DB by design |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ELODIN_DB_PATH` | Directory for the simulation database (record) |
| `ELODIN_ASSETS` | Root for resolving local asset paths at record (default `./assets`) |

## Verification

With `elodin-db run` listening on `2240`:

```bash
export DB_PATH=./my-db
curl -sf -o /dev/null -w "%{http_code}\n" http://127.0.0.1:2241/f22.glb
ls -lh "$DB_PATH/assets/"
```

Expect HTTP `200` and non-empty files under `assets/` after a sim that references those assets.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Icon only, no mesh | `curl http://127.0.0.1:2241/…` → `404` means assets were not persisted; set `ELODIN_DB_PATH` / `db_path` and re-run |
| Port in use | `2240` = Impeller, `2241` = DB Asset Server; do not bind follower Impeller on `2241` |
| Skybox missing on replay | `ls "$DB_PATH/assets/skyboxes/"` for `manifest.ron` and the `.ktx2` |
| Empty `assets/` after sim | `world.run` without `db_path` or `ELODIN_DB_PATH` uses a temp DB |
| Follower missing meshes | Confirm source DB Asset Server responds: `curl http://SOURCE:2241/models/rocket.glb` |

## Adding a new asset type

### A — Simple file referenced by `path=` in KDL

Use this when the schematic stores a **direct relative path** (like a `.glb` mesh or `.png` icon).

1. **KDL** (`libs/impeller2/kdl`) — parse and serialize the path field on the relevant node.
2. **Collect & rewrite** (`libs/impeller2/kdl/src/rewrite.rs`):
   - `collect_local_asset_paths` — include new local paths
   - `collect_db_asset_names` — include `db:` keys (for [follow](/reference/elodin-cli) sync)
   - `rewrite_asset_paths` — rewrite local → `db:…` on record
3. **Persist** — no change if the path appears in collect; `persist_schematic_assets` in `libs/nox-py/src/impeller2_server.rs` is generic.
4. **Follow** — no change if the path appears in `collect_db_asset_names`; full-tree mirror via `GET /__index__` copies all assets.
5. **Editor** — if Bevy can load the format: resolve with `resolve_db_asset_url` and `AssetServer.load(url)`. No blocking HTTP in Bevy systems.
6. **Tests** — unit tests in `impeller2-kdl` (collect/rewrite) and `nox-py` (persist).

### B — Indirect reference (name → manifest → file)

Use this when the KDL stores a **logical name** (like `skybox name=…`).

1. Add a **single resolver** in `impeller2/kdl` (manifest parse → extra storage keys).
2. **Persist** — extend collect after resolving (see `add_local_skybox_cubemap_path` in `impeller2_server.rs`).
3. **Follow** — after syncing the manifest bytes, resolve and fetch dependent files (`assets_http.rs`).
4. **Editor** — either teach the consumer to load via HTTP, or mirror into a local cache (skybox pattern, async via `IoTaskPool`).
5. **Avoid** copying manifest-resolution logic into three places; share one resolver.

## Related docs

- [Schematic KDL](/reference/schematic) — `object_3d`, `icon`, `skybox` syntax
- [Replays](/reference/replays) — legacy replay directory layout (distinct from Elodin DB with `assets/`)
- [Elodin DB overview](/home/db/overview) — database capabilities and follow mode
- [Elodin CLI](/reference/elodin-cli) — `editor`, `elodin-db run`, `--replay`, `--follows`

<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

  mermaid.initialize({
    startOnLoad: false,
    theme: document.body.classList.contains("dark") ? "dark" : "default",
    securityLevel: "loose",
  });

  const nodes = [];
  document.querySelectorAll("pre > code.language-mermaid").forEach((code) => {
    const div = document.createElement("div");
    div.className = "mermaid";
    div.style.margin = "2rem 0";
    div.style.overflowX = "auto";
    div.style.textAlign = "center";
    div.textContent = code.textContent;
    code.parentElement.replaceWith(div);
    nodes.push(div);
  });

  if (nodes.length) {
    await mermaid.run({ nodes });
  }
</script>
