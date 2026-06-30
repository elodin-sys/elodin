# DB-Centric Assets — Phase Checklist (RFD #724)

Branch: `feat/db-centric-assets` · PR #727

Make Elodin-DB the single, complete source of a simulation's assets: the DB
holds the active schematic *and* all of its assets (meshes, icons, window
sub-schematics, terrain). Sim and editor become pure network consumers of the
DB's HTTP asset server.

## Phases

- [x] **Phase 0 — Ingest foundation**: copy the whole `assets/` tree into
  `{db}/assets/` once (`ingest_asset_dir`, `ELODIN_ASSETS`, `--assets`,
  copy-once guard). No consumer changes.
- [x] **Phase 1 — Schematics over HTTP**: editor loads the active schematic
  (`schematic.active`), the library, and window sub-schematics entirely over
  HTTP; `__index__` listing endpoint; drop the window filesystem fallback.
- [x] **Phase 2 — Editor write-back & DB-native open**: `PUT` on the asset
  server (with a write gate); "Save" writes to the DB instead of disk; remove
  file dialogs and the `read_dir` picker. Named schematics: "Save Schematic
  As..." stores under `schematics/<name>.kdl` and "Save Schematic" overwrites
  the currently active key, so several schematics can coexist. "Open
  Schematic..." lists them via `GET /__index__/schematics/` and repoints
  `schematic.active` at the chosen one, so config sync loads it over HTTP and it
  stays the authoritative active schematic. The DB skybox mirror no longer
  fights a local clear: it re-asserts a synced skybox only for an external drift,
  not for a state the user just pushed locally while the `SetDbConfig` echo is in
  flight. (Renaming an existing schematic is deferred — it needs a server-side
  move/`DELETE` the asset server doesn't yet expose.)
- [ ] **Phase 3 — Terrain over HTTP**: ingest `assets/terrains/`; convert the
  `bevy_world_mesh` loader to lazy per-tile HTTP fetch; delete terrain
  filesystem reads.
- [ ] **Phase 4 — Consolidation & cleanup**: remove `ELODIN_ASSETS_DIR`,
  `ELODIN_KDL_DIR`, `schematic.content`/`schematic.path`; generalize follow-mode
  to a full-tree mirror via `__index__`. A single asset code path everywhere.
- [ ] **Phase 5 — Aleph & deploy story**: wire `ELODIN_ASSETS` into Aleph's
  `elodin-db` service (ingest on each fresh boot); make `betaflight-sitl` the
  reference SITL example. A HITL DB becomes a complete, portable record.
- [ ] **Phase 6 — Docs & tests**: rewrite `db-asset-server.md`, `schematic.md`,
  `elodin-cli.md` around the single namespace; tests (copy-once, traversal
  rejection, `__index__`, PUT round-trip, `Window` rewrite/collect, follow
  full-tree mirror, terrain HTTP fetch).

## Status

Phases 0–2 are landed on `feat/db-centric-assets`. Phases 3–6 remain, to be
added as further commits on the same branch (single PR for the whole change).
