# Live Python schematic authoring (Phases 1–3)

Edit a Python schematic and see the Elodin Editor hot-reload it.

## Setup (nix develop)

```bash
nix develop
just install py
just install editor   # installs `elodin` to ~/.cargo/bin
export PATH="$HOME/.cargo/bin:$PATH"
# CLI spawns Python — point it at the venv if needed:
export ELODIN_PYTHON="$(pwd)/.venv/bin/python"
```

## Two terminals (editor launches from the first)

**1. Telemetry + DB + editor** (schematic comes from the DB, not a sticky `--kdl` file):

```bash
uv run python examples/db-client/main.py --db-schematic
```

The embedded DB listens on Impeller `2240` and serves assets (including
`schematics/main.kdl`) on HTTP `2241`. The editor fetches that URL. If you
see `error sending request for url http://127.0.0.1:2241/schematics/main.kdl`,
nothing is listening on 2241 — rebuild the Python wheel so `edb.Server`
starts the asset HTTP server, and make sure 2241 is free.

**2. Watch / push** (rebuilds on save):

```bash
elodin ui watch examples/db-client/schematic.py --db 127.0.0.1:2240
# equivalent:
#   python -m elodin.ui.watch examples/db-client/schematic.py --db 127.0.0.1:2240
```

If you used `--no-editor` on the first command, open the editor separately:

```bash
ELODIN_ASSETS_DIR=./assets elodin editor 127.0.0.1:2240
```

## Try it

1. In `examples/db-client/schematic.py`, rename a graph (e.g. `"Battery (V)"` → `"Battery volts"`).
2. Save — watch prints `pushed schematic.py → …` and the editor reloads.
3. Introduce a syntax error — the editor keeps the last-good schematic and shows
   `Schematic build error: …` in the bottom status bar.
4. Fix the error — watch recovers and pushes again.

## Layout overlay (Phase 4)

Drag a split, then command palette → **Save Layout**. That writes layout-only
state (split shares + window rects), **not** `schematic.py`.

- **Canonical:** DB asset `http://127.0.0.1:2241/schematics/main.overlay.kdl`
  (inside the DB data dir, often a temp path for this demo).
- **Temporary local copy** (for inspection; may go away later):
  `schematics/main.overlay.kdl` under the process cwd (repo root if you
  launched from there). The file starts with a comment saying so.

The next `elodin ui watch` rebuild applies the DB overlay before push. Delete
the overlay asset (or the local copy if you are testing apply from disk) to
return to authored shares. **Save Schematic** still writes a full KDL snapshot.
