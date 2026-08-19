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
