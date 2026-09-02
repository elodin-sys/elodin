# Handoff: Python UI SDK (`elodin.ui`) — branch `feat/python-not-kdl`

For the human taking this over **and** their coding agent. Resume from this branch as it stands; do not re-litigate Phases 0–3 unless something is broken.

**Branch:** `feat/python-not-kdl` @ `c77de3fef` (`ai: Phase 2.`)
**Remote:** `origin/feat/python-not-kdl` is at the same SHA.
**Working tree:** clean except untracked `db.tar.zst` — **do not commit that**.

Local `git status` may say the branch tracks `origin/main` and is “ahead 3”. Those three commits *are* this work (`ai: Phase 0/1/2`). Prefer `origin/feat/python-not-kdl` when pushing.

Do not git-commit unless the human asks. Always `nix develop`. Use `uv` inside the nix shell for Python. Run commands from the repo root.

---

## What this project is

Replace **KDL authoring** with Python (`elodin.ui`). Keep KDL as the **build/wire artifact** (`schematics/main.kdl` + `schematic.active` on the DB). The editor still consumes KDL; Python never runs at view time.

Companion review (not in this repo): `elodin-kdl-ergonomics-review-2026-07-28.md` findings F1–F8. Original design doc was pasted in the kickoff chat (2026-07-28 / implemented starting 2026-08-03). Locked decisions from §3.4:

| Decision | Locked choice |
|---|---|
| Package | `elodin.ui` (inside the existing `elodin` wheel) |
| Bindings | PyO3 over `impeller2_wkt` |
| Artifact | KDL text, unchanged DB channel |
| Expressions | Typed Python frontend → EQL strings today; `eql::Expr` / Tier B–C later |
| KDL authoring | Demote, do not delete (replay of historical DBs is forever) |
| Editor save | Layout overlay in Phase 4 — editor must not rewrite source |

FSW KDLs live at `../fsw/assets/schematics` (sibling of this repo). Corpus copies them under `libs/impeller2/kdl/tests/corpus/sources/fsw/` (21 files, not ~20).

---

## Status vs original gates

| Phase | Intent | Status |
|---|---|---|
| **0** | Golden corpus, `PartialEq`, emit determinism | **Done.** G0: `cargo test -p impeller2-kdl --test golden_corpus` |
| **1** | Builders + emit/parse/write/push; examples match handwritten KDL | **Done.** G1: pytest `test_ui.py` |
| **2** | Typed expr + schema | **Mostly done.** Python `Expr`/`Schema`/`pose()`/`sym_mat3()` emit **EQL strings**. **Not done:** PyO3 over `eql::Expr`; property test random AST → parse → equal AST; reproducing every expression in FSW `main.kdl`. That is leftover G2 work, not a reason to restart Phase 2. |
| **3** | `elodin ui watch`, last-good, editor error banner | **Done enough to demo.** No headless-editor integration test (G3 still wants save→re-render &lt; 1s recorded). |
| **4** | Layout overlay | **Implemented.** Overlay KDL + `ui.apply_overlay` / `extract_overlay`; watch merges `*.overlay.kdl`; editor **Save Layout** writes DB asset plus a temporary local inspection copy. |
| **5** | Fleet migration + `to-python` codegen | **In progress.** `elodin schematic to-python` emits a lossless executable scaffold; corpus model-equality coverage is in `test_ui.py`. Typed-builder rewrite of FSW `main.kdl` remains. |
| **6** | Tier B: real math in editor/eql (faer, per-sample graph eval) | **Not started** |
| **7** | Tier C: JAX/StableHLO display kernels | **Not started** (do not build speculatively) |

The commit message `ai: Phase 2.` also contains Phase 3 (watch CLI, status bar, demo docs). Treat Phases 0–3 as landed on this SHA.

---

## How to verify before changing anything

```bash
nix develop
just install py          # maturin 1.12.6 pin in justfile — do not bump casually
just install editor      # elodin → ~/.cargo/bin

# Rust goldens (Phase 0)
cargo test -p impeller2-kdl --test golden_corpus

# Python UI (Phases 1–2)
python -m pytest libs/nox-py/python/tests/test_ui.py libs/nox-py/python/tests/test_ui_expr.py -v
```

Bless goldens only after intentional parser/serializer changes:

```bash
BLESS_GOLDENS=1 cargo test -p impeller2-kdl --test golden_corpus
```

CI: `cargo fmt`, `cargo test`, `cargo clippy -- -Dwarnings`, `ruff format --check && ruff check --fix`, `alejandra`. For `elodin-db` tests: `CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1`.

---

## Demo the human can run (Phase 3)

Written up in `examples/db-client/UI_WATCH.md`.

```bash
nix develop
export PATH="$HOME/.cargo/bin:$PATH"
export ELODIN_PYTHON="$(pwd)/.venv/bin/python"
```

Terminal 1:

```bash
uv run python examples/db-client/main.py --db-schematic
```

Terminal 2:

```bash
elodin ui watch examples/db-client/schematic.py --db 127.0.0.1:2240
```

`--db-schematic` seeds `schematic.active` from Python and **does not** pass sticky `--kdl`, so watch/reload can work. Edit `examples/db-client/schematic.py`, save, editor should hot-reload. Syntax error → last-good kept; status bar shows `Schematic build error: …` from `DbConfig.metadata["ui.build_error"]`.

---

## Map of the code

| Area | Path |
|---|---|
| Native builders | `libs/nox-py/src/ui/{mod.rs,builders.rs}` |
| Python package | `libs/nox-py/python/elodin/ui/` (`__init__.py`, `expr.py`, `schema.py`, `watch.py`) |
| Re-export | `libs/nox-py/python/elodin/__init__.py` → `from . import ui` |
| `world.schematic` accepts `str \| Schematic` | `libs/nox-py/src/world_builder.rs` |
| Watch CLI | `apps/elodin/src/cli/ui.rs` (`elodin ui watch`) — enum is `UiCommand` to avoid clashing with `std::process::Command` |
| Editor banner | `libs/elodin-editor/src/ui/status_bar.rs` |
| Golden corpus | `libs/impeller2/kdl/tests/corpus/` + `golden_corpus.rs` |
| `PartialEq` on GUI types | `libs/impeller2/wkt/src/gui.rs` |
| Demo | `examples/db-client/{schematic.py,main.py,UI_WATCH.md,schematic.kdl}` |
| Other Python rebuilds | `examples/drone/{motor_panel.py,rate_control_panel.py}` |
| Tests | `libs/nox-py/python/tests/test_ui.py`, `test_ui_expr.py` |
| `watchfiles` dep | `libs/nox-py/pyproject.toml` |

**Native vs Python `ui`:** PyO3 registers the submodule as `elodin.elodin.ui` (same pattern as `monte_carlo`). **Do not** `sys.modules["elodin.ui"] = native_module`. That shadows `python/elodin/ui/` and drops `Expr`/`Schema`/`watch`. There is an explicit comment in `libs/nox-py/src/ui/mod.rs` `register()`.

Builders accept `str | Expr | list` via `extract_eql` (graphs, viewport pos/look_at/up, object_3d, line_3d, joint, vector_arrow).

`Schema.from_db` / `from_json` exist. Strict unknown names / OOB index raise `ExprError`. File:line on those errors is **not** implemented (G2 wanted it).

---

## Dev friction (this machine; likely to recur)

- `nix develop` on a dirty tree often rebuilds `elodinsink` and can sit for 10+ minutes. Prefer `--offline` once the flake is cached, or reuse an already-entered shell.
- Rustup on PATH fights nix `rustc`. Inside the shell, put nix rust-default first (`RUSTC=…`). `just install` documents this.
- JAX/numpy need nix `libstdc++` / `libz`. Run pytest **inside** `nix develop`, not a bare host venv.
- Maturin: pin **1.12.6** (`justfile`). Newer maturin can leave 0-byte `libelodin.so` (PyO3/maturin#3054); justfile deletes those.
- `elodin ui watch` spawns `ELODIN_PYTHON` or `python3` with `-m elodin.ui.watch`. Point `ELODIN_PYTHON` at `.venv/bin/python` or the CLI will miss the wheel.
- Port 2240 in use → `Address already in use` from `edb.Server.start`. Kill leftover `elodin-db`.
- Building `-p elodin` outside a full nix env fails on `alsa.pc` (`PKG_CONFIG_PATH`). Stay in `nix develop`.

Skills: `.agents/skills/elodin-dev/SKILL.md`, `elodin-db`, `elodin-editor-dev`, `nox-py-dev`, `elodin-nix`.

---

## What to do next (agent plan)

Work phases **in order**. Do not start 6–7 unless 4–5 are done or the human explicitly wants math.

### Immediate (close G2/G3 gaps if cheap)

1. Confirm demo still works after `just install py` + `just install editor`.
2. Optional G2 tighten:
   - Attach source locations to `ExprError` (file:line).
   - Property test: random EQL AST → string → `eql::parse` → equal AST (Rust or Python).
   - Do **not** block Phase 4 on migrating FSW `main.kdl`; that is Phase 5.
3. Optional G3: a non-GUI test that watch `--once` pushes KDL and `set_build_error` writes/clears metadata (pytest against embedded DB already covers `push` in `test_ui.py`).

### Phase 4 — Layout overlay (next product work)

Goal: editor owns drag-discovered layout (split shares, active tab, window rects) **without rewriting Python/KDL source**.

1. Define a sparse overlay (layout-only), stored as `{db}/assets/schematics/overlay.kdl` or equivalent.
2. Editor “Save Layout” writes overlay, not full schematic.
3. `ui.apply_overlay(schematic, overlay)` at build/watch time; code values are defaults.
4. Watch loop applies overlay after each successful `build()`.
5. Gate G4: drag + save → rerun Python → shares persist; `git diff` of dashboard source empty; delete overlay → authored layout returns; overlay survives DB replay.

See original §5.4. Open question: per-schematic vs per-workstation overlays.

### Phase 5 — Fleet migration

1. `elodin schematic to-python` (FR-10). The initial lossless scaffold preserves the complete source and promotes `//` lines to Python comments.
2. Migrate `../fsw/assets/schematics` (~21 files). **Author** `main.kdl` in Python (showcase), do not only transliterate.
3. Per-file model equality vs KDL (same G1 trick: `from_kdl(emit(py)) == from_kdl(handwritten)`), reviewed diffs where improvements are intentional.
4. Replace fsw `main_kdl_contract.py` static half with `python dashboards/build_all.py`; keep SITL-CSV data-presence checks.
5. Rollback is always “use the emitted KDL”.

### Phase 6 — Tier B (runtime math)

Only after 4–5, or if graphs-of-formulas are blocking. Extend `eql::Expr` + editor compilation (faer); per-sample eval on the plot path; fix `norm`; `sym_mat3` / Cholesky vs golden data. Minor-version boundary: graph formulas start *working*.

### Phase 7 — Tier C

JAX → StableHLO display kernels in the editor. **Do not start unless Tier B ops are insufficient.**

---

## Known incomplete vs the original spec

- Expressions are **strings**, not `eql::Expr` nodes in Rust.
- G2 “every EQL in `main.kdl` via typed layer” not done.
- `to-python` currently emits a lossless `ui.from_kdl(SOURCE_KDL)` scaffold. It does not yet mechanically translate every KDL node to typed builder calls because the typed builder surface does not cover every parser feature.
- Overlay is per-schematic (`schematics/<stem>.overlay.kdl`), not per-workstation. Active-tab index is not in the model yet, so overlay covers split shares + window rects only.
- `Expr.__add__` parenthesizes (`(a + b)`). Handwritten `examples/db-client/schematic.kdl` chase `pos` was updated to match; model equality is after parse, not byte-identical KDL.
- `test_ui.py` `test_push_to_embedded_server` needs a free DB port; don’t run two demos on 2240.
- Headless `main.py --db-schematic` + concurrent client can hit `Already borrowed` on `latest()` — demo issue, not the SDK path.
- **Fixed 2026-08-31:** `edb.Server.start` did not spawn the DB Asset Server on TCP+1. The editor then failed with `http://127.0.0.1:2241/schematics/main.kdl: error sending request for url`. `libs/nox-py/src/db/server.rs` now calls `spawn_assets_http` like `elodin-db run` / `world.run`. Rebuild the Python wheel after pulling.

---

## Agent constraints (Elodin repo)

- Never `unsafe` Rust.
- No backwards-compat shims unless asked.
- Don’t invent new deps without checking they are maintained.
- Don’t commit; don’t force-push `main`.
- Leave unrelated untracked files (`db.tar.zst`) alone.

Prior implementation chat (this branch’s agent session): [Python UI SDK](b05d04d4-7417-4f62-b14e-bf9d8558bca1).
