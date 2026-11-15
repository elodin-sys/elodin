---
name: branch-regression
description: Compare two git branches (usually the current branch vs main) by running every example on each, in the native editor and in the wasm browser editor, capturing exit codes, logs, and screenshots, then diffing the results. Use when the user asks to regression-test a branch, compare branches, check a branch for example breakage, or A/B the editor output before merging.
---

# Branch Regression

Runs the same example suite on a base branch (usually `main`) and the current
branch, then compares run results, exit codes, new WARN/ERROR log lines, and
screenshots. Do this twice per branch: once in the native editor, then again in
the wasm browser editor. Screenshots are not expected to match 100% (sims are
non-deterministic) — they should be *close* barring an issue. Pair by target:
native base with native head, and wasm base with wasm head. Don't score a native
screenshot against a wasm screenshot.

Requires: Linux host with display + GPU (editor screenshots), `nix develop`,
repo root as cwd.

## Workflow

```
- [ ] 0. Preflight: clean tree, record current branch
- [ ] 1. Base pass: checkout base, build, capture native, then wasm
- [ ] 2. Head pass: checkout original branch, build, capture native, then wasm
- [ ] 3. Compare + visually inspect flagged screenshots (native and wasm)
- [ ] 4. Report (and confirm you are back on the original branch)
```

### 0. Preflight

```bash
git status --porcelain   # MUST be empty
HEAD_BRANCH=$(git branch --show-current)
BASE_BRANCH=main         # or the branch the user named
OUT=ai-context/branch-regression/$(date +%Y-%m-%d)-$BASE_BRANCH-vs-$HEAD_BRANCH
mkdir -p "$OUT"
if [ -f "$OUT/shell-id.txt" ]; then
  ELODIN_SHELL_ID=$(cat "$OUT/shell-id.txt")
else
  ELODIN_SHELL_ID="${ELODIN_SHELL_ID:-branch-regression-$(date +%Y%m%d-%H%M%S)}"
  printf '%s\n' "$ELODIN_SHELL_ID" > "$OUT/shell-id.txt"
fi
export ELODIN_SHELL_ID
```

If there is **any** staged or uncommitted work, STOP and ask the user to commit
or stash it themselves. Never stash, commit, or discard on their behalf.

Use the same `OUT` and `ELODIN_SHELL_ID` when resuming an interrupted run. Never
create a fresh shell ID for each branch or example. A caller may choose any
path-safe name by exporting `ELODIN_SHELL_ID` before starting, for example:

```bash
export ELODIN_SHELL_ID="branch-regression-$(date +%Y%m%d-%H%M%S)"
```

Named shell directories persist for later passes and resumed sessions. Shells
that start without an explicit ID use a numeric process/session ID and contain
`target/shells/$ELODIN_SHELL_ID/garbage-collectable`; the Nix shell may remove
those after the process dies.

### 1 & 2. Capture each branch

For each branch (base first, then head):

```bash
git switch "$BASE_BRANCH"            # then later: git switch "$HEAD_BRANCH"
nix develop --command bash -lc \
  'just install && bash .cursor/skills/branch-regression/capture_branch.sh "$1"' \
  _ "$OUT/base"
# head pass writes to "$OUT/head"
```

`capture_branch.sh` writes `<example>.png`, `<example>.log`, `<example>.exit`
per example plus `commit.txt`. Default example set is the known-good editor
gallery (ball, three-body, drone, rc-jet, apollo-lander, video-stream,
sensor-camera, cube-sat, voyager, geo-frames); pass example names to override.
Headless-only examples (frames, linalg, stablehlo, cube-sat-pysim) are run with
`elodin run` for logs/exit only — add them explicitly if wanted.

Build and capture must run in the same `nix develop` invocation. Each Nix shell
has its own virtual environment, so a separate capture shell cannot import the
Python package installed by `just install`. Both branch passes must also inherit
the same named `ELODIN_SHELL_ID`, which reuses the virtual environment and bin
directory instead of creating a new session. Still run `just install` after
switching branches so each pass tests binaries built from its own commit.

Rules baked into the script (do not work around them):

- One live editor/sim at a time — everything binds TCP 2240. It waits for the
  port to free between examples and group-kills leftovers (s10 children
  respawn on a plain kill).
- Stale DB cleanup for `video-stream` and `voyager` before each run.
- `ELODIN_SCREENSHOT_EXIT=1` + watchdog; a missing/empty PNG is a capture
  failure, not proof of a regression — retry once before flagging.

**Always `git switch` back to `$HEAD_BRANCH` when done or on any error.**
Stop the wasm server before switching branches.

### WASM pass

Run this on the branch just captured, in the same `nix develop` session and
before `git switch`. The browser editor does not launch a simulation. Trunk
serves the editor, and `elodin run` serves the database. See
`apps/elodin-web/README.md`.

Wasm has no `ELODIN_SCREENSHOT` path. Capture with the browser tools. Skip
headless-only examples (`frames`, `linalg`, `stablehlo`, `cube-sat-pysim`);
they have no scene. Use the same gallery as the native pass otherwise.

```bash
# still inside the nix develop used for this branch; OUT is base or head
mkdir -p "$OUT/wasm"
git rev-parse HEAD >"$OUT/wasm/commit.txt"
git branch --show-current >>"$OUT/wasm/commit.txt"
# Builds the wasm-release profile and serves http://127.0.0.1:8080.
# The asset URL is written into <meta name="elodin-web-db">.
# Trunk 0.21 treats NO_COLOR=1 as --no-color=1 and exits.
ELODIN_WEB_DB=http://127.0.0.1:2241 TRUNK_COLOR=never trunk serve --config apps/elodin-web/Trunk.toml
```

Leave Trunk running for every example on this branch. Wait until it is
actually serving before opening a page. The first response is slow: this
profile is fat LTO. A Trunk process left over from the other branch is the
wrong editor; stop it and serve again after `just install`.

For each example, with nothing else bound to TCP 2240:

```bash
example=ball   # then the rest of the gallery
case "$example" in
  video-stream) rm -rf video-stream-db ;;
  voyager) rm -rf examples/voyager/dbs/voyager dbs/voyager ;;
esac
setsid elodin run "examples/$example/main.py" >"$OUT/wasm/$example.sim.log" 2>&1 &
echo $! >"$OUT/wasm/$example.pid"
# wait until ss shows :2240, or the sim log exits
```

Then, in the browser:

- Viewport 1280×720 on both branches. A size mismatch makes the RMSE meaningless.
- Open `http://127.0.0.1:8080/?example=<example>`. The extra parameter is
  ignored by the editor; it forces a new document so the next example does not
  reuse the previous connection. The page reads `<meta name="elodin-web-db">`
  (`http://127.0.0.1:2241`, the asset server on the port after Impeller TCP)
  and opens its WebSocket there.
- Wait until the boot overlay is hidden (`elodin-ready`) and then the same
  settle delay as the native pass (`ELODIN_SCREENSHOT_DELAY`, default 20s).
  The first example also waits for the wasm module to compile; allow several
  minutes once per branch before calling it a hang.
- Save the screenshot to `$OUT/wasm/<example>.png`.
- Read `window.__eloLogs`. Write each `warn` and `error` entry to
  `$OUT/wasm/<example>.log` as a line beginning with `WARN` or `ERROR`, so
  `compare_runs.py` can see it. A panic string is an `ERROR`.
- Write `$OUT/wasm/<example>.exit`: `0` when the canvas is non-empty, the boot
  overlay is hidden, and the sim process is still running. `1` when the sim
  died, the overlay shows a panic, or the canvas is blank. A blank canvas is a
  capture failure — retry that example once before flagging it.

Stop the sim before the next example. `s10` children respawn after a plain
kill, so signal the process group:

```bash
pid=$(cat "$OUT/wasm/$example.pid")
kill -TERM -- -"$pid" 2>/dev/null
sleep 2
kill -KILL -- -"$pid" 2>/dev/null
# wait until :2240 is free
```

After the gallery, stop Trunk. Port 8080 has to be free for the other branch.

Judge wasm against the other branch's wasm capture. These gaps are compiled
out of the browser editor today, so the same gap on both branches is
pre-existing: Cesium terrain, skyboxes, sensor-camera GPU readback, thruster
particles, cinematic Earth, and the native status-bar hardware stats. A panic,
a failed connection, or missing geometry that the other branch still draws is
a regression. Do not require the native status-bar RAM reading on a wasm
screenshot.

### 3. Compare

```bash
nix develop --command uv run python .cursor/skills/branch-regression/compare_runs.py \
  "$OUT/base" "$OUT/head" --rmse-threshold 0.05 --html "$OUT/report.html" \
  > "$OUT/report.md"
```

Prints a markdown table (exit codes, new WARN/ERROR count, screenshot RMSE,
verdict), writes `report.html`, and exits 1 if anything is flagged. The HTML
report must include base/head screenshot thumbnails for every example; each
thumbnail links to the full-size image. Always generate both report formats.
Run it again for the wasm captures:

```bash
nix develop --command uv run python .cursor/skills/branch-regression/compare_runs.py \
  "$OUT/base/wasm" "$OUT/head/wasm" --rmse-threshold 0.05 \
  --html "$OUT/wasm-report.html" > "$OUT/wasm-report.md"
```

Then, for every flagged example, **Read both PNGs** and judge visually:
- Same scene composition (objects, trails, view cube, graph panels populated)?
- Native status bar healthy (RAM > 0, ticks advancing)? Wasm has no hardware
  stats; judge the scene and the console log instead.
- Is the pixel delta explained by sim phase (a ball mid-bounce vs apex) or is
  content actually missing/broken?

RMSE above threshold with equivalent-looking scenes = note and pass.
Missing geometry, blank viewport, dead graphs, or a new panic = regression.

### 4. Report

Summarize per example, native and wasm separately: base vs head exit code,
new WARN/ERROR lines (quote them), RMSE, visual verdict. State the two
commits compared (from `commit.txt` and `wasm/commit.txt`). Distinguish
**regressions** (head worse than base) from **pre-existing issues** (present
in both). A wasm-only failure is still a regression when the base wasm capture
was healthy.

## Interpreting log diffs

`compare_runs.py` only reports WARN/ERROR lines that are *new in head*
(timestamps stripped, deduped) — noisy-but-stable warnings on both branches do
not flag. Lines that differ only by a pointer/tick number may still slip
through; use judgment.

## Gotchas

- Rebuilding between branches is mandatory; a stale `target/release/elodin` or
  Python wheel silently tests the wrong branch. `commit.txt` in each capture
  dir is the audit trail.
- `voyager` needs SPICE kernels under `examples/voyager/nasa_spice_data/`;
  `video-stream` needs the GStreamer plugins from `nix develop`. If a
  prerequisite is missing on *both* branches, drop the example rather than
  flagging it.
- Screenshot delay: script default 20 s (`ELODIN_SCREENSHOT_DELAY`); heavy
  examples may need 25 s. Same delay on both branches, or RMSE is meaningless.
- Output lives under gitignored `ai-context/`; never commit captures.
- When resuming after a crash, locate the existing output directory and restore
  `ELODIN_SHELL_ID` from `shell-id.txt` before entering `nix develop`.
- The wasm editor is a separate build (`trunk serve` / `just web`). Native
  `target/release/elodin` does not exercise it. Stop Trunk before
  `git switch`, then serve again on the next branch.
- Reload between wasm examples. The gallery all uses port 2240, and a reused
  document keeps the previous database connection.
