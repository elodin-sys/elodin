---
name: branch-regression
description: Compare two git branches (usually the current branch vs main) by running every example on each, capturing exit codes, logs, and editor screenshots, then diffing the results. Use when the user asks to regression-test a branch, compare branches, check a branch for example breakage, or A/B the editor output before merging.
---

# Branch Regression

Runs the same example suite on a base branch (usually `main`) and the current
branch, then compares run results, exit codes, new WARN/ERROR log lines, and
screenshots. Screenshots are not expected to match 100% (sims are
non-deterministic) — they should be *close* barring an issue.

Requires: Linux host with display + GPU (editor screenshots), `nix develop`,
repo root as cwd.

## Workflow

```
- [ ] 0. Preflight: clean tree, record current branch
- [ ] 1. Base pass: checkout base, build, capture
- [ ] 2. Head pass: checkout original branch, build, capture
- [ ] 3. Compare + visually inspect flagged screenshots
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

Then, for every flagged example, **Read both PNGs** and judge visually:
- Same scene composition (objects, trails, view cube, graph panels populated)?
- Status bar healthy (RAM > 0, ticks advancing)?
- Is the pixel delta explained by sim phase (a ball mid-bounce vs apex) or is
  content actually missing/broken?

RMSE above threshold with equivalent-looking scenes = note and pass.
Missing geometry, blank viewport, dead graphs, or a new panic = regression.

### 4. Report

Summarize per example: base vs head exit code, new WARN/ERROR lines (quote
them), RMSE, visual verdict. State the two commits compared (from
`commit.txt`). Distinguish **regressions** (head worse than base) from
**pre-existing issues** (present in both).

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
