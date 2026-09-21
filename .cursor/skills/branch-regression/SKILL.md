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
```

If there is **any** staged or uncommitted work, STOP and ask the user to commit
or stash it themselves. Never stash, commit, or discard on their behalf.

### 1 & 2. Capture each branch

For each branch (base first, then head):

```bash
git switch "$BASE_BRANCH"            # then later: git switch "$HEAD_BRANCH"
nix develop --command just install   # rebuild py + editor + db for THIS branch
nix develop --command bash .cursor/skills/branch-regression/capture_branch.sh "$OUT/base"
# head pass writes to "$OUT/head"
```

`capture_branch.sh` writes `<example>.png`, `<example>.log`, `<example>.exit`
per example plus `commit.txt`. Default example set is the known-good editor
gallery (ball, three-body, drone, rc-jet, apollo-lander, video-stream,
sensor-camera, cube-sat, voyager, geo-frames); pass example names to override.
Headless-only examples (frames, linalg, stablehlo, cube-sat-pysim) are run with
`elodin run` for logs/exit only — add them explicitly if wanted.

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
  "$OUT/base" "$OUT/head" --rmse-threshold 0.05
```

Prints a markdown table (exit codes, new WARN/ERROR count, screenshot RMSE,
verdict) and exits 1 if anything is flagged. Save it: `... > "$OUT/report.md"`.

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
