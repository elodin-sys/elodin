---
name: qa-test-plan
description: Author, instantiate, and execute agentic QA test case plans for Elodin releases. Use when the user asks to create a QA plan, test case plan, or release test checklist, add or edit test cases, or run/execute a QA test plan for a release.
---

# QA Test Plan

An agentic QA plan is a markdown document that replaces the traditional QA spreadsheet. Each test case carries machine-actionable steps and objective pass criteria so an agent can execute it autonomously; the filled-out document is both the checklist and the report.

Everything for this skill lives in this directory (`.cursor/skills/qa-test-plan/`):

- Reusable template for new plans: [template.md](template.md)
- **Ready-to-run plan — examples smoke/feature test:** [examples/test-plan.md](examples/test-plan.md) (24 cases: `SDK-001` build + one per `examples/` project). Shared helpers live alongside it in [examples/](examples/) (`run_probe.sh`, `probe.py`).
- **Ready-to-run plan — Elodin-DB deep suite:** [elodin-db/test-plan.md](elodin-db/test-plan.md) (23 cases: generate realistic DBs from the examples — drone, logstream, video-stream, sensor-camera, apollo-lander, rc-jet — then exercise every `elodin-db` subcommand, serving mode, replication, exports, and clients on them). Helper: [elodin-db/serve_probe.py](elodin-db/serve_probe.py); reuses the examples probes.
- New instantiated plans use the same focused-folder pattern: `<yyyy-mm-dd>-<release>/test-plan.md` plus evidence artifacts in that folder.

## Agent entry point

- **"Run/execute the examples QA plan" (or "smoke-test the examples for a release")** → open [examples/test-plan.md](examples/test-plan.md), set the Plan Header (commit, branch, date, environment, executor), then execute cases top-to-bottom following that document's own **Execution Rules**. It is self-contained and every `agent` case has an `AUTHOR-VALIDATED` baseline. Run from the repo root inside `nix develop`. Note the two hard-won prerequisites baked into [examples/run_probe.sh](examples/run_probe.sh): live-run cases bind port `2240` exclusively (never parallelize them), and tearing a run down requires a **process-group kill** (the s10 child restarts on a plain kill).
- **"Run/execute the Elodin-DB QA plan" (or "put elodin-db through its paces")** → open [elodin-db/test-plan.md](elodin-db/test-plan.md) and execute the same way. It first generates a stable of realistic DBs under `/tmp/qa-db/` from the examples, then runs the full `elodin-db` command surface over them; its **Hard-won operational rules** section (group-kill teardown, the s10 first-recipe-exit group cancel, the `tcp+1` asset-server port collision) is required reading before touching live servers.
- **"Create/author a new plan for `<release>`"** → follow *Instantiating a Plan for a Release* below (copy `template.md` into this folder).
- **"Add/edit a test case"** → follow *Authoring Test Cases* below and the case anatomy rules.

There are three workflows: **authoring** test cases, **instantiating** a plan for a release, and **executing** a plan. The execution rules themselves live inside the template (and every copied plan) so a plan is self-contained.

## Authoring Test Cases

### Anatomy of a case

Every case is a markdown block with these fields (see template for exact layout):

| Field | Purpose |
|-------|---------|
| Checkbox + ID + name | `#### - [ ] AREA-NNN — Short name` |
| Priority | P0 release-blocking smoke / P1 core functionality / P2 nice-to-have |
| Mode | `agent`, `agent+visual`, or `manual` |
| Requires | Case IDs that must PASS first (keep chains shallow; most cases require only SDK-001) |
| Description | One or two sentences: what behavior this verifies and why it matters |
| Expected duration | Lets the executor set command timeouts sensibly |
| Steps | Exact shell commands, one per line |
| Pass criteria | Objective checklist; every item must be verifiable from command output or artifacts |
| Result / Evidence / Notes | Filled in during execution, left blank when authoring |

### Rules for agent-executable cases

1. **One behavior per case.** If a case verifies two unrelated things, split it.
2. **Steps are exact commands**, run from repo root inside the Nix shell (`nix develop --command ...`). Never write prose steps like "open the editor and check it works".
3. **Commands must terminate.** Beware: `elodin run <sim.py>` and `elodin editor <sim.py>` keep serving the database even after the sim reaches `max_ticks` — they never exit on their own. The terminating form is bench mode: `uv run python <sim.py> bench --ticks N` (honors `ELODIN_DB_PATH`). For servers, background them and kill them in the same case.
4. **Pass criteria are objective**: exit codes, specific output strings, file existence, row counts, screenshot contents. If a criterion can't be checked mechanically or from a screenshot, the case belongs in `manual` mode.
5. **Cases are independent.** No case may rely on another case's side effects except through the declared Requires line. Use `/tmp/qa-run/<case-id>/` for scratch state and clean it up.
6. **Deterministic where possible**: fixed seeds, fixed tick counts, tolerance ranges for anything timing-dependent.
7. **Never pipe the command under test** (`cmd | tail` reports the pipe's exit code, not `cmd`'s). Redirect to a file in the artifacts directory and inspect that instead.
8. **Background processes must be managed inside one wrapper command**: capture the PID, probe, kill by PID, and `exit` with the probe's code (see DB-002 in the template).

### ID and area conventions

IDs are `AREA-NNN`, stable across releases — never renumber or reuse an ID; retired cases keep their number forever and new cases append. Current areas:

| Prefix | Area |
|--------|------|
| SDK | Python SDK build/install |
| SIM | Simulation runtime (headless) |
| DB | Elodin-DB |
| EDITOR | Elodin Editor |
| RUST, LINT | Build area: workspace tests, CI checks |
| ALEPH | Flight computer (usually `manual` — needs hardware) |

Add new areas as needed; record them in this table so IDs stay consistent.

### Execution modes

| Mode | Meaning |
|------|---------|
| `agent` | Fully autonomous, shell-only |
| `agent+visual` | Agent runs it but verification needs a display/screenshot; BLOCKED on headless machines |
| `manual` | A human must perform it (hardware, subjective judgment); agent marks it SKIPPED and reports it |

## Instantiating a Plan for a Release

1. Copy the template:

```bash
PLAN_DIR=".cursor/skills/qa-test-plan/$(date +%F)-<release>"
mkdir -p "$PLAN_DIR"
cp .cursor/skills/qa-test-plan/template.md "$PLAN_DIR/test-plan.md"
```

2. Fill the Plan Header: release name, `git rev-parse --short HEAD`, `git branch --show-current`, date, environment (note whether a display is available), executor.
3. Scope the plan: delete cases irrelevant to this release, add release-specific cases following the authoring rules. Keep the Summary table in sync with the case blocks — same IDs, same order.
4. If new cases were added that future releases should keep, also add them to [template.md](template.md) so the template stays the source of truth.

## Executing a Plan

The binding rules are in the plan document itself ("Execution Rules" section) — read them first; they travel with every copy. Operationally:

1. Set the header Status to IN PROGRESS.
2. Work through cases strictly in Summary-table order, one at a time. For each case: check Requires, run the steps, verify every pass criterion, write Evidence (exit codes, matching output lines, artifact paths), then set Result in the case block and mirror it in the Summary row.
3. Long builds are normal (SDK-001 up to 30 min, RUST-001 up to 40 min) — use the case's Expected duration to size timeouts instead of assuming a hang.
4. On FAIL, save output to `<artifacts>/<case-id>-fail.log`, diagnose briefly in Notes, and move on. Never fix code mid-run; a QA run measures the release as it is.
5. When all cases have a result: fill the Run Summary counts, list notable issues and follow-ups, set Status to COMPLETE, and report to the user — lead with FAIL/BLOCKED cases and anything a human must do (`manual` cases marked SKIPPED).

## Example: well-formed vs poor case

**Poor (not agent-executable):**

> Steps: Open the editor and load a simulation. Check that everything looks right.

**Well-formed:** see `SIM-001` in [template.md](template.md) — exact command, bounded runtime, three mechanical pass criteria (exit code, summary block present, no traceback/panic).
