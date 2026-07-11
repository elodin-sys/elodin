# QA Test Plan: <release or milestone>

> Copy this file to `.cursor/skills/qa-test-plan/<yyyy-mm-dd>-<release>/test-plan.md` before filling it out.
> Evidence artifacts go in the same folder (see the curated [examples/](examples/) plan for the pattern).

## Plan Header

| Field | Value |
|-------|-------|
| Release / milestone | `<e.g. v0.17.4>` |
| Git commit | `<git rev-parse --short HEAD>` |
| Branch | `<git branch --show-current>` |
| Date started | `<yyyy-mm-dd>` |
| Environment | `<OS, machine, display available yes/no>` |
| Executor | `<agent model / human name>` |
| Status | NOT STARTED / IN PROGRESS / COMPLETE |

## Execution Rules

These rules travel with the plan so any agent can execute it without extra context.

1. Run every command from the repository root, inside the Nix shell (`nix develop --command <cmd>` or an activated shell).
2. Execute cases in the order listed, one at a time. Do not parallelize cases.
3. Before starting a case, check its **Requires** line. If a required case did not PASS, mark this case BLOCKED (do not attempt it) and record which requirement failed.
4. A case is **PASS** only when every item under **Pass criteria** is verified true. Never infer success from the absence of errors alone.
5. Record **Evidence** before writing the result: command exit codes, the specific output lines that satisfy each criterion, and paths to any artifacts saved under the artifacts directory.
6. On **FAIL**: capture the full failing output to `<artifacts>/<case-id>-fail.log`, fill in Notes with a one-paragraph diagnosis, then continue to the next case. Do not attempt fixes mid-run.
7. Stop the run entirely only if a P0 case fails in a way that invalidates everything downstream (e.g. the build itself is broken). Mark remaining cases BLOCKED with a reason.
8. Cases marked `manual` are not attempted by the agent: mark them SKIPPED and list them in the run summary for a human.
9. Cases marked `agent+visual` require a display. On a headless machine mark them BLOCKED with reason "no display".
10. Kill any background processes a case started before moving on, even on failure. Use dedicated temp dirs under `/tmp/qa-run/<case-id>/` and clean them up.
11. After each case: update the case block first (Result, Evidence, Notes), then its row in the Summary table. Update the run summary footer only at the end.

### Result states

| State | Meaning |
|-------|---------|
| PASS | All pass criteria verified true |
| FAIL | At least one criterion false, or the steps errored |
| BLOCKED | Could not attempt (unmet requirement, missing display/hardware) |
| SKIPPED | Deliberately not run this release (out of scope, manual-only) |

## Summary

| # | ID | Name | Area | Priority | Mode | Result |
|---|----|------|------|----------|------|--------|
| 1 | SDK-001 | Toolchain build and install | SDK | P0 | agent | |
| 2 | SIM-001 | Headless simulation completes | Simulation | P0 | agent | |
| 3 | DB-001 | Telemetry write/read round-trip | DB | P0 | agent | |
| 4 | DB-002 | Standalone database server starts | DB | P1 | agent | |
| 5 | RUST-001 | Rust test suite passes | Build | P1 | agent | |
| 6 | LINT-001 | CI lint checks pass | Build | P2 | agent | |
| 7 | EDITOR-001 | Editor renders a simulation | Editor | P1 | agent+visual | |

---

## Test Cases

### SDK

#### - [ ] SDK-001 — Toolchain build and install

- **Priority:** P0 | **Mode:** agent | **Requires:** none
- **Description:** The full toolchain (Python SDK, editor, database) builds from source and installs runnable binaries. Everything else depends on this.
- **Expected duration:** up to 60 min (cold build), ~5 min warm

**Steps**

```bash
nix develop --command just install
nix develop --command elodin --version
nix develop --command elodin-db --version
```

**Pass criteria**

- [ ] `just install` exits 0
- [ ] `elodin --version` exits 0 and prints a version string
- [ ] `elodin-db --version` exits 0 and prints a version string

**Result:**
**Evidence:**
**Notes:**

---

### Simulation

#### - [ ] SIM-001 — Headless simulation completes

- **Priority:** P0 | **Mode:** agent | **Requires:** SDK-001
- **Description:** A bounded example simulation (`examples/ball`, 1200 ticks) runs headless to completion in bench mode. Note: `elodin run <sim.py>` starts a persistent server and never exits — bench mode is the terminating form.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command uv run python examples/ball/main.py bench --ticks 1200
```

**Pass criteria**

- [ ] Command exits 0
- [ ] Output contains the timing block (`tick time`, `build time`, `real_time_factor` lines)
- [ ] Output contains no Python traceback and no Rust panic

**Result:**
**Evidence:**
**Notes:**

---

### DB

#### - [ ] DB-001 — Telemetry write/read round-trip

- **Priority:** P0 | **Mode:** agent | **Requires:** SDK-001
- **Description:** A simulation writes telemetry to a persistent database; the DB CLI reads the same components back.
- **Expected duration:** < 3 min

**Steps**

```bash
mkdir -p /tmp/qa-run/db-001
nix develop --command sh -c 'ELODIN_DB_PATH=/tmp/qa-run/db-001/db uv run python examples/ball/main.py bench --ticks 1200'
nix develop --command elodin-db list-components /tmp/qa-run/db-001/db
nix develop --command elodin-db export --format csv --flatten --output /tmp/qa-run/db-001/export /tmp/qa-run/db-001/db
wc -l /tmp/qa-run/db-001/export/ball.world_pos.csv
rm -rf /tmp/qa-run/db-001
```

**Pass criteria**

- [ ] Simulation exits 0
- [ ] `list-components` output includes `ball.world_pos`
- [ ] Export exits 0 and reports `ball.world_pos.csv` with 1201 rows (initial state + 1200 ticks; ±1 acceptable)

**Result:**
**Evidence:**
**Notes:**

---

#### - [ ] DB-002 — Standalone database server starts

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** `elodin-db` starts as a standalone server and accepts TCP connections on its bind port.
- **Expected duration:** < 1 min

**Steps**

One self-contained command: starts the server, probes the port, kills the server by PID, and propagates the probe's exit code.

```bash
mkdir -p /tmp/qa-run/db-002
nix develop --command sh -c '
  elodin-db run "[::]:2240" /tmp/qa-run/db-002/db --log-level warn &
  DB_PID=$!
  sleep 3
  python3 -c "import socket; socket.create_connection((\"127.0.0.1\", 2240), timeout=5); print(\"TCP OK\")"
  RC=$?
  kill $DB_PID
  exit $RC
'
rm -rf /tmp/qa-run/db-002
```

**Pass criteria**

- [ ] Wrapper command exits 0
- [ ] Output contains `TCP OK`

**Result:**
**Evidence:**
**Notes:**

---

### Build

#### - [ ] RUST-001 — Rust test suite passes

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** The workspace Rust tests pass with the elodin-db thread-limit flags (prevents OOM crashes).
- **Expected duration:** 10–40 min

**Steps**

```bash
nix develop --command sh -c 'CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1 cargo test'
```

**Pass criteria**

- [ ] `cargo test` exits 0
- [ ] Final output reports 0 failed tests

**Result:**
**Evidence:**
**Notes:**

---

#### - [ ] LINT-001 — CI lint checks pass

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** The formatting and lint checks CI enforces all pass in check mode (no files modified).
- **Expected duration:** 5–15 min

**Steps**

```bash
nix develop --command cargo fmt --check
nix develop --command cargo clippy -- -Dwarnings
nix develop --command ruff format --check
nix develop --command ruff check
nix develop --command alejandra --check .
```

**Pass criteria**

- [ ] All five commands exit 0

**Result:**
**Evidence:**
**Notes:**

---

### Editor

#### - [ ] EDITOR-001 — Editor renders a simulation

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** SDK-001
- **Description:** The editor launches with the three-body example and renders the viewport (three colored spheres visible).
- **Expected duration:** < 5 min

**Steps**

```bash
nix develop --command elodin editor examples/three-body/main.py &
EDITOR_PID=$!
sleep 20
kill -0 $EDITOR_PID && echo "EDITOR ALIVE"
# Screenshot the editor window with whatever capture tool the environment provides
# (e.g. gnome-screenshot -w, grim, or agent screenshot tooling); save to
# <artifacts>/EDITOR-001.png, then:
kill $EDITOR_PID
```

**Pass criteria**

- [ ] `EDITOR ALIVE` printed (process survived startup)
- [ ] Screenshot shows the editor viewport with three rendered spheres
- [ ] Screenshot saved to `<artifacts>/EDITOR-001.png`

**Result:**
**Evidence:**
**Notes:**

---

## Run Summary

| Metric | Count |
|--------|-------|
| Total | 7 |
| PASS | |
| FAIL | |
| BLOCKED | |
| SKIPPED | |

**Notable issues:**

**Follow-up items:**
