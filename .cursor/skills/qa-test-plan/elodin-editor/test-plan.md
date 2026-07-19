# QA Test Plan: Elodin Editor deep suite (viewport, panels, playback, video)

> Puts the **Elodin Editor** through its paces: startup/connectivity, Bevy 0.19 viewports and 3D schematics, panels/KDL, graphs, timeline/status bar, and video decode — covering feature-catalog §16–17 plus the screenshot harness used for visual regressions.
> Evidence artifacts + helpers live in `.cursor/skills/qa-test-plan/elodin-editor/`.
> Authored per `.cursor/skills/qa-test-plan/`. Area prefix `EDITOR`, numbered from `EDITOR-100` (the template reserves `EDITOR-001` for a generic smoke case).

## Plan Header

| Field | Value |
|-------|-------|
| Release / milestone | Elodin Editor deep suite |
| Git commit | `5fdb9118` |
| Branch | `main` |
| Date started | `2026-07-19` (author-validation) |
| Environment | macOS Darwin 25, display+GPU yes (author-validation host) |
| Executor | Cursor agent (author-validation) |
| Status | AUTHOR-VALIDATED (cases exercised; Summary Results left blank for release runs) |

## Method: screenshot the real editor, assert pixels + logs

Bench mode and live DB probes prove the data plane. This plan proves the **editor UX surface**:

1. Build a release `elodin` binary (visual loops need speed).
2. Launch `elodin editor examples/<x>/main.py` with the env-gated `EnvScreenshotPlugin` (`ELODIN_SCREENSHOT*`) so Bevy captures the full window (3D + egui) without OS screen-recording permissions.
3. Assert the PNG exists/non-empty, the log shows a clean capture, and an agent (or human) **Reads the image** to confirm the scene-specific facts listed in each case.
4. A few `agent` cases cover connectivity/CLI without pixels; `manual` cases cover interaction (orbit, palette typing, skybox API keys).

Coverage map (feature-catalog §16–17 → cases):

| Catalog area | Cases |
|--------------|-------|
| 16.1 Startup & connectivity | EDITOR-100, EDITOR-101, EDITOR-102 |
| 16.2 Viewport & 3D (objects, trails, view cube, thrusters, terrain, frusta, video tile) | EDITOR-110–118 |
| 16.3 Panels & KDL | EDITOR-110/111 (viewport+graph layouts), EDITOR-120 |
| 16.4 Graphs & data | EDITOR-111, EDITOR-117 |
| 16.5 Playback / timeline / status bar | EDITOR-110 (LIVE/tick/RAM/FPS/TPS) |
| 16.6 Command palette | EDITOR-190 (manual) |
| 16.7 Theming & assets | EDITOR-191 (manual) |
| 17 Video streaming & decoding | EDITOR-118, EDITOR-192 (manual ingestion) |
| Dev harness / feature flags | EDITOR-100, EDITOR-130 |

### Shared helpers

- [capture.sh](capture.sh) — one example → one PNG via `ELODIN_SCREENSHOT*`, with watchdog + stale-DB cleanup for `video-stream` / `voyager`.
- [../../../scripts/ci/screenshot_examples.sh](../../../scripts/ci/screenshot_examples.sh) — batch gallery capture (same plugin).
- Screenshot skill notes: [../../elodin-editor-dev/SKILL.md](../../elodin-editor-dev/SKILL.md) (*Screenshot-driven design, build, and test*).

### Hard-won operational rules

1. **One live editor/sim at a time.** All bind TCP **2240** (asset HTTP on 2241). Never parallelize with `elodin run`, another editor, or monte-carlo workers that also want 2240.
2. **Teardown.** After a failed/hung capture: `pkill -9 -f 'elodin editor'` and confirm `lsof -iTCP:2240 -sTCP:LISTEN` is empty before the next case.
3. **Stale DBs.** `video-stream` → `rm -rf video-stream-db`. `voyager` → clear `examples/voyager/dbs/voyager` (and `dbs/voyager` if present). Time-travel / corrupt DB errors are not Bevy regressions.
4. **Screenshot exit.** Always set `ELODIN_SCREENSHOT_EXIT=1` for agent cases. Killing on a timer alone can lose the async PNG readback.
5. **Delay budget.** Light examples: 12–15 s. Heavy (`apollo-lander`, `sensor-camera`, `voyager`, `video-stream`): 20–25 s. Watchdog ≥ 180 s.
6. **Vision criteria.** Pass criteria that say “screenshot shows …” require the executor to Read the PNG (or equivalent vision/OCR) — file existence alone is not enough.
7. **RAM gauge.** Expect `RAM Usage: X.Y GB` with X.Y > 0 (not `N/A`, not stuck at `0.0`). Bevy's `SystemInformationDiagnosticsPlugin` is **not** the source of truth on macOS.
8. **Recorded DB CLI.** `elodin-db run` takes positional `ADDR` then `PATH` (`elodin-db run "[::]:2250" /tmp/db`). Direct `elodin editor <db-dir>` hits an unfinished `ReplayDir` stub and panics — serve + connect by address instead (EDITOR-102).

## Execution Rules

1. Run every command from the repository root, inside the Nix shell when possible (`nix develop --command <cmd>`). A warm `./target/release/elodin` is acceptable for screenshot cases once EDITOR-100 has built it.
2. Execute cases in Summary order, one at a time. Never parallelize live-editor cases.
3. Check **Requires** first. If unmet, mark BLOCKED.
4. A case is **PASS** only when every **Pass criteria** item is verified true.
5. Record **Evidence** (exit codes, matching log lines, PNG paths, vision notes) before Result.
6. On **FAIL**: save logs under `/tmp/qa-editor/<case-id>-fail.log`, diagnose briefly in Notes, continue. Never fix code mid-run.
7. Stop the whole run only if `SDK-001` or `EDITOR-100` fails; mark the rest BLOCKED.
8. `manual` cases: mark SKIPPED and list for a human.
9. `agent+visual` cases need a display + GPU; on headless machines mark BLOCKED.
10. Kill any editor a case started before moving on.
11. After each case: update the case block, then its Summary row. Fill the run-summary footer at the end.

> **Author-validation note.** Agent cases carry an `AUTHOR-VALIDATED` line with output observed on 2026-07-19 @ `5fdb9118` (macOS Darwin, display + GPU present, warm release build). Treat those as the expected-good baseline; tick counts and RAM floats vary slightly run to run.

### Result states

| State | Meaning |
|-------|---------|
| PASS | All pass criteria verified true |
| FAIL | At least one criterion false, or the steps errored |
| BLOCKED | Could not attempt (unmet requirement, missing GPU/display) |
| SKIPPED | Deliberately not run (manual-only this pass) |

## Summary

| # | ID | Name | Area | Priority | Mode | Result |
|---|----|------|------|----------|------|--------|
| 1 | SDK-001 | Toolchain build and install | SDK | P0 | agent | |
| 2 | EDITOR-100 | Release editor binary builds | Editor | P0 | agent | |
| 3 | EDITOR-101 | Connect editor to a live DB address | Editor | P1 | agent | |
| 4 | EDITOR-102 | Open a recorded DB path | Editor | P1 | agent | |
| 5 | EDITOR-110 | ball — viewport, trail, status bar, RAM | Editor | P0 | agent+visual | |
| 6 | EDITOR-111 | three-body — multi-entity + graphs | Editor | P0 | agent+visual | |
| 7 | EDITOR-112 | drone — GLB + joint animation | Editor | P1 | agent+visual | |
| 8 | EDITOR-113 | rc-jet — FPV / aero viewport | Editor | P1 | agent+visual | |
| 9 | EDITOR-114 | apollo-lander — descent + thrusters | Editor | P1 | agent+visual | |
| 10 | EDITOR-115 | sensor-camera — GPU sensors / frusta | Editor | P1 | agent+visual | |
| 11 | EDITOR-116 | geo-frames — coordinate / terrain | Editor | P2 | agent+visual | |
| 12 | EDITOR-117 | cube-sat — spacecraft + graphs | Editor | P1 | agent+visual | |
| 13 | EDITOR-118 | video-stream — H.264 tile decode | Editor | P1 | agent+visual | |
| 14 | EDITOR-119 | voyager — deep-space + SPICE | Editor | P2 | agent+visual | |
| 15 | EDITOR-120 | --kdl schematic preload | Editor | P2 | agent+visual | |
| 16 | EDITOR-130 | Feature flag builds (inspector) | Editor | P2 | agent | |
| 17 | EDITOR-190 | Command palette interactions | Editor | P1 | manual | |
| 18 | EDITOR-191 | Theme / color scheme switching | Editor | P2 | manual | |
| 19 | EDITOR-192 | External video ingest (OBS/RTSP) | Editor | P2 | manual | |

---

## Test Cases

### SDK

#### - [ ] SDK-001 — Toolchain build and install

- **Priority:** P0 | **Mode:** agent | **Requires:** none
- **Description:** The full toolchain builds from source and installs runnable binaries. Downstream cases need `elodin` / `elodin-db` on PATH (or a release binary from EDITOR-100).
- **Expected duration:** up to 60 min cold, ~5 min warm

**Steps**

```bash
nix develop --command just install
nix develop --command elodin --version
nix develop --command elodin-db --version
```

**Pass criteria**

- [ ] `just install` exits 0
- [ ] `elodin --version` and `elodin-db --version` each exit 0 and print a version string

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: warm `elodin`/`elodin-db` already present from prior nix install; version prints `0.17.4-alpha.0+…`. Full `just install` not re-run this authoring pass (EDITOR-100 rebuilds the editor binary under test).

---

### Build & connectivity

#### - [ ] EDITOR-100 — Release editor binary builds

- **Priority:** P0 | **Mode:** agent | **Requires:** none
- **Description:** Produces `target/release/elodin` used by every screenshot case. Also confirms the editor crate (Bevy 0.19 stack) compiles.
- **Expected duration:** up to 45 min cold, ~2–5 min warm

**Steps**

```bash
mkdir -p /tmp/qa-editor
cargo build -p elodin --release 2>&1 | tee /tmp/qa-editor/EDITOR-100-build.log
test -x target/release/elodin
./target/release/elodin --version
```

**Pass criteria**

- [ ] `cargo build -p elodin --release` exits 0
- [ ] `target/release/elodin` is executable and `--version` prints a version string

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: `./target/release/elodin --version` → `elodin 0.17.4-alpha.0+5fdb9118.dirty` (release binary present; warm rebuild OK).

---

#### - [ ] EDITOR-101 — Connect editor to a live DB address

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Editor can attach to an already-running DB (`elodin editor <host:port>`) instead of spawning a sim from a file — the second startup path in §16.1.
- **Expected duration:** < 3 min

**Steps**

```bash
mkdir -p /tmp/qa-editor/EDITOR-101
rm -rf /tmp/qa-editor/EDITOR-101/db /tmp/qa-editor/EDITOR-101/live.png
lsof -tiTCP:2240 -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true
nix develop --command sh -c '
  set -e
  export ELODIN_DB_PATH=/tmp/qa-editor/EDITOR-101/db
  elodin run examples/ball/main.py > /tmp/qa-editor/EDITOR-101/run.log 2>&1 &
  echo $! > /tmp/qa-editor/EDITOR-101/run.pid
  for i in $(seq 1 60); do
    if lsof -iTCP:2240 -sTCP:LISTEN >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
  sleep 2
  ELODIN_SCREENSHOT=/tmp/qa-editor/EDITOR-101/live.png \
  ELODIN_SCREENSHOT_DELAY=12 \
  ELODIN_SCREENSHOT_EXIT=1 \
    "$PWD/target/release/elodin" editor 127.0.0.1:2240 \
    > /tmp/qa-editor/EDITOR-101/editor.log 2>&1
  status=$?
  # Group-kill: s10 restarts children on a plain kill.
  kill -9 "$(cat /tmp/qa-editor/EDITOR-101/run.pid)" 2>/dev/null || true
  pkill -9 -f "elodin run examples/ball" 2>/dev/null || true
  pkill -9 -f "examples/ball/main.py" 2>/dev/null || true
  exit $status
'
test -s /tmp/qa-editor/EDITOR-101/live.png
rg -n "screenshot written|listening|panic" /tmp/qa-editor/EDITOR-101/editor.log /tmp/qa-editor/EDITOR-101/run.log | head -20
```

**Pass criteria**

- [ ] Non-empty `/tmp/qa-editor/EDITOR-101/live.png`; editor log has `screenshot written`
- [ ] Screenshot shows CONNECTED and a live ball viewport (not the empty startup connect screen)
- [ ] Run log shows the DB listening; no panic/traceback in editor log

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: live.png 4.1 MB; CONNECTED ball viewport + RAM 0.4 GB; run.log listening on `[::]:2240`; editor log `screenshot written`.

---

#### - [ ] EDITOR-102 — Serve a recorded DB and open in editor

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Offline recording target in §16.1. Bench writes a DB, `elodin-db run` serves it, editor connects by address. (Direct `elodin editor <db-dir>` is still a CLI TODO/`ReplayDir` stub and panics — do **not** use that path.)
- **Expected duration:** < 4 min

**Steps**

```bash
mkdir -p /tmp/qa-editor/EDITOR-102
rm -rf /tmp/qa-editor/EDITOR-102/db /tmp/qa-editor/EDITOR-102/replay.png
lsof -tiTCP:2250 -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true
ELODIN_DB_PATH=/tmp/qa-editor/EDITOR-102/db \
  nix develop --command uv run python examples/ball/main.py bench --ticks 600 \
  > /tmp/qa-editor/EDITOR-102/bench.log 2>&1
test -d /tmp/qa-editor/EDITOR-102/db
nix develop --command sh -c '
  set -e
  elodin-db run "[::]:2250" /tmp/qa-editor/EDITOR-102/db \
    > /tmp/qa-editor/EDITOR-102/serve.log 2>&1 &
  echo $! > /tmp/qa-editor/EDITOR-102/serve.pid
  for i in $(seq 1 40); do
    if lsof -iTCP:2250 -sTCP:LISTEN >/dev/null 2>&1; then break; fi
    sleep 0.25
  done
  ELODIN_SCREENSHOT=/tmp/qa-editor/EDITOR-102/replay.png \
  ELODIN_SCREENSHOT_DELAY=12 \
  ELODIN_SCREENSHOT_EXIT=1 \
    "$PWD/target/release/elodin" editor 127.0.0.1:2250 \
    > /tmp/qa-editor/EDITOR-102/editor.log 2>&1
  status=$?
  kill -9 "$(cat /tmp/qa-editor/EDITOR-102/serve.pid)" 2>/dev/null || true
  exit $status
'
test -s /tmp/qa-editor/EDITOR-102/replay.png
rg -n "screenshot written|listening|panic" /tmp/qa-editor/EDITOR-102/editor.log /tmp/qa-editor/EDITOR-102/serve.log | head -20
```

**Pass criteria**

- [ ] Bench exits 0 and writes `/tmp/qa-editor/EDITOR-102/db`
- [ ] Non-empty replay screenshot; editor log contains `screenshot written`
- [ ] Screenshot shows CONNECTED and ball scene content from the recorded DB (not the empty startup screen)
- [ ] No panic in the editor log

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: bench 600 ticks → db; `elodin-db run "[::]:2250" <db>` (positional ADDR PATH); replay.png 4.1 MB shows ball trail TICK 600 CONNECTED RAM 0.4 GB. Note: direct `elodin editor <dir>` still panics (ReplayDir TODO).

---

### Viewport gallery (§16.2)

#### - [ ] EDITOR-110 — ball: viewport, trail, status bar, RAM

- **Priority:** P0 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Lightest full editor path: 3D viewport, `line_3d` trail, velocity vector label, view cube, timeline LIVE/tick, and status-bar FPS/TPS/**RAM** (§16.2 + §16.5).
- **Expected duration:** < 1 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh ball /tmp/qa-editor/EDITOR-110.png 12
rg -n "screenshot written|panic" /tmp/qa-editor/EDITOR-110.log | head -10
```

**Pass criteria**

- [ ] `capture.sh` exits 0; PNG non-empty
- [ ] Screenshot shows orange/brown ball, trajectory trail, and “Ball Velocity” (or equivalent vector label)
- [ ] View cube / navigation gizmo visible in the viewport
- [ ] Status bar shows CONNECTED, FPS, TPS, and `RAM Usage: X.Y GB` with X.Y > 0 (not `N/A`, not `0.0`)
- [ ] Timeline shows LIVE (or recent tick) without panic in the log

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: ball.png ~4.1 MB; Ball Velocity + trail + view cube; status RAM Usage: 0.4 GB (not N/A/0.0); FPS~60 TPS 120 LIVE TICK 1200.

---

#### - [ ] EDITOR-111 — three-body: multi-entity + graphs

- **Priority:** P0 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Multi-body viewport plus telemetry **graph** panels (§16.3 / §16.4) — validates GPU plots alongside 3D.
- **Expected duration:** < 1 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh three-body /tmp/qa-editor/EDITOR-111.png 15
```

**Pass criteria**

- [ ] Non-empty PNG; log has `screenshot written`
- [ ] Screenshot shows multiple bodies (a/b/c style) with orbital trails in the 3D view
- [ ] Schematic / entity tree (or graph tab such as `a.world_pos`) is visible alongside the viewport

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: three-body.png ~2.8 MB; three glowing bodies + golden trails; entity tree a/b/c; RAM 0.4 GB.

---

#### - [ ] EDITOR-112 — drone: GLB + joint animation

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** GLB `object_3d` with animated joints (spinning rotors) — §16.2 3D objects / animate joint.
- **Expected duration:** < 1.5 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh drone /tmp/qa-editor/EDITOR-112.png 18
```

**Pass criteria**

- [ ] Non-empty PNG; no panic in log
- [ ] Screenshot shows a quadrotor / drone mesh (not a fallback sphere-only scene)
- [ ] Viewport chrome (tabs / view cube) present

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: drone mesh + Drone X/Y/Z axes; three graph panels (angle_desired, World Pos, ang_vel_setpoint); TPS 300; RAM 0.5 GB.

---

#### - [ ] EDITOR-113 — rc-jet: FPV / aero viewport

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Fixed-wing example stressing FPV camera + plot_3d / HDR paths that historically crash on render-graph mistakes.
- **Expected duration:** < 1.5 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh rc-jet /tmp/qa-editor/EDITOR-113.png 18
```

**Pass criteria**

- [ ] Non-empty PNG; editor reached screenshot (did not abort on GPU pipeline error)
- [ ] Log has no `wgpu`/`RenderGraph` panic; `screenshot written` present
- [ ] Screenshot shows aircraft / runway / FPV-style view (not a solid pink/black error frame)

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: rc-jet aircraft + orange trail; FPV sensor_camera pane + AoA/Thrust plots; no wgpu panic; RAM 1.2 GB FPS~20.

---

#### - [ ] EDITOR-114 — apollo-lander: descent + thrusters

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** LM descent schematic with thruster / vector overlays (§16.2 icons & effects / SITL).
- **Expected duration:** < 2 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh apollo-lander /tmp/qa-editor/EDITOR-114.png 22
```

**Pass criteria**

- [ ] Non-empty PNG; `screenshot written` in log
- [ ] Screenshot shows the lunar module / descent scene (mesh or lit body above terrain/surface)
- [ ] No panic / fatal render error in log

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: LM descent + DPS/RCS plumes over lunar surface; 8 telemetry plots; Apollo epoch timestamp; RAM 1.1 GB. (CONNECTION ERROR seen once when sim lagged — scene still rendered.)

---

#### - [ ] EDITOR-115 — sensor-camera: GPU sensors / frusta

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** GPU-rendered sensor cameras and frustum-related viewport content (§16.2 frustum overlays / sensor_view).
- **Expected duration:** < 2 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh sensor-camera /tmp/qa-editor/EDITOR-115.png 22
```

**Pass criteria**

- [ ] Non-empty PNG; `screenshot written`
- [ ] Screenshot shows sensor/camera visualization (extra view panels, frustum lines, or dual RGB/thermal panes — not a blank single empty viewport)
- [ ] No panic in log

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: main viewport with frustum boxes + RGB Camera + Thermal panes; RAM 1.6 GB.

---

#### - [ ] EDITOR-116 — geo-frames: coordinate / terrain

- **Priority:** P2 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** ENU/NED/ECEF / geo framing and (when available) world_mesh terrain — §16.2 terrain & globes + §16.3 coordinate root.
- **Expected duration:** < 2 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh geo-frames /tmp/qa-editor/EDITOR-116.png 20
```

**Pass criteria**

- [ ] Non-empty PNG; `screenshot written`
- [ ] Screenshot shows a geo/orbit-style scene (globe/terrain/orbital path — not the ball example)
- [ ] No panic in log

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: textured Earth + ECEF/NED labels + orbit path; entity tree; RAM 0.5 GB.

---

#### - [ ] EDITOR-117 — cube-sat: spacecraft + graphs

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Spacecraft mesh + attitude/control graphs (MEKF/reaction wheels story) — §16.2 + §16.4.
- **Expected duration:** < 1.5 min

**Steps**

```bash
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh cube-sat /tmp/qa-editor/EDITOR-117.png 18
```

**Pass criteria**

- [ ] Non-empty PNG; `screenshot written`
- [ ] Screenshot shows a satellite / cube-sat style mesh in the viewport
- [ ] Graph or monitor panels present with series (when schematic includes them)

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: cube-sat mesh + css_value / att_est graphs; RAM 0.5 GB.

---

#### - [ ] EDITOR-118 — video-stream: H.264 tile decode

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Live H.264 decode into a video tile (§16.2 video in viewport + §17 decoding). `capture.sh` deletes stale `./video-stream-db` first. Needs a longer delay than other examples: s10 builds `elodinsink`, starts `videotestsrc`, and VideoToolbox must decode before the tile leaves “No video”.
- **Expected duration:** < 3 min

**Steps**

```bash
# Prefer nix develop so GStreamer + plugin build match CI.
nix develop --command sh -c '
  export ELODIN_BIN="$PWD/target/release/elodin"
  bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh video-stream /tmp/qa-editor/EDITOR-118.png 40
'
rg -n "screenshot written|VideoToolbox|test-video|panic" /tmp/qa-editor/EDITOR-118.log | head -30
```

**Pass criteria**

- [ ] Non-empty PNG; `screenshot written`
- [ ] Log shows `test-video` msg metadata and at least one `VideoToolbox:` decode line (H.264 reached the editor)
- [ ] Screenshot’s **Test Pattern** tab shows decoded video content (not the blue “No video at this time” placeholder)
- [ ] No Rust panic in the log (GStreamer scanner warnings about `libelodin.dylib` are OK)

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: needs delay≈40s under nix; log shows test-video + VideoToolbox NAL decode; Test Pattern tab shows clock/overlay (not “No video”); Wind graph + RAM 2.3 GB. Shorter delays often leave LOS placeholder.

---

#### - [ ] EDITOR-119 — voyager: deep-space + SPICE

- **Priority:** P2 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** Deep-space trajectory example requiring SPICE kernels under `examples/voyager/nasa_spice_data/`. Clears stale voyager DB dirs.
- **Expected duration:** < 3 min

**Steps**

```bash
test -d examples/voyager/nasa_spice_data
bash .cursor/skills/qa-test-plan/elodin-editor/capture.sh voyager /tmp/qa-editor/EDITOR-119.png 25
```

**Pass criteria**

- [ ] SPICE data directory exists (else BLOCKED with reason)
- [ ] Non-empty PNG; `screenshot written`
- [ ] Screenshot shows a space / trajectory scene (not a crash / empty connect screen)
- [ ] No panic; if log mentions DB time-travel, re-run once after DB delete (helper already deletes)

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: SPICE data present; Sun + orbital arcs + Jupiter; entity tree; RAM 0.6 GB; TPS 0 / TICK 0 early in long-horizon run is OK.

---

### Schematics & builds

#### - [ ] EDITOR-120 — --kdl schematic preload

- **Priority:** P2 | **Mode:** agent+visual | **Requires:** EDITOR-100
- **Description:** CLI `--kdl` preloads a schematic path (§16.1 / §16.3 document lifecycle).
- **Expected duration:** < 1.5 min

**Steps**

```bash
# Prefer a real on-disk schematic used by an example (embedded ball.kdl is not a file).
KDL=examples/drone/motor-panel.kdl
test -f "$KDL"
rm -f /tmp/qa-editor/EDITOR-120.png
ELODIN_SCREENSHOT=/tmp/qa-editor/EDITOR-120.png \
ELODIN_SCREENSHOT_DELAY=18 \
ELODIN_SCREENSHOT_EXIT=1 \
  ./target/release/elodin editor examples/drone/main.py --kdl "$KDL" \
  > /tmp/qa-editor/EDITOR-120.log 2>&1
test -s /tmp/qa-editor/EDITOR-120.png
rg -n "screenshot written|schematic|kdl|panic" /tmp/qa-editor/EDITOR-120.log | head -20
```

**Pass criteria**

- [ ] `--kdl examples/drone/motor-panel.kdl` was used and the editor captured a PNG
- [ ] Non-empty screenshot; editor did not panic on load
- [ ] Screenshot shows a populated schematic (viewport and/or motor panel content present)

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: `--kdl examples/drone/motor-panel.kdl`; log ingested schematics + Loaded window schematic; drone viewport+graphs screenshot OK.

---

#### - [ ] EDITOR-130 — Feature flag builds (inspector)

- **Priority:** P2 | **Mode:** agent | **Requires:** none
- **Description:** Optional cargo feature `inspector` still compiles against Bevy 0.19 (§16.1 editor dev loop features).
- **Expected duration:** up to 20 min cold, ~3 min warm

**Steps**

```bash
cargo check -p elodin --features inspector 2>&1 | tee /tmp/qa-editor/EDITOR-130-check.log
```

**Pass criteria**

- [ ] `cargo check -p elodin --features inspector` exits 0

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-19 @5fdb9118: `cargo check -p elodin --features inspector` Finished OK (1 deprecated CentralPanel warning in elodin-editor).

---

### Manual (§16.6 / §16.7 / §17 ingest)

#### - [ ] EDITOR-190 — Command palette interactions

- **Priority:** P1 | **Mode:** manual | **Requires:** EDITOR-110
- **Description:** §16.6 — invoke palette, filter, create a panel, toggle grid/HDR/wireframe.
- **Expected duration:** 5 min

**Steps**

1. `elodin editor examples/ball/main.py`
2. `Cmd/Ctrl+P` → type `Toggle Grid` → Enter; confirm grid toggles.
3. `Cmd/Ctrl+P` → `Create Graph` → confirm a graph tab appears.
4. Escape dismisses the palette.

**Pass criteria**

- [ ] Palette opens/filters/executes without freeze
- [ ] Grid toggle and Create Graph visibly change the UI

**Result:**
**Evidence:**
**Notes:**

---

#### - [ ] EDITOR-191 — Theme / color scheme switching

- **Priority:** P2 | **Mode:** manual | **Requires:** EDITOR-110
- **Description:** §16.7 color schemes / dark-light mode via palette or theme control.
- **Expected duration:** 3 min

**Steps**

1. Open ball in the editor.
2. Switch color scheme (palette `Set Color Scheme` or UI control) to a non-default (e.g. `eggplant` or `matrix`).
3. Toggle light/dark mode.
4. Confirm colors persist after restart (optional).

**Pass criteria**

- [ ] Scheme and mode changes are immediately visible on chrome + plots

**Result:**
**Evidence:**
**Notes:**

---

#### - [ ] EDITOR-192 — External video ingest (OBS/RTSP)

- **Priority:** P2 | **Mode:** manual | **Requires:** EDITOR-118
- **Description:** §17 OBS SRT / RTSP / webcam pipelines into `elodinsink` — needs external software/hardware.
- **Expected duration:** 15–30 min

**Steps**

1. Follow `docs` / video-stream README for OBS SRT caller → listener → `elodinsink`, **or** `rtsp-streamer` with `RTSP_URL`.
2. Open the editor on the live DB; confirm the video tile updates.
3. Scrub timeline; confirm video stays synchronized with telemetry.

**Pass criteria**

- [ ] Frames appear from the external source (not only `videotestsrc`)
- [ ] Scrubbing keeps video roughly aligned with telemetry

**Result:**
**Evidence:**
**Notes:**

---

## Run Summary

| Metric | Count |
|--------|-------|
| PASS | |
| FAIL | |
| BLOCKED | |
| SKIPPED | |

**Notable issues:**

**Follow-ups:**
