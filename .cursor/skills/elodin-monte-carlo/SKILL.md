---
name: elodin-monte-carlo
description: Develop and calibrate simulations against experimental truth data using elodin monte-carlo. Use when vendoring real telemetry as a reference profile, adding a truth-replay ghost entity, writing campaign specs/hooks and run scoring, reconstructing missing data channels from physics, or iterating on guidance and physics models with campaign feedback.
---

# Truth-Data-Driven Simulation with Elodin Monte Carlo

The most effective way to build a credible simulation is to anchor it to
experimental truth data and use `elodin monte-carlo` as the test harness: every
model change is judged by a 30-run campaign against recorded reality, in under
a minute. This skill codifies that workflow. The canonical worked example is
`examples/apollo-lander/` (with the full methodology in its `WHITEPAPER.md`).

## The development loop

```text
vendor truth data -> build reference module -> sim + truth ghost + graphs
      ^                                                    |
      |                                                    v
 narrow spec.toml <- read report <- run campaign <- score runs vs truth
                         |
                         v
              export worst run CSV -> diagnose time series -> fix model
```

Run the campaign after every meaningful change. Distribution deltas (RMSE,
success rate, margins, dispersion) tell you immediately whether a change
helped, broke something, or just moved noise.

## 1. Vendor truth data with provenance

- Keep the **verbatim source file** untouched in `data/` next to a derived,
  SI-unit version the sim actually loads. Cite the source URL in README/docs.
- Write a `sanity_check()` in the reference module that re-derives the cleaned
  data from the raw file and asserts agreement, plus checks documented anchor
  values. Make `python reference.py` print the profile and the check.
- **Distrust every column.** Real-world traps hit in practice: a "meters"
  column converted at x0.3405 instead of x0.3048 (recompute conversions from
  the original-unit column); digitized charts with out-of-order and duplicate
  timestamps (sort, average duplicates); head samples from a different
  measurement frame (a state-vector update step) that must be trimmed and
  back-extrapolated; a sensor channel that is not the state you want (radar
  *slant range* is not altitude).
- **Cross-validate datasets against each other and against documented
  events**: mission reports, transcripts, and timelines give anchor points
  (event times, velocity callouts, fuel consumed) that catch unit and frame
  errors no amount of smoothing will.

## 2. Build a dependency-free reference module

Put all truth handling in one stdlib-only Python module (see
`examples/apollo-lander/reference.py`) shared by the sim and any external
controller:

- Resample onto a uniform grid, despike with a median filter, smooth with a
  moving average. Only enforce monotonicity if the physics demands it — real
  profiles wiggle, and the wiggles are history worth keeping.
- Expose interpolators (`altitude(t)`, `descent_rate(t)`, ...) plus the raw
  arrays. In `sim.py`, convert once to `jnp.asarray` and close over them in
  systems: they become JIT-time constants (large baked constants are interned
  and shared across campaign workers).

## 3. Reconstruct missing channels from physics

Truth datasets are rarely complete. When a needed channel is missing (e.g.
horizontal velocity existed only as a chart image), reconstruct it:

- Integrate the vehicle dynamics along the channels you *do* have, using
  documented schedules (throttle history, event times) for the rest.
- Allocate consistently: if total thrust is known and the vertical share is
  implied by the recorded altitude, the horizontal share is the remainder —
  this makes the reconstructed profile **flyable by construction**, which
  matters when a controller later tracks it (see section 7).
- Calibrate segment-by-segment through documented anchors so the profile
  passes through known values exactly; integrate the best-documented segment
  unscaled and let it *yield* the unknown initial condition.
- Iterate fixed-point if channels couple (two passes usually converge).
- Document the uncertainty (~±10% is typical) and assert anchors in
  `sanity_check()`.

## 4. Truth ghost: replay reality inside the sim

Render the recorded vehicle next to the simulated one. The robust pattern:

```python
# Kinematic ghost: NO el.Body — gravity/integrator/telemetry systems never match it.
world.spawn([
    StaticSceneObject(el.WorldPos(...)),           # world_pos only
    el.C(TruthMarker, jnp.array([1.0])),           # scopes the playback query
    el.C(Altitude, ...), el.C(VerticalSpeed, ...), # graph channels
], name="lander_truth")

@el.system
def truth_playback(
    tick: el.Query[el.SimulationTick], truth: el.Query[TruthMarker]
) -> el.Query[el.WorldPos, Altitude, VerticalSpeed]:
    t_s = tick[0] * SIM_TIME_STEP
    alt = jnp.interp(t_s, ref_time, ref_altitude)
    ...
    return truth.map((el.WorldPos, Altitude, VerticalSpeed), lambda _: (...))
```

Hard-won rules:

- **Never give the ghost an `el.Body`.** If physics systems match it, gravity
  integrates its velocity while replay snaps its position — sawtooth motion,
  runaway velocities in the graphs, and corrupted exports.
- Do not drive the ghost from `pre_step` DB writes: explicit writes land on
  their own clock and interleave with telemetry commits, producing CSV exports
  with holes. The in-sim `el.SimulationTick` playback keeps truth on exactly
  the simulated vehicle's commit clock, so exports line up row-for-row.
- A marker component is the cheapest way to scope a playback query to one
  entity.

Put sim-vs-truth pairs on every KDL graph, and include raw measurement curves
(e.g. slant range next to true altitude) — visible divergence between a sensor
and the state teaches more than hiding it.

## 5. Campaign as the test harness

- `el.monte_carlo.params_spec(...)` defaults drive single runs;
  `spec.toml` drives campaigns. Keep ranges mirrored, and **anchor ranges in
  data** (a fuel chart pins the propellant load; navigation accuracy pins IC
  dispersions). When control authority is saturated (e.g. a full-throttle
  braking burn), keep IC ranges tight — the real system could not recover big
  errors either, and neither can yours.
- Score every run against truth in `post_step` and emit
  `el.monte_carlo.result(traj_rmse=..., pitch_rmse=..., miss_distance=...,
  soft_landing=...)`. Fit metrics (RMSE vs truth) are what turn a campaign
  from a stress test into a **calibration engine**: report the best-fit run's
  params, narrow `spec.toml` around them, repeat (or automate the loop, see
  `examples/apollo-lander/calibrate.py`).
- **Latch event metrics in-sim, at the event.** A touchdown-speed metric read
  one tick after contact reads the zeroed post-contact state; capture it in
  the same system that detects the event, before state is clobbered.
- Keep the LHS `seed` fixed while iterating so campaign-to-campaign deltas
  reflect your changes, not resampling.
- Use the `[build]` hook in `campaign.toml` to compile external FSW once
  before workers start. Put per-worker resources under `[resources.ports]` and
  read them with `el.monte_carlo.port("name", default)` or
  `ELODIN_MC_PORT_<NAME>`.

## 6. Diagnose with the data, not by staring at code

- Export any run and interrogate it with quick Python:

  ```bash
  elodin-db export dbs/<campaign>/runs/run_0000012/db \
      --format csv --join --flatten -o /tmp/run12
  ```

  Print a time-series table of the suspect channels (state, reference,
  command, actuator) every N seconds — most control/physics bugs are obvious
  in 20 rows.
- Rank runs by the failing metric (`post_run_result.json` per run) and export
  the worst one, not the best.
- **Treat "too perfect" as a bug.** Metrics that are identically zero or
  pegged to a clamp value usually mean the measurement is wrong (read after
  state-clobber, trivially satisfied criterion), not that the system is great.
- Verify exports are clean: row counts equal across entities, no empty cells,
  no physically impossible values. Data quality bugs masquerade as physics.
- Use the editor for visual regressions (attitude behavior, ghost overlap,
  terrain seating); confirm emergent event times against the historical
  record.

## 7. Tracking recorded profiles: control lessons

When a controller tracks the truth profile, the campaign will find every
weakness. Patterns that survived:

- **Feed-forward the reconstructed profile; bound the feedback.** Clamp each
  feedback channel's authority (e.g. ±0.8 m/s²) around the flyable
  feed-forward. Unbounded feedback lets channels fight over a saturated
  actuator and limit-cycle.
- If the feed-forward is *not* flyable (demands more than the actuator), the
  tracker oscillates no matter the gains — fix the reference (section 3), not
  the gains.
- Prefer **one unified tracking law with state-dependent limits** over
  discrete phase switches; let phases emerge (a speed-dependent tilt budget
  reproduces a pitchover; a demand-dependent throttle latch reproduces
  throttle-down).
- Add terminal logic: near the end, stop chasing the time-indexed reference
  and null residual errors — event timing shifts with small tracking errors,
  and arriving early must not mean arriving with the reference's residual
  velocity.
- Capped, slowly-fading position trim erodes IC offsets without destabilizing
  the velocity loop.

## 8. Operational pitfalls

- **Stale processes hold UDP ports.** Prefer direct `world.recipe()` sidecars
  plus s10 readiness probes. Linux campaigns use cgroup teardown to reap
  reparented daemons; `--keep-existing` opts out of scoped pre-reaping.
- Return `valid = False` from `post_run` hooks for infrastructure failures so
  they are reported separately from scored misses.
- CI hygiene per repo rules: `ruff format && ruff check --fix` for the Python,
  `cargo fmt`/`cargo clippy -- -Dwarnings` for external controllers.

## Reference example

`examples/apollo-lander/` exercises every pattern above: vendored NASA
telemetry with unit-bug handling and anchor checks, a physics-based
horizontal-profile reconstruction, a kinematic truth ghost on
`el.SimulationTick`, an external Rust LGC tracking the profile through a SITL
UDP bridge, campaign hooks producing fit metrics and landing-ellipse
statistics, and a calibration loop. Its `WHITEPAPER.md` documents the full
methodology and the honest caveats.
