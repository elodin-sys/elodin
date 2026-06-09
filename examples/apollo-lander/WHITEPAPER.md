# Flying Apollo 11 in Elodin

*A from-scratch, software-in-the-loop recreation of the Apollo 11 powered descent — and a guided tour of how to build a real spacecraft simulation in [Elodin](https://www.elodin.systems/).*

---

## Who this is for

This is an educational explainer for aspiring spacecraft engineers, GNC (guidance,
navigation & control) students, and space enthusiasts. It walks through **every**
piece of the `apollo-lander` example: where the historical data came from, how we
turned noisy 1969 telemetry into a usable reference trajectory, the physics and
flight-control math we implemented, which Elodin features made it possible, and
exactly where each idea lives in the code.

You do not need a background in aerospace to follow along, but you will get the
most out of it if you are comfortable with vectors, basic calculus, and a little
linear algebra. Every formula is paired with a link to the line of code that
implements it.

> **A note on fidelity.** This is a *teaching* simulation, not a flight-certified
> reconstruction. We deliberately favor clarity over completeness and flag every
> simplification. Section 11 collects the honest caveats in one place.

## Table of contents

1. [The historical event](#1-the-historical-event)
2. [System architecture](#2-system-architecture)
3. [Data and provenance](#3-data-and-provenance)
4. [The flight dynamics model](#4-the-flight-dynamics-model)
5. [Composing the simulation with `el.six_dof`](#5-composing-the-simulation-with-elsix_dof)
6. [The guidance law (the "Lunar Guidance Computer")](#6-the-guidance-law-the-lunar-guidance-computer)
7. [Software-in-the-loop: the bridge and lockstep](#7-software-in-the-loop-the-bridge-and-lockstep)
8. [Visualization: the KDL schematic and model scaling](#8-visualization-the-kdl-schematic-and-model-scaling)
9. [Monte Carlo: robustness and calibration](#9-monte-carlo-robustness-and-calibration)
10. [Elodin features used](#10-elodin-features-used)
11. [Modeling decisions and honest caveats](#11-modeling-decisions-and-honest-caveats)
12. [Running and exploring the example](#12-running-and-exploring-the-example)
13. [Exercises for the reader](#13-exercises-for-the-reader)
14. [References](#14-references)

---

## 1. The historical event

On July 20, 1969, the Apollo 11 Lunar Module *Eagle* separated from the Command
Module and began its **powered descent** to the Moon's Sea of Tranquility. A
single throttleable rocket — the Descent Propulsion System (DPS) — slowed the
vehicle from orbital speed to a hover, while 16 small Reaction Control System
(RCS) thrusters held its attitude. Roughly twelve minutes later, Neil Armstrong
took manual control and set *Eagle* down with seconds of fuel margin.

The powered descent is a near-perfect teaching problem:

- **The physics is approachable.** Lunar gravity is a constant `1.62 m/s²`, there
  is no atmosphere (no drag, no wind), and the dominant forces are gravity and a
  single engine.
- **The control problem is real.** The vehicle must simultaneously null its
  horizontal velocity, track a descent profile, manage a shifting center of mass
  as propellant burns, and arrive nearly vertical and slow enough that the
  landing gear survives.
- **The data exists.** NASA's postflight telemetry and 3D assets are public, so
  we can fly *against the real mission* and measure how close we get.

Our simulation clock is set to the real epoch of the first telemetry sample,
`1969-07-20T20:09:53.164Z`, and the simulated *Eagle* flies next to a green
"truth" vehicle that replays the recorded descent. The goal of the example is to
land softly **and** to match the historical trajectory.

---

## 2. System architecture

The example is split into four cooperating pieces. The **plant** (the physics)
lives in Elodin and runs in JAX; the **flight software** (the guidance law) runs
as a *separate process* — exactly as it would on a real vehicle, where the
autopilot is its own computer talking to sensors and actuators over a bus.

```mermaid
flowchart LR
    subgraph SIM["sim.py — plant (Elodin / JAX, 120 Hz)"]
        PHYS["el.six_dof integrator<br/>gravity · DPS thrust · RCS torque<br/>mass & inertia burn-down"]
        TRUTH["truth_playback system<br/>replays the recorded descent<br/>on the kinematic green ghost"]
    end
    subgraph BR["main.py — SITL bridge + harness"]
        POST["post_step():<br/>read state, exchange with FSW,<br/>write throttle + attitude"]
    end
    LGC["controller/ — flight software (Rust)<br/>'Lunar Guidance Computer'<br/>reference-tracking descent law"]
    DB[("Elodin-DB<br/>time-series telemetry")]
    ED["Elodin editor<br/>KDL viewport + live graphs"]

    PHYS <--> POST
    POST -- "UDP: 15×f64 state" --> LGC
    LGC -- "UDP: 6×f64 command" --> POST
    PHYS --> DB --> ED
    TRUTH --> DB
```

**The control loop, once per guidance tick:**

1. The physics integrates one or more steps and publishes the vehicle state.
2. `post_step` reads the kinematics and packs them into a UDP datagram.
3. The Rust controller receives the state, runs the guidance law, and replies
   with a throttle setting and a desired attitude quaternion.
4. `post_step` writes those commands back into the simulation as
   *external-control* components.
5. In-sim JAX systems convert the commands into forces and torques, and the
   integrator advances the state.

This is the essence of **software-in-the-loop (SITL)**: the real flight-control
code flies a simulated vehicle, with the simulator standing in for the physical
world and sensors.

### Repository layout

| File | Role |
| --- | --- |
| [`sim.py`](sim.py) | World definition: components, 6-DOF physics systems, truth replay, KDL schematic |
| [`main.py`](main.py) | Entry point: SITL bridge, scoring, `world.run` |
| [`reference.py`](reference.py) | Turns raw telemetry into a smooth descent reference (stdlib only) |
| [`controller/src/main.rs`](controller/src/main.rs) | The flight software: the guidance law, in Rust |
| [`spec.toml`](spec.toml) | Monte Carlo parameter distributions |
| [`campaign.toml`](campaign.toml) | Campaign config: ports, hooks, one-time `[build]` step |
| [`hooks/score.py`](hooks/score.py) | Per-run pass/fail scoring |
| [`hooks/report.py`](hooks/report.py) | Post-campaign aggregate report |
| [`calibrate.py`](calibrate.py) | Optional automated range-narrowing loop |
| [`data/`](data/) | Raw + derived Apollo 11 telemetry, LM spec sheet |

---

## 3. Data and provenance

Everything physical in this example traces back to a public NASA source.

### 3.1 Telemetry

The descent telemetry comes from the community-maintained transcription of NASA
postflight data at
[`jumpjack/Apollo11LEMdata`](https://github.com/jumpjack/Apollo11LEMdata/blob/master/data.csv).
We keep two copies under [`data/`](data/):

- **[`apollo11_lem_raw.csv`](data/apollo11_lem_raw.csv)** — the verbatim source.
  Each row is a timestamp (`YYMMDDHHMMSS.SSS`), the three IMU **stable-member
  gimbal angles** (inner/middle/outer, in degrees), and the landing-radar
  **slant range** (`RANGE (FT)`).
- **[`apollo11_descent.csv`](data/apollo11_descent.csv)** — a cleaned,
  SI-unit derivation with columns `timestamp_utc, time_s, range_m, inner_deg,
  middle_deg, outer_deg`. Range is converted feet → meters (`× 0.3048`) and time
  is made relative to the first sample.

The very first row fixes our simulation epoch and initial altitude: `t = 0` at
`1969-07-20T20:09:53.164Z`, range ≈ `13,336 m`.

### 3.2 Vehicle mass and propulsion

The mass and engine numbers come from the LM spec sheet in
[`data/lunar_module_spec_sheet.pdf`](data/lunar_module_spec_sheet.pdf):

| Quantity | Value | Used in |
| --- | --- | --- |
| Wet mass (full descent stack) | ≈ 15,065 kg | context |
| Descent-stage dry mass (modeled) | ≈ 6,853 kg | `dry_mass_kg` |
| Descent propellant (full load) | ≈ 8,212 kg | context |
| Propellant at telemetry-window start (modeled) | ≈ 4,100 kg | `propellant_kg` |
| DPS thrust range | 4,670 – 45,040 N (throttleable) | `DPS_MIN/MAX_THRUST_N` |
| DPS specific impulse | ≈ 311 s | `isp_s` |
| RCS thrusters | 16 × 445 N | `RCS_THRUST_N` |
| RCS specific impulse | ≈ 290 s | `RCS_ISP_S` |

The telemetry window opens at ~13 km altitude, after the braking phase had
already burned roughly half of the DPS load — so the simulated vehicle starts
at ≈ 11,200 kg wet (`dry + DPS propellant + RCS propellant`), not the full
15,065 kg. These constants are defined at the top of [`sim.py`](sim.py#L20-L32).

### 3.3 3D assets

Three official NASA glTF models are rendered in the editor:

- **Lunar Module** — [science.nasa.gov/3d-resources/apollo-lunar-module](https://science.nasa.gov/3d-resources/apollo-lunar-module/)
- **Apollo 11 landing site** — [science.nasa.gov/3d-resources/apollo-11-landing-site](https://science.nasa.gov/3d-resources/apollo-11-landing-site/)
  (a 30 km × 30 km height map of the Sea of Tranquility, vertical exaggeration 60×).
- **Moon sphere** — [NASA SVS Moon 3D Models for Web, AR, and Animation](https://svs.gsfc.nasa.gov/14959/)
  (a Lunar Reconnaissance Orbiter imagery/topography model used as the surrounding
  curved lunar ground and horizon).

How we recovered their units and chose the right scale is covered in
[Section 8](#8-visualization-the-kdl-schematic-and-model-scaling).

### 3.4 From noisy telemetry to a clean reference

Raw 1969 landing-radar range is **noisy and non-monotonic** — it spikes, and it
occasionally reports the vehicle *climbing*. We can't guide against that directly,
and we can't use it as "truth" without cleaning it. [`reference.py`](reference.py)
turns the raw measurements into a smooth, physically sensible descent profile
using only the Python standard library (so it has no heavy dependencies and can be
shared by both the simulation and the controller).

The pipeline, implemented in [`build_reference`](reference.py#L131-L170):

1. **Resample** range onto a uniform 1-second grid by linear interpolation
   ([`interp`](reference.py#L72-L83)).
2. **Despike** with a 5-sample median filter ([`_median_filter`](reference.py#L86-L96)),
   which rejects isolated radar glitches without blurring the trend.
3. **Smooth** the altitude with a moving average ([`_moving_average`](reference.py#L99-L110)).
4. **Enforce monotonic descent** — altitude can only ever decrease:

   ```text
   running = +∞
   for each sample a:
       running = min(running, a)
       altitude = max(running, 0)
   ```

5. **Remove the terminal antenna offset** — at touchdown the landing radar still
   reads the antenna's height above the footpads (~4 m), so the cleaned profile
   bottoms out high. Subtracting the terminal value makes the profile reach zero
   exactly at the recorded touchdown time.

6. **Differentiate** for descent rate using a centered difference, then smooth:

   ```text
   rate[i] = (alt[i+1] − alt[i−1]) / (t[i+1] − t[i−1])
   ```

7. **Reconstruct an attitude trend** from the dominant inner gimbal angle,
   smoothed and sign-flipped to a pitch-from-vertical proxy.

The result is exposed as a [`Reference`](reference.py#L113-L128) object with
`altitude(t)`, `descent_rate(t)`, and `pitch(t)` interpolators. A built-in
[`sanity_check()`](reference.py#L173-L214) re-derives the range straight from the
raw file and asserts it matches the cleaned data to sub-millimeter tolerance — run
`python reference.py` to print the profile and the check.

> **Why share one reference module?** The same smoothed profile is the *target*
> for the guidance law **and** the *truth* drawn in green in the editor. Using one
> source of truth keeps "what we asked for" and "what really happened" directly
> comparable.

---

## 4. The flight dynamics model

All of the physics is defined in [`sim.py`](sim.py) as small JAX functions and
composed into a rigid-body integrator. This section walks through each force and
the math behind it.

### 4.1 Coordinate frame and state

The world uses an **ENU** frame (East, North, Up). Altitude is the world `+Z`
component, so the ground is the plane `z = 0`. The vehicle's body `+Z` axis is its
thrust ("up") axis; tilting the body steers the engine.

The vehicle is an [`el.Body`](sim.py#L226-L263), which carries the standard 6-DOF
state Elodin needs:

- `world_pos` — an `el.SpatialTransform` (orientation quaternion + position)
- `world_vel` — an `el.SpatialMotion` (angular + linear velocity)
- `inertia` — an `el.SpatialInertia` (mass + diagonal inertia tensor)

On top of that we attach domain components (defined at
[`sim.py#L87-L159`](sim.py#L87-L159)) such as `Altitude`, `VerticalSpeed`,
`Throttle`, `Propellant`, `RcsTorque`, and the two **external-control** inputs the
flight software writes:

```python
ThrottleCmd   # F64, metadata {"external_control": "true"}
AttitudeSetpoint  # Quaternion, metadata {"external_control": "true"}
```

Marking a component `external_control` tells Elodin its value comes from outside
the physics graph — here, from the SITL bridge.

### 4.2 Gravity

Lunar gravity is a constant downward force scaled by the *current* mass
([`lunar_gravity`](sim.py#L334-L338)):

```text
F_gravity = (0, 0, −g_moon · m)        g_moon = 1.622 m/s² × gravity_scale
```

Because it reads the live `inertia.mass()`, gravity automatically tracks the
vehicle getting lighter as it burns propellant.

### 4.3 The descent engine (throttle, lag, and thrust)

A real engine cannot change thrust instantaneously. [`engine_response`](sim.py#L289-L298)
models the throttle as a **first-order lag** toward the commanded value:

```text
T_cmd  = clip(throttle_cmd, u_min, 1)          u_min = 4670/45040 ≈ 0.104
τ      ← τ + (T_cmd − τ) · α                    α = clip(f_response · Δt, 0, 1)
thrust = τ · T_max · thrust_scale               (0 if out of fuel or landed)
```

`α` is the per-step blend factor; a `throttle_response_hz` of 3 Hz at the 120 Hz
step gives `α ≈ 0.025`, i.e. a ~⅓-second time constant. The throttle floor `u_min`
reflects the DPS's real minimum thrust — the engine cannot be commanded below it.

The thrust acts along body `+Z` and is rotated into the world by the attitude
quaternion `q` ([`apply_main_thrust`](sim.py#L340-L343)):

```text
F_thrust = q ⊗ (0, 0, thrust)
```

This single line is why **attitude is steering**: tilt the vehicle and the same
engine now has a horizontal thrust component to cancel cross-track velocity.

### 4.4 Mass and inertia burn-down

Propellant mass follows the classic rocket mass-flow relation (the differential
form of the Tsiolkovsky equation), in [`mass_props`](sim.py#L300-L320):

```text
ṁ = T / (Isp · g₀)            g₀ = 9.80665 m/s²   (standard gravity)
Δm_DPS = T / (Isp · g₀) · Δt
```

RCS propellant is tracked separately by converting commanded torque back to an
equivalent thruster force (`|τ| / moment_arm`) and burning at the RCS `Isp`. The
live mass then rescales the inertia tensor so rotational dynamics stay consistent:

```text
m = dry_mass + propellant_DPS + propellant_RCS     (remaining propellant masses)
I = I_base · (m / m₀)          (a large "locked" inertia is used once landed)
```

> **Real LM subtlety we approximate:** on the real vehicle the center of mass
> *moved* as the spherical tanks drained, and the engine gimbaled to track it. We
> keep the CoM fixed and let RCS provide control torque — see Section 11.

### 4.5 Attitude control (quaternion-error PD)

The RCS holds the vehicle on the attitude the flight software requests. The
in-sim controller ([`attitude_control`](sim.py#L322-L332)) is a
**proportional-derivative (PD) law on the quaternion error**:

```text
q_err = q⁻¹ ⊗ q_setpoint                         (rotation from current to target)
sign  = +1 if q_err.w ≥ 0 else −1                (take the shortest path)
ω_body = q⁻¹ · ω_world                           (body-frame angular rate)
τ = sign · q_err.xyz ⊙ k_p  −  ω_body ⊙ k_d      (⊙ = per-axis product)
τ = clip(τ, −τ_limit, +τ_limit)                  (RCS authority limit)
```

The intuition: for small errors the vector part of a quaternion is approximately
half the rotation-angle times the rotation axis, so `q_err.xyz` is a clean
proportional error signal. The `−ω_body · k_d` term is damping. The per-axis
torque is then clipped to the available RCS authority,
`4 × 445 N × 2 m ≈ 3,560 N·m`, and applied as a body torque rotated into the world
([`apply_rcs_torque`](sim.py#L345-L347)). The proportional gains scale with the
Monte Carlo `attitude_gain` parameter.

### 4.6 Ground contact

[`ground_contact`](sim.py#L349-L379) latches a landing the first time altitude
crosses zero. On that contact tick — *before* the velocity is zeroed — it records
the vertical impact speed `|v_z|` as `touchdown_speed` and the horizontal impact
speed `‖v_xy‖` as `touchdown_horizontal_speed`, then pins the vehicle: position
`z = 0`, linear and angular velocity zeroed. This is a simple "perfectly
inelastic" stop — enough to score the landing without modeling gear mechanics.
(Latching at contact matters: one tick later the zeroed velocity would score
every landing as a perfect `0 m/s` touchdown.)

### 4.7 Derived telemetry

[`derive_telemetry`](sim.py#L381-L392) computes the human-readable signals shown
in the graphs: altitude and vertical speed (the `z` components), horizontal speed
(`‖v_xy‖`), and pitch-from-vertical:

```text
body_up = q · (0, 0, 1)
pitch   = arccos(clip(body_up_z, −1, 1))    (angle between thrust axis and "up")
```

### 4.8 The truth ghost (in-sim replay)

The green `lander_truth` vehicle is **kinematic**: it is spawned without an
`el.Body` ([`sim.py#L265-L281`](sim.py#L265-L281)), so gravity, the integrator,
and the telemetry-derivation systems never touch it. A dedicated playback
system, [`truth_playback`](sim.py#L408-L434), reads the built-in
`el.SimulationTick`, interpolates the cleaned reference (altitude, descent rate,
pitch trend) with `jnp.interp`, and writes the ghost's pose and telemetry every
tick. A `TruthMarker` component scopes the system's query to the ghost, so the
playback never collides with the simulated lander's physics.

Because the replay runs inside the compiled simulation, the reference arrays
become JIT-time constants, the replay costs a few interpolations per tick, and
the truth telemetry lands on exactly the same 40 Hz commit clock as the
simulated vehicle — so exports line up row-for-row.

---

## 5. Composing the simulation with `el.six_dof`

Elodin builds simulations by **composing small systems** with the `|` operator.
Each `@el.map` function declares the components it reads and writes; Elodin wires
them into a dependency graph and compiles the whole thing (via JAX → StableHLO →
native code) so the per-step physics runs with no Python in the hot loop.

The assembly lives at the end of [`build`](sim.py#L438-L447):

```python
non_effectors = engine_response | attitude_control | mass_props | thrust_visualization
effectors     = lunar_gravity | apply_main_thrust | apply_rcs_torque
system = (
    truth_playback
    | non_effectors
    | el.six_dof(sys=effectors, integrator=el.Integrator.SemiImplicit)
    | ground_contact
    | derive_telemetry
)
```

Read it top to bottom as the per-tick pipeline:

1. **`truth_playback`** replays the recorded descent on the kinematic ghost
   (Section 4.8).
2. **Non-effectors** update throttle, attitude torque, mass/inertia, and the
   visualization vectors.
3. **`el.six_dof`** is the heart of the rigid-body simulation. You hand it a set
   of *effector* systems that accumulate into the `el.Force` (linear + torque)
   component; it integrates translation and rotation together, handling the
   quaternion kinematics for you. We use the **semi-implicit Euler** integrator,
   which is stable and cheap for this kind of problem.
4. **`ground_contact`** clamps the state at touchdown.
5. **`derive_telemetry`** publishes the display signals.

The key idea: **you never write an integrator.** You describe forces and torques
as pure functions of state, and `el.six_dof` does the calculus. Earlier drafts of
this example hand-rolled Euler integration and suffered jitter and stalls;
switching to `el.six_dof` made the descent smooth and let JAX compile the math.

---

## 6. The guidance law (the "Lunar Guidance Computer")

The flight software is a standalone Rust program in
[`controller/src/main.rs`](controller/src/main.rs). It never imports Elodin — it
just reads a state datagram and returns a command, like a real autopilot reading
sensors and driving actuators. Writing it in Rust keeps it fast and deterministic
in lockstep with the simulation.

Its job is **reference-tracking guidance**: follow the cleaned Apollo descent
profile while nulling horizontal drift and staying upright. The law is a cascade
of simple, well-behaved loops ([`command`](controller/src/main.rs#L99-L115)).

### 6.1 Outer loop — altitude to descent-rate command

Track the reference altitude by nudging the commanded descent rate proportionally
to the altitude error, around the reference rate (feed-forward):

```text
ḣ_cmd = clip( ḣ_ref + k_track · (h_ref − h),  −ḣ_max,  −ḣ_min )
```

with `ḣ_max = 120 m/s` and `ḣ_min = 0.5 m/s`. The clamp guarantees the vehicle
is always commanded to descend, never to climb. `ḣ_max` sits *above* the
reference's initial ~97 m/s range-rate — if it were lower, guidance could never
acquire the profile from above and would limit-cycle on the throttle. `ḣ_min`
is the terminal contact rate; the real LM touched down at roughly 0.5 m/s.

### 6.2 Middle loop — descent-rate to vertical acceleration

A proportional loop on the rate error, with gravity fed forward so the engine
first cancels weight:

```text
a_z = max( g + k_vert · (ḣ_cmd − ḣ),  a_z,min )
```

### 6.3 Horizontal nulling and the tilt cone

Cancel horizontal velocity with a proportional law, then **limit the tilt** so the
vehicle never commands an aggressive (or inverted) attitude
([`clamp_horizontal`](controller/src/main.rs#L88-L97)):

```text
a_x = −k_horiz · v_x
a_y = −k_horiz · v_y
‖(a_x, a_y)‖ ≤ a_z · tan(θ_max)          θ_max = 30°
```

Constraining the horizontal acceleration relative to the vertical acceleration
keeps the desired thrust vector inside a 30° cone around "up." This is the fix for
a subtle but catastrophic failure mode: without it, guidance could ask the
vehicle to point its engine sideways or even flip over. If the vehicle needs to
fall faster, it throttles down — it never inverts.

### 6.4 Thrust magnitude to throttle

The desired acceleration vector sets the required thrust; divide by available
thrust to get a throttle command:

```text
a       = (a_x, a_y, a_z)
T_req   = m · ‖a‖
throttle = clip( T_req / (T_max · thrust_scale),  u_min,  1 )
```

### 6.5 Desired acceleration to attitude

Finally, convert the desired acceleration *direction* into an attitude: the body
`+Z` axis should point along `a`. [`quat_from_body_z`](controller/src/main.rs#L76-L86)
builds the **shortest-arc quaternion** that rotates `+Z` onto the unit
acceleration vector (with a guard for the degenerate straight-down case).

The controller returns six doubles: `throttle`, the four quaternion components,
and the commanded descent rate (for logging). The bridge applies one more safety
layer — a **slew limit** of ≤ 8° per update
([`_slew_quat`](main.py#L129-L143)) — so the attitude target moves smoothly even
if guidance changes its mind abruptly.

> **The full cascade in one breath:** altitude error → rate command → vertical
> acceleration; horizontal velocity → horizontal acceleration (cone-limited);
> together they form a thrust vector → throttle + attitude. Every step is a
> first-order loop with an interpretable gain, which is exactly why it is a good
> teaching example.

---

## 7. Software-in-the-loop: the bridge and lockstep

[`main.py`](main.py) is the harness that connects the plant to the flight
software. Elodin exposes `pre_step`/`post_step` callbacks around each physics
step; the example needs only one, and keeps it deliberately thin so the physics
stays in JAX and the only Python work per tick is moving bytes:

- [`post_step`](main.py#L146-L242) — after the step, **read** the simulated
  kinematics, exchange them with the controller over UDP, and **write** the
  resulting `throttle_cmd` and `attitude_setpoint`.

(The truth vehicle needs no callback at all — it is replayed *inside* the
compiled simulation by the `truth_playback` system, Section 4.8.)

### 7.1 The wire protocol

The [`SitlBridge`](main.py#L41-L82) speaks a fixed binary protocol — little-endian
`f64` arrays packed with Python's `struct` and Rust's `to_le_bytes`. Fixed-size
packets mean no parsing ambiguity and no allocation in the loop.

**State (sim → controller), 15 × f64:**

| # | Field | # | Field | # | Field |
| --- | --- | --- | --- | --- | --- |
| 0 | `time_s` | 5 | `world_vel.z` | 10 | `max_thrust` |
| 1 | `altitude` | 6 | `mass` | 11 | `thrust_scale` |
| 2 | `vertical_speed` | 7 | `ref_alt` | 12 | `track_gain` |
| 3 | `world_vel.x` | 8 | `ref_rate` | 13 | `vertical_gain` |
| 4 | `world_vel.y` | 9 | `gravity` | 14 | `horizontal_gain` |

**Command (controller → sim), 6 × f64:** `throttle`, `q_x`, `q_y`, `q_z`, `q_w`,
`rate_cmd`.

Notice the *gains* are sent in the state packet: the Monte Carlo sampler chooses
them per run, so the flight software is re-tuned by the campaign without
recompiling.

### 7.2 Rates and timing

| Clock | Rate | Notes |
| --- | --- | --- |
| Physics step | 120 Hz | `SIMULATION_RATE_HZ` |
| Guidance update | 24 Hz | every 5th tick; matches a modest autopilot rate |
| Truth replay | 120 Hz | in-sim `truth_playback` system, every tick |
| Telemetry to DB | 40 Hz | `telemetry_rate` |
| Editor playback | 30× real time | `default_playback_speed` |

Running guidance slower than the physics is realistic (autopilots run at tens of
Hz, not kHz) and keeps the SITL exchange cheap. The run is configured at
[`world.run`](main.py#L245-L256), which also sets the historical
`start_timestamp` so every telemetry sample is stamped with its 1969 wall-clock
time. Elodin-DB timestamps native writes from the simulation clock, so no manual
time component is needed.

### 7.3 One controller, two launch modes

The same Rust program is launched two ways via an `s10` recipe
([`main.py#L107-L119`](main.py#L107-L119)):

- **Single run** (editor/headless): `el.s10.PyRecipe.cargo(...)` builds and runs
  the controller straight from source — convenient while iterating.
- **Monte Carlo**: the campaign's one-time `[build]` step compiles the release
  binary once, and each worker launches the prebuilt binary with
  `el.s10.PyRecipe.process(...)`. Per-run ports are passed through the
  environment so parallel workers never collide.

---

## 8. Visualization: the KDL schematic and model scaling

Elodin describes its 3D scene and dashboards declaratively in **KDL**
([`apollo-lander.kdl`](apollo-lander.kdl), registered by the
[`world.schematic`](sim.py#L436) call). The schematic lays out:

- a **viewport** ("Tranquility Base") that follows the lander
  (`pos="lander.world_pos.translate_world(10.0, 10.0, 4.0)" look_at="lander.world_pos"`);
- six live **graphs** comparing the simulated vehicle to truth — altitude,
  descent rate, horizontal speed, pitch, throttle, and propellant;
- four **`object_3d`** GLB models: the landing site, the simulated LM, the green
  truth LM, and a full Moon sphere used as the curved horizon backdrop;
- **`line_3d`** trajectory trails (blue = simulated, green = truth);
- **`vector_arrow`** overlays for DPS thrust (orange) and RCS torque (white).

The whole layout uses `coordinate frame="ENU"`, so what you see matches the math.

### 8.1 Figuring out the model units

The GLB models ship with no documented units, and a wrong scale made the lander
visually sink through the terrain. We recovered the real units by reading the glTF
geometry directly (walking the node graph and reading each mesh's `POSITION`
accessor bounding box):

- **Lunar Module** — bounding box ≈ 6.4 m wide, 5.0 m tall, **Y-up**. That is
  glTF's default *meters*, and it matches the real LM (~7 m tall, ~9.4 m gear
  span) to within model fidelity. So [`LANDER_GLB_SCALE = 1.0`](sim.py#L45-L48)
  renders it ~life-size.
- **Landing site** — bounding box 255.5 × 255.5 native units (a ~256-sample
  height-map grid), **Z-up**, with relief spanning ~18.7 units. NASA documents the
  tile as 30 km × 30 km with elevation exaggerated 60×, so **255.5 units ↔
  30,000 m** (≈ 117.4 m/unit). Hence:

  ```text
  TERRAIN_GLB_SCALE = 30000 / 255.5 ≈ 117.4
  ```
- **Moon sphere** — bounding box ≈ 1.94 native units across, centered near the
  origin. We render it at lunar scale using the Moon's mean radius, `1,737.4 km`.
  Since its native radius is ≈ 0.97 units, the KDL uses `scale ≈ 1.8e6`.

### 8.2 Seating the terrain at the ground plane

The height-map's center (the landing point) sits at native elevation ≈ 10.63
units. At the old `scale = 1000` that put the rendered surface ~10,630 m above the
origin — which is exactly why the lander appeared to pass through it while still
kilometers up. The fix ([`sim.py#L49-L59`](sim.py#L49-L59)) is to scale to the
true size **and** lower the whole terrain entity so the landing-point surface sits
at world `z = 0`:

```text
TERRAIN_SEAT_Z = − TERRAIN_GLB_SCALE × TERRAIN_CENTER_NATIVE_Z ≈ −1248 m
```

The `rotate="(-90, 0, 0)"` in the schematic stands the natively Z-up tile upright
in the editor's Y-up render space. Because Elodin's GLB `scale` is a single
uniform factor, the 60× vertical exaggeration cannot be undone here — distant
relief renders too tall — but the immediate landing zone is flat and correctly
seated. (See the [README](README.md) for the knob to tighten the scene.)

> **Lesson:** never trust an asset's scale by eye. A few minutes reading the
> bounding box turns "it looks about right" into a number you can defend.

### 8.3 Adding the Moon-scale horizon

The 30 km landing-site tile provides the local surface detail, but it ends before
the horizon. To give the scene lunar curvature and distant ground, the KDL also
places NASA SVS's LRO Moon GLB around the landing area.

Seating it takes care: the mesh is *not* a perfect sphere — it carries real LRO
topography (vertex radius 0.960–0.973 native units, ±12 km at lunar scale), so
"one mean radius down" can leave the local surface kilometers high or low.
Measuring the transformed mesh directly (the triangle that the world `z` axis
pierces, with the KDL's rotation and `scale = 1,798,000`) puts the under-site
surface ≈ 1,725,022 m above the sphere center. The KDL therefore uses:

```text
moon_center_z = -1,726,250 m
→ local moon surface ≈ −1,228 m
```

i.e. about 1.2 km *below* the touchdown plane — safely beneath the landing-site
tile's 60×-exaggerated valleys (deepest ≈ −1,116 m), so the emissive Moon never
pokes through the near-field terrain. The landing-site GLB remains the precise
near-field surface; the full Moon GLB is the surrounding curved horizon and
visual backdrop.

---

## 9. Monte Carlo: robustness and calibration

The example uses `elodin monte-carlo` for two distinct purposes: proving the
design is **robust** to uncertainty, and **calibrating** it against the real
mission.

### 9.1 The parameter space

[`spec.toml`](spec.toml) declares 16 uncertain parameters sampled by **Latin
Hypercube** (`method = "lhs"`, `n_samples = 30`, `seed = 19690720`). They fall
into three groups:

- **Initial conditions** — altitude, vertical/downrange/crossrange speed, pitch.
- **Vehicle uncertainty** — dry mass, propellant, RCS propellant, thrust scale
  factor, `Isp`, a small gravity scale.
- **Controller gains** — `track_gain`, `vertical_gain`, `horizontal_gain`,
  `attitude_gain`, `throttle_response_hz`.

The Python side mirrors these as a typed [`params_spec`](sim.py#L61-L85) with
defaults and bounds, so the same simulation runs standalone (defaults) or under a
campaign (sampled).

### 9.2 Scoring a run

Each run emits a result via `el.monte_carlo.result(...)`
([`main.py#L217-L242`](main.py#L217-L242)). A landing counts as **soft** when:

```text
landed AND
touchdown_speed ≤ 3 m/s             (vertical, latched at contact)  AND
touchdown_horizontal_speed ≤ 1 m/s  (latched at contact)            AND
upright_dot ≥ 0.94                  (tilt ≲ 20°)                    AND
propellant remaining > 0
```

The 3 m/s vertical limit is a nod to the real LM's landing-gear design. The
per-run [`hooks/score.py`](hooks/score.py) turns this into a `pass`, and each run
also reports `traj_rmse` (RMS altitude error vs. the real profile) and
`pitch_rmse` for calibration.

### 9.3 The campaign and its build step

[`campaign.toml`](campaign.toml) wires up ports, a per-run timeout, the scoring
and report hooks, and a **one-time build step**:

```toml
[build]
command = "cargo"
args = ["build", "--release", "--manifest-path", "examples/apollo-lander/controller/Cargo.toml"]
```

This compiles the Rust controller **once**, before any workers start, and fails
the campaign immediately if it cannot build — so the parallel runs all share one
prebuilt binary instead of each rebuilding (or racing) it.

[`hooks/report.py`](hooks/report.py) aggregates the campaign into
`post_campaign/apollo_lander_report.txt`: landing success rate, touchdown-speed
and fuel-margin distributions, and the **best-fit run** by minimum trajectory
RMSE, including its parameters.

### 9.4 Closing the loop: calibration

This is where Monte Carlo becomes more than a stress test. The best-fit run tells
you *which parameters best reproduce the real Apollo trajectory*. You then narrow
`spec.toml` around those values and run again, watching `traj_rmse` shrink — a
search for the parameter set that matches history.

[`calibrate.py`](calibrate.py) automates exactly that narrowing loop: it reads a
finished campaign, finds the best-fit parameters, writes a tightened spec
(keeping a configurable fraction of each range, centered on the best value), and
launches the next round.

```sh
python examples/apollo-lander/calibrate.py \
  --initial-out dbs/apollo-lander-demo \
  --work-dir dbs/apollo-lander-calibration \
  --rounds 2 --samples 30
```

---

## 10. Elodin features used

| Feature | What it does here | Where |
| --- | --- | --- |
| `el.World` / `el.Body` | The simulated entity and its 6-DOF state | [`sim.py#L226-L263`](sim.py#L226-L263) |
| `el.Component` (+ metadata) | Custom telemetry & `external_control` inputs | [`sim.py#L87-L159`](sim.py#L87-L159) |
| `el.Archetype` | Kinematic, non-integrated entities (truth ghost, terrain) | [`sim.py#L162-L164`](sim.py#L162-L164) |
| `@el.map` + `\|` composition | Pure-function physics systems wired into a graph | [`sim.py#L289-L401`](sim.py#L289-L401) |
| `el.system` + `el.SimulationTick` | The in-sim truth replay (`truth_playback`) | [`sim.py#L408-L434`](sim.py#L408-L434) |
| `el.six_dof` | Rigid-body translational + rotational integration | [`sim.py#L438-L447`](sim.py#L438-L447) |
| `el.Spatial*` / `el.Force` / `el.Inertia` | Spatial-vector state, force/torque, live inertia | throughout `sim.py` |
| `el.monte_carlo` | Typed parameter spec, sampling, per-run results | [`sim.py#L61-L85`](sim.py#L61-L85), [`main.py#L217-L242`](main.py#L217-L242) |
| `world.schematic` (KDL) | Viewport, graphs, GLB models, arrows, trails | [`apollo-lander.kdl`](apollo-lander.kdl), [`sim.py#L436`](sim.py#L436) |
| `world.run` (`post_step`) | The harness + SITL callback | [`main.py#L245-L256`](main.py#L245-L256) |
| `ctx.component_batch_operation` | Batched component reads/writes per tick | [`main.py#L150-L215`](main.py#L150-L215) |
| `el.s10.PyRecipe` | Launching the external FSW process | [`main.py#L107-L119`](main.py#L107-L119) |
| `start_timestamp` | Pins the sim clock to the 1969 mission epoch | [`main.py#L245-L256`](main.py#L245-L256) |
| `elodin monte-carlo` `[build]` | One-time controller build before workers | [`campaign.toml`](campaign.toml) |

---

## 11. Modeling decisions and honest caveats

Good engineering is explicit about its assumptions. This example trades some
realism for clarity, and here is exactly where:

- **Range is used as altitude.** The telemetry reports landing-radar *slant
  range*, not vertical altitude. The true geometry needs the vehicle's attitude
  and the local surface, which the dataset does not include, so we treat range as
  an altitude proxy. It is a documented teaching approximation.
- **Window-start propellant is an estimate.** The telemetry window opens at
  ~13 km, after the braking phase; we model ~4,100 kg of DPS propellant remaining
  (roughly half the full load) and let the campaign sample around it. The
  simulated descent also burns less than the real one did, because the proxy
  trajectory has no large horizontal velocity to kill.
- **Pitch is an illustrative trend, not a reconstruction.** A rigorous vehicle
  attitude from the IMU gimbal angles requires the mission **REFSMMAT** (the
  reference stable-member alignment) we don't have. The green truth vehicle
  matches the recorded *descent profile* faithfully; its attitude — and the
  reported `pitch_rmse` — are an approximate trend from the inner gimbal angle.
- **Fixed center of mass.** The real LM's CoM shifted as spherical tanks drained,
  and the DPS gimbaled to follow it. We hold the CoM fixed and let RCS provide
  control torque. We do scale the inertia tensor with mass.
- **No terrain collision.** Touchdown is detected at the flat plane `z = 0`; the
  rendered terrain relief is cosmetic (and 60× vertically exaggerated by the
  source asset).
- **Idealized sensing.** The controller receives clean state with no sensor noise,
  bias, or latency beyond the guidance-rate quantization. Adding a noisy IMU or a
  radar model is a natural next step (see below).
- **Single rigid body.** We model the descent stack as one rigid body — no
  slosh, no flex, no staging.

None of these affect the *pedagogical* goals: a stable 6-DOF descent, a real SITL
control loop, and a Monte Carlo workflow that scores robustness and calibrates
against history.

---

## 12. Running and exploring the example

All commands run from the repository root inside the `nix develop` shell.

**Watch a single descent in the editor:**

```sh
elodin editor examples/apollo-lander/main.py
```

Look for the simulated LM (with its orange thrust and purple torque arrows)
descending beside the green truth LM, and watch the six graphs converge.

**Run the Monte Carlo campaign:**

```sh
elodin monte-carlo run examples/apollo-lander/main.py \
  --campaign examples/apollo-lander/campaign.toml \
  --spec examples/apollo-lander/spec.toml \
  --out dbs/apollo-lander-demo
```

Then read `dbs/apollo-lander-demo/post_campaign/apollo_lander_report.txt`.

**Inspect the reference profile and data sanity check:**

```sh
python examples/apollo-lander/reference.py
```

> **Tip:** the editor uses database port `2240` by default. Stop any other
> `elodin`, `elodin-db`, Monte Carlo, or FSW process first, or it may connect to
> the same database and mix in unrelated telemetry.

---

## 13. Exercises for the reader

Want to go deeper? Each of these is a self-contained extension:

1. **Add sensor noise.** Corrupt the altitude/velocity in the state packet and
   add a simple filter in the controller. How much noise breaks the soft-landing
   rate?
2. **Model the moving CoM.** Shift the inertia/CoM as propellant burns and add a
   thrust offset torque. Does the RCS keep up?
3. **Fuel-optimal descent.** Replace the proportional vertical loop with a
   gravity-turn or a simple optimal guidance law and minimize propellant in the
   Monte Carlo objective.
4. **Tighten the calibration.** Run several `calibrate.py` rounds and see how low
   you can drive `traj_rmse` against the real Apollo profile.
5. **Abort modes.** Add a "low-gate / high-gate" check and a powered abort if the
   descent rate or attitude leaves a safe envelope.

---

## 14. References

- **Apollo 11 LM telemetry** — [`jumpjack/Apollo11LEMdata`](https://github.com/jumpjack/Apollo11LEMdata/blob/master/data.csv)
  (transcribed NASA postflight data).
- **NASA 3D Resources** — [Apollo Lunar Module](https://science.nasa.gov/3d-resources/apollo-lunar-module/)
  and [Apollo 11 Landing Site](https://science.nasa.gov/3d-resources/apollo-11-landing-site/).
- **NASA SVS Moon model** — [Moon 3D Models for Web, AR, and Animation](https://svs.gsfc.nasa.gov/14959/),
  built from Lunar Reconnaissance Orbiter imagery and topographic data.
- **LM vehicle data** — [`data/lunar_module_spec_sheet.pdf`](data/lunar_module_spec_sheet.pdf).
- **Elodin** — [elodin.systems](https://www.elodin.systems/); see the repository
  skills under `.cursor/skills/` for the simulation SDK, editor, and database.
- **This example's companion doc** — [`README.md`](README.md) for the quick-start
  and operational notes.

---

*Built with Elodin — aerospace's open-source answer to ROS. Ad astra.*


