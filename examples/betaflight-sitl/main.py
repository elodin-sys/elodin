#!/usr/bin/env python3
"""
Betaflight SITL Drone Simulation with Elodin

This is the main entry point for the Betaflight Software-In-The-Loop (SITL)
drone simulation integrating:
- Elodin physics simulation (rigid body dynamics, forces, sensors)
- Betaflight flight controller (running as SITL with GYROPID_SYNC)
- Lockstep time synchronization via post_step callback
- s10 process orchestration for Betaflight lifecycle management

Usage:
    python3 examples/betaflight-sitl/main.py run    # Headless simulation
    elodin run examples/betaflight-sitl/main.py     # Headless with s10
    elodin editor examples/betaflight-sitl/main.py  # With 3D visualization

Prerequisites:
    1. Build Betaflight SITL with GYROPID_SYNC: ./build.sh
    2. (Optional) Configure Betaflight via CLI: socat + screen
"""

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import elodin as el
import jax.numpy as jnp
import numpy as np

from audit import AxisAudit
from baseline import C0Result, evaluate_c0
from config import DEFAULT_CONFIG
from controls import (
    DelayedRcCommand,
    GuidanceMode,
    GuidanceUpdate,
    ManualGuidance,
    RcCommand,
    ScriptedGuidance,
    SemanticControl,
    guidance_mode_from_env,
    semantic_to_rc,
)
from sim import Drone, create_physics_system
from sensors import IMU, create_sensor_system, SensorDataBuffer
from comms import (
    BetaflightSyncBridge,
    RCPacket,
    MAX_RC_CHANNELS,
)


# --- Configuration ---
config = DEFAULT_CONFIG

try:
    guidance_mode = guidance_mode_from_env()
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(2)

audit_value = os.environ.get("RACE_MANUAL_AUDIT", "0")
if audit_value not in ("0", "1"):
    print("ERROR: RACE_MANUAL_AUDIT must be '0' or '1'", file=sys.stderr)
    sys.exit(2)
audit_requested = audit_value == "1"
if audit_requested and guidance_mode is not GuidanceMode.MANUAL:
    print("ERROR: RACE_MANUAL_AUDIT=1 requires RACE_GUIDANCE=manual", file=sys.stderr)
    sys.exit(2)
if audit_requested:
    config.simulation_time = 24.0
config.set_as_global()

if guidance_mode is GuidanceMode.SCRIPTED:
    command_source = ScriptedGuidance()
    initial_control = SemanticControl(angle_mode=False)
else:
    command_source = ManualGuidance()
    initial_control = SemanticControl.safe()


# --- Betaflight Binary Path ---
BETAFLIGHT_PATH = Path(__file__).parent / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"

if not BETAFLIGHT_PATH.exists():
    print(f"ERROR: Betaflight SITL not found at {BETAFLIGHT_PATH}")
    print("Run ./build.sh in examples/betaflight-sitl to build it")
    sys.exit(1)


# --- Clean up stale processes from previous runs ---
# This runs BEFORE s10 starts, so it only affects leftover processes from
# previous interrupted simulations, not the current run's Betaflight.
def cleanup_stale_betaflight():
    """Kill any stale Betaflight SITL processes from previous runs."""
    try:
        subprocess.run(["pkill", "-f", "betaflight_SITL"], capture_output=True, timeout=5)
        time.sleep(0.1)  # Brief pause to let the process terminate
    except Exception:
        pass


# Only cleanup when running with s10 (without --no-s10 flag)
# s10 will start a fresh Betaflight after world.run() begins
if "--no-s10" not in sys.argv:
    cleanup_stale_betaflight()


# --- World Creation ---
world = el.World()

drone = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                linear=jnp.array(config.initial_position),
                angular=el.Quaternion(jnp.array(config.initial_quaternion)),
            ),
            world_vel=el.SpatialMotion(
                linear=jnp.array(config.initial_velocity),
                angular=jnp.array(config.initial_angular_velocity),
            ),
            inertia=el.SpatialInertia(
                mass=config.mass,
                inertia=jnp.array(config.inertia_diagonal),
            ),
        ),
        Drone(),
        IMU(),
    ],
    name="drone",
)

# Editor schematic for visualization
world.schematic(
    """
    tabs {
        hsplit name = "Viewport" {
            viewport name=Viewport pos="drone.world_pos + (0,0,0,0, 10,10,5)" look_at="drone.world_pos" show_grid=#true active=#true
            vsplit share=0.3 {
                graph "drone.motor_command" name="Motor Commands (from Betaflight)"
                graph "drone.motor_thrust" name="Motor Thrust"
                graph "drone.accel" name="Accelerometer"
            }
            vsplit share=0.3 {
                graph "drone.world_pos.linear()" name="Position (ENU)"
                graph "drone.world_vel.linear()" name="Velocity"
                graph "drone.gyro" name="Gyroscope"
            }
        }
    }
    object_3d drone.world_pos {
        glb path="edu-450-v2-drone.glb" scale=10.0
    }
    """,
    "betaflight-sitl.kdl",
)


# --- System ---
physics = create_physics_system(config)
sensors = create_sensor_system(config)
system = physics | sensors


# --- Betaflight Process Management via s10 ---
# Register Betaflight SITL as an s10 process recipe
# s10 will manage the process lifecycle (start/stop) in all execution contexts
betaflight_recipe = el.s10.PyRecipe.process(
    name="Betaflight SITL",
    cmd=str(BETAFLIGHT_PATH),
    cwd=str(Path(__file__).parent),
)
world.recipe(betaflight_recipe)

# Manual input is deliberately a separate s10-supervised process. The default
# scripted example neither builds nor starts it.
if guidance_mode is GuidanceMode.MANUAL:
    controller_path = Path(__file__).parent / "controller"
    stick_mode = os.environ.get("RACE_STICK_MODE", "2")
    if stick_mode not in ("1", "2"):
        raise ValueError("RACE_STICK_MODE must be '1' or '2'")
    controller_args = ["--mode1"] if stick_mode == "1" else []
    if audit_requested:
        controller_args.append("--audit")
    controller_host = os.environ.get("RACE_CONTROLLER_HOST")
    if controller_host:
        controller_args.extend(["--host", controller_host])
    if not controller_args:
        controller_args = None
    controller_binary = os.environ.get("RACE_CONTROLLER_BIN")
    if controller_binary:
        binary = Path(controller_binary).resolve()
        if not binary.is_file():
            raise RuntimeError(f"manual controller binary not found: {binary}")
        manual_controller_recipe = el.s10.PyRecipe.process(
            name="Betaflight manual controller",
            cmd=str(binary),
            args=controller_args,
            ready=el.s10.Ready.delay(100),
            ready_timeout="120s",
        )
    else:
        manual_controller_recipe = el.s10.PyRecipe.cargo(
            name="Betaflight manual controller",
            path=str(controller_path),
            args=controller_args,
            ready=el.s10.Ready.delay(100),
            ready_timeout="120s",
        )
    world.recipe(manual_controller_recipe)

print(f"Betaflight SITL: {BETAFLIGHT_PATH.name}")
print(f"Guidance: {guidance_mode.value}")
print(f"Simulation: {config.simulation_time}s at {config.pid_rate:.0f}Hz PID loop")
print(
    f"Requested sensor rates: gyro={config.gyro_rate:.0f}Hz, accel={config.accel_rate:.0f}Hz, baro={config.baro_rate:.0f}Hz, mag={config.mag_rate:.0f}Hz"
)


# --- SITL State ---
@dataclass
class SITLState:
    """State for SITL synchronization."""

    tick: int = 0
    sim_time: float = 0.0
    motors: np.ndarray = None
    rc_latch: DelayedRcCommand | None = None
    selected_control: SemanticControl | None = None
    rc_telemetry: RcCommand | None = None
    barometer: float | None = None
    magnetometer: np.ndarray | None = None
    manual_values: np.ndarray | None = None
    phase: str = "boot"
    max_motor: float = 0.0
    lockstep_steps: int = 0
    max_altitude: float = float(config.initial_position[2])

    def __post_init__(self):
        if self.motors is None:
            self.motors = np.zeros(4)
        if self.rc_latch is None:
            self.rc_latch = DelayedRcCommand(semantic_to_rc(initial_control))
        if self.selected_control is None:
            self.selected_control = initial_control


# Calculate max ticks for completion detection
MAX_TICKS = int(config.simulation_time / config.dt)

# Shared state (using lists for mutable closure)
bridge = [None]
sensor_buf = [None]
state = [None]
start_time = [None]
last_print = [0.0]
c0_result: list[C0Result | None] = [None]
axis_audit = AxisAudit() if audit_requested else None

# Pre-allocated buffers to avoid allocation in hot loop
_rc_channels_buffer = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
_rc_packet = RCPacket(timestamp=0.0, channels=_rc_channels_buffer)
_rc_channels_command: list[RcCommand | None] = [None]
_zero_sensor_vector = np.zeros(3)
_component_reads = [
    "drone.accel",
    "drone.gyro",
    "drone.world_pos",
    "drone.world_vel",
]
# Manual packets arrive at about 100 Hz. Sampling the DB seam at 1 kHz keeps
# control latency below 1 ms while avoiding an unnecessary array conversion on
# every 125 microsecond physics tick. Freshness is still checked every tick.
_manual_input_tick_interval = max(1, round(config.pid_rate / 1_000.0))
_barometer_tick_interval = config.baro_tick_interval
_magnetometer_tick_interval = config.mag_tick_interval


def sitl_post_step(tick: int, ctx: el.StepContext):
    """
    Post-step callback for lockstep SITL synchronization.

    The ordering is intentional: sensors and the command retained on tick N-1
    are exchanged first; only after the motor response does the selected source
    compute and retain the command that will be sent on tick N+1.
    """
    # Lazy initialization - only start bridge when first tick runs
    if bridge[0] is None:
        print("[SITL] Initializing bridge...")
        bridge[0] = BetaflightSyncBridge(timeout_ms=100)
        sensor_buf[0] = SensorDataBuffer()
        state[0] = SITLState()
        bridge[0].start()
        # Give Betaflight (started by s10) time to fully initialize
        # Betaflight needs time to complete gyro calibration and internal setup
        print("[SITL] Waiting for Betaflight to initialize...")
        time.sleep(2)

        # Warmup phase: Send some initial packets to prime Betaflight's RC processing
        # This helps stabilize the throttle response on fresh starts
        print("[SITL] Sending warmup packets...")
        warmup_buf = SensorDataBuffer()
        warmup_fdm = warmup_buf.build_fdm()
        warmup_channels = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
        semantic_to_rc(initial_control).fill_channels(warmup_channels)
        warmup_rc = RCPacket(timestamp=0.0, channels=warmup_channels)

        warmup_count = 0
        warmup_packets = int(0.5 / config.dt)  # 500ms of warmup at PID rate
        for i in range(warmup_packets):
            try:
                warmup_fdm.timestamp = i * config.dt
                warmup_rc.timestamp = i * config.dt
                bridge[0].step(warmup_fdm, warmup_rc)
                warmup_count += 1
            except TimeoutError:
                pass  # Expected during initial warmup
        print(f"[SITL] Warmup complete ({warmup_count} responses at {config.pid_rate:.0f}Hz)")
        print("[SITL] Bridge ready")

    if start_time[0] is None:
        start_time[0] = time.time()

    s = state[0]
    b = bridge[0]
    buf = sensor_buf[0]

    # Update timing
    s.tick = tick
    s.sim_time = tick * config.dt
    t = s.sim_time

    # Read current state and due lower-rate samples in one DB transaction.
    # component_batch_operation already returns fresh NumPy arrays, so the
    # callback can transfer them to its short-lived FDM buffer without first
    # making another copy.
    accel = _zero_sensor_vector
    gyro = _zero_sensor_vector
    barometer = s.barometer
    magnetometer = s.magnetometer
    manual_values = s.manual_values
    barometer_fresh = tick % _barometer_tick_interval == 0
    magnetometer_fresh = tick % _magnetometer_tick_interval == 0
    manual_input_poll = (
        guidance_mode is GuidanceMode.MANUAL and tick % _manual_input_tick_interval == 0
    )
    reads = _component_reads.copy()
    if barometer_fresh:
        reads.append("drone.baro")
    if magnetometer_fresh:
        reads.append("drone.mag")
    if manual_input_poll:
        reads.append("drone.manual_control")
    sensor_read_succeeded = False
    try:
        sensor_data = ctx.component_batch_operation(reads=reads)
        accel = sensor_data["drone.accel"]
        gyro = sensor_data["drone.gyro"]
        world_pos = sensor_data["drone.world_pos"]
        world_vel = sensor_data["drone.world_vel"]
        if barometer_fresh:
            s.barometer = float(sensor_data["drone.baro"][0])
            barometer = s.barometer
        if magnetometer_fresh:
            s.magnetometer = sensor_data["drone.mag"]
            magnetometer = s.magnetometer
        if manual_input_poll:
            s.manual_values = sensor_data["drone.manual_control"]
            manual_values = s.manual_values
        sensor_read_succeeded = True
        s.max_altitude = max(s.max_altitude, float(world_pos[6]))

        # The batch operation returned new owned arrays. The FDM buffer keeps
        # those arrays only until they are replaced on the next tick, so another
        # copy here would just consume the 125 microsecond lockstep budget.
        buf.update(
            world_pos=world_pos,
            world_vel=world_vel,
            accel=accel,
            gyro=gyro,
            timestamp=t,
            copy=False,
        )
    except RuntimeError as e:
        if tick > 5:
            print(f"[SITL] Warning: Could not read sensor data: {e}")
        buf.timestamp = t

    # Send the command retained on the previous tick. A command change resets
    # all sixteen channels, including the ten unused channels, to deterministic
    # values; unchanged commands reuse that complete packet in the hot loop.
    command_sent = s.rc_latch.command_for_exchange()
    if command_sent != _rc_channels_command[0]:
        command_sent.fill_channels(_rc_channels_buffer)
        _rc_channels_command[0] = command_sent
    fdm = buf.build_fdm()
    _rc_packet.timestamp = t

    try:
        # Synchronous lockstep: send FDM+RC, wait for motor response.
        # Motor order is native Betaflight Quad-X: BR(0), FR(1), BL(2), FL(3).
        steps_before = b.step_count
        s.motors = b.step(fdm, _rc_packet)
        if b.step_count > steps_before:
            s.lockstep_steps += 1
        s.max_motor = max(s.max_motor, float(np.max(s.motors)))
        ctx.write_component("drone.motor_command", s.motors)
        if axis_audit is not None:
            axis_audit.observe(command_sent, gyro, s.motors)
    except TimeoutError:
        pass  # Timeouts expected during bootgrace

    update = GuidanceUpdate(
        sim_time=t,
        tick=tick,
        gyro=gyro,
        accel=accel,
        barometer=barometer,
        barometer_fresh=barometer_fresh and sensor_read_succeeded,
        magnetometer=magnetometer,
        magnetometer_fresh=magnetometer_fresh and sensor_read_succeeded,
    )
    if isinstance(command_source, ScriptedGuidance):
        next_control = command_source.update(update)
        s.phase = command_source.phase
    else:
        command_source.accept_input(manual_values, time.monotonic())
        next_control = command_source.update(update)
        s.phase = "manual" if command_source.fresh else "failsafe"

    # Most guidance ticks select the same command (manual input arrives at
    # about 100 Hz). Preserve the one-tick latch while avoiding an 8 kHz stream
    # of identical allocations and DB telemetry writes.
    if next_control != s.selected_control:
        s.selected_control = next_control
        s.rc_latch.retain_for_next_tick(semantic_to_rc(next_control))
    next_command = s.rc_latch.command_for_exchange()
    if next_command != s.rc_telemetry:
        ctx.write_component("drone.rc_command", next_command.as_array())
        s.rc_telemetry = next_command

    # Print status every second
    if t - last_print[0] >= 1.0:
        armed = "ARMED" if np.any(s.motors > 0.02) else "DISARMED"
        elapsed = time.time() - start_time[0]
        rate = t / elapsed if elapsed > 0 else 0

        # Get current position for debug output
        try:
            pos = np.array(ctx.read_component("drone.world_pos"))
            z_pos = pos[6] if len(pos) > 6 else pos[2]  # world_pos is [qw,qx,qy,qz,x,y,z]
            vel = np.array(ctx.read_component("drone.world_vel"))
            z_vel = vel[5] if len(vel) > 5 else vel[2]  # world_vel is [wx,wy,wz,vx,vy,vz]
            pos_str = f"z={z_pos:+.2f}m vz={z_vel:+.2f}m/s"
        except Exception:
            pos_str = "z=?.??m"

        # DEBUG: Check what motor values the physics is seeing
        try:
            motor_cmd_db = np.array(ctx.read_component("drone.motor_command"))
            motor_thrust = np.array(ctx.read_component("drone.motor_thrust"))
            body_thrust = np.array(ctx.read_component("drone.body_thrust"))
            force = np.array(ctx.read_component("drone.force"))
            world_pos = np.array(ctx.read_component("drone.world_pos"))
            # body_thrust layout: [τx, τy, τz, fx, fy, fz]
            # world_pos layout: [qx, qy, qz, qw, x, y, z] (Elodin scalar-last format)
            quat_xyzw = world_pos[:4]
            debug_str = (
                f"\n    [DEBUG] motor_cmd={motor_cmd_db.sum():.3f} thrust={motor_thrust.sum():.2f}N"
                f"\n    [DEBUG] body_thrust=[{body_thrust[3]:.1f},{body_thrust[4]:.1f},{body_thrust[5]:.1f}]N (linear xyz)"
                f"\n    [DEBUG] force=[{force[3]:.1f},{force[4]:.1f},{force[5]:.1f}]N (linear xyz)"
                f"\n    [DEBUG] quat(xyzw)=[{quat_xyzw[0]:.3f},{quat_xyzw[1]:.3f},{quat_xyzw[2]:.3f},{quat_xyzw[3]:.3f}]"
            )
        except Exception as e:
            debug_str = f"\n    [DEBUG] read failed: {e}"

        print(
            f"  t={t:5.1f}s | {s.phase:8} | {armed:8} | "
            f"motors=[{s.motors[0]:.3f},{s.motors[1]:.3f},{s.motors[2]:.3f},{s.motors[3]:.3f}] | "
            f"{pos_str} | {rate:.1f}x realtime{debug_str}"
        )
        last_print[0] = t

    # Check if simulation is complete - print summary and exit
    if tick >= MAX_TICKS - 1:
        b.stop()
        ctx.stop_recipes()
        elapsed = time.time() - start_time[0]

        # Read final position
        try:
            final_pos = np.array(ctx.read_component("drone.world_pos"))
            final_z = final_pos[6] if len(final_pos) > 6 else final_pos[2]
            s.max_altitude = max(s.max_altitude, float(final_z))
            final_vel = np.array(ctx.read_component("drone.world_vel"))
            final_vz = final_vel[5] if len(final_vel) > 5 else final_vel[2]
        except Exception:
            final_z = 0.0
            final_vz = 0.0

        print()
        print("=" * 50)
        print("Simulation complete!")
        print(
            f"  Simulated: {s.sim_time:.1f}s in {elapsed:.1f}s "
            f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)"
        )
        print(f"  Total ticks: {s.tick}")
        print(f"  Sync steps: {b.step_count}")
        print(f"  Max motor: {s.max_motor:.3f}")
        print(f"  Final position: z={final_z:.2f}m, vz={final_vz:.2f}m/s")
        print()

        if guidance_mode is GuidanceMode.SCRIPTED:
            result = evaluate_c0(
                lockstep_steps=s.lockstep_steps,
                max_motor=s.max_motor,
                initial_altitude=float(config.initial_position[2]),
                max_altitude=s.max_altitude,
            )
            c0_result[0] = result

            # C0 applies only to the default scripted source. Manual mode may
            # legitimately remain disarmed for the entire run.
            if result.passed:
                print("SUCCESS: SITL integration working! Drone took off!")
            elif s.lockstep_steps <= 0:
                print("WARNING: No lockstep motor responses received from Betaflight.")
            elif result.motor_response:
                print("WARNING: Motors responded but drone did not take off.")
                print("  Check physics pipeline: motor_command -> thrust -> force")
            elif s.max_motor > 0.02:
                print("WARNING: Motors armed but no throttle response.")
            else:
                print("WARNING: No motor response. Check Betaflight configuration.")

            print(result.format())
        elif isinstance(command_source, ManualGuidance):
            print(
                f"Manual input ended in {command_source.reason!r}; vehicle commanded disarmed on exit."
            )
            if axis_audit is not None:
                print(axis_audit.format())


# Return the next non-existent filename with auto-incremented
# number if the pattern ends in Xs.
#
# e.g., `next_filename("sim_sitlXXX") -> "sim_sitl001"`
# `next_filename("sim_sitl_mine") -> "sim_sitl_mine"`
def next_filename(pattern: str) -> str:
    match = re.search(r"(X+)$", pattern)
    if not match:
        return pattern

    width = len(match.group(1))
    prefix = pattern[:-width]

    i = 0
    while True:
        fname = f"{prefix}{i:0{width}d}"
        if not os.path.exists(fname):
            return fname
        i += 1


# --- Run Simulation ---
# world.run() creates a CLI - use with:
#   python3 examples/betaflight-sitl/main.py run
#   elodin run examples/betaflight-sitl/main.py
#   elodin editor examples/betaflight-sitl/main.py

db_filename = next_filename("betaflight_dbXXX")
world.run(
    system,
    simulation_rate=config.pid_rate,
    generate_real_time=True,
    max_ticks=config.total_sim_ticks,
    post_step=sitl_post_step,
    db_path=db_filename,
    interactive=False,
)
# `world.run()` won't reach here unless `interactive` is false.
print(f"Wrote database to: {db_filename}")

if not bridge[0]:
    print("\nNo simulation ticks executed.")
    print("Usage: python3 examples/betaflight-sitl/main.py run")
elif c0_result[0] is not None and not c0_result[0].passed:
    sys.exit(1)
elif axis_audit is not None and not axis_audit.passed:
    sys.exit(1)
