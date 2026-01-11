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

from config import DEFAULT_CONFIG
from sim import Drone, create_physics_system
from sensors import IMU, create_sensor_system, SensorDataBuffer
from comms import (
    BetaflightSyncBridge,
    RCPacket,
    MAX_RC_CHANNELS,
)


# --- Configuration ---
config = DEFAULT_CONFIG
config.set_as_global()


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
        glb path="edu-450-v2-drone.glb" rotate="(0.0, 0.0, 0.0)" translate="(0.0, 1.0, 0.0)" scale=10.0
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

print(f"Betaflight SITL: {BETAFLIGHT_PATH.name}")
print(f"Simulation: {config.simulation_time}s at {config.pid_rate:.0f}Hz PID loop")
print(
    f"Sensor rates: gyro={config.gyro_rate:.0f}Hz, accel={config.accel_rate:.0f}Hz, baro={config.baro_rate:.0f}Hz, mag={config.mag_rate:.0f}Hz"
)


# --- SITL State ---
@dataclass
class SITLState:
    """State for SITL synchronization."""

    throttle: int = 1000
    arm: int = 1000
    tick: int = 0
    sim_time: float = 0.0
    motors: np.ndarray = None
    max_motor: float = 0.0

    def __post_init__(self):
        if self.motors is None:
            self.motors = np.zeros(4)


# Test phases (durations in seconds)
BOOTGRACE = 5.0  # Wait for Betaflight to initialize
ARM_DUR = 2.0  # Arming phase
THROTTLE_DUR = 10.0  # Apply throttle (longer for observation)
# Remaining time is disarm phase

# Calculate max ticks for completion detection
MAX_TICKS = int(config.simulation_time / config.sim_time_step)

# Shared state (using lists for mutable closure)
bridge = [None]
sensor_buf = [None]
state = [None]
start_time = [None]
last_print = [0.0]

# Pre-allocated buffers to avoid allocation in hot loop
_rc_channels_buffer = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)


def sitl_post_step(tick: int, ctx: el.StepContext):
    """
    Post-step callback for lockstep SITL synchronization.

    This implements the two-phase synchronization pattern:
    1. Send sensor data (FDM) and RC inputs to Betaflight
    2. Wait for motor response (blocking - this is the lockstep sync point)
    3. Write motor commands back to Elodin-DB via ctx.write_component()

    Following the pattern from ai-context/sitl-example/SITL_EXAMPLE_EXPLAINED.md
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
        warmup_channels[2] = 1000  # Low throttle
        warmup_channels[4] = 1000  # Disarmed
        warmup_rc = RCPacket(timestamp=0.0, channels=warmup_channels)

        warmup_count = 0
        warmup_packets = int(0.5 / config.sim_time_step)  # 500ms of warmup at PID rate
        for i in range(warmup_packets):
            try:
                warmup_fdm.timestamp = i * config.sim_time_step
                warmup_rc.timestamp = i * config.sim_time_step
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
    s.sim_time = tick * config.sim_time_step
    t = s.sim_time

    # Read actual sensor data from physics simulation using batch operation
    # This acquires the DB lock once for all reads, improving performance at high tick rates
    try:
        sensor_data = ctx.component_batch_operation(
            reads=["drone.accel", "drone.gyro", "drone.world_pos", "drone.world_vel"]
        )
        accel = np.array(sensor_data["drone.accel"])  # Body-frame accelerometer
        gyro = np.array(sensor_data["drone.gyro"])  # Body-frame gyroscope
        world_pos = np.array(sensor_data["drone.world_pos"])  # GPS simulation
        world_vel = np.array(sensor_data["drone.world_vel"])  # GPS velocity

        # Update sensor buffer with real physics data
        buf.update(
            world_pos=world_pos,
            world_vel=world_vel,
            accel=accel,
            gyro=gyro,
            timestamp=t,
        )
    except RuntimeError as e:
        # First few ticks may not have data yet
        if tick > 5:
            print(f"[SITL] Warning: Could not read sensor data: {e}")
        buf.timestamp = t

    # Phase logic - determine arm and throttle based on sim time
    if t < BOOTGRACE:
        phase = "boot"
        s.arm = 1000  # Disarmed
        s.throttle = 1000  # Min throttle
    elif t < BOOTGRACE + ARM_DUR:
        phase = "arm"
        s.arm = 1800  # Armed (AUX1 high)
        s.throttle = 1000  # Min throttle during arm
    elif t < BOOTGRACE + ARM_DUR + THROTTLE_DUR:
        phase = "throttle"
        s.arm = 1800  # Stay armed
        s.throttle = 1400  # Mid throttle
    else:
        phase = "disarm"
        s.arm = 1000  # Disarm
        s.throttle = 1000

    # Build RC packet with all channels (reuse pre-allocated buffer)
    channels = _rc_channels_buffer
    channels[0] = 1500  # Roll (center)
    channels[1] = 1500  # Pitch (center)
    channels[2] = s.throttle  # Throttle
    channels[3] = 1500  # Yaw (center)
    channels[4] = s.arm  # AUX1 (arm switch)

    # Build FDM packet with sensor data
    fdm = buf.build_fdm()
    rc = RCPacket(timestamp=t, channels=channels)

    try:
        # Synchronous lockstep: send FDM+RC, wait for motor response
        # Motor order matches Betaflight Quad-X: FR(0), BR(1), BL(2), FL(3)
        # The physics simulation (config.py) uses the same motor layout
        s.motors = b.step(fdm, rc)
        s.max_motor = max(s.max_motor, np.max(s.motors))

        # Write motor commands back to Elodin-DB for physics simulation
        # This uses the StepContext for direct DB access (no TCP overhead)
        ctx.write_component("drone.motor_command", s.motors)
    except TimeoutError:
        pass  # Timeouts expected during bootgrace

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
            f"  t={t:5.1f}s | {phase:8} | {armed:8} | "
            f"motors=[{s.motors[0]:.3f},{s.motors[1]:.3f},{s.motors[2]:.3f},{s.motors[3]:.3f}] | "
            f"{pos_str} | {rate:.1f}x realtime{debug_str}"
        )
        last_print[0] = t

    # Check if simulation is complete - print summary and exit
    if tick >= MAX_TICKS - 1:
        b.stop()
        elapsed = time.time() - start_time[0]

        # Read final position
        try:
            final_pos = np.array(ctx.read_component("drone.world_pos"))
            final_z = final_pos[6] if len(final_pos) > 6 else final_pos[2]
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

        # Success criteria: motors responded AND drone moved
        took_off = final_z > config.initial_position[2] + 0.1  # More than 10cm above start

        if b.step_count > 0 and s.max_motor > 0.06 and took_off:
            print("SUCCESS: SITL integration working! Drone took off!")
        elif b.step_count > 0 and s.max_motor > 0.06:
            print("WARNING: Motors responded but drone did not take off.")
            print("  Check physics pipeline: motor_command -> thrust -> force")
        elif b.step_count > 0 and s.max_motor > 0.02:
            print("WARNING: Motors armed but no throttle response.")
        else:
            print("WARNING: No motor response. Check Betaflight configuration.")


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
    sim_time_step=config.sim_time_step,
    run_time_step=config.sim_time_step,
    max_ticks=int(config.simulation_time / config.sim_time_step),
    post_step=sitl_post_step,
    db_path=db_filename,
    interactive=False,
)
# `world.run()` won't reach here unless `interactive` is false.
print(f"Wrote database to: {db_filename}")

if not bridge[0]:
    print("\nNo simulation ticks executed.")
    print("Usage: python3 examples/betaflight-sitl/main.py run")
