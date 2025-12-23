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

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import elodin as el
import jax.numpy as jnp
import numpy as np

from config import DroneConfig, DEFAULT_CONFIG
from sim import Drone, create_physics_system
from sensors import IMU, create_sensor_system, SensorDataBuffer
from comms import (
    BetaflightSyncBridge,
    RCPacket,
    remap_motors_betaflight_to_elodin,
    MAX_RC_CHANNELS,
)


# --- Configuration ---
config = DEFAULT_CONFIG
config.simulation_time = 11.0
config.sim_time_step = 0.001  # 1kHz physics
config.set_as_global()


# --- Betaflight Binary Path ---
BETAFLIGHT_PATH = Path(__file__).parent / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"

if not BETAFLIGHT_PATH.exists():
    print(f"ERROR: Betaflight SITL not found at {BETAFLIGHT_PATH}")
    print("Run ./build.sh in examples/betaflight-sitl to build it")
    sys.exit(1)


# --- Clean up stale processes ---
def cleanup_stale_processes():
    """Kill any stale Betaflight SITL processes and free ports."""
    # Kill Betaflight
    try:
        subprocess.run(["pkill", "-f", "betaflight_SITL"], capture_output=True, timeout=5)
    except Exception:
        pass
    
    # Free elodin-db port (2240)
    try:
        result = subprocess.run(
            ["lsof", "-ti:2240"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
                except Exception:
                    pass
    except Exception:
        pass
    
    time.sleep(0.5)


def cleanup_on_exit():
    """Cleanup handler for exit - kills Betaflight processes."""
    try:
        subprocess.run(["pkill", "-9", "-f", "betaflight_SITL"], capture_output=True, timeout=2)
    except Exception:
        pass


def signal_handler(signum, frame):
    """Handle Ctrl+C by cleaning up and exiting."""
    print("\n[SITL] Caught interrupt, cleaning up...")
    cleanup_on_exit()
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)

# Only cleanup stale processes when NOT running under s10 orchestration
# When s10 manages us, it passes --no-s10 flag, and it starts Betaflight separately
if "--no-s10" not in sys.argv:
    cleanup_stale_processes()


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
print(f"Simulation: {config.simulation_time}s at {1/config.sim_time_step:.0f}Hz")


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
BOOTGRACE = 5.0   # Wait for Betaflight to initialize
ARM_DUR = 2.0     # Arming phase
THROTTLE_DUR = 3.0  # Apply throttle
# Remaining time is disarm phase

# Calculate max ticks for completion detection
MAX_TICKS = int(config.simulation_time / config.sim_time_step)

# Shared state (using lists for mutable closure)
bridge = [None]
sensor_buf = [None]
state = [None]
start_time = [None]
last_print = [0.0]


def sitl_post_step(tick: int, ctx: el.PostStepContext):
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
        for i in range(500):  # 500ms of warmup at ~1ms per packet
            try:
                warmup_fdm.timestamp = i * 0.001
                warmup_rc.timestamp = i * 0.001
                bridge[0].step(warmup_fdm, warmup_rc)
                warmup_count += 1
            except TimeoutError:
                pass  # Expected during initial warmup
        print(f"[SITL] Warmup complete ({warmup_count} responses)")
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
    
    # Read actual sensor data from physics simulation
    # CRITICAL: These are essential for Betaflight's PID loop
    try:
        accel = np.array(ctx.read_component("drone.accel"))      # Body-frame accelerometer
        gyro = np.array(ctx.read_component("drone.gyro"))        # Body-frame gyroscope
        world_pos = np.array(ctx.read_component("drone.world_pos"))  # GPS simulation
        world_vel = np.array(ctx.read_component("drone.world_vel"))  # GPS velocity
        
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
        s.arm = 1000     # Disarmed
        s.throttle = 1000  # Min throttle
    elif t < BOOTGRACE + ARM_DUR:
        phase = "arm"
        s.arm = 1800     # Armed (AUX1 high)
        s.throttle = 1000  # Min throttle during arm
    elif t < BOOTGRACE + ARM_DUR + THROTTLE_DUR:
        phase = "throttle"
        s.arm = 1800     # Stay armed
        s.throttle = 1400  # Mid throttle
    else:
        phase = "disarm"
        s.arm = 1000     # Disarm
        s.throttle = 1000
    
    # Build RC packet with all channels
    channels = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
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
        motors_bf = b.step(fdm, rc)
        s.motors = remap_motors_betaflight_to_elodin(motors_bf)
        s.max_motor = max(s.max_motor, np.max(s.motors))
        
        # Write motor commands back to Elodin-DB for physics simulation
        # This uses the PostStepContext for direct DB access (no TCP overhead)
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
            # world_pos layout: [qw, qx, qy, qz, x, y, z]
            quat = world_pos[:4]
            debug_str = (
                f"\n    [DEBUG] motor_cmd={motor_cmd_db.sum():.3f} thrust={motor_thrust.sum():.2f}N"
                f"\n    [DEBUG] body_thrust=[{body_thrust[3]:.1f},{body_thrust[4]:.1f},{body_thrust[5]:.1f}]N (linear xyz)"
                f"\n    [DEBUG] force=[{force[3]:.1f},{force[4]:.1f},{force[5]:.1f}]N (linear xyz)"
                f"\n    [DEBUG] quat=[{quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f}]"
            )
        except Exception as e:
            debug_str = f"\n    [DEBUG] read failed: {e}"
        
        print(f"  t={t:5.1f}s | {phase:8} | {armed:8} | "
              f"motors=[{s.motors[0]:.3f},{s.motors[1]:.3f},{s.motors[2]:.3f},{s.motors[3]:.3f}] | "
              f"{pos_str} | {rate:.1f}x realtime{debug_str}")
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
        print(f"  Simulated: {s.sim_time:.1f}s in {elapsed:.1f}s "
              f"({s.sim_time/elapsed if elapsed > 0 else 0:.1f}x realtime)")
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
        
        # Force exit - world.run() may not return cleanly
        cleanup_on_exit()
        os._exit(0)


# --- Run Simulation ---
# world.run() creates a CLI - use with:
#   python3 examples/betaflight-sitl/main.py run
#   elodin run examples/betaflight-sitl/main.py
#   elodin editor examples/betaflight-sitl/main.py

world.run(
    system,
    sim_time_step=config.sim_time_step,
    run_time_step=config.sim_time_step,
    max_ticks=int(config.simulation_time / config.sim_time_step),
    post_step=sitl_post_step,
)


# --- Cleanup (fallback if world.run() returns without triggering post_step exit) ---
cleanup_on_exit()
if not bridge[0]:
    print("\nNo simulation ticks executed.")
    print("Usage: python3 examples/betaflight-sitl/main.py run")
