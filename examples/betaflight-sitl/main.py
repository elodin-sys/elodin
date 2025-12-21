#!/usr/bin/env python3
"""
Betaflight SITL Drone Simulation with Elodin

This is the main entry point for the Betaflight Software-In-The-Loop (SITL)
drone simulation. It integrates:
- Elodin physics simulation (rigid body dynamics, forces, sensors)
- Betaflight flight controller (running as SITL binary with GYROPID_SYNC)
- Synchronous UDP communication bridge
- Lockstep time control via post_step callback

Architecture:
    Each simulation tick:
    1. Elodin runs physics (JAX)
    2. Elodin commits state to DB
    3. post_step callback fires:
       a. Build FDM packet from sensor data
       b. Send FDM + RC to Betaflight (UDP)
       c. Wait for motor response (blocking)
       d. Write motor_command to DB
    4. Next tick begins (using updated motor commands)

Usage:
    elodin editor main.py           # Run with 3D visualization
    python3 main.py bench           # Run headless benchmark
    python3 main.py lockstep        # Run lockstep SITL test

Prerequisites:
    1. Build Betaflight SITL with GYROPID_SYNC: ./build.sh
    2. (Optional) Configure Betaflight via CLI: socat + screen
"""

import atexit
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import elodin as el
import jax.numpy as jnp
import numpy as np

from config import DroneConfig, DEFAULT_CONFIG
from sim import Drone, create_physics_system
from sensors import IMU, create_sensor_system, build_fdm_from_components, SensorDataBuffer
from comms import (
    BetaflightSyncBridge,
    BetaflightBridge,
    RCPacket,
    FDMPacket,
    remap_motors_betaflight_to_elodin,
    MAX_RC_CHANNELS,
)


# --- Global Betaflight Process Tracking ---
# Track the current Betaflight process so we can clean it up on exit
_betaflight_process: Optional[subprocess.Popen] = None


def _cleanup_betaflight_on_exit():
    """Cleanup handler called on Python process exit."""
    global _betaflight_process
    if _betaflight_process is not None:
        try:
            _betaflight_process.terminate()
            _betaflight_process.wait(timeout=2)
        except Exception:
            try:
                _betaflight_process.kill()
            except Exception:
                pass
        _betaflight_process = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    _cleanup_betaflight_on_exit()
    sys.exit(0)


# Register cleanup handlers
atexit.register(_cleanup_betaflight_on_exit)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# --- Configuration ---
config = DEFAULT_CONFIG
config.simulation_time = 60.0  # 60 second simulation
config.sim_time_step = 0.001   # 1kHz physics
config.set_as_global()


def create_world(config: DroneConfig) -> tuple[el.World, el.EntityId]:
    """
    Create the simulation world with a drone entity.
    
    Returns:
        Tuple of (world, drone_entity_id)
    """
    world = el.World()
    
    # Initial state from config
    initial_pos = el.SpatialTransform(
        linear=jnp.array(config.initial_position),
        angular=el.Quaternion(jnp.array(config.initial_quaternion)),
    )
    initial_vel = el.SpatialMotion(
        linear=jnp.array(config.initial_velocity),
        angular=jnp.array(config.initial_angular_velocity),
    )
    inertia = el.SpatialInertia(
        mass=config.mass,
        inertia=jnp.array(config.inertia_diagonal),
    )
    
    # Spawn drone entity with physics and sensor components
    drone = world.spawn(
        [
            el.Body(
                world_pos=initial_pos,
                world_vel=initial_vel,
                inertia=inertia,
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
                viewport name=Viewport pos="drone.world_pos + (0,0,0,0, 3,3,3)" look_at="drone.world_pos" show_grid=#true active=#true
                vsplit share=0.4 {
                    graph "drone.motor_command" name="Motor Commands (from BF)"
                    graph "drone.motor_thrust" name="Motor Thrust"
                    graph "drone.accel" name="Accelerometer"
                }
            }
            vsplit name="State" {
                graph "drone.world_pos.linear" name="Position (ENU)"
                graph "drone.world_vel.linear" name="Velocity"
                graph "drone.gyro" name="Gyroscope"
            }
        }
        """,
        "betaflight-sitl.kdl",
    )
    
    return world, drone


def create_system(config: DroneConfig) -> el.System:
    """Create the complete simulation system (physics + sensors)."""
    physics = create_physics_system(config)
    sensors = create_sensor_system(config)
    return physics | sensors


@dataclass
class SITLState:
    """
    Shared state for SITL synchronization.
    
    This class holds the mutable state accessed by the post_step callback,
    including RC inputs, sensor buffer, and the bridge connection.
    """
    bridge: BetaflightSyncBridge
    sensor_buffer: SensorDataBuffer
    config: DroneConfig
    
    # RC inputs
    throttle: int = 1000
    roll: int = 1500
    pitch: int = 1500
    yaw: int = 1500
    arm: int = 1000  # 1000=disarmed, 1800=armed
    
    # Timing
    tick_count: int = 0
    sim_time: float = 0.0
    
    # Motor output (for logging/debugging)
    last_motors: np.ndarray = None
    
    def __post_init__(self):
        if self.last_motors is None:
            self.last_motors = np.zeros(4)
    
    def set_arm(self, armed: bool):
        """Set arm state."""
        self.arm = 1800 if armed else 1000
    
    def build_rc_packet(self) -> RCPacket:
        """Build RC packet from current state."""
        channels = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
        channels[0] = self.roll
        channels[1] = self.pitch
        channels[2] = self.throttle
        channels[3] = self.yaw
        channels[4] = self.arm  # AUX1 for arming
        return RCPacket(timestamp=self.sim_time, channels=channels)


def create_sitl_step_callback(state: SITLState):
    """
    Create the post_step callback for lockstep SITL synchronization.
    
    This callback is called after each physics tick. It:
    1. Builds FDM packet from sensor buffer
    2. Sends FDM + RC to Betaflight via synchronous bridge
    3. Receives motor response (blocking)
    4. Updates state with new motor values
    
    The motor values will be applied on the next physics tick via the
    external_control component mechanism.
    """
    def sitl_step(tick: int):
        # Update timing
        state.tick_count = tick
        state.sim_time = tick * state.config.sim_time_step
        state.sensor_buffer.timestamp = state.sim_time
        
        # Build FDM packet from sensor buffer
        fdm = state.sensor_buffer.build_fdm()
        
        # Build RC packet
        rc = state.build_rc_packet()
        
        try:
            # Perform synchronous step: send FDM+RC, wait for motors
            motors_bf = state.bridge.step(fdm, rc)
            
            # Remap motors from Betaflight to Elodin order
            motors_el = remap_motors_betaflight_to_elodin(motors_bf)
            state.last_motors = motors_el
            
        except TimeoutError as e:
            # On timeout, keep last motor values
            if state.tick_count > 100:  # Only warn after initialization
                print(f"[SITL] Timeout at tick {tick}: {e}")
        except Exception as e:
            print(f"[SITL] Error at tick {tick}: {e}")
    
    return sitl_step


def cleanup_stale_betaflight():
    """Kill any stale Betaflight SITL processes from previous runs."""
    try:
        result = subprocess.run(
            ["pkill", "-f", "betaflight_SITL"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("Cleaned up stale Betaflight process")
            time.sleep(0.5)  # Give ports time to be released
    except Exception:
        pass  # pkill not found or no processes to kill


def run_with_editor():
    """Run the simulation with the Elodin editor (3D visualization)."""
    print("Starting Betaflight SITL Simulation with Editor")
    print("=" * 50)
    
    # Kill any stale Betaflight processes from previous runs
    cleanup_stale_betaflight()
    
    # Create world and system
    world, drone = create_world(config)
    system = create_system(config)
    
    # Register Betaflight SITL as an external process
    betaflight_path = Path(__file__).parent / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"
    
    if betaflight_path.exists():
        betaflight_recipe = el.s10.PyRecipe.process(
            name="Betaflight SITL",
            cmd=str(betaflight_path),
            cwd=str(Path(__file__).parent),
        )
        world.recipe(betaflight_recipe)
        print(f"Registered Betaflight SITL: {betaflight_path}")
    else:
        print(f"WARNING: Betaflight SITL not found at {betaflight_path}")
        print("         Run ./build.sh to build it first")
    
    # Create SITL state and bridge
    bridge = BetaflightSyncBridge(timeout_ms=50)
    sensor_buffer = SensorDataBuffer()
    state = SITLState(bridge=bridge, sensor_buffer=sensor_buffer, config=config)
    
    # Create post_step callback
    sitl_step = create_sitl_step_callback(state)
    
    print("\nStarting simulation...")
    print("  - Betaflight SITL will be started automatically")
    print("  - Configure via CLI: socat PTY,link=/tmp/bf,rawer TCP:localhost:5761")
    print("  - Use arrow keys or mouse in editor to control view")
    print()
    
    # Start bridge
    bridge.start()
    
    try:
        world.run(
            system,
            sim_time_step=config.sim_time_step,
            run_time_step=1.0 / 60.0,  # 60 FPS for editor
            max_ticks=config.total_sim_ticks,
            post_step=sitl_step,
            interactive=True,
        )
    finally:
        bridge.stop()


def run_lockstep_test():
    """
    Run a lockstep SITL test demonstrating tight timing synchronization.
    
    This test uses SIMULATOR_GYROPID_SYNC for lockstep execution:
    - Each physics tick triggers exactly one Betaflight PID iteration
    - Deterministic, reproducible simulation
    - Can run faster than real-time
    """
    print("Starting Betaflight SITL Lockstep Test")
    print("=" * 50)
    print("This test demonstrates tight timing synchronization using")
    print("SIMULATOR_GYROPID_SYNC for deterministic simulation.")
    
    # Kill any stale Betaflight processes from previous runs
    cleanup_stale_betaflight()
    print()
    
    # Check if Betaflight SITL exists
    betaflight_path = Path(__file__).parent / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"
    if not betaflight_path.exists():
        print(f"ERROR: Betaflight SITL not found at {betaflight_path}")
        print("       Run ./build.sh to build it first")
        print("       (This will also enable SIMULATOR_GYROPID_SYNC)")
        return
    
    # Start Betaflight SITL
    global _betaflight_process
    print(f"Starting Betaflight SITL: {betaflight_path}")
    betaflight_proc = subprocess.Popen(
        [str(betaflight_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).parent),
    )
    _betaflight_process = betaflight_proc  # Track for cleanup on exit
    
    # Give Betaflight time to start
    time.sleep(2)
    
    # Create world and system
    world, drone = create_world(config)
    system = create_system(config)
    
    # Create SITL components
    bridge = BetaflightSyncBridge(timeout_ms=100)
    sensor_buffer = SensorDataBuffer()
    state = SITLState(bridge=bridge, sensor_buffer=sensor_buffer, config=config)
    
    # Track if we ever saw motor response (for success detection)
    max_motor_seen = 0.0
    
    # Test phases
    BOOTGRACE_DURATION = 5.0
    ARM_DURATION = 2.0
    THROTTLE_DURATION = 3.0
    DISARM_DURATION = 1.0
    total_duration = BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION + DISARM_DURATION
    
    # Create post_step callback with phase logic
    def sitl_step_with_phases(tick: int):
        nonlocal max_motor_seen
        state.tick_count = tick
        state.sim_time = tick * state.config.sim_time_step
        state.sensor_buffer.timestamp = state.sim_time
        
        # Determine phase and set RC accordingly
        if state.sim_time < BOOTGRACE_DURATION:
            phase = "bootgrace"
            state.arm = 1000
            state.throttle = 1000
        elif state.sim_time < BOOTGRACE_DURATION + ARM_DURATION:
            phase = "arming"
            state.arm = 1800
            state.throttle = 1000
        elif state.sim_time < BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION:
            phase = "throttle"
            state.arm = 1800
            state.throttle = 1400
        else:
            phase = "disarm"
            state.arm = 1000
            state.throttle = 1000
        
        # Build packets
        fdm = state.sensor_buffer.build_fdm()
        rc = state.build_rc_packet()
        
        try:
            motors_bf = state.bridge.step(fdm, rc)
            motors_el = remap_motors_betaflight_to_elodin(motors_bf)
            state.last_motors = motors_el
            # Track max motor value seen for success detection
            max_motor_seen = max(max_motor_seen, np.max(motors_el))
        except TimeoutError:
            pass  # Keep last motors
    
    # Build executor
    executor = world.build(system)
    
    print(f"\nRunning lockstep simulation ({total_duration:.0f}s total):")
    print(f"  - BOOTGRACE: {BOOTGRACE_DURATION:.0f}s (waiting for Betaflight)")
    print(f"  - ARMING: {ARM_DURATION:.0f}s (AUX1=1800)")
    print(f"  - THROTTLE: {THROTTLE_DURATION:.0f}s (40% power)")
    print(f"  - DISARM: {DISARM_DURATION:.0f}s")
    print()
    
    bridge.start()
    
    try:
        last_print_time = time.time()
        start_time = time.time()
        
        total_ticks = int(total_duration / config.sim_time_step)
        
        for tick in range(total_ticks):
            # Call SITL step (handles phases, sends FDM, receives motors)
            sitl_step_with_phases(tick)
            
            # Run one physics tick
            executor.run(1)
            
            # Print status every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                # Determine current phase
                if state.sim_time < BOOTGRACE_DURATION:
                    phase = "bootgrace"
                elif state.sim_time < BOOTGRACE_DURATION + ARM_DURATION:
                    phase = "arming"
                elif state.sim_time < BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION:
                    phase = "throttle"
                else:
                    phase = "disarm"
                
                motors = state.last_motors
                armed = "ARMED" if np.any(motors > 0.02) else "DISARMED"
                
                # Calculate simulation rate
                elapsed = current_time - start_time
                sim_rate = state.sim_time / elapsed if elapsed > 0 else 0
                
                print(f"  t={state.sim_time:5.1f}s | {phase:10s} | {armed:8s} | "
                      f"motors=[{motors[0]:.3f}, {motors[1]:.3f}, {motors[2]:.3f}, {motors[3]:.3f}] | "
                      f"{sim_rate:.1f}x realtime")
                last_print_time = current_time
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        bridge.stop()
        betaflight_proc.terminate()
        try:
            betaflight_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            betaflight_proc.kill()
        _betaflight_process = None  # Clear so atexit doesn't double-cleanup
    
    # Summary
    elapsed = time.time() - start_time
    example_dir = Path(__file__).parent
    print()
    print("=" * 50)
    print("Lockstep test complete!")
    print(f"  Simulated: {state.sim_time:.1f}s in {elapsed:.1f}s ({state.sim_time/elapsed:.1f}x realtime)")
    print(f"  Total ticks: {state.tick_count}")
    print(f"  Sync steps: {bridge.step_count}")
    print(f"  Max motor value seen: {max_motor_seen:.3f}")
    print()
    if bridge.step_count > 0 and max_motor_seen > 0.1:
        print("SUCCESS: Lockstep SITL integration working!")
    else:
        print("WARNING: No motor response received. Check:")
        print("  1. Betaflight built with SIMULATOR_GYROPID_SYNC enabled")
        print("  2. Arm switch configured: aux 0 0 0 1700 2100 0 0")
        print(f"  3. eeprom.bin present in {example_dir}")


def run_headless_test():
    """
    Run a headless test using the async bridge (legacy mode).
    
    For lockstep testing, use run_lockstep_test() instead.
    """
    print("Starting Betaflight SITL Headless Test (Legacy Mode)")
    print("=" * 50)
    print("NOTE: For lockstep testing, use: python3 main.py lockstep")
    
    # Kill any stale Betaflight processes from previous runs
    cleanup_stale_betaflight()
    print()
    
    # Check if Betaflight SITL exists
    betaflight_path = Path(__file__).parent / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"
    if not betaflight_path.exists():
        print(f"ERROR: Betaflight SITL not found at {betaflight_path}")
        print("       Run ./build.sh to build it first")
        return
    
    # Start Betaflight SITL
    global _betaflight_process
    print(f"Starting Betaflight SITL: {betaflight_path}")
    betaflight_proc = subprocess.Popen(
        [str(betaflight_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).parent),
    )
    _betaflight_process = betaflight_proc  # Track for cleanup on exit
    
    time.sleep(2)
    
    # Create world and system
    world, drone = create_world(config)
    system = create_system(config)
    
    # Create async bridge for legacy mode
    bridge = BetaflightBridge()
    bridge.start()
    
    # Build executor
    executor = world.build(system)
    
    # Test timing
    BOOTGRACE_DURATION = 5.0
    ARM_DURATION = 2.0
    THROTTLE_DURATION = 3.0
    DISARM_DURATION = 1.0
    total_duration = BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION + DISARM_DURATION
    
    print(f"\nRunning simulation ({total_duration:.0f}s total):")
    
    sim_time = 0.0
    rc_arm = 1000
    rc_throttle = 1000
    
    try:
        last_print_time = time.time()
        
        while sim_time < total_duration:
            # Determine phase
            if sim_time < BOOTGRACE_DURATION:
                phase = "bootgrace"
                rc_arm = 1000
                rc_throttle = 1000
            elif sim_time < BOOTGRACE_DURATION + ARM_DURATION:
                phase = "arming"
                rc_arm = 1800
                rc_throttle = 1000
            elif sim_time < BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION:
                phase = "throttle"
                rc_arm = 1800
                rc_throttle = 1400
            else:
                phase = "disarm"
                rc_arm = 1000
                rc_throttle = 1000
            
            # Send RC and FDM
            bridge.send_rc_channels(
                throttle=rc_throttle,
                roll=1500,
                pitch=1500,
                yaw=1500,
                aux=[rc_arm, 1500, 1500, 1500],
                timestamp=sim_time,
            )
            
            fdm = FDMPacket(
                timestamp=sim_time,
                imu_angular_velocity_rpy=np.array([0.0, 0.0, 0.0]),
                imu_linear_acceleration_xyz=np.array([0.0, 0.0, 9.81]),
                imu_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                velocity_xyz=np.array([0.0, 0.0, 0.0]),
                position_xyz=np.array([0.0, 0.0, 0.0]),
                pressure=101325.0,
            )
            bridge.send_fdm(fdm)
            
            # Get motors
            motors = bridge.get_motors()
            
            # Run physics
            executor.run(1)
            sim_time += config.sim_time_step
            
            # Print status
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                armed = "ARMED" if np.any(motors > 0.02) else "DISARMED"
                print(f"  t={sim_time:5.1f}s | {phase:10s} | {armed:8s} | "
                      f"motors=[{motors[0]:.3f}, {motors[1]:.3f}, {motors[2]:.3f}, {motors[3]:.3f}]")
                last_print_time = current_time
            
            time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nInterrupted")
        
    finally:
        bridge.stop()
        betaflight_proc.terminate()
        try:
            betaflight_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            betaflight_proc.kill()
        _betaflight_process = None  # Clear so atexit doesn't double-cleanup
    
    print("\nHeadless test complete!")


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    if "lockstep" in args or "sync" in args:
        run_lockstep_test()
    elif "bench" in args or "headless" in args or "test" in args:
        run_headless_test()
    else:
        run_with_editor()


if __name__ == "__main__":
    main()
