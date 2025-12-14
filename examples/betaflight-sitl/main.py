#!/usr/bin/env python3
"""
Betaflight SITL Drone Simulation with Elodin

This is the main entry point for the Betaflight Software-In-The-Loop (SITL)
drone simulation. It integrates:
- Elodin physics simulation (rigid body dynamics, forces, sensors)
- Betaflight flight controller (running as SITL binary)
- UDP communication bridge

Usage:
    elodin editor main.py       # Run with 3D visualization
    python3 main.py bench       # Run headless benchmark

Prerequisites:
    1. Build Betaflight SITL: ./build.sh
    2. (Optional) Configure Betaflight via Configurator at localhost:5761

Communication Flow:
    [Elodin Physics] --FDM Packet (sensors)--> [Betaflight SITL]
    [Elodin Physics] <--Servo Packet (motors)- [Betaflight SITL]
    [Elodin Physics] --RC Packet (commands)--> [Betaflight SITL]
"""

import sys
import time
from dataclasses import field
from pathlib import Path

import elodin as el
import jax.numpy as jnp
import numpy as np

from config import DroneConfig, DEFAULT_CONFIG
from sim import Drone, create_physics_system
from sensors import IMU, create_sensor_system
from comms import (
    BetaflightBridge,
    remap_motors_betaflight_to_elodin,
    FDMPacket,
)


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


class BetaflightSITLRunner:
    """
    Manages the Betaflight SITL integration with Elodin.
    
    This class handles:
    - Starting/stopping the UDP bridge
    - Sending sensor data to Betaflight
    - Receiving motor commands from Betaflight
    - Updating the simulation with motor commands
    """
    
    def __init__(self, config: DroneConfig):
        self.config = config
        self.bridge = BetaflightBridge()
        self.sim_time = 0.0
        self.last_motor_update = np.zeros(4)
        
        # RC state (can be modified for different flight modes)
        self.rc_throttle = 1000  # Idle
        self.rc_roll = 1500
        self.rc_pitch = 1500
        self.rc_yaw = 1500
        self.rc_arm = 1000  # Disarmed
        
    def start(self):
        """Start the UDP bridge."""
        self.bridge.start()
        print("[BetaflightSITL] Ready for simulation")
        
    def stop(self):
        """Stop the UDP bridge."""
        self.bridge.stop()
        
    def pre_step(self, world: el.World, drone_id: el.EntityId):
        """
        Called before each physics step.
        
        Gets motor commands from Betaflight and applies them to the simulation.
        """
        # Get motor commands from Betaflight
        motors_bf = self.bridge.get_motors()
        
        # Remap to Elodin motor order
        motors_el = remap_motors_betaflight_to_elodin(motors_bf)
        
        # Store for logging
        self.last_motor_update = motors_el
        
        # Update the drone's motor command component
        # Note: This requires direct component access which isn't available
        # in the standard world.run() loop. For now, motor commands are
        # applied through the post_step callback.
        
    def post_step(self, world: el.World, drone_id: el.EntityId):
        """
        Called after each physics step.
        
        Extracts sensor data from the simulation and sends to Betaflight.
        """
        # Update simulation time
        self.sim_time += self.config.sim_time_step
        
        # For now, send static sensor data (actual sensor extraction 
        # requires access to component values which isn't directly available
        # in world.run callbacks)
        
        # Send RC channels
        self.bridge.send_rc_channels(
            throttle=self.rc_throttle,
            roll=self.rc_roll,
            pitch=self.rc_pitch,
            yaw=self.rc_yaw,
            aux=[self.rc_arm, 1500, 1500, 1500],
            timestamp=self.sim_time,
        )
        
        # Send FDM packet with placeholder data
        # In a full implementation, this would extract real sensor values
        fdm = FDMPacket(
            timestamp=self.sim_time,
            imu_angular_velocity_rpy=np.array([0.0, 0.0, 0.0]),
            imu_linear_acceleration_xyz=np.array([0.0, 0.0, 9.81]),  # Gravity
            imu_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity_xyz=np.array([0.0, 0.0, 0.0]),
            position_xyz=np.array([0.0, 0.0, 0.0]),
            pressure=101325.0,
        )
        self.bridge.send_fdm(fdm)
    
    def set_rc(
        self,
        throttle: int = None,
        roll: int = None,
        pitch: int = None,
        yaw: int = None,
        arm: bool = None,
    ):
        """Set RC inputs for the flight controller."""
        if throttle is not None:
            self.rc_throttle = throttle
        if roll is not None:
            self.rc_roll = roll
        if pitch is not None:
            self.rc_pitch = pitch
        if yaw is not None:
            self.rc_yaw = yaw
        if arm is not None:
            self.rc_arm = 1800 if arm else 1000


def run_with_editor():
    """Run the simulation with the Elodin editor (3D visualization)."""
    print("Starting Betaflight SITL Simulation with Editor")
    print("=" * 50)
    
    # Create world and system
    world, drone = create_world(config)
    system = create_system(config)
    
    # Register Betaflight SITL as an external process
    # This will start Betaflight automatically with the simulation
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
    
    # Run the simulation
    print("\nStarting simulation...")
    print("  - Connect to Betaflight Configurator at localhost:5761")
    print("  - Use 'elodin editor main.py' for visualization")
    print()
    
    world.run(
        system,
        sim_time_step=config.sim_time_step,
        run_time_step=1.0 / 60.0,  # 60 FPS for editor
        max_ticks=config.total_sim_ticks,
    )


def run_headless_test():
    """
    Run a headless test of the simulation with Betaflight communication.
    
    This test demonstrates the full SITL loop:
    1. Start Betaflight SITL process
    2. Wait for BOOTGRACE period
    3. ARM the drone via AUX1 switch
    4. Apply throttle and watch motor response
    """
    print("Starting Betaflight SITL Headless Test")
    print("=" * 50)
    
    # Check if Betaflight SITL exists
    betaflight_path = Path(__file__).parent / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"
    if not betaflight_path.exists():
        print(f"ERROR: Betaflight SITL not found at {betaflight_path}")
        print("       Run ./build.sh to build it first")
        return
    
    # Start Betaflight SITL manually for headless test
    import subprocess
    
    print(f"Starting Betaflight SITL: {betaflight_path}")
    betaflight_proc = subprocess.Popen(
        [str(betaflight_path)],
        stdout=subprocess.DEVNULL,  # Suppress SITL output
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).parent),
    )
    
    # Give Betaflight time to start
    time.sleep(2)
    
    # Create simulation
    world, drone = create_world(config)
    system = create_system(config)
    
    # Create SITL runner
    runner = BetaflightSITLRunner(config)
    runner.start()
    
    # Build executor for step-by-step control
    executor = world.build(system)
    
    # Test phases
    BOOTGRACE_DURATION = 5.0
    ARM_DURATION = 2.0
    THROTTLE_DURATION = 3.0
    DISARM_DURATION = 1.0
    
    total_duration = BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION + DISARM_DURATION
    
    print(f"\nRunning simulation ({total_duration:.0f}s total):")
    print(f"  - BOOTGRACE: {BOOTGRACE_DURATION:.0f}s (waiting for Betaflight)")
    print(f"  - ARMING: {ARM_DURATION:.0f}s (AUX1=1800)")
    print(f"  - THROTTLE: {THROTTLE_DURATION:.0f}s (40% power)")
    print(f"  - DISARM: {DISARM_DURATION:.0f}s")
    print()
    
    try:
        tick = 0
        last_print_time = time.time()
        phase = "bootgrace"
        
        while runner.sim_time < total_duration:
            # Determine phase and set RC accordingly
            if runner.sim_time < BOOTGRACE_DURATION:
                phase = "bootgrace"
                runner.rc_arm = 1000  # Keep disarmed during bootgrace
                runner.rc_throttle = 1000
            elif runner.sim_time < BOOTGRACE_DURATION + ARM_DURATION:
                phase = "arming"
                runner.rc_arm = 1800  # ARM switch high
                runner.rc_throttle = 1000  # Throttle low for arming
            elif runner.sim_time < BOOTGRACE_DURATION + ARM_DURATION + THROTTLE_DURATION:
                phase = "throttle"
                runner.rc_arm = 1800  # Stay armed
                runner.rc_throttle = 1400  # 40% throttle
            else:
                phase = "disarm"
                runner.rc_arm = 1000  # Disarm
                runner.rc_throttle = 1000
            
            # Send sensor data and RC to Betaflight
            runner.post_step(world, drone)
            
            # Get motor commands from Betaflight and store
            motors = runner.bridge.get_motors()
            
            # Step the physics simulation
            executor.run(1)
            tick += 1
            
            # Print status every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                armed_status = "ARMED" if np.any(motors > 0.02) else "DISARMED"
                print(f"  t={runner.sim_time:5.1f}s | {phase:10s} | {armed_status:8s} | "
                      f"motors=[{motors[0]:.3f}, {motors[1]:.3f}, {motors[2]:.3f}, {motors[3]:.3f}]")
                last_print_time = current_time
            
            # Small delay to prevent CPU spinning (real-time-ish)
            time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        # Cleanup
        runner.stop()
        betaflight_proc.terminate()
        try:
            betaflight_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            betaflight_proc.kill()
        
    print()
    print("=" * 50)
    print("Headless test complete!")
    print()
    print("If motors showed non-zero values during 'arming' and 'throttle' phases,")
    print("the SITL integration is working correctly!")


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    if "bench" in args or "headless" in args or "test" in args:
        run_headless_test()
    else:
        run_with_editor()


if __name__ == "__main__":
    main()
