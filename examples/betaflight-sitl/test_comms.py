#!/usr/bin/env python3
"""
Standalone UDP Test Script for Betaflight SITL Communication

This script tests the UDP communication with Betaflight SITL without
requiring the full Elodin simulation. It verifies:
1. FDM packets are received by Betaflight
2. RC packets are received by Betaflight
3. Motor/Servo packets are received from Betaflight

Usage:
    1. Start Betaflight SITL: ./betaflight/obj/main/betaflight_SITL.elf
    2. Run this test: python3 test_comms.py

Expected output:
    - Betaflight should show sensor activity
    - Motor values should appear (likely 0 until armed)
"""

import time
import signal
import numpy as np
from comms import (
    BetaflightBridge,
    remap_motors_betaflight_to_elodin,
)


# Global flag for clean shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global running
    print("\nShutting down...")
    running = False


def test_basic_communication():
    """Test basic packet send/receive with Betaflight SITL."""
    print("=" * 60)
    print("Betaflight SITL Communication Test")
    print("=" * 60)
    print()
    print("Make sure Betaflight SITL is running:")
    print("  ./betaflight/obj/main/betaflight_SITL.elf")
    print()
    print("Press Ctrl+C to stop")
    print()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create and start bridge
    bridge = BetaflightBridge()
    bridge.start()

    # Give some time for sockets to initialize
    time.sleep(0.5)

    # Simulation state
    sim_time = 0.0
    dt = 0.001  # 1kHz update rate

    # Counters
    packets_sent = 0
    motor_updates = 0
    last_print_time = time.time()

    def on_motor_update(motors):
        nonlocal motor_updates
        motor_updates += 1

    bridge.set_motor_callback(on_motor_update)

    print("Starting communication loop...")
    print("-" * 60)

    try:
        while running:
            loop_start = time.time()

            # Create simulated sensor data
            # Simulate a stationary drone at ground level with gravity
            angular_velocity = np.array([0.0, 0.0, 0.0])  # No rotation
            linear_acceleration = np.array([0.0, 0.0, 9.81])  # Gravity (up in ENU)
            orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Level
            velocity = np.array([0.0, 0.0, 0.0])  # Stationary
            position = np.array([0.0, 0.0, 0.0])  # At origin

            # Send FDM packet
            bridge.send_state(
                timestamp=sim_time,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                orientation_quat=orientation_quat,
                velocity=velocity,
                position=position,
                pressure=101325.0,
            )
            packets_sent += 1

            # Send RC packet (idle throttle, centered sticks)
            # Arm switch on AUX1 (channel 5) - set to 1800 to try arming
            bridge.send_rc_channels(
                throttle=1000,  # Idle
                roll=1500,  # Centered
                pitch=1500,  # Centered
                yaw=1500,  # Centered
                aux=[1000, 1000, 1000, 1000],  # AUX channels (1000 = disarmed)
                timestamp=sim_time,
            )

            # Get motor values
            motors = bridge.get_motors()
            motors_raw = bridge.get_motors_raw()

            # Print status every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                # Remap motors to Elodin order for display
                motors_elodin = remap_motors_betaflight_to_elodin(motors)

                print(
                    f"Time: {sim_time:.2f}s | "
                    f"Packets: {packets_sent} | "
                    f"Motor updates: {motor_updates} | "
                    f"Connected: {bridge.is_connected}"
                )
                print(
                    f"  Motors (BF order):  [{motors[0]:.3f}, {motors[1]:.3f}, "
                    f"{motors[2]:.3f}, {motors[3]:.3f}]"
                )
                print(
                    f"  Motors (Elodin):    [{motors_elodin[0]:.3f}, {motors_elodin[1]:.3f}, "
                    f"{motors_elodin[2]:.3f}, {motors_elodin[3]:.3f}]"
                )
                print(
                    f"  Raw PWM (first 4):  [{motors_raw[0]:.0f}, {motors_raw[1]:.0f}, "
                    f"{motors_raw[2]:.0f}, {motors_raw[3]:.0f}]"
                )
                print()

                last_print_time = current_time

            # Update simulation time
            sim_time += dt

            # Sleep to maintain approximately real-time
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        bridge.stop()

    print("-" * 60)
    print(f"Test complete. Sent {packets_sent} packets, received {motor_updates} motor updates.")


def test_arm_sequence():
    """
    Test the arming sequence with Betaflight SITL.

    Betaflight arming requires:
    1. BOOTGRACE period expired (~5 seconds after boot)
    2. Low throttle
    3. Stick positions centered
    4. AUX channel switch high (must be configured via CLI first)
    5. Not in CLI mode
    """
    print("=" * 60)
    print("Betaflight SITL Arming Sequence Test")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  1. SITL running: ./betaflight/obj/main/betaflight_SITL.elf")
    print("  2. ARM switch configured via CLI:")
    print("     socat pty,raw,echo=0,link=/tmp/bf tcp:localhost:5761 &")
    print("     screen /tmp/bf")
    print("     # status")
    print("     aux 0 0 0 1700 2100 0 0")
    print("     save")
    print()

    signal.signal(signal.SIGINT, signal_handler)

    bridge = BetaflightBridge()
    bridge.start()
    time.sleep(0.5)

    sim_time = 0.0
    dt = 0.01  # 100Hz for real-time test

    # Phases: bootgrace, arming, armed, throttle_test, disarming
    phase = "bootgrace"
    phase_start_time = 0.0
    last_print_time = time.time()

    print("Starting arming sequence...")
    print("-" * 60)

    try:
        while running:
            loop_start = time.time()

            # Send sensor data (stationary drone)
            bridge.send_state(
                timestamp=sim_time,
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                linear_acceleration=np.array([0.0, 0.0, 9.81]),
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                position=np.array([0.0, 0.0, 0.1]),  # Slight altitude
                pressure=101325.0,
            )

            # Phase-based RC commands
            if phase == "bootgrace":
                # Wait for BOOTGRACE to clear (~5 seconds)
                # Keep AUX1 LOW during this phase
                bridge.send_rc_channels(
                    throttle=1000,
                    roll=1500,
                    pitch=1500,
                    yaw=1500,
                    aux=[1000, 1500, 1500, 1500],  # AUX1 low
                    timestamp=sim_time,
                )
                if sim_time - phase_start_time > 5.0:
                    print(f"[{sim_time:.1f}s] BOOTGRACE cleared, setting AUX1=1800 to ARM...")
                    phase = "arming"
                    phase_start_time = sim_time

            elif phase == "arming":
                # Set arm switch (AUX1 = channel 5) high
                bridge.send_rc_channels(
                    throttle=1000,  # Must be low to arm
                    roll=1500,
                    pitch=1500,
                    yaw=1500,
                    aux=[1800, 1500, 1500, 1500],  # AUX1 high = arm
                    timestamp=sim_time,
                )

                motors = bridge.get_motors()
                # If motors spin up (idle ~5.5%), we're armed
                if np.any(motors > 0.02):
                    print(f"[{sim_time:.1f}s] *** ARMED! *** Motors at idle: {motors}")
                    phase = "armed"
                    phase_start_time = sim_time
                elif sim_time - phase_start_time > 3.0:
                    print(f"[{sim_time:.1f}s] Arming failed after 3 seconds.")
                    print("  Check Betaflight CLI 'status' for arming disable flags.")
                    print("  Make sure ARM switch is configured: aux 0 0 0 1700 2100 0 0")
                    phase = "failed"
                    phase_start_time = sim_time

            elif phase == "armed":
                # Stay armed, then test throttle response
                bridge.send_rc_channels(
                    throttle=1000,
                    roll=1500,
                    pitch=1500,
                    yaw=1500,
                    aux=[1800, 1500, 1500, 1500],
                    timestamp=sim_time,
                )

                if sim_time - phase_start_time > 1.0:
                    print(f"[{sim_time:.1f}s] Testing throttle response...")
                    phase = "throttle_test"
                    phase_start_time = sim_time

            elif phase == "throttle_test":
                # Raise throttle to 40%
                bridge.send_rc_channels(
                    throttle=1400,  # 40% throttle
                    roll=1500,
                    pitch=1500,
                    yaw=1500,
                    aux=[1800, 1500, 1500, 1500],
                    timestamp=sim_time,
                )

                motors = bridge.get_motors()
                if sim_time - phase_start_time > 0.5 and np.any(motors > 0.3):
                    print(f"[{sim_time:.1f}s] *** SUCCESS! *** Motors responding: {motors}")
                    phase = "disarming"
                    phase_start_time = sim_time
                elif sim_time - phase_start_time > 2.0:
                    print(f"[{sim_time:.1f}s] Throttle test timeout. Motors: {motors}")
                    phase = "disarming"
                    phase_start_time = sim_time

            elif phase == "disarming":
                # Disarm by lowering AUX1
                bridge.send_rc_channels(
                    throttle=1000,
                    roll=1500,
                    pitch=1500,
                    yaw=1500,
                    aux=[1000, 1500, 1500, 1500],
                    timestamp=sim_time,
                )

                motors = bridge.get_motors()
                if sim_time - phase_start_time > 1.0:
                    print(f"[{sim_time:.1f}s] DISARMED. Final motors: {motors}")
                    break

            elif phase == "failed":
                # Keep sending data so user can investigate
                bridge.send_rc_channels(
                    throttle=1000,
                    roll=1500,
                    pitch=1500,
                    yaw=1500,
                    aux=[1000, 1500, 1500, 1500],
                    timestamp=sim_time,
                )
                if sim_time - phase_start_time > 5.0:
                    print("Exiting after failure. Check configuration and retry.")
                    break

            # Print status periodically
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                motors = bridge.get_motors()
                print(f"[{sim_time:.1f}s] Phase: {phase:12s} | Motors: {motors}")
                last_print_time = current_time

            sim_time += dt

            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        bridge.stop()

    print("-" * 60)
    print("Arming test complete.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Betaflight SITL UDP communication")
    parser.add_argument("--arm", action="store_true", help="Run arming sequence test")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Test duration in seconds (default: run until Ctrl+C)",
    )

    args = parser.parse_args()

    if args.arm:
        test_arm_sequence()
    else:
        test_basic_communication()


if __name__ == "__main__":
    main()
