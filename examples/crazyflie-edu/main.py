#!/usr/bin/env python3
"""
Crazyflie Educational Simulation - Main Entry Point

This simulation communicates with a C-based SITL binary via UDP,
enabling students to write the same C code that runs on real hardware.

Modes:
    SITL (default): Physics simulation + C SITL binary
    HITL (--hitl): Real sensors from hardware via Crazyradio, control code runs on drone

Run (from repo root):
    ./examples/crazyflie-edu/sitl/build.sh                     # Build SITL first
    elodin editor examples/crazyflie-edu/main.py               # SITL with 3D visualization
    elodin editor examples/crazyflie-edu/main.py --hitl        # HITL with real hardware
    elodin run examples/crazyflie-edu/main.py                  # SITL headless

Keyboard Controls (same in both SITL and HITL):
    Q           - Toggle armed state
    Left Shift  - Blue button (dead man switch, hold to enable motors)
    E/R/T       - Yellow/Green/Red buttons
"""

import atexit
import socket
import struct
import subprocess
import sys
import time
import typing as ty
from dataclasses import dataclass, field
from pathlib import Path

import elodin as el
import jax.numpy as jnp
import numpy as np

from config import CrazyflieConfig, create_default_config
from sim import CrazyflieDrone, create_physics_system, thrust_visualization, propeller_animation
from sensors import IMU, create_imu_system

# Try to import keyboard controller (optional dependency)
try:
    from keyboard_controller import KeyboardController

    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: keyboard_controller not available (install pynput)")

# Try to import cflib for HITL mode
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False

# =============================================================================
# Mode Detection
# =============================================================================

HITL_MODE = "--hitl" in sys.argv
DEFAULT_CF_URI = "radio://0/80/2M/E7E7E7E7E7"

# =============================================================================
# Configuration
# =============================================================================

# UDP ports (matching C SITL)
PORT_SENSORS = 9003  # Python -> C (sensor data)
PORT_MOTORS = 9002  # C -> Python (motor commands)
DEFAULT_HOST = "127.0.0.1"

# SITL binary path
SITL_PATH = Path(__file__).parent / "sitl" / "sitl_main"


# --- Clean up stale processes from previous runs ---
def cleanup_stale_sitl():
    """Kill any stale SITL processes from previous runs."""
    try:
        subprocess.run(["pkill", "-f", "sitl_main"], capture_output=True, timeout=5)
        time.sleep(0.2)  # Brief pause to let the process terminate and ports release
    except Exception:
        pass


# Only cleanup when running directly
if "run" in sys.argv:
    cleanup_stale_sitl()


# =============================================================================
# UDP Packet Structures
# =============================================================================


@dataclass
class SensorPacket:
    """
    Sensor packet sent to C SITL.
    Must match sensor_packet_t in sitl_main.c
    """

    timestamp: float = 0.0
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    is_armed: bool = False
    button_blue: bool = False
    button_yellow: bool = False
    button_green: bool = False
    button_red: bool = False

    # Format: timestamp(d) + gyro(3f) + accel(3f) + buttons(5B) + padding(3B) = 40 bytes
    _FORMAT = "<d3f3f5B3x"
    SIZE = struct.calcsize(_FORMAT)

    def pack(self) -> bytes:
        return struct.pack(
            self._FORMAT,
            self.timestamp,
            self.gyro[0],
            self.gyro[1],
            self.gyro[2],
            self.accel[0],
            self.accel[1],
            self.accel[2],
            int(self.is_armed),
            int(self.button_blue),
            int(self.button_yellow),
            int(self.button_green),
            int(self.button_red),
        )


@dataclass
class MotorPacket:
    """
    Motor packet received from C SITL.
    Must match motor_packet_t in sitl_main.c
    """

    timestamp: float = 0.0
    motor_pwm: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint16))

    # Format: timestamp(d) + motors(4H) = 16 bytes
    _FORMAT = "<d4H"
    SIZE = struct.calcsize(_FORMAT)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MotorPacket":
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[: cls.SIZE])
        return cls(
            timestamp=values[0],
            motor_pwm=np.array(values[1:5], dtype=np.uint16),
        )


# =============================================================================
# SITL Bridge
# =============================================================================


class SITLBridge:
    """
    UDP bridge for communicating with C SITL binary.
    """

    def __init__(self, host: str = DEFAULT_HOST, timeout_ms: int = 100):
        self.host = host
        self.timeout_ms = timeout_ms
        self._sensor_sock: socket.socket | None = None
        self._motor_sock: socket.socket | None = None
        self._step_count = 0
        self._last_motors = np.zeros(4, dtype=np.uint16)

    def start(self) -> None:
        """Initialize UDP sockets."""
        # Socket for sending sensor data
        self._sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Socket for receiving motor data
        self._motor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._motor_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._motor_sock.bind(("0.0.0.0", PORT_MOTORS))
        self._motor_sock.settimeout(self.timeout_ms / 1000.0)

        # Drain any stale packets
        self._motor_sock.setblocking(False)
        try:
            while True:
                self._motor_sock.recv(1024)
        except BlockingIOError:
            pass
        self._motor_sock.setblocking(True)
        self._motor_sock.settimeout(self.timeout_ms / 1000.0)

        print(f"[SITLBridge] Started - sensors->{self.host}:{PORT_SENSORS}, motors<-:{PORT_MOTORS}")

    def stop(self) -> None:
        """Close sockets."""
        if self._sensor_sock:
            self._sensor_sock.close()
        if self._motor_sock:
            self._motor_sock.close()
        print(f"[SITLBridge] Stopped after {self._step_count} steps")

    def step(self, sensor_pkt: SensorPacket) -> np.ndarray:
        """
        Send sensor data and receive motor commands (lockstep).

        Returns:
            Motor PWM values (0-65535) as numpy array
        """
        if not self._sensor_sock or not self._motor_sock:
            raise RuntimeError("Bridge not started")

        # Send sensor packet
        data = sensor_pkt.pack()
        self._sensor_sock.sendto(data, (self.host, PORT_SENSORS))

        # Receive motor response
        try:
            data, _ = self._motor_sock.recvfrom(MotorPacket.SIZE)
            motor_pkt = MotorPacket.from_bytes(data)
            self._last_motors = motor_pkt.motor_pwm
            self._step_count += 1
            return self._last_motors
        except socket.timeout:
            # Return last known motors on timeout
            if self._step_count == 0:
                return np.zeros(4, dtype=np.uint16)
            return self._last_motors

    @property
    def step_count(self) -> int:
        return self._step_count


# =============================================================================
# HITL Bridge (Hardware-In-The-Loop via Crazyradio)
# =============================================================================


class HITLBridge:
    """
    Bridge for HITL mode - connects to real Crazyflie via Crazyradio.

    In HITL mode:
    - Control code runs ON the Crazyflie (user_code.c is flashed)
    - Keyboard inputs are sent as Crazyflie parameters
    - Real sensor data is read for visualization
    """

    def __init__(self, uri: str = DEFAULT_CF_URI):
        self.uri = uri
        self._cf: Crazyflie | None = None
        self._scf: SyncCrazyflie | None = None
        self._connected = False
        self._step_count = 0

        # Latest sensor data from hardware
        self._gyro = np.zeros(3, dtype=np.float32)
        self._accel = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Logging state
        self._log_config: LogConfig | None = None
        self._logging_started = False

    def _log_callback(self, timestamp, data, logconf):
        """Called when new sensor data is received from Crazyflie."""
        # Gyro data (deg/s in firmware, convert to rad/s)
        self._gyro[0] = np.deg2rad(data.get("gyro.x", 0.0))
        self._gyro[1] = np.deg2rad(data.get("gyro.y", 0.0))
        self._gyro[2] = np.deg2rad(data.get("gyro.z", 0.0))

        # Accel data (in g units)
        self._accel[0] = data.get("acc.x", 0.0)
        self._accel[1] = data.get("acc.y", 0.0)
        self._accel[2] = data.get("acc.z", 0.0)

    def _log_error(self, logconf, msg):
        """Called on logging error."""
        print(f"[HITLBridge] Logging error: {msg}")

    def connect(self) -> bool:
        """Connect to Crazyflie and start sensor logging."""
        if not CFLIB_AVAILABLE:
            print("[HITLBridge] ERROR: cflib not available")
            print("Install with: uv pip install cflib")
            return False

        print("[HITLBridge] Initializing Crazyradio drivers...")
        cflib.crtp.init_drivers()

        print(f"[HITLBridge] Connecting to {self.uri}...")

        try:
            self._cf = Crazyflie(rw_cache="./cf_cache")
            self._scf = SyncCrazyflie(self.uri, cf=self._cf)
            self._scf.open_link()

            print("[HITLBridge] Connected! Setting up sensor logging...")

            # Set up logging configuration (100 Hz)
            self._log_config = LogConfig(name="Sensors", period_in_ms=10)
            self._log_config.add_variable("gyro.x", "float")
            self._log_config.add_variable("gyro.y", "float")
            self._log_config.add_variable("gyro.z", "float")
            self._log_config.add_variable("acc.x", "float")
            self._log_config.add_variable("acc.y", "float")
            self._log_config.add_variable("acc.z", "float")

            self._cf.log.add_config(self._log_config)
            self._log_config.data_received_cb.add_callback(self._log_callback)
            self._log_config.error_cb.add_callback(self._log_error)
            self._log_config.start()
            self._logging_started = True

            self._connected = True
            print("[HITLBridge] Sensor logging started!")
            return True

        except Exception as e:
            print(f"[HITLBridge] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Crazyflie."""
        if self._logging_started and self._log_config:
            try:
                self._log_config.stop()
            except Exception:
                pass

        if self._scf:
            try:
                self._scf.close_link()
            except Exception:
                pass

        self._connected = False
        print(f"[HITLBridge] Disconnected after {self._step_count} steps")

    def set_control_params(
        self,
        is_armed: bool,
        button_blue: bool,
        button_yellow: bool = False,
        button_green: bool = False,
        button_red: bool = False,
    ) -> None:
        """Send control inputs to Crazyflie as parameters."""
        if not self._connected or not self._cf:
            return

        try:
            self._cf.param.set_value("userCtrl.armed", int(is_armed))
            self._cf.param.set_value("userCtrl.btnBlue", int(button_blue))
            self._cf.param.set_value("userCtrl.btnYellow", int(button_yellow))
            self._cf.param.set_value("userCtrl.btnGreen", int(button_green))
            self._cf.param.set_value("userCtrl.btnRed", int(button_red))
        except Exception as e:
            # Parameters might not exist if firmware doesn't have userCtrl
            if self._step_count == 0:
                print(f"[HITLBridge] Warning: Could not set parameters: {e}")
                print("[HITLBridge] Make sure firmware has userCtrl parameters")

    def get_sensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Get latest sensor readings."""
        self._step_count += 1
        return self._gyro.copy(), self._accel.copy()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def step_count(self) -> int:
        return self._step_count


# =============================================================================
# Simulation Time Component
# =============================================================================

SimTime = ty.Annotated[
    jnp.ndarray,
    el.Component("sim_time", el.ComponentType.F64),
]


@dataclass
class SimClock(el.Archetype):
    """Simulation clock archetype."""

    sim_time: SimTime = field(default_factory=lambda: jnp.array(0.0))


# =============================================================================
# Control Components
# =============================================================================

# Motor commands from SITL (PWM values 0-65535)
MotorCommand = ty.Annotated[
    jnp.ndarray,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "priority": 100,
            "element_names": "m1,m2,m3,m4",
            "external_control": "true",
        },
    ),
]

# Button components (controlled via keyboard)
ButtonBlue = ty.Annotated[
    jnp.ndarray,
    el.Component("button_blue", el.ComponentType.F64, metadata={"external_control": "true"}),
]
ButtonYellow = ty.Annotated[
    jnp.ndarray,
    el.Component("button_yellow", el.ComponentType.F64, metadata={"external_control": "true"}),
]
ButtonGreen = ty.Annotated[
    jnp.ndarray,
    el.Component("button_green", el.ComponentType.F64, metadata={"external_control": "true"}),
]
ButtonRed = ty.Annotated[
    jnp.ndarray,
    el.Component("button_red", el.ComponentType.F64, metadata={"external_control": "true"}),
]

IsArmedControl = ty.Annotated[
    jnp.ndarray,
    el.Component("is_armed_control", el.ComponentType.F64, metadata={"external_control": "true"}),
]


@dataclass
class Control(el.Archetype):
    """Control state archetype."""

    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))
    is_armed_control: IsArmedControl = field(default_factory=lambda: jnp.array(0.0))
    button_blue: ButtonBlue = field(default_factory=lambda: jnp.array(0.0))
    button_yellow: ButtonYellow = field(default_factory=lambda: jnp.array(0.0))
    button_green: ButtonGreen = field(default_factory=lambda: jnp.array(0.0))
    button_red: ButtonRed = field(default_factory=lambda: jnp.array(0.0))


# =============================================================================
# Systems
# =============================================================================


@el.map
def update_sim_time(time: SimTime) -> SimTime:
    """Advance simulation time."""
    config = CrazyflieConfig.get_global()
    return time + config.dt


# =============================================================================
# World Setup
# =============================================================================


def create_world() -> tuple[el.World, el.EntityId]:
    """Create the simulation world with a Crazyflie drone."""
    config = CrazyflieConfig.get_global()
    w = el.World()

    # Spawn the drone entity
    drone = w.spawn(
        [
            el.Body(
                world_pos=config.spatial_transform,
                inertia=config.spatial_inertia,
            ),
            CrazyflieDrone(),
            IMU(),
            Control(),
            SimClock(),
        ],
        name="crazyflie",
    )

    # Editor schematic
    schematic = """
        theme mode="dark" scheme="default"

        tabs {
            hsplit name="Simulation" {
                viewport name="3D View" pos="crazyflie.world_pos + (0,0,0,0, 0.2, 0.2, 0.2)" look_at="crazyflie.world_pos" show_grid=#true active=#true
                vsplit share=0.35 {
                    graph "crazyflie.gyro" name="Gyroscope (rad/s)"
                    graph "crazyflie.accel" name="Accelerometer (g)"
                    graph "crazyflie.motor_pwm" name="Motor PWM"
                }
            }
            vsplit name="Motors" {
                graph "crazyflie.motor_rpm" name="Motor RPM"
                graph "crazyflie.thrust" name="Motor Thrust (N)"
                graph "crazyflie.motor_command" name="Motor Command (from SITL)"
            }
            vsplit name="Controls" {
                graph "crazyflie.is_armed_control" name="Armed State (Q to toggle)"
                graph "crazyflie.button_blue" name="Blue Button (Shift)"
                graph "crazyflie.button_yellow" name="Yellow Button (E)"
                graph "crazyflie.button_green" name="Green Button (R)"
                graph "crazyflie.button_red" name="Red Button (T)"
            }
        }
        object_3d crazyflie.world_pos {
            glb path="crazyflie.glb" rotate="(0.0, 0.0, 0.0)" translate="(-0.01, 0.0, 0.0)" scale=0.7
            animate joint="Root.Propeller_0" rotation_vector="(0, crazyflie.propeller_angle[0], 0)"
            animate joint="Root.Propeller_1" rotation_vector="(0, crazyflie.propeller_angle[1], 0)"
            animate joint="Root.Propeller_2" rotation_vector="(0, crazyflie.propeller_angle[2], 0)"
            animate joint="Root.Propeller_3" rotation_vector="(0, crazyflie.propeller_angle[3], 0)"
        }

        // Motor position indicators
        vector_arrow "(0.707, -0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M1: FR" show_name=#true body_frame=#true {
            color yellow 10
        }
        vector_arrow "(0.707, 0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M2: FL" show_name=#true body_frame=#true {
            color yellow 10
        }
        vector_arrow "(-0.707, 0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M3: BL" show_name=#true body_frame=#true {
            color yellow 10
        }
        vector_arrow "(-0.707, -0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M4: BR" show_name=#true body_frame=#true {
            color yellow 10
        }

        // Rotor disc visualization
        object_3d "crazyflie.world_pos + (0,0,0,0, 0.0325, -0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color cyan 30
            }
        }
        object_3d "crazyflie.world_pos + (0,0,0,0, 0.0325, 0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color red 30
            }
        }
        object_3d "crazyflie.world_pos + (0,0,0,0, -0.0325, 0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color cyan 30
            }
        }
        object_3d "crazyflie.world_pos + (0,0,0,0, -0.0325, -0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color red 30
            }
        }

        // Thrust visualization arrows
        vector_arrow "crazyflie.thrust_viz_m1" origin="crazyflie.world_pos + (0,0,0,0, 0.0325, -0.0325, 0.013)" body_frame=#true {
            color cyan 20
        }
        vector_arrow "crazyflie.thrust_viz_m2" origin="crazyflie.world_pos + (0,0,0,0, 0.0325, 0.0325, 0.013)" body_frame=#true {
            color red 20
        }
        vector_arrow "crazyflie.thrust_viz_m3" origin="crazyflie.world_pos + (0,0,0,0, -0.0325, 0.0325, 0.013)" body_frame=#true {
            color cyan 20
        }
        vector_arrow "crazyflie.thrust_viz_m4" origin="crazyflie.world_pos + (0,0,0,0, -0.0325, -0.0325, 0.013)" body_frame=#true {
            color red 20
        }
    """

    w.schematic(schematic, "crazyflie-edu.kdl")

    return w, drone


def system(include_physics: bool = True) -> el.System:
    """Create the simulation system.

    Args:
        include_physics: If True (SITL mode), include physics simulation.
                        If False (HITL mode), skip physics (real world is source of truth).
    """
    clock = update_sim_time
    visualization = thrust_visualization | propeller_animation

    if include_physics:
        physics = create_physics_system()
        sensors = create_imu_system()
        return clock | physics | sensors | visualization
    else:
        # HITL mode: no physics, sensors come from hardware
        return clock | visualization


# =============================================================================
# Post-Step Callback (SITL Integration)
# =============================================================================

# Global state
_keyboard_controller = None
_sitl_bridge = None
_sitl_process = None  # Manual process when not using s10
_last_print_time = [0.0]
_init_done = [False]


def check_sitl_binary() -> bool:
    """Check if SITL binary exists and suggest building if not."""
    if not SITL_PATH.exists():
        print(f"\n[ERROR] SITL binary not found at: {SITL_PATH}")
        print("\nTo build the SITL binary, run:")
        print(f"  cd {SITL_PATH.parent}")
        print("  ./build.sh")
        print()
        return False
    return True


def start_sitl_process() -> subprocess.Popen | None:
    """Start the SITL binary as a subprocess."""
    global _sitl_process
    if not SITL_PATH.exists():
        return None

    print(f"[Main] Starting SITL process: {SITL_PATH}")
    _sitl_process = subprocess.Popen(
        [str(SITL_PATH)],
        cwd=str(SITL_PATH.parent),
    )

    # Register cleanup handler
    def cleanup():
        if _sitl_process and _sitl_process.poll() is None:
            print("[Main] Stopping SITL process...")
            _sitl_process.terminate()
            _sitl_process.wait(timeout=2)

    atexit.register(cleanup)
    return _sitl_process


def sitl_post_step(tick: int, ctx: el.StepContext):
    """
    Post-step callback for SITL integration.

    This:
    1. Reads keyboard input
    2. Reads sensor data from simulation
    3. Sends sensor packet to C SITL via UDP
    4. Receives motor commands from C SITL
    5. Writes motor commands to simulation
    """
    global _keyboard_controller, _sitl_bridge

    config = CrazyflieConfig.get_global()
    sim_time = tick * config.dt

    # Initialize on first tick
    if not _init_done[0]:
        _init_done[0] = True

        # Initialize bridge FIRST (binds to motor port 9002)
        _sitl_bridge = SITLBridge(timeout_ms=100)
        _sitl_bridge.start()

        # Now start SITL process (binds to sensor port 9003)
        if _sitl_process is None:
            start_sitl_process()

        # Wait for SITL to be ready
        print("[Main] Waiting for SITL process to start...")
        time.sleep(0.5)

        # Initialize keyboard
        if KEYBOARD_AVAILABLE:
            _keyboard_controller = KeyboardController()
            _keyboard_controller.start()
        else:

            class DummyController:
                def get_state(self):
                    from keyboard_controller import ControllerState

                    return ControllerState(is_armed=False, button_blue=False)

            _keyboard_controller = DummyController()

        print("\n" + "=" * 60)
        print("  SIMULATION STARTED!")
        print("=" * 60)
        print("  Keyboard Controls:")
        print("    Q           - Toggle armed (currently DISARMED)")
        print("    Left Shift  - Blue button (hold to enable motors)")
        print("    E/R/T       - Yellow/Green/Red buttons")
        print("=" * 60 + "\n")

    # Read keyboard state
    kb_state = _keyboard_controller.get_state()

    # Write control inputs for graphing
    ctx.write_component("crazyflie.is_armed_control", np.array([1.0 if kb_state.is_armed else 0.0]))
    ctx.write_component("crazyflie.button_blue", np.array([1.0 if kb_state.button_blue else 0.0]))
    ctx.write_component(
        "crazyflie.button_yellow", np.array([1.0 if kb_state.button_yellow else 0.0])
    )
    ctx.write_component("crazyflie.button_green", np.array([1.0 if kb_state.button_green else 0.0]))
    ctx.write_component("crazyflie.button_red", np.array([1.0 if kb_state.button_red else 0.0]))

    # Read sensor data
    try:
        gyro = np.array(ctx.read_component("crazyflie.gyro"), dtype=np.float32)
        accel = np.array(ctx.read_component("crazyflie.accel"), dtype=np.float32)
    except Exception:
        gyro = np.zeros(3, dtype=np.float32)
        accel = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Build sensor packet
    sensor_pkt = SensorPacket(
        timestamp=sim_time,
        gyro=gyro,
        accel=accel,
        is_armed=kb_state.is_armed,
        button_blue=kb_state.button_blue,
        button_yellow=kb_state.button_yellow,
        button_green=kb_state.button_green,
        button_red=kb_state.button_red,
    )

    # Send to SITL and get motor commands (PWM range 0-65535)
    motor_pwm = _sitl_bridge.step(sensor_pkt)

    # Write motor commands to physics (no conversion needed - all 0-65535)
    motor_pwm_f64 = motor_pwm.astype(np.float64)
    ctx.write_component("crazyflie.motor_pwm", motor_pwm_f64)
    ctx.write_component("crazyflie.motor_command", motor_pwm_f64)

    # Periodic status
    if sim_time - _last_print_time[0] >= 1.0:
        armed_str = "ARMED" if kb_state.is_armed else "DISARMED"
        try:
            thrust = np.array(ctx.read_component("crazyflie.thrust"))
            print(
                f"[{sim_time:6.1f}s] {armed_str} | Blue:{int(kb_state.button_blue)} | "
                f"PWM:[{motor_pwm[0]},{motor_pwm[1]},{motor_pwm[2]},{motor_pwm[3]}] | "
                f"Thrust:[{thrust[0]:.4f},{thrust[1]:.4f},{thrust[2]:.4f},{thrust[3]:.4f}]"
            )
        except Exception as e:
            print(f"[{sim_time:6.1f}s] {armed_str} | Read error: {e}")
        _last_print_time[0] = sim_time


# =============================================================================
# Post-Step Callback (HITL Integration)
# =============================================================================

# HITL-specific global state
_hitl_bridge: HITLBridge | None = None
_hitl_init_done = [False]
_hitl_last_print_time = [0.0]


def hitl_post_step(tick: int, ctx: el.StepContext):
    """
    Post-step callback for HITL mode.

    In HITL mode:
    - Control code runs ON the Crazyflie hardware
    - Keyboard inputs are sent as Crazyflie parameters
    - Real sensor data is streamed back for visualization
    - Physics simulation is disabled (real world is source of truth)
    """
    global _keyboard_controller, _hitl_bridge

    config = CrazyflieConfig.get_global()
    sim_time = tick * config.dt

    # Initialize on first tick
    if not _hitl_init_done[0]:
        _hitl_init_done[0] = True

        if not CFLIB_AVAILABLE:
            print("\n" + "=" * 60)
            print("  ERROR: cflib not installed!")
            print("=" * 60)
            print("  HITL mode requires cflib for Crazyradio communication.")
            print("  Install with: uv pip install cflib")
            print("=" * 60 + "\n")
            return

        # Connect to Crazyflie
        _hitl_bridge = HITLBridge(uri=DEFAULT_CF_URI)
        if not _hitl_bridge.connect():
            print("\n[HITL] Failed to connect to Crazyflie!")
            print("Make sure:")
            print("  1. Crazyradio is plugged in")
            print("  2. Crazyflie is powered on")
            print(f"  3. URI is correct: {DEFAULT_CF_URI}")
            return

        # Initialize keyboard
        if KEYBOARD_AVAILABLE:
            _keyboard_controller = KeyboardController()
            _keyboard_controller.start()
        else:

            class DummyController:
                def get_state(self):
                    from keyboard_controller import ControllerState

                    return ControllerState(is_armed=False, button_blue=False)

            _keyboard_controller = DummyController()

        print("\n" + "=" * 60)
        print("  HITL MODE - Connected to Real Crazyflie!")
        print("=" * 60)
        print("  Control code is running ON the hardware.")
        print("  Keyboard controls same as simulation:")
        print("    Q           - Toggle armed")
        print("    Left Shift  - Blue button (hold to enable motors)")
        print("    E/R/T       - Yellow/Green/Red buttons")
        print("=" * 60 + "\n")

    if not _hitl_bridge or not _hitl_bridge.connected:
        return

    # Read keyboard state
    kb_state = _keyboard_controller.get_state()

    # Send control inputs to Crazyflie as parameters
    _hitl_bridge.set_control_params(
        is_armed=kb_state.is_armed,
        button_blue=kb_state.button_blue,
        button_yellow=kb_state.button_yellow,
        button_green=kb_state.button_green,
        button_red=kb_state.button_red,
    )

    # Write control inputs for visualization graphs
    ctx.write_component("crazyflie.is_armed_control", np.array([1.0 if kb_state.is_armed else 0.0]))
    ctx.write_component("crazyflie.button_blue", np.array([1.0 if kb_state.button_blue else 0.0]))
    ctx.write_component(
        "crazyflie.button_yellow", np.array([1.0 if kb_state.button_yellow else 0.0])
    )
    ctx.write_component("crazyflie.button_green", np.array([1.0 if kb_state.button_green else 0.0]))
    ctx.write_component("crazyflie.button_red", np.array([1.0 if kb_state.button_red else 0.0]))

    # Read real sensor data from Crazyflie
    gyro, accel = _hitl_bridge.get_sensors()

    # Write sensor data to visualization (sensor systems not running in HITL mode)
    ctx.write_component("crazyflie.gyro", gyro)
    ctx.write_component("crazyflie.accel", accel)

    # Periodic status
    if sim_time - _hitl_last_print_time[0] >= 1.0:
        armed_str = "ARMED" if kb_state.is_armed else "DISARMED"
        print(
            f"[HITL {sim_time:6.1f}s] {armed_str} | Blue:{int(kb_state.button_blue)} | "
            f"Gyro:[{gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f}] | "
            f"Accel:[{accel[0]:+.2f},{accel[1]:+.2f},{accel[2]:+.2f}]"
        )
        _hitl_last_print_time[0] = sim_time


# =============================================================================
# Main Entry Point
# =============================================================================

# Create config
config = create_default_config()

# Create world
world, drone_id = create_world()

if HITL_MODE:
    # HITL Mode: Real hardware, no physics simulation
    print("\n" + "=" * 60)
    print("  HITL MODE")
    print("=" * 60)

    if not CFLIB_AVAILABLE:
        print("  ERROR: cflib not installed!")
        print("  Install with: uv pip install cflib")
        print("=" * 60 + "\n")
    else:
        print(f"  Crazyflie URI: {DEFAULT_CF_URI}")
        print("  Physics: DISABLED (real world is source of truth)")
        print(f"  Simulation: {config.simulation_time}s at {1 / config.dt:.0f}Hz")
        print("=" * 60 + "\n")

    # Create system without physics
    sys = system(include_physics=False)

    # Run with HITL post_step
    world.run(
        sys,
        sim_time_step=config.dt,
        run_time_step=1.0 / 60.0,  # 60 FPS for visualization
        max_ticks=config.total_sim_ticks,
        post_step=hitl_post_step,
    )

else:
    # SITL Mode: Physics simulation with C SITL binary
    if not check_sitl_binary():
        print("\n[WARNING] SITL binary not found - simulation will not work correctly")
        print("Build it with: ./examples/crazyflie-edu/sitl/build.sh\n")

    print(f"[Main] SITL binary: {SITL_PATH}")
    print(f"[Main] Simulation: {config.simulation_time}s at {1 / config.dt:.0f}Hz")

    # Create system with physics
    sys = system(include_physics=True)

    # Run with SITL post_step
    world.run(
        sys,
        sim_time_step=config.dt,
        run_time_step=1.0 / 60.0,  # 60 FPS for visualization
        max_ticks=config.total_sim_ticks,
        post_step=sitl_post_step,
    )
