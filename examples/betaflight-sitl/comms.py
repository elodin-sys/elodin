"""
Betaflight SITL UDP Communication Bridge

This module provides UDP packet structures and communication handling
for interfacing Elodin's physics simulation with Betaflight SITL.

Packet Protocol:
    - PORT 9003: FDM packets (simulator -> Betaflight) - sensor data
    - PORT 9004: RC packets (simulator -> Betaflight) - RC channels
    - PORT 9002: Servo packets (Betaflight -> simulator) - normalized motor
    - PORT 9001: Servo raw packets (Betaflight -> simulator) - raw PWM

Coordinate Systems:
    - Elodin uses ENU (East-North-Up) with X=forward, Y=left, Z=up
    - Betaflight SITL expects:
        - Acceleration: NED body frame
        - Gyro: NED body frame (rad/s)
        - Position/Velocity: ENU for GPS mode
        - Quaternion: w, x, y, z

Unit Conversions:
    - Accelerometer: 1G = 256 LSB (Betaflight), 1G = 9.80665 m/s²
    - Gyroscope: 1 deg/s = 16.4 LSB (Betaflight), input in rad/s
"""

import struct
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


# Port definitions matching Betaflight SITL (sitl.c)
PORT_PWM_RAW = 9001  # Betaflight -> Simulator (raw PWM)
PORT_PWM = 9002       # Betaflight -> Simulator (normalized)
PORT_STATE = 9003     # Simulator -> Betaflight (FDM/sensor data)
PORT_RC = 9004        # Simulator -> Betaflight (RC channels)

# Default host
DEFAULT_HOST = "127.0.0.1"

# Conversion constants (from sitl.c)
ACC_SCALE = 256.0 / 9.80665  # Convert m/s² to Betaflight LSB
GYRO_SCALE = 16.4            # Convert deg/s to Betaflight LSB
RAD_TO_DEG = 180.0 / np.pi

# Packet sizes
MAX_RC_CHANNELS = 16
MAX_PWM_CHANNELS = 16


@dataclass
class FDMPacket:
    """
    Flight Dynamics Model packet sent to Betaflight SITL.
    
    Contains simulated sensor data from Elodin physics engine.
    Total size: 216 bytes (all doubles + padding)
    """
    timestamp: float = 0.0                        # seconds
    imu_angular_velocity_rpy: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # rad/s, body frame
    imu_linear_acceleration_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # m/s², NED body frame
    imu_orientation_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )  # w, x, y, z
    velocity_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # m/s, ENU earth frame
    position_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # meters, ENU (lon, lat, alt for GPS)
    pressure: float = 101325.0                    # Pa (sea level default)
    
    # Packet format: 18 doubles = 144 bytes
    # timestamp(1) + gyro(3) + accel(3) + quat(4) + vel(3) + pos(3) + pressure(1) = 18
    _FORMAT = "<18d"
    SIZE = struct.calcsize(_FORMAT)  # 144 bytes
    
    def pack(self) -> bytes:
        """Pack the FDM packet into bytes for UDP transmission."""
        return struct.pack(
            self._FORMAT,
            self.timestamp,
            self.imu_angular_velocity_rpy[0],
            self.imu_angular_velocity_rpy[1],
            self.imu_angular_velocity_rpy[2],
            self.imu_linear_acceleration_xyz[0],
            self.imu_linear_acceleration_xyz[1],
            self.imu_linear_acceleration_xyz[2],
            self.imu_orientation_quat[0],  # w
            self.imu_orientation_quat[1],  # x
            self.imu_orientation_quat[2],  # y
            self.imu_orientation_quat[3],  # z
            self.velocity_xyz[0],
            self.velocity_xyz[1],
            self.velocity_xyz[2],
            self.position_xyz[0],
            self.position_xyz[1],
            self.position_xyz[2],
            self.pressure,
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "FDMPacket":
        """Unpack FDM packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[:cls.SIZE])
        return cls(
            timestamp=values[0],
            imu_angular_velocity_rpy=np.array(values[1:4]),
            imu_linear_acceleration_xyz=np.array(values[4:7]),
            imu_orientation_quat=np.array(values[7:11]),
            velocity_xyz=np.array(values[11:14]),
            position_xyz=np.array(values[14:17]),
            pressure=values[17],
        )


@dataclass
class RCPacket:
    """
    RC (Remote Control) packet sent to Betaflight SITL.
    
    Contains RC channel values (PWM microseconds, typically 1000-2000).
    Standard channel mapping:
        0: Roll, 1: Pitch, 2: Throttle, 3: Yaw
        4-15: Aux channels
    """
    timestamp: float = 0.0
    channels: np.ndarray = field(
        default_factory=lambda: np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
    )
    
    # Format: 1 double + 16 uint16 = 8 + 32 = 40 bytes
    _FORMAT = f"<d{MAX_RC_CHANNELS}H"
    SIZE = struct.calcsize(_FORMAT)
    
    def pack(self) -> bytes:
        """Pack the RC packet into bytes for UDP transmission."""
        return struct.pack(
            self._FORMAT,
            self.timestamp,
            *self.channels[:MAX_RC_CHANNELS]
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "RCPacket":
        """Unpack RC packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[:cls.SIZE])
        return cls(
            timestamp=values[0],
            channels=np.array(values[1:], dtype=np.uint16),
        )


@dataclass
class ServoPacket:
    """
    Servo/Motor output packet received from Betaflight SITL.
    
    Contains normalized motor speeds for quadcopter.
    Values are normalized: [0.0, 1.0] for normal, [-1.0, 1.0] for 3D mode.
    """
    motor_speed: np.ndarray = field(
        default_factory=lambda: np.zeros(4)
    )
    
    # Format: 4 floats = 16 bytes
    _FORMAT = "<4f"
    SIZE = struct.calcsize(_FORMAT)
    
    def pack(self) -> bytes:
        """Pack servo packet into bytes."""
        return struct.pack(self._FORMAT, *self.motor_speed[:4])
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ServoPacket":
        """Unpack servo packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[:cls.SIZE])
        return cls(motor_speed=np.array(values))


@dataclass
class ServoPacketRaw:
    """
    Raw servo/motor output packet received from Betaflight SITL.
    
    Contains raw PWM values (typically 1000-2000 microseconds).
    Supports up to 16 PWM channels.
    """
    motor_count: int = 4
    pwm_output: np.ndarray = field(
        default_factory=lambda: np.full(MAX_PWM_CHANNELS, 1000.0)
    )
    
    # Format: 1 uint16 + 2 bytes padding + 16 floats = 68 bytes
    # C struct has uint16_t motorCount (2 bytes) followed by padding for
    # 4-byte float alignment, then float[16] array
    _FORMAT = f"<Hxx{MAX_PWM_CHANNELS}f"  # xx = 2 padding bytes
    SIZE = struct.calcsize(_FORMAT)  # 68 bytes
    
    def pack(self) -> bytes:
        """Pack raw servo packet into bytes."""
        # Note: struct.pack with 'xx' padding bytes doesn't require values for them
        return struct.pack(
            self._FORMAT,
            self.motor_count,
            *self.pwm_output[:MAX_PWM_CHANNELS]
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ServoPacketRaw":
        """Unpack raw servo packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[:cls.SIZE])
        return cls(
            motor_count=values[0],
            pwm_output=np.array(values[1:]),
        )


class BetaflightBridge:
    """
    UDP communication bridge between Elodin and Betaflight SITL.
    
    Handles:
    - Sending FDM (sensor) packets to Betaflight
    - Sending RC packets to Betaflight
    - Receiving motor/servo packets from Betaflight
    - Coordinate frame conversions (ENU <-> NED)
    
    Usage:
        bridge = BetaflightBridge()
        bridge.start()
        
        # In simulation loop:
        bridge.send_fdm(fdm_packet)
        motors = bridge.get_motors()  # Returns normalized [0, 1] values
        
        bridge.stop()
    """
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        state_port: int = PORT_STATE,
        rc_port: int = PORT_RC,
        pwm_port: int = PORT_PWM,
        pwm_raw_port: int = PORT_PWM_RAW,
    ):
        """
        Initialize the Betaflight bridge.
        
        Args:
            host: IP address of Betaflight SITL (default localhost)
            state_port: Port for FDM packets (default 9003)
            rc_port: Port for RC packets (default 9004)
            pwm_port: Port to receive normalized motor outputs (default 9002)
            pwm_raw_port: Port to receive raw PWM outputs (default 9001)
        """
        self.host = host
        self.state_port = state_port
        self.rc_port = rc_port
        self.pwm_port = pwm_port
        self.pwm_raw_port = pwm_raw_port
        
        # Sockets
        self._state_socket: Optional[socket.socket] = None
        self._rc_socket: Optional[socket.socket] = None
        self._pwm_socket: Optional[socket.socket] = None
        self._pwm_raw_socket: Optional[socket.socket] = None
        
        # State
        self._running = False
        self._motor_lock = threading.Lock()
        self._motors = np.zeros(4)  # Normalized motor values [0, 1]
        self._motors_raw = np.zeros(MAX_PWM_CHANNELS)  # Raw PWM values
        self._last_motor_time = 0.0
        
        # Receiver threads
        self._pwm_thread: Optional[threading.Thread] = None
        self._pwm_raw_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._motor_callback: Optional[Callable[[np.ndarray], None]] = None
    
    def start(self) -> None:
        """Start the bridge and begin listening for motor packets."""
        if self._running:
            return
        
        # Create UDP sockets for sending
        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Create and bind receiving sockets
        self._pwm_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._pwm_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pwm_socket.bind(("0.0.0.0", self.pwm_port))
        self._pwm_socket.settimeout(0.1)
        
        self._pwm_raw_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._pwm_raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pwm_raw_socket.bind(("0.0.0.0", self.pwm_raw_port))
        self._pwm_raw_socket.settimeout(0.1)
        
        self._running = True
        
        # Start receiver threads
        self._pwm_thread = threading.Thread(target=self._pwm_receiver, daemon=True)
        self._pwm_thread.start()
        
        self._pwm_raw_thread = threading.Thread(target=self._pwm_raw_receiver, daemon=True)
        self._pwm_raw_thread.start()
        
        print(f"[BetaflightBridge] Started - sending to {self.host}")
        print(f"  FDM -> port {self.state_port}")
        print(f"  RC  -> port {self.rc_port}")
        print(f"  PWM <- port {self.pwm_port}")
        print(f"  PWM_RAW <- port {self.pwm_raw_port}")
    
    def stop(self) -> None:
        """Stop the bridge and close all sockets."""
        self._running = False
        
        # Wait for threads to finish
        if self._pwm_thread:
            self._pwm_thread.join(timeout=1.0)
        if self._pwm_raw_thread:
            self._pwm_raw_thread.join(timeout=1.0)
        
        # Close sockets
        for sock in [self._state_socket, self._rc_socket, 
                     self._pwm_socket, self._pwm_raw_socket]:
            if sock:
                sock.close()
        
        self._state_socket = None
        self._rc_socket = None
        self._pwm_socket = None
        self._pwm_raw_socket = None
        
        print("[BetaflightBridge] Stopped")
    
    def _pwm_receiver(self) -> None:
        """Background thread to receive normalized motor packets."""
        while self._running:
            try:
                data, addr = self._pwm_socket.recvfrom(ServoPacket.SIZE)
                packet = ServoPacket.from_bytes(data)
                
                with self._motor_lock:
                    self._motors = packet.motor_speed.copy()
                    self._last_motor_time = time.time()
                
                if self._motor_callback:
                    self._motor_callback(packet.motor_speed)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[BetaflightBridge] PWM receive error: {e}")
    
    def _pwm_raw_receiver(self) -> None:
        """Background thread to receive raw PWM packets."""
        while self._running:
            try:
                data, addr = self._pwm_raw_socket.recvfrom(ServoPacketRaw.SIZE)
                packet = ServoPacketRaw.from_bytes(data)
                
                with self._motor_lock:
                    self._motors_raw = packet.pwm_output.copy()
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[BetaflightBridge] PWM_RAW receive error: {e}")
    
    def send_fdm(self, packet: FDMPacket) -> None:
        """
        Send FDM (sensor data) packet to Betaflight SITL.
        
        Args:
            packet: FDMPacket with sensor data
        """
        if self._state_socket:
            data = packet.pack()
            self._state_socket.sendto(data, (self.host, self.state_port))
    
    def send_rc(self, packet: RCPacket) -> None:
        """
        Send RC (remote control) packet to Betaflight SITL.
        
        Args:
            packet: RCPacket with RC channel values
        """
        if self._rc_socket:
            data = packet.pack()
            self._rc_socket.sendto(data, (self.host, self.rc_port))
    
    def send_state(
        self,
        timestamp: float,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        orientation_quat: np.ndarray,
        velocity: np.ndarray,
        position: np.ndarray,
        pressure: float = 101325.0,
    ) -> None:
        """
        Send state update to Betaflight SITL.
        
        This is a convenience method that handles coordinate frame conversion
        from Elodin's ENU frame to Betaflight's expected format.
        
        Args:
            timestamp: Simulation time in seconds
            angular_velocity: Body-frame angular velocity [roll, pitch, yaw] rad/s (ENU)
            linear_acceleration: Body-frame acceleration [x, y, z] m/s² (ENU)
            orientation_quat: Quaternion [w, x, y, z] (Elodin convention)
            velocity: World-frame velocity [east, north, up] m/s
            position: World-frame position [east, north, up] meters
            pressure: Barometric pressure in Pascals
        """
        # Convert from Elodin ENU body frame to Betaflight NED body frame
        # ENU to NED: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu
        accel_ned = np.array([
            linear_acceleration[1],   # ENU-Y -> NED-X
            linear_acceleration[0],   # ENU-X -> NED-Y
            -linear_acceleration[2],  # ENU-Z -> NED-Z (inverted)
        ])
        
        # Gyro conversion: same transform
        gyro_ned = np.array([
            angular_velocity[1],   # pitch rate
            angular_velocity[0],   # roll rate
            -angular_velocity[2],  # yaw rate (inverted)
        ])
        
        # Quaternion needs to be adjusted for frame change
        # For now, pass through as Betaflight handles quaternion internally
        quat = orientation_quat
        
        # Velocity and position are in world frame ENU, which Betaflight expects
        # for virtual GPS mode
        
        packet = FDMPacket(
            timestamp=timestamp,
            imu_angular_velocity_rpy=gyro_ned,
            imu_linear_acceleration_xyz=accel_ned,
            imu_orientation_quat=quat,
            velocity_xyz=velocity,
            position_xyz=position,
            pressure=pressure,
        )
        
        self.send_fdm(packet)
    
    def send_rc_channels(
        self,
        throttle: int = 1000,
        roll: int = 1500,
        pitch: int = 1500,
        yaw: int = 1500,
        aux: Optional[list] = None,
        timestamp: float = 0.0,
    ) -> None:
        """
        Send RC channel values to Betaflight SITL.
        
        Args:
            throttle: Throttle PWM (1000-2000, idle=1000)
            roll: Roll PWM (1000-2000, center=1500)
            pitch: Pitch PWM (1000-2000, center=1500)
            yaw: Yaw PWM (1000-2000, center=1500)
            aux: List of auxiliary channel values
            timestamp: Simulation time
        """
        channels = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
        channels[0] = roll
        channels[1] = pitch
        channels[2] = throttle
        channels[3] = yaw
        
        if aux:
            for i, val in enumerate(aux[:MAX_RC_CHANNELS - 4]):
                channels[4 + i] = val
        
        packet = RCPacket(timestamp=timestamp, channels=channels)
        self.send_rc(packet)
    
    def get_motors(self) -> np.ndarray:
        """
        Get the latest motor values from Betaflight.
        
        Returns:
            Array of 4 normalized motor values [0.0, 1.0]
        """
        with self._motor_lock:
            return self._motors.copy()
    
    def get_motors_raw(self) -> np.ndarray:
        """
        Get the latest raw PWM motor values from Betaflight.
        
        Returns:
            Array of raw PWM values (typically 1000-2000)
        """
        with self._motor_lock:
            return self._motors_raw.copy()
    
    def get_motor_age(self) -> float:
        """
        Get the time since the last motor update was received.
        
        Returns:
            Time in seconds since last motor packet
        """
        with self._motor_lock:
            if self._last_motor_time == 0:
                return float('inf')
            return time.time() - self._last_motor_time
    
    def set_motor_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Set a callback to be invoked when motor values are received.
        
        Args:
            callback: Function that takes motor array as argument
        """
        self._motor_callback = callback
    
    @property
    def is_connected(self) -> bool:
        """Check if we've received recent motor updates from Betaflight."""
        return self.get_motor_age() < 1.0  # Consider connected if update within 1s
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class BetaflightSyncBridge:
    """
    Synchronous Betaflight SITL bridge for lockstep simulation.
    
    This bridge is designed for use with Elodin's post_step callback and
    Betaflight's SIMULATOR_GYROPID_SYNC mode. It provides blocking step()
    calls that:
    1. Send FDM + RC packets to Betaflight
    2. Wait for motor response (blocking with timeout)
    3. Return motor values
    
    This enables deterministic, faster-than-realtime simulation where each
    Elodin physics tick is tightly synchronized with one Betaflight PID iteration.
    
    Usage:
        bridge = BetaflightSyncBridge()
        bridge.start()
        
        # In post_step callback:
        motors = bridge.step(fdm_packet, rc_packet)
        
        bridge.stop()
    """
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        state_port: int = PORT_STATE,
        rc_port: int = PORT_RC,
        pwm_port: int = PORT_PWM,
        timeout_ms: int = 100,
    ):
        """
        Initialize the synchronous Betaflight bridge.
        
        Args:
            host: IP address of Betaflight SITL (default localhost)
            state_port: Port for FDM packets (default 9003)
            rc_port: Port for RC packets (default 9004)
            pwm_port: Port to receive normalized motor outputs (default 9002)
            timeout_ms: Timeout for motor response in milliseconds
        """
        self.host = host
        self.state_port = state_port
        self.rc_port = rc_port
        self.pwm_port = pwm_port
        self.timeout_ms = timeout_ms
        
        # Sockets
        self._state_socket: Optional[socket.socket] = None
        self._rc_socket: Optional[socket.socket] = None
        self._pwm_socket: Optional[socket.socket] = None
        
        # State
        self._started = False
        self._step_count = 0
        self._last_motors = np.zeros(4)
    
    def start(self) -> None:
        """Start the bridge and prepare sockets."""
        if self._started:
            return
        
        # Create UDP sockets for sending
        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Create and bind receiving socket with timeout
        self._pwm_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._pwm_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pwm_socket.bind(("0.0.0.0", self.pwm_port))
        self._pwm_socket.settimeout(self.timeout_ms / 1000.0)
        
        # Drain any stale packets from previous runs
        self._pwm_socket.setblocking(False)
        drained = 0
        try:
            while True:
                self._pwm_socket.recv(1024)
                drained += 1
        except BlockingIOError:
            pass  # No more data to drain
        self._pwm_socket.setblocking(True)
        self._pwm_socket.settimeout(self.timeout_ms / 1000.0)
        if drained > 0:
            print(f"[BetaflightSyncBridge] Drained {drained} stale packet(s)")
        
        self._started = True
        self._step_count = 0
        
        print(f"[BetaflightSyncBridge] Started in lockstep mode")
        print(f"  FDM -> {self.host}:{self.state_port}")
        print(f"  RC  -> {self.host}:{self.rc_port}")
        print(f"  PWM <- 0.0.0.0:{self.pwm_port} (timeout={self.timeout_ms}ms)")
    
    def stop(self) -> None:
        """Stop the bridge and close sockets."""
        if not self._started:
            return
        
        # Close sockets
        for sock in [self._state_socket, self._rc_socket, self._pwm_socket]:
            if sock:
                sock.close()
        
        self._state_socket = None
        self._rc_socket = None
        self._pwm_socket = None
        self._started = False
        
        print(f"[BetaflightSyncBridge] Stopped after {self._step_count} steps")
    
    def step(
        self,
        fdm: FDMPacket,
        rc: RCPacket,
        timeout_ms: Optional[int] = None,
    ) -> np.ndarray:
        """
        Perform one synchronized SITL step.
        
        This is the core synchronization method. When Betaflight is built with
        SIMULATOR_GYROPID_SYNC enabled, it blocks its main loop until an FDM
        packet arrives. After processing, it immediately sends motor outputs.
        
        This method:
        1. Sends FDM packet (sensor data)
        2. Sends RC packet (control inputs)
        3. Waits for motor response (blocking with timeout)
        4. Returns normalized motor values [0.0, 1.0]
        
        Args:
            fdm: FDM packet with sensor data
            rc: RC packet with control inputs
            timeout_ms: Override default timeout (optional)
            
        Returns:
            Array of 4 normalized motor values [0.0, 1.0]
            
        Raises:
            TimeoutError: If no motor response within timeout
            RuntimeError: If bridge not started
        """
        if not self._started:
            raise RuntimeError("Bridge not started - call start() first")
        
        # Send FDM packet (this unblocks Betaflight's GYROPID_SYNC)
        fdm_data = fdm.pack()
        self._state_socket.sendto(fdm_data, (self.host, self.state_port))
        
        # Send RC packet
        rc_data = rc.pack()
        self._rc_socket.sendto(rc_data, (self.host, self.rc_port))
        
        # Wait for motor response (blocking)
        timeout = (timeout_ms or self.timeout_ms) / 1000.0
        self._pwm_socket.settimeout(timeout)
        
        try:
            data, addr = self._pwm_socket.recvfrom(ServoPacket.SIZE)
            packet = ServoPacket.from_bytes(data)
            self._last_motors = packet.motor_speed.copy()
            self._step_count += 1
            return self._last_motors
            
        except socket.timeout:
            # Return last known motors on timeout (first few steps may timeout
            # before Betaflight is fully initialized)
            if self._step_count == 0:
                # During initialization, just return zeros
                return np.zeros(4)
            raise TimeoutError(
                f"No motor response from Betaflight within {timeout*1000:.0f}ms "
                f"(step {self._step_count})"
            )
    
    def step_with_state(
        self,
        timestamp: float,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        orientation_quat: np.ndarray,
        velocity: np.ndarray,
        position: np.ndarray,
        pressure: float,
        throttle: int,
        roll: int,
        pitch: int,
        yaw: int,
        aux: Optional[list] = None,
    ) -> np.ndarray:
        """
        Convenience method to step with raw state values.
        
        Handles ENU to NED coordinate conversion internally.
        
        Args:
            timestamp: Simulation time in seconds
            angular_velocity: Body-frame angular velocity [x, y, z] rad/s (ENU)
            linear_acceleration: Body-frame acceleration [x, y, z] m/s² (ENU)
            orientation_quat: Quaternion [w, x, y, z]
            velocity: World-frame velocity [x, y, z] m/s (ENU)
            position: World-frame position [x, y, z] meters (ENU)
            pressure: Barometric pressure in Pascals
            throttle: Throttle PWM (1000-2000)
            roll: Roll PWM (1000-2000, center=1500)
            pitch: Pitch PWM (1000-2000, center=1500)
            yaw: Yaw PWM (1000-2000, center=1500)
            aux: List of auxiliary channel values
            
        Returns:
            Array of 4 normalized motor values [0.0, 1.0]
        """
        # Convert from Elodin ENU body frame to Betaflight NED body frame
        accel_ned = np.array([
            linear_acceleration[1],   # ENU-Y -> NED-X
            linear_acceleration[0],   # ENU-X -> NED-Y
            -linear_acceleration[2],  # ENU-Z -> NED-Z (inverted)
        ])
        
        gyro_ned = np.array([
            angular_velocity[1],   # pitch rate
            angular_velocity[0],   # roll rate
            -angular_velocity[2],  # yaw rate (inverted)
        ])
        
        fdm = FDMPacket(
            timestamp=timestamp,
            imu_angular_velocity_rpy=gyro_ned,
            imu_linear_acceleration_xyz=accel_ned,
            imu_orientation_quat=orientation_quat,
            velocity_xyz=velocity,
            position_xyz=position,
            pressure=pressure,
        )
        
        # Build RC packet
        channels = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
        channels[0] = roll
        channels[1] = pitch
        channels[2] = throttle
        channels[3] = yaw
        if aux:
            for i, val in enumerate(aux[:MAX_RC_CHANNELS - 4]):
                channels[4 + i] = val
        
        rc = RCPacket(timestamp=timestamp, channels=channels)
        
        return self.step(fdm, rc)
    
    @property
    def step_count(self) -> int:
        """Number of successful steps completed."""
        return self._step_count
    
    @property
    def last_motors(self) -> np.ndarray:
        """Last received motor values."""
        return self._last_motors.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Betaflight motor remapping for Quad-X configuration
# Betaflight motor order (looking from above, front is top):
#   1(FR)  2(BR)
#   4(FL)  3(BL)
# Elodin standard order:
#   0(FR)  1(FL)
#   2(BR)  3(BL)
BETAFLIGHT_TO_ELODIN_MOTOR_MAP = [0, 2, 3, 1]  # BF[0]->EL[0], BF[1]->EL[2], etc.
ELODIN_TO_BETAFLIGHT_MOTOR_MAP = [0, 3, 1, 2]  # EL[0]->BF[0], EL[1]->BF[3], etc.


def remap_motors_betaflight_to_elodin(bf_motors: np.ndarray) -> np.ndarray:
    """
    Remap motor indices from Betaflight order to Elodin order.
    
    Betaflight Quad-X (looking from above):
        Motor 1 (FR) = index 0
        Motor 2 (BR) = index 1  
        Motor 3 (BL) = index 2
        Motor 4 (FL) = index 3
    
    Elodin standard (looking from above):
        Motor 0 (FR)
        Motor 1 (FL)
        Motor 2 (BR)
        Motor 3 (BL)
    """
    return bf_motors[BETAFLIGHT_TO_ELODIN_MOTOR_MAP]


def remap_motors_elodin_to_betaflight(el_motors: np.ndarray) -> np.ndarray:
    """Remap motor indices from Elodin order to Betaflight order."""
    return el_motors[ELODIN_TO_BETAFLIGHT_MOTOR_MAP]


if __name__ == "__main__":
    # Simple test - print packet sizes
    print("Betaflight SITL Packet Sizes:")
    print(f"  FDMPacket:       {FDMPacket.SIZE} bytes")
    print(f"  RCPacket:        {RCPacket.SIZE} bytes")
    print(f"  ServoPacket:     {ServoPacket.SIZE} bytes")
    print(f"  ServoPacketRaw:  {ServoPacketRaw.SIZE} bytes")
    
    # Test packing/unpacking
    fdm = FDMPacket(
        timestamp=1.0,
        imu_angular_velocity_rpy=np.array([0.1, 0.2, 0.3]),
        imu_linear_acceleration_xyz=np.array([0.0, 0.0, 9.81]),
        imu_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity_xyz=np.array([0.0, 0.0, 0.0]),
        position_xyz=np.array([0.0, 0.0, 0.0]),
        pressure=101325.0,
    )
    
    packed = fdm.pack()
    unpacked = FDMPacket.from_bytes(packed)
    
    print(f"\nFDM Pack/Unpack test:")
    print(f"  Original timestamp: {fdm.timestamp}")
    print(f"  Unpacked timestamp: {unpacked.timestamp}")
    print(f"  Accel match: {np.allclose(fdm.imu_linear_acceleration_xyz, unpacked.imu_linear_acceleration_xyz)}")
