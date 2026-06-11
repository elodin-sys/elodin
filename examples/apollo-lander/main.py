#!/usr/bin/env uv run

from __future__ import annotations

import math
import os
import socket
import struct
from pathlib import Path

import elodin as el
import numpy as np

from sim import (
    DEFAULT_MAX_TICKS,
    DPS_FTP_THROTTLE,
    DPS_MAX_THRUST_N,
    GUIDANCE_RATE_HZ,
    LUNAR_GRAVITY,
    PARAMS,
    REFERENCE,
    SIM_TIME_STEP,
    SIMULATION_RATE_HZ,
    SOFT_HORIZONTAL_SPEED_MPS,
    SOFT_VERTICAL_SPEED_MPS,
    START_TIMESTAMP_US,
    TELEMETRY_RATE_HZ,
    UPRIGHT_DOT_MIN,
    build,
    truth_altitude,
    truth_pitch,
)

STATE_PORT_ENV = "ELODIN_MONTE_CARLO_STATE_PORT"
COMMAND_PORT_ENV = "ELODIN_MONTE_CARLO_COMMAND_PORT"
DEFAULT_STATE_PORT = 9013
DEFAULT_COMMAND_PORT = 9012
MAX_TICKS_ENV = "ELODIN_APOLLO_MAX_TICKS"


class SitlBridge:
    def __init__(self, throttle: float, attitude: list[float]) -> None:
        self.state_port = int(os.environ.get(STATE_PORT_ENV, str(DEFAULT_STATE_PORT)))
        self.command_port = int(os.environ.get(COMMAND_PORT_ENV, str(DEFAULT_COMMAND_PORT)))
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_sock.bind(("127.0.0.1", self.command_port))
        self.command_sock.settimeout(0.25)
        self.last_throttle = throttle
        self.last_attitude = list(attitude)
        self.last_rate_setpoint = 0.0

    def step(self, state: dict[str, float | list[float]]) -> tuple[float, list[float], float]:
        world_vel = [float(x) for x in state["world_vel"]]
        world_pos = [float(x) for x in state["world_pos"]]
        payload = struct.pack(
            "<20d",
            float(state["time_s"]),
            float(state["altitude"]),
            float(state["vertical_speed"]),
            world_vel[0],
            world_vel[1],
            world_vel[2],
            float(state["mass"]),
            float(state["ref_alt"]),
            float(state["ref_rate"]),
            float(state["gravity"]),
            float(state["max_thrust"]),
            float(state["thrust_scale"]),
            float(state["track_gain"]),
            float(state["vertical_gain"]),
            float(state["horizontal_gain"]),
            world_pos[0],
            world_pos[1],
            float(state["ref_downrange"]),
            float(state["ref_hspeed"]),
            float(state["ref_hdecel"]),
        )
        self.state_sock.sendto(payload, ("127.0.0.1", self.state_port))
        try:
            raw, _ = self.command_sock.recvfrom(48)
            values = struct.unpack("<6d", raw[:48])
            self.last_throttle = values[0]
            self.last_attitude = [values[1], values[2], values[3], values[4]]
            self.last_rate_setpoint = values[5]
        except (TimeoutError, socket.timeout):
            pass
        return self.last_throttle, self.last_attitude, self.last_rate_setpoint


params = el.monte_carlo.params(PARAMS)
world, system = build(params)
bridge: SitlBridge | None = None
altitude_error_sum = 0.0
pitch_error_sum = 0.0
error_samples = 0
result_emitted = False

initial_altitude = float(params.get("init_altitude_m", truth_altitude(0.0)))
dry_mass = float(params.get("dry_mass_kg", 6_853.0))
thrust_scale = float(params.get("thrust_scale", 1.0))
gravity = LUNAR_GRAVITY * float(params.get("gravity_scale", 1.0))
track_gain = float(params.get("track_gain", 0.06))
vertical_gain = float(params.get("vertical_gain", 0.45))
horizontal_gain = float(params.get("horizontal_gain", 0.05))
init_pitch_deg = float(params.get("init_pitch_deg", -77.0))
max_ticks = int(os.environ.get(MAX_TICKS_ENV, str(DEFAULT_MAX_TICKS)))
guidance_period_ticks = max(1, round(SIMULATION_RATE_HZ / GUIDANCE_RATE_HZ))

# The window opens mid-braking-burn: DPS at the fixed throttle point, vehicle
# pitched back retrograde. Seed the command state to match so the first
# guidance exchanges do not command an attitude transient.
last_throttle = DPS_FTP_THROTTLE
_half_pitch = math.radians(init_pitch_deg) * 0.5
last_attitude = [0.0, math.sin(_half_pitch), 0.0, math.cos(_half_pitch)]
last_rate_setpoint = 0.0

controller_dir = Path(__file__).parent / "controller"
if os.environ.get("ELODIN_MONTE_CARLO_PLANNING") == "1":
    controller_binary = controller_dir / "target" / "release" / "apollo-lander-controller"
    if not controller_binary.exists():
        raise RuntimeError(
            "Apollo Rust controller binary is missing. Run the campaign [build] step "
            "(`cargo build --release --manifest-path examples/apollo-lander/controller/Cargo.toml`) "
            "before generating the Monte Carlo plan."
        )
    controller = el.s10.PyRecipe.process(name="Apollo LGC", cmd=str(controller_binary))
else:
    controller = el.s10.PyRecipe.cargo(name="Apollo LGC", path=str(controller_dir))
world.recipe(controller)


def _normalize_quat(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in q))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm for x in q]


def _slew_quat(current: list[float], target: list[float], max_deg: float = 3.0) -> list[float]:
    current = _normalize_quat(current)
    target = _normalize_quat(target)
    dot = sum(a * b for a, b in zip(current, target))
    if dot < 0.0:
        target = [-x for x in target]
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    angle = 2.0 * math.acos(dot)
    max_angle = math.radians(max_deg)
    if angle <= max_angle or angle < 1e-9:
        return target
    frac = max_angle / angle
    blended = [(1.0 - frac) * a + frac * b for a, b in zip(current, target)]
    return _normalize_quat(blended)


def post_step(tick: int, ctx: el.StepContext) -> None:
    global altitude_error_sum, bridge, error_samples, last_attitude, last_rate_setpoint
    global last_throttle, pitch_error_sum, result_emitted

    reads = ctx.component_batch_operation(
        reads=[
            "lander.world_pos",
            "lander.world_vel",
            "lander.altitude",
            "lander.vertical_speed",
            "lander.horizontal_speed",
            "lander.pitch",
            "lander.propellant",
            "lander.rcs_propellant",
            "lander.landed",
            "lander.touchdown_speed",
            "lander.touchdown_horizontal_speed",
        ]
    )
    world_pos = np.asarray(reads["lander.world_pos"], dtype=np.float64)
    world_vel = np.asarray(reads["lander.world_vel"], dtype=np.float64)
    altitude = float(reads["lander.altitude"][0])
    vertical_speed = float(reads["lander.vertical_speed"][0])
    horizontal_speed = float(reads["lander.horizontal_speed"][0])
    pitch = float(reads["lander.pitch"][0])
    propellant = float(reads["lander.propellant"][0])
    rcs_propellant = float(reads["lander.rcs_propellant"][0])
    landed = float(reads["lander.landed"][0]) > 0.5
    touchdown_speed = float(reads["lander.touchdown_speed"][0])
    touchdown_horizontal_speed = float(reads["lander.touchdown_horizontal_speed"][0])

    t_s = tick * SIM_TIME_STEP
    real_altitude = truth_altitude(t_s)
    truth_pitch_now = truth_pitch(t_s)
    altitude_error_sum += (altitude - real_altitude) ** 2
    pitch_error_sum += (pitch - truth_pitch_now) ** 2
    error_samples += 1

    if tick % guidance_period_ticks == 0 and not landed:
        state = {
            "tick": float(tick),
            "time_s": t_s,
            "altitude": altitude,
            "vertical_speed": vertical_speed,
            "world_pos": [float(x) for x in world_pos[4:7]],
            "world_vel": [float(x) for x in world_vel[3:6]],
            "mass": dry_mass + propellant + rcs_propellant,
            "propellant": propellant,
            "dry_mass": dry_mass,
            "thrust_scale": thrust_scale,
            "gravity": gravity,
            "max_thrust": DPS_MAX_THRUST_N,
            "ref_alt": REFERENCE.altitude(t_s),
            "ref_rate": REFERENCE.descent_rate(t_s),
            "ref_downrange": REFERENCE.downrange(t_s),
            "ref_hspeed": REFERENCE.horizontal_speed(t_s),
            # Feed-forward braking deceleration from the reconstructed profile.
            "ref_hdecel": REFERENCE.horizontal_speed(t_s) - REFERENCE.horizontal_speed(t_s + 1.0),
            "initial_altitude": initial_altitude,
            "track_gain": track_gain,
            "vertical_gain": vertical_gain,
            "horizontal_gain": horizontal_gain,
        }
        if bridge is None:
            bridge = SitlBridge(last_throttle, last_attitude)
        last_throttle, target_attitude, last_rate_setpoint = bridge.step(state)
        last_attitude = _slew_quat(last_attitude, target_attitude)

    ctx.component_batch_operation(
        writes={
            "lander.throttle_cmd": np.array([last_throttle], dtype=np.float64),
            "lander.attitude_setpoint": np.array(last_attitude, dtype=np.float64),
        }
    )

    if not result_emitted and (landed or tick >= max_ticks - 1):
        if not landed:
            touchdown_speed = abs(vertical_speed)
            touchdown_horizontal_speed = horizontal_speed
        traj_rmse = (altitude_error_sum / max(error_samples, 1)) ** 0.5
        pitch_rmse = (pitch_error_sum / max(error_samples, 1)) ** 0.5
        upright_dot = math.cos(math.radians(abs(pitch)))
        # Horizontal distance from the targeted landing site (world origin) at touchdown.
        downrange_miss = math.hypot(float(world_pos[4]), float(world_pos[5]))
        soft_landing = (
            landed
            and touchdown_speed <= SOFT_VERTICAL_SPEED_MPS
            and touchdown_horizontal_speed <= SOFT_HORIZONTAL_SPEED_MPS
            and upright_dot >= UPRIGHT_DOT_MIN
            and propellant > 0.0
        )
        if os.environ.get("ELODIN_MONTE_CARLO_CONTEXT"):
            el.monte_carlo.result(
                touchdown_speed=touchdown_speed,
                horizontal_speed=touchdown_horizontal_speed,
                fuel_remaining=propellant,
                rcs_fuel_remaining=rcs_propellant,
                traj_rmse=traj_rmse,
                pitch_rmse=pitch_rmse,
                downrange_miss=downrange_miss,
                upright_dot=upright_dot,
                landed=landed,
                soft_landing=soft_landing,
            )
        result_emitted = True


world.run(
    system,
    simulation_rate=SIMULATION_RATE_HZ,
    telemetry_rate=TELEMETRY_RATE_HZ,
    default_playback_speed=30.0,
    max_ticks=max_ticks,
    db_path=params.db_path or os.environ.get("ELODIN_DB_PATH"),
    post_step=post_step,
    interactive=False,
    start_timestamp=START_TIMESTAMP_US,
    log_level="warn",
)
