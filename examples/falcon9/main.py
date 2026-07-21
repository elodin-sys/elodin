#!/usr/bin/env uv run
"""Falcon 9 launch-to-landing SITL harness (WHITEPAPER 2, 11-13).

Builds the mission world, launches the Rust flight software via s10, and runs
the UDP lockstep bridge in post_step at the guidance rate. The FSW sees only
sensor outputs; the bridge writes only actuator/valve commands. Scoring
accumulates in-sim; the final Monte Carlo result is emitted at touchdown.
"""

from __future__ import annotations

import json
import math
import os
import socket
import struct
from pathlib import Path

import elodin as el
import numpy as np
from constants import (
    GUIDANCE_RATE_HZ,
    LOX_LOAD_KG,
    LZ1_ALT_M,
    LZ1_LAT_DEG,
    LZ1_LON_DEG,
    PAYLOAD_KG,
    RP1_LOAD_KG,
    SIM_RATE_HZ,
    STAGE2_WET_KG,
    START_TIMESTAMP_US,
    TOUCHDOWN_SOFT_LATERAL_MPS,
    TOUCHDOWN_SOFT_VERTICAL_MPS,
)
from frames import ecef_to_geodetic, geodetic_to_ecef, ned_basis
from reference import build_reference
from sim import SIM_TIME_STEP, build_mission

STATE_FLOATS = 46
CMD_FLOATS = 27
DEFAULT_STATE_PORT = 9114
DEFAULT_COMMAND_PORT = 9115
MAX_TICKS_ENV = "ELODIN_FALCON9_MAX_TICKS"
MISSION_ENV = "ELODIN_FALCON9_MISSION"

# Defaults are the CALIBRATED best-fit against the recorded CRS-12 flight
# (17-round campaign lineage; see README "Calibration Results"). Ranges are
# the exploration priors spec.toml mirrors.
PARAMS = el.monte_carlo.params_spec(
    # Vehicle (EST priors, WHITEPAPER 13.1).
    lox_kg=el.monte_carlo.Param(
        float, default=275_357.0, min=0.97 * LOX_LOAD_KG, max=1.03 * LOX_LOAD_KG
    ),
    rp1_kg=el.monte_carlo.Param(
        float, default=120_449.0, min=0.97 * RP1_LOAD_KG, max=1.03 * RP1_LOAD_KG
    ),
    thrust_scale=el.monte_carlo.Param(float, default=1.0323, min=0.94, max=1.14),
    isp_scale=el.monte_carlo.Param(float, default=1.0215, min=0.97, max=1.03),
    ca_scale=el.monte_carlo.Param(float, default=0.9574, min=0.6, max=1.5),
    cn_scale=el.monte_carlo.Param(float, default=1.3038, min=0.6, max=1.5),
    display_lag_s=el.monte_carlo.Param(float, default=1.0, min=0.0, max=2.5),
    # Guidance (calibrated against the recorded CRS-12 timeline).
    kick_deg=el.monte_carlo.Param(float, default=6.17, min=2.0, max=8.0),
    kick_start_s=el.monte_carlo.Param(float, default=7.81, min=5.0, max=14.0),
    kick_ramp_s=el.monte_carlo.Param(float, default=11.74, min=4.0, max=14.0),
    ascent_throttle=el.monte_carlo.Param(float, default=0.9969, min=0.9, max=1.0),
    bucket_throttle=el.monte_carlo.Param(float, default=0.7105, min=0.6, max=0.9),
    bucket_q_on_pa=el.monte_carlo.Param(float, default=18_942.0, min=12_000.0, max=25_000.0),
    bucket_q_off_pa=el.monte_carlo.Param(float, default=30_000.0, min=20_000.0, max=40_000.0),
    meco_speed_mps=el.monte_carlo.Param(float, default=1_645.1, min=1_600.0, max=1_700.0),
    azimuth_deg=el.monte_carlo.Param(float, default=47.67, min=35.0, max=55.0),
    # Boostback aim bias along the approach course (positive aims past
    # LZ-1; the descent system lands short of the ballistic aim).
    boostback_overshoot_m=el.monte_carlo.Param(float, default=-407.0, min=-4_000.0, max=6_000.0),
    entry_ignite_speed_mps=el.monte_carlo.Param(float, default=1_297.2, min=1_100.0, max=1_400.0),
    entry_dv_mps=el.monte_carlo.Param(float, default=350.0, min=280.0, max=450.0),
    landing_accel_margin=el.monte_carlo.Param(float, default=1.273, min=1.0, max=1.6),
    meco_fpa_deg=el.monte_carlo.Param(float, default=35.27, min=30.0, max=50.0),
    pitch_exp=el.monte_carlo.Param(float, default=0.5626, min=0.3, max=1.0),
    entry_throttle=el.monte_carlo.Param(float, default=0.5725, min=0.57, max=0.85),
    landing_arm_alt_m=el.monte_carlo.Param(float, default=5_630.0, min=4_000.0, max=10_000.0),
    entry_ignite_alt_m=el.monte_carlo.Param(float, default=49_618.0, min=42_000.0, max=58_000.0),
    fsw_cd_s_m2=el.monte_carlo.Param(float, default=41.44, min=15.0, max=45.0),
    # Full-throttle boostback fit the recorded profile best (a throttled burn
    # lengthens the reversal and degrades every downstream segment).
    boostback_throttle=el.monte_carlo.Param(float, default=1.0, min=0.85, max=1.0),
)

params = el.monte_carlo.params(PARAMS)
mission = os.environ.get(MISSION_ENV, "crs12")
REF = build_reference(mission)

# Frozen-parameter overrides (calibrated best-fit replay, CRS-11 holdout):
# a JSON object of param name -> value, taking precedence over sampling.
_overrides = json.loads(os.environ.get("ELODIN_FALCON9_PARAMS_JSON", "{}"))

# Mission-design gates derive from the flown mission's own recorded profile
# (a flight is designed against ITS mission, not another one's). Vehicle
# physics parameters stay frozen from calibration — that split is what makes
# the CRS-11 holdout meaningful.
_mission_gates: dict[str, float] = {}
if mission != "crs12":
    _meco_t = REF.events["meco"]
    _entry_t0, _entry_t1 = REF.events["entry_start"], REF.events["entry_end"]
    _mission_gates = {
        "meco_speed_mps": REF.speed(_meco_t),
        "entry_ignite_speed_mps": REF.speed(_entry_t0),
        "entry_ignite_alt_m": REF.altitude(_entry_t0),
        "entry_dv_mps": REF.speed(_entry_t0) - REF.speed(_entry_t1),
    }


def get_param(key: str, default=None):
    if key in _overrides:
        return _overrides[key]
    if key in _mission_gates:
        return _mission_gates[key]
    return params.get(key, default)


GUIDANCE_KEYS = (
    "kick_deg",
    "kick_start_s",
    "kick_ramp_s",
    "bucket_throttle",
    "bucket_q_on_pa",
    "bucket_q_off_pa",
    "meco_speed_mps",
    "azimuth_deg",
    "boostback_overshoot_m",
    "entry_ignite_speed_mps",
    "entry_dv_mps",
    "landing_accel_margin",
    "ascent_throttle",
    "meco_fpa_deg",
    "pitch_exp",
    "entry_throttle",
    "landing_arm_alt_m",
    "entry_ignite_alt_m",
    "fsw_cd_s_m2",
    "boostback_throttle",
)
guidance_values = [float(get_param(k)) for k in GUIDANCE_KEYS]

lox_kg = float(get_param("lox_kg", LOX_LOAD_KG))
rp1_kg = float(get_param("rp1_kg", RP1_LOAD_KG))
upper_kg = STAGE2_WET_KG + PAYLOAD_KG

world, system = build_mission(
    REF,
    lox_kg=lox_kg,
    rp1_kg=rp1_kg,
    upper_kg=upper_kg,
    thrust_scale=float(get_param("thrust_scale", 1.0)),
    isp_scale=float(get_param("isp_scale", 1.0)),
    ca_scale=float(get_param("ca_scale", 1.0)),
    cn_scale=float(get_param("cn_scale", 1.0)),
    display_lag_s=float(get_param("display_lag_s", 1.0)),
)

LZ1_ECEF = np.asarray(
    geodetic_to_ecef(math.radians(LZ1_LAT_DEG), math.radians(LZ1_LON_DEG), LZ1_ALT_M)
)

guidance_period_ticks = max(1, round(SIM_RATE_HZ / GUIDANCE_RATE_HZ))
max_ticks = int(os.environ.get(MAX_TICKS_ENV, str(int((REF.t_end + 40.0) * SIM_RATE_HZ))))

# --- s10 recipe: run the Rust FSW next to the sim -------------------------------
controller_dir = Path(__file__).parent / "controller"
state_port = el.monte_carlo.port("state", DEFAULT_STATE_PORT)
command_port = el.monte_carlo.port("command", DEFAULT_COMMAND_PORT)
fsw_env = {
    "ELODIN_MC_PORT_STATE": str(state_port),
    "ELODIN_MC_PORT_COMMAND": str(command_port),
    # The FSW's designed ascent reference: the recorded profile it must fly.
    "ELODIN_F9_PROFILE": str(Path(__file__).parent / "data" / mission / "stage1_raw.json"),
}
if os.environ.get("ELODIN_MONTE_CARLO_PLANNING") == "1":
    binary = controller_dir / "target" / "release" / "falcon9-fsw"
    if not binary.exists():
        raise RuntimeError(
            "falcon9-fsw binary missing; run the campaign [[build]] step "
            "(cargo build --release --manifest-path examples/falcon9/controller/Cargo.toml)"
        )
    recipe = el.s10.PyRecipe.process(
        name="Falcon9 FSW",
        cmd=str(binary),
        env=fsw_env,
        ready=el.s10.Ready.delay(100),
        ready_timeout="2s",
    )
else:
    recipe = el.s10.PyRecipe.cargo(
        name="Falcon9 FSW",
        path=str(controller_dir),
        env=fsw_env,
        ready=el.s10.Ready.delay(100),
        ready_timeout="2s",
    )
world.recipe(recipe)


class Bridge:
    """Blocking UDP lockstep exchange with the FSW at the guidance rate."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", command_port))
        self.sock.settimeout(1.0)
        self.last_cmd = np.zeros(CMD_FLOATS)
        self.last_cmd[20] = 1.0  # identity quaternion w
        self._payload = np.zeros(STATE_FLOATS)

    def exchange(self, state: np.ndarray) -> np.ndarray:
        self.sock.sendto(struct.pack(f"<{STATE_FLOATS}d", *state), ("127.0.0.1", state_port))
        try:
            raw, _ = self.sock.recvfrom(CMD_FLOATS * 8)
            self.last_cmd = np.frombuffer(raw[: CMD_FLOATS * 8], dtype="<f8").copy()
        except (TimeoutError, socket.timeout):
            pass
        return self.last_cmd


bridge: Bridge | None = None
result_emitted = False
separated = False
purge_events = 0
purge_open_prev = False
touchdown_state: dict | None = None
DEBUG = os.environ.get("ELODIN_FALCON9_DEBUG") == "1"
last_phase = -1.0
last_debug_t = -1.0e9
phase_events: dict[float, dict] = {}
qbar_peak_descent = 0.0

READS = [
    "booster.imu_accel",
    "booster.imu_gyro",
    "booster.gps_pos",
    "booster.gps_vel",
    "booster.gps_count",
    "booster.radar_range",
    "booster.pressure_meas",
    "booster.propellant_lox",
    "booster.propellant_rp1",
    "booster.landed",
    "booster.world_pos",
    "booster.world_vel",
    "booster.altitude_geodetic",
    "booster.ground_speed",
    "booster.score_state",
    "booster.valve_state",
    "booster.nitrogen_kg",
    "booster.fsw_phase",
    "booster.touchdown_metrics",
    "booster.qbar",
]


def post_step(tick: int, ctx: el.StepContext) -> None:
    global bridge, purge_events, purge_open_prev, result_emitted, separated, touchdown_state

    if tick % guidance_period_ticks != 0 and not (result_emitted or tick >= max_ticks - 1):
        return

    reads = ctx.component_batch_operation(reads=READS)
    t_s = tick * SIM_TIME_STEP

    state = np.zeros(STATE_FLOATS)
    state[0] = t_s
    state[1:4] = reads["booster.imu_accel"]
    state[4:7] = reads["booster.imu_gyro"]
    state[7:10] = reads["booster.gps_pos"]
    state[10:13] = reads["booster.gps_vel"]
    state[13] = reads["booster.gps_count"][0]
    state[14] = reads["booster.radar_range"][0]
    state[16:20] = reads["booster.pressure_meas"]
    state[20] = reads["booster.propellant_lox"][0]
    state[21] = reads["booster.propellant_rp1"][0]
    state[22 : 22 + len(guidance_values)] = guidance_values
    state[43] = reads["booster.landed"][0]

    if bridge is None:
        bridge = Bridge()
    cmd = bridge.exchange(state)

    phase = float(cmd[26])
    if not separated and phase >= 5.0:
        separated = True
    if phase not in phase_events:
        phase_events[phase] = {
            "t": round(t_s, 2),
            "v": round(float(reads["booster.ground_speed"][0]), 1),
            "alt": round(float(reads["booster.altitude_geodetic"][0]), 1),
        }

    global last_phase, last_debug_t
    if DEBUG and (phase != last_phase or t_s - last_debug_t >= 10.0):
        alt = float(reads["booster.altitude_geodetic"][0])
        speed = float(reads["booster.ground_speed"][0])
        prop = float(reads["booster.propellant_lox"][0] + reads["booster.propellant_rp1"][0])
        marker = "PHASE" if phase != last_phase else "     "
        print(
            f"[dbg] {marker} t={t_s:7.2f} phase={int(phase):2d} v={speed:7.1f} "
            f"alt={alt / 1000.0:7.2f} km  prop={prop / 1000.0:6.1f} t  "
            f"ref v={REF.speed(t_s):7.1f} alt={REF.altitude(t_s) / 1000.0:7.2f}"
        )
        last_phase = phase
        last_debug_t = t_s

    writes = {
        "booster.engine_cmd": cmd[0:9],
        "booster.valve_cmd": cmd[9:17],
        "booster.attitude_setpoint": cmd[17:21],
        "booster.ctrl_enable": cmd[21:23],
        "booster.fin_cmd": cmd[23:26],
        "booster.fsw_phase": np.array([phase]),
        "booster.upper_mass": np.array([0.0 if separated else upper_kg]),
    }
    ctx.component_batch_operation(writes=writes)

    # Purge audit: count rising edges of the commanded purge valve.
    purge_open = float(cmd[9 + 7]) > 0.5
    if purge_open and not purge_open_prev:
        purge_events += 1
    purge_open_prev = purge_open

    # Track the descent q-bar peak (physics-plausibility metric).
    global qbar_peak_descent
    if phase >= 8.0:
        qbar_peak_descent = max(qbar_peak_descent, float(reads["booster.qbar"][0]))

    landed = float(reads["booster.landed"][0]) > 0.5
    if landed and touchdown_state is None:
        pos = np.asarray(reads["booster.world_pos"])[4:7]
        lat, lon, _ = (float(x) for x in ecef_to_geodetic(pos))
        ned = np.asarray(ned_basis(lat, lon))
        miss_ned = ned @ (pos - LZ1_ECEF)
        touchdown_state = {
            "t": t_s,
            "pos_err_m": float(np.hypot(miss_ned[0], miss_ned[1])),
            "miss_north_m": float(miss_ned[0]),
            "miss_east_m": float(miss_ned[1]),
        }

    # Let the run continue ~2 s past contact so the shutdown purge is visible.
    if not result_emitted and (
        (landed and touchdown_state is not None and t_s - touchdown_state["t"] >= 2.0)
        or tick >= max_ticks - 1
    ):
        score = np.asarray(reads["booster.score_state"])
        n = max(score[2], 1.0)
        pos = np.asarray(reads["booster.world_pos"])[4:7]
        # Impact kinematics latched by the plant at the contact tick — never
        # read post-contact state for these (the zeroed velocity would score
        # every landing as perfect).
        metrics = np.asarray(reads["booster.touchdown_metrics"])
        pos_err = float(np.linalg.norm(pos - LZ1_ECEF))
        result = {
            "landed": landed,
            "touchdown_t_s": (touchdown_state or {}).get("t", t_s),
            "speed_rmse_mps": float(np.sqrt(score[0] / n)),
            "alt_rmse_m": float(np.sqrt(score[1] / n)),
            "touchdown_vertical_mps": float(metrics[0]),
            "touchdown_lateral_mps": float(metrics[1]),
            "touchdown_tilt_deg": float(metrics[2]),
            "touchdown_pos_err_m": (touchdown_state or {}).get("pos_err_m", pos_err),
            "miss_north_m": (touchdown_state or {}).get("miss_north_m", 0.0),
            "miss_east_m": (touchdown_state or {}).get("miss_east_m", 0.0),
            "prop_remaining_kg": float(
                reads["booster.propellant_lox"][0] + reads["booster.propellant_rp1"][0]
            ),
            "nitrogen_remaining_kg": float(reads["booster.nitrogen_kg"][0]),
            "purge_events": purge_events,
            "qbar_peak_descent_kpa": qbar_peak_descent / 1000.0,
            "final_phase": float(reads["booster.fsw_phase"][0]),
        }
        # Phase entry times as scalar metrics (phase id -> seconds).
        for pid, ev in phase_events.items():
            result[f"phase_{int(pid)}_t_s"] = ev["t"]
        result["soft_landing"] = bool(
            landed
            and result["touchdown_vertical_mps"] <= TOUCHDOWN_SOFT_VERTICAL_MPS
            and result["touchdown_lateral_mps"] <= TOUCHDOWN_SOFT_LATERAL_MPS
            and result["touchdown_tilt_deg"] <= 10.0
            and result["touchdown_pos_err_m"] <= 500.0
            and result["prop_remaining_kg"] > 0.0
        )
        if os.environ.get("ELODIN_MONTE_CARLO_CONTEXT"):
            el.monte_carlo.result(**result)
        else:
            print("[falcon9] result:", result)
            print("[falcon9] phases:", {int(k): v for k, v in sorted(phase_events.items())})
        result_emitted = True


world.run(
    system,
    simulation_rate=SIM_RATE_HZ,
    max_ticks=max_ticks,
    optimize=True,
    db_path=params.db_path or os.environ.get("ELODIN_DB_PATH"),
    post_step=post_step,
    interactive=False,
    start_timestamp=START_TIMESTAMP_US,
    log_level="warn",
)
