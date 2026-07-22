"""Physical constants and vehicle/mission configuration for the Falcon 9 example.

Single source of truth for every number the simulation, tests, and flight
software share. Values marked EST are public estimates or calibration
parameters (priors for spec.toml), per WHITEPAPER.md; the rest are published
or standards-defined figures.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# --- WGS84 / Earth (NGA TR8350.2) -------------------------------------------
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared
MU_EARTH_M3S2 = 3.986004418e14
OMEGA_EARTH_RADPS = 7.292115e-5  # about +Z ECEF (sidereal day 86,164.1 s)
G0 = 9.80665

# --- Simulation rates (WHITEPAPER 14.1) --------------------------------------
SIM_RATE_HZ = 1000.0
SIM_DT_S = 1.0 / SIM_RATE_HZ
GUIDANCE_RATE_HZ = 100.0
GPS_RATE_HZ = 25.0
ALTIMETER_RATE_HZ = 40.0
IMU_RATE_HZ = SIM_RATE_HZ

# --- Reference mission: CRS-12 (data/crs12/mission.json) ---------------------
LAUNCH_EPOCH = datetime(2017, 8, 14, 16, 31, 37, tzinfo=timezone.utc)
START_TIMESTAMP_US = int(LAUNCH_EPOCH.timestamp() * 1_000_000)
PAD_LAT_DEG = 28.60839  # LC-39A
PAD_LON_DEG = -80.60433
PAD_ALT_M = 3.0
LZ1_LAT_DEG = 28.48580  # Landing Zone 1
LZ1_LON_DEG = -80.54440
LZ1_ALT_M = 5.0

# --- Stage 1 geometry and mass (WHITEPAPER 9.4) -------------------------------
STAGE1_LENGTH_M = 47.0  # EST: tank stack + interstage
STAGE1_DIAMETER_M = 3.66
S_REF_M2 = math.pi * STAGE1_DIAMETER_M**2 / 4.0  # ~10.52 m^2
STAGE1_DRY_MASS_KG = 25_600.0  # EST (calibration prior 23-27 t)
STAGE1_PROP_KG = 398_000.0  # EST Block 3/4 load (395-400 t; Block 5: 411 t)
OF_RATIO = 2.33  # EST LOX/RP-1 mixture by mass
LOX_LOAD_KG = STAGE1_PROP_KG * OF_RATIO / (1.0 + OF_RATIO)
RP1_LOAD_KG = STAGE1_PROP_KG / (1.0 + OF_RATIO)
# Departing mass at stage separation: stage 2 wet + Dragon C113 + cargo.
STAGE2_WET_KG = 111_500.0  # EST
PAYLOAD_KG = 7_100.0  # Dragon 1 dry ~4,200 + 2,910 kg CRS-12 cargo
LIFTOFF_MASS_KG = STAGE1_DRY_MASS_KG + STAGE1_PROP_KG + STAGE2_WET_KG + PAYLOAD_KG

# --- Merlin 1D, 2017 Block 3/4 (WHITEPAPER 9.1-9.3) ---------------------------
N_ENGINES = 9
ENGINE_A_E_M2 = 0.681  # derived from published Block 5 SL/vac thrust pair
ENGINE_T_SL_N = 760e3  # EST per-engine sea level (Block 5: 845 kN)
P_SL_PA = 101_325.0
ENGINE_T_VAC_N = ENGINE_T_SL_N + P_SL_PA * ENGINE_A_E_M2  # ~829 kN
ENGINE_ISP_SL_S = 282.0  # EST
ENGINE_ISP_VAC_S = ENGINE_ISP_SL_S * ENGINE_T_VAC_N / ENGINE_T_SL_N  # ~308 s
THROTTLE_MIN = 0.57  # EST
# TEA-TEB relight budget: only the center engine and two neighbors restart.
RELIGHT_CAPABLE_ENGINES = 3
ENGINE_SPINUP_TAU_S = 1.5  # EST: chamber pressure steady ~5 s after command (SRP prior)
ENGINE_SHUTDOWN_TAU_S = 0.35  # EST
ENGINE_THROTTLE_TAU_S = 0.15  # EST: throttle response once running

# --- TVC (WHITEPAPER 10.2) ----------------------------------------------------
TVC_MAX_DEG = 5.0  # EST
TVC_RATE_DPS = 20.0  # EST calibration parameter
TVC_TAU_S = 0.030  # EST actuator lag

# --- Cold-gas RCS (WHITEPAPER 10.3) -------------------------------------------
# Sized by the flip budget (WHITEPAPER 10.3): tau ~ 4e5 N m from a 2-thruster
# pitch pair at ~27 m arm -> ~7.5 kN per thruster (calibration prior 3-15 kN).
RCS_THRUST_PER_THRUSTER_N = 7_500.0  # EST
RCS_VALVE_TAU_S = 0.007  # EST ms-class valve response
RCS_STATION_M = 46.0  # EST: interstage pod station (m from engine plane)

# --- Grid fins (WHITEPAPER 8.4) -----------------------------------------------
N_GRID_FINS = 4
FIN_MAX_DEG = 20.0  # EST
FIN_RATE_DPS = 20.0  # EST
FIN_TAU_S = 0.050  # EST

# --- Tanks and valves (WHITEPAPER 9.5) ----------------------------------------
TANK_P_NOM_PA = 3.5e5  # EST pump-fed ullage pressure
VALVE_TAU_S = 0.015  # EST solenoid open/close response
PURGE_DURATION_S = 5.0  # EST nitrogen purge after every cutoff

# --- Landing legs / contact (WHITEPAPER 11.5, 15) ------------------------------
LANDING_MASS_EST_KG = 27_000.0  # EST: dry + remaining propellant reserve
# Published Falcon 9 landing-leg design limit (~2 m/s impact).
TOUCHDOWN_SOFT_VERTICAL_MPS = 2.0
TOUCHDOWN_SOFT_IMPACT_MPS = 2.0
TOUCHDOWN_SOFT_LATERAL_MPS = 1.5
TOUCHDOWN_SOFT_TILT_DEG = 2.0
TOUCHDOWN_SOFT_POS_ERR_M = 5.0
TOUCHDOWN_SOFT_RATE_DPS = 1.0
# 4-pad contact model (ASDS deck ~52×96 m; legs ~10 m radius, ~90° spacing).
LEG_RADIUS_M = 10.0
LEG_STROKE_M = 0.55
LEG_STIFFNESS_NPM = 4.0e5
LEG_DAMPING_NS_PM = 8.0e4
LEG_FRICTION_MU = 0.55
DECK_HALF_ALONG_M = 26.0  # 52 m barge length / 2 (along-track)
DECK_HALF_CROSS_M = 48.0  # 96 m barge width / 2
