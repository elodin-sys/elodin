# Ardupilot parameters:
# See https://ardupilot.org/copter/docs/parameters.html for more information.

SCHED_LOOP_RATE = 300
MOT_THST_HOVER = 0.35
MOT_THST_EXPO = 0.65
MOT_SPIN_ARM = 0.10
MOT_SPIN_MIN = 0.15
MOT_SPIN_MAX = 0.95
INS_GYRO_FILTER = 40  # 40 Hz for 10 inch props
INS_ACCEL_FILTER = 10
ATC_INPUT_TC = 0.1  # "crisp"
PILOT_Y_RATE_TC = 0.1  # "crisp"

ATC_ACCEL_P_MAX = 110000.0  # (centi-degrees/s^2)
ATC_ACCEL_R_MAX = 110000.0  # (centi-degrees/s^2)
ATC_ACCEL_Y_MAX = 27000.0  # (centi-degrees/s^2)

# Roll, pitch, and yaw angle P gains
ATC_ANG_RLL_P = 4.0
ATC_ANG_PIT_P = 4.0
ATC_ANG_YAW_P = 1.0

# Roll rate PID gains
ATC_RAT_RLL_P = 0.40
ATC_RAT_RLL_I = 0.02
ATC_RAT_RLL_D = 0.05

# Pitch rate PID gains
ATC_RAT_PIT_P = 0.40
ATC_RAT_PIT_I = 0.02
ATC_RAT_PIT_D = 0.05

# Yaw rate PID gains
ATC_RAT_YAW_P = 2.50
ATC_RAT_YAW_I = 0.02
ATC_RAT_YAW_D = 0.01

ATC_RAT_RLL_FLTT = 20.0  # (Hz)
ATC_RAT_RLL_FLTE = 0.0  # (Hz)
ATC_RAT_RLL_FLTD = 20.0  # (Hz)

ATC_RAT_PIT_FLTT = 20.0  # (Hz)
ATC_RAT_PIT_FLTE = 0.0  # (Hz)
ATC_RAT_PIT_FLTD = 20.0  # (Hz)

ATC_RAT_YAW_FLTT = 20.0  # (Hz)
ATC_RAT_YAW_FLTE = 2.5  # (Hz)
ATC_RAT_YAW_FLTD = 20.0  # (Hz)

# Other non-Ardupilot parameters:

# Time it takes for the step response of a motor to reach 63.2% of its final value (s)
MOT_TIME_CONST = 0.1
