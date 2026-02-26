# Ardupilot parameters:
# See https://ardupilot.org/copter/docs/parameters.html for more information.

MOT_SPIN_ARM = 0.10
MOT_SPIN_MIN = 0.12
MOT_SPIN_MAX = 0.95
MOT_PWM_MIN = 1050
MOT_PWM_MAX = 1900

INS_GYRO_FILTER = 40  # 40 Hz for 10 inch props
INS_ACCEL_FILTER = 20

ATC_ACCEL_P_MAX = 110000.0  # (centi-degrees/s^2)
ATC_ACCEL_R_MAX = 110000.0  # (centi-degrees/s^2)
ATC_ACCEL_Y_MAX = 27000.0  # (centi-degrees/s^2)

ATC_RAT_RLL_FLTT = 20.0  # (Hz)
ATC_RAT_RLL_FLTE = 0.0  # (Hz)
ATC_RAT_RLL_FLTD = 10.0  # (Hz)

ATC_RAT_PIT_FLTT = 20.0  # (Hz)
ATC_RAT_PIT_FLTE = 0.0  # (Hz)
ATC_RAT_PIT_FLTD = 10.0  # (Hz)

ATC_RAT_YAW_FLTT = 20.0  # (Hz)
ATC_RAT_YAW_FLTE = 2.5  # (Hz)
ATC_RAT_YAW_FLTD = 0.0  # (Hz)

# Other non-Ardupilot parameters:

# Time it takes for the step response of a motor to reach 63.2% of its final value (s)
MOT_TIME_CONST = 0.1
MOT_PWM_THST_MIN = MOT_PWM_MIN + (MOT_PWM_MAX - MOT_PWM_MIN) * MOT_SPIN_MIN
MOT_PWM_THST_MAX = MOT_PWM_MIN + (MOT_PWM_MAX - MOT_PWM_MIN) * MOT_SPIN_MAX
