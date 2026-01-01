# Simulation Lab 2: Powertrain Identification

In this lab you will identify the characteristics of the Crazyflie's motors and propellers. By the end, you will be able to command a desired thrust force, which is essential for flight control.

**Read the "Deliverables" section before starting work.**

---

## 2.1 Introduction

The Crazyflie's propulsion system converts electrical PWM commands into thrust force through two stages:

```
PWM Command → [Motor] → Angular Velocity → [Propeller] → Thrust Force
```

You will identify two relationships:
1. **PWM → Speed**: How motor speed (rad/s) relates to PWM command (0-255)
2. **Speed → Force**: How thrust force (N) relates to motor speed (rad/s)

### 2.1.1 Why This Matters

To hover, each motor must produce exactly 1/4 of the vehicle's weight:

```
Hover thrust per motor = (mass × g) / 4 = (0.027 kg × 9.81 m/s²) / 4 ≈ 0.066 N
```

To achieve this, you need to know what PWM command produces 0.066 N of thrust.

### 2.1.2 The Mathematical Models

**PWM to Speed (Affine/Linear)**:
```
ω = a + b × PWM
```
Where ω is angular velocity in rad/s, and a, b are constants to determine.

**Speed to Force (Quadratic)**:
```
F = k × ω²
```
Where F is thrust in Newtons, and k is the thrust constant (units: N/(rad/s)²).

---

## 2.2 Part 1: PWM to Speed Mapping

### 2.2.1 Experimental Data

In simulation, we have "measured" the motor speed at various PWM commands using a virtual tachometer. The data is:

| PWM Command | Measured Speed (RPM) | Speed (rad/s) |
|-------------|---------------------|---------------|
| 40          | 4340                | 454.5         |
| 80          | 4610                | 482.8         |
| 120         | 4880                | 511.1         |
| 160         | 5150                | 539.4         |
| 200         | 5420                | 567.6         |
| 240         | 5690                | 595.9         |

**Note**: Convert RPM to rad/s using: `ω (rad/s) = RPM × 2π / 60`

### 2.2.2 Your Task

1. **Plot the data**: Create a scatter plot of PWM (x-axis) vs Speed in rad/s (y-axis)

2. **Fit an affine model**: Find constants `a` and `b` such that:
   ```
   ω = a + b × PWM
   ```

   You can use least-squares fitting:
   ```python
   import numpy as np

   pwm = np.array([40, 80, 120, 160, 200, 240])
   rpm = np.array([4340, 4610, 4880, 5150, 5420, 5690])
   omega = rpm * 2 * np.pi / 60  # Convert to rad/s

   # Fit: omega = a + b * pwm
   A = np.vstack([np.ones_like(pwm), pwm]).T
   a, b = np.linalg.lstsq(A, omega, rcond=None)[0]
   print(f"a = {a:.2f}, b = {b:.4f}")
   ```

3. **Implement the inverse function**: To command a desired speed, you need:
   ```
   PWM = (ω - a) / b
   ```

### 2.2.3 Update Your Code

In `user_code.py`, update the constants:

```python
# PWM to Speed mapping: omega = PWM_TO_SPEED_A + PWM_TO_SPEED_B * pwm
PWM_TO_SPEED_A = ???  # Your value here (rad/s at PWM=0)
PWM_TO_SPEED_B = ???  # Your value here (rad/s per PWM unit)
```

And implement the function:

```python
def pwm_from_speed(desired_speed_rad_s: float) -> float:
    """Convert desired motor speed (rad/s) to PWM command."""
    pwm = (desired_speed_rad_s - PWM_TO_SPEED_A) / PWM_TO_SPEED_B
    return max(0, min(255, pwm))
```

---

## 2.3 Part 2: Speed to Force Mapping

### 2.3.1 The Force Rig

To measure thrust, we use a simulated force rig. The rig is a balance arm with the Crazyflie mounted at one end. By measuring the tilt angle, we can calculate the thrust force.

**Physics**:
When the rig is at equilibrium angle β:
```
4 × F × l = (m_B × g × l + m_R × g × l_R) × sin(β)
```

Where:
- F = thrust per motor (what we want)
- l = 0.150 m (distance from pivot to Crazyflie CoM)
- l_R = 0.070 m (distance from pivot to rig arm CoM)
- m_B = 0.027 kg (Crazyflie mass)
- m_R = 0.0085 kg (rig arm mass)

The accelerometer Z reading gives us the angle:
```
accel_z = g × sin(β)  →  sin(β) = accel_z / g
```

### 2.3.2 Procedure

1. **Enable the force rig** in the simulation (mount the vehicle)

2. **For each test speed**, modify your code to command that speed:
   ```python
   TEST_SPEED = 1000  # rad/s - change this for each test

   def main_loop(state: CrazyflieState) -> None:
       if state.is_armed and state.button_blue:
           pwm = pwm_from_speed(TEST_SPEED)
           state.set_all_motors(pwm)
       else:
           state.motors_off()
   ```

3. **Record data** at these speeds: {1000, 1100, 1200, 1300, 1400} rad/s

4. **For each test**:
   - Run until steady state (~5 seconds)
   - Record the average accelerometer Z reading
   - Calculate the thrust force

### 2.3.3 Data Analysis

For each test point, calculate:

1. **Angle from accelerometer**:
   ```python
   accel_z = <your measured value>  # in g units
   sin_beta = accel_z  # since accel is already in g
   ```

2. **Thrust per motor** (solving the moment balance):
   ```python
   g = 9.81
   l = 0.150
   l_R = 0.070
   m_B = 0.027
   m_R = 0.0085

   # Moment balance: 4*F*l = (m_B*l + m_R*l_R) * g * sin(beta)
   F = (m_B * l + m_R * l_R) * g * sin_beta / (4 * l)
   ```

3. **Fit the thrust constant**: Plot F vs ω² and find k:
   ```python
   # F = k * omega^2
   # Linear regression through origin
   omega_squared = omega ** 2
   k = np.sum(F * omega_squared) / np.sum(omega_squared ** 2)
   ```

### 2.3.4 Update Your Code

In `user_code.py`, update:

```python
# Thrust constant: force = THRUST_CONSTANT * omega^2
THRUST_CONSTANT = ???  # Your value (should be ~10^-8 N/(rad/s)^2)
```

And implement:

```python
def speed_from_force(desired_force_n: float) -> float:
    """Convert desired thrust (N) to motor speed (rad/s)."""
    if desired_force_n <= 0:
        return 0.0
    return math.sqrt(desired_force_n / THRUST_CONSTANT)
```

---

## 2.4 Validation

### 2.4.1 Test: 90% Hover Thrust

Calculate the PWM needed for 90% of hover thrust:

```python
g = 9.81
mass = 0.027  # kg
hover_thrust_total = mass * g  # N
hover_thrust_per_motor = hover_thrust_total / 4  # N

# 90% thrust
test_thrust = 0.9 * hover_thrust_per_motor
test_speed = speed_from_force(test_thrust)
test_pwm = pwm_from_speed(test_speed)

print(f"90% hover thrust: {test_thrust:.4f} N per motor")
print(f"Required speed: {test_speed:.1f} rad/s")
print(f"Required PWM: {test_pwm:.0f}")
```

### 2.4.2 Experiment

1. Remove the force rig
2. Command 90% hover thrust
3. "Drop" the vehicle in simulation
4. **Expected**: Vehicle falls slowly (not quite hovering)

5. Repeat with 110% thrust
6. **Expected**: Vehicle accelerates upward

```python
def main_loop(state: CrazyflieState) -> None:
    if state.is_armed and state.button_blue:
        # Command 90% hover thrust
        thrust_per_motor = 0.9 * (0.027 * 9.81) / 4
        pwm = pwm_from_force(thrust_per_motor)
        state.set_all_motors(pwm)
    else:
        state.motors_off()
```

---

## 2.5 Deliverables

### 1. Contributions
For each team member, describe their contribution.

### 2. PWM-Speed Map [50%]

a. Plot showing PWM command vs measured speed (rad/s). Overlay your fitted line.

b. Report your fit coefficients:
   - a = ___ rad/s
   - b = ___ rad/s per PWM unit

c. Provide the code listing of `pwm_from_speed()`.

d. Comment on the quality of fit and any observations.

### 3. Speed-Force Map [50%]

a. Describe the experimental setup (force rig geometry, measurements taken).

b. Table of your measurements:

| Speed (rad/s) | Accel Z (g) | sin(β) | Thrust/motor (N) |
|---------------|-------------|--------|------------------|
| 1000          |             |        |                  |
| 1100          |             |        |                  |
| 1200          |             |        |                  |
| 1300          |             |        |                  |
| 1400          |             |        |                  |

c. Plot showing ω² (x-axis) vs Force (y-axis). Overlay your quadratic fit.

d. Report your thrust constant:
   - k = ___ N/(rad/s)²

e. Provide the code listing of `speed_from_force()`.

f. Describe your validation results (90% and 110% hover tests).

### 4. Complete Code

Provide the full listing of your `main_loop()` function and utility functions.

---

## 2.6 Next Steps

In **HWLab1**, you will:
- Set up the Crazyflie hardware and radio
- Flash firmware to the real vehicle
- Port your Python code to C

**Important**: Make a backup of your working code! Email it to all team members.

