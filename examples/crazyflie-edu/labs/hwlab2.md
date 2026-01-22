# Hardware Lab 2: Powertrain Validation

In this lab you will validate your powertrain model from SimLab2 using real hardware measurements, then achieve stable hover.

> ⚠️ **SAFETY IS PARAMOUNT**
>
> This lab involves spinning propellers. Students who act irresponsibly will be ejected from the lab.
>
> - Wear safety glasses at all times
> - Keep fingers and faces away from propellers
> - Secure the vehicle before spinning propellers at high speed
> - Have a clear emergency stop plan (power off)
> - Work in a clear area with no obstacles

**Read the "Deliverables" section before starting work.**

---

## 4.1 Equipment

- Crazyflie 2.1 with propellers installed
- Crazyradio PA USB dongle
- Optical tachometer (for RPM measurement)
- Reflective tape strips
- Masking tape (for securing vehicle)
- Scale or force rig (for thrust measurement)
- Safety glasses (REQUIRED)

---

## 4.2 Part 1: PWM to Speed Validation

### 4.2.1 Tachometer Setup

1. **Apply reflective tape**: Place a small strip of reflective tape on one propeller blade
   - Use minimal tape to avoid imbalance
   - Apply smoothly to the flat part of the blade

2. **Secure the vehicle**: Tape the Crazyflie body to the table
   - Ensure propellers can spin freely
   - Vehicle should not be able to fly away

3. **Position tachometer**:
   - Hold ~30cm from propeller
   - Angle slightly (not perpendicular to table) to avoid false readings
   - Ensure you can see the display while measuring

### 4.2.2 Measurement Procedure

> ⚠️ **The tachometer uses a laser. Do not point at eyes.**

For each PWM command in {40, 80, 120, 160, 200, 240}:

1. **Update your code** to command that PWM value:
   ```c
   void user_main_loop(user_state_t* state) {
       if (state->button_blue) {
           user_set_all_motors(state, 10000);  // Test PWM value
       } else {
           user_motors_off(state);
       }
   }
   ```

2. **Measure RPM**:
   - Start motors (press blue button)
   - Point tachometer at reflective strip
   - Hold TEST button, read display
   - Record 3 measurements for repeatability

3. **Record in table**:

| PWM | Run 1 (RPM) | Run 2 (RPM) | Run 3 (RPM) | Average (RPM) |
|-----|-------------|-------------|-------------|---------------|
| 40  |             |             |             |               |
| 80  |             |             |             |               |
| 120 |             |             |             |               |
| 160 |             |             |             |               |
| 200 |             |             |             |               |
| 240 |             |             |             |               |

### 4.2.3 Compare with Simulation

Plot your hardware measurements against the simulation predictions:

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulation values (from SimLab2)
sim_pwm = np.array([40, 80, 120, 160, 200, 240])
sim_rpm = np.array([...])  # Your simulation data

# Hardware measurements
hw_pwm = np.array([40, 80, 120, 160, 200, 240])
hw_rpm = np.array([...])  # Your hardware data

plt.figure(figsize=(10, 6))
plt.plot(sim_pwm, sim_rpm, 'b-o', label='Simulation')
plt.plot(hw_pwm, hw_rpm, 'r-s', label='Hardware')
plt.xlabel('PWM Command')
plt.ylabel('Motor Speed (RPM)')
plt.legend()
plt.grid(True)
plt.title('PWM to Speed: Simulation vs Hardware')
plt.savefig('pwm_speed_comparison.png')
```

### 4.2.4 Update Your Model

If hardware differs significantly from simulation:

1. Recalculate fit coefficients using hardware data
2. Update `PWM_TO_SPEED_A` and `PWM_TO_SPEED_B` in firmware
3. Re-flash and verify

---

## 4.3 Part 2: Speed to Force Validation

### 4.3.1 Force Measurement Options

**Option A: Digital Scale**
- Place Crazyflie upside-down on a scale
- Measure thrust as decrease in apparent weight

**Option B: Simple Force Rig**
- Mount vehicle on a pivot arm (like SimLab2)
- Measure tilt angle with phone inclinometer
- Calculate thrust from moment balance

**Option C: Hover Test**
- Command calculated hover thrust
- Observe if vehicle hovers, climbs, or descends

### 4.3.2 Procedure (using hover test)

The simplest validation is to test hover:

1. **Calculate hover PWM** using your simulation model:
   ```c
   float hoverThrustPerMotor = (0.027 * 9.81) / 4;  // ~0.066 N
   float hoverSpeed = speedFromForce(hoverThrustPerMotor);
   uint16_t hoverPwm = pwmFromSpeed(hoverSpeed);
   ```

2. **Test at 80% hover thrust**:
   - Vehicle should descend slowly when released
   - Confirms thrust < weight

3. **Test at 100% hover thrust**:
   - Vehicle should maintain altitude (approximately)
   - Some drift is expected without attitude control

4. **Test at 120% hover thrust**:
   - Vehicle should climb
   - **Be careful!** Have plenty of ceiling clearance

### 4.3.3 Safety Precautions for Thrust Testing

> ⚠️ **CRITICAL SAFETY**

- **Tether the vehicle**: Use a string attached to a fixed point
- **Start low**: Begin at 50% calculated hover thrust
- **Increment slowly**: Increase by 10% at a time
- **Clear area**: Ensure no people/objects in flight path
- **Emergency stop**: Be ready to release button instantly

### 4.3.4 Record Results

| Test | Thrust (% hover) | PWM | Observed Behavior |
|------|------------------|-----|-------------------|
| 1    | 80%              |     |                   |
| 2    | 90%              |     |                   |
| 3    | 100%             |     |                   |
| 4    | 110%             |     |                   |
| 5    | 120%             |     |                   |

---

## 4.4 Model Refinement

### 4.4.1 If Thrust is Lower Than Expected

Possible causes:
- Battery voltage low
- Motor wear/degradation
- Propeller damage
- Simulation model too optimistic

Solutions:
- Charge battery fully
- Increase thrust constant in model
- Check propellers for chips/cracks

### 4.4.2 If Thrust is Higher Than Expected

Possible causes:
- Simulation model too conservative
- Fresh/clean propellers

Solutions:
- Decrease thrust constant in model
- Re-verify measurements

### 4.4.3 Update Firmware Constants

After validation, update your firmware with calibrated values:

```c
// Updated from hardware measurements
static float PWM_TO_SPEED_A = ???;     // Your calibrated value
static float PWM_TO_SPEED_B = ???;     // Your calibrated value
static float THRUST_CONSTANT = ???;    // Your calibrated value
```

---

## 4.5 Final Hover Test

### 4.5.1 Procedure

1. **Fully charge battery**

2. **Clear the area** (minimum 2m radius)

3. **Tether the vehicle** with a 1m string to a fixed point

4. **Command hover thrust**:
   ```c
   void user_main_loop(user_state_t* state) {
       if (state->is_armed && state->button_blue) {
           float thrust = (0.027f * 9.81f) / 4.0f;  // Hover thrust
           uint16_t pwm = pwm_from_force(thrust);
           user_set_all_motors(state, pwm);
       } else {
           user_motors_off(state);
       }
   }
   ```

5. **Test sequence**:
   - Arm the vehicle
   - Hold blue button
   - Vehicle should lift off and hover (with tether)
   - Release button to descend
   - Disarm

### 4.5.2 Success Criteria

✅ **Success**: Vehicle achieves stable hover at calculated thrust level

⚠️ **Partial Success**: Vehicle hovers but requires thrust adjustment >20%

❌ **Needs Work**: Vehicle cannot achieve hover or is unstable

---

## 4.6 Deliverables

### 1. Contributions
For each team member, describe their contribution.

### 2. PWM-Speed Validation [40%]

a. **Experimental setup**: Photo of tachometer measuring propeller speed

b. **Data table**: All RPM measurements (3 runs per PWM level)

c. **Comparison plot**: Simulation vs hardware measurements

d. **Analysis**: Percent difference between simulation and hardware
   ```
   Error (%) = |Hardware - Simulation| / Simulation × 100
   ```

e. **Updated coefficients** (if changed from simulation)

### 3. Thrust Validation [40%]

a. **Experimental setup**: Photo/description of test method

b. **Results table**: Thrust tests at various levels

c. **Updated thrust constant** (if changed from simulation)

d. **Hover test**: Video evidence or detailed description of hover attempt

### 4. Final Analysis [20%]

a. **Summary table**:

| Parameter | Simulation | Hardware | Difference |
|-----------|------------|----------|------------|
| PWM_TO_SPEED_A | | | |
| PWM_TO_SPEED_B | | | |
| THRUST_CONSTANT | | | |
| Hover PWM | | | |

b. **Discussion**:
   - What were the main sources of error?
   - How well did simulation predict hardware behavior?
   - What would you do differently next time?

c. **Complete firmware code listing** with final calibrated values

---

## 4.7 Congratulations!

You have completed the powertrain identification labs. You now have:

- ✅ A validated model of PWM → Speed → Force
- ✅ Calibrated firmware constants
- ✅ Demonstrated hover capability

### Next Steps (Optional Advanced Labs)

- **Lab 5**: Attitude estimation (using gyro/accel fusion)
- **Lab 6**: Attitude control (PID loops for stabilization)
- **Lab 7**: Position control (autonomous flight)

### Important

**Make a backup of your final calibrated firmware!**

Email the complete source code to all team members. You will need these values for future labs.

