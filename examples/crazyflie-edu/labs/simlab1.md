# Simulation Lab 1: System Setup and Sensor Analysis

In this lab you will set up the Elodin simulation environment and learn how to control the Crazyflie's motors and analyze sensor data.

**Read the "Deliverables" section before starting work**, so that you know what information you need to record as you proceed through the lab.

---

## 1.1 Prerequisites

Before starting this lab, ensure you have:

1. **Elodin installed** - Follow the repository setup instructions
2. **Python environment activated** - The simulation runs in Python

```bash
# From the repository root
nix develop
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
```

---

## 1.2 Running the Simulation

### 1.2.1 Starting the Editor

Launch the Elodin editor with the Crazyflie simulation:

```bash
elodin editor examples/crazyflie-edu/main.py
```

You should see:
- A 3D viewport showing the Crazyflie quadcopter
- Graph panels for sensors (gyroscope, accelerometer) and motors
- The simulation paused, ready to run

### 1.2.2 Editor Controls

| Action | Control |
|--------|---------|
| Play/Pause | Spacebar |
| Step forward | Right arrow |
| Reset simulation | R |
| Rotate camera | Left-click + drag |
| Pan camera | Middle-click + drag |
| Zoom | Scroll wheel |

### 1.2.3 Understanding the Coordinate System

The simulation uses:
- **Body frame**: +X forward, +Y left, +Z up
- **World frame**: +X east, +Y north, +Z up

The colored arrows in the viewport show the body-frame axes.

---

## 1.3 Code Structure

Open the `examples/crazyflie-edu/` directory in your editor. The key files are:

| File | Purpose |
|------|---------|
| `main.py` | Entry point, sets up simulation |
| `config.py` | Crazyflie physical parameters |
| `user_code.py` | **Your code goes here!** |
| `sim.py` | Physics simulation |
| `sensors.py` | IMU sensor simulation |
| `crazyflie_api.py` | API for reading sensors/writing motors |

### 1.3.1 The User Code File

Open `user_code.py`. This is where you write your control code. The structure is:

```python
def main_loop(state: CrazyflieState) -> None:
    """
    Called every control cycle (500 Hz).

    state.gyro[0], gyro[1], gyro[2]  - Angular velocity (rad/s)
    state.accel[0], accel[1], accel[2] - Acceleration (g units)
    state.is_armed      - Vehicle armed state
    state.button_blue   - Blue button pressed

    state.set_motors(m1, m2, m3, m4)  - Set motor commands (0-255)
    state.set_all_motors(cmd)         - Set all motors same value
    state.motors_off()                - Turn all motors off
    """
    # YOUR CODE HERE
    state.motors_off()
```

### 1.3.2 Setting Your Team ID

In `user_code.py`, find and set your team ID:

```python
TEAM_ID = 0  # TODO: Set your team ID here!
```

---

## 1.4 Safety

Even in simulation, we practice good safety habits:

1. **The vehicle must be armed** before motors will spin
2. **The red button** acts as a dead man switch (in hardware, you must hold it)
3. **Start with low motor commands** (50 or less) until you understand the behavior

To arm the vehicle in simulation, you'll use the GUI controls (coming soon) or set `state.is_armed` programmatically for testing.

---

## 1.5 Experiments

### 1.5.1 Experiment 1: Turning the Motors On

**Objective**: Write code to turn on motors when the blue button is pressed.

In `user_code.py`, modify the `main_loop()` function:

```python
def main_loop(state: CrazyflieState) -> None:
    if state.is_armed and state.button_blue:
        # Set all motors to a low test value
        state.set_all_motors(50)  # Don't exceed 50 for now!
    else:
        state.motors_off()
```

**Procedure**:
1. Save your changes to `user_code.py`
2. Restart the simulation (the editor auto-reloads)
3. Arm the vehicle
4. Press the blue button - observe the motors spinning
5. Release the blue button - motors should stop

**Record**:
- Observe the motor RPM graph
- Note the gyroscope readings (should show slight vibration)

### 1.5.2 Experiment 2: Sensor Analysis - Motors Off

**Objective**: Analyze gyroscope and accelerometer data with motors off.

**Procedure**:
1. Ensure motors are off (blue button not pressed)
2. Let the simulation run for at least 30 seconds
3. Observe the sensor graphs

**Analysis**:
- The gyroscope should read near zero (small noise)
- The accelerometer should read approximately [0, 0, 1] g (gravity)

### 1.5.3 Experiment 3: Sensor Analysis - Motors On

**Objective**: Detect vibrations from running motors.

**Procedure**:
1. Turn on motors at command = 50
2. Let run for 30 seconds while recording data
3. Compare gyro readings to motors-off case

**Expected Results**:
- Gyroscope noise should increase due to motor vibrations
- Mean should still be near zero (no net rotation)
- Standard deviation will be higher than motors-off case

### 1.5.4 Experiment 4: Manipulating the Vehicle

**Objective**: Observe sensor response to vehicle motion.

**Procedure**:
1. With motors off, "grab" the vehicle in the GUI
2. Rotate the vehicle to various orientations
3. Observe how accelerometer readings change
4. Drop the vehicle - observe acceleration spike

**Key Observations**:
- Accelerometer measures "specific force" (includes gravity)
- At rest upside-down: accel ≈ [0, 0, -1] g
- During free-fall: accel ≈ [0, 0, 0] g

### 1.5.5 Experiment 5: Noise Enable/Disable

**Objective**: Compare sensor readings with and without simulated noise.

You can disable noise by modifying `config.py`:

```python
sensor_noise: bool = False  # Disable noise
```

Or run with the `--no-noise` flag:

```bash
python examples/crazyflie-edu/main.py --no-noise
```

**Procedure**:
1. Record 20 seconds with noise enabled
2. Restart with noise disabled
3. Record 20 seconds with noise disabled
4. Compare the graphs

---

## 1.6 Data Logging

The simulation logs data that you can analyze. Telemetry is stored in Elodin-DB format and can be exported to CSV.

To analyze logged data, check the `analysis/` folder for example scripts.

---

## 1.7 Deliverables

Submit a report containing:

### 1. Contributions
For each team member, briefly describe their contribution.

### 2. Running the Motors [50%]

a. Provide a listing of your `main_loop()` function in `user_code.py`, clearly identifying the code you added.

b. Include a screenshot of the simulation with motors running.

### 3. Sensor Analysis [50%]

a. Create plots of gyroscope readings (all 3 axes) with motors off and motors on. Label axes clearly with units.

b. Compute the mean and standard deviation of gyro readings for both cases. Present in a table:

| Condition | Gyro X Mean | Gyro X Std | Gyro Y Mean | Gyro Y Std | Gyro Z Mean | Gyro Z Std |
|-----------|-------------|------------|-------------|------------|-------------|------------|
| Motors Off | | | | | | |
| Motors On | | | | | | |

c. Comment on the effect of running motors on the sensor readings.

d. Plot accelerometer readings during vehicle manipulation. Mark the maximum values observed.

e. Compare noise-enabled vs noise-disabled sensor readings. What differences do you observe?

---

## 1.8 Next Steps

In **SimLab2**, you will:
- Characterize the motor powertrain (PWM → speed → force)
- Implement utility functions for control
- Validate by achieving near-hover thrust

---

## Tips

- **Debugging**: Add `print()` statements in your `main_loop()` - they appear in the terminal
- **Reset often**: Press 'R' to reset the simulation if things go wrong
- **Low power first**: Always start with low motor commands (≤50) until you're confident
- **Save frequently**: Keep backups of working code before making changes

