# Simulation Lab 1: System Setup and Sensor Analysis

In this lab you will set up the Elodin simulation environment and learn how to control the Crazyflie's motors and analyze sensor data.

**Read the "Deliverables" section before starting work**, so that you know what information you need to record as you proceed through the lab.

---

## 1.1 Prerequisites

Before starting this lab, ensure you have:

1. **Elodin installed** - Follow the repository setup instructions
2. **Development environment ready** - Nix + Python + C compiler

```bash
# From the repository root
nix develop
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
uv pip install pynput  # For keyboard input
```

---

## 1.2 Running the Simulation

### 1.2.1 Build the SITL Binary

The simulation uses a C binary that runs your control code. Build it first (from repo root):

```bash
./examples/crazyflie-edu/sitl/build.sh
```

### 1.2.2 Starting the Editor

Launch the Elodin editor with the Crazyflie simulation:

```bash
elodin editor examples/crazyflie-edu/main.py
```

You should see:
- A 3D viewport showing the Crazyflie quadcopter
- Graph panels for sensors (gyroscope, accelerometer) and motors
- The simulation paused, ready to run

### 1.2.3 Editor Controls

| Action | Control |
|--------|---------|
| Play/Pause | Spacebar |
| Step forward | Right arrow |
| Reset simulation | R |
| Rotate camera | Left-click + drag |
| Pan camera | Middle-click + drag |
| Zoom | Scroll wheel |

### 1.2.4 Keyboard Controls (Runtime)

| Key | Action |
|-----|--------|
| Q | Toggle armed state |
| Left Shift | Blue button (hold to enable motors) |
| E / R / T | Yellow / Green / Red buttons |

### 1.2.5 Understanding the Coordinate System

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
| `user_code.c` | **Your code goes here!** |
| `user_code.h` | API header for your code |
| `sitl/` | SITL binary (UDP communication) |
| `sim.py` | Physics simulation |
| `sensors.py` | IMU sensor simulation |

### 1.3.1 The User Code File

Open `user_code.c`. This is where you write your control code. The structure is:

```c
#include "user_code.h"

void user_main_loop(user_state_t* state) {
    /*
     * Called every control cycle (500 Hz).
     *
     * Sensors (read-only):
     *   state->sensors.gyro.x/y/z  - Angular velocity (rad/s)
     *   state->sensors.accel.x/y/z - Acceleration (g units)
     *
     * Control inputs (read-only):
     *   state->is_armed      - Vehicle armed state
     *   state->button_blue   - Blue button pressed (dead man switch)
     *
     * Motor control (use helper functions):
     *   user_set_motors(state, m1, m2, m3, m4)  - Set individual motors
     *   user_set_all_motors(state, pwm)         - Set all motors same
     *   user_motors_off(state)                  - Turn all motors off
     */
    
    // YOUR CODE HERE
    user_motors_off(state);
}
```

### 1.3.2 Setting Your Team ID

In `user_code.c`, find and set your team ID:

```c
#define TEAM_ID 0  // TODO: Set your team ID here!
```

### 1.3.3 Rebuilding After Changes

**Important**: After editing `user_code.c`, you must rebuild the SITL binary (from repo root):

```bash
./examples/crazyflie-edu/sitl/build.sh
```

Then restart the simulation (Ctrl+C and re-run, or press R in the editor).

---

## 1.4 Safety

Even in simulation, we practice good safety habits:

1. **The vehicle must be armed** before motors will spin (press Q)
2. **The blue button** acts as a dead man switch (hold Left Shift)
3. **Start with low motor commands** (10000 or less) until you understand the behavior

---

## 1.5 Experiments

### 1.5.1 Experiment 1: Turning the Motors On

**Objective**: Write code to turn on motors when armed and blue button is pressed.

In `user_code.c`, modify the `user_main_loop()` function:

```c
void user_main_loop(user_state_t* state) {
    if (state->is_armed && state->button_blue) {
        // Set all motors to a low test value
        // PWM range is 0-65535, start with ~15%
        user_set_all_motors(state, 10000);
    } else {
        user_motors_off(state);
    }
}
```

**Procedure** (all commands from repo root):
1. Save your changes to `user_code.c`
2. Rebuild: `./examples/crazyflie-edu/sitl/build.sh`
3. Run the simulation: `elodin editor examples/crazyflie-edu/main.py`
4. Press Q to arm the vehicle
5. Hold Left Shift (blue button) - observe the motors spinning
6. Release Left Shift - motors should stop

**Record**:
- Observe the motor RPM graph
- Note the gyroscope readings (should show slight vibration)

### 1.5.2 Experiment 2: Sensor Analysis - Motors Off

**Objective**: Analyze gyroscope and accelerometer data with motors off.

**Procedure**:
1. Ensure motors are off (don't hold blue button)
2. Let the simulation run for at least 30 seconds
3. Observe the sensor graphs

**Analysis**:
- The gyroscope should read near zero (small noise)
- The accelerometer should read approximately [0, 0, 1] g (gravity)

### 1.5.3 Experiment 3: Sensor Analysis - Motors On

**Objective**: Detect vibrations from running motors.

**Procedure**:
1. Turn on motors (arm + hold blue button)
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

a. Provide a listing of your `user_main_loop()` function in `user_code.c`, clearly identifying the code you added.

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

- **Rebuild after changes**: Always run `./examples/crazyflie-edu/sitl/build.sh` after editing `user_code.c`
- **Debugging**: Add `printf()` statements in your code - they appear in the terminal
- **Reset often**: Press 'R' to reset the simulation if things go wrong
- **Low power first**: Always start with low motor commands until you're confident
- **Save frequently**: Keep backups of working code before making changes
