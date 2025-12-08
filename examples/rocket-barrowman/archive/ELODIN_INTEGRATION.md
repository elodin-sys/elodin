# Elodin Integration - Calisto Rocket

## Workflow

### 1. Enter Nix Shell
```bash
cd /home/kush-mahajan/elodin
nix develop
```

### 2. Install Dependencies (if needed)
```bash
# The rocket simulation uses standard Python libraries
# elodin should already be available in the nix shell
```

### 3. Run the Simulation
```bash
python3 examples/rocket-barrowman/main.py
```

This will:
1. Run the Calisto rocket 6-DOF simulation
2. Convert the trajectory to Elodin format
3. Generate `rocket.kdl` schematic
4. Open the Elodin editor with visualization

### 4. Visualize in Editor

The editor will show:
- **3D Viewport**: Rocket trajectory with GLB model
- **Altitude Graph**: Height above ground
- **Velocity Graph**: Speed over time
- **Mach Number**: Supersonic flight indicator
- **Angle of Attack**: Aerodynamic angle
- **Dynamic Pressure**: Aerodynamic loading

## Schematic Features

The `rocket.kdl` schematic includes:
- Rocket GLB model from Elodin assets
- Compass reference at ground level
- Trajectory trail (yellow line)
- Camera follows rocket with offset
- Real-time telemetry graphs

## Current Status

✅ **91.1% apogee accuracy** (3051m vs 3349m RocketPy target)
✅ **Dual parachute system** (Drogue + Main with 1.5s lag)
✅ **Proper descent dynamics** (terminal velocity: 5.36 m/s)
✅ **Full 6-DOF simulation** with quaternion orientation
✅ **RocketPy-compatible** physics and aerodynamics

## Files

- `main.py` - Elodin integration entry point
- `rocket.kdl` - Generated schematic (created on run)
- `flight_solver.py` - Core 6-DOF solver
- `calisto_builder.py` - Rocket definition
- `validation_calisto.py` - Validation against RocketPy

