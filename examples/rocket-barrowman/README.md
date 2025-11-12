# Elodin Rocket Simulation Engine

A high-fidelity 6-DOF (Six Degrees of Freedom) rocket flight simulator with Elodin visualization.

## Features

- **6-DOF Dynamics**: Full quaternion-based orientation tracking with angular velocity
- **Aerodynamic Model**: Barrowman method for lift/drag coefficients with Mach-dependent drag curves
- **Parachute System**: Multi-parachute support with deployment lag, altitude/apogee triggers
- **ISA Atmosphere**: Altitude-varying air density, pressure, temperature, and speed of sound
- **Motor Simulation**: Time-varying thrust and mass flow with realistic burn curves
- **Elodin Integration**: Real-time 3D visualization with telemetry graphs

## Quick Start

### 1. Enter Nix Shell
```bash
cd /home/kush-mahajan/elodin
nix develop
```

### 2. Run Simulation
```bash
python3 examples/rocket-barrowman/main.py
```

This will:
1. Simulate the Calisto rocket (14.4kg, 3.3m apogee)
2. Generate trajectory data
3. Open Elodin editor with 3D visualization

## Architecture

### Core Components

- **`flight_solver.py`** - 6-DOF solver with RK4 integration
- **`rocket_model.py`** - Rocket mass, CG, inertia, and aerodynamics
- **`motor_model.py`** - Motor thrust curve and mass flow
- **`environment.py`** - ISA atmosphere and wind models
- **`calisto_builder.py`** - Calisto rocket definition
- **`main.py`** - Elodin integration entry point

### Physics

#### Equations of Motion
- **Linear**: `F = ma` (thrust + drag + lift + gravity)
- **Angular**: `τ = Iα + ω × (Iω)` (CP-CG moments + damping)
- **Parachute Drag**: `F_drag = -0.5 * ρ * v² * Cd*A * v_unit` (opposes velocity)

#### Aerodynamics
- **Drag**: Mach-dependent CD curve from empirical data
- **Lift**: CNα (normal force coefficient) from Barrowman method
- **Moments**: CP-CG restoring moments + aerodynamic damping

#### Parachute Deployment
- **Triggers**: APOGEE (vertical velocity sign change), ALTITUDE (descent through target), TIME
- **Lag**: Configurable inflation delay (e.g., 1.5s)
- **Multiple Chutes**: Drogue + Main with independent triggers

## Calisto Rocket Specs

- **Mass**: 14.426 kg (dry), 16.676 kg (wet)
- **Length**: 2.338 m
- **Diameter**: 127 mm
- **Motor**: Cesaroni M1670 (3.2s burn, 1670N avg thrust)
- **Fins**: 4x trapezoidal fins (Barrowman aerodynamics)
- **Parachutes**:
  - Drogue: 1.0 m² (deploys at apogee + 1.5s)
  - Main: 10.0 m² (deploys at 800m + 1.5s)

## Visualization

The Elodin schematic (`rocket.kdl`) includes:

- **3D Viewport**: Rocket model with camera tracking
- **Trajectory Trail**: Yellow line showing flight path
- **Telemetry Graphs**:
  - Altitude (m)
  - Velocity (m/s)
  - Mach Number
  - Angle of Attack (rad)
  - Dynamic Pressure (Pa)

## Performance

- **Apogee**: ~3050m (91% accuracy vs reference)
- **Max Velocity**: ~266 m/s (Mach 0.8)
- **Terminal Velocity**: ~5.4 m/s (with both parachutes)
- **Flight Time**: ~200s (to ground impact)

## Extending

### Add a New Rocket

1. Create a builder function in a new file (e.g., `my_rocket_builder.py`)
2. Define components using `openrocket_components.py`
3. Set mass properties, motor, and parachutes
4. Update `main.py` to use your builder

### Modify Aerodynamics

- **Drag Curve**: Edit `calisto_drag_curve.py` or implement custom CD calculation
- **Lift**: Modify CNα scaling in `rocket_model.py`
- **Damping**: Adjust coefficients in `flight_solver.py`

### Add Wind

```python
from dynamic_wind.py import DynamicWindModel

wind_model = DynamicWindModel(
    base_velocity=np.array([5.0, 0.0, 0.0]),  # 5 m/s eastward
    gust_amplitude=2.0,
    gust_frequency=0.1,
)

env = Environment(elevation=1400.0, wind_model=wind_model)
```

## Files

### Core Engine
- `flight_solver.py` - 6-DOF solver (1000 lines)
- `rocket_model.py` - Rocket properties (400 lines)
- `motor_model.py` - Motor simulation (100 lines)
- `environment.py` - Atmosphere models (100 lines)

### Rocket Definition
- `calisto_builder.py` - Calisto rocket (200 lines)
- `calisto_drag_curve.py` - Empirical drag data
- `openrocket_components.py` - Component library (500 lines)
- `openrocket_motor.py` - Motor data structures (300 lines)

### Integration
- `main.py` - Elodin visualization (200 lines)
- `rocket.kdl` - Generated schematic

## License

Part of the Elodin project.

