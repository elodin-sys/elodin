# Rocket Flight Simulator: Technical Whitepaper

## Overview

This document provides a comprehensive technical overview of the Rocket Flight Simulator, a high-fidelity 6-degrees-of-freedom (6-DOF) simulation system for model and high-power rockets. The simulator implements Barrowman aerodynamics, full 3D dynamics, and integrates with real motor databases to provide accurate flight predictions.

## Architecture

### Core Components

1. **Flight Solver (`flight_solver.py`)**
   - 6-DOF dynamics with quaternion-based orientation
   - RK4 numerical integration
   - Real-time mass and inertia updates during motor burn
   - Parachute deployment logic (apogee, altitude, time triggers)

2. **Aerodynamics (`openrocket_aero.py`)**
   - Barrowman method for stability calculations
   - Mach-dependent center of pressure (CP) calculations
   - Drag coefficient estimation
   - Lift and moment calculations

3. **Rocket Model (`rocket_model.py`)**
   - Component-based architecture
   - Mass, center of gravity (CG), and inertia calculations
   - Reference values (length, diameter, area)

4. **Motor Model (`motor_model.py`)**
   - Thrust curve interpolation
   - Propellant mass burn tracking
   - Case mass and propellant CG tracking

5. **Environment (`environment.py`)**
   - International Standard Atmosphere (ISA) model
   - Density, pressure, temperature calculations
   - Wind models (static and dynamic)

### Physics Implementation

#### 6-DOF Dynamics

The simulator solves the full 6-DOF equations of motion:

**Translational Dynamics:**
```
F = m·a
```

Where forces include:
- Thrust (from motor)
- Drag (aerodynamic)
- Lift (aerodynamic)
- Gravity
- Wind effects

**Rotational Dynamics:**
```
τ = I·α + ω × (I·ω)
```

Where moments include:
- Aerodynamic moments (pitch, yaw, roll)
- Thrust misalignment effects
- Parachute deployment effects

#### Barrowman Aerodynamics

The simulator uses the Barrowman method for stability analysis:

- **Center of Pressure (CP)**: Calculated from rocket geometry and Mach number
- **Center of Gravity (CG)**: Calculated from component masses and positions
- **Static Margin**: `(CP - CG) / reference_diameter` in calibers

The CP position varies with Mach number, accounting for compressibility effects.

#### Numerical Integration

The RK4 (Runge-Kutta 4th order) method is used for numerical stability:

```
k1 = f(t, y)
k2 = f(t + h/2, y + h·k1/2)
k3 = f(t + h/2, y + h·k2/2)
k4 = f(t + h, y + h·k3)
y_new = y + (h/6)·(k1 + 2·k2 + 2·k3 + k4)
```

This provides 4th-order accuracy while maintaining numerical stability.

### Mass and Inertia Modeling

The simulator tracks mass and inertia throughout the flight:

- **Initial Mass**: Dry mass + motor mass
- **During Burn**: Mass decreases as propellant burns
- **After Burnout**: Only dry mass + motor case remains
- **CG Shift**: As propellant burns, CG shifts forward

Inertia is recalculated at each time step based on current mass distribution.

### Parachute Deployment

Three deployment triggers are supported:

1. **APOGEE**: Deploys at maximum altitude
2. **ALTITUDE**: Deploys at specified altitude AGL
3. **TIME**: Deploys at specified time after launch

Each parachute has:
- Diameter and drag coefficient
- Deployment delay (inflation time)
- Descent velocity calculation

## AI Rocket Optimizer

### Architecture

The Smart Optimizer (`smart_optimizer.py`) uses a multi-phase search strategy:

1. **Phase 1: Coarse Sampling**
   - Samples motors across the impulse range
   - Quickly identifies viable design space

2. **Phase 2: Fine Search**
   - Dense search around successful designs
   - Cost-optimized exploration

3. **Adaptive Learning**
   - Tracks successful designs
   - Narrows search space based on results

### Impulse Estimation

The optimizer estimates required impulse based on:

- Target altitude
- Rocket mass (payload + structure + motor)
- Drag penalties (diameter, L/D ratio)
- Empirical altitude/impulse ratios

**Calibrated Ratios:**
- Light rockets (<5kg): 1.2 m/N·s
- Medium rockets (5-15kg): 0.7 m/N·s
- Heavy rockets (15-30kg): 0.55 m/N·s
- Very heavy (>30kg): 0.45 m/N·s

### Cost Optimization

The optimizer minimizes total cost:

- **Motor cost**: Based on impulse class
- **Body tube cost**: Scales with diameter and length
- **Fin cost**: Based on material and size
- **Recovery cost**: Parachutes and hardware
- **Hardware cost**: Epoxy, fasteners, paint, etc.

### Tube Selection

The optimizer selects appropriate body tubes:

- **Motor fit**: Tube ID must accommodate motor diameter
- **Payload fit**: For heavy payloads, calculates minimum tube size
- **Cost optimization**: Prefers smaller tubes when possible

## Flight Analysis

### Stability Derivatives

The `FlightAnalyzer` computes stability derivatives from flight data:

**Longitudinal:**
- `C_L_alpha`: Lift coefficient derivative with angle of attack
- `C_m_alpha`: Pitching moment derivative
- `C_D_alpha`: Drag coefficient derivative
- `C_m_q`: Pitch damping derivative

**Lateral-Directional:**
- `C_Y_beta`: Side force derivative
- `C_l_beta`: Roll moment derivative
- `C_n_beta`: Yaw moment derivative
- `C_l_p`, `C_n_r`, `C_n_p`: Rate derivatives

### Dynamic Analysis

**First-Order Terms:**
- Linear acceleration components
- Angular acceleration components
- Velocity rates
- Angular rate rates

**Second-Order Terms:**
- Linear jerk (rate of acceleration change)
- Angular jerk (rate of angular acceleration change)
- AOA and sideslip rates and accelerations

### Energy Analysis

- **Kinetic Energy**: `KE = 0.5·m·v²`
- **Potential Energy**: `PE = m·g·h`
- **Total Energy**: `E = KE + PE`

## Validation

### Calisto Rocket

The simulator is validated against the Calisto rocket:

- **Dry Mass**: 14.426 kg
- **Motor**: Cesaroni M1670 (6026 N·s)
- **Expected Apogee**: ~3350 m
- **Actual Result**: Matches within 5%

### Motor Database

The simulator uses real motor data from ThrustCurve.org:

- Thousands of motors available
- Real thrust curves
- Accurate mass and impulse data
- Local caching for offline use

## Performance

### Simulation Speed

- Typical flight (200s): ~1-2 seconds
- Time step: 0.01s (configurable)
- RK4 integration: 4 function evaluations per step

### Accuracy

- Altitude prediction: ±5% for typical rockets
- Stability margin: ±0.1 calibers
- Apogee timing: ±0.5 seconds

## Limitations

1. **Aerodynamics**: Barrowman method is approximate; CFD would be more accurate
2. **Wind**: Simplified wind models; real weather data not integrated
3. **Structural Dynamics**: Assumes rigid body; no flex or vibration
4. **Recovery**: Simplified parachute model; no inflation dynamics

## Future Enhancements

1. **CFD Integration**: More accurate aerodynamics
2. **Weather Integration**: Real-time wind and atmospheric data
3. **Structural Analysis**: Flex and vibration modeling
4. **Advanced Recovery**: Detailed parachute inflation dynamics
5. **Multi-Stage**: Support for multi-stage rockets

## References

- Barrowman, J. S. (1967). "The Practical Calculation of the Aerodynamic Characteristics of Slender Finned Vehicles." NASA Technical Report.
- See `docs/sources/barrowman_equation.tex` for detailed Barrowman equations.
- ThrustCurve.org for motor database.

## License

Part of the Elodin project.

