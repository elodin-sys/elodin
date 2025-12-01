# Rocket Example

A comprehensive six-degree-of-freedom rocket simulation featuring realistic aerodynamics, thrust curves, wind effects, and PID control systems. This example demonstrates advanced Elodin capabilities including component systems, aerodynamic lookup tables, and EQL (Elodin Query Language) for computing derived values.

## Features

- **6-DOF Physics**: Full six-degree-of-freedom dynamics with spatial forces and torques
- **Aerodynamic Lookup Tables**: Interpolated aerodynamic coefficients based on Mach number, angle of attack, and fin deflection
- **Thrust Curve**: Realistic motor thrust profile over time
- **Wind Effects**: Atmospheric wind modeling affecting aerodynamic forces
- **PID Control**: Pitch control system with PID controller for attitude stabilization
- **EQL Derived Values**: Compute aerodynamic angles and velocity magnitude using EQL queries instead of streaming separate components

## Running the Example

```bash
elodin editor main.py
```

The simulation runs for 5000 ticks by default. The schematic displays:
- A 3D viewport showing the rocket with compass reference
- Control system graphs (trim control, fin deflection, aerodynamic coefficients)
- EQL-derived aerodynamic angles and velocity magnitude

## Key Components

### Rocket Archetype

The `Rocket` archetype includes:

- **`angle_of_attack`**: Computed angle of attack in degrees
- **`aero_coefs`**: Aerodynamic coefficients (Cl, CnR, CmR, CA, CZR, CYR)
- **`center_of_gravity`**: CG position for moment calculations
- **`mach`**: Current Mach number
- **`dynamic_pressure`**: Dynamic pressure for force calculations
- **`aero_force`**: Computed aerodynamic forces and moments
- **`wind`**: Wind velocity vector
- **`v_body`**: Velocity in body frame `[u, v, w]` (for EQL queries)
- **`fin_deflect`**: Fin deflection angle
- **`fin_control`**: Fin control command
- **`fin_control_trim`**: External trim control (can be set via database)
- **`thrust`**: Current motor thrust
- **PID State**: Pitch PID controller state and parameters

## Systems

### Aerodynamics

- **`mach`**: Computes Mach number and dynamic pressure from altitude and velocity
- **`angle_of_attack`**: Calculates angle of attack from body-frame velocity
- **`aero_coefs`**: Interpolates aerodynamic coefficients from lookup tables
- **`aero_forces`**: Converts coefficients to forces and moments
- **`apply_aero_forces`**: Applies aerodynamic forces in world frame

### Propulsion

- **`thrust`**: Interpolates thrust from motor curve based on simulation time
- **`apply_thrust`**: Applies thrust force along body x-axis

### Control Systems

- **`v_rel_accel`**: Computes acceleration relative to velocity direction
- **`v_rel_accel_filtered`**: Low-pass filters acceleration for control
- **`pitch_pid_state`**: Updates PID controller state
- **`pitch_pid_control`**: Computes control output from PID state
- **`fin_control`**: Applies control to fin deflection with Mach scaling

### Physics

- **`gravity`**: Applies gravitational force
- **`compute_v_body`**: Transforms world velocity to body frame (for EQL queries)

## EQL Derived Values

This example demonstrates using EQL (Elodin Query Language) to compute derived aerodynamic values **without** streaming them as separate components. This reduces telemetry bandwidth by computing values on-demand in the database.

### Body Frame Velocity Component

The `v_body` component stores velocity in the body frame as `[u, v, w]`:
- `u` = velocity along body x-axis (forward, aligned with thrust)
- `v` = velocity along body y-axis (sideways)
- `w` = velocity along body z-axis (upward)

This component is computed by the `compute_v_body` system, which transforms world velocity (relative to wind) to body frame using the rocket's orientation.

### EQL Query Plots

The schematic includes three `query_plot` panels that use EQL to compute derived values:

#### Angle of Attack (Alpha)

```eql
(rocket.v_body[2] * -1.0).atan2(rocket.v_body[0].clip(0.000000000001, 999999)).degrees()
```

**Python equivalent:**
```python
alpha_deg = np.degrees(np.arctan2(-w, np.clip(u, 1e-12, None)))
```

**Explanation:**
- `rocket.v_body[2]` is `w` (z-component)
- `rocket.v_body[2] * -1.0` negates it (EQL doesn't support unary minus)
- `rocket.v_body[0].clip(...)` clips `u` to prevent division by zero
- `.atan2(...)` computes the two-argument arctangent
- `.degrees()` converts from radians to degrees

#### Velocity Magnitude

```eql
rocket.v_body.norm()
```

Computes the Euclidean norm (magnitude) of the velocity vector using the `norm()` formula, which expands to `sqrt(u² + v² + w²)`.

### EQL Formulas Used

This example uses the following EQL formulas:

1. **`atan2(y, x)`** - Two-argument arctangent
   - Syntax: `y.atan2(x)`
   - Maps to PostgreSQL `atan2(y, x)`

2. **`degrees(radians)`** - Convert radians to degrees
   - Syntax: `expr.degrees()`
   - Maps to PostgreSQL `degrees(radians)`

3. **`clip(value, min, max)`** - Clamp value between min and max
   - Syntax: `value.clip(min, max)`
   - Maps to PostgreSQL `GREATEST(min, LEAST(value, max))`

4. **`sqrt(x)`** - Square root
   - Syntax: `expr.sqrt()`
   - Maps to PostgreSQL `sqrt(x)`

5. **`norm()`** - Euclidean norm of a vector
   - Syntax: `vector.norm()`
   - Expands to `sqrt(∑ elem²)`

### Benefits of EQL Derived Values

By using EQL to compute derived values:

1. **Reduced Telemetry**: Only stream the base `v_body` component, not derived angles
2. **Flexibility**: Compute different derived values without code changes
3. **Real-time**: Computations happen in the database/query layer
4. **Consistency**: Same formulas used for analysis and visualization

## Aerodynamic Model

The aerodynamic model uses lookup tables with three independent variables:
- **Mach number**: 0.1, 0.5, 0.9
- **Angle of attack**: 0°, 5°, 10°, 15°
- **Fin deflection**: -40°, -20°, 0°, 20°, 40°

Coefficients are interpolated using nearest-neighbor interpolation for Mach and angle of attack, with linear interpolation for fin deflection.

The model computes:
- **Cl**: Roll moment coefficient
- **CnR**: Yaw moment coefficient  
- **CmR**: Pitch moment coefficient
- **CA**: Axial force coefficient
- **CZR**: Normal force coefficient
- **CYR**: Side force coefficient

## Control System

The pitch control system uses a PID controller:
- **Proportional gain**: 1.1
- **Integral gain**: 0.8
- **Derivative gain**: 3.8

The controller tracks acceleration setpoints and adjusts fin deflection to maintain desired pitch attitude. Control effectiveness is scaled by Mach number to account for changing aerodynamic characteristics.

## External Control

The `fin_control_trim` component can be set externally via the database, allowing real-time control of the rocket during simulation. This demonstrates Elodin's capability for hardware-in-the-loop (HITL) testing.

## Customer Use Case

This example directly addresses the customer request for computing aerodynamic angles from velocity vectors in Elodin Editor using EQL, eliminating the need to stream both sensor data AND derived results as components. The EQL queries demonstrate how to reduce telemetry bandwidth by computing derived values on-demand rather than streaming them separately.
