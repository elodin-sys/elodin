# BDX RC Jet Turbine Simulation: Technical Whitepaper

## Executive Summary

This document describes the technical approach for developing an Elodin simulation of the **Elite Aerosports BDX**, a high-performance RC jet turbine aircraft. The simulation targets a "simple but accurate" fidelity level—sufficient for control system development, pilot training scenarios, and flight dynamics visualization—without the complexity of a full system identification or high-fidelity CFD-derived model.

The approach draws from established small-UAV aerodynamic modeling literature while adapting patterns from existing Elodin examples (rocket, drone) to create a maintainable, extensible simulation.

---

## 1. Aircraft Overview

### 1.1 BDX Specifications

The Elite Aerosports BDX is a modern interpretation of the legendary BD-5J, designed for RC sport jet flying and UAS applications.

| Parameter | RC Sport Version | UAS Version |
|-----------|------------------|-------------|
| Wingspan | 2.65 m (104 in) | 2.65 m |
| Length | 2.8 m (110 in) | 2.8 m |
| Empty Weight | 18.1–19 kg (40–42 lb) | 19 kg (42 lb) |
| Max Weight | ~22 kg | 56.7 kg (125 lb) |
| Recommended Thrust | 180–210 N | 210–320 N |
| Max Speed | ~200 kt | >200 kt |
| Maneuverability | High (aerobatic) | Up to 18 g |
| Endurance | 15–20 min typical | ~50 min cruise |

**Sources**: Elite Aerosports product page, AIR-RC specifications

### 1.2 Airfoil Heritage

The BDX descends from the BD-5J design which used:
- **Root**: NACA 64-212
- **Tip**: NACA 64-218

For our simplified model, we treat the wing as having uniform NACA 64-212 characteristics, which provides:
- Good lift-to-drag ratio at cruise
- Well-documented 2D section data
- Reasonable low-Reynolds behavior for RC scale

### 1.3 Estimated Geometric Parameters

For the simulation, we derive or estimate:

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Wing Area | S | 0.75 m² | Estimated from planform |
| Mean Aerodynamic Chord | c̄ | 0.30 m | S/b with taper |
| Wingspan | b | 2.65 m | From spec |
| Horizontal Tail Area | S_h | 0.15 m² | ~20% of wing area |
| Vertical Tail Area | S_v | 0.10 m² | Typical for jets |
| Tail Moment Arm | l_t | 1.2 m | CG to tail AC |

---

## 2. Coordinate System and State Representation

### 2.1 Reference Frames

We adopt the standard aerospace body-fixed frame:
- **X-axis (forward)**: Along fuselage, positive toward nose
- **Y-axis (right)**: Out right wing, positive starboard  
- **Z-axis (down)**: Positive downward (completing right-hand system)

**Note**: Elodin uses Z-up world coordinates, so gravity acts in the -Z direction. Body frame transformations must account for this convention.

### 2.2 State Vector

The simulation state comprises:

**Rigid Body States** (managed by `el.Body`):
- Position: (x, y, z) in NED world frame
- Velocity: (u, v, w) in body frame or (Vx, Vy, Vz) in world frame
- Attitude: Quaternion q = (qw, qx, qy, qz)
- Angular rates: (p, q, r) in body frame (roll, pitch, yaw rates)

**Engine States**:
- Spool speed: n ∈ [0, 1] (normalized)
- Optional: Exhaust gas temperature (for thermal dynamics)

**Actuator States**:
- Elevator deflection: δ_e
- Aileron deflection: δ_a  
- Rudder deflection: δ_r

**Aerodynamic States** (computed each tick):
- Angle of attack: α
- Sideslip angle: β
- Dynamic pressure: q̄
- Mach number: M

---

## 3. Aerodynamic Model

### 3.1 Design Philosophy

We implement a **component-based polynomial aerodynamic model** inspired by:
1. The Skywalker X8 modeling paper (Springer, 2025)
2. Selig's full-envelope UAV aerodynamics framework
3. The high-speed mini-UAV paper (IJAST)

This approach:
- Expresses forces/moments as explicit functions of flight state and control inputs
- Remains numerically stable and computationally efficient
- Can be easily extended with wind tunnel or flight test data

### 3.2 Force and Moment Equations

#### Longitudinal Axis (Lift, Drag, Pitching Moment)

```
C_L = C_L0 + C_Lα·α + C_Lq·(q·c̄)/(2V) + C_Lδe·δ_e

C_D = C_D0 + k·C_L² + C_Dδe·|δ_e|

C_m = C_m0 + C_mα·α + C_mq·(q·c̄)/(2V) + C_mδe·δ_e
```

Where:
- α = angle of attack (rad)
- q = pitch rate (rad/s)
- V = true airspeed (m/s)
- δ_e = elevator deflection (rad)
- k = 1/(π·e·AR) = induced drag factor
- AR = b²/S = aspect ratio
- e ≈ 0.8 = Oswald efficiency factor

#### Lateral-Directional Axis (Side Force, Rolling Moment, Yawing Moment)

```
C_Y = C_Yβ·β + C_Yp·(p·b)/(2V) + C_Yr·(r·b)/(2V) + C_Yδa·δ_a + C_Yδr·δ_r

C_l = C_lβ·β + C_lp·(p·b)/(2V) + C_lr·(r·b)/(2V) + C_lδa·δ_a + C_lδr·δ_r

C_n = C_nβ·β + C_np·(p·b)/(2V) + C_nr·(r·b)/(2V) + C_nδa·δ_a + C_nδr·δ_r
```

Where:
- β = sideslip angle (rad)
- p, r = roll and yaw rates (rad/s)
- δ_a, δ_r = aileron and rudder deflections (rad)

### 3.3 Baseline Aerodynamic Coefficients

These values are estimated for a clean, high-speed RC jet configuration at low angles of attack. They can be refined with XFLR5 analysis or flight test data.

#### Longitudinal Derivatives

| Coefficient | Value | Units | Notes |
|-------------|-------|-------|-------|
| C_L0 | 0.15 | - | Zero-α lift (slight camber) |
| C_Lα | 5.5 | /rad | Typical for AR ~9 wing |
| C_Lq | 8.0 | /rad | Pitch damping (tail effect) |
| C_Lδe | 0.4 | /rad | Elevator effectiveness |
| C_D0 | 0.025 | - | Parasite drag (clean jet) |
| C_Dδe | 0.02 | /rad | Control surface drag |
| k | 0.045 | - | Induced drag factor |
| C_m0 | 0.0 | - | Trim moment (assume trimmed) |
| C_mα | -1.2 | /rad | Static stability (~10% MAC) |
| C_mq | -20.0 | /rad | Pitch damping |
| C_mδe | -1.5 | /rad | Elevator control power |

#### Lateral-Directional Derivatives

| Coefficient | Value | Units | Notes |
|-------------|-------|-------|-------|
| C_Yβ | -0.5 | /rad | Side force due to sideslip |
| C_Yp | 0.0 | /rad | Negligible for most aircraft |
| C_Yr | 0.3 | /rad | Yaw coupling |
| C_Yδa | 0.0 | /rad | Minimal aileron side force |
| C_Yδr | 0.15 | /rad | Rudder side force |
| C_lβ | -0.08 | /rad | Dihedral effect |
| C_lp | -0.5 | /rad | Roll damping |
| C_lr | 0.1 | /rad | Roll due to yaw |
| C_lδa | 0.15 | /rad | Aileron control power |
| C_lδr | 0.01 | /rad | Rudder-roll coupling |
| C_nβ | 0.1 | /rad | Weathercock stability |
| C_np | -0.03 | /rad | Adverse yaw |
| C_nr | -0.15 | /rad | Yaw damping |
| C_nδa | -0.01 | /rad | Aileron adverse yaw |
| C_nδr | -0.1 | /rad | Rudder control power |

### 3.4 Stall Modeling

For the initial model, we implement a simple piecewise stall:

```python
def stall_factor(alpha, alpha_stall=15°):
    """Reduce lift effectiveness beyond stall angle."""
    alpha_deg = abs(degrees(alpha))
    if alpha_deg < alpha_stall:
        return 1.0
    elif alpha_deg < alpha_stall + 10:
        # Linear rolloff
        return 1.0 - 0.5 * (alpha_deg - alpha_stall) / 10
    else:
        return 0.5  # Post-stall plateau
```

This can be extended to include post-stall lift and drag curves if aerobatic flight is required.

### 3.5 Converting Coefficients to Forces

```python
# Dynamic pressure
q_bar = 0.5 * rho * V²

# Aerodynamic forces in body frame
L = q_bar * S * C_L  # Lift (perpendicular to velocity)
D = q_bar * S * C_D  # Drag (opposite to velocity)  
Y = q_bar * S * C_Y  # Side force

# Aerodynamic moments in body frame
L_moment = q_bar * S * b * C_l     # Rolling moment
M_moment = q_bar * S * c_bar * C_m  # Pitching moment
N_moment = q_bar * S * b * C_n     # Yawing moment

# Transform L, D from wind frame to body frame
F_x = -D * cos(α) + L * sin(α)
F_y = Y
F_z = -D * sin(α) - L * cos(α)
```

---

## 4. Propulsion Model

### 4.1 Turbine Dynamics

The BDX uses JetCat-class turbines (P180–P220 range). We model the engine with a first-order spool response:

```
dn/dt = (n_cmd - n) / τ_spool
```

Where:
- n = normalized spool speed (0 to 1)
- n_cmd = throttle command (0 to 1)
- τ_spool = spool time constant (0.4–0.8 s typical)

### 4.2 Static Thrust Map

Thrust is modeled as a quadratic function of spool speed:

```
T(n) = T_max * (a₁·n + a₂·n²)
```

With typical values:
- T_max = 200 N (for P200-class turbine)
- a₁ = 0.2 (linear term)
- a₂ = 0.8 (quadratic term)

This captures the nonlinear throttle response where most thrust comes at high RPM.

### 4.3 Thrust Lapse with Altitude and Speed

For completeness, we include atmospheric effects:

```
T_actual = T(n) * (ρ/ρ₀) * f(M)
```

Where f(M) is a Mach correction factor (approximately 1.0 for M < 0.3, decreasing slightly at higher Mach).

### 4.4 Fuel Consumption (Optional)

For endurance simulation:
```
m_dot_fuel = TSFC * T
```
Where TSFC ≈ 0.12 kg/(N·h) for small turbojets.

---

## 5. Control Surface Actuator Model

### 5.1 First-Order Dynamics with Rate Limiting

Each control surface is modeled as a rate-limited first-order system:

```python
def actuator_dynamics(delta, delta_cmd, tau, delta_dot_max, delta_max, dt):
    # First-order response
    delta_error = delta_cmd - delta
    delta_dot = delta_error / tau
    
    # Rate limiting
    delta_dot = clip(delta_dot, -delta_dot_max, delta_dot_max)
    
    # Integration
    delta_new = delta + delta_dot * dt
    
    # Position limiting
    return clip(delta_new, -delta_max, delta_max)
```

### 5.2 Actuator Parameters

| Surface | τ (s) | δ_dot_max (°/s) | δ_max (°) |
|---------|-------|-----------------|-----------|
| Elevator | 0.05 | 400 | ±25 |
| Aileron | 0.05 | 400 | ±25 |
| Rudder | 0.06 | 350 | ±30 |

---

## 6. Atmosphere Model

### 6.1 International Standard Atmosphere (ISA)

We use a lookup table or analytic ISA model (same as rocket example):

```python
def atmosphere(altitude):
    # Troposphere (h < 11 km)
    T = 288.15 - 0.0065 * altitude  # Temperature (K)
    p = 101325 * (T / 288.15) ** 5.2561  # Pressure (Pa)
    rho = p / (287.05 * T)  # Density (kg/m³)
    a = sqrt(1.4 * 287.05 * T)  # Speed of sound (m/s)
    return T, p, rho, a
```

### 6.2 Wind Model

Wind is modeled as a constant or turbulent vector field:

```python
Wind = el.Component("wind", shape=(3,))  # [Wx, Wy, Wz] in world frame
```

Airspeed is computed as:
```
V_air = V_ground - W
```

---

## 7. Inertia Estimation

### 7.1 Mass Properties

For the RC BDX at ~19 kg:

| Property | Value | Notes |
|----------|-------|-------|
| Mass | 19.0 kg | Typical sport configuration |
| I_xx (roll) | 0.8 kg·m² | Narrow fuselage, spread mass |
| I_yy (pitch) | 2.5 kg·m² | Long moment arm fore/aft |
| I_zz (yaw) | 3.0 kg·m² | Sum of roll + pitch contributions |
| I_xz (cross) | 0.1 kg·m² | Small due to symmetry |

### 7.2 Scaling Approach

These values are estimated using:
1. Component build-up (wing, fuselage, tail, engine)
2. Comparison with Rascal 110 and Multiplex Mentor scaled by (m·L²)
3. Typical non-dimensional inertia ratios for RC jets

---

## 8. Implementation Architecture

### 8.1 Module Structure

Following the drone example pattern:

```
examples/rc-jet/
├── main.py              # Entry point, world setup, run loop
├── config.py            # Aircraft configuration dataclass
├── aero.py              # Aerodynamic model and forces
├── propulsion.py        # Turbine dynamics and thrust
├── actuators.py         # Control surface servo models
├── control.py           # Flight control systems (optional)
├── sim.py               # System composition
└── BDX_Simulation_Whitepaper.md  # This document
```

### 8.2 Component Definitions

```python
# aero.py
AeroCoefs = ty.Annotated[
    jax.Array,
    el.Component(
        "aero_coefs",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "CL,CD,CY,Cl,Cm,Cn"},
    ),
]

AeroForce = ty.Annotated[
    el.SpatialForce,
    el.Component("aero_force", el.ComponentType.SpatialMotionF64),
]

AngleOfAttack = ty.Annotated[jax.Array, el.Component("alpha", el.ComponentType.F64)]
Sideslip = ty.Annotated[jax.Array, el.Component("beta", el.ComponentType.F64)]
```

```python
# propulsion.py
SpoolSpeed = ty.Annotated[
    jax.Array,
    el.Component("spool_speed", el.ComponentType.F64, metadata={"priority": 50}),
]

Thrust = ty.Annotated[
    jax.Array,
    el.Component("thrust", el.ComponentType.F64, metadata={"priority": 51}),
]
```

```python
# actuators.py
ControlSurfaces = ty.Annotated[
    jax.Array,
    el.Component(
        "control_surfaces",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "elevator,aileron,rudder"},
    ),
]

ControlCommands = ty.Annotated[
    jax.Array,
    el.Component(
        "control_commands",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "elevator,aileron,rudder,throttle"},
    ),
]
```

### 8.3 Archetype Definition

```python
@el.dataclass
class BDXJet(el.Archetype):
    # Aerodynamics
    alpha: AngleOfAttack = field(default_factory=lambda: jnp.float64(0.0))
    beta: Sideslip = field(default_factory=lambda: jnp.float64(0.0))
    aero_coefs: AeroCoefs = field(default_factory=lambda: jnp.zeros(6))
    aero_force: AeroForce = field(default_factory=el.SpatialForce)
    dynamic_pressure: DynamicPressure = field(default_factory=lambda: jnp.float64(0.0))
    mach: Mach = field(default_factory=lambda: jnp.float64(0.0))
    
    # Propulsion
    spool_speed: SpoolSpeed = field(default_factory=lambda: jnp.float64(0.2))  # Idle
    thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))
    
    # Actuators
    control_surfaces: ControlSurfaces = field(default_factory=lambda: jnp.zeros(3))
    control_commands: ControlCommands = field(default_factory=lambda: jnp.zeros(4))
    
    # Wind
    wind: Wind = field(default_factory=lambda: jnp.zeros(3))
```

### 8.4 System Graph

```python
# sim.py
def system():
    # Non-effector systems (compute derived quantities)
    non_effectors = (
        compute_airspeed
        | compute_aero_angles
        | compute_dynamic_pressure
        | actuator_dynamics
        | spool_dynamics
        | compute_aero_coefs
        | compute_aero_forces
        | compute_thrust
    )
    
    # Effector systems (apply forces to body)
    effectors = gravity | apply_thrust | apply_aero_forces
    
    # Compose with 6-DOF integrator
    return non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.Rk4)
```

---

## 9. Validation Strategy

### 9.1 Sanity Checks

1. **Trim verification**: Aircraft should maintain level flight at ~40 m/s with ~70% throttle
2. **Static stability**: Positive C_mα (nose-down with positive α perturbation)
3. **Phugoid mode**: Period ~10-15 seconds, lightly damped
4. **Dutch roll**: Period ~2-3 seconds, adequately damped with yaw damper
5. **Roll response**: Time to 60° bank < 1 second at full aileron

### 9.2 Comparison Data

- Compare trim throttle and speed to pilot experience
- Validate stall speed (~25 m/s at 19 kg, sea level)
- Check climb rate at full power (~15-20 m/s)

### 9.3 Tuning Process

1. Start with estimated coefficients
2. Run trim analysis to find level flight condition
3. Apply step inputs to each control and observe response
4. Adjust derivatives to match expected aircraft behavior
5. Optionally correlate with XFLR5 predictions

---

## 10. Future Extensions

### 10.1 Enhanced Aerodynamics
- Full α-sweep lookup tables from XFLR5
- Post-stall modeling for spin simulation
- Ground effect during takeoff/landing
- Compressibility corrections for high-speed flight

### 10.2 Control Systems
- Autopilot implementation (altitude hold, heading hold)
- Rate-based flight control similar to drone example
- Auto-throttle for speed hold

### 10.3 Mission Scenarios
- Takeoff and landing
- Circuit patterns
- Aerobatic maneuvers
- Failure modes (engine-out, surface jam)

### 10.4 Hardware-in-the-Loop
- External control via Impeller2 protocol
- Real RC transmitter input
- Connection to actual flight controller firmware

---

## 11. References

1. **Elite Aerosports BDX Product Page** - Aircraft specifications
2. **AIR-RC BDX Listing** - Dimensions and weight data
3. **Selig, M.S. (2010)** - "Modeling Full-Envelope Aerodynamics of Small UAVs in Realtime", AIAA 2010-7635
4. **Springer (2025)** - "Modeling and identification of a small fixed-wing UAV using estimated aerodynamic angles", CEAS Aeronautical Journal
5. **IJAST** - "Modeling and Control of a Fixed-Wing High-Speed Mini-UAV"
6. **DGLR (2022)** - "Identification of a Turbojet Engine using Multi-Sine Inputs"
7. **BD-5 Wing Reprofile Specs** - Airfoil heritage data
8. **JSBSim/FlightGear Rascal 110** - RC aircraft FDM reference
9. **MathWorks Simulink Drone Reference App** - RC plane modeling patterns

---

## Appendix A: Quick-Start Configuration

```python
# config.py

from dataclasses import dataclass
import numpy as np

@dataclass
class BDXConfig:
    # Geometry
    wingspan: float = 2.65  # m
    wing_area: float = 0.75  # m²
    mean_chord: float = 0.30  # m
    
    # Mass properties
    mass: float = 19.0  # kg
    Ixx: float = 0.8    # kg·m²
    Iyy: float = 2.5    # kg·m²
    Izz: float = 3.0    # kg·m²
    Ixz: float = 0.1    # kg·m²
    
    # Propulsion
    max_thrust: float = 200.0  # N
    spool_tau: float = 0.5     # s
    
    # Actuators
    servo_tau: float = 0.05          # s
    max_deflection: float = 25.0     # degrees
    max_deflection_rate: float = 400 # deg/s
    
    # Initial conditions
    initial_speed: float = 40.0      # m/s
    initial_altitude: float = 100.0  # m
    
    # Simulation
    dt: float = 1/120  # 120 Hz
```

---

## Appendix B: Coefficient Estimation from XFLR5

If you wish to refine the aerodynamic coefficients using XFLR5:

1. **Create geometry**: Build simplified BDX planform (wing, horizontal tail, vertical tail)
2. **Set airfoils**: NACA 64-212 for wing, NACA 0009 for tails
3. **Run VLM analysis**: Sweep α from -5° to 20° at V = 40 m/s
4. **Extract derivatives**: Fit linear slopes to CL(α), Cm(α), etc.
5. **Add control surfaces**: Define elevator, aileron, rudder geometry
6. **Run control sweeps**: Extract C_Lδe, C_mδe, C_lδa, C_nδr, etc.
7. **Export to Python**: Create lookup tables or polynomial fits

---

*Document Version: 1.0*  
*Last Updated: November 2025*  
*Authors: Elodin Development Team*

