# BDX RC Jet Turbine Simulation

A high-fidelity 6-DOF simulation of the Elite Aerosports BDX RC jet turbine aircraft, implementing polynomial aerodynamic models, turbine propulsion dynamics, and control surface servos.

## Overview

This simulation implements the complete flight dynamics of a BDX RC jet following the methodology described in `BDX_Simulation_Whitepaper.md`. The model includes:

- **Aerodynamics**: Polynomial stability derivative model with 20+ coefficients
- **Propulsion**: First-order turbine spool dynamics with thrust mapping
- **Actuators**: Rate-limited servo dynamics for control surfaces
- **6-DOF Dynamics**: RK4 integration for accurate trajectory prediction

## Quick Start

### Prerequisites

Follow the [main README](../../README.md) for Elodin installation.

### Running the Simulation

**Basic command-line run:**
```bash
cd examples/rc-jet
python main.py
```

**With 3D visualization (recommended):**
```bash
elodin editor main.py
```

**Custom parameters:**
```bash
python main.py --altitude 200 --speed 50 --time 120
```

### Command-Line Options

- `--altitude <meters>`: Set initial altitude (default: 100 m)
- `--speed <m/s>`: Set initial airspeed (default: 40 m/s)
- `--time <seconds>`: Set simulation duration (default: 60 s)
- `--build`: Compile mode (faster for batch runs)
- `--save`: Save telemetry to `bdx_telemetry.arrow`

## Flight Plan

The default flight plan (`control.py`) demonstrates various maneuvers:

| Time (s) | Maneuver | Description |
|----------|----------|-------------|
| 0-5 | Steady cruise | Level flight at 70% throttle |
| 5-10 | Pitch up | 5° elevator deflection |
| 10-15 | Roll maneuver | 10° aileron deflection |
| 15-20 | Coordinated turn | Combined aileron + rudder |
| 20-25 | Return to level | Recovery to wings-level |
| 25+ | Steady cruise | Continue level flight |

## Validation

### Expected Behavior

The simulation has been designed to match these physical characteristics:

#### Trim Conditions (Section 9.1 of whitepaper)
- **Cruise speed**: ~40 m/s (78 kt) at 70% throttle
- **Stall speed**: ~25 m/s (49 kt) at 19 kg
- **Climb rate**: 15-20 m/s at full power

#### Stability Characteristics
- **Static stability**: Positive C_mα (nose-down moment with positive α)
- **Phugoid mode**: Period ~10-15 seconds, lightly damped
- **Dutch roll**: Period ~2-3 seconds, adequately damped
- **Roll response**: Time to 60° bank < 1 second at full aileron

### Validation Tests

Run these checks to verify simulation fidelity:

#### 1. Trim Verification
```bash
python main.py --time 30
```
Observe in telemetry:
- Altitude should remain approximately constant (~100m ±5m)
- Angle of attack should stabilize near 0-3°
- Throttle at ~0.7 maintains level flight

#### 2. Stall Behavior
Modify `initial_speed` in `config.py` to 20 m/s and observe:
- Increased angle of attack
- Loss of altitude
- Reduced control effectiveness

#### 3. Roll Response Test
Watch the 10-15s time window:
- Aircraft should roll right with positive aileron
- Roll rate should be responsive
- Recovery should be stable

#### 4. Coordinated Turn
Watch the 15-20s time window:
- Bank angle increases with aileron
- Yaw rate coordinates with rudder
- Altitude may decrease slightly (normal in turns)

### Telemetry Analysis

For detailed analysis, run with `--build --save`:

```bash
python main.py --build --save --time 120
```

This generates `bdx_telemetry.arrow` which can be analyzed with:

```python
import polars as pl

# Load telemetry
df = pl.read_ipc("bdx_telemetry.arrow")

# Analyze altitude stability
print(df.select(["bdx.world_pos", "bdx.alpha", "bdx.thrust"]).describe())

# Plot lift coefficient vs angle of attack
import matplotlib.pyplot as plt
plt.plot(df["bdx.alpha"], df["bdx.aero_coefs"].arr.get(0))
plt.xlabel("Angle of Attack (rad)")
plt.ylabel("Lift Coefficient (CL)")
plt.show()
```

## Configuration

All aircraft parameters are in `config.py`:

### Aircraft Geometry
```python
wingspan = 2.65          # m
wing_area = 0.75         # m²
mean_chord = 0.30        # m
```

### Mass Properties
```python
mass = 19.0              # kg
Ixx = 0.8                # kg·m² (roll inertia)
Iyy = 2.5                # kg·m² (pitch inertia)
Izz = 3.0                # kg·m² (yaw inertia)
```

### Aerodynamic Coefficients
All stability derivatives are in `AeroCoefficients` dataclass:
```python
C_Lalpha = 5.5           # Lift curve slope (/rad)
C_malpha = -1.2          # Pitch stability (/rad)
C_lp = -0.5              # Roll damping (/rad)
# ... and 17 more coefficients
```

### Tuning

To adjust flight characteristics:

1. **Increase stability**: Increase magnitude of `C_malpha` (more negative)
2. **Faster roll**: Increase magnitude of `C_lda` and `C_lp`
3. **More thrust**: Increase `max_thrust` in `PropulsionParams`
4. **Slower servos**: Increase `servo_tau` in `ActuatorParams`

## Module Architecture

```
examples/rc-jet/
├── main.py              # Entry point, world setup, visualization
├── sim.py               # System composition, BDXJet archetype
├── config.py            # Aircraft parameters and configuration
├── aero.py              # Aerodynamic force/moment computation
├── propulsion.py        # Turbine dynamics and thrust
├── actuators.py         # Control surface servo dynamics
├── control.py           # Flight plans and autopilot
├── README.md            # This file
└── BDX_Simulation_Whitepaper.md  # Technical design document
```

### Data Flow

```
Control Commands → Actuator Dynamics → Control Surfaces
                                              ↓
Throttle Command → Spool Dynamics → Thrust → Forces
                                              ↓
Body Velocity → Aero Angles → Aero Coefs → Aero Forces
                                              ↓
                                     6-DOF Integration
                                              ↓
                                  World Position/Velocity
```

## Advanced Usage

### Custom Flight Plans

Modify `control.py` to create custom maneuvers:

```python
@el.system
def custom_flight_plan(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    commands: el.Query[ControlCommands],
) -> el.Query[ControlCommands]:
    time = tick[0] * dt[0]
    
    # Your custom control logic
    elevator = jnp.sin(time * 0.1) * jnp.deg2rad(5.0)
    aileron = 0.0
    rudder = 0.0
    throttle = 0.75
    
    new_cmd = jnp.array([elevator, aileron, rudder, throttle])
    return commands.map(ControlCommands, lambda _: new_cmd)
```

Then integrate it in `sim.py`:

```python
from control import custom_flight_plan

def system() -> el.System:
    non_effectors = (
        custom_flight_plan  # Add your control system here
        | compute_velocity_body
        | compute_aero_angles
        # ... rest of systems
    )
```

### Hardware-in-the-Loop

The simulation supports external control via Impeller2 protocol. Mark control components as external:

```python
# In actuators.py, add external control metadata
ControlCommands = ty.Annotated[
    jax.Array,
    el.Component(
        "control_commands",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "element_names": "elevator,aileron,rudder,throttle",
            "external_control": "true",  # Enable external control
        },
    ),
]
```

Then run with database connection:
```python
# In main.py
world.run(
    sys,
    sim_time_step=config.dt,
    run_time_step=config.dt,  # Real-time execution
    db_addr="0.0.0.0:2240",   # Listen for connections
)
```

### XFLR5 Integration

To improve aerodynamic fidelity using XFLR5:

1. Create BDX geometry in XFLR5 (wing, tail surfaces)
2. Run VLM analysis at cruise speed (40 m/s)
3. Extract stability derivatives from analysis results
4. Update coefficients in `config.py`:

```python
C_Lalpha = <value from XFLR5>  # dCL/dα
C_malpha = <value from XFLR5>  # dCm/dα
# etc.
```

See Appendix B of the whitepaper for detailed XFLR5 workflow.

## Troubleshooting

### Simulation diverges (aircraft tumbles)
- Check initial trim: aircraft may need different throttle setting
- Verify inertia values are reasonable
- Ensure angle of attack stays below stall angle initially

### Aircraft doesn't maintain altitude
- Increase `C_Lalpha` (lift curve slope)
- Increase throttle or `max_thrust`
- Check that weight and lift balance at cruise speed

### Controls are sluggish
- Decrease `servo_tau` for faster servos
- Increase control derivatives (`C_mde`, `C_lda`, `C_ndr`)
- Check rate limits aren't too restrictive

### Unrealistic behavior
- Compare coefficients to similar aircraft (Rascal 110, Mentor)
- Verify sign conventions match aerospace standards
- Check that all units are SI (meters, kg, radians)

## References

- `BDX_Simulation_Whitepaper.md` - Complete technical design
- `references.md` - Source material and related work
- [Elodin Documentation](https://docs.elodin.systems) - Platform documentation

## Contributing

To improve the model:

1. Add wind/turbulence models in `aero.py`
2. Implement post-stall aerodynamics for aerobatics
3. Add landing gear ground contact model
4. Integrate XFLR5-derived coefficients
5. Implement full autopilot with navigation

## License

See repository LICENSE file.

