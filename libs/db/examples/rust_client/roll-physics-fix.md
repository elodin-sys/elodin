# Rocket Roll Physics Fix

## The Problem

The external control trim values were being received and applied correctly, but had no visible effect on the rocket because:

1. **Roll moment was hardcoded to zero** in the aerodynamics model
2. **Single fin deflection model** - the rocket uses one deflection value for all fins, not differential control

## Original Code Issue

In the `aero_coefs` function, the aerodynamic coefficients were:
```python
coefs = jnp.array(
    [
        0.0,  # Cl (roll moment) - ALWAYS ZERO!
        0.0,  # CnR (yaw moment)
        coefs[0] * aoa_sign,  # CmR (pitch moment)
        coefs[1],  # CA (axial force)
        coefs[2] * aoa_sign,  # CZR (normal force)
        0.0,  # CYR (side force)
    ]
)
```

The Cl (roll moment coefficient) was hardcoded to 0.0, meaning no amount of fin deflection would create roll.

## The Fix

Added roll moment generation based on the trim value:

```python
# Add roll moment based on fin deflection to make trim visible
# This simulates differential fin deflection creating roll
roll_effectiveness = 0.01  # Roll moment per degree of trim
cl = fin_trim * roll_effectiveness  # Roll moment from trim only

coefs = jnp.array(
    [
        cl,  # Cl (roll moment) - now affected by trim!
        0.0,  # CnR (yaw moment)
        coefs[0] * aoa_sign,  # CmR (pitch moment)
        coefs[1],  # CA (axial force)
        coefs[2] * aoa_sign,  # CZR (normal force)
        0.0,  # CYR (side force)
    ]
)
```

## Physical Interpretation

This simulates what would happen with differential fin control:
- **Positive trim**: Creates clockwise roll (viewed from behind)
- **Negative trim**: Creates counter-clockwise roll
- **Roll effectiveness**: 0.01 rad/deg gives moderate roll response

In a real rocket with 4 fins:
- Normal fin deflection: All fins move together → pitch/yaw control
- Differential deflection: Opposite fins move oppositely → roll control
- Our trim simulates the differential component

## Visualization Updates

Added graphs to better observe the control effects:
1. **"Trim Control"** - Shows the oscillating trim command
2. **"Fin Deflection"** - Shows total fin angle (PID + trim)
3. **"Aero Coefficients"** - Shows all 6 coefficients including roll

## Testing the Fix

1. Rebuild and run simulation:
```bash
cd libs/nox-py
uvx maturin develop --release --uv
python examples/rocket.py run 0.0.0.0:2240
```

2. Run control client:
```bash
RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240
```

3. Observe in Elodin Editor:
- Rocket should now show visible roll oscillation
- "Aero Coefficients" graph shows Cl varying with trim
- Rocket trajectory will spiral due to roll

## Tuning Roll Response

Adjust `roll_effectiveness` to control sensitivity:
- **0.001**: Very subtle roll (barely visible)
- **0.01**: Moderate roll (current setting)
- **0.05**: Strong roll response
- **0.1**: Very aggressive roll

## Future Improvements

For more realistic modeling:
1. **Individual fin control**: Model 4 separate fins
2. **Fin interaction effects**: Cross-coupling between fins
3. **Reynolds number effects**: Vary effectiveness with speed
4. **Nonlinear effects**: Stall at high deflection angles
