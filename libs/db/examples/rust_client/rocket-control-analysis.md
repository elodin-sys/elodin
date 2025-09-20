# Rocket.py Control System Analysis

## Executive Summary

The rocket simulation has a sophisticated autonomous control system with PID-based fin control. To enable external real-time control from the Rust client, we'll add a **trim component** that works additively with the existing PID controller. This approach maintains stability while allowing external influence - similar to trim tabs on aircraft.

## Current Control Architecture

### Control Flow Diagram (Original)

```
Acceleration Setpoint
        ↓
AccelSetpointSmooth (exponential smoothing)
        ↓
PitchPIDState (error calculation)
        ↓
PitchPIDControl (PID controller output)
        ↓
FinControl (control command)
        ↓
FinDeflect (actual fin angle, -40° to +40°)
        ↓
AeroCoefs (aerodynamic coefficients)
        ↓
AeroForces → Rocket Motion
```

### Proposed Control Flow (With Trim)

```
Acceleration Setpoint
        ↓
AccelSetpointSmooth
        ↓
PitchPIDState
        ↓
PitchPIDControl
        ↓
FinControl
        ↓
FinDeflect + FinControlTrim → Final Fin Angle
        ↓
AeroCoefs (aerodynamic coefficients)
        ↓
AeroForces → Rocket Motion
```

### Key Control Components

#### 1. **Primary Control Inputs**

| Component | Type | Range/Values | Purpose |
|-----------|------|--------------|---------|
| `rocket.fin_deflect` | f64 | -40.0 to +40.0 degrees | Actual fin deflection angle |
| `rocket.fin_control` | f64 | Typically ±0.2 | Incremental fin adjustment per timestep |
| `rocket.accel_setpoint` | f64[2] | [pitch, yaw] m/s² | Target acceleration |

#### 2. **PID Controller Components**

| Component | Current Value | Purpose |
|-----------|---------------|---------|
| `rocket.pitch_pid` | [1.1, 0.8, 3.8] | Kp, Ki, Kd gains |
| `rocket.pitch_pid_state` | [e, i, d] | Error, integral, derivative |
| `rocket.accel_setpoint_smooth` | [0.0, 0.0] | Smoothed setpoint |

#### 3. **Feedback Sensors** (Read-only)

| Component | Purpose |
|-----------|---------|
| `rocket.v_rel_accel_filtered` | Filtered relative acceleration (feedback signal) |
| `rocket.angle_of_attack` | Current angle of attack |
| `rocket.mach` | Current Mach number |
| `rocket.dynamic_pressure` | Current dynamic pressure |

## Control System Analysis

### Autonomous PID Loop (Current Implementation)

The rocket currently flies with a closed-loop PID controller:

1. **Setpoint Processing** (line 413-416):
   - Exponentially smooths the acceleration setpoint
   - Time constant: 0.5 seconds

2. **PID Calculation** (lines 419-438):
   - Error: `e = actual_accel - setpoint`
   - Integral: Clamped between -2.0 and 2.0
   - Output: `fin_control = Kp*e + Ki*i + Kd*d`

3. **Fin Actuation** (lines 441-447):
   - Scales control by Mach number: `fc = fc / (0.1 + mach)`
   - Clamps control increment: ±0.2 per timestep
   - Clamps total deflection: -40° to +40°

### Control Frequency

- **Simulation timestep**: 1/120 seconds (8.33ms)
- **Control loop**: Runs every simulation tick
- **Suitable for real-time**: Yes, matches the configured real-time rate

## Recommended Solution: Trim Control

### The Trim Approach (Additive Control) ✅

**Add `FinControlTrim` as an additive offset to fin deflection**

**Advantages:**
- PID controller remains active for stability
- External control adds to (rather than replaces) autonomous control  
- Simple, clean implementation
- Natural failsafe - trim defaults to 0
- Similar to aircraft trim tabs concept
- Can induce controlled roll rates

**Implementation:**

```python
# Add a new trim component
FinControlTrim = ty.Annotated[jax.Array, el.Component("fin_control_trim", el.ComponentType.F64)]

# Initialize in Rocket dataclass:
fin_control_trim: FinControlTrim = field(default_factory=lambda: jnp.float64(0.0))

# Modify the aero_coefs function to include trim (line 310):
@el.map
def aero_coefs(
    mach: Mach,
    angle_of_attack: AngleOfAttack,
    fin_deflect: FinDeflect,
    fin_trim: FinControlTrim,  # Add trim parameter
) -> AeroCoefs:
    # Apply trim to fin deflection
    effective_fin_deflect = jnp.clip(fin_deflect + fin_trim, -40.0, 40.0)
    
    aero = aero_interp_table(aero_df)
    aoa_sign = jax.lax.cond(
        jnp.abs(angle_of_attack) < 1e-6,
        lambda _: 1.0,
        lambda _: jnp.sign(angle_of_attack),
        operand=None,
    )
    # Use effective fin deflection with trim
    effective_fin_deflect *= aoa_sign
    coords = [
        to_coord(aero_df["Mach"], mach),
        to_coord(aero_df["Delta"], effective_fin_deflect),
        to_coord(aero_df["Alphac"], jnp.abs(angle_of_attack)),
    ]
    # ... rest of function remains the same
```

**Rust Client Usage:**
```rust
// Add positive trim to induce roll
client.set_component_value("rocket.fin_control_trim", 5.0).await?;

// Return to neutral
client.set_component_value("rocket.fin_control_trim", 0.0).await?;

// Negative trim for opposite roll
client.set_component_value("rocket.fin_control_trim", -5.0).await?;
```

### Why Trim is Better Than Override

1. **Safety**: PID controller continues to provide stability
2. **Simplicity**: No mode switching logic needed
3. **Predictability**: Effects are additive and intuitive
4. **Graceful Degradation**: Loss of external control just means trim returns to 0
5. **Coexistence**: External and autonomous control work together harmoniously

### Understanding Trim Effects

The trim value directly adjusts the effective fin deflection angle:

- **Positive Trim** (e.g., +10°): Deflects fins to induce clockwise roll (viewed from behind)
- **Negative Trim** (e.g., -10°): Deflects fins to induce counter-clockwise roll
- **Zero Trim**: No external influence, pure PID control

The actual fin angle becomes: `effective_angle = PID_deflection + trim`

This is clamped to ±40° to respect physical limits. The trim works continuously with the PID, so if the PID commands +5° and trim is +10°, the effective fin deflection is +15°.

## Required Modifications Summary

### Minimal Changes Needed (Trim Approach)

1. **Add one new component to rocket.py:**
   ```python
   FinControlTrim = ty.Annotated[jax.Array, el.Component("fin_control_trim", el.ComponentType.F64)]
   ```

2. **Initialize in Rocket dataclass:**
   ```python
   fin_control_trim: FinControlTrim = field(default_factory=lambda: jnp.float64(0.0))
   ```

3. **Modify `aero_coefs` function** to add trim to fin deflection before computing coefficients:
   ```python
   effective_fin_deflect = jnp.clip(fin_deflect + fin_trim, -40.0, 40.0)
   ```

4. **PID controller remains untouched** - it continues to provide baseline stability

### Control Constraints to Respect

| Constraint | Value | Reason |
|------------|-------|--------|
| Max fin deflection | ±40° | Physical limit |
| Max deflection rate | ~2.4°/timestep | Based on ±0.2 control increment |
| Min control period | 8.33ms | Simulation timestep |
| Mach scaling | fc/(0.1+M) | Control effectiveness decreases with speed |

## Testing Strategy

### Phase 1: Safety Test
1. Start rocket simulation
2. Let it stabilize (5-10 seconds)
3. Send small fin commands (±5°)
4. Verify immediate response
5. Verify return to neutral

### Phase 2: Control Authority
1. Command maximum deflection (±40°)
2. Verify clamping works
3. Test rapid alternation
4. Monitor angle of attack response

### Phase 3: Flight Control
1. Implement pitch-up maneuver
2. Implement pitch-down recovery
3. Test oscillation damping
4. Verify stable flight restoration

## Sample Control Scenarios

### 1. Controlled Roll Maneuver
```rust
// Induce a steady roll rate
client.set_component_value("rocket.fin_control_trim", 10.0).await?;
stellarator::sleep(Duration::from_secs(3)).await;

// Return to neutral
client.set_component_value("rocket.fin_control_trim", 0.0).await?;
```

### 2. Dynamic Stability Adjustment
```rust
// If angle of attack is high, add trim to help recovery
if aoa > 10.0 {
    // Add proportional trim to assist PID controller
    let trim_assist = -aoa * 0.5;  // Gentler than full override
    client.set_component_value("rocket.fin_control_trim", trim_assist).await?;
}
```

### 3. Sinusoidal Roll Pattern
```rust
// Create an oscillating roll pattern
for t in 0..100 {
    let phase = (t as f64) * 0.1;
    let trim = 8.0 * phase.sin();  // ±8° trim oscillation
    client.set_component_value("rocket.fin_control_trim", trim).await?;
    stellarator::sleep(Duration::from_millis(100)).await;
}
```

### 4. Manual Trim Adjustment (Keyboard Control)
```rust
// Respond to keyboard input
match key {
    'a' => {  // Trim left
        current_trim -= 1.0;
        client.set_component_value("rocket.fin_control_trim", current_trim).await?;
    }
    'd' => {  // Trim right  
        current_trim += 1.0;
        client.set_component_value("rocket.fin_control_trim", current_trim).await?;
    }
    's' => {  // Center trim
        current_trim = 0.0;
        client.set_component_value("rocket.fin_control_trim", 0.0).await?;
    }
    _ => {}
}
```

## Implementation Checklist

- [ ] Add `fin_control_trim` component to rocket.py
- [ ] Initialize trim to 0.0 in Rocket dataclass
- [ ] Modify `aero_coefs` function to apply trim
- [ ] Implement `SetComponentValue` message in Rust client
- [ ] Add trim command sending logic to processor.rs
- [ ] Create keyboard handler for trim adjustment (a/d/s keys)
- [ ] Add trim value to telemetry display
- [ ] Test trim effects during stable flight
- [ ] Verify PID controller continues working with trim
- [ ] Document trim control protocol

## Potential Issues and Mitigations

| Issue | Risk | Mitigation |
|-------|------|------------|
| Command latency | Delayed response | Use UDP for commands if TCP too slow |
| Lost commands | Stuck control surface | Implement timeout/failsafe to PID mode |
| Oscillations | Unstable flight | Add rate limiting and damping |
| Mode confusion | Wrong controller active | Clear mode indication in telemetry |
| Simultaneous controllers | Conflicting commands | Priority system or mutex |

## Next Steps

1. **Implement trim component**: Add `FinControlTrim` to rocket.py
2. **Test basic trim**: Verify trim affects roll rate as expected
3. **Add Rust client support**: Implement SetComponentValue message
4. **Create UI**: Add keyboard controls for trim adjustment in the TUI
5. **Enhance**: Consider adding trim for multiple axes if needed
6. **Document**: Create operator guide for trim control usage

## Conclusion

The rocket simulation is perfectly suited for external control integration using the **trim approach**. By adding a `FinControlTrim` component that works additively with the existing PID controller, we achieve the best of both worlds:

- **Stability**: The PID controller continues to maintain basic flight stability
- **Control**: External commands can influence the rocket's behavior through trim adjustments
- **Safety**: Loss of external control naturally returns trim to 0, allowing autonomous flight to continue
- **Simplicity**: Only requires adding one component and modifying one function

This trim-based approach is superior to mode switching because it allows the external control to **collaborate** with the autonomous system rather than replacing it. This is particularly valuable for inducing controlled roll rates or making fine trajectory adjustments while maintaining overall stability.

The implementation is minimal - just add the trim component, apply it in the aerodynamics calculation, and you're ready to send control commands from the Rust client.
