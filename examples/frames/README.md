# Frame Independence Verification Example

This example demonstrates and verifies that physics behaves correctly across different coordinate frames in Elodin.

## Overview

Elodin supports multiple coordinate frame conventions:
- **ENU** (East-North-Up): Standard for terrestrial robotics
- **NED** (North-East-Down): Standard for aviation
- **ECEF** (Earth-Centered Earth-Fixed): Earth-relative positioning
- **ECI** (Earth-Centered Inertial): Low Earth orbit
- **GCRF** (Geocentric Celestial Reference Frame): Deep space

This example runs identical simulations in different frames and verifies that the physics is frame-independent (when properly accounting for coordinate transformations).

## Tests

### 1. Gravity in ENU vs NED

Drops a ball in both ENU and NED frames and verifies:
- In ENU: gravity points in -Z direction (down)
- In NED: gravity points in +Z direction (down in NED convention)
- The magnitudes of displacement and velocity are equal but opposite in sign

### 2. Inertial Frame Equivalence (ECI vs GCRF)

Simulates a two-body orbital system in both ECI and GCRF frames and verifies:
- Both inertial frames produce identical trajectories
- Position and velocity match to within numerical precision

### 3. Energy Conservation

Verifies that total mechanical energy is conserved during simulation:
- Kinetic energy + potential energy = constant
- Tests that the physics integrator is working correctly

## Running the Example

```bash
cd examples/frames
python3 main.py
```

## Expected Output

```
==============================================================
ELODIN FRAME INDEPENDENCE VERIFICATION
==============================================================

This example demonstrates that physics behaves correctly
across different coordinate frames in Elodin.

==============================================================
TEST 1: Gravity in ENU vs NED
==============================================================

Running ENU simulation...
Running NED simulation...

ENU Results:
  Initial Z: 10.0000 m
  Final Z:   5.0913 m
  Delta Z:   -4.9087 m
  Final Vz:  -9.8175 m/s

NED Results:
  Initial Z: -10.0000 m
  Final Z:   -5.0913 m
  Delta Z:   4.9087 m
  Final Vz:  9.8175 m/s

Verification:
  Displacement magnitudes match: True
  Velocity magnitudes match:     True

✅ TEST PASSED: Gravity works correctly in both ENU and NED frames

... (additional tests)

==============================================================
SUMMARY
==============================================================

✅ PASS: Gravity ENU/NED
✅ PASS: Inertial Frames
✅ PASS: Energy Conservation

3/3 tests passed

🎉 All frame verification tests passed!
The configurable frame system is working correctly.
```

## What This Demonstrates

1. **Frame Independence**: Physical laws (Newton's laws) work the same in all coordinate frames
2. **Proper Transformations**: Coordinate transformations are handled correctly
3. **Inertial Frame Equivalence**: Different inertial frames produce identical results
4. **Conservation Laws**: Energy and momentum are conserved regardless of frame choice

## Key Takeaways

- Always explicitly specify the frame when creating a world: `el.World(frame=el.Frame.ENU)`
- Gravity direction changes with frame convention:
  - ENU: `[0, 0, -9.81]` (up is +Z)
  - NED: `[0, 0, +9.81]` (down is +Z)
- Inertial frames (ECI, GCRF) are equivalent for physics simulations
- Position/velocity interpretations must account for frame convention

## Use Cases

Use this example to:
- Verify your Elodin installation is working correctly
- Understand how different frames affect coordinate interpretations
- Test frame transformations in your own simulations
- Debug frame-related issues

## Implementation Notes

The tests use Elodin's built-in physics (`el.six_dof`) with different frame configurations. By comparing datasets from identical initial conditions in different frames, we can verify that:

1. The physics engine is frame-independent
2. Coordinate transformations are correct
3. The frame metadata is properly stored and used

This serves as both an example and a regression test for the frame system.

