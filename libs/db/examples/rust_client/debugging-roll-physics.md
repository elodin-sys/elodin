# Debugging Roll Physics

## The Issue
The rocket isn't showing visible roll even though:
- External control trim values are being received
- Roll effectiveness is set to 0.1 (very aggressive)
- Trim is oscillating ±10°

## Debug Changes Made

### 1. Added Constant Roll Moment
```python
cl = fin_trim * roll_effectiveness  # Should give ±1.0 from ±10° trim
cl = cl + 0.5  # Added constant roll to test physics
```
This adds a constant roll moment of 0.5 to verify the physics pipeline works.

### 2. Added Roll Moment Debug Component
Created `roll_moment_debug` component to directly monitor the Cl coefficient.

### 3. Added Graph to Schematic
Added "Roll Moment (Cl)" graph to visualize the actual roll moment being generated.

## What to Check

### 1. Run the Simulation
```bash
cd libs/nox-py
.venv/bin/python examples/rocket.py run 0.0.0.0:2240
```

### 2. Run Control Client
```bash
RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240
```

### 3. In Elodin Editor
```bash
elodin editor 127.0.0.1:2240
```

Look for:
1. **"Roll Moment (Cl)" graph** - Should show:
   - Base value of 0.5 (constant we added)
   - Oscillation of ±1.0 on top of that (from trim)
   - Total should oscillate between -0.5 and 1.5

2. **"Trim Control" graph** - Should show ±10° oscillation

3. **"Aero Coefficients" graph** - First value (Cl) should match Roll Moment

4. **3D View** - Rocket should now be rolling continuously due to the 0.5 constant

## Possible Issues

### If Roll Moment Shows 0.5 Constant But No Oscillation:
- Trim values aren't being received from database
- Component marked as external_control isn't being read

### If Roll Moment Shows Correct Values But No Roll:
- Issue with aerodynamic force application
- Possible coordinate system mismatch
- Inertia tensor issue

### If No Roll Moment Shows Up:
- System pipeline issue
- Component not initialized properly

## Next Steps

Based on what you see:

1. **If constant roll (0.5) works**: The physics is fine, issue is with trim reception
2. **If nothing works**: Physics pipeline has an issue
3. **If graphs show correct values but no motion**: Force application issue

Once we identify which part is broken, we can fix it properly and remove the debug constant.
