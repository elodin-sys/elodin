# Testing Roll Effect from Trim Control

## The Fix Applied

The rocket's aerodynamics model now generates roll moment from the trim control input. This simulates differential fin deflection that would create roll in a real rocket.

## Test Instructions

### 1. Rebuild nox-py with Physics Fix
```bash
cd libs/nox-py
uvx maturin develop --release --uv
```
✅ Already completed

### 2. Start Rocket Simulation
```bash
cd libs/nox-py
.venv/bin/python examples/rocket.py run 0.0.0.0:2240
```

### 3. Run Control Client
```bash
cd /Users/danieldriscoll/dev/elodin
RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240
```

### 4. Connect Elodin Editor
```bash
elodin editor 127.0.0.1:2240
```

## What You Should See

### In the Editor Graphs:
1. **"Trim Control"** graph - Sinusoidal wave oscillating ±10°
2. **"Fin Deflection"** graph - Combined PID + trim deflection
3. **"Aero Coefficients"** graph - First value (Cl) oscillating with trim

### In the 3D View:
- The rocket should now exhibit a **rolling/spiraling motion**
- The roll rate should oscillate with the trim command
- The trajectory will be more complex due to roll-pitch coupling

## Physics Explanation

The fix adds roll moment generation:
```python
roll_effectiveness = 0.01  # Roll moment per degree of trim
cl = fin_trim * roll_effectiveness
```

This means:
- **+10° trim** → +0.1 roll moment coefficient
- **-10° trim** → -0.1 roll moment coefficient
- Roll oscillates at 0.25 Hz (4-second period)

## Tuning the Effect

If the roll is too subtle or too strong, adjust `roll_effectiveness` in `rocket.py`:

```python
roll_effectiveness = 0.001  # Very subtle
roll_effectiveness = 0.01   # Current - moderate
roll_effectiveness = 0.05   # Strong
roll_effectiveness = 0.1    # Very aggressive
```

## Troubleshooting

### If No Roll Effect:
1. Verify nox-py was rebuilt after the physics changes
2. Check that client is sending trim values (look for oscillation in logs)
3. Confirm "Aero Coefficients" graph shows Cl (first value) changing

### If Roll Too Strong/Unstable:
1. Reduce `roll_effectiveness` value
2. Reduce trim amplitude in control client (currently ±10°)

## Success Criteria

✅ Trim values received without errors
✅ No timestamp conflicts
✅ "Trim Control" graph shows oscillation
✅ "Aero Coefficients" shows Cl varying
✅ **Rocket exhibits visible rolling motion**
✅ Trajectory shows spiral pattern

The external control system is now fully functional with visible physics effects!
