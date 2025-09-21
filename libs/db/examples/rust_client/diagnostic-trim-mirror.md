# Diagnostic: Trim Mirror Test

## Purpose
Prove whether external control data is making it into the NOX compiled system at all.

## What We Added

### 1. TrimMirror Component
A new component that mirrors the trim value + 1000

### 2. Diagnostic System
```python
@el.map
def mirror_trim_diagnostic(fin_trim: FinControlTrim) -> TrimMirror:
    """Diagnostic system: copy trim value to mirror component to verify it's available"""
    return fin_trim + 1000.0  # Add 1000 so we can easily see if it's working
```

## What to Look For

Run the simulation and client, then check the graphs:

### Expected Results

#### If External Control is Working:
- **Trim Control (External)**: Shows oscillating ±10° from client
- **DIAGNOSTIC: Trim Mirror**: Should show 1000 ± 10 (oscillating between 990 and 1010)
- **Roll Moment (Cl)**: Should oscillate ± 1.0

#### If External Control is NOT Working:
- **Trim Control (External)**: Shows oscillating ±10° from client (this comes from DB)
- **DIAGNOSTIC: Trim Mirror**: Stays at 1000 (trim is 0 in the system)
- **Roll Moment (Cl)**: Stays at 0

## The Test

1. Run simulation:
```bash
cd libs/nox-py
.venv/bin/python examples/rocket.py run 0.0.0.0:2240
```

2. Run client:
```bash
./target/release/rust_client --host 127.0.0.1 --port 2240
```

3. Open editor:
```bash
elodin editor 127.0.0.1:2240
```

## Diagnosis

- **If Trim Mirror = 1000 constant**: The `fin_control_trim` value is NOT available in the compiled system
- **If Trim Mirror oscillates 990-1010**: The value IS available but something else is wrong with how it's used

This will definitively tell us whether the external control data is making it into the system pipeline.
