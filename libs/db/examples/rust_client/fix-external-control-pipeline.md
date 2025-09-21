# Fix: External Control Component Pipeline Issue

## The Problem
The external control component `fin_control_trim` was being:
1. ✅ Written to the database by the client
2. ✅ Read from the database by `copy_db_to_world`
3. ❌ NOT being used by the physics calculations

The issue was that the compiled NOX system wasn't aware that this component needed to be dynamically read each tick.

## The Solution

Added an explicit `read_external_trim` system to the pipeline that reads the `fin_control_trim` component:

```python
@el.map
def read_external_trim(fin_trim: FinControlTrim) -> FinControlTrim:
    """Pass-through system to ensure external control component is read from DB"""
    return fin_trim
```

And added it to the system pipeline:
```python
non_effectors = (
    ...
    | read_external_trim  # Ensure external control component is read each tick
    | aero_coefs
    ...
)
```

## Why This Works

The NOX compiler builds a static computation graph at compile time. Components that aren't explicitly read by any system in the pipeline might not be updated from the database values each tick.

By adding a pass-through system that reads the external control component, we ensure:
1. The component is part of the compiled computation graph
2. The value is read from the world (which gets updated from DB) each tick
3. The updated value is available to downstream systems like `aero_coefs`

## Testing

After this fix, you should see:
- Roll Moment graph oscillating between -1.0 and +1.0
- Rocket rolling back and forth with the trim commands
- Trim Control graph showing the ±10° oscillation
