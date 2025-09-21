# Fix for Component Write Issue

## The Problem
The client is sending data to the database, but the simulation isn't reading it. The issue is that:

1. The simulation creates `rocket.fin_control_trim` component when it spawns the rocket
2. The client sends data with VTable ID [3, 0] which creates a NEW stream
3. The simulation reads from the ORIGINAL component, not the client's stream

## The Solution
We need to ensure the client writes to the SAME component that the simulation created, not a parallel stream.

### Current Flow (BROKEN):
```
Simulation: Creates rocket.fin_control_trim = 0.0 → DB
Client: Creates new stream with VTable [3,0] → DB (different component!)
Simulation: Reads rocket.fin_control_trim → Gets 0.0 (original value)
```

### Needed Flow:
```
Simulation: Creates rocket.fin_control_trim = 0.0 → DB
Client: Updates rocket.fin_control_trim → DB (same component)
Simulation: Reads rocket.fin_control_trim → Gets client's value
```

## Implementation Fix

The client needs to:
1. Discover what VTable the simulation is using for its components
2. Use the SAME VTable ID to write data
3. Ensure the component ID matches exactly

OR

The simulation needs to:
1. Not write initial value for external_control components
2. Allow the client to be the sole writer

## Testing

With the debug logging added, we should see:
```
INFO: Copying external control rocket.fin_control_trim from DB: value=10.000
```

If we see:
```
DEBUG: External control component not found in DB: rocket.fin_control_trim
```

Then the client's writes aren't reaching the right component.
