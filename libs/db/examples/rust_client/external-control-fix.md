# External Control Component Fix

## The Final Solution

The simulation no longer writes back `fin_control_trim` to the database, preventing timestamp conflicts with external control clients.

## Implementation

### Modified `commit_world_head` in `libs/nox-ecs/src/impeller2_server.rs`

```rust
// IMPORTANT: Skip writing back components that are marked as external control
// This allows external clients to control these values without conflicts
if component_metadata.name == "fin_control_trim" {
    tracing::trace!("Skipping write-back for external control component: {}", pair_name);
    continue;
}
```

### How It Works

1. **Simulation reads** `rocket.fin_control_trim` from database (via `copy_db_to_world`)
2. **Simulation integrates** the value into its physics calculations
3. **Simulation skips writing back** `fin_control_trim` (marked as external control)
4. **Control client writes** new values with normal timestamps
5. **No conflicts** - only the control client writes to this component

## Advantages

✅ Clean separation of concerns
✅ No timestamp hacks needed
✅ Control client uses normal timestamps
✅ Simulation respects external control values
✅ No time travel errors

## Testing

1. **Rebuild nox-ecs and nox-py**:
   ```bash
   cargo build --release -p nox-ecs
   cd libs/nox-py && uvx maturin develop --release --uv
   ```

2. **Rebuild the Rust client**:
   ```bash
   cargo build --release -p elodin-db-rust-client
   ```

3. **Run the simulation**:
   ```bash
   cd libs/nox-py
   python examples/rocket.py run 0.0.0.0:2240
   ```

4. **Run the control client**:
   ```bash
   RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240
   ```

## Future Improvements

This hardcodes `fin_control_trim` as an external control component. A better solution would be to:

1. **Add metadata flag** to components marking them as "external_control"
2. **Check metadata** in `commit_world_head` instead of hardcoding names
3. **Allow multiple components** to be marked as external control

Example future implementation:
```python
# In rocket.py
FinControlTrim = ty.Annotated[
    jax.Array, 
    el.Component("fin_control_trim", el.ComponentType.F64, 
                 metadata={"external_control": "true"})
]
```

Then in `commit_world_head`:
```rust
if component_metadata.metadata.get("external_control") == Some(&"true".to_string()) {
    continue;  // Skip write-back
}
```

## Summary

This solution properly separates control authority:
- The simulation **reads and uses** external control values
- The simulation **does not write back** these values
- External clients have **exclusive write access** to control components
- No timestamp conflicts or time travel errors
