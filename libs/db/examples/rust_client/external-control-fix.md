# External Control Component Solution

## The Proper, Scalable Solution

Components can now be marked as "external control" in their declaration, preventing the simulation from writing them back to the database while still allowing initialization and reading.

## Implementation

### 1. Component Declaration in Python (`rocket.py`)

```python
FinControlTrim = ty.Annotated[
    jax.Array, 
    el.Component("fin_control_trim", el.ComponentType.F64, 
                 metadata={"external_control": "true"})
]

# Initialize with default value (works before client connects)
fin_control_trim: FinControlTrim = field(default_factory=lambda: jnp.float64(0.0))
```

### 2. Metadata Check in `libs/nox-ecs/src/impeller2_server.rs`

```rust
// Skip writing back components that are marked as external control
// This allows external clients to control these values without conflicts
if component_metadata.metadata.get("external_control") == Some(&"true".to_string()) {
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

## Key Features

✅ **Metadata-based**: No hardcoded component names
✅ **Scalable**: Any component can be marked as external control
✅ **Initialized**: Components have default values before client connects
✅ **Clean**: Simulation reads values but doesn't write them back

## Adding More External Control Components

Simply add the metadata when declaring any component:

```python
# Example: External throttle control
ThrottleOverride = ty.Annotated[
    jax.Array,
    el.Component("throttle_override", el.ComponentType.F64, 
                 metadata={"external_control": "true"})
]

# Example: External guidance target
GuidanceTarget = ty.Annotated[
    jax.Array,
    el.Component("guidance_target", 
                 el.ComponentType(el.PrimitiveType.F64, (3,)),
                 metadata={"external_control": "true", 
                          "element_names": "x,y,z"})
]
```

## Summary

This solution properly separates control authority:
- The simulation **reads and uses** external control values
- The simulation **does not write back** these values
- External clients have **exclusive write access** to control components
- No timestamp conflicts or time travel errors
