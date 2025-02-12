# Elodin

Docs: https://docs.elodin.systems

Elodin is a platform for rapid design, testing, and simulation of drones, satellites, and aerospace control systems.

## Usage

```sh
elodin editor <path/to/src.py>
```

## Templates

Templates can be found at http://app.elodin.systems.

## PySim Usage Examples

```python
# Initialize the simulation
sim = w.to_jax(sys, sim_time_step=1.0 / 20.0)

# Advance the simulation by 500 steps
sim.step(500)

# Retrieve the state of a specific component
state = sim.get_state(component_name="att_est", entity_name="OreSat")
print(state)