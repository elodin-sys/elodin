<h1 align="center">
  <a href="https://www.elodin.systems/">
    <img alt="banner" src="https://github.com/elodin-sys/elodin/assets/1129228/0e0197e9-12ec-42bd-b377-fa3ced2a1b7e">
  </a>
</h1>

Elodin is a platform for rapid design, testing, and simulation of
drones, satellites, and aerospace control systems.

Quick Demo: https://app.elodin.systems/sandbox/hn/cube-sat

Sandbox Alpha: https://app.elodin.systems  
Docs (WIP): https://docs.elodin.systems

This repository is a collection of core libraries:

- `libs/nox`: Tensor library that compiles to XLA (like
JAX, but for Rust).
- `libs/nox-ecs`: ECS framework built to work with Jax and Nox,
that allows you to build your own physics engine.
- `libs/nox-ecs-macros`: Derive macros to generate implementations of
ECS and Nox traits.
- `libs/conduit`: Column-based protocol for transferring ECS data
between different systems.
- `libs/xla-rs`: Rust bindings to XLA's C++ API (originally based on
https://github.com/LaurentMazare/xla-rs).

Join us on Discord: https://discord.gg/agvGJaZXy5!

## Getting Started

1. Setup a new venv with:

```fish 
python3 -m venv .venv
 . .venv/bin/activate.fish # or activate.sh if you don't use fish
```
2. Install `elodins`, and `matplotlib` with

``` fish
pip install elodin matplotlib
```

3. Try running the following code

```python 
import matplotlib.pyplot as plt
import jax.numpy as np
from elodin import *

@system
def gravity(q: Query[WorldPos]) -> Query[Force]:
  return q.map(Force, lambda _p: Force.from_linear(np.array([0.0, -9.81, 0.0])))

@system
def bounce(q: Query[WorldPos, WorldVel]) -> Query[WorldVel]:
  return q.map(WorldVel, lambda p, v: jax.lax.cond(
    jax.lax.max(p.linear()[1], v.linear()[1]) < 0.0,
    lambda _: WorldVel.from_linear(v.linear() * np.array([1.,-1.,1.]) * 0.85),
    lambda _: v,
    operand=None
  ))

w = WorldBuilder()
w.spawn(
    Body(
        world_pos=WorldPos.from_linear(np.array([0.0, 10.0, 0.0])),
        world_vel=WorldVel.from_linear(np.array([0.0, 0.0, 0.0])),
        inertia=Inertia.from_mass(1.0),
    )
)
client = Client.cpu()
exec = w.build(bounce.pipe(six_dof(1.0 / 60.0, gravity)))
t = range(500)
pos = []
for _ in t:
    exec.run(client)
    y = exec.column_array(ComponentId("world_pos"))[0, 5]
    pos.append(y)
fig, ax = plt.subplots()
ax.plot(t, pos)
plt.show()
```


## License

Licensed under either of

 * [Apache License, Version 2.0](LICENSES/Apache-2.0.txt)
 * [MIT License](LICENSES/MIT.txt)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
