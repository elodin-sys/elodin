import elodin
import jax
import jax.numpy as np
from elodin import Component, ComponentType, system, ComponentArray, Archetype, WorldBuilder, Client, ComponentId, Query, WorldPos, WorldAccel, WorldVel, Inertia, Force, Body, six_dof, Material, Mesh

@system
def gravity(q: Query[WorldPos, Force]) -> ComponentArray[Force]:
  return q.map(Force, lambda p, f: Force.from_linear(np.array([0.,0.,1.])))

w = WorldBuilder()
b = Body(
    world_pos = WorldPos.from_linear(np.array([0.,0.,0.])),
    world_vel = WorldVel.from_linear(np.array([1.,0.,0.])),
    world_accel = WorldVel.from_linear(np.array([0.,0.,0.])),
    force = Force.zero(),
    inertia = Inertia.from_mass(1.0),
    mesh = w.insert_asset(Mesh.sphere(1.0)),
    material = w.insert_asset(Material.color(1.0, 1.0, 1.0))
)
w.spawn(b)
exec = w.run(six_dof(1.0 / 60.0, gravity))
