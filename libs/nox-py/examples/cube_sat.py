import elodin
import jax
import jax.numpy as np
import numpy
from elodin import Component, ComponentType, system, ComponentArray, Archetype, WorldBuilder, Client, ComponentId, Query, WorldPos, WorldAccel, WorldVel, Inertia, Force, Body, six_dof, Material, Mesh, Pbr


def lqr_control_mat(j, Q, r):
  d_diag = -1.0 * np.array([np.sqrt( q1i/ri + ji*np.sqrt(q2i/ri) ) for (q1i, q2i, ri, ji) in zip(q[:3], q[3:], r, j)])
  k_diag = -1.0 * np.array([np.sqrt(q2i/ri) for (q2i, ri) in zip(q[3:], r)])
  return (d_diag, k_diag)

j = np.array([0.13, 0.10, 0.05])
q = np.array([12, 12, 12, 12, 12, 12])
r = np.array([1.0, 1.0, 1.0])
(d, k) = lqr_control_mat(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0]))

@system
def control(q: Query[WorldPos, WorldVel]) -> ComponentArray[Force]:
  #return q.map(Force, lambda p, v: Force.from_torque(np.array([0.0,0,0])))
  return q.map(Force, lambda p, v: Force.from_torque(np.clip(v.angular() * d + p.angular().vector()[:3] * k, -0.5, 0.5)))

w = WorldBuilder()
b = Body(
    world_pos = WorldPos.from_linear(np.array([0.,0.,0.])),
    world_vel = WorldVel.from_angular(np.array([5.0,5.,5.])),
    world_accel = WorldVel.from_linear(np.array([0.,0.,0.])),
    force = Force.zero(),
    inertia = Inertia(12.0, j),
    pbr = w.insert_asset(Pbr.from_path("examples/oresat-low.glb")),
)
w.spawn(b)
exec = w.run(six_dof(1.0 / 60.0, control))
