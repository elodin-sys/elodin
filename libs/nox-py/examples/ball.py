import jax
from jax import numpy as np
from jax import random
from elodin import *

TIME_STEP = 1.0 / 120.0
G = 6.6743e-11
R = 6.378e+6
M = 5.972e+24

Wind = Annotated[SpatialForce, Component("wind", ComponentType.SpatialMotionF64)]

@dataclass
class Globals(Archetype):
  seed: Seed
  wind: Wind = SpatialForce.zero()

@system
def sample_wind(s: Query[Seed], q: Query[Wind]) -> Query[Wind]:
   return q.map(Wind, lambda _w: Force.from_linear(0.2 * random.normal(random.key(s[0]), shape=(3,))))

@system
def apply_wind(w: Query[Wind], q: Query[Force]) -> Query[Force]:
   return q.map(Force, lambda f: Force.from_linear(f.force() + w[0].force()))

@system
def gravity(q: Query[Force, Inertia]) -> Query[Force]:
  def gravity_inner(force, inertia):
        m = inertia.mass()
        f = G * M * m / R**2
        return Force.from_linear(force.force() + np.array([0.0, -f, 0.0]))
  return q.map(Force, gravity_inner)

@system
def bounce(q: Query[WorldPos, WorldVel]) -> Query[WorldVel]:
  return q.map(WorldVel, lambda p, v: jax.lax.cond(
    jax.lax.max(p.linear()[1], v.linear()[1]) < 0.0,
    lambda _: WorldVel.from_linear(v.linear() * np.array([1.,-1.,1.]) * 0.85),
    lambda _: v,
    operand=None
  ))

w = WorldBuilder()
w.spawn(Globals(seed=np.int64(1))).metadata(EntityMetadata("Globals"))
w.spawn(
  Body(
    world_pos=WorldPos.from_linear(np.array([0.0, 6.0, 0.0])),
    pbr=w.insert_asset(Pbr(Mesh.sphere(0.4), Material.color(12.7, 9.2, 0.5))),
  )
).metadata(EntityMetadata("Ball"))
effectors = gravity.pipe(apply_wind)
sys = sample_wind.pipe(bounce.pipe(six_dof(TIME_STEP, effectors)))
w.run(sys, TIME_STEP)
