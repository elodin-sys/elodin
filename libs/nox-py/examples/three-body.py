from jax import numpy as np
from jax.numpy import linalg as la
from elodin import *

TIME_STEP = 1.0 / 120.0

GravityEdge = Component[Edge, "gravity_edge", ComponentType.Edge]
G = 6.6743e-11

@dataclass
class GravityConstraint(Archetype):
    a: GravityEdge
    def __init__(self, a: EntityId, b: EntityId):
        self.a = Edge(a, b)


@system
def gravity(q: GraphQuery[GravityEdge, WorldPos, Inertia]) -> Query[Force]:
    def gravity_inner(force, a_pos, a_inertia, b_pos, b_inertia):
        r = a_pos.linear() - b_pos.linear()
        m = a_inertia.mass()
        M = b_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return Force.from_linear(force[3:] - f)
    return q.edge_fold(Force, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), gravity_inner)


w = WorldBuilder()
a = w.spawn(
    Body(
        world_pos=WorldPos.from_linear(np.array([0.8822391241, 0, 0])),
        world_vel=WorldVel.from_linear(np.array([0, 1.0042424155, 0])),
        inertia=Inertia.from_mass(1.0 / G),
        pbr=w.insert_asset(Pbr(Mesh.sphere(0.2), Material.color(25.3, 18.4, 1.0))),
    )
).metadata(EntityMetadata("A")).id()
b = w.spawn(
    Body(
        world_pos=WorldPos.from_linear(np.array([-0.6432718586,0, 0])),
        world_vel=WorldVel.from_linear(np.array([0, -1.6491842814, 0])),
        inertia=Inertia.from_mass(1.0 / G),
        pbr=w.insert_asset(Pbr(Mesh.sphere(0.2), Material.color(10.0, 0.0, 10.0))),
    )
).metadata(EntityMetadata("B")).id()
c = w.spawn(
    Body(
        world_pos=WorldPos.from_linear(np.array([-0.2389672654, 0, 0])),
        world_vel=WorldVel.from_linear(np.array([0,0.6449418659, 0.0])),
        inertia=Inertia.from_mass(1.0 / G),
        pbr=w.insert_asset(Pbr(Mesh.sphere(0.2), Material.color(0.0, 10.0, 10.0))),
    )
).metadata(EntityMetadata("C")).id()
w.spawn(GravityConstraint(a, b)).metadata(EntityMetadata("A -> B"))
w.spawn(GravityConstraint(a, c)).metadata(EntityMetadata("A -> C"))

w.spawn(GravityConstraint(b, c)).metadata(EntityMetadata("B -> C"))
w.spawn(GravityConstraint(b, a)).metadata(EntityMetadata("B -> A"))

w.spawn(GravityConstraint(c, a)).metadata(EntityMetadata("C -> A"))
w.spawn(GravityConstraint(c, b)).metadata(EntityMetadata("C -> B"))

sys = six_dof(TIME_STEP, gravity)
sim = w.run(sys, TIME_STEP)
