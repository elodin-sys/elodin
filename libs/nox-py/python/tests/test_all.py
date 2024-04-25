import pytest
import jax
import jax.numpy as np
from jax import random
from elodin import *
from dataclasses import dataclass


def test_basic_system():
    X = Annotated[jax.Array, Component("x", ComponentType.F32)]
    Y = Annotated[jax.Array, Component("y", ComponentType.F32)]
    E = Annotated[jax.Array, Component("e", ComponentType.F32)]

    @system
    def foo(x: Query[X]) -> Query[X]:
        return x.map(X, lambda x: x * 2)

    @system
    def bar(q: Query[X, Y]) -> Query[X]:
        return q.map(X, lambda x, y: x * y)

    @map
    def baz(x: X, e: E) -> X:
        return x + e

    @dataclass
    class Test(Archetype):
        x: X
        y: Y

    @dataclass
    class Effect(Archetype):
        e: E

    sys = foo.pipe(bar).pipe(baz)
    client = Client.cpu()
    w = WorldBuilder()
    w.spawn(Test(np.array([1.0], dtype="float32"), np.array([500.0], dtype="float32")))
    w.spawn(
        Test(np.array([15.0], dtype="float32"), np.array([500.0], dtype="float32"))
    ).insert(Effect(np.array([15.0], dtype="float32")))
    exec = w.build(sys)
    exec.run(client)
    x1 = exec.column_array(Component.id(X))
    y1 = exec.column_array(Component.id(Y))
    assert (x1 == [1000.0, 15015.0]).all()
    assert (y1 == [500.0, 500.0]).all()
    exec.run(client)
    x1 = exec.column_array(Component.id(X))
    y1 = exec.column_array(Component.id(Y))
    assert (x1 == [1000000.0, 15015015.0]).all()
    assert (y1 == [500.0, 500.0]).all()


def test_six_dof():
    w = WorldBuilder()
    w.spawn(
        Body(
            world_pos=WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=WorldVel.from_linear(np.array([1.0, 0.0, 0.0])),
            inertia=Inertia.from_mass(1.0),
        )
    )
    client = Client.cpu()
    sys = six_dof(1.0 / 60.0)
    exec = w.build(sys)
    exec.run(client)
    x = exec.column_array(Component.id(WorldPos))
    assert (x == [0, 0, 0, 1, 1.0 / 60.0, 0.0, 0.0]).all()


def test_graph():
    X = Annotated[jax.Array, Component("x", ComponentType.F64)]
    E = Annotated[Edge, Component("test_edge", ComponentType.Edge)]

    @dataclass
    class Test(Archetype):
        x: X

    @dataclass
    class EdgeArchetype(Archetype):
        edge: E

    @system
    def fold_test(graph: GraphQuery[E], x: Query[X]) -> Query[X]:
        return graph.edge_fold(x, x, X, np.array(5.0), lambda x, a, b: x + a + b)

    w = WorldBuilder()
    a = w.spawn(Test(np.array([1.0], dtype="float64")))
    b = w.spawn(Test(np.array([2.0], dtype="float64")))
    c = w.spawn(Test(np.array([2.0], dtype="float64")))
    w.spawn(EdgeArchetype(Edge(a.id(), b.id())))
    w.spawn(EdgeArchetype(Edge(a.id(), c.id())))
    w.spawn(EdgeArchetype(Edge(b.id(), c.id())))
    client = Client.cpu()
    exec = w.build(fold_test)
    exec.run(client)
    x = exec.column_array(Component.id(X))
    assert (x == [11.0, 9.0, 2.0]).all()


def test_seed():
    X = Annotated[jax.Array, Component("x", ComponentType.F64)]
    Y = Annotated[jax.Array, Component("y", ComponentType.F64)]

    @system
    def foo(x: Query[X]) -> Query[X]:
        return x.map(X, lambda x: x * 2)

    @system
    def bar(q: Query[X, Y]) -> Query[X]:
        return q.map(X, lambda x, y: x * y)

    @system
    def seed_mul(s: Query[Seed], q: Query[X]) -> Query[X]:
        return q.map(X, lambda x: x * s[0])

    @system
    def seed_sample(s: Query[Seed], q: Query[X, Y]) -> Query[Y]:
        def sample_inner(x, y):
            key = random.key(s[0])
            key = random.fold_in(key, x)
            scaler = random.uniform(key, minval=1.0, maxval=2.0)
            return y * scaler

        return q.map(Y, sample_inner)

    @dataclass
    class Globals(Archetype):
        seed: Seed

    @dataclass
    class Test(Archetype):
        x: X
        y: Y

    sys = foo.pipe(bar).pipe(seed_mul).pipe(seed_sample)
    client = Client.cpu()
    w = WorldBuilder()
    w.spawn(Globals(seed=np.array(2)))
    w.spawn(Test(np.array(1.0), np.array(500.0)))
    w.spawn(Test(np.array(15.0), np.array(500.0)))
    exec = w.build(sys)
    exec.run(client)
    x1 = exec.column_array(Component.id(X))
    y1 = exec.column_array(Component.id(Y))
    assert (x1 == [2000.0, 30000.0]).all()
    assert (y1 >= [500.0, 500.0]).all()
    assert (y1 <= [1000.0, 1000.0]).all()


def test_archetype_name():
    X = Annotated[jax.Array, Component("x", ComponentType.F64)]

    @dataclass
    class TestArchetype(Archetype):
        x: X

    assert TestArchetype.archetype_name() == "test_archetype"
    assert Body.archetype_name() == "body"


def test_spatial_vector_algebra():
    @map
    def double_vec(v: WorldVel) -> WorldVel:
        return v + v

    client = Client.cpu()
    w = WorldBuilder()
    w.spawn(Body(world_vel=WorldVel.from_linear(np.array([1.0, 0.0, 0.0]))))
    exec = w.build(double_vec)
    exec.run(client)
    v = exec.column_array(Component.id(WorldVel))
    assert (v[0][3:] == [2.0, 0.0, 0.0]).all()
