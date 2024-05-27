import jax
import jax.numpy as np
from jax import random
import elodin as el
import typing as ty
from dataclasses import dataclass

X = ty.Annotated[jax.Array, el.Component("x", el.ComponentType.F64)]
Y = ty.Annotated[jax.Array, el.Component("y", el.ComponentType.F64)]
Effect = ty.Annotated[jax.Array, el.Component("e", el.ComponentType.F64)]

E = ty.Annotated[el.Edge, el.Component("test_edge")]


def test_basic_system():
    @el.system
    def foo(x: el.Query[X]) -> el.Query[X]:
        return x.map(X, lambda x: x * 2)

    @el.system
    def bar(q: el.Query[X, Y]) -> el.Query[X]:
        return q.map(X, lambda x, y: x * y)

    @el.map
    def baz(x: X, z: Effect) -> X:
        return x + z

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    @dataclass
    class EffectArchetype(el.Archetype):
        e: Effect

    sys = foo.pipe(bar).pipe(baz)
    w = el.World()
    w.spawn(Test(np.array([1.0]), np.array([500.0])))
    w.spawn(
        [
            Test(np.array([15.0]), np.array([500.0])),
            EffectArchetype(np.array([15.0])),
        ]
    )
    exec = w.build(sys)
    exec.run()
    x1 = exec.column_array(el.Component.id(X))
    y1 = exec.column_array(el.Component.id(Y))
    assert (x1 == [1000.0, 15015.0]).all()
    assert (y1 == [500.0, 500.0]).all()
    exec.run()
    x1 = exec.column_array(el.Component.id(X))
    y1 = exec.column_array(el.Component.id(Y))
    assert (x1 == [1000000.0, 15015015.0]).all()
    assert (y1 == [500.0, 500.0]).all()


def test_six_dof():
    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_linear(np.array([1.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )
    sys = el.six_dof(1.0 / 60.0)
    exec = w.build(sys)
    exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    assert (x == [0.0, 0.0, 0.0, 1.0, 1.0 / 60.0, 0.0, 0.0]).all()


def test_graph():
    @dataclass
    class Test(el.Archetype):
        x: X

    @dataclass
    class EdgeArchetype(el.Archetype):
        edge: E

    @el.system
    def fold_test(graph: el.GraphQuery[E], x: el.Query[X]) -> el.Query[X]:
        return graph.edge_fold(x, x, X, np.array(5.0), lambda x, a, b: x + a + b)

    w = el.World()
    a = w.spawn(Test(np.array([1.0])))
    b = w.spawn(Test(np.array([2.0])))
    c = w.spawn(Test(np.array([2.0])))
    print(a, b, c)
    w.spawn(EdgeArchetype(el.Edge(a, b)))
    w.spawn(EdgeArchetype(el.Edge(a, c)))
    w.spawn(EdgeArchetype(el.Edge(b, c)))
    exec = w.build(fold_test)
    exec.run()
    x = exec.column_array(el.Component.id(X))
    assert (x == [11.0, 9.0, 2.0]).all()


def test_seed():
    @el.system
    def foo(x: el.Query[X]) -> el.Query[X]:
        return x.map(X, lambda x: x * 2)

    @el.system
    def bar(q: el.Query[X, Y]) -> el.Query[X]:
        return q.map(X, lambda x, y: x * y)

    @el.system
    def seed_mul(s: el.Query[el.Seed], q: el.Query[X]) -> el.Query[X]:
        return q.map(X, lambda x: x * s[0])

    @el.system
    def seed_sample(s: el.Query[el.Seed], q: el.Query[X, Y]) -> el.Query[Y]:
        def sample_inner(x, y):
            key = random.key(s[0])
            key = random.fold_in(key, x)
            scaler = random.uniform(key, minval=1.0, maxval=2.0)
            return y * scaler

        return q.map(Y, sample_inner)

    @dataclass
    class Globals(el.Archetype):
        seed: el.Seed

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    sys = foo.pipe(bar).pipe(seed_mul).pipe(seed_sample)
    w = el.World()
    w.spawn(Globals(seed=np.array(2)))
    w.spawn(Test(np.array(1.0), np.array(500.0)))
    w.spawn(Test(np.array(15.0), np.array(500.0)))
    exec = w.build(sys)
    exec.run()
    x1 = exec.column_array(el.Component.id(X))
    y1 = exec.column_array(el.Component.id(Y))
    assert (x1 == [2000.0, 30000.0]).all()
    assert (y1 >= [500.0, 500.0]).all()
    assert (y1 <= [1000.0, 1000.0]).all()


def test_archetype_name():
    @dataclass
    class TestArchetype(el.Archetype):
        x: X

    assert TestArchetype.archetype_name() == "test_archetype"
    assert el.Body.archetype_name() == "body"


def test_spatial_vector_algebra():
    @el.map
    def double_vec(v: el.WorldVel) -> el.WorldVel:
        return v + v

    w = el.World()
    w.spawn(el.Body(world_vel=el.WorldVel.from_linear(np.array([1.0, 0.0, 0.0]))))
    exec = w.build(double_vec)
    exec.run()
    v = exec.column_array(el.Component.id(el.WorldVel))
    assert (v[0][3:] == [2.0, 0.0, 0.0]).all()


def test_six_dof_ang_vel_int():
    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_angular(np.array([0.0, 0.0, 1.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )
    sys = el.six_dof(1.0 / 120.0)
    exec = w.build(sys)
    for _ in range(120):
        exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    # value from Julia and Simulink
    assert np.isclose(
        x.to_numpy()[0],
        np.array([0.0, 0.0, 0.479425538604203, 0.8775825618903728, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()

    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_angular(np.array([0.0, 1.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )
    sys = el.six_dof(1.0 / 120.0)
    exec = w.build(sys)
    for _ in range(120):
        exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    # value from Julia and Simulink
    assert np.isclose(
        x.to_numpy()[0],
        np.array([0.0, 0.479425538604203, 0.0, 0.8775825618903728, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()

    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_angular(np.array([1.0, 1.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )
    sys = el.six_dof(1.0 / 120.0)
    exec = w.build(sys)
    for _ in range(120):
        exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    print(x.to_numpy()[0])
    # value from Julia and Simulink
    assert np.isclose(
        x.to_numpy()[0],
        np.array(
            [0.45936268493243, 0.45936268493243, 0.0, 0.76024459707606, 0.0, 0.0, 0.0]
        ),
        rtol=1e-5,
    ).all()


def test_six_dof_torque():
    @el.map
    def constant_torque(_: el.Force) -> el.Force:
        return el.SpatialForce.from_torque(np.array([1.0, 0.0, 0.0]))

    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_angular(np.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_angular(np.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0, np.array([0.5, 0.75, 0.25])),
        )
    )
    sys = el.six_dof(1.0 / 120.0, constant_torque)
    exec = w.build(sys)
    for _ in range(120):
        exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    assert np.isclose(
        x.to_numpy()[0],
        np.array([0.24740395925454, 0.0, 0.0, 0.96891242171064, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()  # values taken from simulink
    x = exec.column_array(el.Component.id(el.WorldVel))
    assert np.isclose(
        x.to_numpy()[0], np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), rtol=1e-5
    ).all()  # values taken from simulink

    x = exec.column_array(el.Component.id(el.WorldPos))
    print(x.to_numpy()[1])
    assert np.isclose(
        x.to_numpy()[1],
        np.array([0.47942553860408, 0.0, 0.0, 0.87758256189044, 0.0, 0.0, 0.0]),
        rtol=1e-4,
    ).all()  # values taken from simulink
    x = exec.column_array(el.Component.id(el.WorldVel))
    assert np.isclose(
        x.to_numpy()[1], np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), rtol=1e-5
    ).all()  # values taken from simulink


def test_six_dof_force():
    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel.from_angular(np.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )

    @el.map
    def constant_force(_: el.Force) -> el.Force:
        return el.SpatialForce.from_linear(np.array([1.0, 0.0, 0.0]))

    sys = el.six_dof(1.0 / 120.0, constant_force)
    exec = w.build(sys)
    for _ in range(120):
        exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    assert np.isclose(
        x.to_numpy()[0], np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0]), rtol=1e-5
    ).all()  # values taken from simulink
