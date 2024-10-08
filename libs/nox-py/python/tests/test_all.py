import typing as ty
from dataclasses import dataclass

import elodin as el
import jax
import jax.numpy as np
from elodin import ukf
from jax import random

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
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(linear=np.array([1.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )
    sys = el.six_dof(1.0 / 60.0)
    exec = w.build(sys)
    exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    assert np.allclose(x.to_numpy()[0][:4], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(x.to_numpy()[0][4:], np.array([0.01666667, 0.0, 0.0]))


def test_spatial_integration():
    @el.map
    def integrate_velocity(world_pos: el.WorldPos, world_vel: el.WorldVel) -> el.WorldPos:
        linear = world_pos.linear() + world_vel.linear()
        angular = world_pos.angular().integrate_body(world_vel.angular())
        return el.SpatialTransform(linear=linear, angular=angular)

    sys = integrate_velocity
    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(
                linear=np.array([0.0, 0.0, 0.0]),
            ),
            world_vel=el.SpatialMotion(
                linear=np.array([1.0, 0.0, 0.0]),
                angular=np.array([np.pi / 2, 0.0, 0.0]),
            ),
            inertia=el.SpatialInertia(1.0),
        )
    )
    exec = w.build(sys)
    exec.run()
    exec.run()
    pos = exec.column_array(el.Component.name(el.WorldPos))
    assert (pos[4:] == [2.0, 0.0, 0.0]).all()
    assert np.allclose(pos.to_numpy()[0][:4], np.array([0.97151626, 0.0, 0.0, 0.23697292]))


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
    w.spawn(el.Body(world_vel=el.SpatialMotion(linear=np.array([1.0, 0.0, 0.0]))))
    exec = w.build(double_vec)
    exec.run()
    v = exec.column_array(el.Component.id(el.WorldVel))
    assert (v[0][3:] == [2.0, 0.0, 0.0]).all()


def test_six_dof_ang_vel_int():
    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 0.0, 1.0])),
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
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 1.0, 0.0])),
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
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([1.0, 1.0, 0.0])),
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
        np.array([0.45936268493243, 0.45936268493243, 0.0, 0.76024459707606, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()


# def test_six_dof_torque():
#     @el.map
#     def constant_torque(_: el.Force) -> el.Force:
#         return el.SpatialForce(torque=np.array([1.0, 0.0, 0.0]))

#     w = el.World()
#     w.spawn(
#         el.Body(
#             world_pos=el.WorldPos(linear=np.array([0.0, 0.0, 0.0])),
#             world_vel=el.WorldVel(angular=np.array([0.0, 0.0, 0.0])),
#             inertia=el.SpatialInertia(1.0),
#         )
#     )
#     w.spawn(
#         el.Body(
#             world_pos=el.WorldPos(linear=np.array([0.0, 0.0, 0.0])),
#             world_vel=el.WorldVel(angular=np.array([0.0, 0.0, 0.0])),
#             inertia=el.SpatialInertia(1.0, np.array([0.5, 0.75, 0.25])),
#         )
#     )
#     sys = el.six_dof(1.0 / 120.0, constant_torque)
#     exec = w.build(sys)
#     for _ in range(120):
#         exec.run()
#     x = exec.column_array(el.Component.id(el.WorldPos))
#     assert np.isclose(
#         x.to_numpy()[0],
#         np.array([0.24740395925454, 0.0, 0.0, 0.96891242171064, 0.0, 0.0, 0.0]),
#         rtol=1e-5,
#     ).all()  # values taken from simulink
#     x = exec.column_array(el.Component.id(el.WorldVel))
#     assert np.isclose(
#         x.to_numpy()[0], np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), rtol=1e-5
#     ).all()  # values taken from simulink

#     x = exec.column_array(el.Component.id(el.WorldPos))
#     print(x.to_numpy()[1])
#     assert np.isclose(
#         x.to_numpy()[1],
#         np.array([0.47942553860408, 0.0, 0.0, 0.87758256189044, 0.0, 0.0, 0.0]),
#         rtol=1e-4,
#     ).all()  # values taken from simulink
#     x = exec.column_array(el.Component.id(el.WorldVel))
#     assert np.isclose(
#         x.to_numpy()[1], np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), rtol=1e-5
#     ).all()  # values taken from simulink


def test_six_dof_force():
    w = el.World()
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        )
    )

    @el.map
    def constant_force(_: el.Force) -> el.Force:
        print("constant force")
        return el.SpatialForce(linear=np.array([1.0, 0.0, 0.0]))

    sys = el.six_dof(1.0 / 120.0, constant_force)
    exec = w.build(sys)
    for _ in range(120):
        exec.run()
    x = exec.column_array(el.Component.id(el.WorldPos))
    v = exec.column_array(el.Component.id(el.WorldVel))
    a = exec.column_array(el.Component.id(el.WorldAccel))
    print(x.to_numpy())
    print(v.to_numpy())
    print(a.to_numpy())
    assert np.isclose(
        x.to_numpy()[0], np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0]), rtol=1e-5
    ).all()  # values taken from simulink


def test_skew():
    arr = np.array([1.0, 2.0, 3.0])
    assert np.isclose(
        el.skew(arr),
        np.array(
            [
                [0.0, -3.0, 2.0],
                [3.0, 0.0, -1.0],
                [-2.0, 1.0, 0.0],
            ]
        ),
    ).all()


def test_unscented_transform():
    covar_weights = np.array([0.4, 0.1, 0.1])
    mean_weights = np.array([0.4, 0.1, 0.1])
    points = np.array([[1.0, 2.0], [2.0, 4.0], [5.0, 4.0]])
    x, p = ukf.unscented_transform(points, mean_weights, covar_weights)
    assert np.isclose(x, np.array([1.1, 1.6])).all()
    assert np.isclose(p, np.array([[1.606, 1.136], [1.136, 1.216]])).all()


def test_cross_covariance():
    covar_weights = np.array([0.4, 0.1, 0.1])
    x_hat = np.array([1.0, 2.0])
    z_hat = np.array([2.0, 3.0])
    points_x = np.array([[1.0, 2.0], [2.0, 4.0], [5.0, 4.0]])
    points_z = np.array([[2.0, 3.0], [3.0, 5.0], [6.0, 5.0]])

    p = ukf.cross_covar(x_hat, z_hat, points_x, points_z, covar_weights)

    assert np.isclose(p, np.array([[1.7, 1.0], [1.0, 0.8]]), rtol=1e-7).all()


def test_ukf_simple_linear():
    z_std = 0.1
    dt = 0.1

    # Initialize UKFState
    x_hat = np.array([-1.0, 1.0, -1.0, 1.0])
    covar = np.eye(4) * 0.02
    prop_covar = np.array(
        [
            [2.5e-09, 5.0e-08, 0.0e00, 0.0e00],
            [5.0e-08, 1.0e-06, 0.0e00, 0.0e00],
            [0.0e00, 0.0e00, 2.5e-09, 5.0e-08],
            [0.0e00, 0.0e00, 5.0e-08, 1.0e-06],
        ]
    )
    noise_covar = np.diag(np.array([z_std**2, z_std**2]))

    state = ukf.UKFState(x_hat, covar, prop_covar, noise_covar, alpha=0.1, beta=2.0, kappa=-1.0)

    # Define propagation and measurement functions
    def prop_fn(x):
        transition = np.array(
            [
                [1.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return transition @ x

    def measure_fn(x, _):
        return np.array([x[0], x[2]])

    # Measurement data
    zs = np.array(
        [
            [5.17725854e-02, -6.26320583e-03],
            [7.76510642e-01, 9.41957633e-01],
            [2.00841988e00, 2.03743906e00],
            [2.99844618e00, 3.11297309e00],
            [4.02493837e00, 4.18458527e00],
            [5.15192102e00, 4.94829940e00],
            [5.97820965e00, 6.19408601e00],
            [6.90700638e00, 7.09993104e00],
            [8.16174727e00, 7.84922020e00],
            [9.07054319e00, 9.02949139e00],
            [1.00943927e01, 1.02153963e01],
            [1.10528857e01, 1.09655620e01],
            [1.20433517e01, 1.19606005e01],
            [1.30130301e01, 1.30389662e01],
            [1.39492112e01, 1.38780037e01],
            [1.50232252e01, 1.50704141e01],
            [1.59498538e01, 1.60893516e01],
            [1.70097415e01, 1.70585561e01],
            [1.81292609e01, 1.80656253e01],
            [1.90783022e01, 1.89139529e01],
            [1.99490761e01, 1.99682328e01],
            [2.10253265e01, 2.09926241e01],
            [2.18166124e01, 2.19475433e01],
            [2.29619247e01, 2.29313189e01],
            [2.40366414e01, 2.40207406e01],
            [2.50164997e01, 2.50594340e01],
            [2.60602065e01, 2.59104916e01],
            [2.68926856e01, 2.68682419e01],
            [2.81448564e01, 2.81699908e01],
            [2.89114209e01, 2.90161936e01],
            [2.99632302e01, 3.01334351e01],
            [3.09547757e01, 3.09778803e01],
            [3.20168683e01, 3.19300419e01],
            [3.28656686e01, 3.30364708e01],
            [3.39697008e01, 3.40282794e01],
            [3.50369500e01, 3.51215329e01],
            [3.59710004e01, 3.61372108e01],
            [3.70591558e01, 3.69247502e01],
            [3.80522440e01, 3.78498751e01],
            [3.90272805e01, 3.90100329e01],
            [4.00377740e01, 3.98368033e01],
            [4.08131455e01, 4.09728212e01],
            [4.18214855e01, 4.19909894e01],
            [4.31312654e01, 4.29949226e01],
            [4.40398607e01, 4.38911788e01],
            [4.50978163e01, 4.49942054e01],
            [4.61407950e01, 4.59709971e01],
            [4.69322204e01, 4.71080633e01],
            [4.80521531e01, 4.81292422e01],
            [4.91282949e01, 4.90346729e01],
        ]
    )

    # Update state with measurements
    for z in zs:
        state.update(z, prop_fn, measure_fn)

    # Check final state estimate
    expected_x_hat = np.array([48.9118168, 9.96293597, 48.89106226, 9.95283274])
    assert np.isclose(state.x_hat, expected_x_hat, rtol=1e-6).all()
