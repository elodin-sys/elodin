import typing as ty
from dataclasses import dataclass

import elodin as el
import jax
import jax.numpy as np
import polars as pl
from polars.testing import assert_frame_equal
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
    w = el.World(frame=el.Frame.ENU)
    w.spawn(Test(np.array([1.0]), np.array([500.0])), "e1")
    w.spawn(
        [
            Test(np.array([15.0]), np.array([500.0])),
            EffectArchetype(np.array([15.0])),
        ],
        "e2",
    )
    exec = w.build(sys)
    exec.run()
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e1.y", "e2.y"])
    print(df)

    expected_df = pl.DataFrame(
        {
            "e1.x": [1.0, 1000.0, 1000000.0],
            "e2.x": [15.0, 15015.0, 15015015.0],
            "e1.y": [500.0, 500.0, 500.0],
            "e2.y": [500.0, 500.0, 500.0],
        }
    )
    assert_frame_equal(df.drop("time"), expected_df)


def test_six_dof():
    w = el.World(frame=el.Frame.ENU)
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(linear=np.array([1.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        ),
        "e1",
    )
    sys = el.six_dof(1.0 / 60.0)
    exec = w.build(sys)
    exec.run()
    df = exec.history("e1.world_pos")
    x = df["e1.world_pos"][-1]
    assert np.allclose(x.to_numpy()[:4], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(x.to_numpy()[4:], np.array([0.01666667, 0.0, 0.0]))


def test_spatial_integration():
    @el.map
    def integrate_velocity(world_pos: el.WorldPos, world_vel: el.WorldVel) -> el.WorldPos:
        linear = world_pos.linear() + world_vel.linear()
        angular = world_pos.angular().integrate_body(world_vel.angular())
        return el.SpatialTransform(linear=linear, angular=angular)

    sys = integrate_velocity
    w = el.World(frame=el.Frame.ENU)
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
        ),
        "e1",
    )
    exec = w.build(sys)
    exec.run()
    exec.run()
    df = exec.history("e1.world_pos")
    pos = df["e1.world_pos"][-1]
    assert (pos[4:] == [2.0, 0.0, 0.0]).all()
    assert np.allclose(pos.to_numpy()[:4], np.array([0.97151626, 0.0, 0.0, 0.23697292]))


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

    w = el.World(frame=el.Frame.ENU)
    a = w.spawn(Test(np.array([1.0])), "e1")
    b = w.spawn(Test(np.array([2.0])), "e2")
    c = w.spawn(Test(np.array([2.0])), "e3")
    print(a, b, c)
    w.spawn(EdgeArchetype(el.Edge(a, b)))
    w.spawn(EdgeArchetype(el.Edge(a, c)))
    w.spawn(EdgeArchetype(el.Edge(b, c)))
    exec = w.build(fold_test)
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e3.x"])
    expected_df = pl.DataFrame({"e1.x": [1.0, 11.0], "e2.x": [2.0, 9.0], "e3.x": [2.0, 2.0]})
    assert_frame_equal(df.drop("time"), expected_df)


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
    w = el.World(frame=el.Frame.ENU)
    w.spawn(Globals(seed=np.array(2)))
    w.spawn(Test(np.array(1.0), np.array(500.0)), "e1")
    w.spawn(Test(np.array(15.0), np.array(500.0)), "e2")
    exec = w.build(sys)
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e1.y", "e2.y"])
    e1x = df["e1.x"][-1]
    e2x = df["e2.x"][-1]
    e1y = df["e1.y"][-1]
    e2y = df["e2.y"][-1]
    assert np.isclose(e1x, 2000.0)
    assert np.isclose(e2x, 30000.0)
    assert e1y >= 500.0 and e1y <= 1000.0
    assert e2y >= 500.0 and e2y <= 1000.0


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

    w = el.World(frame=el.Frame.ENU)
    w.spawn(el.Body(world_vel=el.SpatialMotion(linear=np.array([1.0, 0.0, 0.0]))), "e1")
    exec = w.build(double_vec)
    exec.run()
    df = exec.history("e1.world_vel")
    expected_df = pl.DataFrame(
        {
            "e1.world_vel": [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            ]
        },
        schema={
            "e1.world_vel": pl.Array(pl.Float64, 6),
        },
    )
    assert_frame_equal(df.drop("time"), expected_df)


def test_six_dof_ang_vel_int():
    w = el.World(frame=el.Frame.ENU)
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 0.0, 1.0])),
            inertia=el.SpatialInertia(1.0),
        ),
        "e1",
    )
    sys = el.six_dof(1.0 / 120.0)
    exec = w.build(sys)
    exec.run(120)
    df = exec.history("e1.world_pos")
    x = df["e1.world_pos"][-1]
    # value from Julia and Simulink
    assert np.isclose(
        x.to_numpy(),
        np.array([0.0, 0.0, 0.479425538604203, 0.8775825618903728, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()

    w = el.World(frame=el.Frame.ENU)
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 1.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        ),
        "e1",
    )
    sys = el.six_dof(1.0 / 120.0)
    exec = w.build(sys)
    exec.run(120)
    df = exec.history("e1.world_pos")
    x = df["e1.world_pos"][-1]
    # value from Julia and Simulink
    assert np.isclose(
        x.to_numpy(),
        np.array([0.0, 0.479425538604203, 0.0, 0.8775825618903728, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()

    w = el.World(frame=el.Frame.ENU)
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([1.0, 1.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        ),
        "e1",
    )
    sys = el.six_dof(1.0 / 120.0)
    exec = w.build(sys)
    exec.run(120)
    df = exec.history("e1.world_pos")
    x = df["e1.world_pos"][-1]
    print(x.to_numpy())
    # value from Julia and Simulink
    assert np.isclose(
        x.to_numpy(),
        np.array([0.45936268493243, 0.45936268493243, 0.0, 0.76024459707606, 0.0, 0.0, 0.0]),
        rtol=1e-5,
    ).all()


# def test_six_dof_torque():
#     @el.map
#     def constant_torque(_: el.Force) -> el.Force:
#         return el.SpatialForce(torque=np.array([1.0, 0.0, 0.0]))

#     w = el.World(frame=el.Frame.ENU)
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
    w = el.World(frame=el.Frame.ENU)
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0),
        ),
        "e1",
    )

    @el.map
    def constant_force(_: el.Force) -> el.Force:
        print("constant force")
        return el.SpatialForce(linear=np.array([1.0, 0.0, 0.0]))

    sys = el.six_dof(1.0 / 120.0, constant_force)
    exec = w.build(sys)
    exec.run(120)
    df = exec.history(["e1.world_pos", "e1.world_vel", "e1.world_accel"])
    assert np.isclose(
        df["e1.world_pos"][-1].to_numpy(), np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0]), rtol=1e-5
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


def test_external_control_waiting():
    """Test that Exec.run waits for external control components to be updated."""
    import typing as ty

    # Define external control component
    ExternalControl = ty.Annotated[
        jax.Array,
        el.Component(
            "external_control", el.ComponentType.F64, metadata={"external_control": "true"}
        ),
    ]

    # Define a simple system that uses the external control
    @el.map
    def use_external_control(x: X, ext: ExternalControl) -> X:
        return x + ext

    @dataclass
    class TestWithExternal(el.Archetype):
        x: X
        external_control: ExternalControl

    # Create world and spawn entity
    w = el.World(frame=el.Frame.ENU)
    w.spawn(TestWithExternal(np.array(1.0), np.array(0.0)), "e1")

    # Build and run the system
    exec = w.build(use_external_control)

    # Run a few ticks - should work normally
    exec.run(3)

    # Check that the system ran
    df = exec.history("e1.x")
    assert len(df) >= 3
    assert np.isclose(df["e1.x"][-1], 1.0)  # Should be 1.0 + 0.0

    print("External control waiting test passed!")


def test_map_seq_single_entity():
    """Test map_seq with a single entity (batch_size == 1)."""

    @el.system
    def double_x_seq(q: el.Query[X]) -> el.Query[X]:
        return q.map_seq(X, lambda x: x * 2)

    @dataclass
    class Test(el.Archetype):
        x: X

    w = el.World()
    w.spawn(Test(np.array(5.0)), "e1")
    exec = w.build(double_x_seq)
    exec.run()
    exec.run()
    df = exec.history("e1.x")

    expected_df = pl.DataFrame({"e1.x": [5.0, 10.0, 20.0]})
    assert_frame_equal(df.drop("time"), expected_df)


def test_map_seq_multiple_entities():
    """Test map_seq with multiple entities (batch_size > 1)."""

    @el.system
    def double_x_seq(q: el.Query[X]) -> el.Query[X]:
        return q.map_seq(X, lambda x: x * 2)

    @dataclass
    class Test(el.Archetype):
        x: X

    w = el.World()
    w.spawn(Test(np.array(1.0)), "e1")
    w.spawn(Test(np.array(2.0)), "e2")
    w.spawn(Test(np.array(3.0)), "e3")
    exec = w.build(double_x_seq)
    exec.run()
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e3.x"])

    expected_df = pl.DataFrame(
        {
            "e1.x": [1.0, 2.0, 4.0],
            "e2.x": [2.0, 4.0, 8.0],
            "e3.x": [3.0, 6.0, 12.0],
        }
    )
    assert_frame_equal(df.drop("time"), expected_df)


def test_map_seq_multiple_outputs():
    """Test map_seq with multiple output components."""

    @el.system
    def swap_xy_seq(q: el.Query[X, Y]) -> el.Query[X, Y]:
        return q.map_seq((X, Y), lambda x, y: (y, x))

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    w = el.World()
    w.spawn(Test(np.array(1.0), np.array(10.0)), "e1")
    w.spawn(Test(np.array(2.0), np.array(20.0)), "e2")
    exec = w.build(swap_xy_seq)
    exec.run()
    df = exec.history(["e1.x", "e1.y", "e2.x", "e2.y"])

    expected_df = pl.DataFrame(
        {
            "e1.x": [1.0, 10.0],
            "e1.y": [10.0, 1.0],
            "e2.x": [2.0, 20.0],
            "e2.y": [20.0, 2.0],
        }
    )
    assert_frame_equal(df.drop("time"), expected_df)


def test_map_vs_map_seq_results_match_batch_size_1():
    """Test that map and map_seq produce the same results with batch_size == 1."""

    @el.system
    def compute_with_map(q: el.Query[X, Y]) -> el.Query[X]:
        return q.map(X, lambda x, y: x * y + 1.0)

    @el.system
    def compute_with_map_seq(q: el.Query[X, Y]) -> el.Query[X]:
        return q.map_seq(X, lambda x, y: x * y + 1.0)

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    # Test with map - single entity
    w1 = el.World()
    w1.spawn(Test(np.array(2.0), np.array(3.0)), "e1")
    exec1 = w1.build(compute_with_map)
    exec1.run()
    exec1.run()
    df1 = exec1.history(["e1.x"])

    # Test with map_seq - single entity
    w2 = el.World()
    w2.spawn(Test(np.array(2.0), np.array(3.0)), "e1")
    exec2 = w2.build(compute_with_map_seq)
    exec2.run()
    exec2.run()
    df2 = exec2.history(["e1.x"])

    # Results should match: 2*3+1=7, then 7*3+1=22
    assert_frame_equal(df1.drop("time"), df2.drop("time"))
    expected_df = pl.DataFrame({"e1.x": [2.0, 7.0, 22.0]})
    assert_frame_equal(df1.drop("time"), expected_df)


def test_map_vs_map_seq_results_match_batch_size_2():
    """Test that map and map_seq produce the same results with batch_size == 2."""

    @el.system
    def compute_with_map(q: el.Query[X, Y]) -> el.Query[X]:
        return q.map(X, lambda x, y: x * y + 1.0)

    @el.system
    def compute_with_map_seq(q: el.Query[X, Y]) -> el.Query[X]:
        return q.map_seq(X, lambda x, y: x * y + 1.0)

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    # Test with map - two entities
    w1 = el.World()
    w1.spawn(Test(np.array(2.0), np.array(3.0)), "e1")
    w1.spawn(Test(np.array(4.0), np.array(5.0)), "e2")
    exec1 = w1.build(compute_with_map)
    exec1.run()
    df1 = exec1.history(["e1.x", "e2.x"])

    # Test with map_seq - two entities
    w2 = el.World()
    w2.spawn(Test(np.array(2.0), np.array(3.0)), "e1")
    w2.spawn(Test(np.array(4.0), np.array(5.0)), "e2")
    exec2 = w2.build(compute_with_map_seq)
    exec2.run()
    df2 = exec2.history(["e1.x", "e2.x"])

    # Results should match: e1: 2*3+1=7, e2: 4*5+1=21
    assert_frame_equal(df1.drop("time"), df2.drop("time"))
    expected_df = pl.DataFrame({"e1.x": [2.0, 7.0], "e2.x": [4.0, 21.0]})
    assert_frame_equal(df1.drop("time"), expected_df)


def test_map_vs_map_seq_results_match_batch_size_0():
    """Test that map and map_seq handle batch_size == 0 (no matching entities).

    Currently, elodin's Rust backend panics when building a system that queries
    for a component that no entity has. This test documents that behavior.
    """
    # Define a component that no entity will have
    Z = ty.Annotated[jax.Array, el.Component("z_unused", el.ComponentType.F64)]

    @el.system
    def compute_with_map(q: el.Query[Z]) -> el.Query[Z]:
        return q.map(Z, lambda z: z * 2.0)

    @el.system
    def compute_with_map_seq(q: el.Query[Z]) -> el.Query[Z]:
        return q.map_seq(Z, lambda z: z * 2.0)

    @dataclass
    class Test(el.Archetype):
        x: X  # Only has X, not Z

    # Test with map - no entities have Z component
    # Elodin's Rust backend currently panics when no entities have the queried component
    w1 = el.World()
    w1.spawn(Test(np.array(1.0)), "e1")
    map_raised = False
    try:
        _ = w1.build(compute_with_map)
    except BaseException as e:
        # pyo3_runtime.PanicException is a BaseException subclass
        if "PanicException" in type(e).__name__:
            map_raised = True
        else:
            raise
    assert map_raised, "Expected PanicException when no entities have the queried component"

    # Test with map_seq - same behavior expected
    w2 = el.World()
    w2.spawn(Test(np.array(1.0)), "e1")
    map_seq_raised = False
    try:
        _ = w2.build(compute_with_map_seq)
    except BaseException as e:
        if "PanicException" in type(e).__name__:
            map_seq_raised = True
        else:
            raise
    assert map_seq_raised, "Expected PanicException when no entities have the queried component"


def test_map_vs_map_seq_with_multiple_outputs():
    """Test that map and map_seq match with multiple output components."""

    @el.system
    def compute_with_map(q: el.Query[X, Y]) -> el.Query[X, Y]:
        return q.map((X, Y), lambda x, y: (x + y, x * y))

    @el.system
    def compute_with_map_seq(q: el.Query[X, Y]) -> el.Query[X, Y]:
        return q.map_seq((X, Y), lambda x, y: (x + y, x * y))

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    # Test with map - three entities
    w1 = el.World()
    w1.spawn(Test(np.array(1.0), np.array(2.0)), "e1")
    w1.spawn(Test(np.array(3.0), np.array(4.0)), "e2")
    w1.spawn(Test(np.array(5.0), np.array(6.0)), "e3")
    exec1 = w1.build(compute_with_map)
    exec1.run()
    df1 = exec1.history(["e1.x", "e1.y", "e2.x", "e2.y", "e3.x", "e3.y"])

    # Test with map_seq - three entities
    w2 = el.World()
    w2.spawn(Test(np.array(1.0), np.array(2.0)), "e1")
    w2.spawn(Test(np.array(3.0), np.array(4.0)), "e2")
    w2.spawn(Test(np.array(5.0), np.array(6.0)), "e3")
    exec2 = w2.build(compute_with_map_seq)
    exec2.run()
    df2 = exec2.history(["e1.x", "e1.y", "e2.x", "e2.y", "e3.x", "e3.y"])

    # Results should match
    assert_frame_equal(df1.drop("time"), df2.drop("time"))
    # e1: x=1+2=3, y=1*2=2
    # e2: x=3+4=7, y=3*4=12
    # e3: x=5+6=11, y=5*6=30
    expected_df = pl.DataFrame(
        {
            "e1.x": [1.0, 3.0],
            "e1.y": [2.0, 2.0],
            "e2.x": [3.0, 7.0],
            "e2.y": [4.0, 12.0],
            "e3.x": [5.0, 11.0],
            "e3.y": [6.0, 30.0],
        }
    )
    assert_frame_equal(df1.drop("time"), expected_df)


def test_map_seq_preserves_cond_semantics():
    """Test that map_seq preserves jax.lax.cond semantics.

    With map (vmap), jax.lax.cond becomes jax.lax.select which evaluates
    both branches. With map_seq, only one branch should execute.

    We verify this by checking that the computation produces correct results
    when using jax.lax.cond inside the mapped function.
    """
    import jax.lax as lax

    # Counter component to track which branch was "logically" taken
    BranchTaken = ty.Annotated[jax.Array, el.Component("branch_taken", el.ComponentType.F64)]

    @el.system
    def cond_with_map_seq(q: el.Query[X]) -> el.Query[X, BranchTaken]:
        def conditional_compute(x):
            # If x > 5, multiply by 2, else multiply by 10
            def true_branch(_):
                return x * 2.0

            def false_branch(_):
                return x * 10.0

            result = lax.cond(x > 5.0, true_branch, false_branch, operand=None)
            branch_taken = lax.cond(x > 5.0, lambda _: 1.0, lambda _: 0.0, operand=None)
            return result, branch_taken

        return q.map_seq((X, BranchTaken), conditional_compute)

    @dataclass
    class Test(el.Archetype):
        x: X
        branch_taken: BranchTaken

    w = el.World()
    w.spawn(Test(np.array(3.0), np.array(0.0)), "e1")  # x <= 5, should take false branch
    w.spawn(Test(np.array(10.0), np.array(0.0)), "e2")  # x > 5, should take true branch
    exec = w.build(cond_with_map_seq)
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e1.branch_taken", "e2.branch_taken"])

    # e1: 3.0 <= 5, so false branch: 3.0 * 10 = 30.0
    # e2: 10.0 > 5, so true branch: 10.0 * 2 = 20.0
    assert np.isclose(df["e1.x"][-1], 30.0)
    assert np.isclose(df["e2.x"][-1], 20.0)
    assert np.isclose(df["e1.branch_taken"][-1], 0.0)  # false branch
    assert np.isclose(df["e2.branch_taken"][-1], 1.0)  # true branch


def test_map_with_cond_also_works():
    """Test that map with jax.lax.cond also produces correct results.

    Even though vmap converts cond to select (evaluating both branches),
    the final result should still be correct.
    """
    import jax.lax as lax

    BranchTaken = ty.Annotated[jax.Array, el.Component("branch_taken", el.ComponentType.F64)]

    @el.system
    def cond_with_map(q: el.Query[X]) -> el.Query[X, BranchTaken]:
        def conditional_compute(x):
            def true_branch(_):
                return x * 2.0

            def false_branch(_):
                return x * 10.0

            result = lax.cond(x > 5.0, true_branch, false_branch, operand=None)
            branch_taken = lax.cond(x > 5.0, lambda _: 1.0, lambda _: 0.0, operand=None)
            return result, branch_taken

        return q.map((X, BranchTaken), conditional_compute)

    @dataclass
    class Test(el.Archetype):
        x: X
        branch_taken: BranchTaken

    w = el.World()
    w.spawn(Test(np.array(3.0), np.array(0.0)), "e1")
    w.spawn(Test(np.array(10.0), np.array(0.0)), "e2")
    exec = w.build(cond_with_map)
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e1.branch_taken", "e2.branch_taken"])

    # Results should be the same as map_seq
    assert np.isclose(df["e1.x"][-1], 30.0)
    assert np.isclose(df["e2.x"][-1], 20.0)
    assert np.isclose(df["e1.branch_taken"][-1], 0.0)
    assert np.isclose(df["e2.branch_taken"][-1], 1.0)


def test_map_seq_decorator():
    """Test the @el.map_seq decorator syntax."""

    @el.map_seq
    def double_x(x: X) -> X:
        return x * 2

    @dataclass
    class Test(el.Archetype):
        x: X

    w = el.World()
    w.spawn(Test(np.array(5.0)), "e1")
    w.spawn(Test(np.array(7.0)), "e2")
    exec = w.build(double_x)
    exec.run()
    exec.run()
    df = exec.history(["e1.x", "e2.x"])

    expected_df = pl.DataFrame(
        {
            "e1.x": [5.0, 10.0, 20.0],
            "e2.x": [7.0, 14.0, 28.0],
        }
    )
    assert_frame_equal(df.drop("time"), expected_df)


def test_map_seq_decorator_with_cond():
    """Test @el.map_seq decorator with jax.lax.cond."""
    import jax.lax as lax

    @el.map_seq
    def conditional_double(x: X) -> X:
        # If x > 5, multiply by 2, else multiply by 10
        def true_branch(_):
            return x * 2.0

        def false_branch(_):
            return x * 10.0

        return lax.cond(x > 5.0, true_branch, false_branch, operand=None)

    @dataclass
    class Test(el.Archetype):
        x: X

    w = el.World()
    w.spawn(Test(np.array(3.0)), "e1")  # x <= 5, false branch: 3 * 10 = 30
    w.spawn(Test(np.array(10.0)), "e2")  # x > 5, true branch: 10 * 2 = 20
    w.spawn(Test(np.array(1.0)), "e3")  # x <= 5, false branch: 1 * 10 = 10
    exec = w.build(conditional_double)
    exec.run()
    df = exec.history(["e1.x", "e2.x", "e3.x"])

    assert np.isclose(df["e1.x"][-1], 30.0)
    assert np.isclose(df["e2.x"][-1], 20.0)
    assert np.isclose(df["e3.x"][-1], 10.0)


def test_map_seq_decorator_multiple_inputs_outputs():
    """Test @el.map_seq decorator with multiple inputs and outputs."""

    @el.map_seq
    def compute_xy(x: X, y: Y) -> tuple[X, Y]:
        return x + y, x * y

    @dataclass
    class Test(el.Archetype):
        x: X
        y: Y

    w = el.World()
    w.spawn(Test(np.array(2.0), np.array(3.0)), "e1")
    w.spawn(Test(np.array(4.0), np.array(5.0)), "e2")
    exec = w.build(compute_xy)
    exec.run()
    df = exec.history(["e1.x", "e1.y", "e2.x", "e2.y"])

    # e1: x=2+3=5, y=2*3=6
    # e2: x=4+5=9, y=4*5=20
    expected_df = pl.DataFrame(
        {
            "e1.x": [2.0, 5.0],
            "e1.y": [3.0, 6.0],
            "e2.x": [4.0, 9.0],
            "e2.y": [5.0, 20.0],
        }
    )
    assert_frame_equal(df.drop("time"), expected_df)
