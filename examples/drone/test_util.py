import elodin as el
import jax.numpy as jnp

import util


def test_q_dist():
    q1 = el.Quaternion.from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.0)
    q2 = el.Quaternion.from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 1.0)
    dist = util.quat_dist(q1, q2)
    assert jnp.isclose(dist, 1.0, atol=1e-6)


def test_skew_symmetric():
    v = jnp.array([1.0, 2.0, 3.0])
    skew = util.skew_symmetric(v)
    expected_skew = jnp.array(
        [
            [0.0, -3.0, 2.0],
            [3.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0],
        ]
    )
    assert jnp.all(jnp.isclose(skew, expected_skew, atol=1e-6))


def test_quat_to_matrix():
    q1 = el.Quaternion.from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)
    mat = util.quat_to_matrix(q1)
    expected_mat = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    assert jnp.all(jnp.isclose(mat, expected_mat, atol=1e-6))

    q2 = el.Quaternion.from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi)
    mat = util.quat_to_matrix(q2)
    expected_mat = jnp.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    assert jnp.all(jnp.isclose(mat, expected_mat, atol=1e-6))


def check_quat_euler_conversion(axis, angle, euler_expected):
    __tracebackhide__ = True
    q_expected = el.Quaternion.from_axis_angle(axis, angle)
    euler = util.quat_to_euler(q_expected)
    assert jnp.all(jnp.isclose(euler, euler_expected, atol=1e-6))
    q = util.euler_to_quat(euler)
    assert jnp.all(jnp.isclose(q.vector(), q_expected.vector(), atol=1e-6))


def test_quat_euler_conversion():
    check_quat_euler_conversion(
        jnp.array([1.0, 0.0, 0.0]), 0.0, jnp.array([0.0, 0.0, 0.0])
    )
    check_quat_euler_conversion(
        jnp.array([1.0, 0.0, 0.0]), jnp.pi / 2, jnp.array([jnp.pi / 2, 0.0, 0.0])
    )
    check_quat_euler_conversion(
        jnp.array([0.0, 1.0, 0.0]), jnp.pi / 2, jnp.array([0.0, jnp.pi / 2, 0.0])
    )
    check_quat_euler_conversion(
        jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2, jnp.array([0.0, 0.0, jnp.pi / 2])
    )


def test_quat_to_axis_angle():
    q = el.Quaternion.from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)
    axis_angle = util.quat_to_axis_angle(q)
    assert jnp.all(jnp.isclose(axis_angle, jnp.array([1.0, 0.0, 0.0]) * jnp.pi))

    q = el.Quaternion.from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi / 2)
    axis_angle = util.quat_to_axis_angle(q)
    assert jnp.all(jnp.isclose(axis_angle, jnp.array([0.0, 1.0, 0.0]) * jnp.pi / 2))


def test_angular_euler_rate_conversion():
    att = el.Quaternion.identity()
    angular_rate = jnp.array([1.0, 2.0, 3.0])
    euler_rate = util.angular_to_euler_rate(att, angular_rate)
    assert jnp.all(jnp.isclose(euler_rate, angular_rate, atol=1e-6))
    angular_rate_recovered = util.euler_to_angular_rate(att, euler_rate)
    assert jnp.all(jnp.isclose(angular_rate_recovered, angular_rate, atol=1e-6))

    att = el.Quaternion.from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi / 2)
    angular_rate = jnp.array([0.0, 2.0, 1.0])
    euler_rate = util.angular_to_euler_rate(att, angular_rate)
    assert jnp.all(jnp.isclose(euler_rate, jnp.array([0.0, -1.0, 2.0]), atol=1e-6))
    angular_rate_recovered = util.euler_to_angular_rate(att, euler_rate)
    assert jnp.all(jnp.isclose(angular_rate_recovered, angular_rate, atol=1e-6))


def test_normalize_angle():
    angle, expected = 3 * jnp.pi, jnp.pi
    assert jnp.isclose(util.normalize_angle(angle), expected, atol=1e-6)

    angle, expected = -3 * jnp.pi, jnp.pi
    assert jnp.isclose(util.normalize_angle(angle), expected, atol=1e-6)

    angle, expected = 0.5 * jnp.pi, 0.5 * jnp.pi
    assert jnp.isclose(util.normalize_angle(angle), expected, atol=1e-6)

    angle, expected = -0.5 * jnp.pi, -0.5 * jnp.pi
    assert jnp.isclose(util.normalize_angle(angle), expected, atol=1e-6)

    angle, expected = 1.5 * jnp.pi, -0.5 * jnp.pi
    assert jnp.isclose(util.normalize_angle(angle), expected, atol=1e-6)
