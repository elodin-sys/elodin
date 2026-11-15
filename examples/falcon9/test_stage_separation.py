"""The second-stage scenery detaches continuously without driving the plant."""

import jax.numpy as jnp
import numpy as np

import elodin as el
from frames import frame_accel, gravity_accel
from sim import SIM_TIME_STEP, rebase_separation_center, upper_stage_visual_step


def test_attached_upper_stage_matches_stack_pose():
    pose = el.SpatialTransform(linear=jnp.array([6_400_000.0, 20.0, 30.0]))
    velocity = el.SpatialMotion(linear=jnp.array([100.0, 200.0, 300.0]))
    result, drift, detached = upper_stage_visual_step(
        pose, velocity, jnp.ones(1), pose, jnp.zeros(3), jnp.zeros(1)
    )
    np.testing.assert_array_equal(result.linear(), pose.linear())
    np.testing.assert_array_equal(result.angular().vector(), pose.angular().vector())
    np.testing.assert_array_equal(drift, velocity.linear())
    assert detached[0] == 0.0


def test_detachment_retains_pose_and_applies_push_only_once():
    previous = el.SpatialTransform(linear=jnp.array([6_400_000.0, 20.0, 30.0]))
    # A later booster pose/velocity must not drag the detached stage along.
    booster = el.SpatialTransform(
        angular=el.Quaternion.from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi),
        linear=jnp.array([-1000.0, -2000.0, -3000.0]),
    )
    velocity = el.SpatialMotion(linear=jnp.array([-100.0, 0.0, 0.0]))
    drift = jnp.array([100.0, 200.0, 300.0])
    result, drift, detached = upper_stage_visual_step(
        booster, velocity, jnp.zeros(1), previous, drift, jnp.zeros(1)
    )
    released = jnp.array([101.0, 200.0, 300.0])
    expected = (
        released
        + (gravity_accel(previous.linear()) + frame_accel(previous.linear(), released))
        * SIM_TIME_STEP
    )
    np.testing.assert_allclose(drift, expected)
    np.testing.assert_allclose(
        result.linear(), previous.linear() + 0.5 * (released + drift) * SIM_TIME_STEP
    )
    np.testing.assert_array_equal(result.angular().vector(), previous.angular().vector())
    assert detached[0] == 1.0
    second, next_drift, _ = upper_stage_visual_step(
        booster, velocity, jnp.zeros(1), result, drift, detached
    )
    expected = (
        drift
        + (gravity_accel(result.linear()) + frame_accel(result.linear(), drift)) * SIM_TIME_STEP
    )
    np.testing.assert_allclose(next_drift, expected)
    assert next_drift[0] < drift[0]  # No second push; radial velocity decelerates.
    assert second.linear()[0] > result.linear()[0]


def test_separation_center_change_preserves_material_positions_and_velocities():
    pose = el.SpatialTransform(
        angular=el.Quaternion.from_axis_angle(jnp.array([0.0, 1.0, 0.0]), 0.7),
        linear=jnp.array([6_400_000.0, 20.0, 30.0]),
    )
    velocity = el.SpatialMotion(
        linear=jnp.array([100.0, 200.0, 300.0]), angular=jnp.array([0.01, -0.02, 0.03])
    )
    old_cg, new_cg = 45.0, 20.0
    rebased, rebased_velocity = rebase_separation_center(pose, velocity, old_cg, new_cg, True)
    for station in [0.0, 37.0, 58.0]:
        old_offset = pose.angular() @ jnp.array([station - old_cg, 0.0, 0.0])
        new_offset = rebased.angular() @ jnp.array([station - new_cg, 0.0, 0.0])
        np.testing.assert_allclose(
            rebased.linear() + new_offset, pose.linear() + old_offset, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            rebased_velocity.linear() + jnp.cross(velocity.angular(), new_offset),
            velocity.linear() + jnp.cross(velocity.angular(), old_offset),
        )
    unchanged, unchanged_velocity = rebase_separation_center(pose, velocity, old_cg, new_cg, False)
    np.testing.assert_array_equal(unchanged.linear(), pose.linear())
    np.testing.assert_array_equal(unchanged_velocity.linear(), velocity.linear())
