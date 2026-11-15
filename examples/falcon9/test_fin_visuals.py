"""Recorded visual deployment and model/actuator binding checks."""

import jax
import jax.numpy as jnp
import numpy as np

from schematic import grid_fin_animations
from sim import FIN_DEPLOY_DURATION_S, SIM_TIME_STEP, fin_deployment_step
from elodin import ui


def test_deployment_starts_after_boostback_and_stays_deployed():
    angle = jnp.zeros(1)
    for phase in range(7):
        np.testing.assert_array_equal(fin_deployment_step(angle, jnp.array([phase])), angle)
    count = round(FIN_DEPLOY_DURATION_S / SIM_TIME_STEP)

    def step(angle, _):
        angle = fin_deployment_step(angle, jnp.array([7.0]))
        return angle, angle

    final, samples = jax.lax.scan(step, angle, None, length=count + 20)
    assert np.all(np.diff(np.asarray(samples[:, 0])) >= 0)
    assert np.all(np.asarray(samples) <= np.pi / 2)
    np.testing.assert_allclose(final, [np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(samples[count // 2 - 1], [np.pi / 4], atol=1e-12)
    np.testing.assert_array_equal(fin_deployment_step(final, jnp.array([11.0])), final)


def test_all_eight_joints_are_bound_to_recorded_state():
    animations = grid_fin_animations()
    assert len(animations) == 8
    text = ui.schematic(
        ui.object_3d("booster.visual_pos", mesh=ui.glb("falcon9-booster.glb"), animate=animations)
    ).emit_kdl()
    for suffix, index in [("", 3), (".001", 2), (".002", 1), (".003", 0)]:
        assert f"grid_fin_deploy{suffix}_f9-v1.2_rig" in text
        assert f"grid_fin_tilt{suffix}_f9-v1.2_rig" in text
        assert f"booster.fin_state[{index}]" in text
    assert text.count("booster.fin_deploy[0]") == 4
    assert "booster.fin_cmd" not in text
