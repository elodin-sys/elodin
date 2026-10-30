import elodin as el
import jax.numpy as jnp
import numpy as np
import pytest


def bodies(n: int) -> el.World:
    w = el.World()
    for i in range(n):
        w.spawn(
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([float(i), 0.0, 10.0])),
                world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 1.0])),
            ),
            name=f"b{i}",
        )
    return w


@pytest.mark.parametrize("n", [1, 2])
def test_get_state_by_entity(n: int):
    sim = bodies(n).to_jax(el.six_dof(), simulation_rate=100.0)
    for i in range(n):
        pos = np.asarray(sim.get_state("world_pos", f"b{i}"))
        assert pos.shape == (7,)
        np.testing.assert_allclose(pos[4:], [float(i), 0.0, 10.0])


@pytest.mark.parametrize("n", [1, 2])
def test_set_state_by_entity(n: int):
    sim = bodies(n).to_jax(el.six_dof(), simulation_rate=100.0)
    target = f"b{n - 1}"
    pos = np.asarray(sim.get_state("world_pos", target)).copy()
    pos[6] = 50.0
    sim.set_state("world_pos", target, jnp.asarray(pos))
    np.testing.assert_allclose(np.asarray(sim.get_state("world_pos", target)), pos)

    # The new value feeds the next step: 1 m/s up for 10 ticks at 100 Hz.
    sim.step(10)
    assert np.asarray(sim.get_state("world_pos", target))[6] == pytest.approx(50.1)
    for i in range(n - 1):
        assert np.asarray(sim.get_state("world_pos", f"b{i}"))[6] == pytest.approx(10.1)


def test_set_state_rejects_wrong_shape():
    sim = bodies(1).to_jax(el.six_dof(), simulation_rate=100.0)
    with pytest.raises(Exception, match="does not match"):
        sim.set_state("world_pos", "b0", jnp.zeros(3))


def test_step_keeps_tick_dtype():
    @el.system
    def reads_tick(
        tick: el.Query[el.SimulationTick], pos: el.Query[el.WorldPos]
    ) -> el.Query[el.WorldPos]:
        return pos

    sim = bodies(1).to_jax(reads_tick | el.six_dof(), simulation_rate=100.0)
    dtype = np.asarray(sim.get_state("tick")).dtype
    sim.step(3)
    tick = np.asarray(sim.get_state("tick"))
    assert tick.dtype == dtype
    assert int(tick) == 3


def test_print_dictionary_single_entity(capsys):
    sim = bodies(1).to_jax(el.six_dof(), simulation_rate=100.0)
    sim.print_dictionary()
    assert "world_pos: b0 (shape: (7,))" in capsys.readouterr().out
