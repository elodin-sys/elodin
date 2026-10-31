import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import pytest

Gain = ty.Annotated[jax.Array, el.Component("jaxsim_gain", el.ComponentType.F64)]
Total = ty.Annotated[jax.Array, el.Component("jaxsim_total", el.ComponentType.F64)]
Ticks = ty.Annotated[jax.Array, el.Component("jaxsim_ticks", el.ComponentType.F64)]


@dataclass
class Counter(el.Archetype):
    gain: Gain = field(default_factory=lambda: jnp.float64(2.0))
    total: Total = field(default_factory=lambda: jnp.float64(0.0))
    ticks: Ticks = field(default_factory=lambda: jnp.float64(0.0))


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


def test_components_no_system_writes_carry_over():
    @el.map
    def accumulate(g: Gain, t: Total) -> Total:
        return t + g

    @el.system
    def count_ticks(tick: el.Query[el.SimulationTick], t: el.Query[Ticks]) -> el.Query[Ticks]:
        return t.map(Ticks, lambda _: tick[0].astype(jnp.float64))

    w = el.World()
    w.spawn(Counter(), "c")
    sim = w.to_jax(accumulate | count_ticks, simulation_rate=100.0)
    sim.step(3)
    assert float(sim.get_state("jaxsim_gain", "c")) == 2.0
    assert float(sim.get_state("jaxsim_total", "c")) == 6.0
    assert float(sim.get_state("jaxsim_ticks", "c")) == 3.0


def test_step_fn_under_jit_vmap_and_scan():
    sim = bodies(2).to_jax(el.six_dof(), simulation_rate=100.0)
    initial = list(sim.state)
    heights = jnp.array([10.0, 20.0, 30.0])

    def start(z):
        pos = sim.get(initial, "world_pos", "b1")
        return sim.set(initial, "world_pos", "b1", pos.at[6].set(z))

    states = jax.jit(jax.vmap(sim.step_fn))(jax.vmap(start)(heights))
    z = jax.vmap(lambda s: sim.get(s, "world_pos", "b1")[6])(states)
    np.testing.assert_allclose(z, heights + 0.01)  # 1 m/s up for one tick at 100 Hz

    scanned = jax.lax.scan(lambda s, _: (sim.step_fn(s), None), initial, None, length=10)[0]
    for a, b in zip(sim.state, initial):
        np.testing.assert_array_equal(a, b)  # pure: sim.state is untouched
    sim.step(10)
    for a, b in zip(scanned, sim.state):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_set_without_entity_writes_every_entity():
    sim = bodies(2).to_jax(el.six_dof(), simulation_rate=100.0)
    pos = np.asarray(sim.get_state("world_pos")).copy()
    pos[:, 6] = [1.0, 2.0]
    sim.set_state("world_pos", None, pos)
    np.testing.assert_allclose(np.asarray(sim.get_state("world_pos", "b1"))[6], 2.0)
