import numpy as np

from sim import RATE_HZ, build


def rows(frame, name: str) -> np.ndarray:
    return np.stack([np.asarray(row.to_numpy()) for row in frame[name]])


def test_two_by_two_world_runs_real_compiled_path():
    world, system, _, _ = build(2, 2)
    execution = world.build(system, simulation_rate=RATE_HZ)

    execution.run(20, show_progress=False)
    frame = execution.history(
        [
            "agent_000.position",
            "agent_001.position",
            "object_000.object_position",
            "object_001.object_position",
        ]
    )

    first_agent = rows(frame, "agent_000.position")
    second_agent = rows(frame, "agent_001.position")
    first_object = rows(frame, "object_000.object_position")
    assert np.all(np.isfinite(first_agent))
    assert np.all(np.isfinite(first_object))
    assert not np.array_equal(first_agent[0], first_agent[-1])
    assert not np.array_equal(first_object[0], first_object[-1])
    assert not np.array_equal(first_agent[-1], second_agent[-1])


def test_cranelift_and_jax_cpu_agree():
    world, system, _, _ = build(2, 2)
    execution = world.build(system, simulation_rate=RATE_HZ)
    execution.run(20, show_progress=False)
    frame = execution.history(["agent_000.position", "agent_001.position"])
    cranelift = np.stack(
        [
            rows(frame, "agent_000.position")[-1],
            rows(frame, "agent_001.position")[-1],
        ]
    )

    jax_world, jax_system, _, _ = build(2, 2)
    simulation = jax_world.to_jax(jax_system, simulation_rate=RATE_HZ)
    simulation.step(20)
    jax_cpu = np.asarray(simulation.get_state("position")).reshape(2, 3)

    np.testing.assert_allclose(cranelift, jax_cpu, rtol=1e-12, atol=1e-12)
