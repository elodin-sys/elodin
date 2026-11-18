"""Prepared handles and the `pre_commit` hook of `World.run`."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Annotated

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_runtime_thread import _available_port

X = Annotated[jax.Array, el.Component("pc_x", el.ComponentType(el.PrimitiveType.F64, (2,)))]
C = Annotated[jax.Array, el.Component("pc_c", el.ComponentType(el.PrimitiveType.F64, (2,)))]
E = Annotated[
    jax.Array,
    el.Component(
        "pc_e",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]
T = Annotated[
    jax.Array,
    el.Component(
        "pc_t", el.ComponentType(el.PrimitiveType.F64, (1,)), metadata={"transient": "true"}
    ),
]


def _world(entities: int = 3):
    world = el.World()
    ids = [
        world.spawn([el.C(X, jnp.array([1.0, 2.0]) * (i + 1)), el.C(C, jnp.zeros(2))], name=f"e{i}")
        for i in range(entities)
    ]
    return world, ids


@el.map
def _step(x: X, c: C) -> X:
    return x + c


def _scenario(name, result_path, db_path):
    world, ids = _world()
    result = {}
    if name in ("history", "time_travel"):
        world.spawn([el.C(E, jnp.zeros(1))], name="probe")
    if name in ("stale", "replaced"):
        # "replaced" prepares these same rows again below.
        world.prepare_component("pc_x", ids if name == "stale" else list(reversed(ids)))
        world.spawn([el.C(X, jnp.zeros(2)), el.C(C, jnp.zeros(2))], name="late")
    exchange = world.prepare_batch(
        {
            "x": world.prepare_component("pc_x", list(reversed(ids)), access="write")
            if name == "conflict"
            else world.prepare_component("pc_x", list(reversed(ids))),
            "c": world.prepare_component("pc_c", list(reversed(ids)), access="write"),
        }
    )

    def exchange_step(tick, ctx):
        with ctx.batch(exchange) as view:
            view.c[...] = 0.5 * view.x
            result["x"] = view.x.tolist()
            result["tick"] = tick
        if name == "record":
            ctx.record(exchange.fields["c"])
        if name == "failure" and tick == 3:
            raise ValueError("controller went away")

    def after_step(tick, ctx):
        result.setdefault("after", []).append(tick)

    if name in ("history", "time_travel"):
        past_x = world.prepare_component_history("e0.pc_x")
        frames = world.prepare_message_history("pc_frames")
        evidence = world.prepare_component_history("probe.pc_e")
        writer = world.prepare_timestamped_writer({"value": "probe.pc_e"})

    def history_step(tick, ctx):
        with ctx.batch(exchange) as view:
            result["sees_earlier_write"] = bool(np.array_equal(view.c, 0.5 * view.x))
        if tick == 5:
            result["exact_now"] = ctx.read_at(past_x, ctx.timestamp, "exact")
            at, value = ctx.read_at(past_x, ctx.timestamp, "at_or_before")
            result["previous"] = [ctx.timestamp - at, value.tolist()]
            result["frame"] = ctx.read_at(frames, ctx.timestamp, "at_or_before")
            result["last_deferred"] = ctx.read_at(evidence, ctx.timestamp - 999, "exact")[
                1
            ].tolist()
            for bad in (
                lambda: ctx.read_at(past_x, 0, "nearest"),
                lambda: ctx.defer_write(writer, 0, {}),
            ):
                try:
                    bad()
                except ValueError as err:
                    result.setdefault("rejected", []).append(str(err))
        at = 0 if name == "time_travel" and tick == 3 else ctx.timestamp + 1
        ctx.defer_write(writer, at, {"value": 10.0 * tick})
        ctx.defer_write(writer, at, {"value": 10.0 * tick + 1})
        if tick == 5:
            result["pending"] = ctx.read_at(evidence, at, "exact")

    hooks = {
        "not_a_hook": [exchange_step, 42],
        "failure": [exchange_step, after_step],
        "history": [exchange_step, history_step],
        "time_travel": [exchange_step, history_step],
    }

    def on_receipt(receipt):
        result["receipt"] = receipt

    try:
        world.run(
            _step,
            simulation_rate=1000.0,
            max_ticks=12,
            pre_commit=hooks.get(name, exchange_step),
            on_receipt=on_receipt,
            interactive=False,
            db_path=db_path,
            log_level="warn",
            synchronous_commits=name == "synchronous",
        )
    except (ValueError, RuntimeError, TypeError) as err:
        result["error"] = f"{type(err).__name__}: {err}"
    Path(result_path).write_text(json.dumps(result))


def _run(tmp_path, name) -> dict:
    result = tmp_path / "result.json"
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""\
            import runpy, sys
            sys.path.insert(0, {str(Path(__file__).parent)!r})
            probe = runpy.run_path({str(Path(__file__).resolve())!r})
            probe['_scenario']({name!r}, {str(result)!r}, {str(tmp_path / "db")!r})
            """
        )
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("ELODIN_PACING_")}
    completed = subprocess.run(
        [sys.executable, str(probe), "run", f"127.0.0.1:{_available_port()}", "--no-s10"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result.read_text())


def test_commands_written_before_the_commit_drive_the_next_tick(tmp_path):
    result = _run(tmp_path, "exchange")
    # The hook at tick t sees x after t + 1 ticks: x0, then x * 1.5 for every tick after.
    x0 = np.array([[1.0, 2.0]]) * np.arange(3, 0, -1)[:, None]
    np.testing.assert_allclose(result["x"], x0 * 1.5 ** result["tick"], rtol=1e-12)
    receipt = result["receipt"]
    assert receipt["pre_commit"]["count"] == receipt["cycles"]
    assert receipt["callbacks"]["prepared_handles"] == 2
    assert receipt["callbacks"]["views"] == 2 * receipt["cycles"]
    assert receipt["failure"] is None
    assert receipt["sync"]["commits"] == "write_behind"


def test_synchronous_commits_turn_off_the_writer_thread(tmp_path):
    assert _run(tmp_path, "synchronous")["receipt"]["sync"]["commits"] == "synchronous"


def test_a_component_written_by_a_system_rejects_a_write_handle(tmp_path):
    assert 'ValueError: a system writes "pc_x"' in _run(tmp_path, "conflict")["error"]


def test_a_failing_hook_stops_later_hooks_and_the_run(tmp_path):
    result = _run(tmp_path, "failure")
    assert result["error"].startswith(
        "RuntimeError: pre_commit hook 0 (_scenario.<locals>.exchange_step) failed at tick 3"
    )
    assert "controller went away" in result["receipt"]["failure"]
    assert result["after"] == [0, 1, 2]
    assert result["receipt"]["failed_hook"] == {
        "index": 0,
        "name": "_scenario.<locals>.exchange_step",
        "committed": False,
    }


def test_hooks_read_committed_history_and_defer_writes_past_the_commit(tmp_path):
    result = _run(tmp_path, "history")
    assert result["sees_earlier_write"]
    # The commit of this tick has not happened yet; the previous tick's has.
    assert result["exact_now"] is None
    assert result["previous"][0] == 1000
    np.testing.assert_allclose(result["previous"][1], [1.5**4, 2 * 1.5**4])
    assert result["frame"] is None
    # Both writes of the previous tick share a timestamp; the second landed last.
    assert result["last_deferred"] == [41.0]
    assert result["pending"] is None
    assert "selection must be" in result["rejected"][0]
    assert 'values must have exactly the keys ["value"]' in result["rejected"][1]
    receipt = result["receipt"]
    assert receipt["sync"]["commits"] == "synchronous"
    callbacks = receipt["callbacks"]
    assert callbacks["history_queries"] == 5
    assert callbacks["history_misses"] == 3
    assert callbacks["deferred_batches"] == callbacks["deferred_rows"] == 2 * receipt["cycles"]
    assert receipt["deferred_writes"]["count"] == receipt["cycles"]


def test_a_deferred_write_that_fails_stops_the_run_after_the_commit(tmp_path):
    result = _run(tmp_path, "time_travel")
    assert "failed after the commit at tick 3" in result["error"]
    assert result["receipt"]["failed_hook"] == {
        "index": 1,
        "name": "_scenario.<locals>.history_step",
        "committed": True,
    }
    assert result["receipt"]["callbacks"]["deferred_failures"] == 1


def test_history_and_timestamped_writers_check_their_components():
    world, _ = _world(1)
    world.spawn([el.C(E, jnp.zeros(1)), el.C(T, jnp.zeros(1))], name="probe")
    with pytest.raises(ValueError, match="no entity component named"):
        world.prepare_component_history("e9.pc_x")
    with pytest.raises(ValueError, match="transient"):
        world.prepare_component_history("probe.pc_t")
    with pytest.raises(ValueError, match="external_control"):
        world.prepare_timestamped_writer({"x": "e0.pc_x"})


def test_spawning_after_prepare_invalidates_the_handle(tmp_path):
    assert "prepared before an entity was spawned" in _run(tmp_path, "stale")["error"]


def test_preparing_again_after_a_spawn_replaces_the_stale_handle(tmp_path):
    assert "error" not in _run(tmp_path, "replaced")


def test_only_manually_recorded_components_take_record_requests(tmp_path):
    assert '"pc_c" is not recorded manually' in _run(tmp_path, "record")["error"]


def test_pre_commit_takes_only_callables_and_native_hooks(tmp_path):
    assert "TypeError: pre_commit takes a callable" in _run(tmp_path, "not_a_hook")["error"]


def test_native_hooks_need_a_library_with_the_symbol():
    world, ids = _world(2)
    batch = world.prepare_batch({"x": world.prepare_component("pc_x", ids)})
    with pytest.raises(ValueError, match="cannot load"):
        el.NativeStepHook.load("/nonexistent/libhook.so", input=batch, symbol="step")
    with pytest.raises(ValueError, match='has no "no_such_step"'):
        el.NativeStepHook.load(el.elodin.__file__, input=batch, symbol="no_such_step")


def test_prepare_rejects_missing_and_ambiguous_rows():
    world, ids = _world(2)
    with pytest.raises(ValueError, match="no component named"):
        world.prepare_component("missing", ids)
    with pytest.raises(ValueError, match="listed twice for writing"):
        world.prepare_component("pc_c", [ids[0], ids[0]], access="write")
    with pytest.raises(ValueError, match="access must be"):
        world.prepare_component("pc_c", ids, access="append")
    with pytest.raises(ValueError, match="same number of rows"):
        world.prepare_batch(
            {
                "x": world.prepare_component("pc_x", ids),
                "c": world.prepare_component("pc_c", ids[:1]),
            }
        )
    handle = world.prepare_component("pc_x", [ids[1], ids[1]])
    assert handle.shape == (2, 2) and handle.access == "read"
