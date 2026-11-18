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

    def on_receipt(receipt):
        result["receipt"] = receipt

    try:
        world.run(
            _step,
            simulation_rate=1000.0,
            max_ticks=12,
            pre_commit=[exchange_step, 42] if name == "not_a_hook" else exchange_step,
            on_receipt=on_receipt,
            interactive=False,
            db_path=db_path,
            log_level="warn",
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


def test_a_component_written_by_a_system_rejects_a_write_handle(tmp_path):
    assert 'ValueError: a system writes "pc_x"' in _run(tmp_path, "conflict")["error"]


def test_a_failing_hook_stops_the_run_with_a_failed_receipt(tmp_path):
    result = _run(tmp_path, "failure")
    assert result["error"].startswith("RuntimeError: pre_commit failed at tick 3")
    assert "controller went away" in result["receipt"]["failure"]


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
