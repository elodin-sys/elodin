"""Deterministic witnesses of `World.run`, compared and replayed."""

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

X = Annotated[jax.Array, el.Component("wx", el.ComponentType(el.PrimitiveType.F64, (2,)))]
C = Annotated[jax.Array, el.Component("wc", el.ComponentType(el.PrimitiveType.F64, (2,)))]


def _world():
    world = el.World()
    ids = [
        world.spawn([el.C(X, jnp.array([1.0, 2.0]) * (i + 1)), el.C(C, jnp.zeros(2))], name=f"e{i}")
        for i in range(3)
    ]
    batch = world.prepare_batch(
        {
            "x": world.prepare_component("wx", ids),
            "c": world.prepare_component("wc", ids, access="write"),
        }
    )
    return world, batch


@el.map
def _step(x: X, c: C) -> X:
    return x + c


def _scenario(options, result_path, db_path):
    world, batch = _world()
    witness = world.prepare_witness(
        {"pair": batch},
        channels={"sent": (batch, 16)},
        external={"reply": (batch, 8)},
        stages=options.get("stages"),
    )
    result = {}

    def exchange(tick, ctx):
        with ctx.batch(batch) as view:
            sent = view.x.copy()
            if options.get("flip") == tick:
                sent.view(np.uint8)[1, 0] ^= 1
            ctx.witness_bytes("sent", sent if options.get("sent_rows") is None else sent[:1])
            reply = ctx.witness_replay("reply")
            if reply is None:
                # Stands for what an outside process answered.
                answer = np.full((3, 1), 1e-3 * tick + options.get("reply_offset", 0.0))
                reply = answer.view(np.uint8)
                ctx.witness_bytes(options.get("channel", "reply"), reply)
            view.c[...] = -1e-3 * view.x + reply.view(np.float64)

    hooks = [exchange]
    if options.get("history"):
        past_x = world.prepare_component_history("e0.wx")
        hooks.append(lambda tick, ctx: ctx.read_at(past_x, ctx.timestamp, "at_or_before"))
    raw = options.get("raw", False)
    try:
        if "replay" in options:
            result["replay"] = world.replay_pre_commit(
                options["replay"], hooks, live=options.get("live", False)
            )
        else:
            trace = el.DeterminismTrace(
                options["path"],
                witness,
                raw=tuple(raw) if isinstance(raw, list) else raw,
                reference=options.get("reference"),
                stop_after_first_mismatch=options.get("stop", False),
            )
            world.run(
                _step,
                simulation_rate=1000.0,
                max_ticks=options.get("ticks", 40),
                pre_commit=hooks,
                on_receipt=lambda receipt: result.update(receipt=receipt),
                interactive=False,
                db_path=db_path,
                log_level="warn",
                determinism_trace=trace,
            )
    except (ValueError, RuntimeError, TypeError, KeyError) as err:
        result["error"] = f"{type(err).__name__}: {err}"
    Path(result_path).write_text(json.dumps(result))


def _run(tmp_path, **options) -> dict:
    case = tmp_path / f"case-{len(list(tmp_path.glob('case-*')))}"
    case.mkdir()
    result = case / "result.json"
    probe = case / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""\
            import runpy, sys
            sys.path.insert(0, {str(Path(__file__).parent)!r})
            probe = runpy.run_path({str(Path(__file__).resolve())!r})
            probe['_scenario']({options!r}, {str(result)!r}, {str(case / "db")!r})
            """
        )
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("ELODIN_PACING_")}
    completed = subprocess.run(
        [sys.executable, str(probe), "run", f"127.0.0.1:{_available_port()}", "--no-s10"],
        cwd=case,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result.read_text())


def _record(tmp_path, name, **options) -> dict:
    return _run(tmp_path, path=str(tmp_path / name), **options)["receipt"]["witness"]


FLIPPED = {
    "tick": 20,
    "stage": "after_hook:0",
    "hook": "_scenario.<locals>.exchange",
    "kind": "channel",
    "record": "sent",
    "entity": "e1",
    "row": 1,
    "byte": 0,
}


def _located(mismatch: dict) -> dict:
    return {key: mismatch[key] for key in FLIPPED}


def test_repeated_runs_record_the_same_witness(tmp_path):
    raw = _record(tmp_path, "raw.witness", raw=True)
    digests = _record(tmp_path, "digests.witness")
    window = _record(tmp_path, "window.witness", raw=[5, 9])
    assert raw["root"] == digests["root"] == window["root"]
    assert (raw["ticks"], raw["raw_ticks"], digests["raw_ticks"], window["raw_ticks"]) == (
        40,
        40,
        0,
        4,
    )
    assert raw["bytes_written"] > digests["bytes_written"] > 0
    assert raw["capture"]["count"] == 40
    assert raw["first_mismatch"] is None and raw["failure"] is None
    comparison = el.determinism.compare(tmp_path / "raw.witness", tmp_path / "digests.witness")
    assert comparison["identical"] and comparison["ticks"] == 40


def test_a_reference_locates_the_first_mismatch_and_keeps_bytes_around_it(tmp_path):
    _record(tmp_path, "baseline.witness", raw=True)
    reference = str(tmp_path / "baseline.witness")
    flipped = _record(tmp_path, "flipped.witness", reference=reference, flip=20)
    mismatch = flipped["first_mismatch"]
    assert _located(mismatch) == FLIPPED
    assert mismatch["right"] == mismatch["left"] ^ 1
    # 16 ticks on each side of the mismatch keep their exact bytes.
    assert flipped["raw_ticks"] == 33

    comparison = el.determinism.compare(reference, tmp_path / "flipped.witness", stop_at_first=True)
    assert comparison["first_mismatch"] == mismatch
    assert (comparison["ticks"], comparison["mismatched_ticks"]) == (21, 1)


def test_a_run_can_stop_at_its_first_mismatch(tmp_path):
    _record(tmp_path, "baseline.witness", ticks=2000)
    result = _run(
        tmp_path,
        path=str(tmp_path / "stopped.witness"),
        reference=str(tmp_path / "baseline.witness"),
        flip=20,
        stop=True,
        ticks=2000,
    )
    assert "error" not in result
    assert result["receipt"]["witness"]["first_mismatch"]["tick"] == 20
    assert result["receipt"]["cycles"] < 2000


def test_replay_reproduces_the_hooks_and_locates_a_difference(tmp_path):
    _record(tmp_path, "baseline.witness", raw=True)
    _record(tmp_path, "flipped.witness", reference=str(tmp_path / "baseline.witness"), flip=20)

    same = _run(tmp_path, replay=str(tmp_path / "baseline.witness"))["replay"]
    assert same["first_mismatch"] is None
    assert (same["ticks"], same["skipped_ticks"], same["records"]) == (40, 0, 40 * 3)

    # The replayed hook does not flip the byte, so it differs from the recording there.
    flipped = _run(tmp_path, replay=str(tmp_path / "flipped.witness"))["replay"]
    assert _located(flipped["first_mismatch"]) == FLIPPED
    assert flipped["skipped_ticks"] == 4

    # A replay feeds back the recorded replies; a live one lets the hook answer itself.
    offset = {"replay": str(tmp_path / "baseline.witness"), "reply_offset": 1.0}
    assert _run(tmp_path, **offset)["replay"]["first_mismatch"] is None
    live = _run(tmp_path, live=True, **offset)["replay"]["first_mismatch"]
    assert (live["tick"], live["kind"], live["record"]) == (0, "field", "pair.c")


def test_replay_needs_exact_bytes_and_has_no_db(tmp_path):
    _record(tmp_path, "digests.witness")
    error = _run(tmp_path, replay=str(tmp_path / "digests.witness"))["error"]
    assert "kept no exact bytes to replay" in error
    _record(tmp_path, "history.witness", raw=True, history=True)
    error = _run(tmp_path, replay=str(tmp_path / "history.witness"), history=True)["error"]
    assert "history reads need the DB, which a replay does not have" in error


def test_channels_take_every_row_of_their_batch(tmp_path):
    error = _run(tmp_path, path=str(tmp_path / "short.witness"), sent_rows=1)["error"]
    assert '"sent" takes 3 rows of 16 bytes, 48 in all; got 16 bytes' in error
    error = _run(tmp_path, path=str(tmp_path / "unknown.witness"), channel="replies")["error"]
    assert 'no witness channel named "replies"' in error


def test_stages_and_references_are_checked(tmp_path):
    error = _run(tmp_path, path=str(tmp_path / "x.witness"), stages=["after_hook:3"])["error"]
    assert 'this run has no stage "after_hook:3"' in error
    _record(tmp_path, "systems.witness", stages=["after_systems"])
    narrowed = el.determinism.compare(tmp_path / "systems.witness", tmp_path / "systems.witness")
    assert narrowed["identical"]
    error = _run(
        tmp_path, path=str(tmp_path / "all.witness"), reference=str(tmp_path / "systems.witness")
    )["error"]
    assert "has a different schema: the stages are" in error


def test_prepare_witness_checks_its_channels_and_stages():
    world, batch = _world()
    other = world.prepare_batch({"x": world.prepare_component("wx", [])})
    with pytest.raises(ValueError, match="must be one of the witnessed batches"):
        world.prepare_witness({"pair": batch}, channels={"sent": (other, 8)})
    with pytest.raises(ValueError, match="declared twice"):
        world.prepare_witness(
            {"pair": batch}, channels={"a": (batch, 8)}, external={"a": (batch, 8)}
        )
    with pytest.raises(ValueError, match="unknown stage"):
        world.prepare_witness({"pair": batch}, stages=["after_everything"])
    witness = world.prepare_witness({"pair": batch})
    with pytest.raises(ValueError, match="raw must be"):
        el.DeterminismTrace("x.witness", witness, raw=(9, 3))
