"""Callbacks on the native simulation thread share one Python thread state."""

import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import textwrap

import pytest


def _runtime_probe(cancel, result_path, db_path):
    import threading
    from typing import Annotated

    import elodin as el
    import jax
    import jax.numpy as jnp

    value_type = Annotated[
        jax.Array,
        el.Component("value", el.ComponentType(el.PrimitiveType.F64, (1,))),
    ]
    world = el.World()
    world.spawn([el.C(value_type, jnp.array([0.0]))], name="probe")

    @el.map
    def step(value: value_type) -> value_type:
        return value + 1

    local = threading.local()
    observations = []
    posts = []

    def observe(phase):
        local.count = getattr(local, "count", 0) + 1
        observations.append((phase, threading.get_ident(), local.count))

    def pre(tick, ctx):
        observe("pre")

    def post(tick, ctx):
        observe("post")
        posts.append(tick)

    def cancelled():
        observe("cancel")
        return cancel and len(posts) >= 12

    world.run(
        step,
        simulation_rate=1000.0,
        backend="cranelift",
        max_ticks=24,
        pre_step=pre,
        post_step=post,
        is_canceled=cancelled,
        interactive=False,
        db_path=db_path,
        log_level="warn",
    )
    Path(result_path).write_text(json.dumps({"observations": observations, "posts": posts}))


def _available_port():
    # The runtime binds Impeller, assets and gRPC on adjacent ports. Avoid
    # the ephemeral range so an outgoing connection cannot consume one.
    for _ in range(100):
        port = random.randrange(20000, 40000)
        listeners = []
        try:
            for offset in range(3):
                listener = socket.socket()
                listeners.append(listener)
                listener.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            continue
        finally:
            for listener in listeners:
                listener.close()
    pytest.fail("no available block of runtime ports")


@pytest.mark.parametrize("cancel", [False, True], ids=["complete", "cancel"])
def test_native_runtime_callback_thread_state(tmp_path, cancel):
    # Isolate runtime threads, signal handlers and listeners from pytest.
    # Use this interpreter's SDK directly; no installed Editor CLI is needed.
    result = tmp_path / "result.json"
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""\
            import runpy
            probe = runpy.run_path({str(Path(__file__).resolve())!r})
            probe['_runtime_probe']({cancel!r}, {str(result)!r}, {str(tmp_path / "db")!r})
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
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    proof = json.loads(result.read_text())
    observations = proof["observations"]
    assert {row[0] for row in observations} == {"pre", "post", "cancel"}
    assert len({row[1] for row in observations}) == 1
    assert [row[2] for row in observations] == list(range(1, len(observations) + 1))
    assert proof["posts"] == list(range(12 if cancel else 24))
