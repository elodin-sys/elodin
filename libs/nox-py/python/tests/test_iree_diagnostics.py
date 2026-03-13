import json
import os
import re
import shutil
import tempfile
import typing as ty
from dataclasses import dataclass

import elodin as el
import jax
import jax.numpy as jnp
import pytest

X = ty.Annotated[jax.Array, el.Component("x_diag", el.ComponentType.F64)]


@dataclass
class TestDiag(el.Archetype):
    x_diag: X


def _world_with_value(value: float = 1.0) -> el.World:
    w = el.World()
    w.spawn(TestDiag(jnp.array([value])), "e1")
    return w


def _extract_report_dir_from_error(message: str) -> str:
    match = re.search(r"Debug artifacts saved to:\s*(.+)", message)
    assert match is not None, f"missing report dir in error: {message}"
    return match.group(1).strip()


def _latest_report_dir(base_dir: str) -> str:
    entries = [
        os.path.join(base_dir, name)
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    assert entries, f"no artifact subdirectories created in {base_dir}"
    return max(entries, key=os.path.getmtime)


def test_iree_lowering_error_shows_real_message():
    @el.map
    def explode(_: X) -> X:
        raise RuntimeError("intentional iree diagnostics failure")

    w = _world_with_value()
    with pytest.raises(RuntimeError) as exc_info:
        w.build(explode, backend="iree")
    message = str(exc_info.value)
    assert "intentional iree diagnostics failure" in message
    assert "python error" not in message.lower()
    assert "Failure stage:" in message


@pytest.mark.skipif(
    shutil.which("iree-compile") is None, reason="iree-compile not available in test env"
)
def test_iree_failure_dumps_artifacts():
    @el.map
    def scale(x: X) -> X:
        return x * 2.0

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_test_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    os.environ["ELODIN_IREE_FLAGS"] = "--iree-no-such-flag"
    try:
        w = _world_with_value()
        with pytest.raises(RuntimeError, match="iree-compile failed") as exc_info:
            w.build(scale, backend="iree")

        report_dir = _extract_report_dir_from_error(str(exc_info.value))
        assert os.path.exists(os.path.join(report_dir, "stablehlo.mlir"))
        assert os.path.exists(os.path.join(report_dir, "iree_compile_stderr.txt"))
        assert os.path.exists(os.path.join(report_dir, "versions.json"))

        with open(os.path.join(report_dir, "versions.json"), encoding="utf-8") as f:
            versions = json.load(f)
        assert "jax" in versions
        assert "python" in versions
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)
        os.environ.pop("ELODIN_IREE_FLAGS", None)


@pytest.mark.skipif(
    shutil.which("iree-compile") is None, reason="iree-compile not available in test env"
)
def test_iree_flags_env_passthrough():
    @el.map
    def scale(x: X) -> X:
        return x * 2.0

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_flags_test_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    os.environ["ELODIN_IREE_FLAGS"] = "--iree-opt-const-eval=false"
    try:
        w = _world_with_value()
        w.build(scale, backend="iree")
        report_dir = _latest_report_dir(dump_dir)
        with open(os.path.join(report_dir, "iree_compile_cmd.sh"), encoding="utf-8") as f:
            cmd = f.read()
        assert "--iree-opt-const-eval=false" in cmd
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)
        os.environ.pop("ELODIN_IREE_FLAGS", None)
