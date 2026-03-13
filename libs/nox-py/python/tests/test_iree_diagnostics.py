import json
import os
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
    def unsupported(x: X) -> X:
        return x.at[0].set(x[0] + 1.0)

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_test_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = _world_with_value()
        with pytest.raises(RuntimeError, match="iree-compile failed"):
            w.build(unsupported, backend="iree")

        assert os.path.exists(os.path.join(dump_dir, "stablehlo.mlir"))
        assert os.path.exists(os.path.join(dump_dir, "iree_compile_stderr.txt"))
        assert os.path.exists(os.path.join(dump_dir, "versions.json"))

        with open(os.path.join(dump_dir, "versions.json"), encoding="utf-8") as f:
            versions = json.load(f)
        assert "jax" in versions
        assert "python" in versions
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


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
        with open(os.path.join(dump_dir, "iree_compile_cmd.sh"), encoding="utf-8") as f:
            cmd = f.read()
        assert "--iree-opt-const-eval=false" in cmd
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)
        os.environ.pop("ELODIN_IREE_FLAGS", None)
