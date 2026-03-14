import platform
import subprocess
import typing as ty
from dataclasses import dataclass

import elodin as el
import jax
import jax.numpy as jnp
import pytest


def _has_nvidia_gpu() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def test_jax_cuda_device_smoke():
    if not _has_nvidia_gpu():
        pytest.skip("No NVIDIA GPU detected on this host")
    devices = jax.devices()
    assert any(d.platform in ("gpu", "cuda") for d in devices), devices


def test_iree_cuda_compile_smoke():
    if not _has_nvidia_gpu():
        pytest.skip("No NVIDIA GPU detected on this host")

    X = ty.Annotated[jax.Array, el.Component("x32_gpu_smoke", el.ComponentType.F32)]

    @el.map
    def step(x: X) -> X:
        return x + jnp.array(1.0, dtype=jnp.float32)

    @dataclass
    class A(el.Archetype):
        x: X

    w = el.World()
    w.spawn(A(jnp.array([1.0], dtype=jnp.float32)), "e1")
    try:
        exec = w.build(step, backend="iree", device="cuda")
    except RuntimeError as exc:
        msg = str(exc)
        if "missing GPU target in #hal.executable.target" in msg:
            pytest.skip(
                "IREE CUDA compiler in the current environment cannot configure cuda-nvptx targets"
            )
        raise
    exec.run(1)
    history = exec.history(["e1.x32_gpu_smoke"])
    assert float(history["e1.x32_gpu_smoke"][-1]) == pytest.approx(2.0)
