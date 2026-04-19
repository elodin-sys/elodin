import importlib.metadata as importlib_metadata
import platform
import subprocess

import jax
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


def _has_jax_cuda_plugin() -> bool:
    try:
        importlib_metadata.version("jax-cuda12-plugin")
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def test_jax_cuda_device_smoke():
    if not _has_nvidia_gpu():
        pytest.skip("No NVIDIA GPU detected on this host")
    if not _has_jax_cuda_plugin():
        pytest.skip("jax-cuda12-plugin is not installed in this environment")
    devices = jax.devices()
    assert any(d.platform in ("gpu", "cuda") for d in devices), devices
