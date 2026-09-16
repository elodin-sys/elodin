"""Display-kernel tracing, hashing, and matrix goldens."""

from __future__ import annotations

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest

import elodin.ui as ui
from elodin.ui.kernel import KernelError, KernelInput, compile_kernel
from elodin.ui.schema import Schema


def _schema() -> Schema:
    return Schema.from_json(
        {
            "components": {
                "drone.nav.covariance": {
                    "shape": [6],
                    "prim_type": "f64",
                    "element_names": ["p00", "p10", "p20", "p11", "p21", "p22"],
                },
                "drone.nav.speed": {"shape": [], "prim_type": "f64"},
            }
        }
    )


def test_kernel_infers_dtype_and_shape():
    schema = _schema()

    @ui.kernel
    def scale_speed(speed):
        return speed * 2.0

    artifact = scale_speed(schema["drone.nav.speed"]).artifact()
    assert artifact.inputs[0].dtype == "f64"
    assert artifact.inputs[0].shape == ()
    assert artifact.outputs[0]["dtype"] == "f64"
    assert artifact.outputs[0]["shape"] == []


def test_kernel_hash_is_deterministic():
    schema = _schema()

    @ui.kernel
    def plus_one(speed):
        return speed + 1.0

    first = plus_one(schema["drone.nav.speed"]).artifact()
    second = plus_one(schema["drone.nav.speed"]).artifact()
    assert first.hash == second.hash
    payload = {
        "batch_size": first.batch_size,
        "inputs": [
            {
                "name": item.name,
                "component": item.component,
                "shape": list(item.shape),
                "dtype": item.dtype,
            }
            for item in first.inputs
        ],
        "outputs": first.outputs,
        "scalar_mlir": first.scalar_mlir,
        "batched_mlir": first.batched_mlir,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert first.hash == hashlib.sha256(canonical).hexdigest()


def test_kernel_captures_closure_constants():
    schema = _schema()
    scale = 3.0

    @ui.kernel
    def scale_speed(speed):
        return speed * scale

    artifact = scale_speed(schema["drone.nav.speed"]).artifact()
    assert "3." in artifact.scalar_mlir or "3" in artifact.scalar_mlir


def test_kernel_rejects_unknown_component():
    schema = Schema.from_json({"components": {}}, strict=True)

    @ui.kernel
    def ident(x):
        return x

    with pytest.raises(Exception, match="unknown component"):
        ident(schema["missing.component"])


def test_kernel_rejects_shape_inference_failure():
    def exploding(_x):
        raise RuntimeError("cannot trace")

    with pytest.raises(KernelError, match="shape inference"):
        compile_kernel(
            exploding,
            [KernelInput(name="x", component="x", shape=(), dtype="f64")],
            validate=False,
        )


def _packed_cholesky(cov):
    p00, p10, p20 = cov[0], cov[1], cov[2]
    p11, p21, p22 = cov[3], cov[4], cov[5]
    l00 = jnp.sqrt(p00)
    l10 = p10 / l00
    l20 = p20 / l00
    l11 = jnp.sqrt(p11 - l10 * l10)
    l21 = (p21 - l20 * l10) / l11
    l22 = jnp.sqrt(p22 - l20 * l20 - l21 * l21)
    z = jnp.float64(0.0)
    return jnp.array([[l00, z, z], [l10, l11, z], [l20, l21, l22]])


def test_cholesky_matches_jax_golden():
    schema = _schema()

    @ui.kernel
    def covariance_cholesky(cov):
        return _packed_cholesky(cov)

    expr = covariance_cholesky(schema["drone.nav.covariance"])
    artifact = expr.artifact()
    assert artifact.outputs[0]["shape"] == [3, 3]
    cov = np.array([4.0, 0.2, 0.1, 2.0, 0.0, 1.0], dtype=np.float64)
    p = np.array(
        [
            [cov[0], cov[1], cov[2]],
            [cov[1], cov[3], cov[4]],
            [cov[2], cov[4], cov[5]],
        ]
    )
    expected = np.linalg.cholesky(p)
    got = covariance_cholesky._func(cov)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=1e-12)


def test_lapack_cholesky_is_rejected_by_cranelift():
    schema = _schema()

    @ui.kernel
    def lapack_cholesky(cov):
        p = jnp.array(
            [
                [cov[0], cov[1], cov[2]],
                [cov[1], cov[3], cov[4]],
                [cov[2], cov[4], cov[5]],
            ]
        )
        return jnp.linalg.cholesky(p)

    with pytest.raises(KernelError, match="Cranelift rejected"):
        lapack_cholesky(schema["drone.nav.covariance"]).artifact()


def test_kernel_rejects_untraceable_python_output():
    def bad(_x):
        return {"x": _x}

    with pytest.raises(KernelError, match="shape inference"):
        compile_kernel(
            bad,
            [KernelInput(name="x", component="x", shape=(), dtype="f64")],
            validate=False,
        )


def test_graph_builder_accepts_kernel_expr():
    schema = _schema()

    @ui.kernel
    def twice(speed):
        return speed * 2.0

    built = ui.schematic(ui.graph(twice(schema["drone.nav.speed"]), name="2x speed"))
    kdl = built.emit_kdl()
    assert "kernel=" in kdl
    assert "schematics/kernels/" in kdl
    assert "drone.nav.speed" in kdl


def test_write_emits_kernel_sidecar(tmp_path):
    schema = _schema()

    @ui.kernel
    def twice(speed):
        return speed * 2.0

    built = ui.schematic(ui.graph(twice(schema["drone.nav.speed"]), name="2x speed"))
    path = tmp_path / "out.kdl"
    ui.write(built, path)
    sidecars = list((tmp_path / "kernels").iterdir())
    assert sidecars
    data = json.loads(sidecars[0].read_bytes())
    assert data["hash"] == sidecars[0].name
    assert "module" in data["scalar_mlir"]
    assert "module" in data["batched_mlir"]
    assert data["hash"] in path.read_text()
