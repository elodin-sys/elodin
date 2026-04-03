import json
import os
import re
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


def _all_report_dirs(base_dir: str) -> list[str]:
    entries = [
        os.path.join(base_dir, name)
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    return sorted(entries)


def _read_json(path: str) -> dict[str, ty.Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _require_iree_compiler() -> None:
    pytest.importorskip("iree.compiler")


def test_iree_lowering_error_shows_real_message():
    @el.map
    def explode(_: X) -> X:
        raise RuntimeError("intentional iree diagnostics failure")

    w = _world_with_value()
    with pytest.raises(RuntimeError) as exc_info:
        w.build(explode, backend="iree-cpu")
    message = str(exc_info.value)
    assert "intentional iree diagnostics failure" in message
    assert "python error" not in message.lower()
    assert "Failure stage:" in message


def test_jax_lower_failure_dumps_artifacts():
    @el.map
    def explode(_: X) -> X:
        raise RuntimeError("intentional iree diagnostics failure")

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_jax_lower_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = _world_with_value()
        with pytest.raises(RuntimeError, match="intentional iree diagnostics failure") as exc_info:
            w.build(explode, backend="iree-cpu")

        report_dir = _extract_report_dir_from_error(str(exc_info.value))
        assert os.path.exists(os.path.join(report_dir, "jax_lower_traceback.txt"))
        assert os.path.exists(os.path.join(report_dir, "versions.json"))
        assert os.path.exists(os.path.join(report_dir, "system_names.txt"))
        assert os.path.exists(os.path.join(report_dir, "compile_context.json"))
        assert os.path.exists(os.path.join(report_dir, "iree_compile_cmd.sh"))
        assert not os.path.exists(os.path.join(report_dir, "stablehlo.mlir"))

        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["compile_origin"] == "primary_system"
        assert context["failure_stage"] == "jax_lower"
        assert context["report_stage"] == "jax_lower_failed"
        assert context["has_singleton_lowering"] is True
        assert context["is_subsystem_diagnostic"] is False
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


def test_iree_failure_dumps_artifacts():
    _require_iree_compiler()

    @el.map
    def scale(x: X) -> X:
        return x * 2.0

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_test_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    os.environ["ELODIN_IREE_FLAGS"] = "--iree-no-such-flag"
    try:
        w = _world_with_value()
        with pytest.raises(RuntimeError, match="iree-compile failed") as exc_info:
            w.build(scale, backend="iree-cpu")

        report_dir = _extract_report_dir_from_error(str(exc_info.value))
        assert os.path.exists(os.path.join(report_dir, "stablehlo.mlir"))
        assert os.path.exists(os.path.join(report_dir, "iree_compile_stderr.txt"))
        assert os.path.exists(os.path.join(report_dir, "versions.json"))
        assert os.path.exists(os.path.join(report_dir, "compile_context.json"))

        with open(os.path.join(report_dir, "versions.json"), encoding="utf-8") as f:
            versions = json.load(f)
        assert "jax" in versions
        assert "python" in versions

        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["compile_origin"] == "primary_system"
        assert context["report_stage"] == "iree_compile_failed"
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)
        os.environ.pop("ELODIN_IREE_FLAGS", None)


def test_iree_flags_env_passthrough():
    _require_iree_compiler()

    @el.map
    def scale(x: X) -> X:
        return x * 2.0

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_flags_test_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    os.environ["ELODIN_IREE_FLAGS"] = "--iree-opt-const-eval=false"
    try:
        w = _world_with_value()
        w.build(scale, backend="iree-cpu")
        report_dir = _latest_report_dir(dump_dir)
        with open(os.path.join(report_dir, "iree_compile_cmd.sh"), encoding="utf-8") as f:
            cmd = f.read()
        assert "--iree-opt-const-eval=false" in cmd
        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["compile_origin"] == "primary_system"
        assert context["report_stage"] == "iree_compile_succeeded"
        assert context["has_singleton_lowering"] is True
        assert "--iree-opt-const-eval=false" in context["effective_iree_flags"]
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)
        os.environ.pop("ELODIN_IREE_FLAGS", None)


def test_mixed_world_singleton_query_uses_local_singleton_lowering():
    _require_iree_compiler()

    RocketOnly = ty.Annotated[
        jax.Array,
        el.Component("rocket_only_diag", el.ComponentType.F64),
    ]

    @dataclass
    class RocketDiag(el.Archetype):
        x_diag: X
        rocket_only_diag: RocketOnly

    @dataclass
    class TargetDiag(el.Archetype):
        x_diag: X

    @el.map
    def scale(rocket_only_diag: RocketOnly) -> RocketOnly:
        return rocket_only_diag * 2.0

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_mixed_singleton_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = el.World()
        w.spawn(RocketDiag(jnp.array([1.0]), jnp.array([2.0])), "rocket")
        w.spawn(TargetDiag(jnp.array([3.0])), "target")
        w.build(scale, backend="iree-cpu")
        report_dir = _latest_report_dir(dump_dir)
        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["has_singleton_lowering"] is True
        assert any(summary["shape"] == [] for summary in context["input_arrays"])
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


def test_mixed_world_length_one_compare_compiles_with_iree():
    _require_iree_compiler()
    import jax.lax as lax

    RocketVec = ty.Annotated[
        jax.Array,
        el.Component("rocket_vec_diag", el.ComponentType(el.PrimitiveType.F64, (1,))),
    ]

    @dataclass
    class RocketVecDiag(el.Archetype):
        x_diag: X
        rocket_vec_diag: RocketVec

    @dataclass
    class TargetDiag(el.Archetype):
        x_diag: X

    @el.map
    def keep_or_increment(rocket_vec_diag: RocketVec) -> RocketVec:
        return lax.select(
            rocket_vec_diag == 1.0,
            rocket_vec_diag,
            rocket_vec_diag + 1.0,
        )

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_singleton_compare_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = el.World()
        w.spawn(RocketVecDiag(jnp.array([1.0]), jnp.array([1.0])), "rocket")
        w.spawn(TargetDiag(jnp.array([3.0])), "target")
        w.build(keep_or_increment, backend="iree-cpu")

        report_dir = _latest_report_dir(dump_dir)
        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["report_stage"] == "iree_compile_succeeded"
        assert context["has_singleton_lowering"] is True
        assert any(summary["shape"] == [1] for summary in context["input_arrays"])

        with open(os.path.join(report_dir, "stablehlo.mlir"), encoding="utf-8") as f:
            stablehlo = f.read()
        assert "stablehlo.compare" in stablehlo
        assert "tensor<1xf64>" in stablehlo
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


def test_map_seq_multiple_entities_lowers_with_while():
    pytest.importorskip("iree.compiler")
    import jax.lax as lax

    Y3 = ty.Annotated[
        jax.Array,
        el.Component("y3_diag", el.ComponentType(el.PrimitiveType.F64, (3,))),
    ]

    @dataclass
    class TestDiagVec(el.Archetype):
        x_diag: X
        y3_diag: Y3

    @el.map_seq
    def branchy(x: X) -> Y3:
        return lax.cond(
            x > 1.5,
            lambda _: jnp.stack([x, x + 1.0, x + 2.0]),
            lambda _: jnp.stack([x, x - 1.0, x - 2.0]),
            operand=None,
        )

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_map_seq_scan_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = el.World()
        w.spawn(TestDiagVec(jnp.array(1.0), jnp.zeros(3)), "e1")
        w.spawn(TestDiagVec(jnp.array(2.0), jnp.zeros(3)), "e2")
        w.build(branchy, backend="iree-cpu")
        report_dir = _latest_report_dir(dump_dir)
        with open(os.path.join(report_dir, "stablehlo.mlir"), encoding="utf-8") as f:
            stablehlo = f.read()
        assert "stablehlo.while" in stablehlo
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


def test_piped_map_seq_compare_emits_noxpr_artifact():
    _require_iree_compiler()
    import jax.lax as lax

    ThreshPipe = ty.Annotated[
        jax.Array,
        el.Component("thresh_pipe_diag", el.ComponentType.F64),
    ]
    RocketVecPipe = ty.Annotated[
        jax.Array,
        el.Component("rocket_vec_pipe_diag", el.ComponentType(el.PrimitiveType.F64, (1,))),
    ]

    @dataclass
    class RocketVecPipeDiag(el.Archetype):
        x_diag: X
        rocket_vec_pipe_diag: RocketVecPipe

    @dataclass
    class ThresholdPipeDiag(el.Archetype):
        thresh_pipe_diag: ThreshPipe

    @el.map
    def passthrough(rocket_vec_pipe_diag: RocketVecPipe) -> RocketVecPipe:
        return rocket_vec_pipe_diag

    @el.system
    def compare_seq(
        thresh_pipe_diag: el.Query[ThreshPipe],
        q: el.Query[RocketVecPipe],
    ) -> el.Query[RocketVecPipe]:
        return q.map_seq(
            RocketVecPipe,
            lambda rocket_vec_pipe_diag: lax.select(
                rocket_vec_pipe_diag == thresh_pipe_diag[0],
                rocket_vec_pipe_diag,
                rocket_vec_pipe_diag + 1.0,
            ),
        )

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_pipe_compare_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = el.World()
        w.spawn(ThresholdPipeDiag(jnp.array(1.0)), "g")
        w.spawn(RocketVecPipeDiag(jnp.array([1.0]), jnp.array([1.0])), "e1")
        w.spawn(RocketVecPipeDiag(jnp.array([2.0]), jnp.array([2.0])), "e2")
        w.build(passthrough.pipe(compare_seq), backend="iree-cpu")

        report_dir = _latest_report_dir(dump_dir)
        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["report_stage"] == "iree_compile_succeeded"
        assert context["compile_origin"] == "primary_system"
        assert context["noxpr_artifact"] == "noxpr.txt"
        assert "noxpr.txt" in context["noxpr_artifacts"]
        assert any(name.startswith("noxpr_call_") for name in context["noxpr_artifacts"])

        with open(os.path.join(report_dir, "noxpr.txt"), encoding="utf-8") as f:
            noxpr = f.read()
        assert "call(comp_id =" in noxpr

        with open(os.path.join(report_dir, "stablehlo.mlir"), encoding="utf-8") as f:
            stablehlo = f.read()
        assert "stablehlo.compare" in stablehlo
        assert "stablehlo.while" in stablehlo
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


def test_unit_extent_broadcasts_are_rewritten_to_reshape():
    _require_iree_compiler()

    Quat = ty.Annotated[
        jax.Array,
        el.Component("quat_outer_diag", el.ComponentType(el.PrimitiveType.F64, (4,))),
    ]
    RotMat = ty.Annotated[
        jax.Array,
        el.Component("quat_outer_matrix_diag", el.ComponentType(el.PrimitiveType.F64, (3, 3))),
    ]

    @dataclass
    class QuatOuterDiag(el.Archetype):
        quat_outer_diag: Quat
        quat_outer_matrix_diag: RotMat

    @el.map
    def quat_outer_matrix(quat_outer_diag: Quat) -> RotMat:
        q0, q1, q2, s = quat_outer_diag
        v = jnp.array([q0, q1, q2])
        return 2.0 * jnp.outer(v, v) + jnp.identity(3) * (s**2 - jnp.dot(v, v))

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_unit_extent_broadcast_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    try:
        w = el.World()
        w.spawn(
            QuatOuterDiag(
                jnp.array([0.1, 0.2, 0.3, 0.9]),
                jnp.zeros((3, 3)),
            ),
            "rocket",
        )
        w.build(quat_outer_matrix, backend="iree-cpu")

        report_dir = _latest_report_dir(dump_dir)
        context = _read_json(os.path.join(report_dir, "compile_context.json"))
        assert context["report_stage"] == "iree_compile_succeeded"
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)


def test_subsystem_diagnostic_artifacts_are_tagged():
    _require_iree_compiler()

    @el.map
    def scale(x: X) -> X:
        return x * 2.0

    dump_dir = tempfile.mkdtemp(prefix="elodin_iree_subsystem_diag_")
    os.environ["ELODIN_IREE_DUMP_DIR"] = dump_dir
    os.environ["ELODIN_IREE_FLAGS"] = "--iree-no-such-flag"
    try:
        w = _world_with_value()
        with pytest.raises(RuntimeError, match="iree-compile failed"):
            w.build(scale, backend="iree-cpu")

        report_dirs = _all_report_dirs(dump_dir)
        origins = {
            _read_json(os.path.join(report_dir, "compile_context.json"))["compile_origin"]
            for report_dir in report_dirs
        }
        assert "primary_system" in origins
        assert "subsystem_diagnostic" in origins
    finally:
        os.environ.pop("ELODIN_IREE_DUMP_DIR", None)
        os.environ.pop("ELODIN_IREE_FLAGS", None)
