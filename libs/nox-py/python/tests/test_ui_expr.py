"""Phase 2 typed expressions + Phase 3 watch helpers."""

from __future__ import annotations

import elodin.ui as ui
from elodin.ui import Expr, ExprError, Schema, pose


def test_expr_ops_emit_eql():
    a = Expr("drone.world_pos")
    b = a + Expr("(0,0,0,0, 1, 0, 0)")
    assert str(b) == "(drone.world_pos + (0,0,0,0, 1, 0, 0))"
    assert str(a[4]) == "drone.world_pos[4]"
    assert str(a.sqrt()) == "drone.world_pos.sqrt()"


def test_schema_index_bounds():
    schema = Schema(
        {
            "NAV.QUAT": {
                "element_names": ["q0", "q1", "q2", "q3"],
                "shape": [4],
            }
        }
    )
    q = schema["NAV.QUAT"]
    assert str(q.q0) == "NAV.QUAT.q0"
    assert str(q[2]) == "NAV.QUAT[2]"
    try:
        _ = q[4]
        raise AssertionError("expected out-of-range")
    except ExprError as exc:
        assert "out of range" in str(exc)


def test_pose_wxyz_order():
    schema = Schema(
        {
            "NAV.QUAT": {"element_names": ["q0", "q1", "q2", "q3"], "shape": [4]},
            "NAV.POS": {"element_names": ["x", "y", "z"], "shape": [3]},
        }
    )
    # Treat q0 as w for this packing demo.
    p = pose(quat=schema["NAV.QUAT"], pos=schema["NAV.POS"], order="wxyz")
    assert "NAV.QUAT.q1" in str(p)
    assert "NAV.POS.x" in str(p)


def test_graph_accepts_expr():
    built = ui.schematic(ui.graph(Expr("drone.thrust"), name="Thrust"))
    assert "drone.thrust" in built.emit_kdl()


def test_g1_still_matches_with_expr_schematic():
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    path = repo / "examples" / "db-client" / "schematic.py"
    spec = importlib.util.spec_from_file_location("db_client_schematic", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    handwritten = ui.from_kdl((repo / "examples" / "db-client" / "schematic.kdl").read_text())
    rebuilt = mod.build()
    assert ui.from_kdl(rebuilt.emit_kdl()) == ui.from_kdl(handwritten.emit_kdl())
