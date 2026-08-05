"""Phase 1 elodin.ui builders — round-trip and G1 model equality."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

import elodin as el
import elodin.db as edb
import elodin.ui as ui

REPO = Path(__file__).resolve().parents[4]
EXAMPLES = REPO / "examples"


def _load_example(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canonical(schematic: ui.Schematic) -> ui.Schematic:
    return ui.from_kdl(schematic.emit_kdl())


def test_from_kdl_emit_idempotent():
    text = (EXAMPLES / "db-client" / "schematic.kdl").read_text()
    once = ui.from_kdl(text)
    twice = ui.from_kdl(once.emit_kdl())
    assert twice == ui.from_kdl(twice.emit_kdl())


def test_graph_builder_roundtrip():
    built = ui.schematic(ui.graph("drone.thrust", name="Thrust"))
    assert ui.from_kdl(built.emit_kdl()) == built


def test_tabs_hsplit_vsplit_roundtrip():
    built = ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.vsplit(
                    ui.graph("a"),
                    ui.graph("b"),
                    share=0.4,
                ),
                ui.graph("c"),
                name="Panel",
            ),
        ),
    )
    assert ui.from_kdl(built.emit_kdl()) == built


def test_write_roundtrip(tmp_path):
    built = ui.schematic(ui.graph("x"))
    path = tmp_path / "out.kdl"
    ui.write(built, path)
    assert ui.from_kdl(path.read_text()) == built


def test_g1_db_client_schematic_equals_handwritten():
    mod = _load_example(EXAMPLES / "db-client" / "schematic.py", "db_client_schematic")
    handwritten = ui.from_kdl((EXAMPLES / "db-client" / "schematic.kdl").read_text())
    assert _canonical(mod.build()) == _canonical(handwritten)


def test_g1_motor_panel_equals_handwritten():
    mod = _load_example(EXAMPLES / "drone" / "motor_panel.py", "motor_panel")
    handwritten = ui.from_kdl((EXAMPLES / "drone" / "motor-panel.kdl").read_text())
    assert _canonical(mod.build()) == _canonical(handwritten)


def test_g1_rate_control_panel_equals_handwritten():
    mod = _load_example(EXAMPLES / "drone" / "rate_control_panel.py", "rate_control_panel")
    handwritten = ui.from_kdl((EXAMPLES / "drone" / "rate-control-panel.kdl").read_text())
    assert _canonical(mod.build()) == _canonical(handwritten)


def test_world_schematic_accepts_ui_schematic():
    world = el.World()
    s = ui.schematic(ui.graph("drone.thrust"))
    world.schematic(s)
    world.schematic(s.emit_kdl())


def test_push_to_embedded_server(tmp_path):
    addr = "127.0.0.1:23551"
    db_path = tmp_path / "db"
    server = edb.Server.start(str(db_path), addr)
    time.sleep(0.3)
    try:
        s = ui.schematic(ui.graph("drone.thrust", name="Thrust"))
        ui.push(s, addr)
        asset = db_path / "assets" / "schematics" / "main.kdl"
        deadline = time.time() + 5
        while time.time() < deadline:
            if asset.exists():
                body = asset.read_text()
                assert "drone.thrust" in body
                return
            time.sleep(0.1)
        pytest.fail(f"asset not written after push: missing {asset}")
    finally:
        server.stop()
