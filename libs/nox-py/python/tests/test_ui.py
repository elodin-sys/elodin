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
KDL_CORPUS = REPO / "libs" / "impeller2" / "kdl" / "tests" / "corpus" / "sources"


def _load_example(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canonical(schematic: ui.Schematic) -> ui.Schematic:
    return ui.from_kdl(schematic.emit_kdl())


@pytest.mark.parametrize(
    "path",
    sorted(KDL_CORPUS.rglob("*.kdl")),
    ids=lambda path: str(path.relative_to(KDL_CORPUS)),
)
def test_to_python_preserves_corpus_model(path: Path):
    source = path.read_text()
    generated = ui.to_python(source, source_name=path.name)
    assert "SOURCE_KDL" not in generated
    assert "ui.from_kdl" not in generated
    namespace = {"__name__": "generated_schematic"}
    exec(compile(generated, str(path.with_suffix(".py")), "exec"), namespace)
    assert _canonical(namespace["build"]()) == _canonical(ui.from_kdl(source))


def test_to_python_preserves_line_comments():
    generated = ui.to_python("// Flight dashboard\nviewport\n")
    assert "# Flight dashboard" in generated
    assert "SOURCE_KDL" not in generated


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


def test_extended_typed_builders_roundtrip():
    visibility = ui.visibility_range(min=50.0, fade_distance=50.0)
    arrow = ui.vector_arrow(
        "(1, 2, 3)",
        color=ui.color(12, 34, 56, 78),
        show_name=False,
        thickness=0.025,
        label_position="0.5",
        frame="ECEF",
    )
    built = ui.schematic(
        ui.viewport(
            cinematic=True,
            bloom=ui.bloom(intensity=0.25),
            ev100=13.5,
            frustums_color=ui.color(1, 2, 3, 4),
            projection_color=ui.color(5, 6, 7),
            frustums_thickness=0.01,
            view_cube_frame="NED",
            smoothing=1.0,
            arrows=[arrow],
        ),
        ui.geo_position_gauge("pose", source="ECEF", display="LLA"),
        ui.orientation_gauge("pose", source="ECEF", display="NED", reference=(0, 0, 0, 1)),
        ui.horizon_gauge("pose", source="ECEF", reference=(0, 0, 0, 1)),
        ui.object_3d(
            "pose",
            ui.ellipsoid(
                error_covariance="cov",
                color=ui.color(0, 188, 212, 40),
                show_grid=True,
                grid_color=ui.color(255, 255, 255, 120),
            ),
            frame="ECEF",
            frame_orientation="NED",
            orientation="absolute",
            icon=ui.icon(
                builtin="rocket_launch",
                color=ui.color(76, 175, 80),
                visibility=visibility,
            ),
            thrusters=[
                ui.thruster(
                    "thrust",
                    (-0.6, 0, 0),
                    direction=(-1, 0, 0),
                    body_frame=True,
                    effect="effects/core.effect",
                    extra_effects=["effects/flame.effect"],
                    light=ui.thruster_light((1.0, 0.5, 0.1), 1_000_000.0),
                )
            ],
            visibility=visibility,
        ),
        ui.window(rect=(0, 0, 50, 100)),
        environment=ui.environment(
            sun=ui.sun(direction=(0.2, -0.8, 0.4)),
            ambient=0.05,
            sky=ui.color(135, 206, 235),
            atmosphere=ui.atmosphere(),
            earth=ui.earth(),
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
    import urllib.error
    import urllib.request

    addr = "127.0.0.1:23551"
    db_path = tmp_path / "db"
    server = edb.Server.start(str(db_path), addr)
    time.sleep(0.3)
    try:
        s = ui.schematic(ui.graph("drone.thrust", name="Thrust"))
        ui.push(s, addr)
        asset = db_path / "assets" / "schematics" / "main.kdl"
        deadline = time.time() + 5
        body = None
        while time.time() < deadline:
            if asset.exists():
                body = asset.read_text()
                break
            time.sleep(0.1)
        if body is None:
            pytest.fail(f"asset not written after push: missing {asset}")
        assert "drone.thrust" in body

        # Editor loads the active schematic from the Asset Server on N+1.
        url = "http://127.0.0.1:23552/schematics/main.kdl"
        http_deadline = time.time() + 5
        last_err: Exception | None = None
        while time.time() < http_deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    fetched = resp.read().decode()
                assert "drone.thrust" in fetched
                return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                time.sleep(0.1)
        pytest.fail(f"asset HTTP GET {url} failed: {last_err}")
    finally:
        server.stop()
