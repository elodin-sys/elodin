"""Regression checks for the n-body example's Python schematic migration."""

from pathlib import Path

import elodin.ui as ui
import pytest
from elodin.ui.watch import _build_schematic, _load_module

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "n-body"
LEGACY_KDL = Path(__file__).with_name("fixtures") / "nbody_schematic.kdl"


@pytest.fixture
def nbody(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return _load_module(EXAMPLE / "schematic.py")


def test_nbody_schematic_matches_legacy_kdl():
    # Use the same zero-argument entry point as the CLI and watcher. This must
    # work without first calling sim.build_world() or initializing sim.BODIES.
    built = _build_schematic(EXAMPLE / "schematic.py")
    source = ui.from_kdl(LEGACY_KDL.read_text())
    assert ui.from_kdl(built.emit_kdl()) == ui.from_kdl(source.emit_kdl())


def test_nbody_schematic_build_is_deterministic(nbody):
    assert nbody.build().emit_kdl() == nbody.build().emit_kdl()
    parsed = ui.from_kdl(nbody.build([]).emit_kdl())
    assert ui.from_kdl(parsed.emit_kdl()) == parsed


@pytest.mark.parametrize("name,naif_id", [("moon", 301), ("helene", 612)])
def test_nbody_schematic_uses_supplied_bodies(nbody, name, naif_id):
    from body_metadata import BODY_META, Body

    body = Body(name, naif_id, BODY_META[name])
    built = nbody.build([body])
    kdl = built.emit_kdl()
    parsed = ui.from_kdl(kdl)
    assert ui.from_kdl(parsed.emit_kdl()) == parsed
    assert f"object_3d {name}.world_pos" in kdl
    assert f"object_3d truth_{name}.truth_world_pos" in kdl
    assert "object_3d earth.world_pos" not in kdl
    assert "builtin=circle" in kdl
    assert "color 180 180 180 120" in kdl
    assert "color 180 180 180 80" in kdl


@pytest.mark.parametrize("include_moons", [False, True])
def test_nbody_standalone_discovery_matches_simulation(nbody, include_moons):
    from body_metadata import CSV_PATHS, load_bodies

    paths = CSV_PATHS
    if include_moons:
        paths = (*paths, EXAMPLE / "moons_truth.csv")
    sim = _load_module(EXAMPLE / "sim.py")
    sim.load_truth(paths)
    assert load_bodies(paths) == sim.BODIES


def test_nbody_world_registers_python_schematic(nbody, monkeypatch):
    sim = _load_module(EXAMPLE / "sim.py")
    registered = []
    world = sim.el.World()

    class World:
        def spawn(self, *args, **kwargs):
            return world.spawn(*args, **kwargs)

        def schematic(self, schematic, path):
            registered.append((schematic, path))
            world.schematic(schematic, path)

    monkeypatch.setattr(sim.el, "World", World)
    sim.build_world()
    [(schematic, path)] = registered
    assert isinstance(schematic, ui.Schematic)
    assert path == "solar-system.kdl"
    source = ui.from_kdl(LEGACY_KDL.read_text())
    assert ui.from_kdl(schematic.emit_kdl()) == ui.from_kdl(source.emit_kdl())
