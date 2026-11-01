from pathlib import Path

import pytest

import bdx_schematic
import visual_check_schematic
from terrain_schematic import TERRAIN_MODE_ENV


SCHEMATICS = (bdx_schematic, visual_check_schematic)
TERRAIN_ASSET = Path(__file__).resolve().parents[3] / "assets" / "mojave_rc_field.glb"


@pytest.mark.parametrize("schematic", SCHEMATICS)
def test_static_terrain_is_the_default(monkeypatch, schematic):
    monkeypatch.delenv(TERRAIN_MODE_ENV, raising=False)
    kdl = schematic.build().emit_kdl()

    assert "glb path=mojave_rc_field.glb" in kdl
    assert "world_mesh mojave_rc_field" not in kdl
    assert "frame=ECEF orientation=absolute" in kdl


@pytest.mark.parametrize("schematic", SCHEMATICS)
def test_generated_atlas_is_opt_in(monkeypatch, schematic):
    monkeypatch.setenv(TERRAIN_MODE_ENV, "atlas")
    kdl = schematic.build().emit_kdl()

    assert 'world_mesh mojave_rc_field translate="(0, 0, 528)" frame=ENU' in kdl
    assert "mojave_rc_field.glb" not in kdl


def test_static_terrain_is_a_materialized_glb():
    with TERRAIN_ASSET.open("rb") as terrain:
        assert terrain.read(4) == b"glTF"


def test_unknown_terrain_mode_is_rejected(monkeypatch):
    monkeypatch.setenv(TERRAIN_MODE_ENV, "unknown")
    with pytest.raises(ValueError, match=TERRAIN_MODE_ENV):
        bdx_schematic.build()
