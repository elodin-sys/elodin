from pathlib import Path

import bdx_schematic
import pytest
import visual_check_schematic

SCHEMATICS = (bdx_schematic, visual_check_schematic)
TERRAIN_ASSET = Path(__file__).resolve().parents[3] / "assets" / "mojave_rc_field.glb"


@pytest.mark.parametrize("schematic", SCHEMATICS)
def test_schematic_uses_static_terrain(schematic):
    kdl = schematic.build().emit_kdl()

    assert "glb path=mojave_rc_field.glb" in kdl
    assert "world_mesh mojave_rc_field" not in kdl
    assert "frame=ECEF orientation=absolute" in kdl


def test_static_terrain_is_a_materialized_glb():
    with TERRAIN_ASSET.open("rb") as terrain:
        assert terrain.read(4) == b"glTF"
