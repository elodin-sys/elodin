"""Entry-point configuration must not silently select a different terrain."""

from dataclasses import replace

import pytest

from bdx_schematic import build
from runtime_config import PREVIEW_CAMERAS, RAW_CAMERAS, RuntimeConfig, TerrainMode, load_config


def test_existing_presets():
    assert load_config({}) == RuntimeConfig(TerrainMode.PLANAR, RAW_CAMERAS)
    assert load_config({"ELODIN_RC_JET_TERRAIN": "planar"}) == load_config({})
    assert load_config({"ELODIN_RC_JET_TERRAIN": "cesium"}) == RuntimeConfig(
        TerrainMode.CESIUM, PREVIEW_CAMERAS
    )


@pytest.mark.parametrize("mode", ["", "Cesium", "cesuim"])
def test_invalid_mode_is_not_a_silent_planar_fallback(mode):
    with pytest.raises(ValueError, match="ELODIN_RC_JET_TERRAIN must be"):
        load_config({"ELODIN_RC_JET_TERRAIN": mode})


@pytest.mark.parametrize("mode", ["planar", "cesium"])
def test_explicit_config_is_independent_of_later_environment_changes(monkeypatch, mode):
    monkeypatch.setenv("ELODIN_RC_JET_TERRAIN", mode)
    config = load_config()
    expected = build().emit_kdl()  # Standalone/watch entry point uses the same preset.
    alternate_cameras = RAW_CAMERAS if mode == "cesium" else PREVIEW_CAMERAS
    config = replace(config, cameras=alternate_cameras)
    monkeypatch.setenv("ELODIN_RC_JET_TERRAIN", "invalid")
    assert build(config).emit_kdl() == expected
    with pytest.raises(ValueError, match="ELODIN_RC_JET_TERRAIN must be"):
        build()


@pytest.mark.parametrize("asset", ["", "0", "-1", "1.5", "wrong"])
def test_invalid_height_source_is_rejected_before_world_creation(asset):
    with pytest.raises(ValueError, match="HEIGHT_ASSET_ID must be a positive integer"):
        load_config({"ELODIN_RC_JET_HEIGHT_ASSET_ID": asset})


def test_rendering_asset_cannot_change_the_physics_height_source():
    config = load_config(
        {
            "ELODIN_RC_JET_TERRAIN": "cesium",
            "CESIUM_ION_ASSET_ID": "2275207",
        }
    )
    assert config.height_asset_id == 1
    assert load_config({"ELODIN_RC_JET_HEIGHT_ASSET_ID": "42"}).height_asset_id == 42
