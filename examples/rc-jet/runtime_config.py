"""RC-jet presets, resolved at entry points and passed explicitly to builders."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class TerrainMode(str, Enum):
    PLANAR = "planar"
    CESIUM = "cesium"


@dataclass(frozen=True)
class CameraSettings:
    color_resolution: tuple[int, int] = (640, 480)
    infrared_resolution: tuple[int, int] = (640, 512)
    color_shadows: bool = True
    color_fps: float = 30.0
    color_format: str = "rgba"
    infrared_fps: float = 60.0
    infrared_format: str = "gray8"


RAW_CAMERAS = CameraSettings()
# Keep both previews on the same cadence and reduce pixel work on shared GPUs.
PREVIEW_CAMERAS = CameraSettings(
    color_resolution=(480, 360),
    infrared_resolution=(480, 384),
    color_shadows=False,
    color_fps=24.0,
    color_format="h264",
    infrared_fps=24.0,
    infrared_format="h264",
)


@dataclass(frozen=True)
class RuntimeConfig:
    terrain: TerrainMode = TerrainMode.PLANAR
    cameras: CameraSettings = RAW_CAMERAS
    height_asset_id: int = 1


def load_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Keep the launcher presets; reject typos instead of silently using planar terrain."""
    environ = os.environ if environ is None else environ
    value = environ.get("ELODIN_RC_JET_TERRAIN", "planar")
    try:
        terrain = TerrainMode(value)
    except ValueError:
        raise ValueError(
            f"ELODIN_RC_JET_TERRAIN must be 'planar' or 'cesium', got {value!r}"
        ) from None
    height_asset = environ.get("ELODIN_RC_JET_HEIGHT_ASSET_ID", "1")
    try:
        height_asset_id = int(height_asset)
        if height_asset_id <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("ELODIN_RC_JET_HEIGHT_ASSET_ID must be a positive integer") from None
    return RuntimeConfig(
        terrain=terrain,
        cameras=PREVIEW_CAMERAS if terrain is TerrainMode.CESIUM else RAW_CAMERAS,
        height_asset_id=height_asset_id,
    )
