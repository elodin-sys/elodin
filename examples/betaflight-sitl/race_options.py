"""Pure startup selection for the Betaflight racing example.

Camera selection is independent of guidance and course; truth guidance requires
an enabled course. Invalid/reserved values fail before recipes run.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from controls import GuidanceMode, guidance_mode_from_env
from course import Course, course_from_env


@dataclass(frozen=True, slots=True)
class RaceOptions:
    guidance: GuidanceMode
    course: Course
    camera_enabled: bool
    record_dir: Path | None = None


def _record_dir(environ: Mapping[str, str], *, camera_enabled: bool) -> Path | None:
    """Validate ``RACE_RECORD``. Unset means do not record."""

    if "RACE_RECORD" not in environ:
        return None
    raw = environ["RACE_RECORD"].strip()
    if raw == "":
        raise ValueError("RACE_RECORD must be a non-empty directory path")
    if not camera_enabled:
        raise ValueError("RACE_RECORD requires RACE_CAMERA=1")
    path = Path(raw).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"RACE_RECORD path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"RACE_RECORD directory is not empty: {path}")
    return path


def race_options_from_env(environ: Mapping[str, str] = os.environ) -> RaceOptions:
    guidance = guidance_mode_from_env(environ)
    course = course_from_env(environ)
    if guidance is GuidanceMode.TRUTH and course.name == "none":
        raise ValueError("RACE_GUIDANCE='truth' requires RACE_COURSE='single' or 'c1_straight'")

    camera = environ.get("RACE_CAMERA", "0")
    if camera not in ("0", "1"):
        raise ValueError(f"RACE_CAMERA must be '0' or '1', got {camera!r}")
    camera_enabled = camera == "1"
    return RaceOptions(
        guidance,
        course,
        camera_enabled,
        _record_dir(environ, camera_enabled=camera_enabled),
    )
