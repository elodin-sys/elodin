"""Pure startup selection for the Betaflight racing example.

Camera selection is independent of guidance and course; only Package E truth
selection constrains the course. Invalid/reserved values fail before recipes run.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from controls import GuidanceMode, guidance_mode_from_env
from course import Course, course_from_env


@dataclass(frozen=True, slots=True)
class RaceOptions:
    guidance: GuidanceMode
    course: Course
    camera_enabled: bool


def race_options_from_env(environ: Mapping[str, str] = os.environ) -> RaceOptions:
    guidance = guidance_mode_from_env(environ)
    course = course_from_env(environ)
    if guidance is GuidanceMode.TRUTH and course.name != "single":
        raise ValueError("RACE_GUIDANCE='truth' currently requires RACE_COURSE='single'")

    camera = environ.get("RACE_CAMERA", "0")
    if camera not in ("0", "1"):
        raise ValueError(f"RACE_CAMERA must be '0' or '1', got {camera!r}")
    return RaceOptions(guidance, course, camera == "1")
