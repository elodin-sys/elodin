"""GPU/Elodin-free startup matrix for the merged Package B + E options."""

import pytest

from controls import GuidanceMode
from race_options import race_options_from_env


@pytest.mark.parametrize(
    ("guidance", "course", "camera"),
    [
        ("scripted", "none", "0"),
        ("scripted", "single", "1"),
        ("manual", "none", "0"),
        ("manual", "single", "1"),
        ("truth", "single", "0"),
        ("truth", "single", "1"),
    ],
)
def test_camera_is_independent_of_command_source_and_course(
    guidance: str, course: str, camera: str
) -> None:
    options = race_options_from_env(
        {"RACE_GUIDANCE": guidance, "RACE_COURSE": course, "RACE_CAMERA": camera}
    )
    assert options.guidance is GuidanceMode(guidance)
    assert options.course.name == course
    assert options.camera_enabled is (camera == "1")


def test_default_startup_is_scripted_camera_off_without_course() -> None:
    options = race_options_from_env({})
    assert options.guidance is GuidanceMode.SCRIPTED
    assert options.course.name == "none"
    assert not options.camera_enabled


@pytest.mark.parametrize("camera", ("0", "1"))
def test_truth_without_single_course_fails_with_or_without_camera(camera: str) -> None:
    with pytest.raises(ValueError, match="truth.*requires RACE_COURSE='single'"):
        race_options_from_env({"RACE_GUIDANCE": "truth", "RACE_CAMERA": camera})


@pytest.mark.parametrize(
    ("env", "error"),
    [
        ({"RACE_GUIDANCE": "vision", "RACE_COURSE": "single", "RACE_CAMERA": "1"}, "reserved"),
        ({"RACE_GUIDANCE": "unknown"}, "unknown RACE_GUIDANCE"),
        ({"RACE_COURSE": "unknown"}, "unknown RACE_COURSE"),
        ({"RACE_COURSE": "c1_straight"}, "reserved for Package F"),
        ({"RACE_CAMERA": "yes"}, "RACE_CAMERA must be '0' or '1'"),
    ],
)
def test_unimplemented_or_unknown_values_fail_at_startup(env: dict[str, str], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        race_options_from_env(env)
