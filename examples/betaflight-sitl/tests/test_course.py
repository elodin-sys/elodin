import math

import pytest
from course import (
    GATE_BAR_THICKNESS_M,
    SATURATED_ORANGE_RGB,
    SINGLE_GATE_INNER_SIZE_M,
    Gate,
    course_bars,
    course_from_env,
    course_from_name,
    gate_schematic,
)


@pytest.mark.parametrize("env", [{}, {"RACE_COURSE": "none"}])
def test_no_course_is_default_or_explicit_and_has_no_geometry(env):
    course = course_from_env(env)

    assert course.name == "none"
    assert course.gates == ()
    assert course.inner_size is None
    assert course_bars(course) == ()
    assert gate_schematic(course) == ""


def test_single_course_has_exact_contract_geometry():
    course = course_from_name("single")

    assert course.name == "single"
    assert len(course.gates) == 1
    gate = course.gates[0]
    assert gate.index == 0
    assert gate.center == (10.0, 0.0, 1.8)
    assert gate.yaw == 0.0
    assert gate.normal == (1.0, 0.0, 0.0)
    assert gate.lateral == (0.0, 1.0, 0.0)
    assert gate.inner_size == SINGLE_GATE_INNER_SIZE_M == 2.5


def test_c1_straight_is_reserved_until_package_f():
    with pytest.raises(ValueError, match="reserved for Package F.*not implemented"):
        course_from_name("c1_straight")


def test_unknown_course_is_rejected_clearly():
    with pytest.raises(ValueError, match=r"unknown RACE_COURSE='oval'"):
        course_from_env({"RACE_COURSE": "oval"})


def test_yawed_gate_local_basis_round_trips():
    gate = Gate(index=0, center=(3.0, -2.0, 5.0), yaw=math.pi / 2.0, inner_size=2.5)

    world = gate.local_to_world((4.0, 1.0, -0.5))

    assert world == pytest.approx((2.0, 2.0, 4.5), abs=1e-12)
    assert gate.world_to_local(world) == pytest.approx((4.0, 1.0, -0.5), abs=1e-12)
    assert gate.normal == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert gate.lateral == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)


def test_gate_render_geometry_is_four_static_box_poses_with_exact_opening():
    course = course_from_name("single")
    bars = course_bars(course)

    assert [bar.name for bar in bars] == [
        "gate_0_top",
        "gate_0_bottom",
        "gate_0_left",
        "gate_0_right",
    ]
    assert len(bars) == 4
    top, bottom, left, right = bars
    half_inner = SINGLE_GATE_INNER_SIZE_M / 2.0
    half_bar = GATE_BAR_THICKNESS_M / 2.0

    # Inner edges are exactly z=1.8 +/- 1.25 and y=+/-1.25.
    assert top.center[2] - top.size[2] / 2.0 == pytest.approx(1.8 + half_inner)
    assert bottom.center[2] + bottom.size[2] / 2.0 == pytest.approx(1.8 - half_inner)
    assert left.center[1] - left.size[1] / 2.0 == pytest.approx(half_inner)
    assert right.center[1] + right.size[1] / 2.0 == pytest.approx(-half_inner)
    assert top.center == pytest.approx((10.0, 0.0, 1.8 + half_inner + half_bar))
    assert all(bar.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0) for bar in bars)


def test_gate_schematic_has_four_saturated_orange_matte_boxes():
    schematic = gate_schematic(course_from_name("single"))

    assert schematic.count("object_3d gate_0_") == 4
    assert schematic.count("orientation=absolute") == 4
    assert schematic.count("box x=") == 4
    rgb = " ".join(str(channel) for channel in SATURATED_ORANGE_RGB)
    assert schematic.count(f"color {rgb}") == 4
    assert "emissivity" not in schematic
