"""Tests for the Betaflight SITL editor schematic variants."""

from course import course_from_name
from schematic import build_schematic


def _kdl(course_name: str, *, audit_enabled: bool, camera_enabled: bool = False) -> str:
    course = course_from_name(course_name)
    return build_schematic(
        course, audit_enabled=audit_enabled, camera_enabled=camera_enabled
    ).emit_kdl()


def test_default_schematic_keeps_chase_camera_and_sensor_graphs() -> None:
    schematic = _kdl("none", audit_enabled=False)

    assert 'pos="drone.world_pos + (0,0,0,0, 10,10,5)"' in schematic
    assert "look_at=drone.world_pos" in schematic
    assert "name=Accelerometer" in schematic
    assert "name=Gyroscope" in schematic
    assert "object_3d gate_" not in schematic
    assert "last_gate_passed" not in schematic
    assert "show_frustums" not in schematic


def test_course_schematic_frames_gate_without_changing_sensor_graphs() -> None:
    schematic = _kdl("single", audit_enabled=False)

    assert 'pos="(0,0,0,1, 4,-2,3)"' in schematic
    assert 'look_at="(10,0,1.8)"' in schematic
    assert "name=Accelerometer" in schematic
    assert "name=Gyroscope" in schematic
    assert schematic.count("object_3d gate_0_") == 4


def test_audit_schematic_uses_oblique_camera_and_referee_graphs() -> None:
    schematic = _kdl("single", audit_enabled=True)

    assert 'pos="(0,0,0,1, 3,-4,3.5)"' in schematic
    assert 'look_at="(9,0,2)"' in schematic
    assert 'name="Referee: Last Gate Passed"' in schematic
    assert 'name="Referee: Gate Pass Times"' in schematic
    assert "name=Accelerometer" not in schematic
    assert "name=Gyroscope" not in schematic
    assert schematic.count("object_3d gate_0_") == 4


def test_camera_schematic_injects_only_the_fpv_fragments() -> None:
    disabled = _kdl("none", audit_enabled=False)
    enabled = _kdl("none", audit_enabled=False, camera_enabled=True)

    assert "sensor_view drone.fpv" not in disabled
    assert "sensor_visible=#false" not in disabled
    assert "plane " not in disabled
    assert "sensor_view drone.fpv" in enabled
    assert 'name="FPV Camera"' in enabled
    assert "show_frustums" not in enabled
    assert "sensor_visible=#false" in enabled
    assert enabled.count("sensor_visible=#false") == 1
    assert "plane " in enabled
    assert 'name="Motor Commands (from Betaflight)"' in enabled
    assert "name=Accelerometer" in enabled
    assert 'pos="drone.world_pos + (0,0,0,0, 10,10,5)"' in enabled


def test_camera_and_course_schematic_combine() -> None:
    schematic = _kdl("single", audit_enabled=False, camera_enabled=True)

    assert "sensor_view drone.fpv" in schematic
    assert schematic.count("object_3d gate_0_") == 4
    assert "sensor_visible=#false" in schematic
    assert schematic.count("sensor_visible=#false") == 1
    assert 'pos="(0,0,0,1, 4,-2,3)"' in schematic
