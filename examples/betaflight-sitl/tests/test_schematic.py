"""Pure tests for the Betaflight SITL editor schematic variants."""

from course import course_from_name, gate_schematic
from schematic import build_schematic


def test_default_schematic_keeps_chase_camera_and_sensor_graphs() -> None:
    schematic = build_schematic(course_from_name("none"), audit_enabled=False)

    assert 'pos="drone.world_pos + (0,0,0,0, 10,10,5)" look_at="drone.world_pos"' in schematic
    assert 'graph "drone.accel" name="Accelerometer"' in schematic
    assert 'graph "drone.gyro" name="Gyroscope"' in schematic
    assert "object_3d gate_" not in schematic
    assert 'graph "drone.last_gate_passed"' not in schematic


def test_course_schematic_frames_gate_without_changing_sensor_graphs() -> None:
    course = course_from_name("single")
    schematic = build_schematic(course, audit_enabled=False)

    assert 'pos="(0,0,0,1, 4,-2,3)" look_at="(10,0,1.8)"' in schematic
    assert 'graph "drone.accel" name="Accelerometer"' in schematic
    assert 'graph "drone.gyro" name="Gyroscope"' in schematic
    assert schematic.count("object_3d gate_0_") == 4
    assert gate_schematic(course) in schematic


def test_audit_schematic_uses_oblique_camera_and_referee_graphs() -> None:
    course = course_from_name("single")
    schematic = build_schematic(course, audit_enabled=True)

    assert 'pos="(0,0,0,1, 3,-4,3.5)" look_at="(9,0,2)"' in schematic
    assert 'graph "drone.last_gate_passed" name="Referee: Last Gate Passed"' in schematic
    assert 'graph "drone.gate_pass_times" name="Referee: Gate Pass Times"' in schematic
    assert 'graph "drone.accel" name="Accelerometer"' not in schematic
    assert 'graph "drone.gyro" name="Gyroscope"' not in schematic
    assert schematic.count("object_3d gate_0_") == 4


def test_camera_schematic_injects_only_the_fpv_fragments() -> None:
    course = course_from_name("none")
    disabled = build_schematic(course, audit_enabled=False)
    enabled = build_schematic(course, audit_enabled=False, camera_enabled=True)

    assert 'sensor_view "drone.fpv"' not in disabled
    assert "show_frustums=#true" not in disabled
    assert "plane width=40" not in disabled
    assert 'sensor_view "drone.fpv" name="FPV Camera"' in enabled
    assert "show_frustums=#true" in enabled
    assert "plane width=40 depth=40" in enabled
    assert enabled.count('graph "drone.motor_command"') == 1
    assert enabled.count('graph "drone.accel"') == 1
    assert 'pos="drone.world_pos + (0,0,0,0, 10,10,5)"' in enabled


def test_camera_and_course_schematic_combine() -> None:
    course = course_from_name("single")
    schematic = build_schematic(course, audit_enabled=False, camera_enabled=True)

    assert 'sensor_view "drone.fpv"' in schematic
    assert schematic.count("object_3d gate_0_") == 4
    assert 'pos="(0,0,0,1, 4,-2,3)"' in schematic
