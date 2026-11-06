"""C1 scene, fixed-width referee telemetry, and shutdown adapter contracts."""

# Only trusted repository AST is executed to isolate the entrypoint's I/O seams.
# ruff: noqa: S102

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from controls import RcCommand
from course import course_from_name
from course_referee import TruthCourseReferee
from exit_status import CallbackExecution, scenario_exit_code
from referee import Referee
from schematic import build_schematic

MAIN = Path(__file__).parents[1] / "main.py"


def test_phase_telemetry_metadata_matches_shared_controller_codes():
    from guidance import GUIDANCE_PHASE_NAMES

    tree = ast.parse((MAIN.parent / "sim.py").read_text())
    metadata = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "enum_values" for key in node.keys)
    )
    values = ast.literal_eval(metadata)["enum_values"]
    assert values == ",".join(f"{code}={name}" for code, name in GUIDANCE_PHASE_NAMES.items())


def test_c1_schematic_has_twelve_distinct_static_bars_and_phase_rc_graphs():
    course = course_from_name("c1_straight")
    kdl = build_schematic(course, audit_enabled=False, camera_enabled=False).emit_kdl()
    assert kdl.count("object_3d gate_") == 12
    for i in range(3):
        assert kdl.count(f"object_3d gate_{i}_") == 4
    assert kdl.count("drone.guidance_mode") == 1
    assert kdl.count("drone.rc_command") == 1
    assert "sensor_view" not in kdl


def test_c1_camera_plane_extends_past_last_gate_and_remains_camera_only():
    course = course_from_name("c1_straight")
    camera_kdl = build_schematic(course, audit_enabled=False, camera_enabled=True).emit_kdl()
    assert "plane width=80.0 depth=80.0" in camera_kdl
    assert course.gates[-1].center[0] < 80 / 2
    assert (
        "plane" not in build_schematic(course, audit_enabled=False, camera_enabled=False).emit_kdl()
    )
    for name in ("none", "single"):
        kdl = build_schematic(
            course_from_name(name), audit_enabled=False, camera_enabled=True
        ).emit_kdl()
        assert "plane width=40.0 depth=40.0" in kdl


def test_spawned_c1_bars_have_world_pose_only_never_body_inertia_or_force():
    from race_runtime import spawn_course

    class World:
        def __init__(self):
            self.entities = []

        def spawn(self, value, name):
            self.entities.append((name, value))
            return len(self.entities)

    world = World()
    assert len(spawn_course(world, course_from_name("c1_straight"))) == 12
    assert len({name for name, _ in world.entities}) == 12
    for _, entity in world.entities:
        assert tuple(entity.__dataclass_fields__) == ("world_pos",)
        assert not any(hasattr(entity, field) for field in ("inertia", "force", "world_vel"))


def test_c1_ordered_crossings_and_all_intermediate_fixed_width_telemetry():
    referee = Referee(course_from_name("c1_straight"))
    assert referee.telemetry_pass_times() == (-1, -1, -1)
    # Flying through gate 2 before gate 0 cannot score a later gate.
    referee.observe_truth((29, 0, 1.8), 0)
    assert referee.observe_truth((31, 0, 1.8), 1) is None
    assert referee.telemetry_pass_times() == (-1, -1, -1)
    times = []
    for i in range(3):
        start = 2 + 2 * i
        x = 10 + 10 * i
        assert referee.observe_truth((x - 1, 0, 1.8), start) is None
        event = referee.observe_truth((x + 1, 0, 1.8), start + 1)
        assert event.gate_index == i and event.pass_time == start + 0.5
        times.append(event.pass_time)
        assert referee.telemetry_pass_times() == tuple(times + [-1.0] * (2 - i))
        assert referee.progress().last_gate_passed == i
        assert referee.progress().next_gate_index == (i + 1 if i < 2 else None)
    result = referee.result()
    assert result.pass_times == tuple(times)
    assert result.format() == (
        "[RACE] course=c1_straight gates_passed=3/3 lap_time=6.500000 "
        "status=COMPLETE pass_times=[2.500000,4.500000,6.500000]"
    )
    referee.observe_truth((29, 0, 1.8), 8)
    assert referee.observe_truth((31, 0, 1.8), 9) is None
    assert referee.result() == result


def shutdown_namespace(*, execution_started=True):
    """Load only real shutdown functions/block, without starting Elodin/SITL.

    The entrypoint is an I/O module. Extracting its small actual shutdown block
    lets exceptions and exactly-once emissions be tested without duplicating it.
    """
    tree = ast.parse(MAIN.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("emit_race_result", "emit_final_results", "finish_recording", "sitl_post_step")
    ]
    run_try = next(n for n in tree.body if isinstance(n, ast.Try) and n.finalbody)
    ns = {
        "race_course": course_from_name("c1_straight"),
        "referee": Referee(course_from_name("c1_straight")),
        "race_result_emitted": [False],
        "race_result_line_count": [0],
        "referee_audit_requested": False,
        "truth_hold_referee": None,
        "command_source": SimpleNamespace(
            phase=SimpleNamespace(name="FINISH"), failure_reason=None
        ),
        "truth_course_referee": TruthCourseReferee(max_sample_gap_s=0.001),
        "truth_course_result": [None],
        "truth_course_result_emitted": [False],
        "bridge": [object() if execution_started else None],
        "recorder_box": [None],
        "simulation_started": [execution_started],
        "callback_execution": CallbackExecution(2),
        "MAX_TICKS": 2,
        "el": SimpleNamespace(StepContext=object),
        "system": None,
        "config": SimpleNamespace(pid_rate=8000, total_sim_ticks=480000),
        "sitl_post_step": None,
        "db_filename": "unused-test-db",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN), "exec"), ns)
    return ns, compile(ast.Module(body=[run_try], type_ignores=[]), str(MAIN), "exec")


def assert_actual_entrypoint_exit(ns, expected_exit):
    """Exercise the real final exit branch, not a test-only aggregation copy."""
    tree = ast.parse(MAIN.read_text())
    final_exit = tree.body[-1]
    assert isinstance(final_exit, ast.If)
    ns.update(
        sys=sys,
        scenario_exit_code=scenario_exit_code,
        c0_result=[None],
        truth_hold_result=[None],
        referee_audit_result=[None],
        axis_audit=None,
        fpv_report=[None],
    )
    code = compile(ast.Module(body=[final_exit], type_ignores=[]), str(MAIN), "exec")
    if expected_exit:
        with pytest.raises(SystemExit) as error:
            exec(code, ns)
        assert error.value.code == expected_exit
    else:
        exec(code, ns)


@pytest.mark.parametrize("exception", (RuntimeError, KeyboardInterrupt))
def test_interrupted_execution_emits_exactly_once_and_preserves_nonzero_exception(
    exception, capsys
):
    ns, run = shutdown_namespace()

    def fail(*args, **kwargs):
        raise exception("interrupted after execution started")

    ns["world"] = SimpleNamespace(run=fail)
    with pytest.raises(exception):
        exec(run, ns)
    ns["emit_final_results"]()  # idempotent even after exceptional callback paths
    output = capsys.readouterr().out
    assert output.count("[RACE]") == output.count("[F-COURSE]") == 1
    assert "status=INCOMPLETE" in output and "status=FAIL" in output
    assert ns["truth_course_result"][0].exit_code == 1


def test_recipe_generation_without_execution_emits_no_final_result(capsys):
    ns, run = shutdown_namespace(execution_started=False)
    ns["world"] = SimpleNamespace(run=lambda *a, **kw: None)
    exec(run, ns)
    assert capsys.readouterr().out == ""


def test_swallowed_first_callback_bridge_setup_failure_is_execution_not_recipe_generation(capsys):
    ns, run = shutdown_namespace(execution_started=False)

    def fail_before_bridge(*args):
        raise RuntimeError("bridge constructor failed before bridge assignment")

    ns["_sitl_post_step"] = fail_before_bridge

    def sdk_run(*args, **kwargs):
        try:
            kwargs["post_step"](0, None)
        except RuntimeError:
            pass  # SDK logs/swallowing a setup exception must still fail F.

    ns["world"] = SimpleNamespace(run=sdk_run)
    exec(run, ns)
    assert ns["bridge"] == [None]
    assert ns["simulation_started"] == [True]
    assert ns["truth_course_result"][0].exit_code == 1
    assert ns["callback_execution"].failure_reason == "callback_error_RuntimeError"
    assert_actual_entrypoint_exit(ns, 1)
    output = capsys.readouterr().out
    assert output.count("[RACE]") == output.count("[F-COURSE]") == 1


def test_normal_callback_and_finally_share_exactly_once_final_result(capsys):
    ns, run = shutdown_namespace()
    ns["world"] = SimpleNamespace(run=lambda *a, **kw: ns["emit_final_results"]())
    exec(run, ns)
    output = capsys.readouterr().out
    assert output.count("[RACE]") == output.count("[F-COURSE]") == 1
    assert ns["race_result_line_count"] == [1]


def qualifying_shutdown_namespace():
    ns, run = shutdown_namespace()
    referee = ns["referee"]
    for i in range(3):
        referee.observe_truth((9 + 10 * i, 0, 1.8), 20 + 5 * i)
        referee.observe_truth((11 + 10 * i, 0, 1.8), 21 + 5 * i)
    grader = TruthCourseReferee(max_sample_gap_s=1.01)
    for t in (57, 58, 59, 59.999875):
        airborne = t == 57
        grader.observe_truth(
            t,
            (0, 0, 0, 1, 32, 0, 1.8 if airborne else 0),
            (0,) * 6,
            RcCommand(arm=1800, throttle=1130) if airborne else RcCommand(),
            (0.13,) * 4 if airborne else (0,) * 4,
            lockstep_response=True,
            race_complete=True,
            guidance_fault=False,
        )
    assert grader.result(referee.result()).passed  # physical evidence really qualifies
    ns["truth_course_referee"] = grader
    return ns, run


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("failure_site", ("bridge_stop", "stop_recipes"))
def test_real_final_callback_cleanup_error_stays_failed_when_sdk_swallows_it(
    error_type, failure_site, capsys
):
    ns, run = qualifying_shutdown_namespace()
    tree = ast.parse(MAIN.read_text())
    body = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sitl_post_step"
    )
    final = body.body[-1]
    assert isinstance(final, ast.If) and "MAX_TICKS" in ast.unparse(final.test)
    final_code = compile(ast.Module(body=[final], type_ignores=[]), str(MAIN), "exec")

    def fail():
        raise error_type("injected final callback cleanup failure")

    ns["b"] = SimpleNamespace(stop=fail if failure_site == "bridge_stop" else lambda: None)
    ctx = SimpleNamespace(stop_recipes=fail)

    def real_final_body(tick, context):
        ns.update(tick=tick, ctx=context)
        exec(final_code, ns)

    ns["_sitl_post_step"] = real_final_body

    def sdk_run(*args, **kwargs):
        for tick in range(2):
            try:
                kwargs["post_step"](tick, ctx)
            except BaseException as error:  # noqa: BLE001 - model SDK's PyErr catch boundary
                # Actual SDK PyErr boundary logs callback failures and continues.
                print(f"SDK logged (and swallowed): {type(error).__name__}")

    ns["world"] = SimpleNamespace(run=sdk_run)
    exec(run, ns)
    result = ns["truth_course_result"][0]
    output = capsys.readouterr().out
    assert output.count("[RACE]") == output.count("[F-COURSE]") == 1
    assert "status=COMPLETE" in output and "status=FAIL" in output
    assert result.failure_reason == f"callback_error_{error_type.__name__}"
    assert (
        scenario_exit_code(
            course_exit_code=result.exit_code,
            execution_exit_code=ns["callback_execution"].exit_code,
        )
        == 1
    )


@pytest.mark.parametrize("ticks", ((0,), (1,), (0, 1)))
def test_actual_adapter_checks_expected_normal_end_not_only_last_physical_sample(ticks, capsys):
    ns, run = qualifying_shutdown_namespace()
    ns["_sitl_post_step"] = lambda *a: None

    def sdk_run(*args, **kwargs):
        for tick in ticks:
            kwargs["post_step"](tick, None)

    ns["world"] = SimpleNamespace(run=sdk_run)
    exec(run, ns)
    result = ns["truth_course_result"][0]
    expected_exit = 0 if ticks == (0, 1) else 1
    assert result.exit_code == ns["callback_execution"].exit_code == expected_exit
    assert ns["callback_execution"].normal_completion is (expected_exit == 0)
    assert_actual_entrypoint_exit(ns, expected_exit)
    output = capsys.readouterr().out
    assert output.count("[RACE]") == output.count("[F-COURSE]") == 1
    if ticks == (0,):
        assert "callback_sequence_incomplete" in output
