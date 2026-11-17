"""Exercise the actual callback with in-memory context/bridge, never live SITL.

Only trusted repository AST is loaded to avoid main.py startup/recipes.
"""
# ruff: noqa: S102

import ast
import dataclasses
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import controls
import numpy as np
import pytest
from baseline import MIN_MOTOR_RESPONSE
from comms import MAX_RC_CHANNELS, RCPacket
from controls import GatePose, GuidanceMode, SemanticControl, TruthGuidanceUpdate, TruthState
from course import course_from_name
from course_referee import TruthCourseReferee
from guidance import GuidancePhase, TruthGuidance, TruthGuidanceConfig
from guidance_cadence import GuidanceCadence, guidance_rate_from_env
from hold_referee import HoldEligibility, StandoffHoldReferee
from race_options import race_options_from_env
from referee import Referee, world_position_from_transform

MAIN = Path(__file__).parents[1] / "main.py"


@pytest.mark.parametrize(
    "environ,expected_hz",
    [
        ({}, 8000),
        ({"RACE_GUIDANCE": "truth", "RACE_COURSE": "single"}, 8000),
        ({"RACE_GUIDANCE": "scripted", "RACE_COURSE": "c1_straight"}, 8000),
        ({"RACE_GUIDANCE": "manual", "RACE_COURSE": "c1_straight"}, 8000),
        ({"RACE_GUIDANCE": "truth", "RACE_COURSE": "c1_straight"}, 100),
        ({"RACE_GUIDANCE": "truth", "RACE_COURSE": "c1_straight", "RACE_GUIDANCE_HZ": "60"}, 60),
        ({"RACE_GUIDANCE": "truth", "RACE_COURSE": "c1_straight", "RACE_GUIDANCE_HZ": "8000"}, 8000),
    ],
)
def test_actual_startup_selects_scenario_default_and_explicit_override(environ, expected_hz):
    # Execute the real configuration block, never the entrypoint's I/O/recipes.
    tree = ast.parse(MAIN.read_text())
    startup = next(node for node in tree.body if isinstance(node, ast.Try))
    ns = {
        "race_options_from_env": lambda: race_options_from_env(environ),
        "GuidanceMode": GuidanceMode,
        "GuidanceCadence": GuidanceCadence,
        "guidance_rate_from_env": lambda mode, hz, **kwargs: guidance_rate_from_env(
            mode, hz, environ, **kwargs
        ),
        "TruthGuidanceConfig": TruthGuidanceConfig,
        "config": SimpleNamespace(pid_rate=8000),
    }
    exec(compile(ast.Module(body=startup.body, type_ignores=[]), str(MAIN), "exec"), ns)
    cadence = ns["guidance_cadence"]
    assert cadence.hz == expected_hz
    assert cadence.full_rate is (expected_hz == 8000)


class MemoryContext:
    def __init__(self):
        self.data = {
            "drone.world_pos": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1]),
            "drone.world_vel": np.zeros(6),
            "drone.accel": np.zeros(3),
            "drone.gyro": np.zeros(3),
            "drone.baro": np.array([0.1]),
            "drone.mag": np.zeros(3),
        }
        self.writes = []
        self.reads = []
        self.fail_read = False
        self.stopped = False

    def component_batch_operation(self, *, reads=None, writes=None):
        if reads is not None:
            self.reads.append(tuple(reads))
            if self.fail_read:
                raise RuntimeError("injected sensor read failure")
            return {name: self.data[name].copy() for name in reads}
        for name, value in writes.items():
            self.write_component(name, value)

    def write_component(self, name, value):
        self.writes.append((name, np.asarray(value).copy()))
        self.data[name] = np.asarray(value).copy()

    def read_component(self, name):
        return self.data[name]

    def stop_recipes(self):
        self.stopped = True


class MemoryBridge:
    def __init__(self):
        self.step_count = 1  # Already initialized; no warmup/recipes.
        self.exchanges = []
        self.timeout = False
        self.no_response = False
        self.motors = np.full(4, 0.1)
        self.stopped = False

    def step(self, fdm, rc):
        self.exchanges.append((fdm.timestamp, rc.timestamp, tuple(rc.channels)))
        if self.timeout:
            raise TimeoutError("injected bridge timeout")
        if not self.no_response:
            self.step_count += 1
        return self.motors.copy()

    def stop(self):
        self.stopped = True


class MemoryBuffer:
    def update(self, **values):
        self.timestamp = values["timestamp"]

    def build_fdm(self):
        return SimpleNamespace(timestamp=self.timestamp)


def callback_namespace(rate=60, course_name="single", source_text=None, max_ticks=10000):
    course = course_from_name(course_name)
    source = TruthGuidance(
        dataclasses.replace(
            TruthGuidanceConfig.for_course(course_name),
            boot_grace_s=0,
        )
    )
    config = SimpleNamespace(dt=1 / 8000, pid_rate=8000, initial_position=(0, 0, 0.1))
    initial = SemanticControl.safe()
    bridge = MemoryBridge()
    referee = Referee(course)
    referee.observe_truth(config.initial_position, 0)
    referee.progress = Mock(wraps=referee.progress)
    referee.observe_truth = Mock(wraps=referee.observe_truth)
    hold = None
    course_grader = None
    if course_name == "single":
        hold = StandoffHoldReferee(course.gates[0], max_sample_gap_s=config.dt * 1.5)
        hold.observe_truth = Mock(wraps=hold.observe_truth)
    else:
        course_grader = TruthCourseReferee(max_sample_gap_s=config.dt * 1.5)
        course_grader.observe_truth = Mock(wraps=course_grader.observe_truth)
    ns = {
        "dataclass": dataclasses.dataclass,
        "np": np,
        "time": time,
        "el": SimpleNamespace(StepContext=object),
        "config": config,
        "initial_control": initial,
        "bridge": [bridge],
        "sensor_buf": [MemoryBuffer()],
        "start_time": [time.time()],
        "last_print": [0.0],
        "state": [None],
        "_zero_sensor_vector": np.zeros(3),
        "_barometer_tick_interval": 17,
        "_magnetometer_tick_interval": 40,
        "_manual_input_tick_interval": 8,
        "_component_reads": ["drone.accel", "drone.gyro", "drone.world_pos", "drone.world_vel"],
        "command_source": source,
        "guidance_mode": GuidanceMode.TRUTH,
        "guidance_cadence": GuidanceCadence(8000, None if rate is None else Fraction(rate)),
        "referee_audit_collector": None,
        "axis_audit": None,
        "fpv": None,
        "_recorder": lambda: None,
        "referee": referee,
        "truth_gate_poses": tuple(
            GatePose(g.index, g.center, g.yaw, g.inner_size) for g in course.gates
        ),
        "truth_hold_referee": hold,
        "truth_course_referee": course_grader,
        "MAX_TICKS": max_ticks,
        "race_course": course,
        "_rc_channels_buffer": np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16),
        "_rc_channels_command": [None],
        "audit_requested": False,
        "referee_audit_requested": False,
        "MIN_MOTOR_RESPONSE": MIN_MOTOR_RESPONSE,
        "HoldEligibility": HoldEligibility,
        "GuidancePhase": GuidancePhase,
        "AuditGuidance": type("AuditGuidance", (), {}),
        "world_position_from_transform": world_position_from_transform,
    }
    for name in (
        "DelayedRcCommand",
        "GuidanceMode",
        "GuidanceUpdate",
        "ManualGuidance",
        "RC_ANGLE",
        "RC_ARMED",
        "RcCommand",
        "ScriptedGuidance",
        "SemanticControl",
        "TruthGuidanceUpdate",
        "TruthState",
        "semantic_to_rc",
    ):
        ns[name] = getattr(controls, name)
    ns["_rc_packet"] = RCPacket(timestamp=0.0, channels=ns["_rc_channels_buffer"])
    tree = ast.parse(source_text or MAIN.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "SITLState")
        or (isinstance(node, ast.FunctionDef) and node.name == "_sitl_post_step")
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAIN), "exec"), ns)
    ns["state"][0] = ns["SITLState"]()
    return ns, MemoryContext()


def step(ns, ctx, tick):
    ns["_sitl_post_step"](tick, ctx)


@pytest.mark.parametrize("rate", [60, 100, 120])
def test_held_ticks_skip_guidance_inputs_and_progress_but_not_io_or_hold_grade(rate):
    ns, ctx = callback_namespace(rate)
    recorder = Mock()
    ns["_recorder"] = lambda: recorder
    samples = [
        SimpleNamespace(frame=object(), frame_time=tick / 8000, fresh=tick % 19 == 0)
        for tick in range(200)
    ]
    ns["fpv"] = Mock()
    ns["fpv"].poll.side_effect = samples
    ctx.read_msg_at, ctx.timestamp = Mock(), 0
    source = ns["command_source"]
    source.update = Mock(wraps=source.update)
    ns["TruthState"] = Mock(wraps=TruthState)
    ns["TruthGuidanceUpdate"] = Mock(wraps=TruthGuidanceUpdate)
    for tick in range(200):
        step(ns, ctx, tick)
    expected_ticks = {60: [0, 134], 100: [0, 80, 160], 120: [0, 67, 134]}[rate]
    assert [call.args[0].tick for call in source.update.call_args_list] == expected_ticks
    assert ns["TruthState"].call_count == len(expected_ticks)
    assert ns["TruthGuidanceUpdate"].call_count == len(expected_ticks)
    assert ns["referee"].progress.call_count == len(expected_ticks)
    assert ns["referee"].observe_truth.call_count == 200
    assert ns["truth_hold_referee"].observe_truth.call_count == 200
    assert [call.kwargs["tick"] for call in recorder.observe.call_args_list] == list(range(200))
    assert ns["fpv"].poll.call_count == 200
    for tick, call in enumerate(recorder.observe.call_args_list):
        assert call.kwargs["frame"] is samples[tick].frame
        assert call.kwargs["frame_time"] == samples[tick].frame_time
        assert call.kwargs["frame_fresh"] is samples[tick].fresh
        assert call.kwargs["barometer_fresh"] is (tick % ns["_barometer_tick_interval"] == 0)
        assert call.kwargs["magnetometer_fresh"] is (tick % ns["_magnetometer_tick_interval"] == 0)
    assert len(ctx.reads) == len(ns["bridge"][0].exchanges) == 200
    assert sum(name == "drone.motor_command" for name, _ in ctx.writes) == 200
    assert ns["guidance_cadence"].forced_updates == 0
    assert all(
        call.args[1] == i * ns["config"].dt
        for i, call in enumerate(ns["truth_hold_referee"].observe_truth.call_args_list)
    )
    for tick, (fdm_t, rc_t, channels) in enumerate(ns["bridge"][0].exchanges):
        assert fdm_t == rc_t == tick * ns["config"].dt
        assert channels[4] == (1000 if tick == 0 else 1800)  # One-tick RC latch.


@pytest.mark.parametrize(
    "fault",
    ["read", "gyro_shape", "gyro_nan", "pos_nan", "vel_nan", "quat_zero", "pos_short", "vel_short"],
)
def test_between_frame_invalid_input_faults_immediately_with_latest_raw_target(fault):
    ns, ctx = callback_namespace()
    source = ns["command_source"]
    step(ns, ctx, 0)  # Arms; next scheduled tick is 134.
    ctx.data["drone.world_pos"][4] = 0.025
    step(ns, ctx, 1)  # Last healthy physics sample, not a guidance sample.
    if fault == "read":
        ctx.fail_read = True
    elif fault == "gyro_shape":
        ctx.data["drone.gyro"] = np.zeros(2)
    elif fault == "gyro_nan":
        ctx.data["drone.gyro"][0] = np.nan
    elif fault == "pos_nan":
        ctx.data["drone.world_pos"][4] = np.nan
    elif fault == "vel_nan":
        ctx.data["drone.world_vel"][3] = np.nan
    elif fault == "quat_zero":
        ctx.data["drone.world_pos"][:4] = 0
    elif fault == "pos_short":
        ctx.data["drone.world_pos"] = np.zeros(6)
    else:
        ctx.data["drone.world_vel"] = np.zeros(5)
    step(ns, ctx, 2)
    assert source.phase is GuidancePhase.FAULT
    assert source.failure_reason == "invalid_truth"
    expected_x = 0.025
    assert source.controller._fault_target[0] == expected_x
    assert ns["guidance_cadence"].scheduled_updates == 1
    assert ns["guidance_cadence"].forced_updates == 1
    # Fault still uses existing armed hold, not a new disarm policy.
    assert ns["state"][0].selected_control.armed
    assert ns["bridge"][0].exchanges[2][2][4] == 1800
    ctx.fail_read = False
    ctx.data["drone.world_pos"] = np.array([0.0, 0.0, 0.0, 1.0, 0.025, 0.0, 0.1])
    ctx.data["drone.world_vel"] = np.zeros(6)
    ctx.data["drone.gyro"] = np.array([0.0, 0.0, 0.2])
    step(ns, ctx, 3)
    assert ns["guidance_cadence"].forced_updates == 2  # Full-rate fault feedback.
    assert ns["state"][0].selected_control.yaw == pytest.approx(0.02)


@pytest.mark.parametrize(
    "course_name,reason", [("single", "gate_crossing"), ("c1_straight", "uncommitted_crossing")]
)
def test_between_frame_crossing_fault_is_not_lost_to_out_and_back(course_name, reason):
    ns, ctx = callback_namespace(course_name=course_name)
    source = ns["command_source"]
    gate = ns["race_course"].gates[0]
    ctx.data["drone.world_pos"][4] = gate.center[0] - 0.01
    step(ns, ctx, 0)
    ctx.data["drone.world_pos"][4] = gate.center[0] + 0.01
    step(ns, ctx, 1)
    assert source.phase is GuidancePhase.FAULT
    assert source.failure_reason == reason
    assert source.phase_started_at == 1 / 8000
    ctx.data["drone.world_pos"][4] = gate.center[0] - 0.01
    step(ns, ctx, 2)
    assert source.phase is GuidancePhase.FAULT


@pytest.mark.parametrize("failure", ["timeout", "no_response", "nonfinite_motor"])
def test_bridge_failures_keep_legacy_policy_and_grade_every_tick(failure):
    ns, ctx = callback_namespace()
    step(ns, ctx, 0)
    b = ns["bridge"][0]
    if failure == "nonfinite_motor":
        b.motors[0] = np.nan
    else:
        setattr(b, failure, True)
    step(ns, ctx, 1)
    assert ns["command_source"].phase is not GuidancePhase.FAULT
    assert ns["guidance_cadence"].forced_updates == 0
    assert not ns["state"][0].motor_response
    assert ns["truth_hold_referee"].observe_truth.call_count == 2
    assert not ns["truth_hold_referee"].observe_truth.call_args.args[2].motor_response


def test_course_grading_and_existing_gap_threshold_stay_at_physics_rate():
    ns, ctx = callback_namespace(course_name="c1_straight")
    for tick in range(200):
        step(ns, ctx, tick)
    grader = ns["truth_course_referee"]
    assert grader.observe_truth.call_count == 200
    assert [call.args[0] for call in grader.observe_truth.call_args_list] == [
        i * ns["config"].dt for i in range(200)
    ]
    assert ns["command_source"].phase is not GuidancePhase.FAULT


def test_normal_termination_still_runs_on_held_tick(capsys):
    ns, ctx = callback_namespace(max_ticks=3)
    for tick in range(3):
        step(ns, ctx, tick)
    assert ctx.stopped and ns["bridge"][0].stopped
    assert ns["truth_hold_referee"].observe_truth.call_count == 3
    assert "scheduled=1 forced=0 updates=1 ticks=3" in capsys.readouterr().out


def test_first_callback_gate_pass_event_faults_hold_on_next_physics_tick():
    ns, ctx = callback_namespace()
    # Gate crossing on the first callback has no previous controller plane
    # sample yet. The referee's public event must still reject the hold next
    # physics tick, rather than waiting until the next scheduled update.
    ctx.data["drone.world_pos"] = np.array([0.0, 0.0, 0.0, 1.0, 10.01, 0.0, 1.8])
    step(ns, ctx, 0)
    assert ns["state"][0].last_gate_passed == 0
    step(ns, ctx, 1)
    assert ns["command_source"].phase is GuidancePhase.FAULT
    assert ns["command_source"].failure_reason == "gate_crossing"
    assert ns["guidance_cadence"].forced_updates == 1


def test_fault_command_is_exchanged_only_on_following_tick():
    ns, ctx = callback_namespace()
    step(ns, ctx, 0)
    previous_command = ns["state"][0].rc_latch.command_for_exchange()
    ctx.fail_read = True
    ctx.data["drone.gyro"][2] = 0.3
    # Failed batch reads leave gyro at the existing zero fallback.
    step(ns, ctx, 1)
    fault_command = ns["state"][0].rc_latch.command_for_exchange()
    assert ns["command_source"].phase is GuidancePhase.FAULT
    exchanged = ns["bridge"][0].exchanges[1][2]
    assert exchanged[:6] == tuple(previous_command.as_array())
    ctx.fail_read = False
    step(ns, ctx, 2)
    assert ns["bridge"][0].exchanges[2][2][:6] == tuple(fault_command.as_array())


def test_fault_on_scheduled_tick_counts_once_not_as_additional_forced_update():
    ns, ctx = callback_namespace()
    step(ns, ctx, 0)
    ctx.fail_read = True
    step(ns, ctx, 134)
    assert ns["command_source"].phase is GuidancePhase.FAULT
    assert ns["guidance_cadence"].scheduled_updates == 2
    assert ns["guidance_cadence"].forced_updates == 0


def test_committed_race_crossing_remains_allowed():
    ns, ctx = callback_namespace(course_name="c1_straight")
    step(ns, ctx, 0)
    source = ns["command_source"]
    source.controller.phase = GuidancePhase.COMMIT
    gate = ns["race_course"].gates[0]
    ctx.data["drone.world_pos"][4] = gate.center[0] + 0.01
    step(ns, ctx, 1)
    assert source.phase is GuidancePhase.COMMIT
    assert ns["guidance_cadence"].forced_updates == 0


def test_skipped_deadlines_use_actual_simulation_time_not_fictitious_physics_dt():
    ns, ctx = callback_namespace()
    source = ns["command_source"]
    source.update = Mock(wraps=source.update)
    step(ns, ctx, 0)
    step(ns, ctx, 1000)
    update = source.update.call_args.args[0]
    assert update.tick == 1000 and update.sim_time == 1000 * ns["config"].dt
    assert ns["guidance_cadence"].scheduled_updates == 2
    step(ns, ctx, 1001)
    assert source.update.call_count == 2


def test_scripted_path_ignores_truth_scheduler_when_knob_unset():
    ns, ctx = callback_namespace(rate=None)
    ns["guidance_mode"] = GuidanceMode.SCRIPTED
    ns["command_source"] = controls.ScriptedGuidance()
    ns["truth_hold_referee"] = None
    ns["guidance_cadence"].due = Mock(side_effect=AssertionError("truth schedule only"))
    ns["GuidanceUpdate"] = Mock(wraps=controls.GuidanceUpdate)
    for tick in range(200):
        step(ns, ctx, tick)
    assert ns["GuidanceUpdate"].call_count == 200
    assert ns["referee"].progress.call_count == 200


@pytest.mark.parametrize("rate", [None, 8000])
def test_fullrate_callback_is_legacy_equivalent(rate):
    # Saved verified HEAD source is evidence, not required to run repository tests.
    # Recreate only the legacy guidance block from the current callback AST by
    # ungating the branch and removing the new scheduler/guard preamble.
    tree = ast.parse(MAIN.read_text())
    callback = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sitl_post_step"
    )
    body = callback.body
    start = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "scheduled" for t in n.targets)
    )
    branch = next(
        i
        for i in range(start, len(body))
        if isinstance(body[i], ast.If)
        and isinstance(body[i].test, ast.BoolOp)
        and ast.unparse(body[i].test) == "scheduled or forced"
    )
    callback.body = body[:start] + body[branch].body[1:] + body[branch + 1 :]
    legacy, oldctx = callback_namespace(rate, source_text=ast.unparse(tree))
    current, ctx = callback_namespace(rate)
    current["command_source"].fast_update_required = Mock(
        side_effect=AssertionError("fullrate guard must be bypassed")
    )
    for tick in range(1000):
        pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0001 * tick, 0.0, 0.1 + tick * 0.00001])
        for context in (ctx, oldctx):
            context.data["drone.world_pos"] = pos.copy()
        step(current, ctx, tick)
        step(legacy, oldctx, tick)
        assert current["state"][0].selected_control == legacy["state"][0].selected_control
    assert current["bridge"][0].exchanges == legacy["bridge"][0].exchanges
    assert len(ctx.writes) == len(oldctx.writes)
    for (name, value), (oldname, oldvalue) in zip(ctx.writes, oldctx.writes):
        assert name == oldname
        np.testing.assert_array_equal(value, oldvalue)
