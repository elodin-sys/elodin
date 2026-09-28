"""Pure tests for the opt-in Package C live referee qualification."""

import pytest
from referee import GatePassEvent, RaceResult
from referee_audit import (
    AUDIT_DURATION_S,
    AUDIT_INITIAL_POSITION,
    AUDIT_INITIAL_VELOCITY,
    RefereeAuditCollector,
    RefereeAuditEvidence,
    audit_initial_condition,
    evaluate_referee_audit,
    referee_audit_from_env,
)


def _passing_evidence(**changes) -> RefereeAuditEvidence:
    values = {
        "events": (GatePassEvent(gate_index=0, pass_time=1.08525),),
        "telemetry_last_gate_passed": 0,
        "telemetry_pass_times": (1.08525, -1.0, -1.0),
        "telemetry_verified_on_later_tick": True,
        "race_result": RaceResult("single", 1, 1, 1.08525, (1.08525,)),
        "first_position": (5.001, 0.0, 4.9),
        "crossing_previous_position": (9.9999, 0.0, 1.8002),
        "crossing_current_position": (10.0001, 0.0, 1.7998),
        "final_position": (11.7, 0.0, 0.0),
        "race_line_count": 1,
        "audit_line_count": 1,
    }
    values.update(changes)
    return RefereeAuditEvidence(**values)


def test_referee_audit_is_disabled_by_default_and_preserves_defaults() -> None:
    parsed = referee_audit_from_env(
        {}, course_name="none", guidance_mode="scripted", manual_audit_requested=False
    )
    position = [1.0, 2.0, 3.0]
    velocity = [4.0, 5.0, 6.0]

    assert parsed.enabled is False
    assert audit_initial_condition(False, position, velocity, 15.0) == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        15.0,
    )
    assert position == [1.0, 2.0, 3.0]
    assert velocity == [4.0, 5.0, 6.0]


def test_referee_audit_fixture_is_controlled_airborne_positive_x_motion() -> None:
    parsed = referee_audit_from_env(
        {"RACE_REFEREE_AUDIT": "1"},
        course_name="single",
        guidance_mode="scripted",
        manual_audit_requested=False,
    )

    assert parsed.enabled is True
    assert audit_initial_condition(True, (0, 0, 0.1), (0, 0, 0), 15.0) == (
        AUDIT_INITIAL_POSITION,
        AUDIT_INITIAL_VELOCITY,
        AUDIT_DURATION_S,
    )
    assert AUDIT_INITIAL_POSITION[0] < 10.0
    assert AUDIT_INITIAL_POSITION[2] > 0.0
    assert AUDIT_INITIAL_VELOCITY[0] > 0.0
    assert AUDIT_INITIAL_VELOCITY[1:] == (0.0, 0.0)
    assert AUDIT_DURATION_S < 5.0  # Scripted guidance remains in its safe boot phase.


@pytest.mark.parametrize(
    ("env", "course", "guidance", "manual_audit", "message"),
    [
        ({"RACE_REFEREE_AUDIT": "yes"}, "single", "scripted", False, "must be '0' or '1'"),
        ({"RACE_REFEREE_AUDIT": "1"}, "none", "scripted", False, "requires RACE_COURSE=single"),
        ({"RACE_REFEREE_AUDIT": "1"}, "single", "manual", False, "requires RACE_GUIDANCE=scripted"),
        ({"RACE_REFEREE_AUDIT": "1"}, "single", "scripted", True, "incompatible"),
    ],
)
def test_referee_audit_rejects_invalid_combinations(
    env, course, guidance, manual_audit, message
) -> None:
    with pytest.raises(ValueError, match=message):
        referee_audit_from_env(
            env,
            course_name=course,
            guidance_mode=guidance,
            manual_audit_requested=manual_audit,
        )


def test_referee_audit_collector_preserves_tick_order_and_evidence() -> None:
    collector = RefereeAuditCollector(AUDIT_INITIAL_POSITION)
    first = (5.001, 0.0, 4.9)
    before = (9.9999, 0.0, 1.8002)
    after = (10.0001, 0.0, 1.7998)
    final = (11.7, 0.0, 0.0)
    event = GatePassEvent(gate_index=0, pass_time=1.08525)

    assert not collector.telemetry_due(1)
    collector.record_truth_read(first)
    collector.record_scoring_tick(first, tick=1, event=None)
    collector.record_truth_read(before)
    collector.record_scoring_tick(before, tick=2, event=None)
    collector.record_truth_read(after)
    collector.record_scoring_tick(after, tick=3, event=event)
    assert not collector.telemetry_due(3)
    assert collector.telemetry_due(4)
    collector.record_telemetry(0, (1.08525, -1.0, -1.0))
    assert not collector.telemetry_due(5)
    collector.record_truth_read(final)
    collector.record_scoring_tick(final, tick=4, event=None)

    expected = _passing_evidence()
    assert (
        collector.evidence(expected.race_result, race_line_count=1, audit_line_count=1) == expected
    )
    assert evaluate_referee_audit(collector.evidence(expected.race_result, 1, 1)).passed


def test_referee_audit_evaluator_accepts_complete_live_evidence() -> None:
    result = evaluate_referee_audit(_passing_evidence())

    assert result.passed is True
    assert result.exit_code == 0
    assert result.failed_checks == ()
    assert result.format() == (
        "[C-REFEREE-AUDIT] gate=0 passes=1 telemetry=true result=COMPLETE "
        "pass_time=1.085250 status=PASS"
    )


@pytest.mark.parametrize(
    ("changes", "failed_check"),
    [
        ({"events": ()}, "event_count"),
        ({"telemetry_last_gate_passed": -1}, "telemetry_last_gate"),
        ({"telemetry_pass_times": (1.08525, 2.0, -1.0)}, "telemetry_unset_slots"),
        ({"telemetry_verified_on_later_tick": False}, "telemetry_later_tick"),
        ({"race_result": RaceResult("single", 0, 1, None, ())}, "race_result"),
        ({"race_line_count": 2}, "race_line_count"),
        ({"audit_line_count": 0}, "audit_line_count"),
        ({"final_position": (10.1, 0.0, 0.0)}, "visible_departure"),
    ],
)
def test_referee_audit_evaluator_rejects_representative_broken_criteria(
    changes, failed_check
) -> None:
    result = evaluate_referee_audit(_passing_evidence(**changes))

    assert result.passed is False
    assert result.exit_code == 1
    assert failed_check in result.failed_checks
    assert result.format().endswith("status=FAIL")
