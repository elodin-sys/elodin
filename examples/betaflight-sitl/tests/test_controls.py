"""Pure tests for the Package D command and manual-input boundary."""

import numpy as np
import pytest

from controls import (
    DelayedRcCommand,
    GatePose,
    GuidanceMode,
    GuidanceUpdate,
    ManualGuidance,
    RcCommand,
    ScriptedGuidance,
    SemanticControl,
    TruthGuidanceUpdate,
    TruthState,
    guidance_mode_from_env,
    semantic_to_rc,
)
from comms import MAX_RC_CHANNELS
from fpv_camera import FRAME_BYTES, PERIOD_US, FpvCamera


def test_default_and_explicit_guidance_selection() -> None:
    assert guidance_mode_from_env({}).value == "scripted"
    assert guidance_mode_from_env({"RACE_GUIDANCE": "manual"}).value == "manual"


@pytest.mark.parametrize("value", ["", "acro", "MANUALS"])
def test_unknown_guidance_fails_at_startup(value: str) -> None:
    with pytest.raises(ValueError, match="unknown RACE_GUIDANCE"):
        guidance_mode_from_env({"RACE_GUIDANCE": value})


def test_truth_guidance_is_enabled_and_vision_remains_reserved() -> None:
    assert guidance_mode_from_env({"RACE_GUIDANCE": "truth"}) is GuidanceMode.TRUTH
    with pytest.raises(ValueError, match="vision.*reserved.*not implemented"):
        guidance_mode_from_env({"RACE_GUIDANCE": "vision"})


def test_semantic_input_maps_to_aetr_aux_channel_order() -> None:
    command = semantic_to_rc(
        SemanticControl(
            roll=0.5,
            pitch=-0.5,
            throttle=0.25,
            yaw=1.0,
            armed=True,
            angle_mode=True,
        )
    )

    assert command.as_array().tolist() == [1750, 1250, 1250, 1000, 1800, 1800]


def test_conversion_clamps_axes_throttle_and_all_filled_channels() -> None:
    command = semantic_to_rc(
        SemanticControl(
            roll=2.0,
            pitch=-2.0,
            throttle=4.0,
            yaw=float("nan"),
            armed=False,
            angle_mode=False,
        )
    )
    channels = np.zeros(MAX_RC_CHANNELS, dtype=np.uint16)

    assert command.fill_channels(channels) is channels
    assert channels[:6].tolist() == [2000, 1000, 2000, 1500, 1000, 1000]
    assert channels[6:].tolist() == [1500] * (MAX_RC_CHANNELS - 6)


def test_rc_command_itself_cannot_carry_out_of_range_pwm() -> None:
    command = RcCommand(roll=-1, pitch=9_000, throttle=float("nan"), yaw=1500)

    assert command.as_array()[:4].tolist() == [1000, 2000, 1000, 1500]


def test_throttle_alone_never_arms() -> None:
    command = semantic_to_rc(SemanticControl(throttle=1.0, armed=False))

    assert command.throttle == 2000
    assert command.arm == 1000


def test_scripted_guidance_preserves_package_a_timing_and_values() -> None:
    source = ScriptedGuidance()

    expected = [
        (0.0, [1500, 1500, 1000, 1500, 1000, 1000], "boot"),
        (5.0, [1500, 1500, 1000, 1500, 1800, 1000], "arm"),
        (7.0, [1500, 1500, 1400, 1500, 1800, 1000], "throttle"),
        (17.0, [1500, 1500, 1000, 1500, 1000, 1000], "disarm"),
    ]
    for sim_time, channels, phase in expected:
        update = _guidance_update(sim_time)
        assert semantic_to_rc(source.update(update)).as_array().tolist() == channels
        assert source.phase == phase


def test_manual_input_maps_all_axes_and_switches_deterministically() -> None:
    source = ManualGuidance()
    raw = np.array([0.2, -0.4, 0.6, -0.8, 1.0, 1.0, 42.0])

    control = _manual_update(source, raw, now=10.0)
    command = semantic_to_rc(control)

    assert source.fresh is True
    assert command.as_array().tolist() == [1600, 1300, 1600, 1900, 1800, 1800]


def test_manual_input_disconnect_is_safe() -> None:
    source = ManualGuidance()

    assert _manual_update(source, None, now=0.0) == SemanticControl.safe()
    assert source.fresh is False
    assert semantic_to_rc(_manual_update(source, None, now=1.0)).as_array().tolist() == [
        1500,
        1500,
        1000,
        1500,
        1000,
        1800,
    ]


def test_manual_input_stale_heartbeat_disarms_and_centers() -> None:
    source = ManualGuidance(stale_after_s=0.25)
    armed = np.array([1.0, -1.0, 0.8, 0.5, 1.0, 1.0, 7.0])

    assert _manual_update(source, armed, now=2.0).armed is True
    assert _manual_update(source, armed, now=2.25).armed is True
    safe = _manual_update(source, armed, now=2.251)

    assert safe == SemanticControl.safe()
    assert source.fresh is False
    assert "stale" in source.reason


def test_new_heartbeat_recovers_after_stale_input() -> None:
    source = ManualGuidance(stale_after_s=0.1)
    first = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    second = first.copy()
    second[-1] = 2.0

    _manual_update(source, first, now=1.0)
    assert _manual_update(source, first, now=1.2) == SemanticControl.safe()
    assert _manual_update(source, second, now=1.21).angle_mode is True
    assert source.fresh is True


def test_disconnect_allows_a_restarted_controller_heartbeat() -> None:
    source = ManualGuidance()
    first = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    assert _manual_update(source, first, now=1.0).roll == 0.2
    assert _manual_update(source, None, now=1.1) == SemanticControl.safe()
    assert _manual_update(source, first, now=1.2).roll == 0.2
    assert source.fresh is True


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros(6),
        np.zeros(7),  # Heartbeat zero is the uninitialized DB value, not a controller sample.
        np.zeros(8),
        np.array([0.0, 0.0, np.nan, 0.0, 1.0, 1.0, 1.0]),
    ],
)
def test_malformed_manual_input_is_safe(raw: np.ndarray) -> None:
    source = ManualGuidance()

    assert _manual_update(source, raw, now=0.0) == SemanticControl.safe()
    assert source.fresh is False


def test_command_selected_on_tick_n_is_exchanged_on_tick_n_plus_one() -> None:
    safe = semantic_to_rc(SemanticControl.safe())
    requested = semantic_to_rc(SemanticControl(roll=0.5, throttle=0.2, armed=True))
    latch = DelayedRcCommand(safe)

    sent_on_tick_n = latch.command_for_exchange()
    latch.retain_for_next_tick(requested)
    sent_on_tick_n_plus_one = latch.command_for_exchange()

    assert sent_on_tick_n == safe
    assert sent_on_tick_n_plus_one == requested


def test_truth_update_copies_and_freezes_truth_and_gate_values() -> None:
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([4.0, 5.0, 6.0])
    orientation = np.array([0.0, 0.0, 0.0, 2.0])
    center = np.array([10.0, 0.0, 1.8])
    truth = TruthState(position, velocity, orientation)
    gate = GatePose(0, center, 0.0, 2.5)
    update = TruthGuidanceUpdate(
        sim_time=0.0,
        tick=0,
        gyro=np.zeros(3),
        accel=np.zeros(3),
        truth_state=truth,
        gate_poses=[gate],
    )

    position[:] = -99.0
    velocity[:] = -99.0
    orientation[:] = -99.0
    center[:] = -99.0

    assert update.truth_state.position_enu == (1.0, 2.0, 3.0)
    assert update.truth_state.velocity_enu == (4.0, 5.0, 6.0)
    assert update.truth_state.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert update.gate_poses == (gate,)
    assert update.gate_poses[0].center_enu == (10.0, 0.0, 1.8)
    assert update.gate_poses[0].normal_enu == (1.0, 0.0, 0.0)
    for name in ("exchanged_armed", "exchanged_angle_mode", "lockstep_response", "motor_response"):
        assert not hasattr(update, name)
    with pytest.raises((AttributeError, TypeError)):
        update.gate_poses[0].center_enu[0] = 2.0


def test_truth_and_base_updates_inherit_b_fpv_frames_without_truth_leak() -> None:
    camera = FpvCamera()
    pixels = np.full(FRAME_BYTES, 20, dtype=np.uint8)

    def read_msg_at(_name: str, _requested: int):
        return (1, pixels)

    first = camera.poll(read_msg_at, 40_000, 0.04)
    second = camera.poll(read_msg_at, 40_000 + PERIOD_US, 0.04 + PERIOD_US / 1e6)
    assert first.fresh and not second.fresh
    assert first.requested_us == 7_000
    assert second.requested_us == 7_000 + PERIOD_US

    shared = dict(sim_time=0.04, tick=320, gyro=np.zeros(3), accel=np.zeros(3))
    truth = TruthState((0.0, 0.0, 0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    gate = GatePose(0, (10.0, 0.0, 1.8), 0.0, 2.5)
    truth_update = TruthGuidanceUpdate(
        **shared,
        truth_state=truth,
        gate_poses=(gate,),
        frame=first.frame,
        frame_sample_time=float(first.requested_us),
        frame_fresh=first.fresh,
    )
    # Camera stays observational even in truth mode. The base/vision update has
    # the same camera fields but no way to access the hidden truth or gate pose.
    base_update = GuidanceUpdate(
        **shared,
        frame=second.frame,
        frame_sample_time=float(second.requested_us),
        frame_fresh=second.fresh,
    )
    pixels[:] = 100
    first.frame[:] = 99
    assert truth_update.frame.shape == (360, 640, 4)
    assert truth_update.frame[0, 0, 0] == 20
    assert truth_update.frame_sample_time == 7_000.0
    assert truth_update.frame_fresh
    assert base_update.frame_sample_time == float(second.requested_us)
    assert not base_update.frame_fresh
    assert not hasattr(base_update, "truth_state")
    assert not hasattr(base_update, "gate_poses")
    with pytest.raises(ValueError):
        truth_update.frame[0, 0, 0] = 0


def test_truth_update_without_camera_retains_none_and_no_freshness() -> None:
    update = TruthGuidanceUpdate(
        sim_time=0.0,
        tick=0,
        gyro=np.zeros(3),
        accel=np.zeros(3),
        truth_state=None,
        gate_poses=(),
    )
    assert update.frame is None
    assert update.frame_sample_time is None
    assert not update.frame_fresh


def test_truth_and_gate_values_reject_invalid_shapes_and_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="position_enu"):
        TruthState((0.0, float("nan"), 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="non-zero quaternion"):
        TruthState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="inner_size_m"):
        GatePose(0, (0.0, 0.0, 0.0), 0.0, 0.0)


def test_guidance_update_owns_immutable_sensor_copies() -> None:
    gyro = np.array([1.0, 2.0, 3.0])
    accel = np.array([4.0, 5.0, 6.0])
    update = GuidanceUpdate(sim_time=1.0, tick=8_000, gyro=gyro, accel=accel)

    gyro[0] = 99.0
    accel[0] = 99.0

    assert update.gyro.tolist() == [1.0, 2.0, 3.0]
    assert update.accel.tolist() == [4.0, 5.0, 6.0]
    assert not hasattr(update, "truth_state")
    assert not hasattr(update, "gate_poses")
    with pytest.raises(ValueError):
        update.gyro[0] = 0.0


def _manual_update(source: ManualGuidance, raw: np.ndarray | None, now: float) -> SemanticControl:
    source.accept_input(raw, now)
    return source.update(_guidance_update(now))


def _guidance_update(sim_time: float) -> GuidanceUpdate:
    return GuidanceUpdate(
        sim_time=sim_time,
        tick=round(sim_time * 8_000),
        gyro=np.zeros(3),
        accel=np.zeros(3),
    )
