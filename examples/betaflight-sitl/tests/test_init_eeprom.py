"""Pure tests for EEPROM configuration and CLI verification."""

import pytest
from init_eeprom import (
    CONFIGURE_COMMANDS,
    EXPECTED_SETTINGS,
    VERIFY_QUERIES,
    verification_failures,
)


def _valid_outputs() -> dict[str, str]:
    outputs = {}
    for query, expected in EXPECTED_SETTINGS:
        outputs[query] = f"{outputs.get(query, '')}\n{expected}"
    # Betaflight reports disabled GPS under Available, and enabled features on
    # their own line. GPS-like names must not be confused with the exact token.
    outputs["feature"] = "Enabled: RX_SERIAL GPS_RESCUE\nAvailable: GPS\nUnavailable:"
    return outputs


def test_configuration_disables_gps_and_mag_trust_without_dropping_existing_settings() -> None:
    assert "feature -GPS" in CONFIGURE_COMMANDS
    assert "set trust_mag = OFF" in CONFIGURE_COMMANDS
    assert {
        "aux 0 0 0 1700 2100 0 0",
        "aux 1 1 1 1700 2100 0 0",
        "set gyro_hardware_lpf = NORMAL",
        "set pid_process_denom = 1",
        "set rc_smoothing = off",
    }.issubset(CONFIGURE_COMMANDS)


def test_verification_accepts_gps_absent_as_an_exact_enabled_feature() -> None:
    assert verification_failures(_valid_outputs()) == ()


def test_verification_requires_exact_stripped_response_lines() -> None:
    outputs = _valid_outputs()
    outputs["get trust_mag"] = "\r\n  trust_mag = OFF \r\n# "

    assert verification_failures(outputs) == ()


@pytest.mark.parametrize(
    ("query", "expected", "misleading_line"),
    [
        (
            "get gyro_hardware_lpf",
            "gyro_hardware_lpf = NORMAL",
            "backup gyro_hardware_lpf = NORMAL",
        ),
        ("get pid_process_denom", "pid_process_denom = 1", "pid_process_denom = 16"),
        ("get rc_smoothing", "rc_smoothing = OFF", "rc_smoothing = OFFISH"),
        ("get trust_mag", "trust_mag = OFF", "other_trust_mag = OFF"),
        (
            "aux",
            "aux 0 0 0 1700 2100 0 0",
            "# expected aux 0 0 0 1700 2100 0 0",
        ),
    ],
)
def test_verification_rejects_setting_substrings(
    query: str, expected: str, misleading_line: str
) -> None:
    outputs = _valid_outputs()
    outputs[query] = f"\r\n{misleading_line}\r\n"

    assert expected in verification_failures(outputs)


def test_verification_rejects_exact_gps_token_but_not_gps_substrings() -> None:
    outputs = _valid_outputs()
    outputs["feature"] = "Enabled: RX_SERIAL GPS GPS_RESCUE\nAvailable:"

    failures = verification_failures(outputs)

    assert "feature -GPS" in failures
    assert not any("GPS_RESCUE" == failure for failure in failures)


def test_verification_requires_trust_mag_off_and_all_existing_settings() -> None:
    outputs = _valid_outputs()
    outputs["get trust_mag"] = "trust_mag = ON"
    outputs["get rc_smoothing"] = "rc_smoothing = ON"

    failures = verification_failures(outputs)

    assert "trust_mag = OFF" in failures
    assert "rc_smoothing = OFF" in failures


def test_verification_fails_closed_when_feature_output_is_missing_or_malformed() -> None:
    outputs = _valid_outputs()
    del outputs["feature"]
    assert "feature -GPS" in verification_failures(outputs)

    outputs["feature"] = "Available: GPS\nUnavailable:"
    assert "feature output with Enabled: line" in verification_failures(outputs)


def test_verification_checks_each_existing_aux_setting() -> None:
    outputs = _valid_outputs()
    outputs["aux"] = "aux 0 0 0 1700 2100 0 0"

    failures = verification_failures(outputs)

    assert "aux 1 1 1 1700 2100 0 0" in failures
    assert "aux 0 0 0 1700 2100 0 0" not in failures


def test_queries_cover_every_expected_setting_once() -> None:
    queries = {query for query, _ in EXPECTED_SETTINGS}
    assert queries == {
        "get gyro_hardware_lpf",
        "get pid_process_denom",
        "get rc_smoothing",
        "get trust_mag",
        "aux",
    }
    assert len(CONFIGURE_COMMANDS) == len(set(CONFIGURE_COMMANDS))
    assert VERIFY_QUERIES == tuple(dict.fromkeys(query for query, _ in EXPECTED_SETTINGS)) + (
        "feature",
    )
    assert len(VERIFY_QUERIES) == len(set(VERIFY_QUERIES))
