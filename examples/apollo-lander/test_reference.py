"""Regression tests for the shared Apollo 11 descent reference profile.

`reference.py` reconstructs the truth profile that drives both the in-sim truth
ghost and the external guidance controller. It carries two constants that its
own comments document as having to match code elsewhere -- the footpad contact
altitude (`sim.FOOTPAD_HEIGHT_M`) and the terminal descent rate
(`MIN_DESCENT_RATE_MPS` in `controller/`) -- but nothing enforced either
coupling, so a change on one side could silently desynchronise the truth
display from the physics or from the guidance law.
"""

from __future__ import annotations

import re
from pathlib import Path

import reference


def test_vendored_data_matches_derived_measurements():
    """reference.py's own cross-check of the derived SI values against the raw CSVs."""
    report = reference.sanity_check()
    assert report["ok"], report


def test_profile_flies_the_descent_to_footpad_contact():
    ref = reference.build_reference()

    # Picks the mission up at landing-radar lock-on: ~38,700 ft, still carrying
    # the residual orbital velocity, ~107 km uprange of the target.
    assert 11_500.0 < ref.altitude_m[0] < 12_500.0
    assert 600.0 < ref.horizontal_speed_mps[0] < 1_100.0
    assert ref.downrange_m[0] < -100_000.0

    # Ends stopped over the targeted site, at footpad contact height.
    assert ref.altitude_m[-1] == reference.FOOTPAD_CONTACT_ALT_M
    assert ref.horizontal_speed_mps[-1] == 0.0
    assert ref.downrange_m[-1] == 0.0

    # The vehicle is descending across the whole profile. Altitude itself is
    # not strictly monotonic -- it is digitized from the mission-report chart
    # and smoothed, so a handful of samples tick up by a few metres.
    assert max(ref.descent_rate_mps) < 0.0


def test_footpad_contact_altitude_matches_sim():
    """FOOTPAD_CONTACT_ALT_M is documented as needing to equal sim.FOOTPAD_HEIGHT_M."""
    # Imported here rather than at module scope so the checks above stay
    # runnable without JAX, matching reference.py's standard-library-only design.
    import sim

    assert reference.FOOTPAD_CONTACT_ALT_M == sim.FOOTPAD_HEIGHT_M


def test_terminal_descent_rate_matches_controller():
    """TERMINAL_CONTACT_RATE_MPS is documented as matching the controller's floor."""
    source = Path(__file__).with_name("controller") / "src" / "main.rs"
    match = re.search(
        r"const\s+MIN_DESCENT_RATE_MPS\s*:\s*f64\s*=\s*([0-9_.]+)",
        source.read_text(),
    )
    assert match is not None, f"MIN_DESCENT_RATE_MPS not found in {source}"
    assert float(match.group(1).replace("_", "")) == reference.TERMINAL_CONTACT_RATE_MPS
