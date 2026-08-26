"""Focused Voyager 1 Jupiter encounter diagnostic contract for issue #794."""

from datetime import datetime, timezone

ENCOUNTER_KERNEL = "vgr1_jup230.bsp"
ENCOUNTER_KERNEL_SHA256 = (
    "e1ea3f72f19b15508bc45979771a36a97d02f33056b76867d444304cb82205c9"
)
PROBE = "VOYAGER 1"
FRAME = "ECLIPJ2000"
OBSERVER = "SUN"
INITIALIZATION_UTC = "1979-02-21T00:00:00Z"
CHECKPOINT_UTCS = (
    "1979-02-21T00:00:00Z",
    "1979-02-24T00:00:00Z",
    "1979-02-28T00:00:00Z",
    "1979-03-04T00:00:00Z",
    "1979-03-05T00:00:00Z",
    "1979-03-06T00:00:00Z",
)
CHECKPOINT_NAMES = (
    "initialization",
    "feb_24",
    "feb_28",
    "mar_04",
    "closest_approach_day",
    "end",
)
CHECKPOINT_ROLES = (
    "anchor",
    "diagnostic",
    "diagnostic",
    "near_encounter",
    "near_encounter",
    "post_encounter",
)

# The PDS data-set envelope is 1979-02-05 through 1979-04-08.
PDS_COVERAGE_UTC = ("1979-02-05T12:00:00Z", "1979-04-08T12:00:00Z")

# Important limitation: this interval is not a clean no-thrust coast. A 1995 JPL
# reanalysis of the Voyager 1 Jupiter tracking data modeled impulsive thruster
# events inside this exact window, including one only 3 h 58 min after our
# initialization epoch. It also modeled piecewise attitude-control accelerations.
# These are not applied by the current gravity-only Chapter 1/2 comparison.
MODELED_THRUSTER_EVENTS_UTC = (
    "1979-02-21T03:58:00Z",
    "1979-03-01T23:00:00Z",
    "1979-03-03T20:00:00Z",
    "1979-03-04T00:00:00Z",
)
MODELED_ACCELERATION_STARTS_UTC = (
    "1979-02-21T18:00:00Z",
)

SOURCES = (
    "https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=VG1-J-SPICE-6-SPK-V2.0",
    "https://ntrs.nasa.gov/api/citations/19790009610/downloads/19790009610.pdf",
    "https://tda.jpl.nasa.gov/progress_report/42-121/121F.pdf",
)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def checkpoints() -> tuple[dict[str, int | str], ...]:
    """Return fixed checkpoint names, roles, epochs, and elapsed seconds."""
    start = parse_utc(INITIALIZATION_UTC)
    return tuple(
        {
            "name": name,
            "role": role,
            "utc": utc,
            "elapsed_seconds": int((parse_utc(utc) - start).total_seconds()),
        }
        for name, role, utc in zip(
            CHECKPOINT_NAMES, CHECKPOINT_ROLES, CHECKPOINT_UTCS, strict=True
        )
    )
