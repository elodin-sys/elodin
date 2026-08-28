"""Focused Voyager 1 Jupiter encounter diagnostic contract for issue #794."""

from datetime import datetime, timezone

ENCOUNTER_KERNEL = "vgr1_jup230.bsp"
ENCOUNTER_KERNEL_SHA256 = (
    "e1ea3f72f19b15508bc45979771a36a97d02f33056b76867d444304cb82205c9"
)
PROBE = "VOYAGER 1"
FRAME = "ECLIPJ2000"
OBSERVER = "SUN"
INITIALIZATION_UTC = "1979-02-22T00:00:00Z"
CHECKPOINT_UTCS = (
    "1979-02-22T00:00:00Z",
    "1979-02-24T00:00:00Z",
    "1979-02-26T00:00:00Z",
    "1979-02-28T00:00:00Z",
)
CHECKPOINT_NAMES = (
    "initialization",
    "feb_24",
    "feb_26",
    "end",
)
CHECKPOINT_ROLES = (
    "anchor",
    "diagnostic",
    "diagnostic",
    "diagnostic",
)

# The PDS data-set envelope is 1979-02-05 through 1979-04-08.
PDS_COVERAGE_UTC = ("1979-02-05T12:00:00Z", "1979-04-08T12:00:00Z")

# Folkner & Haw (1995), Table 1. The selected Feb 22-28 arc is after the
# Feb 21 03:58 modeled impulse and before the Mar 1 23:00 modeled impulse.
# Their reanalysis also estimated small attitude-control accelerations; those
# remain outside the current gravity-only Chapter 1/2 model.
KNOWN_IMPULSIVE_MANEUVER_EVENTS_UTC = (
    "1979-02-04T00:00:00Z",
    "1979-02-05T12:00:00Z",
    "1979-02-09T04:02:00Z",
    "1979-02-17T00:00:00Z",
    "1979-02-18T18:00:00Z",
    "1979-02-19T00:00:00Z",
    "1979-02-21T03:58:00Z",
    "1979-03-01T23:00:00Z",
    "1979-03-03T20:00:00Z",
    "1979-03-04T00:00:00Z",
)
MODELED_ACCELERATION_STARTS_UTC = (
    "1979-02-01T00:00:00Z",
    "1979-02-04T08:30:00Z",
    "1979-02-05T12:00:00Z",
    "1979-02-09T04:00:00Z",
    "1979-02-11T02:00:00Z",
    "1979-02-15T00:00:00Z",
    "1979-02-17T15:00:00Z",
    "1979-02-19T05:00:00Z",
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
