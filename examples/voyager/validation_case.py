"""Frozen Voyager 1 Jupiter encounter validation contract for issue #794."""

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

# The PDS encounter SPK covers 1979-02-05 through 1979-04-08. The NASA
# encounter timeline lists the last pre-encounter trajectory correction on
# Feb 20; the selected Feb 21-Mar 6 window contains no later listed TCM.
PDS_COVERAGE_UTC = ("1979-02-05T12:00:00Z", "1979-04-08T12:00:00Z")
SOURCES = (
    "https://ntrs.nasa.gov/api/citations/19790009614/downloads/19790009614.pdf",
    "https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=VG1-J-SPICE-6-SPK-V2.0",
)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def checkpoints() -> tuple[dict[str, int | str], ...]:
    """Return the fixed checkpoint names, epochs, and elapsed seconds."""
    start = parse_utc(INITIALIZATION_UTC)
    return tuple(
        {
            "name": name,
            "utc": utc,
            "elapsed_seconds": int((parse_utc(utc) - start).total_seconds()),
        }
        for name, utc in zip(CHECKPOINT_NAMES, CHECKPOINT_UTCS, strict=True)
    )
