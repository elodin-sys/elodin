"""Time conversion helpers for the Voyager example.

The tutorial previously hardcoded ``start_timestamp=252_452_400_000_000``.
That constant is 1977-12-31 21:40:00 UTC — 8400 seconds before
``1978-01-01T00:00:00Z``, which is the SPICE epoch the example actually
queries. Propagation uses ``spice.utc2et(START_UTC)`` and was never on that
offset. Only Elodin-DB sample wall-clock labels were wrong.

``utc_epoch_microseconds("1978-01-01T00:00:00")`` is 252_460_800_000_000.
Validation cases use the reconstructed-arc UTC string, so their DB timestamps
follow the same UTC conversion.
"""

from datetime import datetime, timedelta, timezone

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LEGACY_TUTORIAL_DB_EPOCH_US = 252_452_400_000_000
TUTORIAL_START_UTC = "1978-01-01T00:00:00"


def utc_epoch_microseconds(value: str) -> int:
    """Convert an ISO-8601 UTC timestamp to Unix epoch microseconds."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    elapsed = parsed.astimezone(timezone.utc) - UNIX_EPOCH
    return elapsed // timedelta(microseconds=1)
