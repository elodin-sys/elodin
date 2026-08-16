"""Time conversion helpers for the Voyager example."""

from datetime import datetime, timedelta, timezone


UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def utc_epoch_microseconds(value: str) -> int:
    """Convert an ISO-8601 UTC timestamp to Unix epoch microseconds."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    elapsed = parsed.astimezone(timezone.utc) - UNIX_EPOCH
    return elapsed // timedelta(microseconds=1)
