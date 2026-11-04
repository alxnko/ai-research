"""Timestamp helpers used by the scheduler service."""
from datetime import datetime, timedelta, timezone


def format_offset(minutes):
    """Render an offset given in minutes as +HH:MM / -HH:MM, or Z for zero."""
    if minutes == 0:
        return "Z"
    sign = "+" if minutes > 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"{sign}{hours:02d}:{mins:02d}"


def to_local_iso(epoch_seconds, offset_minutes):
    """Format an epoch timestamp in the given UTC offset."""
    tz = timezone(timedelta(minutes=offset_minutes))
    dt = datetime.fromtimestamp(epoch_seconds, tz=tz)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + format_offset(offset_minutes)
