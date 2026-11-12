"""Time formatting shared by detection and the public result."""

from __future__ import annotations

import math
from typing import cast


def as_float(value: object) -> float:
    """Convert a detector or probe value that is already numeric."""
    return float(cast(float, value))


def as_int(value: object) -> int:
    """Convert a detector or probe value that is already an integer."""
    return int(cast(int, value))


def format_time(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(float(seconds) * 1000.0)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def round_seconds(seconds: float) -> float:
    """Round to milliseconds, nudging binary values that land just below a boundary."""
    return round(float(seconds) + 1e-9, 3)


def round_score(value: float | None) -> float | None:
    """Round a measurement the way the frozen JSON export did."""
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 4)
