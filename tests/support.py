"""Shared synthetic shots. No media files."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from media_context_analysis.params import DetectorParams
from media_context_analysis.shots import Shot


def hist(pattern: list[float]) -> NDArray[np.float64]:
    values = np.asarray(pattern, dtype=np.float64)
    return values / values.sum()


def thumb(pattern: list[float]) -> NDArray[np.float64]:
    values = np.asarray(pattern, dtype=np.float64)
    return values / np.linalg.norm(values)


def shot(
    start: float,
    end: float,
    histogram: NDArray[np.float64],
    thumbnail: NDArray[np.float64],
    luma: float = 0.4,
    change: float = 0.3,
    frame_start: int = -1,
    frame_end: int = -1,
) -> Shot:
    return Shot(start, end, histogram, thumbnail, luma, change, frame_start, frame_end)


def params() -> DetectorParams:
    return DetectorParams()
