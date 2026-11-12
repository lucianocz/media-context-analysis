"""Caller settings that do not change the frozen detection thresholds."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from media_context_analysis.errors import InvalidOptionsError


@dataclass(kw_only=True)
class AnalysisOptions:
    """Options for :func:`media_context_analysis.analyze`.

    ``video_threads`` limits FFmpeg video decode. Audio decode stays on one
    thread. Detection thresholds are not configurable: they stay at the
    validated defaults. ``video_stream_index`` selects an FFprobe stream index;
    ``None`` selects the largest valid video, breaking ties by the lowest index.
    """

    ffmpeg: str | None = None
    ffprobe: str | None = None
    timeout_s: float | None = 1800.0
    video_threads: int = 2
    video_stream_index: int | None = None
    include_diagnostics: bool = False
    cancel_event: threading.Event | None = None

    def validate(self) -> None:
        if self.timeout_s is not None and (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise InvalidOptionsError("timeout_s must be a finite number of seconds greater than 0")
        if (
            isinstance(self.video_threads, bool)
            or not isinstance(self.video_threads, int)
            or not 1 <= self.video_threads <= 16
        ):
            raise InvalidOptionsError("video_threads must be an integer from 1 to 16")
        if self.video_stream_index is not None and (
            isinstance(self.video_stream_index, bool)
            or not isinstance(self.video_stream_index, int)
            or self.video_stream_index < 0
        ):
            raise InvalidOptionsError("video_stream_index must be a nonnegative integer or None")
