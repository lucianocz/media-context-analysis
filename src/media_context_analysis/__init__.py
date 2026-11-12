"""Timestamped visual context events for a local video file.

The public entry point is :func:`analyze`. It returns timestamps and evidence.
It does not extract images, read subtitles, or open a translation segment.
"""

from media_context_analysis._version import SCHEMA_VERSION, __version__
from media_context_analysis.analyze import analyze
from media_context_analysis.errors import (
    AnalysisCancelledError,
    AnalysisTimeoutError,
    CorruptMediaError,
    FFmpegFailedError,
    FFmpegNotFoundError,
    InvalidOptionsError,
    MediaContextAnalysisError,
    MediaNotFoundError,
    UnsupportedMediaError,
)
from media_context_analysis.models import (
    AnalysisResult,
    Continuity,
    Diagnostics,
    EventGroup,
    EventScores,
    MediaMetadata,
    Provenance,
    Representatives,
    VisualEvent,
)
from media_context_analysis.options import AnalysisOptions

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisCancelledError",
    "AnalysisOptions",
    "AnalysisResult",
    "AnalysisTimeoutError",
    "Continuity",
    "CorruptMediaError",
    "Diagnostics",
    "EventGroup",
    "EventScores",
    "FFmpegFailedError",
    "FFmpegNotFoundError",
    "InvalidOptionsError",
    "MediaContextAnalysisError",
    "MediaMetadata",
    "MediaNotFoundError",
    "Provenance",
    "Representatives",
    "UnsupportedMediaError",
    "VisualEvent",
    "__version__",
    "analyze",
]
