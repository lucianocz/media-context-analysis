"""Errors raised by the public analysis API."""


class MediaContextAnalysisError(Exception):
    """Base class for failures while analyzing a media file."""


class InvalidOptionsError(MediaContextAnalysisError):
    """``AnalysisOptions`` contains a value the analyzer cannot use."""


class MediaNotFoundError(MediaContextAnalysisError):
    """The path does not exist or is not a file."""


class UnsupportedMediaError(MediaContextAnalysisError):
    """The file is readable but has no usable video stream."""


class CorruptMediaError(MediaContextAnalysisError):
    """The file exists but FFmpeg could not decode a consistent picture."""


class FFmpegNotFoundError(MediaContextAnalysisError):
    """FFmpeg or FFprobe is not installed or the configured path is missing."""


class FFmpegFailedError(MediaContextAnalysisError):
    """FFmpeg or FFprobe exited with an error."""


class AnalysisTimeoutError(MediaContextAnalysisError):
    """The analysis exceeded the configured time limit."""


class AnalysisCancelledError(MediaContextAnalysisError):
    """The caller cancelled the analysis."""
