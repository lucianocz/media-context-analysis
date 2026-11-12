"""Run shot detection and event selection on arrays that are already sampled."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from media_context_analysis.events import detect_events
from media_context_analysis.params import DetectorParams
from media_context_analysis.shots import Shot, detect_shots, estimate_place_threshold
from media_context_analysis.signal import acoustic_change_scores, frame_change_scores


@dataclass
class DetectorRun:
    """In-memory detector output. ``detection`` matches the frozen event document."""

    detection: dict[str, object]
    shots: list[Shot]
    frame_times: NDArray[np.float64]
    frame_scores: NDArray[np.float64]
    place_threshold: float
    place_stats: dict[str, object]
    shot_stats: dict[str, object]
    params: DetectorParams


def run_detector(
    times: NDArray[np.floating],
    hists: NDArray[np.floating],
    thumbs: NDArray[np.floating],
    luma: NDArray[np.floating],
    audio_times: NDArray[np.floating] | None = None,
    audio_scores: NDArray[np.floating] | None = None,
    audio_features: NDArray[np.floating] | None = None,
    params: DetectorParams | None = None,
) -> DetectorRun:
    """Detect events without reading subtitles or writing images.

    Pass either ``audio_scores`` or ``audio_features``. When both are set,
    ``audio_scores`` is used so a cached score vector stays bit-comparable.
    """
    settings = params or DetectorParams()
    scores = audio_scores
    if scores is None and audio_features is not None and len(audio_features):
        scores = acoustic_change_scores(
            audio_features, settings.audio_hop_s, settings.audio_change_window_s
        )
    shots, shot_stats = detect_shots(times, hists, thumbs, luma, settings)
    place_threshold, place_stats = estimate_place_threshold(shots, settings)
    detection = detect_events(shots, place_threshold, settings, audio_times, scores)
    frame_times = np.asarray(times, dtype=np.float64)
    frame_scores = frame_change_scores(hists, thumbs)
    return DetectorRun(
        detection=detection,
        shots=shots,
        frame_times=frame_times,
        frame_scores=frame_scores,
        place_threshold=place_threshold,
        place_stats=place_stats,
        shot_stats=shot_stats,
        params=settings,
    )
