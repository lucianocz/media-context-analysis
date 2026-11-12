"""Public analysis entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from media_context_analysis._version import DETECTOR_BASELINE, SCHEMA_VERSION, __version__
from media_context_analysis.errors import (
    AnalysisCancelledError,
    AnalysisTimeoutError,
    CorruptMediaError,
    MediaContextAnalysisError,
    MediaNotFoundError,
)
from media_context_analysis.ffmpeg import (
    AUDIO_THREADS,
    extract_audio_features,
    extract_visual,
    media_streams,
    probe_file,
    resolve_tool,
    tool_version,
    visual_filter,
)
from media_context_analysis.grouping import group_timeline
from media_context_analysis.models import (
    AnalysisResult,
    Continuity,
    Diagnostics,
    EventGroup,
    Provenance,
    Representatives,
    VisualEvent,
    scores_from_detection,
)
from media_context_analysis.options import AnalysisOptions
from media_context_analysis.params import DetectorParams
from media_context_analysis.pipeline import DetectorRun, run_detector
from media_context_analysis.timestamps import choose_representatives

_DETECTOR_NOTE = (
    "Event selection and thresholds match the visual event detector at the recorded baseline. "
    "Representative timestamps distinguish an immediate pre-transition sample from the "
    "previous-shot midpoint. Related-event groups do not remove events or open "
    "translation segments."
)
_LIBRARY_NAME = "media-context-analysis"


def analyze(path: str | Path, options: AnalysisOptions | None = None) -> AnalysisResult:
    """Analyze a local video file and return timestamps, evidence, and continuity.

    The result identifies when to look. It does not extract, store, or return images.
    """
    settings = options or AnalysisOptions()
    settings.validate()
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise MediaNotFoundError(f"media file not found: {source}")
    if source.stat().st_size <= 0:
        raise CorruptMediaError(f"media file is empty: {source.name}")
    deadline = None if settings.timeout_s is None else time.monotonic() + settings.timeout_s
    _raise_if_cancelled(settings)
    ffmpeg = resolve_tool(settings.ffmpeg, "ffmpeg")
    ffprobe = resolve_tool(settings.ffprobe, "ffprobe")
    ffmpeg_version = tool_version(ffmpeg, deadline=deadline, cancel_event=settings.cancel_event)
    ffprobe_version = tool_version(ffprobe, deadline=deadline, cancel_event=settings.cancel_event)
    _raise_if_cancelled(settings)
    probed = probe_file(source, ffprobe, deadline=deadline, cancel_event=settings.cancel_event)
    video_stream, audio_stream, _subtitle_count = media_streams(
        probed, video_stream_index=settings.video_stream_index
    )
    params = DetectorParams()
    _raise_if_cancelled(settings)
    visual = extract_visual(
        source,
        ffmpeg,
        params,
        video_stream,
        video_threads=settings.video_threads,
        deadline=deadline,
        cancel_event=settings.cancel_event,
    )
    audio_times: NDArray[np.floating] | None = None
    audio_features: NDArray[np.floating] | None = None
    if audio_stream is not None:
        _raise_if_cancelled(settings)
        audio_times, audio_features = extract_audio_features(
            source,
            ffmpeg,
            params,
            deadline=deadline,
            cancel_event=settings.cancel_event,
        )
        if audio_features.size == 0:
            audio_times = None
            audio_features = None
    _raise_if_cancelled(settings)
    run = run_detector(
        cast(NDArray[np.floating], visual["times"]),
        cast(NDArray[np.floating], visual["hists"]),
        cast(NDArray[np.floating], visual["thumbs"]),
        cast(NDArray[np.floating], visual["luma"]),
        audio_times,
        None,
        audio_features,
        params,
    )
    result = _assemble(
        video_stream,
        run,
        settings,
        ffmpeg_version,
        ffprobe_version,
    )
    _raise_if_cancelled(settings)
    if deadline is not None and time.monotonic() > deadline:
        raise AnalysisTimeoutError("analysis timed out")
    return result


def _assemble(
    video_stream: dict[str, object],
    run: DetectorRun,
    settings: AnalysisOptions,
    ffmpeg_version: str | None,
    ffprobe_version: str | None,
) -> AnalysisResult:
    timeline_rows = _event_rows(run.detection.get("timeline"), "timeline")
    raw_rows = _event_rows(run.detection.get("raw_candidates"), "raw_candidates")
    timeline = tuple(_visual_event(row, run) for row in timeline_rows)
    groups = tuple(_group(row) for row in group_timeline(timeline_rows))
    diagnostics = None
    if settings.include_diagnostics:
        diagnostics = _diagnostics(run, raw_rows)
    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        provenance=_provenance(
            run, settings, int(cast(int, video_stream["index"])), ffmpeg_version, ffprobe_version
        ),
        timeline=timeline,
        groups=groups,
        diagnostics=diagnostics,
    )


def _visual_event(row: dict[str, object], run: DetectorRun) -> VisualEvent:
    choice = choose_representatives(
        row,
        run.shots,
        run.frame_times,
        run.frame_scores,
        sample_fps=run.params.sample_fps,
        cut_floor=run.params.cut_floor,
    )
    raw_scores = row["scores"]
    if not isinstance(raw_scores, dict):
        raise MediaContextAnalysisError("detector scores were not an object")
    notes = row["continuity"]
    if not isinstance(notes, list):
        raise MediaContextAnalysisError("detector continuity notes were not a list")
    reasons = row["reasons"]
    evidence = row["evidence"]
    if not isinstance(reasons, list) or not isinstance(evidence, list):
        raise MediaContextAnalysisError("detector evidence was not a list")
    best_shot = float(cast(float, raw_scores["best_framing_similarity"]))
    best_place = float(cast(float, raw_scores["best_place_similarity"]))
    age = raw_scores.get("place_match_age_s")
    return VisualEvent(
        id=str(row["id"]),
        shot_index=int(cast(int, row["shot_index"])),
        time_s=float(cast(float, row["time_s"])),
        time=str(row["time"]),
        end_s=float(cast(float, row["end_s"])),
        end=str(row["end"]),
        category=str(row["category"]),
        priority=str(row["priority"]),
        reasons=tuple(str(item) for item in reasons),
        selected=bool(row["kept"]),
        suggested_context_break=bool(row["suggested_context_break"]),
        forces_translation_segment=False,
        uncertainty=str(row["uncertainty"]),
        evidence=tuple(str(item) for item in evidence),
        continuity=Continuity(
            framing_match=best_shot >= run.params.shot_thr,
            color_match=best_place >= run.place_threshold and age is not None,
            acoustic_vote=str(raw_scores["audio_vote"]),
            short_return=row["category"] == "brief_visual_insert",
            notes=tuple(str(item) for item in notes),
        ),
        scores=scores_from_detection(raw_scores),
        representatives=Representatives.from_dict(choice.to_dict()),
    )


def _group(row: dict[str, object]) -> EventGroup:
    event_ids = row["event_ids"]
    evidence = row["evidence"]
    if not isinstance(event_ids, list) or not isinstance(evidence, list):
        raise MediaContextAnalysisError("event group was incomplete")
    return EventGroup(
        id=str(row["id"]),
        kind=str(row["kind"]),
        event_ids=tuple(str(item) for item in event_ids),
        evidence=tuple(str(item) for item in evidence),
        creates_translation_segment=False,
    )


def _diagnostics(run: DetectorRun, raw_rows: list[dict[str, object]]) -> Diagnostics:
    block = run.detection.get("diagnostics")
    if not isinstance(block, dict):
        raise MediaContextAnalysisError("detector diagnostics were missing")
    rules = run.detection.get("rules")
    thresholds = run.detection.get("thresholds")
    counts = block.get("family_counts")
    times = block.get("family_times")
    band = run.detection.get("audio_band")
    if not isinstance(rules, dict) or not isinstance(thresholds, dict):
        raise MediaContextAnalysisError("detector rules were missing")
    if not isinstance(counts, dict) or not isinstance(times, dict):
        raise MediaContextAnalysisError("detector family diagnostics were missing")
    return Diagnostics(
        shot_count=int(cast(int, run.detection["shot_count"])),
        place_threshold=round(run.place_threshold, 4),
        place_threshold_stats=dict(run.place_stats),
        thresholds=dict(thresholds),
        audio_band=None if band is None else dict(cast(dict[str, object], band)),
        rules=dict(rules),
        shot_stats=dict(run.shot_stats),
        family_counts={str(key): int(value) for key, value in counts.items()},
        family_times={
            str(key): [float(item) for item in value]
            for key, value in times.items()
            if isinstance(value, list)
        },
        compact_before_dedupe=int(cast(int, block["compact_before_dedupe"])),
        raw_candidates=tuple(_visual_event(row, run) for row in raw_rows),
    )


def _provenance(
    run: DetectorRun,
    settings: AnalysisOptions,
    video_stream_index: int,
    ffmpeg_version: str | None,
    ffprobe_version: str | None,
) -> Provenance:
    return Provenance(
        library_name=_LIBRARY_NAME,
        library_version=__version__,
        detector_baseline=DETECTOR_BASELINE,
        detector_note=_DETECTOR_NOTE,
        feature_version=run.params.feature_version,
        subtitles_used=False,
        audio_deletes_events=False,
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
        video_stream_index=video_stream_index,
        video_threads=settings.video_threads,
        audio_threads=AUDIO_THREADS,
        visual_filter=visual_filter(run.params),
        detector_params=run.params.to_dict(),
    )


def _event_rows(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise MediaContextAnalysisError(f"detector {label} was not a list")
    rows: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise MediaContextAnalysisError(f"detector {label} contained a non-object")
        rows.append(item)
    return rows


def _raise_if_cancelled(settings: AnalysisOptions) -> None:
    if settings.cancel_event is not None and settings.cancel_event.is_set():
        raise AnalysisCancelledError("analysis cancelled")
