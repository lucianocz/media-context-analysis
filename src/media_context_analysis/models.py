"""Typed analysis result. Field order in :meth:`AnalysisResult.to_dict` is the JSON contract."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from media_context_analysis._version import SCHEMA_VERSION
from media_context_analysis.timeutil import round_score

CATEGORIES = frozenset(
    {
        "brief_visual_insert",
        "visual_departure",
        "sharp_stable_cut",
        "palette_shift",
        "palette_break",
    }
)
PRIORITIES = frozenset({"high", "medium", "low"})
ACOUSTIC_VOTES = frozenset({"glue", "split", "neutral", "unavailable"})
GROUP_KINDS = frozenset({"brief_insert_return"})
TIMELINE_CATEGORIES = frozenset(
    {"brief_visual_insert", "visual_departure", "sharp_stable_cut", "palette_shift"}
)
_SCHEMA_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class MediaMetadata:
    filename: str
    container: str | None
    duration_s: float
    size_bytes: int
    video_codec: str | None
    width: int | None
    height: int | None
    avg_frame_rate: str | None
    fps: float | None
    audio_present: bool
    audio_codec: str | None
    audio_language: str | None
    subtitle_stream_count: int
    subtitles_read: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "container": self.container,
            "duration_s": self.duration_s,
            "size_bytes": self.size_bytes,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "avg_frame_rate": self.avg_frame_rate,
            "fps": self.fps,
            "audio_present": self.audio_present,
            "audio_codec": self.audio_codec,
            "audio_language": self.audio_language,
            "subtitle_stream_count": self.subtitle_stream_count,
            "subtitles_read": self.subtitles_read,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MediaMetadata:
        _require_false(payload, "subtitles_read")
        return cls(
            filename=_str(payload, "filename"),
            container=_opt_str(payload, "container"),
            duration_s=_float(payload, "duration_s"),
            size_bytes=_int(payload, "size_bytes"),
            video_codec=_opt_str(payload, "video_codec"),
            width=_opt_int(payload, "width"),
            height=_opt_int(payload, "height"),
            avg_frame_rate=_opt_str(payload, "avg_frame_rate"),
            fps=_opt_float(payload, "fps"),
            audio_present=_bool(payload, "audio_present"),
            audio_codec=_opt_str(payload, "audio_codec"),
            audio_language=_opt_str(payload, "audio_language"),
            subtitle_stream_count=_int(payload, "subtitle_stream_count"),
            subtitles_read=False,
        )


@dataclass(frozen=True)
class Provenance:
    library_name: str
    library_version: str
    detector_baseline: str
    detector_note: str
    feature_version: int
    subtitles_used: bool
    audio_deletes_events: bool
    ffmpeg_version: str | None
    ffprobe_version: str | None
    video_threads: int
    audio_threads: int
    visual_filter: str
    detector_params: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "library_name": self.library_name,
            "library_version": self.library_version,
            "detector_baseline": self.detector_baseline,
            "detector_note": self.detector_note,
            "feature_version": self.feature_version,
            "subtitles_used": self.subtitles_used,
            "audio_deletes_events": self.audio_deletes_events,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
            "video_threads": self.video_threads,
            "audio_threads": self.audio_threads,
            "visual_filter": self.visual_filter,
            "detector_params": copy.deepcopy(self.detector_params),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Provenance:
        _require_false(payload, "subtitles_used")
        _require_false(payload, "audio_deletes_events")
        params = payload.get("detector_params")
        if not isinstance(params, dict):
            raise ValueError("provenance.detector_params must be an object")
        return cls(
            library_name=_str(payload, "library_name"),
            library_version=_str(payload, "library_version"),
            detector_baseline=_str(payload, "detector_baseline"),
            detector_note=_str(payload, "detector_note"),
            feature_version=_int(payload, "feature_version"),
            subtitles_used=False,
            audio_deletes_events=False,
            ffmpeg_version=_opt_str(payload, "ffmpeg_version"),
            ffprobe_version=_opt_str(payload, "ffprobe_version"),
            video_threads=_int(payload, "video_threads"),
            audio_threads=_int(payload, "audio_threads"),
            visual_filter=_str(payload, "visual_filter"),
            detector_params=copy.deepcopy(params),
        )


@dataclass(frozen=True)
class EventScores:
    best_place_similarity: float
    best_framing_similarity: float
    adjacent_place_similarity: float
    change_score: float
    luma: float
    luma_step: float
    shot_duration_s: float
    place_match_age_s: float | None
    audio_score: float | None
    audio_vote: str
    near_black: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "best_place_similarity": self.best_place_similarity,
            "best_framing_similarity": self.best_framing_similarity,
            "adjacent_place_similarity": self.adjacent_place_similarity,
            "change_score": self.change_score,
            "luma": self.luma,
            "luma_step": self.luma_step,
            "shot_duration_s": self.shot_duration_s,
            "place_match_age_s": self.place_match_age_s,
            "audio_score": self.audio_score,
            "audio_vote": self.audio_vote,
            "near_black": self.near_black,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EventScores:
        vote = _str(payload, "audio_vote")
        if vote not in ACOUSTIC_VOTES:
            raise ValueError(f"unknown audio_vote {vote!r}")
        return cls(
            best_place_similarity=_float(payload, "best_place_similarity"),
            best_framing_similarity=_float(payload, "best_framing_similarity"),
            adjacent_place_similarity=_float(payload, "adjacent_place_similarity"),
            change_score=_float(payload, "change_score"),
            luma=_float(payload, "luma"),
            luma_step=_float(payload, "luma_step"),
            shot_duration_s=_float(payload, "shot_duration_s"),
            place_match_age_s=_opt_float(payload, "place_match_age_s"),
            audio_score=_opt_float(payload, "audio_score"),
            audio_vote=vote,
            near_black=_bool(payload, "near_black"),
        )


@dataclass(frozen=True)
class Continuity:
    framing_match: bool
    color_match: bool
    acoustic_vote: str
    short_return: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "framing_match": self.framing_match,
            "color_match": self.color_match,
            "acoustic_vote": self.acoustic_vote,
            "short_return": self.short_return,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Continuity:
        vote = _str(payload, "acoustic_vote")
        if vote not in ACOUSTIC_VOTES:
            raise ValueError(f"unknown acoustic_vote {vote!r}")
        return cls(
            framing_match=_bool(payload, "framing_match"),
            color_match=_bool(payload, "color_match"),
            acoustic_vote=vote,
            short_return=_bool(payload, "short_return"),
            notes=_str_tuple(payload, "notes"),
        )


@dataclass(frozen=True)
class Representatives:
    pre_transition_s: float
    pre_transition: str
    pre_transition_basis: str
    context_reference_s: float
    context_reference: str
    context_reference_basis: str
    transition_s: float
    transition: str
    after_s: float
    after: str
    after_basis: str
    return_s: float | None
    return_time: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "pre_transition_s": self.pre_transition_s,
            "pre_transition": self.pre_transition,
            "pre_transition_basis": self.pre_transition_basis,
            "context_reference_s": self.context_reference_s,
            "context_reference": self.context_reference,
            "context_reference_basis": self.context_reference_basis,
            "transition_s": self.transition_s,
            "transition": self.transition,
            "after_s": self.after_s,
            "after": self.after,
            "after_basis": self.after_basis,
            "return_s": self.return_s,
            "return": self.return_time,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Representatives:
        return cls(
            pre_transition_s=_float(payload, "pre_transition_s"),
            pre_transition=_str(payload, "pre_transition"),
            pre_transition_basis=_str(payload, "pre_transition_basis"),
            context_reference_s=_float(payload, "context_reference_s"),
            context_reference=_str(payload, "context_reference"),
            context_reference_basis=_str(payload, "context_reference_basis"),
            transition_s=_float(payload, "transition_s"),
            transition=_str(payload, "transition"),
            after_s=_float(payload, "after_s"),
            after=_str(payload, "after"),
            after_basis=_str(payload, "after_basis"),
            return_s=_opt_float(payload, "return_s"),
            return_time=_opt_str(payload, "return"),
        )


@dataclass(frozen=True)
class VisualEvent:
    id: str
    shot_index: int
    time_s: float
    time: str
    end_s: float
    end: str
    category: str
    priority: str
    reasons: tuple[str, ...]
    selected: bool
    suggested_context_break: bool
    forces_translation_segment: bool
    uncertainty: str
    evidence: tuple[str, ...]
    continuity: Continuity
    scores: EventScores
    representatives: Representatives

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "shot_index": self.shot_index,
            "time_s": self.time_s,
            "time": self.time,
            "end_s": self.end_s,
            "end": self.end,
            "category": self.category,
            "priority": self.priority,
            "reasons": list(self.reasons),
            "selected": self.selected,
            "suggested_context_break": self.suggested_context_break,
            "forces_translation_segment": self.forces_translation_segment,
            "uncertainty": self.uncertainty,
            "evidence": list(self.evidence),
            "continuity": self.continuity.to_dict(),
            "scores": self.scores.to_dict(),
            "representatives": self.representatives.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisualEvent:
        category = _str(payload, "category")
        priority = _str(payload, "priority")
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        if priority not in PRIORITIES:
            raise ValueError(f"unknown priority {priority!r}")
        _require_false(payload, "forces_translation_segment")
        continuity = payload.get("continuity")
        scores = payload.get("scores")
        representatives = payload.get("representatives")
        if (
            not isinstance(continuity, dict)
            or not isinstance(scores, dict)
            or not isinstance(representatives, dict)
        ):
            raise ValueError("event continuity, scores, and representatives must be objects")
        return cls(
            id=_str(payload, "id"),
            shot_index=_int(payload, "shot_index"),
            time_s=_float(payload, "time_s"),
            time=_str(payload, "time"),
            end_s=_float(payload, "end_s"),
            end=_str(payload, "end"),
            category=category,
            priority=priority,
            reasons=_str_tuple(payload, "reasons"),
            selected=_bool(payload, "selected"),
            suggested_context_break=_bool(payload, "suggested_context_break"),
            forces_translation_segment=False,
            uncertainty=_str(payload, "uncertainty"),
            evidence=_str_tuple(payload, "evidence"),
            continuity=Continuity.from_dict(continuity),
            scores=EventScores.from_dict(scores),
            representatives=Representatives.from_dict(representatives),
        )


@dataclass(frozen=True)
class EventGroup:
    id: str
    kind: str
    event_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    creates_translation_segment: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "event_ids": list(self.event_ids),
            "evidence": list(self.evidence),
            "creates_translation_segment": self.creates_translation_segment,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EventGroup:
        kind = _str(payload, "kind")
        if kind not in GROUP_KINDS:
            raise ValueError(f"unknown group kind {kind!r}")
        _require_false(payload, "creates_translation_segment")
        return cls(
            id=_str(payload, "id"),
            kind=kind,
            event_ids=_str_tuple(payload, "event_ids"),
            evidence=_str_tuple(payload, "evidence"),
            creates_translation_segment=False,
        )


@dataclass(frozen=True)
class Diagnostics:
    shot_count: int
    place_threshold: float
    place_threshold_stats: dict[str, object]
    thresholds: dict[str, object]
    audio_band: dict[str, object] | None
    rules: dict[str, object]
    shot_stats: dict[str, object]
    family_counts: dict[str, int]
    family_times: dict[str, list[float]]
    compact_before_dedupe: int
    raw_candidates: tuple[VisualEvent, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "shot_count": self.shot_count,
            "place_threshold": self.place_threshold,
            "place_threshold_stats": copy.deepcopy(self.place_threshold_stats),
            "thresholds": copy.deepcopy(self.thresholds),
            "audio_band": copy.deepcopy(self.audio_band),
            "rules": copy.deepcopy(self.rules),
            "shot_stats": copy.deepcopy(self.shot_stats),
            "family_counts": dict(self.family_counts),
            "family_times": copy.deepcopy(self.family_times),
            "compact_before_dedupe": self.compact_before_dedupe,
            "raw_candidates": [event.to_dict() for event in self.raw_candidates],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Diagnostics:
        raw = payload.get("raw_candidates")
        if not isinstance(raw, list):
            raise ValueError("diagnostics.raw_candidates must be a list")
        raw_items = [item for item in raw if isinstance(item, dict)]
        if len(raw_items) != len(raw):
            raise ValueError("diagnostics.raw_candidates items must be objects")
        return cls(
            shot_count=_int(payload, "shot_count"),
            place_threshold=_float(payload, "place_threshold"),
            place_threshold_stats=copy.deepcopy(_obj(payload, "place_threshold_stats")),
            thresholds=copy.deepcopy(_obj(payload, "thresholds")),
            audio_band=_copied_optional_object(payload, "audio_band"),
            rules=copy.deepcopy(_obj(payload, "rules")),
            shot_stats=copy.deepcopy(_obj(payload, "shot_stats")),
            family_counts=_family_counts(payload),
            family_times=_family_times(payload),
            compact_before_dedupe=_int(payload, "compact_before_dedupe"),
            raw_candidates=tuple(VisualEvent.from_dict(item) for item in raw_items),
        )


@dataclass(frozen=True)
class AnalysisResult:
    """Versioned analysis of one local video file. No image bytes are included."""

    schema_version: str
    media: MediaMetadata
    provenance: Provenance
    timeline: tuple[VisualEvent, ...]
    groups: tuple[EventGroup, ...]
    diagnostics: Diagnostics | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "media": self.media.to_dict(),
            "provenance": self.provenance.to_dict(),
            "timeline": [event.to_dict() for event in self.timeline],
            "groups": [group.to_dict() for group in self.groups],
            "diagnostics": None if self.diagnostics is None else self.diagnostics.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnalysisResult:
        version = _str(payload, "schema_version")
        if _SCHEMA_VERSION_RE.fullmatch(version) is None:
            raise ValueError(f"unsupported schema version {version}")
        if version.split(".", 1)[0] != SCHEMA_VERSION.split(".", 1)[0]:
            raise ValueError(f"unsupported schema version {version}")
        media = payload.get("media")
        provenance = payload.get("provenance")
        timeline = payload.get("timeline")
        groups = payload.get("groups")
        if not isinstance(media, dict) or not isinstance(provenance, dict):
            raise ValueError("media and provenance must be objects")
        if not isinstance(timeline, list) or not isinstance(groups, list):
            raise ValueError("timeline and groups must be lists")
        diagnostics = payload.get("diagnostics")
        parsed_diagnostics = None
        if diagnostics is not None:
            if not isinstance(diagnostics, dict):
                raise ValueError("diagnostics must be an object or null")
            parsed_diagnostics = Diagnostics.from_dict(diagnostics)
        timeline_items = [item for item in timeline if isinstance(item, dict)]
        group_items = [item for item in groups if isinstance(item, dict)]
        if len(timeline_items) != len(timeline):
            raise ValueError("timeline items must be objects")
        if len(group_items) != len(groups):
            raise ValueError("group items must be objects")
        return cls(
            schema_version=version,
            media=MediaMetadata.from_dict(media),
            provenance=Provenance.from_dict(provenance),
            timeline=tuple(VisualEvent.from_dict(item) for item in timeline_items),
            groups=tuple(EventGroup.from_dict(item) for item in group_items),
            diagnostics=parsed_diagnostics,
        )

    @classmethod
    def from_json(cls, text: str) -> AnalysisResult:
        payload = json.loads(text, parse_constant=_reject_json_constant)
        if not isinstance(payload, dict):
            raise ValueError("analysis JSON must be an object")
        return cls.from_dict(payload)


def scores_from_detection(raw: Mapping[str, Any]) -> EventScores:
    age = raw.get("place_match_age_s")
    audio = raw.get("audio_score")
    return EventScores(
        best_place_similarity=_required_score(raw, "best_place_similarity"),
        best_framing_similarity=_required_score(raw, "best_framing_similarity"),
        adjacent_place_similarity=_required_score(raw, "adjacent_place_similarity"),
        change_score=_required_score(raw, "change_score"),
        luma=_required_score(raw, "luma"),
        luma_step=_required_score(raw, "luma_step"),
        shot_duration_s=_required_score(raw, "shot_duration_s"),
        place_match_age_s=None if age is None else round_score(float(cast(float, age))),
        audio_score=None if audio is None else round_score(float(cast(float, audio))),
        audio_vote=str(raw["audio_vote"]),
        near_black=bool(raw["near_black"]),
    )


def _required_score(raw: Mapping[str, Any], key: str) -> float:
    value = round_score(float(cast(float, raw[key])))
    if value is None:
        raise ValueError(f"score {key} is not finite")
    return value


def _require_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise ValueError(f"{key} must be false")


def _str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _opt_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _finite(value: float, key: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def _float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return _finite(float(value), key)


def _opt_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    return _finite(float(value), key)


def _int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _opt_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null")
    return value


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


def _obj(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _opt_obj(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object or null")
    return dict(value)


def _copied_optional_object(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = _opt_obj(payload, key)
    if value is None:
        return None
    return copy.deepcopy(value)


def _family_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in _obj(payload, "family_counts").items():
        try:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            counts[str(key)] = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("diagnostics.family_counts values must be integers") from exc
    return counts


def _family_times(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    times: dict[str, list[float]] = {}
    for key, value in _obj(payload, "family_times").items():
        if not isinstance(value, list):
            raise ValueError("diagnostics.family_times values must be lists of numbers")
        converted: list[float] = []
        try:
            for item in value:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError
                number = float(item)
                if not math.isfinite(number):
                    raise ValueError
                converted.append(number)
        except (TypeError, ValueError) as exc:
            raise ValueError("diagnostics.family_times values must be lists of numbers") from exc
        times[str(key)] = converted
    return times


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON number {value}")
