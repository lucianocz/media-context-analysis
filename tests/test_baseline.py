"""Compare the port with cached research features when MCA_RESEARCH_ROOT is set."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from media_context_analysis.grouping import group_timeline
from media_context_analysis.pipeline import run_detector
from media_context_analysis.signal import acoustic_change_scores
from media_context_analysis.timestamps import choose_representatives

pytestmark = pytest.mark.benchmark

FIELDS = (
    "time_s",
    "end_s",
    "category",
    "priority",
    "shot_index",
    "suggested_context_break",
    "forces_translation_segment",
)
SCORE_FIELDS = (
    "best_place_similarity",
    "best_framing_similarity",
    "adjacent_place_similarity",
    "change_score",
    "luma",
    "luma_step",
    "shot_duration_s",
    "place_match_age_s",
    "audio_score",
    "audio_vote",
    "near_black",
)
CASES = (
    ("anime-validation", "cache/anime-validation", "exports/anime-validation/analysis.json"),
    ("slime", "cache", "results/events/slime.json"),
    ("moviechat-4", "external/moviechat/cache/4", "results/events/4.json"),
    ("moviechat-7", "external/moviechat/cache/7", "results/events/7.json"),
    ("moviechat-8", "external/moviechat/cache/8", "results/events/8.json"),
)


def _root() -> Path | None:
    value = os.environ.get("MCA_RESEARCH_ROOT")
    if not value:
        return None
    return Path(value)


def _rounded(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return value


def _timeline(document: dict[str, object]) -> list[dict[str, object]]:
    detection = document.get("detection")
    if isinstance(detection, dict) and isinstance(detection.get("timeline"), list):
        return [item for item in detection["timeline"] if isinstance(item, dict)]
    timeline = document.get("timeline")
    if not isinstance(timeline, list):
        raise AssertionError("baseline document has no timeline")
    return [item for item in timeline if isinstance(item, dict)]


def _shot_count(document: dict[str, object]) -> int | None:
    detection = document.get("detection")
    if isinstance(detection, dict) and "shot_count" in detection:
        return int(detection["shot_count"])  # type: ignore[arg-type]
    summary = document.get("summary")
    if isinstance(summary, dict) and "shot_count" in summary:
        return int(summary["shot_count"])  # type: ignore[arg-type]
    return None


def _raw_count(document: dict[str, object]) -> int | None:
    detection = document.get("detection")
    if isinstance(detection, dict) and isinstance(detection.get("raw_candidates"), list):
        return len(detection["raw_candidates"])
    summary = document.get("summary")
    if isinstance(summary, dict) and "raw_count" in summary:
        return int(summary["raw_count"])  # type: ignore[arg-type]
    return None


def test_cached_features_match_the_frozen_timeline() -> None:
    root = _root()
    if root is None:
        pytest.skip("MCA_RESEARCH_ROOT is not set")
    ran = 0
    for name, cache, document_path in CASES:
        visual_path = root / cache / "visual.npz"
        audio_path = root / cache / "audio.npz"
        expected_path = root / document_path
        if not visual_path.is_file() or not audio_path.is_file() or not expected_path.is_file():
            continue
        ran += 1
        _compare_case(name, visual_path, audio_path, expected_path)
    if ran == 0:
        pytest.fail("MCA_RESEARCH_ROOT did not contain a cached comparison")


def test_acoustic_function_matches_the_research_module() -> None:
    root = _root()
    if root is None:
        pytest.skip("MCA_RESEARCH_ROOT is not set")
    sys.path.insert(0, str(root))
    from ncs.features import acoustic_change_scores as research_scores

    features = np.random.default_rng(0).normal(size=(80, 8))
    ours = acoustic_change_scores(features, 0.2, 5.0)
    theirs = research_scores(features, 0.2, 5.0)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-6)


def _compare_case(name: str, visual_path: Path, audio_path: Path, expected_path: Path) -> None:
    visual = np.load(visual_path)
    audio = np.load(audio_path)
    document = json.loads(expected_path.read_text(encoding="utf-8"))
    expected = _timeline(document)
    run = run_detector(
        visual["times"],
        visual["hists"],
        visual["thumbs"],
        visual["luma"],
        audio["audio_times"],
        audio["audio_scores"],
    )
    actual = run.detection["timeline"]
    if not isinstance(actual, list):
        raise AssertionError(f"{name} detector timeline was not a list")
    assert len(actual) == len(expected), name
    mismatches: list[tuple[object, ...]] = []
    pre_diffs = 0
    after_diffs = 0
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if not isinstance(right, dict):
            mismatches.append((index, "type", "object", type(right).__name__))
            continue
        for field in FIELDS:
            if left.get(field) != right.get(field):
                mismatches.append((index, field, left.get(field), right.get(field)))
        if list(left.get("reasons", [])) != list(right.get("reasons", [])):
            mismatches.append((index, "reasons", left.get("reasons"), right.get("reasons")))
        left_scores = left.get("scores")
        right_scores = right.get("scores")
        if isinstance(left_scores, dict) and isinstance(right_scores, dict):
            for field in SCORE_FIELDS:
                if _rounded(left_scores.get(field)) != _rounded(right_scores.get(field)):
                    mismatches.append(
                        (index, field, left_scores.get(field), right_scores.get(field))
                    )
        frames = left.get("frames")
        if isinstance(frames, dict):
            choice = choose_representatives(
                right,
                run.shots,
                run.frame_times,
                run.frame_scores,
                sample_fps=run.params.sample_fps,
                cut_floor=run.params.cut_floor,
            )
            if choice.context_reference_s != frames.get("before_s"):
                mismatches.append(
                    (
                        index,
                        "context_reference_s",
                        frames.get("before_s"),
                        choice.context_reference_s,
                    )
                )
            if choice.pre_transition_s != frames.get("before_s"):
                pre_diffs += 1
            if choice.after_s != frames.get("after_s"):
                after_diffs += 1
    assert not mismatches, (name, mismatches[:6])
    shot_count = _shot_count(document)
    if shot_count is not None:
        assert run.detection["shot_count"] == shot_count, name
    raw_count = _raw_count(document)
    raw = run.detection["raw_candidates"]
    if raw_count is not None and isinstance(raw, list):
        assert len(raw) == raw_count, name
    groups = group_timeline(actual)
    ids = {str(item["id"]) for item in actual if isinstance(item, dict)}
    for group in groups:
        assert group["creates_translation_segment"] is False
        assert set(group["event_ids"]) <= ids
    recomputed = acoustic_change_scores(audio["audio_features"], 0.2, 5.0)
    np.testing.assert_allclose(recomputed, audio["audio_scores"], rtol=1e-5, atol=1e-5)
    warnings.warn(
        f"{name}: events={len(actual)} groups={len(groups)} pre_transition_diffs={pre_diffs} after_diffs={after_diffs}",
        stacklevel=2,
    )
