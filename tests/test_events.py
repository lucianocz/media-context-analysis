"""Synthetic checks for the frozen event detector. No video files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from media_context_analysis.events import detect_events, merge_timeline
from media_context_analysis.params import DetectorParams
from tests.support import hist, shot, thumb

ROOT = Path(__file__).resolve().parents[1]


def _glue_audio(moment: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, 40.0, 0.2)
    scores = np.linspace(0.0, 1.0, len(times))
    nearest = int(np.argmin(np.abs(times - moment)))
    scores[nearest] = 0.0
    return times, scores


def test_framing_linked_palette_shift_survives_audio_glue() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    face = thumb([1, 0, 0, 0])
    shots = [
        shot(0, 4, place, face, change=0.2),
        shot(4, 8, place, face, change=0.2),
        shot(8, 12, place, face, change=0.2),
        shot(12, 15, other, face, change=0.2),
        shot(15, 20, other, face, change=0.2),
    ]
    times, scores = _glue_audio(12.0)
    result = detect_events(shots, 0.48, DetectorParams(), times, scores)
    hits = [event for event in result["timeline"] if event["time_s"] == 12.0]
    assert len(hits) == 1
    event = hits[0]
    assert event["category"] == "palette_shift"
    assert event["forces_translation_segment"] is False
    assert event["suggested_context_break"] is False
    assert event["scores"]["audio_vote"] == "glue"
    assert any("framing cosine" in note for note in event["continuity"])


def test_two_dips_six_seconds_apart_both_stay() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    third = hist([0, 0, 0, 0, 0, 0, 1, 0])
    face = thumb([1, 0, 0, 0])
    colors = [place, place, place, other, place, third, place]
    shots = [
        shot(index * 3.0, index * 3.0 + 3.0, color, face) for index, color in enumerate(colors)
    ]
    result = detect_events(shots, 0.48, DetectorParams())
    times = [event["time_s"] for event in result["timeline"]]
    assert 9.0 in times
    assert 15.0 in times
    for event in result["timeline"]:
        assert event["suggested_context_break"] is False
        assert event["forces_translation_segment"] is False


def test_brief_insert_does_not_suggest_a_context_break() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    face = thumb([1, 0, 0, 0])
    other_face = thumb([0, 1, 0, 0])
    shots = [
        shot(0, 4, place, face),
        shot(4, 8, place, face),
        shot(8, 10, other, other_face, luma=0.05),
        shot(10, 14, place, face),
    ]
    result = detect_events(shots, 0.48, DetectorParams())
    hits = [event for event in result["timeline"] if event["time_s"] == 8.0]
    assert len(hits) == 1
    event = hits[0]
    assert event["category"] == "brief_visual_insert"
    assert event["suggested_context_break"] is False
    assert event["forces_translation_segment"] is False
    assert "return_s" in event["frames"]
    assert event["scores"]["near_black"] is True
    blob = json.dumps(event).lower()
    assert "flashback" not in blob
    assert "title" not in blob


def test_mild_dip_above_the_place_threshold_stays_off_the_timeline() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    close = hist([0.7, 0.3, 0, 0, 0, 0, 0, 0])
    face = thumb([1, 0, 0, 0])
    shots = [
        shot(0, 4, place, face, change=0.2),
        shot(4, 8, place, face, change=0.2),
        shot(8, 12, place, face, change=0.2),
        shot(12, 14, close, face, change=0.2),
        shot(14, 18, place, face, change=0.2),
    ]
    result = detect_events(shots, 0.48, DetectorParams())
    assert 12.0 not in [event["time_s"] for event in result["timeline"]]
    assert 12.0 in result["diagnostics"]["family_times"]["local_dip_without_place_gate"]


def test_sharp_cut_is_kept_when_an_earlier_shot_still_matches() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    face = thumb([1, 0, 0, 0])
    other_face = thumb([0, 1, 0, 0])
    shots = [
        shot(0, 5, place, face, change=0.0),
        shot(5, 10, place, face, change=0.2),
        shot(10, 15, place, face, change=0.2),
        shot(15, 20, place, face, change=0.2),
        shot(20, 25, other, other_face, change=0.2),
        shot(25, 31, place, face, change=0.95),
    ]
    result = detect_events(shots, 0.48, DetectorParams())
    hits = [event for event in result["timeline"] if event["time_s"] == 25.0]
    assert len(hits) == 1
    event = hits[0]
    assert event["category"] == "sharp_stable_cut"
    assert event["priority"] == "high"
    assert event["suggested_context_break"] is False
    assert event["forces_translation_segment"] is False
    assert any("color intersection" in note for note in event["continuity"])


def test_detector_runs_without_audio() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    face = thumb([1, 0, 0, 0])
    other_face = thumb([0, 1, 0, 0])
    shots = [
        shot(0, 4, place, face),
        shot(4, 8, place, face),
        shot(8, 14, other, other_face),
    ]
    result = detect_events(shots, 0.48, DetectorParams(), None, None)
    assert result["timeline"]
    assert result["rules"]["subtitles_used"] is False
    assert result["timeline"][0]["scores"]["audio_vote"] == "unavailable"


def test_merge_prefers_the_stronger_type_inside_the_window() -> None:
    earlier = {
        "time_s": 10.0,
        "shot_index": 1,
        "category": "palette_shift",
        "scores": {"best_place_similarity": 0.05},
    }
    later = {
        "time_s": 11.2,
        "shot_index": 2,
        "category": "visual_departure",
        "scores": {"best_place_similarity": 0.4},
    }
    far = {
        "time_s": 13.0,
        "shot_index": 3,
        "category": "palette_shift",
        "scores": {"best_place_similarity": 0.2},
    }
    merged = merge_timeline([earlier, later, far], 1.5)
    assert [item["time_s"] for item in merged] == [11.2, 13.0]
    assert merged[0]["category"] == "visual_departure"


def test_timeline_ids_are_ordered_and_detection_text_names_no_annotations() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    face = thumb([1, 0, 0, 0])
    shots = [shot(0, 4, place, face), shot(4, 8, other, face), shot(8, 14, place, face)]
    result = detect_events(shots, 0.48, DetectorParams())
    times = [event["time_s"] for event in result["timeline"]]
    assert times == sorted(times)
    assert result["timeline"][0]["id"] == "e0001"
    for name in ("events.py", "shots.py", "signal.py", "analyze.py"):
        text = (ROOT / "src" / "media_context_analysis" / name).read_text(encoding="utf-8").lower()
        assert "reference_annotations" not in text
        assert "moviechat" not in text
        assert "rimuru" not in text
        assert "tmdb" not in text


def test_frozen_thresholds_stay_at_the_validated_defaults() -> None:
    settings = DetectorParams()
    assert settings.sample_fps == 4.0
    assert settings.cut_percentile == 90.0
    assert settings.cut_floor == 0.22
    assert settings.min_shot_s == 0.75
    assert settings.place_thr_min == 0.48
    assert settings.place_thr_max == 0.72
    assert settings.shot_thr == 0.84
    assert settings.tau_s == 20.0
    assert settings.brief_insert_s == 4.0
    assert settings.dedupe_s == 1.5
    assert settings.ffmpeg_threads == 2
    assert settings.audio_glue_percentile == 40.0
    assert settings.audio_split_percentile == 70.0
