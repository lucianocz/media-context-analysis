from __future__ import annotations

import numpy as np

from media_context_analysis.events import detect_events
from media_context_analysis.grouping import group_timeline
from media_context_analysis.params import DetectorParams
from media_context_analysis.timestamps import choose_representatives
from tests.support import hist, shot, thumb


def test_pre_transition_is_the_last_sample_and_context_is_the_midpoint() -> None:
    previous = shot(0.0, 10.0, hist([1, 0]), thumb([1, 0]), frame_start=0, frame_end=40)
    current = shot(10.0, 14.0, hist([0, 1]), thumb([0, 1]), frame_start=40, frame_end=56)
    times = np.arange(56) * 0.25
    scores = np.zeros(56)
    event = {
        "shot_index": 1,
        "time_s": 10.0,
        "end_s": 14.0,
        "category": "visual_departure",
        "frames": {},
    }
    choice = choose_representatives(
        event,
        [previous, current],
        times,
        scores,
        sample_fps=4.0,
        cut_floor=0.22,
    )
    assert choice.transition_s == 10.0
    assert choice.context_reference_s == 5.0
    assert choice.context_reference_basis == "previous_shot_midpoint"
    assert choice.pre_transition_s == 9.75
    assert choice.pre_transition_basis == "last_sample_before_cut"
    assert choice.pre_transition_s != choice.context_reference_s
    assert abs(choice.pre_transition_s - choice.transition_s) < abs(
        choice.context_reference_s - choice.transition_s
    )


def test_unstable_pre_frame_steps_back_to_a_stable_sample() -> None:
    previous = shot(0.0, 4.0, hist([1, 0]), thumb([1, 0]), frame_start=0, frame_end=16)
    current = shot(4.0, 8.0, hist([0, 1]), thumb([0, 1]), frame_start=16, frame_end=32)
    times = np.arange(32) * 0.25
    scores = np.zeros(32)
    scores[14] = 0.4
    scores[15] = 0.5
    event = {
        "shot_index": 1,
        "time_s": 4.0,
        "end_s": 8.0,
        "category": "palette_shift",
        "frames": {},
    }
    choice = choose_representatives(
        event, [previous, current], times, scores, sample_fps=4.0, cut_floor=0.22
    )
    assert choice.pre_transition_s == 3.25
    assert choice.pre_transition_basis == "latest_stable_sample_within_1s"


def test_after_uses_the_earliest_stable_sample_instead_of_a_fixed_offset() -> None:
    previous = shot(0.0, 2.0, hist([1, 0]), thumb([1, 0]), frame_start=0, frame_end=8)
    current = shot(2.0, 6.0, hist([0, 1]), thumb([0, 1]), frame_start=8, frame_end=24)
    times = np.arange(24) * 0.25
    scores = np.zeros(24)
    scores[8] = 0.9
    scores[9] = 0.5
    scores[10] = 0.05
    event = {
        "shot_index": 1,
        "time_s": 2.0,
        "end_s": 6.0,
        "category": "sharp_stable_cut",
        "frames": {},
    }
    choice = choose_representatives(
        event, [previous, current], times, scores, sample_fps=4.0, cut_floor=0.22
    )
    assert choice.after_s == 2.5
    assert choice.after_basis == "earliest_stable_sample_within_1s"
    assert choice.transition_s == 2.0


def test_group_links_a_return_without_dropping_either_event() -> None:
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    third = hist([0, 0, 1, 0, 0, 0, 0, 0])
    face = thumb([1, 0, 0, 0])
    other_face = thumb([0, 1, 0, 0])
    third_face = thumb([0, 0, 1, 0])
    shots = [
        shot(0, 4, place, face),
        shot(4, 8, place, face),
        shot(8, 10, other, other_face, luma=0.2),
        shot(10, 11.5, third, third_face, luma=0.2),
        shot(11.5, 16, place, face),
    ]
    result = detect_events(shots, 0.48, DetectorParams())
    groups = group_timeline(result["timeline"])
    assert groups
    assert groups[0]["kind"] == "brief_insert_return"
    assert groups[0]["creates_translation_segment"] is False
    ids = {event["id"] for event in result["timeline"]}
    assert set(groups[0]["event_ids"]) <= ids
    assert len(groups[0]["event_ids"]) >= 2
    times = {event["id"]: event["time_s"] for event in result["timeline"]}
    assert times[groups[0]["event_ids"][0]] == 8.0
