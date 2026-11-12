from __future__ import annotations

import numpy as np

from media_context_analysis.shots import detect_shots, estimate_place_threshold, link_against
from tests.support import hist, params, shot, thumb


def test_detect_shots_finds_one_hard_change() -> None:
    frame_count = 40
    times = np.arange(frame_count) * 0.25
    place = hist([1, 1, 1, 1, 0, 0, 0, 0])
    other = hist([0, 0, 0, 0, 1, 1, 1, 1])
    hists = np.vstack([place] * 16 + [other] * 24)
    thumb_a = thumb([1, 0, 0, 0])
    thumb_b = thumb([0, 0, 1, 0])
    thumbs = np.vstack([thumb_a] * 16 + [thumb_b] * 24)
    luma = np.full(frame_count, 0.4)
    shots, stats = detect_shots(times, hists, thumbs, luma, params())
    assert stats["cut_count"] == 1
    assert len(shots) == 2
    assert abs(shots[1].start_s - 4.0) < 0.02
    assert shots[0].frame_start == 0
    assert shots[0].frame_end == shots[1].frame_start
    assert shots[1].frame_end == frame_count


def test_short_input_is_one_shot() -> None:
    times = np.asarray([0.0])
    hists = hist([1, 0, 0, 0])[None, :]
    thumbs = thumb([1, 0])[None, :]
    luma = np.asarray([0.2])
    shots, stats = detect_shots(times, hists, thumbs, luma, params())
    assert len(shots) == 1
    assert stats["cut_count"] == 0
    assert stats["cut_threshold"] is None


def test_place_threshold_clips_to_the_frozen_floor() -> None:
    place = hist([1, 0, 0, 0])
    other = hist([0, 0, 0, 1])
    face = thumb([1, 0])
    shots = [
        shot(0, 1, place, face),
        shot(1, 2, other, face),
        shot(2, 3, place, face),
    ]
    threshold, stats = estimate_place_threshold(shots, params())
    assert threshold == params().place_thr_min
    assert stats["clipped_threshold"] == params().place_thr_min


def test_link_uses_the_lookback_window_only() -> None:
    place = hist([1, 0, 0, 0])
    face = thumb([1, 0, 0])
    shots = [shot(0, 1, place, face), shot(30, 31, place, face)]
    linked, _place, _shot = link_against(shots, 1, 1, 0.48, 0.84, 20.0)
    assert linked is False
