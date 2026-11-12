from __future__ import annotations

import numpy as np

from media_context_analysis.signal import (
    PcmFeatureReducer,
    acoustic_change_scores,
    cosine,
    frame_change_scores,
    grayscale_thumbnail,
    hsv_histogram,
    intersection,
    spectral_features,
)
from media_context_analysis.timeutil import format_time, round_score, round_seconds


def test_histogram_is_normalized_and_separates_colors() -> None:
    red = np.zeros((54, 96, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros_like(red)
    blue[:, :, 2] = 255
    red_hist = hsv_histogram(red, 16, 4, 4)
    blue_hist = hsv_histogram(blue, 16, 4, 4)
    assert red_hist.shape == (16 * 4 * 4,)
    assert abs(float(red_hist.sum()) - 1.0) < 1e-5
    assert intersection(red_hist, blue_hist) < 0.05


def test_thumbnail_is_unit_norm() -> None:
    rgb = np.zeros((54, 96, 3), dtype=np.uint8)
    rgb[:27, :, 1] = 255
    vector = grayscale_thumbnail(rgb, 16, 9)
    assert vector.shape == (16 * 9,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_cosine_handles_zero_vectors() -> None:
    assert cosine(np.zeros(4), np.ones(4)) == 0.0
    same = np.asarray([1.0, 0.0, 0.0])
    assert cosine(same, same) == 1.0


def test_frame_change_is_zero_until_the_picture_changes() -> None:
    place = np.asarray([1.0, 0.0, 0.0, 0.0])
    other = np.asarray([0.0, 0.0, 0.0, 1.0])
    hists = np.vstack([place, place, other])
    thumbs = hists.copy()
    scores = frame_change_scores(hists, thumbs)
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] > 0.5


def test_streaming_pcm_matches_the_batch_features() -> None:
    rate = 16000
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 0.1, rate).astype(np.float32)
    batch_times, batch_rows = spectral_features(samples, rate, 0.40, 0.20)
    reducer = PcmFeatureReducer(rate, 0.40, 0.20)
    for start in range(0, samples.size, 1000):
        reducer.extend(samples[start : start + 1000])
    times, rows = reducer.finish()
    assert times.shape == batch_times.shape
    assert np.allclose(times, batch_times)
    assert np.allclose(rows, batch_rows)


def test_acoustic_scores_stay_in_range_and_are_deterministic() -> None:
    features = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.2, 0.1, 0.0, 0.4, 0.9, 0.1, 0.2, 0.3],
            [0.8, 0.7, 0.1, 0.1, 0.2, 0.2, 0.4, 0.5],
            [0.8, 0.7, 0.1, 0.1, 0.2, 0.2, 0.4, 0.5],
            [0.0, 0.1, 0.2, 0.8, 0.7, 0.6, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    # The 5 s window is wider than this fixture, so use a short window.
    first = acoustic_change_scores(features, hop_s=0.2, window_s=0.4)
    second = acoustic_change_scores(features, hop_s=0.2, window_s=0.4)
    assert first.shape == (len(features),)
    assert np.allclose(first, second)
    assert float(first.min()) >= 0.0
    # Cosine distance is 1 - cosine, so opposite windows can score up to 2.
    assert float(first.max()) <= 2.0


def test_round_score_drops_non_finite_values() -> None:
    assert round_score(None) is None
    assert round_score(float("nan")) is None
    assert round_score(float("inf")) is None
    assert round_score(0.123456) == 0.1235


def test_time_format_matches_the_frozen_helper() -> None:
    assert format_time(90) == "00:01:30.000"
    assert format_time(128.75) == "00:02:08.750"
    assert format_time(-1) == "00:00:00.000"
    assert round_seconds(1.499999999) == 1.5
