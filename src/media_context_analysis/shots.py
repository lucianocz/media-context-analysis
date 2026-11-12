"""Shot boundaries and look-back links from the frozen detector.

Adapted from narrative-context-segmentation ``segment.py`` at commit
68cc838a008ad279752bcf598e2039fcabac8be0. Context segmentation and subtitle
votes are not part of this library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from media_context_analysis.params import DetectorParams
from media_context_analysis.signal import cosine, frame_change_scores, intersection


@dataclass
class Shot:
    """One visually stable span. ``frame_end`` is exclusive; both are -1 when unknown."""

    start_s: float
    end_s: float
    hist: NDArray[np.float64]
    thumb: NDArray[np.float64]
    luma: float
    change_score: float
    frame_start: int = -1
    frame_end: int = -1


def detect_shots(
    times: NDArray[np.floating],
    hists: NDArray[np.floating],
    thumbs: NDArray[np.floating],
    luma: NDArray[np.floating],
    params: DetectorParams,
) -> tuple[list[Shot], dict[str, object]]:
    scores = frame_change_scores(hists, thumbs)
    if len(scores) < 2:
        step = 1.0 / params.sample_fps
        shot = _make_shot(0, len(times), times, hists, thumbs, luma, scores, step)
        return [shot], {"cut_threshold": None, "cut_count": 0, "frame_count": len(times)}
    threshold = max(
        params.cut_floor, float(np.percentile(scores[1:], params.cut_percentile, method="linear"))
    )
    cuts: list[int] = []
    for index in range(1, len(scores) - 1):
        if (
            scores[index] >= threshold
            and scores[index] >= scores[index - 1]
            and scores[index] > scores[index + 1]
        ):
            cuts.append(index)
    if scores[-1] >= threshold and scores[-1] > scores[-2]:
        cuts.append(len(scores) - 1)
    kept: list[int] = []
    for index in cuts:
        if not kept or float(times[index] - times[kept[-1]]) >= params.min_shot_s:
            kept.append(index)
        elif scores[index] > scores[kept[-1]]:
            kept[-1] = index
    if len(times) >= 2:
        step = float(np.median(np.diff(times)))
    else:
        step = 1.0 / params.sample_fps
    starts = [0, *kept]
    ends = [*kept, len(times)]
    shots = [
        _make_shot(start, end, times, hists, thumbs, luma, scores, step)
        for start, end in zip(starts, ends, strict=True)
    ]
    return shots, {
        "cut_threshold": round(threshold, 4),
        "cut_count": len(kept),
        "frame_count": len(times),
        "change_p50": round(float(np.percentile(scores[1:], 50, method="linear")), 4),
        "change_p90": round(float(np.percentile(scores[1:], 90, method="linear")), 4),
        "change_p99": round(float(np.percentile(scores[1:], 99, method="linear")), 4),
    }


def _make_shot(
    start: int,
    end: int,
    times: NDArray[np.floating],
    hists: NDArray[np.floating],
    thumbs: NDArray[np.floating],
    luma: NDArray[np.floating],
    scores: NDArray[np.floating],
    step: float,
) -> Shot:
    histogram = np.mean(hists[start:end], axis=0)
    total = float(histogram.sum())
    if total > 0:
        histogram = histogram / total
    thumb = np.mean(thumbs[start:end], axis=0)
    norm = float(np.linalg.norm(thumb))
    if norm > 1e-8:
        thumb = thumb / norm
    end_time = float(times[end]) if end < len(times) else float(times[end - 1]) + step
    return Shot(
        start_s=float(times[start]),
        end_s=end_time,
        hist=np.asarray(histogram, dtype=np.float64),
        thumb=np.asarray(thumb, dtype=np.float64),
        luma=float(np.mean(luma[start:end])),
        change_score=float(scores[start]) if start else 0.0,
        frame_start=start,
        frame_end=end,
    )


def estimate_place_threshold(
    shots: list[Shot], params: DetectorParams
) -> tuple[float, dict[str, object]]:
    if len(shots) < 3:
        return params.place_thr_min, {
            "adjacent_count": max(0, len(shots) - 1),
            "raw_percentile": None,
        }
    similarities = np.asarray(
        [intersection(shots[index].hist, shots[index - 1].hist) for index in range(1, len(shots))],
        dtype=np.float64,
    )
    raw = float(np.percentile(similarities, params.place_thr_percentile, method="linear"))
    threshold = min(params.place_thr_max, max(params.place_thr_min, raw))
    return threshold, {
        "adjacent_count": int(similarities.size),
        "raw_percentile": round(raw, 4),
        "clipped_threshold": round(threshold, 4),
        "adjacent_place_p50": round(float(np.percentile(similarities, 50, method="linear")), 4),
        "adjacent_place_p10": round(float(np.percentile(similarities, 10, method="linear")), 4),
        "adjacent_place_p90": round(float(np.percentile(similarities, 90, method="linear")), 4),
    }


def link_against(
    shots: list[Shot],
    query: int,
    limit: int,
    place_thr: float,
    shot_thr: float,
    tau_s: float,
) -> tuple[bool, float, float]:
    best_place = 0.0
    best_shot = 0.0
    linked = False
    index = limit - 1
    query_shot = shots[query]
    while index >= 0 and query_shot.start_s - shots[index].start_s <= tau_s:
        place = intersection(query_shot.hist, shots[index].hist)
        framing = cosine(query_shot.thumb, shots[index].thumb)
        best_place = max(best_place, place)
        best_shot = max(best_shot, framing)
        if place >= place_thr or framing >= shot_thr:
            linked = True
        index -= 1
    return linked, best_place, best_shot


def visual_candidates(
    shots: list[Shot], place_thr: float, params: DetectorParams
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return context-boundary candidates and brief-insert events.

    Only the insert list is used by event detection. Boundaries are retained so
    the frozen insert scan stays intact.
    """
    boundaries: list[dict[str, object]] = []
    inserts: list[dict[str, object]] = []
    index = 1
    while index < len(shots):
        linked, best_place, best_shot = link_against(
            shots, index, index, place_thr, params.shot_thr, params.tau_s
        )
        novelty = 1.0 - max(best_place, best_shot)
        if linked:
            index += 1
            continue
        return_at = None
        for later in range(index + 1, len(shots)):
            if shots[later].start_s - shots[index].start_s > params.brief_insert_s:
                break
            returned, _place, _shot = link_against(
                shots, later, index, place_thr, params.shot_thr, params.tau_s
            )
            if returned:
                return_at = later
                break
        evidence = [
            f"best look-back histogram intersection {best_place:.3f} (link at {place_thr:.3f})",
            f"best look-back framing cosine {best_shot:.3f} (link at {params.shot_thr:.3f})",
            f"look-back window {params.tau_s:.0f}s",
        ]
        if shots[index].luma < 0.08:
            evidence.append(f"mean luminance {shots[index].luma:.3f}, near-black")
        if return_at is not None:
            inserts.append(
                {
                    "time_s": shots[index].start_s,
                    "end_s": shots[return_at].start_s,
                    "category": "brief_visual_insert",
                    "splits_context": False,
                    "visual_novelty": novelty,
                    "best_place_similarity": best_place,
                    "best_shot_similarity": best_shot,
                    "evidence": [
                        *evidence,
                        "following shots return to the preceding visual context",
                        f"insert length {shots[return_at].start_s - shots[index].start_s:.2f}s",
                    ],
                }
            )
            index = return_at
            continue
        boundaries.append(
            {
                "time_s": shots[index].start_s,
                "shot_index": index,
                "visual_novelty": novelty,
                "best_place_similarity": best_place,
                "best_shot_similarity": best_shot,
                "luma": shots[index].luma,
                "evidence": evidence,
            }
        )
        index += 1
    return boundaries, inserts


def percentile_band(
    scores: NDArray[np.floating] | None, params: DetectorParams
) -> dict[str, float | bool] | None:
    if scores is None or len(scores) < 10:
        return None
    low = float(np.percentile(scores, params.audio_glue_percentile, method="linear"))
    high = float(np.percentile(scores, params.audio_split_percentile, method="linear"))
    return {
        "informative": (high - low) >= params.audio_min_spread,
        "p_glue": low,
        "p_split": high,
        "median": float(np.median(scores)),
    }


def audio_score_at(
    times: NDArray[np.floating], scores: NDArray[np.floating], moment: float
) -> float | None:
    if times is None or len(times) == 0:
        return None
    nearest = int(np.argmin(np.abs(times - moment)))
    if abs(float(times[nearest]) - moment) > 1.0:
        return None
    return float(scores[nearest])


def vote_audio(score: float | None, band: dict[str, float | bool] | None) -> int:
    if score is None or band is None or not band["informative"]:
        return 0
    if score >= float(band["p_split"]):
        return 1
    if score <= float(band["p_glue"]):
        return -1
    return 0
