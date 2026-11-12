"""Visual and acoustic events.

Three layers stay separate:

1. Candidates. A place break, an unlinked departure, a short return, or a
   sharp cut into a shot that holds. Context grouping does not delete these.
2. Continuity. Color, framing, a short return, and the acoustic bed are
   attached as evidence. A likely continuation does not remove the event.
3. Selection. The compact timeline keeps the stronger local events. The raw
   list keeps every place break for diagnostics.

No event forces a translation segment. Subtitles are not read. Reference
annotations are not read. Character, place, emotion, and flashback labels
are not assigned.

Adapted from narrative-context-segmentation ``agent_events.py`` at commit
68cc838a008ad279752bcf598e2039fcabac8be0. Representative timestamps are chosen
later and do not change ``time_s``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from media_context_analysis.params import DetectorParams
from media_context_analysis.shots import (
    Shot,
    audio_score_at,
    link_against,
    percentile_band,
    visual_candidates,
    vote_audio,
)
from media_context_analysis.signal import intersection
from media_context_analysis.timeutil import as_float, as_int, format_time, round_seconds

# Same drop the frozen in-context shift uses. It is not a new fitted threshold.
DIP_DROP = 0.15
# Same luminance note the frozen candidate list already records.
NEAR_BLACK_LUMA = 0.08

_RANK = {
    "brief_visual_insert": 0,
    "visual_departure": 1,
    "sharp_stable_cut": 2,
    "palette_shift": 3,
}


def detect_events(
    shots: list[Shot],
    place_thr: float,
    params: DetectorParams | None = None,
    audio_times: NDArray[np.floating] | None = None,
    audio_scores: NDArray[np.floating] | None = None,
) -> dict[str, object]:
    """Build raw candidates and the compact timeline from shots already cut."""
    params = params or DetectorParams()
    observations = _observe(shots, place_thr, params)
    band = percentile_band(audio_scores if audio_scores is not None else np.asarray([]), params)
    sharp_threshold, adjacent_median, luma_threshold = _distribution_thresholds(observations)
    insert_at = _insert_index(shots, place_thr, params)
    dip_indexes = {
        as_int(row["shot_index"])
        for row in observations
        if row["palette_break"] and _local_dip(observations, as_int(row["shot_index"]))
    }
    raw_rows: list[dict[str, object]] = []
    compact_rows: list[dict[str, object]] = []
    for row in observations[1:]:
        reasons = _reasons(row, sharp_threshold, adjacent_median, insert_at, dip_indexes, params)
        if "palette_break" not in reasons and "sharp_stable_cut" not in reasons:
            continue
        event = _event_from_row(
            row, shots, reasons, place_thr, params, band, audio_times, audio_scores, insert_at
        )
        raw_rows.append(event)
        if event["compact_candidate"]:
            compact_rows.append(event)
    timeline = merge_timeline(compact_rows, params.dedupe_s)
    kept = {as_int(event["shot_index"]) for event in timeline}
    for event in raw_rows:
        event["kept"] = as_int(event["shot_index"]) in kept
    _assign_ids(raw_rows, "r")
    _assign_ids(timeline, "e")
    families = _family_times(
        observations,
        sharp_threshold,
        adjacent_median,
        luma_threshold,
        insert_at,
        dip_indexes,
        timeline,
        params,
    )
    return {
        "rules": _rules(place_thr, params),
        "audio_band": _band_record(band),
        "thresholds": {
            "sharp_change": None if sharp_threshold is None else round(sharp_threshold, 4),
            "adjacent_place_median": None if adjacent_median is None else round(adjacent_median, 4),
            "luma_step": None if luma_threshold is None else round(luma_threshold, 4),
        },
        "shot_count": len(shots),
        "raw_candidates": raw_rows,
        "timeline": timeline,
        "diagnostics": {
            "family_counts": {name: len(times) for name, times in families.items()},
            "family_times": families,
            "compact_before_dedupe": len(compact_rows),
            "compact_after_dedupe": len(timeline),
        },
    }


def merge_timeline(events: list[dict[str, object]], dedupe_s: float) -> list[dict[str, object]]:
    """Keep one event per window of ``dedupe_s``, preferring the stronger type.

    The window is measured from the first event in the group, so a chain of
    cuts does not collapse into a single event.
    """
    if not events:
        return []
    ordered = sorted(
        events, key=lambda item: (as_float(item["time_s"]), as_int(item["shot_index"]))
    )
    groups: list[list[dict[str, object]]] = [[ordered[0]]]
    for event in ordered[1:]:
        if as_float(event["time_s"]) - as_float(groups[-1][0]["time_s"]) <= dedupe_s:
            groups[-1].append(event)
        else:
            groups.append([event])
    kept: list[dict[str, object]] = []
    for group in groups:
        winner = min(
            group,
            key=lambda item: (
                _RANK[str(item["category"])],
                as_float(_scores(item)["best_place_similarity"]),
                as_float(item["time_s"]),
                as_int(item["shot_index"]),
            ),
        )
        kept.append(winner)
    kept.sort(key=lambda item: (as_float(item["time_s"]), as_int(item["shot_index"])))
    return kept


def _scores(event: dict[str, object]) -> dict[str, object]:
    scores = event["scores"]
    if not isinstance(scores, dict):
        raise TypeError("event scores must be a dict")
    return scores


def _rules(place_thr: float, params: DetectorParams) -> dict[str, object]:
    return {
        "dip_drop": DIP_DROP,
        "near_black_luma": NEAR_BLACK_LUMA,
        "sharp_change_percentile": params.cut_percentile,
        "sharp_min_duration_s": params.brief_insert_s,
        "sharp_adjacent_comparison": "below the median adjacent-shot color intersection",
        "dedupe_s": params.dedupe_s,
        "place_threshold": round(place_thr, 4),
        "shot_threshold": params.shot_thr,
        "tau_s": params.tau_s,
        "lookback_dip_excludes_neighbors_within_s": None,
        "subtitles_used": False,
        "audio_deletes_events": False,
        "forces_translation_segment": False,
    }


def _band_record(band: dict[str, float | bool] | None) -> dict[str, object] | None:
    if band is None:
        return None
    return {
        "informative": bool(band["informative"]),
        "p_glue": round(as_float(band["p_glue"]), 4),
        "p_split": round(as_float(band["p_split"]), 4),
        "median": round(as_float(band["median"]), 4),
    }


def _observe(
    shots: list[Shot], place_thr: float, params: DetectorParams
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, shot in enumerate(shots):
        duration = shot.end_s - shot.start_s
        if index == 0:
            rows.append(_blank_row(shot, duration))
            continue
        linked, best_place, best_shot = link_against(
            shots, index, index, place_thr, params.shot_thr, params.tau_s
        )
        match_at = _best_place_time(shots, index, params.tau_s)
        previous = shots[index - 1]
        rows.append(
            {
                "shot_index": index,
                "time_s": shot.start_s,
                "end_s": shot.end_s,
                "duration_s": duration,
                "luma": shot.luma,
                "change_score": shot.change_score,
                "adjacent_place": intersection(shot.hist, previous.hist),
                "luma_step": abs(shot.luma - previous.luma),
                "linked": linked,
                "best_place": best_place,
                "best_shot": best_shot,
                "match_at_s": match_at,
                "place_match_age_s": None if match_at is None else shot.start_s - match_at,
                "palette_break": best_place < place_thr,
            }
        )
    return rows


def _blank_row(shot: Shot, duration: float) -> dict[str, object]:
    return {
        "shot_index": 0,
        "time_s": shot.start_s,
        "end_s": shot.end_s,
        "duration_s": duration,
        "luma": shot.luma,
        "change_score": shot.change_score,
        "adjacent_place": 1.0,
        "luma_step": 0.0,
        "linked": True,
        "best_place": 1.0,
        "best_shot": 1.0,
        "match_at_s": shot.start_s,
        "place_match_age_s": 0.0,
        "palette_break": False,
    }


def _best_place_time(shots: list[Shot], query: int, tau_s: float) -> float | None:
    best = -1.0
    match_at = None
    start = shots[query].start_s
    for earlier in range(query - 1, -1, -1):
        if start - shots[earlier].start_s > tau_s:
            break
        place = intersection(shots[query].hist, shots[earlier].hist)
        if place > best:
            best = place
            match_at = shots[earlier].start_s
    if best <= 0.0:
        return None
    return match_at


def _distribution_thresholds(
    observations: list[dict[str, object]],
) -> tuple[float | None, float | None, float | None]:
    body = observations[1:]
    if len(body) < 2:
        return None, None, None
    changes = np.asarray([as_float(row["change_score"]) for row in body], dtype=np.float64)
    adjacent = np.asarray([as_float(row["adjacent_place"]) for row in body], dtype=np.float64)
    steps = np.asarray([as_float(row["luma_step"]) for row in body], dtype=np.float64)
    return (
        as_float(np.percentile(changes, 90, method="linear")),
        as_float(np.median(adjacent)),
        as_float(np.percentile(steps, 90, method="linear")),
    )


def _insert_index(shots: list[Shot], place_thr: float, params: DetectorParams) -> dict[int, float]:
    _boundaries, inserts = visual_candidates(shots, place_thr, params)
    by_time = {as_float(item["time_s"]): as_float(item["end_s"]) for item in inserts}
    return {
        index: by_time[shot.start_s] for index, shot in enumerate(shots) if shot.start_s in by_time
    }


def _local_dip(observations: list[dict[str, object]], index: int) -> bool:
    """True when look-back color is a local minimum and at least DIP_DROP down.

    The frozen in-context shift also skips an event within 10 s of another
    and within 6 s of a segment boundary. Those vetoes are intentionally
    absent here: they hide a second visual event inside one interaction.
    """
    if index <= 0:
        return False
    places = [as_float(row["best_place"]) for row in observations]
    baseline = as_float(np.median(places[max(0, index - 4) : index]))
    if places[index] > baseline - DIP_DROP:
        return False
    if places[index] > places[index - 1]:
        return False
    if index < len(places) - 1 and places[index] > places[index + 1]:
        return False
    return True


def _reasons(
    row: dict[str, object],
    sharp_threshold: float | None,
    adjacent_median: float | None,
    insert_at: dict[int, float],
    dip_indexes: set[int],
    params: DetectorParams,
) -> list[str]:
    reasons = []
    if row["palette_break"]:
        reasons.append("palette_break")
    if not row["linked"]:
        reasons.append("unlinked")
    if as_int(row["shot_index"]) in insert_at:
        reasons.append("brief_return")
    if _is_sharp(row, sharp_threshold, adjacent_median, params.brief_insert_s):
        reasons.append("sharp_stable_cut")
    if as_int(row["shot_index"]) in dip_indexes:
        reasons.append("local_dip")
    return reasons


def _is_sharp(
    row: dict[str, object],
    sharp_threshold: float | None,
    adjacent_median: float | None,
    min_duration_s: float,
) -> bool:
    if sharp_threshold is None or adjacent_median is None:
        return False
    return (
        as_float(row["change_score"]) >= sharp_threshold
        and as_float(row["adjacent_place"]) < adjacent_median
        and as_float(row["duration_s"]) >= min_duration_s
    )


def _is_luma_step(
    row: dict[str, object], luma_threshold: float | None, min_duration_s: float
) -> bool:
    if luma_threshold is None:
        return False
    return (
        as_float(row["luma_step"]) >= luma_threshold
        and as_float(row["duration_s"]) >= min_duration_s
    )


def _primary_category(reasons: list[str]) -> str | None:
    if "brief_return" in reasons:
        return "brief_visual_insert"
    if "unlinked" in reasons:
        return "visual_departure"
    if "sharp_stable_cut" in reasons:
        return "sharp_stable_cut"
    if "local_dip" in reasons:
        return "palette_shift"
    return None


def _event_from_row(
    row: dict[str, object],
    shots: list[Shot],
    reasons: list[str],
    place_thr: float,
    params: DetectorParams,
    band: dict[str, float | bool] | None,
    audio_times: NDArray[np.floating] | None,
    audio_scores: NDArray[np.floating] | None,
    insert_at: dict[int, float],
) -> dict[str, object]:
    audio_score, audio_vote = _audio_at(as_float(row["time_s"]), band, audio_times, audio_scores)
    category = _primary_category(reasons)
    shot_index = as_int(row["shot_index"])
    end_s = insert_at.get(shot_index, as_float(row["end_s"]))
    near_black = as_float(row["luma"]) < NEAR_BLACK_LUMA
    continuity = _continuity(row, place_thr, params, audio_score, audio_vote, band, category, end_s)
    suggested = category == "visual_departure" and audio_vote != "glue"
    previous = shots[shot_index - 1]
    legacy_before = round_seconds((previous.start_s + previous.end_s) / 2.0)
    transition = round_seconds(as_float(row["time_s"]))
    legacy_after = round_seconds(_after_time(as_float(row["time_s"]), as_float(row["end_s"])))
    frames: dict[str, object] = {
        "before_s": legacy_before,
        "transition_s": transition,
        "after_s": legacy_after,
    }
    for key, value in list(frames.items()):
        if isinstance(value, float):
            frames[key.replace("_s", "")] = format_time(value)
    if category == "brief_visual_insert":
        frames["return_s"] = round_seconds(end_s)
        frames["return"] = format_time(end_s)
    if category is None:
        priority = "low"
    elif category == "palette_shift":
        priority = "medium"
    else:
        priority = "high"
    age = row["place_match_age_s"]
    return {
        "shot_index": shot_index,
        "time_s": round_seconds(as_float(row["time_s"])),
        "end_s": round_seconds(end_s),
        "time": format_time(as_float(row["time_s"])),
        "end": format_time(end_s),
        "category": category if category is not None else "palette_break",
        "reasons": list(reasons),
        "priority": priority,
        "compact_candidate": category is not None,
        "kept": False,
        "forces_translation_segment": False,
        "suggested_context_break": suggested,
        "uncertainty": _uncertainty(category, near_black, audio_vote),
        "continuity": continuity,
        "scores": {
            "best_place_similarity": as_float(row["best_place"]),
            "best_framing_similarity": as_float(row["best_shot"]),
            "adjacent_place_similarity": as_float(row["adjacent_place"]),
            "change_score": as_float(row["change_score"]),
            "luma": as_float(row["luma"]),
            "luma_step": as_float(row["luma_step"]),
            "shot_duration_s": as_float(row["duration_s"]),
            "place_match_age_s": None if age is None else as_float(age),
            "audio_score": audio_score,
            "audio_vote": audio_vote,
            "near_black": near_black,
        },
        "frames": frames,
        "evidence": _evidence(
            row, reasons, place_thr, params, audio_score, audio_vote, category, end_s
        ),
    }


def _audio_at(
    moment: float,
    band: dict[str, float | bool] | None,
    times: NDArray[np.floating] | None,
    scores: NDArray[np.floating] | None,
) -> tuple[float | None, str]:
    if times is None or scores is None or band is None or not band["informative"]:
        return None, "unavailable"
    score = audio_score_at(times, scores, moment)
    if score is None:
        return None, "unavailable"
    vote = vote_audio(score, band)
    return score, {1: "split", -1: "glue", 0: "neutral"}[vote]


def _continuity(
    row: dict[str, object],
    place_thr: float,
    params: DetectorParams,
    audio_score: float | None,
    audio_vote: str,
    band: dict[str, float | bool] | None,
    category: str | None,
    end_s: float,
) -> list[str]:
    notes = []
    if as_float(row["best_shot"]) >= params.shot_thr:
        notes.append(
            f"framing cosine {as_float(row['best_shot']):.3f} matches a shot in the "
            f"{params.tau_s:.0f}s look-back"
        )
    age = row["place_match_age_s"]
    if as_float(row["best_place"]) >= place_thr and age is not None:
        notes.append(
            f"color intersection {as_float(row['best_place']):.3f} matches a shot "
            f"{as_float(age):.1f}s earlier"
        )
    if audio_vote == "glue" and audio_score is not None and band is not None:
        notes.append(
            f"acoustic change {audio_score:.3f} is at or below this file's glue level "
            f"{as_float(band['p_glue']):.3f}"
        )
    if category == "brief_visual_insert":
        notes.append(
            f"a shot {end_s - as_float(row['time_s']):.2f}s later matches the look from before "
            "this departure"
        )
    if not notes:
        notes.append("no color, framing, return, or acoustic evidence of continuation")
    return notes


def _uncertainty(category: str | None, near_black: bool, audio_vote: str) -> str:
    if category == "brief_visual_insert":
        text = (
            "Short visual departure that returns to the previous look. "
            "No semantic label is assigned."
        )
    elif category == "visual_departure" and audio_vote == "glue":
        text = "The picture matches nothing in the look-back, but the acoustic bed continues."
    elif category == "visual_departure" and audio_vote == "split":
        text = (
            "The picture matches nothing in the look-back, and the acoustic bed changes. "
            "This does not open a translation segment."
        )
    elif category == "visual_departure":
        text = (
            "The picture matches nothing in the look-back. Audio does not confirm a continuation."
        )
    elif category == "sharp_stable_cut":
        text = (
            "Hard cut into a shot that holds. Color or framing still matches an earlier shot, "
            "so the change may sit inside one situation."
        )
    elif category == "palette_shift":
        text = (
            "Color shifted below the place threshold and the shift is a local minimum. "
            "The surrounding material may continue."
        )
    else:
        text = "Color differs from the look-back window. Not selected for the compact timeline."
    if near_black:
        text += " Mean luminance is very low, which can be a card, a fade, or a dark shot."
    return text


def _evidence(
    row: dict[str, object],
    reasons: list[str],
    place_thr: float,
    params: DetectorParams,
    audio_score: float | None,
    audio_vote: str,
    category: str | None,
    end_s: float,
) -> list[str]:
    lines = [
        (
            f"best look-back color intersection {as_float(row['best_place']):.3f} "
            f"(place link at {place_thr:.3f})"
        ),
        (
            f"best look-back framing cosine {as_float(row['best_shot']):.3f} "
            f"(framing link at {params.shot_thr:.3f})"
        ),
        f"adjacent color intersection {as_float(row['adjacent_place']):.3f}",
        f"shot-start change {as_float(row['change_score']):.3f}",
        f"new shot lasts {as_float(row['duration_s']):.2f}s",
        f"mean luminance {as_float(row['luma']):.3f}",
    ]
    age = row["place_match_age_s"]
    if age is not None:
        lines.append(f"closest color match is {as_float(age):.1f}s earlier")
    if audio_score is None:
        lines.append("acoustic change unavailable")
    else:
        lines.append(f"acoustic change {audio_score:.3f} ({audio_vote})")
    if "local_dip" in reasons:
        lines.append(
            "color intersection is a local minimum and at least "
            f"{DIP_DROP:.2f} below the recent median"
        )
    if "sharp_stable_cut" in reasons:
        lines.append(
            "shot-start change is in the top decile, the previous shot is below the median "
            "color match, "
            "and the new shot holds"
        )
    if category == "brief_visual_insert":
        lines.append(f"return at {format_time(end_s)}")
    if as_float(row["luma"]) < NEAR_BLACK_LUMA:
        lines.append("mean luminance is below 0.08")
    lines.append("this event does not open a translation segment")
    return lines


def _after_time(start: float, end: float) -> float:
    """Frozen after-frame offset. Representative selection may replace it later."""
    duration = end - start
    if duration <= 1.0:
        return (start + end) / 2.0
    return min(start + 0.5, end - 0.05)


def _assign_ids(events: list[dict[str, object]], prefix: str) -> None:
    for number, event in enumerate(events, start=1):
        event["id"] = f"{prefix}{number:04d}"


def _family_times(
    observations: list[dict[str, object]],
    sharp_threshold: float | None,
    adjacent_median: float | None,
    luma_threshold: float | None,
    insert_at: dict[int, float],
    dip_indexes: set[int],
    timeline: list[dict[str, object]],
    params: DetectorParams,
) -> dict[str, list[float]]:
    body = observations[1:]
    palette = [row for row in body if row["palette_break"]]
    dips = [row for row in palette if as_int(row["shot_index"]) in dip_indexes]
    any_dips = [row for row in body if _local_dip(observations, as_int(row["shot_index"]))]
    sharp = [
        row
        for row in body
        if _is_sharp(row, sharp_threshold, adjacent_median, params.brief_insert_s)
    ]
    luma = [row for row in body if _is_luma_step(row, luma_threshold, params.brief_insert_s)]
    unlinked = [row for row in body if not row["linked"]]
    return {
        "palette": [_time(row) for row in palette],
        "local_dip": [_time(row) for row in dips],
        "local_dip_without_place_gate": [_time(row) for row in any_dips],
        "sharp_stable_cut": [_time(row) for row in sharp],
        "luma_step": [_time(row) for row in luma],
        "unlinked": [_time(row) for row in unlinked],
        "brief_insert": [_time(observations[index]) for index in sorted(insert_at)],
        "gap8_lowest_place": [_time(row) for row in _cluster_lowest(palette, 8.0)],
        "dip_run_onset_within_4s": _run_onsets(observations, dip_indexes),
        "compact": [round_seconds(as_float(event["time_s"])) for event in timeline],
    }


def _time(row: dict[str, object]) -> float:
    return round_seconds(as_float(row["time_s"]))


def _cluster_lowest(rows: list[dict[str, object]], gap_s: float) -> list[dict[str, object]]:
    if not rows:
        return []
    groups: list[list[dict[str, object]]] = [[rows[0]]]
    for row in rows[1:]:
        if as_float(row["time_s"]) - as_float(groups[-1][-1]["time_s"]) <= gap_s:
            groups[-1].append(row)
        else:
            groups.append([row])
    return [
        min(group, key=lambda item: (as_float(item["best_place"]), as_float(item["time_s"])))
        for group in groups
    ]


def _run_onsets(observations: list[dict[str, object]], dip_indexes: set[int]) -> list[float]:
    """Start of a place-break run when a deeper dip follows within 4 s.

    Diagnostic only. Using the onset instead of the dip moves the timestamp.
    """
    onsets = []
    index = 1
    while index < len(observations):
        if not observations[index]["palette_break"]:
            index += 1
            continue
        end = index
        while end + 1 < len(observations) and observations[end + 1]["palette_break"]:
            end += 1
        run = observations[index : end + 1]
        run_dips = [
            as_float(row["time_s"]) for row in run if as_int(row["shot_index"]) in dip_indexes
        ]
        if run_dips:
            gap = min(run_dips) - as_float(run[0]["time_s"])
            if 1.5 < gap <= 4.0:
                onsets.append(round_seconds(as_float(run[0]["time_s"])))
        index = end + 1
    return onsets
