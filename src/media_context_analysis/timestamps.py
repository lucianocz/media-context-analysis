"""Representative timestamps for one detected event.

Detection time is unchanged. The frozen export used the middle of the previous
shot as ``before_s``. That middle is an earlier context reference. The frame a
caller wants immediately before the cut is the last stable sample of the
previous shot. After the cut, the first stable sample is preferred over a
fixed half-second offset that can still sit on a dissolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from media_context_analysis.events import _after_time
from media_context_analysis.shots import Shot
from media_context_analysis.timeutil import as_float, as_int, format_time, round_seconds

_PRE_WINDOW_S = 1.0
_AFTER_WINDOW_S = 1.0


@dataclass(frozen=True)
class RepresentativeChoice:
    pre_transition_s: float
    pre_transition_basis: str
    context_reference_s: float
    context_reference_basis: str
    transition_s: float
    after_s: float
    after_basis: str
    return_s: float | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "pre_transition_s": self.pre_transition_s,
            "pre_transition": format_time(self.pre_transition_s),
            "pre_transition_basis": self.pre_transition_basis,
            "context_reference_s": self.context_reference_s,
            "context_reference": format_time(self.context_reference_s),
            "context_reference_basis": self.context_reference_basis,
            "transition_s": self.transition_s,
            "transition": format_time(self.transition_s),
            "after_s": self.after_s,
            "after": format_time(self.after_s),
            "after_basis": self.after_basis,
            "return_s": self.return_s,
            "return": None if self.return_s is None else format_time(self.return_s),
        }
        return payload


def choose_representatives(
    event: dict[str, object],
    shots: list[Shot],
    frame_times: NDArray[np.floating] | None,
    frame_scores: NDArray[np.floating] | None,
    *,
    sample_fps: float,
    cut_floor: float,
) -> RepresentativeChoice:
    """Pick seek times. ``event["time_s"]`` is copied into ``transition_s``."""
    shot_index = as_int(event["shot_index"])
    current = shots[shot_index]
    previous = shots[shot_index - 1]
    transition = round_seconds(as_float(event["time_s"]))
    context = round_seconds((previous.start_s + previous.end_s) / 2.0)
    pre_s, pre_basis = _pre_transition(
        previous, transition, frame_times, frame_scores, sample_fps=sample_fps, cut_floor=cut_floor
    )
    after_s, after_basis = _after(
        current, transition, frame_times, frame_scores, cut_floor=cut_floor
    )
    return_s = _return_time(event)
    return RepresentativeChoice(
        pre_transition_s=round_seconds(pre_s),
        pre_transition_basis=pre_basis,
        context_reference_s=context,
        context_reference_basis="previous_shot_midpoint",
        transition_s=transition,
        after_s=round_seconds(after_s),
        after_basis=after_basis,
        return_s=None if return_s is None else round_seconds(return_s),
    )


def _return_time(event: dict[str, object]) -> float | None:
    if event.get("category") != "brief_visual_insert":
        return None
    frames = event.get("frames")
    if isinstance(frames, dict) and "return_s" in frames:
        return as_float(frames["return_s"])
    return as_float(event["end_s"])


def _indexed(shot: Shot, frame_times: NDArray[np.floating] | None) -> list[int]:
    if frame_times is None or shot.frame_end <= shot.frame_start or shot.frame_start < 0:
        return []
    return list(range(shot.frame_start, min(shot.frame_end, len(frame_times))))


def _pre_transition(
    previous: Shot,
    cut_s: float,
    frame_times: NDArray[np.floating] | None,
    frame_scores: NDArray[np.floating] | None,
    *,
    sample_fps: float,
    cut_floor: float,
) -> tuple[float, str]:
    if frame_times is None:
        return _estimated_pre(previous, cut_s, sample_fps)
    indexes = [
        index
        for index in _indexed(previous, frame_times)
        if float(frame_times[index]) < cut_s - 1e-4
    ]
    if not indexes:
        return _estimated_pre(previous, cut_s, sample_fps)
    last = indexes[-1]
    if frame_scores is None or float(frame_scores[last]) < cut_floor:
        return float(frame_times[last]), "last_sample_before_cut"
    window = [index for index in indexes if float(frame_times[index]) >= cut_s - _PRE_WINDOW_S]
    stable = [index for index in window if float(frame_scores[index]) < cut_floor]
    if stable:
        return float(frame_times[stable[-1]]), "latest_stable_sample_within_1s"
    if window:
        chosen = min(
            window, key=lambda index: (float(frame_scores[index]), -float(frame_times[index]))
        )
        return float(frame_times[chosen]), "lowest_change_within_1s_before_cut"
    return float(frame_times[last]), "last_sample_before_cut"


def _estimated_pre(previous: Shot, cut_s: float, sample_fps: float) -> tuple[float, str]:
    step = 1.0 / sample_fps if sample_fps > 0 else 0.25
    candidate = min(previous.end_s, cut_s) - step
    if candidate < previous.start_s:
        candidate = previous.start_s
    if candidate >= cut_s:
        candidate = max(previous.start_s, cut_s - 0.001)
    return candidate, "estimated_one_sample_before_cut"


def _after(
    current: Shot,
    cut_s: float,
    frame_times: NDArray[np.floating] | None,
    frame_scores: NDArray[np.floating] | None,
    *,
    cut_floor: float,
) -> tuple[float, str]:
    indexes = [
        index
        for index in _indexed(current, frame_times)
        if frame_times is not None
        and cut_s + 1e-4 < float(frame_times[index]) <= cut_s + _AFTER_WINDOW_S
    ]
    if not indexes or frame_times is None or frame_scores is None:
        return _after_time(cut_s, current.end_s), _legacy_after_basis(cut_s, current.end_s)
    stable = [index for index in indexes if float(frame_scores[index]) < cut_floor]
    if stable:
        return float(frame_times[stable[0]]), "earliest_stable_sample_within_1s"
    chosen = min(indexes, key=lambda index: (float(frame_scores[index]), float(frame_times[index])))
    return float(frame_times[chosen]), "lowest_change_within_1s"


def _legacy_after_basis(start: float, end: float) -> str:
    if end - start <= 1.0:
        return "short_shot_midpoint"
    return "fixed_offset_0_5s"
