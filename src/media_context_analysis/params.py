"""Frozen detector settings.

Levels that can be read from a video's own distributions are computed at run
time. Nothing here is chosen by searching reference annotations.

Subtitle and dialogue fields are part of the frozen parameter set so a run can
be compared with the research export. This library does not read subtitles and
does not apply those fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DetectorParams:
    """A priori settings for one reproducible run."""

    feature_version: int = 1
    sample_fps: float = 4.0
    frame_width: int = 96
    frame_height: int = 54
    hue_bins: int = 16
    sat_bins: int = 4
    val_bins: int = 4
    thumb_x: int = 16
    thumb_y: int = 9
    cut_percentile: float = 90.0
    cut_floor: float = 0.22
    min_shot_s: float = 0.75
    place_thr_percentile: float = 35.0
    place_thr_min: float = 0.48
    place_thr_max: float = 0.72
    shot_thr: float = 0.84
    tau_s: float = 20.0
    brief_insert_s: float = 4.0
    audio_sample_rate: int = 16000
    audio_window_s: float = 0.40
    audio_hop_s: float = 0.20
    audio_change_window_s: float = 5.0
    audio_glue_percentile: float = 40.0
    audio_split_percentile: float = 70.0
    audio_min_spread: float = 0.01
    audio_event_percentile: float = 90.0
    audio_event_mad_scale: float = 3.0
    audio_event_min_separation_s: float = 10.0
    lexical_window_s: float = 20.0
    lexical_glue: float = 0.28
    lexical_glue_gap_s: float = 2.5
    lexical_split_gap_s: float = 5.0
    speaker_window_s: float = 15.0
    speaker_persist_s: float = 12.0
    speaker_persist_cues: int = 2
    speaker_glue_jaccard: float = 0.50
    dialogue_pause_s: float = 6.0
    leading_silence_s: float = 8.0
    return_similarity_floor: float = 0.68
    return_margin: float = 0.05
    ffmpeg_threads: int = 2
    dedupe_s: float = 1.5
    methods: tuple[str, ...] = ("visual", "audio", "subtitle", "combined")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["methods"] = list(self.methods)
        return payload
