"""JSON contract for the public result types."""

from __future__ import annotations

import json

import pytest

from media_context_analysis.models import (
    AnalysisResult,
    Continuity,
    Diagnostics,
    EventGroup,
    EventScores,
    Provenance,
    Representatives,
    VisualEvent,
)

TOP_KEYS = ["schema_version", "provenance", "timeline", "groups", "diagnostics"]


def _event(*, category: str = "sharp_stable_cut", return_time: str | None = None) -> VisualEvent:
    return VisualEvent(
        id="e0001",
        shot_index=1,
        time_s=1.0,
        time="00:00:01.000",
        end_s=2.0,
        end="00:00:02.000",
        category=category,
        priority="high",
        reasons=("sharp_stable_cut",),
        selected=True,
        suggested_context_break=False,
        forces_translation_segment=False,
        uncertainty="stable",
        evidence=("shot-start change 0.500",),
        continuity=Continuity(
            framing_match=False,
            color_match=False,
            acoustic_vote="unavailable",
            short_return=category == "brief_visual_insert",
            notes=("no acoustic evidence",),
        ),
        scores=EventScores(
            best_place_similarity=0.5,
            best_framing_similarity=0.25,
            adjacent_place_similarity=0.5,
            change_score=0.5,
            luma=0.4,
            luma_step=0.1,
            shot_duration_s=1.0,
            place_match_age_s=None,
            audio_score=None,
            audio_vote="unavailable",
            near_black=False,
        ),
        representatives=Representatives(
            pre_transition_s=0.75,
            pre_transition="00:00:00.750",
            pre_transition_basis="last_sample_before_cut",
            context_reference_s=0.5,
            context_reference="00:00:00.500",
            context_reference_basis="previous_shot_midpoint",
            transition_s=1.0,
            transition="00:00:01.000",
            after_s=1.25,
            after="00:00:01.250",
            after_basis="earliest_stable_sample_within_1s",
            return_s=None if return_time is None else 2.0,
            return_time=return_time,
        ),
    )


def _result(diagnostics: Diagnostics | None = None) -> AnalysisResult:
    return AnalysisResult(
        schema_version="2.0.0",
        provenance=Provenance(
            library_name="media-context-analysis",
            library_version="0.2.0",
            detector_baseline="68cc838a008ad279752bcf598e2039fcabac8be0",
            detector_note="note",
            feature_version=1,
            subtitles_used=False,
            audio_deletes_events=False,
            ffmpeg_version=None,
            ffprobe_version=None,
            video_stream_index=0,
            video_threads=2,
            audio_threads=1,
            visual_filter="fps=4.0,scale=96:54:flags=area",
            detector_params={"ffmpeg_threads": 2, "methods": ["visual"]},
        ),
        timeline=(_event(),),
        groups=(
            EventGroup(
                id="g0001",
                kind="brief_insert_return",
                event_ids=("e0001", "e0002"),
                evidence=("return lines up",),
                creates_translation_segment=False,
            ),
        ),
        diagnostics=diagnostics,
    )


def test_json_key_order_round_trip_and_same_major_version() -> None:
    result = _result()
    payload = result.to_dict()
    assert list(payload) == TOP_KEYS
    text = result.to_json()
    assert text.endswith("\n")
    assert json.loads(text)["schema_version"] == "2.0.0"
    parsed = AnalysisResult.from_json(text)
    assert parsed.to_dict() == payload
    again = dict(payload)
    again["schema_version"] = "2.9.0"
    again["unused"] = {"ignored": True}
    loaded = AnalysisResult.from_dict(again)
    assert loaded.schema_version == "2.9.0"
    assert loaded.to_json() == AnalysisResult.from_json(loaded.to_json()).to_json()
    representatives = parsed.timeline[0].representatives.to_dict()
    assert list(representatives)[-1] == "return"
    assert "return_time" not in representatives
    assert not hasattr(parsed, "media")
    assert "media" not in payload


def test_other_schema_major_and_forced_flags_are_rejected() -> None:
    payload = _result().to_dict()
    payload["schema_version"] = "1.0.0"
    with pytest.raises(ValueError, match="unsupported schema"):
        AnalysisResult.from_dict(payload)
    payload = _result().to_dict()
    timeline = payload["timeline"]
    assert isinstance(timeline, list)
    timeline[0]["forces_translation_segment"] = True
    with pytest.raises(ValueError, match="forces_translation_segment"):
        AnalysisResult.from_dict(payload)
    payload = _result().to_dict()
    groups = payload["groups"]
    assert isinstance(groups, list)
    groups[0]["creates_translation_segment"] = True
    with pytest.raises(ValueError, match="creates_translation_segment"):
        AnalysisResult.from_dict(payload)


def test_non_object_timeline_items_are_rejected() -> None:
    payload = _result().to_dict()
    timeline = payload["timeline"]
    assert isinstance(timeline, list)
    timeline.append("nope")
    with pytest.raises(ValueError, match="timeline items"):
        AnalysisResult.from_dict(payload)


def test_diagnostics_round_trip() -> None:
    raw = _event(category="palette_break")
    diagnostics = Diagnostics(
        shot_count=2,
        place_threshold=0.48,
        place_threshold_stats={"clipped": True},
        thresholds={"sharp_change": 0.5},
        audio_band=None,
        rules={"subtitles_used": False},
        shot_stats={"count": 2},
        family_counts={"palette_break": 1},
        family_times={"palette_break": [1.0]},
        compact_before_dedupe=1,
        raw_candidates=(raw,),
    )
    text = _result(diagnostics).to_json()
    loaded = AnalysisResult.from_json(text)
    assert loaded.diagnostics is not None
    assert loaded.diagnostics.raw_candidates[0].category == "palette_break"
    assert loaded.diagnostics.to_dict()["audio_band"] is None
    exported = loaded.to_dict()
    assert isinstance(exported["provenance"], dict)
    params = exported["provenance"]["detector_params"]
    assert isinstance(params, dict)
    params["sample_fps"] = 99
    assert loaded.provenance.detector_params.get("sample_fps") != 99


def test_non_finite_numbers_are_rejected() -> None:
    payload = _result().to_dict()
    timeline = payload["timeline"]
    assert isinstance(timeline, list)
    timeline[0]["time_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        AnalysisResult.from_dict(payload)
    with pytest.raises(ValueError, match="invalid JSON number"):
        AnalysisResult.from_json('{"schema_version": "2.0.0", "time_s": NaN}\n')
    counted = _result(
        Diagnostics(
            shot_count=1,
            place_threshold=0.5,
            place_threshold_stats={},
            thresholds={},
            audio_band=None,
            rules={},
            shot_stats={},
            family_counts={"palette_break": 1},
            family_times={"palette_break": [1.0]},
            compact_before_dedupe=0,
            raw_candidates=(),
        )
    ).to_dict()
    diagnostics = counted["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["family_times"] = {"palette_break": [float("inf")]}
    with pytest.raises(ValueError, match="family_times"):
        AnalysisResult.from_dict(counted)
    diagnostics["family_counts"] = {"palette_break": True}
    with pytest.raises(ValueError, match="family_counts"):
        AnalysisResult.from_dict(counted)


def test_schema_version_must_be_numeric_triple() -> None:
    payload = _result().to_dict()
    for version in (
        "2",
        "2.0",
        "2.0.0-garbage",
        "2.x.0",
        "3.0.0",
        "02.0.0",
        "2.01.0",
        "2.0.00",
        "2.0.0\n",
        "2.\u0660.0",
    ):
        payload["schema_version"] = version
        with pytest.raises(ValueError, match="unsupported schema"):
            AnalysisResult.from_dict(payload)


def test_null_diagnostic_counts_are_value_errors() -> None:
    payload = _result(
        Diagnostics(
            shot_count=1,
            place_threshold=0.5,
            place_threshold_stats={},
            thresholds={},
            audio_band=None,
            rules={},
            shot_stats={},
            family_counts={"palette_break": 1},
            family_times={"palette_break": [1.0]},
            compact_before_dedupe=0,
            raw_candidates=(),
        )
    ).to_dict()
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["family_counts"] = {"palette_break": None}
    with pytest.raises(ValueError, match="family_counts"):
        AnalysisResult.from_dict(payload)


@pytest.mark.parametrize("index", [None, True, -1, 1.5, "1"])
def test_effective_stream_index_is_required_and_non_negative(index: object) -> None:
    payload = _result().to_dict()
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    provenance["video_stream_index"] = index
    with pytest.raises(ValueError, match="video_stream_index"):
        AnalysisResult.from_dict(payload)


def test_cli_serializes_canonical_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from media_context_analysis.cli import main

    result = _result()
    monkeypatch.setattr("media_context_analysis.cli.analyze", lambda path, options: result)
    assert main(["clip.mkv"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == TOP_KEYS
    assert payload["schema_version"] == "2.0.0"
    assert payload["provenance"]["video_stream_index"] == 0
