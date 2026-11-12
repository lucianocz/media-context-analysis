"""Optional analysis of the gitignored checkout sample."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from media_context_analysis import SCHEMA_VERSION, analyze
from media_context_analysis._version import DETECTOR_BASELINE
from media_context_analysis.options import AnalysisOptions

pytestmark = pytest.mark.sample


def test_checkout_sample_when_present() -> None:
    path = Path(__file__).resolve().parents[1] / "samples" / "test-sample.mkv"
    if not path.is_file():
        pytest.skip("samples/test-sample.mkv is not in this checkout")
    before = {item.name for item in path.parent.iterdir()}
    result = analyze(path, AnalysisOptions(include_diagnostics=True, timeout_s=600))
    after = {item.name for item in path.parent.iterdir()}
    assert after == before
    assert result.schema_version == SCHEMA_VERSION
    assert "media" not in result.to_dict()
    assert not hasattr(result, "media")
    assert result.provenance.detector_baseline == DETECTOR_BASELINE
    assert result.provenance.audio_deletes_events is False
    assert result.diagnostics is not None
    assert result.diagnostics.shot_count == 44
    assert len(result.timeline) == 8
    assert len(result.groups) == 0
    assert [event.time_s for event in result.timeline] == sorted(
        event.time_s for event in result.timeline
    )
    assert all(event.forces_translation_segment is False for event in result.timeline)
    assert all(group.creates_translation_segment is False for group in result.groups)
    payload = result.to_json()
    assert ".jpg" not in payload
    assert ".png" not in payload
    counts = Counter(event.category for event in result.timeline)
    assert dict(counts) == {
        "visual_departure": 2,
        "palette_shift": 4,
        "sharp_stable_cut": 2,
    }
