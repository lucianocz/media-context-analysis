"""Public analyze() against synthetic media and unusable files."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from media_context_analysis import analyze
from media_context_analysis.errors import (
    AnalysisCancelledError,
    AnalysisTimeoutError,
    CorruptMediaError,
    FFmpegNotFoundError,
    InvalidOptionsError,
    MediaNotFoundError,
    UnsupportedMediaError,
)
from media_context_analysis.ffmpeg import extract_visual, media_streams, probe_file, resolve_tool
from media_context_analysis.options import AnalysisOptions
from media_context_analysis.params import DetectorParams

TOKEN = "SecretSubtitleTokenXYZ"


def _require_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")


def _ffmpeg(*args: str) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-2000:]
        raise AssertionError(detail)


def _video(path: Path, *, audio: bool, cut: bool = False) -> None:
    duration = "4" if cut else "2"
    command = ["-f", "lavfi", "-i", "color=c=red:s=160x90:d=2:r=12"]
    if cut:
        command.extend(["-f", "lavfi", "-i", "color=c=blue:s=160x90:d=2:r=12"])
    if audio:
        command.extend(
            ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={duration}"]
        )
    if cut:
        command.extend(["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]"])
        if audio:
            command.extend(["-map", "2:a", "-shortest"])
    elif audio:
        command.append("-shortest")
    else:
        command.append("-an")
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if audio:
        command.extend(["-c:a", "aac", "-b:a", "64k"])
    command.append(str(path))
    _ffmpeg(*command)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_tools()
    path = tmp_path_factory.mktemp("media") / "clip.mkv"
    _video(path, audio=True, cut=True)
    return path


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(timeout_s=0).validate()
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(timeout_s=float("nan")).validate()
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(timeout_s=float("inf")).validate()
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(timeout_s=float("-inf")).validate()
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(timeout_s=True).validate()  # type: ignore[arg-type]
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(video_threads=0).validate()
    with pytest.raises(InvalidOptionsError):
        AnalysisOptions(video_threads=True).validate()  # type: ignore[arg-type]
    AnalysisOptions(timeout_s=None, video_threads=16).validate()


def test_missing_empty_and_directory(tmp_path: Path) -> None:
    with pytest.raises(MediaNotFoundError):
        analyze(tmp_path / "missing.mkv")
    with pytest.raises(MediaNotFoundError):
        analyze(tmp_path)
    empty = tmp_path / "empty.mkv"
    empty.write_bytes(b"")
    with pytest.raises(CorruptMediaError):
        analyze(empty)


def test_corrupt_audio_only_and_silent(tmp_path: Path) -> None:
    _require_tools()
    corrupt = tmp_path / "corrupt.mkv"
    corrupt.write_bytes(b"\x1a\x45\xdf\xa3not-a-real-container")
    with pytest.raises(CorruptMediaError):
        analyze(corrupt)
    audio_only = tmp_path / "tone.m4a"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=1",
        "-c:a",
        "aac",
        str(audio_only),
    )
    with pytest.raises(UnsupportedMediaError):
        analyze(audio_only)
    silent = tmp_path / "silent.mkv"
    _video(silent, audio=False)
    result = analyze(silent, AnalysisOptions(timeout_s=60))
    assert result.provenance.subtitles_used is False
    assert result.timeline is not None
    assert all(event.scores.audio_vote == "unavailable" for event in result.timeline)


def test_analyze_is_deterministic_and_ignores_subtitle_text(clip: Path, tmp_path: Path) -> None:
    options = AnalysisOptions(include_diagnostics=True, video_threads=1, timeout_s=60)
    first = analyze(clip, options)
    second = analyze(clip, options)
    assert first.timeline
    assert first.to_json() == second.to_json()
    assert first.diagnostics is not None
    assert first.provenance.video_threads == 1
    assert first.provenance.audio_threads == 1
    assert first.provenance.detector_params["ffmpeg_threads"] == 2
    assert ".jpg" not in first.to_json()
    assert ".png" not in first.to_json()
    plain = analyze(clip, AnalysisOptions(timeout_s=60))
    assert plain.diagnostics is None

    cues = tmp_path / "cues.srt"
    cues.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{TOKEN}\n", encoding="utf-8")
    muxed = tmp_path / "with subs.mkv"
    _ffmpeg("-i", str(clip), "-i", str(cues), "-map", "0", "-map", "1", "-c", "copy", str(muxed))
    subtitled = analyze(muxed, AnalysisOptions(timeout_s=60))
    assert TOKEN not in subtitled.to_json()
    assert "with subs.mkv" not in subtitled.to_json()
    assert plain.timeline == subtitled.timeline
    assert "media" not in subtitled.to_dict()


def test_unicode_path_timeout_cancel_and_tool_override(clip: Path, tmp_path: Path) -> None:
    folder = tmp_path / "vidĂ© o"
    folder.mkdir()
    target = folder / "clĂ­p.mkv"
    shutil.copyfile(clip, target)
    found = shutil.which("ffmpeg")
    probe = shutil.which("ffprobe")
    assert found and probe
    result = analyze(target, AnalysisOptions(ffmpeg=found, ffprobe=probe, timeout_s=60))
    assert "clĂ­p.mkv" not in result.to_json()
    assert result.provenance.ffmpeg_version
    with pytest.raises(FFmpegNotFoundError):
        analyze(target, AnalysisOptions(ffmpeg="media-context-analysis-missing-tool"))
    import threading

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelledError):
        analyze(target, AnalysisOptions(cancel_event=cancel))
    with pytest.raises(AnalysisTimeoutError):
        analyze(target, AnalysisOptions(timeout_s=0.01))


def test_relative_filename_starting_with_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_tools()
    _video(tmp_path / "-dash.mp4", audio=False)
    monkeypatch.chdir(tmp_path)
    result = analyze(Path("-dash.mp4"), AnalysisOptions(timeout_s=60))
    assert result.timeline is not None


def test_short_audio_stays_unavailable_and_visual_analysis_continues(tmp_path: Path) -> None:
    _require_tools()
    path = tmp_path / "short-audio.mkv"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=160x90:d=2:r=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=0.2",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(path),
    )
    result = analyze(path, AnalysisOptions(timeout_s=60, include_diagnostics=True))
    assert result.timeline is not None
    assert all(event.scores.audio_vote == "unavailable" for event in result.timeline)
    assert result.diagnostics is not None
    assert result.diagnostics.audio_band is None


def test_attached_picture_is_not_selected_as_video() -> None:
    picture = {
        "codec_type": "video",
        "codec_name": "mjpeg",
        "index": 0,
        "disposition": {"attached_pic": 1},
    }
    real = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 160,
        "height": 90,
        "index": 1,
        "disposition": {"attached_pic": 0},
    }
    audio = {"codec_type": "audio", "codec_name": "aac", "index": 2}
    video, selected_audio, subtitles = media_streams({"streams": [picture, real, audio]})
    assert video["index"] == 1
    assert selected_audio is not None and selected_audio["codec_name"] == "aac"
    assert subtitles == 0
    for flag in (1, True, "1"):
        only = {
            "streams": [{"codec_type": "video", "index": 0, "disposition": {"attached_pic": flag}}]
        }
        with pytest.raises(UnsupportedMediaError):
            media_streams(only)


def test_attached_cover_is_not_the_analyzed_picture(tmp_path: Path) -> None:
    _require_tools()
    cover = tmp_path / "cover.jpg"
    picture = tmp_path / "picture.mkv"
    combined = tmp_path / "combined.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=32x32:d=1:r=1",
        "-frames:v",
        "1",
        str(cover),
    )
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x90:d=2:r=12",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(picture),
    )
    _ffmpeg(
        "-i",
        str(picture),
        "-i",
        str(cover),
        "-map",
        "0:v",
        "-map",
        "1:v",
        "-c",
        "copy",
        "-disposition:v:1",
        "attached_pic",
        str(combined),
    )
    probed = probe_file(
        combined,
        resolve_tool(None, "ffprobe"),
        deadline=None,
        cancel_event=None,
    )
    streams = probed["streams"]
    assert isinstance(streams, list)
    assert any(
        isinstance(stream, dict) and (stream.get("disposition") or {}).get("attached_pic") == 1
        for stream in streams
    )
    selected, _audio, _subtitles = media_streams(probed)
    assert selected["codec_name"] == "h264"
    result = analyze(combined, AnalysisOptions(timeout_s=60))
    assert result.provenance.video_stream_index == selected["index"]
    cover_index = next(
        stream["index"]
        for stream in streams
        if isinstance(stream, dict) and (stream.get("disposition") or {}).get("attached_pic") == 1
    )
    with pytest.raises(UnsupportedMediaError, match="not a usable video"):
        analyze(combined, AnalysisOptions(video_stream_index=cover_index))


def test_visual_decode_uses_the_probed_stream_index(tmp_path: Path) -> None:
    _require_tools()
    path = tmp_path / "two.mkv"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=160x90:d=1:r=12",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x90:d=1:r=12",
        "-map",
        "0:v",
        "-map",
        "1:v",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    ffmpeg = resolve_tool(None, "ffmpeg")
    params = DetectorParams()
    red = extract_visual(
        path,
        ffmpeg,
        params,
        {"index": 0},
        video_threads=1,
        deadline=None,
        cancel_event=None,
    )
    blue = extract_visual(
        path,
        ffmpeg,
        params,
        {"index": 1},
        video_threads=1,
        deadline=None,
        cancel_event=None,
    )
    assert red["hists"].shape[0] > 0
    assert blue["hists"].shape[0] > 0
    assert not np.allclose(red["hists"][0], blue["hists"][0])


@pytest.mark.parametrize("index", [-1, True, 1.5, "1"])
def test_invalid_video_stream_options(index: object) -> None:
    with pytest.raises(InvalidOptionsError, match="video_stream_index"):
        AnalysisOptions(video_stream_index=index).validate()  # type: ignore[arg-type]


def test_video_selection_ignores_invalid_streams_and_orders_by_area_then_index() -> None:
    def video(index: int, width: int, height: int) -> dict[str, object]:
        return {"codec_type": "video", "index": index, "width": width, "height": height}

    candidates = [
        video(9, 320, 180),
        video(3, 320, 180),
        video(0, 160, 90),
        video(1, 0, 500),
        video(2, 900, -1),
        video(-1, 900, 900),
        {"codec_type": "video", "index": 10, "width": True, "height": 900},
        {"codec_type": "video", "index": 11, "height": 900},
        {"codec_type": "video", "width": 900, "height": 900},
    ]
    probed: dict[str, object] = {"streams": candidates}
    assert media_streams(probed)[0]["index"] == 3
    assert media_streams(probed, video_stream_index=0)[0]["index"] == 0
    for index in (1, 2, 10, 11, 99):
        with pytest.raises(UnsupportedMediaError, match=f"index {index}"):
            media_streams(probed, video_stream_index=index)
    with pytest.raises(CorruptMediaError, match="no stream list"):
        media_streams({"streams": "invalid"})


@pytest.fixture(scope="module")
def multi_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_tools()
    path = tmp_path_factory.mktemp("multi-video") / "multiple.mkv"
    # FFmpeg's default-disposition stream is small and constant; the larger
    # streams have a visible cut, so public events prove the decode mapping.
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=160x90:d=4:r=12",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x180:d=2:r=12",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x180:d=2:r=12",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x180:d=4:r=12",
        "-filter_complex",
        "[1:v][2:v]concat=n=2:v=1:a=0[cut]",
        "-map",
        "0:v",
        "-map",
        "[cut]",
        "-map",
        "3:v",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-disposition:v:0",
        "default",
        "-disposition:v:1",
        "0",
        "-disposition:v:2",
        "0",
        str(path),
    )
    return path


def test_default_and_explicit_selection_decode_the_effective_video(multi_video: Path) -> None:
    default = analyze(multi_video, AnalysisOptions(timeout_s=60, include_diagnostics=True))
    explicit = analyze(
        multi_video,
        AnalysisOptions(timeout_s=60, include_diagnostics=True, video_stream_index=1),
    )
    assert default.provenance.video_stream_index == 1
    assert default.to_json() == explicit.to_json()
    assert len(default.timeline) == 1
    assert default.timeline[0].time_s == 2.0
    for index in (0, 2):
        selected = analyze(multi_video, AnalysisOptions(timeout_s=60, video_stream_index=index))
        assert selected.provenance.video_stream_index == index
        assert selected.timeline == ()
    with pytest.raises(UnsupportedMediaError, match="index 99"):
        analyze(multi_video, AnalysisOptions(video_stream_index=99))


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_no_success_if_stopped_during_result_assembly(
    clip: Path, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    module = importlib.import_module("media_context_analysis.analyze")
    assemble = module._assemble
    cancel = threading.Event()
    clock = time.monotonic()

    def stopped_assembly(*args: object, **kwargs: object) -> object:
        result = assemble(*args, **kwargs)
        if stop == "cancel":
            cancel.set()
        else:
            monkeypatch.setattr(module.time, "monotonic", lambda: clock + 120)
        return result

    monkeypatch.setattr(module, "_assemble", stopped_assembly)
    error = AnalysisCancelledError if stop == "cancel" else AnalysisTimeoutError
    with pytest.raises(error):
        analyze(clip, AnalysisOptions(timeout_s=60, cancel_event=cancel))
