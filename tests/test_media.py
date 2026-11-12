"""Public analyze() against synthetic media and unusable files."""

from __future__ import annotations

import shutil
import subprocess
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
    assert result.media.audio_present is False
    assert result.media.subtitles_read is False
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
    assert first.media.audio_present is True
    assert first.media.subtitles_read is False
    assert ".jpg" not in first.to_json()
    assert ".png" not in first.to_json()
    plain = analyze(clip, AnalysisOptions(timeout_s=60))
    assert plain.diagnostics is None

    cues = tmp_path / "cues.srt"
    cues.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{TOKEN}\n", encoding="utf-8")
    muxed = tmp_path / "with subs.mkv"
    _ffmpeg("-i", str(clip), "-i", str(cues), "-map", "0", "-map", "1", "-c", "copy", str(muxed))
    subtitled = analyze(muxed, AnalysisOptions(timeout_s=60))
    assert subtitled.media.subtitle_stream_count >= 1
    assert subtitled.media.subtitles_read is False
    assert TOKEN not in subtitled.to_json()
    assert subtitled.media.filename == "with subs.mkv"


def test_unicode_path_timeout_cancel_and_tool_override(clip: Path, tmp_path: Path) -> None:
    folder = tmp_path / "vidé o"
    folder.mkdir()
    target = folder / "clíp.mkv"
    shutil.copyfile(clip, target)
    found = shutil.which("ffmpeg")
    probe = shutil.which("ffprobe")
    assert found and probe
    result = analyze(target, AnalysisOptions(ffmpeg=found, ffprobe=probe, timeout_s=60))
    assert result.media.filename == "clíp.mkv"
    assert "path" not in result.to_dict()["media"]
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
    assert result.media.filename == "-dash.mp4"
    assert "path" not in result.to_dict()["media"]
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
    assert result.media.audio_present is True
    assert result.media.audio_codec
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
    assert result.media.video_codec == "h264"
    assert result.media.width == 160
    assert result.media.height == 90


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
