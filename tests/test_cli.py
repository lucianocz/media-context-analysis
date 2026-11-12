"""Command-line exit codes. Media failures use a missing path or a monkeypatched analyzer."""

from __future__ import annotations

import tomllib
from pathlib import Path

import media_context_analysis
from media_context_analysis.cli import main
from media_context_analysis.errors import (
    AnalysisCancelledError,
    CorruptMediaError,
    FFmpegFailedError,
)


def test_version_matches_pyproject() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    document = tomllib.loads(project.read_text(encoding="utf-8"))
    assert document["project"]["version"] == media_context_analysis.__version__
    assert document["project"]["license"] == "MIT"


def test_help_and_version_exit_zero(capsys: object) -> None:
    assert main(["--help"]) == 0
    assert main(["--version"]) == 0
    assert main(["--threads", "nope", "clip.mkv"]) == 2
    del capsys


def test_missing_media_and_bad_threads(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.mkv")]) == 3
    assert main(["--threads", "99", str(tmp_path / "missing.mkv")]) == 2
    assert main(["--timeout", "nan", str(tmp_path / "missing.mkv")]) == 2
    assert main(["--timeout", "inf", str(tmp_path / "missing.mkv")]) == 2


def test_output_file_and_missing_parent(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    calls: list[str] = []

    def fake_analyze(path: str, options: object) -> object:
        calls.append(path)
        del options
        return _Payload()

    monkeypatch.setattr("media_context_analysis.cli.analyze", fake_analyze)  # type: ignore[attr-defined]
    destination = tmp_path / "out.json"
    assert main([str(tmp_path / "clip.mkv"), "-o", str(destination), "--diagnostics"]) == 0
    assert destination.read_bytes() == b"{}\n"
    assert calls
    before = len(calls)
    missing = tmp_path / "absent" / "out.json"
    assert main([str(tmp_path / "clip.mkv"), "-o", str(missing)]) == 2
    assert len(calls) == before
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "output directory does not exist" in captured.err


def test_exit_codes_follow_the_exception(monkeypatch: object) -> None:
    def raise_media(path: str, options: object) -> object:
        del path, options
        raise CorruptMediaError("broken")

    monkeypatch.setattr("media_context_analysis.cli.analyze", raise_media)  # type: ignore[attr-defined]
    assert main(["clip.mkv"]) == 3

    def raise_ffmpeg(path: str, options: object) -> object:
        del path, options
        raise FFmpegFailedError("ffmpeg failed")

    monkeypatch.setattr("media_context_analysis.cli.analyze", raise_ffmpeg)  # type: ignore[attr-defined]
    assert main(["clip.mkv"]) == 4

    def raise_cancel(path: str, options: object) -> object:
        del path, options
        raise AnalysisCancelledError("analysis cancelled")

    monkeypatch.setattr("media_context_analysis.cli.analyze", raise_cancel)  # type: ignore[attr-defined]
    assert main(["clip.mkv"]) == 6

    def raise_keyboard(path: str, options: object) -> object:
        del path, options
        raise KeyboardInterrupt

    monkeypatch.setattr("media_context_analysis.cli.analyze", raise_keyboard)  # type: ignore[attr-defined]
    assert main(["clip.mkv"]) == 6

    def raise_other(path: str, options: object) -> object:
        del path, options
        raise RuntimeError("unexpected")

    monkeypatch.setattr("media_context_analysis.cli.analyze", raise_other)  # type: ignore[attr-defined]
    assert main(["clip.mkv"]) == 1


class _Payload:
    def to_json(self) -> str:
        return "{}\n"
