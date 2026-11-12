"""Subprocess limits."""

from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from media_context_analysis.errors import (
    AnalysisCancelledError,
    AnalysisTimeoutError,
    FFmpegNotFoundError,
)
from media_context_analysis.ffmpeg import (
    process_alive,
    resolve_tool,
    run_bounded,
    stream_chunks,
    stream_fixed_frames,
)


def test_missing_executable_raises() -> None:
    with pytest.raises(FFmpegNotFoundError):
        resolve_tool("media-context-analysis-missing-tool", "ffmpeg")
    with pytest.raises(FFmpegNotFoundError):
        run_bounded(["media-context-analysis-missing-tool"], deadline=None, cancel_event=None)


def test_timeout_kills_the_process_quickly() -> None:
    started = time.monotonic()
    with pytest.raises(AnalysisTimeoutError):
        run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            deadline=None,
            cancel_event=None,
            timeout_s=0.3,
        )
    assert time.monotonic() - started < 5


def test_cancel_event_stops_a_running_process() -> None:
    cancel = threading.Event()

    def trip() -> None:
        time.sleep(0.2)
        cancel.set()

    threading.Thread(target=trip, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(AnalysisCancelledError):
        run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            deadline=None,
            cancel_event=cancel,
        )
    assert time.monotonic() - started < 5


def test_cancel_event_checked_before_start() -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelledError):
        run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            deadline=None,
            cancel_event=cancel,
        )


def test_process_alive_does_not_signal_a_live_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert process_alive(process.pid)
        assert process.poll() is None
        pid = process.pid
    finally:
        process.kill()
        process.wait(timeout=5)
    assert process_alive(pid) is False
    assert process_alive(-1) is False


_FRAME_BYTES = 96 * 54 * 3


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        pytest.skip("ffmpeg is required")
    return found


def _ffmpeg_pids() -> set[int]:
    if sys.platform == "win32":
        completed = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            check=False,
        )
        text = completed.stdout.decode("utf-8", "replace")
        pids: set[int] = set()
        for row in csv.reader(text.splitlines()):
            if len(row) >= 2 and row[0].lower() == "ffmpeg.exe":
                try:
                    pids.add(int(row[1]))
                except ValueError:
                    continue
        return pids
    completed = subprocess.run(
        ["ps", "-C", "ffmpeg", "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return {int(line) for line in completed.stdout.split() if line.isdigit()}


def _kill_pids(pids: set[int]) -> None:
    for pid in pids:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _visual_command(ffmpeg: str, duration: str) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s=96x54:d={duration}:r=25",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def _audio_command(ffmpeg: str) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=8",
        "-f",
        "f32le",
        "-",
    ]


def _pause(payload: bytes) -> None:
    del payload
    time.sleep(0.02)


def _call_in_thread(action: object, limit_s: float) -> BaseException | None:
    """Run ``action`` on a daemon thread and kill any ffmpeg it leaves behind."""
    box: dict[str, BaseException | None] = {}
    before = _ffmpeg_pids()

    def run() -> None:
        try:
            action()  # type: ignore[operator]
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(limit_s)
    leaked = _ffmpeg_pids() - before
    if thread.is_alive() or leaked:
        _kill_pids(leaked)
        thread.join(2.0)
    assert not thread.is_alive()
    assert _ffmpeg_pids() - before == set()
    return box.get("error")


def test_slow_visual_callback_times_out_without_a_leftover_process() -> None:
    ffmpeg = _ffmpeg()
    seen = 0
    started = time.monotonic()

    def on_frame(buffer: bytes) -> None:
        nonlocal seen
        del buffer
        seen += 1
        time.sleep(0.02)

    def action() -> None:
        stream_fixed_frames(
            _visual_command(ffmpeg, "8"),
            frame_bytes=_FRAME_BYTES,
            on_frame=on_frame,
            deadline=time.monotonic() + 0.5,
            cancel_event=None,
        )

    error = _call_in_thread(action, 2.5)
    assert isinstance(error, AnalysisTimeoutError)
    assert seen >= 2
    assert time.monotonic() - started < 2.5


def test_slow_visual_callback_can_be_cancelled() -> None:
    ffmpeg = _ffmpeg()
    cancel = threading.Event()

    def trip() -> None:
        time.sleep(0.3)
        cancel.set()

    threading.Thread(target=trip, daemon=True).start()
    started = time.monotonic()

    def action() -> None:
        stream_fixed_frames(
            _visual_command(ffmpeg, "8"),
            frame_bytes=_FRAME_BYTES,
            on_frame=_pause,
            deadline=None,
            cancel_event=cancel,
        )

    error = _call_in_thread(action, 2.5)
    assert isinstance(error, AnalysisCancelledError)
    assert time.monotonic() - started < 2.5


def test_slow_audio_callback_times_out_without_a_leftover_process() -> None:
    ffmpeg = _ffmpeg()
    seen = 0

    def on_chunk(block: bytes) -> None:
        nonlocal seen
        del block
        seen += 1
        time.sleep(0.05)

    def action() -> None:
        stream_chunks(
            _audio_command(ffmpeg),
            on_chunk=on_chunk,
            deadline=time.monotonic() + 0.35,
            cancel_event=None,
        )

    error = _call_in_thread(action, 2.5)
    assert isinstance(error, AnalysisTimeoutError)
    assert seen >= 1


def test_slow_audio_callback_can_be_cancelled() -> None:
    ffmpeg = _ffmpeg()
    cancel = threading.Event()
    seen = 0

    def on_chunk(block: bytes) -> None:
        nonlocal seen
        del block
        seen += 1
        cancel.set()
        time.sleep(0.05)

    def action() -> None:
        stream_chunks(
            _audio_command(ffmpeg),
            on_chunk=on_chunk,
            deadline=None,
            cancel_event=cancel,
        )

    error = _call_in_thread(action, 2.5)
    assert isinstance(error, AnalysisCancelledError)
    assert seen >= 1


def test_concurrent_visual_streams_stop_on_timeout_and_cancel() -> None:
    ffmpeg = _ffmpeg()
    before = _ffmpeg_pids()
    errors: list[BaseException] = []
    ready = threading.Barrier(2)

    def record(action: object) -> None:
        ready.wait(timeout=5)
        try:
            action()  # type: ignore[operator]
        except BaseException as exc:
            errors.append(exc)

    def timeout_action() -> None:
        stream_fixed_frames(
            _visual_command(ffmpeg, "8"),
            frame_bytes=_FRAME_BYTES,
            on_frame=_pause,
            deadline=time.monotonic() + 0.4,
            cancel_event=None,
        )

    cancel = threading.Event()

    def trip() -> None:
        time.sleep(0.25)
        cancel.set()

    threading.Thread(target=trip, daemon=True).start()

    def cancel_action() -> None:
        stream_fixed_frames(
            _visual_command(ffmpeg, "8"),
            frame_bytes=_FRAME_BYTES,
            on_frame=_pause,
            deadline=None,
            cancel_event=cancel,
        )

    workers = [
        threading.Thread(target=record, args=(timeout_action,), daemon=True),
        threading.Thread(target=record, args=(cancel_action,), daemon=True),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(6.0)
    leaked = _ffmpeg_pids() - before
    if any(worker.is_alive() for worker in workers) or leaked:
        _kill_pids(leaked)
        for worker in workers:
            worker.join(2.0)
    assert all(not worker.is_alive() for worker in workers)
    assert _ffmpeg_pids() - before == set()
    assert len(errors) == 2
    assert any(isinstance(error, AnalysisTimeoutError) for error in errors)
    assert any(isinstance(error, AnalysisCancelledError) for error in errors)


def test_fast_visual_stream_still_completes() -> None:
    ffmpeg = _ffmpeg()
    seen = 0

    def on_frame(buffer: bytes) -> None:
        nonlocal seen
        del buffer
        seen += 1

    before = _ffmpeg_pids()
    code, _stderr = stream_fixed_frames(
        _visual_command(ffmpeg, "0.4"),
        frame_bytes=_FRAME_BYTES,
        on_frame=on_frame,
        deadline=time.monotonic() + 10,
        cancel_event=None,
    )
    assert code == 0
    assert seen > 0
    assert _ffmpeg_pids() - before == set()
