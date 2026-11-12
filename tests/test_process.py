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


def _wait_until(action: object, seconds: float = 5) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if action():  # type: ignore[operator]
            return
        time.sleep(0.02)
    assert action()  # type: ignore[operator]


def _running(pid: int) -> bool:
    # Orphaned Unix zombies retain a PID but cannot execute or hold pipes.
    if sys.platform != "win32":
        from pathlib import Path

        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(") ", 1)[1].startswith("Z"):
            return False
    return process_alive(pid)


@pytest.mark.parametrize("leader_exits", [False, True])
def test_cleanup_kills_owned_descendants_only(tmp_path: object, leader_exits: bool) -> None:
    from pathlib import Path

    record = Path(str(tmp_path)) / "descendant.pid"
    child = "import time; time.sleep(30)"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({str(record)!r}).write_text(str(p.pid)); "
        + ("" if leader_exits else "time.sleep(30)")
    )
    outsider = subprocess.Popen([sys.executable, "-c", child])
    try:
        if leader_exits:
            run_bounded([sys.executable, "-c", script], deadline=None, cancel_event=None)
        else:
            with pytest.raises(AnalysisTimeoutError):
                run_bounded(
                    [sys.executable, "-c", script], deadline=None, cancel_event=None, timeout_s=0.5
                )
        pid = int(record.read_text())
        _wait_until(lambda: not _running(pid))
        assert outsider.poll() is None
    finally:
        outsider.kill()
        outsider.wait(timeout=5)


def test_exception_in_queued_consumer_cleans_up(tmp_path: object) -> None:
    from pathlib import Path

    record = Path(str(tmp_path)) / "producer.pid"
    script = (
        "import os,sys,time; from pathlib import Path; "
        f"Path({str(record)!r}).write_text(str(os.getpid())); "
        "sys.stdout.buffer.write(b'x'*1000000);sys.stdout.flush();time.sleep(30)"
    )

    def fail(block: bytes) -> None:
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        stream_chunks(
            [sys.executable, "-c", script], on_chunk=fail, deadline=None, cancel_event=None
        )
    assert not _running(int(record.read_text()))


def test_cancel_kills_producer_while_consumer_is_busy(tmp_path: object) -> None:
    from pathlib import Path

    record = Path(str(tmp_path)) / "producer.pid"
    script = (
        "import os,sys,time; from pathlib import Path; "
        f"Path({str(record)!r}).write_text(str(os.getpid())); "
        "sys.stdout.buffer.write(b'x'*1000000);sys.stdout.flush();time.sleep(30)"
    )
    cancel = threading.Event()

    def slow(block: bytes) -> None:
        cancel.set()
        pid = int(record.read_text())
        _wait_until(lambda: not _running(pid), seconds=2)

    with pytest.raises(AnalysisCancelledError):
        stream_chunks(
            [sys.executable, "-c", script], on_chunk=slow, deadline=None, cancel_event=cancel
        )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object host-death guarantee")
def test_windows_host_death_kills_tree_and_nested_jobs_work(tmp_path: object) -> None:
    from pathlib import Path

    from media_context_analysis.ffmpeg import _popen, _stop

    record = Path(str(tmp_path)) / "descendant.pid"
    child = "import time; time.sleep(30)"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({str(record)!r}).write_text(str(p.pid)); time.sleep(30)"
    )
    host_script = (
        "import sys; from media_context_analysis.ffmpeg import run_bounded; "
        f"run_bounded([sys.executable,'-c',{script!r}], deadline=None, cancel_event=None)"
    )
    # The host itself is inside an owned job. Its subprocess creates a nested job.
    host = _popen([sys.executable, "-c", host_script])
    try:
        _wait_until(record.exists)
        pid = int(record.read_text())
        assert _running(pid)
        host.kill()  # abrupt termination, without Python finally or atexit
        host.wait(timeout=5)
        _wait_until(lambda: not _running(pid))
    finally:
        _stop(host)


def test_reader_accepts_partial_unbuffered_frame_reads() -> None:
    frames: list[bytes] = []
    script = (
        "import sys,time; sys.stdout.buffer.write(b'ab');sys.stdout.flush();"
        "time.sleep(.1);sys.stdout.buffer.write(b'cd');sys.stdout.flush()"
    )
    code, _ = stream_fixed_frames(
        [sys.executable, "-c", script],
        frame_bytes=4,
        on_frame=frames.append,
        deadline=None,
        cancel_event=None,
    )
    assert code == 0
    assert frames == [b"abcd"]


def test_worker_start_failure_does_not_leak_process_or_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import media_context_analysis.ffmpeg as execution

    original = execution._popen
    owned: list[subprocess.Popen[bytes]] = []
    before = set(execution._active)

    def record(command: list[str]) -> subprocess.Popen[bytes]:
        process = original(command)
        owned.append(process)
        return process

    def fail_start(thread: threading.Thread) -> None:
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(execution, "_popen", record)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="worker unavailable"):
        run_bounded(
            [sys.executable, "-c", "import time;time.sleep(30)"], deadline=None, cancel_event=None
        )
    assert execution._active == before
    assert len(owned) == 1
    assert not _running(owned[0].pid)


@pytest.mark.parametrize("stage", ["version", "probe"])
def test_cancellation_during_probing_stages(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    from pathlib import Path

    import media_context_analysis.ffmpeg as execution

    cancel = threading.Event()
    original = execution._popen
    owned: list[subprocess.Popen[bytes]] = []

    def replace(command: list[str]) -> subprocess.Popen[bytes]:
        process = original([sys.executable, "-c", "import time;time.sleep(30)"])
        owned.append(process)
        cancel.set()
        return process

    monkeypatch.setattr(execution, "_popen", replace)
    with pytest.raises(AnalysisCancelledError):
        if stage == "version":
            execution.tool_version("ffprobe", deadline=None, cancel_event=cancel)
        else:
            execution.probe_file(Path("input.mkv"), "ffprobe", deadline=None, cancel_event=cancel)
    assert not _running(owned[0].pid)


def test_run_bounded_preserves_large_stdout_and_stderr_exactly() -> None:
    script = (
        "import sys;sys.stdout.buffer.write(b'o'*2000000);sys.stderr.buffer.write(b'e'*2000000)"
    )
    code, stdout, stderr = run_bounded(
        [sys.executable, "-c", script], deadline=time.monotonic() + 10, cancel_event=None
    )
    assert code == 0
    assert stdout == b"o" * 2000000
    assert stderr == b"e" * 2000000


def test_normal_host_shutdown_cleans_up_active_process(tmp_path: object) -> None:
    from pathlib import Path

    record = Path(str(tmp_path)) / "shutdown.pid"
    script = (
        "import os,time;from pathlib import Path;"
        f"Path({str(record)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    host_script = (
        "import sys,threading,time;from pathlib import Path;"
        "from media_context_analysis.ffmpeg import run_bounded;"
        f"threading.Thread(target=lambda:run_bounded([sys.executable,'-c',{script!r}],"
        "deadline=None,cancel_event=None),daemon=True).start();"
        f"p=Path({str(record)!r});\n"
        "while not p.exists():time.sleep(.01)\n"
    )
    host = subprocess.run([sys.executable, "-c", host_script], capture_output=True, timeout=10)
    assert host.returncode == 0, host.stderr.decode()
    _wait_until(lambda: not _running(int(record.read_text())))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows restricted parent Job Object")
def test_windows_incompatible_parent_job_fails_before_child_runs(tmp_path: object) -> None:
    from pathlib import Path

    marker = Path(str(tmp_path)) / "uncontained.txt"
    child = f"from pathlib import Path;Path({str(marker)!r}).write_text('ran')"
    script = (
        "import ctypes,sys;from ctypes import wintypes as w;"
        "from media_context_analysis._process import _kernel,_check,_ExtendedLimit;"
        "from media_context_analysis.ffmpeg import run_bounded;"
        "from media_context_analysis.errors import FFmpegFailedError;"
        "k=_kernel();j=k.CreateJobObjectW(None,None);_check(j);"
        "limits=_ExtendedLimit();limits.basic.flags=8;limits.basic.active_processes=1;"
        "_check(k.SetInformationJobObject(j,9,ctypes.byref(limits),ctypes.sizeof(limits)));"
        "k.AssignProcessToJobObject.argtypes=(w.HANDLE,w.HANDLE);"
        "k.AssignProcessToJobObject.restype=w.BOOL;"
        "_check(k.AssignProcessToJobObject(j,w.HANDLE(-1)));\n"
        "try:\n"
        f" run_bounded([sys.executable,'-c',{child!r}],deadline=None,cancel_event=None)\n"
        "except FFmpegFailedError:\n print('contained failure')\n"
        "else:\n raise AssertionError('restricted creation unexpectedly succeeded')\n"
    )
    host = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
    assert host.returncode == 0, host.stderr.decode()
    assert host.stdout.strip() == b"contained failure"
    assert not marker.exists()


def test_concurrent_job_close_cannot_close_a_reused_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    import media_context_analysis._process as ownership

    closed: list[int] = []

    class Kernel:
        def CloseHandle(self, handle: int) -> None:
            closed.append(handle)

    monkeypatch.setattr(ownership, "_kernel", Kernel)
    old = ownership.WindowsOwnedProcess.__new__(ownership.WindowsOwnedProcess)
    new = ownership.WindowsOwnedProcess.__new__(ownership.WindowsOwnedProcess)
    old._job_handle = 42
    ownership._jobs.add(old)
    first_closed = threading.Event()
    release = threading.Event()
    original = ownership._close_job

    def pause_after_close(owner: object) -> None:
        original(owner)
        if threading.current_thread().name == "first-closer":
            first_closed.set()
            assert release.wait(5)

    monkeypatch.setattr(ownership, "_close_job", pause_after_close)
    worker = threading.Thread(target=old.close_job, name="first-closer")
    worker.start()
    try:
        assert first_closed.wait(5)
        new._job_handle = 42  # Windows may reuse the numeric handle immediately.
        ownership._jobs.add(new)
        old.close_job()
        assert closed == [42]
        assert new in ownership._jobs
        assert new._job_handle == 42
    finally:
        release.set()
        worker.join(5)
        new.close_job()
    assert closed == [42, 42]
    assert old not in ownership._jobs
    assert new not in ownership._jobs
