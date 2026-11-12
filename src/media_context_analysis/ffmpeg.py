"""Bounded FFmpeg and FFprobe execution.

Video decode uses the configured thread count (default 2). Audio decode uses
one thread. PCM is reduced while it is read. Reader queues hold only a few
chunks, so a slow callback cannot buffer the whole file. The deadline and the
cancel event are checked while those queues are nonempty. No image files are
written.

On Linux each child owns a new session and cleanup kills its process group.
On Windows creation atomically assigns each child to a kill-on-close Job Object.
Incompatible enclosing Windows jobs fail closed before any child can run.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from media_context_analysis._process import WindowsOwnedProcess
from media_context_analysis.errors import (
    AnalysisCancelledError,
    AnalysisTimeoutError,
    CorruptMediaError,
    FFmpegFailedError,
    FFmpegNotFoundError,
    UnsupportedMediaError,
)
from media_context_analysis.params import DetectorParams
from media_context_analysis.signal import PcmFeatureReducer, grayscale_thumbnail, hsv_histogram

AUDIO_THREADS = 1
_POLL_S = 0.05
_STREAM_QUEUE = 8
_active: set[subprocess.Popen[bytes]] = set()
_active_lock = threading.Lock()


def _shutdown_processes() -> None:
    with _active_lock:
        processes = tuple(_active)
    for process in processes:
        _kill(process)


atexit.register(_shutdown_processes)


class _StopMonitor:
    """Kill subprocesses even while a consumer is busy processing queued output."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        deadline: float | None,
        cancel: threading.Event | None,
    ) -> None:
        self.done = threading.Event()
        self.thread = threading.Thread(
            target=self._watch, args=(process, deadline, cancel), daemon=True
        )
        with _active_lock:
            _active.add(process)
        try:
            self.thread.start()
        except BaseException:
            with _active_lock:
                _active.discard(process)
            raise

    def _watch(
        self,
        process: subprocess.Popen[bytes],
        deadline: float | None,
        cancel: threading.Event | None,
    ) -> None:
        while not self.done.wait(_POLL_S):
            if (cancel is not None and cancel.is_set()) or (
                deadline is not None and time.monotonic() > deadline
            ):
                _kill(process)
                return

    def close(self, process: subprocess.Popen[bytes]) -> None:
        self.done.set()
        self.thread.join(timeout=1)


def resolve_tool(explicit: str | None, name: str) -> str:
    """Return an executable path. ``explicit`` may be a name on ``PATH`` or a file path."""
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(explicit)
        if found:
            return found
        raise FFmpegNotFoundError(f"{name} executable not found: {explicit}")
    found = shutil.which(name)
    if not found:
        raise FFmpegNotFoundError(f"{name} is not on PATH")
    return found


def tool_version(
    executable: str, *, deadline: float | None, cancel_event: threading.Event | None
) -> str | None:
    code, stdout, stderr = run_bounded(
        [executable, "-version"],
        deadline=deadline,
        cancel_event=cancel_event,
    )
    if code != 0:
        return None
    text = stdout.decode("utf-8", "replace") or stderr.decode("utf-8", "replace")
    line = text.splitlines()[0].strip() if text.splitlines() else ""
    return line or None


def probe_file(
    video: Path,
    ffprobe: str,
    *,
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> dict[str, object]:
    code, stdout, stderr = run_bounded(
        [
            ffprobe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            _input_argument(video),
        ],
        deadline=deadline,
        cancel_event=cancel_event,
    )
    if code != 0:
        tail = stderr.decode("utf-8", "replace")[-2000:]
        raise CorruptMediaError(f"ffprobe failed for {video.name}: {tail}".rstrip())
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CorruptMediaError(f"ffprobe returned invalid JSON for {video.name}") from exc
    if not isinstance(payload, dict):
        raise CorruptMediaError(f"ffprobe returned unexpected JSON for {video.name}")
    return payload


def media_streams(
    probe_data: dict[str, object],
    *,
    video_stream_index: int | None = None,
) -> tuple[dict[str, object], dict[str, object] | None, int]:
    """Select the largest usable video, first audio, and subtitle count.

    A usable video has a nonnegative integer FFprobe index and positive integer
    dimensions, and is not attached cover art. Equal areas use the lowest index.
    An explicit index must identify one of these usable video streams.
    """
    streams = probe_data.get("streams") or []
    if not isinstance(streams, list):
        raise CorruptMediaError("ffprobe JSON has no stream list")
    videos = [stream for stream in streams if isinstance(stream, dict) and _usable_video(stream)]
    if video_stream_index is not None:
        video = next((stream for stream in videos if stream["index"] == video_stream_index), None)
        if video is None:
            raise UnsupportedMediaError(
                f"video stream index {video_stream_index} is unavailable or is not a usable video"
            )
    else:
        if not videos:
            raise UnsupportedMediaError("no usable video stream")
        video = min(
            videos,
            key=lambda stream: (
                -cast(int, stream["width"]) * cast(int, stream["height"]),
                cast(int, stream["index"]),
            ),
        )
    audio = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    subtitles = sum(
        1
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "subtitle"
    )
    return video, audio if isinstance(audio, dict) else None, subtitles


def _usable_video(stream: dict[str, object]) -> bool:
    if stream.get("codec_type") != "video" or _attached_picture(stream):
        return False
    for name, minimum in (("index", 0), ("width", 1), ("height", 1)):
        value = stream.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return False
    return True


def visual_filter(params: DetectorParams) -> str:
    return f"fps={params.sample_fps},scale={params.frame_width}:{params.frame_height}:flags=area"


def extract_visual(
    video: Path,
    ffmpeg: str,
    params: DetectorParams,
    video_stream: dict[str, object],
    *,
    video_threads: int,
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> dict[str, NDArray[np.floating] | NDArray[np.float32]]:
    frame_bytes = params.frame_width * params.frame_height * 3
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-threads",
        str(video_threads),
        "-i",
        _input_argument(video),
        "-map",
        _stream_map(video_stream),
        "-vf",
        f"{visual_filter(params)},showinfo",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    histograms: list[NDArray[np.float32]] = []
    thumbs: list[NDArray[np.float32]] = []
    luma: list[float] = []

    def on_frame(buffer: bytes) -> None:
        rgb = np.frombuffer(buffer, dtype=np.uint8).reshape(
            params.frame_height, params.frame_width, 3
        )
        histograms.append(hsv_histogram(rgb, params.hue_bins, params.sat_bins, params.val_bins))
        thumbs.append(grayscale_thumbnail(rgb, params.thumb_x, params.thumb_y))
        luma.append(float(rgb.mean()) / 255.0)

    code, stderr = stream_fixed_frames(
        command,
        frame_bytes=frame_bytes,
        on_frame=on_frame,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    text = stderr.decode("utf-8", "replace")
    if code != 0:
        raise FFmpegFailedError(f"ffmpeg visual extraction failed\n{text[-2000:]}")
    times = _pts_times(text)
    if len(times) != len(histograms):
        raise CorruptMediaError(
            f"frame timestamp count {len(times)} does not match decoded frames {len(histograms)}"
        )
    if not histograms:
        raise CorruptMediaError("ffmpeg produced no visual frames")
    return {
        "times": np.asarray(times, dtype=np.float64),
        "hists": np.vstack(histograms).astype(np.float32),
        "thumbs": np.vstack(thumbs).astype(np.float32),
        "luma": np.asarray(luma, dtype=np.float32),
    }


def extract_audio_features(
    video: Path,
    ffmpeg: str,
    params: DetectorParams,
    *,
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        str(AUDIO_THREADS),
        "-i",
        _input_argument(video),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(params.audio_sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    reducer = PcmFeatureReducer(
        params.audio_sample_rate,
        params.audio_window_s,
        params.audio_hop_s,
    )
    pending = bytearray()

    def on_chunk(block: bytes) -> None:
        nonlocal pending
        pending.extend(block)
        usable = len(pending) - (len(pending) % 4)
        if usable <= 0:
            return
        samples = np.frombuffer(bytes(pending[:usable]), dtype=np.float32).copy()
        del pending[:usable]
        reducer.extend(samples)

    code, stderr = stream_chunks(
        command,
        on_chunk=on_chunk,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    if code != 0:
        tail = stderr.decode("utf-8", "replace")[-2000:]
        raise FFmpegFailedError(f"ffmpeg audio extraction failed\n{tail}".rstrip())
    # An empty feature matrix means the track was shorter than one window.
    # The caller keeps the picture and treats the audio as unavailable.
    return reducer.finish()


def run_bounded(
    command: list[str],
    *,
    deadline: float | None,
    cancel_event: threading.Event | None,
    timeout_s: float | None = None,
) -> tuple[int, bytes, bytes]:
    """Run ``command`` and return ``(code, stdout, stderr)``.

    ``timeout_s`` is a convenience for tests. ``deadline`` is an absolute
    ``time.monotonic`` value. The earlier limit wins.
    """
    effective = _deadline(deadline, timeout_s)
    _raise_if_stopped(effective, cancel_event)
    process: subprocess.Popen[bytes] | None = None
    monitor: _StopMonitor | None = None
    out_thread: threading.Thread | None = None
    err_thread: threading.Thread | None = None
    out_parts: list[bytes] = []
    err_parts: list[bytes] = []
    try:
        process = _popen(command)
        monitor = _StopMonitor(process, effective, cancel_event)
        out_thread = threading.Thread(target=_drain, args=(process.stdout, out_parts), daemon=True)
        err_thread = threading.Thread(target=_drain, args=(process.stderr, err_parts), daemon=True)
        out_thread.start()
        err_thread.start()
        _wait(process, effective, cancel_event)
    finally:
        if process is not None:
            _cleanup(process, monitor, out_thread, err_thread)
    assert process is not None
    return (
        int(process.returncode if process.returncode is not None else -1),
        b"".join(out_parts),
        b"".join(err_parts),
    )


def stream_fixed_frames(
    command: list[str],
    *,
    frame_bytes: int,
    on_frame: Callable[[bytes], None],
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> tuple[int, bytes]:
    """Read fixed-size stdout frames. Stderr is captured for ``showinfo`` timestamps."""

    def on_item(item: bytes) -> None:
        if len(item) != frame_bytes:
            raise FFmpegFailedError("ffmpeg reader returned an unexpected frame")
        on_frame(item)

    return _stream(command, deadline, cancel_event, _read_frames, on_item, frame_bytes)


def stream_chunks(
    command: list[str],
    *,
    on_chunk: Callable[[bytes], None],
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> tuple[int, bytes]:
    def on_item(item: bytes) -> None:
        if item:
            on_chunk(item)

    return _stream(command, deadline, cancel_event, _read_chunks, on_item, None)


def _pts_times(stderr: str) -> list[float]:
    import re

    pattern = re.compile(r"\bn:\s*(\d+)\s+pts:\s*\d+\s+pts_time:([0-9]+(?:\.[0-9]+)?)")
    return [float(match.group(2)) for match in pattern.finditer(stderr)]


def _deadline(deadline: float | None, timeout_s: float | None) -> float | None:
    if timeout_s is not None:
        if timeout_s <= 0:
            raise AnalysisTimeoutError("analysis timed out")
        sooner = time.monotonic() + timeout_s
        if deadline is None or sooner < deadline:
            return sooner
    return deadline


def _input_argument(video: Path) -> str:
    """Absolute path, so a relative name such as ``-dash.mp4`` is not an option."""
    return str(video.resolve())


def _attached_picture(stream: dict[str, object]) -> bool:
    disposition = stream.get("disposition")
    if not isinstance(disposition, dict):
        return False
    flag = disposition.get("attached_pic")
    return flag in (1, True) or flag == "1"


def _stream_map(stream: dict[str, object]) -> str:
    """FFmpeg input specifier for one probed stream, never ``0:v:0``.

    ``0:v:0`` is the first video stream, which may be cover art.
    """
    index = stream.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CorruptMediaError("selected video stream has no index")
    return f"0:{index}"


def _popen(command: list[str]) -> subprocess.Popen[bytes]:
    try:
        if sys.platform == "win32":
            return WindowsOwnedProcess(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
            )
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(f"executable not found: {command[0]}") from exc
    except OSError as exc:
        raise FFmpegFailedError(f"could not start {command[0]}: {exc}") from exc


def _stream(
    command: list[str],
    deadline: float | None,
    cancel_event: threading.Event | None,
    reader: Callable[..., None],
    on_item: Callable[[bytes], None],
    frame_bytes: int | None,
) -> tuple[int, bytes]:
    _raise_if_stopped(deadline, cancel_event)
    stop = threading.Event()
    process: subprocess.Popen[bytes] | None = None
    monitor: _StopMonitor | None = None
    out_thread: threading.Thread | None = None
    err_thread: threading.Thread | None = None
    err_parts: list[bytes] = []
    try:
        process = _popen(command)
        monitor = _StopMonitor(process, deadline, cancel_event)
        items: queue.Queue[object] = queue.Queue(maxsize=_STREAM_QUEUE)
        assert process.stdout is not None
        assert process.stderr is not None
        if frame_bytes is None:
            out_thread = threading.Thread(
                target=reader, args=(process.stdout, items, stop), daemon=True
            )
        else:
            out_thread = threading.Thread(
                target=reader,
                args=(process.stdout, frame_bytes, items, stop),
                daemon=True,
            )
        err_thread = threading.Thread(target=_drain, args=(process.stderr, err_parts), daemon=True)
        out_thread.start()
        err_thread.start()
        _consume(process, items, out_thread, deadline, cancel_event, on_item)
    finally:
        stop.set()
        if process is not None:
            _cleanup(process, monitor, out_thread, err_thread)
    assert process is not None
    return int(process.returncode if process.returncode is not None else -1), b"".join(err_parts)


def _consume(
    process: subprocess.Popen[bytes],
    items: queue.Queue[object],
    reader: threading.Thread,
    deadline: float | None,
    cancel_event: threading.Event | None,
    on_item: Callable[[bytes], None],
) -> None:
    while True:
        try:
            item = items.get(timeout=_POLL_S)
        except queue.Empty:
            _raise_if_stopped(deadline, cancel_event, process)
            if process.poll() is not None and not reader.is_alive():
                break
            continue
        # A kill unblocks the reader with an error. Prefer the deadline or cancel.
        _raise_if_stopped(deadline, cancel_event, process)
        if item is None:
            break
        if isinstance(item, BaseException):
            raise FFmpegFailedError(f"reading ffmpeg output failed: {item}") from item
        if not isinstance(item, bytes):
            raise FFmpegFailedError("ffmpeg reader returned an unexpected frame")
        on_item(item)
    _raise_if_stopped(deadline, cancel_event, process)
    _wait(process, deadline, cancel_event)


def _offer(items: queue.Queue[object], item: object, stop: threading.Event) -> None:
    """Queue ``item`` unless ``stop`` is set. A full queue cannot block past one poll."""
    while not stop.is_set():
        try:
            items.put(item, timeout=_POLL_S)
            return
        except queue.Full:
            continue


def _drain(stream: object, parts: list[bytes]) -> None:
    if stream is None or not hasattr(stream, "read"):
        return
    try:
        while True:
            block = cast(BinaryIO, stream).read(1 << 16)
            if not block:
                break
            parts.append(block)
    except Exception:
        return


def _read_frames(
    stream: object,
    frame_bytes: int,
    frames: queue.Queue[object],
    stop: threading.Event,
) -> None:
    try:
        while not stop.is_set():
            pieces = bytearray()
            while len(pieces) < frame_bytes and not stop.is_set():
                block = cast(BinaryIO, stream).read(frame_bytes - len(pieces))
                if not block:
                    break
                pieces.extend(block)
            buffer = bytes(pieces)
            if stop.is_set():
                return
            if len(buffer) < frame_bytes:
                _offer(frames, None, stop)
                return
            _offer(frames, buffer, stop)
    except Exception as exc:
        _offer(frames, exc, stop)


def _read_chunks(stream: object, chunks: queue.Queue[object], stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            block = cast(BinaryIO, stream).read(1 << 16)
            if stop.is_set():
                return
            if not block:
                _offer(chunks, None, stop)
                return
            _offer(chunks, block, stop)
    except Exception as exc:
        _offer(chunks, exc, stop)


def _wait(
    process: subprocess.Popen[bytes], deadline: float | None, cancel_event: threading.Event | None
) -> None:
    while process.poll() is None:
        _raise_if_stopped(deadline, cancel_event, process)
        time.sleep(_POLL_S)
    _raise_if_stopped(deadline, cancel_event, process)


def _raise_if_stopped(
    deadline: float | None,
    cancel_event: threading.Event | None,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelledError("analysis cancelled")
    if deadline is not None and time.monotonic() > deadline:
        raise AnalysisTimeoutError("analysis timed out")


def _kill(process: subprocess.Popen[bytes]) -> None:
    if isinstance(process, WindowsOwnedProcess):
        process.close_job()
        return
    if sys.platform != "win32":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    if process.poll() is None:
        process.kill()


def _cleanup(
    process: subprocess.Popen[bytes],
    monitor: _StopMonitor | None,
    out_thread: threading.Thread | None,
    err_thread: threading.Thread | None,
) -> None:
    if monitor is not None:
        monitor.close(process)
    try:
        _stop(process, close_pipes=False)
        # First let readers drain EOF, preserving bytes already written by the
        # exited process. Closing pipes before joining could discard the tail.
        for worker in (out_thread, err_thread):
            if worker is not None and worker.ident is not None:
                worker.join(timeout=2)
    finally:
        _close_resources(process)
        with _active_lock:
            _active.discard(process)


def _stop(process: subprocess.Popen[bytes], *, close_pipes: bool = True) -> None:
    if getattr(process, "_mca_stopped", False):
        if close_pipes:
            _close_resources(process)
        return
    setattr(process, "_mca_stopped", True)  # noqa: B010
    _kill(process)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill(process)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if close_pipes:
        _close_resources(process)


def _close_resources(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr, process.stdin):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    if isinstance(process, WindowsOwnedProcess):
        process.close_process_handle()


# Imported for tests that inspect the process table without spawning a shell.
def process_alive(pid: int) -> bool:
    """Return whether ``pid`` still exists.

    On Windows ``os.kill(pid, 0)`` terminates the process, so the check uses
    ``OpenProcess`` and does not signal it.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_process_alive(pid: int) -> bool:
    import ctypes

    # WinDLL and get_last_error exist on Windows. Linux type stubs omit them.
    windll = getattr(ctypes, "WinDLL", None)
    last_error = getattr(ctypes, "get_last_error", None)
    if windll is None or last_error is None:
        return False
    kernel32 = windll("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, 0, pid)
    if not handle:
        return int(last_error()) == 5
    code = ctypes.c_ulong()
    queried = int(kernel32.GetExitCodeProcess(handle, ctypes.byref(code)))
    kernel32.CloseHandle(handle)
    if not queried:
        return True
    return int(code.value) == 259
