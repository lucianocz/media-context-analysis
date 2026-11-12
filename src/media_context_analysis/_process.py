"""Atomic Windows subprocess containment using standard-library Win32 bindings.

Only the three child pipe handles are inherited. A job-list creation attribute
assigns the process before its first instruction, including if the host dies
inside CreateProcess. Incompatible enclosing jobs fail closed.
"""

from __future__ import annotations

import atexit
import ctypes
import subprocess
import threading
from ctypes import wintypes as wt
from typing import Any

_jobs: set[WindowsOwnedProcess] = set()
_lock = threading.Lock()


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("lpReserved", wt.LPWSTR),
        ("lpDesktop", wt.LPWSTR),
        ("lpTitle", wt.LPWSTR),
        ("dwX", wt.DWORD),
        ("dwY", wt.DWORD),
        ("dwXSize", wt.DWORD),
        ("dwYSize", wt.DWORD),
        ("dwXCountChars", wt.DWORD),
        ("dwYCountChars", wt.DWORD),
        ("dwFillAttribute", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("wShowWindow", wt.WORD),
        ("cbReserved2", wt.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wt.HANDLE),
        ("hStdOutput", wt.HANDLE),
        ("hStdError", wt.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("startup", _StartupInfo), ("attributes", ctypes.c_void_p)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [("process", wt.HANDLE), ("thread", wt.HANDLE), ("pid", wt.DWORD), ("tid", wt.DWORD)]


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", wt.DWORD),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_processes", wt.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", wt.DWORD),
        ("scheduling", wt.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "read_ops",
            "write_ops",
            "other_ops",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimit),
        ("io", _IoCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


def _kernel() -> Any:
    win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 - Windows-only ctypes API
    kernel = win_dll("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE),
        "SetInformationJobObject": ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL),
        "CloseHandle": ([wt.HANDLE], wt.BOOL),
        "InitializeProcThreadAttributeList": (
            [ctypes.c_void_p, wt.DWORD, wt.DWORD, ctypes.POINTER(ctypes.c_size_t)],
            wt.BOOL,
        ),
        "UpdateProcThreadAttribute": (
            [
                ctypes.c_void_p,
                wt.DWORD,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
            wt.BOOL,
        ),
        "DeleteProcThreadAttributeList": ([ctypes.c_void_p], None),
        "CreateProcessW": (
            [
                wt.LPCWSTR,
                wt.LPWSTR,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wt.BOOL,
                wt.DWORD,
                ctypes.c_void_p,
                wt.LPCWSTR,
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessInfo),
            ],
            wt.BOOL,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = arguments
        function.restype = result
    return kernel


def _check(ok: object) -> None:
    if not ok:
        win_error = getattr(ctypes, "WinError")  # noqa: B009 - Windows-only ctypes API
        last_error = getattr(ctypes, "get_last_error")  # noqa: B009
        raise win_error(last_error())


def _close_job(owner: WindowsOwnedProcess) -> None:
    with _lock:
        handle = owner._job_handle
        owner._job_handle = None
        _jobs.discard(owner)
        if handle is not None:
            _kernel().CloseHandle(handle)


def _shutdown() -> None:
    with _lock:
        owners = tuple(_jobs)
    for owner in owners:
        owner.close_job()


atexit.register(_shutdown)


class WindowsOwnedProcess(subprocess.Popen[bytes]):
    """Popen pipe/wait semantics with atomic job assignment on Windows 10+."""

    _job_handle: int | None = None

    def _execute_child(self, *args: Any, **kwargs: Any) -> None:
        # Popen's private platform hook has the same positional arguments on
        # supported CPython 3.11-3.14. This class is used only by our fixed call.
        if len(args) != 23 or kwargs or not isinstance(args[0], list):
            raise OSError("unsupported CPython Windows process hook; containment unavailable")
        command = args[0]
        p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite = args[10:16]
        kernel = _kernel()
        job = kernel.CreateJobObjectW(None, None)
        _check(job)
        self._job_handle = int(job)
        with _lock:
            _jobs.add(self)
        attributes = None
        initialized = False
        try:
            limits = _ExtendedLimit()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _check(
                kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            )
            size = ctypes.c_size_t()
            kernel.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
            attributes = ctypes.create_string_buffer(size.value)
            _check(kernel.InitializeProcThreadAttributeList(attributes, 2, 0, ctypes.byref(size)))
            initialized = True
            handles = (wt.HANDLE * 3)(int(p2cread), int(c2pwrite), int(errwrite))
            jobs = (wt.HANDLE * 1)(job)
            # HANDLE_LIST and JOB_LIST. The job handle itself is not inherited.
            for attribute, values in ((0x20002, handles), (0x2000D, jobs)):
                _check(
                    kernel.UpdateProcThreadAttribute(
                        attributes, 0, attribute, values, ctypes.sizeof(values), None, None
                    )
                )
            startup = _StartupInfoEx()
            startup.startup.cb = ctypes.sizeof(startup)
            startup.startup.dwFlags = 0x100  # STARTF_USESTDHANDLES
            startup.startup.hStdInput = int(p2cread)
            startup.startup.hStdOutput = int(c2pwrite)
            startup.startup.hStdError = int(errwrite)
            startup.attributes = ctypes.cast(attributes, ctypes.c_void_p)
            info = _ProcessInfo()
            line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
            _check(
                kernel.CreateProcessW(
                    None,
                    line,
                    None,
                    None,
                    True,
                    0x08000000 | 0x00080000,
                    None,
                    None,
                    ctypes.byref(startup),
                    ctypes.byref(info),
                )
            )
            self._child_created = True
            setattr(self, "_handle", getattr(subprocess, "Handle")(info.process))  # noqa: B009, B010
            self.pid = int(info.pid)
            kernel.CloseHandle(info.thread)
        except BaseException:
            self.close_job()
            raise
        finally:
            if initialized:
                kernel.DeleteProcThreadAttributeList(attributes)
            getattr(self, "_close_pipe_fds")(  # noqa: B009
                p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite
            )

    def close_job(self) -> None:
        """Terminate every remaining descendant and close our job handle."""
        _close_job(self)

    def close_process_handle(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.Close()
