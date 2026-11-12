"""Command-line entry point. JSON goes to stdout or to ``--output``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from media_context_analysis._version import __version__
from media_context_analysis.analyze import analyze
from media_context_analysis.errors import (
    AnalysisCancelledError,
    AnalysisTimeoutError,
    CorruptMediaError,
    FFmpegFailedError,
    FFmpegNotFoundError,
    InvalidOptionsError,
    MediaNotFoundError,
    UnsupportedMediaError,
)
from media_context_analysis.options import AnalysisOptions

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_MEDIA = 3
EXIT_FFMPEG = 4
EXIT_TIMEOUT = 5
EXIT_CANCELLED = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media-context-analysis",
        description="Write a versioned JSON timeline of visual events for a local video file.",
    )
    parser.add_argument("path", help="Local video file")
    parser.add_argument("-o", "--output", help="Write JSON to this path instead of stdout")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include raw candidates and detector diagnostics",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Whole-analysis timeout in seconds (default: 1800)",
    )
    parser.add_argument("--ffmpeg", default=None, help="FFmpeg executable name or path")
    parser.add_argument("--ffprobe", default=None, help="FFprobe executable name or path")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="FFmpeg video decode threads (default: 2)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print the exception traceback on failure"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return EXIT_USAGE if code is None else int(code)
    try:
        destination: Path | None = None
        if args.output:
            destination = Path(args.output)
            parent = destination.parent
            if parent != Path() and not parent.exists():
                print(f"output directory does not exist: {parent}", file=sys.stderr)
                return EXIT_USAGE
        options = AnalysisOptions(
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            timeout_s=args.timeout,
            video_threads=args.threads,
            include_diagnostics=args.diagnostics,
        )
        result = analyze(args.path, options)
        payload = result.to_json()
        if destination is not None:
            destination.write_text(payload, encoding="utf-8", newline="\n")
        else:
            _write_stdout(payload)
    except InvalidOptionsError as exc:
        return _fail(exc, EXIT_USAGE, args.verbose)
    except (MediaNotFoundError, UnsupportedMediaError, CorruptMediaError) as exc:
        return _fail(exc, EXIT_MEDIA, args.verbose)
    except (FFmpegNotFoundError, FFmpegFailedError) as exc:
        return _fail(exc, EXIT_FFMPEG, args.verbose)
    except AnalysisTimeoutError as exc:
        return _fail(exc, EXIT_TIMEOUT, args.verbose)
    except AnalysisCancelledError as exc:
        return _fail(exc, EXIT_CANCELLED, args.verbose)
    except KeyboardInterrupt:
        print("analysis cancelled", file=sys.stderr)
        return EXIT_CANCELLED
    except Exception as exc:
        return _fail(exc, EXIT_UNEXPECTED, args.verbose)
    return EXIT_OK


def _write_stdout(payload: str) -> None:
    data = payload.encode("utf-8")
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None and hasattr(buffer, "write"):
        buffer.write(data)
        flush = getattr(buffer, "flush", None)
        if callable(flush):
            flush()
        return
    sys.stdout.write(payload)


def _fail(exc: BaseException, code: int, verbose: bool) -> int:
    if verbose:
        import traceback

        traceback.print_exc()
    else:
        print(str(exc), file=sys.stderr)
    return code
