# Public API

Import the library as `media_context_analysis`. The distribution name is `media-context-analysis`.

## analyze

```python
def analyze(path: str | Path, options: AnalysisOptions | None = None) -> AnalysisResult: ...
```

`path` is a local video file. The result contains seek times and evidence. It does not contain image bytes.

Stages, in order: validate options, reject a missing or empty file, resolve FFmpeg and FFprobe, probe streams, decode visual samples, decode audio when a stream exists, detect events, choose representative timestamps, and group related events. The deadline and the cancel event are checked during each subprocess, including while decoded frames or PCM chunks are still queued, and again between stages. A slow callback is interrupted after it returns. An expired deadline or a set cancel event is never reported as success.

On Linux the FFmpeg process is a new session, and a stop signals that process group. On Windows only the direct child is killed. There is no Job Object. A parent that is hard-killed from outside can leave the child running.

Audio that fails to decode is an error. Audio shorter than one analysis window is ignored and the picture is still analyzed. `audio_present` stays true when that stream exists, and acoustic votes are `unavailable`. A file with no audio stream is analyzed from the picture alone. Cover art (`disposition.attached_pic`) is not the video stream. A file with no real video stream raises `UnsupportedMediaError`. Subtitle streams are counted and are not decoded.

`timeout_s` must be finite and greater than 0, or `None`. Relative names that start with `-` are passed as absolute paths.

## AnalysisOptions

Fields are keyword-only.

| Field | Default | Meaning |
| --- | --- | --- |
| `ffmpeg` | `None` | Executable name or path. `None` uses `PATH`. |
| `ffprobe` | `None` | Executable name or path. `None` uses `PATH`. |
| `timeout_s` | `1800` | Whole-analysis limit in seconds. `None` disables it. Must be finite and positive. |
| `video_threads` | `2` | FFmpeg video decode threads, from 1 to 16. |
| `include_diagnostics` | `False` | Include raw candidates and detector statistics. |
| `cancel_event` | `None` | `threading.Event`. When set, analysis raises `AnalysisCancelledError`. |

`validate()` raises `InvalidOptionsError`. Booleans are rejected for `timeout_s` and `video_threads`. Detection thresholds are not fields on this object.

`detector_params.ffmpeg_threads` in the result stays `2`. `provenance.video_threads` is the thread count actually used.

## AnalysisResult

`to_dict()`, `to_json()`, `from_dict()`, and `from_json()` convert the document. `to_json()` uses UTF-8, two-space indent, LF line endings, and a trailing newline. Non-finite numbers are rejected. Nested dictionaries are copied at the serialization boundary. Key order is part of the contract and is documented in [schema.md](schema.md).

`schema_version` must be `MAJOR.MINOR.PATCH`. A different major, or any other shape, raises `ValueError`. Extra keys are ignored. A true `forces_translation_segment`, `creates_translation_segment`, `subtitles_read`, `subtitles_used`, or `audio_deletes_events` raises `ValueError`.

The supported public surface is `media_context_analysis.__all__`, this command-line interface, and the JSON schema. Other modules are importable and are not a compatibility promise.

## Errors

All analysis failures subclass `MediaContextAnalysisError`.

| Exception | When |
| --- | --- |
| `InvalidOptionsError` | Options are unusable. |
| `MediaNotFoundError` | The path is missing or is not a file. |
| `UnsupportedMediaError` | The file has no video stream. |
| `CorruptMediaError` | The file is empty, or the probe or decode is inconsistent. |
| `FFmpegNotFoundError` | FFmpeg or FFprobe cannot be started from the configured path. |
| `FFmpegFailedError` | FFmpeg or FFprobe exits with an error. |
| `AnalysisTimeoutError` | The deadline is exceeded. The subprocess is killed. |
| `AnalysisCancelledError` | The cancel event is set. The subprocess is killed. |

## Command line

`media-context-analysis` and `python -m media_context_analysis` call the same entry point.

| Argument | Meaning |
| --- | --- |
| `path` | Local video file. |
| `-o`, `--output` | Write JSON to this file instead of stdout. |
| `--diagnostics` | Include diagnostics. |
| `--timeout` | Timeout in seconds. |
| `--ffmpeg`, `--ffprobe` | Tool path or name. |
| `--threads` | Video decode threads. |
| `--verbose` | Print a traceback on failure. |
| `--version` | Print the package version. |

The process does not create a missing output directory. That directory is checked before analysis starts. JSON written to a file uses LF line endings on every platform. Stdout is UTF-8.
