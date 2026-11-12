# Public API

Import the library as `media_context_analysis`. The distribution name is `media-context-analysis`.

## analyze

```python
def analyze(path: str | Path, options: AnalysisOptions | None = None) -> AnalysisResult: ...
```

`path` is a local video file. The result contains seek times and evidence. It does not contain image bytes.

Stages, in order: validate options, reject a missing or empty file, resolve FFmpeg and FFprobe, probe streams, decode visual samples, decode audio when a stream exists, detect events, choose representative timestamps, and group related events. The deadline and the cancel event are checked during each subprocess, including while decoded frames or PCM chunks are still queued, and again between stages. A running callback or detector computation is checked after it returns; Python computation cannot be forcibly interrupted. An expired deadline or a set cancel event is never reported as success.

On Linux each child starts a new session; cleanup kills its process group. Normal interpreter shutdown also kills active owned children. Linux process groups cannot contain descendants that deliberately leave the group, or ensure cleanup after the hosting Python process is forcibly killed. Escaped descendants can retain pipe writers; reader waits are bounded, but those readers may outlive cleanup until the writer closes.

On Windows 10 or newer, each child is assigned atomically at creation to a library-owned Job Object with kill-on-close. Closing the job or terminating the host kills its descendants. If parent-job restrictions prevent containment, process creation fails with `FFmpegFailedError`; there is no uncontained fallback. Only library-owned children are targeted. On Windows, the launcher supports the CPython 3.11 through 3.14 process interface and fails closed if that interface is incompatible.

Audio that fails to decode is an error. Audio shorter than one analysis window is ignored and the picture is still analyzed. Acoustic votes are `unavailable`. A file with no audio stream is analyzed from the picture alone. Cover art and streams without positive dimensions or a valid FFprobe index are excluded. Default selection chooses the largest `width * height`, breaking ties by lowest index. An explicit `video_stream_index` selects that exact eligible stream; an unavailable or ineligible index raises `UnsupportedMediaError`. All video decode maps use the effective index, recorded in provenance. A file with no eligible video raises `UnsupportedMediaError`. Subtitles are not decoded.

`timeout_s` must be finite and greater than 0, or `None`. Relative names that start with `-` are passed as absolute paths.

## AnalysisOptions

Fields are keyword-only.

| Field | Default | Meaning |
| --- | --- | --- |
| `ffmpeg` | `None` | Executable name or path. `None` uses `PATH`. |
| `ffprobe` | `None` | Executable name or path. `None` uses `PATH`. |
| `timeout_s` | `1800` | Whole-analysis limit in seconds. `None` disables it. Must be finite and positive. |
| `video_stream_index` | `None` | FFprobe stream index to analyze; `None` uses the deterministic default. |
| `video_threads` | `2` | FFmpeg video decode threads, from 1 to 16. |
| `include_diagnostics` | `False` | Include raw candidates and detector statistics. |
| `cancel_event` | `None` | `threading.Event`. When set, analysis raises `AnalysisCancelledError`. |

`validate()` raises `InvalidOptionsError`. Booleans are rejected for numeric options. `video_stream_index` must be a non-negative integer or `None`. Detection thresholds are not fields on this object.

`detector_params.ffmpeg_threads` in the result stays `2`. `provenance.video_threads` is the thread count actually used.

## AnalysisResult

`to_dict()`, `to_json()`, `from_dict()`, and `from_json()` convert the document. `to_json()` uses UTF-8, two-space indent, LF line endings, and a trailing newline. Non-finite numbers are rejected. Nested dictionaries are copied at the serialization boundary. Key order is part of the contract and is documented in [schema.md](schema.md).

`schema_version` must be `MAJOR.MINOR.PATCH`, using ASCII integers without leading zeroes. A different major, or any other shape, raises `ValueError`. Extra keys are ignored. A true `forces_translation_segment`, `creates_translation_segment`, `subtitles_used`, or `audio_deletes_events` raises `ValueError`.

The supported public surface is `media_context_analysis.__all__`, this command-line interface, and the JSON schema. Other modules are importable and are not a compatibility promise.

## Errors

All analysis failures subclass `MediaContextAnalysisError`.

| Exception | When |
| --- | --- |
| `InvalidOptionsError` | Options are unusable. |
| `MediaNotFoundError` | The path is missing or is not a file. |
| `UnsupportedMediaError` | The file has no eligible video stream, or the selected index is unavailable. |
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
