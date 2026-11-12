# media-context-analysis

`media-context-analysis` reads a local video file and returns a versioned JSON timeline of visual events. Each event has a time, a category, evidence, uncertainty, and seek timestamps. The library does not extract, store, or return images.

It does not read subtitles, translate text, call a metadata service, or open a translation segment. A suggested context break is advisory.

## Features

- Compact visual timeline: `brief_visual_insert`, `visual_departure`, `sharp_stable_cut`, and `palette_shift`
- Seek times for the sample before the cut, the middle of the previous shot, and the sample after the cut
- An audio vote on each visual event. Audio does not add or delete events
- Optional `brief_insert_return` groups that do not remove events
- Timeouts and cancellation while frames are still being decoded
- JSON schema `2.0.0`, with detector provenance and no source-file metadata

## Requirements

- Python 3.11, 3.12, 3.13, or 3.14
- Windows 10 or newer, or Linux
- FFmpeg and FFprobe on `PATH`, or paths passed in by the caller

FFmpeg is not bundled. Video decode uses two threads by default. Audio decode uses one thread. NumPy is the only required Python dependency.

## Installation

```bash
pip install "git+https://github.com/lucianocz/media-context-analysis.git"
```

This package is not published to PyPI. From a clone, `pip install -e ".[dev]"` installs the test tools.

## Python API

```python
from media_context_analysis import AnalysisOptions, analyze

result = analyze("episode.mkv", AnalysisOptions(include_diagnostics=False))
for event in result.timeline:
    print(event.time, event.category, event.priority)
print(result.to_json())
```

`analyze(path, options=None)` returns an `AnalysisResult`. `AnalysisOptions` fields are keyword-only. Detection thresholds are not caller settings. Event times lie on a 4 fps grid, so a cut can be up to 0.25 s earlier than `time_s`. The document describes detected events and includes no source-file metadata. By default, analysis uses the valid non-cover video stream with the largest pixel area, then the lowest FFprobe index. Set `AnalysisOptions(video_stream_index=2)` to select an index explicitly.

## Command line

```bash
media-context-analysis episode.mkv -o events.json --diagnostics
python -m media_context_analysis episode.mkv --threads 2 --timeout 1800
```

Exit codes: `0` success, `2` usage or invalid options, `3` missing or unusable media, `4` FFmpeg or FFprobe failure, `5` timeout, `6` cancellation, `1` any other error. JSON is written to stdout or to `--output`. The parent directory of `--output` must already exist.

## Documentation

- [Public API](docs/api.md)
- [JSON schema 2.0.0](docs/schema.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
