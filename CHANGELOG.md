# Changelog

## 0.2.0 - 2026-11-12

- Breaking: JSON schema `2.0.0` removes `media` and the `MediaMetadata` model. Results contain only schema version, provenance, timeline, groups, and diagnostics.
- Select the largest eligible video stream by pixel area, with lowest-index tie breaking. `AnalysisOptions.video_stream_index` selects an explicit FFprobe index; provenance records the effective index.
- Contain Windows subprocesses in kill-on-close Job Objects from creation; reject incompatible parent-job restrictions. Preserve Linux process-group cleanup.
- Bound subprocess cleanup for cancellation, timeout, and failures, including queued output.
- Preserve detector thresholds, scoring, event timestamps, categories, priorities, and grouping.

## 0.1.0 - 2026-11-12

Initial release.

- `analyze` API and `media-context-analysis` command
- JSON schema `1.0.0`, with media metadata and detector provenance
- Visual event timeline and representative seek times
- An audio vote on each visual event. Audio does not add or delete events
- `brief_insert_return` groups that do not remove events
- Timeouts and cancellation during decode
- Portable JSON: file name only, LF line endings, and schema `MAJOR.MINOR.PATCH`
