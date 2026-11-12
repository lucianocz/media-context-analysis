# JSON schema 2.0.0

`schema_version` is `MAJOR.MINOR.PATCH` of non-negative ASCII integers without leading zeroes. Package version `0.2.0` emits schema `2.0.0`. Readers accept another `2.x.y` document and reject a different major, a shorter string, or a pre-release suffix.

`to_json()` preserves this key order, uses LF line endings, and rejects non-finite numbers. There is no wall-clock field or source-file metadata. A repeated run matches when the file bytes, the FFmpeg and FFprobe builds, the NumPy build, the platform, and the options match. `ffmpeg_version`, `ffprobe_version`, and `video_threads` follow that environment.

## Top level

1. `schema_version` string
2. `provenance` object
3. `timeline` array of events
4. `groups` array
5. `diagnostics` object or `null`

The former `media` object and `MediaMetadata` Python model are removed. Schema 1 documents are rejected; there is no migration layer.

## provenance

`library_name` is `media-context-analysis`. `library_version` is the package version. `detector_baseline` is the detector revision recorded for this package. `detector_note` describes timestamp and grouping behavior. `feature_version` matches the detector parameters.

`subtitles_used` and `audio_deletes_events` are always false. `ffmpeg_version` and `ffprobe_version` are the first line of each tool's `-version` output, or null. `video_stream_index` is the effective FFprobe video stream index. `video_threads` is the requested video thread count. `audio_threads` is `1`. `visual_filter` is the sample filter without `showinfo`. `detector_params` is the parameter object used for the run, including unused subtitle fields.

## timeline event

`id` (`e0001` upward on the compact timeline), `shot_index`, `time_s`, `time`, `end_s`, `end`, `category`, `priority`, `reasons`, `selected`, `suggested_context_break`, `forces_translation_segment`, `uncertainty`, `evidence`, `continuity`, `scores`, `representatives`.

Compact categories are `brief_visual_insert`, `visual_departure`, `sharp_stable_cut`, and `palette_shift`. Priority is `high`, `medium`, or `low`. `forces_translation_segment` is always false. `selected` is true for compact timeline rows.

`continuity` has `framing_match`, `color_match`, `acoustic_vote` (`glue`, `split`, `neutral`, `unavailable`), `short_return`, and `notes`. Flags are computed from unrounded scores. `notes` is explanatory text.

`scores` stores four-decimal numbers: `best_place_similarity`, `best_framing_similarity`, `adjacent_place_similarity`, `change_score`, `luma`, `luma_step`, `shot_duration_s`, `place_match_age_s`, `audio_score`, `audio_vote`, `near_black`. Missing numeric scores are null.

## representatives

`pre_transition_s`, `pre_transition`, `pre_transition_basis`, `context_reference_s`, `context_reference`, `context_reference_basis`, `transition_s`, `transition`, `after_s`, `after`, `after_basis`, `return_s`, `return`.

Times are seconds, rounded to milliseconds for display, plus an `HH:MM:SS.mmm` string. That rounding is not the sampling resolution. Visual samples are taken at 4 fps, and `time_s` is the first sample of the new shot on that grid, so the cut can be up to 0.25 s earlier than `time_s`.

`context_reference_basis` is `previous_shot_midpoint`. `return` is present for `brief_visual_insert` and null otherwise. Basis strings name the rule that chose the sample, for example `last_sample_before_cut`, `latest_stable_sample_within_1s`, `earliest_stable_sample_within_1s`, `fixed_offset_0_5s`, or `short_shot_midpoint`.

## groups

`id` (`g0001` upward), `kind` (`brief_insert_return`), `event_ids`, `evidence`, `creates_translation_segment`. The last field is always false. A group does not delete or move events. An insert with no related event produces no group.

## diagnostics

Present only when requested. Otherwise the field is null.

`shot_count`, `place_threshold`, `place_threshold_stats`, `thresholds`, `audio_band`, `rules`, `shot_stats`, `family_counts`, `family_times`, `compact_before_dedupe`, `raw_candidates`.

`raw_candidates` uses the same event shape. Ids are assigned to the raw rows first (`r0001` upward) and then to the compact timeline (`e0001` upward). Those two calls share the event dictionaries. A raw row that stays on the timeline is rewritten from `r####` to the compact `e` id. Dropped rows keep the `r` id. The numeric suffix is not an index shared by the two lists. `palette_break` can appear here and does not appear on the compact timeline by itself.

Timeline `category`, `priority`, `acoustic_vote`, and group `kind` are a closed set. A reader rejects an unknown value. Diagnostic object keys are informational.
