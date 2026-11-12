# Security

Report a vulnerability through GitHub private vulnerability reporting on [lucianocz/media-context-analysis](https://github.com/lucianocz/media-context-analysis/security/advisories/new). Do not open a public issue for an unfixed security report.

This library runs FFmpeg and FFprobe on a local file the caller passes. It does not upload media and it does not open a network connection of its own. The JSON result contains the file name and the FFmpeg version strings. It does not contain the absolute path.

A timeout or cancellation stops the FFmpeg process this library started. On Windows that is the direct child. On Linux it is the process group. A parent process that is hard-killed from outside can leave FFmpeg running. There is no Windows Job Object.

Pass only media you intend to decode. The tools are resolved from `PATH` unless the caller sets `ffmpeg` and `ffprobe`.
