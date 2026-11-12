# Agent notes

These rules apply to every coding agent in this repository.

- Follow Semantic Versioning for the Python package and for the JSON schema. The schema version is independent of the package version.
- Use Conventional Commits. Do not add `Co-authored-by` trailers.
- When public behavior changes, update `README.md`, `docs/api.md`, `docs/schema.md`, and `CHANGELOG.md`.
- Keep development progress, research findings, architectural decisions, benchmarks, audits, and unresolved issues in `.dev/`.
- Keep `.dev/README.md` as the index of that internal knowledge.
- Do not commit `.dev/`, local samples, copyrighted media, generated frames, caches, or secrets.
- Preserve the validated detector baseline unless an algorithm change is explicitly requested and regression-tested.
- Do not add application-specific dependencies, image extraction, subtitle analysis, or translation.
- Add or update tests for behavioral changes.
- Before committing, run the relevant tests, Ruff, mypy, and the packaging check.
- Keep source code and documentation in English.
- Do not push, tag, publish, or deploy without explicit authorization.
- Attribute the project only as Luciano (`lucianocz`). Do not add another name or an email address to tracked files.
