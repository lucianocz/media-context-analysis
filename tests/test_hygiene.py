"""The tree that would be published has no credentials, mailboxes, or private paths."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "htmlcov",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".dev",
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cfg"}
EXTRA_NAMES = {".gitignore", ".gitattributes", ".editorconfig", "LICENSE", "NOTICE"}
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+" + "@" + r"[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PRIVATE = (
    "C:" + "\\Users",
    "C:" + "/Users",
    "D:" + "\\",
    "D:" + "/",
)
SECRETS = ("AKIA", "BEGIN PRIVATE KEY", "ghp_", "github_pat_")


def _files() -> list[Path]:
    found: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "test_hygiene.py":
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in EXTRA_NAMES:
            found.append(path)
    return found


def test_tracked_text_has_no_private_markers() -> None:
    failures: list[str] = []
    for path in _files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if EMAIL.search(text):
            failures.append(f"email in {path.relative_to(ROOT)}")
        for marker in PRIVATE:
            if marker in text:
                failures.append(f"private path marker in {path.relative_to(ROOT)}")
                break
        for marker in SECRETS:
            if marker in text:
                failures.append(f"secret marker in {path.relative_to(ROOT)}")
                break
    assert failures == []


def test_package_does_not_write_images_or_read_subtitle_text() -> None:
    package = ROOT / "src" / "media_context_analysis"
    banned = ("imwrite", "Image.open", "cv2", "parse_ass", ".jpg", ".png", "tmdb")
    for path in package.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} contains {token}"


def test_video_containers_are_gitignored() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.mkv" in text
    assert "samples/*.mkv" in text
