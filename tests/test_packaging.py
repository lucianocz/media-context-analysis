"""Build a wheel and install it where the checkout is not importable."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

ROOT = Path(__file__).resolve().parents[1]


def test_wheel_and_sdist_install_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    out.mkdir()
    _run([sys.executable, "-m", "build", "--outdir", str(out)], cwd=ROOT)
    artifacts = sorted(out.iterdir())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    assert len(wheels) == 1
    assert len(sdists) == 1
    _run([sys.executable, "-m", "twine", "check", str(wheels[0]), str(sdists[0])])
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    assert "media_context_analysis/__init__.py" in names
    assert "media_context_analysis/py.typed" in names
    assert not any(name.endswith(".mkv") for name in names)
    with tarfile.open(sdists[0]) as archive:
        packaged = archive.getnames()
    assert not any(name.endswith(".mkv") for name in packaged)
    assert any(name.endswith("README.md") for name in packaged)
    assert any(name.endswith("SECURITY.md") for name in packaged)
    assert any(name.endswith("LICENSE") for name in packaged)
    assert not any(name.endswith("PRE_RELEASE_AUDIT.md") for name in packaged)

    subprocess.run([sys.executable, "-m", "venv", str(tmp_path / "venv")], check=True)
    python = _venv_python(tmp_path / "venv")
    _run([str(python), "-m", "pip", "install", str(wheels[0])])
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    probe = (
        "import media_context_analysis as package; "
        "from media_context_analysis import AnalysisResult, analyze; "
        "assert package.__version__ == '0.1.0'; "
        "assert callable(analyze); "
        "assert AnalysisResult.__name__ == 'AnalysisResult'"
    )
    _run([str(python), "-c", probe], env=env)
    completed = _run([str(python), "-m", "media_context_analysis", "--help"], env=env)
    assert completed.returncode == 0


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{command[0]} exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed
