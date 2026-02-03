from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _load_toml() -> Any:
    try:
        import tomllib  # type: ignore[attr-defined]

        return tomllib
    except ModuleNotFoundError:
        import tomli  # type: ignore[import-not-found]

        return tomli


def read_project_version(pyproject_path: Path) -> str:
    toml = _load_toml()
    data = toml.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version", "")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("project.version is missing in pyproject.toml")
    return version.strip()


def release_log_has_entry(log_path: Path, version: str) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8")
    header_pattern = re.compile(rf"^##\s+{re.escape(version)}\s*$", re.MULTILINE)
    match = header_pattern.search(text)
    if not match:
        return False
    start = match.end()
    next_header = re.compile(r"^##\s+", re.MULTILINE).search(text, start)
    end = next_header.start() if next_header else len(text)
    body = text[start:end]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def clean_dist(dist_path: Path) -> None:
    if dist_path.exists():
        shutil.rmtree(dist_path)


def build_publish_files(dist_path: Path, version: str) -> list[Path]:
    pattern = f"neko_translate-{version}*"
    return sorted(dist_path.glob(pattern))


def ensure_clean_worktree(ignore_warnings: bool) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    changes = result.stdout.strip()
    if changes and not ignore_warnings:
        message = (
            "Uncommitted changes detected. Commit or stash them before releasing, "
            "or rerun with --ignore-git-warnings.\n\n"
            f"{changes}"
        )
        raise RuntimeError(message)


def ensure_lock_up_to_date() -> None:
    result = subprocess.run(
        ["uv", "lock", "--check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        output = result.stdout.strip()
        message = "uv.lock is out of date. Run `uv lock` and commit uv.lock before releasing."
        if output:
            message = f"{message}\n\n{output}"
        raise RuntimeError(message)


def release(pyproject_path: Path, log_path: Path, ignore_git_warnings: bool) -> None:
    ensure_clean_worktree(ignore_git_warnings)
    ensure_lock_up_to_date()
    version = read_project_version(pyproject_path)
    if not release_log_has_entry(log_path, version):
        raise RuntimeError(
            f"Release log entry for version {version} is missing or empty in {log_path}."
        )
    token = os.environ.get("PYPI_TOKEN")
    if not token:
        raise RuntimeError("PYPI_TOKEN is not set.")
    run(["uv", "run", "--extra", "dev", "tox"])
    dist_path = Path("dist")
    clean_dist(dist_path)
    run(["uv", "build"])
    publish_files = build_publish_files(dist_path, version)
    if not publish_files:
        raise RuntimeError(f"No artifacts found for version {version} in {dist_path}.")
    run(["uv", "publish", "--token", token, *[str(p) for p in publish_files]])
    tag = f"v{version}"
    run(["git", "tag", "-f", "-a", tag, "-m", f"Release {tag}"])
    run(["git", "push", "-f", "origin", tag])


def main() -> int:
    parser = argparse.ArgumentParser(description="Release helper for neko-translate.")
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run tests, build, publish, and tag the current version.",
    )
    parser.add_argument(
        "--ignore-git-warnings",
        action="store_true",
        help="Allow releasing with uncommitted git changes.",
    )
    args = parser.parse_args()
    if args.release:
        release(Path("pyproject.toml"), Path("release-log.md"), args.ignore_git_warnings)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
