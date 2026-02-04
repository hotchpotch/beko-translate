from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("beko-translate")
    except PackageNotFoundError:
        return "0.0.0"


def main() -> int:
    from .cli import main as _main

    return _main()


__all__ = ["get_version", "main"]
