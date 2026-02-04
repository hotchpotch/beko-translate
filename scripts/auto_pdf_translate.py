#!/usr/bin/env python3
"""Auto-translate PDFs in ~/Downloads using beko-translate-pdf.

Env overrides:
- AUTO_PDF_TRANSLATE_INPUT_DIR
- AUTO_PDF_TRANSLATE_OUTPUT_DIR
- AUTO_PDF_TRANSLATE_MODEL

Defaults:
- Scans ~/Downloads (non-recursive).
- Outputs to ~/Desktop/pdf_translated.
- Uses --model plamo (override with --model).
- Skips PDFs that already have a corresponding .ja.pdf in the output dir.
- Use --input-dir to scan a different directory.
- Use --arxiv to target arXiv-like filenames only.
- Use --days to limit to PDFs modified within the last N days.
  (Only applied when --days is specified.)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

DEFAULT_INPUT_DIR = Path.home() / "Downloads"
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "pdf_translated"
DEFAULT_MODEL = "plamo"

ENV_INPUT_DIR = "AUTO_PDF_TRANSLATE_INPUT_DIR"
ENV_OUTPUT_DIR = "AUTO_PDF_TRANSLATE_OUTPUT_DIR"
ENV_MODEL = "AUTO_PDF_TRANSLATE_MODEL"


def env_or_default_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


def env_or_default_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


DEFAULT_INPUT_DIR = env_or_default_path(ENV_INPUT_DIR, DEFAULT_INPUT_DIR)
DEFAULT_OUTPUT_DIR = env_or_default_path(ENV_OUTPUT_DIR, DEFAULT_OUTPUT_DIR)
DEFAULT_MODEL = env_or_default_str(ENV_MODEL, DEFAULT_MODEL)
ARXIV_PATTERN = re.compile(r"^\d{4}\.\d{4,5}v\d+( \(\d+\))?\.pdf$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-translate PDFs in ~/Downloads using beko-translate-pdf."
    )
    parser.add_argument(
        "-a",
        "--arxiv",
        action="store_true",
        help="Only include PDFs with arXiv-like filenames.",
    )
    parser.add_argument(
        "--days",
        nargs="?",
        const=3,
        default=None,
        type=float,
        help=(
            "Limit to PDFs modified within the last N days. "
            "If provided without a value, defaults to 3. "
            "No date filtering is applied unless --days is specified."
        ),
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory to scan for PDFs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for translated PDFs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Translation model to use (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def find_pdfs(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.pdf")):
        yield path


def filter_arxiv(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if ARXIV_PATTERN.match(path.name)]


def filter_days(paths: Iterable[Path], days: float) -> List[Path]:
    cutoff = time.time() - (days * 24 * 60 * 60)
    result = []
    for path in paths:
        try:
            if path.stat().st_mtime >= cutoff:
                result.append(path)
        except FileNotFoundError:
            continue
    return result


def has_translated_output(pdf_path: Path, output_dir: Path) -> bool:
    """Return True if a corresponding .ja.pdf exists in output_dir."""
    translated_name = f"{pdf_path.stem}.ja.pdf"
    return (output_dir / translated_name).exists()


def validate_args(args: argparse.Namespace) -> int:
    if not args.input_dir.is_dir():
        sys.stderr.write(f"Input directory not found: {args.input_dir}\n")
        return 1
    if args.days is not None and args.days <= 0:
        sys.stderr.write("--days must be a positive number.\n")
        return 1
    return 0


def main() -> int:
    args = parse_args()

    validation_error = validate_args(args)
    if validation_error:
        return validation_error

    paths = list(find_pdfs(args.input_dir))

    if args.arxiv:
        paths = filter_arxiv(paths)

    if args.days is not None:
        paths = filter_days(paths, args.days)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not paths:
        print("No PDFs found to translate.")
        return 0

    targets = [path for path in paths if not has_translated_output(path, output_dir)]
    if not targets:
        print("All PDFs already have translated outputs.")
        return 0

    cmd = [
        "beko-translate-pdf",
        *[str(p) for p in targets],
        "--output-dir",
        str(output_dir),
        "--model",
        str(args.model),
    ]
    print("[run]", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
