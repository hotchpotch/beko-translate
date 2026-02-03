# CAT-Translate CLI - Agent Notes

## Overview
This project provides a small MLX-based translation CLI for CAT-Translate.
The primary entrypoint is the `cat-translate` command (installed via `uv run`).

## Development Workflow
- Install deps: `uv sync` (or `uv add --dev ...` when adding dev tools)
- Run CLI: `uv run cat-translate --text "Hello" --input-lang en --output-lang ja`
- PDF CLI: `uv run cat-translate-pdf paper.pdf --input en --output ja`
- Interactive (default when no args + tty): `uv run cat-translate`
- Streaming (one-shot): `uv run cat-translate --stream --server never --text "こんにちは"`
- Server mode:
  - Start: `uv run cat-translate server start`
  - Status: `uv run cat-translate server status`
  - Stop: `uv run cat-translate server stop`
  - Use server automatically: `uv run cat-translate --server auto`
- Run tests + lint + typecheck: `uv run tox` (includes MLX integration tests)
  - Lint only: `uv run ruff check .`
  - Typecheck only: `uv run ty check`
  - Tests only: `uv run pytest`
  - MLX integration tests: `RUN_MLX_INTEGRATION=1 uv run pytest -m integration`

## Key Directories
- `cat_translate/`
  - `cli.py`: MLX-only translation CLI implementation
  - `pdf_cli.py`: PDF translation CLI (pdf2zh_next + cat-translate)
  - `__init__.py`: exposes `main`
- `scripts/`
  - `to_mlx.py`: convert HF models to MLX (q4/q8) using `mlx_lm.convert`
  - `translate.py`: example script for HF/MLX translation
- `tests/`
  - unit + optional MLX integration tests
- `output/`
  - local MLX model outputs (when converting locally)
- `.venv/`, `.tox/`
  - local dev environments

## Models
- Default MLX model (remote): `hotchpotch/CAT-Translate-0.8b-mlx-q4`
- Other available MLX repos:
  - `hotchpotch/CAT-Translate-0.8b-mlx-q8`
  - `hotchpotch/CAT-Translate-1.4b-mlx-q4`
  - `hotchpotch/CAT-Translate-1.4b-mlx-q8`

## Notes
- `cat-translate` supports `--text` or stdin input.
- If `--input-lang` and `--output-lang` are omitted, `fast-langdetect` is used to
  detect English/Japanese (k=3) and infer the direction.
- `cat-translate-pdf` defaults to en->ja and uses `hotchpotch/CAT-Translate-1.4b-mlx-q8`.
