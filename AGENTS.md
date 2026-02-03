# NEKO-Translate CLI - Agent Notes

## Overview
This project provides a small MLX-based translation CLI for NEKO-Translate.
The primary entrypoint is the `neko-translate` command (installed via `uv run`).

## Development Workflow
- Install deps: `uv sync` (or `uv add --dev ...` when adding dev tools)
- Run CLI: `uv run neko-translate --text "Hello" --input-lang en --output-lang ja`
- PDF CLI: `uv run neko-translate-pdf paper.pdf --input en --output ja`
- Interactive (default when no args + tty): `uv run neko-translate`
- Streaming (one-shot): `uv run neko-translate --stream --server never --text "こんにちは"`
- Server mode:
  - Start: `uv run neko-translate server start`
  - Status: `uv run neko-translate server status`
  - Stop: `uv run neko-translate server stop`
  - Use server automatically: `uv run neko-translate --server auto`
- Run tests + lint + typecheck: `uv run tox` (includes MLX integration tests)
  - Lint only: `uv run ruff check .`
  - Typecheck only: `uv run ty check`
  - Tests only: `uv run pytest`
  - MLX integration tests: `RUN_MLX_INTEGRATION=1 uv run pytest -m integration`

## Key Directories
- `neko_translate/`
  - `cli.py`: MLX-only translation CLI implementation
  - `pdf_cli.py`: PDF translation CLI (pdf2zh_next + neko-translate)
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
  - `mlx-community/HY-MT1.5-1.8B-4bit`
  - `mlx-community/HY-MT1.5-1.8B-8bit`
  - `mlx-community/HY-MT1.5-7B-4bit`
  - `mlx-community/HY-MT1.5-7B-8bit`

## Notes
- `neko-translate` supports `--text` or stdin input.
- If `--input-lang` and `--output-lang` are omitted, `fast-langdetect` is used to
  detect English/Japanese (k=3) and infer the direction.
- `neko-translate` defaults:
  - model: `hotchpotch/CAT-Translate-0.8b-mlx-q4`
  - sampling: temperature=0.0, top_p=0.98, top_k=0
  - no-repeat: ngram=4, window=128
  - trust remote code: enabled by default
  - KV cache: `max_kv_size = prompt_tokens + max_new_tokens` (auto)
- Model aliases:
  - `cat` -> `hotchpotch/CAT-Translate-1.8b-mlx-q8`
  - `plamo` -> `mlx-community/plamo-2-translate`
- `neko-translate-pdf` defaults:
  - en->ja, model `hotchpotch/CAT-Translate-1.4b-mlx-q8`
  - max-new-tokens=1024
  - pdf2zh args include `--qps 4`
- If `neko-translate-pdf` starts a server itself, it will stop it after the
  translations finish. Existing servers are left running.
- Server uses a Unix domain socket under `~/.config/neko-translate/` with
  `neko-translate.sock` and `server.log` by default. Override with
  `--socket/--log-file` or `NEKO_TRANSLATE_SOCKET/NEKO_TRANSLATE_LOG`.
