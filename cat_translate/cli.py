#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import functools
import os
import sys
import tempfile
import warnings
from typing import Any, Iterable, Iterator

PROMPT_TEMPLATE = "Translate the following {src_lang} text into {tgt_lang}.\n\n{src_text}"
DEFAULT_MLX_MODEL = "hotchpotch/CAT-Translate-0.8b-mlx-q4"
LANG_CODE_MAP = {
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "日本語": "ja",
    "en": "en",
    "eng": "en",
    "english": "en",
}
LANG_NAME_MAP = {
    "ja": "Japanese",
    "en": "English",
}
SUPPORTED_LANGS = {"ja", "en"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate with CAT-Translate (MLX only).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MLX_MODEL,
        help=(
            "MLX model repo or local directory "
            "(default: hotchpotch/CAT-Translate-0.8b-mlx-q4)."
        ),
    )
    parser.add_argument(
        "--text",
        help="Input text. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--input-lang",
        help="Input language (en/ja).",
    )
    parser.add_argument(
        "--output-lang",
        help="Output language (en/ja).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 disables sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizers.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template even if the tokenizer provides one.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and download progress output.",
    )
    return parser.parse_args()


def read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if sys.stdin.isatty():
        raise SystemExit("Provide --text or pipe input via stdin.")
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input text provided via stdin.")
    return data


def normalize_lang(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower()
    normalized = LANG_CODE_MAP.get(key)
    if normalized is None:
        raise SystemExit(
            f"Unsupported language '{value}'. Supported: {sorted(SUPPORTED_LANGS)}"
        )
    return normalized


def detect_lang(text: str) -> str:
    from fast_langdetect import detect

    results = detect(text, k=3, model="auto")
    if isinstance(results, dict):
        results_iter: Iterable[dict[str, Any]] = [results]
    else:
        results_iter = results

    for item in results_iter:
        lang = item.get("lang")
        if isinstance(lang, str) and lang in SUPPORTED_LANGS:
            return lang

    raise SystemExit("Could not detect language as English or Japanese.")


def resolve_languages(args: argparse.Namespace, text: str) -> tuple[str, str]:
    input_lang = normalize_lang(args.input_lang)
    output_lang = normalize_lang(args.output_lang)

    if input_lang is None and output_lang is None:
        input_lang = detect_lang(text)
        output_lang = "ja" if input_lang == "en" else "en"
        if getattr(args, "verbose", False):
            sys.stderr.write(f"[INFO] Detected {input_lang} -> {output_lang}\n")
        return input_lang, output_lang

    if input_lang is None and output_lang is not None:
        input_lang = "ja" if output_lang == "en" else "en"
        return input_lang, output_lang

    if input_lang is not None and output_lang is None:
        output_lang = "ja" if input_lang == "en" else "en"
        return input_lang, output_lang

    if input_lang not in SUPPORTED_LANGS or output_lang not in SUPPORTED_LANGS:
        raise SystemExit("Only English and Japanese are supported.")

    if input_lang == output_lang:
        raise SystemExit("Input and output languages must be different.")

    return input_lang, output_lang


@functools.lru_cache(maxsize=4)
def _load_model(model: str, trust_remote_code: bool):
    from mlx_lm import load

    return load(
        model,
        tokenizer_config={
            "trust_remote_code": True if trust_remote_code else None
        },
    )


def configure_logging(verbose: bool) -> None:
    if verbose:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        disable = getattr(hf_logging, "disable_progress_bars", None)
        if callable(disable):
            disable()
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*mx\\.metal\\.device_info.*",
    )


@contextlib.contextmanager
def silence_stderr(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    original_fd = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            os.dup2(tmp.fileno(), 2)
            try:
                yield
            except Exception:
                os.dup2(original_fd, 2)
                tmp.seek(0)
                data = tmp.read()
                if data:
                    sys.stderr.write(data.decode(errors="replace"))
                raise
            finally:
                os.dup2(original_fd, 2)
    finally:
        os.close(original_fd)


def run_mlx(prompt: str, args: argparse.Namespace) -> str:
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    loaded = _load_model(args.model, args.trust_remote_code)
    model = loaded[0]
    tokenizer = loaded[1]

    messages = [{"role": "user", "content": prompt}]
    if not args.no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt_text = prompt

    sampler = None
    if (args.temperature and args.temperature > 0) or (
        args.top_p and args.top_p < 1.0
    ) or (args.top_k and args.top_k > 0):
        sampler = make_sampler(
            temp=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    gen_kwargs: dict[str, Any] = {"max_tokens": args.max_new_tokens}
    if sampler is not None:
        gen_kwargs["sampler"] = sampler

    return generate(model, tokenizer, prompt_text, **gen_kwargs)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    text = read_text(args)
    input_lang, output_lang = resolve_languages(args, text)

    prompt = PROMPT_TEMPLATE.format(
        src_lang=LANG_NAME_MAP[input_lang],
        tgt_lang=LANG_NAME_MAP[output_lang],
        src_text=text,
    )

    with silence_stderr(not args.verbose):
        translation = run_mlx(prompt, args)
    print(translation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
