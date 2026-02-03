#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

PROMPT_TEMPLATE = "Translate the following {src_lang} text into {tgt_lang}.\n\n{src_text}"
DEFAULT_HF_MODEL = "cyberagent/CAT-Translate-1.4b"
DEFAULT_MLX_MODEL = "hotchpotch/CAT-Translate-0.8b-mlx-q4"
LANG_ALIASES = {
    "en": "English",
    "ja": "Japanese",
}


def normalize_lang(value: str) -> str:
    key = value.strip().lower()
    return LANG_ALIASES.get(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate text with NEKO-Translate (Hugging Face or MLX)."
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "mlx"],
        default="hf",
        help="Backend to use: huggingface (hf) or mlx.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model repo/path. Defaults to HF repo or MLX Hub repo based on backend.",
    )
    parser.add_argument("--src-lang", required=True, help="Source language (e.g., ja/en).")
    parser.add_argument("--tgt-lang", required=True, help="Target language (e.g., en/ja).")
    parser.add_argument("--text", required=True, help="Source text to translate.")
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
    return parser.parse_args()


def extract_hf_text(response: list[dict[str, Any]]) -> str:
    if not response:
        return ""
    generated = response[0].get("generated_text")
    if isinstance(generated, list):
        for item in reversed(generated):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return str(generated)
    if isinstance(generated, str):
        return generated
    return str(generated)


def run_hf(prompt: str, args: argparse.Namespace) -> str:
    from transformers import pipeline

    model = args.model or DEFAULT_HF_MODEL
    pipe = pipeline(
        "text-generation",
        model=model,
        trust_remote_code=args.trust_remote_code,
    )
    messages = [{"role": "user", "content": prompt}]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(
            {
                "temperature": args.temperature,
                "do_sample": True,
                "top_p": args.top_p,
            }
        )
        if args.top_k and args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs["do_sample"] = False

    response = pipe(messages, **gen_kwargs)
    return extract_hf_text(response)


def run_mlx(prompt: str, args: argparse.Namespace) -> str:
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    model_path = args.model or DEFAULT_MLX_MODEL
    loaded = load(
        model_path,
        tokenizer_config={
            "trust_remote_code": True if args.trust_remote_code else None
        },
    )
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
    src_lang = normalize_lang(args.src_lang)
    tgt_lang = normalize_lang(args.tgt_lang)
    prompt = PROMPT_TEMPLATE.format(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_text=args.text,
    )

    if args.backend == "hf":
        translation = run_hf(prompt, args)
    else:
        translation = run_mlx(prompt, args)

    print("-" * 20)
    print("Source Text:")
    print(args.text)
    print("Translation:")
    print(translation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
