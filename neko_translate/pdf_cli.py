#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from . import cli as neko_cli
from .translation_models import resolve_model_alias

DEFAULT_MODEL = "hotchpotch/CAT-Translate-1.4b-mlx-q8"
DEFAULT_INPUT_LANG = "en"
DEFAULT_OUTPUT_LANG = "ja"
DEFAULT_AUTO_TRANSLATE_DIR = Path.home() / "Downloads"

PDF2ZH_COMMAND = os.environ.get("PDF2ZH_COMMAND", "uvx pdf2zh_next")
NEKO_TRANSLATE_COMMAND = os.environ.get("NEKO_TRANSLATE_COMMAND")

LANG_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "日本語": "ja",
    "auto": "auto",
}

DEFAULT_PDF2ZH_ARGS = [
    "--watermark-output-mode",
    "no_watermark",
    "--use-alternating-pages-dual",
    "--qps",
    "4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate PDFs with pdf2zh_next via neko-translate.",
    )
    parser.add_argument(
        "pdf",
        nargs="*",
        type=Path,
        help="PDF files to translate (positional).",
    )
    parser.add_argument(
        "-t",
        "--target-pdf",
        nargs="+",
        action="extend",
        type=Path,
        default=[],
        help="One or more PDF files to translate (repeatable).",
    )
    parser.add_argument(
        "--output-pdf",
        "--output-file",
        dest="output_pdf",
        type=Path,
        default=None,
        help="Output PDF file path (only valid with a single input).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for translated PDFs. "
            "If omitted, outputs are written next to each input PDF."
        ),
    )
    parser.add_argument(
        "-a",
        "--auto-translate-dir",
        type=Path,
        default=DEFAULT_AUTO_TRANSLATE_DIR,
        help="Directory to scan for PDFs when no inputs are specified.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_LANG,
        help="Input language (en/ja/auto). Default: en.",
    )
    parser.add_argument(
        "--output",
        dest="output_lang",
        default=DEFAULT_OUTPUT_LANG,
        help="Output language (en/ja). Default: ja.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX model repo (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--socket",
        default=None,
        help="Unix domain socket path for neko-translate server.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Server log file path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust remote code when loading tokenizers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=neko_cli.DEFAULT_TEMPERATURE,
        help="Sampling temperature. 0 disables sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=neko_cli.DEFAULT_TOP_P,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=neko_cli.DEFAULT_TOP_K,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template even if the tokenizer provides one.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing translated files.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print planned commands and outputs without running translation.",
    )
    parser.add_argument(
        "--no-strip-paren-index",
        action="store_true",
        help='Keep "(1)"-style suffixes instead of stripping them.',
    )
    parser.add_argument(
        "--no-dual",
        action="store_true",
        help="Disable alternating bilingual pages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def normalize_lang(value: str, *, allow_auto: bool) -> str:
    key = value.strip().lower()
    if allow_auto and key == "auto":
        return "auto"
    normalized = LANG_ALIASES.get(key)
    if normalized is None or normalized == "auto":
        raise SystemExit(
            f"Unsupported language '{value}'. Supported: en/ja" + (
                "/auto" if allow_auto else ""
            )
        )
    return normalized


def ensure_command_available(command: str) -> None:
    exe = shlex.split(command)[0]
    if shutil.which(exe) is None:
        sys.stderr.write(f"Command not found in PATH: {exe}\n")
        sys.exit(1)


def load_pdf2zh_args() -> list[str]:
    env_args = os.environ.get("PDF2ZH_ARGS")
    if env_args:
        return shlex.split(env_args)
    return list(DEFAULT_PDF2ZH_ARGS)


def adjust_pdf2zh_args(args: list[str], *, use_dual: bool) -> list[str]:
    if use_dual:
        return args
    return [arg for arg in args if arg != "--use-alternating-pages-dual"]


def _remove_flag(args: list[str], flag: str, has_value: bool) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = has_value
            continue
        cleaned.append(arg)
    return cleaned


def build_pdf2zh_args(
    base_args: list[str],
    *,
    lang_in: str,
    lang_out: str,
    cli_command: str,
) -> list[str]:
    args = list(base_args)
    args = _remove_flag(args, "--lang-in", True)
    args = _remove_flag(args, "--lang-out", True)
    args = _remove_flag(args, "--clitranslator", False)
    args = _remove_flag(args, "--clitranslator-command", True)

    args.append("--clitranslator")
    args += [
        "--lang-in",
        lang_in,
        "--lang-out",
        lang_out,
        "--clitranslator-command",
        cli_command,
    ]
    return args


def strip_trailing_paren_index(stem: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", stem)


def normalize_base(stem: str, strip_paren_index: bool) -> str:
    return strip_trailing_paren_index(stem) if strip_paren_index else stem


def find_pdfs(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.pdf")):
        yield path


def select_latest_by_base(
    paths: Iterable[Path], strip_paren_index: bool
) -> List[Tuple[Path, str, str]]:
    latest: dict[str, Tuple[Path, float, str, str]] = {}
    for path in paths:
        base_raw = path.stem
        base_clean = normalize_base(base_raw, strip_paren_index)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        key = base_clean
        if key not in latest or mtime > latest[key][1]:
            latest[key] = (path, mtime, base_raw, base_clean)
    return [(p, base_raw, base_clean) for p, _mt, base_raw, base_clean in latest.values()]


def is_generated_output(name: str) -> bool:
    return any(
        token in name
        for token in (
            ".ja.",
            ".en.",
            ".translated",
            "no_watermark",
            ".mono.",
            ".dual.",
        )
    )


def cleanup_extras(
    base_prefixes: Iterable[str],
    output_dir: Path,
    keep: Path,
    protected: Iterable[Path],
) -> None:
    prefixes = tuple(base_prefixes)
    if not prefixes:
        return
    protected_paths = set()
    for path in protected:
        try:
            protected_paths.add(path.resolve())
        except FileNotFoundError:
            continue
    for path in output_dir.glob("*.pdf"):
        if not path.name.startswith(prefixes):
            continue
        if not is_generated_output(path.name):
            continue
        try:
            if path.resolve() == keep.resolve():
                continue
            if path.resolve() in protected_paths:
                continue
        except FileNotFoundError:
            continue
        try:
            path.unlink()
        except OSError as exc:
            sys.stderr.write(f"Warning: failed to remove {path}: {exc}\n")


def pick_output_file(paths: Iterable[Path], stem: str) -> Optional[Path]:
    pdfs = [p for p in paths if p.suffix.lower() == ".pdf"]
    if not pdfs:
        return None

    def priority(p: Path) -> tuple:
        name = p.name
        return (
            0 if "no_watermark" in name else 1,
            0 if "dual" in name else 1,
            0 if stem in name else 1,
            -p.stat().st_mtime,
        )

    pdfs.sort(key=priority)
    return pdfs[0]


def translate_pdf(
    pdf_path: Path,
    output_dir: Path,
    target_path: Path,
    base_prefix: str,
    base_clean: str,
    pdf2zh_args: list[str],
) -> bool:
    cmd = (
        shlex.split(PDF2ZH_COMMAND)
        + pdf2zh_args
        + ["--output", str(output_dir), str(pdf_path)]
    )
    print(f"[translate] {pdf_path.name} -> {target_path.name}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        sys.stderr.write(
            f"pdf2zh_next failed for {pdf_path.name} (exit {result.returncode}).\n"
        )
        return False

    output_file = pick_output_file(output_dir.glob("*.pdf"), pdf_path.stem)
    if output_file is None:
        sys.stderr.write(
            f"Translated output for {pdf_path.name} not found in {output_dir}.\n"
        )
        return False

    try:
        output_file.replace(target_path)
    except OSError as exc:
        sys.stderr.write(f"Failed to move output {output_file} -> {target_path}: {exc}\n")
        return False

    cleanup_extras(
        [base_prefix, base_clean],
        output_dir,
        target_path,
        protected=[pdf_path],
    )
    return True


def build_neko_translate_command(
    *,
    model: str,
    socket_path: Path,
    log_path: Path,
    lang_in: str,
    lang_out: str,
    trust_remote_code: bool,
    no_chat_template: bool,
) -> str:
    if NEKO_TRANSLATE_COMMAND:
        base_command = NEKO_TRANSLATE_COMMAND
    else:
        base_command = f"{sys.executable} -m neko_translate.cli"

    command = shlex.split(base_command)
    command += [
        "--server",
        "always",
        "--socket",
        str(socket_path),
        "--log-file",
        str(log_path),
        "--model",
        model,
        "--input-lang",
        lang_in,
        "--output-lang",
        lang_out,
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if no_chat_template:
        command.append("--no-chat-template")

    return " ".join(shlex.quote(part) for part in command)


def wait_for_server_stop(
    socket_path: Path,
    state_path: Path,
    timeout: float = 10.0,
) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if not neko_cli._get_server_status(socket_path, state_path=state_path):
            return True
        time.sleep(0.2)
    return False


def ensure_neko_translate_server(
    *,
    model: str,
    socket_path: Path,
    log_path: Path,
    trust_remote_code: bool,
    verbose: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    no_chat_template: bool,
) -> None:
    state_path = neko_cli.resolve_state_path(None)
    status = neko_cli._get_server_status(socket_path, state_path=state_path)
    if status and status.get("model") != model:
        sys.stderr.write(
            f"[WARN] Server running with model {status.get('model')}; restarting with {model}.\n"
        )
        response = neko_cli._send_request(socket_path, {"type": "stop"}, timeout=2.0)
        if not response or not response.get("ok"):
            raise SystemExit("Failed to stop existing server.")
        if not wait_for_server_stop(socket_path, state_path):
            raise SystemExit("Server did not stop in time.")
        status = None

    if status:
        if verbose:
            sys.stderr.write(f"[INFO] Using existing server (model={status.get('model')}).\n")
        return

    started = neko_cli._start_server(
        model=model,
        socket_path=socket_path,
        log_path=log_path,
        state_path=state_path,
        trust_remote_code=trust_remote_code,
        verbose=verbose,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        no_chat_template=no_chat_template,
    )
    if not started:
        raise SystemExit("Failed to start neko-translate server.")
    if verbose:
        sys.stderr.write(
            f"[INFO] Server started (model={started.get('model')}, socket={socket_path}).\n"
        )


def detect_pdf_language(pdf_path: Path) -> str:
    text = extract_text_for_lang_detect(pdf_path)
    if not text:
        raise SystemExit(f"Failed to extract text for language detection: {pdf_path}")
    from fast_langdetect import detect

    results = detect(text, k=3, model="auto")
    if isinstance(results, dict):
        results_iter: Iterable[dict[str, str]] = [results]
    else:
        results_iter = results

    for item in results_iter:
        lang = item.get("lang")
        if isinstance(lang, str) and lang in {"en", "ja"}:
            return lang
    raise SystemExit(f"Could not detect English/Japanese for: {pdf_path}")


def extract_text_for_lang_detect(pdf_path: Path, max_chars: int = 4000) -> str:
    if shutil.which("pdftotext") is not None:
        try:
            result = subprocess.run(
                ["pdftotext", "-f", "1", "-l", "3", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                check=True,
            )
            text = result.stdout
            return text[:max_chars].strip()
        except subprocess.CalledProcessError:
            pass

    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise SystemExit(
            "pypdf is required for auto detection when pdftotext is unavailable. "
            "Install it with: uv add --dev pypdf"
        ) from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""

    chunks: list[str] = []
    for page in reader.pages[:3]:
        text = page.extract_text() or ""
        if text:
            chunks.append(text)
        if sum(len(c) for c in chunks) >= max_chars:
            break
    return "\n".join(chunks)[:max_chars].strip()


def main() -> int:
    args = parse_args()

    if args.output_pdf and args.output_dir:
        sys.stderr.write("--output-pdf and --output-dir cannot be used together.\n")
        return 1

    input_lang = normalize_lang(args.input, allow_auto=True)
    output_lang = normalize_lang(args.output_lang, allow_auto=False)

    targets_input = [path.expanduser() for path in (args.pdf + args.target_pdf)]
    strip_paren_index = not args.no_strip_paren_index

    if targets_input:
        missing = [path for path in targets_input if not path.is_file()]
        if missing:
            sys.stderr.write("The following target PDFs were not found:\n")
            for path in missing:
                sys.stderr.write(f"  - {path}\n")
            return 1
        targets: List[Tuple[Path, str, str]] = []
        for pdf_path in targets_input:
            base_raw = pdf_path.stem
            base_clean = normalize_base(base_raw, strip_paren_index)
            targets.append((pdf_path, base_raw, base_clean))
    else:
        if args.output_pdf:
            sys.stderr.write("--output-pdf requires exactly one input PDF.\n")
            return 1
        auto_dir = args.auto_translate_dir.expanduser()
        if not auto_dir.is_dir():
            sys.stderr.write(f"Auto-translate directory not found: {auto_dir}\n")
            return 1
        candidates = list(find_pdfs(auto_dir))
        targets = select_latest_by_base(candidates, strip_paren_index)

    if not targets:
        print("No PDFs found to translate.")
        return 0

    if args.output_pdf and len(targets) != 1:
        sys.stderr.write("--output-pdf is only valid with a single input PDF.\n")
        return 1

    ensure_command_available(PDF2ZH_COMMAND)

    socket_path = neko_cli.resolve_socket_path(args.socket)
    log_path = neko_cli.resolve_log_path(args.log_file)
    neko_cli.ensure_directory(socket_path.parent)
    neko_cli.ensure_directory(log_path.parent)
    neko_cli.configure_logging(args.verbose)

    model = resolve_model_alias(args.model, DEFAULT_MODEL)

    if not args.dry_run:
        ensure_neko_translate_server(
            model=model,
            socket_path=socket_path,
            log_path=log_path,
            trust_remote_code=args.trust_remote_code,
            verbose=args.verbose,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            no_chat_template=args.no_chat_template,
        )

    pdf2zh_base_args = adjust_pdf2zh_args(load_pdf2zh_args(), use_dual=not args.no_dual)

    translated: List[Tuple[Path, Path]] = []
    skipped: List[Tuple[Path, Path]] = []
    failed: List[Path] = []
    planned: List[Tuple[Path, Path, str]] = []
    same_lang: List[Tuple[Path, str]] = []

    for pdf_path, base_raw, base_clean in targets:
        lang_in = input_lang
        lang_out = output_lang
        if lang_in == "auto":
            lang_in = detect_pdf_language(pdf_path)
        if lang_in == lang_out:
            sys.stderr.write(
                f"[skip] {pdf_path.name}: input and output are both '{lang_in}'.\n"
            )
            same_lang.append((pdf_path, lang_in))
            continue

        output_code = "ja" if lang_out == "ja" else "en"

        if args.output_pdf:
            dest = args.output_pdf.expanduser()
            dest_dir = dest.parent
            dest_name = dest.name
        else:
            dest_dir = args.output_dir.expanduser() if args.output_dir else pdf_path.parent
            dest_name = f"{base_clean}.{output_code}.pdf"
            dest = dest_dir / dest_name

        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not args.force:
            print(f"[skip] {dest.name} already exists.")
            skipped.append((pdf_path, dest))
            continue

        if dest.exists():
            print(f"[retranslate] Overwriting {dest.name}.")

        cli_command = build_neko_translate_command(
            model=model,
            socket_path=socket_path,
            log_path=log_path,
            lang_in=lang_in,
            lang_out=lang_out,
            trust_remote_code=args.trust_remote_code,
            no_chat_template=args.no_chat_template,
        )
        pdf2zh_args = build_pdf2zh_args(
            pdf2zh_base_args,
            lang_in=lang_in,
            lang_out=lang_out,
            cli_command=cli_command,
        )

        cmd = (
            shlex.split(PDF2ZH_COMMAND)
            + pdf2zh_args
            + ["--output", str(dest_dir), str(pdf_path)]
        )
        cmd_display = " ".join(shlex.quote(part) for part in cmd)

        if args.dry_run:
            print(f"[dry-run] {cmd_display}")
            planned.append((pdf_path, dest, cmd_display))
            continue

        success = translate_pdf(
            pdf_path,
            dest_dir,
            dest,
            base_raw,
            base_clean,
            pdf2zh_args,
        )
        if success:
            print(f"[done] {dest}")
            translated.append((pdf_path, dest))
        else:
            print(f"[fail] {pdf_path.name}")
            failed.append(pdf_path)

    print("\nSummary:")
    if args.dry_run:
        if planned:
            print("Planned translations:")
            for src, out, cmd_display in planned:
                print(f"  {src} -> {out}")
                print(f"    {cmd_display}")
        if skipped:
            print("Skipped (already exists):")
            for src, out in skipped:
                print(f"  {src} -> {out}")
        if same_lang:
            print("Skipped (same language):")
            for src, lang in same_lang:
                print(f"  {src} ({lang})")
        return 0

    if translated:
        print("Translated:")
        for src, out in translated:
            print(f"  {src} -> {out}")
    if skipped:
        print("Skipped (already exists):")
        for src, out in skipped:
            print(f"  {src} -> {out}")
    if same_lang:
        print("Skipped (same language):")
        for src, lang in same_lang:
            print(f"  {src} ({lang})")
    if failed:
        print("Failed:")
        for src in failed:
            print(f"  {src}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
