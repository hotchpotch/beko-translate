#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import warnings
import itertools
from pathlib import Path
from typing import Any, Iterable, Iterator

PROMPT_TEMPLATE = "Translate the following {src_lang} text into {tgt_lang}.\n\n{src_text}"
DEFAULT_MLX_MODEL = "hotchpotch/CAT-Translate-0.8b-mlx-q4"
DEFAULT_SOCKET_NAME = "cat-translate.sock"
DEFAULT_LOG_NAME = "server.log"
DEFAULT_STATE_NAME = "server.json"
SERVER_MODES = ("auto", "always", "never")
DEFAULT_SERVER_MODE = "auto"
DEFAULT_CONFIG_DIR = Path("~/.config/cat-translate").expanduser()
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_NO_CHAT_TEMPLATE = False
STATUS_TIMEOUT = 120.0
SERVER_LISTEN_BACKLOG = 16
TRANSLATE_CONNECT_TIMEOUT = 1.0
TRANSLATE_CONNECT_DEADLINE = 120.0
TRANSLATE_CONNECT_SLEEP = 0.2
LOG_SNIPPET_LIMIT = 20
_REQUEST_COUNTER = itertools.count(1)
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


def build_translate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate with CAT-Translate (MLX only).",
        epilog=(
            "Server commands:\n"
            "  cat-translate server start\n"
            "  cat-translate server stop\n"
            "  cat-translate server status\n"
            "  cat-translate server run\n"
            "Use --socket/--log-file to control server paths."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
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
        default=None,
        help=(
            "Maximum number of new tokens to generate "
            f"(default: {DEFAULT_MAX_NEW_TOKENS})."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. 0 disables sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
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
        default=None,
        help="Disable chat template even if the tokenizer provides one.",
    )
    parser.add_argument(
        "--server",
        choices=SERVER_MODES,
        default=DEFAULT_SERVER_MODE,
        help="Server usage: auto (default), always, or never.",
    )
    parser.add_argument(
        "--socket",
        type=str,
        default=None,
        help="Unix domain socket path (default: ~/.config/cat-translate/cat-translate.sock).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Server log file path (default: ~/.config/cat-translate/server.log).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and download progress output.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens (default: off outside interactive mode).",
    )
    return parser


SERVER_ACTIONS = ("start", "stop", "status", "run")


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage CAT-Translate MLX server. Actions: start, stop, status, run."
        ),
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
        "--socket",
        type=str,
        default=None,
        help="Unix domain socket path (default: ~/.config/cat-translate/cat-translate.sock).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Server log file path (default: ~/.config/cat-translate/server.log).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature. 0 disables sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template even if the tokenizer provides one.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> tuple[str, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "server":
        action = None
        rest = argv[1:]
        if rest and rest[0] in SERVER_ACTIONS:
            action = rest[0]
            rest = rest[1:]
        parser = build_server_parser()
        args = parser.parse_args(rest)
        args.action = action
        return "server", args
    parser = build_translate_parser()
    return "translate", parser.parse_args(argv)


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
    _patch_mx_device_info()
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


def _patch_mx_device_info() -> None:
    try:
        import mlx.core as mx
    except Exception:
        return
    try:
        metal = getattr(mx, "metal", None)
        device_info = getattr(mx, "device_info", None)
        if metal is None or device_info is None:
            return
        setattr(metal, "device_info", device_info)
    except Exception:
        return


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


def ensure_directory(path: Path, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def resolve_socket_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("CAT_TRANSLATE_SOCKET")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_DIR / DEFAULT_SOCKET_NAME


def resolve_log_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("CAT_TRANSLATE_LOG")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_DIR / DEFAULT_LOG_NAME


def resolve_state_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("CAT_TRANSLATE_STATE")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_DIR / DEFAULT_STATE_NAME


def _prepare_prompt(tokenizer: Any, prompt: str, no_chat_template: bool) -> str:
    if not no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def _resolve_generation_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": (
            DEFAULT_MAX_NEW_TOKENS
            if args.max_new_tokens is None
            else args.max_new_tokens
        ),
        "temperature": (
            DEFAULT_TEMPERATURE if args.temperature is None else args.temperature
        ),
        "top_p": (DEFAULT_TOP_P if args.top_p is None else args.top_p),
        "top_k": (DEFAULT_TOP_K if args.top_k is None else args.top_k),
        "no_chat_template": (
            DEFAULT_NO_CHAT_TEMPLATE
            if args.no_chat_template is None
            else args.no_chat_template
        ),
    }


def _build_request_overrides(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.no_chat_template is not None:
        payload["no_chat_template"] = args.no_chat_template
    return payload


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    no_chat_template: bool,
) -> str:
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    prompt_text = _prepare_prompt(tokenizer, prompt, no_chat_template)
    sampler = None
    if (
        (temperature and temperature > 0)
        or (top_p and top_p < 1.0)
        or (top_k and top_k > 0)
    ):
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    gen_kwargs: dict[str, Any] = {"max_tokens": max_new_tokens}
    if sampler is not None:
        gen_kwargs["sampler"] = sampler
    return generate(model, tokenizer, prompt_text, **gen_kwargs)


def _stream_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    no_chat_template: bool,
) -> str:
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_text = _prepare_prompt(tokenizer, prompt, no_chat_template)
    sampler = None
    if (
        (temperature and temperature > 0)
        or (top_p and top_p < 1.0)
        or (top_k and top_k > 0)
    ):
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    gen_kwargs: dict[str, Any] = {"max_tokens": max_new_tokens}
    if sampler is not None:
        gen_kwargs["sampler"] = sampler

    chunks = []
    for response in stream_generate(model, tokenizer, prompt_text, **gen_kwargs):
        if response.text:
            print(response.text, end="", flush=True)
            chunks.append(response.text)
    if chunks:
        print()
    return "".join(chunks)


def _send_request(
    socket_path: Path, payload: dict[str, Any], timeout: float | None = None
) -> dict[str, Any] | None:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if timeout is not None:
            sock.settimeout(timeout)
        sock.connect(str(socket_path))
    except OSError:
        return None

    with sock:
        try:
            file = sock.makefile("rwb")
            data = json.dumps(payload).encode("utf-8") + b"\n"
            file.write(data)
            file.flush()
            line = file.readline()
        except (OSError, TimeoutError):
            return None
        if not line:
            return None
        return json.loads(line.decode("utf-8"))


def _send_request_with_connect_timeout(
    socket_path: Path, payload: dict[str, Any], *, connect_timeout: float
) -> dict[str, Any] | None:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(connect_timeout)
        sock.connect(str(socket_path))
    except OSError:
        return None

    with sock:
        try:
            sock.settimeout(None)
            file = sock.makefile("rwb")
            data = json.dumps(payload).encode("utf-8") + b"\n"
            file.write(data)
            file.flush()
            line = file.readline()
        except (OSError, TimeoutError):
            return None
        if not line:
            return None
        return json.loads(line.decode("utf-8"))


def _read_state(state_path: Path) -> dict[str, Any] | None:
    try:
        data = state_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _write_state(state_path: Path, state: dict[str, Any]) -> None:
    try:
        state_path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        return


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cleanup_stale_resources(socket_path: Path, state_path: Path) -> None:
    state = _read_state(state_path)
    pid = None
    if isinstance(state, dict):
        pid = state.get("pid")
    if isinstance(pid, int) and _is_pid_alive(pid):
        return
    if pid is None and socket_path.exists():
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.2)
            sock.connect(str(socket_path))
        except OSError:
            pass
        else:
            return
        finally:
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
    _remove_path(socket_path)
    _remove_path(state_path)


def _get_server_status(
    socket_path: Path,
    *,
    state_path: Path | None = None,
    timeout: float = STATUS_TIMEOUT,
) -> dict[str, Any] | None:
    if state_path is not None:
        state = _read_state(state_path)
        if isinstance(state, dict):
            pid = state.get("pid")
            if isinstance(pid, int) and _is_pid_alive(pid):
                return {
                    "ok": True,
                    "pid": pid,
                    "model": state.get("model"),
                    "defaults": state.get("defaults"),
                }
            _cleanup_stale_resources(socket_path, state_path)
    response = _send_request(socket_path, {"type": "status"}, timeout=timeout)
    if not response or not response.get("ok"):
        return None
    return response


def _remove_stale_socket(socket_path: Path) -> None:
    try:
        socket_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _wait_for_server(
    socket_path: Path,
    *,
    state_path: Path,
    timeout: float = 60.0,
    status_timeout: float = 1.0,
) -> dict[str, Any] | None:
    start = time.time()
    while time.time() - start < timeout:
        status = _get_server_status(
            socket_path,
            state_path=state_path,
            timeout=status_timeout,
        )
        if status:
            return status
        time.sleep(0.2)
    return None


def _start_server(
    *,
    model: str,
    socket_path: Path,
    log_path: Path,
    state_path: Path,
    trust_remote_code: bool,
    verbose: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    no_chat_template: bool,
) -> dict[str, Any] | None:
    ensure_directory(socket_path.parent)
    ensure_directory(log_path.parent)
    ensure_directory(state_path.parent)
    if socket_path.exists() or state_path.exists():
        status = _get_server_status(socket_path, state_path=state_path)
        if status:
            return status
        _cleanup_stale_resources(socket_path, state_path)

    cmd = [
        sys.executable,
        "-m",
        "cat_translate.cli",
        "server",
        "run",
        "--model",
        model,
        "--socket",
        str(socket_path),
        "--log-file",
        str(log_path),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
    ]
    if no_chat_template:
        cmd.append("--no-chat-template")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if verbose:
        sys.stderr.write(f"[INFO] Starting server: {' '.join(cmd)}\n")

    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    return _wait_for_server(socket_path, state_path=state_path)


def run_mlx(prompt: str, args: argparse.Namespace) -> str:
    loaded = _load_model(args.model, args.trust_remote_code)
    model = loaded[0]
    tokenizer = loaded[1]
    gen_args = _resolve_generation_args(args)
    return _generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=gen_args["max_new_tokens"],
        temperature=gen_args["temperature"],
        top_p=gen_args["top_p"],
        top_k=gen_args["top_k"],
        no_chat_template=gen_args["no_chat_template"],
    )


def _handle_request(
    conn: socket.socket,
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    defaults: dict[str, Any],
) -> bool:
    file = conn.makefile("rwb")
    line = file.readline()
    if not line:
        return False
    try:
        request = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        response = {"ok": False, "error": "invalid_json"}
        _write_response(file, response)
        _log_server_event(
            "invalid_json",
            {"raw": line.decode("utf-8", errors="replace")[:200]},
        )
        return False

    req_type = request.get("type")
    if req_type == "status":
        _log_server_event("status", {})
        response = {
            "ok": True,
            "pid": os.getpid(),
            "model": model_name,
            "defaults": defaults,
        }
        _write_response(file, response)
        return False

    if req_type == "stop":
        _log_server_event("stop", {})
        response = {"ok": True}
        _write_response(file, response)
        return True

    if req_type == "translate":
        request_id = next(_REQUEST_COUNTER)
        prompt = request.get("prompt", "")
        max_new_tokens = request.get("max_new_tokens", defaults["max_new_tokens"])
        temperature = request.get("temperature", defaults["temperature"])
        top_p = request.get("top_p", defaults["top_p"])
        top_k = request.get("top_k", defaults["top_k"])
        no_chat_template = request.get(
            "no_chat_template", defaults["no_chat_template"]
        )
        _log_server_event(
            "translate_start",
            {
                "id": request_id,
                "model": model_name,
                "prompt_len": len(prompt),
                "prompt_head": _log_snippet(prompt),
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "no_chat_template": bool(no_chat_template),
            },
        )
        start_time = time.time()
        try:
            text = _generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                no_chat_template=bool(no_chat_template),
            )
        except Exception as exc:
            duration = time.time() - start_time
            _log_server_event(
                "translate_error",
                {
                    "id": request_id,
                    "duration_sec": round(duration, 3),
                    "error": str(exc),
                },
            )
            response = {"ok": False, "error": "translate_failed"}
            _write_response(file, response)
            return False

        response = {"ok": True, "text": text}
        duration = time.time() - start_time
        _log_server_event(
            "translate_done",
            {
                "id": request_id,
                "duration_sec": round(duration, 3),
                "output_len": len(text),
                "output_head": _log_snippet(text),
            },
        )
        _write_response(file, response)
        return False

    response = {"ok": False, "error": "unknown_request"}
    _write_response(file, response)
    _log_server_event("unknown_request", {"type": req_type})
    return False


def _write_response(file: Any, response: dict[str, Any]) -> None:
    try:
        file.write(json.dumps(response).encode("utf-8") + b"\n")
        file.flush()
    except (BrokenPipeError, OSError):
        return


def _log_snippet(text: str) -> str:
    cleaned = text.replace("\n", " ").replace("\r", " ")
    return cleaned[:LOG_SNIPPET_LIMIT]


def _log_server_event(event: str, payload: dict[str, Any]) -> None:
    data = {"event": event, "ts": time.time(), **payload}
    try:
        sys.stderr.write("[SERVER] " + json.dumps(data) + "\n")
    except Exception:
        return


def _run_server(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    socket_path = resolve_socket_path(args.socket)
    log_path = resolve_log_path(args.log_file)
    state_path = resolve_state_path(None)
    ensure_directory(socket_path.parent)
    ensure_directory(log_path.parent)
    ensure_directory(state_path.parent)

    if socket_path.exists():
        status = _get_server_status(socket_path, state_path=state_path)
        if status:
            sys.stderr.write(
                f"[INFO] Server already running (model={status['model']}).\n"
            )
            return 0
        _cleanup_stale_resources(socket_path, state_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    sock.listen(SERVER_LISTEN_BACKLOG)

    model_name = args.model or DEFAULT_MLX_MODEL
    defaults = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "no_chat_template": args.no_chat_template,
    }
    loaded = _load_model(model_name, args.trust_remote_code)
    model = loaded[0]
    tokenizer = loaded[1]
    _write_state(
        state_path,
        {
            "pid": os.getpid(),
            "model": model_name,
            "defaults": defaults,
            "socket": str(socket_path),
            "log": str(log_path),
            "started_at": time.time(),
        },
    )
    atexit.register(_remove_path, state_path)
    sys.stderr.write(
        "[INFO] Server defaults: "
        + json.dumps({"model": model_name, "defaults": defaults})
        + "\n"
    )

    try:
        should_stop = False
        while not should_stop:
            conn, _ = sock.accept()
            with conn:
                should_stop = _handle_request(
                    conn,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    defaults=defaults,
                )
    finally:
        sock.close()
        _remove_path(state_path)
        _remove_stale_socket(socket_path)
    return 0


def _server_start(args: argparse.Namespace) -> int:
    socket_path = resolve_socket_path(args.socket)
    log_path = resolve_log_path(args.log_file)
    state_path = resolve_state_path(None)
    model = args.model or DEFAULT_MLX_MODEL
    defaults = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "no_chat_template": args.no_chat_template,
    }

    status = _get_server_status(socket_path, state_path=state_path)
    if status:
        if status.get("model") != model:
            msg = (
                f"Server already running with model {status['model']}. "
                "Stop it before starting a different model."
            )
            sys.stderr.write(msg + "\n")
            print(
                json.dumps(
                    {
                        "status": "error",
                        "message": msg,
                        "pid": status.get("pid"),
                        "model": status.get("model"),
                        "socket": str(socket_path),
                        "log": str(log_path),
                        "defaults": status.get("defaults"),
                    },
                    indent=2,
                )
            )
            return 1
        if args.verbose:
            sys.stderr.write(
                f"[INFO] Server already running (model={status['model']}).\n"
            )
        print(
            json.dumps(
                {
                    "status": "running",
                    "pid": status.get("pid"),
                    "model": status.get("model"),
                    "socket": str(socket_path),
                    "log": str(log_path),
                    "message": "already running",
                    "defaults": status.get("defaults"),
                },
                indent=2,
            )
        )
        return 0

    status = _start_server(
        model=model,
        socket_path=socket_path,
        log_path=log_path,
        state_path=state_path,
        trust_remote_code=args.trust_remote_code,
        verbose=args.verbose,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        no_chat_template=args.no_chat_template,
    )
    if not status:
        msg = "Failed to start server."
        sys.stderr.write(msg + "\n")
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": msg,
                    "socket": str(socket_path),
                    "log": str(log_path),
                },
                indent=2,
            )
        )
        return 1
    if args.verbose:
        sys.stderr.write(
            f"[INFO] Server started (model={status['model']}, socket={socket_path}).\n"
        )
    print(
        json.dumps(
            {
                "status": "started",
                "pid": status.get("pid"),
                "model": status.get("model"),
                "socket": str(socket_path),
                "log": str(log_path),
                "defaults": status.get("defaults") or defaults,
            },
            indent=2,
        )
    )
    return 0


def _server_stop(args: argparse.Namespace) -> int:
    socket_path = resolve_socket_path(args.socket)
    log_path = resolve_log_path(args.log_file)
    state_path = resolve_state_path(None)
    defaults = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "no_chat_template": args.no_chat_template,
    }
    status = _get_server_status(socket_path, state_path=state_path)
    if not status:
        if socket_path.exists():
            _cleanup_stale_resources(socket_path, state_path)
        if args.verbose:
            sys.stderr.write("[INFO] No running server found.\n")
        print(
            json.dumps(
                {
                    "status": "stopped",
                    "model": args.model or DEFAULT_MLX_MODEL,
                    "socket": str(socket_path),
                    "log": str(log_path),
                    "message": "not running",
                    "defaults": defaults,
                },
                indent=2,
            )
        )
        return 0

    response = _send_request(socket_path, {"type": "stop"}, timeout=2.0)
    if not response or not response.get("ok"):
        msg = "Failed to stop server."
        sys.stderr.write(msg + "\n")
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": msg,
                    "pid": status.get("pid"),
                    "model": status.get("model"),
                    "socket": str(socket_path),
                    "log": str(log_path),
                    "defaults": status.get("defaults"),
                },
                indent=2,
            )
        )
        return 1
    _wait_for_server(
        socket_path,
        state_path=state_path,
        timeout=2.0,
        status_timeout=0.5,
    )
    if args.verbose:
        sys.stderr.write("[INFO] Server stopped.\n")
    print(
        json.dumps(
            {
                "status": "stopped",
                "pid": status.get("pid"),
                "model": status.get("model"),
                "socket": str(socket_path),
                "log": str(log_path),
                "defaults": status.get("defaults"),
            },
            indent=2,
        )
    )
    return 0


def _server_status(args: argparse.Namespace) -> int:
    socket_path = resolve_socket_path(args.socket)
    log_path = resolve_log_path(args.log_file)
    state_path = resolve_state_path(None)
    status = _get_server_status(socket_path, state_path=state_path)
    if not status:
        print(
            json.dumps(
                {
                    "status": "stopped",
                    "socket": str(socket_path),
                    "log": str(log_path),
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "status": "running",
                "pid": status.get("pid"),
                "model": status.get("model"),
                "socket": str(socket_path),
                "log": str(log_path),
                "defaults": status.get("defaults"),
            },
            indent=2,
        )
    )
    return 0


def _translate_via_server(args: argparse.Namespace, prompt: str) -> str | None:
    socket_path = resolve_socket_path(args.socket)
    overrides = _build_request_overrides(args)
    payload = {
        "type": "translate",
        "prompt": prompt,
        **overrides,
    }
    deadline = time.time() + TRANSLATE_CONNECT_DEADLINE
    while True:
        response = _send_request_with_connect_timeout(
            socket_path,
            payload,
            connect_timeout=TRANSLATE_CONNECT_TIMEOUT,
        )
        if response and response.get("ok"):
            return str(response.get("text", ""))
        if time.time() >= deadline:
            return None
        time.sleep(TRANSLATE_CONNECT_SLEEP)


def _translate_prompt(prompt: str, args: argparse.Namespace) -> int:
    server_mode = args.server
    socket_path = resolve_socket_path(args.socket)
    state_path = resolve_state_path(None)
    model = args.model or DEFAULT_MLX_MODEL

    if args.stream and server_mode != "never":
        if server_mode == "always":
            sys.stderr.write("Streaming is not supported via server.\n")
            return 1
        if args.verbose:
            sys.stderr.write("[INFO] Streaming forces direct execution.\n")
        server_mode = "never"

    if server_mode != "never":
        status = _get_server_status(socket_path, state_path=state_path)
        if status:
            if args.model is not None and status.get("model") != args.model:
                sys.stderr.write(
                    f"Server already running with model {status['model']}. "
                    "Stop it before using a different model.\n"
                )
                return 1
            if args.verbose:
                sys.stderr.write(f"[INFO] Using server at {socket_path}\n")
            translation = _translate_via_server(args, prompt)
            if translation is None:
                sys.stderr.write("Server failed to translate.\n")
                return 1
            print(translation)
            return 0

        if socket_path.exists() or state_path.exists():
            _cleanup_stale_resources(socket_path, state_path)

        gen_args = _resolve_generation_args(args)
        status = _start_server(
            model=model,
            socket_path=socket_path,
            log_path=resolve_log_path(args.log_file),
            state_path=state_path,
            trust_remote_code=args.trust_remote_code,
            verbose=args.verbose,
            max_new_tokens=gen_args["max_new_tokens"],
            temperature=gen_args["temperature"],
            top_p=gen_args["top_p"],
            top_k=gen_args["top_k"],
            no_chat_template=gen_args["no_chat_template"],
        )
        if status:
            if args.verbose:
                sys.stderr.write(f"[INFO] Started server at {socket_path}\n")
            translation = _translate_via_server(args, prompt)
            if translation is None:
                sys.stderr.write("Server failed to translate.\n")
                return 1
            print(translation)
            return 0
        if server_mode == "always":
            sys.stderr.write("Failed to start server.\n")
            return 1

    args.model = model
    gen_args = _resolve_generation_args(args)
    loaded = _load_model(args.model, args.trust_remote_code)
    model_obj = loaded[0]
    tokenizer = loaded[1]
    with silence_stderr(not args.verbose):
        if args.stream:
            _stream_text(
                model_obj,
                tokenizer,
                prompt,
                max_new_tokens=gen_args["max_new_tokens"],
                temperature=gen_args["temperature"],
                top_p=gen_args["top_p"],
                top_k=gen_args["top_k"],
                no_chat_template=gen_args["no_chat_template"],
            )
        else:
            translation = _generate_text(
                model_obj,
                tokenizer,
                prompt,
                max_new_tokens=gen_args["max_new_tokens"],
                temperature=gen_args["temperature"],
                top_p=gen_args["top_p"],
                top_k=gen_args["top_k"],
                no_chat_template=gen_args["no_chat_template"],
            )
            print(translation)
    return 0


def _handle_translate(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    text = read_text(args)
    input_lang, output_lang = resolve_languages(args, text)
    prompt = PROMPT_TEMPLATE.format(
        src_lang=LANG_NAME_MAP[input_lang],
        tgt_lang=LANG_NAME_MAP[output_lang],
        src_text=text,
    )
    return _translate_prompt(prompt, args)


def _run_interactive(args: argparse.Namespace) -> int:
    _setup_readline_history()
    args.stream = True
    if args.verbose:
        sys.stderr.write("[INFO] Interactive mode (type 'exit' to quit).\n")
    configure_logging(args.verbose)

    while True:
        try:
            line = input(">> ")
        except EOFError:
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit", "q"}:
            break
        input_lang, output_lang = resolve_languages(args, line)
        prompt = PROMPT_TEMPLATE.format(
            src_lang=LANG_NAME_MAP[input_lang],
            tgt_lang=LANG_NAME_MAP[output_lang],
            src_text=line,
        )
        _translate_prompt(prompt, args)
    return 0


def _setup_readline_history() -> None:
    try:
        import readline  # noqa: F401
    except Exception:
        return

    try:
        import readline as _readline  # noqa: F401
    except Exception:
        return


def main() -> int:
    mode, args = parse_args()
    if mode == "translate" and args.text is None and sys.stdin.isatty():
        return _run_interactive(args)
    if mode == "server":
        action = args.action or "start"
        if action == "run":
            return _run_server(args)
        if action == "start":
            return _server_start(args)
        if action == "stop":
            return _server_stop(args)
        if action == "status":
            return _server_status(args)
        raise SystemExit(f"Unknown server action: {action}")
    return _handle_translate(args)


if __name__ == "__main__":
    raise SystemExit(main())
