from __future__ import annotations

import argparse

import pytest

import neko_translate.cli as cli

def make_args(**kwargs):
    return argparse.Namespace(**kwargs)


def test_read_text_from_arg() -> None:
    args = make_args(text="hello")
    assert cli.read_text(args) == "hello"


def test_resolve_languages_detects_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_detect(_: str) -> str:
        return "en"

    monkeypatch.setattr(cli, "detect_lang", fake_detect)
    args = make_args(input_lang=None, output_lang=None, verbose=False)
    src, tgt = cli.resolve_languages(args, "hello")
    assert (src, tgt) == ("en", "ja")


def test_resolve_languages_infers_from_input() -> None:
    args = make_args(input_lang="en", output_lang=None, verbose=False)
    src, tgt = cli.resolve_languages(args, "hello")
    assert (src, tgt) == ("en", "ja")


def test_resolve_languages_infers_from_output() -> None:
    args = make_args(input_lang=None, output_lang="ja", verbose=False)
    src, tgt = cli.resolve_languages(args, "hello")
    assert (src, tgt) == ("en", "ja")


def test_resolve_languages_rejects_same_language() -> None:
    args = make_args(input_lang="en", output_lang="en", verbose=False)
    with pytest.raises(SystemExit):
        cli.resolve_languages(args, "hello")


def test_resolve_languages_requires_output_for_non_default_input() -> None:
    args = make_args(input_lang="fr", output_lang=None, verbose=False)
    with pytest.raises(SystemExit):
        cli.resolve_languages(args, "bonjour")


def test_resolve_languages_requires_input_for_non_default_output() -> None:
    args = make_args(input_lang=None, output_lang="fr", verbose=False)
    with pytest.raises(SystemExit):
        cli.resolve_languages(args, "hello")


def test_detect_lang_prefers_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_detect(_: str, k: int = 3, model: str | None = None):
        return [
            {"lang": "fr", "score": 0.9},
            {"lang": "ja", "score": 0.8},
        ]

    monkeypatch.setattr("fast_langdetect.detect", fake_detect, raising=False)
    assert cli.detect_lang("text") == "fr"


def test_detect_lang_handles_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_detect(_: str, k: int = 3, model: str | None = None):
        return {"lang": "en", "score": 0.7}

    monkeypatch.setattr("fast_langdetect.detect", fake_detect, raising=False)
    assert cli.detect_lang("text") == "en"
