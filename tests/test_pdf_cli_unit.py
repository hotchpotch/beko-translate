from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import beko_translate.pdf_cli as pdf_cli


def _write_pdf(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_translate_pdf_picks_mono_output_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "paper.pdf"
    output_dir = tmp_path / "out"
    target_pdf = tmp_path / "paper.ja.pdf"
    output_dir.mkdir()
    _write_pdf(input_pdf, "input")
    _write_pdf(output_dir / "paper.no_watermark.dual.pdf", "dual")
    _write_pdf(output_dir / "paper.no_watermark.mono.pdf", "mono")

    monkeypatch.setattr(
        pdf_cli.subprocess,
        "run",
        lambda cmd: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(pdf_cli, "cleanup_extras", lambda *args, **kwargs: None)

    ok = pdf_cli.translate_pdf(
        input_pdf,
        output_dir,
        target_pdf,
        "paper",
        "paper",
        [],
        "mono",
    )

    assert ok is True
    assert target_pdf.read_text(encoding="utf-8") == "mono"


def test_translate_pdf_picks_dual_output_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "paper.pdf"
    output_dir = tmp_path / "out"
    target_pdf = tmp_path / "paper.ja.pdf"
    output_dir.mkdir()
    _write_pdf(input_pdf, "input")
    _write_pdf(output_dir / "paper.no_watermark.dual.pdf", "dual")
    _write_pdf(output_dir / "paper.no_watermark.mono.pdf", "mono")

    monkeypatch.setattr(
        pdf_cli.subprocess,
        "run",
        lambda cmd: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(pdf_cli, "cleanup_extras", lambda *args, **kwargs: None)

    ok = pdf_cli.translate_pdf(
        input_pdf,
        output_dir,
        target_pdf,
        "paper",
        "paper",
        [],
        "dual",
    )

    assert ok is True
    assert target_pdf.read_text(encoding="utf-8") == "dual"
