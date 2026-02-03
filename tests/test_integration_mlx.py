from __future__ import annotations

import os
import types

import pytest

from cat_translate import cli


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") != "1",
    reason="Set RUN_MLX_INTEGRATION=1 to run MLX integration tests.",
)
def test_translate_with_remote_mlx_model() -> None:
    args = types.SimpleNamespace(
        model="hotchpotch/CAT-Translate-0.8b-mlx-q4",
        trust_remote_code=False,
        no_chat_template=False,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    prompt = cli.PROMPT_TEMPLATE.format(
        src_lang="English",
        tgt_lang="Japanese",
        src_text="Hello, world.",
    )
    translation = cli.run_mlx(prompt, args)
    assert translation
    assert any(
        "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff"
        for ch in translation
    )
