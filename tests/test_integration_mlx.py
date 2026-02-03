from __future__ import annotations

import os
import argparse

import pytest

from neko_translate import cli


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") != "1",
    reason="Set RUN_MLX_INTEGRATION=1 to run MLX integration tests.",
)
def test_translate_with_remote_mlx_model() -> None:
    args = argparse.Namespace(
        model="hotchpotch/CAT-Translate-0.8b-mlx-q4",
        trust_remote_code=False,
        no_chat_template=False,
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    cli._load_model.cache_clear()
    info_before = cli._load_model.cache_info()
    translation = cli.run_mlx("Hello, world.", "en", "ja", args)
    info_after = cli._load_model.cache_info()
    translation_second = cli.run_mlx("Hello, world.", "en", "ja", args)
    info_final = cli._load_model.cache_info()
    assert info_after.misses == info_before.misses + 1
    assert info_final.hits >= info_after.hits + 1
    for output in (translation, translation_second):
        assert output
        assert any(
            "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff"
            for ch in output
        )
