from __future__ import annotations

from neko_translate.translation_models import (
    CATTranslateModel,
    HunyuanTranslationModel,
    PlamoTranslateModel,
    resolve_model_alias,
    resolve_translation_model,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        self.messages = messages
        return "APPLIED"


def test_cat_prompt_uses_language_names() -> None:
    model = CATTranslateModel()
    prompt = model.build_fallback_prompt("ja", "en", "猫です")
    assert "Japanese" in prompt
    assert "English" in prompt


def test_cat_render_prompt_uses_chat_template() -> None:
    model = CATTranslateModel()
    tokenizer = DummyTokenizer()
    rendered = model.render_prompt(
        tokenizer,
        src_lang="ja",
        tgt_lang="en",
        text="猫です",
        no_chat_template=False,
    )
    assert rendered == "APPLIED"
    assert tokenizer.messages is not None


def test_plamo_language_mapping_uses_taiwanese() -> None:
    model = PlamoTranslateModel()
    messages = model.build_messages("zh-hant", "en", "測試")
    assert "input lang=Taiwanese" in messages[0]["content"]


def test_plamo_render_prompt_ignores_chat_template() -> None:
    model = PlamoTranslateModel()
    tokenizer = DummyTokenizer()
    rendered = model.render_prompt(
        tokenizer,
        src_lang="en",
        tgt_lang="ja",
        text="hello",
        no_chat_template=False,
    )
    assert rendered.startswith("<|plamo:op|>dataset")


def test_hunyuan_uses_zh_template_for_zh_targets() -> None:
    model = HunyuanTranslationModel()
    prompt = model.build_fallback_prompt("en", "zh", "hello")
    assert prompt.startswith("将以下文本翻译为中文")


def test_hunyuan_uses_en_template_for_non_zh() -> None:
    model = HunyuanTranslationModel()
    prompt = model.build_fallback_prompt("en", "ja", "hello")
    assert prompt.startswith("Translate the following segment into Japanese")


def test_hunyuan_default_generation_includes_repetition_penalty() -> None:
    model = HunyuanTranslationModel()
    defaults = model.default_generation()
    assert defaults["repetition_penalty"] == 1.05


def test_resolve_translation_model() -> None:
    assert isinstance(
        resolve_translation_model("mlx-community/HY-MT1.5-1.8B-8bit"),
        HunyuanTranslationModel,
    )
    assert isinstance(
        resolve_translation_model("mlx-community/plamo-2-translate"),
        PlamoTranslateModel,
    )
    assert isinstance(
        resolve_translation_model("hotchpotch/CAT-Translate-0.8b-mlx-q4"),
        CATTranslateModel,
    )


def test_resolve_model_alias_preserves_input() -> None:
    assert resolve_model_alias("mlx-community/HY-MT1.5-1.8B-8bit", "default") == (
        "mlx-community/HY-MT1.5-1.8B-8bit"
    )
