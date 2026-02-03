from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

LANG_NAME_MAP = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-hant": "Traditional Chinese",
}


def normalize_lang_code(code: str) -> str:
    return code.strip().lower().replace("_", "-")


def lang_name(code: str) -> str:
    normalized = normalize_lang_code(code)
    if normalized in LANG_NAME_MAP:
        return LANG_NAME_MAP[normalized]
    return code


def trust_remote_code_config(trust_remote_code: bool) -> dict[str, Any]:
    if trust_remote_code:
        return {"trust_remote_code": True}
    return {}


class TranslationModel(ABC):
    add_generation_prompt: bool = True

    @classmethod
    @abstractmethod
    def supports(cls, model_name: str) -> bool:
        raise NotImplementedError

    def tokenizer_config(self, trust_remote_code: bool) -> dict[str, Any]:
        return trust_remote_code_config(trust_remote_code)

    def model_config(self, trust_remote_code: bool) -> dict[str, Any]:
        return trust_remote_code_config(trust_remote_code)

    def configure_tokenizer(self, tokenizer: Any) -> None:
        return

    @abstractmethod
    def build_messages(self, src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def build_fallback_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:
        raise NotImplementedError

    def render_prompt(
        self,
        tokenizer: Any,
        src_lang: str,
        tgt_lang: str,
        text: str,
        no_chat_template: bool,
    ) -> str:
        messages = self.build_messages(src_lang, tgt_lang, text)
        fallback = self.build_fallback_prompt(src_lang, tgt_lang, text)
        if no_chat_template or not hasattr(tokenizer, "apply_chat_template"):
            return fallback
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=self.add_generation_prompt,
            tokenize=False,
        )

    def default_generation(self) -> dict[str, Any]:
        return {}
