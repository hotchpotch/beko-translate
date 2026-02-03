from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

LANG_NAME_MAP = {
    "en": "English",
    "ja": "Japanese",
}

CAT_PROMPT_TEMPLATE = (
    "Translate the following {src_lang} text into {tgt_lang}.\n\n{src_text}"
)

PLAMO_CHAT_TEMPLATE = (
    "{{- \"<|plamo:op|>dataset\\ntranslation\\n\" -}}\n"
    "{% for message in messages %}\n"
    "    {{- '<|plamo:op|>' + message['content']}}\n"
    "    {%- if not loop.last %}\n"
    "        {{- '\\n'}}\n"
    "    {%- endif %}\n"
    "{% endfor %}\n"
)


def _lang_name(code: str) -> str:
    if code in LANG_NAME_MAP:
        return LANG_NAME_MAP[code]
    return code


def _trust_remote_code_config(trust_remote_code: bool) -> dict[str, Any]:
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
        return _trust_remote_code_config(trust_remote_code)

    def model_config(self, trust_remote_code: bool) -> dict[str, Any]:
        return _trust_remote_code_config(trust_remote_code)

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


class CATTranslateModel(TranslationModel):
    add_generation_prompt = True

    @classmethod
    def supports(cls, model_name: str) -> bool:
        return "cat-translate" in model_name.lower()

    def build_messages(self, src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
        prompt = CAT_PROMPT_TEMPLATE.format(
            src_lang=_lang_name(src_lang),
            tgt_lang=_lang_name(tgt_lang),
            src_text=text,
        )
        return [{"role": "user", "content": prompt}]

    def build_fallback_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:
        return CAT_PROMPT_TEMPLATE.format(
            src_lang=_lang_name(src_lang),
            tgt_lang=_lang_name(tgt_lang),
            src_text=text,
        )


class PlamoTranslateModel(TranslationModel):
    add_generation_prompt = False

    @classmethod
    def supports(cls, model_name: str) -> bool:
        lowered = model_name.lower()
        return "plamo-2-translate" in lowered or "plamo_translate" in lowered

    def tokenizer_config(self, trust_remote_code: bool) -> dict[str, Any]:
        config = _trust_remote_code_config(trust_remote_code)
        config["chat_template"] = PLAMO_CHAT_TEMPLATE
        return config

    def model_config(self, trust_remote_code: bool) -> dict[str, Any]:
        return _trust_remote_code_config(trust_remote_code)

    def configure_tokenizer(self, tokenizer: Any) -> None:
        add_eos = getattr(tokenizer, "add_eos_token", None)
        if callable(add_eos):
            add_eos("<|plamo:op|>")

    def build_messages(self, src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
        src = _lang_name(src_lang)
        tgt = _lang_name(tgt_lang)
        text = text.strip()
        messages = [
            {"role": "user", "content": f"input lang={src}\n{text}"},
            {"role": "user", "content": f"output lang={tgt}\n"},
        ]
        return messages

    def build_fallback_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:
        messages = self.build_messages(src_lang, tgt_lang, text)
        parts = ["<|plamo:op|>dataset\ntranslation\n"]
        for idx, message in enumerate(messages):
            parts.append("<|plamo:op|>" + message["content"])
            if idx != len(messages) - 1:
                parts.append("\n")
        return "".join(parts)


def resolve_translation_model(model_name: str) -> TranslationModel:
    for model_cls in (PlamoTranslateModel, CATTranslateModel):
        if model_cls.supports(model_name):
            return model_cls()
    return CATTranslateModel()
