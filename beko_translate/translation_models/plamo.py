from __future__ import annotations

from .base import TranslationModel, normalize_lang_code, trust_remote_code_config

PLAMO_CHAT_TEMPLATE = (
    "{{- \"<|plamo:op|>dataset\\ntranslation\\n\" -}}\n"
    "{% for message in messages %}\n"
    "    {{- '<|plamo:op|>' + message['content']}}\n"
    "    {%- if not loop.last %}\n"
    "        {{- '\\n'}}\n"
    "    {%- endif %}\n"
    "{% endfor %}\n"
)

PLAMO_LANG_NAME_MAP = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-tw": "Taiwanese",
    "zh-hant": "Taiwanese",
    "taiwanese": "Taiwanese",
    "ko": "Korean",
    "ar": "Arabic",
    "it": "Italian",
    "id": "Indonesian",
    "nl": "Dutch",
    "es": "Spanish",
    "th": "Thai",
    "de": "German",
    "fr": "French",
    "vi": "Vietnamese",
    "ru": "Russian",
}


def _plamo_lang_name(code: str) -> str:
    normalized = normalize_lang_code(code)
    return PLAMO_LANG_NAME_MAP.get(normalized, code)


class PlamoTranslateModel(TranslationModel):
    add_generation_prompt = False

    @classmethod
    def supports(cls, model_name: str) -> bool:
        lowered = model_name.lower()
        return "plamo-2-translate" in lowered or "plamo_translate" in lowered

    def tokenizer_config(self, trust_remote_code: bool) -> dict[str, object]:
        config = trust_remote_code_config(trust_remote_code)
        config["chat_template"] = PLAMO_CHAT_TEMPLATE
        return config

    def configure_tokenizer(self, tokenizer: object) -> None:
        add_eos = getattr(tokenizer, "add_eos_token", None)
        if callable(add_eos):
            add_eos("<|plamo:op|>")

    def build_messages(self, src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
        src = _plamo_lang_name(src_lang)
        tgt = _plamo_lang_name(tgt_lang)
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

    def render_prompt(
        self,
        tokenizer: object,
        src_lang: str,
        tgt_lang: str,
        text: str,
        no_chat_template: bool,
    ) -> str:
        return self.build_fallback_prompt(src_lang, tgt_lang, text)

    def default_generation(self) -> dict[str, object]:
        return {
            "temperature": 0.0,
            "top_p": 0.98,
            "top_k": 0,
        }
