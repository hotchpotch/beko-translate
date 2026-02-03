from __future__ import annotations

from .base import TranslationModel, lang_name

CAT_PROMPT_TEMPLATE = (
    "Translate the following {src_lang} text into {tgt_lang}.\n\n{src_text}"
)


class CATTranslateModel(TranslationModel):
    add_generation_prompt = True

    @classmethod
    def supports(cls, model_name: str) -> bool:
        return "cat-translate" in model_name.lower()

    def build_messages(self, src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
        prompt = CAT_PROMPT_TEMPLATE.format(
            src_lang=lang_name(src_lang),
            tgt_lang=lang_name(tgt_lang),
            src_text=text,
        )
        return [{"role": "user", "content": prompt}]

    def build_fallback_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:
        return CAT_PROMPT_TEMPLATE.format(
            src_lang=lang_name(src_lang),
            tgt_lang=lang_name(tgt_lang),
            src_text=text,
        )
