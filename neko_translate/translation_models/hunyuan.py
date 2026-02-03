from __future__ import annotations

from .base import TranslationModel, normalize_lang_code

HUNYUAN_ZH_PROMPT_TEMPLATE = (
    "将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"
)

HUNYUAN_XX_PROMPT_TEMPLATE = (
    "Translate the following segment into {target_language}, without additional "
    "explanation.\n\n{source_text}"
)

HUNYUAN_LANG_NAMES_EN = {
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "tr": "Turkish",
    "ru": "Russian",
    "ar": "Arabic",
    "ko": "Korean",
    "th": "Thai",
    "it": "Italian",
    "de": "German",
    "vi": "Vietnamese",
    "ms": "Malay",
    "id": "Indonesian",
    "tl": "Filipino",
    "hi": "Hindi",
    "zh-hant": "Traditional Chinese",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
    "km": "Khmer",
    "my": "Burmese",
    "fa": "Persian",
    "gu": "Gujarati",
    "ur": "Urdu",
    "te": "Telugu",
    "mr": "Marathi",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "uk": "Ukrainian",
    "bo": "Tibetan",
    "kk": "Kazakh",
    "mn": "Mongolian",
    "ug": "Uyghur",
    "yue": "Cantonese",
}

HUNYUAN_LANG_NAMES_ZH = {
    "zh": "中文",
    "en": "英语",
    "fr": "法语",
    "pt": "葡萄牙语",
    "es": "西班牙语",
    "ja": "日语",
    "tr": "土耳其语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "ko": "韩语",
    "th": "泰语",
    "it": "意大利语",
    "de": "德语",
    "vi": "越南语",
    "ms": "马来语",
    "id": "印尼语",
    "tl": "菲律宾语",
    "hi": "印地语",
    "zh-hant": "繁体中文",
    "pl": "波兰语",
    "cs": "捷克语",
    "nl": "荷兰语",
    "km": "高棉语",
    "my": "缅甸语",
    "fa": "波斯语",
    "gu": "古吉拉特语",
    "ur": "乌尔都语",
    "te": "泰卢固语",
    "mr": "马拉地语",
    "he": "希伯来语",
    "bn": "孟加拉语",
    "ta": "泰米尔语",
    "uk": "乌克兰语",
    "bo": "藏语",
    "kk": "哈萨克语",
    "mn": "蒙古语",
    "ug": "维吾尔语",
    "yue": "粤语",
}


class HunyuanTranslationModel(TranslationModel):
    add_generation_prompt = True

    @classmethod
    def supports(cls, model_name: str) -> bool:
        lowered = model_name.lower()
        return any(token in lowered for token in ("hy-mt", "hy-my", "hunyuan"))

    def _use_zh_prompt(self, src_lang: str, tgt_lang: str) -> bool:
        src_norm = normalize_lang_code(src_lang)
        tgt_norm = normalize_lang_code(tgt_lang)
        return src_norm.startswith("zh") or tgt_norm.startswith("zh")

    def _language_name(self, code: str, *, use_zh_prompt: bool) -> str:
        normalized = normalize_lang_code(code)
        if use_zh_prompt:
            return HUNYUAN_LANG_NAMES_ZH.get(normalized, code)
        return HUNYUAN_LANG_NAMES_EN.get(normalized, code)

    def _build_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:
        use_zh_prompt = self._use_zh_prompt(src_lang, tgt_lang)
        target_name = self._language_name(tgt_lang, use_zh_prompt=use_zh_prompt)
        template = (
            HUNYUAN_ZH_PROMPT_TEMPLATE if use_zh_prompt else HUNYUAN_XX_PROMPT_TEMPLATE
        )
        return template.format(target_language=target_name, source_text=text)

    def build_messages(self, src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
        prompt = self._build_prompt(src_lang, tgt_lang, text)
        return [{"role": "user", "content": prompt}]

    def build_fallback_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:
        return self._build_prompt(src_lang, tgt_lang, text)

    def default_generation(self) -> dict[str, object]:
        return {
            "temperature": 0.7,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
