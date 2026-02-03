from __future__ import annotations

from .base import TranslationModel
from .cat import CATTranslateModel
from .hunyuan import HunyuanTranslationModel
from .plamo import PlamoTranslateModel

MODEL_ALIAS_MAP = {
    "plamo": "mlx-community/plamo-2-translate",
    "cat": "hotchpotch/CAT-Translate-1.8b-mlx-q8",
}


def resolve_translation_model(model_name: str) -> TranslationModel:
    for model_cls in (PlamoTranslateModel, HunyuanTranslationModel, CATTranslateModel):
        if model_cls.supports(model_name):
            return model_cls()
    return CATTranslateModel()


def resolve_model_alias(model_name: str | None, default: str) -> str:
    if model_name is None:
        return default
    candidate = model_name.strip()
    if not candidate:
        return default
    alias = MODEL_ALIAS_MAP.get(candidate.lower())
    if alias:
        return alias
    return model_name


__all__ = [
    "CATTranslateModel",
    "HunyuanTranslationModel",
    "MODEL_ALIAS_MAP",
    "PlamoTranslateModel",
    "TranslationModel",
    "resolve_model_alias",
    "resolve_translation_model",
]
