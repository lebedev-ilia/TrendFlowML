from __future__ import annotations

"""Простейший PII‑фильтр для комментариев.

Цель — заложить каркас PII‑фильтрации (email/phone/url) без тяжёлых зависимостей.
Функции могут быть расширены в следующих фазах (Phase 3).
"""

import re
from dataclasses import dataclass
from typing import Iterable, List


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(\+?\d[\d\s\-().]{7,}\d)",
    re.IGNORECASE,
)
URL_RE = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    re.IGNORECASE,
)


@dataclass
class PIIMatch:
    """Описание найденного PII‑фрагмента."""

    kind: str  # "email" | "phone" | "url"
    value: str


def detect_pii(text: str) -> List[PIIMatch]:
    """Найти базовые PII‑сущности в тексте комментария.

    На MVP‑этапе просто находит вхождения, но не изменяет текст.
    В дальнейшем сюда можно добавить маскирование / удаление.
    """
    matches: list[PIIMatch] = []
    for m in EMAIL_RE.findall(text or ""):
        matches.append(PIIMatch(kind="email", value=m))
    for m in PHONE_RE.findall(text or ""):
        matches.append(PIIMatch(kind="phone", value=m if isinstance(m, str) else m[0]))
    for m in URL_RE.findall(text or ""):
        matches.append(PIIMatch(kind="url", value=m))
    return matches


def has_pii(text: str) -> bool:
    """Проверить, содержит ли текст какую‑либо PII‑информацию."""
    return bool(detect_pii(text))


def mask_pii(text: str) -> str:
    """Простейшее маскирование PII в тексте.

    Для MVP мы заменяем email/phone/url на маркеры [EMAIL], [PHONE], [URL].
    Это поведение может быть изменено позже (например, удаление или частичное маскирование).
    """
    masked = EMAIL_RE.sub("[EMAIL]", text or "")
    masked = PHONE_RE.sub("[PHONE]", masked)
    masked = URL_RE.sub("[URL]", masked)
    return masked


__all__ = ["PIIMatch", "detect_pii", "has_pii", "mask_pii"]


