from __future__ import annotations

"""Вспомогательные функции для вычисления checksum файлов."""

import hashlib
from pathlib import Path


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Посчитать SHA256 для файла и вернуть hex‑строку без префикса."""
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


__all__ = ["compute_sha256"]


