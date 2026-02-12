"""
Утилиты для CLI: время, хеши, атомарные операции.
"""
import os
import time
import uuid
import tempfile
import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def utc_iso_now() -> str:
    """Возвращает текущее UTC время в ISO формате."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(s: str) -> str:
    """Вычисляет SHA256 хеш строки."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def timestamp_now() -> str:
    """Возвращает текущий timestamp с микросекундами."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")


def short_uuid() -> str:
    """Генерирует короткий UUID (первые 8 символов)."""
    return uuid.uuid4().hex[:8]


def atomic_save_npz(path: str, **arrays: Any) -> None:
    """
    Атомарно сохраняет NPZ файл (через временный файл).
    
    Args:
        path: Путь к файлу для сохранения
        **arrays: Массивы для сохранения
    """
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".npz", dir=target_dir)
    os.close(tmp_fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def atomic_write_json(path: str, payload: dict) -> None:
    """
    Атомарно записывает JSON файл (через временный файл).
    
    Args:
        path: Путь к файлу
        payload: Данные для записи
    """
    import json
    
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def as_float(v: Any) -> float:
    """Безопасное преобразование в float (NaN для None/ошибок)."""
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def as_int(v: Any) -> int:
    """Безопасное преобразование в int (-1 для None/ошибок)."""
    try:
        if v is None:
            return -1
        return int(v)
    except Exception:
        return -1

