"""
NPZ савер для tempo_extractor.
"""
import os
from typing import Any, Dict, Optional, Callable

import numpy as np

from ...utils.cli_utils import atomic_save_npz


def _arr_fallback(payload: Dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    """Fallback функция для _arr если она не передана."""
    v = payload.get(key)
    if v is None:
        v = []
    return np.asarray(v, dtype=dtype).reshape(-1)


def save_tempo_npz(
    *,
    out_path: str,
    payload: Dict[str, Any],
    status: str,
    error: Optional[str],
    empty_reason: Optional[str],
    producer_version: str,
    schema_version: str,
    extra_meta: Optional[Dict[str, Any]],
    run_rs_path: str,
    feature_names: list,
    feature_values: list,
    add: Callable[[str, Any], None],
    _arr: Callable[[str, Any], np.ndarray],
    build_meta: Callable[..., np.ndarray],
) -> str:
    """
    Сохраняет NPZ артефакт для tempo_extractor.
    
    Args:
        out_path: Путь для сохранения NPZ файла
        payload: Данные от extractor'а
        status: Статус обработки
        error: Сообщение об ошибке (если есть)
        empty_reason: Причина пустого результата (если есть)
        producer_version: Версия продюсера
        schema_version: Версия схемы
        extra_meta: Дополнительные метаданные
        run_rs_path: Путь к директории run
        feature_names: Список имен фич (будет дополнен)
        feature_values: Список значений фич (будет дополнен)
        add: Функция для добавления фичи
        _arr: Функция для безопасного преобразования в массив
        build_meta: Функция для построения метаданных
    
    Returns:
        Путь к сохраненному файлу
    """
    # Добавляем фичи
    add("tempo_bpm", payload.get("tempo_bpm"))
    add("tempo_bpm_mean", payload.get("tempo_bpm_mean"))
    add("tempo_bpm_median", payload.get("tempo_bpm_median"))
    add("tempo_bpm_std", payload.get("tempo_bpm_std"))
    add("tempo_confidence", payload.get("confidence"))
    add("duration_sec", payload.get("duration"))
    add("sample_rate", payload.get("sample_rate"))
    add("tempo_windowed_bpm_mean", (payload.get("windowed_bpm") or {}).get("bpm_mean"))
    add("tempo_windowed_bpm_median", (payload.get("windowed_bpm") or {}).get("bpm_median"))
    add("tempo_windowed_bpm_std", (payload.get("windowed_bpm") or {}).get("bpm_std"))
    add("segments_count", payload.get("segments_count"))

    tempo_estimates = payload.get("tempo_estimates")
    if tempo_estimates is None:
        tempo_estimates_arr = np.zeros((0,), dtype=np.float32)
    else:
        tempo_estimates_arr = np.asarray(tempo_estimates, dtype=np.float32).reshape(-1)

    windowed = payload.get("windowed_bpm") or {}
    if isinstance(windowed, dict) and windowed:
        w_times = np.asarray(windowed.get("times_sec") or [], dtype=np.float32).reshape(-1)
        w_bpm = np.asarray(windowed.get("bpm") or [], dtype=np.float32).reshape(-1)
    else:
        w_times = np.zeros((0,), dtype=np.float32)
        w_bpm = np.zeros((0,), dtype=np.float32)

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        tempo_estimates=tempo_estimates_arr,
        windowed_times_sec=w_times,
        windowed_bpm=w_bpm,
        warnings=_arr("warnings", dtype=object),
        meta=build_meta(
            producer="tempo_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
            },
        ),
    )
    return out_path

