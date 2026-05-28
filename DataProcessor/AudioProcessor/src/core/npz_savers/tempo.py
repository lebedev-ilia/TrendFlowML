"""
NPZ савер для tempo_extractor (Audit v3).
Canonical axis: segment_start_sec, segment_end_sec, segment_center_sec, segment_mask, bpm_by_segment.
"""
import os
from typing import Any, Dict, Optional, Callable

import numpy as np

from ...utils.cli_utils import atomic_save_npz

TEMPO_CONTRACT_VERSION = "tempo_contract_v1"


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
    # Audit v3: bpm_by_segment stats (was tempo_windowed_bpm_*)
    bpm_by_seg = payload.get("bpm_by_segment")
    windowed = payload.get("windowed_bpm") or {}
    if isinstance(bpm_by_seg, (list, np.ndarray)) and len(bpm_by_seg) > 0:
        bpm_arr = np.asarray(bpm_by_seg, dtype=np.float32).reshape(-1)
        valid = np.isfinite(bpm_arr)
        add("tempo_bpm_by_segment_mean", float(np.mean(bpm_arr[valid])) if np.any(valid) else None)
        add("tempo_bpm_by_segment_median", float(np.median(bpm_arr[valid])) if np.any(valid) else None)
        add("tempo_bpm_by_segment_std", float(np.std(bpm_arr[valid])) if np.sum(valid) > 1 else None)
    else:
        add("tempo_bpm_by_segment_mean", (windowed or {}).get("bpm_mean") if isinstance(windowed, dict) else None)
        add("tempo_bpm_by_segment_median", (windowed or {}).get("bpm_median") if isinstance(windowed, dict) else None)
        add("tempo_bpm_by_segment_std", (windowed or {}).get("bpm_std") if isinstance(windowed, dict) else None)
    add("segments_count", payload.get("segments_count"))

    tempo_estimates = payload.get("tempo_estimates")
    if tempo_estimates is None:
        tempo_estimates_arr = np.zeros((0,), dtype=np.float32)
    else:
        tempo_estimates_arr = np.asarray(tempo_estimates, dtype=np.float32).reshape(-1)

    # Audit v3: canonical axis (segment_* , bpm_by_segment)
    seg_start = payload.get("segment_start_sec")
    seg_end = payload.get("segment_end_sec")
    seg_center = payload.get("segment_center_sec")
    seg_mask = payload.get("segment_mask")
    bpm_by_segment = payload.get("bpm_by_segment")
    if seg_center is not None and bpm_by_segment is not None:
        segment_start_sec = np.asarray(seg_start or [], dtype=np.float32).reshape(-1)
        segment_end_sec = np.asarray(seg_end or [], dtype=np.float32).reshape(-1)
        segment_center_sec = np.asarray(seg_center, dtype=np.float32).reshape(-1)
        segment_mask = np.asarray(seg_mask if seg_mask is not None else [True] * len(segment_center_sec), dtype=bool).reshape(-1)
        bpm_by_segment_arr = np.asarray(bpm_by_segment, dtype=np.float32).reshape(-1)
    else:
        # Legacy run() path: windowed_bpm
        windowed = payload.get("windowed_bpm") or {}
        if isinstance(windowed, dict) and windowed:
            w_times = np.asarray(windowed.get("times_sec") or [], dtype=np.float32).reshape(-1)
            w_bpm = np.asarray(windowed.get("bpm") or [], dtype=np.float32).reshape(-1)
            segment_center_sec = w_times
            bpm_by_segment_arr = w_bpm
            segment_start_sec = np.zeros_like(w_times, dtype=np.float32)
            segment_end_sec = np.zeros_like(w_times, dtype=np.float32)
            segment_mask = np.ones(len(w_times), dtype=bool)
        else:
            segment_start_sec = np.zeros((0,), dtype=np.float32)
            segment_end_sec = np.zeros((0,), dtype=np.float32)
            segment_center_sec = np.zeros((0,), dtype=np.float32)
            segment_mask = np.zeros((0,), dtype=bool)
            bpm_by_segment_arr = np.zeros((0,), dtype=np.float32)

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        tempo_estimates=tempo_estimates_arr,
        segment_start_sec=segment_start_sec,
        segment_end_sec=segment_end_sec,
        segment_center_sec=segment_center_sec,
        segment_mask=segment_mask,
        bpm_by_segment=bpm_by_segment_arr,
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
                "tempo_contract_version": payload.get("tempo_contract_version", TEMPO_CONTRACT_VERSION),
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "tempo_resource_profile": payload.get("tempo_resource_profile"),
            },
        ),
    )
    return out_path

