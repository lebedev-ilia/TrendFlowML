#!/usr/bin/env python3
"""
Планировщик плотных окон для action_recognition (Segmenter, R3).

Design: DataProcessor/docs/design/ACTION_RECOGNITION_V3.md (раздел B).

Проблема: union-sampling даёт разреженные кадры → SlowFast (clip_len=32) получает <32 кадров на
трек → 1 клип/трек. Решение: для action_recognition семплируем **непрерывные окна ≥clip_len
подряд идущих нативных кадров**, расставленные по таймлайну с шагом hop. Детекция+трекер затем
связывают person'ов сквозь плотные кадры окна.

Функции чистые (numpy-free) и юнит-тестируемые — без зависимостей от Segmenter-стейта.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

WINDOW_POLICY_VERSION = "ar_dense_windows_v1"


def plan_dense_windows(
    total_frames_source: int,
    source_fps: float,
    *,
    clip_len: int = 32,
    hop_s: float = 2.0,
    max_windows: int = 48,
    windows_per_min: float = 0.0,
    max_windows_hard: int = 256,
    window_frames: Optional[int] = None,
) -> List[List[int]]:
    """
    Возвращает список окон; каждое окно — список из `window_frames` подряд идущих source-индексов
    (по умолчанию `window_frames = clip_len`). Окна равномерно расставлены по таймлайну с шагом
    ~hop_s секунд, число ограничено эффективным капом. Индексы клампятся в [0, total-1].

    ВАЖНО (ASSESSMENT §1.2, фикс mean_clips_per_track): если окно = clip_len, то компонент делает
    ровно 1 клип на окно → `mean_clips_per_track=1.0`. Делая окно ДЛИННЕЕ clip_len (`window_frames >
    clip_len`), компонент скольжением получает НЕСКОЛЬКО клипов на трек-присутствие в окне. Чтобы
    сохранить бюджет кадров, оркестратор уменьшает число окон пропорционально длине окна.

    Адаптивность: если `windows_per_min > 0`, эффективный кап =
    clamp(round(duration_min * windows_per_min), max_windows, max_windows_hard).

    Крайние случаи:
    - total <= 0 → [].
    - 0 < total < window_frames → одно окно = все доступные кадры.
    """
    total = int(total_frames_source)
    if total <= 0:
        return []
    clip_len = max(1, int(clip_len))
    win = max(clip_len, int(window_frames) if window_frames else clip_len)
    if total < win:
        return [list(range(total))]

    fps = float(source_fps) if source_fps and source_fps > 0 else 25.0
    # адаптивный кап по длительности
    if windows_per_min and windows_per_min > 0:
        duration_min = (total / fps) / 60.0
        adaptive = int(round(duration_min * float(windows_per_min)))
        max_windows = max(int(max_windows), adaptive)
    max_windows = min(int(max_windows), int(max_windows_hard))
    hop_frames = max(1, int(round(float(hop_s) * fps)))
    last_start = total - win
    starts = list(range(0, last_start + 1, hop_frames))
    if starts[-1] != last_start:
        starts.append(last_start)

    # прорежаем до max_windows равномерно (сохраняя первый и последний)
    if max_windows and len(starts) > int(max_windows):
        m = int(max_windows)
        idx = [round(i * (len(starts) - 1) / (m - 1)) for i in range(m)] if m > 1 else [0]
        seen = set()
        picked = []
        for i in idx:
            if i not in seen:
                seen.add(i)
                picked.append(starts[i])
        starts = picked

    return [list(range(s, s + win)) for s in starts]


def windows_to_source_indices(windows: List[List[int]]) -> List[int]:
    """Плоский отсортированный уникальный список source-индексов из окон (для union)."""
    s = set()
    for w in windows:
        s.update(int(x) for x in w)
    return sorted(s)


def map_windows_to_union(
    windows: List[List[int]],
    source_to_union: dict,
) -> List[dict]:
    """
    Переводит окна из source-домена в union-домен, отдавая для каждого окна непрерывный (в union)
    диапазон. Окна, чьи кадры частично не попали в union, усекаются; пустые — отбрасываются.
    Возвращает список {start, end, frame_indices, center} в union-домене.
    """
    out: List[dict] = []
    for w in windows:
        uni = [int(source_to_union[i]) for i in w if i in source_to_union]
        if not uni:
            continue
        uni_sorted = sorted(set(uni))
        center = uni_sorted[len(uni_sorted) // 2]
        out.append({
            "start": int(uni_sorted[0]),
            "end": int(uni_sorted[-1]),
            "frame_indices": uni_sorted,
            "center": int(center),
            "len": int(len(uni_sorted)),
        })
    # дедуп окон с одинаковым диапазоном (после усечения/прорежки могли совпасть)
    seen = set()
    dedup = []
    for wdw in out:
        key = (wdw["start"], wdw["end"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(wdw)
    return dedup
