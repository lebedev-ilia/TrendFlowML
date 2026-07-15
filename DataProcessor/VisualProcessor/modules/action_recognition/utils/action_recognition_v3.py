#!/usr/bin/env python3
"""
Сборка плоского per-clip stream (schema action_recognition_npz_v3).

Design: DataProcessor/docs/design/ACTION_RECOGNITION_V3.md (раздел A).

Заменяет per-track object-dict (v2, `results_json`) единым time-ordered потоком клипов:
эмбеддинг (для Encoder) + классы Kinetics (для аналитиков) + привязка к треку + агрегаты.
Функция `build_v3_arrays` чистая (numpy) и юнит-тестируется без модели/GPU.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

SCHEMA_VERSION_V3 = "action_recognition_npz_v3"
NUM_ACTION_CLASSES = 400  # Kinetics-400


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / np.clip(np.sum(e, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def build_v3_arrays(
    *,
    clip_embeddings: np.ndarray,        # (C, 256) L2
    clip_track_ids: np.ndarray,         # (C,) int  (-1 если нет)
    clip_center_frame_idx: np.ndarray,  # (C,) int (union domain)
    clip_times_s: np.ndarray,           # (C,) float (⊆ union_timestamps_sec)
    clip_logits: Optional[np.ndarray] = None,   # (C, 400) Kinetics-логиты или None
    class_names: Optional[List[str]] = None,    # длина 400 (id->name) или None
    clip_segment_ids: Optional[np.ndarray] = None,  # (C,) id сегмента действия (temporal localization)
    topk: int = 5,
    top_video: int = 10,
) -> Dict[str, Any]:
    """
    Возвращает dict плоских v3-массивов (готов к np.savez_compressed). Клипы сортируются по времени.
    """
    C = int(clip_embeddings.shape[0]) if clip_embeddings.ndim == 2 else 0
    emb_dim = int(clip_embeddings.shape[1]) if clip_embeddings.ndim == 2 and C > 0 else 256

    if C == 0:
        return _empty_v3(emb_dim, topk, class_names)

    emb = np.asarray(clip_embeddings, dtype=np.float32)
    tid = np.asarray(clip_track_ids, dtype=np.int32).reshape(-1)
    cfi = np.asarray(clip_center_frame_idx, dtype=np.int32).reshape(-1)
    ts = np.asarray(clip_times_s, dtype=np.float32).reshape(-1)

    # сортировка по времени (стабильно)
    order = np.argsort(ts, kind="stable")
    emb, tid, cfi, ts = emb[order], tid[order], cfi[order], ts[order]

    if clip_segment_ids is not None and np.asarray(clip_segment_ids).shape[:1] == (C,):
        seg = np.asarray(clip_segment_ids, dtype=np.int32).reshape(-1)[order]
    else:
        seg = np.full((C,), -1, dtype=np.int32)

    out: Dict[str, Any] = {
        "clip_embeddings": emb.astype(np.float32),
        "clip_track_id": tid.astype(np.int32),
        "clip_frame_indices": cfi.astype(np.int32),
        "clip_times_s": ts.astype(np.float32),
        "clip_segment_id": seg.astype(np.int32),
        "clip_count": np.int32(C),
        "num_action_segments": np.int32(int(np.unique(seg[seg >= 0]).size)),
    }

    # классы
    classes_available = clip_logits is not None and np.asarray(clip_logits).shape[:1] == (C,)
    k = int(max(1, min(topk, NUM_ACTION_CLASSES)))
    if classes_available:
        logits = np.asarray(clip_logits, dtype=np.float32)[order]
        probs = _softmax(logits)                                   # (C, 400)
        # top-k per clip
        topk_ids = np.argsort(-probs, axis=1)[:, :k].astype(np.int32)
        topk_probs = np.take_along_axis(probs, topk_ids, axis=1).astype(np.float32)
        out["clip_topk_action_ids"] = topk_ids
        out["clip_topk_probs"] = topk_probs
        # агрегат по видео
        hist = probs.mean(axis=0).astype(np.float32)               # (400,)
        dom = np.argsort(-hist)[:int(top_video)].astype(np.int32)
        out["video_action_hist"] = hist
        out["dominant_action_ids"] = dom
        out["dominant_action_probs"] = hist[dom].astype(np.float32)
    else:
        out["clip_topk_action_ids"] = np.full((C, k), -1, dtype=np.int32)
        out["clip_topk_probs"] = np.zeros((C, k), dtype=np.float32)
        out["video_action_hist"] = np.zeros((NUM_ACTION_CLASSES,), dtype=np.float32)
        out["dominant_action_ids"] = np.full((int(top_video),), -1, dtype=np.int32)
        out["dominant_action_probs"] = np.zeros((int(top_video),), dtype=np.float32)

    out["classes_available"] = np.bool_(classes_available)

    # class_names (стабильная карта 0..399)
    if class_names is not None and len(class_names) == NUM_ACTION_CLASSES:
        cn = [f"{i}:{class_names[i]}" for i in range(NUM_ACTION_CLASSES)]
    else:
        cn = [f"{i}:action_{i}" for i in range(NUM_ACTION_CLASSES)]
    out["class_names"] = np.array(cn, dtype="U")

    # агрегаты по трекам
    valid_tid = tid[tid >= 0]
    uniq = np.unique(valid_tid)
    out["num_tracks"] = np.int32(len(uniq))
    if len(uniq) > 0:
        _, counts = np.unique(valid_tid, return_counts=True)
        out["mean_clips_per_track"] = np.float32(float(np.mean(counts)))
    else:
        out["mean_clips_per_track"] = np.float32(0.0)

    return out


def _empty_v3(emb_dim: int, topk: int, class_names: Optional[List[str]]) -> Dict[str, Any]:
    k = int(max(1, topk))
    if class_names is not None and len(class_names) == NUM_ACTION_CLASSES:
        cn = [f"{i}:{class_names[i]}" for i in range(NUM_ACTION_CLASSES)]
    else:
        cn = [f"{i}:action_{i}" for i in range(NUM_ACTION_CLASSES)]
    return {
        "clip_embeddings": np.zeros((0, emb_dim), dtype=np.float32),
        "clip_track_id": np.zeros((0,), dtype=np.int32),
        "clip_frame_indices": np.zeros((0,), dtype=np.int32),
        "clip_times_s": np.zeros((0,), dtype=np.float32),
        "clip_topk_action_ids": np.zeros((0, k), dtype=np.int32),
        "clip_topk_probs": np.zeros((0, k), dtype=np.float32),
        "clip_segment_id": np.zeros((0,), dtype=np.int32),
        "clip_count": np.int32(0),
        "num_action_segments": np.int32(0),
        "video_action_hist": np.zeros((NUM_ACTION_CLASSES,), dtype=np.float32),
        "dominant_action_ids": np.full((10,), -1, dtype=np.int32),
        "dominant_action_probs": np.zeros((10,), dtype=np.float32),
        "classes_available": np.bool_(False),
        "class_names": np.array(cn, dtype="U"),
        "num_tracks": np.int32(0),
        "mean_clips_per_track": np.float32(0.0),
    }
