"""Общий хелпер сопоставления эмбеддингов с галереей (задача 6 / Q4-подготовка).

Чистый numpy, без сети. Назначение — единая «embeddings-direct» логика top-k,
которую можно переиспользовать в brand/car/face/place хедах вместо per-crop
HTTP /search в Embedding Service (галерея тянется ОДИН раз через
get_all_embeddings(), дальше similarity считается локально).

Контракт:
  * эмбеддинги ОЖИДАЮТСЯ L2-нормализованными → cosine == dot product;
  * порядок строк gallery соответствует label-id (0..A-1), как в semantic-пакете;
  * никаких фолбэков: формы проверяются, иначе ValueError (fail-fast).

Этот модуль НЕ подключён к хедам — он подготовлен для аккуратной миграции
(см. docs/Q4_MIGRATION_EMBEDDINGS_DIRECT.md). Поэтому его внедрение безопасно.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2-нормализация по оси (по умолчанию последней)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def topk_cosine(
    queries: np.ndarray,
    gallery: np.ndarray,
    k: int = 5,
    *,
    assume_normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """top-k по косинусной близости (= dot для L2-normalized).

    Args:
        queries: (M, D) эмбеддинги запросов (кропы/кадры).
        gallery: (A, D) эмбеддинги галереи; строка i == label-id i.
        k: число лучших на запрос (обрезается до A).
        assume_normalized: если False — нормализуем оба входа.

    Returns:
        (idx, score): idx (M, k) int32 — индексы (label-id), score (M, k) float32
        — косинусные близости, отсортированы по убыванию.
    """
    q = np.asarray(queries, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError(f"queries/gallery должны быть 2D, получено {q.shape} / {g.shape}")
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"размерность эмбеддингов не совпадает: D_q={q.shape[1]} D_g={g.shape[1]}")
    if g.shape[0] == 0:
        raise ValueError("пустая галерея")
    if not assume_normalized:
        q = l2_normalize(q)
        g = l2_normalize(g)
    k = int(min(max(k, 1), g.shape[0]))
    sims = q @ g.T  # (M, A)
    # частичный top-k + точная сортировка внутри среза (по убыванию)
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    part_scores = sims[rows, part]
    order = np.argsort(-part_scores, axis=1)
    idx = part[rows, order].astype(np.int32)
    score = part_scores[rows, order].astype(np.float32)
    return idx, score


def aggregate_track_topk(
    per_frame_idx: np.ndarray,
    per_frame_score: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Свести per-frame top-k к per-track top-k агрегацией max-score по label-id.

    Args:
        per_frame_idx: (F, k0) label-id по кадрам трека.
        per_frame_score: (F, k0) близости.
    Returns:
        (labels, scores): по убыванию max-близости; до k элементов.
    """
    pf_idx = np.asarray(per_frame_idx).reshape(-1)
    pf_sc = np.asarray(per_frame_score, dtype=np.float32).reshape(-1)
    if pf_idx.size == 0:
        return np.empty((0,), np.int32), np.empty((0,), np.float32)
    best: dict[int, float] = {}
    for lid, sc in zip(pf_idx.tolist(), pf_sc.tolist()):
        lid = int(lid)
        if lid not in best or sc > best[lid]:
            best[lid] = float(sc)
    items = sorted(best.items(), key=lambda x: -x[1])[: int(max(k, 1))]
    labels = np.array([i for i, _ in items], dtype=np.int32)
    scores = np.array([s for _, s in items], dtype=np.float32)
    return labels, scores
