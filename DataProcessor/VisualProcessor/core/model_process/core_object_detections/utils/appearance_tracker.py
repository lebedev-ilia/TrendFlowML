#!/usr/bin/env python3
"""
Appearance-embedding tracker for core_object_detections.

Design: docs/design/EMBEDDING_TRACKER.md

Восстанавливает persistent track_id (убранный в Audit v3), ассоциируя детекции между
разреженными сэмплированными кадрами по **эмбеддингу бокса** (основной сигнал) + motion-гейту
(вторичный). Эмбеддинг устойчив к разрывам сэмплинга, где IoU/motion рвётся.

Модуль **самодостаточен и не зависит от модели**: association работает на готовых per-detection
эмбеддингах. Эмбеддер (OSNet/CLIP) подаётся снаружи через интерфейс `BoxEmbedder` — это позволяет
юнит-тестировать логику ассоциации на синтетических эмбеддингах без GPU/весов.

Основная точка входа: `track_detections(...) -> track_ids (N, M) int32`  (-1 = не-трек).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy есть в .data_venv; на всякий случай — greedy fallback
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Параметры (совпадают с config core_object_detections.tracking, см. дизайн §5)
# --------------------------------------------------------------------------- #
@dataclass
class TrackerParams:
    classes: Tuple[int, ...] = (0,)          # какие class_id трекаем appearance'ом (0=person)
    sim_gate: float = 0.5                     # мин. cos для матча
    reid_sim_gate: float = 0.6                # мин. cos для воскрешения потерянного трека
    w_app: float = 0.7                        # вес appearance в стоимости
    w_mot: float = 0.3                        # вес motion (нормир. смещение центра)
    max_dist_gate: float = 0.5                # макс. нормир. смещение центра (×gap_scale)
    max_age_steps: int = 3                    # сколько сэмпл-кадров трек «жив» без матча
    max_lost_steps: int = 10                  # сколько держим потерянный трек для re-ID
    gallery_size: int = 16                    # cap галереи эмбеддингов (не используется для EMA, задел)
    ema_alpha: float = 0.5                    # EMA репрезентативного эмбеддинга
    min_track_len: int = 2                    # треки короче — помечаются как шум (в meta), не удаляются

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "TrackerParams":
        cfg = dict(cfg or {})
        classes = cfg.get("classes", [0])
        # допускаем имена классов -> но по контракту person=0; берём числа как есть
        classes_t = tuple(int(c) for c in classes if isinstance(c, (int, float)) or str(c).isdigit()) or (0,)
        return cls(
            classes=classes_t,
            sim_gate=float(cfg.get("sim_gate", 0.5)),
            reid_sim_gate=float(cfg.get("reid_sim_gate", 0.6)),
            w_app=float(cfg.get("w_app", 0.7)),
            w_mot=float(cfg.get("w_mot", 0.3)),
            max_dist_gate=float(cfg.get("max_dist_gate", 0.5)),
            max_age_steps=int(cfg.get("max_age_steps", 3)),
            max_lost_steps=int(cfg.get("max_lost_steps", 10)),
            gallery_size=int(cfg.get("gallery_size", 16)),
            ema_alpha=float(cfg.get("ema_alpha", 0.5)),
            min_track_len=int(cfg.get("min_track_len", 2)),
        )


@dataclass
class _Track:
    track_id: int
    repr_emb: np.ndarray            # L2-normed representative embedding
    last_center: np.ndarray         # (2,) normalized center
    last_frame_pos: int             # позиция в последовательности сэмпл-кадров (не глобальный idx)
    hits: int = 1
    age: int = 0                    # шагов без матча
    first_frame: int = 0            # глобальный frame_index
    last_frame: int = 0             # глобальный frame_index


class BoxEmbedder:
    """Интерфейс эмбеддера бокса. Реализация — в core_object_detections (OSNet/CLIP)."""

    def embed(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """frame_rgb: HxWx3 uint8; boxes_xyxy: (K,4) px. -> (K, d) float32 (не обяз. L2)."""
        raise NotImplementedError


class HistogramBoxEmbedder(BoxEmbedder):
    """
    Zero-dependency baseline-эмбеддер (только numpy+cv2): HSV-гистограмма по трём вертикальным
    зонам бокса (верх/середина/низ — грубый прокси «голова/торс/ноги»). Даёт identity-ish
    дескриптор, достаточный для appearance-ассоциации на коротких разрывах. Прод-замена — OSNet.

    Плюс: пайплайн запускается end-to-end без новых весов/сети. Минус: слабее ReID на близких
    по одежде людях — тогда переключить `tracking.embedder: osnet`.
    """

    def __init__(self, h_bins: int = 8, s_bins: int = 4, zones: int = 3) -> None:
        self.h_bins, self.s_bins, self.zones = int(h_bins), int(s_bins), int(zones)
        try:
            import cv2  # noqa
            self._cv2 = cv2
        except Exception as e:  # pragma: no cover
            raise RuntimeError("HistogramBoxEmbedder требует cv2") from e

    @property
    def dim(self) -> int:
        return self.h_bins * self.s_bins * self.zones

    def embed(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        H, W = frame_rgb.shape[:2]
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        out = np.zeros((len(boxes_xyxy), self.dim), dtype=np.float32)
        for k, b in enumerate(np.asarray(boxes_xyxy, dtype=np.float32)):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
            crop = hsv[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            zone_h = max(1, crop.shape[0] // self.zones)
            parts = []
            for z in range(self.zones):
                zc = crop[z * zone_h:(z + 1) * zone_h if z < self.zones - 1 else crop.shape[0]]
                hist = cv2.calcHist([zc], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
                parts.append(hist.flatten())
            v = np.concatenate(parts).astype(np.float32)
            out[k] = v
        return out


class OSNetBoxEmbedder(BoxEmbedder):
    """
    ReID-эмбеддер бокса на OSNet (torchreid) — сильнее histogram на толпных/длинных видео
    (ASSESSMENT §1.3). Ленивая загрузка: torch + torchreid импортируются в __init__; если их/весов
    нет — RuntimeError с понятным сообщением (вызывающий делает fallback→histogram).

    weights_path: путь к osnet_x1_0 (напр. market1501). Если None — torchreid скачает pretrained
    (нужна сеть) — в offline-проде передавать локальный путь.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu",
                 model_name: str = "osnet_x1_0") -> None:
        try:
            import torch  # noqa
            import torchreid  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OSNetBoxEmbedder требует torch + torchreid (pip install torchreid). "
                f"Импорт не удался: {e}"
            ) from e
        import torch
        import torchreid
        self._torch = torch
        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=(weights_path is None)
        )
        if weights_path:
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
        self.model.eval().to(device)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        try:
            import cv2  # noqa
            self._cv2 = cv2
        except Exception as e:  # pragma: no cover
            raise RuntimeError("OSNetBoxEmbedder требует cv2") from e

    def embed(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        torch = self._torch
        cv2 = self._cv2
        H, W = frame_rgb.shape[:2]
        crops = []
        for b in np.asarray(boxes_xyxy, dtype=np.float32):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            crops.append(cv2.resize(crop, (128, 256)))
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = np.stack(crops).astype(np.float32) / 255.0            # (K,256,128,3)
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        t = (t - self._mean) / self._std
        with torch.inference_mode():
            feats = self.model(t)
        return feats.detach().cpu().numpy().astype(np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    return (x / n).astype(np.float32)


def track_detections(
    *,
    frame_indices: Sequence[int],
    boxes: np.ndarray,            # (N, M, 4) xyxy px
    scores: np.ndarray,          # (N, M)
    class_ids: np.ndarray,       # (N, M)
    valid_mask: np.ndarray,      # (N, M) bool
    centers_norm: Optional[np.ndarray] = None,   # (N, M, 2) in [0,1]; если None — из boxes+frame_wh
    frame_wh: Optional[Tuple[int, int]] = None,  # (W,H) для нормировки центров, если centers_norm=None
    embeddings: Optional[np.ndarray] = None,     # (N, M, d) готовые эмбеддинги (тест/precompute)
    embedder: Optional[BoxEmbedder] = None,      # или эмбеддер + frame_provider
    frame_provider: Optional[Callable[[int], np.ndarray]] = None,  # global_frame_idx -> HxWx3 uint8
    params: Optional[TrackerParams] = None,
    min_person_confidence: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Возвращает (track_ids (N,M) int32, tracks_meta dict).
    track_ids: persistent id ≥0 для трекаемых валидных детекций, -1 иначе.
    Кадры обрабатываются в порядке возрастания frame_index.
    """
    p = params or TrackerParams()
    N, M = valid_mask.shape[:2]
    track_ids = np.full((N, M), -1, dtype=np.int32)
    if N == 0 or M == 0:
        return track_ids, {"num_tracks": 0, "tracks_json": {}}

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    fi = np.asarray(frame_indices, dtype=np.int64).reshape(-1)

    # нормированные центры
    if centers_norm is None:
        if frame_wh is None:
            # без нормировки: используем сырые центры, motion-гейт станет мягче (нормируем позже локально)
            cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
            cy = (boxes[..., 1] + boxes[..., 3]) / 2.0
            diag = float(np.sqrt((boxes[..., 2] - boxes[..., 0]).max() ** 2 + 1) + 1e-6)
            centers_norm = np.stack([cx, cy], axis=-1) / max(diag, 1.0)
        else:
            W, H = frame_wh
            cx = ((boxes[..., 0] + boxes[..., 2]) / 2.0) / max(W - 1, 1)
            cy = ((boxes[..., 1] + boxes[..., 3]) / 2.0) / max(H - 1, 1)
            centers_norm = np.stack([cx, cy], axis=-1).astype(np.float32)
    centers_norm = np.asarray(centers_norm, dtype=np.float32)

    # порядок кадров по возрастанию глобального индекса
    order = np.argsort(fi, kind="stable")

    active: List[_Track] = []
    lost: List[_Track] = []
    next_id = 0
    class_set = set(int(c) for c in p.classes)

    def _emb_for_frame(n: int, slots: List[int]) -> np.ndarray:
        if embeddings is not None:
            e = embeddings[n, slots, :]
            return _l2(np.asarray(e, dtype=np.float32))
        if embedder is not None and frame_provider is not None:
            frame = frame_provider(int(fi[n]))
            bx = boxes[n, slots, :]
            e = embedder.embed(frame, bx)
            return _l2(np.asarray(e, dtype=np.float32))
        raise ValueError("track_detections: нужно либо embeddings, либо (embedder+frame_provider)")

    for step, n in enumerate(order):
        # валидные трекаемые слоты кадра
        slots = [
            j for j in range(M)
            if valid_mask[n, j]
            and int(class_ids[n, j]) in class_set
            and float(scores[n, j]) >= float(min_person_confidence)
        ]
        # состарить все активные треки на 1 шаг (снимем age у сматченных)
        for t in active:
            t.age += 1

        if not slots:
            # никого не матчим; ретайр по возрасту
            active, newly_lost = _retire(active, p)
            lost = _prune_lost(lost + newly_lost, p)
            continue

        emb = _emb_for_frame(n, slots)                    # (K, d) L2
        cen = centers_norm[n, slots, :]                   # (K, 2)
        K = len(slots)

        assigned = [False] * K
        # --- 1. матч с активными треками (Hungarian по гейтованной стоимости) ---
        if active:
            T = len(active)
            repr_mat = _l2(np.stack([t.repr_emb for t in active], axis=0))  # (T,d)
            cos = repr_mat @ emb.T                          # (T,K) в [-1,1]
            cen_t = np.stack([t.last_center for t in active], axis=0)       # (T,2)
            dist = np.linalg.norm(cen_t[:, None, :] - cen[None, :, :], axis=2)  # (T,K)
            gap = np.array([max(step - t.last_frame_pos, 1) for t in active], dtype=np.float32)
            gate = (p.max_dist_gate * gap)[:, None]         # (T,1)
            app = 1.0 - cos                                 # (T,K)
            mot = dist                                       # normalized
            cost = p.w_app * app + p.w_mot * mot
            forbidden = (cos < p.sim_gate) | (dist > gate)
            BIG = 1e6
            cost_g = np.where(forbidden, BIG, cost)
            rows, cols = _assign(cost_g)
            for r, c in zip(rows, cols):
                if cost_g[r, c] >= BIG:
                    continue
                t = active[r]
                t.repr_emb = _l2(p.ema_alpha * t.repr_emb + (1 - p.ema_alpha) * emb[c])
                t.last_center = cen[c]
                t.last_frame_pos = step
                t.last_frame = int(fi[n])
                t.hits += 1
                t.age = 0
                track_ids[n, slots[c]] = t.track_id
                assigned[c] = True

        # --- 2. re-ID из потерянных, затем рождение ---
        for c in range(K):
            if assigned[c]:
                continue
            revived = None
            if lost:
                lost_mat = _l2(np.stack([t.repr_emb for t in lost], axis=0))
                cos_l = lost_mat @ emb[c]
                bi = int(np.argmax(cos_l))
                if float(cos_l[bi]) >= p.reid_sim_gate:
                    revived = lost.pop(bi)
            if revived is not None:
                revived.repr_emb = _l2(p.ema_alpha * revived.repr_emb + (1 - p.ema_alpha) * emb[c])
                revived.last_center = cen[c]
                revived.last_frame_pos = step
                revived.last_frame = int(fi[n])
                revived.hits += 1
                revived.age = 0
                active.append(revived)
                track_ids[n, slots[c]] = revived.track_id
            else:
                t = _Track(
                    track_id=next_id, repr_emb=emb[c], last_center=cen[c],
                    last_frame_pos=step, hits=1, age=0,
                    first_frame=int(fi[n]), last_frame=int(fi[n]),
                )
                next_id += 1
                active.append(t)
                track_ids[n, slots[c]] = t.track_id

        active, newly_lost = _retire(active, p)
        lost = _prune_lost(lost + newly_lost, p)

    # --- meta ---
    all_tracks = active + lost
    tracks_json: Dict[str, Any] = {}
    # соберём длины по track_ids (учёт и уже сброшенных треков через проход по массиву)
    ids, counts = np.unique(track_ids[track_ids >= 0], return_counts=True)
    len_by_id = {int(i): int(c) for i, c in zip(ids, counts)}
    for tid, ln in len_by_id.items():
        tracks_json[str(tid)] = {"len": ln, "class": "person" if 0 in class_set else "obj"}
    meta = {
        "num_tracks": int(len(len_by_id)),
        "mean_track_len": float(np.mean(list(len_by_id.values()))) if len_by_id else 0.0,
        "median_track_len": float(np.median(list(len_by_id.values()))) if len_by_id else 0.0,
        "frac_single_len": (
            float(np.mean([1.0 if v <= 1 else 0.0 for v in len_by_id.values()])) if len_by_id else 0.0
        ),
        "tracks_json": tracks_json,
    }
    return track_ids, meta


def _assign(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    if _HAVE_SCIPY:
        r, c = linear_sum_assignment(cost)
        return list(r), list(c)
    # greedy fallback: по возрастанию стоимости
    T, K = cost.shape
    pairs = sorted(((cost[i, j], i, j) for i in range(T) for j in range(K)))
    used_r, used_c, rr, cc = set(), set(), [], []
    for _, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j); rr.append(i); cc.append(j)
    return rr, cc


def _retire(active: List[_Track], p: TrackerParams) -> Tuple[List[_Track], List[_Track]]:
    keep, lost = [], []
    for t in active:
        if t.age > p.max_age_steps:
            lost.append(t)
        else:
            keep.append(t)
    return keep, lost


def _prune_lost(lost: List[_Track], p: TrackerParams) -> List[_Track]:
    return [t for t in lost if t.age <= (p.max_age_steps + p.max_lost_steps)]
