"""
Содержит:
- библиотечный класс `TextVideoInteractionPipeline` (feature extraction given OCR detections)
- production wrapper `TextScoringModule(BaseModule)`:
  consumer OCR-артефакта (NPZ) и выдача NPZ результатов по стандарту.
"""

from __future__ import annotations

import hashlib
import json
import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

MODULE_NAME = "text_scoring"
VERSION = "2.0.1"
SCHEMA_VERSION = "text_scoring_npz_v2"
ARTIFACT_FILENAME = "text_scoring.npz"


def _resource_profile_snapshot() -> Dict[str, Any]:
    """
    Best-effort resource snapshot for audit/profiling.
    Enabled only when VP_RESOURCE_PROFILE=1|true|yes.
    """
    v = str(os.environ.get("VP_RESOURCE_PROFILE") or "").strip().lower()
    if v not in ("1", "true", "yes", "y", "on"):
        return {}

    out: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        rss = int(getattr(p.memory_info(), "rss", 0) or 0)
        out["rss_bytes"] = rss
        out["rss_mib"] = float(rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                out["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                out["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            except Exception:
                pass
    except Exception:
        pass

    return out


# Stable, fixed list of model-facing scalar features (tabular).
# NOTE: booleans are stored as 0/1 floats for a single float32 vector.
_FEATURE_NAMES_V1: List[str] = [
    # global
    "text_present",
    "text_frames_ratio",
    "text_count_mean",
    "text_count_p95",
    "num_unique_texts",
    # sync / alignment
    "text_action_sync_score",
    "text_motion_alignment",
    "text_motion_alignment_windowed",
    "multimodal_attention_boost_score",
    "multimodal_attention_boost_position",
    # continuity
    "text_on_screen_continuity",
    "text_on_screen_continuity_median",
    "text_on_screen_continuity_max",
    "text_on_screen_continuity_std",
    "text_on_screen_continuity_normalized",
    "text_switch_rate",
    "time_to_first_text_sec",
    "time_to_first_text_position",
    "text_area_fraction",
    # CTA
    "cta_presence",
    "cta_strength",
    "persistent_cta_flag",
    "cta_timestamp",
    "cta_first_timestamp",
    "cta_mean_timestamp",
    "cta_last_timestamp",
    "cta_first_position",
    "cta_mean_position",
    "cta_last_position",
    # readability / extra
    "text_readability_score",
    "ocr_language_entropy",
    "text_movement_speed",
    "text_emphasis_peaks_count",
    # debug-ish but tiny
    "ocr_raw_count",
    "ocr_unique_elements_count",
]


def _as_float_feature(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if bool(v) else 0.0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return float("nan")


def _utc_iso_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl`. Backend tails this file.
    """
    try:
        from pathlib import Path as _Path

        run_rs = _Path(rs_path).resolve()
        rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
        runs_root = rs_base.parent
        platform_id = str(event.get("platform_id") or "")
        video_id = str(event.get("video_id") or "")
        run_id = str(event.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            return
        p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_progress(
    *,
    rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    done: int,
    total: int,
    stage: str,
) -> None:
    if total <= 0:
        return
    progress = float(done) / float(total)
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": _utc_iso_now(),
            "scope": "progress",
            "processor": "visual",
            "component": MODULE_NAME,
            "status": "running",
            "progress": progress,
            "done": int(done),
            "total": int(total),
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


def _times_s_from_union(*, frame_manager: FrameManager, frame_indices: np.ndarray) -> np.ndarray:
    """
    Baseline contract: time-axis is union_timestamps_sec[frame_indices] (no-fallback).
    """
    uts = (frame_manager.meta or {}).get("union_timestamps_sec")
    if not isinstance(uts, list) or not uts:
        raise RuntimeError(f"{MODULE_NAME} | missing/invalid union_timestamps_sec in frames metadata (no-fallback)")
    uts = np.asarray(uts, dtype=np.float32).reshape(-1)
    if frame_indices.size == 0:
        raise RuntimeError(f"{MODULE_NAME} | frame_indices is empty (no-fallback)")
    if int(np.max(frame_indices)) >= int(uts.size) or int(np.min(frame_indices)) < 0:
        raise RuntimeError(f"{MODULE_NAME} | frame_indices out of bounds for union_timestamps_sec (no-fallback)")
    return uts[frame_indices.astype(np.int64)].astype(np.float32)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update((s or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _load_motion_signal_optional(*, module: "TextScoringModule", fi: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Best-effort loader for motion signal aligned to requested frame_indices.
    Prefer `optical_flow/optical_flow.npz` (module output), fallback to `core_optical_flow/flow.npz`.
    Returns (signal_or_none, source_name).
    """
    if module.rs_path is None:
        return None, "none"

    # 1) optical_flow consumer artifact (preferred)
    try:
        p = os.path.join(str(module.rs_path), "optical_flow", "optical_flow.npz")
        if os.path.exists(p):
            npz = np.load(p, allow_pickle=True)
            try:
                idx = npz.get("frame_indices")
                mot = npz.get("motion_norm_per_sec_mean")
                if idx is None or mot is None:
                    return None, "optical_flow_missing_keys"
                idx = np.asarray(idx, dtype=np.int32).reshape(-1)
                mot = np.asarray(mot, dtype=np.float32).reshape(-1)
                mapping = {int(x): i for i, x in enumerate(idx.tolist())}
                pos = [mapping.get(int(x), -1) for x in fi.tolist()]
                if any(p0 < 0 for p0 in pos):
                    return None, "optical_flow_not_covering"
                return mot[np.asarray(pos, dtype=np.int64)], "optical_flow"
            finally:
                try:
                    npz.close()
                except Exception:
                    pass
    except Exception:
        return None, "optical_flow_error"

    # 2) core_optical_flow provider (fallback)
    try:
        p2 = os.path.join(str(module.rs_path), "core_optical_flow", "flow.npz")
        if os.path.exists(p2):
            npz2 = np.load(p2, allow_pickle=True)
            try:
                idx2 = npz2.get("frame_indices")
                mot2 = npz2.get("motion_norm_per_sec_mean")
                if idx2 is None or mot2 is None:
                    return None, "core_optical_flow_missing_keys"
                idx2 = np.asarray(idx2, dtype=np.int32).reshape(-1)
                mot2 = np.asarray(mot2, dtype=np.float32).reshape(-1)
                mapping2 = {int(x): i for i, x in enumerate(idx2.tolist())}
                pos2 = [mapping2.get(int(x), -1) for x in fi.tolist()]
                if any(p0 < 0 for p0 in pos2):
                    return None, "core_optical_flow_not_covering"
                return mot2[np.asarray(pos2, dtype=np.int64)], "core_optical_flow"
            finally:
                try:
                    npz2.close()
                except Exception:
                    pass
    except Exception:
        return None, "core_optical_flow_error"

    return None, "none"

class TextVideoInteractionPipeline:
    """
    Пайплайн для извлечения фичей взаимодействия текста и видео.
    Вход: кадры, OCR с bbox, motion/face/audio пики
    Выход: словарь фичей на видео
    """

    def __init__(
        self,
        video_fps: int = 30,  # legacy/compat only; baseline uses time_s from union_timestamps_sec
        frame_width: int | None = None,
        frame_height: int | None = None,
        alignment_window_seconds: float = 0.5,
        motion_weight: float = 0.4,
        face_weight: float = 0.3,
        audio_weight: float = 0.3,
        min_ocr_confidence: float = 0.4,
    ):
        self.video_fps = float(video_fps)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alignment_window_seconds = float(alignment_window_seconds)
        # нормализуем веса, чтобы сумма была 1.0
        w_sum = motion_weight + face_weight + audio_weight
        if w_sum <= 0:
            self.motion_weight = 1.0
            self.face_weight = 0.0
            self.audio_weight = 0.0
        else:
            self.motion_weight = motion_weight / w_sum
            self.face_weight = face_weight / w_sum
            self.audio_weight = audio_weight / w_sum
        self.min_ocr_confidence = float(min_ocr_confidence)

    @staticmethod
    def _nearest_index(times_s: np.ndarray, t: float) -> int:
        if times_s.size == 0:
            return -1
        return int(np.argmin(np.abs(times_s.astype(np.float32) - float(t))))

    @staticmethod
    def _window_mask(times_s: np.ndarray, t: float, w: float) -> np.ndarray:
        if times_s.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.abs(times_s.astype(np.float32) - float(t)) <= float(w)

    @staticmethod
    def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Intersection over union для bbox"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    @staticmethod
    def _normalize_signal(signal: np.ndarray) -> np.ndarray:
        """Нормализация и сглаживание сигнала для уменьшения шумов"""
        if len(signal) == 0:
            return np.array([])
        signal = np.array(signal, dtype=np.float32)
        signal = gaussian_filter1d(signal, sigma=1)
        max_val = float(signal.max())
        if max_val <= 0:
            return np.zeros_like(signal, dtype=np.float32)
        return signal / max_val

    @staticmethod
    def _zscore(signal: np.ndarray) -> np.ndarray:
        """Перевод сигнала в z-score по видео."""
        if len(signal) == 0:
            return np.array([], dtype=np.float32)
        signal = np.array(signal, dtype=np.float32)
        mean = float(signal.mean())
        std = float(signal.std()) + 1e-6
        return (signal - mean) / std

    @staticmethod
    def _trimmed_mean(values: List[float], proportion_to_cut: float = 0.1) -> float:
        """Робастное среднее: усреднение по центральной части распределения."""
        if not values:
            return 0.0
        arr = np.sort(np.asarray(values, dtype=np.float32))
        n = len(arr)
        k = int(n * proportion_to_cut)
        if k * 2 >= n:
            return float(arr.mean())
        return float(arr[k : n - k].mean())

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Простая нормализация текста: lower + обрезка пробелов и пунктуации по краям."""
        import re

        s = (s or "").lower()
        s = s.strip()
        # Удаляем лишнюю пунктуацию по краям
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        return s

    @staticmethod
    def _normalized_text_similarity(a: str, b: str) -> float:
        """
        Нормализованная похожесть строк (0..1) через Levenshtein-подобное расстояние.
        Реализуем простую динамику, чтобы не тянуть внешние зависимости.
        """
        a = a or ""
        b = b or ""
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        # DP по двум строкам
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,       # удаление
                    dp[j - 1] + 1,   # вставка
                    prev + cost,     # замена
                )
                prev = cur
        dist = dp[lb]
        max_len = max(la, lb)
        return 1.0 - float(dist) / float(max_len)

    @staticmethod
    def _shannon_entropy(counts: Dict[str, int]) -> float:
        """Энтропия распределения языков."""
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        probs = [c / total for c in counts.values() if c > 0]
        return float(-sum(p * math.log(p + 1e-12) for p in probs))

    def _group_ocr_elements(self, ocr_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Группировка OCR-детекций в уникальные текстовые элементы по IoU + текстовой похожести.

        Возвращает:
        - raw_detections: отфильтрованный по confidence список исходных детекций (+ time_s при необходимости)
        - unique_elements: список агрегированных элементов с полями:
            - text_raw, text_norm, language
            - frames, times, first_frame, last_frame, first_time, last_time
            - bbox_median, aggregated_confidence
        """
        if not ocr_data:
            return [], []

        # фильтрация по confidence
        raw_filtered = []
        for d in ocr_data:
            conf = float(d.get("confidence", 1.0))
            if conf < self.min_ocr_confidence:
                continue
            # гарантируем наличие time_s
            if "time_s" not in d:
                d = dict(d)
                d["time_s"] = d["frame"] / self.video_fps
            raw_filtered.append(d)

        if not raw_filtered:
            return [], []

        elements: List[Dict[str, Any]] = []
        for det in raw_filtered:
            frame_idx = det["frame"]
            time_s = det.get("time_s", frame_idx / self.video_fps)
            bbox = det["bbox"]
            text_raw = det.get("text_raw", det.get("text", ""))
            text_norm = det.get("text_norm", self._normalize_text(text_raw))
            language = det.get("language", None)

            matched_idx = None
            best_score = 0.0
            for i, elem in enumerate(elements):
                iou = self._iou(bbox, elem["bbox_median"])
                if iou < 0.6:
                    continue
                sim = self._normalized_text_similarity(text_norm, elem["text_norm"])
                score = 0.5 * iou + 0.5 * sim
                if score > 0.8 and score > best_score:
                    best_score = score
                    matched_idx = i

            if matched_idx is None:
                # создаём новый элемент и сразу инициализируем bbox_median,
                # чтобы его можно было использовать при последующих IoU-сравнениях
                elements.append(
                    {
                        "text_raw": text_raw,
                        "text_norm": text_norm,
                        "language": language,
                        "frames": [frame_idx],
                        "times": [time_s],
                        "bboxes": [bbox],
                        "bbox_median": bbox,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "first_time": time_s,
                        "last_time": time_s,
                        "confidences": [float(det.get("confidence", 1.0))],
                        "is_cta_candidate": bool(det.get("is_cta_candidate", False)),
                    }
                )
            else:
                # обновляем существующий элемент и переоцениваем bbox_median на основе всех bboxes
                elem = elements[matched_idx]
                elem["frames"].append(frame_idx)
                elem["times"].append(time_s)
                elem["bboxes"].append(bbox)
                elem["last_frame"] = frame_idx
                elem["last_time"] = time_s
                elem["confidences"].append(float(det.get("confidence", 1.0)))
                elem["is_cta_candidate"] = elem["is_cta_candidate"] or bool(det.get("is_cta_candidate", False))

                # пересчитываем bbox_median для стабильной работы IoU
                xs1, ys1, xs2, ys2 = [], [], [], []
                for (x1, y1, x2, y2) in elem["bboxes"]:
                    xs1.append(x1)
                    ys1.append(y1)
                    xs2.append(x2)
                    ys2.append(y2)
                elem["bbox_median"] = (
                    float(np.median(xs1)),
                    float(np.median(ys1)),
                    float(np.median(xs2)),
                    float(np.median(ys2)),
                )

        # агрегируем bbox и confidence
        for elem in elements:
            # bbox_median уже посчитан выше; здесь только гарантируем aggregated_confidence
            confs = np.asarray(elem["confidences"], dtype=np.float32)
            elem["aggregated_confidence"] = float(confs.mean()) if confs.size else 0.0

        return raw_filtered, elements

    def _compute_text_area_fraction(self, elements: List[Dict[str, Any]]) -> Tuple[float, List[float]]:
        """
        Оценка доли площади кадра, занятой текстом.
        Возвращает:
        - средняя доля площади текста по уникальным элементам
        - список долей по элементам
        """
        if not elements or not self.frame_width or not self.frame_height:
            return 0.0, []
        frame_area = float(self.frame_width * self.frame_height)
        fractions = []
        for elem in elements:
            x1, y1, x2, y2 = elem["bbox_median"]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            area = w * h
            fractions.append(float(area / (frame_area + 1e-6)))
        if not fractions:
            return 0.0, []
        arr = np.asarray(fractions, dtype=np.float32)
        return float(arr.mean()), fractions

    @staticmethod
    def _readability_score(text_norm: str) -> float:
        """
        Простейший скор читаемости: короткие, хорошо структурированные CTA/заголовки получают больший скор.
        """
        if not text_norm:
            return 0.0
        import re

        # убираем лишние пробелы
        text = re.sub(r"\s+", " ", text_norm.strip())
        words = text.split(" ")
        num_words = len(words)
        num_chars = len(text)
        num_punct = len(re.findall(r"[^\w\s]", text))
        avg_word_len = num_chars / max(num_words, 1)
        punct_ratio = num_punct / max(num_chars, 1)
        # эвристика: 1.0 для коротких заголовков с малым количеством пунктуации
        score = 1.0
        score *= 1.0 / (1.0 + max(0.0, (num_words - 6) / 10.0))
        score *= 1.0 / (1.0 + max(0.0, (avg_word_len - 6) / 10.0))
        score *= 1.0 / (1.0 + 5.0 * punct_ratio)
        return float(max(0.0, min(1.0, score)))

    def extract_features(
        self,
        *,
        ocr_data: List[Dict[str, Any]],
        frame_indices: np.ndarray,
        times_s: np.ndarray,
        motion_signal: np.ndarray,
        face_signal: np.ndarray,
        audio_signal: Optional[np.ndarray] = None,
        enable_text_peaks: bool = False,
        enable_language_entropy: bool = False,
        enable_text_movement_speed: bool = False,
    ) -> Dict[str, Any]:
        """
        ocr_data: List[Dict] = [
            {"frame": 10, "bbox": (x1,y1,x2,y2), "text": "...", "confidence": 0.95, "is_cta": False},
            ...
        ]
        frame_indices/times_s: module sampling group (Segmenter-owned).
        motion_signal/face_signal/audio_signal: arrays aligned to times_s (length N).
        """
        features: Dict[str, Any] = defaultdict(float)

        # --- Preprocess & group OCR detections ---
        raw_ocr, unique_elements = self._group_ocr_elements(ocr_data)

        frame_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        times_s = np.asarray(times_s, dtype=np.float32).reshape(-1)
        motion_signal = np.asarray(motion_signal, dtype=np.float32).reshape(-1)
        face_signal = np.asarray(face_signal, dtype=np.float32).reshape(-1)
        if audio_signal is None:
            audio_signal = np.zeros((times_s.size,), dtype=np.float32)
        else:
            audio_signal = np.asarray(audio_signal, dtype=np.float32).reshape(-1)

        if times_s.shape[0] != frame_indices.shape[0]:
            raise RuntimeError("text_scoring | times_s/frame_indices length mismatch (contract)")
        if motion_signal.shape[0] != times_s.shape[0] or face_signal.shape[0] != times_s.shape[0] or audio_signal.shape[0] != times_s.shape[0]:
            raise RuntimeError("text_scoring | signal arrays must be aligned to times_s (contract)")

        # нормализованные сигналы (0..1) для alignment
        motion_norm = self._normalize_signal(motion_signal)
        face_norm = self._normalize_signal(face_signal)
        audio_norm = self._normalize_signal(audio_signal)

        # z-score для energy-based оконных метрик
        motion_z = self._zscore(motion_signal)

        # ---------- 1. Text → Action / Motion ----------
        text_action_scores_windowed: List[float] = []
        text_motion_align_scores: List[float] = []
        text_motion_align_scores_windowed: List[float] = []
        multimodal_per_text: List[Tuple[float, float]] = []  # (score, time_s)

        # CTA промежуточные агрегаты
        cta_elements_indices: List[int] = []
        cta_multimodal_scores: List[float] = []

        w = float(self.alignment_window_seconds)
        total_video_seconds = float(times_s[-1] - times_s[0]) if times_s.size >= 2 else 0.0

        for idx, elem in enumerate(unique_elements):
            if not elem["frames"]:
                continue
            center_time = float(elem["first_time"])
            m = self._window_mask(times_s, center_time, w)

            # motion z-score в окне (time-based)
            if m.size and np.any(m) and len(motion_z) > 0:
                window_motion = motion_z[m]
                # учитываем максимум/среднее в окне
                max_energy = float(np.max(window_motion))
                mean_energy = float(np.mean(window_motion))
                # комбинируем как среднее max и mean
                window_score = 0.5 * max_energy + 0.5 * mean_energy
                text_action_scores_windowed.append(window_score)

            # Alignment score at nearest sampled frame (time-based)
            i0 = self._nearest_index(times_s, center_time)
            if i0 >= 0 and i0 < int(times_s.size):
                m_val = float(motion_norm[i0])
                f_val = float(face_norm[i0])
                a_val = float(audio_norm[i0])
            else:
                m_val = f_val = a_val = 0.0

            multimodal_score = (
                self.motion_weight * m_val
                + self.face_weight * f_val
                + self.audio_weight * a_val
            )
            text_motion_align_scores.append(multimodal_score)

            # windowed alignment: max within [t-w, t+w]
            if m.size and np.any(m):
                window_multimodal = (
                    self.motion_weight * motion_norm[m]
                    + self.face_weight * face_norm[m]
                    + self.audio_weight * audio_norm[m]
                )
                wmax = float(np.max(window_multimodal)) if window_multimodal.size else 0.0
                text_motion_align_scores_windowed.append(wmax)
                multimodal_per_text.append((wmax, center_time))
            else:
                text_motion_align_scores_windowed.append(multimodal_score)
                multimodal_per_text.append((multimodal_score, center_time))

            # CTA candidate (по флагу из OCR или по тексту)
            text_norm = elem["text_norm"]
            is_cta_flag = bool(elem.get("is_cta_candidate", False))
            is_cta_lexical = False
            if text_norm:
                cta_keywords = [
                    "subscribe",
                    "follow",
                    "like",
                    "link in bio",
                    "click",
                    "watch",
                    "подпишись",
                    "подписаться",
                    "ставь лайк",
                    "ссылка в описании",
                ]
                for kw in cta_keywords:
                    sim = self._normalized_text_similarity(text_norm, self._normalize_text(kw))
                    if sim >= 0.75 or kw in text_norm:
                        is_cta_lexical = True
                        break

            is_cta = is_cta_flag or is_cta_lexical
            if is_cta:
                cta_elements_indices.append(idx)
                cta_multimodal_scores.append(multimodal_score)

        # ---------- Aggregate features ----------
        # Text → Action / Motion: робастное среднее по окнам (z-score motion)
        features["text_action_sync_score"] = self._trimmed_mean(text_action_scores_windowed)

        # Alignment: среднее и "оконное" (максимум в окне)
        features["text_motion_alignment"] = float(
            np.mean(text_motion_align_scores) if text_motion_align_scores else 0.0
        )
        features["text_motion_alignment_windowed"] = float(
            np.mean(text_motion_align_scores_windowed) if text_motion_align_scores_windowed else 0.0
        )

        # Multimodal attention boost: максимум + относительная позиция
        if multimodal_per_text:
            scores_arr = np.asarray([s for s, _ in multimodal_per_text], dtype=np.float32)
            times_arr = np.asarray([t for _, t in multimodal_per_text], dtype=np.float32)
            max_idx = int(np.argmax(scores_arr))
            features["multimodal_attention_boost_score"] = float(scores_arr[max_idx])
            rel_pos = (
                float(times_arr[max_idx] / max(total_video_seconds, 1e-6))
                if total_video_seconds > 0
                else 0.0
            )
            features["multimodal_attention_boost_position"] = rel_pos
        else:
            features["multimodal_attention_boost_score"] = 0.0
            features["multimodal_attention_boost_position"] = 0.0

        # ---------- 2. Text Duration and Continuity ----------
        durations_sec: List[float] = []
        for elem in unique_elements:
            if not elem["frames"]:
                continue
            durations_sec.append(float(elem["last_time"] - elem["first_time"]))

        if durations_sec:
            d_arr = np.asarray(durations_sec, dtype=np.float32)
            mean_dur = float(d_arr.mean())
            features["text_on_screen_continuity"] = mean_dur
            features["text_on_screen_continuity_median"] = float(np.median(d_arr))
            features["text_on_screen_continuity_max"] = float(d_arr.max())
            features["text_on_screen_continuity_std"] = float(d_arr.std())
            features["text_on_screen_continuity_normalized"] = float(
                mean_dur / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
        else:
            features["text_on_screen_continuity"] = 0.0
            features["text_on_screen_continuity_median"] = 0.0
            features["text_on_screen_continuity_max"] = 0.0
            features["text_on_screen_continuity_std"] = 0.0
            features["text_on_screen_continuity_normalized"] = 0.0

        # text_switch_rate: число уникальных элементов / длительность видео
        num_unique_texts = len(unique_elements)
        features["num_unique_texts"] = int(num_unique_texts)
        features["text_switch_rate"] = (
            float(num_unique_texts) / max(total_video_seconds, 1e-6)
            if total_video_seconds > 0
            else 0.0
        )

        # time_to_first_text
        if unique_elements and times_s.size:
            first_time = float(min(elem["first_time"] for elem in unique_elements))
            features["time_to_first_text_sec"] = float(max(0.0, first_time - float(times_s[0])))
            features["time_to_first_text_position"] = float(
                (max(0.0, first_time - float(times_s[0]))) / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
        else:
            features["time_to_first_text_sec"] = None
            features["time_to_first_text_position"] = None

        # text_area_fraction
        mean_text_area_fraction, per_elem_area_frac = self._compute_text_area_fraction(unique_elements)
        features["text_area_fraction"] = mean_text_area_fraction

        # ---------- 3. Call-to-Action (CTA) Detection ----------
        cta_times_sec: List[float] = []
        cta_durations_sec: List[float] = []
        cta_readability_scores: List[float] = []
        cta_confidences: List[float] = []

        for idx in cta_elements_indices:
            elem = unique_elements[idx]
            cta_times_sec.append(elem["first_time"])
            cta_durations_sec.append(float(elem["last_time"] - elem["first_time"]))
            cta_readability_scores.append(self._readability_score(elem["text_norm"]))
            cta_confidences.append(elem.get("aggregated_confidence", 0.0))

        # cta_presence как вероятность (0..1) на основе числа CTA-элементов и их уверенности
        if cta_elements_indices:
            base_prob = min(1.0, len(cta_elements_indices) / max(num_unique_texts, 1) * 1.5)
            conf_mean = float(np.mean(cta_confidences)) if cta_confidences else 0.5
            features["cta_presence"] = float(max(0.0, min(1.0, 0.5 * base_prob + 0.5 * conf_mean)))
        else:
            features["cta_presence"] = 0.0

        if cta_times_sec:
            times_arr = np.asarray(cta_times_sec, dtype=np.float32)
            first_t = float(times_arr.min())
            mean_t = float(times_arr.mean())
            last_t = float(times_arr.max())
            features["cta_first_timestamp"] = first_t
            features["cta_mean_timestamp"] = mean_t
            features["cta_last_timestamp"] = last_t
            features["cta_first_position"] = float(
                first_t / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
            features["cta_mean_position"] = float(
                mean_t / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
            features["cta_last_position"] = float(
                last_t / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
            # оставляем cta_timestamp для обратной совместимости (mean)
            features["cta_timestamp"] = mean_t
        else:
            features["cta_first_timestamp"] = None
            features["cta_mean_timestamp"] = None
            features["cta_last_timestamp"] = None
            features["cta_first_position"] = None
            features["cta_mean_position"] = None
            features["cta_last_position"] = None
            features["cta_timestamp"] = None

        # cta_strength как нормализованный мультимодальный скор в CTA-элементах
        if cta_multimodal_scores:
            c_arr = np.asarray(cta_multimodal_scores, dtype=np.float32)
            features["cta_strength"] = float(np.clip(c_arr.mean(), 0.0, 1.0))
        else:
            features["cta_strength"] = 0.0

        # persistent_cta_flag: CTA, который держится дольше 3 секунд
        persistent = any(dur > 3.0 for dur in cta_durations_sec)
        features["persistent_cta_flag"] = bool(persistent)

        # ---------- 4. Text Emphasis Peaks (optional; often noisy) ----------
        if bool(enable_text_peaks) and text_motion_align_scores:
            scores_arr = np.asarray(text_motion_align_scores, dtype=np.float32)
            peaks, props = find_peaks(scores_arr, prominence=0.1, distance=1)
            features["text_emphasis_peak_flags"] = peaks.tolist()
            features["text_emphasis_peak_prominence"] = (
                props.get("prominences", np.zeros_like(peaks, dtype=np.float32)).tolist()
                if "prominences" in props
                else []
            )
            peak_times = []
            for pi in peaks:
                if pi < len(unique_elements):
                    peak_times.append(unique_elements[int(pi)]["first_time"])
            if peak_times and total_video_seconds > 0:
                features["text_emphasis_peak_positions"] = [float(t / max(total_video_seconds, 1e-6)) for t in peak_times]
            else:
                features["text_emphasis_peak_positions"] = []
        else:
            # omit by default (noise)
            features.pop("text_emphasis_peak_flags", None)
            features.pop("text_emphasis_peak_prominence", None)
            features.pop("text_emphasis_peak_positions", None)

        # ---------- 5. Дополнительные агрегаты ----------
        # text_readability_score (средний по уникальным элементам)
        readability_scores = [
            self._readability_score(elem["text_norm"]) for elem in unique_elements
        ]
        features["text_readability_score"] = float(
            np.mean(readability_scores) if readability_scores else 0.0
        )

        # ocr_language_entropy (optional; often noisy/depends on OCR)
        if bool(enable_language_entropy):
            lang_counts: Dict[str, int] = defaultdict(int)
            for elem in unique_elements:
                lang = elem.get("language")
                if lang:
                    lang_counts[str(lang)] += 1
            features["ocr_language_entropy"] = self._shannon_entropy(lang_counts)

        # text_movement_speed (optional; often noisy and depends on bbox stability + sampling)
        if bool(enable_text_movement_speed):
            movement_speeds = []
            if self.frame_width and self.frame_height:
                diag = math.sqrt(self.frame_width**2 + self.frame_height**2)
                for elem in unique_elements:
                    times = elem.get("times") or []
                    bboxes = elem.get("bboxes") or []
                    if len(times) < 2 or len(bboxes) < 2:
                        continue
                    dist_sum = 0.0
                    time_sum = 0.0
                    for i in range(1, len(times)):
                        (x1a, y1a, x2a, y2a) = bboxes[i - 1]
                        (x1b, y1b, x2b, y2b) = bboxes[i]
                        cxa = (x1a + x2a) / 2.0
                        cya = (y1a + y2a) / 2.0
                        cxb = (x1b + x2b) / 2.0
                        cyb = (y1b + y2b) / 2.0
                        dist = math.sqrt((cxb - cxa) ** 2 + (cyb - cya) ** 2) / (diag + 1e-6)
                        dt = float(times[i] - times[i - 1])
                        if dt > 0:
                            dist_sum += dist
                            time_sum += dt
                    if time_sum > 0:
                        movement_speeds.append(dist_sum / time_sum)
            features["text_movement_speed"] = float(np.mean(movement_speeds) if movement_speeds else 0.0)

        # --- Raw / grouped OCR data for explainability ---
        features["ocr_raw"] = raw_ocr
        features["ocr_unique_elements"] = [
            {
                "text_raw": elem["text_raw"],
                "text_norm": elem["text_norm"],
                "language": elem.get("language"),
                "first_frame": elem["first_frame"],
                "last_frame": elem["last_frame"],
                "first_time": elem["first_time"],
                "last_time": elem["last_time"],
                "bbox_median": elem["bbox_median"],
                "aggregated_confidence": elem["aggregated_confidence"],
            }
            for elem in unique_elements
        ]

        return dict(features)


def _find_ocr_npz(rs_path: str) -> Optional[str]:
    """
    Canonical location (proposed):
    - `<rs_path>/text_ocr/ocr.npz`

    Compatibility:
    - `<rs_path>/ocr/ocr.npz`
    - `<rs_path>/text_scoring/ocr.npz` (legacy custom runs)
    """
    candidates = [
        # Baseline v1: prefer ocr_extractor (core provider)
        os.path.join(rs_path, "ocr_extractor", "ocr.npz"),
        # Legacy/compat:
        os.path.join(rs_path, "text_ocr", "ocr.npz"),
        os.path.join(rs_path, "ocr", "ocr.npz"),
        os.path.join(rs_path, "text_scoring", "ocr.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _load_ocr_npz(path: str) -> List[Dict[str, Any]]:
    """
    Minimal supported schema:
    - key `ocr_raw` -> object array holding list[dict]
    - or key `ocr_data` -> object array holding list[dict]
    Each dict should contain at least: `frame`, `bbox`, `text` (or `text_raw`), `confidence`.
    """
    data = np.load(path, allow_pickle=True)
    raw = data.get("ocr_raw")
    if raw is None:
        raw = data.get("ocr_data")
    if raw is None:
        return []
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw_item = raw.item() if raw.ndim == 0 else raw.tolist()
    else:
        raw_item = raw
    if isinstance(raw_item, list):
        out: List[Dict[str, Any]] = []
        for d in raw_item:
            if not isinstance(d, dict):
                continue
            dd = dict(d)
            # Normalize common field names across OCR producers.
            # ocr_extractor uses det_confidence + text_raw/text_norm.
            if "confidence" not in dd and "det_confidence" in dd:
                try:
                    dd["confidence"] = float(dd.get("det_confidence"))
                except Exception:
                    dd["confidence"] = 1.0
            if "text" not in dd and "text_raw" in dd:
                dd["text"] = dd.get("text_raw")
            if "bbox" in dd and isinstance(dd["bbox"], np.ndarray):
                dd["bbox"] = [float(x) for x in dd["bbox"].tolist()]
            out.append(dd)
        return out
    return []


def _face_presence_signal_from_core_face_landmarks(core_npz: Dict[str, Any]) -> np.ndarray:
    """
    Returns float32 signal in [0,1] with length = number of frames in core_face_landmarks sample.
    """
    face_present = core_npz.get("face_present")
    if face_present is None:
        return np.asarray([], dtype=np.float32)
    fp = np.asarray(face_present)
    if fp.ndim == 1:
        present_any = fp.astype(bool)
    else:
        present_any = np.any(fp.astype(bool), axis=1)
    return present_any.astype(np.float32)


class TextScoringModule(BaseModule):
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    @property
    def supports_batch(self) -> bool:
        """
        TextScoringModule is CPU-only and safe to use in VisualProcessor batch mode.

        Batch processing is implemented via a simple per-video loop that reuses
        the single-video `run()` implementation for each VideoContext, ensuring
        per-run ResultStore isolation and correct NPZ meta.
        """
        return True

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if self.rs_path is None:
            raise ValueError("text_scoring | rs_path is required")
        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        times_s = _times_s_from_union(frame_manager=frame_manager, frame_indices=fi)

        total = int(fi.size)
        retain_raw_ocr_text = bool(config.get("retain_raw_ocr_text", False))
        store_debug_objects = bool(config.get("store_debug_objects", False))
        enable_text_peaks = bool(config.get("enable_text_peaks", False))
        enable_language_entropy = bool(config.get("enable_language_entropy", False))
        enable_text_movement_speed = bool(config.get("enable_text_movement_speed", False))
        use_motion_data = bool(config.get("use_motion_data", False))

        ocr_npz = config.get("ocr_npz")
        if ocr_npz is None:
            ocr_npz = _find_ocr_npz(self.rs_path)

        # Optional face signal
        use_face_data = bool(config.get("use_face_data", False))
        face_signal: Optional[np.ndarray] = None
        if use_face_data:
            core = self.load_core_provider("core_face_landmarks")
            if core is None:
                raise FileNotFoundError("text_scoring | core_face_landmarks requested but not found")
            # Align by frame_indices (strict)
            core_idx = core.get("frame_indices")
            if core_idx is None:
                raise RuntimeError("text_scoring | core_face_landmarks missing frame_indices")
            core_idx = np.asarray(core_idx, dtype=np.int32)
            mapping = {int(x): i for i, x in enumerate(core_idx.tolist())}
            pos = [mapping.get(int(x), -1) for x in fi.tolist()]
            if any(p < 0 for p in pos):
                raise RuntimeError(
                    "text_scoring | core_face_landmarks does not cover requested frame_indices. "
                    "Segmenter must provide consistent sampling if you enable --use-face-data."
                )
            face_all = _face_presence_signal_from_core_face_landmarks(core)
            face_signal = face_all[np.asarray(pos, dtype=np.int64)]

        if not ocr_npz:
            # Valid empty output (OCR not available)
            meta_override = {"status": "empty", "empty_reason": "dependency_missing"}
            feats: Dict[str, Any] = {"text_present": False}
            feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
            feature_values = np.asarray([_as_float_feature(feats.get(k)) for k in _FEATURE_NAMES_V1], dtype=np.float32).reshape(-1)
            return {
                "frame_indices": fi,
                "times_s": times_s,
                "text_present": np.asarray(False),
                "text_presence": np.zeros((total,), dtype=np.bool_),
                "text_count_per_frame": np.zeros((total,), dtype=np.int32),
                "feature_names": feature_names,
                "feature_values": feature_values,
                "ocr_raw": np.asarray([], dtype=object) if store_debug_objects else np.asarray([], dtype=object),
                "ocr_unique_elements": np.asarray([], dtype=object) if store_debug_objects else np.asarray([], dtype=object),
                "ui_payload": {
                    "schema_version": "text_scoring_ui_v1",
                    "status": "empty",
                    "empty_reason": "dependency_missing",
                },
                "__meta_override__": meta_override,
            }

        ocr_data = _load_ocr_npz(str(ocr_npz))
        if not ocr_data:
            meta_override = {"status": "empty", "empty_reason": "no_text_available"}
            feats: Dict[str, Any] = {"text_present": False}
            feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
            feature_values = np.asarray([_as_float_feature(feats.get(k)) for k in _FEATURE_NAMES_V1], dtype=np.float32).reshape(-1)
            return {
                "frame_indices": fi,
                "times_s": times_s,
                "text_present": np.asarray(False),
                "text_presence": np.zeros((total,), dtype=np.bool_),
                "text_count_per_frame": np.zeros((total,), dtype=np.int32),
                "feature_names": feature_names,
                "feature_values": feature_values,
                "ocr_raw": np.asarray([], dtype=object) if store_debug_objects else np.asarray([], dtype=object),
                "ocr_unique_elements": np.asarray([], dtype=object) if store_debug_objects else np.asarray([], dtype=object),
                "ui_payload": {
                    "schema_version": "text_scoring_ui_v1",
                    "status": "empty",
                    "empty_reason": "no_text_available",
                    "ocr_npz": str(ocr_npz),
                },
                "__meta_override__": meta_override,
            }

        # Filter OCR detections to this module's frame_indices (union-domain)
        allowed = set(int(x) for x in fi.tolist())
        ocr_filtered = [d for d in ocr_data if int(d.get("frame", -1)) in allowed]

        # If after filtering there is nothing — still a valid empty result
        if not ocr_filtered:
            meta_override = {"status": "empty", "empty_reason": "no_text_available"}
            feats: Dict[str, Any] = {"text_present": False}
            feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
            feature_values = np.asarray([_as_float_feature(feats.get(k)) for k in _FEATURE_NAMES_V1], dtype=np.float32).reshape(-1)
            return {
                "frame_indices": fi,
                "times_s": times_s,
                "text_present": np.asarray(False),
                "text_presence": np.zeros((total,), dtype=np.bool_),
                "text_count_per_frame": np.zeros((total,), dtype=np.int32),
                "feature_names": feature_names,
                "feature_values": feature_values,
                "ocr_raw": np.asarray([], dtype=object) if store_debug_objects else np.asarray([], dtype=object),
                "ocr_unique_elements": np.asarray([], dtype=object) if store_debug_objects else np.asarray([], dtype=object),
                "ui_payload": {
                    "schema_version": "text_scoring_ui_v1",
                    "status": "empty",
                    "empty_reason": "no_text_available",
                    "ocr_npz": str(ocr_npz),
                    "notes": "ocr_outside_sampling",
                },
                "__meta_override__": meta_override,
            }

        # Ensure OCR rows use union time-axis (no fps fallback)
        idx_to_time = {int(fi[i]): float(times_s[i]) for i in range(int(fi.size))}
        for d in ocr_filtered:
            try:
                fr = int(d.get("frame"))
                if fr in idx_to_time:
                    d["time_s"] = float(idx_to_time[fr])
            except Exception:
                continue

        # Build signals aligned to module sampling group (face-only baseline; motion/audio = zeros).
        motion_weight = float(config.get("motion_weight", 0.0))
        motion_signal = np.zeros((total,), dtype=np.float32)
        motion_source = "none"
        if (use_motion_data or (motion_weight > 0.0)) and total > 0:
            mot, motion_source = _load_motion_signal_optional(module=self, fi=fi)
            if mot is not None and mot.shape[0] == total:
                motion_signal = np.asarray(mot, dtype=np.float32)
            else:
                self.logger.warning(
                    "%s | motion_weight>0 or use_motion_data=true, but motion signal is not available/aligned (source=%s). Using zeros.",
                    MODULE_NAME,
                    str(motion_source),
                )
        face_signal_aligned = np.zeros((total,), dtype=np.float32)
        if face_signal is not None and face_signal.shape[0] == total:
            face_signal_aligned = np.asarray(face_signal, dtype=np.float32)
        audio_signal = np.zeros((total,), dtype=np.float32)

        pipeline = TextVideoInteractionPipeline(
            video_fps=int(round(float(getattr(frame_manager, "fps", 30.0) or 30.0))),
            frame_width=int(getattr(frame_manager, "width", 0) or 0) or None,
            frame_height=int(getattr(frame_manager, "height", 0) or 0) or None,
            alignment_window_seconds=float(config.get("alignment_window_seconds", 0.5)),
            motion_weight=float(motion_weight),
            face_weight=float(config.get("face_weight", 1.0 if use_face_data else 0.0)),
            audio_weight=float(config.get("audio_weight", 0.0)),
            min_ocr_confidence=float(config.get("min_ocr_confidence", 0.4)),
        )

        feats = pipeline.extract_features(
            ocr_data=ocr_filtered,
            frame_indices=fi,
            times_s=times_s,
            motion_signal=motion_signal,
            face_signal=face_signal_aligned,
            audio_signal=audio_signal,
            enable_text_peaks=enable_text_peaks,
            enable_language_entropy=enable_language_entropy,
            enable_text_movement_speed=enable_text_movement_speed,
        )

        # Pipeline already returns `ocr_raw` and `ocr_unique_elements`.
        ocr_raw = feats.pop("ocr_raw", [])
        ocr_unique = feats.pop("ocr_unique_elements", [])
        feats["text_present"] = True

        # Build per-frame text presence/density aligned to fi/times_s
        by_frame_counts: Dict[int, int] = {}
        for d in ocr_filtered:
            try:
                fr = int(d.get("frame"))
                by_frame_counts[fr] = by_frame_counts.get(fr, 0) + 1
            except Exception:
                continue
        text_count = np.asarray([int(by_frame_counts.get(int(fr), 0)) for fr in fi.tolist()], dtype=np.int32)
        text_presence = (text_count > 0).astype(np.bool_)

        # Derived privacy-safe scalar features
        scalar_feats: Dict[str, Any] = dict(feats)
        scalar_feats["text_frames_ratio"] = float(np.mean(text_presence.astype(np.float32))) if text_presence.size else 0.0
        scalar_feats["text_count_mean"] = float(np.mean(text_count.astype(np.float32))) if text_count.size else 0.0
        scalar_feats["text_count_p95"] = float(np.percentile(text_count.astype(np.float32), 95)) if text_count.size else 0.0
        scalar_feats["ocr_raw_count"] = int(len(ocr_raw)) if isinstance(ocr_raw, list) else 0
        scalar_feats["ocr_unique_elements_count"] = int(len(ocr_unique)) if isinstance(ocr_unique, list) else 0

        # compat: expose cta_timestamp as alias of cta_mean_timestamp if present
        if scalar_feats.get("cta_timestamp") is None and scalar_feats.get("cta_mean_timestamp") is not None:
            scalar_feats["cta_timestamp"] = scalar_feats.get("cta_mean_timestamp")

        # Peaks lists -> counts (stable scalar)
        if "text_emphasis_peak_flags" in scalar_feats and isinstance(scalar_feats.get("text_emphasis_peak_flags"), list):
            scalar_feats["text_emphasis_peaks_count"] = int(len(scalar_feats.get("text_emphasis_peak_flags") or []))
        else:
            scalar_feats["text_emphasis_peaks_count"] = float("nan")

        feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
        feature_values = np.asarray([_as_float_feature(scalar_feats.get(k)) for k in _FEATURE_NAMES_V1], dtype=np.float32).reshape(-1)

        # Privacy: raw OCR text stored only if retain_raw_ocr_text=true.
        def _redact_row(r: Dict[str, Any]) -> Dict[str, Any]:
            rr = dict(r)
            txt = str(rr.get("text_norm") or rr.get("text_raw") or rr.get("text") or "")
            rr.pop("text_raw", None)
            rr.pop("text_norm", None)
            rr.pop("text", None)
            rr["text_len"] = int(len(txt))
            rr["text_hash_sha256"] = _sha256_text(txt)
            return rr

        if not retain_raw_ocr_text:
            ocr_raw = [_redact_row(r) for r in ocr_raw if isinstance(r, dict)]
            ocr_unique = [_redact_row(r) for r in ocr_unique if isinstance(r, dict)]

        # ui_payload (privacy-safe)
        ui_payload: Dict[str, Any] = {
            "schema_version": "text_scoring_ui_v1",
            "curves": {
                "text_presence": {"npz_key": "text_presence", "label": "Text present"},
                "text_count_per_frame": {"npz_key": "text_count_per_frame", "label": "OCR count"},
            },
            "markers": {
                "cta_first_timestamp": feats.get("cta_first_timestamp"),
                "cta_mean_timestamp": feats.get("cta_mean_timestamp"),
                "cta_last_timestamp": feats.get("cta_last_timestamp"),
            },
            "flags": {
                "retain_raw_ocr_text": bool(retain_raw_ocr_text),
                "use_face_data": bool(use_face_data),
                "use_motion_data": bool(use_motion_data or (motion_weight > 0.0)),
                "motion_source": str(motion_source),
            },
        }

        # store debug OCR payload only if requested
        if not store_debug_objects:
            ocr_raw = []
            ocr_unique = []

        return {
            "frame_indices": fi,
            "times_s": times_s,
            "text_present": np.asarray(True),
            "text_presence": text_presence,
            "text_count_per_frame": text_count,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "ocr_raw": np.asarray(ocr_raw, dtype=object),
            "ocr_unique_elements": np.asarray(ocr_unique, dtype=object),
            "ui_payload": ui_payload,
        }

    def run(self, frames_dir: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Override BaseModule.run to:
        - write progress events (state_events.jsonl)
        - attach ui_payload into NPZ meta (meta.ui_payload)
        - add stage timings
        """
        import time

        if self.rs_path is None:
            raise RuntimeError(f"{MODULE_NAME} | rs_path is required")
        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise RuntimeError(f"{MODULE_NAME} | frame_indices missing/empty (no-fallback)")

        platform_id = str(metadata.get("platform_id") or "")
        video_id = str(metadata.get("video_id") or "")
        run_id = str(metadata.get("run_id") or "")
        total = int(len(frame_indices))

        _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="start")
        t0 = time.perf_counter()
        resource_profile_before = _resource_profile_snapshot()
        fm = None
        try:
            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="load_deps")
            fm = self.create_frame_manager(frames_dir, metadata)
            t_fm = time.perf_counter()

            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="process")
            results = self.process(frame_manager=fm, frame_indices=frame_indices, config=config or {})
            t_proc = time.perf_counter()

            ui_payload = None
            if isinstance(results, dict) and "ui_payload" in results:
                try:
                    ui_payload = results.pop("ui_payload")
                except Exception:
                    ui_payload = None

            meta_override = None
            if isinstance(results, dict) and "__meta_override__" in results:
                try:
                    meta_override = results.pop("__meta_override__")
                except Exception:
                    meta_override = None

            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": len(frame_indices),
                "frames_dir": frames_dir,
                "platform_id": metadata.get("platform_id"),
                "video_id": metadata.get("video_id"),
                "run_id": metadata.get("run_id"),
                "sampling_policy_version": metadata.get("sampling_policy_version"),
                "config_hash": metadata.get("config_hash"),
                "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                "analysis_fps": metadata.get("analysis_fps"),
                "analysis_width": metadata.get("analysis_width"),
                "analysis_height": metadata.get("analysis_height"),
                "ui_payload": ui_payload,
                "models_used": self.get_models_used(config=config or {}, metadata=metadata or {}),
                # config highlights
                "ocr_npz": (config or {}).get("ocr_npz"),
                "use_face_data": bool((config or {}).get("use_face_data", False)),
                "alignment_window_seconds": float((config or {}).get("alignment_window_seconds", 0.5)),
                "use_motion_data": bool((config or {}).get("use_motion_data", False)),
                "motion_weight": float((config or {}).get("motion_weight", 0.0)),
                "face_weight": float((config or {}).get("face_weight", 1.0)),
                "audio_weight": float((config or {}).get("audio_weight", 0.0)),
                "min_ocr_confidence": float((config or {}).get("min_ocr_confidence", 0.4)),
                "retain_raw_ocr_text": bool((config or {}).get("retain_raw_ocr_text", False)),
                "store_debug_objects": bool((config or {}).get("store_debug_objects", False)),
                "enable_text_peaks": bool((config or {}).get("enable_text_peaks", False)),
                "enable_language_entropy": bool((config or {}).get("enable_language_entropy", False)),
                "enable_text_movement_speed": bool((config or {}).get("enable_text_movement_speed", False)),
            }
            if isinstance(resource_profile_before, dict) and resource_profile_before:
                save_metadata["resource_profile_before"] = dict(resource_profile_before)
            if isinstance(meta_override, dict):
                for k, v in meta_override.items():
                    if isinstance(k, str) and k and (isinstance(v, (str, int, float, bool)) or v is None):
                        save_metadata[k] = v

            # Audit v3: stage timings belong to meta.stage_timings_ms
            save_metadata["stage_timings_ms"] = {
                "frame_manager_ms": float((t_fm - t0) * 1000.0),
                "process_ms": float((t_proc - t_fm) * 1000.0),
                "total_ms": float((t_proc - t0) * 1000.0),
            }

            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=total, total=total, stage="save")
            t_save0 = time.perf_counter()
            out_path = self.save_results(results=results, metadata=save_metadata)
            t_save1 = time.perf_counter()
            try:
                st2 = save_metadata.get("stage_timings_ms") if isinstance(save_metadata.get("stage_timings_ms"), dict) else {}
                st2["save_ms"] = float((t_save1 - t_save0) * 1000.0)
                st2["total_ms"] = float((t_save1 - t0) * 1000.0)
                save_metadata["stage_timings_ms"] = st2
            except Exception:
                pass
            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=total, total=total, stage="done")
            return out_path
        finally:
            try:
                if fm is not None:
                    fm.close()
            except Exception:
                pass

    def process_batch(
        self,
        video_contexts: List["VideoContext"],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Batch-safe wrapper around `run()` for TextScoringModule.

        Strategy (BATCH_PROCESSING_PLAN Stage 0/1 for CPU modules):
        - Iterate over VideoContext instances sequentially (CPU workload).
        - For each video, create a fresh TextScoringModule with its own rs_path.
        - Call `run()` with metadata loaded from VideoContext to keep
          run identity and NPZ meta consistent with single-video mode.
        """
        from utils.video_context import VideoContext  # local import to avoid cycles

        results: List[Dict[str, Any]] = []

        for video_ctx in video_contexts:
            try:
                if not isinstance(video_ctx, VideoContext):
                    raise TypeError(f"video_context must be VideoContext, got {type(video_ctx)}")

                module = TextScoringModule(rs_path=video_ctx.rs_path)
                metadata = video_ctx.load_metadata()

                saved_path = module.run(
                    frames_dir=video_ctx.frames_dir,
                    config=config or {},
                    metadata=metadata,
                )

                results.append(
                    {
                        "video_id": video_ctx.video_id,
                        "status": "ok",
                        "saved_path": saved_path,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "video_id": getattr(video_ctx, "video_id", "unknown"),
                        "status": "error",
                        "error": str(e),
                    }
                )

        return results
