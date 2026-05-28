"""
Модуль для анализа цвета и освещения видео.

Все TODO выполнены:
✓ Оптимизирован код под работу с внешними зависимостями (загрузка scenes из scene_classification)
✓ Модуль оптимизирован под работу с BaseModule
✓ Выход приведен к единому формату для сохранения в npz
"""

import os, sys

_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

import math
import numpy as np
import cv2
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.stats import entropy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, label
from sklearn.cluster import KMeans
from collections import Counter

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.meta_builder import apply_models_meta  # type: ignore

NAME = "ColorLightProcessor"

logger = get_logger(NAME)

class ColorLightProcessor(BaseModule):
    """Процессор для анализа цвета и освещения видео"""

    VERSION = "2.0.2"
    SCHEMA_VERSION = "color_light_npz_v2"
    ARTIFACT_FILENAME = "color_light_features.npz"

    # Fixed model-facing compact frame vector contract (stable dims for models).
    FRAME_COMPACT_KEYS: List[str] = [
        "hue_mean_norm",
        "hue_std_norm",
        "hue_entropy_weighted",
        "sat_mean_norm",
        "val_mean_norm",
        "L_mean_norm",
        "global_contrast_norm",
        "local_contrast_mean_norm",
        "colorfulness_norm",
        "skin_tone_ratio",
        "overexposed_ratio",
        "underexposed_ratio",
        "vignetting_score_norm",
        "soft_light_prob",
        "dominant_lab_a_norm",
        "dominant_lab_b_norm",
    ]
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        max_frames_per_scene: int = 350,
        stride: int = 5,
        store_debug_objects: bool = True,
        **kwargs: Any
    ):
        """
        Args:
            rs_path: Путь к хранилищу результатов
            max_frames_per_scene: Максимальное количество кадров для обработки на сцену
            stride: Шаг для выборки кадров
            store_debug_objects: Сохранять ли тяжёлые debug/analytics объекты (`frames`/`scenes`) в NPZ
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, logger_name="color_light", **kwargs)
        # Deprecated: sampling is controlled by Segmenter (frame_indices from metadata).
        # Keep for backward compatibility, but do not use for sampling.
        self.max_frames_per_scene = max_frames_per_scene
        self.stride = stride
        self.store_debug_objects = bool(store_debug_objects)
        self._last_metadata: Optional[Dict[str, Any]] = None
    
    @property
    def supports_batch(self) -> bool:
        """Поддержка batch processing для color_light (CPU модуль)."""
        return True
    
    def required_dependencies(self) -> List[str]:
        """Возвращает список зависимостей модуля."""
        return ["scene_classification"]
    
    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Возвращает список используемых моделей.
        
        Для color_light: модели не используются (только CPU вычисления на основе OpenCV/scikit-learn).
        Эстетические модели (NIMA/LAION) планируются к интеграции, но пока не подключены.
        """
        return []
    
    def _append_state_event_if_possible(self, *, rs_path: str, event: Dict[str, Any]) -> None:
        """Best-effort writer for state_events.jsonl (backend tails this file)."""
        try:
            from pathlib import Path as _Path
            run_rs = _Path(rs_path).resolve()
            rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
            runs_root = rs_base.parent
            platform_id = str(event.get("platform_id") or "")
            video_id = str(event.get("video_id") or "")
            run_id = str(event.get("run_id") or "")
            if not (platform_id and video_id and run_id):
                # Try to extract from metadata
                if self._last_metadata:
                    platform_id = str(self._last_metadata.get("platform_id") or "")
                    video_id = str(self._last_metadata.get("video_id") or "")
                    run_id = str(self._last_metadata.get("run_id") or "")
            if not (platform_id and video_id and run_id):
                return
            p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            event["platform_id"] = platform_id
            event["video_id"] = video_id
            event["run_id"] = run_id
            line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            with open(p, "ab") as f:
                f.write(line)
        except Exception:
            return
    
    def _build_ui_payload(self, results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build UI payload from results (baseline contract: meta.ui_payload)."""
        try:
            from .presentation import build_presentation
            return build_presentation(results, metadata)
        except Exception as e:
            logger.warning(f"{self.module_name} | Failed to build UI payload: {e}")
            return {
                "component": self.module_name,
                "schema_version": "color_light_ui_v1",
                "error": str(e),
            }

    def _ensure_rgb(self, frame: np.ndarray, color_space: Optional[str]) -> np.ndarray:
        """
        Приводит кадр к RGB, если metadata говорит что он BGR.
        Если color_space неизвестен, оставляем как есть (assume RGB).
        """
        if frame is None:
            raise ValueError("Frame is None")
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        if color_space and str(color_space).upper() == "BGR":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def _compute_rgb_stats(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Базовые RGB‑статистики: только mean/std по каналам.
        Мин/макс, skew и kurtosis убраны как слабоинформативные и сильно коррелирующие.
        """
        features: Dict[str, float] = {}
        for i, channel in enumerate(["R", "G", "B"]):
            channel_data = frame[:, :, i].flatten().astype(np.float32)
            features[f"color_mean_{channel.lower()}"] = float(np.mean(channel_data))
            features[f"color_std_{channel.lower()}"] = float(np.std(channel_data))
        return features
    
    def _compute_hsv_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        HSV‑фичи:
        - hue_mean / hue_std / hue_entropy
        - saturation_mean/std, value_mean/std
        - нормализованные варианты и взвешенная по saturation энтропия hue.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        features: Dict[str, float] = {}

        hue = hsv[:, :, 0].flatten().astype(np.float32)  # [0, 180]
        sat = hsv[:, :, 1].flatten().astype(np.float32)  # [0, 255]
        val = hsv[:, :, 2].flatten().astype(np.float32)  # [0, 255]

        # Базовые статистики
        hue_mean = float(np.mean(hue))
        hue_std = float(np.std(hue))
        features["hue_mean"] = hue_mean
        features["hue_std"] = hue_std

        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
        features["hue_entropy"] = float(entropy(hue_probs + 1e-10))

        # Взвешенная по насыщенности энтропия hue
        sat_norm = sat / 255.0
        hue_hist_w, _ = np.histogram(hue, bins=36, range=(0, 180), weights=sat_norm)
        hue_probs_w = hue_hist_w / (hue_hist_w.sum() + 1e-10)
        features["hue_entropy_weighted"] = float(entropy(hue_probs_w + 1e-10))

        # Saturation
        sat_mean = float(np.mean(sat))
        sat_std = float(np.std(sat))
        features["saturation_mean"] = sat_mean
        features["saturation_std"] = sat_std

        # Value (brightness)
        val_mean = float(np.mean(val))
        val_std = float(np.std(val))
        features["value_mean"] = val_mean
        features["value_std"] = val_std

        # Нормализованные фичи (0–1) для трансформера
        features["hue_mean_norm"] = hue_mean / 180.0
        features["hue_std_norm"] = hue_std / 180.0
        features["sat_mean_norm"] = sat_mean / 255.0
        features["val_mean_norm"] = val_mean / 255.0

        return features
    
    def _compute_lab_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет LAB‑фичи: L_mean, L_contrast, ab_balance + нормализованный L."""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        features: Dict[str, float] = {}

        L = lab[:, :, 0].flatten().astype(np.float32)  # [0, 255]
        L_mean = float(np.mean(L))
        L_std = float(np.std(L))
        features["L_mean"] = L_mean
        features["L_contrast"] = L_std

        a = lab[:, :, 1].flatten().astype(np.float32) - 128.0
        b = lab[:, :, 2].flatten().astype(np.float32) - 128.0
        features["ab_balance"] = float(np.mean(a) - np.mean(b))

        # Нормализованная яркость
        features["L_mean_norm"] = L_mean / 255.0

        return features
    
    def _compute_palette_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Палитра и цветовые признаки.
        - KMeans в Lab‑пространстве (a,b) для устойчивых доминантных цветов.
        - Индекс цветности, warm/cold ratio, skin_tone_ratio, color_palette_entropy.
        - Доминантный цвет возвращается как нормализованные Lab‑координаты a,b.
        """
        features: Dict[str, float] = {}

        h, w = frame.shape[:2]
        sample_size = min(10000, h * w)
        if h * w > sample_size and h > 0 and w > 0:
            step = int(max(1, np.sqrt(h * w / sample_size)))
            sampled_rgb = frame[::step, ::step].reshape(-1, 3)
        else:
            sampled_rgb = frame.reshape(-1, 3)

        # --- KMeans в Lab для доминантного цвета ---
        if len(sampled_rgb) > 0:
            lab = cv2.cvtColor(sampled_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab_flat = lab.reshape(-1, 3).astype(np.float32)
            ab = lab_flat[:, 1:3]  # только a,b

            n_colors = int(min(3, len(ab)))
            if n_colors >= 1:
                if n_colors == 1:
                    centers = ab[:1]
                    labels = np.zeros(len(ab), dtype=int)
                else:
                    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                    kmeans.fit(ab)
                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_

                counts = Counter(labels)
                # выбираем самый частый кластер
                dominant_label = max(counts.items(), key=lambda x: x[1])[0]
                dom_a, dom_b = centers[int(dominant_label)]

                # сохраняем ненормированные и нормализованные координаты
                features["dominant_lab_a"] = float(dom_a)
                features["dominant_lab_b"] = float(dom_b)
                features["dominant_lab_a_norm"] = float((dom_a + 128.0) / 255.0)
                features["dominant_lab_b_norm"] = float((dom_b + 128.0) / 255.0)

        # Colorfulness index (как раньше, по RGB)
        rgb_reshaped = frame.reshape(-1, 3).astype(np.float32)
        if rgb_reshaped.size > 0:
            rg = rgb_reshaped[:, 0] - rgb_reshaped[:, 1]
            yb = 0.5 * (rgb_reshaped[:, 0] + rgb_reshaped[:, 1]) - rgb_reshaped[:, 2]
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rgyb = np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))
            colorfulness = float(std_rg + std_yb + 0.3 * mean_rgyb)
        else:
            colorfulness = 0.0
        features["colorfulness_index"] = colorfulness
        # Нормализованный индекс цветности ~ [0,1] (ожидаем 0–100+)
        features["colorfulness_norm"] = float(min(colorfulness / 100.0, 1.0))

        # Warm vs cold ratio, skin_tone_ratio, color_palette_entropy
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].flatten().astype(np.float32)
        sat = hsv[:, :, 1].flatten().astype(np.float32)
        val = hsv[:, :, 2].flatten().astype(np.float32)

        warm_mask = ((hue >= 0) & (hue <= 30)) | ((hue >= 150) & (hue <= 180))
        warm_count = np.sum(warm_mask)
        cold_count = max(1, len(hue) - warm_count)
        features["warm_vs_cold_ratio"] = float(warm_count / float(cold_count))

        skin_mask = ((hue >= 0) & (hue <= 25)) & (sat >= 20) & (val >= 50)
        features["skin_tone_ratio"] = float(np.sum(skin_mask) / (len(hue) + 1e-10))

        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
        features["color_palette_entropy"] = float(entropy(hue_probs + 1e-10))

        # Цветовые гармонии: оставляем только complementary / analogous как компактные признаки
        harmony_features = self._compute_color_harmonies(hue, sat, val)
        features.update(harmony_features)

        return features
    
    def _compute_color_harmonies(
        self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray
    ) -> Dict[str, float]:
        """
        Компактные цветовые гармонии:
        - color_harmony_complementary_prob
        - color_harmony_analogous_prob
        Триады и split‑complementary убраны для снижения размерности.
        """
        features: Dict[str, float] = {}

        hue_hist, hue_bins = np.histogram(hue, bins=36, range=(0, 180))
        if hue_hist.sum() == 0:
            features["color_harmony_complementary_prob"] = 0.0
            features["color_harmony_analogous_prob"] = 0.0
            return features

        dominant_hue_bin = int(np.argmax(hue_hist))
        dominant_hue = (hue_bins[dominant_hue_bin] + hue_bins[dominant_hue_bin + 1]) / 2.0

        dominant_hue_360 = dominant_hue * 2.0

        # Complementary
        comp_hue = (dominant_hue_360 + 180.0) % 360.0
        comp_hue_180 = comp_hue / 2.0
        comp_range = (
            (hue >= (comp_hue_180 - 15) % 180)
            & (hue <= (comp_hue_180 + 15) % 180)
        ) | ((hue >= 0) & (hue <= (comp_hue_180 + 15 - 180) % 180)) | (
            (hue >= (comp_hue_180 - 15 + 180) % 180) & (hue <= 180)
        )
        comp_ratio = float(np.sum(comp_range) / (len(hue) + 1e-10))
        features["color_harmony_complementary_prob"] = float(min(comp_ratio * 2.0, 1.0))

        # Analogous (±30 градусов вокруг доминирующего)
        anal_range1 = (hue >= (dominant_hue - 30) % 180) & (
            hue <= (dominant_hue + 30) % 180
        )
        anal_range2 = ((hue >= 0) & (hue <= (dominant_hue + 30 - 180) % 180)) | (
            (hue >= (dominant_hue - 30 + 180) % 180) & (hue <= 180)
        )
        anal_range = anal_range1 | anal_range2
        anal_ratio = float(np.sum(anal_range) / (len(hue) + 1e-10))
        features["color_harmony_analogous_prob"] = float(
            min(anal_ratio * 1.2, 1.0)
        )

        return features
    
    def _safe_entropy_from_hist(self, hist: np.ndarray, eps: float = 1e-12) -> float:
        probs = hist.astype(np.float64)
        s = probs.sum()
        if s <= 0:
            return 0.0
        probs = probs / (s + eps)
        probs = probs + eps
        return float(-np.sum(probs * np.log(probs)))

    def _compute_lighting_uniformity(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет фичи равномерности освещения: uniformity_index, center/corner brightness, vignetting.
        Вход: gray — 2D np.ndarray (H, W), dtype=uint8 or numeric.
        Возвращает dict со значениями float.
        """
        features: Dict[str, float] = {}

        if gray.ndim != 2:
            raise ValueError("_compute_lighting_uniformity expects 2D gray image")

        # cast to float for stable stats
        grayf = np.asarray(gray, dtype=np.float32)
        h, w = grayf.shape
        if h == 0 or w == 0:
            # degenerate
            return {
                "lighting_uniformity_index": 0.0,
                "center_brightness": 0.0,
                "corner_brightness": 0.0,
                "vignetting_score": 0.0,
            }

        # grid 3x3
        grid_h, grid_w = 3, 3
        cell_h = max(1, h // grid_h)
        cell_w = max(1, w // grid_w)

        brightness_grid = []
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * cell_h
                x_start = j * cell_w
                # include remainder in last cell
                if i == grid_h - 1:
                    y_end = h
                else:
                    y_end = y_start + cell_h
                if j == grid_w - 1:
                    x_end = w
                else:
                    x_end = x_start + cell_w

                cell = grayf[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    brightness_grid.append(0.0)
                else:
                    brightness_grid.append(float(np.mean(cell)))

        brightness_grid = np.array(brightness_grid, dtype=np.float32)
        uniformity_std = float(np.std(brightness_grid))
        uniformity_mean = float(np.mean(brightness_grid))

        # normalized std: divide by (mean + eps) to be scale invariant
        eps = 1e-6
        norm_std = uniformity_std / (uniformity_mean + eps)

        # uniformity index in (0,1], higher = more uniform
        features["lighting_uniformity_index"] = float(1.0 / (1.0 + norm_std))

        # center brightness (central cell)
        center_idx = (grid_h * grid_w) // 2
        features["center_brightness"] = float(brightness_grid[center_idx])

        # corner brightness (average of 4 corners)
        corner_indices = [0, grid_w - 1, (grid_h - 1) * grid_w, grid_h * grid_w - 1]
        corner_vals = [brightness_grid[idx] for idx in corner_indices if idx < len(brightness_grid)]
        features["corner_brightness"] = float(np.mean(corner_vals)) if corner_vals else 0.0

        # vignetting score: 0 no vignetting, 1 strong (clamped)
        # compute ratio corner/center (safe), invert to have 0..1
        if features["center_brightness"] > 0:
            ratio = features["corner_brightness"] / (features["center_brightness"] + eps)
            # if corners brighter than center ratio>1 -> vignetting negative (we clamp to 0)
            v = 1.0 - min(max(ratio, 0.0), 1.0)
            features["vignetting_score"] = float(max(0.0, min(v, 1.0)))
        else:
            features["vignetting_score"] = 0.0

        return features


    def _compute_lighting_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет фичи освещения: brightness, contrast, entropy, clipping ratios, dynamic range (dB),
        highlight/shadow clipping, local contrast, uniformity (via helper).
        Вход: frame — HxWx3 RGB (uint8) или совместимый numeric array.
        """
        features: Dict[str, float] = {}

        if frame is None:
            # return zeros for robustness
            zero_keys = [
                "brightness_mean", "brightness_std", "brightness_entropy",
                "overexposed_pixels", "underexposed_pixels", "global_contrast",
                "local_contrast", "local_contrast_std", "contrast_entropy",
                "dynamic_range_db", "highlight_clipping_ratio", "shadow_clipping_ratio"
            ]
            return {k: 0.0 for k in zero_keys}

        # ensure numpy array and RGB uint8
        im = np.asarray(frame)
        if im.ndim == 3 and im.shape[-1] == 4:
            im = im[..., :3]
        if im.ndim != 3 or im.shape[-1] != 3:
            # if grayscale 2D, expand
            if im.ndim == 2:
                im = np.stack([im, im, im], axis=-1)
            else:
                raise ValueError(f"_compute_lighting_features: unexpected frame shape {im.shape}")

        # ensure ordering is RGB for cv2.cvtColor conversion to gray; if your frames are BGR, change accordingly
        # Here we assume incoming frames are RGB (consistent with your pipeline earlier).
        rgb = im.astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # 2D uint8

        # Basic brightness stats
        gray_f = gray.astype(np.float32)
        total_pixels = gray_f.size
        eps = 1e-10

        brightness_mean = float(np.mean(gray_f))
        brightness_std = float(np.std(gray_f))
        global_contrast = brightness_std  # RMS contrast = std
        features["brightness_mean"] = brightness_mean
        features["brightness_std"] = brightness_std
        features["global_contrast"] = global_contrast

        # Brightness entropy (histogram over 256 bins)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        features["brightness_entropy"] = self._safe_entropy_from_hist(hist)

        # Over/under exposed ratios (fractions)
        overexposed = int(np.sum(gray >= 250))
        underexposed = int(np.sum(gray <= 5))
        over_ratio = float(overexposed / (total_pixels + eps))
        under_ratio = float(underexposed / (total_pixels + eps))
        features["overexposed_pixels"] = over_ratio
        features["underexposed_pixels"] = under_ratio

        # Highlight & shadow clipping (same as above but slightly different thresholds)
        highlight_threshold = 250
        shadow_threshold = 5
        highlight_clipped = int(np.sum(gray >= highlight_threshold))
        shadow_clipped = int(np.sum(gray <= shadow_threshold))
        features["highlight_clipping_ratio"] = float(highlight_clipped / (total_pixels + eps))
        features["shadow_clipping_ratio"] = float(shadow_clipped / (total_pixels + eps))

        # Contrast entropy (coarser histogram)
        contrast_hist, _ = np.histogram(gray, bins=64, range=(0, 256))
        features["contrast_entropy"] = self._safe_entropy_from_hist(contrast_hist)

        # Dynamic range: use max/min luminance and convert to decibels (20*log10)
        # Protect against zero; add tiny eps
        max_lum = float(np.max(gray_f))
        min_lum = float(np.min(gray_f))
        if min_lum <= 0:
            min_lum = 1e-3
        # ratio in linear domain
        dr_ratio = max_lum / (min_lum + eps)
        # convert to decibels (20*log10 for amplitude-like measure)
        dynamic_range_db = 20.0 * math.log10(dr_ratio + eps)
        features["dynamic_range_db"] = float(max(0.0, dynamic_range_db))

        # Local contrast: sliding non-overlapping windows (window_size adaptive)
        h, w = gray.shape
        # choose window size proportional to smaller dimension but not too large
        window_size = max(4, min(32, min(h, w) // 8))  # sensible defaults
        local_stds = []
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = gray_f[i : min(i + window_size, h), j : min(j + window_size, w)]
                if window.size:
                    local_stds.append(float(np.std(window)))
        if local_stds:
            local_contrast = float(np.mean(local_stds))
            local_contrast_std = float(np.std(local_stds))
            features["local_contrast"] = local_contrast
            features["local_contrast_std"] = local_contrast_std
        else:
            local_contrast = global_contrast
            features["local_contrast"] = float(local_contrast)
            features["local_contrast_std"] = 0.0

        # Lighting uniformity (grid-based) and vignetting via helper
        uniformity_features = self._compute_lighting_uniformity(gray)
        features.update(uniformity_features)

        # Нормализованные lighting‑фичи (0–1) для компактного вектора
        features["global_contrast_norm"] = float(min(global_contrast / 255.0, 1.0))
        features["local_contrast_mean_norm"] = float(min(local_contrast / 255.0, 1.0))
        features["overexposed_ratio"] = over_ratio
        features["underexposed_ratio"] = under_ratio
        features["vignetting_score_norm"] = float(
            uniformity_features.get("vignetting_score", 0.0)
        )

        # Final safety: cast to native floats and ensure keys exist
        for k, v in list(features.items()):
            try:
                features[k] = float(v)
            except Exception:
                features[k] = 0.0

        return features

    def _compute_light_direction(self, frame: np.ndarray) -> Dict[str, float]:
        """Оценивает направление света, количество источников, мягкость/жёсткость."""
        features = {}

        # --- Validate & convert ---
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"_compute_light_direction: unexpected frame shape {frame.shape}")

        rgb = frame.astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        h, w = gray.shape
        eps = 1e-8

        # === 1. GRADIENT-BASED LIGHT DIRECTION (ROBUST CIRCULAR MEAN) ===
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitudes = np.sqrt(grad_x**2 + grad_y**2)
        angles = np.arctan2(grad_y, grad_x)  # radians

        # если нет текстуры — направление неопределено (угол не возвращаем, только источники света)
        if np.sum(magnitudes) < eps:
            pass
        else:
            # оставляем расчёт для возможного дебага, но не сохраняем в фичи,
            # чтобы не засорять финальный вектор сильно шумным признаком
            sin_sum = float(np.sum(np.sin(angles) * (magnitudes + eps)))
            cos_sum = float(np.sum(np.cos(angles) * (magnitudes + eps)))
            _ = np.arctan2(sin_sum, cos_sum)

        # === 2. LIGHT SOURCE COUNT (ROBUST 2D PEAK DETECTION) ===
        # Сглаживаем картинку и ищем bright blobs
        blur = gaussian_filter(gray, sigma=7)
        threshold = np.percentile(blur, 96)

        mask = blur > threshold
        labeled, num_labels = label(mask)  # connected components

        # Ограничиваем количество источников света
        source_count = min(int(num_labels), 5)
        features["light_source_count_estimate"] = float(source_count)

        # === 3. SOFT vs HARD LIGHT (LAPLACIAN VARIANCE + NORMALIZATION) ===
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # нормировка в диапазон [0..1]
        # low variance → soft light; high → hard
        # границы подогнаны под реальные кадры
        soft_min = 50      # ультрамягкий
        hard_max = 600     # ультражёсткий

        if lap_var <= soft_min:
            soft_prob = 1.0
        elif lap_var >= hard_max:
            soft_prob = 0.0
        else:
            # линейное уменьшение мягкости
            soft_prob = 1.0 - (lap_var - soft_min) / (hard_max - soft_min)

        hard_prob = 1.0 - soft_prob

        features["soft_light_probability"] = float(soft_prob)
        features["hard_light_probability"] = float(hard_prob)
        # Дубликат с более коротким именем для компактного вектора
        features["soft_light_prob"] = float(soft_prob)

        return features

        
    def extract_frame_features(
        self,
        frame: np.ndarray,
        frame_idx: int,
        color_space: Optional[str] = None
    ) -> Dict[str, Any]:
        """Извлекает все frame-level фичи для одного кадра"""
        try:
            frame = self._ensure_rgb(frame, color_space)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | extract_frame_features | invalid frame: {e}"
            ) from e

        features = {}

        # RGB статистики (только mean/std)
        try:
            features.update(self._compute_rgb_stats(frame))
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | RGB stats failed: {e}"
            ) from e

        # HSV фичи + нормализованные hue/sat/value
        try:
            features.update(self._compute_hsv_features(frame))
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | HSV features failed: {e}"
            ) from e

        # LAB фичи (L_mean/L_contrast/ab_balance + L_mean_norm)
        try:
            features.update(self._compute_lab_features(frame))
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | LAB features failed: {e}"
            ) from e

        # Палитра, colorfulness, skin_tone_ratio, гармонии
        try:
            features.update(self._compute_palette_features(frame))
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | palette features failed: {e}"
            ) from e

        # Освещение и lighting‑фичи
        try:
            features.update(self._compute_lighting_features(frame))
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | lighting features failed: {e}"
            ) from e

        # Источники света и soft/hard light
        try:
            features.update(self._compute_light_direction(frame))
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | light direction failed: {e}"
            ) from e

        return {
            "frame_idx": frame_idx,
            "features": features
        }
    
    def extract_scene_features(
        self,
        frame_features: List[Dict],
        scene_start: int,
        scene_end: int,
        total_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """Извлекает scene-level фичи из списка frame features"""
        if not frame_features:
            return {}
        
        scene_features = {}
        
        # Базовые метрики сцены
        num_frames = len(frame_features)
        scene_features['num_frames'] = num_frames
        if total_frames and total_frames > 0:
            scene_features['num_frames_norm'] = float(
                min(num_frames / float(total_frames), 1.0)
            )
        
        # Извлекаем значения фич из всех кадров
        feature_arrays = {}
        for frame_feat in frame_features:
            for key, value in frame_feat['features'].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in feature_arrays:
                        feature_arrays[key] = []
                    feature_arrays[key].append(value)
        
        # Усреднение RGB/HSV/LAB фич по сцене
        for key, values in feature_arrays.items():
            if len(values) > 0:
                scene_features[f'{key}_mean'] = float(np.mean(values))
                scene_features[f'{key}_std'] = float(np.std(values))
        
        # Motion + Lighting features
        # Для скорости изменения освещенности используем Value из HSV
        brightness_values = [f['features'].get('value_mean', 0) for f in frame_features]
        if len(brightness_values) > 1:
            brightness_diff = np.diff(brightness_values)
            scene_features['brightness_change_speed'] = float(np.mean(np.abs(brightness_diff)))
            scene_features['scene_flicker_intensity'] = float(np.std(brightness_diff))
        
        # Color change speed (по hue)
        hue_values = [f['features'].get('hue_mean', 0) for f in frame_features]
        if len(hue_values) > 1:
            hue_diff = np.diff(hue_values)
            # Учитываем циклический характер hue
            hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff))
            scene_features['color_change_speed'] = float(np.mean(np.abs(hue_diff)))
            scene_features['color_transition_variance'] = float(np.var(hue_diff))
        
        # Flash events (резкие скачки яркости)
        if len(brightness_values) > 2:
            brightness_diff = np.diff(brightness_values)
            diff_abs = np.abs(brightness_diff)
            # Threshold must be in the same units as diff (Audit v3 fix).
            flash_threshold = float(np.mean(diff_abs) + 2.0 * np.std(diff_abs))
            flash_events = np.sum(diff_abs > flash_threshold)
            scene_features['flash_events_count'] = float(flash_events)
            # Нормированное количество вспышек относительно длины сцены
            scene_features['flash_events_count_norm'] = float(
                flash_events / float(max(1, num_frames - 1))
            )
        
        # Temporal Color Patterns
        if len(frame_features) > 1:
            # Color stability (стабильность цвета)
            color_stability = []
            for i in range(len(frame_features) - 1):
                f1 = frame_features[i]['features']
                f2 = frame_features[i + 1]['features']
                # Используем RGB mean для оценки стабильности
                rgb1 = np.array([f1.get('color_mean_r', 0), f1.get('color_mean_g', 0), f1.get('color_mean_b', 0)])
                rgb2 = np.array([f2.get('color_mean_r', 0), f2.get('color_mean_g', 0), f2.get('color_mean_b', 0)])
                diff = np.linalg.norm(rgb1 - rgb2)
                color_stability.append(diff)
            scene_features['color_stability'] = float(1.0 / (1.0 + np.mean(color_stability)))
            
            # Color temporal entropy
            hue_seq = [f['features'].get('hue_mean', 0) for f in frame_features]
            hue_hist, _ = np.histogram(hue_seq, bins=18)
            hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
            scene_features['color_temporal_entropy'] = float(entropy(hue_probs + 1e-10))
            
            # Color pattern periodicity (простая оценка через автокорреляцию)
            if len(hue_seq) > 3:
                hue_array = np.asarray(hue_seq, dtype=np.float64)
                theta = hue_array * (2.0 * np.pi / 180.0)
                z = np.exp(1j * theta)
                zc = z - np.mean(z)
                autocorr = np.correlate(zc, zc, mode="full")[len(zc) - 1 :]
                autocorr_abs = np.abs(autocorr)
                autocorr_abs = autocorr_abs / (autocorr_abs[0] + 1e-10)
                # Ищем второй пик (первый - это lag=0)
                if len(autocorr_abs) > 2:
                    peaks, _ = find_peaks(autocorr_abs[1:], height=0.3)
                    scene_features['color_pattern_periodicity'] = float(len(peaks) / len(autocorr_abs))
                else:
                    scene_features['color_pattern_periodicity'] = 0.0
            else:
                scene_features['color_pattern_periodicity'] = 0.0
            
            # Scene color shift speed
            if len(hue_values) > 1:
                scene_features['scene_color_shift_speed'] = float(np.mean(np.abs(np.diff(hue_values))))
        
        # Контраст и динамический диапазон (усредненные по сцене)
        contrast_values = [f['features'].get('global_contrast', 0) for f in frame_features]
        if contrast_values:
            scene_features['scene_contrast'] = float(np.mean(contrast_values))
        
        brightness_vals = [f['features'].get('brightness_mean', 0) for f in frame_features]
        if brightness_vals:
            scene_features['dynamic_range'] = float(np.max(brightness_vals) - np.min(brightness_vals))
        
        return scene_features
    
    def _compute_color_style_features(self, all_frame_features: List[Dict]) -> Dict[str, float]:
        """Вычисляет стили цветокоррекции"""
        features = {}
        
        # Собираем статистики по всем кадрам
        hue_means = [f['features'].get('hue_mean', 0) for f in all_frame_features]
        sat_means = [f['features'].get('saturation_mean', 0) for f in all_frame_features]
        rgb_means = []
        for f in all_frame_features:
            rgb_means.append([
                f['features'].get('color_mean_r', 0),
                f['features'].get('color_mean_g', 0),
                f['features'].get('color_mean_b', 0)
            ])
        
        if not rgb_means:
            return {f'style_{k}_prob': 0.0 for k in ['teal_orange', 'film', 'desaturated', 'hyper_saturated', 'vintage', 'tiktok']}
        
        rgb_means = np.array(rgb_means)
        avg_rgb = np.mean(rgb_means, axis=0)
        avg_sat = np.mean(sat_means) if sat_means else 128
        
        # Teal & Orange (теплые и холодные тона одновременно)
        # Оранжевый в RGB: высокий R и G, низкий B
        # Teal: низкий R, средний G, высокий B
        orange_score = (avg_rgb[0] + avg_rgb[1] - avg_rgb[2]) / 255.0
        teal_score = (avg_rgb[2] + avg_rgb[1] - avg_rgb[0]) / 255.0
        features['style_teal_orange_prob'] = float(min(orange_score * teal_score, 1.0))
        
        # Film look (низкая насыщенность, мягкие тона)
        low_sat = avg_sat < 100
        soft_tones = np.std(rgb_means) < 30
        features['style_film_prob'] = float(1.0 if (low_sat and soft_tones) else 0.3)
        
        # Desaturated
        features['style_desaturated_prob'] = float(1.0 - min(avg_sat / 128.0, 1.0))
        
        # Hyper saturated
        features['style_hyper_saturated_prob'] = float(min((avg_sat - 128) / 128.0, 1.0) if avg_sat > 128 else 0.0)
        
        # Vintage (сепия-подобные тона, низкая насыщенность)
        sepia_score = (avg_rgb[0] * 0.393 + avg_rgb[1] * 0.769 + avg_rgb[2] * 0.189) / 255.0
        features['style_vintage_prob'] = float(sepia_score * (1.0 - avg_sat / 255.0))
        
        # TikTok style (высокая насыщенность, яркие цвета)
        high_sat = avg_sat > 150
        bright = np.mean(avg_rgb) > 180
        features['style_tiktok_prob'] = float(1.0 if (high_sat and bright) else 0.2)
        
        return features
    
    def _compute_aesthetic_scores(self, all_frame_features: List[Dict]) -> Dict[str, float]:
        """
        Aesthetic & cinematic scores.
        Реальные модели должны быть подключены отдельно (NIMA / LAION / aesthetic heads).
        Пока модели не подключены — возвращаем NaN + *_present маски.
        """
        features = {
            "nima_mean": float("nan"),
            "nima_std": float("nan"),
            "laion_mean": float("nan"),
            "laion_std": float("nan"),
            "cinematic_lighting_score": float("nan"),
            "professional_look_score": float("nan"),
            "nima_present": 0.0,
            "laion_present": 0.0,
            "cinematic_present": 0.0,
            "professional_present": 0.0,
        }
        return features
    
    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Вычисляет коэффициент Джини"""
        if len(values) == 0:
            return 0.0
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n)
    
    def extract_video_features(self, all_scene_features: Dict[str, Dict], all_frame_features: Dict[str, Dict]) -> Dict[str, Any]:
        """Агрегирует video-level фичи."""
        features = {}

        # -------------------------------
        # 1. Проверки
        # -------------------------------
        if not all_scene_features or not all_frame_features:
            return features

        # -------------------------------
        # 2. Scene-level агрегаты
        # -------------------------------
        scene_agg = {}

        for scene_feat in all_scene_features.values():
            for key, val in scene_feat.items():
                if isinstance(val, (float, int)) and not isinstance(val, bool):
                    scene_agg.setdefault(key, []).append(val)

        for key, vals in scene_agg.items():
            if len(vals) > 0:
                features[f"{key}_mean"] = float(np.mean(vals))
                features[f"{key}_std"] = float(np.std(vals))
                features[f"{key}_min"] = float(np.min(vals))
                features[f"{key}_max"] = float(np.max(vals))

        # -------------------------------
        # 3. Собираем frame-level фичи
        # -------------------------------

        # Собираем ВСЕ кадры в список
        frame_list = []
        for scene_dict in all_frame_features.values():
            for frame_idx, frame_feat in scene_dict.items():
                frame_list.append(frame_feat)

        # Нечего анализировать
        if len(frame_list) == 0:
            return features

        # Универсальный safe getter
        def getf(frame, key, default=0.0):
            return frame.get(key, default)

        # -------------------------------
        # 4. Color/Hue distribution
        # -------------------------------
        hue_values = [getf(f, "hue_mean", 0) for f in frame_list]

        if len(hue_values) > 0:
            hue_hist, _ = np.histogram(hue_values, bins=36)
            hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
            features["color_distribution_entropy"] = float(entropy(hue_probs + 1e-10))
            features["color_distribution_gini"] = float(self._compute_gini_coefficient(np.array(hue_values)))

        # -------------------------------
        # 5. Color style features
        # -------------------------------
        features.update(self._compute_color_style_features(frame_list))

        # -------------------------------
        # 6. Aesthetic scores
        # -------------------------------
        features.update(self._compute_aesthetic_scores(frame_list))

        # -------------------------------
        # 7. Global brightness dynamics
        # -------------------------------
        brightness_values = [getf(f, "brightness_mean", 0) for f in frame_list]

        if len(brightness_values) > 1:
            diff = np.diff(brightness_values)
            features["global_brightness_change_speed"] = float(np.mean(np.abs(diff)))

        # -------------------------------
        # 8. Global color change speed
        # -------------------------------
        if len(hue_values) > 1:
            hue_diff = np.diff(hue_values)
            # корректный hue wrap-around
            hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff))
            features["global_color_change_speed"] = float(np.mean(np.abs(hue_diff)))

        # -------------------------------
        # 9. Strobe transitions
        # -------------------------------
        if len(brightness_values) > 2:
            diff = np.abs(np.diff(brightness_values))
            # Threshold must be defined in the same units as `diff` (Audit v3 fix).
            threshold = float(np.mean(diff) + 2.0 * np.std(diff))
            strobe_count = np.sum(diff > threshold)
            features["strobe_transition_frequency"] = float(strobe_count / len(diff))

        # -------------------------------
        # 10. Color periodicity + color shift
        # -------------------------------
        if len(hue_values) > 3:
            # Hue is circular; use complex representation to avoid wrap-around artifacts.
            hue_arr = np.asarray(hue_values, dtype=np.float64)
            theta = hue_arr * (2.0 * np.pi / 180.0)
            z = np.exp(1j * theta)
            zc = z - np.mean(z)
            autocorr = np.correlate(zc, zc, mode="full")[len(zc) - 1 :]
            autocorr_abs = np.abs(autocorr)
            autocorr_abs /= (autocorr_abs[0] + 1e-10)

            if len(autocorr_abs) > 2:
                peaks, _ = find_peaks(autocorr_abs[1:], height=0.2)
                features["global_color_periodicity"] = float(len(peaks) / len(autocorr_abs))
            else:
                features["global_color_periodicity"] = 0.0

            hue_diff = np.diff(hue_arr)
            hue_diff = np.minimum(np.abs(hue_diff), 180.0 - np.abs(hue_diff))
            features["global_color_shift"] = float(np.mean(np.abs(hue_diff)))

        return features

    
    def _create_sequence_inputs(
        self,
        all_frame_features: Dict[str, Dict[int, Dict]],
        all_scene_features: Dict[str, Dict[str, Any]],
        video_features: Dict[str, Any]
    ) -> Dict[str, List]:
        """
        Создает sequence inputs для трансформера
        
        Args:
            all_frame_features: {scene_label: {frame_idx: {"features": {...}}}}
            all_scene_features: {scene_label: {"feat1":..., "feat2":...}}
            video_features: {"feat": value}
        """
        sequences = {}

        # ============================================================
        # 1) FRAME SEQUENCE → [N_frames_total, D_frame_features]
        # ============================================================
        frame_sequence: List[List[float]] = []

        frame_keys = list(self.FRAME_COMPACT_KEYS)

        def _to_float_or_nan(v: Any) -> float:
            try:
                if v is None:
                    return float("nan")
                fv = float(v)
                return fv
            except Exception:
                return float("nan")

        # Собираем все кадры ПЛОСКО по всем сценам
        for scene_label, frames_dict in all_frame_features.items():
            for frame_idx, frame_feat in sorted(frames_dict.items(), key=lambda x: x[0]):
                feat_dict = frame_feat["features"]
                frame_vector = [
                    _to_float_or_nan(feat_dict.get(k)) for k in frame_keys
                ]
                frame_sequence.append(frame_vector)

        sequences["frames"] = frame_sequence

        # ============================================================
        # 2) SCENE SEQUENCE → [N_scenes, D_scene_features]
        # ============================================================
        scene_sequence = []

        for scene_label, scene_feat in all_scene_features.items():
            numeric_keys = sorted([
                k for k, v in scene_feat.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ])
            scene_vector = [float(scene_feat[k]) for k in numeric_keys]
            scene_sequence.append(scene_vector)

        sequences["scenes"] = scene_sequence

        # ============================================================
        # 3) GLOBAL → [D_global_features]
        # ============================================================
        global_sequence = []

        numeric_keys = sorted([
            k for k, v in video_features.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ])

        global_sequence = [float(video_features[k]) for k in numeric_keys]
        sequences["global"] = global_sequence

        return sequences

    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Главный метод обработки видео (интерфейс BaseModule).
        
        Args:
            frame_manager: FrameManager для доступа к кадрам
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля (не используется, но требуется BaseModule)
        
        Returns:
            Словарь с результатами в формате для сохранения в npz
        """
        if not frame_indices:
            raise RuntimeError(f"{self.module_name} | process | frame_indices is empty")
        
        # Baseline contract: stage timings
        stage_timings_ms: Dict[str, float] = {}
        t0 = time.time()
        
        # Stage: initialization
        t_stage = time.time()
        self.initialize()  # Гарантируем инициализацию (если еще не была)
        stage_timings_ms["initialization"] = (time.time() - t_stage) * 1000.0
        
        # Stage: load_deps
        t_stage = time.time()
        # Загружаем scenes из scene_classification
        scene_data = self.load_dependency_results("scene_classification")
        stage_timings_ms["load_deps"] = (time.time() - t_stage) * 1000.0
        
        if scene_data is None:
            raise RuntimeError(
                f"{self.module_name} | process | scene_classification не найдены. "
                f"Убедитесь, что модуль scene_classification запущен перед этим модулем. "
                f"rs_path: {self.rs_path}"
            )
        
        # Canonical contract (audit): scene_classification exports `scenes` (dict) in NPZ.
        scenes = None
        if isinstance(scene_data, dict):
            if isinstance(scene_data.get("scenes"), dict):
                scenes = scene_data.get("scenes")
            elif isinstance(scene_data.get("scenes_raw"), dict):
                # legacy alias
                scenes = scene_data.get("scenes_raw")
        
        if scenes is None:
            raise ValueError(
                f"{self.module_name} | process | Не удалось извлечь scenes из данных scene_classification. "
                f"Проверьте формат сохраненных данных."
            )
        
        # Обработка кадров по сценам
        all_frame_features = {}
        all_scene_features = {}
        sequence_frame_indices: List[int] = []

        meta = getattr(frame_manager, "meta", {}) or {}
        union_timestamps = meta.get("union_timestamps_sec")
        total_frames = int(meta.get("total_frames") or meta.get("num_frames") or 0)
        color_space = meta.get("color_space")

        if union_timestamps is None:
            raise RuntimeError(
                f"{self.module_name} | process | union_timestamps_sec missing in frames metadata"
            )
        if not isinstance(union_timestamps, (list, tuple, np.ndarray)):
            raise RuntimeError(
                f"{self.module_name} | process | union_timestamps_sec has invalid type: {type(union_timestamps)}"
            )
        union_timestamps = np.asarray(union_timestamps, dtype=np.float32)

        allowed_set = set(int(x) for x in frame_indices)
        
        # Stage: process_frames
        t_stage = time.time()
        scene_items = []
        for scene_id, scene in scenes.items():
            if not isinstance(scene, dict) or "indices" not in scene:
                continue
            indices = list(scene.get("indices") or [])
            if not indices:
                continue
            scene_items.append((int(indices[0]), scene_id, scene))

        # Precompute per-scene frame indices to have a correct progress denominator.
        scene_tasks: List[Tuple[str, str, str, np.ndarray, int, int]] = []
        for _, scene_id, scene in sorted(scene_items, key=lambda x: x[0]):
            if not isinstance(scene, dict) or "indices" not in scene:
                continue
            indices = list(scene.get("indices") or [])
            if not indices:
                continue
            scene_label = str(scene.get("scene_label") or "unknown")
            # Avoid collisions when the same label appears in multiple disjoint scenes.
            scene_key = f"{scene_label}__{scene_id}"
            start_frame = int(indices[0])
            end_frame = int(indices[-1])

            # строго следуем Segmenter: пересэмплинг запрещён
            scene_frame_indices = [int(idx) for idx in indices if int(idx) in allowed_set]
            if not scene_frame_indices:
                continue
            # Убираем дубликаты и сортируем (сохраняем временной порядок)
            sfi = np.unique(np.asarray(scene_frame_indices, dtype=np.int32))
            if sfi.size == 0:
                continue
            scene_tasks.append((scene_key, scene_label, str(scene_id), sfi, start_frame, end_frame))

        total_frames_to_process = int(sum(int(t[3].size) for t in scene_tasks))
        if total_frames_to_process <= 0:
            total_frames_to_process = int(max(1, len(frame_indices)))

        frames_done = 0
        progress_interval = max(1, total_frames_to_process // 15)  # ~15 updates/run
        run_meta = getattr(frame_manager, "meta", {}) or {}
        total_scenes = int(len(scene_tasks))

        for i, (scene_key, scene_label, scene_id_str, scene_frame_indices, start_frame, end_frame) in enumerate(scene_tasks):
            all_frame_features[scene_key] = {}
            scene_frame_features: List[Dict[str, Any]] = []

            # ==== FEATURE EXTRACTION ====
            for k, frame_idx in enumerate(scene_frame_indices.tolist()):
                frame = frame_manager.get(int(frame_idx))
                frame_feat = self.extract_frame_features(frame, int(frame_idx), color_space=color_space)

                scene_frame_features.append(frame_feat)
                all_frame_features[scene_key][int(frame_idx)] = frame_feat
                sequence_frame_indices.append(int(frame_idx))

                logger.info(
                    f"Сцена {i+1}/{total_scenes} | Кадр {k+1}/{int(scene_frame_indices.size)} обработан"
                )

                # Baseline contract: granular progress (>=10 updates per run)
                if frames_done % progress_interval == 0 or frames_done == total_frames_to_process - 1:
                    self._append_state_event_if_possible(
                        rs_path=self.rs_path or "",
                        event={
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "scope": "progress",
                            "processor": "visual",
                            "component": self.module_name,
                            "status": "running",
                            "progress": float(frames_done) / float(max(1, total_frames_to_process)),
                            "done": int(frames_done),
                            "total": int(total_frames_to_process),
                            "stage": "process_frames",
                            "platform_id": run_meta.get("platform_id"),
                            "video_id": run_meta.get("video_id"),
                            "run_id": run_meta.get("run_id"),
                        },
                    )

                frames_done += 1

            # ==== SCENE FEATURES ====
            if scene_frame_features:
                scene_feat = self.extract_scene_features(
                    scene_frame_features,
                    start_frame,
                    end_frame,
                    total_frames=total_frames if total_frames > 0 else None
                )
                scene_feat = dict(scene_feat or {})
                scene_feat["scene_label"] = scene_label
                scene_feat["scene_id"] = str(scene_id_str)
                all_scene_features[scene_key] = scene_feat

            logger.info(f"Сцена {i+1}/{total_scenes} | Scene-level фичи извлечены")
        
        stage_timings_ms["process_frames"] = (time.time() - t_stage) * 1000.0
        
        # Stage: post_process (aggregate video features)
        t_stage = time.time()
        # Если после фильтрации нет ни одного кадра — валидная пустота
        if not sequence_frame_indices:
            d = int(len(self.FRAME_COMPACT_KEYS))
            stage_timings_ms["post_process"] = (time.time() - t_stage) * 1000.0
            stage_timings_ms["total"] = (time.time() - t0) * 1000.0
            result = {
                "frames": {},
                "scenes": {},
                "video_features": {},
                "sequence_inputs": {"frames": [], "scenes": [], "global": []},
                "frame_indices": np.asarray([], dtype=np.int32),
                "times_s": np.asarray([], dtype=np.float32),
                "sequence_frame_indices": np.asarray([], dtype=np.int32),
                "sequence_times_s": np.asarray([], dtype=np.float32),
                "frame_compact_features": np.full((0, d), np.nan, dtype=np.float32),
                "frame_compact_feature_names": np.asarray(list(self.FRAME_COMPACT_KEYS), dtype=object),
                "frame_compact_frame_indices": np.asarray([], dtype=np.int32),
                "aggregated": {
                    "frame_compact": {
                        "feature_names": np.asarray(list(self.FRAME_COMPACT_KEYS), dtype=object),
                        "mean": np.full((d,), np.nan, dtype=np.float32),
                        "std": np.full((d,), np.nan, dtype=np.float32),
                        "p25": np.full((d,), np.nan, dtype=np.float32),
                        "p50": np.full((d,), np.nan, dtype=np.float32),
                        "p75": np.full((d,), np.nan, dtype=np.float32),
                        "rows": 0,
                        "valid_rows": 0,
                    }
                },
                "_status": "empty",
                "_empty_reason": "after_filt_empty",
                "_stage_timings_ms": stage_timings_ms,
            }
            return result

        # Video-level фичи
        video_features = self.extract_video_features(all_scene_features, all_frame_features)

        logger.info(f"Видео фичи извлечены")

        # Формируем последовательности для трансформера на основе компактных фич
        sequence_inputs = self._create_sequence_inputs(
            all_frame_features, all_scene_features, video_features
        )

        # Fixed model-facing compact arrays (stable for models).
        frame_compact_feature_names = np.asarray(list(self.FRAME_COMPACT_KEYS), dtype=object)
        frame_compact_features = np.asarray(sequence_inputs.get("frames") or [], dtype=np.float32)
        if frame_compact_features.ndim == 1:
            frame_compact_features = frame_compact_features.reshape((0, int(frame_compact_feature_names.size)))
        if frame_compact_features.ndim != 2 or frame_compact_features.shape[1] != int(frame_compact_feature_names.size):
            raise RuntimeError(
                f"{self.module_name} | process | frame_compact_features shape mismatch: got {frame_compact_features.shape}, expected (*,{int(frame_compact_feature_names.size)})"
            )
        frame_compact_frame_indices = np.asarray(sequence_frame_indices, dtype=np.int32)

        # Aggregated (tabular) stats for baseline heads.
        def _nan_stats(mat: np.ndarray) -> Dict[str, Any]:
            if mat.size == 0:
                d = int(frame_compact_feature_names.size)
                nanv = np.full((d,), np.nan, dtype=np.float32)
                return {
                    "feature_names": frame_compact_feature_names,
                    "mean": nanv,
                    "std": nanv,
                    "p25": nanv,
                    "p50": nanv,
                    "p75": nanv,
                    "rows": 0,
                    "valid_rows": 0,
                }
            valid_row = np.all(np.isfinite(mat), axis=1)
            sub = mat[valid_row]
            if sub.size == 0:
                d = int(frame_compact_feature_names.size)
                nanv = np.full((d,), np.nan, dtype=np.float32)
                return {
                    "feature_names": frame_compact_feature_names,
                    "mean": nanv,
                    "std": nanv,
                    "p25": nanv,
                    "p50": nanv,
                    "p75": nanv,
                    "rows": int(mat.shape[0]),
                    "valid_rows": 0,
                }
            return {
                "feature_names": frame_compact_feature_names,
                "mean": np.nanmean(sub, axis=0).astype(np.float32),
                "std": np.nanstd(sub, axis=0).astype(np.float32),
                "p25": np.nanpercentile(sub, 25, axis=0).astype(np.float32),
                "p50": np.nanpercentile(sub, 50, axis=0).astype(np.float32),
                "p75": np.nanpercentile(sub, 75, axis=0).astype(np.float32),
                "rows": int(mat.shape[0]),
                "valid_rows": int(sub.shape[0]),
            }

        aggregated = {
            "frame_compact": _nan_stats(frame_compact_features),
        }
        
        stage_timings_ms["post_process"] = (time.time() - t_stage) * 1000.0
        stage_timings_ms["total"] = (time.time() - t0) * 1000.0

        # Сортированный список индексов для контрактных полей
        unique_frame_indices = np.unique(np.asarray(sequence_frame_indices, dtype=np.int32))
        # times_s строго из union_timestamps_sec
        try:
            times_s = union_timestamps[unique_frame_indices].astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | process | failed to build times_s from union_timestamps_sec: {e}"
            ) from e
        try:
            sequence_times_s = union_timestamps[np.asarray(sequence_frame_indices, dtype=np.int32)].astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"{self.module_name} | process | failed to build sequence_times_s: {e}"
            ) from e

        # Формируем результат
        result = {
            "frames": all_frame_features if self.store_debug_objects else {},
            "scenes": all_scene_features if self.store_debug_objects else {},
            "video_features": video_features,
            # sequence_inputs is kept as compat/debug (variable dims for scenes/global).
            "sequence_inputs": sequence_inputs,
            "frame_indices": unique_frame_indices.astype(np.int32),
            "times_s": times_s.astype(np.float32),
            "sequence_frame_indices": np.asarray(sequence_frame_indices, dtype=np.int32),
            "sequence_times_s": sequence_times_s.astype(np.float32),
            "frame_compact_features": frame_compact_features.astype(np.float32),
            "frame_compact_feature_names": frame_compact_feature_names,
            "frame_compact_frame_indices": frame_compact_frame_indices,
            "aggregated": aggregated,
            "_status": "ok",
            "_empty_reason": None,
            "_stage_timings_ms": stage_timings_ms,
        }

        return result
    
    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Полный цикл обработки с baseline contracts (stage_timings_ms, state_events progress).
        """
        if metadata is None:
            metadata = self.load_metadata(frames_dir)
        self._last_metadata = metadata

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise ValueError(f"{self.module_name} | Нет кадров для обработки")

        # Baseline contract: stage timings
        stage_timings_ms: Dict[str, float] = {}
        t0 = time.time()

        def _resource_profile_snapshot() -> Dict[str, Any]:
            """
            Best-effort, env-gated resource snapshot for Audit 4.2.
            """
            if str(os.environ.get("VP_RESOURCE_PROFILE", "")).strip().lower() not in ("1", "true", "yes", "on"):
                return {}
            snap: Dict[str, Any] = {}
            try:
                import psutil  # type: ignore
                snap["rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                pass
            return snap

        resource_profile_before = _resource_profile_snapshot()

        frame_manager = None
        try:
            # Stage: start
            self._append_state_event_if_possible(
                rs_path=self.rs_path or "",
                event={
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "scope": "progress",
                    "processor": "visual",
                    "component": self.module_name,
                    "status": "running",
                    "stage": "start",
                    "platform_id": metadata.get("platform_id"),
                    "video_id": metadata.get("video_id"),
                    "run_id": metadata.get("run_id"),
                },
            )
            
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            self.logger.info(
                f"{self.module_name} | Начало обработки {len(frame_indices)} кадров"
            )

            # Stage: process (includes load_deps, process_frames, aggregate)
            results = self.process(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                config=config
            )
            
            # Extract stage_timings from results
            stage_timings_ms = results.get("_stage_timings_ms", {})
            stage_timings_ms["total"] = (time.time() - t0) * 1000.0

            status = results.get("_status", "ok")
            empty_reason = results.get("_empty_reason")

            # Baseline contract: все обязательные meta поля
            save_metadata: Dict[str, Any] = {
                # Базовые поля
                "producer": self.module_name,
                "producer_version": self.VERSION,
                "schema_version": self.SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                # Run identity (обязательно)
                "platform_id": metadata.get("platform_id"),
                "video_id": metadata.get("video_id"),
                "run_id": metadata.get("run_id"),
                "config_hash": metadata.get("config_hash"),
                "sampling_policy_version": metadata.get("sampling_policy_version"),
                "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                # Статус
                "status": status,
                "empty_reason": empty_reason,
                # Дополнительные поля
                "total_frames": metadata.get("total_frames"),
                "processed_frames": len(frame_indices),
                "frames_dir": frames_dir,
                "analysis_fps": metadata.get("analysis_fps"),
                "analysis_width": metadata.get("analysis_width"),
                "analysis_height": metadata.get("analysis_height"),
            }
            # PR-3: model meta (even if empty — must be present for contract stability)
            save_metadata = apply_models_meta(
                save_metadata, models_used=self.get_models_used(config=config or {}, metadata=metadata or {})
            )

            # Audit v3: reproducibility / config highlights
            save_metadata["store_debug_objects"] = bool(self.store_debug_objects)
            # Deprecated sampling knobs kept for compat (do not affect sampling anymore)
            save_metadata["max_frames_per_scene"] = int(self.max_frames_per_scene)
            save_metadata["stride"] = int(self.stride)
            # Hard-coded algorithmic knobs (to keep results reproducible across refactors)
            save_metadata["hue_hist_bins"] = 36
            save_metadata["palette_sample_size"] = 10000
            save_metadata["palette_kmeans_max_colors"] = 3
            save_metadata["palette_kmeans_random_state"] = 42
            save_metadata["palette_kmeans_n_init"] = 10

            # Audit v3: sampling policy provenance
            save_metadata["module_sampling_policy_version"] = "segmenter_axis_v1"
            # Contract highlights (fixed model-facing compact vector)
            save_metadata["frame_compact_dim"] = int(len(self.FRAME_COMPACT_KEYS))
            if resource_profile_before:
                save_metadata["resource_profile_before"] = resource_profile_before

            # Stage: save
            t_stage = time.time()
            saved_path = self.save_results(
                results=results,
                metadata=save_metadata,
                use_compressed=False
            )
            stage_timings_ms["save"] = (time.time() - t_stage) * 1000.0
            
            # Emit done stage
            self._append_state_event_if_possible(
                rs_path=self.rs_path or "",
                event={
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "scope": "progress",
                    "processor": "visual",
                    "component": self.module_name,
                    "status": status,
                    "stage": "done",
                    "platform_id": metadata.get("platform_id"),
                    "video_id": metadata.get("video_id"),
                    "run_id": metadata.get("run_id"),
                },
            )

            self.logger.info(
                f"{self.module_name} | Обработка завершена. Результаты сохранены: {saved_path}"
            )
            return saved_path
        finally:
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception as e:
                    self.logger.exception(
                        f"{self.module_name} | Ошибка при закрытии FrameManager: {e}"
                    )

    def save_results(
        self,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        use_compressed: bool = True,
        embeddings_key: Optional[str] = None
    ) -> str:
        """
        Переопределяем сохранение, чтобы корректно выставлять status/empty_reason и добавлять ui_payload/stage_timings_ms.
        """
        meta = dict(metadata or {})
        status = results.get("_status")
        empty_reason = results.get("_empty_reason")
        stage_timings_ms = results.get("_stage_timings_ms")
        
        if status in ("ok", "empty", "error"):
            meta["status"] = status
            meta["empty_reason"] = empty_reason
        
        # Baseline contract (Audit v3): stage_timings_ms must be present in meta (top-level).
        if not isinstance(stage_timings_ms, dict):
            stage_timings_ms = {}
        meta["stage_timings_ms"] = stage_timings_ms
        # Backward-compat: keep older nesting too.
        if "summary" not in meta:
            meta["summary"] = {}
        if isinstance(meta.get("summary"), dict):
            meta["summary"]["stage_timings_ms"] = stage_timings_ms
        
        # Baseline contract: ui_payload in meta
        if status == "ok" and not meta.get("ui_payload"):
            try:
                meta["ui_payload"] = self._build_ui_payload(results, meta)
            except Exception as e:
                logger.warning(f"{self.module_name} | Failed to build UI payload: {e}")
        
        clean_results = dict(results)
        clean_results.pop("_status", None)
        clean_results.pop("_empty_reason", None)
        clean_results.pop("_stage_timings_ms", None)
        return super().save_results(
            results=clean_results,
            metadata=meta,
            use_compressed=use_compressed,
            embeddings_key=embeddings_key
        )
