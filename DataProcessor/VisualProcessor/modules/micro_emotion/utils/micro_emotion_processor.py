"""
Micro Emotion Processor - Optimized version
Обрабатывает данные OpenFace и извлекает оптимизированные фичи:
- Ключевые AU (10-14) с baseline subtraction
- PCA для остальных AU
- Компактные метрики pose, gaze, landmarks
- Micro-expressions detection
- Per-frame векторы для VisualTransformer

Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (core_face_landmarks)
    2. ✅ Использование результатов core провайдеров вместо прямых вызовов моделей
    3. ✅ Интеграция с BaseModule через класс MicroEmotionModule
    4. ✅ Единый формат вывода для сохранения в npz
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Sequence
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Добавляем путь для импорта BaseModule
_vp_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _vp_root not in sys.path:
    sys.path.insert(0, _vp_root)
_repo_root = os.path.dirname(_vp_root)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger

from modules.micro_emotion.utils.openface_analyzer import OpenFaceAnalyzer  # type: ignore

# Ключевые AU для UGC/вовлечённости
KEY_AUS = ['AU06', 'AU12', 'AU04', 'AU01', 'AU02', 'AU25', 'AU26', 'AU07', 'AU23', 'AU45', 'AU43', 'AU15', 'AU20', 'AU10']

# AU для micro-expressions detection
MICROEXPR_AU_COMBINATIONS = {
    'smile': ['AU06', 'AU12'],
    'surprise': ['AU01', 'AU02', 'AU25', 'AU26'],
    'frown': ['AU04', 'AU15'],
    'disgust': ['AU09', 'AU10'],
}

MODULE_NAME = "micro_emotion"
VERSION = "2.0.2"
SCHEMA_VERSION = "micro_emotion_npz_v3"
ARTIFACT_FILENAME = "micro_emotion.npz"

LOGGER = get_logger(MODULE_NAME)

# Best-effort resource profiling (env-gated; consistent across VisualProcessor modules)
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

# Fixed model-facing scalar feature set (video-level), tabular contract.
# Values are stored as:
# - feature_names: object[F]
# - feature_values: float32[F]
#
# Missing/unavailable values are stored as NaN.
_FEATURE_NAMES_V1: Tuple[str, ...] = (
    # Presence / counts
    "has_faces",
    "frames_n",
    "frames_with_face",
    "frames_processed_openface",
    # Micro-expressions (scalars)
    "microexpr_count",
    "microexpr_rate_per_min",
    "microexpr_max_intensity",
    # High-level ratios / scores
    "smile_ratio",
    "eye_contact_ratio",
    "blink_rate_per_min",
    "eye_contact_score",
    "pose_stability_score",
    "face_presence_ratio",
    # Reliability flags / proxies
    "au_quality_overall",
    "au_quality_reliable",
    "landmark_visibility_mean",
    "landmark_visibility_reliable",
    "occlusion_flag",
    "lighting_flag",
    # Pose aggregates
    "pose_Rx_mean",
    "pose_Rx_std",
    "pose_Rx_min",
    "pose_Rx_max",
    "pose_Ry_mean",
    "pose_Ry_std",
    "pose_Ry_min",
    "pose_Ry_max",
    "pose_Rz_mean",
    "pose_Rz_std",
    "pose_Rz_min",
    "pose_Rz_max",
    "pose_Tz_mean",
    "pose_Tz_std",
    # Gaze aggregates
    "gaze_x_mean",
    "gaze_x_std",
    "gaze_y_mean",
    "gaze_y_std",
    "gaze_centered_ratio",
    # Landmarks / geometry aggregates
    "mouth_opening_mean",
    "mouth_opening_std",
    "smile_width_mean",
    "smile_width_std",
    "face_asymmetry_score",
    "head_depth_variation",
    # PCA summaries
    "au_pca_var_explained_1",
    "au_pca_var_explained_2",
    "au_pca_var_explained_3",
    "au_pca_var_explained_4",
    "au_pca_var_explained_5",
    "landmarks_pca_1",
    "landmarks_pca_2",
    "landmarks_pca_3",
    "landmarks_pca_4",
    "landmarks_pca_5",
    # Key AU aggregates (flattened video-level)
    "AU04_mean",
    "AU04_std",
    "AU04_peak_count",
    "AU06_mean",
    "AU06_std",
    "AU06_peak_count",
    "AU07_mean",
    "AU07_std",
    "AU07_peak_count",
    "AU12_mean",
    "AU12_std",
    "AU12_peak_count",
    "AU15_mean",
    "AU15_std",
    "AU15_peak_count",
    "AU25_mean",
    "AU25_std",
    "AU25_peak_count",
    "AU26_mean",
    "AU26_std",
    "AU26_peak_count",
)

# Model-facing compact vector contract (fixed 22 dims).
COMPACT22_FEATURE_NAMES: List[str] = [
    "time_norm",
    "face_presence_flag",
    "AU12_delta_norm",
    "AU06_delta_norm",
    "AU04_delta_norm",
    "AU25_delta_norm",
    "AU25_presence_rate_1s",
    "blink_flag",
    "pose_Ry_norm",
    "pose_Rx_norm",
    "gaze_centered_flag",
    "gaze_x_norm",
    "gaze_y_norm",
    "mouth_opening_norm",
    "face_asymmetry_score",
    "microexpr_recent_count",
    "au_pca_0",
    "au_pca_1",
    "au_pca_2",
    "au_quality_flag",
    "pose_Rz_norm",
    "pose_Tz_norm",
]


# -------------------------
# Progress to state_events.jsonl (PR-5) — same mechanism as frames_composition
# -------------------------
def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    try:
        run_rs = Path(rs_path).resolve()
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


def _times_s_from_union(*, metadata: Dict[str, Any], frame_indices: np.ndarray) -> np.ndarray:
    uts = metadata.get("union_timestamps_sec")
    if uts is None:
        raise RuntimeError("micro_emotion | metadata missing union_timestamps_sec (no-fallback)")
    uts = np.asarray(uts, dtype=np.float32).reshape(-1)
    if uts.size == 0:
        raise RuntimeError("micro_emotion | union_timestamps_sec is empty")
    if frame_indices.size == 0:
        raise RuntimeError("micro_emotion | frame_indices is empty")
    if int(np.max(frame_indices)) >= int(uts.size) or int(np.min(frame_indices)) < 0:
        raise RuntimeError("micro_emotion | frame_indices out of bounds for union_timestamps_sec")
    times_s = uts[frame_indices.astype(np.int64)]
    if times_s.size >= 2 and not bool(np.all(np.diff(times_s) >= -1e-6)):
        raise RuntimeError("micro_emotion | times_s is not monotonic (unexpected union timeline)")
    return times_s.astype(np.float32)


def _pack_openface_68_landmarks_for_ui(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Pack OpenFace 68 x/y landmarks for UI only.
    We use pixel coords as produced by OpenFace (x_0..x_67, y_0..y_67).
    """
    x_cols = [f"x_{i}" for i in range(68)]
    y_cols = [f"y_{i}" for i in range(68)]
    if not all(c in df.columns for c in (x_cols + y_cols)):
        return None
    x = df[x_cols].to_numpy(dtype=np.float32)
    y = df[y_cols].to_numpy(dtype=np.float32)
    pts = np.stack([x, y], axis=2)  # (M, 68, 2)
    # frame_union provides mapping to union frame index
    fu = df["frame_union"].to_numpy(dtype=np.int32) if "frame_union" in df.columns else None
    return {
        "schema_version": "openface_landmarks68_ui_v1",
        "frame_indices": fu.tolist() if fu is not None else [],
        "landmarks_xy": pts.tolist(),
    }


class MicroEmotionProcessor:
    """Обработчик данных OpenFace с оптимизацией для VisualTransformer"""
    
    def __init__(
        self,
        fps: int = 30,
        microexpr_smoothing_sigma: float = 0.05,  # 0.03-0.1s
        microexpr_delta_threshold: float = 0.4,  # raw intensity change
        microexpr_max_duration_frames: int = 15,  # 0.5s at 30fps
        microexpr_min_peak_distance_frames: int = 6,  # 0.2s at 30fps
        gaze_centered_threshold: float = 10.0,  # degrees
        pca_components: int = 3,
        au_confidence_threshold: float = 0.5,
    ):
        """
        fps: кадров в секунду
        microexpr_smoothing_sigma: сглаживание для micro-expressions (в секундах)
        microexpr_delta_threshold: порог изменения интенсивности для micro-expression
        microexpr_max_duration_frames: максимальная длительность micro-expression в кадрах
        microexpr_min_peak_distance_frames: минимальное расстояние между пиками
        gaze_centered_threshold: порог для определения взгляда в камеру (градусы)
        pca_components: количество PCA компонент для AU
        au_confidence_threshold: порог уверенности AU
        """
        self.fps = fps
        self.microexpr_sigma = microexpr_smoothing_sigma * fps  # в кадрах
        self.microexpr_delta_threshold = microexpr_delta_threshold
        self.microexpr_max_duration_frames = microexpr_max_duration_frames
        self.microexpr_min_peak_distance_frames = microexpr_min_peak_distance_frames
        self.gaze_centered_threshold = gaze_centered_threshold
        self.pca_components = pca_components
        self.au_confidence_threshold = au_confidence_threshold
        
        self.pca_au = None
        self.pca_landmarks = None
        self.au_baseline = None
        
    def _smooth(self, x: np.ndarray, sigma: float = None) -> np.ndarray:
        """Сглаживание сигнала"""
        if x is None or len(x) == 0:
            return np.array([])
        if sigma is None:
            sigma = self.microexpr_sigma
        return gaussian_filter1d(x.astype(float), sigma=sigma)
    
    def _normalize_01(self, x: np.ndarray) -> np.ndarray:
        """Нормализация в [0, 1]"""
        if x is None or len(x) == 0:
            return np.array([])
        x = np.array(x, dtype=float)
        mi, ma = x.min(), x.max()
        if ma - mi < 1e-9:
            return np.zeros_like(x)
        return (x - mi) / (ma - mi)
    
    def _z_normalize(self, x: np.ndarray, mean: float = None, std: float = None) -> Tuple[np.ndarray, float, float]:
        """Z-нормализация"""
        if x is None or len(x) == 0:
            return np.array([]), 0.0, 1.0
        x = np.array(x, dtype=float)
        if mean is None:
            mean = float(x.mean())
        if std is None:
            std = float(x.std()) + 1e-9
        return (x - mean) / std, mean, std
    
    def compute_au_baseline(self, df: pd.DataFrame, au_columns: List[str]) -> Dict[str, float]:
        """
        Вычисляет baseline (нейтральное состояние) для каждого AU.
        Использует нижние 20% кадров по общей активности AU.
        """
        if len(df) == 0:
            return {au: 0.0 for au in au_columns}
        
        # Вычисляем общую активность AU для каждого кадра
        au_intensity_cols = [col for col in au_columns if col.endswith('_r')]
        if len(au_intensity_cols) == 0:
            return {au: 0.0 for au in au_columns}
        
        total_activity = df[au_intensity_cols].sum(axis=1)
        
        # Выбираем нижние 20% кадров как нейтральные
        threshold = np.percentile(total_activity, 20)
        neutral_frames = df[total_activity <= threshold]
        
        baseline = {}
        for au in au_columns:
            intensity_col = f"{au}_r"
            if intensity_col in neutral_frames.columns:
                baseline[au] = float(neutral_frames[intensity_col].mean())
            else:
                baseline[au] = 0.0
        
        return baseline
    
    def extract_key_au_features(
        self,
        df: pd.DataFrame,
        au_baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Извлекает фичи для ключевых AU.
        Для каждого AU: intensity_mean, intensity_std, presence_rate, peak_count, intensity_delta_mean
        """
        features = {}
        
        if au_baseline is None:
            au_baseline = {}
        
        for au in KEY_AUS:
            intensity_col = f"{au}_r"
            presence_col = f"{au}_c"
            
            if intensity_col not in df.columns:
                features[au] = {
                    'intensity_mean': 0.0,
                    'intensity_std': 0.0,
                    'presence_rate': 0.0,
                    'peak_count': 0,
                    'intensity_delta_mean': 0.0,
                }
                continue
            
            intensities = df[intensity_col].fillna(0.0).values
            baseline = au_baseline.get(au, 0.0)
            
            # Baseline subtraction
            intensities_delta = intensities - baseline
            
            # Presence rate
            if presence_col in df.columns:
                presence = df[presence_col].fillna(0.0).values
                presence_rate = float(np.mean(presence > 0.5))
            else:
                # Infer presence from intensity
                presence_rate = float(np.mean(intensities > 0.1))
            
            # Peak detection для интенсивности
            smoothed = self._smooth(intensities, sigma=self.microexpr_sigma)
            peaks, _ = find_peaks(
                smoothed,
                height=baseline + 1.5 * np.std(smoothed),
                distance=self.microexpr_min_peak_distance_frames,
            )
            peak_count = len(peaks)
            
            features[au] = {
                'intensity_mean': float(np.mean(intensities)),
                'intensity_std': float(np.std(intensities)),
                'presence_rate': presence_rate,
                'peak_count': peak_count,
                'intensity_delta_mean': float(np.mean(intensities_delta)),
            }
        
        return features
    
    def compute_au_pca(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[np.ndarray, Optional[PCA]]:
        """
        Вычисляет PCA для всех AU интенсивностей (кроме ключевых).
        Возвращает проекции и модель PCA.
        """
        # Получаем все AU колонки интенсивности
        all_au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        non_key_au_cols = [col for col in all_au_cols if col.replace('_r', '') not in KEY_AUS]
        
        if len(non_key_au_cols) == 0:
            # Если нет неключевых AU, используем все AU
            non_key_au_cols = all_au_cols
        
        if len(non_key_au_cols) == 0:
            return np.zeros((len(df), self.pca_components)), None
        
        au_matrix = df[non_key_au_cols].fillna(0.0).values
        n_samples, n_features = au_matrix.shape
        # sklearn PCA requires n_components <= min(n_samples, n_features); after dropping bad
        # OpenFace rows n_samples can be small → would raise ValueError and exit 3 from CLI.
        max_comp = int(min(self.pca_components, n_samples, n_features)) if n_samples and n_features else 0

        if fit:
            if max_comp < 1:
                self.pca_au = None
                return np.zeros((len(df), self.pca_components), dtype=np.float32), None
            self.pca_au = PCA(n_components=max_comp)
            pca_features = self.pca_au.fit_transform(au_matrix)
        else:
            if self.pca_au is None:
                return np.zeros((len(df), self.pca_components), dtype=np.float32), None
            pca_features = self.pca_au.transform(au_matrix)

        if pca_features.shape[1] < self.pca_components:
            pad = self.pca_components - int(pca_features.shape[1])
            pca_features = np.hstack(
                [pca_features, np.zeros((pca_features.shape[0], pad), dtype=pca_features.dtype)]
            )
        return pca_features.astype(np.float32), self.pca_au
    
    def detect_micro_expressions(
        self,
        df: pd.DataFrame,
        au_baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Детектирует micro-expressions как быстрые вспышки AU интенсивности.
        """
        if au_baseline is None:
            au_baseline = {}
        
        microexpr_timestamps = []
        microexpr_types = []
        microexpr_intensities = []
        
        for expr_type, au_list in MICROEXPR_AU_COMBINATIONS.items():
            # Комбинируем AU для данного типа выражения
            combined_intensity = None
            
            for au in au_list:
                intensity_col = f"{au}_r"
                if intensity_col not in df.columns:
                    continue
                
                intensities = df[intensity_col].fillna(0.0).values
                baseline = au_baseline.get(au, 0.0)
                intensities_delta = intensities - baseline
                intensities_norm = self._normalize_01(intensities_delta)
                
                if combined_intensity is None:
                    combined_intensity = intensities_norm
                else:
                    combined_intensity = np.maximum(combined_intensity, intensities_norm)
            
            if combined_intensity is None or len(combined_intensity) == 0:
                continue
            
            # Сглаживание
            smoothed = self._smooth(combined_intensity, sigma=self.microexpr_sigma)
            
            # Детекция пиков
            threshold = np.mean(smoothed) + 1.5 * np.std(smoothed)
            peaks, properties = find_peaks(
                smoothed,
                height=threshold,
                distance=self.microexpr_min_peak_distance_frames,
                width=(1, self.microexpr_max_duration_frames),
            )
            
            for peak_idx in peaks:
                timestamp = peak_idx / self.fps
                intensity = float(smoothed[peak_idx])
                microexpr_timestamps.append(timestamp)
                microexpr_types.append(expr_type)
                microexpr_intensities.append(intensity)
        
        # Сортируем по времени
        if len(microexpr_timestamps) > 0:
            sorted_indices = np.argsort(microexpr_timestamps)
            microexpr_timestamps = [microexpr_timestamps[i] for i in sorted_indices]
            microexpr_types = [microexpr_types[i] for i in sorted_indices]
            microexpr_intensities = [microexpr_intensities[i] for i in sorted_indices]
        
        # Распределение типов
        types_distribution = {}
        for expr_type in MICROEXPR_AU_COMBINATIONS.keys():
            count = sum(1 for t in microexpr_types if t == expr_type)
            types_distribution[expr_type] = count
        
        duration_minutes = len(df) / (self.fps * 60.0) if len(df) > 0 else 1.0
        
        return {
            'microexpr_count': len(microexpr_timestamps),
            'microexpr_rate_per_min': len(microexpr_timestamps) / duration_minutes if duration_minutes > 0 else 0.0,
            'microexpr_max_intensity': float(max(microexpr_intensities)) if microexpr_intensities else 0.0,
            'microexpr_types_distribution': types_distribution,
            'microexpr_timestamps': microexpr_timestamps,
            'microexpr_types': microexpr_types,
        }
    
    def compute_pose_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет оптимизированные фичи позы головы"""
        features = {}
        
        for axis in ['Rx', 'Ry', 'Rz']:
            col = f'pose_{axis}'
            if col not in df.columns:
                features[f'pose_{axis}_mean'] = 0.0
                features[f'pose_{axis}_std'] = 0.0
                features[f'pose_{axis}_min'] = 0.0
                features[f'pose_{axis}_max'] = 0.0
                continue
            
            values = df[col].fillna(0.0).values
            features[f'pose_{axis}_mean'] = float(np.mean(values))
            features[f'pose_{axis}_std'] = float(np.std(values))
            features[f'pose_{axis}_min'] = float(np.min(values))
            features[f'pose_{axis}_max'] = float(np.max(values))
        
        # Pose stability score
        rx_std = features.get('pose_Rx_std', 0.0)
        ry_std = features.get('pose_Ry_std', 0.0)
        rz_std = features.get('pose_Rz_std', 0.0)
        total_std = np.sqrt(rx_std**2 + ry_std**2 + rz_std**2)
        # Нормализуем (предполагаем max std ~30 градусов)
        max_expected_std = 30.0
        pose_stability_score = float(np.clip(1.0 - (total_std / max_expected_std), 0.0, 1.0))
        features['pose_stability_score'] = pose_stability_score
        
        # Tz (приближение/удаление)
        if 'pose_Tz' in df.columns:
            tz_values = df['pose_Tz'].fillna(0.0).values
            features['pose_Tz_mean'] = float(np.mean(tz_values))
            features['pose_Tz_std'] = float(np.std(tz_values))
        else:
            features['pose_Tz_mean'] = 0.0
            features['pose_Tz_std'] = 0.0
        
        return features
    
    def compute_gaze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет фичи направления взгляда"""
        features = {}
        
        for axis in ['x', 'y']:
            col = f'gaze_angle_{axis}'
            if col not in df.columns:
                features[f'gaze_{axis}_mean'] = 0.0
                features[f'gaze_{axis}_std'] = 0.0
                continue
            
            values = df[col].fillna(0.0).values
            features[f'gaze_{axis}_mean'] = float(np.mean(values))
            features[f'gaze_{axis}_std'] = float(np.std(values))
        
        # Gaze centered ratio (взгляд в камеру)
        if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
            gaze_x = df['gaze_angle_x'].fillna(0.0).values
            gaze_y = df['gaze_angle_y'].fillna(0.0).values
            
            centered = (np.abs(gaze_x) < self.gaze_centered_threshold) & \
                       (np.abs(gaze_y) < self.gaze_centered_threshold)
            gaze_centered_ratio = float(np.mean(centered))
        else:
            gaze_centered_ratio = 0.0
        
        features['gaze_centered_ratio'] = gaze_centered_ratio
        
        # Blink rate (AU45/AU43)
        blink_rate = 0.0
        if 'AU45_c' in df.columns:
            au45_presence = df['AU45_c'].fillna(0.0).values
            # Blink: короткая вспышка presence (< 0.25s)
            blink_frames = int(0.25 * self.fps)
            blink_count = 0
            i = 0
            while i < len(au45_presence):
                if au45_presence[i] > 0.5:
                    # Начало blink
                    blink_duration = 0
                    while i < len(au45_presence) and au45_presence[i] > 0.5:
                        blink_duration += 1
                        i += 1
                    if blink_duration <= blink_frames:
                        blink_count += 1
                else:
                    i += 1
            
            duration_minutes = len(df) / (self.fps * 60.0) if len(df) > 0 else 1.0
            blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0.0
        
        features['blink_rate_per_min'] = blink_rate
        
        # Eye contact score (gaze centered + blink rate)
        eye_contact_score = (gaze_centered_ratio * 0.7) + (np.clip(blink_rate / 20.0, 0.0, 1.0) * 0.3)
        features['eye_contact_score'] = float(eye_contact_score)
        
        return features
    
    def compute_landmark_features(self, df: pd.DataFrame, fit: bool = True) -> Dict[str, Any]:
        """Вычисляет компактные геометрические признаки из landmarks"""
        features = {}
        
        # Извлекаем landmarks (2D)
        landmark_cols_2d = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
        
        if len(landmark_cols_2d) >= 68 * 2:  # 68 точек × 2 координаты
            # Mouth opening (расстояние между верхней и нижней губой, нормализованное по межглазному расстоянию)
            # Landmarks для губ: 48-67 (примерно)
            # Межглазное расстояние: между точками глаз (примерно 36 и 45)
            
            # Упрощенная версия: используем средние координаты верхней и нижней губы
            upper_lip_y = df[[f'y_{i}' for i in range(51, 54)]].mean(axis=1).values if all(f'y_{i}' in df.columns for i in range(51, 54)) else None
            lower_lip_y = df[[f'y_{i}' for i in range(57, 60)]].mean(axis=1).values if all(f'y_{i}' in df.columns for i in range(57, 60)) else None
            
            if upper_lip_y is not None and lower_lip_y is not None:
                mouth_opening = np.abs(upper_lip_y - lower_lip_y)
                # Нормализация по межглазному расстоянию (упрощенно)
                interocular_dist = 1.0  # placeholder
                if 'x_36' in df.columns and 'x_45' in df.columns:
                    interocular_dist = np.sqrt(
                        (df['x_36'] - df['x_45'])**2 + (df['y_36'] - df['y_45'])**2
                    ).mean()
                if interocular_dist > 0:
                    mouth_opening_norm = mouth_opening / interocular_dist
                else:
                    mouth_opening_norm = mouth_opening
                
                features['mouth_opening_mean'] = float(np.mean(mouth_opening_norm))
                features['mouth_opening_std'] = float(np.std(mouth_opening_norm))
            else:
                features['mouth_opening_mean'] = 0.0
                features['mouth_opening_std'] = 0.0
            
            # Smile width (расстояние между уголками губ)
            if 'x_48' in df.columns and 'x_54' in df.columns:
                smile_width = np.sqrt(
                    (df['x_48'] - df['x_54'])**2 + (df['y_48'] - df['y_54'])**2
                ).values
                features['smile_width_mean'] = float(np.mean(smile_width))
                features['smile_width_std'] = float(np.std(smile_width))
            else:
                features['smile_width_mean'] = 0.0
                features['smile_width_std'] = 0.0
            
            # Face asymmetry (корреляция L-R landmark distances)
            # Упрощенно: используем симметричные точки
            asymmetry_scores = []
            for i in range(17):  # Контур лица
                left_idx = i
                right_idx = 16 - i
                if f'x_{left_idx}' in df.columns and f'x_{right_idx}' in df.columns:
                    left_x = df[f'x_{left_idx}'].values
                    right_x = df[f'x_{right_idx}'].values
                    # Центр лица
                    center_x = (df['x_30'].values + df['x_33'].values) / 2 if 'x_30' in df.columns and 'x_33' in df.columns else df['x_30'].values
                    left_dist = np.abs(left_x - center_x)
                    right_dist = np.abs(right_x - center_x)
                    if len(left_dist) > 1 and len(right_dist) > 1:
                        corr = np.corrcoef(left_dist, right_dist)[0, 1]
                        if not np.isnan(corr):
                            asymmetry_scores.append(1.0 - abs(corr))  # 1 - correlation = asymmetry
            
            if asymmetry_scores:
                features['face_asymmetry_score'] = float(np.mean(asymmetry_scores))
            else:
                features['face_asymmetry_score'] = 0.0
            
            # PCA для landmarks (если нужно)
            landmark_matrix = []
            for i in range(68):
                if f'x_{i}' in df.columns and f'y_{i}' in df.columns:
                    landmark_matrix.append(df[f'x_{i}'].values)
                    landmark_matrix.append(df[f'y_{i}'].values)
            
            if len(landmark_matrix) > 0:
                landmark_matrix = np.array(landmark_matrix).T
                if fit:
                    lm_n, lm_f = landmark_matrix.shape
                    lm_comp = min(5, lm_n, lm_f)
                    if lm_comp >= 1:
                        self.pca_landmarks = PCA(n_components=lm_comp)
                        landmarks_pca = self.pca_landmarks.fit_transform(landmark_matrix)
                    else:
                        self.pca_landmarks = None
                        landmarks_pca = np.zeros((len(df), 5))
                else:
                    if self.pca_landmarks is not None:
                        landmarks_pca = self.pca_landmarks.transform(landmark_matrix)
                    else:
                        landmarks_pca = np.zeros((len(df), 5))
                
                for i in range(min(5, landmarks_pca.shape[1])):
                    features[f'landmarks_pca_{i+1}'] = float(np.mean(landmarks_pca[:, i]))
        else:
            features['mouth_opening_mean'] = 0.0
            features['mouth_opening_std'] = 0.0
            features['smile_width_mean'] = 0.0
            features['smile_width_std'] = 0.0
            features['face_asymmetry_score'] = 0.0
        
        # 3D landmarks features
        if 'X_30' in df.columns:  # Nose tip
            nose_z = df['Z_30'].fillna(0.0).values if 'Z_30' in df.columns else None
            if nose_z is not None:
                features['head_depth_variation'] = float(np.std(nose_z))
            else:
                features['head_depth_variation'] = 0.0
        else:
            features['head_depth_variation'] = 0.0
        
        return features
    
    def compute_per_frame_vectors(
        self,
        df: pd.DataFrame,
        au_baseline: Optional[Dict[str, float]] = None,
        au_pca_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Создает per-frame векторы для VisualTransformer (~16-24 числа).
        """
        n_frames = len(df)
        if n_frames == 0:
            return np.zeros((0, 22), dtype=float)
        
        vectors = []
        total_duration = n_frames / self.fps
        
        # Вычисляем per-frame признаки.
        # ВАЖНО: элементы вектора собираются СТРОГО в порядке COMPACT22_FEATURE_NAMES.
        # Раньше порядок append() (pose_Rz/pose_Tz шли на позициях 10-11, gaze/mouth сдвинуты)
        # НЕ совпадал с COMPACT22_FEATURE_NAMES → compact22 отдавался с перепутанными метками
        # (Encoder читал не те столбцы). Теперь собираем по имени и раскладываем по контракту.
        for idx in range(n_frames):
            row = df.iloc[idx]

            # time_norm
            time_norm = idx / max(n_frames - 1, 1)
            # face_presence_flag
            face_presence = 1.0 if row.get('success', 0) > 0.5 else 0.0
            # Key AU intensity deltas
            au_delta = {}
            for au in ['AU12', 'AU06', 'AU04', 'AU25']:
                baseline = au_baseline.get(au, 0.0) if au_baseline else 0.0
                intensity = float(row.get(f"{au}_r", 0.0))
                au_delta[au] = (intensity - baseline) / 5.0  # ~[0,1], max intensity ~5
            # AU25 presence rate in short window
            if 'AU25_c' in df.columns:
                window_start = max(0, idx - int(0.5 * self.fps))
                window_end = min(n_frames, idx + int(0.5 * self.fps))
                au25_window = df['AU25_c'].iloc[window_start:window_end]
                au25_presence_rate = float(au25_window.mean()) if len(au25_window) > 0 else 0.0
            else:
                au25_presence_rate = 0.0
            # Blink flag (AU45 presence)
            blink_flag = float(row.get('AU45_c', 0.0) > 0.5)
            # Pose normalized
            pose_ry = float(row.get('pose_Ry', 0.0)) / 90.0
            pose_rx = float(row.get('pose_Rx', 0.0)) / 90.0
            pose_rz = float(row.get('pose_Rz', 0.0)) / 90.0
            pose_tz = float(row.get('pose_Tz', 0.0)) / 100.0
            # Gaze
            gaze_x_raw = float(row.get('gaze_angle_x', 0.0))
            gaze_y_raw = float(row.get('gaze_angle_y', 0.0))
            gaze_centered = 1.0 if (abs(gaze_x_raw) < self.gaze_centered_threshold and abs(gaze_y_raw) < self.gaze_centered_threshold) else 0.0
            gaze_x = gaze_x_raw / 30.0
            gaze_y = gaze_y_raw / 30.0
            # Mouth opening normalized
            mouth_opening = 0.0
            if 'y_51' in row.index and 'y_57' in row.index:
                mouth_opening = abs(float(row['y_51']) - float(row['y_57']))
            mouth_opening = mouth_opening / 50.0
            # Face asymmetry / microexpr_recent — placeholder (per-frame не вычисляются; отдельный PR)
            face_asymmetry = 0.0
            microexpr_recent = 0.0
            # AU PCA (3)
            if au_pca_features is not None and idx < len(au_pca_features):
                au_pca = [float(x) for x in au_pca_features[idx, :3].tolist()]
            else:
                au_pca = [0.0, 0.0, 0.0]
            # AU quality flag — placeholder (отдельный PR)
            au_quality = 1.0

            # Раскладка СТРОГО по COMPACT22_FEATURE_NAMES.
            vec = [
                time_norm,               # 0  time_norm
                face_presence,           # 1  face_presence_flag
                au_delta['AU12'],        # 2  AU12_delta_norm
                au_delta['AU06'],        # 3  AU06_delta_norm
                au_delta['AU04'],        # 4  AU04_delta_norm
                au_delta['AU25'],        # 5  AU25_delta_norm
                au25_presence_rate,      # 6  AU25_presence_rate_1s
                blink_flag,              # 7  blink_flag
                pose_ry,                 # 8  pose_Ry_norm
                pose_rx,                 # 9  pose_Rx_norm
                gaze_centered,           # 10 gaze_centered_flag
                gaze_x,                  # 11 gaze_x_norm
                gaze_y,                  # 12 gaze_y_norm
                mouth_opening,           # 13 mouth_opening_norm
                face_asymmetry,          # 14 face_asymmetry_score
                microexpr_recent,        # 15 microexpr_recent_count
                au_pca[0],               # 16 au_pca_0
                au_pca[1],               # 17 au_pca_1
                au_pca[2],               # 18 au_pca_2
                au_quality,              # 19 au_quality_flag
                pose_rz,                 # 20 pose_Rz_norm
                pose_tz,                 # 21 pose_Tz_norm
            ]

            # Enforce strict compact22 contract.
            if len(vec) != 22:
                raise RuntimeError(f"MicroEmotionProcessor | compact22 vector length mismatch: got={len(vec)} expected=22")
            vectors.append(vec)
        
        return np.array(vectors, dtype=float)
    
    def compute_video_level_aggregates(
        self,
        df: pd.DataFrame,
        key_au_features: Dict[str, Any],
        microexpr_features: Dict[str, Any],
        pose_features: Dict[str, Any],
        gaze_features: Dict[str, Any],
        landmark_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Вычисляет видео-уровневые агрегаты"""
        aggregates = {}
        
        # Key AU aggregates
        for au in ['AU06', 'AU12', 'AU04', 'AU25', 'AU26', 'AU7', 'AU15']:
            au_name = au.replace('AU7', 'AU07')
            if au_name in key_au_features:
                au_data = key_au_features[au_name]
                aggregates[f'{au_name}_mean'] = au_data['intensity_mean']
                aggregates[f'{au_name}_std'] = au_data['intensity_std']
                aggregates[f'{au_name}_min'] = 0.0  # Placeholder
                aggregates[f'{au_name}_max'] = 0.0  # Placeholder
                aggregates[f'{au_name}_median'] = 0.0  # Placeholder
                aggregates[f'{au_name}_peak_count'] = au_data['peak_count']
        
        # Micro-expressions
        aggregates.update(microexpr_features)
        
        # Smile ratio
        if 'AU12_r' in df.columns and 'AU06_r' in df.columns:
            smile_threshold = 1.0
            smile_frames = (df['AU12_r'].fillna(0.0) + df['AU06_r'].fillna(0.0)) > smile_threshold
            aggregates['smile_ratio'] = float(smile_frames.mean())
        else:
            aggregates['smile_ratio'] = 0.0
        
        # Eye contact ratio
        aggregates['eye_contact_ratio'] = gaze_features.get('gaze_centered_ratio', 0.0)
        
        # Blink rate
        aggregates['blink_rate_per_min'] = gaze_features.get('blink_rate_per_min', 0.0)
        
        # Pose stability
        aggregates['pose_stability_score'] = pose_features.get('pose_stability_score', 0.0)
        
        # Face presence ratio
        if 'success' in df.columns:
            aggregates['face_presence_ratio'] = float(df['success'].mean())
        else:
            aggregates['face_presence_ratio'] = 0.0
        
        # Landmark features
        aggregates['avg_mouth_opening'] = landmark_features.get('mouth_opening_mean', 0.0)
        
        # AU PCA variance explained
        if self.pca_au is not None:
            for i, var_exp in enumerate(self.pca_au.explained_variance_ratio_[:5]):
                aggregates[f'au_pca_var_explained_{i+1}'] = float(var_exp)
        
        return aggregates
    
    def compute_reliability_flags(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Вычисляет флаги надёжности"""
        flags = {}
        
        # AU quality score
        au_confidence_cols = [col for col in df.columns if col.endswith('_c')]
        if len(au_confidence_cols) > 0:
            au_quality_scores = []
            for col in au_confidence_cols:
                quality = df[col].fillna(0.0).values
                au_quality_scores.extend(quality)
            au_quality_overall = float(np.mean(au_quality_scores)) if au_quality_scores else 0.0
        else:
            au_quality_overall = 0.0
        
        flags['au_quality_overall'] = au_quality_overall
        flags['au_quality_reliable'] = au_quality_overall > self.au_confidence_threshold
        
        # Landmark visibility
        landmark_cols = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
        if len(landmark_cols) > 0:
            # Простая проверка: считаем видимыми landmarks с ненулевыми координатами
            visible_count = 0
            total_count = 0
            for col in landmark_cols[:68*2]:  # 68 landmarks × 2
                values = df[col].fillna(0.0).values
                visible = np.sum(values != 0.0)
                visible_count += visible
                total_count += len(values)
            landmark_visibility_mean = visible_count / total_count if total_count > 0 else 0.0
        else:
            landmark_visibility_mean = 0.0
        
        flags['landmark_visibility_mean'] = landmark_visibility_mean
        flags['landmark_visibility_reliable'] = landmark_visibility_mean > 0.8
        
        # Occlusion flag
        flags['occlusion_flag'] = landmark_visibility_mean < 0.7
        
        # Lighting flag (упрощенно)
        flags['lighting_flag'] = False  # Placeholder
        
        return flags
    
    def process_openface_dataframe(
        self,
        df: pd.DataFrame,
        fit_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Главный метод обработки DataFrame OpenFace.
        Возвращает оптимизированные фичи.
        """
        if len(df) == 0:
            return {
                'success': False,
                'features': {},
                'per_frame_vectors': np.zeros((0, 22)),
                'reliability_flags': {},
            }
        
        # Вычисляем baseline для AU
        au_columns = [col.replace('_r', '') for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        au_baseline = self.compute_au_baseline(df, au_columns) if fit_models else {}
        
        # Извлекаем ключевые AU фичи
        key_au_features = self.extract_key_au_features(df, au_baseline)
        
        # PCA для остальных AU
        au_pca_features, pca_model = self.compute_au_pca(df, fit=fit_models)
        
        # Детекция micro-expressions
        microexpr_features = self.detect_micro_expressions(df, au_baseline)
        
        # Pose features
        pose_features = self.compute_pose_features(df)
        
        # Gaze features
        gaze_features = self.compute_gaze_features(df)
        
        # Landmark features
        landmark_features = self.compute_landmark_features(df, fit=fit_models)
        
        # Per-frame vectors
        per_frame_vectors = self.compute_per_frame_vectors(df, au_baseline, au_pca_features)
        
        # Видео-уровневые агрегаты
        video_aggregates = self.compute_video_level_aggregates(
            df, key_au_features, microexpr_features, pose_features, gaze_features, landmark_features
        )
        
        # Reliability flags
        reliability_flags = self.compute_reliability_flags(df)
        
        # Объединяем все фичи
        features = {
            **key_au_features,
            **pose_features,
            **gaze_features,
            **landmark_features,
            **video_aggregates,
            **reliability_flags,
        }
        
        return {
            'success': True,
            'features': features,
            'per_frame_vectors': per_frame_vectors,
            'reliability_flags': reliability_flags,
            'microexpr_features': microexpr_features,
            'au_baseline': au_baseline,
            'pca_models': {
                'au_pca': pca_model,
                'landmarks_pca': self.pca_landmarks,
            },
        }


class MicroEmotionModule(BaseModule):
    """
    Модуль для извлечения micro-emotion фичей из данных OpenFace.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует MicroEmotionProcessor для обработки DataFrame OpenFace.
    
    Может работать с:
    - Готовым DataFrame (переданным через config)
    - CSV файлом OpenFace (загружается автоматически)
        - face presence из core_face_landmarks для фильтрации кадров (face_detection удалён)
    """
    
    MODULE_NAME = MODULE_NAME
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    def __init__(
        self,
        rs_path: Optional[str] = None,
        fps: int = 30,
        microexpr_smoothing_sigma: float = 0.05,
        microexpr_delta_threshold: float = 0.4,
        microexpr_max_duration_frames: int = 15,
        microexpr_min_peak_distance_frames: int = 6,
        gaze_centered_threshold: float = 10.0,
        pca_components: int = 3,
        au_confidence_threshold: float = 0.5,
        use_face_detection: bool = False,
        # OpenFace runtime
        docker_image: str = "openface/openface:latest",
        openface_batch_size: int = 64,
        device: str = "cuda",
        # gating/progress
        feature_groups: str = "default",
        progress_every_frames: int = 50,
        **kwargs: Any
    ):
        """
        Инициализация MicroEmotionModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            fps: Кадров в секунду
            microexpr_smoothing_sigma: Сглаживание для micro-expressions (в секундах)
            microexpr_delta_threshold: Порог изменения интенсивности для micro-expression
            microexpr_max_duration_frames: Максимальная длительность micro-expression в кадрах
            microexpr_min_peak_distance_frames: Минимальное расстояние между пиками
            gaze_centered_threshold: Порог для определения взгляда в камеру (градусы)
            pca_components: Количество PCA компонент для AU
            au_confidence_threshold: Порог уверенности AU
            use_face_detection: Устаревший флаг. Если True — фильтруем кадры по `core_face_landmarks.face_present`
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        self.fps = fps
        self.use_face_detection = use_face_detection
        self.docker_image = str(docker_image)
        self.openface_batch_size = max(1, int(openface_batch_size))
        self.device = str(device or "cuda")
        if self.device != "cuda":
            raise RuntimeError("micro_emotion | policy: only cuda is allowed for OpenFace runtime")
        self.feature_groups = str(feature_groups or "default")
        self.progress_every_frames = max(1, int(progress_every_frames))
        
        # Инициализируем процессор
        self.processor = MicroEmotionProcessor(
            fps=fps,
            microexpr_smoothing_sigma=microexpr_smoothing_sigma,
            microexpr_delta_threshold=microexpr_delta_threshold,
            microexpr_max_duration_frames=microexpr_max_duration_frames,
            microexpr_min_peak_distance_frames=microexpr_min_peak_distance_frames,
            gaze_centered_threshold=gaze_centered_threshold,
            pca_components=pca_components,
            au_confidence_threshold=au_confidence_threshold,
        )
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Опциональные зависимости:
        - core_face_landmarks: для фильтрации кадров по face_present (если включён use_face_detection)
        """
        # Per owner decision: this component works only with faces and relies on core_face_landmarks.
        return ["core_face_landmarks"]

    def _load_frames_metadata(self, frames_dir: str) -> Dict[str, Any]:
        meta_path = os.path.join(frames_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"micro_emotion | metadata.json not found in {frames_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_core_face_landmarks(self) -> Dict[str, Any]:
        core = self.load_core_provider("core_face_landmarks", file_name=None)
        if not isinstance(core, dict):
            raise RuntimeError("micro_emotion | failed to load core_face_landmarks")
        return core

    def _face_present_any_aligned(
        self, *, core: Dict[str, Any], want_frame_indices: np.ndarray
    ) -> np.ndarray:
        fi = core.get("frame_indices")
        fp = core.get("face_present")
        if fi is None or fp is None:
            raise RuntimeError("micro_emotion | core_face_landmarks missing frame_indices/face_present")
        fi = np.asarray(fi, dtype=np.int32).reshape(-1)
        fp = np.asarray(fp, dtype=bool)
        if fp.shape[0] != fi.shape[0]:
            raise RuntimeError("micro_emotion | core_face_landmarks face_present shape mismatch")
        # Приводим к виду (N,) — any по осям лиц
        if fp.ndim == 1:
            face_any = fp.astype(bool)
        else:
            face_any = np.any(fp, axis=1).astype(bool)

        # Базовая инварианта: core_face_landmarks не должен иметь дубликатов frame_indices
        unique_fi, uniq_idx = np.unique(fi, return_index=True)
        if unique_fi.size != fi.size:
            # если есть дубликаты, берём любой (первый) и логируем предупреждение
            LOGGER.warning("micro_emotion | core_face_landmarks.frame_indices contains duplicates; using first occurrence per index")
            fi = unique_fi
            face_any = face_any[uniq_idx]

        # Строим мапу frame_index -> face_present_any
        fi_to_face: Dict[int, bool] = {int(idx): bool(v) for idx, v in zip(fi.tolist(), face_any.tolist())}

        # Возвращаем булев массив длины want_frame_indices; если индекс отсутствует у core_face_landmarks,
        # считаем, что лица нет (False) — это безопаснее, чем падать при несовпадении выборок.
        out = np.zeros_like(want_frame_indices, dtype=bool)
        for i, idx in enumerate(want_frame_indices.tolist()):
            out[i] = bool(fi_to_face.get(int(idx), False))
        return out

    def _build_ui_payload(
        self,
        *,
        frame_indices: np.ndarray,
        times_s: np.ndarray,
        face_present_any: np.ndarray,
        openface_df: Optional[pd.DataFrame],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": "micro_emotion_ui_v1",
            "component": MODULE_NAME,
            "frame_indices": frame_indices.tolist(),
            "times_s": times_s.tolist(),
            "face_present_any": face_present_any.astype(bool).tolist(),
            "summary": results.get("summary", {}) or {},
            "microexpr_features": results.get("microexpr_features", {}) or {},
        }
        # OpenFace 68 landmarks for backend UI (only here, not in model-facing arrays).
        if openface_df is not None:
            of68 = _pack_openface_68_landmarks_for_ui(openface_df)
            if of68 is not None:
                payload["openface_landmarks68"] = of68
        return payload

    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Override BaseModule.run() to allow meta.ui_payload (dict) while preserving baseline identity checks.
        """
        if self.rs_path is None:
            raise RuntimeError("micro_emotion | rs_path is required")

        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise RuntimeError("micro_emotion | frame_indices missing/empty (no-fallback)")

        resource_profile_before = _resource_profile_snapshot()

        fm = None
        try:
            fm = self.create_frame_manager(frames_dir, metadata)

            # Stage timings
            t0 = time.perf_counter()

            # Load upstream core_face_landmarks
            core = self._load_core_face_landmarks()

            t_deps = time.perf_counter()

            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            times_s = _times_s_from_union(metadata=metadata, frame_indices=fi_np)
            face_present_any = self._face_present_any_aligned(core=core, want_frame_indices=fi_np)

            if not bool(np.any(face_present_any)):
                # Valid empty (no faces in video)
                empty_results = {
                    "frame_indices": fi_np,
                    "times_s": times_s,
                    "face_present_any": face_present_any,
                    "frame_feature_names": np.asarray([], dtype=object),
                    "frame_features": np.full((int(fi_np.size), 0), np.nan, dtype=np.float32),
                    "compact22": np.full((int(fi_np.size), 22), np.nan, dtype=np.float32),
                    "compact22_feature_names": np.asarray(COMPACT22_FEATURE_NAMES, dtype=object),
                    "event_times_s": np.asarray([], dtype=np.float32),
                    "event_type_id": np.asarray([], dtype=np.int16),
                    "event_strength": np.asarray([], dtype=np.float32),
                    "feature_names": np.asarray(_FEATURE_NAMES_V1, dtype=object),
                    "feature_values": np.full((len(_FEATURE_NAMES_V1),), np.nan, dtype=np.float32),
                    # micro_emotion_npz_v3 requires this key; omitting it fails validate_npz (allow_extra_keys=false).
                    "microexpr_features": {},
                    "summary": {
                        "success": False,
                        "total_frames": int(fi_np.size),
                        "frames_processed_openface": 0,
                        "fps": int(self.fps),
                        "stage_timings_ms": {
                            "deps_load_ms": float((t_deps - t0) * 1000.0),
                        },
                    },
                }
                # Fill minimal scalar features for empty.
                try:
                    name_to_idx = {str(n): i for i, n in enumerate(empty_results["feature_names"].tolist())}
                    fv = empty_results["feature_values"]
                    fv[name_to_idx["has_faces"]] = 0.0
                    fv[name_to_idx["frames_n"]] = float(fi_np.size)
                    fv[name_to_idx["frames_with_face"]] = 0.0
                    fv[name_to_idx["frames_processed_openface"]] = 0.0
                    fv[name_to_idx["face_presence_ratio"]] = 0.0
                except Exception:
                    pass
                ui_payload = self._build_ui_payload(
                    frame_indices=fi_np,
                    times_s=times_s,
                    face_present_any=face_present_any,
                    openface_df=None,
                    results=empty_results,
                )
                save_metadata = {
                    "total_frames": metadata.get("total_frames"),
                    "processed_frames": int(fi_np.size),
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
                    "status": "empty",
                    "empty_reason": "no_faces_in_video",
                    "ui_payload": ui_payload,
                }
                if isinstance(resource_profile_before, dict) and resource_profile_before:
                    save_metadata["resource_profile_before"] = dict(resource_profile_before)
                save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
                # Audit v3: stage timings must be present in meta.
                try:
                    st = empty_results.get("summary", {}).get("stage_timings_ms", {})
                    save_metadata["stage_timings_ms"] = st if isinstance(st, dict) else {}
                except Exception:
                    save_metadata["stage_timings_ms"] = {}

                # Reproducibility: config highlights
                save_metadata["module_sampling_policy_version"] = "segmenter_axis_v1"
                save_metadata["face_frames_sampling_policy_version"] = "core_face_landmarks_face_present_v1"
                save_metadata["docker_image"] = str(self.docker_image)
                save_metadata["openface_batch_size"] = int(self.openface_batch_size)
                save_metadata["device"] = str(self.device or "cuda")
                save_metadata["feature_groups"] = str(self.feature_groups or "default")
                save_metadata["fps"] = int(self.fps)
                save_metadata["microexpr_smoothing_sigma"] = float(getattr(self.processor, "microexpr_smoothing_sigma", 0.0))
                save_metadata["microexpr_delta_threshold"] = float(getattr(self.processor, "microexpr_delta_threshold", 0.0))
                save_metadata["microexpr_max_duration_frames"] = int(getattr(self.processor, "microexpr_max_duration_frames", 0))
                save_metadata["microexpr_min_peak_distance_frames"] = int(getattr(self.processor, "microexpr_min_peak_distance_frames", 0))
                save_metadata["gaze_centered_threshold"] = float(getattr(self.processor, "gaze_centered_threshold", 0.0))
                save_metadata["pca_components"] = int(getattr(self.processor, "pca_components", 0))
                save_metadata["au_confidence_threshold"] = float(getattr(self.processor, "au_confidence_threshold", 0.0))
                return self.save_results(results=empty_results, metadata=save_metadata, use_compressed=False)

            # Run process() (this will run OpenFace and compute outputs)
            results = self.process(frame_manager=fm, frame_indices=frame_indices, config={**(config or {}), "_frames_dir": frames_dir, "_metadata": metadata})

            t_proc = time.perf_counter()

            # UI payload: include OpenFace 68 landmarks only here
            openface_df = results.pop("_openface_df_for_ui", None)
            ui_payload = self._build_ui_payload(
                frame_indices=fi_np,
                times_s=times_s,
                face_present_any=face_present_any,
                openface_df=openface_df if isinstance(openface_df, pd.DataFrame) else None,
                results=results,
            )

            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": int(fi_np.size),
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
            }
            if isinstance(resource_profile_before, dict) and resource_profile_before:
                save_metadata["resource_profile_before"] = dict(resource_profile_before)

            # Attach stage timings into summary (baseline profiling requirement)
            try:
                summary = results.get("summary") if isinstance(results, dict) else None
                if isinstance(summary, dict):
                    st = summary.get("stage_timings_ms") if isinstance(summary.get("stage_timings_ms"), dict) else {}
                    st["deps_load_ms"] = float((t_deps - t0) * 1000.0)
                    st["process_ms"] = float((t_proc - t_deps) * 1000.0)
                    summary["stage_timings_ms"] = st
                    results["summary"] = summary
            except Exception:
                pass

            save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
            # Audit v3: stage timings must also be present in NPZ meta for render/QA.
            try:
                st = results.get("summary", {}).get("stage_timings_ms", {})
                save_metadata["stage_timings_ms"] = st if isinstance(st, dict) else {}
            except Exception:
                save_metadata["stage_timings_ms"] = {}

            # Reproducibility: config highlights + sampling policy
            save_metadata["module_sampling_policy_version"] = "segmenter_axis_v1"
            save_metadata["face_frames_sampling_policy_version"] = "core_face_landmarks_face_present_v1"
            save_metadata["docker_image"] = str(self.docker_image)
            save_metadata["openface_batch_size"] = int(self.openface_batch_size)
            save_metadata["device"] = str(self.device or "cuda")
            save_metadata["feature_groups"] = str(self.feature_groups or "default")
            save_metadata["fps"] = int(self.fps)
            save_metadata["microexpr_smoothing_sigma"] = float(getattr(self.processor, "microexpr_smoothing_sigma", 0.0))
            save_metadata["microexpr_delta_threshold"] = float(getattr(self.processor, "microexpr_delta_threshold", 0.0))
            save_metadata["microexpr_max_duration_frames"] = int(getattr(self.processor, "microexpr_max_duration_frames", 0))
            save_metadata["microexpr_min_peak_distance_frames"] = int(getattr(self.processor, "microexpr_min_peak_distance_frames", 0))
            save_metadata["gaze_centered_threshold"] = float(getattr(self.processor, "gaze_centered_threshold", 0.0))
            save_metadata["pca_components"] = int(getattr(self.processor, "pca_components", 0))
            save_metadata["au_confidence_threshold"] = float(getattr(self.processor, "au_confidence_threshold", 0.0))
            return self.save_results(results=results, metadata=save_metadata, use_compressed=False)
        finally:
            if fm is not None:
                try:
                    fm.close()
                except Exception:
                    pass
    
    def _load_openface_dataframe(
        self,
        config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Загружает DataFrame OpenFace из различных источников.
        
        Приоритет:
        1. DataFrame переданный напрямую в config['openface_dataframe']
        2. Путь к CSV в config['openface_csv_path']
        3. Автоматический поиск CSV в rs_path/micro_emotion/
        4. Загрузка из результатов других модулей (если есть)
        
        Args:
            config: Конфигурация модуля
            
        Returns:
            DataFrame OpenFace или None, если не найден
        """
        # OpenFace CSV headers carry leading spaces (", AU01_r", ", pose_Rx", ", success", ", x_0").
        # pandas keeps them unless skipinitialspace=True, so bare-name lookups miss and every
        # OpenFace-derived feature collapses to zero/NaN. Normalize column names on every load path.
        def _norm_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if isinstance(df, pd.DataFrame):
                df.columns = [str(c).strip() for c in df.columns]
            return df

        # 1. Прямая передача DataFrame
        if 'openface_dataframe' in config and config['openface_dataframe'] is not None:
            df = config['openface_dataframe']
            if isinstance(df, pd.DataFrame):
                self.logger.info("Используется переданный DataFrame OpenFace")
                return _norm_cols(df)

        # 2. Путь к CSV в config
        csv_path = config.get('openface_csv_path')
        if csv_path and os.path.exists(csv_path):
            self.logger.info(f"Загружаем CSV OpenFace из {csv_path}")
            try:
                return _norm_cols(pd.read_csv(csv_path))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки CSV {csv_path}: {e}")

        # 3. Автоматический поиск CSV в rs_path
        if self.rs_path:
            micro_emotion_dir = os.path.join(self.rs_path, "micro_emotion")
            if os.path.exists(micro_emotion_dir):
                csv_files = list(Path(micro_emotion_dir).glob("*.csv"))
                if csv_files:
                    # Берем последний по времени модификации
                    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
                    self.logger.info(f"Найден CSV OpenFace: {csv_path}")
                    try:
                        return _norm_cols(pd.read_csv(str(csv_path)))
                    except Exception as e:
                        self.logger.warning(f"Ошибка загрузки CSV {csv_path}: {e}")
        
        # 4. Попытка загрузить из результатов других модулей
        # (если есть сохраненный DataFrame в npz)
        if self.rs_path:
            try:
                deps = self.load_all_dependencies()
                # Проверяем, есть ли сохраненные данные OpenFace
                for module_name, data in deps.items():
                    if data and isinstance(data, dict):
                        if 'openface_dataframe' in data:
                            df = data['openface_dataframe']
                            if isinstance(df, pd.DataFrame):
                                self.logger.info(f"Загружен DataFrame из {module_name}")
                                return df
            except Exception:
                pass
        
        return None
    
    def _filter_frame_indices_by_face_presence(
        self,
        frame_indices: List[int]
    ) -> List[int]:
        """
        Фильтрует индексы кадров по `core_face_landmarks.face_present`.
        
        Args:
            frame_indices: Исходный список индексов кадров
            
        Returns:
            Отфильтрованный список индексов кадров с лицами
        """
        if not self.use_face_detection:
            return frame_indices
        
        try:
            core = self.load_core_provider("core_face_landmarks", file_name=None)
            if not core or "frame_indices" not in core or "face_present" not in core:
                self.logger.warning("core_face_landmarks missing; cannot filter frames, using all frames")
                return frame_indices

            fi = np.asarray(core["frame_indices"], dtype=np.int32)
            fp = np.asarray(core["face_present"], dtype=bool)
            if fi.shape[0] != fp.shape[0]:
                self.logger.warning("core_face_landmarks shape mismatch; cannot filter frames, using all frames")
                return frame_indices

            frames_with_faces = set(int(x) for x in fi[fp].tolist())
            filtered = sorted(set(int(x) for x in frame_indices) & frames_with_faces)
            self.logger.info(
                f"Отфильтровано кадров: {len(frame_indices)} -> {len(filtered)} "
                f"(с лицами: {len(frames_with_faces)})"
            )
            return filtered
            
        except Exception as e:
            self.logger.warning(
                f"Ошибка фильтрации по core_face_landmarks: {e}. "
                "Используем все кадры."
            )
            return frame_indices
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля:
                - openface_dataframe: DataFrame OpenFace (опционально)
                - openface_csv_path: Путь к CSV OpenFace (опционально)
                - fit_models: Флаг для обучения PCA моделей (по умолчанию True)
                
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - features: словарь с агрегированными фичами
            - per_frame_vectors: numpy массив [N_frames, 22] для VisualTransformer
            - reliability_flags: флаги надёжности
            - microexpr_features: фичи micro-expressions
            - summary: метаданные обработки
        """
        # NOTE: this component is baseline-ready and runs OpenFace itself.
        # It still supports passing precomputed OpenFace DataFrame/CSV via config for debugging,
        # but production path runs docker OpenFace on face-present frames.

        frames_dir = config.get("_frames_dir") if isinstance(config, dict) else None
        metadata = config.get("_metadata") if isinstance(config, dict) else None
        if not isinstance(frames_dir, str) or not frames_dir:
            # best-effort fallback: try to read frames_dir from FrameManager
            frames_dir = getattr(frame_manager, "frames_dir", None)
        if not isinstance(metadata, dict):
            # best-effort load metadata.json
            try:
                metadata = self._load_frames_metadata(str(frames_dir))
            except Exception:
                metadata = {}

        fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        times_s = _times_s_from_union(metadata=metadata, frame_indices=fi_np)

        core = self._load_core_face_landmarks()
        face_present_any = self._face_present_any_aligned(core=core, want_frame_indices=fi_np)

        # Positions to run OpenFace on (union indices)
        face_pos = np.where(face_present_any)[0].astype(np.int32)
        face_union_indices = [int(fi_np[p]) for p in face_pos.tolist()]

        # Emit progress: treat non-face frames as "already done"
        total = int(fi_np.size)
        done_base = int(total - len(face_union_indices))
        _emit_progress(
            rs_path=str(self.rs_path),
            platform_id=str(metadata.get("platform_id") or ""),
            video_id=str(metadata.get("video_id") or ""),
            run_id=str(metadata.get("run_id") or ""),
            done=done_base,
            total=total,
            stage="filter_faces",
        )

        # If caller provided OpenFace df/csv explicitly, prefer it (debug).
        df = self._load_openface_dataframe(config)
        openface_df_for_ui: Optional[pd.DataFrame] = None

        stage_timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        if df is None or len(df) == 0:
            # Run docker OpenFace in batches (GPU-only)
            analyzer = OpenFaceAnalyzer(docker_image=self.docker_image, use_gpu=True)
            dfs: List[pd.DataFrame] = []
            processed_face = 0
            for start in range(0, len(face_union_indices), int(self.openface_batch_size)):
                batch = face_union_indices[start : start + int(self.openface_batch_size)]
                frames_bgr: List[np.ndarray] = []
                for uidx in batch:
                    frame_rgb = frame_manager.get(int(uidx))
                    # OpenFace expects images; color-space isn't critical after JPEG, but we keep BGR for cv2.
                    frames_bgr.append(np.ascontiguousarray(frame_rgb[:, :, ::-1]))
                t_run0 = time.perf_counter()
                res = analyzer.analyze_frames(
                    frames_bgr=frames_bgr,
                    union_frame_indices=batch,
                    output_prefix=f"batch_{start}",
                    keep_tmp=False,
                )
                stage_timings.setdefault("openface_run_ms", 0.0)
                stage_timings["openface_run_ms"] += float((time.perf_counter() - t_run0) * 1000.0)
                dfs.append(res.dataframe)

                processed_face += len(batch)
                # progress
                _emit_progress(
                    rs_path=str(self.rs_path),
                    platform_id=str(metadata.get("platform_id") or ""),
                    video_id=str(metadata.get("video_id") or ""),
                    run_id=str(metadata.get("run_id") or ""),
                    done=int(done_base + processed_face),
                    total=total,
                    stage="openface",
                )

            if not dfs:
                raise RuntimeError("micro_emotion | OpenFace produced no results (unexpected; should have faces)")
            df = pd.concat(dfs, ignore_index=True)
            openface_df_for_ui = df

        stage_timings["total_openface_plus_load_ms"] = float((time.perf_counter() - t0) * 1000.0)

        # Ensure mapping column exists and is valid (no partial failures allowed)
        if "frame_union" not in df.columns:
            raise RuntimeError("micro_emotion | OpenFace DataFrame missing frame_union mapping")
        mapped = df["frame_union"].astype(int).values
        # Раньше любые отрицательные значения считались фатальной ошибкой (partial failure).
        # На практике OpenFace иногда возвращает отдельные битые строки, поэтому:
        #   - выбрасываем строки с frame_union < 0,
        #   - логируем предупреждение, но продолжаем работать на валидных данных.
        if np.any(mapped < 0):
            bad_rows = int(np.sum(mapped < 0))
            LOGGER.warning(
                "micro_emotion | OpenFace mapping contains %d invalid rows (frame_union < 0); "
                "dropping these rows and continuing best-effort",
                bad_rows,
            )
            df = df[mapped >= 0].reset_index(drop=True)
            mapped = df["frame_union"].astype(int).values

        if mapped.size == 0:
            raise RuntimeError("micro_emotion | OpenFace produced no valid rows after filtering invalid mapping")

        # Раньше требовалось строгое покрытие всех face_union_indices (got == want).
        # В реальных данных OpenFace может не вернуть детекцию для части кадров с лицом.
        # Теперь:
        #   - требуем ненулевое пересечение с face_union_indices,
        #   - если частичное покрытие, логируем warning, но продолжаем.
        got = set(int(x) for x in mapped.tolist())
        want = set(int(x) for x in face_union_indices)
        intersection = got & want
        if not intersection:
            raise RuntimeError(
                "micro_emotion | OpenFace mapping has no overlap with expected face_union_indices "
                f"(got={sorted(list(got))[:10]}, want_sample={sorted(list(want))[:10]})"
            )
        if got != want:
            missing = sorted(list(want - got))[:10]
            extra = sorted(list(got - want))[:10]
            LOGGER.warning(
                "micro_emotion | partial OpenFace results: missing=%s extra=%s (continuing best-effort)",
                missing,
                extra,
            )

        # Prepare df_filtered for MicroEmotionProcessor (use union-domain frame index in `frame`)
        df_filtered = df.copy()
        df_filtered["frame"] = df_filtered["frame_union"].astype(int)

        # Process via MicroEmotionProcessor (fit_models always true for now; no persistent PCA)
        t_proc0 = time.perf_counter()
        processed = self.processor.process_openface_dataframe(df_filtered, fit_models=True)
        stage_timings["micro_emotion_features_ms"] = float((time.perf_counter() - t_proc0) * 1000.0)

        if not processed.get('success', False):
            raise RuntimeError("micro_emotion | processing OpenFace DataFrame failed")
        
        # Build model-facing dense time-series aligned to primary frame_indices:
        # - frame_features (N,F) with NaN for frames without faces
        # - frame_feature_names
        #
        # Default is wide (~40-80). Additionally we store compact22 as separate array.

        per_frame_vectors = processed.get("per_frame_vectors")
        if isinstance(per_frame_vectors, list):
            per_frame_vectors = np.asarray(per_frame_vectors, dtype=np.float32)
        elif not isinstance(per_frame_vectors, np.ndarray):
            per_frame_vectors = np.asarray(per_frame_vectors, dtype=np.float32)

        # Create aligned compact22 features: (N, 22) fixed contract.
        if per_frame_vectors.ndim != 2 or int(per_frame_vectors.shape[1]) != 22:
            raise RuntimeError(
                f"micro_emotion | per_frame_vectors must have shape (M,22) (contract). Got: {per_frame_vectors.shape}"
            )
        compact = np.full((int(fi_np.size), 22), np.nan, dtype=np.float32)
        # Map DataFrame rows (frame union indices) to positions
        pos_by_union = {int(u): i for i, u in enumerate(fi_np.tolist())}
        # MicroEmotionProcessor outputs vectors per row order of df_filtered.
        # Ensure df_filtered is sorted by frame (union index) for deterministic mapping
        df_sorted = df_filtered.sort_values("frame").reset_index(drop=True)
        vec_sorted = per_frame_vectors
        if vec_sorted.shape[0] != len(df_sorted):
            raise RuntimeError("micro_emotion | per_frame_vectors length mismatch with DataFrame (policy: error)")
        for i_row in range(len(df_sorted)):
            u = int(df_sorted.loc[i_row, "frame"])
            pos = pos_by_union.get(u, None)
            if pos is None:
                continue
            compact[pos] = vec_sorted[i_row]

        # Wide frame_features: include key AU deltas + pose/gaze + geometry + PCA components when available.
        # We pull raw columns from df_sorted (OpenFace) and align.
        wide_names: List[str] = []
        wide_cols: List[np.ndarray] = []

        def add_wide(name: str, values_by_pos: np.ndarray) -> None:
            wide_names.append(name)
            wide_cols.append(values_by_pos.astype(np.float32))

        # time_norm and face_present
        time_norm = (times_s - float(times_s[0])) / max(1e-6, float(times_s[-1] - times_s[0]))
        add_wide("time_norm", time_norm.astype(np.float32))
        add_wide("face_present_any", face_present_any.astype(np.float32))

        # Raw AU intensity deltas for key AUs (baseline subtraction already done inside processor features,
        # but for per-frame we approximate by using raw AU intensities and subtract baseline computed there if available).
        au_baseline = processed.get("au_baseline") if isinstance(processed.get("au_baseline"), dict) else {}
        for au in KEY_AUS:
            col = f"{au}_r"
            if col in df_sorted.columns:
                v = df_sorted[col].fillna(np.nan).to_numpy(dtype=np.float32)
                baseline = float(au_baseline.get(au, 0.0)) if isinstance(au_baseline, dict) else 0.0
                v = v - baseline
                aligned = np.full((int(fi_np.size),), np.nan, dtype=np.float32)
                for i_row in range(len(df_sorted)):
                    u = int(df_sorted.loc[i_row, "frame"])
                    pos = pos_by_union.get(u, None)
                    if pos is not None:
                        aligned[pos] = v[i_row]
                add_wide(f"{au}_delta", aligned)

        # Pose/gaze
        for name in ["pose_Rx", "pose_Ry", "pose_Rz", "pose_Tz", "gaze_angle_x", "gaze_angle_y"]:
            if name in df_sorted.columns:
                v = df_sorted[name].fillna(np.nan).to_numpy(dtype=np.float32)
                aligned = np.full((int(fi_np.size),), np.nan, dtype=np.float32)
                for i_row in range(len(df_sorted)):
                    u = int(df_sorted.loc[i_row, "frame"])
                    pos = pos_by_union.get(u, None)
                    if pos is not None:
                        aligned[pos] = v[i_row]
                add_wide(name, aligned)

        frame_feature_names = np.asarray(wide_names, dtype=object)
        frame_features = np.stack(wide_cols, axis=1) if wide_cols else np.full((int(fi_np.size), 0), np.nan, dtype=np.float32)

        # Events stream for micro-expressions (use processor output microexpr_features)
        microexpr_features = processed.get("microexpr_features", {}) or {}
        ev_times = np.asarray(microexpr_features.get("microexpr_timestamps", []), dtype=np.float32).reshape(-1)
        ev_types = microexpr_features.get("microexpr_types", []) or []
        # Encode type ids deterministically
        type_map = {"smile": 1, "surprise": 2, "frown": 3, "disgust": 4}
        ev_type_id = np.asarray([type_map.get(str(t), 0) for t in ev_types], dtype=np.int16).reshape(-1)
        # Strength is optional
        ev_strength = np.asarray(microexpr_features.get("microexpr_intensities", []), dtype=np.float32).reshape(-1)
        if ev_strength.size != ev_times.size:
            ev_strength = np.ones_like(ev_times, dtype=np.float32)

        # Aggregate scalar features from processor
        features = processed.get("features", {}) or {}
        features_clean: Dict[str, Any] = {}
        for k, v in features.items():
            if isinstance(v, (int, float, bool)):
                features_clean[k] = float(v) if isinstance(v, bool) else v
            elif isinstance(v, (list, tuple)):
                try:
                    features_clean[k] = np.asarray(v, dtype=np.float32)
                except Exception:
                    features_clean[k] = np.asarray(v, dtype=object)
            elif isinstance(v, np.ndarray):
                features_clean[k] = v
            else:
                # avoid huge objects in model-facing; keep only scalars/arrays here
                pass

        # Summary
        summary = {
            "success": True,
            "total_frames": int(fi_np.size),
            "frames_with_face": int(np.sum(face_present_any)),
            "frames_processed_openface": int(len(df_sorted)),
            "fps": int(self.fps),
            "stage_timings_ms": stage_timings,
        }

        # Model-facing scalar features: fixed tabular list.
        scalar_overrides: Dict[str, float] = {
            "has_faces": 1.0,
            "frames_n": float(fi_np.size),
            "frames_with_face": float(np.sum(face_present_any)),
            "frames_processed_openface": float(len(df_sorted)),
        }
        tab_values: List[float] = []
        for name in _FEATURE_NAMES_V1:
            if name in scalar_overrides:
                tab_values.append(float(scalar_overrides[name]))
                continue
            v = features_clean.get(name)
            if isinstance(v, (int, float, np.integer, np.floating, np.bool_)):
                tab_values.append(float(v))
            elif isinstance(v, bool):
                tab_values.append(1.0 if v else 0.0)
            else:
                tab_values.append(float("nan"))
        feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
        feature_values = np.asarray(tab_values, dtype=np.float32).reshape(-1)

        # Progress finalize
        _emit_progress(
            rs_path=str(self.rs_path),
            platform_id=str(metadata.get("platform_id") or ""),
            video_id=str(metadata.get("video_id") or ""),
            run_id=str(metadata.get("run_id") or ""),
            done=int(total),
            total=int(total),
            stage="finalize",
        )

        result = {
            "frame_indices": fi_np.astype(np.int32),
            "times_s": times_s.astype(np.float32),
            "face_present_any": face_present_any.astype(bool),
            "frame_feature_names": frame_feature_names,
            "frame_features": frame_features.astype(np.float32),
            "compact22": compact.astype(np.float32),
            "compact22_feature_names": np.asarray(COMPACT22_FEATURE_NAMES, dtype=object),
            "event_times_s": ev_times,
            "event_type_id": ev_type_id,
            "event_strength": ev_strength,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "microexpr_features": microexpr_features,
            "summary": summary,
            # pass df for UI payload embedding into meta; removed before saving
            "_openface_df_for_ui": openface_df_for_ui if openface_df_for_ui is not None else df_sorted,
        }
        
        self.logger.info(
            f"micro_emotion | done: primary={int(fi_np.size)} face_frames={int(np.sum(face_present_any))} openface_rows={len(df_sorted)}"
        )
        
        return result

