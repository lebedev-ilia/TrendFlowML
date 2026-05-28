"""
video_pacing

Содержит:
- `VideoPacingPipelineVisualOptimized` (feature extraction on sampled frames)
- `VideoPacingModule(BaseModule)` — NPZ output + strict frame_indices + core provider integration
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from scipy.stats import entropy
from scipy.signal import find_peaks
from typing import List, Dict, Optional, Any, Tuple

import warnings

warnings.filterwarnings("ignore")

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager


MODULE_NAME = "video_pacing"
VERSION = "2.0.1"
SCHEMA_VERSION = "video_pacing_npz_v3"
ARTIFACT_FILENAME = "video_pacing_features.npz"


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
    # basic
    "video_length_seconds",
    "shots_count",
    "shot_duration_mean",
    "shot_duration_median",
    "shot_duration_min",
    "shot_duration_max",
    "shot_duration_std",
    "shot_duration_mean_normalized",
    "cuts_variance",
    "cuts_per_10s",
    "cuts_per_10s_max",
    "cuts_per_10s_median",
    # shot distribution (optional blocks still included as NaN if absent)
    "shot_duration_entropy",
    "shot_length_gini",
    "short_shot_fraction",
    "quick_cut_burst_count",
    "tempo_entropy",
    # histograms (flattened)
    "shot_length_histogram_5bins_0",
    "shot_length_histogram_5bins_1",
    "shot_length_histogram_5bins_2",
    "shot_length_histogram_5bins_3",
    "shot_length_histogram_5bins_4",
    "cut_density_map_8bins_0",
    "cut_density_map_8bins_1",
    "cut_density_map_8bins_2",
    "cut_density_map_8bins_3",
    "cut_density_map_8bins_4",
    "cut_density_map_8bins_5",
    "cut_density_map_8bins_6",
    "cut_density_map_8bins_7",
    # pace curve
    "pace_curve_slope",
    "pace_curve_slope_normalized",
    "pace_curve_peaks_mean_prominence",
    "pace_curve_dominant_period_sec",
    "pace_curve_power_at_period",
    # motion
    "mean_motion_speed_per_shot",
    "motion_speed_median",
    "motion_speed_variance",
    "motion_speed_90perc",
    "share_of_high_motion_frames",
    "share_of_high_motion_shots",
    "motion_shot_corr",
    # semantic change
    "frame_embedding_diff_mean",
    "frame_embedding_diff_std",
    "high_change_frames_ratio",
    "scene_embedding_jumps",
    "semantic_change_burst_count",
    # color pacing
    "color_change_rate_mean",
    "color_change_rate_std",
    "color_change_bursts",
    "saturation_change_rate",
    "brightness_change_rate",
    # lighting pacing
    "luminance_spikes_per_minute",
    # structural pacing
    "intro_speed",
    "main_speed",
    "climax_speed",
    "pacing_symmetry",
]


def _as_float_feature(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if bool(v) else 0.0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return float("nan")


def _get_vec_elem(d: Dict[str, Any], key: str, i: int) -> float:
    v = d.get(key)
    try:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if 0 <= int(i) < int(arr.size):
            return float(arr[int(i)])
    except Exception:
        pass
    return float("nan")


def _utc_iso_now() -> str:
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

def _require_union_times_s(frame_manager: FrameManager, frame_indices: List[int]) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth for time axis.
    No-fallback: if missing/invalid -> error.
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("video_pacing | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("video_pacing | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)
    fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
    if fi.size == 0:
        raise RuntimeError("video_pacing | frame_indices is empty (no-fallback)")
    if int(np.max(fi)) >= int(uts.shape[0]):
        raise RuntimeError("video_pacing | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[fi]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("video_pacing | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust scale)."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def gini_coefficient(values: np.ndarray) -> float:
    """Gini для неотрицательного массива."""
    if values.size == 0:
        return 0.0
    vals = values.astype(np.float64)
    if np.any(vals < 0):
        vals = vals - vals.min()
    if np.allclose(vals, 0):
        return 0.0
    vals_sorted = np.sort(vals)
    n = vals_sorted.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * vals_sorted) / (n * np.sum(vals_sorted))) - (n + 1) / n)


def _load_core_optical_flow_npz(rs_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Loads core_optical_flow NPZ:
    <rs_path>/core_optical_flow/flow.npz
    """
    if not rs_path:
        return None
    p = os.path.join(rs_path, "core_optical_flow", "flow.npz")
    if not os.path.isfile(p):
        return None
    try:
        data = np.load(p, allow_pickle=True)
        idx = data.get("frame_indices")
        curve = data.get("motion_norm_per_sec_mean")
        if idx is None or curve is None:
            return None
        return {
            "frame_indices": np.asarray(idx, dtype=np.int32),
            "motion_norm_per_sec_mean": np.asarray(curve, dtype=np.float32),
            "meta": data.get("meta"),
        }
    except Exception:
        return None


def _load_core_clip_npz(rs_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Пытается загрузить CLIP‑эмбеддинги из core_clip провайдера.

    Ожидается файл:
    <rs_path>/core_clip/embeddings.npz
    """
    if not rs_path:
        return None

    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        return None

    try:
        data = np.load(core_path, allow_pickle=True)
        frame_indices = data.get("frame_indices")
        emb = data.get("frame_embeddings")
        if frame_indices is None or emb is None:
            return None
        return {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "frame_embeddings": np.asarray(emb, dtype=np.float32),
            "created_at": data.get("created_at"),
            "version": data.get("version"),
            "model_name": data.get("model_name"),
        }
    except Exception:
        return None


class VideoPacingPipelineVisualOptimized:
    def __init__(
        self,
        frame_manager,
        frame_indices,
        clip_model_name: str = "ViT-B/32",
        batch_size: int = 32,
        downscale_factor: float = 0.25,
        min_shot_length_seconds: float = 0.15,
        shot_detect_k: float = 6.0,
        rs_path: Optional[str] = None,
        cut_shot_boundaries_frame_indices: Optional[List[int]] = None,
        enable_entropy_features: bool = False,
        enable_histograms: bool = False,
        enable_pace_curve_peaks: bool = False,
        enable_periodicity: bool = False,
        enable_bursts: bool = False,
    ):
        """
        batch_size: батч для CLIP
        downscale_factor: для Optical Flow и color/lighting features
        """
        self.batch_size = int(batch_size)
        self.downscale_factor = float(downscale_factor)
        self.enable_entropy_features = bool(enable_entropy_features)
        self.enable_histograms = bool(enable_histograms)
        self.enable_pace_curve_peaks = bool(enable_pace_curve_peaks)
        self.enable_periodicity = bool(enable_periodicity)
        self.enable_bursts = bool(enable_bursts)

        # Загружаем кадры через FrameManager
        self.frame_manager = frame_manager
        self.frame_indices = [int(i) for i in frame_indices]
        self.total_frames = len(frame_indices)
        # Strict time-axis contract (Segmenter)
        self.times_s = _require_union_times_s(self.frame_manager, self.frame_indices)
        self.video_length_seconds = float(max(self.times_s[-1] - self.times_s[0], 0.0)) if self.times_s.size else 0.0
        self.min_shot_length_seconds = float(min_shot_length_seconds)
        self.shot_detect_k = float(shot_detect_k)
        self.rs_path = rs_path

        # CLIP модель удалена - используем только core_clip

        # Cache resized frames to avoid repeated resize/conversion work across multiple feature blocks.
        # Must be initialized before any method that may call `_get_resize_frame`.
        self._resize_cache: Dict[int, np.ndarray] = {}

        # Определяем шоты и сцены
        if cut_shot_boundaries_frame_indices is not None:
            self.shot_boundaries = self._shot_boundaries_from_cut_detection(
                [int(x) for x in cut_shot_boundaries_frame_indices]
            )
        else:
            self.shot_boundaries = self._detect_shots_with_merging()

    def _get_resize_frame(self, idx):
        i = int(idx)
        if i in self._resize_cache:
            return self._resize_cache[i]
        fr = cv2.resize(
            self.frame_manager.get(i),
            (0, 0),
            fx=self.downscale_factor,
            fy=self.downscale_factor,
        )
        self._resize_cache[i] = fr
        return fr

    def _shot_boundaries_from_cut_detection(self, shot_boundaries_frame_indices: List[int]) -> List[int]:
        """
        Convert cut_detection union-frame boundaries to POSITIONS within this module's sampled sequence.
        Also merges too-short shots using time axis.
        """
        if not shot_boundaries_frame_indices:
            raise RuntimeError("video_pacing | cut_detection shot boundaries empty (no-fallback)")

        # map union frame idx -> time via Segmenter source-of-truth
        uts = (self.frame_manager.meta or {}).get("union_timestamps_sec")
        if not isinstance(uts, list) or not uts:
            raise RuntimeError("video_pacing | missing union_timestamps_sec (contract)")
        uts_np = np.asarray(uts, dtype=np.float32).reshape(-1)

        b = []
        for x in shot_boundaries_frame_indices:
            xi = int(x)
            if 0 <= xi < int(uts_np.size):
                b.append(xi)
        if not b:
            raise RuntimeError("video_pacing | cut_detection boundaries invalid (no-fallback)")

        # sort by time
        b = sorted(set(b))
        bt = uts_np[np.asarray(b, dtype=np.int32)]
        order = np.argsort(bt)
        b = [b[int(i)] for i in order.tolist()]
        bt = bt[order]

        # Ensure starts at this module's first sampled time
        if float(bt[0]) > float(self.times_s[0]) + 1e-6:
            bt = np.concatenate([np.asarray([self.times_s[0]], dtype=np.float32), bt], axis=0)

        # Merge too-short shots: drop boundaries that open a too-short segment
        keep = [0]
        for i in range(1, int(bt.size)):
            dt_s = float(bt[i] - bt[keep[-1]])
            if dt_s < float(self.min_shot_length_seconds) and len(keep) > 1:
                continue
            keep.append(i)
        bt = bt[np.asarray(keep, dtype=np.int32)]

        # Convert to positions by nearest sampled frame time
        pos = [0]
        for t in bt[1:]:
            j = int(np.argmin(np.abs(self.times_s - float(t))))
            pos.append(int(np.clip(j, 0, self.total_frames - 1)))
        pos = sorted(set(int(x) for x in pos))
        if pos[0] != 0:
            pos = [0] + pos
        return pos

    # -------------------------
    # Shot Detection with SSIM
    # -------------------------

    def _safe_ssim(self, img1, img2):
        h, w = img1.shape[:2]
        min_side = min(h, w)

        # No-fallback: if Segmenter produced frames too small for SSIM, treat as invalid input.
        if min_side < 3:
            raise RuntimeError("video_pacing | frames too small for SSIM (min_side < 3). Check Segmenter sampling/resolution.")

        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            raise RuntimeError("video_pacing | frames too small for SSIM window (win_size < 3).")

        return ssim(
            img1,
            img2,
            channel_axis=-1,
            win_size=win_size
        )

    def _detect_shots_with_merging(self) -> List[int]:
        """
        Shot boundary detection on sampled frames.

        Design goals (audit):
        - no hard-coded global thresholds tied to FPS / sampling density
        - robust thresholds derived from per-video statistics (MAD)
        - merging too-short shots uses time axis (union_timestamps_sec)
        """
        if not self.frame_indices:
            return [0]

        # IMPORTANT: store boundaries as POSITIONS in `self.frame_indices` list (0..N-1),
        # not as union frame indices. This keeps all downstream computations consistent.
        # 1-pass feature extraction for transitions
        ssim_scores: List[float] = []
        chi_scores: List[float] = []
        edge_scores: List[float] = []
        vdiff_scores: List[float] = []

        prev_idx = self.frame_indices[0]
        prev_frame = self._get_resize_frame(prev_idx)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)
        prev_v = float(np.mean(prev_hsv[:, :, 2]))
        prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
        prev_hist = cv2.normalize(prev_hist, None).flatten()
        prev_edges = cv2.Canny(prev_gray, 50, 150)

        for pos in range(1, len(self.frame_indices)):
            idx = self.frame_indices[pos]
            curr_frame = self._get_resize_frame(idx)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
            curr_v = float(np.mean(curr_hsv[:, :, 2]))

            ssim_score = float(self._safe_ssim(prev_frame, curr_frame))
            curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
            curr_hist = cv2.normalize(curr_hist, None).flatten()
            chi_sq = float(0.5 * np.sum(((prev_hist - curr_hist) ** 2) / (prev_hist + curr_hist + 1e-6)))

            curr_edges = cv2.Canny(curr_gray, 50, 150)
            edge_diff = float(np.mean(cv2.absdiff(prev_edges, curr_edges) > 0))
            v_diff = float(abs(curr_v - prev_v))

            ssim_scores.append(ssim_score)
            chi_scores.append(chi_sq)
            edge_scores.append(edge_diff)
            vdiff_scores.append(v_diff)

            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_hsv = curr_hsv
            prev_v = curr_v
            prev_hist = curr_hist
            prev_edges = curr_edges

        ssim_arr = np.asarray(ssim_scores, dtype=np.float32)
        chi_arr = np.asarray(chi_scores, dtype=np.float32)
        edge_arr = np.asarray(edge_scores, dtype=np.float32)
        vdiff_arr = np.asarray(vdiff_scores, dtype=np.float32)

        if ssim_arr.size == 0:
            return [0]

        # Robust per-video thresholds (MAD-based)
        k = float(self.shot_detect_k)
        ssim_med, chi_med, edge_med, vdiff_med = map(float, [np.median(ssim_arr), np.median(chi_arr), np.median(edge_arr), np.median(vdiff_arr)])
        ssim_thr = float(ssim_med - k * (_mad(ssim_arr) + 1e-9))
        chi_thr = float(chi_med + k * (_mad(chi_arr) + 1e-9))
        edge_thr = float(edge_med + k * (_mad(edge_arr) + 1e-9))
        vdiff_thr = float(vdiff_med + k * (_mad(vdiff_arr) + 1e-9))

        # Decision rule: a cut is a strong semantic/visual discontinuity vs local baseline
        cut_mask = (ssim_arr < ssim_thr) & ((chi_arr > chi_thr) | (edge_arr > edge_thr) | (vdiff_arr > vdiff_thr))
        cut_positions = (np.nonzero(cut_mask)[0] + 1).astype(np.int32)  # +1: transition i corresponds to boundary at pos i+1

        shot_positions = [0] + [int(x) for x in cut_positions.tolist()]

        # Объединяем слишком короткие шоты
        if len(shot_positions) <= 1:
            return [0]

        # shot boundaries in POSITIONS
        boundaries = sorted(set(int(x) for x in shot_positions))
        merged = [boundaries[0]]
        last_start = boundaries[0]
        for b in boundaries[1:]:
            dur_s = float(self.times_s[b] - self.times_s[last_start])
            if dur_s < self.min_shot_length_seconds:
                # не открываем новый шот, просто сливаем
                continue
            merged.append(b)
            last_start = b

        if not merged:
            merged = [0]
        return merged

    # -------------------------
    # Shot Features
    # -------------------------
    def extract_shot_features(self) -> Dict:
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        # ensure start boundary exists
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)
        if durations_sec.size == 0:
            return {}

        # базовые статистики
        mean_dur = float(np.mean(durations_sec))
        med_dur = float(np.median(durations_sec))
        min_dur = float(np.min(durations_sec))
        max_dur = float(np.max(durations_sec))
        std_dur = float(np.std(durations_sec))

        # Optional: entropy / gini / histograms are often noisy on small number of shots.
        dur_entropy = None
        gini = None
        if bool(self.enable_entropy_features):
            hist_counts, _ = np.histogram(durations_sec, bins=20)
            dur_entropy = float(entropy(hist_counts + 1e-9))
            gini = gini_coefficient(durations_sec)

        # нормализация длительности на длину видео
        norm_mean = float(mean_dur / max(self.video_length_seconds, 1e-6))

        # short_shot_fraction (<0.5 s)
        short_threshold = 0.5
        short_shot_fraction = float(
            np.mean(durations_sec < short_threshold) if durations_sec.size > 0 else 0.0
        )

        # quick_cut_burst_count: >=3 cut за 1 секунду (time-axis, union_timestamps_sec)
        cut_times = bt[1:].astype(np.float32)  # exclude t0
        quick_cut_burst_count = 0
        if cut_times.size >= 3:
            i = 0
            while i < len(cut_times):
                j = i + 1
                while j < len(cut_times) and cut_times[j] - cut_times[i] <= 1.0:
                    j += 1
                if j - i >= 3:
                    quick_cut_burst_count += 1
                i += 1

        hist_fracs = None
        tempo_entropy_val = None
        if bool(self.enable_histograms):
            # shot length histogram bins (very_short, short, medium, long, very_long)
            bins_sec = np.array([0.0, 0.3, 0.7, 1.5, 3.0, np.inf], dtype=np.float32)
            hist_counts_5, _ = np.histogram(durations_sec, bins=bins_sec)
            total_shots = float(hist_counts_5.sum()) if hist_counts_5.sum() > 0 else 1.0
            hist_fracs = (hist_counts_5 / total_shots).tolist()
            if bool(self.enable_entropy_features):
                tempo_entropy_val = float(entropy(hist_counts_5 + 1e-9))

        # cuts per 10 seconds (max/median over sliding 10s windows)
        window = 10.0
        if cut_times.size > 0 and self.video_length_seconds > 0:
            cut_rel = (cut_times - float(self.times_s[0])).astype(np.float32)
            t_edges = np.arange(0.0, self.video_length_seconds + window, window, dtype=np.float32)
            cuts_per_window, _ = np.histogram(cut_rel, bins=t_edges)
            cuts_per_10s_series = cuts_per_window / window
            cuts_per_10s_max = float(cuts_per_10s_series.max())
            cuts_per_10s_median = float(np.median(cuts_per_10s_series))
            cuts_per_10s_global = float(cuts_per_window.sum() / max(self.video_length_seconds / window, 1e-6))
        else:
            cuts_per_10s_series = np.array([0.0], dtype=np.float32)
            cuts_per_10s_max = 0.0
            cuts_per_10s_median = 0.0
            cuts_per_10s_global = 0.0

        # cut_density_map по 8 бинам времени
        if cut_times.size > 0 and self.video_length_seconds > 0:
            cut_rel = (cut_times - float(self.times_s[0])).astype(np.float32)
            bins8 = np.linspace(0.0, self.video_length_seconds, 9, dtype=np.float32)
            cuts8, _ = np.histogram(cut_rel, bins=bins8)
            cut_density_map = (cuts8 / max(self.video_length_seconds / 8.0, 1e-6)).tolist()
        else:
            cut_density_map = [0.0] * 8

        out = {
            "shot_duration_mean": mean_dur,
            "shot_duration_median": med_dur,
            "shot_duration_min": min_dur,
            "shot_duration_max": max_dur,
            "shot_duration_std": std_dur,
            "shot_duration_mean_normalized": norm_mean,
            "cuts_per_10s": cuts_per_10s_global,
            "cuts_per_10s_max": cuts_per_10s_max,
            "cuts_per_10s_median": cuts_per_10s_median,
            "cuts_variance": float(np.var(durations_sec)),
            "short_shot_fraction": short_shot_fraction,
            "quick_cut_burst_count": int(quick_cut_burst_count),
            "cut_density_map_8bins": cut_density_map,
            "shots_count": int(durations_sec.size),
        }
        if dur_entropy is not None:
            out["shot_duration_entropy"] = float(dur_entropy)
        if gini is not None:
            out["shot_length_gini"] = float(gini)
        if hist_fracs is not None:
            out["shot_length_histogram_5bins"] = list(hist_fracs)
        if tempo_entropy_val is not None:
            out["tempo_entropy"] = float(tempo_entropy_val)
        return out

    def extract_pace_curve(self) -> Dict:
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)
        if durations_sec.size == 0:
            return {}

        x = np.arange(len(durations_sec), dtype=np.float32)
        if len(durations_sec) >= 2:
            # простая регрессия (псевдо-робастность достигается за счёт клиппинга лог-длительностей)
            y = np.log1p(durations_sec)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        pace_curve_slope = slope
        pace_curve_slope_normalized = float(slope * float(np.mean(durations_sec))) if float(np.mean(durations_sec)) > 0 else 0.0

        pace_curve_peaks = 0
        pace_curve_peaks_mean_prominence = 0.0
        peak_positions: List[float] = []
        if bool(self.enable_pace_curve_peaks):
            # пики (локальные максимумы длительностей = замедления)
            if len(durations_sec) >= 3:
                peaks, props = find_peaks(
                    durations_sec,
                    prominence=np.std(durations_sec) if np.std(durations_sec) > 0 else 0.0,
                    distance=1,
                )
                pace_curve_peaks = int(len(peaks))
                peak_prom = props.get("prominences", np.zeros_like(peaks, dtype=np.float32))
                pace_curve_peaks_mean_prominence = float(peak_prom.mean()) if peak_prom.size else 0.0
                peak_positions = (peaks / max(len(durations_sec) - 1, 1)).astype(np.float32).tolist()

        dominant_period_sec = 0.0
        power_at_period = 0.0
        if bool(self.enable_periodicity):
            # периодичность: по автокорреляции, вернуть период в секундах и мощность
            durations_centered = durations_sec - np.mean(durations_sec)
            autocorr = np.correlate(durations_centered, durations_centered, mode="full")
            mid = len(autocorr) // 2
            autocorr = autocorr[mid + 1 :]
            if autocorr.size > 0 and np.max(autocorr) > 0:
                autocorr_norm = autocorr / np.max(autocorr)
                best_lag = int(np.argmax(autocorr_norm) + 1)
                dominant_period_sec = float(best_lag * np.mean(durations_sec))
                power_at_period = float(autocorr_norm[best_lag - 1])

        out = {
            "pace_curve_slope": pace_curve_slope,
            "pace_curve_slope_normalized": pace_curve_slope_normalized,
        }
        if bool(self.enable_pace_curve_peaks):
            out["pace_curve_peaks"] = int(pace_curve_peaks)
            out["pace_curve_peaks_mean_prominence"] = float(pace_curve_peaks_mean_prominence)
            out["pace_curve_peak_positions"] = list(peak_positions)
        if bool(self.enable_periodicity):
            out["pace_curve_dominant_period_sec"] = float(dominant_period_sec)
            out["pace_curve_power_at_period"] = float(power_at_period)
        return out

    # -------------------------
    # Motion / Optical Flow
    # -------------------------
    def extract_motion_features(self) -> Dict:
        """
        Motion‑фичи. Используем только core_optical_flow (RAFT).
        """
        core = _load_core_optical_flow_npz(self.rs_path)
        if core is None:
            raise RuntimeError("video_pacing | core_optical_flow not found (required)")
        core_idx = core["frame_indices"]
        core_curve = core["motion_norm_per_sec_mean"]
        if core_curve is None or core_curve.size == 0:
            raise RuntimeError("video_pacing | core_optical_flow curve is empty")

        # Align to this module's frame_indices (union-domain). Must be fully covered.
        want = np.asarray(self.frame_indices, dtype=np.int32)
        mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
        pos = [mapping.get(int(fi), -1) for fi in want.tolist()]
        if any(p < 0 for p in pos):
            raise RuntimeError(
                "video_pacing | core_optical_flow.frame_indices does not cover this module's frame_indices. "
                "Segmenter must produce consistent sampling for dependent components."
            )
        core_curve = core_curve[np.asarray(pos, dtype=np.int64)]

        # лёгкое сглаживание (ignore NaNs from the first element)
        if np.isnan(core_curve[0]):
            core_curve = core_curve.copy()
            core_curve[0] = 0.0
        if core_curve.size >= 3:
            flow_mags_smooth = np.convolve(core_curve, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        else:
            flow_mags_smooth = core_curve

        # пер-кадровые метрики
        mean_motion = float(np.mean(flow_mags_smooth))
        median_motion = float(np.median(flow_mags_smooth))
        var_motion = float(np.var(flow_mags_smooth))
        perc90_motion = float(np.percentile(flow_mags_smooth, 90))

        high_thr = float(np.percentile(flow_mags_smooth, 75))
        share_high_frames = float(np.mean(flow_mags_smooth > high_thr))

        # пер-шотовые motion-агрегаты и корреляция с длиной шота
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)

        shot_motion_means = []
        for i in range(len(boundaries_pos)):
            start = boundaries_pos[i]
            end = boundaries_pos[i + 1] if i + 1 < len(boundaries_pos) else int(self.total_frames)
            # core_optical_flow curve is per-frame; the first element is typically 0 (no previous frame).
            local = flow_mags_smooth[min(start + 1, flow_mags_smooth.size) : min(end, flow_mags_smooth.size)]
            if local.size > 0:
                shot_motion_means.append(float(np.mean(local)))
            else:
                shot_motion_means.append(0.0)

        motion_shot_corr = 0.0
        if len(shot_motion_means) == len(durations_sec) and len(durations_sec) > 1:
            x = np.asarray(durations_sec, dtype=np.float32)
            y = np.asarray(shot_motion_means, dtype=np.float32)
            if np.std(x) > 0 and np.std(y) > 0:
                motion_shot_corr = float(np.corrcoef(x, y)[0, 1])

        share_high_motion_shots = 0.0
        if shot_motion_means:
            thr_shot = float(np.percentile(shot_motion_means, 75))
            share_high_motion_shots = float(
                np.mean(np.asarray(shot_motion_means, dtype=np.float32) > thr_shot)
            )

        return {
            "mean_motion_speed_per_shot": mean_motion,
            "motion_speed_median": median_motion,
            "motion_speed_variance": var_motion,
            "motion_speed_90perc": perc90_motion,
            "share_of_high_motion_frames": share_high_frames,
            "share_of_high_motion_shots": share_high_motion_shots,
            "motion_shot_corr": motion_shot_corr,
        }

    def _get_clip_frame(self, idx):
        frame = self.frame_manager.get(idx)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        return frame

    # -------------------------
    # CLIP embeddings (batched)
    # -------------------------
    def extract_content_change_rate(self) -> Dict:
        # Используем только core_clip - обязательное требование
        core = _load_core_clip_npz(self.rs_path)
        if core is None:
            raise RuntimeError("video_pacing | core_clip not found (required)")
        frame_indices = core["frame_indices"]
        embeddings = core["frame_embeddings"]

        # Align to this module's frame_indices (union-domain). If not fully covered, treat as missing.
        want = np.asarray(self.frame_indices, dtype=np.int32)
        mapping = {int(fi): i for i, fi in enumerate(frame_indices.tolist())}
        pos = [mapping.get(int(fi), -1) for fi in want.tolist()]
        if any(p < 0 for p in pos):
            raise RuntimeError(
                "video_pacing | core_clip.frame_indices does not cover this module's frame_indices. "
                "Segmenter must produce consistent sampling for dependent components."
            )
        embeddings = embeddings[np.asarray(pos, dtype=np.int64)]

        # cosine distance между соседними эмбеддингами
        # нормируем эмбеддинги
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        emb_norm = embeddings / norms
        cos_sim = np.sum(emb_norm[1:] * emb_norm[:-1], axis=1)
        cos_dist = 1.0 - cos_sim  # 0..2
        # Normalize by dt to be robust to variable sampling density.
        dt = np.diff(self.times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        cos_rate = (cos_dist.astype(np.float32) / dt).astype(np.float32)

        # сглаживание
        if cos_rate.size >= 7:
            kernel_size = 5
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            diff_smooth = np.convolve(cos_rate, kernel, mode="same")
        else:
            diff_smooth = cos_rate

        mean_diff = float(np.mean(diff_smooth))
        std_diff = float(np.std(diff_smooth))

        thr75 = float(np.percentile(diff_smooth, 75))
        high_change_ratio = float(np.mean(diff_smooth > thr75))

        thr_jump = mean_diff + 2.0 * std_diff
        scene_jumps = int(np.sum(diff_smooth > thr_jump))

        burst_count = 0
        if bool(self.enable_bursts):
            # semantic_change_burst_count: >=3 high-change transitions within any 5 seconds window
            high_mask = diff_smooth > thr_jump
            if np.any(high_mask):
                trans_times = ((self.times_s[1:] + self.times_s[:-1]) * 0.5).astype(np.float32)
                high_times = trans_times[high_mask]
                window_s = 5.0
                i = 0
                while i < high_times.size:
                    j = i + 1
                    while j < high_times.size and float(high_times[j] - high_times[i]) <= window_s:
                        j += 1
                    if j - i >= 3:
                        burst_count += 1
                    i = j

        return {
            "frame_embedding_diff_mean": mean_diff,
            "frame_embedding_diff_std": std_diff,
            "high_change_frames_ratio": high_change_ratio,
            "scene_embedding_jumps": scene_jumps,
            **({"semantic_change_burst_count": int(burst_count)} if bool(self.enable_bursts) else {}),
        }

    # -------------------------
    # Color & Lighting Pacing
    # -------------------------
    def extract_color_pacing(self) -> Dict:
        hist_diffs = []
        if not self.frame_indices:
            return {}
        prev_frame = self._get_resize_frame(self.frame_indices[0])
        for idx in self.frame_indices[1:]:
            frame = self._get_resize_frame(idx)
            lab1 = rgb2lab(prev_frame)
            lab2 = rgb2lab(frame)
            deltaE = np.sqrt(np.sum((lab1-lab2)**2, axis=2))
            hist_diffs.append(np.mean(deltaE))
            prev_frame = frame
        hist_diffs = np.array(hist_diffs, dtype=np.float32)
        dt = np.diff(self.times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        hist_rate = (hist_diffs / dt).astype(np.float32)

        # локальный baseline (скользящее среднее) для дельты цвета
        if hist_rate.size >= 7:
            kernel = np.ones(7, dtype=np.float32) / 7.0
            baseline = np.convolve(hist_rate, kernel, mode="same")
        else:
            baseline = np.full_like(hist_rate, float(np.mean(hist_rate)) if hist_rate.size else 0.0)
        hist_diffs_detrended = hist_rate - baseline

        saturation = np.asarray(
            [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:, :, 1])
            for idx in self.frame_indices
            ],
            dtype=np.float32,
        )
        brightness = np.asarray(
            [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:, :, 2])
            for idx in self.frame_indices
            ],
            dtype=np.float32,
        )

        color_change_bursts = 0
        if bool(self.enable_bursts):
            # color_change_bursts по detrended DeltaE
            if hist_diffs_detrended.size > 0:
                thr_color = float(np.mean(hist_diffs_detrended) + 2.0 * np.std(hist_diffs_detrended))
                peaks, _ = find_peaks(hist_diffs_detrended, height=thr_color, distance=1)
                color_change_bursts = int(len(peaks))

        sat_rate = np.diff(saturation) / dt if saturation.size >= 2 else np.asarray([], dtype=np.float32)
        bri_rate = np.diff(brightness) / dt if brightness.size >= 2 else np.asarray([], dtype=np.float32)

        return {
            "color_change_rate_mean": float(np.mean(hist_rate)) if hist_rate.size else 0.0,
            "color_change_rate_std": float(np.std(hist_rate)) if hist_rate.size else 0.0,
            "saturation_change_rate": float(np.std(sat_rate)) if sat_rate.size else 0.0,
            "brightness_change_rate": float(np.std(bri_rate)) if bri_rate.size else 0.0,
            **({"color_change_bursts": int(color_change_bursts)} if bool(self.enable_bursts) else {}),
        }

    def extract_lighting_pacing(self) -> Dict:
        lum = np.asarray(
            [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2GRAY))
            for idx in self.frame_indices
            ],
            dtype=np.float32,
        )
        if lum.size < 2:
            return {
                "luminance_spikes_per_minute": 0.0,
            }

        dt = np.diff(self.times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        lum_rate = np.diff(lum) / dt

        # Robust spike threshold (MAD-based). Avoid FFT because sampling is non-uniform.
        med = float(np.median(lum_rate))
        thr = float(abs(med) + 6.0 * (_mad(lum_rate) + 1e-9))
        spikes = np.abs(lum_rate - med) > thr
        spikes_count = int(np.sum(spikes))

        time_minutes = max(self.video_length_seconds / 60.0, 1e-6)
        lum_spikes_per_minute = float(spikes_count / time_minutes)

        return {
            "luminance_spikes_per_minute": lum_spikes_per_minute,
        }

    # -------------------------
    # Structural Pacing
    # -------------------------
    def extract_structural_pacing(self) -> Dict:
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)
        if durations_sec.size == 0:
            return {}
        n = len(durations_sec)
        quarter = max(n // 4, 1)
        intro = float(np.median(durations_sec[:quarter]))
        main = float(np.median(durations_sec[quarter : 3 * quarter]))
        climax = float(np.median(durations_sec[3 * quarter :]))
        overall_med = float(np.median(durations_sec))

        if overall_med > 0:
            pacing_symmetry = float((climax - intro) / overall_med)
        else:
            pacing_symmetry = 0.0

        return {
            "intro_speed": intro,
            "main_speed": main,
            "climax_speed": climax,
            "pacing_symmetry": pacing_symmetry,
        }

    # -------------------------
    # Full Pipeline
    # -------------------------
    def extract_all_features(self) -> Dict:
        features = {}
        # Основные визуальные метрики
        features.update(self.extract_shot_features())
        features.update(self.extract_pace_curve())
        # Hard deps: core_optical_flow + core_clip (contract)
        features.update(self.extract_motion_features())
        features.update(self.extract_content_change_rate())
        features.update(self.extract_color_pacing())
        features.update(self.extract_lighting_pacing())
        features.update(self.extract_structural_pacing())

        # Note: AV sync / per-person / object pacing can be added later as separate optional blocks.

        return features


class VideoPacingModule(BaseModule):
    """
    BaseModule wrapper for `video_pacing`.

    Контракты:
    - `frame_indices` приходят только из metadata (Segmenter).
    - Кадры в FrameManager — RGB.
    - Core providers:
      - `core_optical_flow` (motion curve)
      - `core_clip` (semantic content-change rate)
    """

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    # Baseline: fixed artifact filename (run_id already provides uniqueness in path).
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    def __init__(self, rs_path: Optional[str] = None, downscale_factor: float = 0.25, **kwargs: Any):
        super().__init__(rs_path=rs_path, logger_name=self.module_name, **kwargs)
        self._downscale_factor = float(downscale_factor)

    def required_dependencies(self) -> List[str]:
        # Baseline policy: shot boundaries are provided by cut_detection (module dep).
        return ["core_optical_flow", "core_clip", "cut_detection"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if not frame_indices:
            raise ValueError("video_pacing | frame_indices is empty")

        min_frames = int(config.get("min_frames", 30))
        if len(frame_indices) < min_frames:
            raise RuntimeError(f"video_pacing | too few frames: N={len(frame_indices)} < min_frames={min_frames} (no-fallback)")

        downscale = float(config.get("downscale_factor", self._downscale_factor))
        min_shot_len_s = float(config.get("min_shot_length_seconds", 0.15))
        shot_detect_k = float(config.get("shot_detect_k", 6.0))
        # feature gating (remove noisy by default)
        enable_entropy_features = bool(config.get("enable_entropy_features", False))
        enable_histograms = bool(config.get("enable_histograms", False))
        enable_pace_curve_peaks = bool(config.get("enable_pace_curve_peaks", False))
        enable_periodicity = bool(config.get("enable_periodicity", False))
        enable_bursts = bool(config.get("enable_bursts", False))

        # Strict time-axis (Segmenter source-of-truth)
        times_s = _require_union_times_s(frame_manager, frame_indices)

        # Shot boundaries from cut_detection (no-fallback).
        cd = self.load_dependency_results("cut_detection", format="npz")
        if not isinstance(cd, dict):
            raise RuntimeError("video_pacing | cut_detection results missing/invalid (no-fallback)")
        det = cd.get("detections")
        if not isinstance(det, dict):
            raise RuntimeError("video_pacing | cut_detection missing detections dict (no-fallback)")
        sb = det.get("shot_boundaries_frame_indices")
        if not isinstance(sb, list) or not sb:
            raise RuntimeError("video_pacing | cut_detection missing shot_boundaries_frame_indices (no-fallback)")
        shot_boundaries_frame_indices_cd = [int(x) for x in sb]

        pipeline = VideoPacingPipelineVisualOptimized(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            downscale_factor=downscale,
            min_shot_length_seconds=min_shot_len_s,
            shot_detect_k=shot_detect_k,
            rs_path=self.rs_path,
            cut_shot_boundaries_frame_indices=shot_boundaries_frame_indices_cd,
            enable_entropy_features=enable_entropy_features,
            enable_histograms=enable_histograms,
            enable_pace_curve_peaks=enable_pace_curve_peaks,
            enable_periodicity=enable_periodicity,
            enable_bursts=enable_bursts,
        )

        raw_features = pipeline.extract_all_features()
        if not isinstance(raw_features, dict):
            raw_features = {}

        frame_indices_np = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        # Export union-frame indices directly from cut_detection output (dedup + sorted).
        shot_boundaries_frame_indices = np.asarray(sorted(set(int(x) for x in shot_boundaries_frame_indices_cd)), dtype=np.int32)

        # Add lightweight time-series for UI (aligned to this module sampling).
        # motion curve aligned
        core_flow = self.load_core_provider("core_optical_flow", file_name="flow.npz")
        if not isinstance(core_flow, dict) or core_flow.get("frame_indices") is None or core_flow.get("motion_norm_per_sec_mean") is None:
            raise RuntimeError("video_pacing | core_optical_flow artifact missing/invalid (no-fallback)")
        fi_flow = np.asarray(core_flow["frame_indices"], dtype=np.int32)
        motion_curve = np.asarray(core_flow["motion_norm_per_sec_mean"], dtype=np.float32).reshape(-1)
        mapping = {int(x): i for i, x in enumerate(fi_flow.tolist())}
        pos = [mapping.get(int(x), -1) for x in frame_indices_np.tolist()]
        if any(p < 0 for p in pos):
            raise RuntimeError("video_pacing | core_optical_flow does not cover requested frame_indices (no-fallback)")
        motion_aligned = motion_curve[np.asarray(pos, dtype=np.int64)].astype(np.float32)

        # semantic change rate aligned (N, with 0 at first)
        core_clip = self.load_core_provider("core_clip", file_name="embeddings.npz")
        if not isinstance(core_clip, dict) or core_clip.get("frame_indices") is None or core_clip.get("frame_embeddings") is None:
            raise RuntimeError("video_pacing | core_clip artifact missing/invalid (no-fallback)")
        fi_clip = np.asarray(core_clip["frame_indices"], dtype=np.int32)
        emb = np.asarray(core_clip["frame_embeddings"], dtype=np.float32)
        mapping2 = {int(x): i for i, x in enumerate(fi_clip.tolist())}
        pos2 = [mapping2.get(int(x), -1) for x in frame_indices_np.tolist()]
        if any(p < 0 for p in pos2):
            raise RuntimeError("video_pacing | core_clip does not cover requested frame_indices (no-fallback)")
        emb_aligned = emb[np.asarray(pos2, dtype=np.int64)]
        norms = np.linalg.norm(emb_aligned, axis=1, keepdims=True) + 1e-9
        en = emb_aligned / norms
        if en.shape[0] >= 2:
            cos_sim_next = np.sum(en[1:] * en[:-1], axis=1).astype(np.float32)
            cos_dist_next = (1.0 - cos_sim_next).astype(np.float32)
            dt = np.diff(np.asarray(times_s, dtype=np.float32)).astype(np.float32)
            dt = np.maximum(dt, 1e-3)
            rate = (cos_dist_next / dt).astype(np.float32)
            semantic_change_rate = np.concatenate([np.asarray([0.0], dtype=np.float32), rate], axis=0)
        else:
            semantic_change_rate = np.zeros((en.shape[0],), dtype=np.float32)

        # color change rate aligned (mean LAB delta / dt, 0 at first)
        if frame_indices_np.size >= 2:
            # reuse pipeline cache (downscaled frames) by recomputing mean lab quickly
            lab_means = []
            for idx in frame_indices_np.tolist():
                fr = cv2.resize(frame_manager.get(int(idx)), (0, 0), fx=downscale, fy=downscale)
                lab = cv2.cvtColor(fr, cv2.COLOR_RGB2LAB)
                lab_means.append(np.asarray(np.mean(lab.reshape(-1, 3), axis=0), dtype=np.float32))
            lab_means = np.asarray(lab_means, dtype=np.float32)
            d = np.linalg.norm(lab_means[1:] - lab_means[:-1], axis=1).astype(np.float32)
            dt = np.diff(np.asarray(times_s, dtype=np.float32)).astype(np.float32)
            dt = np.maximum(dt, 1e-3)
            rate = (d / dt).astype(np.float32)
            color_change_rate = np.concatenate([np.asarray([0.0], dtype=np.float32), rate], axis=0)
        else:
            color_change_rate = np.zeros((frame_indices_np.size,), dtype=np.float32)

        # Build stable tabular model-facing scalars (no object-dict in NPZ)
        video_len_s = float(times_s[-1] - times_s[0]) if isinstance(times_s, np.ndarray) and times_s.size >= 2 else 0.0
        scalars: Dict[str, Any] = dict(raw_features)
        scalars.setdefault("video_length_seconds", float(video_len_s))

        # Flatten histogram vectors into scalar slots
        for j in range(5):
            scalars[f"shot_length_histogram_5bins_{j}"] = _get_vec_elem(scalars, "shot_length_histogram_5bins", j)
        for j in range(8):
            scalars[f"cut_density_map_8bins_{j}"] = _get_vec_elem(scalars, "cut_density_map_8bins", j)

        feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
        feature_values = np.asarray([_as_float_feature(scalars.get(k)) for k in _FEATURE_NAMES_V1], dtype=np.float32).reshape(-1)

        return {
            "frame_indices": frame_indices_np,
            "times_s": np.asarray(times_s, dtype=np.float32),
            "shot_boundary_frame_indices": shot_boundaries_frame_indices,
            "motion_norm_per_sec_mean": motion_aligned,
            "semantic_change_rate_per_sec": semantic_change_rate,
            "color_change_rate_per_sec": color_change_rate,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "ui_payload": {
                "schema_version": "video_pacing_ui_v1",
                "curves": {
                    "motion_norm_per_sec_mean": {"npz_key": "motion_norm_per_sec_mean", "label": "Motion (per-sec mean)"},
                    "semantic_change_rate_per_sec": {"npz_key": "semantic_change_rate_per_sec", "label": "Semantic change (/s)"},
                    "color_change_rate_per_sec": {"npz_key": "color_change_rate_per_sec", "label": "Color change (/s)"},
                },
                "markers": {
                    "shot_boundaries": [int(x) for x in shot_boundaries_frame_indices.tolist()],
                },
            },
        }

    def run(self, frames_dir: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Override BaseModule.run to add progress events, ui_payload in meta, and stage timings.
        """
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
                "downscale_factor": float((config or {}).get("downscale_factor", self._downscale_factor)),
                "min_shot_length_seconds": float((config or {}).get("min_shot_length_seconds", 0.15)),
                "shot_detect_k": float((config or {}).get("shot_detect_k", 6.0)),
                "min_frames": int((config or {}).get("min_frames", 30)),
                "enable_entropy_features": bool((config or {}).get("enable_entropy_features", False)),
                "enable_histograms": bool((config or {}).get("enable_histograms", False)),
                "enable_pace_curve_peaks": bool((config or {}).get("enable_pace_curve_peaks", False)),
                "enable_periodicity": bool((config or {}).get("enable_periodicity", False)),
                "enable_bursts": bool((config or {}).get("enable_bursts", False)),
            }
            if isinstance(resource_profile_before, dict) and resource_profile_before:
                save_metadata["resource_profile_before"] = dict(resource_profile_before)

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