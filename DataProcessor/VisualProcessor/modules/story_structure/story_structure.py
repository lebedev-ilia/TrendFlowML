"""
story_structure (Tier‑0 baseline)

Baseline mode:
- Computes story/energy/coherence proxies from sampled frames.
- Hard dependencies: core_clip, core_optical_flow, core_face_landmarks.
- Time axis is strictly `union_timestamps_sec` (no fallback).
- Per-second normalization for change signals (robust to sampling density).

Legacy / non-baseline experiments are moved to `legacy_story_structure.py` and MUST use ModelManager.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

MODULE_NAME = "story_structure"
VERSION = "3.0"
SCHEMA_VERSION = "story_structure_npz_v1"
ARTIFACT_FILENAME = "story_structure.npz"


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


def _unbox_object_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _require_union_times_s(frame_manager: FrameManager, frame_indices: np.ndarray) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth for time axis.
    No-fallback: if missing/invalid -> error.
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("story_structure | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("story_structure | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)

    if frame_indices.size == 0:
        raise RuntimeError("story_structure | frame_indices is empty (no-fallback)")
    if int(np.max(frame_indices)) >= int(uts.shape[0]):
        raise RuntimeError("story_structure | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[frame_indices.astype(np.int32)]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("story_structure | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _load_npz_meta_models_used(npz: np.lib.npyio.NpzFile) -> List[Dict[str, Any]]:
    meta = _unbox_object_scalar(npz.get("meta"))
    if isinstance(meta, dict):
        mu = meta.get("models_used")
        if isinstance(mu, list):
            return [x for x in mu if isinstance(x, dict)]
    return []


def _align_by_frame_indices(core_idx: np.ndarray, want_idx: np.ndarray, *, who: str) -> np.ndarray:
    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in want_idx.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            f"{MODULE_NAME} | {who}.frame_indices does not cover requested frame_indices. "
            "Segmenter must produce consistent sampling for dependent components."
        )
    return np.asarray(pos, dtype=np.int64)


def _load_core_clip_embeddings_aligned(rs_path: str, fi: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    p = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{MODULE_NAME} | missing core_clip embeddings: {p}")
    npz = np.load(p, allow_pickle=True)
    core_idx = npz.get("frame_indices")
    core_emb = npz.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError(f"{MODULE_NAME} | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)
    pos = _align_by_frame_indices(core_idx, fi, who="core_clip")
    return core_emb[pos], _load_npz_meta_models_used(npz)


def _load_core_optical_flow_aligned(rs_path: str, fi: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    p = os.path.join(rs_path, "core_optical_flow", "flow.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{MODULE_NAME} | missing core_optical_flow flow: {p}")
    npz = np.load(p, allow_pickle=True)
    core_idx = npz.get("frame_indices")
    curve = npz.get("motion_norm_per_sec_mean")
    if core_idx is None or curve is None:
        raise RuntimeError(f"{MODULE_NAME} | core_optical_flow flow.npz missing keys frame_indices/motion_norm_per_sec_mean")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    curve = np.asarray(curve, dtype=np.float32)
    pos = _align_by_frame_indices(core_idx, fi, who="core_optical_flow")
    return curve[pos], _load_npz_meta_models_used(npz)


def _load_core_face_any_present_aligned(rs_path: str, fi: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      - any_face_present (N,) bool
      - models_used
      - provider_meta (best-effort, unboxed)
    """
    p = os.path.join(rs_path, "core_face_landmarks", "landmarks.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{MODULE_NAME} | missing core_face_landmarks landmarks: {p}")
    npz = np.load(p, allow_pickle=True)
    core_idx = npz.get("frame_indices")
    face_present = npz.get("face_present")
    if core_idx is None or face_present is None:
        raise RuntimeError(f"{MODULE_NAME} | core_face_landmarks landmarks.npz missing keys frame_indices/face_present")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    face_present = np.asarray(face_present, dtype=bool)
    if face_present.ndim == 1:
        any_face = face_present
    else:
        any_face = np.any(face_present, axis=1)
    pos = _align_by_frame_indices(core_idx, fi, who="core_face_landmarks")
    meta = _unbox_object_scalar(npz.get("meta"))
    return np.asarray(any_face[pos], dtype=bool), _load_npz_meta_models_used(npz), meta if isinstance(meta, dict) else {}


def _downsample_to_fixed(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if m <= 0:
        return np.asarray([], dtype=np.float32)
    if x.size == 0:
        return np.zeros((m,), dtype=np.float32)
    if x.size == 1:
        return np.full((m,), float(x[0]), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=int(x.size), dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=int(m), dtype=np.float32)
    return np.interp(xq, xp, x.astype(np.float32)).astype(np.float32)


class StoryStructureBaselineModule(BaseModule):
    """
    Tier‑0 baseline story_structure.
    """

    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    def __init__(self, rs_path: Optional[str] = None, max_frames: int = 200, **kwargs: Any):
        super().__init__(rs_path=rs_path, logger_name=self.module_name, **kwargs)
        self._max_frames = int(max_frames)
        self._last_models_used: List[Dict[str, Any]] = []

    def required_dependencies(self) -> List[str]:
        return ["core_clip", "core_optical_flow", "core_face_landmarks"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(self._last_models_used or [])

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not frame_indices:
            raise ValueError(f"{MODULE_NAME} | frame_indices is empty")
        if self.rs_path is None:
            raise ValueError(f"{MODULE_NAME} | rs_path is required")

        min_frames = int(config.get("min_frames", 30))
        max_frames = int(config.get("max_frames", self._max_frames))
        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        if int(fi.size) < int(min_frames):
            raise RuntimeError(f"{MODULE_NAME} | too few frames: N={int(fi.size)} < min_frames={int(min_frames)} (no-fallback)")
        if max_frames > 0 and int(fi.size) > int(max_frames):
            raise RuntimeError(
                f"{MODULE_NAME} | too many frames: N={int(fi.size)} > max_frames={int(max_frames)} (no-fallback). "
                "Fix Segmenter sampling for story_structure."
            )

        times_s = _require_union_times_s(frame_manager, fi)
        dt = np.diff(times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        video_len_s = float(times_s[-1] - times_s[0]) if times_s.size >= 2 else 0.0
        t_deps = time.perf_counter()

        emb, mu_clip = _load_core_clip_embeddings_aligned(self.rs_path, fi)
        emb_n = _normalize_rows(np.asarray(emb, dtype=np.float32))
        motion, mu_flow = _load_core_optical_flow_aligned(self.rs_path, fi)
        any_face, mu_face, face_meta = _load_core_face_any_present_aligned(self.rs_path, fi)

        # store combined models_used for meta/model_signature
        self._last_models_used = []
        self._last_models_used.extend(mu_clip)
        self._last_models_used.extend(mu_flow)
        self._last_models_used.extend(mu_face)

        t_curves0 = time.perf_counter()
        # embedding change rate (per-second)
        if emb_n.shape[0] >= 2:
            sim_next = np.sum(emb_n[1:] * emb_n[:-1], axis=1).astype(np.float32)
            diff_next = (1.0 - sim_next).astype(np.float32)
            diff_rate = (diff_next / dt).astype(np.float32)
        else:
            sim_next = np.asarray([], dtype=np.float32)
            diff_next = np.asarray([], dtype=np.float32)
            diff_rate = np.asarray([], dtype=np.float32)

        # align to frames (pad 0 at first frame)
        if diff_rate.size:
            emb_rate_curve = np.concatenate([np.asarray([0.0], dtype=np.float32), diff_rate], axis=0)
        else:
            emb_rate_curve = np.zeros((emb_n.shape[0],), dtype=np.float32)

        # motion curve is already per-second mean magnitude (core_optical_flow contract)
        motion_curve = np.asarray(motion, dtype=np.float32)
        if motion_curve.shape != emb_rate_curve.shape:
            raise RuntimeError(f"{MODULE_NAME} | motion curve shape mismatch after alignment")

        # Smooth and z-score component curves, then combine
        sigma = float(config.get("energy_smoothing_sigma", 1.0))
        sigma = max(0.0, sigma)
        emb_s = gaussian_filter1d(emb_rate_curve, sigma=sigma).astype(np.float32) if emb_rate_curve.size else emb_rate_curve
        mot_s = gaussian_filter1d(motion_curve, sigma=sigma).astype(np.float32) if motion_curve.size else motion_curve
        emb_z = _zscore(emb_s)
        mot_z = _zscore(mot_s)

        combined = (0.5 * emb_z + 0.5 * mot_z).astype(np.float32)
        combined_s = gaussian_filter1d(combined, sigma=sigma).astype(np.float32) if combined.size else combined
        story_energy_curve = _zscore(combined_s)
        t_curves1 = time.perf_counter()

        t_markers0 = time.perf_counter()
        # Hook window: min(5s, 15% of video). If sampling yields too few points, extend to cover at least 3 frames.
        hook_len_s = float(min(5.0, 0.15 * video_len_s)) if video_len_s > 0 else 0.0
        hook_end_t = float(times_s[0] + hook_len_s) if times_s.size else 0.0
        hook_mask = times_s <= hook_end_t if times_s.size else np.zeros((fi.size,), dtype=bool)
        if int(np.sum(hook_mask)) < 3 and fi.size >= 3:
            hook_mask = np.zeros((fi.size,), dtype=bool)
            hook_mask[:3] = True
            hook_end_t = float(times_s[min(2, times_s.size - 1)])

        hook_dur_s = float(max(hook_end_t - float(times_s[0]), 1e-6)) if times_s.size else 1e-6

        hook_emb = story_energy_curve[hook_mask] if story_energy_curve.size else np.asarray([], dtype=np.float32)
        hook_visual_surprise_score = float(np.mean(hook_emb)) if hook_emb.size else float("nan")
        hook_visual_surprise_std = float(np.std(hook_emb)) if hook_emb.size else float("nan")

        hook_motion = mot_s[hook_mask] if mot_s.size else np.asarray([], dtype=np.float32)
        if hook_motion.size:
            hook_motion_intensity = float(np.mean(hook_motion))
            p75 = float(np.percentile(hook_motion, 75))
            p90 = float(np.percentile(hook_motion, 90))
            cut_frames = hook_motion > p75
            spike_frames = hook_motion > p90
            hook_cut_rate = float(np.sum(cut_frames) / hook_dur_s)
            hook_motion_spikes = int(np.sum(spike_frames))
            hook_rhythm_score = float(
                (np.sum(hook_motion[spike_frames]) / (np.mean(hook_motion) + 1e-6)) if np.any(spike_frames) else 0.0
            )
        else:
            hook_motion_intensity = float("nan")
            hook_cut_rate = float("nan")
            hook_motion_spikes = 0
            hook_rhythm_score = float("nan")

        hook_face_presence = float(np.mean(any_face[hook_mask])) if any_face.size and np.any(hook_mask) else 0.0

        # climax = max energy
        if story_energy_curve.size:
            climax_pos = int(np.argmax(story_energy_curve))
            climax_frame = int(fi[climax_pos])
            climax_time = float(times_s[climax_pos])
            climax_strength = float(combined_s[climax_pos]) if combined_s.size else float("nan")
            climax_strength_z = float(story_energy_curve[climax_pos])
            climax_position_norm = float(climax_pos / max(len(fi) - 1, 1))
        else:
            climax_pos = -1
            climax_frame = -1
            climax_time = float("nan")
            climax_strength = float("nan")
            climax_strength_z = float("nan")
            climax_position_norm = float("nan")

        # peaks
        peaks_idx = np.asarray([], dtype=np.int32)
        peaks_times_s = np.asarray([], dtype=np.float32)
        peaks_values = np.asarray([], dtype=np.float32)
        if story_energy_curve.size >= 4:
            p90 = float(np.percentile(story_energy_curve, 90))
            peaks, props = find_peaks(story_energy_curve, height=p90)
            peaks_idx = np.asarray(peaks, dtype=np.int32)
            number_of_peaks = int(peaks_idx.size)
            if peaks_idx.size:
                peaks_times_s = np.asarray(times_s[peaks_idx], dtype=np.float32)
                peaks_values = np.asarray(story_energy_curve[peaks_idx], dtype=np.float32)
        else:
            number_of_peaks = 0
        t_markers1 = time.perf_counter()

        # time from hook to climax (normalized by video length)
        if video_len_s > 0 and np.isfinite(climax_time):
            if climax_time <= hook_end_t:
                time_from_hook_to_climax = 0.0
            else:
                time_from_hook_to_climax = float((climax_time - hook_end_t) / max(video_len_s, 1e-6))
        else:
            time_from_hook_to_climax = float("nan")

        # hook energy ratio (raw combined, not z)
        if combined_s.size and np.any(hook_mask):
            hook_energy = float(np.mean(combined_s[hook_mask]))
            avg_energy = float(np.mean(combined_s))
            hook_to_avg_energy_ratio = float(hook_energy / (avg_energy + 1e-6))
        else:
            hook_to_avg_energy_ratio = float("nan")

        # face-based global proxies (safe even if core_face_landmarks is empty)
        main_character_screen_time = float(np.mean(any_face)) if any_face.size else 0.0
        if any_face.size >= 2:
            switches = int(np.sum(np.diff(any_face.astype(np.int8)) != 0))
            speaker_switch_rate = float(switches / max(any_face.size - 1, 1))
            speaker_switches_per_minute = float(switches / max(video_len_s / 60.0, 1e-6)) if video_len_s > 0 else float("nan")
        else:
            speaker_switch_rate = float("nan")
            speaker_switches_per_minute = float("nan")

        # Text (baseline): OCR -> CLIP text (triton) topic shift curve
        t_text0 = time.perf_counter()
        text_mode = str(config.get("text_mode", "ocr_clip_text")).strip().lower()
        clip_text_model_spec = str(config.get("clip_text_model_spec", "clip_text_triton")).strip()
        clip_text_batch_size = int(config.get("clip_text_batch_size", 64))

        topic_shift_curve = np.full((int(fi.size),), np.nan, dtype=np.float32)
        topic_shift_curve_present = False
        topic_shift_peaks_idx = np.asarray([], dtype=np.int32)

        # Status policy from user: if OCR is missing/empty in OCR-mode -> status=empty
        meta_override: Dict[str, Any] = {"status": "ok", "empty_reason": None}
        ocr_reason = None
        if text_mode in ("ocr_clip_text", "ocr"):
            ocr_rows, ocr_reason = self._load_ocr_rows_optional(rs_path=str(self.rs_path))
            if not ocr_rows:
                meta_override["status"] = "empty"
                meta_override["empty_reason"] = str(ocr_reason or "dependency_missing")
            else:
                texts = self._texts_per_frame_from_ocr_rows(fi=fi, ocr_rows=ocr_rows, max_chars=int(config.get("ocr_max_chars_per_frame", 256)))
                topic_shift_curve, topic_shift_curve_present, topic_shift_peaks_idx = self._topic_shift_curve_from_texts(
                    times_s=times_s,
                    texts=texts,
                    clip_text_model_spec=clip_text_model_spec,
                    batch_size=clip_text_batch_size,
                    config=config,
                )
        t_text1 = time.perf_counter()

        features: Dict[str, Any] = {
            "n_frames": int(fi.size),
            "min_frames": int(min_frames),
            "max_frames": int(max_frames),
            "video_length_seconds": float(video_len_s),
            # hook
            "hook_visual_surprise_score": hook_visual_surprise_score,
            "hook_visual_surprise_std": hook_visual_surprise_std,
            "hook_motion_intensity": hook_motion_intensity,
            "hook_cut_rate": hook_cut_rate,
            "hook_motion_spikes": int(hook_motion_spikes),
            "hook_rhythm_score": hook_rhythm_score,
            "hook_face_presence": float(hook_face_presence),
            # climax
            "climax_timestamp": int(climax_frame),  # union-frame index
            "climax_time_sec": float(climax_time),
            "climax_position_normalized": float(climax_position_norm),
            "climax_strength": float(climax_strength),
            "climax_strength_normalized": float(climax_strength_z),
            "number_of_peaks": int(number_of_peaks),
            "time_from_hook_to_climax": float(time_from_hook_to_climax),
            "hook_to_avg_energy_ratio": float(hook_to_avg_energy_ratio),
            # character proxies
            "main_character_screen_time": float(main_character_screen_time),
            "speaker_switch_rate": float(speaker_switch_rate),
            "speaker_switches_per_minute": float(speaker_switches_per_minute),
            # text (OCR -> CLIP)
            "topic_shift_curve_present": bool(topic_shift_curve_present),
            "topic_shift_peaks_count": int(topic_shift_peaks_idx.size),
            "clip_text_model_spec": clip_text_model_spec if clip_text_model_spec else "unknown",
            # trace
            "core_face_landmarks_empty_reason": face_meta.get("empty_reason") if isinstance(face_meta, dict) else None,
            "ocr_empty_reason": str(ocr_reason) if ocr_reason is not None else None,
        }

        # ui_payload: keep it small (pointers + markers), no heavy arrays duplicated.
        ui_payload: Dict[str, Any] = {
            "schema_version": "story_structure_ui_v1",
            "curves": {
                "story_energy_curve": {"npz_key": "story_energy_curve", "label": "Story energy (z)"},
                "motion_norm_per_sec_mean": {"npz_key": "motion_norm_per_sec_mean", "label": "Motion (per-sec mean)"},
                "embedding_change_rate_per_sec": {"npz_key": "embedding_change_rate_per_sec", "label": "Embedding change rate (/s)"},
                "topic_shift_curve": {"npz_key": "topic_shift_curve", "label": "Topic shift (/s)"} if text_mode in ("ocr_clip_text", "ocr") else None,
                "any_face_present": {"npz_key": "any_face_present", "label": "Face present"} if True else None,
            },
            "markers": {
                "hook_window": {"t_start_s": float(times_s[0]) if times_s.size else 0.0, "t_end_s": float(hook_end_t)},
                "climax": {"t_s": float(climax_time), "frame_index": int(climax_frame), "strength_z": float(climax_strength_z)},
            },
            "peaks": {
                "energy": [
                    {"t_s": float(peaks_times_s[i]), "frame_index": int(fi[int(peaks_idx[i])]), "value_z": float(peaks_values[i])}
                    for i in range(int(peaks_idx.size))
                ]
            },
            "flags": {
                "topic_shift_present": bool(topic_shift_curve_present),
                "faces_any_present": bool(np.any(any_face)) if any_face.size else False,
            },
        }

        return {
            "summary": {
                "stage_timings_ms": {
                    "deps_ms": float((t_deps - t0) * 1000.0),
                    "compute_curves_ms": float((t_curves1 - t_curves0) * 1000.0),
                    "hooks_climax_ms": float((t_markers1 - t_markers0) * 1000.0),
                    "text_ms": float((t_text1 - t_text0) * 1000.0),
                }
            },
            "frame_indices": fi,
            "times_s": times_s.astype(np.float32),
            "embedding_sim_next": sim_next,
            "embedding_diff_next": diff_next,
            "embedding_change_rate_per_sec": emb_rate_curve.astype(np.float32),
            "motion_norm_per_sec_mean": mot_s.astype(np.float32),
            "any_face_present": np.asarray(any_face, dtype=bool),
            "story_energy_curve": story_energy_curve.astype(np.float32),
            "story_energy_curve_downsampled_128": _downsample_to_fixed(story_energy_curve, 128),
            "story_energy_peaks_idx": peaks_idx.astype(np.int32),
            "story_energy_peaks_times_s": peaks_times_s.astype(np.float32),
            "story_energy_peaks_values_z": peaks_values.astype(np.float32),
            "topic_shift_curve": topic_shift_curve.astype(np.float32),
            "topic_shift_curve_present": np.asarray(bool(topic_shift_curve_present)),
            "topic_shift_peaks_idx": topic_shift_peaks_idx.astype(np.int32),
            "features": features,
            "ui_payload": ui_payload,
            "__meta_override__": meta_override,
        }

    def run(self, frames_dir: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Override BaseModule.run to:
        - write progress events (state_events.jsonl)
        - attach ui_payload into NPZ meta (meta.ui_payload), not as a top-level NPZ key
        - add stage timings into results["summary"]["stage_timings_ms"]
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
        fm = None
        try:
            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="load_deps")
            fm = self.create_frame_manager(frames_dir, metadata)
            t_fm = time.perf_counter()

            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="compute_curves")
            results = self.process(frame_manager=fm, frame_indices=frame_indices, config=config or {})
            t_proc = time.perf_counter()
            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="hooks_climax")

            # Move ui_payload into meta
            ui_payload = None
            if isinstance(results, dict) and "ui_payload" in results:
                try:
                    ui_payload = results.pop("ui_payload")
                except Exception:
                    ui_payload = None

            # Apply __meta_override__ (status/empty_reason)
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
            }
            if isinstance(meta_override, dict):
                for k, v in meta_override.items():
                    if isinstance(k, str) and k and (isinstance(v, (str, int, float, bool)) or v is None):
                        save_metadata[k] = v

            # stage timings into results["summary"]
            if isinstance(results, dict):
                summ = results.get("summary")
                if not isinstance(summ, dict):
                    summ = {}
                    results["summary"] = summ
                st = summ.get("stage_timings_ms") if isinstance(summ.get("stage_timings_ms"), dict) else {}
                st["load_deps_ms"] = float((t_fm - t0) * 1000.0)
                st["process_ms"] = float((t_proc - t_fm) * 1000.0)
                st["total_ms"] = float((t_proc - t0) * 1000.0)
                summ["stage_timings_ms"] = st

            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=total, total=total, stage="save")
            t_save0 = time.perf_counter()
            out_path = self.save_results(results=results, metadata=save_metadata)
            t_save1 = time.perf_counter()
            try:
                if isinstance(results, dict):
                    summ3 = results.get("summary")
                    if isinstance(summ3, dict):
                        st3 = summ3.get("stage_timings_ms") if isinstance(summ3.get("stage_timings_ms"), dict) else {}
                        st3["save_ms"] = float((t_save1 - t_save0) * 1000.0)
                        st3["total_ms"] = float((t_save1 - t0) * 1000.0)
                        summ3["stage_timings_ms"] = st3
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

    @staticmethod
    def _load_npz_dict(path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        z = np.load(path, allow_pickle=True)
        out: Dict[str, Any] = {}
        for k in z.files:
            v = z[k]
            out[k] = _unbox_object_scalar(v)
        return out

    @staticmethod
    def _find_ocr_npz_paths(rs_path: str) -> List[str]:
        # canonical first, then compatibility locations used by other components
        return [
            os.path.join(rs_path, "ocr_extractor", "ocr.npz"),
            os.path.join(rs_path, "text_ocr", "ocr.npz"),
            os.path.join(rs_path, "ocr", "ocr.npz"),
            os.path.join(rs_path, "text_scoring", "ocr.npz"),
        ]

    def _load_ocr_rows_optional(self, rs_path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Returns (rows, empty_reason_or_none).
        Policy: missing/empty OCR is a valid empty for story_structure (user choice).
        """
        for p in self._find_ocr_npz_paths(rs_path):
            if not os.path.isfile(p):
                continue
            try:
                d = self._load_npz_dict(p)
                meta = d.get("meta")
                if isinstance(meta, dict):
                    st = str(meta.get("status") or "")
                    if st == "empty":
                        return [], str(meta.get("empty_reason") or "no_text_available")
                rows = d.get("ocr_raw")
                if rows is None:
                    return [], "no_text_available"
                rows_list = [x for x in np.asarray(rows, dtype=object).tolist() if isinstance(x, dict)]
                if not rows_list:
                    return [], "no_text_available"
                return rows_list, None
            except Exception:
                return [], "dependency_missing"
        return [], "dependency_missing"

    @staticmethod
    def _texts_per_frame_from_ocr_rows(*, fi: np.ndarray, ocr_rows: List[Dict[str, Any]], max_chars: int) -> List[str]:
        by_frame: Dict[int, List[str]] = {}
        for r in ocr_rows:
            try:
                fr = int(r.get("frame"))
                txt = str(r.get("text_norm") or r.get("text_raw") or "").strip()
                if not txt:
                    continue
                by_frame.setdefault(fr, []).append(txt)
            except Exception:
                continue

        out: List[str] = []
        for fr in fi.tolist():
            toks = by_frame.get(int(fr), [])
            if not toks:
                out.append("")
                continue
            # de-dup while preserving order
            seen = set()
            uniq: List[str] = []
            for t in toks:
                if t in seen:
                    continue
                seen.add(t)
                uniq.append(t)
            s = " | ".join(uniq)
            s = s[: max(0, int(max_chars))]
            out.append(s)
        return out

    @staticmethod
    def _load_triton_spec_via_model_manager(model_spec_name: str, triton_http_url: Optional[str] = None) -> dict:
        import os
        from dp_models import get_global_model_manager, ModelManagerError  # type: ignore

        # Get triton_http_url from parameter, env var, or config
        if not triton_http_url:
            triton_http_url = os.environ.get("TRITON_HTTP_URL")
        
        mm = get_global_model_manager()
        try:
            rm = mm.get(model_name=str(model_spec_name))
            rp = rm.spec.runtime_params or {}
            handle = rm.handle or {}
            client = None
            if isinstance(handle, dict):
                client = handle.get("client")
            
            # If client is None or rp doesn't have triton_http_url, try to create from env/config
            if client is None or not rp.get("triton_http_url"):
                if triton_http_url:
                    from dp_triton import TritonHttpClient, TritonError  # type: ignore
                    client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
                    if not client.ready():
                        raise TritonError(
                            f"{MODULE_NAME} | Triton is not ready at {triton_http_url}",
                            error_code="triton_unavailable",
                        )
                    # Update runtime_params with triton_http_url and ensure default params are set
                    if not isinstance(rp, dict):
                        rp = {}
                    rp["triton_http_url"] = str(triton_http_url)
                    # Ensure default parameters are set if missing (from clip_text_triton.yaml spec)
                    if not rp.get("model_name"):
                        rp["model_name"] = "clip_text"
                    if not rp.get("model_version"):
                        rp["model_version"] = "1"
                    if not rp.get("input_name"):
                        rp["input_name"] = "INPUT__0"
                    if not rp.get("output_name"):
                        rp["output_name"] = "OUTPUT__0"
                    if not rp.get("datatype"):
                        rp["datatype"] = "INT64"
                    models_used_entry = rm.models_used_entry if hasattr(rm, 'models_used_entry') else None
                else:
                    raise RuntimeError(f"{MODULE_NAME} | ModelManager returned empty Triton client handle for: {model_spec_name} and triton_http_url not provided (set --triton-http-url or TRITON_HTTP_URL env var)")
            else:
                # Use triton_http_url from runtime_params if available
                if not triton_http_url and rp.get("triton_http_url"):
                    triton_http_url = str(rp.get("triton_http_url"))
                models_used_entry = rm.models_used_entry if hasattr(rm, 'models_used_entry') else None
            
            if client is None:
                raise RuntimeError(f"{MODULE_NAME} | Failed to create Triton client for: {model_spec_name}")
            if not isinstance(rp, dict) or not rp:
                raise RuntimeError(f"{MODULE_NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
            return {"client": client, "rp": rp, "models_used_entry": models_used_entry}
        except ModelManagerError as e:
            # If ModelManager fails but we have triton_http_url, create client directly with default params
            if triton_http_url:
                # This is expected when spec uses ${TRITON_HTTP_URL} - ModelManager doesn't expand env vars during validation
                # We handle it gracefully with fallback
                from dp_triton import TritonHttpClient, TritonError  # type: ignore
                client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
                if not client.ready():
                    raise TritonError(
                        f"{MODULE_NAME} | Triton is not ready at {triton_http_url}",
                        error_code="triton_unavailable",
                    )
                # Use default parameters for clip_text model (from spec_catalog/vision/clip_text_triton.yaml)
                rp = {
                    "triton_http_url": str(triton_http_url),
                    "model_name": "clip_text",
                    "model_version": "1",
                    "input_name": "INPUT__0",
                    "output_name": "OUTPUT__0",
                    "datatype": "INT64",
                }
                return {"client": client, "rp": rp, "models_used_entry": None}
            raise RuntimeError(f"{MODULE_NAME} | ModelManager failed for {model_spec_name}: {e} and triton_http_url not provided")

    @staticmethod
    def _triton_infer(*, client, model_name: str, model_version: Optional[str], input_name: str, input_tensor: np.ndarray, output_name: str, datatype: str) -> np.ndarray:
        res = client.infer(
            model_name=str(model_name),
            model_version=str(model_version) if model_version else None,
            input_name=str(input_name),
            input_tensor=input_tensor,
            output_name=str(output_name),
            datatype=str(datatype),
        )
        return np.asarray(res.output)

    def _compute_clip_text_embeddings_triton(self, *, texts: List[str], clip_text_model_spec: str, batch_size: int, triton_http_url: Optional[str] = None) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Returns (B,512) float32 L2-normalized embeddings. Also returns models_used_entry (if available).
        """
        import clip  # type: ignore

        txt_mm = self._load_triton_spec_via_model_manager(str(clip_text_model_spec), triton_http_url=triton_http_url)
        rp = txt_mm.get("rp") if isinstance(txt_mm, dict) else None
        client = txt_mm.get("client") if isinstance(txt_mm, dict) else None
        if not isinstance(rp, dict) or client is None:
            raise RuntimeError(f"{MODULE_NAME} | invalid triton spec handle for: {clip_text_model_spec}")
        model_name = str(rp.get("model_name") or "clip_text")
        model_version = rp.get("model_version")
        input_name = str(rp.get("input_name") or "input")
        output_name = str(rp.get("output_name") or "output")
        datatype = str(rp.get("datatype") or "INT64")

        if not texts:
            return np.zeros((0, 512), dtype=np.float32), txt_mm.get("models_used_entry") if isinstance(txt_mm, dict) else None

        toks = clip.tokenize(texts)  # (B,77) int64
        toks_np = np.asarray(toks.cpu().numpy(), dtype=np.int64)
        out_list: List[np.ndarray] = []
        bs = max(1, int(batch_size))
        for i in range(0, int(toks_np.shape[0]), bs):
            chunk = toks_np[i : i + bs]
            arr = self._triton_infer(
                client=client,
                model_name=model_name,
                model_version=str(model_version) if model_version else None,
                input_name=input_name,
                input_tensor=chunk,
                output_name=output_name,
                datatype=datatype,
            )
            arr = np.asarray(arr, dtype=np.float32)
            # Backward compatible: handle different output shapes
            # Case 1: (B, 512) - already pooled embeddings
            if arr.ndim == 2 and arr.shape[0] == chunk.shape[0] and arr.shape[1] == 512:
                pass  # Already correct shape
            # Case 2: (B, 1, 512) - single token embedding
            elif arr.ndim == 3 and int(arr.shape[1]) == 1:
                arr = arr[:, 0, :]
            # Case 3: (B, 77, 512) - per-token embeddings, need to extract EOT position
            elif arr.ndim == 3 and arr.shape[0] == chunk.shape[0] and arr.shape[1] == 77 and arr.shape[2] == 512:
                # EOT position per row: OpenAI CLIP uses a large token id for EOT (argmax works)
                eot_pos = np.argmax(chunk, axis=1).astype(np.int64)
                eot_pos = np.clip(eot_pos, 0, 76)
                rows = np.arange(chunk.shape[0], dtype=np.int64)
                arr = arr[rows, eot_pos, :]  # (B, 512)
            else:
                raise RuntimeError(f"{MODULE_NAME} | clip_text output has invalid shape: {arr.shape} (expected (B,512), (B,1,512), or (B,77,512))")
            # Final validation
            if arr.ndim != 2:
                raise RuntimeError(f"{MODULE_NAME} | clip_text output has invalid shape after processing: {arr.shape}")
            if arr.shape[0] != chunk.shape[0]:
                raise RuntimeError(f"{MODULE_NAME} | clip_text batch mismatch: out_B={arr.shape[0]} in_B={chunk.shape[0]}")
            if int(arr.shape[1]) != 512:
                raise RuntimeError(f"{MODULE_NAME} | clip_text output has invalid D (expected 512): {arr.shape}")
            out_list.append(arr)
        emb = np.concatenate(out_list, axis=0).astype(np.float32) if out_list else np.zeros((0, 512), dtype=np.float32)
        emb = _normalize_rows(emb) if emb.size else emb
        return emb, txt_mm.get("models_used_entry") if isinstance(txt_mm, dict) else None

    def _topic_shift_curve_from_texts(
        self,
        *,
        times_s: np.ndarray,
        texts: List[str],
        clip_text_model_spec: str,
        batch_size: int,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        Compute topic_shift_curve aligned to frames:
        - For frames with OCR text -> CLIP text embedding
        - Topic shift = (1 - cosine_sim(prev, cur)) / dt
        Returns: (curve (N,), present_flag, peaks_idx)
        """
        N = int(len(texts))
        if N == 0:
            return np.zeros((0,), dtype=np.float32), False, np.asarray([], dtype=np.int32)

        present = np.asarray([bool(t.strip()) for t in texts], dtype=bool)
        if not np.any(present):
            return np.full((N,), np.nan, dtype=np.float32), False, np.asarray([], dtype=np.int32)

        texts_present = [texts[i] for i in range(N) if bool(present[i])]
        triton_http_url = config.get("triton_http_url") if config else None
        emb, mm_entry = self._compute_clip_text_embeddings_triton(texts=texts_present, clip_text_model_spec=clip_text_model_spec, batch_size=batch_size, triton_http_url=triton_http_url)
        if mm_entry and isinstance(mm_entry, dict):
            existing_names = {str(m.get("model_name") or "") for m in (self._last_models_used or []) if isinstance(m, dict)}
            if str(mm_entry.get("model_name") or "") not in existing_names:
                self._last_models_used.append(mm_entry)

        full = np.full((N, int(emb.shape[1] if emb.ndim == 2 else 512)), np.nan, dtype=np.float32)
        j = 0
        for i in range(N):
            if not present[i]:
                continue
            full[i] = emb[j]
            j += 1

        curve = np.full((N,), np.nan, dtype=np.float32)
        if N >= 2:
            dt = np.diff(times_s).astype(np.float32)
            dt = np.maximum(dt, 1e-3)
            for i in range(1, N):
                if not (present[i] and present[i - 1]):
                    continue
                sim = float(np.sum(full[i] * full[i - 1]))
                curve[i] = float((1.0 - sim) / float(dt[i - 1]))

        peaks_idx = np.asarray([], dtype=np.int32)
        try:
            finite = curve[np.isfinite(curve)]
            if finite.size >= 4:
                thr = float(np.percentile(finite, 90))
                peaks, _ = find_peaks(np.nan_to_num(curve, nan=-1e9), height=thr)
                peaks_idx = np.asarray(peaks, dtype=np.int32)
        except Exception:
            peaks_idx = np.asarray([], dtype=np.int32)

        return curve.astype(np.float32), True, peaks_idx


