"""
uniqueness (Tier‑0 baseline)

Baseline mode:
- No reference videos.
- Computes intra-video repetition/diversity proxies from `core_clip` embeddings on sampled frames.

Contract:
- `frame_indices` come strictly from Segmenter metadata (union-domain).
- time-axis strictly from `union_timestamps_sec` (per-second temporal change).
- hard dependency: `core_clip/embeddings.npz` must fully cover this module's `frame_indices` (no-fallback).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

MODULE_NAME = "uniqueness"
VERSION = "1.0.2"
SCHEMA_VERSION = "uniqueness_npz_v4"


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
    "repeat_threshold_is_otsu",
    "repeat_threshold_used",
    "repeat_threshold_raw",
    "repeat_threshold_quality",
    "repeat_threshold_min",
    "repeat_threshold_max",
    "repeat_threshold_bins",
    "max_frames",
    "repetition_ratio",
    "max_sim_to_other_mean",
    "max_sim_to_other_p95",
    "pairwise_sim_mean",
    "pairwise_sim_p95",
    "cos_dist_next_mean",
    "cos_dist_next_p95",
    "temporal_change_mean",
    "diversity_score",
    "effective_unique_frames",
    "effective_unique_ratio",
    "n_frames",
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


def _otsu_threshold_and_quality_0_1(x: np.ndarray, bins: int = 128) -> Tuple[float, float]:
    """
    Otsu threshold for values assumed in [0,1].
    Returns (threshold in [0,1], quality in [0,1] approx).
    """
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return 0.97, 0.0
    v = np.clip(v, 0.0, 1.0)
    hist, bin_edges = np.histogram(v, bins=int(bins), range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    if float(hist.sum()) <= 0:
        return 0.97, 0.0
    p = hist / float(hist.sum())
    omega = np.cumsum(p)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 1e-12] = np.nan
    sigma_b2 = ((mu_t * omega - mu) ** 2) / denom
    idx = int(np.nanargmax(sigma_b2))
    thr = float(centers[idx])
    # quality: between-class variance ratio (0..1-ish); 0 if distribution is flat.
    try:
        sb2 = float(np.nanmax(sigma_b2))
        vt = float(np.var(v.astype(np.float64)))
        q = float(sb2 / (vt + 1e-12)) if vt > 0 else 0.0
        q = float(np.clip(q, 0.0, 1.0))
    except Exception:
        q = 0.0
    return thr, q

def _unbox_object_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _require_union_times_s(frame_manager: FrameManager, frame_indices: np.ndarray) -> np.ndarray:
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("uniqueness | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("uniqueness | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)

    if frame_indices.size == 0:
        raise RuntimeError("uniqueness | frame_indices is empty (no-fallback)")
    if int(np.max(frame_indices)) >= int(uts.shape[0]):
        raise RuntimeError("uniqueness | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[frame_indices.astype(np.int32)]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("uniqueness | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _load_core_clip_embeddings_aligned(
    rs_path: str, want_frame_indices: np.ndarray
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load core_clip embeddings and align to requested frame_indices (union-domain).
    Requires full coverage (no gaps). No fallback.
    Returns (embeddings_aligned, core_clip_models_used_best_effort).
    """
    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"uniqueness | missing core_clip embeddings: {core_path}")
    data = np.load(core_path, allow_pickle=True)
    core_idx = data.get("frame_indices")
    core_emb = data.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError("uniqueness | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)

    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in want_frame_indices.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            "uniqueness | core_clip does not cover requested frame_indices. "
            "Segmenter must provide consistent indices across core_clip and this module."
        )

    # Best-effort: read upstream models_used for reproducibility.
    models_used: List[Dict[str, Any]] = []
    meta = _unbox_object_scalar(data.get("meta"))
    if isinstance(meta, dict):
        mu = meta.get("models_used")
        if isinstance(mu, list):
            models_used = [x for x in mu if isinstance(x, dict)]

    return core_emb[np.asarray(pos, dtype=np.int64)], models_used


class UniquenessBaselineModule(BaseModule):
    """
    Baseline version of uniqueness:
    - No external reference videos.
    - Computes intra-video repetition/diversity proxies using `core_clip` embeddings.
    
    Batch processing support:
    - Batch-safe: uses per-video rs_path (no shared mutable state between videos).
    - Default process_batch() from BaseModule handles sequential processing of multiple videos.
    - No GPU batching required (CPU-only module).
    """

    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    # Baseline: fixed artifact filename (run_id already provides uniqueness in path).
    ARTIFACT_FILENAME = "uniqueness.npz"

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    def __init__(
        self,
        rs_path: Optional[str] = None,
        repeat_threshold: float = 0.97,
        max_frames: int = 200,
        **kwargs: Any,
    ):
        super().__init__(rs_path=rs_path, logger_name=MODULE_NAME, **kwargs)
        self._repeat_threshold = float(repeat_threshold)
        self._max_frames = int(max_frames)
        self._last_core_clip_models_used: List[Dict[str, Any]] = []

    def required_dependencies(self) -> List[str]:
        return ["core_clip"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # This module does not run ML models itself; include upstream core_clip model signature for reproducibility.
        return list(self._last_core_clip_models_used or [])

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()

        if not frame_indices:
            raise ValueError("uniqueness | frame_indices is empty")
        if self.rs_path is None:
            raise ValueError("uniqueness | rs_path is required")

        thr_mode = str(config.get("repeat_threshold_mode", "auto")).strip().lower()
        thr_min = float(config.get("repeat_threshold_min", 0.90))
        thr_max = float(config.get("repeat_threshold_max", 0.99))
        repeat_thr_fixed = float(config.get("repeat_threshold", self._repeat_threshold))
        thr_bins = int(config.get("repeat_threshold_bins", 128))
        max_frames = int(config.get("max_frames", self._max_frames))
        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)

        if max_frames > 0 and int(fi.size) > int(max_frames):
            raise RuntimeError(
                f"uniqueness | too many frames for NxN similarity: N={int(fi.size)} > max_frames={int(max_frames)}. "
                "Fix Segmenter sampling for uniqueness (no-fallback)."
            )

        times_s = _require_union_times_s(frame_manager, fi)
        emb, core_clip_models_used = _load_core_clip_embeddings_aligned(self.rs_path, fi)
        self._last_core_clip_models_used = core_clip_models_used

        if emb.ndim != 2 or emb.shape[0] != fi.shape[0]:
            raise RuntimeError("uniqueness | invalid embeddings shape after alignment")

        emb_n = _normalize_rows(emb)
        n = int(emb_n.shape[0])

        # Pairwise similarity matrix (N x N)
        sim = emb_n @ emb_n.T
        np.fill_diagonal(sim, -np.inf)

        max_sim_other = np.max(sim, axis=1).astype(np.float32) if n > 0 else np.asarray([], dtype=np.float32)
        # Auto-calibrate repeat threshold from distribution (Otsu) with safety clamp.
        if thr_mode in ("auto", "otsu"):
            thr_raw, thr_quality = _otsu_threshold_and_quality_0_1(max_sim_other, bins=int(thr_bins))
            repeat_thr_used = float(np.clip(thr_raw, min(thr_min, thr_max), max(thr_min, thr_max)))
            thr_mode_used = "otsu"
        else:
            thr_raw = float("nan")
            thr_quality = float("nan")
            repeat_thr_used = float(repeat_thr_fixed)
            thr_mode_used = "fixed"

        repetition_ratio = float(np.mean(max_sim_other >= repeat_thr_used)) if n > 0 else float("nan")

        max_sim_mean = float(np.mean(max_sim_other)) if max_sim_other.size else float("nan")
        max_sim_p95 = float(np.percentile(max_sim_other, 95)) if max_sim_other.size else float("nan")

        if n >= 2:
            iu = np.triu_indices(n, k=1)
            sim_ut = (emb_n @ emb_n.T)[iu].astype(np.float32)
            pairwise_sim_mean = float(np.mean(sim_ut))
            pairwise_sim_p95 = float(np.percentile(sim_ut, 95))
        else:
            sim_ut = np.asarray([], dtype=np.float32)
            pairwise_sim_mean = float("nan")
            pairwise_sim_p95 = float("nan")

        if n >= 2:
            cos_sim_next = np.sum(emb_n[1:] * emb_n[:-1], axis=1).astype(np.float32)
            cos_dist_next = (1.0 - cos_sim_next).astype(np.float32)
            dt = np.diff(times_s).astype(np.float32)
            dt = np.maximum(dt, 1e-3)
            change_rate = (cos_dist_next / dt).astype(np.float32)
            temporal_change_mean = float(np.mean(change_rate))
        else:
            cos_dist_next = np.asarray([], dtype=np.float32)
            temporal_change_mean = float("nan")

        cos_dist_mean = float(np.mean(cos_dist_next)) if cos_dist_next.size else float("nan")
        cos_dist_p95 = float(np.percentile(cos_dist_next, 95)) if cos_dist_next.size else float("nan")

        diversity_score = float(
            np.clip(
                1.0 - (pairwise_sim_mean if not np.isnan(pairwise_sim_mean) else 0.0),
                0.0,
                1.0,
            )
        )

        effective_unique_frames = int(np.sum(max_sim_other < float(repeat_thr_used))) if max_sim_other.size else 0
        effective_unique_ratio = float(effective_unique_frames / max(n, 1)) if n > 0 else float("nan")

        scalar_feats: Dict[str, Any] = {
            "repeat_threshold_is_otsu": bool(thr_mode_used == "otsu"),
            "repeat_threshold_used": float(repeat_thr_used),
            "repeat_threshold_raw": float(thr_raw),
            "repeat_threshold_quality": float(thr_quality),
            "repeat_threshold_min": float(thr_min),
            "repeat_threshold_max": float(thr_max),
            "repeat_threshold_bins": int(thr_bins),
            "max_frames": int(max_frames),
            "repetition_ratio": float(repetition_ratio),
            "max_sim_to_other_mean": float(max_sim_mean),
            "max_sim_to_other_p95": float(max_sim_p95),
            "pairwise_sim_mean": float(pairwise_sim_mean),
            "pairwise_sim_p95": float(pairwise_sim_p95),
            "cos_dist_next_mean": float(cos_dist_mean),
            "cos_dist_next_p95": float(cos_dist_p95),
            "temporal_change_mean": float(temporal_change_mean),
            "diversity_score": float(diversity_score),
            "effective_unique_frames": int(effective_unique_frames),
            "effective_unique_ratio": float(effective_unique_ratio),
            "n_frames": int(n),
        }

        feature_names = np.asarray(list(_FEATURE_NAMES_V1), dtype=object)
        feature_values = np.asarray([_as_float_feature(scalar_feats.get(k)) for k in _FEATURE_NAMES_V1], dtype=np.float32).reshape(-1)

        # ui_payload (lightweight, pointers only + top repeats)
        ui_topk = int(config.get("ui_topk", 8))
        ui_topk = max(0, min(int(ui_topk), int(n)))
        top_items = []
        top_unique = []
        if ui_topk > 0 and max_sim_other.size:
            order = np.argsort(-max_sim_other)[:ui_topk]
            for idx in order.tolist():
                top_items.append(
                    {
                        "rank": int(len(top_items) + 1),
                        "i": int(idx),
                        "frame_index": int(fi[int(idx)]),
                        "t_s": float(times_s[int(idx)]),
                        "max_sim_to_other": float(max_sim_other[int(idx)]),
                    }
                )
            order2 = np.argsort(max_sim_other)[:ui_topk]
            for idx in order2.tolist():
                top_unique.append(
                    {
                        "rank": int(len(top_unique) + 1),
                        "i": int(idx),
                        "frame_index": int(fi[int(idx)]),
                        "t_s": float(times_s[int(idx)]),
                        "max_sim_to_other": float(max_sim_other[int(idx)]),
                    }
                )

        return {
            "frame_indices": fi,
            "times_s": np.asarray(times_s, dtype=np.float32),
            "max_sim_to_other": max_sim_other,
            "cos_dist_next": cos_dist_next,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "ui_payload": {
                "schema_version": "uniqueness_ui_v1",
                "curves": {
                    "max_sim_to_other": {"npz_key": "max_sim_to_other", "label": "Max similarity to any other frame"},
                    "cos_dist_next": {"npz_key": "cos_dist_next", "label": "Cosine distance to next frame"},
                },
                "top_repeats": top_items,
                "top_unique": top_unique,
                "repeat_threshold_mode": str(thr_mode_used),
            },
        }

    def run(self, frames_dir: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Override BaseModule.run to:
        - write progress events (state_events.jsonl)
        - attach ui_payload into NPZ meta (meta.ui_payload), not as a top-level NPZ key
        - add stage timings
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

            _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=total, stage="compute")
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
                "repeat_threshold_mode": str((config or {}).get("repeat_threshold_mode", "auto")),
                "repeat_threshold": float((config or {}).get("repeat_threshold", self._repeat_threshold)),
                "repeat_threshold_min": float((config or {}).get("repeat_threshold_min", 0.90)),
                "repeat_threshold_max": float((config or {}).get("repeat_threshold_max", 0.99)),
                "repeat_threshold_bins": int((config or {}).get("repeat_threshold_bins", 128)),
                "ui_topk": int((config or {}).get("ui_topk", 8)),
                "max_frames": int((config or {}).get("max_frames", self._max_frames)),
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


