"""
Baseline similarity_metrics (wide baseline v1).

Production goals:
- Compare a video to a reference set (dp_models) across multiple modalities:
  visual (core_clip), editing/pacing (cut_detection + video_pacing), quality/style (shot_quality),
  audio (AudioProcessor artifacts), text embeddings (TextProcessor artifacts), emotion (optional; faces may be absent).
- Strict contracts: NPZ-only, fixed schema+filename, time-axis alignment, no-fallback coverage for required deps.
- UI: provide `meta.ui_payload` for graphs + top-K reference matches.

Important:
- Heavy / experimental similarity code is moved to `similarity_metrics_library.py` to avoid scipy/sklearn deps here.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

MODULE_NAME = "similarity_metrics"
VERSION = "2.0.2"
SCHEMA_VERSION = "similarity_metrics_npz_v3"
ARTIFACT_FILENAME = "results.npz"


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


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
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


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": _utc_iso_now(),
            "scope": "progress",
            "processor": "visual",
            "component": MODULE_NAME,
            "status": "running",
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x) + 1e-9)
    return (x / n).astype(np.float32)


def _times_s_from_union(*, frame_manager: FrameManager, frame_indices: np.ndarray) -> np.ndarray:
    uts = (frame_manager.meta or {}).get("union_timestamps_sec")
    if not isinstance(uts, list) or not uts:
        raise RuntimeError(f"{MODULE_NAME} | missing/invalid union_timestamps_sec (no-fallback)")
    uts = np.asarray(uts, dtype=np.float32).reshape(-1)
    if int(np.max(frame_indices)) >= int(uts.size) or int(np.min(frame_indices)) < 0:
        raise RuntimeError(f"{MODULE_NAME} | frame_indices out of bounds for union_timestamps_sec")
    return uts[frame_indices.astype(np.int64)].astype(np.float32)


def _load_core_clip_embeddings_aligned(rs_path: str, want_frame_indices: np.ndarray) -> np.ndarray:
    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"similarity_metrics | missing core_clip embeddings: {core_path}")
    data = np.load(core_path, allow_pickle=True)
    core_idx = data.get("frame_indices")
    core_emb = data.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError("similarity_metrics | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)

    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in want_frame_indices.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            "similarity_metrics | core_clip does not cover requested frame_indices. "
            "Segmenter must provide consistent indices across core_clip and this module."
        )
    return core_emb[np.asarray(pos, dtype=np.int64)]


def _load_core_clip_embeddings_with_indices(rs_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load core_clip embeddings and their frame_indices without enforcing any alignment.
    Used by similarity_metrics as a single source-of-truth for the time axis.
    """
    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"similarity_metrics | missing core_clip embeddings: {core_path}")
    data = np.load(core_path, allow_pickle=True)
    core_idx = data.get("frame_indices")
    core_emb = data.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError("similarity_metrics | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)
    if core_idx.ndim != 1 or core_emb.ndim != 2 or core_emb.shape[0] != core_idx.shape[0]:
        raise RuntimeError(
            f"similarity_metrics | core_clip embeddings invalid shapes: "
            f"frame_indices.shape={core_idx.shape}, frame_embeddings.shape={core_emb.shape}"
        )
    return core_idx, core_emb


def _extract_meta_models_used(npz_dict: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Best-effort extraction of models_used from loaded NPZ dicts (core providers typically store meta as dict/object).
    """
    if not isinstance(npz_dict, dict):
        return []
    meta = npz_dict.get("meta")
    if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
        try:
            meta = meta.item()
        except Exception:
            meta = None
    if not isinstance(meta, dict):
        return []
    mu = meta.get("models_used")
    if isinstance(mu, list):
        return [m for m in mu if isinstance(m, dict)]
    return []


_MODALITIES_ORDER: List[str] = ["clip", "audio_clap", "text", "pacing", "quality", "emotion", "overall"]


def _nan_topn_stats(x: np.ndarray, *, top_n: int) -> Dict[str, float]:
    """
    NaN-aware stats for reference similarity scores:
    - mean_topn: mean of top-N finite values
    - max: max finite
    - p10: 10th percentile of finite values
    """
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {"mean_topn": float("nan"), "max": float("nan"), "p10": float("nan")}
    v_sorted = np.sort(v)[::-1]
    k = int(min(max(int(top_n), 1), int(v_sorted.size)))
    topk = v_sorted[:k]
    return {
        "mean_topn": float(np.mean(topk)),
        "max": float(v_sorted[0]),
        "p10": float(np.percentile(v, 10)),
    }


def _build_fixed_feature_vector(
    *,
    base_feats: Dict[str, Any],
    reference_present: bool,
    sims_by_modality: Dict[str, np.ndarray],
    present_by_modality: Dict[str, bool],
    top_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build stable `feature_names/feature_values` for models.
    All features are present; missing modalities => NaN metrics + present_flag=0.
    """
    # base coherence feats (always)
    names: List[str] = [
        "n_frames",
        "centroid_sim_mean",
        "centroid_sim_std",
        "centroid_sim_p10",
        "centroid_sim_p90",
        "temporal_sim_mean",
        "temporal_sim_std",
    ]
    vals: List[float] = [
        float(base_feats.get("n_frames", float("nan"))),
        float(base_feats.get("centroid_sim_mean", float("nan"))),
        float(base_feats.get("centroid_sim_std", float("nan"))),
        float(base_feats.get("centroid_sim_p10", float("nan"))),
        float(base_feats.get("centroid_sim_p90", float("nan"))),
        float(base_feats.get("temporal_sim_mean", float("nan"))),
        float(base_feats.get("temporal_sim_std", float("nan"))),
    ]

    # reference presence (as float for tabular consumers)
    names.append("reference_present_float")
    vals.append(1.0 if bool(reference_present) else 0.0)

    # per-modality present flags + stats
    ref_mean_topn_by_mod: Dict[str, float] = {}
    for mod in _MODALITIES_ORDER:
        pres = bool(present_by_modality.get(mod, False)) and bool(reference_present)
        names.append(f"modality_{mod}_present")
        vals.append(1.0 if pres else 0.0)

        st = {"mean_topn": float("nan"), "max": float("nan"), "p10": float("nan")}
        if pres and mod in sims_by_modality and isinstance(sims_by_modality.get(mod), np.ndarray):
            st = _nan_topn_stats(np.asarray(sims_by_modality[mod], dtype=np.float32), top_n=int(top_n))
        ref_mean_topn_by_mod[mod] = float(st["mean_topn"])

        names.append(f"reference_similarity_{mod}_mean_topn")
        vals.append(float(st["mean_topn"]))
        names.append(f"reference_similarity_{mod}_max")
        vals.append(float(st["max"]))
        names.append(f"reference_similarity_{mod}_p10")
        vals.append(float(st["p10"]))

    # Uniqueness: prefer overall if present, else clip
    uniq = float("nan")
    if bool(reference_present):
        base_sim = ref_mean_topn_by_mod.get("overall")
        if not np.isfinite(base_sim):
            base_sim = ref_mean_topn_by_mod.get("clip")
        if base_sim is not None and np.isfinite(base_sim):
            uniq = float(1.0 - float(base_sim))
    names.append("uniqueness_score")
    vals.append(float(uniq))
    names.append("uniqueness_clip")
    vals.append(float(1.0 - ref_mean_topn_by_mod["clip"]) if (bool(reference_present) and np.isfinite(ref_mean_topn_by_mod.get("clip", float("nan")))) else float("nan"))
    names.append("uniqueness_overall")
    vals.append(float(1.0 - ref_mean_topn_by_mod["overall"]) if (bool(reference_present) and np.isfinite(ref_mean_topn_by_mod.get("overall", float("nan")))) else float("nan"))

    return np.asarray(names, dtype=object), np.asarray(vals, dtype=np.float32).reshape(-1)


def _load_reference_embeddings_npz(path: str) -> np.ndarray:
    """
    Expected NPZ keys (baseline contract):
    - `video_embeddings` shape (M, D) float32 OR
    - `embeddings` shape (M, D) float32
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"similarity_metrics | reference embeddings npz not found: {path}")
    data = np.load(path, allow_pickle=True)
    emb = data.get("video_embeddings")
    if emb is None:
        emb = data.get("embeddings")
    if emb is None:
        raise RuntimeError("similarity_metrics | reference npz missing video_embeddings/embeddings")
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise RuntimeError("similarity_metrics | reference embeddings has invalid shape")
    return emb


def _load_npz_required(rs_path: str, component: str, fixed_name: Optional[str] = None) -> Dict[str, Any]:
    d = os.path.join(rs_path, component)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{MODULE_NAME} | missing dependency dir: {d}")
    if fixed_name:
        p = os.path.join(d, fixed_name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{MODULE_NAME} | missing dependency file: {p}")
        return dict(np.load(p, allow_pickle=True))
    # fallback: pick newest npz in dir (still fail-fast if none)
    npz_files = [os.path.join(d, x) for x in os.listdir(d) if x.lower().endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"{MODULE_NAME} | missing dependency npz in: {d}")
    npz_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dict(np.load(npz_files[0], allow_pickle=True))


def _load_npz_optional(rs_path: str, component: str, fixed_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        return _load_npz_required(rs_path, component, fixed_name=fixed_name)
    except Exception:
        return None


def _load_reference_pack_dir(*, dp_models_root: str, reference_set_id: str) -> str:
    # dp_models/bundled_models/similarity/reference_sets/<set_id>/...
    p = os.path.join(dp_models_root, "bundled_models", "similarity", "reference_sets", reference_set_id)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{MODULE_NAME} | reference set not found in dp_models: {p}")
    return p


def _load_reference_pack_v1(pack_dir: str) -> Dict[str, Any]:
    """
    Reference pack contract (wide baseline v1):
    - manifest.json: contains schema_version + ids
    - clip_video_embeddings.npy: (M, D)
    - clap_audio_embeddings.npy: (M, D)
    - text_primary_embeddings.npy: (M, D)
    - pacing_features.npy: (M, Kp)
    - shot_quality_features.npy: (M, Kq)
    - emotion_embeddings.npy: (M, Ke)  (may be NaN rows if faces missing, but file must exist)
    """
    man = os.path.join(pack_dir, "manifest.json")
    if not os.path.isfile(man):
        raise FileNotFoundError(f"{MODULE_NAME} | reference pack missing manifest.json: {man}")
    with open(man, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if str(manifest.get("schema_version")) != "similarity_reference_pack_v1":
        raise RuntimeError(f"{MODULE_NAME} | unsupported reference pack schema: {manifest.get('schema_version')}")

    def req_npy(name: str) -> np.ndarray:
        p = os.path.join(pack_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{MODULE_NAME} | reference pack missing file: {p}")
        return np.load(p, allow_pickle=True)

    ref_ids = np.asarray(manifest.get("reference_video_ids") or [], dtype=object)
    if ref_ids.size == 0:
        raise RuntimeError(f"{MODULE_NAME} | reference pack has empty reference_video_ids")

    clip = np.asarray(req_npy("clip_video_embeddings.npy"), dtype=np.float32)
    clap = np.asarray(req_npy("clap_audio_embeddings.npy"), dtype=np.float32)
    text = np.asarray(req_npy("text_primary_embeddings.npy"), dtype=np.float32)
    pacing = np.asarray(req_npy("pacing_features.npy"), dtype=np.float32)
    quality = np.asarray(req_npy("shot_quality_features.npy"), dtype=np.float32)
    emo = np.asarray(req_npy("emotion_embeddings.npy"), dtype=np.float32)

    M = int(ref_ids.size)
    for arr, nm in [(clip, "clip"), (clap, "clap"), (text, "text"), (pacing, "pacing"), (quality, "quality"), (emo, "emotion")]:
        if arr.ndim != 2 or int(arr.shape[0]) != M:
            raise RuntimeError(f"{MODULE_NAME} | reference pack invalid shape for {nm}: {arr.shape}, expected (M,*) with M={M}")

    return {
        "manifest": manifest,
        "reference_video_ids": ref_ids,
        "pacing_feature_keys": list(manifest.get("pacing_feature_keys") or []),
        "emotion_feature_keys": list(manifest.get("emotion_feature_keys") or []),
        "shot_quality_feature_names": list(manifest.get("shot_quality_feature_names") or []),
        "clip_video_embeddings": _normalize_rows(clip),
        "clap_audio_embeddings": _normalize_rows(clap),
        "text_primary_embeddings": _normalize_rows(text),
        "pacing_features": _normalize_rows(pacing),
        "shot_quality_features": _normalize_rows(quality),
        "emotion_embeddings": _normalize_rows(emo),
    }


class SimilarityBaselineModule(BaseModule):
    """
    Wide baseline v1:
    - intra-video coherence from core_clip
    - reference similarity across multiple modalities (dp_models reference pack)
    """

    @property
    def module_name(self) -> str:
        return "similarity_metrics"

    def __init__(
        self,
        rs_path: Optional[str] = None,
        top_n: int = 10,
        reference_embeddings_npz: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(rs_path=rs_path, logger_name=self.module_name, **kwargs)
        self._top_n = int(top_n)
        self._reference_embeddings_npz = reference_embeddings_npz

    MODULE_NAME = MODULE_NAME
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    def required_dependencies(self) -> List[str]:
        # Only core_clip is strictly required; audio (clap_extractor) is optional.
        return ["core_clip"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if not frame_indices:
            raise ValueError("similarity_metrics | frame_indices is empty")
        if self.rs_path is None:
            raise ValueError("similarity_metrics | rs_path is required")

        t0 = time.perf_counter()
        cfg = config or {}
        top_n = int(cfg.get("top_n", self._top_n))
        ui_topk = int(cfg.get("ui_topk") or 5)
        ui_topk = max(1, min(50, ui_topk))
        enable_overall_score = bool(cfg.get("enable_overall_score")) if "enable_overall_score" in cfg else False
        weights = cfg.get("overall_weights") if isinstance(cfg.get("overall_weights"), dict) else None

        platform_id = str((frame_manager.meta or {}).get("platform_id") or "")
        video_id = str((frame_manager.meta or {}).get("video_id") or "")
        run_id = str((frame_manager.meta or {}).get("run_id") or "")
        _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="start")

        # Audit v3 strict axis policy:
        # - Segmenter owns module axis (`frame_indices`), no-fallback.
        # - core_clip MUST have exactly the same frame_indices (strict equality) for coherent multimodal joins.
        requested = np.asarray(sorted({int(i) for i in frame_indices}), dtype=np.int32).reshape(-1)
        if requested.size == 0:
            raise ValueError("similarity_metrics | frame_indices is empty")

        core_idx, core_emb = _load_core_clip_embeddings_with_indices(self.rs_path)
        if core_idx.shape != requested.shape or not bool(np.all(core_idx.astype(np.int32) == requested)):
            raise RuntimeError(
                "similarity_metrics | frame_indices mismatch vs core_clip (strict). "
                "Segmenter must provide consistent indices across core_clip and this module."
            )

        fi = requested.astype(np.int32, copy=False)
        times_s = _times_s_from_union(frame_manager=frame_manager, frame_indices=fi)
        emb_n = _normalize_rows(np.asarray(core_emb, dtype=np.float32))

        # Intra-video coherence: similarity to centroid
        centroid = np.mean(emb_n, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        centroid_sims = (emb_n @ centroid).astype(np.float32)

        _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="coherence")

        features: Dict[str, Any] = {
            "n_frames": int(emb_n.shape[0]),
            "centroid_sim_mean": float(np.mean(centroid_sims)),
            "centroid_sim_std": float(np.std(centroid_sims)),
            "centroid_sim_p10": float(np.percentile(centroid_sims, 10)),
            "centroid_sim_p90": float(np.percentile(centroid_sims, 90)),
        }

        # Temporal coherence: consecutive similarity
        if emb_n.shape[0] >= 2:
            sim_next = np.sum(emb_n[1:] * emb_n[:-1], axis=1).astype(np.float32)
            features.update(
                {
                    "temporal_sim_mean": float(np.mean(sim_next)),
                    "temporal_sim_std": float(np.std(sim_next)),
                }
            )
        else:
            sim_next = np.asarray([], dtype=np.float32)
            features.update({"temporal_sim_mean": float("nan"), "temporal_sim_std": float("nan")})

        # Optional: reference similarity (centroid vs reference embeddings)
        _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="load_deps")

        # Audio (CLAP) — optional: if absent, we just skip audio modality.
        clap_npz = _load_npz_optional(str(self.rs_path), "clap_extractor", fixed_name="clap_extractor_features.npz")
        clap_vec = None
        if clap_npz is not None:
            clap_emb = clap_npz.get("embedding")
            if clap_emb is not None:
                clap_vec = _normalize_vec(np.asarray(clap_emb, dtype=np.float32))

        # Optional deps (faces/text/ocr): missing is allowed by policy.
        shot_npz = _load_npz_optional(str(self.rs_path), "shot_quality", fixed_name="shot_quality.npz")
        pacing_npz = _load_npz_optional(str(self.rs_path), "video_pacing", fixed_name="video_pacing_features.npz")
        text_npz = _load_npz_optional(str(self.rs_path), "text_processor", fixed_name="text_features.npz")
        micro_npz = _load_npz_optional(str(self.rs_path), "micro_emotion", fixed_name="micro_emotion.npz")

        # Build modality vectors (deterministic, no heuristics).
        vec_clip = centroid.astype(np.float32)

        # Text: prefer primary_embedding from TextProcessor NPZ (privacy-safe).
        vec_text = None
        text_present = False
        if text_npz is not None:
            meta = text_npz.get("meta")
            if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
                meta = meta.item()
            if isinstance(meta, dict):
                text_present = bool(meta.get("status") == "ok")
            if "primary_embedding_present" in text_npz and "primary_embedding" in text_npz:
                present = bool(np.asarray(text_npz.get("primary_embedding_present")).item())
                if present:
                    vec_text = _normalize_vec(np.asarray(text_npz.get("primary_embedding"), dtype=np.float32))
                    text_present = True

        # Pacing: small vector from video_pacing if present
        vec_pacing = None
        if pacing_npz is not None:
            fobj = pacing_npz.get("features")
            if isinstance(fobj, np.ndarray) and fobj.dtype == object and fobj.shape == ():
                fobj = fobj.item()
            if isinstance(fobj, dict):
                # Default: deterministic scalar subset, but if reference pack provides keys we will project later.
                scalar_items = {str(k): v for k, v in fobj.items() if isinstance(v, (int, float, np.floating, np.integer))}
                keys = sorted([k for k in scalar_items.keys() if k.startswith("pacing_") or k.startswith("shot_") or k.startswith("motion_")])[:64]
                if keys:
                    v = np.asarray([float(scalar_items.get(k, float("nan"))) for k in keys], dtype=np.float32)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    vec_pacing = _normalize_vec(v)

        # Shot quality: vectorize a subset of impl_meta/aggregates if present
        vec_quality = None
        if shot_npz is not None:
            feats_mean = shot_npz.get("shot_features_mean")
            if isinstance(feats_mean, np.ndarray) and feats_mean.ndim == 2 and feats_mean.size > 0:
                # aggregate across shots
                v = np.nanmean(np.asarray(feats_mean, dtype=np.float32), axis=0)
                if v.size > 0:
                    vec_quality = _normalize_vec(v)

        # Emotion: optional; if faces missing, it's ok to be absent/NaN.
        vec_emotion = None
        if micro_npz is not None:
            fobj = micro_npz.get("features")
            if isinstance(fobj, np.ndarray) and fobj.dtype == object and fobj.shape == ():
                fobj = fobj.item()
            if isinstance(fobj, dict) and fobj:
                # Stable, low-noise subset from micro_emotion video-level aggregates + reliability flags.
                want = [
                    "smile_ratio",
                    "eye_contact_ratio",
                    "blink_rate_per_min",
                    "pose_stability_score",
                    "face_presence_ratio",
                    "au_quality_overall",
                    "AU06_mean",
                    "AU12_mean",
                    "AU04_mean",
                    "AU25_mean",
                    "AU26_mean",
                    "AU07_mean",
                    "AU15_mean",
                    "AU06_peak_count",
                    "AU12_peak_count",
                    "AU04_peak_count",
                    "AU25_peak_count",
                    "AU26_peak_count",
                    "AU07_peak_count",
                    "AU15_peak_count",
                ]
                vec_emotion = _normalize_vec(np.asarray([float(fobj.get(k, float("nan"))) for k in want], dtype=np.float32))

        # Reference pack (strict if provided)
        reference_present = False
        ref_set_id = cfg.get("reference_set_id")
        dp_models_root = cfg.get("dp_models_root") or os.environ.get("DP_MODELS_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dp_models")
        ref_pack = None
        if ref_set_id:
            pack_dir = _load_reference_pack_dir(dp_models_root=str(dp_models_root), reference_set_id=str(ref_set_id))
            ref_pack = _load_reference_pack_v1(pack_dir)
            reference_present = True

            # If reference pack defines keys, rebuild optional vectors to match pack dims (strict).
            pfk = ref_pack.get("pacing_feature_keys") or []
            if pfk and pacing_npz is not None:
                fobj = pacing_npz.get("features")
                if isinstance(fobj, np.ndarray) and fobj.dtype == object and fobj.shape == ():
                    fobj = fobj.item()
                if isinstance(fobj, dict):
                    v = np.asarray([float(fobj.get(k, float("nan"))) for k in pfk], dtype=np.float32)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    vec_pacing = _normalize_vec(v)

            efk = ref_pack.get("emotion_feature_keys") or []
            if efk and micro_npz is not None:
                fobj = micro_npz.get("features")
                if isinstance(fobj, np.ndarray) and fobj.dtype == object and fobj.shape == ():
                    fobj = fobj.item()
                if isinstance(fobj, dict):
                    vec_emotion = _normalize_vec(np.asarray([float(fobj.get(k, float("nan"))) for k in efk], dtype=np.float32))

            # shot_quality feature_names strict match if provided by pack
            sq_names = ref_pack.get("shot_quality_feature_names") or []
            if sq_names and shot_npz is not None:
                cur_names = shot_npz.get("feature_names")
                if cur_names is not None:
                    cur_names = [str(x) for x in np.asarray(cur_names, dtype=object).tolist()]
                    if cur_names != list(sq_names):
                        raise RuntimeError(f"{MODULE_NAME} | shot_quality feature_names mismatch vs reference pack (strict)")

        # Similarities per modality
        sims_by_modality: Dict[str, Any] = {}
        topk_refs: List[Dict[str, Any]] = []
        present_by_modality: Dict[str, bool] = {
            "clip": True,
            "audio_clap": bool(clap_vec is not None),
            "text": bool(vec_text is not None),
            "pacing": bool(vec_pacing is not None),
            "quality": bool(vec_quality is not None),
            "emotion": bool(vec_emotion is not None),
            "overall": False,
        }
        if reference_present and ref_pack is not None:
            _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="reference_similarity")
            ref_ids = ref_pack["reference_video_ids"]
            sim_clip = (ref_pack["clip_video_embeddings"] @ vec_clip).astype(np.float32)

            # Audio similarity: only if we have CLAP embedding for this video; otherwise NaN.
            sim_clap = np.full_like(sim_clip, np.nan, dtype=np.float32)
            if clap_vec is not None:
                sim_clap = (ref_pack["clap_audio_embeddings"] @ clap_vec).astype(np.float32)

            # strict for reference modalities: if we don't have target vector, we still compute as NaN (target missing allowed for faces/text),
            # but reference pack must contain files (already enforced in loader).
            sim_text = np.full_like(sim_clip, np.nan, dtype=np.float32)
            if vec_text is not None:
                sim_text = (ref_pack["text_primary_embeddings"] @ vec_text).astype(np.float32)
            sim_pacing = np.full_like(sim_clip, np.nan, dtype=np.float32)
            if vec_pacing is not None:
                sim_pacing = (ref_pack["pacing_features"] @ vec_pacing).astype(np.float32)
            sim_quality = np.full_like(sim_clip, np.nan, dtype=np.float32)
            if vec_quality is not None:
                sim_quality = (ref_pack["shot_quality_features"] @ vec_quality).astype(np.float32)
            sim_emotion = np.full_like(sim_clip, np.nan, dtype=np.float32)
            if vec_emotion is not None:
                sim_emotion = (ref_pack["emotion_embeddings"] @ vec_emotion).astype(np.float32)

            sims_by_modality = {
                "clip": sim_clip,
                "audio_clap": sim_clap,
                "text": sim_text,
                "pacing": sim_pacing,
                "quality": sim_quality,
                "emotion": sim_emotion,
            }

            # Overall score (optional) - deterministic weighted nanmean
            overall = None
            if enable_overall_score:
                w = {
                    "clip": 1.0,
                    "audio_clap": 1.0,
                    "text": 1.0,
                    "pacing": 1.0,
                    "quality": 1.0,
                    "emotion": 1.0,
                }
                if weights:
                    for k, v in weights.items():
                        if k in w:
                            w[k] = float(v)
                num = np.zeros_like(sim_clip, dtype=np.float32)
                den = np.zeros_like(sim_clip, dtype=np.float32)
                for k, sarr in sims_by_modality.items():
                    ww = float(w.get(k, 0.0))
                    if ww <= 0:
                        continue
                    mask = ~np.isnan(sarr)
                    num[mask] += ww * sarr[mask]
                    den[mask] += ww
                overall = np.full_like(sim_clip, np.nan, dtype=np.float32)
                ok = den > 0
                overall[ok] = (num[ok] / den[ok]).astype(np.float32)
                sims_by_modality["overall"] = overall
                present_by_modality["overall"] = bool(np.any(np.isfinite(overall)))

            # Rank by overall if present else by clip
            rank_key = "overall" if ("overall" in sims_by_modality) else "clip"
            scores = sims_by_modality[rank_key]
            order = np.argsort(-scores.astype(np.float32))[: min(ui_topk, int(scores.size))]
            for idx in order.tolist():
                topk_refs.append(
                    {
                        "reference_video_id": str(ref_ids[int(idx)]),
                        "score": float(scores[int(idx)]) if not np.isnan(scores[int(idx)]) else None,
                        "scores_by_modality": {
                            k: (float(v[int(idx)]) if not np.isnan(v[int(idx)]) else None) for k, v in sims_by_modality.items()
                        },
                    }
                )

            # top-n aggregations for clip (legacy fields)
            sims_sorted = np.sort(sim_clip)[::-1]
            k2 = min(max(top_n, 1), sims_sorted.size)
            topk = sims_sorted[:k2]
            features.update(
                {
                    "reference_similarity_mean_topn": float(np.mean(topk)),
                    "reference_similarity_max": float(sims_sorted[0]) if sims_sorted.size else float("nan"),
                    "reference_similarity_p10": float(np.percentile(sim_clip, 10)),
                }
            )
        else:
            features.update({"reference_similarity_mean_topn": float("nan"), "reference_similarity_max": float("nan"), "reference_similarity_p10": float("nan")})

        t1 = time.perf_counter()
        _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="done")

        # Stable tabular features for models (fixed order, includes per-modality scores + present flags + uniqueness)
        feat_names, feat_vals = _build_fixed_feature_vector(
            base_feats=features,
            reference_present=bool(reference_present),
            sims_by_modality={k: np.asarray(v, dtype=np.float32) for k, v in sims_by_modality.items() if isinstance(v, np.ndarray)},
            present_by_modality=present_by_modality,
            top_n=int(top_n),
        )

        return {
            "frame_indices": fi,
            "times_s": times_s,
            "centroid_sims": centroid_sims,
            "temporal_sim_next": sim_next,
            "reference_present": np.asarray(bool(reference_present)),
            "feature_names": np.asarray(feat_names, dtype=object),
            "feature_values": feat_vals,
            "ui_payload": {
                "schema_version": "similarity_metrics_ui_v1",
                "reference_set_id": str(ref_set_id) if ref_set_id else None,
                "topk_refs": topk_refs,
                "text_present": bool(text_present),
                "audio_required_present": True,
                "present_by_modality": {k: bool(v) for k, v in present_by_modality.items()},
            },
        }

    def run(
                self,
                frames_dir: str,
                config: Dict[str, Any],
                metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Override BaseModule.run to:
        - attach ui_payload into NPZ meta (meta.ui_payload), not as a top-level NPZ key
        - add stage timings into meta.stage_timings_ms
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

        t0 = time.perf_counter()
        resource_profile_before = _resource_profile_snapshot()
        fm = None
        try:
            fm = self.create_frame_manager(frames_dir, metadata)
            t_fm = time.perf_counter()

            results = self.process(frame_manager=fm, frame_indices=frame_indices, config=config or {})
            t_proc = time.perf_counter()

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

            cfg_run = config or {}
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
            }
            if "top_n" in cfg_run:
                save_metadata["top_n"] = int(cfg_run.get("top_n", self._top_n))
            rid = cfg_run.get("reference_set_id")
            if rid is not None and str(rid).strip():
                save_metadata["reference_set_id"] = str(rid).strip()
            if "enable_overall_score" in cfg_run:
                save_metadata["enable_overall_score"] = bool(cfg_run.get("enable_overall_score"))
            if isinstance(resource_profile_before, dict) and resource_profile_before:
                save_metadata["resource_profile_before"] = dict(resource_profile_before)
            if isinstance(meta_override, dict):
                for k, v in meta_override.items():
                    if isinstance(k, str) and k and (isinstance(v, (str, int, float, bool)) or v is None):
                        save_metadata[k] = v

            # stage timings (meta.stage_timings_ms)
            save_metadata["stage_timings_ms"] = {
                "frame_manager_ms": float((t_fm - t0) * 1000.0),
                "process_ms": float((t_proc - t_fm) * 1000.0),
                "total_ms": float((t_proc - t0) * 1000.0),
            }

            # models_used (best-effort)
            try:
                # Prefer propagating from core_clip meta (consumer module).
                cc = self.load_core_provider("core_clip", file_name="embeddings.npz")
                save_metadata["models_used"] = _extract_meta_models_used(cc)
                if not save_metadata["models_used"]:
                    save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
            except Exception:
                save_metadata["models_used"] = []

            saved_path = self.save_results(results=results, metadata=save_metadata)
            return saved_path
        finally:
            if fm is not None:
                try:
                    fm.close()
                except Exception:
                    pass
    
    # ==================== C. Style & Composition Similarity ====================
    
    def compute_style_similarity(
        self,
        video_visual_features: Dict[str, Any],
        reference_visual_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть визуального стиля и композиции.
        
        Args:
            video_visual_features: Визуальные фичи текущего видео (цвет, свет, типы кадров, монтаж, движение)
            reference_visual_features_list: Список визуальных фичей референсных видео
            
        Returns:
            Словарь с метриками визуальной схожести
        """
        if len(reference_visual_features_list) == 0:
            return {
                'color_histogram_similarity': 0.0,
                'lighting_pattern_similarity': 0.0,
                'shot_type_distribution_similarity': 0.0,
                'cut_rate_similarity': 0.0,
                'motion_pattern_similarity': 0.0
            }
        
        color_sims = []
        lighting_sims = []
        shot_type_sims = []
        cut_rate_sims = []
        motion_sims = []
        
        for ref_features in reference_visual_features_list:
            # Color histogram similarity
            if 'color_histogram' in video_visual_features and 'color_histogram' in ref_features:
                hist1 = np.array(video_visual_features['color_histogram']).flatten()
                hist2 = np.array(ref_features['color_histogram']).flatten()
                if len(hist1) == len(hist2):
                    # Cosine similarity для гистограмм
                    hist1_norm = hist1 / (np.linalg.norm(hist1) + 1e-10)
                    hist2_norm = hist2 / (np.linalg.norm(hist2) + 1e-10)
                    color_sim = np.dot(hist1_norm, hist2_norm)
                    color_sims.append(color_sim)
            
            # Lighting pattern similarity
            if 'lighting_features' in video_visual_features and 'lighting_features' in ref_features:
                light1 = np.array(video_visual_features['lighting_features'])
                light2 = np.array(ref_features['lighting_features'])
                if light1.shape == light2.shape:
                    light_sim = 1.0 - cosine(light1.flatten(), light2.flatten())
                    lighting_sims.append(max(0.0, light_sim))
            
            # Shot type distribution similarity
            if 'shot_type_distribution' in video_visual_features and 'shot_type_distribution' in ref_features:
                dist1 = np.array(video_visual_features['shot_type_distribution'])
                dist2 = np.array(ref_features['shot_type_distribution'])
                if len(dist1) == len(dist2):
                    # Нормализуем распределения
                    dist1_norm = dist1 / (dist1.sum() + 1e-10)
                    dist2_norm = dist2 / (dist2.sum() + 1e-10)
                    # Earth Mover's Distance или cosine similarity
                    shot_sim = 1.0 - wasserstein_distance(dist1_norm, dist2_norm) / (np.max(dist1_norm) + np.max(dist2_norm) + 1e-10)
                    shot_type_sims.append(max(0.0, min(1.0, shot_sim)))
            
            # Cut rate similarity
            if 'cut_rate' in video_visual_features and 'cut_rate' in ref_features:
                cut1 = float(video_visual_features['cut_rate'])
                cut2 = float(ref_features['cut_rate'])
                # Нормализованная разница
                max_cut = max(abs(cut1), abs(cut2), 1.0)
                cut_sim = 1.0 - abs(cut1 - cut2) / max_cut
                cut_rate_sims.append(max(0.0, cut_sim))
            
            # Motion pattern similarity
            if 'motion_pattern' in video_visual_features and 'motion_pattern' in ref_features:
                motion1 = np.array(video_visual_features['motion_pattern'])
                motion2 = np.array(ref_features['motion_pattern'])
                if len(motion1) == len(motion2):
                    # Корреляция между паттернами движения
                    try:
                        corr, _ = pearsonr(motion1, motion2)
                        motion_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        motion_sims.append(0.0)
        
        return {
            'color_histogram_similarity': float(np.mean(color_sims)) if color_sims else 0.0,
            'lighting_pattern_similarity': float(np.mean(lighting_sims)) if lighting_sims else 0.0,
            'shot_type_distribution_similarity': float(np.mean(shot_type_sims)) if shot_type_sims else 0.0,
            'cut_rate_similarity': float(np.mean(cut_rate_sims)) if cut_rate_sims else 0.0,
            'motion_pattern_similarity': float(np.mean(motion_sims)) if motion_sims else 0.0
        }
    
    # ==================== D. Text & OCR Similarity ====================
    
    def compute_text_similarity(
        self,
        video_text_features: Dict[str, Any],
        reference_text_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть текста и OCR.
        
        Args:
            video_text_features: Текстовые фичи текущего видео (OCR embeddings, layout, timing)
            reference_text_features_list: Список текстовых фичей референсных видео
            
        Returns:
            Словарь с метриками текстовой схожести
        """
        if len(reference_text_features_list) == 0:
            return {
                'ocr_text_semantic_similarity': 0.0,
                'text_layout_similarity': 0.0,
                'text_timing_similarity': 0.0
            }
        
        ocr_sims = []
        layout_sims = []
        timing_sims = []
        
        for ref_features in reference_text_features_list:
            # OCR text semantic similarity
            if 'ocr_embedding' in video_text_features and 'ocr_embedding' in ref_features:
                emb1 = np.array(video_text_features['ocr_embedding'])
                emb2 = np.array(ref_features['ocr_embedding'])
                if emb1.shape == emb2.shape:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
                    ocr_sim = np.dot(emb1_norm, emb2_norm)
                    ocr_sims.append(ocr_sim)
            
            # Text layout similarity (позиции, длина, font size)
            if 'text_layout' in video_text_features and 'text_layout' in ref_features:
                layout1 = np.array(video_text_features['text_layout'])
                layout2 = np.array(ref_features['text_layout'])
                if len(layout1) == len(layout2):
                    layout_sim = 1.0 - cosine(layout1.flatten(), layout2.flatten())
                    layout_sims.append(max(0.0, layout_sim))
            
            # Text timing similarity
            if 'text_timing' in video_text_features and 'text_timing' in ref_features:
                timing1 = np.array(video_text_features['text_timing'])
                timing2 = np.array(ref_features['text_timing'])
                if len(timing1) == len(timing2):
                    try:
                        corr, _ = pearsonr(timing1, timing2)
                        timing_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        timing_sims.append(0.0)
        
        return {
            'ocr_text_semantic_similarity': float(np.mean(ocr_sims)) if ocr_sims else 0.0,
            'text_layout_similarity': float(np.mean(layout_sims)) if layout_sims else 0.0,
            'text_timing_similarity': float(np.mean(timing_sims)) if timing_sims else 0.0
        }
    
    # ==================== E. Audio / Speech Similarity ====================
    
    def compute_audio_similarity(
        self,
        video_audio_features: Dict[str, Any],
        reference_audio_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть аудио характеристик.
        
        Args:
            video_audio_features: Аудио фичи текущего видео (embeddings, tempo, energy)
            reference_audio_features_list: Список аудио фичей референсных видео
            
        Returns:
            Словарь с метриками аудио схожести
        """
        if len(reference_audio_features_list) == 0:
            return {
                'audio_embedding_similarity': 0.0,
                'speech_content_similarity': 0.0,
                'music_tempo_similarity': 0.0,
                'audio_energy_pattern_similarity': 0.0
            }
        
        audio_emb_sims = []
        speech_sims = []
        tempo_sims = []
        energy_sims = []
        
        for ref_features in reference_audio_features_list:
            # Audio embedding similarity
            if 'audio_embedding' in video_audio_features and 'audio_embedding' in ref_features:
                emb1 = np.array(video_audio_features['audio_embedding'])
                emb2 = np.array(ref_features['audio_embedding'])
                if emb1.shape == emb2.shape:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
                    audio_sim = np.dot(emb1_norm, emb2_norm)
                    audio_emb_sims.append(audio_sim)
            
            # Speech content similarity (по ASR)
            if 'speech_embedding' in video_audio_features and 'speech_embedding' in ref_features:
                speech1 = np.array(video_audio_features['speech_embedding'])
                speech2 = np.array(ref_features['speech_embedding'])
                if speech1.shape == speech2.shape:
                    speech1_norm = speech1 / (np.linalg.norm(speech1) + 1e-10)
                    speech2_norm = speech2 / (np.linalg.norm(speech2) + 1e-10)
                    speech_sim = np.dot(speech1_norm, speech2_norm)
                    speech_sims.append(speech_sim)
            
            # Music tempo similarity
            if 'tempo' in video_audio_features and 'tempo' in ref_features:
                tempo1 = video_audio_features['tempo']
                tempo2 = ref_features['tempo']
                max_tempo = max(abs(tempo1), abs(tempo2), 1.0)
                tempo_sim = 1.0 - abs(tempo1 - tempo2) / max_tempo
                tempo_sims.append(max(0.0, tempo_sim))
            
            # Audio energy pattern similarity
            if 'energy_pattern' in video_audio_features and 'energy_pattern' in ref_features:
                energy1 = np.array(video_audio_features['energy_pattern'])
                energy2 = np.array(ref_features['energy_pattern'])
                if len(energy1) == len(energy2):
                    try:
                        corr, _ = pearsonr(energy1, energy2)
                        energy_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        energy_sims.append(0.0)
        
        return {
            'audio_embedding_similarity': float(np.mean(audio_emb_sims)) if audio_emb_sims else 0.0,
            'speech_content_similarity': float(np.mean(speech_sims)) if speech_sims else 0.0,
            'music_tempo_similarity': float(np.mean(tempo_sims)) if tempo_sims else 0.0,
            'audio_energy_pattern_similarity': float(np.mean(energy_sims)) if energy_sims else 0.0
        }
    
    # ==================== F. Emotion & Behavior Similarity ====================
    
    def compute_emotion_behavior_similarity(
        self,
        video_emotion_features: Dict[str, Any],
        reference_emotion_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть эмоций и поведения.
        
        Args:
            video_emotion_features: Фичи эмоций/поведения текущего видео (emotion curve, pose, behavior)
            reference_emotion_features_list: Список фичей эмоций/поведения референсных видео
            
        Returns:
            Словарь с метриками схожести эмоций и поведения
        """
        if len(reference_emotion_features_list) == 0:
            return {
                'emotion_curve_similarity': 0.0,
                'pose_motion_similarity': 0.0,
                'behavior_pattern_similarity': 0.0
            }
        
        emotion_sims = []
        pose_sims = []
        behavior_sims = []
        
        for ref_features in reference_emotion_features_list:
            # Emotion curve similarity
            if 'emotion_curve' in video_emotion_features and 'emotion_curve' in ref_features:
                curve1 = np.array(video_emotion_features['emotion_curve'])
                curve2 = np.array(ref_features['emotion_curve'])
                if len(curve1) == len(curve2):
                    try:
                        corr, _ = pearsonr(curve1, curve2)
                        emotion_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        emotion_sims.append(0.0)
            
            # Pose motion similarity
            if 'pose_motion' in video_emotion_features and 'pose_motion' in ref_features:
                pose1 = np.array(video_emotion_features['pose_motion'])
                pose2 = np.array(ref_features['pose_motion'])
                if pose1.shape == pose2.shape:
                    pose1_norm = pose1 / (np.linalg.norm(pose1) + 1e-10)
                    pose2_norm = pose2 / (np.linalg.norm(pose2) + 1e-10)
                    pose_sim = np.dot(pose1_norm.flatten(), pose2_norm.flatten())
                    pose_sims.append(pose_sim)
            
            # Behavior pattern similarity
            if 'behavior_pattern' in video_emotion_features and 'behavior_pattern' in ref_features:
                behavior1 = np.array(video_emotion_features['behavior_pattern'])
                behavior2 = np.array(ref_features['behavior_pattern'])
                if len(behavior1) == len(behavior2):
                    try:
                        corr, _ = pearsonr(behavior1, behavior2)
                        behavior_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        behavior_sims.append(0.0)
        
        return {
            'emotion_curve_similarity': float(np.mean(emotion_sims)) if emotion_sims else 0.0,
            'pose_motion_similarity': float(np.mean(pose_sims)) if pose_sims else 0.0,
            'behavior_pattern_similarity': float(np.mean(behavior_sims)) if behavior_sims else 0.0
        }
    
    # ==================== G. Temporal / Pacing Similarity ====================
    
    def compute_temporal_similarity(
        self,
        video_pacing_features: Dict[str, Any],
        reference_pacing_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть временного ритма и pacing.
        
        Args:
            video_pacing_features: Фичи pacing текущего видео (pacing curve, shot duration, scene length)
            reference_pacing_features_list: Список фичей pacing референсных видео
            
        Returns:
            Словарь с метриками схожести временного ритма
        """
        if len(reference_pacing_features_list) == 0:
            return {
                'pacing_curve_similarity': 0.0,
                'shot_duration_distribution_similarity': 0.0,
                'scene_length_similarity': 0.0,
                'temporal_pattern_novelty': 1.0
            }
        
        pacing_sims = []
        shot_duration_sims = []
        scene_length_sims = []
        
        for ref_features in reference_pacing_features_list:
            # Pacing curve similarity
            if 'pacing_curve' in video_pacing_features and 'pacing_curve' in ref_features:
                curve1 = np.array(video_pacing_features['pacing_curve'])
                curve2 = np.array(ref_features['pacing_curve'])
                if len(curve1) == len(curve2):
                    try:
                        corr, _ = pearsonr(curve1, curve2)
                        pacing_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        pacing_sims.append(0.0)
            
            # Shot duration distribution similarity
            if 'shot_duration_distribution' in video_pacing_features and 'shot_duration_distribution' in ref_features:
                dist1 = np.array(video_pacing_features['shot_duration_distribution'])
                dist2 = np.array(ref_features['shot_duration_distribution'])
                if len(dist1) == len(dist2):
                    dist1_norm = dist1 / (dist1.sum() + 1e-10)
                    dist2_norm = dist2 / (dist2.sum() + 1e-10)
                    # Wasserstein distance
                    wd = wasserstein_distance(dist1_norm, dist2_norm)
                    max_wd = np.max(dist1_norm) + np.max(dist2_norm)
                    shot_sim = 1.0 - wd / (max_wd + 1e-10)
                    shot_duration_sims.append(max(0.0, min(1.0, shot_sim)))
            
            # Scene length similarity
            if 'scene_lengths' in video_pacing_features and 'scene_lengths' in ref_features:
                len1 = np.array(video_pacing_features['scene_lengths'])
                len2 = np.array(ref_features['scene_lengths'])
                if len(len1) > 0 and len(len2) > 0:
                    # Сравниваем средние и std
                    mean1, std1 = np.mean(len1), np.std(len1)
                    mean2, std2 = np.mean(len2), np.std(len2)
                    max_mean = max(abs(mean1), abs(mean2), 1.0)
                    max_std = max(abs(std1), abs(std2), 1.0)
                    mean_sim = 1.0 - abs(mean1 - mean2) / max_mean
                    std_sim = 1.0 - abs(std1 - std2) / max_std
                    scene_sim = (mean_sim + std_sim) / 2.0
                    scene_length_sims.append(max(0.0, scene_sim))
        
        # Temporal pattern novelty = 1 - mean similarity
        mean_pacing_sim = np.mean(pacing_sims) if pacing_sims else 0.0
        
        return {
            'pacing_curve_similarity': float(np.mean(pacing_sims)) if pacing_sims else 0.0,
            'shot_duration_distribution_similarity': float(np.mean(shot_duration_sims)) if shot_duration_sims else 0.0,
            'scene_length_similarity': float(np.mean(scene_length_sims)) if scene_length_sims else 0.0,
            'temporal_pattern_novelty': float(1.0 - mean_pacing_sim)
        }
    
    # ==================== H. High-level Comparative Scores ====================
    
    def compute_high_level_scores(
        self,
        all_similarity_metrics: Dict[str, float],
        reference_videos_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет высокоуровневые сравнительные оценки.
        
        Args:
            all_similarity_metrics: Словарь со всеми метриками схожести из категорий A-G
            reference_videos_metadata: Метаданные референсных видео (для trend_alignment и viral_pattern)
            
        Returns:
            Словарь с высокоуровневыми оценками
        """
        # Overall similarity score = взвешенная сумма базовых аспектов.
        # Это эвристический скор, рекомендовано переобучать веса отдельно.
        weights = self.similarity_weights
        
        semantic_score = all_similarity_metrics.get('semantic_similarity_mean', 0.0)
        topics_score = all_similarity_metrics.get('topic_overlap_score', 0.0)
        visual_score = float(np.mean([
            all_similarity_metrics.get('color_histogram_similarity', 0.0),
            all_similarity_metrics.get('lighting_pattern_similarity', 0.0),
            all_similarity_metrics.get('shot_type_distribution_similarity', 0.0),
        ]))
        text_score = all_similarity_metrics.get('ocr_text_semantic_similarity', 0.0)
        audio_score = all_similarity_metrics.get('audio_embedding_similarity', 0.0)
        emotion_score = all_similarity_metrics.get('emotion_curve_similarity', 0.0)
        temporal_score = all_similarity_metrics.get('pacing_curve_similarity', 0.0)
        
        overall_similarity = (
            weights['semantic'] * semantic_score +
            weights['topics'] * topics_score +
            weights['visual'] * visual_score +
            weights['text'] * text_score +
            weights['audio'] * audio_score +
            weights['emotion'] * emotion_score +
            weights['temporal'] * temporal_score
        )
        overall_similarity = float(np.clip(overall_similarity, 0.0, 1.0))
        
        uniqueness_score = float(1.0 - overall_similarity)
        
        # Trend alignment / viral pattern по умолчанию = overall_similarity;
        # предполагается, что в продакшене будут обучены отдельные агрегаторы.
        trend_alignment = overall_similarity
        viral_pattern = overall_similarity
        
        return {
            'overall_similarity_score': overall_similarity,
            'uniqueness_score': uniqueness_score,
            'trend_alignment_score': trend_alignment,
            'viral_pattern_score': viral_pattern,
        }
    
    # ==================== I. Group / Batch Metrics ====================
    
    def compute_batch_metrics(
        self,
        video_embeddings: List[np.ndarray],
        video_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет групповые метрики для батча видео.
        
        Args:
            video_embeddings: Список embeddings всех видео в батче
            video_features_list: Список фичей всех видео в батче
            
        Returns:
            Словарь с групповыми метриками
        """
        if len(video_embeddings) < 2:
            return {
                'cluster_similarity_mean': 0.0,
                'inter_video_variance_topics': 0.0,
                'inter_video_variance_emotions': 0.0,
                'inter_video_variance_editing': 0.0,
                'inter_video_variance_audio': 0.0
            }
        
        # Cluster similarity metrics (средняя схожесть между всеми парами)
        pairwise_similarities = []
        for i in range(len(video_embeddings)):
            for j in range(i + 1, len(video_embeddings)):
                emb1 = video_embeddings[i] / (np.linalg.norm(video_embeddings[i]) + 1e-10)
                emb2 = video_embeddings[j] / (np.linalg.norm(video_embeddings[j]) + 1e-10)
                sim = np.dot(emb1, emb2)
                pairwise_similarities.append(sim)
        
        cluster_similarity = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
        
        # Inter-video variance по различным аспектам
        topics_variance = 0.0
        emotions_variance = 0.0
        editing_variance = 0.0
        audio_variance = 0.0
        
        if video_features_list:
            # Topics variance
            topic_vectors = []
            for features in video_features_list:
                if 'topic_embedding' in features:
                    topic_vectors.append(features['topic_embedding'])
            if topic_vectors:
                topics_variance = float(np.var([np.linalg.norm(t) for t in topic_vectors]))
            
            # Emotions variance
            emotion_means = []
            for features in video_features_list:
                if 'emotion_mean' in features:
                    emotion_means.append(features['emotion_mean'])
            if emotion_means:
                emotions_variance = float(np.var(emotion_means))
            
            # Editing variance (cut rate)
            cut_rates = []
            for features in video_features_list:
                if 'cut_rate' in features:
                    cut_rates.append(features['cut_rate'])
            if cut_rates:
                editing_variance = float(np.var(cut_rates))
            
            # Audio variance (tempo)
            tempos = []
            for features in video_features_list:
                if 'tempo' in features:
                    tempos.append(features['tempo'])
            if tempos:
                audio_variance = float(np.var(tempos))
        
        return {
            'cluster_similarity_mean': float(cluster_similarity),
            'inter_video_variance_topics': float(topics_variance),
            'inter_video_variance_emotions': float(emotions_variance),
            'inter_video_variance_editing': float(editing_variance),
            'inter_video_variance_audio': float(audio_variance)
        }
    
    # ==================== Main Method ====================
    
    def extract_all(
        self,
        video_embedding: np.ndarray,
        reference_embeddings: List[np.ndarray],
        video_topics: Optional[Union[List[str], np.ndarray, Dict[str, float]]] = None,
        reference_topics_list: Optional[List[Union[List[str], np.ndarray, Dict[str, float]]]] = None,
        video_visual_features: Optional[Dict[str, Any]] = None,
        reference_visual_features_list: Optional[List[Dict[str, Any]]] = None,
        video_text_features: Optional[Dict[str, Any]] = None,
        reference_text_features_list: Optional[List[Dict[str, Any]]] = None,
        video_audio_features: Optional[Dict[str, Any]] = None,
        reference_audio_features_list: Optional[List[Dict[str, Any]]] = None,
        video_emotion_features: Optional[Dict[str, Any]] = None,
        reference_emotion_features_list: Optional[List[Dict[str, Any]]] = None,
        video_pacing_features: Optional[Dict[str, Any]] = None,
        reference_pacing_features_list: Optional[List[Dict[str, Any]]] = None,
        reference_videos_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Главный метод для вычисления всех метрик схожести.
        
        Args:
            video_embedding: Embedding текущего видео
            reference_embeddings: Список embeddings референсных видео
            video_topics: Темы текущего видео
            reference_topics_list: Список тем референсных видео
            video_visual_features: Визуальные фичи текущего видео
            reference_visual_features_list: Список визуальных фичей референсных видео
            video_text_features: Текстовые фичи текущего видео
            reference_text_features_list: Список текстовых фичей референсных видео
            video_audio_features: Аудио фичи текущего видео
            reference_audio_features_list: Список аудио фичей референсных видео
            video_emotion_features: Фичи эмоций/поведения текущего видео
            reference_emotion_features_list: Список фичей эмоций/поведения референсных видео
            video_pacing_features: Фичи pacing текущего видео
            reference_pacing_features_list: Список фичей pacing референсных видео
            reference_videos_metadata: Метаданные референсных видео (опционально)
            
        Returns:
            Словарь со всеми метриками схожести
        """
        features = {}
        
        # A. Semantic similarity
        semantic_metrics = self.compute_semantic_similarity(video_embedding, reference_embeddings)
        features.update(semantic_metrics)
        
        # B. Topic overlap
        if video_topics is not None and reference_topics_list is not None:
            topic_metrics = self.compute_topic_overlap(video_topics, reference_topics_list)
            features.update(topic_metrics)
        else:
            features.update({
                'topic_overlap_score': 0.0,
                'topic_diversity_comparison': 0.0,
                'key_concept_match_ratio': 0.0
            })
        
        # C. Style & Composition
        if video_visual_features is not None and reference_visual_features_list is not None:
            style_metrics = self.compute_style_similarity(video_visual_features, reference_visual_features_list)
            features.update(style_metrics)
        else:
            features.update({
                'color_histogram_similarity': 0.0,
                'lighting_pattern_similarity': 0.0,
                'shot_type_distribution_similarity': 0.0,
                'cut_rate_similarity': 0.0,
                'motion_pattern_similarity': 0.0
            })
        
        # D. Text & OCR
        if video_text_features is not None and reference_text_features_list is not None:
            text_metrics = self.compute_text_similarity(video_text_features, reference_text_features_list)
            features.update(text_metrics)
        else:
            features.update({
                'ocr_text_semantic_similarity': 0.0,
                'text_layout_similarity': 0.0,
                'text_timing_similarity': 0.0
            })
        
        # E. Audio / Speech
        if video_audio_features is not None and reference_audio_features_list is not None:
            audio_metrics = self.compute_audio_similarity(video_audio_features, reference_audio_features_list)
            features.update(audio_metrics)
        else:
            features.update({
                'audio_embedding_similarity': 0.0,
                'speech_content_similarity': 0.0,
                'music_tempo_similarity': 0.0,
                'audio_energy_pattern_similarity': 0.0
            })
        
        # F. Emotion & Behavior
        if video_emotion_features is not None and reference_emotion_features_list is not None:
            emotion_metrics = self.compute_emotion_behavior_similarity(
                video_emotion_features, reference_emotion_features_list
            )
            features.update(emotion_metrics)
        else:
            features.update({
                'emotion_curve_similarity': 0.0,
                'pose_motion_similarity': 0.0,
                'behavior_pattern_similarity': 0.0
            })
        
        # G. Temporal / Pacing
        if video_pacing_features is not None and reference_pacing_features_list is not None:
            temporal_metrics = self.compute_temporal_similarity(video_pacing_features, reference_pacing_features_list)
            features.update(temporal_metrics)
        else:
            features.update({
                'pacing_curve_similarity': 0.0,
                'shot_duration_distribution_similarity': 0.0,
                'scene_length_similarity': 0.0,
                'temporal_pattern_novelty': 1.0
            })
        
        # H. High-level scores
        high_level = self.compute_high_level_scores(features, reference_videos_metadata)
        features.update(high_level)
        
        return {
            'features': features,
            'all_metrics': features  # Для обратной совместимости
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Пример использования
    import numpy as np
    
    # Создаем экземпляр
    similarity = SimilarityMetrics(top_n=10)
    
    # Примерные данные текущего видео
    video_embedding = np.random.randn(512)
    video_topics = ["cooking", "tutorial", "food"]
    
    # Референсные видео
    reference_embeddings = [np.random.randn(512) for _ in range(5)]
    reference_topics_list = [
        ["cooking", "recipe"],
        ["gaming", "tutorial"],
        ["cooking", "food", "tutorial"],
        ["travel", "vlog"],
        ["cooking", "diy"]
    ]
    
    # Вычисляем метрики
    result = similarity.extract_all(
        video_embedding=video_embedding,
        reference_embeddings=reference_embeddings,
        video_topics=video_topics,
        reference_topics_list=reference_topics_list
    )
    
    print("Similarity metrics:")
    for key, value in result['features'].items():
        print(f"  {key}: {value:.4f}")
