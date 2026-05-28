#!/usr/bin/env python3
"""
car_semantics (semantic head, v1)

Car recognition using Embedding Service:
- Detects car objects from core_object_detections
- Extracts crops with padding
- Searches for similar cars via Embedding Service
- Returns per-track top-K car identifications (make, model, segment)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import hashlib

_vp_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _vp_root not in sys.path:
    sys.path.insert(0, _vp_root)
elif sys.path[0] != _vp_root:
    try:
        sys.path.remove(_vp_root)
    except ValueError:
        pass
    sys.path.insert(0, _vp_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta  # type: ignore  # noqa: E402
from utils.embedding_service_errors import EmbeddingServiceUnavailableError  # type: ignore  # noqa: E402


def _load_crop_utils():
    import importlib.util

    p = Path(__file__).resolve().parent / "utils" / "crop_utils.py"
    spec = importlib.util.spec_from_file_location("car_semantics.crop_utils", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"car_semantics | crop_utils not found: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.crop_with_padding, mod.select_best_crop_for_track


crop_with_padding, select_best_crop_for_track = _load_crop_utils()

try:
    from utils.embedding_service_client import EmbeddingServiceClient
except ImportError:
    try:
        from embedding_service_client import EmbeddingServiceClient
    except ImportError:
        import importlib.util
        _current_dir = Path(__file__).parent
        _utils_path = _current_dir / "utils" / "embedding_service_client.py"
        if _utils_path.exists():
            spec = importlib.util.spec_from_file_location("embedding_service_client", str(_utils_path))
            if spec and spec.loader:
                _mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_mod)
                EmbeddingServiceClient = _mod.EmbeddingServiceClient
            else:
                raise RuntimeError("car_semantics | Failed to load EmbeddingServiceClient from utils")
        else:
            raise ImportError("car_semantics | embedding_service_client not found. Expected at: %s" % _utils_path)

NAME = "car_semantics"
VERSION = "0.2"
SCHEMA_VERSION = "car_semantics_npz_v2"
ARTIFACT_FILENAME = "car_semantics.npz"
LOGGER = get_logger(NAME)

# Car category for Embedding Service
CAR_CATEGORY = "car"
TOP_K = 5  # Contract: fixed K for semantic heads v1+


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _atomic_save_npz(out_path: str, **kwargs: Any) -> None:
    """
    Atomic NPZ save (tmp -> replace).
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **kwargs)
    os.replace(tmp_path, out_path)


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (backend tails this file).
    """
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
        event["platform_id"] = platform_id
        event["video_id"] = video_id
        event["run_id"] = run_id
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
    """Emit stage event to state_events.jsonl."""
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
            "status": "running",
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


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
    """Emit progress event to state_events.jsonl."""
    if total <= 0:
        return
    progress = float(done) / float(total)
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
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


def _require_frame_indices(meta: dict) -> List[int]:
    block = meta.get("core_object_detections")
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | frames metadata missing core_object_detections.frame_indices (no-fallback)"
        )
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(
            f"{NAME} | core_object_detections.frame_indices empty/invalid (no-fallback)"
        )
    return [int(x) for x in frame_indices]


def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME} | required artifact not found: {path}")
    z = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in z.files:
        v = z[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _get_class_id_to_name(class_names: np.ndarray) -> Dict[int, str]:
    """Parse class_names array into id->name mapping"""
    result: Dict[int, str] = {}
    for item in class_names:
        item_str = str(item)
        if ":" in item_str:
            try:
                class_id_str, class_name = item_str.split(":", 1)
                result[int(class_id_str)] = class_name
            except Exception:
                continue
    return result


def _extract_car_metadata(metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract make, model, segment from metadata dictionary.

    Args:
        metadata: Metadata dictionary from Embedding Service result

    Returns:
        Tuple of (make, model, segment) as strings.
        Returns empty strings if not found in metadata.
    """
    make = str(metadata.get("make", "")).strip()
    model = str(metadata.get("model", "")).strip()
    segment = str(metadata.get("segment", "")).strip()
    return make, model, segment


def main() -> int:
    ap = argparse.ArgumentParser("car_semantics")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rs-path", required=True)
    ap.add_argument(
        "--embedding-service-url",
        default=None,
        help="Embedding Service URL (default: from EMBEDDING_SERVICE_URL env or http://localhost:8001)",
    )
    ap.add_argument(
        "--topk", type=int, default=TOP_K, help=f"Top-K results (default: {TOP_K})"
    )
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help=(
            "DEPRECATED name (kept for backward compatibility). "
            "Used ONLY as confidence threshold for *_is_confident_top1 flags. "
            "Contract: MUST NOT gate top-K results. Range: 0.0-1.0."
        ),
    )
    ap.add_argument(
        "--confidence-threshold-top1",
        type=float,
        default=None,
        help=(
            "Confidence threshold for *_is_confident_top1 flags (does NOT gate top-K). "
            "If not set, falls back to --similarity-threshold for backward compatibility."
        ),
    )
    ap.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Maximum number of tracks to process (cost control)",
    )
    ap.add_argument(
        "--max-dets-per-frame",
        type=int,
        default=None,
        help="Maximum detections per frame (cost control)",
    )
    ap.add_argument(
        "--proposal-classes",
        type=str,
        default="car",
        help=(
            "Comma-separated list of proposal class names from core_object_detections taxonomy "
            '(default: "car").'
        ),
    )
    ap.add_argument(
        "--pad-ratio",
        type=float,
        default=0.15,
        help="Padding ratio for crops (default: 0.15 = 15%%)",
    )
    ap.add_argument(
        "--use-sharpness",
        action="store_true",
        help="Use sharpness metric for selecting best crop per track",
    )
    args = ap.parse_args()

    # Contract: fixed K (downstream expects consistent shape)
    if int(args.topk) != int(TOP_K):
        raise RuntimeError(
            f"{NAME} | topk must be fixed to {TOP_K} by contract; got {args.topk}"
        )

    confidence_threshold_top1 = (
        float(args.confidence_threshold_top1)
        if args.confidence_threshold_top1 is not None
        else float(args.similarity_threshold)
    )
    if not (0.0 <= confidence_threshold_top1 <= 1.0):
        raise RuntimeError(
            f"{NAME} | confidence_threshold_top1 out of range [0,1]: {confidence_threshold_top1}"
        )

    proposal_classes = [
        s.strip()
        for s in str(args.proposal_classes or "").split(",")
        if str(s).strip()
    ]
    if not proposal_classes:
        raise RuntimeError(f"{NAME} | proposal_classes is empty (contract)")

    # Initialize timing dictionary
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Load metadata
    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    
    # Extract run identity for state_events
    platform_id = str(meta.get("platform_id") or "")
    video_id = str(meta.get("video_id") or "")
    run_id = str(meta.get("run_id") or "")
    
    # Baseline contract: emit start stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="start",
    )
    
    t_load_deps = time.perf_counter()
    timings["initialization"] = t_load_deps - t0
    
    frame_indices = _require_frame_indices(meta)

    # Timestamps
    uts = (
        meta.get("union_timestamps_sec")
        or meta.get("union_timestamps_s")
        or meta.get("times_s")
    )
    if uts is None:
        raise RuntimeError(
            f"{NAME} | metadata.json missing union_timestamps_sec (contract)"
        )
    uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
    fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
        raise RuntimeError(
            f"{NAME} | frame_indices out of range for union_timestamps_sec"
        )
    times_s = uts_arr[fi_np].astype(np.float32)

    # Baseline contract: emit load_deps stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )
    
    # Load detections
    detections_path = os.path.join(
        str(args.rs_path), "core_object_detections", "detections.npz"
    )
    detections = _load_npz(detections_path)
    
    t_load_deps_end = time.perf_counter()
    timings["load_deps"] = t_load_deps_end - t_load_deps

    # Parse detections
    boxes = np.asarray(detections.get("boxes"), dtype=np.float32)  # (N, MAX, 4)
    scores = np.asarray(detections.get("scores"), dtype=np.float32)  # (N, MAX)
    class_ids = np.asarray(detections.get("class_ids"), dtype=np.int32)  # (N, MAX)
    valid_mask = np.asarray(
        detections.get("valid_mask"), dtype=bool
    )  # (N, MAX)
    class_names = np.asarray(
        detections.get("class_names"), dtype="U"
    )  # (M,)
    class_id_to_name = _get_class_id_to_name(class_names)

    # Validate array shapes
    if boxes.shape[:2] != scores.shape or boxes.shape[:2] != class_ids.shape or boxes.shape[:2] != valid_mask.shape:
        raise RuntimeError(
            f"{NAME} | Mismatched detection array shapes: "
            f"boxes={boxes.shape}, scores={scores.shape}, "
            f"class_ids={class_ids.shape}, valid_mask={valid_mask.shape}"
        )
    if len(frame_indices) != boxes.shape[0]:
        raise RuntimeError(
            f"{NAME} | Mismatched frame count: "
            f"frame_indices={len(frame_indices)}, boxes.shape[0]={boxes.shape[0]}"
        )

    # Proposal classes (contract): explicit mapping by class name
    name_to_id = {str(v): int(k) for k, v in class_id_to_name.items()}
    missing = [c for c in proposal_classes if c not in name_to_id]
    if missing:
        raise RuntimeError(
            f"{NAME} | proposal_classes not found in core_object_detections taxonomy: {missing} "
            f"(available example: {sorted(list(name_to_id.keys()))[:15]}...)"
        )
    allowed_class_ids = sorted({int(name_to_id[c]) for c in proposal_classes})

    proposal_mask = (valid_mask.astype(bool)) & np.isin(class_ids, np.asarray(allowed_class_ids, dtype=np.int32))

    # Initialize Embedding Service client (fail-fast)
    embedding_client = EmbeddingServiceClient(base_url=args.embedding_service_url)
    embedding_client._ensure_url()

    # Label-space (fail-fast if empty)
    labels = embedding_client.get_labels(category=CAR_CATEGORY)
    if not labels:
        raise RuntimeError(
            f"{NAME} | Embedding Service category '{CAR_CATEGORY}' has 0 labels (fail-fast)"
        )

    # Deterministic label-space (UUID -> stable int)
    labels_canon = []
    for r in labels:
        if not isinstance(r, dict):
            continue
        if not r.get("id"):
            continue
        labels_canon.append(
            {
                "id": str(r.get("id")),
                "name": str(r.get("name") or ""),
                "embedding_model": str(r.get("embedding_model") or ""),
                "embedding_dim": int(r.get("embedding_dim") or 0),
            }
        )
    labels_canon.sort(key=lambda x: x["id"])
    if not labels_canon:
        raise RuntimeError(
            f"{NAME} | Embedding Service returned invalid labels for category='{CAR_CATEGORY}' (fail-fast)"
        )

    db_digest = _sha256_hex(
        json.dumps(labels_canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    semantic_object_ids = np.asarray([r["id"] for r in labels_canon], dtype="U")
    semantic_label_names = np.asarray(
        [f"{i}:{r['name']}" for i, r in enumerate(labels_canon)],
        dtype="U",
    )
    uuid_to_int = {str(r["id"]): int(i) for i, r in enumerate(labels_canon)}
    A = int(semantic_label_names.shape[0])
    threshold_per_label_arr = np.full((A,), np.nan, dtype=np.float32)

    # Best-effort parsed taxonomy from label name (analytics/debug only)
    semantic_label_make = np.empty((A,), dtype="U")
    semantic_label_model = np.empty((A,), dtype="U")
    for i in range(A):
        name = str(labels_canon[i].get("name") or "")
        if "_" in name:
            parts = [p for p in name.split("_") if p]
            make = parts[0] if parts else ""
            model = "_".join(parts[1:]) if len(parts) > 1 else ""
        else:
            make, model = "", name
        semantic_label_make[i] = make
        semantic_label_model[i] = model

    # Create FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=int(meta.get("chunk_size", 32)),
        cache_size=int(meta.get("cache_size", 2)),
    )

    # Build candidate detections (per-detection surrogate tracks; tracking removed upstream)
    candidates: List[Tuple[float, int, int, float, np.ndarray, int]] = []
    for frame_pos in range(int(boxes.shape[0])):
        dets = np.where(proposal_mask[frame_pos])[0].astype(int)
        if dets.size == 0:
            continue
        if args.max_dets_per_frame is not None:
            k = int(args.max_dets_per_frame)
            if k > 0 and dets.size > k:
                ss = scores[frame_pos, dets].astype(np.float32)
                order = np.argsort(-ss)[:k]
                dets = dets[order]
        for det_idx in dets.tolist():
            bbox = boxes[frame_pos, det_idx].copy()
            sc = float(scores[frame_pos, det_idx])
            area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            metric = sc * area
            cls = int(class_ids[frame_pos, det_idx])
            candidates.append((metric, int(frame_pos), int(det_idx), sc, bbox, cls))

    candidates.sort(key=lambda x: float(x[0]), reverse=True)
    if args.max_tracks is not None:
        mt = int(args.max_tracks)
        if mt > 0 and len(candidates) > mt:
            candidates = candidates[:mt]

    # Provider status/empty semantics:
    # - empty is valid only when there are no valid proposals (with correct deps/db)
    status = "ok" if candidates else "empty"
    empty_reason = None if candidates else "no_car_proposals"

    # Baseline contract: emit process_frames stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="process_frames",
    )
    
    t_process_start = time.perf_counter()

    # Output arrays (NaN/-1 policy)
    n_frames = int(len(frame_indices))
    max_dets = int(boxes.shape[1])
    n_tracks = int(len(candidates))

    track_ids_arr = np.arange(n_tracks, dtype=np.int32)
    track_present_mask = np.ones((n_tracks,), dtype=bool)
    track_topk_ids = np.full((n_tracks, TOP_K), -1, dtype=np.int32)
    track_topk_scores = np.full((n_tracks, TOP_K), np.nan, dtype=np.float32)
    track_is_confident_top1 = np.zeros((n_tracks,), dtype=bool)

    frame_topk_ids = np.full((n_frames, TOP_K), -1, dtype=np.int32)
    frame_topk_scores = np.full((n_frames, TOP_K), np.nan, dtype=np.float32)
    frame_is_confident_top1 = np.zeros((n_frames,), dtype=bool)

    det_present_mask = np.zeros((n_frames, max_dets), dtype=bool)
    det_topk_ids = np.full((n_frames, max_dets, TOP_K), -1, dtype=np.int32)
    det_topk_scores = np.full((n_frames, max_dets, TOP_K), np.nan, dtype=np.float32)
    det_is_confident_top1 = np.zeros((n_frames, max_dets), dtype=bool)

    # QA helpers for offline render assets
    track_best_frame_pos = np.full((n_tracks,), -1, dtype=np.int32)
    track_best_det_idx = np.full((n_tracks,), -1, dtype=np.int32)
    track_best_bbox_xyxy = np.full((n_tracks, 4), np.nan, dtype=np.float32)
    track_best_det_score = np.full((n_tracks,), np.nan, dtype=np.float32)
    track_best_class_id = np.full((n_tracks,), -1, dtype=np.int32)

    # Process each candidate detection (surrogate track)
    for track_pos, (_metric, frame_pos, det_idx, det_score, bbox, cls) in enumerate(candidates):
        track_best_frame_pos[track_pos] = int(frame_pos)
        track_best_det_idx[track_pos] = int(det_idx)
        track_best_bbox_xyxy[track_pos, :] = np.asarray(bbox, dtype=np.float32).reshape(4)
        track_best_det_score[track_pos] = float(det_score)
        track_best_class_id[track_pos] = int(cls)

        frame_idx_global = int(frame_indices[int(frame_pos)])
        try:
            frame = frame_manager.get(frame_idx_global)
        except Exception as e:
            raise RuntimeError(f"{NAME} | failed to load frame {frame_idx_global}: {e}") from e

        crop = crop_with_padding(frame, bbox, pad_ratio=float(args.pad_ratio))
        try:
            results = embedding_client.search(
                category=CAR_CATEGORY,
                image=crop,
                top_k=TOP_K,
                similarity_threshold=0.0,  # contract: no gating
                max_retries=3,
                retry_delay=1.0,
            )
        except RuntimeError as e:
            LOGGER.warning(
                f"{NAME} | Embedding Service search failed for track {track_pos} (degraded to empty): {e}"
            )
            results = []

        for k, r in enumerate((results or [])[:TOP_K]):
            try:
                obj_id = str(r.get("id") or "")
                if obj_id not in uuid_to_int:
                    raise RuntimeError(
                        f"{NAME} | Embedding Service returned unknown label id={obj_id!r} not present in labels (db_digest mismatch)"
                    )
                lid = int(uuid_to_int[obj_id])
                sc = float(r.get("similarity", np.nan))
            except Exception as e:
                raise RuntimeError(f"{NAME} | invalid search result format: {r!r} ({e})") from e
            track_topk_ids[track_pos, k] = int(lid)
            track_topk_scores[track_pos, k] = float(sc)

        # Confidence flags (top1 only)
        top1_id = int(track_topk_ids[track_pos, 0])
        top1_sc = float(track_topk_scores[track_pos, 0])
        track_is_confident_top1[track_pos] = bool(
            (top1_id >= 0) and np.isfinite(top1_sc) and (top1_sc >= float(confidence_threshold_top1))
        )

        det_present_mask[int(frame_pos), int(det_idx)] = True
        det_topk_ids[int(frame_pos), int(det_idx), :] = track_topk_ids[track_pos, :]
        det_topk_scores[int(frame_pos), int(det_idx), :] = track_topk_scores[track_pos, :]
        det_is_confident_top1[int(frame_pos), int(det_idx)] = track_is_confident_top1[track_pos]

        # progress (>=10 updates)
        if (track_pos + 1) % max(1, max(n_tracks // 15, 1)) == 0 or (track_pos + 1) == n_tracks:
            _emit_progress(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=int(track_pos + 1),
                total=int(n_tracks),
                stage="process_frames",
            )

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # Frame-level aggregation: deduplicate labels across detections in frame
    for frame_pos in range(n_frames):
        best: Dict[int, float] = {}
        dets = np.where(det_present_mask[frame_pos])[0].astype(int)
        for det_idx in dets.tolist():
            for k in range(TOP_K):
                lid = int(det_topk_ids[frame_pos, det_idx, k])
                sc = float(det_topk_scores[frame_pos, det_idx, k])
                if lid < 0 or not np.isfinite(sc):
                    continue
                if (lid not in best) or (sc > float(best[lid])):
                    best[lid] = float(sc)
        if best:
            pairs = sorted(best.items(), key=lambda x: float(x[1]), reverse=True)[:TOP_K]
            for k, (lid, sc) in enumerate(pairs):
                frame_topk_ids[frame_pos, k] = int(lid)
                frame_topk_scores[frame_pos, k] = float(sc)
        top1_id = int(frame_topk_ids[frame_pos, 0])
        top1_sc = float(frame_topk_scores[frame_pos, 0])
        frame_is_confident_top1[frame_pos] = bool(
            (top1_id >= 0) and np.isfinite(top1_sc) and (top1_sc >= float(confidence_threshold_top1))
        )

    # Build metadata
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(
            f"{NAME} | frames metadata missing required run identity keys: {missing}"
        )
    
    output_meta: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": status,
        "empty_reason": empty_reason,
        "embedding_service_url": embedding_client.base_url,
        "category": CAR_CATEGORY,
        "top_k": int(TOP_K),
        "confidence_threshold_top1": float(confidence_threshold_top1),
        "proposal_classes": list(proposal_classes),
        "proposal_class_ids": list(map(int, allowed_class_ids)),
        "pad_ratio": float(args.pad_ratio),
        "use_sharpness": bool(args.use_sharpness),
        "max_tracks": (int(args.max_tracks) if args.max_tracks is not None else None),
        "max_dets_per_frame": (int(args.max_dets_per_frame) if args.max_dets_per_frame is not None else None),
        # DB provenance
        "db_name": "embedding_service",
        "db_version": "v1",
        "db_digest": db_digest,
        "db_path": f"{embedding_client.base_url}/categories/{CAR_CATEGORY}",
        "labels_count": int(A),
        # Summary
        "tracks_total": int(n_tracks),
        "tracks_present": int(np.sum(track_present_mask)),
        "dets_present": int(np.sum(det_present_mask)),
    }
    
    # Required run identity fields
    for k in required_run_keys:
        output_meta[k] = meta.get(k)
    
    # Required by contract (baseline may use "unknown")
    output_meta["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

    # PR-3: model system baseline
    from utils.meta_builder import model_used
    output_meta = apply_models_meta(
        output_meta,
        models_used=[
            model_used(
                model_name="embedding_service",
                model_version="v1",
                runtime="http",
                engine="http",
                precision="fp32",
                device="cpu",
            )
        ],
    )
    
    # Baseline contract: stage_timings_ms in meta
    timings["saving"] = 0.0  # Will be updated after save
    timings["total"] = time.perf_counter() - t0
    stage_timings_ms: Dict[str, float] = {}
    for key, value in timings.items():
        stage_timings_ms[key] = float(value) * 1000.0
    output_meta["stage_timings_ms"] = stage_timings_ms

    # Baseline contract: emit save stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="save",
    )
    
    output_dir = os.path.join(str(args.rs_path), NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, ARTIFACT_FILENAME)

    def _build_npz_payload(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_indices": fi_np,
            "times_s": times_s,
            # label space
            "semantic_label_names": semantic_label_names,
            "semantic_object_ids": semantic_object_ids,
            "threshold_per_label_arr": threshold_per_label_arr,
            "semantic_label_make": semantic_label_make,
            "semantic_label_model": semantic_label_model,
            # track axis
            "track_ids": track_ids_arr,
            "track_present_mask": track_present_mask,
            "track_topk_ids": track_topk_ids,
            "track_topk_scores": track_topk_scores,
            "track_is_confident_top1": track_is_confident_top1,
            # frame axis
            "frame_topk_ids": frame_topk_ids,
            "frame_topk_scores": frame_topk_scores,
            "frame_is_confident_top1": frame_is_confident_top1,
            # per-detection axis
            "det_present_mask": det_present_mask,
            "det_topk_ids": det_topk_ids,
            "det_topk_scores": det_topk_scores,
            "det_is_confident_top1": det_is_confident_top1,
            # QA helpers
            "track_best_frame_pos": track_best_frame_pos,
            "track_best_det_idx": track_best_det_idx,
            "track_best_bbox_xyxy": track_best_bbox_xyxy,
            "track_best_det_score": track_best_det_score,
            "track_best_class_id": track_best_class_id,
            # meta
            "meta": np.asarray(meta_dict, dtype=object),
            "meta_json": np.asarray(
                json.dumps(meta_dict, ensure_ascii=False, sort_keys=True),
                dtype="U",
            ),
        }

    # Two-pass write: measure saving time and persist final meta.stage_timings_ms
    t_save_start = time.perf_counter()
    _atomic_save_npz(output_path, **_build_npz_payload(output_meta))
    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0
    output_meta["stage_timings_ms"] = {k: float(v) * 1000.0 for k, v in timings.items()}
    _atomic_save_npz(output_path, **_build_npz_payload(output_meta))

    # Validate artifact (meta + schema if known)
    from utils.artifact_validator import validate_npz  # type: ignore

    ok, issues, _ = validate_npz(output_path)
    if not ok:
        try:
            os.remove(output_path)
        except Exception:
            pass
        msgs = "; ".join([f"{i.level}:{i.message}" for i in issues if getattr(i, "level", "") == "error"])
        raise RuntimeError(f"{NAME} | saved artifact failed validation: {msgs}")

    # Baseline contract: emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )

    LOGGER.info(
        f"{NAME} | Saved results: {output_path} "
        f"(status={status}, tracks_total={n_tracks}, dets_present={int(np.sum(det_present_mask))}, frames={n_frames}, labels={A})"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EmbeddingServiceUnavailableError as ex:
        print(f"{NAME}: {ex}", file=sys.stderr)
        raise SystemExit(1) from None

