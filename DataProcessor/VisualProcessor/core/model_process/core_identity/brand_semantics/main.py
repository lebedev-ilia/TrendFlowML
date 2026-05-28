#!/usr/bin/env python3
"""
brand_semantics (semantic head, Audit v3)

Brand recognition using Embedding Service:
- Detects logo regions from core_object_detections
- Extracts crops with padding
- Searches for similar brands via Embedding Service
- Returns per-detection / per-frame / per-track top-K brand identifications
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    spec = importlib.util.spec_from_file_location("brand_semantics.crop_utils", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"brand_semantics | crop_utils not found: {p}")
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
                raise RuntimeError("brand_semantics | Failed to load EmbeddingServiceClient from utils")
        else:
            raise ImportError("brand_semantics | embedding_service_client not found. Expected at: %s" % _utils_path)

NAME = "brand_semantics"
VERSION = "0.2"
SCHEMA_VERSION = "brand_semantics_npz_v2"
ARTIFACT_FILENAME = "brand_semantics.npz"
LOGGER = get_logger(NAME)

# Brand category for Embedding Service
BRAND_CATEGORY = "brand"
TOP_K = 5  # Contract: fixed K=5 for semantic-head v1


def _atomic_save_npz(out_path: str, **kwargs: Any) -> None:
    """Atomic NPZ save (avoid partially-written artifacts)."""
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".",
        suffix=".npz",
        dir=out_dir,
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **kwargs)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    """
    Extract and validate frame indices from metadata.

    Args:
        meta: Metadata dictionary from metadata.json

    Returns:
        List of frame indices as integers

    Raises:
        RuntimeError: If frame_indices are missing or invalid
    """
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
    """
    Load NPZ file and handle object arrays properly.

    Args:
        path: Path to NPZ file

    Returns:
        Dictionary with loaded arrays and objects

    Raises:
        RuntimeError: If file not found
    """
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
    """
    Parse class_names array into id->name mapping.

    Expected format: ["id:name", "id:name", ...]

    Args:
        class_names: Array of class name strings in "id:name" format

    Returns:
        Dictionary mapping class ID to class name
    """
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


def main() -> int:
    """
    Main entry point for brand_semantics component.

    This function processes video frames to recognize brands/logo regions:
    1. Loads detections from core_object_detections
    2. Groups detections by tracks
    3. Extracts crops with padding from frames
    4. Selects best crop per track based on quality metrics
    5. Searches for similar brands via Embedding Service
    6. Outputs results in NPZ format

    The component follows the semantic head contract:
    - Requires core_object_detections.frame_indices
    - Aligns output to same frame_indices
    - Outputs per-track and per-frame top-K results

    Returns:
        0 on success, non-zero on error

    Example:
        ```bash
        python main.py \
            --frames-dir /path/to/frames \
            --rs-path /path/to/result_store \
            --embedding-service-url http://localhost:8001 \
            --topk 5 \
            --similarity-threshold 0.7 \
            --max-tracks 100 \
            --pad-ratio 0.15 \
            --use-sharpness
        ```
    """
    ap = argparse.ArgumentParser(
        "brand_semantics",
        description="Brand recognition component using Embedding Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --frames-dir frames/ --rs-path results/ --embedding-service-url http://localhost:8001

  # With cost controls
  python main.py --frames-dir frames/ --rs-path results/ --max-tracks 50 --max-dets-per-frame 5

  # With quality settings
  python main.py --frames-dir frames/ --rs-path results/ --pad-ratio 0.20 --use-sharpness
        """
    )
    ap.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing video frames and metadata.json"
    )
    ap.add_argument(
        "--rs-path",
        required=True,
        help="Result store path (e.g., result_store/platform/video/run)"
    )
    ap.add_argument(
        "--embedding-service-url",
        default=None,
        help="Embedding Service URL (default: from EMBEDDING_SERVICE_URL env or http://localhost:8001)"
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=TOP_K,
        help=f"Number of top results to return (contract: must be {TOP_K})"
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
        "--proposal-classes",
        type=str,
        default="logo_region,text_region",
        help=(
            "Comma-separated list of proposal class names from core_object_detections taxonomy "
            '(default: "logo_region,text_region").'
        ),
    )
    ap.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Maximum number of tracks to process (cost control). "
             "If specified, only top tracks by length are processed."
    )
    ap.add_argument(
        "--max-dets-per-frame",
        type=int,
        default=None,
        help=(
            "Cost control. Maximum number of detections kept per track before selecting best crop "
            "(historical flag name; applies to per-track candidate detections)."
        ),
    )
    ap.add_argument(
        "--pad-ratio",
        type=float,
        default=0.15,
        help="Padding ratio for crops (default: 0.15 = 15%% padding on each side)"
    )
    ap.add_argument(
        "--use-sharpness",
        action="store_true",
        help="Use sharpness (Laplacian variance) as additional metric for selecting best crop per track. "
             "This helps select sharper images which may have better recognition quality."
    )
    args = ap.parse_args()

    # Contract: fixed K=5 (downstream encoder expects consistent shape)
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

    # Resolve proposal classes to class_ids (strict: fail-fast if taxonomy mismatch)
    available_class_names = set(class_id_to_name.values())
    missing = [c for c in proposal_classes if c not in available_class_names]
    if missing:
        raise RuntimeError(
            f"{NAME} | proposal_classes not found in core_object_detections taxonomy: {missing}"
        )
    allowed_class_ids = {
        int(cid) for cid, cname in class_id_to_name.items() if cname in set(proposal_classes)
    }
    if not allowed_class_ids:
        raise RuntimeError(
            f"{NAME} | resolved proposal_classes produced empty class_id set (contract)"
        )

    # Get tracks (if available) or generate per-detection tracks
    if "tracks" in detections:
        tracks = np.asarray(detections.get("tracks"), dtype=np.int32)  # (N, MAX)
        # Validate tracks shape
        if tracks.shape != boxes.shape:
            raise RuntimeError(
                f"{NAME} | tracks shape {tracks.shape} != boxes shape {boxes.shape}"
            )
    else:
        # WARNING: Generating per-detection track IDs - this breaks tracking!
        # Each detection will get a unique track ID, which may not be correct.
        # For proper tracking, tracks should come from core_object_detections.
        LOGGER.warning(
            f"{NAME} | tracks not found in detections.npz. "
            f"Generating per-detection track IDs - this may produce incorrect results. "
            f"Each detection will be treated as a separate track. "
            f"Consider ensuring core_object_detections provides proper tracking."
        )
        track_counter = 0
        tracks = np.full_like(class_ids, -1, dtype=np.int32)
        for i in range(boxes.shape[0]):
            for j in range(boxes.shape[1]):
                if valid_mask[i, j]:
                    tracks[i, j] = track_counter
                    track_counter += 1

    # Initialize Embedding Service client
    embedding_client = EmbeddingServiceClient(base_url=args.embedding_service_url)
    
    # Baseline contract: fail-fast if Embedding Service is unavailable (EmbeddingServiceUnavailableError)
    embedding_client._ensure_url()

    # Load label-space (db provenance + deterministic UUID->int32 mapping)
    labels = embedding_client.get_labels(category=BRAND_CATEGORY)
    if not labels:
        raise RuntimeError(
            f"{NAME} | Embedding Service category '{BRAND_CATEGORY}' has 0 labels (fail-fast)"
        )

    def _canon_label_row(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(r.get("id") or ""),
            "name": str(r.get("name") or ""),
            "embedding_model": str(r.get("embedding_model") or ""),
            "embedding_dim": int(r.get("embedding_dim") or 0),
            "updated_at": str(r.get("updated_at") or ""),
        }

    labels_canon = [_canon_label_row(r) for r in labels]
    labels_canon = [r for r in labels_canon if r["id"]]
    if not labels_canon:
        raise RuntimeError(
            f"{NAME} | Embedding Service returned invalid labels for '{BRAND_CATEGORY}' (no ids)"
        )
    labels_canon.sort(key=lambda r: r["id"])

    db_digest = _sha256_hex(
        json.dumps(labels_canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )

    uuid_to_int: Dict[str, int] = {r["id"]: i for i, r in enumerate(labels_canon)}
    semantic_object_ids = np.asarray([r["id"] for r in labels_canon], dtype="U")
    semantic_label_names = np.asarray(
        [f"{i}:{labels_canon[i]['name']}" for i in range(len(labels_canon))],
        dtype="U",
    )
    threshold_per_label_arr = np.full((len(labels_canon),), np.nan, dtype=np.float32)

    embedding_models = sorted({r["embedding_model"] for r in labels_canon if r["embedding_model"]})
    embedding_model = embedding_models[0] if len(embedding_models) == 1 else ""

    # Create FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=int(meta.get("chunk_size", 32)),
        cache_size=int(meta.get("cache_size", 2)),
    )

    # Group detections by track
    track_detections: Dict[int, List[Tuple[int, int, float, np.ndarray]]] = (
        defaultdict(list)
    )  # track_id -> [(frame_idx, det_idx, score, bbox)]

    for frame_idx, frame_idx_global in enumerate(frame_indices):
        for det_idx in range(boxes.shape[1]):
            if not valid_mask[frame_idx, det_idx]:
                continue

            # Filter by allowed proposal classes (strict set)
            if int(class_ids[frame_idx, det_idx]) not in allowed_class_ids:
                continue

            track_id = int(tracks[frame_idx, det_idx])
            if track_id < 0:
                continue

            score = float(scores[frame_idx, det_idx])
            bbox = boxes[frame_idx, det_idx].copy()  # (x1, y1, x2, y2)

            track_detections[track_id].append(
                (frame_idx, det_idx, score, bbox)
            )

    # Apply cost control: max_tracks
    if args.max_tracks and len(track_detections) > args.max_tracks:
        # Sort by track length (number of detections) and keep top tracks
        sorted_tracks = sorted(
            track_detections.items(), key=lambda x: len(x[1]), reverse=True
        )
        track_detections = dict(sorted_tracks[: args.max_tracks])
        LOGGER.info(
            f"{NAME} | Limited to {args.max_tracks} tracks (cost control)"
        )

    # Baseline contract: emit process_frames stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="process_frames",
    )
    
    t_process_start = time.perf_counter()
    # Build output arrays (semantic-head contract v1, brand_semantics_npz_v2)
    track_ids_sorted = sorted(track_detections.keys())
    n_frames = int(len(frame_indices))
    max_dets = int(boxes.shape[1])
    n_tracks = int(len(track_ids_sorted))

    track_ids_arr = np.asarray(track_ids_sorted, dtype=np.int32)
    track_present_mask = np.zeros((n_tracks,), dtype=bool)
    track_topk_ids = np.full((n_tracks, TOP_K), -1, dtype=np.int32)
    track_topk_scores = np.full((n_tracks, TOP_K), np.nan, dtype=np.float32)
    track_is_confident_top1 = np.zeros((n_tracks,), dtype=bool)

    # Debug/QA helpers for render assets (best crop coordinates per track)
    track_best_frame_pos = np.full((n_tracks,), -1, dtype=np.int32)
    track_best_det_idx = np.full((n_tracks,), -1, dtype=np.int32)
    track_best_bbox_xyxy = np.full((n_tracks, 4), np.nan, dtype=np.float32)
    track_best_det_score = np.full((n_tracks,), np.nan, dtype=np.float32)
    track_best_class_id = np.full((n_tracks,), -1, dtype=np.int32)

    det_present_mask = np.zeros((n_frames, max_dets), dtype=bool)
    det_topk_ids = np.full((n_frames, max_dets, TOP_K), -1, dtype=np.int32)
    det_topk_scores = np.full((n_frames, max_dets, TOP_K), np.nan, dtype=np.float32)
    det_is_confident_top1 = np.zeros((n_frames, max_dets), dtype=bool)

    frame_topk_ids = np.full((n_frames, TOP_K), -1, dtype=np.int32)
    frame_topk_scores = np.full((n_frames, TOP_K), np.nan, dtype=np.float32)
    frame_is_confident_top1 = np.zeros((n_frames,), dtype=bool)

    processed_tracks = 0
    total_tracks = n_tracks

    for track_pos, track_id in enumerate(track_ids_sorted):
        detections_list = track_detections[track_id]

        # Cost control: limit candidate detections per track
        if args.max_dets_per_frame and len(detections_list) > args.max_dets_per_frame:
            detections_list = sorted(detections_list, key=lambda x: x[2], reverse=True)[
                : args.max_dets_per_frame
            ]

        crops: List[np.ndarray] = []
        scores_list: List[float] = []
        areas_list: List[float] = []
        crop_meta: List[Tuple[int, int, float, np.ndarray, int]] = []

        for frame_pos, det_idx, det_score, bbox in detections_list:
            frame_idx_global = frame_indices[frame_pos]
            try:
                frame = frame_manager.get(frame_idx_global)
            except Exception as e:
                LOGGER.warning(f"{NAME} | Failed to load frame {frame_idx_global}: {e}")
                continue

            crop = crop_with_padding(frame, bbox, pad_ratio=args.pad_ratio)
            crops.append(crop)
            scores_list.append(float(det_score))
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            areas_list.append(area)
            crop_meta.append(
                (int(frame_pos), int(det_idx), float(det_score), bbox.astype(np.float32), int(class_ids[frame_pos, det_idx]))
            )

        if not crops:
            continue

        best_idx, best_crop = select_best_crop_for_track(
            crops, scores_list, areas_list, use_sharpness=args.use_sharpness
        )
        best_frame_pos, best_det_idx, best_det_score, best_bbox, best_class_id = crop_meta[int(best_idx)]

        # Contract: no gating by thresholds -> always request top-K with similarity_threshold=0.0
        try:
            results = embedding_client.search(
                category=BRAND_CATEGORY,
                image=best_crop,
                top_k=TOP_K,
                similarity_threshold=0.0,
                max_retries=3,
                retry_delay=1.0,
            )
        except RuntimeError as e:
            LOGGER.warning(
                f"{NAME} | Embedding Service search failed for track {track_id} (degraded to empty): {e}"
            )
            results = []

        track_present_mask[track_pos] = True
        track_best_frame_pos[track_pos] = int(best_frame_pos)
        track_best_det_idx[track_pos] = int(best_det_idx)
        track_best_bbox_xyxy[track_pos, :] = best_bbox.reshape(4)
        track_best_det_score[track_pos] = float(best_det_score)
        track_best_class_id[track_pos] = int(best_class_id)

        for k, r in enumerate(results[:TOP_K]):
            oid = str(r.get("id") or "")
            if oid not in uuid_to_int:
                continue
            track_topk_ids[track_pos, k] = int(uuid_to_int[oid])
            track_topk_scores[track_pos, k] = float(r.get("similarity", np.nan))

        top1 = float(track_topk_scores[track_pos, 0]) if not np.isnan(track_topk_scores[track_pos, 0]) else float("nan")
        if not np.isnan(top1) and top1 >= confidence_threshold_top1:
            track_is_confident_top1[track_pos] = True

        # Fill per-detection outputs for all candidate detections in this track (post cost-control)
        for frame_pos, det_idx, _det_score, _bbox in detections_list:
            det_present_mask[int(frame_pos), int(det_idx)] = True
            det_topk_ids[int(frame_pos), int(det_idx), :] = track_topk_ids[track_pos, :]
            det_topk_scores[int(frame_pos), int(det_idx), :] = track_topk_scores[track_pos, :]
            if track_is_confident_top1[track_pos]:
                det_is_confident_top1[int(frame_pos), int(det_idx)] = True

        processed_tracks += 1
        if total_tracks > 0 and (
            processed_tracks % max(1, (total_tracks // 15)) == 0
            or processed_tracks == total_tracks
        ):
            _emit_progress(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=processed_tracks,
                total=total_tracks,
                stage="process_frames",
            )

    # Frame-level aggregation: deduplicate by label_id, take best similarity across detections
    for frame_pos in range(n_frames):
        best_by_label: Dict[int, float] = {}
        for det_idx in range(max_dets):
            if not det_present_mask[frame_pos, det_idx]:
                continue
            for k in range(TOP_K):
                lid = int(det_topk_ids[frame_pos, det_idx, k])
                sc = float(det_topk_scores[frame_pos, det_idx, k])
                if lid < 0 or np.isnan(sc):
                    continue
                prev = best_by_label.get(lid)
                if prev is None or sc > prev:
                    best_by_label[lid] = sc

        if not best_by_label:
            continue

        items = sorted(best_by_label.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        for k, (lid, sc) in enumerate(items):
            frame_topk_ids[frame_pos, k] = int(lid)
            frame_topk_scores[frame_pos, k] = float(sc)

        top1 = float(frame_topk_scores[frame_pos, 0])
        if not np.isnan(top1) and top1 >= confidence_threshold_top1:
            frame_is_confident_top1[frame_pos] = True

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # Build metadata (semantic-head contract v1)
    required_run_keys = [
        "platform_id",
        "video_id",
        "run_id",
        "sampling_policy_version",
        "config_hash",
    ]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(
            f"{NAME} | frames metadata missing required run identity keys: {missing}"
        )

    has_any_proposals = bool(track_detections)
    has_any_present_track = bool(np.any(track_present_mask))
    if not has_any_proposals:
        status = "empty"
        empty_reason = "no_logo_proposals"
    elif not has_any_present_track:
        status = "empty"
        empty_reason = "no_valid_crops"
    else:
        status = "ok"
        empty_reason = None

    output_meta: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": status,
        "empty_reason": empty_reason,
        # DB provenance
        "db_name": "embedding_service",
        "db_version": "v1",
        "db_digest": db_digest,
        "db_path": f"{embedding_client.base_url}/categories/{BRAND_CATEGORY}",
        # Service/runtime
        "embedding_service_url": embedding_client.base_url,
        "brand_category": BRAND_CATEGORY,
        "embedding_model": embedding_model,
        # Config highlights
        "topk": TOP_K,
        "confidence_threshold_top1": confidence_threshold_top1,
        "proposal_classes": proposal_classes,
        "pad_ratio": float(args.pad_ratio),
        "use_sharpness": bool(args.use_sharpness),
        "max_tracks": int(args.max_tracks) if args.max_tracks is not None else None,
        "max_dets_per_track": int(args.max_dets_per_frame) if args.max_dets_per_frame is not None else None,
        # Stats
        "tracks_total": int(n_tracks),
        "tracks_present": int(np.sum(track_present_mask)),
        "dets_present": int(np.sum(det_present_mask)),
    }

    for k in required_run_keys:
        output_meta[k] = meta.get(k)
    output_meta["dataprocessor_version"] = str(
        meta.get("dataprocessor_version") or "unknown"
    )

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

    # Stage timings (ms)
    timings["saving"] = 0.0
    timings["total"] = time.perf_counter() - t0
    output_meta["stage_timings_ms"] = {
        k: float(v) * 1000.0 for k, v in timings.items()
    }

    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="save",
    )

    output_dir = os.path.join(str(args.rs_path), NAME)
    output_path = os.path.join(output_dir, ARTIFACT_FILENAME)

    def _build_npz_payload(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_indices": fi_np,
            "times_s": times_s,
            # label space
            "semantic_label_names": semantic_label_names,
            "semantic_object_ids": semantic_object_ids,
            "threshold_per_label_arr": threshold_per_label_arr,
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
            # QA helpers for render assets
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
    output_meta["stage_timings_ms"] = {
        k: float(v) * 1000.0 for k, v in timings.items()
    }
    _atomic_save_npz(output_path, **_build_npz_payload(output_meta))

    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )

    LOGGER.info(
        f"{NAME} | Saved results: {output_path} "
        f"(tracks_total={n_tracks}, tracks_present={int(np.sum(track_present_mask))}, "
        f"dets_present={int(np.sum(det_present_mask))}, frames={n_frames}, labels={len(labels_canon)})"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EmbeddingServiceUnavailableError as ex:
        print(f"{NAME}: {ex}", file=sys.stderr)
        raise SystemExit(1) from None

