#!/usr/bin/env python3
"""
brand_semantics (semantic head, v1)

Brand recognition using Embedding Service:
- Detects logo regions from core_object_detections
- Extracts crops with padding
- Searches for similar brands via Embedding Service
- Returns per-track top-K brand identifications
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

_vp_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
if _vp_root not in sys.path:
    sys.path.append(_vp_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta  # type: ignore  # noqa: E402

from crop_utils import crop_with_padding, select_best_crop_for_track
from embedding_service_client import EmbeddingServiceClient

NAME = "brand_semantics"
VERSION = "0.1"
SCHEMA_VERSION = "brand_semantics_npz_v1"
ARTIFACT_FILENAME = "brand_semantics.npz"
LOGGER = get_logger(NAME)

# Brand category for Embedding Service
BRAND_CATEGORY = "brand"
TOP_K = 5


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
        help=f"Number of top results to return per track/frame (default: {TOP_K})"
    )
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity threshold for results (default: 0.0, range: 0.0-1.0)"
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
        help="Maximum detections per frame for track selection (cost control). "
             "If specified, only top detections by score are kept per frame."
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

    # Find logo_region class ID
    logo_class_id = None
    for cid, cname in class_id_to_name.items():
        if cname in ["logo_region", "text_region", "brand"]:
            logo_class_id = cid
            break

    if logo_class_id is None:
        LOGGER.warning(
            f"{NAME} | logo_region class not found in taxonomy, using all detections"
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
    
    # Baseline contract: fail-fast if Embedding Service is unavailable
    try:
        embedding_client._ensure_url()
    except Exception as e:
        raise RuntimeError(
            f"{NAME} | Embedding Service unavailable at {embedding_client.base_url}: {e} (fail-fast)"
        ) from e

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

            # Filter by class if logo_class_id specified
            if logo_class_id is not None:
                if class_ids[frame_idx, det_idx] != logo_class_id:
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

    # Early validation: проверка доступности Embedding Service и тестовый запрос
    embedding_service_available = True
    try:
        # Проверка health endpoint
        embedding_client._ensure_url()
        
        # Тестовый запрос с первым треком для проверки работоспособности search endpoint
        if track_detections:
            first_track_id = next(iter(track_detections.keys()))
            first_detections_list = track_detections[first_track_id]
            if first_detections_list:
                # Получаем первый кадр и делаем кроп
                frame_idx, det_idx, score, bbox = first_detections_list[0]
                frame_idx_global = frame_indices[frame_idx]
                try:
                    test_frame = frame_manager.get(frame_idx_global)
                    test_crop = crop_with_padding(test_frame, bbox, pad_ratio=args.pad_ratio)
                    test_results = embedding_client.search(
                        category=BRAND_CATEGORY,
                        image=test_crop,
                        top_k=1,  # Минимальный запрос для теста
                        similarity_threshold=0.0,
                        max_retries=1,  # Одна попытка для теста
                        retry_delay=0.5,
                    )
                    # Если тест прошел успешно, продолжаем обработку
                    LOGGER.info(f"{NAME} | Embedding Service test request successful, proceeding with all tracks")
                except Exception as test_error:
                    # Тестовый запрос не прошел - сервис недоступен или возвращает ошибки
                    embedding_service_available = False
                    LOGGER.warning(
                        f"{NAME} | Embedding Service test request failed: {test_error}. "
                        f"Skipping all tracks to avoid repeated errors. "
                        f"Check Embedding Service status and ensure category '{BRAND_CATEGORY}' is configured."
                    )
    except Exception as health_error:
        # Health check не прошел
        embedding_service_available = False
        LOGGER.warning(
            f"{NAME} | Embedding Service health check failed: {health_error}. "
            f"Skipping all tracks. Check Embedding Service status."
        )

    # Process each track (или пропустить, если сервис недоступен)
    track_results: Dict[
        int, Tuple[List[Dict[str, Any]], np.ndarray]
    ] = {}  # track_id -> (search_results, crop)

    total_tracks = len(track_detections)
    processed_tracks = 0
    failed_tracks = 0  # Count tracks that failed with errors (not empty results)
    
    if not embedding_service_available:
        # Сервис недоступен - пропускаем обработку всех треков
        LOGGER.warning(
            f"{NAME} | Embedding Service unavailable, skipping {total_tracks} tracks"
        )
        processed_tracks = total_tracks
    else:
        # Сервис доступен - обрабатываем все треки
    for track_id, detections_list in track_detections.items():
        # Apply cost control: max_dets_per_frame
        if args.max_dets_per_frame and len(detections_list) > args.max_dets_per_frame:
            # Sort by score and keep top detections
            detections_list = sorted(
                detections_list, key=lambda x: x[2], reverse=True
            )[: args.max_dets_per_frame]

        # Select best crop for track
        crops = []
        scores_list = []
        areas_list = []

        for frame_idx, det_idx, score, bbox in detections_list:
            # Get frame
            frame_idx_global = frame_indices[frame_idx]
            try:
                frame = frame_manager.get(frame_idx_global)
            except Exception as e:
                LOGGER.warning(
                    f"{NAME} | Failed to load frame {frame_idx_global}: {e}"
                )
                continue

            # Crop with padding
            crop = crop_with_padding(frame, bbox, pad_ratio=args.pad_ratio)
            crops.append(crop)

            # Calculate area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas_list.append(area)
            scores_list.append(score)

        if not crops:
            continue

        # Select best crop
        try:
            best_idx, best_crop = select_best_crop_for_track(
                crops, scores_list, areas_list, use_sharpness=args.use_sharpness
            )
        except Exception as e:
            LOGGER.warning(
                f"{NAME} | Failed to select best crop for track {track_id}: {e}"
            )
            continue

        # Search in Embedding Service with retry
        try:
            results = embedding_client.search(
                category=BRAND_CATEGORY,
                image=best_crop,
                top_k=args.topk,
                similarity_threshold=args.similarity_threshold,
                max_retries=3,
                retry_delay=1.0,
            )
            if not results:
                # Empty results - log warning but continue (this is OK, brand not found)
                LOGGER.warning(
                    f"{NAME} | Embedding Service returned empty results for track {track_id}"
                )
            track_results[track_id] = (results, best_crop)
        except Exception as e:
            failed_tracks += 1
            LOGGER.error(
                f"{NAME} | Embedding Service search failed for track {track_id} after retries: {e}"
            )
            # Continue without this track instead of failing completely
            # But we'll check at the end if ALL tracks failed
            continue
        
        processed_tracks += 1
        # Baseline contract: granular progress (>=10 updates)
        if processed_tracks % max(1, (total_tracks // 15)) == 0 or processed_tracks == total_tracks:
            _emit_progress(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=processed_tracks,
                total=total_tracks,
                stage="process_frames",
            )

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # Baseline contract: fail-fast if all tracks failed with errors
    # (empty results are OK, but errors on all tracks indicate service problem)
    if total_tracks > 0 and failed_tracks == total_tracks and len(track_results) == 0:
        raise RuntimeError(
            f"{NAME} | All {total_tracks} tracks failed with Embedding Service errors. "
            f"Service may be misconfigured or unavailable. "
            f"Check Embedding Service logs and ensure it's running correctly."
        )

    # Build output arrays
    unique_track_ids = sorted(track_results.keys())
    n_frames = len(frame_indices)
    n_tracks = len(unique_track_ids)

    # Track-level arrays
    track_ids_arr = np.asarray(unique_track_ids, dtype=np.int32)  # (T,)
    track_topk_ids = np.zeros((n_tracks, args.topk), dtype=np.int32)  # (T, K)
    track_topk_scores = np.zeros((n_tracks, args.topk), dtype=np.float32)  # (T, K)

    # Frame-level arrays (aggregate per-frame)
    frame_topk_ids = np.zeros((n_frames, args.topk), dtype=np.int32)  # (N, K)
    frame_topk_scores = np.zeros((n_frames, args.topk), dtype=np.float32)  # (N, K)

    # Build semantic_label_names from results
    all_brand_ids: Dict[str, int] = {}  # brand_name -> label_id
    label_id_counter = 0

    for track_idx, track_id in enumerate(unique_track_ids):
        results, _ = track_results[track_id]

        # Track-level top-K
        for k, result in enumerate(results[: args.topk]):
            brand_name = result.get("name", "unknown")
            similarity = float(result.get("similarity", 0.0))
            brand_id = result.get("id", "")

            # Map brand name to label ID
            if brand_name not in all_brand_ids:
                all_brand_ids[brand_name] = label_id_counter
                label_id_counter += 1

            label_id = all_brand_ids[brand_name]
            track_topk_ids[track_idx, k] = label_id
            track_topk_scores[track_idx, k] = similarity

    # Build semantic_label_names array
    brand_names_list = sorted(all_brand_ids.keys(), key=lambda x: all_brand_ids[x])
    semantic_label_names = np.asarray(
        [f"{all_brand_ids[name]}:{name}" for name in brand_names_list], dtype="U"
    )

    # Frame-level aggregation (for each frame, find best match from tracks)
    # Fixed: deduplication by brand_name - take best similarity for each brand
    for frame_idx in range(n_frames):
        frame_idx_global = frame_indices[frame_idx]
        brand_scores: Dict[str, float] = {}  # brand_name -> max_similarity

        for track_id, detections_list in track_detections.items():
            if track_id not in track_results:
                continue

            # Check if track has detection on this frame
            has_detection = any(
                det_frame_idx == frame_idx for det_frame_idx, _, _, _ in detections_list
            )

            if not has_detection:
                continue

            results, _ = track_results[track_id]
            for result in results[: args.topk]:
                brand_name = result.get("name", "unknown")
                similarity = float(result.get("similarity", 0.0))

                if brand_name in all_brand_ids:
                    # Deduplicate: take best similarity for each brand
                    if brand_name not in brand_scores or similarity > brand_scores[brand_name]:
                        brand_scores[brand_name] = similarity

        # Sort by similarity and take top-K (deduplicated)
        frame_results = [
            (similarity, all_brand_ids[brand_name])
            for brand_name, similarity in brand_scores.items()
        ]
        frame_results.sort(key=lambda x: x[0], reverse=True)
        for k, (similarity, label_id) in enumerate(frame_results[: args.topk]):
            frame_topk_ids[frame_idx, k] = label_id
            frame_topk_scores[frame_idx, k] = similarity

    # Build metadata
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(
            f"{NAME} | frames metadata missing required run identity keys: {missing}"
        )
    
    output_meta = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ok",
        "empty_reason": None,
        "embedding_service_url": embedding_client.base_url,
        "brand_category": BRAND_CATEGORY,
        "topk": args.topk,
        "similarity_threshold": args.similarity_threshold,
        "pad_ratio": args.pad_ratio,
        "use_sharpness": args.use_sharpness,
        "num_tracks": n_tracks,
        "num_brands": len(all_brand_ids),
    }
    
    # Required run identity fields
    for k in required_run_keys:
        output_meta[k] = meta.get(k)
    
    # Required by contract (baseline may use "unknown")
    output_meta["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

    # Add models_used (using model_used helper)
    from utils.meta_builder import model_used
    models_used_list = [
        model_used(
            model_name="embedding_service",
            model_version="v1",
            runtime="http",
            engine="http",
            precision="fp32",
            device="cpu",  # Embedding Service runs on server
        )
    ]
    from utils.meta_builder import apply_models_meta
    output_meta = apply_models_meta(output_meta, models_used=models_used_list)
    
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
    
    t_save_start = time.perf_counter()

    # Save output
    output_dir = os.path.join(str(args.rs_path), NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, ARTIFACT_FILENAME)

    np.savez_compressed(
        output_path,
        frame_indices=fi_np,
        times_s=times_s,
        track_ids=track_ids_arr,
        track_topk_ids=track_topk_ids,
        track_topk_scores=track_topk_scores,
        frame_topk_ids=frame_topk_ids,
        frame_topk_scores=frame_topk_scores,
        semantic_label_names=semantic_label_names,
        meta=np.asarray(output_meta, dtype=object),
    )
    
    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0
    
    # Update stage_timings_ms with final timings
    stage_timings_ms = {}
    for key, value in timings.items():
        stage_timings_ms[key] = float(value) * 1000.0
    output_meta["stage_timings_ms"] = stage_timings_ms

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
        f"(tracks={n_tracks}, brands={len(all_brand_ids)}, frames={n_frames})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

