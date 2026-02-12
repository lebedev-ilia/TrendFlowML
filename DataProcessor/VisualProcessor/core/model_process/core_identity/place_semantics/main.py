#!/usr/bin/env python3
"""
place_semantics (semantic head, v1)

Place recognition using Embedding Service:
- Processes frames from core_object_detections.frame_indices
- Searches for similar places via Embedding Service
- Groups frames by places into tracks (temporal segmentation)
- Returns per-track and per-frame top-K place identifications
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

import numpy as np

_vp_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
if _vp_root not in sys.path:
    sys.path.append(_vp_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta, model_used  # type: ignore  # noqa: E402

from embedding_service_client import EmbeddingServiceClient

NAME = "place_semantics"
VERSION = "0.1"
SCHEMA_VERSION = "place_semantics_npz_v1"
ARTIFACT_FILENAME = "place_semantics.npz"
LOGGER = get_logger(NAME)

# Place category for Embedding Service
PLACE_CATEGORY = "place"
TOP_K = 5  # Top-5 for places


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


def _group_frames_by_place(
    frame_results: List[List[Dict[str, Any]]],
    frame_indices: List[int],
    times_s: np.ndarray,
    min_track_length: int = 3,
    max_gap_sec: float = 5.0,
) -> Dict[int, List[int]]:
    """
    Group frames into tracks based on place recognition results.
    
    Algorithm:
    1. For each frame, get top-1 place (highest similarity)
    2. Group consecutive frames with the same top-1 place into tracks
    3. Merge tracks if gap is small (max_gap_sec)
    4. Filter out tracks shorter than min_track_length
    
    Args:
        frame_results: List of search results per frame (from Embedding Service)
        frame_indices: List of frame indices
        times_s: Timestamps for frames
        min_track_length: Minimum number of frames in a track
        max_gap_sec: Maximum gap between frames to merge tracks
        
    Returns:
        Dictionary mapping track_id -> list of frame indices
    """
    if not frame_results or not frame_indices:
        return {}
    
    # Get top-1 place for each frame (use place name as identifier)
    frame_place_names: List[Optional[str]] = []
    frame_place_scores: List[float] = []
    
    for results in frame_results:
        if results:
            top_result = results[0]
            place_name = top_result.get("name", "unknown")
            similarity = float(top_result.get("similarity", 0.0))
            # Use place name as identifier (will be mapped to label_id later)
            frame_place_names.append(place_name)
            frame_place_scores.append(similarity)
        else:
            frame_place_names.append(None)
            frame_place_scores.append(0.0)
    
    # Group consecutive frames with same place
    tracks: Dict[int, List[int]] = {}
    current_track_id = 0
    current_place_name = None
    current_track_frames: List[int] = []
    
    for i, (frame_idx, place_name) in enumerate(zip(frame_indices, frame_place_names)):
        if place_name is None:
            # No place detected - start new track
            if current_track_frames:
                if len(current_track_frames) >= min_track_length:
                    tracks[current_track_id] = current_track_frames
                current_track_id += 1
                current_track_frames = []
            current_place_name = None
            continue
        
        if place_name == current_place_name:
            # Same place - continue track
            current_track_frames.append(frame_idx)
        else:
            # Different place - start new track
            if current_track_frames:
                if len(current_track_frames) >= min_track_length:
                    tracks[current_track_id] = current_track_frames
                current_track_id += 1
            
            # Check if we should merge with previous track (small gap)
            if current_place_name is not None and i > 0:
                gap_sec = float(times_s[i]) - float(times_s[i - 1])
                if gap_sec <= max_gap_sec:
                    # Merge with previous track
                    prev_track_id = current_track_id - 1
                    if prev_track_id in tracks:
                        tracks[prev_track_id].extend(current_track_frames)
                        current_track_frames = tracks[prev_track_id]
                        del tracks[prev_track_id]
                        current_track_id -= 1
            
            current_place_name = place_name
            current_track_frames = [frame_idx]
    
    # Add last track
    if current_track_frames and len(current_track_frames) >= min_track_length:
        tracks[current_track_id] = current_track_frames
    
    return tracks


def main() -> int:
    ap = argparse.ArgumentParser("place_semantics")
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
        help="Minimum similarity threshold (default: 0.0)",
    )
    ap.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Minimum number of frames in a track (default: 3)",
    )
    ap.add_argument(
        "--max-gap-sec",
        type=float,
        default=5.0,
        help="Maximum gap between frames to merge tracks (default: 5.0 seconds)",
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
    
    # Initialize Embedding Service client
    embedding_client = EmbeddingServiceClient(base_url=args.embedding_service_url)

    # Create FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=int(meta.get("chunk_size", 32)),
        cache_size=int(meta.get("cache_size", 2)),
    )
    
    t_load_deps_end = time.perf_counter()
    timings["load_deps"] = t_load_deps_end - t_load_deps

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
        
        # Тестовый запрос с первым кадром для проверки работоспособности search endpoint
        if frame_indices:
            try:
                test_frame = frame_manager.get(frame_indices[0])
                test_results = embedding_client.search(
                    category=PLACE_CATEGORY,
                    image=test_frame,
                    top_k=1,  # Минимальный запрос для теста
                    similarity_threshold=0.0,
                    max_retries=1,  # Одна попытка для теста
                    retry_delay=0.5,
                )
                # Если тест прошел успешно, продолжаем обработку
                LOGGER.info(f"{NAME} | Embedding Service test request successful, proceeding with all frames")
            except Exception as test_error:
                # Тестовый запрос не прошел - сервис недоступен или возвращает ошибки
                embedding_service_available = False
                LOGGER.warning(
                    f"{NAME} | Embedding Service test request failed: {test_error}. "
                    f"Skipping all frames to avoid repeated errors. "
                    f"Check Embedding Service status and ensure category '{PLACE_CATEGORY}' is configured."
                )
    except Exception as health_error:
        # Health check не прошел
        embedding_service_available = False
        LOGGER.warning(
            f"{NAME} | Embedding Service health check failed: {health_error}. "
            f"Skipping all frames. Check Embedding Service status."
        )

    # Process each frame (или пропустить, если сервис недоступен)
    frame_results: List[List[Dict[str, Any]]] = []
    total_frames = len(frame_indices)
    processed_frames = 0
    
    if not embedding_service_available:
        # Сервис недоступен - заполняем пустыми результатами для всех кадров
        LOGGER.warning(
            f"{NAME} | Embedding Service unavailable, filling {total_frames} frames with empty results"
        )
        frame_results = [[] for _ in range(total_frames)]
        processed_frames = total_frames
    else:
        # Сервис доступен - обрабатываем все кадры
        for frame_idx, frame_idx_global in enumerate(frame_indices):
            try:
                frame = frame_manager.get(frame_idx_global)
            except Exception as e:
                LOGGER.warning(
                    f"{NAME} | Failed to load frame {frame_idx_global}: {e}"
                )
                frame_results.append([])
                processed_frames += 1
                continue

            # Search in Embedding Service with retry
            try:
                results = embedding_client.search(
                    category=PLACE_CATEGORY,
                    image=frame,
                    top_k=args.topk,
                    similarity_threshold=args.similarity_threshold,
                    max_retries=3,
                    retry_delay=1.0,
                )
                frame_results.append(results)
            except Exception as e:
                LOGGER.error(
                    f"{NAME} | Embedding Service search failed for frame {frame_idx_global} after retries: {e}"
                )
                frame_results.append([])
            
            processed_frames += 1
            # Baseline contract: granular progress (>=10 updates)
            if processed_frames % max(1, (total_frames // 15)) == 0 or processed_frames == total_frames:
                _emit_progress(
                    rs_path=args.rs_path,
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=processed_frames,
                    total=total_frames,
                    stage="process_frames",
                )

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # Group frames into tracks
    tracks = _group_frames_by_place(
        frame_results,
        frame_indices,
        times_s,
        min_track_length=args.min_track_length,
        max_gap_sec=args.max_gap_sec,
    )

    # Build semantic_label_names from results
    all_place_ids: Dict[str, int] = {}  # place_name -> label_id
    label_id_counter = 0

    for results in frame_results:
        for result in results[: args.topk]:
            place_name = result.get("name", "unknown")
            if place_name not in all_place_ids:
                all_place_ids[place_name] = label_id_counter
                label_id_counter += 1

    # Build semantic_label_names array
    place_names_list = sorted(all_place_ids.keys(), key=lambda x: all_place_ids[x])
    semantic_label_names = np.asarray(
        [f"{all_place_ids[name]}:{name}" for name in place_names_list], dtype="U"
    )

    # Build output arrays
    unique_track_ids = sorted(tracks.keys())
    n_frames = len(frame_indices)
    n_tracks = len(unique_track_ids)

    # Track-level arrays
    track_ids_arr = np.asarray(unique_track_ids, dtype=np.int32)  # (T,)
    track_topk_ids = np.zeros((n_tracks, args.topk), dtype=np.int32)  # (T, K)
    track_topk_scores = np.zeros((n_tracks, args.topk), dtype=np.float32)  # (T, K)
    track_present_mask = np.ones((n_tracks,), dtype=np.bool_)  # (T,)
    track_is_confident_top1 = np.zeros((n_tracks,), dtype=np.bool_)  # (T,)

    # Frame-level arrays
    frame_topk_ids = np.full((n_frames, args.topk), -1, dtype=np.int32)  # (N, K)
    frame_topk_scores = np.full((n_frames, args.topk), np.nan, dtype=np.float32)  # (N, K)
    frame_is_confident_top1 = np.zeros((n_frames,), dtype=np.bool_)  # (N,)

    # Fill track-level arrays
    for track_idx, track_id in enumerate(unique_track_ids):
        track_frame_indices = tracks[track_id]
        
        # Aggregate results for this track
        track_place_scores: Dict[str, float] = {}  # place_name -> max_similarity
        
        for frame_idx_local, frame_idx_global in enumerate(frame_indices):
            if frame_idx_global not in track_frame_indices:
                continue
            
            results = frame_results[frame_idx_local]
            for result in results[: args.topk]:
                place_name = result.get("name", "unknown")
                similarity = float(result.get("similarity", 0.0))
                
                if place_name in all_place_ids:
                    # Deduplicate: take best similarity for each place
                    if place_name not in track_place_scores or similarity > track_place_scores[place_name]:
                        track_place_scores[place_name] = similarity
        
        # Sort by similarity and take top-K
        track_results_sorted = [
            (similarity, all_place_ids[place_name])
            for place_name, similarity in track_place_scores.items()
        ]
        track_results_sorted.sort(key=lambda x: x[0], reverse=True)
        
        for k, (similarity, label_id) in enumerate(track_results_sorted[: args.topk]):
            track_topk_ids[track_idx, k] = label_id
            track_topk_scores[track_idx, k] = similarity
        
        # Track confidence flag (top-1)
        if track_results_sorted:
            top1_score = track_results_sorted[0][0]
            track_is_confident_top1[track_idx] = bool(
                np.isfinite(top1_score) and top1_score >= args.similarity_threshold
            )

    # Fill frame-level arrays
    for frame_idx_local, results in enumerate(frame_results):
        frame_place_scores: Dict[str, float] = {}  # place_name -> max_similarity
        
        for result in results[: args.topk]:
            place_name = result.get("name", "unknown")
            similarity = float(result.get("similarity", 0.0))
            
            if place_name in all_place_ids:
                # Deduplicate: take best similarity for each place
                if place_name not in frame_place_scores or similarity > frame_place_scores[place_name]:
                    frame_place_scores[place_name] = similarity
        
        # Sort by similarity and take top-K
        frame_results_sorted = [
            (similarity, all_place_ids[place_name])
            for place_name, similarity in frame_place_scores.items()
        ]
        frame_results_sorted.sort(key=lambda x: x[0], reverse=True)
        
        for k, (similarity, label_id) in enumerate(frame_results_sorted[: args.topk]):
            frame_topk_ids[frame_idx_local, k] = label_id
            frame_topk_scores[frame_idx_local, k] = similarity
        
        # Frame confidence flag (top-1)
        if frame_results_sorted:
            top1_score = frame_results_sorted[0][0]
            frame_is_confident_top1[frame_idx_local] = bool(
                np.isfinite(top1_score) and top1_score >= args.similarity_threshold
            )

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
        "place_category": PLACE_CATEGORY,
        "topk": args.topk,
        "similarity_threshold": args.similarity_threshold,
        "min_track_length": args.min_track_length,
        "max_gap_sec": args.max_gap_sec,
        "num_tracks": n_tracks,
        "num_places": len(all_place_ids),
    }
    
    # Required run identity fields
    for k in required_run_keys:
        output_meta[k] = meta.get(k)
    
    # Required by contract (baseline may use "unknown")
    output_meta["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

    # Add models_used (using model_used helper)
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

    # Threshold arrays (aligned with semantic_label_names)
    threshold_per_label_arr = np.full((len(semantic_label_names),), np.nan, dtype=np.float32)

    np.savez_compressed(
        output_path,
        frame_indices=fi_np,
        times_s=times_s,
        track_ids=track_ids_arr,
        track_topk_ids=track_topk_ids,
        track_topk_scores=track_topk_scores,
        track_present_mask=track_present_mask,
        track_is_confident_top1=track_is_confident_top1,
        frame_topk_ids=frame_topk_ids,
        frame_topk_scores=frame_topk_scores,
        frame_is_confident_top1=frame_is_confident_top1,
        semantic_label_names=semantic_label_names,
        threshold_per_label_arr=threshold_per_label_arr,
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
        f"(tracks={n_tracks}, places={len(all_place_ids)}, frames={n_frames})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
