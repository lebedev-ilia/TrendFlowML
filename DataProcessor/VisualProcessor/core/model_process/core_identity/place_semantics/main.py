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
import hashlib
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
from utils.meta_builder import apply_models_meta, model_used  # type: ignore  # noqa: E402
from utils.embedding_service_errors import EmbeddingServiceUnavailableError  # type: ignore  # noqa: E402

# Import EmbeddingServiceClient (try utils directory first, then fallback)
try:
    from utils.embedding_service_client import EmbeddingServiceClient
except ImportError:
    # Fallback: try direct import (should work if path is set correctly)
    try:
        from embedding_service_client import EmbeddingServiceClient
    except ImportError:
        # Last resort: try to load from file directly
        _embedding_client_path = Path(__file__).parent / "utils" / "embedding_service_client.py"
        if _embedding_client_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("embedding_service_client", str(_embedding_client_path))
            if spec and spec.loader:
                embedding_service_client_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(embedding_service_client_module)
                EmbeddingServiceClient = embedding_service_client_module.EmbeddingServiceClient
            else:
                raise ImportError("Failed to load EmbeddingServiceClient from utils/embedding_service_client.py")
        else:
            raise ImportError(
                f"place_semantics | embedding_service_client not found. "
                f"Expected at: {_embedding_client_path}"
            )

NAME = "place_semantics"
VERSION = "0.2"
SCHEMA_VERSION = "place_semantics_npz_v2"
ARTIFACT_FILENAME = "place_semantics.npz"
LOGGER = get_logger(NAME)

# Place category for Embedding Service
PLACE_CATEGORY = "place"
TOP_K = 5  # Top-5 for places


def _place_embeddings_to_matrix(rows: List[Any]) -> np.ndarray:
    """Stack DB place embeddings into (M, D) float32; raises ValueError if ragged or empty."""
    out: List[np.ndarray] = []
    for emb in rows:
        a = np.asarray(emb, dtype=np.float32).reshape(-1)
        if a.size == 0:
            raise ValueError("empty place embedding row")
        out.append(a)
    if not out:
        raise ValueError("no place embedding rows")
    d0 = int(out[0].shape[0])
    for i, r in enumerate(out):
        if int(r.shape[0]) != d0:
            raise ValueError(
                f"ragged place embeddings: row0 dim {d0} vs row{i} dim {int(r.shape[0])}"
            )
    return np.stack(out, axis=0)


def _sha256_hex(s: str) -> str:
    """Compute SHA256 hex digest of a string."""
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


def _load_npz(path: str) -> Dict[str, Any]:
    """Load NPZ file and handle object arrays properly."""
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
    z.close()
    return out


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
        "--threshold-global",
        type=float,
        default=0.23,
        help="Global threshold for is_confident flag (default: 0.23, used only for flags, not gating)",
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
    ap.add_argument(
        "--http-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout (seconds) for Embedding Service (default: 120)",
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
    
    # Load core_clip embeddings (required by schema)
    core_clip_path = os.path.join(str(args.rs_path), "core_clip", "embeddings.npz")
    clip_npz = _load_npz(core_clip_path)
    clip_meta = clip_npz.get("meta")
    upstream_models_used: List[Dict[str, Any]] = []
    upstream_model_signature: Any = None
    if isinstance(clip_meta, dict):
        if isinstance(clip_meta.get("models_used"), list):
            upstream_models_used = clip_meta.get("models_used") or []
        upstream_model_signature = clip_meta.get("model_signature")

    clip_fi = np.asarray(clip_npz.get("frame_indices"), dtype=np.int32).reshape(-1)
    clip_emb = np.asarray(clip_npz.get("frame_embeddings"), dtype=np.float32)
    if clip_fi.size == 0 or clip_emb.size == 0:
        raise RuntimeError(
            f"{NAME} | core_clip embeddings.npz missing frame_indices/frame_embeddings (no-fallback)"
        )
    if clip_emb.ndim != 2 or clip_emb.shape[0] != clip_fi.shape[0]:
        raise RuntimeError(f"{NAME} | core_clip frame_embeddings invalid shape: {clip_emb.shape}")

    # Normalize embeddings (L2 norm)
    def _l2norm_rows(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
        return x / n
    
    clip_emb = _l2norm_rows(clip_emb)

    clip_map: Dict[int, int] = {int(u): int(i) for i, u in enumerate(clip_fi.tolist())}
    sel_rows: List[int] = []
    for u in frame_indices:
        if int(u) not in clip_map:
            raise RuntimeError(
                f"{NAME} | core_clip embeddings do not cover required frame_index={u} (no-fallback)"
            )
        sel_rows.append(int(clip_map[int(u)]))
    frame_emb = clip_emb[np.asarray(sel_rows, dtype=np.int32)]  # (N, D)
    
    # Initialize Embedding Service client (fail-fast if unavailable → EmbeddingServiceUnavailableError)
    embedding_client = EmbeddingServiceClient(
        base_url=args.embedding_service_url, timeout=args.http_timeout
    )
    embedding_client._ensure_url()

    # Load label-space (db provenance + deterministic UUID->int32 mapping)
    labels = embedding_client.get_labels(category=PLACE_CATEGORY)
    if not labels:
        raise RuntimeError(
            f"{NAME} | Embedding Service category '{PLACE_CATEGORY}' has 0 labels (fail-fast)"
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
            f"{NAME} | Embedding Service returned invalid labels for '{PLACE_CATEGORY}' (no ids)"
        )
    labels_canon.sort(key=lambda r: r["id"])

    db_digest = _sha256_hex(
        json.dumps(labels_canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )

    # Build canonical label space (stable UUID->int32 mapping)
    uuid_to_int: Dict[str, int] = {r["id"]: i for i, r in enumerate(labels_canon)}
    canonical_semantic_object_ids = np.asarray([r["id"] for r in labels_canon], dtype="U")
    canonical_semantic_label_names = np.asarray(
        [f"{i}:{labels_canon[i]['name']}" for i in range(len(labels_canon))],
        dtype="U",
    )
    
    embedding_models = sorted({r["embedding_model"] for r in labels_canon if r["embedding_model"]})
    embedding_model = embedding_models[0] if len(embedding_models) == 1 else ""
    
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

    n_frames = len(frame_indices)
    frame_results: List[List[Dict[str, Any]]] = [[] for _ in range(n_frames)]
    
    # Try to use embeddings directly (much faster than HTTP requests per frame)
    embedding_service_available = True
    use_embedding_direct = True
    place_embeddings: Optional[List[Dict[str, Any]]] = None
    
    try:
        # Get all place embeddings from Embedding Service once, then compare locally
        place_embeddings = embedding_client.get_all_embeddings(
            category=PLACE_CATEGORY,
            embedding_model=embedding_model if embedding_model else None,
        )
        if not place_embeddings or len(place_embeddings) == 0:
            LOGGER.warning(f"{NAME} | No place embeddings available, falling back to image search")
            use_embedding_direct = False
        else:
            LOGGER.info(f"{NAME} | Using direct embedding comparison with {len(place_embeddings)} places (10-50x faster)")
    except Exception as emb_error:
        LOGGER.warning(
            f"{NAME} | Failed to get place embeddings: {emb_error}. "
            f"Falling back to image search."
        )
        use_embedding_direct = False

    direct_path_completed = False
    K = int(args.topk)
    if use_embedding_direct and place_embeddings:
        place_emb_by_uuid: Dict[str, Dict[str, Any]] = {
            str(p.get("id") or ""): p for p in place_embeddings if p.get("id")
        }

        place_emb_raw: List[Any] = []
        place_uuid_list: List[str] = []
        for uuid in canonical_semantic_object_ids:
            uid = str(uuid)
            if uid in place_emb_by_uuid:
                place_emb_raw.append(place_emb_by_uuid[uid]["embedding"])
                place_uuid_list.append(uid)

        if not place_emb_raw:
            LOGGER.warning(
                f"{NAME} | No place embeddings match canonical label space, falling back to image search"
            )
        else:
            try:
                place_emb_matrix = _place_embeddings_to_matrix(place_emb_raw)
                fdim = int(frame_emb.shape[1])
                if place_emb_matrix.shape[1] != fdim:
                    raise ValueError(
                        f"place embedding dim {place_emb_matrix.shape[1]} != "
                        f"frame_emb dim {fdim} (check embedding_model vs core_clip)"
                    )

                place_norms = np.linalg.norm(place_emb_matrix, axis=1, keepdims=True)
                place_norms = np.where(place_norms > 1e-10, place_norms, 1.0)
                place_emb_normalized = place_emb_matrix / place_norms

                frame_emb_normalized = _l2norm_rows(frame_emb)

                similarities = np.dot(frame_emb_normalized, place_emb_normalized.T)

                similarity_threshold = float(args.similarity_threshold)
                if similarity_threshold > 0:
                    similarities = np.where(
                        similarities >= similarity_threshold, similarities, -1.0
                    )

                topk_indices = np.argsort(similarities, axis=1)[:, -K:][:, ::-1]
                topk_similarities = np.take_along_axis(similarities, topk_indices, axis=1)

                for frame_idx in range(n_frames):
                    frame_result = []
                    for k_idx in range(K):
                        emb_idx = int(topk_indices[frame_idx, k_idx])
                        if emb_idx >= len(place_uuid_list):
                            continue
                        place_uuid = place_uuid_list[emb_idx]
                        similarity = float(topk_similarities[frame_idx, k_idx])

                        if similarity < similarity_threshold:
                            continue

                        place_data = place_emb_by_uuid[place_uuid]
                        frame_result.append(
                            {
                                "id": str(place_uuid),
                                "name": str(place_data.get("name", "unknown")),
                                "similarity": similarity,
                                "metadata": place_data.get("metadata", {}),
                            }
                        )
                    frame_results[frame_idx] = frame_result
                direct_path_completed = True
            except Exception as e:
                LOGGER.warning(
                    f"{NAME} | Direct place embedding comparison failed ({e}); "
                    f"falling back to image search"
                )
                frame_results = [[] for _ in range(n_frames)]

    if direct_path_completed:
        _emit_progress(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            done=n_frames,
            total=n_frames,
            stage="process_frames",
        )
    elif not embedding_service_available:
        # Service unavailable - fill with empty results
        LOGGER.warning(
            f"{NAME} | Embedding Service unavailable, filling {n_frames} frames with empty results"
        )
        processed_frames = n_frames
        _emit_progress(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            done=processed_frames,
            total=n_frames,
            stage="process_frames",
        )
    else:
        # Fallback: Use image-based search (slower, but works if embeddings unavailable)
        from utils.frame_manager import FrameManager
        frame_manager = FrameManager(
            frames_dir=args.frames_dir,
            chunk_size=int(meta.get("chunk_size", 32)),
            cache_size=int(meta.get("cache_size", 2)),
        )
        
        LOGGER.warning(f"{NAME} | Using image-based search (slower). Consider using embeddings for better performance.")
        processed_frames = 0
        for frame_idx, frame_idx_global in enumerate(frame_indices):
            try:
                frame = frame_manager.get(frame_idx_global)
            except Exception as e:
                LOGGER.warning(f"{NAME} | Failed to load frame {frame_idx_global}: {e}")
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
                frame_results[frame_idx] = results
            except Exception as e:
                LOGGER.warning(
                    f"{NAME} | Embedding Service search failed for frame {frame_idx_global} after retries: {e}"
                )
            
            processed_frames += 1
            # Baseline contract: granular progress (>=10 updates)
            if processed_frames % max(1, n_frames // 10) == 0:
                _emit_progress(
                    rs_path=args.rs_path,
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=processed_frames,
                    total=n_frames,
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

    # Map search results to canonical label space (via UUID)
    # Results from Embedding Service contain "id" (UUID) and "name"
    # We map them to canonical label space using uuid_to_int
    results_by_uuid: Dict[str, Dict[str, Any]] = {}  # uuid -> result (for deduplication)
    for results in frame_results:
        for result in results[: args.topk]:
            result_uuid = result.get("id", "")
            if result_uuid and result_uuid in uuid_to_int:
                # Deduplicate: take best similarity for each UUID
                if result_uuid not in results_by_uuid:
                    results_by_uuid[result_uuid] = result
                else:
                    existing_sim = float(results_by_uuid[result_uuid].get("similarity", 0.0))
                    new_sim = float(result.get("similarity", 0.0))
                    if new_sim > existing_sim:
                        results_by_uuid[result_uuid] = result

    # Use canonical label space (already built above)
    semantic_label_names = canonical_semantic_label_names
    semantic_object_ids = canonical_semantic_object_ids

    # Build output arrays
    unique_track_ids = sorted(tracks.keys())
    n_frames = len(frame_indices)
    n_tracks = len(unique_track_ids)

    # Track-level arrays
    track_ids_arr = np.asarray(unique_track_ids, dtype=np.int32)  # (T,)
    track_topk_ids = np.full((n_tracks, args.topk), -1, dtype=np.int32)  # (T, K) -1 for missing
    track_topk_scores = np.full((n_tracks, args.topk), np.nan, dtype=np.float32)  # (T, K) NaN for missing
    track_present_mask = np.ones((n_tracks,), dtype=np.bool_)  # (T,)
    track_is_confident_top1 = np.zeros((n_tracks,), dtype=np.bool_)  # (T,)
    track_topk_evidence_frame_indices = np.full((n_tracks, args.topk), -1, dtype=np.int32)  # (T, K)

    # Frame-level arrays
    frame_topk_ids = np.full((n_frames, args.topk), -1, dtype=np.int32)  # (N, K)
    frame_topk_scores = np.full((n_frames, args.topk), np.nan, dtype=np.float32)  # (N, K)
    frame_is_confident_top1 = np.zeros((n_frames,), dtype=np.bool_)  # (N,)

    # Fill track-level arrays
    for track_idx, track_id in enumerate(unique_track_ids):
        track_frame_indices = tracks[track_id]
        
        # Aggregate results for this track (by UUID -> label_id)
        # Also track best frame for each place
        track_place_scores: Dict[int, float] = {}  # label_id -> max_similarity
        track_place_best_frames: Dict[int, int] = {}  # label_id -> frame_idx_local with max similarity
        
        for frame_idx_local, frame_idx_global in enumerate(frame_indices):
            if frame_idx_global not in track_frame_indices:
                continue
            
            results = frame_results[frame_idx_local]
            for result in results[: args.topk]:
                result_uuid = result.get("id", "")
                similarity = float(result.get("similarity", 0.0))
                
                if result_uuid and result_uuid in uuid_to_int:
                    label_id = uuid_to_int[result_uuid]
                    # Deduplicate: take best similarity for each place
                    if label_id not in track_place_scores or similarity > track_place_scores[label_id]:
                        track_place_scores[label_id] = similarity
                        track_place_best_frames[label_id] = frame_idx_local
        
        # Sort by similarity and take top-K
        track_results_sorted = [
            (similarity, label_id)
            for label_id, similarity in track_place_scores.items()
        ]
        track_results_sorted.sort(key=lambda x: x[0], reverse=True)
        
        for k, (similarity, label_id) in enumerate(track_results_sorted[: args.topk]):
            track_topk_ids[track_idx, k] = label_id
            track_topk_scores[track_idx, k] = similarity
            # Find evidence frame (union frame index where similarity is maximum for this place in this track)
            if label_id in track_place_best_frames:
                best_frame_idx_local = track_place_best_frames[label_id]
                track_topk_evidence_frame_indices[track_idx, k] = int(frame_indices[best_frame_idx_local])
        
        # Track confidence flag (top-1) - use threshold_global
        if track_results_sorted:
            top1_score = track_results_sorted[0][0]
            track_is_confident_top1[track_idx] = bool(
                np.isfinite(top1_score) and top1_score >= args.threshold_global
            )

    # Fill frame-level arrays
    for frame_idx_local, results in enumerate(frame_results):
        frame_place_scores: Dict[int, float] = {}  # label_id -> max_similarity
        
        for result in results[: args.topk]:
            result_uuid = result.get("id", "")
            similarity = float(result.get("similarity", 0.0))
            
            if result_uuid and result_uuid in uuid_to_int:
                label_id = uuid_to_int[result_uuid]
                # Deduplicate: take best similarity for each place
                if label_id not in frame_place_scores or similarity > frame_place_scores[label_id]:
                    frame_place_scores[label_id] = similarity
        
        # Sort by similarity and take top-K
        frame_results_sorted = [
            (similarity, label_id)
            for label_id, similarity in frame_place_scores.items()
        ]
        frame_results_sorted.sort(key=lambda x: x[0], reverse=True)
        
        for k, (similarity, label_id) in enumerate(frame_results_sorted[: args.topk]):
            frame_topk_ids[frame_idx_local, k] = label_id
            frame_topk_scores[frame_idx_local, k] = similarity
        
        # Frame confidence flag (top-1) - use threshold_global
        if frame_results_sorted:
            top1_score = frame_results_sorted[0][0]
            frame_is_confident_top1[frame_idx_local] = bool(
                np.isfinite(top1_score) and top1_score >= args.threshold_global
            )

    # Build metadata
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(
            f"{NAME} | frames metadata missing required run identity keys: {missing}"
        )
    
    # Count unique places found in results
    places_found_uuids = set()
    for results in frame_results:
        for result in results[: args.topk]:
            result_uuid = result.get("id", "")
            if result_uuid and result_uuid in uuid_to_int:
                places_found_uuids.add(result_uuid)
    
    # Determine status and empty_reason
    threshold_global = float(args.threshold_global)
    status = "ok"
    empty_reason = None
    
    # Check if we have any places found
    if len(places_found_uuids) == 0:
        # No places found - check if it's because service was unavailable
        if not embedding_service_available:
            # Service was unavailable - this should have been caught earlier (fail-fast),
            # but if we got here, it means service became unavailable during processing
            status = "empty"
            empty_reason = "embedding_service_unavailable_during_processing"
        else:
            # Service was available but no places found
            status = "empty"
            empty_reason = "no_places_detected"
    
    output_meta: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": status,
        "empty_reason": empty_reason,
        "embedding_service_url": embedding_client.base_url,
        "place_category": PLACE_CATEGORY,
        "topk": args.topk,
        "similarity_threshold": args.similarity_threshold,
        "threshold_global": threshold_global,
        "min_track_length": args.min_track_length,
        "max_gap_sec": args.max_gap_sec,
        "num_tracks": n_tracks,
        "num_places": len(places_found_uuids),
        "num_frames": n_frames,
        # DB provenance
        "db_name": "embedding_service",
        "db_version": "v1",
        "db_digest": db_digest,
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
    # Add upstream models from core_clip
    if upstream_models_used:
        models_used_list.extend(upstream_models_used)
    output_meta = apply_models_meta(output_meta, models_used=models_used_list)
    
    # Add upstream model signature (for provenance chaining)
    if upstream_model_signature:
        output_meta["core_clip_model_signature"] = upstream_model_signature
    
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

    # Create meta_json (cross-venv safe)
    output_meta_json = json.dumps(output_meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    np.savez_compressed(
        output_path,
        frame_indices=fi_np,
        times_s=times_s,
        semantic_label_names=semantic_label_names,
        semantic_object_ids=semantic_object_ids,
        threshold_per_label_arr=threshold_per_label_arr,
        track_ids=track_ids_arr,
        track_topk_ids=track_topk_ids,
        track_topk_scores=track_topk_scores,
        track_present_mask=track_present_mask,
        track_is_confident_top1=track_is_confident_top1,
        track_topk_evidence_frame_indices=track_topk_evidence_frame_indices,
        frame_topk_ids=frame_topk_ids,
        frame_topk_scores=frame_topk_scores,
        frame_is_confident_top1=frame_is_confident_top1,
        meta=np.asarray(output_meta, dtype=object),
        meta_json=np.asarray(output_meta_json, dtype=object),
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
        f"(tracks={n_tracks}, places={len(places_found_uuids)}, frames={n_frames})"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EmbeddingServiceUnavailableError as ex:
        print(f"{NAME}: {ex}", file=sys.stderr)
        raise SystemExit(1) from None
