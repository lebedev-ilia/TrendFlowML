#!/usr/bin/env python3
"""
core_face_identity (semantic head, Audit v3)

Face identity recognition using Embedding Service:
- Extracts face crops from core_face_landmarks
- Searches for similar faces via Embedding Service
- Returns per-frame top-K face identifications with deterministic label-space
- Schema v2: db_digest, meta_json, face_bbox_xyxy for render assets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
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
from utils.meta_builder import apply_models_meta, model_used  # type: ignore  # noqa: E402
from utils.embedding_service_errors import EmbeddingServiceUnavailableError  # type: ignore  # noqa: E402

# Import Embedding Service client (try utils directory first, then fallback)
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
                raise RuntimeError("face_identity | Failed to load EmbeddingServiceClient from utils")
        else:
            raise ImportError(
                "face_identity | embedding_service_client not found. "
                f"Expected at: {_utils_path}"
            )

NAME = "core_face_identity"
VERSION = "0.2"
SCHEMA_VERSION = "core_face_identity_npz_v2"
ARTIFACT_FILENAME = "face_identity.npz"
LOGGER = get_logger(NAME)

# Face category for Embedding Service
FACE_CATEGORY = "face"
TOP_K = 5  # Contract: fixed K=5 for semantic-head v1


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    return out


def _require_frame_indices_from_landmarks(landmarks_npz: Dict[str, Any]) -> List[int]:
    """
    Extract frame indices from core_face_landmarks, filtering only frames with faces.

    Args:
        landmarks_npz: Loaded NPZ from core_face_landmarks

    Returns:
        List of frame indices where faces were found

    Raises:
        RuntimeError: If frame_indices or face_present are missing or invalid
    """
    frame_indices_all = landmarks_npz.get("frame_indices")
    if frame_indices_all is None:
        raise RuntimeError(
            f"{NAME} | core_face_landmarks.landmarks.npz missing frame_indices (no-fallback)"
        )
    frame_indices_all = np.asarray(frame_indices_all, dtype=np.int32).reshape(-1)
    
    # Get face_present to filter frames with faces
    face_present = landmarks_npz.get("face_present")
    if face_present is None:
        raise RuntimeError(
            f"{NAME} | core_face_landmarks.landmarks.npz missing face_present (no-fallback)"
        )
    face_present = np.asarray(face_present, dtype=bool)
    
    # Filter: keep only frames where at least one face is present
    if face_present.ndim == 1:
        # (N,) - one face per frame
        has_face = face_present
    elif face_present.ndim == 2:
        # (N, FACES) - multiple faces per frame
        has_face = np.any(face_present, axis=1)
    else:
        raise RuntimeError(
            f"{NAME} | core_face_landmarks.face_present has invalid shape: {face_present.shape}"
        )
    
    if len(has_face) != len(frame_indices_all):
        raise RuntimeError(
            f"{NAME} | frame_indices length ({len(frame_indices_all)}) != face_present length ({len(has_face)})"
        )
    
    # Filter frame_indices to only those with faces
    frame_indices_with_faces = frame_indices_all[has_face].tolist()
    
    if not frame_indices_with_faces:
        # Valid empty: no faces found in video
        LOGGER.warning(f"{NAME} | No frames with faces found in core_face_landmarks")
        return []
    
    return [int(x) for x in frame_indices_with_faces]


def _extract_face_bbox_from_landmarks(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract bounding box from face landmarks.

    Args:
        landmarks: Face landmarks array (468, 3) or similar shape

    Returns:
        Bounding box as [x1, y1, x2, y2] or None if invalid
    """
    if landmarks is None or landmarks.size == 0:
        return None
    
    # Filter out NaN values
    valid_mask = ~np.isnan(landmarks[:, 0])
    if not np.any(valid_mask):
        return None
    
    valid_landmarks = landmarks[valid_mask]
    if valid_landmarks.size == 0:
        return None
    
    # Get bounding box from valid landmarks
    x_coords = valid_landmarks[:, 0]
    y_coords = valid_landmarks[:, 1]
    
    x1 = float(np.min(x_coords))
    y1 = float(np.min(y_coords))
    x2 = float(np.max(x_coords))
    y2 = float(np.max(y_coords))
    
    # Add small padding (5%)
    w = x2 - x1
    h = y2 - y1
    pad_x = w * 0.05
    pad_y = h * 0.05
    
    return np.array([
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        x2 + pad_x,
        y2 + pad_y,
    ], dtype=np.float32)


def _crop_face(frame_rgb: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop face from frame using bounding box.

    Args:
        frame_rgb: Frame as RGB uint8 array (H, W, 3)
        bbox: Bounding box as [x1, y1, x2, y2]

    Returns:
        Cropped face image or None if invalid
    """
    if frame_rgb is None or bbox is None:
        return None
    
    h, w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    
    # Clamp to frame bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = frame_rgb[y1:y2, x1:x2, :].copy()
    if crop.size == 0:
        return None
    
    return crop


def main() -> int:
    ap = argparse.ArgumentParser(
        "core_face_identity",
        description="Face identity recognition component using Embedding Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --frames-dir frames/ --rs-path results/ --embedding-service-url http://localhost:8001

  # With custom top-K and threshold
  python main.py --frames-dir frames/ --rs-path results/ --topk 10 --similarity-threshold 0.7
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
        help=f"Number of top results to return per frame (default: {TOP_K})"
    )
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity threshold for results (default: 0.0, range: 0.0-1.0)"
    )
    ap.add_argument(
        "--http-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout (seconds) for Embedding Service /search and related calls (default: 120)",
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

    # Baseline contract: emit load_deps stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )
    
    # Load core_face_landmarks first (no-fallback) - this is the source of truth for frame indices
    landmarks_path = os.path.join(
        str(args.rs_path), "core_face_landmarks", "landmarks.npz"
    )
    landmarks_npz = _load_npz(landmarks_path)
    
    # Extract frame indices ONLY from frames where faces were found
    frame_indices = _require_frame_indices_from_landmarks(landmarks_npz)
    
    t_load_deps_end = time.perf_counter()
    timings["load_deps"] = t_load_deps_end - t_load_deps
    
    # Handle valid empty: no faces found
    if not frame_indices:
        # Write valid empty artifact
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not meta.get(k)]
        if missing:
            raise RuntimeError(
                f"{NAME} | frames metadata missing required run identity keys: {missing}"
            )
        
        # Initialize Embedding Service client for db_digest (fail-fast → EmbeddingServiceUnavailableError)
        embedding_client = EmbeddingServiceClient(
            base_url=args.embedding_service_url, timeout=args.http_timeout
        )
        embedding_client._ensure_url()
        
        labels = embedding_client.get_labels(category=FACE_CATEGORY)
        if not labels:
            raise RuntimeError(
                f"{NAME} | Embedding Service category '{FACE_CATEGORY}' has 0 labels (fail-fast)"
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
                f"{NAME} | Embedding Service returned invalid labels for '{FACE_CATEGORY}' (no ids)"
            )
        labels_canon.sort(key=lambda r: r["id"])
        db_digest = _sha256_hex(
            json.dumps(labels_canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        semantic_object_ids = np.asarray([r["id"] for r in labels_canon], dtype="U")
        semantic_label_names = np.asarray(
            [f"{i}:{labels_canon[i]['name']}" for i in range(len(labels_canon))],
            dtype="U",
        )
        embedding_models = sorted({r["embedding_model"] for r in labels_canon if r["embedding_model"]})
        embedding_model = embedding_models[0] if len(embedding_models) == 1 else ""
        
        meta_out: Dict[str, Any] = {
            "producer": NAME,
            "producer_version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "empty",
            "empty_reason": "no_faces_in_video",
        }
        for k in required_run_keys:
            meta_out[k] = meta.get(k)
        meta_out["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")
        meta_out["embedding_service_url"] = embedding_client.base_url
        meta_out["category"] = FACE_CATEGORY
        meta_out["top_k"] = args.topk
        meta_out["similarity_threshold"] = args.similarity_threshold
        meta_out["n_frames"] = 0
        meta_out["total_faces_processed"] = 0
        meta_out["db_name"] = "embedding_service"
        meta_out["db_version"] = "v1"
        meta_out["db_digest"] = db_digest
        if embedding_model:
            meta_out["embedding_model"] = embedding_model
        meta_out = apply_models_meta(meta_out, models_used=[])
        
        out_dir = os.path.join(str(args.rs_path), NAME)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, ARTIFACT_FILENAME)
        
        # Baseline contract: emit save stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="save",
        )
        
        t_save = time.perf_counter()
        timings["saving"] = 0.0  # Will be updated after save
        timings["total"] = time.perf_counter() - t0
        
        # Baseline contract: stage_timings_ms in meta
        stage_timings_ms: Dict[str, float] = {}
        for key, value in timings.items():
            stage_timings_ms[key] = float(value) * 1000.0
        meta_out["stage_timings_ms"] = stage_timings_ms
        
        def _build_npz_payload_empty(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "frame_indices": np.array([], dtype=np.int32),
                "times_s": np.array([], dtype=np.float32),
                "semantic_label_names": semantic_label_names,
                "semantic_object_ids": semantic_object_ids,
                "face_ids": np.array([], dtype=np.int32).reshape(0, args.topk),
                "face_names": np.array([], dtype="U256").reshape(0, args.topk),
                "face_similarities": np.array([], dtype=np.float32).reshape(0, args.topk),
                "face_bbox_xyxy": np.array([], dtype=np.float32).reshape(0, 4),
                "meta": np.asarray(meta_dict, dtype=object),
                "meta_json": np.asarray(
                    json.dumps(meta_dict, ensure_ascii=False, sort_keys=True),
                    dtype="U",
                ),
            }
        
        _atomic_save_npz(out_path, **_build_npz_payload_empty(meta_out))
        timings["saving"] = time.perf_counter() - t_save
        timings["total"] = time.perf_counter() - t0
        meta_out["stage_timings_ms"] = {
            k: float(v) * 1000.0 for k, v in timings.items()
        }
        _atomic_save_npz(out_path, **_build_npz_payload_empty(meta_out))
        
        # Baseline contract: emit done stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="done",
        )
        
        LOGGER.info(f"{NAME} | No faces found, wrote empty artifact: {out_path}")
        return 0

    # Timestamps (contract: use union_timestamps_sec)
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
    
    # Get all frame_indices from landmarks for mapping
    landmarks_fi_all = np.asarray(landmarks_npz.get("frame_indices"), dtype=np.int32).reshape(-1)

    # Extract face landmarks and presence flags
    face_landmarks_all = landmarks_npz.get("face_landmarks")  # (N_all, FACES, 468, 3)
    face_present_all = landmarks_npz.get("face_present")  # (N_all, FACES) bool
    
    if face_landmarks_all is None:
        raise RuntimeError(
            f"{NAME} | core_face_landmarks.landmarks.npz missing face_landmarks (no-fallback)"
        )
    if face_present_all is None:
        raise RuntimeError(
            f"{NAME} | core_face_landmarks.landmarks.npz missing face_present (no-fallback)"
        )
    
    face_landmarks_all = np.asarray(face_landmarks_all, dtype=np.float32)
    face_present_all = np.asarray(face_present_all, dtype=bool)
    
    # Create mapping from global frame index to position in landmarks arrays
    landmarks_fi_map: Dict[int, int] = {int(fi): int(i) for i, fi in enumerate(landmarks_fi_all)}
    
    # Filter landmarks arrays to only frames with faces
    n_frames = len(frame_indices)
    landmark_positions = [landmarks_fi_map[fi] for fi in frame_indices]
    
    face_landmarks = face_landmarks_all[landmark_positions]  # (N, FACES, 468, 3)
    face_present = face_present_all[landmark_positions]  # (N, FACES) bool
    
    max_faces = face_landmarks.shape[1] if len(face_landmarks.shape) > 1 else 1

    # Baseline contract: emit process_frames stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="process_frames",
    )
    
    t_process_start = time.perf_counter()

    # Initialize Embedding Service client
    embedding_client = EmbeddingServiceClient(
        base_url=args.embedding_service_url, timeout=args.http_timeout
    )
    
    # Fail-fast: проверка доступности Embedding Service
    embedding_client._ensure_url()

    # Load label-space (db provenance + deterministic UUID->int32 mapping)
    labels = embedding_client.get_labels(category=FACE_CATEGORY)
    if not labels:
        raise RuntimeError(
            f"{NAME} | Embedding Service category '{FACE_CATEGORY}' has 0 labels (fail-fast)"
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
            f"{NAME} | Embedding Service returned invalid labels for '{FACE_CATEGORY}' (no ids)"
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

    embedding_models = sorted({r["embedding_model"] for r in labels_canon if r["embedding_model"]})
    embedding_model = embedding_models[0] if len(embedding_models) == 1 else ""

    # Create FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=int(meta.get("chunk_size", 32)),
        cache_size=int(meta.get("cache_size", 2)),
    )

    # Prepare output arrays
    # Per-frame top-K results: (N, K)
    face_ids = np.full((n_frames, args.topk), -1, dtype=np.int32)
    face_names = np.full((n_frames, args.topk), "", dtype="U256")
    face_similarities = np.full((n_frames, args.topk), 0.0, dtype=np.float32)
    # Bbox for top-1 face per frame (for render assets)
    face_bbox_xyxy = np.full((n_frames, 4), np.nan, dtype=np.float32)

    # Early validation: проверка доступности Embedding Service и тестовый запрос
    embedding_service_available = True
    try:
        # Проверка health endpoint
        embedding_client._ensure_url()
        
        # Тестовый запрос с первым кадром, где есть лицо, для проверки работоспособности search endpoint
        test_frame_idx = None
        test_face_idx = None
        for frame_idx in range(n_frames):
            for face_idx in range(max_faces):
                if face_idx < face_present.shape[1] and face_present[frame_idx, face_idx]:
                    test_frame_idx = frame_idx
                    test_face_idx = face_idx
                    break
            if test_frame_idx is not None:
                break
        
        if test_frame_idx is not None and test_face_idx is not None:
            try:
                frame_idx_global = frame_indices[test_frame_idx]
                test_frame = frame_manager.get(frame_idx_global)
                landmarks = face_landmarks[test_frame_idx, test_face_idx]
                bbox = _extract_face_bbox_from_landmarks(landmarks)
                if bbox is not None:
                    face_crop = _crop_face(test_frame, bbox)
                    if face_crop is not None:
                        test_results = embedding_client.search(
                            category=FACE_CATEGORY,
                            image=face_crop,
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
                    f"Check Embedding Service status and ensure category '{FACE_CATEGORY}' is configured."
                )
    except Exception as health_error:
        # Health check не прошел
        embedding_service_available = False
        LOGGER.warning(
            f"{NAME} | Embedding Service health check failed: {health_error}. "
            f"Skipping all frames. Check Embedding Service status."
        )

    # Process each frame (или пропустить, если сервис недоступен)
    total_faces_processed = 0
    failed_faces = 0
    
    if not embedding_service_available:
        # Сервис недоступен - пропускаем обработку всех кадров
        LOGGER.warning(
            f"{NAME} | Embedding Service unavailable, skipping {n_frames} frames"
        )
    else:
        # Сервис доступен - обрабатываем все кадры
        for frame_idx in range(n_frames):
            frame_idx_global = frame_indices[frame_idx]
            
            # Get frame
            try:
                frame = frame_manager.get(frame_idx_global)
            except Exception as e:
                LOGGER.warning(
                    f"{NAME} | Failed to load frame {frame_idx_global}: {e}"
                )
                continue
            
            # Process all faces in this frame
            frame_results: List[Tuple[float, int, str, Optional[np.ndarray]]] = []  # (similarity, id, name, bbox)
            best_bbox: Optional[np.ndarray] = None
            
            for face_idx in range(max_faces):
                if face_idx >= face_present.shape[1] or not face_present[frame_idx, face_idx]:
                    continue
                
                # Extract face landmarks
                landmarks = face_landmarks[frame_idx, face_idx]  # (468, 3)
                
                # Get bounding box from landmarks
                bbox = _extract_face_bbox_from_landmarks(landmarks)
                if bbox is None:
                    continue
                
                # Crop face
                face_crop = _crop_face(frame, bbox)
                if face_crop is None:
                    continue
                
                # Search in Embedding Service
                try:
                    search_results = embedding_client.search(
                        category=FACE_CATEGORY,
                        image=face_crop,
                        top_k=args.topk,
                        similarity_threshold=args.similarity_threshold,
                    )
                    
                    for result in search_results:
                        face_uuid = str(result.get("id", ""))
                        face_name = str(result.get("name", ""))
                        similarity = float(result.get("similarity", 0.0))
                        
                        # Map UUID to int32 index (deterministic label-space)
                        face_id_int = uuid_to_int.get(face_uuid, -1)
                        if face_id_int < 0:
                            # UUID not in label-space - это ошибка консистентности базы
                            raise RuntimeError(
                                f"{NAME} | Face UUID {face_uuid} not found in label-space. "
                                f"This indicates database inconsistency: label-space was loaded with {len(labels_canon)} labels, "
                                f"but search returned UUID not in that set. Database may have changed between label-space load and search."
                            )
                        
                        frame_results.append((similarity, face_id_int, face_name, bbox.copy()))
                        total_faces_processed += 1
                        
                except Exception as e:
                    failed_faces += 1
                    LOGGER.warning(
                        f"{NAME} | Embedding Service search failed for frame {frame_idx_global}, face {face_idx}: {e}"
                    )
                    continue
            
            # Sort by similarity and take top-K (deduplicate by name)
            frame_results.sort(key=lambda x: x[0], reverse=True)
            
            # Deduplicate by name (keep best similarity for each name)
            seen_names: Dict[str, Tuple[float, int, str, Optional[np.ndarray]]] = {}
            for similarity, face_id, face_name, bbox in frame_results:
                if face_name and face_name not in seen_names:
                    seen_names[face_name] = (similarity, face_id, face_name, bbox)
                elif face_name and similarity > seen_names[face_name][0]:
                    seen_names[face_name] = (similarity, face_id, face_name, bbox)
            
            # Fill output arrays (NaN-policy: -1 for ids, 0.0 for scores where no result)
            deduplicated_results = sorted(seen_names.values(), key=lambda x: x[0], reverse=True)
            for k, (similarity, face_id_int, face_name, bbox) in enumerate(deduplicated_results[: args.topk]):
                face_ids[frame_idx, k] = face_id_int
                face_names[frame_idx, k] = face_name
                face_similarities[frame_idx, k] = similarity
                # Save bbox for top-1 (for render assets)
                if k == 0 and bbox is not None:
                    face_bbox_xyxy[frame_idx, :] = bbox
                    best_bbox = bbox
            
            # Baseline contract: granular progress (>=10 updates)
            processed = frame_idx + 1
            if processed % max(1, (n_frames // 15)) == 0 or processed == n_frames:
                _emit_progress(
                    rs_path=args.rs_path,
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=processed,
                    total=n_frames,
                    stage="process_frames",
                )

    frame_manager.close()
    
    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start
    
    # Fail-fast: если все лица упали с ошибками
    total_faces_attempted = total_faces_processed + failed_faces
    if total_faces_attempted > 0 and failed_faces == total_faces_attempted and total_faces_processed == 0:
        raise RuntimeError(
            f"{NAME} | All {total_faces_attempted} faces failed with Embedding Service errors. "
            "Service may be misconfigured or unavailable."
        )

    # Build metadata (contract: all required fields)
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(
            f"{NAME} | frames metadata missing required run identity keys: {missing}"
        )

    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ok" if total_faces_processed > 0 else "empty",
        "empty_reason": None if total_faces_processed > 0 else "no_faces_in_video",
    }
    
    # Required run identity fields
    for k in required_run_keys:
        meta_out[k] = meta.get(k)
    
    # Required by contract (baseline may use "unknown")
    meta_out["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")
    
    # Component-specific metadata
    meta_out["embedding_service_url"] = embedding_client.base_url
    meta_out["category"] = FACE_CATEGORY
    meta_out["top_k"] = args.topk
    meta_out["similarity_threshold"] = args.similarity_threshold
    meta_out["total_faces_processed"] = total_faces_processed
    meta_out["n_frames"] = n_frames
    
    # DB provenance (reproducibility)
    meta_out["db_name"] = "embedding_service"
    meta_out["db_version"] = "v1"
    meta_out["db_digest"] = db_digest
    if embedding_model:
        meta_out["embedding_model"] = embedding_model

    # Add models_used (contract: required if models are used)
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
    meta_out = apply_models_meta(meta_out, models_used=models_used_list)

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
    out_dir = os.path.join(str(args.rs_path), NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ARTIFACT_FILENAME)
    
    timings["saving"] = 0.0  # Will be updated after save
    timings["total"] = time.perf_counter() - t0
    
    def _build_npz_payload(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_indices": fi_np,
            "times_s": times_s,
            # Label space (deterministic, derived from Embedding Service)
            "semantic_label_names": semantic_label_names,
            "semantic_object_ids": semantic_object_ids,
            # Per-frame top-K results
            "face_ids": face_ids,  # (N, K) int32, -1 where no result
            "face_names": face_names,  # (N, K) str, "" where no result
            "face_similarities": face_similarities,  # (N, K) float32, 0.0 where no result
            # Bbox for top-1 face per frame (for render assets)
            "face_bbox_xyxy": face_bbox_xyxy,  # (N, 4) float32, NaN where no face
            # Meta
            "meta": np.asarray(meta_dict, dtype=object),
            "meta_json": np.asarray(
                json.dumps(meta_dict, ensure_ascii=False, sort_keys=True),
                dtype="U",
            ),
        }

    # Two-pass write: measure saving time and persist final meta.stage_timings_ms
    t_save_start = time.perf_counter()
    _atomic_save_npz(out_path, **_build_npz_payload(meta_out))
    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0
    meta_out["stage_timings_ms"] = {
        k: float(v) * 1000.0 for k, v in timings.items()
    }
    _atomic_save_npz(out_path, **_build_npz_payload(meta_out))

    # Baseline contract: emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )

    LOGGER.info(
        f"{NAME} | Saved results: {out_path} "
        f"(frames={n_frames}, faces_processed={total_faces_processed})"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EmbeddingServiceUnavailableError as ex:
        print(f"{NAME}: {ex}", file=sys.stderr)
        raise SystemExit(1) from None
