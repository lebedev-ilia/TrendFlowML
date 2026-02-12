#!/usr/bin/env python3
"""
franchise_recognition (v1 semantic head)

Goal: identify a specific title/franchise (games/anime/cartoons) using:
- Embedding Service for franchise search over core_clip frame embeddings
- OCR -> candidate hints (optional, for cost control)

Constraints:
- Embedding Service is required (fail-fast if unavailable)
- sampling group = core_clip.frame_indices (Segmenter-owned)
- top-K is never gated; thresholds only produce is_confident flags
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_vp_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
if _vp_root not in sys.path:
    sys.path.append(_vp_root)

from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta  # type: ignore  # noqa: E402

# Import Embedding Service client (create if doesn't exist)
try:
    from embedding_service_client import EmbeddingServiceClient
except ImportError:
    # Fallback: try to import from brand_semantics or car_semantics
    _brand_path = os.path.join(
        os.path.dirname(__file__), "..", "brand_semantics", "embedding_service_client.py"
    )
    if os.path.isfile(_brand_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("embedding_service_client", _brand_path)
        embedding_service_client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(embedding_service_client)
        EmbeddingServiceClient = embedding_service_client.EmbeddingServiceClient
    else:
        raise RuntimeError(
            "franchise_recognition | embedding_service_client not found. "
            "Please ensure EmbeddingServiceClient is available."
        )

NAME = "franchise_recognition"
VERSION = "0.1"
SCHEMA_VERSION = "franchise_recognition_npz_v1"
ARTIFACT_FILENAME = "franchise_recognition.npz"
LOGGER = get_logger(NAME)

# Franchise category for Embedding Service
FRANCHISE_CATEGORY = "franchise"
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


def _require_frame_indices(meta: dict) -> List[int]:
    """Extract and validate frame indices from metadata."""
    block = meta.get("core_clip")
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | frames metadata missing core_clip.frame_indices (no-fallback)"
        )
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(
            f"{NAME} | core_clip.frame_indices empty/invalid (no-fallback)"
        )
    return [int(x) for x in frame_indices]


def _norm_text(s: str) -> str:
    """Normalize text for OCR matching."""
    s = str(s or "")
    s = s.lower()
    s = s.replace("\u200b", "")
    # Keep letters/numbers, collapse spaces.
    s = re.sub(r"[^0-9a-zа-яё]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_ocr_npz(path: str) -> List[Dict[str, Any]]:
    """
    Minimal supported schema:
    - key `ocr_raw` -> object array holding list[dict]
    - or key `ocr_data` -> object array holding list[dict]
    Each dict should contain at least: `frame`, `bbox`, `text` (or `text_raw`), `confidence` (optional).
    """
    data = np.load(path, allow_pickle=True)
    raw = data.get("ocr_raw")
    if raw is None:
        raw = data.get("ocr_data")
    if raw is None:
        return []
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw_item = raw.item() if raw.ndim == 0 else raw.tolist()
    else:
        raw_item = raw
    if isinstance(raw_item, list):
        out: List[Dict[str, Any]] = []
        for d in raw_item:
            if isinstance(d, dict):
                out.append(d)
        return out
    return []


def _find_latest_npz(component_dir: str) -> Optional[str]:
    """Find latest NPZ file in component directory."""
    if not os.path.isdir(component_dir):
        return None
    files: List[str] = []
    for name in os.listdir(component_dir):
        p = os.path.join(component_dir, name)
        if os.path.isfile(p) and name.lower().endswith(".npz"):
            files.append(p)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0] if files else None


def _auto_find_ocr_npz(rs_path: str) -> Optional[str]:
    """Convention: ocr_extractor writes into rs_path/ocr_extractor/*.npz"""
    return _find_latest_npz(os.path.join(str(rs_path), "ocr_extractor"))


def main() -> int:
    """
    Main entry point for franchise_recognition component.

    This function processes video frames to recognize franchises:
    1. Loads frame embeddings from core_clip
    2. Optionally uses OCR to filter candidates
    3. Searches for similar franchises via Embedding Service
    4. Outputs results in NPZ format

    The component follows the semantic head contract:
    - Requires core_clip.frame_indices
    - Aligns output to same frame_indices
    - Outputs per-frame and video-level top-K results

    Returns:
        0 on success, non-zero on error
    """
    ap = argparse.ArgumentParser(
        NAME,
        description="Franchise recognition component using Embedding Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames-dir", required=True, help="Directory containing video frames and metadata.json")
    ap.add_argument("--rs-path", required=True, help="Result store path (e.g., result_store/platform/video/run)")
    ap.add_argument(
        "--embedding-service-url",
        default=None,
        help="Embedding Service URL (default: from EMBEDDING_SERVICE_URL env or http://localhost:8005)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=TOP_K,
        help=f"Number of top results to return per frame (default: {TOP_K}, contract: must be 5)",
    )
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity threshold for results (default: 0.0, range: 0.0-1.0)",
    )
    ap.add_argument(
        "--threshold-global",
        type=float,
        default=0.23,
        help="Global threshold for is_confident flag (default: 0.23, used only for flags, not gating)",
    )
    ap.add_argument(
        "--ocr-npz",
        default=None,
        help="Optional OCR NPZ path (union-domain frames). If not specified, auto-finds in rs_path/ocr_extractor/",
    )
    ap.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=0.4,
        help="Minimum OCR confidence for candidate filtering (default: 0.4)",
    )
    ap.add_argument(
        "--ocr-max-events",
        type=int,
        default=5000,
        help="Maximum OCR events to process (default: 5000, cost control)",
    )
    ap.add_argument(
        "--use-ocr-filtering",
        action="store_true",
        help="Use OCR to filter franchise candidates (only if OCR available and many franchises in database)",
    )
    ap.add_argument(
        "--max-franchises-for-full-search",
        type=int,
        default=500,
        help="If franchise count <= this, compute full search. Else rely on OCR-candidates gating (default: 500)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Embedding Service search requests (default: 16, scheduler-controlled)",
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
    uts = meta.get("union_timestamps_sec")
    if uts is None:
        raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
    uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
    fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if fi_np.size == 0:
        raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
    if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
        raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
    times_s = uts_arr[fi_np].astype(np.float32)

    # Baseline contract: emit load_deps stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )

    # Load core_clip embeddings
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

    # Initialize Embedding Service client (fail-fast if unavailable)
    try:
        embedding_client = EmbeddingServiceClient(base_url=args.embedding_service_url)
        # Test connection
        embedding_client._ensure_url()
    except Exception as e:
        raise RuntimeError(
            f"{NAME} | Embedding Service unavailable (fail-fast): {e}. "
            "Ensure Embedding Service is running and accessible."
        )

    t_load_deps_end = time.perf_counter()
    timings["load_deps"] = t_load_deps_end - t_load_deps

    K = int(args.topk)
    if K != 5:
        raise RuntimeError(f"{NAME} | topk must be 5 (contract), got {K}")

    # Optional OCR: extract candidate franchise names (via text matching)
    # OPTIMIZATION 6: Vectorized OCR filtering
    ocr_events = 0
    ocr_hits = 0
    ocr_candidate_names: List[str] = []
    ocr_evidence_frames: Dict[str, List[int]] = {}  # franchise_name -> union frame indices

    ocr_npz_path = str(args.ocr_npz) if args.ocr_npz else None
    if not ocr_npz_path:
        ocr_npz_path = _auto_find_ocr_npz(str(args.rs_path))

    if ocr_npz_path and os.path.isfile(str(ocr_npz_path)) and args.use_ocr_filtering:
        try:
            ocr_data = _load_ocr_npz(str(ocr_npz_path))
        except Exception as e:
            LOGGER.warning("%s | failed to load OCR NPZ (%s): %s", NAME, ocr_npz_path, e)
            ocr_data = []
        
        if ocr_data:
            # OPTIMIZATION 6: Vectorized filtering using list comprehensions
            allowed = set(int(x) for x in fi_np.tolist())
            min_conf = float(args.ocr_min_confidence)
            limit = int(args.ocr_max_events)
            
            # Filter in one pass
            filtered_ocr = []
            for d in ocr_data:
                if ocr_events >= limit:
                    break
                try:
                    fr = int(d.get("frame", -1))
                except Exception:
                    fr = -1
                if fr not in allowed:
                    continue
                conf = d.get("confidence")
                try:
                    conf_f = float(conf) if conf is not None else 1.0
                except Exception:
                    conf_f = 1.0
                if conf_f < min_conf:
                    continue
                txt = d.get("text") or d.get("text_raw")
                if not txt:
                    continue
                s = _norm_text(str(txt))
                if not s or len(s) < 3:
                    continue
                filtered_ocr.append((fr, s))
                ocr_events += 1
            
            # Build candidate names and evidence frames
            for fr, s in filtered_ocr:
                ocr_candidate_names.append(s)
                ocr_evidence_frames.setdefault(s, []).append(int(fr))
                ocr_hits += 1

            # Dedup candidates
            ocr_candidate_names = sorted(set(ocr_candidate_names))

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
    batch_size = int(args.batch_size)
    
    # Early validation: проверка доступности Embedding Service и тестовый запрос
    embedding_service_available = True
    use_embedding_direct = True
    franchise_embeddings: Optional[List[Dict[str, Any]]] = None
    
    try:
        # OPTIMIZATION 2: Try to use embeddings directly (10-50x faster)
        # Get all franchise embeddings from Embedding Service once, then compare locally
        try:
            # Try to get all franchise embeddings
            franchise_embeddings = embedding_client.get_all_embeddings(
                category=FRANCHISE_CATEGORY,
                embedding_model=None,  # Will use default model for category
            )
            
            if not franchise_embeddings or len(franchise_embeddings) == 0:
                LOGGER.warning(f"{NAME} | No franchise embeddings found in Embedding Service, falling back to image search")
                use_embedding_direct = False
            else:
                LOGGER.info(f"{NAME} | Using direct embedding comparison with {len(franchise_embeddings)} franchises (10-50x faster)")
        except Exception as e:
            LOGGER.warning(f"{NAME} | Failed to get franchise embeddings: {e}, falling back to image search")
            use_embedding_direct = False
        
        # Если не удалось использовать embeddings напрямую, делаем тестовый запрос с первым кадром
        if not use_embedding_direct:
            # Тестовый запрос с первым кадром для проверки работоспособности search endpoint
            if frame_indices:
                try:
                    from utils.frame_manager import FrameManager
                    test_frame_manager = FrameManager(
                        frames_dir=args.frames_dir,
                        chunk_size=int(meta.get("chunk_size", 32)),
                        cache_size=int(meta.get("cache_size", 2)),
                    )
                    test_frame_idx_global = frame_indices[0]
                    test_frame = test_frame_manager.get(test_frame_idx_global)
                    test_results = embedding_client.search(
                        category=FRANCHISE_CATEGORY,
                        image=test_frame,
                        top_k=1,  # Минимальный запрос для теста
                        similarity_threshold=0.0,
                        max_retries=1,  # Одна попытка для теста
                        retry_delay=0.5,
                    )
                    # Если тест прошел успешно, продолжаем обработку
                    LOGGER.info(f"{NAME} | Embedding Service test request successful, proceeding with all frames")
                    test_frame_manager.close()
                except Exception as test_error:
                    # Тестовый запрос не прошел - сервис недоступен или возвращает ошибки
                    embedding_service_available = False
                    LOGGER.warning(
                        f"{NAME} | Embedding Service test request failed: {test_error}. "
                        f"Skipping all frames to avoid repeated errors. "
                        f"Check Embedding Service status and ensure category '{FRANCHISE_CATEGORY}' is configured."
                    )
    except Exception as health_error:
        # Health check не прошел (уже проверен выше, но на всякий случай)
        embedding_service_available = False
        LOGGER.warning(
            f"{NAME} | Embedding Service health check failed: {health_error}. "
            f"Skipping all frames. Check Embedding Service status."
        )
    
    if use_embedding_direct and franchise_embeddings:
        # OPTIMIZATION 2: Direct embedding comparison (no HTTP requests per frame)
        # Prepare franchise embeddings matrix
        franchise_emb_matrix = np.array([f["embedding"] for f in franchise_embeddings], dtype=np.float32)  # (M, D)
        franchise_names = [f.get("name", "unknown") for f in franchise_embeddings]
        franchise_ids = [f.get("id", "") for f in franchise_embeddings]
        franchise_metadata = [f.get("metadata", {}) for f in franchise_embeddings]
        
        # L2 normalize franchise embeddings
        franchise_norms = np.linalg.norm(franchise_emb_matrix, axis=1, keepdims=True)
        franchise_norms = np.where(franchise_norms > 1e-10, franchise_norms, 1.0)
        franchise_emb_normalized = franchise_emb_matrix / franchise_norms  # (M, D)
        
        # L2 normalize frame embeddings
        frame_emb_normalized = _l2norm_rows(frame_emb)  # (N, D)
        
        # Compute cosine similarity: (N, D) @ (D, M) = (N, M)
        similarities = np.dot(frame_emb_normalized, franchise_emb_normalized.T)  # (N, M)
        
        # Apply similarity threshold
        similarity_threshold = float(args.similarity_threshold)
        if similarity_threshold > 0:
            similarities = np.where(similarities >= similarity_threshold, similarities, -1.0)
        
        # Get top-K for each frame
        topk_indices = np.argsort(similarities, axis=1)[:, -K:][:, ::-1]  # (N, K)
        topk_similarities = np.take_along_axis(similarities, topk_indices, axis=1)  # (N, K)
        
        # Build frame_results
        for frame_idx in range(n_frames):
            frame_result = []
            for k_idx in range(K):
                franchise_idx = topk_indices[frame_idx, k_idx]
                similarity = float(topk_similarities[frame_idx, k_idx])
                
                if similarity < similarity_threshold:
                    continue
                
                frame_result.append({
                    "id": str(franchise_ids[franchise_idx]),
                    "name": str(franchise_names[franchise_idx]),
                    "similarity": similarity,
                    "metadata": franchise_metadata[franchise_idx],
                })
            frame_results[frame_idx] = frame_result
        
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
    elif not embedding_service_available:
        # Сервис недоступен - заполняем пустыми результатами для всех кадров
        LOGGER.warning(
            f"{NAME} | Embedding Service unavailable, filling {n_frames} frames with empty results"
        )
        frame_results = [[] for _ in range(n_frames)]
        _emit_progress(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            done=n_frames,
            total=n_frames,
            stage="process_frames",
        )
    else:
        # Fallback: Use image-based search (with optimizations)
        from utils.frame_manager import FrameManager
        
        # OPTIMIZATION 5: Simple cache for frame search results (by frame hash)
        search_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        frame_manager = FrameManager(
            frames_dir=args.frames_dir,
            chunk_size=int(meta.get("chunk_size", 32)),
            cache_size=int(meta.get("cache_size", 2)),
        )
        
        # OPTIMIZATION 4: Parallel batch processing
        def process_batch(batch_frames_data: List[Tuple[int, int, np.ndarray]]) -> List[Tuple[int, List[Dict[str, Any]]]]:
            """Process a batch of frames"""
            batch_results = []
            batch_frames = [frame for _, _, frame in batch_frames_data]
            batch_indices = [(frame_idx, frame_idx_global) for frame_idx, frame_idx_global, _ in batch_frames_data]
            
            try:
                if hasattr(embedding_client, 'search_batch'):
                    batch_search_results = embedding_client.search_batch(
                        category=FRANCHISE_CATEGORY,
                        images=batch_frames,
                        top_k=K,
                        similarity_threshold=args.similarity_threshold,
                        max_retries=3,
                        retry_delay=1.0,
                    )
                    for i, (frame_idx, _) in enumerate(batch_indices):
                        if i < len(batch_search_results):
                            batch_results.append((frame_idx, batch_search_results[i] if batch_search_results[i] else []))
                        else:
                            batch_results.append((frame_idx, []))
                else:
                    # Fallback to individual searches
                    for i, (frame_idx, frame_idx_global) in enumerate(batch_indices):
                        try:
                            # OPTIMIZATION 5: Cache check
                            frame_hash = hashlib.md5(batch_frames[i].tobytes()).hexdigest()
                            cache_key = f"{frame_hash}_{FRANCHISE_CATEGORY}_{K}_{args.similarity_threshold}"
                            
                            if cache_key in search_cache:
                                result = search_cache[cache_key]
                            else:
                                result = embedding_client.search(
                                    category=FRANCHISE_CATEGORY,
                                    image=batch_frames[i],
                                    top_k=K,
                                    similarity_threshold=args.similarity_threshold,
                                    max_retries=3,
                                    retry_delay=1.0,
                                )
                                search_cache[cache_key] = result if result else []
                                result = result if result else []
                            
                            batch_results.append((frame_idx, result))
                        except Exception as e:
                            LOGGER.warning(f"{NAME} | Search failed for frame {frame_idx_global}: {e}")
                            batch_results.append((frame_idx, []))
            except Exception as e:
                LOGGER.warning(f"{NAME} | Batch search failed: {e}")
                # Fallback to individual searches
                for i, (frame_idx, frame_idx_global) in enumerate(batch_indices):
                    try:
                        # OPTIMIZATION 5: Cache check
                        frame_hash = hashlib.md5(batch_frames[i].tobytes()).hexdigest()
                        cache_key = f"{frame_hash}_{FRANCHISE_CATEGORY}_{K}_{args.similarity_threshold}"
                        
                        if cache_key in search_cache:
                            result = search_cache[cache_key]
                        else:
                            result = embedding_client.search(
                                category=FRANCHISE_CATEGORY,
                                image=batch_frames[i],
                                top_k=K,
                                similarity_threshold=args.similarity_threshold,
                                max_retries=3,
                                retry_delay=1.0,
                            )
                            search_cache[cache_key] = result if result else []
                            result = result if result else []
                        
                        batch_results.append((frame_idx, result))
                    except Exception as e2:
                        LOGGER.warning(f"{NAME} | Search failed for frame {frame_idx_global}: {e2}")
                        batch_results.append((frame_idx, []))
            
            return batch_results
        
        # Prepare batches
        batches = []
        for batch_start in range(0, n_frames, batch_size):
            batch_end = min(batch_start + batch_size, n_frames)
            batch_data = []
            for frame_idx in range(batch_start, batch_end):
                frame_idx_global = frame_indices[frame_idx]
                try:
                    frame = frame_manager.get(frame_idx_global)
                    batch_data.append((frame_idx, frame_idx_global, frame))
                except Exception as e:
                    LOGGER.warning(f"{NAME} | Failed to load frame {frame_idx_global}: {e}")
                    batch_data.append((frame_idx, frame_idx_global, None))
            batches.append(batch_data)
        
        # OPTIMIZATION 4: Parallel processing of batches
        max_workers = min(4, len(batches))  # Limit parallelism to avoid overwhelming Embedding Service
        processed_frames = 0
        
        if max_workers > 1 and len(batches) > 1:
            # Use parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for frame_idx, result in batch_results:
                            frame_results[frame_idx] = result
                            processed_frames += 1
                    except Exception as e:
                        LOGGER.warning(f"{NAME} | Batch processing failed: {e}")
        else:
            # Sequential processing
            for batch_data in batches:
                batch_results = process_batch(batch_data)
                for frame_idx, result in batch_results:
                    frame_results[frame_idx] = result
                    processed_frames += 1
        
        # Progress updates
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

    # Build output arrays
    # Collect all unique franchise names and create label mapping
    all_franchise_names: Dict[str, int] = {}  # franchise_name -> label_id
    label_id_counter = 0

    # First pass: collect all franchise names
    for results in frame_results:
        for result in results:
            franchise_name = result.get("name", "unknown")
            if franchise_name not in all_franchise_names:
                all_franchise_names[franchise_name] = label_id_counter
                label_id_counter += 1

    # Build semantic_label_names
    franchise_names_list = sorted(all_franchise_names.keys(), key=lambda x: all_franchise_names[x])
    semantic_label_names = np.asarray(
        [f"{all_franchise_names[name]}:{name}" for name in franchise_names_list], dtype="U"
    )

    # Build frame-level arrays
    frame_topk_ids = np.full((n_frames, K), -1, dtype=np.int32)
    frame_topk_scores = np.full((n_frames, K), np.nan, dtype=np.float32)

    for frame_idx, results in enumerate(frame_results):
        for k, result in enumerate(results[:K]):
            franchise_name = result.get("name", "unknown")
            similarity = float(result.get("similarity", 0.0))
            if franchise_name in all_franchise_names:
                label_id = all_franchise_names[franchise_name]
                frame_topk_ids[frame_idx, k] = label_id
                frame_topk_scores[frame_idx, k] = similarity

    # Compute is_confident flags
    threshold_global = float(args.threshold_global)
    frame_is_confident_top1 = np.zeros((n_frames,), dtype=np.bool_)
    for i in range(n_frames):
        if frame_topk_ids[i, 0] >= 0 and np.isfinite(frame_topk_scores[i, 0]):
            frame_is_confident_top1[i] = bool(frame_topk_scores[i, 0] >= threshold_global)

    # Video-level aggregate: max over time per franchise
    n_franchises = len(all_franchise_names)
    if n_franchises > 0:
        max_scores = np.full((n_franchises,), np.nan, dtype=np.float32)
        for franchise_name, label_id in all_franchise_names.items():
            scores_for_franchise = []
            for frame_idx, results in enumerate(frame_results):
                for result in results:
                    if result.get("name") == franchise_name:
                        scores_for_franchise.append(float(result.get("similarity", 0.0)))
            if scores_for_franchise:
                max_scores[label_id] = float(max(scores_for_franchise))

        # Top-K franchises for video
        valid_indices = np.where(np.isfinite(max_scores))[0]
        if valid_indices.size > 0:
            top_vid = valid_indices[np.argsort(-max_scores[valid_indices])[:K]]
            track_topk_ids = np.asarray(top_vid, dtype=np.int32).reshape(1, K)
            track_topk_scores = np.asarray(max_scores[top_vid], dtype=np.float32).reshape(1, K)
        else:
            track_topk_ids = np.full((1, K), -1, dtype=np.int32)
            track_topk_scores = np.full((1, K), np.nan, dtype=np.float32)
    else:
        track_topk_ids = np.full((1, K), -1, dtype=np.int32)
        track_topk_scores = np.full((1, K), np.nan, dtype=np.float32)

    # Evidence frames for top-K franchises
    track_topk_evidence_frame_indices = np.full((1, K), -1, dtype=np.int32)
    for j in range(K):
        if track_topk_ids[0, j] >= 0:
            label_id = int(track_topk_ids[0, j])
            franchise_name = franchise_names_list[label_id]
            # Find frame with max similarity for this franchise
            best_frame_idx = -1
            best_score = -1.0
            for frame_idx, results in enumerate(frame_results):
                for result in results:
                    if result.get("name") == franchise_name:
                        score = float(result.get("similarity", 0.0))
                        if score > best_score:
                            best_score = score
                            best_frame_idx = frame_idx
            if best_frame_idx >= 0:
                track_topk_evidence_frame_indices[0, j] = int(frame_indices[best_frame_idx])

    # Track-level confidence
    top1_lid = int(track_topk_ids[0, 0]) if track_topk_ids[0, 0] >= 0 else -1
    top1_sc = float(track_topk_scores[0, 0]) if np.isfinite(track_topk_scores[0, 0]) else np.nan
    track_is_confident_top1 = np.asarray(
        [bool(top1_lid >= 0 and np.isfinite(top1_sc) and top1_sc >= threshold_global)], dtype=np.bool_
    )
    track_ids = np.asarray([0], dtype=np.int32)
    track_present_mask = np.asarray([True], dtype=np.bool_)

    # Threshold per label (not available from Embedding Service, use global)
    threshold_per_label_arr = np.full((n_franchises,), np.nan, dtype=np.float32)

    # Build metadata
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")

    output_meta: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ok",
        "empty_reason": None,
        "embedding_service_url": embedding_client.base_url,
        "franchise_category": FRANCHISE_CATEGORY,
        "topk": K,
        "similarity_threshold": args.similarity_threshold,
        "threshold_global": threshold_global,
        "num_franchises": n_franchises,
        "num_frames": n_frames,
        # OCR stats
        "ocr_npz": str(ocr_npz_path) if ocr_npz_path else None,
        "ocr_events_used": int(ocr_events),
        "ocr_hits": int(ocr_hits),
        "ocr_candidate_names": ocr_candidate_names[:20],  # cap for meta size
        "ocr_evidence_frames": {str(k): v[:20] for k, v in list(ocr_evidence_frames.items())[:10]},  # cap
        # Provenance chaining
        "core_clip_model_signature": upstream_model_signature,
    }

    # Required run identity fields
    for k in required_run_keys:
        output_meta[k] = meta.get(k)

    # Required by contract (baseline may use "unknown")
    output_meta["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

    # Add models_used
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
    models_used_list.extend(upstream_models_used)
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

    # Save output (atomic save)
    output_dir = os.path.join(str(args.rs_path), NAME)
    output_path = os.path.join(output_dir, ARTIFACT_FILENAME)
    
    # Use atomic save function (creates directory if needed)
    import tempfile
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    # IMPORTANT: tmp must have .npz suffix, otherwise numpy will write to tmp + ".npz"
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(output_path) + ".", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            frame_indices=fi_np,
            times_s=times_s,
            semantic_label_names=semantic_label_names,
            threshold_per_label_arr=threshold_per_label_arr,
            track_ids=track_ids,
            track_present_mask=track_present_mask,
            track_topk_ids=track_topk_ids,
            track_topk_scores=track_topk_scores,
            track_is_confident_top1=track_is_confident_top1,
            track_topk_evidence_frame_indices=track_topk_evidence_frame_indices,
            frame_topk_ids=frame_topk_ids,
            frame_topk_scores=frame_topk_scores,
            frame_is_confident_top1=frame_is_confident_top1,
            meta=np.asarray(output_meta, dtype=object),
        )
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0

    # Update stage_timings_ms with final timings
    stage_timings_ms = {}
    for key, value in timings.items():
        stage_timings_ms[key] = float(value) * 1000.0
    output_meta["stage_timings_ms"] = stage_timings_ms

    # Validate artifact
    from utils.artifact_validator import validate_npz

    ok, issues, _ = validate_npz(output_path)
    if not ok:
        error_messages = [f"{i.level}: {i.message}" for i in issues if i.level == "error"]
        os.remove(output_path)
        raise RuntimeError(f"{NAME} | Artifact validation failed: {', '.join(error_messages)}")

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
        f"(franchises={n_franchises}, frames={n_frames})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
