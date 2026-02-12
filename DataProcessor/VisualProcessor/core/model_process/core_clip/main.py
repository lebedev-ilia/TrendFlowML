import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np      # type: ignore
import torch            # type: ignore
from PIL import Image   # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)
_repo_root = os.path.dirname(_path)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.resource_probe import pick_device
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz


NAME = "core_clip"
VERSION = "2.0"
SCHEMA_VERSION = "core_clip_npz_v1"
ARTIFACT_FILENAME = "embeddings.npz"
LOGGER = get_logger(NAME)

PROMPTS_VERSION = "v3_2026-01-16"


# Timing instrumentation
@contextmanager
def _time_block(name: str, timings: Dict[str, float]):
    """Context manager for timing a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = elapsed


def _print_timing_json(timings: Dict[str, float]) -> None:
    """Print timing information as JSON to stderr (for benchmark parsing)."""
    try:
        # Pretty print table for human readability
        if timings:
            total_time = timings.get("total", sum(timings.values()))
            print(f"\n{NAME} | Timing summary:", file=sys.stderr, flush=True)
            print(f"{'Stage':<30} {'Time (s)':>12} {'%':>8}", file=sys.stderr, flush=True)
            print("-" * 52, file=sys.stderr, flush=True)
            
            # Sort by time (descending) for better readability
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
            for stage, time_val in sorted_timings:
                if stage == "total":
                    continue
                pct = (time_val / total_time * 100) if total_time > 0 else 0.0
                print(f"{stage:<30} {time_val:>12.4f} {pct:>7.1f}%", file=sys.stderr, flush=True)
            
            if "total" in timings:
                print("-" * 52, file=sys.stderr, flush=True)
                print(f"{'TOTAL':<30} {timings['total']:>12.4f} {'100.0':>7}%", file=sys.stderr, flush=True)
            print("", file=sys.stderr, flush=True)
        
    except Exception:
        pass  # Ignore errors in timing output


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


def _extract_model_size_from_name(model_name: str) -> Optional[str]:
    """
    Extract model size (224/336/448) from model name for versioning.
    
    Note: Text embeddings are model-agnostic (same for 224/336/448 CLIP models),
    but we version by image model size for clarity and cache organization.
    This allows easy identification and management of caches for different model sizes.
    """
    # Try to extract size from model name (e.g., "clip_image_224" -> "224")
    if "_224" in model_name:
        return "224"
    elif "_336" in model_name:
        return "336"
    elif "_448" in model_name:
        return "448"
    return None


def _get_text_embeddings_cache_key(
    all_prompts: List[str], 
    triton_model_name: str, 
    triton_model_version: Optional[str], 
    prompts_version: str,
    model_size: Optional[str] = None
) -> str:
    """
    Generate cache key for text embeddings based on prompts and model.
    
    Text embeddings are model-agnostic (same for 224/336/448), but we version by model size
    for clarity and potential future model-specific optimizations.
    """
    # Include prompts IN ORDER (order matters for slices), model name, model version, prompts version, and model size in hash
    # Order is critical because slices (sl_shot, sl_aes, etc.) depend on the exact order
    prompts_str = "\n".join(all_prompts)  # Keep original order, not sorted!
    
    # Extract model size if not provided
    if model_size is None:
        model_size = _extract_model_size_from_name(triton_model_name)
    
    # Version includes: model name, version, prompts version, and model size
    model_info = f"{triton_model_name}:{triton_model_version or 'default'}:{prompts_version}:{model_size or 'unknown'}"
    payload = f"{prompts_str}\n{model_info}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _get_text_embeddings_cache_path(cache_key: str, model_size: Optional[str] = None) -> Optional[str]:
    """
    Get path to cached text embeddings file.
    
    Cache structure: {root}/cache/core_clip_text_embeddings/{model_size}/{cache_key}.npz
    This allows easy identification and management of caches for different model sizes.
    
    Tries DP_MODELS_ROOT first, falls back to auto-detecting bundled_models relative to repo root.
    """
    # Try DP_MODELS_ROOT first
    root = os.environ.get("DP_MODELS_ROOT")
    
    # Fallback: try to auto-detect bundled_models relative to repo root (same logic as Places365)
    if not isinstance(root, str) or not root.strip():
        # This file is at: VisualProcessor/core/model_process/core_clip/main.py
        # We need to find: DataProcessor/dp_models/bundled_models/
        current_dir = os.path.abspath(__file__)
        # Go up from core_clip/main.py to find DataProcessor directory
        # core_clip -> model_process -> core -> VisualProcessor -> DataProcessor
        for _ in range(6):  # Enough levels to reach DataProcessor
            current_dir = os.path.dirname(current_dir)
            # Check if we're at DataProcessor level
            if os.path.basename(current_dir) == "DataProcessor":
                candidate_bundled = os.path.join(current_dir, "dp_models", "bundled_models")
                if os.path.isdir(candidate_bundled):
                    root = candidate_bundled
                    break
    
    if not isinstance(root, str) or not root.strip():
        return None
    
    # Organize cache by model size for better management
    if model_size:
        cache_dir = os.path.join(os.path.abspath(root), "cache", "core_clip_text_embeddings", f"size_{model_size}")
    else:
        cache_dir = os.path.join(os.path.abspath(root), "cache", "core_clip_text_embeddings", "unknown")
    
    # Create directory with all parent directories
    try:
        os.makedirs(cache_dir, exist_ok=True, mode=0o755)
        # Verify directory was created
        if not os.path.isdir(cache_dir):
            LOGGER.warning(f"{NAME} | Cache directory was not created: {cache_dir}")
            return None
    except OSError as e:
        LOGGER.warning(f"{NAME} | Failed to create cache directory {cache_dir}: {e}")
        return None
    
    cache_file_path = os.path.join(cache_dir, f"{cache_key}.npz")
    LOGGER.debug(f"{NAME} | Cache path resolved: {cache_file_path}")
    return cache_file_path


def _load_cached_text_embeddings(cache_path: str) -> Optional[np.ndarray]:
    """Load cached text embeddings if available."""
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data.get("embeddings")
        if embeddings is None:
            return None
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as e:
        rel_cache_path = os.path.relpath(cache_path, os.getcwd()) if os.path.exists(cache_path) else cache_path
        LOGGER.warning(f"{NAME} | Failed to load cached text embeddings from {rel_cache_path}: {e}")
        return None


def _save_text_embeddings_cache(
    cache_path: str, 
    embeddings: np.ndarray, 
    prompts_count: int,
    model_size: Optional[str] = None,
    prompts_version: str = PROMPTS_VERSION
) -> None:
    """
    Save text embeddings to cache with versioning metadata.
    
    Cache includes:
    - embeddings: text embeddings array
    - prompts_count: number of prompts
    - model_size: model size (224/336/448) for identification
    - prompts_version: version of prompts for reproducibility
    - created_at: timestamp for cache management
    
    Text embeddings are model-agnostic (same for 224/336/448), but we version by model size
    for clarity and potential future model-specific optimizations.
    """
    try:
        # Ensure parent directory exists (defensive check, in case makedirs in _get_text_embeddings_cache_path failed)
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True, mode=0o755)
                if not os.path.isdir(cache_dir):
                    raise OSError(f"Directory {cache_dir} was not created")
            except OSError as dir_err:
                LOGGER.error(f"{NAME} | Failed to create cache directory {cache_dir}: {dir_err}")
                raise
        
        # Use tempfile.mkstemp for atomic save (same approach as _atomic_save_npz)
        # Important: numpy adds ".npz" automatically if the filename doesn't end with ".npz".
        # So we need to use a pattern that ends with .npz
        import tempfile
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(cache_path)[:-4] + ".",  # Remove .npz from basename
            suffix=".npz",
            dir=cache_dir
        )
        os.close(fd)
        
        cache_data = {
            "embeddings": embeddings,
            "prompts_count": prompts_count,
            "model_size": model_size or "unknown",
            "prompts_version": prompts_version,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        
        try:
            # Save to temporary file first
            np.savez_compressed(tmp_path, **cache_data)
            
            # Verify tmp file was created
            if not os.path.exists(tmp_path):
                raise OSError(f"Temporary cache file was not created: {tmp_path}")
            
            # Atomically replace
            os.replace(tmp_path, cache_path)
        except Exception:
            # Clean up temp file on error
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
        rel_cache_path = os.path.relpath(cache_path, os.getcwd()) if os.path.exists(cache_path) else cache_path
        size_info = f" (size={model_size})" if model_size else ""
        LOGGER.info(f"{NAME} | ✓ Cached text embeddings to {rel_cache_path} ({prompts_count} prompts{size_info})")
    except Exception as e:
        rel_cache_path = os.path.relpath(cache_path, os.getcwd()) if cache_path and os.path.exists(cache_path) else cache_path
        LOGGER.warning(f"{NAME} | Failed to save text embeddings cache to {rel_cache_path}: {e}")

# Prompt sets used by downstream modules + (optionally) future popularity heads.
# IMPORTANT:
# - Prompts are part of the model-facing contract for reproducibility.
# - Keep them compact, stable, and broadly applicable across content categories.
SHOT_QUALITY_PROMPTS: List[str] = [
    "high-quality professional video, sharp focus, good lighting",
    "cinematic video, stable camera, good composition",
    "good smartphone video, sharp and stable",
    "action camera footage, fast motion sports, wide angle",
    "screen recording of a phone or computer display",
    "webcam video call, low resolution",
    "security camera CCTV footage, low quality",
    "blurry shaky handheld video, low quality",
    "dark noisy low-light video",
    "overexposed washed out bright video",
]

# Prompt sets used by downstream modules (scene_classification).
# These are kept here so downstream modules can be strictly "core_clip-only" and offline.
SCENE_AESTHETIC_PROMPTS: List[str] = [
    "beautiful scenic landscape, aesthetic",
    "vibrant colors, pleasing lighting",
    "clean composition with clear subject",
    "blurry out-of-focus scene",
    "cluttered messy scene",
    "dull flat lighting, low contrast",
]

SCENE_LUXURY_PROMPTS: List[str] = [
    "luxury lifestyle, premium brand, high-end",
    "expensive car or supercar, luxury",
    "fine dining, gourmet food, upscale restaurant",
    "budget cheap product, low-end",
    "simple homemade meal, casual",
    "everyday ordinary scene, not luxurious",
]

SCENE_ATMOSPHERE_PROMPTS: List[str] = [
    "exciting energetic intense atmosphere",
    "calm relaxing peaceful atmosphere",
    "funny playful lighthearted atmosphere",
    "romantic warm cozy atmosphere",
    "tense dramatic suspenseful atmosphere",
    "sad emotional melancholic atmosphere",
]

# A small, universal topic set for popularity-oriented downstream heads.
# This is intentionally low-cardinality (not taxonomy): we want stable coarse signals.
POPULARITY_TOPIC_PROMPTS: List[str] = [
    "sports highlight, match, competition",
    "travel vlog, vacation, scenic views",
    "food cooking recipe, tasty meal",
    "fitness workout training",
    "dance or music performance",
    "gaming gameplay",
    "beauty makeup fashion tutorial",
    "educational how-to tutorial",
    "comedy prank funny moment",
    "cute pets animals",
]

# Prompt set used by `modules/cut_detection` (stylized transition classification).
# IMPORTANT: these prompts are embedded via CLIP text encoder here (core provider),
# so downstream modules do NOT load CLIP weights (single source-of-truth + no-network).
CUT_DETECTION_TRANSITION_PROMPTS: List[str] = [
    "hard cut",
    "fade",
    "dissolve",
    "whip pan",
    "zoom transition",
    "wipe transition",
    "slide transition",
    "glitch transition",
    "flash transition",
    "luma wipe transition",
]

_CLIP_IMG_SIZE = 224
_CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _parse_places365_categories(raw_text: str) -> List[str]:
    """
    Parse Places365 categories file (e.g. bundled_models/visual/places365/categories_places365.txt).
    Keep hierarchical subcategories like 'apartment_building/outdoor' (do NOT truncate to 'outdoor').
    """
    cats: List[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        raw_label = " ".join(parts[:-1]).strip()
        segs = raw_label.split("/")
        if len(segs) >= 3 and segs[0] == "":
            label = "/".join(segs[2:])
        else:
            label = raw_label.lstrip("/").strip()
        cats.append(label)
    return cats


def _load_places365_prompts_from_bundle() -> List[str]:
    """
    Deterministic Places365 prompt list (365 prompts), derived from bundled categories.
    Requires DP_MODELS_ROOT to point at a folder that contains visual/places365/categories_places365.txt.
    Falls back to auto-detecting bundled_models relative to repo root if DP_MODELS_ROOT is not set.
    """
    # Try DP_MODELS_ROOT first
    root = os.environ.get("DP_MODELS_ROOT")
    cat_path = None
    
    if isinstance(root, str) and root.strip():
        candidate = os.path.join(os.path.abspath(root), "visual", "places365", "categories_places365.txt")
        if os.path.isfile(candidate):
            cat_path = candidate
    
    # Fallback: try to auto-detect bundled_models relative to repo root
    if cat_path is None:
        # This file is at: VisualProcessor/core/model_process/core_clip/main.py
        # We need to find: DataProcessor/dp_models/bundled_models/visual/places365/categories_places365.txt
        current_dir = os.path.abspath(__file__)
        # Go up from core_clip/main.py to find DataProcessor directory
        # core_clip -> model_process -> core -> VisualProcessor -> DataProcessor
        for _ in range(6):  # Enough levels to reach DataProcessor
            current_dir = os.path.dirname(current_dir)
            # Check if we're at DataProcessor level
            if os.path.basename(current_dir) == "DataProcessor":
                candidate_bundled = os.path.join(
                    current_dir, "dp_models", "bundled_models", 
                    "visual", "places365", "categories_places365.txt"
                )
                if os.path.isfile(candidate_bundled):
                    cat_path = candidate_bundled
                    # Show relative path for cleaner output
                    rel_path = os.path.relpath(cat_path, current_dir) if current_dir else cat_path
                    LOGGER.info(f"{NAME} | Auto-detected Places365 categories: {rel_path}")
                    break
    
    if cat_path is None or not os.path.isfile(cat_path):
        root_hint = root if (isinstance(root, str) and root.strip()) else "not set"
        raise RuntimeError(
            f"{NAME} | Places365 categories file not found. "
            f"DP_MODELS_ROOT={root_hint}. "
            f"Please set DP_MODELS_ROOT to point to bundled_models directory "
            f"(e.g., export DP_MODELS_ROOT=/path/to/DataProcessor/dp_models/bundled_models), "
            f"or ensure the file exists at: dp_models/bundled_models/visual/places365/categories_places365.txt"
        )
    
    cats = _parse_places365_categories(open(cat_path, "r", encoding="utf-8").read())
    if len(cats) < 100:
        # Show relative path for cleaner error message
        rel_path = os.path.relpath(cat_path, os.getcwd()) if os.path.exists(cat_path) else cat_path
        raise RuntimeError(f"{NAME} | Places365 categories look too short (n={len(cats)}): {rel_path}")
    # CLIP prompt normalization: replace separators with spaces.
    prompts = [f"a photo of a {c.replace('_', ' ').replace('/', ' ')}" for c in cats]
    return prompts


def _load_triton_spec_via_model_manager(model_spec_name: str) -> dict:
    """
    Resolve Triton model spec via dp_models.ModelManager (no-network, reproducible).
    Returns dict with keys:
      - client: TritonHttpClient
      - rp: runtime_params
      - models_used_entry: dict (model_used)
    """
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    rp = rm.spec.runtime_params or {}
    handle = rm.handle or {}
    client = None
    if isinstance(handle, dict):
        client = handle.get("client")
    if client is None:
        raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
    return {"client": client, "rp": rp, "models_used_entry": rm.models_used_entry}

def _clip_preprocess_batch(frames_rgb_uint8: List[np.ndarray], *, image_size: int) -> np.ndarray:
    """
    Preprocess RGB uint8 frames to CLIP float32 tensor: (B,3,224,224).
    This intentionally avoids loading model weights (needed for Triton runtime).
    """
    s = int(image_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid image_size={image_size}")
    out: List[np.ndarray] = []
    for fr in frames_rgb_uint8:
        img = Image.fromarray(fr)
        img = img.resize((s, s), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) RGB
        arr = (arr - _CLIP_MEAN) / (_CLIP_STD + 1e-12)
        arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
        out.append(arr.astype(np.float32))
    return np.stack(out, axis=0) if out else np.zeros((0, 3, s, s), dtype=np.float32)


def _clip_resize_batch_uint8(frames_rgb_uint8: List[np.ndarray], *, image_size: int) -> np.ndarray:
    """
    Resize RGB uint8 frames to fixed square size and keep UINT8 NHWC.
    Used for baseline GPU contract where Triton ensemble owns normalize/layout conversion.
    """
    s = int(image_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid image_size={image_size}")
    out: List[np.ndarray] = []
    for fr in frames_rgb_uint8:
        img = Image.fromarray(fr)
        img = img.resize((s, s), resample=Image.BICUBIC)
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out, axis=0) if out else np.zeros((0, s, s, 3), dtype=np.uint8)

def _triton_infer_embeddings(
    *,
    client,
    model_name: str,
    model_version: Optional[str],
    input_name: str,
    input_tensor: np.ndarray,
    output_name: str,
    datatype: str,
) -> np.ndarray:
    res = client.infer(
        model_name=model_name,
        model_version=model_version,
        input_name=input_name,
        input_tensor=input_tensor,
        output_name=output_name,
        datatype=datatype,
    )
    out = np.asarray(res.output, dtype=np.float32)
    # L2 normalize (standard CLIP practice)
    norms = np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9
    return out / norms


def _require_frame_indices(meta: dict, name: str) -> List[int]:
    """
    Strict contract: frame sampling is owned by Segmenter/DataProcessor.
    Providers MUST use metadata[name].frame_indices and MUST NOT fallback.
    """
    block = meta.get(name)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{name} | metadata missing '{name}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(f"{name} | metadata '{name}.frame_indices' is empty/invalid.")
    return [int(x) for x in frame_indices]

def _require_union_timestamps_sec(meta: dict, *, total_frames: int) -> np.ndarray:
    """
    Contract: union_timestamps_sec is the source-of-truth time axis.
    Returns float32 array shape (total_frames,).
    """
    uts = meta.get("union_timestamps_sec") or meta.get("union_timestamps_s") or meta.get("times_s")
    if uts is None:
        raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
    arr = np.asarray(uts, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != int(total_frames):
        raise RuntimeError(
            f"{NAME} | union_timestamps_sec length mismatch: got {int(arr.shape[0])}, expected total_frames={int(total_frames)}"
        )
    return arr

def _atomic_save_npz(out_path: str, **kwargs) -> None:
    """
    Atomic NPZ save: write to tmp file in the same directory, then os.replace().
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    # Important: numpy adds ".npz" automatically if the filename doesn't end with ".npz".
    # If we used suffix=".tmp", we'd end up replacing out_path with an empty tmp file.
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".npz", dir=out_dir)
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


def compute_text_embeddings(
    model,
    device: str,
    prompts: List[str],
) -> np.ndarray:
    import clip  # type: ignore

    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        emb = model.encode_text(text)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    return emb.detach().cpu().numpy().astype(np.float32)


def init_clip(model_name: str, preferred_device: str = "auto") -> Tuple[torch.nn.Module, callable, str]:
    import clip # type: ignore

    device = pick_device(preferred_device)
    # Optional: pin OpenAI CLIP weights cache root (offline-friendly).
    dl_root = os.environ.get("DP_CLIP_WEIGHTS_DIR")
    if isinstance(dl_root, str) and dl_root.strip():
        model, preprocess = clip.load(model_name, device=device, download_root=str(dl_root).strip())
    else:
        model, preprocess = clip.load(model_name, device=device)

    model.eval()

    LOGGER.info(
        f"{NAME} | CLIP initialized | model: {model_name} | device: {device}"
    )

    return model, preprocess, device


def compute_clip_embeddings(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_name: str,
    batch_size: int,
) -> np.ndarray:

    if not frame_indices:
        LOGGER.warning(f"{NAME} | No frame indices provided")
        return np.zeros((0, 0), dtype=np.float32)

    model, preprocess, device = init_clip(model_name)

    n_frames = len(frame_indices)

    embeddings_out = None
    embed_dim = None

    try:
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                batch_ids = frame_indices[start : start + batch_size]

                images = []
                for idx in batch_ids:
                    frame = frame_manager.get(idx)
                    img = Image.fromarray(frame)
                    images.append(preprocess(img))

                batch_tensor = torch.stack(images).to(device)
                
                emb = model.encode_image(batch_tensor)

                # L2 normalization (standard CLIP practice)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)

                emb_np = emb.cpu().numpy().astype(np.float32)

                if embeddings_out is None:
                    embed_dim = emb_np.shape[1]
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)

                embeddings_out[start : start + len(batch_ids)] = emb_np

                if start % (batch_size * 10) == 0:
                    LOGGER.info(
                        f"{NAME} | processed {start + len(batch_ids)}/{n_frames}"
                    )
    finally:
        del model
        torch.cuda.empty_cache()

    return embeddings_out


def main():
    parser = argparse.ArgumentParser(description="Production CLIP per-frame embedding extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--weights-digest", default="unknown")
    parser.add_argument("--engine", default="torch")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--runtime", default="inprocess", choices=["inprocess", "triton"])
    # Triton (HTTP v2) options (used when --runtime=triton)
    parser.add_argument("--triton-http-url", default=None)
    # Prefer ModelManager specs for Triton (recommended; overrides explicit triton_* args when provided).
    parser.add_argument("--triton-image-model-spec", default=None, help="dp_models spec name (e.g., clip_image_triton)")
    parser.add_argument("--triton-text-model-spec", default=None, help="dp_models spec name (e.g., clip_text_triton)")
    # image embeddings
    parser.add_argument("--triton-image-model-name", default=None)
    parser.add_argument("--triton-image-model-version", default=None)
    parser.add_argument("--triton-image-input-name", default="INPUT__0")
    parser.add_argument("--triton-image-output-name", default="OUTPUT__0")
    parser.add_argument("--triton-image-datatype", default="FP32")
    # text embeddings (required by shot_quality_* contract)
    parser.add_argument("--triton-text-model-name", default=None)
    parser.add_argument("--triton-text-model-version", default=None)
    parser.add_argument("--triton-text-input-name", default="INPUT__0")
    parser.add_argument("--triton-text-output-name", default="OUTPUT__0")
    parser.add_argument("--triton-text-datatype", default="INT64")
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="openai_clip_224",
        choices=["openai_clip_224", "openai_clip_336", "openai_clip_448"],
        help="Image preprocess preset used for Triton runtime (controls resize).",
    )
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (must be provided by scheduler/orchestrator)")
    parser.add_argument("--triton-timeout-sec", type=float, default=60.0, help="Triton HTTP client timeout in seconds (default: 60.0, increased for text inference with many prompts)")
    parser.add_argument("--disable-text-cache", action="store_true", help="Disable text embeddings cache (for benchmarks)")
    args = parser.parse_args()

    # Initialize timing dictionary
    timings: Dict[str, float] = {}

    # Extract run identity for state_events
    meta_path = os.path.join(args.frames_dir, "metadata.json")
    meta = load_metadata(meta_path, NAME)
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

    with _time_block("total", timings):
        with _time_block("initialization", timings):
            total_frames = int(meta["total_frames"])

            frame_indices = _require_frame_indices(meta, NAME)
            if len(frame_indices) <= 0:
                # Contract: empty is not allowed for core_clip in baseline.
                raise RuntimeError(f"{NAME} | empty frame_indices is not allowed (no-fallback)")
            LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")
            union_timestamps_sec = _require_union_timestamps_sec(meta, total_frames=total_frames)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            if np.any(fi_np < 0) or np.any(fi_np >= int(total_frames)):
                raise RuntimeError(f"{NAME} | frame_indices out of range for total_frames={total_frames}")
            times_s = union_timestamps_sec[fi_np].astype(np.float32)

            frame_manager = FrameManager(
                frames_dir=args.frames_dir,
                chunk_size=meta.get("chunk_size", 32),
                cache_size=meta.get("cache_size", 2),
            )
            
            # Baseline contract: emit load_deps stage
            _emit_stage(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                stage="load_deps",
            )

        runtime = str(args.runtime or "inprocess").strip().lower()
        if runtime not in ("inprocess", "triton"):
            raise RuntimeError(f"{NAME} | invalid runtime: {runtime}")

        # In triton runtime we MUST compute both image and text embeddings via Triton (no local model inference).
        model = None
        preprocess = None
        device = "cpu"
        if runtime == "inprocess":
            with _time_block("model_init", timings):
                model, preprocess, device = init_clip(args.model_name, preferred_device="auto")
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise RuntimeError(f"{NAME} | --batch-size must be > 0 (scheduler-controlled); got {batch_size}")
        try:
            client = None
            if runtime == "triton":
                with _time_block("triton_init", timings):
                    img_mm = None
                    txt_mm = None
                    # Recommended path: resolve Triton params via ModelManager specs.
                    if args.triton_image_model_spec and args.triton_text_model_spec:
                        img_mm = _load_triton_spec_via_model_manager(str(args.triton_image_model_spec))
                        txt_mm = _load_triton_spec_via_model_manager(str(args.triton_text_model_spec))
                        client = img_mm["client"]
                    else:
                        # Legacy path: explicit Triton args (kept for backward compatibility).
                        if not args.triton_http_url:
                            # Allow orchestrator to provide the endpoint via env (still explicit; no silent defaults).
                            env_url = os.environ.get("TRITON_HTTP_URL")
                            if isinstance(env_url, str) and env_url.strip():
                                args.triton_http_url = env_url.strip()
                        if not args.triton_http_url:
                            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url (or TRITON_HTTP_URL env) (no-fallback)")

                        from dp_triton import TritonHttpClient, TritonError  # local import (repo code)

                        client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=float(args.triton_timeout_sec))
                        if not client.ready():
                            raise TritonError(
                                f"{NAME} | Triton is not ready at {args.triton_http_url}",
                                error_code="triton_unavailable",
                            )

                        if not args.triton_image_model_name:
                            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-image-model-name")
                        if not args.triton_text_model_name:
                            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-text-model-name")

            # Baseline contract: emit process_frames stage
            _emit_stage(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                stage="process_frames",
            )
            
            # --- image embeddings ---
            image_inference_time = 0.0
            image_preprocessing_time = 0.0
            image_frame_loading_time = 0.0
            
            with _time_block("image_embeddings_total", timings):
                n_frames = len(frame_indices)
                embeddings_out = None
                embed_dim = None
                with torch.no_grad():
                    start = 0
                    while start < n_frames:
                        batch_ids = frame_indices[start : start + batch_size]
                        
                        frame_load_start = time.perf_counter()
                        images: List[np.ndarray] = [frame_manager.get(idx) for idx in batch_ids]
                        image_frame_loading_time += time.perf_counter() - frame_load_start

                        if runtime == "triton":
                            assert client is not None
                            preset = str(args.triton_preprocess_preset or "openai_clip_224").strip().lower()
                            if preset == "openai_clip_224":
                                image_size = 224
                            elif preset == "openai_clip_336":
                                image_size = 336
                            elif preset == "openai_clip_448":
                                image_size = 448
                            else:
                                raise RuntimeError(f"{NAME} | unknown triton_preprocess_preset: {preset}")

                            if "img_mm" in locals() and img_mm is not None:
                                rp = img_mm["rp"]
                                triton_model_name = str(rp.get("triton_model_name"))
                                triton_model_version = str(rp.get("triton_model_version") or "") or None
                                triton_input_name = str(rp.get("triton_input_name"))
                                triton_output_name = str(rp.get("triton_output_name"))
                                triton_datatype = str(rp.get("triton_input_datatype") or "FP32")
                            else:
                                triton_model_name = str(args.triton_image_model_name)
                                triton_model_version = str(args.triton_image_model_version) if args.triton_image_model_version else None
                                triton_input_name = str(args.triton_image_input_name)
                                triton_output_name = str(args.triton_image_output_name)
                                triton_datatype = str(args.triton_image_datatype)
                            dt = str(triton_datatype or "FP32").strip().upper()
                            
                            preprocess_start = time.perf_counter()
                            if dt == "UINT8":
                                inp = _clip_resize_batch_uint8(images, image_size=image_size)  # (B,S,S,3) uint8
                            else:
                                inp = _clip_preprocess_batch(images, image_size=image_size)  # (B,3,S,S) float32
                            image_preprocessing_time += time.perf_counter() - preprocess_start
                            
                            infer_start = time.perf_counter()
                            try:
                                emb_np = _triton_infer_embeddings(
                                    client=client,
                                    model_name=triton_model_name,
                                    model_version=triton_model_version,
                                    input_name=triton_input_name,
                                    input_tensor=inp,
                                    output_name=triton_output_name,
                                    datatype=triton_datatype,
                                )
                            except Exception as e:
                                raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e
                            image_inference_time += time.perf_counter() - infer_start
                        else:
                            assert model is not None and preprocess is not None
                            imgs = [preprocess(Image.fromarray(fr)) for fr in images]
                            batch_tensor = torch.stack(imgs).to(device)
                            emb = model.encode_image(batch_tensor)
                            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
                            emb_np = emb.detach().cpu().numpy().astype(np.float32)

                        if embeddings_out is None:
                            embed_dim = int(emb_np.shape[1])
                            embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)
                        embeddings_out[start : start + len(batch_ids)] = emb_np
                        
                        # Baseline contract: granular progress (>=10 updates)
                        processed = start + len(batch_ids)
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
                        
                        if start % (batch_size * 10) == 0:
                            LOGGER.info(f"{NAME} | processed {processed}/{n_frames}")
                        start += len(batch_ids)
            
            timings["image_frame_loading"] = image_frame_loading_time
            timings["image_preprocessing"] = image_preprocessing_time
            timings["image_inference"] = image_inference_time

            embeddings = embeddings_out if embeddings_out is not None else np.zeros((0, 0), dtype=np.float32)

            # Baseline contract: emit post_process stage
            _emit_stage(
                rs_path=args.rs_path,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                stage="post_process",
            )

            # --- text embeddings for downstream modules ---
            # Compute all prompt embeddings in one batch, then split.
            with _time_block("text_embeddings_prep", timings):
                places365_prompts = _load_places365_prompts_from_bundle()
                all_prompts: List[str] = []
                all_prompts.extend(SHOT_QUALITY_PROMPTS)
                all_prompts.extend(SCENE_AESTHETIC_PROMPTS)
                all_prompts.extend(SCENE_LUXURY_PROMPTS)
                all_prompts.extend(SCENE_ATMOSPHERE_PROMPTS)
                all_prompts.extend(CUT_DETECTION_TRANSITION_PROMPTS)
                all_prompts.extend(POPULARITY_TOPIC_PROMPTS)
                all_prompts.extend(places365_prompts)

                n_shot = len(SHOT_QUALITY_PROMPTS)
                n_aes = len(SCENE_AESTHETIC_PROMPTS)
                n_lux = len(SCENE_LUXURY_PROMPTS)
                n_atm = len(SCENE_ATMOSPHERE_PROMPTS)
                n_cut = len(CUT_DETECTION_TRANSITION_PROMPTS)
                n_pop = len(POPULARITY_TOPIC_PROMPTS)
                n_p365 = len(places365_prompts)
                sl_shot = slice(0, n_shot)
                sl_aes = slice(n_shot, n_shot + n_aes)
                sl_lux = slice(n_shot + n_aes, n_shot + n_aes + n_lux)
                sl_atm = slice(n_shot + n_aes + n_lux, n_shot + n_aes + n_lux + n_atm)
                sl_cut = slice(n_shot + n_aes + n_lux + n_atm, n_shot + n_aes + n_lux + n_atm + n_cut)
                sl_pop = slice(
                    n_shot + n_aes + n_lux + n_atm + n_cut,
                    n_shot + n_aes + n_lux + n_atm + n_cut + n_pop,
                )
                sl_p365 = slice(
                    n_shot + n_aes + n_lux + n_atm + n_cut + n_pop,
                    n_shot + n_aes + n_lux + n_atm + n_cut + n_pop + n_p365,
                )

            if runtime == "triton":
                assert client is not None
                import clip  # type: ignore

                # Determine model info for cache key
                if "txt_mm" in locals() and txt_mm is not None:
                    rp = txt_mm["rp"]
                    triton_model_name = str(rp.get("triton_model_name"))
                    triton_model_version = str(rp.get("triton_model_version") or "") or None
                    triton_input_name = str(rp.get("triton_input_name"))
                    triton_output_name = str(rp.get("triton_output_name"))
                    triton_datatype = str(rp.get("triton_input_datatype") or "INT64")
                else:
                    triton_model_name = str(args.triton_text_model_name)
                    triton_model_version = str(args.triton_text_model_version) if args.triton_text_model_version else None
                    triton_input_name = str(args.triton_text_input_name)
                    triton_output_name = str(args.triton_text_output_name)
                    triton_datatype = str(args.triton_text_datatype)
                
                # Determine model size from preset or image model name for versioning
                # Text embeddings are model-agnostic, but we version by image model size for clarity
                model_size = None
                if runtime == "triton":
                    preset = str(args.triton_preprocess_preset or "openai_clip_224").strip().lower()
                    if preset == "openai_clip_224":
                        model_size = "224"
                    elif preset == "openai_clip_336":
                        model_size = "336"
                    elif preset == "openai_clip_448":
                        model_size = "448"
                    else:
                        # Try to extract from image model name
                        if "img_mm" in locals() and img_mm is not None:
                            img_rp = img_mm["rp"]
                            img_model_name = str(img_rp.get("triton_model_name", ""))
                        else:
                            img_model_name = str(args.triton_image_model_name or "")
                        model_size = _extract_model_size_from_name(img_model_name)
                    LOGGER.info(f"{NAME} | Determined model_size={model_size} from preset={preset}")
                
                # Try to load from cache first (unless disabled)
                all_text_embeddings = None
                if not args.disable_text_cache:
                    cache_key = _get_text_embeddings_cache_key(
                        all_prompts, 
                        triton_model_name, 
                        triton_model_version, 
                        PROMPTS_VERSION,
                        model_size=model_size
                    )
                    cache_path = _get_text_embeddings_cache_path(cache_key, model_size=model_size)
                    
                    if cache_path:
                        rel_cache_path = os.path.relpath(cache_path, os.getcwd()) if os.path.exists(cache_path) else cache_path
                        LOGGER.info(f"{NAME} | Checking cache: {rel_cache_path} (model_size={model_size}, cache_key={cache_key[:8]}...)")
                        cached_emb = _load_cached_text_embeddings(cache_path)
                        if cached_emb is not None and cached_emb.shape[0] == len(all_prompts):
                            all_text_embeddings = cached_emb
                            LOGGER.info(f"{NAME} | Loaded {len(all_prompts)} text embeddings from cache ({rel_cache_path})")
                            timings["text_inference"] = 0.0  # Cache hit - no inference time
                        elif cached_emb is not None:
                            LOGGER.warning(f"{NAME} | Cache mismatch: cached has {cached_emb.shape[0]} prompts, expected {len(all_prompts)}")
                        else:
                            LOGGER.debug(f"{NAME} | Cache miss: will compute text embeddings")
                    else:
                        LOGGER.debug(f"{NAME} | Cache path unavailable (DP_MODELS_ROOT not set?)")
                else:
                    cache_path = None
                    LOGGER.debug(f"{NAME} | Text cache disabled")
                
                # If not cached, compute embeddings via Triton
                if all_text_embeddings is None:
                    toks = clip.tokenize(all_prompts)  # (P,77) int64
                    toks_np = toks.detach().cpu().numpy().astype(np.int64)
                    
                    # Text encoder returns per-token embeddings (B,77,512) in the new export,
                    # or (B,512) in older backward-compatible exports.
                    # We select EOT embedding OUTSIDE the model using argmax(tokens) per row.
                    # NOTE: clip_text is typically configured with max_batch_size=64.
                    # We must chunk large prompt lists (e.g. Places365=365 prompts) to avoid HTTP 400.
                    text_inference_time = 0.0
                    seq_chunks: List[np.ndarray] = []
                    max_text_batch = 64
                    for start in range(0, int(toks_np.shape[0]), int(max_text_batch)):
                        chunk = toks_np[start : start + int(max_text_batch)]
                        infer_start = time.perf_counter()
                        try:
                            seq = _triton_infer_embeddings(
                                client=client,
                                model_name=triton_model_name,
                                model_version=triton_model_version,
                                input_name=triton_input_name,
                                input_tensor=chunk,
                                output_name=triton_output_name,
                                datatype=triton_datatype,
                            )
                        except Exception as e:
                            raise RuntimeError(f"{NAME} | Triton text infer failed: {e}") from e
                        text_inference_time += time.perf_counter() - infer_start
                        seq_chunks.append(np.asarray(seq))
                    timings["text_inference"] = text_inference_time
                    
                    seq = np.concatenate(seq_chunks, axis=0) if seq_chunks else np.zeros((0, 512), dtype=np.float32)

                    with _time_block("text_embeddings_postproc", timings):
                        arr = np.asarray(seq, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[-1] == 512:
                            # Backward-compatible: old clip_text returned (B,512)
                            all_text_embeddings = arr.astype(np.float32)
                        else:
                            if arr.ndim != 3 or arr.shape[1] != 77 or arr.shape[2] != 512:
                                raise RuntimeError(f"{NAME} | clip_text output has invalid shape: {arr.shape}")
                            B = int(arr.shape[0])
                            if B != int(toks_np.shape[0]):
                                raise RuntimeError(f"{NAME} | clip_text batch mismatch: out_B={B}, in_B={int(toks_np.shape[0])}")
                            eot_pos = np.argmax(toks_np, axis=1).astype(np.int64)  # (B,)
                            eot_pos = np.clip(eot_pos, 0, 76)
                            rows = np.arange(B, dtype=np.int64)
                            all_text_embeddings = arr[rows, eot_pos, :].astype(np.float32)  # (B,512)
                    
                    # Save to cache for future use (unless disabled)
                    # Text embeddings are computed once per model size and reused for all videos
                    if not args.disable_text_cache and cache_path:
                        LOGGER.info(f"{NAME} | Saving text embeddings to cache (model_size={model_size}, prompts={len(all_prompts)})")
                        _save_text_embeddings_cache(
                            cache_path, 
                            all_text_embeddings, 
                            len(all_prompts),
                            model_size=model_size,
                            prompts_version=PROMPTS_VERSION
                        )
                    elif not args.disable_text_cache:
                        LOGGER.warning(f"{NAME} | Cannot save cache: cache_path is None")
            else:
                # inprocess runtime: determine model size from model_name
                assert model is not None
                model_size_inprocess = _extract_model_size_from_name(args.model_name) if hasattr(args, 'model_name') else None
                
                # Try to load from cache first (unless disabled)
                all_text_embeddings = None
                cache_path_inprocess = None
                if not args.disable_text_cache:
                    # For inprocess, use model_name as triton_model_name equivalent
                    cache_key_inprocess = _get_text_embeddings_cache_key(
                        all_prompts,
                        args.model_name if hasattr(args, 'model_name') else "inprocess",
                        None,  # no version for inprocess
                        PROMPTS_VERSION,
                        model_size=model_size_inprocess
                    )
                    cache_path_inprocess = _get_text_embeddings_cache_path(cache_key_inprocess, model_size=model_size_inprocess)
                    
                    if cache_path_inprocess:
                        cached_emb = _load_cached_text_embeddings(cache_path_inprocess)
                        if cached_emb is not None and cached_emb.shape[0] == len(all_prompts):
                            all_text_embeddings = cached_emb
                            rel_cache_path = os.path.relpath(cache_path_inprocess, os.getcwd()) if os.path.exists(cache_path_inprocess) else cache_path_inprocess
                            LOGGER.info(f"{NAME} | Loaded {len(all_prompts)} text embeddings from cache ({rel_cache_path})")
                            timings["text_inference"] = 0.0  # Cache hit - no inference time
                
                # If not cached, compute embeddings
                if all_text_embeddings is None:
                    with _time_block("text_inference", timings):
                        all_text_embeddings = compute_text_embeddings(
                            model=model,
                            device=device,
                            prompts=all_prompts,
                        )
                    
                    # Save to cache for future use (unless disabled)
                    if not args.disable_text_cache and cache_path_inprocess:
                        _save_text_embeddings_cache(
                            cache_path_inprocess,
                            all_text_embeddings,
                            len(all_prompts),
                            model_size=model_size_inprocess,
                            prompts_version=PROMPTS_VERSION
                        )

            shot_quality_text_embeddings = np.asarray(all_text_embeddings[sl_shot], dtype=np.float32)
            scene_aesthetic_text_embeddings = np.asarray(all_text_embeddings[sl_aes], dtype=np.float32)
            scene_luxury_text_embeddings = np.asarray(all_text_embeddings[sl_lux], dtype=np.float32)
            scene_atmosphere_text_embeddings = np.asarray(all_text_embeddings[sl_atm], dtype=np.float32)
            cut_detection_transition_text_embeddings = np.asarray(all_text_embeddings[sl_cut], dtype=np.float32)
            popularity_topic_text_embeddings = np.asarray(all_text_embeddings[sl_pop], dtype=np.float32)
            places365_text_embeddings = np.asarray(all_text_embeddings[sl_p365], dtype=np.float32)

        finally:
            if model is not None:
                del model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        LOGGER.info(
            f"{NAME} | embeddings computed | shape: {embeddings.shape}"
        )

        frame_manager.close()

        # Baseline contract: emit save stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="save",
        )
        
        with _time_block("saving", timings):
                out_dir = os.path.join(args.rs_path, NAME)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, ARTIFACT_FILENAME)

                created_at = datetime.utcnow().isoformat()
                runtime_meta = "triton-gpu" if runtime == "triton" else "inprocess"
                # Device semantics:
                # - inprocess: actual compute device ("cuda" or "cpu")
                # - triton: we assume GPU-backed triton by default in this project ("cuda")
                device_meta = "cuda" if runtime == "triton" else str(device)
                # Baseline contract: convert timings to stage_timings_ms (milliseconds)
                stage_timings_ms: Dict[str, float] = {}
                for key, value in timings.items():
                    stage_timings_ms[key] = float(value) * 1000.0  # Convert seconds to milliseconds
                
                # Log stage timings for profiling
                LOGGER.info(f"{NAME} | stage timings (ms): {', '.join([f'{k}={v:.1f}' for k, v in sorted(stage_timings_ms.items())])}")
                
                meta_out = {
                    "producer": NAME,
                    "producer_version": VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "created_at": created_at,
                    "status": "ok",
                    "empty_reason": None,
                    "model_name": args.model_name,
                    "total_frames": int(total_frames),
                    "batch_size": int(batch_size),
                    "runtime": runtime_meta,
                    "device": device_meta,
                    "stage_timings_ms": stage_timings_ms,
                }
                required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
                missing = [k for k in required_run_keys if not meta.get(k)]
                if missing:
                    raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
                for k in required_run_keys:
                    meta_out[k] = meta.get(k)
                # Required by contract (baseline may use "unknown")
                dpv = meta.get("dataprocessor_version") or "unknown"
                meta_out["dataprocessor_version"] = str(dpv)
                meta_out["prompts_version"] = PROMPTS_VERSION

                # PR-3: model system baseline (core_clip may use both image+text encoders on Triton).
                models_used_list = []
                if runtime == "triton" and "img_mm" in locals() and "txt_mm" in locals() and img_mm is not None and txt_mm is not None:
                    # Use ModelManager entries (preferred path)
                    models_used_list.append(img_mm["models_used_entry"])
                    models_used_list.append(txt_mm["models_used_entry"])
                elif runtime == "triton":
                    # Legacy Triton path: create entries from explicit args
                    # Image encoder
                    triton_image_model_name = str(args.triton_image_model_name) if args.triton_image_model_name else "unknown"
                    triton_image_model_version = str(args.triton_image_model_version) if args.triton_image_model_version else "unknown"
                    models_used_list.append(
                        model_used(
                            model_name=triton_image_model_name,
                            model_version=triton_image_model_version,
                            weights_digest="unknown",  # Not available in legacy path
                            runtime=runtime_meta,
                            engine="triton",
                            precision=str(args.triton_image_datatype or "unknown").lower(),
                            device=device_meta,
                        )
                    )
                    # Text encoder
                    triton_text_model_name = str(args.triton_text_model_name) if args.triton_text_model_name else "unknown"
                    triton_text_model_version = str(args.triton_text_model_version) if args.triton_text_model_version else "unknown"
                    models_used_list.append(
                        model_used(
                            model_name=triton_text_model_name,
                            model_version=triton_text_model_version,
                            weights_digest="unknown",  # Not available in legacy path
                            runtime=runtime_meta,
                            engine="triton",
                            precision=str(args.triton_text_datatype or "unknown").lower(),
                            device=device_meta,
                        )
                    )
                else:
                    # Inprocess runtime: single model
                    models_used_list.append(
                        model_used(
                            model_name=str(args.model_name),
                            model_version=str(args.model_version or "unknown"),
                            weights_digest=str(args.weights_digest or "unknown"),
                            runtime=runtime_meta,
                            engine=str(args.engine or "unknown"),
                            precision=str(args.precision or "unknown"),
                            device=device_meta,
                        )
                    )
                meta_out = apply_models_meta(meta_out, models_used=models_used_list)

                _atomic_save_npz(
                    out_path,
                    # legacy fields (kept)
                    version=VERSION,
                    created_at=created_at,
                    model_name=args.model_name,
                    total_frames=total_frames,
                    frame_indices=np.array(frame_indices, dtype=np.int32),
                    times_s=times_s,
                    frame_embeddings=embeddings,
                    # downstream contract for shot_quality
                    shot_quality_prompts=np.array(SHOT_QUALITY_PROMPTS, dtype=object),
                    shot_quality_text_embeddings=shot_quality_text_embeddings,
                    # downstream contract for scene_classification semantics (zero-shot prompts)
                    scene_aesthetic_prompts=np.array(SCENE_AESTHETIC_PROMPTS, dtype=object),
                    scene_aesthetic_text_embeddings=scene_aesthetic_text_embeddings,
                    scene_luxury_prompts=np.array(SCENE_LUXURY_PROMPTS, dtype=object),
                    scene_luxury_text_embeddings=scene_luxury_text_embeddings,
                    scene_atmosphere_prompts=np.array(SCENE_ATMOSPHERE_PROMPTS, dtype=object),
                    scene_atmosphere_text_embeddings=scene_atmosphere_text_embeddings,
                    # downstream contract for cut_detection (stylized transitions, CLIP zero-shot)
                    cut_detection_transition_prompts=np.array(CUT_DETECTION_TRANSITION_PROMPTS, dtype=object),
                    cut_detection_transition_text_embeddings=cut_detection_transition_text_embeddings,
                    # popularity-oriented coarse topic prompts (optional downstream heads)
                    popularity_topic_prompts=np.array(POPULARITY_TOPIC_PROMPTS, dtype=object),
                    popularity_topic_text_embeddings=popularity_topic_text_embeddings,
                    # downstream contract: Places365 zero-shot label embeddings for fusion
                    places365_prompts=np.array(places365_prompts, dtype=object),
                    places365_text_embeddings=places365_text_embeddings,
                    # canonical meta (required by artifact_validator)
                    meta=np.asarray(meta_out, dtype=object),
                )

                ok, issues, _ = validate_npz(out_path)
                if not ok:
                    msg = "; ".join([f"{i.level}:{i.message}" for i in issues])
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    raise RuntimeError(f"{NAME} | saved artifact failed validation: {msg}")

        rel_out_path = os.path.relpath(out_path, os.getcwd()) if os.path.exists(out_path) else out_path
        LOGGER.info(f"{NAME} | Saved result: {rel_out_path}")
        
        # Baseline contract: emit done stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="done",
        )
        
        # Output timing information for benchmark parsing
        _print_timing_json(timings)


if __name__ == "__main__":
    main()