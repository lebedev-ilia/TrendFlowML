#!/VisualProcessor/core/model_process/.model_process_venv python3
"""
Production-ready depth extraction (MiDaS family) via Triton.

Design decisions and behavior (short):
- We assume FrameManager.get(idx) returns an RGB uint8 HxWx3 image (this matches other modules).
  If your FrameManager returns BGR images, set --frames-bgr to True.
- Outputs:
  * <rs_path>/core_depth_midas/depth.npz  -- compressed NPZ containing:
      - depth_maps: float32 array (N, out_h, out_w)
      - frame_indices: int32 (N,)
      - meta (dict, object-array)  # includes created_at, models_used, run identity, etc.
- Triton-only: preprocessing lives in Triton; no torch.hub / no local torch engine.
"""

import sys
from pathlib import Path
# Ensure VisualProcessor root is first on path so "utils" resolves to VisualProcessor/utils (subprocess entry point).
_vp = Path(__file__).resolve().parent  # core_depth_midas/
for _ in range(3):
    _vp = _vp.parent  # -> model_process -> core -> VisualProcessor
sys.path.insert(0, str(_vp))

import argparse
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np      # type: ignore
import cv2              # type: ignore

# repo root (needed for dp_triton)
_root = str(_vp.parent)
if _root not in sys.path:
    sys.path.append(_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

VERSION = "2.2"
NAME = "core_depth_midas"
SCHEMA_VERSION = "core_depth_midas_npz_v3"
ARTIFACT_FILENAME = "depth.npz"
LOGGER = get_logger(NAME)


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


def _atomic_save_npz(out_path: str, **kwargs: Any) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
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
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")  # type: ignore[name-defined]
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
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


def _require_union_times_s(meta: Dict[str, Any], frame_indices: List[int]) -> np.ndarray:
    union_ts = meta.get("union_timestamps_sec")
    if not isinstance(union_ts, list) or not union_ts:
        raise RuntimeError(f"{NAME} | frames metadata missing 'union_timestamps_sec' (strict time axis, no-fallback).")
    ts = np.asarray(union_ts, dtype=np.float32).reshape(-1)
    idx = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    try:
        out = ts[idx].astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"{NAME} | failed to index union_timestamps_sec by frame_indices: {e}") from e
    return out


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


def _preset_to_input_size(preset: str) -> int:
    p = str(preset or "").strip().lower()
    if p in ("midas_256", "256"):
        return 256
    if p in ("midas_384", "384"):
        return 384
    if p in ("midas_512", "512"):
        return 512
    raise ValueError(f"{NAME} | unknown triton_preprocess_preset: {preset!r}")


def _prep_batch_rgb_uint8(frames: List[np.ndarray], *, input_size: int, frames_are_bgr: bool) -> np.ndarray:
    """
    Minimal client-side formatting for Triton (NOT full preprocessing):
    - ensure RGB
    - resize to (S,S)
    - keep UINT8 NHWC (baseline GPU contract)

    Full preprocessing (normalize/layout conversion to model FP32 NCHW) lives in Triton ensemble.
    """
    s = int(input_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid input_size={input_size}")
    out: List[np.ndarray] = []
    for fr in frames:
        if frames_are_bgr:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        # Resize (square) for preset
        fr_r = cv2.resize(fr, (s, s), interpolation=cv2.INTER_AREA)
        # Keep UINT8 NHWC
        out.append(np.asarray(fr_r, dtype=np.uint8))
    if not out:
        return np.zeros((0, s, s, 3), dtype=np.uint8)
    return np.stack(out, axis=0).astype(np.uint8)


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Production MiDaS depth extractor")
    parser.add_argument("--frames-dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--rs-path", type=str, required=True, help="Path to VisualProcessor result_store")
    # Triton-only policy (prod): local torch/onnx engines are removed.
    parser.add_argument("--runtime", type=str, default="triton", choices=["triton"], help="Runtime (prod: triton only)")
    parser.add_argument("--triton-http-url", type=str, default=None)
    # Preferred: resolve Triton params via ModelManager specs (recommended; overrides explicit triton_* args when provided).
    parser.add_argument("--triton-model-spec", type=str, default=None, help="dp_models spec name (e.g., midas_256_triton)")
    parser.add_argument("--triton-model-name", type=str, default=None)
    parser.add_argument("--triton-model-version", type=str, default=None)
    parser.add_argument("--triton-input-name", type=str, default="INPUT__0")
    parser.add_argument("--triton-output-name", type=str, default="OUTPUT__0")
    # Triton ensemble expects UINT8 NHWC input.
    parser.add_argument("--triton-datatype", type=str, default="UINT8")
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="midas_384",
        choices=["midas_256", "midas_384", "midas_512"],
        help="Input preset (square size) for Triton depth model.",
    )
    parser.add_argument("--model-version", type=str, default="unknown")
    parser.add_argument("--weights-digest", type=str, default="unknown")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--out-width", type=int, default=384, help="Output width of saved depth maps (downsampled) to store in NPZ")
    parser.add_argument("--out-height", type=int, default=384, help="Output height of saved depth maps (downsampled) to store in NPZ")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (must be provided by scheduler/orchestrator)")
    parser.add_argument("--frames-bgr", action="store_true", help="Set if FrameManager returns BGR images instead of RGB")
    args = parser.parse_args()

    # Triton-only mode: device is not observable from client reliably; we assume GPU-backed Triton.
    runtime = str(args.runtime or "triton").strip().lower()
    if runtime != "triton":
        raise RuntimeError(f"{NAME} | runtime must be triton (no-fallback), got: {runtime}")

    # Load metadata
    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta.get("total_frames", 0))
    platform_id = str(meta.get("platform_id") or "")
    video_id = str(meta.get("video_id") or "")
    run_id = str(meta.get("run_id") or "")

    # Stage timings (seconds → later converted to ms in meta)
    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # Emit start stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="start",
    )

    # Initialization stage
    t_init_start = time.perf_counter()

    # Strict sampling contract: Segmenter must provide per-provider indices in metadata[NAME].frame_indices.
    block = meta.get(NAME)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | metadata missing '{NAME}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{NAME} | metadata '{NAME}.frame_indices' is empty/invalid.")
    frame_indices = [int(x) for x in frame_indices_raw]
    LOGGER.info(f"{NAME} | main | sampled frames: {len(frame_indices)} / total={total_frames}")
    if len(frame_indices) <= 0:
        raise RuntimeError(f"{NAME} | empty frame_indices is not allowed (no-fallback)")

    # Strict time axis (no-fallback) + write times_s into the artifact.
    times_s = _require_union_times_s(meta, frame_indices)

    # Initialize FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )
    LOGGER.info(
        f"{NAME} | main | FrameManager initialized (chunk_size={meta.get('chunk_size', 32)}, cache_size={meta.get('cache_size', 2)})"
    )

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"{NAME} | --batch-size must be > 0 (scheduler-controlled); got {batch_size}")

    # Prepare output dir
    core_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(core_dir, exist_ok=True)

    # Triton client (repo-local)
    from dp_triton import TritonHttpClient, TritonError  # type: ignore

    mm_entry = None
    if args.triton_model_spec:
        mm_entry = _load_triton_spec_via_model_manager(str(args.triton_model_spec))
        client = mm_entry["client"]
        rp = mm_entry["rp"]
        # Override explicit args from runtime_params (single source-of-truth).
        args.triton_http_url = str(rp.get("triton_http_url") or args.triton_http_url or "")
        args.triton_model_name = str(rp.get("triton_model_name") or args.triton_model_name or "")
        args.triton_model_version = str(rp.get("triton_model_version") or "") or None
        args.triton_input_name = str(rp.get("triton_input_name") or args.triton_input_name)
        args.triton_output_name = str(rp.get("triton_output_name") or args.triton_output_name)
        args.triton_datatype = str(rp.get("triton_input_datatype") or args.triton_datatype)
    else:
        if not args.triton_http_url or not str(args.triton_http_url).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec (no-fallback)")
        if not args.triton_model_name or not str(args.triton_model_name).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec (no-fallback)")
        client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=240.0)
        if not client.ready():
            raise TritonError(f"{NAME} | Triton is not ready at {args.triton_http_url}", error_code="triton_unavailable")

    input_size = _preset_to_input_size(str(args.triton_preprocess_preset))
    out_size = (int(args.out_height), int(args.out_width))  # (H,W)

    # End of initialization
    timings["initialization"] = float(time.perf_counter() - t_init_start)

    # Emit load_deps stage (FrameManager + Triton client ready)
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )

    # Compute depth maps (batched)
    # NOTE: Performance characteristics:
    # - Triton ensemble `midas_256` includes CPU preprocessing (preprocess_midas_256) + GPU inference (midas_256_onnx)
    # - For batch_size=1: CPU preprocessing overhead is minimal, each request is fast
    # - For batch_size=8: CPU preprocessing becomes bottleneck, coordination overhead between CPU/GPU steps
    # - Post-processing (resize, mean, std, percentiles) is done sequentially per frame, not optimized for batches
    # - Recommendation: Use batch_size=1 for unit-cost, batch_size=4-8 only if throughput is critical and CPU preprocessing is optimized
    t_infer_start = time.perf_counter()
    try:
        n = len(frame_indices)
        out_h, out_w = out_size
        depth_maps = np.full((n, out_h, out_w), np.nan, dtype=np.float32)
        depth_mean = np.full((n,), np.nan, dtype=np.float32)
        depth_std = np.full((n,), np.nan, dtype=np.float32)
        depth_p05 = np.full((n,), np.nan, dtype=np.float32)
        depth_p95 = np.full((n,), np.nan, dtype=np.float32)

        # Emit process_frames stage
        _emit_stage(
            rs_path=args.rs_path,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            stage="process_frames",
        )

        for start in range(0, n, batch_size):
            t_batch_start = time.perf_counter()
            batch_ids = frame_indices[start : start + batch_size]
            
            # Load frames
            t_load_start = time.perf_counter()
            frames = [frame_manager.get(i) for i in batch_ids]
            t_load = time.perf_counter() - t_load_start
            
            # Preprocess
            t_prep_start = time.perf_counter()
            inp = _prep_batch_rgb_uint8(frames, input_size=input_size, frames_are_bgr=bool(args.frames_bgr))
            t_prep = time.perf_counter() - t_prep_start
            
            # Triton inference
            t_infer_start = time.perf_counter()
            try:
                res = client.infer(
                    model_name=str(args.triton_model_name),
                    model_version=str(args.triton_model_version) if args.triton_model_version else None,
                    input_name=str(args.triton_input_name),
                    input_tensor=inp,
                    output_name=str(args.triton_output_name),
                    datatype=str(args.triton_datatype),
                )
            except Exception as e:
                raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e
            t_infer = time.perf_counter() - t_infer_start

            out = np.asarray(res.output, dtype=np.float32)
            # Expect (B,1,h,w) or (B,h,w)
            if out.ndim == 4 and out.shape[1] == 1:
                out = out[:, 0, :, :]
            if out.ndim != 3:
                raise RuntimeError(f"{NAME} | Triton output has invalid shape: {out.shape}")
            if out.shape[0] != len(batch_ids):
                raise RuntimeError(f"{NAME} | Triton output batch mismatch: got {out.shape[0]} expected {len(batch_ids)}")

            # Post-process (оптимизированная версия с векторизацией)
            t_post_start = time.perf_counter()
            batch_size_actual = out.shape[0]
            
            # Resize для каждого кадра (cv2 не поддерживает batch resize напрямую)
            resized_maps = []
            for i_local in range(batch_size_actual):
                m = out[i_local]
                dm = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                if not np.isfinite(dm).any():
                    raise RuntimeError(f"{NAME} | invalid depth map produced (NaN/empty) at frame_idx={batch_ids[i_local]}")
                resized_maps.append(dm)
            
            # Конвертируем в numpy array для векторизованных операций
            resized_batch = np.array(resized_maps, dtype=np.float32)  # (B, out_h, out_w)
            
            # Векторизованное вычисление mean и std для всего батча
            depth_maps_flat = resized_batch.reshape(batch_size_actual, -1)  # (B, out_h * out_w)
            means = np.mean(depth_maps_flat, axis=1, dtype=np.float32)  # (B,)
            stds = np.std(depth_maps_flat, axis=1, dtype=np.float32)  # (B,)
            
            # Вычисление percentiles (нужно для каждого кадра отдельно)
            finite_mask = np.isfinite(depth_maps_flat)  # (B, out_h * out_w)
            for i_local in range(batch_size_actual):
                vv = depth_maps_flat[i_local][finite_mask[i_local]]
                if vv.size:
                    depth_p05[start + i_local] = float(np.percentile(vv, 5))
                    depth_p95[start + i_local] = float(np.percentile(vv, 95))
            
            # Сохраняем результаты векторизованно
            global_indices = np.arange(start, start + batch_size_actual, dtype=np.int32)
            depth_maps[global_indices] = resized_batch
            depth_mean[global_indices] = means
            depth_std[global_indices] = stds
            
            t_post = time.perf_counter() - t_post_start
            
            t_batch_total = time.perf_counter() - t_batch_start
            if len(batch_ids) > 0:
                ms_per_frame = (t_batch_total * 1000.0) / len(batch_ids)
                LOGGER.debug(
                    f"{NAME} | batch[{start}:{start+len(batch_ids)}] | "
                    f"load={t_load*1000:.1f}ms prep={t_prep*1000:.1f}ms "
                    f"infer={t_infer*1000:.1f}ms post={t_post*1000:.1f}ms "
                    f"total={t_batch_total*1000:.1f}ms ({ms_per_frame:.1f}ms/frame)"
                )

            processed = min(start + batch_size, n)
            if processed % max(1, n // 15) == 0 or processed == n:
                _emit_progress(
                    rs_path=args.rs_path,
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=processed,
                    total=n,
                    stage="process_frames",
                )
    finally:
        timings["depth_inference_total"] = float(time.perf_counter() - t_infer_start)
        # Always close frame manager and free GPU memory
        try:
            frame_manager.close()
        except Exception:
            pass

    # Emit post_process stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="post_process",
    )

    # -------------------------
    # Backend-friendly proxies (Audit v3)
    # -------------------------
    # Normalized maps (robust per-frame scale using p05/p95) + proxy scores.
    # MiDaS depth is relative; normalized maps are convenient for backend visualization and downstream heuristics.
    eps = np.float32(1e-6)
    denom = (depth_p95 - depth_p05).astype(np.float32)
    denom = np.where(np.isfinite(denom) & (denom > 0), denom, eps).astype(np.float32)
    depth_maps_norm = ((depth_maps - depth_p05[:, None, None]) / denom[:, None, None]).astype(np.float32)
    depth_maps_norm = np.clip(depth_maps_norm, 0.0, 1.0).astype(np.float32)

    depth_range_robust = (depth_p95 - depth_p05).astype(np.float32)  # (N,)
    foreground_background_separation_proxy = (depth_range_robust / (depth_std.astype(np.float32) + eps)).astype(np.float32)

    # Complexity proxy: mean absolute gradient magnitude on normalized maps.
    gx = np.abs(np.diff(depth_maps_norm, axis=2))  # (N,H,W-1)
    gy = np.abs(np.diff(depth_maps_norm, axis=1))  # (N,H-1,W)
    depth_complexity_score = (0.5 * (gx.mean(axis=(1, 2)) + gy.mean(axis=(1, 2)))).astype(np.float32)

    # Preview maps for backend: pick up to K maps uniformly over sampled timeline.
    preview_k = 10
    n_prev = int(min(preview_k, depth_maps.shape[0]))
    if n_prev <= 0:
        raise RuntimeError(f"{NAME} | internal error: n_prev<=0 with N={depth_maps.shape[0]}")
    if n_prev == depth_maps.shape[0]:
        sel = np.arange(depth_maps.shape[0], dtype=np.int64)
    else:
        sel = np.unique(np.round(np.linspace(0, depth_maps.shape[0] - 1, n_prev)).astype(np.int64))
        # If rounding caused fewer unique indices, pad deterministically.
        if sel.size < n_prev:
            missing = n_prev - int(sel.size)
            tail = np.arange(depth_maps.shape[0] - 1, -1, -1, dtype=np.int64)
            for t in tail:
                if int(t) not in set(map(int, sel.tolist())):
                    sel = np.append(sel, t)
                    missing -= 1
                    if missing <= 0:
                        break
            sel = np.sort(sel.astype(np.int64))

    preview_frame_indices = np.asarray(frame_indices, dtype=np.int32)[sel]
    preview_times_s = times_s.astype(np.float32)[sel]
    preview_depth_maps = depth_maps[sel].astype(np.float32, copy=False)
    preview_depth_maps_norm = depth_maps_norm[sel].astype(np.float32, copy=False)

    out_path = os.path.join(core_dir, ARTIFACT_FILENAME)
    created_at = datetime.utcnow().isoformat()

    # Convert timings to milliseconds for meta
    timings["total"] = float(time.perf_counter() - t_total_start)
    stage_timings_ms: Dict[str, float] = {k: float(v) * 1000.0 for k, v in timings.items()}

    # Log stage timings for profiling
    LOGGER.info(f"{NAME} | stage timings (ms): {', '.join([f'{k}={v:.1f}' for k, v in sorted(stage_timings_ms.items())])}")

    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "status": "ok",
        "empty_reason": None,
        "model_name": str(args.triton_model_name),
        "total_frames": int(total_frames),
        "out_width": int(args.out_width),
        "out_height": int(args.out_height),
        "batch_size": int(batch_size),
        "runtime": "triton-gpu",
        "device": "cuda",
        "triton_preprocess_preset": str(args.triton_preprocess_preset),
        "stage_timings_ms": stage_timings_ms,
        # Backend contract
        "backend_proxy_version": "core_depth_midas_backend_proxy_v1",
        "preview_k": int(n_prev),
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
        meta_out[k] = meta.get(k)
    # Baseline: dataprocessor_version must always be present (production should pass the real version).
    meta_out["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

    # PR-3: model system baseline
    # Prefer ModelManager-provided identity when --triton-model-spec is used.
    models_used_list: List[Dict[str, Any]] = []
    if isinstance(mm_entry, dict) and isinstance(mm_entry.get("models_used_entry"), dict):
        models_used_list = [mm_entry["models_used_entry"]]
        meta_out["triton_model_spec"] = str(args.triton_model_spec)
        meta_out["triton_model_name"] = str(args.triton_model_name)
    else:
        models_used_list = [
            model_used(
                model_name=str(args.triton_model_name),
                model_version=str(args.model_version or "unknown"),
                weights_digest=str(args.weights_digest or "unknown"),
                runtime="triton-gpu",
                engine="onnx",
                precision=str(args.precision or "unknown"),
                device="cuda",
            )
        ]
    meta_out = apply_models_meta(meta_out, models_used=models_used_list)
    rp_before = _resource_profile_snapshot()
    if isinstance(rp_before, dict) and rp_before:
        meta_out["resource_profile_before"] = dict(rp_before)

    _atomic_save_npz(
        out_path,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        times_s=times_s.astype(np.float32),
        depth_maps=depth_maps,  # shape (N, out_h, out_w), dtype float32
        depth_maps_norm=depth_maps_norm,  # (N,out_h,out_w) float32 in [0,1]
        depth_mean=depth_mean.astype(np.float32),
        depth_std=depth_std.astype(np.float32),
        depth_p05=depth_p05.astype(np.float32),
        depth_p95=depth_p95.astype(np.float32),
        depth_range_robust=depth_range_robust.astype(np.float32),
        depth_complexity_score=depth_complexity_score.astype(np.float32),
        foreground_background_separation_proxy=foreground_background_separation_proxy.astype(np.float32),
        preview_frame_indices=preview_frame_indices.astype(np.int32),
        preview_times_s=preview_times_s.astype(np.float32),
        preview_depth_maps=preview_depth_maps.astype(np.float32, copy=False),
        preview_depth_maps_norm=preview_depth_maps_norm.astype(np.float32, copy=False),
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    ok, issues, _ = validate_npz(
        out_path,
        required_meta_keys=[
            "producer",
            "producer_version",
            "schema_version",
            "created_at",
            "platform_id",
            "video_id",
            "run_id",
            "config_hash",
            "sampling_policy_version",
            "dataprocessor_version",
            "status",
            "empty_reason",
            "models_used",
            "model_signature",
        ],
    )
    if not ok:
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError(f"{NAME} | saved artifact failed validation: {issues}")

    # Show relative path for cleaner output
    rel_out_path = os.path.relpath(out_path, os.getcwd()) if os.path.exists(out_path) else out_path
    LOGGER.info(f"{NAME} | main | Saved NPZ artifact: {rel_out_path} | created_at={created_at}")

    # Emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )


if __name__ == "__main__":
    main()
