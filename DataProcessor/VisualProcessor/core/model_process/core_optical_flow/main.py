#!/usr/bin/env python3
"""
core_optical_flow

Tier-0 core provider: optical flow motion curve via Triton.

Contract:
- Segmenter provides `metadata["core_optical_flow"]["frame_indices"]` (union-domain indices).
- No sampling fallback is allowed.
- Frames from FrameManager.get(idx) are RGB uint8 (HxWx3).

Output:
- <rs_path>/core_optical_flow/flow.npz
  Keys:
    - frame_indices: int32 (N,)
    - times_s: float32 (N,)                 # union_timestamps_sec[frame_indices] (strict time axis)
    - motion_norm_per_sec_mean: float32 (N,)  # mean flow magnitude / dt / max(H,W); 0 for first frame
    - dt_seconds: float32 (N,)                # NaN for first frame
    - meta: object(dict)
"""

from __future__ import annotations

import sys
from pathlib import Path
_vp = Path(__file__).resolve().parent
for _ in range(3):
    _vp = _vp.parent
sys.path.insert(0, str(_vp))

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

_root = str(_vp.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

NAME = "core_optical_flow"
VERSION = "2.2"
SCHEMA_VERSION = "core_optical_flow_npz_v3"
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


def _compute_affine_camera_from_flow(
    *,
    flow_dx: np.ndarray,
    flow_dy: np.ndarray,
    dt: float,
    norm: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute compact camera-motion proxies from a single flow map.
    Returns: (cam_scale, cam_rotation_rad, cam_tx_per_sec_norm, cam_ty_per_sec_norm, cam_shake_std_norm, bg_ratio)
    Deterministic (grid sampling).
    """
    try:
        import cv2  # type: ignore
        import math

        dx = np.asarray(flow_dx, dtype=np.float32)
        dy = np.asarray(flow_dy, dtype=np.float32)
        mag = np.hypot(dx, dy).astype(np.float32)

        # background mask: low-motion pixels (robust threshold)
        vv = mag[np.isfinite(mag)]
        if vv.size <= 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        thr = float(np.percentile(vv, 40))
        bg = mag <= max(thr, 1e-6)
        bg_ratio = float(np.mean(bg))

        # shakiness: std of background magnitude (per-sec, normalized)
        bg_mag = mag[bg]
        if bg_mag.size > 0 and dt > 0 and norm > 0:
            cam_shake_std_norm = float(np.std(bg_mag) / max(dt, 1e-6) / max(norm, 1.0))
        else:
            cam_shake_std_norm = float("nan")

        # downsample to a moderate size for affine estimation
        h, w = dx.shape
        tgt = 64
        if h != tgt or w != tgt:
            dx_s = cv2.resize(dx, (tgt, tgt), interpolation=cv2.INTER_AREA).astype(np.float32)
            dy_s = cv2.resize(dy, (tgt, tgt), interpolation=cv2.INTER_AREA).astype(np.float32)
            mag_s = cv2.resize(mag, (tgt, tgt), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            dx_s, dy_s, mag_s = dx, dy, mag

        # deterministic grid sampling
        ys, xs = np.mgrid[0:tgt, 0:tgt]
        stride = 4
        xs_g = xs[::stride, ::stride].reshape(-1).astype(np.float32)
        ys_g = ys[::stride, ::stride].reshape(-1).astype(np.float32)
        dx_g = dx_s[::stride, ::stride].reshape(-1).astype(np.float32)
        dy_g = dy_s[::stride, ::stride].reshape(-1).astype(np.float32)
        mag_g = mag_s[::stride, ::stride].reshape(-1).astype(np.float32)

        # background points selection on grid
        vvg = mag_g[np.isfinite(mag_g)]
        if vvg.size <= 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), cam_shake_std_norm, bg_ratio
        thr_g = float(np.percentile(vvg, 40))
        m_bg = np.isfinite(dx_g) & np.isfinite(dy_g) & np.isfinite(mag_g) & (mag_g <= max(thr_g, 1e-6))
        if int(np.sum(m_bg)) < 12:
            # fallback to all finite points
            m_bg = np.isfinite(dx_g) & np.isfinite(dy_g)
        if int(np.sum(m_bg)) < 12:
            return float("nan"), float("nan"), float("nan"), float("nan"), cam_shake_std_norm, bg_ratio

        pts = np.stack([xs_g[m_bg], ys_g[m_bg]], axis=1).astype(np.float32)
        disp = np.stack([dx_g[m_bg], dy_g[m_bg]], axis=1).astype(np.float32)
        src = pts
        dst = pts + disp

        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or not np.isfinite(M).all():
            return float("nan"), float("nan"), float("nan"), float("nan"), cam_shake_std_norm, bg_ratio

        a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
        cam_scale = float(math.sqrt(a * a + b * b))
        cam_rotation = float(math.atan2(c, a))
        cam_tx_per_sec_norm = float(tx / max(dt, 1e-6) / max(norm, 1.0))
        cam_ty_per_sec_norm = float(ty / max(dt, 1e-6) / max(norm, 1.0))
        return cam_scale, cam_rotation, cam_tx_per_sec_norm, cam_ty_per_sec_norm, cam_shake_std_norm, bg_ratio
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")


def _compute_direction_stats(*, flow_dx: np.ndarray, flow_dy: np.ndarray) -> Tuple[float, float, float]:
    """
    Direction stats computed from flow vectors:
    returns (sin_mean, cos_mean, dispersion) with magnitude-weighted averaging, NaN if undefined.
    """
    dx = np.asarray(flow_dx, dtype=np.float32)
    dy = np.asarray(flow_dy, dtype=np.float32)
    mag = np.hypot(dx, dy).astype(np.float32)
    vv = mag[np.isfinite(mag)]
    if vv.size <= 0:
        return float("nan"), float("nan"), float("nan")
    thr = max(float(np.percentile(vv, 50)), 1e-4)
    m = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag) & (mag > thr)
    if int(np.sum(m)) < 16:
        return float("nan"), float("nan"), float("nan")
    w = mag[m]
    denom = float(np.sum(w)) + 1e-9
    sinv = dy[m] / (mag[m] + 1e-9)
    cosv = dx[m] / (mag[m] + 1e-9)
    sin_mean = float(np.sum(w * sinv) / denom)
    cos_mean = float(np.sum(w * cosv) / denom)
    R = float(np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean))
    dispersion = float(1.0 - R)
    return sin_mean, cos_mean, dispersion


def _compute_divergence_consistency(*, flow_dx: np.ndarray, flow_dy: np.ndarray) -> Tuple[float, float]:
    """
    Lightweight divergence proxy on a downsampled grid.
    Returns (div_abs_mean, flow_consistency=1/(1+div_abs_mean)).
    """
    try:
        import cv2  # type: ignore

        dx = np.asarray(flow_dx, dtype=np.float32)
        dy = np.asarray(flow_dy, dtype=np.float32)
        tgt = 64
        if dx.shape != (tgt, tgt):
            dx = cv2.resize(dx, (tgt, tgt), interpolation=cv2.INTER_AREA).astype(np.float32)
            dy = cv2.resize(dy, (tgt, tgt), interpolation=cv2.INTER_AREA).astype(np.float32)
        # div = d/dx dx + d/dy dy
        d_dx = np.gradient(dx, axis=1)
        d_dy = np.gradient(dy, axis=0)
        div = d_dx + d_dy
        v = div[np.isfinite(div)]
        if v.size <= 0:
            return float("nan"), float("nan")
        div_abs_mean = float(np.mean(np.abs(v)))
        flow_consistency = float(1.0 / (1.0 + div_abs_mean))
        return div_abs_mean, flow_consistency
    except Exception:
        return float("nan"), float("nan")

def _atomic_save_npz(out_path: str, **kwargs) -> None:
    """
    Atomic NPZ save:
    write to tmp file in same dir, then os.replace().
    IMPORTANT: suffix must be '.npz' so numpy does not append another extension.
    """
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


def _require_frame_indices(meta: dict, name: str) -> List[int]:
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


def _get_union_timestamps_sec(frame_manager: FrameManager) -> Optional[np.ndarray]:
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        return None
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        return None
    try:
        return np.asarray(ts, dtype=np.float32)
    except Exception:
        return None


def _require_union_times_s(frame_manager: FrameManager, frame_indices: List[int]) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth.
    No-fallback: if missing/invalid -> error.
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError(f"{NAME} | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError(f"{NAME} | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)
    fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
    if fi.size == 0:
        raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
    if int(np.max(fi)) >= int(uts.shape[0]):
        raise RuntimeError(f"{NAME} | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[fi]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError(f"{NAME} | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _preset_to_input_size(preset: str) -> int:
    p = str(preset or "").strip().lower()
    if p in ("raft_256", "256"):
        return 256
    if p in ("raft_384", "384"):
        return 384
    if p in ("raft_512", "512"):
        return 512
    raise ValueError(f"{NAME} | unknown triton_preprocess_preset: {preset!r}")


def _prep_batch_rgb_uint8(frames: List[np.ndarray], *, input_size: int) -> np.ndarray:
    """
    Minimal client-side formatting for Triton (NOT full preprocessing):
    - resize to (S,S)
    - keep UINT8 NHWC (baseline GPU contract)

    Full preprocessing (normalize/layout conversion to model FP32 NCHW) lives in Triton ensemble.
    """
    import cv2  # type: ignore

    s = int(input_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid input_size={input_size}")
    out: List[np.ndarray] = []
    for fr in frames:
        fr_r = cv2.resize(fr, (s, s), interpolation=cv2.INTER_AREA)
        out.append(np.asarray(fr_r, dtype=np.uint8))
    if not out:
        return np.zeros((0, s, s, 3), dtype=np.uint8)
    return np.stack(out, axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="core_optical_flow (Triton) motion curve extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    # Triton-only policy (prod): local torch engine is removed.
    parser.add_argument("--runtime", type=str, default="triton", choices=["triton"], help="Runtime (prod: triton only)")
    parser.add_argument("--triton-http-url", type=str, default=None)
    # Preferred: resolve Triton params via ModelManager specs (recommended; overrides explicit triton_* args when provided).
    parser.add_argument("--triton-model-spec", type=str, default=None, help="dp_models spec name (e.g., raft_256_triton)")
    parser.add_argument("--triton-model-name", type=str, default=None)
    parser.add_argument("--triton-model-version", type=str, default=None)
    parser.add_argument("--triton-input0-name", type=str, default="INPUT0__0")
    parser.add_argument("--triton-input1-name", type=str, default="INPUT1__0")
    parser.add_argument("--triton-output-name", type=str, default="OUTPUT__0")
    # Triton ensemble expects UINT8 NHWC inputs.
    parser.add_argument("--triton-datatype", type=str, default="UINT8")
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="raft_256",
        choices=["raft_256", "raft_384", "raft_512"],
        help="Input preset (square size) for Triton optical-flow model.",
    )
    parser.add_argument("--model-version", type=str, default="unknown")
    parser.add_argument("--weights-digest", type=str, default="unknown")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of frame-pairs per Triton request (scheduler-controlled). For unit-cost set to 1.",
    )
    args = parser.parse_args()

    runtime = str(args.runtime or "triton").strip().lower()
    if runtime != "triton":
        raise RuntimeError(f"{NAME} | runtime must be triton (no-fallback), got: {runtime}")

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

    frame_indices = _require_frame_indices(meta, NAME)
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")

    if len(frame_indices) < 2:
        raise RuntimeError(f"{NAME} | frame_indices must contain at least 2 frames (no-fallback)")

    # Triton client (repo-local)
    from dp_triton import TritonHttpClient, TritonError  # type: ignore

    mm_entry = None
    client: Any = None
    if args.triton_model_spec:
        mm_entry = _load_triton_spec_via_model_manager(str(args.triton_model_spec))
        client = mm_entry["client"]
        rp = mm_entry["rp"]
        args.triton_http_url = str(rp.get("triton_http_url") or args.triton_http_url or "")
        args.triton_model_name = str(rp.get("triton_model_name") or args.triton_model_name or "")
        args.triton_model_version = str(rp.get("triton_model_version") or "") or None
        args.triton_input0_name = str(rp.get("triton_input0_name") or args.triton_input0_name)
        args.triton_input1_name = str(rp.get("triton_input1_name") or args.triton_input1_name)
        args.triton_output_name = str(rp.get("triton_output_name") or args.triton_output_name)
        args.triton_datatype = str(rp.get("triton_input_datatype") or args.triton_datatype)
    else:
        if not args.triton_http_url or not str(args.triton_http_url).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec (no-fallback)")
        if not args.triton_model_name or not str(args.triton_model_name).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec (no-fallback)")
        _tmo = float(os.environ.get("DP_TRITON_HTTP_TIMEOUT_SEC", "120.0"))
        client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=_tmo)
    if client is None:
        raise RuntimeError(f"{NAME} | Triton client not initialized")
    if not client.ready():
        raise TritonError(f"{NAME} | Triton is not ready at {args.triton_http_url}", error_code="triton_unavailable")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    # Strict time axis (no-fallback)
    times_s = _require_union_times_s(frame_manager, frame_indices)
    idx_np = np.asarray(frame_indices, dtype=np.int32)
    n = int(idx_np.size)

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

    dt_seconds = np.full((n,), np.nan, dtype=np.float32)
    motion_norm_per_sec = np.full((n,), np.nan, dtype=np.float32)

    # Audit v3 (v3 schema): compact per-frame flow/camera stats for downstream models
    flow_mag_std_per_sec_norm = np.full((n,), np.nan, dtype=np.float32)
    flow_mag_p95_per_sec_norm = np.full((n,), np.nan, dtype=np.float32)
    flow_dx_mean_per_sec_norm = np.full((n,), np.nan, dtype=np.float32)
    flow_dy_mean_per_sec_norm = np.full((n,), np.nan, dtype=np.float32)
    flow_dir_sin_mean = np.full((n,), np.nan, dtype=np.float32)
    flow_dir_cos_mean = np.full((n,), np.nan, dtype=np.float32)
    flow_dir_dispersion = np.full((n,), np.nan, dtype=np.float32)
    flow_div_abs_mean = np.full((n,), np.nan, dtype=np.float32)
    flow_consistency = np.full((n,), np.nan, dtype=np.float32)
    cam_affine_scale = np.full((n,), np.nan, dtype=np.float32)
    cam_affine_rotation = np.full((n,), np.nan, dtype=np.float32)
    cam_tx_per_sec_norm = np.full((n,), np.nan, dtype=np.float32)
    cam_ty_per_sec_norm = np.full((n,), np.nan, dtype=np.float32)
    cam_shake_std_norm = np.full((n,), np.nan, dtype=np.float32)
    bg_ratio = np.full((n,), np.nan, dtype=np.float32)

    # Backend preview (Audit v3): store K downsampled flow magnitude maps (normalized to [0,1])
    preview_k = 10
    n_pairs = max(0, n - 1)
    n_prev = int(min(preview_k, n_pairs))
    if n_prev <= 0:
        raise RuntimeError(f"{NAME} | requires at least 2 frames (>=1 pair); got n={n}")
    if n_prev == n_pairs:
        preview_pair_pos = np.arange(1, n, dtype=np.int32)  # positions in [1..n-1]
    else:
        preview_pair_pos = np.unique(np.round(np.linspace(1, n - 1, n_prev)).astype(np.int32))
        if preview_pair_pos.size < n_prev:
            missing = n_prev - int(preview_pair_pos.size)
            tail = np.arange(n - 1, 0, -1, dtype=np.int32)
            seen = set(map(int, preview_pair_pos.tolist()))
            for t in tail:
                if int(t) not in seen:
                    preview_pair_pos = np.append(preview_pair_pos, t)
                    seen.add(int(t))
                    missing -= 1
                    if missing <= 0:
                        break
            preview_pair_pos = np.sort(preview_pair_pos.astype(np.int32))
    preview_slot_by_pos = {int(p): int(i) for i, p in enumerate(preview_pair_pos.tolist())}

    preview_prev_frame_indices = idx_np[preview_pair_pos - 1].astype(np.int32, copy=False)
    preview_cur_frame_indices = idx_np[preview_pair_pos].astype(np.int32, copy=False)
    preview_prev_times_s = times_s[preview_pair_pos - 1].astype(np.float32, copy=False)
    preview_cur_times_s = times_s[preview_pair_pos].astype(np.float32, copy=False)

    preview_map_h = 64
    preview_map_w = 64
    preview_flow_mag_map_norm = np.full((int(preview_pair_pos.size), preview_map_h, preview_map_w), np.nan, dtype=np.float32)

    # Flow inference stage
    t_infer_start = time.perf_counter()

    # Emit process_frames stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="process_frames",
    )

    try:
        input_size = _preset_to_input_size(str(args.triton_preprocess_preset))

        # First element is defined but has no dt
        motion_norm_per_sec[0] = 0.0
        # For compact per-frame stats, frame 0 has no prev pair -> NaN by design.

        # Pre-compute dt from strict time axis (no-fallback)
        if times_s.size >= 2:
            dt_seconds[1:] = np.maximum(np.diff(times_s), 1e-6).astype(np.float32)

        # Batched pair inference:
        # pairs are (frame[i-1], frame[i]) for i=1..N-1 and we store result at position i.
        max_bs = max(1, int(args.batch_size))
        i = 1
        while i < n:
            t_batch_start = time.perf_counter()
            j = min(n, i + max_bs)
            
            # Load frames
            t_load_start = time.perf_counter()
            prev_frames: List[np.ndarray] = []
            cur_frames: List[np.ndarray] = []
            dts: List[float] = []
            for k in range(i, j):
                prev_frames.append(frame_manager.get(int(idx_np[k - 1])))
                cur_frames.append(frame_manager.get(int(idx_np[k])))
                dts.append(float(dt_seconds[k]))
            t_load = time.perf_counter() - t_load_start

            # Preprocess
            t_prep_start = time.perf_counter()
            inp0 = _prep_batch_rgb_uint8(prev_frames, input_size=input_size)
            inp1 = _prep_batch_rgb_uint8(cur_frames, input_size=input_size)
            t_prep = time.perf_counter() - t_prep_start
            
            # Triton inference
            t_infer_start = time.perf_counter()
            try:
                out0 = client.infer_two_inputs(
                    model_name=str(args.triton_model_name),
                    model_version=str(args.triton_model_version) if args.triton_model_version else None,
                    input0_name=str(args.triton_input0_name),
                    input0_tensor=inp0,
                    input1_name=str(args.triton_input1_name),
                    input1_tensor=inp1,
                    output_name=str(args.triton_output_name),
                    datatype=str(args.triton_datatype),
                )
            except AttributeError:
                raise RuntimeError(
                    f"{NAME} | dp_triton client missing infer_two_inputs(). "
                    f"Please update dp_triton to support 2-input models."
                )
            except Exception as e:
                raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e
            t_infer = time.perf_counter() - t_infer_start

            flow = np.asarray(out0.output, dtype=np.float32)
            # Expect (B,2,H,W)
            if flow.ndim != 4 or flow.shape[1] != 2:
                raise RuntimeError(f"{NAME} | Triton output has invalid shape: {flow.shape}")
            if flow.shape[0] != inp0.shape[0]:
                raise RuntimeError(f"{NAME} | Triton output batch mismatch: outB={flow.shape[0]} inB={inp0.shape[0]}")

            # Post-process (compute motion norm)
            t_post_start = time.perf_counter()
            # Compute mean magnitude per pair (векторизовано для всего батча)
            # mag: (B,H,W) - используем np.hypot для более эффективного вычисления
            mag = np.hypot(flow[:, 0], flow[:, 1])  # Более эффективно чем sqrt(square + square)
            mag_mean = mag.reshape(mag.shape[0], -1).mean(axis=1).astype(np.float32)
            mag_std = mag.reshape(mag.shape[0], -1).std(axis=1).astype(np.float32)
            mag_p95 = np.quantile(mag.reshape(mag.shape[0], -1), 0.95, axis=1).astype(np.float32)
            dx_mean = flow[:, 0].reshape(flow.shape[0], -1).mean(axis=1).astype(np.float32)
            dy_mean = flow[:, 1].reshape(flow.shape[0], -1).mean(axis=1).astype(np.float32)
            norm = float(max(flow.shape[2], flow.shape[3], 1))
            dts_np = np.asarray(dts, dtype=np.float32)
            vals = (mag_mean / np.maximum(dts_np, 1e-6)) / float(max(norm, 1.0))
            mag_std_vals = (mag_std / np.maximum(dts_np, 1e-6)) / float(max(norm, 1.0))
            mag_p95_vals = (mag_p95 / np.maximum(dts_np, 1e-6)) / float(max(norm, 1.0))
            dx_mean_vals = (dx_mean / np.maximum(dts_np, 1e-6)) / float(max(norm, 1.0))
            dy_mean_vals = (dy_mean / np.maximum(dts_np, 1e-6)) / float(max(norm, 1.0))

            # Backend preview maps for selected pairs (best-effort, deterministic).
            # NOTE: Only K maps are kept, downsampled to (64,64) and normalized to [0,1] per map.
            try:
                import cv2  # type: ignore

                for i_local in range(int(mag.shape[0])):
                    pos = int(i + i_local)  # position in [1..n-1]
                    slot = preview_slot_by_pos.get(pos)
                    if slot is None:
                        continue
                    mm = cv2.resize(mag[i_local].astype(np.float32), (preview_map_w, preview_map_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                    vv = mm[np.isfinite(mm)]
                    if vv.size <= 0:
                        continue
                    p05 = float(np.percentile(vv, 5))
                    p95 = float(np.percentile(vv, 95))
                    denom = max(float(p95 - p05), 1e-6)
                    mmn = (mm - p05) / denom
                    mmn = np.clip(mmn, 0.0, 1.0).astype(np.float32)
                    preview_flow_mag_map_norm[int(slot)] = mmn
            except Exception:
                pass

            if not np.all(np.isfinite(vals)):
                bad = np.where(~np.isfinite(vals))[0]
                raise RuntimeError(f"{NAME} | invalid motion value(s) at batch offsets: {bad.tolist()}")
            motion_norm_per_sec[i:j] = vals.astype(np.float32)
            flow_mag_std_per_sec_norm[i:j] = mag_std_vals.astype(np.float32)
            flow_mag_p95_per_sec_norm[i:j] = mag_p95_vals.astype(np.float32)
            flow_dx_mean_per_sec_norm[i:j] = dx_mean_vals.astype(np.float32)
            flow_dy_mean_per_sec_norm[i:j] = dy_mean_vals.astype(np.float32)

            # Direction / divergence / camera motion: per-item (batch size small, computed on demand)
            for i_local in range(int(flow.shape[0])):
                pos = int(i + i_local)  # 1..N-1
                dtv = float(dts_np[int(i_local)])
                dx_map = flow[int(i_local), 0]
                dy_map = flow[int(i_local), 1]
                s, c, disp = _compute_direction_stats(flow_dx=dx_map, flow_dy=dy_map)
                flow_dir_sin_mean[pos] = np.float32(s)
                flow_dir_cos_mean[pos] = np.float32(c)
                flow_dir_dispersion[pos] = np.float32(disp)
                div_abs, cons = _compute_divergence_consistency(flow_dx=dx_map, flow_dy=dy_map)
                flow_div_abs_mean[pos] = np.float32(div_abs)
                flow_consistency[pos] = np.float32(cons)
                sc, rot, txv, tyv, shake, bgr = _compute_affine_camera_from_flow(
                    flow_dx=dx_map, flow_dy=dy_map, dt=dtv, norm=float(max(norm, 1.0))
                )
                cam_affine_scale[pos] = np.float32(sc)
                cam_affine_rotation[pos] = np.float32(rot)
                cam_tx_per_sec_norm[pos] = np.float32(txv)
                cam_ty_per_sec_norm[pos] = np.float32(tyv)
                cam_shake_std_norm[pos] = np.float32(shake)
                bg_ratio[pos] = np.float32(bgr)
            t_post = time.perf_counter() - t_post_start
            
            t_batch_total = time.perf_counter() - t_batch_start
            if len(prev_frames) > 0:
                ms_per_pair = (t_batch_total * 1000.0) / len(prev_frames)
                LOGGER.debug(
                    f"{NAME} | batch[{i}:{j}] | "
                    f"load={t_load*1000:.1f}ms prep={t_prep*1000:.1f}ms "
                    f"infer={t_infer*1000:.1f}ms post={t_post*1000:.1f}ms "
                    f"total={t_batch_total*1000:.1f}ms ({ms_per_pair:.1f}ms/pair)"
                )

            # Emit progress (at least ~10-15 updates per video)
            processed = j
            if processed % max(1, (n - 1) // 15) == 0 or processed == n:
                _emit_progress(
                    rs_path=args.rs_path,
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=processed - 1,  # processed pairs (i starts from 1)
                    total=n - 1,  # total pairs (n-1 pairs for n frames)
                    stage="process_frames",
                )
                LOGGER.info(f"{NAME} | processed {processed}/{n}")
            i = j
    finally:
        timings["flow_inference_total"] = float(time.perf_counter() - t_infer_start)
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

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flow.npz")

    # Saving stage
    t_save_start = time.perf_counter()
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    
    # Initialize meta_out
    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "status": "ok",
        "empty_reason": None,
    }
    
    for k in required_run_keys:
        meta_out[k] = meta.get(k)
    # Baseline: dataprocessor_version must be present (prod: real version; baseline may be 'unknown').
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

    # Backend contract metadata
    meta_out["backend_proxy_version"] = "core_optical_flow_backend_proxy_v1"
    meta_out["preview_k"] = int(preview_pair_pos.size)
    meta_out["preview_map_size"] = [int(preview_map_h), int(preview_map_w)]

    # Convert timings to milliseconds for meta (before saving)
    timings["saving"] = 0.0  # Will be measured after save
    timings["total"] = float(time.perf_counter() - t_total_start)
    stage_timings_ms: Dict[str, float] = {k: float(v) * 1000.0 for k, v in timings.items()}
    meta_out["stage_timings_ms"] = stage_timings_ms
    rp_before = _resource_profile_snapshot()
    if isinstance(rp_before, dict) and rp_before:
        meta_out["resource_profile_before"] = dict(rp_before)

    _atomic_save_npz(
        out_path,
        frame_indices=idx_np.astype(np.int32),
        times_s=times_s.astype(np.float32),
        motion_norm_per_sec_mean=motion_norm_per_sec.astype(np.float32),
        dt_seconds=dt_seconds.astype(np.float32),
        flow_mag_std_per_sec_norm=flow_mag_std_per_sec_norm.astype(np.float32),
        flow_mag_p95_per_sec_norm=flow_mag_p95_per_sec_norm.astype(np.float32),
        flow_dx_mean_per_sec_norm=flow_dx_mean_per_sec_norm.astype(np.float32),
        flow_dy_mean_per_sec_norm=flow_dy_mean_per_sec_norm.astype(np.float32),
        flow_dir_sin_mean=flow_dir_sin_mean.astype(np.float32),
        flow_dir_cos_mean=flow_dir_cos_mean.astype(np.float32),
        flow_dir_dispersion=flow_dir_dispersion.astype(np.float32),
        flow_div_abs_mean=flow_div_abs_mean.astype(np.float32),
        flow_consistency=flow_consistency.astype(np.float32),
        cam_affine_scale=cam_affine_scale.astype(np.float32),
        cam_affine_rotation=cam_affine_rotation.astype(np.float32),
        cam_tx_per_sec_norm=cam_tx_per_sec_norm.astype(np.float32),
        cam_ty_per_sec_norm=cam_ty_per_sec_norm.astype(np.float32),
        cam_shake_std_norm=cam_shake_std_norm.astype(np.float32),
        bg_ratio=bg_ratio.astype(np.float32),
        preview_pair_pos=preview_pair_pos.astype(np.int32),
        preview_prev_frame_indices=preview_prev_frame_indices.astype(np.int32, copy=False),
        preview_cur_frame_indices=preview_cur_frame_indices.astype(np.int32, copy=False),
        preview_prev_times_s=preview_prev_times_s.astype(np.float32, copy=False),
        preview_cur_times_s=preview_cur_times_s.astype(np.float32, copy=False),
        preview_flow_mag_map_norm=preview_flow_mag_map_norm.astype(np.float32, copy=False),
        meta=np.asarray(meta_out, dtype=object),
    )
    timings["saving"] = float(time.perf_counter() - t_save_start)
    stage_timings_ms["saving"] = float(timings["saving"] * 1000.0)
    # Update total timing (includes saving)
    timings["total"] = float(time.perf_counter() - t_total_start)
    stage_timings_ms["total"] = float(timings["total"] * 1000.0)

    # Log stage timings for profiling
    LOGGER.info(f"{NAME} | stage timings (ms): {', '.join([f'{k}={v:.1f}' for k, v in sorted(stage_timings_ms.items())])}")

    # Emit save stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="save",
    )

    # Extra sanity checks beyond validator (shapes + strict time axis presence)
    if times_s.shape != (n,):
        raise RuntimeError(f"{NAME} | internal error: times_s shape mismatch: {times_s.shape} vs (N,) N={n}")
    if motion_norm_per_sec.shape != (n,):
        raise RuntimeError(f"{NAME} | internal error: motion_norm_per_sec shape mismatch: {motion_norm_per_sec.shape} vs (N,) N={n}")
    if dt_seconds.shape != (n,):
        raise RuntimeError(f"{NAME} | internal error: dt_seconds shape mismatch: {dt_seconds.shape} vs (N,) N={n}")

    required_meta_keys = [
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
    ]
    ok, issues, _ = validate_npz(out_path, required_meta_keys=required_meta_keys)
    if not ok:
        try:
            os.remove(out_path)
        except Exception:
            pass
        msgs = "; ".join([f"{i.level}:{i.message}" for i in issues if getattr(i, "level", "") == "error"])
        raise RuntimeError(f"{NAME} | saved artifact failed validation: {msgs}")

    # Emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()


