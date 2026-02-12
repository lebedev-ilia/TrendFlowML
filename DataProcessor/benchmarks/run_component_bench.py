#!/usr/bin/env python3
"""
Component-level benchmark harness with resource monitoring.

This script:
1. Measures system resources (GPU VRAM, RAM, CPU/GPU utilization) BEFORE Triton startup
2. Waits for user to start Triton (manual step)
3. Measures resources AFTER Triton startup
4. Runs component (e.g., core_clip, core_depth_midas, core_optical_flow) with specified units (1 frame, 10 frames, or full video)
5. Monitors resources during component execution
6. Measures resources AFTER component execution
7. Generates HTML, JSON, and table reports with all metrics

Usage:
    python benchmarks/run_component_bench.py \
        --component core_clip \
        --video-path /path/to/video.mp4 \
        --frames-count 1 \
        --triton-http-url http://localhost:8000 \
        [--wait-triton]  # if set, waits for user to start Triton manually

Supported components:
    - core_clip: CLIP embeddings extraction
    - core_depth_midas: Depth estimation via MiDaS
    - core_optical_flow: Optical flow motion curve (requires at least 2 frames)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add DynamicBatch to path for ResourceMonitor
_dynamicbatch_path = Path(__file__).parent.parent.parent / "DynamicBatch"
sys.path.insert(0, str(_dynamicbatch_path))

from dynamicbatch.resource_monitor import ResourceMonitor
from dynamicbatch.system_probe import probe_cpu_mem_mb, probe_gpu_mem_mb

# Import enhanced resource monitor
_benchmarks_path = Path(__file__).parent
sys.path.insert(0, str(_benchmarks_path))

from resource_monitor_enhanced import EnhancedResourceMonitor

# Add VisualProcessor to path for component execution
_visual_processor_path = Path(__file__).parent.parent.parent / "VisualProcessor"
sys.path.insert(0, str(_visual_processor_path))


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""

    timestamp_iso: str
    cpu_mem_total_mb: Optional[int] = None
    cpu_mem_used_mb: Optional[int] = None
    cpu_mem_free_mb: Optional[int] = None
    gpu_mem_total_mb: Optional[int] = None
    gpu_mem_used_mb: Optional[int] = None
    gpu_mem_free_mb: Optional[int] = None
    cpu_util_pct: Optional[float] = None
    gpu_util_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceTimeSeries:
    """Time series data point for resource monitoring."""
    timestamp_iso: str
    elapsed_sec: float
    cpu_util_pct: Optional[float] = None
    cpu_mem_used_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceMetrics:
    """Resource metrics for a benchmark run."""

    before_triton: ResourceSnapshot
    after_triton: ResourceSnapshot
    after_component: ResourceSnapshot
    peaks: Dict[str, Any]  # From ResourceMonitor
    time_series: List[ResourceTimeSeries]  # Time series during component execution
    peak_timestamps: Dict[str, str]  # Timestamps when peaks occurred

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before_triton": self.before_triton.to_dict(),
            "after_triton": self.after_triton.to_dict(),
            "after_component": self.after_component.to_dict(),
            "peaks": self.peaks,
            "time_series": [ts.to_dict() for ts in self.time_series],
            "peak_timestamps": self.peak_timestamps,
        }


def _get_resource_snapshot() -> ResourceSnapshot:
    """Get current resource snapshot."""
    cpu_mem = probe_cpu_mem_mb()
    gpu_mem = probe_gpu_mem_mb()

    # Get CPU/GPU utilization via nvidia-smi (best-effort)
    cpu_util = None
    gpu_util = None

    # CPU utilization via /proc/stat (simple approach)
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("cpu "):
                    # Simple approach: parse but don't calculate delta here
                    # We'll use ResourceMonitor for proper utilization
                    break
    except Exception:
        pass

    # GPU utilization via nvidia-smi
    import shutil

    if shutil.which("nvidia-smi"):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,nounits,noheader",
            ]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if p.returncode == 0 and p.stdout:
                gpu_util = float(p.stdout.strip().splitlines()[0])
        except Exception:
            pass

    return ResourceSnapshot(
        timestamp_iso=datetime.utcnow().isoformat(),
        cpu_mem_total_mb=cpu_mem.total_mb if cpu_mem else None,
        cpu_mem_used_mb=cpu_mem.used_mb if cpu_mem else None,
        cpu_mem_free_mb=cpu_mem.free_mb if cpu_mem else None,
        gpu_mem_total_mb=gpu_mem.total_mb if gpu_mem else None,
        gpu_mem_used_mb=gpu_mem.used_mb if gpu_mem else None,
        gpu_mem_free_mb=gpu_mem.free_mb if gpu_mem else None,
        cpu_util_pct=cpu_util,
        gpu_util_pct=gpu_util,
    )


def _wait_for_triton(triton_url: str, timeout_sec: float = 300.0, check_interval_sec: float = 2.0) -> bool:
    """Wait for Triton to become available."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout_sec
    health_url = f"{triton_url.rstrip('/')}/v2/health/ready"

    print(f"[bench] Waiting for Triton at {triton_url}...")
    start_time = time.time()
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"[bench] Triton is ready!")
                    return True
        except Exception:
            pass
        time.sleep(check_interval_sec)
        elapsed = time.time() - start_time
        if int(elapsed) % 10 < check_interval_sec and elapsed > 0:
            remaining = int(deadline - time.time())
            print(f"[bench] Still waiting for Triton... ({remaining}s remaining)")

    print(f"[bench] ERROR: Triton did not become ready within {timeout_sec}s")
    return False


def _find_existing_frames_dir(video_id: str) -> Optional[str]:
    """Search for existing frames_dir in Segmenter/data."""
    # Segmenter/data is at DataProcessor/Segmenter/data
    # __file__ is DataProcessor/benchmarks/run_component_bench.py
    segmenter_data = Path(__file__).parent.parent / "Segmenter" / "data"
    
    if not segmenter_data.exists():
        return None
    
    # Search for video_id directory in Segmenter/data
    video_dir = segmenter_data / video_id
    if video_dir.exists():
        frames_dir = video_dir / "video"
        metadata_path = frames_dir / "metadata.json"
        if metadata_path.exists():
            # Check if there are batch files
            batch_files = list(frames_dir.glob("batch*.npy"))
            if batch_files:
                return str(frames_dir)
    
    return None


def _ensure_frames_dir(video_path: str, frames_dir: Optional[str], out_base: str) -> str:
    """Ensure frames_dir exists, create if needed via Segmenter."""
    # If explicitly provided and exists, use it
    if frames_dir and os.path.exists(os.path.join(frames_dir, "metadata.json")):
        return frames_dir

    video_id = Path(video_path).stem
    
    # Try to find existing frames_dir in Segmenter/data
    existing_frames_dir = _find_existing_frames_dir(video_id)
    if existing_frames_dir:
        print(f"[bench] Found existing frames_dir: {existing_frames_dir}")
        return existing_frames_dir
    
    # Try output directory
    frames_dir_target = os.path.join(out_base, video_id, "video")
    metadata_path = os.path.join(frames_dir_target, "metadata.json")
    if os.path.exists(metadata_path):
        batch_files = list(Path(frames_dir_target).glob("batch*.npy"))
        if batch_files:
            print(f"[bench] Found existing frames_dir in output: {frames_dir_target}")
            return frames_dir_target

    # Run Segmenter to create frames_dir
    print(f"[bench] Running Segmenter to create frames_dir...")
    # Segmenter is in DataProcessor/Segmenter/segmenter.py
    # __file__ is DataProcessor/benchmarks/run_component_bench.py
    # So we need to go up one level (parent = benchmarks, parent.parent = DataProcessor)
    segmenter_script = Path(__file__).parent.parent / "Segmenter" / "segmenter.py"
    if not segmenter_script.exists():
        raise FileNotFoundError(f"Segmenter script not found: {segmenter_script}")

    cmd = [
        sys.executable,
        str(segmenter_script),
        "--video-path",
        video_path,
        "--output",
        out_base,
        "--platform-id",
        "bench",
        f"--video-id={video_id}",
        "--run-id",
        "bench_run",
        "--sampling-policy-version",
        "v1",
        "--config-hash",
        "bench",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Segmenter failed: {result.stdout}")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Segmenter did not create metadata.json at {metadata_path}")

    return frames_dir_target


def _prepare_frame_indices(frames_dir: str, frames_count: Optional[int], full_video: bool) -> List[int]:
    """Prepare frame indices based on request."""
    metadata_path = os.path.join(frames_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_frames = int(meta.get("total_frames", 0))
    if total_frames <= 0:
        raise RuntimeError(f"Invalid total_frames in metadata: {total_frames}")

    if full_video:
        # Use all frames
        frame_indices = list(range(total_frames))
    elif frames_count is not None and frames_count > 0:
        # Use first N frames
        frame_indices = list(range(min(frames_count, total_frames)))
    else:
        # Default: 1 frame
        frame_indices = [0]

    return frame_indices


def _update_metadata_frame_indices(frames_dir: str, component_name: str, frame_indices: List[int]) -> None:
    """Update metadata.json with frame_indices for component."""
    metadata_path = os.path.join(frames_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if component_name not in meta:
        meta[component_name] = {}
    meta[component_name]["frame_indices"] = frame_indices

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _extract_models_used(component: str, rs_path: str) -> List[Dict[str, Any]]:
    """Extract models_used information from component output."""
    try:
        import numpy as np
        
        if component == "core_clip":
            npz_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
        elif component == "core_depth_midas":
            npz_path = os.path.join(rs_path, "core_depth_midas", "depth.npz")
        elif component == "core_optical_flow":
            npz_path = os.path.join(rs_path, "core_optical_flow", "flow.npz")
        elif component == "core_object_detections":
            npz_path = os.path.join(rs_path, "core_object_detections", "detections.npz")
        else:
            return []
        
        if not os.path.exists(npz_path):
            print(f"[bench] DEBUG: NPZ file not found: {npz_path}")
            return []
        
        data = np.load(npz_path, allow_pickle=True)
        meta = data.get("meta")
        if meta is None:
            print(f"[bench] DEBUG: 'meta' key not found in NPZ file")
            return []
        
        # Convert numpy object to dict if needed
        if hasattr(meta, 'item'):
            meta = meta.item()
        
        if not isinstance(meta, dict):
            print(f"[bench] DEBUG: 'meta' is not a dict, type: {type(meta)}")
            return []
        
        models_used = meta.get("models_used", [])
        if not models_used:
            print(f"[bench] DEBUG: 'models_used' is empty or missing in meta")
            print(f"[bench] DEBUG: Available meta keys: {list(meta.keys())}")
            return []
        
        if isinstance(models_used, (list, tuple)):
            # Convert numpy arrays to lists if needed
            result = []
            for m in models_used:
                if hasattr(m, 'item'):
                    m = m.item()
                if isinstance(m, dict):
                    # Convert numpy types to native Python types
                    cleaned = {}
                    for k, v in m.items():
                        if isinstance(v, (np.integer, np.floating)):
                            cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
                        elif isinstance(v, np.ndarray):
                            cleaned[k] = v.tolist()
                        elif isinstance(v, (list, tuple)):
                            cleaned[k] = [float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x) for x in v]
                        else:
                            cleaned[k] = str(v) if v is not None else None
                    result.append(cleaned)
            return result
        else:
            print(f"[bench] DEBUG: 'models_used' is not a list/tuple, type: {type(models_used)}")
        return []
    except Exception as e:
        import traceback
        print(f"[bench] WARNING: Failed to extract models_used: {e}")
        print(f"[bench] DEBUG: Traceback: {traceback.format_exc()}")
        return []


def _run_component(
    component: str,
    frames_dir: str,
    rs_path: str,
    triton_http_url: str,
    batch_size: int,
    component_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Run component and return execution info."""
    if component == "core_clip":
        return _run_core_clip(
            frames_dir=frames_dir,
            rs_path=rs_path,
            triton_http_url=triton_http_url,
            batch_size=batch_size,
            **component_args,
        )
    elif component == "core_depth_midas":
        return _run_core_depth_midas(
            frames_dir=frames_dir,
            rs_path=rs_path,
            triton_http_url=triton_http_url,
            batch_size=batch_size,
            **component_args,
        )
    elif component == "core_optical_flow":
        return _run_core_optical_flow(
            frames_dir=frames_dir,
            rs_path=rs_path,
            triton_http_url=triton_http_url,
            batch_size=batch_size,
            **component_args,
        )
    elif component == "core_object_detections":
        return _run_core_object_detections(
            frames_dir=frames_dir,
            rs_path=rs_path,
            triton_http_url=triton_http_url,
            batch_size=batch_size,
            **component_args,
        )
    else:
        raise ValueError(f"Unsupported component: {component}")


def _run_core_clip(
    frames_dir: str,
    rs_path: str,
    triton_http_url: str,
    batch_size: int,
    triton_image_model_spec: Optional[str] = None,
    triton_text_model_spec: Optional[str] = None,
    triton_image_model_name: Optional[str] = None,
    triton_text_model_name: Optional[str] = None,
    triton_preprocess_preset: str = "openai_clip_224",
    triton_image_datatype: str = "UINT8",
    triton_text_datatype: str = "INT64",
    triton_http_timeout_sec: float = 60.0,
    dp_models_root: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run core_clip component."""
    # core_clip is in DataProcessor/VisualProcessor/core/model_process/core_clip/main.py
    # __file__ is DataProcessor/benchmarks/run_component_bench.py
    # So we need to go up one level (parent = benchmarks, parent.parent = DataProcessor)
    core_clip_script = (
        Path(__file__).parent.parent / "VisualProcessor" / "core" / "model_process" / "core_clip" / "main.py"
    )
    if not core_clip_script.exists():
        raise FileNotFoundError(f"core_clip script not found: {core_clip_script}")

    # Auto-detect preprocess preset from model name if not explicitly set
    # This ensures preset matches the model's expected input size
    # Determine which model name will be used
    actual_model_name = triton_image_model_name
    if not actual_model_name:
        # If using model_spec, we can't auto-detect, so keep default
        # If using defaults, it will be clip_image_224
        if not (triton_image_model_spec and triton_text_model_spec):
            actual_model_name = "clip_image_224"  # default
    
    if actual_model_name:
        if "336" in actual_model_name and triton_preprocess_preset == "openai_clip_224":
            triton_preprocess_preset = "openai_clip_336"
            print(f"[bench] Auto-detected preset: {triton_preprocess_preset} from model name: {actual_model_name}")
        elif "448" in actual_model_name and triton_preprocess_preset == "openai_clip_224":
            triton_preprocess_preset = "openai_clip_448"
            print(f"[bench] Auto-detected preset: {triton_preprocess_preset} from model name: {actual_model_name}")
    
    cmd = [
        sys.executable,
        str(core_clip_script),
        "--frames-dir",
        frames_dir,
        "--rs-path",
        rs_path,
        "--runtime",
        "triton",
        "--batch-size",
        str(int(batch_size)),
        "--model-name",
        "ViT-B/32",  # placeholder, not used for triton
        "--triton-http-url",
        triton_http_url,
        "--triton-preprocess-preset",
        triton_preprocess_preset,
        "--triton-timeout-sec",
        str(float(triton_http_timeout_sec)),
        "--disable-text-cache",  # Disable cache for benchmarks
    ]

    # Preferred: use model_spec (via ModelManager)
    if triton_image_model_spec and triton_text_model_spec:
        cmd.extend(["--triton-image-model-spec", triton_image_model_spec])
        cmd.extend(["--triton-text-model-spec", triton_text_model_spec])
    # Fallback: use explicit model names
    elif triton_image_model_name and triton_text_model_name:
        cmd.extend(["--triton-image-model-name", triton_image_model_name])
        cmd.extend(["--triton-text-model-name", triton_text_model_name])
    else:
        # Default model names (common baseline setup)
        cmd.extend(["--triton-image-model-name", "clip_image_224"])
        cmd.extend(["--triton-text-model-name", "clip_text"])
    
    # Add datatype parameters (important for UINT8 vs FP32)
    cmd.extend(["--triton-image-datatype", triton_image_datatype])
    cmd.extend(["--triton-text-datatype", triton_text_datatype])

    # Prepare environment variables
    env = os.environ.copy()
    if triton_http_url and "TRITON_HTTP_URL" not in env:
        env["TRITON_HTTP_URL"] = triton_http_url
    if dp_models_root:
        env["DP_MODELS_ROOT"] = dp_models_root

    start_time = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    duration_sec = time.perf_counter() - start_time

    # Parse timing information from stderr
    component_timing = {}
    if result.stderr:
        for line in result.stderr.splitlines():
            if line.startswith("__BENCHMARK_TIMING__:"):
                try:
                    json_str = line.split(":", 1)[1]
                    timing_data = json.loads(json_str)
                    component_timing = timing_data.get("component_timing", {})
                except Exception:
                    pass

    return {
        "duration_sec": duration_sec,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "component_timing": component_timing,
    }


def _run_core_depth_midas(
    frames_dir: str,
    rs_path: str,
    triton_http_url: str,
    batch_size: int,
    triton_model_spec: Optional[str] = None,
    triton_model_name: Optional[str] = None,
    triton_preprocess_preset: str = "midas_384",
    triton_datatype: str = "UINT8",
    out_width: int = 384,
    out_height: int = 384,
    frames_bgr: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run core_depth_midas component."""
    core_depth_midas_script = (
        Path(__file__).parent.parent / "VisualProcessor" / "core" / "model_process" / "core_depth_midas" / "main.py"
    )
    if not core_depth_midas_script.exists():
        raise FileNotFoundError(f"core_depth_midas script not found: {core_depth_midas_script}")

    cmd = [
        sys.executable,
        str(core_depth_midas_script),
        "--frames-dir",
        frames_dir,
        "--rs-path",
        rs_path,
        "--runtime",
        "triton",
        "--batch-size",
        str(int(batch_size)),
        "--triton-http-url",
        triton_http_url,
        "--triton-preprocess-preset",
        triton_preprocess_preset,
        "--triton-datatype",
        triton_datatype,
        "--out-width",
        str(int(out_width)),
        "--out-height",
        str(int(out_height)),
    ]

    # Preferred: use model_spec (via ModelManager)
    if triton_model_spec:
        cmd.extend(["--triton-model-spec", triton_model_spec])
    # Fallback: use explicit model name
    elif triton_model_name:
        cmd.extend(["--triton-model-name", triton_model_name])
    else:
        # Default model name (common baseline setup)
        cmd.extend(["--triton-model-name", "midas_384"])

    if frames_bgr:
        cmd.append("--frames-bgr")

    # Prepare environment variables
    env = os.environ.copy()
    if triton_http_url and "TRITON_HTTP_URL" not in env:
        env["TRITON_HTTP_URL"] = triton_http_url

    start_time = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    duration_sec = time.perf_counter() - start_time

    # Parse timing information from stderr (if available)
    component_timing = {}
    if result.stderr:
        for line in result.stderr.splitlines():
            if line.startswith("__BENCHMARK_TIMING__:"):
                try:
                    json_str = line.split(":", 1)[1]
                    timing_data = json.loads(json_str)
                    component_timing = timing_data.get("component_timing", {})
                except Exception:
                    pass

    return {
        "duration_sec": duration_sec,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "component_timing": component_timing,
    }


def _run_core_object_detections(
    frames_dir: str,
    rs_path: str,
    triton_http_url: str,
    batch_size: int,
    runtime: str = "triton",
    triton_model_spec: Optional[str] = None,
    triton_model_name: Optional[str] = None,
    triton_model_version: Optional[str] = None,
    triton_preprocess_preset: str = "yolo11x_640",
    box_threshold: float = 0.6,
    iou_threshold: float = 0.3,
    model_path: Optional[str] = None,
    device: str = "auto",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run core_object_detections component."""
    core_od_script = (
        Path(__file__).parent.parent
        / "VisualProcessor"
        / "core"
        / "model_process"
        / "core_object_detections"
        / "main.py"
    )
    if not core_od_script.exists():
        raise FileNotFoundError(f"core_object_detections script not found: {core_od_script}")

    runtime = str(runtime or "triton").strip().lower()

    cmd = [
        sys.executable,
        str(core_od_script),
        "--frames-dir",
        frames_dir,
        "--rs-path",
        rs_path,
        "--batch-size",
        str(int(batch_size)),
        "--box-threshold",
        str(float(box_threshold)),
        "--iou-threshold",
        str(float(iou_threshold)),
        "--runtime",
        runtime,
    ]

    if runtime == "triton":
        if triton_model_spec:
            cmd.extend(["--triton-model-spec", triton_model_spec])
        if triton_http_url:
            cmd.extend(["--triton-http-url", triton_http_url])
        if triton_model_name:
            cmd.extend(["--triton-model-name", triton_model_name])
        if triton_model_version:
            cmd.extend(["--triton-model-version", triton_model_version])
        cmd.extend(["--triton-preprocess-preset", triton_preprocess_preset])
    else:
        if model_path:
            cmd.extend(["--model", model_path])
        cmd.extend(["--device", device])

    env = os.environ.copy()
    if triton_http_url and "TRITON_HTTP_URL" not in env:
        env["TRITON_HTTP_URL"] = triton_http_url

    start_time = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    duration_sec = time.perf_counter() - start_time

    component_timing: Dict[str, float] = {}

    return {
        "duration_sec": duration_sec,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "component_timing": component_timing,
    }

def _run_core_optical_flow(
    frames_dir: str,
    rs_path: str,
    triton_http_url: str,
    batch_size: int,
    triton_model_spec: Optional[str] = None,
    triton_model_name: Optional[str] = None,
    triton_model_version: Optional[str] = None,
    triton_preprocess_preset: str = "raft_256",
    triton_datatype: str = "UINT8",
    triton_input0_name: str = "INPUT0__0",
    triton_input1_name: str = "INPUT1__0",
    triton_output_name: str = "OUTPUT__0",
    model_version: str = "unknown",
    weights_digest: str = "unknown",
    precision: str = "fp32",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run core_optical_flow component."""
    core_optical_flow_script = (
        Path(__file__).parent.parent / "VisualProcessor" / "core" / "model_process" / "core_optical_flow" / "main.py"
    )
    if not core_optical_flow_script.exists():
        raise FileNotFoundError(f"core_optical_flow script not found: {core_optical_flow_script}")

    cmd = [
        sys.executable,
        str(core_optical_flow_script),
        "--frames-dir",
        frames_dir,
        "--rs-path",
        rs_path,
        "--runtime",
        "triton",
        "--batch-size",
        str(int(batch_size)),
        "--triton-http-url",
        triton_http_url,
        "--triton-preprocess-preset",
        triton_preprocess_preset,
        "--triton-datatype",
        triton_datatype,
        "--triton-input0-name",
        triton_input0_name,
        "--triton-input1-name",
        triton_input1_name,
        "--triton-output-name",
        triton_output_name,
        "--model-version",
        model_version,
        "--weights-digest",
        weights_digest,
        "--precision",
        precision,
    ]

    # Preferred: use model_spec (via ModelManager)
    if triton_model_spec:
        cmd.extend(["--triton-model-spec", triton_model_spec])
    # Fallback: use explicit model name
    elif triton_model_name:
        cmd.extend(["--triton-model-name", triton_model_name])
        if triton_model_version:
            cmd.extend(["--triton-model-version", triton_model_version])
    else:
        # Default model name (common baseline setup)
        cmd.extend(["--triton-model-name", "raft_256"])

    # Prepare environment variables
    env = os.environ.copy()
    if triton_http_url and "TRITON_HTTP_URL" not in env:
        env["TRITON_HTTP_URL"] = triton_http_url

    start_time = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    duration_sec = time.perf_counter() - start_time

    # Parse timing information from stderr (if available)
    component_timing = {}
    if result.stderr:
        for line in result.stderr.splitlines():
            if line.startswith("__BENCHMARK_TIMING__:"):
                try:
                    json_str = line.split(":", 1)[1]
                    timing_data = json.loads(json_str)
                    component_timing = timing_data.get("component_timing", {})
                except Exception:
                    pass

    return {
        "duration_sec": duration_sec,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "component_timing": component_timing,
    }


def _generate_html_report(results: Dict[str, Any], out_path: str) -> None:
    """Generate HTML report."""
    # Calculate deltas
    delta_gpu_triton = _calc_delta(results, 'gpu_mem_used_mb', 'before_triton', 'after_triton')
    delta_gpu_component = _calc_delta(results, 'gpu_mem_used_mb', 'after_triton', 'after_component')
    delta_ram_triton = _calc_delta(results, 'cpu_mem_used_mb', 'before_triton', 'after_triton')
    delta_ram_component = _calc_delta(results, 'cpu_mem_used_mb', 'after_triton', 'after_component')
    
    # Extract models used
    models_used = results.get('models_used', [])
    models_html = ""
    if models_used:
        models_html = "<table><tr><th>Model Name</th><th>Version</th><th>Runtime</th><th>Engine</th><th>Device</th></tr>"
        for m in models_used:
            model_name = m.get('model_name', 'N/A')
            model_version = m.get('model_version', 'N/A')
            runtime = m.get('runtime', 'N/A')
            engine = m.get('engine', 'N/A')
            device = m.get('device', 'N/A')
            models_html += f"<tr><td>{model_name}</td><td>{model_version}</td><td>{runtime}</td><td>{engine}</td><td>{device}</td></tr>"
        models_html += "</table>"
    else:
        models_html = "<p>No model information available</p>"
    
    # Extract peak timestamps
    peak_timestamps = results.get('resources', {}).get('peak_timestamps', {})
    cpu_peak_time = peak_timestamps.get('cpu_util_peak_elapsed_sec', 'N/A')
    cpu_peak_ts = peak_timestamps.get('cpu_util_peak', 'N/A')
    gpu_peak_time = peak_timestamps.get('gpu_util_peak_elapsed_sec', 'N/A')
    gpu_peak_ts = peak_timestamps.get('gpu_util_peak', 'N/A')
    
    # Extract time series for charts
    time_series = results.get('resources', {}).get('time_series', [])
    time_series_js = "[]"
    if time_series:
        # Prepare data for Chart.js
        cpu_data = []
        gpu_data = []
        cpu_mem_data = []
        gpu_mem_data = []
        labels = []
        for ts in time_series[:1000]:  # Limit to 1000 points for performance
            elapsed = ts.get('elapsed_sec', 0)
            labels.append(f"{elapsed:.2f}")
            cpu_data.append(ts.get('cpu_util_pct') if ts.get('cpu_util_pct') is not None else None)
            gpu_data.append(ts.get('gpu_util_pct') if ts.get('gpu_util_pct') is not None else None)
            cpu_mem_data.append(ts.get('cpu_mem_used_mb') if ts.get('cpu_mem_used_mb') is not None else None)
            gpu_mem_data.append(ts.get('gpu_mem_used_mb') if ts.get('gpu_mem_used_mb') is not None else None)
        
        time_series_js = f"""
        {{
            labels: {json.dumps(labels)},
            cpu_util: {json.dumps(cpu_data)},
            gpu_util: {json.dumps(gpu_data)},
            cpu_mem: {json.dumps(cpu_mem_data)},
            gpu_mem: {json.dumps(gpu_mem_data)}
        }}
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Component Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            background: #e8f5e9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            font-weight: bold;
            color: #2e7d32;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        .chart-container {{
            margin: 20px 0;
            height: 300px;
            position: relative;
        }}
        canvas {{
            max-height: 300px;
        }}
        .peak-highlight {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>Component Benchmark Report</h1>
        <p class="timestamp">Generated: {datetime.utcnow().isoformat()}Z</p>

        <h2>Test Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Component</td><td>{results.get('component', 'N/A')}</td></tr>
            <tr><td>Video Path</td><td>{results.get('video_path', 'N/A')}</td></tr>
            <tr><td>Frames Count</td><td>{results.get('frames_count', 'N/A')}</td></tr>
            <tr><td>Batch Size</td><td>{results.get('batch_size', 'N/A')}</td></tr>
            <tr><td>Triton URL</td><td>{results.get('triton_http_url', 'N/A')}</td></tr>
            <tr><td>Component Duration</td><td>{results.get('component_duration_sec', 0):.3f} seconds</td></tr>
        </table>

        <h2>Models Used</h2>
        {models_html}

        <h2>Resource Metrics</h2>
        
        <h3>Before Triton Startup</h3>
        <div class="metric">
            <span class="metric-label">GPU VRAM:</span> {results.get('resources', {}).get('before_triton', {}).get('gpu_mem_used_mb', 'N/A')} / {results.get('resources', {}).get('before_triton', {}).get('gpu_mem_total_mb', 'N/A')} MB
            <br>
            <span class="metric-label">GPU Utilization:</span> {results.get('resources', {}).get('before_triton', {}).get('gpu_util_pct', 'N/A')}%
            <br>
            <span class="metric-label">RAM:</span> {results.get('resources', {}).get('before_triton', {}).get('cpu_mem_used_mb', 'N/A')} / {results.get('resources', {}).get('before_triton', {}).get('cpu_mem_total_mb', 'N/A')} MB
        </div>

        <h3>After Triton Startup</h3>
        <div class="metric">
            <span class="metric-label">GPU VRAM:</span> {results.get('resources', {}).get('after_triton', {}).get('gpu_mem_used_mb', 'N/A')} / {results.get('resources', {}).get('after_triton', {}).get('gpu_mem_total_mb', 'N/A')} MB
            <br>
            <span class="metric-label">GPU Utilization:</span> {results.get('resources', {}).get('after_triton', {}).get('gpu_util_pct', 'N/A')}%
            <br>
            <span class="metric-label">RAM:</span> {results.get('resources', {}).get('after_triton', {}).get('cpu_mem_used_mb', 'N/A')} / {results.get('resources', {}).get('after_triton', {}).get('cpu_mem_total_mb', 'N/A')} MB
        </div>

        <h3>After Component Execution</h3>
        <div class="metric">
            <span class="metric-label">GPU VRAM:</span> {results.get('resources', {}).get('after_component', {}).get('gpu_mem_used_mb', 'N/A')} / {results.get('resources', {}).get('after_component', {}).get('gpu_mem_total_mb', 'N/A')} MB
            <br>
            <span class="metric-label">GPU Utilization:</span> {results.get('resources', {}).get('after_component', {}).get('gpu_util_pct', 'N/A')}%
            <br>
            <span class="metric-label">RAM:</span> {results.get('resources', {}).get('after_component', {}).get('cpu_mem_used_mb', 'N/A')} / {results.get('resources', {}).get('after_component', {}).get('cpu_mem_total_mb', 'N/A')} MB
        </div>

        <h3>Peak Resource Usage (During Component Execution)</h3>
        <div class="metric">
            <span class="metric-label">Peak GPU VRAM:</span> {results.get('resources', {}).get('peaks', {}).get('vram_used_peak_mb', 'N/A')} MB
            {f'<br><span class="timestamp">At: {gpu_peak_time}s elapsed ({gpu_peak_ts})</span>' if gpu_peak_time != 'N/A' else ''}
            <br>
            <span class="metric-label">Peak GPU Utilization:</span> {results.get('resources', {}).get('peaks', {}).get('gpu_util_peak_pct', 'N/A')}%
            {f'<br><span class="timestamp">At: {gpu_peak_time}s elapsed ({gpu_peak_ts})</span>' if gpu_peak_time != 'N/A' else ''}
            <br>
            <span class="metric-label">Peak RAM:</span> {results.get('resources', {}).get('peaks', {}).get('ram_used_peak_mb', 'N/A')} MB
            <br>
            <span class="metric-label">Peak CPU Utilization:</span> {results.get('resources', {}).get('peaks', {}).get('cpu_util_peak_pct', 'N/A')}%
            {f'<br><span class="timestamp">⚠️ Peak CPU load at: {cpu_peak_time}s elapsed ({cpu_peak_ts})</span>' if cpu_peak_time != 'N/A' else ''}
        </div>

        <h2>Resource Summary Table</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Before Triton</th>
                <th>After Triton</th>
                <th>After Component</th>
                <th>Peak (During Component)</th>
                <th>Triton Delta</th>
                <th>Component Delta</th>
            </tr>
            <tr>
                <td>GPU VRAM Used (MB)</td>
                <td>{results.get('resources', {}).get('before_triton', {}).get('gpu_mem_used_mb', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_triton', {}).get('gpu_mem_used_mb', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_component', {}).get('gpu_mem_used_mb', 'N/A')}</td>
                <td>{results.get('resources', {}).get('peaks', {}).get('vram_used_peak_mb', 'N/A')}</td>
                <td>{delta_gpu_triton}</td>
                <td>{delta_gpu_component}</td>
            </tr>
            <tr>
                <td>GPU Utilization (%)</td>
                <td>{results.get('resources', {}).get('before_triton', {}).get('gpu_util_pct', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_triton', {}).get('gpu_util_pct', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_component', {}).get('gpu_util_pct', 'N/A')}</td>
                <td>{results.get('resources', {}).get('peaks', {}).get('gpu_util_peak_pct', 'N/A')}</td>
                <td>N/A</td>
                <td>N/A</td>
            </tr>
            <tr>
                <td>RAM Used (MB)</td>
                <td>{results.get('resources', {}).get('before_triton', {}).get('cpu_mem_used_mb', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_triton', {}).get('cpu_mem_used_mb', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_component', {}).get('cpu_mem_used_mb', 'N/A')}</td>
                <td>{results.get('resources', {}).get('peaks', {}).get('ram_used_peak_mb', 'N/A')}</td>
                <td>{delta_ram_triton}</td>
                <td>{delta_ram_component}</td>
            </tr>
            <tr>
                <td>CPU Utilization (%)</td>
                <td>{results.get('resources', {}).get('before_triton', {}).get('cpu_util_pct', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_triton', {}).get('cpu_util_pct', 'N/A')}</td>
                <td>{results.get('resources', {}).get('after_component', {}).get('cpu_util_pct', 'N/A')}</td>
                <td>{results.get('resources', {}).get('peaks', {}).get('cpu_util_peak_pct', 'N/A')}</td>
                <td>N/A</td>
                <td>N/A</td>
            </tr>
        </table>
        
        <h2>Component Execution</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Success</td><td>{'Yes' if results.get('component_success', False) else 'No'}</td></tr>
            <tr><td>Return Code</td><td>{results.get('component_returncode', 'N/A')}</td></tr>
            <tr><td>Duration (seconds)</td><td>{results.get('component_duration_sec', 0):.3f}</td></tr>
        </table>

        <h2>Component Timing Breakdown</h2>
        {_format_timing_table(results.get('component_timing', {}))}

        <h2>Resource Usage Over Time</h2>
        <div class="chart-container">
            <canvas id="cpuChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="gpuChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="memChart"></canvas>
        </div>

        {f'<div class="peak-highlight"><strong>⚠️ CPU Peak Load Detected:</strong><br>Peak CPU utilization ({results.get("resources", {}).get("peaks", {}).get("cpu_util_peak_pct", "N/A")}%) occurred at {cpu_peak_time}s elapsed ({cpu_peak_ts})</div>' if cpu_peak_time != 'N/A' and results.get('resources', {}).get('peaks', {}).get('cpu_util_peak_pct', 0) and results.get('resources', {}).get('peaks', {}).get('cpu_util_peak_pct', 0) > 90 else ''}
    </div>
    <script>
        const timeSeries = {time_series_js};
        
        if (timeSeries.labels && timeSeries.labels.length > 0) {{
            // CPU Utilization Chart
            const cpuCtx = document.getElementById('cpuChart').getContext('2d');
            new Chart(cpuCtx, {{
                type: 'line',
                data: {{
                    labels: timeSeries.labels,
                    datasets: [{{
                        label: 'CPU Utilization (%)',
                        data: timeSeries.cpu_util,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'CPU Utilization (%)'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time Elapsed (seconds)'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'CPU Utilization Over Time'
                        }}
                    }}
                }}
            }});

            // GPU Utilization Chart
            const gpuCtx = document.getElementById('gpuChart').getContext('2d');
            new Chart(gpuCtx, {{
                type: 'line',
                data: {{
                    labels: timeSeries.labels,
                    datasets: [{{
                        label: 'GPU Utilization (%)',
                        data: timeSeries.gpu_util,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'GPU Utilization (%)'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time Elapsed (seconds)'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'GPU Utilization Over Time'
                        }}
                    }}
                }}
            }});

            // Memory Usage Chart
            const memCtx = document.getElementById('memChart').getContext('2d');
            new Chart(memCtx, {{
                type: 'line',
                data: {{
                    labels: timeSeries.labels,
                    datasets: [{{
                        label: 'CPU RAM Used (MB)',
                        data: timeSeries.cpu_mem,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1,
                        yAxisID: 'y'
                    }}, {{
                        label: 'GPU VRAM Used (MB)',
                        data: timeSeries.gpu_mem,
                        borderColor: 'rgb(255, 206, 86)',
                        backgroundColor: 'rgba(255, 206, 86, 0.2)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            type: 'linear',
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'CPU RAM (MB)'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'GPU VRAM (MB)'
                            }},
                            grid: {{
                                drawOnChartArea: false
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time Elapsed (seconds)'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Memory Usage Over Time'
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[bench] Generated HTML report: {out_path}")


def _format_timing_table(timing: Dict[str, float]) -> str:
    """Format timing breakdown as HTML table."""
    if not timing:
        return "<p>No timing information available</p>"
    
    total = timing.get('total', 0)
    rows = []
    rows.append("<table><tr><th>Phase</th><th>Time (seconds)</th><th>Percentage</th></tr>")
    
    # Sort by time descending
    sorted_timing = sorted(timing.items(), key=lambda x: x[1], reverse=True)
    for phase, time_sec in sorted_timing:
        if phase == 'total':
            continue
        pct = (time_sec / total * 100) if total > 0 else 0
        rows.append(f"<tr><td>{phase}</td><td>{time_sec:.3f}</td><td>{pct:.1f}%</td></tr>")
    
    if total > 0:
        rows.append(f"<tr><td><strong>Total</strong></td><td><strong>{total:.3f}</strong></td><td><strong>100.0%</strong></td></tr>")
    
    rows.append("</table>")
    return "\n".join(rows)


def _calc_delta(results: Dict[str, Any], key: str, before_key: str, after_key: str) -> str:
    """Calculate delta between two snapshots."""
    before = results.get("resources", {}).get(before_key, {}).get(key)
    after = results.get("resources", {}).get(after_key, {}).get(key)
    if before is None or after is None:
        return "N/A"
    try:
        delta = after - before
        return f"+{delta}" if delta >= 0 else str(delta)
    except Exception:
        return "N/A"


def main() -> None:
    # Preprocess sys.argv to handle values starting with '-' for --video-path
    # Convert "--video-path -value" to "--video-path=-value"
    argv = sys.argv[:]
    for i, arg in enumerate(argv):
        if arg == "--video-path" and i + 1 < len(argv):
            next_arg = argv[i + 1]
            if next_arg.startswith("-") and not next_arg.startswith("--"):
                # Value starts with '-', combine with '=' to prevent argparse from treating it as an option
                argv[i] = f"{arg}={next_arg}"
                argv.pop(i + 1)
                sys.argv = argv
                break
    
    parser = argparse.ArgumentParser(
        description="Component-level benchmark with resource monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        choices=["core_clip", "core_depth_midas", "core_optical_flow", "core_object_detections"],
        help="Component name (core_clip, core_depth_midas, core_optical_flow, or core_object_detections)",
    )
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Path to frames directory (will be created via Segmenter if not provided)",
    )
    parser.add_argument(
        "--frames-count",
        type=int,
        default=None,
        help="Number of frames to process (default: 1, use --full-video for all frames)",
    )
    parser.add_argument("--full-video", action="store_true", help="Process full video (all frames)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for component")
    parser.add_argument("--triton-http-url", type=str, required=True, help="Triton HTTP URL")
    parser.add_argument(
        "--wait-triton",
        action="store_true",
        help="Wait for Triton to become available (useful if starting manually)",
    )
    parser.add_argument("--triton-timeout", type=float, default=300.0, help="Timeout for Triton wait (seconds)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/out_component/<timestamp>)",
    )
    parser.add_argument(
        "--triton-image-model-spec",
        type=str,
        default=None,
        help="Triton image model spec (for core_clip)",
    )
    parser.add_argument(
        "--triton-text-model-spec",
        type=str,
        default=None,
        help="Triton text model spec (for core_clip)",
    )
    parser.add_argument(
        "--triton-image-model-name",
        type=str,
        default=None,
        help="Triton image model name (for core_clip, default: clip_image_224)",
    )
    parser.add_argument(
        "--triton-text-model-name",
        type=str,
        default=None,
        help="Triton text model name (for core_clip, default: clip_text)",
    )
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="openai_clip_224",
        choices=["openai_clip_224", "openai_clip_336", "openai_clip_448", "midas_256", "midas_384", "midas_512", "raft_256", "raft_384", "raft_512"],
        help="Triton preprocess preset (for core_clip: openai_clip_224/336/448, for core_depth_midas: midas_256/384/512, for core_optical_flow: raft_256/384/512)",
    )
    parser.add_argument(
        "--triton-http-timeout-sec",
        type=float,
        default=60.0,
        help="Triton HTTP client timeout in seconds (default: 60.0, increased for text inference with many prompts)",
    )
    parser.add_argument(
        "--triton-image-datatype",
        type=str,
        default="UINT8",
        choices=["UINT8", "FP32"],
        help="Triton image input datatype (default: UINT8 for clip_image_224)",
    )
    parser.add_argument(
        "--triton-text-datatype",
        type=str,
        default="INT64",
        choices=["INT64"],
        help="Triton text input datatype (default: INT64)",
    )
    parser.add_argument(
        "--dp-models-root",
        type=str,
        default=None,
        help="DP_MODELS_ROOT environment variable (required for core_clip Places365 prompts). If not set, uses env var DP_MODELS_ROOT",
    )
    # Arguments for core_depth_midas, core_optical_flow and core_object_detections (Triton)
    parser.add_argument(
        "--triton-model-spec",
        type=str,
        default=None,
        help="Triton model spec (for core_depth_midas: midas_384_triton, for core_optical_flow: raft_256_triton/raft_384_triton/raft_512_triton, for core_object_detections: yolo11x_640_triton, etc.)",
    )
    parser.add_argument(
        "--triton-model-name",
        type=str,
        default=None,
        help="Triton model name (for core_depth_midas: midas_384, for core_optical_flow: raft_256/raft_384/raft_512, for core_object_detections: yolo11x_320/640/960)",
    )
    parser.add_argument(
        "--triton-datatype",
        type=str,
        default="UINT8",
        choices=["UINT8", "FP32"],
        help="Triton input datatype (for core_depth_midas, default: UINT8)",
    )
    parser.add_argument(
        "--out-width",
        type=int,
        default=384,
        help="Output width for depth maps (for core_depth_midas, default: 384)",
    )
    parser.add_argument(
        "--out-height",
        type=int,
        default=384,
        help="Output height for depth maps (for core_depth_midas, default: 384)",
    )
    parser.add_argument(
        "--frames-bgr",
        action="store_true",
        help="Set if FrameManager returns BGR images instead of RGB (for core_depth_midas)",
    )
    # Arguments for core_optical_flow
    parser.add_argument(
        "--triton-model-version",
        type=str,
        default=None,
        help="Triton model version (for core_optical_flow, optional)",
    )
    parser.add_argument(
        "--triton-input0-name",
        type=str,
        default="INPUT0__0",
        help="Triton input0 name (for core_optical_flow, default: INPUT0__0)",
    )
    parser.add_argument(
        "--triton-input1-name",
        type=str,
        default="INPUT1__0",
        help="Triton input1 name (for core_optical_flow, default: INPUT1__0)",
    )
    parser.add_argument(
        "--triton-output-name",
        type=str,
        default="OUTPUT__0",
        help="Triton output name (for core_optical_flow, default: OUTPUT__0)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="unknown",
        help="Model version for metadata (for core_optical_flow, default: unknown)",
    )
    parser.add_argument(
        "--weights-digest",
        type=str,
        default="unknown",
        help="Weights digest for metadata (for core_optical_flow, default: unknown)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for metadata (for core_optical_flow, default: fp32)",
    )

    # Arguments for core_object_detections
    parser.add_argument(
        "--od-runtime",
        type=str,
        default="triton",
        choices=["triton", "ultralytics"],
        help="Runtime for core_object_detections (default: triton)",
    )
    parser.add_argument(
        "--od-model",
        type=str,
        default=None,
        help="Path to local YOLO weights (for runtime=ultralytics or class-name resolution for runtime=triton)",
    )
    parser.add_argument(
        "--od-triton-preprocess-preset",
        type=str,
        default="yolo11x_640",
        choices=["yolo11x_320", "yolo11x_640", "yolo11x_960"],
        help="Preprocess preset for core_object_detections Triton YOLO model",
    )
    parser.add_argument(
        "--od-box-threshold",
        type=float,
        default=0.6,
        help="Box confidence threshold for object detections",
    )
    parser.add_argument(
        "--od-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for NMS/tracking in core_object_detections",
    )
    parser.add_argument(
        "--od-device",
        type=str,
        default="auto",
        help="Device for runtime=ultralytics in core_object_detections ('auto'|'cpu'|'cuda')",
    )

    args = parser.parse_args()
    
    # Validate component-specific arguments
    if args.component == "core_depth_midas":
        # Update preprocess preset choices for midas
        if args.triton_preprocess_preset not in ["midas_256", "midas_384", "midas_512"]:
            args.triton_preprocess_preset = "midas_384"
    elif args.component == "core_optical_flow":
        # Update preprocess preset choices for raft
        if args.triton_preprocess_preset not in ["raft_256", "raft_384", "raft_512"]:
            args.triton_preprocess_preset = "raft_256"
    elif args.component == "core_object_detections":
        # Ensure OD Triton preset is valid
        if args.od_triton_preprocess_preset not in ["yolo11x_320", "yolo11x_640", "yolo11x_960"]:
            args.od_triton_preprocess_preset = "yolo11x_640"

    # Setup output directory
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(Path(__file__).parent, "out_component", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[bench] Output directory: {out_dir}")

    # Step 1: Measure resources BEFORE Triton startup
    print("[bench] Step 1: Measuring resources BEFORE Triton startup...")
    before_triton = _get_resource_snapshot()
    print(f"[bench] GPU VRAM before Triton: {before_triton.gpu_mem_used_mb}/{before_triton.gpu_mem_total_mb} MB")
    print(f"[bench] RAM before Triton: {before_triton.cpu_mem_used_mb}/{before_triton.cpu_mem_total_mb} MB")

    # Step 2: Wait for Triton if requested
    if args.wait_triton:
        print("[bench] Step 2: Waiting for Triton to become available (automatic)...")
        if not _wait_for_triton(args.triton_http_url, timeout_sec=args.triton_timeout):
            print("[bench] ERROR: Failed to wait for Triton")
            return
    else:
        print("[bench] " + "=" * 70)
        print("[bench] Step 2: MANUAL STEP - Start Triton now")
        print("[bench] " + "-" * 70)
        print("[bench] Please:")
        print("[bench]   1. Start Triton Inference Server in another terminal")
        print(f"[bench]   2. Ensure it's available at: {args.triton_http_url}")
        print("[bench]   3. Press Enter here when Triton is ready")
        print("[bench] " + "=" * 70)
        input()
        
        # Verify Triton is actually available
        print("[bench] Verifying Triton is available...")
        if not _wait_for_triton(args.triton_http_url, timeout_sec=10.0):
            print("[bench] WARNING: Triton does not appear to be available yet.")
            print("[bench] Continuing anyway, but benchmarks may fail.")
        else:
            print("[bench] ✓ Triton is available!")

    # Step 3: Measure resources AFTER Triton startup
    print("[bench] Step 3: Measuring resources AFTER Triton startup...")
    time.sleep(2.0)  # Give Triton a moment to stabilize
    after_triton = _get_resource_snapshot()
    print(f"[bench] GPU VRAM after Triton: {after_triton.gpu_mem_used_mb}/{after_triton.gpu_mem_total_mb} MB")
    print(f"[bench] RAM after Triton: {after_triton.cpu_mem_used_mb}/{after_triton.cpu_mem_total_mb} MB")

    # Step 4: Prepare frames_dir and frame_indices
    print("[bench] Step 4: Preparing frames_dir...")
    frames_dir = _ensure_frames_dir(args.video_path, args.frames_dir, out_dir)
    
    # For core_optical_flow, ensure at least 2 frames (required for frame pairs)
    frames_count_for_prep = args.frames_count
    if args.component == "core_optical_flow" and frames_count_for_prep is not None and frames_count_for_prep < 2:
        print(f"[bench] WARNING: core_optical_flow requires at least 2 frames (got {frames_count_for_prep}). Setting to 2.")
        frames_count_for_prep = 2
    
    frame_indices = _prepare_frame_indices(frames_dir, frames_count_for_prep, args.full_video)
    
    # Validate minimum frame count for core_optical_flow
    if args.component == "core_optical_flow" and len(frame_indices) < 2:
        print(f"[bench] ERROR: core_optical_flow requires at least 2 frames, but only {len(frame_indices)} frame(s) available.")
        print(f"[bench] Please use --frames-count 2 or more, or --full-video if video has at least 2 frames.")
        return
    
    _update_metadata_frame_indices(frames_dir, args.component, frame_indices)
    print(f"[bench] Processing {len(frame_indices)} frame(s): {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    # Step 5: Start resource monitoring and run component
    print("[bench] Step 5: Running component with resource monitoring...")
    monitor = EnhancedResourceMonitor(interval_sec=0.25)
    monitor.start()

    rs_path = os.path.join(out_dir, "result_store")
    os.makedirs(rs_path, exist_ok=True)

    # Set DP_MODELS_ROOT if provided or use environment variable
    dp_models_root = args.dp_models_root or os.environ.get("DP_MODELS_ROOT")
    if dp_models_root:
        os.environ["DP_MODELS_ROOT"] = dp_models_root
        print(f"[bench] Using DP_MODELS_ROOT: {dp_models_root}")
    elif args.component == "core_clip":
        print("[bench] WARNING: DP_MODELS_ROOT not set. core_clip may fail when loading Places365 prompts.")

    # Prepare component-specific arguments
    if args.component == "core_clip":
        component_args = {
            "triton_image_model_spec": args.triton_image_model_spec,
            "triton_text_model_spec": args.triton_text_model_spec,
            "triton_image_model_name": args.triton_image_model_name,
            "triton_text_model_name": args.triton_text_model_name,
            "triton_preprocess_preset": args.triton_preprocess_preset,
            "triton_image_datatype": args.triton_image_datatype,
            "triton_text_datatype": args.triton_text_datatype,
            "triton_http_timeout_sec": args.triton_http_timeout_sec,
            "dp_models_root": dp_models_root,
        }
    elif args.component == "core_depth_midas":
        component_args = {
            "triton_model_spec": args.triton_model_spec,
            "triton_model_name": args.triton_model_name,
            "triton_preprocess_preset": args.triton_preprocess_preset,
            "triton_datatype": args.triton_datatype,
            "out_width": args.out_width,
            "out_height": args.out_height,
            "frames_bgr": args.frames_bgr,
        }
    elif args.component == "core_optical_flow":
        component_args = {
            "triton_model_spec": args.triton_model_spec,
            "triton_model_name": args.triton_model_name,
            "triton_model_version": args.triton_model_version,
            "triton_preprocess_preset": args.triton_preprocess_preset,
            "triton_datatype": args.triton_datatype,
            "triton_input0_name": args.triton_input0_name,
            "triton_input1_name": args.triton_input1_name,
            "triton_output_name": args.triton_output_name,
            "model_version": args.model_version,
            "weights_digest": args.weights_digest,
            "precision": args.precision,
        }
    elif args.component == "core_object_detections":
        component_args = {
            "runtime": args.od_runtime,
            "triton_model_spec": args.triton_model_spec,
            "triton_model_name": args.triton_model_name,
            "triton_model_version": args.triton_model_version,
            "triton_preprocess_preset": args.od_triton_preprocess_preset,
            "box_threshold": args.od_box_threshold,
            "iou_threshold": args.od_iou_threshold,
            "model_path": args.od_model,
            "device": args.od_device,
        }
    else:
        component_args = {}

    exec_result = _run_component(
        component=args.component,
        frames_dir=frames_dir,
        rs_path=rs_path,
        triton_http_url=args.triton_http_url,
        batch_size=args.batch_size,
        component_args=component_args,
    )

    monitor.stop(timeout_sec=2.0)
    peaks = monitor.peaks.to_dict()
    time_series = [ts.to_dict() for ts in monitor.time_series]
    peak_timestamps = monitor.peak_timestamps

    # Extract models_used from component output
    # Try to extract even if component failed (file might have been created before failure)
    models_used = _extract_models_used(args.component, rs_path)
    if models_used:
        print(f"[bench] Found {len(models_used)} model(s) used:")
        for i, m in enumerate(models_used):
            model_name = m.get("model_name", "unknown")
            model_version = m.get("model_version", "unknown")
            print(f"[bench]   {i+1}. {model_name} (version: {model_version})")
    elif exec_result["success"]:
        print(f"[bench] WARNING: Component succeeded but models_used could not be extracted")

    if not exec_result["success"]:
        print(f"[bench] ERROR: Component execution failed with returncode {exec_result['returncode']}")
        if exec_result.get('stdout'):
            print(f"[bench] stdout: {exec_result['stdout']}")
        if exec_result.get('stderr'):
            print(f"[bench] stderr: {exec_result['stderr']}")

    # Step 6: Measure resources AFTER component execution
    print("[bench] Step 6: Measuring resources AFTER component execution...")
    time.sleep(1.0)  # Give system a moment to stabilize
    after_component = _get_resource_snapshot()
    print(f"[bench] GPU VRAM after component: {after_component.gpu_mem_used_mb}/{after_component.gpu_mem_total_mb} MB")
    print(f"[bench] RAM after component: {after_component.cpu_mem_used_mb}/{after_component.cpu_mem_total_mb} MB")

    # Step 7: Generate reports
    print("[bench] Step 7: Generating reports...")

    # Convert time series to ResourceTimeSeries objects
    time_series_objects = [ResourceTimeSeries(**ts) for ts in time_series]

    resources = ResourceMetrics(
        before_triton=before_triton,
        after_triton=after_triton,
        after_component=after_component,
        peaks=peaks,
        time_series=time_series_objects,
        peak_timestamps=peak_timestamps,
    )

    results = {
        "component": args.component,
        "video_path": args.video_path,
        "frames_dir": frames_dir,
        "frames_count": len(frame_indices),
        "frame_indices": frame_indices,
        "batch_size": args.batch_size,
        "triton_http_url": args.triton_http_url,
        "component_duration_sec": exec_result["duration_sec"],
        "component_success": exec_result["success"],
        "component_returncode": exec_result["returncode"],
        "component_timing": exec_result.get("component_timing", {}),
        "models_used": models_used,
        "resources": resources.to_dict(),
        "created_at": datetime.utcnow().isoformat(),
    }

    # JSON report
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[bench] Generated JSON report: {json_path}")

    # HTML report
    html_path = os.path.join(out_dir, "report.html")
    _generate_html_report(results, html_path)

    # Summary table (print to console)
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Component: {args.component}")
    print(f"Frames processed: {len(frame_indices)}")
    print(f"Component duration: {exec_result['duration_sec']:.3f} seconds")
    
    # Models used
    if models_used:
        print(f"\nModels Used ({len(models_used)}):")
        for i, m in enumerate(models_used, 1):
            model_name = m.get('model_name', 'unknown')
            model_version = m.get('model_version', 'unknown')
            runtime = m.get('runtime', 'unknown')
            engine = m.get('engine', 'unknown')
            device = m.get('device', 'unknown')
            print(f"  {i}. {model_name} (v{model_version}, {runtime}, {engine}, {device})")
    else:
        print("\nModels Used: Not available")
    
    print(f"\nResource Metrics:")
    print(f"  GPU VRAM before Triton: {before_triton.gpu_mem_used_mb}/{before_triton.gpu_mem_total_mb} MB")
    print(f"  GPU VRAM after Triton: {after_triton.gpu_mem_used_mb}/{after_triton.gpu_mem_total_mb} MB")
    print(f"  GPU VRAM after component: {after_component.gpu_mem_used_mb}/{after_component.gpu_mem_total_mb} MB")
    print(f"  GPU VRAM peak: {peaks.get('vram_used_peak_mb', 'N/A')} MB")
    if peak_timestamps.get('vram_peak_elapsed_sec'):
        print(f"    ⏱️  Peak VRAM at: {peak_timestamps.get('vram_peak_elapsed_sec')}s elapsed ({peak_timestamps.get('vram_peak', 'N/A')})")
    print(f"  RAM before Triton: {before_triton.cpu_mem_used_mb}/{before_triton.cpu_mem_total_mb} MB")
    print(f"  RAM after Triton: {after_triton.cpu_mem_used_mb}/{after_triton.cpu_mem_total_mb} MB")
    print(f"  RAM after component: {after_component.cpu_mem_used_mb}/{after_component.cpu_mem_total_mb} MB")
    print(f"  RAM peak: {peaks.get('ram_used_peak_mb', 'N/A')} MB")
    if peak_timestamps.get('ram_peak_elapsed_sec'):
        print(f"    ⏱️  Peak RAM at: {peak_timestamps.get('ram_peak_elapsed_sec')}s elapsed ({peak_timestamps.get('ram_peak', 'N/A')})")
    print(f"  CPU utilization peak: {peaks.get('cpu_util_peak_pct', 'N/A')}%")
    if peak_timestamps.get('cpu_util_peak_elapsed_sec'):
        peak_cpu_time = peak_timestamps.get('cpu_util_peak_elapsed_sec', 'N/A')
        peak_cpu_ts = peak_timestamps.get('cpu_util_peak', 'N/A')
        peak_cpu_pct = peaks.get('cpu_util_peak_pct', 0)
        print(f"    ⚠️  Peak CPU load at: {peak_cpu_time}s elapsed ({peak_cpu_ts})")
        if peak_cpu_pct and peak_cpu_pct > 90:
            print(f"    ⚠️  WARNING: CPU was heavily loaded ({peak_cpu_pct:.1f}%) - consider optimizing or scaling")
        # Estimate which phase based on elapsed time
        duration = exec_result['duration_sec']
        if peak_cpu_time != 'N/A' and duration > 0:
            try:
                peak_time_float = float(peak_cpu_time)
                phase_pct = (peak_time_float / duration) * 100
                if phase_pct < 25:
                    phase = "Initialization/Model Loading"
                elif phase_pct < 50:
                    phase = "Early Processing"
                elif phase_pct < 75:
                    phase = "Mid Processing"
                else:
                    phase = "Late Processing/Saving"
                print(f"    📍 Estimated phase: {phase} ({phase_pct:.1f}% of execution time)")
            except Exception:
                pass
    print(f"  GPU utilization peak: {peaks.get('gpu_util_peak_pct', 'N/A')}%")
    if peak_timestamps.get('gpu_util_peak_elapsed_sec'):
        print(f"    ⏱️  Peak GPU load at: {peak_timestamps.get('gpu_util_peak_elapsed_sec')}s elapsed ({peak_timestamps.get('gpu_util_peak', 'N/A')})")
    
    # Component timing breakdown
    component_timing = exec_result.get('component_timing', {})
    if component_timing:
        print(f"\nComponent Timing Breakdown:")
        total = component_timing.get('total', 0)
        sorted_timing = sorted([(k, v) for k, v in component_timing.items() if k != 'total'], key=lambda x: x[1], reverse=True)
        for phase, time_sec in sorted_timing:
            pct = (time_sec / total * 100) if total > 0 else 0
            print(f"  {phase}: {time_sec:.3f}s ({pct:.1f}%)")
        if total > 0:
            print(f"  Total: {total:.3f}s")
    
    print("=" * 80)
    print(f"\nReports saved to: {out_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")


if __name__ == "__main__":
    main()

