#!/usr/bin/env python3
"""
Multi-threaded component benchmark harness.

This script runs a component (e.g., core_clip) in multiple threads simultaneously
and measures system resources (GPU VRAM, RAM, CPU/GPU utilization) during parallel execution.

Usage:
    python benchmarks/run_component_parallel_bench.py \
        --component core_clip \
        --video-path /path/to/video.mp4 \
        --threads 4 \
        --frames-count 10 \
        --triton-http-url http://localhost:8000 \
        [--wait-triton]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add DynamicBatch to path for ResourceMonitor
_dynamicbatch_path = Path(__file__).parent.parent.parent / "DynamicBatch"
sys.path.insert(0, str(_dynamicbatch_path))

from dynamicbatch.resource_monitor import ResourceMonitor
from dynamicbatch.system_probe import probe_cpu_mem_mb, probe_gpu_mem_mb

# Import enhanced resource monitor
_benchmarks_path = Path(__file__).parent
sys.path.insert(0, str(_benchmarks_path))

from resource_monitor_enhanced import EnhancedResourceMonitor


@dataclass
class ThreadResult:
    """Result from a single thread execution."""
    thread_id: int
    run_id: str
    success: bool
    duration_sec: float
    returncode: int
    stdout: str
    stderr: str
    component_timing: Dict[str, float]
    start_time: float
    end_time: float


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


def _get_resource_snapshot() -> ResourceSnapshot:
    """Get current resource snapshot."""
    cpu_mem = probe_cpu_mem_mb()
    gpu_mem = probe_gpu_mem_mb()

    # Get GPU utilization via nvidia-smi
    gpu_util = None
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
        cpu_util_pct=None,  # Will be tracked by monitor
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
    segmenter_data = Path(__file__).parent.parent / "Segmenter" / "data"
    
    if not segmenter_data.exists():
        return None
    
    video_dir = segmenter_data / video_id
    if video_dir.exists():
        frames_dir = video_dir / "video"
        metadata_path = frames_dir / "metadata.json"
        if metadata_path.exists():
            batch_files = list(frames_dir.glob("batch*.npy"))
            if batch_files:
                return str(frames_dir)
    
    return None


def _ensure_frames_dir(video_path: str, frames_dir: Optional[str], out_base: str) -> str:
    """Ensure frames_dir exists, create if needed via Segmenter."""
    if frames_dir and os.path.exists(os.path.join(frames_dir, "metadata.json")):
        return frames_dir

    video_id = Path(video_path).stem
    
    existing_frames_dir = _find_existing_frames_dir(video_id)
    if existing_frames_dir:
        print(f"[bench] Found existing frames_dir: {existing_frames_dir}")
        return existing_frames_dir
    
    frames_dir_target = os.path.join(out_base, video_id, "video")
    metadata_path = os.path.join(frames_dir_target, "metadata.json")
    if os.path.exists(metadata_path):
        batch_files = list(Path(frames_dir_target).glob("batch*.npy"))
        if batch_files:
            print(f"[bench] Found existing frames_dir in output: {frames_dir_target}")
            return frames_dir_target

    print(f"[bench] Running Segmenter to create frames_dir...")
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
        frame_indices = list(range(total_frames))
    elif frames_count is not None and frames_count > 0:
        frame_indices = list(range(min(frames_count, total_frames)))
    else:
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


def _run_core_clip_thread(
    thread_id: int,
    frames_dir: str,
    rs_path: str,
    run_id: str,
    triton_http_url: str,
    batch_size: int,
    component_args: Dict[str, Any],
) -> ThreadResult:
    """Run core_clip component in a thread."""
    core_clip_script = (
        Path(__file__).parent.parent / "VisualProcessor" / "core" / "model_process" / "core_clip" / "main.py"
    )
    if not core_clip_script.exists():
        raise FileNotFoundError(f"core_clip script not found: {core_clip_script}")

    triton_image_model_spec = component_args.get("triton_image_model_spec")
    triton_text_model_spec = component_args.get("triton_text_model_spec")
    triton_image_model_name = component_args.get("triton_image_model_name")
    triton_text_model_name = component_args.get("triton_text_model_name")
    triton_preprocess_preset = component_args.get("triton_preprocess_preset", "openai_clip_224")
    triton_image_datatype = component_args.get("triton_image_datatype", "UINT8")
    triton_text_datatype = component_args.get("triton_text_datatype", "INT64")
    triton_http_timeout_sec = component_args.get("triton_http_timeout_sec", 60.0)
    dp_models_root = component_args.get("dp_models_root")

    # Auto-detect preprocess preset from model name
    actual_model_name = triton_image_model_name
    if not actual_model_name:
        if not (triton_image_model_spec and triton_text_model_spec):
            actual_model_name = "clip_image_224"
    
    if actual_model_name:
        if "336" in actual_model_name and triton_preprocess_preset == "openai_clip_224":
            triton_preprocess_preset = "openai_clip_336"
        elif "448" in actual_model_name and triton_preprocess_preset == "openai_clip_224":
            triton_preprocess_preset = "openai_clip_448"

    # Create unique rs_path for this thread
    thread_rs_path = os.path.join(rs_path, run_id)
    os.makedirs(thread_rs_path, exist_ok=True)

    cmd = [
        sys.executable,
        str(core_clip_script),
        "--frames-dir",
        frames_dir,
        "--rs-path",
        thread_rs_path,
        "--runtime",
        "triton",
        "--batch-size",
        str(int(batch_size)),
        "--model-name",
        "ViT-B/32",
        "--triton-http-url",
        triton_http_url,
        "--triton-preprocess-preset",
        triton_preprocess_preset,
        "--triton-timeout-sec",
        str(float(triton_http_timeout_sec)),
        "--disable-text-cache",
    ]

    if triton_image_model_spec and triton_text_model_spec:
        cmd.extend(["--triton-image-model-spec", triton_image_model_spec])
        cmd.extend(["--triton-text-model-spec", triton_text_model_spec])
    elif triton_image_model_name and triton_text_model_name:
        cmd.extend(["--triton-image-model-name", triton_image_model_name])
        cmd.extend(["--triton-text-model-name", triton_text_model_name])
    else:
        cmd.extend(["--triton-image-model-name", "clip_image_224"])
        cmd.extend(["--triton-text-model-name", "clip_text"])
    
    cmd.extend(["--triton-image-datatype", triton_image_datatype])
    cmd.extend(["--triton-text-datatype", triton_text_datatype])

    env = os.environ.copy()
    if triton_http_url and "TRITON_HTTP_URL" not in env:
        env["TRITON_HTTP_URL"] = triton_http_url
    if dp_models_root:
        env["DP_MODELS_ROOT"] = dp_models_root

    start_time = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    end_time = time.perf_counter()
    duration_sec = end_time - start_time

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

    return ThreadResult(
        thread_id=thread_id,
        run_id=run_id,
        success=result.returncode == 0,
        duration_sec=duration_sec,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        component_timing=component_timing,
        start_time=start_time,
        end_time=end_time,
    )


def _run_core_depth_midas_thread(
    thread_id: int,
    frames_dir: str,
    rs_path: str,
    run_id: str,
    triton_http_url: str,
    batch_size: int,
    component_args: Dict[str, Any],
) -> ThreadResult:
    """Run core_depth_midas component in a thread."""
    core_depth_midas_script = (
        Path(__file__).parent.parent / "VisualProcessor" / "core" / "model_process" / "core_depth_midas" / "main.py"
    )
    if not core_depth_midas_script.exists():
        raise FileNotFoundError(f"core_depth_midas script not found: {core_depth_midas_script}")

    triton_model_spec = component_args.get("triton_model_spec")
    triton_model_name = component_args.get("triton_model_name")
    triton_preprocess_preset = component_args.get("triton_preprocess_preset", "midas_384")
    triton_datatype = component_args.get("triton_datatype", "UINT8")
    out_width = component_args.get("out_width", 384)
    out_height = component_args.get("out_height", 384)
    frames_bgr = component_args.get("frames_bgr", False)

    # Create unique rs_path for this thread
    thread_rs_path = os.path.join(rs_path, run_id)
    os.makedirs(thread_rs_path, exist_ok=True)

    cmd = [
        sys.executable,
        str(core_depth_midas_script),
        "--frames-dir",
        frames_dir,
        "--rs-path",
        thread_rs_path,
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

    if triton_model_spec:
        cmd.extend(["--triton-model-spec", triton_model_spec])
    elif triton_model_name:
        cmd.extend(["--triton-model-name", triton_model_name])
    else:
        cmd.extend(["--triton-model-name", "midas_384"])

    if frames_bgr:
        cmd.append("--frames-bgr")

    env = os.environ.copy()
    if triton_http_url and "TRITON_HTTP_URL" not in env:
        env["TRITON_HTTP_URL"] = triton_http_url

    start_time = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    end_time = time.perf_counter()
    duration_sec = end_time - start_time

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

    return ThreadResult(
        thread_id=thread_id,
        run_id=run_id,
        success=result.returncode == 0,
        duration_sec=duration_sec,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        component_timing=component_timing,
        start_time=start_time,
        end_time=end_time,
    )


def _generate_html_report(results: Dict[str, Any], out_path: str) -> None:
    """Generate HTML report for parallel benchmark."""
    threads = results.get("threads", [])
    num_threads = len(threads)
    successful = sum(1 for t in threads if t.get("success", False))
    failed = num_threads - successful

    # Calculate statistics
    durations = [t.get("duration_sec", 0) for t in threads if t.get("success", False)]
    avg_duration = sum(durations) / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0

    # Resource metrics
    resources = results.get("resources", {})
    peaks = resources.get("peaks", {})
    time_series = resources.get("time_series", [])

    # Prepare time series data for charts
    time_series_js = "[]"
    if time_series:
        cpu_data = []
        gpu_data = []
        cpu_mem_data = []
        gpu_mem_data = []
        labels = []
        for ts in time_series[:1000]:
            elapsed = ts.get("elapsed_sec", 0)
            labels.append(f"{elapsed:.2f}")
            cpu_data.append(ts.get("cpu_util_pct"))
            gpu_data.append(ts.get("gpu_util_pct"))
            cpu_mem_data.append(ts.get("cpu_mem_used_mb"))
            gpu_mem_data.append(ts.get("gpu_mem_used_mb"))
        
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
    <title>Parallel Component Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
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
        .chart-container {{
            margin: 20px 0;
            height: 300px;
            position: relative;
        }}
        .success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .failure {{
            color: #f44336;
            font-weight: bold;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>Parallel Component Benchmark Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}Z</p>

        <h2>Test Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Component</td><td>{results.get('component', 'N/A')}</td></tr>
            <tr><td>Video Path</td><td>{results.get('video_path', 'N/A')}</td></tr>
            <tr><td>Frames Count</td><td>{results.get('frames_count', 'N/A')}</td></tr>
            <tr><td>Batch Size</td><td>{results.get('batch_size', 'N/A')}</td></tr>
            <tr><td>Number of Threads</td><td>{num_threads}</td></tr>
            <tr><td>Successful Runs</td><td class="success">{successful}</td></tr>
            <tr><td>Failed Runs</td><td class="failure">{failed}</td></tr>
            <tr><td>Triton URL</td><td>{results.get('triton_http_url', 'N/A')}</td></tr>
        </table>

        <h2>Execution Statistics</h2>
        <div class="metric">
            <span class="metric-label">Average Duration:</span> {avg_duration:.3f} seconds<br>
            <span class="metric-label">Min Duration:</span> {min_duration:.3f} seconds<br>
            <span class="metric-label">Max Duration:</span> {max_duration:.3f} seconds<br>
            <span class="metric-label">Total Wall Time:</span> {results.get('total_wall_time_sec', 0):.3f} seconds<br>
            <span class="metric-label">Throughput:</span> {successful / results.get('total_wall_time_sec', 1):.2f} runs/second
        </div>

        <h2>Thread Results</h2>
        <table>
            <tr>
                <th>Thread ID</th>
                <th>Run ID</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Return Code</th>
            </tr>
"""
    for thread in threads:
        status_class = "success" if thread.get("success", False) else "failure"
        status_text = "✓ Success" if thread.get("success", False) else "✗ Failed"
        html += f"""
            <tr>
                <td>{thread.get('thread_id', 'N/A')}</td>
                <td>{thread.get('run_id', 'N/A')}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{thread.get('duration_sec', 0):.3f}</td>
                <td>{thread.get('returncode', 'N/A')}</td>
            </tr>
"""

    html += """
        </table>

        <h2>Peak Resource Usage</h2>
        <div class="metric">
            <span class="metric-label">Peak CPU Utilization:</span> {peaks.get('cpu_util_peak_pct', 'N/A')}%<br>
            <span class="metric-label">Peak RAM:</span> {peaks.get('ram_used_peak_mb', 'N/A')} MB<br>
            <span class="metric-label">Peak GPU Utilization:</span> {peaks.get('gpu_util_peak_pct', 'N/A')}%<br>
            <span class="metric-label">Peak GPU VRAM:</span> {peaks.get('vram_used_peak_mb', 'N/A')} MB
        </div>

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
                            text: 'CPU Utilization Over Time (Parallel Execution)'
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
                            text: 'GPU Utilization Over Time (Parallel Execution)'
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
                            text: 'Memory Usage Over Time (Parallel Execution)'
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
        description="Multi-threaded component benchmark with resource monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--component", type=str, required=True, help="Component name (e.g., core_clip)")
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
    parser.add_argument("--threads", type=int, required=True, help="Number of parallel threads")
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
        help="Output directory (default: benchmarks/out_component_parallel/<timestamp>)",
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
        choices=["openai_clip_224", "openai_clip_336", "openai_clip_448", "midas_256", "midas_384", "midas_512"],
        help="Triton preprocess preset (for core_clip: openai_clip_224/336/448, for core_depth_midas: midas_256/384/512)",
    )
    parser.add_argument(
        "--triton-http-timeout-sec",
        type=float,
        default=60.0,
        help="Triton HTTP client timeout in seconds",
    )
    parser.add_argument(
        "--triton-image-datatype",
        type=str,
        default="UINT8",
        choices=["UINT8", "FP32"],
        help="Triton image input datatype (default: UINT8)",
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
        help="DP_MODELS_ROOT environment variable",
    )
    # Arguments for core_depth_midas
    parser.add_argument(
        "--triton-model-spec",
        type=str,
        default=None,
        help="Triton model spec (for core_depth_midas, e.g., midas_384_triton)",
    )
    parser.add_argument(
        "--triton-model-name",
        type=str,
        default=None,
        help="Triton model name (for core_depth_midas, default: midas_384)",
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

    args = parser.parse_args()
    
    # Validate component
    if args.component not in ["core_clip", "core_depth_midas"]:
        raise ValueError(f"Unsupported component: {args.component}. Supported: core_clip, core_depth_midas")
    
    # Validate component-specific arguments
    if args.component == "core_depth_midas":
        # Update preprocess preset choices for midas
        if args.triton_preprocess_preset not in ["midas_256", "midas_384", "midas_512"]:
            args.triton_preprocess_preset = "midas_384"

    # Setup output directory
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(Path(__file__).parent, "out_component_parallel", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[bench] Output directory: {out_dir}")

    # Step 1: Measure resources BEFORE Triton startup
    print("[bench] Step 1: Measuring resources BEFORE Triton startup...")
    before_triton = _get_resource_snapshot()
    print(f"[bench] GPU VRAM before Triton: {before_triton.gpu_mem_used_mb}/{before_triton.gpu_mem_total_mb} MB")
    print(f"[bench] RAM before Triton: {before_triton.cpu_mem_used_mb}/{before_triton.cpu_mem_total_mb} MB")

    # Step 2: Wait for Triton if requested
    if args.wait_triton:
        print("[bench] Step 2: Waiting for Triton to become available...")
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
        
        print("[bench] Verifying Triton is available...")
        if not _wait_for_triton(args.triton_http_url, timeout_sec=10.0):
            print("[bench] WARNING: Triton does not appear to be available yet.")
            print("[bench] Continuing anyway, but benchmarks may fail.")
        else:
            print("[bench] ✓ Triton is available!")

    # Step 3: Measure resources AFTER Triton startup
    print("[bench] Step 3: Measuring resources AFTER Triton startup...")
    time.sleep(2.0)
    after_triton = _get_resource_snapshot()
    print(f"[bench] GPU VRAM after Triton: {after_triton.gpu_mem_used_mb}/{after_triton.gpu_mem_total_mb} MB")
    print(f"[bench] RAM after Triton: {after_triton.cpu_mem_used_mb}/{after_triton.cpu_mem_total_mb} MB")

    # Step 4: Prepare frames_dir and frame_indices
    print("[bench] Step 4: Preparing frames_dir...")
    frames_dir = _ensure_frames_dir(args.video_path, args.frames_dir, out_dir)
    frame_indices = _prepare_frame_indices(frames_dir, args.frames_count, args.full_video)
    _update_metadata_frame_indices(frames_dir, args.component, frame_indices)
    print(f"[bench] Processing {len(frame_indices)} frame(s): {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    # Step 5: Start resource monitoring and run components in parallel
    print(f"[bench] Step 5: Running {args.threads} component instances in parallel with resource monitoring...")
    monitor = EnhancedResourceMonitor(interval_sec=0.25)
    monitor.start()

    rs_path = os.path.join(out_dir, "result_store")
    os.makedirs(rs_path, exist_ok=True)

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
    else:
        component_args = {}

    # Run components in parallel
    total_start_time = time.perf_counter()
    thread_results: List[ThreadResult] = []
    
    # Select the appropriate thread function based on component
    if args.component == "core_clip":
        thread_func = _run_core_clip_thread
    elif args.component == "core_depth_midas":
        thread_func = _run_core_depth_midas_thread
    else:
        raise ValueError(f"Unsupported component: {args.component}")
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for thread_id in range(args.threads):
            run_id = f"thread_{thread_id}_{uuid.uuid4().hex[:8]}"
            future = executor.submit(
                thread_func,
                thread_id=thread_id,
                frames_dir=frames_dir,
                rs_path=rs_path,
                run_id=run_id,
                triton_http_url=args.triton_http_url,
                batch_size=args.batch_size,
                component_args=component_args,
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                thread_results.append(result)
                status = "✓" if result.success else "✗"
                print(f"[bench] Thread {result.thread_id} ({result.run_id}): {status} ({result.duration_sec:.3f}s)")
            except Exception as e:
                print(f"[bench] Thread failed with exception: {e}")
    
    total_end_time = time.perf_counter()
    total_wall_time = total_end_time - total_start_time

    monitor.stop(timeout_sec=2.0)
    peaks = monitor.peaks.to_dict()
    time_series = [ts.to_dict() for ts in monitor.time_series]
    peak_timestamps = monitor.peak_timestamps

    # Step 6: Measure resources AFTER execution
    print("[bench] Step 6: Measuring resources AFTER parallel execution...")
    time.sleep(1.0)
    after_execution = _get_resource_snapshot()
    print(f"[bench] GPU VRAM after execution: {after_execution.gpu_mem_used_mb}/{after_execution.gpu_mem_total_mb} MB")
    print(f"[bench] RAM after execution: {after_execution.cpu_mem_used_mb}/{after_execution.cpu_mem_total_mb} MB")

    # Step 7: Generate reports
    print("[bench] Step 7: Generating reports...")

    successful_threads = [r for r in thread_results if r.success]
    durations = [r.duration_sec for r in successful_threads]
    avg_duration = sum(durations) / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0

    results = {
        "component": args.component,
        "video_path": args.video_path,
        "frames_dir": frames_dir,
        "frames_count": len(frame_indices),
        "frame_indices": frame_indices,
        "batch_size": args.batch_size,
        "num_threads": args.threads,
        "triton_http_url": args.triton_http_url,
        "total_wall_time_sec": total_wall_time,
        "threads": [asdict(r) for r in thread_results],
        "statistics": {
            "successful": len(successful_threads),
            "failed": len(thread_results) - len(successful_threads),
            "avg_duration_sec": avg_duration,
            "min_duration_sec": min_duration,
            "max_duration_sec": max_duration,
            "throughput_runs_per_sec": len(successful_threads) / total_wall_time if total_wall_time > 0 else 0,
        },
        "resources": {
            "before_triton": before_triton.to_dict(),
            "after_triton": after_triton.to_dict(),
            "after_execution": after_execution.to_dict(),
            "peaks": peaks,
            "time_series": time_series,
            "peak_timestamps": peak_timestamps,
        },
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

    # Summary
    print("\n" + "=" * 80)
    print("PARALLEL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Component: {args.component}")
    print(f"Threads: {args.threads}")
    print(f"Frames per thread: {len(frame_indices)}")
    print(f"Successful: {len(successful_threads)}/{len(thread_results)}")
    print(f"Total wall time: {total_wall_time:.3f} seconds")
    print(f"Average duration per thread: {avg_duration:.3f} seconds")
    print(f"Min duration: {min_duration:.3f} seconds")
    print(f"Max duration: {max_duration:.3f} seconds")
    print(f"Throughput: {len(successful_threads) / total_wall_time:.2f} runs/second")
    print(f"\nPeak Resource Usage:")
    print(f"  CPU utilization: {peaks.get('cpu_util_peak_pct', 'N/A')}%")
    print(f"  RAM: {peaks.get('ram_used_peak_mb', 'N/A')} MB")
    print(f"  GPU utilization: {peaks.get('gpu_util_peak_pct', 'N/A')}%")
    print(f"  GPU VRAM: {peaks.get('vram_used_peak_mb', 'N/A')} MB")
    print("=" * 80)
    print(f"\nReports saved to: {out_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")


if __name__ == "__main__":
    main()

