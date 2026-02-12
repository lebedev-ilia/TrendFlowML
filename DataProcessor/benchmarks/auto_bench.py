#!/usr/bin/env python3
"""
Automated benchmark runner for core_optical_flow and core_object_detections components.

This script:
1. Cleans system resources
2. Starts Triton server in Docker (detached mode)
3. Waits for Triton to be ready
4. Runs benchmarks with different batch sizes and frame counts for both components
5. Aggregates results and prints summary

Usage:
    python benchmarks/auto_bench.py
"""

import json
import os
import time
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple

# Configuration
ROOT = Path(__file__).parent.parent
PYTHON_PATH = ROOT / "VisualProcessor" / ".vp_venv" / "bin" / "python3"
CLEANUP_SCRIPT = ROOT / "benchmarks" / "cleanup_system.py"
BENCH_SCRIPT = ROOT / "benchmarks" / "run_component_bench.py"
AGGREGATE_SCRIPT = ROOT / "benchmarks" / "aggregate_benchmark_results.py"
OUT_DIR = ROOT / "benchmarks" / "out_component"
SUMMARY_DIR = ROOT / "benchmarks" / "summary"

# Benchmark configuration
# Components to benchmark
COMPONENTS = [
    {
        "name": "core_optical_flow",
        "presets": ["raft_256", "raft_384", "raft_512"],
        "frames_list": [2, 4, 8, 32, 64, 100, 304],
        "models_dir_pattern": "models_{preset}",  # e.g., models_raft_256
    },
    # {
    #     "name": "core_object_detections",
    #     "presets": ["yolo11x_320", "yolo11x_640", "yolo11x_960"],
    #     "frames_list": [1, 5, 10, 50, 100, 304],
    #     "models_dir_pattern": "models",  # All YOLO models in triton/models/
    # },
]

VIDEO_PATH = "example/example_videos/-F71yZij1Uc.mp4"
TRITON_URL = "http://localhost:8000"
BATCHES = [1, 8, 16]
ATTEMPTS = 4

# For core_object_detections additional parameters
OD_BOX_THRESHOLD = 0.6
OD_IOU_THRESHOLD = 0.3

# Docker configuration
DOCKER_IMAGE = "nvcr.io/nvidia/tritonserver:24.08-py3"
DOCKER_CONTAINER_NAME = "triton-bench"


def wait_for_triton(url: str, timeout_sec: float = 300.0, check_interval_sec: float = 2.0) -> bool:
    """Wait for Triton to become available."""
    health_url = f"{url.rstrip('/')}/v2/health/ready"
    deadline = time.time() + timeout_sec
    
    print(f"[auto_bench] Waiting for Triton at {url}...")
    start_time = time.time()
    
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start_time
                    print(f"[auto_bench] ✓ Triton is ready! (took {elapsed:.1f}s)")
                    return True
        except Exception:
            pass
        
        time.sleep(check_interval_sec)
        elapsed = time.time() - start_time
        if int(elapsed) % 10 < check_interval_sec and elapsed > 0:
            remaining = int(deadline - time.time())
            print(f"[auto_bench] Still waiting for Triton... ({remaining}s remaining)")
    
    print(f"[auto_bench] ERROR: Triton did not become ready within {timeout_sec}s")
    return False


def refresh_sudo_cache() -> bool:
    """
    Продлевает кеш sudo пароля, чтобы не запрашивать его повторно.
    Использует 'sudo -v' для обновления timestamp.
    """
    try:
        # sudo -v обновляет timestamp кеша без выполнения команды
        result = subprocess.run(
            ["sudo", "-v"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return True
    except subprocess.CalledProcessError:
        # Если sudo -v не сработал (например, пароль не был введен ранее),
        # это не критично - просто попробуем выполнить команду
        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def cleanup_system() -> bool:
    """Run system cleanup script."""
    # Продлеваем кеш sudo перед каждым cleanup, чтобы не запрашивать пароль
    refresh_sudo_cache()
    
    print("[auto_bench] Cleaning system...")
    try:
        cmd = ["sudo", str(PYTHON_PATH), str(CLEANUP_SCRIPT), "--force"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[auto_bench] ✓ Cleanup successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[auto_bench] ERROR: Cleanup failed: {e.stderr}")
        return False


def start_triton_container(models_dir: Path) -> Optional[str]:
    """
    Start Triton server in Docker (detached mode). Returns container ID.
    
    Uses the main models directory (triton/models) which contains all models
    (RAFT, YOLO, etc.) for both components.
    """
    # Stop existing container if running
    print("[auto_bench] Stopping existing Triton container (if any)...")
    subprocess.run(
        ["docker", "stop", DOCKER_CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "rm", DOCKER_CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    print("[auto_bench] Starting Triton server in Docker...")
    # Убеждаемся, что путь абсолютный и существует
    models_dir_abs = models_dir.resolve()
    if not models_dir_abs.exists():
        print(f"[auto_bench] ERROR: Models directory does not exist: {models_dir_abs}")
        return None
    
    print(f"[auto_bench] Using models directory: {models_dir_abs}")
    cmd = [
        "docker", "run",
        "-d",  # detached mode
        "--name", DOCKER_CONTAINER_NAME,
        "--rm",
        "--gpus", "all",
        "--shm-size=1g",
        "-p", "8000:8000",
        "-p", "8001:8001",
        "-p", "8002:8002",
        "-v", f"{models_dir_abs}:/models:ro",
        DOCKER_IMAGE,
        "tritonserver",
        "--model-repository=/models",
    ]
    
    # Выводим команду для отладки
    print(f"[auto_bench] Docker command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        container_id = result.stdout.strip()
        print(f"[auto_bench] ✓ Triton container started: {container_id[:12]}")
        
        # Проверяем, что контейнер действительно запущен и не упал сразу
        time.sleep(2)  # Даём контейнеру время на запуск
        check_cmd = ["docker", "ps", "-q", "-f", f"id={container_id}"]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True)
        if not check_result.stdout.strip():
            # Контейнер не запущен, проверяем логи
            print(f"[auto_bench] WARNING: Container {container_id[:12]} is not running. Checking logs...")
            logs_cmd = ["docker", "logs", container_id]
            logs_result = subprocess.run(logs_cmd, capture_output=True, text=True, timeout=5)
            if logs_result.stdout:
                print(f"[auto_bench] Container logs (stdout):\n{logs_result.stdout[-1000:]}")
            if logs_result.stderr:
                print(f"[auto_bench] Container logs (stderr):\n{logs_result.stderr[-1000:]}")
            return None
        
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"[auto_bench] ERROR: Failed to start Docker container")
        if e.stdout:
            print(f"[auto_bench] stdout: {e.stdout}")
        if e.stderr:
            print(f"[auto_bench] stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"[auto_bench] ERROR: Unexpected error starting container: {e}")
        return None


def stop_triton_container() -> bool:
    """Stop Triton Docker container."""
    print("[auto_bench] Stopping Triton container...")
    try:
        subprocess.run(
            ["docker", "stop", DOCKER_CONTAINER_NAME],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[auto_bench] ✓ Triton container stopped")
        return True
    except subprocess.CalledProcessError:
        print("[auto_bench] WARNING: Container was not running")
        return False


def run_benchmark(
    component: str,
    batch_size: int,
    frames_count: int,
    attempt: int,
    preprocess_preset: str,
    wait_triton: bool = False,
    **component_kwargs,
) -> Tuple[bool, bool]:
    """
    Run a single benchmark.

    Args:
        component: Component name (core_optical_flow or core_object_detections)
        batch_size: Batch size
        frames_count: Number of frames
        attempt: Attempt number
        preprocess_preset: Preprocess preset (e.g., raft_256, yolo11x_640)
        wait_triton: Whether to wait for Triton automatically
        **component_kwargs: Additional component-specific arguments

    Returns:
        (success, is_oom)
    """
    print(
        f"[auto_bench] Running benchmark: component={component}, prep={preprocess_preset}, "
        f"batch={batch_size}, frames={frames_count}, attempt={attempt+1}/{ATTEMPTS}"
    )
    
    cmd = [
        str(PYTHON_PATH),
        str(BENCH_SCRIPT),
        "--component", component,
        "--video-path", VIDEO_PATH,
        "--triton-http-url", TRITON_URL,
        # We start Triton ourselves and also pass a shorter timeout for safety
        "--triton-timeout", "30.0",
        "--triton-preprocess-preset", preprocess_preset,
        "--batch-size", str(batch_size),
        "--frames-count", str(frames_count),
    ]
    
    if wait_triton:
        cmd.append("--wait-triton")
    
    # Add component-specific arguments
    if component == "core_object_detections":
        cmd.extend(["--od-triton-preprocess-preset", preprocess_preset])
        cmd.extend(["--od-box-threshold", str(component_kwargs.get("box_threshold", OD_BOX_THRESHOLD))])
        cmd.extend(["--od-iou-threshold", str(component_kwargs.get("iou_threshold", OD_IOU_THRESHOLD))])
        if component_kwargs.get("triton_model_spec"):
            cmd.extend(["--triton-model-spec", component_kwargs["triton_model_spec"]])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(ROOT))
        
        # Проверяем, что results.json действительно создан
        # Ищем последнюю созданную директорию в out_component
        out_component_dir = ROOT / "benchmarks" / "out_component"
        if out_component_dir.exists():
            result_dirs = sorted([d for d in out_component_dir.iterdir() if d.is_dir()])
            if result_dirs:
                latest_dir = result_dirs[-1]
                results_json = latest_dir / "results.json"
                if not results_json.exists():
                    print(f"[auto_bench] WARNING: Benchmark completed but results.json not found in {latest_dir}")
                    print(f"[auto_bench] This may indicate the benchmark failed before saving results")
                    if result.stdout:
                        print(f"[auto_bench] stdout (last 500 chars): {result.stdout[-500:]}")
                    if result.stderr:
                        print(f"[auto_bench] stderr (last 500 chars): {result.stderr[-500:]}")
                    return False, False
        
        print(f"[auto_bench] ✓ Benchmark completed successfully")
        return True, False
    except subprocess.CalledProcessError as e:
        print(f"[auto_bench] ERROR: Benchmark failed with returncode {e.returncode}")
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if e.stdout:
            print(f"[auto_bench] stdout (last 1000 chars):\n{e.stdout[-1000:]}")
        if e.stderr:
            print(f"[auto_bench] stderr (last 1000 chars):\n{e.stderr[-1000:]}")
        
        # Проверяем, создалась ли директория, но без results.json
        out_component_dir = ROOT / "benchmarks" / "out_component"
        if out_component_dir.exists():
            result_dirs = sorted([d for d in out_component_dir.iterdir() if d.is_dir()])
            if result_dirs:
                latest_dir = result_dirs[-1]
                results_json = latest_dir / "results.json"
                if not results_json.exists():
                    print(f"[auto_bench] Directory {latest_dir.name} was created but results.json is missing")
                    # Показываем содержимое директории
                    files = list(latest_dir.iterdir())
                    if files:
                        print(f"[auto_bench] Files in directory: {[f.name for f in files]}")
                    else:
                        print(f"[auto_bench] Directory is empty")
        
        # Best-effort OOM detection
        combined = (stdout + "\n" + stderr).lower()
        oom_markers = [
            "cuda out of memory",
            "cuda_error_out_of_memory",
            "out of memory",
            "resourceexhaustederror",
            "memory allocation failed",
        ]
        is_oom = any(m in combined for m in oom_markers)
        if is_oom:
            print("[auto_bench] Detected potential OOM condition.")
        return False, is_oom


def aggregate_results(name: str) -> Optional[dict]:
    """Aggregate benchmark results and return summary data."""
    print(f"[auto_bench] Aggregating results for {name}...")
    
    if not OUT_DIR.exists():
        print(f"[auto_bench] WARNING: Output directory {OUT_DIR} does not exist")
        return None
    
    cmd = [
        str(PYTHON_PATH),
        str(AGGREGATE_SCRIPT),
        str(OUT_DIR),
        name,
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(ROOT))
        
        # Read aggregated results
        summary_file = SUMMARY_DIR / f"res_{name}.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[auto_bench] ✓ Aggregation successful")
            return data
        else:
            print(f"[auto_bench] WARNING: Summary file {summary_file} not found")
            return None
    except subprocess.CalledProcessError as e:
        print(f"[auto_bench] ERROR: Aggregation failed: {e.stderr}")
        return None


def cleanup_output_dir() -> None:
    """Remove output directory after aggregation."""
    if OUT_DIR.exists():
        print(f"[auto_bench] Cleaning output directory...")
        subprocess.run(["rm", "-rf", str(OUT_DIR)], check=False)
        print(f"[auto_bench] ✓ Output directory cleaned")


def print_summary(data: dict, name: str) -> None:
    """Print summary of benchmark results."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {name}")
    print(f"{'='*80}")
    
    metrics = [
        "Duration (s)",
        "Peak CPU %",
        "Peak GPU %",
        "Component Delta VRAM (MB)",
        "Component Delta RAM (MB)",
    ]
    
    for metric in metrics:
        value = data.get(metric, "N/A")
        print(f"  {metric}: {value}")
    
    print(f"{'='*80}\n")


def main() -> None:
    """Main benchmark execution loop."""
    print("="*80)
    print("AUTOMATED BENCHMARK RUNNER")
    print("="*80)
    print(f"Components: {', '.join([c['name'] for c in COMPONENTS])}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Batches: {BATCHES}")
    print(f"Attempts per configuration: {ATTEMPTS}")
    print("="*80)
    
    # Validate Python path
    if not PYTHON_PATH.exists():
        print(f"ERROR: Python not found at {PYTHON_PATH}")
        sys.exit(1)
    
    # Продлеваем кеш sudo в начале, чтобы не запрашивать пароль во время долгого выполнения
    print("[auto_bench] Refreshing sudo cache...")
    if refresh_sudo_cache():
        print("[auto_bench] ✓ Sudo cache refreshed (will not ask for password during execution)")
    else:
        print("[auto_bench] ⚠ Could not refresh sudo cache (may ask for password later)")
    
    # Create summary directory
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks for each component
    for component_config in COMPONENTS:
        component_name = component_config["name"]
        presets = component_config["presets"]
        frames_list = component_config["frames_list"]
        models_dir_pattern = component_config.get("models_dir_pattern", "models")
        
        print(f"\n{'#'*80}")
        print(f"COMPONENT: {component_name}")
        print(f"{'#'*80}")
        print(f"Presets: {presets}")
        print(f"Frames: {frames_list}")
        
        # Run benchmarks for each preset
        for prep in presets:
            print(f"\n{'='*80}")
            print(f"PRESET: {prep}")
            print(f"{'='*80}")
            
            # Определяем директорию моделей для этого preset
            if "{preset}" in models_dir_pattern:
                models_dir_name = models_dir_pattern.format(preset=prep)
            else:
                models_dir_name = models_dir_pattern
            models_dir = ROOT / "triton" / models_dir_name
            
            if not models_dir.exists():
                print(f"[auto_bench] ERROR: Models directory not found: {models_dir}")
                print(f"[auto_bench] Skipping preset {prep}")
                continue
            
            print(f"[auto_bench] Using models directory: {models_dir}")
            
            for batch in BATCHES:
                for frames in frames_list:
                    print(f"\n{'='*80}")
                    print(f"CONFIGURATION: component={component_name}, prep={prep}, batch={batch}, frames={frames}")
                    print(f"{'='*80}")
                    
                    # Run multiple attempts
                    for attempt in range(ATTEMPTS):
                        print(f"\n[auto_bench] Attempt {attempt+1}/{ATTEMPTS}")
                        
                        # Step 1: Clean system before each attempt
                        if not cleanup_system():
                            print(f"[auto_bench] WARNING: Cleanup failed for attempt {attempt+1}, continuing...")
                        
                        time.sleep(5)  # Give system time to stabilize
                        
                        # Step 2: Stop existing Triton container (if any)
                        stop_triton_container()
                        time.sleep(1)  # Brief pause
                        
                        # Step 3: Start Triton container with preset-specific models directory
                        container_id = start_triton_container(models_dir=models_dir)
                        if not container_id:
                            print(f"[auto_bench] ERROR: Failed to start Triton for attempt {attempt+1}, skipping...")
                            continue
                        
                        # Step 4: Wait for Triton to be ready
                        print(f"[auto_bench] Waiting for Triton to be ready...")
                        if not wait_for_triton(TRITON_URL, timeout_sec=120.0):
                            print(f"[auto_bench] ERROR: Triton did not become ready for attempt {attempt+1}, skipping...")
                            stop_triton_container()
                            continue
                        
                        # Adaptive batch size for OOM protection
                        current_batch = batch
                        tried_batches = set()
                        
                        # Prepare component-specific kwargs
                        component_kwargs = {}
                        if component_name == "core_object_detections":
                            # Use yolo11x_640_triton spec (or could be made configurable)
                            component_kwargs["triton_model_spec"] = "yolo11x_640_triton"
                            component_kwargs["box_threshold"] = OD_BOX_THRESHOLD
                            component_kwargs["iou_threshold"] = OD_IOU_THRESHOLD
                        
                        try:
                            while True:
                                if current_batch in tried_batches:
                                    # Avoid infinite loops
                                    print(f"[auto_bench] WARNING: Batch size {current_batch} already tried, giving up.")
                                    break
                                tried_batches.add(current_batch)
                                
                                success, is_oom = run_benchmark(
                                    component=component_name,
                                    batch_size=current_batch,
                                    frames_count=frames,
                                    attempt=attempt,
                                    preprocess_preset=prep,
                                    wait_triton=True,  # Let benchmark wait for Triton automatically
                                    **component_kwargs,
                                )
                                
                                if success:
                                    break
                                
                                # If OOM and batch > 1, reduce batch size and retry
                                if is_oom and current_batch > 1:
                                    next_batch = max(1, current_batch // 2)
                                    if next_batch == current_batch:
                                        next_batch = 1
                                    print(
                                        f"[auto_bench] OOM detected for batch={current_batch}. "
                                        f"Retrying with smaller batch={next_batch}."
                                    )
                                    current_batch = next_batch
                                    continue
                                
                                # Non-OOM failure or already at batch=1
                                print(
                                    f"[auto_bench] WARNING: Attempt {attempt+1} failed "
                                    f"for component={component_name}, prep={prep}, "
                                    f"batch={current_batch}, frames={frames}."
                                )
                                break
                        finally:
                            # Step 6: Stop Triton container after each attempt
                            stop_triton_container()
                            time.sleep(1)  # Brief pause before next attempt
                    
                    # Aggregate results after all attempts for this configuration
                    name = f"{component_name}_{prep}_{batch}_{frames}"
                    
                    # Проверяем, есть ли хотя бы один results.json перед агрегацией
                    out_component_dir = ROOT / "benchmarks" / "out_component"
                    has_results = False
                    if out_component_dir.exists():
                        result_dirs = sorted([d for d in out_component_dir.iterdir() if d.is_dir()])
                        for result_dir in result_dirs:
                            results_json = result_dir / "results.json"
                            if results_json.exists():
                                has_results = True
                                break
                    
                    if has_results:
                        data = aggregate_results(name)
                        if data:
                            print_summary(data, name)
                        else:
                            print(f"[auto_bench] WARNING: Could not aggregate results for {name}")
                    else:
                        print(f"[auto_bench] WARNING: No successful attempts found for {name}. Skipping aggregation.")
                        print(f"[auto_bench] All {ATTEMPTS} attempts failed before saving results.json")
                    
                    # Cleanup output directory
                    cleanup_output_dir()
                    
                    # Small delay between configurations
                    time.sleep(2)
    
    print("\n[auto_bench] ✓ All benchmarks completed!")


if __name__ == "__main__":
    main()
