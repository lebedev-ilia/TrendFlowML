#!/usr/bin/env python3
"""
Скрипт для прогона всех компонентов core с Triton моделями и генерации HTML отчета.

Использование:
    python run_all_core_components.py \
        --frames-dir /path/to/frames_dir \
        --rs-path /path/to/result_store \
        --triton-http-url http://localhost:8000 \
        --out-dir /path/to/output \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger

logger = get_logger("run_all_core_components")


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources."""
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
class ComponentResult:
    """Result of running a component."""
    name: str
    success: bool
    duration_sec: float
    error: Optional[str] = None
    before_snapshot: Optional[ResourceSnapshot] = None
    after_snapshot: Optional[ResourceSnapshot] = None
    peak_ram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    ram_delta_mb: Optional[float] = None
    vram_delta_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "duration_sec": self.duration_sec,
            "error": self.error,
            "before_snapshot": self.before_snapshot.to_dict() if self.before_snapshot else None,
            "after_snapshot": self.after_snapshot.to_dict() if self.after_snapshot else None,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "ram_delta_mb": self.ram_delta_mb,
            "vram_delta_mb": self.vram_delta_mb,
        }


def probe_cpu_mem_mb() -> Optional[Tuple[int, int, int]]:
    """Probe CPU memory usage in MB. Returns (total, used, free)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            lines = f.readlines()
            total = None
            available = None
            for line in lines:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024  # KB to MB
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) // 1024  # KB to MB
                if total is not None and available is not None:
                    break
            if total is not None and available is not None:
                used = total - available
                return (total, used, available)
    except Exception:
        pass
    return None


def probe_gpu_mem_mb() -> Optional[Tuple[int, int, int]]:
    """Probe GPU memory usage in MB. Returns (total, used, free)."""
    import shutil
    if not shutil.which("nvidia-smi"):
        return None
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,nounits,noheader",
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if p.returncode == 0 and p.stdout:
            parts = p.stdout.strip().splitlines()[0].split(",")
            if len(parts) >= 3:
                total = int(parts[0].strip())
                used = int(parts[1].strip())
                free = int(parts[2].strip())
                return (total, used, free)
    except Exception:
        pass
    return None


def get_resource_snapshot() -> ResourceSnapshot:
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
        cpu_mem_total_mb=cpu_mem[0] if cpu_mem else None,
        cpu_mem_used_mb=cpu_mem[1] if cpu_mem else None,
        cpu_mem_free_mb=cpu_mem[2] if cpu_mem else None,
        gpu_mem_total_mb=gpu_mem[0] if gpu_mem else None,
        gpu_mem_used_mb=gpu_mem[1] if gpu_mem else None,
        gpu_mem_free_mb=gpu_mem[2] if gpu_mem else None,
        gpu_util_pct=gpu_util,
    )


class ResourceMonitor:
    """Monitor resources during component execution."""
    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak_ram_mb: Optional[float] = None
        self._peak_vram_mb: Optional[float] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start monitoring."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Monitor loop."""
        while not self._stop.wait(self.interval_sec):
            cpu_mem = probe_cpu_mem_mb()
            gpu_mem = probe_gpu_mem_mb()
            with self._lock:
                if cpu_mem and cpu_mem[1] is not None:
                    if self._peak_ram_mb is None or cpu_mem[1] > self._peak_ram_mb:
                        self._peak_ram_mb = float(cpu_mem[1])
                if gpu_mem and gpu_mem[1] is not None:
                    if self._peak_vram_mb is None or gpu_mem[1] > self._peak_vram_mb:
                        self._peak_vram_mb = float(gpu_mem[1])

    @property
    def peak_ram_mb(self) -> Optional[float]:
        with self._lock:
            return self._peak_ram_mb

    @property
    def peak_vram_mb(self) -> Optional[float]:
        with self._lock:
            return self._peak_vram_mb


def run_component(
    component_name: str,
    frames_dir: str,
    rs_path: str,
    triton_http_url: str,
    batch_size: int,
    component_args: Dict[str, Any],
) -> ComponentResult:
    """Run a single component and return result."""
    logger.info(f"Running component: {component_name}")

    # Get initial snapshot
    before_snapshot = get_resource_snapshot()
    initial_ram = before_snapshot.cpu_mem_used_mb or 0
    initial_vram = before_snapshot.gpu_mem_used_mb or 0

    # Start resource monitor
    monitor = ResourceMonitor()
    monitor.start()

    # Build command
    component_scripts = {
        "core_clip": "core_clip/main.py",
        "core_object_detections": "core_object_detections/main.py",
        "core_optical_flow": "core_optical_flow/main.py",
        "core_depth_midas": "core_depth_midas/main.py",
        "core_face_landmarks": "core_face_landmarks/main.py",
        "ocr_extractor": "ocr_extractor/main.py",
        "brand_semantics": "core_identity/brand_semantics/main.py",
        "car_semantics": "core_identity/car_semantics/main.py",
        "content_domain": "core_identity/content_domain/main.py",
        "face_identity": "core_identity/face_identity/main.py",
        "franchise_recognition": "core_identity/franchise_recognition/main.py",
        "place_semantics": "core_identity/place_semantics/main.py",
    }

    script_path = Path(__file__).parent / component_scripts[component_name]
    if not script_path.exists():
        monitor.stop()
        return ComponentResult(
            name=component_name,
            success=False,
            duration_sec=0.0,
            error=f"Script not found: {script_path}",
        )

    # Determine Python executable - use venv for core_face_landmarks
    python_exec = sys.executable
    if component_name == "core_face_landmarks":
        venv_path = Path(__file__).parent / "core_face_landmarks" / ".core_face_landmarks_venv"
        venv_python = venv_path / "bin" / "python"
        if venv_python.exists():
            python_exec = str(venv_python)
            logger.info(f"Using venv Python for {component_name}: {python_exec}")
        else:
            logger.warning(f"Venv Python not found at {venv_python}, using system Python")

    cmd = [
        python_exec,
        str(script_path),
        "--frames-dir", frames_dir,
        "--rs-path", rs_path,
    ]

    # Add component-specific arguments
    # Note: core_face_landmarks and identity components don't accept --batch-size
    components_without_batch_size = {
        "core_face_landmarks",
        "ocr_extractor",
        "brand_semantics",
        "car_semantics",
        "content_domain",
        "face_identity",
        "franchise_recognition",
        "place_semantics",
    }
    if component_name not in components_without_batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    if component_name == "core_clip":
        cmd.extend(["--runtime", "triton"])
        # Audit v3: ModelManager-only specs (no legacy Triton args)
        if not component_args.get("triton_image_model_spec") or not component_args.get("triton_text_model_spec"):
            raise ValueError(
                "core_clip requires both triton_image_model_spec and triton_text_model_spec (ModelManager-only)"
            )
        cmd.extend(["--triton-image-model-spec", component_args["triton_image_model_spec"]])
        cmd.extend(["--triton-text-model-spec", component_args["triton_text_model_spec"]])
        cmd.extend(["--triton-preprocess-preset", component_args.get("triton_preprocess_preset", "openai_clip_224")])
    elif component_name == "core_object_detections":
        # core_object_detections uses ultralytics runtime (not triton) in baseline
        cmd.extend(["--runtime", "ultralytics"])
        # Add model path for YOLO model
        if component_args.get("model_path"):
            cmd.extend(["--model", component_args["model_path"]])
        # Device selection
        if component_args.get("device"):
            cmd.extend(["--device", component_args["device"]])
    elif component_name == "core_optical_flow":
        cmd.extend(["--runtime", "triton"])
        cmd.extend(["--triton-http-url", triton_http_url])
        if component_args.get("triton_model_spec"):
            cmd.extend(["--triton-model-spec", component_args["triton_model_spec"]])
        else:
            cmd.extend(["--triton-model-name", component_args.get("triton_model_name", "raft_256")])
            if component_args.get("triton_model_version"):
                cmd.extend(["--triton-model-version", component_args["triton_model_version"]])
        cmd.extend(["--triton-datatype", component_args.get("triton_datatype", "UINT8")])
        cmd.extend(["--triton-preprocess-preset", component_args.get("triton_preprocess_preset", "raft_256")])
    elif component_name == "core_depth_midas":
        cmd.extend(["--runtime", "triton"])
        cmd.extend(["--triton-http-url", triton_http_url])
        if component_args.get("triton_model_spec"):
            cmd.extend(["--triton-model-spec", component_args["triton_model_spec"]])
        else:
            cmd.extend(["--triton-model-name", component_args.get("triton_model_name", "midas_256")])
            if component_args.get("triton_model_version"):
                cmd.extend(["--triton-model-version", component_args["triton_model_version"]])
        cmd.extend(["--triton-datatype", component_args.get("triton_datatype", "UINT8")])
        cmd.extend(["--triton-preprocess-preset", component_args.get("triton_preprocess_preset", "midas_256")])
    elif component_name == "core_face_landmarks":
        # core_face_landmarks doesn't use Triton, uses MediaPipe
        # Required flags for baseline
        cmd.extend(["--use-face-mesh"])
        cmd.extend(["--use-person-mask"])
        # Optional flags (can be enabled via component_args if needed)
        if component_args.get("use_pose"):
            cmd.extend(["--use-pose"])
        if component_args.get("use_hands"):
            cmd.extend(["--use-hands"])
    elif component_name == "ocr_extractor":
        # ocr_extractor doesn't use Triton, uses tesseract
        # Optional arguments
        if component_args.get("engine"):
            cmd.extend(["--engine", str(component_args["engine"])])
        if component_args.get("rec_model_spec"):
            cmd.extend(["--rec-model-spec", str(component_args["rec_model_spec"])])
        if component_args.get("ppocr_img_h") is not None:
            cmd.extend(["--ppocr-img-h", str(component_args["ppocr_img_h"])])
        if component_args.get("ppocr_img_w") is not None:
            cmd.extend(["--ppocr-img-w", str(component_args["ppocr_img_w"])])
        if component_args.get("min_rec_score") is not None:
            cmd.extend(["--min-rec-score", str(component_args["min_rec_score"])])
        if component_args.get("proposal_class"):
            cmd.extend(["--proposal-class", str(component_args["proposal_class"])])
        if component_args.get("min_det_score") is not None:
            cmd.extend(["--min-det-score", str(component_args["min_det_score"])])
        if component_args.get("max_boxes_per_frame") is not None:
            cmd.extend(["--max-boxes-per-frame", str(component_args["max_boxes_per_frame"])])
        if component_args.get("max_total_boxes") is not None:
            cmd.extend(["--max-total-boxes", str(component_args["max_total_boxes"])])
        if component_args.get("crop_margin_frac") is not None:
            cmd.extend(["--crop-margin-frac", str(component_args["crop_margin_frac"])])
        if component_args.get("tesseract_lang"):
            cmd.extend(["--tesseract-lang", component_args["tesseract_lang"]])
        if component_args.get("tesseract_psm"):
            cmd.extend(["--tesseract-psm", str(component_args["tesseract_psm"])])
        if bool(component_args.get("retain_raw_ocr_text")):
            cmd.extend(["--retain-raw-ocr-text"])
    elif component_name == "brand_semantics":
        # brand_semantics uses Embedding Service
        if component_args.get("embedding_service_url"):
            cmd.extend(["--embedding-service-url", component_args["embedding_service_url"]])
    elif component_name == "car_semantics":
        # car_semantics uses Embedding Service
        if component_args.get("embedding_service_url"):
            cmd.extend(["--embedding-service-url", component_args["embedding_service_url"]])
    elif component_name == "content_domain":
        # content_domain uses Triton via ModelManager
        if component_args.get("clip_text_model_spec"):
            cmd.extend(["--clip-text-model-spec", component_args["clip_text_model_spec"]])
        if component_args.get("domain_db_dir"):
            cmd.extend(["--domain-db-dir", component_args["domain_db_dir"]])
    elif component_name == "face_identity":
        # face_identity uses Embedding Service
        if component_args.get("embedding_service_url"):
            cmd.extend(["--embedding-service-url", component_args["embedding_service_url"]])
    elif component_name == "franchise_recognition":
        # franchise_recognition uses Triton via ModelManager
        if component_args.get("clip_text_model_spec"):
            cmd.extend(["--clip-text-model-spec", component_args["clip_text_model_spec"]])
        if component_args.get("franchise_db_dir"):
            cmd.extend(["--franchise-db-dir", component_args["franchise_db_dir"]])
    elif component_name == "place_semantics":
        # place_semantics uses core_clip embeddings
        if component_args.get("places_db_dir"):
            cmd.extend(["--places-db-dir", component_args["places_db_dir"]])

    # Run component
    start_time = time.perf_counter()
    try:
        env = os.environ.copy()
        if not str(env.get("TRITON_HTTP_URL") or "").strip():
            env["TRITON_HTTP_URL"] = str(triton_http_url).strip()
        if not str(env.get("DP_MODELS_ROOT") or "").strip():
            # Deterministic local default for bundled assets/caches.
            dp_root = Path(__file__).resolve().parents[2]
            cand = dp_root / "dp_models" / "bundled_models"
            if cand.is_dir():
                env["DP_MODELS_ROOT"] = str(cand)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
        )
        duration_sec = time.perf_counter() - start_time

        monitor.stop()

        # Get final snapshot
        after_snapshot = get_resource_snapshot()
        final_ram = after_snapshot.cpu_mem_used_mb or 0
        final_vram = after_snapshot.gpu_mem_used_mb or 0

        peak_ram = monitor.peak_ram_mb
        peak_vram = monitor.peak_vram_mb

        ram_delta = (final_ram - initial_ram) if (final_ram and initial_ram) else None
        vram_delta = (final_vram - initial_vram) if (final_vram and initial_vram) else None

        success = result.returncode == 0
        error = None
        if not success:
            error = result.stderr[:500] if result.stderr else f"Exit code: {result.returncode}"

        return ComponentResult(
            name=component_name,
            success=success,
            duration_sec=duration_sec,
            error=error,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            peak_ram_mb=peak_ram,
            peak_vram_mb=peak_vram,
            ram_delta_mb=ram_delta,
            vram_delta_mb=vram_delta,
        )
    except Exception as e:
        monitor.stop()
        duration_sec = time.perf_counter() - start_time
        return ComponentResult(
            name=component_name,
            success=False,
            duration_sec=duration_sec,
            error=str(e),
        )


def _load_npz(npz_path: str) -> Dict[str, Any]:
    """Load NPZ file and unbox object arrays."""
    data = np.load(npz_path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def extract_frame_thumbnail(frame: np.ndarray, max_size: int = 800) -> str:
    """Convert frame to base64 JPEG data URI."""
    h, w = frame.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame

    # Convert RGB to BGR for cv2
    if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame_resized

    _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def _run_quality_script(
    quality_script: Path,
    html_filename: str,
    frames_dir: str,
    rs_path: str,
    max_frames: int = 10,
) -> Optional[str]:
    """Run a quality report script and extract HTML body content."""
    if not quality_script.exists():
        return None
    
    try:
        out_dir = tempfile.mkdtemp()
        cmd = [
            sys.executable,
            str(quality_script),
            "--frames-dir", frames_dir,
            "--rs-path", rs_path,
            "--out-dir", out_dir,
            "--max-frames", str(max_frames),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            # Try exact filename first
            html_path = Path(out_dir) / html_filename
            if not html_path.exists():
                # If exact filename doesn't exist, search for HTML files matching pattern
                # Some scripts use dynamic filenames (e.g., with video_id)
                pattern_parts = html_filename.replace("_quality_report.html", "").split("_")
                if pattern_parts:
                    # Search for files starting with the component name
                    search_pattern = f"{pattern_parts[0]}*quality*.html"
                    html_files = list(Path(out_dir).glob(search_pattern))
                    if html_files:
                        html_path = html_files[0]  # Use first match
                    else:
                        # Fallback: search for any HTML file
                        html_files = list(Path(out_dir).glob("*.html"))
                        if html_files:
                            html_path = html_files[0]
            
            if html_path.exists():
                html_content = html_path.read_text(encoding="utf-8")
                # Extract body content from the generated HTML
                body_start = html_content.find("<body>")
                body_end = html_content.find("</body>")
                if body_start >= 0 and body_end >= 0:
                    body_content = html_content[body_start + 6:body_end].strip()
                    # Remove container div wrapper if present
                    if body_content.startswith("<div class=\"container\">"):
                        body_content = body_content[24:]
                    if body_content.endswith("</div>"):
                        body_content = body_content[:-6]
                    return body_content
                return html_content
        else:
            logger.warning(
                f"Quality script {quality_script.name} failed with code {result.returncode}: "
                f"{result.stderr[:200] if result.stderr else 'No error message'}"
            )
    except Exception as e:
        logger.warning(f"Failed to run quality script {quality_script.name}: {e}")
    return None


def generate_quality_demo_html(
    component_name: str,
    frames_dir: str,
    rs_path: str,
    max_frames: int = 10,
) -> Optional[str]:
    """Generate quality demonstration HTML for a component."""
    try:
        # Mapping of component names to their quality report scripts and output HTML filenames
        # Note: Some scripts (core_clip, core_optical_flow, core_depth_midas) require --video-path
        # and are designed for standalone use, so we use fallback visualizations for them
        quality_scripts = {
            "core_object_detections": (
                Path(__file__).parent / "core_object_detections" / "quality_report" / "demo_core_object_detections_quality.py",
                "detections_quality_report.html",
            ),
            "core_face_landmarks": (
                Path(__file__).parent / "core_face_landmarks" / "quality_report" / "demo_core_face_landmarks_quality.py",
                "landmarks_quality_report.html",
            ),
            "ocr_extractor": (
                Path(__file__).parent / "ocr_extractor" / "quality_report" / "demo_ocr_extractor_quality.py",
                "ocr_extractor_quality_report.html",
            ),
            "brand_semantics": (
                Path(__file__).parent / "core_identity" / "brand_semantics" / "quality_report" / "demo_brand_semantics_quality.py",
                "brand_semantics_quality_report.html",
            ),
            "car_semantics": (
                Path(__file__).parent / "core_identity" / "car_semantics" / "quality_report" / "demo_car_semantics_quality.py",
                "car_semantics_quality_report.html",
            ),
            "content_domain": (
                Path(__file__).parent / "core_identity" / "content_domain" / "quality_report" / "demo_content_domain_quality.py",
                "content_domain_quality_report.html",
            ),
            "franchise_recognition": (
                Path(__file__).parent / "core_identity" / "franchise_recognition" / "quality_report" / "demo_franchise_recognition_quality.py",
                "franchise_recognition_quality_report.html",
            ),
            "place_semantics": (
                Path(__file__).parent / "core_identity" / "place_semantics" / "quality_report" / "demo_place_semantics_quality.py",
                "place_semantics_quality_report.html",
            ),
        }
        
        # Components that require --video-path (standalone scripts, use fallback instead)
        standalone_scripts = {"core_clip", "core_depth_midas", "core_optical_flow"}
        
        # Check if component has a quality report script that works with our interface
        if component_name in quality_scripts:
            quality_script, html_filename = quality_scripts[component_name]
            return _run_quality_script(quality_script, html_filename, frames_dir, rs_path, max_frames)
        
        # For standalone scripts, skip and use fallback below
        if component_name in standalone_scripts:
            pass  # Will use fallback visualization below
        
        # Fallback: simple visualizations for components without dedicated scripts
        if component_name == "core_clip":
            # Simple visualization for CLIP embeddings
            npz_path = Path(rs_path) / "core_clip" / "embeddings.npz"
            if npz_path.exists():
                data = _load_npz(str(npz_path))
                frame_indices = data.get("frame_indices")
                embeddings = data.get("frame_embeddings")
                if frame_indices is not None and embeddings is not None:
                    frame_manager = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
                    thumbnails = []
                    n_frames = min(max_frames, len(frame_indices))
                    step = max(1, len(frame_indices) // n_frames)
                    for i in range(0, len(frame_indices), step):
                        try:
                            frame = frame_manager.get(int(frame_indices[i]))
                            thumb = extract_frame_thumbnail(frame, max_size=400)
                            thumbnails.append(thumb)
                        except Exception:
                            continue
                    frame_manager.close()
                    if thumbnails:
                        html = f"""
                        <div class="component-quality">
                            <h3>CLIP Embeddings Visualization</h3>
                            <p>Processed {len(frame_indices)} frames. Embedding shape: {embeddings.shape}</p>
                            <div class="thumbnails-grid">
                        """
                        for thumb in thumbnails[:max_frames]:
                            html += f'<img src="{thumb}" style="max-width: 200px; margin: 5px;">'
                        html += """
                            </div>
                        </div>
                        """
                        return html
        elif component_name == "core_depth_midas":
            # Simple visualization for depth maps
            npz_path = Path(rs_path) / "core_depth_midas" / "depth.npz"
            if npz_path.exists():
                data = _load_npz(str(npz_path))
                frame_indices = data.get("frame_indices")
                depth_maps = data.get("depth_maps")
                if frame_indices is not None and depth_maps is not None:
                    frame_manager = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
                    thumbnails = []
                    n_frames = min(max_frames, len(frame_indices))
                    step = max(1, len(frame_indices) // n_frames)
                    for i in range(0, len(frame_indices), step):
                        try:
                            frame = frame_manager.get(int(frame_indices[i]))
                            depth = depth_maps[i]
                            # Normalize depth for visualization
                            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-9) * 255).astype(np.uint8)
                            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                            thumb = extract_frame_thumbnail(depth_colored, max_size=400)
                            thumbnails.append(thumb)
                        except Exception:
                            continue
                    frame_manager.close()
                    if thumbnails:
                        html = f"""
                        <div class="component-quality">
                            <h3>Depth Maps Visualization</h3>
                            <p>Processed {len(frame_indices)} frames. Depth shape: {depth_maps.shape}</p>
                            <div class="thumbnails-grid">
                        """
                        for thumb in thumbnails[:max_frames]:
                            html += f'<img src="{thumb}" style="max-width: 200px; margin: 5px;">'
                        html += """
                            </div>
                        </div>
                        """
                        return html
        elif component_name == "core_optical_flow":
            # Simple visualization for optical flow
            npz_path = Path(rs_path) / "core_optical_flow" / "flow.npz"
            if npz_path.exists():
                data = _load_npz(str(npz_path))
                frame_indices = data.get("frame_indices")
                motion_norm = data.get("motion_norm_per_sec_mean")
                if frame_indices is not None and motion_norm is not None:
                    html = f"""
                    <div class="component-quality">
                        <h3>Optical Flow Motion Curve</h3>
                        <p>Processed {len(frame_indices)} frames. Mean motion per second: {np.mean(motion_norm):.4f}</p>
                        <p>Motion range: [{np.min(motion_norm):.4f}, {np.max(motion_norm):.4f}]</p>
                    </div>
                    """
                    return html
    except Exception as e:
        logger.warning(f"Failed to generate quality demo for {component_name}: {e}")
    return None


def create_html_report(
    results: List[ComponentResult],
    frames_dir: str,
    rs_path: str,
    out_html: str,
    initial_snapshot: ResourceSnapshot,
    final_snapshot: ResourceSnapshot,
) -> None:
    """Create comprehensive HTML report."""
    total_duration = sum(r.duration_sec for r in results)
    total_peak_ram = max((r.peak_ram_mb or 0) for r in results) if results else 0
    total_peak_vram = max((r.peak_vram_mb or 0) for r in results) if results else 0

    initial_ram = initial_snapshot.cpu_mem_used_mb or 0
    initial_vram = initial_snapshot.gpu_mem_used_mb or 0
    final_ram = final_snapshot.cpu_mem_used_mb or 0
    final_vram = final_snapshot.gpu_mem_used_mb or 0

    total_ram_delta = final_ram - initial_ram if (final_ram and initial_ram) else None
    total_vram_delta = final_vram - initial_vram if (final_vram and initial_vram) else None

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Core Components Execution Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .components-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .components-table th {{
            background: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .components-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .components-table tr:hover {{
            background: #f5f5f5;
        }}
        .success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .error {{
            color: #f44336;
            font-weight: bold;
        }}
        .component-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        .component-quality {{
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 5px;
        }}
        .thumbnails-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .thumbnails-grid img {{
            border: 2px solid #ddd;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Core Components Execution Report</h1>
        <p><strong>Generated:</strong> {datetime.utcnow().isoformat()}</p>

        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{total_duration:.2f} s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak RAM</div>
                <div class="stat-value">{total_peak_ram:.0f} MB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak VRAM</div>
                <div class="stat-value">{total_peak_vram:.0f} MB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">RAM Delta</div>
                <div class="stat-value">{total_ram_delta:+.0f} MB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">VRAM Delta</div>
                <div class="stat-value">{total_vram_delta:+.0f} MB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Components Run</div>
                <div class="stat-value">{len(results)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Successful</div>
                <div class="stat-value">{sum(1 for r in results if r.success)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value">{sum(1 for r in results if not r.success)}</div>
            </div>
        </div>

        <h2>Component Results</h2>
        <table class="components-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Peak RAM (MB)</th>
                    <th>Peak VRAM (MB)</th>
                    <th>RAM Delta (MB)</th>
                    <th>VRAM Delta (MB)</th>
                </tr>
            </thead>
            <tbody>
"""

    for result in results:
        status_class = "success" if result.success else "error"
        status_text = "✓ Success" if result.success else "✗ Failed"
        # Format values safely
        peak_ram_str = f"{result.peak_ram_mb:.0f}" if result.peak_ram_mb is not None else "N/A"
        peak_vram_str = f"{result.peak_vram_mb:.0f}" if result.peak_vram_mb is not None else "N/A"
        ram_delta_str = f"{result.ram_delta_mb:+.0f}" if result.ram_delta_mb is not None else "N/A"
        vram_delta_str = f"{result.vram_delta_mb:+.0f}" if result.vram_delta_mb is not None else "N/A"
        html += f"""
                <tr>
                    <td><strong>{result.name}</strong></td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.duration_sec:.2f}</td>
                    <td>{peak_ram_str}</td>
                    <td>{peak_vram_str}</td>
                    <td>{ram_delta_str}</td>
                    <td>{vram_delta_str}</td>
                </tr>
"""
        if result.error:
            html += f"""
                <tr>
                    <td colspan="7" style="color: #f44336; font-size: 0.9em; padding-left: 30px;">
                        Error: {result.error}
                    </td>
                </tr>
"""

    html += """
            </tbody>
        </table>

        <h2>Quality Demonstrations</h2>
"""

    for result in results:
        if result.success:
            quality_html = generate_quality_demo_html(result.name, frames_dir, rs_path, max_frames=10)
            if quality_html:
                html += f"""
        <div class="component-section">
            <h3>{result.name}</h3>
            {quality_html}
        </div>
"""
            else:
                html += f"""
        <div class="component-section">
            <h3>{result.name}</h3>
            <p>Quality demonstration not available for this component.</p>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved: {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Run all core components and generate HTML report")
    parser.add_argument("--frames-dir", required=True, help="Path to frames directory")
    parser.add_argument("--rs-path", required=True, help="Path to result_store")
    parser.add_argument("--triton-http-url", required=True, help="Triton HTTP URL (e.g., http://localhost:8000)")
    parser.add_argument("--out-dir", required=True, help="Output directory for HTML report")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for components")
    parser.add_argument("--components", nargs="+", default=None, help="Components to run (default: all)")
    # Component-specific Triton model overrides
    parser.add_argument("--triton-image-model-spec", default=None, help="Override Triton image model spec for core_clip")
    parser.add_argument("--triton-text-model-spec", default=None, help="Override Triton text model spec for core_clip")
    parser.add_argument("--triton-flow-model-name", default=None, help="Override Triton model name for core_optical_flow")
    parser.add_argument("--triton-depth-model-name", default=None, help="Override Triton model name for core_depth_midas")
    parser.add_argument("--detection-model-path", default=None, help="Path to YOLO model file for core_object_detections (e.g., yolo11x_41_best.pt)")
    parser.add_argument("--detection-device", default=None, help="Device for core_object_detections (auto|cpu|cuda)")
    args = parser.parse_args()

    # Default components to run
    all_components = [
        "core_clip",
        "core_object_detections",
        "core_optical_flow",
        "core_depth_midas",
        "core_face_landmarks",
        "ocr_extractor",
        "brand_semantics",
        "car_semantics",
        "content_domain",
        "face_identity",
        "franchise_recognition",
        "place_semantics",
    ]

    components_to_run = args.components if args.components else all_components

    # Get initial snapshot
    initial_snapshot = get_resource_snapshot()

    # Component-specific arguments
    # Default model names match the user's Triton model list:
    # clip_image_224, clip_text, midas_256, raft_256
    component_args = {
        "core_clip": {
            # Audit v3: ModelManager-only specs
            "triton_image_model_spec": args.triton_image_model_spec or "clip_image_224_triton",
            "triton_text_model_spec": args.triton_text_model_spec or "clip_text_triton",
            "triton_preprocess_preset": "openai_clip_224",
        },
        "core_object_detections": {
            # core_object_detections uses ultralytics runtime (not triton) in baseline
            # Use relative path - component will resolve it via DP_MODELS_ROOT if needed
            "model_path": args.detection_model_path or "visual/yolo/yolo11x_41_best.pt",
            "device": args.detection_device or "auto",  # Will use cuda if available, else cpu
        },
        "core_optical_flow": {
            "triton_model_name": args.triton_flow_model_name or "raft_256",
            "triton_model_version": "1",
            "triton_datatype": "UINT8",
            "triton_preprocess_preset": "raft_256",
        },
        "core_depth_midas": {
            "triton_model_name": args.triton_depth_model_name or "midas_256",
            "triton_model_version": "1",
            "triton_datatype": "UINT8",
            "triton_preprocess_preset": "midas_256",
        },
        "core_face_landmarks": {
            "use_pose": True,  # Optional: enable pose landmarks
            "use_hands": True,  # Optional: enable hand landmarks
        },
        "ocr_extractor": {
            "engine": "tesseract",
            "rec_model_spec": "ppocr_rec_onnx_v1_inprocess",
            "ppocr_img_h": 48,
            "ppocr_img_w": 320,
            "min_rec_score": 0.0,
            "proposal_class": "text_region",
            "min_det_score": 0.5,
            "max_boxes_per_frame": 5,
            "max_total_boxes": 5000,
            "crop_margin_frac": 0.02,
            "tesseract_lang": "eng+rus",
            "tesseract_psm": 6,
            "retain_raw_ocr_text": False,
        },
        "brand_semantics": {
            "embedding_service_url": os.environ.get("EMBEDDING_SERVICE_URL") or "http://localhost:8001",
        },
        "car_semantics": {
            "embedding_service_url": os.environ.get("EMBEDDING_SERVICE_URL") or "http://localhost:8001",
        },
        "content_domain": {
            "clip_text_model_spec": "clip_text_triton",
            "domain_db_dir": "dp_models/bundled_models/semantics/content_domain/v1",
        },
        "face_identity": {
            "embedding_service_url": os.environ.get("EMBEDDING_SERVICE_URL") or "http://localhost:8001",
        },
        "franchise_recognition": {
            "clip_text_model_spec": "clip_text_triton",
            "franchise_db_dir": "dp_models/bundled_models/semantics/franchises/v1",
        },
        "place_semantics": {
            "places_db_dir": None,  # Optional: will use default if not set
        },
    }

    # Run all components
    results: List[ComponentResult] = []
    start_time = time.perf_counter()

    for component_name in components_to_run:
        logger.info(f"Running component: {component_name}")
        result = run_component(
            component_name=component_name,
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            triton_http_url=args.triton_http_url,
            batch_size=args.batch_size,
            component_args=component_args.get(component_name, {}),
        )
        results.append(result)
        if result.success:
            logger.info(f"✓ {component_name} completed in {result.duration_sec:.2f}s")
        else:
            logger.error(f"✗ {component_name} failed: {result.error}")

    total_duration = time.perf_counter() - start_time

    # Get final snapshot
    final_snapshot = get_resource_snapshot()

    # Generate HTML report
    out_html = os.path.join(args.out_dir, "core_components_report.html")
    create_html_report(
        results=results,
        frames_dir=args.frames_dir,
        rs_path=args.rs_path,
        out_html=out_html,
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Execution Summary")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Components run: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.success)}")
    print(f"Failed: {sum(1 for r in results if not r.success)}")
    print(f"\nReport saved: {out_html}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

