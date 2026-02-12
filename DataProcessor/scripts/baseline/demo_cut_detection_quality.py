#!/usr/bin/env python3
"""
Демонстрация качества cut_detection: визуализация результатов на тестовом видео.

Скрипт:
1. Запускает Segmenter для создания frames_dir (или использует существующий)
2. Запускает cut_detection
3. Создаёт HTML визуализацию с timeline, thumbnails, графиками сигналов
4. Выводит статистику в консоль

Использование:
    python scripts/baseline/demo_cut_detection_quality.py \
        --video-path /path/to/video.mp4 \
        --out-dir /path/to/output \
        [--frames-dir /path/to/existing/frames] \
        [--preset default|quality|fast]
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent.parent / "VisualProcessor"
sys.path.insert(0, str(_visual_processor_path))

from modules.cut_detection.cut_detection import CutDetectionPipeline
from utils.frame_manager import FrameManager
from utils.logger import get_logger

logger = get_logger("demo_cut_detection_quality")


def run_segmenter(
    video_path: str,
    out_dir: str,
    visual_cfg_path: Optional[str] = None,
    analysis_width: Optional[int] = None,
    analysis_height: Optional[int] = None,
    analysis_fps: Optional[float] = None,
) -> str:
    """Запускает Segmenter для создания frames_dir."""
    logger.info("Running Segmenter for video: %s", video_path)
    
    segmenter_script = Path(__file__).parent.parent.parent / "Segmenter" / "segmenter.py"
    if not segmenter_script.exists():
        raise FileNotFoundError(f"Segmenter script not found: {segmenter_script}")
    
    video_id = Path(video_path).stem
    frames_dir = os.path.join(out_dir, video_id, "video")
    
    # Проверка на существующие batch файлы и metadata.json
    metadata_path = os.path.join(frames_dir, "metadata.json")
    if os.path.exists(metadata_path):
        # Проверяем наличие batch файлов
        import glob
        batch_files = glob.glob(os.path.join(frames_dir, "batch*.npy"))
        if batch_files:
            logger.info("Found existing frames_dir with batch files: %s (%d batches)", frames_dir, len(batch_files))
            # Проверяем наличие config_hash в metadata
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if not metadata.get("config_hash"):
                    logger.warning("metadata.json exists but missing config_hash, adding it...")
                    metadata["config_hash"] = "demo_demo_run"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    logger.info("Added config_hash to existing metadata.json")
            except Exception as e:
                logger.warning("Failed to check/add config_hash: %s", e)
            return frames_dir
        else:
            logger.info("metadata.json exists but no batch files found, will regenerate")
    
    
    cmd = [
        sys.executable,
        str(segmenter_script),
        "--video-path", video_path,
        "--output", out_dir,
        "--platform-id", "demo",
        "--video-id", video_id,
        "--run-id", "demo_run",
        "--sampling-policy-version", "v1",
        "--config-hash", "demo_demo_run",  # Required by BaseModule
    ]
    
    if visual_cfg_path:
        cmd.extend(["--visual-cfg-path", visual_cfg_path])
    if analysis_width:
        cmd.extend(["--analysis-width", str(analysis_width)])
    if analysis_height:
        cmd.extend(["--analysis-height", str(analysis_height)])
    if analysis_fps:
        cmd.extend(["--analysis-fps", str(analysis_fps)])
    
    logger.info("Running Segmenter: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Segmenter failed:\n{result.stdout}\n{result.stderr}")
    
    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Segmenter did not create frames_dir: {frames_dir}")
    
    # Убеждаемся, что config_hash присутствует в metadata (на случай если Segmenter его не добавил)
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not metadata.get("config_hash"):
            logger.warning("Segmenter did not add config_hash, adding it manually...")
            metadata["config_hash"] = "demo_demo_run"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info("Added config_hash to metadata.json")
    except Exception as e:
        logger.warning("Failed to verify/add config_hash: %s", e)
    
    logger.info("Segmenter completed. Frames dir: %s", frames_dir)
    return frames_dir


def run_cut_detection(
    frames_dir: str,
    rs_path: str,
    preset: str = "default",
    device: str = "cpu",
    prefer_core_optical_flow: bool = False,
) -> str:
    """Запускает cut_detection и возвращает путь к сохранённому NPZ."""
    logger.info("Running cut_detection with preset: %s", preset)
    
    # Map preset to parameters
    preset_map = {
        "quality": {"ssim": 640, "flow": 384, "cascade": False},
        "default": {"ssim": 512, "flow": 320, "cascade": False},
        "fast": {"ssim": 384, "flow": 256, "cascade": True},
    }
    pm = preset_map.get(preset, preset_map["default"])
    
    pipeline = CutDetectionPipeline(
        rs_path=rs_path,
        fps=30.0,  # actual fps taken from metadata
        device=device,
        clip_zero_shot=False,  # baseline: no CLIP for demo
        use_deep_features=False,  # baseline: no deep features
        use_adaptive_thresholds=True,
        use_semantic_clustering=False,
        ssim_max_side=int(pm["ssim"]),
        flow_max_side=int(pm["flow"]),
        hard_cuts_cascade=bool(pm["cascade"]),
        prefer_core_optical_flow=prefer_core_optical_flow,
        write_model_facing_npz=True,
    )
    
    config = {}
    saved_path = pipeline.run(frames_dir=frames_dir, config=config)
    
    logger.info("cut_detection completed. Results: %s", saved_path)
    return saved_path


def load_cut_detection_results(npz_path: str) -> Dict[str, Any]:
    """Загружает результаты cut_detection из NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    
    # Unbox object arrays
    result = {}
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            try:
                result[key] = value.item()
            except Exception:
                result[key] = value
        else:
            result[key] = value
    
    return result


def extract_frame_thumbnail(frame_manager: FrameManager, frame_idx: int, max_size: int = 200) -> Optional[str]:
    """Извлекает кадр и сохраняет как base64 для HTML."""
    try:
        frame = frame_manager.get(frame_idx)
        # Resize for thumbnail
        h, w = frame.shape[:2]
        scale = min(max_size / max(h, w), 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > 0 and new_w > 0:
            frame_resized = cv2.resize(frame, (new_w, new_h))
            # Convert RGB to BGR for cv2.imencode
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.warning("Failed to extract thumbnail for frame %d: %s", frame_idx, e)
    return None


def create_html_visualization(
    results: Dict[str, Any],
    frame_manager: FrameManager,
    video_path: str,
    out_html: str,
    npz_path: Optional[str] = None,
) -> None:
    """Создаёт HTML визуализацию результатов cut_detection."""
    logger.info("Creating HTML visualization: %s", out_html)
    
    # Extract data
    features = results.get("features", {})
    detections = results.get("detections", {})
    frame_indices = results.get("frame_indices", np.array([]))
    times_s = results.get("times_s", np.array([]))
    
    # Hard cuts
    hard_cut_pos = detections.get("hard_cut_pos", [])
    hard_cut_frame_indices = detections.get("hard_cut_frame_indices", [])
    hard_cut_strengths = detections.get("hard_cut_strengths", [])
    
    # Soft cuts
    soft_events = detections.get("soft_events", [])
    
    # Motion cuts
    motion_cut_pos = detections.get("motion_cut_pos", [])
    
    # Jump cuts
    jump_cut_pos = detections.get("jump_cut_pos", [])
    
    # Shot boundaries
    shot_boundaries_pos = detections.get("shot_boundaries_pos", [])
    
    # Load model-facing NPZ if available (for signal curves)
    signal_curves = {}
    # Try to find model-facing NPZ in the same directory as main NPZ
    if npz_path:
        npz_abs_path = os.path.abspath(npz_path)
        npz_dir = os.path.dirname(npz_abs_path)
    else:
        # Fallback: try to find it from results metadata
        model_facing_path_from_meta = results.get("model_facing_npz_path", "")
        if model_facing_path_from_meta:
            npz_dir = os.path.dirname(os.path.abspath(model_facing_path_from_meta))
        else:
            npz_dir = ""
    
    if npz_dir:
        # Look for model-facing NPZ files in cut_detection subdirectory
        model_facing_pattern = os.path.join(npz_dir, "cut_detection_model_facing_*.npz")
        model_facing_files = glob.glob(model_facing_pattern)
        if model_facing_files:
            # Use the most recent one
            model_facing_path = max(model_facing_files, key=os.path.getmtime)
            try:
                mf_data = np.load(model_facing_path, allow_pickle=True)
                for key in ["hist_diff_l1", "ssim_drop", "flow_mag", "hard_score"]:
                    if key in mf_data.files:
                        arr = mf_data[key]
                        if isinstance(arr, np.ndarray):
                            signal_curves[key] = arr.tolist()
                logger.info("Loaded signal curves from: %s", model_facing_path)
            except Exception as e:
                logger.warning("Failed to load model-facing NPZ: %s", e)
    
    # Extract thumbnails for hard cuts
    thumbnails = []
    for i, cut_pos in enumerate(hard_cut_pos[:20]):  # Limit to first 20 cuts
        if cut_pos < len(frame_indices):
            frame_idx = int(frame_indices[int(cut_pos)])
            # Get frame before and after cut
            thumb_before = extract_frame_thumbnail(frame_manager, max(0, frame_idx - 1)) if frame_idx > 0 else None
            thumb_after = extract_frame_thumbnail(frame_manager, min(frame_manager.total_frames - 1, frame_idx))
            
            cut_time = float(times_s[int(cut_pos)]) if cut_pos < len(times_s) else 0.0
            strength = float(hard_cut_strengths[i]) if i < len(hard_cut_strengths) else 0.0
            
            thumbnails.append({
                "cut_index": i,
                "time_s": cut_time,
                "strength": strength,
                "frame_before": thumb_before,
                "frame_after": thumb_after,
            })
    
    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>cut_detection Quality Demo - {Path(video_path).name}</title>
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
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .timeline-container {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 5px;
        }}
        .timeline {{
            position: relative;
            height: 100px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .timeline-marker {{
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            cursor: pointer;
        }}
        .timeline-marker.hard-cut {{
            background: #f44336;
            z-index: 3;
        }}
        .timeline-marker.soft-cut {{
            background: #ff9800;
            z-index: 2;
        }}
        .timeline-marker.motion-cut {{
            background: #2196F3;
            z-index: 2;
        }}
        .timeline-marker.jump-cut {{
            background: #9C27B0;
            z-index: 4;
        }}
        .timeline-marker:hover {{
            opacity: 0.7;
        }}
        .timeline-label {{
            position: absolute;
            top: -20px;
            font-size: 10px;
            color: #666;
            white-space: nowrap;
        }}
        .thumbnails {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .thumbnail-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background: white;
        }}
        .thumbnail-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #333;
        }}
        .thumbnail-pair {{
            display: flex;
            gap: 10px;
        }}
        .thumbnail-pair img {{
            max-width: 140px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 5px;
        }}
        .chart {{
            height: 200px;
            margin: 20px 0;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>cut_detection Quality Demo</h1>
        <div class="info">
            <strong>Video:</strong> {Path(video_path).name}<br>
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Total frames:</strong> {len(frame_indices)}<br>
            <strong>Video duration:</strong> {float(times_s[-1]) if len(times_s) > 0 else 0:.2f} seconds
        </div>
        
        <h2>Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Hard Cuts</div>
                <div class="stat-value">{features.get('hard_cuts_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hard Cuts/min</div>
                <div class="stat-value">{features.get('hard_cuts_per_minute', 0):.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Soft Cuts</div>
                <div class="stat-value">{features.get('fade_in_count', 0) + features.get('fade_out_count', 0) + features.get('dissolve_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Motion Cuts</div>
                <div class="stat-value">{features.get('motion_cuts_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Jump Cuts</div>
                <div class="stat-value">{features.get('jump_cuts_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Shots</div>
                <div class="stat-value">{len(shot_boundaries_pos) - 1 if shot_boundaries_pos else 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Shot Length</div>
                <div class="stat-value">{features.get('avg_shot_length', 0):.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Scenes</div>
                <div class="stat-value">{features.get('scene_count', 0)}</div>
            </div>
        </div>
        
        <h2>Timeline</h2>
        <div class="timeline-container">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #f44336;"></div>
                    <span>Hard Cut</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff9800;"></div>
                    <span>Soft Cut</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3;"></div>
                    <span>Motion Cut</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0;"></div>
                    <span>Jump Cut</span>
                </div>
            </div>
            <div class="timeline" id="timeline">
"""
    
    # Add timeline markers
    video_duration = float(times_s[-1]) if len(times_s) > 0 else 1.0
    
    for cut_pos in hard_cut_pos:
        if cut_pos < len(times_s):
            cut_time = float(times_s[int(cut_pos)])
            left_pct = (cut_time / video_duration) * 100 if video_duration > 0 else 0
            html_content += f'<div class="timeline-marker hard-cut" style="left: {left_pct}%;" title="Hard cut at {cut_time:.2f}s"></div>\n'
    
    for event in soft_events[:50]:  # Limit to first 50
        start_pos = event.get("start", 0)
        if start_pos < len(times_s):
            event_time = float(times_s[int(start_pos)])
            left_pct = (event_time / video_duration) * 100 if video_duration > 0 else 0
            html_content += f'<div class="timeline-marker soft-cut" style="left: {left_pct}%;" title="Soft cut ({event.get("type", "unknown")}) at {event_time:.2f}s"></div>\n'
    
    for cut_pos in motion_cut_pos[:50]:  # Limit to first 50
        if cut_pos < len(times_s):
            cut_time = float(times_s[int(cut_pos)])
            left_pct = (cut_time / video_duration) * 100 if video_duration > 0 else 0
            html_content += f'<div class="timeline-marker motion-cut" style="left: {left_pct}%;" title="Motion cut at {cut_time:.2f}s"></div>\n'
    
    for cut_pos in jump_cut_pos[:50]:  # Limit to first 50
        if cut_pos < len(times_s):
            cut_time = float(times_s[int(cut_pos)])
            left_pct = (cut_time / video_duration) * 100 if video_duration > 0 else 0
            html_content += f'<div class="timeline-marker jump-cut" style="left: {left_pct}%;" title="Jump cut at {cut_time:.2f}s"></div>\n'
    
    html_content += """
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px; color: #666;">
                <span>0s</span>
                <span>{video_duration:.1f}s</span>
            </div>
        </div>
"""
    
    # Add signal charts if available
    if signal_curves:
        html_content += """
        <h2>Detection Signals</h2>
        <div class="chart-container">
"""
        for signal_name, signal_data in signal_curves.items():
            if not signal_data or len(signal_data) == 0:
                continue
            
            # Create time axis
            pair_times = []
            if len(times_s) >= 2:
                pair_times = [(float(times_s[i]) + float(times_s[i+1])) / 2.0 for i in range(len(times_s) - 1)]
            
            if len(pair_times) != len(signal_data):
                pair_times = list(range(len(signal_data)))
            
            # Filter NaN values for plotting
            valid_data = [(t, v) for t, v in zip(pair_times, signal_data) if np.isfinite(v)]
            if not valid_data:
                continue
            
            chart_id = f"chart_{signal_name}"
            html_content += f"""
            <div class="chart" id="{chart_id}"></div>
            <script>
                var trace_{signal_name} = {{
                    x: {[t for t, v in valid_data]},
                    y: {[v for t, v in valid_data]},
                    type: 'scatter',
                    mode: 'lines',
                    name: '{signal_name}',
                    line: {{color: '#2196F3'}}
                }};
                var layout_{signal_name} = {{
                    title: '{signal_name}',
                    xaxis: {{title: 'Time (s)'}},
                    yaxis: {{title: 'Value'}},
                    height: 200,
                    margin: {{l: 50, r: 20, t: 40, b: 40}}
                }};
                Plotly.newPlot('{chart_id}', [trace_{signal_name}], layout_{signal_name});
            </script>
"""
        html_content += """
        </div>
"""
    
    # Add thumbnails
    if thumbnails:
        html_content += """
        <h2>Hard Cuts (First 20)</h2>
        <div class="thumbnails">
"""
        for thumb in thumbnails:
            html_content += f"""
            <div class="thumbnail-card">
                <h3>Cut #{thumb['cut_index'] + 1} at {thumb['time_s']:.2f}s (strength: {thumb['strength']:.3f})</h3>
                <div class="thumbnail-pair">
"""
            if thumb['frame_before']:
                html_content += f'<img src="{thumb["frame_before"]}" alt="Before">'
            if thumb['frame_after']:
                html_content += f'<img src="{thumb["frame_after"]}" alt="After">'
            html_content += """
                </div>
            </div>
"""
        html_content += """
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info("HTML visualization saved: %s", out_html)


def print_statistics(results: Dict[str, Any]) -> None:
    """Выводит статистику в консоль."""
    features = results.get("features", {})
    detections = results.get("detections", {})
    
    print("\n" + "="*70)
    print("cut_detection Quality Statistics")
    print("="*70)
    
    print("\n📊 Hard Cuts:")
    print(f"  Total: {features.get('hard_cuts_count', 0)}")
    print(f"  Per minute: {features.get('hard_cuts_per_minute', 0):.2f}")
    print(f"  Mean strength: {features.get('hard_cut_strength_mean', 0):.3f}")
    
    print("\n🎬 Soft Transitions:")
    print(f"  Fade in: {features.get('fade_in_count', 0)}")
    print(f"  Fade out: {features.get('fade_out_count', 0)}")
    print(f"  Dissolve: {features.get('dissolve_count', 0)}")
    print(f"  Avg fade duration: {features.get('avg_fade_duration', 0):.2f}s")
    
    print("\n🎥 Motion-based Cuts:")
    print(f"  Total: {features.get('motion_cuts_count', 0)}")
    print(f"  Whip pan: {features.get('whip_pan_transitions_count', 0)}")
    print(f"  Zoom: {features.get('zoom_transition_count', 0)}")
    print(f"  Speed ramp: {features.get('speed_ramp_cuts_count', 0)}")
    
    print("\n👤 Jump Cuts:")
    print(f"  Total: {features.get('jump_cuts_count', 0)}")
    print(f"  Per minute: {features.get('jump_cut_ratio_per_minute', 0):.2f}")
    print(f"  Mean intensity: {features.get('jump_cut_intensity', 0):.3f}")
    
    print("\n🎞️ Shot Statistics:")
    shot_boundaries = detections.get("shot_boundaries_pos", [])
    print(f"  Total shots: {len(shot_boundaries) - 1 if shot_boundaries else 0}")
    print(f"  Avg shot length: {features.get('avg_shot_length', 0):.2f}s")
    print(f"  Median shot length: {features.get('median_shot_length', 0):.2f}s")
    print(f"  Short shots ratio: {features.get('short_shots_ratio', 0):.3f}")
    print(f"  Long shots ratio: {features.get('long_shots_ratio', 0):.3f}")
    
    print("\n🎭 Scene Statistics:")
    print(f"  Total scenes: {features.get('scene_count', 0)}")
    print(f"  Avg scene length: {features.get('avg_scene_length_shots', 0):.1f} shots")
    print(f"  Scene to shot ratio: {features.get('scene_to_shot_ratio', 0):.3f}")
    
    print("\n⏱️ Timing Statistics:")
    print(f"  Median cut interval: {features.get('median_cut_interval', 0):.2f}s")
    print(f"  Cut interval std: {features.get('cut_interval_std', 0):.2f}s")
    print(f"  Cut rhythm uniformity: {features.get('cut_rhythm_uniformity_score', 0):.3f}")
    
    print("\n" + "="*70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Демонстрация качества cut_detection с визуализацией",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Путь к тестовому видео (mp4/mkv/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Директория для выходных файлов (frames_dir, rs_path, HTML)",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Существующий frames_dir (если указан, Segmenter пропускается)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["quality", "default", "fast"],
        help="Preset для cut_detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Устройство для обработки",
    )
    parser.add_argument(
        "--prefer-core-optical-flow",
        action="store_true",
        help="Использовать core_optical_flow если доступен",
    )
    parser.add_argument(
        "--visual-cfg-path",
        type=str,
        default=None,
        help="Путь к VisualProcessor/config.yaml для Segmenter",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=None,
        help="Ширина для анализа (передаётся в Segmenter)",
    )
    parser.add_argument(
        "--analysis-height",
        type=int,
        default=None,
        help="Высота для анализа (передаётся в Segmenter)",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=None,
        help="FPS для анализа (передаётся в Segmenter)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    os.makedirs(args.out_dir, exist_ok=True)
    video_id = Path(args.video_path).stem
    
    # Step 1: Run Segmenter (or use existing frames_dir)
    if args.frames_dir and os.path.exists(os.path.join(args.frames_dir, "metadata.json")):
        frames_dir = args.frames_dir
        logger.info("Using provided frames_dir: %s", frames_dir)
    else:
        frames_dir = run_segmenter(
            video_path=args.video_path,
            out_dir=args.out_dir,
            visual_cfg_path=args.visual_cfg_path,
            analysis_width=args.analysis_width,
            analysis_height=args.analysis_height,
            analysis_fps=args.analysis_fps,
        )
    
    # Step 2: Run cut_detection
    rs_path = os.path.join(args.out_dir, video_id, "result_store")
    os.makedirs(rs_path, exist_ok=True)
    
    npz_path = run_cut_detection(
        frames_dir=frames_dir,
        rs_path=rs_path,
        preset=args.preset,
        device=args.device,
        prefer_core_optical_flow=args.prefer_core_optical_flow,
    )
    
    # Step 3: Load results
    logger.info("Loading results from: %s", npz_path)
    results = load_cut_detection_results(npz_path)
    
    # Step 4: Print statistics
    print_statistics(results)
    
    # Step 5: Create HTML visualization
    frame_manager = FrameManager(frames_dir)
    try:
        html_path = os.path.join(args.out_dir, f"cut_detection_quality_demo_{video_id}.html")
        create_html_visualization(
            results=results,
            frame_manager=frame_manager,
            video_path=args.video_path,
            out_html=html_path,
            npz_path=npz_path,
        )
        print(f"\n✅ HTML visualization saved: {html_path}")
        print(f"   Open in browser to review quality")
        print(f"\n📋 Review checklist:")
        print(f"   - Check timeline: are cuts marked at correct positions?")
        print(f"   - Review thumbnails: do hard cuts show actual scene transitions?")
        print(f"   - Check statistics: are values reasonable for this video type?")
        print(f"   - Verify signal curves: do spikes correspond to detected cuts?")
    finally:
        frame_manager.close()
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()

