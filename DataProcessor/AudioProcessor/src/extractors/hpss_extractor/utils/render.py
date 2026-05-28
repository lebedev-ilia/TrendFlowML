"""
Renderer для hpss_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def render_hpss_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Генерирует render-context JSON для hpss_extractor.

    Args:
        npz_data: Данные из NPZ файла
        meta: Метаданные из NPZ

    Returns:
        Render-context JSON для UI
    """
    render_context: Dict[str, Any] = {
        "component_name": meta.get("producer", "hpss_extractor"),
        "status": meta.get("status", "error"),
        "error": meta.get("error"),
        "empty_reason": meta.get("empty_reason"),
        "processing_time": meta.get("processing_time"),
        "device_used": meta.get("device_used"),
        "contract_version": meta.get("hpss_contract_version"),
        "features_enabled": meta.get("features_enabled", []),
        "metrics": {},
        "time_series": {},
        "segments_data": {},
    }

    if render_context["status"] == "ok":
        # Extract scalar features from feature_names/feature_values
        feature_names = npz_data.get("feature_names", [])
        feature_values = npz_data.get("feature_values", [])
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        if isinstance(feature_values, np.ndarray):
            feature_values = feature_values.tolist()

        features = {}
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                features[name] = feature_values[i]

        # Extract payload (if exists)
        payload = npz_data.get("payload")
        if isinstance(payload, np.ndarray) and payload.dtype == object:
            payload = payload.item() if payload.size == 1 else {}
        if not isinstance(payload, dict):
            payload = {}
        
        # Merge features into payload (features take precedence)
        payload = {**payload, **features}
        
        features_enabled = render_context["features_enabled"]

        # Energy metrics
        if "energy_metrics" in features_enabled:
            render_context["metrics"]["harmonic_share"] = float(payload.get("hpss_harmonic_share", 0.0))
            render_context["metrics"]["percussive_share"] = float(payload.get("hpss_percussive_share", 0.0))
            render_context["metrics"]["energy_total"] = float(payload.get("hpss_energy_total", 0.0))
            render_context["metrics"]["energy_harmonic"] = float(payload.get("hpss_energy_harmonic", 0.0))
            render_context["metrics"]["energy_percussive"] = float(payload.get("hpss_energy_percussive", 0.0))
            render_context["metrics"]["harmonic_stability"] = float(payload.get("hpss_harmonic_stability", 0.0))
            render_context["metrics"]["percussive_stability"] = float(payload.get("hpss_percussive_stability", 0.0))
            render_context["metrics"]["separation_quality"] = float(payload.get("hpss_separation_quality", 0.0))
            render_context["metrics"]["balance_score"] = float(payload.get("hpss_balance_score", 0.0))
            render_context["metrics"]["dominance"] = payload.get("hpss_dominance") or meta.get("hpss_dominance") or "unknown"

            # Segment-level aggregates (if run_segments was used)
            if "hpss_harmonic_share_mean" in payload:
                render_context["metrics"]["harmonic_share_mean"] = float(payload.get("hpss_harmonic_share_mean", 0.0))
                render_context["metrics"]["harmonic_share_std"] = float(payload.get("hpss_harmonic_share_std", 0.0))
                render_context["metrics"]["percussive_share_mean"] = float(payload.get("hpss_percussive_share_mean", 0.0))
                render_context["metrics"]["percussive_share_std"] = float(payload.get("hpss_percussive_share_std", 0.0))

        # Spectral features
        if "spectral_features" in features_enabled:
            render_context["metrics"]["harmonic_centroid_mean"] = float(payload.get("hpss_harmonic_centroid_mean", 0.0))
            render_context["metrics"]["harmonic_centroid_std"] = float(payload.get("hpss_harmonic_centroid_std", 0.0))
            render_context["metrics"]["harmonic_bandwidth_mean"] = float(payload.get("hpss_harmonic_bandwidth_mean", 0.0))
            render_context["metrics"]["harmonic_bandwidth_std"] = float(payload.get("hpss_harmonic_bandwidth_std", 0.0))
            render_context["metrics"]["harmonic_rolloff_mean"] = float(payload.get("hpss_harmonic_rolloff_mean", 0.0))
            render_context["metrics"]["harmonic_rolloff_std"] = float(payload.get("hpss_harmonic_rolloff_std", 0.0))
            render_context["metrics"]["percussive_centroid_mean"] = float(payload.get("hpss_percussive_centroid_mean", 0.0))
            render_context["metrics"]["percussive_centroid_std"] = float(payload.get("hpss_percussive_centroid_std", 0.0))
            render_context["metrics"]["percussive_bandwidth_mean"] = float(payload.get("hpss_percussive_bandwidth_mean", 0.0))
            render_context["metrics"]["percussive_bandwidth_std"] = float(payload.get("hpss_percussive_bandwidth_std", 0.0))
            render_context["metrics"]["percussive_rolloff_mean"] = float(payload.get("hpss_percussive_rolloff_mean", 0.0))
            render_context["metrics"]["percussive_rolloff_std"] = float(payload.get("hpss_percussive_rolloff_std", 0.0))

        # Time series
        if "time_series" in features_enabled:
            harmonic_share_series = npz_data.get("hpss_harmonic_share_series")
            if harmonic_share_series is not None:
                if isinstance(harmonic_share_series, np.ndarray):
                    if harmonic_share_series.size > 0:
                        render_context["time_series"]["harmonic_share_series"] = harmonic_share_series.tolist()
                    else:
                        render_context["time_series"]["harmonic_share_series"] = []
                elif isinstance(harmonic_share_series, list) and len(harmonic_share_series) > 0:
                    render_context["time_series"]["harmonic_share_series"] = harmonic_share_series
                else:
                    render_context["time_series"]["harmonic_share_series"] = []
            else:
                render_context["time_series"]["harmonic_share_series"] = []

            percussive_share_series = npz_data.get("hpss_percussive_share_series")
            if percussive_share_series is not None:
                if isinstance(percussive_share_series, np.ndarray):
                    if percussive_share_series.size > 0:
                        render_context["time_series"]["percussive_share_series"] = percussive_share_series.tolist()
                    else:
                        render_context["time_series"]["percussive_share_series"] = []
                elif isinstance(percussive_share_series, list) and len(percussive_share_series) > 0:
                    render_context["time_series"]["percussive_share_series"] = percussive_share_series
                else:
                    render_context["time_series"]["percussive_share_series"] = []
            else:
                render_context["time_series"]["percussive_share_series"] = []

        # Segment data (Audit v3: segment_start_sec, segment_end_sec, segment_center_sec)
        def _to_list(arr):
            if arr is None:
                return []
            if isinstance(arr, np.ndarray):
                return arr.tolist() if arr.size > 0 else []
            return list(arr) if arr else []

        seg_start = _to_list(npz_data.get("segment_start_sec"))
        seg_end = _to_list(npz_data.get("segment_end_sec"))
        seg_center = _to_list(npz_data.get("segment_center_sec"))
        seg_mask = _to_list(npz_data.get("segment_mask"))
        if seg_center:
            render_context["segments_data"]["segment_centers_sec"] = seg_center
            render_context["segments_data"]["segment_durations_sec"] = [
                float(seg_end[i]) - float(seg_start[i]) for i in range(len(seg_center))
            ] if seg_end else []
            render_context["segments_data"]["segment_mask"] = seg_mask
        else:
            # Backward compat: segment_centers_sec, segment_durations_sec
            seg_centers_legacy = _to_list(npz_data.get("segment_centers_sec"))
            seg_dur_legacy = _to_list(npz_data.get("segment_durations_sec"))
            render_context["segments_data"]["segment_centers_sec"] = seg_centers_legacy
            render_context["segments_data"]["segment_durations_sec"] = seg_dur_legacy
        if render_context["segments_data"].get("segment_centers_sec"):
            render_context["segments_data"]["segments_count"] = int(payload.get("segments_count", 0))

        # Metadata
        render_context["metadata"] = {
            "sample_rate": int(payload.get("sample_rate", 0)),
            "n_fft": int(payload.get("n_fft", 0)),
            "hop_length": int(payload.get("hop_length", 0)),
            "duration": float(payload.get("duration", 0.0)),
            "hpss_frames": int(payload.get("hpss_frames", 0)),
            "hpss_kernel_size": int(payload.get("hpss_kernel_size", 0)),
            "hpss_margin": float(payload.get("hpss_margin", 0.0)),
            "hpss_power": float(payload.get("hpss_power", 0.0)),
        }

    return render_context


def render_hpss_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерирует HTML-страницу для отладки визуализации hpss_extractor.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML

    Returns:
        Путь к сохранённому HTML файлу
    """
    try:
        npz_data = load_npz(npz_path)
        meta = extract_meta(npz_data)
        render_context = render_hpss_extractor(npz_data, meta)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPSS Extractor Debug Render</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #0056b3; }}
        pre {{ background-color: #e9e9e9; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        .chart-container {{ position: relative; height: 300px; width: 100%; margin-bottom: 20px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background-color: #eaf4ff; padding: 10px; border-radius: 5px; border: 1px solid #cce0ff; }}
        .metric-card strong {{ color: #0056b3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HPSS Extractor Debug Render</h1>
        <p><strong>Component:</strong> {render_context.get("component_name")}</p>
        <p><strong>Status:</strong> {render_context.get("status")}</p>
        {"<p><strong>Error:</strong> " + render_context["error"] + "</p>" if render_context.get("error") else ""}
        {"<p><strong>Empty Reason:</strong> " + render_context["empty_reason"] + "</p>" if render_context.get("empty_reason") else ""}
        <p><strong>Processing Time:</strong> {f"{render_context.get('processing_time', 0.0):.2f}" if render_context.get("processing_time") is not None else "N/A"} seconds</p>
        <p><strong>Device Used:</strong> {render_context.get("device_used") or "N/A"}</p>
        <p><strong>Contract Version:</strong> {render_context.get("contract_version") or "N/A"}</p>
        <p><strong>Features Enabled:</strong> {', '.join(render_context.get("features_enabled", [])) if render_context.get("features_enabled") else "None"}</p>

        <h2>Metrics</h2>
        <div class="metric-grid">
"""

        for key, value in render_context["metrics"].items():
            if isinstance(value, (int, float)):
                html_content += f"""
            <div class="metric-card">
                <strong>{key.replace('_', ' ').title()}:</strong> {value:.4f}
            </div>
"""
            else:
                html_content += f"""
            <div class="metric-card">
                <strong>{key.replace('_', ' ').title()}:</strong> {value}
            </div>
"""
        html_content += "</div>"

        # Time series charts (offline vanilla canvas, no CDN)
        if render_context["time_series"].get("harmonic_share_series"):
            harmonic_series_json = json.dumps(render_context["time_series"]["harmonic_share_series"])
            percussive_series_json = json.dumps(render_context["time_series"].get("percussive_share_series", []))
            html_content += f"""
        <h2>Time Series: Harmonic vs Percussive Share</h2>
        <div class="chart-container">
            <canvas id="shareSeriesChart" width="800" height="280"></canvas>
        </div>
        <script>
            (function() {{
                const harmonicSeries = {harmonic_series_json};
                const percussiveSeries = {percussive_series_json};
                const canvas = document.getElementById('shareSeriesChart');
                const ctx = canvas.getContext('2d');
                const w = canvas.width, h = canvas.height;
                const pad = {{ top: 20, right: 20, bottom: 30, left: 50 }};
                const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
                const n = Math.max(harmonicSeries.length, percussiveSeries.length);
                if (n === 0) return;
                const maxY = 1, minY = 0;
                function toX(i) {{ return pad.left + (i / Math.max(1, n - 1)) * plotW; }}
                function toY(v) {{ return pad.top + plotH - (v - minY) / (maxY - minY) * plotH; }}
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, w, h);
                ctx.strokeStyle = '#ddd';
                ctx.lineWidth = 1;
                for (let i = 0; i <= 5; i++) {{
                    const y = pad.top + (i / 5) * plotH;
                    ctx.beginPath();
                    ctx.moveTo(pad.left, y);
                    ctx.lineTo(w - pad.right, y);
                    ctx.stroke();
                }}
                ctx.fillStyle = '#333';
                ctx.font = '12px sans-serif';
                ctx.fillText('Frame', w / 2 - 20, h - 5);
                ctx.save();
                ctx.translate(15, h / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.fillText('Share (0.0-1.0)', 0, 0);
                ctx.restore();
                function drawLine(data, color) {{
                    if (data.length < 2) return;
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(toX(0), toY(data[0]));
                    for (let i = 1; i < data.length; i++)
                        ctx.lineTo(toX(i), toY(data[i]));
                    ctx.stroke();
                }}
                drawLine(harmonicSeries, 'rgb(75, 192, 192)');
                drawLine(percussiveSeries, 'rgb(255, 99, 132)');
                ctx.fillStyle = 'rgb(75, 192, 192)';
                ctx.fillRect(w - pad.right - 100, pad.top, 12, 12);
                ctx.fillStyle = '#333';
                ctx.fillText('Harmonic', w - pad.right - 82, pad.top + 10);
                ctx.fillStyle = 'rgb(255, 99, 132)';
                ctx.fillRect(w - pad.right - 100, pad.top + 18, 12, 12);
                ctx.fillStyle = '#333';
                ctx.fillText('Percussive', w - pad.right - 82, pad.top + 28);
            }})();
        </script>
"""

        # Segment-level data (offline vanilla canvas)
        if render_context["segments_data"].get("segment_centers_sec"):
            segment_centers_json = json.dumps(render_context["segments_data"]["segment_centers_sec"])
            segment_durations_json = json.dumps(render_context["segments_data"]["segment_durations_sec"])
            html_content += f"""
        <h2>Segment-Level Data</h2>
        <div class="chart-container">
            <canvas id="segmentDataChart" width="800" height="280"></canvas>
        </div>
        <script>
            (function() {{
                const segmentCenters = {segment_centers_json};
                const segmentDurations = {segment_durations_json};
                const canvas = document.getElementById('segmentDataChart');
                const ctx = canvas.getContext('2d');
                const w = canvas.width, h = canvas.height;
                const pad = {{ top: 20, right: 20, bottom: 30, left: 50 }};
                const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
                const pts = segmentCenters.map((c, i) => ({{ x: c, y: (segmentDurations[i] || 0) }}));
                if (pts.length === 0) return;
                const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
                const minX = Math.min(...xs), maxX = Math.max(...xs) || 1;
                const minY = Math.min(...ys), maxY = Math.max(...ys) || 1;
                const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
                function toX(x) {{ return pad.left + (x - minX) / rangeX * plotW; }}
                function toY(y) {{ return pad.top + plotH - (y - minY) / rangeY * plotH; }}
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, w, h);
                ctx.strokeStyle = '#ddd';
                ctx.lineWidth = 1;
                ctx.strokeRect(pad.left, pad.top, plotW, plotH);
                ctx.fillStyle = '#333';
                ctx.font = '12px sans-serif';
                ctx.fillText('Center Time (s)', w / 2 - 50, h - 5);
                ctx.save();
                ctx.translate(15, h / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.fillText('Duration (s)', 0, 0);
                ctx.restore();
                ctx.fillStyle = 'rgba(153, 102, 255, 0.6)';
                ctx.strokeStyle = 'rgba(153, 102, 255, 1)';
                ctx.lineWidth = 1;
                pts.forEach(p => {{
                    ctx.beginPath();
                    ctx.arc(toX(p.x), toY(p.y), 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                }});
            }})();
        </script>
"""

        html_content += f"""
        <h2>Raw Render Context JSON</h2>
        <pre>{json.dumps(render_context, indent=2, ensure_ascii=False)}</pre>
    </div>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path
    except Exception as e:
        logger.error(f"Error generating HTML render for hpss_extractor: {e}")
        return f"Error generating HTML: {e}"


__all__ = ["render_hpss_extractor", "render_hpss_extractor_html"]

