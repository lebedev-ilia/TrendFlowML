"""
Renderer для hpss_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta


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
            render_context["metrics"]["dominance"] = payload.get("hpss_dominance", "unknown")

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

        # Segment centers and durations (if run_segments was used)
        segment_centers_sec = npz_data.get("segment_centers_sec")
        segment_durations_sec = npz_data.get("segment_durations_sec")
        if segment_centers_sec is not None:
            if isinstance(segment_centers_sec, np.ndarray):
                if segment_centers_sec.size > 0:
                    render_context["segments_data"]["segment_centers_sec"] = segment_centers_sec.tolist()
                else:
                    render_context["segments_data"]["segment_centers_sec"] = []
            elif isinstance(segment_centers_sec, list) and len(segment_centers_sec) > 0:
                render_context["segments_data"]["segment_centers_sec"] = segment_centers_sec
            else:
                render_context["segments_data"]["segment_centers_sec"] = []
        else:
            render_context["segments_data"]["segment_centers_sec"] = []
        
        if segment_durations_sec is not None:
            if isinstance(segment_durations_sec, np.ndarray):
                if segment_durations_sec.size > 0:
                    render_context["segments_data"]["segment_durations_sec"] = segment_durations_sec.tolist()
                else:
                    render_context["segments_data"]["segment_durations_sec"] = []
            elif isinstance(segment_durations_sec, list) and len(segment_durations_sec) > 0:
                render_context["segments_data"]["segment_durations_sec"] = segment_durations_sec
            else:
                render_context["segments_data"]["segment_durations_sec"] = []
        else:
            render_context["segments_data"]["segment_durations_sec"] = []
        
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

        # Time series charts
        if render_context["time_series"].get("harmonic_share_series"):
            harmonic_series_json = json.dumps(render_context["time_series"]["harmonic_share_series"])
            percussive_series_json = json.dumps(render_context["time_series"].get("percussive_share_series", []))
            html_content += f"""
        <h2>Time Series: Harmonic vs Percussive Share</h2>
        <div class="chart-container">
            <canvas id="shareSeriesChart"></canvas>
        </div>
        <script>
            const harmonicSeries = {harmonic_series_json};
            const percussiveSeries = {percussive_series_json};
            const labels = Array.from({{length: harmonicSeries.length}}, (_, i) => i);
            const shareCtx = document.getElementById('shareSeriesChart').getContext('2d');
            new Chart(shareCtx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Harmonic Share',
                        data: harmonicSeries,
                        fill: false,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }}, {{
                        label: 'Percussive Share',
                        data: percussiveSeries,
                        fill: false,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Frame Index'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Share (0.0-1.0)'
                            }},
                            min: 0,
                            max: 1
                        }}
                    }}
                }}
            }});
        </script>
"""

        # Segment-level data (if run_segments was used)
        if render_context["segments_data"].get("segment_centers_sec"):
            segment_centers_json = json.dumps(render_context["segments_data"]["segment_centers_sec"])
            segment_durations_json = json.dumps(render_context["segments_data"]["segment_durations_sec"])
            html_content += f"""
        <h2>Segment-Level Data</h2>
        <div class="chart-container">
            <canvas id="segmentDataChart"></canvas>
        </div>
        <script>
            const segmentCenters = {segment_centers_json};
            const segmentDurations = {segment_durations_json};
            const segmentCtx = document.getElementById('segmentDataChart').getContext('2d');
            new Chart(segmentCtx, {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Segments',
                        data: segmentCenters.map((center, i) => ({{ x: center, y: segmentDurations[i] }})),
                        backgroundColor: 'rgba(153, 102, 255, 0.6)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            type: 'linear',
                            position: 'bottom',
                            title: {{
                                display: true,
                                text: 'Center Time (seconds)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Duration (seconds)'
                            }}
                        }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return `Center: ${{context.parsed.x.toFixed(2)}}s, Duration: ${{context.parsed.y.toFixed(2)}}s`;
                                }}
                            }}
                        }}
                    }}
                }}
            }});
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

