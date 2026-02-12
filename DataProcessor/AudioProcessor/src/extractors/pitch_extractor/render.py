"""
Renderer для pitch_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_pitch_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для pitch_extractor."""
    render = {
        "component": "pitch_extractor",
        "summary": {},
        "basic_stats": {},
        "stability_metrics": {},
        "delta_features": {},
        "method_stats": {},
        "time_series": {},
    }
    
    # Extract scalar features
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
    
    # Summary
    render["summary"] = {
        "sample_rate": int(features.get("sample_rate", 22050)),
        "device_used": "cuda" if features.get("device_used", 0.0) > 0.5 else "cpu",
        "f0_method": meta.get("f0_method", "unknown"),
    }
    
    # Basic stats (feature-gated)
    if "f0_mean" in features:
        render["basic_stats"] = {
            "f0_mean": float(features.get("f0_mean", 0.0)),
            "f0_std": float(features.get("f0_std", 0.0)),
            "f0_min": float(features.get("f0_min", 0.0)),
            "f0_max": float(features.get("f0_max", 0.0)),
            "f0_median": float(features.get("f0_median", 0.0)),
        }
        
        # Additional ML/analytics metrics
        if "pitch_contour_smoothness" in features:
            render["basic_stats"].update({
                "pitch_contour_smoothness": float(features.get("pitch_contour_smoothness", 0.0)),
                "pitch_jump_count": int(features.get("pitch_jump_count", 0)),
                "pitch_centroid": float(features.get("pitch_centroid", 0.0)),
                "pitch_skewness": float(features.get("pitch_skewness", 0.0)),
                "pitch_kurtosis": float(features.get("pitch_kurtosis", 0.0)),
            })
        
        # Pitch octave distribution
        pitch_octave_distribution = npz_data.get("pitch_octave_distribution")
        if pitch_octave_distribution is not None:
            if isinstance(pitch_octave_distribution, np.ndarray) and pitch_octave_distribution.dtype == object:
                pitch_octave_distribution = pitch_octave_distribution.item() if pitch_octave_distribution.size == 1 else {}
            if isinstance(pitch_octave_distribution, dict):
                render["basic_stats"]["pitch_octave_distribution"] = {str(k): float(v) for k, v in pitch_octave_distribution.items()}
    
    # Stability metrics (feature-gated)
    if "pitch_variation" in features:
        render["stability_metrics"] = {
            "pitch_variation": float(features.get("pitch_variation", 0.0)),
            "pitch_stability": float(features.get("pitch_stability", 0.0)),
            "pitch_range": float(features.get("pitch_range", 0.0)),
        }
    
    # Delta features (feature-gated)
    if "f0_delta_mean" in features:
        render["delta_features"] = {
            "f0_delta_mean": float(features.get("f0_delta_mean", 0.0)),
            "f0_delta_std": float(features.get("f0_delta_std", 0.0)),
            "f0_delta_abs_mean": float(features.get("f0_delta_abs_mean", 0.0)),
        }
    
    # Method stats (feature-gated)
    if "f0_mean_pyin" in features:
        render["method_stats"] = {
            "pyin": {
                "f0_mean": float(features.get("f0_mean_pyin", 0.0)),
                "f0_std": float(features.get("f0_std_pyin", 0.0)),
                "f0_min": float(features.get("f0_min_pyin", 0.0)),
                "f0_max": float(features.get("f0_max_pyin", 0.0)),
                "f0_median": float(features.get("f0_median_pyin", 0.0)),
                "f0_count": int(features.get("f0_count_pyin", 0)),
                "voiced_fraction": float(features.get("voiced_fraction_pyin", 0.0)),
                "voiced_probability_mean": float(features.get("voiced_probability_mean_pyin", 0.0)),
            },
            "yin": {
                "f0_mean": float(features.get("f0_mean_yin", 0.0)),
                "f0_std": float(features.get("f0_std_yin", 0.0)),
                "f0_min": float(features.get("f0_min_yin", 0.0)),
                "f0_max": float(features.get("f0_max_yin", 0.0)),
                "f0_median": float(features.get("f0_median_yin", 0.0)),
                "f0_count": int(features.get("f0_count_yin", 0)),
            },
        }
        
        # torchcrepe stats (if used)
        if "f0_mean_torchcrepe" in features:
            render["method_stats"]["torchcrepe"] = {
                "f0_mean": float(features.get("f0_mean_torchcrepe", 0.0)),
                "f0_std": float(features.get("f0_std_torchcrepe", 0.0)),
                "f0_min": float(features.get("f0_min_torchcrepe", 0.0)),
                "f0_max": float(features.get("f0_max_torchcrepe", 0.0)),
                "f0_median": float(features.get("f0_median_torchcrepe", 0.0)),
                "f0_count": int(features.get("f0_count_torchcrepe", 0)),
            }
    
    # Time series (feature-gated)
    f0_series_pyin = npz_data.get("f0_series_pyin")
    f0_series_yin = npz_data.get("f0_series_yin")
    f0_series_torchcrepe = npz_data.get("f0_series_torchcrepe")
    f0_series = npz_data.get("f0_series")
    segment_centers_sec = npz_data.get("segment_centers_sec")
    segment_durations_sec = npz_data.get("segment_durations_sec")
    
    if f0_series_pyin is not None or f0_series_yin is not None or f0_series_torchcrepe is not None or f0_series is not None:
        render["time_series"] = {}
        if f0_series_pyin is not None:
            if isinstance(f0_series_pyin, np.ndarray):
                f0_series_pyin = f0_series_pyin.tolist()
            render["time_series"]["f0_series_pyin"] = f0_series_pyin
        if f0_series_yin is not None:
            if isinstance(f0_series_yin, np.ndarray):
                f0_series_yin = f0_series_yin.tolist()
            render["time_series"]["f0_series_yin"] = f0_series_yin
        if f0_series_torchcrepe is not None:
            if isinstance(f0_series_torchcrepe, np.ndarray):
                f0_series_torchcrepe = f0_series_torchcrepe.tolist()
            render["time_series"]["f0_series_torchcrepe"] = f0_series_torchcrepe
        if f0_series is not None:
            if isinstance(f0_series, np.ndarray):
                f0_series = f0_series.tolist()
            render["time_series"]["f0_series"] = f0_series
        if segment_centers_sec is not None:
            if isinstance(segment_centers_sec, np.ndarray):
                segment_centers_sec = segment_centers_sec.tolist()
            render["time_series"]["segment_centers_sec"] = segment_centers_sec
        if segment_durations_sec is not None:
            if isinstance(segment_durations_sec, np.ndarray):
                segment_durations_sec = segment_durations_sec.tolist()
            render["time_series"]["segment_durations_sec"] = segment_durations_sec
    
    return render


def render_pitch_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML render для pitch_extractor (debug mode).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_pitch_extractor(npz_data, meta)
    
    # Extract data for visualization
    summary = render.get("summary", {})
    basic_stats = render.get("basic_stats", {})
    stability_metrics = render.get("stability_metrics", {})
    delta_features = render.get("delta_features", {})
    method_stats = render.get("method_stats", {})
    time_series = render.get("time_series", {})
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pitch Extractor Debug View</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #e3f2fd; padding: 15px; border-radius: 5px; }}
        .metric-label {{ font-size: 0.9em; color: #1976d2; }}
        .metric-value {{ font-size: 1.3em; font-weight: bold; color: #0d47a1; }}
        .section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 5px; }}
        .chart-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pitch Extractor Debug View</h1>
        <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
        <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
        <p><strong>Contract Version:</strong> {meta.get('pitch_contract_version', 'unknown')}</p>
        <p><strong>Method:</strong> {summary.get('f0_method', 'unknown')}</p>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Sample Rate (Hz)</div>
                <div class="stat-value">{summary.get('sample_rate', 22050)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Device</div>
                <div class="stat-value">{summary.get('device_used', 'cpu')}</div>
            </div>
        </div>
"""
    
    # Basic stats
    if basic_stats:
        html_content += """
        <div class="section">
            <h2>Basic Statistics</h2>
            <div class="metrics">
"""
        for key, value in basic_stats.items():
            if key != "pitch_octave_distribution":
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value:.2f if isinstance(value, float) else value}</div>
                </div>
"""
        html_content += """
            </div>
"""
        if "pitch_octave_distribution" in basic_stats:
            html_content += """
            <h3>Pitch Octave Distribution</h3>
            <div class="summary">
"""
            for octave, ratio in basic_stats["pitch_octave_distribution"].items():
                html_content += f"""
                <div class="stat-card">
                    <div class="stat-label">{octave}</div>
                    <div class="stat-value">{ratio:.1%}</div>
                </div>
"""
            html_content += """
            </div>
"""
        html_content += """
        </div>
"""
    
    # Stability metrics
    if stability_metrics:
        html_content += """
        <div class="section">
            <h2>Stability Metrics</h2>
            <div class="metrics">
"""
        for key, value in stability_metrics.items():
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                <div class="metric-value">{value:.2f if isinstance(value, float) else value}</div>
            </div>
"""
        html_content += """
            </div>
        </div>
"""
    
    # Delta features
    if delta_features:
        html_content += """
        <div class="section">
            <h2>Delta Features</h2>
            <div class="metrics">
"""
        for key, value in delta_features.items():
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                <div class="metric-value">{value:.2f if isinstance(value, float) else value}</div>
            </div>
"""
        html_content += """
            </div>
        </div>
"""
    
    # Method stats
    if method_stats:
        html_content += """
        <div class="section">
            <h2>Method Statistics</h2>
"""
        for method, stats in method_stats.items():
            html_content += f"""
            <h3>{method.upper()}</h3>
            <div class="metrics">
"""
            for key, value in stats.items():
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value:.2f if isinstance(value, float) else value}</div>
                </div>
"""
            html_content += """
            </div>
"""
        html_content += """
        </div>
"""
    
    # Time series visualization
    if time_series:
        html_content += """
        <div class="section">
            <h2>Time Series</h2>
            <div class="chart-container" id="pitch-chart"></div>
        </div>
        <script>
            var timeSeries = """ + json.dumps(time_series) + """;
            var traces = [];
            if (timeSeries.f0_series_pyin && timeSeries.f0_series_pyin.length > 0) {
                traces.push({
                    x: Array.from({length: timeSeries.f0_series_pyin.length}, (_, i) => i),
                    y: timeSeries.f0_series_pyin,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'PYIN',
                    line: {color: 'blue'}
                });
            }
            if (timeSeries.f0_series_yin && timeSeries.f0_series_yin.length > 0) {
                traces.push({
                    x: Array.from({length: timeSeries.f0_series_yin.length}, (_, i) => i),
                    y: timeSeries.f0_series_yin,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'YIN',
                    line: {color: 'green'}
                });
            }
            if (timeSeries.f0_series_torchcrepe && timeSeries.f0_series_torchcrepe.length > 0) {
                traces.push({
                    x: Array.from({length: timeSeries.f0_series_torchcrepe.length}, (_, i) => i),
                    y: timeSeries.f0_series_torchcrepe,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'torchcrepe',
                    line: {color: 'red'}
                });
            }
            if (timeSeries.f0_series && timeSeries.segment_centers_sec) {
                traces.push({
                    x: timeSeries.segment_centers_sec,
                    y: timeSeries.f0_series,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Segments',
                    line: {color: 'purple'}
                });
            }
            if (traces.length > 0) {
                Plotly.newPlot('pitch-chart', traces, {
                    title: 'Pitch (f0) Time Series',
                    xaxis: {title: 'Time (samples or seconds)'},
                    yaxis: {title: 'Frequency (Hz)'}
                });
            }
        </script>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved pitch HTML render to {output_path}")
    return output_path

__all__ = ["render_pitch_extractor", "render_pitch_extractor_html"]
