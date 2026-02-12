"""
Renderer для spectral_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_spectral_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для spectral_extractor."""
    render = {
        "component": "spectral_extractor",
        "summary": {},
        "basic_features": {},
        "contrast": {},
        "advanced_features": {},
        "time_series": {},
        "additional_metrics": {},
    }
    
    # Extract payload
    payload = npz_data.get("payload")
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item() if payload.size == 1 else {}
    if not isinstance(payload, dict):
        payload = {}
    
    # Summary
    render["summary"] = {
        "sample_rate": int(payload.get("sample_rate", 22050)),
        "hop_length": int(payload.get("hop_length", 512)),
        "n_fft": int(payload.get("n_fft", 2048)),
        "device_used": str(payload.get("device_used", "cpu")),
        "duration": float(payload.get("duration", 0.0)),
        "segments_count": int(payload.get("segments_count", 0)) if "segments_count" in payload else None,
    }
    
    # Basic features (feature-gated)
    if "spectral_centroid_stats" in payload:
        render["basic_features"] = {
            "centroid": payload.get("spectral_centroid_stats", {}),
            "bandwidth": payload.get("spectral_bandwidth_stats", {}),
            "flatness": payload.get("spectral_flatness_stats", {}),
            "rolloff": payload.get("spectral_rolloff_stats", {}),
            "zcr": payload.get("zcr_stats", {}),
        }
    
    # Contrast (feature-gated)
    if "spectral_contrast_stats" in payload:
        render["contrast"] = {
            "stats": payload.get("spectral_contrast_stats", {}),
            "variance": float(payload.get("spectral_contrast_variance", 0.0)),
        }
        if "spectral_contrast_bands" in payload:
            contrast_bands = payload.get("spectral_contrast_bands")
            if isinstance(contrast_bands, list):
                render["contrast"]["bands"] = contrast_bands
    
    # Advanced features (feature-gated)
    if "spectral_slope_stats" in payload:
        render["advanced_features"] = {
            "slope": payload.get("spectral_slope_stats", {}),
            "slope_stability": float(payload.get("spectral_slope_stability", 0.0)),
        }
        if "spectral_flatness_db_stats" in payload:
            render["advanced_features"]["flatness_db"] = payload.get("spectral_flatness_db_stats", {})
    
    # Additional ML/analytics metrics
    if "spectral_centroid_median" in payload:
        render["additional_metrics"] = {
            "centroid_median": float(payload.get("spectral_centroid_median", 0.0)),
            "bandwidth_ratio": float(payload.get("spectral_bandwidth_ratio", 0.0)),
            "rolloff_ratio": float(payload.get("spectral_rolloff_ratio", 0.0)),
            "flatness_entropy": float(payload.get("spectral_flatness_entropy", 0.0)),
            "features_correlation": payload.get("spectral_features_correlation", {}),
        }
    
    # Time series (feature-gated)
    time_series_keys = ["centroid_series", "bandwidth_series", "flatness_series", "rolloff_series", "zcr_series", "contrast_series", "slope_series"]
    segment_keys = ["segment_centers_sec", "segment_durations_sec"]
    
    has_time_series = any(key in payload for key in time_series_keys)
    if has_time_series:
        render["time_series"] = {}
        for key in time_series_keys:
            if key in payload:
                series = payload.get(key)
                if isinstance(series, list):
                    render["time_series"][key] = series
                elif isinstance(series, np.ndarray):
                    render["time_series"][key] = series.tolist()
        for key in segment_keys:
            if key in payload:
                series = payload.get(key)
                if isinstance(series, list):
                    render["time_series"][key] = series
                elif isinstance(series, np.ndarray):
                    render["time_series"][key] = series.tolist()
    
    return render


def render_spectral_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага spectral_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_spectral_extractor(npz_data, meta)
    
    # Helper function to safely get float value
    def safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except (ValueError, TypeError):
            return default
    
    # Build HTML sections
    summary_html = f"""
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Sample Rate:</strong> {render['summary'].get('sample_rate', 'N/A')} Hz</p>
        <p><strong>Hop Length:</strong> {render['summary'].get('hop_length', 'N/A')} samples</p>
        <p><strong>N_FFT:</strong> {render['summary'].get('n_fft', 'N/A')} samples</p>
        <p><strong>Device:</strong> {render['summary'].get('device_used', 'N/A')}</p>
        <p><strong>Duration:</strong> {safe_float(render['summary'].get('duration', 0.0)):.2f} sec</p>
"""
    if render['summary'].get('segments_count') is not None:
        summary_html += f"        <p><strong>Segments Count:</strong> {render['summary'].get('segments_count', 'N/A')}</p>\n"
    summary_html += "    </div>\n"
    
    # Basic features section
    basic_features_html = ""
    if render.get('basic_features'):
        rows = []
        for name, stats in [
            ('Centroid', render['basic_features'].get('centroid', {})),
            ('Bandwidth', render['basic_features'].get('bandwidth', {})),
            ('Flatness', render['basic_features'].get('flatness', {})),
            ('Rolloff', render['basic_features'].get('rolloff', {})),
            ('ZCR', render['basic_features'].get('zcr', {})),
        ]:
            if stats:
                rows.append(f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{safe_float(stats.get('mean', 0.0)):.4f}</td>
                <td>{safe_float(stats.get('std', 0.0)):.4f}</td>
                <td>{safe_float(stats.get('min', 0.0)):.4f}</td>
                <td>{safe_float(stats.get('max', 0.0)):.4f}</td>
                <td>{safe_float(stats.get('median', 0.0)):.4f}</td>
            </tr>
""")
        if rows:
            basic_features_html = f"""
    <div class="stats">
        <h2>Basic Features</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
            </tr>
            {''.join(rows)}
        </table>
    </div>
"""
    
    # Additional metrics section
    additional_metrics_html = ""
    if render.get('additional_metrics'):
        additional_metrics = render.get('additional_metrics', {})
        centroid_median = safe_float(additional_metrics.get('centroid_median', 0.0))
        bandwidth_ratio = safe_float(additional_metrics.get('bandwidth_ratio', 0.0))
        rolloff_ratio = safe_float(additional_metrics.get('rolloff_ratio', 0.0))
        flatness_entropy = safe_float(additional_metrics.get('flatness_entropy', 0.0))
        additional_metrics_html = f"""
    <div class="stats">
        <h2>Additional Metrics</h2>
        <ul>
            <li><strong>Centroid Median:</strong> {centroid_median:.4f} Hz</li>
            <li><strong>Bandwidth Ratio:</strong> {bandwidth_ratio:.4f}</li>
            <li><strong>Rolloff Ratio:</strong> {rolloff_ratio:.4f}</li>
            <li><strong>Flatness Entropy:</strong> {flatness_entropy:.4f}</li>
        </ul>
    </div>
"""
    
    # Time series section
    time_series_html = ""
    if render.get('time_series'):
        time_series_keys = ', '.join(render['time_series'].keys())
        time_series_html = f"""
    <div class="time-series">
        <h2>Time Series</h2>
        <p>Time series data available: {time_series_keys}</p>
    </div>
"""
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Spectral Extractor Debug</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .stats {{ margin: 10px 0; }}
        .stats table {{ border-collapse: collapse; width: 100%; }}
        .stats th, .stats td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .stats th {{ background-color: #4CAF50; color: white; }}
        .time-series {{ margin: 20px 0; }}
        .chart {{ width: 100%; height: 300px; border: 1px solid #ddd; margin: 10px 0; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Spectral Extractor Debug</h1>
    
{summary_html}
{basic_features_html}
{additional_metrics_html}
{time_series_html}
    
    <div class="summary">
        <h2>Raw Data (JSON)</h2>
        <pre>{json.dumps(render, indent=2)}</pre>
    </div>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path

__all__ = ["render_spectral_extractor", "render_spectral_extractor_html"]
