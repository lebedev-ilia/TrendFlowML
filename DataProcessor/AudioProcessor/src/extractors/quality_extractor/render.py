"""
Renderer для quality_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_quality_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для quality_extractor."""
    render = {
        "component": "quality_extractor",
        "summary": {},
        "basic_metrics": {},
        "dynamic_metrics": {},
        "frame_analysis": {},
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
        "device_used": str(payload.get("device_used", "cpu")),
        "duration": float(payload.get("duration", 0.0)),
        "segments_count": int(payload.get("segments_count", 0)) if "segments_count" in payload else None,
    }
    
    # Basic metrics (feature-gated)
    if "dc_offset" in payload:
        render["basic_metrics"] = {
            "dc_offset": float(payload.get("dc_offset", 0.0)),
            "dc_offset_abs": float(payload.get("dc_offset_abs", 0.0)),
            "clipping_ratio": float(payload.get("clipping_ratio", 0.0)),
            "crest_factor_db": float(payload.get("crest_factor_db", 0.0)),
        }
    
    # Dynamic metrics (feature-gated)
    if "dynamic_range_db" in payload:
        render["dynamic_metrics"] = {
            "dynamic_range_db": float(payload.get("dynamic_range_db", 0.0)),
            "snr_db": float(payload.get("snr_db", 0.0)),
        }
        if "dynamic_range_stability" in payload:
            render["dynamic_metrics"]["dynamic_range_stability"] = float(payload.get("dynamic_range_stability", 0.0))
        if "snr_stability" in payload:
            render["dynamic_metrics"]["snr_stability"] = float(payload.get("snr_stability", 0.0))
    
    # Frame analysis (feature-gated)
    if "frame_levels_distribution" in payload:
        render["frame_analysis"] = {
            "frame_levels_distribution": payload.get("frame_levels_distribution", {}),
        }
    
    # Additional ML/analytics metrics
    if "clipping_segments_count" in payload:
        render["additional_metrics"] = {
            "clipping_segments_count": int(payload.get("clipping_segments_count", 0)),
            "crest_factor_median": float(payload.get("crest_factor_median", 0.0)),
            "quality_score": float(payload.get("quality_score", 0.0)),
        }
    
    # Time series (feature-gated)
    time_series_keys = ["frame_levels_db_series", "frame_rms_series", "clipping_segments_series", "dc_offset_series", "clipping_ratio_series", "crest_factor_db_series", "dynamic_range_db_series", "snr_db_series"]
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


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Безопасное преобразование значения в float для форматирования в HTML.
    
    Args:
        value: Значение для преобразования
        default: Значение по умолчанию, если преобразование не удалось
    
    Returns:
        float значение
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def render_quality_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага quality_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_quality_extractor(npz_data, meta)
    
    # Безопасное извлечение данных для форматирования
    summary = render.get("summary", {})
    basic_metrics = render.get("basic_metrics", {})
    dynamic_metrics = render.get("dynamic_metrics", {})
    additional_metrics = render.get("additional_metrics", {})
    time_series = render.get("time_series", {})
    
    # Форматируем значения заранее, чтобы избежать вложенных f-строк
    sample_rate = summary.get('sample_rate', 'N/A')
    device_used = summary.get('device_used', 'N/A')
    duration = f"{safe_float(summary.get('duration', 0.0)):.2f}"
    segments_count_html = f"<p><strong>Segments Count:</strong> {summary.get('segments_count', 'N/A')}</p>" if summary.get('segments_count') is not None else ""
    
    # Basic metrics HTML
    basic_metrics_html = ""
    if basic_metrics:
        dc_offset = f"{safe_float(basic_metrics.get('dc_offset', 0.0)):.6f}"
        dc_offset_abs = f"{safe_float(basic_metrics.get('dc_offset_abs', 0.0)):.6f}"
        clipping_ratio = f"{safe_float(basic_metrics.get('clipping_ratio', 0.0)):.4f}"
        crest_factor_db = f"{safe_float(basic_metrics.get('crest_factor_db', 0.0)):.2f}"
        basic_metrics_html = f"""
    <div class="metrics">
        <h2>Basic Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td><strong>DC Offset</strong></td>
                <td>{dc_offset}</td>
            </tr>
            <tr>
                <td><strong>DC Offset (abs)</strong></td>
                <td>{dc_offset_abs}</td>
            </tr>
            <tr>
                <td><strong>Clipping Ratio</strong></td>
                <td>{clipping_ratio}</td>
            </tr>
            <tr>
                <td><strong>Crest Factor (dB)</strong></td>
                <td>{crest_factor_db} dB</td>
            </tr>
        </table>
    </div>
"""
    
    # Dynamic metrics HTML
    dynamic_metrics_html = ""
    if dynamic_metrics:
        dynamic_range_db = f"{safe_float(dynamic_metrics.get('dynamic_range_db', 0.0)):.2f}"
        snr_db = f"{safe_float(dynamic_metrics.get('snr_db', 0.0)):.2f}"
        dr_stability_row = ""
        if dynamic_metrics.get('dynamic_range_stability') is not None:
            dr_stability = f"{safe_float(dynamic_metrics.get('dynamic_range_stability', 0.0)):.4f}"
            dr_stability_row = f"<tr><td><strong>Dynamic Range Stability</strong></td><td>{dr_stability}</td></tr>"
        snr_stability_row = ""
        if dynamic_metrics.get('snr_stability') is not None:
            snr_stability = f"{safe_float(dynamic_metrics.get('snr_stability', 0.0)):.4f}"
            snr_stability_row = f"<tr><td><strong>SNR Stability</strong></td><td>{snr_stability}</td></tr>"
        dynamic_metrics_html = f"""
    <div class="metrics">
        <h2>Dynamic Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td><strong>Dynamic Range (dB)</strong></td>
                <td>{dynamic_range_db} dB</td>
            </tr>
            <tr>
                <td><strong>SNR (dB)</strong></td>
                <td>{snr_db} dB</td>
            </tr>
            {dr_stability_row}
            {snr_stability_row}
        </table>
    </div>
"""
    
    # Additional metrics HTML
    additional_metrics_html = ""
    if additional_metrics:
        clipping_segments_count = additional_metrics.get('clipping_segments_count', 0)
        crest_factor_median = f"{safe_float(additional_metrics.get('crest_factor_median', 0.0)):.2f}"
        quality_score = f"{safe_float(additional_metrics.get('quality_score', 0.0)):.4f}"
        additional_metrics_html = f"""
    <div class="metrics">
        <h2>Additional Metrics</h2>
        <ul>
            <li><strong>Clipping Segments Count:</strong> {clipping_segments_count}</li>
            <li><strong>Crest Factor Median:</strong> {crest_factor_median} dB</li>
            <li><strong>Quality Score:</strong> {quality_score}</li>
        </ul>
    </div>
"""
    
    # Time series HTML
    time_series_html = ""
    if time_series:
        time_series_keys_str = ', '.join(time_series.keys())
        time_series_html = f"""
    <div class="time-series">
        <h2>Time Series</h2>
        <p>Time series data available: {time_series_keys_str}</p>
    </div>
"""
    
    # Raw JSON
    raw_json = json.dumps(render, indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quality Extractor Debug</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .metrics {{ margin: 10px 0; }}
        .metrics table {{ border-collapse: collapse; width: 100%; }}
        .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metrics th {{ background-color: #4CAF50; color: white; }}
        .time-series {{ margin: 20px 0; }}
        .chart {{ width: 100%; height: 300px; border: 1px solid #ddd; margin: 10px 0; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Quality Extractor Debug</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Sample Rate:</strong> {sample_rate} Hz</p>
        <p><strong>Device:</strong> {device_used}</p>
        <p><strong>Duration:</strong> {duration} sec</p>
        {segments_count_html}
    </div>
    {basic_metrics_html}
    {dynamic_metrics_html}
    {additional_metrics_html}
    {time_series_html}
    
    <div class="summary">
        <h2>Raw Data (JSON)</h2>
        <pre>{raw_json}</pre>
    </div>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path

__all__ = ["render_quality_extractor", "render_quality_extractor_html"]
