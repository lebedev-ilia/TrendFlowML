"""
Renderer для mfcc_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_mfcc_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для mfcc_extractor."""
    render = {
        "component": "mfcc_extractor",
        "summary": {},
        "basic_features": {},
        "deltas": {},
        "time_series": {},
        "additional_metrics": {},
    }
    
    # Helper function to safely extract values from NPZ
    # Note: load_npz converts numpy arrays to lists, so we need to handle both
    def safe_get(key, default=None):
        value = npz_data.get(key, default)
        if value is None:
            return default
        # Handle numpy arrays (if not converted yet)
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            elif value.dtype == object and value.size == 1:
                return value.item()
        # Handle lists (converted from numpy arrays by load_npz)
        if isinstance(value, list):
            if len(value) == 0:
                return default
            # For scalar values stored as single-element lists
            if len(value) == 1 and not isinstance(value[0], (list, np.ndarray)):
                return value[0]
        return value
    
    # Helper to check if value is non-empty (handles lists and arrays)
    def is_non_empty(value):
        if value is None:
            return False
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, np.ndarray):
            return value.size > 0
        return value != 0 and value != 0.0
    
    # Summary - read directly from NPZ
    render["summary"] = {
        "sample_rate": int(safe_get("sample_rate", 22050)),
        "n_mfcc": int(safe_get("n_mfcc", 13)),
        "n_fft": int(safe_get("n_fft", 2048)),
        "hop_length": int(safe_get("hop_length", 512)),
        "n_mels": int(safe_get("n_mels", 128)),
        "fmin": float(safe_get("fmin", 0.0)),
        "fmax": float(safe_get("fmax", 0.0)) if safe_get("fmax") is not None else None,
        "device_used": str(safe_get("device_used", "cpu")),
        "duration": float(safe_get("duration", 0.0)),
        "segments_count": int(safe_get("segments_count", 0)) if is_non_empty(safe_get("segments_count")) else None,
    }
    
    # Basic features (feature-gated) - read from NPZ
    mfcc_features = safe_get("mfcc_features")
    if mfcc_features is not None and is_non_empty(mfcc_features):
        # Get shape from list or array
        if isinstance(mfcc_features, list):
            # For 2D list, get shape
            if len(mfcc_features) > 0 and isinstance(mfcc_features[0], list):
                shape = [len(mfcc_features), len(mfcc_features[0])]
            else:
                shape = [len(mfcc_features)]
            dtype = "float32"  # Default dtype for lists from NPZ
        else:
            shape = list(mfcc_features.shape) if hasattr(mfcc_features, 'shape') else []
            dtype = str(mfcc_features.dtype) if hasattr(mfcc_features, 'dtype') else "float32"
        
        render["basic_features"] = {
            "shape": shape,
            "dtype": dtype,
        }
        
        # Statistics from NPZ
        mfcc_mean = safe_get("mfcc_mean")
        mfcc_std = safe_get("mfcc_std")
        mfcc_min = safe_get("mfcc_min")
        mfcc_max = safe_get("mfcc_max")
        mfcc_median = safe_get("mfcc_median")
        
        if is_non_empty(mfcc_mean) or is_non_empty(mfcc_std):
            stats = {}
            if is_non_empty(mfcc_mean):
                stats["mfcc_mean"] = mfcc_mean if isinstance(mfcc_mean, list) else (mfcc_mean.tolist() if isinstance(mfcc_mean, np.ndarray) else mfcc_mean)
            if is_non_empty(mfcc_std):
                stats["mfcc_std"] = mfcc_std if isinstance(mfcc_std, list) else (mfcc_std.tolist() if isinstance(mfcc_std, np.ndarray) else mfcc_std)
            if is_non_empty(mfcc_min):
                stats["mfcc_min"] = mfcc_min if isinstance(mfcc_min, list) else (mfcc_min.tolist() if isinstance(mfcc_min, np.ndarray) else mfcc_min)
            if is_non_empty(mfcc_max):
                stats["mfcc_max"] = mfcc_max if isinstance(mfcc_max, list) else (mfcc_max.tolist() if isinstance(mfcc_max, np.ndarray) else mfcc_max)
            if is_non_empty(mfcc_median):
                stats["mfcc_median"] = mfcc_median if isinstance(mfcc_median, list) else (mfcc_median.tolist() if isinstance(mfcc_median, np.ndarray) else mfcc_median)
            if stats:
                render["basic_features"]["statistics"] = stats
    
    # Deltas (feature-gated) - read from NPZ
    delta_mean = safe_get("delta_mean")
    if is_non_empty(delta_mean):
        render["deltas"] = {}
        render["deltas"]["delta_mean"] = delta_mean if isinstance(delta_mean, list) else (delta_mean.tolist() if isinstance(delta_mean, np.ndarray) else delta_mean)
        delta_std = safe_get("delta_std")
        if is_non_empty(delta_std):
            render["deltas"]["delta_std"] = delta_std if isinstance(delta_std, list) else (delta_std.tolist() if isinstance(delta_std, np.ndarray) else delta_std)
        delta_delta_mean = safe_get("delta_delta_mean")
        if is_non_empty(delta_delta_mean):
            render["deltas"]["delta_delta_mean"] = delta_delta_mean if isinstance(delta_delta_mean, list) else (delta_delta_mean.tolist() if isinstance(delta_delta_mean, np.ndarray) else delta_delta_mean)
        delta_delta_std = safe_get("delta_delta_std")
        if is_non_empty(delta_delta_std):
            render["deltas"]["delta_delta_std"] = delta_delta_std if isinstance(delta_delta_std, list) else (delta_delta_std.tolist() if isinstance(delta_delta_std, np.ndarray) else delta_delta_std)
    
    # Additional ML/analytics metrics - read from NPZ
    mfcc_energy = safe_get("mfcc_energy")
    if is_non_empty(mfcc_energy) and mfcc_energy != 0.0:
        render["additional_metrics"] = {
            "mfcc_energy": float(mfcc_energy),
            "mfcc_centroid": float(safe_get("mfcc_centroid", 0.0)),
            "mfcc_bandwidth": float(safe_get("mfcc_bandwidth", 0.0)),
            "mfcc_skewness": float(safe_get("mfcc_skewness", 0.0)),
            "mfcc_kurtosis": float(safe_get("mfcc_kurtosis", 0.0)),
            "mfcc_correlation": float(safe_get("mfcc_correlation", 0.0)),
            "mfcc_stability": float(safe_get("mfcc_stability", 0.0)),
        }
    
    # Time series (feature-gated) - read from NPZ
    time_series_keys = ["mfcc_series", "delta_series", "delta_delta_series"]
    segment_keys = ["segment_centers_sec", "segment_durations_sec"]
    
    has_time_series = any(is_non_empty(safe_get(key)) for key in time_series_keys)
    if has_time_series:
        render["time_series"] = {}
        for key in time_series_keys:
            series = safe_get(key)
            if is_non_empty(series):
                if isinstance(series, list):
                    # Get shape from list
                    if len(series) > 0 and isinstance(series[0], list):
                        shape = [len(series), len(series[0])]
                        # Flatten for visualization (sample every Nth point if too large)
                        flat_series = []
                        for row in series:
                            flat_series.extend(row if isinstance(row, list) else [row])
                        # Sample if too large (max 10000 points for visualization)
                        if len(flat_series) > 10000:
                            step = len(flat_series) // 10000
                            flat_series = flat_series[::step]
                        render["time_series"][key] = {
                            "shape": shape,
                            "dtype": "float32",
                            "sample_count": sum(len(s) if isinstance(s, list) else 1 for s in series) if isinstance(series, list) else len(series),
                            "data": flat_series[:10000] if len(flat_series) > 10000 else flat_series,  # Limit for JSON
                        }
                    else:
                        shape = [len(series)]
                        # Sample if too large
                        sampled = series[::max(1, len(series) // 10000)] if len(series) > 10000 else series
                        render["time_series"][key] = {
                            "shape": shape,
                            "dtype": "float32",
                            "sample_count": len(series),
                            "data": sampled[:10000],
                        }
                elif isinstance(series, np.ndarray):
                    # Sample if too large
                    if series.size > 10000:
                        step = series.size // 10000
                        sampled = series[::step]
                    else:
                        sampled = series
                    render["time_series"][key] = {
                        "shape": list(series.shape),
                        "dtype": str(series.dtype),
                        "sample_count": int(series.size),
                        "data": sampled.tolist()[:10000],
                    }
        for key in segment_keys:
            series = safe_get(key)
            if is_non_empty(series):
                if isinstance(series, list):
                    render["time_series"][key] = series
                elif isinstance(series, np.ndarray):
                    render["time_series"][key] = series.tolist()
    
    return render


def render_mfcc_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага mfcc_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Load NPZ directly to get raw arrays (not converted to lists)
    import numpy as np
    npz_raw = np.load(npz_path, allow_pickle=True)
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_mfcc_extractor(npz_data, meta)
    
    # Безопасное извлечение данных для форматирования
    summary = render.get("summary", {})
    basic_features = render.get("basic_features", {})
    deltas = render.get("deltas", {})
    additional_metrics = render.get("additional_metrics", {})
    time_series = render.get("time_series", {})
    
    # Форматируем значения заранее, чтобы избежать вложенных f-строк
    def safe_float(value, default=0.0):
        """Безопасное преобразование в float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    # Extract time series data for visualization
    mfcc_series_data = None
    delta_series_data = None
    delta_delta_series_data = None
    segment_centers = time_series.get("segment_centers_sec", [])
    
    # Try to get data from render context or load directly from NPZ
    if "mfcc_series" in time_series and "data" in time_series["mfcc_series"]:
        mfcc_series_data = time_series["mfcc_series"]["data"]
    elif "mfcc_series" in npz_raw.files:
        arr = npz_raw["mfcc_series"]
        if arr.size > 0:
            # Flatten if 2D and sample if too large
            if arr.ndim == 2:
                arr = arr.flatten()
            if arr.size > 10000:
                step = arr.size // 10000
                arr = arr[::step]
            mfcc_series_data = arr.tolist()[:10000]
    
    if "delta_series" in time_series and "data" in time_series["delta_series"]:
        delta_series_data = time_series["delta_series"]["data"]
    elif "delta_series" in npz_raw.files:
        arr = npz_raw["delta_series"]
        if arr.size > 0:
            if arr.ndim == 2:
                arr = arr.flatten()
            if arr.size > 10000:
                step = arr.size // 10000
                arr = arr[::step]
            delta_series_data = arr.tolist()[:10000]
    
    if "delta_delta_series" in time_series and "data" in time_series["delta_delta_series"]:
        delta_delta_series_data = time_series["delta_delta_series"]["data"]
    elif "delta_delta_series" in npz_raw.files:
        arr = npz_raw["delta_delta_series"]
        if arr.size > 0:
            if arr.ndim == 2:
                arr = arr.flatten()
            if arr.size > 10000:
                step = arr.size // 10000
                arr = arr[::step]
            delta_delta_series_data = arr.tolist()[:10000]
    
    # Statistics for distribution charts
    mfcc_mean_stats = basic_features.get("statistics", {}).get("mfcc_mean", [])
    mfcc_std_stats = basic_features.get("statistics", {}).get("mfcc_std", [])
    
    # Build HTML with Plotly charts
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>🎵 MFCC Extractor Debug</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        h1 {{ 
            font-size: 2.5em; 
            margin-bottom: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        h2 {{ 
            font-size: 1.8em; 
            margin: 30px 0 15px 0; 
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        h3 {{
            font-size: 1.3em;
            margin: 20px 0 10px 0;
            color: #764ba2;
        }}
        .summary {{ 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .chart-container {{
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .meta-info {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .meta-info p {{
            margin: 5px 0;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 MFCC Extractor</h1>
        <div class="meta-info">
            <p><strong>Status:</strong> <span style="color: {'green' if meta.get('status') == 'ok' else 'orange' if meta.get('status') == 'empty' else 'red'}">{meta.get('status', 'unknown')}</span></p>
            <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
            <p><strong>Contract Version:</strong> {meta.get('mfcc_contract_version', 'unknown')}</p>
        </div>
        
        <h2>📊 Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Sample Rate</div>
                <div class="stat-value">{summary.get('sample_rate', 'N/A')} Hz</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">N MFCC</div>
                <div class="stat-value">{summary.get('n_mfcc', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">N FFT</div>
                <div class="stat-value">{summary.get('n_fft', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hop Length</div>
                <div class="stat-value">{summary.get('hop_length', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">N Mels</div>
                <div class="stat-value">{summary.get('n_mels', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Device</div>
                <div class="stat-value">{summary.get('device_used', 'N/A')}</div>
            </div>
            {f'<div class="stat-card"><div class="stat-label">Segments</div><div class="stat-value">{summary.get("segments_count", "N/A")}</div></div>' if summary.get('segments_count') is not None else ''}
        </div>
"""
    
    # Additional metrics section
    if additional_metrics:
        html_content += """
        <div class="section">
            <h2>📈 Additional Metrics</h2>
            <div class="metrics">
"""
        for key, value in additional_metrics.items():
            formatted_value = f"{safe_float(value, 0.0):.4f}" if isinstance(value, (int, float)) else str(value)
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
"""
        html_content += """
            </div>
        </div>
"""
    
    # MFCC Statistics Distribution
    if mfcc_mean_stats and len(mfcc_mean_stats) > 0:
        html_content += f"""
        <div class="section">
            <h2>📊 MFCC Coefficients Statistics</h2>
            <h3>Mean Values by Coefficient</h3>
            <div class="chart-container">
                <div id="mfcc-mean-chart" style="height: 400px;"></div>
            </div>
            <script>
                var meanData = [{{
                    x: {json.dumps(list(range(len(mfcc_mean_stats))))},
                    y: {json.dumps(mfcc_mean_stats)},
                    type: 'bar',
                    marker: {{
                        color: 'rgba(102, 126, 234, 0.7)',
                        line: {{ color: 'rgba(102, 126, 234, 1.0)', width: 1 }}
                    }},
                    name: 'MFCC Mean'
                }}];
                var meanLayout = {{
                    title: {{
                        text: 'MFCC Mean Values by Coefficient',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'MFCC Coefficient Index' }},
                    yaxis: {{ title: 'Mean Value' }},
                    height: 400
                }};
                Plotly.newPlot('mfcc-mean-chart', meanData, meanLayout);
            </script>
"""
        if mfcc_std_stats and len(mfcc_std_stats) > 0:
            html_content += f"""
            <h3>Standard Deviation by Coefficient</h3>
            <div class="chart-container">
                <div id="mfcc-std-chart" style="height: 400px;"></div>
            </div>
            <script>
                var stdData = [{{
                    x: {json.dumps(list(range(len(mfcc_std_stats))))},
                    y: {json.dumps(mfcc_std_stats)},
                    type: 'bar',
                    marker: {{
                        color: 'rgba(118, 75, 162, 0.7)',
                        line: {{ color: 'rgba(118, 75, 162, 1.0)', width: 1 }}
                    }},
                    name: 'MFCC Std'
                }}];
                var stdLayout = {{
                    title: {{
                        text: 'MFCC Standard Deviation by Coefficient',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'MFCC Coefficient Index' }},
                    yaxis: {{ title: 'Standard Deviation' }},
                    height: 400
                }};
                Plotly.newPlot('mfcc-std-chart', stdData, stdLayout);
            </script>
"""
        html_content += """
        </div>
"""
    
    # Time series charts
    if mfcc_series_data and len(mfcc_series_data) > 0:
        time_indices = list(range(len(mfcc_series_data)))
        html_content += f"""
        <div class="section">
            <h2>📈 Time Series</h2>
            <h3>MFCC Series</h3>
            <div class="chart-container">
                <div id="mfcc-series-chart" style="height: 400px;"></div>
            </div>
            <script>
                var seriesData = [{{
                    x: {json.dumps(time_indices)},
                    y: {json.dumps(mfcc_series_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#667eea', width: 1 }},
                    name: 'MFCC Series'
                }}];
                var seriesLayout = {{
                    title: {{
                        text: 'MFCC Time Series',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'MFCC Value' }},
                    height: 400
                }};
                Plotly.newPlot('mfcc-series-chart', seriesData, seriesLayout);
            </script>
"""
        
        if delta_series_data and len(delta_series_data) > 0:
            delta_time_indices = list(range(len(delta_series_data)))
            html_content += f"""
            <h3>Delta Series</h3>
            <div class="chart-container">
                <div id="delta-series-chart" style="height: 400px;"></div>
            </div>
            <script>
                var deltaData = [{{
                    x: {json.dumps(delta_time_indices)},
                    y: {json.dumps(delta_series_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#764ba2', width: 1 }},
                    name: 'Delta Series'
                }}];
                var deltaLayout = {{
                    title: {{
                        text: 'Delta (First Derivative) Time Series',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Delta Value' }},
                    height: 400
                }};
                Plotly.newPlot('delta-series-chart', deltaData, deltaLayout);
            </script>
"""
        
        if delta_delta_series_data and len(delta_delta_series_data) > 0:
            delta_delta_time_indices = list(range(len(delta_delta_series_data)))
            html_content += f"""
            <h3>Delta-Delta Series</h3>
            <div class="chart-container">
                <div id="delta-delta-series-chart" style="height: 400px;"></div>
            </div>
            <script>
                var deltaDeltaData = [{{
                    x: {json.dumps(delta_delta_time_indices)},
                    y: {json.dumps(delta_delta_series_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#f093fb', width: 1 }},
                    name: 'Delta-Delta Series'
                }}];
                var deltaDeltaLayout = {{
                    title: {{
                        text: 'Delta-Delta (Second Derivative) Time Series',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Delta-Delta Value' }},
                    height: 400
                }};
                Plotly.newPlot('delta-delta-series-chart', deltaDeltaData, deltaDeltaLayout);
            </script>
"""
        
        html_content += """
        </div>
"""
    
    # Distribution histogram
    if mfcc_series_data and len(mfcc_series_data) > 0:
        html_content += f"""
        <div class="section">
            <h2>📊 Distribution</h2>
            <h3>MFCC Values Distribution</h3>
            <div class="chart-container">
                <div id="mfcc-distribution-chart" style="height: 400px;"></div>
            </div>
            <script>
                var distData = [{{
                    x: {json.dumps(mfcc_series_data)},
                    type: 'histogram',
                    marker: {{
                        color: 'rgba(102, 126, 234, 0.7)',
                        line: {{ color: 'rgba(102, 126, 234, 1.0)', width: 1 }}
                    }},
                    name: 'MFCC Distribution',
                    nbinsx: 50
                }}];
                var distLayout = {{
                    title: {{
                        text: 'Distribution of MFCC Values',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'MFCC Value' }},
                    yaxis: {{ title: 'Frequency' }},
                    height: 400
                }};
                Plotly.newPlot('mfcc-distribution-chart', distData, distLayout);
            </script>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path

__all__ = ["render_mfcc_extractor", "render_mfcc_extractor_html"]
