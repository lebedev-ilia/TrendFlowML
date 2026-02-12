"""
Renderer для mel_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_mel_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для mel_extractor."""
    render = {
        "component": "mel_extractor",
        "summary": {},
        "basic_features": {},
        "statistics": {},
        "spectral_features": {},
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
        "n_fft": int(safe_get("n_fft", 2048)),
        "hop_length": int(safe_get("hop_length", 512)),
        "n_mels": int(safe_get("n_mels", 128)),
        "fmin": float(safe_get("fmin", 0.0)),
        "fmax": float(safe_get("fmax", 0.0)) if safe_get("fmax") is not None else None,
        "power": float(safe_get("power", 2.0)),
        "device_used": str(safe_get("device_used", "cpu")),
        "duration": float(safe_get("duration", 0.0)),
        "segments_count": int(safe_get("segments_count", 0)) if is_non_empty(safe_get("segments_count")) else None,
    }
    
    # Basic features (feature-gated) - read from NPZ
    mel_shape_0 = safe_get("mel_shape_0")
    mel_shape_1 = safe_get("mel_shape_1")
    if mel_shape_0 is not None and mel_shape_1 is not None:
        render["basic_features"] = {
            "mel_shape": (int(mel_shape_0), int(mel_shape_1)),
            "mel_elements": int(safe_get("mel_elements", 0)),
        }
    
    # Statistics (feature-gated) - read from NPZ
    mel_mean = safe_get("mel_mean")
    if is_non_empty(mel_mean):
        render["statistics"] = {}
        if is_non_empty(mel_mean):
            render["statistics"]["mel_mean"] = mel_mean if isinstance(mel_mean, list) else (mel_mean.tolist() if isinstance(mel_mean, np.ndarray) else mel_mean)
        mel_std = safe_get("mel_std")
        if is_non_empty(mel_std):
            render["statistics"]["mel_std"] = mel_std if isinstance(mel_std, list) else (mel_std.tolist() if isinstance(mel_std, np.ndarray) else mel_std)
        mel_min = safe_get("mel_min")
        if is_non_empty(mel_min):
            render["statistics"]["mel_min"] = mel_min if isinstance(mel_min, list) else (mel_min.tolist() if isinstance(mel_min, np.ndarray) else mel_min)
        mel_max = safe_get("mel_max")
        if is_non_empty(mel_max):
            render["statistics"]["mel_max"] = mel_max if isinstance(mel_max, list) else (mel_max.tolist() if isinstance(mel_max, np.ndarray) else mel_max)
        freq_mean = safe_get("freq_mean")
        if is_non_empty(freq_mean):
            render["statistics"]["freq_mean"] = freq_mean if isinstance(freq_mean, list) else (freq_mean.tolist() if isinstance(freq_mean, np.ndarray) else freq_mean)
        freq_std = safe_get("freq_std")
        if is_non_empty(freq_std):
            render["statistics"]["freq_std"] = freq_std if isinstance(freq_std, list) else (freq_std.tolist() if isinstance(freq_std, np.ndarray) else freq_std)
    
    # Spectral features (feature-gated) - read from NPZ
    spectral_centroid = safe_get("spectral_centroid")
    if is_non_empty(spectral_centroid):
        render["spectral_features"] = {}
        render["spectral_features"]["spectral_centroid"] = spectral_centroid if isinstance(spectral_centroid, list) else (spectral_centroid.tolist() if isinstance(spectral_centroid, np.ndarray) else spectral_centroid)
        spectral_bandwidth = safe_get("spectral_bandwidth")
        if is_non_empty(spectral_bandwidth):
            render["spectral_features"]["spectral_bandwidth"] = spectral_bandwidth if isinstance(spectral_bandwidth, list) else (spectral_bandwidth.tolist() if isinstance(spectral_bandwidth, np.ndarray) else spectral_bandwidth)
    
    # Additional ML/analytics metrics - read from NPZ
    mel_energy = safe_get("mel_energy")
    if is_non_empty(mel_energy) and mel_energy != 0.0:
        render["additional_metrics"] = {
            "mel_energy": float(mel_energy),
            "mel_centroid": float(safe_get("mel_centroid", 0.0)),
            "mel_bandwidth": float(safe_get("mel_bandwidth", 0.0)),
            "mel_rolloff": float(safe_get("mel_rolloff", 0.0)),
            "mel_flatness": float(safe_get("mel_flatness", 0.0)),
            "mel_stability": float(safe_get("mel_stability", 0.0)),
        }
    
    # Time series (feature-gated) - read from NPZ
    time_series_keys = ["mel_series"]
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


def render_mel_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага mel_extractor результатов.
    
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
    render = render_mel_extractor(npz_data, meta)
    
    # Безопасное извлечение данных для форматирования
    summary = render.get("summary", {})
    basic_features = render.get("basic_features", {})
    statistics = render.get("statistics", {})
    spectral_features = render.get("spectral_features", {})
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
    mel_series_data = None
    segment_centers = time_series.get("segment_centers_sec", [])
    
    # Try to get data from render context or load directly from NPZ
    if "mel_series" in time_series and "data" in time_series["mel_series"]:
        mel_series_data = time_series["mel_series"]["data"]
    elif "mel_series" in npz_raw.files:
        arr = npz_raw["mel_series"]
        if arr.size > 0:
            # Flatten if 2D and sample if too large
            if arr.ndim == 2:
                arr = arr.flatten()
            if arr.size > 10000:
                step = arr.size // 10000
                arr = arr[::step]
            mel_series_data = arr.tolist()[:10000]
    
    # Statistics for distribution charts
    mel_mean_stats = statistics.get("mel_mean", [])
    mel_std_stats = statistics.get("mel_std", [])
    spectral_centroid_data = spectral_features.get("spectral_centroid", [])
    spectral_bandwidth_data = spectral_features.get("spectral_bandwidth", [])
    
    # Ensure lists are not empty and have valid data
    if isinstance(mel_mean_stats, np.ndarray):
        mel_mean_stats = mel_mean_stats.tolist() if mel_mean_stats.size > 0 else []
    if isinstance(mel_std_stats, np.ndarray):
        mel_std_stats = mel_std_stats.tolist() if mel_std_stats.size > 0 else []
    if isinstance(spectral_centroid_data, np.ndarray):
        spectral_centroid_data = spectral_centroid_data.tolist() if spectral_centroid_data.size > 0 else []
    if isinstance(spectral_bandwidth_data, np.ndarray):
        spectral_bandwidth_data = spectral_bandwidth_data.tolist() if spectral_bandwidth_data.size > 0 else []
    
    # Ensure we have lists, not other types
    if not isinstance(mel_mean_stats, list):
        mel_mean_stats = []
    if not isinstance(mel_std_stats, list):
        mel_std_stats = []
    if not isinstance(spectral_centroid_data, list):
        spectral_centroid_data = []
    if not isinstance(spectral_bandwidth_data, list):
        spectral_bandwidth_data = []
    
    # Build HTML with Plotly charts
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>🎵 Mel Extractor Debug</title>
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
        <h1>🎵 Mel Extractor</h1>
        <div class="meta-info">
            <p><strong>Status:</strong> <span style="color: {'green' if meta.get('status') == 'ok' else 'orange' if meta.get('status') == 'empty' else 'red'}">{meta.get('status', 'unknown')}</span></p>
            <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
            <p><strong>Contract Version:</strong> {meta.get('mel_contract_version', 'unknown')}</p>
        </div>
        
        <h2>📊 Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Sample Rate</div>
                <div class="stat-value">{summary.get('sample_rate', 'N/A')} Hz</div>
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
                <div class="stat-label">Power</div>
                <div class="stat-value">{summary.get('power', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Device</div>
                <div class="stat-value">{summary.get('device_used', 'N/A')}</div>
            </div>
            {f'<div class="stat-card"><div class="stat-label">Segments</div><div class="stat-value">{summary.get("segments_count", "N/A")}</div></div>' if summary.get('segments_count') is not None else ''}
        </div>
"""
    
    # Always show at least basic info, even if no features enabled
    if not basic_features and not statistics and not spectral_features and not additional_metrics:
        html_content += """
        <div class="section">
            <h2>⚠️ No Features Enabled</h2>
            <p>No features were enabled for this extraction. Please enable at least one feature flag in the configuration.</p>
        </div>
"""
    
    # Basic features section
    if basic_features:
        mel_shape = basic_features.get("mel_shape", "N/A")
        mel_elements = basic_features.get("mel_elements", "N/A")
        html_content += f"""
        <div class="section">
            <h2>📈 Basic Features</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Mel Shape</div>
                    <div class="metric-value">{mel_shape}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mel Elements</div>
                    <div class="metric-value">{mel_elements}</div>
                </div>
            </div>
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
    
    # Mel Statistics Distribution
    if mel_mean_stats and isinstance(mel_mean_stats, list) and len(mel_mean_stats) > 0:
        html_content += f"""
        <div class="section">
            <h2>📊 Mel Statistics</h2>
            <h3>Mean Values by Mel Bin</h3>
            <div class="chart-container">
                <div id="mel-mean-chart" style="height: 400px;"></div>
            </div>
            <script>
                var meanData = [{{
                    x: {json.dumps(list(range(len(mel_mean_stats))))},
                    y: {json.dumps(mel_mean_stats)},
                    type: 'bar',
                    marker: {{
                        color: 'rgba(102, 126, 234, 0.7)',
                        line: {{ color: 'rgba(102, 126, 234, 1.0)', width: 1 }}
                    }},
                    name: 'Mel Mean'
                }}];
                var meanLayout = {{
                    title: {{
                        text: 'Mel Mean Values by Bin',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Mel Bin Index' }},
                    yaxis: {{ title: 'Mean Value (dB)' }},
                    height: 400
                }};
                Plotly.newPlot('mel-mean-chart', meanData, meanLayout);
            </script>
"""
        if mel_std_stats and len(mel_std_stats) > 0:
            html_content += f"""
            <h3>Standard Deviation by Mel Bin</h3>
            <div class="chart-container">
                <div id="mel-std-chart" style="height: 400px;"></div>
            </div>
            <script>
                var stdData = [{{
                    x: {json.dumps(list(range(len(mel_std_stats))))},
                    y: {json.dumps(mel_std_stats)},
                    type: 'bar',
                    marker: {{
                        color: 'rgba(118, 75, 162, 0.7)',
                        line: {{ color: 'rgba(118, 75, 162, 1.0)', width: 1 }}
                    }},
                    name: 'Mel Std'
                }}];
                var stdLayout = {{
                    title: {{
                        text: 'Mel Standard Deviation by Bin',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Mel Bin Index' }},
                    yaxis: {{ title: 'Standard Deviation (dB)' }},
                    height: 400
                }};
                Plotly.newPlot('mel-std-chart', stdData, stdLayout);
            </script>
"""
        html_content += """
        </div>
"""
    
    # Spectral features charts
    if spectral_centroid_data and isinstance(spectral_centroid_data, list) and len(spectral_centroid_data) > 0:
        centroid_time_indices = list(range(len(spectral_centroid_data)))
        html_content += f"""
        <div class="section">
            <h2>📈 Spectral Features</h2>
            <h3>Spectral Centroid</h3>
            <div class="chart-container">
                <div id="spectral-centroid-chart" style="height: 400px;"></div>
            </div>
            <script>
                var centroidData = [{{
                    x: {json.dumps(centroid_time_indices)},
                    y: {json.dumps(spectral_centroid_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#667eea', width: 2 }},
                    name: 'Spectral Centroid'
                }}];
                var centroidLayout = {{
                    title: {{
                        text: 'Spectral Centroid Over Time',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Frequency (Hz)' }},
                    height: 400
                }};
                Plotly.newPlot('spectral-centroid-chart', centroidData, centroidLayout);
            </script>
"""
        
        if spectral_bandwidth_data and len(spectral_bandwidth_data) > 0:
            bandwidth_time_indices = list(range(len(spectral_bandwidth_data)))
            html_content += f"""
            <h3>Spectral Bandwidth</h3>
            <div class="chart-container">
                <div id="spectral-bandwidth-chart" style="height: 400px;"></div>
            </div>
            <script>
                var bandwidthData = [{{
                    x: {json.dumps(bandwidth_time_indices)},
                    y: {json.dumps(spectral_bandwidth_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#764ba2', width: 2 }},
                    name: 'Spectral Bandwidth'
                }}];
                var bandwidthLayout = {{
                    title: {{
                        text: 'Spectral Bandwidth Over Time',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Frequency (Hz)' }},
                    height: 400
                }};
                Plotly.newPlot('spectral-bandwidth-chart', bandwidthData, bandwidthLayout);
            </script>
"""
        
        html_content += """
        </div>
"""
    
    # Time series chart
    if mel_series_data and isinstance(mel_series_data, list) and len(mel_series_data) > 0:
        time_indices = list(range(len(mel_series_data)))
        html_content += f"""
        <div class="section">
            <h2>📈 Time Series</h2>
            <h3>Mel Series</h3>
            <div class="chart-container">
                <div id="mel-series-chart" style="height: 400px;"></div>
            </div>
            <script>
                var seriesData = [{{
                    x: {json.dumps(time_indices)},
                    y: {json.dumps(mel_series_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#667eea', width: 1 }},
                    name: 'Mel Series'
                }}];
                var seriesLayout = {{
                    title: {{
                        text: 'Mel Time Series',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Mel Value (dB)' }},
                    height: 400
                }};
                Plotly.newPlot('mel-series-chart', seriesData, seriesLayout);
            </script>
        </div>
"""
    
    # Distribution histogram
    if mel_series_data and len(mel_series_data) > 0:
        html_content += f"""
        <div class="section">
            <h2>📊 Distribution</h2>
            <h3>Mel Values Distribution</h3>
            <div class="chart-container">
                <div id="mel-distribution-chart" style="height: 400px;"></div>
            </div>
            <script>
                var distData = [{{
                    x: {json.dumps(mel_series_data)},
                    type: 'histogram',
                    marker: {{
                        color: 'rgba(102, 126, 234, 0.7)',
                        line: {{ color: 'rgba(102, 126, 234, 1.0)', width: 1 }}
                    }},
                    name: 'Mel Distribution',
                    nbinsx: 50
                }}];
                var distLayout = {{
                    title: {{
                        text: 'Distribution of Mel Values',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Mel Value (dB)' }},
                    yaxis: {{ title: 'Frequency' }},
                    height: 400
                }};
                Plotly.newPlot('mel-distribution-chart', distData, distLayout);
            </script>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Mel extractor HTML render saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save mel extractor HTML render to {output_path}: {e}")
        raise

__all__ = ["render_mel_extractor", "render_mel_extractor_html"]
