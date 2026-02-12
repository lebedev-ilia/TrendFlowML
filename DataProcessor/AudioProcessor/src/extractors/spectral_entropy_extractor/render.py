"""
Renderer для spectral_entropy_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta


def render_spectral_entropy_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для spectral_entropy_extractor."""
    render = {
        "component": "spectral_entropy_extractor",
        "summary": {},
        "entropy_info": {},
        "flatness_info": {},
        "spread_info": {},
        "statistics": {},
        "time_series": {},
        "dynamics": {},
    }

    features_enabled = meta.get("features_enabled", [])

    # Extract payload (similar to key_extractor and band_energy_extractor render.py)
    payload = npz_data.get("payload")
    if payload is not None:
        if isinstance(payload, np.ndarray):
            if payload.dtype == object:
                # Object array - extract dict
                if payload.size == 1:
                    payload = payload.item()
                elif payload.size > 1:
                    # Multi-element array - try to extract first element
                    try:
                        payload = payload.item() if payload.ndim == 0 else payload[0].item() if payload.size > 0 else {}
                    except (ValueError, IndexError):
                        payload = {}
                else:
                    payload = {}
            else:
                # Non-object array - not a dict
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    
    # Fallback: always try to enrich payload from feature_names/feature_values and meta
    # Extract from feature_names/feature_values
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    
    # Enrich payload from feature_names/feature_values (if not already present)
    for i, name in enumerate(feature_names):
        if i < len(feature_values) and (name not in payload or payload[name] is None or payload[name] == 0):
            val = feature_values[i]
            # Skip NaN/None/Inf values
            if val is not None and not (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
                payload[name] = val
    
    # Enrich payload from meta (if not already present)
    for key in ["sample_rate", "n_fft", "hop_length", "use_mel", "n_mels", "smoothing_window", "duration", "segments_count"]:
        if key in meta and (key not in payload or payload.get(key) is None or payload.get(key) == 0):
            payload[key] = meta[key]

    # Summary - try payload first, then feature_names/feature_values, then meta
    # Helper to get value from multiple sources
    def get_value(key, default=0):
        if key in payload and payload[key] is not None:
            return payload[key]
        # Try feature_names/feature_values
        feature_names = npz_data.get("feature_names", [])
        feature_values = npz_data.get("feature_values", [])
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        if isinstance(feature_values, np.ndarray):
            feature_values = feature_values.tolist()
        for i, name in enumerate(feature_names):
            if name == key and i < len(feature_values):
                val = feature_values[i]
                # Handle NaN/None
                if val is None or (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
                    break
                return val
        # Try meta
        if key in meta and meta[key] is not None:
            return meta[key]
        return default
    
    render["summary"] = {
        "sample_rate": get_value("sample_rate", 0),
        "n_fft": get_value("n_fft", 0),
        "hop_length": get_value("hop_length", 0),
        "use_mel": get_value("use_mel", False) if isinstance(get_value("use_mel", False), bool) else bool(get_value("use_mel", False)),
        "n_mels": get_value("n_mels", 0),
        "smoothing_window": get_value("smoothing_window", 0),
        "duration": get_value("duration", 0.0),
        "segments_count": get_value("segments_count", 0),
    }

    # Entropy info - try payload first, then feature_names/feature_values, then meta
    entropy_stats = payload.get("spectral_entropy_stats", {})
    
    # Try to get mean/std from feature_names/feature_values if not in payload
    entropy_mean = None
    entropy_std = None
    if entropy_stats:
        entropy_mean = entropy_stats.get("mean")
        entropy_std = entropy_stats.get("std")
    else:
        # Try feature_names/feature_values
        for i, name in enumerate(feature_names):
            if name == "spectral_entropy_mean" and i < len(feature_values):
                val = feature_values[i]
                if val is not None and not (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
                    entropy_mean = val
            elif name == "spectral_entropy_std" and i < len(feature_values):
                val = feature_values[i]
                if val is not None and not (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
                    entropy_std = val
    
    # Get variance from payload or meta (always available from additional_metrics)
    entropy_variance = payload.get("spectral_entropy_variance") or meta.get("spectral_entropy_variance", 0.0)
    entropy_min = payload.get("spectral_entropy_min") or meta.get("spectral_entropy_min")
    entropy_max = payload.get("spectral_entropy_max") or meta.get("spectral_entropy_max")
    
    render["entropy_info"] = {
        "mean": float(entropy_mean) if entropy_mean is not None else 0.0,
        "std": float(entropy_std) if entropy_std is not None else 0.0,
        "min": float(entropy_min) if (entropy_min is not None and "extended_stats" in features_enabled) else None,
        "max": float(entropy_max) if (entropy_max is not None and "extended_stats" in features_enabled) else None,
        "variance": float(entropy_variance) if entropy_variance is not None else 0.0,
    }

    # Flatness info
    if "flatness" in features_enabled:
        flatness_stats = payload.get("spectral_flatness_stats", {})
        render["flatness_info"] = {
            "mean": flatness_stats.get("mean", 0.0),
            "std": flatness_stats.get("std", 0.0),
            "min": flatness_stats.get("min", 0.0) if "extended_stats" in features_enabled else None,
            "max": flatness_stats.get("max", 0.0) if "extended_stats" in features_enabled else None,
            "variance": payload.get("spectral_flatness_variance", 0.0),
        }

    # Spread info
    if "spread" in features_enabled:
        spread_stats = payload.get("spectral_spread_stats", {})
        render["spread_info"] = {
            "mean": spread_stats.get("mean", 0.0),
            "std": spread_stats.get("std", 0.0),
            "min": spread_stats.get("min", 0.0) if "extended_stats" in features_enabled else None,
            "max": spread_stats.get("max", 0.0) if "extended_stats" in features_enabled else None,
            "variance": payload.get("spectral_spread_variance", 0.0),
        }

    # Statistics if enabled
    if "basic_stats" in features_enabled:
        render["statistics"] = {
            "entropy": entropy_stats,
        }
        if "flatness" in features_enabled:
            render["statistics"]["flatness"] = payload.get("spectral_flatness_stats", {})
        if "spread" in features_enabled:
            render["statistics"]["spread"] = payload.get("spectral_spread_stats", {})

    # Time series if enabled
    if "time_series" in features_enabled:
        # Try payload first, then direct NPZ arrays
        entropy_series = payload.get("spectral_entropy_series")
        flatness_series = payload.get("spectral_flatness_series")
        spread_series = payload.get("spectral_spread_series")
        
        if not entropy_series and "spectral_entropy_series" in npz_data:
            entropy_series = npz_data["spectral_entropy_series"]
        if not flatness_series and "spectral_flatness_series" in npz_data:
            flatness_series = npz_data["spectral_flatness_series"]
        if not spread_series and "spectral_spread_series" in npz_data:
            spread_series = npz_data["spectral_spread_series"]

        if isinstance(entropy_series, np.ndarray):
            entropy_series = entropy_series.tolist()
        elif entropy_series is None:
            entropy_series = []
            
        if isinstance(flatness_series, np.ndarray):
            flatness_series = flatness_series.tolist()
        elif flatness_series is None:
            flatness_series = []
            
        if isinstance(spread_series, np.ndarray):
            spread_series = spread_series.tolist()
        elif spread_series is None:
            spread_series = []

        render["time_series"] = {
            "spectral_entropy_series": entropy_series,
            "spectral_flatness_series": flatness_series if flatness_series else None,
            "spectral_spread_series": spread_series if spread_series else None,
        }

    # Dynamics if enabled
    if "dynamics" in features_enabled:
        # Try payload first, then meta
        entropy_stability = payload.get("spectral_entropy_stability") or meta.get("spectral_entropy_stability", 0.0)
        entropy_transitions_count = payload.get("spectral_entropy_transitions_count") or meta.get("spectral_entropy_transitions_count", 0)
        entropy_transitions_rate = payload.get("spectral_entropy_transitions_rate") or meta.get("spectral_entropy_transitions_rate", 0.0)
        entropy_distribution = payload.get("spectral_entropy_distribution") or meta.get("spectral_entropy_distribution", {})
        entropy_diversity = payload.get("spectral_entropy_diversity") or meta.get("spectral_entropy_diversity", 0.0)
        
        render["dynamics"] = {
            "spectral_entropy_stability": float(entropy_stability) if entropy_stability is not None else 0.0,
            "spectral_entropy_transitions_count": int(entropy_transitions_count) if entropy_transitions_count is not None else 0,
            "spectral_entropy_transitions_rate": float(entropy_transitions_rate) if entropy_transitions_rate is not None else 0.0,
            "spectral_entropy_distribution": entropy_distribution if isinstance(entropy_distribution, dict) else {},
            "spectral_entropy_diversity": float(entropy_diversity) if entropy_diversity is not None else 0.0,
        }

    return render


def render_spectral_entropy_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага spectral_entropy_extractor результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла

    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_spectral_entropy_extractor(npz_data, meta)

    summary = render.get("summary", {})
    entropy_info = render.get("entropy_info", {})
    flatness_info = render.get("flatness_info", {})
    spread_info = render.get("spread_info", {})
    time_series = render.get("time_series", {})
    dynamics = render.get("dynamics", {})

    # Prepare data for visualization
    entropy_series = time_series.get("spectral_entropy_series", [])
    flatness_series = time_series.get("spectral_entropy_series", [])
    spread_series = time_series.get("spectral_spread_series", [])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Spectral Entropy Extractor Debug</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Spectral Entropy Extractor Debug</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Sample Rate:</span>
            <span class="metric-value">{summary.get("sample_rate", 0)} Hz</span>
        </div>
        <div class="metric">
            <span class="metric-label">N_FFT:</span>
            <span class="metric-value">{summary.get("n_fft", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Hop Length:</span>
            <span class="metric-value">{summary.get("hop_length", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Use Mel:</span>
            <span class="metric-value">{summary.get("use_mel", False)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">{summary.get("duration", 0.0):.2f}s</span>
        </div>
        {f'<div class="metric"><span class="metric-label">Segments Count:</span><span class="metric-value">{summary.get("segments_count", 0)}</span></div>' if summary.get("segments_count", 0) > 0 else ''}
    </div>

    <div class="section">
        <h2>Spectral Entropy</h2>
        <div class="metric">
            <span class="metric-label">Mean:</span>
            <span class="metric-value">{entropy_info.get("mean", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Std:</span>
            <span class="metric-value">{entropy_info.get("std", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Variance:</span>
            <span class="metric-value">{entropy_info.get("variance", 0.0):.4f}</span>
        </div>
        {f'<div class="metric"><span class="metric-label">Min:</span><span class="metric-value">{entropy_info.get("min", 0.0):.4f}</span></div>' if entropy_info.get("min") is not None else ''}
        {f'<div class="metric"><span class="metric-label">Max:</span><span class="metric-value">{entropy_info.get("max", 0.0):.4f}</span></div>' if entropy_info.get("max") is not None else ''}
    </div>

    {f'''
    <div class="section">
        <h2>Spectral Flatness</h2>
        <div class="metric">
            <span class="metric-label">Mean:</span>
            <span class="metric-value">{flatness_info.get("mean", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Std:</span>
            <span class="metric-value">{flatness_info.get("std", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Variance:</span>
            <span class="metric-value">{flatness_info.get("variance", 0.0):.4f}</span>
        </div>
        {f'<div class="metric"><span class="metric-label">Min:</span><span class="metric-value">{flatness_info.get("min", 0.0):.4f}</span></div>' if flatness_info.get("min") is not None else ''}
        {f'<div class="metric"><span class="metric-label">Max:</span><span class="metric-value">{flatness_info.get("max", 0.0):.4f}</span></div>' if flatness_info.get("max") is not None else ''}
    </div>
    ''' if flatness_info else ''}

    {f'''
    <div class="section">
        <h2>Spectral Spread</h2>
        <div class="metric">
            <span class="metric-label">Mean:</span>
            <span class="metric-value">{spread_info.get("mean", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Std:</span>
            <span class="metric-value">{spread_info.get("std", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Variance:</span>
            <span class="metric-value">{spread_info.get("variance", 0.0):.4f}</span>
        </div>
        {f'<div class="metric"><span class="metric-label">Min:</span><span class="metric-value">{spread_info.get("min", 0.0):.4f}</span></div>' if spread_info.get("min") is not None else ''}
        {f'<div class="metric"><span class="metric-label">Max:</span><span class="metric-value">{spread_info.get("max", 0.0):.4f}</span></div>' if spread_info.get("max") is not None else ''}
    </div>
    ''' if spread_info else ''}

    {f'''
    <div class="section">
        <h2>Time Series</h2>
        <div id="time-series-plot"></div>
    </div>
    ''' if entropy_series else ''}

    {f'''
    <div class="section">
        <h2>Dynamics</h2>
        <div class="metric">
            <span class="metric-label">Stability:</span>
            <span class="metric-value">{dynamics.get("spectral_entropy_stability", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Transitions Count:</span>
            <span class="metric-value">{dynamics.get("spectral_entropy_transitions_count", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Transitions Rate:</span>
            <span class="metric-value">{dynamics.get("spectral_entropy_transitions_rate", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Diversity:</span>
            <span class="metric-value">{dynamics.get("spectral_entropy_diversity", 0.0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Distribution:</span>
            <span class="metric-value">{json.dumps(dynamics.get("spectral_entropy_distribution", {}))}</span>
        </div>
    </div>
    ''' if dynamics else ''}

    <script>
        var entropySeries = {json.dumps(entropy_series)};
        var flatnessSeries = {json.dumps(flatness_series)};
        var spreadSeries = {json.dumps(spread_series)};

        // Time series plot
        {f'''
        if (entropySeries.length > 0) {{
            var traces = [];
            
            // Entropy
            traces.push({{
                x: Array.from({{length: entropySeries.length}}, (_, i) => i),
                y: entropySeries,
                mode: 'lines',
                type: 'scatter',
                name: 'Spectral Entropy',
                line: {{ width: 2, color: 'blue' }}
            }});

            // Flatness
            ''' + ('''
            if (flatnessSeries.length > 0) {{
                traces.push({{
                    x: Array.from({{length: flatnessSeries.length}}, (_, i) => i),
                    y: flatnessSeries,
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Spectral Flatness',
                    line: {{ width: 2, color: 'red' }},
                    yaxis: 'y2'
                }});
            }}
            ''' if flatness_series else '') + '''

            // Spread
            ''' + ('''
            if (spreadSeries.length > 0) {{
                traces.push({{
                    x: Array.from({{length: spreadSeries.length}}, (_, i) => i),
                    y: spreadSeries,
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Spectral Spread',
                    line: {{ width: 2, color: 'green' }},
                    yaxis: 'y3'
                }});
            }}
            ''' if spread_series else '') + '''

            var layout = {{
                title: 'Spectral Entropy Time Series',
                xaxis: {{ title: 'Frame Index' }},
                yaxis: {{ title: 'Entropy (bits)', side: 'left' }},
                {f'yaxis2: {{ title: "Flatness", overlaying: "y", side: "right" }},' if flatness_series else ''}
                {f'yaxis3: {{ title: "Spread", overlaying: "y", side: "right", anchor: "free", position: 1.0 }},' if spread_series else ''}
                height: 500,
                showlegend: true
            }};
            Plotly.newPlot('time-series-plot', traces, layout);
        }}
        ''' if entropy_series else ''}
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


__all__ = ["render_spectral_entropy_extractor", "render_spectral_entropy_extractor_html"]

