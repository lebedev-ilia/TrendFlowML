"""
Renderer для onset_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_onset_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для onset_extractor."""
    render = {
        "component": "onset_extractor",
        "summary": {},
        "basic_features": {},
        "interval_stats": {},
        "rhythmic_metrics": {},
        "timeline": {},
    }
    
    # Helper function to safely get value from NPZ data
    def safe_get(key: str, default: Any = None) -> Any:
        value = npz_data.get(key, default)
        if value is None:
            return default
        # Handle numpy arrays (if not converted yet by load_npz)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            if value.size == 1:
                val = value.item()
                # Convert NaN/Inf to None for JSON compatibility
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        return None
                return val
            # For multi-element arrays, convert to list
            return value.tolist()
        # Handle lists (converted from numpy arrays by load_npz)
        if isinstance(value, list):
            if len(value) == 0:
                return default
            # For scalar values stored as single-element lists
            if len(value) == 1 and not isinstance(value[0], (list, np.ndarray)):
                val = value[0]
                # Convert NaN/Inf to None for JSON compatibility
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        return None
                return val
            return value
        # Convert NaN/Inf to None for JSON compatibility
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return None
        return value
    
    # Extract feature_names and feature_values to build features dict
    feature_names = safe_get("feature_names", [])
    feature_values = safe_get("feature_values", [])
    
    # Build features dict from feature_names/feature_values
    features = {}
    if isinstance(feature_names, list) and isinstance(feature_values, list):
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                value = feature_values[i]
                # Convert NaN/Inf to None for JSON compatibility
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        features[name] = None
                    else:
                        features[name] = value
                else:
                    features[name] = value
    
    # Helper to get value from features dict or directly from NPZ
    def get_feature(key: str, default: Any = None) -> Any:
        # First try features dict (from feature_names/feature_values)
        if key in features:
            return features[key]
        # Then try direct NPZ key
        return safe_get(key, default)
    
    # Extract features_enabled from meta
    features_enabled = meta.get("features_enabled", [])
    
    # Summary (always available) - read from NPZ directly
    render["summary"] = {
        "backend": get_feature("backend", meta.get("backend", "unknown")),
        "sample_rate": get_feature("sample_rate"),
        "hop_length": get_feature("hop_length"),
        "segments_count": get_feature("segments_count"),
    }
    
    # Basic features
    if "basic_features" in features_enabled:
        render["basic_features"] = {
            "onset_count": get_feature("onset_count"),
            "onset_density_per_sec": get_feature("onset_density_per_sec"),
            "insufficient_onsets": get_feature("insufficient_onsets"),
        }
    
    # Interval stats
    if "interval_stats" in features_enabled:
        render["interval_stats"] = {
            "avg_interval_sec": get_feature("avg_interval_sec"),
            "interval_std": get_feature("interval_std"),
            "interval_min": get_feature("interval_min"),
            "interval_max": get_feature("interval_max"),
            "interval_median": get_feature("interval_median"),
        }
    
    # Rhythmic metrics
    if "rhythmic_metrics" in features_enabled:
        render["rhythmic_metrics"] = {
            "onset_regularity_score": get_feature("onset_regularity_score"),
            "onset_clustering_score": get_feature("onset_clustering_score"),
            "onset_tempo_estimate": get_feature("onset_tempo_estimate"),
            "onset_syncopation_score": get_feature("onset_syncopation_score"),
            "onset_strength_mean": get_feature("onset_strength_mean"),
            "onset_strength_std": get_feature("onset_strength_std"),
            "onset_density_variance": get_feature("onset_density_variance"),
            "onset_tempo_consistency": get_feature("onset_tempo_consistency"),
        }
    
    # Timeline (time series) - onset_times is stored as separate key in NPZ
    onset_times = safe_get("onset_times", [])
    if not isinstance(onset_times, list):
        onset_times = []
    
    render["timeline"] = {
        "onset_times": onset_times,
        "duration": get_feature("duration"),
    }
    
    return render


def render_onset_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага onset_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла
        
    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_onset_extractor(npz_data, meta)
    
    onset_times = render.get("timeline", {}).get("onset_times", [])
    duration = render.get("timeline", {}).get("duration", 0.0)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Onset Extractor Debug</title>
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
    <h1>Onset Extractor Debug</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Backend:</span>
            <span class="metric-value">{render.get("summary", {}).get("backend", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Sample Rate:</span>
            <span class="metric-value">{render.get("summary", {}).get("sample_rate", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Hop Length:</span>
            <span class="metric-value">{render.get("summary", {}).get("hop_length", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Segments Count:</span>
            <span class="metric-value">{render.get("summary", {}).get("segments_count", "N/A")}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Basic Features</h2>
        <div class="metric">
            <span class="metric-label">Onset Count:</span>
            <span class="metric-value">{render.get("basic_features", {}).get("onset_count", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Onset Density (per sec):</span>
            <span class="metric-value">{render.get("basic_features", {}).get("onset_density_per_sec", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Insufficient Onsets:</span>
            <span class="metric-value">{render.get("basic_features", {}).get("insufficient_onsets", "N/A")}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Interval Statistics</h2>
        <div class="metric">
            <span class="metric-label">Avg Interval (sec):</span>
            <span class="metric-value">{render.get("interval_stats", {}).get("avg_interval_sec", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Interval Std:</span>
            <span class="metric-value">{render.get("interval_stats", {}).get("interval_std", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Interval Min:</span>
            <span class="metric-value">{render.get("interval_stats", {}).get("interval_min", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Interval Max:</span>
            <span class="metric-value">{render.get("interval_stats", {}).get("interval_max", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Interval Median:</span>
            <span class="metric-value">{render.get("interval_stats", {}).get("interval_median", "N/A")}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Rhythmic Metrics</h2>
        <div class="metric">
            <span class="metric-label">Regularity Score:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_regularity_score", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Clustering Score:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_clustering_score", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tempo Estimate (BPM):</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_tempo_estimate", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Syncopation Score:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_syncopation_score", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Strength Mean:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_strength_mean", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Strength Std:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_strength_std", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Density Variance:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_density_variance", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tempo Consistency:</span>
            <span class="metric-value">{render.get("rhythmic_metrics", {}).get("onset_tempo_consistency", "N/A")}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Onset Timeline</h2>
        <div id="timeline-plot"></div>
    </div>
    
    <script>
        var onsetTimes = {json.dumps(onset_times)};
        var duration = {duration};
        
        var trace = {{
            x: onsetTimes,
            y: Array(onsetTimes.length).fill(1),
            mode: 'markers',
            type: 'scatter',
            marker: {{ size: 10, color: 'red' }},
            name: 'Onsets'
        }};
        
        var layout = {{
            title: 'Onset Timeline',
            xaxis: {{ title: 'Time (seconds)', range: [0, duration] }},
            yaxis: {{ title: 'Onset Events', range: [0, 2] }},
            height: 400
        }};
        
        Plotly.newPlot('timeline-plot', [trace], layout);
    </script>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path

__all__ = ["render_onset_extractor", "render_onset_extractor_html"]
