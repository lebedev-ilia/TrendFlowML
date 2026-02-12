"""
Renderer для band_energy_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta


def render_band_energy_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для band_energy_extractor."""
    render = {
        "component": "band_energy_extractor",
        "summary": {},
        "band_info": {},
        "statistics": {},
        "balance_metrics": {},
        "time_series": {},
        "dynamics": {},
    }

    features_enabled = meta.get("features_enabled", [])

    # Extract payload (similar to key_extractor render.py)
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
    
    # Fallback: if payload is empty, try to reconstruct from NPZ arrays and meta
    if not payload:
        # Extract from feature_names/feature_values
        feature_names = npz_data.get("feature_names", [])
        feature_values = npz_data.get("feature_values", [])
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        if isinstance(feature_values, np.ndarray):
            feature_values = feature_values.tolist()
        
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                payload[name] = feature_values[i]
        
        # Extract from direct NPZ arrays
        if "band_energies" in npz_data:
            band_energies_arr = npz_data["band_energies"]
            if isinstance(band_energies_arr, np.ndarray):
                payload["band_energies"] = band_energies_arr.tolist()
        if "band_energy_shares" in npz_data:
            band_shares_arr = npz_data["band_energy_shares"]
            if isinstance(band_shares_arr, np.ndarray):
                payload["band_energy_shares"] = band_shares_arr.tolist()
        
        # Extract from meta
        if "band_edges" in meta:
            payload["band_edges"] = meta["band_edges"]
        if "method" in meta:
            payload["method"] = meta["method"]

    # Get band_edges from payload or meta
    band_edges = payload.get("band_edges") or meta.get("band_edges", [])
    
    # Summary
    render["summary"] = {
        "band_edges": band_edges,
        "num_bands": len(band_edges) if band_edges else 0,
        "total_energy": payload.get("total_energy", 0.0),
        "method": payload.get("method") or meta.get("method", "unknown"),
        "sample_rate": payload.get("sample_rate", 0),
        "n_fft": payload.get("n_fft", 0),
        "hop_length": payload.get("hop_length", 0),
        "duration": payload.get("duration", 0.0),
    }

    # Band info - try payload first, then direct NPZ arrays
    band_energies = payload.get("band_energies")
    band_shares = payload.get("band_energy_shares")
    
    # Fallback to direct NPZ arrays if not in payload
    if not band_energies and "band_energies" in npz_data:
        band_energies = npz_data["band_energies"]
    if not band_shares and "band_energy_shares" in npz_data:
        band_shares = npz_data["band_energy_shares"]
    
    if isinstance(band_energies, np.ndarray):
        band_energies = band_energies.tolist()
    elif band_energies is None:
        band_energies = []
    
    if isinstance(band_shares, np.ndarray):
        band_shares = band_shares.tolist()
    elif band_shares is None:
        band_shares = []

    render["band_info"] = {
        "band_energies": band_energies,
        "band_energy_shares": band_shares,
        "dominant_band": int(np.argmax(band_shares)) if band_shares else 0,
    }

    # Statistics if enabled
    if "basic_stats" in features_enabled:
        # Try payload first, then direct NPZ arrays
        band_energy_mean = payload.get("band_energy_mean")
        band_energy_std = payload.get("band_energy_std")
        band_energy_median = payload.get("band_energy_median")
        
        if not band_energy_mean and "band_energy_mean" in npz_data:
            band_energy_mean = npz_data["band_energy_mean"]
        if not band_energy_std and "band_energy_std" in npz_data:
            band_energy_std = npz_data["band_energy_std"]
        if not band_energy_median and "band_energy_median" in npz_data:
            band_energy_median = npz_data["band_energy_median"]
        
        render["statistics"] = {
            "mean": band_energy_mean.tolist() if isinstance(band_energy_mean, np.ndarray) else (band_energy_mean or []),
            "std": band_energy_std.tolist() if isinstance(band_energy_std, np.ndarray) else (band_energy_std or []),
            "median": band_energy_median.tolist() if isinstance(band_energy_median, np.ndarray) else (band_energy_median or []),
        }

    if "extended_stats" in features_enabled:
        # Try payload first, then direct NPZ arrays
        band_energy_min = payload.get("band_energy_min")
        band_energy_max = payload.get("band_energy_max")
        band_energy_p25 = payload.get("band_energy_p25")
        band_energy_p75 = payload.get("band_energy_p75")
        
        if not band_energy_min and "band_energy_min" in npz_data:
            band_energy_min = npz_data["band_energy_min"]
        if not band_energy_max and "band_energy_max" in npz_data:
            band_energy_max = npz_data["band_energy_max"]
        if not band_energy_p25 and "band_energy_p25" in npz_data:
            band_energy_p25 = npz_data["band_energy_p25"]
        if not band_energy_p75 and "band_energy_p75" in npz_data:
            band_energy_p75 = npz_data["band_energy_p75"]
        
        if "statistics" not in render:
            render["statistics"] = {}
        render["statistics"].update({
            "min": band_energy_min.tolist() if isinstance(band_energy_min, np.ndarray) else (band_energy_min or []),
            "max": band_energy_max.tolist() if isinstance(band_energy_max, np.ndarray) else (band_energy_max or []),
            "p25": band_energy_p25.tolist() if isinstance(band_energy_p25, np.ndarray) else (band_energy_p25 or []),
            "p75": band_energy_p75.tolist() if isinstance(band_energy_p75, np.ndarray) else (band_energy_p75 or []),
        })

    # Balance metrics if enabled
    if "balance_metrics" in features_enabled:
        # Try payload first, then meta
        band_balance_score = payload.get("band_balance_score") or meta.get("band_balance_score", 0.0)
        band_dominance = payload.get("band_dominance") or meta.get("band_dominance", 0)
        band_contrast = payload.get("band_contrast") or meta.get("band_contrast", 0.0)
        
        render["balance_metrics"] = {
            "band_balance_score": float(band_balance_score) if band_balance_score is not None else 0.0,
            "band_dominance": int(band_dominance) if band_dominance is not None else 0,
            "band_contrast": float(band_contrast) if band_contrast is not None else 0.0,
        }

    # Time series if enabled
    if "time_series" in features_enabled:
        # Try payload first, then direct NPZ arrays
        band_energy_ts = payload.get("band_energy_ts")
        segment_centers = payload.get("segment_centers_sec")
        
        if not band_energy_ts and "band_energy_ts" in npz_data:
            band_energy_ts = npz_data["band_energy_ts"]
        if not segment_centers and "segment_centers_sec" in npz_data:
            segment_centers = npz_data["segment_centers_sec"]

        if isinstance(band_energy_ts, np.ndarray):
            band_energy_ts = band_energy_ts.tolist()
        elif band_energy_ts is None:
            band_energy_ts = []
            
        if isinstance(segment_centers, np.ndarray):
            segment_centers = segment_centers.tolist()
        elif segment_centers is None:
            segment_centers = []

        render["time_series"] = {
            "band_energy_ts": band_energy_ts,
            "segment_centers_sec": segment_centers,
        }

    # Dynamics if enabled
    if "dynamics" in features_enabled:
        # Try payload first, then meta
        band_energy_stability = payload.get("band_energy_stability") or meta.get("band_energy_stability", 0.0)
        band_transitions = payload.get("band_transitions") or meta.get("band_transitions", [])
        band_transitions_count = payload.get("band_transitions_count") or meta.get("band_transitions_count", 0)
        band_transitions_rate = payload.get("band_transitions_rate") or meta.get("band_transitions_rate", 0.0)
        band_distribution = payload.get("band_distribution") or meta.get("band_distribution", {})
        band_diversity = payload.get("band_diversity") or meta.get("band_diversity", 0)
        
        render["dynamics"] = {
            "band_energy_stability": float(band_energy_stability) if band_energy_stability is not None else 0.0,
            "band_transitions": band_transitions if isinstance(band_transitions, list) else [],
            "band_transitions_count": int(band_transitions_count) if band_transitions_count is not None else 0,
            "band_transitions_rate": float(band_transitions_rate) if band_transitions_rate is not None else 0.0,
            "band_distribution": band_distribution if isinstance(band_distribution, dict) else {},
            "band_diversity": int(band_diversity) if band_diversity is not None else 0,
        }

    return render


def render_band_energy_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага band_energy_extractor результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла

    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_band_energy_extractor(npz_data, meta)

    summary = render.get("summary", {})
    band_info = render.get("band_info", {})
    statistics = render.get("statistics", {})
    balance_metrics = render.get("balance_metrics", {})
    time_series = render.get("time_series", {})
    dynamics = render.get("dynamics", {})

    # Prepare data for visualization
    band_edges = summary.get("band_edges", [])
    band_energies = band_info.get("band_energies", [])
    band_shares = band_info.get("band_energy_shares", [])
    segment_centers = time_series.get("segment_centers_sec", [])
    band_energy_ts = time_series.get("band_energy_ts", [])

    # Band names
    band_names = [f"Band {i+1}" for i in range(len(band_edges))]
    if len(band_edges) == 3:
        band_names = ["Low", "Mid", "High"]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Band Energy Extractor Debug</title>
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
    <h1>Band Energy Extractor Debug</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Number of Bands:</span>
            <span class="metric-value">{summary.get("num_bands", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Energy:</span>
            <span class="metric-value">{summary.get("total_energy", 0.0):.2e}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Method:</span>
            <span class="metric-value">{summary.get("method", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">{summary.get("duration", 0.0):.2f}s</span>
        </div>
    </div>

    <div class="section">
        <h2>Band Energies</h2>
        <div id="band-energies-plot"></div>
    </div>

    <div class="section">
        <h2>Band Energy Shares</h2>
        <div id="band-shares-plot"></div>
    </div>

    {f'''
    <div class="section">
        <h2>Balance Metrics</h2>
        <div class="metric">
            <span class="metric-label">Balance Score:</span>
            <span class="metric-value">{balance_metrics.get("band_balance_score", 0.0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Dominant Band:</span>
            <span class="metric-value">{band_names[balance_metrics.get("band_dominance", 0)] if balance_metrics.get("band_dominance") is not None else "N/A"}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Contrast:</span>
            <span class="metric-value">{balance_metrics.get("band_contrast", 0.0):.3f}</span>
        </div>
    </div>
    ''' if balance_metrics else ''}

    {f'''
    <div class="section">
        <h2>Statistics</h2>
        <div id="statistics-plot"></div>
    </div>
    ''' if statistics else ''}

    {f'''
    <div class="section">
        <h2>Band Energy Over Time</h2>
        <div id="band-timeline-plot"></div>
    </div>
    ''' if segment_centers and band_energy_ts else ''}

    {f'''
    <div class="section">
        <h2>Dynamics</h2>
        <div class="metric">
            <span class="metric-label">Stability:</span>
            <span class="metric-value">{dynamics.get("band_energy_stability", 0.0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Transitions Count:</span>
            <span class="metric-value">{dynamics.get("band_transitions_count", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Transitions Rate:</span>
            <span class="metric-value">{dynamics.get("band_transitions_rate", 0.0):.4f} transitions/sec</span>
        </div>
        <div class="metric">
            <span class="metric-label">Band Diversity:</span>
            <span class="metric-value">{dynamics.get("band_diversity", 0)}</span>
        </div>
    </div>
    ''' if dynamics else ''}

    <script>
        var bandNames = {json.dumps(band_names)};
        var bandEnergies = {json.dumps(band_energies)};
        var bandShares = {json.dumps(band_shares)};
        var segmentCenters = {json.dumps(segment_centers)};
        var bandEnergyTs = {json.dumps(band_energy_ts)};
        var statistics = {json.dumps(statistics)};

        // Band energies plot
        var traceEnergies = {{
            x: bandNames,
            y: bandEnergies,
            type: 'bar',
            marker: {{ color: 'blue' }}
        }};
        var layoutEnergies = {{
            title: 'Band Energies',
            xaxis: {{ title: 'Band' }},
            yaxis: {{ title: 'Energy', type: 'log' }},
            height: 400
        }};
        Plotly.newPlot('band-energies-plot', [traceEnergies], layoutEnergies);

        // Band shares plot (pie chart)
        var traceShares = {{
            labels: bandNames,
            values: bandShares,
            type: 'pie',
            textinfo: 'label+percent',
            textposition: 'outside',
        }};
        var layoutShares = {{
            title: 'Band Energy Shares',
            height: 400
        }};
        Plotly.newPlot('band-shares-plot', [traceShares], layoutShares);

        // Statistics plot
        {f'''
        if (statistics.mean && statistics.mean.length > 0) {{
            var traceMean = {{
                x: bandNames,
                y: statistics.mean,
                type: 'bar',
                name: 'Mean',
                marker: {{ color: 'blue' }}
            }};
            var traceStd = {{
                x: bandNames,
                y: statistics.std,
                type: 'bar',
                name: 'Std',
                marker: {{ color: 'red' }}
            }};
            var layoutStats = {{
                title: 'Band Energy Statistics',
                xaxis: {{ title: 'Band' }},
                yaxis: {{ title: 'Value' }},
                barmode: 'group',
                height: 400
            }};
            Plotly.newPlot('statistics-plot', [traceMean, traceStd], layoutStats);
        }}
        ''' if statistics else ''}

        // Band energy timeline plot
        {f'''
        if (segmentCenters.length > 0 && bandEnergyTs.length > 0) {{
            var traces = [];
            for (var i = 0; i < bandEnergyTs.length; i++) {{
                traces.push({{
                    x: segmentCenters,
                    y: bandEnergyTs[i],
                    mode: 'lines+markers',
                    type: 'scatter',
                    name: bandNames[i],
                    line: {{ width: 2 }}
                }});
            }}
            var layoutTimeline = {{
                title: 'Band Energy Over Time',
                xaxis: {{ title: 'Time (seconds)' }},
                yaxis: {{ title: 'Energy', type: 'log' }},
                height: 500,
                showlegend: true
            }};
            Plotly.newPlot('band-timeline-plot', traces, layoutTimeline);
        }}
        ''' if segment_centers and band_energy_ts else ''}
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


__all__ = ["render_band_energy_extractor", "render_band_energy_extractor_html"]

