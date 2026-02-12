"""
Renderer для tempo_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_tempo_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для tempo_extractor."""
    render = {
        "component": "tempo_extractor",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "warnings": [],
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
    
    # Summary - извлекаем из features или напрямую из npz_data
    # confidence может быть в features как "confidence" или "tempo_confidence"
    confidence = features.get("confidence") or features.get("tempo_confidence", 0.0)
    
    render["summary"] = {
        "tempo_bpm": features.get("tempo_bpm", 0.0),
        "tempo_bpm_mean": features.get("tempo_bpm_mean", 0.0),
        "tempo_bpm_median": features.get("tempo_bpm_median", 0.0),
        "tempo_bpm_std": features.get("tempo_bpm_std", 0.0),
        "tempo_confidence": float(confidence),
        "segments_count": int(features.get("segments_count", 0)),
    }
    
    # Warnings
    warnings = npz_data.get("warnings")
    if warnings is not None:
        if isinstance(warnings, np.ndarray):
            warnings = warnings.tolist()
        render["warnings"] = warnings if isinstance(warnings, list) else []
    
    # Timeline data (windowed BPM if available)
    windowed_times_sec = npz_data.get("windowed_times_sec")
    windowed_bpm = npz_data.get("windowed_bpm")
    
    if windowed_times_sec is not None and windowed_bpm is not None:
        if isinstance(windowed_times_sec, np.ndarray):
            windowed_times_sec = windowed_times_sec.tolist()
        if isinstance(windowed_bpm, np.ndarray):
            windowed_bpm = windowed_bpm.tolist()
        
        timeline = []
        for i, (time_sec, bpm) in enumerate(zip(windowed_times_sec, windowed_bpm)):
            timeline.append({
                "time_sec": float(time_sec),
                "bpm": float(bpm),
                "window_index": i,
            })
        render["timeline"] = timeline
        
        # Distribution of BPM
        if timeline:
            bpms = [t["bpm"] for t in timeline]
            render["distributions"]["bpm"] = {
                "min": float(np.min(bpms)) if bpms else 0.0,
                "max": float(np.max(bpms)) if bpms else 0.0,
                "mean": float(np.mean(bpms)) if bpms else 0.0,
                "std": float(np.std(bpms)) if bpms else 0.0,
                "median": float(np.median(bpms)) if bpms else 0.0,
            }
    
    # Distribution of tempo_estimates
    tempo_estimates = npz_data.get("tempo_estimates")
    if tempo_estimates is not None:
        if isinstance(tempo_estimates, np.ndarray):
            tempo_estimates = tempo_estimates.tolist()
        if isinstance(tempo_estimates, list) and tempo_estimates:
            render["distributions"]["tempo_estimates"] = {
                "min": float(np.min(tempo_estimates)),
                "max": float(np.max(tempo_estimates)),
                "mean": float(np.mean(tempo_estimates)),
                "std": float(np.std(tempo_estimates)),
                "median": float(np.median(tempo_estimates)),
            }
    
    return render


def render_tempo_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага tempo_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    ap_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(ap_root / "src") not in sys.path:
        sys.path.insert(0, str(ap_root / "src"))
    from core.renderer import load_npz, extract_meta  # type: ignore
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_tempo_extractor(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    warnings = render.get("warnings", [])
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        bpms = [t.get("bpm", 0.0) for t in timeline]
        timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times])},
            datasets: [{{
                label: 'BPM',
                data: {json.dumps(bpms)},
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1,
                pointRadius: 3
            }}]
        }};
        """
    
    # Prepare tempo estimates distribution data
    tempo_estimates_dist = distributions.get("tempo_estimates", {})
    bpm_dist = distributions.get("bpm", {})
    
    # Warnings display
    warnings_html = ""
    if warnings:
        warnings_list = ", ".join(warnings) if isinstance(warnings, list) else str(warnings)
        warnings_html = f"""
        <div class="warnings">
            <h2>Warnings</h2>
            <div class="warning-badge">{warnings_list}</div>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tempo Extractor Debug Render</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; font-weight: bold; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin: 20px 0; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .warnings {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #ffc107; }}
        .warning-badge {{ background-color: #ffc107; color: #856404; padding: 8px 12px; border-radius: 4px; display: inline-block; font-weight: bold; }}
        .meta-info {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 20px 0; font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Tempo Extractor Debug Render</h1>
        
        <div class="meta-info">
            <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
            <p><strong>Producer:</strong> {meta.get('producer_version', 'unknown')}</p>
            <p><strong>Schema Version:</strong> {meta.get('schema_version', 'unknown')}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Tempo BPM</strong>
                    <span class="metric-value">{summary.get('tempo_bpm', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>BPM Mean</strong>
                    <span class="metric-value">{summary.get('tempo_bpm_mean', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>BPM Median</strong>
                    <span class="metric-value">{summary.get('tempo_bpm_median', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>BPM Std</strong>
                    <span class="metric-value">{summary.get('tempo_bpm_std', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>Confidence</strong>
                    <span class="metric-value">{summary.get('tempo_confidence', 0.0):.3f}</span>
                </div>
            </div>
        </div>
        
        {warnings_html}
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: BPM Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
        {f'''
        <div class="distributions">
            <h2>Windowed BPM Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{bpm_dist.get('min', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{bpm_dist.get('max', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{bpm_dist.get('mean', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{bpm_dist.get('std', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{bpm_dist.get('median', 0.0):.2f} BPM</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if bpm_dist else ''}
        
        {f'''
        <div class="distributions">
            <h2>Tempo Estimates Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{tempo_estimates_dist.get('min', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{tempo_estimates_dist.get('max', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{tempo_estimates_dist.get('mean', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{tempo_estimates_dist.get('std', 0.0):.2f} BPM</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{tempo_estimates_dist.get('median', 0.0):.2f} BPM</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if tempo_estimates_dist else ''}
    </div>
    
    {f'''
    <script>
        {timeline_js}
        const ctx = document.getElementById('timelineChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: timelineData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'BPM (Beats Per Minute)'
                        }},
                        suggestedMin: 40,
                        suggestedMax: 220
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time (seconds)'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return 'BPM: ' + context.parsed.y.toFixed(2);
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
    ''' if timeline else ''}
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Saved Tempo HTML render to {output_path}")
    return output_path


__all__ = ["render_tempo_extractor", "render_tempo_extractor_html"]

