"""
Renderer для voice_quality_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta


def render_voice_quality_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для voice_quality_extractor."""
    render = {
        "component": "voice_quality_extractor",
        "summary": {},
        "jitter_metrics": {},
        "shimmer_metrics": {},
        "hnr_metrics": {},
        "f0_stats": {},
        "quality_scores": {},
        "timeline": [],
        "distributions": {},
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

    # Extract payload
    payload = npz_data.get("payload")
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item() if payload.size == 1 else {}
    if not isinstance(payload, dict):
        payload = {}

    # Summary
    render["summary"] = {
        "sample_rate": int(payload.get("sample_rate", 22050)),
        "duration": float(payload.get("duration", 0.0)),
        "f0_method": str(payload.get("f0_method", "unknown")),
        "f0_fmin": float(payload.get("f0_fmin", 50.0)),
        "f0_fmax": float(payload.get("f0_fmax", 500.0)),
        "segments_count": int(payload.get("segments_count", 0)) if "segments_count" in payload else None,
    }

    # Jitter metrics (feature-gated)
    if "vq_jitter" in payload:
        render["jitter_metrics"] = {
            "jitter": float(payload.get("vq_jitter", 0.0)),
        }
        if "vq_jitter_mean" in payload:
            render["jitter_metrics"]["jitter_mean"] = float(payload.get("vq_jitter_mean", 0.0))
        if "vq_jitter_std" in payload:
            render["jitter_metrics"]["jitter_std"] = float(payload.get("vq_jitter_std", 0.0))
        if "vq_jitter_min" in payload:
            render["jitter_metrics"]["jitter_min"] = float(payload.get("vq_jitter_min", 0.0))
        if "vq_jitter_max" in payload:
            render["jitter_metrics"]["jitter_max"] = float(payload.get("vq_jitter_max", 0.0))

    # Shimmer metrics (feature-gated)
    if "vq_shimmer" in payload:
        render["shimmer_metrics"] = {
            "shimmer": float(payload.get("vq_shimmer", 0.0)),
        }
        if "vq_shimmer_mean" in payload:
            render["shimmer_metrics"]["shimmer_mean"] = float(payload.get("vq_shimmer_mean", 0.0))
        if "vq_shimmer_std" in payload:
            render["shimmer_metrics"]["shimmer_std"] = float(payload.get("vq_shimmer_std", 0.0))
        if "vq_shimmer_min" in payload:
            render["shimmer_metrics"]["shimmer_min"] = float(payload.get("vq_shimmer_min", 0.0))
        if "vq_shimmer_max" in payload:
            render["shimmer_metrics"]["shimmer_max"] = float(payload.get("vq_shimmer_max", 0.0))

    # HNR metrics (feature-gated)
    if "vq_hnr_like_db" in payload:
        render["hnr_metrics"] = {
            "hnr_db": float(payload.get("vq_hnr_like_db", 0.0)),
        }
        if "vq_hnr_mean" in payload:
            render["hnr_metrics"]["hnr_mean"] = float(payload.get("vq_hnr_mean", 0.0))
        if "vq_hnr_std" in payload:
            render["hnr_metrics"]["hnr_std"] = float(payload.get("vq_hnr_std", 0.0))
        if "vq_hnr_min" in payload:
            render["hnr_metrics"]["hnr_min"] = float(payload.get("vq_hnr_min", 0.0))
        if "vq_hnr_max" in payload:
            render["hnr_metrics"]["hnr_max"] = float(payload.get("vq_hnr_max", 0.0))

    # F0 stats (feature-gated)
    if "vq_f0_mean" in payload:
        render["f0_stats"] = {
            "f0_mean": float(payload.get("vq_f0_mean", 0.0)),
            "f0_std": float(payload.get("vq_f0_std", 0.0)),
            "f0_min": float(payload.get("vq_f0_min", 0.0)),
            "f0_max": float(payload.get("vq_f0_max", 0.0)),
        }
        if "vq_f0_median" in payload:
            render["f0_stats"]["f0_median"] = float(payload.get("vq_f0_median", 0.0))
        if "vq_f0_stability" in payload:
            render["f0_stats"]["f0_stability"] = float(payload.get("vq_f0_stability", 0.0))
        if "vq_voice_presence_ratio" in payload:
            render["f0_stats"]["voice_presence_ratio"] = float(payload.get("vq_voice_presence_ratio", 0.0))

    # Quality scores
    if "vq_voice_quality_score" in payload:
        render["quality_scores"] = {
            "voice_quality_score": float(payload.get("vq_voice_quality_score", 0.0)),
        }
        if "vq_breathiness_score" in payload:
            render["quality_scores"]["breathiness_score"] = float(payload.get("vq_breathiness_score", 0.0))

    # Timeline (f0 if available)
    f0 = payload.get("f0")
    f0_npy = payload.get("f0_npy")
    segment_centers_sec = payload.get("segment_centers_sec")

    if f0 is not None:
        if isinstance(f0, np.ndarray):
            f0 = f0.tolist()
        render["timeline"] = [
            {"index": i, "f0": float(f0_val)} for i, f0_val in enumerate(f0)
        ]
    elif f0_npy is not None:
        # F0 saved to .npy file
        render["timeline"] = []  # Would need to load from file for full timeline
    elif segment_centers_sec is not None:
        # Segment-based timeline
        if isinstance(segment_centers_sec, np.ndarray):
            segment_centers_sec = segment_centers_sec.tolist()
        render["timeline"] = [
            {"segment_index": i, "center_sec": float(center_sec)} for i, center_sec in enumerate(segment_centers_sec)
        ]

    # Distributions
    if render["timeline"] and "f0" in render["timeline"][0]:
        f0_list = [t["f0"] for t in render["timeline"] if "f0" in t]
        if f0_list:
            render["distributions"]["f0"] = {
                "min": float(np.min(f0_list)),
                "max": float(np.max(f0_list)),
                "mean": float(np.mean(f0_list)),
                "std": float(np.std(f0_list)),
                "median": float(np.median(f0_list)),
            }

    return render


def render_voice_quality_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага voice_quality_extractor результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла

    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_voice_quality_extractor(npz_data, meta)

    summary = render.get("summary", {})
    jitter_metrics = render.get("jitter_metrics", {})
    shimmer_metrics = render.get("shimmer_metrics", {})
    hnr_metrics = render.get("hnr_metrics", {})
    f0_stats = render.get("f0_stats", {})
    quality_scores = render.get("quality_scores", {})
    timeline = render.get("timeline", [])

    # Pre-compute HTML sections to avoid nested f-strings
    segments_count_html = f"<div class='metric'><span class='metric-label'>Segments Count:</span><span class='metric-value'>{summary.get('segments_count', 'N/A')}</span></div>" if summary.get('segments_count') is not None else ""
    
    jitter_section = ""
    if jitter_metrics:
        jitter_mean_html = f"<div class='metric'><span class='metric-label'>Jitter Mean:</span><span class='metric-value'>{jitter_metrics.get('jitter_mean', 'N/A')}</span></div>" if 'jitter_mean' in jitter_metrics else ""
        jitter_std_html = f"<div class='metric'><span class='metric-label'>Jitter Std:</span><span class='metric-value'>{jitter_metrics.get('jitter_std', 'N/A')}</span></div>" if 'jitter_std' in jitter_metrics else ""
        jitter_min_html = f"<div class='metric'><span class='metric-label'>Jitter Min:</span><span class='metric-value'>{jitter_metrics.get('jitter_min', 'N/A')}</span></div>" if 'jitter_min' in jitter_metrics else ""
        jitter_max_html = f"<div class='metric'><span class='metric-label'>Jitter Max:</span><span class='metric-value'>{jitter_metrics.get('jitter_max', 'N/A')}</span></div>" if 'jitter_max' in jitter_metrics else ""
        jitter_section = f"""
    <div class="section">
        <h2>Jitter Metrics</h2>
        <div class="metric">
            <span class="metric-label">Jitter:</span>
            <span class="metric-value">{jitter_metrics.get('jitter', 'N/A')}</span>
        </div>
        {jitter_mean_html}
        {jitter_std_html}
        {jitter_min_html}
        {jitter_max_html}
    </div>
    """
    
    shimmer_section = ""
    if shimmer_metrics:
        shimmer_mean_html = f"<div class='metric'><span class='metric-label'>Shimmer Mean:</span><span class='metric-value'>{shimmer_metrics.get('shimmer_mean', 'N/A')}</span></div>" if 'shimmer_mean' in shimmer_metrics else ""
        shimmer_std_html = f"<div class='metric'><span class='metric-label'>Shimmer Std:</span><span class='metric-value'>{shimmer_metrics.get('shimmer_std', 'N/A')}</span></div>" if 'shimmer_std' in shimmer_metrics else ""
        shimmer_min_html = f"<div class='metric'><span class='metric-label'>Shimmer Min:</span><span class='metric-value'>{shimmer_metrics.get('shimmer_min', 'N/A')}</span></div>" if 'shimmer_min' in shimmer_metrics else ""
        shimmer_max_html = f"<div class='metric'><span class='metric-label'>Shimmer Max:</span><span class='metric-value'>{shimmer_metrics.get('shimmer_max', 'N/A')}</span></div>" if 'shimmer_max' in shimmer_metrics else ""
        shimmer_section = f"""
    <div class="section">
        <h2>Shimmer Metrics</h2>
        <div class="metric">
            <span class="metric-label">Shimmer:</span>
            <span class="metric-value">{shimmer_metrics.get('shimmer', 'N/A')}</span>
        </div>
        {shimmer_mean_html}
        {shimmer_std_html}
        {shimmer_min_html}
        {shimmer_max_html}
    </div>
    """
    
    hnr_section = ""
    if hnr_metrics:
        hnr_mean_html = f"<div class='metric'><span class='metric-label'>HNR Mean:</span><span class='metric-value'>{hnr_metrics.get('hnr_mean', 'N/A')}</span></div>" if 'hnr_mean' in hnr_metrics else ""
        hnr_std_html = f"<div class='metric'><span class='metric-label'>HNR Std:</span><span class='metric-value'>{hnr_metrics.get('hnr_std', 'N/A')}</span></div>" if 'hnr_std' in hnr_metrics else ""
        hnr_min_html = f"<div class='metric'><span class='metric-label'>HNR Min:</span><span class='metric-value'>{hnr_metrics.get('hnr_min', 'N/A')}</span></div>" if 'hnr_min' in hnr_metrics else ""
        hnr_max_html = f"<div class='metric'><span class='metric-label'>HNR Max:</span><span class='metric-value'>{hnr_metrics.get('hnr_max', 'N/A')}</span></div>" if 'hnr_max' in hnr_metrics else ""
        hnr_section = f"""
    <div class="section">
        <h2>HNR Metrics</h2>
        <div class="metric">
            <span class="metric-label">HNR (dB):</span>
            <span class="metric-value">{hnr_metrics.get('hnr_db', 'N/A')}</span>
        </div>
        {hnr_mean_html}
        {hnr_std_html}
        {hnr_min_html}
        {hnr_max_html}
    </div>
    """
    
    f0_section = ""
    if f0_stats:
        f0_median_html = f"<div class='metric'><span class='metric-label'>F0 Median:</span><span class='metric-value'>{f0_stats.get('f0_median', 'N/A')} Hz</span></div>" if 'f0_median' in f0_stats else ""
        f0_stability_html = f"<div class='metric'><span class='metric-label'>F0 Stability:</span><span class='metric-value'>{f0_stats.get('f0_stability', 'N/A')}</span></div>" if 'f0_stability' in f0_stats else ""
        f0_presence_html = f"<div class='metric'><span class='metric-label'>Voice Presence Ratio:</span><span class='metric-value'>{f0_stats.get('voice_presence_ratio', 'N/A')}</span></div>" if 'voice_presence_ratio' in f0_stats else ""
        f0_section = f"""
    <div class="section">
        <h2>F0 Statistics</h2>
        <div class="metric">
            <span class="metric-label">F0 Mean:</span>
            <span class="metric-value">{f0_stats.get('f0_mean', 'N/A')} Hz</span>
        </div>
        <div class="metric">
            <span class="metric-label">F0 Std:</span>
            <span class="metric-value">{f0_stats.get('f0_std', 'N/A')} Hz</span>
        </div>
        <div class="metric">
            <span class="metric-label">F0 Range:</span>
            <span class="metric-value">{f0_stats.get('f0_min', 'N/A')} - {f0_stats.get('f0_max', 'N/A')} Hz</span>
        </div>
        {f0_median_html}
        {f0_stability_html}
        {f0_presence_html}
    </div>
    """
    
    quality_section = ""
    if quality_scores:
        breathiness_html = f"<div class='metric'><span class='metric-label'>Breathiness Score:</span><span class='metric-value'>{quality_scores.get('breathiness_score', 'N/A')}</span></div>" if 'breathiness_score' in quality_scores else ""
        quality_section = f"""
    <div class="section">
        <h2>Quality Scores</h2>
        <div class="metric">
            <span class="metric-label">Voice Quality Score:</span>
            <span class="metric-value">{quality_scores.get('voice_quality_score', 'N/A')}</span>
        </div>
        {breathiness_html}
    </div>
    """

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Voice Quality Extractor Debug</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ color: #333; }}
        .timeline {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Voice Quality Extractor Debug</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Sample Rate:</span>
            <span class="metric-value">{summary.get('sample_rate', 'N/A')} Hz</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">{summary.get('duration', 0.0):.2f} sec</span>
        </div>
        <div class="metric">
            <span class="metric-label">F0 Method:</span>
            <span class="metric-value">{summary.get('f0_method', 'N/A')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">F0 Range:</span>
            <span class="metric-value">{summary.get('f0_fmin', 0.0):.0f} - {summary.get('f0_fmax', 0.0):.0f} Hz</span>
        </div>
        {segments_count_html}
    </div>
    {jitter_section}
    {shimmer_section}
    {hnr_section}
    {f0_section}
    {quality_section}

    <div class="section">
        <h2>F0 Timeline</h2>
        <div id="f0-plot"></div>
    </div>

    <script>
        var timeline = {json.dumps(timeline)};
        var duration = {summary.get('duration', 0.0)};

        if (timeline.length > 0 && timeline[0].f0 !== undefined) {{
            var indices = timeline.map(function(t) {{ return t.index; }});
            var f0Values = timeline.map(function(t) {{ return t.f0; }});
            var trace = {{
                x: indices,
                y: f0Values,
                mode: 'lines+markers',
                type: 'scatter',
                name: 'F0',
                line: {{ color: 'blue' }},
                marker: {{ size: 4 }}
            }};

            var layout = {{
                title: 'F0 Timeline',
                xaxis: {{ title: 'Frame Index' }},
                yaxis: {{ title: 'F0 (Hz)', range: [0, Math.max(...f0Values) * 1.1] }},
                height: 400
            }};

            Plotly.newPlot('f0-plot', [trace], layout);
        }} else {{
            document.getElementById('f0-plot').innerHTML = '<p>F0 timeline not available (feature not enabled or saved to .npy file)</p>';
        }}
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


__all__ = ["render_voice_quality_extractor", "render_voice_quality_extractor_html"]

