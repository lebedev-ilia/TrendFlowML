"""
Renderer для frames_composition: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _svg_line_chart(
    *,
    times_s: np.ndarray,
    values: np.ndarray,
    title: str,
    stroke: str = "#2563eb",
    width: int = 960,
    height: int = 220,
    pad: int = 18,
) -> str:
    if times_s.size == 0 or values.size == 0:
        return ""

    m = np.isfinite(times_s) & np.isfinite(values)
    if not np.any(m):
        return ""

    x = times_s[m].astype(np.float64, copy=False)
    y = values[m].astype(np.float64, copy=False)
    if x.size < 2:
        return ""

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin:
        xmax = xmin + 1e-6
    if ymax <= ymin:
        ymax = ymin + 1e-6

    def sx(v: float) -> float:
        return pad + (v - xmin) / (xmax - xmin) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - ymin) / (ymax - ymin) * (height - 2 * pad)

    pts = " ".join(f"{sx(float(xx)):.2f},{sy(float(yy)):.2f}" for xx, yy in zip(x, y))
    title_esc = _esc(title)
    return f"""
<div class="chart">
  <div class="chart-title">{title_esc}</div>
  <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{title_esc}">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#e5e7eb"/>
    <polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>
    <text x="{pad}" y="{pad}" font-size="12" fill="#6b7280">{_esc(f'[{ymin:.3f} .. {ymax:.3f}]')}</text>
  </svg>
</div>
""".strip()


def render_frames_composition(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для frames_composition."""
    render = {
        "component": "frames_composition",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "key_facts": [],
        "top_examples": {},
        "anti_top_examples": {},
        "config_highlights": {},
    }
    
    # Extract data
    frame_indices = npz_data.get("frame_indices")
    times_s = npz_data.get("times_s")
    feature_names = npz_data.get("feature_names")
    feature_values = npz_data.get("feature_values")
    frame_feature_names = npz_data.get("frame_feature_names")
    frame_feature_values = npz_data.get("frame_feature_values")
    
    # Convert to numpy arrays if needed
    if frame_indices is not None:
        if isinstance(frame_indices, list):
            frame_indices = np.array(frame_indices, dtype=np.int32)
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = np.asarray(frame_indices, dtype=np.int32)
        else:
            frame_indices = None
    
    if times_s is not None:
        if isinstance(times_s, list):
            times_s = np.array(times_s, dtype=np.float32)
        elif isinstance(times_s, np.ndarray):
            times_s = np.asarray(times_s, dtype=np.float32)
        else:
            times_s = None
    
    if frame_feature_values is not None:
        if isinstance(frame_feature_values, list):
            frame_feature_values = np.array(frame_feature_values, dtype=np.float32)
        elif isinstance(frame_feature_values, np.ndarray):
            frame_feature_values = np.asarray(frame_feature_values, dtype=np.float32)
        else:
            frame_feature_values = None
    
    # Extract feature names
    if feature_names is not None:
        if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
            feature_names = [str(f) for f in feature_names.flatten()]
        elif isinstance(feature_names, list):
            feature_names = [str(f) for f in feature_names]
        else:
            feature_names = []
    else:
        feature_names = []
    
    if frame_feature_names is not None:
        if isinstance(frame_feature_names, np.ndarray) and frame_feature_names.dtype == object:
            frame_feature_names = [str(f) for f in frame_feature_names.flatten()]
        elif isinstance(frame_feature_names, list):
            frame_feature_names = [str(f) for f in frame_feature_names]
        else:
            frame_feature_names = []
    else:
        frame_feature_names = []
    
    # Summary statistics from video-level features
    if feature_names and feature_values is not None:
        if isinstance(feature_values, list):
            feature_values = np.array(feature_values, dtype=np.float32)
        elif isinstance(feature_values, np.ndarray):
            feature_values = np.asarray(feature_values, dtype=np.float32)
        
        # Build summary from video-level features
        summary = {
            "frames_count": int(len(frame_indices)) if frame_indices is not None else 0,
        }
        
        # Extract key metrics from feature_names/feature_values
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                val = float(feature_values[i]) if np.isfinite(feature_values[i]) else None
                if val is not None:
                    # Store important metrics
                    if "has_faces" in name:
                        summary["has_faces"] = bool(val > 0.5)
                    elif "frames_n" in name:
                        summary["frames_n"] = int(val)
                    elif "style_dominant_id" in name:
                        summary["style_dominant_id"] = int(val)
                    elif "style_prob__" in name:
                        if "style_probabilities" not in summary:
                            summary["style_probabilities"] = {}
                        style_name = name.replace("style_prob__", "").replace("__mean", "")
                        summary["style_probabilities"][style_name] = val
        
        render["summary"] = summary
    
    # Define key features for timeline and distributions
    key_features = {
        "saliency_center_offset": None,
        "symmetry_score": None,
        "negative_space_ratio": None,
        "edge_density": None,
        "line_strength": None,
        "face_present": None,
        "object_count": None,
        "depth_mean": None,
        "style_minimalist": None,
        "style_cinematic": None,
        "style_vlog": None,
        "style_product_centered": None,
    }
    
    # Timeline data (per-frame features)
    timeline = []
    if frame_feature_values is not None and frame_feature_names and times_s is not None and frame_indices is not None:
        n_frames = len(frame_indices)
        n_features = len(frame_feature_names)
        
        # Find indices of key features for timeline
        
        for i, name in enumerate(frame_feature_names):
            for key in key_features:
                if key in name:
                    key_features[key] = i
                    break
        
        for i in range(n_frames):
            if i >= len(times_s) or i >= len(frame_indices):
                break
            
            frame_idx = int(frame_indices[i])
            time_sec = float(times_s[i])
            
            timeline_entry = {
                "frame_index": frame_idx,
                "time_sec": time_sec,
            }
            
            # Add key features if available
            for key, idx in key_features.items():
                if idx is not None and idx < n_features and i < frame_feature_values.shape[0]:
                    val = float(frame_feature_values[i, idx]) if np.isfinite(frame_feature_values[i, idx]) else None
                    if val is not None:
                        timeline_entry[key] = val
            
            timeline.append(timeline_entry)
    
    render["timeline"] = timeline
    
    # Distribution statistics for per-frame features
    distributions = {}
    
    if timeline and frame_feature_names:
        # Collect values for each key feature
        feature_vals_map = {}
        for key in key_features:
            if key_features[key] is not None:
                idx = key_features[key]
                vals = []
                for entry in timeline:
                    if key in entry and entry[key] is not None:
                        vals.append(entry[key])
                if vals:
                    feature_vals_map[key] = np.array(vals, dtype=np.float32)
        
        # Compute distributions
        for key, vals in feature_vals_map.items():
            if vals.size > 0:
                distributions[key] = {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                }
    
    render["distributions"] = distributions
    
    return render


def render_frames_composition_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага frames_composition результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(vp_root / "modules") not in sys.path:
        sys.path.insert(0, str(vp_root / "modules"))
    
    # Try to import from utils if renderer exists
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except ImportError:
        # Fallback: direct load
        def load_npz(path: str):
            data = np.load(path, allow_pickle=True)
            result = {}
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.dtype == object:
                        result[key] = arr.item() if arr.size == 1 else arr.tolist()
                    else:
                        result[key] = arr.tolist() if arr.size > 0 else []
                else:
                    result[key] = arr
            return result
        
        def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
            meta = npz_data.get("meta")
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                return meta.item() if meta.size == 1 else meta.tolist()
            return meta if isinstance(meta, dict) else {}
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_frames_composition(npz_data, meta)
    
    timeline = render.get("timeline", []) or []
    summary = render.get("summary", {}) or {}
    distributions = render.get("distributions", {}) or {}

    # Key facts (Audit v3)
    key_facts: List[str] = []
    status = str(meta.get("status") or "unknown")
    empty_reason = meta.get("empty_reason")
    key_facts.append(f"status: {status}" + (f" ({empty_reason})" if empty_reason else ""))
    if meta.get("producer_version"):
        key_facts.append(f"producer_version: {meta.get('producer_version')}")
    if meta.get("schema_version"):
        key_facts.append(f"schema_version: {meta.get('schema_version')}")
    if meta.get("sampling_policy_version"):
        key_facts.append(f"sampling_policy_version: {meta.get('sampling_policy_version')}")
    if isinstance(summary.get("frames_count"), int):
        key_facts.append(f"frames_count: {summary.get('frames_count')}")

    config_highlights = {k: meta.get(k) for k in ["feature_set", "features", "num_workers"] if k in meta}
    stage_timings = meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {}
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Offline SVG charts
    charts_html = ""
    if timeline:
        t_arr = np.asarray([t.get("time_sec", np.nan) for t in timeline], dtype=np.float64)
        def _arr(name: str) -> np.ndarray:
            return np.asarray([t.get(name, np.nan) for t in timeline], dtype=np.float64)
        charts_html = "\n".join(
            [
                _svg_line_chart(times_s=t_arr, values=_arr("saliency_center_offset"), title="saliency_center_offset", stroke="#2563eb"),
                _svg_line_chart(times_s=t_arr, values=_arr("symmetry_score"), title="symmetry_score", stroke="#14b8a6"),
                _svg_line_chart(times_s=t_arr, values=_arr("negative_space_ratio"), title="negative_space_ratio", stroke="#f59e0b"),
                _svg_line_chart(times_s=t_arr, values=_arr("edge_density"), title="edge_density", stroke="#a855f7"),
                _svg_line_chart(times_s=t_arr, values=_arr("line_strength"), title="line_strength", stroke="#ef4444"),
            ]
        )

    def _topk(metric: str, k: int = 5, reverse: bool = True) -> List[Dict[str, Any]]:
        rows: List[Tuple[float, Dict[str, Any]]] = []
        for t in timeline:
            v = t.get(metric)
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                rows.append((float(v), t))
        rows.sort(key=lambda x: x[0], reverse=reverse)
        out: List[Dict[str, Any]] = []
        for v, t in rows[:k]:
            out.append({"time_sec": t.get("time_sec"), "frame_index": t.get("frame_index"), metric: v})
        return out

    top_examples = {
        "saliency_center_offset_top": _topk("saliency_center_offset", k=5, reverse=True),
        "edge_density_top": _topk("edge_density", k=5, reverse=True),
    }
    anti_top_examples = {
        "saliency_center_offset_low": _topk("saliency_center_offset", k=5, reverse=False),
        "edge_density_low": _topk("edge_density", k=5, reverse=False),
    }
    
    # Style probabilities display
    style_probs_html = ""
    style_probs = summary.get("style_probabilities", {})
    if style_probs:
        style_probs_html = "<div class='card'><h3>Style Probabilities</h3><pre class='mono'>"
        style_probs_html += _esc(json.dumps(style_probs, ensure_ascii=False, indent=2))
        style_probs_html += "</pre></div>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>frames_composition — debug render (offline)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; background: #f9fafb; color: #111827; }}
    .container {{ background: #fff; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb; max-width: 1100px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
    h2 {{ margin: 18px 0 10px 0; font-size: 16px; }}
    .muted {{ color: #6b7280; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }}
    .chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .chart-title {{ font-weight: 600; margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ color: #374151; background: #f9fafb; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    pre.mono {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1><code>frames_composition</code> — debug render (offline)</h1>
    <div class="muted">NPZ: {_esc(os.path.basename(npz_path))}</div>

    <h2>Key facts</h2>
    <div class="card">
      <ul>
        {''.join(f'<li>{_esc(x)}</li>' for x in key_facts)}
      </ul>
    </div>

    <h2>Summary</h2>
    <div class="grid">
      <div class="card"><div class="muted">frames_count</div><div><strong>{_esc(summary.get('frames_count', 0))}</strong></div></div>
      <div class="card"><div class="muted">has_faces</div><div><strong>{_esc(summary.get('has_faces', False))}</strong></div></div>
      <div class="card"><div class="muted">style_dominant_id</div><div><strong>{_esc(summary.get('style_dominant_id', 'N/A'))}</strong></div></div>
    </div>

    {style_probs_html}

    <h2>Timeline charts (offline SVG)</h2>
    {f'<div class="charts">{charts_html}</div>' if charts_html else '<p class="muted">No timeline data available</p>'}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Saliency Offset</th>
                        <th>Symmetry Score</th>
                        <th>Negative Space</th>
                        <th>Edge Density</th>
                        <th>Line Strength</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'min')}</td>
                        <td>{format_dist_value('symmetry_score', 'min')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'min')}</td>
                        <td>{format_dist_value('edge_density', 'min')}</td>
                        <td>{format_dist_value('line_strength', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'max')}</td>
                        <td>{format_dist_value('symmetry_score', 'max')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'max')}</td>
                        <td>{format_dist_value('edge_density', 'max')}</td>
                        <td>{format_dist_value('line_strength', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'mean')}</td>
                        <td>{format_dist_value('symmetry_score', 'mean')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'mean')}</td>
                        <td>{format_dist_value('edge_density', 'mean')}</td>
                        <td>{format_dist_value('line_strength', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'std')}</td>
                        <td>{format_dist_value('symmetry_score', 'std')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'std')}</td>
                        <td>{format_dist_value('edge_density', 'std')}</td>
                        <td>{format_dist_value('line_strength', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'median')}</td>
                        <td>{format_dist_value('symmetry_score', 'median')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'median')}</td>
                        <td>{format_dist_value('edge_density', 'median')}</td>
                        <td>{format_dist_value('line_strength', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}

    <h2>Top / anti-top frames</h2>
    <div class="grid">
      <div class="card"><div class="chart-title">Top saliency_center_offset</div><pre class="mono">{_esc(json.dumps(top_examples.get('saliency_center_offset_top', []), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><div class="chart-title">Low saliency_center_offset</div><pre class="mono">{_esc(json.dumps(anti_top_examples.get('saliency_center_offset_low', []), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><div class="chart-title">Top edge_density</div><pre class="mono">{_esc(json.dumps(top_examples.get('edge_density_top', []), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><div class="chart-title">Low edge_density</div><pre class="mono">{_esc(json.dumps(anti_top_examples.get('edge_density_low', []), ensure_ascii=False, indent=2))}</pre></div>
    </div>

    <h2>Config highlights</h2>
    <div class="card"><pre class="mono">{_esc(json.dumps(config_highlights, ensure_ascii=False, indent=2))}</pre></div>

    <h2>Stage timings (ms)</h2>
    <div class="card"><pre class="mono">{_esc(json.dumps(stage_timings, ensure_ascii=False, indent=2))}</pre></div>
  </div>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Frames Composition HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_frames_composition", "render_frames_composition_html"]

