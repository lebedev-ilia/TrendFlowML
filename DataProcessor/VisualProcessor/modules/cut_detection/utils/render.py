"""
Renderer для cut_detection: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

def _to_list_int(x: Any) -> List[int]:
    try:
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, list):
            out: List[int] = []
            for v in x:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    out.append(int(v))
            return out
    except Exception:
        pass
    return []


def render_cut_detection(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для cut_detection."""
    render = {
        "component": "cut_detection",
        "key_facts": {},
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract data
    features = npz_data.get("features")
    detections = npz_data.get("detections")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if times_s is not None:
        if isinstance(times_s, list):
            times_s = np.array(times_s, dtype=np.float32)
        elif isinstance(times_s, np.ndarray):
            times_s = np.asarray(times_s, dtype=np.float32)
        else:
            times_s = None
    
    if frame_indices is not None:
        if isinstance(frame_indices, list):
            frame_indices = np.array(frame_indices, dtype=np.int32)
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = np.asarray(frame_indices, dtype=np.int32)
        else:
            frame_indices = None
    
    # Extract features
    if features is not None:
        if isinstance(features, np.ndarray) and features.dtype == object:
            features = features.item() if features.size == 1 else features.tolist()
        if not isinstance(features, dict):
            features = {}
    else:
        features = {}
    
    # Extract detections
    if detections is not None:
        if isinstance(detections, np.ndarray) and detections.dtype == object:
            detections = detections.item() if detections.size == 1 else detections.tolist()
        if not isinstance(detections, dict):
            detections = {}
    else:
        detections = {}
    
    # Summary statistics
    render["summary"] = {
        "frames_count": int(len(times_s)) if times_s is not None and len(times_s) > 0 else 0,
        "hard_cuts_count": int(features.get("hard_cuts_count", 0)),
        "hard_cuts_per_minute": float(features.get("hard_cuts_per_minute", 0.0)),
        "hard_cut_strength_mean": float(features.get("hard_cut_strength_mean", 0.0)),
        "fade_in_count": int(features.get("fade_in_count", 0)),
        "fade_out_count": int(features.get("fade_out_count", 0)),
        "dissolve_count": int(features.get("dissolve_count", 0)),
        "motion_cuts_count": int(features.get("motion_cuts_count", 0)),
        "jump_cuts_count": int(features.get("jump_cuts_count", 0)),
        "cuts_per_minute": float(features.get("cuts_per_minute", 0.0)),
        "avg_shot_length": float(features.get("avg_shot_length", 0.0)),
        "median_shot_length": float(features.get("median_shot_length", 0.0)),
        "scene_count": int(features.get("scene_count", 0)),
    }

    if isinstance(meta, dict):
        render["key_facts"] = {
            "schema_version": meta.get("schema_version", "unknown"),
            "producer_version": meta.get("producer_version", "unknown"),
            "status": meta.get("status", "unknown"),
            "empty_reason": meta.get("empty_reason"),
            "model_facing_npz_path": npz_data.get("model_facing_npz_path"),
        }
    
    # Timeline data: cuts and transitions over time
    timeline = []
    if times_s is not None and len(times_s) > 0:
        # Hard cuts
        hard_cuts = (
            detections.get("hard_cut_pos")
            or detections.get("hard_cut_indices")
            or detections.get("hard_cuts")
            or []
        )
        hard_cuts_set = set(_to_list_int(hard_cuts))
        
        # Soft events
        soft_events = detections.get("soft_events", [])
        if isinstance(soft_events, np.ndarray) and soft_events.dtype == object:
            soft_events = soft_events.item() if soft_events.size == 1 else soft_events.tolist()
        if not isinstance(soft_events, list):
            soft_events = []
        
        # Motion cuts
        motion_cuts = (
            detections.get("motion_cut_pos")
            or detections.get("motion_cut_indices")
            or detections.get("motion_cuts")
            or []
        )
        motion_cuts_set = set(_to_list_int(motion_cuts))
        
        # Jump cuts
        jump_cuts = (
            detections.get("jump_cut_pos")
            or detections.get("jump_cut_indices")
            or detections.get("jump_cuts")
            or []
        )
        jump_cuts_set = set(_to_list_int(jump_cuts))
        
        # Build timeline
        for i, time_sec in enumerate(times_s):
            frame_idx = int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i
            
            # Check if this frame is a cut
            is_hard_cut = i in hard_cuts_set
            is_motion_cut = i in motion_cuts_set
            is_jump_cut = i in jump_cuts_set
            
            # Find soft events at this time
            active_soft_events = []
            for event in soft_events:
                if isinstance(event, dict):
                    start_idx = event.get("start", -1)
                    end_idx = event.get("end", -1)
                    if start_idx <= i <= end_idx:
                        active_soft_events.append(event.get("type", "unknown"))
            
            timeline.append({
                "time_sec": float(time_sec),
                "frame_index": frame_idx,
                "is_hard_cut": bool(is_hard_cut),
                "is_motion_cut": bool(is_motion_cut),
                "is_jump_cut": bool(is_jump_cut),
                "active_soft_events": active_soft_events,
            })
        
        render["timeline"] = timeline
    
    # Distribution statistics
    distributions = {}
    
    # Hard cut strength distribution
    hard_cut_strength_mean = float(features.get("hard_cut_strength_mean", 0.0))
    hard_cut_strength_p25 = float(features.get("hard_cut_strength_p25", 0.0))
    hard_cut_strength_p50 = float(features.get("hard_cut_strength_p50", 0.0))
    hard_cut_strength_p75 = float(features.get("hard_cut_strength_p75", 0.0))
    
    distributions["hard_cut_strength"] = {
        "min": 0.0,
        "max": 1.0,
        "mean": hard_cut_strength_mean,
        "p25": hard_cut_strength_p25,
        "p50": hard_cut_strength_p50,
        "p75": hard_cut_strength_p75,
    }
    
    # Shot length distribution
    avg_shot_length = float(features.get("avg_shot_length", 0.0))
    median_shot_length = float(features.get("median_shot_length", 0.0))
    min_shot_length = float(features.get("min_shot_length", 0.0))
    max_shot_length = float(features.get("max_shot_length", 0.0))
    
    distributions["shot_length"] = {
        "min": min_shot_length,
        "max": max_shot_length,
        "mean": avg_shot_length,
        "median": median_shot_length,
    }
    
    # Cut interval distribution
    median_cut_interval = float(features.get("median_cut_interval", 0.0)) if not np.isnan(float(features.get("median_cut_interval", float("nan")))) else 0.0
    min_cut_interval = float(features.get("min_cut_interval", 0.0)) if not np.isnan(float(features.get("min_cut_interval", float("nan")))) else 0.0
    max_cut_interval = float(features.get("max_cut_interval", 0.0)) if not np.isnan(float(features.get("max_cut_interval", float("nan")))) else 0.0
    cut_interval_mean = median_cut_interval  # Use median as mean approximation
    
    distributions["cut_interval"] = {
        "min": min_cut_interval,
        "max": max_cut_interval,
        "mean": cut_interval_mean,
        "median": median_cut_interval,
    }
    
    render["distributions"] = distributions
    
    # Top events summary
    render["top_events"] = {
        "hard_cuts": int(features.get("hard_cuts_count", 0)),
        "fade_in": int(features.get("fade_in_count", 0)),
        "fade_out": int(features.get("fade_out_count", 0)),
        "dissolve": int(features.get("dissolve_count", 0)),
        "motion_cuts": int(features.get("motion_cuts_count", 0)),
        "jump_cuts": int(features.get("jump_cuts_count", 0)),
    }

    # Top hard cuts by strength (QA helper)
    try:
        hard_strengths = detections.get("hard_cut_strengths") or []
        hard_positions = _to_list_int(
            detections.get("hard_cut_pos")
            or detections.get("hard_cut_indices")
            or detections.get("hard_cuts")
            or []
        )
        hs = np.asarray(hard_strengths, dtype=np.float32).reshape(-1)
        m = int(min(len(hard_positions), int(hs.size)))
        if m > 0 and times_s is not None:
            pairs = []
            for j in range(m):
                pos = int(hard_positions[j])
                strength = float(hs[j]) if np.isfinite(hs[j]) else float("nan")
                ti = max(0, min(pos, int(len(times_s) - 1)))
                pairs.append({"pos": pos, "time_s": float(times_s[ti]), "strength": strength})
            pairs = sorted(pairs, key=lambda x: (-(x["strength"] if np.isfinite(x["strength"]) else -1e9)))
            render["top"] = {"hard_cuts_by_strength": pairs[:20]}
    except Exception:
        pass
    
    return render


def render_cut_detection_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML debug страницу для cut_detection.
    
    Args:
        npz_path: Путь к NPZ файлу с результатами
        output_path: Путь для сохранения HTML файла
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(vp_root / "core" / "model_process") not in sys.path:
        sys.path.insert(0, str(vp_root / "core" / "model_process"))
    
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
    render = render_cut_detection(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    top_events = render.get("top_events", {})
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        hard_cuts = [1.0 if t.get("is_hard_cut") else 0.0 for t in timeline]
        motion_cuts = [1.0 if t.get("is_motion_cut") else 0.0 for t in timeline]
        jump_cuts = [1.0 if t.get("is_jump_cut") else 0.0 for t in timeline]
        
        # Build datasets array
        datasets = []
        
        if any(hard_cuts):
            datasets.append({
                "label": "Hard Cuts",
                "data": hard_cuts,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y",
                "pointRadius": 3,
            })
        
        if any(motion_cuts):
            datasets.append({
                "label": "Motion Cuts",
                "data": motion_cuts,
                "borderColor": "rgb(54, 162, 235)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "tension": 0.1,
                "yAxisID": "y",
                "pointRadius": 3,
            })
        
        if any(jump_cuts):
            datasets.append({
                "label": "Jump Cuts",
                "data": jump_cuts,
                "borderColor": "rgb(255, 206, 86)",
                "backgroundColor": "rgba(255, 206, 86, 0.2)",
                "tension": 0.1,
                "yAxisID": "y",
                "pointRadius": 3,
            })
        
        if datasets:
            # Format time labels
            time_labels = [f"{t:.2f}s" for t in times]
            timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps(time_labels)},
            datasets: {json.dumps(datasets)}
        }};
        """
    
    # Prepare top events table
    top_events_rows = ''.join([
        f'''
        <tr>
            <td>{event_name.replace('_', ' ').title()}</td>
            <td>{count}</td>
        </tr>
        ''' for event_name, count in top_events.items() if count > 0
    ])

    key_facts = render.get("key_facts", {}) if isinstance(render, dict) else {}
    top_hard = ((render.get("top") or {}).get("hard_cuts_by_strength") or []) if isinstance(render, dict) else []
    top_hard_rows = ""
    for e in top_hard:
        if not isinstance(e, dict):
            continue
        try:
            pos = e.get("pos", "")
            ts = float(e.get("time_s", 0.0))
            strength = float(e.get("strength", 0.0))
            top_hard_rows += f"""
            <tr>
                <td>{pos}</td>
                <td>{ts:.3f}</td>
                <td>{strength:.4f}</td>
            </tr>
            """
        except Exception:
            continue
    
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>cut_detection — render (dev-only)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin: 20px 0; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .top-events {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .top-events table {{ width: 100%; border-collapse: collapse; }}
        .top-events th, .top-events td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .top-events th {{ background-color: #0056b3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>cut_detection — render (dev-only)</h1>
        <p><em>Offline HTML (no CDN). NPZ is source-of-truth.</em></p>

        <div class="top-events">
            <h2>Key facts</h2>
            <table>
                <tbody>
                    <tr><td><strong>schema_version</strong></td><td>{key_facts.get('schema_version','unknown')}</td></tr>
                    <tr><td><strong>producer_version</strong></td><td>{key_facts.get('producer_version','unknown')}</td></tr>
                    <tr><td><strong>status</strong></td><td>{key_facts.get('status','unknown')}</td></tr>
                    <tr><td><strong>empty_reason</strong></td><td>{key_facts.get('empty_reason')}</td></tr>
                    <tr><td><strong>model_facing_npz_path</strong></td><td>{key_facts.get('model_facing_npz_path')}</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Hard Cuts</strong>
                    <span class="metric-value">{summary.get('hard_cuts_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Hard Cuts/Min</strong>
                    <span class="metric-value">{summary.get('hard_cuts_per_minute', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>Hard Cut Strength</strong>
                    <span class="metric-value">{summary.get('hard_cut_strength_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Fade In</strong>
                    <span class="metric-value">{summary.get('fade_in_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Fade Out</strong>
                    <span class="metric-value">{summary.get('fade_out_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Dissolve</strong>
                    <span class="metric-value">{summary.get('dissolve_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Cuts</strong>
                    <span class="metric-value">{summary.get('motion_cuts_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Jump Cuts</strong>
                    <span class="metric-value">{summary.get('jump_cuts_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Shot Length</strong>
                    <span class="metric-value">{summary.get('avg_shot_length', 0.0):.2f}s</span>
                </div>
                <div class="metric-card">
                    <strong>Median Shot Length</strong>
                    <span class="metric-value">{summary.get('median_shot_length', 0.0):.2f}s</span>
                </div>
                <div class="metric-card">
                    <strong>Scene Count</strong>
                    <span class="metric-value">{summary.get('scene_count', 0)}</span>
                </div>
            </div>
        </div>

        <div class="top-events">
            <h2>Top hard cuts (by strength)</h2>
            <table>
                <thead>
                    <tr>
                        <th>pos</th>
                        <th>time_s</th>
                        <th>strength</th>
                    </tr>
                </thead>
                <tbody>
                    {top_hard_rows if top_hard_rows else '<tr><td colspan="3">No data</td></tr>'}
                </tbody>
            </table>
        </div>
        
        {f'''
        <div class="top-events">
            <h2>Top Events</h2>
            <table>
                <thead>
                    <tr>
                        <th>Event Type</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {top_events_rows}
                </tbody>
            </table>
        </div>
        ''' if top_events_rows else ''}
        
        <p><strong>Timeline</strong>: доступен в `render_context.json` как `timeline[]` (без JS‑графиков, чтобы HTML был offline).</p>
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Hard Cut Strength</th>
                        <th>Shot Length</th>
                        <th>Cut Interval</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('hard_cut_strength', {}).get('min', 0.0):.4f}</td>
                        <td>{distributions.get('shot_length', {}).get('min', 0.0):.2f}s</td>
                        <td>{distributions.get('cut_interval', {}).get('min', 0.0):.2f}s</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('hard_cut_strength', {}).get('max', 0.0):.4f}</td>
                        <td>{distributions.get('shot_length', {}).get('max', 0.0):.2f}s</td>
                        <td>{distributions.get('cut_interval', {}).get('max', 0.0):.2f}s</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('hard_cut_strength', {}).get('mean', 0.0):.4f}</td>
                        <td>{distributions.get('shot_length', {}).get('mean', 0.0):.2f}s</td>
                        <td>{distributions.get('cut_interval', {}).get('mean', 0.0):.2f}s</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{distributions.get('hard_cut_strength', {}).get('p50', 0.0):.4f}</td>
                        <td>{distributions.get('shot_length', {}).get('median', 0.0):.2f}s</td>
                        <td>{distributions.get('cut_interval', {}).get('median', 0.0):.2f}s</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
    </div>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved Cut Detection HTML render to {output_path}")
    return output_path


__all__ = ["render_cut_detection", "render_cut_detection_html"]

