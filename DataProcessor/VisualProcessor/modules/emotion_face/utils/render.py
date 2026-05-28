"""
Renderer для emotion_face: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _to_1d_float(arr: Any) -> np.ndarray:
    try:
        if arr is None:
            return np.asarray([], dtype=np.float32)
        if isinstance(arr, np.ndarray):
            return np.asarray(arr, dtype=np.float32).reshape(-1)
        if isinstance(arr, list):
            out = []
            for v in arr:
                if v is None:
                    out.append(np.nan)
                else:
                    out.append(float(v))
            return np.asarray(out, dtype=np.float32).reshape(-1)
    except Exception:
        pass
    return np.asarray([], dtype=np.float32)


def _svg_line_chart(
    *,
    times_s: Any,
    values: Any,
    width: int = 1000,
    height: int = 240,
    stroke: str = "#2563eb",
    title: str = "",
) -> str:
    t = _to_1d_float(times_s)
    v = _to_1d_float(values)
    n = int(min(t.size, v.size))
    if n <= 1:
        return "<div class='muted'>No data</div>"
    t = t[:n]
    v = v[:n]
    max_points = 500
    step = max(1, int(np.ceil(float(n) / float(max_points))))
    t = t[::step]
    v = v[::step]
    finite = np.isfinite(t) & np.isfinite(v)
    if not np.any(finite):
        return "<div class='muted'>No finite data</div>"
    t_f = t[finite]
    v_f = v[finite]
    t_min = float(np.min(t_f))
    t_max = float(np.max(t_f))
    v_min = float(np.min(v_f))
    v_max = float(np.max(v_f))
    if abs(t_max - t_min) < 1e-9:
        t_max = t_min + 1e-6
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1e-6
    pad_x = 40
    pad_y = 24
    w = max(240, int(width))
    h = max(120, int(height))
    x0, y0 = pad_x, pad_y
    x1, y1 = w - pad_x, h - pad_y

    def _sx(tt: float) -> float:
        return x0 + (tt - t_min) / (t_max - t_min) * (x1 - x0)

    def _sy(vv: float) -> float:
        return y1 - (vv - v_min) / (v_max - v_min) * (y1 - y0)

    path_parts: List[str] = []
    pen_down = False
    for tt, vv, ok in zip(t.tolist(), v.tolist(), finite.tolist()):
        if not ok:
            pen_down = False
            continue
        x = _sx(float(tt))
        y = _sy(float(vv))
        if not pen_down:
            path_parts.append(f"M {x:.2f} {y:.2f}")
            pen_down = True
        else:
            path_parts.append(f"L {x:.2f} {y:.2f}")

    path_d = " ".join(path_parts)
    esc_title = (title or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div class="chart">
      <div class="chart-title">{esc_title}</div>
      <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc_title}">
        <rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff" stroke="#e5e7eb"/>
        <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#e5e7eb"/>
        <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#e5e7eb"/>
        <path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="2"/>
        <text x="{x0}" y="{y0 - 8}" font-size="12" fill="#6b7280">{v_max:.3f}</text>
        <text x="{x0}" y="{y1 + 16}" font-size="12" fill="#6b7280">{v_min:.3f}</text>
        <text x="{x0}" y="{h - 4}" font-size="12" fill="#6b7280">{t_min:.2f}s</text>
        <text x="{x1 - 40}" y="{h - 4}" font-size="12" fill="#6b7280">{t_max:.2f}s</text>
      </svg>
    </div>
    """.strip()


def render_emotion_face(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для emotion_face."""
    render = {
        "component": "emotion_face",
        "key_facts": {},
        "summary": {},
        "timeline": [],
        "distributions": {},
    }

    if isinstance(meta, dict):
        render["key_facts"] = {
            "schema_version": meta.get("schema_version"),
            "producer_version": meta.get("producer_version"),
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "module_sampling_policy_version": meta.get("module_sampling_policy_version"),
            "face_frames_sampling_policy_version": meta.get("face_frames_sampling_policy_version"),
            "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        }
    
    # Extract emotion data
    sequence_features = npz_data.get("sequence_features", {})
    if not isinstance(sequence_features, dict):
        sequence_features = {}
    
    frame_indices = sequence_features.get("frame_indices")
    times_s = sequence_features.get("times_s")
    valence = sequence_features.get("valence")
    arousal = sequence_features.get("arousal")
    intensity = sequence_features.get("intensity")
    emotion_confidence = sequence_features.get("emotion_confidence")
    emotion_probs = sequence_features.get("emotion_probs")
    dominant_emotion_id = sequence_features.get("dominant_emotion_id")
    face_count = sequence_features.get("face_count")
    
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
    
    if valence is not None:
        if isinstance(valence, list):
            valence = np.array(valence, dtype=np.float32)
        elif isinstance(valence, np.ndarray):
            valence = np.asarray(valence, dtype=np.float32)
        else:
            valence = None
    
    if arousal is not None:
        if isinstance(arousal, list):
            arousal = np.array(arousal, dtype=np.float32)
        elif isinstance(arousal, np.ndarray):
            arousal = np.asarray(arousal, dtype=np.float32)
        else:
            arousal = None
    
    if intensity is not None:
        if isinstance(intensity, list):
            intensity = np.array(intensity, dtype=np.float32)
        elif isinstance(intensity, np.ndarray):
            intensity = np.asarray(intensity, dtype=np.float32)
        else:
            intensity = None
    
    if emotion_confidence is not None:
        if isinstance(emotion_confidence, list):
            emotion_confidence = np.array(emotion_confidence, dtype=np.float32)
        elif isinstance(emotion_confidence, np.ndarray):
            emotion_confidence = np.asarray(emotion_confidence, dtype=np.float32)
        else:
            emotion_confidence = None
    
    # Summary statistics
    if frame_indices is not None and frame_indices.size > 0:
        n_frames = int(frame_indices.shape[0])
        
        render["summary"] = {
            "frames_count": n_frames,
        }
        
        if valence is not None and valence.size > 0:
            valid_valence = valence[np.isfinite(valence)]
            if valid_valence.size > 0:
                render["summary"]["valence_mean"] = float(np.mean(valid_valence))
                render["summary"]["valence_std"] = float(np.std(valid_valence))
                render["summary"]["valence_min"] = float(np.min(valid_valence))
                render["summary"]["valence_max"] = float(np.max(valid_valence))
        
        if arousal is not None and arousal.size > 0:
            valid_arousal = arousal[np.isfinite(arousal)]
            if valid_arousal.size > 0:
                render["summary"]["arousal_mean"] = float(np.mean(valid_arousal))
                render["summary"]["arousal_std"] = float(np.std(valid_arousal))
                render["summary"]["arousal_min"] = float(np.min(valid_arousal))
                render["summary"]["arousal_max"] = float(np.max(valid_arousal))
        
        if intensity is not None and intensity.size > 0:
            valid_intensity = intensity[np.isfinite(intensity)]
            if valid_intensity.size > 0:
                render["summary"]["intensity_mean"] = float(np.mean(valid_intensity))
                render["summary"]["intensity_std"] = float(np.std(valid_intensity))
                render["summary"]["intensity_min"] = float(np.min(valid_intensity))
                render["summary"]["intensity_max"] = float(np.max(valid_intensity))
        
        if emotion_confidence is not None and emotion_confidence.size > 0:
            valid_conf = emotion_confidence[np.isfinite(emotion_confidence)]
            if valid_conf.size > 0:
                render["summary"]["emotion_confidence_mean"] = float(np.mean(valid_conf))
                render["summary"]["emotion_confidence_std"] = float(np.std(valid_conf))
        
        if face_count is not None and face_count.size > 0:
            valid_face_count = face_count[face_count >= 0]
            if valid_face_count.size > 0:
                render["summary"]["face_count_mean"] = float(np.mean(valid_face_count))
                render["summary"]["face_count_max"] = int(np.max(valid_face_count))
                render["summary"]["faces_found_frames"] = int(np.sum(valid_face_count > 0))
        
        # Dominant emotion distribution
        if dominant_emotion_id is not None and dominant_emotion_id.size > 0:
            emotion_classes = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
            valid_emotion_ids = dominant_emotion_id[dominant_emotion_id >= 0]
            if valid_emotion_ids.size > 0:
                unique, counts = np.unique(valid_emotion_ids, return_counts=True)
                emotion_dist = {}
                for emo_id, count in zip(unique, counts):
                    if 0 <= int(emo_id) < len(emotion_classes):
                        emotion_dist[emotion_classes[int(emo_id)]] = int(count)
                render["summary"]["dominant_emotion_distribution"] = emotion_dist
    
    # Timeline data (per-frame statistics)
    if frame_indices is not None and times_s is not None and valence is not None and arousal is not None:
        n = min(frame_indices.size, times_s.size, valence.size, arousal.size)
        if n > 0:
            timeline = []
            for i in range(n):
                if not (np.isfinite(times_s[i]) and np.isfinite(valence[i]) and np.isfinite(arousal[i])):
                    continue
                timeline.append({
                    "frame_index": int(frame_indices[i]),
                    "time_s": float(times_s[i]),
                    "valence": float(valence[i]),
                    "arousal": float(arousal[i]),
                    "intensity": float(intensity[i]) if intensity is not None and i < intensity.size and np.isfinite(intensity[i]) else None,
                    "emotion_confidence": float(emotion_confidence[i]) if emotion_confidence is not None and i < emotion_confidence.size and np.isfinite(emotion_confidence[i]) else None,
                    "face_count": int(face_count[i]) if face_count is not None and i < face_count.size else 0,
                })
            render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if valence is not None and valence.size > 0:
        valid_valence = valence[np.isfinite(valence)]
        if valid_valence.size > 0:
            distributions["valence"] = {
                "mean": float(np.mean(valid_valence)),
                "std": float(np.std(valid_valence)),
                "min": float(np.min(valid_valence)),
                "max": float(np.max(valid_valence)),
                "median": float(np.median(valid_valence)),
                "p25": float(np.percentile(valid_valence, 25)),
                "p75": float(np.percentile(valid_valence, 75)),
            }
    
    if arousal is not None and arousal.size > 0:
        valid_arousal = arousal[np.isfinite(arousal)]
        if valid_arousal.size > 0:
            distributions["arousal"] = {
                "mean": float(np.mean(valid_arousal)),
                "std": float(np.std(valid_arousal)),
                "min": float(np.min(valid_arousal)),
                "max": float(np.max(valid_arousal)),
                "median": float(np.median(valid_arousal)),
                "p25": float(np.percentile(valid_arousal, 25)),
                "p75": float(np.percentile(valid_arousal, 75)),
            }
    
    if intensity is not None and intensity.size > 0:
        valid_intensity = intensity[np.isfinite(intensity)]
        if valid_intensity.size > 0:
            distributions["intensity"] = {
                "mean": float(np.mean(valid_intensity)),
                "std": float(np.std(valid_intensity)),
                "min": float(np.min(valid_intensity)),
                "max": float(np.max(valid_intensity)),
                "median": float(np.median(valid_intensity)),
                "p25": float(np.percentile(valid_intensity, 25)),
                "p75": float(np.percentile(valid_intensity, 75)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_emotion_face_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага emotion_face результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent
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
    render = render_emotion_face(npz_data, meta)
    
    timeline = render.get("timeline", [])
    key_facts = render.get("key_facts", {})
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})

    charts_html = ""
    top_rows: List[Tuple[int, float, float]] = []
    anti_rows: List[Tuple[int, float, float]] = []
    if timeline:
        times = [t.get("time_s", 0.0) for t in timeline]

        def _col(name: str) -> List[Optional[float]]:
            return [t.get(name) for t in timeline]

        charts_html = "\n".join(
            [
                _svg_line_chart(times_s=times, values=_col("valence"), title="valence", stroke="#f43f5e"),
                _svg_line_chart(times_s=times, values=_col("arousal"), title="arousal", stroke="#0ea5e9"),
                _svg_line_chart(times_s=times, values=_col("intensity"), title="intensity", stroke="#f59e0b"),
                _svg_line_chart(times_s=times, values=_col("emotion_confidence"), title="emotion_confidence", stroke="#22c55e"),
                _svg_line_chart(times_s=times, values=_col("face_count"), title="face_count", stroke="#111827"),
            ]
        )

        def _safe_float(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                fv = float(v)
                return fv if np.isfinite(fv) else None
            except Exception:
                return None

        scored_int: List[Tuple[float, int, float]] = []
        scored_conf: List[Tuple[float, int, float]] = []
        for t in timeline:
            fi = int(t.get("frame_index", -1))
            ts = float(t.get("time_s", 0.0))
            it = _safe_float(t.get("intensity"))
            cf = _safe_float(t.get("emotion_confidence"))
            if it is not None:
                scored_int.append((it, fi, ts))
            if cf is not None:
                scored_conf.append((cf, fi, ts))
        scored_int.sort(key=lambda x: x[0], reverse=True)
        scored_conf.sort(key=lambda x: x[0])
        for it, fi, ts in scored_int[:5]:
            top_rows.append((fi, ts, it))
        for cf, fi, ts in scored_conf[:5]:
            anti_rows.append((fi, ts, cf))
    
    # Format distribution values
    def format_dist_value(category: str, stat: str) -> str:
        if category not in distributions:
            return "-"
        val = distributions[category].get(stat)
        if val is None:
            return "-"
        return f"{val:.4f}"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Face Debug Render</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .muted {{ color: #6b7280; }}
        .keyfacts {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #dee2e6; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin: 20px 0; }}
        .chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
        .chart-title {{ font-weight: 600; margin-bottom: 6px; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .examples table {{ width: 100%; border-collapse: collapse; }}
        .examples th, .examples td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .examples th {{ background-color: #111827; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Emotion Face Debug Render</h1>

        <div class="keyfacts">
            <h2>Key facts</h2>
            <div class="metric-grid">
                <div class="metric-card"><strong>schema_version</strong><span class="metric-value">{key_facts.get('schema_version')}</span></div>
                <div class="metric-card"><strong>producer_version</strong><span class="metric-value">{key_facts.get('producer_version')}</span></div>
                <div class="metric-card"><strong>status</strong><span class="metric-value">{key_facts.get('status')}</span></div>
                <div class="metric-card"><strong>empty_reason</strong><span class="metric-value">{key_facts.get('empty_reason')}</span></div>
            </div>
            <div class="muted">stage_timings_ms: {json.dumps(key_facts.get("stage_timings_ms", {}), ensure_ascii=False)}</div>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Valence Mean</strong>
                    <span class="metric-value">{summary.get('valence_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Arousal Mean</strong>
                    <span class="metric-value">{summary.get('arousal_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Intensity Mean</strong>
                    <span class="metric-value">{summary.get('intensity_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Emotion Confidence Mean</strong>
                    <span class="metric-value">{summary.get('emotion_confidence_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Faces Found Frames</strong>
                    <span class="metric-value">{summary.get('faces_found_frames', 0)}</span>
                </div>
            </div>
        </div>
        
        <h2>Timeline charts (offline)</h2>
        {f'<div class="charts">{charts_html}</div>' if timeline else '<p class="muted">No timeline data available</p>'}

        <div class="examples">
            <h2>Top / Anti-top examples</h2>
            <h3>Top intensity frames</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>intensity</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in top_rows])}
              </tbody>
            </table>
            ''' if top_rows else '<div class="muted">No data</div>'}

            <h3>Anti-top confidence frames</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>emotion_confidence</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in anti_rows])}
              </tbody>
            </table>
            ''' if anti_rows else '<div class="muted">No data</div>'}
        </div>
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Valence</th>
                        <th>Arousal</th>
                        <th>Intensity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('valence', 'mean')}</td>
                        <td>{format_dist_value('arousal', 'mean')}</td>
                        <td>{format_dist_value('intensity', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('valence', 'std')}</td>
                        <td>{format_dist_value('arousal', 'std')}</td>
                        <td>{format_dist_value('intensity', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('valence', 'min')}</td>
                        <td>{format_dist_value('arousal', 'min')}</td>
                        <td>{format_dist_value('intensity', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('valence', 'max')}</td>
                        <td>{format_dist_value('arousal', 'max')}</td>
                        <td>{format_dist_value('intensity', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('valence', 'median')}</td>
                        <td>{format_dist_value('arousal', 'median')}</td>
                        <td>{format_dist_value('intensity', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>P25</strong></td>
                        <td>{format_dist_value('valence', 'p25')}</td>
                        <td>{format_dist_value('arousal', 'p25')}</td>
                        <td>{format_dist_value('intensity', 'p25')}</td>
                    </tr>
                    <tr>
                        <td><strong>P75</strong></td>
                        <td>{format_dist_value('valence', 'p75')}</td>
                        <td>{format_dist_value('arousal', 'p75')}</td>
                        <td>{format_dist_value('intensity', 'p75')}</td>
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
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Emotion Face HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_emotion_face", "render_emotion_face_html"]

