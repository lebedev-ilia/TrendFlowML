"""
Renderer для scene_classification: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

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
    max_points = 600
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


def render_scene_classification(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для scene_classification."""
    render = {
        "component": "scene_classification",
        "key_facts": {},
        "config_highlights": {},
        "summary": {},
        "timeline": [],
        "distributions": {},
        "scenes": [],
    }

    if isinstance(meta, dict):
        render["key_facts"] = {
            "schema_version": meta.get("schema_version"),
            "producer_version": meta.get("producer_version"),
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        }
        render["config_highlights"] = {
            "label_fusion": meta.get("label_fusion"),
            "min_scene_seconds": meta.get("min_scene_seconds"),
            "runtime": meta.get("runtime"),
            "model_arch": meta.get("model_arch"),
            "input_size": meta.get("input_size"),
            "batch_size": meta.get("batch_size"),
            "temporal_smoothing": meta.get("temporal_smoothing"),
            "smoothing_window": meta.get("smoothing_window"),
            "enable_advanced_features": meta.get("enable_advanced_features"),
        }
    
    # Extract data
    frame_indices = npz_data.get("frame_indices")
    times_s = npz_data.get("times_s")
    scene_ids = npz_data.get("scene_ids")
    scene_label = npz_data.get("scene_label")
    frame_scene_id = npz_data.get("frame_scene_id")
    frame_topk_ids = npz_data.get("frame_topk_ids")
    frame_topk_probs = npz_data.get("frame_topk_probs")
    frame_entropy = npz_data.get("frame_entropy")
    frame_top1_prob = npz_data.get("frame_top1_prob")
    frame_top1_top2_gap = npz_data.get("frame_top1_top2_gap")
    start_frame = npz_data.get("start_frame")
    end_frame = npz_data.get("end_frame")
    start_time_s = npz_data.get("start_time_s")
    end_time_s = npz_data.get("end_time_s")
    length_seconds = npz_data.get("length_seconds")
    mean_score = npz_data.get("mean_score")
    class_entropy_mean = npz_data.get("class_entropy_mean")
    top1_prob_mean = npz_data.get("top1_prob_mean")
    mean_aesthetic_score = npz_data.get("mean_aesthetic_score")
    mean_luxury_score = npz_data.get("mean_luxury_score")
    scenes_raw = npz_data.get("scenes") or npz_data.get("scenes_raw")
    
    # Convert to numpy arrays if needed
    def to_array(data, dtype=None):
        if data is None:
            return None
        if isinstance(data, list):
            return np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            return np.asarray(data, dtype=dtype) if dtype else data
        elif isinstance(data, np.ndarray) and data.dtype == object:
            return data
        return None
    
    frame_indices = to_array(frame_indices, np.int32)
    times_s = to_array(times_s, np.float32)
    scene_ids = to_array(scene_ids, object) if scene_ids is not None else None
    scene_label = to_array(scene_label, object) if scene_label is not None else None
    frame_scene_id = to_array(frame_scene_id, np.int32) if frame_scene_id is not None else None
    frame_topk_ids = to_array(frame_topk_ids, np.int32) if frame_topk_ids is not None else None
    frame_topk_probs = to_array(frame_topk_probs, np.float32) if frame_topk_probs is not None else None
    frame_entropy = to_array(frame_entropy, np.float32) if frame_entropy is not None else None
    frame_top1_prob = to_array(frame_top1_prob, np.float32) if frame_top1_prob is not None else None
    frame_top1_top2_gap = to_array(frame_top1_top2_gap, np.float32) if frame_top1_top2_gap is not None else None
    start_frame = to_array(start_frame, np.int32) if start_frame is not None else None
    end_frame = to_array(end_frame, np.int32) if end_frame is not None else None
    start_time_s = to_array(start_time_s, np.float32) if start_time_s is not None else None
    end_time_s = to_array(end_time_s, np.float32) if end_time_s is not None else None
    length_seconds = to_array(length_seconds, np.float32) if length_seconds is not None else None
    mean_score = to_array(mean_score, np.float32) if mean_score is not None else None
    class_entropy_mean = to_array(class_entropy_mean, np.float32) if class_entropy_mean is not None else None
    top1_prob_mean = to_array(top1_prob_mean, np.float32) if top1_prob_mean is not None else None
    mean_aesthetic_score = to_array(mean_aesthetic_score, np.float32) if mean_aesthetic_score is not None else None
    mean_luxury_score = to_array(mean_luxury_score, np.float32) if mean_luxury_score is not None else None
    
    # Summary statistics
    if frame_indices is not None and frame_indices.size > 0:
        n_frames = len(frame_indices)
        n_scenes = len(scene_ids) if scene_ids is not None and scene_ids.size > 0 else 0
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "scenes_count": int(n_scenes),
        }
        
        if n_scenes > 0 and length_seconds is not None and length_seconds.size > 0:
            valid_lengths = length_seconds[np.isfinite(length_seconds)]
            if valid_lengths.size > 0:
                render["summary"]["avg_scene_length_seconds"] = float(np.mean(valid_lengths))
                render["summary"]["min_scene_length_seconds"] = float(np.min(valid_lengths))
                render["summary"]["max_scene_length_seconds"] = float(np.max(valid_lengths))
        
        if frame_top1_prob is not None and frame_top1_prob.size > 0:
            valid_probs = frame_top1_prob[np.isfinite(frame_top1_prob)]
            if valid_probs.size > 0:
                render["summary"]["top1_prob_mean"] = float(np.mean(valid_probs))
                render["summary"]["top1_prob_std"] = float(np.std(valid_probs))
                render["summary"]["top1_prob_min"] = float(np.min(valid_probs))
                render["summary"]["top1_prob_max"] = float(np.max(valid_probs))
        
        if frame_entropy is not None and frame_entropy.size > 0:
            valid_entropy = frame_entropy[np.isfinite(frame_entropy)]
            if valid_entropy.size > 0:
                render["summary"]["entropy_mean"] = float(np.mean(valid_entropy))
                render["summary"]["entropy_std"] = float(np.std(valid_entropy))
        
        if mean_aesthetic_score is not None and mean_aesthetic_score.size > 0:
            valid_aesthetic = mean_aesthetic_score[np.isfinite(mean_aesthetic_score)]
            if valid_aesthetic.size > 0:
                render["summary"]["aesthetic_score_mean"] = float(np.mean(valid_aesthetic))
                render["summary"]["aesthetic_score_std"] = float(np.std(valid_aesthetic))
        
        if mean_luxury_score is not None and mean_luxury_score.size > 0:
            valid_luxury = mean_luxury_score[np.isfinite(mean_luxury_score)]
            if valid_luxury.size > 0:
                render["summary"]["luxury_score_mean"] = float(np.mean(valid_luxury))
                render["summary"]["luxury_score_std"] = float(np.std(valid_luxury))
    
    # Timeline data (per-frame)
    if frame_indices is not None and times_s is not None and len(frame_indices) == len(times_s):
        timeline = []
        for i in range(len(frame_indices)):
            frame_idx = int(frame_indices[i])
            time_sec = float(times_s[i])
            
            timeline_entry = {
                "frame_index": frame_idx,
                "time_sec": time_sec,
            }
            
            if frame_scene_id is not None and i < len(frame_scene_id):
                timeline_entry["scene_id"] = str(scene_ids[frame_scene_id[i]]) if scene_ids is not None and frame_scene_id[i] < len(scene_ids) else None
                timeline_entry["scene_label"] = str(scene_label[frame_scene_id[i]]) if scene_label is not None and frame_scene_id[i] < len(scene_label) else None
            
            if frame_top1_prob is not None and i < len(frame_top1_prob):
                timeline_entry["top1_prob"] = float(frame_top1_prob[i]) if np.isfinite(frame_top1_prob[i]) else None
            
            if frame_entropy is not None and i < len(frame_entropy):
                timeline_entry["entropy"] = float(frame_entropy[i]) if np.isfinite(frame_entropy[i]) else None
            
            if frame_top1_top2_gap is not None and i < len(frame_top1_top2_gap):
                timeline_entry["top1_top2_gap"] = float(frame_top1_top2_gap[i]) if np.isfinite(frame_top1_top2_gap[i]) else None
            
            if frame_topk_ids is not None and i < len(frame_topk_ids):
                topk_ids = frame_topk_ids[i]
                if isinstance(topk_ids, np.ndarray):
                    timeline_entry["topk_ids"] = [int(x) for x in topk_ids.tolist() if np.isfinite(x)]
                else:
                    timeline_entry["topk_ids"] = []
            
            if frame_topk_probs is not None and i < len(frame_topk_probs):
                topk_probs = frame_topk_probs[i]
                if isinstance(topk_probs, np.ndarray):
                    timeline_entry["topk_probs"] = [float(x) for x in topk_probs.tolist() if np.isfinite(x)]
                else:
                    timeline_entry["topk_probs"] = []
            
            timeline.append(timeline_entry)
        
        render["timeline"] = timeline
    
    # Scenes data
    if scene_ids is not None and scene_ids.size > 0:
        scenes_list = []
        for i in range(len(scene_ids)):
            scene_id = str(scene_ids[i]) if isinstance(scene_ids[i], (str, np.str_)) else f"s{i:04d}"
            scene_data = {
                "scene_id": scene_id,
                "scene_label": str(scene_label[i]) if scene_label is not None and i < len(scene_label) else "",
            }
            
            if start_frame is not None and i < len(start_frame):
                scene_data["start_frame"] = int(start_frame[i])
            if end_frame is not None and i < len(end_frame):
                scene_data["end_frame"] = int(end_frame[i])
            if start_time_s is not None and i < len(start_time_s):
                scene_data["start_time_s"] = float(start_time_s[i]) if np.isfinite(start_time_s[i]) else None
            if end_time_s is not None and i < len(end_time_s):
                scene_data["end_time_s"] = float(end_time_s[i]) if np.isfinite(end_time_s[i]) else None
            if length_seconds is not None and i < len(length_seconds):
                scene_data["length_seconds"] = float(length_seconds[i]) if np.isfinite(length_seconds[i]) else None
            
            if mean_score is not None and i < len(mean_score):
                scene_data["mean_score"] = float(mean_score[i]) if np.isfinite(mean_score[i]) else None
            if class_entropy_mean is not None and i < len(class_entropy_mean):
                scene_data["entropy_mean"] = float(class_entropy_mean[i]) if np.isfinite(class_entropy_mean[i]) else None
            if top1_prob_mean is not None and i < len(top1_prob_mean):
                scene_data["top1_prob_mean"] = float(top1_prob_mean[i]) if np.isfinite(top1_prob_mean[i]) else None
            
            if mean_aesthetic_score is not None and i < len(mean_aesthetic_score):
                scene_data["aesthetic_score"] = float(mean_aesthetic_score[i]) if np.isfinite(mean_aesthetic_score[i]) else None
            if mean_luxury_score is not None and i < len(mean_luxury_score):
                scene_data["luxury_score"] = float(mean_luxury_score[i]) if np.isfinite(mean_luxury_score[i]) else None
            
            scenes_list.append(scene_data)
        
        render["scenes"] = scenes_list
    
    # Distributions
    distributions = {}
    
    if frame_top1_prob is not None and frame_top1_prob.size > 0:
        valid_probs = frame_top1_prob[np.isfinite(frame_top1_prob)]
        if valid_probs.size > 0:
            distributions["top1_prob"] = {
                "min": float(np.min(valid_probs)),
                "max": float(np.max(valid_probs)),
                "mean": float(np.mean(valid_probs)),
                "std": float(np.std(valid_probs)),
                "median": float(np.median(valid_probs)),
                "p25": float(np.percentile(valid_probs, 25)),
                "p75": float(np.percentile(valid_probs, 75)),
                "p05": float(np.percentile(valid_probs, 5)),
                "p95": float(np.percentile(valid_probs, 95)),
            }
    
    if frame_entropy is not None and frame_entropy.size > 0:
        valid_entropy = frame_entropy[np.isfinite(frame_entropy)]
        if valid_entropy.size > 0:
            distributions["entropy"] = {
                "min": float(np.min(valid_entropy)),
                "max": float(np.max(valid_entropy)),
                "mean": float(np.mean(valid_entropy)),
                "std": float(np.std(valid_entropy)),
                "median": float(np.median(valid_entropy)),
                "p25": float(np.percentile(valid_entropy, 25)),
                "p75": float(np.percentile(valid_entropy, 75)),
                "p05": float(np.percentile(valid_entropy, 5)),
                "p95": float(np.percentile(valid_entropy, 95)),
            }
    
    if length_seconds is not None and length_seconds.size > 0:
        valid_lengths = length_seconds[np.isfinite(length_seconds)]
        if valid_lengths.size > 0:
            distributions["scene_length_seconds"] = {
                "min": float(np.min(valid_lengths)),
                "max": float(np.max(valid_lengths)),
                "mean": float(np.mean(valid_lengths)),
                "std": float(np.std(valid_lengths)),
                "median": float(np.median(valid_lengths)),
                "p25": float(np.percentile(valid_lengths, 25)),
                "p75": float(np.percentile(valid_lengths, 75)),
                "p05": float(np.percentile(valid_lengths, 5)),
                "p95": float(np.percentile(valid_lengths, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_scene_classification_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага scene_classification результатов.
    
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
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))
    
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
    render = render_scene_classification(npz_data, meta)
    
    timeline = render.get("timeline", [])
    key_facts = render.get("key_facts", {})
    config_h = render.get("config_highlights", {})
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    scenes = render.get("scenes", [])
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    charts_html = ""
    top_entropy: List[Tuple[int, float, float]] = []
    anti_conf: List[Tuple[int, float, float]] = []
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        ent = [t.get("entropy") for t in timeline]
        top1 = [t.get("top1_prob") for t in timeline]
        gap = [t.get("top1_top2_gap") for t in timeline]

        charts_html = "\n".join(
            [
                _svg_line_chart(times_s=times, values=ent, title="frame_entropy", stroke="#ef4444"),
                _svg_line_chart(times_s=times, values=top1, title="frame_top1_prob", stroke="#22c55e"),
                _svg_line_chart(times_s=times, values=gap, title="frame_top1_top2_gap", stroke="#0ea5e9"),
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

        ent_scored: List[Tuple[float, int, float]] = []
        prob_scored: List[Tuple[float, int, float]] = []
        for t in timeline:
            fi = int(t.get("frame_index", -1))
            ts = float(t.get("time_sec", 0.0))
            e = _safe_float(t.get("entropy"))
            p = _safe_float(t.get("top1_prob"))
            if e is not None:
                ent_scored.append((e, fi, ts))
            if p is not None:
                prob_scored.append((p, fi, ts))
        ent_scored.sort(key=lambda x: x[0], reverse=True)
        prob_scored.sort(key=lambda x: x[0])
        for e, fi, ts in ent_scored[:8]:
            top_entropy.append((fi, ts, e))
        for p, fi, ts in prob_scored[:8]:
            anti_conf.append((fi, ts, p))
    
    # Prepare scenes table
    scenes_table_html = ""
    if scenes:
        scenes_rows = []
        for scene in scenes:
            scene_id = scene.get("scene_id", "")
            scene_label = scene.get("scene_label", "")
            start_time = scene.get("start_time_s")
            end_time = scene.get("end_time_s")
            length = scene.get("length_seconds")
            top1_prob = scene.get("top1_prob_mean")
            entropy = scene.get("entropy_mean")
            
            # Format values safely
            start_time_str = f"{start_time:.2f}s" if start_time is not None and isinstance(start_time, (int, float)) else "N/A"
            end_time_str = f"{end_time:.2f}s" if end_time is not None and isinstance(end_time, (int, float)) else "N/A"
            length_str = f"{length:.2f}s" if length is not None and isinstance(length, (int, float)) else "N/A"
            top1_prob_str = f"{top1_prob:.4f}" if top1_prob is not None and isinstance(top1_prob, (int, float)) else "N/A"
            entropy_str = f"{entropy:.4f}" if entropy is not None and isinstance(entropy, (int, float)) else "N/A"
            
            scenes_rows.append(f"""
                <tr>
                    <td>{scene_id}</td>
                    <td>{scene_label}</td>
                    <td>{start_time_str}</td>
                    <td>{end_time_str}</td>
                    <td>{length_str}</td>
                    <td>{top1_prob_str}</td>
                    <td>{entropy_str}</td>
                </tr>
            """)
        
        scenes_table_html = f"""
        <h2>Scenes</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Scene ID</th>
                    <th>Label</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Length</th>
                    <th>Top-1 Prob Mean</th>
                    <th>Entropy Mean</th>
                </tr>
            </thead>
            <tbody>
                {''.join(scenes_rows)}
            </tbody>
        </table>
        """
    
    # Format summary values safely
    def safe_format(value, default=0, format_str="{:.2f}"):
        if value is None or not isinstance(value, (int, float)):
            return str(default)
        try:
            return format_str.format(float(value))
        except (ValueError, TypeError):
            return str(default)
    
    avg_scene_length_val = summary.get('avg_scene_length_seconds', 0)
    avg_scene_length_str = safe_format(avg_scene_length_val, 0, "{:.2f}") + "s"
    
    top1_prob_mean_val = summary.get('top1_prob_mean', 0)
    top1_prob_mean_str = safe_format(top1_prob_mean_val, 0, "{:.4f}")
    
    entropy_mean_val = summary.get('entropy_mean', 0)
    entropy_mean_str = safe_format(entropy_mean_val, 0, "{:.4f}")
    
    aesthetic_score_mean_val = summary.get('aesthetic_score_mean', 0)
    aesthetic_score_mean_str = safe_format(aesthetic_score_mean_val, 0, "{:.4f}")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scene Classification Debug</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            font-weight: normal;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .chart-container {{
            margin: 30px 0;
            position: relative;
            height: 400px;
        }}
        .muted {{ color: #6b7280; }}
        .keyfacts {{ background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin: 16px 0; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin: 16px 0; }}
        .chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
        .chart-title {{ font-weight: 600; margin-bottom: 6px; }}
        .examples table {{ width: 100%; border-collapse: collapse; }}
        .examples th, .examples td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .examples th {{ background-color: #111827; color: white; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th,
        .data-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .data-table th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        .data-table tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Scene Classification Debug</h1>

        <div class="keyfacts">
            <h2>Key facts</h2>
            <div class="muted">{json.dumps(key_facts, ensure_ascii=False)}</div>
        </div>

        <div class="keyfacts">
            <h2>Config highlights</h2>
            <div class="muted">{json.dumps(config_h, ensure_ascii=False)}</div>
        </div>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Frames Count</h3>
                <div class="value">{summary.get('frames_count', 'N/A')}</div>
            </div>
            <div class="summary-card">
                <h3>Scenes Count</h3>
                <div class="value">{summary.get('scenes_count', 'N/A')}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Scene Length</h3>
                <div class="value">{avg_scene_length_str}</div>
            </div>
            <div class="summary-card">
                <h3>Top-1 Prob Mean</h3>
                <div class="value">{top1_prob_mean_str}</div>
            </div>
            <div class="summary-card">
                <h3>Entropy Mean</h3>
                <div class="value">{entropy_mean_str}</div>
            </div>
            <div class="summary-card">
                <h3>Aesthetic Score Mean</h3>
                <div class="value">{aesthetic_score_mean_str}</div>
            </div>
        </div>
        
        <h2>Timeline (offline)</h2>
        {f'<div class="charts">{charts_html}</div>' if timeline else '<div class="muted">No timeline data available</div>'}

        <div class="examples">
            <h2>Top / Anti-top examples</h2>
            <h3>Top entropy frames (most uncertain)</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>entropy</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in top_entropy])}
              </tbody>
            </table>
            ''' if top_entropy else '<div class="muted">No data</div>'}

            <h3>Anti-top top1_prob frames (least confident)</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>top1_prob</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in anti_conf])}
              </tbody>
            </table>
            ''' if anti_conf else '<div class="muted">No data</div>'}
        </div>
        
        {scenes_table_html}
        
        <h2>Distributions</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Top-1 Prob</h3>
                <div>Min: {format_dist_value('top1_prob', 'min')}</div>
                <div>Max: {format_dist_value('top1_prob', 'max')}</div>
                <div>Mean: {format_dist_value('top1_prob', 'mean')}</div>
                <div>Std: {format_dist_value('top1_prob', 'std')}</div>
            </div>
            <div class="summary-card">
                <h3>Entropy</h3>
                <div>Min: {format_dist_value('entropy', 'min')}</div>
                <div>Max: {format_dist_value('entropy', 'max')}</div>
                <div>Mean: {format_dist_value('entropy', 'mean')}</div>
                <div>Std: {format_dist_value('entropy', 'std')}</div>
            </div>
            <div class="summary-card">
                <h3>Scene Length</h3>
                <div>Min: {format_dist_value('scene_length_seconds', 'min')}</div>
                <div>Max: {format_dist_value('scene_length_seconds', 'max')}</div>
                <div>Mean: {format_dist_value('scene_length_seconds', 'mean')}</div>
                <div>Std: {format_dist_value('scene_length_seconds', 'std')}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


__all__ = ["render_scene_classification", "render_scene_classification_html"]

