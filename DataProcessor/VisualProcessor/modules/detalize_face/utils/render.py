"""
Renderer для detalize_face: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def _to_1d_float(arr: Any) -> np.ndarray:
    """Best-effort convert to 1D float32 numpy array (NaNs preserved if possible)."""
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
    """
    Minimal offline-friendly SVG line chart.
    - Handles NaNs by breaking the path into segments.
    - Downsamples to ~500 points for file size.
    """
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

    path_parts = []
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


def render_detalize_face(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для detalize_face."""
    render = {
        "component": "detalize_face",
        "key_facts": {},
        "summary": {},
        "timeline": [],
        "distributions": {},
        "faces": [],
    }

    # Key facts from meta (best-effort)
    if isinstance(meta, dict):
        st = meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {}
        render["key_facts"] = {
            "schema_version": meta.get("schema_version"),
            "producer_version": meta.get("producer_version"),
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "module_sampling_policy_version": meta.get("module_sampling_policy_version"),
            "stage_timings_ms": st,
        }
    
    # Extract face data
    face_present = npz_data.get("face_present")
    processed_mask = npz_data.get("processed_mask")
    primary_valid = npz_data.get("primary_valid")
    primary_tracking_id = npz_data.get("primary_tracking_id")
    primary_compact_features = npz_data.get("primary_compact_features")
    face_count = npz_data.get("face_count")
    primary_gaze = npz_data.get("primary_gaze_at_camera_prob")
    primary_blink_rate = npz_data.get("primary_blink_rate")
    primary_attention = npz_data.get("primary_attention_score")
    primary_quality = npz_data.get("primary_quality_proxy_score")
    primary_sharpness = npz_data.get("primary_face_sharpness")
    primary_occlusion = npz_data.get("primary_occlusion_proxy")
    primary_speech = npz_data.get("primary_speech_activity_prob")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    faces_agg = npz_data.get("faces_agg")
    summary = npz_data.get("summary")
    
    # Convert to numpy arrays if needed
    def _to_1d_bool(x: Any) -> Optional[np.ndarray]:
        try:
            if x is None:
                return None
            if isinstance(x, list):
                return np.asarray(x, dtype=bool).reshape(-1)
            if isinstance(x, np.ndarray):
                return np.asarray(x, dtype=bool).reshape(-1)
        except Exception:
            return None
        return None

    face_present = _to_1d_bool(face_present)
    processed_mask = _to_1d_bool(processed_mask)
    primary_valid = _to_1d_bool(primary_valid)

    if primary_tracking_id is not None:
        try:
            if isinstance(primary_tracking_id, list):
                primary_tracking_id = np.asarray(primary_tracking_id, dtype=np.int32).reshape(-1)
            elif isinstance(primary_tracking_id, np.ndarray):
                primary_tracking_id = np.asarray(primary_tracking_id, dtype=np.int32).reshape(-1)
            else:
                primary_tracking_id = None
        except Exception:
            primary_tracking_id = None

    if primary_compact_features is not None:
        try:
            if isinstance(primary_compact_features, list):
                primary_compact_features = np.asarray(primary_compact_features, dtype=np.float32)
            elif isinstance(primary_compact_features, np.ndarray):
                primary_compact_features = np.asarray(primary_compact_features, dtype=np.float32)
            else:
                primary_compact_features = None
        except Exception:
            primary_compact_features = None

    if face_count is not None:
        if isinstance(face_count, list):
            face_count = np.array(face_count, dtype=np.float32)
        elif isinstance(face_count, np.ndarray):
            face_count = np.asarray(face_count, dtype=np.float32)
        else:
            face_count = None
    
    if primary_gaze is not None:
        if isinstance(primary_gaze, list):
            primary_gaze = np.array(primary_gaze, dtype=np.float32)
        elif isinstance(primary_gaze, np.ndarray):
            primary_gaze = np.asarray(primary_gaze, dtype=np.float32)
        else:
            primary_gaze = None
    
    if primary_blink_rate is not None:
        if isinstance(primary_blink_rate, list):
            primary_blink_rate = np.array(primary_blink_rate, dtype=np.float32)
        elif isinstance(primary_blink_rate, np.ndarray):
            primary_blink_rate = np.asarray(primary_blink_rate, dtype=np.float32)
        else:
            primary_blink_rate = None
    
    if primary_attention is not None:
        if isinstance(primary_attention, list):
            primary_attention = np.array(primary_attention, dtype=np.float32)
        elif isinstance(primary_attention, np.ndarray):
            primary_attention = np.asarray(primary_attention, dtype=np.float32)
        else:
            primary_attention = None
    
    if primary_quality is not None:
        if isinstance(primary_quality, list):
            primary_quality = np.array(primary_quality, dtype=np.float32)
        elif isinstance(primary_quality, np.ndarray):
            primary_quality = np.asarray(primary_quality, dtype=np.float32)
        else:
            primary_quality = None
    
    if primary_sharpness is not None:
        if isinstance(primary_sharpness, list):
            primary_sharpness = np.array(primary_sharpness, dtype=np.float32)
        elif isinstance(primary_sharpness, np.ndarray):
            primary_sharpness = np.asarray(primary_sharpness, dtype=np.float32)
        else:
            primary_sharpness = None
    
    if primary_occlusion is not None:
        if isinstance(primary_occlusion, list):
            primary_occlusion = np.array(primary_occlusion, dtype=np.float32)
        elif isinstance(primary_occlusion, np.ndarray):
            primary_occlusion = np.asarray(primary_occlusion, dtype=np.float32)
        else:
            primary_occlusion = None
    
    if primary_speech is not None:
        if isinstance(primary_speech, list):
            primary_speech = np.array(primary_speech, dtype=np.float32)
        elif isinstance(primary_speech, np.ndarray):
            primary_speech = np.asarray(primary_speech, dtype=np.float32)
        else:
            primary_speech = None
    
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
    
    # Summary statistics
    if summary is not None:
        if isinstance(summary, np.ndarray) and summary.dtype == object:
            summary = summary.item() if summary.size == 1 else {}
        if isinstance(summary, dict):
            render["summary"] = {
                "axis_frames": int(summary.get("axis_frames", summary.get("total_frames", 0))),
                "processed_frames": int(summary.get("processed_frames", 0)),
                "frames_with_faces_total": int(summary.get("frames_with_faces_total", summary.get("frames_with_faces", 0))),
                "frames_with_faces_processed": int(summary.get("frames_with_faces_processed", 0)),
                "total_faces": int(summary.get("total_faces", 0)),
                "primary_faces": int(summary.get("primary_faces", 0)),
                "avg_faces_per_processed_face_frame": float(summary.get("avg_faces_per_processed_face_frame", summary.get("avg_faces_per_frame", 0.0))),
            }
    
    # Timeline data
    if times_s is not None and face_count is not None:
        n = len(times_s)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i
            time_sec = float(times_s[i]) if np.isfinite(times_s[i]) else 0.0
            fc = float(face_count[i]) if i < len(face_count) and np.isfinite(face_count[i]) else None
            fp = bool(face_present[i]) if face_present is not None and i < len(face_present) else None
            pm = bool(processed_mask[i]) if processed_mask is not None and i < len(processed_mask) else None
            pv = bool(primary_valid[i]) if primary_valid is not None and i < len(primary_valid) else None
            tid = int(primary_tracking_id[i]) if primary_tracking_id is not None and i < len(primary_tracking_id) else None
            c_norm = None
            if primary_compact_features is not None and isinstance(primary_compact_features, np.ndarray):
                try:
                    if primary_compact_features.ndim == 2 and i < primary_compact_features.shape[0]:
                        c_norm = float(np.linalg.norm(primary_compact_features[i, :]))
                except Exception:
                    c_norm = None
            gaze = float(primary_gaze[i]) if primary_gaze is not None and i < len(primary_gaze) and np.isfinite(primary_gaze[i]) else None
            blink = float(primary_blink_rate[i]) if primary_blink_rate is not None and i < len(primary_blink_rate) and np.isfinite(primary_blink_rate[i]) else None
            attn = float(primary_attention[i]) if primary_attention is not None and i < len(primary_attention) and np.isfinite(primary_attention[i]) else None
            qual = float(primary_quality[i]) if primary_quality is not None and i < len(primary_quality) and np.isfinite(primary_quality[i]) else None
            sharp = float(primary_sharpness[i]) if primary_sharpness is not None and i < len(primary_sharpness) and np.isfinite(primary_sharpness[i]) else None
            occ = float(primary_occlusion[i]) if primary_occlusion is not None and i < len(primary_occlusion) and np.isfinite(primary_occlusion[i]) else None
            speech = float(primary_speech[i]) if primary_speech is not None and i < len(primary_speech) and np.isfinite(primary_speech[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "face_present": fp,
                "processed_mask": pm,
                "primary_valid": pv,
                "primary_tracking_id": tid,
                "primary_compact_norm": c_norm,
                "face_count": fc,
                "primary_gaze_at_camera_prob": gaze,
                "primary_blink_rate": blink,
                "primary_attention_score": attn,
                "primary_quality_proxy_score": qual,
                "primary_face_sharpness": sharp,
                "primary_occlusion_proxy": occ,
                "primary_speech_activity_prob": speech,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if face_count is not None:
        valid_counts = face_count[np.isfinite(face_count)]
        if valid_counts.size > 0:
            distributions["face_count"] = {
                "min": float(np.min(valid_counts)),
                "max": float(np.max(valid_counts)),
                "mean": float(np.mean(valid_counts)),
                "std": float(np.std(valid_counts)),
                "median": float(np.median(valid_counts)),
                "p25": float(np.percentile(valid_counts, 25)),
                "p75": float(np.percentile(valid_counts, 75)),
            }
    
    if primary_gaze is not None:
        valid_gaze = primary_gaze[np.isfinite(primary_gaze)]
        if valid_gaze.size > 0:
            distributions["primary_gaze_at_camera_prob"] = {
                "min": float(np.min(valid_gaze)),
                "max": float(np.max(valid_gaze)),
                "mean": float(np.mean(valid_gaze)),
                "std": float(np.std(valid_gaze)),
                "median": float(np.median(valid_gaze)),
                "p25": float(np.percentile(valid_gaze, 25)),
                "p75": float(np.percentile(valid_gaze, 75)),
            }
    
    if primary_blink_rate is not None:
        valid_blink = primary_blink_rate[np.isfinite(primary_blink_rate)]
        if valid_blink.size > 0:
            distributions["primary_blink_rate"] = {
                "min": float(np.min(valid_blink)),
                "max": float(np.max(valid_blink)),
                "mean": float(np.mean(valid_blink)),
                "std": float(np.std(valid_blink)),
                "median": float(np.median(valid_blink)),
                "p25": float(np.percentile(valid_blink, 25)),
                "p75": float(np.percentile(valid_blink, 75)),
            }
    
    if primary_attention is not None:
        valid_attn = primary_attention[np.isfinite(primary_attention)]
        if valid_attn.size > 0:
            distributions["primary_attention_score"] = {
                "min": float(np.min(valid_attn)),
                "max": float(np.max(valid_attn)),
                "mean": float(np.mean(valid_attn)),
                "std": float(np.std(valid_attn)),
                "median": float(np.median(valid_attn)),
                "p25": float(np.percentile(valid_attn, 25)),
                "p75": float(np.percentile(valid_attn, 75)),
            }
    
    if primary_quality is not None:
        valid_qual = primary_quality[np.isfinite(primary_quality)]
        if valid_qual.size > 0:
            distributions["primary_quality_proxy_score"] = {
                "min": float(np.min(valid_qual)),
                "max": float(np.max(valid_qual)),
                "mean": float(np.mean(valid_qual)),
                "std": float(np.std(valid_qual)),
                "median": float(np.median(valid_qual)),
                "p25": float(np.percentile(valid_qual, 25)),
                "p75": float(np.percentile(valid_qual, 75)),
            }
    
    if primary_sharpness is not None:
        valid_sharp = primary_sharpness[np.isfinite(primary_sharpness)]
        if valid_sharp.size > 0:
            distributions["primary_face_sharpness"] = {
                "min": float(np.min(valid_sharp)),
                "max": float(np.max(valid_sharp)),
                "mean": float(np.mean(valid_sharp)),
                "std": float(np.std(valid_sharp)),
                "median": float(np.median(valid_sharp)),
                "p25": float(np.percentile(valid_sharp, 25)),
                "p75": float(np.percentile(valid_sharp, 75)),
            }
    
    if primary_occlusion is not None:
        valid_occ = primary_occlusion[np.isfinite(primary_occlusion)]
        if valid_occ.size > 0:
            distributions["primary_occlusion_proxy"] = {
                "min": float(np.min(valid_occ)),
                "max": float(np.max(valid_occ)),
                "mean": float(np.mean(valid_occ)),
                "std": float(np.std(valid_occ)),
                "median": float(np.median(valid_occ)),
                "p25": float(np.percentile(valid_occ, 25)),
                "p75": float(np.percentile(valid_occ, 75)),
            }
    
    if primary_speech is not None:
        valid_speech = primary_speech[np.isfinite(primary_speech)]
        if valid_speech.size > 0:
            distributions["primary_speech_activity_prob"] = {
                "min": float(np.min(valid_speech)),
                "max": float(np.max(valid_speech)),
                "mean": float(np.mean(valid_speech)),
                "std": float(np.std(valid_speech)),
                "median": float(np.median(valid_speech)),
                "p25": float(np.percentile(valid_speech, 25)),
                "p75": float(np.percentile(valid_speech, 75)),
            }
    
    render["distributions"] = distributions
    
    # Faces aggregate info
    if faces_agg is not None:
        if isinstance(faces_agg, np.ndarray) and faces_agg.dtype == object:
            faces_agg = faces_agg.item() if faces_agg.size == 1 else {}
        if isinstance(faces_agg, dict):
            render["faces"] = [
                {
                    "tracking_id": int(track_id),
                    "frames_count": int(agg.get("frames_count", 0)) if isinstance(agg, dict) else 0,
                }
                for track_id, agg in faces_agg.items()
            ]
    
    return render


def render_detalize_face_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага detalize_face результатов.
    
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
    render = render_detalize_face(npz_data, meta)
    
    timeline = render.get("timeline", [])
    key_facts = render.get("key_facts", {})
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    faces = render.get("faces", [])
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Prepare SVG charts (offline, no CDN / no JS)
    charts_html = ""
    top_rows = []
    anti_rows = []
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]

        def _col(name: str):
            return [t.get(name) for t in timeline]

        charts_html = "\n".join(
            [
                _svg_line_chart(times_s=times, values=_col("face_present"), title="face_present (mask)", stroke="#111827"),
                _svg_line_chart(times_s=times, values=_col("processed_mask"), title="processed_mask (mask)", stroke="#374151"),
                _svg_line_chart(times_s=times, values=_col("primary_valid"), title="primary_valid (mask)", stroke="#6b7280"),
                _svg_line_chart(times_s=times, values=_col("face_count"), title="face_count", stroke="#0ea5e9"),
                _svg_line_chart(times_s=times, values=_col("primary_compact_norm"), title="primary_compact_norm (L2)", stroke="#7c3aed"),
                _svg_line_chart(times_s=times, values=_col("primary_attention_score"), title="primary_attention_score", stroke="#f59e0b"),
                _svg_line_chart(times_s=times, values=_col("primary_quality_proxy_score"), title="primary_quality_proxy_score", stroke="#22c55e"),
                _svg_line_chart(times_s=times, values=_col("primary_occlusion_proxy"), title="primary_occlusion_proxy", stroke="#ef4444"),
                _svg_line_chart(times_s=times, values=_col("primary_gaze_at_camera_prob"), title="primary_gaze_at_camera_prob", stroke="#6366f1"),
                _svg_line_chart(times_s=times, values=_col("primary_blink_rate"), title="primary_blink_rate", stroke="#14b8a6"),
                _svg_line_chart(times_s=times, values=_col("primary_speech_activity_prob"), title="primary_speech_activity_prob", stroke="#a855f7"),
            ]
        )

        def _safe_float(v):
            try:
                if v is None:
                    return None
                fv = float(v)
                return fv if np.isfinite(fv) else None
            except Exception:
                return None

        scored_attn = []
        scored_qual = []
        for t in timeline:
            fi = int(t.get("frame_index", -1))
            ts = float(t.get("time_sec", 0.0))
            a = _safe_float(t.get("primary_attention_score"))
            q = _safe_float(t.get("primary_quality_proxy_score"))
            if a is not None:
                scored_attn.append((a, fi, ts))
            if q is not None:
                scored_qual.append((q, fi, ts))

        scored_attn.sort(key=lambda x: x[0], reverse=True)
        scored_qual.sort(key=lambda x: x[0])

        for a, fi, ts in scored_attn[:5]:
            top_rows.append((fi, ts, a))
        for q, fi, ts in scored_qual[:5]:
            anti_rows.append((fi, ts, q))
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detalize Face Debug Render</title>
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
        .faces-list {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .examples table {{ width: 100%; border-collapse: collapse; }}
        .examples th, .examples td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .examples th {{ background-color: #111827; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Detalize Face Debug Render</h1>

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
                    <strong>Total Frames</strong>
                    <span class="metric-value">{summary.get('total_frames', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Processed Frames</strong>
                    <span class="metric-value">{summary.get('processed_frames', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Frames with Faces</strong>
                    <span class="metric-value">{summary.get('frames_with_faces', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Total Faces</strong>
                    <span class="metric-value">{summary.get('total_faces', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Primary Faces</strong>
                    <span class="metric-value">{summary.get('primary_faces', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Faces per Frame</strong>
                    <span class="metric-value">{summary.get('avg_faces_per_frame', 0.0):.2f}</span>
                </div>
            </div>
        </div>
        
        <h2>Timeline charts (offline)</h2>
        {f'<div class="charts">{charts_html}</div>' if timeline else '<p class="muted">No timeline data available</p>'}

        <div class="examples">
            <h2>Top / Anti-top examples</h2>
            <h3>Top attention frames (primary_attention_score)</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>score</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in top_rows])}
              </tbody>
            </table>
            ''' if top_rows else '<div class="muted">No data</div>'}

            <h3>Anti-top quality frames (primary_quality_proxy_score)</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>score</th></tr></thead>
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
                        <th>Face Count</th>
                        <th>Gaze at Camera</th>
                        <th>Blink Rate</th>
                        <th>Attention Score</th>
                        <th>Quality Score</th>
                        <th>Sharpness</th>
                        <th>Occlusion</th>
                        <th>Speech Activity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('face_count', 'mean')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'mean')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'mean')}</td>
                        <td>{format_dist_value('primary_attention_score', 'mean')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'mean')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'mean')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'mean')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('face_count', 'std')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'std')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'std')}</td>
                        <td>{format_dist_value('primary_attention_score', 'std')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'std')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'std')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'std')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('face_count', 'min')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'min')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'min')}</td>
                        <td>{format_dist_value('primary_attention_score', 'min')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'min')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'min')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'min')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('face_count', 'max')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'max')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'max')}</td>
                        <td>{format_dist_value('primary_attention_score', 'max')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'max')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'max')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'max')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'max')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
        
        {f'''
        <div class="faces-list">
            <h2>Tracked Faces</h2>
            <p>Total tracks: {len(faces)}</p>
            <ul>
                {''.join([f'<li>Track {f.get("tracking_id", 0)}: {f.get("frames_count", 0)} frames</li>' for f in faces[:10]])}
                {f'<li>... and {len(faces) - 10} more tracks</li>' if len(faces) > 10 else ''}
            </ul>
        </div>
        ''' if faces else ''}
    </div>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved Detalize Face HTML render to {output_path}")
    return output_path


__all__ = ["render_detalize_face", "render_detalize_face_html"]

