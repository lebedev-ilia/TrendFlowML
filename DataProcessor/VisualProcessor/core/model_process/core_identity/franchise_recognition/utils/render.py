"""
Renderer для franchise_recognition: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_franchise_recognition(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для franchise_recognition."""
    render = {
        "component": "franchise_recognition",
        "summary": {},
        "key_facts": {},
        "config_highlights": {},
        "qa_hints": {},
        "timeline": [],
        "distributions": {},
        "top_examples": [],
        "anti_top_examples": [],
    }
    
    # Extract franchise recognition data
    frame_topk_ids = npz_data.get("frame_topk_ids")
    frame_topk_scores = npz_data.get("frame_topk_scores")
    frame_is_confident_top1 = npz_data.get("frame_is_confident_top1")
    semantic_label_names = npz_data.get("semantic_label_names")
    track_topk_ids = npz_data.get("track_topk_ids")
    track_topk_scores = npz_data.get("track_topk_scores")
    track_is_confident_top1 = npz_data.get("track_is_confident_top1")
    track_topk_evidence_frame_indices = npz_data.get("track_topk_evidence_frame_indices")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if frame_topk_ids is not None:
        if isinstance(frame_topk_ids, list):
            frame_topk_ids = np.array(frame_topk_ids, dtype=np.int32)
        elif isinstance(frame_topk_ids, np.ndarray):
            frame_topk_ids = np.asarray(frame_topk_ids, dtype=np.int32)
        else:
            frame_topk_ids = None
    else:
        frame_topk_ids = None
    
    if frame_topk_scores is not None:
        if isinstance(frame_topk_scores, list):
            frame_topk_scores = np.array(frame_topk_scores, dtype=np.float32)
        elif isinstance(frame_topk_scores, np.ndarray):
            frame_topk_scores = np.asarray(frame_topk_scores, dtype=np.float32)
        else:
            frame_topk_scores = None
    else:
        frame_topk_scores = None
    
    if frame_is_confident_top1 is not None:
        if isinstance(frame_is_confident_top1, list):
            frame_is_confident_top1 = np.array(frame_is_confident_top1, dtype=np.bool_)
        elif isinstance(frame_is_confident_top1, np.ndarray):
            frame_is_confident_top1 = np.asarray(frame_is_confident_top1, dtype=np.bool_)
        else:
            frame_is_confident_top1 = None
    else:
        frame_is_confident_top1 = None
    
    if semantic_label_names is not None:
        if isinstance(semantic_label_names, list):
            semantic_label_names = np.array(semantic_label_names, dtype="U")
        elif isinstance(semantic_label_names, np.ndarray):
            semantic_label_names = np.asarray(semantic_label_names, dtype="U")
        else:
            semantic_label_names = None
    else:
        semantic_label_names = None
    
    if times_s is not None:
        if isinstance(times_s, list):
            times_s = np.array(times_s, dtype=np.float32)
        elif isinstance(times_s, np.ndarray):
            times_s = np.asarray(times_s, dtype=np.float32)
        else:
            times_s = None
    else:
        times_s = None
    
    if frame_indices is not None:
        if isinstance(frame_indices, list):
            frame_indices = np.array(frame_indices, dtype=np.int32)
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = np.asarray(frame_indices, dtype=np.int32)
        else:
            frame_indices = None
    else:
        frame_indices = None
    
    # Convert track arrays
    if track_topk_ids is not None:
        if isinstance(track_topk_ids, list):
            track_topk_ids = np.array(track_topk_ids, dtype=np.int32)
        elif isinstance(track_topk_ids, np.ndarray):
            track_topk_ids = np.asarray(track_topk_ids, dtype=np.int32)
        else:
            track_topk_ids = None
    else:
        track_topk_ids = None
    
    if track_topk_scores is not None:
        if isinstance(track_topk_scores, list):
            track_topk_scores = np.array(track_topk_scores, dtype=np.float32)
        elif isinstance(track_topk_scores, np.ndarray):
            track_topk_scores = np.asarray(track_topk_scores, dtype=np.float32)
        else:
            track_topk_scores = None
    else:
        track_topk_scores = None
    
    if track_is_confident_top1 is not None:
        if isinstance(track_is_confident_top1, list):
            track_is_confident_top1 = np.array(track_is_confident_top1, dtype=np.bool_)
        elif isinstance(track_is_confident_top1, np.ndarray):
            track_is_confident_top1 = np.asarray(track_is_confident_top1, dtype=np.bool_)
        else:
            track_is_confident_top1 = None
    else:
        track_is_confident_top1 = None
    
    # Summary statistics
    if frame_topk_ids is not None and frame_topk_ids.size > 0:
        n_frames = frame_topk_ids.shape[0] if frame_topk_ids.ndim >= 2 else 1
        topk = frame_topk_ids.shape[1] if frame_topk_ids.ndim >= 2 else 1
        
        # Extract top-1 franchise for each frame
        top1_ids = frame_topk_ids[:, 0] if frame_topk_ids.ndim >= 2 else frame_topk_ids
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores is not None and frame_topk_scores.ndim >= 2 else None
        
        # Count unique franchises
        unique_franchises = np.unique(top1_ids)
        unique_franchises = unique_franchises[unique_franchises >= 0]  # Filter out -1 (invalid)
        
        # Count confident predictions
        confident_count = int(np.sum(frame_is_confident_top1)) if frame_is_confident_top1 is not None else 0
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "topk": int(topk),
            "unique_franchises_count": int(len(unique_franchises)),
            "confident_predictions_count": confident_count,
            "confident_predictions_ratio": float(confident_count / n_frames) if n_frames > 0 else 0.0,
        }
        
        if top1_scores is not None and top1_scores.size > 0:
            valid_scores = top1_scores[np.isfinite(top1_scores)]
            if valid_scores.size > 0:
                render["summary"]["top1_score_mean"] = float(np.mean(valid_scores))
                render["summary"]["top1_score_std"] = float(np.std(valid_scores))
                render["summary"]["top1_score_min"] = float(np.min(valid_scores))
                render["summary"]["top1_score_max"] = float(np.max(valid_scores))
                render["summary"]["top1_score_median"] = float(np.median(valid_scores))
        
        # Video-level aggregate (track)
        if track_topk_ids is not None and track_topk_ids.size > 0:
            track_top1_id = int(track_topk_ids[0, 0]) if track_topk_ids.ndim >= 2 else int(track_topk_ids[0])
            track_top1_score = float(track_topk_scores[0, 0]) if track_topk_scores is not None and track_topk_scores.ndim >= 2 else None
            track_confident = bool(track_is_confident_top1[0]) if track_is_confident_top1 is not None and track_is_confident_top1.size > 0 else False
            
            # Get franchise name
            franchise_name = "unknown"
            if semantic_label_names is not None:
                for label_str in semantic_label_names:
                    if isinstance(label_str, str) and ":" in label_str:
                        label_id_str, label_name = label_str.split(":", 1)
                        try:
                            if int(label_id_str) == track_top1_id:
                                franchise_name = label_name
                                break
                        except (ValueError, TypeError):
                            continue
            
            render["summary"]["video_franchise"] = {
                "franchise_id": track_top1_id,
                "franchise_name": franchise_name,
                "score": track_top1_score,
                "is_confident": track_confident,
            }
        
        # Top franchises by frequency
        if len(unique_franchises) > 0:
            franchise_counts = {}
            for franchise_id in top1_ids:
                if franchise_id >= 0:
                    franchise_counts[int(franchise_id)] = franchise_counts.get(int(franchise_id), 0) + 1
            
            # Sort by frequency
            sorted_franchises = sorted(franchise_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            top_franchises = []
            for franchise_id, count in sorted_franchises:
                franchise_name = "unknown"
                if semantic_label_names is not None:
                    for label_str in semantic_label_names:
                        if isinstance(label_str, str) and ":" in label_str:
                            label_id_str, label_name = label_str.split(":", 1)
                            try:
                                if int(label_id_str) == franchise_id:
                                    franchise_name = label_name
                                    break
                            except (ValueError, TypeError):
                                continue
                
                top_franchises.append({
                    "franchise_id": int(franchise_id),
                    "franchise_name": franchise_name,
                    "count": int(count),
                    "ratio": float(count / n_frames) if n_frames > 0 else 0.0,
                })
            
            render["summary"]["top_franchises"] = top_franchises
    
    # Timeline data (per-frame franchise predictions)
    if frame_topk_ids is not None and times_s is not None and frame_indices is not None:
        # Ensure all are numpy arrays
        if isinstance(frame_topk_ids, np.ndarray):
            n = frame_topk_ids.shape[0] if frame_topk_ids.ndim >= 1 else 0
        else:
            n = len(frame_topk_ids) if hasattr(frame_topk_ids, '__len__') else 0
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            
            # Top-1 franchise
            top1_id = int(frame_topk_ids[i, 0]) if frame_topk_ids.ndim >= 2 else int(frame_topk_ids[i])
            top1_score = float(frame_topk_scores[i, 0]) if frame_topk_scores is not None and frame_topk_scores.ndim >= 2 else None
            is_confident = bool(frame_is_confident_top1[i]) if frame_is_confident_top1 is not None and i < len(frame_is_confident_top1) else False
            
            # Get franchise name
            franchise_name = "unknown"
            if semantic_label_names is not None:
                for label_str in semantic_label_names:
                    if isinstance(label_str, str) and ":" in label_str:
                        label_id_str, label_name = label_str.split(":", 1)
                        try:
                            if int(label_id_str) == top1_id:
                                franchise_name = label_name
                                break
                        except (ValueError, TypeError):
                            continue
            
            # Top-K scores
            topk_scores = []
            if frame_topk_scores is not None and frame_topk_scores.ndim >= 2:
                for k in range(min(5, frame_topk_scores.shape[1])):
                    topk_scores.append(float(frame_topk_scores[i, k]) if np.isfinite(frame_topk_scores[i, k]) else None)
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "top1_franchise_id": top1_id if top1_id >= 0 else None,
                "top1_franchise_name": franchise_name if top1_id >= 0 else None,
                "top1_score": top1_score,
                "is_confident": is_confident,
                "topk_scores": topk_scores,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if frame_topk_scores is not None and frame_topk_scores.size > 0:
        # Top-1 scores distribution
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores.ndim >= 2 else frame_topk_scores
        valid_scores = top1_scores[np.isfinite(top1_scores)]
        if valid_scores.size > 0:
            distributions["top1_scores"] = {
                "min": float(np.min(valid_scores)),
                "max": float(np.max(valid_scores)),
                "mean": float(np.mean(valid_scores)),
                "std": float(np.std(valid_scores)),
                "median": float(np.median(valid_scores)),
                "p25": float(np.percentile(valid_scores, 25)),
                "p75": float(np.percentile(valid_scores, 75)),
                "p05": float(np.percentile(valid_scores, 5)),
                "p95": float(np.percentile(valid_scores, 95)),
            }
    
    render["distributions"] = distributions
    
    # Key facts
    render["key_facts"] = {
        "schema_version": meta.get("schema_version", "unknown"),
        "producer_version": meta.get("producer_version", "unknown"),
        "franchise_category": meta.get("franchise_category", "unknown"),
        "embedding_service_url": meta.get("embedding_service_url", "unknown"),
        "num_frames": summary.get("frames_count", 0) if summary else 0,
        "num_franchises": summary.get("unique_franchises_count", 0) if summary else 0,
        "stage_timings_ms": meta.get("stage_timings_ms", {}),
    }
    
    # Config highlights
    render["config_highlights"] = {
        "topk": meta.get("topk", 5),
        "similarity_threshold": meta.get("similarity_threshold", 0.0),
        "threshold_global": meta.get("threshold_global", 0.23),
        "use_ocr_filtering": meta.get("ocr_npz") is not None,
    }
    
    # QA hints
    render["qa_hints"] = {
        "normal_top1_score_range": "0.3-0.9 (typical for correct recognitions)",
        "anomaly_low_scores": "top1_score < 0.2 may indicate no franchise present or poor quality",
        "anomaly_high_variance": "std > 0.3 may indicate inconsistent recognition",
        "check_confident_ratio": "confident_predictions_ratio should be > 0.5 for videos with clear franchise content",
    }
    
    # Top/anti-top examples (based on top1_score)
    if timeline:
        # Sort by top1_score
        sorted_timeline = sorted(
            [t for t in timeline if t.get("top1_score") is not None],
            key=lambda x: x.get("top1_score", 0.0),
            reverse=True
        )
        
        # Top examples (highest scores)
        render["top_examples"] = sorted_timeline[:10] if len(sorted_timeline) >= 10 else sorted_timeline
        
        # Anti-top examples (lowest scores, but only if they have a franchise)
        anti_top = sorted(
            [t for t in timeline if t.get("top1_score") is not None and t.get("top1_franchise_id") is not None],
            key=lambda x: x.get("top1_score", 1.0),
            reverse=False
        )
        render["anti_top_examples"] = anti_top[:10] if len(anti_top) >= 10 else anti_top
    
    return render


def render_franchise_recognition_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать offline HTML mini-dashboard для franchise_recognition (NO CDN, vanilla JS + SVG).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent.parent
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
    render = render_franchise_recognition(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    key_facts = render.get("key_facts", {})
    config_highlights = render.get("config_highlights", {})
    qa_hints = render.get("qa_hints", {})
    top_examples = render.get("top_examples", [])
    anti_top_examples = render.get("anti_top_examples", [])
    
    def _esc(s: Any) -> str:
        """Escape HTML special characters."""
        if s is None:
            return ""
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    
    def format_dist_value(dist_key: str, stat_key: str) -> str:
        """Format distribution statistic value."""
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Build SVG timeline chart (offline, no CDN)
    svg_timeline = ""
    if timeline:
        scores = [t.get("top1_score") for t in timeline if t.get("top1_score") is not None]
        scores_num = [float(s) for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
        if scores_num:
            smin = min(scores_num)
            smax = max(scores_num)
            if smax - smin < 1e-9:
                smin = 0.0
                smax = 1.0
            W, H = 1000, 200
            pts = []
            for i, t in enumerate(timeline):
                s = t.get("top1_score")
                if not isinstance(s, (int, float)) or np.isnan(s):
                    continue
                x = 0 if len(timeline) <= 1 else int(round((i / max(1, len(timeline) - 1)) * (W - 1)))
                denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
                y = int(round((1.0 - ((float(s) - smin) / denom)) * (H - 1)))
                pts.append(f"{x},{y}")
            
            if pts:
                svg_timeline = f"""<svg viewBox='0 0 {W} {H}' width='100%' height='{H}' style='background:#0b1020;border:1px solid #1f2a44;border-radius:8px;display:block'>
                    <polyline fill='none' stroke='#4fd1c5' stroke-width='2' points='{' '.join(pts)}'/>
                    <text x='10' y='20' fill='#a9b8e6' font-size='12'>{_esc(f"min={smin:.3f}, max={smax:.3f}")}</text>
                </svg>"""
    
    # Key facts HTML
    stage_timings = key_facts.get("stage_timings_ms", {})
    key_facts_html = f"""
        <div><div class="k">schema_version</div><div class="v">{_esc(key_facts.get("schema_version", "unknown"))}</div></div>
        <div><div class="k">producer_version</div><div class="v">{_esc(key_facts.get("producer_version", "unknown"))}</div></div>
        <div><div class="k">franchise_category</div><div class="v">{_esc(key_facts.get("franchise_category", "unknown"))}</div></div>
        <div><div class="k">frames</div><div class="v">{_esc(key_facts.get("num_frames", 0))}</div></div>
        <div><div class="k">franchises</div><div class="v">{_esc(key_facts.get("num_franchises", 0))}</div></div>
        <div><div class="k">db_digest</div><div class="v"><span class="pill">{_esc(str(meta.get("db_digest", ""))[:16])}</span></div></div>
        <div><div class="k">total_time_ms</div><div class="v">{_esc(f"{stage_timings.get('total', 0):.0f}")}</div></div>
        <div><div class="k">process_frames_ms</div><div class="v">{_esc(f"{stage_timings.get('process_frames', 0):.0f}")}</div></div>
    """
    
    # Config highlights HTML
    config_highlights_html = f"""
        <div><div class="k">topk</div><div class="v">{_esc(config_highlights.get("topk", 5))}</div></div>
        <div><div class="k">similarity_threshold</div><div class="v">{_esc(config_highlights.get("similarity_threshold", 0.0))}</div></div>
        <div><div class="k">threshold_global</div><div class="v">{_esc(config_highlights.get("threshold_global", 0.23))}</div></div>
        <div><div class="k">use_ocr_filtering</div><div class="v">{_esc("Yes" if config_highlights.get("use_ocr_filtering", False) else "No")}</div></div>
    """
    
    # QA hints HTML
    qa_hints_html = f"""
        <ul style="margin:8px 0; padding-left:20px; font-size:13px; line-height:1.6;">
            <li><strong>Normal top1_score range:</strong> {_esc(qa_hints.get("normal_top1_score_range", "N/A"))}</li>
            <li><strong>Anomaly (low scores):</strong> {_esc(qa_hints.get("anomaly_low_scores", "N/A"))}</li>
            <li><strong>Anomaly (high variance):</strong> {_esc(qa_hints.get("anomaly_high_variance", "N/A"))}</li>
            <li><strong>Check confident_ratio:</strong> {_esc(qa_hints.get("check_confident_ratio", "N/A"))}</li>
        </ul>
    """
    
    # Top franchises table
    top_franchises = summary.get("top_franchises", [])
    top_franchises_rows = "".join([
        f"<tr><td>{_esc(f.get('franchise_name', 'unknown'))}</td><td>{_esc(f.get('count', 0))}</td><td>{_esc(f'{f.get(\"ratio\", 0.0):.2%}')}</td></tr>"
        for f in top_franchises[:10]
    ])
    
    # Top/anti-top examples
    top_examples_html = "".join([
        f"<tr><td>{_esc(e.get('frame_index', ''))}</td><td>{_esc(f\"{e.get('time_sec', 0.0):.2f}s\")}</td><td>{_esc(e.get('top1_franchise_name', 'unknown'))}</td><td>{_esc(f\"{e.get('top1_score', 0.0):.4f}\")}</td><td>{_esc('Yes' if e.get('is_confident', False) else 'No')}</td></tr>"
        for e in top_examples[:10]
    ])
    
    anti_top_examples_html = "".join([
        f"<tr><td>{_esc(e.get('frame_index', ''))}</td><td>{_esc(f\"{e.get('time_sec', 0.0):.2f}s\")}</td><td>{_esc(e.get('top1_franchise_name', 'unknown'))}</td><td>{_esc(f\"{e.get('top1_score', 0.0):.4f}\")}</td><td>{_esc('Yes' if e.get('is_confident', False) else 'No')}</td></tr>"
        for e in anti_top_examples[:10]
    ])
    
    # Timeline data for interactive table (first 3000 frames)
    timeline_limited = timeline[:3000]
    timeline_data_js = json.dumps([{
        "frame_index": t.get("frame_index"),
        "time_sec": t.get("time_sec", 0.0),
        "franchise_name": t.get("top1_franchise_name", ""),
        "top1_score": t.get("top1_score"),
        "is_confident": t.get("is_confident", False),
    } for t in timeline_limited])
    
    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>franchise_recognition — offline mini-dashboard</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, Arial; background:#0b1020; color:#e8eefc; margin:0; }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:24px; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:24px; padding-bottom:16px; border-bottom:1px solid #1f2a44; }}
    .title {{ font-size:20px; font-weight:600; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #1f2a44; font-size:11px; color:#a9b8e6; font-family:monospace; }}
    .nav {{ display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }}
    .nav a {{ color:#7aa2ff; text-decoration:none; padding:6px 12px; border-radius:6px; border:1px solid #1f2a44; }}
    .nav a:hover {{ background:#1f2a44; }}
    .card {{ background:#111a33; border:1px solid #1f2a44; border-radius:12px; padding:16px; margin:16px 0; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
    .k {{ color:#a9b8e6; font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
    .v {{ font-size:18px; }}
    h2 {{ margin-top:0; margin-bottom:12px; font-size:18px; }}
    h3 {{ margin-top:16px; margin-bottom:8px; font-size:16px; }}
    table {{ width:100%; border-collapse: collapse; font-size:13px; }}
    th, td {{ padding:8px; border-bottom:1px solid #1f2a44; text-align:left; }}
    th {{ position:sticky; top:0; background:#111a33; cursor:pointer; user-select:none; }}
    th:hover {{ background:#1a2544; }}
    th.sorted-asc::after {{ content: ' ▲'; font-size:10px; }}
    th.sorted-desc::after {{ content: ' ▼'; font-size:10px; }}
    tbody tr:hover {{ background:#1a2544; }}
    .controls {{ margin-bottom:12px; display:flex; gap:8px; flex-wrap:wrap; }}
    .controls input {{ padding:6px 10px; border:1px solid #1f2a44; border-radius:6px; background:#0b1020; color:#e8eefc; font-size:13px; }}
    .controls input[type="search"] {{ flex:1; min-width:200px; }}
    .section {{ margin:32px 0; }}
    .example-card {{ background:#1a2544; border:1px solid #2a3554; border-radius:8px; padding:12px; margin:8px 0; }}
    .example-card .meta {{ font-size:11px; color:#a9b8e6; margin-top:4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title">franchise_recognition — render (dev-only)</div>
      <div class="pill">offline • mini-dashboard</div>
    </div>

    <div class="nav">
      <a href="#overview">Overview</a>
      <a href="#qa">QA (top/anti-top)</a>
      <a href="#timeline">Timeline</a>
      <a href="#tables">Tables</a>
      <a href="#meta">Meta</a>
    </div>

    <div id="overview" class="section">
      <div class="card">
        <h2>What is this?</h2>
        <p style="margin:8px 0; line-height:1.6;">
          This component recognizes specific franchises/titles (games/anime/cartoons) in video frames using Embedding Service + CLIP frame embeddings.
          It writes per-frame and video-level top-K franchise IDs and similarity scores to NPZ.
        </p>
      </div>

      <div class="card">
        <h2>Key facts</h2>
        <div class="grid">
          {key_facts_html}
        </div>
      </div>

      <div class="card">
        <h2>Config highlights</h2>
        <div class="grid">
          {config_highlights_html}
        </div>
      </div>

      <div class="card">
        <h2>How to QA</h2>
        {qa_hints_html}
      </div>
    </div>

    <div id="qa" class="section">
      <h2>QA peaks</h2>
      <div class="card">
        <h3>Top examples (highest similarity scores)</h3>
        <table id="topExamplesTable">
          <thead>
            <tr>
              <th data-k="frame_index">frame</th>
              <th data-k="time_sec">time_s</th>
              <th data-k="franchise_name">franchise</th>
              <th data-k="top1_score">score</th>
              <th data-k="is_confident">confident</th>
            </tr>
          </thead>
          <tbody>
            {top_examples_html if top_examples_html else "<tr><td colspan='5'>No examples</td></tr>"}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h3>Anti-top examples (lowest scores, but with franchise)</h3>
        <table id="antiTopExamplesTable">
          <thead>
            <tr>
              <th data-k="frame_index">frame</th>
              <th data-k="time_sec">time_s</th>
              <th data-k="franchise_name">franchise</th>
              <th data-k="top1_score">score</th>
              <th data-k="is_confident">confident</th>
            </tr>
          </thead>
          <tbody>
            {anti_top_examples_html if anti_top_examples_html else "<tr><td colspan='5'>No examples</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>

    <div id="timeline" class="section">
      <div class="card">
        <h2>Timeline: Top-1 Franchise Score Over Time</h2>
        {svg_timeline if svg_timeline else "<p>No timeline data available</p>"}
        <div class="k" style="margin-top:8px">NPZ остаётся source-of-truth; это dev-only render.</div>
      </div>
    </div>

    <div id="tables" class="section">
      <div class="card">
        <h2>Top franchises by frequency</h2>
        <table id="topFranchisesTable">
          <thead>
            <tr>
              <th data-k="franchise_name">Franchise Name</th>
              <th data-k="count">Count</th>
              <th data-k="ratio">Ratio</th>
            </tr>
          </thead>
          <tbody>
            {top_franchises_rows if top_franchises_rows else "<tr><td colspan='3'>No franchises found</td></tr>"}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>All frames (search/sort) — first 3000</h2>
        <div class="controls">
          <input id="searchInput" type="search" placeholder="search (frame_index/time_sec/franchise_name)..." />
          <input id="minScoreFilter" type="number" step="0.01" min="0" max="1" placeholder="min top1_score" />
          <label style="display:flex; align-items:center; gap:6px; color:#a9b8e6; font-size:12px;">
            <input type="checkbox" id="confidentOnly" />
            Confident only
          </label>
        </div>
        <div style="overflow:auto; border:1px solid #1f2a44; border-radius:8px; max-height:600px;">
          <table id="timelineTable">
            <thead>
              <tr>
                <th data-k="frame_index">frame</th>
                <th data-k="time_sec">time_s</th>
                <th data-k="franchise_name">franchise</th>
                <th data-k="top1_score">score</th>
                <th data-k="is_confident">confident</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2>Distribution statistics</h2>
        <table>
          <thead>
            <tr>
              <th>Statistic</th>
              <th>Top-1 Score</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><strong>Min</strong></td><td>{format_dist_value('top1_scores', 'min')}</td></tr>
            <tr><td><strong>Max</strong></td><td>{format_dist_value('top1_scores', 'max')}</td></tr>
            <tr><td><strong>Mean</strong></td><td>{format_dist_value('top1_scores', 'mean')}</td></tr>
            <tr><td><strong>Std</strong></td><td>{format_dist_value('top1_scores', 'std')}</td></tr>
            <tr><td><strong>Median</strong></td><td>{format_dist_value('top1_scores', 'median')}</td></tr>
            <tr><td><strong>P25</strong></td><td>{format_dist_value('top1_scores', 'p25')}</td></tr>
            <tr><td><strong>P75</strong></td><td>{format_dist_value('top1_scores', 'p75')}</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div id="meta" class="section">
      <div class="card">
        <h2>Metadata</h2>
        <pre style="background:#0b1020; padding:12px; border-radius:6px; overflow:auto; font-size:11px; font-family:monospace;">{_esc(json.dumps(meta, indent=2, ensure_ascii=False))}</pre>
      </div>
    </div>
  </div>

  <script>
    // Timeline data
    const timelineData = {timeline_data_js};
    
    // Table sorting (vanilla JS, offline)
    function setupTableSorting(tableId) {{
      const table = document.getElementById(tableId);
      if (!table) return;
      const thead = table.querySelector('thead');
      const tbody = table.querySelector('tbody');
      if (!thead || !tbody) return;
      
      const headers = thead.querySelectorAll('th[data-k]');
      let sortKey = null;
      let sortAsc = true;
      
      headers.forEach(header => {{
        header.addEventListener('click', () => {{
          const key = header.getAttribute('data-k');
          if (sortKey === key) {{
            sortAsc = !sortAsc;
          }} else {{
            sortKey = key;
            sortAsc = true;
          }}
          
          // Update header classes
          headers.forEach(h => {{
            h.classList.remove('sorted-asc', 'sorted-desc');
          }});
          header.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
          
          // Sort rows
          const rows = Array.from(tbody.querySelectorAll('tr'));
          rows.sort((a, b) => {{
            const aVal = a.cells[Array.from(headers).indexOf(header)]?.textContent || '';
            const bVal = b.cells[Array.from(headers).indexOf(header)]?.textContent || '';
            const aNum = parseFloat(aVal);
            const bNum = parseFloat(bVal);
            if (!isNaN(aNum) && !isNaN(bNum)) {{
              return sortAsc ? aNum - bNum : bNum - aNum;
            }}
            return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
          }});
          
          rows.forEach(row => tbody.appendChild(row));
        }});
      }});
    }}
    
    // Search and filter for timeline table
    function setupTimelineTable() {{
      const table = document.getElementById('timelineTable');
      if (!table) return;
      const tbody = table.querySelector('tbody');
      if (!tbody) return;
      
      // Render initial rows
      function renderRows(data) {{
        tbody.innerHTML = data.map(t => `
          <tr>
            <td>${{t.frame_index}}</td>
            <td>${{t.time_sec.toFixed(2)}}s</td>
            <td>${{t.franchise_name || 'unknown'}}</td>
            <td>${{t.top1_score !== null && t.top1_score !== undefined ? t.top1_score.toFixed(4) : 'N/A'}}</td>
            <td>${{t.is_confident ? 'Yes' : 'No'}}</td>
          </tr>
        `).join('');
      }}
      
      renderRows(timelineData);
      
      const searchInput = document.getElementById('searchInput');
      const minScoreFilter = document.getElementById('minScoreFilter');
      const confidentOnly = document.getElementById('confidentOnly');
      
      function filterRows() {{
        const search = (searchInput?.value || '').toLowerCase();
        const minScore = minScoreFilter?.value ? parseFloat(minScoreFilter.value) : null;
        const onlyConfident = confidentOnly?.checked || false;
        
        const filtered = timelineData.filter(t => {{
          if (search && !(
            String(t.frame_index).includes(search) ||
            String(t.time_sec).includes(search) ||
            (t.franchise_name || '').toLowerCase().includes(search)
          )) return false;
          if (minScore !== null && (t.top1_score === null || t.top1_score === undefined || t.top1_score < minScore)) return false;
          if (onlyConfident && !t.is_confident) return false;
          return true;
        }});
        
        renderRows(filtered);
      }}
      
      if (searchInput) searchInput.addEventListener('input', filterRows);
      if (minScoreFilter) minScoreFilter.addEventListener('input', filterRows);
      if (confidentOnly) confidentOnly.addEventListener('change', filterRows);
    }}
    
    // Initialize all tables
    setupTableSorting('topFranchisesTable');
    setupTableSorting('topExamplesTable');
    setupTableSorting('antiTopExamplesTable');
    setupTimelineTable();
  </script>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Franchise Recognition HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_franchise_recognition", "render_franchise_recognition_html"]

