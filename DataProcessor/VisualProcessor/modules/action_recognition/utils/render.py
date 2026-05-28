"""
Renderer for action_recognition: render-context JSON + fully-offline HTML (no CDN).

Audit v3:
- No Chart.js CDN; offline-only HTML.
- action_recognition produces per-track embeddings and analytics payload; render should be lightweight.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List, Tuple

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


def _unbox_object(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
    }


def render_action_recognition(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    tracks = npz_data.get("tracks")
    if isinstance(tracks, list):
        tracks = np.asarray(tracks, dtype=np.int32)
    elif isinstance(tracks, np.ndarray):
        tracks = np.asarray(tracks, dtype=np.int32).reshape(-1)
    else:
        tracks = np.asarray([], dtype=np.int32)

    embeddings = _unbox_object(npz_data.get("embeddings"))
    results_json = _unbox_object(npz_data.get("results_json"))
    ui_payload = meta.get("ui_payload", {})

    per_track = []
    timeline_data = []
    top_jumps = []
    
    if isinstance(embeddings, (list, tuple)) and len(embeddings) == int(tracks.size):
        for i in range(int(tracks.size)):
            emb_i = embeddings[i]
            track_id = int(tracks[i])
            try:
                emb_arr = np.asarray(emb_i, dtype=np.float32)
                n_clips = int(emb_arr.shape[0]) if emb_arr.ndim == 2 else int(emb_arr.size > 0)
                dim = int(emb_arr.shape[1]) if emb_arr.ndim == 2 else int(emb_arr.size)
                norms = np.linalg.norm(emb_arr, axis=1).astype(np.float32) if emb_arr.ndim == 2 else np.asarray([], dtype=np.float32)
                
                # Получаем метрики из results_json
                track_metrics = {}
                if isinstance(results_json, (list, tuple)) and i < len(results_json):
                    track_metrics = results_json[i] if isinstance(results_json[i], dict) else {}
                
                stability = track_metrics.get("stability")
                stability_centroid_dist = track_metrics.get("stability_centroid_dist")
                max_temporal_jump = track_metrics.get("max_temporal_jump")
                mean_temporal_jump = track_metrics.get("mean_temporal_jump")
                clip_center_times_s = track_metrics.get("clip_center_times_s", [])
                temporal_jumps = track_metrics.get("temporal_jumps", [])
                
                per_track.append(
                    {
                        "track_id": track_id,
                        "num_clips": n_clips,
                        "embedding_dim": dim,
                        "clip_norm_mean": float(np.mean(norms)) if norms.size else float("nan"),
                        "stability": float(stability) if stability is not None and not (isinstance(stability, float) and np.isnan(stability)) else None,
                        "stability_centroid_dist": float(stability_centroid_dist) if stability_centroid_dist is not None and not (isinstance(stability_centroid_dist, float) and np.isnan(stability_centroid_dist)) else None,
                        "max_temporal_jump": float(max_temporal_jump) if max_temporal_jump is not None and not (isinstance(max_temporal_jump, float) and np.isnan(max_temporal_jump)) else None,
                        "mean_temporal_jump": float(mean_temporal_jump) if mean_temporal_jump is not None and not (isinstance(mean_temporal_jump, float) and np.isnan(mean_temporal_jump)) else None,
                    }
                )
                
                # Timeline data для графика
                if isinstance(clip_center_times_s, (list, tuple)) and len(clip_center_times_s) == len(norms):
                    for j, (t, norm) in enumerate(zip(clip_center_times_s, norms)):
                        timeline_data.append({
                            "track_id": track_id,
                            "time_s": float(t),
                            "embedding_norm": float(norm),
                            "clip_index": j,
                        })
                
                # Top jumps
                if isinstance(temporal_jumps, (list, tuple)) and isinstance(clip_center_times_s, (list, tuple)):
                    for j, jump in enumerate(temporal_jumps):
                        if j < len(clip_center_times_s):
                            top_jumps.append({
                                "track_id": track_id,
                                "clip_index": j + 1,
                                "time_s": float(clip_center_times_s[j + 1]) if j + 1 < len(clip_center_times_s) else None,
                                "jump": float(jump) if not (isinstance(jump, float) and np.isnan(jump)) else None,
                            })
            except Exception as e:
                logger.debug(f"Error processing track {i}: {e}")
                per_track.append({"track_id": int(tracks[i]), "num_clips": 0, "embedding_dim": 0, "clip_norm_mean": float("nan")})

    # Сортируем top jumps
    top_jumps = sorted([j for j in top_jumps if j.get("jump") is not None], key=lambda x: x["jump"], reverse=True)[:10]

    key_facts: Dict[str, Any] = {
        "tracks_count": int(tracks.size),
        "results_rows": int(len(results_json)) if isinstance(results_json, (list, tuple)) else 0,
    }
    
    # Summary из ui_payload
    summary = ui_payload.get("summary", {}) if isinstance(ui_payload, dict) else {}

    return {
        "component": "action_recognition",
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "status": str(meta.get("status") or ""),
        "key_facts": key_facts,
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        "per_track": per_track,
        "timeline_data": timeline_data,
        "top_jumps": top_jumps,
        "summary": summary,
    }


def render_action_recognition_html(npz_path: str, output_path: str) -> str:
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except Exception:
        def load_npz(path: str):
            data = np.load(path, allow_pickle=True)
            result: Dict[str, Any] = {}
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    result[key] = arr.item() if arr.size == 1 else arr.tolist()
                else:
                    result[key] = arr
            return result

        def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
            meta2 = npz_data.get("meta")
            if isinstance(meta2, np.ndarray) and meta2.dtype == object:
                return meta2.item() if meta2.size == 1 else {}
            return meta2 if isinstance(meta2, dict) else {}

    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    ctx = render_action_recognition(npz_data, meta)

    key_facts = ctx.get("key_facts") or {}
    stage_timings = ctx.get("stage_timings_ms") or {}
    per_track = ctx.get("per_track") or []
    timeline_data = ctx.get("timeline_data") or []
    top_jumps = ctx.get("top_jumps") or []
    summary = ctx.get("summary") or {}

    status = _esc(ctx.get("status") or "")
    schema_version = _esc(ctx.get("schema_version") or "")
    producer_version = _esc(ctx.get("producer_version") or "")

    # Summary metrics
    summary_html = ""
    if summary:
        avg_stab = f"{summary.get('avg_stability', 0):.3f}" if isinstance(summary.get('avg_stability'), (int, float)) else 'N/A'
        avg_stab_cd = f"{summary.get('avg_stability_centroid_dist', 0):.3f}" if isinstance(summary.get('avg_stability_centroid_dist'), (int, float)) else 'N/A'
        avg_max_jump = f"{summary.get('avg_max_temporal_jump', 0):.3f}" if isinstance(summary.get('avg_max_temporal_jump'), (int, float)) else 'N/A'
        avg_mean_jump = f"{summary.get('avg_mean_temporal_jump', 0):.3f}" if isinstance(summary.get('avg_mean_temporal_jump'), (int, float)) else 'N/A'
        summary_html = f"""
    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Summary Metrics</div>
      <div class="kv">
        <div>Total Tracks</div><div>{summary.get('num_tracks', 'N/A')}</div>
        <div>Total Clips</div><div>{summary.get('num_clips_total', 'N/A')}</div>
        <div>Avg Stability</div><div>{avg_stab}</div>
        <div>Avg Stability Centroid Dist</div><div>{avg_stab_cd}</div>
        <div>Avg Max Temporal Jump</div><div>{avg_max_jump}</div>
        <div>Avg Mean Temporal Jump</div><div>{avg_mean_jump}</div>
      </div>
    </div>"""

    # Stability bar chart
    stability_chart = ""
    if per_track:
        stability_vals = [r.get("stability") for r in per_track if r.get("stability") is not None]
        if stability_vals:
            max_stab = max(stability_vals)
            min_stab = min(stability_vals)
            range_stab = max_stab - min_stab if max_stab > min_stab else 1.0
            bars = []
            for i, r in enumerate(per_track):
                stab = r.get("stability")
                if stab is not None:
                    height = int(((stab - min_stab) / range_stab) * 200) if range_stab > 0 else 100
                    bars.append(f'<rect x="{i * 30}" y="{200 - height}" width="25" height="{height}" fill="#3b82f6" opacity="0.7"/>')
                    bars.append(f'<text x="{i * 30 + 12}" y="220" text-anchor="middle" font-size="10">{r.get("track_id")}</text>')
            if bars:
                stability_chart = f"""
    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Stability by Track</div>
      <svg width="{max(300, len(per_track) * 30)}" height="240" style="border: 1px solid #e2e8f0;">
        {"".join(bars)}
      </svg>
    </div>"""

    # Timeline chart (embedding norms)
    timeline_chart = ""
    if timeline_data:
        # Группируем по track_id
        tracks_timeline = {}
        for item in timeline_data:
            tid = item.get("track_id")
            if tid not in tracks_timeline:
                tracks_timeline[tid] = []
            tracks_timeline[tid].append(item)
        
        # Находим диапазоны
        all_times = [item.get("time_s") for item in timeline_data if item.get("time_s") is not None]
        all_norms = [item.get("embedding_norm") for item in timeline_data if item.get("embedding_norm") is not None]
        if all_times and all_norms:
            min_time = min(all_times)
            max_time = max(all_times)
            min_norm = min(all_norms)
            max_norm = max(all_norms)
            time_range = max_time - min_time if max_time > min_time else 1.0
            norm_range = max_norm - min_norm if max_norm > min_norm else 1.0
            
            colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
            paths = []
            for idx, (tid, items) in enumerate(tracks_timeline.items()):
                color = colors[idx % len(colors)]
                points = []
                for item in sorted(items, key=lambda x: x.get("time_s", 0)):
                    t = item.get("time_s")
                    n = item.get("embedding_norm")
                    if t is not None and n is not None:
                        x = int(((t - min_time) / time_range) * 800) if time_range > 0 else 0
                        y = int(200 - ((n - min_norm) / norm_range) * 200) if norm_range > 0 else 100
                        points.append(f"{x},{y}")
                if points:
                    paths.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.7"/>')
                    paths.append(f'<text x="10" y="{20 + idx * 20}" fill="{color}" font-size="12">Track {tid}</text>')
            
            if paths:
                timeline_chart = f"""
    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Timeline: Embedding Norms</div>
      <svg width="850" height="250" style="border: 1px solid #e2e8f0;">
        {"".join(paths)}
      </svg>
    </div>"""

    # Top jumps table
    top_jumps_table = ""
    if top_jumps:
        rows = []
        for j in top_jumps[:10]:
            time_s_val = f"{j.get('time_s', 0):.2f}" if j.get('time_s') is not None else 'N/A'
            jump_val = f"{j.get('jump', 0):.3f}" if j.get('jump') is not None else 'N/A'
            rows.append(
                "<tr>"
                f"<td>{_esc(j.get('track_id', ''))}</td>"
                f"<td>{_esc(j.get('clip_index', ''))}</td>"
                f"<td>{_esc(time_s_val)}</td>"
                f"<td>{_esc(jump_val)}</td>"
                "</tr>"
            )
        top_jumps_table = f"""
    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Top 10 Clips with Highest Temporal Jumps</div>
      <table>
        <thead><tr><th>Track ID</th><th>Clip Index</th><th>Time (s)</th><th>Jump</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""

    # Track details table
    track_rows = ""
    if isinstance(per_track, list) and per_track:
        rows = []
        for r in per_track[:200]:
            if not isinstance(r, dict):
                continue
            clip_norm_mean_val = f"{r.get('clip_norm_mean', 0):.3f}" if isinstance(r.get('clip_norm_mean'), (int, float)) and not np.isnan(r.get('clip_norm_mean', 0)) else 'N/A'
            stability_val = f"{r.get('stability', 0):.3f}" if r.get('stability') is not None else 'N/A'
            max_jump_val = f"{r.get('max_temporal_jump', 0):.3f}" if r.get('max_temporal_jump') is not None else 'N/A'
            rows.append(
                "<tr>"
                f"<td>{_esc(r.get('track_id'))}</td>"
                f"<td>{_esc(r.get('num_clips'))}</td>"
                f"<td>{_esc(r.get('embedding_dim'))}</td>"
                f"<td>{_esc(clip_norm_mean_val)}</td>"
                f"<td>{_esc(stability_val)}</td>"
                f"<td>{_esc(max_jump_val)}</td>"
                "</tr>"
            )
        track_rows = "".join(rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>action_recognition — render</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 12px 0; }}
    .muted {{ color: #64748b; }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; }}
    .kv div {{ padding: 2px 0; border-bottom: 1px dashed #e2e8f0; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
    th {{ color: #334155; font-weight: 700; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div style="font-size:18px; font-weight:700;">action_recognition</div>
      <div class="muted">schema: <code>{schema_version}</code> · producer_version: <code>{producer_version}</code> · status: <code>{status}</code></div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Key facts</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{_esc(v)}</div>" for k,v in key_facts.items()])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Stage timings (ms)</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{float(v):.2f}</div>" for k,v in stage_timings.items()])}
      </div>
    </div>
{summary_html}
{stability_chart}
{timeline_chart}
{top_jumps_table}
    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Tracks Details</div>
      {("<table><thead><tr><th>track</th><th>num_clips</th><th>dim</th><th>clip_norm_mean</th><th>stability</th><th>max_jump</th></tr></thead><tbody>"+track_rows+"</tbody></table>") if track_rows else "<div class='muted'>No track data</div>"}
    </div>
  </div>
</body>
</html>
"""

    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


__all__ = ["render_action_recognition", "render_action_recognition_html"]


