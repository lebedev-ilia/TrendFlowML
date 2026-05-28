"""
Audit v3 renderer for face_identity.

- `render_face_identity(...)` builds human-friendly render_context JSON (saved by VisualProcessor).
- `render_face_identity_html(...)` builds offline `render.html` (NO CDN) + optional assets in `_render/assets/`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_vp_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
if _vp_root not in sys.path:
    sys.path.append(_vp_root)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from utils.frame_manager import FrameManager  # type: ignore
except Exception:  # pragma: no cover
    FrameManager = None  # type: ignore


def _np(x: Any, dtype: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)
    if isinstance(x, list):
        return np.asarray(x, dtype=dtype)
    return None


def _parse_label_space(
    semantic_label_names: Optional[np.ndarray],
    semantic_object_ids: Optional[np.ndarray],
) -> Tuple[Dict[int, str], Dict[int, str]]:
    id_to_name: Dict[int, str] = {}
    id_to_uuid: Dict[int, str] = {}
    if semantic_label_names is not None:
        for s in semantic_label_names.tolist():
            ss = str(s)
            if ":" not in ss:
                continue
            try:
                i_str, name = ss.split(":", 1)
                i = int(i_str)
                id_to_name[i] = name
            except Exception:
                continue
    if semantic_object_ids is not None:
        for i, oid in enumerate(semantic_object_ids.tolist()):
            id_to_uuid[int(i)] = str(oid)
    return id_to_name, id_to_uuid


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _compute_score_stats(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {}
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {}
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "count": int(vals.size),
    }


def _maybe_write_face_crop(
    *,
    frames_dir: Optional[str],
    assets_dir: Optional[str],
    frame_indices: np.ndarray,
    frame_pos: int,
    bbox_xyxy: np.ndarray,
) -> Optional[str]:
    """Extract and save face crop for render asset."""
    if not frames_dir or not assets_dir:
        return None
    if cv2 is None or FrameManager is None:
        return None
    if frame_pos < 0 or frame_pos >= int(frame_indices.shape[0]):
        return None
    if bbox_xyxy is None or not np.all(np.isfinite(bbox_xyxy)):
        return None

    frame_idx_global = int(frame_indices[int(frame_pos)])
    fn = f"face_frame_{frame_idx_global}.jpg"
    out_path = os.path.join(assets_dir, fn)
    if os.path.isfile(out_path):
        return fn

    try:
        fm = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
        frame = fm.get(frame_idx_global)
        h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
        x1_i = max(0, min(int(round(bbox_xyxy[0])), w_img - 1))
        y1_i = max(0, min(int(round(bbox_xyxy[1])), h_img - 1))
        x2_i = max(x1_i + 1, min(int(round(bbox_xyxy[2])), w_img))
        y2_i = max(y1_i + 1, min(int(round(bbox_xyxy[3])), h_img))
        crop = frame[y1_i:y2_i, x1_i:x2_i, :].copy()
        if crop.size == 0:
            try:
                fm.close()
            except Exception:
                pass
            return None
        os.makedirs(assets_dir, exist_ok=True)
        ok = cv2.imwrite(out_path, crop)
        try:
            fm.close()
        except Exception:
            pass
        return fn if ok else None
    except Exception:
        return None


def render_face_identity(
    npz_data: Dict[str, Any],
    meta: Dict[str, Any],
    render_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Render-context for face_identity (Audit v3).
    
    Returns render_context dict with:
    - summary: key statistics
    - key_facts: schema_version, producer_version, db_digest, etc.
    - config_highlights: important config parameters
    - qa_hints: how to interpret results
    - distributions: score statistics
    - top_faces: top faces by count/score
    - examples: top/anti-top examples with assets
    - timeline: per-frame data
    """
    render_env = render_env or {}
    frames_dir = render_env.get("frames_dir")
    assets_dir = render_env.get("assets_dir")
    
    face_ids = _np(npz_data.get("face_ids"), np.int32)
    face_names = _np(npz_data.get("face_names"), "U256")
    face_similarities = _np(npz_data.get("face_similarities"), np.float32)
    frame_indices = _np(npz_data.get("frame_indices"), np.int32)
    times_s = _np(npz_data.get("times_s"), np.float32)
    face_bbox_xyxy = _np(npz_data.get("face_bbox_xyxy"), np.float32)
    semantic_label_names = _np(npz_data.get("semantic_label_names"), "U")
    semantic_object_ids = _np(npz_data.get("semantic_object_ids"), "U")
    
    id_to_name, id_to_uuid = _parse_label_space(semantic_label_names, semantic_object_ids)
    
    n_frames = int(face_ids.shape[0]) if face_ids is not None and face_ids.size > 0 else 0
    topk = int(face_ids.shape[1]) if face_ids is not None and face_ids.ndim > 1 and face_ids.size > 0 else 5
    
    # Summary statistics
    valid_mask = (face_ids != -1) if face_ids is not None else np.array([], dtype=bool)
    valid_similarities = face_similarities[valid_mask] if face_similarities is not None and valid_mask.size > 0 else np.array([])
    
    unique_face_ids = set()
    unique_face_names = set()
    if face_ids is not None and face_ids.size > 0:
        for frame_idx in range(n_frames):
            for k in range(topk):
                face_id = int(face_ids[frame_idx, k])
                if face_id != -1:
                    unique_face_ids.add(face_id)
                    if face_names is not None:
                        face_name = str(face_names[frame_idx, k]).strip()
                        if face_name:
                            unique_face_names.add(face_name)
    
    confident_count = int(np.sum(valid_similarities > 0.7)) if valid_similarities.size > 0 else 0
    
    top1_similarities = []
    if face_similarities is not None and face_similarities.shape[0] > 0:
        top1_similarities = face_similarities[:, 0].tolist()
        top1_similarities = [s for s in top1_similarities if s > 0.0]
    
    # Top faces statistics
    face_stats: Dict[str, Dict[str, Any]] = {}
    if face_names is not None and face_ids is not None and face_ids.size > 0:
        for frame_idx in range(n_frames):
            for k in range(topk):
                face_id = int(face_ids[frame_idx, k])
                if face_id == -1:
                    continue
                face_name = str(face_names[frame_idx, k]).strip() if face_names is not None else ""
                similarity = float(face_similarities[frame_idx, k]) if face_similarities is not None else 0.0
                if not face_name:
                    face_name = id_to_name.get(face_id, f"face_{face_id}")
                if face_name not in face_stats:
                    face_stats[face_name] = {
                        "face_id": face_id,
                        "count": 0,
                        "total_score": 0.0,
                        "max_score": 0.0,
                        "min_score": 1.0,
                    }
                face_stats[face_name]["count"] += 1
                face_stats[face_name]["total_score"] += similarity
                face_stats[face_name]["max_score"] = max(face_stats[face_name]["max_score"], similarity)
                face_stats[face_name]["min_score"] = min(face_stats[face_name]["min_score"], similarity)
    
    top_faces = []
    for face_name, stats in face_stats.items():
        top_faces.append({
            "face_name": face_name,
            "face_id": stats["face_id"],
            "count": stats["count"],
            "avg_score": stats["total_score"] / stats["count"],
            "max_score": stats["max_score"],
            "min_score": stats["min_score"],
        })
    top_faces.sort(key=lambda x: (x["count"], x["avg_score"]), reverse=True)
    
    # Examples: top and anti-top
    examples = []
    if face_similarities is not None and face_similarities.shape[0] > 0:
        # Top examples (highest scores)
        top1_scores_with_pos = [(float(face_similarities[i, 0]), i) for i in range(n_frames) if face_similarities[i, 0] > 0.0]
        top1_scores_with_pos.sort(reverse=True)
        top_examples = top1_scores_with_pos[:10]
        
        # Anti-top examples (lowest scores > 0)
        anti_top_examples = top1_scores_with_pos[-10:] if len(top1_scores_with_pos) >= 10 else []
        
        for score, pos in top_examples + anti_top_examples:
            face_id = int(face_ids[pos, 0]) if face_ids is not None else -1
            face_name = str(face_names[pos, 0]).strip() if face_names is not None else ""
            if not face_name and face_id >= 0:
                face_name = id_to_name.get(face_id, f"face_{face_id}")
            asset = None
            if frames_dir and assets_dir and face_bbox_xyxy is not None and pos < face_bbox_xyxy.shape[0]:
                asset = _maybe_write_face_crop(
                    frames_dir=str(frames_dir) if frames_dir else None,
                    assets_dir=str(assets_dir) if assets_dir else None,
                    frame_indices=frame_indices,
                    frame_pos=pos,
                    bbox_xyxy=face_bbox_xyxy[pos, :],
                )
            examples.append({
                "frame_pos": pos,
                "frame_index": int(frame_indices[pos]) if frame_indices is not None else pos,
                "time_sec": float(times_s[pos]) if times_s is not None else 0.0,
                "top1_face_id": face_id,
                "top1_face_name": face_name,
                "top1_score": score,
                "is_confident": score > 0.7,
                "asset": asset,
            })
    
    # Timeline
    timeline = []
    if times_s is not None and len(times_s) == n_frames:
        for i in range(min(n_frames, 3000)):  # Limit for performance
            frame_data = {
                "pos": i,
                "frame_index": int(frame_indices[i]) if frame_indices is not None else i,
                "time_sec": float(times_s[i]),
            }
            if face_ids is not None and i < face_ids.shape[0]:
                top1_id = int(face_ids[i, 0])
                top1_name = str(face_names[i, 0]).strip() if face_names is not None and i < face_names.shape[0] else ""
                top1_score = float(face_similarities[i, 0]) if face_similarities is not None and i < face_similarities.shape[0] else 0.0
                frame_data["top1_face_id"] = top1_id if top1_id != -1 else None
                frame_data["top1_face_name"] = top1_name if top1_name else None
                frame_data["top1_score"] = top1_score if top1_score > 0.0 else None
                frame_data["is_confident"] = top1_score > 0.7
            timeline.append(frame_data)
    
    render = {
        "component": "face_identity",
        "summary": {
            "frames": n_frames,
            "topk": topk,
            "unique_faces_count": len(unique_face_ids),
            "unique_face_names_count": len(unique_face_names),
            "total_identifications": int(np.sum(valid_mask)) if valid_mask.size > 0 else 0,
            "confident_predictions_count": confident_count,
            "confident_predictions_ratio": float(confident_count / max(1, np.sum(valid_mask))) if valid_mask.size > 0 else 0.0,
            "status": str(meta.get("status") or "unknown"),
            "empty_reason": meta.get("empty_reason"),
        },
        "key_facts": {
            "schema_version": meta.get("schema_version"),
            "producer_version": meta.get("producer_version"),
            "db_digest": (str(meta.get("db_digest") or "")[:16] + "...") if meta.get("db_digest") else None,
            "embedding_model": meta.get("embedding_model"),
            "stage_timings_ms": meta.get("stage_timings_ms", {}),
        },
        "config_highlights": {
            "top_k": meta.get("top_k"),
            "similarity_threshold": meta.get("similarity_threshold"),
            "category": meta.get("category"),
        },
        "qa_hints": [
            "Проверьте, что top примеры визуально похожи на известных людей, а не на шум.",
            "Если много confident=false, возможно порог similarity_threshold слишком высокий или база маленькая.",
            "Если много confident=true при явном мусоре — порог слишком низкий или качество face crops плохое.",
            "Проверьте стабильность: одинаковые люди должны иметь похожие similarity scores на соседних кадрах.",
        ],
        "distributions": {
            "top1_scores": _compute_score_stats(np.array(top1_similarities)) if top1_similarities else {},
            "all_scores": _compute_score_stats(valid_similarities) if valid_similarities.size > 0 else {},
        },
        "top_faces": top_faces[:50],
        "examples": examples,
        "timeline": timeline,
    }
    return render


def render_face_identity_html(
    npz_path: str,
    output_path: str,
    *,
    frames_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
) -> str:
    """Offline HTML render (no external deps)."""
    # Load NPZ (allow pickle for meta dict)
    z = np.load(npz_path, allow_pickle=True)
    npz_data: Dict[str, Any] = {k: z[k] for k in z.files}
    meta = {}
    try:
        m = npz_data.get("meta")
        if isinstance(m, np.ndarray) and m.dtype == object and m.shape == ():
            meta = m.item() if isinstance(m.item(), dict) else {}
    except Exception:
        meta = {}

    # Render context + assets
    render = render_face_identity(
        {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v) for k, v in npz_data.items()},
        meta,
        {"frames_dir": frames_dir, "assets_dir": assets_dir},
    )

    examples = render.get("examples", []) or []
    timeline = render.get("timeline", []) or []
    top_faces = render.get("top_faces", []) or []
    key_facts = render.get("key_facts", {}) or {}
    cfg = render.get("config_highlights", {}) or {}
    summary = render.get("summary", {}) or {}

    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>face_identity — Audit v3 render</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; color: #111; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    code, pre {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #fff; }}
    img.thumb {{ max-width: 220px; border-radius: 8px; border: 1px solid #ddd; }}
    .pill {{ display: inline-block; padding: 2px 8px; border: 1px solid #ddd; border-radius: 999px; font-size: 12px; }}
    .pill.ok {{ background: #eefaf0; border-color: #bfe7c6; }}
    .pill.empty {{ background: #fff6e6; border-color: #ffd59a; }}
    .pill.error {{ background: #ffecec; border-color: #ffb3b3; }}
    input {{ padding: 8px; width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>face_identity <span class="pill {esc(summary.get('status', 'unknown'))}">{esc(summary.get('status', 'unknown'))}</span></h1>
  <p class="muted">NPZ is source-of-truth. This page is dev-only and works offline.</p>

  <div class="banner" style="background: #fff6e6; border: 1px solid #ffd59a; border-radius: 8px; padding: 12px; margin: 16px 0;">
    <strong>Privacy notice:</strong> This page contains face crops with personal data (faces). Use only for local QA in dev; do not publish or share. Assets are stored in <code>_render/assets/</code>.
  </div>

  <h2>Key facts</h2>
  <div class="grid">
    <div class="card"><div class="muted">schema_version</div><div><code>{esc(key_facts.get('schema_version'))}</code></div></div>
    <div class="card"><div class="muted">producer_version</div><div><code>{esc(key_facts.get('producer_version'))}</code></div></div>
    <div class="card"><div class="muted">db_digest</div><div><code>{esc(key_facts.get('db_digest'))}</code></div></div>
    <div class="card"><div class="muted">embedding_model</div><div><code>{esc(key_facts.get('embedding_model'))}</code></div></div>
    <div class="card"><div class="muted">frames</div><div><code>{esc(summary.get('frames', 0))}</code></div></div>
    <div class="card"><div class="muted">unique_faces</div><div><code>{esc(summary.get('unique_faces_count', 0))}</code></div></div>
  </div>

  <h2>Config highlights</h2>
  <div class="grid">
    <div class="card"><div class="muted">top_k</div><div><code>{esc(cfg.get('top_k'))}</code></div></div>
    <div class="card"><div class="muted">similarity_threshold</div><div><code>{esc(cfg.get('similarity_threshold'))}</code></div></div>
    <div class="card"><div class="muted">category</div><div><code>{esc(cfg.get('category'))}</code></div></div>
  </div>

  <h2>Examples (top / anti-top)</h2>
  <div class="grid">
    {''.join([f'''<div class="card">
      <div><strong>frame_index</strong>: <code>{esc(e.get("frame_index"))}</code></div>
      <div><strong>time</strong>: <code>{esc(e.get("time_sec")):.1f}s</code></div>
      <div><strong>top1</strong>: <code>{esc(e.get("top1_face_name"))}</code> <span class="muted">({esc(e.get("top1_score"))})</span></div>
      <div><strong>confident</strong>: <code>{esc(e.get("is_confident"))}</code></div>
      {f'<div style="margin-top:8px"><img class="thumb" src="assets/{esc(e.get("asset"))}" alt="face crop"></div>' if e.get("asset") else '<div class="muted" style="margin-top:8px">asset not available</div>'}
    </div>''' for e in examples[:20]])}
  </div>

  <h2>Top faces (by frame top-1 count)</h2>
  <div class="card">
    <table>
      <thead><tr><th>name</th><th>count</th><th>avg_score</th><th>max_score</th><th>min_score</th><th>face_id</th></tr></thead>
      <tbody>
        {''.join([f"<tr><td>{esc(b.get('face_name'))}</td><td>{esc(b.get('count'))}</td><td>{esc(b.get('avg_score')):.3f}</td><td>{esc(b.get('max_score')):.3f}</td><td>{esc(b.get('min_score')):.3f}</td><td>{esc(b.get('face_id'))}</td></tr>" for b in top_faces[:50]])}
      </tbody>
    </table>
  </div>

  <h2>Timeline (first 3000 frames)</h2>
  <div class="card">
    <input id="q" placeholder="filter by face name...">
    <table id="tl">
      <thead><tr><th>pos</th><th>frame_index</th><th>time_s</th><th>top1_name</th><th>top1_score</th><th>confident</th></tr></thead>
      <tbody>
        {''.join([f"<tr><td>{esc(r.get('pos'))}</td><td>{esc(r.get('frame_index'))}</td><td>{esc(r.get('time_sec')):.2f}</td><td>{esc(r.get('top1_face_name') or '')}</td><td>{esc(r.get('top1_score') or 0.0):.3f}</td><td>{esc(r.get('is_confident', False))}</td></tr>" for r in timeline])}
      </tbody>
    </table>
  </div>

  <h2>How to QA</h2>
  <div class="card">
    <ul>
      {''.join([f"<li>{esc(x)}</li>" for x in (render.get("qa_hints", []) or [])])}
    </ul>
    <div class="muted">Tip: if assets are missing, ensure VisualProcessor renderer passed frames_dir/assets_dir.</div>
  </div>
</div>
<script>
  const q = document.getElementById('q');
  const rows = Array.from(document.querySelectorAll('#tl tbody tr'));
  q.addEventListener('input', () => {{
    const needle = (q.value || '').toLowerCase();
    for (const r of rows) {{
      const name = (r.children[3].textContent || '').toLowerCase();
      r.style.display = (!needle || name.includes(needle)) ? '' : 'none';
    }}
  }});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved Face Identity HTML render to {output_path}")
    return output_path


__all__ = ["render_face_identity", "render_face_identity_html"]
