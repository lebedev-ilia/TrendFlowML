"""
Audit v3 renderer for brand_semantics.

- `render_brand_semantics(...)` builds human-friendly render_context JSON (saved by VisualProcessor).
- `render_brand_semantics_html(...)` builds offline `render.html` (NO CDN) + optional assets in `_render/assets/`.
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

from crop_utils import crop_with_padding


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


def _maybe_write_track_crop(
    *,
    frames_dir: Optional[str],
    assets_dir: Optional[str],
    frame_indices: np.ndarray,
    track_id: int,
    track_best_frame_pos: int,
    track_best_bbox_xyxy: np.ndarray,
    pad_ratio: float,
) -> Optional[str]:
    if not frames_dir or not assets_dir:
        return None
    if cv2 is None or FrameManager is None:
        return None
    if track_best_frame_pos < 0 or track_best_frame_pos >= int(frame_indices.shape[0]):
        return None
    if not np.all(np.isfinite(track_best_bbox_xyxy)):
        return None

    frame_idx_global = int(frame_indices[int(track_best_frame_pos)])
    fn = f"track_{int(track_id)}_frame_{frame_idx_global}.jpg"
    out_path = os.path.join(assets_dir, fn)
    if os.path.isfile(out_path):
        return fn

    try:
        fm = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
        frame = fm.get(frame_idx_global)
        crop = crop_with_padding(frame, tuple(track_best_bbox_xyxy.tolist()), pad_ratio=float(pad_ratio))
        os.makedirs(assets_dir, exist_ok=True)
        ok = cv2.imwrite(out_path, crop)
        try:
            fm.close()
        except Exception:
            pass
        return fn if ok else None
    except Exception:
        return None


def render_brand_semantics(
    npz_data: Dict[str, Any],
    meta: Dict[str, Any],
    render_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Render-context for brand_semantics (Audit v3).

    If render_env contains `frames_dir` + `assets_dir`, we also emit a few example crop assets.
    """
    render_env = render_env or {}
    frames_dir = render_env.get("frames_dir")
    assets_dir = render_env.get("assets_dir")

    frame_indices = _np(npz_data.get("frame_indices"), np.int32)
    times_s = _np(npz_data.get("times_s"), np.float32)

    track_ids = _np(npz_data.get("track_ids"), np.int32)
    track_present_mask = _np(npz_data.get("track_present_mask"), bool)
    track_topk_ids = _np(npz_data.get("track_topk_ids"), np.int32)
    track_topk_scores = _np(npz_data.get("track_topk_scores"), np.float32)
    track_is_confident_top1 = _np(npz_data.get("track_is_confident_top1"), bool)

    track_best_frame_pos = _np(npz_data.get("track_best_frame_pos"), np.int32)
    track_best_bbox_xyxy = _np(npz_data.get("track_best_bbox_xyxy"), np.float32)

    frame_topk_ids = _np(npz_data.get("frame_topk_ids"), np.int32)
    frame_topk_scores = _np(npz_data.get("frame_topk_scores"), np.float32)
    frame_is_confident_top1 = _np(npz_data.get("frame_is_confident_top1"), bool)

    semantic_label_names = _np(npz_data.get("semantic_label_names"), "U")
    semantic_object_ids = _np(npz_data.get("semantic_object_ids"), "U")

    id_to_name, id_to_uuid = _parse_label_space(semantic_label_names, semantic_object_ids)

    n_frames = int(frame_indices.shape[0]) if frame_indices is not None else 0
    n_tracks = int(track_ids.shape[0]) if track_ids is not None else 0
    n_tracks_present = int(np.sum(track_present_mask)) if track_present_mask is not None else 0
    n_labels = int(semantic_label_names.shape[0]) if semantic_label_names is not None else 0

    # Frame score stats
    top1_frame_scores = None
    if frame_topk_scores is not None and frame_topk_scores.ndim == 2 and frame_topk_scores.shape[1] >= 1:
        top1_frame_scores = frame_topk_scores[:, 0]
    elif frame_topk_scores is not None and frame_topk_scores.ndim == 1:
        top1_frame_scores = frame_topk_scores

    # Top brands by frame top-1 count
    top_brands: List[Dict[str, Any]] = []
    if frame_topk_ids is not None and frame_topk_ids.ndim == 2 and frame_topk_ids.shape[1] >= 1:
        top1_ids = frame_topk_ids[:, 0]
        counts: Dict[int, int] = {}
        for lid in top1_ids.tolist():
            li = int(lid)
            if li < 0:
                continue
            counts[li] = counts.get(li, 0) + 1
        for lid, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            top_brands.append(
                {
                    "label_id": int(lid),
                    "name": id_to_name.get(int(lid), "unknown"),
                    "object_id": id_to_uuid.get(int(lid), ""),
                    "frames_count_top1": int(cnt),
                }
            )

    # Examples: top / anti-top tracks by top1 score
    examples: List[Dict[str, Any]] = []
    if (
        track_ids is not None
        and track_present_mask is not None
        and track_topk_scores is not None
        and track_topk_scores.ndim == 2
        and track_topk_scores.shape[1] >= 1
    ):
        top1_track_scores = track_topk_scores[:, 0]
        items = []
        for i in range(n_tracks):
            if not bool(track_present_mask[i]):
                continue
            sc = float(top1_track_scores[i])
            if not np.isfinite(sc):
                continue
            items.append((i, sc))
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        picks = items_sorted[:6] + list(reversed(items_sorted[-6:])) if items_sorted else []

        pad_ratio = float(meta.get("pad_ratio", 0.15) or 0.15)
        for i, sc in picks:
            tid = int(track_ids[i])
            top1_lid = int(track_topk_ids[i, 0]) if track_topk_ids is not None else -1
            top1_name = id_to_name.get(top1_lid, "unknown") if top1_lid >= 0 else None
            asset = None
            if (
                frame_indices is not None
                and track_best_frame_pos is not None
                and track_best_bbox_xyxy is not None
                and i < int(track_best_frame_pos.shape[0])
                and i < int(track_best_bbox_xyxy.shape[0])
            ):
                asset = _maybe_write_track_crop(
                    frames_dir=str(frames_dir) if frames_dir else None,
                    assets_dir=str(assets_dir) if assets_dir else None,
                    frame_indices=frame_indices,
                    track_id=tid,
                    track_best_frame_pos=int(track_best_frame_pos[i]),
                    track_best_bbox_xyxy=track_best_bbox_xyxy[i, :],
                    pad_ratio=pad_ratio,
                )
            examples.append(
                {
                    "track_pos": int(i),
                    "track_id": tid,
                    "top1_label_id": int(top1_lid) if top1_lid >= 0 else None,
                    "top1_name": top1_name,
                    "top1_score": float(sc),
                    "is_confident_top1": bool(track_is_confident_top1[i]) if track_is_confident_top1 is not None else False,
                    "asset": asset,
                }
            )

    # Timeline rows (compact)
    timeline: List[Dict[str, Any]] = []
    if frame_indices is not None and times_s is not None and frame_topk_scores is not None and frame_topk_ids is not None:
        for i in range(min(n_frames, 3000)):  # cap for huge runs
            lid = int(frame_topk_ids[i, 0]) if frame_topk_ids.ndim == 2 and frame_topk_ids.shape[1] >= 1 else -1
            sc = float(frame_topk_scores[i, 0]) if frame_topk_scores.ndim == 2 and frame_topk_scores.shape[1] >= 1 else float("nan")
            timeline.append(
                {
                    "pos": int(i),
                    "frame_index": int(frame_indices[i]),
                    "time_s": float(times_s[i]),
                    "top1_label_id": int(lid) if lid >= 0 else None,
                    "top1_name": id_to_name.get(int(lid), "unknown") if lid >= 0 else None,
                    "top1_score": _safe_float(sc),
                    "is_confident_top1": bool(frame_is_confident_top1[i]) if frame_is_confident_top1 is not None else False,
                }
            )

    render = {
        "component": "brand_semantics",
        "summary": {
            "frames": n_frames,
            "tracks_total": n_tracks,
            "tracks_present": n_tracks_present,
            "labels": n_labels,
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
            "proposal_classes": meta.get("proposal_classes"),
            "confidence_threshold_top1": meta.get("confidence_threshold_top1"),
            "pad_ratio": meta.get("pad_ratio"),
            "use_sharpness": meta.get("use_sharpness"),
            "max_tracks": meta.get("max_tracks"),
            "max_dets_per_track": meta.get("max_dets_per_track"),
        },
        "qa_hints": [
            "Проверьте, что top/anti-top примеры визуально похожи на логотипы/бренды, а не на шум (фон, текстуры).",
            "Если много confident=false, возможно порог confidence_threshold_top1 слишком высокий или база маленькая.",
            "Если много confident=true при явном мусоре — порог слишком низкий или proposals слишком широкие (proposal_classes).",
        ],
        "distributions": {
            "frame_top1_scores": _compute_score_stats(top1_frame_scores) if top1_frame_scores is not None else {},
        },
        "top_brands": top_brands,
        "examples": examples,
        "timeline": timeline,
    }
    return render


def render_brand_semantics_html(
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
    render = render_brand_semantics(
        {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v) for k, v in npz_data.items()},
        meta,
        {"frames_dir": frames_dir, "assets_dir": assets_dir},
    )

    examples = render.get("examples", []) or []
    timeline = render.get("timeline", []) or []
    top_brands = render.get("top_brands", []) or []
    key_facts = render.get("key_facts", {}) or {}
    cfg = render.get("config_highlights", {}) or {}

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
  <title>brand_semantics — Audit v3 render</title>
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
  <h1>brand_semantics <span class="pill {esc(render['summary']['status'])}">{esc(render['summary']['status'])}</span></h1>
  <p class="muted">NPZ is source-of-truth. This page is dev-only and works offline.</p>

  <h2>Key facts</h2>
  <div class="grid">
    <div class="card"><div class="muted">schema_version</div><div><code>{esc(key_facts.get('schema_version'))}</code></div></div>
    <div class="card"><div class="muted">producer_version</div><div><code>{esc(key_facts.get('producer_version'))}</code></div></div>
    <div class="card"><div class="muted">db_digest</div><div><code>{esc(key_facts.get('db_digest'))}</code></div></div>
    <div class="card"><div class="muted">embedding_model</div><div><code>{esc(key_facts.get('embedding_model'))}</code></div></div>
  </div>

  <h2>Config highlights</h2>
  <div class="grid">
    <div class="card"><div class="muted">proposal_classes</div><div><code>{esc(cfg.get('proposal_classes'))}</code></div></div>
    <div class="card"><div class="muted">confidence_threshold_top1</div><div><code>{esc(cfg.get('confidence_threshold_top1'))}</code></div></div>
    <div class="card"><div class="muted">pad_ratio</div><div><code>{esc(cfg.get('pad_ratio'))}</code></div></div>
    <div class="card"><div class="muted">use_sharpness</div><div><code>{esc(cfg.get('use_sharpness'))}</code></div></div>
  </div>

  <h2>Examples (top / anti-top)</h2>
  <div class="grid">
    {''.join([f'''<div class="card">
      <div><strong>track_id</strong>: <code>{esc(e.get("track_id"))}</code></div>
      <div><strong>top1</strong>: <code>{esc(e.get("top1_name"))}</code> <span class="muted">({esc(e.get("top1_score"))})</span></div>
      <div><strong>confident</strong>: <code>{esc(e.get("is_confident_top1"))}</code></div>
      {f'<div style="margin-top:8px"><img class="thumb" src="assets/{esc(e.get("asset"))}" alt="crop"></div>' if e.get("asset") else '<div class="muted" style="margin-top:8px">asset not available</div>'}
    </div>''' for e in examples])}
  </div>

  <h2>Top brands (by frame top-1 count)</h2>
  <div class="card">
    <table>
      <thead><tr><th>name</th><th>frames_count_top1</th><th>label_id</th><th>object_id</th></tr></thead>
      <tbody>
        {''.join([f"<tr><td>{esc(b.get('name'))}</td><td>{esc(b.get('frames_count_top1'))}</td><td>{esc(b.get('label_id'))}</td><td><code>{esc(b.get('object_id'))}</code></td></tr>" for b in top_brands[:50]])}
      </tbody>
    </table>
  </div>

  <h2>Timeline (first 3000 frames)</h2>
  <div class="card">
    <input id="q" placeholder="filter by brand name...">
    <table id="tl">
      <thead><tr><th>pos</th><th>frame_index</th><th>time_s</th><th>top1_name</th><th>top1_score</th><th>confident</th></tr></thead>
      <tbody>
        {''.join([f"<tr><td>{esc(r.get('pos'))}</td><td>{esc(r.get('frame_index'))}</td><td>{esc(r.get('time_s'))}</td><td>{esc(r.get('top1_name'))}</td><td>{esc(r.get('top1_score'))}</td><td>{esc(r.get('is_confident_top1'))}</td></tr>" for r in timeline])}
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
    logger.info(f"Saved Brand Semantics HTML render to {output_path}")
    return output_path


__all__ = ["render_brand_semantics", "render_brand_semantics_html"]

