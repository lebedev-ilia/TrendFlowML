"""
Audit v3 renderer for car_semantics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

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

    try:
        fm = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
        fr_idx = int(frame_indices[int(track_best_frame_pos)])
        frame = fm.get(fr_idx)
        fm.close()
    except Exception:
        return None

    try:
        crop = crop_with_padding(
            frame,
            tuple(map(float, track_best_bbox_xyxy.tolist())),
            pad_ratio=float(pad_ratio),
        )
    except Exception:
        return None

    rel_name = f"track_{int(track_id):06d}.jpg"
    out_path = os.path.join(assets_dir, rel_name)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = bool(cv2.imwrite(out_path, crop))
        if not ok:
            return None
        return os.path.join("assets", rel_name)
    except Exception:
        return None


def render_car_semantics(
    npz_data: Dict[str, Any],
    meta: Dict[str, Any],
    render_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build render_context JSON.
    """
    frames_dir = str((render_env or {}).get("frames_dir") or "") or None
    assets_dir = str((render_env or {}).get("assets_dir") or "") or None

    frame_indices = _np(npz_data.get("frame_indices"), np.int32)
    times_s = _np(npz_data.get("times_s"), np.float32)

    semantic_label_names = _np(npz_data.get("semantic_label_names"), "U")
    semantic_object_ids = _np(npz_data.get("semantic_object_ids"), "U")
    id_to_name, id_to_uuid = _parse_label_space(semantic_label_names, semantic_object_ids)

    track_ids = _np(npz_data.get("track_ids"), np.int32)
    track_topk_ids = _np(npz_data.get("track_topk_ids"), np.int32)
    track_topk_scores = _np(npz_data.get("track_topk_scores"), np.float32)
    track_is_confident_top1 = _np(npz_data.get("track_is_confident_top1"), bool)

    frame_topk_ids = _np(npz_data.get("frame_topk_ids"), np.int32)
    frame_topk_scores = _np(npz_data.get("frame_topk_scores"), np.float32)
    frame_is_confident_top1 = _np(npz_data.get("frame_is_confident_top1"), bool)

    track_best_frame_pos = _np(npz_data.get("track_best_frame_pos"), np.int32)
    track_best_bbox_xyxy = _np(npz_data.get("track_best_bbox_xyxy"), np.float32)

    render: Dict[str, Any] = {
        "component": "car_semantics",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "top_labels": [],
        "examples": {"top": [], "anti_top": []},
    }

    N = int(frame_indices.shape[0]) if frame_indices is not None else 0
    T = int(track_ids.shape[0]) if track_ids is not None else 0
    A = int(semantic_label_names.shape[0]) if semantic_label_names is not None else 0

    render["summary"] = {
        "frames_count": N,
        "tracks_count": T,
        "labels_count": A,
        "status": str(meta.get("status") or "unknown"),
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "db_digest_prefix": str(meta.get("db_digest") or "")[:12],
        "confidence_threshold_top1": float(meta.get("confidence_threshold_top1") or 0.0) if meta else 0.0,
        "proposal_classes": meta.get("proposal_classes"),
        "stage_timings_ms": meta.get("stage_timings_ms"),
    }

    # Timeline (frame-level)
    if times_s is not None and frame_topk_ids is not None and frame_topk_scores is not None:
        for i in range(int(min(times_s.shape[0], frame_topk_scores.shape[0]))):
            top1_id = int(frame_topk_ids[i, 0]) if frame_topk_ids.shape[1] > 0 else -1
            top1_score = float(frame_topk_scores[i, 0]) if frame_topk_scores.shape[1] > 0 else float("nan")
            name = id_to_name.get(top1_id, "") if top1_id >= 0 else ""
            render["timeline"].append(
                {
                    "pos": int(i),
                    "frame_idx": int(frame_indices[i]) if frame_indices is not None else int(i),
                    "time_s": float(times_s[i]),
                    "top1_id": int(top1_id),
                    "top1_name": str(name),
                    "top1_uuid": str(id_to_uuid.get(top1_id, "")),
                    "top1_score": None if not np.isfinite(top1_score) else float(top1_score),
                    "confident": bool(frame_is_confident_top1[i]) if frame_is_confident_top1 is not None else False,
                }
            )

    # Distributions
    if track_topk_scores is not None and track_topk_scores.size > 0:
        render["distributions"]["track_top1_scores"] = _compute_score_stats(track_topk_scores[:, 0])
        render["distributions"]["track_topk_scores"] = _compute_score_stats(track_topk_scores.reshape(-1))
    if frame_topk_scores is not None and frame_topk_scores.size > 0:
        render["distributions"]["frame_top1_scores"] = _compute_score_stats(frame_topk_scores[:, 0])
        render["distributions"]["frame_topk_scores"] = _compute_score_stats(frame_topk_scores.reshape(-1))

    # Top labels by confident top1 frames
    if frame_topk_ids is not None and frame_is_confident_top1 is not None:
        counts: Dict[int, int] = {}
        for i in range(int(frame_topk_ids.shape[0])):
            if not bool(frame_is_confident_top1[i]):
                continue
            lid = int(frame_topk_ids[i, 0])
            if lid < 0:
                continue
            counts[lid] = int(counts.get(lid, 0) + 1)
        top = sorted(counts.items(), key=lambda x: int(x[1]), reverse=True)[:15]
        render["top_labels"] = [
            {
                "label_id": int(lid),
                "label_name": str(id_to_name.get(int(lid), "")),
                "label_uuid": str(id_to_uuid.get(int(lid), "")),
                "confident_frames": int(cnt),
            }
            for lid, cnt in top
        ]

    # Examples (top / anti-top) from track_top1 scores
    if track_ids is not None and track_topk_ids is not None and track_topk_scores is not None:
        top1 = track_topk_scores[:, 0].astype(np.float32, copy=False)
        finite_idx = np.where(np.isfinite(top1))[0].astype(int)
        if finite_idx.size > 0:
            order_desc = finite_idx[np.argsort(-top1[finite_idx])]
            for idx in order_desc[:8].tolist():
                tid = int(track_ids[idx])
                lid = int(track_topk_ids[idx, 0])
                score = float(track_topk_scores[idx, 0])
                asset = None
                if (
                    frame_indices is not None
                    and track_best_frame_pos is not None
                    and track_best_bbox_xyxy is not None
                ):
                    asset = _maybe_write_track_crop(
                        frames_dir=frames_dir,
                        assets_dir=assets_dir,
                        frame_indices=frame_indices,
                        track_id=tid,
                        track_best_frame_pos=int(track_best_frame_pos[idx]),
                        track_best_bbox_xyxy=track_best_bbox_xyxy[idx],
                        pad_ratio=float(meta.get("pad_ratio") or 0.15),
                    )
                render["examples"]["top"].append(
                    {
                        "track_id": tid,
                        "label_id": lid,
                        "label_name": str(id_to_name.get(lid, "")),
                        "score": score,
                        "confident": bool(track_is_confident_top1[idx]) if track_is_confident_top1 is not None else False,
                        "asset_path": asset,
                    }
                )

            order_asc = finite_idx[np.argsort(top1[finite_idx])]
            for idx in order_asc[:8].tolist():
                tid = int(track_ids[idx])
                lid = int(track_topk_ids[idx, 0])
                score = float(track_topk_scores[idx, 0])
                asset = None
                if (
                    frame_indices is not None
                    and track_best_frame_pos is not None
                    and track_best_bbox_xyxy is not None
                ):
                    asset = _maybe_write_track_crop(
                        frames_dir=frames_dir,
                        assets_dir=assets_dir,
                        frame_indices=frame_indices,
                        track_id=tid,
                        track_best_frame_pos=int(track_best_frame_pos[idx]),
                        track_best_bbox_xyxy=track_best_bbox_xyxy[idx],
                        pad_ratio=float(meta.get("pad_ratio") or 0.15),
                    )
                render["examples"]["anti_top"].append(
                    {
                        "track_id": tid,
                        "label_id": lid,
                        "label_name": str(id_to_name.get(lid, "")),
                        "score": score,
                        "confident": bool(track_is_confident_top1[idx]) if track_is_confident_top1 is not None else False,
                        "asset_path": asset,
                    }
                )

    return render


def render_car_semantics_html(
    npz_path: str,
    output_path: str,
    *,
    frames_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
) -> str:
    """
    Offline HTML render (NO CDN). Prefers `render_context.json` in the same directory.
    """
    render_dir = os.path.dirname(os.path.abspath(output_path))
    ctx_path = os.path.join(render_dir, "render_context.json")
    ctx: Dict[str, Any]
    if os.path.exists(ctx_path):
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    else:
        z = np.load(npz_path, allow_pickle=True)
        npz_data = {k: z[k] for k in z.files}
        meta_obj = npz_data.get("meta")
        meta = (
            meta_obj.item()
            if isinstance(meta_obj, np.ndarray) and meta_obj.dtype == object and meta_obj.shape == ()
            else {}
        )
        ctx = render_car_semantics(npz_data, meta, {"frames_dir": frames_dir, "assets_dir": assets_dir})

    summary = ctx.get("summary") or {}
    timeline = ctx.get("timeline") or []
    top_labels = ctx.get("top_labels") or []
    ex_top = ((ctx.get("examples") or {}).get("top") or [])
    ex_anti = ((ctx.get("examples") or {}).get("anti_top") or [])

    # Simple offline plot: SVG polyline for frame top-1 scores
    scores = [t.get("top1_score") for t in timeline]
    scores_num = [float(s) for s in scores if isinstance(s, (int, float))]
    smin = min(scores_num) if scores_num else 0.0
    smax = max(scores_num) if scores_num else 1.0
    W, H = 1000, 180
    pts = []
    for i, t in enumerate(timeline):
        s = t.get("top1_score")
        if not isinstance(s, (int, float)):
            continue
        x = 0 if len(timeline) <= 1 else int(round((i / (len(timeline) - 1)) * (W - 1)))
        denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
        y = int(round((1.0 - ((float(s) - smin) / denom)) * (H - 1)))
        pts.append(f"{x},{y}")
    svg = (
        f"<svg viewBox='0 0 {W} {H}' width='100%' height='{H}' "
        f"style='background:#0b1020;border:1px solid #1f2a44;border-radius:8px'>"
        + (f"<polyline fill='none' stroke='#4fd1c5' stroke-width='2' points='{' '.join(pts)}'/>" if pts else "")
        + "</svg>"
    )

    def _esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _img(asset_path: Any) -> str:
        if not asset_path:
            return ""
        return f"<img class='thumb' src='{_esc(asset_path)}' loading='lazy' />"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>car_semantics (offline render)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, Arial; background:#0b1020; color:#e8eefc; margin:0; }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:24px; }}
    a {{ color:#7aa2ff; }}
    .card {{ background:#111a33; border:1px solid #1f2a44; border-radius:12px; padding:16px; margin:16px 0; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
    .k {{ color:#a9b8e6; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .v {{ font-size:18px; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ padding:8px; border-bottom:1px solid #1f2a44; text-align:left; font-size:14px; }}
    th {{ position:sticky; top:0; background:#111a33; }}
    .thumb {{ width:180px; height:auto; border-radius:10px; border:1px solid #1f2a44; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #1f2a44; font-size:12px; color:#a9b8e6; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>car_semantics — offline mini-dashboard</h1>
    <div class="card">
      <div class="grid">
        <div><div class="k">status</div><div class="v">{_esc(summary.get("status"))}</div></div>
        <div><div class="k">schema_version</div><div class="v">{_esc(summary.get("schema_version"))}</div></div>
        <div><div class="k">producer_version</div><div class="v">{_esc(summary.get("producer_version"))}</div></div>
        <div><div class="k">frames</div><div class="v">{_esc(summary.get("frames_count"))}</div></div>
        <div><div class="k">tracks</div><div class="v">{_esc(summary.get("tracks_count"))}</div></div>
        <div><div class="k">labels</div><div class="v">{_esc(summary.get("labels_count"))}</div></div>
        <div><div class="k">db_digest</div><div class="v"><span class="pill">{_esc(summary.get("db_digest_prefix"))}</span></div></div>
        <div><div class="k">conf_th_top1</div><div class="v">{_esc(summary.get("confidence_threshold_top1"))}</div></div>
      </div>
    </div>

    <div class="card">
      <h2>Timeline (frame top-1 score)</h2>
      {svg}
      <div class="k" style="margin-top:8px">NPZ остаётся source-of-truth; это dev-only render.</div>
    </div>

    <div class="card">
      <h2>Top labels (by confident top-1 frames)</h2>
      <table>
        <thead><tr><th>label</th><th>uuid</th><th>confident_frames</th></tr></thead>
        <tbody>
          {"".join([f"<tr><td>{_esc(r.get('label_name'))}</td><td>{_esc(r.get('label_uuid'))}</td><td>{_esc(r.get('confident_frames'))}</td></tr>" for r in top_labels])}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Examples (Top / Anti-top tracks by top-1 score)</h2>
      <h3>Top</h3>
      <table>
        <thead><tr><th>crop</th><th>track_id</th><th>label</th><th>score</th><th>confident</th></tr></thead>
        <tbody>
          {"".join([f"<tr><td>{_img(r.get('asset_path'))}</td><td>{_esc(r.get('track_id'))}</td><td>{_esc(r.get('label_name'))}</td><td>{_esc(r.get('score'))}</td><td>{_esc(r.get('confident'))}</td></tr>" for r in ex_top])}
        </tbody>
      </table>
      <h3 style="margin-top:16px">Anti-top</h3>
      <table>
        <thead><tr><th>crop</th><th>track_id</th><th>label</th><th>score</th><th>confident</th></tr></thead>
        <tbody>
          {"".join([f"<tr><td>{_img(r.get('asset_path'))}</td><td>{_esc(r.get('track_id'))}</td><td>{_esc(r.get('label_name'))}</td><td>{_esc(r.get('score'))}</td><td>{_esc(r.get('confident'))}</td></tr>" for r in ex_anti])}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("car_semantics | html render saved: %s", output_path)
    return output_path


