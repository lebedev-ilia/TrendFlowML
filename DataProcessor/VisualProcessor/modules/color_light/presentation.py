from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _safe_list(values: List[float]) -> List[float]:
    out = []
    for v in values:
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(float("nan"))
            else:
                out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


def _histogram(values: List[float], bins: int = 10, vmin: float = 0.0, vmax: float = 1.0) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if isinstance(v, (int, float)) and not math.isnan(v)], dtype=np.float32)
    if arr.size == 0:
        return {"bins": bins, "range": [vmin, vmax], "counts": [0] * bins}
    hist, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    return {
        "bins": bins,
        "range": [vmin, vmax],
        "counts": [int(x) for x in hist.tolist()],
        "edges": [float(e) for e in edges.tolist()],
    }


def build_presentation(results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    frames = results.get("frames", {}) or {}
    video_features = results.get("video_features", {}) or {}
    seq_indices = results.get("sequence_frame_indices") or []
    seq_times = results.get("sequence_times_s") or []

    # Build frame feature map: frame_idx -> features dict
    frame_feat_map: Dict[int, Dict[str, Any]] = {}
    for scene_dict in frames.values():
        for frame_idx, frame_obj in (scene_dict or {}).items():
            feat = (frame_obj or {}).get("features", {})
            try:
                frame_feat_map[int(frame_idx)] = feat
            except Exception:
                continue

    # Ordered frame features
    ordered_features = []
    for idx in seq_indices:
        feat = frame_feat_map.get(int(idx), {})
        ordered_features.append(feat)

    hue_vals = _safe_list([f.get("hue_mean_norm", float("nan")) for f in ordered_features])
    colorfulness_vals = _safe_list([f.get("colorfulness_norm", float("nan")) for f in ordered_features])
    brightness_vals = _safe_list([
        (f.get("brightness_mean", float("nan")) / 255.0) if isinstance(f.get("brightness_mean"), (int, float)) else float("nan")
        for f in ordered_features
    ])
    contrast_vals = _safe_list([f.get("global_contrast_norm", float("nan")) for f in ordered_features])

    presentation = {
        "schema_version": "color_light_presentation_v1",
        "run_identity": {
            "platform_id": metadata.get("platform_id"),
            "video_id": metadata.get("video_id"),
            "run_id": metadata.get("run_id"),
            "config_hash": metadata.get("config_hash"),
            "sampling_policy_version": metadata.get("sampling_policy_version"),
        },
        "summary": {
            "color_distribution_entropy": video_features.get("color_distribution_entropy"),
            "color_distribution_gini": video_features.get("color_distribution_gini"),
            "cinematic_lighting_score": video_features.get("cinematic_lighting_score"),
            "professional_look_score": video_features.get("professional_look_score"),
        },
        "style_probs": {k: v for k, v in video_features.items() if k.startswith("style_")},
        "video_features": video_features,
        "timeline": {
            "times_s": [float(t) for t in seq_times] if isinstance(seq_times, (list, np.ndarray)) else [],
            "hue_mean_norm": hue_vals,
            "colorfulness_norm": colorfulness_vals,
            "brightness_mean_norm": brightness_vals,
            "global_contrast_norm": contrast_vals,
        },
        "distributions": {
            "hue_mean_norm_hist": _histogram(hue_vals, bins=12, vmin=0.0, vmax=1.0),
            "colorfulness_norm_hist": _histogram(colorfulness_vals, bins=12, vmin=0.0, vmax=1.0),
            "brightness_mean_norm_hist": _histogram(brightness_vals, bins=12, vmin=0.0, vmax=1.0),
        },
    }
    return presentation


def _sparkline_svg(values: List[float], width: int = 600, height: int = 120) -> str:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return f'<svg width="{width}" height="{height}"></svg>'
    vmin = min(vals)
    vmax = max(vals)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    points = []
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)) or math.isnan(v):
            continue
        x = int((i / max(1, len(values) - 1)) * (width - 4)) + 2
        y = int((1.0 - (v - vmin) / (vmax - vmin)) * (height - 4)) + 2
        points.append(f"{x},{y}")
    pts = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="#2E7DFF" stroke-width="2" points="{pts}" />'
        "</svg>"
    )


def write_presentation(
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    base_dir: str
) -> Tuple[str, str]:
    base = Path(base_dir)
    run_id = str(metadata.get("run_id") or "unknown_run")
    out_dir = base / "color_light" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_presentation(results, metadata)

    json_path = out_dir / "color_light_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    timeline = data.get("timeline", {})
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Color & Light — Presentation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 6px; }}
    .section {{ margin-top: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; }}
    .mono {{ font-family: monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Color & Light — Presentation</h1>
  <div class="mono">run_id: {data.get("run_identity", {}).get("run_id")}</div>

  <div class="section grid">
    <div class="card">
      <h3>Hue mean (norm)</h3>
      {_sparkline_svg(timeline.get("hue_mean_norm", []))}
    </div>
    <div class="card">
      <h3>Colorfulness (norm)</h3>
      {_sparkline_svg(timeline.get("colorfulness_norm", []))}
    </div>
    <div class="card">
      <h3>Brightness mean (norm)</h3>
      {_sparkline_svg(timeline.get("brightness_mean_norm", []))}
    </div>
    <div class="card">
      <h3>Global contrast (norm)</h3>
      {_sparkline_svg(timeline.get("global_contrast_norm", []))}
    </div>
  </div>

  <div class="section card">
    <h3>Style probabilities</h3>
    <pre class="mono">{json.dumps(data.get("style_probs", {}), ensure_ascii=False, indent=2)}</pre>
  </div>

  <div class="section card">
    <h3>Summary</h3>
    <pre class="mono">{json.dumps(data.get("summary", {}), ensure_ascii=False, indent=2)}</pre>
  </div>
</body>
</html>
"""
    html_path = out_dir / "color_light_summary.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(json_path), str(html_path)

