"""
Renderer for rhythmic_extractor (Audit v3): render-context JSON + offline HTML (vanilla canvas, no CDN).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def _to_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size > 0 else []
    if isinstance(v, list):
        return v
    return []


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def render_rhythmic_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Generate render-context for rhythmic_extractor (Audit v3)."""
    render: Dict[str, Any] = {
        "component": "rhythmic_extractor",
        "summary": {},
        "model_facing": {},
        "analytics": {},
        "beat_events": {
            "beat_times_sec": [],
            "beat_times_sec_npy": meta.get("beat_times_sec_npy") if isinstance(meta, dict) else None,
            "beat_segment_index": [],
            "beat_segment_index_npy": meta.get("beat_segment_index_npy") if isinstance(meta, dict) else None,
        },
        "canonical_axis": {},
        "distributions": {},
    }

    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()

    feats: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            feats[str(name)] = feature_values[i]

    render["summary"] = {
        "status": meta.get("status") if isinstance(meta, dict) else None,
        "schema_version": meta.get("schema_version") if isinstance(meta, dict) else None,
        "producer_version": meta.get("producer_version") if isinstance(meta, dict) else None,
        "backend": meta.get("backend") if isinstance(meta, dict) else None,
        "hop_length": meta.get("hop_length") if isinstance(meta, dict) else None,
        "sampling_family_used": meta.get("sampling_family_used") if isinstance(meta, dict) else None,
    }

    # Frozen model-facing subset (tabular)
    for k in [
        "rhythm_tempo_bpm",
        "rhythm_beats_count",
        "rhythm_beat_density",
        "rhythm_regularity",
        "rhythm_tempo_variation",
        "rhythm_beat_consistency",
        "duration_sec",
        "sample_rate",
        "segments_count",
    ]:
        if k in feats:
            render["model_facing"][k] = _safe_float(feats.get(k))

    # Analytics scalars (stored as dedicated NPZ keys)
    for k in [
        "rhythm_avg_period_sec",
        "rhythm_period_std_sec",
        "rhythm_median_period_sec",
        "rhythm_min_period_sec",
        "rhythm_max_period_sec",
        "rhythm_median_bpm",
        "rhythm_ibi_tempo_bpm",
        "rhythm_tempo_mean",
        "rhythm_tempo_std",
        "rhythm_tempo_min",
        "rhythm_tempo_max",
        "rhythm_syncopation_score",
        "rhythm_polyrhythm_score",
        "rhythm_beat_strength_mean",
        "rhythm_beat_strength_std",
        "rhythm_metrical_stability",
    ]:
        if k in npz_data:
            render["analytics"][k] = _safe_float(npz_data.get(k))

    render["canonical_axis"] = {
        "segment_start_sec": _to_list(npz_data.get("segment_start_sec")),
        "segment_end_sec": _to_list(npz_data.get("segment_end_sec")),
        "segment_center_sec": _to_list(npz_data.get("segment_center_sec")),
        "segment_mask": _to_list(npz_data.get("segment_mask")),
    }

    render["beat_events"]["beat_times_sec"] = _to_list(npz_data.get("beat_times_sec"))
    render["beat_events"]["beat_segment_index"] = _to_list(npz_data.get("beat_segment_index"))

    bt = render["beat_events"]["beat_times_sec"]
    if bt and len(bt) >= 2:
        intervals = np.diff(np.asarray(bt, dtype=np.float32))
        intervals = intervals[np.isfinite(intervals)]
        if intervals.size > 0:
            render["distributions"]["intervals"] = {
                "min": float(np.min(intervals)),
                "max": float(np.max(intervals)),
                "mean": float(np.mean(intervals)),
                "std": float(np.std(intervals)),
                "median": float(np.median(intervals)),
            }

    return render


def render_rhythmic_extractor_html(npz_path: str, output_path: str) -> str:
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_rhythmic_extractor(npz_data, meta)

    npz_dir = Path(npz_path).parent

    beat_times = render.get("beat_events", {}).get("beat_times_sec") or []
    beat_times_npy = render.get("beat_events", {}).get("beat_times_sec_npy")
    if (not beat_times) and isinstance(beat_times_npy, str) and beat_times_npy:
        try:
            p = Path(beat_times_npy)
            if not p.is_absolute():
                p = npz_dir / beat_times_npy
            if p.exists():
                beat_times = np.load(str(p)).astype(np.float32).reshape(-1).tolist()
        except Exception:
            pass

    summary = render.get("summary", {})
    mf = render.get("model_facing", {})
    an = render.get("analytics", {})
    dist = render.get("distributions", {})
    raw_json = json.dumps(render, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rhythmic Extractor Debug</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, Arial, sans-serif; margin: 20px; background: #f4f4f4; color: #111; }}
    .container {{ background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ color: #0b3d91; }}
    .meta {{ background: #f7f7f7; padding: 12px; border-radius: 8px; margin: 12px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .card {{ border: 1px solid #eee; border-radius: 8px; padding: 10px; }}
    .k {{ color: #444; font-weight: 700; font-size: 0.95em; }}
    .v {{ font-variant-numeric: tabular-nums; font-weight: 800; font-size: 1.1em; }}
    canvas {{ width: 100%; max-width: 1100px; height: 240px; border: 1px solid #eee; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; }}
    th {{ background: #0b3d91; color: #fff; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Rhythmic Extractor Debug</h1>
    <div class="meta">
      <div><strong>Status:</strong> {summary.get('status','unknown')}</div>
      <div><strong>Schema:</strong> {summary.get('schema_version','unknown')} | <strong>Producer:</strong> {summary.get('producer_version','unknown')}</div>
      <div><strong>Backend:</strong> {summary.get('backend','unknown')} | <strong>Hop:</strong> {summary.get('hop_length','unknown')}</div>
      <div><strong>Sampling family used:</strong> {summary.get('sampling_family_used','unknown')}</div>
    </div>

    <h2>Model-facing (tabular)</h2>
    <div class="grid">
      <div class="card"><div class="k">tempo_bpm</div><div class="v">{mf.get('rhythm_tempo_bpm')}</div></div>
      <div class="card"><div class="k">beats_count</div><div class="v">{mf.get('rhythm_beats_count')}</div></div>
      <div class="card"><div class="k">beat_density</div><div class="v">{mf.get('rhythm_beat_density')}</div></div>
      <div class="card"><div class="k">regularity</div><div class="v">{mf.get('rhythm_regularity')}</div></div>
      <div class="card"><div class="k">tempo_variation</div><div class="v">{mf.get('rhythm_tempo_variation')}</div></div>
      <div class="card"><div class="k">beat_consistency</div><div class="v">{mf.get('rhythm_beat_consistency')}</div></div>
    </div>

    <h2>Beat timeline</h2>
    <canvas id="beatsCanvas" width="1100" height="240"></canvas>

    <h2>Intervals distribution</h2>
    <table>
      <thead><tr><th>stat</th><th>value</th></tr></thead>
      <tbody>
        <tr><td><strong>min</strong></td><td>{(dist.get('intervals',{}).get('min'))}</td></tr>
        <tr><td><strong>median</strong></td><td>{(dist.get('intervals',{}).get('median'))}</td></tr>
        <tr><td><strong>mean</strong></td><td>{(dist.get('intervals',{}).get('mean'))}</td></tr>
        <tr><td><strong>std</strong></td><td>{(dist.get('intervals',{}).get('std'))}</td></tr>
        <tr><td><strong>max</strong></td><td>{(dist.get('intervals',{}).get('max'))}</td></tr>
      </tbody>
    </table>

    <h2>Analytics (selected)</h2>
    <pre>{json.dumps(an, indent=2)}</pre>

    <h2>Raw render-context (JSON)</h2>
    <pre>{raw_json}</pre>
  </div>

  <script>
    const beatTimes = {json.dumps(beat_times)};

    function isFiniteNum(x) {{
      const v = Number(x);
      return Number.isFinite(v);
    }}

    function drawBeats(canvasId, xs) {{
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      const pad = 30;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0,0,w,h);
      ctx.strokeStyle = '#ddd';
      ctx.strokeRect(0.5,0.5,w-1,h-1);

      const pts = xs.filter(isFiniteNum).map(Number);
      if (pts.length === 0) {{
        ctx.fillStyle = '#666';
        ctx.font = '14px sans-serif';
        ctx.fillText('no beat_times_sec (enable beat_times or check *_npy path)', 12, 24);
        return;
      }}
      let xMin = pts[0], xMax = pts[0];
      for (const x of pts) {{ if (x < xMin) xMin = x; if (x > xMax) xMax = x; }}
      if (xMax <= xMin) xMax = xMin + 1;
      const xScale = (w - 2*pad) / (xMax - xMin);

      ctx.strokeStyle = '#bbb';
      ctx.beginPath();
      ctx.moveTo(pad, h-pad);
      ctx.lineTo(w-pad, h-pad);
      ctx.stroke();

      ctx.strokeStyle = '#eb4d4b';
      ctx.lineWidth = 1;
      for (const t of pts) {{
        const x = pad + (t - xMin) * xScale;
        ctx.beginPath();
        ctx.moveTo(x, pad);
        ctx.lineTo(x, h-pad);
        ctx.stroke();
      }}

      ctx.fillStyle = '#666';
      ctx.font = '12px sans-serif';
      ctx.fillText('time (sec)', w-pad-70, h-10);
      ctx.fillText(`beats: ${{pts.length}}`, 12, 18);
    }}

    drawBeats('beatsCanvas', beatTimes);
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, output_path)
    return output_path


__all__ = ["render_rhythmic_extractor", "render_rhythmic_extractor_html"]

