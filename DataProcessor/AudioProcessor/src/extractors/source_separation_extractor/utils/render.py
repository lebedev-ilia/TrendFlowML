"""
Renderer for source_separation_extractor (Audit v3): render-context JSON + offline HTML (vanilla canvas, no CDN).
"""

import os
import json
import logging
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


def render_source_separation_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
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

    so = npz_data.get("source_order")
    if isinstance(so, np.ndarray) and so.dtype == object:
        so = so.tolist()
    source_order = so if isinstance(so, list) else ["vocals", "drums", "bass", "other"]

    render: Dict[str, Any] = {
        "component": "source_separation_extractor",
        "summary": {
            "status": meta.get("status") if isinstance(meta, dict) else None,
            "empty_reason": meta.get("empty_reason") if isinstance(meta, dict) else None,
            "schema_version": meta.get("schema_version") if isinstance(meta, dict) else None,
            "producer_version": meta.get("producer_version") if isinstance(meta, dict) else None,
            "model_name": meta.get("model_name") if isinstance(meta, dict) else None,
            "weights_digest": meta.get("weights_digest") if isinstance(meta, dict) else None,
            "source_order": source_order,
        },
        "model_facing": feats,
        "axis": {
            "segment_start_sec": _to_list(npz_data.get("segment_start_sec")),
            "segment_end_sec": _to_list(npz_data.get("segment_end_sec")),
            "segment_center_sec": _to_list(npz_data.get("segment_center_sec")),
            "segment_mask": _to_list(npz_data.get("segment_mask")),
        },
        "vectors": {
            "share_mean": _to_list(npz_data.get("share_mean")),
            "share_std": _to_list(npz_data.get("share_std")),
            "source_distribution_ratio": _to_list(npz_data.get("source_distribution_ratio")),
            "source_segments_count": _to_list(npz_data.get("source_segments_count")),
            "source_duration_sec": _to_list(npz_data.get("source_duration_sec")),
        },
        "sequences": {
            "share_sequence": _to_list(npz_data.get("share_sequence")),
            "energy_sequence": _to_list(npz_data.get("energy_sequence")),
        },
    }

    # Optional analytics scalars (if present)
    analytics = {}
    for k in [
        "source_entropy_mean",
        "source_entropy_std",
        "energy_balance_mean",
        "vocals_presence_ratio",
        "drums_flux",
        "bass_floor_p20",
    ]:
        if k in npz_data:
            v = npz_data.get(k)
            if isinstance(v, np.ndarray) and v.size == 1:
                analytics[k] = float(v.item())
            elif isinstance(v, (int, float, np.number)):
                analytics[k] = float(v)
    if analytics:
        render["analytics"] = analytics

    return render


def render_source_separation_extractor_html(npz_path: str, output_path: str) -> str:
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_source_separation_extractor(npz_data, meta)

    source_order = render.get("summary", {}).get("source_order") or ["vocals", "drums", "bass", "other"]
    share_mean = render.get("vectors", {}).get("share_mean") or []
    share_seq = render.get("sequences", {}).get("share_sequence") or []
    mask = render.get("axis", {}).get("segment_mask") or []

    raw_json = json.dumps(render, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Source Separation Debug</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, Arial, sans-serif; margin: 20px; background: #f4f4f4; color: #111; }}
    .container {{ background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ color: #2b2d42; }}
    .meta {{ background: #f7f7f7; padding: 12px; border-radius: 8px; margin: 12px 0; }}
    canvas {{ width: 100%; max-width: 1100px; height: 260px; border: 1px solid #eee; border-radius: 8px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Source Separation Extractor Debug</h1>
    <div class="meta">
      <div><strong>Status:</strong> {render.get('summary',{}).get('status','unknown')} {(' | ' + str(render.get('summary',{}).get('empty_reason'))) if render.get('summary',{}).get('empty_reason') else ''}</div>
      <div><strong>Schema:</strong> {render.get('summary',{}).get('schema_version','unknown')} | <strong>Producer:</strong> {render.get('summary',{}).get('producer_version','unknown')}</div>
      <div><strong>Model:</strong> {render.get('summary',{}).get('model_name','unknown')} | <strong>Weights:</strong> {render.get('summary',{}).get('weights_digest','')}</div>
    </div>

    <h2>Mean shares</h2>
    <canvas id="meanCanvas" width="1100" height="260"></canvas>

    <h2>Share sequence (masked segments skipped)</h2>
    <canvas id="seqCanvas" width="1100" height="260"></canvas>

    <h2>Raw render-context (JSON)</h2>
    <pre>{raw_json}</pre>
  </div>

  <script>
    const sourceOrder = {json.dumps(source_order)};
    const meanShares = {json.dumps(share_mean)};
    const shareSeq = {json.dumps(share_seq)};
    const mask = {json.dumps(mask)};

    function drawBar(canvasId, labels, values) {{
      const c = document.getElementById(canvasId);
      const ctx = c.getContext('2d');
      const w = c.width, h = c.height;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = '#fff'; ctx.fillRect(0,0,w,h);
      const pad = 40;
      const innerW = w - 2*pad, innerH = h - 2*pad;
      const n = Math.min(labels.length, values.length);
      if (n === 0) {{
        ctx.fillStyle = '#666'; ctx.font = '14px sans-serif';
        ctx.fillText('mean shares not available', 12, 24);
        return;
      }}
      const bw = innerW / n * 0.7;
      const gap = innerW / n * 0.3;
      const colors = ['#ef476f','#ffd166','#06d6a0','#118ab2'];
      for (let i=0;i<n;i++) {{
        const v = Number(values[i]);
        const x = pad + i*(bw+gap);
        const barH = Math.max(0, Math.min(innerH, innerH * (v/1.0)));
        const y = pad + (innerH - barH);
        ctx.fillStyle = colors[i % 4];
        ctx.fillRect(x, y, bw, barH);
        ctx.fillStyle = '#111';
        ctx.font = '12px sans-serif';
        ctx.fillText(labels[i], x, h - pad + 14);
        ctx.fillText((Number.isFinite(v) ? v.toFixed(3) : 'NaN'), x, y - 6);
      }}
    }}

    function drawLines(canvasId, seq, maskArr) {{
      const c = document.getElementById(canvasId);
      const ctx = c.getContext('2d');
      const w = c.width, h = c.height;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = '#fff'; ctx.fillRect(0,0,w,h);
      const pad = 40;
      const innerW = w - 2*pad, innerH = h - 2*pad;
      if (!Array.isArray(seq) || seq.length === 0) {{
        ctx.fillStyle = '#666'; ctx.font = '14px sans-serif';
        ctx.fillText('share_sequence not saved (enable flag)', 12, 24);
        return;
      }}
      const idx = [];
      for (let i=0;i<seq.length;i++) {{
        const ok = (Array.isArray(maskArr) && maskArr.length === seq.length) ? Boolean(maskArr[i]) : true;
        if (ok) idx.push(i);
      }}
      if (idx.length === 0) {{
        ctx.fillStyle = '#666'; ctx.font = '14px sans-serif';
        ctx.fillText('all segments masked', 12, 24);
        return;
      }}
      const colors = ['#ef476f','#ffd166','#06d6a0','#118ab2'];
      for (let s=0;s<4;s++) {{
        ctx.beginPath();
        for (let j=0;j<idx.length;j++) {{
          const i = idx[j];
          const row = seq[i];
          const v = (Array.isArray(row) && row.length>=4) ? Number(row[s]) : 0;
          const x = pad + (innerW * j / (idx.length - 1 || 1));
          const y = pad + (innerH * (1 - v));
          if (j === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }}
        ctx.strokeStyle = colors[s]; ctx.lineWidth = 2; ctx.stroke();
      }}
      ctx.fillStyle = '#666'; ctx.font = '12px sans-serif';
      ctx.fillText('y: share (0..1), x: segment index (masked removed)', pad, pad-10);
    }}

    drawBar('meanCanvas', sourceOrder, meanShares);
    drawLines('seqCanvas', shareSeq, mask);
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, output_path)
    return output_path


__all__ = ["render_source_separation_extractor", "render_source_separation_extractor_html"]

