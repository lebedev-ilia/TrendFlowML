"""
Offline renderer for tempo_extractor (Audit v3).

- No external CDNs (offline-only).
- Reads NPZ v1 keys: segment_center_sec, bpm_by_segment (fallback: windowed_times_sec, windowed_bpm).
"""

import os
import json
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        xf = float(x)
        if np.isnan(xf):
            return int(default)
        return int(xf)
    except Exception:
        return int(default)


def render_tempo_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для tempo_extractor (Audit v3)."""
    render = {
        "component": "tempo_extractor",
        "meta": {
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "schema_version": meta.get("schema_version"),
        },
        "summary": {},
        "timeline": [],
        "distributions": {},
        "warnings": [],
        "axis": {},
        "per_segment": {},
    }

    # Extract scalar features
    feature_names = _to_list(npz_data.get("feature_names", []))
    feature_values = _to_list(npz_data.get("feature_values", []))
    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            features[str(name)] = feature_values[i]

    confidence = features.get("confidence") or features.get("tempo_confidence", 0.0)
    render["summary"] = {
        "tempo_bpm": features.get("tempo_bpm", 0.0),
        "tempo_bpm_mean": features.get("tempo_bpm_mean", 0.0),
        "tempo_bpm_median": features.get("tempo_bpm_median", 0.0),
        "tempo_bpm_std": features.get("tempo_bpm_std", 0.0),
        "tempo_confidence": float(confidence),
        "segments_count": _safe_int(features.get("segments_count", 0), default=0),
    }

    # Warnings
    warnings = npz_data.get("warnings")
    if warnings is not None:
        render["warnings"] = _to_list(warnings) if isinstance(warnings, (list, np.ndarray)) else []

    # Timeline: Audit v3 canonical (segment_center_sec, bpm_by_segment) or legacy (windowed_times_sec, windowed_bpm)
    seg_center = npz_data.get("segment_center_sec")
    bpm_by_seg = npz_data.get("bpm_by_segment")
    if seg_center is not None and bpm_by_seg is not None:
        times_sec = _to_list(seg_center)
        bpms = _to_list(bpm_by_seg)
        seg_mask = _to_list(npz_data.get("segment_mask"))
    else:
        times_sec = _to_list(npz_data.get("windowed_times_sec"))
        bpms = _to_list(npz_data.get("windowed_bpm"))
        seg_mask = [True] * max(len(times_sec), len(bpms))

    render["axis"] = {
        "segment_center_sec": times_sec,
        "segment_mask": seg_mask,
    }
    render["per_segment"] = {"bpm_by_segment": bpms}

    timeline = []
    for i in range(min(len(times_sec), len(bpms))):
        ok = seg_mask[i] if i < len(seg_mask) else True
        timeline.append({
            "time_sec": float(times_sec[i]),
            "bpm": float(bpms[i]),
            "window_index": i,
            "mask": bool(ok),
        })
    render["timeline"] = timeline

    # Distribution of BPM (masked segments only)
    valid_bpms = [t["bpm"] for t in timeline if t.get("mask", True) and np.isfinite(t["bpm"])]
    if valid_bpms:
        render["distributions"]["bpm"] = {
            "min": float(np.min(valid_bpms)),
            "max": float(np.max(valid_bpms)),
            "mean": float(np.mean(valid_bpms)),
            "std": float(np.std(valid_bpms)),
            "median": float(np.median(valid_bpms)),
        }

    # Distribution of tempo_estimates
    tempo_estimates = npz_data.get("tempo_estimates")
    if tempo_estimates is not None:
        te_list = _to_list(tempo_estimates)
        if te_list:
            render["distributions"]["tempo_estimates"] = {
                "min": float(np.min(te_list)),
                "max": float(np.max(te_list)),
                "mean": float(np.mean(te_list)),
                "std": float(np.std(te_list)),
                "median": float(np.median(te_list)),
            }

    return render


def render_tempo_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Offline HTML render (vanilla canvas, no CDN).
    """
    import sys
    from pathlib import Path
    ap_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(ap_root / "src") not in sys.path:
        sys.path.insert(0, str(ap_root / "src"))
    from ....core.renderer import load_npz, extract_meta

    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    ctx = render_tempo_extractor(npz_data, meta)

    timeline = ctx.get("timeline", [])
    summary = ctx.get("summary", {})
    distributions = ctx.get("distributions", {})
    warnings = ctx.get("warnings", [])
    bpm_dist = distributions.get("bpm", {})
    tempo_estimates_dist = distributions.get("tempo_estimates", {})

    # Points for canvas (masked only, finite BPM)
    pts = []
    for t in timeline:
        if not t.get("mask", True):
            continue
        bpm = t.get("bpm")
        if bpm is None or (isinstance(bpm, float) and not np.isfinite(bpm)):
            continue
        pts.append([float(t.get("time_sec", 0)), float(bpm)])

    warnings_html = ""
    if warnings:
        warnings_list = ", ".join(str(w) for w in warnings)
        warnings_html = f"""
        <div class="warnings">
            <h2>Warnings</h2>
            <div class="warning-badge">{warnings_list}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tempo Extractor Debug Render</title>
    <style>
        :root {{ --bg: #0b0f14; --panel: #101823; --text: #e6edf3; --muted: #8aa2b2; }}
        body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: var(--bg); color: var(--text); }}
        .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        .card {{ background: var(--panel); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px; margin-top: 16px; }}
        h1 {{ margin: 0 0 8px 0; font-size: 18px; }}
        .meta {{ color: var(--muted); font-size: 12px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin-top: 12px; }}
        .kpi {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; }}
        .kpi .k {{ color: var(--muted); font-size: 11px; }}
        .kpi .v {{ font-size: 14px; margin-top: 4px; }}
        canvas {{ width: 100%; height: 260px; background: rgba(0,0,0,0.22); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }}
        table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
        th, td {{ border: 1px solid rgba(255,255,255,0.12); padding: 8px; text-align: left; }}
        th {{ background: rgba(255,255,255,0.06); color: var(--muted); }}
        .warnings {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #ffc107; }}
        .warning-badge {{ background: #ffc107; color: #856404; padding: 8px 12px; border-radius: 4px; display: inline-block; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h1>Tempo Extractor Debug Render</h1>
            <div class="meta" id="metaLine"></div>
            <div class="grid" id="kpis"></div>
        </div>
        {warnings_html}
        <div class="card">
            <div class="meta">BPM over time (segment_center_sec vs bpm_by_segment)</div>
            <canvas id="plotBpm"></canvas>
        </div>
        {f'''
        <div class="card">
            <h2 style="font-size:16px; margin:0 0 8px 0;">BPM by segment distribution</h2>
            <table>
                <thead><tr><th>Statistic</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td><strong>Min</strong></td><td>{bpm_dist.get('min', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Max</strong></td><td>{bpm_dist.get('max', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Mean</strong></td><td>{bpm_dist.get('mean', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Std</strong></td><td>{bpm_dist.get('std', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Median</strong></td><td>{bpm_dist.get('median', 0.0):.2f} BPM</td></tr>
                </tbody>
            </table>
        </div>
        ''' if bpm_dist else ''}
        {f'''
        <div class="card">
            <h2 style="font-size:16px; margin:0 0 8px 0;">Tempo estimates distribution</h2>
            <table>
                <thead><tr><th>Statistic</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td><strong>Min</strong></td><td>{tempo_estimates_dist.get('min', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Max</strong></td><td>{tempo_estimates_dist.get('max', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Mean</strong></td><td>{tempo_estimates_dist.get('mean', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Std</strong></td><td>{tempo_estimates_dist.get('std', 0.0):.2f} BPM</td></tr>
                    <tr><td><strong>Median</strong></td><td>{tempo_estimates_dist.get('median', 0.0):.2f} BPM</td></tr>
                </tbody>
            </table>
        </div>
        ''' if tempo_estimates_dist else ''}
    </div>

    <script>
        const ctx = {json.dumps(ctx)};
        const meta = ctx.meta || {{}};
        document.getElementById('metaLine').textContent =
            `status=${{meta.status ?? 'unknown'}} schema=${{meta.schema_version ?? 'unknown'}} empty_reason=${{meta.empty_reason ?? '—'}} segments=${{ctx.summary?.segments_count ?? 0}}`;

        const kpiKeys = ['tempo_bpm', 'tempo_bpm_mean', 'tempo_bpm_median', 'tempo_bpm_std', 'tempo_confidence'];
        const kpisEl = document.getElementById('kpis');
        for (const k of kpiKeys) {{
            const v = ctx.summary?.[k];
            if (v === undefined) continue;
            const d = document.createElement('div');
            d.className = 'kpi';
            d.innerHTML = `<div class="k">${{k}}</div><div class="v">${{(v ?? '—')}}</div>`;
            kpisEl.appendChild(d);
        }}

        function plotBpm() {{
            const c = document.getElementById('plotBpm');
            const dpr = window.devicePixelRatio || 1;
            const w = Math.floor(c.clientWidth * dpr);
            const h = Math.floor(c.clientHeight * dpr);
            c.width = w; c.height = h;
            const g = c.getContext('2d');
            g.clearRect(0,0,w,h);

            const pts = {json.dumps(pts)};
            if (!pts.length) {{
                g.fillStyle = 'rgba(255,255,255,0.6)';
                g.font = `${{14*dpr}}px ui-sans-serif`;
                g.fillText('no valid segments to plot', 14*dpr, 24*dpr);
                return;
            }}

            let xmin = pts[0][0], xmax = pts[0][0], ymin = pts[0][1], ymax = pts[0][1];
            for (const [x,y] of pts) {{
                xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
                ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
            }}
            const pad = 18*dpr;
            const xspan = (xmax - xmin) || 1.0;
            const yspan = (ymax - ymin) || 1.0;
            ymin = Math.max(40, Math.min(ymin, 220) - 10);
            ymax = Math.min(220, Math.max(ymax, 40) + 10);

            g.strokeStyle = 'rgba(255,255,255,0.10)';
            g.lineWidth = 1;
            for (let i=0;i<=10;i++) {{
                const xx = pad + (i/10) * (w - pad*2);
                g.beginPath(); g.moveTo(xx, pad); g.lineTo(xx, h-pad); g.stroke();
            }}
            for (let i=0;i<=6;i++) {{
                const yy = pad + (i/6) * (h - pad*2);
                g.beginPath(); g.moveTo(pad, yy); g.lineTo(w-pad, yy); g.stroke();
            }}

            function X(x) {{ return pad + ((x - xmin) / xspan) * (w - pad*2); }}
            function Y(y) {{ return (h - pad) - ((y - ymin) / (ymax - ymin || 1)) * (h - pad*2); }}

            g.strokeStyle = '#ff6384';
            g.lineWidth = 2*dpr;
            g.beginPath();
            for (let i=0;i<pts.length;i++) {{
                const [x,y] = pts[i];
                const px = X(x), py = Y(y);
                if (i===0) g.moveTo(px, py); else g.lineTo(px, py);
            }}
            g.stroke();

            g.fillStyle = '#ff6384';
            for (const [x,y] of pts) {{
                const px = X(x), py = Y(y);
                g.beginPath(); g.arc(px, py, 3*dpr, 0, Math.PI*2); g.fill();
            }}
        }}

        plotBpm();
        window.addEventListener('resize', plotBpm);
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)

    logger.info("Saved Tempo HTML render to %s", output_path)
    return output_path


__all__ = ["render_tempo_extractor", "render_tempo_extractor_html"]
