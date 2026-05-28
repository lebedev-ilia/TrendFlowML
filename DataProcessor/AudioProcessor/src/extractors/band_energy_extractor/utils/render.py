"""
Renderer для band_energy_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def render_band_energy_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для band_energy_extractor."""
    render = {
        "component": "band_energy_extractor",
        "summary": {},
        "band_info": {},
        "balance_metrics": {},
        "time_series": {},
    }

    features_enabled = meta.get("features_enabled", [])

    # Canonical arrays (Audit v3): no payload dict in NPZ.
    band_edges_hz = npz_data.get("band_edges_hz")
    band_energy_shares = npz_data.get("band_energy_shares")

    if isinstance(band_edges_hz, np.ndarray):
        band_edges_hz_list = band_edges_hz.astype(float).tolist()
    else:
        band_edges_hz_list = []

    if isinstance(band_energy_shares, np.ndarray):
        band_shares_list = band_energy_shares.astype(float).tolist()
    else:
        band_shares_list = []
    
    # Summary
    render["summary"] = {
        "band_edges_hz": band_edges_hz_list,
        "num_bands": len(band_edges_hz_list) if band_edges_hz_list else 0,
        "method": meta.get("method", "unknown"),
        "sample_rate": meta.get("sample_rate", 0),
        "n_fft": meta.get("n_fft", 0),
        "hop_length": meta.get("hop_length", 0),
        "duration": meta.get("duration", 0.0),
    }

    band_edges = band_edges_hz_list
    render["band_info"] = {
        "band_energy_shares": band_shares_list,
        "dominant_band": int(np.argmax(band_shares_list)) if band_shares_list else 0,
    }

    # Balance metrics if enabled
    if "balance_metrics" in features_enabled:
        band_balance_score = meta.get("band_balance_score", 0.0)
        band_dominance = meta.get("band_dominance", 0)
        band_contrast = meta.get("band_contrast", 0.0)
        
        render["balance_metrics"] = {
            "band_balance_score": float(band_balance_score) if band_balance_score is not None else 0.0,
            "band_dominance": int(band_dominance) if band_dominance is not None else 0,
            "band_contrast": float(band_contrast) if band_contrast is not None else 0.0,
        }

    # Time series if enabled
    if "time_series" in features_enabled:
        segment_centers = npz_data.get("segment_centers_sec")
        segment_mask = npz_data.get("segment_mask")
        band_shares_by_segment = npz_data.get("band_shares_by_segment")

        if isinstance(segment_centers, np.ndarray):
            segment_centers = segment_centers.astype(float).tolist()
        else:
            segment_centers = []

        if isinstance(segment_mask, np.ndarray):
            segment_mask = segment_mask.astype(bool).tolist()
        else:
            segment_mask = []

        if isinstance(band_shares_by_segment, np.ndarray):
            band_shares_by_segment = band_shares_by_segment.astype(float).tolist()
        else:
            band_shares_by_segment = []

        render["time_series"] = {
            "segment_centers_sec": segment_centers,
            "segment_mask": segment_mask,
            "band_shares_by_segment": band_shares_by_segment,
        }

    return render


def render_band_energy_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага band_energy_extractor результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла

    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_band_energy_extractor(npz_data, meta)

    summary = render.get("summary", {})
    band_info = render.get("band_info", {})
    balance_metrics = render.get("balance_metrics", {})
    time_series = render.get("time_series", {})

    # Prepare data for visualization
    band_edges = summary.get("band_edges_hz", [])
    band_shares = band_info.get("band_energy_shares", [])
    segment_centers = time_series.get("segment_centers_sec", [])
    segment_mask = time_series.get("segment_mask", [])
    band_shares_by_segment = time_series.get("band_shares_by_segment", [])

    # Band names
    band_names = [f"Band {i+1}" for i in range(len(band_edges))]
    if len(band_edges) == 3:
        band_names = ["Low", "Mid", "High"]

    # Simple offline HTML: no external libs.
    def _bar_row(label: str, value: float) -> str:
        v = float(value) if value is not None else 0.0
        v = max(0.0, min(1.0, v))
        pct = int(round(v * 100.0))
        return f"""
        <div class="row">
          <div class="lbl">{label}</div>
          <div class="bar"><div class="fill" style="width:{pct}%"></div></div>
          <div class="val">{v:.3f}</div>
        </div>
        """

    shares_rows = ""
    for i, name in enumerate(band_names):
        v = band_shares[i] if i < len(band_shares) else 0.0
        shares_rows += _bar_row(name, v)

    seq_info = ""
    if segment_centers and band_shares_by_segment:
        valid = sum(1 for x in segment_mask if x) if isinstance(segment_mask, list) else 0
        seq_info = f"<div class='metric'><b>Segments:</b> N={len(segment_centers)}, valid={valid}</div>"

    def _fnum(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Band Energy Extractor Debug</title>
    <style>
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ color: #333; }}
        .row {{ display: grid; grid-template-columns: 90px 1fr 60px; gap: 10px; align-items: center; margin: 8px 0; }}
        .lbl {{ font-weight: 600; }}
        .bar {{ height: 14px; background: #eee; border-radius: 999px; overflow: hidden; }}
        .fill {{ height: 100%; background: linear-gradient(90deg, #2563eb, #0ea5e9); }}
        .val {{ text-align: right; font-variant-numeric: tabular-nums; }}
    </style>
</head>
<body>
    <h1>Band Energy Extractor Debug</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Number of Bands:</span>
            <span class="metric-value">{summary.get("num_bands", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Method:</span>
            <span class="metric-value">{summary.get("method", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">{_fnum(summary.get("duration"), 0.0):.2f}s</span>
        </div>
        {seq_info}
    </div>

    <div class="section">
        <h2>Band Energy Shares</h2>
        {shares_rows}
    </div>

    {f'''
    <div class="section">
        <h2>Balance Metrics</h2>
        <div class="metric">
            <span class="metric-label">Balance Score:</span>
            <span class="metric-value">{balance_metrics.get("band_balance_score", 0.0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Dominant Band:</span>
            <span class="metric-value">{band_names[balance_metrics.get("band_dominance", 0)] if balance_metrics.get("band_dominance") is not None else "N/A"}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Contrast:</span>
            <span class="metric-value">{_fnum(balance_metrics.get("band_contrast"), 0.0):.3f}</span>
        </div>
    </div>
    ''' if balance_metrics else ''}
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


__all__ = ["render_band_energy_extractor", "render_band_energy_extractor_html"]

