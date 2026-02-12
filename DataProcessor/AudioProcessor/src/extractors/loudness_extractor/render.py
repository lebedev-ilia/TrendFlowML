"""
Renderer для loudness_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_loudness_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для loudness_extractor."""
    render = {
        "component": "loudness_extractor",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract scalar features
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    
    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            features[name] = feature_values[i]
    
    # Summary - извлекаем из features
    lufs_present_flag = bool(features.get("lufs_present", 0.0) > 0.5) if isinstance(features.get("lufs_present"), (int, float)) else False
    lufs_value = features.get("loudness_lufs") or features.get("lufs")
    # Проверяем, что lufs не NaN
    if lufs_value is not None:
        try:
            lufs_float = float(lufs_value)
            if np.isnan(lufs_float):
                lufs_value = None
                lufs_present_flag = False
        except (ValueError, TypeError):
            lufs_value = None
            lufs_present_flag = False
    
    render["summary"] = {
        "rms": features.get("loudness_rms") or features.get("rms", 0.0),
        "peak": features.get("loudness_peak") or features.get("peak", 0.0),
        "dbfs": features.get("loudness_dbfs") or features.get("dbfs", 0.0),
        "rms_mean": features.get("segment_rms_mean", 0.0),
        "rms_std": features.get("segment_rms_std", 0.0),
        "peak_mean": features.get("peak_mean", 0.0),
        "dbfs_mean": features.get("dbfs_mean", 0.0),
        "lufs_present": lufs_present_flag,
        "lufs": float(lufs_value) if lufs_value is not None and not np.isnan(float(lufs_value)) else None,
        "segments_count": int(features.get("segments_count", 0)),
    }
    
    # Timeline data
    segment_centers_sec = npz_data.get("segment_centers_sec")
    segment_rms = npz_data.get("segment_rms")
    segment_dbfs = npz_data.get("segment_dbfs")
    segment_lufs = npz_data.get("segment_lufs")
    
    if segment_centers_sec is not None:
        if isinstance(segment_centers_sec, np.ndarray):
            segment_centers_sec = segment_centers_sec.tolist()
        
        timeline = []
        for i, center_sec in enumerate(segment_centers_sec):
            entry = {
                "center_sec": float(center_sec),
                "segment_index": i,
            }
            
            if segment_rms is not None:
                if isinstance(segment_rms, np.ndarray):
                    segment_rms_list = segment_rms.tolist()
                else:
                    segment_rms_list = segment_rms
                if i < len(segment_rms_list):
                    entry["rms"] = float(segment_rms_list[i])
            
            if segment_dbfs is not None:
                if isinstance(segment_dbfs, np.ndarray):
                    segment_dbfs_list = segment_dbfs.tolist()
                else:
                    segment_dbfs_list = segment_dbfs
                if i < len(segment_dbfs_list):
                    entry["dbfs"] = float(segment_dbfs_list[i])
            
            if segment_lufs is not None and render["summary"]["lufs_present"]:
                if isinstance(segment_lufs, np.ndarray):
                    segment_lufs_list = segment_lufs.tolist()
                else:
                    segment_lufs_list = segment_lufs
                if i < len(segment_lufs_list):
                    entry["lufs"] = float(segment_lufs_list[i])
            
            timeline.append(entry)
        render["timeline"] = timeline
        
        # Distributions
        if timeline:
            rms_values = [t.get("rms") for t in timeline if "rms" in t]
            dbfs_values = [t.get("dbfs") for t in timeline if "dbfs" in t]
            lufs_values = [t.get("lufs") for t in timeline if "lufs" in t]
            
            if rms_values:
                render["distributions"]["rms"] = {
                    "min": float(np.min(rms_values)),
                    "max": float(np.max(rms_values)),
                    "mean": float(np.mean(rms_values)),
                    "std": float(np.std(rms_values)),
                }
            
            if dbfs_values:
                render["distributions"]["dbfs"] = {
                    "min": float(np.min(dbfs_values)),
                    "max": float(np.max(dbfs_values)),
                    "mean": float(np.mean(dbfs_values)),
                    "std": float(np.std(dbfs_values)),
                }
            
            if lufs_values:
                render["distributions"]["lufs"] = {
                    "min": float(np.min(lufs_values)),
                    "max": float(np.max(lufs_values)),
                    "mean": float(np.mean(lufs_values)),
                    "std": float(np.std(lufs_values)),
                }
    
    return render


def render_loudness_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага loudness_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    ap_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(ap_root / "src") not in sys.path:
        sys.path.insert(0, str(ap_root / "src"))
    from core.renderer import load_npz, extract_meta  # type: ignore
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_loudness_extractor(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Prepare timeline data for charts
    timeline_js_rms = ""
    timeline_js_dbfs = ""
    timeline_js_lufs = ""
    
    if timeline:
        times = [t.get("center_sec", 0.0) for t in timeline]
        rms_values = [t.get("rms", 0.0) for t in timeline if "rms" in t]
        dbfs_values = [t.get("dbfs", 0.0) for t in timeline if "dbfs" in t]
        lufs_values = [t.get("lufs", 0.0) for t in timeline if "lufs" in t and not np.isnan(t.get("lufs", float("nan")))]
        
        if rms_values:
            timeline_js_rms = f"""
        const rmsTimelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times[:len(rms_values)]])},
            datasets: [{{
                label: 'RMS',
                data: {json.dumps(rms_values)},
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                pointRadius: 2
            }}]
        }};
        """
        
        if dbfs_values:
            timeline_js_dbfs = f"""
        const dbfsTimelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times[:len(dbfs_values)]])},
            datasets: [{{
                label: 'dBFS',
                data: {json.dumps(dbfs_values)},
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1,
                pointRadius: 2
            }}]
        }};
        """
        
        if lufs_values:
            timeline_js_lufs = f"""
        const lufsTimelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times[:len(lufs_values)]])},
            datasets: [{{
                label: 'LUFS',
                data: {json.dumps(lufs_values)},
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                tension: 0.1,
                pointRadius: 2
            }}]
        }};
        """
    
    # Формируем JavaScript код для графиков
    chart_js_code = ""
    if timeline_js_rms:
        chart_js_code += """
        const rmsCtx = document.getElementById('rmsChart').getContext('2d');
        new Chart(rmsCtx, {
            type: 'line',
            data: rmsTimelineData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'RMS'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });
        """
    if timeline_js_dbfs:
        chart_js_code += """
        const dbfsCtx = document.getElementById('dbfsChart').getContext('2d');
        new Chart(dbfsCtx, {
            type: 'line',
            data: dbfsTimelineData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'dBFS (dB)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });
        """
    if timeline_js_lufs:
        chart_js_code += """
        const lufsCtx = document.getElementById('lufsChart').getContext('2d');
        new Chart(lufsCtx, {
            type: 'line',
            data: lufsTimelineData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'LUFS'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });
        """
    
    # Формируем полный JavaScript блок
    script_content = ""
    if timeline_js_rms or timeline_js_dbfs or timeline_js_lufs:
        script_content = f"""
    <script>
        {timeline_js_rms or ''}
        {timeline_js_dbfs or ''}
        {timeline_js_lufs or ''}
        {chart_js_code}
    </script>
    """
    
    lufs_present = summary.get("lufs_present", False)
    lufs_value = summary.get("lufs")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loudness Extractor Debug Render</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; font-weight: bold; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin: 20px 0; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .meta-info {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 20px 0; font-size: 0.9em; color: #666; }}
        .lufs-info {{ background-color: {'#d4edda' if lufs_present else '#fff3cd'}; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid {'#c3e6cb' if lufs_present else '#ffc107'}; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Loudness Extractor Debug Render</h1>
        
        <div class="meta-info">
            <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
            <p><strong>Producer:</strong> {meta.get('producer_version', 'unknown')}</p>
            <p><strong>Schema Version:</strong> {meta.get('schema_version', 'unknown')}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>RMS Mean</strong>
                    <span class="metric-value">{summary.get('rms_mean', 0.0):.6f}</span>
                </div>
                <div class="metric-card">
                    <strong>RMS Std</strong>
                    <span class="metric-value">{summary.get('rms_std', 0.0):.6f}</span>
                </div>
                <div class="metric-card">
                    <strong>Peak Mean</strong>
                    <span class="metric-value">{summary.get('peak_mean', 0.0):.6f}</span>
                </div>
                <div class="metric-card">
                    <strong>dBFS Mean</strong>
                    <span class="metric-value">{summary.get('dbfs_mean', 0.0):.2f} dB</span>
                </div>
                <div class="metric-card">
                    <strong>LUFS Present</strong>
                    <span class="metric-value">{'Yes' if lufs_present else 'No'}</span>
                </div>
                {f'''
                <div class="metric-card">
                    <strong>LUFS</strong>
                    <span class="metric-value">{lufs_value:.2f} LUFS</span>
                </div>
                ''' if lufs_present and lufs_value is not None else ''}
            </div>
        </div>
        
        <div class="lufs-info">
            <p><strong>LUFS Status:</strong> {'LUFS computation available (pyloudnorm installed)' if lufs_present else 'LUFS computation not available (pyloudnorm not installed or failed)'}</p>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: RMS Over Time</h2>
            <canvas id="rmsChart"></canvas>
        </div>
        ''' if timeline_js_rms else '<p>No RMS timeline data available</p>'}
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: dBFS Over Time</h2>
            <canvas id="dbfsChart"></canvas>
        </div>
        ''' if timeline_js_dbfs else '<p>No dBFS timeline data available</p>'}
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: LUFS Over Time</h2>
            <canvas id="lufsChart"></canvas>
        </div>
        ''' if timeline_js_lufs else ''}
        
        {f'''
        <div class="distributions">
            <h2>RMS Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('rms', {}).get('min', 0.0):.6f}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('rms', {}).get('max', 0.0):.6f}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('rms', {}).get('mean', 0.0):.6f}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('rms', {}).get('std', 0.0):.6f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions.get('rms') else ''}
        
        {f'''
        <div class="distributions">
            <h2>dBFS Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('dbfs', {}).get('min', 0.0):.2f} dB</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('dbfs', {}).get('max', 0.0):.2f} dB</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('dbfs', {}).get('mean', 0.0):.2f} dB</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('dbfs', {}).get('std', 0.0):.2f} dB</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions.get('dbfs') else ''}
        
        {f'''
        <div class="distributions">
            <h2>LUFS Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('lufs', {}).get('min', 0.0):.2f} LUFS</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('lufs', {}).get('max', 0.0):.2f} LUFS</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('lufs', {}).get('mean', 0.0):.2f} LUFS</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('lufs', {}).get('std', 0.0):.2f} LUFS</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions.get('lufs') else ''}
    </div>
    
    {script_content}
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Saved Loudness HTML render to {output_path}")
    return output_path


__all__ = ["render_loudness_extractor", "render_loudness_extractor_html"]

