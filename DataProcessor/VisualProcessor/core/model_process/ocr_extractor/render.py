"""
Renderer для ocr_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


def render_ocr_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для ocr_extractor."""
    render = {
        "component": "ocr_extractor",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract OCR data
    ocr_raw = npz_data.get("ocr_raw")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to list if needed
    if ocr_raw is not None:
        if isinstance(ocr_raw, np.ndarray):
            if ocr_raw.dtype == object:
                ocr_raw = ocr_raw.tolist()
            else:
                ocr_raw = ocr_raw.tolist()
        elif not isinstance(ocr_raw, list):
            ocr_raw = []
    else:
        ocr_raw = []
    
    if times_s is not None:
        if isinstance(times_s, list):
            times_s = np.array(times_s, dtype=np.float32)
        elif isinstance(times_s, np.ndarray):
            times_s = np.asarray(times_s, dtype=np.float32)
        else:
            times_s = None
    
    if frame_indices is not None:
        if isinstance(frame_indices, list):
            frame_indices = np.array(frame_indices, dtype=np.int32)
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = np.asarray(frame_indices, dtype=np.int32)
        else:
            frame_indices = None
    
    # Summary statistics
    if ocr_raw and len(ocr_raw) > 0:
        # Extract statistics from OCR results
        text_lengths = []
        det_confidences = []
        frames_with_text = set()
        
        for ocr_item in ocr_raw:
            if isinstance(ocr_item, dict):
                text_raw = ocr_item.get("text_raw", "")
                text_norm = ocr_item.get("text_norm", "")
                det_conf = ocr_item.get("det_confidence", 0.0)
                frame_idx = ocr_item.get("frame", -1)
                
                if text_norm:
                    text_lengths.append(len(text_norm))
                    frames_with_text.add(frame_idx)
                if det_conf > 0:
                    det_confidences.append(float(det_conf))
        
        text_lengths_arr = np.array(text_lengths) if text_lengths else np.array([])
        det_confidences_arr = np.array(det_confidences) if det_confidences else np.array([])
        
        render["summary"] = {
            "frames_count": int(len(frame_indices)) if frame_indices is not None else 0,
            "total_ocr_results": len(ocr_raw),
            "frames_with_text": len(frames_with_text),
            "frames_with_text_ratio": float(len(frames_with_text) / len(frame_indices)) if frame_indices is not None and len(frame_indices) > 0 else 0.0,
            "text_length_mean": float(np.mean(text_lengths_arr)) if text_lengths_arr.size > 0 else 0.0,
            "text_length_std": float(np.std(text_lengths_arr)) if text_lengths_arr.size > 0 else 0.0,
            "text_length_min": int(np.min(text_lengths_arr)) if text_lengths_arr.size > 0 else 0,
            "text_length_max": int(np.max(text_lengths_arr)) if text_lengths_arr.size > 0 else 0,
            "text_length_median": float(np.median(text_lengths_arr)) if text_lengths_arr.size > 0 else 0.0,
            "det_confidence_mean": float(np.mean(det_confidences_arr)) if det_confidences_arr.size > 0 else 0.0,
            "det_confidence_std": float(np.std(det_confidences_arr)) if det_confidences_arr.size > 0 else 0.0,
            "det_confidence_min": float(np.min(det_confidences_arr)) if det_confidences_arr.size > 0 else 0.0,
            "det_confidence_max": float(np.max(det_confidences_arr)) if det_confidences_arr.size > 0 else 0.0,
            "det_confidence_median": float(np.median(det_confidences_arr)) if det_confidences_arr.size > 0 else 0.0,
        }
        
        # Timeline data (per-frame OCR statistics)
        if times_s is not None and frame_indices is not None:
            timeline = []
            frame_to_ocr = {}
            
            # Group OCR results by frame
            for ocr_item in ocr_raw:
                if isinstance(ocr_item, dict):
                    frame_idx = ocr_item.get("frame", -1)
                    if frame_idx not in frame_to_ocr:
                        frame_to_ocr[frame_idx] = []
                    frame_to_ocr[frame_idx].append(ocr_item)
            
            # Build timeline
            for i, frame_idx in enumerate(frame_indices):
                frame_idx_int = int(frame_idx)
                time_sec = float(times_s[i]) if i < len(times_s) else 0.0
                ocr_items = frame_to_ocr.get(frame_idx_int, [])
                
                ocr_count = len(ocr_items)
                avg_confidence = 0.0
                if ocr_items:
                    confidences = [item.get("det_confidence", 0.0) for item in ocr_items if isinstance(item, dict)]
                    if confidences:
                        avg_confidence = float(np.mean(confidences))
                
                timeline.append({
                    "frame_index": frame_idx_int,
                    "time_sec": time_sec,
                    "ocr_count": ocr_count,
                    "average_confidence": avg_confidence if ocr_count > 0 else None,
                })
            
            render["timeline"] = timeline
        
        # Distribution statistics
        distributions = {}
        
        if text_lengths_arr.size > 0:
            distributions["text_length"] = {
                "min": int(np.min(text_lengths_arr)),
                "max": int(np.max(text_lengths_arr)),
                "mean": float(np.mean(text_lengths_arr)),
                "std": float(np.std(text_lengths_arr)),
                "median": float(np.median(text_lengths_arr)),
                "p25": float(np.percentile(text_lengths_arr, 25)),
                "p75": float(np.percentile(text_lengths_arr, 75)),
                "p05": float(np.percentile(text_lengths_arr, 5)),
                "p95": float(np.percentile(text_lengths_arr, 95)),
            }
        
        if det_confidences_arr.size > 0:
            distributions["det_confidence"] = {
                "min": float(np.min(det_confidences_arr)),
                "max": float(np.max(det_confidences_arr)),
                "mean": float(np.mean(det_confidences_arr)),
                "std": float(np.std(det_confidences_arr)),
                "median": float(np.median(det_confidences_arr)),
                "p25": float(np.percentile(det_confidences_arr, 25)),
                "p75": float(np.percentile(det_confidences_arr, 75)),
                "p05": float(np.percentile(det_confidences_arr, 5)),
                "p95": float(np.percentile(det_confidences_arr, 95)),
            }
        
        render["distributions"] = distributions
        
        # Top text samples (for debugging)
        text_samples = []
        for ocr_item in ocr_raw[:10]:  # Top 10 samples
            if isinstance(ocr_item, dict):
                text_samples.append({
                    "text_norm": ocr_item.get("text_norm", ""),
                    "text_raw": ocr_item.get("text_raw", ""),
                    "det_confidence": ocr_item.get("det_confidence", 0.0),
                    "frame": ocr_item.get("frame", -1),
                })
        render["summary"]["text_samples"] = text_samples
    else:
        render["summary"] = {
            "frames_count": int(len(frame_indices)) if frame_indices is not None else 0,
            "total_ocr_results": 0,
            "frames_with_text": 0,
            "frames_with_text_ratio": 0.0,
        }
    
    return render


def render_ocr_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага ocr_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(vp_root / "core" / "model_process") not in sys.path:
        sys.path.insert(0, str(vp_root / "core" / "model_process"))
    
    # Try to import from utils if renderer exists
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except ImportError:
        # Fallback: direct load
        def load_npz(path: str):
            data = np.load(path, allow_pickle=True)
            result = {}
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.dtype == object:
                        result[key] = arr.item() if arr.size == 1 else arr.tolist()
                    else:
                        result[key] = arr.tolist() if arr.size > 0 else []
                else:
                    result[key] = arr
            return result
        
        def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
            meta = npz_data.get("meta")
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                return meta.item() if meta.size == 1 else meta.tolist()
            return meta if isinstance(meta, dict) else {}
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_ocr_extractor(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    text_samples = summary.get("text_samples", [])
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Generate HTML for text samples separately to avoid nested f-string issues
    text_samples_html = ""
    if text_samples:
        text_samples_rows = []
        for sample in text_samples:
            frame = sample.get('frame', -1)
            text_norm = sample.get('text_norm', '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text_raw = sample.get('text_raw', '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            confidence = sample.get('det_confidence', 0.0)
            text_samples_rows.append(f'''
                    <tr>
                        <td>{frame}</td>
                        <td class="text-norm">{text_norm}</td>
                        <td>{text_raw}</td>
                        <td>{confidence:.3f}</td>
                    </tr>''')
        text_samples_html = f'''
        <div class="text-samples">
            <h2>Text Samples (Top 10)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Frame</th>
                        <th>Text (Normalized)</th>
                        <th>Text (Raw)</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(text_samples_rows)}
                </tbody>
            </table>
        </div>
        '''
    
    # Prepare timeline data for chart
    timeline_js = ""
    y1_scale_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        ocr_counts = [t.get("ocr_count", 0) for t in timeline]
        avg_confidences = [t.get("average_confidence") for t in timeline if t.get("average_confidence") is not None]
        
        # Build datasets array
        datasets = [{
            "label": "OCR Count",
            "data": ocr_counts,
            "borderColor": "rgb(75, 192, 192)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "tension": 0.1,
            "yAxisID": "y"
        }]
        
        if avg_confidences:
            datasets.append({
                "label": "Average Confidence",
                "data": avg_confidences,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
            y1_scale_js = """,
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Average Confidence'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }"""
        
        # Format time labels
        time_labels = [f"{t:.2f}s" for t in times]
        timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps(time_labels)},
            datasets: {json.dumps(datasets)}
        }};
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Extractor Debug Render</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin: 20px 0; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .text-samples {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .text-samples table {{ width: 100%; border-collapse: collapse; }}
        .text-samples th, .text-samples td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .text-samples th {{ background-color: #0056b3; color: white; }}
        .text-samples .text-norm {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OCR Extractor Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Total OCR Results</strong>
                    <span class="metric-value">{summary.get('total_ocr_results', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Frames with Text</strong>
                    <span class="metric-value">{summary.get('frames_with_text', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Frames with Text Ratio</strong>
                    <span class="metric-value">{summary.get('frames_with_text_ratio', 0.0):.2%}</span>
                </div>
                <div class="metric-card">
                    <strong>Text Length Mean</strong>
                    <span class="metric-value">{summary.get('text_length_mean', 0.0):.1f}</span>
                </div>
                <div class="metric-card">
                    <strong>Det Confidence Mean</strong>
                    <span class="metric-value">{summary.get('det_confidence_mean', 0.0):.3f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: OCR Count and Confidence Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Text Length</th>
                        <th>Det Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('text_length', 'min')}</td>
                        <td>{format_dist_value('det_confidence', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('text_length', 'max')}</td>
                        <td>{format_dist_value('det_confidence', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('text_length', 'mean')}</td>
                        <td>{format_dist_value('det_confidence', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('text_length', 'std')}</td>
                        <td>{format_dist_value('det_confidence', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('text_length', 'median')}</td>
                        <td>{format_dist_value('det_confidence', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
        
        {text_samples_html if text_samples else ''}
    </div>
    
    {f'''
    <script>
        {timeline_js}
        const ctx = document.getElementById('timelineChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: timelineData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'OCR Count'
                        }}
                    }}{y1_scale_js}
                }}
            }}
        }});
    </script>
    ''' if timeline else ''}
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved OCR Extractor HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_ocr_extractor", "render_ocr_extractor_html"]

