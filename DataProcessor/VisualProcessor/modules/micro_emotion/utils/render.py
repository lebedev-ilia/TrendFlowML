"""
Renderer для micro_emotion: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _to_1d_float(arr: Any) -> np.ndarray:
    try:
        if arr is None:
            return np.asarray([], dtype=np.float32)
        if isinstance(arr, np.ndarray):
            return np.asarray(arr, dtype=np.float32).reshape(-1)
        if isinstance(arr, list):
            out = []
            for v in arr:
                if v is None:
                    out.append(np.nan)
                else:
                    out.append(float(v))
            return np.asarray(out, dtype=np.float32).reshape(-1)
    except Exception:
        pass
    return np.asarray([], dtype=np.float32)


def _svg_line_chart(
    *,
    times_s: Any,
    values: Any,
    width: int = 1000,
    height: int = 240,
    stroke: str = "#2563eb",
    title: str = "",
) -> str:
    t = _to_1d_float(times_s)
    v = _to_1d_float(values)
    n = int(min(t.size, v.size))
    if n <= 1:
        return "<div class='muted'>No data</div>"
    t = t[:n]
    v = v[:n]
    max_points = 600
    step = max(1, int(np.ceil(float(n) / float(max_points))))
    t = t[::step]
    v = v[::step]
    finite = np.isfinite(t) & np.isfinite(v)
    if not np.any(finite):
        return "<div class='muted'>No finite data</div>"
    t_f = t[finite]
    v_f = v[finite]
    t_min = float(np.min(t_f))
    t_max = float(np.max(t_f))
    v_min = float(np.min(v_f))
    v_max = float(np.max(v_f))
    if abs(t_max - t_min) < 1e-9:
        t_max = t_min + 1e-6
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1e-6
    pad_x = 40
    pad_y = 24
    w = max(240, int(width))
    h = max(120, int(height))
    x0, y0 = pad_x, pad_y
    x1, y1 = w - pad_x, h - pad_y

    def _sx(tt: float) -> float:
        return x0 + (tt - t_min) / (t_max - t_min) * (x1 - x0)

    def _sy(vv: float) -> float:
        return y1 - (vv - v_min) / (v_max - v_min) * (y1 - y0)

    path_parts: List[str] = []
    pen_down = False
    for tt, vv, ok in zip(t.tolist(), v.tolist(), finite.tolist()):
        if not ok:
            pen_down = False
            continue
        x = _sx(float(tt))
        y = _sy(float(vv))
        if not pen_down:
            path_parts.append(f"M {x:.2f} {y:.2f}")
            pen_down = True
        else:
            path_parts.append(f"L {x:.2f} {y:.2f}")

    path_d = " ".join(path_parts)
    esc_title = (title or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div class="chart">
      <div class="chart-title">{esc_title}</div>
      <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc_title}">
        <rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff" stroke="#e5e7eb"/>
        <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#e5e7eb"/>
        <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#e5e7eb"/>
        <path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="2"/>
        <text x="{x0}" y="{y0 - 8}" font-size="12" fill="#6b7280">{v_max:.3f}</text>
        <text x="{x0}" y="{y1 + 16}" font-size="12" fill="#6b7280">{v_min:.3f}</text>
        <text x="{x0}" y="{h - 4}" font-size="12" fill="#6b7280">{t_min:.2f}s</text>
        <text x="{x1 - 40}" y="{h - 4}" font-size="12" fill="#6b7280">{t_max:.2f}s</text>
      </svg>
    </div>
    """.strip()


def _as_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32)
    return None


def _as_int_array(x, dtype=np.int32) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)
    return None


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def render_micro_emotion(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Генерировать render-context для micro_emotion.

    Структура:
    - summary: видео-уровневые агрегаты (microexpr, улыбка, зрительный контакт, надёжность и т.п.)
    - timeline: per-frame таймлайн (взгляд, присутствие лица, proxy-интенсивности)
    - distributions: распределения по ключевым метрикам
    - events: упрощённый список micro-expressions
    """
    render: Dict[str, Any] = {
        "component": "micro_emotion",
        "key_facts": {},
        "config_highlights": {},
        "summary": {},
        "timeline": [],
        "distributions": {},
        "events": [],
    }

    if isinstance(meta, dict):
        render["key_facts"] = {
            "schema_version": meta.get("schema_version"),
            "producer_version": meta.get("producer_version"),
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        }
        render["config_highlights"] = {
            "docker_image": meta.get("docker_image"),
            "openface_batch_size": meta.get("openface_batch_size"),
            "feature_groups": meta.get("feature_groups"),
            "fps": meta.get("fps"),
            "microexpr_smoothing_sigma": meta.get("microexpr_smoothing_sigma"),
            "microexpr_delta_threshold": meta.get("microexpr_delta_threshold"),
            "microexpr_max_duration_frames": meta.get("microexpr_max_duration_frames"),
            "microexpr_min_peak_distance_frames": meta.get("microexpr_min_peak_distance_frames"),
            "gaze_centered_threshold": meta.get("gaze_centered_threshold"),
            "pca_components": meta.get("pca_components"),
            "au_confidence_threshold": meta.get("au_confidence_threshold"),
        }

    # --- Извлечение базовых данных из NPZ ---
    frame_indices = _as_int_array(npz_data.get("frame_indices"), dtype=np.int32)
    times_s = _as_float_array(npz_data.get("times_s"))
    face_present_any = _as_float_array(npz_data.get("face_present_any"))

    frame_feature_names_raw = npz_data.get("frame_feature_names")
    frame_features = npz_data.get("frame_features")

    if isinstance(frame_feature_names_raw, np.ndarray) and frame_feature_names_raw.dtype == object:
        frame_feature_names: List[str] = [str(x) for x in frame_feature_names_raw.flatten()]
    elif isinstance(frame_feature_names_raw, list):
        frame_feature_names = [str(x) for x in frame_feature_names_raw]
    else:
        frame_feature_names = []

    if isinstance(frame_features, list):
        frame_features = np.asarray(frame_features, dtype=np.float32)
    elif isinstance(frame_features, np.ndarray):
        frame_features = np.asarray(frame_features, dtype=np.float32)
    else:
        frame_features = None

    # Video-level scalar features (tabular)
    features: Dict[str, Any] = {}
    feature_names = npz_data.get("feature_names")
    feature_values = npz_data.get("feature_values")
    try:
        if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
            feature_names_list = [str(x) for x in feature_names.flatten()]
        elif isinstance(feature_names, list):
            feature_names_list = [str(x) for x in feature_names]
        else:
            feature_names_list = []
        if isinstance(feature_values, np.ndarray):
            feature_values_arr = np.asarray(feature_values, dtype=np.float32).reshape(-1)
        elif isinstance(feature_values, list):
            feature_values_arr = np.asarray(feature_values, dtype=np.float32).reshape(-1)
        else:
            feature_values_arr = np.asarray([], dtype=np.float32)
        for i, n in enumerate(feature_names_list):
            if i < int(feature_values_arr.size):
                v = float(feature_values_arr[i])
                features[str(n)] = v
    except Exception:
        features = {}
    microexpr_features = npz_data.get("microexpr_features") or {}
    summary_raw = npz_data.get("summary") or {}

    # --- Summary-блок ---
    def _safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            if not np.isfinite(f):
                return None
            return f
        except Exception:
            return None

    total_frames = int(summary_raw.get("total_frames") or (len(frame_indices) if frame_indices is not None else 0))
    frames_with_face = int(summary_raw.get("frames_with_face") or (int(np.sum(face_present_any > 0.5)) if face_present_any is not None else 0))
    fps = int(summary_raw.get("fps") or meta.get("analysis_fps") or 0)

    # Видео-уровневые агрегаты из features/microexpr_features
    smile_ratio = _safe_float(features.get("smile_ratio"))
    eye_contact_ratio = _safe_float(features.get("eye_contact_ratio") or features.get("gaze_centered_ratio"))
    blink_rate_per_min = _safe_float(features.get("blink_rate_per_min") or features.get("blink_rate"))
    pose_stability_score = _safe_float(features.get("pose_stability_score"))
    face_presence_ratio = _safe_float(features.get("face_presence_ratio"))
    au_quality_overall = _safe_float(features.get("au_quality_overall"))
    landmark_visibility_mean = _safe_float(features.get("landmark_visibility_mean"))

    microexpr_count = int(microexpr_features.get("microexpr_count") or 0)
    microexpr_rate_per_min = _safe_float(microexpr_features.get("microexpr_rate_per_min"))
    microexpr_max_intensity = _safe_float(microexpr_features.get("microexpr_max_intensity"))
    microexpr_types_distribution = microexpr_features.get("microexpr_types_distribution") or {}

    render["summary"] = {
        "frames_total": total_frames,
        "frames_with_face": frames_with_face,
        "fps": fps or None,
        "smile_ratio": smile_ratio,
        "eye_contact_ratio": eye_contact_ratio,
        "blink_rate_per_min": blink_rate_per_min,
        "pose_stability_score": pose_stability_score,
        "face_presence_ratio": face_presence_ratio,
        "au_quality_overall": au_quality_overall,
        "landmark_visibility_mean": landmark_visibility_mean,
        "microexpr_count": microexpr_count,
        "microexpr_rate_per_min": microexpr_rate_per_min,
        "microexpr_max_intensity": microexpr_max_intensity,
        "microexpr_types_distribution": microexpr_types_distribution,
        "success": bool(summary_raw.get("success", True)),
    }

    # --- Таймлайн по кадрам ---
    timeline: List[Dict[str, Any]] = []
    if (
        frame_indices is not None
        and times_s is not None
        and frame_features is not None
        and frame_features.ndim == 2
        and len(frame_feature_names) == frame_features.shape[1]
    ):
        name_to_idx = {name: i for i, name in enumerate(frame_feature_names)}

        def col(name: str) -> Optional[np.ndarray]:
            idx = name_to_idx.get(name)
            if idx is None or idx < 0 or idx >= frame_features.shape[1]:
                return None
            return frame_features[:, idx].astype(np.float32)

        # Вытащим несколько ключевых сигналов по кадрам, если доступны:
        au12_delta = col("AU12_delta")
        au06_delta = col("AU06_delta")
        pose_ry = col("pose_Ry")
        gaze_x = col("gaze_angle_x")
        gaze_y = col("gaze_angle_y")

        N = min(len(frame_indices), len(times_s))
        if face_present_any is not None:
            N = min(N, len(face_present_any))

        for i in range(N):
            entry: Dict[str, Any] = {
                "frame_index": int(frame_indices[i]),
                "time_sec": float(times_s[i]),
            }
            if face_present_any is not None:
                entry["face_present_any"] = bool(face_present_any[i] > 0.5)
            if au12_delta is not None:
                entry["AU12_delta"] = float(au12_delta[i])
            if au06_delta is not None:
                entry["AU06_delta"] = float(au06_delta[i])
            if pose_ry is not None:
                entry["pose_Ry"] = float(pose_ry[i])
            if gaze_x is not None:
                entry["gaze_angle_x"] = float(gaze_x[i])
            if gaze_y is not None:
                entry["gaze_angle_y"] = float(gaze_y[i])
            timeline.append(entry)

    render["timeline"] = timeline

    # --- Список событий micro-expressions ---
    ev_times = _as_float_array(npz_data.get("event_times_s"))
    ev_type_id = _as_int_array(npz_data.get("event_type_id"), dtype=np.int16)
    ev_strength = _as_float_array(npz_data.get("event_strength"))
    ev_types_str = (microexpr_features.get("microexpr_types") or []) if isinstance(microexpr_features.get("microexpr_types"), list) else []

    type_id_to_name = {1: "smile", 2: "surprise", 3: "frown", 4: "disgust"}

    events_list: List[Dict[str, Any]] = []
    if ev_times is not None and ev_type_id is not None:
        M = min(len(ev_times), len(ev_type_id))
        if ev_strength is not None:
            M = min(M, len(ev_strength))
        for i in range(M):
            t = float(ev_times[i])
            tid = int(ev_type_id[i])
            strength = float(ev_strength[i]) if ev_strength is not None and i < len(ev_strength) else None
            # Попробуем взять человекочитаемое имя из массива строк, иначе из словаря маппинга
            type_name = None
            if i < len(ev_types_str):
                type_name = str(ev_types_str[i])
            else:
                type_name = type_id_to_name.get(tid)
            events_list.append(
                {
                    "time_sec": t,
                    "type_id": tid,
                    "type_name": type_name,
                    "strength": strength,
                }
            )
    render["events"] = events_list

    # --- Распределения по ключевым скалярам ---
    distributions: Dict[str, Any] = {}

    # Распределение AU12_delta / AU06_delta по кадрам (если есть)
    if frame_features is not None and frame_features.ndim == 2 and len(frame_feature_names) == frame_features.shape[1]:
        for key in ["AU12_delta", "AU06_delta", "pose_Ry"]:
            idx = None
            for name in frame_feature_names:
                if name == key:
                    idx = name_to_idx.get(name)
                    break
            if idx is not None:
                vals = frame_features[:, idx]
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    distributions[key] = _stats(vals)

    # Распределение microexpr по типам (сохраняем как есть)
    if microexpr_types_distribution:
        distributions["microexpr_types_distribution"] = {
            "raw": dict(microexpr_types_distribution)
        }

    render["distributions"] = distributions
    return render


def render_micro_emotion_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага micro_emotion результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML

    Returns:
        Путь к сохранённому HTML файлу
    """
    # Импорт здесь, чтобы избежать циклических зависимостей
    import sys
    from pathlib import Path

    vp_root = Path(__file__).resolve().parent.parent.parent
    if str(vp_root / "modules") not in sys.path:
        sys.path.insert(0, str(vp_root / "modules"))

    # Пытаемся использовать общие utils.renderer
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except ImportError:
        # Fallback: прямой load
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
    render_ctx = render_micro_emotion(npz_data, meta)

    summary = render_ctx.get("summary", {})
    timeline = render_ctx.get("timeline", [])
    events = render_ctx.get("events", [])
    distributions = render_ctx.get("distributions", {})

    charts_html = ""
    top_au12: List[Tuple[int, float, float]] = []
    anti_au12: List[Tuple[int, float, float]] = []
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        face_present_series = [1.0 if t.get("face_present_any") else 0.0 for t in timeline]
        au12_series = [t.get("AU12_delta") for t in timeline]
        au06_series = [t.get("AU06_delta") for t in timeline]

        charts_html = "\n".join(
            [
                _svg_line_chart(times_s=times, values=face_present_series, title="face_present_any", stroke="#111827"),
                _svg_line_chart(times_s=times, values=au12_series, title="AU12_delta", stroke="#ef4444"),
                _svg_line_chart(times_s=times, values=au06_series, title="AU06_delta", stroke="#f59e0b"),
            ]
        )

        def _safe_float(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                fv = float(v)
                return fv if np.isfinite(fv) else None
            except Exception:
                return None

        scored: List[Tuple[float, int, float]] = []
        scored_anti: List[Tuple[float, int, float]] = []
        for t in timeline:
            fi = int(t.get("frame_index", -1))
            ts = float(t.get("time_sec", 0.0))
            v = _safe_float(t.get("AU12_delta"))
            if v is None:
                continue
            scored.append((v, fi, ts))
            scored_anti.append((v, fi, ts))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored_anti.sort(key=lambda x: x[0])
        for v, fi, ts in scored[:8]:
            top_au12.append((fi, ts, v))
        for v, fi, ts in scored_anti[:8]:
            anti_au12.append((fi, ts, v))

    # Подготовка таблички распределений microexpr типов
    microexpr_types_html = ""
    types_dist = distributions.get("microexpr_types_distribution", {}).get("raw") or {}
    if types_dist:
        rows = "".join(
            f"<tr><td>{t}</td><td>{int(c)}</td></tr>"
            for t, c in sorted(types_dist.items(), key=lambda x: x[0])
        )
        microexpr_types_html = f"""
        <div class="microexpr-types">
            <h3>Micro-expressions by type</h3>
            <table>
                <thead><tr><th>Type</th><th>Count</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    # Helper для распределений
    def format_dist_value(dist_key: str, stat_key: str) -> str:
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            try:
                return f"{float(dist[stat_key]):.4f}"
            except Exception:
                return "N/A"
        return "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micro Emotion Debug Render</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.1em; color: #333; }}
        .muted {{ color: #6b7280; }}
        .keyfacts {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #dee2e6; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin: 20px 0; }}
        .chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
        .chart-title {{ font-weight: 600; margin-bottom: 6px; }}
        .examples table {{ width: 100%; border-collapse: collapse; }}
        .examples th, .examples td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .examples th {{ background-color: #111827; color: white; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .microexpr-types table {{ width: 100%; border-collapse: collapse; }}
        .microexpr-types th, .microexpr-types td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .microexpr-types th {{ background-color: #0056b3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Micro Emotion Debug Render</h1>

        <div class="keyfacts">
            <h2>Key facts</h2>
            <div class="muted">{json.dumps(render_ctx.get("key_facts", {}), ensure_ascii=False)}</div>
            <div class="muted">stage_timings_ms: {json.dumps((render_ctx.get("key_facts", {}) or {}).get("stage_timings_ms", {}), ensure_ascii=False)}</div>
        </div>

        <div class="keyfacts">
            <h2>Config highlights</h2>
            <div class="muted">{json.dumps(render_ctx.get("config_highlights", {}), ensure_ascii=False)}</div>
        </div>

        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames (total / with face)</strong>
                    <span class="metric-value">{summary.get('frames_total', 0)} / {summary.get('frames_with_face', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Microexpr count / rate</strong>
                    <span class="metric-value">{summary.get('microexpr_count', 0)} / {summary.get('microexpr_rate_per_min', 0.0) or 0.0:.2f} per min</span>
                </div>
                <div class="metric-card">
                    <strong>Smile ratio</strong>
                    <span class="metric-value">{(summary.get('smile_ratio') or 0.0):.3f}</span>
                </div>
                <div class="metric-card">
                    <strong>Eye contact ratio</strong>
                    <span class="metric-value">{(summary.get('eye_contact_ratio') or 0.0):.3f}</span>
                </div>
                <div class="metric-card">
                    <strong>Blink rate (per min)</strong>
                    <span class="metric-value">{(summary.get('blink_rate_per_min') or 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>Pose stability</strong>
                    <span class="metric-value">{(summary.get('pose_stability_score') or 0.0):.3f}</span>
                </div>
                <div class="metric-card">
                    <strong>Face presence ratio</strong>
                    <span class="metric-value">{(summary.get('face_presence_ratio') or 0.0):.3f}</span>
                </div>
                <div class="metric-card">
                    <strong>AU & landmarks quality</strong>
                    <span class="metric-value">
                        AU {summary.get('au_quality_overall') if summary.get('au_quality_overall') is not None else 'N/A'} /
                        LM {(summary.get('landmark_visibility_mean') or 0.0):.3f}
                    </span>
                </div>
            </div>
        </div>

        <h2>Timeline charts (offline)</h2>
        {f'<div class="charts">{charts_html}</div>' if timeline else '<p class="muted">No timeline data available</p>'}

        <div class="examples">
            <h2>Top / Anti-top examples</h2>
            <h3>Top AU12_delta frames</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>AU12_delta</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in top_au12])}
              </tbody>
            </table>
            ''' if top_au12 else '<div class="muted">No data</div>'}

            <h3>Anti-top AU12_delta frames</h3>
            {f'''
            <table>
              <thead><tr><th>frame_index</th><th>time_s</th><th>AU12_delta</th></tr></thead>
              <tbody>
                {''.join([f'<tr><td>{fi}</td><td>{ts:.2f}</td><td>{sc:.4f}</td></tr>' for (fi, ts, sc) in anti_au12])}
              </tbody>
            </table>
            ''' if anti_au12 else '<div class="muted">No data</div>'}
        </div>

        <div class="distributions">
            <h2>Distributions (per-frame)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Median</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>AU12_delta</strong></td>
                        <td>{format_dist_value('AU12_delta', 'min')}</td>
                        <td>{format_dist_value('AU12_delta', 'max')}</td>
                        <td>{format_dist_value('AU12_delta', 'mean')}</td>
                        <td>{format_dist_value('AU12_delta', 'std')}</td>
                        <td>{format_dist_value('AU12_delta', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>AU06_delta</strong></td>
                        <td>{format_dist_value('AU06_delta', 'min')}</td>
                        <td>{format_dist_value('AU06_delta', 'max')}</td>
                        <td>{format_dist_value('AU06_delta', 'mean')}</td>
                        <td>{format_dist_value('AU06_delta', 'std')}</td>
                        <td>{format_dist_value('AU06_delta', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>pose_Ry</strong></td>
                        <td>{format_dist_value('pose_Ry', 'min')}</td>
                        <td>{format_dist_value('pose_Ry', 'max')}</td>
                        <td>{format_dist_value('pose_Ry', 'mean')}</td>
                        <td>{format_dist_value('pose_Ry', 'std')}</td>
                        <td>{format_dist_value('pose_Ry', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        {microexpr_types_html}
    </div>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Micro Emotion HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_micro_emotion", "render_micro_emotion_html"]


