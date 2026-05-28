"""
Renderer for core_face_landmarks: render-context JSON + offline HTML mini-dashboard + preview assets.

Audit v3 rule: render.html must work offline (no CDN) and provide a QA-friendly mini-dashboard:
- navigation/sections
- key facts (schema/models/timings)
- top/anti-top peaks with jump-to
- interactive tables (search/sort)
- preview examples with overlays saved into _render/assets/
- explicit privacy banner (faces).
"""

import os
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _svg_multi_line_chart(
    *,
    times_s: List[float],
    series: List[Tuple[str, List[float], str]],
    title: str,
    width: int = 960,
    height: int = 240,
    pad: int = 18,
) -> str:
    if not times_s or not series:
        return ""
    # Use only finite points for bounds.
    x = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return ""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        xmax = xmin + 1e-6
    # y bounds across all series
    ys = []
    for _, vals, _ in series:
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        m = np.isfinite(v)
        if np.any(m):
            ys.append(v[m])
    if not ys:
        return ""
    ycat = np.concatenate(ys, axis=0)
    ymin, ymax = float(np.min(ycat)), float(np.max(ycat))
    if ymax <= ymin:
        ymax = ymin + 1e-6

    def sx(v: float) -> float:
        return pad + (v - xmin) / (xmax - xmin) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - ymin) / (ymax - ymin) * (height - 2 * pad)

    polys = []
    legend = []
    for name, vals, color in series:
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        n = min(int(x.size), int(v.size))
        pts = []
        for i in range(n):
            if not np.isfinite(v[i]):
                continue
            pts.append(f"{sx(float(x[i])):.2f},{sy(float(v[i])):.2f}")
        if len(pts) >= 2:
            polys.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>')
            legend.append(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;"><span style="width:10px;height:10px;background:{color};display:inline-block;border-radius:2px;"></span>{_esc(name)}</span>')

    if not polys:
        return ""
    return f"""
<div class="chart-container">
  <h2>{_esc(title)}</h2>
  <div style="margin:6px 0 10px 0; color:#6b7280; font-size:13px;">{' '.join(legend)}</div>
  <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{_esc(title)}">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#e5e7eb"/>
    {''.join(polys)}
    <text x="{pad}" y="{pad}" font-size="12" fill="#6b7280">{_esc(f'[{ymin:.2f} .. {ymax:.2f}]')}</text>
  </svg>
</div>
""".strip()

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _compute_face_bbox_and_area_norm(face_lm: np.ndarray) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
    """
    Compute bbox (x1,y1,x2,y2) and area in normalized coords from a single face landmarks array (468,3).
    Returns (bbox, area_norm). If invalid -> (None, 0.0).
    """
    if face_lm is None or not isinstance(face_lm, np.ndarray) or face_lm.size == 0:
        return None, 0.0
    # face_lm: (468,3) in normalized coords; may have NaNs if face missing
    xs = face_lm[:, 0]
    ys = face_lm[:, 1]
    m = (~np.isnan(xs)) & (~np.isnan(ys))
    if not np.any(m):
        return None, 0.0
    x1 = float(np.min(xs[m]))
    y1 = float(np.min(ys[m]))
    x2 = float(np.max(xs[m]))
    y2 = float(np.max(ys[m]))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    return (x1, y1, x2, y2), float(area)

def _sort_table(rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
    def k(r):
        v = r.get(key)
        try:
            return float(v)
        except Exception:
            return -1e9
    return sorted(rows, key=k, reverse=reverse)

def _pick_examples(
    rows: List[Dict[str, Any]],
    *,
    k_total: int = 12,
    k_peaks: int = 6,
    k_antipeaks: int = 6,
) -> List[Dict[str, Any]]:
    peaks = [r for r in rows if r.get("face_area_norm", 0.0) > 0.0]
    peaks = _sort_table(peaks, "face_area_norm", reverse=True)[:k_peaks]

    anti = [r for r in rows if (r.get("face_mesh_ran") and not r.get("has_face"))]
    # Prefer "strong person but no face" frames if available
    anti = _sort_table(anti, "person_present", reverse=True)[:k_antipeaks]

    out: List[Dict[str, Any]] = []
    seen = set()
    for r in peaks + anti:
        fi = r.get("frame_index")
        if fi in seen:
            continue
        seen.add(fi)
        out.append(r)
        if len(out) >= k_total:
            break
    return out


def render_core_face_landmarks(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для core_face_landmarks."""
    render = {
        "component": "core_face_landmarks",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract landmarks data
    face_landmarks = npz_data.get("face_landmarks")  # filtered (default)
    face_landmarks_raw = npz_data.get("face_landmarks_raw")
    face_present = npz_data.get("face_present")
    person_present = npz_data.get("person_present")
    face_mesh_ran = npz_data.get("face_mesh_ran")
    pose_landmarks = npz_data.get("pose_landmarks")
    pose_present = npz_data.get("pose_present")
    hands_landmarks = npz_data.get("hands_landmarks")
    hands_present = npz_data.get("hands_present")
    has_any_face = npz_data.get("has_any_face")
    has_any_pose = npz_data.get("has_any_pose")
    has_any_hands = npz_data.get("has_any_hands")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if face_landmarks is not None:
        if isinstance(face_landmarks, list):
            face_landmarks = np.array(face_landmarks, dtype=np.float32)
        elif isinstance(face_landmarks, np.ndarray):
            face_landmarks = np.asarray(face_landmarks, dtype=np.float32)
        else:
            face_landmarks = None

    if face_landmarks_raw is not None:
        if isinstance(face_landmarks_raw, list):
            face_landmarks_raw = np.array(face_landmarks_raw, dtype=np.float32)
        elif isinstance(face_landmarks_raw, np.ndarray):
            face_landmarks_raw = np.asarray(face_landmarks_raw, dtype=np.float32)
        else:
            face_landmarks_raw = None
    
    if face_present is not None:
        if isinstance(face_present, list):
            face_present = np.array(face_present, dtype=bool)
        elif isinstance(face_present, np.ndarray):
            face_present = np.asarray(face_present, dtype=bool)
        else:
            face_present = None
    
    if pose_landmarks is not None:
        if isinstance(pose_landmarks, list):
            pose_landmarks = np.array(pose_landmarks, dtype=np.float32)
        elif isinstance(pose_landmarks, np.ndarray):
            pose_landmarks = np.asarray(pose_landmarks, dtype=np.float32)
        else:
            pose_landmarks = None
    
    if pose_present is not None:
        if isinstance(pose_present, list):
            pose_present = np.array(pose_present, dtype=bool)
        elif isinstance(pose_present, np.ndarray):
            pose_present = np.asarray(pose_present, dtype=bool)
        else:
            pose_present = None
    
    if hands_landmarks is not None:
        if isinstance(hands_landmarks, list):
            hands_landmarks = np.array(hands_landmarks, dtype=np.float32)
        elif isinstance(hands_landmarks, np.ndarray):
            hands_landmarks = np.asarray(hands_landmarks, dtype=np.float32)
        else:
            hands_landmarks = None
    
    if hands_present is not None:
        if isinstance(hands_present, list):
            hands_present = np.array(hands_present, dtype=bool)
        elif isinstance(hands_present, np.ndarray):
            hands_present = np.asarray(hands_present, dtype=bool)
        else:
            hands_present = None
    
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

    if person_present is not None:
        person_present = np.asarray(person_present, dtype=bool).reshape(-1)
    if face_mesh_ran is not None:
        face_mesh_ran = np.asarray(face_mesh_ran, dtype=bool).reshape(-1)
    
    # Summary statistics
    n_frames = 0
    if frame_indices is not None:
        n_frames = len(frame_indices)
    elif face_landmarks is not None:
        n_frames = face_landmarks.shape[0] if face_landmarks.ndim >= 1 else 0
    elif pose_landmarks is not None:
        n_frames = pose_landmarks.shape[0] if pose_landmarks.ndim >= 1 else 0
    
    render["summary"] = {
        "frames_count": int(n_frames),
        "has_any_face": bool(has_any_face) if has_any_face is not None else False,
        "has_any_pose": bool(has_any_pose) if has_any_pose is not None else False,
        "has_any_hands": bool(has_any_hands) if has_any_hands is not None else False,
        "status": str(meta.get("status", "unknown")),
        "empty_reason": meta.get("empty_reason"),
        "schema_version": str(meta.get("schema_version", "unknown")),
        "producer_version": str(meta.get("producer_version", "unknown")),
        "models_used": meta.get("models_used", []),
        "person_frames_count": _safe_int(meta.get("person_frames_count", 0)),
        "face_mesh_frames_count": _safe_int(meta.get("face_mesh_frames_count", 0)),
        "temporal_filter_enabled": bool(meta.get("temporal_filter_enabled", False)),
    }
    
    # Face statistics
    if face_present is not None and face_present.size > 0:
        if face_present.ndim == 1:
            # Single face per frame
            face_count_per_frame = face_present.astype(int)
        elif face_present.ndim == 2:
            # Multiple faces per frame
            face_count_per_frame = np.sum(face_present.astype(int), axis=1)
        else:
            face_count_per_frame = np.array([0])
        
        frames_with_face = np.sum(face_count_per_frame > 0)
        render["summary"]["face_frames_count"] = int(frames_with_face)
        render["summary"]["face_frames_percentage"] = float(frames_with_face / n_frames * 100) if n_frames > 0 else 0.0
        render["summary"]["face_count_mean"] = float(np.mean(face_count_per_frame))
        render["summary"]["face_count_max"] = int(np.max(face_count_per_frame)) if face_count_per_frame.size > 0 else 0
    else:
        render["summary"]["face_frames_count"] = 0
        render["summary"]["face_frames_percentage"] = 0.0
        render["summary"]["face_count_mean"] = 0.0
        render["summary"]["face_count_max"] = 0
    
    # Pose statistics
    if pose_present is not None and pose_present.size > 0:
        frames_with_pose = np.sum(pose_present.astype(int))
        render["summary"]["pose_frames_count"] = int(frames_with_pose)
        render["summary"]["pose_frames_percentage"] = float(frames_with_pose / n_frames * 100) if n_frames > 0 else 0.0
    else:
        render["summary"]["pose_frames_count"] = 0
        render["summary"]["pose_frames_percentage"] = 0.0
    
    # Hands statistics
    if hands_present is not None and hands_present.size > 0:
        if hands_present.ndim == 1:
            hands_count_per_frame = hands_present.astype(int)
        elif hands_present.ndim == 2:
            hands_count_per_frame = np.sum(hands_present.astype(int), axis=1)
        else:
            hands_count_per_frame = np.array([0])
        
        frames_with_hands = np.sum(hands_count_per_frame > 0)
        render["summary"]["hands_frames_count"] = int(frames_with_hands)
        render["summary"]["hands_frames_percentage"] = float(frames_with_hands / n_frames * 100) if n_frames > 0 else 0.0
        render["summary"]["hands_count_mean"] = float(np.mean(hands_count_per_frame))
        render["summary"]["hands_count_max"] = int(np.max(hands_count_per_frame)) if hands_count_per_frame.size > 0 else 0
    else:
        render["summary"]["hands_frames_count"] = 0
        render["summary"]["hands_frames_percentage"] = 0.0
        render["summary"]["hands_count_mean"] = 0.0
        render["summary"]["hands_count_max"] = 0
    
    # Timeline data (also acts as table data)
    if times_s is not None and frame_indices is not None and n_frames > 0:
        timeline = []
        
        for i in range(min(n_frames, len(times_s), len(frame_indices))):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            
            # Face presence
            has_face = False
            face_count = 0
            face_area_norm = 0.0
            if face_present is not None:
                if face_present.ndim == 1 and i < len(face_present):
                    has_face = bool(face_present[i])
                    face_count = 1 if has_face else 0
                elif face_present.ndim == 2 and i < face_present.shape[0]:
                    has_face = bool(np.any(face_present[i]))
                    face_count = int(np.sum(face_present[i].astype(int)))

            # Face area from RAW landmarks (use first face slot), only if face_present==True.
            if has_face and face_landmarks_raw is not None and isinstance(face_landmarks_raw, np.ndarray):
                try:
                    if face_landmarks_raw.ndim >= 4 and i < face_landmarks_raw.shape[0]:
                        # (N, FACES, 468, 3)
                        lm0 = np.asarray(face_landmarks_raw[i, 0], dtype=np.float32)
                        _, face_area_norm = _compute_face_bbox_and_area_norm(lm0)
                except Exception:
                    face_area_norm = 0.0
            
            # Pose presence
            has_pose = False
            if pose_present is not None and i < len(pose_present):
                has_pose = bool(pose_present[i])
            
            # Hands presence
            has_hands = False
            hands_count = 0
            if hands_present is not None:
                if hands_present.ndim == 1 and i < len(hands_present):
                    has_hands = bool(hands_present[i])
                    hands_count = 1 if has_hands else 0
                elif hands_present.ndim == 2 and i < hands_present.shape[0]:
                    has_hands = bool(np.any(hands_present[i]))
                    hands_count = int(np.sum(hands_present[i].astype(int)))
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "person_present": bool(person_present[i]) if isinstance(person_present, np.ndarray) and i < person_present.shape[0] else False,
                "face_mesh_ran": bool(face_mesh_ran[i]) if isinstance(face_mesh_ran, np.ndarray) and i < face_mesh_ran.shape[0] else False,
                "has_face": has_face,
                "face_count": face_count,
                "face_area_norm": float(face_area_norm),
                "has_pose": has_pose,
                "has_hands": has_hands,
                "hands_count": hands_count,
            })
        
        render["timeline"] = timeline

        # QA peaks (top/anti-top) for jump-to in HTML
        rows = timeline
        render["qa"] = {
            "top_face_area": _sort_table([r for r in rows if r.get("face_area_norm", 0.0) > 0.0], "face_area_norm", reverse=True)[:10],
            "anti_no_face_when_ran": [r for r in rows if (r.get("face_mesh_ran") and not r.get("has_face"))][:10],
        }
    
    # Distributions
    distributions = {}
    
    if face_present is not None and face_present.size > 0:
        if face_present.ndim == 1:
            face_counts = face_present.astype(int)
        elif face_present.ndim == 2:
            face_counts = np.sum(face_present.astype(int), axis=1)
        else:
            face_counts = np.array([0])
        
        if face_counts.size > 0:
            distributions["face_count_per_frame"] = {
                "min": int(np.min(face_counts)),
                "max": int(np.max(face_counts)),
                "mean": float(np.mean(face_counts)),
                "std": float(np.std(face_counts)),
                "median": float(np.median(face_counts)),
            }
    
    if hands_present is not None and hands_present.size > 0:
        if hands_present.ndim == 1:
            hands_counts = hands_present.astype(int)
        elif hands_present.ndim == 2:
            hands_counts = np.sum(hands_present.astype(int), axis=1)
        else:
            hands_counts = np.array([0])
        
        if hands_counts.size > 0:
            distributions["hands_count_per_frame"] = {
                "min": int(np.min(hands_counts)),
                "max": int(np.max(hands_counts)),
                "mean": float(np.mean(hands_counts)),
                "std": float(np.std(hands_counts)),
                "median": float(np.median(hands_counts)),
            }
    
    render["distributions"] = distributions
    
    return render

# NOTE: HTML renderer must accept frames_dir/assets_dir kwargs (see utils/renderer.py).
def render_core_face_landmarks_html(
    npz_path: str,
    output_path: str,
    *,
    frames_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
) -> str:
    """
    Offline HTML mini-dashboard for core_face_landmarks + optional preview assets.
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))

    from utils.renderer import load_npz, extract_meta  # type: ignore
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    rc = render_core_face_landmarks(npz_data, meta)

    rows: List[Dict[str, Any]] = list(rc.get("timeline", []) or [])
    summary = rc.get("summary", {}) or {}
    qa = rc.get("qa", {}) or {}

    # Generate preview assets (best-effort)
    preview_cards: List[Dict[str, Any]] = []
    if frames_dir and assets_dir and rows:
        try:
            import cv2  # type: ignore
            from utils.frame_manager import FrameManager  # type: ignore

            os.makedirs(assets_dir, exist_ok=True)

            # Pick examples: peaks + anti-peaks
            examples = _pick_examples(rows, k_total=12, k_peaks=6, k_antipeaks=6)
            # Load arrays for overlay drawing (filtered landmarks)
            z = np.load(npz_path, allow_pickle=True)
            # Prefer RAW landmarks for visualization (filtered may be interpolated/smoothed).
            face_lm_arr = z["face_landmarks_raw"] if "face_landmarks_raw" in z.files else (z["face_landmarks"] if "face_landmarks" in z.files else None)
        except Exception as e:
            logger.warning(f"core_face_landmarks | render | could not init preview generator: {e}")
            examples = []
            face_lm_arr = None

        if examples:
            try:
                fm = FrameManager(frames_dir)
            except Exception as e:
                logger.warning(f"core_face_landmarks | render | FrameManager init failed: {e}")
                fm = None

            if fm is not None:
                try:
                    if isinstance(face_lm_arr, np.ndarray):
                        face_lm_arr = np.asarray(face_lm_arr, dtype=np.float32)
                    else:
                        face_lm_arr = None

                    for r in examples:
                        fi = int(r["frame_index"])
                        # Find position i in primary list
                        # (rows are aligned; easiest lookup by index in rows list)
                        pos = None
                        # small N, linear scan ok
                        for j, rr in enumerate(rows):
                            if int(rr.get("frame_index")) == fi:
                                pos = j
                                break
                        if pos is None:
                            continue

                        fr = fm.get(fi)  # RGB uint8
                        h, w = int(fr.shape[0]), int(fr.shape[1])

                        # Downscale for privacy + small artifacts
                        max_side = max(h, w)
                        scale = 1.0
                        target_max = 640
                        if max_side > target_max:
                            scale = float(target_max) / float(max_side)
                            fr = cv2.resize(fr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                            h, w = int(fr.shape[0]), int(fr.shape[1])

                        # Draw overlay: bbox + landmarks (first face slot)
                        # Only draw face overlay when face_present==True for that row (avoid boxes on missing frames).
                        row_has_face = bool(r.get("has_face", False))
                        if row_has_face and face_lm_arr is not None and face_lm_arr.ndim >= 4 and pos < face_lm_arr.shape[0]:
                            lm0 = np.asarray(face_lm_arr[pos, 0], dtype=np.float32)  # (468,3)
                            bbox, _ = _compute_face_bbox_and_area_norm(lm0)
                            # Convert to BGR for cv2 drawing
                            img = fr[:, :, ::-1].copy()
                            if bbox is not None:
                                x1, y1, x2, y2 = bbox
                                p1 = (int(round(x1 * w)), int(round(y1 * h)))
                                p2 = (int(round(x2 * w)), int(round(y2 * h)))
                                cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
                            # Landmarks
                            xs = lm0[:, 0]
                            ys = lm0[:, 1]
                            m = (~np.isnan(xs)) & (~np.isnan(ys))
                            idxs = np.nonzero(m)[0]
                            # Limit draw cost: draw every 2nd point
                            for k in idxs[::2]:
                                x = int(round(float(xs[k]) * w))
                                y = int(round(float(ys[k]) * h))
                                cv2.circle(img, (x, y), 1, (255, 255, 0), -1)
                        else:
                            img = fr[:, :, ::-1].copy()

                        # Deterministic filename
                        tag = f"fi={fi}|pos={pos}|schema={summary.get('schema_version','')}"
                        fn = f"face_landmarks_{_sha1_text(tag)[:12]}.jpg"
                        out_img_path = os.path.join(assets_dir, fn)
                        if not os.path.exists(out_img_path):
                            cv2.imwrite(out_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                        preview_cards.append(
                            {
                                "frame_index": fi,
                                "time_sec": float(r.get("time_sec", 0.0)),
                                "face_area_norm": float(r.get("face_area_norm", 0.0)),
                                "has_face": bool(r.get("has_face")),
                                "person_present": bool(r.get("person_present")),
                                "face_mesh_ran": bool(r.get("face_mesh_ran")),
                                "asset_relpath": f"assets/{fn}",
                            }
                        )
                finally:
                    try:
                        fm.close()
                    except Exception:
                        pass

    # Interactive table data
    table_rows = rows
    # Limit embedding size in HTML: keep only key fields
    table_rows_compact = [
        {
            "frame_index": int(r.get("frame_index", 0)),
            "time_sec": float(r.get("time_sec", 0.0)),
            "person_present": bool(r.get("person_present", False)),
            "face_mesh_ran": bool(r.get("face_mesh_ran", False)),
            "has_face": bool(r.get("has_face", False)),
            "face_count": int(r.get("face_count", 0)),
            "face_area_norm": float(r.get("face_area_norm", 0.0)),
            "has_pose": bool(r.get("has_pose", False)),
            "has_hands": bool(r.get("has_hands", False)),
            "hands_count": int(r.get("hands_count", 0)),
        }
        for r in table_rows
    ]

    key_facts = {
        "status": summary.get("status"),
        "empty_reason": summary.get("empty_reason"),
        "schema_version": summary.get("schema_version"),
        "producer_version": summary.get("producer_version"),
        "frames_count": summary.get("frames_count"),
        "person_frames_count": summary.get("person_frames_count"),
        "face_mesh_frames_count": summary.get("face_mesh_frames_count"),
        "face_frames_count": summary.get("face_frames_count"),
        "face_frames_percentage": summary.get("face_frames_percentage"),
        "temporal_filter_enabled": summary.get("temporal_filter_enabled"),
    }

    # Minimal config highlights from meta
    cfg_hi = {
        "use_person_mask": True,
        "person_window_radius": meta.get("person_window_radius"),
        "face_mesh_max_num_faces": meta.get("face_mesh_max_num_faces"),
        "face_mesh_min_tracking_confidence": meta.get("face_mesh_min_tracking_confidence"),
        "temporal_filter_enabled": meta.get("temporal_filter_enabled"),
        "temporal_filter_min_cutoff": meta.get("temporal_filter_min_cutoff"),
        "temporal_filter_beta": meta.get("temporal_filter_beta"),
    }

    # Offline HTML (no CDN)
    data_json = json.dumps(
        {
            "key_facts": key_facts,
            "config_highlights": cfg_hi,
            "qa": qa,
            "table_rows": table_rows_compact,
            "previews": preview_cards,
        },
        ensure_ascii=False,
    )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>core_face_landmarks — render</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #0f172a;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #60a5fa;
      --danger: #fb7185;
      --ok: #34d399;
      --border: rgba(255,255,255,0.08);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
    }}
    body {{ margin:0; background: linear-gradient(180deg, #070a14, var(--bg)); color: var(--text); font-family: var(--sans); }}
    a {{ color: var(--accent); text-decoration: none; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 18px 16px 60px; }}
    .topbar {{ display:flex; gap:12px; align-items:center; justify-content:space-between; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .pill {{ font-family: var(--mono); font-size: 12px; padding: 6px 10px; border:1px solid var(--border); border-radius:999px; color: var(--muted); }}
    .nav {{ display:flex; gap:10px; flex-wrap:wrap; margin: 14px 0 6px; }}
    .nav a {{ padding: 7px 10px; border:1px solid var(--border); border-radius:10px; color: var(--text); }}
    .banner {{ margin: 14px 0; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border); background: rgba(255,255,255,0.04); }}
    .banner strong {{ display:block; margin-bottom: 4px; }}
    .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }}
    .card {{ grid-column: span 6; padding: 14px; border-radius: 14px; border: 1px solid var(--border); background: rgba(255,255,255,0.03); }}
    .card h3 {{ margin: 0 0 10px; font-size: 14px; color: var(--muted); font-weight: 600; letter-spacing: .02em; text-transform: uppercase; }}
    .kv {{ display:grid; grid-template-columns: 220px 1fr; gap: 8px 12px; font-family: var(--mono); font-size: 12px; }}
    .k {{ color: var(--muted); }}
    .v {{ color: var(--text); }}
    .section {{ margin-top: 18px; }}
    .section h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .controls {{ display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0; }}
    input[type="search"], input[type="number"] {{
      background: rgba(0,0,0,0.25); border:1px solid var(--border); color: var(--text);
      padding: 8px 10px; border-radius: 10px; outline: none; font-family: var(--mono); font-size: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }}
    th, td {{ padding: 8px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ cursor: pointer; color: var(--muted); position: sticky; top: 0; background: rgba(9, 12, 25, 0.95); }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .tag-ok {{ color: var(--ok); }}
    .tag-bad {{ color: var(--danger); }}
    .gallery {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .shot {{ border:1px solid var(--border); border-radius: 14px; overflow:hidden; background: rgba(255,255,255,0.03); }}
    .shot img {{ width:100%; display:block; }}
    .shot .meta {{ padding: 10px 12px; font-family: var(--mono); font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title">core_face_landmarks — render (dev-only)</div>
      <div class="pill">offline • mini-dashboard • privacy-aware</div>
    </div>

    <div class="nav">
      <a href="#overview">Overview</a>
      <a href="#qa">QA (top/anti-top)</a>
      <a href="#previews">Previews</a>
      <a href="#table">Table</a>
    </div>

    <div class="banner">
      <strong>Privacy notice</strong>
      This page may contain downscaled frames with face landmarks overlays. Use only for local QA in dev; do not publish or share.
    </div>

    <div id="overview" class="grid">
      <div class="card">
        <h3>Key facts</h3>
        <div id="keyFacts" class="kv"></div>
      </div>
      <div class="card">
        <h3>Config highlights</h3>
        <div id="cfgHi" class="kv"></div>
      </div>
    </div>

    <div id="qa" class="section">
      <h2>QA peaks</h2>
      <div class="grid">
        <div class="card">
          <h3>Top face area (norm)</h3>
          <div id="topPeaks"></div>
        </div>
        <div class="card">
          <h3>Anti-top: ran FaceMesh but no face detected</h3>
          <div id="antiPeaks"></div>
        </div>
      </div>
    </div>

    <div id="previews" class="section">
      <h2>Preview examples</h2>
      <div id="gallery" class="gallery"></div>
      <div style="color: var(--muted); font-family: var(--mono); font-size: 12px; margin-top: 8px;">
        If gallery is empty: frames_dir/assets_dir were not available during render generation.
      </div>
    </div>

    <div id="table" class="section">
      <h2>All frames (search/sort)</h2>
      <div class="controls">
        <input id="q" type="search" placeholder="search (frame_index/time)..." />
        <input id="minArea" type="number" step="0.0001" placeholder="min face_area_norm" />
      </div>
      <div style="overflow:auto; border:1px solid var(--border); border-radius: 14px;">
        <table id="tbl">
          <thead>
            <tr>
              <th data-k="frame_index">frame</th>
              <th data-k="time_sec">time_s</th>
              <th data-k="person_present">person</th>
              <th data-k="face_mesh_ran">ran</th>
              <th data-k="has_face">face</th>
              <th data-k="face_count">face_n</th>
              <th data-k="face_area_norm">face_area</th>
              <th data-k="hands_count">hands_n</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <script>
      const DATA = {data_json};

      function kv(el, obj) {{
        const rows = Object.entries(obj || {{}}).map(([k,v]) => {{
          const vv = (v === null || v === undefined) ? "" : (typeof v === "object" ? JSON.stringify(v) : String(v));
          return `<div class="k">${{k}}</div><div class="v">${{vv}}</div>`;
        }}).join("");
        el.innerHTML = rows;
      }}

      kv(document.getElementById("keyFacts"), DATA.key_facts);
      kv(document.getElementById("cfgHi"), DATA.config_highlights);

      function peaksList(rows) {{
        if (!rows || !rows.length) return `<div style="color: var(--muted); font-family: var(--mono); font-size: 12px;">(empty)</div>`;
        const items = rows.slice(0,10).map(r => {{
          const a = (r.face_area_norm ?? 0).toFixed(4);
          const href = `#row-${{r.frame_index}}`;
          return `<div style="font-family: var(--mono); font-size: 12px; margin: 6px 0;">
            <a href="${{href}}">frame=${{r.frame_index}}</a> time=${{(r.time_sec??0).toFixed(2)}}s area=${{a}} person=${{r.person_present?1:0}} ran=${{r.face_mesh_ran?1:0}} face=${{r.has_face?1:0}}
          </div>`;
        }}).join("");
        return items;
      }}

      document.getElementById("topPeaks").innerHTML = peaksList(DATA.qa?.top_face_area || []);
      document.getElementById("antiPeaks").innerHTML = peaksList(DATA.qa?.anti_no_face_when_ran || []);

      // Gallery
      const g = document.getElementById("gallery");
      const previews = DATA.previews || [];
      g.innerHTML = previews.map(p => {{
        const meta = `frame=${{p.frame_index}} time=${{p.time_sec.toFixed(2)}}s area=${{(p.face_area_norm||0).toFixed(4)}} person=${{p.person_present?1:0}} ran=${{p.face_mesh_ran?1:0}} face=${{p.has_face?1:0}}`;
        return `<div class="shot">
          <img src="${{p.asset_relpath}}" loading="lazy"/>
          <div class="meta">${{meta}}</div>
        </div>`;
      }}).join("");

      // Table
      let rows = DATA.table_rows || [];
      let sortKey = "frame_index";
      let sortDir = 1;

      function rowMatches(r, q, minArea) {{
        if (minArea !== null && (r.face_area_norm ?? 0) < minArea) return false;
        if (!q) return true;
        const s = `${{r.frame_index}} ${{r.time_sec}}`.toLowerCase();
        return s.includes(q);
      }}

      function renderTable() {{
        const q = (document.getElementById("q").value || "").trim().toLowerCase();
        const minAreaRaw = (document.getElementById("minArea").value || "").trim();
        const minArea = minAreaRaw ? Number(minAreaRaw) : null;

        const out = rows
          .filter(r => rowMatches(r, q, minArea))
          .sort((a,b) => {{
            const av = a[sortKey], bv = b[sortKey];
            const na = (typeof av === "number") ? av : Number(av);
            const nb = (typeof bv === "number") ? bv : Number(bv);
            if (!Number.isNaN(na) && !Number.isNaN(nb)) return (na - nb) * sortDir;
            return String(av).localeCompare(String(bv)) * sortDir;
          }});

        const tb = document.querySelector("#tbl tbody");
        tb.innerHTML = out.map(r => {{
          const yn = (b) => b ? `<span class="tag-ok">yes</span>` : `<span class="tag-bad">no</span>`;
          return `<tr id="row-${{r.frame_index}}">
            <td>${{r.frame_index}}</td>
            <td>${{r.time_sec.toFixed(2)}}</td>
            <td>${{yn(r.person_present)}}</td>
            <td>${{yn(r.face_mesh_ran)}}</td>
            <td>${{yn(r.has_face)}}</td>
            <td>${{r.face_count}}</td>
            <td>${{(r.face_area_norm||0).toFixed(4)}}</td>
            <td>${{r.hands_count}}</td>
          </tr>`;
        }}).join("");
      }}

      document.getElementById("q").addEventListener("input", renderTable);
      document.getElementById("minArea").addEventListener("input", renderTable);
      document.querySelectorAll("#tbl thead th").forEach(th => {{
        th.addEventListener("click", () => {{
          const k = th.getAttribute("data-k");
          if (!k) return;
          if (sortKey === k) sortDir *= -1;
          else {{ sortKey = k; sortDir = 1; }}
          renderTable();
        }});
      }});
      renderTable();
    </script>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved Core Face Landmarks HTML render to {output_path}")
    return output_path
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        face_counts = [t.get("face_count", 0) for t in timeline]
        hands_counts = [t.get("hands_count", 0) for t in timeline]
        
        timeline_svg = _svg_multi_line_chart(
            times_s=[float(t) for t in times],
            series=[
                ("Face Count", [float(x) for x in face_counts], "#4bc0c0"),
                ("Hands Count", [float(x) for x in hands_counts], "#ff6384"),
            ],
            title="Timeline: Face and Hands Count Over Time",
        )
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Core Face Landmarks Debug Render</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Core Face Landmarks Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Has Any Face</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_any_face', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Face Frames</strong>
                    <span class="metric-value">{summary.get('face_frames_count', 0)} ({summary.get('face_frames_percentage', 0.0):.1f}%)</span>
                </div>
                <div class="metric-card">
                    <strong>Face Count Mean</strong>
                    <span class="metric-value">{summary.get('face_count_mean', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>Face Count Max</strong>
                    <span class="metric-value">{summary.get('face_count_max', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Has Any Pose</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_any_pose', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Pose Frames</strong>
                    <span class="metric-value">{summary.get('pose_frames_count', 0)} ({summary.get('pose_frames_percentage', 0.0):.1f}%)</span>
                </div>
                <div class="metric-card">
                    <strong>Has Any Hands</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_any_hands', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Hands Frames</strong>
                    <span class="metric-value">{summary.get('hands_frames_count', 0)} ({summary.get('hands_frames_percentage', 0.0):.1f}%)</span>
                </div>
                <div class="metric-card">
                    <strong>Hands Count Mean</strong>
                    <span class="metric-value">{summary.get('hands_count_mean', 0.0):.2f}</span>
                </div>
            </div>
        </div>
        
        {timeline_svg if timeline else '<p>No timeline data available</p>'}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Face Count</th>
                        <th>Hands Count</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('min', 0)}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('min', 0)}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('max', 0)}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('max', 0)}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('mean', 0.0):.2f}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('mean', 0.0):.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('std', 0.0):.2f}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('std', 0.0):.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('median', 0.0):.2f}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('median', 0.0):.2f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
    </div>
    
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved Core Face Landmarks HTML render to {output_path}")
    return output_path


__all__ = ["render_core_face_landmarks", "render_core_face_landmarks_html"]

