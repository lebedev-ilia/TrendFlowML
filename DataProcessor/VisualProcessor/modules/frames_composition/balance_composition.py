"""
frames_composition (baseline-ready)
----------------------------------
Source of truth: NPZ artifact written via BaseModule (no JSON artifacts).

Key baseline policies implemented here:
- No-fallback: frame_indices must be provided by Segmenter in frames_dir/metadata.json[frames_composition.frame_indices]
- NPZ meta contract via BaseModule.save_results() + artifact_validator.validate_npz()
- Valid empty: if there are no faces in the video -> status="empty", empty_reason="no_faces_in_video"
- Fixed artifact filename (per-run unique by run_id path)
- Feature gating: all output features are controllable via CLI args (features/feature_set)
- Internal parallelism: concurrent per-frame compute with safe frame loading
- Progress reporting: append-only events to state_events.jsonl (PR-5) when available

Notes:
- This module does NOT load ML models directly (it consumes core providers).
- FrameManager.get() is assumed to return RGB uint8 frames (per Segmenter contract).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger


MODULE_NAME = "frames_composition"
VERSION = "1.0"
SCHEMA_VERSION = "frames_composition_npz_v1"
ARTIFACT_FILENAME = "frames_composition.npz"

LOGGER = get_logger(MODULE_NAME)


# -------------------------
# Progress to state_events.jsonl (PR-5)
# -------------------------
def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort append to:
      <runs_root>/state/<platform>/<video>/<run>/state_events.jsonl
    where:
      rs_path = <rs_base>/<platform>/<video>/<run>
      runs_root = dirname(rs_base)
    """
    try:
        run_rs = Path(rs_path).resolve()
        # rs_path/<platform>/<video>/<run> => rs_base = parents[2]
        rs_base = run_rs.parents[2]
        runs_root = rs_base.parent
        platform_id = str(event.get("platform_id") or "")
        video_id = str(event.get("video_id") or "")
        run_id = str(event.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            return
        p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_progress(
    *,
    rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    done: int,
    total: int,
    stage: str,
) -> None:
    if total <= 0:
        return
    progress = float(done) / float(total)
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": _utc_iso_now(),
            "scope": "progress",
            "processor": "visual",
            "component": MODULE_NAME,
            "status": "running",
            "progress": progress,
            "done": int(done),
            "total": int(total),
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


# -------------------------
# Feature gating
# -------------------------
DEFAULT_FEATURE_SET = "default"


def _parse_csv_set(s: Optional[str]) -> set[str]:
    if not s:
        return set()
    parts = [p.strip() for p in str(s).split(",")]
    return {p for p in parts if p}


def _enabled_groups(feature_set: str, features_csv: Optional[str]) -> set[str]:
    """
    Groups control which feature families are computed and exported.
    """
    explicit = _parse_csv_set(features_csv)
    if explicit:
        return explicit

    fs = str(feature_set or DEFAULT_FEATURE_SET).strip().lower()
    if fs in ("all", "full"):
        return {
            "anchors",
            "balance",
            "symmetry",
            "negative_space",
            "complexity",
            "leading_lines",
            "depth",
            "objects",
            "faces",
            "style",
        }
    if fs in ("ml", "model"):
        return {
            "anchors",
            "balance",
            "symmetry",
            "negative_space",
            "complexity",
            "leading_lines",
            "depth",
            "objects",
            "faces",
        }
    # default
    return {
        "anchors",
        "balance",
        "symmetry",
        "negative_space",
        "complexity",
        "leading_lines",
        "depth",
        "objects",
        "faces",
        "style",
    }


# -------------------------
# Core provider helpers (NPZ)
# -------------------------
def _unbox_obj(v: Any) -> Any:
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
        try:
            return v.item()
        except Exception:
            return v
    return v


def _require_npz_meta(d: Dict[str, Any], provider: str) -> Dict[str, Any]:
    meta = d.get("meta")
    meta = _unbox_obj(meta)
    if isinstance(meta, dict):
        return meta
    raise RuntimeError(f"{MODULE_NAME} | {provider} | missing/invalid meta in NPZ")


@dataclass(frozen=True)
class CoreObjects:
    frame_indices: np.ndarray  # (N,) int32
    boxes: np.ndarray  # (N, K, 4) float32
    valid_mask: np.ndarray  # (N, K) bool


@dataclass(frozen=True)
class CoreFaces:
    frame_indices: np.ndarray  # (N,) int32
    face_present: np.ndarray  # (N,) bool
    face_landmarks: np.ndarray  # (N, 468, 3) float32 (normalized) with NaN for missing
    has_any_face: bool
    provider_status: str
    provider_empty_reason: Optional[str]


@dataclass(frozen=True)
class CoreDepth:
    frame_indices: np.ndarray  # (N,) int32
    depth_mean: np.ndarray  # (N,) float32
    depth_std: np.ndarray  # (N,) float32
    depth_p05: np.ndarray  # (N,) float32
    depth_p95: np.ndarray  # (N,) float32


def _require_aligned_frame_indices(name: str, want: np.ndarray, got: np.ndarray) -> None:
    if want.shape != got.shape or not np.all(want == got):
        raise RuntimeError(f"{MODULE_NAME} | frame_indices mismatch vs {name} (no-fallback)")


def _load_core_objects(d: Dict[str, Any], want_fi: np.ndarray) -> CoreObjects:
    meta = _require_npz_meta(d, "core_object_detections")
    fi = np.asarray(d.get("frame_indices"), dtype=np.int32).reshape(-1)
    _require_aligned_frame_indices("core_object_detections", want_fi, fi)
    boxes = np.asarray(d.get("boxes"), dtype=np.float32)
    valid = np.asarray(d.get("valid_mask")).astype(bool)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise RuntimeError(f"{MODULE_NAME} | core_object_detections invalid boxes shape: {boxes.shape}")
    if valid.ndim != 2 or valid.shape[:2] != boxes.shape[:2]:
        raise RuntimeError(f"{MODULE_NAME} | core_object_detections invalid valid_mask shape: {valid.shape}")
    # Note: core_object_detections can be status=empty; still valid dependency.
    _ = meta
    return CoreObjects(frame_indices=fi, boxes=boxes, valid_mask=valid)


def _load_core_faces(d: Dict[str, Any], want_fi: np.ndarray) -> CoreFaces:
    meta = _require_npz_meta(d, "core_face_landmarks")
    fi = np.asarray(d.get("frame_indices"), dtype=np.int32).reshape(-1)
    _require_aligned_frame_indices("core_face_landmarks", want_fi, fi)
    face_present = np.asarray(d.get("face_present")).astype(bool).reshape(-1)
    face = np.asarray(d.get("face_landmarks"), dtype=np.float32)
    if face_present.shape[0] != want_fi.shape[0]:
        raise RuntimeError(f"{MODULE_NAME} | core_face_landmarks invalid face_present shape: {face_present.shape}")
    if face.ndim != 3 or face.shape[0] != want_fi.shape[0] or face.shape[1] != 468 or face.shape[2] != 3:
        raise RuntimeError(f"{MODULE_NAME} | core_face_landmarks invalid face_landmarks shape: {face.shape}")
    has_any_face = bool(meta.get("has_any_face", False))
    provider_status = str(meta.get("status") or "error")
    provider_empty_reason = meta.get("empty_reason")
    return CoreFaces(
        frame_indices=fi,
        face_present=face_present,
        face_landmarks=face,
        has_any_face=has_any_face,
        provider_status=provider_status,
        provider_empty_reason=str(provider_empty_reason) if provider_empty_reason else None,
    )


def _load_core_depth(d: Dict[str, Any], want_fi: np.ndarray) -> CoreDepth:
    _ = _require_npz_meta(d, "core_depth_midas")
    fi = np.asarray(d.get("frame_indices"), dtype=np.int32).reshape(-1)
    _require_aligned_frame_indices("core_depth_midas", want_fi, fi)
    # Prefer precomputed stats (present in current core_depth_midas)
    dm = np.asarray(d.get("depth_mean"), dtype=np.float32).reshape(-1)
    ds = np.asarray(d.get("depth_std"), dtype=np.float32).reshape(-1)
    p05 = np.asarray(d.get("depth_p05"), dtype=np.float32).reshape(-1)
    p95 = np.asarray(d.get("depth_p95"), dtype=np.float32).reshape(-1)
    for name, arr in (("depth_mean", dm), ("depth_std", ds), ("depth_p05", p05), ("depth_p95", p95)):
        if arr.shape != want_fi.shape:
            raise RuntimeError(f"{MODULE_NAME} | core_depth_midas missing/invalid {name} shape: {arr.shape}")
    return CoreDepth(frame_indices=fi, depth_mean=dm, depth_std=ds, depth_p05=p05, depth_p95=p95)


# -------------------------
# Per-frame feature extraction (RGB contract)
# -------------------------
def _rgb_to_gray01(frame_rgb: np.ndarray) -> np.ndarray:
    # FrameManager contract: RGB uint8
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def _saliency_proxy(frame_rgb: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray01(frame_rgb)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) / 8.0
    contrast = np.abs(cv2.filter2D(gray, -1, kernel))
    sal = 0.5 * gradient_magnitude + 0.5 * contrast
    mn = float(np.min(sal))
    mx = float(np.max(sal))
    if mx - mn < 1e-6:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - mn) / (mx - mn)).astype(np.float32)


def _center_of_mass_offset01(weight_map01: np.ndarray) -> float:
    h, w = weight_map01.shape[:2]
    total = float(np.sum(weight_map01))
    if total <= 1e-6:
        return 0.0
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cx = float(np.sum(xx * weight_map01) / total)
    cy = float(np.sum(yy * weight_map01) / total)
    dx = cx - (w / 2.0)
    dy = cy - (h / 2.0)
    dist = float(np.sqrt(dx * dx + dy * dy))
    max_dist = float(np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2))
    return float(dist / max(max_dist, 1e-6))


def _symmetry_scores(frame_rgb: np.ndarray) -> Tuple[float, float, float]:
    gray = _rgb_to_gray01(frame_rgb)
    h_flip = cv2.flip(gray, 1)
    v_flip = cv2.flip(gray, 0)
    hc = float(np.corrcoef(gray.reshape(-1), h_flip.reshape(-1))[0, 1])
    vc = float(np.corrcoef(gray.reshape(-1), v_flip.reshape(-1))[0, 1])
    hc = float(np.nan_to_num(hc, nan=0.0))
    vc = float(np.nan_to_num(vc, nan=0.0))
    score = float(np.mean([hc, vc]))
    return score, hc, vc


def _complexity(frame_rgb: np.ndarray) -> Tuple[float, float, float, float]:
    gray_u8 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    gray = gray_u8.astype(np.float32) / 255.0
    kernel = np.ones((5, 5), np.float32) / 25.0
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_var = cv2.filter2D((gray - local_mean) ** 2, -1, kernel)
    texture = float(np.mean(local_var))

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hue_std = float(hsv[:, :, 0].std() / 180.0)
    sat_mean = float(hsv[:, :, 1].mean() / 255.0)
    return edge_density, texture, hue_std, sat_mean


def _leading_lines(frame_rgb: np.ndarray) -> Tuple[float, float, float, float, int, int, int, int]:
    gray_u8 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_u8, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_thinned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(edges_thinned, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 3  # dominant=none (3)
    lines = lines.reshape(-1, 4)
    lengths = []
    angles = []
    for x1, y1, x2, y2 in lines:
        lengths.append(float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
        ang = float(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        ang = float((ang + 180.0) % 180.0)
        angles.append(ang)
    total_len = float(np.sum(lengths))
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    h = int(np.sum([(60.0 < a < 120.0) for a in angles]))
    v = int(np.sum([(a < 30.0 or a > 150.0) for a in angles]))
    d = int(len(angles) - h - v)
    # dominant: 0=horizontal,1=vertical,2=diagonal,3=none
    if h >= v and h >= d and h > 0:
        dom = 0
    elif v >= d and v > 0:
        dom = 1
    elif d > 0:
        dom = 2
    else:
        dom = 3
    # strength normalized by frame area
    H, W = gray_u8.shape[:2]
    strength = float(min(1.0, total_len / max(1.0, float(H * W))))
    # crude convergence proxy: 1 - mean distance between line midpoints / diag
    mids = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in lines.tolist()]
    if len(mids) < 2:
        conv = 0.0
    else:
        ds = []
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                dx = mids[i][0] - mids[j][0]
                dy = mids[i][1] - mids[j][1]
                ds.append(float(np.sqrt(dx * dx + dy * dy)))
        diag = float(np.sqrt(float(H * H + W * W)))
        conv = float(1.0 - (float(np.mean(ds)) / max(diag, 1e-6)))
    return strength, total_len, avg_len, conv, len(lines), h, v, dom


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _bbox_area_ratio(b: np.ndarray, *, H: int, W: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in b]
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return float(area / max(1.0, float(H * W)))


def _intersect_area_ratio(b: np.ndarray, *, x_left: float, x_right: float, H: int, W: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in b]
    ix1 = max(x1, x_left)
    ix2 = min(x2, x_right)
    if ix2 <= ix1:
        return 0.0
    area = max(0.0, (ix2 - ix1)) * max(0.0, (y2 - y1))
    return float(area / max(1.0, float(H * W)))


def _face_bbox_from_landmarks(face468: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    # face468: (468,3) normalized in [0..1] with NaN for missing
    xy = face468[:, :2]
    m = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if not np.any(m):
        return None
    xs = xy[m, 0]
    ys = xy[m, 1]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def _anchor_distance(x: float, y: float) -> Tuple[float, int]:
    # returns (min_distance, closest_type_id): 0=thirds,1=golden,2=center
    pts: List[Tuple[float, float, int]] = []
    for tx in (1.0 / 3.0, 2.0 / 3.0):
        for ty in (1.0 / 3.0, 2.0 / 3.0):
            pts.append((tx, ty, 0))
    phi = 1.618033988749895
    for tx in (1.0 / phi, (phi - 1.0)):
        for ty in (1.0 / phi, (phi - 1.0)):
            pts.append((tx, ty, 1))
    pts.append((0.5, 0.5, 2))
    best = (1e9, 2)
    for px, py, tid in pts:
        d = float(np.sqrt((x - px) ** 2 + (y - py) ** 2))
        if d < best[0]:
            best = (d, tid)
    # normalize: max possible distance in unit square from center to corner
    max_d = float(np.sqrt(0.5**2 + 0.5**2))
    return float(best[0] / max_d), int(best[1])


def _style_probs(
    *,
    complexity: float,
    neg_space: float,
    obj_density: float,
    depth_std: float,
    bokeh_proxy: float,
    center_offset: float,
    symmetry: float,
    thirds_alignment: float,
    face_centering: float,
) -> Tuple[float, float, float, float]:
    """
    Heuristic style probabilities for UI explainability.
    Returns probs for: minimalist, cinematic, vlog, product_centered
    """
    eps = 1e-6
    minimalist = 0.45 * (1.0 - complexity) + 0.35 * neg_space + 0.20 * (1.0 - obj_density)
    cinematic = 0.35 * depth_std + 0.25 * (1.0 - center_offset) + 0.20 * (1.0 - symmetry) + 0.20 * bokeh_proxy
    vlog = 0.45 * face_centering + 0.35 * (1.0 - complexity) + 0.20 * obj_density
    product = 0.45 * obj_density + 0.30 * thirds_alignment + 0.25 * bokeh_proxy
    xs = np.clip(np.array([minimalist, cinematic, vlog, product], dtype=np.float32), 0.0, 1.0)
    s = float(xs.sum()) + eps
    xs = xs / s
    return float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3])


def _aggregate_stats(x: np.ndarray, prefix: str) -> Tuple[List[str], List[float]]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return [f"{prefix}__all_non_finite"], [1.0]
    return (
        [
            f"{prefix}__mean",
            f"{prefix}__std",
            f"{prefix}__p10",
            f"{prefix}__p50",
            f"{prefix}__p90",
            f"{prefix}__min",
            f"{prefix}__max",
        ],
        [
            float(np.mean(finite)),
            float(np.std(finite)),
            float(np.percentile(finite, 10)),
            float(np.percentile(finite, 50)),
            float(np.percentile(finite, 90)),
            float(np.min(finite)),
            float(np.max(finite)),
        ],
    )


# -------------------------
# Module
# -------------------------
class FramesCompositionModule(BaseModule):
    MODULE_NAME = MODULE_NAME
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    @property
    def supports_batch(self) -> bool:
        """Поддержка batch processing для frames_composition (CPU модуль)."""
        return True

    def required_dependencies(self) -> List[str]:
        # hard deps (no-fallback): all three must exist, but valid empty allowed for core_face_landmarks
        return ["core_object_detections", "core_face_landmarks", "core_depth_midas"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = config or {}
        enabled = _enabled_groups(str(cfg.get("feature_set") or DEFAULT_FEATURE_SET), cfg.get("features"))
        num_workers = int(cfg.get("num_workers") or max(1, min(8, (os.cpu_count() or 4))))

        # Run identity + time axis (strict, no-fallback)
        meta = getattr(frame_manager, "meta", None) or {}
        platform_id = str(meta.get("platform_id") or "")
        video_id = str(meta.get("video_id") or "")
        run_id = str(meta.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            raise RuntimeError(f"{MODULE_NAME} | frames metadata missing run identity (platform_id/video_id/run_id)")

        union_ts = meta.get("union_timestamps_sec")
        if not isinstance(union_ts, list) or not union_ts:
            raise RuntimeError(f"{MODULE_NAME} | frames metadata missing union_timestamps_sec (no-fallback)")
        ts = np.asarray(union_ts, dtype=np.float32).reshape(-1)

        fi = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        if fi.size == 0:
            raise RuntimeError(f"{MODULE_NAME} | empty frame_indices is not allowed (no-fallback)")
        if np.any(fi < 0) or np.any(fi >= ts.shape[0]):
            raise RuntimeError(f"{MODULE_NAME} | frame_indices out of range for union_timestamps_sec (no-fallback)")
        if not np.all(fi[1:] >= fi[:-1]) or np.unique(fi).size != fi.size:
            raise RuntimeError(f"{MODULE_NAME} | frame_indices must be sorted+unique (no-fallback)")
        times_s = ts[fi].astype(np.float32)

        # Load dependencies (NPZ)
        core_obj_raw = self.load_core_provider("core_object_detections")
        core_face_raw = self.load_core_provider("core_face_landmarks")
        core_depth_raw = self.load_core_provider("core_depth_midas")
        if core_obj_raw is None:
            raise RuntimeError(f"{MODULE_NAME} | missing required dependency: core_object_detections (no-fallback)")
        if core_face_raw is None:
            raise RuntimeError(f"{MODULE_NAME} | missing required dependency: core_face_landmarks (no-fallback)")
        if core_depth_raw is None:
            raise RuntimeError(f"{MODULE_NAME} | missing required dependency: core_depth_midas (no-fallback)")

        core_obj = _load_core_objects(core_obj_raw, fi)
        core_faces = _load_core_faces(core_face_raw, fi)
        core_depth = _load_core_depth(core_depth_raw, fi)

        # Valid empty: no faces in the whole run
        has_any_face = bool(core_faces.has_any_face) or bool(np.any(core_faces.face_present))
        if not has_any_face:
            # Output empty but still write aligned arrays.
            # Features are all NaN (no misleading zeros).
            feature_names = np.asarray(["has_faces"], dtype=object)
            feature_values = np.asarray([0.0], dtype=np.float32)

            frame_feature_names = np.asarray(["face_present"], dtype=object)
            frame_feature_values = np.full((fi.size, 1), np.nan, dtype=np.float32)
            frame_feature_values[:, 0] = 0.0

            return {
                "frame_indices": fi.astype(np.int32),
                "times_s": times_s.astype(np.float32),
                "feature_names": feature_names,
                "feature_values": feature_values,
                "frame_feature_names": frame_feature_names,
                "frame_feature_values": frame_feature_values,
                "__meta_override__": {"status": "empty", "empty_reason": "no_faces_in_video"},
            }

        # Per-frame extraction
        # We keep the vector size stable; groups may set some dims to NaN.
        frame_feat_names: List[str] = []
        cols: List[np.ndarray] = []

        # Always export presence flags for downstream
        face_present_f = core_faces.face_present.astype(np.float32)
        frame_feat_names.append("face_present")
        cols.append(face_present_f)

        # Depth stats (always required; no empty)
        if "depth" in enabled:
            for nm, arr in (
                ("depth_mean", core_depth.depth_mean),
                ("depth_std", core_depth.depth_std),
                ("depth_p05", core_depth.depth_p05),
                ("depth_p95", core_depth.depth_p95),
            ):
                frame_feat_names.append(nm)
                cols.append(arr.astype(np.float32))
        else:
            for nm in ("depth_mean", "depth_std", "depth_p05", "depth_p95"):
                frame_feat_names.append(nm)
                cols.append(np.full((fi.size,), np.nan, dtype=np.float32))

        # Object stats from core_object_detections
        # We keep them cheap: counts + coverage proxies.
        valid = core_obj.valid_mask
        boxes = core_obj.boxes
        obj_count = np.sum(valid, axis=1).astype(np.float32)
        frame_feat_names.append("object_count")
        cols.append(obj_count)

        # Main bbox area ratio and coverage (sum of bbox areas capped at 1)
        def _bbox_stats_for_frame(i: int, H: int, W: int) -> Tuple[float, float, float]:
            m = valid[i]
            if not np.any(m):
                return 0.0, 0.0, 0.0
            bs = boxes[i][m]
            ars = np.array([_bbox_area_ratio(b, H=H, W=W) for b in bs], dtype=np.float32)
            max_ar = float(np.max(ars)) if ars.size else 0.0
            cov = float(min(1.0, float(np.sum(ars))))
            # left-right coverage imbalance proxy
            left = float(np.sum([_intersect_area_ratio(b, x_left=0.0, x_right=W / 2.0, H=H, W=W) for b in bs]))
            right = float(np.sum([_intersect_area_ratio(b, x_left=W / 2.0, x_right=float(W), H=H, W=W) for b in bs]))
            left = float(min(1.0, left))
            right = float(min(1.0, right))
            lr_balance = float(1.0 - abs(left - right))
            return max_ar, cov, lr_balance

        H = int(getattr(frame_manager, "height", 0) or meta.get("height") or 0)
        W = int(getattr(frame_manager, "width", 0) or meta.get("width") or 0)
        if H <= 0 or W <= 0:
            # fallback to actual frame shape on first frame (safe)
            fr0 = frame_manager.get(int(fi[0]))
            H, W = int(fr0.shape[0]), int(fr0.shape[1])

        max_area = np.full((fi.size,), np.nan, dtype=np.float32)
        cover = np.full((fi.size,), np.nan, dtype=np.float32)
        lr_cov_balance = np.full((fi.size,), np.nan, dtype=np.float32)
        for i in range(fi.size):
            a, c, b = _bbox_stats_for_frame(i, H, W)
            max_area[i] = float(a)
            cover[i] = float(c)
            lr_cov_balance[i] = float(b)
        frame_feat_names.append("object_max_area_ratio")
        cols.append(max_area)
        frame_feat_names.append("object_bbox_coverage_ratio")
        cols.append(cover)

        # Anchors and face geometry (requires faces)
        face_center_x = np.full((fi.size,), np.nan, dtype=np.float32)
        face_center_y = np.full((fi.size,), np.nan, dtype=np.float32)
        face_area_ratio = np.full((fi.size,), np.nan, dtype=np.float32)
        anchor_dist = np.full((fi.size,), np.nan, dtype=np.float32)
        anchor_type = np.full((fi.size,), np.nan, dtype=np.float32)
        thirds_alignment = np.full((fi.size,), np.nan, dtype=np.float32)

        if "faces" in enabled or "anchors" in enabled:
            for i in range(fi.size):
                if not bool(core_faces.face_present[i]):
                    continue
                bb = _face_bbox_from_landmarks(core_faces.face_landmarks[i])
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                face_center_x[i] = cx
                face_center_y[i] = cy
                face_area_ratio[i] = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
                if "anchors" in enabled:
                    d, t = _anchor_distance(cx, cy)
                    anchor_dist[i] = float(d)
                    anchor_type[i] = float(t)
                    thirds_alignment[i] = float(max(0.0, 1.0 - d)) if t == 0 else float(max(0.0, 1.0 - d) * 0.7)
        frame_feat_names.extend(["face_center_x", "face_center_y", "face_area_ratio"])
        cols.extend([face_center_x, face_center_y, face_area_ratio])
        frame_feat_names.extend(["anchor_distance", "anchor_type_id", "thirds_alignment"])
        cols.extend([anchor_dist, anchor_type, thirds_alignment])

        # Image-based features (parallel with safe frame reads)
        # We compute these regardless of face_present per frame (they are generic), but gated.
        need_balance = "balance" in enabled
        need_sym = "symmetry" in enabled
        need_neg = "negative_space" in enabled
        need_cmp = "complexity" in enabled
        need_lines = "leading_lines" in enabled
        need_style = "style" in enabled

        # Pre-alloc
        saliency_offset = np.full((fi.size,), np.nan, dtype=np.float32)
        sym_score = np.full((fi.size,), np.nan, dtype=np.float32)
        sym_h = np.full((fi.size,), np.nan, dtype=np.float32)
        sym_v = np.full((fi.size,), np.nan, dtype=np.float32)
        neg_ratio = np.full((fi.size,), np.nan, dtype=np.float32)
        neg_balance_lr = np.full((fi.size,), np.nan, dtype=np.float32)
        edge_density = np.full((fi.size,), np.nan, dtype=np.float32)
        texture = np.full((fi.size,), np.nan, dtype=np.float32)
        hue_std = np.full((fi.size,), np.nan, dtype=np.float32)
        sat_mean = np.full((fi.size,), np.nan, dtype=np.float32)
        line_strength = np.full((fi.size,), np.nan, dtype=np.float32)
        line_count = np.full((fi.size,), np.nan, dtype=np.float32)
        line_conv = np.full((fi.size,), np.nan, dtype=np.float32)
        dom_line = np.full((fi.size,), np.nan, dtype=np.float32)

        style_min = np.full((fi.size,), np.nan, dtype=np.float32)
        style_cin = np.full((fi.size,), np.nan, dtype=np.float32)
        style_vlog = np.full((fi.size,), np.nan, dtype=np.float32)
        style_prod = np.full((fi.size,), np.nan, dtype=np.float32)

        # Progress cadence: >=10 updates per run
        total = int(fi.size)
        step = max(1, int(total // 20))

        # Safe access to FrameManager cache structures (not thread-safe)
        fm_lock = None
        try:
            import threading

            fm_lock = threading.Lock()
        except Exception:
            fm_lock = None

        def _get_frame(idx: int) -> np.ndarray:
            if fm_lock is None:
                fr = frame_manager.get(idx)
            else:
                with fm_lock:
                    fr = frame_manager.get(idx)
            # materialize copy to avoid memmap lifetime issues in worker threads
            return np.asarray(fr).copy()

        def _work(pos: int, idx: int) -> Tuple[int, Dict[str, float]]:
            fr = _get_frame(int(idx))
            out: Dict[str, float] = {}
            if need_balance:
                out["saliency_center_offset"] = float(_center_of_mass_offset01(_saliency_proxy(fr)))
            if need_sym:
                s, hsc, vsc = _symmetry_scores(fr)
                out["symmetry_score"] = float(s)
                out["symmetry_h"] = float(hsc)
                out["symmetry_v"] = float(vsc)
            if need_neg:
                # negative space via bbox coverage proxy (cheap)
                cov = float(cover[pos]) if np.isfinite(cover[pos]) else 0.0
                out["negative_space_ratio"] = float(max(0.0, 1.0 - cov))
                out["neg_space_balance_lr"] = float(lr_cov_balance[pos]) if np.isfinite(lr_cov_balance[pos]) else float("nan")
            if need_cmp:
                ed, tex, hs, sm = _complexity(fr)
                out["edge_density"] = float(ed)
                out["texture_entropy"] = float(tex)
                out["hue_std"] = float(hs)
                out["saturation_mean"] = float(sm)
            if need_lines:
                strength, _tot, _avg, conv, cnt, _h, _v, dom = _leading_lines(fr)
                out["line_strength"] = float(strength)
                out["line_count"] = float(cnt)
                out["convergence_score"] = float(conv)
                out["dominant_line_id"] = float(dom)
            if need_style:
                # style based on already computed/scheduled signals (best-effort)
                cplx = float(out.get("edge_density", np.nan))
                cplx = _clip01(cplx) if np.isfinite(cplx) else 0.5
                neg = float(out.get("negative_space_ratio", np.nan))
                neg = _clip01(neg) if np.isfinite(neg) else 0.5
                od = float(obj_count[pos] / 8.0)
                od = _clip01(od)
                ds = float(core_depth.depth_std[pos])
                ds = _clip01(ds) if np.isfinite(ds) else 0.0
                bokeh_proxy = float(_clip01((core_depth.depth_p95[pos] - core_depth.depth_p05[pos]) if np.isfinite(core_depth.depth_p95[pos]) and np.isfinite(core_depth.depth_p05[pos]) else 0.0))
                center_off = float(out.get("saliency_center_offset", 0.0)) if np.isfinite(out.get("saliency_center_offset", 0.0)) else 0.0
                sym = float(out.get("symmetry_score", 0.0)) if np.isfinite(out.get("symmetry_score", 0.0)) else 0.0
                ta = float(thirds_alignment[pos]) if np.isfinite(thirds_alignment[pos]) else 0.0
                if np.isfinite(face_center_x[pos]):
                    face_centering = float(1.0 - abs(float(face_center_x[pos]) - 0.5) * 2.0)
                else:
                    face_centering = 0.0
                p0, p1, p2, p3 = _style_probs(
                    complexity=cplx,
                    neg_space=neg,
                    obj_density=od,
                    depth_std=ds,
                    bokeh_proxy=bokeh_proxy,
                    center_offset=_clip01(center_off),
                    symmetry=_clip01(sym),
                    thirds_alignment=_clip01(ta),
                    face_centering=_clip01(face_centering),
                )
                out["style_minimalist"] = p0
                out["style_cinematic"] = p1
                out["style_vlog"] = p2
                out["style_product_centered"] = p3
            return pos, out

        # Work scheduling
        import concurrent.futures as cf

        done = 0
        stage = "frames"
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
            inflight: Dict[cf.Future, int] = {}
            max_inflight = max(4, num_workers * 2)
            for pos, idx in enumerate(fi.tolist()):
                fut = ex.submit(_work, pos, int(idx))
                inflight[fut] = pos
                if len(inflight) >= max_inflight:
                    for fut2 in cf.as_completed(list(inflight.keys()), timeout=None):
                        _pos, out = fut2.result()
                        if "saliency_center_offset" in out:
                            saliency_offset[_pos] = float(out["saliency_center_offset"])
                        if "symmetry_score" in out:
                            sym_score[_pos] = float(out["symmetry_score"])
                            sym_h[_pos] = float(out.get("symmetry_h", np.nan))
                            sym_v[_pos] = float(out.get("symmetry_v", np.nan))
                        if "negative_space_ratio" in out:
                            neg_ratio[_pos] = float(out["negative_space_ratio"])
                            neg_balance_lr[_pos] = float(out.get("neg_space_balance_lr", np.nan))
                        if "edge_density" in out:
                            edge_density[_pos] = float(out["edge_density"])
                            texture[_pos] = float(out.get("texture_entropy", np.nan))
                            hue_std[_pos] = float(out.get("hue_std", np.nan))
                            sat_mean[_pos] = float(out.get("saturation_mean", np.nan))
                        if "line_strength" in out:
                            line_strength[_pos] = float(out["line_strength"])
                            line_count[_pos] = float(out.get("line_count", np.nan))
                            line_conv[_pos] = float(out.get("convergence_score", np.nan))
                            dom_line[_pos] = float(out.get("dominant_line_id", np.nan))
                        if "style_minimalist" in out:
                            style_min[_pos] = float(out["style_minimalist"])
                            style_cin[_pos] = float(out["style_cinematic"])
                            style_vlog[_pos] = float(out["style_vlog"])
                            style_prod[_pos] = float(out["style_product_centered"])

                        done += 1
                        inflight.pop(fut2, None)
                        if done % step == 0 or done == total:
                            _emit_progress(
                                rs_path=str(self.rs_path or ""),
                                platform_id=platform_id,
                                video_id=video_id,
                                run_id=run_id,
                                done=done,
                                total=total,
                                stage=stage,
                            )
                        break
            # drain remaining
            for fut in cf.as_completed(list(inflight.keys())):
                _pos, out = fut.result()
                if "saliency_center_offset" in out:
                    saliency_offset[_pos] = float(out["saliency_center_offset"])
                if "symmetry_score" in out:
                    sym_score[_pos] = float(out["symmetry_score"])
                    sym_h[_pos] = float(out.get("symmetry_h", np.nan))
                    sym_v[_pos] = float(out.get("symmetry_v", np.nan))
                if "negative_space_ratio" in out:
                    neg_ratio[_pos] = float(out["negative_space_ratio"])
                    neg_balance_lr[_pos] = float(out.get("neg_space_balance_lr", np.nan))
                if "edge_density" in out:
                    edge_density[_pos] = float(out["edge_density"])
                    texture[_pos] = float(out.get("texture_entropy", np.nan))
                    hue_std[_pos] = float(out.get("hue_std", np.nan))
                    sat_mean[_pos] = float(out.get("saturation_mean", np.nan))
                if "line_strength" in out:
                    line_strength[_pos] = float(out["line_strength"])
                    line_count[_pos] = float(out.get("line_count", np.nan))
                    line_conv[_pos] = float(out.get("convergence_score", np.nan))
                    dom_line[_pos] = float(out.get("dominant_line_id", np.nan))
                if "style_minimalist" in out:
                    style_min[_pos] = float(out["style_minimalist"])
                    style_cin[_pos] = float(out["style_cinematic"])
                    style_vlog[_pos] = float(out["style_vlog"])
                    style_prod[_pos] = float(out["style_product_centered"])
                done += 1
                if done % step == 0 or done == total:
                    _emit_progress(
                        rs_path=str(self.rs_path or ""),
                        platform_id=platform_id,
                        video_id=video_id,
                        run_id=run_id,
                        done=done,
                        total=total,
                        stage=stage,
                    )

        LOGGER.info(
            "%s | processed frames=%d workers=%d elapsed_ms=%d",
            MODULE_NAME,
            total,
            num_workers,
            int((time.time() - t0) * 1000),
        )

        # attach computed arrays into frame columns (respect gating)
        def _maybe(name: str, arr: np.ndarray, enabled_group: str) -> None:
            frame_feat_names.append(name)
            cols.append(arr.astype(np.float32) if enabled_group in enabled else np.full((fi.size,), np.nan, dtype=np.float32))

        _maybe("saliency_center_offset", saliency_offset, "balance")
        _maybe("symmetry_score", sym_score, "symmetry")
        _maybe("symmetry_h", sym_h, "symmetry")
        _maybe("symmetry_v", sym_v, "symmetry")
        _maybe("negative_space_ratio", neg_ratio, "negative_space")
        _maybe("neg_space_balance_lr", neg_balance_lr, "negative_space")
        _maybe("edge_density", edge_density, "complexity")
        _maybe("texture_entropy", texture, "complexity")
        _maybe("hue_std", hue_std, "complexity")
        _maybe("saturation_mean", sat_mean, "complexity")
        _maybe("line_strength", line_strength, "leading_lines")
        _maybe("line_count", line_count, "leading_lines")
        _maybe("convergence_score", line_conv, "leading_lines")
        _maybe("dominant_line_id", dom_line, "leading_lines")

        # style (UI explainability)
        _maybe("style_minimalist", style_min, "style")
        _maybe("style_cinematic", style_cin, "style")
        _maybe("style_vlog", style_vlog, "style")
        _maybe("style_product_centered", style_prod, "style")

        frame_feature_names = np.asarray(frame_feat_names, dtype=object)
        frame_feature_values = np.stack([np.asarray(c, dtype=np.float32).reshape(-1) for c in cols], axis=1).astype(np.float32)

        # Aggregated video-level features (preferred by DatasetBuilder): feature_names + feature_values
        fnames: List[str] = []
        fvals: List[float] = []

        # Basic
        fnames.extend(["has_faces", "frames_n"])
        fvals.extend([1.0, float(fi.size)])

        # Stats for each frame feature (except the raw categorical id fields; keep only numeric useful)
        for j, n in enumerate(frame_feat_names):
            if n in ("anchor_type_id", "dominant_line_id"):
                continue
            names_j, vals_j = _aggregate_stats(frame_feature_values[:, j], prefix=n)
            fnames.extend(names_j)
            fvals.extend(vals_j)

        # Add style distribution ratios (video-level, UI/ML friendly)
        if "style" in enabled:
            # dominant style by mean probability
            means = {
                "minimalist": float(np.nanmean(style_min)),
                "cinematic": float(np.nanmean(style_cin)),
                "vlog": float(np.nanmean(style_vlog)),
                "product_centered": float(np.nanmean(style_prod)),
            }
            dom = max(means.items(), key=lambda x: x[1])[0]
            # store the 4 mean probs as features
            for k, v in means.items():
                fnames.append(f"style_prob__{k}__mean")
                fvals.append(float(v))
            # store dominant style id (stable mapping)
            dom_id = {"minimalist": 0.0, "cinematic": 1.0, "vlog": 2.0, "product_centered": 3.0}.get(dom, -1.0)
            fnames.append("style_dominant_id")
            fvals.append(float(dom_id))

        feature_names = np.asarray(fnames, dtype=object)
        feature_values = np.asarray(fvals, dtype=np.float32).reshape(-1)

        return {
            "frame_indices": fi.astype(np.int32),
            "times_s": times_s.astype(np.float32),
            "feature_names": feature_names,
            "feature_values": feature_values,
            "frame_feature_names": frame_feature_names,
            "frame_feature_values": frame_feature_values,
        }


