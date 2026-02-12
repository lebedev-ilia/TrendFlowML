#!/VisualProcessor/core/model_process/.model_process_venv python3

import argparse
import math
import os
import sys
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread, Event
import threading

import cv2              # type: ignore
import numpy as np      # type: ignore

# Suppress MediaPipe verbose logs (GL context, EGL, inference feedback manager)
# Must be set BEFORE importing mediapipe
os.environ["GLOG_minloglevel"] = "2"  # Suppress INFO, WARNING (keep ERROR, FATAL)
os.environ["GLOG_stderrthreshold"] = "2"  # Only ERROR and FATAL to stderr
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "0")  # Keep GPU enabled, just suppress logs

# Redirect stderr during MediaPipe import to suppress initialization logs
import contextlib
from io import StringIO

_stderr_suppress = StringIO()
with contextlib.redirect_stderr(_stderr_suppress):
    import mediapipe as mp

# Suppress MediaPipe/absl logging after import
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# Suppress warnings from protobuf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype") 

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used

VERSION = "2.0"
NAME = "core_face_landmarks"
SCHEMA_VERSION = "core_face_landmarks_npz_v1"
ARTIFACT_FILENAME = "landmarks.npz"
LOGGER = get_logger(NAME)

# Progress context for state_events (used inside processing functions)
_PROGRESS_CONTEXT: Dict[str, Any] = {}


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (backend tails this file).
    """
    try:
        run_rs = Path(rs_path).resolve()
        rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
        runs_root = rs_base.parent
        platform_id = str(event.get("platform_id") or "")
        video_id = str(event.get("video_id") or "")
        run_id = str(event.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            return
        p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        event["platform_id"] = platform_id
        event["video_id"] = video_id
        event["run_id"] = run_id
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
            "status": "running",
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


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
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
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


def _set_progress_context(rs_path: str, platform_id: str, video_id: str, run_id: str) -> None:
    _PROGRESS_CONTEXT.clear()
    _PROGRESS_CONTEXT.update(
        {
            "rs_path": rs_path,
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        }
    )


def _clear_progress_context() -> None:
    _PROGRESS_CONTEXT.clear()


def _emit_progress_frames(done: int, total: int, stage: str = "process_frames") -> None:
    if total <= 0:
        return
    ctx = _PROGRESS_CONTEXT
    if not ctx:
        return
    _emit_progress(
        rs_path=str(ctx.get("rs_path") or ""),
        platform_id=str(ctx.get("platform_id") or ""),
        video_id=str(ctx.get("video_id") or ""),
        run_id=str(ctx.get("run_id") or ""),
        done=done,
        total=total,
        stage=stage,
    )

POSE_LANDMARKS = 33
POSE_DIMS = 4          # x, y, z, visibility

HAND_LANDMARKS = 21
HAND_DIMS = 3          # x, y, z

FACE_LANDMARKS = 468
FACE_DIMS = 3          # x, y, z

# Profiling
class Profiler:
    """Simple profiler for timing different stages."""
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def time(self, stage: str):
        """Context manager for timing a stage."""
        return _TimingContext(self, stage)
    
    def add(self, stage: str, duration: float):
        """Add a timing measurement."""
        with self._lock:
            self.timings[stage].append(duration)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics."""
        summary = {}
        with self._lock:
            for stage, times in self.timings.items():
                if times:
                    summary[stage] = {
                        "total": sum(times),
                        "mean": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "count": len(times),
                    }
        return summary


class _TimingContext:
    def __init__(self, profiler: Profiler, stage: str):
        self.profiler = profiler
        self.stage = stage
        self.start = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        self.profiler.add(self.stage, duration)


# Temporal filtering
class OneEuroFilter:
    """One Euro filter for temporal smoothing of landmarks."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
    
    def __call__(self, x: float, dt: float = 1.0 / 30.0) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        
        # Estimate derivative
        dx = (x - self.x_prev) / dt if dt > 0 else 0.0
        
        # Smooth derivative
        edx = self._smooth(self.dx_prev, dx, self.d_cutoff, dt)
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Smooth value
        x_filtered = self._smooth(self.x_prev, x, cutoff, dt)
        
        self.x_prev = x_filtered
        self.dx_prev = edx
        
        return x_filtered
    
    @staticmethod
    def _smooth(prev: float, curr: float, cutoff: float, dt: float) -> float:
        """Exponential smoothing."""
        if dt <= 0:
            return curr
        tau = 1.0 / (2 * math.pi * cutoff)
        alpha = dt / (tau + dt)
        return alpha * curr + (1 - alpha) * prev


def apply_temporal_filter(
    data: np.ndarray,
    fps: float = 30.0,
    use_one_euro: bool = True,
    min_cutoff: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Apply temporal filtering to landmark data.
    
    Args:
        data: Shape (n_frames, ..., n_dims) with NaN for missing values
        fps: Frames per second for temporal filtering
        use_one_euro: Use OneEuro filter, otherwise linear interpolation only
        min_cutoff: Minimum cutoff frequency for OneEuro
        beta: Beta parameter for OneEuro (higher = more responsive)
    
    Returns:
        Filtered data with same shape
    """
    if data is None or data.size == 0:
        return data
    
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    filtered = data.copy()
    n_frames = data.shape[0]
    
    # Flatten for processing (except frame dimension)
    original_shape = data.shape
    flat_shape = (n_frames, -1)
    flat_data = data.reshape(flat_shape)
    flat_filtered = filtered.reshape(flat_shape)
    
    n_features = flat_data.shape[1]
    
    for feat_idx in range(n_features):
        # Extract time series for this feature
        series = flat_data[:, feat_idx]
        
        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(series)
        if not np.any(valid_mask):
            continue
        
        valid_indices = np.where(valid_mask)[0]
        
        # Linear interpolation for missing values
        if len(valid_indices) < n_frames:
            # Interpolate missing values
            interp_series = np.interp(
                np.arange(n_frames),
                valid_indices,
                series[valid_indices]
            )
        else:
            interp_series = series.copy()
        
        # Apply OneEuro filter if enabled
        if use_one_euro and len(valid_indices) > 1:
            filter_obj = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            for i in range(n_frames):
                interp_series[i] = filter_obj(interp_series[i], dt)
        
        flat_filtered[:, feat_idx] = interp_series
    
    return flat_filtered.reshape(original_shape)

def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME} | missing required artifact: {path}")
    d = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in d.files:
        v = d[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _parse_class_names(arr: Any) -> Dict[str, int]:
    """
    class_names in core_object_detections is an array of strings like 'id:name'.
    Returns mapping name -> id.
    """
    out: Dict[str, int] = {}
    if arr is None:
        return out
    try:
        xs = np.asarray(arr).reshape(-1).tolist()
    except Exception:
        return out
    for s in xs:
        try:
            ss = str(s)
            if ":" not in ss:
                continue
            a, b = ss.split(":", 1)
            out[str(b).strip()] = int(a)
        except Exception:
            continue
    return out


def _pick_detection_stride(n_frames: int, *, target: int, min_frames: int, max_frames: int) -> int:
    """
    Stage-1 sampling: choose stride on the PRIMARY frame_indices list (union-domain).
    We want ~target frames to run lightweight face detection on (bounded by min/max).
    """
    n = max(0, int(n_frames))
    if n <= 0:
        return 1
    t = max(1, int(target))
    mn = max(1, int(min_frames))
    mx = max(mn, int(max_frames))
    desired = int(max(mn, min(mx, t)))
    # stride >= 1
    return max(1, int(math.ceil(float(n) / float(desired))))


def _pick_window_radius(stride: int, *, min_radius: int = 1, max_radius: int = 5) -> int:
    """
    Stage-2 policy: if stage-1 detection stride is large, expand window a bit.
    This keeps "face neighbourhood" coverage without running FaceMesh everywhere.
    """
    s = max(1, int(stride))
    r = max(min_radius, int(math.ceil(s / 5.0)))
    return int(min(max_radius, r))


def init_face_detector(cfg):
    # MediaPipe lightweight face detector (faster than FaceMesh).
    # Uses an internal tracker-like temporal consistency when frames are close enough.
    return mp.solutions.face_detection.FaceDetection(
        model_selection=int(getattr(cfg, "face_detection_model_selection", 0)),
        min_detection_confidence=float(getattr(cfg, "face_detection_min_confidence", 0.5)),
    )


def _stage1_detect_faces(
    frame_manager: FrameManager,
    frame_indices_primary: List[int],
    cfg,
) -> Tuple[List[int], List[int], int]:
    """
    Stage-1: lightweight face detection on a sparse subset of PRIMARY indices.
    Returns:
      - det_primary_pos: positions (0..N-1) in primary list where we ran detection
      - det_face_pos: subset of det_primary_pos where at least one face was detected
      - stride used
    """
    n = len(frame_indices_primary)
    stride = _pick_detection_stride(
        n,
        target=int(getattr(cfg, "face_detection_target_frames", 50)),
        min_frames=int(getattr(cfg, "face_detection_min_frames", 20)),
        max_frames=int(getattr(cfg, "face_detection_max_frames", 200)),
    )
    det_primary_pos = list(range(0, n, stride))
    det_face_pos: List[int] = []

    fd = init_face_detector(cfg)
    try:
        for j, pos in enumerate(det_primary_pos):
            idx = frame_indices_primary[pos]
            fr_rgb = frame_manager.get(idx)
            # MediaPipe expects RGB, FrameManager.get() returns RGB - pass directly
            res = fd.process(fr_rgb)
            if getattr(res, "detections", None):
                det_face_pos.append(int(pos))
            if j % 30 == 0:
                LOGGER.info(f"{NAME} | stage1 | processed {j + 1}/{len(det_primary_pos)} det frames")
    finally:
        try:
            fd.close()
        except Exception:
            pass

    return det_primary_pos, det_face_pos, stride


def _stage2_select_face_mesh_positions(
    n_frames: int,
    det_face_positions: List[int],
    stride: int,
    cfg,
) -> List[int]:
    """
    Stage-2: choose which PRIMARY frame positions will run FaceMesh.
    Policy: for each detected face position, include itself + a small window around it.
    """
    n = max(0, int(n_frames))
    if n <= 0:
        return []
    if not det_face_positions:
        return []
    # window radius can be explicitly overridden, otherwise derived from stride.
    if getattr(cfg, "face_mesh_window_radius", None) is not None:
        r = int(getattr(cfg, "face_mesh_window_radius"))
        r = max(0, r)
    else:
        r = _pick_window_radius(stride)

    sel = set()
    for p in det_face_positions:
        pp = int(p)
        for q in range(pp - r, pp + r + 1):
            if 0 <= q < n:
                sel.add(int(q))
    return sorted(sel)


def init_pose(cfg):
    return mp.solutions.pose.Pose(
        static_image_mode=cfg.pose_static_image_mode,
        model_complexity=cfg.pose_model_complexity,
        enable_segmentation=cfg.pose_enable_segmentation,
        min_detection_confidence=cfg.pose_min_detection_confidence,
        min_tracking_confidence=cfg.pose_min_tracking_confidence,
    )


def init_hands(cfg):
    return mp.solutions.hands.Hands(
        static_image_mode=cfg.hands_static_image_mode,
        max_num_hands=cfg.hands_max_num_hands,
        model_complexity=cfg.hands_model_complexity,
        min_detection_confidence=cfg.hands_min_detection_confidence,
        min_tracking_confidence=cfg.hands_min_tracking_confidence,
    )


def init_face(cfg):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=cfg.face_mesh_static_image_mode,
        refine_landmarks=cfg.face_mesh_refine_landmarks,
        max_num_faces=cfg.face_mesh_max_num_faces,
        min_detection_confidence=cfg.face_mesh_min_detection_confidence,
        min_tracking_confidence=cfg.face_mesh_min_tracking_confidence,
    )


def _process_single_frame(
    frame_rgb: np.ndarray,
    i: int,
    face_mesh_positions: Optional[set[int]],
    cfg: Any,
    pose_data: Optional[np.ndarray],
    hands_data: Optional[np.ndarray],
    face_data: Optional[np.ndarray],
    pose_present: Optional[np.ndarray],
    hands_present: Optional[np.ndarray],
    face_present: Optional[np.ndarray],
    mp_pose,
    mp_hands,
    mp_face,
    profiler: Optional[Profiler] = None,
) -> Dict[str, Any]:
    """
    Process a single frame with MediaPipe models.
    Returns dict with results for this frame.
    """
    result = {
        "i": i,
        "pose": None,
        "hands": None,
        "face": None,
        "pose_present": False,
        "hands_present": [False] * (cfg.hands_max_num_hands if cfg.use_hands else 0),
        "face_present": [False] * (cfg.face_mesh_max_num_faces if cfg.use_face_mesh else 0),
    }

    # MediaPipe expects RGB, FrameManager.get() returns RGB - pass directly

    if mp_pose:
        with profiler.time("inference.pose") if profiler else _null_context():
            res = mp_pose.process(frame_rgb)
        if res.pose_landmarks:
            result["pose_present"] = True
            landmarks = []
            for lm in res.pose_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
            result["pose"] = landmarks

    if mp_hands:
        with profiler.time("inference.hands") if profiler else _null_context():
            res = mp_hands.process(frame_rgb)
        if res.multi_hand_landmarks:
            hands_list = []
            for h, hand in enumerate(res.multi_hand_landmarks):
                if h >= cfg.hands_max_num_hands:
                    break
                result["hands_present"][h] = True
                landmarks = []
                for lm in hand.landmark:
                    landmarks.append((lm.x, lm.y, lm.z))
                hands_list.append(landmarks)
            result["hands"] = hands_list

    if mp_face:
        # Skip expensive FaceMesh if stage2 did not select this frame.
        if face_mesh_positions is None or i in face_mesh_positions:
            with profiler.time("inference.face") if profiler else _null_context():
                res = mp_face.process(frame_rgb)
            if res.multi_face_landmarks:
                max_landmarks = face_data.shape[2] if face_data is not None else FACE_LANDMARKS
                faces_list = []
                for f, face in enumerate(res.multi_face_landmarks):
                    if f >= cfg.face_mesh_max_num_faces:
                        break
                    result["face_present"][f] = True
                    landmarks = []
                    for j, lm in enumerate(face.landmark):
                        if j >= max_landmarks:
                            break
                        landmarks.append((lm.x, lm.y, lm.z))
                    faces_list.append(landmarks)
                result["face"] = faces_list

    return result


class _null_context:
    """Null context manager for optional profiling."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def process_video(
    frame_manager: FrameManager,
    frame_indices: List[int],
    cfg,
    *,
    face_mesh_positions_override: Optional[set[int]] = None,
    profiler: Optional[Profiler] = None,
    enable_async: bool = True,
    enable_parallel: bool = False,
    num_workers: int = 2,
    enable_temporal_filter: bool = True,
    fps: float = 30.0,
):

    n_frames = len(frame_indices)

    pose_data = (
        np.full((n_frames, POSE_LANDMARKS, POSE_DIMS), np.nan, dtype=np.float32)
        if cfg.use_pose else None
    )

    hands_data = (
        np.full(
            (n_frames, cfg.hands_max_num_hands, HAND_LANDMARKS, HAND_DIMS),
            np.nan,
            dtype=np.float32,
        )
        if cfg.use_hands else None
    )

    face_data = (
        np.full(
            (n_frames, cfg.face_mesh_max_num_faces, FACE_LANDMARKS, FACE_DIMS),
            np.nan,
            dtype=np.float32,
        )
        if cfg.use_face_mesh else None
    )

    # Validity masks: explicit "worked but empty" signals for downstream modules.
    pose_present = np.zeros((n_frames,), dtype=bool) if cfg.use_pose else None
    hands_present = (
        np.zeros((n_frames, cfg.hands_max_num_hands), dtype=bool) if cfg.use_hands else None
    )
    face_present = (
        np.zeros((n_frames, cfg.face_mesh_max_num_faces), dtype=bool) if cfg.use_face_mesh else None
    )

    # Stage-1/2 acceleration for face_mesh:
    # - stage1: sparse face detection on PRIMARY indices
    # - stage2: run FaceMesh only near detected faces, but OUTPUT stays aligned to PRIMARY indices.
    face_mesh_positions: Optional[set[int]] = None
    det_stride = 1
    if cfg.use_face_mesh:
        if face_mesh_positions_override is not None:
            face_mesh_positions = set(face_mesh_positions_override)
            LOGGER.info(
                f"{NAME} | person-mask | primary_frames={n_frames} face_mesh_frames={len(face_mesh_positions)}"
            )
        else:
            det_pos, det_face_pos, det_stride = _stage1_detect_faces(frame_manager, frame_indices, cfg)
            sel_pos = _stage2_select_face_mesh_positions(n_frames, det_face_pos, det_stride, cfg)
            face_mesh_positions = set(sel_pos)
            LOGGER.info(
                f"{NAME} | stage1/2 | primary_frames={n_frames} det_frames={len(det_pos)} "
                f"faces_found={len(det_face_pos)} face_mesh_frames={len(sel_pos)} stride={det_stride}"
            )

    # Initialize MediaPipe models
    mp_pose = init_pose(cfg) if cfg.use_pose else None
    mp_hands = init_hands(cfg) if cfg.use_hands else None
    # If override is provided and empty, skip FaceMesh init entirely (still produce aligned NaNs).
    mp_face = (
        init_face(cfg)
        if (cfg.use_face_mesh and not (face_mesh_positions is not None and len(face_mesh_positions) == 0))
        else None
    )

    LOGGER.info(f"{NAME} | Models initialized")

    if profiler is None:
        profiler = Profiler()
    
    # Initialize profiler
    total_start = time.perf_counter()
    
    try:
        if enable_async and not enable_parallel:
            # Async producer/consumer pattern with prefetch
            _process_video_async(
                frame_manager, frame_indices, cfg,
                face_mesh_positions,
                pose_data, hands_data, face_data,
                pose_present, hands_present, face_present,
                profiler, n_frames,
                mp_pose, mp_hands, mp_face,
            )
        elif enable_parallel:
            # Parallel processing with worker pool
            _process_video_parallel(
                frame_manager, frame_indices, cfg,
                face_mesh_positions,
                pose_data, hands_data, face_data,
                pose_present, hands_present, face_present,
                profiler, n_frames, num_workers,
                mp_pose, mp_hands, mp_face,
            )
        else:
            # Sequential processing (baseline)
            _process_video_sequential(
                frame_manager, frame_indices, cfg,
                face_mesh_positions,
                pose_data, hands_data, face_data,
                pose_present, hands_present, face_present,
                profiler, n_frames,
                mp_pose, mp_hands, mp_face,
            )
    finally:
        if mp_pose:
            mp_pose.close()
        if mp_hands:
            mp_hands.close()
        if mp_face:
            mp_face.close()
    
    total_duration = time.perf_counter() - total_start
    profiler.add("total", total_duration)
    
    # Apply temporal filtering if enabled
    if enable_temporal_filter:
        with profiler.time("postproc.temporal_filter"):
            if pose_data is not None:
                pose_data = apply_temporal_filter(pose_data, fps=fps)
            if hands_data is not None:
                hands_data = apply_temporal_filter(hands_data, fps=fps)
            if face_data is not None:
                face_data = apply_temporal_filter(face_data, fps=fps)
    
    # Log profiling summary
    summary = profiler.summary()
    LOGGER.info(f"{NAME} | Profiling summary:")
    for stage, stats in summary.items():
        LOGGER.info(f"  {stage}: total={stats['total']:.3f}s, mean={stats['mean']:.3f}s, count={stats['count']}")
    
    return pose_data, hands_data, face_data, pose_present, hands_present, face_present


def _process_video_sequential(
    frame_manager: FrameManager,
    frame_indices: List[int],
    cfg: Any,
    face_mesh_positions: Optional[set[int]],
    pose_data: Optional[np.ndarray],
    hands_data: Optional[np.ndarray],
    face_data: Optional[np.ndarray],
    pose_present: Optional[np.ndarray],
    hands_present: Optional[np.ndarray],
    face_present: Optional[np.ndarray],
    profiler: Profiler,
    n_frames: int,
    mp_pose,
    mp_hands,
    mp_face,
):
    """Sequential processing (baseline)."""
    total = len(frame_indices)
    for i, frame_idx in enumerate(frame_indices):
        with profiler.time("io.frame_load"):
            frame_rgb = frame_manager.get(frame_idx)
        
        result = _process_single_frame(
            frame_rgb, i, face_mesh_positions, cfg,
            pose_data, hands_data, face_data,
            pose_present, hands_present, face_present,
            mp_pose, mp_hands, mp_face,
            profiler,
        )
        
        # Store results
        with profiler.time("postproc.store"):
            if result["pose"] and pose_data is not None:
                pose_present[i] = result["pose_present"]
                for j, lm in enumerate(result["pose"]):
                    pose_data[i, j] = lm
            
            if result["hands"] and hands_data is not None:
                for h, hand_landmarks in enumerate(result["hands"]):
                    if h >= cfg.hands_max_num_hands:
                        break
                    hands_present[i, h] = result["hands_present"][h]
                    for j, lm in enumerate(hand_landmarks):
                        hands_data[i, h, j] = lm
            
            if result["face"] and face_data is not None:
                for f, face_landmarks in enumerate(result["face"]):
                    if f >= cfg.face_mesh_max_num_faces:
                        break
                    face_present[i, f] = result["face_present"][f]
                    for j, lm in enumerate(face_landmarks):
                        face_data[i, f, j] = lm
        
        if i % 30 == 0:
            LOGGER.info(f"{NAME} | processed {i + 1}/{n_frames} frames")

        # Granular progress (sequential path)
        done = i + 1
        if done % max(1, total // 15) == 0 or done == total:
            _emit_progress_frames(done=done, total=total, stage="process_frames")


def _process_video_async(
    frame_manager: FrameManager,
    frame_indices: List[int],
    cfg: Any,
    face_mesh_positions: Optional[set[int]],
    pose_data: Optional[np.ndarray],
    hands_data: Optional[np.ndarray],
    face_data: Optional[np.ndarray],
    pose_present: Optional[np.ndarray],
    hands_present: Optional[np.ndarray],
    face_present: Optional[np.ndarray],
    profiler: Profiler,
    n_frames: int,
    mp_pose,
    mp_hands,
    mp_face,
):
    """Async producer/consumer with prefetch."""
    frame_queue: Queue = Queue(maxsize=8)  # Prefetch buffer (optimized: increased from 4 to 8 for better I/O overlap)
    stop_event = Event()
    
    def producer():
        """Producer: load frames ahead."""
        try:
            for i, frame_idx in enumerate(frame_indices):
                with profiler.time("io.frame_load"):
                    frame_rgb = frame_manager.get(frame_idx)
                frame_queue.put((i, frame_idx, frame_rgb))
            frame_queue.put(None)  # Sentinel
        except Exception as e:
            LOGGER.error(f"{NAME} | producer error: {e}")
            frame_queue.put(None)
    
    def consumer():
        """Consumer: process frames."""
        processed = 0
        while True:
            item = frame_queue.get()
            if item is None:
                break
            
            i, frame_idx, frame_rgb = item
            
            result = _process_single_frame(
                frame_rgb, i, face_mesh_positions, cfg,
                pose_data, hands_data, face_data,
                pose_present, hands_present, face_present,
                mp_pose, mp_hands, mp_face,
                profiler,
            )
            
            # Store results
            with profiler.time("postproc.store"):
                if result["pose"] and pose_data is not None:
                    pose_present[i] = result["pose_present"]
                    for j, lm in enumerate(result["pose"]):
                        pose_data[i, j] = lm
                
                if result["hands"] and hands_data is not None:
                    for h, hand_landmarks in enumerate(result["hands"]):
                        if h >= cfg.hands_max_num_hands:
                            break
                        hands_present[i, h] = result["hands_present"][h]
                        for j, lm in enumerate(hand_landmarks):
                            hands_data[i, h, j] = lm
                
                if result["face"] and face_data is not None:
                    for f, face_landmarks in enumerate(result["face"]):
                        if f >= cfg.face_mesh_max_num_faces:
                            break
                        face_present[i, f] = result["face_present"][f]
                        for j, lm in enumerate(face_landmarks):
                            face_data[i, f, j] = lm
            
            processed += 1
            if processed % 30 == 0:
                LOGGER.info(f"{NAME} | processed {processed}/{n_frames} frames")

            # Granular progress (async path)
            if processed % max(1, n_frames // 15) == 0 or processed == n_frames:
                _emit_progress_frames(done=processed, total=n_frames, stage="process_frames")
            
            frame_queue.task_done()
    
    # Start producer and consumer threads
    prod_thread = Thread(target=producer, name="frame_producer", daemon=True)
    cons_thread = Thread(target=consumer, name="frame_consumer", daemon=True)
    
    prod_thread.start()
    cons_thread.start()
    
    prod_thread.join()
    cons_thread.join()


def _process_video_parallel(
    frame_manager: FrameManager,
    frame_indices: List[int],
    cfg: Any,
    face_mesh_positions: Optional[set[int]],
    pose_data: Optional[np.ndarray],
    hands_data: Optional[np.ndarray],
    face_data: Optional[np.ndarray],
    pose_present: Optional[np.ndarray],
    hands_present: Optional[np.ndarray],
    face_present: Optional[np.ndarray],
    profiler: Profiler,
    n_frames: int,
    num_workers: int,
    mp_pose,
    mp_hands,
    mp_face,
):
    """
    Parallel processing with worker pool.
    
    WARNING: MediaPipe models are NOT thread-safe when sharing instances across threads.
    This implementation creates separate model instances per worker to avoid SIGSEGV.
    However, parallel processing with multiple models (pose + hands + face) can still be unstable.
    Recommended: use enable_async=True with enable_parallel=False for better stability.
    """
    # Process in chunks to maintain order
    # Optimized: reduced multiplier from 4 to 2 for better load balancing
    chunk_size = max(1, n_frames // (num_workers * 2))
    chunks = []
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunks.append((start, end))
    
    def process_chunk(start: int, end: int, worker_id: int):
        """
        Process a chunk of frames with worker-specific model instances.
        Each worker gets its own MediaPipe model instances to avoid thread-safety issues.
        """
        # Create separate model instances for this worker
        worker_mp_pose = init_pose(cfg) if cfg.use_pose else None
        worker_mp_hands = init_hands(cfg) if cfg.use_hands else None
        worker_mp_face = (
            init_face(cfg)
            if (cfg.use_face_mesh and not (face_mesh_positions is not None and len(face_mesh_positions) == 0))
            else None
        )
        
        try:
            chunk_results = []
            for i in range(start, end):
                frame_idx = frame_indices[i]
                with profiler.time("io.frame_load"):
                    frame_rgb = frame_manager.get(frame_idx)
                
                result = _process_single_frame(
                    frame_rgb, i, face_mesh_positions, cfg,
                    pose_data, hands_data, face_data,
                    pose_present, hands_present, face_present,
                    worker_mp_pose, worker_mp_hands, worker_mp_face,
                    profiler,
                )
                chunk_results.append((i, result))
            return chunk_results
        finally:
            # Cleanup worker-specific models
            if worker_mp_pose:
                try:
                    worker_mp_pose.close()
                except Exception:
                    pass
            if worker_mp_hands:
                try:
                    worker_mp_hands.close()
                except Exception:
                    pass
            if worker_mp_face:
                try:
                    worker_mp_face.close()
                except Exception:
                    pass
    
    # Process chunks in parallel with worker-specific models
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_chunk, start, end, worker_id)
            for worker_id, (start, end) in enumerate(chunks)
        ]
        
        all_results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
            except Exception as e:
                LOGGER.error(f"{NAME} | parallel | chunk processing failed: {e}")
                # Continue with other chunks
        
        # Sort by frame index to maintain order
        all_results.sort(key=lambda x: x[0])
        
        # Store results
        with profiler.time("postproc.store"):
            for i, result in all_results:
                if result["pose"] and pose_data is not None:
                    pose_present[i] = result["pose_present"]
                    for j, lm in enumerate(result["pose"]):
                        pose_data[i, j] = lm
                
                if result["hands"] and hands_data is not None:
                    for h, hand_landmarks in enumerate(result["hands"]):
                        if h >= cfg.hands_max_num_hands:
                            break
                        hands_present[i, h] = result["hands_present"][h]
                        for j, lm in enumerate(hand_landmarks):
                            hands_data[i, h, j] = lm
                
                if result["face"] and face_data is not None:
                    for f, face_landmarks in enumerate(result["face"]):
                        if f >= cfg.face_mesh_max_num_faces:
                            break
                        face_present[i, f] = result["face_present"][f]
                        for j, lm in enumerate(face_landmarks):
                            face_data[i, f, j] = lm
        
        LOGGER.info(f"{NAME} | processed {n_frames}/{n_frames} frames (parallel)")

        # Granular progress (parallel path)
        _emit_progress_frames(done=n_frames, total=n_frames, stage="process_frames")


def main():
    parser = argparse.ArgumentParser(description="Production MediaPipe landmarks extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--use-pose", action="store_true")
    parser.add_argument("--use-hands", action="store_true")
    parser.add_argument("--use-face-mesh", action="store_true")
    # Baseline policy: use core_object_detections person mask to decide which frames are eligible for face analysis.
    parser.add_argument("--use-person-mask", action="store_true", help="Run face mesh only on frames where core_object_detections detected class 'person' (baseline policy).")
    parser.add_argument("--person-window-radius", type=int, default=0, help="Optional +/- radius (in primary positions) around person frames. Default 0 (strict).")
    # Stage-1 lightweight face detection (used to decide where to run FaceMesh)
    parser.add_argument("--face-detection-target-frames", type=int, default=50)
    parser.add_argument("--face-detection-min-frames", type=int, default=20)
    parser.add_argument("--face-detection-max-frames", type=int, default=200)
    parser.add_argument("--face-detection-model-selection", type=int, default=0, choices=[0, 1])
    parser.add_argument("--face-detection-min-confidence", type=float, default=0.5)
    # Stage-2 FaceMesh window (primary index positions)
    parser.add_argument(
        "--face-mesh-window-radius",
        type=int,
        default=None,
        help="If set, run FaceMesh on detected frames plus +/- radius neighbours (in primary frame_indices positions). "
             "If not set, derived from detection stride.",
    )
    parser.add_argument("--pose-static-image-mode", action="store_true")
    parser.add_argument("--pose-model-complexity", type=int, default=2)
    parser.add_argument("--pose-enable-segmentation", action="store_true")
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--pose-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--hands-static-image-mode", action="store_true")
    parser.add_argument("--hands-max-num-hands", type=int, default=2)
    parser.add_argument("--hands-model-complexity", type=int, default=1)
    parser.add_argument("--hands-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--hands-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--face-mesh-static-image-mode", action="store_true")
    parser.add_argument("--face-mesh-max-num-faces", type=int, default=1)
    parser.add_argument("--face-mesh-refine-landmarks", action="store_true")
    parser.add_argument("--face-mesh-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--face-mesh-min-tracking-confidence", type=float, default=0.5)
    # Optimization options
    parser.add_argument("--enable-async", action="store_true", default=True, help="Enable async producer/consumer pattern (default: True)")
    parser.add_argument("--disable-async", dest="enable_async", action="store_false", help="Disable async processing")
    parser.add_argument("--enable-parallel", action="store_true", default=False, help="Enable parallel processing with worker pool")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: auto-detect based on CPU count)")
    parser.add_argument("--enable-temporal-filter", action="store_true", default=True, help="Enable temporal filtering (OneEuro filter) (default: True)")
    parser.add_argument("--disable-temporal-filter", dest="enable_temporal_filter", action="store_false", help="Disable temporal filtering")
    parser.add_argument("--temporal-filter-min-cutoff", type=float, default=1.0, help="OneEuro filter min_cutoff (default: 1.0)")
    parser.add_argument("--temporal-filter-beta", type=float, default=0.0, help="OneEuro filter beta (default: 0.0)")
    parser.add_argument("--enable-profiling", action="store_true", default=True, help="Enable performance profiling (default: True)")
    args = parser.parse_args()
    
    # Auto-detect optimal number of workers if not specified
    if args.num_workers is None or args.num_workers <= 0:
        cpu_count = os.cpu_count() or 4
        # Use 50-75% of CPU cores for workers (leave some for system)
        args.num_workers = max(2, min(8, int(cpu_count * 0.75)))
        LOGGER.info(f"{NAME} | auto-detected num_workers={args.num_workers} (CPU count={cpu_count})")

    # Baseline policy: core_face_landmarks must produce face_mesh outputs (shot_quality depends on it).
    if not bool(args.use_face_mesh):
        raise RuntimeError(f"{NAME} | baseline requires --use-face-mesh (no-fallback)")
    # Baseline policy: person-mask is required (face_landmarks depends on core_object_detections).
    if not bool(args.use_person_mask):
        raise RuntimeError(f"{NAME} | baseline requires --use-person-mask (no-fallback)")

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta["total_frames"])
    platform_id = str(meta.get("platform_id") or "")
    video_id = str(meta.get("video_id") or "")
    run_id = str(meta.get("run_id") or "")

    # Stage timings (seconds; later converted to ms and stored in meta)
    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # Emit start stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="start",
    )
    
    # Get union_timestamps_sec for times_s (required by baseline contract)
    union_timestamps_sec = meta.get("union_timestamps_sec")
    if union_timestamps_sec is None:
        raise RuntimeError(f"{NAME} | metadata.json missing 'union_timestamps_sec' (strict time axis, no-fallback)")
    union_ts = np.asarray(union_timestamps_sec, dtype=np.float32).reshape(-1)

    # Strict sampling contract: Segmenter must provide per-provider indices in metadata[NAME].frame_indices.
    block = meta.get(NAME)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | metadata missing '{NAME}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{NAME} | metadata '{NAME}.frame_indices' is empty/invalid.")
    frame_indices = [int(x) for x in frame_indices_raw]
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")
    
    # Extract times_s from union_timestamps_sec (required by baseline contract)
    fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if np.any(fi_np < 0) or np.any(fi_np >= int(union_ts.shape[0])):
        raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
    times_s = union_ts[fi_np].astype(np.float32)

    # Load dependency: core_object_detections (no-fallback) to get person mask.
    det_path = os.path.join(args.rs_path, "core_object_detections", "detections.npz")
    det = _load_npz(det_path)
    fi_det = np.asarray(det.get("frame_indices"), dtype=np.int32).reshape(-1)
    if fi_det.size == 0:
        raise RuntimeError(f"{NAME} | core_object_detections.detections.npz missing frame_indices (no-fallback)")
    if fi_det.shape[0] != len(frame_indices) or not np.all(fi_det == np.asarray(frame_indices, dtype=np.int32)):
        raise RuntimeError(f"{NAME} | frame_indices mismatch vs core_object_detections (no-fallback)")

    class_ids = np.asarray(det.get("class_ids"), dtype=np.int32)
    valid_mask = np.asarray(det.get("valid_mask"))
    if class_ids.ndim != 2 or valid_mask.ndim != 2:
        raise RuntimeError(f"{NAME} | core_object_detections invalid shapes for class_ids/valid_mask (no-fallback)")
    if class_ids.shape != valid_mask.shape:
        raise RuntimeError(f"{NAME} | core_object_detections class_ids/valid_mask shape mismatch (no-fallback)")

    name_to_id = _parse_class_names(det.get("class_names"))
    if "person" not in name_to_id:
        raise RuntimeError(f"{NAME} | core_object_detections.class_names missing 'person' (no-fallback)")
    person_id = int(name_to_id["person"])
    person_present = np.any((valid_mask.astype(bool)) & (class_ids == person_id), axis=1)
    person_positions = np.nonzero(person_present)[0].astype(int).tolist()
    radius = max(0, int(args.person_window_radius))
    if radius > 0 and person_positions:
        expanded = set()
        n = len(frame_indices)
        for p in person_positions:
            for q in range(int(p) - radius, int(p) + radius + 1):
                if 0 <= q < n:
                    expanded.add(int(q))
        person_positions = sorted(expanded)
    face_mesh_positions_override = set(person_positions)
    LOGGER.info(
        f"{NAME} | person-mask | person_id={person_id} person_frames={int(np.sum(person_present))} "
        f"face_mesh_frames={len(face_mesh_positions_override)} radius={radius}"
    )

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    # Get FPS for temporal filtering
    fps = float(meta.get("fps", 30.0))
    if fps <= 0:
        fps = 30.0
        LOGGER.warning(f"{NAME} | Invalid fps in metadata, using default 30.0")

    # Initialize profiler if enabled
    profiler = Profiler() if args.enable_profiling else None

    # Set progress context for processing functions
    _set_progress_context(rs_path=args.rs_path, platform_id=platform_id, video_id=video_id, run_id=run_id)

    # Emit load_deps stage (FrameManager + deps ready)
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )

    t_process_start = time.perf_counter()

    pose, hands, face, pose_present, hands_present, face_present = process_video(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        cfg=args,
        face_mesh_positions_override=face_mesh_positions_override,
        profiler=profiler,
        enable_async=args.enable_async,
        enable_parallel=args.enable_parallel,
        num_workers=args.num_workers,
        enable_temporal_filter=args.enable_temporal_filter,
        fps=fps,
    )

    timings["process_video"] = float(time.perf_counter() - t_process_start)

    # Clear context (no further progress from workers)
    _clear_progress_context()

    frame_manager.close()

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ARTIFACT_FILENAME)

    # Global availability flags for human-friendly consumers
    has_any_face = bool(np.any(face_present)) if face_present is not None else False
    has_any_pose = bool(np.any(pose_present)) if pose_present is not None else False
    has_any_hands = bool(np.any(hands_present)) if hands_present is not None else False

    # Valid empty reasons:
    # - no_faces_in_video is a valid empty for this provider (NOT an error)
    face_empty_reason: Optional[str] = None
    pose_empty_reason: Optional[str] = None
    hands_empty_reason: Optional[str] = None
    if args.use_face_mesh and not has_any_face:
        face_empty_reason = "no_faces_in_video"
    if args.use_pose and not has_any_pose:
        pose_empty_reason = "no_pose_detected"
    if args.use_hands and not has_any_hands:
        hands_empty_reason = "no_hands_detected"

    # Provider-level status: empty only for face-mesh absence (baseline consumer logic expects this)
    status = "empty" if face_empty_reason else "ok"
    empty_reason = face_empty_reason

    # Convert profiler timings into milliseconds (per-stage totals)
    stage_timings_ms: Dict[str, float] = {}
    if profiler is not None:
        summary = profiler.summary()
        for stage, stats in summary.items():
            total_s = float(stats.get("total", 0.0))
            stage_timings_ms[f"{stage}_total"] = total_s * 1000.0

    # Add top-level total timing
    timings["total"] = float(time.perf_counter() - t_total_start)
    for stage, sec in timings.items():
        stage_timings_ms[f"{stage}_total"] = float(sec) * 1000.0

    meta_out = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "status": status,
        "empty_reason": empty_reason,
        "model_name": "mediapipe",
        "total_frames": int(total_frames),
        # extended empty reasons (optional)
        "face_empty_reason": face_empty_reason,
        "pose_empty_reason": pose_empty_reason,
        "hands_empty_reason": hands_empty_reason,
        # dependency policy info
        "person_mask_enabled": True,
        "person_class_id": int(person_id),
        "person_frames_count": int(np.sum(person_present)),
        "person_window_radius": int(radius),
        "stage_timings_ms": stage_timings_ms,
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
        meta_out[k] = meta.get(k)

    # Required by contract (baseline may use "unknown")
    meta_out["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")

    # PR-3: model system baseline (mediapipe models are internal; treat as a model component)
    meta_out["device"] = str(meta_out.get("device") or "cpu")
    # Deterministic digest: best-effort "weights token" based on mediapipe version + key config params.
    mp_ver = str(mp.__version__ if hasattr(mp, "__version__") else "unknown")
    digest_payload: Dict[str, Any] = {
        "engine": "mediapipe",
        "mediapipe_version": mp_ver,
        "use_pose": bool(args.use_pose),
        "use_hands": bool(args.use_hands),
        "use_face_mesh": True,
        "pose": {
            "static_image_mode": bool(args.pose_static_image_mode),
            "model_complexity": int(args.pose_model_complexity),
            "enable_segmentation": bool(args.pose_enable_segmentation),
            "min_detection_confidence": float(args.pose_min_detection_confidence),
            "min_tracking_confidence": float(args.pose_min_tracking_confidence),
        },
        "hands": {
            "static_image_mode": bool(args.hands_static_image_mode),
            "max_num_hands": int(args.hands_max_num_hands),
            "model_complexity": int(args.hands_model_complexity),
            "min_detection_confidence": float(args.hands_min_detection_confidence),
            "min_tracking_confidence": float(args.hands_min_tracking_confidence),
        },
        "face_mesh": {
            "static_image_mode": bool(args.face_mesh_static_image_mode),
            "max_num_faces": int(args.face_mesh_max_num_faces),
            "refine_landmarks": bool(args.face_mesh_refine_landmarks),
            "min_detection_confidence": float(args.face_mesh_min_detection_confidence),
            "min_tracking_confidence": float(args.face_mesh_min_tracking_confidence),
        },
    }
    digest_text = json.dumps(digest_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    weights_digest = hashlib.sha256(digest_text.encode("utf-8")).hexdigest()
    meta_out = apply_models_meta(
        meta_out,
        models_used=[
            model_used(
                model_name="mediapipe",
                model_version=mp_ver,
                weights_digest=weights_digest,
                runtime="inprocess",
                engine="mediapipe",
                precision="fp32",
                device=str(meta_out.get("device") or "cpu"),
            )
        ],
    )

    np.savez_compressed(
        out_path,
        # legacy fields (kept)
        version=VERSION,
        created_at=meta_out["created_at"],
        model_name="mediapipe",
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        # times_s (required by baseline contract: union_timestamps_sec[frame_indices])
        times_s=times_s,
        pose_landmarks=pose,
        hands_landmarks=hands,
        face_landmarks=face,
        pose_present=pose_present,
        hands_present=hands_present,
        face_present=face_present,
        has_any_face=np.asarray(has_any_face),
        has_any_pose=np.asarray(has_any_pose),
        has_any_hands=np.asarray(has_any_hands),
        empty_reason=np.asarray(empty_reason, dtype=object),
        face_empty_reason=np.asarray(face_empty_reason, dtype=object),
        pose_empty_reason=np.asarray(pose_empty_reason, dtype=object),
        hands_empty_reason=np.asarray(hands_empty_reason, dtype=object),
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")

    # Emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )


if __name__ == "__main__":
    main()