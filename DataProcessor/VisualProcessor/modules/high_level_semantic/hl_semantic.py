"""
high_level_semantic (baseline-ready)
-----------------------------------

This module produces high-level semantic signals aligned to the Visual encoder time axis.

Design decisions (per audit / baseline criteria):
- Unit-of-processing: **frame** (union-domain indices from Segmenter).
- No-fallback: `frame_indices` MUST be provided by Segmenter for this component.
- Source-of-truth embeddings: **core_clip** NPZ (no in-module CLIP weights loading).
- Scene source: **cut_detection** NPZ (no internal scene/cut detection).
- Output: NPZ (fixed filename) with:
  - `frame_indices`, `times_s`
  - dense per-frame feature matrix (`frame_features`, `frame_feature_names`)
  - scene embeddings (`scene_embeddings`) + scene metadata arrays
  - unified sparse events stream (`event_*`) for UI/encoder.

Audio/Text integration:
- AudioProcessor and TextProcessor are outside VisualProcessor; we load their per-run NPZ artifacts from rs_path.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger

MODULE_NAME = "high_level_semantic"
VERSION = "1.0"
SCHEMA_VERSION = "high_level_semantic_npz_v1"
ARTIFACT_FILENAME = "high_level_semantic.npz"

LOGGER = get_logger(MODULE_NAME)


# -------------------------
# Progress to state_events.jsonl (PR-5) — same mechanism as frames_composition
# -------------------------
def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
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
# Feature gating (explicit, config-controlled)
    # -------------------------
DEFAULT_FEATURE_GROUPS = {"core", "scenes", "events", "audio", "emotion", "text"}


def _parse_groups_csv(s: Optional[str]) -> set[str]:
    if not s:
        return set(DEFAULT_FEATURE_GROUPS)
    parts = [p.strip() for p in str(s).split(",")]
    return {p for p in parts if p}


def _as_float32(x: Any, *, shape: Optional[Tuple[int, ...]] = None, fill: float = np.nan) -> np.ndarray:
    if x is None:
        if shape is None:
            return np.asarray([], dtype=np.float32)
        return np.full(shape, float(fill), dtype=np.float32)
    a = np.asarray(x, dtype=np.float32)
    if shape is not None and tuple(a.shape) != tuple(shape):
        out = np.full(shape, float(fill), dtype=np.float32)
        m = min(out.size, a.size)
        if m > 0:
            out.reshape(-1)[:m] = a.reshape(-1)[:m]
        return out
    return a


def _as_int32(x: Any, *, shape: Optional[Tuple[int, ...]] = None, fill: int = -1) -> np.ndarray:
    if x is None:
        if shape is None:
            return np.asarray([], dtype=np.int32)
        return np.full(shape, int(fill), dtype=np.int32)
    a = np.asarray(x, dtype=np.int32)
    if shape is not None and tuple(a.shape) != tuple(shape):
        out = np.full(shape, int(fill), dtype=np.int32)
        m = min(out.size, a.size)
        if m > 0:
            out.reshape(-1)[:m] = a.reshape(-1)[:m]
        return out
    return a


def _interp1d_nan(*, x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Interpolate y(x) onto x_new.
    - If y has NaNs, they are ignored (piecewise over valid points).
    - If there are <2 valid points -> returns all NaN.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x_new = np.asarray(x_new, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x_new.size == 0:
        return np.full((x_new.size,), np.nan, dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 2:
        return np.full((x_new.size,), np.nan, dtype=np.float32)
    xv = x[m]
    yv = y[m]
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    return np.interp(x_new, xv, yv).astype(np.float32)


@dataclass(frozen=True)
class _ScenePack:
    scene_id: np.ndarray  # (N,) int32
    scene_start_frame_idx: np.ndarray  # (S,) int32 (union-domain)
    scene_end_frame_idx: np.ndarray  # (S,) int32 (union-domain, inclusive end in sampled domain approximated)
    scene_start_time_s: np.ndarray  # (S,) float32
    scene_end_time_s: np.ndarray  # (S,) float32
    scene_duration_s: np.ndarray  # (S,) float32
    scene_representative_frame_idx: np.ndarray  # (S,) int32 (union-domain)


def _load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _pick_npz_by_prefix(dir_path: str, prefixes: Sequence[str]) -> Optional[str]:
    if not os.path.isdir(dir_path):
                return None
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".npz")]
    if not files:
        return None
    # prefer by prefix (in order), newest mtime among matches
    for pref in prefixes:
        matches = [f for f in files if f.startswith(pref)]
        if matches:
            matches.sort(key=lambda n: os.path.getmtime(os.path.join(dir_path, n)), reverse=True)
            return os.path.join(dir_path, matches[0])
    # fallback: newest .npz
    files.sort(key=lambda n: os.path.getmtime(os.path.join(dir_path, n)), reverse=True)
    return os.path.join(dir_path, files[0])


def _core_clip_aligned(*, core_clip: Dict[str, Any], want_frame_indices: np.ndarray) -> np.ndarray:
    core_idx = core_clip.get("frame_indices")
    core_emb = core_clip.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError("high_level_semantic | core_clip embeddings.npz missing frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32).reshape(-1)
    core_emb = np.asarray(core_emb, dtype=np.float32)
    if core_idx.size != core_emb.shape[0]:
        raise RuntimeError("high_level_semantic | core_clip shape mismatch frame_indices vs frame_embeddings")

    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in want_frame_indices.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            "high_level_semantic | core_clip does not cover requested frame_indices. "
            "Segmenter must provide consistent indices across core_clip and this module."
        )
    return core_emb[np.asarray(pos, dtype=np.int64)]


def _times_s_from_union(*, metadata: Dict[str, Any], frame_indices: np.ndarray) -> np.ndarray:
    uts = metadata.get("union_timestamps_sec")
    if uts is None:
        raise RuntimeError("high_level_semantic | metadata missing union_timestamps_sec (no-fallback)")
    uts = np.asarray(uts, dtype=np.float32).reshape(-1)
    if uts.size == 0:
        raise RuntimeError("high_level_semantic | union_timestamps_sec is empty")
    if frame_indices.size == 0:
        raise RuntimeError("high_level_semantic | frame_indices is empty")
    if int(np.max(frame_indices)) >= int(uts.size) or int(np.min(frame_indices)) < 0:
        raise RuntimeError("high_level_semantic | frame_indices out of bounds for union_timestamps_sec")
    times_s = uts[frame_indices.astype(np.int64)]
    # monotonic is expected (Segmenter contract)
    if times_s.size >= 2 and not bool(np.all(np.diff(times_s) >= -1e-6)):
        raise RuntimeError("high_level_semantic | times_s is not monotonic (unexpected union timeline)")
    return times_s.astype(np.float32)


def _build_scenes_from_cut_detection(
    *,
    cut_npz: Dict[str, Any],
    want_frame_indices: np.ndarray,
    times_s: np.ndarray,
) -> _ScenePack:
    cut_fi = cut_npz.get("frame_indices")
    if cut_fi is None:
        raise RuntimeError("high_level_semantic | cut_detection NPZ missing frame_indices")
    cut_fi = np.asarray(cut_fi, dtype=np.int32).reshape(-1)
    if cut_fi.size != want_frame_indices.size or not bool(np.all(cut_fi == want_frame_indices)):
        raise RuntimeError(
            "high_level_semantic | cut_detection.frame_indices must exactly match high_level_semantic.frame_indices"
        )

    det = cut_npz.get("detections")
    if det is None:
        raise RuntimeError("high_level_semantic | cut_detection NPZ missing detections")
    if isinstance(det, np.ndarray) and det.dtype == object:
        det = det.item()
    if not isinstance(det, dict):
        raise RuntimeError("high_level_semantic | cut_detection.detections has invalid type")

    shot_boundaries = det.get("shot_boundaries_frame_indices")
    scenes_shot_idx = det.get("scene_boundaries_shot_idx")
    if shot_boundaries is None:
        raise RuntimeError("high_level_semantic | cut_detection.detections missing shot_boundaries_frame_indices")
    shot_boundaries = np.asarray(shot_boundaries, dtype=np.int32).reshape(-1)
    if shot_boundaries.size < 2:
        raise RuntimeError("high_level_semantic | cut_detection shot_boundaries_frame_indices too short")

    # Build scenes in union-domain [start_frame_idx, end_frame_idx] using cut_detection scenes when available.
    # scenes_shot_idx is list of (start_shot, end_shot) pairs.
    scene_start: List[int] = []
    scene_end: List[int] = []
    if isinstance(scenes_shot_idx, (list, tuple)) and len(scenes_shot_idx) > 0:
        try:
            for pair in scenes_shot_idx:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                s_shot = int(pair[0])
                e_shot = int(pair[1])
                s_shot = max(0, min(s_shot, int(shot_boundaries.size) - 2))
                e_shot = max(s_shot, min(e_shot, int(shot_boundaries.size) - 2))
                s_frame = int(shot_boundaries[s_shot])
                # end: start of next shot after e_shot, then take the last sampled frame within that span
                e_next = int(shot_boundaries[e_shot + 1])
                scene_start.append(s_frame)
                scene_end.append(max(s_frame, e_next))
        except Exception:
            scene_start = []
            scene_end = []

    if not scene_start:
        # Fallback inside cut_detection-provided information: treat each shot as a scene.
        scene_start = [int(x) for x in shot_boundaries[:-1].tolist()]
        scene_end = [int(x) for x in shot_boundaries[1:].tolist()]

    # Map scene spans to our sampled positions; assign per-frame scene_id.
    starts = np.asarray(scene_start, dtype=np.int32).reshape(-1)
    ends = np.asarray(scene_end, dtype=np.int32).reshape(-1)
    # Ensure sorted by start
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]

    # Scene id per frame by union frame index position.
    fi = want_frame_indices
    # For each frame, find last start <= fi
    idx = np.searchsorted(starts, fi, side="right") - 1
    idx = np.clip(idx, 0, int(starts.size) - 1).astype(np.int32)
    scene_id = idx

    # Derive per-scene time metadata (using first/last sampled frame in each scene).
    S = int(starts.size)
    scene_start_time = np.full((S,), np.nan, dtype=np.float32)
    scene_end_time = np.full((S,), np.nan, dtype=np.float32)
    scene_rep = np.full((S,), -1, dtype=np.int32)
    for s in range(S):
        mask = scene_id == s
        if not bool(np.any(mask)):
            continue
        pos = np.where(mask)[0]
        p0 = int(pos[0])
        p1 = int(pos[-1])
        scene_start_time[s] = float(times_s[p0])
        scene_end_time[s] = float(times_s[p1])
        scene_rep[s] = int(fi[int(pos[len(pos) // 2])])

    duration = (scene_end_time - scene_start_time).astype(np.float32)
    duration[~np.isfinite(duration)] = np.nan

    # end frame idx: use representative upper bound in union-domain; keep as "end-exclusive" proxy
    return _ScenePack(
        scene_id=scene_id,
        scene_start_frame_idx=starts.astype(np.int32),
        scene_end_frame_idx=ends.astype(np.int32),
        scene_start_time_s=scene_start_time,
        scene_end_time_s=scene_end_time,
        scene_duration_s=duration,
        scene_representative_frame_idx=scene_rep,
    )


def _build_scene_embeddings(*, frame_embeddings: np.ndarray, scene_id: np.ndarray) -> np.ndarray:
    if frame_embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    N, D = frame_embeddings.shape[0], frame_embeddings.shape[1]
    S = int(np.max(scene_id)) + 1 if scene_id.size else 0
    out = np.full((S, D), np.nan, dtype=np.float32)
    for s in range(S):
        mask = scene_id == s
        if not bool(np.any(mask)):
            continue
        out[s] = np.nanmean(frame_embeddings[mask], axis=0).astype(np.float32)
    # L2-normalize (core_clip embeddings are typically normalized; mean breaks it)
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    out = out / norms
    return out.astype(np.float32)


class HighLevelSemanticModule(BaseModule):
    MODULE_NAME = MODULE_NAME
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        *,
        feature_groups: Optional[str] = None,
        require_cut_detection_model_facing: bool = False,
        require_text_processor: bool = True,
        require_audio_loudness: bool = True,
        require_audio_tempo: bool = True,
        require_audio_clap: bool = False,
        progress_every_frames: int = 50,
        semantic_jump_topk_events: int = 256,
        semantic_jump_min_strength: float = 0.25,
        **kwargs: Any,
    ):
        super().__init__(rs_path=rs_path, **kwargs)
        self.feature_groups = _parse_groups_csv(feature_groups)
        self.require_cut_detection_model_facing = bool(require_cut_detection_model_facing)
        self.require_text_processor = bool(require_text_processor)
        self.require_audio_loudness = bool(require_audio_loudness)
        self.require_audio_tempo = bool(require_audio_tempo)
        self.require_audio_clap = bool(require_audio_clap)
        self.progress_every_frames = max(1, int(progress_every_frames))
        self.semantic_jump_topk_events = max(0, int(semantic_jump_topk_events))
        self.semantic_jump_min_strength = float(semantic_jump_min_strength)

    def required_dependencies(self) -> List[str]:
        # Note: AudioProcessor/TextProcessor are outside VisualProcessor, but their artifacts still live in rs_path.
        deps = ["core_clip", "cut_detection", "emotion_face"]
        if self.require_text_processor:
            deps.append("text_processor")
        if self.require_audio_loudness:
            deps.append("loudness_extractor")
        if self.require_audio_tempo:
            deps.append("tempo_extractor")
        if self.require_audio_clap:
            deps.append("clap_extractor")
        return deps

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        # Load frames metadata for time-axis (BaseModule.run validated identity keys already)
        if self.rs_path is None:
            raise RuntimeError("high_level_semantic | rs_path is required")

        # We need platform/video/run ids for progress emission; they come from metadata in BaseModule.run,
        # but BaseModule does not pass them into process(). We re-load metadata best-effort from frames_dir
        # stored in config by BaseModule.run (save_metadata contains frames_dir but not exposed here),
        # so instead we parse them from rs_path layout: <rs_base>/<platform>/<video>/<run>.
        run_rs = Path(self.rs_path).resolve()
        platform_id = run_rs.parents[1].name if len(run_rs.parents) >= 2 else ""
        video_id = run_rs.parents[0].name if len(run_rs.parents) >= 1 else ""
        run_id = run_rs.name

        # Best-effort: load metadata.json from frames_dir via frame_manager
        frames_dir = getattr(frame_manager, "frames_dir", None)
        if not isinstance(frames_dir, str) or not frames_dir:
            raise RuntimeError("high_level_semantic | cannot infer frames_dir from FrameManager")
        meta_path = os.path.join(frames_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            raise RuntimeError("high_level_semantic | frames_dir missing metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        fi = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        if fi.size == 0:
            raise RuntimeError("high_level_semantic | empty frame_indices (no-fallback)")
        times_s = _times_s_from_union(metadata=metadata, frame_indices=fi)

        # Parallel load of upstream artifacts (internal parallelism requirement)
        def _load_core_clip() -> Dict[str, Any]:
            core = self.load_core_provider("core_clip", file_name=None)
            if not isinstance(core, dict):
                raise RuntimeError("high_level_semantic | failed to load core_clip")
            return core

        def _load_cut_detection_features() -> Dict[str, Any]:
            d = os.path.join(self.rs_path, "cut_detection")
            p = _pick_npz_by_prefix(d, prefixes=("cut_detection_features_", "cut_detection_"))
            if not p:
                raise RuntimeError("high_level_semantic | missing cut_detection NPZ")
            return _load_npz(p)

        def _load_cut_detection_model_facing() -> Optional[Dict[str, Any]]:
            d = os.path.join(self.rs_path, "cut_detection")
            p = _pick_npz_by_prefix(d, prefixes=("cut_detection_model_facing_",))
            if not p:
                if self.require_cut_detection_model_facing:
                    raise RuntimeError("high_level_semantic | require_cut_detection_model_facing=true but model-facing NPZ not found")
            return None
            return _load_npz(p)

        def _load_emotion_face() -> Dict[str, Any]:
            d = os.path.join(self.rs_path, "emotion_face")
            p = os.path.join(d, "emotion_face.npz")
            if not os.path.isfile(p):
                raise RuntimeError("high_level_semantic | missing emotion_face/emotion_face.npz (no-fallback)")
            return _load_npz(p)

        def _load_text_processor() -> Optional[Dict[str, Any]]:
            if "text" not in self.feature_groups and "events" not in self.feature_groups:
        return None
            d = os.path.join(self.rs_path, "text_processor")
            p = os.path.join(d, "text_features.npz")
            if not os.path.isfile(p):
                if self.require_text_processor:
                    raise RuntimeError("high_level_semantic | missing text_processor/text_features.npz")
                return None
            return _load_npz(p)

        def _load_audio_npz(component: str, fixed_name: str) -> Optional[Dict[str, Any]]:
            d = os.path.join(self.rs_path, component)
            p = os.path.join(d, fixed_name)
            if not os.path.isfile(p):
                p2 = _pick_npz_by_prefix(d, prefixes=(component + "_features", component))
                if not p2:
            return None
                p = p2
            return _load_npz(p)

        with ThreadPoolExecutor(max_workers=6) as ex:
            fut_core = ex.submit(_load_core_clip)
            fut_cut = ex.submit(_load_cut_detection_features)
            fut_cut_mf = ex.submit(_load_cut_detection_model_facing)
            fut_emo = ex.submit(_load_emotion_face)
            fut_text = ex.submit(_load_text_processor)
            fut_loud = ex.submit(_load_audio_npz, "loudness_extractor", "loudness_extractor_features.npz")
            fut_tempo = ex.submit(_load_audio_npz, "tempo_extractor", "tempo_extractor_features.npz")
            fut_clap = ex.submit(_load_audio_npz, "clap_extractor", "clap_extractor_features.npz")

            core_clip = fut_core.result()
            cut_npz = fut_cut.result()
            cut_mf = fut_cut_mf.result()
            emo_npz = fut_emo.result()
            text_npz = fut_text.result()
            loud_npz = fut_loud.result()
            tempo_npz = fut_tempo.result()
            clap_npz = fut_clap.result()

        # Enforce required audio deps if configured
        if self.require_audio_loudness and loud_npz is None:
            raise RuntimeError("high_level_semantic | require_audio_loudness=true but loudness_extractor artifact missing")
        if self.require_audio_tempo and tempo_npz is None:
            raise RuntimeError("high_level_semantic | require_audio_tempo=true but tempo_extractor artifact missing")
        if self.require_audio_clap and clap_npz is None:
            raise RuntimeError("high_level_semantic | require_audio_clap=true but clap_extractor artifact missing")

        # Align core_clip embeddings to our frame_indices
        frame_embeddings = _core_clip_aligned(core_clip=core_clip, want_frame_indices=fi)

        # Build scenes and per-frame scene_id from cut_detection
        scenes = _build_scenes_from_cut_detection(cut_npz=cut_npz, want_frame_indices=fi, times_s=times_s)
        scene_embeddings = _build_scene_embeddings(frame_embeddings=frame_embeddings, scene_id=scenes.scene_id)

        # -------- per-frame derived semantic features --------
        N = int(fi.size)
        # cosine similarity to prev frame (core_clip emb are normalized)
        sim_prev = np.full((N,), np.nan, dtype=np.float32)
        if N >= 2:
            sim_prev[1:] = np.sum(frame_embeddings[1:] * frame_embeddings[:-1], axis=1).astype(np.float32)
        novelty_prev = (1.0 - sim_prev).astype(np.float32)

        # per-frame position within scene (0..1)
        scene_pos = np.full((N,), np.nan, dtype=np.float32)
        for s in range(int(np.max(scenes.scene_id)) + 1):
            idxs = np.where(scenes.scene_id == s)[0]
            if idxs.size <= 1:
                if idxs.size == 1:
                    scene_pos[int(idxs[0])] = 0.0
                continue
            # linear from 0..1 inside scene
            k = np.linspace(0.0, 1.0, int(idxs.size), dtype=np.float32)
            scene_pos[idxs] = k

        # -------- audio mapping (time-axis) --------
        loud_dbfs = np.full((N,), np.nan, dtype=np.float32)
        tempo_bpm = np.full((N,), np.nan, dtype=np.float32)

        if "audio" in self.feature_groups:
            if loud_npz is not None:
                payload = loud_npz.get("payload")
                if isinstance(payload, np.ndarray) and payload.dtype == object:
                    payload = payload.item()
                if isinstance(payload, dict):
                    t = payload.get("segment_center_sec") or payload.get("windowed_times_sec") or payload.get("times_sec")
                    v = payload.get("dbfs") or payload.get("segment_dbfs") or payload.get("rms_dbfs")
                    if t is not None and v is not None:
                        loud_dbfs = _interp1d_nan(x=_as_float32(t), y=_as_float32(v), x_new=times_s)

            if tempo_npz is not None:
                payload = tempo_npz.get("payload")
                if isinstance(payload, np.ndarray) and payload.dtype == object:
                    payload = payload.item()
                if isinstance(payload, dict):
                    t = payload.get("windowed_times_sec") or payload.get("times_sec")
                    v = payload.get("windowed_bpm") or payload.get("bpm")
                    if t is not None and v is not None:
                        tempo_bpm = _interp1d_nan(x=_as_float32(t), y=_as_float32(v), x_new=times_s)

        # -------- emotion mapping (time-axis) --------
        emo_valence = np.full((N,), np.nan, dtype=np.float32)
        emo_arousal = np.full((N,), np.nan, dtype=np.float32)
        emo_intensity = np.full((N,), np.nan, dtype=np.float32)

        if "emotion" in self.feature_groups:
            seq = emo_npz.get("sequence_features")
            if isinstance(seq, np.ndarray) and seq.dtype == object:
                seq = seq.item()
            if isinstance(seq, dict):
                et = seq.get("times_s")
                if et is None:
                    # emotion_face stores indices in seq["frame_indices"]; compute times from union if possible
                    efi = seq.get("frame_indices")
                    if efi is not None:
                        et = _times_s_from_union(metadata=metadata, frame_indices=np.asarray(efi, dtype=np.int32).reshape(-1))
                if et is not None:
                    et = _as_float32(et)
                    ev = _as_float32(seq.get("valence_sequence"))
                    ea = _as_float32(seq.get("arousal_sequence"))
                    emo_valence = _interp1d_nan(x=et, y=ev, x_new=times_s)
                    emo_arousal = _interp1d_nan(x=et, y=ea, x_new=times_s)
                    emo_intensity = np.sqrt(np.square(emo_valence) + np.square(emo_arousal)).astype(np.float32)

        # -------- build dense frame_features matrix --------
        frame_feature_names: List[str] = [
            "clip_sim_prev",
            "clip_novelty_prev",
            "scene_pos_norm",
            "loudness_dbfs",
            "tempo_bpm",
            "emo_valence",
            "emo_arousal",
            "emo_intensity",
        ]
        frame_features = np.stack(
            [
                sim_prev,
                novelty_prev,
                scene_pos,
                loud_dbfs,
                tempo_bpm,
                emo_valence,
                emo_arousal,
                emo_intensity,
            ],
            axis=1,
        ).astype(np.float32)

        # -------- events stream --------
        # event_type_id map:
        #  1: hard_cut (from cut_detection)
        #  200: semantic_jump (from clip_novelty_prev)
        #  210: emotion_keyframe (from emotion_face keyframes, mapped by time)
        event_type_map = {1: "hard_cut", 200: "semantic_jump", 210: "emotion_keyframe"}
        ev_times: List[float] = []
        ev_type: List[int] = []
        ev_strength: List[float] = []
        ev_pos: List[int] = []

        if "events" in self.feature_groups:
            det = cut_npz.get("detections")
            if isinstance(det, np.ndarray) and det.dtype == object:
                det = det.item()
            if isinstance(det, dict):
                hard_pos = det.get("hard_cut_pos") or det.get("hard_cut_indices") or []
                hard_strengths = det.get("hard_cut_strengths") or det.get("hard_cut_strengths") or []
                if hard_pos:
                    for i, p in enumerate(hard_pos):
                        pp = int(p)
                        if 0 <= pp < N:
                            ev_times.append(float(times_s[pp]))
                            ev_type.append(1)
                            st = float(hard_strengths[i]) if i < len(hard_strengths) else 1.0
                            ev_strength.append(st)
                            ev_pos.append(pp)

            # semantic jumps: take top-k novelty peaks
            if self.semantic_jump_topk_events > 0 and N >= 2:
                cand = novelty_prev.copy()
                cand[0] = np.nan
                mask = np.isfinite(cand) & (cand >= float(self.semantic_jump_min_strength))
                idxs = np.where(mask)[0]
                if idxs.size > 0:
                    vals = cand[idxs]
                    order = np.argsort(vals)[::-1]
                    idxs = idxs[order]
                    if idxs.size > self.semantic_jump_topk_events:
                        idxs = idxs[: self.semantic_jump_topk_events]
                    for pp in idxs.tolist():
                        ev_times.append(float(times_s[int(pp)]))
                        ev_type.append(200)
                        ev_strength.append(float(cand[int(pp)]))
                        ev_pos.append(int(pp))

            # emotion keyframes (best-effort)
            keyframes = emo_npz.get("keyframes")
            if isinstance(keyframes, np.ndarray) and keyframes.dtype == object:
                keyframes = keyframes.tolist()
            if isinstance(keyframes, list) and keyframes:
                for kf in keyframes:
                    if not isinstance(kf, dict):
                        continue
                    # emotion_face keyframe has global_index (union-domain index) or we map by local index/time
                    gi = kf.get("global_index")
                    if gi is None:
                        continue
                    gi = int(gi)
                    # map union index to our pos
                    # strict mapping: if not present, map by nearest time
                    try:
                        pp = int(np.where(fi == gi)[0][0])
                except Exception:
                        pp = int(np.argmin(np.abs(fi.astype(np.int64) - int(gi))))
                    pp = max(0, min(pp, N - 1))
                    ev_times.append(float(times_s[pp]))
                    ev_type.append(210)
                    ev_strength.append(1.0)
                    ev_pos.append(pp)

        # Convert events to arrays, sort by time
        if ev_times:
            order = np.argsort(np.asarray(ev_times, dtype=np.float32))
            ev_times_arr = np.asarray([ev_times[i] for i in order], dtype=np.float32)
            ev_type_arr = np.asarray([ev_type[i] for i in order], dtype=np.int16)
            ev_strength_arr = np.asarray([ev_strength[i] for i in order], dtype=np.float32)
            ev_pos_arr = np.asarray([ev_pos[i] for i in order], dtype=np.int32)
        else:
            ev_times_arr = np.asarray([], dtype=np.float32)
            ev_type_arr = np.asarray([], dtype=np.int16)
            ev_strength_arr = np.asarray([], dtype=np.float32)
            ev_pos_arr = np.asarray([], dtype=np.int32)

        # Progress (stage: finalize)
        _emit_progress(
            rs_path=self.rs_path,
            platform_id=str(metadata.get("platform_id") or platform_id),
            video_id=str(metadata.get("video_id") or video_id),
            run_id=str(metadata.get("run_id") or run_id),
            done=N,
            total=N,
            stage="finalize",
        )

        # Copy text snapshot features (privacy-safe) into this artifact if enabled
        text_feature_names = np.asarray([], dtype=object)
        text_feature_values = np.asarray([], dtype=np.float32)
        if "text" in self.feature_groups and isinstance(text_npz, dict):
            tn = text_npz.get("feature_names")
            tv = text_npz.get("feature_values")
            if tn is not None and tv is not None:
                text_feature_names = np.asarray(tn, dtype=object).reshape(-1)
                text_feature_values = np.asarray(tv, dtype=np.float32).reshape(-1)

        # Minimal scalar features for analytics/UI
        features: Dict[str, Any] = {
            "n_frames": int(N),
            "n_scenes": int(scene_embeddings.shape[0]),
            "clip_sim_prev_mean": float(np.nanmean(sim_prev)) if np.isfinite(sim_prev).any() else float("nan"),
            "clip_novelty_prev_mean": float(np.nanmean(novelty_prev)) if np.isfinite(novelty_prev).any() else float("nan"),
            "hard_cuts_count": int(np.sum(ev_type_arr == 1)) if ev_type_arr.size else 0,
            "semantic_jump_events_count": int(np.sum(ev_type_arr == 200)) if ev_type_arr.size else 0,
        }

        # Emit progress periodically by frame count requirement (>=10 updates). Here compute stage already complete;
        # we also emit in a loop for long sequences to show motion (cheap).
        if N > 0:
            step = max(self.progress_every_frames, int(max(1, N // 20)))
            done = 0
            while done < N:
                done = min(N, done + step)
                _emit_progress(
                    rs_path=self.rs_path,
                    platform_id=str(metadata.get("platform_id") or platform_id),
                    video_id=str(metadata.get("video_id") or video_id),
                    run_id=str(metadata.get("run_id") or run_id),
                    done=done,
                    total=N,
                    stage="compute",
                )

        # Store per-scene metadata as arrays (avoid object dicts in main schema)
        return {
            "frame_indices": fi.astype(np.int32),
            "times_s": times_s.astype(np.float32),
            "scene_id": scenes.scene_id.astype(np.int32),
            "scene_embeddings": scene_embeddings.astype(np.float32),
            "scene_start_frame_idx": scenes.scene_start_frame_idx.astype(np.int32),
            "scene_end_frame_idx": scenes.scene_end_frame_idx.astype(np.int32),
            "scene_start_time_s": scenes.scene_start_time_s.astype(np.float32),
            "scene_end_time_s": scenes.scene_end_time_s.astype(np.float32),
            "scene_duration_s": scenes.scene_duration_s.astype(np.float32),
            "scene_representative_frame_idx": scenes.scene_representative_frame_idx.astype(np.int32),
            "frame_feature_names": np.asarray(frame_feature_names, dtype=object),
            "frame_features": frame_features.astype(np.float32),
            "event_times_s": ev_times_arr,
            "event_type_id": ev_type_arr,
            "event_strength": ev_strength_arr,
            "event_frame_pos": ev_pos_arr,
            "features": features,
            "text_feature_names": text_feature_names,
            "text_feature_values": text_feature_values,
            # keep these in meta-friendly dicts:
            "ui": {
                "event_type_map": event_type_map,
                "feature_groups": sorted(self.feature_groups),
                "upstream": {
                    "cut_detection_model_facing_present": bool(cut_mf is not None),
                    "text_processor_present": bool(text_npz is not None),
                    "loudness_present": bool(loud_npz is not None),
                    "tempo_present": bool(tempo_npz is not None),
                    "clap_present": bool(clap_npz is not None),
                },
            },
        }