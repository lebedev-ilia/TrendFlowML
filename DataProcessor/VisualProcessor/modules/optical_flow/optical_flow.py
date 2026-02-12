"""
optical_flow (module)

Production policy:
- This module is a CONSUMER of `core_optical_flow` (NPZ) and MUST NOT compute RAFT itself.
- No JSON artifacts in result_store (NPZ-only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.video_context import VideoContext


MODULE_NAME = "optical_flow"
VERSION = "1.0"
SCHEMA_VERSION = "optical_flow_npz_v1"
ARTIFACT_FILENAME = "optical_flow.npz"


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (PR-5).
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


def _times_s_from_union(*, metadata: Dict[str, Any], frame_indices: np.ndarray) -> np.ndarray:
    uts = metadata.get("union_timestamps_sec")
    if uts is None:
        raise RuntimeError("optical_flow | metadata missing union_timestamps_sec (no-fallback)")
    uts = np.asarray(uts, dtype=np.float32).reshape(-1)
    if uts.size == 0:
        raise RuntimeError("optical_flow | union_timestamps_sec is empty")
    if frame_indices.size == 0:
        raise RuntimeError("optical_flow | frame_indices is empty")
    if int(np.max(frame_indices)) >= int(uts.size) or int(np.min(frame_indices)) < 0:
        raise RuntimeError("optical_flow | frame_indices out of bounds for union_timestamps_sec")
    times_s = uts[frame_indices.astype(np.int64)]
    if times_s.size >= 2 and not bool(np.all(np.diff(times_s) >= -1e-6)):
        raise RuntimeError("optical_flow | times_s is not monotonic (unexpected union timeline)")
    return times_s.astype(np.float32)


def _load_core_optical_flow(module: BaseModule) -> Dict[str, Any]:
    core = module.load_core_provider("core_optical_flow", file_name="flow.npz")
    if not isinstance(core, dict):
        raise FileNotFoundError("optical_flow | missing dependency: core_optical_flow/flow.npz")
    idx = core.get("frame_indices")
    curve = core.get("motion_norm_per_sec_mean")
    if idx is None or curve is None:
        raise RuntimeError("optical_flow | core_optical_flow/flow.npz missing keys frame_indices/motion_norm_per_sec_mean")
    return {
        "frame_indices": np.asarray(idx, dtype=np.int32).reshape(-1),
        "motion_norm_per_sec_mean": np.asarray(curve, dtype=np.float32).reshape(-1),
        "meta": core.get("meta"),
    }


class OpticalFlowModule(BaseModule):
    MODULE_NAME = MODULE_NAME
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME

    @property
    def supports_batch(self) -> bool:
        """Optical flow module supports batch processing (CPU-only consumer)."""
        return True

    def required_dependencies(self) -> List[str]:
        return ["core_optical_flow"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Baseline-ready consumer:
        - consumes `core_optical_flow/flow.npz`
        - aligns to THIS module's Segmenter-owned frame_indices
        - produces a time-axis aligned curve + aggregates
        """
        # NOTE: we intentionally ignore frame_manager (consumer-only).
        if self.rs_path is None:
            raise ValueError("optical_flow | rs_path is required")
        if not frame_indices:
            raise ValueError("optical_flow | frame_indices is empty")
        metadata = config.get("_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("optical_flow | internal error: _metadata missing (run() must provide)")

        want = np.asarray([int(i) for i in frame_indices], dtype=np.int32).reshape(-1)
        times_s = _times_s_from_union(metadata=metadata, frame_indices=want)

        core = _load_core_optical_flow(self)
        core_idx = core["frame_indices"]
        core_curve = core["motion_norm_per_sec_mean"]

        # Propagate upstream empty if core provider is empty
        core_meta = None
        try:
            core_meta = core.get("meta")
            if isinstance(core_meta, np.ndarray) and core_meta.dtype == object and core_meta.shape == ():
                core_meta = core_meta.item()
        except Exception:
            core_meta = core.get("meta")
        if isinstance(core_meta, dict) and core_meta.get("status") == "empty":
            empty_reason = core_meta.get("empty_reason") or "dependency_missing"
            return {
                "frame_indices": want,
                "times_s": times_s,
                "motion_norm_per_sec_mean": np.full((int(want.size),), np.nan, dtype=np.float32),
                "features": {},
                "summary": {
                    "success": False,
                    "total_frames": int(want.size),
                    "reason": "dependency_empty",
                },
                "__meta_override__": {"status": "empty", "empty_reason": str(empty_reason)},
            }

        # Align curve to module sampling (use NaN for missing frames)
        mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
        curve = np.full((int(want.size),), np.nan, dtype=np.float32)
        missing_count = 0
        for j, fi in enumerate(want.tolist()):
            p = mapping.get(int(fi), -1)
            if p < 0:
                # Missing frame: use NaN (will be ignored in statistics)
                missing_count += 1
                curve[j] = np.nan
            else:
                curve[j] = float(core_curve[int(p)])
        
        # Log warning if some frames are missing (but don't fail)
        if missing_count > 0:
            try:
                self.logger.warning(
                    f"optical_flow | {missing_count}/{want.size} frame_indices not found in core_optical_flow. "
                    f"Using NaN for missing frames (will be ignored in statistics)."
                )
            except Exception:
                pass  # Best-effort logging

        # Aggregates (ignore NaN, and ignore first element if it represents 'no prev frame')
        curve_for_stats = curve[1:] if curve.size >= 2 else curve
        features: Dict[str, Any] = {
            "motion_curve_mean": float(np.nanmean(curve_for_stats)) if curve_for_stats.size else float("nan"),
            "motion_curve_median": float(np.nanmedian(curve_for_stats)) if curve_for_stats.size else float("nan"),
            "motion_curve_p90": float(np.nanpercentile(curve_for_stats, 90)) if curve_for_stats.size else float("nan"),
            "motion_curve_variance": float(np.nanvar(curve_for_stats)) if curve_for_stats.size else float("nan"),
        }

        # Minimal UI payload (NPZ meta only)
        ui_payload = {
            "schema_version": "optical_flow_ui_v1",
            "frame_indices": want.tolist(),
            "times_s": times_s.tolist(),
            "motion_norm_per_sec_mean": curve.tolist(),
        }

        return {
            "frame_indices": want,
            "times_s": times_s,
            "motion_norm_per_sec_mean": curve.astype(np.float32),
            "features": features,
            "ui_payload": ui_payload,
        }

    def run(self, frames_dir: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Custom run() to support:
        - progress events (unit=frame)
        - stage timings
        - passing metadata into process() without changing BaseModule API
        """
        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise ValueError(f"{self.module_name} | Нет кадров для обработки")

        platform_id = str(metadata.get("platform_id"))
        video_id = str(metadata.get("video_id"))
        run_id = str(metadata.get("run_id"))
        total = int(len(frame_indices))

        t0 = time.perf_counter()
        _emit_progress(
            rs_path=str(self.rs_path or ""),
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            done=0,
            total=total,
            stage="start",
        )

        frame_manager = None
        try:
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            t_fm = time.perf_counter()

            # process() requires metadata, pass it via config
            cfg = dict(config or {})
            cfg["_metadata"] = metadata

            results = self.process(frame_manager=frame_manager, frame_indices=frame_indices, config=cfg)
            t_proc = time.perf_counter()

            # Pull ui_payload out of results into meta (avoid storing large json as a top-level key)
            ui_payload = None
            if isinstance(results, dict) and "ui_payload" in results:
                try:
                    ui_payload = results.pop("ui_payload")
                except Exception:
                    ui_payload = None

            # stage timings in summary (baseline profiling criterion 1.14)
            stage_timings_ms = {
                "frame_manager_ms": float((t_fm - t0) * 1000.0),
                "process_ms": float((t_proc - t_fm) * 1000.0),
            }
            if isinstance(results, dict):
                summary = results.get("summary")
                if not isinstance(summary, dict):
                    summary = {}
                summary["stage_timings_ms"] = stage_timings_ms
                results["summary"] = summary

            _emit_progress(
                rs_path=str(self.rs_path or ""),
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=max(1, total // 2),
                total=total,
                stage="processed",
            )

            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": len(frame_indices),
                "frames_dir": frames_dir,
                "platform_id": metadata.get("platform_id"),
                "video_id": metadata.get("video_id"),
                "run_id": metadata.get("run_id"),
                "sampling_policy_version": metadata.get("sampling_policy_version"),
                "config_hash": metadata.get("config_hash"),
                "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                "analysis_fps": metadata.get("analysis_fps"),
                "analysis_width": metadata.get("analysis_width"),
                "analysis_height": metadata.get("analysis_height"),
            }
            if ui_payload is not None:
                save_metadata["ui_payload"] = ui_payload

            # models_used: consumer relies on core_optical_flow (Triton RAFT), so no direct models here
            save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})

            # Apply __meta_override__ if present
            meta_override = None
            if isinstance(results, dict) and "__meta_override__" in results:
                try:
                    meta_override = results.pop("__meta_override__")
                except Exception:
                    meta_override = None
            if isinstance(meta_override, dict) and meta_override:
                for k, v in meta_override.items():
                    if isinstance(k, str) and k and (isinstance(v, (str, int, float, bool)) or v is None):
                        save_metadata[k] = v

            saved_path = self.save_results(results=results, metadata=save_metadata, use_compressed=False)

            _emit_progress(
                rs_path=str(self.rs_path or ""),
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=total,
                total=total,
                stage="done",
            )

            return saved_path
        finally:
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception:
                    pass

    def process_batch(
        self,
        video_contexts: List[VideoContext],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Batch processing для optical_flow.
        
        Optical flow module - это consumer модуль, который обрабатывает данные из core_optical_flow.
        Batch processing выполняется последовательно для каждого видео (CPU-only операция).
        
        Args:
            video_contexts: Список VideoContext для каждого видео
            config: Конфигурация модуля
            
        Returns:
            Список результатов для каждого видео
        """
        results = []
        
        for video_ctx in video_contexts:
            try:
                # Создаем модуль для каждого видео с его rs_path
                module = OpticalFlowModule(rs_path=video_ctx.rs_path)
                
                # Загружаем метаданные
                metadata = video_ctx.load_metadata()
                
                # Выполняем обработку
                saved_path = module.run(
                    frames_dir=video_ctx.frames_dir,
                    config=config,
                    metadata=metadata
                )
                
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "ok",
                    "saved_path": saved_path,
                })
            except Exception as e:
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "error",
                    "error": str(e),
                })
        
        return results


