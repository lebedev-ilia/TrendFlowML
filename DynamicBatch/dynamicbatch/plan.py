from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cost_provider import CostProvider, CostQuery, DbCostProvider, FileCostProvider
from .resource_costs import UnitCost, gpu_mem_per_task_mb
from .runtime_cfg import VisualBatchOverrides, build_visual_runtime_cfg, write_temp_yaml
from .subprocess_runner import RunResult, run_subprocess
from .system_probe import probe_gpu_mem_mb


@dataclass(frozen=True)
class SchedulerKnobs:
    max_parallel: int = 1
    backoff_sec: float = 2.0
    oom_retries: int = 3
    headroom_ratio: float = 0.25
    min_headroom_mb: int = 1024


def _compute_batch_size(
    *,
    free_vram_mb: int,
    per_task_mb: int,
    knobs: SchedulerKnobs,
    max_batch_cap: int,
) -> int:
    free = max(0, int(free_vram_mb))
    headroom = max(int(knobs.min_headroom_mb), int(round(float(free) * float(knobs.headroom_ratio))))
    budget = max(0, free - headroom)
    denom = max(1, int(per_task_mb))
    bs = budget // denom if budget > 0 else 0
    bs = max(1, min(int(bs), int(max_batch_cap)))
    return int(bs)


def _halve_or_one(x: int) -> int:
    xx = max(1, int(x))
    if xx <= 1:
        return 1
    return max(1, xx // 2)


def build_visual_overrides_from_costs(
    *,
    cost_provider: CostProvider,
    prefer_clip_branch: str = "336",
    prefer_midas_branch: str = "384",
    knobs: SchedulerKnobs,
    sim_free_vram_mb: Optional[int] = None,
) -> Tuple[VisualBatchOverrides, Dict[str, int]]:
    """
    Computes scheduler-controlled batch sizes for baseline core providers.
    Returns (overrides, debug_info).
    """
    gpu = probe_gpu_mem_mb()
    free_vram_mb = int(gpu.free_mb) if gpu is not None else 0
    if sim_free_vram_mb is not None:
        free_vram_mb = max(0, int(sim_free_vram_mb))

    clip_cost = cost_provider.get_cost(CostQuery(component_id="core_clip.clip_image", prefer_branch=prefer_clip_branch))
    midas_cost = cost_provider.get_cost(CostQuery(component_id="core_depth_midas.midas", prefer_branch=prefer_midas_branch))

    # MVP caps (must be <= Triton max_batch_size / component constraints)
    clip_cap = 64
    midas_cap = 8
    raft_cap = 4
    yolo_cap = 8

    clip_mem = gpu_mem_per_task_mb(clip_cost, default_mb=32)
    midas_mem = gpu_mem_per_task_mb(midas_cost, default_mb=96)

    clip_bs = _compute_batch_size(
        free_vram_mb=free_vram_mb,
        per_task_mb=clip_mem,
        knobs=knobs,
        max_batch_cap=clip_cap,
    )
    midas_bs = _compute_batch_size(
        free_vram_mb=free_vram_mb,
        per_task_mb=midas_mem,
        knobs=knobs,
        max_batch_cap=midas_cap,
    )

    # For components without costs yet, keep conservative caps (MVP).
    raft_bs = 1
    yolo_bs = 1

    dbg = {
        "free_vram_mb": free_vram_mb,
        "clip_mem_mb": clip_mem,
        "midas_mem_mb": midas_mem,
        "clip_bs": clip_bs,
        "midas_bs": midas_bs,
        "raft_bs": raft_bs,
        "yolo_bs": yolo_bs,
        "clip_cap": clip_cap,
        "midas_cap": midas_cap,
        "raft_cap": raft_cap,
        "yolo_cap": yolo_cap,
    }

    return (
        VisualBatchOverrides(
            core_clip_batch_size=int(clip_bs),
            core_depth_midas_batch_size=int(midas_bs),
            core_optical_flow_batch_size=int(raft_bs),
            core_object_detections_batch_size=int(yolo_bs),
        ),
        dbg,
    )


def run_dataprocessor_job(
    *,
    repo_root: str,
    video_path: str,
    rs_base: str,
    visual_cfg_path: str,
    dag_stage: str,
    profile_path: Optional[str],
    dp_models_root: Optional[str],
    triton_http_url: Optional[str],
    dp_python: str,
    knobs: SchedulerKnobs,
    costs_dir: str,
    costs_provider: str,
    db_dsn: Optional[str],
    db_table: str,
    dry_run: bool,
    platform_id: str = "local",
    video_id: Optional[str] = None,
    run_id: Optional[str] = None,
    sim_free_vram_mb: Optional[int] = None,
    audio_segment_parallelism: Optional[int] = None,
    audio_max_inflight: Optional[int] = None,
    audio_clap_batch_size: Optional[int] = None,
) -> RunResult:
    """
    Runs a single DataProcessor job with scheduler-generated VisualProcessor config overrides.
    """
    dp_root = os.path.join(repo_root, "DataProcessor")
    dp_main = os.path.join(dp_root, "main.py")
    if str(costs_provider).lower() == "db":
        if not db_dsn:
            raise RuntimeError("--costs-provider=db requires --db-dsn")
        cost_provider: CostProvider = DbCostProvider(dsn=str(db_dsn), table=str(db_table))
    else:
        resource_costs_dir = costs_dir
        cost_provider = FileCostProvider(resource_costs_dir=resource_costs_dir)

    overrides, _dbg = build_visual_overrides_from_costs(cost_provider=cost_provider, knobs=knobs, sim_free_vram_mb=sim_free_vram_mb)
    runtime_cfg = build_visual_runtime_cfg(base_cfg_path=visual_cfg_path, overrides=overrides)
    tmp_visual_cfg = write_temp_yaml(runtime_cfg)
    tmp_cfg_paths: List[str] = [tmp_visual_cfg]

    env: Dict[str, str] = {}
    if dp_models_root:
        env["DP_MODELS_ROOT"] = str(dp_models_root)
    if triton_http_url:
        env["TRITON_HTTP_URL"] = str(triton_http_url)

    try:
        vid = video_id or os.path.splitext(os.path.basename(video_path))[0]
        rid = run_id or (time.strftime("%Y%m%d-%H%M%S") + "_" + os.urandom(3).hex())
        cmd = [
            str(dp_python or "python3"),
            dp_main,
            "--video-path",
            os.path.abspath(video_path),
            "--rs-base",
            os.path.abspath(rs_base),
            "--visual-cfg-path",
            os.path.abspath(tmp_visual_cfg),
            "--dag-stage",
            str(dag_stage),
            "--platform-id",
            str(platform_id),
            f"--video-id={vid}",
            "--run-id",
            str(rid),
        ]
        if audio_segment_parallelism is not None:
            cmd.extend(["--audio-segment-parallelism", str(int(audio_segment_parallelism))])
        if audio_max_inflight is not None:
            cmd.extend(["--audio-max-inflight", str(int(audio_max_inflight))])
        if audio_clap_batch_size is not None:
            cmd.extend(["--audio-clap-batch-size", str(int(audio_clap_batch_size))])
        if profile_path:
            cmd.extend(["--profile-path", os.path.abspath(profile_path)])

        if dry_run:
            return RunResult(ok=True, returncode=0, stdout="DRY_RUN", stderr="", oom=False, cmd=cmd, env=env)

        # OOM loop: retry by halving batch sizes in Visual cfg (MVP).
        cur_cfg_path = tmp_visual_cfg
        last: Optional[RunResult] = None
        for attempt in range(int(knobs.oom_retries)):
            if attempt > 0:
                time.sleep(float(knobs.backoff_sec))
            last = run_subprocess(cmd, env=env, cwd=dp_root)
            if last.ok:
                return last
            if not last.oom:
                return last

            # OOM: halve key batch sizes, rewrite cfg and retry.
            try:
                import yaml as _yaml  # type: ignore

                with open(cur_cfg_path, "r", encoding="utf-8") as f:
                    c = _yaml.safe_load(f) or {}
                if not isinstance(c, dict):
                    return last
                core_flags = c.get("core_providers") or {}
                if not isinstance(core_flags, dict):
                    core_flags = {}
                for k in ("core_clip", "core_depth_midas", "core_optical_flow", "core_object_detections"):
                    if not bool(core_flags.get(k)):
                        continue
                    node = c.get(k)
                    if not isinstance(node, dict):
                        continue
                    bs = node.get("batch_size")
                    if isinstance(bs, int):
                        node["batch_size"] = _halve_or_one(bs)
                        c[k] = node
                new_path = write_temp_yaml(c, prefix="vp_sched_retry_", suffix=".yaml")
                cur_cfg_path = new_path
                tmp_cfg_paths.append(new_path)
                # Replace in cmd
                idx = cmd.index("--visual-cfg-path")
                cmd[idx + 1] = os.path.abspath(cur_cfg_path)
            except Exception:
                return last

        return last or RunResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="scheduler_internal_error: no attempts executed",
            oom=False,
            cmd=cmd,
            env=env,
        )
    finally:
        # Best-effort cleanup of temp cfg files.
        for p in tmp_cfg_paths:
            try:
                os.remove(p)
            except Exception:
                pass


