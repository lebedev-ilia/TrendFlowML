from __future__ import annotations

import argparse
import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import List, Optional, Dict, Any

from .plan import SchedulerKnobs, run_dataprocessor_job
from .cost_provider import CostProvider, CostQuery, DbCostProvider, FileCostProvider
from .resource_costs import gpu_mem_per_task_mb
from .resource_monitor import ResourceMonitor
from .report_index_html import RunRow, write_index_html
from .state_level1 import Level1StateStore
from .system_probe import probe_cpu_mem_mb, probe_gpu_mem_mb


def _default_repo_root() -> str:
    # DynamicBatch/ is at repo root; if executed from within package, be robust.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dynamicbatch",
        description="DynamicBatch scheduler (MVP): batch level 1 + resource-aware batch_size overrides + OOM retry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-path", action="append", default=[], help="Local video path (repeatable)")
    p.add_argument("--input-json", type=str, default=None, help="Path to JSON list of local video paths")

    p.add_argument("--repo-root", type=str, default=_default_repo_root(), help="TrendFlowML repo root")
    p.add_argument(
        "--rs-base",
        type=str,
        default=None,
        help="Base result_store for runs (default: DataProcessor/VisualProcessor/result_store)",
    )
    p.add_argument(
        "--visual-cfg-template",
        type=str,
        default=None,
        help="Base VisualProcessor config.yaml to override (default: try configs/visual_triton_baseline_gpu_local.yaml, else DataProcessor/VisualProcessor/config.yaml)",
    )
    p.add_argument("--dag-stage", type=str, default="baseline", choices=["baseline", "v1", "v2"])
    p.add_argument("--platform-id", type=str, default="local", help="platform_id to pass to DataProcessor runs")
    p.add_argument("--profile-path", type=str, default=None, help="Optional DataProcessor analysis profile YAML")
    p.add_argument("--dp-models-root", type=str, default=None, help="DP_MODELS_ROOT to pass to workers")
    p.add_argument("--triton-http-url", type=str, default=None, help="TRITON_HTTP_URL to pass to workers")
    p.add_argument("--dp-python", type=str, default="python3", help="Python executable to run DataProcessor/main.py (venv-aware)")

    p.add_argument("--costs-provider", type=str, default="file", choices=["file", "db"], help="Where scheduler reads benchmark costs from")
    p.add_argument("--db-dsn", type=str, default=None, help="Postgres DSN for benchmark registry (required when --costs-provider=db)")
    p.add_argument("--db-table", type=str, default="benchmark_costs_v1", help="Benchmark registry table name (Postgres)")

    p.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Max concurrent DataProcessor runs. 0 means auto (scheduler picks a safe value).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned commands, do not execute")

    p.add_argument("--oom-retries", type=int, default=3)
    p.add_argument("--backoff-sec", type=float, default=2.0)
    p.add_argument("--sim-gpu-free-mb", type=int, default=None, help="Simulate available free VRAM (MB) for planning batch sizes")
    p.add_argument("--headroom-ratio", type=float, default=0.25, help="Scheduler headroom ratio (fraction of free VRAM reserved)")
    p.add_argument("--min-headroom-mb", type=int, default=1024, help="Scheduler minimum VRAM headroom (MB)")
    p.add_argument("--post-validate", action="store_true", help="After each run: validate all NPZ artifacts under run_rs_path")
    p.add_argument("--post-html", action="store_true", help="After each run: generate HTML quality report for new heads (domain/franchise/ocr)")
    p.add_argument("--post-out-dir", type=str, default=None, help="Output dir for post-step artifacts (default: DataProcessor/docs/baseline/out_dynamicbatch)")

    # AudioProcessor knobs (scheduler-controlled L2/L3)
    p.add_argument("--audio-segment-parallelism", type=int, default=None, help="AudioProcessor: concurrent segment workers (if supported)")
    p.add_argument("--audio-max-inflight", type=int, default=None, help="AudioProcessor: max in-flight segment tasks (safety cap)")
    p.add_argument("--audio-clap-batch-size", type=int, default=None, help="AudioProcessor: CLAP micro-batch size (may increase VRAM)")
    return p.parse_args(argv)


def _load_inputs(args: argparse.Namespace) -> List[str]:
    vids: List[str] = []
    vids.extend([str(x) for x in (args.video_path or []) if x])
    if args.input_json:
        import json

        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            vids.extend([str(x) for x in payload if isinstance(x, str) and x])
    # normalize + de-dupe preserving order
    seen = set()
    out: List[str] = []
    for v in vids:
        vv = os.path.abspath(v)
        if vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def _safe_load_json(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def _make_cost_provider(*, args: argparse.Namespace, costs_dir: str) -> CostProvider:
    if str(args.costs_provider).lower() == "db":
        if not args.db_dsn:
            raise RuntimeError("--costs-provider=db requires --db-dsn")
        return DbCostProvider(dsn=str(args.db_dsn), table=str(args.db_table))
    return FileCostProvider(resource_costs_dir=str(costs_dir))


def _auto_max_parallel(
    *,
    free_vram_mb: Optional[int],
    cost_provider: CostProvider,
    headroom_ratio: float,
    min_headroom_mb: int,
    n_videos: int,
) -> int:
    """
    Conservative auto-parallel choice based on VRAM budget.
    We approximate per-video VRAM peak as the max of major GPU components' per-task deltas.
    """
    if not free_vram_mb or int(free_vram_mb) <= 0:
        return 1
    free = int(free_vram_mb)
    headroom = max(int(min_headroom_mb), int(round(float(free) * float(headroom_ratio))))
    budget = max(0, free - headroom)
    if budget <= 0:
        return 1

    clip_cost = cost_provider.get_cost(CostQuery(component_id="core_clip.clip_image", prefer_branch="336"))
    midas_cost = cost_provider.get_cost(CostQuery(component_id="core_depth_midas.midas", prefer_branch="384"))
    # RAFT costs may be missing in registry; keep a conservative fallback.
    raft_mb = 512
    clip_mb = gpu_mem_per_task_mb(clip_cost, default_mb=64)
    midas_mb = gpu_mem_per_task_mb(midas_cost, default_mb=128)

    per_video_peak_mb = max(int(clip_mb), int(midas_mb), int(raft_mb))
    if per_video_peak_mb <= 0:
        return 1
    k = max(1, budget // per_video_peak_mb)
    return int(max(1, min(int(k), int(n_videos))))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = os.path.abspath(args.repo_root)
    dp_root = os.path.join(repo_root, "DataProcessor")

    rs_base = args.rs_base or os.path.join(dp_root, "VisualProcessor", "result_store")
    if args.visual_cfg_template:
        visual_cfg = str(args.visual_cfg_template)
    else:
        preferred = os.path.join(repo_root, "configs", "visual_triton_baseline_gpu_local.yaml")
        visual_cfg = preferred if os.path.exists(preferred) else os.path.join(dp_root, "VisualProcessor", "config.yaml")

    # If profile not provided explicitly, try a repo-level default profile (enables audio tier-0).
    if not args.profile_path:
        prof = os.path.join(repo_root, "configs", "profile_triton_baseline_gpu_local.yaml")
        if os.path.exists(prof):
            args.profile_path = prof

    costs_dir = os.path.join(dp_root, "docs", "models_docs", "resource_costs")
    state_dir = os.path.join(repo_root, "DynamicBatch", "_state")
    store = Level1StateStore(state_dir=state_dir)
    store.init()

    videos = _load_inputs(args)
    if not videos:
        print("No inputs. Provide --video-path or --input-json.", file=sys.stderr)
        return 2

    # One-time probes for logs/visibility (MVP).
    cpu = probe_cpu_mem_mb()
    gpu = probe_gpu_mem_mb()
    if args.sim_gpu_free_mb is not None and gpu is not None:
        # Keep total/used as-is for reporting; override only free for planner.
        gpu = type(gpu)(total_mb=int(gpu.total_mb), used_mb=int(gpu.used_mb), free_mb=max(0, int(args.sim_gpu_free_mb)))

    # Auto-pick max_parallel if requested (0).
    selected_max_parallel = int(args.max_parallel) if args.max_parallel is not None else 0
    if int(selected_max_parallel) <= 0:
        try:
            cp = _make_cost_provider(args=args, costs_dir=costs_dir)
            selected_max_parallel = _auto_max_parallel(
                free_vram_mb=(int(gpu.free_mb) if gpu is not None else None),
                cost_provider=cp,
                headroom_ratio=float(args.headroom_ratio),
                min_headroom_mb=int(args.min_headroom_mb),
                n_videos=len(videos),
            )
        except Exception:
            selected_max_parallel = 1
    selected_max_parallel = int(max(1, selected_max_parallel))

    store.emit_event(
        {
            "event": "scheduler_start",
            "cpu_mem_mb": asdict(cpu) if cpu else None,
            "gpu_mem_mb": asdict(gpu) if gpu else None,
            "n_videos": len(videos),
            "max_parallel_selected": int(selected_max_parallel),
        }
    )
    if args.dry_run:
        print(f"[DynamicBatch] dry-run: n_videos={len(videos)} max_parallel={int(selected_max_parallel)}")
        if cpu:
            print(f"[DynamicBatch] cpu_mem_mb: total={cpu.total_mb} used={cpu.used_mb} free={cpu.free_mb}")
        if gpu:
            print(f"[DynamicBatch] gpu_mem_mb: total={gpu.total_mb} used={gpu.used_mb} free={gpu.free_mb}")

    knobs = SchedulerKnobs(
        max_parallel=int(selected_max_parallel),
        backoff_sec=float(args.backoff_sec),
        oom_retries=int(args.oom_retries),
        headroom_ratio=float(args.headroom_ratio),
        min_headroom_mb=int(args.min_headroom_mb),
    )

    # MVP execution: concurrent jobs with a hard max_parallel.
    # NOTE: deeper scheduling (VRAM gating, mixed workloads) will be layered on top.
    rc = 0
    started_at = time.time()
    monitor: Optional[ResourceMonitor] = None
    if not args.dry_run:
        monitor = ResourceMonitor(interval_sec=0.25)
        monitor.start()

    run_rows: List[RunRow] = []
    with ThreadPoolExecutor(max_workers=knobs.max_parallel) as ex:
        futs = {}
        for v in videos:
            base = os.path.splitext(os.path.basename(v))[0]
            run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + os.urandom(3).hex()
            run_key = f"{base}__{run_id}"
            run_rows.append(
                RunRow(
                    run_key=run_key,
                    video_path=str(v),
                    video_id=str(base),
                    run_id=str(run_id),
                    status="queued",
                    run_rs_path=os.path.join(os.path.abspath(rs_base), str(args.platform_id), str(base), str(run_id)),
                )
            )
            store.update_run(run_key, {"status": "queued", "video_path": v})
            store.emit_event({"event": "job_queued", "run_key": run_key, "video_path": v})
            fut = ex.submit(
                run_dataprocessor_job,
                repo_root=repo_root,
                video_path=v,
                rs_base=rs_base,
                visual_cfg_path=visual_cfg,
                dag_stage=args.dag_stage,
                profile_path=args.profile_path,
                dp_models_root=args.dp_models_root,
                triton_http_url=args.triton_http_url,
                dp_python=str(args.dp_python),
                knobs=knobs,
                costs_dir=costs_dir,
                costs_provider=str(args.costs_provider),
                db_dsn=str(args.db_dsn) if args.db_dsn else None,
                db_table=str(args.db_table),
                dry_run=bool(args.dry_run),
                platform_id=str(args.platform_id),
                video_id=str(base),
                run_id=str(run_id),
                sim_free_vram_mb=int(args.sim_gpu_free_mb) if args.sim_gpu_free_mb is not None else None,
                audio_segment_parallelism=(int(args.audio_segment_parallelism) if args.audio_segment_parallelism is not None else None),
                audio_max_inflight=(int(args.audio_max_inflight) if args.audio_max_inflight is not None else None),
                audio_clap_batch_size=(int(args.audio_clap_batch_size) if args.audio_clap_batch_size is not None else None),
            )
            futs[fut] = (run_key, v, base, run_id)
            store.update_run(run_key, {"status": "running"})
            store.emit_event({"event": "job_started", "run_key": run_key})
            for rr in run_rows:
                if rr.run_key == run_key:
                    rr.status = "running"
                    break

        for fut in as_completed(list(futs.keys())):
            run_key, v, base, run_id = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                store.update_run(run_key, {"status": "error", "error": str(e)})
                store.emit_event({"event": "job_failed", "run_key": run_key, "error": str(e)})
                rc = 1
                continue

            if args.dry_run:
                env_str = " ".join([f"{k}={v}" for k, v in (res.env or {}).items()])
                if env_str:
                    print(env_str)
                print(" ".join(res.cmd))

            if res.ok:
                store.update_run(run_key, {"status": "success", "returncode": res.returncode})
                store.emit_event({"event": "job_finished", "run_key": run_key, "status": "success"})
                for rr in run_rows:
                    if rr.run_key == run_key:
                        rr.status = "success"
                        rr.returncode = int(res.returncode)
                        rr.oom = bool(res.oom)
                        break
                # Scheduler runtime report (plan vs fact): written under run_rs_path by processors.
                if not args.dry_run:
                    try:
                        run_rs_path = os.path.join(os.path.abspath(rs_base), str(args.platform_id), str(base), str(run_id))
                        rep_path = os.path.join(run_rs_path, "_reports", "scheduler_runtime_report.json")
                        rep = _safe_load_json(rep_path)
                        if rep:
                            store.emit_event(
                                {
                                    "event": "runtime_report",
                                    "run_key": run_key,
                                    "report_path": rep_path,
                                    "schema_version": rep.get("schema_version"),
                                    "created_at": rep.get("created_at"),
                                    "per_processor": rep.get("per_processor"),
                                    "scheduler_knobs": rep.get("scheduler_knobs"),
                                }
                            )
                    except Exception as e:
                        store.emit_event({"event": "runtime_report_failed", "run_key": run_key, "error": str(e)})
                # Post-step: validate + HTML reports (best-effort).
                if (args.post_validate or args.post_html) and (not args.dry_run):
                    try:
                        out_base = args.post_out_dir or os.path.join(dp_root, "docs", "baseline", "out_dynamicbatch")
                        run_rs_path = os.path.join(os.path.abspath(rs_base), str(args.platform_id), str(base), str(run_id))
                        os.makedirs(out_base, exist_ok=True)
                        out_dir = os.path.join(os.path.abspath(out_base), str(base), str(run_id))
                        os.makedirs(out_dir, exist_ok=True)
                        qa_script = os.path.join(dp_root, "scripts", "baseline", "run_quality_suite_for_run.py")
                        qa_cmd = [str(args.dp_python), qa_script, "--run-rs-path", run_rs_path, "--out-dir", out_dir]
                        if args.post_validate:
                            qa_cmd.append("--validate")
                        if args.post_html:
                            qa_cmd.append("--html")
                        _ = run_dataprocessor_job  # keep linter happy about imports (no-op)
                        import subprocess as _sp
                        _sp.run(qa_cmd, check=False, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
                        store.emit_event({"event": "post_done", "run_key": run_key, "out_dir": out_dir})
                        for rr in run_rows:
                            if rr.run_key == run_key:
                                rr.out_dir = str(out_dir)
                                break
                    except Exception as e:
                        store.emit_event({"event": "post_failed", "run_key": run_key, "error": str(e)})
            else:
                store.update_run(
                    run_key,
                    {
                        "status": "error",
                        "returncode": res.returncode,
                        "oom": bool(res.oom),
                    },
                )
                store.emit_event({"event": "job_finished", "run_key": run_key, "status": "error", "oom": bool(res.oom)})
                for rr in run_rows:
                    if rr.run_key == run_key:
                        rr.status = "error"
                        rr.returncode = int(res.returncode)
                        rr.oom = bool(res.oom)
                        break
                rc = 1

    # Stop monitor and report peaks.
    peaks: Optional[Dict[str, Any]] = None
    if monitor is not None:
        monitor.stop(timeout_sec=2.0)
        peaks = monitor.peaks.to_dict()
        store.emit_event({"event": "resource_peaks", "peaks": peaks})

    # Build a run-level HTML index for manual QA (per-video quality.html already exists).
    if (args.post_validate or args.post_html) and (not args.dry_run):
        try:
            out_base = args.post_out_dir or os.path.join(dp_root, "docs", "baseline", "out_dynamicbatch")
            idx_path = write_index_html(out_base=os.path.abspath(out_base), rows=run_rows, scheduler_peaks=peaks)
            store.emit_event({"event": "post_index_done", "index_path": idx_path})
        except Exception as e:
            store.emit_event({"event": "post_index_failed", "error": str(e)})

    store.emit_event({"event": "scheduler_finished", "duration_sec": float(time.time() - started_at), "rc": rc, "max_parallel_selected": int(selected_max_parallel)})
    return int(rc)


