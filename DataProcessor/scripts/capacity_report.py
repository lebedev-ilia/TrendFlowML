from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _coerce_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _percentile(sorted_vals: List[int], p: float) -> Optional[int]:
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    # Nearest-rank (simple, stable)
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    k = max(0, min(len(sorted_vals) - 1, k))
    return sorted_vals[k]


@dataclass
class RunDurations:
    run_dir: str
    platform_id: str
    video_id: str
    run_id: str
    total_ms: int
    cpu_ms: int
    cuda_ms: int
    by_component_ms: Dict[str, int]
    by_component_status: Dict[str, str]


def _iter_run_dirs(results_root: str, *, max_runs: Optional[int]) -> Iterable[str]:
    # dp_results/<platform>/<video_id>/<run_id>/manifest.json
    # dp_results/youtube/test_x/test_x/manifest.json also exists historically; we look for manifest.json and treat its dir as run_dir.
    found = 0
    for root, _dirs, files in os.walk(results_root):
        if "manifest.json" in files:
            yield root
            found += 1
            if max_runs is not None and found >= max_runs:
                return


def _parse_manifest(run_dir: str) -> Optional[RunDurations]:
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    m = _load_json(manifest_path)
    run = m.get("run") if isinstance(m.get("run"), dict) else {}
    comps_raw = m.get("components") if isinstance(m.get("components"), list) else []
    comps: List[Dict[str, Any]] = [c for c in comps_raw if isinstance(c, dict)]

    platform_id = _coerce_str(run.get("platform_id") or "")
    video_id = _coerce_str(run.get("video_id") or "")
    run_id = _coerce_str(run.get("run_id") or "")

    total_ms = 0
    cpu_ms = 0
    cuda_ms = 0
    by_component_ms: Dict[str, int] = {}
    by_component_status: Dict[str, str] = {}

    for c in comps:
        name = _coerce_str(c.get("name")).strip() or "unknown"
        status = _coerce_str(c.get("status")).strip() or "unknown"
        by_component_status[name] = status

        d = _coerce_int(c.get("duration_ms"))
        if not isinstance(d, int) or d < 0:
            continue
        by_component_ms[name] = d
        total_ms += d

        dev = _coerce_str(c.get("device_used")).strip().lower()
        if dev in ("cuda", "gpu"):
            cuda_ms += d
        else:
            cpu_ms += d

    return RunDurations(
        run_dir=os.path.abspath(run_dir),
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        total_ms=total_ms,
        cpu_ms=cpu_ms,
        cuda_ms=cuda_ms,
        by_component_ms=by_component_ms,
        by_component_status=by_component_status,
    )


def _fmt_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "n/a"
    sec = ms / 1000.0
    if sec < 120:
        return f"{sec:.1f}s"
    return f"{sec/60.0:.1f}m"


def _summarize_int_series(vals: List[int]) -> Dict[str, Any]:
    vals2 = sorted(int(v) for v in vals if isinstance(v, int))
    return {
        "n": len(vals2),
        "min_ms": vals2[0] if vals2 else None,
        "p50_ms": _percentile(vals2, 50),
        "p90_ms": _percentile(vals2, 90),
        "p95_ms": _percentile(vals2, 95),
        "p99_ms": _percentile(vals2, 99),
        "max_ms": vals2[-1] if vals2 else None,
        "sum_ms": sum(vals2) if vals2 else 0,
    }


def _estimate_throughput(summary: Dict[str, Any], *, gpus: Optional[int]) -> Dict[str, Any]:
    # Very rough: assume GPU-bound if cuda_ms dominates. Use p50_cuda_ms if available.
    cuda = summary.get("cuda_ms") or {}
    cpu = summary.get("cpu_ms") or {}

    p50_cuda = cuda.get("p50_ms")
    p95_cuda = cuda.get("p95_ms")
    sum_cuda_ms = cuda.get("sum_ms", 0) or 0

    # GPU-hours for the dataset represented by this report.
    gpu_hours = float(sum_cuda_ms) / (1000.0 * 60.0 * 60.0) if sum_cuda_ms else 0.0

    est: Dict[str, Any] = {"gpu_hours_total": gpu_hours}
    if gpus and isinstance(p50_cuda, int) and p50_cuda > 0:
        est["videos_per_hour_per_gpu_p50"] = (3600.0 * 1000.0) / float(p50_cuda)
        est["videos_per_hour_cluster_p50"] = est["videos_per_hour_per_gpu_p50"] * float(gpus)
    if gpus and isinstance(p95_cuda, int) and p95_cuda > 0:
        est["videos_per_hour_per_gpu_p95"] = (3600.0 * 1000.0) / float(p95_cuda)
        est["videos_per_hour_cluster_p95"] = est["videos_per_hour_per_gpu_p95"] * float(gpus)
    return est


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Capacity report from dp_results manifests (p50/p90/p95, cpu vs cuda, gpu-hours).")
    p.add_argument("--results-root", default="DataProcessor/dp_results", help="Root directory with dp_results")
    p.add_argument("--platform-id", default="", help="Optional: only runs whose manifest.run.platform_id matches")
    p.add_argument("--max-runs", type=int, default=500, help="Safety cap for scanned runs (default 500)")
    p.add_argument("--out", default="", help="Optional: output JSON path (default: <results-root>/_reports/capacity_report.json)")
    p.add_argument("--gpus", type=int, default=0, help="Optional: GPU count for throughput estimates")
    args = p.parse_args(argv)

    results_root = os.path.abspath(args.results_root)
    want_platform = (args.platform_id or "").strip()
    max_runs = int(args.max_runs) if args.max_runs is not None else None
    gpus = int(args.gpus) if int(args.gpus) > 0 else None

    runs: List[RunDurations] = []
    for run_dir in _iter_run_dirs(results_root, max_runs=max_runs):
        rd = _parse_manifest(run_dir)
        if rd is None:
            continue
        if want_platform and rd.platform_id != want_platform:
            continue
        runs.append(rd)

    total_ms = [r.total_ms for r in runs]
    cpu_ms = [r.cpu_ms for r in runs]
    cuda_ms = [r.cuda_ms for r in runs]

    # Per-component distributions (duration_ms), across all runs where present.
    by_comp: Dict[str, List[int]] = {}
    by_comp_status: Dict[str, Dict[str, int]] = {}
    for r in runs:
        for name, d in r.by_component_ms.items():
            by_comp.setdefault(name, []).append(int(d))
        for name, st in r.by_component_status.items():
            by_comp_status.setdefault(name, {})
            by_comp_status[name][st] = by_comp_status[name].get(st, 0) + 1

    per_component = {}
    for name, vals in sorted(by_comp.items(), key=lambda kv: kv[0]):
        per_component[name] = {
            "duration_ms": _summarize_int_series(vals),
            "status_counts": dict(sorted((by_comp_status.get(name) or {}).items(), key=lambda kv: kv[0])),
        }

    summary = {
        "total_ms": _summarize_int_series(total_ms),
        "cpu_ms": _summarize_int_series(cpu_ms),
        "cuda_ms": _summarize_int_series(cuda_ms),
    }

    payload = {
        "schema_version": "capacity_report_v1",
        "created_at": _utc_iso_now(),
        "results_root": results_root,
        "filters": {"platform_id": want_platform or None, "max_runs": max_runs},
        "runs_count": len(runs),
        "summary": summary,
        "per_component": per_component,
        "estimates": _estimate_throughput(summary, gpus=gpus),
    }

    out_path = os.path.abspath(args.out) if args.out else os.path.join(results_root, "_reports", "capacity_report.json")
    _atomic_write_json(out_path, payload)

    # Human output (concise)
    print(f"runs: {len(runs)}")
    print(f"total p50/p95: {_fmt_ms(summary['total_ms']['p50_ms'])} / {_fmt_ms(summary['total_ms']['p95_ms'])}")
    print(f"cuda  p50/p95: {_fmt_ms(summary['cuda_ms']['p50_ms'])} / {_fmt_ms(summary['cuda_ms']['p95_ms'])}")
    print(f"cpu   p50/p95: {_fmt_ms(summary['cpu_ms']['p50_ms'])} / {_fmt_ms(summary['cpu_ms']['p95_ms'])}")
    if payload["estimates"].get("gpu_hours_total") is not None:
        print(f"gpu_hours_total: {payload['estimates']['gpu_hours_total']:.2f}")
    print(f"out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

