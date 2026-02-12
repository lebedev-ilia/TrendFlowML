#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import defaultdict
import math
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _p95(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    if n == 1:
        return float(ys[0])
    # Linear interpolation between closest ranks
    pos = 0.95 * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    w = pos - float(lo)
    return float((1.0 - w) * ys[lo] + w * ys[hi])


def _mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(float(x) for x in xs) / float(len(xs)))


def _extract_gpu0_used_mb(gpu_mem_mb: Any) -> Optional[float]:
    d = _as_dict(gpu_mem_mb)
    v = d.get("gpu0_used_mb")
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="import_benchmarks_results_to_db",
        description="Import DataProcessor/benchmarks/out/*/results.jsonl into Postgres benchmark registry (benchmark_costs_v1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db-dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@localhost:5432/db")
    ap.add_argument("--db-table", default="benchmark_costs_v1")

    ap.add_argument("--results-jsonl", action="append", default=[], help="Path to results.jsonl (repeatable)")
    ap.add_argument("--bench-out-dir", default=None, help="Path to benchmarks/out/<run_dir> (will read results.jsonl inside)")
    ap.add_argument("--bench", default="model_level_triton", help="Filter: keep only rows where row.bench == value")
    ap.add_argument("--min-samples", type=int, default=5, help="Minimum ok samples per group to write aggregated row")
    ap.add_argument("--keep-batches", default="1,8", help="Comma-separated batch sizes to import (scheduler-facing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore
    except Exception as e:
        raise RuntimeError("This importer requires psycopg2 (or psycopg2-binary).") from e

    paths: List[str] = []
    for p in (args.results_jsonl or []):
        if p:
            paths.append(os.path.abspath(p))
    if args.bench_out_dir:
        cand = os.path.join(os.path.abspath(args.bench_out_dir), "results.jsonl")
        paths.append(cand)
    # de-dupe
    seen = set()
    uniq: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    paths = uniq

    if not paths:
        raise SystemExit("No inputs. Provide --results-jsonl or --bench-out-dir.")

    keep_batches: set[int] = set()
    for x in str(args.keep_batches).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            keep_batches.add(int(x))
        except Exception:
            continue
    if not keep_batches:
        keep_batches = {1}

    # Group ok samples by registry identity + model_batch_size knob.
    # key -> samples
    Key = Tuple[
        str,  # component_id
        str,  # component_part
        str,  # owner
        str,  # stage
        str,  # unit
        str,  # runtime
        Optional[str],  # model_signature
        Optional[str],  # model_branch
        str,  # device_profile_json
        str,  # input_bucket_json
        str,  # knobs_json
        str,  # artifact_uri
        str,  # producer_version
        str,  # git_commit
        bool,  # git_dirty
    ]
    lat: DefaultDict[Key, List[float]] = defaultdict(list)
    rss: DefaultDict[Key, List[float]] = defaultdict(list)
    gpu0: DefaultDict[Key, List[Optional[float]]] = defaultdict(list)
    base_rss: DefaultDict[Key, List[Optional[float]]] = defaultdict(list)
    base_gpu0: DefaultDict[Key, List[Optional[float]]] = defaultdict(list)

    for path in paths:
        if not os.path.isfile(path):
            continue
        for row in _iter_jsonl(path):
            if args.bench is not None and row.get("bench") != args.bench:
                continue
            if row.get("status") != "ok":
                continue

            component_id = _as_str(row.get("component_id"))
            unit = _as_str(row.get("unit"))
            runtime = _as_str(row.get("runtime"))
            owner = _as_str(row.get("owner")) or "dataprocessor"
            stage = _as_str(row.get("stage")) or "baseline"
            component_part = _as_str(row.get("component_part")) or "whole"
            model_signature = _as_str(row.get("model_signature"))
            model_branch = _as_str(row.get("model_branch"))
            batch = _as_int(row.get("batch"))
            latency_ms = _as_float(row.get("latency_ms"))
            if not component_id or not unit or not runtime or batch is None or latency_ms is None:
                continue
            if batch not in keep_batches:
                continue

            device_profile = _as_dict(row.get("device_profile"))
            input_bucket = _as_dict(row.get("input_bucket"))
            knobs = _as_dict(row.get("knobs"))
            knobs = dict(knobs)
            knobs["model_batch_size"] = int(batch)

            artifact_uri = _as_str(row.get("artifact_uri")) or ("file://" + os.path.abspath(path))
            producer_version = _as_str(row.get("producer_version")) or "unknown"
            git_commit = _as_str(row.get("git_commit")) or "unknown"
            git_dirty = bool(row.get("git_dirty") or False)

            key: Key = (
                component_id,
                component_part,
                owner,
                stage,
                unit,
                runtime,
                model_signature,
                model_branch,
                json.dumps(device_profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                json.dumps(input_bucket, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                json.dumps(knobs, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                artifact_uri,
                producer_version,
                git_commit,
                git_dirty,
            )

            lat[key].append(float(latency_ms))

            r = _as_float(row.get("rss_mb"))
            if r is not None:
                rss[key].append(float(r))

            g0 = _extract_gpu0_used_mb(row.get("gpu_mem_mb"))
            gpu0[key].append(g0)

            br = _as_float(row.get("base_rss_mb"))
            base_rss[key].append(br)

            bg0 = _extract_gpu0_used_mb(row.get("base_gpu_mem_mb"))
            base_gpu0[key].append(bg0)

    # Build aggregated rows
    inserts: List[Dict[str, Any]] = []
    for key, samples in lat.items():
        if len(samples) < int(args.min_samples):
            continue
        (
            component_id,
            component_part,
            owner,
            stage,
            unit,
            runtime,
            model_signature,
            model_branch,
            device_profile_json,
            input_bucket_json,
            knobs_json,
            artifact_uri,
            producer_version,
            git_commit,
            git_dirty,
        ) = key

        knobs_obj = json.loads(knobs_json)
        bs = int(knobs_obj.get("model_batch_size") or 1)

        mean_batch = _mean(samples)
        p95_batch = _p95(samples)
        mean_unit = float(mean_batch) / float(max(1, bs))
        p95_unit = float(p95_batch) / float(max(1, bs))

        cpu_peak = float(max(rss.get(key) or [float("nan")]))

        # Best-effort VRAM (device-level, not tritonserver). Still useful for rough sizing.
        g0_vals = [x for x in (gpu0.get(key) or []) if x is not None]
        g0_peak = float(max(g0_vals)) if g0_vals else float("nan")
        g0_before_vals = [x for x in (base_gpu0.get(key) or []) if x is not None]
        g0_before = float(max(g0_before_vals)) if g0_before_vals else float("nan")
        g0_after = float(g0_vals[-1]) if g0_vals else float("nan")
        g0_delta = float(g0_peak - g0_before) if (math.isfinite(g0_peak) and math.isfinite(g0_before)) else float("nan")
        g0_drift = float(g0_after - g0_before) if (math.isfinite(g0_after) and math.isfinite(g0_before)) else float("nan")

        metrics: Dict[str, Any] = {
            "status": "ok",
            "samples": int(len(samples)),
            "model_batch_size": int(bs),
            "latency_ms_mean_stable_per_batch": float(mean_batch),
            "latency_ms_p95_per_batch": float(p95_batch),
            "latency_ms_mean_stable_per_unit": float(mean_unit),
            "latency_ms_p95": float(p95_unit),
            "cpu_rss_peak_mb": float(cpu_peak),
            # legacy keys for scheduler (best-effort mapping)
            "vram_triton_peak_mb": float(g0_peak) if math.isfinite(g0_peak) else None,
            "vram_triton_delta_run_mb": float(g0_delta) if math.isfinite(g0_delta) else None,
            "vram_triton_drift_mb": float(g0_drift) if math.isfinite(g0_drift) else None,
            "restart_recommended": False,
            "restart_reason": None,
            # explicit device-level fields
            "vram_device_before_mb": float(g0_before) if math.isfinite(g0_before) else None,
            "vram_device_peak_mb": float(g0_peak) if math.isfinite(g0_peak) else None,
            "vram_device_after_mb": float(g0_after) if math.isfinite(g0_after) else None,
            "vram_device_delta_run_mb": float(g0_delta) if math.isfinite(g0_delta) else None,
            "vram_device_drift_mb": float(g0_drift) if math.isfinite(g0_drift) else None,
        }

        inserts.append(
            {
                "component_id": component_id,
                "component_part": component_part,
                "owner": owner,
                "stage": stage,
                "unit": unit,
                "runtime": runtime,
                "model_signature": model_signature,
                "model_branch": model_branch,
                "device_profile_json": device_profile_json,
                "input_bucket_json": input_bucket_json,
                "knobs_json": knobs_json,
                "producer_version": producer_version,
                "git_commit": git_commit,
                "git_dirty": bool(git_dirty),
                "schema_version": "benchmark_costs_v1",
                "metrics_json": json.dumps(metrics, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                "artifact_uri": artifact_uri,
            }
        )

    if args.dry_run:
        print(f"[dry-run] files={len(paths)} aggregated_rows={len(inserts)}")
        return 0

    # Write to Postgres (append-only + close previous active with same identity payload).
    with psycopg2.connect(args.db_dsn) as conn:
        with conn.cursor() as cur:
            for r in inserts:
                close_sql = f"""
                UPDATE {args.db_table}
                SET valid_to = NOW()
                WHERE valid_to IS NULL
                  AND component_id = %s
                  AND component_part = %s
                  AND owner = %s
                  AND stage IS NOT DISTINCT FROM %s
                  AND unit = %s
                  AND runtime = %s
                  AND model_signature IS NOT DISTINCT FROM %s
                  AND model_branch IS NOT DISTINCT FROM %s
                  AND device_profile = %s::jsonb
                  AND input_bucket = %s::jsonb
                  AND knobs = %s::jsonb
                """
                cur.execute(
                    close_sql,
                    (
                        r["component_id"],
                        r["component_part"],
                        r["owner"],
                        r["stage"],
                        r["unit"],
                        r["runtime"],
                        r["model_signature"],
                        r["model_branch"],
                        r["device_profile_json"],
                        r["input_bucket_json"],
                        r["knobs_json"],
                    ),
                )

                ins_sql = f"""
                INSERT INTO {args.db_table} (
                  id, component_id, component_part, owner, stage, unit, runtime,
                  model_signature, model_branch,
                  input_bucket, knobs, device_profile,
                  producer_version, git_commit, git_dirty, schema_version,
                  metrics, artifact_uri, created_at, valid_from, valid_to
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s,
                  %s, %s,
                  %s::jsonb, %s::jsonb, %s::jsonb,
                  %s, %s, %s, %s,
                  %s::jsonb, %s, NOW(), NOW(), NULL
                )
                """
                cur.execute(
                    ins_sql,
                    (
                        str(uuid.uuid4()),
                        r["component_id"],
                        r["component_part"],
                        r["owner"],
                        r["stage"],
                        r["unit"],
                        r["runtime"],
                        r["model_signature"],
                        r["model_branch"],
                        r["input_bucket_json"],
                        r["knobs_json"],
                        r["device_profile_json"],
                        r["producer_version"],
                        r["git_commit"],
                        bool(r["git_dirty"]),
                        r["schema_version"],
                        r["metrics_json"],
                        r["artifact_uri"],
                    ),
                )

    print(f"Imported aggregated rows: {len(inserts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


