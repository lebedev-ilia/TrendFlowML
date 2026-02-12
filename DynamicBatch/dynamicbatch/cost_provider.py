from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from .resource_costs import UnitCost, load_resource_costs_dir


@dataclass(frozen=True)
class CostQuery:
    """
    MVP query shape. Will expand to include:
    - device_profile
    - input_bucket
    - knobs/preset
    - stage (baseline/v1/v2)
    - component_part (whole/substep)
    """

    component_id: str
    component_part: str = "whole"  # whole | substep:<name>
    runtime: Optional[str] = None
    stage: Optional[str] = None
    model_signature: Optional[str] = None
    prefer_branch: Optional[str] = None
    # Best-effort: JSONB subset constraints (for DB queries). For file provider these are ignored.
    device_profile_subset: Optional[Dict] = None
    input_bucket_subset: Optional[Dict] = None


class CostProvider(Protocol):
    def list_costs(self) -> List[UnitCost]: ...

    def get_cost(self, q: CostQuery) -> Optional[UnitCost]: ...


class FileCostProvider:
    """
    Seed provider: reads DataProcessor/docs/models_docs/resource_costs/*.json.
    """

    def __init__(self, resource_costs_dir: str):
        self.resource_costs_dir = str(resource_costs_dir)
        self._cache: Optional[List[UnitCost]] = None

    def list_costs(self) -> List[UnitCost]:
        if self._cache is None:
            self._cache = load_resource_costs_dir(self.resource_costs_dir)
        return list(self._cache)

    def get_cost(self, q: CostQuery) -> Optional[UnitCost]:
        rows = [c for c in self.list_costs() if c.component == q.component_id]
        if not rows:
            return None
        rows.sort(key=lambda r: (str(r.model_branch), str(r.unit)))
        if q.prefer_branch is not None:
            for r in rows:
                if str(r.model_branch) == str(q.prefer_branch):
                    return r
        return rows[0]


class DbCostProvider:
    """
    Stub for DB-backed registry (Postgres).

    MVP behavior:
    - implement interface, but raise a clear error until DB integration is wired.
    - scheduler can still run via FileCostProvider.
    """

    def __init__(self, dsn: str, table: str = "benchmark_costs_v1"):
        self.dsn = str(dsn)
        self.table = str(table)

    def list_costs(self) -> List[UnitCost]:
        raise NotImplementedError("Use get_cost() with a specific query in MVP.")

    def get_cost(self, q: CostQuery) -> Optional[UnitCost]:
        try:
            import psycopg2  # type: ignore
            import psycopg2.extras  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "DbCostProvider requires psycopg2. Install it in the environment running DynamicBatch "
                "or use --costs-provider=file."
            ) from e

        # MVP: pick active row (valid_to IS NULL) with best match.
        # JSONB subset constraints are optional; when provided we use @> operator.
        where = ["valid_to IS NULL", "component_id = %(component_id)s", "component_part = %(component_part)s"]
        params: Dict[str, object] = {"component_id": q.component_id, "component_part": q.component_part}

        if q.runtime:
            where.append("runtime = %(runtime)s")
            params["runtime"] = q.runtime
        if q.stage:
            where.append("(stage = %(stage)s OR stage IS NULL)")
            params["stage"] = q.stage

        # model_signature: use IS NOT DISTINCT FROM to allow NULL matching.
        where.append("model_signature IS NOT DISTINCT FROM %(model_signature)s")
        params["model_signature"] = q.model_signature

        if q.device_profile_subset:
            where.append("device_profile @> %(device_profile_subset)s::jsonb")
            params["device_profile_subset"] = psycopg2.extras.Json(q.device_profile_subset)
        if q.input_bucket_subset:
            where.append("input_bucket @> %(input_bucket_subset)s::jsonb")
            params["input_bucket_subset"] = psycopg2.extras.Json(q.input_bucket_subset)

        if q.prefer_branch:
            where.append("model_branch = %(model_branch)s")
            params["model_branch"] = str(q.prefer_branch)

        sql = f"""
        SELECT
          component_id,
          unit,
          COALESCE(model_branch, '') AS model_branch,
          metrics
        FROM {self.table}
        WHERE {' AND '.join(where)}
        ORDER BY valid_from DESC
        LIMIT 1
        """

        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()

        if not row:
            return None

        metrics = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, dict):
            metrics = {}
        return UnitCost(
            component=str(row.get("component_id") or ""),
            unit=str(row.get("unit") or ""),
            model_branch=str(row.get("model_branch") or ""),
            latency_ms_per_unit=float(metrics.get("latency_ms_mean_stable_per_unit")) if metrics.get("latency_ms_mean_stable_per_unit") is not None else None,
            cpu_rss_peak_mb=float(metrics.get("cpu_rss_peak_mb")) if metrics.get("cpu_rss_peak_mb") is not None else None,
            vram_triton_peak_mb=float(metrics.get("vram_triton_peak_mb")) if metrics.get("vram_triton_peak_mb") is not None else None,
            vram_triton_delta_run_mb=float(metrics.get("vram_triton_delta_run_mb")) if metrics.get("vram_triton_delta_run_mb") is not None else None,
        )


