#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from add_targets import compute_targets_from_record
from build_training_table import _extract_features_from_npz, _iter_run_manifests
from utils_bigjson import load_video_records_subset


def _parse_iso8601(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        # handle trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_yaml_minimal(path: str) -> Dict[str, Any]:
    """
    Minimal YAML loader for our feature_spec.yaml without external deps.
    Supports only mappings/lists/scalars in the specific structure we use.
    If you need full YAML, install pyyaml and replace this function.
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # fallback: very small subset parser (enough for our current file)
        # This fallback is intentionally minimal; we encourage installing PyYAML.
        data: Dict[str, Any] = {}
        current_key: Optional[str] = None
        current_list: Optional[List[Dict[str, Any]]] = None
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if not line.startswith(" "):  # top-level key
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v:
                        data[k] = v
                        current_key = None
                        current_list = None
                    else:
                        data[k] = []
                        current_key = k
                        current_list = data[k]
                else:
                    if current_key is None or current_list is None:
                        continue
                    # list item: "  - name: foo"
                    s = line.strip()
                    if s.startswith("- "):
                        item: Dict[str, Any] = {}
                        rest = s[2:]
                        if ":" in rest:
                            kk, vv = rest.split(":", 1)
                            item[kk.strip()] = vv.strip()
                        current_list.append(item)
                    else:
                        # continuation key for last dict
                        if not current_list:
                            continue
                        if ":" in s:
                            kk, vv = s.split(":", 1)
                            current_list[-1][kk.strip()] = vv.strip()
        return data


@dataclass
class DatasetRow:
    keys: Dict[str, Any]
    features: Dict[str, Any]


def _extract_npz_features_for_manifest_component(c: Dict[str, Any]) -> Dict[str, float]:
    name = c.get("name")
    if not isinstance(name, str) or not name:
        return {}
    arts = c.get("artifacts") or []
    npz_path = None
    if isinstance(arts, list):
        for a in arts:
            if isinstance(a, dict) and isinstance(a.get("path"), str) and str(a.get("path")).lower().endswith(".npz"):
                npz_path = a.get("path")
                break
    if npz_path and os.path.exists(npz_path):
        feats, _meta = _extract_features_from_npz(name, npz_path)
        return feats
    return {}


def build_feature_rows(
    rs_base: str,
    *,
    allowed_components: Optional[set[str]] = None,
    required_components: Optional[set[str]] = None,
) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    for manifest_path, manifest in _iter_run_manifests(rs_base):
        run = manifest.get("run") or {}
        platform_id = run.get("platform_id") or ""
        video_id = run.get("video_id") or ""
        run_id = run.get("run_id") or ""
        config_hash = run.get("config_hash") or ""
        sampling_policy_version = run.get("sampling_policy_version") or ""

        keys = {
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": sampling_policy_version,
            "manifest_path": manifest_path,
        }

        feats: Dict[str, Any] = {}
        # temporal from manifest
        feats["manifest_created_at"] = _safe_str(run.get("created_at"))
        feats["analysis_fps"] = _safe_float(run.get("analysis_fps"))
        feats["analysis_width"] = _safe_float(run.get("analysis_width"))
        feats["analysis_height"] = _safe_float(run.get("analysis_height"))

        comps = manifest.get("components") or []
        seen_components: set[str] = set()
        if isinstance(comps, list):
            for c in comps:
                if not isinstance(c, dict):
                    continue
                name = c.get("name")
                if not isinstance(name, str) or not name:
                    continue
                if allowed_components is not None and name not in allowed_components:
                    continue
                seen_components.add(name)

                status = c.get("status")
                val = -1.0
                if status == "ok":
                    val = 1.0
                elif status == "empty":
                    val = 0.0
                feats[f"component_status__{name}"] = val

                feats.update(_extract_npz_features_for_manifest_component(c))

        # Ensure required components always have a status column (missing -> -1).
        if required_components:
            for rc in required_components:
                key = f"component_status__{rc}"
                if key not in feats:
                    feats[key] = -1.0

        rows.append(DatasetRow(keys=keys, features=feats))
    return rows


def add_snapshot_and_targets(
    rows: List[DatasetRow],
    *,
    data_json_path: str,
    require_14_21: bool = True,
) -> Tuple[List[DatasetRow], Dict[str, Any]]:
    video_ids = {str(r.keys.get("video_id") or "") for r in rows if str(r.keys.get("video_id") or "")}
    subset, stats = load_video_records_subset(data_json_path, include_video_ids=video_ids)

    out_rows: List[DatasetRow] = []
    dropped_missing_required = 0
    missing_video_record = 0
    for r in rows:
        vid = str(r.keys.get("video_id") or "")
        rec = subset.get(vid)
        if rec is None:
            missing_video_record += 1
            continue

        meta = rec.get("metadata") or {}
        s0 = rec.get("snapshot_0") or {}

        feats = dict(r.features)

        # snapshot_0 fixed fields (convert counters to floats for tabular)
        feats["views_0"] = _safe_float(s0.get("viewCount"))
        feats["likes_0"] = _safe_float(s0.get("likeCount"))
        feats["comments_0"] = _safe_float(s0.get("commentCount"))
        feats["channel_subscribers_0"] = _safe_float(s0.get("subscriberCount"))
        feats["channel_total_views_0"] = _safe_float(s0.get("viewCount_channel"))
        feats["channel_total_videos_0"] = _safe_float(s0.get("videoCount"))

        # temporal fields
        feats["publishedAt"] = _safe_str(meta.get("publishedAt"))
        feats["language"] = _safe_str(meta.get("language"))
        # duration fallback
        dur = meta.get("duration_seconds")
        if dur is None:
            dur = meta.get("duration")
        feats["duration_sec"] = _safe_float(dur)

        # derived: video_age_hours_at_snapshot1 (snapshot_0 time approximated by manifest created_at)
        dt_created = _parse_iso8601(feats.get("manifest_created_at"))
        dt_pub = _parse_iso8601(feats.get("publishedAt"))
        if dt_created and dt_pub:
            feats["video_age_hours_at_snapshot1"] = (dt_created - dt_pub).total_seconds() / 3600.0
        else:
            feats["video_age_hours_at_snapshot1"] = float("nan")

        # targets (log1p deltas vs snapshot_0)
        t = compute_targets_from_record(rec)
        feats["target_views_7d"] = t.y_views_7d
        feats["target_views_14d"] = t.y_views_14d
        feats["target_views_21d"] = t.y_views_21d
        feats["target_likes_7d"] = t.y_likes_7d
        feats["target_likes_14d"] = t.y_likes_14d
        feats["target_likes_21d"] = t.y_likes_21d
        feats["mask_7d"] = t.m_7d
        feats["mask_14d"] = t.m_14d
        feats["mask_21d"] = t.m_21d

        if require_14_21 and (t.m_14d <= 0.0 or t.m_21d <= 0.0):
            dropped_missing_required += 1
            continue

        out_rows.append(DatasetRow(keys=r.keys, features=feats))

    meta_out = {
        "targets_source": data_json_path,
        "video_ids_requested": len(video_ids),
        "records_seen_in_data_json": stats.total_records_seen,
        "records_loaded_from_data_json": stats.total_records_yielded,
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "rows_missing_video_record": missing_video_record,
        "rows_dropped_missing_required_targets": dropped_missing_required,
    }
    return out_rows, meta_out


def rows_to_columnar(rows: List[DatasetRow]) -> Tuple[List[Dict[str, Any]], List[str]]:
    dict_rows: List[Dict[str, Any]] = []
    cols: List[str] = []
    for r in rows:
        row = {**r.keys, **r.features}
        dict_rows.append(row)
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    # stable ordering: keys first (common), then sorted rest
    key_order = ["platform_id", "video_id", "run_id", "config_hash", "sampling_policy_version", "manifest_path"]
    rest = sorted([c for c in cols if c not in key_order])
    ordered_cols = key_order + rest
    return dict_rows, ordered_cols


def write_dataset(rows: List[DatasetRow], out_path: str) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dict_rows, cols = rows_to_columnar(rows)
    wrote = {"format": None, "path": str(out), "columns": cols}

    # Try parquet (pyarrow), else csv
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(dict_rows, columns=cols)
        if out.suffix.lower() == ".parquet":
            df.to_parquet(out, index=False)
            wrote["format"] = "parquet"
        else:
            df.to_csv(out, index=False)
            wrote["format"] = "csv"
        wrote["rows"] = int(df.shape[0])
        return wrote
    except Exception:
        # fallback: jsonl (always available)
        jsonl = out.with_suffix(out.suffix + ".jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            for row in dict_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        wrote["format"] = "jsonl_fallback"
        wrote["path"] = str(jsonl)
        wrote["rows"] = len(dict_rows)
        return wrote


def compute_dataset_fingerprint(rows: List[DatasetRow], feature_schema_version: str) -> str:
    # Deterministic fingerprint based on run identity keys + schema version.
    parts: List[str] = []
    for r in rows:
        parts.append(
            "|".join(
                [
                    str(r.keys.get("platform_id") or ""),
                    str(r.keys.get("video_id") or ""),
                    str(r.keys.get("run_id") or ""),
                    str(r.keys.get("config_hash") or ""),
                    str(r.keys.get("sampling_policy_version") or ""),
                ]
            )
        )
    parts.sort()
    return _sha256_text(feature_schema_version + "\n" + "\n".join(parts))


def main() -> int:
    p = argparse.ArgumentParser(description="Build full baseline dataset: features + snapshot_0 + targets")
    p.add_argument("--rs-base", type=str, required=True, help="Base result_store directory (where manifest.json are)")
    p.add_argument("--data-json", type=str, required=True, help="Path to data_00.json (video_id -> metadata + snapshot_0..3)")
    p.add_argument(
        "--feature-spec",
        type=str,
        default=str(Path(__file__).with_name("feature_spec.yaml")),
        help="Path to feature_spec.yaml",
    )
    p.add_argument("--out-dataset", type=str, required=True, help="Output dataset path (.parquet preferred, else .csv)")
    p.add_argument("--out-metadata", type=str, required=True, help="Output dataset_metadata.json path")
    p.add_argument("--require-14-21", action="store_true", help="Drop rows missing required 14d/21d targets")
    p.add_argument("--enforce-required-components", action="store_true", help="Drop/error rows missing required components (ok|empty allowed)")
    p.add_argument("--required-policy", type=str, default="drop", choices=["drop", "error"], help="What to do if required components missing/error")
    args = p.parse_args()

    spec = _load_yaml_minimal(args.feature_spec)
    feature_schema_version = str(spec.get("feature_schema_version") or "unknown")
    comps = spec.get("baseline_components") or []
    allowed_components = set()
    required_components = set()
    for c in comps:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            allowed_components.add(str(c["name"]))
            if str(c.get("required")).lower() == "true" or c.get("required") is True:
                required_components.add(str(c["name"]))

    rows = build_feature_rows(
        args.rs_base,
        allowed_components=allowed_components or None,
        required_components=required_components if args.enforce_required_components else None,
    )
    rows, targets_meta = add_snapshot_and_targets(rows, data_json_path=args.data_json, require_14_21=args.require_14_21)

    # Enforce required components after targets join (so metadata reports are accurate).
    missing_required_rows = 0
    if args.enforce_required_components and required_components:
        filtered: List[DatasetRow] = []
        for r in rows:
            bad = []
            for rc in required_components:
                v = float(r.features.get(f"component_status__{rc}", -1.0))
                if v < 0.0:  # missing or error
                    bad.append(rc)
            if bad:
                missing_required_rows += 1
                if args.required_policy == "error":
                    raise RuntimeError(f"Missing/error required components for video_id={r.keys.get('video_id')}: {bad}")
                continue
            filtered.append(r)
        rows = filtered

    wrote = write_dataset(rows, args.out_dataset)
    fingerprint = compute_dataset_fingerprint(rows, feature_schema_version)

    metadata = {
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_fingerprint": fingerprint,
        "feature_schema_version": feature_schema_version,
        "feature_spec_path": args.feature_spec,
        "rs_base": args.rs_base,
        "targets": targets_meta,
        "required_components": {
            "enabled": bool(args.enforce_required_components),
            "policy": args.required_policy,
            "required": sorted(list(required_components)),
            "rows_dropped_missing_required_components": missing_required_rows,
        },
        "dataset": wrote,
    }

    out_meta = Path(args.out_metadata)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] dataset -> {wrote['path']} ({wrote['format']}, rows={wrote.get('rows')})")
    print(f"[ok] metadata -> {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


