#!/usr/bin/env python3
"""
build_training_table.py  —  DatasetBuilder stage C1

Deterministically build a FLAT feature table from the per-run result_store,
driven by feature_spec.yaml (v0-real). One row per (video_id, run_id).

It reuses Models/baseline/common/npz_features.py for NPZ parsing (single source
of truth for how an artifact becomes tabular features) — this file only decides
WHICH components/columns are kept and glues in identity + temporal + snapshot_0.

Output: parquet (default) or csv + <out>.metadata.json (fingerprint, spec hash).

Leakage note: every feature here comes from video frames/audio + snapshot_0 +
run metadata. Targets (future snapshots) are added separately by add_targets.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# --- make Models/baseline/common importable ---------------------------------
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]  # .../TrendFlowML
sys.path.insert(0, str(REPO_ROOT / "Models"))
from baseline.common.npz_features import extract_features_from_npz  # noqa: E402


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# feature_spec handling
# ---------------------------------------------------------------------------
def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spec_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _make_column_filter(spec: Dict[str, Any]):
    g = spec.get("global_column_exclude") or {}
    substrings = list(g.get("substrings") or [])
    suffixes = list(g.get("suffixes") or [])

    def keep(col_full: str, col_local: str, per_comp_substrings: List[str]) -> bool:
        # substring patterns are tested against BOTH the full prefixed name and
        # the component-local name (e.g. "__version" lives in the full name,
        # while a bare "version" local also matches via col_local).
        for s in substrings:
            if s in col_full or s in col_local:
                return False
        for s in per_comp_substrings:
            if s in col_full or s in col_local:
                return False
        for suf in suffixes:
            if col_local.endswith(suf) or col_full.endswith(suf):
                return False
        return True

    return keep


# ---------------------------------------------------------------------------
# result_store walking
# ---------------------------------------------------------------------------
def iter_runs(result_store: str, platform: str = "youtube"):
    """Yield (video_id, run_id, run_dir, manifest_dict) for every run with a manifest."""
    base = Path(result_store) / platform
    if not base.is_dir():
        return
    for video_dir in sorted(base.iterdir()):
        if not video_dir.is_dir():
            continue
        for run_dir in sorted(video_dir.iterdir()):
            mpath = run_dir / "manifest.json"
            if not mpath.is_file():
                continue
            try:
                manifest = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            yield video_dir.name, run_dir.name, run_dir, manifest


def _component_npz_paths(run_dir: Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in manifest.get("components") or []:
        name = c.get("name")
        if not name:
            continue
        for a in c.get("artifacts") or []:
            if a.get("type") == "npz" and isinstance(a.get("path"), str):
                p = run_dir / a["path"]
                if p.exists():
                    out.setdefault(name, str(p))
                    break
    return out


def _run_richness(manifest: Dict[str, Any]) -> int:
    """How many components produced an ok status — used to pick the best run per video."""
    return sum(1 for c in (manifest.get("components") or []) if c.get("status") == "ok")


def _extract_component_features(
    comp_to_paths: Dict[str, List[str]],
    included: List[str],
    per_comp_excl: Dict[str, List[str]],
    keep_col,
) -> Tuple[Dict[str, float], int]:
    """Single source of truth for turning per-component NPZ paths into filtered
    tabular features. Accepts MULTIPLE npz per component (e.g. cut_detection emits
    a features file + a model_facing file) and merges them."""
    row: Dict[str, float] = {}
    n_feat = 0
    for comp in included:
        for p in comp_to_paths.get(comp, []):
            try:
                feats, _ = extract_features_from_npz(comp, p)
            except Exception:
                continue
            excl = per_comp_excl.get(comp, [])
            for k, v in feats.items():
                local = k[len(comp) + 2:] if k.startswith(comp + "__") else k
                if not keep_col(k, local, excl):
                    continue
                if k not in row:
                    n_feat += 1
                row[k] = v
    return row, n_feat


# ---------------------------------------------------------------------------
# direct-rs walking (Agent A corpus_out: rs/<component>/*.npz, NO manifest.json)
# ---------------------------------------------------------------------------
def iter_rs_videos(rs_root: str):
    """Yield (video_id, {component: [npz_paths]}) for a corpus_out-style tree
    laid out as <rs_root>/<video_id>/rs/<component>/*.npz (Agent A's run_corpus.sh
    output, which has no per-run manifest.json)."""
    base = Path(rs_root)
    if not base.is_dir():
        return
    for video_dir in sorted(base.iterdir()):
        rsd = video_dir / "rs"
        if not rsd.is_dir():
            continue
        comp_to_paths: Dict[str, List[str]] = {}
        for comp_dir in sorted(rsd.iterdir()):
            if not comp_dir.is_dir():
                continue
            npzs = sorted(str(p) for p in comp_dir.glob("*.npz"))
            if npzs:
                comp_to_paths[comp_dir.name] = npzs
        if comp_to_paths:
            yield video_dir.name, comp_to_paths


def build_from_rs(
    rs_root: str,
    spec_path: str,
    *,
    platform: str = "youtube",
) -> Tuple["pd.DataFrame", Dict[str, Any]]:  # type: ignore # noqa: F821
    """C1 variant: build the flat feature table directly from a corpus_out rs/ tree
    (no manifest.json). Same feature_spec filtering as build(); one row per video."""
    import pandas as pd

    spec = load_spec(spec_path)
    keep_col = _make_column_filter(spec)
    comp_specs = spec.get("components") or []
    included = [c["name"] for c in comp_specs]
    per_comp_excl = {c["name"]: list(c.get("column_exclude") or []) for c in comp_specs}
    snap_fields = spec.get("snapshot0_fields") or []
    temporal_fields = spec.get("temporal_fields") or []

    rows: List[Dict[str, Any]] = []
    for vid, comp_to_paths in iter_rs_videos(rs_root):
        row: Dict[str, Any] = {"platform_id": platform, "video_id": vid, "run_id": "rs_direct"}
        feats, n_feat = _extract_component_features(comp_to_paths, included, per_comp_excl, keep_col)
        row.update(feats)
        row["_n_features"] = n_feat
        row["_n_components_present"] = len(comp_to_paths)
        for sf in snap_fields:
            row.setdefault(sf, float("nan"))
        for tf in temporal_fields:
            row.setdefault(tf, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    id_like = ["platform_id", "video_id", "run_id", "_n_features", "_n_components_present",
               "video_age_hours_at_snapshot1"] + list(snap_fields)
    id_present = [c for c in id_like if c in df.columns]
    feat_cols = sorted([c for c in df.columns if c not in id_present])
    df = df[id_present + feat_cols]

    meta = {
        "created_at": _now_utc(),
        "feature_schema_version": spec.get("feature_schema_version"),
        "feature_spec_hash": _spec_hash(spec_path),
        "source": "direct_rs",
        "rs_root": os.path.abspath(rs_root),
        "platform": platform,
        "all_runs": False,
        "n_rows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()) if len(df) else 0,
        "n_feature_columns": int(len(feat_cols)),
        "included_components": included,
        "skipped": {},
    }
    return df, meta


# ---------------------------------------------------------------------------
# snapshot_0 / metadata extraction from run manifest (best-effort)
# ---------------------------------------------------------------------------
def _extract_run_meta(manifest: Dict[str, Any]) -> Dict[str, Any]:
    run = manifest.get("run") or {}
    meta: Dict[str, Any] = {
        "platform_id": run.get("platform_id"),
        "publishedAt": run.get("publishedAt") or run.get("published_at"),
        "channel_id": run.get("channel_id") or run.get("channelId"),
        "channelTitle": run.get("channelTitle") or run.get("channel_title"),
        "config_hash": run.get("config_hash"),
        "analysis_fps": run.get("analysis_fps"),
        "duration_sec": run.get("duration_sec") or run.get("duration_seconds"),
        "manifest_created_at": run.get("created_at"),
        "language": run.get("language"),
    }
    return meta


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(
    result_store: str,
    spec_path: str,
    *,
    platform: str = "youtube",
    all_runs: bool = False,
    require_ok_min: int = 1,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:  # type: ignore # noqa: F821
    import pandas as pd

    spec = load_spec(spec_path)
    keep_col = _make_column_filter(spec)
    comp_specs = spec.get("components") or []
    included = [c["name"] for c in comp_specs]
    per_comp_excl = {c["name"]: list(c.get("column_exclude") or []) for c in comp_specs}
    snap_fields = spec.get("snapshot0_fields") or []
    temporal_fields = spec.get("temporal_fields") or []

    # group runs per video to allow "best run only"
    per_video: Dict[str, List[Tuple[str, Path, Dict[str, Any]]]] = {}
    for vid, rid, rdir, manifest in iter_runs(result_store, platform):
        per_video.setdefault(vid, []).append((rid, rdir, manifest))

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {"low_richness": 0}
    for vid, runs in per_video.items():
        # richest first
        runs_sorted = sorted(runs, key=lambda t: _run_richness(t[2]), reverse=True)
        chosen = runs_sorted if all_runs else runs_sorted[:1]
        for rid, rdir, manifest in chosen:
            if _run_richness(manifest) < require_ok_min:
                skipped["low_richness"] += 1
                continue
            comp_paths = _component_npz_paths(rdir, manifest)
            row: Dict[str, Any] = {
                "platform_id": platform,
                "video_id": vid,
                "run_id": rid,
                "manifest_path": str(rdir / "manifest.json"),
            }
            row.update(_extract_run_meta(manifest))

            comp_to_paths = {c: [p] for c, p in comp_paths.items()}
            feats, n_feat = _extract_component_features(comp_to_paths, included, per_comp_excl, keep_col)
            row.update(feats)
            row["_n_features"] = n_feat

            # snapshot_0 fields — placeholder columns (filled from HF metadata if
            # available in the manifest run block; else NaN, native-missing).
            run = manifest.get("run") or {}
            for sf in snap_fields:
                row.setdefault(sf, run.get(sf, float("nan")))
            for tf in temporal_fields:
                if tf not in row or row[tf] is None:
                    row[tf] = run.get(tf, float("nan"))

            rows.append(row)

    df = pd.DataFrame(rows)
    # stable column order: ids first, then sorted feature cols
    id_like = [
        "platform_id", "video_id", "run_id", "channel_id", "channelTitle",
        "publishedAt", "config_hash", "manifest_path", "language",
        "manifest_created_at", "duration_sec", "analysis_fps",
        "video_age_hours_at_snapshot1", "_n_features",
    ] + list(snap_fields)
    id_present = [c for c in id_like if c in df.columns]
    feat_cols = sorted([c for c in df.columns if c not in id_present])
    df = df[id_present + feat_cols]

    meta = {
        "created_at": _now_utc(),
        "feature_schema_version": spec.get("feature_schema_version"),
        "feature_spec_hash": _spec_hash(spec_path),
        "result_store": os.path.abspath(result_store),
        "platform": platform,
        "all_runs": all_runs,
        "n_rows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()) if len(df) else 0,
        "n_feature_columns": int(len(feat_cols)),
        "included_components": included,
        "skipped": skipped,
    }
    return df, meta


def _fingerprint(df: "pd.DataFrame") -> str:  # type: ignore # noqa: F821
    import pandas as pd  # noqa: F401
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(str(df.shape).encode())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build flat feature table from result_store (DatasetBuilder C1)")
    ap.add_argument("--result-store", default=str(REPO_ROOT / "storage" / "result_store"))
    ap.add_argument("--rs-root", default="", help="Direct corpus_out rs/ tree (<root>/<vid>/rs/<comp>/*.npz, no manifest). Overrides --result-store.")
    ap.add_argument("--feature-spec", default=str(THIS.parent / "feature_spec.yaml"))
    ap.add_argument("--platform", default="youtube")
    ap.add_argument("--out", required=True, help="Output path (.parquet or .csv)")
    ap.add_argument("--all-runs", action="store_true", help="Emit every run (default: richest run per video)")
    ap.add_argument("--require-ok-min", type=int, default=1, help="Min ok components to keep a run")
    args = ap.parse_args()

    if args.rs_root:
        df, meta = build_from_rs(args.rs_root, args.feature_spec, platform=args.platform)
    else:
        df, meta = build(
            args.result_store, args.feature_spec,
            platform=args.platform, all_runs=args.all_runs, require_ok_min=args.require_ok_min,
        )
    meta["dataset_fingerprint"] = _fingerprint(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    elif out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError(f"Unsupported output format: {out.suffix}")

    (out.parent / (out.stem + ".metadata.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] {meta['n_rows']} rows x {meta['n_feature_columns']} feat cols "
          f"({meta['n_videos']} videos) -> {out}")
    print(f"[ok] fingerprint={meta['dataset_fingerprint']} spec={meta['feature_schema_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
