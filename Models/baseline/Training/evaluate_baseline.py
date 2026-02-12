#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Models.baseline.Training.train_baseline import split_hybrid_time_channel  # noqa: E402
from Models.baseline.Training.utils_metrics import age_bucket, compute_metrics, compute_metrics_by_bucket  # noqa: E402


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_dataset(path: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Evaluation requires pandas. Install it in your venv.") from e

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower().endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported dataset format: {path}")


def load_training_manifest(model_dir: str) -> Dict[str, Any]:
    p = Path(model_dir) / "training_run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_models(model_dir: str, training_manifest: Dict[str, Any]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    family = str(training_manifest.get("model_family") or "unknown")
    models_root = Path(model_dir) / "models"

    bundles = training_manifest.get("bundles") or []
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("training_run_manifest.json missing bundles[] (expected new format)")

    for b in bundles:
        if not isinstance(b, dict):
            continue
        bname = b.get("name")
        if not isinstance(bname, str) or not bname:
            continue
        horizons = b.get("horizons") or []
        if not isinstance(horizons, list):
            continue
        for h in horizons:
            if not isinstance(h, dict):
                continue
            hname = h.get("name")
            if not isinstance(hname, str) or not hname:
                continue
            key = f"{bname}__{hname}"
            if family == "catboost":
                from catboost import CatBoostRegressor  # type: ignore

                m = CatBoostRegressor()
                m.load_model(str(models_root / bname / f"{hname}.cbm"))
                models[key] = m
            elif family == "lightgbm":
                import lightgbm as lgb  # type: ignore

                models[key] = lgb.Booster(model_file=str(models_root / bname / f"{hname}.txt"))
            else:
                with open(models_root / bname / f"{hname}.pkl", "rb") as f:
                    models[key] = pickle.load(f)
    return models


def predict(model: Any, X: "pd.DataFrame", *, model_family: str) -> List[float]:
    if model_family == "lightgbm":
        y = model.predict(X)
        return [float(v) for v in list(y)]
    y = model.predict(X)
    return [float(v) for v in list(y)]


def _stable_sample_video_ids(video_ids: List[str], n: int, seed: int) -> List[str]:
    if len(video_ids) <= n:
        return video_ids
    # stable hash sort (not random.shuffle) to keep determinism across envs
    def key(v: str) -> str:
        return hashlib.sha256((str(seed) + "::" + v).encode("utf-8")).hexdigest()

    video_ids_sorted = sorted(video_ids, key=key)
    return video_ids_sorted[:n]


def render_md_report(metrics: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Baseline evaluation report")
    lines.append("")
    lines.append(f"- created_at: `{metrics.get('created_at')}`")
    lines.append(f"- model_dir: `{metrics.get('model_dir')}`")
    lines.append(f"- model_version: `{metrics.get('model_version')}`")
    lines.append(f"- dataset: `{metrics.get('dataset_path')}`")
    lines.append(f"- eval_set: `{metrics.get('eval_set')}`")
    lines.append("")

    per = metrics.get("per_bundle") or {}
    for bundle_name, bundle_metrics in per.items():
        lines.append(f"## {bundle_name}")
        horizons = bundle_metrics.get("horizons") or {}
        for hname, hmetrics in horizons.items():
            lines.append(f"### {hname}")
            ov = (hmetrics.get("overall") or {})
            lines.append(f"- overall: n={ov.get('n')} spearman={ov.get('spearman')} mae={ov.get('mae')}")
            byb = hmetrics.get("by_age_bucket") or {}
            if byb:
                lines.append("- by_age_bucket:")
                for b, row in byb.items():
                    lines.append(f"  - {b}: n={row.get('n')} spearman={row.get('spearman')} mae={row.get('mae')}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate baseline models (quality gate)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--eval-set", type=str, default="test", choices=["train", "val", "test", "holdout", "regression_mini"])
    p.add_argument("--holdout-video-ids", type=str, default="", help="Optional JSON list of video_ids for holdout")
    p.add_argument("--regression-video-ids", type=str, default="", help="Optional JSON list of video_ids for regression mini")
    p.add_argument(
        "--golden-set-dir",
        type=str,
        default="",
        help="Optional directory with {holdout.json,regression_mini.json} (overrides *-video-ids if present)",
    )
    p.add_argument("--regression-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--channel-col", type=str, default="channel_id")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    tr = load_training_manifest(args.model_dir)
    model_family = str(tr.get("model_family") or "unknown")
    model_version = tr.get("model_version")

    feature_names = tr.get("feature_names") or []
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("training_run_manifest.json missing feature_names[]")

    # Split logic should match training
    channel_col = args.channel_col
    if channel_col not in df.columns:
        if "channelTitle" in df.columns:
            channel_col = "channelTitle"
        else:
            channel_col = "video_id"
    split = split_hybrid_time_channel(df, channel_col=channel_col, published_col="publishedAt")
    df = df.assign(_split=split)

    # Prepare eval subset
    if args.eval_set in ("train", "val", "test"):
        df_eval = df[df["_split"] == args.eval_set].copy()
        eval_set = args.eval_set
    elif args.eval_set == "holdout":
        ids_path = args.holdout_video_ids
        if args.golden_set_dir:
            ids_path = str(Path(args.golden_set_dir) / "holdout.json")
        if not ids_path:
            raise ValueError("Need --holdout-video-ids or --golden-set-dir when --eval-set=holdout")
        vids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        vids_set = set([str(v) for v in vids])
        df_eval = df[df["video_id"].astype(str).isin(vids_set)].copy()
        eval_set = "holdout"
    else:
        ids_path = args.regression_video_ids
        if args.golden_set_dir:
            ids_path = str(Path(args.golden_set_dir) / "regression_mini.json")
        if ids_path:
            vids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
            vids_list = [str(v) for v in vids]
        else:
            vids_list = [str(v) for v in df[df["_split"] == "test"]["video_id"].astype(str).tolist()]
        vids_sel = set(_stable_sample_video_ids(vids_list, args.regression_size, args.seed))
        df_eval = df[df["video_id"].astype(str).isin(vids_sel)].copy()
        eval_set = "regression_mini"

    # Build X matrix
    import pandas as pd  # type: ignore

    X = df_eval.reindex(columns=feature_names)

    # Missing handling parity (median imputation)
    imp = tr.get("imputation") or {}
    if isinstance(imp, dict) and imp.get("strategy") == "median":
        med = imp.get("medians_by_feature") or {}
        if isinstance(med, dict):
            for fn in feature_names:
                if fn in med:
                    X[fn] = X[fn].fillna(float(med[fn]))

    models = load_models(args.model_dir, tr)

    # Metrics
    per_bundle: Dict[str, Any] = {}

    # age buckets for stratification
    if "video_age_hours_at_snapshot1" in df_eval.columns:
        ages = [float(v) for v in df_eval["video_age_hours_at_snapshot1"].astype(float).tolist()]
    else:
        ages = [float("nan")] * int(df_eval.shape[0])
    buckets = [age_bucket(a) for a in ages]

    for bundle_name in ["views", "likes"]:
        per_bundle[bundle_name] = {"horizons": {}}
        for horizon in ["7d", "14d", "21d"]:
            key = f"{bundle_name}__{horizon}"
            m = models.get(key)
            if m is None:
                continue

            target_col = f"target_{bundle_name}_{horizon}"
            if target_col not in df_eval.columns:
                raise ValueError(f"Missing target column in dataset: {target_col}")

            mask_col = "mask_7d" if horizon == "7d" else None
            if mask_col and mask_col not in df_eval.columns:
                raise ValueError(f"Missing mask column: {mask_col}")

            if mask_col:
                mask = df_eval[mask_col].astype(float) > 0.0
                y_true = [float(v) for v in df_eval.loc[mask, target_col].astype(float).tolist()]
                y_pred = predict(m, X.loc[mask], model_family=model_family)
                buckets_h = [b for b, keep in zip(buckets, mask.tolist()) if keep]
            else:
                y_true = [float(v) for v in df_eval[target_col].astype(float).tolist()]
                y_pred = predict(m, X, model_family=model_family)
                buckets_h = buckets

            overall = compute_metrics(y_true, y_pred)
            by_bucket = compute_metrics_by_bucket(y_true=y_true, y_pred=y_pred, buckets=buckets_h)

            per_bundle[bundle_name]["horizons"][horizon] = {
                "overall": {"n": overall.n, "spearman": overall.spearman, "mae": overall.mae},
                "by_age_bucket": {k: {"n": v.n, "spearman": v.spearman, "mae": v.mae} for k, v in by_bucket.items()},
            }

    metrics = {
        "created_at": _now_utc(),
        "model_dir": args.model_dir,
        "model_family": model_family,
        "model_version": model_version,
        "dataset_path": args.dataset,
        "eval_set": eval_set,
        "n_rows": int(df_eval.shape[0]),
        "channel_col_used": channel_col,
        "per_bundle": per_bundle,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_md_report(metrics), encoding="utf-8")

    print(f"[ok] wrote metrics -> {out_dir / 'metrics.json'}")
    print(f"[ok] wrote report  -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


