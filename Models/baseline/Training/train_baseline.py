#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
# Make local baseline modules importable when running as a script.
sys.path.insert(0, str(REPO_ROOT))

from Models.baseline.Training.utils_metrics import age_bucket, compute_metrics, compute_metrics_by_bucket  # noqa: E402


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso8601(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
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


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def load_dataset(path: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Training requires pandas. Install it in your venv.") from e

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower().endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported dataset format: {path}")


def select_numeric_features(df: "pd.DataFrame", *, exclude_cols: set[str]) -> Tuple["pd.DataFrame", List[str]]:
    import pandas as pd  # type: ignore

    cols: List[str] = []
    dropped_const = 0
    dropped_allnan = 0
    for c in df.columns:
        if c in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c]
        nfin = s[s.notna()]
        # Drop degenerate columns: all-NaN or constant (<=1 unique value). They carry
        # no signal AND break HistGradientBoosting's binning on numpy>=2/py3.14
        # ("window shape cannot be larger than input array shape" in _find_binning_thresholds).
        if len(nfin) == 0:
            dropped_allnan += 1
            continue
        if nfin.nunique() <= 1:
            dropped_const += 1
            continue
        cols.append(c)
    if dropped_const or dropped_allnan:
        print(f"[features] dropped {dropped_const} constant + {dropped_allnan} all-NaN columns "
              f"-> {len(cols)} usable numeric features")
    return df[cols], cols


def split_hybrid_time_channel(
    df: "pd.DataFrame",
    *,
    channel_col: str,
    published_col: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> "pd.Series":
    """
    Deterministic hybrid split:
      - compute per-channel min publishedAt
      - sort channels by that min time
      - assign channels into train/val/test by fractions
    """
    import pandas as pd  # type: ignore

    if channel_col not in df.columns:
        raise ValueError(f"Missing channel column for group split: {channel_col}")
    if published_col not in df.columns:
        raise ValueError(f"Missing publishedAt column: {published_col}")

    # parse publishedAt to datetime (UTC)
    pub_dt = df[published_col].apply(_parse_iso8601)
    # fill missing with epoch to keep deterministic ordering (will go to train)
    pub_dt = pub_dt.fillna(datetime(1970, 1, 1, tzinfo=timezone.utc))

    tmp = pd.DataFrame({channel_col: df[channel_col].astype(str), "_pub": pub_dt})
    g = tmp.groupby(channel_col)["_pub"].min().sort_values()
    channels = list(g.index)

    n = len(channels)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_set = set(channels[:n_train])
    val_set = set(channels[n_train : n_train + n_val])
    # rest -> test

    def _assign(ch: str) -> str:
        if ch in train_set:
            return "train"
        if ch in val_set:
            return "val"
        return "test"

    split = df[channel_col].astype(str).apply(_assign)
    return split


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_col: str
    mask_col: Optional[str]


def build_model_specs() -> List[ModelSpec]:
    # 6 outputs (views/likes × 7/14/21). Every horizon carries a per-horizon mask:
    # real data (pre_final_data) has videos missing an individual follow-up snapshot,
    # so 14d/21d targets can be NaN too, not just 7d. Respect the mask for all three
    # (mask columns default to 1.0 when a horizon is always present, e.g. smoke set).
    return [
        ModelSpec("views_7d", "target_views_7d", "mask_7d"),
        ModelSpec("views_14d", "target_views_14d", "mask_14d"),
        ModelSpec("views_21d", "target_views_21d", "mask_21d"),
        ModelSpec("likes_7d", "target_likes_7d", "mask_7d"),
        ModelSpec("likes_14d", "target_likes_14d", "mask_14d"),
        ModelSpec("likes_21d", "target_likes_21d", "mask_21d"),
    ]


def fit_model(
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    *,
    model_family: str,
    seed: int,
) -> Any:
    if model_family == "catboost":
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except Exception as e:
            raise RuntimeError("CatBoost is not installed. Install catboost or use --model-family sklearn.") from e
        model = CatBoostRegressor(
            random_seed=seed,
            loss_function="RMSE",
            depth=8,
            learning_rate=0.05,
            iterations=2000,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train)
        return model

    if model_family == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:
            raise RuntimeError("LightGBM is not installed. Install lightgbm or use --model-family sklearn.") from e
        model = lgb.LGBMRegressor(
            random_state=seed,
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        model.fit(X_train, y_train)
        return model

    if model_family == "sklearn":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sklearn is not installed. Install scikit-learn or use catboost/lightgbm in your training venv."
            ) from e
        model = HistGradientBoostingRegressor(
            random_state=seed,
            max_depth=8,
            learning_rate=0.05,
            max_iter=500,
        )
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Unknown model family: {model_family}")


def predict_model(model: Any, X: "pd.DataFrame") -> List[float]:
    y = model.predict(X)
    # normalize to python floats
    return [float(v) for v in list(y)]


def main() -> int:
    p = argparse.ArgumentParser(description="Train baseline boosting models (M5)")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset (.parquet/.csv/.jsonl)")
    p.add_argument("--dataset-metadata", type=str, required=False, help="Path to dataset_metadata.json (optional)")
    p.add_argument("--feature-spec", type=str, required=False, help="Path to feature_spec.yaml (copied into artifact)")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for model artifacts")
    p.add_argument("--model-family", type=str, default="catboost", choices=["catboost", "lightgbm", "sklearn"])
    p.add_argument("--model-version", type=str, default="", help="Pinned model version string (stored in manifest)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--channel-col", type=str, default="channel_id", help="Column for channel-group split (preferred)")
    args = p.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = args.model_version.strip() or out_dir.name

    df = load_dataset(args.dataset)

    # If no channel_id yet, fall back (degraded) to channelTitle, else video_id as last resort
    channel_col = args.channel_col
    if channel_col not in df.columns:
        if "channelTitle" in df.columns:
            channel_col = "channelTitle"
        else:
            channel_col = "video_id"

    split = split_hybrid_time_channel(df, channel_col=channel_col, published_col="publishedAt")
    df = df.assign(_split=split)

    # Build feature matrix (numeric only)
    exclude = {
        "_split",
        "platform_id",
        "video_id",
        "run_id",
        "config_hash",
        "sampling_policy_version",
        "manifest_path",
        "publishedAt",
        "language",
        "manifest_created_at",
    }
    # targets + masks
    for c in ["target_views_7d", "target_views_14d", "target_views_21d", "target_likes_7d", "target_likes_14d", "target_likes_21d"]:
        exclude.add(c)
    for c in ["mask_7d", "mask_14d", "mask_21d"]:
        exclude.add(c)

    X_all, feature_names = select_numeric_features(df, exclude_cols=exclude)

    # Missing handling:
    # - catboost/lightgbm: keep NaN
    # - sklearn: median imputation (store params for eval/inference parity)
    imputation: Dict[str, Any] = {"strategy": "none"}
    if args.model_family == "sklearn":
        med = X_all.median(numeric_only=True)
        X_all = X_all.fillna(med)
        imputation = {"strategy": "median", "medians_by_feature": {k: float(v) for k, v in med.to_dict().items()}}

    artifact = {
        "created_at": _now_utc(),
        "seed": args.seed,
        "model_family": args.model_family,
        "model_version": model_version,
        "channel_col_used": channel_col,
        "feature_names": feature_names,
        "targets": [],
        "imputation": imputation,
        "dataset": {
            "path": args.dataset,
            "metadata_path": args.dataset_metadata,
        },
    }

    # Optional: attach dataset fingerprint if provided
    if args.dataset_metadata and Path(args.dataset_metadata).exists():
        try:
            meta = json.loads(Path(args.dataset_metadata).read_text(encoding="utf-8"))
            artifact["dataset"]["dataset_fingerprint"] = meta.get("dataset_fingerprint")
            artifact["dataset"]["feature_schema_version"] = meta.get("feature_schema_version")
        except Exception:
            pass

    # copy feature_spec snapshot (reproducibility)
    if args.feature_spec and Path(args.feature_spec).exists():
        shutil.copyfile(args.feature_spec, out_dir / "feature_spec.yaml")

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {"created_at": _now_utc(), "splits": {}, "per_model": {}}

    for split_name in ["train", "val", "test"]:
        metrics["splits"][split_name] = int((df["_split"] == split_name).sum())

    # We package as two bundles (views/likes), each with 3 horizon heads.
    bundles = {
        "views": build_model_specs()[0:3],
        "likes": build_model_specs()[3:6],
    }
    artifact["bundles"] = []

    for bundle_name, specs in bundles.items():
        (models_dir / bundle_name).mkdir(parents=True, exist_ok=True)
        bundle_entry = {"name": bundle_name, "horizons": []}

        for spec in specs:
            if spec.target_col not in df.columns:
                raise ValueError(f"Missing target column: {spec.target_col}")

            mask = None
            if spec.mask_col is not None:
                if spec.mask_col not in df.columns:
                    raise ValueError(f"Missing mask column: {spec.mask_col}")
                mask = df[spec.mask_col].astype(float)

            # train on train split (and on masked rows if needed)
            is_train = df["_split"] == "train"
            if mask is not None:
                is_train = is_train & (mask > 0.0)

            X_train = X_all[is_train]
            y_train = df.loc[is_train, spec.target_col].astype(float)

            model = fit_model(X_train, y_train, model_family=args.model_family, seed=args.seed)

            # save model under bundle/<horizon>.* (7d/14d/21d)
            horizon = spec.name.split("_")[-1]
            model_path = (models_dir / bundle_name) / f"{horizon}.pkl"
            if args.model_family == "catboost":
                model_path = (models_dir / bundle_name) / f"{horizon}.cbm"
                model.save_model(str(model_path))
            elif args.model_family == "lightgbm":
                model_path = (models_dir / bundle_name) / f"{horizon}.txt"
                model.booster_.save_model(str(model_path))
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            # evaluate on val/test (respect mask for 7d metrics)
            model_metrics: Dict[str, Any] = {"model_path": str(model_path), "target_col": spec.target_col, "mask_col": spec.mask_col}

            for eval_split in ["val", "test"]:
                is_eval = df["_split"] == eval_split
                if mask is not None:
                    is_eval = is_eval & (mask > 0.0)

                X_eval = X_all[is_eval]
                y_true = [float(v) for v in df.loc[is_eval, spec.target_col].astype(float).tolist()]
                y_pred = predict_model(model, X_eval)

                overall = compute_metrics(y_true, y_pred)

                ages = (
                    [float(v) for v in df.loc[is_eval, "video_age_hours_at_snapshot1"].astype(float).tolist()]
                    if "video_age_hours_at_snapshot1" in df.columns
                    else [float("nan")] * len(y_true)
                )
                buckets = [age_bucket(a) for a in ages]
                by_bucket = compute_metrics_by_bucket(y_true=y_true, y_pred=y_pred, buckets=buckets)

                model_metrics[eval_split] = {
                    "overall": {"n": overall.n, "spearman": overall.spearman, "mae": overall.mae},
                    "by_age_bucket": {k: {"n": v.n, "spearman": v.spearman, "mae": v.mae} for k, v in by_bucket.items()},
                }

            metrics["per_model"][f"{bundle_name}_{horizon}"] = model_metrics
            bundle_entry["horizons"].append(
                {
                    "name": horizon,  # 7d/14d/21d
                    "target_col": spec.target_col,
                    "mask_col": spec.mask_col,
                    "model_path": str(model_path),
                }
            )

            # Keep legacy flat list too (easy to diff/debug)
            artifact["targets"].append({"name": spec.name, "target_col": spec.target_col, "mask_col": spec.mask_col})

        artifact["bundles"].append(bundle_entry)

    # write manifests
    (out_dir / "training_run_manifest.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


