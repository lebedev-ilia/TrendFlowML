#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
# Make local baseline modules importable when running as a script.
sys.path.insert(0, str(REPO_ROOT))

from Models.baseline.common.npz_features import extract_features_from_npz, find_first_npz_artifact  # noqa: E402


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def load_training_manifest(model_dir: str) -> Dict[str, Any]:
    p = Path(model_dir) / "training_run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_models(model_dir: str, training_manifest: Dict[str, Any]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    family = str(training_manifest.get("model_family") or "unknown")
    models_root = Path(model_dir) / "models"

    # New format: bundles (views/likes) with 3 horizons each.
    bundles = training_manifest.get("bundles") or []
    if isinstance(bundles, list) and bundles:
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

    # Legacy format: flat list of targets
    for t in training_manifest.get("targets") or []:
        name = t.get("name")
        if not isinstance(name, str) or not name:
            continue
        if family == "catboost":
            from catboost import CatBoostRegressor  # type: ignore

            m = CatBoostRegressor()
            m.load_model(str(models_root / f"{name}.cbm"))
            models[name] = m
        elif family == "lightgbm":
            import lightgbm as lgb  # type: ignore

            models[name] = lgb.Booster(model_file=str(models_root / f"{name}.txt"))
        else:
            with open(models_root / f"{name}.pkl", "rb") as f:
                models[name] = pickle.load(f)
    return models


def predict_one(model: Any, X_row: List[float], *, model_family: str) -> float:
    if model_family == "lightgbm":
        # Booster expects 2d matrix-like
        return float(model.predict([X_row])[0])
    return float(model.predict([X_row])[0])


def extract_features_from_run(
    *,
    manifest_path: str,
    allowed_components: Optional[set[str]] = None,
) -> Dict[str, float]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    feats: Dict[str, float] = {}

    comps = manifest.get("components") or []
    if isinstance(comps, list):
        for c in comps:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            if not isinstance(name, str) or not name:
                continue
            if allowed_components is not None and name not in allowed_components:
                continue

            status = c.get("status")
            val = -1.0
            if status == "ok":
                val = 1.0
            elif status == "empty":
                val = 0.0
            feats[f"component_status__{name}"] = val

            npz_path = find_first_npz_artifact(c)
            if npz_path:
                comp_feats, _meta = extract_features_from_npz(name, npz_path)
                feats.update(comp_feats)

    # manifest-level temporal (kept numeric)
    run = manifest.get("run") or {}
    feats["analysis_fps"] = _safe_float(run.get("analysis_fps"))
    feats["analysis_width"] = _safe_float(run.get("analysis_width"))
    feats["analysis_height"] = _safe_float(run.get("analysis_height"))
    return feats


def main() -> int:
    p = argparse.ArgumentParser(description="Baseline inference: run artifacts -> prediction JSON")
    p.add_argument("--rs-base", type=str, required=True, help="Base result_store directory")
    p.add_argument("--platform-id", type=str, required=True)
    p.add_argument("--video-id", type=str, required=True)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--model-dir", type=str, required=True, help="Training artifacts directory")
    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--feature-spec", type=str, default="", help="Optional feature_spec.yaml (for required-components enforcement)")
    p.add_argument("--enforce-required-components", action="store_true")
    p.add_argument("--required-policy", type=str, default="degraded", choices=["degraded", "error"])
    args = p.parse_args()

    manifest_path = Path(args.rs_base) / args.platform_id / args.video_id / args.run_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    tr = load_training_manifest(args.model_dir)
    feature_names = tr.get("feature_names") or []
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("training_run_manifest.json missing feature_names[]")

    model_family = str(tr.get("model_family") or "unknown")
    models = load_models(args.model_dir, tr)

    feats = extract_features_from_run(manifest_path=str(manifest_path), allowed_components=None)
    # Optional: required-components enforcement (ok|empty allowed; missing/error -> degraded/error).
    required_components: List[str] = []
    if args.feature_spec:
        try:
            import yaml  # type: ignore

            spec = yaml.safe_load(Path(args.feature_spec).read_text(encoding="utf-8")) or {}
            for c in spec.get("baseline_components") or []:
                if isinstance(c, dict) and c.get("required") is True and isinstance(c.get("name"), str):
                    required_components.append(str(c["name"]))
        except Exception:
            pass

    missing_required: List[str] = []
    if args.enforce_required_components and required_components:
        for c in required_components:
            v = feats.get(f"component_status__{c}", -1.0)
            if float(v) < 0.0:
                missing_required.append(c)

    prediction_status = "ok"
    if missing_required:
        prediction_status = "degraded"
        if args.required_policy == "error":
            raise RuntimeError(f"Missing/error required components: {missing_required}")

    X_row = [float(feats.get(f, float('nan'))) for f in feature_names]

    # Missing handling parity with training (if training stored imputation params).
    imp = tr.get("imputation") or {}
    if isinstance(imp, dict) and imp.get("strategy") == "median":
        med = imp.get("medians_by_feature") or {}
        if isinstance(med, dict):
            X_row = [float(med.get(fn, x)) if (isinstance(x, float) and math.isnan(x)) else x for fn, x in zip(feature_names, X_row)]

    preds: Dict[str, Any] = {}
    bundles = tr.get("bundles") or []
    if isinstance(bundles, list) and bundles:
        for b in bundles:
            if not isinstance(b, dict):
                continue
            bname = b.get("name")
            if not isinstance(bname, str) or not bname:
                continue
            preds[bname] = {}
            horizons = b.get("horizons") or []
            if not isinstance(horizons, list):
                continue
            for h in horizons:
                hname = h.get("name")
                if not isinstance(hname, str) or not hname:
                    continue
                key = f"{bname}__{hname}"
                m = models.get(key)
                if m is None:
                    continue
                preds[bname][hname] = predict_one(m, X_row, model_family=model_family)
    else:
        # legacy flat predictions
        preds_flat: Dict[str, float] = {}
    for t in tr.get("targets") or []:
        name = t.get("name")
        if not isinstance(name, str) or not name:
            continue
        m = models.get(name)
        if m is None:
            continue
            preds_flat[name] = predict_one(m, X_row, model_family=model_family)
        preds = preds_flat

    out = {
        "platform_id": args.platform_id,
        "video_id": args.video_id,
        "run_id": args.run_id,
        "created_at": _now_utc(),
        "prediction_status": prediction_status if len(preds) > 0 else "degraded",
        "model_family": model_family,
        "model_version": tr.get("model_version"),
        "feature_schema_version": (tr.get("dataset") or {}).get("feature_schema_version"),
        "model_dir": args.model_dir,
        "missing_required_components": missing_required,
        "predictions_log1p_delta": preds,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote prediction -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


