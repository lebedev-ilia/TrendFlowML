#!/usr/bin/env python3
from __future__ import annotations

"""
Smoke E2E for baseline:
  dataset -> train -> eval(regression_mini) -> predict(one run)

This is meant as a sanity check / CI-friendly harness (no servers).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Models.baseline.Training.train_baseline import main as train_main  # noqa: E402
from Models.baseline.Training.evaluate_baseline import main as eval_main  # noqa: E402
from Models.baseline.Inference.predict_baseline import main as predict_main  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Baseline smoke E2E")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset-metadata", type=str, required=True)
    p.add_argument("--feature-spec", type=str, required=True)
    p.add_argument("--work-dir", type=str, required=True, help="Output directory for artifacts/eval")
    p.add_argument("--model-family", type=str, default="sklearn", choices=["catboost", "lightgbm", "sklearn"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rs-base", type=str, required=True, help="result_store base for inference smoke")
    p.add_argument("--platform-id", type=str, required=True)
    p.add_argument("--video-id", type=str, required=True)
    p.add_argument("--run-id", type=str, required=True)
    args = p.parse_args()

    work = Path(args.work_dir)
    model_dir = work / "model"
    eval_dir = work / "eval_regression_mini"
    pred_path = work / "prediction.json"
    work.mkdir(parents=True, exist_ok=True)

    # Train
    sys.argv = [
        "train_baseline.py",
        "--dataset",
        args.dataset,
        "--dataset-metadata",
        args.dataset_metadata,
        "--feature-spec",
        args.feature_spec,
        "--out-dir",
        str(model_dir),
        "--model-family",
        args.model_family,
        "--seed",
        str(args.seed),
    ]
    train_main()

    # Golden sets are optional; evaluate will sample deterministically if not provided.
    sys.argv = [
        "evaluate_baseline.py",
        "--dataset",
        args.dataset,
        "--model-dir",
        str(model_dir),
        "--out-dir",
        str(eval_dir),
        "--eval-set",
        "regression_mini",
        "--seed",
        str(args.seed),
    ]
    eval_main()

    # Predict one run
    sys.argv = [
        "predict_baseline.py",
        "--rs-base",
        args.rs_base,
        "--platform-id",
        args.platform_id,
        "--video-id",
        args.video_id,
        "--run-id",
        args.run_id,
        "--model-dir",
        str(model_dir),
        "--out-json",
        str(pred_path),
        "--feature-spec",
        args.feature_spec,
        "--enforce-required-components",
        "--required-policy",
        "degraded",
    ]
    predict_main()

    print(f"[ok] smoke_e2e finished -> {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


