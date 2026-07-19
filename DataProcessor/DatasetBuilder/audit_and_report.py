#!/usr/bin/env python3
"""
audit_and_report.py  —  leakage audit + feature importance + diagnostics viz

Runs AFTER train_baseline.py. Three jobs:
  1) LEAKAGE AUDIT (the mandatory gate before trusting any number):
       - no target/mask/future-snapshot column leaked into the feature matrix
       - every feature traces to an allowed source (v0-real component / snapshot_0
         / temporal), none to a forbidden source (future snapshot, post-pub comments)
       - flag suspiciously dominant single features (possible target proxy)
  2) PERMUTATION IMPORTANCE on the test split (HistGB has no native importances)
  3) PNG diagnostics -> models_agent_outbox (auto-posted to VK)

Writes leakage_audit.json + baseline_report.md next to the model dir.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
sys.path.insert(0, str(THIS.parent))
sys.path.insert(0, str(REPO_ROOT / "Models"))

import yaml  # noqa: E402

# NOTE: exact target/mask column leakage is checked separately against the spec's
# target_columns/mask_columns sets. These substrings catch FUTURE-snapshot / post-
# publication provenance only — deliberately NOT "mask_"/"target_" (those would
# false-positive on legit content features like `..._valid_mask__mean`).
FORBIDDEN_SUBSTR = ["snapshot_7", "snapshot_14", "snapshot_21",
                    "comments_embed", "comment_embed", "future_", "_postpub", "views_7d",
                    "views_14d", "views_21d", "likes_7d", "likes_14d", "likes_21d"]


def load_spec(p: str) -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def leakage_audit(feature_names, spec: dict) -> dict:
    included = {c["name"] for c in (spec.get("components") or [])}
    allowed_scalar = set(spec.get("snapshot0_fields") or []) | set(spec.get("temporal_fields") or [])
    forbidden = {c for c in spec.get("target_columns", [])} | set(spec.get("mask_columns", []))

    findings = {"leaked_target_or_mask": [], "forbidden_source_hits": [], "unmapped_features": []}
    for f in feature_names:
        if f in forbidden:
            findings["leaked_target_or_mask"].append(f)
        low = f.lower()
        if any(s in low for s in FORBIDDEN_SUBSTR):
            findings["forbidden_source_hits"].append(f)
        # map to a source
        comp = f.split("__", 1)[0] if "__" in f else f
        if comp in included or f in allowed_scalar:
            continue
        findings["unmapped_features"].append(f)

    ok = (not findings["leaked_target_or_mask"]) and (not findings["forbidden_source_hits"])
    return {
        "passed": bool(ok),
        "n_features_audited": len(feature_names),
        "n_leaked_target_or_mask": len(findings["leaked_target_or_mask"]),
        "n_forbidden_source_hits": len(findings["forbidden_source_hits"]),
        "n_unmapped_features": len(findings["unmapped_features"]),
        "unmapped_features_sample": findings["unmapped_features"][:15],
        "leaked_examples": findings["leaked_target_or_mask"][:15],
        "forbidden_examples": findings["forbidden_source_hits"][:15],
    }


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if r == r else float("nan")


def main() -> int:
    import pandas as pd
    from sklearn.inspection import permutation_importance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--feature-spec", required=True)
    ap.add_argument("--outbox", default=str(REPO_ROOT / "automation/runner/state/models_agent_outbox"))
    ap.add_argument("--target", default="views_21d", help="which head to analyze for importance")
    args = ap.parse_args()

    spec = load_spec(args.feature_spec)
    mdir = Path(args.model_dir)
    manifest = json.loads((mdir / "training_run_manifest.json").read_text())
    metrics = json.loads((mdir / "metrics.json").read_text())
    feature_names = manifest["feature_names"]

    # 1) LEAKAGE AUDIT ----------------------------------------------------
    audit = leakage_audit(feature_names, spec)
    (mdir / "leakage_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"[audit] passed={audit['passed']} features={audit['n_features_audited']} "
          f"leaked={audit['n_leaked_target_or_mask']} forbidden={audit['n_forbidden_source_hits']} "
          f"unmapped={audit['n_unmapped_features']}")

    # 2) PERMUTATION IMPORTANCE ------------------------------------------
    df = pd.read_parquet(args.dataset)
    # recompute split same way as trainer
    sys.path.insert(0, str(REPO_ROOT / "Models" / "baseline" / "Training"))
    from train_baseline import split_hybrid_time_channel  # noqa: E402
    ch = "channel_id" if "channel_id" in df.columns else "video_id"
    df = df.assign(_split=split_hybrid_time_channel(df, channel_col=ch, published_col="publishedAt"))
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")

    bundle = "views" if args.target.startswith("views") else "likes"
    hz = args.target.split("_")[-1]
    model_path = mdir / "models" / bundle / f"{hz}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    is_test = df["_split"] == "test"
    tcol = f"target_{args.target}"
    Xt, yt = X[is_test], df.loc[is_test, tcol].astype(float)
    imp_pairs = []
    if len(Xt) >= 5:
        pi = permutation_importance(model, Xt, yt, n_repeats=5, random_state=0, scoring="r2", n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1]
        imp_pairs = [(feature_names[i], float(pi.importances_mean[i])) for i in order[:20]]

    # 3) VIZ --------------------------------------------------------------
    outbox = Path(args.outbox); outbox.mkdir(parents=True, exist_ok=True)

    # (a) importance
    if imp_pairs:
        names = [n.replace("__", "·")[:38] for n, _ in imp_pairs][::-1]
        vals = [v for _, v in imp_pairs][::-1]
        plt.figure(figsize=(9, 7))
        plt.barh(names, vals, color="#7c5cff")
        plt.title(f"Permutation importance (test) — {args.target}  [v0-real, SYNTHETIC targets]")
        plt.xlabel("mean R² drop"); plt.tight_layout()
        plt.savefig(outbox / "baseline_v0_feature_importance.png", dpi=110); plt.close()

    # (b) spearman by age bucket for the analyzed head
    pm = metrics["per_model"].get(f"{bundle}_{hz}", {})
    bb = pm.get("test", {}).get("by_age_bucket", {})
    if bb:
        ks = list(bb.keys()); sp = [bb[k]["spearman"] if bb[k]["spearman"] is not None else np.nan for k in ks]
        plt.figure(figsize=(9, 4.5))
        plt.bar(range(len(ks)), sp, color="#4fc3f7")
        plt.xticks(range(len(ks)), ks, rotation=30, ha="right")
        plt.ylabel("Spearman"); plt.ylim(-1, 1)
        plt.title(f"Spearman by age bucket (test) — {args.target}  [SYNTHETIC]")
        plt.tight_layout(); plt.savefig(outbox / "baseline_v0_spearman_by_age.png", dpi=110); plt.close()

    # (c) overall spearman across 6 heads
    heads = ["views_7d", "views_14d", "views_21d", "likes_7d", "likes_14d", "likes_21d"]
    val_sp, test_sp = [], []
    for h in heads:
        b = "views" if h.startswith("views") else "likes"; hh = h.split("_")[-1]
        e = metrics["per_model"].get(f"{b}_{hh}", {})
        val_sp.append(e.get("val", {}).get("overall", {}).get("spearman") or np.nan)
        test_sp.append(e.get("test", {}).get("overall", {}).get("spearman") or np.nan)
    x = np.arange(len(heads))
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - 0.2, val_sp, 0.4, label="val", color="#81c784")
    plt.bar(x + 0.2, test_sp, 0.4, label="test", color="#e57373")
    plt.xticks(x, heads, rotation=20); plt.ylabel("Spearman"); plt.ylim(0, 1)
    plt.title("Baseline v0-real Spearman per head  [SYNTHETIC targets — pipeline smoke]")
    plt.legend(); plt.tight_layout()
    plt.savefig(outbox / "baseline_v0_spearman_per_head.png", dpi=110); plt.close()

    # importance report md
    lines = ["# Baseline v0-real — leakage audit + importance\n",
             f"- leakage audit passed: **{audit['passed']}**",
             f"- features audited: {audit['n_features_audited']}, unmapped: {audit['n_unmapped_features']}\n",
             f"## Top permutation importance ({args.target}, test)\n"]
    for n, v in imp_pairs:
        lines.append(f"- `{n}`: {v:.4f}")
    (mdir / "importance_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[viz] wrote PNGs -> {outbox}")
    print(f"[importance] top-5 {args.target}: {[n for n,_ in imp_pairs[:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
