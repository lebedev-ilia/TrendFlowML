#!/usr/bin/env python3
"""
residual_content_signal.py — does CONTENT carry signal BEYOND snapshot_0?

The v2 CV showed content doesn't improve accuracy at N=291. But that could be pure
p>>n (too few rows to *fit* 500 features), not "content is uninformative". This test
separates the two:

  1. Fit an S0-only (snapshot_0 + metadata) model with out-of-fold CV -> OOF predictions.
  2. residual = y - pred_S0  (what snapshot_0 could NOT explain).
  3. For each content feature, Spearman(feature, residual) across videos.

If some content features correlate with the residual, content carries incremental
signal that MORE DATA would let a model exploit -> scaling is worth it, and these are
the features to watch. If all correlations ~0, content is genuinely uninformative for
this target regardless of N.

Output: residual_content_signal_<head>.csv (ranked) + console top-15.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

THIS = Path(__file__).resolve()
OUT = THIS.parent
ID_COLS = {"video_id", "channel_id", "channelTitle", "publishedAt", "run_id",
           "platform_id", "language", "_n_features", "_n_components_present",
           "manifest_path", "config_hash", "manifest_created_at"}


def _usable(df, cols):
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        f = s[s.notna()]
        if len(f) and f.nunique() > 1:
            keep.append(c)
    return keep


def oof_predict(X, y, groups, k=5, max_iter=300):
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(k, len(set(groups))))
    for tr, te in gkf.split(X, y, groups):
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med)
        Xte = np.where(np.isfinite(X[te]), X[te], med)
        m = HistGradientBoostingRegressor(random_state=1337, max_depth=6,
                                          learning_rate=0.05, max_iter=max_iter,
                                          l2_regularization=1.0)
        m.fit(Xtr, y[tr])
        pred[te] = m.predict(Xte)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heads", nargs="+", default=["views_21d", "likes_21d"])
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    content = _usable(df, [c for c in df.columns if "__" in c])
    s0 = _usable(df, [c for c in df.columns if c not in ID_COLS and "__" not in c
                      and not c.startswith(("target_", "mask_"))
                      and pd.api.types.is_numeric_dtype(df[c])])
    print(f"S0={len(s0)} content={len(content)}")

    for head in args.heads:
        tcol, mcol = f"target_{head}", f"mask_{head.split('_')[-1]}"
        if head.startswith("likes"):
            mcol = f"mask_{head.split('_')[-1]}"
        sub = df[df[mcol].astype(float) > 0].reset_index(drop=True)
        y = sub[tcol].astype(float).values
        groups = sub["channel_id"].astype(str).fillna(sub["video_id"].astype(str)).values
        Xs0 = sub[s0].apply(pd.to_numeric, errors="coerce").values
        pred = oof_predict(Xs0, y, groups)
        resid = y - pred
        base_rho = spearmanr(y, pred).correlation

        recs = []
        Xc = sub[content].apply(pd.to_numeric, errors="coerce")
        for c in content:
            col = Xc[c].values
            m = np.isfinite(col) & np.isfinite(resid)
            if m.sum() < 30 or np.unique(col[m]).size < 2:
                continue
            rho = spearmanr(col[m], resid[m]).correlation
            # also raw corr with target (for context)
            rho_y = spearmanr(col[m], y[m]).correlation
            recs.append({"feature": c, "component": c.split("__")[0],
                         "spearman_vs_residual": round(float(rho), 4),
                         "abs_resid": abs(float(rho)),
                         "spearman_vs_target": round(float(rho_y), 4)})
        r = pd.DataFrame(recs).sort_values("abs_resid", ascending=False)
        r.drop(columns="abs_resid").to_csv(OUT / f"residual_content_signal_{head}.csv", index=False)
        print(f"\n=== {head}: S0 OOF Spearman={base_rho:.3f}  (residual = unexplained) ===")
        print(f"top content features by |Spearman(feature, residual)|:")
        for _, row in r.head(15).iterrows():
            print(f"  {row['feature'][:52]:52} resid={row['spearman_vs_residual']:+.3f}  raw_y={row['spearman_vs_target']:+.3f}")
        # summary: how many content feats exceed |0.15| vs residual
        strong = (r["abs_resid"] >= 0.15).sum()
        print(f"  content features with |resid corr|>=0.15: {strong}/{len(r)}")


if __name__ == "__main__":
    raise SystemExit(main())
