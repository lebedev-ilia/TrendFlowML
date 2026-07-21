#!/usr/bin/env python3
"""
v2_cv_experiment.py — robust (GroupKFold-by-channel) test of whether CONTENT features
help at N=291 once we (a) drop constant/redundant content dims and (b) select a lean
top-K per fold. Replaces the noisy single 80/10/10 split (test n~30) of exp_0005/0006.

Feature sets compared, per head:
  S0        : snapshot_0 + metadata numeric (the ~22 non-content features)
  S0+lean   : S0 + top-K content features (redundancy-pruned pool, |Spearman(feat,y)|
              selected ON THE TRAIN FOLD ONLY -> no selection leakage)
  S0+full   : S0 + all (non-constant) content features (the p>>n case)

Evaluation: 5-fold GroupKFold on channel_id; predictions pooled across folds, then
Spearman on the pooled (y_true, y_pred). Emits v2_cv_results.csv + v2_cv_compare.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

THIS = Path(__file__).resolve()
OUT = THIS.parent
REPO = THIS.parents[3]

ID_COLS = {"video_id", "channel_id", "channelTitle", "publishedAt", "run_id",
           "platform_id", "language", "_n_features", "_n_components_present",
           "manifest_path", "config_hash", "manifest_created_at"}
HEADS = [("views_7d", "target_views_7d", "mask_7d"),
         ("views_14d", "target_views_14d", "mask_14d"),
         ("views_21d", "target_views_21d", "mask_21d"),
         ("likes_7d", "target_likes_7d", "mask_7d"),
         ("likes_14d", "target_likes_14d", "mask_14d"),
         ("likes_21d", "target_likes_21d", "mask_21d")]


def _prune_redundant(content_cols, redundant_csv, thr=0.98):
    """Union-Find over |Spearman|>=thr pairs; keep 1 representative per cluster."""
    if not Path(redundant_csv).exists():
        return content_cols
    rp = pd.read_csv(redundant_csv)
    rp = rp[rp["spearman"] >= thr]
    parent = {c: c for c in content_cols}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    for _, r in rp.iterrows():
        union(r["a"], r["b"])
    reps = {}
    for c in content_cols:
        reps.setdefault(find(c), c)
    kept = sorted(set(reps.values()))
    return kept


def _fit_eval(X_tr, y_tr, X_te, seed=1337, max_iter=200):
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(X_tr), X_tr, med)
    Xte = np.where(np.isfinite(X_te), X_te, med)
    m = HistGradientBoostingRegressor(random_state=seed, max_depth=6,
                                      learning_rate=0.05, max_iter=max_iter,
                                      l2_regularization=1.0)
    m.fit(Xtr, y_tr)
    return m.predict(Xte)


def _usable(df, cols):
    """drop constant / all-nan cols over the whole df (unsupervised, no leak)."""
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        f = s[s.notna()]
        if len(f) and f.nunique() > 1:
            keep.append(c)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--redundant", default=str(OUT / "redundant_pairs.csv"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--max-iter", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    content_all = [c for c in df.columns if "__" in c]
    s0_all = [c for c in df.columns if c not in ID_COLS and "__" not in c
              and not c.startswith(("target_", "mask_"))
              and pd.api.types.is_numeric_dtype(df[c])]
    content_all = _usable(df, content_all)
    s0_all = _usable(df, s0_all)
    content_pruned = _prune_redundant(content_all, args.redundant)
    print(f"features: S0={len(s0_all)} content_all={len(content_all)} "
          f"content_pruned(redundancy)={len(content_pruned)}")

    groups_all = df["channel_id"].astype(str).fillna(df["video_id"].astype(str)).values
    rows = []

    for name, tcol, mcol in HEADS:
        sub = df[df[mcol].astype(float) > 0].reset_index(drop=True)
        if len(sub) < 40:
            continue
        y = sub[tcol].astype(float).values
        groups = sub["channel_id"].astype(str).fillna(sub["video_id"].astype(str)).values
        n_groups = len(set(groups))
        k = min(args.k, n_groups)
        gkf = GroupKFold(n_splits=k)

        # pooled predictions per feature-set
        pools = {s: (np.zeros(len(sub)) + np.nan) for s in ["S0", "S0+lean", "S0+full"]}
        Xs0 = sub[s0_all].apply(pd.to_numeric, errors="coerce").values
        Xc_all = sub[content_all].apply(pd.to_numeric, errors="coerce").values
        Xc_pr = sub[content_pruned].apply(pd.to_numeric, errors="coerce").values

        for tr, te in gkf.split(sub, y, groups):
            # S0
            pools["S0"][te] = _fit_eval(Xs0[tr], y[tr], Xs0[te], max_iter=args.max_iter)
            # S0+full
            Xf_tr = np.hstack([Xs0[tr], Xc_all[tr]]); Xf_te = np.hstack([Xs0[te], Xc_all[te]])
            pools["S0+full"][te] = _fit_eval(Xf_tr, y[tr], Xf_te, max_iter=args.max_iter)
            # S0+lean: select top-K pruned content by |Spearman(feat,y)| on TRAIN only
            scores = []
            ytr = y[tr]
            for j in range(Xc_pr.shape[1]):
                col = Xc_pr[tr, j]
                mask = np.isfinite(col)
                if mask.sum() < 10 or np.unique(col[mask]).size < 2:
                    scores.append(0.0); continue
                rho = spearmanr(col[mask], ytr[mask]).correlation
                scores.append(0.0 if rho != rho else abs(rho))
            top = np.argsort(scores)[::-1][:args.topk]
            Xl_tr = np.hstack([Xs0[tr], Xc_pr[tr][:, top]])
            Xl_te = np.hstack([Xs0[te], Xc_pr[te][:, top]])
            pools["S0+lean"][te] = _fit_eval(Xl_tr, y[tr], Xl_te, max_iter=args.max_iter)

        rec = {"head": name, "n": len(sub), "n_groups": n_groups}
        for s in pools:
            rho = spearmanr(y, pools[s]).correlation
            rec[s] = round(float(rho), 4)
        rows.append(rec)
        print(f"  {name:11} n={len(sub):3} groups={n_groups:3} | "
              f"S0={rec['S0']:.3f}  S0+lean={rec['S0+lean']:.3f}  S0+full={rec['S0+full']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "v2_cv_results.csv", index=False)

    # PNG
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(len(res)); w = 0.26
    plt.figure(figsize=(10, 5.3))
    plt.bar(x - w, res["S0"], w, label="S0 (snap0+meta)", color="#4cc9f0")
    plt.bar(x, res["S0+lean"], w, label=f"S0+lean top-{args.topk}", color="#7c5cff")
    plt.bar(x + w, res["S0+full"], w, label="S0+full content", color="#e07b7b")
    plt.xticks(x, res["head"], rotation=20); plt.ylabel("pooled CV Spearman"); plt.ylim(0, 1)
    plt.title(f"Corpus-300 v2: {args.k}-fold GroupKFold(channel) — does lean content help?")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "v2_cv_compare.png", dpi=115); plt.close()
    print("wrote v2_cv_results.csv + v2_cv_compare.png")


if __name__ == "__main__":
    raise SystemExit(main())
