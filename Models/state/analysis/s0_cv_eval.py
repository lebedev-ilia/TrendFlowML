#!/usr/bin/env python3
"""s0_cv_eval.py — robust GroupKFold(channel) CV for a snapshot_0+metadata dataset
(no content features). Gives the definitive large-N "baseline to beat" per head."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ID = {"video_id","channel_id","channelTitle","publishedAt","run_id","platform_id","language"}
HEADS = [("views_7d","mask_7d"),("views_14d","mask_14d"),("views_21d","mask_21d"),
         ("likes_7d","mask_7d"),("likes_14d","mask_14d"),("likes_21d","mask_21d")]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True)
    ap.add_argument("--k",type=int,default=5); ap.add_argument("--max-iter",type=int,default=300)
    a=ap.parse_args()
    df=pd.read_parquet(a.dataset).reset_index(drop=True)
    feats=[c for c in df.columns if c not in ID and not c.startswith(("target_","mask_"))
           and "__" not in c and pd.api.types.is_numeric_dtype(df[c])]
    feats=[c for c in feats if pd.to_numeric(df[c],errors="coerce").nunique()>1]
    print(f"rows={len(df)} channels={df['channel_id'].nunique()} S0 feats={len(feats)}")
    for head,mcol in HEADS:
        sub=df[df[mcol].astype(float)>0].reset_index(drop=True)
        y=sub[f"target_{head}"].astype(float).values
        g=sub["channel_id"].astype(str).fillna(sub["video_id"].astype(str)).values
        X=sub[feats].apply(pd.to_numeric,errors="coerce").values
        pred=np.full(len(y),np.nan)
        for tr,te in GroupKFold(n_splits=min(a.k,len(set(g)))).split(X,y,g):
            med=np.nanmedian(X[tr],axis=0); med=np.where(np.isfinite(med),med,0.0)
            Xtr=np.where(np.isfinite(X[tr]),X[tr],med); Xte=np.where(np.isfinite(X[te]),X[te],med)
            m=HistGradientBoostingRegressor(random_state=1337,max_depth=6,learning_rate=0.05,
                                            max_iter=a.max_iter,l2_regularization=1.0)
            m.fit(Xtr,y[tr]); pred[te]=m.predict(Xte)
        print(f"  {head:11} n={len(sub):5} groups={len(set(g)):5} Spearman={spearmanr(y,pred).correlation:.4f}")

if __name__=="__main__": raise SystemExit(main())
