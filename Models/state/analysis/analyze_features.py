#!/usr/bin/env python3
"""
analyze_features.py — Phase 1 feature analysis of Agent A's 300-video corpus run.

Consumes:
  - content feature table (build_from_rs output parquet)
  - corpus300.json (strata / duration / repo)
  - the local corpus_npz rs/ tree (for Segmenter frame-count analysis)

Emits into Models/state/analysis/:
  - feature_stats.csv          per-feature: %NaN, %const, mean/std/min/max, nunique
  - const_features.csv         features constant across the whole corpus (drop candidates)
  - redundant_pairs.csv        feature-feature |Spearman|>=0.98 pairs (redundancy)
  - segmenter_frames.csv       per-video per-component sampled-frame counts vs duration
  - PNGs: nan_by_component, feat_count_by_component, segmenter_frames_vs_duration
Numbers only; the narrative goes in FEATURE_ANALYSIS_300CORPUS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
OUT = THIS.parent
REPO = THIS.parents[3]
sys.path.insert(0, str(REPO / "Models" / "baseline" / "common"))


def _component_of(col: str) -> str:
    return col.split("__", 1)[0] if "__" in col else "_meta"


def feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if "__" in c]
    rows = []
    n = len(df)
    for c in feat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s)]
        nun = int(finite.nunique())
        rows.append({
            "feature": c,
            "component": _component_of(c),
            "pct_nan": round(100 * (1 - len(finite) / n), 2) if n else 100.0,
            "nunique": nun,
            "is_constant": nun <= 1,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "std": float(finite.std()) if len(finite) else np.nan,
            "min": float(finite.min()) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def redundant_pairs(df: pd.DataFrame, thr: float = 0.98) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if "__" in c]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    # keep non-constant, <50% NaN
    keep = [c for c in feat_cols if np.isfinite(X[c]).sum() > 0.5 * len(df) and X[c].nunique() > 1]
    Xk = X[keep].fillna(X[keep].median())
    if len(keep) < 2:
        return pd.DataFrame(columns=["a", "b", "spearman"])
    corr = Xk.corr(method="spearman").abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v) and v >= thr:
                pairs.append({"a": cols[i], "b": cols[j], "spearman": round(float(v), 4)})
    return pd.DataFrame(pairs).sort_values("spearman", ascending=False)


def segmenter_frames(corpus_npz: Path, corpus_json: Path) -> pd.DataFrame:
    """Per-video per-component sampled-frame count from NPZ frame_indices."""
    meta = {c["video_id"]: c for c in json.loads(corpus_json.read_text())}
    rows = []
    for vdir in sorted(corpus_npz.iterdir()):
        rsd = vdir / "rs"
        if not rsd.is_dir():
            continue
        vid = vdir.name
        dur = meta.get(vid, {}).get("duration")
        cell = meta.get(vid, {}).get("cell", "")
        for cdir in sorted(rsd.iterdir()):
            for npz in cdir.glob("*.npz"):
                try:
                    with np.load(npz, allow_pickle=True) as z:
                        fi = None
                        for k in z.files:
                            if k == "frame_indices" or k.endswith("frame_indices"):
                                fi = z[k]
                                break
                        nfr = int(np.asarray(fi).reshape(-1).size) if fi is not None else np.nan
                except Exception:
                    nfr = np.nan
                rows.append({"video_id": vid, "component": cdir.name, "npz": npz.name,
                             "n_frames": nfr, "duration": dur, "cell": cell})
    return pd.DataFrame(rows)


def main() -> int:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--corpus-json", default=str(REPO / "DataProcessor/docs/corpus_run_report/corpus300.json"))
    ap.add_argument("--corpus-npz", default=str(REPO / "storage/corpus_npz"))
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    print(f"table: {df.shape[0]} videos x {df.shape[1]} cols")

    st = feature_stats(df)
    st.to_csv(OUT / "feature_stats.csv", index=False)
    const = st[st["is_constant"]].sort_values("component")
    const.to_csv(OUT / "const_features.csv", index=False)
    print(f"features: {len(st)}  constant-across-corpus: {len(const)}  "
          f">50%NaN: {int((st['pct_nan']>50).sum())}")

    rp = redundant_pairs(df)
    rp.to_csv(OUT / "redundant_pairs.csv", index=False)
    print(f"redundant |Spearman|>=0.98 pairs: {len(rp)}")

    sf = segmenter_frames(Path(args.corpus_npz), Path(args.corpus_json))
    sf.to_csv(OUT / "segmenter_frames.csv", index=False)
    print(f"segmenter rows: {len(sf)} ({sf['video_id'].nunique()} videos)")

    # ---- PNGs ----
    # NaN by component
    nanc = st.groupby("component")["pct_nan"].mean().sort_values()
    plt.figure(figsize=(9, 6)); plt.barh(nanc.index, nanc.values, color="#e07b7b")
    plt.xlabel("mean %NaN across features"); plt.title("Corpus-300: mean %NaN by component")
    plt.tight_layout(); plt.savefig(OUT / "nan_by_component.png", dpi=110); plt.close()

    # feature count by component (non-const)
    live = st[~st["is_constant"]].groupby("component")["feature"].count().sort_values()
    allc = st.groupby("component")["feature"].count()
    plt.figure(figsize=(9, 6)); plt.barh(live.index, live.values, color="#7c5cff")
    plt.xlabel("non-constant features"); plt.title("Corpus-300: live (non-const) features by component")
    for i, comp in enumerate(live.index):
        plt.text(live[comp], i, f" {live[comp]}/{allc[comp]}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "feat_count_by_component.png", dpi=110); plt.close()

    # segmenter frames vs duration (core_clip)
    cc = sf[(sf["component"] == "core_clip") & sf["n_frames"].notna() & sf["duration"].notna()]
    if len(cc):
        plt.figure(figsize=(8, 6)); plt.scatter(cc["duration"], cc["n_frames"], s=14, alpha=0.6, color="#4cc9f0")
        plt.xlabel("duration (s)"); plt.ylabel("sampled frames (core_clip)")
        plt.title("Segmenter: core_clip sampled frames vs duration")
        plt.tight_layout(); plt.savefig(OUT / "segmenter_frames_vs_duration.png", dpi=110); plt.close()

    print("wrote CSVs + PNGs to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
