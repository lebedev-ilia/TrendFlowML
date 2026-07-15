#!/usr/bin/env python3
"""Хелпер валидации color_light (запускать на поде, venv c numpy).

Подкоманды:
  nan   <npz>                  — NaN/Inf/const-статы по frame_compact_features + gini/aesthetic из video_features.
  diff  <npzA> <npzB>          — golden max|Δ| по frame_compact_features + aggregated.frame_compact (CPU-детерм).
  cross <npz1> <npz2> ...       — межвидовая вариация: mean компакт-dim по роликам, флаг «константа между видео».
  empty <scene_npz> <out_npz>  — крафт edge: сдвиг индексов всех сцен на +999999 (нет пересечения → after_filt_empty).
"""
import sys, numpy as np

FRAME_COMPACT_KEYS = [
    "hue_mean_norm","hue_std_norm","hue_entropy_weighted","sat_mean_norm","val_mean_norm",
    "L_mean_norm","global_contrast_norm","local_contrast_mean_norm","colorfulness_norm",
    "skin_tone_ratio","overexposed_ratio","underexposed_ratio","vignetting_score_norm",
    "soft_light_prob","dominant_lab_a_norm","dominant_lab_b_norm",
]

def load(p):
    return np.load(p, allow_pickle=True)

def get_compact(z):
    return np.asarray(z["frame_compact_features"], dtype=np.float64)

def cmd_nan(npz):
    z = load(npz)
    x = get_compact(z)
    print(f"file={npz}")
    print(f"compact shape={x.shape} dtype={np.asarray(z['frame_compact_features']).dtype}")
    names = [str(s) for s in np.asarray(z.get("frame_compact_feature_names", FRAME_COMPACT_KEYS))]
    print(f"names==FRAME_COMPACT_KEYS: {names == FRAME_COMPACT_KEYS}")
    if x.size:
        nan = int(np.isnan(x).sum()); inf = int(np.isinf(x).sum())
        print(f"NaN={nan} ({100*nan/x.size:.4f}%)  Inf={inf} ({100*inf/x.size:.4f}%)")
        stds = np.nanstd(x, axis=0)
        alive = int((stds > 0).sum())
        print(f"per-dim std>0: {alive}/{x.shape[1]}")
        for i, k in enumerate(FRAME_COMPACT_KEYS):
            print(f"  [{i:2d}] {k:26s} std={stds[i]:.5f} min={np.nanmin(x[:,i]):.4f} max={np.nanmax(x[:,i]):.4f}")
    # video_features gini + aesthetic
    vf = z.get("video_features")
    if vf is not None:
        try:
            vfd = vf.item() if hasattr(vf, "item") else dict(vf)
        except Exception:
            vfd = {}
        for k in ["color_distribution_gini","color_distribution_entropy","nima_mean","nima_std",
                  "laion_mean","laion_std","cinematic_lighting_score","professional_look_score",
                  "hue_mean_mean","hue_mean_std"]:
            if k in vfd:
                v = vfd[k]
                fin = np.isfinite(v) if isinstance(v,(int,float)) else "n/a"
                print(f"  vf.{k} = {v}  finite={fin}")
        nan_keys = [k for k,v in vfd.items() if isinstance(v,(int,float)) and not np.isfinite(v)]
        print(f"video_features: total={len(vfd)} NaN/Inf-keys={len(nan_keys)}: {sorted(nan_keys)}")

def _agg_arr(z):
    agg = z["aggregated"].item() if hasattr(z["aggregated"], "item") else z["aggregated"]
    fc = agg["frame_compact"]
    return np.concatenate([np.asarray(fc[k],dtype=np.float64).ravel() for k in ["mean","std","p25","p50","p75"]])

def cmd_diff(a, b):
    za, zb = load(a), load(b)
    xa, xb = get_compact(za), get_compact(zb)
    print(f"shapes: {xa.shape} vs {xb.shape}")
    if xa.shape == xb.shape:
        d = np.nan_to_num(xa) - np.nan_to_num(xb)
        print(f"compact max|Δ| = {np.max(np.abs(d)) if d.size else 0.0:.3e}")
        print(f"NaN-mask equal: {np.array_equal(np.isnan(xa), np.isnan(xb))}")
    else:
        print("SHAPE MISMATCH")
    try:
        aa, ab = _agg_arr(za), _agg_arr(zb)
        da = np.nan_to_num(aa) - np.nan_to_num(ab)
        print(f"aggregated.frame_compact max|Δ| = {np.max(np.abs(da)) if da.size else 0.0:.3e}")
    except Exception as e:
        print(f"agg diff err: {e}")

def cmd_cross(npzs):
    means = []
    for p in npzs:
        x = get_compact(load(p))
        means.append(np.nanmean(x, axis=0) if x.size else np.full(16, np.nan))
    M = np.vstack(means)
    print(f"videos={len(npzs)}")
    cross_std = np.nanstd(M, axis=0)
    varying = int((cross_std > 1e-6).sum())
    print(f"cross-video: dims with std>1e-6: {varying}/16")
    for i,k in enumerate(FRAME_COMPACT_KEYS):
        print(f"  [{i:2d}] {k:26s} cross-std={cross_std[i]:.5f} vals={np.round(M[:,i],4).tolist()}")

def cmd_empty(scene_npz, out_npz):
    z = load(scene_npz)
    d = {k: z[k] for k in z.files}
    scenes = d.get("scenes")
    scenes = scenes.item() if hasattr(scenes, "item") else scenes
    patched = {}
    for sid, sc in dict(scenes).items():
        sc = dict(sc)
        if "indices" in sc and sc["indices"] is not None:
            sc["indices"] = [int(i)+999999 for i in list(sc["indices"])]
        patched[sid] = sc
    d["scenes"] = np.array(patched, dtype=object)
    np.savez(out_npz, **d)
    print(f"wrote {out_npz}: сдвинул индексы {len(patched)} сцен на +999999")

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "nan": cmd_nan(sys.argv[2])
    elif c == "diff": cmd_diff(sys.argv[2], sys.argv[3])
    elif c == "cross": cmd_cross(sys.argv[2:])
    elif c == "empty": cmd_empty(sys.argv[2], sys.argv[3])
    else: print("unknown", c); sys.exit(2)
