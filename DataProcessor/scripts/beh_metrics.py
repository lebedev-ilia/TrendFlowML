#!/usr/bin/env python3
"""Метрики behavioral по критериям C1-C4/U2-U3 для набора NPZ. Вывод JSON."""
import sys, json, glob, os
import numpy as np

MAIN_SEQ = "seq_num_hands"  # основной блок: NaN строго ↔ ¬landmarks_present
POSE_SEQ = ["seq_arm_openness", "seq_pose_expansion", "seq_body_lean_angle", "seq_balance_offset", "seq_shoulder_angle"]
MOUTH_SEQ = ["seq_mouth_width_norm", "seq_mouth_height_norm", "seq_mouth_area_norm", "seq_mouth_open_ratio", "seq_speech_activity_proxy"]

def load(npz):
    d = np.load(npz, allow_pickle=True)
    meta = d["meta"].reshape(-1)[0] if "meta" in d.files else {}
    agg = d["aggregated"].reshape(-1)[0] if "aggregated" in d.files else {}
    return d, meta, agg

def analyze(npz):
    d, meta, agg = load(npz)
    fi = np.asarray(d["frame_indices"]).astype(np.int64)
    ts = np.asarray(d["times_s"], dtype=np.float64)
    lp = np.asarray(d["landmarks_present"]).astype(bool)
    N = int(fi.size); nlp = int(lp.sum())
    seq_keys = [k for k in d.files if k.startswith("seq_")]
    # U2 ось
    u2 = {"monotonic_fi": bool(np.all(np.diff(fi) > 0)) if N > 1 else True,
          "times_nondecr": bool(np.all(np.diff(ts) >= -1e-6)) if N > 1 else True,
          "times_nan_pct": float(np.mean(~np.isfinite(ts)) * 100),
          "tsnorm_in01": None}
    if "seq_timestamp_norm" in d.files:
        tn = np.asarray(d["seq_timestamp_norm"], dtype=np.float64)
        fin = np.isfinite(tn)
        u2["tsnorm_in01"] = bool(np.all((tn[fin] >= -1e-6) & (tn[fin] <= 1 + 1e-6)))
        u2["tsnorm_nan_pct"] = float(np.mean(~fin) * 100)
    # U3 health
    inf_count = 0
    for k in seq_keys:
        a = np.asarray(d[k], dtype=np.float64)
        inf_count += int(np.sum(np.isinf(a)))
    # gesture prob сумма на кадрах с landmarks
    gk = [k for k in d.files if k.startswith("seq_gesture_prob_")]
    gsum_ok = None
    if gk and nlp > 0:
        M = np.stack([np.asarray(d[k], dtype=np.float64) for k in gk], axis=1)  # (N, G)
        row_sum = M[lp].sum(axis=1)
        finite_rows = np.isfinite(row_sum)
        gsum_ok = {"n_gestures": len(gk),
                   "sum_close_1_pct": float(np.mean(np.abs(row_sum[finite_rows] - 1.0) < 1e-3) * 100) if finite_rows.any() else None,
                   "prob_in01": bool(np.all((M[np.isfinite(M)] >= -1e-9) & (M[np.isfinite(M)] <= 1 + 1e-9)))}
    # C1 основной seq строго ↔ ¬lp
    c1 = None
    if MAIN_SEQ in d.files:
        a = np.asarray(d[MAIN_SEQ], dtype=np.float64)
        fin = np.isfinite(a)
        c1 = {"nan_at_lp_true": int(np.sum(~fin & lp)),   # ожид 0
              "finite_at_lp_false": int(np.sum(fin & ~lp)), # ожид 0
              "strict_match": bool(np.sum(~fin & lp) == 0 and np.sum(fin & ~lp) == 0)}
    # C2 вторичные опоры: доля finite при lp=True
    def finite_at_lp(keys):
        out = {}
        for k in keys:
            if k in d.files:
                a = np.asarray(d[k], dtype=np.float64)
                fin_lp = np.isfinite(a) & lp
                out[k] = {"finite_at_lp": int(fin_lp.sum()), "lp_true": nlp,
                          "pct": round(100 * fin_lp.sum() / nlp, 1) if nlp else None}
        return out
    c2 = {"pose": finite_at_lp(POSE_SEQ), "mouth": finite_at_lp(MOUTH_SEQ)}
    # C3 body_lean не константа (по конечным кадрам)
    c3 = None
    if "seq_body_lean_angle" in d.files:
        bl = np.asarray(d["seq_body_lean_angle"], dtype=np.float64)
        blf = bl[np.isfinite(bl)]
        c3 = {"n_finite": int(blf.size), "std": float(np.std(blf)) if blf.size else None,
              "min": float(blf.min()) if blf.size else None, "max": float(blf.max()) if blf.size else None,
              "all_eq_1": bool(blf.size > 0 and np.allclose(blf, 1.0))}
    # C4 aggregated
    nan_fields = []
    scalar_fields = 0
    for k, v in (agg.items() if isinstance(agg, dict) else []):
        if isinstance(v, (int, float)):
            scalar_fields += 1
            if isinstance(v, float) and not np.isfinite(v):
                nan_fields.append(k)
    c4 = {"n_top_fields": len(agg) if isinstance(agg, dict) else 0,
          "n_scalar": scalar_fields, "nan_fields": nan_fields,
          "avg_engagement": agg.get("avg_engagement") if isinstance(agg, dict) else None,
          "gesture_rate_per_sec": agg.get("gesture_rate_per_sec") if isinstance(agg, dict) else None}
    return {"npz": npz, "N": N, "landmarks_present_true": nlp,
            "status": meta.get("status"), "empty_reason": meta.get("empty_reason"),
            "producer_version": meta.get("producer_version"),
            "U2": u2, "U3": {"inf_in_seq": inf_count, "gesture": gsum_ok},
            "C1": c1, "C2": c2, "C3_body_lean": c3, "C4_agg": c4}

if __name__ == "__main__":
    pats = sys.argv[1:]
    files = []
    for p in pats:
        files += glob.glob(p)
    res = [analyze(f) for f in sorted(files)]
    print(json.dumps(res, ensure_ascii=False, indent=1))
