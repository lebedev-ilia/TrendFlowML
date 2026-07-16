#!/usr/bin/env python3
"""
Валидационный скрипт clap_extractor:
- U5: golden determinism (2 прогона → diff=0)
- U6: разные длины (5 vs 30 сегментов)
- C1: embedding quality
- C2: embedding_sequence alignment
- C3: segment_mask

Запуск: python3 DataProcessor/clap_synth_validate.py
"""
import os
import sys
import json
import shutil
import subprocess
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AP_VENV_PYTHON = REPO_ROOT / "DataProcessor/AudioProcessor/.ap_venv/bin/python3"
RUN_CLI = REPO_ROOT / "DataProcessor/AudioProcessor/run_cli.py"
FRAMES_DIR_BASE = REPO_ROOT / "storage/frames_dir"
VAL_SCRIPT = REPO_ROOT / "DataProcessor/AudioProcessor/src/extractors/clap_extractor/utils/validate_clap.py"
DP_MODELS_ROOT = str(REPO_ROOT / "DataProcessor/dp_models/bundled_models")
TMP_BASE = Path("/tmp/clap_validate_test")

ENV = {**os.environ,
       "DP_MODELS_ROOT": DP_MODELS_ROOT,
       "HF_HUB_OFFLINE": "1",
       "TRANSFORMERS_OFFLINE": "1"}

results = []

def run_ap(video_id: str, run_id: str, max_segments: int | None = None) -> Path:
    """Запустить AudioProcessor для одного видео, вернуть путь к NPZ."""
    frames_dir = FRAMES_DIR_BASE / video_id
    rs_base = TMP_BASE / run_id
    rs_base.mkdir(parents=True, exist_ok=True)

    # Если нужно ограничить сегменты — патчим временный segments.json
    if max_segments is not None:
        seg_src = frames_dir / "audio/segments.json"
        with open(seg_src) as f:
            segs = json.load(f)
        clap_segs = segs.get("families", {}).get("clap", {}).get("segments", [])
        if len(clap_segs) > max_segments:
            segs["families"]["clap"]["segments"] = clap_segs[:max_segments]
        tmp_frames = TMP_BASE / f"frames_{run_id}_{video_id}"
        if tmp_frames.exists():
            shutil.rmtree(tmp_frames)
        shutil.copytree(frames_dir, tmp_frames)
        with open(tmp_frames / "audio/segments.json", "w") as f:
            json.dump(segs, f)
        frames_dir = tmp_frames

    cmd = [
        str(AP_VENV_PYTHON), str(RUN_CLI),
        "--frames-dir", str(frames_dir),
        "--video-id", video_id,
        "--run-id", run_id,
        "--extractors", "clap",
        "--rs-base", str(rs_base),
        "--platform-id", "youtube",
        "--device", "cuda",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV,
                       cwd=str(REPO_ROOT / "DataProcessor/AudioProcessor"))
    # Найти выходной NPZ
    npzs = list(rs_base.rglob("clap_extractor_features.npz"))
    if not npzs:
        raise RuntimeError(f"NPZ not found for {run_id}. rc={r.returncode}\nSTDOUT={r.stdout[-500:]}\nSTDERR={r.stderr[-500:]}")
    return npzs[0]


def check(name: str, ok: bool, msg: str = ""):
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name}{': ' + msg if msg else ''}")
    results.append((name, ok, msg))


# ============ Подготовка ============
if TMP_BASE.exists():
    shutil.rmtree(TMP_BASE)
TMP_BASE.mkdir(parents=True)

# Найти видео с достаточным числом clap-сегментов
video_ids = []
for vd in FRAMES_DIR_BASE.iterdir():
    seg_path = vd / "audio/segments.json"
    if not seg_path.exists():
        continue
    with open(seg_path) as f:
        segs = json.load(f)
    clap_n = len(segs.get("families", {}).get("clap", {}).get("segments", []))
    if segs.get("audio_present", True) and clap_n >= 5:
        video_ids.append((vd.name, clap_n))

video_ids.sort(key=lambda x: x[1])
print(f"Доступные видео с clap >= 5 сегментов: {video_ids}")

if not video_ids:
    print("❌ Нет доступных видео!")
    sys.exit(1)

vid_short, n_short = video_ids[0]   # короткое (мало сегментов)
vid_long, n_long = video_ids[-1]    # длинное (много сегментов)
print(f"Выбраны: short={vid_short} ({n_short} seg), long={vid_long} ({n_long} seg)")

# ============ U5: Golden ============
print("\n=== U5: Golden детерминизм ===")
try:
    npz1 = run_ap(vid_long, "golden_run1")
    npz2 = run_ap(vid_long, "golden_run2")
    z1 = np.load(str(npz1), allow_pickle=True)
    z2 = np.load(str(npz2), allow_pickle=True)
    emb1 = np.asarray(z1["embedding"], dtype=np.float32)
    emb2 = np.asarray(z2["embedding"], dtype=np.float32)
    seq1 = np.asarray(z1["embedding_sequence"], dtype=np.float32)
    seq2 = np.asarray(z2["embedding_sequence"], dtype=np.float32)
    max_diff = float(np.max(np.abs(emb1 - emb2)))
    seq_diff = float(np.nanmax(np.abs(seq1 - seq2))) if seq1.size else 0.0
    check("U5 embedding diff=0", max_diff == 0.0, f"max|Δ|={max_diff:.2e}")
    check("U5 seq diff=0", seq_diff == 0.0, f"max|Δ|={seq_diff:.2e}")
    z1.close(); z2.close()
except Exception as e:
    check("U5 golden", False, f"EXCEPTION: {e}")

# ============ U6: Разные длины ============
print("\n=== U6: Разные длины ===")
for seg_count in [5, min(n_long, 30)]:
    try:
        npz = run_ap(vid_long, f"u6_n{seg_count}", max_segments=seg_count)
        z = np.load(str(npz), allow_pickle=True)
        meta = z["meta"].item()
        status = meta.get("status")
        n_seg = int(np.asarray(z["segment_start_sec"]).size)
        emb = np.asarray(z["embedding"], dtype=np.float32)
        check(f"U6 N={seg_count} status=ok", status == "ok", f"status={status}")
        check(f"U6 N={seg_count} seg_n match", n_seg == seg_count, f"seg_n={n_seg}")
        check(f"U6 N={seg_count} embedding finite", np.all(np.isfinite(emb)), f"nan={np.isnan(emb).sum()}")
        z.close()
    except Exception as e:
        check(f"U6 N={seg_count}", False, f"EXCEPTION: {e}")

# ============ C1: Embedding quality ============
print("\n=== C1: Embedding quality ===")
try:
    npz = run_ap(vid_long, "c1_quality")
    z = np.load(str(npz), allow_pickle=True)
    emb = np.asarray(z["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    dim = int(emb.shape[0])
    fn = list(np.asarray(z["feature_names"], dtype=object))
    fv = np.asarray(z["feature_values"], dtype=np.float32)
    ep = bool(z["embedding_present"])
    check("C1 dim=512", dim == 512, f"dim={dim}")
    check("C1 norm∈[0.5,1.5]", 0.5 <= norm <= 1.5, f"norm={norm:.4f}")
    check("C1 NaN=0", np.isnan(emb).sum() == 0, f"nan={np.isnan(emb).sum()}")
    check("C1 embedding_present=True", ep, f"ep={ep}")
    check("C1 feature_names count=5", len(fn) == 5, f"fn={fn}")
    check("C1 fv NaN=0", np.isnan(fv).sum() == 0, f"fv_nan={np.isnan(fv).sum()}")
    z.close()
except Exception as e:
    check("C1 quality", False, f"EXCEPTION: {e}")

# ============ C2: embedding_sequence alignment ============
print("\n=== C2: Alignment ===")
try:
    npz = run_ap(vid_long, "c2_align")
    z = np.load(str(npz), allow_pickle=True)
    es = np.asarray(z["embedding_sequence"])
    ss = np.asarray(z["segment_start_sec"])
    sm = np.asarray(z["segment_mask"])
    check("C2 seq shape[0] = seg_n", es.shape[0] == len(ss), f"es.shape={es.shape} seg_n={len(ss)}")
    check("C2 seq shape[1] = 512", es.shape[1] == 512, f"es.shape={es.shape}")
    check("C2 masked rows NaN", np.all(np.isnan(es[~sm])) if np.any(~sm) else True,
          f"unmasked_nan={np.isnan(es[~sm]).all() if np.any(~sm) else 'n/a'}")
    z.close()
except Exception as e:
    check("C2 alignment", False, f"EXCEPTION: {e}")

# ============ C3: segment_mask ============
print("\n=== C3: segment_mask ===")
try:
    npz = run_ap(vid_short, "c3_mask_short")
    z = np.load(str(npz), allow_pickle=True)
    sm = np.asarray(z["segment_mask"])
    check("C3 at least 1 True", sm.sum() >= 1, f"mask_sum={sm.sum()}/{len(sm)}")
    z.close()
except Exception as e:
    check("C3 mask", False, f"EXCEPTION: {e}")

# ============ U1 validate script ============
print("\n=== U1: validate_clap.py на свежем NPZ ===")
try:
    npz = run_ap(vid_long, "u1_validate")
    r = subprocess.run([str(AP_VENV_PYTHON), str(VAL_SCRIPT), str(npz), "--struct"],
                       capture_output=True, text=True)
    check("U1 rc=0 + VALID", r.returncode == 0 and "VALID" in r.stdout,
          f"rc={r.returncode} out={r.stdout.strip()[:60]}")
except Exception as e:
    check("U1 validate", False, f"EXCEPTION: {e}")

# ============ Итог ============
print("\n" + "="*50)
pass_count = sum(1 for _, ok, _ in results if ok)
fail_count = len(results) - pass_count
print(f"ИТОГ: {pass_count}/{len(results)} PASS ({fail_count} FAIL)")
for name, ok, msg in results:
    if not ok:
        print(f"  ❌ {name}: {msg}")

sys.exit(0 if fail_count == 0 else 1)
