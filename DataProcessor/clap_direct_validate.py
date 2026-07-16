#!/usr/bin/env python3
"""
Прямая валидация CLAPExtractor без run_cli.py.
Использует реальный audio.wav из storage/frames_dir.
"""
import os, sys, json, time
import numpy as np
from pathlib import Path

# Пути
REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["DP_MODELS_ROOT"] = str(REPO_ROOT / "DataProcessor/dp_models/bundled_models")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, str(REPO_ROOT / "DataProcessor"))
AP_SRC = REPO_ROOT / "DataProcessor/AudioProcessor/src"
sys.path.insert(0, str(AP_SRC))
# Нужно чтобы import src.* работал из AudioProcessor директории
sys.path.insert(0, str(REPO_ROOT / "DataProcessor/AudioProcessor"))

import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from src.extractors.clap_extractor import CLAPExtractor

# Найти видео
FRAMES_BASE = REPO_ROOT / "storage/frames_dir"
video_data = []
for vd in sorted(FRAMES_BASE.iterdir()):
    seg_path = vd / "audio/segments.json"
    audio_path = vd / "audio/audio.wav"
    if not seg_path.exists() or not audio_path.exists():
        continue
    with open(seg_path) as f:
        segs = json.load(f)
    clap_segs = segs.get("families", {}).get("clap", {}).get("segments", [])
    if segs.get("audio_present", True) and len(clap_segs) >= 5:
        video_data.append((vd.name, clap_segs, str(audio_path)))

video_data.sort(key=lambda x: len(x[1]))
print(f"Видео: {[(v, len(s)) for v,s,_ in video_data]}")

results = []
def check(name, ok, msg=""):
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name}{': ' + msg if msg else ''}")
    results.append((name, ok, msg))

# === Загрузка модели ===
print("\n=== Загрузка CLAP (1.8GB) ===")
t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = CLAPExtractor(device=device, sample_rate=48000)
extractor._load_model()
print(f"Загружено за {time.time()-t0:.1f}s")

# Выбираем видео
vid_name, segs_long, audio_long = video_data[-1]
vid_short, segs_short, audio_short = video_data[0]
print(f"long={vid_name} ({len(segs_long)} seg), short={vid_short} ({len(segs_short)} seg)")

# === U5: Golden ===
print("\n=== U5: Golden (2 прогона) ===")
r1 = extractor.run_segments(audio_long, "/tmp", segs_long)
r2 = extractor.run_segments(audio_long, "/tmp", segs_long)
emb1 = np.asarray(r1.payload["embedding"], dtype=np.float32)
emb2 = np.asarray(r2.payload["embedding"], dtype=np.float32)
diff = float(np.max(np.abs(emb1 - emb2)))
seq1 = np.asarray(r1.payload["embedding_sequence"], dtype=np.float32)
seq2 = np.asarray(r2.payload["embedding_sequence"], dtype=np.float32)
seq_diff = float(np.nanmax(np.abs(seq1 - seq2)))
check("U5 embedding diff=0", diff == 0.0, f"max|Δ|={diff:.2e}")
check("U5 seq diff=0", seq_diff == 0.0, f"max|Δ|={seq_diff:.2e}")

# === U6: Разные длины ===
print("\n=== U6: Разные длины ===")
for n in [5, min(len(segs_long), 30)]:
    seg_subset = segs_long[:n]
    r = extractor.run_segments(audio_long, "/tmp", seg_subset)
    if r.success:
        emb = np.asarray(r.payload["embedding"], dtype=np.float32)
        seg_n = len(r.payload["segment_start_sec"])
        check(f"U6 N={n} ok", r.success, f"seg_n={seg_n}")
        check(f"U6 N={n} embedding finite", np.all(np.isfinite(emb)), f"nan={np.isnan(emb).sum()}")
    else:
        check(f"U6 N={n}", False, f"error={r.error}")

# === C1: Embedding quality ===
print("\n=== C1: Embedding quality ===")
emb = emb1
norm = float(np.linalg.norm(emb))
check("C1 dim=512", emb.shape[0] == 512, f"dim={emb.shape[0]}")
check("C1 norm∈[0.5,1.5]", 0.5 <= norm <= 1.5, f"norm={norm:.4f}")
check("C1 NaN=0", np.isnan(emb).sum() == 0)
print(f"    clap_norm={r1.payload.get('clap_norm'):.4f}")
print(f"    clap_magnitude_mean={r1.payload.get('clap_magnitude_mean'):.4f}")
print(f"    segments_count={r1.payload.get('segments_count')}")

# === C2: Alignment ===
print("\n=== C2: embedding_sequence alignment ===")
es = seq1
ss = np.asarray(r1.payload["segment_start_sec"])
sm = np.asarray(r1.payload["segment_mask"])
check("C2 seq N=seg_n", es.shape[0] == len(ss), f"es={es.shape[0]} seg={len(ss)}")
check("C2 seq D=512", es.shape[1] == 512, f"D={es.shape[1]}")
masked_rows = es[~sm]
if len(masked_rows) > 0:
    check("C2 masked=NaN", np.all(np.isnan(masked_rows)), f"n_masked={len(masked_rows)}")
else:
    check("C2 all segments valid", True, f"mask_sum={sm.sum()}/{len(sm)}")

# === C3: segment_mask ===
print("\n=== C3: segment_mask ===")
check("C3 mask_sum>=1", sm.sum() >= 1, f"sum={sm.sum()}/{len(sm)}")

# === Итог ===
print("\n" + "="*50)
p = sum(1 for _,ok,_ in results if ok)
f = len(results)-p
print(f"ИТОГ: {p}/{len(results)} PASS ({f} FAIL)")
for n,ok,m in results:
    if not ok:
        print(f"  ❌ {n}: {m}")

sys.exit(0 if f==0 else 1)
