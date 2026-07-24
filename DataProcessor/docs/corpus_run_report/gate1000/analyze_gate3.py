#!/usr/bin/env python3
"""Анализ Gate 3 (1000 видео) из report1000 CSV → статистика + графики.
Запуск: <venv-с-mpl+pandas>/bin/python analyze_gate3.py"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "report1000")
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
OUTBOX = "/home/ilya/Рабочий стол/TrendFlowML/automation/runner/state/deepdive_outbox"
os.makedirs(OUTBOX, exist_ok=True)

pvc = pd.read_csv(os.path.join(REP, "per_video_component.csv"))
pv  = pd.read_csv(os.path.join(REP, "per_video.csv"))

# порядок стадий пайплайна
ORDER = ["download","segmenter","core_clip","core_depth_midas","core_optical_flow",
         "cut_detection","scene_classification","video_pacing","uniqueness"]
SHORT = {"download":"download","segmenter":"segment","core_clip":"clip","core_depth_midas":"depth",
         "core_optical_flow":"flow","cut_detection":"cut","scene_classification":"scene",
         "video_pacing":"pacing","uniqueness":"uniq"}
comps = [c for c in ORDER if c in set(pvc["component"])]
ok = pvc[pvc["status"]=="ok"].copy()

def stat(col):
    d={}
    for c in comps:
        v=pd.to_numeric(ok[ok["component"]==c][col],errors="coerce").dropna()
        d[c]=v
    return d

# ---- FIG 1: разбивка времени по компонентам (p50/p95) ----
wall=stat("wall_s")
p50=[wall[c].median() if len(wall[c]) else 0 for c in comps]
p95=[wall[c].quantile(.95) if len(wall[c]) else 0 for c in comps]
x=np.arange(len(comps))
fig,ax=plt.subplots(figsize=(11,5.5))
ax.bar(x-0.2,p50,0.4,label="p50",color="#6c5ce7")
ax.bar(x+0.2,p95,0.4,label="p95",color="#a29bfe")
for i,(a,b) in enumerate(zip(p50,p95)):
    ax.text(i-0.2,a,f"{a:.0f}",ha="center",va="bottom",fontsize=8)
    ax.text(i+0.2,b,f"{b:.0f}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in comps],rotation=20)
ax.set_ylabel("wall-time, с/видео"); ax.set_title("Gate 3 (1000 видео): время по компонентам\nflow+cut+clip+depth — 4 самых дорогих; сумма ≈ 306с/видео (p50)")
ax.legend(); ax.grid(axis="y",alpha=.3); fig.tight_layout()
f1=os.path.join(OUT,"gate3_component_time.png"); fig.savefig(f1,dpi=120); plt.close(fig)

# ---- FIG 2: CPU% vs GPU-util по компонентам (CPU-bound доказательство) ----
cpu=stat("cpu_pct"); gpu=stat("gpu_util_mean")
cpu_p50=[cpu[c].median() if len(cpu[c]) else np.nan for c in comps]
gpu_p50=[gpu[c].median() if len(gpu[c]) else np.nan for c in comps]
fig,ax=plt.subplots(figsize=(11,5.5))
ax.bar(x-0.2,cpu_p50,0.4,label="CPU% (p50)",color="#e17055")
ax.bar(x+0.2,gpu_p50,0.4,label="GPU util% (mean, p50)",color="#00b894")
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in comps],rotation=20)
ax.set_ylabel("%"); ax.set_title("Gate 3: CPU vs GPU по компонентам\nпайплайн CPU-bound (GPU util низкий) → дешёвая GPU + высокий N оправданы")
ax.legend(); ax.grid(axis="y",alpha=.3); fig.tight_layout()
f2=os.path.join(OUT,"gate3_cpu_vs_gpu.png"); fig.savefig(f2,dpi=120); plt.close(fig)

# ---- FIG 3: распределение total wall / видео ----
tw=pd.to_numeric(pv["total_wall_s"],errors="coerce").dropna()
fig,ax=plt.subplots(figsize=(10,5))
ax.hist(tw,bins=50,color="#0984e3",alpha=.8)
for q,lab,col in [(.5,"p50","#d63031"),(.95,"p95","#e17055")]:
    ax.axvline(tw.quantile(q),color=col,ls="--",label=f"{lab}={tw.quantile(q):.0f}с")
ax.set_xlabel("total wall-time, с/видео"); ax.set_ylabel("видео"); ax.legend()
ax.set_title(f"Gate 3: распределение времени обработки видео (n={len(tw)})\nmean={tw.mean():.0f}с, разброс от коротких к длинным (corpus1000: 4-893с)")
ax.grid(axis="y",alpha=.3); fig.tight_layout()
f3=os.path.join(OUT,"gate3_video_wall_hist.png"); fig.savefig(f3,dpi=120); plt.close(fig)

# ---- FIG 4: OK / Fail(краш rc≠0) / Absent(нет выхода) по компонентам ----
okc =[len(pvc[(pvc["component"]==c)&(pvc["status"]=="ok")]) for c in comps]
failc=[len(pvc[(pvc["component"]==c)&(pvc["status"]=="fail")]) for c in comps]
absc=[len(pvc[(pvc["component"]==c)&(pvc["status"]=="absent")]) for c in comps]
fig,ax=plt.subplots(figsize=(11,5.5))
ax.bar(x,okc,0.6,label="OK",color="#00b894")
ax.bar(x,failc,0.6,bottom=okc,label="Fail (краш rc≠0)",color="#d63031")
ax.bar(x,absc,0.6,bottom=[o+f for o,f in zip(okc,failc)],label="Absent (нет выхода/каскад)",color="#fdcb6e")
for i,(o,fl,ab) in enumerate(zip(okc,failc,absc)):
    lbl=[]
    if fl: lbl.append(f"{fl}F")
    if ab: lbl.append(f"{ab}A")
    if lbl: ax.text(i,o+fl+ab,"+".join(lbl),ha="center",va="bottom",fontsize=8,color="#2d3436")
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in comps],rotation=20)
ax.set_ylabel("видео"); ax.set_ylim(0,1080)
ax.set_title("Gate 3: OK / Fail(краш) / Absent(нет выхода) по компонентам\nscene 497 краш = L13 (cuDNN на RTX 2000 Ada, фикс готов); остальные фейлы 2-13 — транзиенты, absent — каскад/таймаут")
ax.legend(); ax.grid(axis="y",alpha=.3); fig.tight_layout()
f4=os.path.join(OUT,"gate3_ok_fail.png"); fig.savefig(f4,dpi=120); plt.close(fig)

# копия в outbox (для VK)
import shutil
for f in [f1,f2,f3,f4]:
    shutil.copy(f, os.path.join(OUTBOX, os.path.basename(f)))

# ---- текстовая сводка ----
print("=== Gate 3 анализ ===")
print(f"видео: {len(pv)}, total wall p50={tw.median():.0f}с p95={tw.quantile(.95):.0f}с mean={tw.mean():.0f}с")
print("\ntime/component (p50с, доля от суммы p50):")
tot=sum(p50)
for c,v in sorted(zip(comps,p50),key=lambda z:-z[1]):
    print(f"  {SHORT[c]:8} {v:6.1f}с  {100*v/tot:4.1f}%")
print(f"\nCPU% p50 (медиана по компонентам с torch/cv): {np.nanmedian([v for v in cpu_p50 if v]):.0f}%")
print(f"GPU util mean p50 (макс среди компонентов): {np.nanmax([v for v in gpu_p50 if not np.isnan(v)]):.0f}%")
print("\nфейлы (краш rc≠0):", {SHORT[c]:f for c,f in zip(comps,failc) if f})
print("absent (нет выхода):", {SHORT[c]:a for c,a in zip(comps,absc) if a})
print(f"\nграфики: {OUT}/ + скопированы в deepdive_outbox")
