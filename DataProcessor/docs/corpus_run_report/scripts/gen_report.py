#!/usr/bin/env python3
"""Comprehensive batch report: per-video x per-component metrics (time, CPU/RAM/GPU load, peaks) + aggregates."""
import json, os, glob, csv, statistics as st, datetime
BASE='/workspace/corpus_out'
OUTDIR='/workspace/report'
os.makedirs(OUTDIR, exist_ok=True)
COMPONENTS=['download','segmenter','core_clip','core_depth_midas','core_optical_flow',
            'cut_detection','scene_classification','video_pacing','uniqueness']

def parse_gpu(path):
    """gpu_samples.csv -> list of (epoch, util, mem_mib)."""
    rows=[]
    if not os.path.exists(path): return rows
    for l in open(path):
        p=[x.strip() for x in l.split(',')]
        if len(p)<3: continue
        try:
            ep=datetime.datetime.strptime(p[0].split('.')[0], '%Y/%m/%d %H:%M:%S').timestamp()
            rows.append((ep, float(p[1]), float(p[2])))
        except Exception:
            continue
    return rows

def window(gpu, t0, t1):
    """max/mean util+mem within [t0,t1]."""
    sel=[(u,m) for (e,u,m) in gpu if t0 is not None and t1 is not None and t0-1<=e<=t1+1]
    if not sel: return (None,None,None,None)
    u=[x[0] for x in sel]; m=[x[1] for x in sel]
    return (max(u), round(st.mean(u),1), max(m), round(st.mean(m),1))

def pct(a,p):
    if not a: return None
    a=sorted(a); return round(a[int(round((p/100)*(len(a)-1)))],2)

vids=sorted(os.listdir(BASE))
per_comp={c:{'ok':0,'fail':0,'wall':[],'rss':[],'user':[],'sys':[],'cpu':[],'gpu_util':[],'gpu_mem':[]} for c in COMPONENTS}
video_rows=[]      # per-video summary
pvc_rows=[]        # per (video,component)
done=[]; partial=[]

for v in vids:
    mp=os.path.join(BASE,v,'metrics.jsonl')
    if not os.path.exists(mp): continue
    gpu=parse_gpu(os.path.join(BASE,v,'gpu_samples.csv'))
    stages={}; summ=None
    for l in open(mp):
        l=l.strip()
        if not l: continue
        try: o=json.loads(l)
        except: continue
        if o.get('stage')=='_summary': summ=o
        else: stages[o['stage']]=o
    complete = summ is not None
    (done if complete else partial).append(v)
    npz=len(glob.glob(os.path.join(BASE,v,'rs','*','*.npz')))
    tot_wall=0.0
    for c in COMPONENTS:
        s=stages.get(c)
        if not s:
            pvc_rows.append({'video':v,'component':c,'status':'absent'}); continue
        rc=s.get('rc',1)
        if rc==0: per_comp[c]['ok']+=1
        else: per_comp[c]['fail']+=1
        wall=s.get('wall_s'); rss=s.get('max_rss_kb'); usr=s.get('user_s'); sy=s.get('sys_s'); cpu=s.get('cpu_pct')
        gu,gum,gm,gmm=window(gpu, s.get('t_start'), s.get('t_end'))
        if wall: per_comp[c]['wall'].append(wall); tot_wall+=wall
        if rss: per_comp[c]['rss'].append(rss)
        if usr: per_comp[c]['user'].append(usr)
        if sy: per_comp[c]['sys'].append(sy)
        if cpu: per_comp[c]['cpu'].append(cpu)
        if gu is not None: per_comp[c]['gpu_util'].append(gu)
        if gm is not None: per_comp[c]['gpu_mem'].append(gm)
        pvc_rows.append({'video':v,'component':c,'status':'ok' if rc==0 else 'fail','rc':rc,
            'wall_s':round(wall,2) if wall else None,'user_s':usr,'sys_s':sy,'cpu_pct':cpu,
            'max_rss_mb':round(rss/1024,1) if rss else None,
            'minor_faults':s.get('minor_faults'),'vol_ctx_sw':s.get('vol_ctx_sw'),
            'gpu_util_max':gu,'gpu_util_mean':gum,'gpu_mem_peak_mib':gm,'gpu_mem_mean_mib':gmm})
    video_rows.append({'video':v,'status':'complete' if complete else 'partial','npz_files':npz,
        'total_wall_s':round(tot_wall,1),
        'gpu_util_max':(summ or {}).get('gpu_util_max'),'gpu_util_mean':(summ or {}).get('gpu_util_mean'),
        'gpu_util_p95':(summ or {}).get('gpu_util_p95'),
        'gpu_mem_peak_mib':(summ or {}).get('gpu_mem_peak_mib'),'gpu_mem_mean_mib':(summ or {}).get('gpu_mem_mean_mib'),
        'gpu_samples':(summ or {}).get('n_samples')})

# ---- write per-(video,component) CSV ----
pvc_fields=['video','component','status','rc','wall_s','user_s','sys_s','cpu_pct','max_rss_mb',
            'minor_faults','vol_ctx_sw','gpu_util_max','gpu_util_mean','gpu_mem_peak_mib','gpu_mem_mean_mib']
with open(os.path.join(OUTDIR,'per_video_component.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=pvc_fields); w.writeheader()
    for r in pvc_rows: w.writerow({k:r.get(k) for k in pvc_fields})
# ---- write per-video CSV ----
vf=['video','status','npz_files','total_wall_s','gpu_util_max','gpu_util_mean','gpu_util_p95','gpu_mem_peak_mib','gpu_mem_mean_mib','gpu_samples']
with open(os.path.join(OUTDIR,'per_video.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=vf); w.writeheader()
    for r in video_rows: w.writerow({k:r.get(k) for k in vf})

# ---- per-component aggregates ----
comp_stats={}
for c in COMPONENTS:
    d=per_comp[c]
    def agg(a): return {'p50':pct(a,50),'p95':pct(a,95),'mean':round(st.mean(a),2) if a else None,'max':round(max(a),2) if a else None} if a else None
    comp_stats[c]={'ok':d['ok'],'fail':d['fail'],
        'wall_s':agg(d['wall']),'cpu_pct':agg(d['cpu']),'user_s':agg(d['user']),'sys_s':agg(d['sys']),
        'max_rss_mb':agg([x/1024 for x in d['rss']]),
        'gpu_util_pct':agg(d['gpu_util']),'gpu_mem_mib':agg(d['gpu_mem'])}

# ---- overall ----
all_summ=[r for r in video_rows if r['status']=='complete']
gpu_peaks=[r['gpu_mem_peak_mib'] for r in all_summ if r['gpu_mem_peak_mib']]
gpu_utils=[r['gpu_util_max'] for r in all_summ if r['gpu_util_max'] is not None]
walls=[r['total_wall_s'] for r in all_summ]
npzs=[r['npz_files'] for r in all_summ]
overall={'generated':datetime.datetime.now().isoformat(),
    'videos_complete':len(done),'videos_partial':len(partial),
    'total_per_video_wall_s':{'p50':pct(walls,50),'p95':pct(walls,95),'mean':round(st.mean(walls),1) if walls else None},
    'gpu_mem_peak_mib_max':max(gpu_peaks) if gpu_peaks else None,
    'gpu_util_max_observed':max(gpu_utils) if gpu_utils else None,
    'npz_per_video':{'p50':pct(npzs,50),'min':min(npzs) if npzs else None,'max':max(npzs) if npzs else None}}
report={'overall':overall,'per_component':comp_stats}
json.dump(report,open(os.path.join(OUTDIR,'batch_report.json'),'w'),indent=2,ensure_ascii=False)

# ---- Markdown ----
md=['# Batch Report — DataProcessor corpus run','',
    f"Generated: {overall['generated']}",'',
    f"**Videos complete:** {len(done)} | **partial/failed:** {len(partial)}",
    f"**GPU mem peak (max over videos):** {overall['gpu_mem_peak_mib_max']} MiB | **GPU util max:** {overall['gpu_util_max_observed']}%",
    f"**Per-video wall time:** p50={overall['total_per_video_wall_s']['p50']}s p95={overall['total_per_video_wall_s']['p95']}s",
    f"**NPZ per video:** p50={overall['npz_per_video']['p50']} (min {overall['npz_per_video']['min']}, max {overall['npz_per_video']['max']})",'',
    '## Per-component (over complete+partial videos)','',
    '| Component | OK | Fail | wall p50/p95 s | CPU% p50/p95 | RSS MB p95 | GPU util p95 | GPU mem MiB p95 |',
    '|---|---|---|---|---|---|---|---|']
for c in COMPONENTS:
    s=comp_stats[c]
    def g(x,k): return (x or {}).get(k) if x else None
    md.append(f"| {c} | {s['ok']} | {s['fail']} | {g(s['wall_s'],'p50')}/{g(s['wall_s'],'p95')} | "
              f"{g(s['cpu_pct'],'p50')}/{g(s['cpu_pct'],'p95')} | {g(s['max_rss_mb'],'p95')} | "
              f"{g(s['gpu_util_pct'],'p95')} | {g(s['gpu_mem_mib'],'p95')} |")
md+=['','## Files','',
     '- `per_video_component.csv` — full per-(video,component) metrics (time, CPU user/sys/%, RSS, faults, ctx-sw, per-component GPU util/mem)',
     '- `per_video.csv` — per-video summary (total wall, GPU peak/util/mem, NPZ count, status)',
     '- `batch_report.json` — machine-readable aggregates for TF Agent M',
     '- raw: each `corpus_out/<id>/metrics.jsonl`, `gpu_samples.csv`, `.time_*` (full /usr/bin/time -v)']
open(os.path.join(OUTDIR,'BATCH_REPORT.md'),'w').write('\n'.join(md))
print('REPORT WRITTEN to /workspace/report/')
print('videos_complete=%d partial=%d'%(len(done),len(partial)))
print('per_video_component.csv rows=%d'%len(pvc_rows))
