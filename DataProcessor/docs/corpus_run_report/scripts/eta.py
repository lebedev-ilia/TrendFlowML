import json,os,re,statistics as st
rows=json.load(open('/workspace/corpus300.json'))
dur={r['video_id']:(r.get('duration') or 0) for r in rows}
# done set
done=set()
for d in os.listdir('/workspace/corpus_out'):
    m='/workspace/corpus_out/%s/metrics.jsonl'%d
    if os.path.exists(m) and '_summary' in open(m).read(): done.add(d)
# per-video wall from batch log (OK lines)
wall={}
for l in open('/workspace/batch_progress.log'):
    mt=re.search(r'\] OK (\S+) repo=\S+ (\d+)s',l)
    if mt: wall[mt.group(1)]=int(mt.group(2))
# model points: done videos with known wall+duration
pts=[(dur[v],wall[v]) for v in done if v in wall and dur.get(v,0)>0]
# piecewise: frames ~ min(duration*rate, cap). Processing ~ a + b*effective_frames.
# effective 'size' proxy = min(duration, DCAP) since sampling caps long videos
DCAP=100.0  # sampling budget saturates around ~100s (rate 4fps, cap ~400 frames)
def size(dv): return min(dv,DCAP)
xs=[size(d) for d,w in pts]; ys=[w for d,w in pts]
n=len(pts); mx=sum(xs)/n; my=sum(ys)/n
b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
a=my-b*mx
def pred(dv): return max(a+b*size(dv), 60)
# remaining
remaining=[v for v in dur if v not in done]
pred_secs=[pred(dur[v]) for v in remaining]
total=sum(pred_secs)
# also empirical mean for sanity
emp=st.mean(ys)
print('model: wall ~ %.0f + %.2f*min(dur,%ds)  (n=%d fit points)'%(a,b,DCAP,n))
print('done=%d remaining=%d'%(len(done),len(remaining)))
print('remaining dur buckets: <30s=%d 30-100s=%d 100-300s=%d >300s=%d'%(
  sum(1 for v in remaining if dur[v]<30),sum(1 for v in remaining if 30<=dur[v]<100),
  sum(1 for v in remaining if 100<=dur[v]<300),sum(1 for v in remaining if dur[v]>=300)))
print('predicted remaining wall: %.1f h (model)  |  %.1f h (naive mean %.0fs x%d)'%(total/3600, emp*len(remaining)/3600, emp, len(remaining)))
print('mean predicted per remaining video: %.0fs'%(total/len(remaining)))
