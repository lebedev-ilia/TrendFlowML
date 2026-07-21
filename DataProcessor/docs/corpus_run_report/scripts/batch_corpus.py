import json, subprocess, os, time
os.makedirs('/workspace/.dltmp',exist_ok=True)
open('/workspace/batch.pid','w').write(str(os.getpid()))
rows=json.load(open('/workspace/corpus300.json'))
done=0; fail=0
print('BATCH_START total=%d'%len(rows),flush=True)
for i,r in enumerate(rows,1):
    v=r['video_id']; repo=r['repo']
    mp='/workspace/corpus_out/%s/metrics.jsonl'%v
    if os.path.exists(mp) and '_summary' in open(mp).read():
        done+=1; print('[%d/%d] SKIP %s'%(i,len(rows),v),flush=True); continue
    t0=time.time()
    try:
        subprocess.run(['bash','/workspace/run_corpus.sh',v,repo,'6'],timeout=1500)
    except subprocess.TimeoutExpired:
        print('[%d/%d] TIMEOUT %s'%(i,len(rows),v),flush=True)
    dt=time.time()-t0
    ok = os.path.exists(mp) and '_summary' in open(mp).read()
    status = 'OK' if ok else 'FAIL'
    if ok: done+=1
    else: fail+=1
    print('[%d/%d] %s %s repo=%s %.0fs | done=%d fail=%d'%(i,len(rows),status,v,repo,dt,done,fail),flush=True)
print('BATCH_COMPLETE done=%d fail=%d'%(done,fail),flush=True)
