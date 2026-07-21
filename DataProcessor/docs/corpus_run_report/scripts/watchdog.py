import time, os, subprocess, urllib.request, datetime
LOG='/workspace/watchdog.log'
def log(m):
    open(LOG,'a').write('%s %s\n'%(datetime.datetime.now().strftime('%H:%M:%S'),m))
def batch_alive():
    try:
        pid=int(open('/workspace/batch.pid').read().strip()); os.kill(pid,0); return True
    except: return False
def batch_complete():
    try: return 'BATCH_COMPLETE' in open('/workspace/batch_progress.log').read()
    except: return False
def triton_ok():
    try:
        urllib.request.urlopen('http://localhost:8000/v2/health/ready',timeout=5); return True
    except: return False
def start_batch():
    subprocess.Popen("nohup setsid bash -c 'python3 /workspace/batch_corpus.py >> /workspace/batch_progress.log 2>&1' </dev/null >/dev/null 2>&1 &",shell=True)
def start_triton():
    subprocess.Popen("nohup setsid bash /workspace/start_triton.sh >> /workspace/triton_srv.log 2>&1 </dev/null >/dev/null 2>&1 &",shell=True)
log('watchdog start')
while True:
    if batch_complete(): log('BATCH COMPLETE - watchdog exit'); break
    if not triton_ok():
        log('triton DOWN -> restart'); start_triton(); time.sleep(45)
    if not batch_alive() and not batch_complete():
        log('batch DEAD -> restart'); start_batch(); time.sleep(25)
    time.sleep(60)
