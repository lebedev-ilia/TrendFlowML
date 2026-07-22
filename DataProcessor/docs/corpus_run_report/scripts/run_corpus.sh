#!/bin/bash
# usage: run_corpus.sh <video_id> <repo> <fps>
source /workspace/venv/bin/activate
DP=/workspace/TrendFlowML/DataProcessor
export DP_MODELS_ROOT=$DP/dp_models
export PYTHONPATH=$DP/VisualProcessor:$DP:$PYTHONPATH
export TRITON_HTTP_URL=http://localhost:8000
export $(grep -E "^HF_TOKEN=" /workspace/TrendFlowML/automation/fetcher/.env 2>/dev/null | head -1)
VID="$1"; REPO="$2"; FPS="${3:-6}"
OUT=/workspace/corpus_out/$VID; mkdir -p "$OUT"
RS="$OUT/rs"; mkdir -p "$RS"
CFG=/workspace/TrendFlowML/configs/visual_triton_baseline_gpu_local.yaml
MET="$OUT/metrics.jsonl"; : > "$MET"
LOG="$OUT/run.log"; : > "$LOG"
DL=$(mktemp -d /workspace/.dltmp/dlXXXXXX 2>/dev/null || mktemp -d)
# --- GPU sampler (фон) ---
GPUCSV="$OUT/gpu_samples.csv"; : > "$GPUCSV"
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$GPUCSV" 2>/dev/null; sleep 1; done ) &
SAMP=$!
stage(){ # stage <name> <cmd...>
  local name="$1"; shift
  local t0=$(date +%s.%N)
  /usr/bin/time -v -o "$OUT/.time_$name" "$@" >> "$LOG" 2>&1; local rc=$?
  local t1=$(date +%s.%N)
  local wall=$(echo "$t1 - $t0" | bc)
  local tf="$OUT/.time_$name"
  local rss=$(grep "Maximum resident" "$tf" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  local usr=$(grep "User time" "$tf" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
  local sy=$(grep "System time" "$tf" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
  local cpu=$(grep "Percent of CPU" "$tf" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  local flt=$(grep "Minor (reclaiming" "$tf" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  local ctx=$(grep "Voluntary context" "$tf" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  echo "{\"stage\":\"$name\",\"rc\":$rc,\"wall_s\":$wall,\"max_rss_kb\":${rss:-0},\"user_s\":${usr:-0},\"sys_s\":${sy:-0},\"cpu_pct\":${cpu:-0},\"minor_faults\":${flt:-0},\"vol_ctx_sw\":${ctx:-0},\"t_start\":$t0,\"t_end\":$t1}" >> "$MET"
  echo "STAGE $name rc=$rc wall=${wall}s" >> "$LOG"
  [ $rc -ne 0 ] && echo "  FAIL $name" 
  return $rc
}
# --- download ---
T0=$(date +%s.%N)
python -c "from huggingface_hub import hf_hub_download; import shutil; p=hf_hub_download(\"Ilialebedev/$REPO\",filename=\"$VID.mp4\",repo_type=\"dataset\",local_dir=\"$DL\"); print(p)" >> "$LOG" 2>&1 || { echo "DL_FAIL"; kill $SAMP 2>/dev/null; exit 2; }
MP4="$DL/$VID.mp4"
echo "{\"stage\":\"download\",\"rc\":0,\"wall_s\":$(echo "$(date +%s.%N) - $T0"|bc)}" >> "$MET"
# --- segmenter ---
stage segmenter python $DP/Segmenter/segmenter.py --video-path="$MP4" --output "$OUT/seg" --visual-cfg-path "$CFG" --platform-id youtube --video-id="$VID" --run-id "corpus_$VID" --sampling-policy-version corpus_v1 --config-hash local --dataprocessor-version corpus_local --analysis-fps "$FPS" --analysis-width 480
FD="$OUT/seg/$VID/video"
if [ ! -f "$FD/metadata.json" ]; then echo "SEG_FAIL"; rm -rf "$DL"; kill $SAMP 2>/dev/null; exit 3; fi
# --- visual components ---
stage core_clip python $DP/VisualProcessor/core/model_process/core_clip/main.py --frames-dir "$FD" --rs-path "$RS" --runtime triton --triton-image-model-spec clip_image_224_triton --triton-text-model-spec clip_text_triton --triton-preprocess-preset openai_clip_224 --batch-size 16
stage core_depth_midas python $DP/VisualProcessor/core/model_process/core_depth_midas/main.py --frames-dir "$FD" --rs-path "$RS" --runtime triton --triton-model-spec midas_256_triton --triton-http-url $TRITON_HTTP_URL --batch-size 16 --triton-preprocess-preset midas_256
stage core_optical_flow python $DP/VisualProcessor/core/model_process/core_optical_flow/main.py --frames-dir "$FD" --rs-path "$RS" --runtime triton --triton-model-spec raft_256_triton --triton-http-url $TRITON_HTTP_URL --batch-size 16 --triton-preprocess-preset raft_256
stage cut_detection python $DP/VisualProcessor/modules/cut_detection/main.py --frames-dir "$FD" --rs-path "$RS" --no-use-clip
stage scene_classification python $DP/VisualProcessor/modules/scene_classification/main.py --frames-dir "$FD" --rs-path "$RS" --runtime inprocess --model-arch resnet50 --device cuda --label-fusion places --enable-advanced-features
stage video_pacing python $DP/VisualProcessor/modules/video_pacing/main.py --frames-dir "$FD" --rs-path "$RS" --enable-entropy-features --enable-histograms
stage uniqueness python $DP/VisualProcessor/modules/uniqueness/main.py --frames-dir "$FD" --rs-path "$RS"
# --- finalize metrics ---
kill $SAMP 2>/dev/null
python3 - << PY >> "$MET"
import json
gpu_util=[]; gpu_mem=[]
try:
  for l in open("$GPUCSV"):
    p=[x.strip() for x in l.split(",")]
    if len(p)>=3:
      gpu_util.append(float(p[1])); gpu_mem.append(float(p[2]))
except: pass
import statistics as st
def p95(a):
  if not a: return None
  a=sorted(a); return a[int(round(0.95*(len(a)-1)))]
o={"stage":"_summary","gpu_util_max":max(gpu_util) if gpu_util else None,"gpu_util_mean":round(st.mean(gpu_util),1) if gpu_util else None,"gpu_util_p95":p95(gpu_util),"gpu_mem_peak_mib":max(gpu_mem) if gpu_mem else None,"gpu_mem_mean_mib":round(st.mean(gpu_mem),1) if gpu_mem else None,"n_samples":len(gpu_util)}
print(json.dumps(o))
PY
rm -rf "$DL" "$OUT/seg" 2>/dev/null
echo "CORPUS_DONE $VID"
