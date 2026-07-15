#!/usr/bin/env bash
# =============================================================================
# Запуск GPU-валидации action_recognition в k8s одной командой (без Cursor).
#
# Делает: PVC → helper-pod → заливает входные .mp4 → запускает GPU-Job → ждёт →
# забирает артефакты (/io/output) в репозиторий для анализа Claude → чистит helper.
#
# Использование:
#   NS=trendflow \
#   FIXTURES_DIR=DataProcessor/docs/component_reports/action_recognition/fixtures \
#   ./k8s/jobs/run_ar_validation.sh
#
# Переменные: NS (namespace), FIXTURES_DIR (локальные .mp4), SECONDS_LIMIT (0=полный),
#             FPS, DEVICE, OUT_DIR (куда забрать артефакты).
# =============================================================================
set -Eeuo pipefail
NS="${NS:-trendflow}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS="$ROOT/k8s/jobs"
FIXTURES_DIR="${FIXTURES_DIR:-$ROOT/DataProcessor/docs/component_reports/action_recognition/fixtures}"
OUT_DIR="${OUT_DIR:-$ROOT/DataProcessor/docs/component_reports/action_recognition/artifacts/gpu}"
SECONDS_LIMIT="${SECONDS_LIMIT:-0}"; FPS="${FPS:-25}"; DEVICE="${DEVICE:-cuda}"
HELPER="ar-val-helper"

log(){ printf '\033[1;36m[ar-val]\033[0m %s\n' "$*"; }
k(){ kubectl -n "$NS" "$@"; }

log "1/6 PVC ar-validation-io"
k apply -f "$JOBS/ar-validation-io-pvc.yaml"

log "2/6 helper-pod для заливки входных клипов"
cat <<YAML | k apply -f -
apiVersion: v1
kind: Pod
metadata: { name: $HELPER, labels: { app: action-recognition, component: io-helper } }
spec:
  restartPolicy: Never
  containers:
  - name: helper
    image: busybox:1.36
    command: ["sh","-c","mkdir -p /io/input /io/output && sleep 36000"]
    volumeMounts: [{ name: io, mountPath: /io }]
  volumes: [{ name: io, persistentVolumeClaim: { claimName: ar-validation-io } }]
YAML
k wait --for=condition=Ready "pod/$HELPER" --timeout=180s

log "3/6 заливаю фикстуры из $FIXTURES_DIR"
shopt -s nullglob
for f in "$FIXTURES_DIR"/*.mp4; do
  log "  cp $(basename "$f")"
  k cp "$f" "$HELPER:/io/input/$(basename "$f")"
done

log "4/6 запускаю GPU-Job (SECONDS_LIMIT=$SECONDS_LIMIT FPS=$FPS DEVICE=$DEVICE)"
k delete job ar-validation --ignore-not-found
# патчим env под параметры запуска
python3 - "$JOBS/ar-validation-job.yaml" "$SECONDS_LIMIT" "$FPS" "$DEVICE" <<'PY' | kubectl -n "${NS:-trendflow}" apply -f -
import sys,re
y=open(sys.argv[1]).read()
y=y.replace('value: "0" }','value: "%s" }'%sys.argv[2],1)  # SECONDS_LIMIT (первый value:"0")
y=re.sub(r'(name: FPS, value: )"[^"]*"', r'\1"%s"'%sys.argv[3], y)
y=re.sub(r'(name: DEVICE, value: )"[^"]*"', r'\1"%s"'%sys.argv[4], y)
sys.stdout.write(y)
PY

log "5/6 жду завершения Job (до 2ч)"
k wait --for=condition=complete job/ar-validation --timeout=7200s \
  || { log "Job не complete — логи ниже"; k logs job/ar-validation --tail=80 || true; }

log "6/6 забираю артефакты → $OUT_DIR"
mkdir -p "$OUT_DIR"
k cp "$HELPER:/io/output" "$OUT_DIR" || log "cp output failed"
log "готово. Артефакты: $OUT_DIR"
log "helper-pod оставлен для докачки; удалить: kubectl -n $NS delete pod $HELPER"
