# GPU-Job: валидация action_recognition в k8s (без Cursor)

Прогоняет цепочку **Segmenter → core_object_detections (+appearance-tracker) → action_recognition v3**
на **GPU-нодах** твоего кластера и складывает артефакты, которые Claude потом сам анализирует.
Заменяет ручной прогон Cursor'ом: ты запускаешь одну команду, дальше — на Claude.

## Что внутри
| Файл | Назначение |
|---|---|
| `ar-validation-io-pvc.yaml` | RWX PVC `ar-validation-io` — вход (`/io/input`) + выход (`/io/output`) |
| `ar-validation-job.yaml` | GPU-Job (образ `trendflow-dataprocessor`, `nvidia.com/gpu: 1`, models-pvc RO) |
| `run_ar_validation.sh` | оркестрация: залить фикстуры → запустить Job → дождаться → забрать артефакты в репо |

Движок прогона — `DataProcessor/scripts/run_ar_local.py` (тот же, что для CPU-smoke; на GPU
запускается с `--device cuda`, полным клипом `--seconds 0`, `--fps 25`).

## Предпосылки
1. Кластер поднят по `k8s/` (namespace `trendflow`, `models-pvc` заполнен model-download Job'ом,
   `PriorityClass trendflow-gpu-high` из `governance.yaml`).
2. GPU-ноды помечены `accelerator=nvidia-gpu` (как `dataprocessor-worker`).
3. `ar-validation-io` PVC использует **RWX** storage class (nfs/cephfs) — впиши его в PVC-манифест.
4. `kubectl` настроен на кластер; образ `trendflow-dataprocessor:latest` доступен в registry.

## Запуск (одна команда)
```bash
NS=trendflow ./k8s/jobs/run_ar_validation.sh
# параметры (опц.):
#   FIXTURES_DIR=<папка с .mp4>   SECONDS_LIMIT=0(полный)|N   FPS=25   DEVICE=cuda
#   OUT_DIR=<куда забрать артефакты>  (по умолчанию .../action_recognition/artifacts/gpu)
```
Скрипт зальёт `.mp4` из `FIXTURES_DIR`, запустит Job на GPU, дождётся завершения и **скопирует
`/io/output` в репозиторий** (`artifacts/gpu/`). После этого Claude сам разбирает npz и пишет REPORT.

## Что на выходе (per video, в `artifacts/gpu/<video_id>/`)
- `rs/action_recognition/action_recognition_features.npz` (v3: penultimate-эмбеддинг, классы,
  `clip_track_id`, `clip_segment_id`, агрегаты) + `metrics.{json,prom}`;
- `rs/core_object_detections/detections.npz` (+ `track_ids`);
- `summary.json` (clip_count, mean_clips_per_track, embedding_mode/dim, статус);
- `run.log` (все стадии + вывод обоих валидаторов).

## Тонкая настройка Job
Параметры передаются env'ами Job (`FIXTURES`, `SECONDS_LIMIT`, `FPS`, `DEVICE`) — их патчит
`run_ar_validation.sh`, либо правь `ar-validation-job.yaml` напрямую. Action_recognition-флаги
(`--embedding-mode penultimate`, `--localization track_anchored`, `--tubelet-crop true`,
`window_len_mult` через Segmenter cfg) зашиты в `run_ar_local.py` — меняются там.

## Ограничение
Диск: dense-окна кэшируют кадры; Job удаляет `seg/` и `clip.mp4` после NPZ. Для больших батчей —
следи за размером `ar-validation-io` PVC (по умолчанию 50Gi).
