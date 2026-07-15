# Карта проекта TrendFlow (актуальная, по итогам анализа 2026-06-30)

Составлена по сквозному проходу (`ANALYSIS_PLAN.md`, прогресс — `ANALYSIS_CHECKLIST.md`).
Зрелость: 🟢 готово к проду · 🟡 работает, нужны доработки · 🔴 пробел.

## Поток данных (E2E)

```
Сайт (Next.js+FastAPI) ──POST /api/runs──▶ Fetcher ──finalize/trigger──▶ Backend
                                              │ (видео+метаданные в S3/MinIO)
                                              ▼
                                        DataProcessor (main.py / api+worker)
                                              │ Segmenter → frames/audio (union-сэмплинг)
                                  ┌───────────┼───────────┐
                              Visual        Audio        Text   ── читают dp_models / Triton / Embedding Service
                                  └───────────┼───────────┘
                                      manifest.json + NPZ (source-of-truth) в S3
                                              ▼
                                        Models (baseline / v1 transformer) → прогноз
```

## Компоненты и зрелость

| Компонент | Что | Зрелость | Заметки |
|---|---|---|---|
| **Fetcher** | сбор 200k (multi-platform, dataset_collector) | 🟢 | backpressure, circuit_breaker, idempotency, resume, balancer, HF-координация, кампании до 100k/мес |
| **Segmenter** | единый источник `frame_indices` | 🟢 | union-сэмплинг, адаптивная плотность, `union_timestamps_sec`=SoT; проверить версионирование sampling_policy |
| **DataProcessor api/worker** | оркестрация обработки | 🟢 | Redis Streams + consumer groups, idempotency, recovery, state_machine; состояние в Redis |
| **DataProcessor main.py** | subprocess-оркестратор процессоров | 🟡 | 1500 строк, spawn Audio/Text/Visual на видео; opt-кандидат (тёплые воркеры) |
| **VisualProcessor** | 11 core + 19 modules | 🟢/🟡 | core_object_detections=точка YOLO (ждёт чекпоинт); core_identity=6 хедов (places/franchise без баз) |
| **AudioProcessor** | 22 экстрактора (Audit v3) | 🟢 | no-network через dp_models/triton; DSP librosa CPU-тяжёлый (opt) |
| **TextProcessor** | 22 экстрактора (Audit v3) | 🟢 | централ. model_registry, privacy/token-only; e5-large не в едином репо |
| **dp_models (ModelManager)** | 8 провайдеров, spec_catalog | 🟢 | no-network/fail-fast/digests; DP_MODELS_ROOT=DataProcessor/dp_models |
| **Embedding Service** | faiss + category-менеджеры | 🟡 | реальный FastAPI-сервис, но **нет в k8s** (нужно для мульти-ноды) |
| **Triton** | ONNX (CLIP/midas/raft/places365) | 🟢 | модели провижатся из `trendflow_artifact_0_1` (multi-source манифест) |
| **Backend** | сайт-оркестратор (FastAPI+Celery) | 🟢 | +метрики RED/Celery, структурные логи+correlation_id, OTel, надёжность Celery (добавлено) |
| **Models** | baseline + v1 (encoder/transformer/text) | 🟡 | контракты заданы; нужен прогон на датасете с таргетами |
| **DynamicBatch** | планировщик масштаба | 🟢 | cost-модель, level-1 batching, OOM-backoff, Postgres-registry |

## Модели и веса

- Единый HF dataset **`Ilialebedev/trendflow_models`** (3.4 ГБ): веса под `DataProcessor/dp_models/...` + семантические базы. Манифест — `configs/models_manifest.json` (sha256, версии, multi-source). Загрузка — `DataProcessor/scripts/download_models.py`.
- Triton ONNX (5.16 ГБ) — из `trendflow_artifact_0_1` (приватный → нужен `HF_TOKEN`).
- Публичные базовые модели (e5-large, wavlm, wav2vec2-base, CLAP-630k, places365, slowfast, pyannote) — **не в репо** (`public_base_models`), тянутся отдельно → пробел для offline-мультиноды.

## Развёртывание

- **k8s/** (kustomize): namespace, postgres/redis/minio/triton, backend(+migrate/ingress), dataprocessor(api/worker/hpa), models-pvc+download Job, governance (PriorityClass/Quota/PDB), KEDA (опц.). **Нет: Embedding Service.**
- **docker-compose.prod.yml** — весь стек на одной машине (17 сервисов).
- **bootstrap.sh** — от `git clone` до запуска (venvs/.data_venv/.fetcher_venv, модели, стек, smoke).
- Наблюдаемость: метрики `dataprocessor_*` + backend RED + Fetcher; Prometheus/Grafana/Jaeger; структурные логи + correlation_id; OTel во всех 3 сервисах.

## Зона владельца (ассистент не делает)

1. Ручная разметка semantic-баз (фото качаются `_tools/download_reference_images.py` → разметка → `build_gallery.py`).
2. Адаптация алгоритмов под особенности YOLO-датасета (опишет позже).
3. Предоставить корпус 200k + таргеты для доказательства пользы фич; ротировать YouTube/HF-ключи.

> Полный список доработок и порядок — `IMPLEMENTATION_PLAN.md`.
