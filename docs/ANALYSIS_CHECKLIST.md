# Чеклист анализа проекта

Статусы: ⬜ не начато · 🔄 в работе · ✅ проанализировано.
План — [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md).

## Фазы / папки

| Фаза | Область | Статус | Заметки |
|---|---|---|---|
| 0 | Корень, архитектура, configs, example | ✅ | корень почищен; индекс+CLAUDE.md актуализированы |
| 1 | Fetcher (+ dataset_collector, photo-download для разметки) | ✅ | зрелый движок 200k; убраны секреты из git; найден gap по downloader'ам |
| 2 | DataProcessor: оркестрация (main/api/dag/queue/triton/storage/docker) | ✅ | зрелый api-сервис (Redis Streams+idempotency+recovery); main.py = subprocess-оркестратор; дубль очередей |
| 3 | Segmenter | ✅ | зрелый, единый источник frame_indices; union-сэмплинг; README чистить |
| 4 | VisualProcessor (core 11 + modules 19) | ✅ | core_identity разобран в аудите 6.1; сделан единый downloader фото для разметки |
| 5 | AudioProcessor | ✅ | зрелый (Audit v3, 22 экстрактора, no-network); DSP librosa-тяжёлый — opt-кандидат |
| 6 | TextProcessor | ✅ | зрелый (22 экстр., централ. model_registry, no-network, privacy); e5-large не в едином репо |
| 7 | dp_models / embedding_service / triton | ✅ | dp_models зрелый (8 провайдеров); ES = микросервис, но НЕ в k8s |
| 8 | Качество фич и тулинг (tools/scripts/qa/monitoring) | ✅ | полный тулинг (quality_audit/qa_pipeline/drift/golden/build_training_matrix); нужен прогон на корпусе |
| 9 | Backend | ✅ | разобран ранее; +метрики/логи/трейсинг/надёжность добавлены мной |
| 10 | Models | ✅ | baseline + v1 (encoder/transformer/text) + контракты; Логика заскаффолжена, нужен прогон с таргетами |
| 11 | DynamicBatch | ✅ | планировщик cost-модель/level-1/OOM-backoff/Postgres-registry (разобран ранее) |
| 12 | Развёртывание/инфра (k8s/docker/monitoring) | ✅ | построено мной (k8s kustomize, compose.prod, bootstrap, метрики); gap: ES в k8s |
| 13 | Синтез → PROJECT_MAP + вход в IMPLEMENTATION_PLAN | ✅ | созданы `docs/PROJECT_MAP.md` и `docs/IMPLEMENTATION_PLAN.md` |

## Лог изменений документов

Формат: `действие` — путь — что сделано.

**Фаза 1 (2026-06-30) — Fetcher:**
- `untrack+gitignore` — `Fetcher/youtube_working_keys.txt`, `youtube_key_check_results.json`, `celerybeat-schedule`, `coverage.xml` — секреты/артефакты убраны из git (файлы на диске сохранены).
- `перенесено` — `d.py` → `Fetcher/scripts/smoke_metadata.py`.
- ⚠️ **ВЛАДЕЛЬЦУ:** YouTube API-ключи были в git-истории → **ротировать**.
- Оценка масштаба: dataset_collector зрелый (мультиплатформа youtube/tiktok/twitch/instagram/rutube, HF-координация, backpressure, circuit_breaker, idempotency, rate_limiter, resume, snapshots, balancer, worker_leases; кампании вкл. `dataset_campaign_100k_monthly.json`). База под 200k сильная.

**Фаза 0 (2026-06-30):**
- `удалено` — 21× `.DS_Store` по репо + `configs/FINAL_BENCH_TABLE copy.md` — очевидный мусор.
- `перенесено` — `doc.md` → `docs/PRODUCT_VISION.md` — продуктовое видение/Q&A в правильное место.
- `удалено` — `poradoc.md` — 3 e2e-команды, перекрыты `bootstrap.sh` + E2E-доками (содержимое залогировано в истории).
- `обновлено` — `docs/MAIN_INDEX.md` — добавлена секция «Прод-готовность, модели и анализ (2026-06)» со ссылками на новые доки.
- `обновлено` — `CLAUDE.md` — блок «Актуальное (2026-06-30)»: единый HF-репо, bootstrap/k8s, ссылки на план анализа, зона владельца.

## Находки по веткам (сырьё для плана реализации)

### Логика (польза фич)
- [logic/models] **Architecture review моделей** — `Models/docs/ARCHITECTURE_REVIEW.md`: углублённые рекомендации (Tweedie/quantile loss, modality dropout+learned tokens, temporal split, Spearman+интервалы, baseline-first+ablation, early-engagement velocity, sampling_policy↔dataset_version) + пронумерованный список изменений по всем файлам `Models/`. Протокол компонентов дополнен §0.1 «модельная пригодность выхода» (типы token-stream, seq в NPZ не debug-.npy, time-axis, вклад в baseline). Найден конкретный gap: `pitch` держит f0-контур вне NPZ.
- [logic] **C4 (загрузчики) ✅ частично:** провайдер `dp_models/providers/transformers_pretrained.py` (offline `from_pretrained`, engine=`transformers`) зарегистрирован в ModelManager; specs `vision/videomae_kinetics400_inprocess.yaml` и `audio/whisper_small_hf_transformers.yaml` загружаемы. Компилируется. **Follow-up:** переключить модули `action_recognition`(slowfast→videomae) и `asr_extractor`(openai→HF) + адаптировать I/O (feature-quality). (Спринт 3)
- [logic] Segmenter — единый источник `frame_indices` (пер-компонентные бюджеты, shared sampling group, union-сэмплинг, адаптивная плотность, `union_timestamps_sec` = SoT). **A6 ✅ проверено:** `sampling_policy_version` уже входит в `config_hash` (idempotency key, `AudioProcessor/src/core/config_hash.py:62`) и пишется в `run_meta` Segmenter — версионирование есть, правки не нужны. Рекомендация: вести changelog версий политики при изменении сэмплинга. Почистить README Segmenter (псевдо-идеи vs реальный контракт). (Спринт 1)
- [logic/owner] Скрипты скачивания фото для разметки semantic-баз есть **только** для брендов
  (`brand_semantics/utils/download_logos_wikimedia.py`). Для cars/celebs/places/franchise — нет.
  Сделать единый downloader (вход для ручной разметки владельца) — в Фазе 4 (VisualProcessor).
  **✅ СДЕЛАНО (Фаза 4):** `dp_models/bundled_models/semantics/_tools/download_reference_images.py` —
  Wikimedia Commons по 5 доменам (brands/cars/celebs/places/franchise), проверен (dry-run + реальная загрузка).
  Поток для владельца: download → ручная разметка → convert_known_to_semantics → build_gallery → build_db_manifest → preflight.
- [logic] `Models/`: контракты (ENCODER_CONTRACT, MODEL_CONTRACTS_V1, TARGETS_SPLITS_METRICS) задают интерфейс фич; есть baseline (train/predict) + v1 (encoder/transformer/text) + `build_v1_dataset_index`/`build_training_matrix`. Доказательство пользы фич (Q5 feature-importance) = прогон на стабильном датасете с таргетами (зона владельца: данные/таргеты). (Фаза 10)
- [logic] VisualProcessor: core (core_clip, core_object_detections=точка YOLO, core_face_landmarks, core_depth_midas,
  core_optical_flow, ocr_extractor, core_identity=6 хедов [см. аудит 6.1]) + 19 modules (scene_classification,
  action_recognition, shot_quality, emotion_face, cut_detection, high_level_semantic и др.) — консьюмеры core-провайдеров.
  Детальный per-head аудит качества — `DataProcessor/docs/AUDIT_KACHESTVA_FICHEY_CORE_IDENTITY.md`. (Фаза 4)

### Масштабируемость / развёртывание
- [scale] DP api-сервис готов к мульти-ноде: Redis Streams + consumer groups (`queue:high/normal/low`, группа `workers`), `idempotency`, `recovery`, `state_machine`, `checkpoint`. Состояние в Redis, не in-memory. (Фаза 2)
- [scale/deploy] **🔴→✅ КРИТИЧЕСКИЙ ФИКС:** Postgres Service назывался `postgres-service`, но StatefulSet `serviceName: postgres` и ВСЕ клиенты (backend DSN/ES/backups/DP) шли на host `postgres` → DNS не резолвился, весь стек не достучался бы до БД. Переименовал Service в `postgres` + поправил fetcher-DSN. Добавлен линт `k8s/validate_manifests.py` (secretKeyRef/configMap/PVC/image/service-DNS) — прогон OK (57 док, 9 сервисов). Запускать перед `kubectl apply`. (Спринт 1)
- [scale/deploy] **A5 ✅ СДЕЛАНО:** `k8s/infrastructure/backups.yaml` — CronJob `pg_dump` всех БД (trendflow/fetcher_db/embeddings) на backup-PVC с ротацией (RETENTION_DAYS) + Job lifecycle MinIO (TTL сырого видео 30д). result_store чистится retention-CronJob (P8). Валидно. (Спринт 1)
- [scale/deploy] ~~**Embedding Service нет в k8s**~~ **✅ A1 СДЕЛАНО:** `k8s/embedding_service/` (Deployment+Service+faiss-PVC+initContainer создаёт БД `embeddings`), `EMBEDDING_SERVICE_URL` проброшен в DP api/worker; паритет в `docker-compose.prod.yml` (+ `embeddings` БД в `deploy/postgres-init.sql`). Замечен рассинхрон дефолтного порта клиента (8001) vs сервера (8005) — снят явным `EMBEDDING_SERVICE_URL`. (Спринт 1)
- [scale] ~~Базовые публичные модели НЕ в едином репо~~ **✅ A2 СДЕЛАНО:** оркестратор `DataProcessor/scripts/provision_base_models.py` (реестр e5/source_separation/pyannote[gated]/wavlm/wav2vec2/CLAP/places365, стратегии script/hf_snapshot/manual, канонические пути под DP_MODELS_ROOT, `--list/--dry-run/--only`). Док: `DataProcessor/docs/BASE_MODELS_PROVISION.md`. Проверен (--list/--dry-run). Реальная загрузка — на машине с сетью+venv. (Спринт 1)
- [scale/arch] ~~**Дубль очередей**~~ **✅ A3 СДЕЛАНО:** канонический = Redis Streams (`api/services/queue`+`worker`, подключён к `/process`, покрыт тестами). `dp_queue` (Celery) никем не импортируется → помечен deprecated (`dp_queue/__init__.py`), документ `DataProcessor/docs/DATAPROCESSOR_QUEUE_CANONICAL.md`. Кандидат на удаление после проверки внешних вызовов. (Спринт 1)

### Оптимизации
- [opt] AudioProcessor DSP-экстракторы (mel/mfcc/chroma/spectral/tempo/pitch и др.) на librosa = CPU-тяжёлые; на 200k проверить векторизацию/кэш/общий STFT. Также 2 файла с `from_pretrained` (asr render, speaker example) — убедиться, что не в прод-пути (no-network). (Фаза 5)
- [opt] `DataProcessor/main.py` (1500 строк) — монолитный subprocess-оркестратор (Audio/Text/Visual отдельными процессами на видео, ручной сброс CUDA-кэша). Кандидат на профилирование/тёплые воркеры/переиспользование процессов. (Фаза 2)
- [opt] ~~`configs/hf_artifacts_manifest.json` устарел~~ **✅ B6:** помечен `_deprecated` (перекрыт `models_manifest.json`+`download_models.py`). (Спринт 1)
- [deploy] **✅ Линты обоих путей:** `k8s/validate_manifests.py` + `deploy/validate_compose.py` (depends_on/volumes/build/secret/PVC/service-DNS). Оба зелёные; добавлены в DEPLOYMENT.md как шаг перед apply/up. Compose (18 сервисов) без проблем; k8s — после фикса DNS postgres.
- [deploy] **✅ A7:** канонический `docs/DEPLOYMENT.md` (k8s/compose/bootstrap/модели/наблюдаемость/бэкапы), фоновые доки на него ссылаются. (Спринт 1)
- [opt] `d.py` в корне — smoke-тул Fetcher (YouTube metadata), не на месте → перенести в `Fetcher/scripts/` (Фаза 1).
