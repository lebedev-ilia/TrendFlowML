# План доведения TrendFlow до прод-запуска на 200k видео / мульти-нода

Составлен после полного анализа проекта (`PROJECT_MAP.md`, `ANALYSIS_CHECKLIST.md`).
Три ветки: **Логика**, **Масштабируемость/развёртывание**, **Оптимизации**.
Метки: **[я]** — делаю сам · **[владелец]** — только ты · приоритет H/M/L.

Принцип: не ломать контракты (NPZ=SoT, no-network, idempotency, общий `frame_indices`),
каждое изменение — проверяемо (тест/golden-compare/preflight), доки актуализируются.

---

## Ветка A — Масштабируемость / развёртывание (база для 200k мульти-нода)

| # | Задача | Кто | Приор. | Зависит |
|---|---|---|---|---|
| A1 | **Embedding Service в k8s**: Deployment+Service+PVC (персистентный faiss), wiring в DP (`EMBEDDING_SERVICE_URL`) | [я] | H | — |
| A2 | **Провижен публичных базовых моделей offline**: e5-large, wavlm, wav2vec2-base, CLAP-630k, places365, slowfast, pyannote — в bootstrap (`scripts/save_*`/`download_*`) или в репо/манифест | [я] | H | — |
| A3 | **Развести дубль очередей** (`dp_queue` Celery vs `api/services/worker` Redis Streams): зафиксировать канонический, второй — убрать/пометить legacy | [я] | H | — |
| A4 | **Мульти-нодный E2E прогон-канарейка** (1–5к видео): apply -k k8s, model-download Job, KEDA-скейл воркера, проверка S3-shared result_store | [я]+[владелец] | H | A1,A2 |
| A5 | DB-миграции/бэкапы/TTL: backup-CronJob Postgres, lifecycle MinIO, проверить retention | [я] | M | — |
| A6 | Версионирование sampling_policy в Segmenter (в idempotency key) — проверить/добавить | [я] | M | — |
| A7 | Чистка инфра-доков: единый `DEPLOYMENT.md` (свести DEPLOYMENT_GUIDE/QUICKSTART/K8S_FIRST/CONTAINER_GRANULARITY) | [я] | L | — |

## Ветка B — Оптимизации (убрать медленный код без деградации логики)

| # | Задача | Кто | Приор. | Зависит |
|---|---|---|---|---|
| B1 | **Профилирование на канарейке**: `capacity_report.py` + `dataprocessor_component_stage_seconds` → топ-N самых дорогих компонентов | [я] | H | A4 |
| B2 | **main.py: тёплые воркеры / переиспользование процессов** вместо spawn Audio/Text/Visual на каждое видео (сохранив изоляцию памяти) | [я] | H | B1 |
| B3 | **Q4 embeddings-direct** для brand/car/face (галерея 1 раз, локальный `topk_cosine` вместо HTTP-на-кроп); фундамент `_shared/gallery_match.py` готов + тесты | [я] | M | B1, golden-compare |
| B4 | **Audio DSP** (mel/mfcc/chroma/spectral/tempo): общий STFT/кэш, векторизация; проверить librosa vs более быстрые либы | [я] | M | B1 |
| B5 | Triton dynamic batching для тяжёлых core (CLIP/midas/raft) — включить cross-video micro-batching (сейчас batch=1) | [я] | M | B1 |
| B6 | Удалить устаревшее: `configs/hf_artifacts_manifest.json` (→ deprecated в `HF_ARTIFACTS_SYNC.md`), прочий выявленный мусор | [я] | L | — |

## Ветка C — Логика (польза фич для моделей и аналитиков)

| # | Задача | Кто | Приор. | Зависит |
|---|---|---|---|---|
| C1 | **Прогон качества фич на корпусе**: `feature_qa_pipeline.py` (coverage/nan/const/дисперсия/повторяемость) по канарейке → отчёт | [я] | H | A4 |
| C2 | **Feature importance / ablation** против таргета на стабильном `dataset_v1` (baseline → SHAP/важность) → решение какие фичи держать | [я]+[владелец] | H | C1, таргеты |
| C3 | **Semantic-базы**: тулинг готов (`download_reference_images`→`convert`→`build_gallery`→`build_db_manifest`→preflight). Разметка — владелец; places/franchise сейчас без баз | [владелец] | H | — |
| C4 | **Починить spec-несоответствия**: загрузчики под VideoMAE (action_recognition) и HF-whisper; завести wiring emotion_recognition | [я] | M | — |
| C5 | **Аналитический выход**: визуализации/интерпретации/распределения по компонентам для Pro-режима (вход — `PRODUCT_VISION.md`, `WEBSITE_REQUIREMENTS.md`) | [я] | M | C1 |
| C6 | Адаптация алгоритмов под особенности YOLO-датасета | [владелец] | H | YOLO-чекпоинт |
| C7 | Сверка таксономии YOLO v1 ↔ ожидания хедов (logo/text/person/car proposals) | [я] | M | YOLO-чекпоинт |

---

## Рекомендуемый порядок исполнения (мной, автономно)

**Спринт 1 (разблокирует масштаб):** A1 (ES в k8s) → A2 (offline базовые модели) → A3 (очереди) → A5 (бэкапы/TTL) → A6 (sampling_policy).
**Спринт 2 (после канарейки A4 — нужен твой кластер/данные):** B1 (профиль) → B2 (тёплые воркеры) → B4 (DSP) → B5 (triton batching) → C1 (качество фич).
**Спринт 3 (логика):** B3 (Q4) → C4 (spec-загрузчики) → C5 (аналитический выход) → A7/B6 (чистка доков/мусора).
**По мере поступления входов:** C2 (таргеты), C3 (разметка владельцем), C6/C7 (YOLO-чекпоинт).

## Что нужно от владельца (разблокировки)
1. Кластер/ноды (или подтверждение compose-staging) для канарейки A4.
2. Корпус-выборка + таргеты (просмотры/лайки 14/21д) для C1/C2.
3. Ручная разметка semantic-баз (C3) по готовому тулингу.
4. YOLO-чекпоинт (C6/C7) — когда обучится.
5. Ротация YouTube/HF-ключей (были в истории git).

## Definition of Done (проект готов к 200k мульти-нода)
- `kubectl apply -k k8s/` поднимает ВСЁ (вкл. ES), модели провижатся, канарейка проходит зелёный E2E на ≥2 нодах.
- Профиль оптимизирован: per-video GPU-время измерено, топ-узкие места устранены без деградации логики.
- Качество фич доказано на датасете (отчёт + feature importance), мусорные фичи отключены через deprecation.
- Наблюдаемость: дашборды (queue/latency/GPU/ошибки) + алерты + SLO; логи с correlation_id; трейсинг.
- Документация актуальна (PROJECT_MAP + DEPLOYMENT + per-component), мусор убран.
- Открытые зоны владельца (разметка, YOLO-адаптации) — изолированы и не блокируют запуск каркаса.
