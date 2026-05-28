# Главный индекс документации Models

Этот документ служит единой точкой входа для навигации по всей документации обучаемых моделей TrendFlow. Каждый раздел содержит краткое описание документов и ссылки на полные версии.

**Главный индекс проекта**: [../../docs/MAIN_INDEX.md](../../docs/MAIN_INDEX.md) — навигация по всей документации TrendFlowML (DataProcessor, Backend, DynamicBatch, Models, Fetcher)

---

## Контракты

### MODEL_INTERFACE_V2.md
**Краткое описание**: **Финальный (v2) интерфейс Models ↔ DataProcessor**. Фиксирует двоичный путь потребления (tabular FeatureSpec для baseline и TokenStreams/TokenSpec для vNext), правила tokenizer/learned pooling (вместо uniform bins), model-driven sampling через SamplingPlan, privacy/retention требования, и строгие правила versioning (`model_interface_version`, `token_stream_schema_version`, `feature_spec_version`, `token_spec_version`, `sampling_plan_version`).

**Полный документ**: [contracts/MODEL_INTERFACE_V2.md](contracts/MODEL_INTERFACE_V2.md)

### MODEL_CONTRACTS_V1.md
**Краткое описание**: Финальные контракты v1.0 (source-of-truth) для всех обучаемых моделей TrendFlow. Определяет что именно считаем "нашими обучаемыми моделями" (Encoder v0/v1, Baseline, v1 predictor, v2 predictor), prediction time и входные данные (snapshot_0, таргеты как future snapshots 7d/14d/21d, age buckets), таргеты и нормализацию (views/likes, горизонты, log1p функция), split/metrics/golden sets (hybrid time-split, Spearman как north star, holdout и regression mini), архитектуру моделей, feature schema versioning, freeze policy для baseline.

**Полный документ**: [contracts/MODEL_CONTRACTS_V1.md](contracts/MODEL_CONTRACTS_V1.md)

### ENCODER_CONTRACT.md
**Краткое описание**: Контракт Encoder (VisualEncoder + AudioEncoder) — компонента, который приводит variable-length последовательности к fixed-budget представлению для моделей прогноза. Определяет source-of-truth time axis (union_timestamps_sec для visual, seconds для audio), типы входов (dense time-series, sparse events, precomputed embeddings), выходной контракт per modality (global_embedding, summary_tokens, summary_times_s, summary_mask), адаптивные бюджеты (K=64/96/128 по duration_sec, D=768), алгоритм Encoder v0 (deterministic baseline с uniform time-binning и линейной проекцией), план Encoder v1 (trainable, end-to-end с v1 transformer).

**Полный документ**: [contracts/ENCODER_CONTRACT.md](contracts/ENCODER_CONTRACT.md)

### TARGETS_SPLITS_METRICS.md
**Краткое описание**: Определяет таргеты, сплиты и метрики для baseline, v1 и v2 моделей. Описывает prediction time (в любой момент времени, вход snapshot_0), поля snapshot_0 (views_0, likes_0, comments_0, channel stats, comments_text_list_0), таргеты (views/likes на горизонтах 7d/14d/21d, log1p функция для дельт), loss weights (базовые веса и обучаемые веса в v1), сплиты (hybrid time-split по publishedAt + channel-group split), метрики (Spearman как north star, MAE, Spearman по age buckets), golden sets (holdout 2000 видео, regression mini 200 видео).

**Полный документ**: [contracts/TARGETS_SPLITS_METRICS.md](contracts/TARGETS_SPLITS_METRICS.md)

### BASELINE_MODEL.md
**Краткое описание**: Контракт Baseline модели (Boosting) — контрольной точки качества и production fallback в degraded-mode. Определяет входы (7 visual modules, 3 audio extractors, snapshot_0 fields, required core providers), выходы (2 модели views/likes, каждая multi-output на горизонты 7/14/21), freeze policy (запрет изменения алгоритмов после начала baseline dataset collection, обязательный bump feature_schema_version), режимы схемы, версионирование.

**Полный документ**: [contracts/BASELINE_MODEL.md](contracts/BASELINE_MODEL.md)

### V1_TRANSFORMER_MODEL.md
**Краткое описание**: Контракт v1 predictor (Transformers) — основной multimodal модели предсказания, обучаемой end-to-end вместе с trainable encoder. Описывает high-level архитектуру (VisualEncoder, AudioEncoder, text/comments tokens, FusionTransformer с cross-attention, multi-head outputs), time encoding (time embeddings через MLP), обработку comments/text (embeddings per-comment, агрегация в Kc=4..8 tokens, без raw текста), выходы (6 значений views/likes × 7/14/21, masked loss для 7d), uncertainty (quantile heads p10/p50/p90), loss balancing (обучаемые веса горизонтов), compute budget.

**Полный документ**: [contracts/V1_TRANSFORMER_MODEL.md](contracts/V1_TRANSFORMER_MODEL.md)

### V2_CONTEXT_MODEL.md
**Краткое описание**: Контракт v2 predictor (v1 + external context) — модели, которая корректирует v1 prediction с учётом внешнего контекста (trends/news) в воспроизводимом виде. Описывает архитектуру (ContextBuilder строит context_features, ContextAdjustmentModel корректирует v1 prediction), контракт context_features (набор именованных фичей + context_schema_version, сохранение как артефакт run), TTL и деградацию (TTL 48 часов, fallback на v1 при недоступности/просрочке, prediction_status="degraded").

**Полный документ**: [contracts/V2_CONTEXT_MODEL.md](contracts/V2_CONTEXT_MODEL.md)

### MODEL_SYSTEM_RULES.md
**Краткое описание**: Канонические правила для версионирования, кэширования, воспроизводимости и policy-решений вокруг ML-моделей (компоненты DataProcessor и обученные модели прогноза). Определяет термины и версии (producer_version, dataprocessor_version, schema_version, feature_schema_version, model_signature), model signature (обязательные поля models_used[], использование в idempotency/cache key, manifest.json), mapping component → model:version для Triton, правила кэширования и воспроизводимости, error codes и taxonomy, OOM policy, observability.

**Полный документ**: [contracts/MODEL_SYSTEM_RULES.md](contracts/MODEL_SYSTEM_RULES.md)

### PREDICTION_REPORT_CONTRACT.md
**Краткое описание**: Контракт формата prediction_report.json — machine-readable отчёта, который backend отдаёт frontend для отображения прогона модели. Определяет top-level fields (schema_version, job_id, platform_id, video_id, run_ids, timestamps, status, errors), stages (DataCollection, DataProcessor, TextEmbedding, Encoder, Fusion, Heads, Postprocess с таймингами и артефактами), artifacts (список артефактов с путями и метаданными), heads (выходы моделей с интервалами uncertainty), интервалы прогноза (prediction intervals для каждого head), принципы (без raw текста, натуральные единицы для пользователя, model_signature для аудита).

**Полный документ**: [contracts/PREDICTION_REPORT_CONTRACT.md](contracts/PREDICTION_REPORT_CONTRACT.md)

---

## Планы разработки

### plan_dev/README.md
**Краткое описание**: Индекс планов разработки (engineering plans) для обучаемых моделей. Описывает структуру планов (baseline и v1), ссылки на контракты как source-of-truth, расположение документов в plan_dev/.

**Полный документ**: [plan_dev/README.md](plan_dev/README.md)

### plan_dev/BASELINE_DEV_PLAN.md
**Краткое описание**: Полный план разработки Baseline модели (Boosting). Определяет цель baseline (контрольная точка качества, production fallback, быстрый цикл обучения, табличный view для объяснимости), зафиксированные входы/выходы (targets, baseline inputs schema v1.0, baseline outputs), milestones M0-M9 (freeze baseline feature set, DatasetBuilder, обучение baseline, evaluation pipeline, production integration, monitoring, оптимизации), acceptance criteria для каждого milestone, технические детали реализации.

**Полный документ**: [plan_dev/BASELINE_DEV_PLAN.md](plan_dev/BASELINE_DEV_PLAN.md)

### plan_dev/V1_DEV_PLAN.md
**Краткое описание**: Полный план разработки v1 модели (Transformers + trainable Encoder). Определяет цель v1 (multimodal модель с trainable encoder, text tokens, multi-horizon/multi-target прогноз, uncertainty через quantile heads), scope и ключевые инварианты (только snapshot_0 + артефакты DataProcessor, targets log1p(delta), encoder контракт, no-network/fail-fast, reproducibility), milestones V0-V12 (подготовка model-ready датасета, encoder v1, fusion transformer, training pipeline, evaluation, production integration, оптимизации), acceptance criteria, технические детали архитектуры и обучения.

**Полный документ**: [plan_dev/V1_DEV_PLAN.md](plan_dev/V1_DEV_PLAN.md)

---

## Roadmaps

### roadmaps/BASELINE_TO_TRAINING_ROADMAP.md
**Краткое описание**: План доведения TrendFlow/DataProcessor до обучения baseline (CatBoost/LightGBM). Перенесено из DataProcessor/docs/baseline/. Определяет Definition of Done (сквозной прогон DataProcessor, run identity, dataset builder, обучение baseline, evaluation pipeline, production readiness), документы правды (DATAPROCESSOR_AUDIT, BASELINE_IMPLEMENTATION_PLAN, ML_TARGETS_AND_TRAINING, контракты), этапы реализации (подготовка инфраструктуры, валидация артефактов, построение датасета, обучение baseline, evaluation, production integration), acceptance criteria, технические детали.

**Полный документ**: [roadmaps/BASELINE_TO_TRAINING_ROADMAP.md](roadmaps/BASELINE_TO_TRAINING_ROADMAP.md)

---

## Q&A

### QA/CONTRACTS_QA.md
**Краткое описание**: Живой протокол в формате "вопрос → ответ" для фиксации решений по контрактам и системе. Содержит multi-round Q&A (Round 1+) по источникам данных, time semantics, leakage, prediction time, таргетам, сплитам, метрикам, версионированию, кэшированию, воспроизводимости, privacy, edge cases. Формат работы: раунды вопросов → ответы → фиксация решений как FINAL/TEMP/OPEN. Используется для закрытия "серых зон" и фиксации каноничных решений до их включения в контракты/код/CI.

**Полный документ**: [QA/CONTRACTS_QA.md](QA/CONTRACTS_QA.md)

---

## Source Migrations

Документы в `source_migrations/` — это исторические документы, перенесённые из других репозиториев (DataProcessor/docs/) для сохранения контекста и ссылок. Они могут быть устаревшими; актуальные контракты находятся в `contracts/`.

### source_migrations/FEATURE_ENCODER_CONTRACT.md
**Краткое описание**: (Migrated) Feature Encoder Contract v1 — унификация variable-length выходов компонентов для Transformers. Перенесено из DataProcessor/docs/models_docs/. Описывает цель encoder'а (нормализация, выравнивание по времени, сжатие до fixed-budget токенов), модальности (Visual, Audio, Text), входы encoder'а (time-axis, канонические типы выходов компонентов), выходной контракт, алгоритмы encoder v0/v1. Исторический документ; актуальный контракт: `contracts/ENCODER_CONTRACT.md`.

**Полный документ**: [source_migrations/FEATURE_ENCODER_CONTRACT.md](source_migrations/FEATURE_ENCODER_CONTRACT.md)

### source_migrations/ML_TARGETS_AND_TRAINING.md
**Краткое описание**: (Migrated) ML targets и обучение (полуфинал). Перенесено из DataProcessor/docs/baseline/. Описывает таргеты (multi-target views/likes, multi-horizon 7/14/21d), вычисление таргетов (дельты, log1p нормализация), входы модели и leakage (только snapshot1), cold-start и возраст видео, сплиты (time-split + channel-group split), архитектуры (baseline → v2). Исторический документ; актуальный контракт: `contracts/TARGETS_SPLITS_METRICS.md`.

**Полный документ**: [source_migrations/ML_TARGETS_AND_TRAINING.md](source_migrations/ML_TARGETS_AND_TRAINING.md)

### source_migrations/MODEL_SYSTEM_RULES.md
**Краткое описание**: (Migrated) Model system rules (MVP) — полуфинал. Перенесено из DataProcessor/docs/models_docs/. Описывает термины и версии, model signature, mapping component → model:version, правила кэширования, error codes, OOM policy, observability. Исторический документ; актуальный контракт: `contracts/MODEL_SYSTEM_RULES.md`.

**Полный документ**: [source_migrations/MODEL_SYSTEM_RULES.md](source_migrations/MODEL_SYSTEM_RULES.md)

### source_migrations/MODELS_Q.md
**Краткое описание**: (Migrated) MODELS_Q (Q&A). Перенесено из DataProcessor/docs/models_docs/. Содержит Q&A Round 1+ по версионированию моделей, воспроизводимости, Triton, кэшированию, совместимости версий, error handling, observability. Исторический документ; актуальные решения: `QA/CONTRACTS_QA.md` и `contracts/MODEL_SYSTEM_RULES.md`.

**Полный документ**: [source_migrations/MODELS_Q.md](source_migrations/MODELS_Q.md)

---

## Структура документации

Models/docs организован по функциональным разделам:

- **`contracts/`**: Финальные контракты v1.0 (source-of-truth) для всех обучаемых моделей
- **`plan_dev/`**: Планы разработки (engineering plans) для baseline и v1
- **`roadmaps/`**: Долгосрочные планы и milestones
- **`QA/`**: Q&A протоколы для фиксации решений
- **`source_migrations/`**: Исторические документы, перенесённые из других репозиториев

---

## Интеграция с DataProcessor

Models интегрирован с DataProcessor:

- **Входные данные**: Models читает артефакты (NPZ) из DataProcessor result_store
- **Encoder**: Encoder работает после всех компонентов DataProcessor, приводит variable-length последовательности к fixed-budget
- **Feature schema**: Models использует feature_schema_version для версионирования схемы фичей
- **Model signature**: Models следует правилам MODEL_SYSTEM_RULES.md для версионирования и воспроизводимости
- **Targets**: Models использует snapshot_0 и future snapshots для построения таргетов
- **Artifacts**: Models сохраняет prediction_report.json и другие артефакты в result_store

---

## Статистика

- **Всего контрактов**: 8
- **Планов разработки**: 2 (baseline, v1)
- **Roadmaps**: 1
- **Q&A документов**: 1
- **Source migrations**: 4
- **Реализованных моделей**: 2 (baseline, v1)
- **Baseline компонентов**: 3 модуля (Training, Inference, common)
- **v1 компонентов**: 6 модулей (encoder, model, training, data, text, common)

---

## Реализация моделей

### baseline/
**Краткое описание**: Реализация Baseline модели (Boosting) — контрольной точки качества и production fallback в degraded-mode. Следует контракту `contracts/BASELINE_MODEL.md` и плану разработки `plan_dev/BASELINE_DEV_PLAN.md`. Baseline использует табличные фичи из DataProcessor артефактов (7 visual modules, 3 audio extractors, snapshot_0 fields, required core providers), обучает 2 модели (views/likes) с multi-output на горизонты 7/14/21d, поддерживает CatBoost и sklearn как fallback.

**Расположение**: `Models/baseline/`

**Основные компоненты**:
- **Training/**: Обучение baseline моделей (M5 из плана разработки)
- **Inference/**: Инференс для production (M6 из плана разработки)
- **common/**: Общие утилиты для извлечения фичей из NPZ
- **README.md**: Полная документация по использованию baseline

#### baseline/README.md
**Краткое описание**: Основная документация Baseline модели. Описывает расположение компонентов (dataset builder в DataProcessor/DatasetBuilder/, training/evaluation/inference в Models/baseline/), контракты (targets, splits, quality gate), quickstart end-to-end (сбор датасета, обучение, генерация golden sets, evaluation, инференс), формат артефактов обучения (training_run_manifest.json, bundles для views/likes, метрики).

**Расположение**: `Models/baseline/README.md`

#### baseline/Training/
**Краткое описание**: Модуль обучения Baseline моделей (M5 из `plan_dev/BASELINE_DEV_PLAN.md`). Загружает baseline dataset, выполняет hybrid split (time + channel-group), обучает baseline regressor per output (views/likes × 7/14/21d, 7d masked), сохраняет воспроизводимые артефакты моделей и метрики. Поддерживает CatBoost и sklearn как fallback.

**Расположение**: `Models/baseline/Training/`

**Основные файлы**:
- **train_baseline.py**: Главный скрипт обучения baseline моделей. Загружает dataset, выполняет split, обучает модели для views/likes на горизонтах 7/14/21d, сохраняет training_run_manifest.json и артефакты моделей.
- **evaluate_baseline.py**: Скрипт evaluation (quality gate). Вычисляет метрики (Spearman на log1p(delta), MAE, Spearman по age buckets), генерирует metrics.json и report.md для различных eval sets (test, holdout, regression_mini).
- **generate_golden_sets.py**: Генератор фиксированных evaluation sets (golden sets), ключённых по dataset_fingerprint. Создаёт holdout (2000 видео) и regression_mini (200 видео) для воспроизводимой оценки качества.
- **smoke_e2e.py**: End-to-end smoke тест: dataset → train → eval(regression_mini) → predict(one run). Проверяет корректность всего пайплайна baseline.
- **utils_metrics.py**: Утилиты для вычисления метрик (Spearman, MAE, метрики по age buckets).
- **README.md**: Документация по использованию Training модуля.

#### baseline/Inference/
**Краткое описание**: Модуль инференса Baseline моделей (M6 из `plan_dev/BASELINE_DEV_PLAN.md`). Извлекает фичи из per-run артефактов (manifest + NPZ) используя ту же логику, что и training, загружает обученные baseline артефакты, генерирует детерминированный prediction JSON для backend/UI. Поддерживает enforcement required components и degraded mode.

**Расположение**: `Models/baseline/Inference/`

**Основные файлы**:
- **predict_baseline.py**: Главный скрипт инференса. Читает артефакты из result_store, извлекает фичи через npz_features, загружает обученные модели, генерирует prediction JSON с prediction_status (ok/degraded), missing_required_components, model_version, feature_schema_version, predictions_log1p_delta (views/likes × 7/14/21d).
- **README.md**: Документация по использованию Inference модуля.

#### baseline/common/
**Краткое описание**: Общие утилиты для Baseline, используемые в training и inference. Изолированы от зависимостей DataProcessor python-модулей для независимости.

**Расположение**: `Models/baseline/common/`

**Основные файлы**:
- **npz_features.py**: Извлечение фичей из NPZ артефактов без зависимости от DataProcessor модулей. Содержит функции для чтения NPZ файлов, извлечения числовых массивов, агрегации статистик (mean, max, quantiles), обработки meta секций, поиска NPZ артефактов по component name.

### v1/
**Краткое описание**: Реализация v1 модели (Transformers + trainable Encoder) — основной multimodal модели предсказания. Следует контрактам `contracts/V1_TRANSFORMER_MODEL.md` и `contracts/MODEL_CONTRACTS_V1.md`, плану разработки `plan_dev/V1_DEV_PLAN.md`. v1 использует trainable encoder (VisualEncoder/AudioEncoder) end-to-end, учитывает text/comments через несколько text tokens (Kc=4..8), выдаёт multi-horizon/multi-target прогноз (6 значений), uncertainty через quantile heads (p10/p50/p90).

**Расположение**: `Models/v1/`

**Основные компоненты**:
- **encoder/**: Encoder v0 (deterministic) и v1 (trainable)
- **model/**: Архитектура v1 transformer модели
- **training/**: Обучение и evaluation v1
- **data/**: Построение v1 dataset index
- **text/**: Обработка text/comments tokens
- **common/**: Общие утилиты (split, bigjson utils)
- **README.md**: Полная документация по использованию v1

#### v1/README.md
**Краткое описание**: Основная документация v1 модели. Описывает mapping milestones к коду (V0-V5), требования к входным данным (v1_dataset_index.parquet с указателями на артефакты, snapshot_0 полями, таргетами; v1_text_index.parquet опционально), quick start для skeleton, использование text tokens (V3), trainable encoder (V4), quantile heads (V5), golden sets для evaluation.

**Расположение**: `Models/v1/README.md`

#### v1/encoder/
**Краткое описание**: Реализация Encoder (VisualEncoder + AudioEncoder) для приведения variable-length последовательностей к fixed-budget представлению. Encoder v0 — deterministic baseline с uniform time-binning и линейной проекцией. Encoder v1 — trainable, обучается end-to-end вместе с v1 transformer.

**Расположение**: `Models/v1/encoder/`

**Основные файлы**:
- **encoder_v0.py**: Реализация Encoder v0 (deterministic) согласно `contracts/ENCODER_CONTRACT.md`. Выполняет uniform time-binning на K бинов (адаптивно 64/96/128 по duration_sec), вычисляет статистики (mean, max, p50, p90) для каждого ряда в каждом бине, линейную проекцию в D=768, pooled representation для global_embedding. Сложность O(N) по длине входной последовательности.
- **encoder_v1.py**: Реализация Encoder v1 (trainable). Обучается end-to-end вместе с v1 transformer, использует learnable параметры для проекции и pooling. Следует тому же контракту выхода, что и encoder_v0.

#### v1/model/
**Краткое описание**: Архитектура v1 transformer модели. Реализует multimodal transformer system с cross-attention fusion между модальностями, multi-head outputs для views/likes × 7/14/21d, quantile heads для uncertainty (p10/p50/p90).

**Расположение**: `Models/v1/model/`

**Основные файлы**:
- **v1_skeleton.py**: Skeleton архитектуры v1 transformer. Определяет структуру модели (VisualEncoder, AudioEncoder, TextEncoder, FusionTransformer с cross-attention, multi-head outputs, quantile heads), forward pass, loss computation с обучаемыми весами горизонтов, uncertainty estimation через quantile regression.

#### v1/training/
**Краткое описание**: Модуль обучения и evaluation v1 модели. Реализует training loop для v1 transformer, evaluation pipeline с метриками (Spearman, MAE, quantile coverage), генерацию golden sets, smoke тесты.

**Расположение**: `Models/v1/training/`

**Основные файлы**:
- **train_v1_skeleton.py**: Главный скрипт обучения v1. Загружает v1_dataset_index, опционально v1_text_index, инициализирует encoder (v0 или v1), v1 transformer, выполняет training loop с loss balancing, сохраняет checkpoints, метрики, training manifest.
- **evaluate_v1.py**: Скрипт evaluation v1. Загружает checkpoint, вычисляет метрики (Spearman на log1p(delta), MAE, quantile coverage для p10/p50/p90), генерирует отчёты для различных eval sets (test, holdout, regression_mini).
- **generate_v1_golden_sets.py**: Генератор фиксированных evaluation sets для v1, ключённых по v1_dataset_fingerprint. Создаёт holdout и regression_mini для воспроизводимой оценки качества.

#### v1/data/
**Краткое описание**: Модуль построения v1 dataset index (V0 из плана разработки). Создаёт "лёгкий" индекс с указателями на per-run артефакты (manifest + key NPZ paths), snapshot_0/meta полями, таргетами/masks.

**Расположение**: `Models/v1/data/`

**Основные файлы**:
- **build_v1_dataset_index.py**: Скрипт построения v1_dataset_index.parquet. Читает data.json с видео, result_store с артефактами, извлекает указатели на core_clip_npz_path, segmenter_metadata_path, snapshot_0 поля (views_0, likes_0, comments_0, channel stats, duration_sec, publishedAt), таргеты (target_views_{7d|14d|21d}, target_likes_{7d|14d|21d} на log1p(delta) scale), masks. Сохраняет v1_dataset_index.parquet и v1_dataset_metadata.json с fingerprint.

#### v1/text/
**Краткое описание**: Модуль обработки text/comments tokens для v1 (V3 из плана разработки). Строит per-video text artifacts и index mapping без raw текста (только embeddings и агрегаты).

**Расположение**: `Models/v1/text/`

**Основные файлы**:
- **build_text_embeddings.py**: Скрипт построения v1_text_index.parquet и text NPZ артефактов. Читает comments_text_list_0 из data.json, извлекает embeddings per-comment через sentence-transformers (no-network policy), агрегирует в Kc=4..8 tokens (attention pooling/top-K информативных), сохраняет text_tokens и text_mask в NPZ, создаёт v1_text_index с указателями на text_npz_path.
- **text_encoder.py**: Утилиты для encoding текста в tokens. Реализует извлечение embeddings, агрегацию в фиксированное количество tokens, обработку missing/empty комментариев.

#### v1/common/
**Краткое описание**: Общие утилиты для v1, используемые в различных модулях.

**Расположение**: `Models/v1/common/`

**Основные файлы**:
- **split.py**: Утилиты для hybrid split (time-split по publishedAt + channel-group split по channel_id). Реализует детерминированное разбиение датасета на train/val/test/holdout с учётом временных и канальных зависимостей.
- **utils_bigjson.py**: Утилиты для работы с большими JSON файлами (data.json). Реализует streaming чтение, фильтрацию подмножеств видео, обработку больших объёмов данных без загрузки всего файла в память.

### __init__.py
**Краткое описание**: Python package marker для Models. Экспортирует публичный API модуля Models (если требуется в будущем).

**Расположение**: `Models/__init__.py`

---

## Связь с DataProcessor: Audit v4 (эмпирическая оценка фич)

Когда в DataProcessor обновляют экстракторы и NPZ, коммутирующие контракты моделей — **второй слой проверки** после Audit v3: статистика реальных выходов, вырожденные распределения, полезность полей для `ENCODER_CONTRACT` / tabular FeatureSpec. Источник процедуры и критериев:

- [`DataProcessor/docs/audit_v4/AUDIT_4_CRITERIA_AND_PLAN.md`](../../DataProcessor/docs/audit_v4/AUDIT_4_CRITERIA_AND_PLAN.md)

---

## Быстрая навигация

### Для начала работы
1. Прочитайте [contracts/MODEL_CONTRACTS_V1.md](contracts/MODEL_CONTRACTS_V1.md) — общий обзор всех моделей
2. Изучите [contracts/ENCODER_CONTRACT.md](contracts/ENCODER_CONTRACT.md) — контракт Encoder
3. Ознакомьтесь с [contracts/TARGETS_SPLITS_METRICS.md](contracts/TARGETS_SPLITS_METRICS.md) — таргеты и метрики

### Для разработки baseline
1. [contracts/BASELINE_MODEL.md](contracts/BASELINE_MODEL.md) — контракт baseline
2. [plan_dev/BASELINE_DEV_PLAN.md](plan_dev/BASELINE_DEV_PLAN.md) — план разработки
3. [roadmaps/BASELINE_TO_TRAINING_ROADMAP.md](roadmaps/BASELINE_TO_TRAINING_ROADMAP.md) — roadmap до обучения

### Для разработки v1
1. [contracts/V1_TRANSFORMER_MODEL.md](contracts/V1_TRANSFORMER_MODEL.md) — контракт v1
2. [plan_dev/V1_DEV_PLAN.md](plan_dev/V1_DEV_PLAN.md) — план разработки

### Для вопросов и уточнений
1. [QA/CONTRACTS_QA.md](QA/CONTRACTS_QA.md) — Q&A протоколы
2. [contracts/MODEL_SYSTEM_RULES.md](contracts/MODEL_SYSTEM_RULES.md) — системные правила

### Для работы с кодом
1. [baseline/README.md](../../baseline/README.md) — документация Baseline реализации
2. [v1/README.md](../../v1/README.md) — документация v1 реализации
3. [baseline/Training/](../../baseline/Training/) — обучение baseline
4. [baseline/Inference/](../../baseline/Inference/) — инференс baseline
5. [v1/training/](../../v1/training/) — обучение v1
6. [v1/encoder/](../../v1/encoder/) — Encoder v0/v1

