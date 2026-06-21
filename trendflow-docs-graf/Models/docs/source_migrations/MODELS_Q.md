## (Migrated) MODELS_Q (Q&A)

Источник: `DataProcessor/docs/models_docs/MODELS_Q.md` (перенесено без смысловых правок).

---

## TrendFlow — MODELS Q&A (Round 1)

Формат работы:
- Я добавляю вопросы в конец файла.
- Ты отвечаешь прямо под каждым вопросом (кратко, но однозначно).
- Если по ответам появляются новые вопросы — я дописываю их ниже отдельным блоком "Round N+1".

---

### Round 1 — вопросы

#### 1) Версионирование моделей и воспроизводимость

- **Q1. Версионирование моделей в NPZ meta**: в документации упоминается, что в `meta` каждого NPZ нужно фиксировать `model_name` и `model_version` (если использовался Triton/ML модель). Как именно это должно работать:
  - для core providers (CLIP, MiDaS, YOLO, RAFT, Mediapipe) — фиксируем версию модели в `producer_version` или отдельное поле `model_version`?
  - для модулей, использующих модели (emotion_face, scene_classification, etc.) — как версионируем?
  - для моделей из HuggingFace/transformers — используем полный путь (например, `"openai/clip-vit-base-patch32"`) или отдельно `model_name` + `model_version`?
  - **A**:
    - `producer_version` = версия кода компонента (semver/commit), **не** версия модели.
    - В `meta` пишем отдельный блок `models_used` (список), где **на каждую реально вызванную ML-модель** фиксируем:
      - `model_name` (каноническое имя: HF repo id / Triton model name / локальный алиас)
      - `model_version` (строго: HF `revision`/commit sha, Triton `version`, или наш semver/tag)
      - `weights_digest` (sha256/etag/commit) — чтобы различать «одинаковый version, разные веса»
      - `runtime` (`triton` | `inprocess`) + `engine` (`torch`/`onnx`/`tensorrt`) + `precision` (`fp32`/`fp16`) + `device` (`cuda:0`/`cpu`)
    - **Core providers**: пишем все под‑модели, которые они используют (например, `core_clip` → CLIP weights + tokenizer).
    - **Модули**: пишем те модели, которые они вызывают напрямую (если модуль только читает артефакт core provider — у модуля `models_used` может быть пустым, а зависимость фиксируем через `dependencies` в `meta`).
    - **HuggingFace/transformers**: `model_name` = полный путь (`"openai/clip-vit-base-patch32"`), `model_version` = pinned `revision` (commit sha или tag), не “latest”.

- **Q2. Версионирование моделей в Triton**: в `PRODUCTION_ARCHITECTURE.md` указано, что версии моделей фиксируются в `dataprocessor_version` (все run'ы одной версии DataProcessor используют одинаковые версии моделей). Но как это согласуется с возможностью обновления отдельных моделей:
  - если мы обновили только одну модель (например, YOLO с v11.0 на v11.1), нужно ли бампить `dataprocessor_version`?
  - или `dataprocessor_version` бампится только при изменении всего пайплайна, а версии отдельных моделей хранятся в `triton_models.yaml` и прокидываются в `meta` NPZ?
  - **A**:
    - `dataprocessor_version` бампим **только** при изменении кода пайплайна/контрактов/схем (producer/schema/feature extraction), а **не** при апдейте одной модели.
    - Версии моделей живут в `triton_models.yaml` (или в DB-конфиге профиля анализа) и **пинятся на run** (run получает конкретный mapping `component → model:version`).
    - В `meta` NPZ всегда сохраняем **фактический** mapping, использованный в этом run (см. Q1), чтобы воспроизводимость не зависела от “текущего” `triton_models.yaml`.

- **Q3. Совместимость версий моделей при кэшировании**: если у нас есть кэш артефакта от старой версии модели, но мы обновили модель:
  - должны ли мы автоматически инвалидировать кэш (пересчитать артефакт с новой моделью)?
  - или разрешаем использовать старый кэш, но помечаем в `manifest.json`, что артефакт создан со старой версией?
  - как это влияет на idempotency ключ `(platform_id, video_id, run_id, component, config_hash, sampling_policy_version, producer_version, schema_version, model_version*)`?
  - **A**:
    - Idempotency/cache key **обязан включать** `model_signature` (минимум: `model_name + model_version + weights_digest + engine + precision`), иначе кэш станет невалидным при апдейте модели.
    - Базовая политика: **любое изменение `model_signature` = новый ключ = автоматическая “инвалидация”** (старый артефакт остаётся, но не переиспользуется).
    - Опционально (позже): “мягкая совместимость” через `model_compatibility_token`, который явно объявляет, что output совместим (тогда можно разрешить reuse), но это требует явного решения/тестов.
    - В `manifest.json` всегда пишем: `producer_version`, `schema_version`, `model_signature`, `cache_hit` и ссылку на исходный артефакт (если reused).

- **Q4. Версионирование baseline/v1/v2 моделей прогноза**: в `ML_TARGETS_AND_TRAINING.md` описаны три уровня моделей (baseline CatBoost/LightGBM, v1 late-fusion, v2 multimodal transformer). Как версионируем эти модели:
  - отдельные версии для каждого типа (например, `baseline_v1.2.3`, `v1_late_fusion_v2.0.1`, `v2_transformer_v3.1.0`)?
  - или единая версия для всех типов, но с полем `model_type`?
  - как это отражается в inference pipeline и в JSON для фронта?
  - **A**:
    - Версионируем **раздельно по типам**: `baseline_*`, `v1_late_fusion_*`, `v2_transformer_*` (semver + training run id).
    - В inference pipeline всегда явно выбираем `(prediction_model_type, prediction_model_version)` из конфигурации run.
    - В JSON для фронта добавляем блок:
      - `prediction.model.type`, `prediction.model.version`, `prediction.model.run_id`
      - `prediction.model.features_schema_version` (важно для отладки)

#### 2) Развертывание моделей и Triton

- **Q5. Миграция на Triton (план)**: сейчас модели загружаются напрямую в процесс (например, через `torch.load`, `transformers.from_pretrained`, `model_registry`). Когда и как планируем мигрировать на Triton:
  - все модели сразу или поэтапно (сначала самые тяжёлые)?
  - какие модели приоритетны для Triton (GPU-heavy: CLIP, YOLO, emotion_face, scene_classification)?
  - как обеспечим обратную совместимость во время миграции (fallback на прямую загрузку, если Triton недоступен)?
  - **A**: Модели переводим на Triton для каждой глобальной смены моделей, например для baseline нужны 5 моделей, вот их и переводим. Далее baseline обучили, прошли полностью, переходим на v1, где нужны уже 10 моделей их и переводим. Пока никаких Fallback, Triton обязателен.

- **Q6. Конфигурация Triton models (детали)**: в `triton_models.yaml` нужно хранить mapping `component_name → model_name:version`. Но как это работает для:
  - моделей с несколькими вариантами (например, CLIP: `ViT-B/32`, `ViT-L/14`, `RN50` — какой выбираем для `core_clip`)?
  - моделей с параметрами (например, YOLO: `yolo11n.pt`, `yolo11s.pt`, `yolo11x.pt` — выбор зависит от `config.yaml`)?
  - моделей, которые могут быть разными в разных профилях анализа (например, `scene_classification` может использовать `resnet18` или `efficientnet_b0`)?
  - **A**: Выбор моделей будет в ЛК на сайте также как и их настройка. У нас должны быть скомпелированы все модели для всех возможных параметров которые достпны на сайте (какие конкретно определим позже). Как тебе план, давай рекомендации.

- **Q7. Triton batching и динамический batch size**: Triton поддерживает dynamic batching. Как это интегрируется с нашим динамическим батчингом на уровне DataProcessor:
  - Triton сам решает batch size или мы передаём готовый batch?
  - как учитываем ресурсные требования компонентов (`component_resource_requirements` в БД) при формировании batch для Triton?
  - нужно ли конфигурировать `max_batch_size` в Triton model config отдельно для каждого компонента?
  - **A**: У нас своя система батчирования основаная больше на оптимизации памяти. Наши батчи мы передаем в Triton (они уже оптимальны для доступной памяти). Остальное реши сам.

- **Q8. Triton health checks и fallback**: если Triton недоступен или модель не загружена:
  - должны ли компоненты автоматически fallback на прямую загрузку модели (как сейчас)?
  - или это считается ошибкой и run должен упасть (согласно no-fallback policy)?
  - как различаем "Triton временно недоступен" (retry) от "модель не найдена" (fail-fast)?
  - **A**: Triton будет запускаться локально и должен быть достпуен всегда. Никаких fallback.

- **Q9. Triton model repository и обновления**: как обновляем модели в Triton:
  - через CI/CD при деплое DataProcessor (автоматически)?
  - вручную через админку/API (для hot-fix)?
  - как обеспечиваем zero-downtime обновление (rolling update, A/B тестирование)?
  - нужно ли версионировать model repository (например, `models/emotion_net/v1.2/` vs `models/emotion_net/v1.3/`)?
  - **A**:
    - Обновление моделей: **через CI/CD** (основной путь) + **ручной hot-fix** (ограниченно, с аудит‑логом).
    - Model repository **версионируем папками**: `models/<model_name>/<version>/...` + отдельный `config.pbtxt`/metadata per version.
    - Zero-downtime: держим **старую и новую версии параллельно**, переключение происходит на уровне `triton_models.yaml`/DB профиля (A/B по пользователям/профилям возможно).
    - После прогрева/валидации новую версию делаем default, старую — оставляем на “grace period”, потом архивируем.

---

### Round 2 — вопросы (следующие уточнения перед реализацией)

#### 1) Кэш, совместимость и ключи

- **Q1. Политика “model_compatibility_token”**: хотим ли мы сразу вводить совместимость между версиями модели (reuse артефактов при апдейте), или в MVP фиксируем правило “новая модель = новый кэш” без исключений?
  - **A**: в MVP фиксируем правило “новая модель = новый кэш” без исключений

- **Q2. Детерминизм**: какой допуск по воспроизводимости считаем нормой (например, “строго одинаково” vs “допускаем небольшие fp16/engine расхождения”)?
  - **A**: допускаем небольшие fp16/engine расхождения, но:
    - на одном и том же железе/engine/precision должны получать “практически одинаковый” результат (детерминизм best-effort)
    - для кросс-окружений (другая GPU/другой engine) считаем это другой версией (`model_signature` отличается), поэтому сравнение делаем с tolerances
    - в `meta/manifest` фиксируем: seed, engine, precision, device, версии CUDA/cuDNN (если применимо)

- **Q3. Храним ли старые артефакты**: сколько времени/версий артефактов держим (TTL/GC), и можно ли удалять всё, что не относится к “последним N версиям моделей/профилей”?
  - **A**: да, делаем GC:
    - hard cap: `hard_cap_days = 60`
    - держим минимум: последний успешный `run_id` на ключ `(platform_id, video_id, config_hash, sampling_policy_version, model_signature-set)` + N=2 предыдущих
    - промежуточные/временные файлы (frames_dir и т.п.) — короткий TTL (7 дней)
---

## Навигация

[Models](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
