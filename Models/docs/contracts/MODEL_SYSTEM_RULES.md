## Model system rules (MVP)

Этот документ задаёт **канонические правила** для версионирования, кэширования, воспроизводимости и policy-решений вокруг ML-моделей (как компонентов DataProcessor, так и наших обученных моделей прогноза).

Источник решений: ранее велся в `DataProcessor/docs/models_docs/MODEL_SYSTEM_RULES.md` и Q&A `MODELS_Q.md`. Перенесено в `Models/docs/` как source-of-truth.

---

### 1) Термины и “версии”

- **`producer_version`**: версия кода конкретного компонента (core provider / module / audio / text).
- **`dataprocessor_version`**: версия оркестратора/пайплайна (DataProcessor релиз). Не равна версии моделей.
- **`schema_version`**: версия схемы NPZ артефакта.
- **`feature_schema_version`**: версия схемы фичей для моделей прогноза (training/inference).
- **`model_signature`**: идентификатор “какой именно моделью и как” был получен результат.

Ключевое правило: апдейт одной модели **не требует** bump `dataprocessor_version`. Версии моделей живут отдельно и входят в `model_signature`.

---

### 2) Model signature (обязательно)

Для каждого NPZ артефакта в `meta` фиксируем список `models_used[]` (может быть пустым, если компонент моделей не вызывал).

Для каждой записи `models_used[]` фиксируем минимум:
- `model_name`
- `model_version` (pinned)
- `weights_digest`
- `runtime`: `triton` | `inprocess`
- `engine`: `torch` | `onnx` | `tensorrt`
- `precision`: `fp32` | `fp16`
- `device`: `cuda:0` / `cuda:1` / `cpu`

`model_signature` компонента = функция от (`models_used` + `engine/precision/device`) и используется:
- в idempotency/cache key
- в `manifest.json`
- для дебага/аудита

Важно: смена `engine` (например, TensorRT) считается другой версией (другой `model_signature`).

---

### 3) Mapping `component → model:version` (Triton)

- source-of-truth: профиль анализа в БД
- любые YAML/seed-конфиги допустимы только для dev/bootstrap
- per-run сохраняем resolved mapping в `manifest.json` и/или в `meta` NPZ

---

### 4) Кэширование и idempotency

Политика MVP: **“новая модель = новый кэш” без исключений**.

Idempotency/cache key компонента должен включать:
- `platform_id`, `video_id`
- `component`
- `config_hash`
- `sampling_policy_version`
- `producer_version`
- `schema_version`
- `model_signature`

---

### 5) Детерминизм

- допускаем небольшие расхождения из-за FP16/движка/разных GPU
- в пределах одного окружения детерминизм best-effort
- в meta/manifest фиксируем seed, engine, precision, device, версии CUDA/cuDNN (если применимо)

---

### 6) Retention / GC

- hard cap: 60 дней
- держим минимум: последний успешный `run_id` на ключ `(platform_id, video_id, config_hash, sampling_policy_version, model_signature-set)` + N=2 предыдущих
- временные данные (например `frames_dir`) — короткий TTL (например 7 дней)

---

### 7) Ошибки загрузки моделей / Triton

- no-fallback: выбранная модель недоступна → fail-fast
- retry допускается только для transient ошибок (timeout/503)
- в manifest фиксируем `error_code` (`model_load_failed`, `insufficient_gpu_memory`, `triton_unavailable`, ...)

---

### 8) OOM / batch_size / resource-aware execution

- batch_size выбираем **до запуска** на основе доступной памяти + чеклиста
- использованный batch_size фиксируем в meta
- при OOM допускается уменьшение batch_size и retry (если поддерживается)

---

### 9) Precision / engine

- default GPU precision: FP16 (если валидировано), иначе FP32
- precision/engine настраиваются per model и входят в `model_signature`

---

### 10) Multi-GPU scheduling

Принято: компоненты одного run могут выполняться на разных GPU.

---

### 11) Prediction: выбор модели, fallback, confidence

- default: самая новая stable (обычно v2)
- degraded-mode: если v2 упал → пробуем v1 → затем baseline
- confidence: дефолт `confidence_score` + интервал p10/p90 (если посчитано)


