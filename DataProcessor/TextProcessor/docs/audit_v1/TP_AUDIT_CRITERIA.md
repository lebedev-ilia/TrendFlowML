# Критерии аудита TextProcessor и его Extractors (TP audit criteria)

**Версия**: v1.0  
**Дата**: 2026-01-29  
**Применимо к**: `DataProcessor/TextProcessor` (процессор) и всем `TextProcessor/src/extractors/*` (экстракторам)

---

## 0. Обзор: что мы аудируем и зачем

Этот документ фиксирует **обязательные критерии** для приведения TextProcessor к baseline‑контрактам TrendFlow/DataProcessor и дальнейшего доведения каждого extractor до полуфинального “production‑grade” уровня.

Аудит должен подтвердить, что:
- TextProcessor и каждый extractor **соответствуют контрактам входа/выхода**, storage и meta.
- Соблюдается **no‑fallback policy** (ошибки — явные, empty — валидная пустота).
- Соблюдается **privacy/no‑raw по умолчанию**, а любые raw‑режимы — строго gated.
- Любые ML‑модели загружаются **только через `dp_models` (ModelManager)**, без сетевых загрузок.
- Есть воспроизводимость, измерения производительности, quality‑валидация и UI‑рендер (presentation layer) из NPZ.

---

## 0.1 Термины и уровни аудита

### A) Уровень “TextProcessor компонент”

Код‑входные точки и orchestration:
- `TextProcessor/run_cli.py`
- `TextProcessor/src/core/main_processor.py` (оркестратор экстракторов)
- `TextProcessor/src/schemas/models.py` (контракт `VideoDocument`)
- интеграция с `manifest.json` + `artifact_validator`

### B) Уровень “Extractor компонент”

Подкомпонент TextProcessor, реализующий `BaseExtractor.extract(doc) -> Dict` и возвращающий:
- `result` (фичи/служебные данные для последующих экстракторов),
- `timings_s` (тайминги),
- `system` (снапшот ресурсов),
- `device`, `version`, `error` (если применимо).

**Важно**: в baseline аудит делается **по каждому extractor**: он считается “мини‑компонентом” со своими входами/зависимостями/моделями/кэшем/фичами/производительностью.

---

## 0.2 Source-of-truth документы (обязательные ссылки)

- Базовые критерии: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- Контракты:  
  - `docs/contracts/CONTRACTS_OVERVIEW.md`  
  - `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`  
  - `docs/contracts/ERROR_HANDLING_AND_EDGE_CASES.md`  
  - `docs/contracts/PRIVACY_AND_RETENTION.md`  
  - `docs/contracts/LLM_RENDERING.md`
- Модельная система: `dp_models/*` + `docs/models_docs/*` (ModelManager правила)

---

## 1. Архитектурное соответствие (TextProcessor / Extractor)

### 1.1 Интерфейсы и границы ответственности

**TextProcessor как компонент**:
- [ ] Имеет CLI entrypoint (argparse) и может быть вызван оркестратором DataProcessor.
- [ ] Работает строго в рамках **per-run result_store** (см. 1.4).
- [ ] Обновляет `manifest.json` через общий механизм (upsert) и **не пишет произвольные JSON артефакты** в `result_store` (кроме `manifest.json`).
- [ ] **Строгая загрузка экстракторов (обязательное правило)**: если extractor указан в конфиге, но не смог импортироваться/создаться → run должен завершаться `status="error"` (fail-fast), а не “тихий skip”.

**Extractor как компонент**:
- [ ] Реализует `BaseExtractor.extract(doc) -> Dict[str, Any]`.
- [ ] Не делает “скрытых” глобальных сайд‑эффектов (сетевые вызовы, скачивание моделей, запись raw текста в логи/файлы) без явной политики/флага.
- [ ] Если extractor требует специфичный вход (например, transcript или comments) — это требование **декларировано** и **проверяется fail-fast** (см. 1.3 / 1.6).

**Evidence**:
- путь к файлам + номера строк (интерфейс, проверки, место вызова в pipeline);
- пример запуска `TextProcessor/run_cli.py` с минимальными флагами.

### 1.2 Контракты входа (VideoDocument)

- [ ] Входная единица обработки: **один `VideoDocument`** (JSON) по схеме `TextProcessor/src/schemas/models.py`.
- [ ] Источники текста (title/description/transcripts/comments) явно интерпретируются как:
  - **required** для конкретного extractor’а или профиля,
  - **optional** (валидный empty).
- [ ] Для transcript допускается privacy‑safe форма: `transcripts_token_ids` → декодирование через `dp_models` (shared tokenizer), без сетевых загрузок.
- [ ] Любые эвристики выбора источника (например, `choose_best_transcript`) считаются **policy** и должны быть:
  - задокументированы,
  - стабильны,
  - не маскировать отсутствие required входа.

**Evidence**:
- схема `VideoDocument`;
- пример `VideoDocument.json` (без PII) и его разбор;
- тест/скрипт, который валидирует вход.

### 1.3 No-fallback policy (fail-fast vs valid empty)

**Правило**: если extractor/профиль требует вход, а вход отсутствует/невалиден → **`raise` / status=error**, а не “молчаливый default”.

- [ ] Отсутствие обязательного input JSON / ошибка парсинга → fail-fast на уровне orchestrator/CLI.
- [ ] Отсутствие обязательного поля/источника текста для required extractor’а → fail-fast.
- [ ] Любая ошибка модели/инференса/валидации → `status="error"` + `error_code` + понятное сообщение (см. `ERROR_HANDLING_AND_EDGE_CASES.md`).
- [ ] **Observability (обязательное правило)**: любые ошибки и любые “empty” статусы **должны логгироваться** (без PII) и отражаться в `manifest.json` (component status/error_code/notes).

**Исключение**: “валидная пустота” (см. 1.6) — когда данных действительно нет и это нормальный кейс.

**Evidence**:
- фрагменты кода с проверками и явными ошибками;
- отсутствие “тихих” except/pass, которые превращают error в ok.

### 1.4 Per-run storage (ResultStore)

**TextProcessor должен писать результаты только в**:

`result_store/<platform_id>/<video_id>/<run_id>/`

Обязательные файлы:
- `manifest.json`
- `text_processor/text_features.npz` (канонический путь)

- [ ] Имя NPZ файла стабильное (fixed name). Перезапуск run не оставляет “старых” артефактов.
- [ ] Запись NPZ и manifest атомарная (tmp → `os.replace()`).
- [ ] Любые дополнительные файлы в `result_store` разрешены только если они:
  - являются **sub-artifacts** (например, `.npy` большой матрицы) и
  - явно перечислены в `manifest.json.artifacts[]` и/или в NPZ как ссылки,
  - не нарушают запрет на “произвольные JSON” (кроме `manifest.json`).

**TextProcessor-specific правило sub-artifacts (`.npy`) — обязательный стандарт**:
- [ ] Любые промежуточные/вспомогательные векторные артефакты `*.npy` (embeddings/aggregates/matrices) для TextProcessor должны сохраняться **per-run** внутри:
  - `result_store/<platform_id>/<video_id>/<run_id>/text_processor/_artifacts/*.npy`
- [ ] Эти `*.npy` файлы обязаны быть перечислены в `manifest.json.components[].artifacts[]` (type=`"npy"`).
- [ ] Extractor’ы, которые пишут `*.npy`, должны принимать `artifacts_dir` через параметры (config/UI) и **не** использовать глобальные директории (`TREND_TEXT_ARTIFACTS_DIR`) как source-of-truth.
- [ ] Запрещено полагаться на `glob + mtime` по глобальным директориям для выбора “последнего эмбеддинга” (недетерминировано, cross-run leakage).
- [ ] В `result`/NPZ нельзя возвращать **абсолютные** пути к `*.npy`; допускаются только:
  - числовые признаки (`features_flat`),
  - relpath внутри `text_processor/_artifacts/` (если это нужно для intra-run), либо хранение relpath только in-memory (см. 4.1).

**Критически важно (TextProcessor‑специфика)**:
- “кеши” и “артефакты” — разные сущности.
  - **Cache**: content-addressed, не обязателен для корректности, может жить вне `result_store` (например, `TREND_TEXT_CACHE_DIR`) и иметь TTL.
  - **Artifact**: часть результата run; должен быть внутри `run_rs_path` и отражён в `manifest.json`.

**Evidence**:
- фактический путь артефактов;
- содержимое `manifest.json` для компонента `text_processor`.

### 1.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="text_npz_v1"` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Обязательные ключи:
- `feature_names: object[str]`
- `feature_values: float32[]`
- `payload: object(dict)` — **privacy-safe summary** (без raw текста по умолчанию)
- `meta: object(dict)` — обязательные поля meta

Обязательные поля `meta` (минимум):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`, иначе null)
- `models_used[]`, `model_signature` (если используются модели)

**Extractor-level требования**:
- [ ] Каждая фича имеет стабильное имя (без случайных suffix/prefix).
- [ ] Единицы измерения и диапазоны (где применимо) зафиксированы в README extractor’а.
- [ ] Missing values: NaN + маски (`*_present`), а не “0 по умолчанию”, если 0 — валидное значение.

**Evidence**:
- пример NPZ `text_features.npz` + вывод валидатора `validate_npz(...)`.

### 1.6 Valid empty outputs (TextProcessor)

Валидная пустота означает: данные отсутствуют по причине отсутствия текста/комментов/политики приватности, но pipeline продолжает работать.

Стандартные значения `empty_reason` (канонические):
- `no_text_available`
- `comments_missing_or_disabled`
- `ocr_disabled_by_policy` (если в будущем будет OCR‑текст входом TextProcessor)
- `dependency_missing` (только если это реально “опциональная зависимость отключена профилем”, иначе это error)

Требования:
- [ ] При `status="empty"`: `empty_reason` обязательно.
- [ ] Фичи при empty должны быть:
  - либо NaN (и соответствующие `*_present=false`),
  - либо явно документированный “empty-safe” набор (например, длины=0), если это семантически корректно.
- [ ] Empty не должен скрывать ошибки парсинга/модели/валидации.
- [ ] **Observability**: empty-case должен быть явно зафиксирован (лог/manifest notes), чтобы downstream и UI могли объяснить пользователю “почему данных нет”.

**Evidence**:
- пример `VideoDocument` без текста → NPZ `status=empty`.

---

## 2. Model system: no-network + ModelManager (обязательное правило)

### 2.1 Запрещённые практики (fail audit)

В TextProcessor/Extractors запрещено:
- сетевые загрузки моделей/весов/данных во время run:
  - `from_pretrained(...)` без `local_files_only`,
  - `torch.hub.load(...)`,
  - `requests.get(...)` для весов,
  - любые “pretrained=True” с implicit download,
  - runtime `pip install` / `nltk.download(...)` и т.п.

### 2.2 Обязательное правило: `dp_models` (ModelManager)

- [ ] Любая ML‑модель (embedders, rerankers, tokenizers, topic models и т.п.) резолвится через `dp_models`.
- [ ] Спеки моделей заданы в `dp_models/spec_catalog/text/*.yaml` (или другом каноническом каталоге).
- [ ] Артефакты лежат в `DP_MODELS_ROOT` и валидируются fail-fast (missing weights → error).
- [ ] В `meta.models_used[]` фиксируются: `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`.

**Особое требование для Tokenizer**:
- [ ] Декодирование `transcripts_token_ids` использует только “shared tokenizer” из `dp_models` и не сохраняет raw transcript в артефакты по умолчанию.

**Evidence**:
- ссылка на код резолва через `dp_models`;
- пример `models_used[]` и `model_signature` в NPZ.

---

## 3. Privacy / Retention / PII (TextProcessor‑специфика)

Источник правил: `docs/contracts/PRIVACY_AND_RETENTION.md`.

### 3.1 Default: no raw

- [ ] По умолчанию TextProcessor **не сохраняет raw**:
  - title/description/transcript/comments,
  - OCR‑текст (если появится),
  - длинные списки фраз/топиков, которые могут реконструировать исходный текст.
- [ ] В `payload` хранится только privacy‑safe summary: длины, счётчики, хэши, timings, наборы ключей, но не содержимое.
- [ ] В логах запрещено PII: только длины/счётчики/хэши/статусы.

### 3.2 Debug-only raw режим (строгий gating)

Если нужен raw для локального дебага:
- [ ] Это включается **явным флагом** (например, `--store-raw-payload`).
- [ ] Raw сохраняется только в `_tmp_*` области и **не считается source-of-truth**.
- [ ] В production профилях флаг отключён; наличие raw в result_store — блокирующая проблема.

**Evidence**:
- grep‑проверка, что raw не попадает в `payload`/логи без флага;
- пример структуры `_tmp_*` и подтверждение, что она исключена из “source-of-truth”.

---

## 4. Зависимости между extractors и между процессорами

### 4.1 Внутри TextProcessor (extractor graph)

- [ ] Если extractor зависит от результатов предыдущих (например, cosine метрики зависят от эмбеддингов), зависимость:
  - описана в README,
  - отражена в “execution plan” (devices_config / порядок),
  - проверяется fail-fast, если required вход отсутствует.
- [ ] “Передача данных” между extractors делается через `current_doc` и/или через `payload.results_by_extractor`, без скрытых глобальных синглтонов.
- [ ] Для передачи **путей/идентификаторов sub-artifacts** между extractor’ами в рамках одного run допускается in-memory реестр в `VideoDocument` (например `doc.tp_artifacts`), при этом:
  - он не является persisted contract,
  - не должен попадать в `features_flat`/NPZ payload как raw пути,
  - нужен для детерминизма (без glob/mtime).

### 4.2 Между процессорами (cross-processor dependencies)

- [ ] Любые cross‑modal фичи (Text ↔ OCR / Visual / Audio) считаются зависимостями от других компонентов DataProcessor и должны:
  - опираться на NPZ source-of-truth других компонентов (а не на JSON),
  - иметь чётко описанный контракт входа/выхода,
  - иметь понятную empty/error семантику.

**Evidence**:
- список зависимостей и пример, как они резолвятся в run_rs_path.

---

## 5. Наблюдаемость: progress + stage timings

### 5.1 Промежуточный прогресс (обязательный критерий)

TextProcessor должен уметь репортить прогресс выполнения для UI/backend.

Требования:
- [ ] Есть stage-based прогресс минимум по стадиям:
  - `load_input`
  - `run_extractors`
  - `save_npz`
  - `validate_artifact`
  - `update_manifest`
- [ ] Во время `run_extractors` прогресс обновляется **по мере завершения экстракторов** (не по таймеру), минимум 10 обновлений на run (если экстракторов ≥10; иначе — по каждому).
- [ ] Формат прогресса **машиночитаем** и безопасен (без raw текста).

Рекомендованный формат события (stdout JSON‑line, проксируется оркестратором):
- `platform_id`, `video_id`, `run_id`
- `component="text_processor"`
- `stage_id`, `stage_name`
- `extractor` (если применимо)
- `progress_pct` ∈ [0..100]
- `ts`

**Evidence**:
- пример реального output (лог/JSON‑lines) с ≥10 обновлениями.

### 5.2 Stage timings (обязательный критерий)

- [ ] В NPZ (в `meta` или отдельной секции `summary`) сохранены тайминги стадий:
  - минимум: `load_input_ms`, `run_extractors_ms`, `save_npz_ms`, `validate_npz_ms`, `update_manifest_ms`.
- [ ] Также сохраняются per-extractor timings (агрегировано) — например `timings_by_extractor` в `payload` (privacy-safe).

**Evidence**:
- NPZ содержит `stage_timings_ms` (dict) + ссылка на код измерений.

---

## 6. Feature contract: управление выходными фичами (feature gating)

Цель: управляемо включать/выключать группы фичей (стоимость/качество/IO/NPZ size).

Требования:
- [ ] Есть механизм выбора фич через CLI/конфиг:
  - `--features <csv>` и/или `--feature-set <name>` и/или `--features-json <json>`.
- [ ] Есть “default feature set” и он документирован.
- [ ] В `meta` фиксируются:
  - `features_enabled[]`
  - `features_produced[]`
- [ ] Нет “скрытых” фич: если фича может появиться в output, она перечислена и gated.

**Evidence**:
- примеры запуска с разными feature sets;
- NPZ meta отражает включённые фичи.

---

## 7. Производительность и ресурсы (TextProcessor + Extractors)

### 7.1 Обязательные измерения

Для TextProcessor и для каждого “тяжёлого” extractor’а (embedding/topic/cosine/top‑k):
- [ ] latency per unit (unit = `VideoDocument`) и/или per sub-unit (например, per comment/per chunk) — явно задокументировано.
- [ ] CPU RSS peak.
- [ ] GPU VRAM peak (если используется GPU).
- [ ] Spikes (p95/p99) + правило детектора spikes.

### 7.2 Где хранить результаты измерений

Источник правды: `docs/models_docs/resource_costs/text_processor_costs_v1.json`  
Дополнительно допускается детализация по экстракторам: `text_processor_<extractor>_costs_v1.json`.

### 7.3 Требования к batching / OOM

- [ ] Для GPU‑extractors есть конфигурируемый `batch_size`.
- [ ] При OOM поддерживается управляемый rollback batch_size (см. `ERROR_HANDLING_AND_EDGE_CASES.md`), либо fail-fast с понятным error_code.

**Evidence**:
- JSON resource_costs;
- команды запуска бенчмарка и воспроизводимость результатов.

---

## 8. Проверка качества выхода (quality validation)

### 8.1 Минимальные sanity-checks (обязательны)

Для каждого extractor’а:
- [ ] диапазоны значений разумны (no NaN там, где обязаны быть числа; нет inf; нет отрицательных там, где нельзя);
- [ ] консистентность связных фичей (например, `*_present` ↔ значение);
- [ ] статистические инварианты (пример: cosine ∈ [-1, 1], длины ≥0).

### 8.2 Human-friendly визуализация / UI render (must-have для TP)

TextProcessor должен иметь функционал, который превращает NPZ output в **UI‑дружелюбный** формат (distribution, top features, flags, timings), без использования LLM как источника истины.

Требования:
- [ ] Есть deterministic “renderer” (python модуль/скрипт) который читает `text_features.npz` и строит render-context JSON (без raw текста).
- [ ] Этот render-context может быть использован LLM (см. `LLM_RENDERING.md`) только как вход для генерации текста.

**Evidence**:
- скрипт/модуль рендера + пример JSON;
- пример HTML/графиков (опционально), либо структуры данных для фронта.

---

## 9. Документация (обязательные разделы)

### 9.1 README TextProcessor

В `TextProcessor/README.md` должны быть разделы:
- **Input contract** (VideoDocument, required/optional sources)
- **Output contract** (NPZ schema, пути, meta)
- **Models** (через dp_models, no-network)
- **Parallelization** (внутренний/внешний параллелизм, thread-safety, GPU sharing)
- **Performance characteristics** (ссылка на resource_costs)
- **Quality validation & human-friendly inspection** (sanity + render)
- **Features** (feature gating, default set)

### 9.2 README каждого extractor’а

В `TextProcessor/src/extractors/<name>/README.md` обязаны быть:
- входы: какие поля `VideoDocument` нужны и какие зависимости от других extractor’ов
- выходы: список фичей + единицы/диапазоны + masks/empty semantics
- модели (если есть): spec name / runtime / engine / precision / device
- параметры: допустимые значения + дефолты + влияние на стоимость (Δ latency / Δ cost — best-effort)
- параллелизм/батчинг/лимиты и поведение при OOM
- качество: sanity checks + как визуально посмотреть результат

**Evidence**:
- ссылки на README, отсутствие дублирования “контента” в других местах.

---

## 10. Шаблон “карточки аудита” для extractor’а (заполняется на каждый)

Рекомендуемый файл:

`TextProcessor/docs/audits/<EXTRACTOR>_AUDIT.md` *(если `docs/` пустой — создать)*

Секции:
1. Summary (ok/needs fixes)
2. Contract compliance checklist (по разделам 1–9)
3. Models used (dp_models specs, weights_digest)
4. Features list + gating status
5. Performance (resource_costs ссылки)
6. Quality validation (sanity + render snapshots)
7. Open issues + fix plan (PR list)

---

## 11. Чек-лист аудитора (коротко)

### Архитектура / контракты
- [ ] per-run storage + manifest upsert
- [ ] NPZ meta обязательные поля + validate_npz
- [ ] no-fallback policy соблюдён
- [ ] empty semantics корректны (canonical empty_reason)

### Модели / приватность
- [ ] dp_models only, no downloads
- [ ] models_used/model_signature корректны
- [ ] no raw text in artifacts/logs by default

### Наблюдаемость / качество / ресурсы
- [ ] progress events есть и безопасны
- [ ] stage timings сохранены
- [ ] resource_costs измерены
- [ ] есть sanity checks + UI render


