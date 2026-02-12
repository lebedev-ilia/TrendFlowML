# Audit: TextProcessor Orchestrator/Writer (`run_cli.py` + `main_processor.py`)

**Дата**: 2026-01-29  
**Статус**: `done`  
**Критерии**: `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

---

## 1) Summary

Orchestrator и writer приведены к production‑политике TextProcessor:
- **Error handling per extractor**: каждый extractor обёрнут в try/except, ошибки собираются в `errors_by_extractor`, required extractors fail-fast
- **Status aggregation**: статусы extractors (`ok`/`empty`/`error`) собираются и агрегируются в orchestrator‑уровне
- **Features_flat conflict detection**: обнаружение дубликатов ключей от разных extractors с логированием
- **Models_used collection**: автоматический сбор `models_used` из всех extractors (вместо хардкода)
- **Required vs optional extractors**: поддержка `required_extractors` списка для fail-fast политики
- **NPZ validation before write**: валидация NPZ в памяти перед атомарной записью (не пишем невалидные артефакты)
- **Structured logging**: логирование ошибок/empty/конфликтов без PII в `_logs/` или stderr
- **Orchestrator metrics**: метрики `tp_orchestrator_*` в `features_flat` (total_extractors, successful_count, failed_count, empty_count, total_duration_ms, feature_conflicts_count)
- **Primary embedding flag**: `--include-primary-embedding` / `--no-include-primary-embedding` для контроля включения primary_embedding в NPZ

---

## 2) Контракт входа

**`run_cli.py`** (CLI entrypoint):
- `--input-json`: путь к `VideoDocument` JSON (required)
- `--run-rs-path`: явный путь к per-run result_store (optional, иначе строится из `--rs-base`/`--platform-id`/`--video-id`/`--run-id`)
- `--devices-config-json`: JSON mapping device → extractor names (optional)
- `--extractor-params-json`: JSON mapping extractor name → params dict (optional)
- `--disabled-extractors`: comma-separated список extractor names для отключения (optional)
- `--enable-embeddings`: включить GPU embedders (optional)
- `--no-strict-extractors`: не fail-fast при ошибках создания extractors (NOT recommended)
- `--include-primary-embedding` / `--no-include-primary-embedding`: контроль включения primary_embedding в NPZ (default: True)
- `--log-dir`: директория для structured logs (optional, default: `<run_rs_path>/_logs/`)

**`MainProcessor`** (orchestrator):
- `devices_config`: Dict[str, Union[str, List[str]]] — mapping device → extractor names
- `extractor_params`: Dict[str, Dict[str, Any]] — параметры per extractor
- `strict`: bool — fail-fast при ошибках создания extractors (default: True)
- `artifacts_dir`: str — путь к per-run sub-artifacts directory (optional)
- `required_extractors`: List[str] — список extractor names, которые обязательны для успешного run (optional)
- `logger`: logging.Logger — structured logger (optional, default: root logger)

---

## 3) Контракт выхода

**NPZ артефакт**: `text_processor/text_features.npz`
- `feature_names: object[str]` — отсортированные ключи из `features_flat`
- `feature_values: float32[]` — значения (соответствуют `feature_names`)
- `payload: object(dict)` — privacy-safe summary (без raw текста)
- `primary_embedding: float32[D]` — primary embedding vector (если `--include-primary-embedding` и embeddings enabled)
- `primary_embedding_present: bool`
- `primary_embedding_source: str`
- `primary_embedding_model: str`
- `meta: object(dict)` — run identity, версии, статус, `models_used[]`

**Manifest**: `manifest.json` (upsert через `RunManifest.upsert_component()`)
- `status`: `ok` | `empty` | `error`
- `empty_reason`: причина пустоты (если `status=empty`)
- `error`: сообщение об ошибке (если `status=error`)
- `error_code`: код ошибки (если `status=error`)
- `artifacts[]`: список артефактов (NPZ + sub-artifacts `*.npy`)

**Structured logs**: `_logs/text_processor.log` (если `--log-dir` задан или stderr не TTY)
- Логирование ошибок extractors (без PII)
- Логирование empty причин
- Логирование конфликтов features_flat
- Логирование валидации NPZ

---

## 4) Error handling и status propagation

**Per-extractor error handling**:
- Каждый `ext.extract()` обёрнут в try/except
- Ошибки собираются в `errors_by_extractor[extractor_name] = error_message`
- Если required extractor падает → fail-fast (RuntimeError)
- Если optional extractor падает → логируется warning, run продолжается

**Status aggregation**:
- `MainProcessor.run()` собирает `status_by_extractor` из каждого extractor result
- Агрегирует orchestrator‑уровень статус:
  - Если все extractors `empty` → `status="empty"`, `empty_reason="all_extractors_empty"`
  - Если required extractor `error` → `status="error"` (уже fail-fast выше)
  - Если только optional extractors failed и нет successful → `status="error"`
  - Иначе → `status="ok"`

**NPZ validation**:
- NPZ валидируется **перед** атомарной записью (временный файл → validate → atomic move)
- Если валидация не прошла → не пишем файл, `status="error"`, логируем ошибки

---

## 5) Features_flat merge и конфликты

**Merge policy**: last-wins (как было)
**Conflict detection**: добавлена проверка дубликатов ключей
- При обнаружении конфликта логируется warning: `"features_flat conflict: {key} (from {extractor}, previously from another extractor)"`
- Конфликты собираются в `features_flat_conflicts` список
- Метрика `tp_orchestrator_feature_conflicts_count` добавляется в `features_flat`

---

## 6) Models_used collection

**Автоматический сбор**:
- `MainProcessor.run()` собирает `models_used` из `part["result"].get("meta", {}).get("models_used", [])` каждого extractor
- Мержит в `features["models_used"]` (список всех уникальных моделей)
- `run_cli.py` использует это для `meta.models_used` в NPZ (вместо хардкода)

---

## 7) Required vs optional extractors

**Политика**:
- `required_extractors: List[str]` — список extractor names, которые обязательны
- Если required extractor не создался → fail-fast (RuntimeError)
- Если required extractor вернул `status="error"` → fail-fast (RuntimeError)
- Если required extractor не выполнился → fail-fast (RuntimeError в конце `run()`)

---

## 8) Orchestrator metrics

**Метрики в `features_flat`**:
- `tp_orchestrator_total_extractors`: общее количество extractors
- `tp_orchestrator_successful_count`: количество успешных (`status="ok"`)
- `tp_orchestrator_failed_count`: количество failed (`status="error"`)
- `tp_orchestrator_empty_count`: количество empty (`status="empty"`)
- `tp_orchestrator_total_duration_ms`: общее время выполнения (ms)
- `tp_orchestrator_feature_conflicts_count`: количество конфликтов в features_flat

---

## 9) Structured logging

**Логирование**:
- Если `--log-dir` задан или stderr не TTY → логи в файл `_logs/text_processor.log` + stderr
- Иначе → только stderr
- Формат: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Логируются: ошибки extractors, empty причины, конфликты features_flat, валидация NPZ

---

## 10) Primary embedding flag

**Контроль включения**:
- `--include-primary-embedding` (default: True) — включить primary_embedding в NPZ
- `--no-include-primary-embedding` — отключить (экономия места, если не нужно)
- Работает только если `--enable-embeddings` и найден `EmbeddingSourceIdExtractor`

---

## 11) Atomic writes

**NPZ**: `_atomic_save_npz()` использует `tempfile.mkstemp()` → `np.savez_compressed()` → `os.replace()` (атомарно)

**Manifest**: `RunManifest.flush()` использует `_atomic_write_json()` (tmp → replace) из `VisualProcessor/utils/manifest.py`

**Raw payload** (debug): `--store-raw-payload` использует tmp → replace

---

## 12) Registry validation

**Хардкод registry** (явность лучше auto-discovery):
- `_get_registry_entry()` возвращает `(module_paths, class_name, rel_file_path)` для каждого extractor
- При старте можно добавить валидацию импорта всех зарегистрированных extractors (future enhancement)

---

## 13) Открытые задачи для будущих улучшений

1. Добавить валидацию импорта всех зарегистрированных extractors при старте `MainProcessor`
2. Добавить метрики per-extractor в `features_flat` (например, `tp_orchestrator_extractor_{name}_status`)
3. Добавить retry политику для transient errors (если понадобится)
4. Добавить timeout per extractor (если понадобится)

---

## 14) Compliance с TP_AUDIT_CRITERIA

- ✅ **1.1 Интерфейсы**: CLI entrypoint, per-run result_store, manifest upsert, строгая загрузка extractors
- ✅ **1.2 Контракты входа**: `VideoDocument` JSON, явные источники текста
- ✅ **1.3 No-fallback policy**: fail-fast для required extractors, явные ошибки
- ✅ **1.4 Per-run storage**: все артефакты в `run_rs_path`, sub-artifacts в `_artifacts/`, manifest upsert
- ✅ **1.5 NPZ schema**: `text_npz_v1`, обязательные ключи, meta с `models_used[]`
- ✅ **1.6 Valid empty**: `status="empty"` с `empty_reason`, NaN для missing features
- ✅ **Observability**: structured logging, manifest notes, error codes

---

**Evidence**:
- `TextProcessor/run_cli.py` (lines 198-560)
- `TextProcessor/src/core/main_processor.py` (lines 16-313)
- Примеры NPZ: `result_store/*/text_processor/text_features.npz`
- Примеры manifest: `result_store/*/manifest.json`

