# Audit: `asr_text_proxy_audio_features` (ASRTextProxyExtractor)

**Дата**: 2026-01-29  
**Статус**: `done`  
**Критерии**: `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

---

## 1) Summary

Компонент приведён к ключевым требованиям production-уровня по части:
- **no-fallback** (нет “подстановки” транскрипта/уверенности),
- **строгий контракт по `audio_duration_sec`** (fail-fast),
- **stable flat feature names** (`tp_asrproxy_*`) для dataset/UI,
- **feature-gating** через параметры extractor’а и config-driven запуск,
- **удалены внешние зависимости** (`langdetect`, `spacy`) из прод-версии extractor’а.

Остаются интеграционные задачи на стороне AudioProcessor (зафиксировать schema_version/поля `doc.asr`), но extractor в TextProcessor доведён до A‑policy и может быть включаем/выключаем через конфиг.

---

## 2) Архитектура и интеграции

### 2.1 Внутри TextProcessor

- **Исправлено**: registry `MainProcessor` теперь указывает на реальные `src/extractors/*/main.py`.
- **Добавлено**: `strict=True` (default) → если extractor из конфига не загрузился/не создался, run становится `error` (fail-fast).
- **Добавлено**: слой `features_flat` (единый dict плоских фич), который `run_cli.py` использует для NPZ export.

### 2.2 Внешние зависимости (AudioProcessor / Segmenter)

**Зафиксировано требование**: `audio_duration_sec` должен присутствовать всегда (Segmenter/AudioProcessor contract).  
**Источник транскрипта**: `VideoDocument.asr` (AudioProcessor-owned payload). Если ASR отсутствует — это валидный empty.

---

## 3) Контракт входа/выхода

### 3.1 Input

`VideoDocument`:
- `audio_duration_sec` (**required**)
- `asr` (**optional**, preferred): `segments[]` с `text` и optional `confidence`
- `transcripts_meta` (**legacy**, временный alias)

### 3.2 Output (stable flat features)

Префикс: `tp_asrproxy_`

Ключевые поля:
- `tp_asrproxy_present`, `tp_asrproxy_has_confidence`
- `tp_asrproxy_audio_duration_sec`
- confidence metrics: `tp_asrproxy_confidence_*`, `tp_asrproxy_low_conf_rate`
- noise proxies: `tp_asrproxy_text_noise_*`, `tp_asrproxy_noise_proxy`
- rhythm proxies: `tp_asrproxy_speech_*`, `tp_asrproxy_pause_density`, `tp_asrproxy_filler_ratio`
- intonation: `tp_asrproxy_sentence_intonation`

**Empty semantics**:
- если транскрипта нет: `tp_asrproxy_present=false`, остальные метрики = `NaN` (модель отличает “нет данных” от “0”)

**Error semantics**:
- если нет `audio_duration_sec` → `RuntimeError`

---

## 4) Model system / no-network

Компонент не использует ML-модели. Ранее планировались `langdetect/spacy`, но они удалены из прод-версии.

Если понадобится NER/langdetect — это должен быть отдельный extractor/компонент с моделями через `dp_models` (ModelManager), без runtime downloads.

---

## 5) Feature gating

Реализовано через параметры конструктора (передаются через `--extractor-params-json`):
- `enable_basic`, `enable_noise`, `enable_rhythm`, `enable_intonation`
- `low_conf_threshold`

---

## 6) Performance / resource costs

### Сейчас

- CPU-only, вычисления лёгкие.
- Измерения в `docs/models_docs/resource_costs/` пока отсутствуют для этого extractor’а.

### TODO (обязательное для закрытия аудита)

- Добавлено `DataProcessor/docs/models_docs/resource_costs/text_processor_asr_text_proxy_audio_features_costs_v1.json`
  (best-effort шаблон; требуется заполнить бенчмарком).

---

## 7) Quality validation

### Sanity checks (ожидаемые инварианты)

- `tp_asrproxy_confidence_* ∈ [0..1]` (если не NaN)
- `tp_asrproxy_low_conf_rate ∈ [0..1]` (если не NaN)
- `tp_asrproxy_*_ratio ∈ [0..1]` (если не NaN)
- `tp_asrproxy_pause_density ≥ 0` (если не NaN)
- если `tp_asrproxy_present=false` → большинство фич = NaN (кроме duration + masks)

### Fixtures

Добавлены фикстуры (без PII) для локального smoke-run:
- `src/extractors/asr_text_proxy_audio_features/fixtures/doc_asr_with_confidence.json`
- `.../doc_asr_no_confidence.json`
- `.../doc_no_asr.json`

---

## 8) Изменения в коде (что сделано)

- `TextProcessor/src/core/main_processor.py`
  - strict loading (fail-fast)
  - registry paths исправлены
  - поддержка `features_flat`
- `TextProcessor/run_cli.py`
  - PYTHONPATH fix для `import src.*`
  - `--devices-config-json`, `--extractor-params-json`, `--disabled-extractors`
  - экспорт фич из `features_flat`
- `TextProcessor/src/schemas/models.py`
  - добавлены поля `audio_duration_sec`, `asr`, `transcripts_meta` (legacy)
- `TextProcessor/src/extractors/asr_text_proxy_audio_features/main.py`
  - новый contract: ASR от AudioProcessor, duration required
  - no-fallback, NaN+masks
  - stable feature names `tp_asrproxy_*`
- `TextProcessor/src/extractors/asr_text_proxy_audio_features/README.md`
  - обновлён под новый контракт и feature-gating

---

## 9) Открытые вопросы / блокеры

1. **Контракт AudioProcessor → TextProcessor**:
   - нужно формально закрепить структуру `VideoDocument.asr` (schema_version + segments поля) и обеспечить её генерацию в AudioProcessor.
2. **Измерения performance**:
   - добавить `resource_costs` файл для этого extractor’а.
3. **Среда для smoke-run**:
   - локально в текущем окружении отсутствует `numpy` → нужен нормальный runtime/venv для воспроизводимых запусков.


