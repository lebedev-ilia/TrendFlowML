# Audit: AudioProcessor Orchestrator/Writer (`run_cli.py`)

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

Orchestrator и writer приведены к production‑политике AudioProcessor:
- **Progress reporting**: JSON-lines в stdout с stage-based прогрессом (load_input, run_extractors, save_npz, validate_artifact, update_manifest, complete)
- **Stage timings**: тайминги стадий сохраняются в NPZ meta (`stage_timings_ms`, `timings_by_extractor`)
- **Resource metrics**: summary resource metrics в NPZ meta (`resource_metrics.cpu_rss_peak_mb`, `resource_metrics.gpu_vram_peak_mb`)
- **Error handling**: строгий fail-fast для extractors (если указан в `--extractors`, но не смог инициализироваться → error), опционально `--no-strict-extractors` для graceful degradation
- **Retry логика**: retry политики для Triton (503, 504, connection timeout) — 2 попытки с exponential backoff
- **OOM fallback**: автоматический fallback при OOM для CLAP (уменьшение batch_size, максимум 2 попытки)
- **UI Render**: генерация render-context JSON для каждого extractor'а в `_render/render_context.json`
- **Manifest integration**: render artifacts добавляются в `manifest.json.artifacts[]` (type=`"render"`)

---

## 2) Контракт входа

**`run_cli.py`** (CLI entrypoint):
- `--frames-dir`: **обязателен**. Путь к Segmenter output dir с `audio/audio.wav` и `audio/segments.json` (Segmenter contract)
- `--video-path`: опционален (metadata-only). AudioProcessor **не** извлекает аудио из видео
- `--rs-base`: base result_store path (required)
- `--run-rs-path`: явный путь к per-run result_store (optional)
- `--platform-id`, `--video-id`, `--run-id`: run identity (optional, auto-generated)
- `--sampling-policy-version`, `--dataprocessor-version`: версии (required)
- `--extractors`: comma-separated список extractors (default: `clap,tempo,loudness`)
- `--device`: устройство для обработки (`auto`/`cpu`/`cuda`, default: `auto`)
- `--segment-parallelism`, `--max-inflight`, `--clap-batch-size`: scheduler knobs
- `--asr-model-size`, `--diarization-model-size`, `--emotion-model-size`, `--source-separation-model-size`: размеры моделей
- `--speech-analysis-pitch`: включить pitch extraction
- `--no-strict-extractors`: graceful degradation вместо fail-fast (для дебага)
- `--write-legacy-manifest`: писать legacy manifest (deprecated)

---

## 3) Контракт выхода

**NPZ артефакты**: отдельные NPZ для каждого extractor'а
- `result_store/.../<component_name>/<component_name>_features.npz`
- Схема: `schema_version="audio_npz_v1"`
- Обязательные поля `meta`:
  - `stage_timings_ms`: тайминги стадий обработки
  - `timings_by_extractor`: тайминги по extractor'ам
  - `resource_metrics`: summary resource metrics (CPU RSS peak, GPU VRAM peak)
  - `models_used[]`, `model_signature`: модели через dp_models
  - `scheduler_knobs`: применённые knobs (segment_parallelism, max_inflight, clap_batch_size)

**Manifest**: `manifest.json` (upsert через `RunManifest.upsert_component()`)
- `status`: `ok` | `empty` | `error`
- `empty_reason`: причина пустоты (если `status=empty`)
- `error`: сообщение об ошибке (если `status=error`)
- `artifacts[]`: список артефактов (NPZ + render-context JSON)

**Render-context JSON**: `_render/render_context.json` (для каждого extractor'а)
- Timeline данные (сегменты с временными метками и значениями)
- Статистики (mean, std, min, max, distributions)
- Quality flags (warnings, confidence scores)

**Scheduler runtime report**: `_reports/scheduler_runtime_report.json`
- Детальные resource metrics (CPU RSS peak, GPU VRAM peak)
- Per-extractor timings и статусы
- Applied scheduler knobs

---

## 4) Progress Reporting

**Stage-based прогресс** (JSON-lines в stdout):
- `load_input` (5%): загрузка audio/segments.json
- `run_extractors` (10-80%): запуск extractors, обновляется по мере завершения каждого extractor'а
- `save_npz` (80%): сохранение NPZ артефактов
- `validate_artifact` (85%, per component): валидация NPZ
- `update_manifest` (95%, per component): обновление manifest.json
- `complete` (100%): завершение

**Формат события**:
```json
{
  "platform_id": "...",
  "video_id": "...",
  "run_id": "...",
  "component": "...",
  "stage_id": "...",
  "stage_name": "...",
  "progress_pct": N,
  "extractor": "..." (optional),
  "ts": "..."
}
```

---

## 5) Stage Timings

**Измеряемые стадии**:
- `load_input_ms`: загрузка audio/segments.json
- `run_extractors_ms`: общее время выполнения extractors
- `save_npz_ms`: сохранение всех NPZ артефактов
- `validate_npz_ms`: валидация NPZ (суммируется по всем компонентам)
- `update_manifest_ms`: обновление manifest.json (суммируется по всем компонентам)

**Per-extractor timings**:
- `timings_by_extractor[extractor_key]`: `wall_ms`, `reported_ms`

**Сохранение**: в NPZ meta (`stage_timings_ms`, `timings_by_extractor`)

---

## 6) Resource Metrics

**Summary metrics** (в NPZ meta):
- `resource_metrics.cpu_rss_peak_mb`: пиковое использование CPU RSS (MB)
- `resource_metrics.gpu_vram_peak_mb`: пиковое использование GPU VRAM (MB)

**Детальные metrics** (в `scheduler_runtime_report.json`):
- `rss_peak_mb`, `gpu_used_peak_mb`
- `per_extractor`: детальные тайминги и статусы по каждому extractor'у

**Мониторинг**: continuous sampling через background thread (каждые 0.2s)

---

## 7) Error Handling

**Строгий fail-fast** (по умолчанию):
- Если extractor указан в `--extractors`, но не смог инициализироваться → `RuntimeError` (fail-fast)
- Отсутствие обязательного family в `segments.json` → `RuntimeError` (no-fallback)

**Graceful degradation** (с `--no-strict-extractors`):
- Если extractor не смог инициализироваться → warning, skip extractor, continue run
- Полезно для дебага, но не рекомендуется в production

**Retry логика для Triton**:
- Triton-backed extractors (diarization, emotion, source separation)
- Retry на transient errors: 503, 504, connection timeout, "triton" в error message
- 2 попытки с exponential backoff (1.0s, 2.0s)

**OOM fallback для CLAP**:
- Автоматическое уменьшение `batch_size` при OOM
- Максимум 2 попытки: `batch_size → batch_size // 2 → 1`
- Логирование каждого fallback

---

## 8) UI Render

**Генерация render-context**:
- Модуль: `src/core/renderer.py`
- Функция: `render_component(npz_path, component_name, output_dir)`
- Результат: `_render/render_context.json` в директории компонента

**Поддерживаемые extractors**:
- `clap_extractor`: timeline с embedding norms, статистики
- `tempo_extractor`: timeline с BPM, распределения, warnings
- `loudness_extractor`: timeline с RMS/dBFS/LUFS, распределения
- `asr_extractor`: timeline с token counts (privacy-safe, без raw текста)
- `speaker_diarization_extractor`: timeline с speaker segments, статистики по спикерам

**Интеграция с manifest**:
- Render artifacts добавляются в `manifest.json.artifacts[]` (type=`"render"`)

---

## 9) Models System

**Все модели через dp_models**:
- CLAP: `laion_clap` (inprocess, torch, fp32)
- Whisper (ASR): `whisper_{size}_inprocess` (inprocess, torch, fp16/fp32)
- Speaker Diarization: `speaker_diarization_{size}_triton` (triton, onnx/tensorrt, fp16/fp32)
- Emotion Diarization: `emotion_diarization_{size}_triton` (triton, onnx/tensorrt, fp16/fp32)
- Source Separation: `source_separation_{size}_triton` (triton, onnx/tensorrt, fp16/fp32)

**Метаданные моделей**:
- Сохраняются в NPZ meta: `models_used[]` (model_name, model_version, weights_digest, runtime, engine, precision, device)
- `model_signature`: агрегированный signature всех моделей

---

## 10) Segmenter Contract

**Входные данные**:
- `audio/audio.wav`: готовит Segmenter
- `audio/segments.json`: contract `audio_segments_v1` с families (primary, clap, tempo, asr, diarization, emotion, source_separation)

**No-fallback policy**:
- Отсутствие `audio/audio.wav` или `audio/segments.json` → `RuntimeError`
- Отсутствие обязательного family для required extractor'а → `RuntimeError`

**Legacy mode**:
- Удалён. AudioProcessor работает только по Segmenter contract (`--frames-dir` обязателен).

---

## 11) Atomic Writes

**NPZ**: `_atomic_save_npz()` использует `tempfile.mkstemp()` → `np.savez_compressed()` → `os.replace()` (атомарно)

**Manifest**: `RunManifest.flush()` использует `_atomic_write_json()` (tmp → replace) из `VisualProcessor/utils/manifest.py`

**Render-context**: `render_component()` использует tmp → replace для атомарной записи

---

## 12) Открытые задачи для будущих улучшений

1. **Feature gating**: поддержка feature sets (baseline/standard/full) для выбора подмножества фичей внутри extractor'а
2. **Рефакторинг архитектуры**: вынести orchestration в отдельный класс `AudioProcessorOrchestrator` (аналог TextProcessor MainProcessor)
3. **Кэширование моделей**: убедиться, что ModelManager правильно кэширует модели между extractors
4. **Дополнительные агрегаты**: добавить временные агрегаты (mean, std, min, max) и quality flags для каждого extractor'а

---

## 13) Compliance Checklist

### Архитектура / контракты
- [x] per-run storage + manifest upsert
- [x] NPZ meta обязательные поля + validate_npz
- [x] no-fallback policy соблюдён
- [x] empty semantics корректны (canonical empty_reason)
- [x] Segmenter contract соблюдён

### Модели / приватность
- [x] dp_models only, no downloads
- [x] models_used/model_signature корректны
- [x] no raw audio in artifacts/logs by default

### Наблюдаемость / качество / ресурсы
- [x] progress events есть и безопасны
- [x] stage timings сохранены
- [x] resource_metrics сохранены (summary в NPZ, details в report)
- [x] есть sanity checks + UI render

### Error handling
- [x] строгий fail-fast для extractors (по умолчанию)
- [x] retry логика для Triton
- [x] OOM fallback для CLAP

