# Audit: `voice_quality_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`voice_quality_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: поддерживает `run_segments()` для работы на сегментах от Segmenter (family `voice_quality`)
- ✅ **No-fallback policy**: fail-fast при ошибках оценки f0 и вычисления метрик (no-fallback для всех операций)
- ✅ **Model system**: не использует ML модели через ModelManager (signal processing: librosa, numpy)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `voice_quality_extractor_features.npz`, .npy файлы в per-run storage
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (5 групп, все opt-in)
- ✅ **Error handling**: детальные error codes (6 типов)
- ✅ **Segments validation**: полная валидация структуры сегментов (для `run_segments()`)
- ✅ **Output validation**: полная валидация выходных данных (диапазоны, NaN/inf, консистентность)
- ✅ **Parameter validation**: полная валидация входных параметров (fail-fast)
- ✅ **Progress reporting**: обновление прогресса для каждого этапа и сегмента
- ✅ **UI Render**: renderer реализован в `src/extractors/voice_quality_extractor/render.py` + HTML renderer для дебага
- ✅ **Contract versioning**: `voice_quality_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional metrics**: jitter_mean/std/min/max, shimmer_mean/std/min/max, hnr_mean/std/min/max, f0_stability, voice_presence_ratio, voice_quality_score, breathiness_score
- ✅ **Optional audio normalization**: флаг `--voice-quality-enable-audio-normalization`
- ✅ **F0 method selection**: поддержка YIN, PYIN, torchcrepe с явным выбором метода
- ✅ **Optional integration with pitch_extractor**: использование результатов `pitch_extractor` для более точных оценок f0

---

## 2) Contract Compliance Checklist

### 2.1 Архитектурное соответствие

#### 1.1 Интерфейсы и границы ответственности

- ✅ Реализует `BaseExtractor.run()` и `BaseExtractor.run_segments()`
- ✅ Не делает скрытых глобальных сайд-эффектов
- ✅ Требования к входу декларированы и проверяются fail-fast

**Evidence**:
- `src/extractors/voice_quality_extractor/main.py`: класс `VoiceQualityExtractor` наследует `BaseExtractor`
- Методы `run()` и `run_segments()` реализованы
- Валидация входных параметров в `_validate_parameters()`

#### 1.2 Контракты входа (Segmenter contract)

- ✅ Использует `audio/audio.wav` и `audio/segments.json` (family `voice_quality`)
- ✅ Для `run_segments()` читает `families.voice_quality.segments[]`
- ✅ Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- ✅ Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `run_segments()` принимает `segments: List[Dict[str, Any]]` из Segmenter
- Использует `audio_utils.load_audio_segment()` с `start_sample` и `end_sample`
- В `run_cli.py`: проверка наличия `families.voice_quality.segments[]` (fail-fast)

#### 1.3 No-fallback policy

- ✅ Отсутствие обязательного `audio/audio.wav` → fail-fast на уровне CLI
- ✅ Отсутствие обязательного `audio/segments.json` → fail-fast на уровне CLI
- ✅ Отсутствие обязательного family `voice_quality` → fail-fast
- ✅ Пустой список segments → fail-fast (`ValueError("segments is empty (no-fallback)")`)
- ✅ Ошибка оценки f0 → fail-fast (`RuntimeError` с error_code)
- ✅ Ошибка вычисления метрик → fail-fast (`RuntimeError` с error_code)

**Evidence**:
- `_estimate_f0()`: raise `RuntimeError` при ошибке оценки f0 (no-fallback)
- `run_segments()`: проверка `if not segments: raise ValueError("segments is empty (no-fallback)")`
- В `run_cli.py`: проверка наличия `families.voice_quality.segments[]` (fail-fast)

#### 1.4 Per-run storage

- ✅ Имя NPZ файла стабильное: `voice_quality_extractor_features.npz`
- ✅ Запись NPZ атомарная (tmp → `os.replace()`)
- ✅ Sub-artifacts (`.npy`) сохраняются в `result_store/<platform_id>/<video_id>/<run_id>/voice_quality_extractor/_artifacts/*.npy`
- ✅ `.npy` файлы регистрируются в `manifest.json.artifacts[]`

**Evidence**:
- `_save_time_series_npy()`: сохранение больших массивов (>10000 элементов) в `.npy`
- В `run_cli.py`: `extractor.artifacts_dir = str(Path(run_rs_path) / "voice_quality_extractor" / "_artifacts")`
- Регистрация `.npy` файлов в `manifest.json` через `_atomic_save_npz()`

#### 1.5 NPZ schema + meta contract

- ✅ Schema version: `audio_npz_v1`
- ✅ Обязательные ключи: `feature_names`, `feature_values`, `payload`, `meta`
- ✅ Обязательные поля meta: `producer`, `producer_version`, `schema_version`, `created_at`, `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`, `status`, `empty_reason`, `device_used`, `voice_quality_contract_version`, `features_enabled`

**Evidence**:
- В `run_cli.py`: `_save_component_npz()` для `voice_quality_extractor` с правильными полями meta
- `payload` содержит `voice_quality_contract_version` и `_features_enabled`

#### 1.6 Valid empty outputs

- ✅ При `status="empty"`: `empty_reason` обязательно
- ✅ Фичи при empty: NaN или явно документированный "empty-safe" набор
- ✅ Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- В `run_cli.py`: обработка empty статуса с `empty_reason`
- Empty-case логируется и отражается в `manifest.json`

### 2.2 Model system

- ✅ Не использует ML модели через ModelManager (signal processing: librosa, numpy)
- ✅ Нет сетевых загрузок моделей/весов/данных
- ✅ Использует только локальные библиотеки (librosa, numpy, опционально torchcrepe)

**Evidence**:
- `_estimate_f0()`: использует `librosa.yin()`, `librosa.pyin()`, или `torchcrepe.predict()`
- Нет вызовов `dp_models` или `ModelManager`
- Нет сетевых загрузок

### 2.3 Segmenter contract

- ✅ Использует `audio/segments.json` (contract `audio_segments_v1`)
- ✅ Читает `families.voice_quality.segments[]`
- ✅ Использует `start_sample/end_sample` для загрузки сегментов
- ✅ Не генерирует сегменты сам

**Evidence**:
- `run_segments()` принимает `segments: List[Dict[str, Any]]` из Segmenter
- Использует `audio_utils.load_audio_segment()` с `start_sample` и `end_sample`
- В `run_cli.py`: проверка наличия `families.voice_quality.segments[]`

### 2.4 Зависимости между extractors

- ✅ Опциональная интеграция с `pitch_extractor`: если доступны результаты `pitch_extractor`, используются их f0 значения
- ✅ Зависимость документирована в README (раздел "Feature Dependencies")
- ✅ Проверка наличия `pitch_payload` в `_estimate_f0()`

**Evidence**:
- `_estimate_f0()`: проверка `if self.pitch_payload is not None:`
- В `run_cli.py`: передача `pitch_payload` из результатов `pitch_extractor`
- README содержит раздел "Feature Dependencies"

### 2.5 Наблюдаемость

#### 5.1 Промежуточный прогресс

- ✅ Stage-based прогресс: `load_input`, `run_extractors`, `save_npz`, `validate_artifact`, `update_manifest`
- ✅ Прогресс обновляется по мере завершения сегментов (каждые 10% при `run_segments()`)
- ✅ Формат прогресса машиночитаем (JSON-line)

**Evidence**:
- `run_segments()`: `progress_callback` вызывается каждые 10% сегментов
- В `run_cli.py`: `voice_quality_progress_callback` для эмиссии прогресса

#### 5.2 Stage timings

- ✅ Тайминги стадий сохранены в NPZ meta: `load_input_ms`, `run_extractors_ms`, `save_npz_ms`, `validate_npz_ms`, `update_manifest_ms`
- ✅ Per-extractor timings сохранены в `meta.timings_by_extractor`

**Evidence**:
- В `run_cli.py`: `stage_timings` и `timings_by_extractor` сохраняются в NPZ meta

### 2.6 Feature contract

- ✅ Механизм выбора фич через CLI: `--voice-quality-enable-jitter`, `--voice-quality-enable-shimmer`, `--voice-quality-enable-hnr`, `--voice-quality-enable-f0-stats`, `--voice-quality-enable-time-series`
- ✅ Все фичи opt-in (по умолчанию все выключены)
- ✅ В `meta` фиксируются: `features_enabled[]`, `features_produced[]`
- ✅ Нет "скрытых" фич: все фичи перечислены и gated

**Evidence**:
- 5 персональных флагов для каждой группы фичей
- `payload` содержит `_features_enabled` список
- В `run_cli.py`: feature-gated сохранение в NPZ

### 2.7 Производительность и ресурсы

- ✅ Latency per unit задокументирована: ~1.5 секунды для полного аудио, ~0.1-0.2 секунды на сегмент
- ✅ CPU RSS peak: низкие-умеренные (YIN/PYIN, оконные операции, автокорреляция)
- ✅ GPU VRAM peak: не используется (CPU-only)
- ✅ Segment-level parallelism поддерживается через `segment_parallelism` и `max_inflight`

**Evidence**:
- README содержит раздел "Performance characteristics"
- `run_segments()` поддерживает параллельную обработку через `ThreadPoolExecutor`

### 2.8 Проверка качества выхода

#### 8.1 Sanity-checks

- ✅ Диапазоны значений разумны: jitter, shimmer ∈ [0.0, 1.0], HNR ∈ [-100.0, 100.0]
- ✅ Консистентность связных фичей: f0_min ≤ f0_mean ≤ f0_max
- ✅ Статистические инварианты: quality scores ∈ [0.0, 1.0], voice_presence_ratio ∈ [0.0, 1.0]

**Evidence**:
- `_validate_output()`: проверка диапазонов, NaN/inf, консистентности
- Валидация выполняется перед сохранением NPZ

#### 8.2 Human-friendly визуализация

- ✅ Deterministic renderer: `render_voice_quality_extractor()` в `src/extractors/voice_quality_extractor/render.py`
- ✅ HTML renderer для дебага: `render_voice_quality_extractor_html()`
- ✅ README содержит раздел "Visualization" с рекомендациями по визуализации

**Evidence**:
- `src/extractors/voice_quality_extractor/render.py`: реализованы оба renderer'а
- README содержит раздел "Visualization" с описанием типов графиков и интерактивных элементов

### 2.9 Документация

- ✅ README содержит все обязательные разделы:
  - Input contract (Segmenter contract)
  - Output contract (NPZ schema, пути, meta)
  - Models (не используется, но документировано)
  - Feature dependencies (явное описание зависимостей)
  - Feature gating (описание всех флагов)
  - Параметры (допустимые значения + дефолты)
  - Параллелизм/батчинг/лимиты
  - Качество (sanity checks + визуализация)
  - Visualization (рекомендации по визуализации для UI/сайта)

**Evidence**:
- `src/extractors/voice_quality_extractor/README.md`: содержит все обязательные разделы

---

## 3) Models used

- **Нет ML моделей**: экстрактор использует только signal processing библиотеки (librosa, numpy, опционально torchcrepe)
- **librosa**: для оценки f0 (YIN, PYIN)
- **torchcrepe** (опционально): для точной оценки f0
- **numpy**: для численных операций

---

## 4) Features list + gating status

### 4.1 Feature groups

1. **Jitter метрики** (`--voice-quality-enable-jitter`):
   - `vq_jitter`, `vq_jitter_mean`, `vq_jitter_std`, `vq_jitter_min`, `vq_jitter_max`

2. **Shimmer метрики** (`--voice-quality-enable-shimmer`):
   - `vq_shimmer`, `vq_shimmer_mean`, `vq_shimmer_std`, `vq_shimmer_min`, `vq_shimmer_max`

3. **HNR метрики** (`--voice-quality-enable-hnr`):
   - `vq_hnr_like_db`, `vq_hnr_mean`, `vq_hnr_std`, `vq_hnr_min`, `vq_hnr_max`

4. **F0 статистики** (`--voice-quality-enable-f0-stats`):
   - `vq_f0_mean`, `vq_f0_std`, `vq_f0_min`, `vq_f0_max`, `vq_f0_median`, `vq_f0_stability`, `vq_voice_presence_ratio`

5. **Временные серии** (`--voice-quality-enable-time-series`):
   - `f0`, `amps`, `hnr_vals` (или пути к `.npy` файлам)

6. **Quality scores** (автоматически, если включены jitter, shimmer и HNR):
   - `vq_voice_quality_score`, `vq_breathiness_score`

### 4.2 Feature dependencies

- **Jitter** зависит от оценки f0 (требует `f0_method` и `f0_fmin`/`f0_fmax`)
- **Shimmer** не зависит от других фичей
- **HNR** не зависит от других фичей
- **F0 stats** зависит от оценки f0 (требует `f0_method` и `f0_fmin`/`f0_fmax`)
- **Quality scores** зависят от jitter, shimmer и HNR (все три должны быть включены)
- **Time series** зависит от включённых фичей (f0, amps, hnr_vals)

---

## 5) Performance

### 5.1 Resource costs

- **CPU**: низкие-умеренные (YIN/PYIN, оконные операции, автокорреляция)
- **GPU**: не используется (CPU-only)
- **Estimated duration**: ~1.5 секунды для типичного аудио файла (полное аудио)
- **Per-segment**: ~0.1-0.2 секунды на сегмент (зависит от размера сегмента)

### 5.2 Параметры производительности

- `hnr_frame_ms`: большие значения → меньше окон → быстрее, но менее детально
- `rms_mask_threshold`: влияет на количество обрабатываемых данных
- `f0_method`: YIN быстрее PYIN, torchcrepe медленнее (но точнее)
- Размер аудио: линейная зависимость от длительности

---

## 6) Quality validation

### 6.1 Sanity checks

- ✅ Jitter, shimmer ∈ [0.0, 1.0]
- ✅ HNR ∈ [-100.0, 100.0] (типично)
- ✅ F0 stats: f0_min ≤ f0_mean ≤ f0_max
- ✅ Voice presence ratio ∈ [0.0, 1.0]
- ✅ Quality scores ∈ [0.0, 1.0]
- ✅ Проверка на NaN/inf во всех метриках

### 6.2 Render snapshots

- ✅ JSON renderer: `render_voice_quality_extractor()` генерирует render-context JSON
- ✅ HTML renderer: `render_voice_quality_extractor_html()` генерирует HTML debug страницу
- ✅ README содержит раздел "Visualization" с рекомендациями по визуализации

---

## 7) Open issues + fix plan

### 7.1 Completed

- ✅ Добавлен `run_segments()` для обработки сегментов из Segmenter
- ✅ Изменён fallback на fail-fast для f0 оценки
- ✅ Добавлен feature gating (5 персональных флагов)
- ✅ Добавлено сохранение временных серий в `.npy` для больших массивов
- ✅ Добавлены дополнительные ML/analytics метрики
- ✅ Добавлена полная валидация выходных данных и параметров
- ✅ Добавлен progress reporting
- ✅ Добавлен contract versioning
- ✅ Создан UI renderer (JSON + HTML)
- ✅ Интегрирован в `run_cli.py` с аргументами
- ✅ Добавлены детальные error codes
- ✅ Добавлена опциональная нормализация и дополнительные параметры f0
- ✅ Добавлена опциональная интеграция с `pitch_extractor`
- ✅ Обновлён README и создан audit файл

### 7.2 Future improvements

- Возможность использования более точных методов оценки f0 (например, через интеграцию с `pitch_extractor`)
- Дополнительные метрики качества голоса (например, на основе спектрального анализа)
- Оптимизация производительности для больших аудио файлов

---

## 8) Compliance Summary

### Архитектура / контракты
- ✅ per-run storage + manifest upsert
- ✅ NPZ meta обязательные поля + validate_npz
- ✅ no-fallback policy соблюдён
- ✅ empty semantics корректны (canonical empty_reason)
- ✅ Segmenter contract соблюдён (audio/segments.json, families)

### Модели / воспроизводимость
- ✅ Нет ML моделей (signal processing только)
- ✅ Нет сетевых загрузок
- ✅ scheduler_knobs зафиксированы в meta

### Наблюдаемость / качество / ресурсы
- ✅ progress events есть и безопасны
- ✅ stage timings сохранены
- ✅ resource_costs задокументированы
- ✅ есть sanity checks + UI render

---

## 9) Implementation Details

### 9.1 F0 Estimation

- **Методы**: YIN (librosa), PYIN (librosa), torchcrepe (опционально)
- **Fail-fast**: при ошибке оценки f0 → `RuntimeError` с error_code `voice_quality_f0_estimation_failed`
- **Опциональная интеграция**: если доступны результаты `pitch_extractor`, используются их f0 значения

### 9.2 Feature Gating

- Все фичи opt-in (по умолчанию все выключены)
- 5 персональных флагов для каждой группы фичей
- Quality scores автоматически включаются, если включены jitter, shimmer и HNR

### 9.3 Per-run Storage

- Большие временные серии (>10000 элементов) сохраняются в `.npy` файлы
- Путь: `result_store/<platform_id>/<video_id>/<run_id>/voice_quality_extractor/_artifacts/*.npy`
- Регистрация в `manifest.json.artifacts[]`

### 9.4 Error Codes

- `voice_quality_audio_load_failed`: ошибка загрузки аудио
- `voice_quality_f0_estimation_failed`: ошибка оценки f0
- `voice_quality_librosa_failed`: ошибка librosa
- `voice_quality_insufficient_data`: недостаточно данных
- `voice_quality_validation_failed`: ошибка валидации
- `voice_quality_unknown`: неизвестная ошибка

---

## 10) References

- **Audit criteria**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Artifacts and schemas**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Error handling**: `docs/contracts/ERROR_HANDLING_AND_EDGE_CASES.md`
- **Component code**: `src/extractors/voice_quality_extractor/main.py`
- **Renderer code**: `src/extractors/voice_quality_extractor/render.py`
- **README**: `src/extractors/voice_quality_extractor/README.md`

