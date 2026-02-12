# Audit: `rhythmic_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`rhythmic_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: поддерживает `run_segments()` для работы на сегментах от Segmenter (family `rhythmic`)
- ✅ **No-fallback policy**: fail-fast при ошибках выбранного backend (no-fallback для backend selection)
- ✅ **Model system**: не использует ML модели через ModelManager (signal processing: librosa/essentia)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `rhythmic_extractor_features.npz`, .npy файлы в per-run storage
- ✅ **Feature gating**: персональные флаги для каждой группы фичей (5 групп, все opt-in)
- ✅ **Error handling**: детальные error codes (6 типов)
- ✅ **Segments validation**: полная валидация структуры сегментов (для `run_segments()`)
- ✅ **Output validation**: полная валидация выходных данных (диапазоны, NaN/inf, консистентность)
- ✅ **Parameter validation**: полная валидация входных параметров (fail-fast)
- ✅ **Progress reporting**: обновление прогресса для каждого этапа и сегмента
- ✅ **UI Render**: renderer реализован в `src/extractors/rhythmic_extractor/render.py` + HTML renderer для дебага
- ✅ **Contract version**: `rhythmic_contract_version` для валидации совместимости с downstream extractors
- ✅ **Additional metrics**: rhythm_syncopation_score, rhythm_polyrhythm_score, rhythm_beat_strength_mean/std, rhythm_metrical_stability, rhythm_tempo_variation, rhythm_beat_consistency, rhythm_tempo_mean/std/min/max
- ✅ **Optional normalization**: опциональная нормализация аудио через флаг (по умолчанию выключена)
- ✅ **Backend selection**: явный выбор backend (librosa/essentia) через CLI, no-fallback policy
- ✅ **Additional parameters**: дополнительные параметры для librosa (start_bpm, std_bpm, ac_size, max_tempo)

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run(input_uri, tmp_path)` и `BaseExtractor.run_segments(input_uri, tmp_path, segments)`
- [x] Не делает скрытых глобальных сайд-эффектов (модели не загружаются, signal processing только)
- [x] Требование специфичного входа декларировано: `audio/audio.wav` (обязательно), `audio/segments.json` family `rhythmic` (для `run_segments()`) (см. README)

**Evidence**:
- `src/extractors/rhythmic_extractor/main.py:490` — метод `run()`
- `src/extractors/rhythmic_extractor/main.py:576` — метод `run_segments()`
- `src/extractors/rhythmic_extractor/main.py:500` — проверка входного файла

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + опционально `audio/segments.json` family `rhythmic`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()` (в `run_segments()`)
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/rhythmic_extractor/main.py:576` — метод `run_segments()` принимает сегменты
- `src/extractors/rhythmic_extractor/main.py:625-635` — загрузка сегментов через `AudioUtils.load_audio_segment()`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("rhythmic | segments is empty (no-fallback)")`
- [x] Ошибка выбранного backend → fail-fast с детальным error_code (no-fallback для backend selection)
- [x] Ошибка вычисления метрики → fail-fast с детальным error_code
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/rhythmic_extractor/main.py:590` — проверка segments
- `src/extractors/rhythmic_extractor/main.py:310-350` — fail-fast для backend selection (no-fallback)
- `src/extractors/rhythmic_extractor/main.py:558-566` — fail-fast для валидации выходных данных

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `rhythmic_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] .npy файлы сохраняются в `result_store/<platform_id>/<video_id>/<run_id>/rhythmic_extractor/_artifacts/` (per-run storage)
- [x] .npy файлы регистрируются в `manifest.json.artifacts[]` (через `meta.artifacts`)

**Evidence**:
- `run_cli.py:952` — сохранение через `_save_component_npz()` с фиксированным именем
- `src/extractors/rhythmic_extractor/main.py:420-440` — метод `_save_beat_times_npy()` сохраняет в per-run storage
- `run_cli.py:952-1020` — сохранение NPZ с artifacts в meta

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `beat_times: float32[N]` — временные метки ударов (feature-gated)
- [x] `segment_centers_sec: float32[L]` — центры сегментов (feature-gated, для `run_segments()`)
- [x] `segment_durations_sec: float32[L]` — длительности сегментов (feature-gated, для `run_segments()`)
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (не используется, signal processing)
- [x] `device_used`
- [x] `rhythmic_contract_version` (для валидации совместимости)

**Evidence**:
- `run_cli.py:952-1020` — сохранение NPZ с обязательными полями
- `src/extractors/rhythmic_extractor/main.py:550-556` — добавление contract version

### 2.6 Valid empty outputs

- [x] При `status="empty"`: `empty_reason` обязательно
- [x] Фичи при empty: NaN для числовых значений, пустые массивы для массивов
- [x] Empty не скрывает ошибки парсинга/модели/валидации

**Evidence**:
- `run_cli.py:2046-2057` — обработка empty статуса
- `src/extractors/rhythmic_extractor/main.py:490-575` — обработка empty случаев

### 2.7 Model system: no-network + ModelManager

- [x] ML модели не используются (signal processing only)
- [x] `models_used[]` пустой
- [x] Библиотеки (librosa, essentia) не требуют сетевых загрузок

**Evidence**:
- `src/extractors/rhythmic_extractor/main.py:23` — dependencies: ["librosa", "numpy"]
- `src/extractors/rhythmic_extractor/main.py:310-350` — использование librosa/essentia без сетевых загрузок

### 2.8 Segmenter contract: audio/segments.json

- [x] Использует `families.rhythmic.segments[]` из `audio/segments.json`
- [x] Проверяет `schema_version="audio_segments_v1"` (fail-fast)
- [x] Использует `start_sample/end_sample` для загрузки сегментов

**Evidence**:
- `run_cli.py:1819` — проверка наличия `families.rhythmic.segments`
- `src/extractors/rhythmic_extractor/main.py:625-635` — загрузка сегментов через `AudioUtils.load_audio_segment()`

### 2.9 Наблюдаемость: progress + stage timings

- [x] Stage-based прогресс: `load_input`, `run_extractors`, `save_npz`, `validate_artifact`, `update_manifest`
- [x] Progress reporting для каждого этапа (в `run()`)
- [x] Progress reporting для каждого сегмента (в `run_segments()`, каждые 10%)
- [x] Stage timings сохранены в NPZ meta

**Evidence**:
- `src/extractors/rhythmic_extractor/main.py:490-575` — progress reporting в `run()`
- `src/extractors/rhythmic_extractor/main.py:640-680` — progress reporting в `run_segments()`
- `run_cli.py:2108` — сохранение stage_timings_ms в meta

### 2.10 Feature contract: управление выходными фичами

- [x] Feature gating через CLI/конфиг: `--rhythmic-enable-basic-metrics`, `--rhythmic-enable-interval-stats`, `--rhythmic-enable-regularity-metrics`, `--rhythmic-enable-beat-times`, `--rhythmic-enable-tempo-metrics`
- [x] Персональные флаги для каждой группы фичей (5 групп, все opt-in)
- [x] В `meta` фиксируются: `features_enabled[]`, `features_produced[]`

**Evidence**:
- `run_cli.py:1307-1311` — CLI аргументы для feature gating
- `src/extractors/rhythmic_extractor/main.py:550-556` — добавление `_features_enabled` в payload
- `run_cli.py:952-1020` — сохранение `features_enabled` в meta

### 2.11 Производительность и ресурсы

- [x] Latency per unit: задокументировано в README (~1.2 секунд для типичного аудио файла)
- [x] CPU RSS peak: не измерено (CPU-only, умеренные требования)
- [x] GPU VRAM peak: не используется
- [x] Batching: поддерживается через `run_segments()` с `segment_parallelism`

**Evidence**:
- `src/extractors/rhythmic_extractor/README.md:160-170` — Performance characteristics
- `src/extractors/rhythmic_extractor/main.py:576` — метод `run_segments()` с поддержкой параллелизма

### 2.12 Проверка качества выхода

- [x] Sanity checks: диапазоны значений (tempo_bpm ∈ [40, 300], regularity ∈ [0, 1], beat_density ∈ [0, 10])
- [x] NaN/inf проверка: все метрики проверяются на NaN и inf
- [x] Консистентность: проверка согласованности `tempo_bpm` и `avg_period` (допуск 10 BPM)
- [x] UI render: renderer реализован в `src/extractors/rhythmic_extractor/render.py`

**Evidence**:
- `src/extractors/rhythmic_extractor/main.py:244-300` — метод `_validate_output()`
- `src/extractors/rhythmic_extractor/render.py:1-250` — renderer для JSON и HTML

### 2.13 Документация

- [x] README содержит все обязательные разделы: Input contract, Output contract, Models, Feature dependencies, Configuration, Parameters, Parallelization, Quality, Visualization
- [x] README содержит раздел "Visualization" с рекомендациями по визуализации данных для UI/сайта

**Evidence**:
- `src/extractors/rhythmic_extractor/README.md` — полная документация

---

## 3) Models Used

ML модели **не используются** (signal processing only).

**Библиотеки:**
- **librosa**: основная библиотека для beat tracking (default backend)
- **essentia** (опционально): более точный алгоритм beat tracking (если доступен)

---

## 4) Features List + Gating Status

### Feature Groups (все opt-in, default: False)

1. **Basic metrics** (`--rhythmic-enable-basic-metrics`):
   - `rhythm_tempo_bpm`: темп в ударах в минуту (BPM)
   - `rhythm_beats_count`: количество обнаруженных битов
   - `rhythm_beat_density`: плотность ударов (beats/sec)

2. **Interval stats** (`--rhythmic-enable-interval-stats`):
   - `rhythm_avg_period_sec`: средний период между ударами
   - `rhythm_period_std_sec`: стандартное отклонение периодов
   - `rhythm_median_period_sec`: медианный период между ударами
   - `rhythm_min_period_sec`: минимальный период между ударами
   - `rhythm_max_period_sec`: максимальный период между ударами

3. **Regularity metrics** (`--rhythmic-enable-regularity-metrics`):
   - `rhythm_regularity`: коэффициент регулярности ритма (0-1)
   - `rhythm_syncopation_score`: мера синкопированности
   - `rhythm_polyrhythm_score`: мера полиритмичности
   - `rhythm_beat_strength_mean`: средняя сила ударов
   - `rhythm_beat_strength_std`: стандартное отклонение силы ударов
   - `rhythm_metrical_stability`: метрическая стабильность

4. **Beat times** (`--rhythmic-enable-beat-times`):
   - `beat_times`: массив временных меток ударов (float32[N])
   - Если `beat_times.size > 10000`, сохраняется в `.npy` файл

5. **Tempo metrics** (`--rhythmic-enable-tempo-metrics`):
   - `rhythm_median_bpm`: медианный темп
   - `rhythm_tempo_variation`: вариация темпа
   - `rhythm_beat_consistency`: консистентность ударов (0-1)
   - `rhythm_tempo_mean/std/min/max`: статистика темпа по сегментам (для `run_segments()`)

---

## 5) Performance

**Resource costs**:
- **CPU**: умеренные (beat tracking требует вычислений)
- **GPU**: не используется
- **Estimated duration**: ~1.2 секунд для типичного аудио файла

**Параметры производительности**:
- `hop_length`: меньшие значения → выше разрешение → точнее, но медленнее
- `sample_rate`: более высокие значения → точнее, но медленнее
- Essentia обычно быстрее, чем librosa для beat tracking

---

## 6) Quality Validation

### Sanity Checks

- ✅ **Диапазоны**: `tempo_bpm ∈ [40, 300]`, `regularity ∈ [0, 1]`, `beat_density ∈ [0, 10]`
- ✅ **NaN/inf**: Проверка на наличие NaN и inf во всех метриках
- ✅ **Консистентность**: Проверка согласованности `tempo_bpm` и `avg_period` (допуск 10 BPM)

### UI Render

- ✅ **JSON renderer**: `render_rhythmic_extractor()` для генерации render-context JSON
- ✅ **HTML renderer**: `render_rhythmic_extractor_html()` для локального дебага (debug-only)

**Evidence**:
- `src/extractors/rhythmic_extractor/render.py:1-250` — renderer реализация

---

## 7) Open Issues + Fix Plan

Нет открытых проблем. Все требования выполнены.

---

## 8) Compliance Summary

| Критерий | Статус | Примечания |
|----------|--------|------------|
| Segmenter contract | ✅ | Поддерживает `run_segments()` для family `rhythmic` |
| No-fallback policy | ✅ | Fail-fast для backend selection, no-fallback |
| Model system | ✅ | Signal processing only, no ML models |
| NPZ schema | ✅ | `audio_npz_v1` с обязательными полями |
| Per-run storage | ✅ | Фиксированное имя, .npy файлы в per-run storage |
| Feature gating | ✅ | 5 групп фичей, все opt-in |
| Error handling | ✅ | 6 типов error codes |
| Validation | ✅ | Полная валидация параметров и выходных данных |
| Progress reporting | ✅ | Progress для каждого этапа и сегмента |
| UI Render | ✅ | JSON + HTML renderer |
| Contract versioning | ✅ | `rhythmic_contract_version` |
| Documentation | ✅ | Полная документация в README |

**Общий статус**: ✅ **done** — все критерии выполнены.

