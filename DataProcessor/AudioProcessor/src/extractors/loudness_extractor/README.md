## `loudness_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Считает **громкость/динамику**:
- RMS, peak, dBFS,
- опционально LUFS (если установлен `pyloudnorm`),
- статистики по short-term RMS.

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `primary` (окна вокруг time‑anchors)

Если `segments` пустой → **error**.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/loudness_extractor/loudness_extractor_features.npz` (**фиксированное имя**)

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Полезные поля payload:
- `rms`, `peak`, `dbfs`, `lufs` (может быть NaN)
- `segment_rms_*` агрегаты
- `lufs_present` (bool)

### Важно (по коду)

- Экстрактор считает **frame-wise RMS stats** (`frame_rms_*` + `frame_rms_stats_vector`) на базе `frame_length/hop_length`.
- LUFS вычисляется **best-effort**: если `pyloudnorm` отсутствует или падает — `lufs=None`, `lufs_present=false`.

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

### Progress Reporting

`loudness_extractor` поддерживает `progress_callback` для отображения прогресса обработки:
- Прогресс обновляется каждые 10% сегментов
- Отображается количество обработанных сегментов и процент выполнения
- Поддерживается как последовательная, так и параллельная обработка

### Обработка ошибок

**Политика NO FALLBACK**:
- Отсутствие segments → `ValueError("segments is empty (no-fallback)")`
- Некорректный входной файл → ошибка с описанием
- Ошибки обработки сегментов → логируются и пробрасываются дальше

**Логирование**:
- `_log_extraction_start()` вызывается в начале `run_segments()`
- `_log_extraction_success()` вызывается при успешном завершении
- `_log_extraction_error()` вызывается при ошибках

### Render

`loudness_extractor` поддерживает генерацию render-context JSON и HTML debug страницы:
- **Render-context JSON**: содержит summary, timeline (RMS, dBFS, LUFS), distributions
- **HTML debug страница**: интерактивные графики для RMS, dBFS и LUFS (если доступен) по времени
- Управление через флаги в `global_config.yaml`: `render.enable_render` и `render.enable_html_render`

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/loudness_extractor_costs_v1.json`

**Единица обработки**: `audio_window` (Segmenter `families.primary`)

**Evidence**:
- micro‑bench: `scripts/baseline/run_loudness_extractor_micro.py`

## Quality validation & human-friendly inspection

Human‑friendly demo:
- `scripts/baseline/demo_loudness_extractor_quality.py` (HTML: timeline `segment_dbfs/segment_rms` + sanity checks)


