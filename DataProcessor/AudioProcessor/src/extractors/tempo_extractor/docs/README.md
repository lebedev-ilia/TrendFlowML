## `tempo_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Оценивает **темп (BPM)** и простые ритмические признаки на базе `librosa`.

**Версия**: 2.0.1  
**Категория**: rhythm  
**GPU**: не требуется  
**schema_version**: `tempo_extractor_npz_v1` (см. `SCHEMA.md`, `schemas/tempo_extractor_npz_v1.json`)

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `tempo` (sliding windows для устойчивого BPM)

Если `segments` пустой → **error**.

#### Sampling policy (tempo windows)

`Segmenter` строит family=`tempo` **адаптивно и нелинейно по длительности** (sub-linear), чтобы:
- на коротких видео можно было брать почти “1 секунда = 1 окно” (например ~32s → ~32 окна),
- на длинных видео рост замедлялся (например ~10 минут → ~300 окон, а не 600).

Параметры кривой (`k/min/max/linear_until/cap_duration`) сохраняются в `audio/segments.json` (см. `docs/contracts/SEGMENTER_CONTRACT.md`) и подбираются по tradeoff **cost ↔ quality**.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/tempo_extractor/tempo_extractor_features.npz` (**фиксированное имя**)

Схема: `tempo_extractor_npz_v1` (Audit v3, см. `SCHEMA.md`).

#### Audit v4 — заметки по NPZ

- На reference **A**: tabular **F=11**, NaN **0**; **`device_used`** в **`meta`**; ось **`tempo`**: **N=12** на этом run; `tempo_estimates` — длинный вектор с полного трека.
- См. `SCHEMA.md` §Audit v4 про **`duration_sec`** и отсутствие `features_enabled` в meta.
- Observability (audit v4.2): `meta.stage_timings_ms`, `meta.tempo_resource_profile` (env: `AP_TEMPO_RESOURCE_PROFILE=1`)

Полезные поля NPZ:
- `tempo_bpm_*` (mean/median/std)
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`, `bpm_by_segment` (canonical axis)
- `tempo_confidence`, `warnings`
- `empty_reason` (например `tempo_all_segments_failed`)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

### Progress Reporting

`tempo_extractor` поддерживает `progress_callback` для отображения прогресса обработки:
- Прогресс обновляется каждые 10% сегментов
- Отображается количество обработанных сегментов и процент выполнения
- Поддерживается как последовательная, так и параллельная обработка

### Обработка ошибок

**Политика (Audit v3)**:
- Отсутствие segments → `ValueError("segments is empty (no-fallback)")`
- Некорректный входной файл → ошибка с описанием
- Partial segment failures → `segment_mask=False` для failed сегментов, `bpm_by_segment=NaN`
- All segments failed → `status="empty"`, `empty_reason="tempo_all_segments_failed"`

**Логирование**:
- `_log_extraction_start()` вызывается в начале `run_segments()`
- `_log_extraction_success()` вызывается при успешном завершении
- `_log_extraction_error()` вызывается при ошибках

### Конфигурация

**Параметры конфигурации компонента**:

| Параметр | Тип | Значение по умолчанию | Допустимые значения | Описание |
|----------|-----|----------------------|---------------------|----------|
| `device` | str | `"auto"` | `"auto"` \| `"cpu"` \| `"cuda"` | Устройство для обработки |
| `sample_rate` | int | `22050` | `> 0` | Частота дискретизации (Hz) |
| `hop_length` | int | `512` | `> 0` | Размер hop для onset detection |
| `aggregate` | str | `"median"` | `"median"` \| `"mean"` | Метод агрегации BPM оценок |
| `average_channels` | bool | `true` | `true` \| `false` | Усреднять каналы для многоканального аудио |
| `windowed_bpm` | bool | `false` | `true` \| `false` | Включить пер-оконные последовательности BPM (для `run()`) |
| `window_sec` | float | `15.0` | `> 0` | Размер окна в секундах (для `windowed_bpm=true`) |
| `step_sec` | float | `5.0` | `> 0` | Шаг окна в секундах (для `windowed_bpm=true`) |

**Пример конфигурации**:
```python
{
    "device": "auto",
    "sample_rate": 22050,
    "hop_length": 512,
    "aggregate": "median",
    "average_channels": true,
    "windowed_bpm": false,
}
```

### Выходные поля

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Глобальные метрики (всегда присутствуют)

- `tempo_bpm`: основной BPM (median или mean, в зависимости от `aggregate`)
- `tempo_bpm_mean`: среднее значение BPM
- `tempo_bpm_median`: медиана BPM
- `tempo_bpm_std`: стандартное отклонение BPM
- `tempo_estimates`: массив всех оценок BPM (float32[])
- `confidence`: уверенность оценки (0.0-1.0, вычисляется как `1.0 / (1.0 + std/mean)`)
- `warnings`: список предупреждений (например, `["low_confidence"]`, `["tempo_out_of_range"]`, `["signal_too_quiet"]`)
- `sample_rate`: частота дискретизации (Hz)
- `hop_length`: размер hop для onset detection
- `duration`: длительность аудио (секунды)
- `device_used`: устройство обработки

#### Canonical axis (Audit v3, для `run_segments()`)

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: границы и центр сегментов (float32[])
- `segment_mask`: маска валидных сегментов (bool[], False = failed)
- `bpm_by_segment`: BPM для каждого сегмента (float32[], NaN для failed)
- `segments_count`: количество сегментов

#### Legacy (для `run()` с `windowed_bpm=true`)

- `windowed_bpm`: словарь с `times_sec`, `bpm`, `bpm_mean`, `bpm_median`, `bpm_std`

### Render (dev-only)

Offline HTML render для отладки результатов tempo_extractor:

- **Без CDN**: vanilla `<canvas>`, без Chart.js
- **Вход**: NPZ файл (`tempo_extractor_features.npz`)
- **Выход**: HTML страница с графиком BPM vs segment_center_sec, summary, distributions

Запуск (из корня AudioProcessor):

```bash
python -c "
from src.extractors.tempo_extractor.utils.render import render_tempo_extractor_html
render_tempo_extractor_html('result_store/.../tempo_extractor/tempo_extractor_features.npz', 'debug_tempo.html')
"
```

### Важно (по коду)

- Экстрактор вычисляет глобальные `tempo_bpm_*` по валидным сегментам (или full-track fallback).
- Canonical axis (`segment_*`, `bpm_by_segment`) — основной формат для NPZ v1.


