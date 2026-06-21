## `loudness_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Считает **громкость/динамику**:
- RMS, peak, dBFS,
- опционально LUFS (если установлен `pyloudnorm`),
- статистики по short-term RMS.

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `primary` (окна вокруг time‑anchors)

Пустота/ошибки:
- если `audio/segments.json` содержит `audio_present=false` → экстрактор **не запускается**, а артефакт имеет `status="empty"` (валидный кейс Audit v3).
- если `audio_present=true`, но `families.primary.segments` отсутствует/пустой → **error** (no-fallback).

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/loudness_extractor/loudness_extractor_features.npz` (**фиксированное имя**)

Схема: `loudness_extractor_npz_v2` (см. `schemas/loudness_extractor_npz_v2.json`, `docs/SCHEMA.md`).

Audit v3 rollout:
- `schema_version="loudness_extractor_npz_v2"` (machine schema) — см. `docs/SCHEMA.md`

#### Audit v4 — сводка NPZ

| Группа | Заметка |
|--------|---------|
| `feature_names` / `feature_values` | **F=18**, все float; порядок фиксирован [`npz_savers/loudness.py`](../../../core/npz_savers/loudness.py). |
| `lufs_present` | Скаляр **bool** отдельно от tabular. |
| `segment_*` | Ось **N** = числу `families.primary.segments` (на общем e2e run **N** может отличаться от `chroma`/`key` и т.д.). |
| Глобальные vs окна | Глобальные **rms/peak/dbfs/lufs** и **frame_*** считаются по **полному** клипу после обработки сегментов; ряды `segment_*` — по окнам. |

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): покомпонентные тайминги (ms), пишутся в NPZ meta
- `meta.loudness_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_LOUDNESS_RESOURCE_PROFILE=1`

#### Полезные поля payload:

**Глобальные метрики (всегда включены):**
- `rms`: RMS значение по всему треку (float)
- `peak`: пиковое значение по всему треку (float)
- `dbfs`: dBFS значение (20*log10(rms + eps)) (float)
- `lufs`: LUFS значение (float, может быть None если pyloudnorm недоступен)
- `lufs_present`: флаг наличия LUFS (bool)
- `sample_rate`: частота дискретизации (int)
- `duration`: длительность аудио в секундах (float)
- `frame_length`: размер окна для frame-wise RMS (int)
- `hop_length`: размер hop для frame-wise RMS (int)
- `frames_count`: количество кадров для frame-wise RMS (int)

**Frame-wise RMS статистики (всегда включены):**
- `frame_rms_mean`: среднее значение frame-wise RMS (float)
- `frame_rms_std`: стандартное отклонение frame-wise RMS (float)
- `frame_rms_median`: медиана frame-wise RMS (float)
- `frame_rms_p10`: 10-й перцентиль frame-wise RMS (float)
- `frame_rms_p90`: 90-й перцентиль frame-wise RMS (float)
- `frame_rms_stats_vector`: вектор статистик [mean, std, median, p10, p90] (list[float])

**Сегментные метрики (для `run_segments()`):**
- `segments_count`: количество обработанных сегментов (int)
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: временная ось (float32[N])
- `segment_mask`: маска валидных сегментов (bool[N]; false для failed)
- `segment_rms`: RMS по каждому сегменту (float32[N])
- `segment_peak`: пиковое значение по каждому сегменту (float32[N])
- `segment_dbfs`: dBFS по каждому сегменту (float32[N])
- `segment_lufs`: LUFS по каждому сегменту (float32[N], может содержать NaN)

**Агрегированные статистики по сегментам (для `run_segments()`):**
- `segment_rms_mean`: среднее RMS по сегментам (float)
- `segment_rms_std`: стандартное отклонение RMS по сегментам (float)
- `segment_rms_median`: медиана RMS по сегментам (float)
- `segment_rms_p10`: 10-й перцентиль RMS по сегментам (float)
- `segment_rms_p90`: 90-й перцентиль RMS по сегментам (float)

**Метаданные:**
- `device_used`: устройство обработки (str)

### Конфигурация

#### Через global_config.yaml

```yaml
audio:
  extractors:
    loudness:
      enabled: true
      sample_rate: 22050
      frame_length: 2048
      hop_length: 512
      mix_to_mono: true
```

#### Через Python API

```python
from src.extractors.loudness_extractor import LoudnessExtractor

extractor = LoudnessExtractor(
    device="auto",           # "auto" | "cuda" | "cpu"
    sample_rate=22050,       # Частота дискретизации
    frame_length=2048,       # Размер окна для frame-wise RMS
    hop_length=512,          # Размер hop для frame-wise RMS
    mix_to_mono=True,        # Сводить стерео в моно
)
```

### Алгоритм работы

#### Метод `run()` (полное аудио):

1. **Загрузка аудио**: через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. **Сведение в моно**: если `mix_to_mono=True`, усреднение каналов
3. **Вычисление глобальных метрик**:
   - RMS: `sqrt(mean(x^2))`
   - Peak: `max(abs(x))`
   - dBFS: `20*log10(rms + eps)`
4. **Frame-wise RMS**: вычисление RMS по окнам размера `frame_length` с hop `hop_length`
5. **Статистики frame-wise RMS**: mean, std, median, p10, p90
6. **LUFS** (best-effort): если `pyloudnorm` доступен, вычисление integrated loudness

#### Метод `run_segments()` (сегменты от Segmenter):

1. **Загрузка сегментов**: из `families.primary.segments[]` от Segmenter
2. **Обработка каждого сегмента**: вычисление RMS, peak, dBFS, LUFS для каждого сегмента
3. **Параллельная обработка**: поддержка `segment_parallelism` и `max_inflight` для параллельной обработки сегментов
4. **Агрегация результатов**: вычисление статистик по сегментам (mean, std, median, p10, p90)
5. **Глобальные метрики**: также вычисляются для полного аудио файла

### Важно (по коду)

- Экстрактор считает **frame-wise RMS stats** (`frame_rms_*` + `frame_rms_stats_vector`) на базе `frame_length/hop_length`.
- LUFS вычисляется **best-effort**: если `pyloudnorm` отсутствует или падает — `lufs=None`, `lufs_present=false`.
- Для `run_segments()` всегда вычисляются глобальные метрики по полному аудио файлу (backward compatibility).
- Сегменты обрабатываются параллельно если `segment_parallelism > 1`.

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

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("loudness_extractor_features.npz", allow_pickle=True)
payload = data["payload"].item()

# Глобальные метрики
rms = payload["rms"]
peak = payload["peak"]
dbfs = payload["dbfs"]
lufs = payload.get("lufs")  # Может быть None
lufs_present = payload["lufs_present"]

print(f"RMS: {rms:.6f}, Peak: {peak:.6f}, dBFS: {dbfs:.2f} dB")
if lufs_present:
    print(f"LUFS: {lufs:.2f}")
```

#### Анализ сегментов (для run_segments)

```python
# Сегментные данные (Audit v3 strict alignment)
segment_centers = payload["segment_center_sec"]
segment_mask = payload["segment_mask"]
segment_rms = payload["segment_rms"]
segment_dbfs = payload["segment_dbfs"]

# Агрегированные статистики
rms_mean = payload["segment_rms_mean"]
rms_std = payload["segment_rms_std"]
rms_median = payload["segment_rms_median"]

print(f"Segment RMS: mean={rms_mean:.6f}, std={rms_std:.6f}, median={rms_median:.6f}")
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (`families.primary.segments[]`)
- **pyloudnorm** (опционально): библиотека для вычисления LUFS
- **AudioUtils**: загрузка и предобработка аудио
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
