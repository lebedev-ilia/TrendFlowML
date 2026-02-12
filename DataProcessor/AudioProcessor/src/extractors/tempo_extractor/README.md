## `tempo_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Оценивает **темп (BPM)** и простые ритмические признаки на базе `librosa`.

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

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Полезные поля payload:
- `tempo_bpm_*` (mean/median/std)
- `windowed_times_sec`, `windowed_bpm` (последовательности по окнам)
- `confidence`, `warnings`
- `warnings` (например `low_confidence`, `tempo_out_of_range`)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

### Progress Reporting

`tempo_extractor` поддерживает `progress_callback` для отображения прогресса обработки:
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

### Важно (по коду)

- Экстрактор вычисляет глобальные `tempo_bpm_*` **по всему аудиотреку** (backward compatibility).
- Пер-оконные последовательности `windowed_times_sec/windowed_bpm` включаются параметром `windowed_bpm` в конструкторе.


