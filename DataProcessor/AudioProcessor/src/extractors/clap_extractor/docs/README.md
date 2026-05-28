## `clap_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Считает **семантический аудио‑эмбеддинг** CLAP по Segmenter‑окнам и отдаёт:
- эмбеддинг по каждому окну,
- агрегат по всему видео (**robust aggregation** по валидным сегментам).

### Входы (строго, no‑fallback)

- **`audio/audio.wav`**: готовит Segmenter.
- **`audio/segments.json`**: contract `audio_segments_v1`, family: `clap` (короткие окна на нелинейной кривой).

Если `segments` пустой → **error**.

#### Sampling policy (clap windows)

Segmenter строит family=`clap` по **универсальной нелинейной кривой**:
- на коротких видео можно близко к 1:1 (секунда → окно),
- на длинных видео рост замедляется и упирается в `max_windows`.

Параметры кривой (`k/min/max/linear_until/cap_duration`) лежат в `audio/segments.json` (см. `docs/contracts/SEGMENTER_CONTRACT.md`) и подбираются по tradeoff **cost ↔ quality** для конкретного компонента.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/clap_extractor/clap_extractor_features.npz` (**фиксированное имя**)

Схема (Audit v3): `clap_extractor_npz_v1` (см. `schemas/clap_extractor_npz_v1.json` и `SCHEMA.md`).

#### Audit v4 — сводка NPZ

| Ключ / группа | Форма | Tier | Заметка |
|---------------|-------|------|---------|
| `feature_names` / `feature_values` | F=5 | model_facing | Порядок имён **фиксирован савером**: `embedding_dim`, `clap_norm`, `clap_magnitude_mean`, `clap_magnitude_std`, `segments_count`. |
| `embedding` | `float32[D]` | model_facing | Агрегат (robust aggregation по валидным строкам `embedding_sequence`). |
| `embedding_sequence` | `float32[N,D]` | model_facing | Строгое выравнивание с Segmenter; для `segment_mask=false` — NaN в строке. |
| `embedding_present` | `bool` | analytics | `True`, если агрегат ненулевой длины. |
| `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`, `segment_embedding_norm` | `float32[N]` / `bool[N]` | analytics | Ось времён и валидность окон. |
| `meta` | object | debug | В т.ч. `models_used`, `max_audio_length_sec`, trim‑статистика. |

Ключевые поля NPZ:
- `embedding` (`float32[D]`, model_facing): агрегированный эмбеддинг по видео (robust aggregation)
- `embedding_sequence` (`float32[N,D]`, model_facing): эмбеддинги по каждому Segmenter‑окну (strict-aligned)
  - для `segment_mask=false` строка содержит `NaN`
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec` (`float32[N]`, analytics): time-axis
- `segment_mask` (`bool[N]`, analytics): валидность сегмента (strict alignment; missing → mask=false + NaN)
- `segment_embedding_norm` (`float32[N]`, analytics): L2‑норма per‑segment embedding (NaN для masked)
- `embedding_present` (`bool`, analytics): агрегированный embedding присутствует

Скалярные фичи (tabular, model_facing) — **порядок в `feature_names` как в савере**:
- `embedding_dim` (int, обычно 512)
- `clap_norm` (float): L2 норма агрегированного `embedding` (совпадает с `np.linalg.norm(embedding)`)
- `clap_magnitude_mean`, `clap_magnitude_std` (float): по элементам **агрегата** (`|embedding|` mean/std)
- `segments_count` (int): количество **валидных** окон (`sum(segment_mask)`)

Audit v3 observability:
- `max_audio_length_sec` (float): эффективный лимит длины окна
- `trimmed_segments_count` (int) и `trimmed_ratio` (float): доля сегментов, где был trim до `max_audio_length_sec`

Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): покомпонентные тайминги (ms), пишутся в NPZ meta
- `meta.clap_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_CLAP_RESOURCE_PROFILE=1`

### Особенности реализации (по коду)

- **Runtime**: inprocess (PyTorch). Модель загружается лениво при первом вызове.
- **No-network policy**: загрузка весов строго локально через `dp_models` (дополнительно выставляются offline env для HF).
- **Окна**: берутся из `audio/segments.json` family=`clap` (Segmenter sampling curve).
- **Ограничение длины окна**: модель ограничивает эффективную длину аудио (`max_audio_length=10.0` секунд). Если окно длиннее — сигнал обрезается, и это репортится через `trimmed_ratio`.
- **Sample rate**: по умолчанию 48000 Hz (требуется для CLAP модели)
- **Batch processing**: API `extract_batch_segments()` сохранён, но в audited режиме использует per-file `run_segments()` (корректность-first, строгий mask контракт)
- **Progress reporting**: обновление прогресса каждые 10% сегментов/батчей (если `progress_callback` установлен)
- **Методы**: 
  - `run_segments()`: основной метод для production (использует сегменты от Segmenter)
  - `run()`: **disabled** в audited режиме (только segment-based контракт)

### Модель / ModelManager

Модель CLAP грузится **строго локально** через `dp_models`:
- spec: `dp_models/spec_catalog/audio/laion_clap.yaml`
- артефакт: `${DP_MODELS_ROOT}/audio/laion_clap/clap_ckpt.pt`

Сетевые загрузки запрещены.

### Производительность / batching

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/clap_extractor_costs_v1.json`

**Единица обработки**: `audio_window` (Segmenter `families.clap`)

**Evidence**:
- micro‑bench: `scripts/baseline/run_clap_extractor_micro.py`

## Quality validation & human-friendly inspection

Human‑friendly demo:
- `scripts/baseline/demo_clap_extractor_quality.py` (HTML: norms + cosine stability + sanity checks)


