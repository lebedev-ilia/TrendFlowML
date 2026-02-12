## `clap_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Считает **семантический аудио‑эмбеддинг** CLAP по Segmenter‑окнам и отдаёт:
- эмбеддинг по каждому окну,
- агрегат (mean) по всему видео.

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

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Полезные поля payload (внутри NPZ):
- `embedding` (`float32[D]`)
- `embedding_sequence` (`float32[N,D]`)
- `segment_centers_sec` (`float32[N]`)
- `device_used`

### Особенности реализации (по коду)

- **Runtime**: inprocess (PyTorch). Модель загружается лениво при первом вызове.
- **No-network policy**: загрузка весов строго локально через `dp_models` (дополнительно выставляются offline env для HF).
- **Окна**: берутся из `audio/segments.json` family=`clap` (Segmenter sampling curve).
- **Ограничение длины окна**: модель ограничивает эффективную длину аудио (`max_audio_length`), очень длинные окна будут обрезаться.

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


