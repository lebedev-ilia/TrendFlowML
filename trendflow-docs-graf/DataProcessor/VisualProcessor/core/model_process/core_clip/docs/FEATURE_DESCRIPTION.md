# core_clip — описание фич (Audit v2/v3)

**Компонент:** `core_clip` (VisualProcessor **core** provider)  
**producer** в `meta` NPZ: `core_clip`  
**producer_version (код `main.py`):** `2.1` (см. `VERSION`)  
**schema_version NPZ:** `core_clip_npz_v2`  
**Артефакт:** `<result_store>/<platform_id>/<video_id>/<run_id>/core_clip/embeddings.npz` (см. [README.md](../README.md), `ARTIFACT_FILENAME`)

## Назначение

Эмбеддинги **CLIP** по кадрам (inprocess или **Triton** image+text), нуллшот-промпты и скоры для downstream (shot quality, сцена/стиль, cut, popularity, **Places365** top-K), плюс `consecutive_cosine_prev` между соседними кадрами.

## Ключи NPZ (обзор)

- Ось **N** (`frame_indices`, `times_s`, `frame_embeddings` (N×D), per-frame score-матрицы, `consecutive_cosine_prev`, `places365_topk_*` на (N, K)).  
- Текстовые эмбеддинги и промпты: отдельные массивы (не длина N) — `*_prompts`, `*_text_embeddings` для соответствующих голов.  
- Видео-уровень Places365: `places365_video_topk_*` — длина K.  
- **meta** — `model_name`, `batch_size`, `runtime` (`triton-gpu` / `inprocess`), `device`, `prompts_version`, `export_prompt_scores`, `places365_topk_k`, `backend_proxy_version`, `stage_timings_ms`, `models_used`, …

## Тайминги (`stage_timings_ms` → `meta_timing_*` в CSV)

В типичном Triton-ране встречаются в т.ч.: `initialization`, `triton_init`, `model_init`, `image_embeddings_total` (+ вложенно `image_frame_loading`, `image_preprocessing`, `image_inference`), `text_embeddings_prep`, `text_embeddings_postproc`, `text_inference`, `saving`, `total` (имена = ключи в meta; в плоском отчёте `meta_timing_<key>`).

## Нормальные диапазоны (QA / флаг `--ranges` в валидаторе)

| Поле / группа | Ожидание (finite, если не оговорено иное) |
|---------------|------------------------------------------|
| `consecutive_cosine_prev` | **∈ [−1, 1]** (косинус соседних image-эмбеддингов; первый кадр допускается **NaN**) |
| `*_scores` (все per-frame score-матрицы, softmax по промптам) | **∈ [0, 1]**; строка обычно **≈** сумма 1 (допуск в валидаторе) |
| `places365_topk_scores` | **∈ [0, 1]** |
| `places365_video_topk_scores` | **∈ [0, 1]** |
| `frame_embeddings` | строки как правило **L2-нормированы** (‖v‖ ≈ 1, допуск в валидаторе) |
| `times_s` | **Неубывающий** ряд (union-ось кадра) |
| `len(frame_indices)` vs `meta.total_frames` | **N ≤ total_frames** (семпл ⊆ union) |
| `meta.batch_size` | **1…256** (см. `view_csv_feature_qa`); `meta.places365_topk_k` **1…32** |

## Валидатор

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_clip/utils/validate_core_clip_npz.py \
  <path/to/embeddings.npz> [--struct] [--qa] [--ranges]
```

Пакетно (struct-эквивалент, без `--qa`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_clip/utils/validate_core_clip_npz.py \
  --results-base /path/to/storage/result_store --platform-id youtube
```

- QA / melt: `view_csv_feature_qa.json` / `view_csv_melt_interesting.json` → **`core_clip`**.

## Схема

[SCHEMA.md](SCHEMA.md), [README.md](../README.md), machine schema: `DataProcessor/VisualProcessor/schemas/core_clip_npz_v2.json`.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
