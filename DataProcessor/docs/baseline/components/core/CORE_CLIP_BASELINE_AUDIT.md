# Аудит соответствия core_clip требованиям baseline

**Дата проверки**: 2026-01-14  
**Компонент**: `core_clip` (Core provider, Tier‑0 baseline)  
**Расположение**: `VisualProcessor/core/model_process/core_clip/`  
**Runtime (prod)**: `triton` (GPU-only)  
**Статус аудита**: ✅ **CLOSED** (2026-01-14)

## Резюме

`core_clip` — базовый provider CLIP эмбеддингов для sampled кадров (image encoder) + фиксированных prompt-наборов (text encoder), чтобы downstream модули могли использовать CLIP без загрузки весов и без сети.

Текущее состояние: компонент приведён к baseline‑контракту (no-fallback sampling, atomic save, runtime validation, обязательный `dataprocessor_version`, сохранение `times_s`, batching loop fix, text path batching в Triton).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **8/10**
- **Логика алгоритмов**: **9/10**
- **Логика глобального взаимодействия**: **8/10**
- **Оптимизации (параллелизм, батчинг)**: **8/10**

## ✅ Соответствие требованиям

### 1. Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **строго** читает `frame_indices` из `frames_dir/metadata.json[core_clip.frame_indices]`
- При отсутствии/пустоте `frame_indices` → **fail-fast** (no‑fallback)
- `frame_indices` — union-domain (контракт Segmenter)

### 2. Time axis → `times_s`

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `embeddings.npz` выполняется атомарно (tmp → `os.replace`)
- После сохранения выполняется `artifact_validator.validate_npz()`
- При провале валидации — файл удаляется и компонент падает (fail‑fast)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"` (dev/default)
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна)

### 5. Batch size (scheduler-controlled)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `batch_size` задаётся извне (верхний scheduler/DynamicBatching), auto-batching внутри компонента запрещён
- Исправлен баг в батчинг‑цикле: шаг увеличивается на `len(batch_ids)` (без пропусков кадров)

### 6. Triton runtime: image + text inference

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- В режиме `runtime=triton` **и image, и text** эмбеддинги считаются через Triton
- Text путь оптимизирован: все prompts отправляются в Triton одним батчем (лучше использует dynamic batching)

### 7. NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `embeddings.npz` содержит:
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `frame_embeddings (N,D) float32`
- `{group}_prompts (P,) object` + `{group}_text_embeddings (P,D) float32` для:
  - `shot_quality_*`
  - `scene_aesthetic_*`
  - `scene_luxury_*`
  - `scene_atmosphere_*`
  - `cut_detection_transition_*`
- `popularity_topic_prompts (Pp,) object` + `popularity_topic_text_embeddings (Pp,D) float32` (coarse topics for popularity heads)
- `meta` (dict, object-array)

### 8. Stage timings + progress (state_events)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент измеряет время ключевых стадий через `timings` и сохраняет их в `meta.stage_timings_ms` (миллисекунды).
- Гарантировано присутствуют, как минимум, тайминги:
  - `initialization`, `image_embeddings_total`, `text_embeddings_prep`, `text_inference`, `saving`, `total`
- Компонент пишет прогресс в `state_events.jsonl` cо стадиями:
  - `start → load_deps → process_frames → post_process → save → done`
- Для `process_frames` отправляется **гранулярный** прогресс (не менее ~10–15 обновлений на видео):
  - `progress ∈ [0,1]`, `done`, `total` (кол-во обработанных кадров)

## 📊 Performance / resource costs (baseline unit-cost)

Источник: `docs/models_docs/resource_costs/core_clip_costs_v1.json` (unit-cost, `model-batch-size=1`)  
Evidence: `storage/reports/out/checklist-clip-b1/` (B=1), `storage/reports/out/checklist-clip-b8/` (B=8).

Примечание про VRAM:
- В baseline мы фиксируем VRAM **по процессу `tritonserver`** (`vram_triton_*`) как delta/peak, а не «полный VRAM GPU».

## 🔍 Quality validation (минимальный набор)

- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- Sanity на эмбеддинги: L2-нормы (ожидаемо близко к 1.0 при нормализации в модели/постпроцессе)
- Cosine-sim sanity: одинаковые кадры/соседние кадры должны иметь более высокий cosine similarity, чем разные сцены
- Human-friendly demo (HTML): `scripts/baseline/demo_core_clip_quality.py`

## Вопросы / открытые решения

1. **Владение prompt-наборами**: оставляем prompts внутри `core_clip` (как “source-of-truth”) или переносим в отдельный общий модуль/конфиг baseline?
2. **Production `dataprocessor_version`**: откуда именно его берём (env/manifest/commit tag) и кто гарантирует, что не `"unknown"`?
3. **Spikes для `clip_image_336`/`clip_text`**: приемлемо ли, или хотим менять параметры измерения/прогрева/серий (или включать restart policy на 6GB)?

## Update: prompts v2 + popularity topics (2026-01-14)

Изменение:
- Обновлены prompt-наборы для более качественных/конкретных формулировок.
- Добавлен небольшой универсальный набор `popularity_topic_*` (спорт/путешествия/еда/…): как coarse‑signals для задачи “предсказать популярность”.
- В `meta` добавлено поле `prompts_version` для воспроизводимости.

DEFERRED:
- “Качество prompts” (coverage по категориям, отсутствие bias, устойчивость) — отдельная итерация с человеческой проверкой и/или оффлайн-оценкой. Сейчас фиксируем только текущую версию и совместимость контрактов.

## Update: duration-based sampling budget (2026-01-14)

Наблюдение (по демо-прогону на коротком видео): primary sampling group мог получаться слишком плотным (например, сотни кадров на ~30 сек).

Изменение:
- В `Segmenter` для primary visual sampling group добавлен **duration-based budget** (cap по длительности), чтобы выборка была **легче** на длинных видео и не была близка к 1:1 на коротких.
- Добавлено логирование: `total_frames_source`, `fps`, `duration_s`, `requested_max`, `rate_fps`, `budget_n`, `chosen_n`.

Файл: `Segmenter/segmenter.py` (`_apply_primary_visual_sampling_group`).

Политика по шагу (целевые значения):
- Используем **непрерывную** кривую `target_gap_sec = f(duration_s)` (без ступенек).
- Ориентиры (приблизительно): 5min→1s, 10min→2s, 20min→3–4s (≈3.5s), далее плавное разрежение.

### Result: 20‑minute demo run (evidence)

Видео: `-F71yZij1Uc`  
Длительность (оценка): ~**642s** (≈ 10.7 min), `fps=25`, `approx_frame_count=16057`

Фактическая выборка (после budget):
- `frames_dir.total_frames` (union saved): **951**
- `core_clip.frame_indices` (N): **161** (≈ 0.25 fps по времени, шаг ~4.0s)

Ключевые sanity‑метрики:
- `times_s` согласован с `union_timestamps_sec[frame_indices]` (max abs error ≈ **2.7e‑5s**)
- эмбеддинги L2‑нормированы (норма ≈ **1.0**)


