## Component: `content_domain` (semantic head, v1)

### Назначение

`content_domain` определяет домен контента по кадрам (union-domain):
- игра / аниме / мульт / live-action / screen-recording (и др.)

Это **semantic head** поверх `core_clip`:
- использует `core_clip` frame embeddings
- делает CLIP text-retrieval (prompt ensemble) по небольшой базе доменов
- пишет per-frame top‑K + per-video aggregate (track=1)

### Интеграция с Embedding Service

Компонент **может использовать Embedding Service** для хранения эталонных эмбеддингов доменов контента:

1. **Категория**: `content_domain` (может быть расширена до отдельных категорий)
2. **Модель**: `clip_224` (для быстрой классификации доменов)
3. **Использование**:
   - Хранение эталонных эмбеддингов для каждого домена (game, anime, cartoon, live-action и т.д.)
   - Поиск наиболее похожего домена через Embedding Service (`POST /search`)
   - Добавление новых доменов через API (`POST /objects/add`)

**Примечание**: Текущая реализация использует CLIP text-retrieval через Triton напрямую, но может быть расширена для использования Embedding Service для хранения эталонных embeddings доменов.

**Потенциальное использование**:
```python
# Добавить эталонный домен
POST http://localhost:8001/objects/add
{
    "category": "content_domain",
    "name": "anime",
    "image": <representative_frame>,
    "metadata": {"domain_type": "anime", "prompts": ["anime style", "animated"]}
}

# Определить домен контента
POST http://localhost:8001/search
{
    "category": "content_domain",
    "image": <frame_image>,
    "top_k": 1,
    "similarity_threshold": 0.6
}
```

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_clip.frame_indices` (sampling group, source-of-truth для этого компонента)
  - `union_timestamps_sec`
- `rs_path/core_clip/embeddings.npz`
- offline база доменов: `--domain-db-dir DataProcessor/dp_models/bundled_models/semantics/content_domain/v1`
  - `manifest.json`, `domains.jsonl`, optional `thresholds.json`

### Output (NPZ)

Путь: `rs_path/content_domain/content_domain.npz`

**Artifact filename**: `content_domain.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `content_domain_npz_v2`

Краткий аудит фич / melt / QA: **`docs/FEATURE_DESCRIPTION.md`**, валидатор артефакта: `utils/validate_content_domain.py`.

Ключи (v2):
- `frame_indices (N,) int32` (строго = `core_clip.frame_indices`)
- `times_s (N,) float32`
- `semantic_label_names (A,) str` (`"id:name"`)
- `threshold_per_label_arr (A,) float32` (NaN если нет)
- `track_ids (1,) int32` (=0)
- `track_present_mask (1,) bool`
- `track_topk_ids (1,5) int32`, `track_topk_scores (1,5) float32` (max over time)
- `track_is_confident_top1 (1,) bool`
- `frame_topk_ids (N,5) int32`, `frame_topk_scores (N,5) float32`
- `frame_is_confident_top1 (N,) bool`
- `meta` (object dict) + `meta_json () str`: стандартный meta + `db_*` + thresholds + `models_used`/`model_signature` (включая chaining от `core_clip`)

Схемы:
- Human schema: `SCHEMA.md` (рядом с компонентом)
- Machine schema: `VisualProcessor/schemas/content_domain_npz_v2.json`

**Meta обязательные поля** (baseline contract):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (может быть "unknown" в baseline)
- `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- `models_used[]` (если используются модели)
- `stage_timings_ms` (dict): тайминги стадий выполнения в миллисекундах:
  - `initialization`: инициализация компонента
  - `load_deps`: загрузка зависимостей (`core_clip`, domain db)
  - `process_frames`: вычисление text embeddings и классификация доменов
  - `saving`: сохранение артефакта
  - `total`: общее время выполнения

### No-fallback / empty semantics

- Если нет `core_clip.frame_indices` или `union_timestamps_sec` → **error**.
- Если `core_clip/embeddings.npz` не покрывает все required `frame_indices` → **error**.
- Если domain db отсутствует/битая/пустая → **error (fail-fast)**.

### Progress / state events

Компонент публикует прогресс выполнения в `state_events.jsonl` (baseline contract):

**Стадии выполнения**:
- `start` → `load_deps` → `process_frames` → `save` → `done`

**Гранулярный прогресс**:
- Во время стадии `process_frames` компонент публикует прогресс обработки кадров (≥10 обновлений)
- Формат события: `{"progress": 0.0-1.0, "done": int, "total": int, "stage": "process_frames"}`

**Использование**: Backend сайта может читать `state_events.jsonl` для отображения прогресса анализа в реальном времени.

### Runtime modes

Компонент использует **Triton** для вычисления CLIP text embeddings:

- **Triton mode** (baseline):
  - CLIP text encoder через Triton HTTP API
  - Модель: `clip_text` (Triton model name)
  - Input: tokenized text prompts (INT64)
  - Output: text embeddings (FP32, 512-dim)

**Конфигурация Triton**:
- `clip_text_model_spec`: spec name из ModelManager (например, `clip_text_triton`)
- `triton_http_url`: URL Triton HTTP сервера (или через переменную окружения `TRITON_HTTP_URL`)
- Дефолтные параметры (если не указаны в spec):
  - `triton_model_name`: `"clip_text"`
  - `triton_model_version`: `"1"`
  - `triton_input_name`: `"INPUT__0"`
  - `triton_output_name`: `"OUTPUT__0"`
  - `triton_input_datatype`: `"INT64"`

### Параметры конфигурации

Все параметры принимаются через аргументы командной строки или конфигурацию:

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--domain-db-dir` | str | `dp_models/bundled_models/semantics/content_domain/v1` | Путь к базе доменов (manifest.json, domains.jsonl) |
| `--clip-text-model-spec` | str | `clip_text_triton` | Spec name из ModelManager для CLIP text encoder |
| `--triton-http-url` | str | `None` | Triton HTTP URL (может быть установлен через `TRITON_HTTP_URL` env var) |
| `--topk` | int | `5` | Количество топ доменов для каждого кадра (фиксировано: 5) |
| `--threshold-global` | float | `0.23` | **DEPRECATED** fallback для `--confidence-threshold-top1` |
| `--confidence-threshold-top1` | float | `None` | Порог для `*_is_confident_top1` (не режет top‑K) |
| `--thresholds-json` | str | `None` | Путь к JSON файлу с порогами для каждого домена |

**Конфигурация в `global_config.yaml`**:
```yaml
content_domain:
  domain_db_dir: "dp_models/bundled_models/semantics/content_domain/v1"
  clip_text_model_spec: "clip_text_triton"
  triton_http_url: "http://localhost:8000"  # или через TRITON_HTTP_URL env var
  topk: 5
  threshold_global: 0.23
  thresholds_json: ""
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

## Parallelization

- **Внутренний**: обрабатывает sampled кадры, используя CLIP text embeddings через Triton (batch inference для всех prompts доменов).
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage).

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор frame embeddings из всех видео → группировка в батчи → batch inference через CLIP text encoder (Triton) → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Вычисление text embeddings один раз**: text embeddings для всех доменов вычисляются один раз для всех видео (значительное ускорение)
  - **Векторизованные вычисления**: cosine similarity вычисляется векторизованно для всего батча
  - **Переиспользование Triton клиента**: клиент создается один раз и используется для всех батчей

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **2-5x** (за счет переиспользования text embeddings и лучшего использования GPU)
- Для single video: **1.1-1.2x** (за счет оптимизации вычислений)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Top-K consistency**: top-1 домен должен быть стабильным по времени (без резких скачков)
- **Score distribution**: scores должны быть в разумном диапазоне (обычно 0.2-0.8 для уверенных предсказаний)
- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- **Track aggregate**: per-video track должен соответствовать доминирующему домену по времени

### Human-friendly визуализация (Render System)

`content_domain` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/content_domain/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по доменам (frames_count, domains_count, top1_score_mean/std/min/max, confident_frames_count/ratio)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, top1_domain_id, top1_domain_name, top1_score, is_confident)
- **Distributions**: распределения top1_scores и topk_scores (min, max, mean, std, median, percentiles)
- **Top domains**: топ домены по количеству кадров и среднему score

Render-context может быть использован:
- **LLM** для генерации текстовых описаний домена контента видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions, domain pie charts)
- **Debugging**: быстрая проверка качества классификации доменов без загрузки NPZ

**HTML debug страница** (опционально, dev-only):
- Путь: `result_store/.../content_domain/_render/render.html`
- Offline mini-dashboard (без CDN):
  - Timeline (SVG)
  - Таблица top domains
  - Дистрибуции score’ов

**Конфигурация** (в `global_config.yaml`):
```yaml
content_domain:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

Audit v3: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

### Бенчмарки / resource_costs

Компонент должен иметь unit-cost benchmark (batch=1) и `resource_costs` запись после первого стабильного прогона.

**Типичные характеристики производительности**:
- **Latency per frame**: зависит от количества доменов в базе (обычно 10-50 доменов)
- **GPU VRAM**: минимальное (только для CLIP text encoder inference через Triton)
- **CPU RAM**: ~50-100 MB (для хранения frame embeddings и text embeddings)

**Единица обработки**: `frame` (один кадр с top-K доменами)
---

## Навигация

[VisualProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
