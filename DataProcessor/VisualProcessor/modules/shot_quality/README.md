# `shot_quality` (Audit v3)

Модуль оценивает **техническое качество** видео на уровне:
- **кадров** (frame-level признаки по выборке `frame_indices`)
- **шотов** (shot-level агрегаты поверх результатов `cut_detection`)

Модуль рассчитан на **GPU full‑quality** режим и работает **строго** по контракту выборки кадров: индексы задаёт Segmenter/DataProcessor в `metadata.json`.

---

## Зависимости (обязательные)

Модуль требует, чтобы до него были успешно запущены и сохранили артефакты:

- **`core_clip`** → `rs_path/core_clip/embeddings.npz`
  - обязательно должен содержать:
    - `frame_indices (N,) int32`
    - `frame_embeddings (N, D) float32`
    - `shot_quality_prompts (P,) object`
    - `shot_quality_text_embeddings (P, D) float32`
- **`core_depth_midas`** → `rs_path/core_depth_midas/depth.npz`
  - `frame_indices (N,) int32`
  - `depth_maps (N, H, W) float32`
- **`core_object_detections`** → `rs_path/core_object_detections/detections.npz`
  - `frame_indices (N,) int32`
  - `boxes (N, MAX, 4) float32`, `valid_mask (N, MAX) bool`, `class_ids (N, MAX) int32`
- **`core_face_landmarks`** → `rs_path/core_face_landmarks/landmarks.npz`
  - `frame_indices (N,) int32`
  - `face_landmarks (N, FACES, 468, 3) float32`
  - `face_present (N, FACES) bool`
  - `has_any_face bool`, `empty_reason object` (например `"no_faces_in_video"`)
- **`cut_detection`** → `rs_path/cut_detection/<...>.npz` (через `BaseModule.save_results`)
  - используется для шот-сегментации (hard cuts) и shot-level агрегатов.

**Важно**: никаких fallback. Если зависимость отсутствует/ключи отсутствуют/индексы не совпадают — модуль делает `raise`.

---

## Входы

### 1) Кадры
Через `FrameManager.get(frame_idx)` из `frames_dir`. В проекте `FrameManager` возвращает `HxWx3 uint8` кадр (ожидается **RGB**).

### 2) Выборка кадров (Sampling)
`shot_quality` не выбирает кадры сам.

Segmenter обязан записать в `frames_dir/metadata.json`:

```json
{
  "shot_quality": {
    "frame_indices": [0, 5, 10, 15]
  }
}
```

### Рекомендация по “умной” выборке (для Segmenter)
Цель — одинаковая информативность для видео в диапазоне **120 … 36000** кадров.

Рекомендуемая стратегия (описательная, не реализована в модуле):
- **target_N**: 240–1200 кадров (например, 600 как центр)
- **stratified uniform**: равномерно по времени + обязательные кадры начала/середины/конца
- если есть shot boundaries (на уровне Segmenter) — **per-shot sampling**:
  - минимум 1 кадр на шот
  - плюс дополнительные кадры пропорционально длине шота, но с cap

---

## Выход (NPZ)

Модуль сохраняет **фиксированный** артефакт через `BaseModule.save_results()` в директорию:
`rs_path/shot_quality/`

**Version**: 2.0.2  
**Schema**: `shot_quality_npz_v3`  
**Artifact filename**: `shot_quality.npz`

### Ключи

- **`frame_indices`**: `(N,) int32`
- **`times_s`**: `(N,) float32` — временная ось строго из `union_timestamps_sec[frame_indices]` (no-fallback)
- **`feature_names`**: `(F,) object` — имена признаков в `frame_features`
- **`frame_features`**: `(N, F) float32`
- **`frame_feature_present_ratio`**: `(F,) float32` — доля finite по каждой колонке `frame_features` (помогает моделям интерпретировать NaN)
- **`quality_probs`**: `(N, P) float16` — вероятности zero-shot классов качества (через `core_clip`)
- **`shot_ids`**: `(N,) int32` — принадлежность каждого кадра шоту
- **`shot_start_frame`**: `(S,) int32`
- **`shot_end_frame`**: `(S,) int32`
- **`shot_frame_count`**: `(S,) int32` — число sampled кадров в шоте
- **`shot_features_mean/std/min/max`**: `(S, F) float32` — агрегаты по кадрам шота
- **`shot_frame_feature_present_ratio`**: `(S,F) float32` — доля finite в `frame_features` внутри каждого шота (QA/model)
- **`shot_quality_topk_ids` / `shot_quality_topk_probs`**: `(S,K)` — top‑K quality классы по средним вероятностям внутри шота
- **`shot_quality_conf_mean` / `shot_quality_entropy_mean`**: `(S,)` — shot-level confidence/entropy
- **`impl_meta`**: **внутри `meta.impl_meta`** (debug) — модульный словарь (маппинги категорий, prompts и др.)

**Важно про `meta`:** canonical `meta` всегда добавляется `BaseModule.save_results()` и содержит run identity keys, status/empty_reason и т.д.

### `meta.ui_payload` (для сайта)

Компонент пишет в `meta.ui_payload` JSON payload (schema `shot_quality_ui_v1`) для UI:
- список сцен/шотов (start/end/frame_count) + per-shot top‑K quality labels (ids+probs)
- `frame_indices` + `times_s`
- **графики**: `frame_confidence`, `frame_entropy`
- **распределение**: top‑K классов по средним вероятностям на всём видео (`video_mean_probs_topk_*`)
- флаги наличия лиц и причину пустоты по лицам

Важно: текст промптов **не хранится**. UI должен маппить `class_id` → label по `prompts.version` / `prompts.sha256`.

### “Нет лиц” — это нормально
Если на видео нет лиц, это **валидный OK результат**:
- `core_face_landmarks` может быть `status="empty"` с `empty_reason="no_faces_in_video"`
- `shot_quality` остаётся `meta.status="ok"` (non-face метрики и `quality_probs` вычисляются)
- `face_*` признаки в `frame_features` будут `NaN`

### Human-friendly распаковка

```python
import numpy as np

data = np.load(".../shot_quality/shot_quality.npz", allow_pickle=True)

frame_indices = data["frame_indices"]
feature_names = data["feature_names"].tolist()
X = data["frame_features"]  # (N,F)

# frame-level dict (легко смотреть/логировать)
frames = {
    int(frame_indices[i]): {feature_names[j]: float(X[i, j]) for j in range(X.shape[1])}
    for i in range(len(frame_indices))
}

# shot-level
S = int(data["shot_start_frame"].shape[0])
shot_means = data["shot_features_mean"]
shots = [
    {
        "start_frame": int(data["shot_start_frame"][s]),
        "end_frame": int(data["shot_end_frame"][s]),
        "frame_count": int(data["shot_frame_count"][s]),
        "mean": {feature_names[j]: float(shot_means[s, j]) for j in range(shot_means.shape[1])},
    }
    for s in range(S)
]
```

### Как найти “последний” артефакт

```python
from pathlib import Path
import numpy as np

# фиксированное имя в result_store/.../shot_quality/shot_quality.npz
p = Path(".../result_store/<platform>/<video>/<run_id>/shot_quality/shot_quality.npz")
data = np.load(p, allow_pickle=True)
```

### Проверка NPZ (валидатор)

```bash
cd /path/to/TrendFlowML
export PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor
DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/shot_quality/utils/validate_shot_quality_npz.py \
  storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/shot_quality/shot_quality.npz \
  --struct --qa --ranges
```

Пакетный обход: `--results-base storage/result_store --platform-id youtube` (ищет `shot_quality/shot_quality.npz`). Подробности: [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md).

---

## Фичи (кратко)

Фичи организованы в матрицу `frame_features` и описаны именами в `feature_names`. Сейчас включены группы:
- sharpness / blur
- noise / ISO proxies
- exposure / contrast
- color / cast / fidelity
- compression artifacts
- lens proxies (vignetting, CA, …) — **только preset=quality** (по умолчанию выключены)
- fog/haziness proxy
- temporal (flicker, rolling shutter) — rolling_shutter **только preset=quality** (по умолчанию выключен)
- depth (mean/std/gradient)
- object detections summary
- face ROI quality (по `core_face_landmarks`)

Классы `quality_probs` соответствуют `core_clip["shot_quality_prompts"]` (P = 10).
Для воспроизводимости в `impl_meta` фиксируются:
- `shot_quality_prompts_version`
- `shot_quality_prompts_sha256` (без хранения текста промптов)

---

## GPU / batching

Текущая реализация вычисления `quality_probs` — **numpy‑only (CPU)**. Параметр `device` сохранён как конфигурационное поле (информативно).

Для контроля памяти матмала используется **явный** параметр:
- `matmul_chunk_size` (через config), default `2048` (без эвристик).

## Batch Processing

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео:

- **Batch-safe**: использует per-video rs_path (нет shared mutable state между видео).
- **Дефолтный process_batch()**: последовательная обработка каждого видео через BaseModule.
- **GPU batching**: не требуется (CPU-only модуль).
- **supports_batch**: возвращает `False` (компонент не реализует оптимизированный GPU batching, но поддерживает последовательную обработку батчей через BaseModule).

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): без изменений (компонент работает через subprocess)
- Для single video: без изменений

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными.

## Features / presets

Поддерживаемые пресеты:
- `fast`: выключает entropy-heavy метрики и rolling_shutter/lens
- `default`: включает entropy-heavy, но выключает rolling_shutter/lens
- `quality`: включает rolling_shutter + lens группу

Ключи конфигурации (прокидываются через сайт):
- `preset: fast|default|quality`
- `enable_entropy_features: bool`
- `enable_rolling_shutter: bool`
- `enable_lens_features: bool`
- `analysis_max_dim: int`
- `matmul_chunk_size: int`
- `progress_every_n_frames: int`
- `ui_topk: int` (default: 3)

## Human-friendly визуализация (Render System)

`shot_quality` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/shot_quality/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по shot quality метрикам (frames_count, features_count, shots_count, avg_frame_confidence)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, shot_id, sharpness, quality_confidence)
- **Distributions**: распределения метрик (sharpness_tenengrad, noise_level_luma, quality_confidence) с min, max, mean, std, median, percentiles
- **Shots**: информация о shot boundaries и длительности shots

Render-context может быть использован:
- **LLM** для генерации текстовых описаний качества видео
- **Frontend** для построения графиков и визуализаций (timeline charts с sharpness и quality confidence, distributions метрик, shot boundaries)
- **Debugging**: быстрая проверка качества shot quality метрик без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../shot_quality/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: sharpness и quality confidence по времени
  - Shots summary: таблица со всеми shots (shot_id, start_frame, end_frame, frame_count)
  - Distributions: статистики по метрикам
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
shot_quality:
  preset: "default"
  analysis_max_dim: 320
  feature_flags:
    enable_entropy_features: true
    enable_rolling_shutter: false
    enable_lens_features: false
  matmul_chunk_size: 2048
  progress_every_n_frames: 25
  ui_topk: 3
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).


