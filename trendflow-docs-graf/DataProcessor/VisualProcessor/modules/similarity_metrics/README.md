# Модуль `similarity_metrics`

## Описание

Модуль `similarity_metrics` вычисляет метрики схожести для видео:
- **intra-video coherence** (покадровые графики на `core_clip`)
- **reference similarity** (сравнение с reference set из `dp_models`) по нескольким модальностям (visual/audio/text/pacing/quality/emotion).

### Production Policy

- ✅ Модуль является **CONSUMER** компонента `core_clip` (NPZ)
- ✅ Основной режим: **intra-video coherence** (схожесть кадров внутри видео)
- ✅ Опциональный режим: **reference similarity** (если задан `reference_set_id` из `dp_models`)
- ✅ Результаты сохраняются **только в NPZ формате** (без JSON артефактов)
- ✅ Использует данные из `core_clip/embeddings.npz` как **единственный source-of-truth для time‑axis**
- ✅ **Audio optional**: отсутствие `clap_extractor` допустимо, аудио‑модальность помечается как `NaN` и исключается из агрегатов

### Архитектура модуля

Модуль содержит два класса/части:

1. **`SimilarityBaselineModule`** — production baseline v1 (numpy-only, wide baseline)
2. **`SimilarityMetrics`** — library-only (scipy/sklearn), находится в `similarity_metrics_library.py` и **не используется** baseline пайплайном

## Зависимости

### Обязательные зависимости

- **`core_clip`** — компонент, который вычисляет embeddings для ключевых кадров видео и сохраняет результаты в `result_store/.../core_clip/embeddings.npz`

### Опциональные зависимости (модальности)

- `clap_extractor` — audio embeddings (AudioProcessor); если артефакт отсутствует или нет эмбеддинга, аудио‑схожесть маркируется как `NaN` (optional, отсутствие допустимо)
- `shot_quality/shot_quality.npz` (quality/style)
- `video_pacing/*.npz` (pacing features)
- `text_processor/text_features.npz` (TextProcessor; использует `primary_embedding` если `primary_embedding_present=true`)
- `micro_emotion/micro_emotion.npz` (emotion; если нет лиц — это допустимо)
- OCR отсутствие допустимо (в baseline v1 similarity_metrics не требует OCR)

### Требования к входным данным

Модуль ожидает наличие файла:
```
<rs_path>/core_clip/embeddings.npz
```

Файл должен содержать следующие ключи:
- `frame_indices` — массив индексов кадров (int32)
- `frame_embeddings` — массив embeddings кадров (float32, shape: [N, D])

### Требования к согласованности данных

- Segmenter должен обеспечивать согласованную выборку кадров для зависимых компонентов
- **Audit v3 strict**: `similarity_metrics.frame_indices` должны **строго совпадать** с `core_clip.frame_indices` (иначе модуль делает `raise`)

## Использование

### Программный интерфейс

```python
from modules.similarity_metrics.similarity_metrics import SimilarityBaselineModule
from utils.frame_manager import FrameManager

# Инициализация модуля
module = SimilarityBaselineModule(
    rs_path="/path/to/result_store",
    top_n=10,
)

# Обработка кадров
frame_manager = FrameManager(frames_dir="/path/to/frames")
frame_indices = [0, 5, 10, 15, 20]  # Индексы кадров для обработки
config = {
    "top_n": 10,
    "reference_set_id": "niche_ref_v1",
    "ui_topk": 5,
}

# Выполнение обработки
results = module.process(
    frame_manager=frame_manager,
    frame_indices=frame_indices,
    config=config
)
```

### CLI интерфейс

```bash
python -m modules.similarity_metrics.main \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --top-n 10 \
    --reference-set-id niche_ref_v1 \
    --ui-topk 5 \
    --log-level INFO
```

### Reference set (dp_models)

`reference_set_id` должен указывать на пакет:
`dp_models/bundled_models/similarity/reference_sets/<reference_set_id>/`

Пакет содержит `manifest.json` (`schema_version="similarity_reference_pack_v1"`) и `.npy` матрицы эмбеддингов по модальностям.

**Параметры CLI:**
- `--frames-dir` (обязательный) — директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный) — путь к хранилищу результатов (`result_store`)
- `--top-n` (опционально) — количество топ видео для усреднения метрик reference similarity (по умолчанию: `10`)
- `--reference-set-id` (опционально) — reference set id в `dp_models`
- `--ui-topk` (опционально) — top‑K reference videos в `meta.ui_payload`
- `--enable-overall-score` (опционально) — считать общий скор (по умолчанию выключен)
- `--log-level` (опционально) — уровень логирования: `DEBUG`, `INFO`, `WARN`, `ERROR` (по умолчанию: `INFO`)

### Интеграция в пайплайн

Модуль автоматически вызывается через основной пайплайн `VisualProcessor` при наличии конфигурации:

```yaml
modules:
  similarity_metrics:
    enabled: true
    top_n: 10
    reference_set_id: null  # dp_models reference set id (optional)
    ui_topk: 5
```

## Выходные данные

### Формат сохранения

Результаты сохраняются в NPZ формате:
```
<rs_path>/similarity_metrics/results.npz
```

**Version**: 2.0.2  
**Schema**: `similarity_metrics_npz_v3`  
**Artifact filename**: `results.npz`

### Структура выходных данных

```python
{
    "frame_indices": np.ndarray[int32],           # Индексы обработанных кадров
    "times_s": np.ndarray[float32],               # union_timestamps_sec[frame_indices]
    "centroid_sims": np.ndarray[float32],         # Схожесть каждого кадра к центроиду
    "temporal_sim_next": np.ndarray[float32],     # Схожесть соседних кадров
    "reference_present": np.ndarray[bool],        # Флаг наличия reference_set_id
    "feature_names": np.ndarray[object],          # Имена агрегатов (tabular)
    "feature_values": np.ndarray[float32]         # Значения агрегатов (tabular)
}
```

### `meta.ui_payload`

`meta.ui_payload` (schema `similarity_metrics_ui_v1`) содержит:
- графики coherence (через `centroid_sims`, `temporal_sim_next`, `times_s`)
- `topk_refs[]`: top‑K reference videos + per-modality scores
- `text_present`: флаг наличия текстовых фичей
- `audio_required_present`: UI флаг (всегда `True`; audio является optional, если отсутствует → `NaN`)

### Описание полей

#### `frame_indices`
- **Тип**: `np.ndarray[int32]`
- **Описание**: Массив индексов кадров, для которых были вычислены метрики схожести
- **Пример**: `[0, 5, 10, 15, 20]`

#### `centroid_sims`
- **Тип**: `np.ndarray[float32]`
- **Описание**: Покадровая схожесть каждого кадра к центроиду всех кадров (intra-video coherence)
- **Размерность**: Соответствует количеству кадров в `frame_indices`
- **Диапазон**: `[-1.0, 1.0]` (косинусная схожесть)
- **Интерпретация**: Высокие значения означают, что кадр семантически близок к "центру" видео

#### `temporal_sim_next`
- **Тип**: `np.ndarray[float32]`
- **Описание**: Схожесть каждого кадра с предыдущим кадром (временная согласованность)
- **Размерность**: `len(frame_indices) - 1` (первый кадр не имеет предыдущего)
- **Диапазон**: `[-1.0, 1.0]` (косинусная схожесть)
- **Интерпретация**: Высокие значения означают плавные переходы между кадрами

#### `reference_present`
- **Тип**: `np.ndarray[bool]`
- **Описание**: Флаг, указывающий, был ли включен reference set
- **Значение**: `True` если `reference_set_id` был предоставлен, иначе `False`

#### `feature_names` / `feature_values`
- **Тип**: `object[F]` / `float32[F]`
- **Описание**: Агрегированные статистические признаки схожести в tabular-формате.
- **Порядок**: `feature_names` отсортирован лексикографически (stable).

**Содержимое агрегатов (примерно):**

```python
{
    # Intra-video coherence метрики (всегда присутствуют)
    "n_frames": int,                          # Количество обработанных кадров
    "centroid_sim_mean": float,               # Средняя схожесть к центроиду
    "centroid_sim_std": float,                # Стандартное отклонение схожести к центроиду
    "centroid_sim_p10": float,                # 10-й перцентиль схожести к центроиду
    "centroid_sim_p90": float,                # 90-й перцентиль схожести к центроиду
    "temporal_sim_mean": float,               # Средняя схожесть соседних кадров
    "temporal_sim_std": float,                # Стандартное отклонение схожести соседних кадров
    
    # Reference similarity метрики (NaN если reference не предоставлен)
    "reference_similarity_mean_topn": float,  # Средняя схожесть с топ-N референсными видео
    "reference_similarity_max": float,        # Максимальная схожесть с референсными видео
    "reference_similarity_p10": float         # 10-й перцентиль схожести с референсными видео
}
```

**Обработка NaN значений:**
- Если `reference_set_id` не задан, все `reference_similarity_*` метрики будут `float("nan")`
- Если кадров меньше 2, `temporal_sim_mean` и `temporal_sim_std` будут `float("nan")`

### Пример загрузки результатов

```python
import numpy as np

# Загрузка результатов
data = np.load("result_store/.../similarity_metrics/results.npz", allow_pickle=True)

frame_indices = data["frame_indices"]
centroid_sims = data["centroid_sims"]
temporal_sims = data["temporal_sim_next"]
reference_present = data["reference_present"].item()
feat_names = data["feature_names"].astype(object).reshape(-1).tolist()
feat_vals = data["feature_values"].astype(np.float32).reshape(-1).tolist()
features = dict(zip(feat_names, feat_vals))

print(f"Обработано кадров: {len(frame_indices)}")
print(f"Средняя схожесть к центроиду: {features['centroid_sim_mean']:.4f}")
print(f"Средняя временная схожесть: {features['temporal_sim_mean']:.4f}")

if reference_present:
    print(f"Максимальная схожесть с референсами: {features['reference_similarity_max']:.4f}")
else:
    print("Reference embeddings не использовались")
```

## Алгоритм работы

1. **Загрузка данных из `core_clip`**
   - Модуль загружает файл `core_clip/embeddings.npz`
   - Извлекает `frame_indices` и `frame_embeddings`

2. **Сопоставление индексов кадров**
   - Создается маппинг между индексами `core_clip` и запрошенными индексами модуля
   - Проверяется, что все запрошенные индексы присутствуют в `core_clip`

3. **Нормализация embeddings**
   - Все embeddings нормализуются по L2-норме для вычисления косинусной схожести

4. **Вычисление intra-video coherence**
   - Вычисляется центроид всех кадров (среднее нормализованных embeddings)
   - Для каждого кадра вычисляется схожесть с центроидом → `centroid_sims`
   - Для каждой пары соседних кадров вычисляется схожесть → `temporal_sim_next`

5. **Вычисление reference similarity (опционально)**
   - Если задан `reference_set_id`, загружается reference pack из `dp_models`
   - Вычисляется схожесть центроида видео с каждым референсным видео
   - Вычисляются статистики: mean (топ-N), max, p10

6. **Агрегация признаков**
   - Вычисляются статистические признаки: mean, std, p10, p90 для всех метрик
   - Результаты упаковываются в tabular `feature_names/feature_values`

7. **Сохранение результатов**
   - Результаты сохраняются в NPZ формате через `BaseModule.run()`

## Обработка ошибок

### `FileNotFoundError: similarity_metrics | missing core_clip embeddings`
- **Причина**: Отсутствует файл `core_clip/embeddings.npz`
- **Решение**: Убедитесь, что компонент `core_clip` был выполнен перед запуском модуля

### `RuntimeError: similarity_metrics | core_clip does not cover requested frame_indices`
- **Причина**: Индексы кадров модуля не покрываются индексами `core_clip`
- **Решение**: Убедитесь, что Segmenter генерирует согласованную выборку кадров для всех зависимых компонентов

### `FileNotFoundError: similarity_metrics | reference embeddings npz not found`
- **Причина**: Задан `reference_set_id`, но reference pack не найден/невалиден в `dp_models`
- **Решение**: Проверьте путь к файлу или уберите параметр, если reference similarity не требуется

### `RuntimeError: similarity_metrics | reference npz missing video_embeddings/embeddings`
- **Причина**: NPZ файл с reference embeddings не содержит ожидаемых ключей
- **Решение**: Убедитесь, что файл содержит ключ `video_embeddings` или `embeddings`

### `ValueError: similarity_metrics | frame_indices is empty`
- **Причина**: Передан пустой список индексов кадров
- **Решение**: Убедитесь, что `frame_indices` содержит хотя бы один индекс

### `ValueError: similarity_metrics | rs_path is required`
- **Причина**: Не указан путь к хранилищу результатов
- **Решение**: Укажите `rs_path` при инициализации модуля

## Конфигурация

### Параметры модуля

```python
config = {
    "top_n": 10,  # Количество топ видео для усреднения reference similarity
    "reference_set_id": "niche_ref_v1"
}
```

### Параметры инициализации

```python
SimilarityBaselineModule(
    rs_path="/path/to/result_store",
    top_n=10,  # По умолчанию: 10
    reference_set_id=None  # По умолчанию: None (только intra-video coherence)
)
```

## Производительность

- **Время выполнения**: Зависит от количества кадров и размера reference set (обычно < 1 секунды для типичных видео без reference, до нескольких секунд с большим reference set)
- **Память**: Минимальное потребление (только загрузка и обработка embeddings)
- **Зависимости**: Требует предварительного выполнения `core_clip`

## Интерпретация метрик

### Intra-video coherence метрики

- **`centroid_sim_mean`** (высокое значение = хорошо):
  - Показывает, насколько кадры видео семантически согласованы
  - Высокие значения означают, что видео имеет единую тему/стиль
  - Низкие значения могут указывать на разнообразие или несогласованность

- **`centroid_sim_std`** (низкое значение = хорошо):
  - Показывает стабильность схожести кадров к центроиду
  - Низкие значения означают равномерную согласованность
  - Высокие значения могут указывать на резкие переходы

- **`temporal_sim_mean`** (высокое значение = хорошо):
  - Показывает плавность переходов между кадрами
  - Высокие значения означают плавное развитие сюжета
  - Низкие значения могут указывать на резкие смены сцен

### Reference similarity метрики

- **`reference_similarity_max`**:
  - Показывает максимальную схожесть с референсными видео
  - Высокие значения означают, что видео похоже на что-то из референсного набора
  - Низкие значения означают уникальность

- **`reference_similarity_mean_topn`**:
  - Средняя схожесть с топ-N наиболее похожими референсными видео
  - Более устойчивая метрика, чем max
  - Показывает общую близость к референсному набору

## Связанные компоненты

- **`core_clip`** — компонент, вычисляющий embeddings для ключевых кадров видео
- **`high_level_semantic`** — модуль, который может предоставлять video-level embeddings для reference-based сравнения (в будущем)

## Будущие улучшения

Модуль содержит библиотечный класс `SimilarityMetrics`, который реализует расширенный набор метрик схожести:

- Семантическая схожесть (embeddings)
- Тематическое пересечение (topics/concepts)
- Визуальный стиль (цвет, свет, композиция)
- Текст и OCR
- Аудио характеристики
- Эмоции и поведение
- Временной ритм (pacing)

Эти метрики могут быть интегрированы в production-версию модуля в будущем, когда будут доступны соответствующие входные данные из других модулей.

## Примечания

- Модуль является частью архитектуры, где вычисление embeddings вынесено в отдельный компонент `core_clip` для переиспользования
- Все результаты сохраняются только в NPZ формате (без JSON) для совместимости с пайплайном
- Агрегированные признаки (`features`) оптимизированы для использования в табличных данных (tabular-friendly)
- Baseline-версия фокусируется на intra-video coherence, что является важной метрикой качества видео
- Reference similarity является опциональной функциональностью и может быть использована для анализа трендов и уникальности

## Дополнительная документация

Подробное описание всех возможных фичей модуля (включая расширенные метрики из класса `SimilarityMetrics`) доступно в файле `FEATURES_DESCRIPTION.md`.
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
