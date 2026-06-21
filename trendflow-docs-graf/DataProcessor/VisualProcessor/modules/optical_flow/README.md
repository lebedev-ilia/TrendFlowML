# Модуль `optical_flow`

- **Контракт NPZ, melt/QA, примеры команд:** [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md)
- **Валидатор:** `utils/validate_optical_flow_npz.py` (`--struct`, `--qa`, `--ranges` или батч `--results-base`)

## Описание

Модуль `optical_flow` — это модуль-потребитель (consumer), который обрабатывает данные оптического потока, полученные от компонента `core_optical_flow`. Модуль **не вычисляет RAFT самостоятельно**, а использует предварительно вычисленные данные движения для извлечения агрегированных признаков.

### Production Policy

- ✅ Модуль является **CONSUMER** компонента `core_optical_flow` (NPZ)
- ✅ Модуль **НЕ вычисляет RAFT** самостоятельно
- ✅ Результаты сохраняются **только в NPZ формате** (без JSON артефактов)
- ✅ Использует данные из `core_optical_flow/flow.npz`

## Зависимости

### Обязательные зависимости

- **`core_optical_flow`** — компонент, который вычисляет оптический поток (RAFT) и сохраняет результаты в `result_store/.../core_optical_flow/flow.npz`

### Требования к входным данным

Модуль ожидает наличие файла:
```
<rs_path>/core_optical_flow/flow.npz
```

Файл должен содержать следующие ключи:
- `frame_indices` — массив индексов кадров (int32)
- `motion_norm_per_sec_mean` — кривая движения, нормализованная на секунду (float32)
- `meta` (опционально) — метаданные

### Требования к согласованности данных

- `frame_indices` модуля `optical_flow` должны быть **подмножеством** `core_optical_flow.frame_indices`
- Segmenter должен обеспечивать согласованную выборку кадров для зависимых компонентов
- При несоответствии индексов модуль использует `NaN` для отсутствующих кадров (с предупреждением в логе)
- Статистика корректно обрабатывает NaN значения (игнорирует их при вычислении mean, median, p90, variance)

## Использование

## Sampling / units-of-processing requirements

- **Sampling owner**: Segmenter (источник истины — `metadata.json`).
- Модуль **не генерирует выборку сам** и не делает fallback на “все кадры”.
- Единица обработки: **frame** (значение `motion_norm_per_sec_mean` соответствует кадру из `frame_indices`, но основано на потоке между соседними кадрами).
- Time-axis: `times_s = union_timestamps_sec[frame_indices]`.

### Программный интерфейс

```python
from modules.optical_flow.optical_flow import OpticalFlowModule
from utils.frame_manager import FrameManager

# Инициализация модуля
module = OpticalFlowModule(rs_path="/path/to/result_store")

# Обработка кадров
frame_manager = FrameManager(frames_dir="/path/to/frames")
frame_indices = [0, 5, 10, 15, 20]  # Индексы кадров для обработки
config = {}  # Конфигурация модуля (опционально)

# Выполнение обработки
results = module.process(
    frame_manager=frame_manager,
    frame_indices=frame_indices,
    config=config
)
```

### CLI интерфейс

```bash
python -m modules.optical_flow.main \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --log-level INFO
```

**Параметры CLI:**
- `--frames-dir` (обязательный) — директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный) — путь к хранилищу результатов (`result_store`)
- `--log-level` (опционально) — уровень логирования: `DEBUG`, `INFO`, `WARN`, `ERROR` (по умолчанию: `INFO`)

### Интеграция в пайплайн

Модуль автоматически вызывается через основной пайплайн `VisualProcessor` при наличии конфигурации:

```yaml
modules:
  optical_flow:
    enabled: true
```

## Выходные данные

### Формат сохранения

Результаты сохраняются в NPZ формате:
```
<rs_path>/optical_flow/optical_flow.npz
```

### Структура выходных данных (Audit v3)

```python
{
    "frame_indices": np.ndarray[int32],      # Индексы обработанных кадров
    "times_s": np.ndarray[float32],          # Временная ось: union_timestamps_sec[frame_indices]
    "motion_norm_per_sec_mean": np.ndarray[float32],  # Кривая движения (нормализованная на секунду)
    "frame_feature_names": np.ndarray[object], # Имена per-frame compact фич
    "frame_feature_values": np.ndarray[float32],# Per-frame compact фичи (N,D)
    "feature_names": np.ndarray[object],     # Имена агрегатов (табличный формат)
    "feature_values": np.ndarray[float32]    # Значения агрегатов (табличный формат)
}

# В NPZ meta также сохраняется:
# - ui_payload: dict с данными для фронтенда (schema_version, frame_indices, times_s, motion_norm_per_sec_mean)
# - stage_timings_ms: dict (baseline contract)
```

### Описание полей

#### `frame_indices`
- **Тип**: `np.ndarray[int32]`
- **Описание**: Массив индексов кадров, для которых были вычислены признаки движения
- **Пример**: `[0, 5, 10, 15, 20]`

#### `motion_norm_per_sec_mean`
- **Тип**: `np.ndarray[float32]`
- **Описание**: Покадровая кривая движения, нормализованная на секунду (px/sec)
- **Размерность**: Соответствует количеству кадров в `frame_indices`
- **Примечание**: Первый элемент обычно равен 0.0 (нет предыдущего кадра для сравнения)

#### `feature_names` / `feature_values`
- **Тип**: `object[F]` / `float32[F]`
- **Описание**: табличные агрегаты для baseline моделей/QA.

Текущий фиксированный набор:
- `motion_curve_mean`
- `motion_curve_median`
- `motion_curve_p90`
- `motion_curve_variance`
- `missing_frame_ratio`
- `cam_shake_std_mean`
- `cam_rotation_abs_mean`
- `cam_translation_abs_mean`
- `flow_consistency_mean`

**Обработка NaN значений:**
- Для статистик используются `nan*` функции.
- Первый элемент кривой может быть "особым" (нет предыдущего кадра); статистики считаются по `curve[1:]` (если доступно).

#### Stage timings

Audit v3: тайминги пишутся в `meta.stage_timings_ms`.

#### Метаданные NPZ (`meta`)

NPZ meta содержит baseline identity keys +:
- `ui_payload`: JSON для фронта (внутри NPZ meta, не отдельный JSON-артефакт). Содержит `schema_version`, `frame_indices`, `times_s`, `motion_norm_per_sec_mean` для построения графиков на фронтенде.

### Пример загрузки результатов

```python
import numpy as np

# Загрузка результатов
data = np.load("result_store/.../optical_flow/optical_flow.npz", allow_pickle=True)

frame_indices = data["frame_indices"]
motion_curve = data["motion_norm_per_sec_mean"]
names = data["feature_names"].astype(object).reshape(-1).tolist()
vals = data["feature_values"].astype(np.float32).reshape(-1).tolist()
features = dict(zip(names, vals))

print(f"Обработано кадров: {len(frame_indices)}")
print(f"Среднее движение: {features.get('motion_curve_mean', float('nan')):.2f} px/sec")
print(f"Медианное движение: {features['motion_curve_median']:.2f} px/sec")
print(f"90-й перцентиль: {features['motion_curve_p90']:.2f} px/sec")
```

## Алгоритм работы

1. **Загрузка данных из `core_optical_flow`**
   - Модуль загружает файл `core_optical_flow/flow.npz`
   - Извлекает `frame_indices` и `motion_norm_per_sec_mean`

2. **Сопоставление индексов кадров**
   - Создается маппинг между индексами `core_optical_flow` и запрошенными индексами модуля
   - Для отсутствующих кадров используется `NaN` (с предупреждением в логе)

3. **Извлечение кривой движения**
   - Извлекается подмножество кривой движения для запрошенных кадров
   - Отсутствующие кадры заполняются `NaN` (игнорируются в статистике)

4. **Вычисление агрегатов**
   - Вычисляются статистические признаки: mean, median, p90, variance
   - Результаты упаковываются в табличный формат `feature_names/feature_values`

5. **Сохранение результатов**
   - Результаты сохраняются в NPZ формате через `BaseModule.run()`

## Обработка ошибок

### `FileNotFoundError`
- **Причина**: Отсутствует файл `core_optical_flow/flow.npz`
- **Решение**: Убедитесь, что компонент `core_optical_flow` был выполнен перед запуском модуля

### Предупреждение о несоответствии индексов
- **Причина**: Некоторые индексы кадров модуля отсутствуют в `core_optical_flow.frame_indices`
- **Поведение**: Модуль использует `NaN` для отсутствующих кадров и логирует предупреждение
- **Решение**: Убедитесь, что Segmenter генерирует согласованную выборку кадров для всех зависимых компонентов (рекомендуется, но не обязательно)

### `ValueError: optical_flow | frame_indices is empty`
- **Причина**: Передан пустой список индексов кадров
- **Решение**: Убедитесь, что `frame_indices` содержит хотя бы один индекс

### `ValueError: optical_flow | rs_path is required`
- **Причина**: Не указан путь к хранилищу результатов
- **Решение**: Укажите `rs_path` при инициализации модуля

## Конфигурация

Модуль не требует специальной конфигурации для обработки. Все параметры обработки определяются компонентом `core_optical_flow`.

### Render конфигурация

Модуль поддерживает генерацию render-context JSON и HTML debug страницы:

```yaml
modules:
  optical_flow:
    render:
      enable_render: true  # Генерировать render-context JSON (for LLM/frontend)
      enable_html_render: true  # Генерировать HTML debug страницу
```

**Render-context JSON** содержит:
- **Summary**: статистики по движению (frames_count, motion_curve_mean/median/p90/variance/std/min/max)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, motion_norm_per_sec_mean)
- **Distributions**: распределения motion_norm_per_sec_mean (min, max, mean, std, median, percentiles)

**HTML debug страница** содержит:
- Offline SVG графики (без CDN): timeline движения по времени
- Таблицы со статистиками распределений
- Summary метрики в удобном формате

Путь к render файлам:
- JSON: `<rs_path>/optical_flow/_render/render_context.json`
- HTML: `<rs_path>/optical_flow/_render/render.html`

## Batch Processing

Модуль поддерживает **batch processing** для одновременной обработки нескольких видео:

- **CPU-only операция**: модуль является consumer и не требует GPU
- **Последовательная обработка**: каждое видео обрабатывается последовательно (CPU-bound операция)
- **Изоляция**: каждый видео имеет свой `rs_path` для артефактов

**Конфигурация batch processing** (в `global_config.yaml`):
```yaml
visual:
  batch_processing:
    enabled: true
    enable_gpu_batching: true  # Не используется для optical_flow (CPU-only)
    enable_cpu_parallel: true  # Может использоваться для параллельной обработки видео
```

**Использование**:
- Batch processing автоматически активируется при вызове `run_batch()` в `VisualProcessor/main.py`
- Модуль реализует `supports_batch = True` и метод `process_batch()`

## Производительность

- **Время выполнения**: Зависит от количества кадров (обычно < 1 секунды для типичных видео)
- **Память**: Минимальное потребление (только загрузка и обработка кривой движения)
- **Зависимости**: Требует предварительного выполнения `core_optical_flow`
- **Batch processing**: Поддерживает обработку нескольких видео последовательно

## Связанные компоненты

- **`core_optical_flow`** — компонент, вычисляющий оптический поток (RAFT)
- **`video_pacing`** — модуль, использующий данные движения для анализа темпа видео
- **`cut_detection`** — модуль, использующий данные движения для детекции склеек

## Примечания

- Модуль является частью архитектуры, где вычисление оптического потока вынесено в отдельный компонент `core_optical_flow` для переиспользования
- Все результаты сохраняются только в NPZ формате (без JSON) для совместимости с пайплайном
- Агрегированные признаки (`features`) оптимизированы для использования в табличных данных (tabular-friendly)

## Quality validation & human-friendly inspection

### Human-friendly визуализация (Render System)

`optical_flow` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/optical_flow/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по движению (frames_count, motion_curve_mean, motion_curve_median, motion_curve_p90, motion_curve_variance, motion_curve_std/min/max)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, motion_norm_per_sec_mean)
- **Distributions**: распределения motion_norm_per_sec_mean (min, max, mean, std, median, percentiles)

Render-context может быть использован:
- **LLM** для генерации текстовых описаний движения в видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions)
- **Debugging**: быстрая проверка качества анализа движения без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../optical_flow/_render/render.html`
- Содержит offline графики (SVG, без CDN):
  - Timeline: motion_norm_per_sec_mean по времени
  - Distributions: статистики по motion_norm_per_sec_mean
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
visual:
  modules:
    optical_flow:
      render:
        enable_render: true  # Генерировать render-context JSON
        enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

## История изменений

- **Production policy**: Модуль переведен на использование данных из `core_optical_flow` вместо самостоятельного вычисления RAFT
- **Формат вывода**: Убраны JSON артефакты, используется только NPZ формат
- **Упрощение**: Убраны сложные статистические фичи, оставлены только стабильные агрегаты
- **Обработка отсутствующих кадров**: Модуль теперь использует `NaN` для отсутствующих кадров вместо падения с ошибкой
- **Batch processing**: Добавлена поддержка batch processing для обработки нескольких видео
- **Render система**: Добавлена генерация render-context JSON и HTML debug страницы
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
