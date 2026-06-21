# Action Recognition Module

Модуль распознавания действий в видео на основе архитектуры **SlowFast** (Meta AI Research). Компонент извлекает временные эмбеддинги и агрегированные метрики для анализа действий людей в видео.

**Документация и проверка NPZ:** `docs/FEATURE_DESCRIPTION.md` (ключи, meta, melt/QA), `docs/SCHEMA.md`, `docs/FEATURES_DESCRIPTION.md` · `utils/validate_action_recognition.py` `<npz>` `--struct` / `--qa` (для старого отчёта: `--legacy`).

## 📋 Содержание

- [⚠️ Статус качества и доработки](#️-статус-качества-и-доработки)
- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Зависимости](#зависимости)
- [Установка](#установка)
- [Использование](#использование)
- [Структура выходных данных](#структура-выходных-данных)
- [Параметры конфигурации](#параметры-конфигурации)
- [Технические детали](#технические-детали)
- [Производительность](#производительность)

## ⚠️ Статус качества и доработки

**ВНИМАНИЕ**: Модуль требует **доработки качества** и **более тщательной проверки**.

### Текущий статус

- ✅ **Базовая функциональность**: Модуль успешно проходит smoke-тесты и генерирует корректные NPZ артефакты
- ✅ **Схема и контракты**: Соответствует `action_recognition_npz_v2`, все обязательные поля присутствуют
- ⚠️ **Качество алгоритмов**: Требуется валидация на репрезентативных датасетах
- ⚠️ **Метрики стабильности**: Алгоритмы кластеризации (PCA+KMeans) требуют проверки на различных типах действий
- ⚠️ **Сегментация треков**: Логика группировки person детекций в сегменты требует валидации на edge cases
- ⚠️ **Временная согласованность**: Проверка корректности `temporal_jumps` и `clip_center_times_s` на различных видео

### Результаты smoke-тестов

**Тест 1** (`audit3_action_recognition_smoke_2`, video2.mp4, 58.8s):
- Обработано: 235 кадров детекций
- Найдено person детекций: 33 кадра
- Создано сегментов: 15 треков
- Статус: ✅ Успешно, NPZ артефакт сгенерирован

**Тест 2** (`audit3_action_recognition_smoke_4`, video1.mp4, 28.8s):
- Обработано: 115 кадров детекций
- Найдено person детекций: 26 кадров
- Создано сегментов: 18 треков
- Статус: ✅ Успешно, NPZ артефакт сгенерирован

### Области для доработки

1. **Валидация метрик стабильности**:
   - Проверка корректности `stability` и `stability_centroid_dist` на различных типах действий
   - Валидация порогов для кластеризации (число кластеров K)
   - Проверка на edge cases (очень короткие/длинные треки)

2. **Улучшение сегментации**:
   - Валидация параметров `segment_gap_sec` и `min_person_confidence` на различных видео
   - Проверка корректности группировки кадров с person детекциями
   - Обработка случаев с множественными людьми в кадре

3. **Временная согласованность**:
   - Проверка корректности `temporal_jumps` (должны быть `num_clips-1` элементов)
   - Валидация `clip_center_times_s` относительно `union_timestamps_sec`
   - Проверка согласованности `clip_center_frame_indices` с реальными кадрами

4. **Качество эмбеддингов**:
   - Валидация нормализации (L2) эмбеддингов
   - Проверка проекции 2048d → 256d
   - Анализ распределения эмбеддингов на различных типах действий

5. **Render и визуализация**:
   - Проверка корректности HTML render на различных данных
   - Валидация offline SVG графиков (timeline, stability, distributions)
   - Проверка fallback логики для `metric__*` ключей

### Рекомендации по проверке

1. Запустить на репрезентативном датасете (различные типы видео, длительности, количество людей)
2. Сравнить метрики с ground truth (если доступен)
3. Провести визуальный анализ HTML render для различных видео
4. Проверить edge cases (очень короткие видео, видео без людей, множественные люди)
5. Валидировать согласованность с downstream компонентами

## Обзор

Модуль `action_recognition` использует предобученную модель **SlowFast R50** для анализа действий в видео. SlowFast — это dual-pathway CNN архитектура, специально разработанная для анализа движения и темпа в видео:

- **Slow pathway**: обрабатывает семантическую информацию (каждый 16-й кадр)
- **Fast pathway**: обрабатывает движение и темп (каждый 2-й кадр)

Модуль извлекает два типа признаков:
1. **Sequence Features** — временные последовательности эмбеддингов для VisualTransformer
2. **Aggregate Features** — агрегированные статистики для MLP/Tabular Head

### Основные возможности

- ✅ Распознавание действий на основе SlowFast (ResNet-50)
- ✅ Извлечение нормализованных эмбеддингов (256d) для каждого клипа
- ✅ Вычисление метрик стабильности, разнообразия и временной структуры
- ✅ Поддержка обработки нескольких треков (людей) параллельно
- ✅ Интеграция с BaseModule для единообразия с другими модулями
- ✅ Безопасная обработка ошибок и управление памятью
- ✅ Поддержка GPU и CPU

## Архитектура

### Используемая модель

- **Модель**: `slowfast_r50` из `pytorchvideo.models.hub`
- **Предобучение**: Kinetics-400 (400 классов действий)
- **Входной размер**: 224×224 пикселей
- **Длина клипа**: по умолчанию 32 кадра (минимум для T_slow=8, T_fast=32 для SlowFast)
- **Alpha**: 4 (T_fast / T_slow, по умолчанию)
- **Размерность эмбеддингов**: 
  - Raw: 2048d (извлеченные из модели)
  - Проекция: 256d (для VisualTransformer)

### Обработка данных

1. **Загрузка детекций**: Модуль загружает детекции из `detections.npz` (результаты `core_object_detections`)
2. **Генерация сегментов**: Модуль генерирует сегменты из детекций "person" (class_id=0), группируя последовательные кадры с person детекциями
3. **Создание клипов**: Последовательности кадров разбиваются на перекрывающиеся клипы
4. **Препроцессинг**: Кадры приводятся к 224×224, нормализуются (ImageNet statistics), дополняются до точной длины clip_len и кратности alpha
5. **Извлечение признаков**: SlowFast обрабатывает клипы через slow и fast пути
6. **Проекция**: Эмбеддинги проецируются в 256d и нормализуются (L2)
7. **Агрегация**: Вычисляются метрики для каждого сегмента

## Sampling requirements

Компонент **не генерирует sampling** самостоятельно. `frame_indices` берутся из `frames_dir/metadata.json` (Segmenter‑owned).

### Требования к выборке кадров

- **Единица обработки**: кадр (union‑domain)
- **Источник индексов**: `metadata.json` секция `action_recognition.frame_indices` (Segmenter генерирует)
- **Минимальная длительность видео**: 5 сек
- **Максимальная длительность видео**: 20 мин
- **Требования к `union_timestamps_sec`**: должен быть доступен для временной сегментации (используется для разрыва сегментов по временному порогу)

### Рекомендуемая стратегия sampling (Segmenter‑owned)

Универсальная нелинейная кривая:
- **type**: `ease_out_power`
- **k**: `0.7`
- **min_units**: `120` (минимум кадров для коротких видео)
- **max_units**: `1600` (максимум кадров для длинных видео)
- **linear_until_sec**: `60` (линейная выборка до 60 сек)
- **cap_duration_sec**: `1200` (20 минут, максимальная длительность)

### No-fallback policy

- Если `action_recognition.frame_indices` отсутствует или пустой → **error** (fail-fast)
- Если `core_object_detections` отсутствует → **error** (fail-fast)
- Если нет person детекций → **valid empty** (`status="empty"`, `empty_reason="no_person_detections"`)

## Зависимости

### Внешние зависимости

Модуль требует результаты модуля `core_object_detections`:
- Файл `detections.npz` должен содержать:
  - `frame_indices (N,) int32`: индексы кадров (union-domain)
  - `boxes (N, MAX, 4) float32`: bounding boxes (xyxy)
  - `scores (N, MAX) float32`: confidence scores
  - `class_ids (N, MAX) int32`: идентификаторы классов
  - `valid_mask (N, MAX) bool`: маска валидных детекций
  - `class_names (M,) str`: маппинг "id:name" для классов
  - `meta` или `meta_json`: метаданные (meta_json предпочтительнее для совместимости между виртуальными окружениями)

**Примечание**: Трекинг удален из `core_object_detections`. Модуль генерирует сегменты из детекций "person" (class_id=0), группируя последовательные кадры с person детекциями в сегменты.

Также требуется локальная модель SlowFast, загружаемая через **ModelManager** (`DataProcessor/dp_models`).

## Models

### GPU Models

1. **SlowFast R50** (action recognition)
   - **Triton**: ❌ Нет (in-process)
   - **ModelManager spec name**: `slowfast_r50_action_recognition`
   - **Runtime**: `inprocess`
   - **Engine**: `torch`
   - **Precision**: `fp32`
   - **Device**: `cuda` (если доступно) / `cpu`
   - **Local weights**: `dp_models/bundled_models/visual/action_recognition/slowfast_r50/slowfast_r50.pyth`

### CPU Models

Отсутствуют (используется тот же SlowFast в `cpu` режиме при отсутствии GPU).

### Python зависимости

```python
torch>=1.9.0
pytorchvideo>=0.1.5  # Для slowfast_r50 модели
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
```

### Системные требования

- **GPU**: Рекомендуется (CUDA) для ускорения inference
- **RAM**: Минимум 4GB, рекомендуется 8GB+
- **Диск**: ~500MB для модели (загружается автоматически при первом использовании)

## Установка

Модуль является частью VisualProcessor и не требует отдельной установки. Убедитесь, что все зависимости установлены:

```bash
pip install torch torchvision numpy opencv-python scikit-learn
```

Также необходимо указать путь к локальным моделям:

```bash
export DP_MODELS_ROOT="/abs/path/to/DataProcessor/dp_models"
```

## Использование

### CLI интерфейс

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/results \
    --clip-len 16 \
    --stride 8 \
    --batch-size 8 \
    --embedding-dim 256 \
    --alpha 4 \
    --model-name slowfast_r50_action_recognition \
    --log-level INFO
```

#### Параметры CLI

- `--frames-dir` (обязательный): Директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный): Путь к хранилищу результатов (ResultsStore)
- `--clip-len` (по умолчанию: 32): Длина клипа в кадрах (T_fast для SlowFast, минимум 32 для T_slow=8)
- `--stride` (по умолчанию: clip_len//2): Шаг скользящего окна
- `--batch-size` (по умолчанию: 8): Размер батча для inference
- `--embedding-dim` (по умолчанию: 256): Размерность эмбеддингов
- `--alpha` (по умолчанию: 4): SlowFast alpha (T_fast / T_slow)
- `--model-name` (по умолчанию: slowfast_r50_action_recognition): ModelManager spec
- `--device` (опционально): Устройство `cuda`/`cpu` (если не задано — policy из ModelManager)
- `--log-level` (по умолчанию: INFO): Уровень логирования (DEBUG/INFO/WARN/ERROR)

### Программный интерфейс

```python
from action_recognition_slowfast import SlowFastActionRecognizer
from utils.frame_manager import FrameManager

# Инициализация
recognizer = SlowFastActionRecognizer(
    rs_path="/path/to/results",
    clip_len=32,
    batch_size=8,
    embedding_dim=256,
    alpha=4,
    model_name="slowfast_r50_action_recognition",
    device="cuda"  # или "cpu"
)

# Полный цикл (load metadata + process + save)
saved_path = recognizer.run(
    frames_dir="/path/to/frames",
    config={
        "clip_len": 32,
        "batch_size": 8,
        "embedding_dim": 256,
        "model_name": "slowfast_r50_action_recognition",
    }
)
```

## Структура выходных данных

Результаты сохраняются в формате `.npz` (compressed numpy) и содержат:

### Метаданные (meta)

NPZ `meta` содержит стандартные baseline поля (`producer`, `schema_version`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`, `status`, `empty_reason`, `models_used`, и т.д.), а также:

```python
{
    "total_frames": int,
    "processed_frames": int,
    "clip_len": int,
    "stride": int,
    "batch_size": int,
    "embedding_dim": int,
    "processed_tracks": int,
    "model_name": str,
    "ui_payload": { ... }  # JSON для фронта (см. ниже)
}
```

### Результаты по трекам

Для каждого трека (ключ — `track_id`) возвращается словарь:

#### Sequence Features (для VisualTransformer)

- **`embedding_normed_256d`** (`np.ndarray`, shape: `[num_clips, 256]`)
  - L2-нормализованные эмбеддинги для каждого клипа
  - Используются для обучения временных паттернов в VisualTransformer

#### Aggregate Features (для MLP/Tabular Head)

**Core Dynamics:**
- `max_temporal_jump` (`float`): Максимальный скачок между соседними клипами (L2 по normed)
- `mean_temporal_jump` (`float`): Средний скачок (L2 по normed)
- `stability` (`float`): Стабильность действий (PCA+KMeans, доля самой длинной непрерывной серии одинаковых кластеров, 0-1)
- `stability_centroid_dist` (`float`): Альтернативная метрика стабильности: среднее расстояние до центроида кластера (меньше = более стабильные действия)
- `num_switches` (`int`): Количество переключений между кластерами (из KMeans labels)
- `num_clips` (`int`): Количество клипов для трека
- `track_frame_count` (`int`): Количество кадров в треке

**Дополнительно для UI/диагностики:**
- `clip_center_frame_indices` (`List[int]`)
- `clip_center_times_s` (`List[float]`, если доступен `union_timestamps_sec`)
- `temporal_jumps` (`List[float]`, per‑clip)

### Пример структуры

```python
results = {
    0: {  # track_id
        "embedding_normed_256d": np.array([[0.1, 0.2, ...], ...]),  # [N, 256]
        "max_temporal_jump": 0.45,
        "mean_temporal_jump": 0.21,
        "stability": 0.78,
        "stability_centroid_dist": 0.15,
        "num_switches": 3,
        "num_clips": 12,
        "track_frame_count": 160,
        "clip_center_times_s": [0.5, 1.0, ...],
        "temporal_jumps": [0.0, 0.2, ...],
        "embedding_dim": 256
    },
    1: { ... },  # другой трек
    ...
}
```

Подробное описание всех фичей см. в [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md).

## Features

Компонент извлекает следующие фичи (все включены по умолчанию):

### Sequence Features (per-track, per-clip)

- **`embedding_normed_256d`** (`np.ndarray`, shape: `[num_clips, 256]`)
  - L2-нормализованные эмбеддинги для каждого клипа
  - Единица: per-clip
  - Влияние на стоимость: основная часть стоимости (GPU inference)
  - Используется для: VisualTransformer, временной анализ

### Aggregate Features (per-track)

**Core Dynamics:**
- `max_temporal_jump` (`float`): Максимальный скачок между соседними клипами
- `mean_temporal_jump` (`float`): Средний скачок между соседними клипами
- `stability` (`float`): Стабильность действий (PCA+KMeans, доля самой длинной непрерывной серии одинаковых кластеров, 0-1)
- `stability_centroid_dist` (`float`): Альтернативная метрика стабильности: среднее расстояние до центроида кластера (меньше = более стабильные действия)
- `num_switches` (`int`): Количество переключений между кластерами (из KMeans labels)
- `num_clips` (`int`): Количество клипов для трека
- `track_frame_count` (`int`): Количество кадров в треке

**UI/Diagnostics:**
- `clip_center_frame_indices` (`List[int]`): Индексы центров клипов (union‑domain)
- `clip_center_times_s` (`List[float]`): Времена центров клипов (если доступен `union_timestamps_sec`)
- `temporal_jumps` (`List[float]`): Скачки между соседними клипами (per‑clip)

**Примечание**: Все фичи включены по умолчанию. Управление фичами через конфиг не реализовано (все фичи обязательны для downstream компонентов).

## UI / Presentation payload

Компонент формирует JSON‑payload для фронта и сохраняет его в `meta.ui_payload` внутри NPZ. Это **не отдельный JSON‑артефакт** (соответствует контракту: source‑of‑truth = NPZ).

Минимальный формат (MVP):
- `summary`: число треков/клипов, средняя стабильность, средние jump‑метрики
- `tracks[]`: per‑track метрики + упрощённые временные графики (`clip_center_times_s`, `temporal_jumps`)

## Параметры конфигурации

### Основные параметры

| Параметр | Тип | По умолчанию | Описание | Δ latency | Δ cost |
|----------|-----|--------------|----------|-----------|--------|
| `clip_len` | `int` | `16` | Длина клипа в кадрах (T_fast, рекомендуется минимум 32 для T_slow=8) | +5-10 ms/clip при увеличении до 64 | +20-30% |
| `alpha` | `int` | `4` | SlowFast alpha (T_fast / T_slow) | 0% | 0% |
| `stride` | `int` | `clip_len // 2` | Шаг скользящего окна | -10-15% при уменьшении stride | -10-15% |
| `batch_size` | `int` | `8` | Размер батча для inference | -20-30% при увеличении с 4 до 16 (GPU) | -20-30% |
| `embedding_dim` | `int` | `256` | Размерность эмбеддингов после проекции | +2-3 ms/clip при увеличении до 512 | +5-10% |
| `device` | `str` | `"cuda"` или `"cpu"` | Устройство для обработки | +500-1000% на CPU vs GPU | +500-1000% |
| `seed` | `int` | `42` | Seed для детерминированности | 0% | 0% |

**Источник оценки**: бенчмарки на RTX 3090 (GPU) и Intel i7-9700K (CPU), измерение времени на клип.

### Рекомендации по настройке

- **`clip_len`**: 
  - По умолчанию 16 кадров (в коде)
  - Рекомендуется минимум 32 кадра для SlowFast (T_slow=8, требуется моделью)
  - Увеличение до 64 улучшает качество, но замедляет обработку (~+30% latency)
  - Должно быть кратно `alpha` (по умолчанию 4)
- **`batch_size`**:
  - GPU (8GB): 4-8
  - GPU (16GB+): 8-16
  - CPU: 1-2
  - Увеличение batch_size даёт значительное ускорение на GPU (до ~30% при увеличении с 4 до 16)
- **`stride`**:
  - Меньший stride = больше клипов = более детальный анализ
  - Рекомендуется: `clip_len // 2` для баланса качества и скорости
  - Уменьшение stride увеличивает количество клипов линейно, что пропорционально увеличивает стоимость

### Примеры конфигурации

**Минимальная (быстрая)**:
```yaml
action_recognition:
  clip_len: 32  # Минимум для SlowFast (T_slow=8)
  stride: 16
  batch_size: 16
  embedding_dim: 256
  alpha: 4
  segment_gap_sec: 0.5  # Временной порог для разрыва сегмента
  min_person_confidence: 0.3  # Минимальный confidence для person детекций
```

**Стандартная (рекомендуемая)**:
```yaml
action_recognition:
  clip_len: 32  # Минимум для SlowFast (T_slow=8)
  stride: 16
  batch_size: 8
  embedding_dim: 256
  alpha: 4
  segment_gap_sec: 0.5
  min_person_confidence: 0.3
```

**Качественная (медленная)**:
```yaml
action_recognition:
  clip_len: 64  # Улучшенное качество
  stride: 32
  batch_size: 4
  embedding_dim: 512
  alpha: 4
```

## Технические детали

### Препроцессинг

1. **Resize**: Кадры приводятся к 224×224 (bilinear interpolation)
2. **Нормализация**: ImageNet statistics (`mean=[0.45, 0.45, 0.45]`, `std=[0.225, 0.225, 0.225]`)
3. **Формат**: Преобразование в `[C, T, H, W]` для PyTorch
4. **Паддинг**: Клипы дополняются до точной длины `clip_len` и кратности `alpha` (повторением последнего кадра)
5. **Валидация**: Проверка минимальной длины (T_slow >= 8)

### SlowFast обработка

1. **Slow pathway**: разреженная выборка (каждый alpha-й кадр, T_slow = T_fast / alpha)
2. **Fast pathway**: все кадры (T_fast = clip_len)
3. **Fusion**: Модель объединяет оба пути для извлечения признаков
4. **Пример**: при clip_len=32, alpha=4 → T_slow=8, T_fast=32

### Проекция эмбеддингов

- **Вход**: 2048d (raw embeddings из SlowFast)
- **Проекция**: Linear layer с Xavier initialization
- **Нормализация**: L2 нормализация на устройстве (GPU/CPU)
- **Выход**: 256d нормализованные эмбеддинги

### Агрегация метрик

- **Кластеризация**: KMeans на PCA-редуцированных эмбеддингах (для стабильности)
- **Временные метрики**: Вычисляются на основе нормализованных эмбеддингов
- **Обработка ошибок**: ошибки модели не скрываются (no‑fallback); для недостаточного числа клипов метрики могут быть `NaN`

### Управление памятью

- Автоматическая очистка GPU кэша после каждого батча
- Обработка ошибок с освобождением ресурсов
- Гарантированное закрытие FrameManager в `finally` блоках

## Parallelization

### Внутренний параллелизм

- **Батчинг**: обрабатывает клипы батчами размера `batch_size` (контролируется через конфиг)
- **GPU inference**: использует PyTorch для параллельной обработки батчей на GPU
- **Thread-safety**: компонент не thread-safe (не предназначен для параллельного использования в одном процессе)

### Внешний параллелизм

- **Per-video**: можно запускать несколько экземпляров параллельно на разных видео (разные `run_id`)
- **Per-run isolation**: каждый `run_id` имеет изолированное хранилище результатов
- **Ограничения**: 
  - Зависит от доступной VRAM/CPU
  - Рекомендуется ограничивать количество параллельных GPU‑задач оркестратором (1 компонент на GPU)
  - При параллельном выполнении на CPU ограничение по количеству процессов зависит от доступной RAM

### Комбинированный подход

- Внутренний батчинг (batch_size=8-16) + внешний запуск на разных GPU (по одному компоненту на GPU)
- Для CPU: можно запускать несколько процессов параллельно, но каждый процесс будет использовать 1-2 CPU потока

## Performance characteristics

**Источник данных**: `docs/models_docs/resource_costs/action_recognition_costs_v1.json`  
**Единица обработки**: `clip` (32 кадра по умолчанию, минимум для SlowFast)

### Resource costs (measured)

**Типичные значения (preset="default", GPU, batch_size=8)**:

| Resolution | Latency per clip | CPU RAM peak | GPU VRAM peak | Notes |
|------------|------------------|--------------|---------------|-------|
| 1920x1080 | ~15-25 ms | ~1-2 GB | ~2-4 GB | typical |
| 1280x720 | ~12-20 ms | ~0.8-1.5 GB | ~1.5-3 GB | typical |

**Для видео с N клипами**: Total latency ≈ N × latency_per_clip

**Полные данные**: см. `docs/models_docs/resource_costs/action_recognition_costs_v1.json`

### Бенчмарки (приблизительные)

На GPU (NVIDIA RTX 3090, 24GB):
- **Скорость**: ~50-100 клипов/сек (зависит от batch_size)
- **Память**: ~2-4GB для batch_size=8

На CPU (Intel i7-9700K):
- **Скорость**: ~5-10 клипов/сек
- **Память**: ~1-2GB

### Оптимизация

1. **Используйте GPU**: Критично для приемлемой скорости
2. **Увеличьте batch_size**: До предела памяти GPU
3. **Увеличьте stride**: Для длинных треков (меньше клипов)
4. **Обрабатывайте треки параллельно**: На уровне пайплайна (не в модуле)

### Ограничения

- Модуль требует результаты `core_object_detections` (детекции объектов)
- Трекинг удален из `core_object_detections`; модуль генерирует сегменты из детекций "person"
- Короткие сегменты (< clip_len кадров) дополняются последним кадром
- Обработка очень длинных сегментов может быть медленной (рассмотрите увеличение stride)
- Минимальная длина клипа: 32 кадра (для T_slow=8)

## Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: обработка каждого видео через subprocess с оптимизацией конфигурации
- **Оптимизации производительности**:
  - Переиспользование конфигурации для всех видео в батче
  - Параллельная обработка видео (если включено в конфигурации)
  - Оптимизация передачи параметров через CLI

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **1.5-3x** (за счет оптимизации конфигурации и параллельной обработки)
- Для single video: без изменений (компонент работает через subprocess)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

## Human-friendly визуализация (Render System)

`action_recognition` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/action_recognition/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по распознаванию действий (tracks_count, total_clips, avg_stability, avg_stability_centroid_dist, avg_max_temporal_jump, avg_mean_temporal_jump)
- **Timeline**: данные по каждому клипу (track_id, clip_index, clip_center_time_s, temporal_jump, embedding_norm, frame_idx)
- **Distributions**: распределения метрик (stability, stability_centroid_dist, max_temporal_jump, mean_temporal_jump, embedding_norm) с min, max, mean, std, median, percentiles
- **Tracks**: информация о каждом треке с метриками и временными данными
- **Track stability**: per-track stability значения для визуализации
- **Top jumps**: Top 10 клипов с наибольшими temporal jumps (для примеров)

Render-context может быть использован:
- **LLM** для генерации текстовых описаний действий в видео
- **Frontend** для построения графиков и визуализаций (timeline charts с temporal jumps, distributions метрик, track summaries)
- **Debugging**: быстрая проверка качества распознавания действий без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../action_recognition/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - **Timeline**: embedding norms по времени для каждого трека (multi-track line chart)
  - **Stability by Track**: bar chart со stability для каждого трека
  - **Top 10 Clips with Highest Temporal Jumps**: таблица с примерами клипов с наибольшими скачками (track_id, clip_index, time, jump, frame_idx)
  - **Distributions**: статистики по метрикам (stability, stability_centroid_dist, max_temporal_jump, embedding_norm_mean) с min, max, mean, std, median
  - **Summary metrics**: ключевые показатели в удобном формате (num_tracks, total_clips, avg_stability, avg_stability_centroid_dist, avg_max_temporal_jump, avg_mean_temporal_jump)

**Конфигурация** (в `global_config.yaml`):
```yaml
action_recognition:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

## Quality validation & human-friendly inspection

### ⚠️ Текущий статус валидации

**ВНИМАНИЕ**: Модуль требует **доработки качества** и **более тщательной проверки**. Smoke-тесты пройдены успешно, но полная валидация на репрезентативных датасетах еще не проведена.

### Как проверить качество выхода компонента

#### 1. Автоматическая оценка (если доступна)
- ✅ Запустить компонент на эталонных видео (short/long, single/multi person) — **требуется**
- ⚠️ Сверить распределения `max_temporal_jump`, `stability`, `num_switches` — **требуется валидация**
- ⚠️ Проверить корректность метрик на различных типах действий — **требуется**

#### 2. Human-friendly визуализация
- ✅ Построить графики `temporal_jumps` по времени (`clip_center_times_s`) — доступно через HTML render
- ✅ Показать top‑K клипы с наибольшим `max_temporal_jump` — реализовано в HTML render
- ⚠️ Демо‑скрипт: `VisualProcessor/modules/action_recognition/quality_report/demo_action_recognition_quality.py` — **требуется проверка**
- ✅ Использовать HTML render страницу для интерактивного просмотра результатов — доступно

#### 3. Статистическая валидация
- ⚠️ `0.0 <= stability <= 1.0` (или `NaN` если недостаточно клипов) — **требуется проверка на различных данных**
- ⚠️ `max_temporal_jump` не должен быть NaN при `num_clips >= 2` — **требуется валидация**
- ⚠️ `num_clips` согласован с `clip_len/stride` — **требуется проверка**
- ⚠️ `temporal_jumps` содержит `num_clips-1` элементов — **требуется валидация**

#### 4. Интеграция с downstream модулями
- ⚠️ Убедиться, что downstream читают `embedding_normed_256d` — **требуется проверка**
- ⚠️ Проверить корректность временной оси (clip_center_times_s) — **требуется валидация**

#### 5. Валидация алгоритмов (критично)
- ⚠️ **Кластеризация (PCA+KMeans)**: Проверить корректность выбора числа кластеров K и валидацию на различных типах действий
- ⚠️ **Сегментация треков**: Валидация логики группировки person детекций на edge cases
- ⚠️ **Временная согласованность**: Проверка корректности `temporal_jumps` и `clip_center_times_s` на различных видео
- ⚠️ **Нормализация эмбеддингов**: Валидация L2 нормализации и проекции 2048d → 256d

### Команды для теста

```bash
export DP_MODELS_ROOT="/abs/path/to/DataProcessor/dp_models"

# 1) Запуск action_recognition (после Segmenter + core_object_detections)
python VisualProcessor/modules/action_recognition/main.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --clip-len 32 \
  --alpha 4 \
  --batch-size 8 \
  --model-name slowfast_r50_action_recognition

# 2) Генерация HTML отчета
python VisualProcessor/modules/action_recognition/quality_report/demo_action_recognition_quality.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --out-html /path/to/action_recognition_quality.html
```

**Статус**: 
- ✅ Smoke-тесты пройдены (см. результаты выше)
- ⚠️ Полная валидация качества на репрезентативных датасетах **требуется**
- ⚠️ Визуальная проверка HTML render на различных видео **рекомендуется**

## Примеры использования

### Базовый пример

```python
from action_recognition_slowfast import SlowFastActionRecognizer

recognizer = SlowFastActionRecognizer(
    rs_path="./results",
    clip_len=32,
    alpha=4,
    batch_size=8
)

# Загрузка и обработка
metadata = recognizer.load_metadata("./frames")
frame_indices = recognizer.get_frame_indices(metadata)
frame_manager = recognizer.create_frame_manager("./frames", metadata)

results = recognizer.process(
    frame_manager=frame_manager,
    frame_indices=frame_indices
)

# Сохранение
recognizer.save_results(
    results=results,
    metadata={"total_frames": len(frame_indices)}
)
```

### Настройка параметров

```python
recognizer = SlowFastActionRecognizer(
    rs_path="./results",
    clip_len=32,        # Более длинные клипы
    stride=16,          # Меньший шаг = больше клипов
    batch_size=16,     # Больший батч (требует больше памяти)
    embedding_dim=512, # Большая размерность
    device="cuda",
    seed=123
)
```

## Обработка ошибок

- **Отсутствие person детекций**: результат помечается как `status="empty"` с `empty_reason="no_person_detections"`
- **Ошибки модели / отсутствие весов**: модуль падает (no‑fallback)
- **Некорректные данные**: выбрасывает `RuntimeError`/`ValueError` с описанием проблемы

## Логирование

Модуль использует стандартный logger из `utils.logger`:

```python
from utils.logger import get_logger

logger = get_logger("action_recognition")
```

Уровни логирования:
- **DEBUG**: Детальная информация о каждом батче
- **INFO**: Общая информация о процессе
- **WARN**: Предупреждения (пропущенные треки, ошибки)
- **ERROR**: Критические ошибки

## Интеграция с пайплайном

Модуль интегрирован с VisualProcessor через `BaseModule`:

- Автоматическая загрузка зависимостей через `load_core_provider()`
- Стандартизированное сохранение результатов через `save_results()`
- Единый интерфейс `process()` для всех модулей

## Дополнительная документация

- [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md) — подробное описание всех извлекаемых фичей
- [BaseModule documentation](../../base_module.py) — базовый класс для модулей

## Лицензия

Модуль использует предобученную модель SlowFast из PyTorch, которая распространяется под лицензией BSD.

## Авторы

Компонент разработан как часть проекта TrendFlowML для анализа видео контента.
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
