# Color & Light Analysis Module

Модуль для комплексного анализа цвета и освещения видео. Извлекает покадровые (frame-level), сценовые (scene-level) и видеоуровневые (video-level) признаки для анализа визуального стиля, цветокоррекции и качества освещения.

## 📋 Содержание

- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Зависимости](#зависимости)
- [Установка](#установка)
- [Использование](#использование)
- [Структура выходных данных](#структура-выходных-данных)
- [Параметры конфигурации](#параметры-конфигурации)
- [Извлекаемые признаки](#извлекаемые-признаки)
- [Технические детали](#технические-детали)
- [Sampling / units-of-processing requirements](#sampling--units-of-processing-requirements)
- [Models](#models)
- [Parallelization](#parallelization)
- [Performance characteristics](#performance-characteristics)
- [Quality validation & human-friendly inspection](#quality-validation--human-friendly-inspection)
- [Presentation output (для UI)](#presentation-output-для-ui)

## Обзор

Модуль `color_light` выполняет глубокий анализ цветовых характеристик и освещения видео на трех уровнях:

1. **Frame-level** — компактные нормализованные признаки для каждого кадра (для VisualTransformer)
2. **Scene-level** — агрегированные признаки по сценам с временными паттернами
3. **Video-level** — глобальные метрики стиля, эстетики и цветокоррекции

### Основные возможности

- ✅ Анализ цвета в HSV и Lab цветовых пространствах
- ✅ Извлечение доминантных цветов через KMeans кластеризацию
- ✅ Оценка освещения: контраст, яркость, равномерность, виньетирование
- ✅ Определение характеристик источников света (мягкий/жёсткий, количество)
- ✅ Вычисление стилей цветокоррекции (Teal & Orange, Film, Vintage, TikTok)
- ✅ Эстетические оценки (NIMA/LAION — план интеграции, сейчас значения будут NaN при отсутствии моделей)
- ✅ Временной анализ: стабильность цвета, периодичность, вспышки
- ✅ Интеграция с BaseModule для единообразия с другими модулями

## Архитектура

### Обработка данных

1. **Загрузка сцен**: Модуль загружает информацию о сценах из `scene_classification`
2. **Выборка кадров**: строго используются `frame_indices`, выданные Segmenter (модуль не пересэмплирует)
3. **Извлечение признаков**: Для каждого кадра вычисляются:
   - RGB статистики (mean/std)
   - HSV признаки (hue, saturation, value)
   - Lab признаки (яркость, контраст, баланс тёплых/холодных тонов)
   - Палитра и доминантные цвета
   - Освещение и контраст
   - Характеристики источников света
4. **Агрегация**: Признаки агрегируются на уровне сцен и всего видео
5. **Формирование последовательностей**: Создаются компактные векторы для VisualTransformer

### Используемые алгоритмы

- **KMeans кластеризация** (scikit-learn) для доминантных цветов в Lab пространстве
- **Гистограммный анализ** для энтропии и распределений
- **Автокорреляция** для определения периодичности цветовых паттернов
- **Gradient-based анализ** для оценки направления света
- **Laplacian variance** для определения мягкости/жёсткости света

## Зависимости

### Внешние зависимости

Модуль требует результаты модуля `scene_classification`:
- Файл результатов должен содержать словарь `scenes` с информацией о сценах
- Каждая сцена должна иметь `indices` (список индексов кадров) и `scene_label`

### Python зависимости

```python
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

### Системные требования

- **RAM**: Минимум 2GB, рекомендуется 4GB+
- **CPU**: Многопоточная обработка не требуется, но рекомендуется многоядерный процессор
- **Диск**: Минимальные требования (только для сохранения результатов)

## Установка

Модуль является частью VisualProcessor и не требует отдельной установки. Убедитесь, что все зависимости установлены:

```bash
pip install numpy opencv-python scipy scikit-learn
```

Или используйте файл requirements:

```bash
pip install -r requirements.txt
```

## Использование

### CLI интерфейс

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/results \
    --max-frames-per-scene 350 \
    --stride 5 \
    --log-level INFO \
    --presentation-dir /path/to/VisualProcessor/presentation
```

#### Параметры CLI

- `--frames-dir` (обязательный): Директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный): Путь к хранилищу результатов (ResultsStore)
- `--max-frames-per-scene` (по умолчанию: 350): deprecated (sampling контролируется Segmenter)
- `--stride` (по умолчанию: 5): deprecated (sampling контролируется Segmenter)
- `--log-level` (по умолчанию: INFO): Уровень логирования (DEBUG/INFO/WARN/ERROR)
- `--presentation-dir` (опционально): папка для UI‑presentation JSON/HTML (вне result_store)

### Программный интерфейс

```python
from modules.color_light.processor import ColorLightProcessor
from utils.frame_manager import FrameManager

# Инициализация
processor = ColorLightProcessor(
    rs_path="/path/to/results",
    max_frames_per_scene=350,
    stride=5
)

# Полный запуск (с валидацией meta)
saved_path = processor.run(
    frames_dir="/path/to/frames",
    config={}
)
```

## Структура выходных данных

Результаты сохраняются в формате NPZ через `BaseModule.save_results()`. Структура данных:

```python
{
    "frames": {
        "scene_key": {  # scene_key = "{scene_label}__{scene_id}"
            frame_idx: {
                "frame_idx": int,
                "features": {
                    # Frame-level признаки (см. раздел "Извлекаемые признаки")
                }
            }
        }
    },
    "scenes": {
        "scene_key": {
            # Scene-level признаки
            "scene_label": str,
            "scene_id": str,
            "num_frames": int,
            "num_frames_norm": float,
            # Агрегированные покадровые признаки
            "{feature}_mean": float,
            "{feature}_std": float,
            # Временные паттерны
            "brightness_change_speed": float,
            "scene_flicker_intensity": float,
            "color_change_speed": float,
            # ... и другие
        }
    },
    "video_features": {
        # Video-level признаки
        # Агрегаты по сценам
        "{feature}_mean": float,
        "{feature}_std": float,
        "{feature}_min": float,
        "{feature}_max": float,
        # Стили цветокоррекции
        "style_teal_orange_prob": float,
        "style_film_prob": float,
        # ... и другие
    },
    "sequence_inputs": {
        "frames": [[...]],  # N x D_frame (компактные нормализованные векторы)
        "scenes": [[...]],  # N_scenes x D_scene
        "global": [...]     # D_global
    },
    "frame_indices": [...],          # int32, отсортированные, уникальные (union domain)
    "times_s": [...],                # float32, union_timestamps_sec[frame_indices]
    "sequence_frame_indices": [...], # int32, порядок соответствует sequence_inputs["frames"]
    "sequence_times_s": [...]        # float32, union_timestamps_sec[sequence_frame_indices]
}
```

### Компактный вектор кадра

Каждый кадр в `sequence_inputs["frames"]` представлен вектором из ~16-18 нормализованных признаков (все значения в диапазоне [0, 1]):

1. `hue_mean_norm` — средний hue (нормированный)
2. `hue_std_norm` — стандартное отклонение hue
3. `hue_entropy_weighted` — энтропия hue, взвешенная по насыщенности
4. `sat_mean_norm` — средняя насыщенность
5. `val_mean_norm` — средняя яркость Value
6. `L_mean_norm` — средняя яркость Lab
7. `global_contrast_norm` — глобальный контраст
8. `local_contrast_mean_norm` — локальный контраст
9. `colorfulness_norm` — индекс цветности
10. `skin_tone_ratio` — доля пикселей кожи
11. `overexposed_ratio` — доля переэкспонированных пикселей
12. `underexposed_ratio` — доля недоэкспонированных пикселей
13. `vignetting_score_norm` — оценка виньетирования
14. `soft_light_prob` — вероятность мягкого света
15. `dominant_lab_a_norm` — доминантный цвет (a-канал Lab)
16. `dominant_lab_b_norm` — доминантный цвет (b-канал Lab)

## Параметры конфигурации

### Параметры инициализации

- `rs_path` (str, обязательный): Путь к хранилищу результатов
**Важно:** sampling контролируется Segmenter. Параметры ниже **оставлены для обратной совместимости**, но не влияют на выборку кадров.

- `max_frames_per_scene` (int, по умолчанию: 350): deprecated
- `stride` (int, по умолчанию: 5): deprecated

### Рекомендации по выбору параметров

Рекомендации по sampling формируются на уровне Segmenter (policy).

## Извлекаемые признаки

### Frame-level признаки

#### Цвет в HSV
- `hue_mean`, `hue_std` — статистики hue (0-180)
- `hue_mean_norm`, `hue_std_norm` — нормализованные версии (0-1)
- `hue_entropy` — энтропия распределения hue (36 бинов)
- `hue_entropy_weighted` — энтропия, взвешенная по насыщенности
- `saturation_mean`, `saturation_std` — статистики насыщенности
- `sat_mean_norm` — нормализованная насыщенность
- `value_mean`, `value_std` — статистики яркости
- `val_mean_norm` — нормализованная яркость

#### Цвет в Lab
- `L_mean` — средняя яркость (0-255)
- `L_contrast` — стандартное отклонение L-канала
- `ab_balance` — баланс тёплых/холодных тонов
- `L_mean_norm` — нормализованная яркость

#### Палитра и доминантные цвета
- `dominant_lab_a`, `dominant_lab_b` — координаты доминантного цвета в Lab
- `dominant_lab_a_norm`, `dominant_lab_b_norm` — нормализованные координаты
- `colorfulness_index` — индекс цветности (rg/yb)
- `colorfulness_norm` — нормализованный индекс цветности
- `warm_vs_cold_ratio` — отношение тёплых к холодным тонам
- `skin_tone_ratio` — доля пикселей кожи
- `color_palette_entropy` — энтропия палитры
- `color_harmony_complementary_prob` — вероятность комплементарной гармонии
- `color_harmony_analogous_prob` — вероятность аналогичной гармонии

#### Освещение и контраст
- `brightness_mean`, `brightness_std` — статистики яркости
- `global_contrast` — RMS-контраст (std по яркости)
- `global_contrast_norm` — нормализованный контраст
- `local_contrast`, `local_contrast_std` — локальный контраст по окнам
- `local_contrast_mean_norm` — нормализованный локальный контраст
- `brightness_entropy`, `contrast_entropy` — энтропия гистограмм
- `dynamic_range_db` — динамический диапазон в децибелах
- `overexposed_pixels`, `underexposed_pixels` — доли пере/недоэкспонированных пикселей
- `overexposed_ratio`, `underexposed_ratio` — нормализованные версии
- `highlight_clipping_ratio`, `shadow_clipping_ratio` — доли клиппинга
- `lighting_uniformity_index` — индекс равномерности освещения
- `center_brightness`, `corner_brightness` — яркость центра и углов
- `vignetting_score` — оценка виньетирования (0-1)
- `vignetting_score_norm` — нормализованная версия

#### Источники света
- `light_source_count_estimate` — оценка количества источников света (0-5)
- `soft_light_probability`, `hard_light_probability` — вероятности мягкого/жёсткого света
- `soft_light_prob` — укороченный алиас для компактного вектора

### Scene-level признаки

#### Агрегированные покадровые признаки
Для каждой числовой покадровой фичи автоматически вычисляются:
- `{feature}_mean` — среднее по кадрам сцены
- `{feature}_std` — стандартное отклонение по кадрам сцены

#### Метаданные сцены
- `num_frames` — количество обработанных кадров
- `num_frames_norm` — нормированная длина сцены (0-1)
- `scene_label` — метка сцены из scene_classification
- `scene_id` — ID сцены

#### Временные паттерны
- `brightness_change_speed` — средняя скорость изменения яркости
- `scene_flicker_intensity` — интенсивность мерцания
- `flash_events_count` — количество вспышек
- `flash_events_count_norm` — нормализованное количество вспышек
- `color_change_speed` — скорость изменения цвета
- `color_transition_variance` — дисперсия цветовых переходов
- `color_stability` — стабильность цвета (1 / (1 + mean_color_diff))
- `color_temporal_entropy` — энтропия временной последовательности hue
- `color_pattern_periodicity` — периодичность цветовых паттернов
- `scene_color_shift_speed` — скорость цветового сдвига
- `scene_contrast` — средний контраст по сцене
- `dynamic_range` — динамический диапазон яркости в сцене

### Video-level признаки

#### Агрегаты по сценам
Для каждой числовой сценовой фичи вычисляются:
- `{feature}_mean` — среднее по сценам
- `{feature}_std` — стандартное отклонение
- `{feature}_min` — минимум
- `{feature}_max` — максимум

#### Распределения по кадрам
- `color_distribution_entropy` — энтропия распределения hue по всему видео
- `color_distribution_gini` — коэффициент Джини для распределения оттенков

#### Стили цветокоррекции
- `style_teal_orange_prob` — вероятность стиля Teal & Orange
- `style_film_prob` — вероятность кинематографического стиля
- `style_desaturated_prob` — вероятность десатурации
- `style_hyper_saturated_prob` — вероятность гипернасыщенности
- `style_vintage_prob` — вероятность винтажного стиля
- `style_tiktok_prob` — вероятность стиля TikTok

#### Эстетические оценки
- `nima_mean`, `nima_std` — оценки эстетики на основе контраста
- `laion_mean`, `laion_std` — оценки эстетики на основе цветности
- `cinematic_lighting_score` — оценка кинематографического освещения (0-1)
- `professional_look_score` — оценка профессиональности кадра (0-1)

#### Глобальная динамика
- `global_brightness_change_speed` — глобальная скорость изменения яркости
- `global_color_change_speed` — глобальная скорость изменения цвета
- `strobe_transition_frequency` — частота стробоскопических переходов
- `global_color_periodicity` — глобальная периодичность цветовых паттернов
- `global_color_shift` — глобальный цветовой сдвиг

## Технические детали

### Алгоритм обработки

1. **Загрузка сцен**: Модуль загружает информацию о сценах из `scene_classification` через `load_dependency_results()`
2. **Итерация по сценам**: Для каждой сцены:
   - Определяются индексы кадров сцены
   - Берутся только `frame_indices`, заданные Segmenter (без пересэмплинга)
   - Для каждого кадра извлекаются frame-level признаки
3. **Агрегация на уровне сцены**: Вычисляются scene-level признаки из покадровых
4. **Агрегация на уровне видео**: Вычисляются video-level признаки из сценовых
5. **Формирование последовательностей**: Создаются компактные векторы для VisualTransformer

### Обработка ошибок

- Ошибки при извлечении фичей приводят к `RuntimeError` (no-fallback)
- При отсутствии зависимостей (`scene_classification`) модуль выбрасывает `RuntimeError`

### Производительность

- **Источник метрик**: `docs/models_docs/resource_costs/color_light_costs_v1.json`
- **Единица обработки**: `frame`
- **Статус**: требуется измерение latency/RAM/VRAM

### Логирование

Модуль использует стандартный logger через `get_logger("color_light")`:
- **INFO**: Прогресс обработки сцен и кадров
- **ERROR**: Ошибки при обработке кадров
- **WARN**: Предупреждения (если есть)

## Интеграция с пайплайном

Модуль интегрирован с VisualProcessor через `BaseModule`:

- Автоматическая загрузка зависимостей через `load_dependency_results("scene_classification")`
- Стандартизированное сохранение результатов через `save_results()`
- Единый интерфейс `process()` для всех модулей
- Использование `FrameManager` (RGB contract; при `color_space=BGR` кадры конвертируются)

### Порядок выполнения

Модуль должен выполняться **после** модуля `scene_classification`, так как требует его результатов.

## Дополнительная документация

- [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md) — подробное описание всех извлекаемых фичей с математическими формулами
- [BaseModule documentation](../../base_module.py) — базовый класс для модулей

## Sampling / units-of-processing requirements

- **Источник выборки**: Segmenter является единственным владельцем sampling.
- Модуль **не** генерирует кадры сам и **не** пересэмплирует `frame_indices`.
- `frame_indices` читаются из `frames_dir/metadata.json` в секции `color_light.frame_indices` (union domain).
- Временная ось: `times_s = union_timestamps_sec[frame_indices]`.

## Models

> В текущем коде оценки эстетики не используют реальные модели. Ниже — рекомендуемые варианты для интеграции:

### GPU Models (предложение)
1. **NIMA (InceptionV2/MobileNetV2)**  
   - **Runtime**: inprocess  
   - **Engine**: torch / onnx  
   - **Precision**: fp16  
   - **Device**: cuda  
2. **LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + linear head)**  
   - **Runtime**: inprocess или triton  
   - **Engine**: torch / onnx / tensorrt  
   - **Precision**: fp16  
   - **Device**: cuda  

### CPU Models (опционально)
1. **NIMA (MobileNetV2)**  
   - **Runtime**: inprocess  
   - **Engine**: onnx / torch  
   - **Precision**: fp32  
   - **Device**: cpu  

## Parallelization

- **Внутренний параллелизм**: возможен батчинг вычисления фичей по кадрам (TODO: реализовать безопасный батчинг).
- **Внешний параллелизм**: допускается запуск нескольких экземпляров на разных `run_id` (изоляция по per-run storage).
- **GPU**: используется только если будут подключены эстетические модели.

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/color_light_costs_v1.json`  
**Единица обработки**: `frame`  
**Статус**: требуется измерение latency/RAM/VRAM

## Quality validation & human-friendly inspection

### Как проверить качество выхода компонента

1. **Автоматическая оценка**: базовые sanity checks по диапазонам значений (см. FEATURES_DESCRIPTION).  
2. **Human‑friendly визуализация**: HTML отчет со скриншотами и графиками признаков.  
3. **Интеграция с downstream**: проверить, что `sequence_inputs["frames"]` корректно читается моделями.  

Скрипт генерации HTML отчета:
```
python VisualProcessor/modules/color_light/quality_report/demo_color_light_quality.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --out-html /path/to/color_light_quality.html
```

## Presentation output (для UI)

Компонент умеет генерировать JSON/HTML summary (вне `result_store`):
```
python VisualProcessor/modules/color_light/main.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --presentation-dir /path/to/VisualProcessor/presentation
```

## Примеры использования

### Базовый пример

```python
from modules.color_light.processor import ColorLightProcessor

processor = ColorLightProcessor(rs_path="./results")

saved_path = processor.run(
    frames_dir="/path/to/frames_dir",
    config={}
)

# Доступ к результатам
results = processor._load_npz(saved_path)
video_features = results["video_features"]
print(f"Cinematic Lighting Score: {video_features['cinematic_lighting_score']:.3f}")
print(f"Professional Look Score: {video_features['professional_look_score']:.3f}")
```

### Анализ стиля цветокоррекции

```python
video_features = results["video_features"]

styles = {
    "Teal & Orange": video_features.get("style_teal_orange_prob", 0),
    "Film": video_features.get("style_film_prob", 0),
    "Vintage": video_features.get("style_vintage_prob", 0),
    "TikTok": video_features.get("style_tiktok_prob", 0),
}

dominant_style = max(styles.items(), key=lambda x: x[1])
print(f"Доминирующий стиль: {dominant_style[0]} ({dominant_style[1]:.2%})")
```

## Лицензия

Модуль является частью проекта TrendFlowML и использует стандартные библиотеки с открытым исходным кодом (numpy, opencv-python, scipy, scikit-learn).

## Авторы

Компонент разработан как часть проекта TrendFlowML для анализа видео контента.

