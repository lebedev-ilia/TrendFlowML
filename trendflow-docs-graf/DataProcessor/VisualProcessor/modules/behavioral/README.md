# Behavioral Analysis Module

Модуль комплексного анализа поведения людей в видео. Компонент извлекает детальные признаки жестов рук, языка тела, активности речи, вовлеченности, уверенности и признаков стресса на основе MediaPipe landmarks.

## 📋 Содержание

- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Зависимости](#зависимости)
- [Установка](#установка)
- [Использование](#использование)
- [Структура выходных данных](#структура-выходных-данных)
- [Параметры конфигурации](#параметры-конфигурации)
- [Технические детали](#технические-детали)
- [Производительность](#производительность)

## Обзор

Модуль `behavioral` выполняет комплексный анализ поведения людей в видео на основе ключевых точек (landmarks) от MediaPipe. Модуль анализирует:

- **Жесты рук**: детальная классификация 10+ типов жестов (указание, открытая ладонь, кулак, OK, победа и др.)
- **Язык тела**: поза, открытость/закрытость, доминантность, наклон тела, баланс
- **Активность речи**: динамика рта, прокси-метрики речевой активности
- **Вовлеченность**: индекс вовлеченности на основе жестов, позы и речи
- **Уверенность**: индекс уверенности/доминантности на основе языка тела
- **Стресс**: детекция признаков стресса и тревожности (моргание, self-touch, ёрзание)

Модуль извлекает два типа признаков:
1. **Sequence Features** — непрерывные признаки для каждого кадра (для VisualTransformer)
2. **Aggregate Features** — агрегированные статистики по всему видео (для MLP/Tabular Head)

### Основные возможности

- ✅ Детальная классификация жестов рук (10+ типов) с soft-представлением
- ✅ Анализ языка тела с непрерывными физическими сигналами
- ✅ Детекция активности речи через динамику рта
- ✅ Вычисление индексов вовлеченности, уверенности и стресса
- ✅ Извлечение sequence features для временного моделирования
- ✅ Агрегированные метрики для высокоуровневого анализа
- ✅ Интеграция с BaseModule для единообразия с другими модулями
- ✅ Работа с numpy массивами landmarks (оптимизировано)

## Архитектура

### Используемые технологии

Модуль использует результаты модуля `core_face_landmarks`, который предоставляет:
- **Pose landmarks**: 33 ключевые точки тела (MediaPipe Pose)
- **Hand landmarks**: до 2 рук, по 21 точке на руку (MediaPipe Hands)
- **Face landmarks**: до 1 лица, 468 точек (MediaPipe Face Mesh)

### Обработка данных

1. **Загрузка landmarks**: Модуль загружает landmarks из `core_face_landmarks` (файл `landmarks.npz`)
2. **Обработка кадров**: Для каждого кадра извлекаются признаки поведения
3. **Классификация жестов**: Анализ состояния пальцев и классификация жестов
4. **Анализ позы**: Вычисление физических параметров языка тела
5. **Динамика речи**: Анализ движения губ и прокси-метрики речи
6. **Детекция стресса**: Анализ моргания, self-touch, ёрзания
7. **Агрегация**: Вычисление метрик по всему видео

### Компоненты модуля

- **`HandGestureClassifier`**: Классификация жестов рук (hard и soft представления)
- **`BodyLanguageAnalyzer`**: Анализ языка тела и позы
- **`SpeechBehaviorAnalyzer`**: Анализ динамики рта и речи
- **`StressAnalyzer`**: Детекция признаков стресса
- **`EngagementAnalyzer`**: Заглушка (логика перенесена на уровень агрегации)
- **`ConfidenceAnalyzer`**: Заглушка (логика перенесена на уровень агрегации)

## Зависимости

### Внешние зависимости

Модуль требует результаты модуля `core_face_landmarks`:
- Файл `landmarks.npz` должен содержать:
  - `frame_indices`: массив индексов кадров
  - `pose_landmarks`: numpy массив формы `(n_frames, 33, 4)` где 4 = [x, y, z, visibility]
  - `hands_landmarks`: numpy массив формы `(n_frames, max_num_hands, 21, 3)` где 3 = [x, y, z]
  - `face_landmarks`: numpy массив формы `(n_frames, max_num_faces, 468, 3)` где 3 = [x, y, z]

### Python зависимости

```python
numpy>=1.19.0
```

Модуль работает только с numpy массивами и не требует MediaPipe напрямую (landmarks предоставляются через `core_face_landmarks`).

### Системные требования

- **RAM**: Минимум 2GB, рекомендуется 4GB+
- **CPU**: Обработка выполняется на CPU (легковесные вычисления)
- **Диск**: Минимальные требования (только для сохранения результатов)

## Установка

Модуль является частью VisualProcessor и не требует отдельной установки. Убедитесь, что модуль `core_face_landmarks` выполнен перед запуском `behavioral`.

## Использование

### CLI интерфейс

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/results \
    --log-level INFO
```

#### Параметры CLI

- `--frames-dir` (обязательный): Директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный): Путь к хранилищу результатов (ResultsStore)
- `--log-level` (по умолчанию: INFO): Уровень логирования (DEBUG/INFO/WARN/ERROR)
- `--ui-json-path` (опционально): Путь для экспорта UI JSON из NPZ

### Программный интерфейс

```python
from modules.behavioral.behavior_analyzer import BehaviorAnalyzer
from utils.frame_manager import FrameManager

# Инициализация
analyzer = BehaviorAnalyzer(
    rs_path="/path/to/results"
)

# Загрузка метаданных
metadata = analyzer.load_metadata("/path/to/frames")
frame_indices = analyzer.get_frame_indices(metadata)

# Создание FrameManager
frame_manager = analyzer.create_frame_manager("/path/to/frames", metadata)

try:
    # Обработка
    results = analyzer.process(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        config={}
    )
    
    # Сохранение результатов
    saved_path = analyzer.save_results(
        results=results,
        metadata={"total_frames": len(frame_indices)},
        use_compressed=False  # per-frame формат
    )
finally:
    frame_manager.close()
```

## Структура выходных данных

Артефакт: `result_store/<platform_id>/<video_id>/<run_id>/behavioral/behavioral_features.npz`

- **Сводка ключей, meta → CSV, melt/QA:** `docs/FEATURE_DESCRIPTION.md`

### Schema (Audit v3)

- **schema_version**: `behavioral_npz_v1`
- **producer_version**: `2.0.1`
- **Human schema**: [SCHEMA.md](./SCHEMA.md)
- **Machine schema**: `DataProcessor/VisualProcessor/schemas/behavioral_npz_v1.json`

### Обязательные поля

- `frame_indices (N,) int32` — индексы кадров (union‑domain).
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]` (source‑of‑truth).
- `landmarks_present (N,) bool` — маска кадров, где есть данные из `core_face_landmarks`.
- `frame_results (N,) object` — сериализованные per‑frame результаты (словарь для каждого кадра).
- `aggregated (object)` — агрегированные метрики по всему видео.

### Sequence features (для ML)

Все временные признаки сохраняются как массивы длины `N`:
- `seq_num_hands`
- `seq_hands_visibility`
- `seq_hand_motion_energy`
- `seq_arm_openness`
- `seq_pose_expansion`
- `seq_body_lean_angle`
- `seq_balance_offset`
- `seq_shoulder_angle`
- `seq_shoulder_angle_velocity`
- `seq_head_position_x_norm`, `seq_head_position_y_norm`
- `seq_head_motion_energy`, `seq_head_stability`
- `seq_mouth_width_norm`, `seq_mouth_height_norm`, `seq_mouth_area_norm`
- `seq_mouth_velocity`, `seq_mouth_open_ratio`, `seq_speech_activity_proxy`
- `seq_blink_flag`, `seq_blink_rate_short`, `seq_self_touch_flag`, `seq_fidgeting_energy`
- `seq_timestamp_norm`

Вероятности жестов по классам:
- `seq_gesture_prob_<gesture>` для каждого жеста (`pointing`, `open_palm`, `hands_on_hips`, `self_touch`, `fist`, `thumbs_up`, `thumbs_down`, `victory`, `ok`, `rock`, `call_me`, `love`).

### Per‑frame результаты (frame_results)

Для каждого кадра возвращается словарь:

#### Frame-level результаты

- **`hand_gestures`** (`List[str]`): Список распознанных жестов в кадре
- **`num_hands`** (`int`): Количество обнаруженных рук (0, 1 или 2)
- **`body_language`** (`Dict`): Результаты анализа языка тела
  - `posture`: `"standing"` или `"sitting"`
  - `open_posture`, `closed_posture`, `power_pose`, `rigidity`, `relaxed`: булевы флаги
  - `arm_openness`, `pose_expansion`, `body_lean_angle`, `balance_offset`, `shoulder_angle`: непрерывные признаки
- **`speech_behavior`** (`Dict`): Результаты анализа речи
  - `mouth_width_norm`, `mouth_height_norm`, `mouth_area_norm`: нормализованные параметры рта
  - `mouth_velocity`: скорость изменения площади рта
  - `mouth_open_ratio`: соотношение открытия рта
  - `speech_activity_proxy`: прокси-метрика активности речи (0.0-1.0)
- **`stress`** (`Dict`): Признаки стресса
  - `blink_flag`: флаг моргания (0/1)
  - `blink_rate_short`: частота моргания за короткое окно
  - `self_touch_flag`: флаг self-touch жестов (0/1)
  - `fidgeting_energy`: энергия ёрзания
- **`timestamp`** (`float`): Временная метка кадра в секундах (`times_s`)
- **`landmarks_present`** (`bool`): Есть ли landmarks от `core_face_landmarks` для кадра

#### Sequence Features (для VisualTransformer)

Словарь `sequence_features` содержит непрерывные признаки для каждого кадра:

**Жесты:**
- `num_hands` (`int`): Количество рук
- `hands_visibility` (`int`): Видимость рук (0/1)
- `gesture_probs` (`Dict[str, float]`): Распределение вероятностей по жестам (soft representation)
- `hand_motion_energy` (`float`): Энергия движения рук

**Язык тела:**
- `arm_openness` (`float`): Открытость рук (wrist_distance / shoulder_width)
- `pose_expansion` (`float`): Расширение позы (отношение площади человека к площади кадра)
- `body_lean_angle` (`float`): Угол наклона тела (-1.0 до 1.0, назад → вперед)
- `balance_offset` (`float`): Смещение центра масс (-1.0 до 1.0, влево → вправо)
- `shoulder_angle` (`float`): Угол плеч в градусах
- `shoulder_angle_velocity` (`float`): Скорость изменения угла плеч

**Голова:**
- `head_position_x_norm` (`float`): Нормализованная X-позиция головы (0.0-1.0)
- `head_position_y_norm` (`float`): Нормализованная Y-позиция головы (0.0-1.0)
- `head_motion_energy` (`float`): Энергия движения головы
- `head_stability` (`float`): Стабильность головы (обратная к motion_energy)

**Речь:**
- `mouth_width_norm`, `mouth_height_norm`, `mouth_area_norm`: нормализованные параметры рта
- `mouth_velocity`: скорость изменения площади рта
- `mouth_open_ratio`: соотношение открытия рта
- `speech_activity_proxy`: прокси-метрика активности речи

**Стресс:**
- `blink_flag`, `blink_rate_short`: признаки моргания
- `self_touch_flag`: флаг self-touch жестов
- `fidgeting_energy`: энергия ёрзания

**Временные:**
- `timestamp_norm` (`float`): Нормализованное время (0.0-1.0, относительно длительности видео)

#### Aggregate Features (для MLP/Tabular Head)

Агрегированные метрики вычисляются через метод `_aggregate_results()`:

**Вовлеченность:**
- `avg_engagement`, `max_engagement`: средний и максимальный индекс вовлеченности
- `engagement_variance`: вариативность вовлеченности
- `engagement_peaks`: количество пиков вовлеченности
- `early_engagement_mean`, `late_engagement_mean`: средние значения в начале и конце видео
- `engagement_contrast`: контраст между пиком и средним

**Уверенность:**
- `avg_confidence`, `max_confidence`: средний и максимальный индекс уверенности
- `confidence_variance`: вариативность уверенности
- `confidence_peak_count`: количество пиков уверенности
- `confidence_contrast`: контраст между пиком и средним

**Стресс:**
- `avg_stress`, `max_stress`: средний и максимальный уровень стресса
- `stress_spike_count`: количество всплесков стресса
- `stress_duration_ratio`: доля времени с высоким стрессом (>0.5)
- `stress_contrast`: контраст между пиком и средним

**Жесты:**
- `gesture_counts` (`Dict[str, int]`): Количество каждого типа жеста
- `gesture_rate_per_sec` (`float`): Частота жестов в секунду
- `gesture_entropy_mean` (`float`): Средняя энтропия распределения жестов
- `dominant_gesture_ratio` (`float`): Доля доминирующего жеста
- `gesture_switching_rate` (`float`): Частота смены жестов

**Язык тела:**
- `avg_arm_openness`: Средняя открытость рук
- `avg_pose_expansion`: Среднее расширение позы
- `body_motion_energy_mean`, `body_motion_energy_var`: Средняя энергия движения тела и её вариативность

**Речь:**
- `speech_activity_ratio`: Доля времени с активной речью (>0.5)
- `speech_burstiness`: Концентрация речевой активности
- `mouth_rhythm_score`: Ритмичность речи (стандартное отклонение)

**Временные паттерны:**
- `early_late_ratios` (`Dict`): Соотношения между началом и концом видео
  - `engagement`: соотношение вовлеченности
  - `speech_activity`: соотношение речевой активности
  - `gesture_rate`: частота жестов

**Видимость:**
- `hands_visibility_ratio`: Доля кадров с видимыми руками
- `face_visibility_ratio`: Доля кадров с видимым лицом
- `center_bias_mean`: Среднее смещение от центра кадра

### Пример структуры

```python
results = {
    0: {  # frame_idx
        "hand_gestures": ["pointing", "open_palm"],
        "num_hands": 2,
        "body_language": {
            "posture": "standing",
            "arm_openness": 1.5,
            "pose_expansion": 0.3,
            ...
        },
        "speech_behavior": {
            "speech_activity_proxy": 0.7,
            ...
        },
        "stress": {
            "blink_flag": 0,
            "fidgeting_energy": 0.001,
            ...
        },
        "sequence_features": {
            "num_hands": 2,
            "gesture_probs": {"pointing": 0.8, "open_palm": 0.2, ...},
            "arm_openness": 1.5,
            "speech_activity_proxy": 0.7,
            "timestamp_norm": 0.1,
            ...
        },
        "timestamp": 0.33
    },
    1: { ... },  # следующий кадр
    ...
}
```

Подробное описание всех фичей см. в [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md).

### UI payload

Компонент формирует JSON‑payload для фронта и сохраняет его в `meta.ui_payload` внутри NPZ (source‑of‑truth остаётся NPZ).
Также доступен экспорт JSON через CLI флаг `--ui-json-path`.

## Параметры конфигурации

Модуль имеет один параметр, влияющий на “вес” артефакта (без изменения схемы):

- **`store_debug_objects`** (`bool`, default: `true`):
  - `true`: сохраняет полный `frame_results` и `hand_gestures` (удобно для QA/аудита).
  - `false`: сохраняет `frame_results` как пустые dict’ы и `hand_gestures` как пустые списки (существенно меньше NPZ по размеру; для production).
  
Остальные настройки заложены в алгоритмах анализа.

### Внутренние параметры

- **Окно истории для речи**: 10 кадров (в `SpeechBehaviorAnalyzer`)
- **Окно истории для стресса**: 30 кадров (в `StressAnalyzer`)
- **Порог моргания (EAR)**: 0.2 (в `StressAnalyzer`)
- **Порог self-touch**: wrist.y < 0.3 (в `HandGestureClassifier`)
- **Epsilon для soft-жестов**: 1e-3 (в `HandGestureClassifier`)

## Технические детали

### Классификация жестов

1. **Определение состояния пальцев**: Для каждого пальца проверяется позиция кончика относительно сустава (PIP)
   - Большой палец: проверка по оси X
   - Остальные пальцы: проверка по оси Y

2. **Hard классификация**: Последовательная проверка условий для каждого типа жеста
   - Возвращает первый подходящий жест или `"unknown"`

3. **Soft классификация**: Распределение вероятностей по всем жестам
   - Все жесты получают базовый вес (epsilon)
   - Найденный жест получает повышенный вес (1.0)
   - Вектор нормализуется до суммы 1.0

### Анализ языка тела

Модуль вычисляет непрерывные физические сигналы вместо дискретных поз:

- **arm_openness**: Отношение расстояния между запястьями к ширине плеч
- **pose_expansion**: Отношение площади bounding box человека к площади кадра
- **body_lean_angle**: Нормализованный угол наклона тела (-1.0 до 1.0)
- **balance_offset**: Смещение центра масс относительно центра кадра (-1.0 до 1.0)
- **shoulder_angle**: Абсолютный угол плеч в градусах

Старые дискретные флаги (`open_posture`, `closed_posture`, `power_pose`, etc.) сохраняются для обратной совместимости, но не используются в новой схеме.

### Анализ речи

1. **Извлечение параметров рта**: Используются индексы точек губ из MediaPipe Face Mesh
2. **Вычисление динамики**: Ширина, высота, площадь рта
3. **Скорость изменения**: Мгновенная скорость изменения площади рта
4. **Прокси-метрика речи**: Sigmoid от нормализованной скорости изменения

### Детекция стресса

1. **Моргание (EAR)**: Eye Aspect Ratio для обоих глаз
   - Моргание детектируется при EAR < 0.2
   - Частота моргания вычисляется за окно истории

2. **Self-touch**: Детектируется через классификацию жестов
   - Если рука близко к лицу (wrist.y < 0.3), фиксируется self-touch

3. **Ёрзание**: Вариативность позиции носа за окно истории
   - Высокая вариативность указывает на ёрзание

### Агрегация метрик

Агрегированные метрики вычисляются из sequence features:

- **Вовлеченность**: Комбинация `speech_activity_proxy`, `arm_openness`, `body_lean_angle`
- **Уверенность**: Комбинация `arm_openness`, `body_lean_angle`
- **Стресс**: Комбинация `blink_rate_short`, `self_touch_flag`, `fidgeting_energy`

Все метрики нормализуются через sigmoid и взвешиваются для получения финальных индексов.

### Обработка отсутствующих данных

- **Отсутствие landmarks**: кадр сохраняется с NaN в seq‑признаках и `landmarks_present=false` (логируем)
- **NaN значения**: проверка на NaN перед использованием landmarks
- **Невидимые точки**: Проверка visibility для pose landmarks (< 0.5 считается невидимым)

## Sampling requirements (baseline)

- `frame_indices` берутся **только** из `frames_dir/metadata.json` (`behavioral.frame_indices`).
- Источник истины времени: `union_timestamps_sec` (компонент **не** вычисляет время сам).
- Segmenter — единственный владелец sampling.

## Models

Компонент не использует ML‑модели напрямую. Он потребляет результат `core_face_landmarks` (MediaPipe landmarks).

## Parallelization

- **Внутренний параллелизм**: отсутствует, обработка последовательная (CPU).
- **Внешний параллелизм**: допускается параллельный запуск на разных `run_id`.
- **Ограничения**: общий `rs_path` должен быть изолирован per‑run (`platform_id/video_id/run_id`).

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/behavioral_costs_v1.json` (TBD)

**Единица обработки**: `frame`

**Типичные значения**: TBD (нужны измерения latency/RAM).

## Производительность и ограничения (кратко)

- CPU‑only, лёгкие numpy‑операции
- зависит от `core_face_landmarks` (обязательная зависимость)

## Примеры использования

### Базовый пример

```python
from modules.behavioral.behavior_analyzer import BehaviorAnalyzer

analyzer = BehaviorAnalyzer(
    rs_path="./results"
)

# Загрузка и обработка
metadata = analyzer.load_metadata("./frames")
frame_indices = analyzer.get_frame_indices(metadata)
frame_manager = analyzer.create_frame_manager("./frames", metadata)

results = analyzer.process(
    frame_manager=frame_manager,
    frame_indices=frame_indices,
    config={}
)

# Сохранение
analyzer.save_results(
    results=results,
    metadata={"total_frames": len(frame_indices)}
)
```

### Агрегация результатов

```python
# После обработки можно получить агрегированные метрики
aggregated = analyzer._aggregate_results(list(results.values()))

print(f"Средняя вовлеченность: {aggregated['avg_engagement']:.2f}")
print(f"Средняя уверенность: {aggregated['avg_confidence']:.2f}")
print(f"Средний стресс: {aggregated['avg_stress']:.2f}")
print(f"Статистика жестов: {aggregated['gesture_counts']}")
```

## Обработка ошибок

Модуль включает безопасную обработку ошибок:

- **Отсутствие core_face_landmarks**: Выбрасывает `RuntimeError` с описанием проблемы
- **Неполные данные landmarks**: Выбрасывает `ValueError` с описанием отсутствующих данных
- **Кадры без landmarks**: Пропускаются с предупреждением в лог
- **NaN значения**: Автоматически фильтруются перед использованием

## Логирование

Модуль использует стандартный logger из `utils.logger`:

```python
from utils.logger import get_logger

logger = get_logger("behavior_analyzer")
```

Уровни логирования:
- **DEBUG**: Детальная информация о каждом кадре
- **INFO**: Общая информация о процессе (каждые 20 кадров)
- **WARN**: Предупреждения (пропущенные кадры, отсутствующие landmarks)
- **ERROR**: Критические ошибки

## Интеграция с пайплайном

Модуль интегрирован с VisualProcessor через `BaseModule`:

- Автоматическая загрузка зависимостей через `load_core_provider("core_face_landmarks")`
- Стандартизированное сохранение результатов через `save_results()`
- Единый интерфейс `process()` для всех модулей
- Поддержка per-frame формата результатов
- Поддержка batch processing через `process_batch()` (использует дефолтную реализацию из BaseModule)

### Порядок выполнения в пайплайне

1. **core_face_landmarks** → извлечение landmarks (pose, hands, face)
2. **behavioral** → анализ поведения на основе landmarks

## Batch Processing

Модуль поддерживает **batch processing** для одновременной обработки нескольких видео:

- **CPU processing**: модуль использует дефолтный `process_batch()` из `BaseModule`, который последовательно обрабатывает каждое видео
- **Изоляция**: каждый видео имеет свой `rs_path` для артефактов
- **Поддержка**: модуль реализует `supports_batch = True` для интеграции с batch processing системой

**Конфигурация batch processing** (в `global_config.yaml`):
```yaml
visual:
  batch_processing:
    enabled: true
    max_video_workers: 4
    enable_video_parallel: false
    enable_cpu_parallel: true
```

**Использование**:
- Batch processing автоматически активируется при вызове `run_batch()` в `VisualProcessor/main.py`
- Модуль реализует `supports_batch = True` и использует дефолтный метод `process_batch()`

## Quality validation & human-friendly inspection

### Human-friendly визуализация (Render System)

`behavioral` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/behavioral/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по поведенческим признакам (frames_count, landmarks_present_ratio, avg_engagement, avg_confidence, avg_stress, gesture_rate_per_sec, hands_visibility_ratio, face_visibility_ratio)
- **Timeline**: данные по каждому кадру (frame_index, time_s, landmarks_present, sequence features)
- **Distributions**: распределения ключевых признаков (speech_activity_proxy, arm_openness, body_lean_angle, hand_motion_energy, blink_rate_short, fidgeting_energy) и распределение жестов

Render-context может быть использован:
- **LLM** для генерации текстовых описаний поведения в видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions)
- **Debugging**: быстрая проверка качества поведенческого анализа без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../behavioral/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: speech_activity, arm_openness, body_lean_angle по времени
  - Distributions: статистики по ключевым признакам
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
visual:
  inline_config:
    modules:
      behavioral:
        render:
          enable_render: true  # Генерировать render-context JSON
          enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

### Legacy HTML-отчёт

HTML-отчёт: `quality_report/demo_behavioral_quality.py`

Пример:

```bash
python VisualProcessor/modules/behavioral/quality_report/demo_behavioral_quality.py \
  --npz-path /path/to/behavioral_features.npz \
  --out-html /path/to/behavioral_report.html
```

Скрипт строит базовый HTML‑отчёт с временными графиками и summary.

### Статистическая валидация (baseline)

- диапазоны значений по ключевым series‑фичам (0..1 где применимо)
- отсутствие неожиданных NaN вне `landmarks_present=false`
- корректность `times_s` и выравнивание с `frame_indices`

### Ожидаемые предупреждения

Модуль может выдавать следующие предупреждения, которые являются нормальными и не критичными:

1. **`PR-6: exec_order missing enabled components: ['behavioral']`**
   - Информационное предупреждение: модуль включен, но не указан в `execution_order` конфига
   - Модуль выполнится после компонентов из `exec_order`
   - Не критично, можно игнорировать или добавить `behavioral` в `execution_order`

2. **`N кадров отсутствуют в core_face_landmarks. Заполнены NaN и отмечены masks.`**
   - Нормальное предупреждение: в некоторых кадрах не обнаружены лица/тела
   - Модуль корректно обрабатывает это, заполняя NaN значениями
   - Типично для видео, где не все кадры содержат людей

## Дополнительная документация

- [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md) — подробное описание всех извлекаемых фичей
- [BaseModule documentation](../../base_module.py) — базовый класс для модулей

## Примечания

### Новая схема sequence features

Модуль переведен на новую схему работы с sequence features:
- Вместо дискретных меток используются непрерывные признаки
- Soft-представление жестов вместо hard-классификации
- Физические сигналы вместо интерпретированных метрик
- Высокоуровневые метрики (engagement, confidence) вычисляются на уровне агрегации

### Обратная совместимость

Старые дискретные флаги (`open_posture`, `closed_posture`, `power_pose`, etc.) сохраняются в `body_language` для обратной совместимости, но не используются в новой схеме sequence features.

## Лицензия

Модуль использует алгоритмы анализа поведения, разработанные для проекта TrendFlowML.

## Авторы

Компонент разработан как часть проекта TrendFlowML для анализа поведения людей в видео контенте.
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
