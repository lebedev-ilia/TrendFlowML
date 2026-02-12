# Модуль `text_scoring` — Анализ взаимодействия текста и видео

## Описание

Модуль `text_scoring` анализирует взаимодействие текста с видео, извлекая признаки синхронизации текста с движением, детекции call-to-action (CTA), непрерывности отображения текста и пиков акцента. Это **Tier-0 baseline** модуль, который работает как **consumer** OCR-артефакта от внешнего компонента.

### Production Policy

- ✅ Модуль является **CONSUMER** OCR-артефакта (NPZ) от внешнего компонента (TextProcessor/OCR service)
- ✅ Модуль **НЕ выполняет OCR** самостоятельно (никакого EasyOCR/pyTesseract внутри)
- ✅ Если OCR-артефакт не найден — модуль **не падает**, а возвращает валидный empty результат
- ✅ Результаты сохраняются **только в NPZ формате** (без JSON артефактов)
- ✅ Опционально использует данные из `core_face_landmarks` для мультимодального анализа

## Архитектура

Модуль наследуется от `BaseModule` и реализует следующий интерфейс:

```python
class TextScoringModule(BaseModule):
    def process(self, frame_manager, frame_indices, config) -> Dict[str, Any]
```

Внутренний пайплайн `TextVideoInteractionPipeline` выполняет:
1. Группировку OCR-детекций в уникальные текстовые элементы
2. Вычисление мультимодальных метрик (motion + face + audio alignment)
3. Детекцию CTA (call-to-action)
4. Анализ непрерывности и акцентов текста

## Алгоритм работы

### 1. Загрузка OCR данных

Модуль ищет OCR-артефакт в следующих локациях (в порядке приоритета):
- `<rs_path>/text_ocr/ocr.npz` (canonical location)
- `<rs_path>/ocr/ocr.npz` (compatibility)
- `<rs_path>/text_scoring/ocr.npz` (legacy)

Поддерживаемый минимальный формат OCR NPZ:
- Ключ `ocr_raw` или `ocr_data` → `list[dict]`
- Каждый dict содержит минимум: `frame`, `bbox`, `text`/`text_raw`, `confidence`
- Опционально: `text_norm`, `language`, `is_cta_candidate`, `time_s`

### 2. Фильтрация и группировка OCR

1. **Фильтрация по confidence**: удаляются детекции с `confidence < 0.4` (по умолчанию)
2. **Группировка в уникальные элементы**:
   - Дедупликация по IoU bbox (> 0.6) и нормализованной текстовой похожести (> 0.8)
   - Агрегация: медианный bbox, средняя confidence, временные границы (first_frame, last_frame)

### 3. Мультимодальный анализ

Для каждого уникального текстового элемента:

1. **Окно выравнивания**: `[t - w, t + w]`, где `w = 0.5 секунды` (по умолчанию)
2. **Motion signal**: z-score нормализация движения в окне
3. **Face signal**: нормализация наличия лиц (0..1) из `core_face_landmarks` (опционально)
4. **Audio signal**: нормализация аудио энергии (0..1) — placeholder (не используется в baseline)
5. **Multimodal alignment**: взвешенная комбинация `w_motion * motion + w_face * face + w_audio * audio`

### 4. Детекция CTA

CTA определяется как комбинация:
- Флагов `is_cta_candidate` из OCR
- Лексического анализа: нечеткое совпадение с CTA-ключевыми словами (EN/RU) по нормализованному тексту (fuzzy match ≥ 0.75)

Поддерживаемые CTA-ключевые слова:
- EN: "subscribe", "follow", "like", "link in bio", "click", "watch"
- RU: "подпишись", "подписаться", "ставь лайк", "ссылка в описании"

### 5. Вычисление фичей

Модуль извлекает 5 категорий фичей (см. раздел "Выходные данные").

## Входные данные

### Обязательные входы

1. **`frames_dir`**: Директория Segmenter с:
   - `metadata.json` с обязательными полями:
     - `story_structure.frame_indices`: индексы кадров для обработки (union-domain, 0..N-1)
     - `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash` (run identity keys)

2. **`rs_path`**: Путь к хранилищу результатов (result_store), содержащий:
   - OCR-артефакт: `text_ocr/ocr.npz` (или альтернативные локации)
   - Опционально: `core_face_landmarks/landmarks.npz` (если `use_face_data=True`)

### Конфигурация (config)

```python
{
    "ocr_npz": None,              # Опционально: путь к OCR NPZ (override)
    "use_face_data": False,        # Использовать core_face_landmarks для alignment
    "alignment_window_seconds": 0.5,  # Окно выравнивания (секунды)
    "motion_weight": 0.4,         # Вес движения в multimodal alignment
    "face_weight": 0.3,           # Вес лица в multimodal alignment
    "audio_weight": 0.3,          # Вес аудио в multimodal alignment
    "min_ocr_confidence": 0.4      # Минимальная confidence для OCR детекций
}
```

### Требования к OCR NPZ

Минимальная схема:
```python
{
    "ocr_raw": [  # или "ocr_data"
        {
            "frame": int,           # Индекс кадра
            "bbox": [x1, y1, x2, y2],  # Координаты текста
            "text": str,            # Исходный текст (или "text_raw")
            "confidence": float,    # Уверенность OCR (0.0-1.0)
            "text_norm": str,       # Опционально: нормализованный текст
            "language": str,        # Опционально: язык текста
            "is_cta_candidate": bool,  # Опционально: флаг CTA
            "time_s": float         # Опционально: время в секундах
        },
        ...
    ]
}
```

## Выходные данные

Результаты сохраняются в NPZ файл:
```
result_store/<platform_id>/<video_id>/<run_id>/text_scoring/text_scoring.npz
```

### Структура NPZ файла

#### Массивы (numpy)

| Ключ | Формат | Описание |
|------|--------|----------|
| `frame_indices` | `(N,) int32` | Индексы обработанных кадров (union-domain) |
| `times_s` | `(N,) float32` | Time-axis: `union_timestamps_sec[frame_indices]` (no-fallback) |
| `text_present` | `bool` | Наличие текста в видео |
| `text_presence` | `(N,) bool` | Есть ли OCR детекции на кадре (privacy-safe) |
| `text_count_per_frame` | `(N,) int32` | Кол-во OCR детекций на кадре (privacy-safe) |
| `ocr_raw` | `(M,) object` | Список сырых OCR-детекций (после фильтрации) |
| `ocr_unique_elements` | `(K,) object` | Список уникальных текстовых элементов |

#### Словарь `features` (object)

**1. Text → Action / Motion Correlation**

- `text_action_sync_score` (float): Робастная оценка синхронизации текста с движением (trimmed-mean оконных z-score motion)
- `text_motion_alignment` (float): Средняя оценка мультимодального выравнивания текста с моментами активности
- `text_motion_alignment_windowed` (float): Оконная версия alignment (максимум в окне [t-w, t+w])
- `multimodal_attention_boost_score` (float): Максимальная оценка мультимодального выравнивания (по всем элементам)
- `multimodal_attention_boost_position` (float): Относительная позиция (0..1) текста с максимальным alignment

**2. Text Duration and Continuity**

- `text_on_screen_continuity` (float): Средняя длительность отображения уникального текста (секунды)
- `text_on_screen_continuity_median` (float): Медианная длительность
- `text_on_screen_continuity_max` (float): Максимальная длительность
- `text_on_screen_continuity_std` (float): Стандартное отклонение длительности
- `text_on_screen_continuity_normalized` (float): Средняя длительность, нормализованная на длину видео
- `text_switch_rate` (float): Частота смены текста (уникальных элементов / секунда)
- `num_unique_texts` (int): Количество уникальных текстовых элементов
- `time_to_first_text_sec` (float | None): Время до появления первого текста (секунды)
- `time_to_first_text_position` (float | None): Нормализованная позиция первого текста (0..1)
- `text_area_fraction` (float): Средняя доля площади кадра, занимаемая текстом

**3. Call-to-Action (CTA) Detection**

- `cta_presence` (float): Оценка вероятности наличия CTA (0..1)
- `cta_timestamp` (float | None): Средний временной момент появления CTA (секунды, обратная совместимость)
- `cta_first_timestamp` (float | None): Время первого CTA (секунды)
- `cta_mean_timestamp` (float | None): Среднее время CTA (секунды)
- `cta_last_timestamp` (float | None): Время последнего CTA (секунды)
- `cta_first_position` (float | None): Относительная позиция первого CTA (0..1)
- `cta_mean_position` (float | None): Относительная позиция среднего CTA (0..1)
- `cta_last_position` (float | None): Относительная позиция последнего CTA (0..1)
- `cta_strength` (float): Средняя сила CTA (усреднённое multimodal alignment для CTA-элементов, 0..1)
- `persistent_cta_flag` (bool): Флаг наличия "стойкого" CTA (удерживается > 3 секунд)

**4. Text Emphasis Peaks**

- `text_emphasis_peak_flags` (list[int]): Список индексов текстовых элементов, где alignment образует пики
- `text_emphasis_peak_prominence` (list[float]): Значения prominence для каждого пика
- `text_emphasis_peak_positions` (list[float]): Относительные позиции пиков в видео (0..1)

**5. Дополнительные метрики**

- `text_readability_score` (float): Средний скор читаемости текста (0..1, учитывает длину, среднюю длину слова, пунктуацию)
- `ocr_language_entropy` (float): Энтропия распределения языков по уникальным элементам
- `text_movement_speed` (float): Средняя скорость движения текстовых элементов (в долях диагонали кадра / секунда)

**6. Метаданные**

- `ocr_npz` (str): Путь к использованному OCR NPZ файлу
- `text_present` (bool): Наличие текста в видео
- `meta.status` = `ok|empty`
- `meta.empty_reason` (str | None): причина empty

#### Структура `ocr_raw` (object array)

Каждый элемент содержит (privacy-aware):
- `frame`: индекс кадра
- `time_s`: время в секундах
- `bbox`: координаты текста `[x1, y1, x2, y2]`
- `text_raw` / `text_norm` / `text` **только если** `retain_raw_ocr_text=true`
- иначе: `text_len`, `text_hash_sha256`
- `confidence`: уверенность OCR (0.0–1.0)
- `language`: язык текста (если доступен)
- `is_cta_candidate`: флаг потенциального CTA

#### Структура `ocr_unique_elements` (object array)

Каждый элемент содержит (privacy-aware):
- `text_raw`/`text_norm` **только если** `retain_raw_ocr_text=true`
- иначе: `text_len`, `text_hash_sha256`
- `language`: язык текста
- `first_frame` / `last_frame`: границы появления элемента
- `first_time` / `last_time`: временные границы (секунды)
- `bbox_median`: медианный bbox по всем появлениям
- `aggregated_confidence`: средняя уверенность по всем кадрам

## Зависимости

### Hard dependencies (обязательные)

Модуль **не требует** обязательных core‑зависимостей: OCR может прийти из `ocr_extractor` или внешней text‑ветки.
Baseline preference: `rs_path/ocr_extractor/ocr.npz` (если присутствует), иначе legacy/compat paths.

### Optional dependencies

1. **`core_face_landmarks`** (опционально, если `use_face_data=True`)
   - Файл: `rs_path/core_face_landmarks/landmarks.npz`
   - Ключи: `frame_indices`, `face_present`
   - Использование: мультимодальный alignment (face signal)
   - **Примечание**: Если запрошен, но не найден → `FileNotFoundError`

### Требования к согласованности данных

- Если `use_face_data=True`, `frame_indices` модуля должны быть **подмножеством** `core_face_landmarks.frame_indices`
- Segmenter должен обеспечивать согласованную выборку кадров для зависимых компонентов
- При несоответствии индексов модуль выбросит `RuntimeError`

## Использование

### CLI интерфейс

```bash
python -m modules.text_scoring.main \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    [--ocr-npz /path/to/custom/ocr.npz] \
    [--use-face-data] \
    [--alignment-window-seconds 0.5] \
    [--retain-raw-ocr-text] \
    [--log-level INFO]
```

**Параметры CLI:**
- `--frames-dir` (обязательный) — директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный) — путь к хранилищу результатов (`result_store`)
- `--ocr-npz` (опционально) — путь к OCR NPZ файлу (override canonical location)
- `--use-face-data` (опционально) — использовать `core_face_landmarks` для мультимодального анализа
- `--log-level` (опционально) — уровень логирования: `DEBUG`, `INFO`, `WARN`, `ERROR` (по умолчанию: `INFO`)

### Программный интерфейс

```python
from modules.text_scoring.text_scoring import TextScoringModule
from utils.frame_manager import FrameManager

# Инициализация модуля
module = TextScoringModule(rs_path="/path/to/result_store")

# Конфигурация
config = {
    "ocr_npz": None,  # опционально: путь к OCR NPZ
    "use_face_data": False,  # использовать face landmarks
    "alignment_window_seconds": 0.5,  # окно выравнивания
    # baseline C: face-only weights
    "motion_weight": 0.0,
    "face_weight": 1.0,
    "audio_weight": 0.0,
    "min_ocr_confidence": 0.4,
    # privacy
    "retain_raw_ocr_text": False,
    # noisy extras (disabled by default)
    "enable_text_peaks": False,
    "enable_language_entropy": False,
    "enable_text_movement_speed": False,
}

# Запуск
saved_path = module.run(
    frames_dir="/path/to/frames",
    config=config
)
```

### Интеграция в pipeline

Модуль автоматически вызывается через VisualProcessor pipeline, если указан в конфигурации:

```yaml
visual_modules:
  - name: text_scoring
    config:
      use_face_data: false
      alignment_window_seconds: 0.5
      min_ocr_confidence: 0.4
```

## Обработка ошибок

### Валидные empty состояния

Модуль **не падает** при отсутствии OCR-артефакта, а возвращает валидный empty результат:

```python
{
    "frame_indices": fi,
    "text_present": np.asarray(False),
    "features": {
        "text_present": False,
        "empty_reason": "ocr_not_available" | "ocr_empty" | "ocr_outside_sampling"
    },
    "ocr_raw": np.asarray([], dtype=object),
    "ocr_unique_elements": np.asarray([], dtype=object),
}
```

**Возможные `empty_reason`:**
- `"ocr_not_available"`: OCR NPZ файл не найден ни в одной из локаций
- `"ocr_empty"`: OCR NPZ найден, но пуст (нет детекций)
- `"ocr_outside_sampling"`: OCR детекции есть, но не попадают в `frame_indices` модуля

### Fail-fast ошибки

Модуль выбрасывает ошибки в следующих случаях:

1. **Отсутствие `rs_path`**: `ValueError("text_scoring | rs_path is required")`
2. **Запрошен `use_face_data`, но `core_face_landmarks` не найден**: `FileNotFoundError`
3. **Несовпадение `frame_indices` с `core_face_landmarks`**: `RuntimeError` (Segmenter должен обеспечить консистентность)

## Методы вычисления

### Группировка OCR-детекций

1. **Фильтрация по confidence**: `confidence >= min_ocr_confidence` (по умолчанию 0.4)
2. **Дедупликация**:
   - IoU bbox > 0.6
   - Нормализованная текстовая похожесть (Levenshtein-подобное расстояние) > 0.8
   - Комбинированный скор: `0.5 * IoU + 0.5 * text_similarity > 0.8`
3. **Агрегация**: медианный bbox, средняя confidence, временные границы

### Нормализация сигналов

- **Gaussian smoothing**: `scipy.ndimage.gaussian_filter1d(signal, sigma=1)`
- **Min-max normalization**: `signal / max(signal)` → [0, 1]
- **Z-score normalization**: `(signal - mean) / std` (для energy-based метрик)

### Мультимодальный alignment

```python
alignment = w_motion * motion_norm + w_face * face_norm + w_audio * audio_norm
```

Веса нормализуются так, чтобы `w_motion + w_face + w_audio = 1.0`.

### CTA детекция

1. **Лексический анализ**: нечеткое совпадение с CTA-ключевыми словами (fuzzy match ≥ 0.75 или подстрока)
2. **Флаги из OCR**: `is_cta_candidate` из исходных детекций
3. **Агрегация**: `cta_presence = 0.5 * (num_cta / num_unique_texts * 1.5) + 0.5 * mean_confidence`

### Text Emphasis Peaks

Используется `scipy.signal.find_peaks` с параметрами:
- `prominence=0.1`: минимальная выдающаяся высота пика
- `distance=1`: минимальная дистанция между пиками

## Производительность

- **Сложность**: O(N * M), где N — количество уникальных текстовых элементов, M — размер окна выравнивания
- **Память**: O(N + M) для массивов сигналов и группированных элементов
- **Время выполнения**: ~0.1-1 секунда для типичного видео (зависит от количества OCR-детекций)

## Примеры использования результатов

### Загрузка результатов

```python
import numpy as np

# Загрузка NPZ
data = np.load("text_scoring_features_*.npz", allow_pickle=True)

# Извлечение features
features = data["features"].item()  # object array → dict
text_present = data["text_present"].item()

if text_present:
    # Text-action sync
    sync_score = features["text_action_sync_score"]
    alignment = features["text_motion_alignment"]
    
    # CTA detection
    cta_presence = features["cta_presence"]
    cta_timestamp = features["cta_timestamp"]
    
    # Text continuity
    continuity = features["text_on_screen_continuity"]
    switch_rate = features["text_switch_rate"]
    
    # Unique elements
    unique_elements = data["ocr_unique_elements"]
    for elem in unique_elements:
        print(f"Text: {elem['text_raw']}, Duration: {elem['last_time'] - elem['first_time']:.2f}s")
else:
    print(f"No text found: {features.get('empty_reason')}")
```

### Анализ CTA

```python
features = data["features"].item()

if features["cta_presence"] > 0.5:
    print(f"CTA detected with strength: {features['cta_strength']:.2f}")
    print(f"First CTA at: {features['cta_first_timestamp']:.2f}s")
    print(f"Persistent CTA: {features['persistent_cta_flag']}")
else:
    print("No CTA detected")
```

### Визуализация текстовых пиков

```python
import matplotlib.pyplot as plt

features = data["features"].item()
peak_positions = features["text_emphasis_peak_positions"]
peak_prominence = features["text_emphasis_peak_prominence"]

plt.figure(figsize=(12, 4))
plt.bar(peak_positions, peak_prominence, width=0.01, alpha=0.7)
plt.xlabel("Video Position (0..1)")
plt.ylabel("Peak Prominence")
plt.title("Text Emphasis Peaks")
plt.show()
```

## Связанные компоненты

- **TextProcessor/OCR service**: создаёт OCR-артефакт (`ocr.npz`)
- **Segmenter**: генерирует `frame_indices` и метаданные
- **core_face_landmarks**: предоставляет информацию о лицах (опционально)
- **core_optical_flow**: может использоваться для motion signal (future enhancement)

## Примечания

1. **Consumer архитектура**: Модуль не выполняет OCR, а потребляет готовый артефакт
2. **Graceful degradation**: Модуль не падает при отсутствии OCR, возвращает валидный empty результат
3. **Multimodal integration**: Опциональная интеграция с face landmarks для улучшения alignment
4. **CTA detection**: Комбинация лексического анализа и флагов из OCR
5. **Robust statistics**: Используются trimmed-mean и z-score нормализация для устойчивости к выбросам
6. **Text grouping**: Дедупликация по IoU и текстовой похожести для агрегации повторяющихся детекций

## История изменений

- **Baseline v1**: Модуль переведен на consumer архитектуру (не выполняет OCR самостоятельно)
- **Multimodal alignment**: Добавлена поддержка face landmarks для улучшения синхронизации
- **CTA detection**: Улучшена детекция через комбинацию лексического анализа и флагов
- **Empty handling**: Добавлена graceful обработка отсутствия OCR-артефакта

