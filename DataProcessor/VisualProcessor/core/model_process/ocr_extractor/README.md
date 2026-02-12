## Component: `ocr_extractor` (core provider, v1)

### Назначение

`ocr_extractor` делает OCR по bbox‑кропам класса `text_region` из `core_object_detections`.

Идея (зафиксировано):
- `core_object_detections` даёт **только регионы текста** (`text_region`) — это *не OCR*.
- `ocr_extractor` берёт эти bbox и запускает OCR движок, формируя сырой список детекций текста.
- downstream (`franchise_recognition`, `text_scoring`) используют этот артефакт, но должны быть устойчивы к его отсутствию.

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_object_detections.frame_indices` (sampling group)
  - `union_timestamps_sec`
- `rs_path/core_object_detections/detections.npz`:
  - `boxes/scores/class_ids/valid_mask/class_names`

### OCR engine (v1)

MVP v1 использует **`tesseract` бинарник** через subprocess.

Если `tesseract` не установлен в системе — компонент пишет валидный `empty` (`empty_reason="dependency_missing"`).

### Output (NPZ)

Пишется в: `rs_path/ocr_extractor/ocr.npz`

Ключи (v1):
- `frame_indices (N,) int32` — sampling group (= `core_object_detections.frame_indices`)
- `times_s (N,) float32`
- `ocr_raw` — object array (scalar) со значением `list[dict]`, где dict содержит минимум:
  - `frame` (int, union-domain)
  - `time_s` (float)
  - `bbox` ([x1,y1,x2,y2])
  - `text_raw` (str)
  - `text_norm` (str)
  - `det_confidence` (float) — score из `core_object_detections` для bbox
  - `engine` (str) — например `"tesseract"`
  - `lang` (str) — язык движка (например `"eng+rus"`)
- `meta` — стандартный meta (`producer/schema/status/empty_reason/...`)

### Cost controls

- `--max-boxes-per-frame` (default: 5)
- `--min-det-score` (default: 0.5)
- `--max-total-boxes` (default: 5000)

### Stage timings и progress

Компонент измеряет время выполнения ключевых стадий и сохраняет их в `meta.stage_timings_ms`:

- `initialization` — загрузка `metadata.json`, валидация `frame_indices`
- `load_deps` — проверка наличия tesseract, загрузка `core_object_detections/detections.npz`
- `process_frames` — OCR обработка всех bbox-кропов
- `saving` — формирование `meta` и атомарная запись NPZ
- `total` — общее время работы компонента

Компонент публикует прогресс в `state_events.jsonl`:
- Стадии: `start → load_deps → process_frames → save → done`
- Гранулярный прогресс во время `process_frames` (≥10 обновлений по кадрам)

---

## Models

### CPU Models

1. **Tesseract OCR** (text recognition)
   - **Triton**: ❌ Нет (CLI бинарник через subprocess)
   - **Runtime**: `inprocess` (subprocess)
   - **Engine**: `tesseract` (CLI)
   - **Precision**: N/A (не ML-модель)
   - **Device**: `cpu`
   - **Model path**: системный бинарник `tesseract` (должен быть установлен в системе)
   - **Languages**: поддерживаются через `--tesseract-lang` (например, `eng+rus`)
   - **PSM modes**: поддерживаются через `--tesseract-psm` (default: 6)

**Примечание**: 
- Tesseract не является ML-моделью в смысле ModelManager, поэтому `models_used[]` пустой
- Если tesseract не установлен, компонент пишет валидный empty artifact с `empty_reason="dependency_missing"`

---

## Parallelization

### Внутренний параллелизм

- **Последовательная обработка**: Компонент обрабатывает bbox-кропы последовательно (tesseract subprocess)
- **Ограничения**: Tesseract CLI не поддерживает параллельную обработку из одного процесса
- **Cost controls**: Параметры `--max-boxes-per-frame` и `--max-total-boxes` ограничивают количество обрабатываемых кропов

### Внешний параллелизм

- **Можно запускать несколько экземпляров параллельно** на разных видео (разные `run_id`)
- **Требования к изоляции**:
  - Разные `run_id` для каждого видео
  - Разные пути `result_store` (обеспечивается через `platform_id/video_id/run_id`)
  - Изоляция CPU: каждый экземпляр запускает свой subprocess tesseract

### Комбинированный подход

- Внешний запуск на разных видео/CPU (по одному компоненту на CPU core для оптимальной производительности)
- Thread-safety: компонент thread-safe для параллельного запуска на разных видео

---

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/ocr_extractor_costs_v1.json` (планируется)

**Единица обработки**: `bbox_crop` (один bbox-кроп, обработанный через tesseract)

**Типичные значения (preset="default")**:

| Resolution | Latency per bbox | CPU RAM peak | Notes |
|------------|------------------|--------------|-------|
| 320p | TBD ms | TBD MB | measurements pending |
| 640p | TBD ms | TBD MB | measurements pending |
| 960p | TBD ms | TBD MB | measurements pending |

**Для видео с N кадрами и M bbox**: Total latency ≈ M × latency_per_bbox (зависит от количества `text_region` детекций)

**Полные данные**: см. `docs/models_docs/resource_costs/ocr_extractor_costs_v1.json` (планируется)

---

## Sampling / units-of-processing requirements

**Требования к выборке кадров**:

Компонент использует shared sampling group через `core_object_detections.frame_indices`. Все компоненты этой группы должны работать на **одном и том же** primary `frame_indices` (иначе downstream падает из-за mismatch).

Требования к выборке:

- **Coverage**: обязательно покрывать начало/середину/конец видео и быть равномерной по времени
- **Непрерывная кривая**: количество кадров должно зависеть от длительности видео через непрерывную монотонную функцию (без скачков)
- **Минимальное значение**: минимум 50 кадров (для коротких видео)
- **Максимальное значение**: максимум 1500 кадров (cap для длинных видео)
- **Целевое значение**: зависит от длительности через кривую `target_gap_sec = f(duration_s)`

**Рекомендуемая политика выборки** (Segmenter-owned):

- `target_gap_sec = f(duration_s)` — непрерывная монотонная кривая, построенная через log‑log интерполяцию по anchor‑точкам
- `budget_n = round(duration_s / target_gap_sec)` (и затем `N = min(requested_max, budget_n)`)

Ориентиры по кривой (приблизительно):
- **≈ 5 минут**: `target_gap_sec ≈ 1s`
- **≈ 10 минут**: `target_gap_sec ≈ 2s`
- **≈ 20 минут**: `target_gap_sec ≈ 3–4s` (целимся около **3.5s**)

**Требования к разрешению**:

- **min shorter side**: 320 px
- **target**: 640 px
- **max useful**: 1080 px
- **апскейл запрещён** (только downscale)

**No-fallback policy**:
- Если `core_object_detections.frame_indices` отсутствует или пустой → **RuntimeError** (no-fallback)
- Если `union_timestamps_sec` отсутствует → **RuntimeError** (no-fallback)
- Если `core_object_detections/detections.npz` отсутствует → **RuntimeError** (no-fallback)

**Важно**: Segmenter является единственным владельцем sampling (компонент не генерирует семплинг сам).

---

## Quality validation & human-friendly inspection

### Как проверить качество выхода компонента

#### 1. Human-friendly визуализация

Для визуальной проверки качества OCR результатов используйте скрипт:

```bash
python3 DataProcessor/VisualProcessor/core/model_process/ocr_extractor/quality_report/demo_ocr_extractor_quality.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store \
  --out-dir /path/to/output \
  --max-frames 20
```

Скрипт создаёт HTML отчёт с:
- **Визуализацией OCR**: кадры с нарисованными bbox и распознанным текстом
- **Статистикой**: общее количество OCR результатов, средние значения confidence
- **Распределением языков**: статистика по языкам распознавания
- **Таблицей OCR результатов**: для каждого кадра отдельная таблица со всеми распознанными текстами

**Что проверять визуально**:
- ✅ Корректность распознавания текста (текст соответствует содержимому bbox)
- ✅ Корректность bbox (bbox соответствуют текстовым регионам на изображении)
- ❌ False positives (ложное распознавание текста там, где его нет)
- ❌ False negatives (пропущенный текст)
- ❌ Ошибки распознавания (неправильные символы, опечатки)

#### 2. Статистическая валидация

**Ожидаемые диапазоны значений** (для типичных видео):

- **Количество OCR результатов на кадр**: 0-5 (зависит от количества `text_region` детекций)
- **Text length**: 1-100 символов (зависит от типа текста)
- **Det confidence**: >0.5 для валидных детекций (threshold=0.5 по умолчанию)

**Проверка разумности**:
- Отсутствие аномальных значений (NaN где не ожидается)
- `frame_indices` отсортированы и уникальны
- `times_s` соответствует `union_timestamps_sec[frame_indices]`
- `ocr_raw` содержит валидные структуры данных

#### 3. Интеграция с downstream модулями

Компонент используется следующими downstream компонентами:
- `franchise_recognition`: использует OCR результаты для распознавания франшиз
- `text_scoring`: использует OCR результаты для оценки текста

**Проверка**: Убедитесь, что downstream компоненты корректно читают артефакты и устойчивы к отсутствию артефакта (valid empty).


