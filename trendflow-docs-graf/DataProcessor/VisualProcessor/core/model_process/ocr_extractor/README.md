## Component: `ocr_extractor` (core provider, v2 / Audit v3)

**Версия**: 0.2  
**Schema Version**: `ocr_extractor_npz_v2`  
**Категория**: core provider

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

### OCR engine (v2)

Компонент поддерживает выбор OCR движка (engine) через конфиг/CLI:

- **`ppocr_rec_onnx` (recommended / best)**: распознавание текста через **ONNXRuntime** (PPOCR-recognizer),
  веса и словарь загружаются **локально** через `dp_models.ModelManager` (no-network).
- **`tesseract`**: распознавание через системный бинарник `tesseract` (CLI).

**Audit v3 decision**:
- `engine=ppocr_rec_onnx`: hard dependency = локальный model pack в `DP_MODELS_ROOT` (через ModelManager).
- `engine=tesseract`: hard dependency = установленный `tesseract` в системе.

**Важно (про запуск через VisualProcessor / cfg_path профили)**:
- CLI дефолт у `ocr_extractor` = `engine=tesseract`.
- Если вы запускаете VisualProcessor с внешним YAML (`--cfg-path` / `profile.visual.cfg_path`),
  **нужно явно задать** `ocr_extractor.engine="ppocr_rec_onnx"` и `rec_model_spec="ppocr_rec_onnx_v1_inprocess"`
  в этом YAML (иначе вы случайно протестируете tesseract вместо ONNX движка).

### Output (NPZ)

Пишется в: `rs_path/ocr_extractor/ocr.npz`

Ключи (v2):
- `frame_indices (N,) int32` — sampling group (= `core_object_detections.frame_indices`)
- `times_s (N,) float32`
- `ocr_raw (M,) object` — список OCR-детекций, где каждый элемент — `dict` (union-domain), минимум:
  - `frame` (int)
  - `time_s` (float)
  - `bbox` ([x1,y1,x2,y2])
  - `det_confidence` (float) — score из `core_object_detections` для bbox
  - `engine` (str) — `"ppocr_rec_onnx"` или `"tesseract"`
  - `lang` (str|None) — например `"eng+rus"` (актуально для tesseract)
  - `rec_confidence` (float|None) — confidence proxy распознавания (актуально для `ppocr_rec_onnx`)
  - **Если `retain_raw_ocr_text=true`**:
    - `text_raw` (str)
    - `text_norm` (str)
  - **Если `retain_raw_ocr_text=false`** (default):
    - `text_sha256` (str) — SHA256 от `text_norm`
    - `text_len` (int)
- `meta` — стандартный meta (`producer/schema/status/empty_reason/...`) + `retain_raw_ocr_text`
- `meta_json` — JSON string дубль `meta` (для кросс-языковой совместимости)

### Параметры конфигурации компонента

Все параметры принимаются через аргументы командной строки:

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--engine` | str | `tesseract` | OCR engine: `tesseract` или `ppocr_rec_onnx` (recommended) |
| `--rec-model-spec` | str | `ppocr_rec_onnx_v1_inprocess` | dp_models ModelManager spec (только для `ppocr_rec_onnx`) |
| `--ppocr-img-h` | int | `48` | Высота входа recognizer (только для `ppocr_rec_onnx`) |
| `--ppocr-img-w` | int | `320` | Ширина входа recognizer (только для `ppocr_rec_onnx`) |
| `--min-rec-score` | float | `0.0` | Минимальный `rec_confidence` (только для `ppocr_rec_onnx`) |
| `--frames-dir` | str | required | Путь к директории с кадрами |
| `--rs-path` | str | required | Путь к result_store |
| `--proposal-class` | str | `text_region` | Класс объектов для OCR (обычно `text_region`) |
| `--min-det-score` | float | `0.5` | Минимальный score детекции для обработки |
| `--max-boxes-per-frame` | int | `5` | Максимальное количество bbox на кадр |
| `--max-total-boxes` | int | `5000` | Максимальное общее количество bbox для обработки |
| `--crop-margin-frac` | float | `0.02` | Доля отступа вокруг bbox при кропе (2% по умолчанию) |
| `--tesseract-lang` | str | `eng+rus` | Языки для Tesseract OCR (например, `eng+rus`) |
| `--tesseract-psm` | int | `6` | PSM режим Tesseract (Page Segmentation Mode) |
| `--retain-raw-ocr-text` | flag | `false` | Сохранять `text_raw/text_norm` в `ocr_raw` (dev/debug). По умолчанию raw OCR **не сохраняем** (см. `docs/contracts/PRIVACY_AND_RETENTION.md`) |

**Cost controls**:
- `--max-boxes-per-frame` ограничивает количество bbox на кадр
- `--min-det-score` фильтрует детекции по confidence
- `--max-total-boxes` ограничивает общее количество обрабатываемых bbox

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
- Если `engine=tesseract` и `tesseract` не установлен — компонент **fail-fast (error)** (hard dependency)

2. **PP-OCR recognizer (ONNXRuntime)** (text recognition, recommended)
   - **Triton**: ❌ Нет (in-process ONNXRuntime)
   - **Runtime**: `inprocess`
   - **Engine**: `onnxruntime_onnx` (через `dp_models.ModelManager`)
   - **Precision**: `fp32`
   - **Device**: `auto` (CPU/CUDA EP выбирается по доступности ONNXRuntime)
   - **Model spec**: `ppocr_rec_onnx_v1_inprocess`
   - **Local artifacts** (DP_MODELS_ROOT):
     - `bundled_models/visual/ocr/ppocr_rec_onnx_v1/model.onnx`
     - `bundled_models/visual/ocr/ppocr_rec_onnx_v1/dict.txt`

---

## Parallelization

### Внутренний параллелизм

- **По умолчанию**: последовательная обработка bbox-кропов в single-video запуске.
- **Ограничения**:
  - `tesseract` CLI не параллелится “внутри” одного процесса (subprocess per-crop).
  - `ppocr_rec_onnx` может быть распараллелен (см. batch utilities VisualProcessor).
- **Cost controls**: Параметры `--max-boxes-per-frame` и `--max-total-boxes` ограничивают количество обрабатываемых кропов

### Внешний параллелизм

- **Можно запускать несколько экземпляров параллельно** на разных видео (разные `run_id`)
- **Требования к изоляции**:
  - Разные `run_id` для каждого видео
  - Разные пути `result_store` (обеспечивается через `platform_id/video_id/run_id`)
  - Изоляция CPU: каждый экземпляр запускает свои subprocess/ORT session в рамках процесса

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

### Empty/error semantics

Компонент поддерживает **valid empty** артефакты:
- Если нет текста на всех обработанных bbox → компонент создаёт валидный empty artifact с:
  - `status="empty"`
  - `empty_reason="no_text_available"`
  - `ocr_raw` — пустой массив

**No-fallback policy**:
- Отсутствие/пустота `core_object_detections.frame_indices` → **RuntimeError** (no-fallback)
- Отсутствие `union_timestamps_sec` → **RuntimeError** (no-fallback)
- Отсутствие `core_object_detections/detections.npz` → **RuntimeError** (no-fallback)
- Несоответствие `frame_indices` между `metadata.json` и `detections.npz` → **RuntimeError** (no-fallback)
- Отсутствие engine-deps:
  - `engine=tesseract` и `tesseract` не установлен → **RuntimeError** (hard dependency)
  - `engine=ppocr_rec_onnx` и отсутствуют локальные артефакты ModelManager (`weights_missing`) → **RuntimeError** (hard dependency)

### Artifact save / validation (baseline contract)

- NPZ сохраняется **атомарно** (tmp → `os.replace`).
- После записи артефакт проходит `artifact_validator.validate_npz()` (fail-fast).

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

---

## Render (dev-only): как читать OCR человеку

Цель рендера `ocr_extractor`: чтобы человек, открыв `render.html`, понял:

- “какой текст нашёлся” и **где он находится** в видео,
- насколько OCR **доверяет** распознаванию,
- есть ли проблемы (ложный текст, пропуски, мусорные символы),
- и не нарушаем ли мы privacy (сохраняем raw text или только hashes).

### Где лежат файлы рендера

- `result_store/<platform_id>/<video_id>/<run_id>/ocr_extractor/_render/render_context.json`
- `result_store/.../ocr_extractor/_render/render.html`

### Что показывает текущий HTML (MVP сейчас)

Текущий шаблонный рендер показывает базовую статистику (через render-context):

- summary: сколько OCR событий (`len(ocr_raw)`), какие engine использовались
- timeline: распределение OCR по времени (если построено)

Но этого недостаточно для QA качества текста.

### Что ДОЛЖНО появиться в персонализированном рендере (target)

Минимально обязательное для “90% понимания человеком”:

- **Кадры с bbox + текстом**:
  - K=12 равномерно по видео: на кадре рисуем bbox `text_region` + распознанный текст рядом
  - отдельный режим “только top-confidence” и “только low-confidence” (для поиска ошибок)
- **Топ-таблица OCR событий**:
  - топ-50 по `rec_confidence` (и анти-топ-50) с `time_s`, `frame`, `bbox`, `text_norm`
- **Распределения**:
  - histogram `rec_confidence` (для `ppocr_rec_onnx`)
  - histogram `det_confidence` (score bbox из `core_object_detections`)
  - количество OCR событий на кадр (0..max_boxes_per_frame)
- **Дубликаты / “залипание” текста**:
  - сгруппировать повторяющийся `text_norm` по времени (полезно для watermark/подписей)

### Privacy / retention (обязательно объяснить в рендере)

- Если `retain_raw_ocr_text=true` (dev/debug):
  - в `ocr_raw` храним `text_raw`/`text_norm` → **это чувствительные данные**
- Если `retain_raw_ocr_text=false` (production default):
  - `text_raw/text_norm` не сохраняем
  - вместо этого: `text_sha256`, `text_len` (и при необходимости безопасная статистика по charset)

Рендер должен явно показывать предупреждение, если raw text сохранён.

### Время выполнения (что смотреть)

- `meta.stage_timings_ms.process_frames` — основная стоимость OCR.
- `meta.models_used[]` — какой recognizer использовался (для воспроизводимости).

### Параметры конфига, которые сильнее всего меняют результат

- `proposal_class` (обычно `text_region`) — какие bbox берём из `core_object_detections`
- `min_det_score` / `box_threshold` upstream — сколько bbox дойдёт до OCR
- `max_boxes_per_frame` / `max_total_boxes` — cost control
- `engine`:
  - `ppocr_rec_onnx` (рекомендовано)
  - `tesseract` (fallback)
- `ppocr_img_h/ppocr_img_w` — размер входа recognizer (качество/скорость)
- `min_rec_score` — фильтр распознанного текста по confidence proxy

### Конфигурация render

```yaml
ocr_extractor:
  render:
    enable_render: true
    enable_html_render: true
```
---

## Навигация

[VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
