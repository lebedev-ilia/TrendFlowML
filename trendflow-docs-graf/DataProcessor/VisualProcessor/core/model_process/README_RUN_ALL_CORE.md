# Запуск всех компонентов core с Triton

Скрипт `run_all_core_components.py` позволяет прогнать все компоненты core с использованием моделей Triton и сгенерировать HTML отчет с метриками производительности и демонстрацией качества.

## Использование

### Базовый запуск

```bash
python run_all_core_components.py \
    --frames-dir /path/to/frames_dir \
    --rs-path /path/to/result_store \
    --triton-http-url http://localhost:8000 \
    --out-dir /path/to/output \
    --batch-size 16
```

### Параметры

#### Обязательные параметры

- `--frames-dir`: Путь к директории с кадрами (должна содержать `metadata.json`)
- `--rs-path`: Путь к result_store (куда будут сохранены результаты компонентов)
- `--triton-http-url`: URL Triton сервера (например, `http://localhost:8000`)
- `--out-dir`: Директория для сохранения HTML отчета

#### Опциональные параметры

- `--batch-size`: Размер батча для компонентов (по умолчанию: 16)
- `--components`: Список компонентов для запуска (по умолчанию: все)
  - Доступные компоненты core: `core_clip`, `core_object_detections`, `core_optical_flow`, `core_depth_midas`, `core_face_landmarks`
  - Доступные компоненты identity: `brand_semantics`, `car_semantics`, `content_domain`, `franchise_recognition`, `place_semantics`
  - Другие: `ocr_extractor`
  - Примечание: `face_identity` исключен из списка по умолчанию (требует проверки production-версии)

#### Параметры для переопределения моделей Triton

- `--triton-image-model-name`: Имя модели Triton для image encoder в `core_clip` (по умолчанию: `clip_image_224`)
- `--triton-text-model-name`: Имя модели Triton для text encoder в `core_clip` (по умолчанию: `clip_text`)
- `--triton-image-model-spec`: Spec имя модели через ModelManager для image encoder
- `--triton-text-model-spec`: Spec имя модели через ModelManager для text encoder
- `--triton-flow-model-name`: Имя модели Triton для `core_optical_flow` (по умолчанию: `raft_256`)
- `--triton-depth-model-name`: Имя модели Triton для `core_depth_midas` (по умолчанию: `midas_256`)
- `--detection-model-path`: Путь к файлу модели YOLO для `core_object_detections` (по умолчанию: `visual/yolo/yolo11x_41_best.pt`, разрешается через `DP_MODELS_ROOT`)
- `--detection-device`: Устройство для `core_object_detections` (по умолчанию: `auto` - использует cuda если доступен, иначе cpu)

### Примеры

#### Запуск только определенных компонентов

```bash
python run_all_core_components.py \
    --frames-dir /path/to/frames_dir \
    --rs-path /path/to/result_store \
    --triton-http-url http://localhost:8000 \
    --out-dir /path/to/output \
    --batch-size 16 \
    --components core_clip core_object_detections
```

#### Использование ONNX моделей

```bash
python run_all_core_components.py \
    --frames-dir /path/to/frames_dir \
    --rs-path /path/to/result_store \
    --triton-http-url http://localhost:8000 \
    --out-dir /path/to/output \
    --batch-size 16 \
    --triton-image-model-name clip_image_224_onnx \
    --triton-text-model-name clip_text_onnx \
    --triton-flow-model-name raft_256_onnx \
    --triton-depth-model-name midas_256_onnx
```

## Структура отчета

HTML отчет содержит:

1. **Сводка (Summary)**:
   - Общее время выполнения
   - Пиковая память RAM и VRAM
   - Дельта памяти с момента запуска
   - Количество успешных/неуспешных компонентов

2. **Таблица результатов компонентов**:
   - Статус выполнения
   - Время выполнения каждого компонента
   - Пиковая память RAM/VRAM для каждого компонента
   - Дельта памяти для каждого компонента

3. **Демонстрация качества**:
   - Для `core_object_detections`: кадры с боксами и классами
   - Для `core_face_landmarks`: кадры с landmarks лиц, позы и рук
   - Для `core_clip`: визуализация обработанных кадров
   - Для `core_depth_midas`: визуализация карт глубины
   - Для `core_optical_flow`: статистика по оптическому потоку

## Требования

- Python 3.8+
- Доступ к Triton Inference Server
- Все необходимые зависимости для компонентов core
- `nvidia-smi` для мониторинга VRAM (опционально)

## Модели и Runtime

### Triton модели

Скрипт по умолчанию использует следующие модели Triton:

- `clip_image_224` - для image embeddings в `core_clip`
- `clip_text` - для text embeddings в `core_clip`
- `raft_256` - для оптического потока в `core_optical_flow`
- `midas_256` - для оценки глубины в `core_depth_midas`

### Локальные модели (не Triton)

- `core_object_detections` использует **ultralytics runtime** (не Triton) с моделью YOLO `yolo11x_41_best.pt`
- `core_face_landmarks` использует **MediaPipe** локально (не Triton)
- `ocr_extractor` использует **tesseract CLI** локально (не Triton)

### Компоненты с Embedding Service

- `brand_semantics` - распознавание брендов/логотипов через Embedding Service
- `car_semantics` - распознавание автомобилей через Embedding Service
- `face_identity` - идентификация известных людей через Embedding Service

### Компоненты с Triton через ModelManager

- `content_domain` - классификация домена контента (игра/аниме/мультфильм и т.д.) через CLIP text encoder
- `franchise_recognition` - распознавание франшиз/названий через CLIP text encoder
- `place_semantics` - распознавание мест через core_clip embeddings

## Примечания

- Скрипт мониторит использование памяти в реальном времени
- Для каждого компонента собираются метрики до и после выполнения
- Quality reports для `core_object_detections` и `core_face_landmarks` генерируются автоматически
- Для остальных компонентов создается простая визуализация результатов
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
