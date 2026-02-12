# TextProcessor

TextProcessor — процессор текстовой модальности. Он извлекает табличные **snapshot‑фичи** из доступного текстового входа (title/description/transcript/comments) и сохраняет результат в **per‑run `result_store`**.

## Контракт входа

- **Единица обработки (CLI, single)**: один `VideoDocument` (JSON) через `--input-json`.
- **Единица обработки (CLI, batch)**: несколько `VideoDocument` через `--text-input-dir` или `--text-input-json-list`.
- **Единица обработки (Python API)**: один или несколько `VideoDocument` (см. `MainProcessor.run_batch()`).
- **Источник**: upstream (backend/ingestion) формирует `VideoDocument` и передаёт путь в CLI.
- **Требования (no‑fallback)**:
  - Если TextProcessor включён в профиль как required, отсутствие входного JSON или ошибка парсинга → run должен падать на уровне orchestrator.
  - Модели/эмбеддеры должны грузиться **только локально** (no‑network policy).

### Формат `VideoDocument`

См. `TextProcessor/src/schemas/models.py`.

Ключевые поля:
- `title: str`
- `description: str`
- `transcripts: Dict[str, str]` (например `{"whisper": "...", "youtube_auto": "..."}`)
- `transcripts_token_ids: Dict[str, List[int]]` (опционально; предпочтительно вместо raw transcript)
- `comments: List[{text: str}]` (опционально)

## Контракт выхода (result_store)

TextProcessor пишет **один NPZ артефакт**:

- `result_store/<platform_id>/<video_id>/<run_id>/text_processor/text_features.npz`

и апдейтит:

- `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`

### NPZ schema

Схема: `schema_version="text_npz_v1"` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Минимальные ключи:
- `feature_names: object[str]`
- `feature_values: float32[]`
- `payload: object(dict)` — **privacy‑safe summary**, без raw текста по умолчанию
- `meta: object(dict)` — run identity + версии + статус

Если включены embeddings (`--enable-embeddings`), TextProcessor также сохраняет privacy‑safe “primary embedding” для downstream сравнения:
- `primary_embedding: float32[D]` (вектор)
- `primary_embedding_present: bool`
- `primary_embedding_source: str` (basename файла источника)
- `primary_embedding_model: str` (best-effort)

### Privacy / raw текст

По умолчанию TextProcessor **не сохраняет raw текст** в NPZ.

Для локального дебага есть флаг:
- `--store-raw-payload` → пишет raw payload в `result_store/.../_tmp_text/raw_payload.json` (не source‑of‑truth).

## Запуск (CLI)

### Single-document режим

Standalone (в локальный `_runs/result_store`):

```bash
python3 TextProcessor/run_cli.py \
  --input-json /path/to/video_document.json \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id> \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown
```

Embedding‑ветка (тяжелее, может требовать GPU):

```bash
python3 TextProcessor/run_cli.py \
  --input-json /path/to/video_document.json \
  --enable-embeddings
```

### Batch режим

Обработка всех `.json` файлов в директории:

```bash
python3 TextProcessor/run_cli.py \
  --input-dir /path/to/documents \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id>
```

Обработка конкретного списка файлов:

```bash
python3 TextProcessor/run_cli.py \
  --input-json-list doc1.json,doc2.json,doc3.json \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id>
```

Через верхний оркестратор (`DataProcessor/main.py`):

```bash
python3 main.py \
  --run-text \
  --text-input-dir /path/to/documents \
  --global-config configs/global_config.yaml \
  --output dp_output \
  --rs-base dp_results
```

**Batch processing флаги**:
- `--batch-max-workers <N>`: количество параллельных воркеров для CPU extractors (по умолчанию: auto)
- `--no-batch-gpu`: отключить GPU batching (обрабатывать GPU extractors последовательно)
- `--no-batch-cpu-parallel`: отключить CPU параллелизм (обрабатывать CPU extractors последовательно)

**Дополнительные флаги**:
- `--include-primary-embedding` / `--no-include-primary-embedding`: контроль включения primary_embedding в NPZ (default: True)
- `--log-dir <path>`: директория для structured logs (default: `<run_rs_path>/_logs/`)

## Orchestrator (MainProcessor)

### Batch API (Python)

Начиная со Stage-0/1 рефакторинга под multi-document, `MainProcessor` поддерживает:

- `run(doc)`: обработка одного документа
- `run_batch([doc1, doc2, ...])`: обработка нескольких документов

**Важно (Stage-1, изоляция артефактов)**:
- В `run_batch()` каждый документ получает **свой** `artifacts_dir` внутри базового `_artifacts/` (подпапки вида `doc_00000/`, `doc_00001/`, ...),
  чтобы fixed-name `.npy` (например, `title_embedding.npy`, `transcript_{source}_agg_mean.npy`) не конфликтовали при обработке нескольких документов.
- `doc.tp_artifacts` сбрасывается для каждого документа на старте обработки (изоляция in-memory реестра).

**Stage-2/3 (GPU batching)**:
- `TitleEmbedder`, `HashtagEmbedder`, `TranscriptChunkEmbedder`, `CommentsEmbedder` реализуют `extract_batch()` для батчевого кодирования.
- Для сохранения совместимого layout они используют per-doc директорию из `doc._tp_artifacts_dir` (заполняется orchestrator'ом в `run_batch()`).

**Stage-4 (CPU parallelism + граф зависимостей)**:
- `run_batch()` использует граф зависимостей для группировки extractors по уровням.
- GPU extractors с `supports_batch=True` обрабатываются батчем для всех документов одновременно.
- CPU extractors обрабатываются параллельно через `ThreadPoolExecutor` внутри каждого уровня зависимостей.
- Параметры: `max_workers`, `enable_gpu_batching`, `enable_cpu_parallel` (можно задать в `__init__()` или передать в `run_batch()`).

**Конфигурация через global_config.yaml**:
```yaml
text:
  batch_processing:
    enabled: true
    max_workers: null  # auto (использует os.cpu_count())
    enable_gpu_batching: true
    enable_cpu_parallel: true
```

**CLI batch режим** (через `DataProcessor/main.py` или `TextProcessor/run_cli.py`):
```bash
# Обработка всех .json файлов в директории
python3 main.py --run-text --text-input-dir /path/to/documents --global-config configs/global_config.yaml

# Обработка конкретного списка файлов
python3 main.py --run-text --text-input-json-list doc1.json,doc2.json,doc3.json --global-config configs/global_config.yaml

# Тонкая настройка через CLI флаги
python3 main.py --run-text --text-input-dir /path/to/documents \
  --batch-max-workers 8 \
  --no-batch-gpu \  # отключить GPU batching
  --no-batch-cpu-parallel  # отключить CPU параллелизм
```

**Структура результатов в batch режиме**:
Каждый документ сохраняется в отдельную директорию:
```
{rs_base}/youtube/{video_id}/{doc_name}/{config_hash}/text_processor/text_features.npz
```

**Production результаты** (6 документов, полный набор extractors):
- Время обработки: 94.79s (15.80s/doc)
- Все документы успешно обработаны и валидированы

**Error handling**:
- Каждый extractor обёрнут в try/except, ошибки собираются в `errors_by_extractor`
- Required extractors (через `required_extractors` параметр) fail-fast при ошибках
- Optional extractors логируют warning и продолжают run

**Status aggregation**:
- Статусы extractors (`ok`/`empty`/`error`) собираются и агрегируются
- Если все extractors `empty` → orchestrator `status="empty"`
- Если required extractor `error` → orchestrator `status="error"` (fail-fast)

**Features_flat merge**:
- Last-wins merge с обнаружением конфликтов (дубликаты ключей логируются)
- Метрика `tp_orchestrator_feature_conflicts_count` в `features_flat`

**Models_used collection**:
- Автоматический сбор `models_used` из всех extractors (вместо хардкода)
- Мержится в `features["models_used"]` для NPZ `meta`

**Orchestrator metrics** (в `features_flat`):
- `tp_orchestrator_total_extractors`: общее количество extractors
- `tp_orchestrator_successful_count`: количество успешных
- `tp_orchestrator_failed_count`: количество failed
- `tp_orchestrator_empty_count`: количество empty
- `tp_orchestrator_total_duration_ms`: общее время выполнения (ms)
- `tp_orchestrator_feature_conflicts_count`: количество конфликтов

**NPZ validation**:
- NPZ валидируется **перед** атомарной записью (не пишем невалидные артефакты)
- Если валидация не прошла → `status="error"`, файл не пишется

**Structured logging**:
- Логирование ошибок/empty/конфликтов без PII
- Логи в `_logs/text_processor.log` (если `--log-dir` задан или stderr не TTY)

## UI Render (Human-friendly визуализация)

TextProcessor генерирует **render-context JSON** для визуализации и LLM:

- `result_store/.../text_processor/_render/render_context.json`

Render-context содержит:
- **Summary**: общая статистика (количество фич, статус, версии)
- **Features**: все фичи из NPZ (privacy-safe, без raw текста)
- **Extractors**: детализированные render-context'ы для каждого extractor'а (если есть render.py)

### Глобальный renderer

Модуль: `src/core/renderer.py`
- Функция: `render_text_processor(npz_path, output_dir)` — генерирует render-context для всего TextProcessor
- Автоматически группирует фичи по extractor'ам (по префиксам `tp_<extractor>_*`)
- Динамически загружает per-extractor renderer'ы из `src/extractors/<name>/render.py`

### Per-extractor renderer'ы

Каждый extractor может иметь свой файл `render.py` с функцией:
```python
def render_<extractor_name>(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Генерировать render-context для extractor'а."""
    return {
        "component": "<extractor_name>",
        "summary": {...},
        ...
    }
```

Пример: `src/extractors/lexico_static_features/render.py`

Render-context может быть использован:
- LLM для генерации текстовых описаний (см. `docs/contracts/LLM_RENDERING.md`)
- Frontend для визуализаций (статистики, распределения, метрики)

**Документация**:
- `docs/RENDERER_GUIDE.md` - руководство по созданию renderer'ов
- `docs/RENDERER_CHECKLIST.md` - чеклист требований и шаблоны для приведения extractors к production-ready состоянию

## Политика моделей (ModelManager)

TextProcessor использует единый `ModelManager` (`dp_models`) и работает в режиме **no‑network**:

- модели SentenceTransformers должны быть описаны в `dp_models/spec_catalog/text/*.yaml`
- артефакты должны лежать в `DP_MODELS_ROOT` (см. `env.example`)

Текущий локальный embedding‑модельный дефолт: `sentence-transformers/all-MiniLM-L6-v2`.


