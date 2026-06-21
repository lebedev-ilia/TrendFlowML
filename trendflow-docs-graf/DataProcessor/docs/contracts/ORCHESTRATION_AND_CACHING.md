# Оркестрация, DAG, кэширование (полуфинал)

## 1) Где живёт orchestrator

Полуфинал:
- Оркестратор должен быть на уровне **DataProcessor** (единый DAG для Visual/Audio/Text).
- VisualProcessor может иметь внутренний запуск компонентов, но “истинный” граф и решения required/optional — на уровне DataProcessor.

Текущее состояние (baseline v0, референс):
- `DataProcessor/main.py` — реальный orchestrator “1 видео → 1 run”.
- `VisualProcessor/main.py` — внутренний исполнитель core/providers + modules.

## 2) Required vs optional (качество)

Гибридный режим:
- **required компоненты** (training schema) → **fail-fast**
- **optional компоненты** (аналитика/доп. фичи) → **best-effort**

Полуфинальное уточнение (Round 1 + Round 2):
- На этапе разработки/обучения (baseline+v2) мы целимся в **качество и предсказуемость**, поэтому:
  - по умолчанию **все включённые в профиль анализа компоненты считаются required** (fail-fast),
  - “best-effort”/optional допустим только если компонент явно помечен как optional в профиле анализа (см. `PRODUCT_CONTRACT.md`).

**Правило partial failures (Round 2)**:
- Если компонент упал, но он не участвует в prediction schema (не является required для прогноза), то:
  - run считается `status="ok"` (если все required компоненты успешны),
  - component_status="error" для упавшего компонента,
  - в UI показывается предупреждение о частичной ошибке,
  - prediction всё равно выдаётся (если required компоненты ok).

## 3) Idempotency ключ

Компонент должен быть идемпотентным по ключу:
- `(platform_id, video_id, component, config_hash, sampling_policy_version, producer_version, schema_version, model_signature)`

Где `model_signature` — функция от реально использованных моделей/версий/весов + engine/precision/device (см. `docs/models_docs/MODEL_SYSTEM_RULES.md`).

Если артефакт по ключу уже существует и валиден — пересчёт не делаем.

## 4) Задачи (Celery/очереди)

Рекомендация по этапам:
- старт: “1 видео = 1 job” (внутри job выполняем DAG по компонентам)
- масштабирование: “1 компонент = 1 task” (больше параллелизма, но сложнее кэш)

Параллелизм внутри одного видео (текущий baseline v0):
- Модули VisualProcessor могут выполняться параллельно.
- GPU-задачи ограничиваются `gpu_max_concurrent` (по умолчанию `auto` → 1 на малой VRAM).
- Manifest апдейтится последовательно из оркестратора, чтобы избежать гонок записи.

## 5) Artifact index (кэш по видео)

Нужен быстрый индекс (manifest/таблица) по ключу:
- `(platform_id, video_id, config_hash, sampling_policy_version, dataprocessor_version)`
→ ссылка на `latest_success_run_id` и артефакты.

Примечание:
- `dataprocessor_version` — версия кода пайплайна, а модельная часть кэша обеспечивается через `model_signature` на уровне компонентов (см. `docs/models_docs/MODEL_SYSTEM_RULES.md`).

Политика кэша “последние 10k видео” трактуется как **10k уникальных video_id** (heavy compute слой).

Полуфинальное уточнение (Round 1):
- Повторный запрос того же видео в проде должен учитывать “возраст анализа”.
- Рекомендуемая политика: если последний успешный run свежее `cache_ttl_days` (дефолт 3 дня, настраивается) — используем кэш, иначе пересчитываем.

## 6) Наблюдаемость

Минимум:
- timings per component
- GPU/CPU mem (если есть)
- status ok/empty/error
- причины empty (empty_reason)

Эти данные пишем в `manifest.json` и/или в БД.

## 7) Frames_dir retention

Полуфинал (Round 1):
- Union-кадры (`frames_dir`) храним **7 дней** для дебага/повторного рендера, затем удаляем.

## 8) Dep-lock (dependency locks) для параллельного исполнения процессоров (draft → FINAL for Audit v3)

### 8.1 Проблема

Целевое состояние: **AudioProcessor / TextProcessor / VisualProcessor** могут исполняться **параллельно** (в рамках одного `run_id`),
но между ними (и между компонентами внутри них) существуют зависимости.
Нельзя блокировать весь процессор только потому, что одному компоненту нужен артефакт, который ещё не готов.

### 8.2 Рекомендованная модель (правильная абстракция)

**FINAL (Audit v3)**: планирование выполняется **на уровне компонентов**, а не на уровне “процессор дошёл до точки”.

- Source-of-truth граф зависимостей: `docs/reference/component_graph.yaml` (узлы имеют `owner_processor`).
- Orchestrator/Scheduler поддерживает “ready queue”:
  - компонент становится runnable, когда выполнены **все hard deps** (`depends_on_components`);
  - компоненты без зависимостей разных процессоров могут стартовать параллельно.
- Процессор (Audio/Text/Visual) в проде становится по сути worker’ом, который берет runnable компоненты своего `owner_processor`
  и выполняет их (subprocess или native).

Почему так лучше:
- Dep-lock получается “из коробки” через DAG (не нужен polling в коде компонентов).
- Устраняется класс ошибок “TextProcessor дошёл до OCR слишком рано” — OCR‑зависимые экстракторы просто отдельные узлы DAG,
  которые будут запланированы позже.

### 8.3 Hard deps vs soft deps в dep-lock модели

- **hard dependency (`depends_on_components`)**:
  - scheduler **не стартует** компонент до тех пор, пока dependency‑компонент не перешел в terminal state
    (`success|empty|error|skipped`, в терминах state).
  - компонент сам выбирает политику: если ему нужен именно `success` (а `empty` не приемлем) — это должно быть задокументировано
    в README компонента и проверено (fail-fast).
- **soft dependency (`soft_dependencies`)**:
  - scheduler **не обязан ждать** (может стартовать компонент сразу),
  - компонент должен быть устойчив к отсутствию артефакта/данных и выдать валидный `empty`/degraded результат,
    либо (опционально) повторно выполнить под‑стадию после появления soft‑dep (только если это явно описано и не ломает идемпотентность).

### 8.4 Как именно “видим” готовность зависимостей (без гонок)

**FINAL (Audit v3)**: в параллельном режиме **не используем `manifest.json` как lock‑примитив**, потому что он multi-writer и
без доп. протокола может терять обновления при конкурентных upsert’ах.

Для dep-lock используем одно из:
- **State files (single-writer)**: `state/<platform>/<video>/<run>/state_<processor>.json`
  - Каждый процессор владеет своим файлом (один писатель) → нет гонок.
  - Orchestrator может агрегировать в `run_state.json` (Level‑2 snapshot).
- **Artifact existence + atomic writes**:
  - компонент пишет NPZ **атомарно** (`tmp → os.replace`) → читатели никогда не увидят partially-written NPZ.
  - готовность можно проверять через `storage.exists(npz_path)` + опционально `artifact_validator.validate_npz()`.
- **Journal events**: `state_events.jsonl` как append-only stream для UI/наблюдаемости (не как единственный lock).

### 8.5 Что фиксируем для текущего аудита

- `component_graph.yaml` остаётся source-of-truth для hard/soft deps.
- Пока baseline orchestrator исполняет процессоры последовательно, но **контракты компонентов** должны быть написаны так,
  чтобы при переходе к scheduler по компонентам ничего не пришлось перепридумывать.
- Для будущих TextProcessor ↔ VisualProcessor зависимостей (например OCR):
  - OCR‑зависимые экстракторы TextProcessor должны быть отдельными узлами DAG с `depends_on_components: [ocr_extractor]`
    (или `soft_dependencies`, если хотим best-effort).
---

## Навигация

[DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
