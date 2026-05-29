# DataProcessor Portfolio Normalization Plan

Этот документ фиксирует рабочий порядок приведения `DataProcessor` к портфолио- и production-ready уровню.
Цель: сделать проект понятным для найма и технического интервью, и одновременно чистым/надежным для последующего выпуска в прод, без потери текущей функциональности.

**Статус (2026-05-28): Waves 0–6 завершены.** Entry point: [../README.md](../README.md) · [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md)

**Phase 7 (post):** [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) · [reference/COMPONENT_GRAPH_INDEX.md](reference/COMPONENT_GRAPH_INDEX.md)

## Цели для портфолио

- Понятная архитектура: как проходит данные от входа до `result_store`.
- Предсказуемая структура папок и единый стиль документации.
- Явные контракты модулей: входы, выходы, зависимости, ограничения.
- Воспроизводимый запуск: quickstart, smoke, проверки артефактов.
- Чистый narrative: что уже продакшен-готово, что в работе, что legacy.

## Production-ready цели (равноприоритетно)

- Эксплуатационная предсказуемость: одинаковое поведение локально, в CI и в deployment.
- Fail-fast и явные ошибки вместо "тихих" fallback-сценариев.
- Наблюдаемость: метрики, логи, runbooks и понятная диагностика деградаций.
- Детерминизм артефактов и воспроизводимость run'ов.
- Ясные границы между source code, runtime-данными и внешними артефактами.

## Рабочий формат

Мы идем по папкам последовательно. На каждый шаг:

1. Инвентаризация (`что есть сейчас`).
2. Классификация (`core / active / legacy / generated / temporary`).
3. Нормализация структуры и имен.
4. Нормализация документации.
5. Фиксация результата в этом плане.

## Порядок прохода (waves)

### Wave 0 — базовая навигация и рамки (done)

Scope:
- `docs/`
- корневые входные точки (`main.py`, `api/`, `scripts/`)

DoD:
- Есть единый документ маршрута (этот файл).
- В `docs/MAIN_INDEX.md` есть ссылка на этот план.
- Зафиксированы правила статусов и приоритетов.

### Wave 1 — корень DataProcessor и архитектурный "скелет" (done)

Scope:
- корневые папки `DataProcessor/*` (только top-level)
- выделение `runtime/generated` директорий из "читаемой" структуры

DoD:
- Таблица "назначение папки -> владелец -> статус".
- Выделены папки, которые не должны восприниматься как source code (`dp_output`, `dp_results`, `state`, `_profiles_cache`, `__pycache__` и т.п.).
- Обновлен quick map для нового читателя.

### Wave 2 — AudioProcessor (крупный блок, done — docs/navigation)

Scope:
- `AudioProcessor/src/core`
- `AudioProcessor/src/extractors/*`
- `AudioProcessor/docs` и extractor-level docs

DoD:
- Единый шаблон extractor README/FEATURE docs.
- Четкая карта зависимостей extractors.
- Отдельно отмечены compute-heavy, model-heavy и optional компоненты.

### Wave 3 — TextProcessor (done — docs/navigation)

Scope:
- `TextProcessor/src/extractors/*`
- `TextProcessor/docs/*`

DoD:
- Единый шаблон описания экстракторов и схем.
- Убраны дубли и рассинхрон между README и реальным кодом.
- Понятный порядок запуска и валидации.

### Wave 4 — VisualProcessor (done — docs/navigation)

Scope:
- `VisualProcessor/core/*`
- `VisualProcessor/modules/*`
- связанная схема/валидация/доки

DoD:
- Единая карта core vs modules.
- Сверка contracts/schemas/docs для ключевых модулей.
- Ясно обозначены экспериментальные или нестабильные части.

### Wave 5 — API / orchestration / monitoring / tools (done)

Scope:
- `api/`, `dag/`, `monitoring/`, `tools/`, `scripts/`
- интеграционные runbooks

DoD:
- Нет разрозненных "параллельных" инструкций для запуска.
- У каждого operational сценария есть одна canonical doc entry.
- Скрипты классифицированы: setup, audit, migration, debug, one-off.

### Wave 6 — финальная упаковка для портфолио (done)

Scope:
- итоговые документы и narrative

DoD:
- `README`/index отвечают на вопросы "что это", "как запустить", "как проверять", "какие результаты".
- Есть раздел "ограничения и технический долг" (честный и инженерно зрелый).
- Есть checklist для демонстрации на собеседовании.

## Статусы задач

- `todo` — еще не брали в работу.
- `in_progress` — сейчас в работе.
- `blocked` — есть внешний блокер.
- `done` — закрыто и проверено.

## Трекер прогресса

- [x] `Wave 0`: создан управляющий план нормализации.
- [x] `Wave 0`: связать все ключевые индексы (`docs/MAIN_INDEX.md`, `docs/COMPONENTS_DESC.md`, `scripts/MAIN_INDEX.md`) через единый "старт здесь".
- [x] `Wave 1`: инвентаризация и классификация top-level папок `DataProcessor`.
- [x] `Wave 1`: каноничная навигация `docs/TOP_LEVEL_LAYOUT.md` (source / config / runtime / artifacts).
- [x] `Wave 2`: нормализация документации и навигации `AudioProcessor`.
- [x] `Wave 2`: `NORMALIZATION_WAVE2.md`, `EXTRACTOR_DEPENDENCIES.md`, MAIN_INDEX, doc trio 21/21.
- [x] `Wave 3`: нормализация документации и навигации `TextProcessor`.
- [x] `Wave 3`: `NORMALIZATION_WAVE3.md`, `EXTRACTOR_DEPENDENCIES.md`, doc layout 22/22.
- [x] `Wave 4`: нормализация документации и навигации `VisualProcessor`.
- [x] `Wave 4`: `NORMALIZATION_WAVE4.md`, `EXTRACTOR_DEPENDENCIES.md`, doc audit 29 components.
- [x] `Wave 5`: API / orchestration / monitoring / scripts (`NORMALIZATION_WAVE5.md`, `env.example`).
- [x] `Wave 6`: `DataProcessor/README.md`, `PORTFOLIO_INTERVIEW_GUIDE.md`.

Операционный журнал прогресса: `DataProcessor/docs/PORTFOLIO_PROGRESS_LOG.md`

## Принципы выполнения

- Не ломаем рабочий пайплайн ради "красоты" структуры.
- Сначала ясность и контракты, затем косметика.
- Любое перемещение файлов сопровождается обновлением ссылок в документах.
- Для больших блоков делаем маленькие итерации с измеримым результатом.
- Любое решение проверяем по двум осям: `portfolio clarity` и `production readiness`.
- Не добавляем "демо-only" решений, которые ухудшают прод-путь.

## Wave 1: черновая инвентаризация top-level

Ниже первичная классификация корневых папок `DataProcessor`. Это рабочая версия; уточняется по мере прохода.

### Source/Core (основной код)

- `AudioProcessor`
- `TextProcessor`
- `VisualProcessor`
- `Segmenter`
- `api`
- `common`
- `embedding_service`
- `dag`
- `dp_queue`
- `state`
- `storage`
- `qa`
- `tools`
- `scripts`
- `configs`
- `profiles`
- `docs`
- `monitoring`
- `docker`
- `triton`

### Infra/Models/Artifacts (инфраструктура и артефакты)

- `dp_models`
- `dp_triton`
- `faiss_indices`
- `wav2vec2_checkpoint`

### Runtime/Generated (не source code)

- `dp_output`
- `dp_results`
- `_profiles_cache`
- `__pycache__`

### Что делаем следующим шагом в Wave 1

- Для каждой top-level папки фиксируем роль, владельца и policy (`editable` / `generated` / `external artifact`).
- Выносим runtime-папки в отдельный раздел навигации, чтобы не мешались с кодом.
- Добавляем "куда смотреть новичку сначала" (5-7 директорий максимум).

## Wave 1: inventory + policy + action (v1)

### Как читать

- `policy=editable`: часть кодовой базы, редактируем в обычном потоке.
- `policy=generated`: runtime/generated данные; не считаются source code.
- `policy=artifact`: внешние модели/индексы/пакеты данных; управляются отдельно.

### Точки входа (files)

- `main.py` — orchestrator entrypoint, `policy=editable`, action: оставить как главный CLI/API старт.
- `docker-compose.yml` — локальный/интеграционный стек, `policy=editable`, action: позже сверить с runbooks в Wave 5.
- `env.example` — шаблон окружения, `policy=editable`, action: привести к минимальному canonical env-list (Wave 5).
- `pytest.ini` — тестовый baseline, `policy=editable`, action: сверить покрытие smoke/test путей (Wave 5).
- `requirements-api.txt`, `requirements-test.txt` — dependency boundaries, `policy=editable`, action: проверить на дубли и drift (Wave 5).

### Core source directories (owner/action)

- `AudioProcessor` — owner: audio pipeline, `policy=editable`, action: глубокая нормализация в Wave 2.
- `TextProcessor` — owner: text features pipeline, `policy=editable`, action: глубокая нормализация в Wave 3.
- `VisualProcessor` — owner: visual features pipeline, `policy=editable`, action: глубокая нормализация в Wave 4.
- `Segmenter` — owner: sampling/time-axis contract, `policy=editable`, action: выделить как фундаментальный upstream модуль.
- `api` — owner: service API layer, `policy=editable`, action: нормализация совместно с Wave 5.
- `embedding_service` — owner: semantic DB/index integration, `policy=editable`, action: проверить контракты с процессорами.
- `common` — owner: shared utilities/contracts, `policy=editable`, action: ограничить разрастание "misc" helper-кода.
- `configs` — owner: runtime/config contracts, `policy=editable`, action: разнести stable vs experiment конфиги.
- `scripts` — owner: operational tooling, `policy=editable`, action: классифицировать по назначению (Wave 5).
- `tools` — owner: diagnostics/audits utilities, `policy=editable`, action: объединить навигацию с `scripts`.
- `dag` — owner: orchestration graph/logics, `policy=editable`, action: зафиксировать текущий source-of-truth DAG path.
- `dp_queue` — owner: Celery integration, `policy=editable`, action: документировать контракт задач с backend (Wave 5).
- `state` — owner: run/processor state library, `policy=editable`, action: не путать с runtime JSON в storage (`state/<platform>/...`).
- `storage` — owner: FS/S3 abstraction, `policy=editable`, action: canonical entry — `storage/MAIN_INDEX.md`.
- `qa` — owner: feature QA helpers, `policy=editable`, action: связать с `tools/feature_qa_pipeline.py` и audit docs.
- `profiles` — owner: analysis profiles (YAML), `policy=config`, action: описать mapping backend ↔ `profiles/*.yaml`.
- `monitoring` — owner: observability stack, `policy=editable`, action: свести документы в единый runbook.
- `docs` — owner: project knowledge base, `policy=editable`, action: удерживать единый индекс и актуальный прогресс.
- `docker` — owner: container build/runtime assets, `policy=editable`, action: сверить с `docker-compose.yml`.
- `triton` — owner: inference deployment assets, `policy=editable`, action: документировать жизненный цикл моделей в Wave 5.

### Artifacts and model assets

- `dp_models` — `policy=artifact`, action: не смешивать с source code; поддерживать через model/runbook tooling.
- `dp_triton` — `policy=artifact`, action: считать deploy artifact зоной; описать versioning/promotion path.
- `faiss_indices` — `policy=artifact`, action: выделить в "managed data" с правилами обновления.
- `wav2vec2_checkpoint` — `policy=artifact`, action: документировать происхождение и воспроизводимость.

### Runtime/generated directories (must stay out of source navigation)

- `dp_output` — `policy=generated`, action: исключить из "читаемого кода", хранить по retention policy.
- `dp_results` — `policy=generated`, action: явно маркировать как runtime result store.
- `_profiles_cache` — `policy=generated`, action: cache-only зона; добавить в `.gitignore` при отсутствии.
- `__pycache__` — `policy=generated`, action: технический python cache.

### Новичку смотреть сначала (short path)

- `docs/MAIN_INDEX.md`
- `docs/contracts/CONTRACTS_OVERVIEW.md`
- `Segmenter`
- `AudioProcessor`
- `TextProcessor`
- `VisualProcessor`
- `api`

### Wave 1 status

- `done`: первичная классификация top-level entries.
- `done`: уточнены `qa`, `profiles`, `storage`, `dp_queue`, `state`.
- `done`: каноничная навигация — [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md).
- `next`: Wave 2 (`AudioProcessor`).
