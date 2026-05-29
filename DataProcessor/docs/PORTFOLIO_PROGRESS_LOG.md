# DataProcessor Portfolio + Production Progress Log

Этот журнал фиксирует фактический прогресс нормализации `DataProcessor`.
Формат записи: дата, этап, что сделано, артефакты, следующий шаг.

**Полный итог сессии:** [PORTFOLIO_SESSION_SUMMARY_2026-05-29.md](PORTFOLIO_SESSION_SUMMARY_2026-05-29.md)

## 2026-05-28 — старт нормализации

### Entry 001 — foundation

- Stage: `Wave 0`
- Status: `done`
- Сделано:
  - создан маршрут нормализации `PORTFOLIO_NORMALIZATION_PLAN.md`
  - добавлена единая точка входа в `docs/MAIN_INDEX.md`, `docs/COMPONENTS_DESC.md`, `scripts/MAIN_INDEX.md`
  - зафиксированы двойные цели: `portfolio clarity` + `production readiness`
- Артефакты:
  - `DataProcessor/docs/PORTFOLIO_NORMALIZATION_PLAN.md`
  - `DataProcessor/docs/MAIN_INDEX.md`
  - `DataProcessor/docs/COMPONENTS_DESC.md`
  - `DataProcessor/scripts/MAIN_INDEX.md`
- Next:
  - начать `Wave 1` с формальной inventory/policy/action классификации top-level.

### Entry 002 — top-level inventory v1

- Stage: `Wave 1`
- Status: `in_progress`
- Сделано:
  - выполнена первичная инвентаризация top-level entries в `DataProcessor`
  - добавлены `policy` категории: `editable`, `generated`, `artifact`
  - для core/artifact/runtime директорий добавлены action-решения
  - зафиксирован "short path" для нового инженера
- Артефакты:
  - `DataProcessor/docs/PORTFOLIO_NORMALIZATION_PLAN.md` (секция `Wave 1: inventory + policy + action (v1)`)
- Next:
  - уточнить владельцев и границы для `qa`, `profiles`, `storage`
  - создать в документации отдельный navigation-блок для runtime/artifact директорий
  - после закрытия Wave 1 перейти к Wave 2 (`AudioProcessor`)

### Entry 003 — Wave 1 closed

- Stage: `Wave 1`
- Status: `done`
- Сделано:
  - уточнены границы: `qa` (QA helpers), `profiles` (YAML config), `storage` (FS/S3 lib + MAIN_INDEX)
  - исправлена классификация: `dp_queue`, `state` — source code, не runtime
  - создан [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md) — каноничная навигация top-level
  - в `.gitignore` добавлен `_profiles_cache/` (runtime cache)
- Артефакты:
  - `DataProcessor/docs/TOP_LEVEL_LAYOUT.md`
  - `DataProcessor/docs/PORTFOLIO_NORMALIZATION_PLAN.md` (Wave 1 → done)
  - `.gitignore`
- Next:
  - `Wave 2`: инвентаризация `AudioProcessor` (core + extractors + docs).

### Entry 004 — Wave 2 AudioProcessor started

- Stage: `Wave 2`
- Status: `in_progress`
- Сделано:
  - инвентаризация 21 extractors и структуры `AudioProcessor/`
  - создан [NORMALIZATION_WAVE2.md](../AudioProcessor/docs/NORMALIZATION_WAVE2.md)
  - зафиксированы проблемы: 6 дублей `FEATURE_DESCRIPTION.md`, рассинхрон ссылок в MAIN_INDEX
  - категоризация extractors (tier-0, model-heavy, spectral, speech)
- Артефакты:
  - `DataProcessor/AudioProcessor/docs/NORMALIZATION_WAVE2.md`
  - `DataProcessor/AudioProcessor/docs/README.md` (ссылка на Wave 2)
- Next:
  - исправить ссылки в `AudioProcessor/docs/MAIN_INDEX.md`
  - убрать дубли `FEATURE_DESCRIPTION.md` (6 extractors)

### Entry 005 — AudioProcessor doc links + FEATURE_DESCRIPTION stubs

- Stage: `Wave 2`
- Status: `in_progress`
- Сделано:
  - исправлены битые ссылки в `AudioProcessor/docs/MAIN_INDEX.md` (hpss, mfcc, quality)
  - 6 корневых `FEATURE_DESCRIPTION.md` заменены на stub → `docs/FEATURE_DESCRIPTION.md`
  - зафиксировано: stub vs docs содержимое ранее различалось (merge при необходимости отдельно)
- Артефакты:
  - `AudioProcessor/docs/MAIN_INDEX.md`
  - `AudioProcessor/src/extractors/*/FEATURE_DESCRIPTION.md` (6 stubs)
- Next:
  - единообразить текст ссылок в MAIN_INDEX (все → `docs/README.md`)
  - dependency map extractors
  - prod smoke checklist в NORMALIZATION_WAVE2

### Entry 006 — dependency map + smoke checklist + MAIN_INDEX links

- Stage: `Wave 2`
- Status: `in_progress`
- Сделано:
  - создан [EXTRACTOR_DEPENDENCIES.md](../AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md):
    - Segmenter families → extractors
    - optional shared_features deps
    - speech_analysis conditional deps
    - mermaid flow, prod smoke checklist (7 шагов)
  - унифицированы ссылки в `AudioProcessor/docs/MAIN_INDEX.md` → `docs/README.md`
  - ссылки в `AudioProcessor/README.md`, `AudioProcessor/docs/README.md`
- Артефакты:
  - `AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md`
  - `AudioProcessor/docs/MAIN_INDEX.md`
  - `AudioProcessor/README.md`
- Next:
  - аудит наличия SCHEMA.md у всех 21 extractors
  - закрыть Wave 2, старт Wave 3 (TextProcessor)

### Entry 007 — Wave 2 doc layout verified

- Stage: `Wave 2`
- Status: `done` (документация и навигация; кодовой рефакторинг extractors — отдельно)
- Сделано:
  - проверено: все 21 extractor имеют `docs/{README,SCHEMA,FEATURE_DESCRIPTION}.md`
  - DoD Wave 2 по документации закрыт
- Next:
  - `Wave 3`: TextProcessor — аналогичный проход (inventory, deps, docs)

### Entry 008 — Wave 3 TextProcessor started

- Stage: `Wave 3`
- Status: `in_progress`
- Сделано:
  - создан стартовый план [NORMALIZATION_WAVE3.md](../TextProcessor/docs/NORMALIZATION_WAVE3.md)
  - зафиксировано: 22 extractors, сильная база audit_v3 docs
- Next:
  - doc trio audit для 22 extractors
  - EXTRACTOR_DEPENDENCIES + smoke checklist (TextProcessor)

### Entry 009 — Wave 3 TextProcessor closed

- Stage: `Wave 3`
- Status: `done`
- Сделано:
  - создан [EXTRACTOR_DEPENDENCIES.md](../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md): tier 1–22, ASR/tags/diarization, corpus packs, smoke checklist
  - проверено: 22/22 extractors — `README.md`, `SCHEMA.md`, `docs/FEATURE_DESCRIPTION.md`
  - зафиксирован canonical layout Text (отличается от Audio: README в корне extractor)
  - [NORMALIZATION_WAVE3.md](../TextProcessor/docs/NORMALIZATION_WAVE3.md) → done
- Артефакты:
  - `TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md`
  - `TextProcessor/docs/NORMALIZATION_WAVE3.md`
  - `TextProcessor/docs/MAIN_INDEX.md`
- Next:
  - `Wave 4`: VisualProcessor

### Entry 010 — Wave 4 VisualProcessor started

- Stage: `Wave 4`
- Status: `in_progress`
- Сделано:
  - inventory v1: core providers (clip, depth, od, flow, ocr, identity/*) + 17 modules
  - создан [NORMALIZATION_WAVE4.md](../VisualProcessor/docs/NORMALIZATION_WAVE4.md)
  - runtime policy: `VisualProcessor/result_store`, `VisualProcessor/state` ≠ source
  - ссылки в `VisualProcessor/README.md`, `docs/MAIN_INDEX.md`
- Артефакты:
  - `VisualProcessor/docs/NORMALIZATION_WAVE4.md`
- Next:
  - doc coverage scan core + modules
  - `VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md`

### Entry 011 — Wave 4 VisualProcessor closed

- Stage: `Wave 4`
- Status: `done`
- Сделано:
  - doc scan: 6 core + 6 identity + 17 modules (README/SCHEMA/FEATURE)
  - создан [EXTRACTOR_DEPENDENCIES.md](../VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md): baseline DAG, extended modules, prod checklist
  - исключение: `failing_module` (test), `face_identity` → `docs/SCHEMA.md`
  - [NORMALIZATION_WAVE4.md](../VisualProcessor/docs/NORMALIZATION_WAVE4.md) → done
- Артефакты:
  - `VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md`
  - `VisualProcessor/docs/NORMALIZATION_WAVE4.md`
- Next:
  - `Wave 5`: API, monitoring, scripts classification

### Entry 012 — Wave 5 started

- Stage: `Wave 5`
- Status: `in_progress`
- Сделано:
  - inventory `api/`, `dag/`, `monitoring/`, `scripts/`
  - создан [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md): canonical entry points, script classes, prod checklist
  - ссылка в `docs/MAIN_INDEX.md`
- Next:
  - Wave 6: portfolio narrative + interview checklist

### Entry 013 — env.example cleanup + drift table

- Stage: `Wave 5`
- Status: `in_progress`
- Сделано:
  - `env.example`: убран дубль `DP_MODELS_ROOT`, локальный путь; добавлены `TREND_STORAGE_*`, ссылка на API docs
  - в NORMALIZATION_WAVE5 §9 — таблица drift API storage vs storage adapter
- Next:
  - закрыть Wave 5, начать Wave 6 (PORTFOLIO_README / interview checklist)

### Entry 014 — Wave 6 portfolio pack

- Stage: `Wave 6`
- Status: `done`
- Сделано:
  - создан [DataProcessor/README.md](../README.md) — entry point: архитектура, quickstart, масштаб, принципы
  - создан [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md) — demo flow, checklist, Q&A, tech debt
  - обновлены `docs/MAIN_INDEX.md`, repo `docs/MAIN_INDEX.md`
  - Wave 5 закрыт (`NORMALIZATION_WAVE5.md` → done)
- Артефакты:
  - `DataProcessor/README.md`
  - `DataProcessor/docs/PORTFOLIO_INTERVIEW_GUIDE.md`
- Итог:
  - Waves 0–6 документационной нормализации **закрыты** (см. PORTFOLIO_NORMALIZATION_PLAN.md)

## 2026-05-29 — Phase 7: demo + DAG

### Entry 015 — component_graph baseline extended + demo runbook

- Stage: `Phase 7` (post-normalization)
- Status: `done`
- Сделано:
  - расширен `docs/reference/component_graph.yaml` (baseline): +15 visual modules/identity heads
  - создан [COMPONENT_GRAPH_INDEX.md](reference/COMPONENT_GRAPH_INDEX.md)
  - создан [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) — Demo A–E
  - валидация DAG: baseline 33 nodes, audio_extended 6, topo OK
- Артефакты:
  - `docs/reference/component_graph.yaml`
  - `docs/PORTFOLIO_DEMO_RUNBOOK.md`
- Next:
  - stage `text_processor_full` в component_graph (22 text extractors)
  - `configs/portfolio_demo.yaml` — единый лёгкий профиль для Demo D

### Entry 016 — итог сессии (финальная фиксация)

- Stage: `Session summary`
- Status: `done`
- Сделано:
  - сводный отчёт [PORTFOLIO_SESSION_SUMMARY_2026-05-29.md](PORTFOLIO_SESSION_SUMMARY_2026-05-29.md)
  - задокументированы все Waves 0–6, Phase 7, файлы, метрики, backlog
- Итог для пользователя:
  - **Документационно и для портфолио — готово**
  - **Runtime — рекомендуется один smoke по DEMO_RUNBOOK**
  - **Git — commit не выполнялся в сессии**
