# DataProcessor — итог сессии нормализации (2026-05-28 / 2026-05-29)

Полный отчёт о работе по приведению `DataProcessor` к виду для **портфолио** и **production-ready** документации.  
Журнал по шагам: [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md)

---

## 1. Цели сессии

1. **Портфолио** — понятная навигация, narrative для собеседования.
2. **Production** — честные контракты, ops-путь, без «демо-костылей» в документации.
3. **Процесс** — последовательный проход по папкам с фиксацией прогресса.

Принцип: любое решение оценивалось по двум осям — `portfolio clarity` и `production readiness`.

---

## 2. Созданные документы (новые файлы)

| Файл | Назначение |
|------|------------|
| [PORTFOLIO_NORMALIZATION_PLAN.md](PORTFOLIO_NORMALIZATION_PLAN.md) | Маршрут Waves 0–6, DoD, трекер |
| [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md) | Операционный журнал (Entries 001–016) |
| [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md) | Карта корня: source / config / runtime / artifacts |
| [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md) | Pitch, demo flow, Q&A, tech debt |
| [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) | Сценарии Demo A–E, команды, troubleshooting |
| [../README.md](../README.md) | Главный entry point DataProcessor |
| [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md) | API, monitoring, scripts, env drift |
| [reference/COMPONENT_GRAPH_INDEX.md](reference/COMPONENT_GRAPH_INDEX.md) | Индекс stages DAG + валидация |

### AudioProcessor

| Файл | Назначение |
|------|------------|
| [../AudioProcessor/docs/NORMALIZATION_WAVE2.md](../AudioProcessor/docs/NORMALIZATION_WAVE2.md) | Wave 2, DoD, inventory |
| [../AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md) | 21 extractor, Segmenter families, smoke checklist |

### TextProcessor

| Файл | Назначение |
|------|------------|
| [../TextProcessor/docs/NORMALIZATION_WAVE3.md](../TextProcessor/docs/NORMALIZATION_WAVE3.md) | Wave 3, doc layout |
| [../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md) | 22 extractor, tier order, ASR/tags |

### VisualProcessor

| Файл | Назначение |
|------|------------|
| [../VisualProcessor/docs/NORMALIZATION_WAVE4.md](../VisualProcessor/docs/NORMALIZATION_WAVE4.md) | Wave 4, core vs modules |
| [../VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md) | Baseline DAG, 29 components, prod checklist |

---

## 3. Изменённые файлы (ключевые)

| Файл | Суть изменения |
|------|----------------|
| [MAIN_INDEX.md](MAIN_INDEX.md) | Ссылки на portfolio, demo, DAG, Wave 5 |
| [COMPONENTS_DESC.md](COMPONENTS_DESC.md) | Ссылка на план нормализации |
| [../scripts/MAIN_INDEX.md](../scripts/MAIN_INDEX.md) | Ссылка на план |
| [reference/component_graph.yaml](reference/component_graph.yaml) | +15 visual nodes в baseline; segmenter в audio_extended |
| [../env.example](../env.example) | Убран дубль `DP_MODELS_ROOT`, добавлен `TREND_STORAGE_*`, ссылка на API env docs |
| [../../.gitignore](../../.gitignore) | `_profiles_cache/` |
| [../../docs/MAIN_INDEX.md](../../docs/MAIN_INDEX.md) | Ссылки на DataProcessor README и interview guide |
| `AudioProcessor/docs/MAIN_INDEX.md` | Исправлены ссылки на `docs/README.md` |
| `AudioProcessor/README.md` | Ссылки на Wave 2 |
| 6× `AudioProcessor/.../FEATURE_DESCRIPTION.md` (корень) | Stub → canonical `docs/FEATURE_DESCRIPTION.md` |
| `TextProcessor/docs/MAIN_INDEX.md` | Wave 3 + EXTRACTOR_DEPENDENCIES |
| `VisualProcessor/README.md`, `docs/MAIN_INDEX.md` | Wave 4 + deps |

---

## 4. Waves — что сделано

### Wave 0 — рамки

- Единый план нормализации.
- Ссылки «старт здесь» в `docs/MAIN_INDEX.md`, `COMPONENTS_DESC.md`, `scripts/MAIN_INDEX.md`.
- Двойная цель: portfolio + production.

### Wave 1 — корень DataProcessor

- Инвентаризация top-level папок.
- [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md).
- **Исправления классификации:**
  - `dp_queue/`, `state/` — **source code** (не runtime).
  - `profiles/` — **config** (не artifact).
  - `qa/`, `storage/` — уточнены роли.

### Wave 2 — AudioProcessor

- 21 extractor, карта зависимостей.
- Проверка doc trio: `docs/README.md` + `docs/SCHEMA.md` + `docs/FEATURE_DESCRIPTION.md`.
- Исправлены битые ссылки в `docs/MAIN_INDEX.md`.
- 6 дублей `FEATURE_DESCRIPTION.md` в корне → stub на `docs/`.

### Wave 3 — TextProcessor

- 22 extractor, tier 0–3, ASR обязателен в audit.
- Canonical layout: `README.md` + `SCHEMA.md` + `docs/FEATURE_DESCRIPTION.md` (отличается от Audio).

### Wave 4 — VisualProcessor

- Inventory: 6 core + 6 identity + 17 modules.
- Doc audit: 29/29 (кроме `failing_module` без FEATURE — test utility).
- Runtime: `VisualProcessor/result_store`, `state/` — не source.

### Wave 5 — API / ops

- [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md): canonical entry points, классы scripts.
- `env.example` очищен; таблица env drift (API vs storage adapter).

### Wave 6 — portfolio pack

- [../README.md](../README.md).
- [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md).

### Phase 7 — DAG + demo

- `component_graph.yaml`: baseline **33** узла (+15 visual).
- `audio_extended`: +`segmenter` для валидатора (6 узлов).
- [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md).
- [COMPONENT_GRAPH_INDEX.md](reference/COMPONENT_GRAPH_INDEX.md).

---

## 5. Метрики (масштаб проекта)

| Компонент | Количество |
|-----------|------------|
| Audio extractors | 21 |
| Text extractors | 22 |
| Visual components | 29 |
| **Итого feature-компонентов** | **~72** |
| Новых/обновлённых doc-файлов в сессии | ~20 новых + ~15 изменённых |
| DAG baseline nodes | 33 |

---

## 6. Что НЕ делалось в этой сессии

- Массовый рефакторинг кода extractors.
- Прогоны smoke/E2E на машине пользователя.
- Git commit / push.
- Полный DAG для всех 22 text extractors (`text_processor_full` stage).
- Унификация Text README в `docs/` (зафиксирован текущий layout как canonical).
- Сокращение `COMPONENTS_DESC.md` (~5k строк).
- `configs/portfolio_demo.yaml` (единый лёгкий профиль) — в backlog.

---

## 7. Готовность к демонстрации (без тестов)

| Критерий | Статус |
|----------|--------|
| Понятная точка входа | ✅ `DataProcessor/README.md` |
| Архитектура и зависимости | ✅ TOP_LEVEL + 3× EXTRACTOR_DEPENDENCIES |
| Собеседование | ✅ PORTFOLIO_INTERVIEW_GUIDE |
| Живое демо (инструкции) | ✅ PORTFOLIO_DEMO_RUNBOOK |
| Prod narrative | ✅ Wave 5, tech debt честно |
| Runtime «всё зелёное» | ⚠️ нужен smoke на локальной машине |
| GitHub | ⚠️ изменения не закоммичены |

---

## 8. Backlog (следующие шаги)

1. **Commit** всех doc-изменений в одну ветку (`system-testing` / feature branch).
2. **Smoke** Demo B (audio 21/21) или Demo A (visual) — 30–60 мин.
3. `text_processor_full` в `component_graph.yaml`.
4. `configs/portfolio_demo.yaml` для Demo D.
5. Дедупликация `docs/MAIN_INDEX.md` (двойные блоки API architecture).
6. По желанию: сократить `COMPONENTS_DESC.md` или оставить только index.

---

## 9. Быстрая навигация (старт)

```
DataProcessor/README.md
  → PORTFOLIO_INTERVIEW_GUIDE.md   (собеседование)
  → PORTFOLIO_DEMO_RUNBOOK.md      (живое демо)
  → TOP_LEVEL_LAYOUT.md            (структура папок)
  → PORTFOLIO_PROGRESS_LOG.md      (что делали по шагам)
  → PORTFOLIO_NORMALIZATION_PLAN.md (план waves)
```

---

## 10. Итог одним абзацем

За сессию `DataProcessor` получил **единую систему навигации и документации**: от корня репозитория до каждого процессора (Audio/Text/Visual), с картами зависимостей, расширенным DAG, runbook для демо и честным списком техдолга. Код пайплайна не переписывался — работа была про **ясность для портфолио и сопровождения prod**. Для показа на интервью репозиторий **готов**; для уверенности в runtime стоит один раз прогнать сценарий из Demo Runbook на своём окружении и закоммитить изменения.
