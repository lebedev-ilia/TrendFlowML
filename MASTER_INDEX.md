# TrendFlow — MASTER INDEX (единая точка входа для новой сессии/модели)

> Прочитай этот файл первым — он даёт карту всех важных документов проекта с 1-строчным описанием и
> порядком чтения. Пути указаны от корня репо `/home/ilya/Рабочий стол/TrendFlowML/`.
> Быстрый онбординг: раздел 1 → 2 → 10 (как работает автоматизация) → 3 (текущая задача).

---

## 0. Что за проект (1 абзац)
TrendFlow — мультимодальная AI-система предсказания популярности видео (просмотры/лайки через 14 и 21 день).
Видео разбирается на 75+ компонентов-экстракторов (visual/audio/text), их выходы идут в ML-модель (Encoder+Fusion).
Текущая работа: доведение каждого компонента DataProcessor до прод-готовности на GPU (RunPod), автономно двумя
агентами под управлением из VK. Потребители выходов: **Models (Encoder, seq)** и **аналитики (агрегаты)**.

## 1. Точки входа (читать первыми)
- **`CLAUDE.md`** — полный контекст проекта (архитектура, компоненты, решения). Главный вводный файл.
- **`automation/runner/AGENT_CONTEXT.md`** (раздел 5) — сжатый handoff: что делаем, статус, как устроена автоматизация, что дальше (бывший `automation/CONTEXT_HANDOFF.md`, удалён).
- **`docs/MAIN_INDEX.md`** — индекс общесистемной документации.
- Под-индексы: `DataProcessor/docs/MAIN_INDEX.md`, `Models/docs/MAIN_INDEX.md`, `backend/docs/MAIN_INDEX.md`,
  `Fetcher/docs/INDEX.md`.

## 2. Продукт, архитектура, дорожная карта (`docs/`)
- `docs/PRODUCT_VISION.md` — видение продукта, для кого и зачем.
- `docs/MAIN_README.md` — обзор архитектуры и ML-системы.
- `docs/PROJECT_MAP.md` / `docs/PROD_ARCH_GAP_MAP.md` — карта проекта и разрывы до прода.
- `docs/PRODUCT_ROADMAP_TO_PRODUCTION.md` — дорожная карта до продакшена.
- `docs/LOAD_AND_SCALING_PLAN.md`, `docs/K8S_FIRST_APPROACH.md`, `docs/CONTAINER_GRANULARITY.md` — масштабирование.
- `docs/WEBSITE_REQUIREMENTS.md`, `docs/DESIGN_SYSTEM.md` — требования и дизайн сайта.
- `docs/DEPLOYMENT_GUIDE.md` / `docs/DEPLOYMENT_QUICKSTART.md` — деплой.

## 3. DataProcessor — ГЛАВНАЯ ТЕКУЩАЯ ЗАДАЧА (`DataProcessor/docs/`)
- **`COMPONENT_VALIDATION_CHECKLIST.md`** — статусы валидации компонентов (✅/⬜) + подтверждённые фичи. Источник очереди.
- **`COMPONENT_VALIDATION_PROTOCOL.md`** — как валидировать компонент (критерий «выход пригоден», 4 оси).
- `COMPONENT_CONTRACTS.md` — контракты входов/выходов компонентов.
- `COMPONENTS_DESC_INDEX.md` / `COMPONENTS_DESC.md` — описание всех компонентов.
- `LOGIC_ERRORS_FOR_CLAUDE.md` — реестр известных логических багов (L1…).
- `FEATURE_QUALITY_PLAYBOOK.md`, `FEATURE_COVERAGE_AUDIT.md` — качество/покрытие фич.
- `MODEL_FEATURE_SET_ROADMAP.md`, `DATAPROCESSOR_QUEUE_CANONICAL.md` — набор фич под модель, каноничная очередь.
- `component_reports/<component>/` — отчёты валидации (REPORT_*.md), CRITERIA.md, VERIFICATION_*.md.
- **`COMPONENT_DEEP_DIVE_PROTOCOL.md`** / **`COMPONENT_DEEP_DIVE_CHECKLIST.md`** (новое, 2026-07-17) —
  ОТДЕЛЬНАЯ от валидации задача: глубокий разбор компонента (функционал/вход/выход/фичи/оптимизации/
  слабые места/рекомендации/интерпретируемость/польза для моделей и аналитиков + оценки 1–5), пишется
  ПОСЛЕ штампа валидации. Итог — `component_reports/<component>/FINAL_REPORT.md`.
- Подпапки: `contracts/`, `architecture/`, `reference/`, `audit_v3/`, `audit_v4/`.

## 4. Models — ML-модель (`Models/docs/`)
- `MAIN_INDEX.md`, `ARCHITECTURE_REVIEW.md` — обзор и ревью архитектуры модели.
- **`contracts/ENCODER_CONTRACT.md`** — что Encoder ждёт от seq-выходов компонентов (ключевой контракт).
- `contracts/BASELINE_MODEL.md`, `V1_TRANSFORMER_MODEL.md`, `V2_CONTEXT_MODEL.md` — baseline и версии модели.
- `contracts/TARGETS_SPLITS_METRICS.md`, `PREDICTION_REPORT_CONTRACT.md`, `MODEL_SYSTEM_RULES.md`.

## 5. Backend — FastAPI + PostgreSQL + Celery (`backend/docs/`)
- `MAIN_INDEX.md`, `OVERVIEW.md` — обзор бэкенда и оркестрации.
- `API.md`, `DATABASE.md` / `DATABASE_ARCH.md`, `STORAGE_LAYOUT.md`, `PROFILES.md`.
- `RUNS_AND_WORKERS.md`, `EVENTS_AND_LOGGING.md`, `OPERATIONS.md`, `SECURITY.md`.
- `E2E_RUNBOOK.md`, `E2E_FULL_CHECKLIST.md` — сквозные прогоны. `reference/DATAPROCESSOR_CONTRACT.md` — контракт с DP.

## 6. Fetcher — загрузка видео (`Fetcher/docs/`)
- `INDEX.md`, `CORE_INGESTION.md`, `PIPELINE_ORCHESTRATION.md`, `BACKEND_CONTRACTS.md`.
- `PLATFORM_ADAPTERS.md`, `QUEUE_ORCHESTRATION.md`, `RATE_LIMITING_AND_LOCKS.md`, `RUNBOOKS.md`.

## 7. DynamicBatch — динамический батчинг (`DynamicBatch/docs/`)
- `IMPLEMENTATION_PLAN.md`, `DYNAMIC_BATCHING_CHECKLIST.md`, `BENCHMARK_REGISTRY_CONTRACT.md`.

## 8. Site — фронтенд (внешний репо)
- `/home/ilya/Рабочий стол/site/SITE_SPECIFICATION.md` — полная спецификация сайта (Next.js MVP). Вне этого репо.

## 9. Деплой / инфраструктура
- `k8s/` — Kubernetes-манифесты. `deploy/`, `docker-compose.prod.yml` — прод-развёртывание.
- `configs/` — профили анализа (`visual_*`, `profile_*`).

## 10. Автоматизация агентов (`automation/`, `automation/runner/`) — КАК ВСЁ РАБОТАЕТ СЕЙЧАС
- **`automation/runner/README.md`** — как запускать и что делает раннер (главный операционный документ).
- **`automation/runner/AGENT_CONTEXT.md`** — ЕДИНЫЙ контекст для агентов из VK (заменяет прежние
  `system_prompt.md` / `system_prompt_verify.md` / `AGENT_ONBOARDING.md` / `DECISIONS_AND_LESSONS.md` /
  `VERIFICATION_GUIDE.md` / `CONTEXT_HANDOFF.md` / `RESOURCE_TIMING_LEDGER.md`, ныне удалённые). Разделы:
  0 — общие правила (язык/краткость/лимиты), 1 — системный промпт рабочего агента, 2 — системный промпт
  верификатора, 3 — онбординг (суть проекта, протокол компонента), 4 — ТЗ верификации, 5 — устройство
  автоматизации (агенты/боты/деньги/запуск, бывший CONTEXT_HANDOFF), 6 — тайминги/ресурсы по компонентам
  (append-only), 7 — решения и уроки/институциональная память (append-only, читать и дополнять!).
  `agent_runner.py` собирает системный промпт из разделов 0+1 (компонент) или 0+2 (verify) программно.
- `automation/AUTOMATION_SINGLE_AGENT_PLAN.md`, `automation/SESSION_PLAYBOOK.md` — план/протокол автономной работы.
- `automation/runpod_ssh/POD_CONNECTION.md`, `RUNPOD_SETUP_GUIDE.md` — доступ к RunPod-подам.
- Ключевые модули раннера: `agent_runner.py` (Первый агент, цикл, работает непрерывно по ВСЕМ
  компонентам), `assistant.py` (Второй агент, VK-бот + периодический контроль/фикс/рестарт Первого),
  `supervisor.py` (быстрый авто-ответчик Первому — часть Второго агента), `hub.py` (VK), `budget.py`
  (только учёт трат, БЕЗ дневного лимита), `limits.py` + `claude_limits_scraper.py` (единственный
  реальный лимит — % Claude 5ч/неделя, 95%/97%), `runpod_api.py` (+ `account_balance()`, только
  предупреждение) + `runpod_gpu_scraper.py` + `runpod_pod_browser.py` (поды/цены/фолбэк, потолок
  $0.30/$0.60ч), `podmanager.py`, `settings.py`, `hooks.py`.

## 10b. Fetcher-инфраструктура (`automation/fetcher/`) — ОТДЕЛЬНАЯ система, не трогать из runner
3 постоянных CPU-пода RunPod (`fetcher-main`, `fetcher-worker-b`, `fetcher-worker-c`, 2vCPU/8ГБ,
$0.08/ч каждый, свой Network Volume 15ГБ) — сбор YouTube-датасета (`Fetcher/fetcher/dataset_collector/`),
работают непрерывно (никогда не гасятся). См. `automation/fetcher/README.md`. Защищены от массовых
операций ML-раннера через `state/machines.json` (`policy=persistent`, `kind=fetcher`) — см.
`AGENT_CONTEXT.md` раздел 3.4 и шапку `automation/runner/runpod_api.py`. Третий VK-бот-наблюдатель
(`watchdog.py`, Haiku) + часовой отчёт метрик (`hourly_report.py`, чистый код).

## 11. Куда писать/обновлять (для агентов)
- Прогресс компонента → `automation/runner/state/progress/<component>.md`; контекст сессии → `state/last_session.md`.
- Отчёты → `DataProcessor/docs/component_reports/<component>/`; штампы → `COMPONENT_VALIDATION_CHECKLIST.md`.
- Уроки/грабли → `automation/runner/AGENT_CONTEXT.md` (раздел 7). Новые важные доки → добавь ссылку СЮДА и в `MAIN_INDEX`.
- Автогенерация полного списка всех .md: `python automation/runner/build_docs_index.py` → `automation/DOCS_INDEX.md`.

---
*Зона владельца (не трогать): ручная разметка semantic-баз; адаптация под YOLO-датасет. Язык общения — русский.*
