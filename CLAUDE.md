# TrendFlow — Контекст проекта для Claude

> Этот файл автоматически читается Claude Code при работе в этой директории.
> Он содержит полный контекст проекта TrendFlow для AI-ассистента.

---

## Что такое TrendFlow

TrendFlow — мультимодальная AI-система для анализа и предсказания популярности видеоконтента (просмотры/лайки через 2 и 3 недели). Анализирует визуальный, аудио и текстовый контент видео через 75+ специализированных компонентов-экстракторов.

**Аудитория**: (1) обычные креаторы — хотят загрузить видео и получить прогноз + рекомендации; (2) профессиональные аналитики — нужны факторы роста, графики, распределения, сравнения.

**Текущий статус**: MLService почти реализован и имеет высший приоритет; backend-каркас сайта готов; frontend в разработке; модели в разработке (baseline + v1 трансформер).

---

## Архитектура системы

```
Пользователь
    │
    ▼
[Site] Next.js (frontend) + FastAPI (backend) + PostgreSQL
    │
    ▼
[MLService]
    ├── Fetcher        — загрузка видео с платформ (YouTube, Twitch, TikTok)
    ├── DataProcessor  — извлечение признаков (ядро системы)
    │   ├── VisualProcessor  (29 компонентов)
    │   ├── AudioProcessor   (24 компонента)
    │   ├── TextProcessor    (22 компонента)
    │   ├── Segmenter        — выборка кадров/сегментов
    │   ├── Embedding Service — база эмбеддингов (CLIP, CLAP, бренды, авто)
    │   ├── Triton           — инференс ONNX моделей (CLIP, RAFT, MiDaS, Places365)
    │   └── DP_Models        — ModelManager для всех весов
    └── Models         — ML-модели предсказания популярности
        ├── AudioEncoder + VisualEncoder
        ├── AudioHead + VisualHead (Transformer)
        ├── Fusion (cross-attention)
        └── Prediction Heads (views/likes × 14/21 дней)
```

---

## Компоненты — краткое описание

### Site (Next.js + FastAPI)

**Папка**: `/home/ilya/Рабочий стол/site/`
**Спецификация**: `site/SITE_SPECIFICATION.md` (полный дизайн и функционал MVP)

Дизайн: тёмная тема, фиолетово-голубые акценты, Three.js фон, Inter шрифт.

Страницы:
- Landing с 3D-фоном (Three.js), описанием модальностей, тарифами
- Auth: email + Google/GitHub OAuth, NextAuth.js
- Создание анализа: 3-шаговый wizard (URL/upload → метаданные → конфигурация)
- Страница прогресса: WebSocket real-time с превью кадров и bbox
- Страница результатов: вкладки по компонентам, графики, сравнение видео
- Личный кабинет: конфигурации, история анализов, биллинг, настройки

Tech stack сайта: Next.js (App Router), TypeScript, Tailwind CSS, shadcn/ui, Three.js / React Three Fiber, Framer Motion, NextAuth.js.

Backend сайта: FastAPI, PostgreSQL (`core.*` schema), Redis, Celery, SQLAlchemy, Alembic.

### Fetcher

**Папка**: `Fetcher/`
**Индекс**: `Fetcher/docs/INDEX.md`

Загружает видео с YouTube (v1). Собирает: заголовок, описание, комментарии, данные канала, скачивает видео. Передаёт артефакты в DataProcessor. После `finalize` вызывает `POST /api/runs/{run_id}/trigger-processing` на backend.

### DataProcessor

**Папка**: `DataProcessor/`
**Индекс**: `DataProcessor/docs/MAIN_INDEX.md`

Source-of-truth результатов: `manifest.json` и NPZ артефакты.
Конфигурируется через YAML-профили (profile.yaml → visual.cfg).

Каждый компонент: получает сегменты/кадры → обрабатывает → выдаёт `[N, M]` признаков.

**VisualProcessor** (29 компонентов): object detection, face analysis, scene classification, pose estimation, shot quality, brand semantics, car semantics, similarity metrics, и др.

**AudioProcessor** (24 компонента): speaker diarization, music detection, audio features, RMS, spectral features, ASR. Сильно зависит от Segmenter для сегментов аудио.

**TextProcessor** (22 компонента): title/description embeddings, ASR text, hashtag analysis, semantic clusters, embedding shifts. Не имеет доступа к аудио — запрашивает ASR у AudioProcessor.

**Triton**: хранит ONNX модели (CLIP, RAFT, MiDaS, Places365), отдельный сервис.

**Embedding Service**: база эмбеддингов для семантических компонентов (бренды, авто, аниме).

### Backend (FastAPI)

**Папка**: `backend/`
**Индекс**: `backend/docs/MAIN_INDEX.md`
**Обзор**: `backend/docs/OVERVIEW.md`

REST API + WebSocket. Оркестрирует DataProcessor через Celery.

Ключевые endpoints:
- `POST /api/runs` — запуск ingestion по YouTube URL
- `WS /api/runs/{run_id}/events` — live-прогресс
- `POST /api/workspaces/{id}/videos/{id}/analysis` — анализ загруженного видео
- `POST /api/runs/{run_id}/trigger-processing` — вызов от Fetcher после finalize

Database schema: `core.*` в PostgreSQL (users, workspaces, channels, videos, analysis_jobs, ingestion_runs, predictions, artifacts).

**Поток YouTube**: `POST /api/runs` → Fetcher → finalize → `trigger-processing` → DataProcessor → manifest.json → обновление БД.

**Поток upload**: `POST /api/videos` → upload → `POST /api/analysis` → Celery → DataProcessor → manifest.json.

### Models (ML)

**Папка**: `Models/`
**Индекс**: `Models/docs/MAIN_INDEX.md`
**Контракты**: `Models/docs/contracts/`

Таргет: просмотры и лайки через 14 и 21 день (4 значения на видео).

Архитектура v1:
```
[VisualProcessor output] → AudioEncoder → AudioHead (Transformer + time_mlp)
[AudioProcessor output]  → VisualEncoder → VisualHead (Transformer + time_mlp)
                                               ↓
                                      Fusion (cross-attention: video+audio+text+metadata+temporal)
                                               ↓
                                      Prediction Heads (views/likes × 14/21)
```

Learnable Pooling: M локальных токенов → K summary-токенов (фиксированное K).
Positional encoding: time embedding через MLP от `t_center / duration_sec`.

Baseline модель: XGBoost/LightGBM (сначала), затем v1 трансформер.
Планируемая интерпретируемость: SHAP (optional, internal/debug), top_modalities evidence.

### DynamicBatch

**Папка**: `DynamicBatch/`
Система динамического батчинга для оптимизации обработки множества видео. Управляет очередями и распределением ресурсов.

### Конфигурации

**Папка**: `configs/`
`visual_triton_baseline_gpu_local.yaml` — конфиг VisualProcessor.
`profile_triton_baseline_gpu_local.yaml` — профиль DataProcessor (audio tier-0 + VisualProcessor cfg).

---

## Важные технические решения

- **Source-of-truth**: `manifest.json` + NPZ артефакты (не БД)
- **БД** только как индекс и ускоритель, не как источник истины
- **Celery + Redis** для очередей задач (не Kafka для MVP)
- **Celery beat** для polling статуса Fetcher
- **WebSocket** для live-прогресса (events из `state_events.jsonl`)
- **Profiles** (YAML) — конфигурация анализа (profil_hash = sha256 от JSON с сортировкой)
- **Triton** запускается как отдельный микросервис на отдельном порту
- **ModelManager** — не отдельный сервис в v1, компоненты импортируют функции напрямую
- **platform_id="youtube"** зафиксирован до расширения на другие платформы
- **Сравнение видео** через cosine similarity по CLIP/CLAP/text эмбеддингам

---

## Структура этого Obsidian Vault

```
trendflow-docs-graf/
├── CLAUDE.md                    ← этот файл (главный контекст для Claude)
├── docs/
│   ├── MAIN_INDEX.md            ← главный индекс всего vault
│   ├── MAIN_README.md           ← описание архитектуры и ML-системы
│   ├── PRODUCT_ROADMAP_TO_PRODUCTION.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── ...
├── site/
│   ├── SITE_SPECIFICATION.md    ← полная спецификация сайта (MVP)
│   └── AGENTS.md / CLAUDE.md
├── backend/
│   └── docs/                    ← полная документация backend
│       ├── OVERVIEW.md
│       ├── API.md
│       ├── DATABASE.md
│       └── ...
├── DataProcessor/
│   └── docs/                    ← документация DataProcessor (самый объёмный раздел)
├── Fetcher/
│   └── docs/
├── Models/
│   └── docs/
├── DynamicBatch/
│   └── docs/
├── configs/                     ← конфиги профилей
├── k8s/                         ← Kubernetes манифесты
└── storage/                     ← runbooks QA
```

---

## Как работать с этим Vault

**Навигация**: `docs/MAIN_INDEX.md` — единая точка входа для поиска любого документа.

**Для конкретной задачи:**
- Работа с сайтом → `site/SITE_SPECIFICATION.md`
- Backend API → `backend/docs/API.md`, `backend/docs/OVERVIEW.md`
- Компоненты DataProcessor → `DataProcessor/docs/MAIN_INDEX.md`
- ML-модели → `Models/docs/contracts/`
- Fetcher → `Fetcher/docs/INDEX.md`
- E2E тесты → `backend/docs/E2E_RUNBOOK.md`, `backend/docs/E2E_FULL_CHECKLIST.md`
- Деплой → `docs/DEPLOYMENT_GUIDE.md`, `docs/DEPLOYMENT_QUICKSTART.md`

---

## Расположение кода (реальные репозитории)

- **TrendFlowML**: `/home/ilya/Рабочий стол/TrendFlowML/` — основной монорепо (backend + scripts)
- **Site**: `/home/ilya/Рабочий стол/site/` — Next.js frontend
- **Другие компоненты** (DataProcessor, Fetcher, Models): в отдельных репозиториях или папках внутри TrendFlowML

---

*Обновлено: 2026-06-21. При изменениях в проекте обновляй этот файл.*
