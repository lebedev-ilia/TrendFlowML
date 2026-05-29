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

## 2026-05-29 — Phase 8: production hardening (старт)

### Entry 017 — PRODUCTION_HARDENING_PLAN

- Stage: `Phase 8`
- Status: `in_progress`
- Сделано:
  - создан [PRODUCTION_HARDENING_PLAN.md](PRODUCTION_HARDENING_PLAN.md) (P0–P4)
  - ссылка из PORTFOLIO_NORMALIZATION_PLAN.md
- Next:
  - **P0.1–P0.2:** preflight + `run_smoke_all_components.sh` на этом ПК

### Entry 018 — P0.1 + P0.2 audio smoke (основной ПК)

- Stage: `Phase 8` P0
- Status: `done` (с оговоркой по emotion_diarization)
- Ветка: `system-testing`
- P0.1 Preflight:
  - `DP_MODELS_ROOT=/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_models/bundled_models`
  - ключевые артефакты: whisper, clap, source_separation, pyannote, wavlm — OK
  - `prepare_hf_cache.sh` — OK
  - `dp_models_selftest.py` — 3/3 OK
- P0.2 Smoke (`run_smoke_all_components.sh`, ~27 мин):
  - скрипт: **21/21 exit 0**
  - `validate_smoke_results.sh`: **20/21 NPZ**
  - **emotion_diarization**: AudioProcessor exit 1 (`No CUDA GPUs are available`), NPZ нет; extractor `required: false` → `main.py` всё равно exit 0
- Результаты: `DataProcessor/dp_results/smoke_test/`
- Заметки:
  - `AudioProcessor/.env` указывает старый путь (`Рабочий стол/...`) — для этого диска экспортировать `DP_MODELS_ROOT` явно или обновить `.env`
  - для prod-gate smoke: после прогона всегда `validate_smoke_results.sh`
- Next: P0.3 text smoke, P0.4 visual minimal; P4: smoke script → fail если NPZ нет

### Entry 019 — P0.3 text smoke + P0.4 visual (частично)

- Stage: `Phase 8` P0
- Status: `done` (P0.3 полный; P0.4 частичный)
- P0.3 Text smoke (`smoke_each_extractor_audit_v3.py --scenario-index 0`):
  - **22/22 OK**, ~158 s, exit 0
  - venv: `TextProcessor/.tp_venv`
- P0.4 Visual minimal:
  - **Segmenter:** `storage/frames_dir/-Q6fnPIybEI/video/` — 48 union frames (~67 s)
  - **core_object_detections:** `detections.npz` OK (~3 min, ultralytics CPU)
  - **action_recognition:** FAIL `No CUDA GPUs are available` (SlowFast, `device: cuda` в YAML)
  - Артефакты: `storage/result_store_ar_minimal/youtube/-Q6fnPIybEI/ar_minimal_cli_001/`
  - `VisualProcessor/main.py` exit 0 при падении AR (как audio optional)
- Блокеры GPU на этом ПК: `emotion_diarization` (audio), `action_recognition` (visual)
- Next: P0.5 закрыт; P1; на GPU-машине повторить AR + emotion; P4: exit code / NPZ gate

### Entry 020 — после краша ПК: выводы + P1.1

- Stage: `Phase 8` (recovery)
- Status: `done`
- Проверка после краша:
  - ветка `system-testing`, smoke NPZ на диске — **OK**
  - незакоммичены: docs Phase 8, `PRODUCTION_HARDENING_PLAN.md`
- Выводы: см. § «Уроки после OOM/краша» в PRODUCTION_HARDENING_PLAN.md
- Доработки:
  - `configs/portfolio_demo.yaml` — tier-0 audio (clap/tempo/loudness), без GPU extractors
  - `run_smoke_all_components.sh` — NPZ validate gate, skip `emotion_diarization` без CUDA
  - `.gitignore` — `storage/frames_dir/`, `storage/result_store*/`, `storage/state/`
- Проверка `portfolio_demo.yaml`: run_1 OK (~8 min, clap+tempo+loudness), `dp_results/portfolio_demo/`
- Next: P1.2 env alignment; commit doc+config changes; GPU-машина — AR + emotion

### Entry 021 — P1.2 env + P1.3 text_processor_full + configs README

- Stage: `Phase 8` P1
- Status: `done`
- P1.2 Env alignment:
  - создан [ENV_ALIGNMENT.md](ENV_ALIGNMENT.md) — local vs docker vs E2E, матрица переменных
  - обновлён [env.example](../env.example) — блок local dev + ссылка на guide
- P1.3 DAG:
  - stage `text_processor_full` в [component_graph.yaml](reference/component_graph.yaml) — **22 nodes**, topo OK
  - обновлён [COMPONENT_GRAPH_INDEX.md](reference/COMPONENT_GRAPH_INDEX.md)
  - [EXTRACTOR_DEPENDENCIES.md](../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md) — ссылка на full stage
- P2.1 (частично): [configs/README.md](../configs/README.md)
- P1.4: удалён дубль `DATAPROCESSOR_API_ARCHITECTURE` / `API_DEVELOPMENT_CHECKLIST` в [MAIN_INDEX.md](MAIN_INDEX.md)
- Next: commit Phase 8; GPU — emotion + AR; P3 CI smoke

### Entry 022 — commit prep + CI smoke workflow

- Stage: `Phase 8` P3
- Status: `done`
- Сделано:
  - восстановлен [configs/README.md](../configs/README.md) (index + global config reference)
  - CI: [dataprocessor-smoke.yml](../../.github/workflows/dataprocessor-smoke.yml)
  - [CI_SMOKE.md](CI_SMOKE.md)
- Next: `git commit` Phase 8 на `system-testing`

### Entry 023 — push + E2E runbook

- Stage: `Phase 8` P3
- Status: `done` (push); P3.4 E2E — runbook, runtime pending
- Git:
  - `git pull --rebase origin system-testing` (merge fetcher `9e397a4`)
  - `git push origin system-testing` → `f187912`
- E2E:
  - создан [E2E_PREFLIGHT.md](E2E_PREFLIGHT.md)
  - полный `start_e2e_stack.sh` не запускался (CVAT/docker на портах; тяжёлый прогон)
- Next: поднять E2E infra в чистом shell; P3.4 checklist в E2E_PREFLIGHT §4

### Entry 024 — E2E infra + stack + run (P3.4 partial)

- Stage: `Phase 8` P3.4
- Status: `done` (segmenter-only E2E); multimodal — pending GPU/profile
- Infra: `setup_e2e_infra.sh` OK (postgres/redis/minio, prometheus :9091)
- Stack: `start_e2e_stack.sh` — 8000/8001/8002/8005 healthy
- E2E run: `e2e_run_to_complete.py --with-dataprocessor --timeout 2400`
  - run_id: `63048b78-74ac-457b-97bd-fa5f8a772a5c`
  - video: `-Q6fnPIybEI` (mock download)
  - result: ingestion **completed**, DP **segmenter success** (~6s), audio/visual/text off in default profile
- Code: `backend/scripts/e2e_env.sh` — default `DP_MODELS_ROOT`, `TORCH_HOME`, `HF_HOME`
- Logs: `backend/.e2e/logs/20260529-135506/`
- Next: E2E with `portfolio_demo.yaml` or `global_config`; GPU AR/emotion; commit Entry 024

### Entry 025 — E2E audio tier-0 (portfolio_demo)

- Stage: `Phase 8` P3.4
- Status: `done` (segmenter + audio tier-0)
- Config: `E2E_USE_PORTFOLIO_DEMO_CONFIG=1` → `TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG_PATH` в `e2e_env.sh`
- Stack: `start_e2e_stack.sh --no-stop` (run id `20260529-140304`)
- E2E run: `e2e_run_to_complete.py --with-dataprocessor --timeout 3600`
  - run_id: `203b6361-870d-4c45-8362-89ce3dbfa4ba`
  - video: `-Q6fnPIybEI` (mock)
  - segmenter: **success** (~5.9s)
  - audio: **success** (~189s) — clap 33s, loudness 9s, tempo 0.9s
  - ingestion: **completed** (~4 min total)
- Code: `backend/scripts/e2e_env.sh` — toggle `E2E_USE_PORTFOLIO_DEMO_CONFIG`
- Logs: `backend/.e2e/logs/e2e_portfolio_audio_*.log`, stack `20260529-140304`
- Next: full multimodal (`global_config` + GPU/Triton); GPU — emotion + AR; commit Entry 025

### Entry 026 — P4 smoke gates + visual validate

- Stage: `Phase 8` P4
- Status: `done` (CPU-only scope)
- Fixes:
  - `run_smoke_all_components.sh` — **`continue 2`** для GPU-only skip (раньше `emotion_diarization` всё равно запускался на CPU)
  - новый `validate_visual_minimal.sh` — frames_dir + `detections.npz` schema; AR skip без CUDA
- Проверка: validate visual на артефактах Entry 019 — **OK** (detections ✅, AR skip)
- `.gitignore` — `storage/videos/`, `storage/__health_check__/`
- CI: `bash -n validate_visual_minimal.sh` в dataprocessor-smoke workflow
- Next: GPU-машина — полный audio 21/21 + AR; E2E `global_config`; optional `--strict` exit в `main.py`

### Entry 027 — P2.2 profiles + P2.5 API worker + GPU runbook

- Stage: `Phase 8` P2
- Status: `done` (docs)
- Новые документы:
  - [PROFILES_MAPPING.md](PROFILES_MAPPING.md) — profile JSON ↔ global_config ↔ visual cfg; E2E toggles
  - [API_WORKER_RUNBOOK.md](API_WORKER_RUNBOOK.md) — POST /process → Redis → worker → main.py → result_store
  - [GPU_MACHINE_RUNBOOK.md](GPU_MACHINE_RUNBOOK.md) — чеклист 21/21 audio + AR + full E2E на CUDA
- CI: `py_compile preflight_triton.py` в dataprocessor-smoke
- Next: выполнить GPU_MACHINE_RUNBOOK на ноутбуке/GPU-ПК; Entry 028 с результатами
