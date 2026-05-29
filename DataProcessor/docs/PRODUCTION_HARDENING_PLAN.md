# DataProcessor — Production Hardening (Phase 8+)

**Статус:** `in_progress` (старт после Waves 0–6 и Phase 7)  
**Предпосылка:** документация и навигация готовы ([PORTFOLIO_SESSION_SUMMARY_2026-05-29.md](PORTFOLIO_SESSION_SUMMARY_2026-05-29.md)).  
**Цель:** не «красота репо», а **предсказуемый runtime**, **операционный путь** и **закрытые контракты** для релиза.

Связанные документы:
- [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md) — ops checklist (§6)
- [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) — сценарии проверки
- [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md) — source vs runtime vs artifacts

---

## Где мы сейчас

| Область | Статус |
|---------|--------|
| Навигация, extractor docs, DAG index | ✅ |
| Код extractors / рефакторинг | ⏳ не трогали |
| Smoke 21+22+visual на этой машине | ⏳ |
| Env единый слой (API + storage) | ⏳ drift зафиксирован, не унифицирован |
| `component_graph.yaml` text full | ⏳ tier0 только |
| CI gate на smoke | ⏳ |
| E2E stack | ⏳ по runbook |

**Критерий «prod-ready v1»:** все пункты P0 + P1 закрыты, P2 — по приоритету релиза.

### Уроки после OOM/краша ПК (2026-05-29)

| Фактор | Риск | Митигация |
|--------|------|-----------|
| 21× audio smoke подряд (~30+ мин) | RAM/CPU spike | Не гонять полный smoke + visual в одной сессии; tier-0: `configs/portfolio_demo.yaml` |
| YOLO 48 кадров на CPU (~3 мин) | Нагрузка | Visual minimal — отдельный шаг; AR только на GPU |
| `main.py` exit 0 при `required: false` | Ложный PASS | `validate_smoke_results.sh` в конце `run_smoke_all_components.sh` |
| Артефакты в `storage/` | Засор git | `storage/frames_dir/`, `storage/result_store*/` в `.gitignore` |
| Старый `DP_MODELS_ROOT` в `.env` | Тихий fail | Экспорт явного пути под диск `/media/ilya/...` |

**После краша:** smoke-артефакты в `dp_results/smoke_test/` сохранились; перезапуск smoke не обязателен, если validate 20/21 устраивает.

---

## P0 — Доказать, что пайплайн живой (1–2 дня)

Без этого любая «чистка» кода — вслепую.

| # | Задача | Команда / артефакт | DoD |
|---|--------|-------------------|-----|
| P0.1 | Preflight models | `DP_MODELS_ROOT`, `docs/models_docs/` | Нет missing weights |
| P0.2 | Audio smoke 21/21 | `./DataProcessor/scripts/run_smoke_all_components.sh` | Exit 0, NPZ в `dp_results/smoke_test/` |
| P0.3 | Text smoke (минимум) | `TextProcessor/scripts/smoke_each_extractor_audit_v3.py` (1 scenario) | Exit 0 |
| P0.4 | Visual minimal | Demo A из [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md) | manifest + core NPZ |
| P0.5 | Зафиксировать результат | Entry в [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md) | Дата, ветка, что упало |

**Следующий шаг прямо сейчас:** P0.1 → P0.2 на `system-testing`.

---

## P1 — Операционный контур (3–5 дней)

| # | Задача | Scope | DoD |
|---|--------|-------|-----|
| P1.1 | `configs/portfolio_demo.yaml` | `configs/` | Один YAML для Demo D / локального prod-like run |
| P1.2 | Env alignment | `env.example`, `api/docs/ENVIRONMENT_VARIABLES.md`, `backend/scripts/e2e_env.sh` | Таблица «одна переменная — одно имя» или явный adapter doc |
| P1.3 | `text_processor_full` stage | `docs/reference/component_graph.yaml` | 22 text nodes, validator OK |
| P1.4 | Legacy run instructions | `docs/MAIN_INDEX.md`, дубли API blocks | Один canonical path на сценарий |
| P1.5 | `.gitignore` merge policy | repo root | `_profiles_cache/` + Fetcher paths в `main` (после merge веток) |

---

## P2 — Контракты и конфиги (1–2 недели)

| # | Задача | Scope | DoD |
|---|--------|-------|-----|
| P2.1 | `configs/` classification | `configs/global_config.yaml`, `audit_v3/`, `e2e_*` | `configs/README.md`: stable / audit / e2e / deprecated |
| P2.2 | `profiles/` mapping | `profiles/*.yaml` + backend | Документ: platform → profile → processors |
| P2.3 | COMPONENTS_DESC index-only | `docs/COMPONENTS_DESC.md` | Индекс + ссылки на extractor docs; тело не дублировать |
| P2.4 | Storage contract test | `storage/` | FS roundtrip + S3 path (если MinIO в docker) |
| P2.5 | API worker path | `api/`, `dp_queue/` | Один runbook: submit → worker → result_store |

---

## P3 — Надёжность и наблюдаемость (параллельно с P2)

| # | Задача | DoD |
|---|--------|-----|
| P3.1 | Triton preflight в CI/local gate | `scripts/preflight_triton.py` в чеклисте |
| P3.2 | Prometheus targets | `monitoring/` — scrape up |
| P3.3 | Smoke в CI | GitHub Action или локальный `make smoke` |
| P3.4 | E2E перед релизом | `backend/scripts/start_e2e_stack.sh` зелёный |

---

## P4 — Кодовая чистка (после P0, точечно)

Не массовый рефакторинг — **по pain points** из smoke/E2E.

| Приоритет | Область | Типичная проблема |
|-----------|---------|-------------------|
| 1 | `common/`, дубли helpers | Разрастание misc |
| 2 | Silent fallback в extractors | Нарушение no-fallback policy |
| 3 | Hardcoded paths | `DP_MODELS_ROOT`, frames_dir |
| 4 | Triton vs in-process drift | Разное поведение local/prod |
| 5 | `failing_module` | Оставить test-only, не в prod profile |

---

## Порядок волн (рекомендуемый)

```
P0 smoke ──► P1 ops/config ──► P2 contracts ──► P3 CI/E2E
                    │                                    │
                    └──────────── P4 code fixes ◄────────┘
                              (только где упало)
```

---

## Трекер Phase 8

- [x] P0.1 preflight models (2026-05-29, основной ПК)
- [x] P0.2 audio smoke — **20/21 NPZ** (emotion_diarization: CUDA required; smoke exit 21/21 из‑за `required: false`)
- [x] P0.3 text smoke — **22/22** scenario 0 (~158 s)
- [x] P0.4 visual minimal — **частично:** segmenter + `detections.npz` OK; `action_recognition` — CUDA
- [x] P0.5 progress log entry (Entry 018–019)
- [x] P1.1 portfolio_demo.yaml (+ smoke script NPZ gate, GPU skip)
- [x] P1.2 env alignment — [ENV_ALIGNMENT.md](ENV_ALIGNMENT.md), `env.example`
- [x] P1.3 text_processor_full DAG (22 nodes, validated)
- [x] P1.4 dedupe MAIN_INDEX run paths — удалён дубль API architecture/checklist блоков
- [x] P2.1 configs README — [configs/README.md](../configs/README.md)
- [x] P3.3 smoke in CI — [.github/workflows/dataprocessor-smoke.yml](../../.github/workflows/dataprocessor-smoke.yml), [CI_SMOKE.md](CI_SMOKE.md)
- [x] P3.4 E2E stack green — segmenter (Entry 024) + **audio tier-0** (Entry 025, `E2E_USE_PORTFOLIO_DEMO_CONFIG`)
- [x] P4 smoke gates (partial) — `continue 2` GPU skip fix, `validate_visual_minimal.sh`

Журнал: [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md)
