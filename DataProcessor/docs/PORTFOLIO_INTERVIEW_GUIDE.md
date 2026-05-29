# DataProcessor — Portfolio & Interview Guide

Руководство для демонстрации **DataProcessor** на собеседовании и в портфолио.  
Двойная цель: показать инженерную зрелость **и** честно обозначить границы v1.

Связано: [../README.md](../README.md) · [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md)

---

## Elevator pitch (30–60 сек)

> DataProcessor — это мультимодальный ML-пайплайн для анализа видео: из одного ролика извлекаются сотни структурированных признаков по видео, аудио и тексту. Архитектура модульная: Segmenter задаёт sampling, три процессора пишут версионированные NPZ-артефакты, контракты fail-fast без «тихих» fallback. Есть путь к продакшену: HTTP API, очередь, Triton для GPU, observability. Я приводил документацию и структуру к единому виду для сопровождения и релиза.

---

## Что показать на собеседовании (рекомендуемый порядок, ~15–20 мин)

### 1. Архитектура (2 мин)

- [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md) — что код, что runtime, что артефакты
- [contracts/CONTRACTS_OVERVIEW.md](contracts/CONTRACTS_OVERVIEW.md) — NPZ, Segmenter, no-fallback

### 2. Один вертикальный срез (5 мин)

Выберите **один** сценарий и пройдите его целиком:

| Срез | Что рассказать | Документ |
|------|----------------|----------|
| **Audio ASR → Text** | Segmenter → `asr_extractor` → TextProcessor | [TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../TextProcessor/docs/EXTRACTOR_DEPENDENCIES.md) |
| **Visual baseline** | segmenter → core_clip → cut_detection | [VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../VisualProcessor/docs/EXTRACTOR_DEPENDENCIES.md) |
| **Tier-0 audio** | clap + loudness + tempo | [AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md](../AudioProcessor/docs/EXTRACTOR_DEPENDENCIES.md) |

### 3. Качество и контракты (3 мин)

- Пример `README.md` + `SCHEMA.md` у компонента (например `cut_detection` или `asr_extractor`)
- `schema_version` в NPZ meta — воспроизводимость

### 4. Production path (3 мин)

- [architecture/PRODUCTION_ARCHITECTURE.md](architecture/PRODUCTION_ARCHITECTURE.md)
- [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md) — API, monitoring, E2E
- [monitoring/README.md](../monitoring/README.md) — Grafana/Prometheus

### 5. Масштаб команды / solo (2 мин)

- 21 + 22 + 29 компонентов, audit v3/v4, dependency maps по процессорам
- [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md) — как вели рефакторинг документации

---

## Demo runbook (пошагово)

Полные сценарии A–E (visual / audio / text / full CLI / E2E): [PORTFOLIO_DEMO_RUNBOOK.md](PORTFOLIO_DEMO_RUNBOOK.md)

---

## Checklist перед демо

| # | Проверка |
|---|----------|
| 1 | `DP_MODELS_ROOT` указывает на реальный bundled_models |
| 2 | Короткое тестовое видео (5–120 с) в `example/example_videos/` |
| 3 | Smoke audio: `./DataProcessor/scripts/run_smoke_all_components.sh` (опционально) |
| 4 | Text smoke: `TextProcessor/scripts/smoke_each_extractor_audit_v3.py` (1 scenario) |
| 5 | Открыт `dp_results/.../manifest.json` + один `.npz` для показа структуры |
| 6 | Знаете 3 принципа: no-fallback, Segmenter sampling, ModelManager offline |

---

## Команды «на столе»

```bash
# Минимальный прогон
python3 DataProcessor/main.py \
  --video-path example/example_videos/video1.mp4 \
  --global-config DataProcessor/configs/global_config.yaml \
  --platform-id youtube --video-id interview_demo --run-id run_1

# Просмотр manifest
find DataProcessor/dp_results -name manifest.json | head -1 | xargs cat | python3 -m json.tool | head -40
```

E2E (если спрашивают про full stack): `backend/scripts/start_e2e_stack.sh`

---

## Вопросы, которые хорошо «ломают» джуна vs сеньора (и ваши ответы)

| Вопрос | Хороший ответ |
|--------|----------------|
| Почему NPZ, а не JSON? | Плотные массивы, скорость, единый meta; JSON только для manifest/render |
| Кто владеет frame_indices? | Только Segmenter |
| Что если нет audio track? | Empty semantics в meta, не подделка нулей в model-facing |
| Как версионируете модели? | `models_used[]`, `model_signature`, `schema_version` |
| Как масштабировать на 100k видео? | [VisualProcessor/docs/PRODUCTION_READINESS_AND_SCALE_PLAN.md](../VisualProcessor/docs/PRODUCTION_READINESS_AND_SCALE_PLAN.md), K8s + Triton + queue |

---

## Tech debt и границы v1

Честный список для интервью (не слабость, а зрелость):

| Тема | Статус | Комментарий |
|------|--------|-------------|
| `component_graph.yaml` | Расширен | baseline 33 узла; text tier2+ вне yaml — см. COMPONENT_GRAPH_INDEX |
| Triton coverage | Частичный | CLIP, MiDaS, RAFT, Places365; не все модели на Triton |
| ModelManager API | In-process | Нет отдельного HTTP-сервиса dp_models |
| Frontend | Вне scope DP | Backend каркас есть; UI в планах ([doc.md](../../doc.md)) |
| Env naming | Два слоя | `TREND_STORAGE_*` vs API `STORAGE_*` — [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md) §9 |
| `failing_module` | Test only | Демо optional component failure |
| micro_emotion | Heavy dep | OpenFace Docker — ops cost |
| Corpus packs (text) | Частично | FAISS/top-k packs — analytics until `pack_digest` fixed |
| Дубли в MAIN_INDEX | Косметика | Исторические блоки API architecture 2× |

**Не техдолг (намеренно):** строгий fail-fast, offline models, audit trails v3/v4.

---

## Карта нормализации (что уже сделано)

| Wave | Результат |
|------|-----------|
| 0–1 | План, TOP_LEVEL_LAYOUT, индексы |
| 2 | AudioProcessor: deps + 21 docs |
| 3 | TextProcessor: deps + 22 docs |
| 4 | VisualProcessor: deps + 29 docs |
| 5 | API/ops map, env.example |
| 6 | Этот guide + [README.md](../README.md) |

---

## Ссылки для README портфолио / резюме

Скопируйте в проект на GitHub:

- Repo: TrendFlowML / `DataProcessor/`
- Highlight: *Multimodal video feature pipeline, 70+ components, contract-driven NPZ artifacts*
- Docs entry: `DataProcessor/README.md`
- Deep dive: `DataProcessor/docs/PORTFOLIO_INTERVIEW_GUIDE.md`
