# DataProcessor — Top-Level Layout

Каноничная карта корня `DataProcessor/`: что является исходным кодом, что конфигом, что runtime-данными, что внешними артефактами.

Связанные документы:
- [PORTFOLIO_NORMALIZATION_PLAN.md](PORTFOLIO_NORMALIZATION_PLAN.md) — этапы нормализации
- [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md) — журнал прогресса
- [MAIN_INDEX.md](MAIN_INDEX.md) — полный индекс документации

---

## 1. Source code (редактируемый код)

| Путь | Владелец / роль | Prod-заметки |
|------|-----------------|--------------|
| `main.py` | Orchestrator CLI entrypoint | Единая точка локального/E2E запуска пайплайна |
| `AudioProcessor/` | Аудио-модальность, extractors | Зависит от Segmenter contract |
| `TextProcessor/` | Текстовая модальность, extractors | ASR tokens, embeddings, aggregators |
| `VisualProcessor/` | Визуальная модальность, core + modules | GPU/Triton paths |
| `Segmenter/` | Sampling, frames_dir, audio segments | Единственный владелец frame_indices |
| `api/` | HTTP API + worker | Production service layer |
| `embedding_service/` | Semantic DB / FAISS integration | Offline bases + search |
| `common/` | Shared utilities | Не раздувать «misc» |
| `dag/` | Component dependency graph | Source-of-truth для порядка компонентов |
| `dp_queue/` | Celery app + tasks | **Код**, не runtime-очередь; backend шлёт задачи сюда |
| `state/` | Run/processor state managers | **Код** (`managers.py`, `enums.py`); runtime JSON пишется в storage |
| `storage/` | FS/S3 abstraction | [storage/MAIN_INDEX.md](../storage/MAIN_INDEX.md) |
| `qa/` | QA helpers (`component_feature_qa.py`) | Валидация meta/CSV по rules; не тестовые данные |
| `tools/` | Audit/diagnostics scripts | Связать навигацию с `scripts/` |
| `scripts/` | Setup, models, baseline demos | [scripts/MAIN_INDEX.md](../scripts/MAIN_INDEX.md) |
| `configs/` | YAML configs (global, processors) | Разделить stable vs experiment |
| `monitoring/` | Prometheus/Grafana | Runbooks в Wave 5 |
| `docker/` | Container assets | Сверка с `docker-compose.yml` |
| `triton/` | Triton model repositories (ONNX) | Deploy artifacts + `config.pbtxt` |
| `docs/` | Документация и контракты | Единый индекс + progress log |

---

## 2. Configuration (редактируемые конфиги, не код)

| Путь | Назначение | Prod-заметки |
|------|------------|--------------|
| `profiles/` | Профили анализа (YAML) | Включают audio/text/visual; backend может мапить в БД |
| `env.example` | Шаблон переменных окружения | Canonical env-list — Wave 5 |
| `pytest.ini` | Pytest defaults | Smoke/integration paths |
| `requirements-api.txt`, `requirements-test.txt` | Dependency boundaries | Проверить drift с venv |

---

## 3. Runtime / generated (не source code)

Данные создаются при run'ах. **Не ревьюить как код.** В git — через `.gitignore` где применимо.

| Путь | Назначение | Git / retention |
|------|------------|-----------------|
| `dp_results/` | Локальный result_store (NPZ, manifest, renders) | В `.gitignore` (`dp_results/`) |
| `dp_output/` | Локальный frames_dir / промежуточные кадры | В `.gitignore` (`dp_output/`) |
| `_profiles_cache/` | Кэш разрешённых профилей per run_id | Рекомендуется в `.gitignore` (runtime cache) |
| `__pycache__/` | Python bytecode | Стандартный ignore |

**Важно:** runtime state **файлы** (`run_state.json`, `state_events.jsonl`) хранятся в storage по ключам `state/<platform>/<video>/<run>/`, а не в репозитории в `DataProcessor/state/` (там только библиотека).

---

## 4. External artifacts (модели, индексы, чекпоинты)

Управляются отдельно (HF sync, bootstrap, deployment). Не смешивать с прикладным кодом.

| Путь | Назначение | Prod-заметки |
|------|------------|--------------|
| `dp_models/` | Bundled models root (ModelManager) | Offline-first; см. `docs/models_docs/` |
| `dp_triton/` | Triton HTTP client helpers | Код клиента; модели — в `triton/` |
| `faiss_indices/` | Локальные FAISS индексы | В `.gitignore`; воспроизводимый rebuild |
| `wav2vec2_checkpoint/` | Checkpoint для ASR-related paths | Artifact; origin в model docs |

---

## 5. Short path для нового инженера

1. [docs/contracts/CONTRACTS_OVERVIEW.md](contracts/CONTRACTS_OVERVIEW.md)
2. [docs/TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md) (этот файл)
3. `Segmenter/` → `AudioProcessor/` → `TextProcessor/` → `VisualProcessor/`
4. `main.py` + `configs/global_config.yaml`
5. `api/` (если service path)
6. [docs/PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md) — что уже нормализовано

---

## 6. Исправления классификации (Wave 1)

Ранее ошибочно отнесены к `generated`:
- `dp_queue/` — **source** (Celery integration)
- `state/` — **source** (state management library)

Ранее ошибочно отнесены к `artifact`:
- `profiles/` — **configuration** (YAML профили анализа)
---

## Навигация

[Module README](../README.md) · [DataProcessor](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
