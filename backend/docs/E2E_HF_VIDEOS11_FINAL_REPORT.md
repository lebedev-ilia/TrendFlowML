# Итоговый отчёт: E2E HF videos11 (mock → real-video → core_identity)

**Дата:** 2026-07-02  
**Платформа:** Linux, RTX 2060 6 GiB (локальный E2E-хост)  
**Датасет:** [Ilialebedev/videos11](https://huggingface.co/datasets/Ilialebedev/videos11) — 5 коротких YouTube-клипов (~10–30 s)

Связанные документы:

- Пошаговый runbook: [E2E_RUNBOOK.md](E2E_RUNBOOK.md) (§0.14–§0.17)
- Чеклист с галочками: [E2E_FULL_CHECKLIST.md](E2E_FULL_CHECKLIST.md) §4.3
- Логические баги DP: [`DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md`](../../DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md)

---

## 1. Цель работы

Довести **полный E2E** TrendFlow (Backend → Fetcher → DataProcessor → все процессоры) до состояния, когда:

1. **Mock green** (3 s fixture) проходит §0.1 (оркестрация) и §0.2 (качество NPZ).
2. **Real HF videos11** (5 коротких mp4) проходит тот же gate на **реальном** аудио/видео/ASR.
3. Включён scope **`core_identity`** (semantic heads, в т.ч. `face_identity` через Embedding Service).
4. Результаты **воспроизводимы** и **зафиксированы** в документации + CI fixture.

---

## 2. Уровни валидации (что чем доказывается)

| Уровень | Команда | Доказывает |
|---------|---------|------------|
| **Smoke** | `e2e_run_to_complete.py --with-dataprocessor` | Связка Backend ↔ Fetcher ↔ DP (Segmenter only) |
| **Mock green §0.1** | `e2e_run_full_green.sh` | Fetcher 7/7 + все включённые processors `success` на 3 s mock |
| **Mock green §0.2** | `e2e_validate_output_quality.py` | NPZ-контракт, finite ratio, expected-empty на mock |
| **HF real §0.1+§0.2** | `e2e_run_hf_videos11.py --count 5` | + реальный segmenter duration, `asr_wired_to_text` |
| **HF + core_identity** | `… --with-core-identity` | + `face_identity` / ES match, auto face seed |
| **Регрессия** | `e2e_verify_hf_results.py --with-core-identity` | Повторная §0.1+§0.2 по сохранённым run_id без полного E2E |
| **CI fixture gate** | `pytest tests/unit/test_hf_videos11_results_fixture.py` | Контракт 5/5 green в committed JSON (без `storage/`) |

---

## 3. Финальный статус (2026-07-02)

| Компонент / gate | Статус |
|------------------|--------|
| Mock green §0.1 + §0.2 | ✅ |
| HF videos11 5/5 §0.1 + §0.2 (real audio) | ✅ |
| HF 5/5 + **core_identity** + auto face seed | ✅ |
| **micro_emotion** + OpenFace docker | ✅ 5/5 |
| **L6** manifest ↔ NPZ sync (`face_identity`) | ✅ |
| **L12** normalized bbox + ES deps | ✅ |
| Регрессия `e2e_verify --with-core-identity` | ✅ `worst_exit=0` |
| CI fixture + `hf-e2e-regression.yml` | ✅ |

---

## 4. Валидированные прогоны HF + core_identity

Источник истины: `backend/tests/fixtures/hf_videos11_results.json` (копия `backend/.e2e/state/hf_videos11_results.json`).

| video_id | duration | run_id | face_identity | micro_emotion |
|----------|----------|--------|---------------|---------------|
| `-4WRepA-bss` | 10.7 s | `dbf7eecc-ba7e-4ce2-ad54-ab86914b4177` | ok, n=5, match `hf_seed_-4WRepA-bss` | ok, 5 face / 4 OpenFace |
| `-3Mbinqzig4` | 17.3 s | `cb0c3383-b1eb-4dc4-ac3d-cc5e4404c750` | ok, n=8, match `hf_seed_-3Mbinqzig4` | ok, 8 / 7 |
| `-4RHVBIikn8` | 29.6 s | `21a279af-a6cd-4810-957c-29aaabed7a2e` | ok, n=38, match `hf_seed_-4RHVBIikn8` | ok, 38 / 37 |
| `-0InsUQNwIQ` | 16.3 s | `6c3ab2dc-e10e-45d1-a9e8-6c98aaa5a034` | ok, n=40, match `hf_seed_-0InsUQNwIQ` | ok, 40 / 39 |
| `-8WeWWOpxHk` | 10.6 s | `2a592446-0c4b-453a-8f38-ec7884732287` | valid empty (`no_faces_in_video`) | empty (no faces) |

Артефакты: `storage/result_store/youtube/{video_id}/{run_id}/` (локально, не в git).

Ранние прогоны без core_identity / до L9 (mock 3 s) — superseded; см. worklog §0.15 в [E2E_RUNBOOK.md](E2E_RUNBOOK.md).

---

## 5. Исправленные логические ошибки (L1–L12)

Полные описания: [`LOGIC_ERRORS_FOR_CLAUDE.md`](../../DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md).

| ID | Проблема | Fix / workaround |
|----|----------|------------------|
| **L9** | Fetcher worker не видит HF mp4 → segmenter 3 s mock | Копия `hf_videos11/{id}.mp4` → `example/example_videos/` в `e2e_run_hf_videos11.py` |
| **L10** | Stale NPZ ломает §0.2 (`audio_too_short`) | `_clear_caches()`: wipe `result_store` + `state` перед прогоном |
| **L8** | ASR не попадает в TextProcessor | `--real-video`, autogen `_tmp/text_input_autogen.json` в `DataProcessor/main.py` |
| **L11** | EmoNet vendor + OpenFace docker | `vendor_emonet.py`, `setup_e2e_openface.sh`, `OPENFACE_DOCKER_IMAGE` |
| **L12** | `face_identity`: n_frames>0, processed=0 | Normalized bbox в `_crop_face`; ES deps `protobuf`/`ml_dtypes` |
| **L6** | manifest `ok` vs NPZ `empty` | `_component_artifact_dir`: `face_identity` → `core_face_identity/` в `VisualProcessor/main.py` |
| L1–L5, L7 | min_frames, uniqueness, empty_reason, … | E2E-патчи в `e2e_full_max_run.py`, компоненты DP |

---

## 6. Новая инфраструктура и скрипты

### E2E / валидация

| Файл | Назначение |
|------|------------|
| `backend/scripts/e2e_run_hf_videos11.py` | Batch HF: mock-path, cache wipe, `--with-core-identity`, auto seed, `--sync-fixture` |
| `backend/scripts/e2e_verify_hf_results.py` | Регрессия §0.1+§0.2 по `hf_videos11_results.json` |
| `backend/scripts/e2e_validate_output_quality.py` | §0.2 quality gate (NPZ, L6 warnings) |
| `backend/scripts/ci_sync_hf_results_fixture.sh` | Sync `.e2e/state` → CI fixture |

### OpenFace + core_identity

| Файл | Назначение |
|------|------------|
| `backend/scripts/setup_e2e_openface.sh` | Docker OpenFace (`algebr/openface` → `openface/openface:latest`) |
| `DataProcessor/embedding_service/scripts/seed_e2e_hf_face_from_video.py` | Face seed в ES (InsightFace / landmarks crop) |
| `DataProcessor/embedding_service/requirements-e2e.txt` | `protobuf`, `ml_dtypes` для ArcFace |

### CI

| Файл | Назначение |
|------|------------|
| `backend/tests/fixtures/hf_videos11_results.json` | Committed snapshot 5/5 green |
| `backend/tests/unit/test_hf_videos11_results_fixture.py` | Pytest gate (4 теста) |
| `.github/workflows/hf-e2e-regression.yml` | Fixture gate (ubuntu) + artifact-verify (manual, self-hosted GPU) |
| `.github/workflows/backend-ci.yml` | + шаг HF fixture gate |

---

## 7. Ключевые команды (воспроизведение)

```bash
# Инфра + стек
./backend/scripts/start_e2e_stack.sh --with-infra

# HF dataset (один раз)
python example/scripts/download_hf_videos11_samples.py --count 5

# Полный HF batch + core_identity + CI fixture
cd backend && source scripts/e2e_env.sh && source .venv/bin/activate
export DP_MODELS_ROOT="$PWD/../DataProcessor/dp_models/bundled_models"
python scripts/e2e_run_hf_videos11.py --count 5 --with-triton-docker \
  --with-core-identity --sync-fixture

# Регрессия (нужен storage/result_store на диске)
python scripts/e2e_verify_hf_results.py --with-core-identity

# CI fixture (без storage)
pytest tests/unit/test_hf_videos11_results_fixture.py -v
```

**Секреты:** `HF_TOKEN` для pyannote → `backend/.e2e/secrets.env` (gitignored).  
**Embedding Service:** `http://localhost:8005`. **Triton:** `http://127.0.0.1:8010` (docker).

---

## 8. Операционные заметки

1. **Дубликаты Fetcher Celery worker** — задача `download_video` может «пропасть»; симптом: зависание на `DOWNLOADING_VIDEO` 10+ мин. Решение: один worker после `source e2e_env.sh`.
2. **Placeholder face seed из SQL** (`setup_e2e_infra.sql`) не даёт real match — нужен `seed_e2e_hf_face_from_video.py` или auto seed в runner.
3. **§0.2 warnings** — `similarity_metrics` ~38% finite ratio ожидаем (sparse NaN by design), не error.
4. **Артефакты не в git** — полная регрессия на CI только на self-hosted GPU (`hf-e2e-regression` → workflow_dispatch).

---

## 9. Что остаётся вне scope этой итерации

| Задача | Статус |
|--------|--------|
| Prod/k8s: Triton, ES, OpenFace sidecar | Не делалось |
| Расширение HF dataset (>5 видео) | Не делалось |
| Автоматический nightly artifact-verify на self-hosted | Workflow есть, runner — manual dispatch |
| L4 product fix (`similarity_metrics` NaN policy) | Warning only |

---

## 10. Хронология (кратко)

1. Mock green + §0.2 gate + pyannote (`HF_TOKEN`).
2. HF videos11 download; первый batch без L9 — ложный «green» на 3 s mock.
3. L9/L10/L8 fixes → **5/5 real HF** без core_identity.
4. OpenFace + EmoNet (L11) → `micro_emotion` на HF.
5. Embedding Service + face seed + L12 → `face_identity` match.
6. **5/5 + core_identity** + auto seed.
7. L6 manifest sync; re-run `-8WeWWOpxHk` с корректным `empty`.
8. CI fixture + pytest + GitHub Actions workflows.
9. Документация: §0.14–§0.17 runbook, checklist §4.3, этот отчёт.

---

*Документ зафиксирован по итогам сессии E2E 2026-06-30 … 2026-07-02. При новом green batch обновите fixture: `./backend/scripts/ci_sync_hf_results_fixture.sh`.*
