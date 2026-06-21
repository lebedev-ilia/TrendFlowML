# AudioProcessor — Wave 2 Normalization

Этап нормализации `AudioProcessor` (portfolio + production).  
Управляющий план: [../../docs/PORTFOLIO_NORMALIZATION_PLAN.md](../../docs/PORTFOLIO_NORMALIZATION_PLAN.md)  
Журнал: [../../docs/PORTFOLIO_PROGRESS_LOG.md](../../docs/PORTFOLIO_PROGRESS_LOG.md)

---

## Статус: `done` (документация и навигация; 2026-05-28)

---

## 1. Структура модуля (as-is)

| Путь | Роль | Policy |
|------|------|--------|
| `main.py`, `run_cli.py` | CLI entrypoints | editable |
| `src/core/` | Orchestrator, dependency resolver, NPZ saver | editable |
| `src/extractors/*` | 21 extractor | editable |
| `src/api/` | FastAPI (если используется отдельно) | editable |
| `schemas/` | Machine NPZ schemas | editable |
| `config/` | AudioProcessor configs | config |
| `docs/` | Документация, audit, testing | editable |
| `scripts/` | Smoke/full per-extractor scripts | editable |

**Extractors (21):**  
`asr_extractor`, `band_energy_extractor`, `chroma_extractor`, `clap_extractor`, `emotion_diarization_extractor`, `hpss_extractor`, `key_extractor`, `loudness_extractor`, `mel_extractor`, `mfcc_extractor`, `onset_extractor`, `pitch_extractor`, `quality_extractor`, `rhythmic_extractor`, `source_separation_extractor`, `speaker_diarization_extractor`, `spectral_entropy_extractor`, `spectral_extractor`, `speech_analysis_extractor`, `tempo_extractor`, `voice_quality_extractor`

---

## 2. Каноничный layout документации extractor

Целевая структура (одинаково для всех 21):

```text
src/extractors/<name>/
  docs/
    README.md              # контракт: вход, зависимости, выход, sampling family
    SCHEMA.md              # human schema (NPZ keys)
    FEATURE_DESCRIPTION.md # описание фич для audit/portfolio
    TESTING_REPORT.md      # опционально, результаты прогонов
  utils/                   # validate_*.py, render
  main.py / extractor impl
```

**Prod-правило:** один canonical путь к README — `docs/README.md`, не корень extractor.

---

## 3. Найденные проблемы (Wave 2 inventory)

### 3.1 Дубликаты `FEATURE_DESCRIPTION.md`

Файл есть и в `docs/`, и в корне extractor (устаревший дубль):

| Extractor | Дубль в корне |
|-----------|---------------|
| `asr_extractor` | да |
| `band_energy_extractor` | да |
| `chroma_extractor` | да |
| `clap_extractor` | да |
| `emotion_diarization_extractor` | да |
| `hpss_extractor` | да |

**Action (done v1):** корневые файлы заменены на stub → `docs/FEATURE_DESCRIPTION.md`. Содержимое в `docs/` и корне **различалось** — при необходимости ручной merge устаревших фрагментов из git history.

### 3.2 Рассинхрон ссылок в `docs/MAIN_INDEX.md`

Часть ссылок указывает на `../src/extractors/<name>/README.md`, фактические README — в `docs/README.md`.

**Action (done v1):** исправлены битые ссылки (`hpss`, `mfcc`, `quality` → `docs/README.md`).

### 3.3 Runtime / dev артефакты

- `docs/LAST_FULL_RUN_LOG.md` — dev log с локальными путями; не canonical runbook.
- Smoke/full results: `dp_results/smoke_test/`, `dp_results/full_test/` — generated.

**Action:** LAST_FULL_RUN_LOG — перенести в progress/audit или пометить как historical; не использовать как entry doc.

---

## 4. Категории extractors (для prod narrative)

| Категория | Extractors | Prod-заметки |
|-----------|------------|--------------|
| **Tier-0 / baseline** | `clap_extractor`, `loudness_extractor`, `asr_extractor` | Required paths, строгие контракты v1/v2 |
| **Model-heavy / GPU** | `asr_extractor`, `clap_extractor`, `emotion_diarization_extractor`, `source_separation_extractor`, `speaker_diarization_extractor` | ModelManager-only, offline |
| **Spectral family** | `band_energy`, `chroma`, `spectral`, `spectral_entropy`, `mel`, `mfcc`, `pitch` | Shared Segmenter family `spectral` / dedicated |
| **Speech / diarization** | `asr`, `speech_analysis`, `speaker_diarization`, `voice_quality` | Downstream TextProcessor зависит от ASR |
| **Optional / analytics** | многие с feature-gates | Документировать optional vs model_facing |

---

## 5. DoD Wave 2

- [x] Все 21 extractor: единый layout `docs/{README,SCHEMA,FEATURE_DESCRIPTION}.md` (проверено 2026-05-28)
- [x] Корневые `FEATURE_DESCRIPTION.md` (6 шт.) — stub → canonical `docs/`
- [x] `docs/MAIN_INDEX.md` — ссылки унифицированы → `docs/README.md`
- [x] `README.md` (корень AudioProcessor) — ссылки на Wave 2, EXTRACTOR_DEPENDENCIES, progress log
- [x] Зависимости: [EXTRACTOR_DEPENDENCIES.md](EXTRACTOR_DEPENDENCIES.md) (families, optional deps, smoke checklist)
- [x] Prod smoke checklist в EXTRACTOR_DEPENDENCIES §7

---

## 6. Следующий шаг (immediate)

1. Добавить ссылки Wave 2 / EXTRACTOR_DEPENDENCIES в корневой `AudioProcessor/README.md`.
2. Проверить layout всех 21 extractors (наличие SCHEMA.md).
3. Закрыть Wave 2 → Wave 3 (`TextProcessor`).
---

## Навигация

[README](README.md) · [Module README](../README.md) · [AudioProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
