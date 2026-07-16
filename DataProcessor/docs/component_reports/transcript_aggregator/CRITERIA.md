# Критерии приёмки: transcript_aggregator

**Компонент:** `TranscriptAggregatorExtractor` v1.3.0  
**Схема:** `transcript_aggregator_output_v1` (19 ключей `tp_tragg_*`)  
**Дата:** 2026-07-17  
**Одобрено:** владелец (авто-штамп при 100% PASS)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Метод проверки |
|------|----------|----------------|
| U1 | Батч-валидатор rc=0 (28/28 NPZ) | `validate_transcript_aggregator_text_npz.py --results-base storage/result_store --platform-id youtube` |
| U2 | Ось времени N/A — компонент выдаёт только скалярные аггрегаты + .npy эмбеддинги (нет seq) | структурный анализ |
| U3 | core-поля finite 22/22; agg_mean L2-норма=1.0; cosine_sim между видео ∈ [0.705–0.884] | numpy проверка |
| U4 | missing chunk file + require_chunks=False → present=0, error=None (no crash) | синтетический тест |
| U5 | golden: mean max\|Δ\|=0.0, max max\|Δ\|=0.0 (чистый numpy/torch, детерминирован) | 2 прогона на storage chunks |
| U6 | N=1 чанк и N=3 чанка работают: norm=1.0, finite | синтетический тест |

---

## Критерии компонента (C1–C4)

| Критерий | Описание | Порог / исключение |
|----------|----------|--------------------|
| C1 | `emit_extra_metrics=False` (дефолт) → `n_chunks_*` / `std_*` = NaN | **NaN by design** — норма, не дефект |
| C2 | `youtube_auto present=0` во всех ok-видео | **by design** — нет youtube auto-captions в датасете Fetcher |
| C3 | agg embedding L2-норма = 1.0 (все ok-видео) | норма = 1.0 ± 1e-4 |
| C4 | cosine_sim между разными видео (различимость) | ∈ [0.7, 0.9] на корпусе 5+ видео |

---

## Примечания

- Компонент CPU-only, GPU не нужен.
- Модель (`intfloat/multilingual-e5-large`) используется **только для метаданных** (weights_digest через dp_models), forward pass отсутствует.
- В production рекомендуется `emit_extra_metrics=True` для мониторинга n_chunks.
- `require_chunks=True` только при жёстком требовании наличия транскрипта (не default).
