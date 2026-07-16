# Критерии приёмки: embedding_stats_extractor

**Согласовано**: 2026-07-17  
**Версия кода**: 1.2.0  
**Тип**: CPU-only, scalar/agg (нет временной оси)

## Универсальные гейты (U1–U6)

| Гейт | Описание | Ожидается |
|------|----------|-----------|
| U1 | validate_embedding_stats_extractor rc=0 на всех storage NPZ | PASS |
| U2 | Ось времени (нет — scalar/agg) | N/A → тривиально PASS |
| U3 | Числа корректны при present=1: l2_variance≥0, topvar≥0, entropy_norm∈[0,1], perplexity≥1 | PASS |
| U4 | Expected-empty (нет чанков): present=0, validate rc=0 | PASS |
| U5 | Golden-детерминизм: 2 прогона → diff=0 (pure numpy) | diff=0 |
| U6 | Разные N чанков (1/2/10/100) — без падений | PASS |

## Специфические критерии (C1–C4)

| Критерий | Описание |
|----------|----------|
| C1 | **NaN by design при present=0**: ровно 13 полей = NaN: `l2_variance`, `n_chunks`, `dim`, `topvar_1..8`, `load_ms`, `compute_ms` (при emit_extra_metrics=False, дефолт). Остальные поля finite. |
| C2 | **l2_variance > 0** при N≥2 отличных чанках (сигнал различимости вектора дисперсии). |
| C3 | **emit_extra_metrics=False → load_ms/compute_ms = NaN** (by design, задокументировано в SCHEMA.md). При True — конечные значения. |
| C4 | **topic_entropy_norm ∈ [0,1], perplexity ≥ 1** при valid topic_probs (present=1, invalid=False). |
