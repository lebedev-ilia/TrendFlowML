# CRITERIA: hashtag_embedder

**Версия**: v1.x | **Дата**: 2026-07-16

## Универсальные гейты (U1–U6)

| Гейт | Критерий |
|------|----------|
| U1 | Валидатор rc=0 (batch) |
| U2 | feature_names.size == feature_values.size; dim∈{384,768,1024} |
| U3 | l2_norm≈1.0 при present=1; nan_rate=0 |
| U4 | Absent hashtags: graceful (present=0, rc=0) |
| U5 | Golden: max\|Δ\|=0.0 (sentence-transformers детерминирован) |
| U6 | n_input_tags или l2_norm CV>0% (хоть что-то различимо) |

## Специфичные критерии (C1–C3)

| Код | Критерий |
|-----|----------|
| C1 | dim=1024, shape=(1024,) |
| C2 | l2_norm=1.0000 (±0.001) при present=1 |
| C3 | sim=1.0 при одинаковых тегах — by design (все видео с mock n_unique=3 tags) |

## Примечание

- Все 6 тестовых видео имеют n_input=3, n_unique=3 одинаковых тега → sim=1.0 by design.
