# CRITERIA.md — embedding_source_id_extractor

**Версия компонента:** 1.3.0  
**Дата согласования:** 2026-07-17  
**Тип валидации:** только синтетика (в storage нет text_features.npz)

---

## Универсальные хард-гейты

| ID | Критерий | Порог |
|----|----------|-------|
| U1 | validate_schema / validate_structure / validate_ranges rc=0 на синтетическом NPZ (ok-path + error-path) | rc=0 |
| U2 | N/A — нет оси времени (текстовый агрегатор) | — |
| U3 | all-finite в ok-path features_flat; policy one-hot сумма=1; primary_type one-hot ∈{0,1} | 0 нарушений |
| U4 | absent artifacts + strict_missing_primary=False → valid NPZ (present=0, error-код в payload), без RuntimeError | rc=0, no crash |
| U5 | golden=0: повтор с одинаковым input → bit-identical (SHA256 + pure numpy детерминирован) | max\|Δ\|=0 |
| U6 | N/A — компонент не зависит от длины видео | — |

## Специфичные критерии

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | Все 5 политик (transcript_first/title_first/description_first/title_only/transcript_only) корректно устанавливают policy one-hot=1 и выбирают правильный источник | 5/5 |
| C2 | Все 6 error-кодов (no_embedding_found/unsafe_relpath/embedding_file_missing/embedding_load_failed/embedding_empty/embedding_non_finite) воспроизводятся при strict=False → present=0, valid NPZ | 6/6 |
| C3 | vector_id совпадает с ручным SHA256(float32.C-order bytes)[:24] для ≥3 различных синтетических векторов | 3/3 совпадений |
| C4 | 0 NaN в ok-path features_flat (все 13 полей конечны) | 0 NaN |

## Замечание к вердикту

⚠️ Валидирована на синтетических эмбеддингах.
При появлении text_features.npz в реальном storage рекомендуется:
- Провести batch_runs на 3–5 реальных видео (разные язык/длина текста).
- Повторить C2/C3 на реальных файлах.
- Обновить вердикт в REPORT.

Текущий штамп: «допуск на синтетику v1.3.0».

## NaN-политика

- **ok-path:** 0 NaN (все 13 tp_embid_* конечны)
- **error-path (strict=False):** features_flat конечны (флаги 0/1), payload содержит {"error": "<код>"}
- **Нет полей NaN by design** в ok-path
