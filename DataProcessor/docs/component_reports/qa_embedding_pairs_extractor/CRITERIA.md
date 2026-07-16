# Критерии приёмки: qa_embedding_pairs_extractor

Согласованы 2026-07-17 (Второй агент от имени владельца).

## Универсальные хард-гейты (U)

| Гейт | Описание | Порог |
|------|----------|-------|
| U1 | Валидатор rc=0 | 100% NPZ pass |
| U2 | Ось времени | **НЕ ПРИМЕНИМА** — текстовый компонент, не использует times_sec/frame_indices; asr.segments читаются как текст (не как временная последовательность) |
| U3 | Finite/health/различимость | 4 NaN by design при present=0 (см. C1); при present=1 — все 34 ключа finite |
| U4 | Expected-empty | disabled=True → disabled_by_policy=1, present=0; нет вопросов → present=0 (valid empty, артефакты не пишутся) |
| U5 | Golden-детерминизм | max\|Δ\| features_flat = 0.0; max\|Δ\| embeddings = 0.0 (ST CPU детерминирован) |
| U6 | Разные длины / объёмы текста | Синтетически: 0/1/5 вопросов, разные источники; storage: 22 ok-NPZ из 5 разных видео ✓ |

## Специфические критерии (C)

| Критерий | Описание | Порог |
|----------|----------|-------|
| C1 | NaN by design при present=0 | Ровно 4 ключа: `embedding_dim`, `questions_per_min`, `questions_per_1k_chars`, `mean_cosine_to_centroid` = NaN. Это 4/34 = 11.8%. При `emit_extra_metrics=False` (прод-дефолт) — всегда 4 NaN независимо от present. |
| C2 | Policy one-hot | `sum(policy_asr_only + policy_asr_then_legacy + policy_legacy_only) == 1.0` ±0.001 |
| C3 | Инвариант счётчиков при present=1 | `sum(q_title + q_description + q_transcript + q_comments) == num_questions` ±0.5 |
| C4 | Dedup корректен | Повторяющиеся вопросы (по canonical form) не дублируются в выходе |
| C5 | Shape/dtype артефакта | При present=1: `qa_question_embeddings.npy` → shape (N, 1024), dtype float32, L2-нормы ∈ [0.999, 1.001], всё finite. Артефакт хранится в `_artifacts/` по дизайну TextProcessor (не нарушение протокола 0.1.2 — паттерн аналогичен `title_embedding.npy`, `description_embedding.npy`). |

## NaN by design (явные исключения)

- `tp_qa_embedding_dim` = NaN при present=0 (нет эмбеддингов)
- `tp_qa_questions_per_min` = NaN при emit_extra_metrics=False (прод) ИЛИ если audio_duration_sec не задан
- `tp_qa_questions_per_1k_chars` = NaN при emit_extra_metrics=False (прод) ИЛИ нет вопросов
- `tp_qa_mean_cosine_to_centroid` = NaN при present=0 ИЛИ emit_extra_metrics=False
- `tp_qa_mean_cosine_to_centroid_present` = 0 при отсутствии mean_cosine

Все остальные 30 ключей при status=ok всегда finite.
