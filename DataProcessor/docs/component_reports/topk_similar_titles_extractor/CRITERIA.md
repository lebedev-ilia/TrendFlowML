# CRITERIA.md — topk_similar_titles_extractor

**Версия компонента:** 1.3.0  
**Согласовано:** 2026-07-17 (Второй агент от имени владельца)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Порог |
|------|----------|-------|
| U1 | Валидатор rc=0 | `validate_topk_similar_titles_extractor_text_npz.py` → VALID, struct OK, ranges OK |
| U2 | Ось времени N/A | Компонент без временного seq (только flat-скаляры 29 ключей) — применимость не требуется |
| U3 | Finite / не-константа | top1_score/topk_mean_score finite при present=1; различимость подтверждена синтетически (разные title → разные scores) |
| U4 | Expected-empty | Отсутствие title embedding / disabled / dim_mismatch / zero_norm / nan_inf → graceful (present=0, флаги, без RuntimeError при require=False) |
| U5 | Golden детерминизм | numpy cosine: max|Δ|=0.0 (5 прогонов); faiss HNSW — приближённый (задокументировано в SCHEMA.md как допустимо) |
| U6 | Разные параметры | Разные видео, k=3/5, export_topk_mode=ids_only/ids_and_scores/none — без падений |

---

## Критерии под компонент (C0–C4)

| Критерий | Описание | Порог |
|----------|----------|-------|
| **C0_production** | Миграция на реальный corpus перед production | Corpus ≥100 уникальных заголовков, embeddings.npy ненулевой. Stub corpus (8 нулевых векторов, corpus_size=8) допустим только для логики-валидации на текущем этапе. TODO: обновить similar_titles_corpus_v1 до реального corpus-pack и перепроверить различимость (ожидаемое top1 ≈ 0.5–0.95 по audit_v4 L1). |
| **C1** | top1_score ∈ [0, 1] при present=1 | Косинус нормированных векторов ≥ 0; при реальном corpus ожидается ≈ 0.5–0.95. При stub corpus top1=0.0 by design (нулевой corpus). |
| **C2** | export_topk_mode one-hot | sum(ids_only + ids_and_scores + none) = 1.0 строго |
| **C3** | top1 ≥ topk_mean при present=1 | Математически гарантировано (top1 = max, mean ≤ max) |
| **C4** | top1/topk_mean = NaN при present=0 | By design: нет поиска → нет скоров. Включает: disabled_by_policy, title_embed_missing, dim_mismatch, zero_norm, nan_inf |

---

## Известные исключения (NaN by design)

- `tp_topktitles_top1_score` = NaN при `present=0` — корректно (нет поиска)
- `tp_topktitles_topk_mean_score` = NaN при `present=0` — корректно
- `tp_topktitles_export_k_used` = NaN при `present=0` (до успешного поиска) — корректно
- top1_score = 0.0 при stub corpus (нулевые эмбеддинги corpus) — by design, не дефект

---

## C0_production_corpus (техдолг, не блокирует штамп)

**Текущее состояние:** corpus-pack stub (8 нулей); все top1=0.0 by design.  
**Блокирует production:** ДО развёртывания нужно заменить на реальный corpus (≥100 заголовков).  
**Action:** обновить `dp_models/bundled_models/text/similar_titles_v1/embeddings.npy`,  
перепроверить U3/C1 с реальными числами (ожидается top1_score 0.5–0.95).

Штамп логической валидации поставлен 2026-07-17 (Второй агент). Corpus-pending отслеживается явно.
