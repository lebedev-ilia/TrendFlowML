# Критерии приёмки: semantics_topics_keyphrases

**Дата согласования:** 2026-07-17  
**Версия компонента:** SemanticTopicExtractor v2.1.0  
**Схема выхода:** semantics_topics_keyphrases_output_v1 (116 ключей `tp_topics_*`)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий | Порог | Примечание |
|------|----------|-------|------------|
| U1 | Batch-валидатор rc=0 на всех NPZ с полным срезом | 100% OK | Исторические NPZ без tp_topics_* (старые прогоны) исключаются (архивные артефакты) |
| U2 | Ось времени согласована | N/A | Текстовый компонент, нет временной оси |
| U3 | Различимость: top1_id ≥ 2 уникальных значений на ≥ 6 видео | ≥ 2 уник. | На 6 видео storage: {1,2} — ✓ |
| U4 | Expected-empty: disabled=True и empty-text → 116 ключей, present=0 | rc=0 | Синтетически; torch не нужен (после фикса) |
| U5 | Golden-детерминизм: max\|Δ\| = 0.0 (без timing-полей) | 0.0 | 16 прогонов одного видео: max\|Δ\|=0.0 ✓ |
| U6 | Разные длины текстов (разные chars) отрабатывают без падений | pass/fail | chars диапазон 166–169+ на storage-данных |

---

## Критерии компонента (C1–C4)

| Критерий | Описание | Порог |
|----------|----------|-------|
| C1 | `tp_topics_extra_*` = NaN при `emit_extra_metrics=False` | 100% NaN (by design) — не дефект |
| C2 | `tp_topics_kp_top*_hash01/len` = NaN при `export_keyphrases_mode=none` | 100% NaN (by design) — режим экспорта "none" не заполняет хеш-слоты |
| C3 | `tp_topics_topic_top6..8_*` = NaN при `top_k_slots=5` | 100% NaN (by design) — слоты 6-8 заполняются только при top_k_slots≥6 |
| C4 | `tp_topics_entropy_topk` ∈ [0, ln(top_k_topics)] при present=1; `tp_topics_topic_top1_prob` ∈ [0,1] | ∈ допустимом диапазоне | На storage: entropy ∈ [1.603, 1.604], prob ∈ [0.235, 0.242] |

---

## NaN-политика (явная документация)

- **`extra_*` поля (5 штук):** NaN при `emit_extra_metrics=False` — штатно, не дефект
- **`kp_top*_hash01` и `kp_top*_len` (32 поля):** NaN при `mode=none` — штатно; при `mode=hashed` должны быть финитны в заполненных слотах
- **`topic_topN_*` слоты выше top_k_slots:** NaN — штатно
- **Disabled/empty-text путь:** present=0, ~67 NaN-полей — штатно

---

## GPU-прогон для финального штампа

Требуется прогон на 5–6 видео разных длин (10s–8min, разный контент) на поде RunPod.  
Проверяет: U1, U3, U6 на реальных данных; подтверждает torch-инференс работает корректно.
