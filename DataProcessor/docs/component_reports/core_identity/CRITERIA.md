# Критерии приёмки: core_identity (группа из 5+1 компонентов)

**Согласовано:** 2026-07-16  
**Компоненты:** place_semantics (v0.2), brand_semantics (v0.2), car_semantics (v0.2), content_domain (v0.2), franchise_recognition (v0.2)  
**face_identity:** ПРОПУСК — требует ручной разметки базы лиц (зона владельца)

---

## Ограничения текущей сессии

- **Embedding Service недоступен** → U5/U6 для place/brand/car/franchise/content_domain пропускаются (одобрено владельцем)
- **Triton недоступен** → U5/U6 для content_domain (CLIP text-retrieval) пропускаются
- U1-U4 проверяются на реальных NPZ из storage (116 штук)

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий | Применимость |
|------|----------|-------------|
| U1 | validate_*.py rc=0 (--struct) на ≥5 NPZ каждого компонента | все 5 |
| U2 | frame_indices строго возрастающие, times_s монотонны | все 5 |
| U3 | Различимость: status=ok → track_topk_scores ∈ [0,1]; status=empty → track_ids пустой | все 5 |
| U4 | status=empty при отсутствии совпадений — valid empty, не exception | подтверждено реальными NPZ |
| U5 | Golden детерминизм (повтор → diff=0) | ПРОПУСК (нет Embedding Service / Triton) |
| U6 | Разные длины (N=5/30/200) без падений | ПРОПУСК (нет Embedding Service / Triton) |

## Критерии компонентов (C1–C3)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | status=ok → frame_topk_scores∈[0,1] (finite), semantic_label_names непустой | ≥1 уникальный лейбл |
| C2 | status=empty → track_ids пустой (len=0), frame_topk_ids все ≤ -1 или 0 | len(track_ids)=0 |
| C3 | content_domain: labels ∈ {2..N} (game/live_action/cartoon/etc.), conf_tracks≥1 при ok | labels≥2, conf_tracks≥1 |

## Исключения

- **face_identity:** SKIPPED (требует ручной разметки), статус `skipped_manual_labeling_required`
- **U5/U6 всех 5 компонентов:** SKIPPED (нет внешних сервисов, одобрено владельцем)
- **franchise_recognition scores=0.0:** нормально при отсутствии franchise в видео (empty tracks допустимы при status=ok)
