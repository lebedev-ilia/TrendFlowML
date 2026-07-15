# Q4: миграция brand/car/face на embeddings-direct (план)

Цель — убрать per-crop HTTP `/search` в Embedding Service из горячего пути
`brand_semantics`, `car_semantics`, `face_identity` и считать similarity локально
(как уже сделано в `place_semantics`). Это снижает латентность/стоимость на 200k.

Изменение трогает горячий путь хедов → делается **отдельным PR с review** и
прогоном на пилотной пачке. Безопасный фундамент уже готов и протестирован:
`core_identity/_shared/gallery_match.py` (чистый numpy, 4/4 теста).

## Что меняем (паттерн, одинаковый для 3 хедов)

Было (per-crop round-trip):
```
для каждого кропа:  client.search(category=..., image=crop, top_k=k)   # HTTP на кроп
```

Станет (embeddings-direct):
```
# 1) галерея — ОДИН раз на запуск
labels, gallery = client.get_all_embeddings(category=...)   # (A,), (A, D) L2-norm
# 2) эмбеддинги кропов — батчем через тот же extractor, что и при наполнении базы
crop_embs = embed_crops_batch(crops)                        # (M, D) L2-norm
# 3) similarity и top-k — локально, без сети
idx, score = gallery_match.topk_cosine(crop_embs, gallery, k)   # (M,k),(M,k)
# 4) per-track агрегация
labels_tr, scores_tr = gallery_match.aggregate_track_topk(idx_per_track, score_per_track, k)
```

`idx` — это индексы строк галереи; маппинг в канонический label-space берём из
`labels`/`semantic_label_names` (как уже делает place/face: get_labels()).

## Пошагово (на каждый хед)

1. Добавить в `embedding_service_client.py` метод `get_all_embeddings(category)` —
   у `place_semantics` он уже есть; переиспользовать/вынести в общий клиент.
2. Заменить цикл `search` на: один `get_all_embeddings` + локальный `topk_cosine`.
   Эмбеддинги кропов считать батчем (тот же CLIP-extractor, что в `sync_known_*`).
3. Сохранить контракт NPZ без изменений (те же поля/схема v2), только источник
   similarity другой → числа должны совпасть с прежними в пределах допуска.
4. Проверка регрессии: `golden_batch_compare.py` (старый vs новый прогон, §5
   playbook) на одной пачке — близость чисел в пределах `--rel-eps/--abs-eps`.

## Критерии готовности

- `golden_batch_compare` без значимых расхождений на пилоте.
- p95 времени хеда снизился (метрики `dataprocessor_component_stage_seconds`).
- No-network в рантайме сохранён (только локальная арифметика + один pull галереи).

## Риски

- Разный препроцессинг кропа при локальном эмбеддинге vs тот, что был на стороне ES
  → строго использовать ту же модель/препроцесс (`clip_image_336` и т.д.).
- Кэш галереи: инвалидация при смене `db_version`/`db_digest`.

## Статус
- ✅ `_shared/gallery_match.py` + тесты (фундамент).
- ⏳ внедрение в 3 хеда — отдельный PR с review (горячий путь).
