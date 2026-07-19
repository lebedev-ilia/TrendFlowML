# FINAL REPORT — `core_identity/franchise_recognition`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Часть группы `core_identity`. Общий отчёт — [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md).
> Родственник place/brand/car (retrieval), но **мультимодальный** (CLIP-кадры **+ OCR-текст**).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `franchise_recognition` (VisualProcessor core_identity, semantic head, Tier-2) |
| Версия кода | `0.2` |
| Схема NPZ | `franchise_recognition_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/franchise_recognition/franchise_recognition.npz` |
| Модель | **CLIP-224** (core_clip frame-эмбеддинги) + **OCR-текст** + **Embedding Service** (kNN vs franchise-DB) |
| Hard deps | `core_clip` + Embedding Service; soft — `ocr_extractor` (текст) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_identity ✅ (2026-07-16, авто) |
| Отчёт валидации | [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md), [`CRITERIA.md`](../CRITERIA.md) |
| Код | `core/model_process/core_identity/franchise_recognition/main.py` |

## 2. Резюме

`franchise_recognition` — **распознавание конкретного тайтла/франшизы** (игры, аниме, мультфильмы, IP) через
**мультимодальный retrieval**: фьюзит визуальные CLIP-эмбеддинги кадров (`core_clip`) и **OCR-текст** (названия/
логотипы на экране) и ищет франшизу в базе **Embedding Service**. На реальном корпусе все 23 NPZ формально
`status=ok`, **но фактически пусто**: franchise-DB — **плейсхолдер-стаб** (label space `A=1` = `'0:e2e_seed_
placeholder'`), а OCR-текст пуст везде (нет `text_region`-детектора). Компонент эмитит **вырожденный 1 трек на
видео со score 0.0/NaN** — «нашёл франшизу», но с нулевой уверенностью. Это **хуже чистого empty** — вводящий в
заблуждение `status=ok` без реального сигнала. Ни одной франшизы за корпус не распознано. Качество не проверено
(U5/U6 SKIP). v0.2.

## 3. Функционал

Tier-2, semantic head, мультимодальный:

1. **Визуальный канал** — CLIP-эмбеддинги кадров `core_clip` (общая выборка Segmenter).
2. **Текстовый канал** — OCR-текст (`ocr_extractor`): названия/логотипы франшиз на экране (`_load_ocr_npz`).
3. **Embedding Service kNN** — ищет франшизу в базе (`category=franchise`) по обоим сигналам.
4. Группирует в трек с `track_topk_evidence_frame_indices` (где франшиза видна).
5. Возвращает per-track/frame top-K франшиз + флаги уверенности.

**Зачем продукту:** конкретная франшиза/IP — **признак жанра/фандома**: let's-play (какая игра), аниме-обзор
(какое аниме), фан-контент (какой IP). Ценно для категоризации и рекомендаций внутри фандомных вертикалей.
Мультимодальность оправдана: франшиза узнаётся и по визуальному стилю, и по тексту-логотипу.

## 4. Вход

- **`core_clip`** (hard, no-fallback) — `frame_embeddings` + `frame_indices`.
- **`ocr_extractor`** (soft) — OCR-текст для текстового канала (пуст на реальных данных).
- **Embedding Service** (hard, fail-fast) — franchise-DB (`db_digest`); нет базы/совпадений → нет реального матча.
- **`union_timestamps_sec`** — ось.

## 5. Выход

- **Label-space:** `semantic_label_names (A)` (сейчас 1 = placeholder), `semantic_object_ids`, `threshold_per_label_arr`.
- **Per-track:** `track_ids`, `track_topk_ids/scores`, `track_topk_evidence_frame_indices`, `track_is_confident_top1`.
- **Per-frame:** `frame_topk_ids/scores`, `frame_is_confident_top1`. Ось: `frame_indices`, `times_s`.
- **На реальных данных:** 1 вырожденный трек, `track_topk_scores` (1,5) = 0.0/NaN, conf=False.

## 6. Фичи (важное/неочевидное)

- **Мультимодальность (CLIP + OCR)** — сильный дизайн: франшиза = визуальный стиль ИЛИ текст-логотип. Но **оба
  канала мертвы**: franchise-DB стаб + OCR пуст (нет text_region-детектора).
- **`status=ok` без сигнала (антипаттерн)** — в отличие от place/brand/car (чистый `status=empty` при нет-матчей),
  franchise эмитит вырожденный трек со score 0.0/NaN и `status=ok`. Downstream может принять «ok» за наличие
  франшизы → вводящее в заблуждение поведение (стоит переводить в empty при нулевой уверенности).
- **franchise-DB = `e2e_seed_placeholder`** — единственная seed-метка, реальной базы франшиз нет.
- **`evidence_frame_indices`** — где франшиза замечена (для UI) — хорошая идея, но пусто.

## 7. Архитектура / алгоритм

- **CLIP-224 (core_clip) + OCR-текст → Embedding Service kNN (franchise category)** → temporal track.
- Retrieval, не обучается; качество ∝ franchise-DB + OCR. **v0.2.** Детерминизм не проверен (U5 SKIP).

## 8. Оптимизации

- **Reuse core_clip-эмбеддингов** (не эмбеддит кадры заново).
- **Мультимодальный фьюз** (визуал + текст) — потенциально точнее одного канала.
- **evidence-кадры** — компактная привязка «где франшиза».
- **db_digest provenance**.

## 9. Слабые места

- **Оба канала мертвы → 0 франшиз (главное).** franchise-DB стаб (`A=1` placeholder) + OCR-текст пуст (нет
  text_region-детектора). Компонент **не распознал ни одной франшизы** за корпус.
- **`status=ok` вместо empty при нулевой уверенности (антипаттерн)** — эмитит вырожденный трек со score 0/NaN;
  вводит downstream в заблуждение (лучше `status=empty`). Хуже, чем чистый empty у place/brand/car.
- **Тройная зависимость** — core_clip (есть) + franchise-DB (стаб) + OCR (пуст); два из трёх отсутствуют.
- **Качество не проверено** — U5/U6 SKIP, 0 матчей.
- **v0.2**, узкая вертикаль (фандом/let's-play/аниме).
- **Наследует OCR-блокер** — text_region-детектор (общий с ocr/brand/text_scoring).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Перевести нулевую уверенность в `status=empty`** — не эмитить вырожденный ok-трек со score 0/NaN
   (антипаттерн, вводит в заблуждение).
2. **[выс.] Наполнить franchise-DB** — реальные франшизы/IP (игры/аниме/мультфильмы) вместо placeholder.
3. **[выс.] Оживить OCR-канал** — text_region-детектор (общий блокер) для текстового узнавания франшиз-логотипов.
4. **[сред.] Проверить мультимодальный фьюз** на фандом-контенте (let's-play/аниме-обзор), U5/U6.
5. **[низ.] Оценить продуктовый scope** — узкая вертикаль; приоритет ниже brand.

## 11. Рекомендации по архитектуре / связям

- **Общий text_region-детектор** — единый блокер для franchise/brand/ocr/text_scoring; решать в первую очередь.
- **Общий Embedding Service + retrieval** с place/brand/car — наполнять базы согласованно.
- **Мультимодальный фьюз** — reuse и core_clip, и OCR; закрепить, когда оба канала живы.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct | 23 NPZ | 23/23 VALID | схема ок |
| U2 ось | 23 | fi↑, ts монотонны | ось ок |
| U3 различимость | 23 | все ok, scores∈[0,1], ok_with_nonzero=0 | реального матча нет |
| U5/U6 | — | SKIP | не проверено |
| **Реальный storage (мой прогон)** | **6 видео** | **все ok, но 1 вырожденный трек, score 0/NaN, franchise-DB A=1 placeholder** | 0 франшиз; оба канала мертвы; status=ok вводит в заблуждение |

Вывод: схема/ось валидны, но **реального распознавания нет** (стаб-база + пустой OCR), а `status=ok` при нулевой
уверенности — антипаттерн.

## 13. Интерпретируемость

- **Потенциально отличная** (в фандом-нише): «в видео: франшиза X» + evidence-кадры.
- **Сейчас нечего показывать** — placeholder, score 0. `status=ok` не должен трактоваться как «франшиза найдена».

## 14. Польза для моделей

**Нишевая, фактически нулевая.** Франшиза/IP — признак фандом-вертикали; sparse-выход, оба канала мертвы →
0 сигнала. `status=ok` без уверенности может даже навредить (ложный сигнал). Потенциал ограничен нишей.

## 15. Польза для аналитиков

**Нишевая, фактически нулевая.** «Какая игра/аниме/IP в видео» ценно для фандом-контента, но на корпусе пусто, а
`status=ok` вводит в заблуждение. До наполнения базы и OCR — ничего.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Мультимодальный дизайн осмыслен, но узкая ниша + оба канала мертвы |
| 5. Выход (контракт) | 3 | Богатый per-track/frame + evidence; но status=ok-антипаттерн |
| 6. Фичи | 2 | Фьюз CLIP+OCR продуман, но placeholder-DB + пустой OCR |
| 8. Оптимизации | 3 | Reuse core_clip, мультимодальность, evidence |
| 9. Слабые места (инверсно) | 1 | Оба канала мертвы, status=ok-антипаттерн, тройная зависимость, v0.2 |
| 12. Результаты тестов | 2 | Схема/ось ок, но 0 матчей, U5/U6 SKIP |
| 13. Интерпретируемость | 3 | «Франшиза X» понятно (когда есть); сейчас нечего |
| 14. Польза для моделей | 2 | Нишевый потенциал, факт=0 (+риск ложного ok) |
| 15. Польза для аналитиков | 2 | Нишевый потенциал, факт=0 |

### Итоговые оценки

- **Польза для моделей: 2/5.** Франшиза/IP — признак узкой фандом-вертикали; мультимодальный дизайн (CLIP+OCR)
  разумен, но оба канала мертвы (placeholder-DB + пустой OCR) → 0 сигнала, а `status=ok` при нулевой уверенности
  рискует дать ложный сигнал. Фактическая польза нулевая.
- **Польза для аналитиков: 2/5.** «Какая франшиза/IP» ценно для фандом-контента, но на всём корпусе пусто и
  вводящий в заблуждение `status=ok`. Балл отражает факт (0 сигнала) и узкую нишу.

## 17. Источники

- `core/model_process/core_identity/franchise_recognition/main.py`, `docs/SCHEMA.md`, `utils/*`
- `DataProcessor/docs/component_reports/core_identity/{REPORT_2026-07-16.md, CRITERIA.md}`
- Cross-ref: `core_clip` (визуал), `ocr_extractor` (текст-канал, пуст), `brand_semantics`/`content_domain` (retrieval/zero-shot), Embedding Service
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/franchise_recognition/franchise_recognition.npz`
  (**все status=ok, но 1 вырожденный трек, score 0/NaN, franchise-DB A=1 placeholder; OCR пуст**)

## 18. Визуализации

![franchise_recognition overview](franchise_recognition_overview.png)

`franchise_recognition_overview.png`: слева — все 6 видео `status=ok`, но вырожденно (1 трек, score 0/NaN,
0 реальных франшиз); справа — мультимодальный механизм (CLIP-кадры + OCR-текст → Embedding Service kNN) и
**двойная деградация** (franchise-DB = `e2e_seed_placeholder` стаб; OCR пуст — нет text_region-детектора) +
антипаттерн `status=ok` при нулевой уверенности. Подтверждает: дизайн осмыслен, но реального сигнала нет и статус
вводит в заблуждение.
