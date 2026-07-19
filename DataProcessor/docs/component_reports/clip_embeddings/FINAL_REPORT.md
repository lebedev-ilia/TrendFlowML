# FINAL REPORT — `clip_embeddings` (action_recognition v3 seq-token)

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> **Что это на самом деле:** `clip_embeddings` — не отдельный модуль, а **главный model-facing выход
> компонента `action_recognition`** (schema `action_recognition_npz_v3`). В деп-дайв-чеклист он попал
> отдельной строкой, потому что `stamped_components()` разбирает фич-леджер валидации и подхватил его как
> «заштампованное имя». Разбирается как самостоятельная фича-группа; сёстры — `clip_times_s/clip_frame_indices`
> и `clip_track_id`. Полный разбор родителя — см. будущий `action_recognition/FINAL_REPORT.md`.

## 1. Метаданные

| Поле | Значение |
|---|---|
| Фича-группа | `clip_embeddings` (+ `clip_track_id`, `clip_times_s`, `clip_frame_indices` — ось) |
| Родитель | `action_recognition` (VisualProcessor `BaseModule`, Tier-2, GPU-backbone) |
| Схема NPZ (целевая) | `action_recognition_npz_v3`, `clip_embeddings (C,D)` float32 L2 |
| Артефакт | `result_store/<platform>/<video>/<run>/action_recognition/action_recognition_features.npz` |
| Модель | **SlowFast R50** (Kinetics-400, torchvision/pytorchvideo); penultimate-фичи через forward-hook |
| Размерность D | динамическая: **≈2304** для slowfast_r50 (penultimate), fallback 400 (logits) / 256 (v2) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → action_recognition v3 ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05_FINAL.md`](../action_recognition/REPORT_2026-07-05_FINAL.md), `RUN_RESULT_v3*.md`, `ASSESSMENT_action_recognition.md` |
| Контракт | [`COMPONENT_CONTRACTS.md`](../../COMPONENT_CONTRACTS.md) → `action_recognition → Models/Encoder` |
| Код | `modules/action_recognition/utils/action_recognition_v3.py` (`build_v3_arrays`), `utils/backbones.py` (penultimate-hook) |

## 2. Резюме

`clip_embeddings` — это **action-токен** системы: для каждого клипа (окно ≥`clip_len` кадров вокруг трека
человека) SlowFast прогоняет видео-клип и выдаёт **penultimate-вектор** (вход классификатора, ≈2304-d),
который L2-нормируется и кладётся в плоский, отсортированный по времени стрим `(C, D)`. Это единственный в
системе выход, несущий **пространственно-временную динамику действия** (не по кадру, а по короткому ролику),
и он прямо предназначен как seq-токен для VisualEncoder. Ось клипа задаётся `clip_times_s`/`clip_frame_indices`
(⊆ union), привязка к персонажу — `clip_track_id` (из appearance-трекера core_object_detections). В v3-валидации
доказан: `(44,2304)` L2/finite, golden побитово идентичен, ~4 клипа/трек. **Главная оговорка:** в реальном
storage-корпусе стрим `clip_embeddings` пока **не материализован** — все артефакты там ещё v2 (256-d
per-track object-array), а машинная `SCHEMA.md` тоже описывает v2 (доковый дрейф).

## 3. Функционал

Стоит в Tier-2 визуальной цепочки, **после** core_object_detections (нужны person-детекции + `track_ids`) и
Segmenter (dense-окна). Пайплайн токена:

1. Segmenter даёт плотные окна `clip_len × window_len_mult` (v3.2 = 96 кадров) вокруг треков.
2. `core_object_detections` (schema v3, appearance-tracker) даёт `track_ids` — какой человек в кадре.
3. Для каждого клипа собирается тензор кадров → SlowFast инференс → **два выхода**: логиты (400 Kinetics) и
   **penultimate-фичи** (снимаются forward-pre-hook'ом со входа классификатора).
4. `build_v3_arrays` сортирует клипы по времени, L2-нормирует эмбеддинги, пришивает `clip_track_id`,
   `clip_times_s`, `clip_frame_indices`, `clip_segment_id`, считает агрегаты (Kinetics-гистограмма и т.д.).

**Зачем продукту:** большинство визуальных фич — покадровые (CLIP, depth, detections). `clip_embeddings` —
**единственная фича «что происходит в движении»**: танец, спорт, разговор, готовка. Для предсказания
популярности тип и динамика действия — сильный семантический сигнал (жанр/формат контента), а плотность
~4–5 клипов на трек даёт Encoder'у **траекторию действия** во времени, а не одну метку.

## 4. Вход

Контракт (`COMPONENT_CONTRACTS.md`, hard-deps):

- **person-детекции** (`core_object_detections/detections.npz`: boxes/scores/class_ids person) — hard; нет
  person → весь `action_recognition` = valid `empty` (`no_person_detections`), `clip_embeddings=(0,D)`.
- **`track_ids`** (schema v3 appearance-tracker, `(N,MAX) int32`, `-1`=невалидный слот) — hard для связи клип↔трек.
- **общий `frame_indices`** (shared sampling group) — hard, должен совпадать с сэмплом action_recognition.
- **Segmenter dense-окна** ≥`clip_len` — иначе SlowFast = 1 клип/трек (была корневая проблема v1).
- **`union_timestamps_sec`** — ось для `clip_times_s` (⊆ union, монотонно).
- **SlowFast веса** (backbone, GPU/CPU, fp16-опция) — внешняя предобученная сеть.

## 5. Выход

Плоский per-clip стрим (v3), `clip_embeddings` — центральный ключ:

- **`clip_embeddings (C, D)` float32, L2** — model_facing, hard. C = число клипов (по всем трекам, отсортировано
  по времени), D — размерность penultimate-фич backbone (**динамическая**, в `meta.embedding_dim`; режим в
  `meta.embedding_mode`: `penultimate`|`projection_fallback`). Encoder читает D динамически.
- **Ось (сёстры):** `clip_times_s (C,)` (⊆ union, монотонно), `clip_frame_indices (C,)` (центр клипа, union),
  `clip_track_id (C,) int32` (`-1` если нет трека), `clip_segment_id (C,)` (temporal localization).
- **Аналитика (soft):** `clip_topk_action_ids/probs`, `class_names (400,)`, `video_action_hist (400,)`,
  `dominant_action_ids/probs`, `num_tracks`, `mean_clips_per_track`, `clip_count`, `num_action_segments`.

Смысл группировки: `clip_embeddings` + ось = **seq-вход Encoder'а**; классы/гистограмма = аналитика.
`clip_track_id` позволяет Encoder'у/аналитику собрать траекторию конкретного персонажа.

## 6. Фичи (важное/неочевидное)

- **Эмбеддинг = настоящие penultimate-фичи, НЕ проекция логитов.** Ключевое улучшение v3: снимается вход
  классификатора (2304-d у slowfast_r50), а не 400-мерные логиты. Penultimate несёт богатую сжатую
  репрезентацию действия, логиты — только «уверенность по 400 классам». Это делает токен пригодным для
  Encoder'а как полноценный семантический вектор.
- **L2-нормировка** (`F.normalize(penult, p=2, dim=1)`) — все эмбеддинги на единичной сфере → косинус =
  скалярное произведение, стабильно для attention и temporal-jump метрик.
- **Динамическая D + `embedding_mode`** — при отказе hook'а (нет `classifier`-модуля) fallback на
  L2(logits), D=400, режим `projection_fallback`. Encoder обязан читать D из меты, не хардкодить.
- **`clip_track_id`** привязывает клип к персонажу через appearance-трекер (эмбеддинг бокса) — чинит
  фрагментацию (v1: 205 треков → v3: ~26/11/16). Плотность ~4 клипа/трек = траектория действия.
- **Сортировка по времени (stable argsort)** — стрим монотонен по `clip_times_s`, что важно для позиционного
  кодирования Encoder'а (time-embedding от `t_center/duration`).

## 7. Алгоритм / архитектура

- **Модель:** SlowFast R50, Kinetics-400 (внешняя предобученная, не дообучается). Backbone-абстракция
  (`backbones.py`) поддерживает также VideoMAE/VideoMAEv2/Hiera как альтернативы (по baseline-ablation).
- **Снятие эмбеддинга:** `register_forward_pre_hook` на модуле-классификаторе перехватывает его вход
  (`_penult_buf`); если тензор >2D — усредняется по промежуточным осям до `(B, hidden)`. Затем L2.
- **Сложность:** тяжёлый 3D-CNN инференс на клип (окно 96 кадров). Бюджет кадров 1536 (16 окон×96),
  диск ~10.7 GB/видео. fp16-опция на CUDA. Детерминизм: fp32-backbone + детерминированный трекер → golden ×2 идентичны.
- **Где идёт:** GPU (прод), CPU-возможен. Батч-путь v3-safe.

## 8. Оптимизации

- **penultimate через hook** без модификации модели — не нужно резать голову вручную, работает для любого
  HF-video-классификатора (общий `_HFVideoBackbone`).
- **Косто-нейтральные окна (v3.2):** окно = `clip_len×3`, окон втрое меньше → тот же бюджет 1536 кадров, но
  `mean_clips_per_track` вырос с 1.0 до 4.0 (плотнее траектория за те же деньги — осознанное решение).
- **fp16-опция** на CUDA, батч клипов (`batch_size`), очистка frames после NPZ (`cleanup_frames_after_npz`).
- **Чистая `build_v3_arrays` (numpy)** — юнит-тестируется без модели/GPU, golden-детерминизм тривиален.
- **L2 + stable-sort** — дешёвая нормализация формы токена под Encoder.

## 9. Слабые места

- **Реального стрима нет в storage-корпусе (главное).** Все 6 storage-видео — **stale v2**
  (`action_recognition_npz_v2`, `embeddings` = object-array per-track 256-d), а доминирующий ролик
  `-Q6fnPIybEI` вообще `empty` (no_person). Т.е. `clip_embeddings (C,2304)` доказан только на **тест-прогонах
  v3** (44/65 клипов), но **батч на v3 не пере-прогнан**. Это блокер для обучения Encoder на реальных данных.
- **Доковый дрейф:** `docs/SCHEMA.md` всё ещё описывает v2 (`embedding_normed_256d (num_clips,256)`,
  «embedding_dim always 256»). Контракт (v3.1, 2304) живёт в `COMPONENT_CONTRACTS.md` и коде, но не в SCHEMA.md.
- **Зависимость от person-детекций.** Нет людей → `empty`. Контент без людей (природа, графика, скринкаст)
  вообще не получает action-токена. Для не-people-видео это тихая дыра в фичах.
- **`clip_segment_id` слабый:** `num_action_segments = num_tracks` (1 сегмент/трек) — change-point внутри
  трека не срабатывает (эмбеддинги трека похожи). Аналитическая тонкость, на сам токен не влияет.
- **D нестабильна между backbone'ами** (2304 slowfast / 768 VideoMAE / 400 fallback) — Encoder обязан
  подстраиваться; смена backbone = смена размерности входа.
- **Дорого и диск-ёмко** (~10.7 GB/видео кадров, тяжёлый 3D-CNN) — для 200k обязательна политика очистки кадров.
- L-номера в `LOGIC_ERRORS_FOR_CLAUDE.md` по этой фиче нет (история багов — фрагментация треков, паддинг —
  закрыта итерациями v2→v3.2).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Пере-прогнать storage-корпус на v3** — сейчас реальных `clip_embeddings (C,2304)` нет ни в одном
   артефакте; без этого Encoder нельзя обучать на реальном action-сигнале. Блокер продакшн-масштаба.
2. **[выс.] Обновить `SCHEMA.md` до v3** — устранить доковый дрейф (v2 256-d → v3 динамический penultimate),
   иначе потребитель (Encoder) получит противоречивые контракты.
3. **[сред.] Зафиксировать один прод-backbone и D** (slowfast_r50 / 2304) — чтобы Encoder не подстраивал
   размерность входа; альтернативы (VideoMAEv2/Hiera) — только через baseline-ablation.
4. **[сред.] Fallback для не-people-контента** — сейчас `empty` при отсутствии людей; рассмотреть
   scene-level action-эмбеддинг (клип целиком без трека) как минимальный сигнал для природа/графика/скринкаст.
5. **[низ.] Улучшить `clip_segment_id`** (сегментация по логит-классу, а не по эмбеддингу) — если аналитике
   нужны тонкие границы действий; на model-токен не влияет.

## 11. Рекомендации по архитектуре / связям

- **Единая размерность-политика с Encoder:** зафиксировать чтение D из `meta.embedding_dim` на стороне
  Models (уже в контракте) и запретить хардкод 256/2304 — чтобы смена backbone не ломала обучение.
- **Shared sampling group** (core_object_detections ↔ action_recognition) — гарантировать на Segmenter
  совпадение `frame_indices`, иначе клип↔трек рассыпается (общий риск с core-провайдерами).
- **Reuse appearance-трекера:** `track_ids` из core_object_detections уже переиспользуются здесь — закрепить,
  не считать трекинг дважды.
- **Провенанс backbone в meta** (`model_id`, `embedding_mode`) обязателен, чтобы Encoder знал, на каких
  фичах учился (важно при миграции весов).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| v3.2 валидация (NPZ) 4:35 | 44 клипа, 11 треков | `(44,2304)` L2/finite, класс. ✅, golden идентичен | токен корректен, детерминирован |
| v3.2 валидация 8:00 | 65 клипов, 16 треков | `(·,2304)` L2, median 5 кл/трек | плотность траектории ок |
| Контроль 2:47 | no_person | `empty` (`no_person_detections`), golden ✅ | valid-empty корректен |
| Различимость треков | групповая сцена | intra≫inter, tubelet разные действия разным трекам | эмбеддинг разделяет персонажей |
| Классы Kinetics | сверка с роликами | clarinet/harmonica, yoga/marching правдоподобны | семантика осмысленна |
| `mean_clips_per_track` | v3.2 | **4.0/4.06** (главный критерий >1 закрыт) | траектория, а не одна метка |
| **Реальный storage-корпус (мой прогон)** | **6 видео, все v2** | `clip_embeddings` отсутствует; 5×v2-256d + 1 empty | **v3-стрим не материализован в батче** |

Вывод: как **логика/контракт** токен доказан (v3-гейты зелёные, golden=0, penultimate-фичи, плотность>1). Как
**прод-данные** — незрелый: реальные артефакты ещё v2, батч на v3 не пере-прогнан, SCHEMA.md отстаёт.

## 13. Интерпретируемость

- **Есть:** `clip_topk_action_ids/probs` + `class_names` (Kinetics) — человекочитаемая метка действия на клип;
  `video_action_hist`/`dominant_action_*` — «какие действия в видео»; `render.py` (dev).
- **Добавить:** словесная подпись «в видео преобладает: танец/готовка/спорт» из `dominant_action_ids`;
  timeline действий по `clip_times_s` с метками классов; сам 2304-d эмбеддинг не интерпретируем напрямую —
  для пользователя показывать классы, а эмбеддинг оставить внутренним model-входом.

## 14. Польза для моделей

**Высокая и уникальная — при условии материализации.** `clip_embeddings` — единственный вход, несущий
**пространственно-временную семантику действия** (движение во времени, а не покадрово), в идеальной для
Encoder форме: плоский `(C, D)` L2-стрим, отсортированный по времени, с осью и track-привязкой. Тип и динамика
действия правдоподобно сильно коррелируют с форматом/жанром контента → с удержанием/популярностью. penultimate
(2304-d) вместо логитов даёт богатую репрезентацию. **Но:** сейчас реального стрима в корпусе нет (всё v2/256d),
и при отсутствии людей токен пуст — обе вещи ограничивают фактическую пользу до пере-прогона.

## 15. Польза для аналитиков

Сам эмбеддинг аналитику не показывают (2304-d непонятен), но его аналитические спутники ценны:
`dominant_action_ids/probs` и `video_action_hist` → «какие действия в видео и какое доминирует»;
`clip_topk` по времени → лента действий; `num_tracks`/`mean_clips_per_track` → сколько людей и как долго в кадре.
Для сравнения видео полезен action-профиль (гистограмма Kinetics). Оговорки: не-people-видео дают пусто;
на реальных данных пока v2 (per-track, без стрима).

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Единственный источник семантики действия во времени; прямой seq-токен Encoder'а |
| 5. Выход (контракт) | 4 | Чистый L2 `(C,D)` стрим + ось + track-link; D динамична, но задокументирована в мете |
| 6. Фичи | 4 | penultimate вместо логитов, L2, track-траектория — сильно; D-нестабильность между backbone |
| 8. Оптимизации | 4 | Косто-нейтральные окна (cl/trk 1→4), hook без правки модели, fp16, golden-детерминизм |
| 9. Слабые места (инверсно) | 2 | Реального v3-стрима нет в корпусе, SCHEMA.md отстал, пусто без людей, дорого/диск |
| 12. Результаты тестов | 3 | v3-гейты зелёные+golden=0, но только на тест-прогонах; батч не пере-прогнан на v3 |
| 13. Интерпретируемость | 4 | Kinetics-классы/гистограмма понятны; эмбеддинг внутренний (и не должен быть виден) |
| 14. Польза для моделей | 4 | Уникальная ось «действие во времени», идеальная форма; ограничена материализацией/no-people |
| 15. Польза для аналитиков | 3 | Action-профиль полезен; сам эмбеддинг скрыт, на реальных данных пока v2 |

### Итоговые оценки

- **Польза для моделей: 4/5.** Даёт Encoder'у уникальный и семантически богатый action-токен (penultimate
  2304-d, L2, time-ordered, track-linked) — потенциально один из самых информативных визуальных входов для
  жанра/формата контента. Балл держит ниже 5 только незрелость данных: реального v3-стрима в корпусе нет
  (всё v2), и токен пуст на не-people-видео — обе вещи надо закрыть до обучения.
- **Польза для аналитиков: 3/5.** Напрямую эмбеддинг не показывается; ценность аналитику несут его спутники
  (доминирующие Kinetics-действия, action-гистограмма, число/плотность треков). Ограничивают пустота на
  не-people-контенте и то, что реальные артефакты пока v2 без стрима.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/action_recognition/utils/action_recognition_v3.py` (`build_v3_arrays`)
- `DataProcessor/VisualProcessor/modules/action_recognition/utils/backbones.py` (penultimate forward-hook, L2)
- `DataProcessor/VisualProcessor/modules/action_recognition/docs/SCHEMA.md` (⚠ описывает v2, доковый дрейф)
- `DataProcessor/docs/component_reports/action_recognition/REPORT_2026-07-05_FINAL.md`, `RUN_RESULT_v3*.md`, `ASSESSMENT_action_recognition.md`
- `DataProcessor/docs/COMPONENT_CONTRACTS.md` (action_recognition → Models/Encoder; core_object_detections → action_recognition)
- `DataProcessor/docs/COMPONENT_VALIDATION_CHECKLIST.md` (фич-леджер action_recognition v3, штамп 2026-07-05)
- `DataProcessor/docs/design/ACTION_RECOGNITION_V3.md` (дизайн окон/трекера/стрима)
- Реальные артефакты: 12× `storage/result_store/youtube/*/*/action_recognition/action_recognition_features.npz`
  (**6 уникальных видео, все schema v2**, 256-d per-track object-array, 1 empty)

## 18. Визуализации

![clip_embeddings overview](clip_embeddings_overview.png)

`clip_embeddings_overview.png`: слева — v3 **валидационные** прогоны (44/65 клипов, D=2304 penultimate, ~4 клипа/трек,
контроль empty) — токен доказан; справа — реальный **storage-корпус (всё v2, 256-d per-track, 1 empty)** —
подтверждает главный вывод: v3-стрим `clip_embeddings (C,2304)` пока не материализован в батче, нужен пере-прогон.
