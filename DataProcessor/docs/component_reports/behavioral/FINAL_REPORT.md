# FINAL REPORT — `behavioral`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `behavioral` (VisualProcessor `BaseModule`, Tier-2, CPU-only) |
| Версия кода | `2.0.1` |
| Схема NPZ | `behavioral_npz_v1` |
| Артефакт | `result_store/<platform>/<video>/<run>/behavioral/behavioral_features.npz` |
| Модель | **нет** — numpy-эвристики по pose+hands+face-mesh landmarks |
| Hard deps | Segmenter + `core_face_landmarks` (pose + hands + face-mesh тиры) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → behavioral ✅ (2026-07-12) |
| Отчёт валидации | [`REPORT_2026-07-12.md`](REPORT_2026-07-12.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `modules/behavioral/utils/behavior_analyzer.py` |

## 2. Резюме

`behavioral` — **анализатор языка тела и жестов**: из pose/hands/face-mesh landmarks считает **43 seq-признака**
(число рук, 12 классов жестов, движение/стабильность головы, моргание, self-touch, раскрытие рук `arm_openness`,
`pose_expansion`, `body_lean`, `shoulder_angle`, mouth/speech-прокси) + **33 video-агрегата** (engagement,
confidence, stress, gesture_rate/entropy). NaN построен **иерархически** (`landmarks_present ⊇ pose-тир ⊇
mouth/face-тир`) — by design. На реальном корпусе **ядро работает, где есть landmarks**, но компонент сильно
**ограничен контентом**: pose-тир 42–100% NaN (нужно тело в кадре), руки почти не видны (num_hands≈0.2 → 12
жестов в основном неактивны). Плюс: `body_lean_angle` в реальном батче **насыщается к константе 1.0 на части
видео** — пре-фикс баг (`*5.0`-множитель), фикс закоммичен, но **батч не пере-прогнан**. golden Δ=0.

## 3. Функционал

Стоит в Tier-2, после `core_face_landmarks` (расширенный: pose+hands+face-mesh). По кадрам:

- **Руки/жесты** — `num_hands`, `hands_visibility`, `hand_motion_energy`, 12 `gesture_prob_*` (thumbs_up/down,
  victory, ok, fist, open_palm, pointing, self_touch, love, rock, call_me, hands_on_hips), `self_touch_flag`.
- **Голова** — `head_position_x/y_norm`, `head_motion_energy`, `head_stability`, `blink_flag/rate`.
- **Поза (pose-тир)** — `arm_openness`, `pose_expansion`, `body_lean_angle`, `balance_offset`, `shoulder_angle(_velocity)`.
- **Рот/речь (face-тир)** — `mouth_*`, `speech_activity_proxy`.
- **Агрегаты** — engagement/confidence/stress (эвристические интерпретации), gesture_rate/entropy/switching.

**Зачем продукту:** язык тела — **сильный сигнал харизмы и вовлечённости**: активная жестикуляция, открытая
поза, зрительный контакт, стабильность. Это model-вход (харизма ↔ engagement) и понятная креатору аналитика
(«закрытая поза», «мало жестов», «высокий стресс»).

## 4. Вход

- **`core_face_landmarks`** (hard) — pose + hands + face-mesh landmarks (расширенный режим); нет person →
  `status=empty, skipped_due_to_person_mask_no_person`; нет лиц (только pose) → тоже возможен empty.
- **Segmenter** `frame_indices` + `union_timestamps_sec` — ось.
- Иерархия опор: face-landmarks → pose/hands → mouth; NaN по недоступным тирам.

## 5. Выход

- **Model-facing seq (N,F):** 43 `seq_*` признака + `seq_landmarks_present` (основная маска) + `seq_timestamp_norm`.
  Разрежённость by design: тиры pose/mouth NaN, когда нет соответствующих landmarks.
- **Аналитика:** `aggregated` (33 скаляра: engagement/confidence/stress/gesture-статистики).
- **Ось:** `frame_indices`, `times_s`.
- **NaN-иерархия:** `landmarks_present=True` → core-поля finite; pose-поля NaN все-вместе при нет-pose;
  mouth-поля NaN все-вместе при нет-face-mesh (0 частичных NaN внутри тира — структурно).

## 6. Фичи (важное/неочевидное)

- **Иерархическая NaN-маска (by design)** — `landmarks_present ⊇ pose ⊇ mouth`: элегантно, но означает **высокую
  разрежённость** на YouTube-контенте (talking-head без тела → pose-тир весь NaN 57%+).
- **12 gesture_probs — бимодальны**: 47% кадров = 0.0 (num_hands=0, руки не видны), 53% ≈softmax. На реальном
  корпусе num_hands≈0.2 → руки редки → большинство жестов неактивны почти всегда.
- **`body_lean_angle` — пре-фикс баг в батче**: `lean_raw*5.0 → clip(1.0)` → **константа 1.0** на видео с крупным
  lean. В storage std=0 (≡1.0) на 2/4 ok-видео. Фикс (убран `*5.0`) закоммичен, но батч не пере-прогнан → на
  части корпуса фича мёртвая.
- **engagement/confidence/stress-агрегаты** — эвристические «психологические» интерпретации разреженных
  landmark-сигналов; звучат ценно, но без валидации против реального восприятия — спекулятивны.
- **`self_touch`, `fidgeting_energy`** — прокси нервозности/неуверенности; интересны, но зависят от видимости рук.

## 7. Алгоритм / архитектура

- **Чистый CPU** (numpy): геометрические эвристики по landmarks (углы, расстояния, энергия движения), softmax
  жестов по эвристическим признакам кисти. Модели нет.
- **Сложность:** 3–4 c/видео (N=34…300); upstream (OD YOLO11l + landmarks pose+hands+face) доминирует (16–44 c).
- **Детерминизм:** golden max|Δ|=0.0 (чистый numpy).

## 8. Оптимизации

- **Reuse pose/hands/face landmarks** из core_face_landmarks — не детектит заново.
- **Иерархическая маска** — компактная семантика разрежённости (одна маска + isfinite по тирам).
- **Бимодальный gesture-softmax** — «нет рук»=0 явно, не шум.
- **Фикс body_lean** (убран saturating `*5.0`) — в коде (но не в батче).

## 9. Слабые места

- **Сильная ограниченность контентом (главное функциональное).** pose-тир 42–100% NaN, руки почти не видны
  (num_hands≈0.2) → на типичном YouTube-talking-head большая часть 43 признаков (жесты, поза) неактивна/NaN.
  Компонент раскрывается только на контенте с полным телом и активными руками (танцы/спорт/презентации).
- **`body_lean_angle` ≡1.0 в батче (пре-фикс баг)** — на 2/4 ok-видео константа (saturation `*5.0`); фикс не
  пере-прогнан. Ключевой дефект audit v4, устранён в коде, но не в данных.
- **engagement/confidence/stress — спекулятивные эвристики** — «психологические» агрегаты из разреженных
  геометрических сигналов, не валидированы против реального восприятия; риск ложной уверенности для аналитика.
- **Высокая NaN-доля агрегатов на degraded-роликах** (ItRcDFKFiSU 35% NaN — нет pose-тира) — C4 «PASS с оговоркой».
- **Дубль face-стека** — landmarks/emotion_face/micro_emotion/detalize_face/behavioral — пять компонентов на лице/теле.
- **12 жестов — узкий словарь** (в основном «руки-символы»), не покрывает естественную жестикуляцию речи.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Пере-прогнать батч с фиксом body_lean** — сейчас на части корпуса фича ≡1.0 (пре-фикс).
2. **[сред.] Валидировать/пометить engagement/confidence/stress** как эвристики — либо калибровать против
   разметки, либо честно подписать «оценочно», чтобы не вводить аналитика в заблуждение.
3. **[сред.] Явный content-flag** «есть ли полное тело/руки» — чтобы downstream знал, что behavioral-сигнал
   применим (а не путал NaN-разрежённость с «нейтральным поведением»).
4. **[низ.] Расширить жестовый словарь** до естественной жестикуляции речи (beat/iconic gestures), а не только руки-символы.
5. **[низ.] Слить face-стек** — общий landmark-слой для 5 компонентов.

## 11. Рекомендации по архитектуре / связям

- **Единый landmark-pipeline** (pose+hands+face) как source-of-truth для behavioral/detalize_face/emotion — снизить
  дублирование гейтинга и стоимость.
- **Reuse pose/hands закреплён** — правильно; убедиться в едином tracking_id между body-компонентами.
- **Контракт маски** `landmarks_present` + per-tier isfinite с Encoder — обязателен (высокая разрежённость).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate | 7 (набор B) | rc=0 VALID | контракт ок |
| U2 ось времени | 7 | fi↑, ts/timestamp_norm 0 NaN, ∈[0,1] | ось ок |
| U3 finite/health | 7 | 0 Inf; gesture-probs бимодальны (47%=0, 53%≈1) | health ок |
| U4 expected-empty | синт + 2 real core-empty | landmarks_present=0, seq NaN, aggregated валиден, rc=0 | пустой путь ок |
| U5 golden | ×2 | max\|Δ\|=0.0 | детерминизм |
| U6 разные длины | N=34…300 | все ок | масштаб ок |
| C1 NaN↔маска (core) | 0% NaN на present-кадрах (0/1344) | ядро строго по маске |
| C2 иерархия тиров | pose 57.22% / mouth 56.18% NaN, 0 частичных | структурно by design |
| C3 body_lean (после фикса) | std=0.239, 546/575 уникальных (на поде) | различимо ПОСЛЕ фикса |
| C4 aggregated | 33 поля, NaN 0–35% (структурны) | варьируется |
| **Реальный storage (мой прогон)** | 6 видео (4 ok, 2 empty) | body_lean ≡1.0 на 2/4; pose_NaN 0–100%; num_hands≈0.2 | body_lean пре-фикс, сильная разрежённость |

Вывод: **ядро (руки/голова/жесты) строго по маске и детерминировано**, но `body_lean` в батче местами ≡1.0
(пре-фикс), а весь компонент сильно ограничен контентом (разрежённость pose/hands).

## 13. Интерпретируемость

- **Потенциально высокая:** язык тела понятен — «открытая/закрытая поза», «активная жестикуляция», «нервозность».
- **Добавить:** словесная сводка «уверенная открытая поза / зажатая, мало жестов»; overlay skeleton на превью;
  но engagement/confidence/stress подписать как оценочные. `render.py` есть.

## 14. Польза для моделей

**Средняя, ограниченная контентом.** Язык тела/жесты — правдоподобный сигнал харизмы/вовлечённости, 43-мерный
dense-seq с маской — годная форма для Encoder. **Но** на типичном корпусе разрежённость огромна (pose/hands
часто NaN), `body_lean` местами ≡1.0 (пре-фикс), а «психологические» агрегаты спекулятивны. Полезен как
дополнительная фича на body-контенте (танцы/спорт/презентации), слаб на talking-heads.

## 15. Польза для аналитиков

**Средняя.** «Язык тела»: поза, жесты, нервозность, engagement — привлекательная и понятная аналитика. Но
надёжна только на контенте с телом/руками; engagement/confidence/stress — эвристики без валидации (риск ложной
уверенности); body_lean в текущих данных местами мёртв.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Богатый язык тела (43 seq + 33 agg); но раскрывается лишь на body-контенте |
| 5. Выход (контракт) | 4 | Dense-seq + иерархическая маска + агрегаты; высокая разрежённость |
| 6. Фичи | 3 | Ядро (руки/голова/жесты) ок; body_lean пре-фикс, жесты редки, агрегаты спекулятивны |
| 8. Оптимизации | 4 | Reuse landmarks, иерархич. маска, бимодальный softmax, golden=0 |
| 9. Слабые места (инверсно) | 2 | Контент-ограниченность, body_lean≡1.0, спекулятивные агрегаты, дубль стека |
| 12. Результаты тестов | 3 | Гейты PASS + строгая маска, но разрежённость и body_lean в реальных данных |
| 13. Интерпретируемость | 4 | Язык тела понятен (агрегаты — оценочно) |
| 14. Польза для моделей | 3 | Годная форма, но контент-ограничен и body_lean пре-фикс |
| 15. Польза для аналитиков | 3 | Понятно, но надёжно лишь на body-контенте; агрегаты спекулятивны |

### Итоговые оценки

- **Польза для моделей: 3/5.** Язык тела/жесты — правдоподобный сигнал харизмы в удобной dense-seq+маска форме.
  Но на типичном YouTube-корпусе огромная разрежённость (pose/hands часто NaN), `body_lean` местами ≡1.0
  (пре-фикс, не пере-прогнан), а «психологические» агрегаты спекулятивны — фактическая польза средняя, выше на body-контенте.
- **Польза для аналитиков: 3/5.** «Язык тела» (поза/жесты/нервозность/engagement) — понятная и привлекательная
  аналитика, но надёжна только когда в кадре есть тело/руки, agg-метрики engagement/confidence/stress —
  неоткалиброванные эвристики, а body_lean в текущих данных местами мёртв.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/behavioral/utils/behavior_analyzer.py`, `main.py`, `utils/validate_behavioral.py`
- `DataProcessor/VisualProcessor/modules/behavioral/docs/{SCHEMA.md, ALGORITHM_RECOMMENDATIONS.md}`
- `DataProcessor/docs/component_reports/behavioral/{REPORT_2026-07-12.md, CRITERIA.md}`
- Cross-ref: `core_face_landmarks` (pose+hands+face dep), face-стек (`emotion_face`/`micro_emotion`/`detalize_face`)
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/behavioral/behavioral_features.npz`
  (4 ok / 2 empty; **body_lean ≡1.0 на 2/4 (пре-фикс); pose_NaN 0–100%; num_hands≈0.2**)

## 18. Визуализации

![behavioral overview](behavioral_overview.png)

`behavioral_overview.png`: слева — `body_lean_angle` std по видео: **=0 (≡1.0) на 2/4 ok-видео** (пре-фикс
saturation-баг `*5.0`, не пере-прогнан); центр — pose-тир NaN% (0–100%, высокая разрежённость — нужно тело в
кадре); справа — mean num_hands ≈0.2 (руки почти не видны → 12 классов жестов в основном неактивны).
Подтверждает: ядро работает по маске, но компонент сильно ограничен контентом и body_lean местами мёртв в батче.
