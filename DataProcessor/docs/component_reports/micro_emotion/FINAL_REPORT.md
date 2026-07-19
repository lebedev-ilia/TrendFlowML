# FINAL REPORT — `micro_emotion`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `micro_emotion` (VisualProcessor `BaseModule`, Tier-2, Docker-OpenFace) |
| Версия кода | `2.0.2` |
| Схема NPZ | `micro_emotion_npz_v*` |
| Артефакт | `result_store/<platform>/<video>/<run>/micro_emotion/micro_emotion.npz` |
| Модель | **OpenFace** (FeatureExtraction: Action Units FACS, pose, gaze, landmarks) — Docker `--gpus all` |
| Hard dep | лица (через core_face_landmarks / детекции) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → micro_emotion ✅ (2026-07-13) |
| Отчёт валидации | [`REPORT_2026-07-13.md`](REPORT_2026-07-13.md), [`CRITERIA.md`](CRITERIA.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L11 (OpenFace docker infra gap) |
| Код | `modules/micro_emotion/utils/{micro_emotion_processor.py, openface_analyzer.py}` |

## 2. Резюме

`micro_emotion` — **анализатор микровыражений и лицевой мимики** на базе **OpenFace** (FACS Action Units, поза
головы, направление взгляда, раскрытие рта, моргание). Выдаёт **compact22** (22-мерный per-frame вектор AU/pose/
gaze для Encoder), детекцию микро-выражений (`microexpr_count`/события), 75 video-агрегатов (smile_ratio,
eye_contact, blink_rate, pose/gaze статистики) и PCA-скаляры. **На реальном корпусе компонент фактически мёртв:**
из-за бага чтения OpenFace-CSV (заголовки с ведущим пробелом) **21 из 22 колонок compact22 — константа**, все 75
агрегатов **= 0.0**, `microexpr_count=0` везде, `frame_features` усечён до (N,2). OpenFace физически отрабатывал
(processed 4–39 кадров), но его выход **никогда не парсился корректно**. Три бага (2 корневых + 1 валидатор)
найдены и исправлены 2026-07-13, но **только синтетически** — перегенерация 23 storage-NPZ требует docker-OpenFace
(L11 infra gap), который на поде недоступен. Итог: логика починена, реальные данные — мусор-константа.

## 3. Функционал

Стоит в Tier-2, гейтится лицами. Пайплайн:

1. OpenFace (Docker) прогоняет FeatureExtraction по кадрам-с-лицами → CSV с AU (интенсивности `_r` + наличие
   `_c`), позой головы (pose_Rx/Ry/Rz/Tz), взглядом (gaze_angle), landmarks, success-флагом.
2. `openface_analyzer.py` парсит CSV → per-frame величины.
3. `micro_emotion_processor.py` собирает **compact22** (AU-дельты, поза, взгляд, рот, моргание, PCA), детектит
   **микро-выражения** (спайки AU06/AU12/etc → smile/surprise/frown/disgust события), считает 75 агрегатов.

**Зачем продукту:** микро-мимика — **тонкий сигнал эмоц. вовлечённости и харизмы** спикера: улыбки, зрительный
контакт, живость лица, честность/напряжение (микровыражения). Это дополняет `emotion_face` (грубые эмоции)
детальным FACS-уровнем: model-сигнал (харизма ↔ engagement) + аналитика («мало зрительного контакта», «искренняя улыбка»).

## 4. Вход

- **Лица** (core_face_landmarks / детекции) — нет лиц → `status=empty, no_faces_in_video`.
- **Кадры** для OpenFace.
- **OpenFace Docker-образ** (`openface/openface:latest`; L11 — pull denied на поде, нужен `algebr/openface` алиас).
- **`union_timestamps_sec`** + Segmenter `frame_indices` — ось.

## 5. Выход

- **Model-facing:** `compact22 (N,22)` float32 + `compact22_feature_names` (AU-дельты/pose/gaze/mouth/blink/PCA).
- **Микро-выражения:** `microexpr_features`, `event_type_id`/`event_times_s`/`event_strength`, `microexpr_count`.
- **Video-агрегаты:** `feature_names`/`feature_values` (75): smile_ratio, eye_contact_ratio, blink_rate_per_min,
  pose/gaze mean/std/min/max, au_quality, occlusion/lighting-флаги.
- **`frame_features (N,2)`** (time_norm, face_present — усечён багом), `face_present_any`, `summary`.
- **NaN-политика:** вне-лицевые кадры/видео без лиц → all-NaN compact22.

## 6. Фичи (важное/неочевидное)

- **compact22 — FACS Action Unit вектор** (задумано): AU-интенсивности/дельты + поза + взгляд + рот — богатое
  описание мимики для Encoder. **На реальных данных мёртв: 21/22 колонки константа** (только time_norm жив).
- **Микро-выражения** — детекция коротких AU-спайков (smile/surprise/frown/disgust). **microexpr_count=0 везде** в батче.
- **eye_contact / blink / pose stability** — потенциально ценные «поведенческие» метрики спикера. **Все = 0.0** в батче.
- **`frames_processed_openface`** (4–39) и **`au_quality_overall`** (0.0–0.28) — **единственные не-нулевые**
  величины: доказывают, что OpenFace ЗАПУСКАЛСЯ и обрабатывал кадры, но его AU/pose/gaze-выход не был прочитан.
- **compact22 label-баг** (bug #2): порядок append не совпадал с `COMPACT22_FEATURE_NAMES` → Encoder получал
  **перепутанные метки AU** (столбец «gaze» содержал pose_Rz). Исправлено, но батч не перегенерирован.

## 7. Алгоритм / архитектура

- **Модель:** OpenFace (внешний C++ инструмент, FACS AU + pose/gaze), запуск через **Docker** `--gpus all`;
  постобработка — numpy/pandas/scipy/sklearn (PCA).
- **Сложность:** OpenFace на кадры-с-лицами (Docker overhead); постобработка лёгкая.
- **Детерминизм:** постобработка golden max|Δ|=0.0; реальный docker-OpenFace детерминизм/версия образа не проверены.
- **Хрупкость:** docker-in-docker недоступен на поде → реальный прогон не воспроизводился при валидации.

## 8. Оптимизации

- **compact22** — компактный фикс-размерный вектор для Encoder (когда жив).
- **PCA-сжатие** AU/landmarks в скаляры (var_explained).
- **Гейтинг по лицам** — OpenFace только на кадрах-с-лицами.
- **Синтетическая валидация постобработки** — позволила найти/починить логику без docker (умный обход инфры).

## 9. Слабые места

- **Весь выход мёртв в реальном батче (главное).** 21/22 compact22-колонок константа, **все 75 агрегатов = 0.0**,
  microexpr=0 на всех видео. OpenFace отрабатывал (processed 4–39 кадров), но leading-space CSV-баг сделал
  каждое AU/pose/gaze значение дефолтным. Компонент **не даёт ни одного usable-сигнала** в storage.
- **Фиксы только синтетические** — 3 бага (leading-space, mislabeled compact22, empty-shape) исправлены и
  проверены на синтетике, но **23 storage-NPZ не перегенерированы** (нужен docker-OpenFace) → в данных всё ещё
  баг. Хуже: до фикса #2 Encoder получал бы **перепутанные метки AU**.
- **Docker-OpenFace — тяжёлая хрупкая инфра (L11).** `openface/openface:latest` pull denied; нужен алиас
  `algebr/openface`, `--entrypoint bash`, толерантность к segfault. Docker-in-docker недоступен на поде.
- **3 placeholder-колонки** (`face_asymmetry_score`, `microexpr_recent_count`, `au_quality_flag`) не заполнены — отдельный PR.
- **frame_features усечён до (N,2)** — богатый per-frame выход схлопнут багом; расширение — отдельный PR.
- **Дубль с emotion_face** — обе про эмоции лица; два хрупких инфра-стека (EmoNet + OpenFace) вместо одного.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер] Перегенерировать storage через docker-OpenFace с фиксами** — сейчас весь выход = константа/0;
   без перегенерации компонент бесполезен в данных. Нужна рабочая OpenFace-среда (L11 fix: `setup_e2e_openface.sh`).
2. **[выс.] Подтвердить git-коммит 3 фиксов** (openface_analyzer strip, compact22 order, validator empty).
3. **[выс.] Стабилизировать OpenFace-инфру** — зафиксировать образ/версию, entrypoint, toleration segfault;
   иначе прод-200k не воспроизводим.
4. **[сред.] Заполнить 3 placeholder-колонки + расширить frame_features** (отдельный PR из отчёта).
5. **[сред.] Рассмотреть слияние с emotion_face** — единый face-emotion слой вместо двух хрупких стеков.

## 11. Рекомендации по архитектуре / связям

- **OpenFace как сервис/пре-провайдер** — если и micro_emotion, и другие захотят AU, поднять OpenFace один раз
  (как Triton), а не docker-per-video.
- **Единый face-pipeline** (landmarks → emotion_face → micro_emotion) с общим source-of-truth лиц — уменьшить
  дублирование гейтинга и инфра-точек отказа.
- **compact22 label-контракт** — зафиксировать строгую сверку порядка колонок в CI (bug #2 был именно про это).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate | 4 ok + empty | rc=0 | схема валидна (но контент мёртв) |
| U2 ось времени | 23 NPZ | fi↑, ts монотонны | ось ок |
| U3 health/shape | compact22 (N,22) f32, 0 Inf | форма ок |
| U4 expected-empty | нет лиц | status=empty, all-NaN, rc=0 | пустой путь ок |
| U5 golden (постобработка) | ×2 | max\|Δ\|=0.0 | детерминизм постобработки |
| U6 разные длины | N=12/43/65/119 | rc=0 | масштаб ок |
| C1–C3 (после фикса, СИНТЕТИКА) | N=90 | 17/22 non-const, PCA finite, microexpr=8 | логика чинена **синтетически** |
| **Реальный storage (мой прогон)** | 6 видео (4 ok, 2 empty) | **compact22 1/22 non-const; все 75 агрегатов=0.0; microexpr=0; frame_features (N,2)** | **выход полностью мёртв (пре-фикс баг)** |
| — доказательство запуска OpenFace | 4 ok | frames_processed=4–39, au_quality 0–0.28 | OpenFace работал, но CSV не читался |

Вывод: **схема/детерминизм валидны, логика починена (синтетически), но реальный выход — мусор-константа**;
без перегенерации через docker-OpenFace компонент бесполезен.

## 13. Интерпретируемость

- **Потенциально отличная** (когда жив): зрительный контакт, улыбки, микро-выражения, поза головы — понятнейшие
  креатору «поведенческие» инсайты спикера. `render.py` есть.
- **Сейчас нечего показывать** — всё 0. После перегенерации: «мало зрительного контакта», «искренние улыбки N%»,
  таймлайн микро-выражений.

## 14. Польза для моделей

**Потенциально заметная, фактически нулевая.** FACS Action Units / микро-мимика — тонкий сигнал харизмы и
эмоц. вовлечённости спикера, богатый (compact22) вход для Encoder. **Но** в реальных данных compact22 — константа
(21/22), все агрегаты 0, а до фикса #2 метки были перепутаны (риск тихого отравления). Фактическая польза = 0 до
перегенерации; потенциал — 3–4 после docker-OpenFace-прогона.

## 15. Польза для аналитиков

**Потенциально высокая, фактически нулевая.** «Зрительный контакт, улыбки, живость лица, микро-выражения» —
ценнейшая поведенческая аналитика для говорящих-голов (влоги/интервью). Но на реальных данных всё = 0. До
перегенерации аналитик не получает ничего.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Ценная FACS-ниша (микро-мимика), но целиком на хрупком docker-OpenFace |
| 5. Выход (контракт) | 3 | Богатый compact22+75 агрегатов+события; в данных мёртв/усечён |
| 6. Фичи | 2 | AU/gaze/микро-выражения задуманы сильно, но в батче константа/0 |
| 8. Оптимизации | 3 | compact22/PCA/синт-валидация умны; docker-инфра хрупкая |
| 9. Слабые места (инверсно) | 1 | Весь выход мёртв в данных, фиксы лишь синтетические, docker L11, label-баг |
| 12. Результаты тестов | 2 | Схема/детерминизм ок, но реальный контент мусор-константа |
| 13. Интерпретируемость | 4 | Поведение спикера предельно понятно (когда жив) |
| 14. Польза для моделей | 2 | Потенциал есть, факт=0 (константа + был label-shuffle) |
| 15. Польза для аналитиков | 2 | Потенциал высок, факт=0 (всё 0 в батче) |

### Итоговые оценки

- **Польза для моделей: 2/5.** FACS Action Units / микро-мимика — тонкий сигнал харизмы, богатый compact22-вход
  для Encoder. Но в реальном батче compact22 — константа (21/22), агрегаты=0, а пре-фикс #2 давал перепутанные
  метки. Фактическая польза нулевая до перегенерации через docker-OpenFace; логика починена лишь синтетически.
- **Польза для аналитиков: 2/5.** «Зрительный контакт, улыбки, микро-выражения, поза» — потенциально одна из
  самых ценных поведенческих аналитик для говорящих-голов, но на всём реальном корпусе выход = 0. Балл отражает
  факт (данные мертвы), а не потенциал (3–4 после рабочего OpenFace-прогона).

## 17. Источники

- `DataProcessor/VisualProcessor/modules/micro_emotion/utils/{micro_emotion_processor.py, openface_analyzer.py, validate_micro_emotion.py, render.py}`
- `DataProcessor/VisualProcessor/modules/micro_emotion/{main.py, docs/SCHEMA.md}`
- `DataProcessor/docs/component_reports/micro_emotion/{REPORT_2026-07-13.md, CRITERIA.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L11 OpenFace docker + EmoNet vendor infra gap)
- Cross-ref: `emotion_face` (родственная эмоц.-ветка), `core_face_landmarks` (лица)
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/micro_emotion/micro_emotion.npz`
  (4 ok / 2 empty; **compact22 1/22 non-const, все 75 агрегатов=0.0, microexpr=0 — пре-фикс баг, OpenFace processed 4–39 кадров**)

## 18. Визуализации

![micro_emotion overview](micro_emotion_overview.png)

`micro_emotion_overview.png`: слева — число не-константных колонок compact22 по видео: **1/22** в реальном
батче (только time_norm) против 17/22 после фикса на синтетике (зелёный пунктир); справа — сводка 3 багов
(leading-space CSV → 20/22 константа, mislabeled compact22, empty-shape) и inфра-блокера (docker-OpenFace pull
denied, L11). Подтверждает: логика починена синтетически, но реальный выход мёртв и требует перегенерации через
рабочий OpenFace.
