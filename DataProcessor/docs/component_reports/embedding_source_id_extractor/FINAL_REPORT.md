# FINAL REPORT — `embedding_source_id_extractor`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `embedding_source_id_extractor` (EmbeddingSourceIdExtractor v1.3.0, TextProcessor) |
| Артефакт | tp_embid_* (**13 скаляров**) + `embedding_source_id` (vector_id/meta) в render_context |
| Тип | **инфраструктурный** (плоскость retrieval/индексации), не фича-экстрактор |
| Модель | **нет** — детерминированный выбор источника + sha256-хэш |
| Hard dep | эмбеддинги (transcript_agg / title / description) от эмбеддеров/агрегаторов |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → embedding_source_id_extractor ✅ (2026-07-17) |
| Отчёт валидации | [`REPORT_2026-07-17.md`](REPORT_2026-07-17.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `DataProcessor/TextProcessor/src/extractors/embedding_source_id_extractor/main.py` (365) |

## 2. Резюме

`embedding_source_id_extractor` — **инфраструктурный компонент retrieval-плоскости**, а не фича-экстрактор. Выбирает
**«первичный» эмбеддинг** видео детерминированно по политике (`transcript_first`: транскрипт если есть, иначе title/
description) и считает **portable `vector_id`** = sha256[:24] по C-order little-endian байтам float32 (content-
addressable, без привязки к пути) → ID для vector store (FAISS «найти похожие видео», дедупликация). **Единственный
ЗДОРОВЫЙ компонент в текущей derived-серии:** `present=1` у **всех 22/22**, все флаги чисты. Корректно: у 5 видео с
реальным транскриптом primary=transcript → **5 уникальных ID**; у 17 без транскрипта fallback на title → **один общий
ID** `f20b4d16…` (mock-title → идентичный эмбеддинг → идентичный хэш — правильное поведение, вскрывает mock). Низкие
итоговые баллы — **из-за инфра-роли (не предиктор), НЕ поломки**.

## 3. Функционал

Работает после эмбеддеров/агрегаторов:

1. **Выбор primary** по `primary_source_policy` (transcript_first/title_first/description_first/title_only/
   transcript_only) — с fallback-цепочкой.
2. **Загрузка** выбранного эмбеддинга (safe-join relpath, finite-check).
3. **vector_id** = sha256[:24] по float32-байтам (детерминированный, портируемый, privacy-safe — без абсолютных путей).
4. Пишет 13 tp_embid_* (present, policy one-hot, primary-source one-hot, флаги) + `embedding_source_id` (vector_id,
   vector_store_uri, model-meta, primary_source, relpath).

**Зачем продукту:** стабильный content-addressable ID — **фундамент retrieval-плоскости**: индексация видео в FAISS,
«найти похожие», дедупликация одинакового контента, ссылка на эмбеддинг без пути. Это plumbing под фичу «сравнение
видео», а не сам инсайт.

## 4. Вход

- **Эмбеддинги** из `doc.tp_artifacts`: transcript_agg_mean (реальный, 5/22), title/description (mock).
- Нет эмбеддинга → strict-raise или present=0 (по флагу).

## 5. Выход

- **13 tp_embid_* скаляров:** present, strict_missing_primary_enabled, 5 policy one-hot, 3 primary-source one-hot
  (transcript/title/description), 3 флага (unsafe_relpath/primary_embed_missing/nan_inf).
- **`embedding_source_id` (meta):** vector_id, vector_store_uri, model_name/version/weights_digest, embedding_relpath,
  primary_source.
- **NaN-политика:** нет эмбеддинга → strict-raise (strict_missing_primary=True) либо present=0 + error-код.

## 6. Фичи (важное/неочевидное)

- **present=1 у всех 22 (главное)** — компонент отработал везде; все диагност-флаги 0. Здоровый.
- **primary_is_transcript=5 / primary_is_title=17** — политика transcript_first корректно: транскрипт где есть, иначе
  title. Отражает, у каких видео реально был ASR.
- **vector_id: 5 уникальных (transcript) + 1 общий на 17 (title mock)** — content-addressable хэш работает верно;
  17 mock-заголовков коллапсируют в один ID (симптом mock, не баг компонента; но «найти похожие» по ним слепнет).
- **model_version=unknown** — meta не заполнена (weights_digest unknown); ID не привязан к версии модели.
- **13 скаляров — метаданные, не предиктор** — говорят «какой источник первичен / есть ли транскрипт», слабый
  косвенный сигнал для модели.

## 7. Архитектура / алгоритм

- **Детерминированный выбор + sha256-хэш** (numpy + hashlib). Своей модели нет.
- **Portable ID:** явный byteswap к little-endian перед хэшем → одинаковый ID на разных архитектурах.
- **Безопасность:** safe-join relpath (path-traversal guard), privacy-safe (без абсолютных путей в результате).
- **Сложность:** тривиальная. **Детерминизм:** полный (по построению).

## 8. Оптимизации

- **Content-addressable ID** — дедупликация «из коробки» (одинаковый контент → одинаковый ID).
- **Portable byteswap** — кросс-архитектурная стабильность ID.
- **Policy-fallback** — устойчивость к отсутствию источника (transcript→title→description).
- **Privacy-safe** — no path leak; vector_id вместо пути.
- **strict/soft режимы** (strict_missing_primary) — гибкость для пайплайна.

## 9. Слабые места

- **Не предиктор** — 13 скаляров суть метаданные; для модели почти нет прямого сигнала (это его роль, не дефект).
- **model_version=unknown / weights_digest unknown** — ID не привязан к версии эмбеддера; при смене модели два
  разных эмбеддинга-по-версии могут не различаться по meta.
- **vector_store_uri='faiss://semantic_titles_v1'** — хардкод-дефолт, «titles» вводит в заблуждение (primary=
  transcript у 5). Наименование не отражает мультиисточник.
- **17 видео → один vector_id** (mock-title) — «найти похожие» по ним бесполезно (инходит от mock, но влияет на UX).
- **Хэш по значениям без модели-версии** — теоретическая коллизия между версиями эмбеддера (низкий риск).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[сред.] Заполнить model_version/weights_digest** в esid — привязать ID к версии эмбеддера (воспроизводимость
   retrieval при апдейте модели).
2. **[сред.] Включить версию модели в vector_id** (или хранить рядом) — избежать кросс-версионных коллизий.
3. **[низ.] Переименовать/динамить vector_store_uri** — отражать реальный источник (не только «titles»).
4. **[низ.] Реальные title/desc** — уберёт коллапс 17 видео в один ID (наследованный mock).

## 11. Рекомендации по архитектуре / связям

- **embedding_source_id → FAISS-индекс → «найти похожие»/сравнение видео** — ключевая связь с продуктовой фичой
  сравнения (backend/аналитика).
- **Единый vector_id для мультимодального retrieval** — согласовать с визуальными/аудио-ID (общий namespace дедупа).
- **Версионирование индекса** — при смене эмбеддера пере-индексировать по новым ID (нужен model_version binding).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1–U6 (отчёт) | 28 | авто-штамп | схема/гейты ок |
| **Реальный storage (мой прогон)** | 22 видео | **present=1 у всех 22, флаги чисты; 5 уникальных vector_id (transcript) + 1 общий на 17 (title mock)** | здоров и корректен; content-addressable ID работает; mock коллапсирует title-видео |

Вывод: **корректный, здоровый инфра-компонент** — работает 22/22, детерминированный portable ID; ограничен инфра-
ролью (не предиктор), model_version unknown и mock-коллапсом title-видео.

## 13. Интерпретируемость

- **vector_id опаков** (хэш) — не для человека; но `primary_source` понятен («ID построен по транскрипту/заголовку»).
- **Добавить:** в UI сравнения — «похоже на видео X, Y (по эмбеддингу транскрипта)»; сам ID остаётся служебным.

## 14. Польза для моделей

**Низкая (по роли).** Это инфра-компонент, не фича-экстрактор: 13 скаляров — метаданные (какой источник первичен,
есть ли транскрипт). Прямого предиктивного сигнала почти нет (кроме слабого «наличие транскрипта» через
primary_is_transcript). Модели он служит косвенно — как backbone retrieval, не как признак. Балл отражает узкую роль,
не поломку.

## 15. Польза для аналитиков

**Умеренная (инфра-бэкбон).** `vector_id` — фундамент фичи «найти похожие видео»/дедупликации/сравнения (FAISS).
Компонент **здоров и корректен** (22/22, чистые флаги, детерминированный ID). Ограничивают: mock-коллапс 17 title-
видео в один ID (нет реального «похожие» по ним), model_version unbound, и то, что сам ID — служебный, а не прямой
инсайт. При реальных данных — надёжная основа сравнения контента.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Полностью выполняет инфра-задачу (выбор+ID), 22/22 |
| 5. Выход (контракт) | 4 | Чистые 13 скаляров + esid-meta; privacy-safe |
| 6. Фичи | 3 | Source-флаги — осмысленные метаданные, не предиктор |
| 8. Оптимизации | 4 | Content-addressable, portable byteswap, fallback, privacy |
| 9. Слабые места (инверсно) | 3 | Не предиктор, model_version unknown, uri naming, mock-коллапс |
| 12. Результаты тестов | 4 | 22/22 present, флаги чисты, ID корректен |
| 13. Интерпретируемость | 3 | primary_source понятен; vector_id служебный |
| 14. Польза для моделей | 2 | Инфра, не предиктор; слабый косвенный сигнал |
| 15. Польза для аналитиков | 3 | Backbone retrieval/сравнения; здоров, но plumbing + mock |

### Итоговые оценки

- **Польза для моделей: 2/5.** Инфраструктурный компонент, а не фича-экстрактор: 13 скаляров — метаданные (первичный
  источник, наличие транскрипта), прямого предиктивного сигнала почти нет. Модели служит косвенно (backbone для
  retrieval), не как признак. Балл отражает **узкую инфра-роль, не поломку** — сам компонент здоров (22/22, чистый).
- **Польза для аналитиков: 3/5.** `vector_id` — надёжный фундамент фичи «найти похожие видео»/дедупликации/сравнения
  контента (FAISS). Компонент корректен и детерминирован. Ограничивают: mock-коллапс 17 title-видео в один ID,
  model_version unbound, служебная (не прямая) природа ID. При реальных данных — прочная основа сравнительной аналитики.

## 17. Источники

- `DataProcessor/TextProcessor/src/extractors/embedding_source_id_extractor/{main.py, SCHEMA.md, docs/FEATURE_DESCRIPTION.md}`
- `DataProcessor/docs/component_reports/embedding_source_id_extractor/{REPORT_2026-07-17.md, CRITERIA.md}`
- Cross-ref: `transcript_aggregator` (transcript_agg — primary у 5), title/description embedders (mock — общий ID у 17), FAISS/vector store (потребитель ID; backend «сравнение видео»)
- Реальные артефакты: 22× tp_embid_* (13) + `embedding_source_id` в render_context.json
  (**present=1 у всех 22; флаги чисты; primary_is_transcript=5 (уникальные vector_id) / primary_is_title=17 (общий ID f20b4d16… — mock); model_version=unknown**)

## 18. Визуализации

![embedding_source_id overview](embedding_source_id_overview.png)

`embedding_source_id_overview.png`: слева — разбивка primary-источника: 5 transcript (реальный → 5 уникальных
vector_id) / 17 title-fallback (mock → 1 общий ID), present=1 у всех 22; справа — сводка: инфра-компонент retrieval-
плоскости (portable content-addressable vector_id для FAISS «найти похожие»/дедуп), здоров и корректен (чистые флаги),
но не предиктор, model_version unknown, mock коллапсирует title-видео. Подтверждает: единственный здоровый derived-
компонент подряд; низкие баллы — из-за инфра-роли, не поломки.
