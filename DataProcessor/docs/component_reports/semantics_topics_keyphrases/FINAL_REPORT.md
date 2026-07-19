# FINAL REPORT — `semantics_topics_keyphrases`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `semantics_topics_keyphrases` (SemanticTopicExtractor v2.1.0, TextProcessor) |
| Артефакт | tp_topics_* (**116 полей**, крупнейший text-блок) + `tp_topics_keyphrase_embeddings.npy` |
| Модель | **intfloat/multilingual-e5-large** (эмбеддинг RU/EN) + taxonomy-retrieval |
| Taxonomy DB | `dp_models/bundled_models/text/topics_v1/topics.jsonl` (**e2e_stub: 8 фейк-топиков**) |
| Hard deps | текст (ASR/title/description) + topics DB (e5-large инференс) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → semantics_topics_keyphrases ✅ (2026-07-17) |
| Отчёт валидации | [`REPORT_2026-07-17.md`](REPORT_2026-07-17.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `DataProcessor/TextProcessor/src/extractors/semantics_topics_keyphrases/` (main.py 756, topics_db.py 234) |

## 2. Резюме

`semantics_topics_keyphrases` — **самый амбициозный text-компонент**: тематическая классификация видео + ключевые
фразы + стиль-флаги. Механика **образцовая**: e5-large multilingual (RU/EN), retrieval по taxonomy (эмбеддинг текста
× эмбеддинги промптов топиков → softmax-распределение с entropy/perplexity), извлечение keyphrases (n-gram TF/позиция)
+ их эмбеддинги (1024-d), стиль-флаги (FAQ/CTA/instructional/audience), кэш prompt-эмбеддингов, privacy-режимы
none/hashed/raw. **Но на реальных данных — тройной обрыв сигнала:** (1) **taxonomy = e2e_stub** — 8 фейковых топиков
`topic_0..topic_7` (group `e2e_stub`, промпт = собственное имя), классификация бессмысленна; (2) **has_asr=0 везде** —
топики считаются по **mock-title (166–169 символов), НЕ по реальному транскрипту** (policy `asr_only`, но `doc.asr` не
прокинут — L8); (3) **title mock** — 2 уникальных значения на 21 видео. Итог: сильнейший дизайн, нулевой факт.

## 3. Функционал

Работает в TextProcessor. Пайплайн:

1. **Сбор текста** — по `transcript_source_policy` (asr_only/asr_then_legacy/legacy_only) + title + description → full_text.
2. **Topic-distribution** — e5-large эмбеддит full_text; matmul с эмбеддингами промптов таксона; max-по-топику →
   top-k; softmax(temperature=0.07) → prob; entropy_topk/norm/perplexity.
3. **Keyphrases** — детерминированный n-gram скоринг (TF × 1/(1+pos) × length-bonus), top-k фраз + их e5-эмбеддинги.
4. **Style-флаги** — счёт `?` (FAQ), словари RU для instructional/audience/CTA.
5. Пишет 116 tp_topics_* + артефакт keyphrase-эмбеддингов.

**Зачем продукту:** «про что видео» (тема) + «ключевые фразы» + «стиль подачи» — **основа семантического
позиционирования**: категоризация каталога, поиск похожего контента по теме, инсайт «в этой нише заходит».

## 4. Вход

- **Текст:** ASR-транскрипт (policy asr_only — **но `doc.asr` не прокинут → has_asr=0**), title/description (mock).
- **Taxonomy DB** (topics.jsonl через dp_models) — **сейчас e2e_stub 8 топиков**.
- **e5-large** — эмбеддинг-инференс.
- Нет текста → present=0.

## 5. Выход

- **116 tp_topics_* фич:** present/policy-флаги, text_chars/has_asr/title/desc, **8 topic-слотов** (id/score/prob),
  entropy/perplexity, **16 keyphrase-слотов** (present/hash01/len), keyphrases_count/dim, keyphrase_score top1/mean,
  4 style-флага, config/timings/digests.
- **Артефакт:** `tp_topics_keyphrase_embeddings.npy` (n_kp × 1024).
- **NaN-политика:** пустые слоты NaN; нет текста → present=0.

## 6. Фичи (важное/неочевидное)

- **topic_top1_id ∈ {1,2}, prob≈0.24 (главное)** — матч full_text против **stub-строк** `topic_0..topic_7`; топик
  «выигрывает» случайно (низкий prob 0.24, entropy 1.6 ≈ равномерно) → **сигнал пустой**.
- **has_asr=0 на всех 21** — реальный транскрипт **не участвует**; топики по mock-title 166–169 символов. Двойной
  промах: даже был бы реальный таксон — классифицируется не контент, а mock-заголовок.
- **2 уникальных значения на 21 видео** — (166ch→top1=2) ×5, (169ch→top1=1) ×16. Полностью mock-driven.
- **keyphrases_count=10, dim=1024** — keyphrase-эмбеддинги считаются (e5 работает), но из mock-title → мусорные фразы.
- **style-флаги все 0** — FAQ/CTA/instructional/audience не найдены (текст = крошечный mock-заголовок; на реальном
  транскрипте с «подпишитесь»/«вы» — сработали бы; сильная идея, простаивает).
- **entropy/perplexity top-k** — качественная мера «размытости» темы; на stub бесполезна, на реальном таксоне ценна.
- **privacy-режимы keyphrases (none/hashed/raw)** — продуманный контроль утечки текста (hashed01/len вместо строк).

## 7. Архитектура / алгоритм

- **e5-large multilingual** (RU/EN, 1024-d) — сильная SOTA-модель эмбеддингов, единственный text-компонент с реальным
  многоязычным ретривером.
- **Retrieval-topic:** эмбеддинг текста · эмбеддинги промптов топиков → max-по-топику → softmax. Гибко: топики
  задаются промптами в jsonl, не требует переобучения (zero-shot).
- **Keyphrases:** детерминированный n-gram (не YAKE — задокументировано как будущий upgrade).
- **Сложность:** e5-large инференс (CPU) — самый тяжёлый text-компонент; кэш prompt-эмбеддингов амортизирует таксон.
- **Детерминизм:** golden (заявлен PASS).

## 8. Оптимизации

- **Кэш prompt-эмбеддингов** (cache_dir, TTL, LRU-prune по MB) — таксон эмбеддится раз, не на каждое видео. Умно.
- **Zero-shot retrieval** — топики меняются правкой jsonl без переобучения.
- **Privacy-режимы** (none/hashed/raw) keyphrases — контроль утечки.
- **Slot-clamping** (topic 8, kp 16) + config-фрагмент — стабильная схема.
- **Ленивый import torch** (после disabled/empty путей) — исправленный баг.

## 9. Слабые места

- **Taxonomy = e2e_stub (корневой блокер)** — 8 фейк-топиков `topic_0..topic_7`, промпт = имя. Классификация
  бессмысленна. Нужен реальный таксон (жанры/ниши YouTube с RU/EN промптами).
- **has_asr=0 — ASR не прокинут (L8)** — топики по mock-title, не по транскрипту. Даже с реальным таксоном
  классифицируется заголовок, а не контент.
- **Title mock** — 2 варианта на 21 видео.
- **Keyphrases n-gram простой** (не YAKE/KeyBERT) — на реальном тексте фразы будут грубее.
- **Style-словари RU-only** — не-русский контент не покрыт.
- **Тяжесть e5-large на CPU** — самый дорогой text-компонент; на масштабе — узкое место.
- **116 полей — много config/slot-балласта** — реально информативных немного.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер] Реальный таксон** — заменить e2e_stub на продуктовый taxonomy (жанры/ниши, RU+EN промпты). Без
   этого топики фиктивны.
2. **[выс., блокер] Прокинуть ASR** (has_asr=0) — классифицировать транскрипт, а не mock-title (L8-паттерн).
3. **[сред.] Keyphrases → YAKE/KeyBERT** — качественнее n-gram (задокументировано автором как upgrade).
4. **[сред.] Мультиязычные style-словари** — не только RU.
5. **[низ.] Урезать config-балласт** 116 полей (в meta).

## 11. Рекомендации по архитектуре / связям

- **Единый e5-large ретривер** — шэрить эмбеддинг текста с `semantic_cluster_extractor` (тоже кластеризует текст) и
  transcript-эмбеддерами: один инференс на видео вместо нескольких.
- **Топики → каталог/поиск похожих** — при реальном таксоне ключ к «найти похожие по теме» (продуктовая фича).
- **ASR-first policy** — топики по контенту (транскрипт), заголовок как вторичный сигнал.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1–U6 (отчёт) | 27/28 | авто-штамп (1 архивный NPZ исключён) | схема/гейты ок |
| **Реальный storage (мой прогон)** | 21 видео с tp_topics | **taxonomy=stub; has_asr=0; top1∈{1,2} prob 0.24; 2 уник. значения; style=0** | тройной обрыв: stub-таксон + нет ASR + mock-title |

Вывод: **механизм образцов (e5-large retrieval, keyphrases, style, кэш), но факт нулевой** — фиктивный таксон,
классификация не контента (mock-title), 2 значения на 21 видео.

## 13. Интерпретируемость

- **Потенциально лучшая в text-секции** — «видео про [тему], ключевые фразы: …, стиль: FAQ/CTA» — предельно понятно
  и actionable (в отличие от опаковых эмбеддингов).
- **Сейчас нечего показывать** — топики фиктивны, фразы из mock-title. После реального таксона + ASR — сильнейший
  объяснимый семантический слой (тема + фразы + стиль на человеческом языке).

## 14. Польза для моделей

**Низкая (факт), высокая (потенциал).** Topic-distribution + keyphrase-эмбеддинги + style — богатейший
семантический сигнал (тема ↔ ниша ↔ спрос, стиль ↔ формат). Но факт: топики против stub-таксона по mock-title
(2 значения на 21 видео) → почти нулевой вклад, keyphrase-эмбеддинги из mock. При реальном таксоне + ASR — один из
сильнейших text-предикторов.

## 15. Польза для аналитиков

**Низкая (факт), высокая (потенциал).** «Про что видео, ключевые фразы, стиль подачи» — ценнейший инсайт
позиционирования («ваша ниша X, заходят фразы Y, формат FAQ»). Но на данных топики фиктивны (stub), фразы из
mock-title, style=0. После реального таксона + ASR — топовая объяснимая аналитика контента.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Тема+фразы+стиль — амбициозный полезный замысел |
| 5. Выход (контракт) | 3 | 116 фич, богато, но config/slot-балласт |
| 6. Фичи | 1 | Stub-таксон, has_asr=0, 2 значения на 21 видео |
| 8. Оптимизации | 4 | Кэш prompt-эмб, zero-shot, privacy, ленивый torch |
| 9. Слабые места (инверсно) | 1 | Тройной блокер (stub+ASR+mock), тяжесть e5 |
| 12. Результаты тестов | 2 | Гейты ok, но факт фиктивен |
| 13. Интерпретируемость | 3 | Замысел лучший в секции; сейчас нечего показать |
| 14. Польза для моделей | 2 | Сильнейший потенциал, факт≈0 |
| 15. Польза для аналитиков | 2 | Топ-потенциал позиционирования, факт≈0 |

### Итоговые оценки

- **Польза для моделей: 2/5.** Тематическая классификация + keyphrase-эмбеддинги + стиль — потенциально **сильнейший
  text-предиктор** (тема ↔ ниша ↔ спрос). Механика образцовая (e5-large multilingual, retrieval-softmax с entropy,
  кэш, zero-shot). Но на реальных данных **тройной обрыв**: taxonomy = e2e_stub (8 фейк-топиков), has_asr=0 (топики по
  mock-title, не транскрипту), title mock (2 значения на 21 видео). Фактический вклад ≈0; потенциал 4–5 после
  реального таксона + ASR-wiring.
- **Польза для аналитиков: 2/5.** «Про что видео, ключевые фразы, стиль» — ценнейший объяснимый инсайт
  позиционирования, потенциально топ text-аналитика. Но факт фиктивен (stub-таксон, mock-фразы, style=0). После
  реального таксона + ASR — одна из сильнейших; сейчас — нулевой сигнал.

## 17. Источники

- `DataProcessor/TextProcessor/src/extractors/semantics_topics_keyphrases/{main.py, topics_db.py, render.py, SCHEMA.md, docs/FEATURE_DESCRIPTION.md}`
- `dp_models/bundled_models/text/topics_v1/topics.jsonl` (**e2e_stub, 8 топиков**), `dp_models/spec_catalog/text/topics_taxonomy_v1.yaml`
- `DataProcessor/docs/component_reports/semantics_topics_keyphrases/{REPORT_2026-07-17.md, CRITERIA.md}`
- Cross-ref: `semantic_cluster_extractor` (родственный e5-кластеризатор — кандидат на шэринг инференса), `asr_extractor`/`transcript_aggregator` (транскрипт — не прокинут сюда, L8), `lexico_static_features` (родственная лексика)
- Реальные артефакты: 21× tp_topics_* (116) + keyphrase_embeddings.npy в storage
  (**taxonomy=stub topic_0..7; has_asr=0 везде; top1∈{1,2} prob 0.24; 2 уникальных значения; style-флаги=0; 7 видео без tp_topics — архивные прогоны**)

## 18. Визуализации

![semantics_topics overview](semantics_topics_overview.png)

`semantics_topics_overview.png`: слева — «воронка обрыва сигнала»: механизм ~100% → stub-таксон ~35% → без ASR ~15%
→ mock-title ~5% реальной информативности; справа — сводка: образцовый дизайн (e5-large multilingual, retrieval-
softmax+entropy, keyphrases+эмбеддинги, style-флаги, кэш, privacy-режимы) vs мёртвый факт (taxonomy=e2e_stub 8 фейк-
топиков, has_asr=0, top1 prob 0.24, 2 значения на 21 видео, style=0). Подтверждает: сильнейший text-семантический
замысел с тройным обрывом — нужен реальный таксон + ASR-wiring.
