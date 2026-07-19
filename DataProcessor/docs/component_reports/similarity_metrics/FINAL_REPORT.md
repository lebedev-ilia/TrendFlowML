# FINAL REPORT — `similarity_metrics`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `similarity_metrics` (VisualProcessor `BaseModule`, Tier-3, CPU-only) |
| Версия кода | `2.0.2` |
| Схема NPZ | `similarity_metrics_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/similarity_metrics/results.npz` |
| Модель | **нет** — numpy cosine-similarity над эмбеддингами |
| Hard dep | `core_clip` (frame-эмбеддинги) |
| Soft deps | audio_clap, text, pacing, quality, emotion (для мультимодальной reference-части) |
| Reference pack | `dp_models/bundled_models/similarity/reference_sets/` — **ПУСТ** |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → similarity_metrics ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L4 (status=ok при ~60% NaN) |
| Код | `modules/similarity_metrics/utils/similarity_metrics.py` |

## 2. Резюме

`similarity_metrics` — движок **сравнения видео**: считает (а) **внутривидео-когерентность** (`centroid_sim` —
насколько кадры похожи на центроид видео; `temporal_sim_next` — покадровая близость) и (б) **мультимодальное
сравнение с референс-набором** (`reference_similarity_*` по 7 модальностям: clip/audio_clap/text/pacing/quality/
emotion/overall) + **uniqueness** (насколько видео уникально относительно корпуса). На реальном корпусе **живёт
только когерентная половина (15/39 фич)**, а **вся reference/uniqueness-часть мертва (24/39 = 62% NaN)** —
`reference_sets/` пуст → `reference_present=False` на всех видео. Именно reference/uniqueness — **ключевая
продуктовая фича** («сравнение видео через cosine similarity», CLAUDE.md), и она вся NaN. Когерентная часть
работает, но слабо различима (centroid_sim 0.83–0.99, CV 4.8% — все видео самоподобны). golden Δ=0; есть мёртвый
код (8 legacy-методов) и L4 (status=ok при 62% NaN).

## 3. Функционал

Tier-3, после core_clip (+ опц. мультимодальные фичи). Две группы метрик:

1. **Когерентность (alive):** `centroid_sim` (mean/std/p10/p90) — косинус кадра к среднему эмбеддингу видео
   (насколько видео «однородно»); `temporal_sim_next` (mean/std) — близость соседних кадров (плавность/
   повторяемость); `n_frames` + флаги наличия модальностей.
2. **Reference/uniqueness (dead):** `reference_similarity_*` по 7 модальностям (mean_topn/max/p10) — насколько
   видео похоже на референс-набор (похожие видео); `uniqueness_score/clip/overall` — обратное (насколько уникально).

**Зачем продукту:** сравнение видео — **центральная фича** для аналитиков (найти похожие ролики, оценить
уникальность/новизну контента, позиционирование в нише) и для рекомендаций. Когерентность — прокси
«однородность/повторяемость» видео. Uniqueness — «насколько ваш контент выделяется».

## 4. Вход

- **`core_clip`** (hard, no-fallback) — `frame_embeddings` + `frame_indices`; нет → `FileNotFoundError`.
- **Мультимодальные (soft):** audio_clap/text/pacing/quality/emotion — для reference-части; флаги `modality_*_present`.
- **Reference pack** (`reference_sets/`) — эмбеддинги референс-видео; **пуст** → reference-часть NaN.
- **`union_timestamps_sec`** + `frame_indices` — ось.

## 5. Выход

- **Alive (15):** `centroid_sim_mean/std/p10/p90`, `temporal_sim_mean/std`, `n_frames`, `reference_present_float`,
  `modality_{clip,audio_clap,text,pacing,quality,emotion,overall}_present`.
- **Dead (24, NaN):** `reference_similarity_{clip,audio_clap,text,pacing,quality,emotion,overall}_{mean_topn,max,p10}`
  (21) + `uniqueness_{score,clip,overall}` (3).
- **Массивы:** `centroid_sims (N)`, `temporal_sim_next (N-1)`, ось `frame_indices`/`times_s`.
- **F=39 стабильно** при любом N.

## 6. Фичи (важное/неочевидное)

- **`centroid_sim`** — самоподобие видео: высокое (0.98) = однородное/статичное (talking-head, короткое); ниже
  (0.83) = разнообразное (много сцен). На данных 0.83–0.99, но **CV всего 4.8%** — слабо различает (все видео
  довольно самоподобны в CLIP-пространстве).
- **`temporal_sim_next`** — покадровая гладкость (0.86–0.996): высокое = мало смены/статика.
- **Мультимодальный reference-дизайн (7 модальностей)** — сравнение не только по картинке, но и по звуку/тексту/
  темпу/качеству/эмоциям: сильная идея для «похожие видео». **Вся мертва** (нет reference pack).
- **`uniqueness_*`** — «насколько уникально» (обратное reference-сходству); тоже NaN. Пересекается с отдельным
  компонентом `uniqueness` (дублирование концепции — см. §11).
- **`modality_*_present` флаги** — какие модальности были доступны; полезная мета для трактовки reference-NaN.

## 7. Архитектура / алгоритм

- **Чистый CPU** (numpy cosine); reference-часть = kNN эмбеддинга видео против reference-набора.
- **Сложность:** O(N) для когерентности + O(N×|ref|) для reference (когда есть). Дёшев.
- **Детерминизм:** golden max|Δ|=0.0.
- **Мёртвый код:** 8 legacy-методов (`compute_style/text/audio/emotion_similarity`, `extract_all` и т.д.)
  используют незадекларированные scipy-функции + отсутствующий `self.similarity_weights` — не в production.

## 8. Оптимизации

- **Reuse core_clip-эмбеддингов** (+ мультимодальных) — не считает заново.
- **Стабильный F=39** — фикс-размер вектора при любом N.
- **modality-present флаги** — дешёвая трактовка NaN (L4-паттерн).
- **Cosine на numpy** — легко/детерминированно.

## 9. Слабые места

- **62% фич мертвы (главное).** Вся reference/uniqueness-часть (24/39) = NaN, т.к. **reference pack пуст**. Это
  **самая ценная половина** (сравнение видео = центральная продуктовая фича) — и она не работает ни на одном видео.
- **Когерентность слабо различима** — centroid_sim CV 4.8% (все ~0.9); как фича для модели/аналитика мало
  информативна (видео в CLIP-пространстве все довольно самоподобны).
- **Мёртвый код** (8 методов) — техдолг, путает; использует отсутствующие зависимости.
- **L4: `status=ok` при 62% NaN** — потребовался schema-aware валидатор (сделан); но семантически «ok» с
  большинством NaN спорно (лучше отражать reference_present в статусе).
- **Дублирование uniqueness** — `uniqueness_*` фичи пересекаются с отдельным компонентом `uniqueness`.
- **Reference pack — инфра/данные** — наполнение требует корпуса референс-видео (какие? как отобраны?).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер] Собрать reference pack** — эмбеддинги референс-корпуса (по нишам/популярности); без него
   вся ценная reference/uniqueness-часть мертва. Определить, с чем сравнивать (топ ниши? весь корпус?).
2. **[выс.] Удалить/вынести мёртвый код** (8 legacy-методов) — техдолг, отсутствующие зависимости.
3. **[сред.] Пересмотреть статус при reference-NaN** — не `status=ok` с 62% NaN, а явный флаг `reference_present`
   в статусе/отдельный degraded-статус.
4. **[сред.] Усилить когерентность** — сейчас CV 4.8%; возможно добавить сегментное самоподобие (разнообразие
   между сценами) как более различимую фичу.
5. **[сред.] Объединить с `uniqueness`** — устранить дублирование концепции uniqueness в двух компонентах.

## 11. Рекомендации по архитектуре / связям

- **Единый reference-бэкенд** для similarity_metrics + uniqueness — общий reference pack + один cosine-движок.
- **Reuse core_clip/мультимодальных** закреплён — правильно; убедиться в согласованности осей.
- **Reference pack как сервис** (аналог Embedding Service) — сравнение видео против растущего корпуса.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate | 23 batch + 6 синт | 23/23 OK | схема ок |
| U2 ось | 23 | times_s монотонны | ось ок |
| U3 диапазоны | 23 | centroid/temporal ∈[-1,1], finite при ok | когерентность здорова |
| U4 expected-empty | N=1 / no-dep | temporal len=0, no core_clip→error | edge-случаи ок |
| U5 golden | синт | max\|Δ\|=0.0 | детерминизм |
| C1 NaN by design | 23 | 24/39 NaN (ref_present=False) | reference мёртв by design |
| C2 вариативность | 23 | centroid_sim CV **4.8%** (0.83–0.985) | слабая различимость |
| **Реальный storage (мой прогон)** | 6 видео | 15 alive / **24 dead**; centroid 0.83–0.99; ref_present=False | когерентность жива (слабо), reference мёртв |

Вывод: **когерентная половина корректна, но слабо различима; ценная reference/uniqueness-половина мертва** (нет pack).

## 13. Интерпретируемость

- **Когерентность:** «однородное vs разнообразное видео» — умеренно понятно, но слабый сигнал.
- **Reference/uniqueness (когда живо):** «ваше видео на N% уникально / похоже на …» — очень понятный ценный
  инсайт; сейчас недоступен.
- **Добавить:** после reference pack — «похожие видео», «уникальность vs ниша»; сейчас показывать нечего кроме когерентности.

## 14. Польза для моделей

**Низкая в текущем виде.** Когерентность (centroid/temporal_sim) слабо различима (CV 4.8%) → мало сигнала для
Encoder. Ценная reference/uniqueness-часть (мультимодальное сходство с корпусом — сильный сигнал новизны/
позиционирования) **вся NaN**. После reference pack потенциал вырос бы существенно; сейчас — почти нет.

## 15. Польза для аналитиков

**Ограниченная.** «Однородность видео» — слабый инсайт. Главная ценность — **сравнение видео и uniqueness**
(центральная фича продукта) — мертва (нет pack). После наполнения reference — станет одной из ключевых аналитик
(«похожие ролики», «уникальность контента»). Сейчас аналитик получает только слабую когерентность.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Ценный дизайн (сравнение видео, 7 модальностей), но 62% мертво |
| 5. Выход (контракт) | 3 | F=39 стабилен + modality-флаги; но 24/39 NaN, dead code |
| 6. Фичи | 2 | Когерентность слаба (CV 4.8%); reference/uniqueness мертвы |
| 8. Оптимизации | 3 | Reuse эмбеддингов, F-стабильность; мёртвый код техдолг |
| 9. Слабые места (инверсно) | 2 | 62% NaN, слабая когерентность, dead code, L4, дубль uniqueness |
| 12. Результаты тестов | 3 | Гейты PASS + golden=0, но ценная половина мертва |
| 13. Интерпретируемость | 3 | Uniqueness понятен (когда есть); когерентность слаба |
| 14. Польза для моделей | 2 | Когерентность слаба, reference мёртв |
| 15. Польза для аналитиков | 2 | Ключевая фича (сравнение) мертва; когерентность слаба |

### Итоговые оценки

- **Польза для моделей: 2/5.** В текущем виде даёт только слабо-различимую когерентность (CV 4.8%), а ценная
  мультимодальная reference/uniqueness-часть — сильный сигнал новизны/позиционирования — вся NaN (нет reference
  pack). Потенциал 3–4 после наполнения pack; фактически почти нет сигнала.
- **Польза для аналитиков: 2/5.** Сравнение видео и uniqueness — центральная продуктовая аналитика — не работает
  (62% фич мертвы); аналитик получает лишь слабую «однородность». После reference pack ценность вырастет до
  ключевой, но балл отражает факт.

## 17. Источники

- `modules/similarity_metrics/utils/{similarity_metrics.py, similarity_metrics_library.py, validate_similarity_metrics_npz.py}`, `main.py`
- `modules/similarity_metrics/docs/SCHEMA.md`
- `DataProcessor/docs/component_reports/similarity_metrics/{REPORT_2026-07-16.md, CRITERIA.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L4 status=ok при 60% NaN)
- Cross-ref: `core_clip` (эмбеддинги), `uniqueness` (дубль концепции), reference pack `dp_models/bundled_models/similarity/reference_sets/` (пуст)
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/similarity_metrics/results.npz`
  (**все status=ok; 15 alive / 24 dead NaN; centroid_sim 0.83–0.99 CV 4.8%; reference_present=False**)

## 18. Визуализации

![similarity_metrics overview](similarity_metrics_overview.png)

`similarity_metrics_overview.png`: слева — когерентность (centroid/temporal_sim) по 6 видео: все высоки (0.83–0.99,
CV~5%) → **живо, но слабо различимо**; справа — раскладка 39 фич: 15 alive (когерентность+флаги) vs **24 dead NaN**
(вся reference_similarity_* по 7 модальностям + uniqueness — центральная фича «сравнение видео» из CLAUDE.md),
причина — пустой reference pack; + мёртвый код (8 методов) и L4. Подтверждает: работает слабая половина, ценная —
мертва до наполнения reference pack.
