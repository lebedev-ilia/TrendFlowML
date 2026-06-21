# Content Domain + Franchise/Title Recognition — Q&A (working doc)

Цель: реализовать 2 новых компонента VisualProcessor для:
- **определения домена контента** (мультик/аниме/игра/скрин-рекординг/реал и т.п.)
- **определения “что именно за франшиза/тайтл”** (какая игра/какое аниме/какой мульт), насколько это возможно.

Этот файл — рабочий Q&A: я задаю вопросы, ты отвечаешь **прямо под ними**, после чего мы фиксируем:
архитектуру, зависимости, алгоритмы v1, контракты артефактов и план реализации.

---

## 0) Контекст (что уже есть в репо)

Связанные source-of-truth документы:
- `docs/contracts/SEGMENTER_CONTRACT.md` — `union_timestamps_sec` и правила sampling.
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md` — meta/NPZ, `times_s`, `validate_npz`.
- `docs/models_docs/SCHEMA_SEMANTIC_HEADS_NPZ.md` — готовый контракт top‑K retrieval head (пер‑кадр/пер‑трек).
- `docs/baseline/components/visual/SCENE_CLASSIFICATION_BASELINE_AUDIT.md` — рекомендация: отдельные модули `content_domain` и `game_title`.

Уже реализованные “референс” компоненты:
- `core_clip` — frame embeddings + text embeddings (Triton).
- `core_place_semantics` — retrieval поверх `core_clip` (пример архитектуры).
- `text_scoring` — consumer внешнего OCR NPZ (важно для title recognition).

---

## 1) Предлагаемые 2 компонента (черновое именование)

### 1.1 `content_domain` (coarse routing)

Назначение: дать устойчивый сигнал **какой тип контента** в видео.

Примеры доменов (не финально):
- `live_action_real`
- `animation_cartoon`
- `animation_anime`
- `video_game`
- `screen_recording_ui` (в т.ч. туториалы/презентации/софт)
- `sports_broadcast` (опционально)

База/модели:
- v1 без датасета: CLIP zero-shot (prompt-ensemble) поверх `core_clip.frame_embeddings`.

### 1.2 `franchise_recognition` (game/anime/cartoon title)

Назначение: определить **какая именно франшиза/тайтл** (если возможно и уместно).

Главная оговорка (важно): в большинстве случаев “какая игра” определяется не по общему визуальному стилю, а по **UI тексту** и “screen layout” → поэтому v1 почти всегда должен опираться на OCR/текст.

База/модели:
- v1: гибрид OCR → кандидаты + CLIP retrieval/verification по оффлайн базе (опционально).

---

## 2) Contract proposal (чтобы не плодить новый формат)

Я предлагаю оба компонента делать в стиле “semantic head” (см. `SCHEMA_SEMANTIC_HEADS_NPZ.md`), потому что:
- это уже стандартизировано под top‑K ids+scores,
- удобно для encoder’а,
- прозрачно для scheduler’а (model_signature, db_digest, unit costs).

Черновой формат (одинаковая форма, разные label spaces):
- `frame_indices (N,) int32` — union-domain, строго от Segmenter
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `semantic_label_names (A,) str` — `"id:name"`
- `frame_topk_ids (N,K) int32`, `frame_topk_scores (N,K) float32`
- `track_ids (1,) int32` + `track_topk_* (1,K)` — агрегация по видео (max/mean pool)
- `*_is_confident_top1` — только флаг, **без обрезания top‑K**
- `meta` — стандарт + `models_used[]`, `model_signature`, `db_*` (если есть база)

Если ты хочешь, чтобы это были “VisualProcessor modules”, а не `core_*` providers — ок: важнее контракт/NPZ и зависимости, место в дереве можно решить отдельно.

---

## 3) Вопросы (Round 1) — ответь под каждым

### 3.1 Product scope / output expectations

**Q1.1** Где это будет использоваться в первую очередь?
- A) только в popularity‑модели (encoder features)
- B) ещё и в аналитике/дашбордах (человеко‑читаемые результаты)

**A1.1**: и там и там. При этом модельные фичи, также можно адаптировать под аналитику.

**Q1.2** Нужен ли результат **per-frame** (кривая по времени), или достаточно **per-video** (1 итоговый домен/тайтл)?
Я рекомендую: per-frame + per-video aggregate (и то, и то).

**A1.2**: Да **per-frame** нужен и я даже больше скажу, на одном кадре может быть два класса, например если какой то блогер снимает себя на веб-камеру + демонстрация экрана с какой то игрой.

**Q1.3** Какой минимальный список доменов ты хочешь в v1? (5–8 доменов, без мелкой детализации)

**A1.3**: Касательно классификации на: Игра, Аниме, Мультик и тд. то тут достаточно 3-6, можешь придумать еще от себя 2-3. А касательно расспознавания конкретной игры или мультика, то тут нужно больше классов, например взять топ-100 самых популярных игр, аниме и мультиков. Но сложность заключаеться в том что бы аддаптировать эти алгоритмы к тому что если например выходит какая то новая игра то алгоритмы должны это учитывать и модель в том числе, так как это огромный фактор популярности.

**Q1.4** Для “тайтла” (второй компонент): мы ограничиваемся только **играми**, или это общий компонент для:
- games + anime + cartoons (одна “franchise” база)
- или делаем только games v1, остальное позже

**A1.4**: для всех

### 3.2 Baseline / deployment / fail-fast

**Q1.5** Эти 2 компонента входят в baseline DAG (`docs/reference/component_graph.yaml: stages.baseline`) или это `v1`/`extended` стадия?
Важно, потому что baseline подразумевает строгий audit/bench/resource_costs сразу.

**A1.5**: baseline

**Q1.6** No-fallback policy по данным:
- если нет OCR (или OCR NPZ отсутствует) — `franchise_recognition` должен быть `empty` (валидно), или `error` (fail-fast)?

**A1.6**: я думаю что OCR не обязателен для этого и можно использовать другие алгоритмы для классфификации и распознавания

### 3.3 Dependencies & sampling

**Q1.7** Подтверди зависимости:
- `content_domain` зависит от `segmenter` + `core_clip` (обязательно).
- `franchise_recognition` зависит от `segmenter` + (A) OCR artifact (если есть) + (B) `core_clip` (если делаем визуальную verification).

**A1.7**: Да

**Q1.8** Sampling: достаточно ли `core_clip` sampling (N≈200) для domain+title, или нужен отдельный sampling budget?
Варианты:
- A) reuse `core_clip` frame_indices (простота + alignment)
- B) отдельная группа от Segmenter (дороже, но можно плотнее в “text-heavy moments”)

**A1.8**: Давай пока возьмем A

### 3.4 Label space / базы / языки

**Q1.9** Язык выходных label names:
- A) EN canonical ids + RU aliases в базе (как в semantic bases guide)
- B) сразу RU labels

**A1.9**: Зависит от качества, лучше коненчо RU, но если это повлияет на качество то можно и EN. Некритично

**Q1.10** Размер базы “тайтлов” (порядок величины) на v1:
- 200? 1k? 10k?

**A1.10**: уже говорил в **A1.3**. Давай для начала возьмем 100. Если все получиться, потом увеличим.

**Q1.11** Откуда будем брать базу тайтлов (source-of-truth)?
Варианты:
- A) ручной curated список (старт)
- B) выгрузка из IGDB/Steam/Wiki, но **offline** пакет с лицензиями (no-network at runtime)
- C) смесь

**A1.11**: Давай попробуем и то и то. Я могу искать руками, но можно и алгоритмами, что бы не тратить кучу времени.

**Q1.12** Нужны ли “платформенные” различия (PC/console/mobile) или не важно?

**A1.12**: Нужны, но это не супер важно

### 3.5 Algorithm choices (v1)

**Q1.13** `content_domain` v1:
Ок ли стартовать с CLIP zero-shot (prompt ensemble) без обучения?

**A1.13**: Наверное норм, все будет зависеть от качества выхода.

**Q1.14** `franchise_recognition` v1:
Какой минимум сигнала обязателен?
- A) OCR-first (из текста интерфейса) → нормализуем строки → сопоставляем с базой (aliases)
- B) OCR + CLIP verification (визуально подтверждаем топ‑кандидатов)
- C) CLIP-only retrieval по gallery изображений (скорее дорого/неустойчиво без большой базы)

**A1.14**: OCR + CLIP verification

**Q1.15** Требования к качеству (что считаем “успехом” на старте):
- domain: точность top‑1 на 30‑видео golden set?
- title: хотя бы top‑5 часто содержит правильный тайтл, если есть UI текст?

**A1.15**: Пока незнаю, реши сам

### 3.6 Output format & evidence

**Q1.16** Нужны ли “evidence frames” для дебага/аналитики:
- например топ‑5 кадров, которые сильнее всего поддерживают выбранный домен/тайтл.

**A1.16**: Да

**Q1.17** Как фиксируем “не уверен”:
я предлагаю: всегда писать top‑K ids+scores, а уверенность — через `*_is_confident_top1` (thresholds per-label или global).
Ок?

**A1.17**: Ок

---

## 4) Следующий шаг после ответов

После твоих ответов я подготовлю:
- финальные имена компонентов + место в дереве (`modules/` vs `core/model_process/`)
- DAG зависимости в `docs/reference/component_graph.yaml`
- README каждого компонента (в папке компонента) + ссылку/индекс в docs
- точный NPZ schema_version и список ключей
- v1 алгоритм + cost controls + план бенчмарков/resource_costs

---

## 5) Resolved decisions (фиксируем по твоим ответам)

- **Use cases**: и popularity‑модель, и аналитика/дашборды (A1.1).
- **Output granularity**:
  - per-frame top‑K обязателен,
  - допускается multi-label per frame (например webcam + screen) (A1.2).
- **Domain labels v1**: 3–6 основных доменов + 2–3 дополнительных по инициативе реализации (A1.3).
- **Franchise scope**: games + anime + cartoons в одном head’е (A1.4).
- **DAG stage**: baseline (A1.5).
- **OCR policy**:
  - OCR полезен, но не обязателен как hard‑dependency (A1.6),
  - v1 алгоритм всё равно: OCR + CLIP verification (A1.14),
  - если OCR отсутствует — компонент обязан продолжать работу (через CLIP‑only retrieval/verification).
- **Dependencies**: `segmenter` + `core_clip` (A1.7), sampling = reuse `core_clip.frame_indices` (A1.8).
- **Language**: prefer RU если не ухудшает качество, иначе EN (A1.9).
- **Franchise DB size**: стартуем с 100 (A1.10), затем расширяем.
- **DB sourcing**: смесь ручного куруирования + автоматизации (A1.11).
- **Platform differences**: нужны, но не критично в v1 (A1.12).
- **Confidence**: always output top‑K, `*_is_confident_top1` для “уверен/не уверен” (A1.17).
- **Evidence**: нужны evidence frames (A1.16).

## 6) Proposed architecture (v1, чтобы начать код)

Я предлагаю реализовать **2 core providers** в `VisualProcessor/core/model_process/` (аналогично `core_place_semantics`):

1) `content_domain`
   - input: `core_clip/embeddings.npz` (frame embeddings) + `core_clip.frame_indices` из frames metadata
   - method: CLIP text retrieval по небольшой базе доменов (prompt ensemble)
   - output: per-frame top‑K + per-video aggregate (track=1)

2) `franchise_recognition`
   - input: `core_clip/embeddings.npz` + (optional) OCR NPZ
   - method:
     - OCR → кандидаты (aliases match) + “evidence frames”
     - CLIP verification (text prompts vs frame embeddings)
     - fallback when OCR missing: CLIP-only retrieval over label space
   - output: per-frame top‑K + per-video aggregate (track=1) + evidence frame indices for top‑K

В обоих:
- NPZ contract: на базе `SCHEMA_SEMANTIC_HEADS_NPZ.md` (top‑K ids+scores, no hard gating).
- Базы/labels: offline packages в `DataProcessor/dp_models/bundled_models/semantics/...` с `manifest.json` + `*.jsonl` + optional `thresholds.json`.
---

## Навигация

[README](README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
