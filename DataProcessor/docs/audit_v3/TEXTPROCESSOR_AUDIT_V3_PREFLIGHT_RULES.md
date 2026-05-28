# Audit v3 — TextProcessor preflight rules (source-of-truth)

Дата: 2026-04-01  
Статус: **FINAL** (рабочий пакет правил старта аудита TextProcessor)

Документ фиксирует **preflight** для Audit v3 **TextProcessor**: обязательный путь с ASR, smoke-набор с контролируемыми текстами, порядок всех 22 экстракторов, единая модель эмбеддингов, corpus/FAISS packs, политика `tags_extractor`, структура отчётов и run-log.

**Быстрые ссылки (минимум контекста для LLM)**:

| Назначение | Путь |
|------------|------|
| Карта всех text extractors | `DataProcessor/TextProcessor/docs/MAIN_INDEX.md` |
| Общие критерии Audit v3 (Text §6) | `DataProcessor/docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md` |
| Решения Audit v3 | `DataProcessor/docs/audit_v3/DECISIONS_AND_RULES.md` |
| NPZ / meta / privacy | `DataProcessor/docs/contracts/ARTIFACTS_AND_SCHEMAS.md`, `PRIVACY_AND_RETENTION.md` |
| Система схем | `DataProcessor/docs/contracts/SCHEMAS_SYSTEM.md` |
| Интерфейс моделей v2 | `Models/docs/contracts/MODEL_INTERFACE_V2.md` |
| Индекс описаний компонентов | `DataProcessor/docs/COMPONENTS_DESC_INDEX.md` → секция TextProcessor |
| Отчёты по компонентам (заполняются по ходу аудита) | `DataProcessor/TextProcessor/docs/audit_v3/components/` |
| Лог прогонов | `DataProcessor/docs/audit_v3/RUN_LOG.md` |

---

## 0) Решения владельца (зафиксированы из Q&A)

1. **Smoke-набор**: расширить за счёт **качественно сгенерированных** текстов и метаданных; эталон проверяется вручную владельцем.
2. **Upstream**: аудит TextProcessor выполняется **только с работающим ASR** (Segmenter → AudioProcessor `asr_extractor` → конвейёр текста). Пропускать ASR для «только метаданные» в рамках Audit v3 **нельзя**.
3. **Scope**: **все 22** экстрактора; порядок — см. §2.
4. **Corpus / FAISS packs** (similar titles, cluster entropy и т.д.): допускается **выстроить и задокументировать по ходу аудита**; после фиксации — обязательны `pack_version` / `pack_digest` (или эквивалент) в артефакте или README компонента. До фиксации model_facing для корпус-зависимых сигналов — по правилам `AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md` §6.4 (по умолчанию **analytics**, не model_facing).
5. **Единая модель эмбеддингов (Audit v3)**: **intfloat/multilingual-e5-large** через **ModelManager / `dp_models`** (offline). Спецификация: `DataProcessor/dp_models/spec_catalog/text/intfloat_multilingual-e5-large.yaml`. Обоснование: единая размерность **1024**, сильная многоязычность (в т.ч. RU), уже принята в `DataProcessor/configs/global_config.yaml` для text-embedder’ов. Во время аудита **не смешивать** с `all-MiniLM-L6-v2` и др. без bump схемы и явного Decision Record.
6. **`tags_extractor`**: обязан выполняться **до** любых экстракторов, которые рассчитывают на «очищенные» от хэштегов `title`/`description` (см. §5). Топологический порядок в раннере должен это гарантировать; аудит фиксирует нарушение как **fail**.
7. **Структура артефактов аудита**: как у Audio — этот файл + per-component отчёты в `TextProcessor/docs/audit_v3/components/<component>_AUDIT_V3_REPORT.md` (при необходимости `*_AUDIT_V3_QUESTIONS_R1.md`).

---

## 1) Audit pack (smoke + validation) — рабочий контракт

### 1.1 Видео (для реального ASR)

Использовать **audio-present** набор (как в Audio preflight), чтобы Segmenter/`audio.wav`/`segments.json` и **asr_extractor** отрабатывали на живом аудио:

- `example/example_videos/video1.mp4`
- `example/example_videos/video2.mp4`
- `example/example_videos/video3.mp4`

При необходимости **empty / no-audio** регрессий TextProcessor — опционально добавить `*_fixed.mp4` отдельным подпунктом в `RUN_LOG.md` (не заменяет основной text+ASR pack).

### 1.2 Тексты и метаданные (VideoDocument) — качественная генерация

**Назначение**: воспроизводимые, осмысленные **title / description / comments** (и при необходимости поля канала), согласованные с темой видео, чтобы человек мог проверить лексику, эмбеддинги, темы, Q&A-эвристики.

**Рабочая директория (рекомендуемая)**:

- `example/text_audit_v3_smoke/`

**Шаблон фикстуры (копировать в сценарий)**:

- `example/text_audit_v3_smoke/_template/` — `video_document.example.json`, `video_document.schema.json`, краткие правила в `README.md`.

**20 готовых разнообразных сценариев (inference + training)**:

- `example/text_audit_v3_smoke/scenarios/audit_v3_20_scenarios.json` (+ `README.md`, `generate_20_scenarios.py`).
- **Автопроверка TextProcessor по сценариям (dev)**: `DataProcessor/TextProcessor/scripts/smoke_each_extractor_audit_v3.py` — для каждого из 22 экстракторов минимальная цепочка зависимостей, опционально все сценарии подряд (`--all-scenarios`). Подробности и команды: `example/text_audit_v3_smoke/scenarios/README.md`.

**Конвенция (минимум)**:

- Один каталог или префикс на сценарий, например `example/text_audit_v3_smoke/<scenario_id>/`.
- **Обязательно**: `video_document.json` (или эквивалент), совместимый с `TextProcessor/src/schemas/models.py` (`VideoDocument`), с явным `video_id`, соответствующим прогону result_store.
- **Обязательно**: короткий `README.md` в каталоге сценария: что за кейс (язык, длина, наличие хэштегов, токсичность/краевые случаи), ожидаемое поведение extractors на уровне «sanity».
- Комментарии: достаточное число и дисперсия (короткие/длинные, разные лайки, дедуп), чтобы отработали `comments_embedder` / `comments_aggregator`.

Точные имена файлов могут быть уточнены при первом прогоне; **source-of-truth для аудита — наличие документированного набора + запись в `RUN_LOG.md`**.

### 1.3 Связка «видео ↔ документ»

В прогоне DataProcessor должны совпасть:

- идентификаторы **`platform_id` / `video_id` / `run_id`**;
- путь к **VideoDocument** для TextProcessor (`processors.text.input_json` или эквивалент в CLI/worker);
- артефакт **ASR** в том же `run_id`, от которого TextProcessor читает transcript/tokens (контракт см. в README `asr_extractor` и text extractors).

---

## 2) Порядок аудита всех 22 экстракторов (working order)

Порядок выбран по **Tier зависимостей** (`TextProcessor/docs/MAIN_INDEX.md`). Внутри Tier при конфликте — топологический порядок графа зависимостей в коде/конфиге.

**Tier-0 (независимые, сначала)**

1. `tags_extractor` — **первым среди text**, если включён shared документ с последующими embedder’ами (мутация `doc`).
2. `lexico_static_features`
3. `asr_text_proxy_audio_features`

**Канон транскрипта для полного Audit v3**: `LexicalStatsExtractor` с **`transcript_source_policy="asr_only"`** (без fallback на legacy `transcripts`). Профили с `asr_then_legacy` — только для явно помеченных degraded/dev прогонов в `RUN_LOG.md`. Опция **`require_transcript=true`** — для строгих прогонов с обязательным текстом ASR.

**Tier-1 (эмбеддинги; единая модель §0 п.5)**

4. `title_embedder`
5. `description_embedder`
6. `hashtag_embedder` (зависит от тегов, зафиксированных `tags_extractor`)
7. `transcript_chunk_embedder` (ASR)
8. `comments_embedder`
9. `speaker_turn_embeddings_aggregator` (после transcript/diarization — см. §4)

**Tier-2 (агрегации / пары / темы)**

10. `transcript_aggregator`
11. `comments_aggregator`
12. `qa_embedding_pairs_extractor`
13. `embedding_pair_topk_extractor`
14. `semantics_topics_keyphrases`

**Tier-3 (метрики / корпус / кластеры)**

15. `embedding_stats_extractor`
16. `cosine_metrics_extractor`
17. `title_embedding_cluster_entropy_extractor`
18. `title_to_hashtag_cosine_extractor`
19. `semantic_cluster_extractor`
20. `topk_similar_titles_extractor`
21. `embedding_shift_indicator_extractor`
22. `embedding_source_id_extractor`

При изменении порядка по факту графа — **зафиксировать причину** в `RUN_LOG.md` и в отчёте по затронутому компоненту.

---

## 3) ASR и аудит (обязательно)

- В профиле аудита **включены** Segmenter, AudioProcessor с **`asr_extractor`**, затем TextProcessor с **тем же run**.
- Транскрипт для text-конвейера: **истина из ASR** (token/text policy — см. контракты ASR и README конкретных extractors; без противоречий privacy § в критериях).
- Если ASR дал `empty`/`error` — TextProcessor ведёт себя согласно **empty semantics** компонента; аудит фиксирует, что поведение соответствует `SCHEMA.md` и не маскирует fail-fast там, где он требуется.

---

## 4) Зависимость от speaker diarization

`SpeakerTurnEmbeddingsAggregator` опирается на диаризацию из AudioProcessor.

**Preflight**: для полного прохода **рекомендуется** включить `speaker_diarization_extractor` в том же run, что и ASR.

Если диаризация отключена — аудит обязан задокументировать **фактическое** поведение (empty / degraded) в README + отчёте компонента; это не отменяет требование **ASR**.

---

## 5) Политика `tags_extractor` (hard)

- В конфиге аудита **`tags_extractor.mutate_doc_clean_texts`** и извлечение хэштегов должны быть согласованы с документацией; downstream embedder’ы title/description **не должны** получать «сырой» текст с `#...`, если контракт компонента предполагает очистку.
- Проверка аудита: в отчёте `tags_extractor` + одного зависимого embedder’а (например `title_embedder`) явно указано, что порядок выполнения и мутации документа **проверены**.

---

## 6) ModelManager-only и эмбеддинги

- Все ML-модели текста в audited режиме: **только** через `dp_models` / ModelManager, **no-network**.
- В `meta`: `models_used[]`, `model_signature`, **weights_digest** — по факту resolved spec (см. `DataProcessor/common/meta_builder.py`).
- Запрещено в audited состоянии подменять `intfloat/multilingual-e5-large` на другую модель без отдельного решения и bump схем.

---

## 7) Corpus / FAISS packs (выстраиваются в ходе аудита)

Заготовка путей/версий (плейсхолдеры до фиксации pack’ов):  
`DataProcessor/TextProcessor/config/corpus_packs.placeholder.yaml`

Компоненты: прежде всего `topk_similar_titles_extractor`, `title_embedding_cluster_entropy_extractor`, опционально связанные с `semantic_cluster` / taxonomy.

**Минимум при объявлении pack «закрытым» для model_facing**:

- Версия: `pack_version` (строка).
- Целостность: `pack_digest` (хэш каталога/файлов или реестра).
- Политика anti-leakage: time-frozen / train-infer parity — текстом в `SCHEMA.md` или Decision Record.
- Путь к артефакту в репозитории или инструкция сборки — в README компонента.

До этого сигналы таких компонентов в baseline модели — **analytics** (см. общие критерии §6.4).

---

## 8) Схемы и валидация

Как в `AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md` §3:

- Human: `DataProcessor/TextProcessor/src/extractors/<name>/SCHEMA.md`
- Machine: `DataProcessor/TextProcessor/schemas/<schema_version>.json`
- Известные схемы — runtime fail-fast (`SCHEMAS_SYSTEM.md`).

---

## 9) Per-component отчёты и run-log

**Отчёт по компоненту** (шаблон смысла — как у Audio):

- по файлу: `DataProcessor/TextProcessor/docs/audit_v3/components/<component_name>_AUDIT_V3_REPORT.md`
- минимальные секции: TL;DR, versions/schema, inputs, outputs (tiers), empty semantics, privacy, sampling/requirements, acceptance, ссылка на запись в `RUN_LOG.md`

**`RUN_LOG.md`** после существенных изменений:

- команда / профиль / `config_hash`
- список text extractors
- для каждого сценария: какие поля VideoDocument использованы, сколько комментариев после отбора, число чанков транскрипта, `D=1024`, доля mask/empty для token-ready полей (если есть)

---

## 10) LLM-исполнитель (Composer 2 Fast) — как пользоваться этим документом

1. Открыть **этот preflight** + **§6** в `AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md`.
2. Взять **один** `component_name` из §2 списка.
3. Навигация по коду/докам: `TextProcessor/docs/MAIN_INDEX.md` → README экстрактора → `SCHEMA.md` → `main.py`.
4. Не загружать в контекст весь репозиторий: только перечисленные файлы + NPZ/render пути из `RUN_LOG.md`.
5. Закрытие компонента = отчёт в `TextProcessor/docs/audit_v3/components/` + запись в `RUN_LOG.md`.

---

## Связанные документы (повтор)

- `DataProcessor/docs/audit_v3/AUDIOPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md` — аналог для Audio (путь ASR, segments, empty audio).
- `docs/PRODUCT_ROADMAP_TO_PRODUCTION.md` — Фаза 1 (Text Audit v3).
