# Audit v4 — общий итог по трём процессорам (L1, набор **A**)

**Дата сводки:** 2026-04-06  
**Дополнение (TextProcessor L2 / итог по компонентам):** 2026-04-15  
**Сводка Audit v4.2 (L2) по всем процессорам:** [AUDIT_V4_2_L2_CROSS_PROCESSORS_SUMMARY.md](AUDIT_V4_2_L2_CROSS_PROCESSORS_SUMMARY.md)  
**Общий опорный run (набор A):** `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`  
**План и критерии:** [AUDIT_4_CRITERIA_AND_PLAN.md](AUDIT_4_CRITERIA_AND_PLAN.md)  
**Журнал:** [RUN_LOG.md](RUN_LOG.md)  
**Индекс отчётов:** [components/README.md](components/README.md)

## Сводная таблица по подсистемам

| Подсистема | Сводка L1 | Компонентов с отчётом **L1** | Артефакт(ы) на **A** (типично) |
|------------|-----------|-------------------------------|----------------------------------|
| **AudioProcessor** | [AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md) | **21** | Per-extractor NPZ под `audio_processor/` / сегментные режимы |
| **TextProcessor** | [AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md) | **22** | Агрегат `text_processor/text_features.npz` + `_artifacts/*.npy` |
| **VisualProcessor** | [AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md) | **23** (17 модулей + 6 core) | Per-module / core NPZ под `visual_processor/` и связанные пути |
| **Итого** | — | **66** | Один e2e **A** как горизонтальная склейка |

## Оценка волны (L1, условно)

| Метрика | Значение | Комментарий |
|---------|----------|--------------|
| Оценка **AudioProcessor** (программный итог в сводке) | **~8 / 10** | Субъективный итог волны + акцент на классах дефектов tabular/meta; часть строк в NPZ шла в NaN до фиксов саверов |
| Оценка **TextProcessor** (среднее по таблице вердиктов) | **≈8.15 / 10** (~**8.2**) | 22 строки; коридор **~7.7–8.4** по компонентам |
| Оценка **VisualProcessor** (среднее по вердиктам в сводке) | **~8.3 / 10** | 23 отчёта; выброс вниз **`action_recognition` ~7** |
| **Сводный балл трёх подсистем** | **~8.1–8.2 / 10** | Простое среднее «головных» цифр **~(8.0 + 8.15 + 8.3) / 3 ≈ 8.15**; с учётом лёгкого завышения отдельных аудио-строк — разумно держать формулировку **~8.1–8.2** для всей L1-волны |

Интерпретация: по всем трём процессорам **L1 закрывает «контракт ↔ эмпирика на одном run»**, но **не** закрывает **§8 DoD**, **§4.8 golden**, наборы **B/C**. Итог **~8.1–8.2** означает «сильный draft»: архитектура признаков и склейка e2e на **A** в целом состоятельны, а узкие места уже видны по отчётам.

## TextProcessor — итог по компонентам (L1 + L2 tooling)

Каноническая развёрнутая таблица по всем **22** экстракторам (вердикт L1 на **A**, краткие заметки, список следующих шагов по каждому имени): [AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md). Журнал прогонов и путей **L2**: [RUN_LOG.md](RUN_LOG.md). Инженерный bridge после L2: [components/audit_4_2/README.md](components/audit_4_2/README.md) (папка `audit_4_2/text_processor/`).

### L1 (набор **A**)

- Отчёты: по одному `*_audit_v4.md` на компонент в [components/text_processor/](components/text_processor/).
- Оценка волны: среднее по вердиктам **≈8.15 / 10** (~**8.2**); коридор **~7.7–8.4**. Минимум на **A**: **`speaker_turn_embeddings_aggregator`** (~7.7). Максимум: **`embedding_pair_topk_extractor`**, **`embedding_source_id_extractor`** (~8.4).
- Артефакт: агрегат `…/text_processor/text_features.npz` (`feature_names` / `feature_values`) плюс связанные `_artifacts/*.npy` и `_render/*.html` где применимо.
- Общий долг L1-уровня (для всех имён): наборы **B/C**, **§4.8** golden, прогоны с иными флагами (**`emit_extra_metrics`**, второй источник транскрипта, FAISS vs numpy и т.д.) — см. нумерованный список в конце текстовой сводки TextProcessor.

### L2 (целевые **A+B**, **5** путей из `result_store`)

- Для **всех 22** компонентов заведены скрипты статистики (в дереве экстракторов, путь в каждой записи `RUN_LOG`), выходной JSON: `storage/audit_v4/<component>_l2/<component>_audit_v4_stats.json`.
- **Фактический паттерн на текущем storage:** на **2/5** путях `meta.status=ok` у `text_processor` и в NPZ есть табличный срез соответствующего префикса `tp_*` (и проверки артефактов там, где скрипт это делает); на **3/5** путях пайплайн `text_processor` падает до полного табличного слоя (`feature_names` пустой, детали в поле `text_processor_error` / `dataset_quality` в JSON).
- Следствие: **полный L2 по TextProcessor на этом наборе run формально blocked** — нужны **5/5** успешных e2e `text_processor` на тех же id (или пересборка `result_store` / исправление корневой ошибки пайплайна). Блокер **сквозной** для всех экстракторов, а не специфичный отдельному модулю.
- Канонические отчёты подняты до уровня **L2** в части методологии (наборы A+B, ссылки на JSON, блок про блокировку) по мере обновления; несоответствие «заявлено 5 путей / фактически 2» везде явно помечено.

### Компактная матрица по именам (L1 на **A**)

| Компонент | L1 (~) | Заметка на **A** (одна строка) |
|-----------|--------|--------------------------------|
| `asr_text_proxy_audio_features` | 8.2 | `tp_asrproxy_*` (37); нужны B/C для пустых/token путей |
| `comments_embedder` | 7.8 | `tp_commentsemb_*`; при `emit_extra_metrics=false` — NaN в диагностике; матрица **(5,1024)**, L2≈1 |
| `comments_aggregator` | 8.2 | 39 ключей; NaN по std/timing от конфига |
| `description_embedder` | 8.2 | `tp_descemb_*`; вектор **(1024,)**, L2=1; `emit_extra_metrics` в коде не используется |
| `cosine_metrics_extractor` | 8.3 | `tp_cos_*`; whisper; часть полей NaN при aggregates + `emit_extra_metrics=false` |
| `embedding_pair_topk_extractor` | 8.4 | 69 ключей; 1 чанк → в основном top1 |
| `embedding_shift_indicator_extractor` | 8.0 | мало чанков vs `require_min_chunks` → `present=0` |
| `embedding_source_id_extractor` | 8.4 | `tp_embid_*`; vector_id от агрегата транскрипта |
| `embedding_stats_extractor` | 7.9 | мало чанков → `present=0`, дисперсии NaN |
| `hashtag_embedder` | 8.3 | `tp_hashemb_*`; `emit_extra_metrics` не меняет таблицу |
| `lexico_static_features` | 8.2 | `tp_lex_*` (67); NaN без эмодзи / при выкл. NER |
| `qa_embedding_pairs_extractor` | 7.8 | на A нет вопросов → `present=0` |
| `semantic_cluster_extractor` | 8.3 | `use_faiss` vs `backend_faiss=0` (numpy) |
| `semantics_topics_keyphrases` | 8.2 | `tp_topics_*` (116); много слотов NaN от режимов export |
| `speaker_turn_embeddings_aggregator` | 7.7 | нет diar+ASR на A → `present=0` |
| `tags_extractor` | 8.2 | `tp_tags_*` (43); пустые top-слоты → NaN |
| `title_embedder` | 8.2 | `tp_titleemb_*`; `title_embedding.npy`, L2=1 |
| `title_embedding_cluster_entropy_extractor` | 8.2 | numpy backend при желании FAISS |
| `title_to_hashtag_cosine_extractor` | 8.3 | `tp_titlehashcos_*`; cosine из артефактов |
| `topk_similar_titles_extractor` | 8.2 | `tp_topktitles_*`; top‑K по корпусу, numpy path |
| `transcript_chunk_embedder` | 8.2 | `tp_tchunk_*`; whisper 1×(1,1024); youtube_auto off на A |
| `transcript_aggregator` | 8.2 | `tp_tragg_*` (19); много NaN при `emit_extra_metrics=false`; agg `.npy` на happy-path |

## Сквозные темы (все три процессора)

1. **Дисциплина tabular vs meta.** Audio: строки и категориальные поля не должны попадать во float-NPZ как NaN. Text/Visual: часть диагностики сознательно **NaN** при **`emit_extra_metrics=false`** или масках — потребитель обязан читать флаги **`present` / маски**.
2. **Один reference **A** не покрывает хвосты.** Короткое аудио, пустые семантики, второй источник транскрипта, другой Segmenter (N кадров), edge OCR — требуют **B/C** и повторов **A** после правок.
3. **FAISS / numpy.** Text (и местами общая инфраструктура): конфиг «use faiss» vs фактический **backend** при отсутствии пакета — везде, где есть inner-product по матрицам, нужно явно различать.
4. **Согласованность enabled vs факт.** Audio (`speech_analysis` / pitch), Text (`embedding_stats` при малом числе чанков) — метаданные «включено» должны совпадать с реально смёрженными ветками.
5. **Визуальные вероятности и NaN-оси.** Visual: top‑k скоры не обязаны суммироваться в 1; первый кадр / отсутствие лица дают ожидаемые NaN — контракт для моделей должен быть явным.

## Сильные стороны кросс-процессорно

- Единый **набор A** позволяет сопоставлять трассы (ASR → текст → визуальные модули, опирающиеся на те же run id).
- **Machine-schema + префиксы** (`tp_*`, NPZ-ключи аудио/визуала) в L1 в основном **согласованы** с прогоном.
- Отчёты заведены **покомпонентно**; общий долг (B/C, golden) явно вынесен в сводках.

## Общие следующие шаги

1. Зафиксировать **git commit** и обновить **RUN_LOG** после стабилизации аудио/визуальных фиксов.  
2. **Повторный прогон A** там, где менялись саверы, схемы или продюсеры (**action_recognition**, audio tabular и т.д.).  
3. **§4.8** (golden / hash) по свежим артефакатам для пилотных компонентов.  
4. **Набор B** (диверсификация) и **C** (edge) — по плану §3.  
5. Для TextProcessor — **разблокировать L2**: **5/5** успешных `text_processor` на стандартных путях A+B (сейчас **2/5** OK, **3/5** error — см. JSON `dataset_quality` и `RUN_LOG`); для Visual — унифицировать **`models_used`** где пусто при работающем upstream.

## До набора B (N≥5): что сделать в коде, логике и документах

Цель: на **B** измерять **распределения и хвосты**, а не «тихие» NaN, сломанный tabular и рассинхрон схема↔продюсер. Опора: [**§4.1a** табличных типов](AUDIT_4_CRITERIA_AND_PLAN.md), L1-сводки Audio/Text/Visual.

### Блокеры и повтор A (желательно до B)

| Область | Действие |
|---------|-----------|
| **Visual** | **action_recognition:** схема / `ResultsStore` / продюцер / доки — **сделано в коде**; осталось **повторить A** и обновить артефакты в storage. |
| **Audio** | **npz_savers / tabular:** категориальные поля (**`device_used`**, **`backend`**, **`f0_method`** и т.п.) не через **`as_float` → NaN**; вынести в **meta** или typed ключи. Дополнительно: **spectral_extractor** — полный **`run_segments`** payload (`hop_length`, `n_fft`, `duration` и др. по отчёту). |
| **Audio** | **`meta.features_enabled` ↔ фактический merge** (как **speech_analysis** / pitch): не заявлять ветки без смёрженных данных. |
| **Репро** | В **RUN_LOG**: **git commit**, **config_hash**, ссылка на полный **e2e**-лог для **A** (план §2). |

### TextProcessor: логика и конфиг

| Действие | Зачем на B |
|----------|------------|
| **`emit_extra_metrics` / `compute_std`:** подключить к **`features_flat`** там, где заявлено в README, или убрать из документации. | **Сделано (доки):** README **title/description/hashtag embedder** — `emit_extra_metrics` не гейтит `features_flat` в v1.2.0 (согласовано со SCHEMA); **transcript_aggregator** — явное предупреждение про NaN и включение флагов для B. Логика экстракторов с настоящим гейтингом (**tchunk**, **comments_embedder**, …) без изменений — там по-прежнему нужен **`emit_extra_metrics=true`** в YAML для заполнения. |
| Документировать **FAISS в конфиге vs фактический backend** (numpy fallback) для кластеризации / top‑K корпуса. | **Сделано:** [`TextProcessor/docs/FAISS_AND_NUMPY_BACKEND.md`](../../TextProcessor/docs/FAISS_AND_NUMPY_BACKEND.md) + ссылка в [`MAIN_INDEX.md`](../../TextProcessor/docs/MAIN_INDEX.md). |
| Опционально: один **повторный A** с **`emit_extra_metrics=true`** на цепочке transcript/stats/cosine. | Операционный шаг для storage / §4.8 — по плану после смены профиля. |

### VisualProcessor: контракты

| Действие | Зачем |
|----------|--------|
| **`meta.models_used`** там, где модель работала, а поле пустое. | Частично: **`shot_quality`** (CLIP из `impl_meta`); остальные модули — по мере разбора отчётов. |
| В **SCHEMA/README**: top‑k / logits **не обязаны** суммироваться в 1; маски (**face**, **flow**, первый кадр). | Частично: уточнено для **`shot_quality`** (`quality_probs` vs `shot_quality_topk_probs`); остальное — при необходимости по компонентам. |
| **core_object_detections:** в контракте явно — **нет track id**. | **Сделано:** `docs/SCHEMA.md` (contract notes). |

### Инструментарий и курация B

| Действие | Зачем |
|----------|--------|
| Скрипт/ноутбук: по **N** путям — доли **NaN/Inf**, **shape/dtype**, временная ось vs Segmenter (**§4.1–4.2** плана). | Масштабируемый L2. |
| Отобрать **≥5** видео (длительность, речь/музыка/тишина, плотность текста/OCR, разный **N** кадров); записать id в **RUN_LOG**. | Соответствие **набору B** §3. |
| **Набор C** (edge) подготовить отдельно; не обязателен для первого захода **B**. | План §3. |

### Можно отложить до после первого B

- Полное **§8 DoD** и **`passed`** в RUN_LOG (**L3**).  
- **§4.8** на всех 66 компонентах — достаточно пилота на критичных цепочках.  
- Новые L1 по TextProcessor-экстракторам **вне** текущих **22** с отчётом — не блокер **B** по уже покрытым компонентам (**22/22** L1 + L2 tooling закрыты на момент дополнения 2026-04-15).

---

**Дочерние сводки (детализация):**

- [AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md)  
- [AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md)  
- [AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md)
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
