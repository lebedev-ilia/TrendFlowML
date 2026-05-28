# TextProcessor — Audit v4: общий итог (L1, набор **A**)

**Дата сводки:** 2026-04-06  
**Опорный run (набор A):** `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`  
**План и критерии:** [AUDIT_4_CRITERIA_AND_PLAN.md](AUDIT_4_CRITERIA_AND_PLAN.md)  
**Журнал прогонов:** [RUN_LOG.md](RUN_LOG.md)  
**Каталог отчётов:** [components/README.md](components/README.md)

## Статус волны

Покрыты **22** компонента с отчётами **L1 (draft)** на артефакте **A**. Остальные экстракторы TextProcessor — вне сводки до отдельных отчётов.

| Компонент | Отчёт | Вердикт (L1) | Кратко на **A** |
|-----------|--------|--------------|-----------------|
| `asr_text_proxy_audio_features` | [asr_text_proxy_audio_features_audit_v4.md](components/text_processor/asr_text_proxy_audio_features_audit_v4.md) | **~8.2**/10 | 37 `tp_asrproxy_*`; merge NPZ; нужны B/C для empty/token path |
| `comments_embedder` | [comments_embedder_audit_v4.md](components/text_processor/comments_embedder_audit_v4.md) | **~7.8**/10 | 18 `tp_commentsemb_*`; **`emit_extra_metrics=false`** → NaN в диагностике и **`artifact_written`**; матрица **(5,1024)** L2≈1 |
| `comments_aggregator` | [comments_aggregator_audit_v4.md](components/text_processor/comments_aggregator_audit_v4.md) | **~8.2**/10 | 39 ключей трёх префиксов; схема ↔ NPZ; **5×1024** эмбей; NaN в `*_std`/timing по конфигу; нужны B/C |
| `description_embedder` | [description_embedder_audit_v4.md](components/text_processor/description_embedder_audit_v4.md) | **~8.2**/10 | 19 `tp_descemb_*`; **(1024,)** L2=1; **CUDA/fp16**; **`emit_extra_metrics`** в коде не используется |
| `cosine_metrics_extractor` | [cosine_metrics_extractor_audit_v4.md](components/text_processor/cosine_metrics_extractor_audit_v4.md) | **~8.3**/10 | 39 `tp_cos_*`; косинусы конечны; transcript **whisper**; **`comments_mode=aggregates`** → NaN в `tp_cos_tc_*`/timings при `emit_extra_metrics=false` |
| `embedding_pair_topk_extractor` | [embedding_pair_topk_extractor_audit_v4.md](components/text_processor/embedding_pair_topk_extractor_audit_v4.md) | **~8.4**/10 | 69 ключей; title–desc cos = **`tp_cos_title_desc`**; **1** чанк → только **top1**; extra → **NaN** `n_chunks`/источник |
| `embedding_shift_indicator_extractor` | [embedding_shift_indicator_extractor_audit_v4.md](components/text_processor/embedding_shift_indicator_extractor_audit_v4.md) | **~8.0**/10 | 27 `tp_embshift_*`; **`n_chunks=1` меньше `require_min_chunks=2`** → **`present=0`**, косинусы **NaN**; whisper; **`emit_extra_metrics=false`** → timing **NaN** |
| `embedding_source_id_extractor` | [embedding_source_id_extractor_audit_v4.md](components/text_processor/embedding_source_id_extractor_audit_v4.md) | **~8.4**/10 | 13 `tp_embid_*`; **`transcript_first`**; **`vector_id`** = sha256(**`transcript_combined_agg_mean.npy`**); nested в **`payload`** |
| `embedding_stats_extractor` | [embedding_stats_extractor_audit_v4.md](components/text_processor/embedding_stats_extractor_audit_v4.md) | **~7.9**/10 | 39 `tp_embstats_*`; **1** чанк &lt; **`min_chunks_required=2`** → **`present=0`**, дисперсия/topvar/**`n_chunks`/`dim`** **NaN**; topic entropy **конечна**; **`emit_extra_metrics=false`** → timing **NaN** |
| `hashtag_embedder` | [hashtag_embedder_audit_v4.md](components/text_processor/hashtag_embedder_audit_v4.md) | **~8.3**/10 | 23 `tp_hashemb_*`; **3** тега → **`hashtag_embedding.npy` (1024,) L2=1**; **CUDA**/fp16; тайминги конечны; **`emit_extra_metrics`** в коде не влияет на выход |
| `lexico_static_features` | [lexico_static_features_audit_v4.md](components/text_processor/lexico_static_features_audit_v4.md) | **~8.2**/10 | 67 `tp_lex_*`; title/desc/transcript **present**; **ASR**; **NaN** `emoji_diversity` (нет эмодзи), **NaN** `named_entity_density` (**enabled=0**); **`load_ms`** всегда **0** |
| `qa_embedding_pairs_extractor` | [qa_embedding_pairs_extractor_audit_v4.md](components/text_processor/qa_embedding_pairs_extractor_audit_v4.md) | **~7.8**/10 | 34 `tp_qa_*`; **`present=0`**, **`num_questions=0`** — нет `?`-сегментов с вопросительным словом; **`qa_question_embeddings.npy`** нет; **NaN** dim/extra при **`emit_extra_metrics=false`** |
| `semantic_cluster_extractor` | [semantic_cluster_extractor_audit_v4.md](components/text_processor/semantic_cluster_extractor_audit_v4.md) | **~8.3**/10 | 31 `tp_semclust_*`; **`present=1`**, **cluster id 25**, similarity **≈0.79**; **`use_faiss`** vs **`backend_faiss=0`** (numpy path); extra-поля **NaN** при **`emit_extra_metrics=false`** |
| `semantics_topics_keyphrases` | [semantics_topics_keyphrases_audit_v4.md](components/text_processor/semantics_topics_keyphrases_audit_v4.md) | **~8.2**/10 | 116 `tp_topics_*`; topics+**10** KPE **(10,1024)**; **`export_keyphrases_mode_none`** → **`kp_top*`** **NaN** в таблице; слоты тем **>5** **NaN**; **`emit_extra_metrics=false`** → **`tp_topics_extra_*`** **NaN** |
| `speaker_turn_embeddings_aggregator` | [speaker_turn_embeddings_aggregator_audit_v4.md](components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md) | **~7.7**/10 | 17 `tp_spkemb_*`; **`present=0`** — нет **diar+тайминг ASR** и нет **legacy speakers**; **`.npy`** нет; **`emit_extra_metrics=false`** → пять config-полей **NaN** |
| `tags_extractor` | [tags_extractor_audit_v4.md](components/text_processor/tags_extractor_audit_v4.md) | **~8.2**/10 | **43** `tp_tags_*` (**28** базовых + слоты **`top1..5`**); **3** уникальных тега; **`export_hashtags_mode_none`**; пустые слоты → **NaN** hash/len |
| `title_embedder` | [title_embedder_audit_v4.md](components/text_processor/title_embedder_audit_v4.md) | **~8.2**/10 | **16** `tp_titleemb_*`; **`title_embedding.npy` (1024,) L2=1**; **CUDA**, **`fp16=0`**; кеш **off**; **`emit_extra_metrics`** не влияет на таблицу |
| `title_embedding_cluster_entropy_extractor` | [title_embedding_cluster_entropy_extractor_audit_v4.md](components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md) | **~8.2**/10 | **24** `tp_titleclent_*`; **`present=1`**; **`use_faiss=1`** / **`backend_faiss=0`** (numpy); **`emit_extra_metrics=false`** → **5× NaN** (dims, margin, ms) |
| `title_to_hashtag_cosine_extractor` | [title_to_hashtag_cosine_extractor_audit_v4.md](components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md) | **~8.3**/10 | **11** `tp_titlehashcos_*`; **`present=1`**, cosine **≈0.847**; title+hashtag **emb** из **`tp_artifacts`**; флаги **0** |
| `topk_similar_titles_extractor` | [topk_similar_titles_extractor_audit_v4.md](components/text_processor/topk_similar_titles_extractor_audit_v4.md) | **~8.2**/10 | **29** `tp_topktitles_*`; **`present=1`**, **K=5**, corpus **≈18k**, **D=1024**; **FAISS off** → **numpy** top‑K; **top1≈0.93** |
| `transcript_chunk_embedder` | [transcript_chunk_embedder_audit_v4.md](components/text_processor/transcript_chunk_embedder_audit_v4.md) | **~8.2**/10 | **16** `tp_tchunk_*`; **whisper** **1** чанк **(1,1024)**; **youtube_auto** **0**; **`emit_extra_metrics=false`** → **5× NaN** |
| `transcript_aggregator` | [transcript_aggregator_audit_v4.md](components/text_processor/transcript_aggregator_audit_v4.md) | **~8.2**/10 | **19** `tp_tragg_*`; **whisper+combined** **present**, **youtube_auto** **0**; **`emit_extra_metrics=false`** → **9× NaN**; **agg `.npy`** есть |

**Сводная оценка по волне:** среднее арифметическое по **22** вердиктам в таблице **≈8.15 / 10** (до одного десятичного **~8.2 / 10**; в тексте ниже для краткости — **~8.1–8.2** как «высокий L1-draft»).

## Итоговая оценка TextProcessor (L1, набор **A**)

**Что сделано.** Зафиксированы отчёты **L1 (draft)** для **22** экстракторов/срезов в [`components/text_processor/`](components/text_processor/): все строки таблицы выше имеют отдельный `*_audit_v4.md`. Контракты проверены на одном e2e reference (**набор A**): табличный слой — **`text_features.npz`** (**`feature_names`** / **`feature_values`**), плюс связанные **`_artifacts/*.npy`** и **`_render/*.html`** где применимо.

**Сводная метрика по вердиктам.** Минимум **~7.7** (`speaker_turn_embeddings_aggregator`), максимум **~8.4** (`embedding_pair_topk_extractor`, `embedding_source_id_extractor`). Типичный коридор **7.9–8.3**: большинство компонентов получают **~8.2** за чёткое соответствие схема ↔ NPZ ↔ код на happy-path **A**.

**Сильные стороны подсистемы (по волне).**

- **Схемы и стабильные префиксы** (`tp_*`): почти везде жёсткие списки ключей и понятные флаги `*_present` / ошибок.
- **Цепочка эмбеддингов** на **A** согласована: title / description / hashtag / чанки транскрипта / агрегаты транскрипта / корпусный top‑K читаются из одного run.
- **Документированность L1**: каждый отчёт привязан к плану (**§4.x**), отдельно отмечены ожидаемые **NaN** (часто из‑за **`emit_extra_metrics=false`** или веток «мало чанков»).

**Риски и ограничения (общие для TextProcessor на **A**).**

- Один reference **не раскрывает** хвосты: много компонентов на **A** видят **один чанк** транскрипта или **`present=0`** по опциональным веткам — нужны **B/C** из списка «Следующие шаги».
- **`emit_extra_metrics` / `compute_std`:** в ряде прогонов диагностические поля в NPZ остаются **NaN**; это ожидаемо, но усложняет отладку без второго прогона. Для **title/description/hashtag embedder** флаг `emit_extra_metrics` в v1.2.0 **не** меняет `features_flat` (см. обновлённые README / SCHEMA); для **transcript_aggregator** и др. с реальным гейтингом — включайте флаг в профиле прогона.
- **FAISS:** где в конфиге «хотим faiss», на **A** часто **`backend_faiss=0`** (модуль недоступен) — numpy/fallback должен явно учитываться в интерпретации метрик. Сводка: [`TextProcessor/docs/FAISS_AND_NUMPY_BACKEND.md`](../../TextProcessor/docs/FAISS_AND_NUMPY_BACKEND.md).

**Вывод.** TextProcessor по Audit v4 (**L1**, набор **A**) выглядит **зрелым на уровне контрактов и e2e-склейки**: средняя оценка волны **~8.15/10** отражает сильное соответствие данных кодам при признании, что **golden (§4.8)** и **B/C** ещё не закрыты и смещают оценку с «draft» к «подкреплёно данными».

## Структура артефактов

- Агрегированный NPZ: `…/text_processor/text_features.npz` (`text_npz_v1`).
- HTML QA: `…/text_processor/_render/<extractor>_report.html`.

## Следующие шаги

1. L1-отчёты для остальных экстракторов (срезы по префиксам в `text_features.npz`, плюс связанные `_artifacts/` где есть).  
2. Наборы **B/C** и **§4.8** для уже закрытых L1-компонентов.  
3. При пилотах с token-only ASR — зафиксировать **shared_tokenizer_v1** в метаданных или отчёте.  
4. Для **`comments_aggregator`** — эмпирика без эмбеддингов, с весами, с `emit_extra_metrics=true` и `compute_std=true`.  
5. Для **`comments_embedder`** — прогон с **`emit_extra_metrics=true`** (заполнение `artifact_written`, таймингов, digest) и **B/C** (нет комментариев, batch path).  
6. Для **`cosine_metrics_extractor`** — **`comments_mode=matrix`**, **`emit_extra_metrics=true`**, альтернативные источники транскрипта и кейсы **require_*** / dim mismatch.  
7. Для **`description_embedder`** — длинное описание (**N_chunks>1**), пустое описание, **cache hit**, убрать или подключить **`emit_extra_metrics`**.  
8. Для **`embedding_pair_topk_extractor`** — многочисленные чанки (**top_k_slots** заполнены), **`emit_extra_metrics=true`**, FAISS/**`use_faiss_mode`**, legacy `transcript_chunks`.  
9. Для **`embedding_shift_indicator_extractor`** — run с **≥2** чанками (или **`require_min_chunks=1`**), **`emit_extra_metrics=true`**, **`compute_extra_cosines=true`**, кейсы **`require_transcript_chunks`**.  
10. Для **`embedding_source_id_extractor`** — политики **`title_first`** / **`strict_missing_primary`**, пустой реестр, согласование **render summary** vs **`payload.embedding_source_id`**.  
11. Для **`embedding_stats_extractor`** — run с **≥2** чанками (или **`min_chunks_required=1`**), **`emit_extra_metrics=true`**, кейсы без topic / invalid **`topic_probs`**, **`require_chunks=true`** при недостатке чанков.  
12. Для **`hashtag_embedder`** — пустые хештеги / **`require_hashtags`**, **`extract_batch`**, **`use_frequencies`**, **`aggregation` ≠ mean**, disk **cache hit**, убрать или подключить **`emit_extra_metrics`** в коде.  
13. Для **`lexico_static_features`** — legacy transcript, **`require_transcript`**, выключенные группы, длинные тексты с усечением, run с эмодзи (**`emoji_diversity`** конечен), отдельный NER-extractor vs заглушка **`named_entity_density`**.  
14. Для **`qa_embedding_pairs_extractor`** — контент с вопросительными конструкциями (**≥1** вопрос), **`emit_extra_metrics=true`**, **`require_min_questions`**, optional **`qa_question_hashes`/`source_ids`**, альтернативные **`question_langs`/слова**.  
15. Для **`semantic_cluster_extractor`** — **`emit_extra_metrics=true`**, прогон с **`backend_faiss=1`**, fallback по слотам, **`require_faiss`**, dim mismatch / отсутствие эмбеддингов.  
16. Для **`semantics_topics_keyphrases`** — **`export_keyphrases_mode=hashed`**, **`emit_extra_metrics=true`**, **`raw`** в payload, выключенные ветки (keyphrases/topics), пустой текст.  
17. Для **`speaker_turn_embeddings_aggregator`** — run с **diarization + ASR** (перекрытие по времени), **legacy `doc.speakers`**, **`emit_extra_metrics=true`**, **`require_input=true`** при отсутствии входа.  
18. Для **`tags_extractor`** — **`export_hashtags_mode` raw/hashed**, **`merge_json_hashtags`**, усечения/parse cap, пустой title и **`require_title`**, **`hashtags_disabled_by_policy`**, **`top_k_slots` > 5.  
19. Для **`title_embedder`** — пустой title (стабильные **NaN**), **`require_title`**, **`compute_embedding=false`**, **CPU** vs **CUDA**, дисковый **cache hit**, **`extract_batch`**, убрать или подключить **`emit_extra_metrics`**.  
20. Для **`title_embedding_cluster_entropy_extractor`** — **`emit_extra_metrics=true`**, **`export_topk_distribution`**, **FAISS** установлен vs fallback, **`top_k_slots`** &gt; **8** (clamp), отсутствие **`relpath`**, **`dim_mismatch`**, **`require_title_embedding`** / **`require_faiss`**.  
21. Для **`title_to_hashtag_cosine_extractor`** — только title или только hashtag (**NaN** cosine), **`unsafe` relpath**, **`dim_mismatch`**, **пустой / битый** `.npy` (**zero_norm**), **`require_title_embedding`** / **`require_hashtag_embedding`**.  
22. Для **`topk_similar_titles_extractor`** — **`enabled=false`**, режимы **`export_topk_mode`**, усечение **`max_export_k`**, корпус &gt; **`max_corpus_for_numpy`** без FAISS, **`require_faiss`** / **`allow_numpy_large_corpus`**, кеш индекса, сравнение **HNSW** vs **numpy**.  
23. Для **`transcript_chunk_embedder`** — **`youtube_auto`**, несколько чанков / overlap, **`emit_extra_metrics=true`**, disk **cache hit**, **`require_asr`**, **`emit_confidence_metrics=false`**, GPU/**fp16** vs CPU.  
24. Для **`transcript_aggregator`** — оба источника (**youtube_auto** + **whisper**), **`emit_extra_metrics=true`**, **`compute_std=true`**, **`require_chunks`**, legacy **`transcript_chunks`**, выключение **mean/max/combined** / **`write_artifacts`**.
