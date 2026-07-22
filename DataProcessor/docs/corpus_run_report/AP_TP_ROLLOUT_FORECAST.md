# Фаза 6 — Прогноз AudioProcessor (21) + TextProcessor (22)

AP и TP **не участвовали** в 300-видео прогоне (отдельные venv, не подключены к `run_corpus.sh`). Это
прогноз по коду + портфолио; реальные числа — после смоука (5-10 видео). Оценки помечены как прогноз.

## Инфраструктура
- **venv:** `AudioProcessor/.ap_venv` = **7.4 ГБ**, `TextProcessor/.tp_venv` = **6.8 ГБ** — оба тяжёлые →
  **тот же import-tax**, что у VP (импорт с сетевого FS ~десятки секунд/subprocess). Системный рычаг
  «venv на локальный SSD» (Фаза 4, измерено −31с/импорт) применим и здесь.
- **Запуск:** AP — `run_cli.py --extractors <список>` (конфиг-driven, `run_extractors`); TP — `run_cli.py`.
  `run_corpus.sh` нужно обобщить, чтобы дергать AP/TP с их venv (не visual venv).

## AudioProcessor (21 экстрактор)
- **tier-0 (4):** `clap_extractor` (CLAP-эмбеддинг, тяжёлая модель), `tempo_extractor`, `loudness_extractor`,
  `asr_extractor` (Whisper/ASR — тяжёлая, ключ для TP). Остальные 17 — производные (mel/mfcc/spectral/chroma/
  pitch/onset/rhythmic/hpss/key/…), в основном librosa/numpy на CPU + несколько torch-моделей.
- **Параллелизуемость:** большинство экстракторов независимы по входу (аудио-сегменты от Segmenter) →
  параллелизуемы внутри видео. Тяжёлые GPU-модели: `clap`, `asr`, `source_separation`, `emotion_diarization`
  (WavLM), `speaker_diarization` (pyannote) — конкурируют за GPU/CPU.
- **Известные блокеры:**
  1. **`voice_quality_extractor` config-drift** (портфолио §3.6): дефолт деплоя `yin + 22050 Hz`, а штамп
     валидации был `yin + 16000` → на реальных данных RuntimeError «No valid f0 values» (24/29 NaN в прогоне).
     **Починить ПЕРЕД массовым прогоном** (иначе 100% фейл компонента, как чуть не вышло с video_pacing).
     Проверить в смоуке: реально ли yin+22050 падает, и какой sr даёт валидный f0.
  2. **`emotion_diarization_extractor`** требует WavLM HF-кэш (`scripts/prepare_hf_cache.sh`, см.
     `AudioProcessor/docs/TESTING_GUIDE.md`) — включить подготовку кэша в setup-скрипт свежего пода
     (не полагаться, что кэш «просто есть»).
  3. **`speaker_diarization`** — pyannote требует `config.yaml` модели (портфолио §заметки: на volume были
     пустые поддиректории → scp модели). Проверить наличие на volume перед прогоном.
  4. **`source_separation_extractor`** — портфолио §4: ложный silence-гейт + неправдоподобная доля вокала;
     алгоритмически под вопросом (но технически запускается).

## TextProcessor (22 экстрактора)
- **tier-0 (3):** `tags_extractor`, `lexico_static_features`, `asr_text_proxy_audio_features`.
- **tier-1 embeddings (6):** title/description/hashtag/transcript_chunk/comments embedders + speaker_turn.
- **tier-2 (5) / tier-3 (8):** производные (cosine/topk/cluster/shift/…) поверх эмбеддингов.
### 🟢 Дизайн AP→TP ASR-моста (разобран в коде — конкретный план сборки)
Формат согласован, мост НЕ требует глубокого рефактора:
- **asr_extractor** (Whisper, inprocess) выдаёт **token-IDs, не текст** (приватность/размер): NPZ содержит
  `token_ids_by_segment` + `segment_start_sec` + `segment_end_sec` (+ timings).
- **TP `asr_text_proxy`** уже умеет читать `doc.asr` и **декодить токены транзитно** через tokenizer
  (`tok.decode(ids, skip_special_tokens=True)`) — token-only путь уже реализован.
- **VideoDocument.asr** ожидает dict: `{"schema_version":.., "segments":[{text,confidence,start_sec,end_sec}]}`
  ИЛИ token-only: `{"token_ids_by_segment":.., "segment_start_sec":.., "segment_end_sec":..}`.
- **Мост (что построить):** после AP asr_extractor → прочитать его NPZ → собрать `doc.asr` dict (token-only
  форма) → создать `VideoDocument(asr=..., title=.., description=..)` из метаданных видео → передать в TP
  `MainProcessor.run(document)`. TP сам декодит токены. Это скрипт-мост (~маппинг NPZ→dict), не рефактор.
- **Что нужно для сборки+валидации:** полный AP-пайплайн (`.ap_venv` + Whisper-модель + аудио от Segmenter)
  должен отработать asr_extractor до TP. Т.е. интеграция AP+TP в `run_corpus` — это и есть основная работа
  «все компоненты» (VP уже работает; AP/TP — следующий крупный блок).

- **ГЛАВНЫЙ БЛОКЕР (исходный) — ASR-проброс `doc.asr` НЕ подключён в оркестраторе** (портфолио §3.3, подтверждено в коде:
  `speaker_turn_embeddings_aggregator/main.py:227` — «doc.asr … NOT yet wired in orchestrator»). TP не имеет
  доступа к аудио сам — берёт ASR из `doc.asr`, который наполняет AudioProcessor. Пока связка AP→doc.asr→TP
  не выстроена, пострадают: `asr_text_proxy_audio_features`, `lexico_static_features` (транскрипт-часть),
  `qa_embedding_pairs_extractor`, `semantics_topics_keyphrases`, `speaker_turn_embeddings_aggregator`,
  `transcript_chunk_embedder`, `transcript_aggregator` — систематически пустые поля (как было с ocr_extractor).
  **Выстроить AP→TP ASR-передачу ПЕРЕД прогоном TP на реальных видео.**
- **Мок-данные:** title/description/hashtag эмбеддинги в валидации были на 1-3 фикстурах → каскад мёртвых
  производных (портфолио §3.2). На реальном корпусе (мои 300 видео + реальные метаданные из pre_final_data,
  которые уже использует Agent B) это оживёт — но нужно передать реальные title/description в TP-документ.

## Зависимости и порядок (важно для параллелизма)
- **Кросс-процессорная последовательность:** TP (transcript/asr/speaker) **ждёт ASR от AP** → AP должен
  отработать раньше TP на том же видео. Внутри AP — параллельно; внутри TP — tier-1 → tier-2/3 (эмбеддинги
  раньше производных). То есть на видео: `AP (parallel extractors) → TP tier-1 → TP tier-2/3`.
- Это влияет на пайплайн-граф: нельзя гнать TP параллельно с AP на одном видео (зависимость по ASR).

## Прогноз диска (доп. МБ/видео)
- AP: mel/spectrogram-выходы тяжёлые; clap/asr эмбеддинги средние; скаляры дёшевы. Грубо **+30-80 МБ/видео**.
- TP: эмбеддинги (title/desc/hashtag/transcript/comments) — векторы, средние; производные дёшевы. **+15-40 МБ/видео**.
- Полный VP+AP+TP: ~**400-550 МБ/видео** → 300 видео ≈ 120-165 ГБ (**превышает том 120ГБ** → обязательна
  выгрузка в HF по ходу, см. Фаза 7 / disk-прогноз).

## Рекомендация Фазы 6
1. **Перед прогоном AP:** починить `voice_quality` config (смоук-проверка sr/f0), подготовить WavLM-кэш,
   проверить pyannote config на volume.
2. **Перед прогоном TP:** выстроить **AP→doc.asr→TP** передачу ASR (иначе 7 TP-компонентов пустые) + передать
   реальные title/description в TP-документ.
3. **Смоук 5-10 видео** на подмножестве AP+TP → реальные числа времени/памяти/диска, потом точечные доработки.
4. Учесть кросс-процессорную зависимость (TP после AP) при проектировании параллелизма (Фаза 4/7).

*Все числа — прогноз до смоука. Смоук требует поднятия AP/TP venv на поде (отдельно от visual venv).*
