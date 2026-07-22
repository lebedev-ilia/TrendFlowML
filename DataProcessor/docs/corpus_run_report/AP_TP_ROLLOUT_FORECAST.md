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
### ✅ AP→TP ASR-мост УЖЕ реализован и подключён в оркестраторе (поправка 2026-07-22)
**Ранее (ниже, зачёркнуто) я записал это как «главный блокер — не подключено». Это было НЕВЕРНО** —
опиралось на устаревший комментарий в компоненте `speaker_turn_embeddings_aggregator/main.py:227`
(«doc.asr … NOT yet wired in orchestrator»), который относится к самому компоненту, а НЕ к top-level
оркестратору. Прочитал реальный оркестратор — мост есть и подключён:
- **`DataProcessor/main.py` → `_autogen_text_input_from_asr()` (строки 1110–1234)**: при `--run-text`, если
  явный text-input не задан, читает `asr_extractor_features.npz`, извлекает `token_ids_by_segment` +
  `segment_start_sec/end_sec` (+ lang/quality-мета), берёт `audio_duration_sec` из Segmenter
  `audio/segments.json`, собирает privacy-safe `doc.asr` (`schema_version=asr_payload_v2`, token-only, БЕЗ
  сырого текста) и пишет `_tmp/text_input_autogen.json`. Подключено в поток `--run-text` (строки 1237–1240).
- **TP-сторона** (`asr_text_proxy_audio_features/main.py::_extract_asr_payload`) уже читает этот `doc.asr`:
  token-only ветка декодит через `shared_tokenizer_v1` (`Tokenizer.from_file`, `tok.decode(ids,
  skip_special_tokens=True)`), транзитно, без персиста текста.
- Ключи, которые читает мост, **побайтово совпадают** с тем, что пишет `AudioProcessor/src/core/npz_savers/
  asr.py` (token_ids_by_segment, segment_start_sec/end_sec/center_sec, lang_code/conf_by_segment,
  segment_quality_by_segment, asr_text_contract_version). `shared_tokenizer_v1` лежит локально
  (`dp_models/bundled_models/text/shared_tokenizer_v1`).
- **Мост в первом коммите** (`git log -S _autogen_text_input_from_asr` → `4c45b91 first commit`) — то есть
  существовал всё это время; «Строй» = не писать заново, а **валидировать end-to-end**.

**Что реально осталось (не «построить мост», а «проверить связку на реальном видео»):**
1. Поднять `.ap_venv` (asr_extractor/Whisper) + `.tp_venv` (shared_tokenizer) на поде — сейчас на поде только
   VP+Triton. Это основная инфра-работа блока «все компоненты».
2. Запустить DP с `--run-audio --run-text --asr-enable-token-sequences` на 5–10 видео → убедиться, что
   autogen-JSON создаётся, `doc.asr.token_ids_by_segment` непустой, TP `asr_text_proxy` даёт
   `tp_asrproxy_present=1` + непустые фичи.
3. Проверить, что реальные `title/description` (из Fetcher/pre_final_data) попадают в TP-документ (сейчас
   autogen кладёт `title=""/description=""` — для asr_text_proxy не критично, но для title/desc-эмбеддеров
   нужно передать реальные метаданные, не пустые).

~~**ГЛАВНЫЙ БЛОКЕР (исходный, ОШИБКА) — ASR-проброс `doc.asr` НЕ подключён в оркестраторе**~~ (портфолио §3.3
цитировал `speaker_turn_.../main.py:227` — комментарий устарел; оркестратор подключает мост). TP берёт ASR из
`doc.asr`, который наполняет autogen-мост выше. Компоненты, зависящие от транскрипта (`asr_text_proxy`,
`lexico_static` транскрипт-часть, `qa_embedding_pairs`, `semantics_topics_keyphrases`, `speaker_turn`,
`transcript_chunk_embedder`, `transcript_aggregator`) оживут, как только AP отработает asr_extractor перед TP —
т.е. это вопрос **прогона AP+TP на поде**, а не отсутствующего кода моста.
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
2. **Перед прогоном TP:** мост **AP→doc.asr→TP уже есть** (`_autogen_text_input_from_asr`, см. выше) — не
   строить, а **валидировать** (asr NPZ → autogen-JSON → TP декод). Дополнительно: передать реальные
   title/description в TP-документ (autogen сейчас кладёт пустые — не критично для asr_text_proxy, критично
   для title/desc-эмбеддеров).
3. **Смоук 5-10 видео** на подмножестве AP+TP → реальные числа времени/памяти/диска, потом точечные доработки.
4. Учесть кросс-процессорную зависимость (TP после AP) при проектировании параллелизма (Фаза 4/7).

*Все числа — прогноз до смоука. Смоук требует поднятия AP/TP venv на поде (отдельно от visual venv).*
