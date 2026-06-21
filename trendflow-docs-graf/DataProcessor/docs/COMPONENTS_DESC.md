# Описание компонентов DataProcessor

Старт нормализации для портфолио: `DataProcessor/docs/PORTFOLIO_NORMALIZATION_PLAN.md`

Для быстрого поиска - DataProcessor/docs/COMPONENTS_DESC_INDEX.md

## AudioProcessor

### Общее описание

AudioProcessor — процессор аудио модальности, извлекающий аудио признаки из аудио дорожки видео. Сохраняет результаты в per-run `result_store`.

### Структура модулей

**API** (`src/api/`):
- `main.py`: FastAPI приложение для HTTP API
- `endpoints.py`: REST endpoints для обработки аудио

**Core** (`src/core/`):
- `main_processor.py`: MainProcessor — главный координатор extractors, оркестрация обработки
- `base_extractor.py`: Базовый класс для всех extractors
- `audio_utils.py`: Утилиты для работы с аудио (загрузка, ресемплирование, нормализация)
- `segments_loader.py`: Загрузка и валидация `audio/segments.json`
- `dependency_resolver.py`: Разрешение зависимостей между extractors
- `extractor_runner.py`: Запуск extractors с учетом зависимостей
- `batch_processor.py`: Батчевая обработка нескольких видео
- `npz_saver.py`: Сохранение результатов в NPZ формат
- `model_resolver.py`: Разрешение метаданных моделей через ModelManager
- `renderer.py`: Генерация render-context для визуализации
- `cli_args.py`: Парсинг аргументов командной строки
- `config_hash.py`: Создание config_hash для идемпотентности

**Schemas** (`src/schemas/`):
- `models.py`: Pydantic модели для валидации данных

### Segment Policy

AudioProcessor использует **Segmenter contract** (`audio_segments_v1`):
- Входные данные: `audio/audio.wav` и `audio/segments.json` от Segmenter
- Сегменты определяются через **families** в `segments.json`:
  - `primary`: короткие окна вокруг time-anchors
  - `clap`: короткие окна на нелинейной кривой
  - `asr`: длинные окна для ASR (10-30 секунд)
  - `tempo`: длинные sliding windows
  - `spectral`: сегменты для спектральных extractors (в т.ч. band_energy/pitch/spectral_entropy — shared family)
  - `chroma`: сегменты для хрома-фич
  - И другие families для специфичных extractors
- Segmenter является единственным владельцем sampling — extractors не генерируют сегменты сами
- Параметры нелинейной кривой (window_sec, min/max windows) определяются Segmenter и передаются через `segments.json`

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет `audio/audio.wav` и `audio/segments.json`
- **ModelManager** (`dp_models`): управление моделями (Whisper, CLAP и др.), строго локальная загрузка без сети
- **TextProcessor**: может использовать результаты ASR (token IDs) для дальнейшей обработки текста
- **Embedding Service**: может использовать CLAP embeddings для семантического поиска

---

## asr_extractor

### Краткое описание

Извлекает транскрипцию речи через Whisper ASR (inprocess, без сети). Выход — token IDs из shared tokenizer (для downstream TextProcessor), опционально raw text по сегментам.

**Версия**: 2.2.0  
**Категория**: speech  
**GPU**: preferred (может работать на CPU, но медленнее)

### Извлекаемые фичи

**Основные фичи** (feature-gated):
- `token_ids_by_segment`: списки token IDs для каждого сегмента (если `enable_token_sequences=True`)
- `lang_id_by_segment`: языковые ID для каждого сегмента
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: временные границы сегментов

**Агрегаты и статистики** (feature-gated):
- `token_counts`: количество токенов по сегментам (если `enable_token_counts=True`)
- `token_total`: общее количество токенов (если `enable_token_total=True`)
- `token_density_per_sec`: средняя плотность токенов (если `enable_token_density=True`)
- `speech_rate_wpm`: слова в минуту (если `enable_speech_rate=True`)
- `lang_distribution`: распределение языков (если `enable_lang_distribution=True`)
- `segments_with_speech`: количество сегментов с речью (если `enable_segments_with_speech=True`)
- `avg_segment_duration_sec`: средняя длительность сегмента (если `enable_avg_segment_duration=True`)
- `token_variance`: статистическая дисперсия token counts (если `enable_token_variance=True`)

**Зависимости между фичами**:
- `token_total` зависит от `token_counts` (сумма всех counts)
- `token_density_per_sec` зависит от `token_total` и `segment_duration`
- `speech_rate_wpm` зависит от `token_total` (приблизительный расчёт: tokens / 1.3 / duration_min)
- `token_variance` зависит от `token_counts`
- `segments_with_speech` зависит от `token_ids_by_segment`

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- **TextProcessor**: декодирует token IDs в текст через shared tokenizer (`shared_tokenizer_v1`)
- **speech_analysis_extractor**: может использовать результаты ASR для анализа речи

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка Whisper моделей (`whisper_small_inprocess`, `whisper_medium_inprocess`, `whisper_large_inprocess`) и shared tokenizer (`shared_tokenizer_v1`)
- **TextProcessor**: использует token IDs для декодирования в текст
- **Segmenter**: предоставляет ASR windows (`families.asr`) — длинные окна (10-30 секунд)

### Segment Policy

- Использует family `asr` из `audio/segments.json`
- Сегменты — длинные окна (обычно 10-30 секунд) для устойчивой транскрипции
- Параметры сохраняются в `audio/segments.json` (Segmenter contract)
- Если `segments` пустой → error (no-fallback policy)

---

## band_energy_extractor

### Краткое описание

Извлекает энергии по частотным полосам (низ/середина/высокие) и их доли. Поддерживает фиксированные полосы или мел-шкалу, опционально возвращает временные ряды (per-frame энергии).

**Версия**: 2.0.2  
**Категория**: spectral  
**GPU**: не требуется

### Извлекаемые фичи

**Основной результат** (всегда включен):
- `band_edges`: границы полос `[(lo, hi), ...]` в Hz
- `band_energies`: суммарные энергии по полосам
- `band_energy_shares`: доли энергии по полосам (нормализованные, сумма = 1.0)
- `total_energy`: общая энергия сигнала

**Базовые статистики** (feature-gated):
- `band_energy_mean`, `band_energy_std`, `band_energy_median` (если `enable_basic_stats=True`)

**Расширенные статистики** (feature-gated):
- `band_energy_min`, `band_energy_max`, `band_energy_p25`, `band_energy_p75` (если `enable_extended_stats=True`)

**Временные серии** (feature-gated):
- `band_energy_ts`: временные ряды энергий по полосам (если `enable_time_series=True`)

**Метрики баланса** (feature-gated):
- `band_balance_score`, `band_dominance`, `band_contrast` (если `enable_balance_metrics=True`)

**Метрики динамики** (feature-gated, для `run_segments()`):
- `band_energy_stability`, `band_transitions`, `band_transitions_count`, `band_transitions_rate`, `band_distribution`, `band_diversity` (если `enable_dynamics=True`)

**Зависимости между фичами**:
- `band_energy_stability` зависит от `enable_dynamics` и `enable_time_series`
- `band_transitions` и `band_transitions_count` зависят от `enable_dynamics` и `enable_time_series`
- `band_balance_score` зависит от `enable_balance_metrics` и вычисления `band_energy_shares`

**Upstream зависимости**:
- **spectral_extractor** (опционально): может использовать предвычисленный `stft_magnitude` и `frequencies` из `shared_features` для оптимизации

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.band_energy.segments[]`)
- **librosa**: основная библиотека для STFT и мел-шкалы
- **essentia** (опционально): альтернативный метод обработки

### Segment Policy

- **Audit v3**: uses shared sampling family `spectral` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## chroma_extractor

### Краткое описание

Извлекает хрома-признаки (12-полосный профиль классов высот, pitch class profile) с автоматической оценкой строя, нормализацией и статистическими агрегатами. Хрома-фичи отражают гармоническое содержание аудио.

**Версия**: 2.0.2  
**Категория**: spectral  
**GPU**: не требуется

### Извлекаемые фичи

**Статистические агрегаты** (feature-gated):
- `chroma_mean`, `chroma_std`, `chroma_min`, `chroma_max` (если `enable_basic_stats=True`)
- `chroma_median`, `chroma_p25`, `chroma_p75` (если `enable_extended_stats=True`)
- `chroma_stats_vector`: конкатенированный вектор всех статистик (если `enable_stats_vector=True`)

**Временные ряды** (feature-gated):
- `chroma`: полная хрома-спектрограмма (12 x frames) (если `enable_time_series=True`)

**Дополнительные метрики** (always computed):
- `tuning_estimate`: оценка строя (semitones)
- `chroma_dominant_class`: индекс доминирующего хрома-класса
- `chroma_dominant_energy`: энергия доминирующего класса
- `chroma_harmonic_stability`: стабильность гармонического содержания
- `chroma_entropy`: энтропия распределения хрома-классов
- `chroma_contrast`: контраст между классами
- `chroma_centroid`: центроид распределения
- `chroma_rolloff`: rolloff частоты (95% энергии)

**Зависимости между фичами**:
- `chroma_stats_vector` зависит от `enable_basic_stats` и/или `enable_extended_stats`
- Все статистики зависят от вычисления хрома-спектрограммы
- Дополнительные метрики зависят от вычисления статистик

**Upstream зависимости**:
- Нет зависимостей от других extractors

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.chroma.segments[]`)
- **librosa**: основная библиотека для хрома-фич (CQT или STFT-based) и оценки строя

### Segment Policy

- Использует family `chroma` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## clap_extractor

### Краткое описание

Извлекает семантические аудио эмбеддинги CLAP (Contrastive Language-Audio Pre-training) по Segmenter-окнам. Отдаёт эмбеддинг по каждому окну и агрегат (mean) по всему видео.

**Версия**: 1.1.0 (Audit v3)  
**Категория**: advanced  
**GPU**: preferred (может работать на CPU)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `embedding`: агрегированный эмбеддинг по видео (robust aggregation по валидным сегментам), shape `[512]`
- `embedding_sequence`: эмбеддинги по каждому сегменту (strict-aligned; masked → NaN), shape `[N, 512]`
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: time-axis, shape `[N]`
- `segment_mask`: валидность сегмента (strict alignment), shape `[N]`
- `segments_count`: количество **валидных** сегментов (`sum(segment_mask)`)
- `embedding_dim`: размерность эмбеддинга (512)

**Статистики** (always computed):
- `clap_norm`: норма агрегированного эмбеддинга
- `clap_magnitude_mean`: среднее абсолютное значение эмбеддинга
- `clap_magnitude_std`: стандартное отклонение абсолютных значений
- `clap_non_zero_count`: количество ненулевых компонент

**Зависимости между фичами**:
- `embedding` зависит от `embedding_sequence` (mean по всем сегментам)
- Статистики зависят от `embedding`

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- **Embedding Service**: может использовать CLAP embeddings для семантического поиска
- **ML модели**: могут использовать embeddings как входные признаки

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка CLAP модели (`laion_clap`) строго локально, без сетевых загрузок
- **Segmenter**: предоставляет CLAP windows (`families.clap`) — короткие окна на нелинейной кривой
- **Embedding Service**: может использовать embeddings для семантического поиска и кластеризации

### Segment Policy

- Использует family `clap` из `audio/segments.json`
- Сегменты — короткие окна на универсальной нелинейной кривой:
  - На коротких видео можно близко к 1:1 (секунда → окно)
  - На длинных видео рост замедляется и упирается в `max_windows`
- Параметры кривой (`k/min/max/linear_until/cap_duration`) лежат в `audio/segments.json`
- Ограничение длины окна: модель ограничивает эффективную длину аудио (`max_audio_length=10.0` секунд)
- Если `segments` пустой → error (no-fallback policy)

### Schema (Audit v3)

- `schema_version`: `clap_extractor_npz_v1`
- NPZ keys/tiers/shape фиксируются в `DataProcessor/AudioProcessor/schemas/clap_extractor_npz_v1.json` и `AudioProcessor/src/extractors/clap_extractor/SCHEMA.md`

---

## emotion_diarization_extractor

### Краткое описание

Извлекает эмоциональную диаризацию (распознавание эмоций по временным окнам) через SpeechBrain Speech_Emotion_Diarization модель. Работает с сегментами от Segmenter и возвращает вероятности эмоций для каждого окна, а также агрегированные метрики.

**Версия**: 3.1.0 (Audit v3)  
**Категория**: speech  
**GPU**: preferred (SpeechBrain model, requires GPU memory)

### Извлекаемые фичи

**Обязательные поля** (Audit v3):
- `segments_total`: количество окон от Segmenter
- `segments_count`: количество валидных окон (`sum(segment_mask)`)
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: time-axis (strict-aligned)
- `segment_mask`: маска валидных окон (strict-aligned)
- `emotion_id`: доминирующая эмоция по окнам (invalid → `-1`)
- `emotion_confidence`: confidence по окнам (invalid → `NaN`)
- `emotion_labels`: список названий эмоций

**Model-facing агрегаты (Audit v3, frozen subset)**:
- `emotion_entropy`
- `dominant_emotion_id`, `dominant_emotion_prob`
- `emotion_transitions_count`, `emotion_stability_score`, `emotion_diversity_score`

**Опционально (feature-gated)**:
- `emotion_probs: [N,C]` (heavy analytics)
- `emotion_mean_probs: [C]`
- `emotion_distribution` / `emotion_segments_per_emotion` / `emotion_duration_per_emotion` (dict-like)
- `emotion_quality_metrics` (dict-like)

**Зависимости между фичами**:
- `dominant` зависит от `ids` или `confidence` (использует `emotion_id` для вычисления transitions, distribution, stability)
- `quality_metrics` зависит от `confidence` и `mean_probs`

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- нет жёстких downstream контрактов (используется как самостоятельный источник фич)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SpeechBrain моделей (`emotion_diarization_small_inprocess`, `emotion_diarization_large_inprocess`) строго локально
- **SpeechBrainProvider** (`dp_models`): провайдер для загрузки SpeechBrain моделей
- **SpeechBrain**: библиотека для загрузки и использования Speech_Emotion_Diarization (локальная копия в `speechbrain/`)
- **Segmenter**: предоставляет сегменты (family: `emotion`)

### Segment Policy

- Использует family `emotion` из `audio/segments.json`
- Audit v3: только `run_segments()` (full-audio mode отключён)
- Минимальная длительность: <5s → `status="empty"`, `empty_reason="audio_too_short"`
- Если `segments` пустой → error (no-fallback policy)
- Тихое аудио → `status="empty"`, `empty_reason="audio_silent"` (если silence detection включен)

### Schema (Audit v3)

- `schema_version`: `emotion_diarization_extractor_npz_v1`
- Human schema: `AudioProcessor/src/extractors/emotion_diarization_extractor/SCHEMA.md`

---

## hpss_extractor

### Краткое описание

Извлекает Harmonic-Percussive Source Separation (HPSS) признаки — разложение аудио на гармоническую и перкуссионную компоненты. Вычисляет доли энергии каждой компоненты, спектральные фичи из разделённых компонент, и опционально сохраняет восстановленные временные сигналы.

**Версия**: 2.0.2  
**Категория**: source_separation  
**GPU**: не требуется

### Извлекаемые фичи

**Энергетические метрики** (feature-gated):
- `hpss_harmonic_share`, `hpss_percussive_share`: доли энергии компонент (0.0-1.0)
- `hpss_energy_total`, `hpss_energy_harmonic`, `hpss_energy_percussive`: энергии компонент
- `hpss_harmonic_stability`, `hpss_percussive_stability`: стабильность компонент
- `hpss_separation_quality`: качество разделения
- `hpss_balance_score`: баланс между компонентами
- `hpss_dominance`: доминирующая компонента ("harmonic", "percussive", "mixed")

**Спектральные фичи** (feature-gated):
- `hpss_harmonic_centroid_mean/std`, `hpss_harmonic_bandwidth_mean/std`, `hpss_harmonic_rolloff_mean/std`: спектральные характеристики гармонической компоненты
- `hpss_percussive_centroid_mean/std`, `hpss_percussive_bandwidth_mean/std`, `hpss_percussive_rolloff_mean/std`: спектральные характеристики перкуссионной компоненты

**Временные сигналы** (feature-gated):
- `hpss_harmonic_npy`, `hpss_percussive_npy`: пути к сохраненным .npy файлам с восстановленными сигналами

**Временные серии** (feature-gated):
- `hpss_harmonic_share_series`, `hpss_percussive_share_series`: временные серии долей энергии

**Зависимости между фичами**:
- Спектральные фичи требуют вычисления HPSS разложения (автоматически выполняется)
- Waveforms требуют вычисления HPSS разложения (автоматически выполняется)
- Временные серии требуют включения энергетических метрик (для вычисления shares по времени)

**Upstream зависимости**:
- Нет зависимостей от других extractors

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (family `hpss`)
- **librosa**: основная библиотека для HPSS разложения (`librosa.decompose.hpss`) и спектральных фичей

### Segment Policy

- Использует family `hpss` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## key_extractor

### Краткое описание

Определяет тональность (ключ) аудио — основной тональный центр и лад (мажор/минор). Использует шаблоны Krumhansl-Schmuckler для корреляции с хрома-профилем. Поддерживает явный выбор метода (Essentia/librosa/auto) и segment-based обработку для отслеживания изменений тональности.

**Версия**: 2.0.2  
**Категория**: music_theory  
**GPU**: не требуется

### Извлекаемые фичи

**Основной результат** (всегда включен):
- `key_name`: название тональности (str, например "C", "C#", "D", ..., "B")
- `key_mode`: лад (str, "major" или "minor")
- `key_confidence`: уверенность в определении (float, 0.0-1.0)
- `key_confidence_category`: категория уверенности ("high" | "medium" | "low" | "very_low")
- `key_low_confidence_warning`: флаг предупреждения о низкой уверенности
- `method`: использованный метод ("essentia" | "librosa")

**Детальные оценки** (feature-gated):
- `key_scores`: оценки для всех 24 возможных тональностей (если `enable_detailed_scores=True`)

**Топ-K альтернативных тональностей** (feature-gated):
- `key_top_k`: топ-K тональностей с наивысшими оценками (если `enable_top_k=True`)

**Временные серии** (feature-gated, для `run_segments()`):
- `key_names_sequence`, `key_modes_sequence`, `key_confidences_sequence`: последовательности по сегментам (если `enable_time_series=True`)

**Детекция смены тональности** (feature-gated, для `run_segments()`):
- `key_transitions`: список переходов между тональностями (если `enable_key_changes=True`)
- `key_transitions_count`, `key_transitions_rate`: количество и частота переходов

**Метрики стабильности** (feature-gated, для `run_segments()`):
- `key_stability_score`: доля времени в доминирующей тональности
- `key_confidence_mean/std/min/max`: статистики уверенности
- `key_distribution`: распределение времени по тональностям
- `key_diversity`: количество уникальных тональностей
- `key_detection_quality`: метрика качества (confidence × stability)

**Зависимости между фичами**:
- `key_top_k` зависит от `enable_top_k` и требует `enable_detailed_scores=True`
- `key_transitions` и `key_transitions_count` зависят от `enable_key_changes` и `enable_time_series`
- Все метрики стабильности зависят от `enable_stability_metrics` и `enable_time_series`
- `key_detection_quality` зависит от `key_stability_score` и `key_confidence`

**Upstream зависимости**:
- **chroma_extractor** (опционально): может использовать предвычисленный `chroma` из `shared_features` для оптимизации

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.key.segments[]`)
- **librosa**: основная библиотека для хрома-фич и оценки тональности (Krumhansl-Schmuckler profiles)
- **essentia** (опционально): альтернативный метод определения тональности (`essentia.standard.KeyExtractor`)
- **chroma_extractor**: может предоставлять `chroma` в `shared_features` для оптимизации

### Segment Policy

- Использует family `key` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Сегменты короче 0.5 секунды пропускаются (слишком короткие для точного определения тональности)
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## loudness_extractor

### Краткое описание

Извлекает метрики громкости и динамики аудио: RMS, peak, dBFS, опционально LUFS (если доступен pyloudnorm), а также frame-wise RMS статистики для анализа краткосрочной динамики.

**Версия**: 1.1.0  
**Категория**: loudness  
**GPU**: не требуется

### Извлекаемые фичи

**Основные метрики** (всегда включены):
- `rms`: RMS значение по всему треку (float)
- `peak`: пиковое значение по всему треку (float)
- `dbfs`: dBFS значение (20*log10(rms + eps)) (float)
- `lufs`: LUFS значение (float, может быть None если pyloudnorm недоступен)
- `lufs_present`: флаг наличия LUFS (bool)

**Frame-wise RMS статистики** (всегда включены):
- `frame_rms_mean`, `frame_rms_std`, `frame_rms_median`, `frame_rms_p10`, `frame_rms_p90`: статистики по frame-wise RMS
- `frame_rms_stats_vector`: вектор статистик [mean, std, median, p10, p90]

**Сегментные метрики** (для `run_segments()`):
- `segment_rms`, `segment_peak`, `segment_dbfs`, `segment_lufs`: метрики по каждому сегменту (float32[N])
- `segment_centers_sec`: центры сегментов в секундах (float32[N])
- `segment_rms_mean`, `segment_rms_std`, `segment_rms_median`, `segment_rms_p10`, `segment_rms_p90`: агрегированные статистики по сегментам

**Зависимости между фичами**:
- `lufs` зависит от наличия `pyloudnorm` (best-effort, не критично)
- Frame-wise статистики зависят от вычисления frame-wise RMS
- Сегментные статистики зависят от сегментных метрик

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.primary.segments[]`)
- **pyloudnorm** (опционально): библиотека для вычисления LUFS (best-effort, не критично)
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `primary` из `audio/segments.json`
- Сегменты — короткие окна вокруг time-anchors
- Параметры сохраняются в `audio/segments.json` (Segmenter contract)
- Если `segments` пустой → error (no-fallback policy)

---

## mel_extractor

### Краткое описание

Извлекает Mel-спектрограмму — частотно-временное представление аудио в мел-шкале (mel scale), которая лучше соответствует восприятию звука человеком. Вычисляет Mel-спектрограмму в децибелах, статистические агрегаты и спектральные характеристики (спектральный центроид и полоса пропускания).

**Версия**: 2.0.1  
**Категория**: spectral  
**GPU**: preferred (использует GPU если доступен, так как дает прирост скорости)

### Извлекаемые фичи

**Базовые фичи** (feature-gated: `enable_basic_features`):
- `mel_spectrogram_npy`: путь к сохраненному .npy файлу с полной Mel-спектрограммой (shape: `(n_mels, frames)`, единицы: децибелы)
- `mel_shape`, `mel_elements`: форма и количество элементов спектрограммы

**Статистики** (feature-gated: `enable_statistics`):
- `mel_mean`, `mel_std`, `mel_min`, `mel_max`: статистики по времени для каждого mel bin (float32[n_mels])
- `freq_mean`, `freq_std`: статистики по частотам для каждого кадра (float32[frames])
- `mel_stats_vector`: конкатенированный вектор статистик (feature-gated: `enable_stats_vector`)

**Спектральные характеристики** (feature-gated: `enable_spectral_features`):
- `spectral_centroid`: спектральный центроид по времени (float32[frames], Hz)
- `spectral_bandwidth`: полоса пропускания по времени (float32[frames], Hz)

**Дополнительные метрики** (всегда включены, если включены basic_features):
- `mel_energy`: общая энергия Mel-спектрограммы
- `mel_centroid_mean/std`, `mel_bandwidth_mean/std`: агрегированные спектральные характеристики
- `mel_spectrogram_entropy`, `mel_spectrogram_contrast`: энтропия и контраст

**Временные серии** (feature-gated: `enable_time_series`):
- `mel_series`: полная временная серия Mel-спектрограммы
- `segment_centers_sec`, `segment_durations_sec`: временные метки сегментов (для `run_segments()`)

**Зависимости между фичами**:
- Статистики зависят от `enable_basic_features` (требуют вычисления Mel-спектрограммы)
- Спектральные характеристики зависят от `enable_basic_features` и `enable_spectral_features`
- `mel_stats_vector` зависит от `enable_statistics` и `enable_stats_vector`
- Дополнительные метрики зависят от `enable_basic_features`

**Upstream зависимости**:
- Нет зависимостей от других extractors

**Downstream зависимости**:
- **mfcc_extractor**: использует Mel-спектрограмму как промежуточный шаг для MFCC

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.mel.segments[]`)
- **torchaudio**: основная библиотека для Mel-спектрограммы (STFT → Mel-фильтры → power/magnitude → dB)
- **torch**: для GPU-ускорения (опционально)

### Segment Policy

- Использует family `mel` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Минимальная длительность сегмента: **100 мс** (для точности Mel-спектрограммы)
- Если family отсутствует → error (no-fallback policy)

---

## mfcc_extractor

### Краткое описание

Извлекает MFCC (Mel-frequency cepstral coefficients) — кепстральные коэффициенты в мел-шкале, широко используемые для анализа речи и музыки. MFCC представляют спектральную форму сигнала в компактном виде и эффективны для распознавания речи, классификации аудио и других задач машинного обучения.

**Версия**: 2.0.1  
**Категория**: spectral  
**GPU**: preferred (может работать на CPU, но GPU ускоряет обработку длинных файлов)

### Извлекаемые фичи

**Базовые фичи** (feature-gated: `enable_basic_features`):
- `mfcc_features`: массив MFCC коэффициентов (shape: `(n_mfcc, frames)`)
- `mfcc_statistics`: словарь со статистиками (`mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max`)

**Дельты** (feature-gated: `enable_deltas`):
- `delta_mean`, `delta_std`: статистики первых дельт (производных по времени)
- `delta_delta_mean`, `delta_delta_std`: статистики вторых дельт
- `delta_series`, `delta_delta_series`: временные серии дельт (feature-gated: `enable_time_series`)

**Дополнительные метрики** (всегда включены, если включены basic_features):
- `mfcc_energy`: энергия первого MFCC коэффициента
- `mfcc_centroid`, `mfcc_bandwidth`: центроид и полоса пропускания MFCC
- `mfcc_skewness`, `mfcc_kurtosis`: асимметрия и эксцесс распределения
- `mfcc_correlation`, `mfcc_stability`: корреляция и стабильность MFCC

**Временные серии** (feature-gated: `enable_time_series`):
- `mfcc_series`: полная временная серия MFCC
- `segment_centers_sec`, `segment_durations_sec`: временные метки сегментов (для `run_segments()`)

**Зависимости между фичами**:
- Статистики зависят от `enable_basic_features` (требуют вычисления MFCC)
- Дельты зависят от `enable_basic_features` и `enable_deltas`
- Временные серии дельт зависят от `enable_time_series` и `enable_deltas`
- Дополнительные метрики зависят от `enable_basic_features`

**Upstream зависимости**:
- Нет зависимостей от других extractors (вычисляет Mel-спектрограмму внутри)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.mfcc.segments[]`)
- **torchaudio**: основная библиотека для MFCC (Mel-спектрограмма → логарифм → DCT)
- **torch**: для GPU-ускорения (опционально)

### Segment Policy

- Использует family `mfcc` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Минимальная длительность сегмента: **100 мс** (для точности MFCC)
- Если family отсутствует → error (no-fallback policy)

---

## onset_extractor

### Краткое описание

Определяет онсеты (атаки звука) — моменты начала новых звуковых событий в аудио сигнале. Онсеты важны для анализа ритма, сегментации музыки, обнаружения ударных инструментов и других задач музыкального анализа.

**Версия**: 2.0.0  
**Категория**: rhythm  
**GPU**: не требуется (CPU-only обработка)

### Извлекаемые фичи

**Базовые фичи** (feature-gated: `enable_basic_features`):
- `onset_times`: массив времен онсетов в секундах (float32[N])
- `onset_count`: количество обнаруженных онсетов (int)
- `onset_density_per_sec`: плотность онсетов (онсетов/сек) (float)
- `insufficient_onsets`: флаг недостаточного количества онсетов (bool)

**Статистики интервалов** (feature-gated: `enable_interval_stats`):
- `avg_interval_sec`, `interval_std`, `interval_min`, `interval_max`, `interval_median`: статистики интервалов между онсетами

**Ритмические метрики** (feature-gated: `enable_rhythmic_metrics`):
- `onset_regularity_score`: регулярность ритма (0-1)
- `onset_clustering_score`: мера кластеризации онсетов (0-1)
- `onset_tempo_estimate`: оценка BPM из интервалов
- `onset_syncopation_score`: мера синкопированности (0-1)
- `onset_strength_mean/std`: статистики силы онсетов
- `onset_density_variance`: вариация плотности онсетов
- `onset_tempo_consistency`: согласованность с tempo_extractor (0-1, если доступен)

**Временные серии** (feature-gated: `enable_time_series`):
- `onset_times`: массив времен онсетов (если `onset_times.size > 10000`, сохраняется в .npy файл)

**Зависимости между фичами**:
- Статистики интервалов зависят от `enable_interval_stats` и `enable_basic_features` (требуют минимум 2 онсета)
- Ритмические метрики зависят от `enable_rhythmic_metrics` и `enable_basic_features`
- `onset_tempo_consistency` зависит от доступности результатов `tempo_extractor`

**Upstream зависимости**:
- **tempo_extractor** (опционально): может использоваться для валидации/улучшения результатов (метрика `onset_tempo_consistency`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.onset.segments[]`)
- **librosa**: основная библиотека для обнаружения онсетов (default backend)
- **essentia** (опционально): более точный алгоритм обнаружения онсетов (если доступен, no-fallback policy)
- **tempo_extractor**: опциональная интеграция для валидации результатов

### Segment Policy

- Использует family `onset` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## pitch_extractor

### Краткое описание

Извлекает основную частоту (f0) из аудио сигнала с использованием нескольких алгоритмов: PYIN, YIN и опционально torchcrepe (CREPE). Автоматически выбирает лучший метод на основе качества оценки и вычисляет статистические метрики высоты тона.

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: optional (torchcrepe может использовать GPU, но не требуется)

### Извлекаемые фичи

**Базовые статистики** (feature-gated: `enable_basic_stats`):
- `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_median`: статистики основной частоты (Hz)
- `f0_method`: выбранный метод (`"pyin"`, `"yin"`, `"torchcrepe"` или `"none"`)
- `pitch_contour_smoothness`: гладкость контура pitch (0-1)
- `pitch_jump_count`: количество больших скачков pitch (>2 semitones)
- `pitch_centroid`, `pitch_skewness`, `pitch_kurtosis`: статистические характеристики распределения pitch
- `pitch_octave_distribution`: распределение pitch по октавам

**Метрики стабильности** (feature-gated: `enable_stability_metrics`):
- `pitch_variation`: вариация высоты тона (std отклонение разностей)
- `pitch_stability`: стабильность высоты тона (0-1, где 1 = стабильная)
- `pitch_range`: диапазон высоты тона (max - min, Hz)

**Delta-признаки** (feature-gated: `enable_delta_features`):
- `f0_delta_mean`, `f0_delta_std`, `f0_delta_abs_mean`: статистики изменений f0 между кадрами

**Статистики по методам** (feature-gated: `enable_method_stats`):
- `f0_mean_pyin`, `f0_std_pyin`, `f0_min_pyin`, `f0_max_pyin`, `f0_median_pyin`: статистики PYIN
- `voiced_fraction_pyin`, `voiced_probability_mean_pyin`: метрики озвученности PYIN
- `f0_mean_yin`, `f0_std_yin`, `f0_min_yin`, `f0_max_yin`, `f0_median_yin`: статистики YIN
- `f0_mean_torchcrepe`, `f0_std_torchcrepe`, `f0_min_torchcrepe`, `f0_max_torchcrepe`, `f0_median_torchcrepe`: статистики torchcrepe (если используется)

**Временные серии** (feature-gated: `enable_time_series`):
- `f0_series_pyin`, `f0_series_yin`, `f0_series_torchcrepe`: временные серии f0 для каждого метода
- `f0_series`: агрегированная временная серия f0 (для `run_segments()`)
- `segment_centers_sec`, `segment_durations_sec`: временные метки сегментов (для `run_segments()`)

**Зависимости между фичами**:
- Метрики стабильности зависят от `enable_basic_stats` (требуют `f0_mean`, `f0_std`, `f0_min`, `f0_max`)
- Delta-признаки зависят от `enable_basic_stats` (требуют временную серию f0)
- Статистики по методам независимы от других фичей
- Временные серии независимы от других фичей

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- **speech_analysis_extractor**: может использовать pitch_extractor для анализа речи

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.pitch.segments[]`)
- **librosa**: библиотека для PYIN и YIN алгоритмов (classic backend)
- **torchcrepe** (опционально): нейросетевая модель для оценки f0 (если выбран как backend, no-fallback policy)
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `pitch` из `audio/segments.json` (опционально, для `run_segments()`)
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует и используется `run_segments()` → error (no-fallback policy)
- Поддерживает режим `run()` для полного аудио (legacy mode)

---

## quality_extractor

### Краткое описание

Извлекает базовые метрики качества аудио для оценки технического состояния записи. Легковесный экстрактор без тяжелых зависимостей, предназначенный для быстрой оценки качества аудио сигнала.

**Версия**: 2.0.0  
**Категория**: quality  
**GPU**: не требуется (CPU-only)

### Извлекаемые фичи

**Базовые метрики** (feature-gated: `enable_basic_metrics`):
- `dc_offset`: среднее смещение постоянной составляющей (DC offset) (нормализованные амплитуды)
- `dc_offset_abs`: абсолютное значение DC offset
- `clipping_ratio`: доля отсечённых сэмплов (clipping) (0.0-1.0)
- `crest_factor_db`: отношение пика к RMS в децибелах (≥ 0 dB)

**Динамические метрики** (feature-gated: `enable_dynamic_metrics`):
- `dynamic_range_db`: динамический диапазон (разница между 95 и 5 перцентилями уровня) (≥ 0 dB)
- `snr_db`: грубая оценка отношения сигнал/шум (≥ 0 dB)

**Анализ кадров** (feature-gated: `enable_frame_analysis`):
- `frame_levels_distribution`: распределение уровней кадров (mean, std, min, max, median)

**Дополнительные метрики** (всегда включены, если включены basic_metrics):
- `clipping_segments_count`: количество сегментов с клиппингом (для `run_segments()`)
- `crest_factor_median`: медиана crest factor по кадрам (для `run_segments()`)
- `dynamic_range_stability`: стабильность dynamic range (0-1)
- `snr_stability`: стабильность SNR (0-1)
- `quality_score`: композитная оценка качества на основе всех метрик (0.0-1.0)

**Временные серии** (feature-gated: `enable_time_series`):
- `frame_levels_db_series`, `frame_rms_series`: временные серии уровней и RMS по кадрам
- `clipping_segments_series`, `dc_offset_series`: временные серии клиппинга и DC offset
- `clipping_ratio_series`, `crest_factor_db_series`: временные серии clipping ratio и crest factor
- `dynamic_range_db_series`, `snr_db_series`: временные серии dynamic range и SNR
- `segment_centers_sec`, `segment_durations_sec`: временные метки сегментов (для `run_segments()`)

**Зависимости между фичами**:
- `dc_offset_abs` зависит от `enable_basic_metrics` (требует `dc_offset`)
- `dynamic_range_stability` и `snr_stability` зависят от `enable_dynamic_metrics`
- `quality_score` зависит от всех базовых и динамических метрик
- `frame_levels_distribution` зависит от `enable_frame_analysis`
- Временные серии клиппинга зависят от `enable_basic_metrics` и `enable_frame_analysis`

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.quality.segments[]`)
- **numpy**: все вычисления выполняются через numpy (векторизованные операции)
- **AudioUtils**: загрузка и предобработка аудио (нормализация, конвертация форматов)

### Segment Policy

- Использует family `quality` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Минимальная длительность сегмента: **50 мс** (для точности метрик)
- Если family отсутствует → error (no-fallback policy)

---

## rhythmic_extractor

### Краткое описание

Извлекает ритмические метрики из аудио сигнала: beat tracking (отслеживание битов), регулярность ритма, плотность ударов, статистику интервалов между ударами и дополнительные ML/analytics метрики. Использует librosa или Essentia для beat tracking (явный выбор backend, no-fallback policy).

**Версия**: 2.0.0  
**Категория**: rhythm  
**GPU**: не требуется (CPU-only обработка)

### Извлекаемые фичи

**Базовые метрики** (feature-gated: `enable_basic_metrics`):
- `rhythm_tempo_bpm`: темп в ударах в минуту (BPM) (float, типичные значения: 60-180)
- `rhythm_beats_count`: количество обнаруженных битов (int)
- `rhythm_beat_density`: плотность ударов (количество ударов в секунду) (float)

**Статистики интервалов** (feature-gated: `enable_interval_stats`):
- `rhythm_avg_period_sec`, `rhythm_period_std_sec`, `rhythm_median_period_sec`: статистики периодов между ударами
- `rhythm_min_period_sec`, `rhythm_max_period_sec`: минимальный и максимальный периоды

**Метрики регулярности** (feature-gated: `enable_regularity_metrics`):
- `rhythm_regularity`: коэффициент регулярности ритма (0-1, где 1 = идеально регулярный)
- `rhythm_syncopation_score`: мера синкопированности
- `rhythm_polyrhythm_score`: мера полиритмичности
- `rhythm_beat_strength_mean/std`: статистики силы ударов
- `rhythm_metrical_stability`: метрическая стабильность

**Метрики темпа** (feature-gated: `enable_tempo_metrics`):
- `rhythm_median_bpm`: медианный темп, вычисленный из медианного периода (BPM)
- `rhythm_tempo_variation`: вариация темпа (коэффициент вариации интервалов)
- `rhythm_beat_consistency`: консистентность ударов (0-1, где 1 = идеально консистентный)
- `rhythm_tempo_mean/std/min/max`: статистики темпа по сегментам (для `run_segments()`)

**Временные метки ударов** (feature-gated: `enable_beat_times`):
- `beat_times`: массив временных меток ударов в секундах (float32[N])
- `segment_beat_times`: список массивов временных меток ударов для каждого сегмента (для `run_segments()`)

**Временные метки сегментов** (для `run_segments()`):
- `segment_centers_sec`, `segment_durations_sec`: центры и длительности сегментов

**Зависимости между фичами**:
- Статистики интервалов зависят от `enable_interval_stats` и `enable_basic_metrics` (требуют минимум 2 удара)
- Метрики регулярности зависят от `enable_regularity_metrics` и `enable_basic_metrics`
- Метрики темпа зависят от `enable_tempo_metrics` и `enable_basic_metrics`
- Временные метки ударов независимы от других фичей

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.rhythmic.segments[]`)
- **librosa**: основная библиотека для beat tracking (default backend)
- **essentia** (опционально): более точный алгоритм beat tracking (если доступен, no-fallback policy)
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `rhythmic` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## source_separation_extractor

### Краткое описание

Извлекает доли энергии источников (vocals, drums, bass, other) из аудио сигнала с использованием inprocess PyTorch модели разделения источников. Работает на уровне окон сегментации и вычисляет энергетические доли для каждого сегмента, а также агрегированные статистики (transitions, distribution, stability, balance).

**Версия**: 3.0.0  
**Категория**: source_separation  
**GPU**: preferred (PyTorch модель, требует GPU memory)

### Извлекаемые фичи

**Последовательности долей** (feature-gated: `enable_share_sequence`):
- `share_sequence`: массив долей энергии по сегментам, shape `[N, 4]` float32 (порядок: vocals, drums, bass, other)
- `energy_sequence`: массив абсолютных энергий по сегментам (feature-gated: `enable_energy_sequence`)

**Базовые статистики** (feature-gated: `enable_share_mean`, `enable_share_std`):
- `share_mean`: средние доли по всем сегментам, shape `[4]` float32
- `share_std`: стандартные отклонения долей по всем сегментам, shape `[4]` float32

**Дополнительные агрегаты** (вычисляются автоматически):
- `dominant_source_id`: ID доминирующего источника (argmax share_mean)
- `dominant_source_share`: доля доминирующего источника
- `source_balance_score`: метрика баланса источников (0 = один доминирует, 1 = равномерное распределение)
- `source_transitions_count`: количество переходов между доминирующими источниками (требует `enable_share_sequence`)
- `source_distribution`: распределение времени по источникам (требует `enable_share_sequence`)
- `source_stability_score`: метрика стабильности источников (требует `enable_share_sequence`)

**Расширенные фичи** (автоматически вычисляются если включен `share_sequence`):
- **Transition features**: `{source}_delta_mean/std/max` для каждого источника (vocals, drums, bass, other)
- **Stability features**: `{source}_stability` для каждого источника
- **Distribution features**: `{source}_mean_share`, `{source}_dominance_ratio` для каждого источника
- **Energy balance**: `source_entropy_mean/std`, `energy_balance_mean`
- **Musical heuristics**: `vocals_presence_ratio`, `drums_flux`, `bass_floor_p20`

**Метрики качества** (feature-gated: `enable_quality_metrics`):
- `source_quality_metrics`: словарь с метриками качества разделения (min, max, mean, std для shares и energies)

**Зависимости между фичами**:
- `dominant_source_id/share` и `source_balance_score` зависят от `enable_share_mean`
- `source_transitions_count`, `source_distribution`, `source_stability_score` зависят от `enable_share_sequence`
- Расширенные фичи автоматически вычисляются если включен `enable_share_sequence`
- `quality_metrics` зависят от `share_mean`, `share_std`, `share_sequence`, `energy_sequence`

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- Может использоваться в других компонентах как зависимость (требует `share_sequence` или `share_mean`)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): управление моделями и спецификациями (`source_separation_large_inprocess`)
- **TorchStateDictProvider** (`dp_models`): провайдер для загрузки PyTorch моделей
- **Segmenter**: генерация сегментов (family: `source_separation`)
- **torchaudio**: преобразование в mel спектрограммы (log-mel features)
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `source_separation` из `audio/segments.json`
- Работает только с сегментами от Segmenter (метод `run()` не поддерживается)
- Минимальная длительность аудио: **5 секунд** (иначе ошибка)
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## speaker_diarization_extractor

### Краткое описание

Выполняет диаризацию спикеров (определение "кто говорит когда") на основе полного аудио. Audit v3 целевой контракт: ModelManager-only (no-network), Segmenter-owned family `diarization` (одно full-audio окно), результаты публикуются как token-ready turn arrays (`turn_start_sec`, `turn_end_sec`, `turn_speaker_id`) и структурные per-speaker массивы (без object dict).

**Версия**: 3.1.0  
**Категория**: speech  
**GPU**: preferred

### Извлекаемые фичи

**Speaker turns (token-ready)**:
- `turn_start_sec`, `turn_end_sec`, `turn_speaker_id` (+ `turn_mask`)

**Per-speaker arrays**:
- `speaker_duration_sec`, `speaker_time_ratio`, `speaker_turns_count_by_speaker`

**Обязательные поля** (всегда присутствуют):
- `speaker_count`: количество обнаруженных спикеров (int)
- `speaker_ids`: список ID спикеров (list[int], `0..speaker_count-1`)
- `duration`: общая длительность аудио (float)
- `model_name`: имя модели диаризации (str)
- `weights_digest`: digest весов (str)

**Зависимости между фичами**:
- `speaker_durations` зависит от `speaker_stats` (использует `total_duration` для вычисления `speaker_time_ratios`)
- Все фичи зависят от результатов диаризации и транскрипции

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- **speech_analysis_extractor**: использует результаты diarization как зависимость (требует `speaker_segments`)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): управление моделями и спецификациями (strict offline)
- **pyannote.audio**: библиотека для speaker diarization Pipeline (`pyannote/speaker-diarization`)
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Требует Segmenter family `diarization` с ровно 1 сегментом (full-audio окно)
- Реальная диаризация выполняется на полном аудио (pyannote), но входной window используется для валидации sampling контракта

---

## spectral_entropy_extractor

### Краткое описание

Извлекает спектральную энтропию и связанные метрики из аудио сигнала. Спектральная энтропия измеряет распределенность энергии по частотному спектру — высокие значения указывают на равномерное распределение (белый шум), низкие — на концентрированную энергию (тональные звуки). Дополнительно вычисляются spectral flatness (спектральная плоскость) и spectral spread (разброс частот).

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: не требуется

### Извлекаемые фичи

**Базовые статистики** (feature-gated: `enable_basic_stats`):
- `spectral_entropy_stats`: статистики энтропии (mean, std, min, max, p25, p75)
- `spectral_entropy_variance`: дисперсия энтропии

**Flatness метрики** (feature-gated: `enable_flatness`):
- `spectral_flatness_stats`: статистики flatness (mean, std, min, max, p25, p75)
- `spectral_flatness_variance`: дисперсия flatness

**Spread метрики** (feature-gated: `enable_spread`):
- `spectral_spread_stats`: статистики spread (mean, std, min, max, p25, p75)
- `spectral_spread_variance`: дисперсия spread

**Временные серии** (feature-gated: `enable_time_series`):
- `spectral_entropy_series`: временная серия энтропии
- `spectral_flatness_series`: временная серия flatness (требует `enable_flatness`)
- `spectral_spread_series`: временная серия spread (требует `enable_spread`)

**Метрики динамики** (feature-gated: `enable_dynamics`, для `run_segments()`):
- `spectral_entropy_stability`: стабильность энтропии (variance)
- `spectral_entropy_transitions_count/rate`: количество и частота переходов
- `spectral_entropy_distribution`: распределение времени по уровням энтропии (low/medium/high)
- `spectral_entropy_diversity`: разнообразие значений энтропии (0.0-1.0)

**Расширенные статистики** (feature-gated: `enable_extended_stats`):
- `min`, `max`, `p25`, `p75` для всех метрик (требует соответствующие базовые фичи)

**Зависимости между фичами**:
- `spectral_flatness_stats` зависит от `enable_flatness`
- `spectral_spread_stats` зависит от `enable_spread`
- Все метрики динамики зависят от `enable_dynamics` и `enable_time_series`
- Расширенные статистики зависят от `enable_extended_stats` и соответствующих базовых фичей

**Upstream зависимости**:
- **spectral_extractor** (опционально): может использовать предвычисленный `stft_magnitude` или `mel_spectrogram` из `shared_features` для оптимизации

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.spectral_entropy.segments[]`)
- **librosa**: основная библиотека для STFT, Mel-спектрограмм и обработки аудио
- **spectral_extractor**: опциональная интеграция через `shared_features` для переиспользования спектрограммы
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует shared family `spectral` из `audio/segments.json` (Audit v3 shared-family policy)
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Минимальная длительность аудио: **1 секунда** (иначе ошибка)
- Если family отсутствует → error (no-fallback policy)

---

## spectral_extractor

### Краткое описание

Извлекает базовые спектральные признаки из аудио сигнала: спектральный центроид, ширина полосы, плоскостность, rolloff, скорость пересечения нуля (ZCR), контраст и дополнительные метрики (спектральный наклон и плоскостность в дБ).

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: не требуется

### Извлекаемые фичи

**Базовые признаки** (feature-gated: `enable_basic_features`):
- `spectral_centroid_stats`: статистики центроида спектра (mean, std, min, max, median) в Hz
- `spectral_bandwidth_stats`: статистики ширины полосы (mean, std, min, max, median) в Hz
- `spectral_flatness_stats`: статистики плоскостности спектра (mean, std, min, max, median), диапазон [0.0, 1.0]
- `spectral_rolloff_stats`: статистики частоты rolloff (mean, std, min, max, median) в Hz
- `zcr_stats`: статистики скорости пересечения нуля (mean, std, min, max, median), диапазон [0.0, 1.0]

**Контраст** (feature-gated: `enable_contrast`):
- `spectral_contrast_stats`: статистики спектрального контраста (mean, std, min, max, median)
- `spectral_contrast_bands`: полные данные контраста по частотным полосам (если `keep_contrast_bands=True`)
- `spectral_contrast_variance`: дисперсия контраста по полосам

**Продвинутые признаки** (feature-gated: `enable_advanced_features`):
- `spectral_slope_stats`: статистики спектрального наклона (dB/Hz)
- `spectral_flatness_db_stats`: статистики плоскостности в децибелах (dB)
- `spectral_slope_stability`: стабильность наклона

**Дополнительные метрики** (всегда включены, если включены basic_features):
- `spectral_centroid_median`: медиана центроида
- `spectral_bandwidth_ratio`: относительная ширина полосы (bandwidth / centroid)
- `spectral_rolloff_ratio`: относительный rolloff (rolloff / sample_rate)
- `spectral_flatness_entropy`: энтропия плоскостности
- `spectral_features_correlation`: корреляция между признаками (словарь парных корреляций)

**Временные серии** (feature-gated: `enable_time_series`):
- `centroid_series`, `bandwidth_series`, `flatness_series`, `rolloff_series`, `zcr_series`: временные серии признаков
- `contrast_series`, `slope_series`: временные серии контраста и наклона (если включены)
- `segment_centers_sec`, `segment_durations_sec`: временные метки сегментов (для `run_segments()`)

**Зависимости между фичами**:
- `spectral_flatness_db_stats` зависит от `spectral_flatness_stats` (требует `enable_basic_features`)
- `spectral_features_correlation` зависит от всех базовых признаков (требует `enable_basic_features`)
- `spectral_contrast_variance` зависит от `spectral_contrast_stats` (требует `enable_contrast`)
- `spectral_slope_stability` зависит от `spectral_slope_stats` (требует `enable_advanced_features`)

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- **band_energy_extractor** (опционально): может использовать предвычисленный `stft_magnitude` и `frequencies` из `shared_features` для оптимизации
- **spectral_entropy_extractor** (опционально): может использовать предвычисленный `stft_magnitude` или `mel_spectrogram` из `shared_features` для оптимизации

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.spectral.segments[]`)
- **librosa**: основная библиотека для всех спектральных операций (FFT, spectral features)
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `spectral` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## speech_analysis_extractor

### Краткое описание

Предоставляет компактный "обзор речи" путем комбинирования результатов нескольких экстракторов: ASR (токены распознавания речи через Whisper), speaker diarization (определение спикеров через pyannote.audio + whisperx) и опционально pitch (анализ высоты тона). Не сохраняет сырой текст транскрипции.

**Версия**: 2.1.0  
**Категория**: speech  
**GPU**: preferred (через зависимые компоненты ASR/diarization)

### Извлекаемые фичи

**ASR метрики** (feature-gated: `enable_asr_metrics`):
- `asr_segments_count`: количество ASR сегментов
- `asr_token_total`: общее количество токенов
- `asr_token_mean`, `asr_token_std`: среднее и стандартное отклонение количества токенов на сегмент
- `asr_token_density_per_sec`: плотность токенов (токенов/секунду)
- `asr_speech_rate_wpm`: скорость речи (слов в минуту)
- `asr_lang_distribution`: распределение языков по сегментам
- `asr_lang_id_by_segment`: идентификаторы языка для каждого сегмента

**Diarization метрики** (feature-gated: `enable_diarization_metrics`):
- `diar_segments_count`: количество сегментов diarization
- `speaker_count`: количество уникальных спикеров
- `dominant_speaker_share`: доля доминирующего спикера (0.0-1.0)
- `speaker_balance_score`: метрика баланса спикеров (0 = один доминирует, 1 = равномерное распределение)
- `speaker_transitions_count`: количество переходов между спикерами
- `speaker_ids`: список идентификаторов спикеров

**Pitch метрики** (feature-gated: `enable_pitch_metrics`, требует `pitch_enabled=True`):
- `pitch_f0_mean`, `pitch_f0_std`: среднее и стандартное отклонение основной частоты (Hz)
- `pitch_f0_min`, `pitch_f0_max`, `pitch_f0_range`: минимальное, максимальное значение и диапазон f0
- `pitch_stability`: метрика стабильности pitch (0 = нестабильная, 1 = стабильная)
- `pitch_distribution`: распределение pitch по октавам

**Зависимости между фичами**:
- Все ASR метрики зависят от результата `asr_extractor` (обязательная зависимость, fail-fast если не предоставлен)
- Все diarization метрики зависят от результата `speaker_diarization_extractor` (обязательная зависимость, fail-fast если не предоставлен)
- Все pitch метрики зависят от результата `pitch_extractor` (обязательная зависимость, fail-fast если не предоставлен)

**Upstream зависимости**:
- **asr_extractor** (обязательно, если `enable_asr_metrics=True`): использует результаты ASR из `extractor_results`
- **speaker_diarization_extractor** (обязательно, если `enable_diarization_metrics=True`): использует результаты diarization из `extractor_results`
- **pitch_extractor** (обязательно, если `pitch_enabled=True` и `enable_pitch_metrics=True`): использует результаты pitch из `extractor_results`

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): управление моделями через зависимые компоненты (Whisper для ASR, pyannote.audio для diarization)
- **Segmenter**: предоставляет сегменты для ASR (`families.asr`) и diarization (`families.diarization`)
- **AudioUtils**: загрузка и предобработка аудио для silence detection

### Segment Policy

- Использует два семейства сегментов из `audio/segments.json`:
  - `families.asr.segments[]`: длинные окна для ASR анализа (10-30 секунд)
  - `families.diarization.segments[]`: окна для diarization
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Минимальная длительность аудио: **5 секунд** (иначе ошибка)
- Если сегменты пустые → error (no-fallback policy)
- Тихое аудио → `status="empty"`, `empty_reason="audio_missing_or_extract_failed"` (если silence detection включен)

---

## tempo_extractor

### Краткое описание

Оценивает темп (BPM) и простые ритмические признаки на базе librosa. Использует onset-энергию и beat tracking для оценки темпа.

**Версия**: 1.1.0  
**Категория**: rhythm  
**GPU**: не требуется

### Извлекаемые фичи

**Глобальные метрики** (всегда присутствуют):
- `tempo_bpm`: основной BPM (median или mean, в зависимости от `aggregate`)
- `tempo_bpm_mean`, `tempo_bpm_median`, `tempo_bpm_std`: статистики BPM
- `tempo_estimates`: массив всех оценок BPM (float32[])
- `confidence`: уверенность оценки (0.0-1.0, вычисляется как `1.0 / (1.0 + std/mean)`)
- `warnings`: список предупреждений (например, `["low_confidence"]`, `["tempo_out_of_range"]`, `["signal_too_quiet"]`)

**Пер-оконные последовательности** (для `run_segments()`):
- `windowed_bpm`: словарь с полями:
  - `times_sec`: центры сегментов в секундах (float32[])
  - `bpm`: BPM для каждого сегмента (float32[])
  - `bpm_mean`, `bpm_median`, `bpm_std`: статистики BPM по сегментам
- `segments_count`: количество обработанных сегментов

**Пер-оконные последовательности** (для `run()` с `windowed_bpm=true`):
- `windowed_bpm`: словарь с полями:
  - `times_sec`: временные метки окон в секундах (float32[])
  - `bpm`: BPM для каждого окна (float32[])
  - `bpm_mean`, `bpm_median`, `bpm_std`: статистики BPM по окнам

**Зависимости между фичами**:
- `confidence` зависит от `tempo_bpm_std` и `tempo_bpm_mean`
- `warnings` зависят от `tempo_bpm_median`, `confidence` и уровня сигнала
- `windowed_bpm` статистики зависят от последовательностей `bpm` по сегментам/окнам

**Upstream зависимости**:
- Нет зависимостей от других extractors (работает независимо)

**Downstream зависимости**:
- **onset_extractor** (опционально): может использоваться для валидации/улучшения результатов (метрика `onset_tempo_consistency`)

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.tempo.segments[]`) — длинные sliding windows
- **librosa**: основная библиотека для onset detection и beat tracking
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `tempo` из `audio/segments.json`
- Сегменты — длинные sliding windows для устойчивой оценки BPM
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если `segments` пустой → error (no-fallback policy)

---

## voice_quality_extractor

### Краткое описание

Извлекает метрики качества голоса для оценки стабильности и гармоничности голоса: jitter (вариативность f0), shimmer (вариативность амплитуды) и HNR-подобная метрика (Harmonic-to-Noise Ratio).

**Версия**: 2.0.0  
**Категория**: voice  
**GPU**: optional (torchcrepe может использовать GPU для ускорения f0 estimation)

### Извлекаемые фичи

**Jitter метрики** (feature-gated: `enable_jitter`):
- `vq_jitter`: вариативность основной частоты (f0), нормализованная (0.0-1.0)
- `vq_jitter_mean`, `vq_jitter_std`, `vq_jitter_min`, `vq_jitter_max`: статистики разностей f0

**Shimmer метрики** (feature-gated: `enable_shimmer`):
- `vq_shimmer`: вариативность амплитуды, нормализованная (0.0-1.0)
- `vq_shimmer_mean`, `vq_shimmer_std`, `vq_shimmer_min`, `vq_shimmer_max`: статистики разностей амплитуд

**HNR метрики** (feature-gated: `enable_hnr`):
- `vq_hnr_like_db`: HNR-подобная метрика в децибелах (dB)
- `vq_hnr_mean`, `vq_hnr_std`, `vq_hnr_min`, `vq_hnr_max`: статистики HNR по окнам

**F0 статистики** (feature-gated: `enable_f0_stats`):
- `vq_f0_mean`, `vq_f0_std`, `vq_f0_min`, `vq_f0_max`, `vq_f0_median`: статистики f0 (Hz)
- `vq_f0_stability`: стабильность f0 (0.0-1.0, коэффициент вариации)
- `vq_voice_presence_ratio`: доля времени с присутствием голоса (0.0-1.0)

**Quality scores** (всегда включены, если включены jitter, shimmer и HNR):
- `vq_voice_quality_score`: композитная оценка качества голоса (0.0-1.0)
- `vq_breathiness_score`: оценка "дыхательности" голоса (0.0-1.0)

**Временные серии** (feature-gated: `enable_time_series`):
- `f0`, `amps`, `hnr_vals`: временные серии f0, амплитуд и HNR значений (или пути к `.npy` файлам)
- `segment_centers_sec`, `segment_durations_sec`: временные метки сегментов (для `run_segments()`)

**Зависимости между фичами**:
- Jitter зависит от оценки f0 (требует `f0_method` и `f0_fmin`/`f0_fmax`)
- Shimmer не зависит от других фичей
- HNR не зависит от других фичей
- F0 stats зависит от оценки f0
- Quality scores зависят от jitter, shimmer и HNR (все три должны быть включены)
- Time series зависит от включённых фичей

**Upstream зависимости**:
- **pitch_extractor** (опционально): может использовать результаты f0 из `pitch_payload` для оптимизации (избежание повторной оценки f0)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет аудио файл и сегменты (`families.voice_quality.segments[]`)
- **librosa**: библиотека для YIN/PYIN алгоритмов оценки f0
- **torchcrepe** (опционально): библиотека для точной оценки f0 (если `f0_method="torchcrepe"`)
- **pitch_extractor** (опционально): интеграция для использования более точных оценок f0
- **AudioUtils**: загрузка и предобработка аудио

### Segment Policy

- Использует family `voice_quality` из `audio/segments.json`
- Сегменты определяются Segmenter через нелинейную кривую
- Параметры кривой сохраняются в `audio/segments.json`
- Если family отсутствует → error (no-fallback policy)

---

## TextProcessor

### Общее описание

TextProcessor — процессор текстовой модальности, извлекающий текстовые признаки из метаданных видео (заголовок, описание, транскрипция, комментарии). Сохраняет результаты в per-run `result_store` и артефакты.

### Структура модулей

**Core** (`src/core/`):
- `main_processor.py`: MainProcessor — главный координатор extractors, оркестрация обработки
- `base_extractor.py`: Базовый класс для всех extractors
- `model_registry.py`: Реестр моделей для переиспользования между extractors
- `text_utils.py`: Утилиты для работы с текстом (нормализация, токенизация)
- `path_utils.py`: Утилиты для работы с путями к артефактам
- `renderer.py`: Генерация render-context для визуализации
- `metrics.py`: Метрики системы и производительности

**Schemas** (`src/schemas/`):
- `models.py`: Pydantic модели для валидации данных (VideoDocument)

**Extractors** (`src/extractors/`):
- Различные extractors для обработки текстовых данных

### Взаимосвязи с модулями системы

- **AudioProcessor**: предоставляет ASR транскрипцию через `doc.asr` (token IDs или сегменты с текстом)
- **ModelManager** (`dp_models`): управление моделями (SentenceTransformer и др.), строго локальная загрузка без сети
- **Embedding Service**: может использовать эмбеддинги для семантического поиска
- **Segmenter**: предоставляет метаданные видео (заголовок, описание, комментарии)

---

## asr_text_proxy_audio_features

### Краткое описание

Извлекает audio-like proxy признаки из текста ASR транскрипции для оценки качества распознавания, "шумности" и ритма речи без прямого анализа звукового сигнала. Работает с транскрипцией от AudioProcessor.

**Версия**: 1.2.0  
**Категория**: text-based audio proxy  
**GPU**: не требуется

Таблица и диапазоны: `TextProcessor/src/extractors/asr_text_proxy_audio_features/docs/FEATURE_DESCRIPTION.md`. Валидатор среза в `text_features.npz`: `.../utils/validate_asr_text_proxy_text_npz.py`.

### Извлекаемые фичи

**Конфиг / аудит (отражают параметры конструктора):**
- `tp_asrproxy_enabled`, `tp_asrproxy_basic_enabled`, `tp_asrproxy_noise_enabled`, `tp_asrproxy_rhythm_enabled`, `tp_asrproxy_intonation_enabled` (0/1)
- `tp_asrproxy_require_asr_text_enabled`, `tp_asrproxy_strict_document_duration_enabled` (0/1)
- `tp_asrproxy_low_conf_threshold`, `tp_asrproxy_words_per_minute_baseline`, `tp_asrproxy_max_text_chars` (порог, baseline WPM, лимит символов)

**Presence и размер**:
- `tp_asrproxy_present`: наличие транскрипта (0/1)
- `tp_asrproxy_has_confidence`: наличие confidence хотя бы у одного сегмента (0/1)
- `tp_asrproxy_segments_count`: количество сегментов ASR
- `tp_asrproxy_text_chars`, `tp_asrproxy_word_count`: размеры текста
- `tp_asrproxy_confidence_present_rate`: доля сегментов, у которых задано поле `confidence` (не `None`)

**Длительность и деградация:**
- `tp_asrproxy_audio_duration_sec`: итоговая длительность (с)
- `tp_asrproxy_duration_from_payload_flag`: длительность взята из ASR payload (0/1)
- `tp_asrproxy_duration_invalid_flag`: невалидная длительность (обычно до NPZ не доходит)

**Флаги валидации / качества ввода:**
- `tp_asrproxy_text_truncated_flag`, `tp_asrproxy_asr_schema_invalid_flag`, `tp_asrproxy_conf_invalid_flag`, `tp_asrproxy_token_decode_failed_flag` (0/1)

**Confidence метрики** (feature-gated: `enable_basic`):
- `tp_asrproxy_confidence_mean`, `tp_asrproxy_confidence_std`: статистики confidence
- `tp_asrproxy_confidence_chunked_min`: минимум **средних** confidence по блокам (~10 чанков по списку confidence)
- `tp_asrproxy_low_conf_rate`: доля сегментов с низкой confidence (`<` threshold)

**Noise proxies** (feature-gated: `enable_noise`; агрегат также использует `low_conf_rate` при `enable_basic`):
- `tp_asrproxy_text_noise_rare_ratio`: доля редких слов (длина > 12 или много символов/цифр)
- `tp_asrproxy_text_noise_oov_ratio`: доля out-of-vocabulary токенов (мало букв)
- `tp_asrproxy_noise_proxy`: агрегированный proxy шумности (ограниченный `min(1, mean(...))` по доступным rare_ratio / low_conf_rate)
- `tp_asrproxy_noise_proxy_present`: флаг, что агрегат определён (0/1)

**Rhythm метрики** (feature-gated: `enable_rhythm`):
- `tp_asrproxy_speech_rate_wpm`: скорость речи (слов в минуту)
- `tp_asrproxy_speech_rate_wpm_ratio_to_baseline`: отношение WPM к `words_per_minute_baseline` (в конфиге)
- `tp_asrproxy_speech_char_density`: плотность символов (символов/секунду)
- `tp_asrproxy_pause_density`: эвристика пауз: запятые/точка с запятой/двоеточие на «предложение» (счётчики `.` `?` `!`)
- `tp_asrproxy_filler_ratio`: доля слов из небольшого filler-лексикона

**Intonation метрики** (feature-gated: `enable_intonation`):
- `tp_asrproxy_sentence_intonation`: доля восклицательных/вопросительных предложений

**Зависимости между фичами**:
- `noise_proxy` зависит от `enable_noise` и/или `enable_basic` (агрегирует rare_ratio и low_conf_rate)
- Все rhythm метрики зависят от `audio_duration_sec` (требуется для вычисления WPM и density)
- Intonation зависит от наличия транскрипта

**Upstream зависимости**:
- **AudioProcessor** (обязательно): предоставляет ASR транскрипцию через `doc.asr` (preferred) или `doc.transcripts_meta` (legacy)
- **Segmenter/AudioProcessor**: предоставляет `audio_duration_sec` (required, fail-fast если отсутствует)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **AudioProcessor**: источник ASR транскрипции (`doc.asr.segments[]` с полями `text`, `confidence`, `start_sec`, `end_sec`)
- **Segmenter**: предоставляет `audio_duration_sec` (контракт: Segmenter извлекает аудио, AudioProcessor предоставляет duration в TextProcessor)

### Segment Policy

- Не использует сегментацию (работает с полной транскрипцией)
- Требует `audio_duration_sec` (fail-fast если отсутствует)
- Если транскрипта нет → valid empty (NaN + masks, без эвристических fallback)
- Текст обрезается до `max_text_chars` (по умолчанию 200000) если превышает лимит

---

## comments_aggregator

### Краткое описание

Агрегирует эмбеддинги комментариев (уже вычисленные `CommentsEmbedder`) в единые векторы-представители с использованием двух стратегий: взвешенное среднее (с весами: likes × authority × recency) и медиана по компонентам.

**Версия**: 1.3.0  
**Категория**: embedding aggregation  
**GPU**: не требуется

Таблица, зеркала и диапазоны: `TextProcessor/src/extractors/comments_aggregator/docs/FEATURE_DESCRIPTION.md`. Валидатор среза: `.../utils/validate_comments_aggregator_text_npz.py`. Ровно **39** плоских ключей: **22** `tp_commentsagg_*`, **12** `tp_comments_agg_*`, **5** `tp_cagg_*` (дубликаты по смыслу для back-compat).

### Извлекаемые фичи

**Основные метрики**:
- `tp_commentsagg_present`: агрегаты вычислены (0/1) — не «файл существует», а факт векторов mean/median
- `tp_commentsagg_count`, `tp_commentsagg_dim`: `N` и `D` матрицы; при valid-empty `dim` = **NaN**
- `tp_commentsagg_mean_std` / `tp_commentsagg_median_std`: `mean` по координатам от `std` по комментариям; **NaN**, если `compute_std=false` или выключен соответствующий `compute_*`

**Конфиг (аудит)**: `tp_commentsagg_compute_mean_enabled`, `compute_median_enabled`, `compute_std_enabled`, `write_artifacts_enabled`, `require_comment_embeddings_enabled` (0/1)

**Weights**:
- `tp_commentsagg_weights_applied`: 0/1 — перемножались ли факторы весов (после `selected_indices`)
- `tp_commentsagg_weights_mask_likes` / `authority` / `recency`: 0/1, какой сигнал участвовал
- `tp_commentsagg_weights_align_present`, `tp_commentsagg_weights_align_shape_ok`: индексы выравнивания и `len == N`

**Безопасность / качество ввода**:
- `tp_commentsagg_dim_mismatch_flag`: 1.0, если `np.ndarray` был, но форма не (N>0, D>0)
- `tp_commentsagg_unsafe_relpath_flag`: выход relpath за пределы `artifacts_dir`

**Артефакты**:
- `tp_commentsagg_artifact_mean_written`, `tp_commentsagg_artifact_median_written` (0/1)

**Extra-тайминги (мс)**: `tp_commentsagg_agg_mean_ms`, `tp_commentsagg_agg_median_ms` — только при `emit_extra_metrics=True`; иначе **NaN** (и **NaN**, если отключён соответствующий `compute_*`).

**Legacy**: `tp_comments_agg_*` (включая `compute_std` / `compute_mean` / `compute_median` как **флаги**) и `tp_cagg_*` — зеркала canonical; `count`/`present`/`dim`/std-слоты должны совпадать.

**Зависимости между фичами**:
- `mean_std` / `median_std` зависят от `compute_std` и соответствующего `compute_mean` / `compute_median`
- `weights_applied` — от наличия весов и выравнивания индексов с матрицей

**Upstream зависимости**:
- **CommentsEmbedder** (обязательно): создаёт эмбеддинги комментариев и регистрирует их в `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`
- **CommentsEmbedder**: создаёт `selected_indices` для выравнивания весов (`doc.tp_artifacts["comments"]["selected_indices_relpath"]`)

**Downstream зависимости**:
- **cosine_metrics_extractor**: может использовать агрегированные эмбеддинги для вычисления косинусного сходства с транскрипцией

### Взаимосвязи с модулями системы

- **CommentsEmbedder**: источник эмбеддингов комментариев (матрица `N×D` в `comments_embeddings.npy`)
- **VideoDocument**: опциональные веса через `doc.comments_likes`, `doc.comments_authority`, `doc.comments_recency`
- **Artifacts storage**: сохраняет агрегированные векторы в per-run `text_processor/_artifacts/` (`comments_agg_mean.npy`, `comments_agg_median.npy`)

### Segment Policy

- Не использует сегментацию (работает с полной матрицей эмбеддингов)
- Если эмбеддинги отсутствуют → valid empty (`tp_commentsagg_present=0`), fail-fast при `require_comment_embeddings=True`
- Веса выравниваются через `selected_indices` от `CommentsEmbedder` (если доступны)
- Агрегированные векторы L2-нормализованы для использования в косинусной метрике

---

## comments_embedder

### Краткое описание

Извлекает L2-нормализованные эмбеддинги для комментариев видео с использованием sentence-transformers модели, строго через `dp_models` (offline/no-network). Поддерживает батчинг, детерминированный отбор/лимиты, optional cache и per-run sub-artifact.

**Версия**: 1.3.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

Док+диапазоны+валидатор: `TextProcessor/src/extractors/comments_embedder/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_comments_embedder_text_npz.py`. Всего **18** полей `tp_commentsemb_*`.

### Извлекаемые фичи

**Core (8, ключи не гейтятся `emit_extra_metrics`)**: `tp_commentsemb_present`, `count`, `dim`, `n_input`, `n_deduped`, `n_selected`, `total_chars_used`, `truncated_by_total_chars_flag`. На valid-empty: `count=0`, `dim=NaN` и т.д.; при `compute_embeddings=False` (отбор без encode) `count=NaN`, `present=0`.

**Extra (10)**: `cache_enabled`, `cache_hit`, `fp16`, `device_cuda`, `model_digest_u24`, `compute_enabled`, `write_artifact_enabled`, `artifact_written`, `select_ms`, `encode_ms`. При **`emit_extra_metrics=False`** (типично) **все 10 — NaN** в NPZ. При **`emit_extra_metrics=True`**: 0/1 / мс; в **`extract_batch`** `cache_hit=NaN` (единый encode).

**Семантика**:
- `tp_commentsemb_model_digest_u24` — `int(weights_digest[:6], 16)` (идентификатор весов)
- `tp_commentsemb_encode_ms`: в `extract` — wall encode; в `extract_batch` — **доля** общего batch-encode, пропорциональная числу комментариев документа

**Кэш (extra при включённых метриках)**:
- `cache_hit` в `extract` — 0/1; в `extract_batch` с extras — **NaN** (не «промах кеша»)

**Зависимости между фичами**:
- `n_selected` / отбор: `selection_policy`, лайки/рецензия, `max_comments`, `max_total_chars`
- При `present=1` ожидается `count` ≈ `n_selected` и `dim` > 0
- `encode_ms` (extra): зависит от `extract` vs `extract_batch` (см. схему)

**Upstream зависимости**:
- **VideoDocument**: предоставляет комментарии через `doc.comments` (список объектов с полем `text`)
- **VideoDocument**: опциональные веса через `doc.comments_likes`, `doc.comments_recency` для политики отбора

**Downstream зависимости**:
- **comments_aggregator**: использует созданные эмбеддинги для агрегации
- **cosine_metrics_extractor**: может использовать эмбеддинги для вычисления косинусного сходства

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`comments_embeddings.npy`, `comments_selected_indices.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["embeddings"]["comments"]` для downstream extractors

### Segment Policy

- Не использует сегментацию (работает с полным списком комментариев)
- Если комментариев нет → valid empty (`tp_commentsemb_present=0`, артефакт не создаётся)
- Отбор комментариев: детерминированная политика (`by_likes_then_recency`, `by_likes`, `by_recency`, `first_k`) с лимитами (`max_comments`, `max_total_chars`)
- Дедупликация: удаление дубликатов по `normalize_whitespace(text)` (если `dedup_comments=True`)
- Фильтрация: удаление пустых и слишком коротких комментариев (`min_chars_per_comment`)
- Обрезка: каждый комментарий обрезается до `max_chars_per_comment` (по умолчанию 400)
- Батчинг: поддерживает `extract_batch()` для обработки нескольких документов одновременно

---

## cosine_metrics_extractor

### Краткое описание

Вычисляет метрики косинусного сходства между различными текстовыми эмбеддингами видео: заголовком, описанием, транскрипцией и комментариями. Загружает эмбеддинги из артефактов, созданных другими экстракторами.

**Версия**: 1.3.0  
**Категория**: similarity metrics  
**GPU**: не требуется  

**Machine schema**: `cosine_metrics_extractor_output_v1` (39 ключей). Human: `TextProcessor/src/extractors/cosine_metrics_extractor/SCHEMA.md`. Диапазоны и чеклист: `.../cosine_metrics_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_cosine_metrics_extractor_text_npz.py`.

### Извлекаемые фичи

**Косинусные метрики** (feature-gated):
- `tp_cos_title_desc`: косинусное сходство заголовка и описания (если `compute_title_desc=True`)
- `tp_cos_title_transcript`: косинусное сходство заголовка и транскрипции (если `compute_title_transcript=True`)
- `tp_cos_desc_transcript`: косинусное сходство описания и транскрипции (если `compute_desc_transcript=True`)
- `tp_cos_transcript_comments_mean`: среднее косинусное сходство транскрипции и комментариев (если `compute_transcript_comments_mean=True`)
- `tp_cos_transcript_comments_median`: медианное косинусное сходство транскрипции и комментариев (если `compute_transcript_comments_median=True`)

**Presence флаги** (всегда включены):
- `tp_cos_title_present`: наличие эмбеддинга заголовка (0/1)
- `tp_cos_desc_present`: наличие эмбеддинга описания (0/1)
- `tp_cos_transcript_present`: наличие эмбеддинга транскрипции (0/1)
- `tp_cos_comments_present`: наличие эмбеддингов комментариев (0/1)

**Feature-gating флаги** (всегда включены):
- `tp_cos_title_desc_enabled`: включена ли метрика title↔description (0/1)
- `tp_cos_title_transcript_enabled`: включена ли метрика title↔transcript (0/1)
- `tp_cos_desc_transcript_enabled`: включена ли метрика desc↔transcript (0/1)
- `tp_cos_transcript_comments_mean_enabled`: включена ли метрика transcript↔comments (mean) (0/1)
- `tp_cos_transcript_comments_median_enabled`: включена ли метрика transcript↔comments (median) (0/1)

**Диагностика** (всегда включены):
- `tp_cos_dim_mismatch_flag`: несовпадение размерностей или ошибка математики (0/1)
- `tp_cos_pair_dim_mismatch_flag`: проблема в title/desc/transcript парах (0/1)
- `tp_cos_tc_dim_mismatch_flag`: проблема в transcript↔comments ветке (0/1)
- `tp_cos_zero_norm_flag`: встречен вырожденный вектор с нормой ~0 (0/1)
- `tp_cos_unsafe_relpath_flag`: входной relpath небезопасный/вне `artifacts_dir` (0/1)

**Empty причины** (privacy-safe, только если релевантны включённым метрикам):
- `tp_cos_empty_no_title`: отсутствует заголовок (0/1)
- `tp_cos_empty_no_desc`: отсутствует описание (0/1)
- `tp_cos_empty_no_transcript`: отсутствует транскрипция (0/1)
- `tp_cos_empty_no_comments`: отсутствуют комментарии (0/1)

**Источник agg_mean транскрипта** (one-hot, фиксированный набор ключей):
- `tp_cos_transcript_agg_source_whisper`, `tp_cos_transcript_agg_source_youtube_auto`, `tp_cos_transcript_agg_source_combined`

**Политики конфигурации** (зеркала):
- `tp_cos_require_any_metric_enabled`, `tp_cos_require_title_enabled`, `tp_cos_require_description_enabled`, `tp_cos_require_transcript_enabled`, `tp_cos_require_comments_for_tc_enabled`
- `tp_cos_emit_extra_metrics_enabled`

**Дополнительные метрики** (ключи всегда в `features_flat`; при `emit_extra_metrics=False` тайминги и matrix-статы → **NaN**):
- `tp_cos_load_ms`, `tp_cos_compute_ms`
- `tp_cos_comments_mode_aggregates`, `tp_cos_comments_mode_matrix` (зеркало режима; неизвестный режим → **0/0**)
- `tp_cos_tc_n_comments_used`, `tp_cos_tc_sims_std`, `tp_cos_tc_sims_p95` (в основном для **matrix**)

**Зависимости между фичами**:
- Все косинусные метрики зависят от наличия соответствующих эмбеддингов
- `transcript_comments_mean/median` зависят от `comments_mode` (aggregates или matrix)
- Empty причины зависят от включённых метрик (вычисляются только для релевантных пар)

**Upstream зависимости**:
- **title_embedder** (опционально): создаёт эмбеддинг заголовка (`doc.tp_artifacts["embeddings"]["title"]["relpath"]`)
- **description_embedder** (опционально): создаёт эмбеддинг описания (`doc.tp_artifacts["embeddings"]["description"]["relpath"]`)
- **transcript_aggregator** (опционально): создаёт агрегированный эмбеддинг транскрипции (`doc.tp_artifacts["transcripts"][source]["agg_mean_relpath"]`)
- **comments_aggregator** (опционально): создаёт агрегированные эмбеддинги комментариев (`doc.tp_artifacts["comments"]["agg_mean_relpath"]`, `agg_median_relpath`)
- **comments_embedder** (опционально, для matrix режима): создаёт матрицу эмбеддингов комментариев (`doc.tp_artifacts["embeddings"]["comments"]["relpath"]`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)

### Segment Policy

- Не использует сегментацию (работает с агрегированными эмбеддингами)
- Если эмбеддинги отсутствуют → соответствующие метрики становятся `NaN` (valid empty)
- Fail-fast политики: `require_title`, `require_description`, `require_transcript`, `require_comments_for_tc` (вызывают RuntimeError если обязательный вход отсутствует)
- Приоритет транскрипции: `transcript_source_priority` (только `whisper` / `youtube_auto` / `combined`; по умолчанию в коде **`whisper` → youtube_auto**; `combined` — при явном указании в конфиге)
- Режим комментариев: `comments_mode` (`aggregates` использует агрегаты от `comments_aggregator`, `matrix` использует матрицу от `comments_embedder`)
- Вырожденные векторы (норма ~0): метрики становятся `NaN` (а не 0.0), `tp_cos_zero_norm_flag=1`

---

## description_embedder

### Краткое описание

Извлекает L2-нормализованные эмбеддинги для описаний видео (description) с использованием sentence-transformers модели, строго через `dp_models` (offline/no-network). Поддерживает обработку длинных текстов через token-aware chunking и агрегацию с attention-weighted pooling.

**Версия**: 1.2.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

Док+диапазоны+валидатор: `TextProcessor/src/extractors/description_embedder/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_description_embedder_text_npz.py`. Срез: **19** ключей `tp_descemb_*`.

### Извлекаемые фичи

**Основные метрики** (всегда включены):
- `tp_descemb_present`: эмбеддинг вычислен (0/1)
- `tp_descemb_dim`: размерность эмбеддинга
- `tp_descemb_norm_raw`: L2-норма необработанного вектора (до нормализации, NaN если `compute_raw_norm=False`)
- `tp_descemb_l2_norm`: L2-норма нормализованного вектора (должна быть ~1.0)

**Presence и конфигурация** (всегда включены):
- `tp_descemb_description_present`: присутствует ли описание в документе (0/1)
- `tp_descemb_compute_enabled`: включено ли вычисление эмбеддинга (0/1)
- `tp_descemb_write_artifact_enabled`: включено ли сохранение артефакта (0/1)
- `tp_descemb_artifact_written`: был ли артефакт успешно записан (0/1)
- `tp_descemb_cache_enabled`: включено ли кеширование (0/1)
- `tp_descemb_cache_hit`: 0/1 при успешном пути encode; **NaN** в шаблоне valid-empty; при `cache_enabled=False` на ветке encode задаётся **0.0**

**Model метрики**:
- `tp_descemb_fp16`: использовался ли режим float16 (0/1)
- `tp_descemb_device_cuda`: использовалось ли устройство CUDA (0/1)
- `tp_descemb_model_digest_u24`: первые 24 бита хеша модели (для идентификации)

**Chunking и pooling метрики**:
- `tp_descemb_pooling_length_weighted`: использовалась ли стратегия length_weighted_mean (0/1)
- `tp_descemb_n_chunks`: количество чанков, на которые был разбит текст (NaN если не применимо)
- `tp_descemb_avg_chunk_tokens`: среднее количество токенов в чанке (NaN если не применимо)

**Timing метрики** (NaN если не применимо):
- `tp_descemb_chunk_ms`: время разбиения на чанки (миллисекунды)
- `tp_descemb_encode_ms`: время кодирования через модель (миллисекунды)
- `tp_descemb_pool_ms`: время агрегации (pooling) эмбеддингов чанков (миллисекунды)

**Зависимости между фичами**:
- `norm_raw` зависит от `compute_raw_norm` (NaN если отключено)
- `n_chunks` и `avg_chunk_tokens` зависят от длины текста (NaN если текст короткий и не требует chunking)
- `cache_hit` зависит от `cache_enabled` (NaN если кеш отключен)

**Upstream зависимости**:
- **VideoDocument**: предоставляет описание через `doc.description` (str)

**Downstream зависимости**:
- **cosine_metrics_extractor**: может использовать эмбеддинг для вычисления косинусного сходства
- **embedding_pair_topk_extractor**: может использовать эмбеддинг для вычисления сходства с транскрипцией

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей строго локально, без сетевых загрузок
- **ModelManager** (`dp_models): загрузка shared tokenizer (`shared_tokenizer_v1`) для token-aware chunking
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`description_embedding.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["embeddings"]["description"]` для downstream extractors

### Segment Policy

- Не использует сегментацию (работает с полным текстом описания)
- Если описание пустое → valid empty (`tp_descemb_present=0`, артефакт не создаётся)
- Token-aware chunking: длинные тексты разбиваются на чанки по `max_chunk_tokens_model` (по умолчанию 512) через `shared_tokenizer_v1`
- Pooling стратегии: `mean`, `length_weighted_mean` (default), `max`, `logsumexp`
- L2 нормализация: финальный вектор всегда L2-нормализован (норма ≈ 1.0)
- Кеширование: опциональное кеширование по SHA256(content + model_name + config) для избежания повторных вычислений

---

## embedding_pair_topk_extractor

### Краткое описание

Вычисляет топ-K наиболее похожих чанков транскрипта для заголовка видео на основе косинусного сходства эмбеддингов. Также вычисляет косинусное сходство между заголовком и описанием. Поддерживает эффективный поиск через FAISS для больших корпусов.

**Версия**: 1.3.0  
**Категория**: similarity search  
**GPU**: не требуется (опционально для cross-encoder, но отключено по умолчанию)

Док+диапазоны+валидатор: `TextProcessor/src/extractors/embedding_pair_topk_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_embedding_pair_topk_extractor_text_npz.py`. Срез: **69** ключей `tp_embpair_*` + legacy `tp_pairtopk_*` (machine: `embedding_pair_topk_extractor_output_v1`).

### Извлекаемые фичи

**Основные метрики** (feature-gated):
- `tp_embpair_present`: присутствует ли хотя бы одна вычисленная метрика (0/1)
- `tp_embpair_title_desc_cosine`: косинусное сходство заголовок↔описание (если `compute_title_desc=True`)
- `tp_embpair_title_transcript_topk_max`: максимум среди топ-K сходств (если `export_topk_summary=True`)
- `tp_embpair_title_transcript_topk_mean`: среднее среди топ-K сходств (если `export_topk_summary=True`)

**Presence флаги** (всегда включены):
- `tp_embpair_title_present`: присутствует ли эмбеддинг заголовка (0/1)
- `tp_embpair_desc_present`: присутствует ли эмбеддинг описания (0/1)
- `tp_embpair_transcript_chunks_present`: присутствует ли матрица эмбеддингов чанков транскрипта (0/1)
- `tp_embpair_title_desc_present`: успешно ли вычислено сходство заголовок↔описание (0/1)
- `tp_embpair_title_transcript_topk_present`: успешно ли вычислен топ-K поиск (0/1)

**Топ-K слоты** (схема: ровно **8** фиксированных ключей `top1..top8`; `export_topk_slots` / `export_topk_indices` заполняют префикс длины `min(top_k_slots, 8)`):
- `tp_embpair_title_transcript_top{1..8}`: сходства (NaN если слот не экспортирован / нет данных)
- `tp_embpair_title_transcript_top{1..8}_idx`: индексы чанков (если `export_topk_indices=True`, privacy-safe)
- дубликаты: `tp_pairtopk_title_transcript_top{1..8}` = те же значения, что `tp_embpair_*` (back-compat)

**Feature-gating флаги** (всегда включены):
- `tp_embpair_enabled`: включен ли экстрактор (0/1)
- `tp_embpair_disabled_by_policy`: отключен ли по политике (0/1)
- `tp_embpair_compute_title_desc_enabled`: включено ли вычисление заголовок↔описание (0/1)
- `tp_embpair_compute_title_transcript_topk_enabled`: включен ли топ-K поиск (0/1)
- `tp_embpair_export_topk_slots_enabled`: включен ли экспорт слотов (0/1)
- `tp_embpair_export_topk_indices_enabled`: включен ли экспорт индексов (0/1)
- `tp_embpair_export_topk_summary_enabled`: включен ли экспорт сводки (0/1)

**Диагностические флаги** (всегда включены):
- `tp_embpair_dim_mismatch_flag`: несовпадение размерностей эмбеддингов (0/1)
- `tp_embpair_unsafe_relpath_flag`: небезопасный relpath (path traversal) (0/1)
- `tp_embpair_nan_inf_flag`: обнаружены NaN/Inf в эмбеддингах (0/1)
- `tp_embpair_zero_norm_flag`: обнаружены вырожденные векторы (норма ~0) (0/1)
- `tp_embpair_used_legacy_key_flag`: использовался ли legacy ключ для транскрипта (0/1)

**Конфигурационные параметры** (всегда включены):
- `tp_embpair_top_k`: запрошенное количество топ-K кандидатов
- `tp_embpair_top_k_slots` / `tp_embpair_top_k_slots_requested` / `tp_embpair_top_k_slots_clamped`: эффективные слоты, значение из конфига до клампа, флаг клампа к схеме
- `tp_embpair_schema_slots_max`: **8** (жёсткий потолок Audit v3)
- `tp_embpair_use_faiss_mode_auto/never/always`: one-hot режима FAISS (0/1, ровно один = 1)
- `tp_embpair_min_corpus_for_faiss`: минимальный размер корпуса для использования FAISS
- `tp_embpair_require_faiss_enabled`: требуется ли FAISS (0/1)
- `tp_embpair_require_title_embedding_enabled`: требуется ли эмбеддинг заголовка (0/1)
- `tp_embpair_require_description_embedding_enabled`: требуется ли эмбеддинг описания (0/1)
- `tp_embpair_require_transcript_chunks_enabled`: требуется ли матрица чанков транскрипта (0/1)

**Дополнительные метрики** (ключи в схеме всегда; при `emit_extra_metrics=False` — **NaN**):
- `tp_embpair_n_chunks`: количество чанков в транскрипте
- `tp_embpair_transcript_source_whisper` / `youtube_auto` / `combined`: one-hot выбранного источника (0/1; сумма 0 или 1)
- `tp_embpair_use_faiss_mode`: скаляр 0 / 0.5 / 1 (never / auto / always), отдельно от triplet `tp_embpair_use_faiss_mode_*`
- `tp_embpair_require_faiss`: дубль политики `require_faiss` (0/1)

**Зависимости между фичами**:
- Все топ-K метрики зависят от `compute_title_transcript_topk` и наличия эмбеддингов
- `title_desc_cosine` зависит от `compute_title_desc` и наличия обоих эмбеддингов
- Топ-K слоты зависят от `export_topk_slots` (всегда присутствуют в схеме, но заполняются только если включено)
- Индексы зависят от `export_topk_indices` (privacy-safe, без текста)

**Upstream зависимости**:
- **title_embedder** (опционально): создаёт эмбеддинг заголовка (`doc.tp_artifacts["embeddings"]["title"]["relpath"]`)
- **description_embedder** (опционально): создаёт эмбеддинг описания (`doc.tp_artifacts["embeddings"]["description"]["relpath"]`)
- **transcript_chunk_embedder** (опционально): создаёт матрицу эмбеддингов чанков транскрипта (`doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)
- **FAISS** (опционально): библиотека для эффективного поиска в больших корпусах (автоматический fallback на NumPy если недоступен)

### Segment Policy

- Не использует сегментацию (работает с агрегированными эмбеддингами)
- Если эмбеддинги отсутствуют → valid empty (NaN + `*_present=0`), fail-fast при `require_*` флагах
- Приоритет транскрипции: `transcript_source_priority` (по умолчанию: `whisper → youtube_auto`)
- FAISS режим: `auto` (использует FAISS если корпус >= `min_corpus_for_faiss`), `never` (всегда NumPy), `always` (всегда пытается FAISS)
- Cross-encoder reranking: запрещён по умолчанию (fail-fast, требует raw chunk texts + dp_models spec + privacy gating)
- Вырожденные векторы (норма ~0): метрики становятся `NaN` (а не 0.0), `tp_embpair_zero_norm_flag=1`

---

## embedding_shift_indicator_extractor

### Краткое описание

Обнаруживает семантический сдвиг в транскрипте видео путём сравнения эмбеддингов начала и конца транскрипта. Вычисляет косинусное сходство между усреднёнными эмбеддингами начальных и конечных чанков и устанавливает флаг сдвига, если сходство ниже порога.

**Версия**: 1.3.0  
**Категория**: semantic analysis  
**GPU**: не требуется

Док+диапазоны+валидатор: `TextProcessor/src/extractors/embedding_shift_indicator_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_embedding_shift_indicator_extractor_text_npz.py`. Срез: **27** ключей `tp_embshift_*` (`embedding_shift_indicator_extractor_output_v1`).

### Извлекаемые фичи

**Контракт Audit v3:** **27** фиксированных ключей — `TextProcessor/src/extractors/embedding_shift_indicator_extractor/SCHEMA.md`, machine `embedding_shift_indicator_extractor_output_v1.json`.

**Основные метрики** (всегда включены):
- `tp_embshift_present`: **1.0** только если основной **`cosine_begin_end`** валиден (конечное число); иначе **0.0**
- `tp_embshift_n_chunks`: количество чанков в транскрипте
- `tp_embshift_n_window_chunks`: размер окна для усреднения (адаптивный)
- `tp_embshift_dim`: размерность эмбеддингов
- `tp_embshift_cosine_begin_end`: косинусное сходство между начальным и конечным окнами (NaN если данных недостаточно/zero-norm/NaN-Inf)
- `tp_embshift_shift_flag`: флаг сдвига (1.0 если cosine < threshold, 0.0 иначе, NaN если `compute_shift_flag=False` или cosine invalid)
- `tp_embshift_cosine_threshold`: порог косинусного сходства для определения сдвига
- `tp_embshift_margin`: разница между cosine и threshold (cosine - threshold, NaN если cosine invalid)

**Дополнительные метрики** (feature-gated: `compute_extra_cosines`):
- `tp_embshift_cosine_first_last`: косинусное сходство между первым и последним чанками
- `tp_embshift_mean_cosine_last_to_start_window`: среднее косинусное сходство между последними чанками и начальным окном

**Feature-gating флаги** (всегда включены):
- `tp_embshift_enabled`: включен ли экстрактор (0/1)
- `tp_embshift_disabled_by_policy`: отключен ли по политике (0/1)
- `tp_embshift_require_transcript_chunks_enabled`: требуется ли матрица чанков транскрипта (0/1)
- `tp_embshift_require_min_chunks`: минимальное число чанков для расчёта
- `tp_embshift_compute_shift_flag_enabled`: включен ли бинарный флаг сдвига (0/1)
- `tp_embshift_compute_extra_cosines_enabled`: включены ли дополнительные косинусы (0/1)
- `tp_embshift_emit_extra_metrics_enabled`: зеркало **`emit_extra_metrics`** в конфиге

**Source флаги** (всегда включены):
- `tp_embshift_source_used_whisper`: использовался ли источник whisper (0/1)
- `tp_embshift_source_used_youtube_auto`: использовался ли источник youtube_auto (0/1)
- `tp_embshift_used_legacy_key_flag`: использовался ли legacy ключ для транскрипта (0/1)

**Диагностические флаги** (всегда включены):
- `tp_embshift_unsafe_relpath_flag`: небезопасный relpath (path traversal) (0/1)
- `tp_embshift_chunk_embed_missing_flag`: безопасный relpath, но файла нет или ошибка **`.npy`**
- `tp_embshift_dim_mismatch_flag`: несовпадение размерностей или невалидная форма матрицы (0/1)
- `tp_embshift_zero_norm_flag`: обнаружены вырожденные векторы (норма ~0) (0/1)
- `tp_embshift_nan_inf_flag`: обнаружены NaN/Inf в эмбеддингах (0/1)

**Timing метрики** (в схеме всегда; числа только при **`emit_extra_metrics=True`**):
- `tp_embshift_load_ms`, `tp_embshift_compute_ms`: **NaN** при **`emit_extra_metrics=False`**

**Зависимости между фичами**:
- `shift_flag` зависит от `compute_shift_flag` и валидности `cosine_begin_end`
- `margin` зависит от валидности `cosine_begin_end` (NaN если cosine invalid)
- Дополнительные косинусы зависят от `compute_extra_cosines`
- `n_window_chunks` адаптивно вычисляется как `min(n_window_chunks, max(1, n_chunks // 2))`

**Upstream зависимости**:
- **transcript_chunk_embedder**: матрица в **`doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`** (канон; не требует ключа **`transcript_chunks`**) или legacy **`transcript_chunks`[][].embeddings_relpath**

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает матрицу эмбеддингов из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)

### Segment Policy

- Не использует сегментацию (работает с полной матрицей эмбеддингов чанков)
- Если эмбеддинги отсутствуют → valid empty (`tp_embshift_present=0`, NaN метрики), fail-fast при `require_transcript_chunks=True`
- Минимальное количество чанков: `require_min_chunks` (по умолчанию 2, fail-fast при `require_transcript_chunks=True` если недостаточно)
- Приоритет транскрипции: `transcript_source_priority` (по умолчанию: `whisper → youtube_auto`)
- Адаптивный размер окна: `min(n_window_chunks, max(1, n_chunks // 2))` для стабильного усреднения
- Вырожденные векторы (норма ~0): метрики становятся `NaN` (а не 0.0), `tp_embshift_zero_norm_flag=1`
- Интерпретация: высокое `cosine_begin_end` (близко к 1.0) означает семантическую консистентность, низкое — сдвиг темы

---

## qa_embedding_pairs_extractor

### Краткое описание

Извлекает вопросоподобные фразы из различных источников текста (заголовок, описание, транскрипт, комментарии) и вычисляет их L2-нормализованные эмбеддинги. Сохраняет эмбеддинги вопросов в артефакты для последующего использования в задачах поиска похожих вопросов или анализа FAQ-контента.

**Версия**: 1.3.0  
**Категория**: question extraction, embeddings  
**GPU**: поддерживается (cuda), опционально fp16

Док+диапазоны+валидатор: `TextProcessor/src/extractors/qa_embedding_pairs_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_qa_embedding_pairs_extractor_text_npz.py`. Срез: **34** ключа `tp_qa_*` (`qa_embedding_pairs_extractor_output_v1`).

### Извлекаемые фичи

**Основные метрики** (всегда включены):
- `tp_qa_present`: вопросы найдены и эмбеддинги вычислены (0/1)
- `tp_qa_num_questions`: общее количество найденных вопросов
- `tp_qa_embedding_dim`: размерность эмбеддингов

**Количество вопросов по источникам**:
- `tp_qa_q_title`: количество вопросов из заголовка
- `tp_qa_q_description`: количество вопросов из описания
- `tp_qa_q_transcript`: количество вопросов из транскрипта
- `tp_qa_q_comments`: количество вопросов из комментариев

**Политики и конфигурация**:
- `tp_qa_enabled`: включен ли экстрактор (0/1)
- `tp_qa_disabled_by_policy`: отключен ли политикой (0/1)
- `tp_qa_allow_legacy_transcripts`: разрешены ли legacy транскрипты (0/1)
- `tp_qa_transcript_source_policy_asr_only/asr_then_legacy/legacy_only`: политика выбора источника транскрипта (0/1)
- `tp_qa_use_title/use_description/use_transcript/use_comments`: feature-gating флаги (0/1)

**Параметры извлечения**:
- `tp_qa_require_min_questions`: минимальное требуемое количество вопросов
- `tp_qa_max_questions_total`: максимальное общее количество вопросов
- `tp_qa_max_questions_per_source`: максимальное количество вопросов на источник
- `tp_qa_max_comments`: максимальное количество комментариев для обработки
- `tp_qa_max_transcript_chars`: максимальное количество символов транскрипта
- `tp_qa_min_chars_per_question`: минимальная длина вопроса
- `tp_qa_max_question_chars`: максимальная длина вопроса
- `tp_qa_dedup_questions`: включена ли дедупликация вопросов (0/1)

**Опциональные артефакты**:
- `tp_qa_write_question_hashes_artifact_enabled`: включена ли запись хешей (0/1)
- `tp_qa_write_question_source_ids_artifact_enabled`: включена ли запись source IDs (0/1)
- `tp_qa_hashes_written`: были ли записаны хеши (0/1)
- `tp_qa_source_ids_written`: были ли записаны source IDs (0/1)

**Дополнительные метрики** (feature-gated: `emit_extra_metrics`):
- `tp_qa_questions_per_min`: количество вопросов в минуту (требует `audio_duration_sec`)
- `tp_qa_questions_per_1k_chars`: количество вопросов на 1000 символов
- `tp_qa_mean_cosine_to_centroid`: среднее косинусное сходство до центроида

**Зависимости между фичами**:
- Все метрики зависят от наличия вопросов в источниках
- `questions_per_min` зависит от `audio_duration_sec`
- `mean_cosine_to_centroid` зависит от наличия минимум 2 вопросов

**Upstream зависимости**:
- **AudioProcessor** (опционально): предоставляет ASR транскрипцию через `doc.asr.segments[].text` (preferred)
- **VideoDocument**: предоставляет `title`, `description`, `comments` (список объектов с полем `text`)
- **VideoDocument** (legacy, опционально): предоставляет `transcripts` (только если `allow_legacy_transcripts=True`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`qa_question_embeddings.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["qa"]["question_embeddings"]` для downstream extractors

### Segment Policy

- Не использует сегментацию (работает с полным текстом из источников)
- Если вопросов не найдено → valid empty (`tp_qa_present=0`, артефакты не создаются)
- Фильтрация вопросов: regex паттерн по question words (RU/EN), обязателен знак вопроса `?` или `？`
- Дедупликация: удаление дубликатов по canonical форме вопроса (casefold, нормализация пробелов)
- Лимиты: `max_questions_per_source` на источник, `max_questions_total` общий
- Fail-fast: если `require_min_questions > 0` и `num_questions < require_min_questions` → RuntimeError

---

## semantic_cluster_extractor

### Краткое описание

Определяет семантический кластер для видео по эмбеддингу **title** / **description** / **hashtag**: проекция **PCA**, ближайший **центроид** (cosine / inner product на L2-нормированных векторах). Ассеты таксономии — **`dp_models`** (`clusters_spec_name`, обычно `semantic_clusters_v1`).

**Версия**: 1.3.0  
**Категория**: clustering, classification  
**GPU**: не требуется (опционально FAISS для ускорения)

### Извлекаемые фичи

**Контракт Audit v3**: **31** фиксированный scalar в `features_flat` — см. `DataProcessor/TextProcessor/src/extractors/semantic_cluster_extractor/SCHEMA.md` и `semantic_cluster_extractor_output_v1.json` (`allow_extra_keys: false`; tier **analytics**).

Док+диапазоны+валидатор: `TextProcessor/src/extractors/semantic_cluster_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_semantic_cluster_extractor_text_npz.py`. Срез: **31** ключа `tp_semclust_*` (`semantic_cluster_extractor_output_v1`).

Кратко:
- **Зеркала политик:** `require_primary_source`, `require_embedding`, `use_faiss`, `require_faiss`, `emit_extra_metrics`
- **One-hot `primary_source` из конфига:** `tp_semclust_config_primary_*`
- **`tp_semclust_*_present`:** **1.0** только после **успешной** загрузки соответствующего **`.npy`** (не «наличие relpath»)
- **One-hot выбранного источника:** `tp_semclust_source_*`; **`tp_semclust_fallback_used`** если взяли не `primary_source`
- **Диагностика:** `tp_semclust_unsafe_relpath_flag`; `tp_semclust_*_embed_missing_flag` (безопасный relpath, но файл/CRC/пустой вектор — отдельно от unsafe)
- **Метрики:** `tp_semclust_present`, `tp_semclust_id`, `tp_semclust_similarity`, `tp_semclust_distance`; **NaN** при empty / dim mismatch по правилам кода
- **Extra-блок** (dims, margin, compute_ms, n_clusters): в схеме всегда; при **`emit_extra_metrics=False`** — **NaN** (см. `main.py` `_apply_extra_block`)
- **`tp_semclust_dim_mismatch_flag`**, **`tp_semclust_backend_faiss`**

**`semantic_cluster_meta`:** на всех ветках — `clusters_spec_*`, `cluster_db_version`, **`backend`** (`faiss_ip` | `numpy_cosine`). Верхний уровень ответа: **`model_*`/`weights_digest`** = **`null`**; **`system`** с **`pre_init`/`post_init`**, **`gpu_peak_mb`**.

**Зависимости между фичами**:
- `similarity` / `distance` / `present` согласованы с выбранным вектором и таксономией
- `fallback_used` — если фактический слот ≠ `primary_source`
- `dim_mismatch_flag` — несовместимость размерности эмбеддинга и PCA/модели

**Upstream зависимости**:
- **title_embedder** (опционально): создаёт эмбеддинг заголовка (`doc.tp_artifacts["embeddings"]["title"]["relpath"]`)
- **description_embedder** (опционально): создаёт эмбеддинг описания (`doc.tp_artifacts["embeddings"]["description"]["relpath"]`)
- **hashtag_embedder** (опционально): создаёт эмбеддинг хештегов (`doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]`); нужен в DAG при **`primary_source`** или fallback **`hashtag`**

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка PCA и центроидов через spec (`semantic_clusters_v1`), строго локально, без сетевых загрузок
- **FAISS** (опционально): библиотека для быстрого поиска ближайших соседей (автоматический fallback на NumPy если недоступен)
- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов

### Segment Policy

- Не использует сегментацию (работает с агрегированными эмбеддингами)
- `tp_semclust_{title,description,hashtag}_present` — успешная загрузка соответствующего `.npy`, а не только наличие `relpath` в `tp_artifacts`
- Если эмбеддинги отсутствуют → valid empty (`tp_semclust_present=0`, NaN метрики), fail-fast при `require_embedding=True`
- Политика источника: `primary_source` (title/description/hashtag) с опциональным fallback через `allow_fallback_sources`
- Dim mismatch: если размерность не совпадает → `dim_mismatch_flag=1`, fail-fast при `require_embedding=True`
- Кластеры — фиксированная таксономия (стабильные ID) + словарь `clusters.jsonl` для интерпретации

---

## semantics_topics_keyphrases

### Краткое описание

Извлекает глобальные (сопоставимые между видео) темы из текста через retrieval по фиксированной taxonomy (bundled `topics.jsonl` + embeddings через `dp_models`), а также ключевые фразы и дешёвые стилистические proxy-флаги. Компонент больше не обучает темы per-video (BERTopic/KMeans) — это было несопоставимо между видео.

**Версия**: 2.1.0  
**Категория**: topic modeling, keyphrase extraction, style analysis  
**GPU**: поддерживается (cuda), опционально fp16

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/semantics_topics_keyphrases/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_semantics_topics_keyphrases_text_npz.py`. Срез: **116** ключей `tp_topics_*` (`semantics_topics_keyphrases_output_v1`).

**Основные флаги** (всегда включены):
- `tp_topics_present`: наличие данных (1.0 если есть текст, 0.0 если пусто)
- `tp_topics_disabled_by_policy`: экстрактор отключен через `enabled=False` (1.0 если отключен)

**Присутствие данных**:
- `tp_topics_text_chars`: количество символов в объединенном тексте
- `tp_topics_has_asr`: наличие ASR транскрипта (1.0/0.0)
- `tp_topics_has_title`: наличие заголовка (1.0/0.0)
- `tp_topics_has_description`: наличие описания (1.0/0.0)

**Topics (retrieval)** (feature-gated: `enable_topic_distribution`):
- `tp_topics_topic_top1_id`: ID топ-1 темы
- `tp_topics_topic_top1_score`: сходство с топ-1 темой
- `tp_topics_topic_top1_prob`: вероятность топ-1 темы (softmax)
- `tp_topics_topic_top{i}_id`, `tp_topics_topic_top{i}_score`, `tp_topics_topic_top{i}_prob` (i=1..top_k_slots; стабильная схема)
- `tp_topics_entropy_topk`: энтропия Шеннона распределения вероятностей по топ-K темам
- `tp_topics_entropy_topk_norm`: нормализованная энтропия (энтропия / log(K))
- `tp_topics_perplexity_topk`: perplexity = exp(энтропия)

**Keyphrases** (feature-gated: `enable_keyphrases`):
- `tp_topics_keyphrases_count`: количество извлеченных ключевых фраз
- `tp_topics_keyphrase_score_top1`: оценка топ-1 ключевой фразы
- `tp_topics_keyphrase_score_mean`: средняя оценка всех ключевых фраз
- `tp_topics_keyphrases_dim`: размерность эмбеддингов ключевых фраз (если `enable_keyphrase_embeddings=True`)

**Keyphrases (privacy-safe export)** (feature-gated: `export_keyphrases_mode="hashed"`):
- `tp_topics_kp_top{i}_present`: наличие ключевой фразы в слоте i (1.0/0.0)
- `tp_topics_kp_top{i}_hash01`: хеш ключевой фразы (первый байт SHA256)
- `tp_topics_kp_top{i}_len`: длина ключевой фразы в символах (i=1..keyphrase_slots)

**Style proxies** (feature-gated: `enable_style_flags`):
- `tp_topics_style_faq_qmarks`: количество предложений, заканчивающихся на "?"
- `tp_topics_style_instructional_flag`: присутствие инструктивных ключевых слов (1.0/0.0)
- `tp_topics_style_audience_flag`: присутствие обращений к аудитории (1.0/0.0)
- `tp_topics_style_cta_flag`: присутствие призывов к действию (1.0/0.0)

**Конфигурационные флаги**:
- `tp_topics_enable_topic_distribution`: включено ли извлечение тем (1.0/0.0)
- `tp_topics_enable_keyphrases`: включено ли извлечение ключевых фраз (1.0/0.0)
- `tp_topics_enable_keyphrase_embeddings`: включены ли эмбеддинги ключевых фраз (1.0/0.0)
- `tp_topics_export_keyphrases_mode_raw/hashed/none`: режим экспорта (1.0/0.0)
- `tp_topics_enable_style_flags`: включены ли стилистические флаги (1.0/0.0)
- `tp_topics_allow_legacy_transcripts`: разрешены ли legacy транскрипты (1.0/0.0)
- `tp_topics_top_k_topics`: количество тем для извлечения
- `tp_topics_top_k_slots`: количество слотов для топ-K тем
- `tp_topics_temperature`: температура для softmax

**Зависимости между фичами**:
- Все topics метрики зависят от `enable_topic_distribution` и наличия текста
- Keyphrases метрики зависят от `enable_keyphrases` и наличия текста
- Style метрики зависят от `enable_style_flags` и наличия текста
- `entropy_topk` и `perplexity_topk` зависят от вычисления probabilities тем

**Upstream зависимости**:
- **AudioProcessor** (опционально): предоставляет ASR транскрипцию через `doc.asr.segments[].text` (preferred)
- **VideoDocument**: предоставляет `title`, `description`
- **VideoDocument** (legacy, опционально): предоставляет `transcripts` (только если `allow_legacy_transcripts=True`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка embedding модели (`intfloat/multilingual-e5-large`) и topics DB (`topics_taxonomy_v1`), строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Topics DB**: резолвится через `dp_models` spec, содержит список тем с промптами на русском и английском языках
- **Artifacts storage**: сохраняет эмбеддинги ключевых фраз в per-run `text_processor/_artifacts/` (`tp_topics_keyphrase_embeddings.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["topics"]` для downstream extractors
- **Cache**: prompt embeddings кешируются в `default_cache_dir()/tp_topics_db/*.npy` (не `result_store`)

### Segment Policy

- Не использует сегментацию (работает с полным объединенным текстом: transcript + title + description)
- Если текста нет → valid empty (`tp_topics_present=0`, артефакты не создаются)
- Текст обрезается до `max_text_chars` (по умолчанию 20000) если превышает лимит
- Topics retrieval: cosine similarity через dot product на нормализованных векторах, агрегация prompt→topic через `max`
- Keyphrases: deterministic lightweight scorer (n-граммы 1-3 слова, фильтрация стоп-слов, score = tf × (1 / (1 + first_position)) × length_bonus)
- Приоритет транскрипции: `transcript_source_policy` (asr_only/asr_then_legacy/legacy_only)

---

## speaker_turn_embeddings_aggregator

### Краткое описание

Агрегирует эмбеддинги speaker turns в per-speaker агрегаты (mean/max). Компонент предназначен для downstream-метрик и UI-индикаторов "multi-speaker / speaker diversity", при этом соблюдает A-policy: no raw, determinism, dp_models.

**Версия**: 1.3.0  
**Категория**: embedding aggregation, speaker analysis  
**GPU**: поддерживается (cuda), опционально fp16

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_speaker_turn_embeddings_aggregator_text_npz.py`. Срез: **17** ключей `tp_spkemb_*` (`speaker_turn_embeddings_aggregator_output_v1`).

**Основные метрики** (всегда включены):
- `tp_spkemb_present`: наличие обработанных спикеров (0/1)
- `tp_spkemb_speakers_total`: общее количество спикеров
- `tp_spkemb_speakers_embedded`: количество спикеров, для которых вычислены эмбеддинги
- `tp_spkemb_turns_total`: общее количество реплик (speaker turns)

**Конфигурационные флаги**:
- `tp_spkemb_write_artifacts`: включена ли запись артефактов (0/1)
- `tp_spkemb_compute_mean`: включено ли вычисление среднего (0/1)
- `tp_spkemb_compute_max`: включено ли вычисление максимума (0/1)

**Флаги режима входа**:
- `tp_spkemb_input_present`: наличие входных данных (0/1)
- `tp_spkemb_input_mode_diar_asr`: использован режим diarization + ASR (0/1)
- `tp_spkemb_input_mode_legacy_doc_speakers`: использован legacy режим doc.speakers (0/1)
- `tp_spkemb_asr_present`: наличие ASR сегментов (0/1)
- `tp_spkemb_diar_present`: наличие diarization сегментов (0/1)

**Дополнительные метрики** (feature-gated: `emit_extra_metrics`):
- `tp_spkemb_batch_size`: размер батча
- `tp_spkemb_max_speakers`: максимальное количество спикеров
- `tp_spkemb_max_turns_per_speaker`: максимальное количество реплик на спикера
- `tp_spkemb_min_chars_per_turn`: минимальная длина реплики
- `tp_spkemb_max_chars_per_turn`: максимальная длина реплики

**Зависимости между фичами**:
- Все метрики зависят от наличия входных данных (speaker diarization или legacy doc.speakers)
- `speakers_embedded` зависит от успешного вычисления эмбеддингов для спикеров
- `turns_total` зависит от количества реплик во входных данных

**Upstream зависимости**:
- **speaker_diarization_extractor** (опционально): предоставляет `doc.speaker_diarization["speaker_segments"]` с полями `speaker_id`, `start_sec`, `end_sec`
- **asr_extractor** (опционально): предоставляет `doc.asr["segments"]` с полями `text`, `start_sec`, `end_sec`
- **VideoDocument** (legacy, опционально): предоставляет `doc.speakers` (Dict[str, Dict] со структурами `{name, description}`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`speaker_<speaker_id>_mean.npy`, `speaker_<speaker_id>_max.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["speakers"]["embeddings"]` для downstream extractors

### Segment Policy

- Не использует сегментацию (работает с полным набором speaker turns)
- Если входных данных нет → valid empty (`tp_spkemb_present=0`), fail-fast при `require_input=True`
- Два режима входа:
  - **Preferred**: diarization + ASR → сопоставление ASR-сегментов diar-сегментам по overlap во времени
  - **Legacy**: `doc.speakers` → группировка по `name`, детерминированное назначение `speaker_id` (spk000, spk001, ...)
- Фильтрация реплик: `min_chars_per_turn`, `max_chars_per_turn`, дедупликация (case-insensitive)
- Лимиты: `max_speakers` на общее количество спикеров, `max_turns_per_speaker` на количество реплик на спикера
- Агрегация: mean (среднее по всем репликам) и max (max pooling по компонентам), оба L2-нормализованы

---

## embedding_source_id_extractor

### Краткое описание

Генерирует переносимый стабильный идентификатор (`vector_id`) для primary embedding и возвращает privacy-safe метаданные для интеграции с vector store. Выбирает детерминированно primary embedding из `doc.tp_artifacts` по заданной политике приоритета.

**Версия**: 1.3.0  
**Категория**: metadata  
**GPU**: не требуется  

Док+диапазоны+валидатор: `TextProcessor/src/extractors/embedding_source_id_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_embedding_source_id_extractor_text_npz.py`. Срез: **13** ключей `tp_embid_*` (`embedding_source_id_extractor_output_v1`).

**Machine schema**: `embedding_source_id_extractor_output_v1` (**13** ключей, `allow_extra_keys: false`). Human: `TextProcessor/src/extractors/embedding_source_id_extractor/SCHEMA.md`.

### Извлекаемые фичи

**`features_flat`** (фиксированный набор, audit v3):
- `tp_embid_present`: успешно загружен конечный вектор без NaN/inf (0/1)
- `tp_embid_strict_missing_primary_enabled`: зеркало флага `strict_missing_primary`
- `tp_embid_policy_*`: one-hot политики (`transcript_first`, `title_first`, `description_first`, `title_only`, `transcript_only`)
- `tp_embid_primary_is_*`: one-hot выбранного типа источника (`transcript`, `title`, `description`); при отсутствии primary — все 0
- `tp_embid_unsafe_relpath_flag`, `tp_embid_primary_embed_missing_flag`, `tp_embid_nan_inf_flag`

**Метаданные** (в `result.embedding_source_id`, не в `features_flat`):
- `vector_id`, `vector_store_uri`, `embedding_relpath`, `primary_source`
- `model_name`: из upstream meta или отсутствует (не смешивается с версией)
- `model_version`: из upstream **`model_version`** либо fallback конфига
- `weights_digest`: из upstream или `unknown`
- при `strict_missing_primary=False` на ошибочной ветке: ключ **`error`** (`no_embedding_found` | `unsafe_relpath` | `embedding_file_missing` | `embedding_load_failed` | `embedding_empty` | `embedding_non_finite`)

Верхний уровень ответа экстрактора: **`model_name` / `model_version` / `weights_digest`** = **`null`** (дубли только во вложенном блоке).

**Зависимости между фичами**:
- `tp_embid_present` = 1 только при успешной загрузке и конечности вектора
- при `strict_missing_primary=True` отсутствие primary, unsafe path, отсутствие файла, ошибка загрузки, пустой вектор, NaN/inf → **RuntimeError**

**Upstream зависимости**:
- **title_embedder** (опционально): создаёт `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- **description_embedder** (опционально): создаёт `doc.tp_artifacts["embeddings"]["description"]["relpath"]`
- **transcript_aggregator** (опционально): создаёт `doc.tp_artifacts["transcripts"][source]["agg_mean_relpath"]` (canonical) или `doc.tp_artifacts["transcript_aggregates"][source]["agg_mean_relpath"]` (legacy)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)

### Segment Policy

- Не использует сегментацию (работает с агрегированными эмбеддингами)
- Политика выбора источника: `primary_source_policy` (`transcript_first` по умолчанию, `title_first`, `description_first`, `title_only`, `transcript_only`)
- Приоритет transcript: combined → whisper → youtube_auto (сначала canonical, затем legacy fallback)
- При `strict_missing_primary=False`: soft empty с полным **`features_flat`** и кодом **`embedding_source_id.error`** на соответствующей ветке
- `vector_id` вычисляется детерминированно по float32 значениям вектора (не зависит от путей)

---

## embedding_stats_extractor

### Краткое описание

Вычисляет статистические метрики по эмбеддингам чанков транскрипта: дисперсию эмбеддингов между чанками (L2-норма дисперсии и top-k компонентных дисперсий) и энтропию topic distribution (если доступно). Используется для анализа вариативности представления текста и смешения тем.

**Версия**: 1.2.0  
**Категория**: embedding statistics  
**GPU**: не требуется  

Док+диапазоны+валидатор: `TextProcessor/src/extractors/embedding_stats_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_embedding_stats_extractor_text_npz.py`. Срез: **39** ключей `tp_embstats_*` (`embedding_stats_extractor_output_v1`).

**Machine schema**: `embedding_stats_extractor_output_v1` (39 ключей, `allow_extra_keys: false`). Human: `TextProcessor/src/extractors/embedding_stats_extractor/SCHEMA.md`.

### Извлекаемые фичи

**Базовые статистики** (feature-gated: `enabled`):
- `tp_embstats_present`: метрики вычислены (0/1, только если `n_chunks >= min_chunks_required`)
- `tp_embstats_l2_variance`: L2-норма вектора дисперсий по компонентам (float, NaN если empty)
- `tp_embstats_topvar_1..tp_embstats_topvar_8`: фиксированные 8 слотов (экспорт до эффективного `top_k_slots` после клампа ≤ 8)
- `tp_embstats_n_chunks`: количество чанков (float)
- `tp_embstats_dim`: размерность эмбеддинга (float)

**Topic entropy метрики** (feature-gated: `compute_topic_entropy`):
- `tp_embstats_topic_entropy`: энтропия top-K topic distribution (float, NaN если topic probs отсутствуют/невалидны)
- `tp_embstats_topic_entropy_norm`: нормированная энтропия H/log(K) (float, NaN если не применимо)
- `tp_embstats_topic_perplexity`: e^H (float, NaN если не применимо)
- `tp_embstats_topic_entropy_present`: наличие topic entropy (0/1)
- `tp_embstats_topic_probs_present`: наличие topic probs (0/1)
- `tp_embstats_topic_probs_invalid_flag`: флаг невалидности topic probs (0/1)

**Source tracking**:
- `tp_embstats_source_used_whisper`, `tp_embstats_source_used_youtube_auto` (0/1): фиксированная пара; приоритет конфига фильтруется к этим ключам (`whisper` по умолчанию)

**Диагностические флаги**:
- `tp_embstats_unsafe_relpath_flag`, `tp_embstats_dim_mismatch_flag`, `tp_embstats_nan_inf_flag`, `tp_embstats_used_legacy_key_flag`

**Зависимости между фичами**:
- `tp_embstats_l2_variance` и `tp_embstats_topvar_*` зависят от наличия чанков (`n_chunks >= min_chunks_required`)
- `tp_embstats_topic_entropy` зависит от `compute_topic_entropy` и наличия `doc.tp_artifacts["topics"]["topk_distribution"]["topic_probs"]`
- Все метрики зависят от `enabled` (если `enabled=False` → `tp_embstats_disabled_by_policy=1`, метрики NaN)

**Upstream зависимости**:
- **transcript_chunk_embedder** (обязательно, если `require_chunks=True`): создаёт матрицу эмбеддингов чанков (`doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` или legacy `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]`)
- **semantics_topics_keyphrases** (опционально): создаёт `doc.tp_artifacts["topics"]["topk_distribution"]["topic_probs"]` (in-memory, если `enable_topic_distribution=true`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает матрицу эмбеддингов из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)

### Segment Policy

- Не использует сегментацию (работает с полной матрицей эмбеддингов чанков)
- Приоритет источников: `transcript_source_priority` (по умолчанию `["whisper", "youtube_auto"]`)
- Минимальное количество чанков: `min_chunks_required` (по умолчанию 2, fail-fast при `require_chunks=True` если недостаточно)
- Если чанки отсутствуют → valid empty (`tp_embstats_present=0`, метрики NaN), fail-fast при `require_chunks=True`
- Topic entropy вычисляется только если `semantics_topics_keyphrases` предоставил `topic_probs` в `doc.tp_artifacts`
- Вырожденные векторы (NaN/Inf): `tp_embstats_nan_inf_flag=1`, метрики NaN, fail-fast при `require_chunks=True`

---

## hashtag_embedder

### Краткое описание

Извлекает L2-нормализованный эмбеддинг для хештегов видео (агрегация по списку `doc.hashtags`) с использованием sentence-transformers модели, строго через `dp_models` (offline/no-network). Поддерживает батчинг, детерминированный canonicalization/лимиты, опциональный cache и per-run artifact.

**Версия**: 1.2.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

Док+диапазоны+валидатор: `TextProcessor/src/extractors/hashtag_embedder/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_hashtag_embedder_text_npz.py`. Срез: **23** ключа `tp_hashemb_*` (`hashtag_embedder_output_v1`).

### Извлекаемые фичи

**Основные метрики** (всегда включены):
- `tp_hashemb_present`: эмбеддинг вычислен (0/1)
- `tp_hashemb_dim`: размерность эмбеддинга (float)
- `tp_hashemb_tag_count`: количество уникальных тегов после canonicalization/limit (float)
- `tp_hashemb_l2_norm`: L2-норма финального эмбеддинга (float, должна быть ~1.0)

**Политики и входы**:
- `tp_hashemb_require_hashtags_enabled`: включен ли fail-fast при отсутствии хештегов (0/1)
- `tp_hashemb_disabled_by_policy_hint`: хинт от upstream TagsExtractor о том, что хештеги отключены политикой (0/1)
- `tp_hashemb_n_input_tags`: количество входных тегов до canonicalization (float)
- `tp_hashemb_n_unique_tags`: количество уникальных тегов после canonicalization (float)
- `tp_hashemb_n_tags_truncated`: количество отброшенных тегов из-за лимита `max_tags` (float)

**Feature gating**:
- `tp_hashemb_compute_enabled`: включено ли вычисление эмбеддинга (0/1)
- `tp_hashemb_write_artifact_enabled`: включена ли запись артефакта (0/1)
- `tp_hashemb_artifact_written`: был ли записан артефакт (0/1)

**Кеш**:
- `tp_hashemb_cache_enabled`: включен ли кеш (0/1)
- `tp_hashemb_cache_hit`: было ли попадание в кеш (0/1 или NaN если кеш отключен)

**Модель и устройство**:
- `tp_hashemb_model_digest_u24`: первые 24 бита digest модели (float)
- `tp_hashemb_fp16`: используется ли float16 (0/1)
- `tp_hashemb_device_cuda`: используется ли CUDA (0/1)

**Тайминги**:
- `tp_hashemb_encode_ms`: время кодирования в миллисекундах (float, NaN если не применимо)
- `tp_hashemb_agg_ms`: время агрегации в миллисекундах (float, NaN если не применимо)

**Параметры агрегации**:
- `tp_hashemb_use_frequencies`: используются ли частоты тегов как веса (0/1)
- `tp_hashemb_agg_mean`, `tp_hashemb_agg_max`, `tp_hashemb_agg_logsumexp`: one-hot флаги типа агрегации (0/1)

**Зависимости между фичами**:
- `tp_hashemb_present` зависит от `compute_embedding` и наличия хештегов после canonicalization
- `tp_hashemb_tag_count` зависит от количества уникальных тегов после canonicalization и лимита `max_tags`
- `tp_hashemb_cache_hit` зависит от `cache_enabled` (NaN если кеш отключен)
- `tp_hashemb_artifact_written` зависит от `write_artifact` и успешного сохранения

**Upstream зависимости**:
- **TagsExtractor** (опционально): формирует канонический список в `doc.hashtags` при `mutate_doc_hashtags` и (**`enable_extract_hashtags`** для inline **или** **`merge_json_hashtags`** с непустым входным JSON-списком)

**Downstream зависимости**:
- Нет явных downstream зависимостей (эмбеддинг доступен через `doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]`)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`hashtag_embedding.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["embeddings"]["hashtag"]` для downstream extractors

### Segment Policy

- Не использует сегментацию (работает с полным списком хештегов)
- Если хештегов нет → valid empty (`tp_hashemb_present=0`, артефакт не создаётся), fail-fast при `require_hashtags=True`
- Canonicalization: strip hash prefix, casefold, dedup, sort, truncate до `max_tags` и `max_tag_len`
- Агрегация: `mean` (по умолчанию), `max`, `logsumexp` с опциональным взвешиванием по частотам (`use_frequencies`)
- L2 нормализация: финальный вектор всегда L2-нормализован (норма ≈ 1.0)
- Кеширование: опциональное кеширование по SHA256(content + model_name + config) для избежания повторных вычислений
- Батчинг: поддерживает `extract_batch()` для обработки нескольких документов одновременно

---

## lexico_static_features

### Краткое описание

Извлекает детерминированные лексические/статические признаки из текстовых полей видео: заголовка (title), описания (description) и транскрипта (transcript). Компонент не использует тяжёлые NLP-модели (spaCy/langdetect) и не требует сети; любые модели должны быть отдельными extractor'ами через `dp_models`.

**Версия**: 1.2.0  
**Категория**: lexical features  
**GPU**: не требуется

Док+диапазоны+валидатор: `TextProcessor/src/extractors/lexico_static_features/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_lexico_static_features_text_npz.py`. Срез: **67** ключей `tp_lex_*` (`lexico_static_features_output_v1`).

### Извлекаемые фичи

**Presence флаги** (всегда включены):
- `tp_lex_present_title`, `tp_lex_present_description`, `tp_lex_present_transcript`: наличие полей (0/1)
- `tp_lex_present_any`: наличие хотя бы одного поля (0/1)

**Feature gating флаги** (всегда включены):
- `tp_lex_enabled`, `tp_lex_disabled_by_policy`: включен ли экстрактор (0/1)
- `tp_lex_group_title_enabled`, `tp_lex_group_description_enabled`, `tp_lex_group_transcript_enabled`: включены ли группы фич (0/1)
- `tp_lex_group_emoji_enabled`, `tp_lex_group_clickbait_enabled`: включены ли опциональные группы (0/1)

**Title фичи** (feature-gated: `enable_title`):
- `tp_lex_title_len_words`, `tp_lex_title_len_chars`: длина в словах и символах
- `tp_lex_title_avg_word_len`: средняя длина слова
- `tp_lex_title_exclamation_count`, `tp_lex_title_question_count`: количество восклицательных/вопросительных знаков
- `tp_lex_title_emoji_count`: количество эмодзи (если `enable_emoji=True`)
- `tp_lex_title_type_token_ratio`: отношение уникальных слов к общему количеству (лексическое разнообразие)
- `tp_lex_title_punctuation_ratio`: доля знаков пунктуации
- `tp_lex_title_capital_words_ratio`: доля слов в верхнем регистре
- `tp_lex_title_question_prefix_flag`: наличие вопросительных слов в начале (0/1)
- `tp_lex_title_number_presence`: наличие чисел (0/1)
- `tp_lex_title_time_mention_flag`: наличие упоминаний времени/даты (0/1)
- `tp_lex_title_clickbait_score`: оценка clickbait (0.0-1.0, если `enable_clickbait_heuristic=True`)
- `tp_lex_title_stopword_ratio`: доля стоп-слов

**Description фичи** (feature-gated: `enable_description`):
- `tp_lex_description_len_words`: длина в словах
- `tp_lex_description_num_urls`: количество URL-адресов
- `tp_lex_description_num_mentions`: количество упоминаний (@username)
- `tp_lex_description_has_timestamps_flag`: наличие временных меток (0/1)
- `tp_lex_description_emoji_count`: количество эмодзи (если `enable_emoji=True`)

**Transcript фичи** (feature-gated: `enable_transcript`):
- `tp_lex_transcript_len_words`: длина в словах
- `tp_lex_transcript_avg_sentence_len`: средняя длина предложения в словах
- `tp_lex_transcript_question_ratio`: доля вопросительных предложений
- `tp_lex_transcript_lexical_diversity`: лексическое разнообразие (отношение уникальных слов к общему количеству)
- `tp_lex_transcript_rare_word_ratio`: доля "редких" слов (длиннее 12 символов, как прокси)
- `tp_lex_transcript_stopword_ratio`: доля стоп-слов
- `tp_lex_transcript_readability_score`: прокси читаемости (avg_sentence_len / avg_word_len)
- `tp_lex_transcript_orthographic_error_rate`: доля "неправильно сформированных" токенов (прокси орфографических ошибок)
- `tp_lex_transcript_avg_token_frequency_percentile`: прокси частоты токенов (на основе нормализованной длины слова)

**Combined фичи**:
- `tp_lex_emoji_diversity`: разнообразие эмодзи (отношение уникальных эмодзи к общему количеству) по всем полям
- `tp_lex_punctuation_entropy`: энтропия распределения знаков пунктуации (title + description)
- `tp_lex_special_character_ratio`: доля специальных символов (не буквы/цифры/пробелы)
- `tp_lex_upper_lower_ratio_title`: отношение заглавных букв к строчным в заголовке
- `tp_lex_named_entity_density`: зарезервировано (всегда NaN), `tp_lex_named_entity_density_enabled=0`

**Truncation метрики**:
- `tp_lex_*_chars_used`, `tp_lex_*_chars_kept`, `tp_lex_*_truncated_flag`: метрики обрезки текста (если заданы `max_*_chars`)

**Source tracking**:
- `tp_lex_transcript_source_policy_*`: one-hot флаги политики источника транскрипта (`asr_only`, `asr_then_legacy`, `legacy_only`)
- `tp_lex_transcript_source_used_asr`, `tp_lex_transcript_source_used_legacy`, `tp_lex_transcript_source_used_none`: какой источник был использован (0/1)

**Зависимости между фичами**:
- Все title фичи зависят от `enable_title` и наличия `doc.title`
- Все description фичи зависят от `enable_description` и наличия `doc.description`
- Все transcript фичи зависят от `enable_transcript` и наличия транскрипта (ASR или legacy)
- Emoji фичи зависят от `enable_emoji` и наличия пакета `emoji` (fail-fast при `emoji_policy=required`)
- Clickbait фичи зависят от `enable_clickbait_heuristic` и `enable_title`

**Upstream зависимости**:
- **AudioProcessor** (опционально): предоставляет ASR транскрипцию через `doc.asr.segments[].text` (preferred source)
- **VideoDocument**: предоставляет `doc.title`, `doc.description`, `doc.transcripts` (legacy, опционально)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **AudioProcessor**: источник ASR транскрипции (`doc.asr.segments[]` с полем `text`)
- **VideoDocument**: схема документа с текстовыми полями
- **text_utils**: утилиты для нормализации текста (`normalize_whitespace`)

### Segment Policy

- Не использует сегментацию (работает с полными текстами полей)
- Transcript source policy: `asr_only` (по умолчанию, только `doc.asr`), `asr_then_legacy` (ASR → legacy fallback), `legacy_only` (только `doc.transcripts`)
- Если поля отсутствуют → valid empty (метрики NaN, presence flags = 0)
- Truncation: опциональная обрезка текста по `max_title_chars`, `max_description_chars`, `max_transcript_chars`
- Emoji dependency: `emoji_policy=required` → fail-fast если пакет отсутствует, `emoji_policy=optional` → валидный empty для emoji фич
- NLP модели (spaCy/langdetect) намеренно не используются (production policy: no-network + ModelManager packaging)

---

## tags_extractor

### Краткое описание

Извлекает хэштеги из заголовка и описания видео, удаляет токены `#<tag>` из текста и опционально выполняет in-memory мутации документа для downstream extractors. Поддерживает Unicode-нормализацию, детерминированный отбор и privacy-safe экспорт.

**Версия**: 1.2.0  
**Категория**: text  
**GPU**: не требуется

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/tags_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_tags_extractor_text_npz.py`. Срез: **28** базовых + **3K** `tp_tags_top{i}_*` ( **K = `top_k_slots`**; `tags_extractor_output_v1`, `allow_extra_keys: true`).

**Presence и политики** (всегда включены):
- `tp_tags_title_present`, `tp_tags_description_present`: наличие полей (0/1)
- `tp_tags_group_extract_enabled`, `tp_tags_group_mutate_clean_texts_enabled`, `tp_tags_group_mutate_hashtags_enabled`, `tp_tags_group_merge_json_hashtags_enabled`: флаги конфигурации
- `tp_tags_require_title_enabled`, `tp_tags_hashtags_disabled_by_policy`: политики обработки
- `tp_tags_export_cleaned_texts_mode_none/raw`, `tp_tags_export_hashtags_mode_none/raw/hashed`: one-hot режимы экспорта

**Counts и densities** (всегда включены):
- `tp_tags_title_hashtag_found_count`, `tp_tags_description_hashtag_found_count`, `tp_tags_hashtag_total_found_count` (inline, `total` = title+desc), `tp_tags_json_hashtag_merged_count` (только JSON), `tp_tags_hashtag_unique_count` (после merge)
- `tp_tags_title_hashtag_density_per_char`, `tp_tags_description_hashtag_density_per_char`: found / len(окна парсинга)
- `tp_tags_hashtag_avg_len`, `tp_tags_hashtag_max_len`: статистики длины хэштегов

**Усечение** (всегда включены):
- `tp_tags_title_parse_capped_flag`, `tp_tags_description_parse_capped_flag`: исходное поле длиннее `max_parse_chars` (хвост не сканировали)
- `tp_tags_title_truncated_flag`, `tp_tags_description_truncated_flag`, `tp_tags_hashtags_truncated_flag`: очищенные строки / лимит уникальных тегов

**Privacy-safe top-K слоты** (feature-gated: `top_k_slots`):
- `tp_tags_top{i}_present`, `tp_tags_top{i}_hash01`, `tp_tags_top{i}_len`: топ-K хэштегов (i=1..top_k_slots)

**Зависимости между фичами**:
- Все counts зависят от `enable_extract_hashtags`
- Top-K слоты зависят от `top_k_slots` и количества найденных хэштегов
- Truncation флаги зависят от `max_text_chars` и `max_tags_total`

**Upstream зависимости**:
- **VideoDocument**: предоставляет `doc.title` и `doc.description` (опционально, fail-fast при `require_title=True`)

**Downstream зависимости**:
- **hashtag_embedder**: использует `doc.hashtags` (создаётся через `mutate_doc_hashtags=True` и `enable_extract_hashtags=True`)
- **title_embedder/description_embedder**: могут использовать очищенные тексты из `doc.title/doc.description` (если `mutate_doc_clean_texts=True`)

### Взаимосвязи с модулями системы

- **VideoDocument**: in-memory мутации через `doc.title`, `doc.description`, `doc.hashtags` (не персистируются в result)
- **doc.tp_artifacts**: privacy-safe маркер `doc.tp_artifacts["tags"]["hashtags_disabled_by_policy"]` для downstream extractors

### Segment Policy

- Не использует сегментацию (работает с полными текстами заголовка и описания)
- Unicode нормализация: `unicode_normalization` (по умолчанию `NFKC`, можно `NONE|NFKC|NFC|NFKD|NFD`)
- Правила извлечения хэштегов: первый символ — только буквы/цифры (L/M/N), последующие — L/M/N + `_` + `-`, boundary правило (не матчим `abc#tag`)
- Нормализация тегов: `casefold()` для дедупликации
- Лимиты: `max_text_chars` (по умолчанию 5000), `max_tags_total` (по умолчанию 64), `max_tag_len` (по умолчанию 64)
- Privacy: raw outputs только при явном включении `export_*_mode` (по умолчанию `none`)
- Если заголовок отсутствует → valid empty (`tp_tags_title_present=0`), fail-fast при `require_title=True`

---

## title_embedder

### Краткое описание

Извлекает L2-нормализованные эмбеддинги для заголовков видео с использованием sentence-transformers моделей, строго через `dp_models` (offline/no-network). Поддерживает батчинг, дисковый кеш, GPU ускорение и возвращает как нормализованные векторы, так и L2-нормы необработанных векторов.

**Версия**: 1.2.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/title_embedder/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_title_embedder_text_npz.py`. Срез: **16** ключей `tp_titleemb_*` (`title_embedder_output_v1`).

**Основные метрики** (всегда включены):
- `tp_titleemb_present`: эмбеддинг вычислен (0/1)
- `tp_titleemb_dim`: размерность эмбеддинга (float)
- `tp_titleemb_norm_raw`: L2-норма необработанного вектора (NaN если `compute_raw_norm=False`)
- `tp_titleemb_l2_norm`: L2-норма нормализованного вектора (должна быть ~1.0)

**Presence и конфигурация** (всегда включены):
- `tp_titleemb_title_present`: присутствует ли заголовок в документе (0/1)
- `tp_titleemb_require_title_enabled`: включен ли fail-fast при отсутствии заголовка (0/1)
- `tp_titleemb_compute_enabled`: включено ли вычисление эмбеддинга (0/1)
- `tp_titleemb_write_artifact_enabled`: включено ли сохранение артефакта (0/1)
- `tp_titleemb_artifact_written`: был ли артефакт успешно записан (0/1)
- `tp_titleemb_cache_enabled`: включено ли кеширование (0/1)
- `tp_titleemb_cache_hit`: попадание в кеш (0/1 или NaN если кеш отключен)

**Model метрики** (всегда включены):
- `tp_titleemb_fp16`: использовался ли режим float16 (0/1)
- `tp_titleemb_device_cuda`: использовалось ли устройство CUDA (0/1)
- `tp_titleemb_model_digest_u24`: **6** шестнадцатеричных символов `weights_digest` → целое **0…0xFFFFFF** (u24)
- `tp_titleemb_compute_raw_norm` (0/1): считать ли L2-норму **до** нормировки (иначе `norm_raw` = NaN)

**Timing** (всегда включены):
- `tp_titleemb_encode_ms`: время `encode` (мс; NaN/vacuum на путях без энкода)

**Зависимости между фичами**:
- `norm_raw` зависит от `compute_raw_norm` (NaN если отключено)
- `cache_hit`: при `cache_enabled=0` — **0** (или **NaN** на путях пустого title), при **1** + успехе — 0/1
- `artifact_written` зависит от `write_artifact` и успешного сохранения

**Upstream зависимости**:
- **VideoDocument**: предоставляет заголовок через `doc.title` (str)

**Downstream зависимости**:
- **cosine_metrics_extractor**: может использовать эмбеддинг для вычисления косинусного сходства
- **embedding_pair_topk_extractor**: может использовать эмбеддинг для вычисления сходства с транскрипцией
- **title_embedding_cluster_entropy_extractor**: использует эмбеддинг для вычисления энтропии кластеров
- **title_to_hashtag_cosine_extractor**: использует эмбеддинг для вычисления сходства с хэштегами

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`title_embedding.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["embeddings"]["title"]` для downstream extractors
- **Cache**: опциональное дисковое кеширование по SHA256(content + model_name + weights_digest) в `default_cache_dir()/embed_cache/`

### Segment Policy

- Не использует сегментацию (работает с полным текстом заголовка)
- Если заголовок пустой → valid empty (`tp_titleemb_present=0`, артефакт не создаётся), fail-fast при `require_title=True`
- L2 нормализация: финальный вектор всегда L2-нормализован (норма ≈ 1.0)
- Кеширование: опциональное кеширование на диск (TTL + лимиты по количеству/размеру), атомарное сохранение через временные файлы
- Батчинг: поддерживает `extract_batch()` для обработки нескольких документов одновременно
- GPU: опциональное использование CUDA с fp16 для ускорения

---

## title_embedding_cluster_entropy_extractor

### Краткое описание

Вычисляет энтропию распределения эмбеддинга заголовка по кластерам через общую таксономию (`semantic_clusters_v1`). Проецирует title embedding через PCA, вычисляет cosine similarity к центроидам, применяет softmax с температурой к top-K и вычисляет энтропию (а также нормализованную энтропию и perplexity).

**Версия**: 1.3.0  
**Категория**: clustering, entropy  
**GPU**: не требуется (опционально FAISS для ускорения)

**Machine schema**: `title_embedding_cluster_entropy_extractor_output_v1` (24 ключа). Human: `TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md`.

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_title_embedding_cluster_entropy_extractor_text_npz.py`. Срез: **24** ключа `tp_titleclent_*` (`title_embedding_cluster_entropy_extractor_output_v1`).

**Контракт Audit v3**: ровно **24** ключа `tp_titleclent_*` (`allow_extra_keys: false`); на empty ветках метрики **NaN**, зеркала конфигурации заполняются всегда.

**Зеркала политик**: `tp_titleclent_emit_extra_metrics_enabled`, `tp_titleclent_require_title_embedding_enabled`, `tp_titleclent_use_faiss_enabled`, `tp_titleclent_require_faiss_enabled`, `tp_titleclent_export_topk_distribution_enabled`.

**Top-K (кламп)**: `tp_titleclent_schema_top_k_slots_max` (= 8), `tp_titleclent_top_k_slots_requested`, `tp_titleclent_top_k_slots` (после клампа), `tp_titleclent_top_k_slots_clamped`.

**Основные метрики**:
- `tp_titleclent_present`: успешный счёт (0/1)
- `tp_titleclent_entropy_raw`, `tp_titleclent_entropy_norm` (при K≤1 нормировка 0.0), `tp_titleclent_perplexity` (NaN если empty)
- `tp_titleclent_top_k_used`, `tp_titleclent_distinct_clusters_topk` (NaN если empty)
- `tp_titleclent_temperature`

**Presence и диагностика**:
- `tp_titleclent_title_present`, `tp_titleclent_dim_mismatch_flag`, `tp_titleclent_backend_faiss`

**Дополнительные метрики** (`emit_extra_metrics=True` и успех; иначе **NaN**):
- `tp_titleclent_n_clusters`, `tp_titleclent_model_orig_dim`, `tp_titleclent_model_reduced_dim`, `tp_titleclent_margin_top2`, `tp_titleclent_compute_ms`

**Зависимости между фичами**:
- Счётные метрики (энтропия и т.д.) зависят от наличия валидного title embedding совместимой размерности
- `entropy_norm`: при **K≤1** зафиксировано **0.0**; иначе **H / log(K)**
- `perplexity` зависит от `entropy_raw`
- `margin_top2` только при **`emit_extra_metrics=True`** и **K≥2**

**Upstream зависимости**:
- **title_embedder** (обязательно): создаёт эмбеддинг заголовка (`doc.tp_artifacts["embeddings"]["title"]["relpath"]`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка PCA и центроидов через spec (`semantic_clusters_v1`), строго локально, без сетевых загрузок
- **FAISS** (опционально): библиотека для быстрого поиска ближайших соседей (автоматический fallback на NumPy если недоступен)
- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)

### Segment Policy

- Не использует сегментацию (работает с агрегированным эмбеддингом заголовка)
- Если эмбеддинг отсутствует → valid empty (`tp_titleclent_present=0`, метрики NaN), fail-fast при `require_title_embedding=True`
- Dim mismatch: если размерность не совпадает с `orig_dim` PCA → `dim_mismatch_flag=1`, fail-fast при `require_title_embedding=True`
- Алгоритм: PCA проекция → cosine similarity к центроидам → top-K → softmax( / temperature) → entropy/perplexity
- Центроиды автоматически L2-нормализуются для корректного косинусного сходства
- Интерпретация: низкая энтропия (~0-1) = четкая принадлежность, высокая энтропия (~2+) = неопределенность/межкластерное расположение

---

## title_to_hashtag_cosine_extractor

### Краткое описание

Вычисляет косинусное сходство между эмбеддингом заголовка и эмбеддингом хэштегов видео. Загружает эмбеддинги из артефактов через relpath из `doc.tp_artifacts` (детерминированный доступ, без glob/mtime).

**Версия**: 1.2.0  
**Категория**: similarity metric  
**GPU**: не требуется

**Machine schema**: `title_to_hashtag_cosine_extractor_output_v1` (11 ключей). Human: `TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md`.

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_title_to_hashtag_cosine_extractor_text_npz.py`. Срез: **11** ключей `tp_titlehashcos_*` (`title_to_hashtag_cosine_extractor_output_v1`).

**Контракт Audit v3**: ровно **11** ключей `tp_titlehashcos_*`; **без** legacy `tp_title_hashtag_cosine_*`; **без** полей `enabled`/`disabled_by_policy` в `features_flat` (включение экстрактора — список прогона / `global_config`).

**Основные метрики**:
- `tp_titlehashcos_present`: метрика вычислена (0/1)
- `tp_titlehashcos_cosine`: косинусное сходство заголовок↔хэштеги (float, [-1, 1] или NaN)

**Зеркала opt-in fail-fast**:
- `tp_titlehashcos_require_title_embedding_enabled`, `tp_titlehashcos_require_hashtag_embedding_enabled`

**Presence**:
- `tp_titlehashcos_title_present`, `tp_titlehashcos_hashtag_present`

**Диагностика**:
- `tp_titlehashcos_unsafe_relpath_flag`: path traversal при безопасном join
- `tp_titlehashcos_title_embed_missing_flag` / `tp_titlehashcos_hashtag_embed_missing_flag`: задан relpath, путь безопасен, но файл отсутствует или не прочитан / пустой вектор
- `tp_titlehashcos_dim_mismatch_flag`, `tp_titlehashcos_zero_norm_flag`

**Зависимости между фичами**:
- `cosine` зависит от успешной загрузки обоих эмбеддингов, совпадения размерности и ненулевых норм после нормализации
- `unsafe` и per-side `embed_missing` взаимно исключают некоторые комбинации (нет double-count одной ошибки)

**Upstream зависимости**:
- **title_embedder** (опционально): создаёт эмбеддинг заголовка (`doc.tp_artifacts["embeddings"]["title"]["relpath"]`)
- **hashtag_embedder** (опционально): создаёт эмбеддинг хэштегов (`doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)

### Segment Policy

- Не использует сегментацию (работает с агрегированными эмбеддингами)
- Если эмбеддинги отсутствуют → valid empty (NaN + `*_present=0`), fail-fast при `require_*` флагах
- Алгоритм: L2-нормализация обоих векторов → скалярное произведение (dot product)
- Вырожденные векторы (норма ~0): метрики становятся `NaN` (а не 0.0), `tp_titlehashcos_zero_norm_flag=1`
- Диапазон значений: [-1, 1], где 1.0 = максимальное сходство, 0.0 = ортогональность, -1.0 = максимальное различие
- Интерпретация: высокое сходство указывает на семантическую согласованность заголовка и хэштегов

---

## topk_similar_titles_extractor

### Краткое описание

Находит топ-K наиболее похожих заголовков из статического корпуса по эмбеддингу текущего заголовка видео. Использует **FAISS HNSW** (inner product на L2-нормированных векторах; **приближённый** top-K относительно полного numpy перебора) или **numpy** fallback. Корпус загружается строго через **`dp_models`** (offline, fail-fast).

**Версия**: 1.3.0  
**Категория**: similarity search  
**GPU**: не требуется (FAISS на CPU)

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/topk_similar_titles_extractor/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_topk_similar_titles_extractor_text_npz.py`. Срез: **29** ключей `tp_topktitles_*` (`topk_similar_titles_extractor_output_v1`).

**Контракт Audit v3:** **29** фиксированных ключей `features_flat` — `DataProcessor/TextProcessor/src/extractors/topk_similar_titles_extractor/SCHEMA.md`, machine `topk_similar_titles_extractor_output_v1.json` (tier **analytics** до фиксации corpus pack).

**Основные метрики**:
- `tp_topktitles_present`: результаты найдены (0/1)
- `tp_topktitles_top1_score`, `tp_topktitles_topk_mean_score` (NaN если empty)

**Зеркала конфигурации** (в т.ч. FAISS/numpy лимиты, cache, export one-hot, `k`, размеры корпуса): см. `SCHEMA.md`.

**Диагностические флаги**:
- `tp_topktitles_unsafe_relpath_flag`
- `tp_topktitles_title_embed_missing_flag`: безопасный join, но **нет файла** или **ошибка чтения** title `.npy` (отдельно от dim mismatch / NaN / zero-norm)
- `tp_topktitles_dim_mismatch_flag`, `tp_topktitles_zero_norm_flag`, `tp_topktitles_nan_inf_flag`

**Payload** (`result.topk_similar_corpus_titles`): на всех ветках **`corpus`** (spec, digest, `backend`, и т.д.); опционально **`topk_similar_ids`** / **`topk_similar_scores`** по `export_topk_mode`.

**Зависимости между фичами**:
- Все метрики зависят от наличия валидного title embedding и корпуса
- `top1_score` и `topk_mean_score` зависят от успешного поиска
- `export_k_used` зависит от `export_topk_mode` и `max_export_k`

**Upstream зависимости**:
- **title_embedder**: создаёт эмбеддинг заголовка (`doc.tp_artifacts["embeddings"]["title"]["relpath"]`); при **`require_title_embedding=false`** отсутствие артефакта → valid empty
- **ModelManager** (`dp_models`): корпус через spec (например `similar_titles_corpus_v1`)

**Downstream зависимости**:
- Нет явных downstream зависимостей

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка корпуса (embeddings.npy + ids.json) строго локально, без сетевых загрузок
- **FAISS** (опционально): библиотека для эффективного поиска похожих векторов (автоматический fallback на NumPy если недоступен)
- **Artifacts storage**: загружает эмбеддинги из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)
- **Cache**: process-level кеш индекса/корпуса по ключу `(spec+weights_digest+backend+hnsw params)` с TTL и max_entries

### Segment Policy

- Не использует сегментацию (работает с агрегированным эмбеддингом заголовка)
- Если эмбеддинг отсутствует → valid empty (`tp_topktitles_present=0`, метрики NaN), fail-fast при `require_title_embedding=True`
- Алгоритм поиска: FAISS HNSW (inner product на L2-нормализованных векторах) или numpy fallback (матричное умножение)
- Все векторы автоматически L2-нормализуются для корректного косинусного сходства
- Экспорт результатов: `export_topk_mode` (`none`/`ids_only`/`ids_and_scores`) с лимитом `max_export_k`
- Dim mismatch: если размерность не совпадает с корпусом → `dim_mismatch_flag=1`, fail-fast при `require_title_embedding=True`
- Корпус — offline asset в `dp_models` (не создаётся этим компонентом)

---

## transcript_aggregator

### Краткое описание

Агрегирует эмбеддинги чанков транскрипта в единые векторные представления. Использует два метода агрегации: взвешенное среднее (weighted mean) с экспоненциальным затуханием и опциональными весами уверенности ASR, а также max pooling. Обрабатывает несколько источников транскрипта (whisper, youtube_auto) и создает комбинированные агрегаты.

**Версия**: 1.3.0  
**Категория**: text embeddings aggregation  
**GPU**: не требуется (только tensor операции на CPU)

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/transcript_aggregator/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_transcript_aggregator_text_npz.py`. Срез: **19** ключей `tp_tragg_*` (`transcript_aggregator_output_v1`); тайминги экстрактора в `text_features.npz` — `payload.timings_by_extractor["TranscriptAggregatorExtractor"]` (проверка `--timings`).

**Основные метрики** (всегда включены):
- `tp_tragg_present`: агрегаты вычислены (0/1)
- `tp_tragg_present_whisper`: флаг наличия whisper агрегата (0/1)
- `tp_tragg_present_youtube`: флаг наличия youtube_auto агрегата (0/1)
- `tp_tragg_present_combined`: флаг наличия комбинированного агрегата (0/1)

**Конфигурационные параметры** (всегда включены):
- `tp_tragg_decay_rate`: коэффициент экспоненциального затухания (float)
- `tp_tragg_compute_std`: включено ли вычисление std (0/1)
- `tp_tragg_compute_mean`: включено ли вычисление weighted mean (0/1)
- `tp_tragg_compute_max`: включено ли вычисление max pooling (0/1)
- `tp_tragg_compute_combined`: включено ли вычисление комбинированного агрегата (0/1)
- `tp_tragg_write_artifacts`: включена ли запись артефактов (0/1)

**Дополнительные метрики** (feature-gated: `emit_extra_metrics`):
- `tp_tragg_{source}_n_chunks`: количество чанков по источнику (float, NaN если источник отсутствует)
- `tp_tragg_{source}_mean_std`: стандартное отклонение для weighted mean (float, NaN если не вычисляется)
- `tp_tragg_{source}_max_std`: стандартное отклонение для max pooling (float, NaN если не вычисляется)

**Зависимости между фичами**:
- Все метрики зависят от наличия эмбеддингов чанков от `transcript_chunk_embedder`
- `present_combined` зависит от `compute_combined` и наличия хотя бы одного источника
- Дополнительные метрики зависят от `emit_extra_metrics` и `compute_std`

**Upstream зависимости**:
- **transcript_chunk_embedder** (обязательно): создаёт эмбеддинги чанков (`doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` или legacy `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]`)

**Downstream зависимости**:
- **cosine_metrics_extractor**: может использовать агрегированные эмбеддинги для вычисления косинусного сходства
- **embedding_source_id_extractor**: может использовать агрегированные эмбеддинги как primary embedding

### Взаимосвязи с модулями системы

- **Artifacts storage**: загружает эмбеддинги чанков из per-run `text_processor/_artifacts/` через relpath из `doc.tp_artifacts`
- **VideoDocument**: использует in-memory registry `doc.tp_artifacts` для детерминированной загрузки эмбеддингов
- **Path safety**: защита от path traversal (relpath обязан оставаться внутри `artifacts_dir`)
- **Artifacts storage**: сохраняет агрегированные векторы в per-run `text_processor/_artifacts/` (`transcript_{source}_agg_mean.npy`, `transcript_{source}_agg_max.npy`, `transcript_combined_agg_mean.npy`, `transcript_combined_agg_max.npy`)

### Segment Policy

- Не использует сегментацию (работает с полной матрицей эмбеддингов чанков)
- Если эмбеддинги отсутствуют → valid empty (`tp_tragg_present=0`, метрики NaN/0), fail-fast при `require_chunks=True`
- Алгоритмы агрегации:
  - **Weighted mean**: экспоненциальное затухание по позиции (`decay_rate`) × ASR confidence (если доступно) → нормализация весов → L2-нормализация результата
  - **Max pooling**: максимум по каждому измерению → L2-нормализация результата
- Обработка источников: независимая обработка whisper и youtube_auto, опционально комбинированный агрегат из всех источников
- ASR confidence: используется только для whisper источника (если доступно в `doc.tp_artifacts["transcripts"]["whisper"]["chunk_confidence"]`)
- Атомарная запись: сохранение через временные файлы `.tmp.npy` с последующим `os.replace()`

---

## transcript_chunk_embedder

### Краткое описание

Извлекает эмбеддинги по чанкам из транскрипта видео. Разбивает транскрипт на перекрывающиеся чанки (по предложениям или ASR segments) и генерирует векторные представления для каждого чанка с использованием sentence-transformers моделей. Поддерживает обработку нескольких источников транскрипта (whisper, youtube_auto) независимо.

**Версия**: 1.3.0  
**Категория**: text embeddings  
**GPU**: опционально (если указан `device="cuda"`)

### Извлекаемые фичи

Док+диапазоны+валидатор: `TextProcessor/src/extractors/transcript_chunk_embedder/docs/FEATURE_DESCRIPTION.md` · `.../utils/validate_transcript_chunk_embedder_text_npz.py`. Срез: **16** ключей `tp_tchunk_*` (`transcript_chunk_embedder_output_v1`); тайминг — `payload.timings_by_extractor["TranscriptChunkEmbedder"]` (поле `total`, `--timings`).

**Основные метрики** (всегда включены):
- `tp_tchunk_present`: эмбеддинги вычислены (0/1)
- `tp_tchunk_sources_count`: количество обработанных источников (float)
- `tp_tchunk_whisper_present`, `tp_tchunk_youtube_auto_present`: флаги наличия по источникам (0/1)
- `tp_tchunk_whisper_chunks`, `tp_tchunk_youtube_chunks`: количество чанков по источникам (float)
- `tp_tchunk_embedding_dim`: размерность эмбеддинга (float, NaN если empty)

**Confidence метрики** (feature-gated: `emit_confidence_metrics`):
- `tp_tchunk_conf_present`: наличие confidence для whisper источника (0/1)
- `tp_tchunk_conf_mean`, `tp_tchunk_conf_min`, `tp_tchunk_conf_max`: статистики confidence (float, NaN если не применимо)

**Дополнительные метрики** (feature-gated: `emit_extra_metrics`):
- `tp_tchunk_batch_size`: размер батча (float)
- `tp_tchunk_max_chunk_tokens_model`: максимальное количество токенов в чанке (float)
- `tp_tchunk_overlap_ratio`: коэффициент перекрытия между чанками (float)
- `tp_tchunk_max_chunks_total`: максимальное количество чанков (float)
- `tp_tchunk_cache_enabled`: включено ли кеширование (0/1)

**Зависимости между фичами**:
- Все метрики зависят от наличия транскрипта (ASR или legacy)
- Confidence метрики зависят от `emit_confidence_metrics` и наличия confidence в ASR segments
- Дополнительные метрики зависят от `emit_extra_metrics`

**Upstream зависимости**:
- **AudioProcessor** (опционально): предоставляет ASR транскрипцию через `doc.asr.segments[]` (preferred для whisper источника)
- **VideoDocument** (legacy, опционально): предоставляет `doc.transcripts["youtube_auto"]` (для youtube_auto источника)
- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей и shared tokenizer (`shared_tokenizer_v1`) строго локально

**Downstream зависимости**:
- **transcript_aggregator**: использует созданные эмбеддинги чанков для агрегации

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SentenceTransformer моделей и shared tokenizer строго локально, без сетевых загрузок
- **ModelRegistry**: переиспользование моделей между extractors
- **Artifacts storage**: сохраняет эмбеддинги в per-run `text_processor/_artifacts/` (`transcript_{source}_chunk_embeddings.npy`)
- **VideoDocument**: регистрирует relpath в `doc.tp_artifacts["transcripts"][source]` (canonical) и `doc.tp_artifacts["transcript_chunks"][source]` (legacy) для downstream extractors
- **Cache**: опциональное кеширование по privacy-safe transcript_id (hash от token_ids или текста) + weights_digest в `default_cache_dir()/transcript_embed/`

### Segment Policy

- Не использует сегментацию (работает с полным транскриптом)
- Если транскрипта нет → valid empty (`tp_tchunk_present=0`, метрики NaN/0), fail-fast при `require_asr=True` или `require_any_source=True`
- Token-aware chunking: точный подсчет токенов через `shared_tokenizer_v1` (dp_models), не приблизительный
- Chunking стратегии:
  - **whisper**: chunking по ASR segments с сохранением confidence mapping
  - **youtube_auto/другие**: sentence-based chunking (regex по `.`, `!`, `?`)
- Overlap: сохранение перекрытия между чанками (`overlap_ratio`, по умолчанию 0.15 = 15%) для сохранения контекста
- Лимиты: `max_chunk_tokens_model` (по умолчанию 256), `max_chunks_total` (по умолчанию 256, cost cap)
- L2 нормализация: все эмбеддинги нормализованы к единичной длине
- Батчинг: поддерживает `extract_batch()` для обработки нескольких документов одновременно
- Кеширование: опциональное кеширование на диск (TTL + лимиты по количеству/размеру), атомарное сохранение через временные файлы

---

## VisualProcessor

### Общее описание

VisualProcessor — процессор визуальной модальности, извлекающий визуальные признаки из кадров видео. Сохраняет результаты в per-run `result_store` в виде отдельных NPZ артефактов для каждого компонента.

### Структура модулей

**Core Components** (`core/model_process/`):
- Базовые провайдеры, извлекающие низкоуровневые признаки: `core_clip`, `core_object_detections`, `core_optical_flow`, `core_depth_midas`, `core_face_landmarks`, `ocr_extractor`, `core_identity/*`

**Modules** (`modules/`):
- Высокоуровневые модули, анализирующие признаки: `cut_detection`, `shot_quality`, `video_pacing`, `scene_classification`, `story_structure`, `emotion_face`, `detalize_face`, `behavioral`, `action_recognition`, `color_light`, `frames_composition`, `similarity_metrics`, `uniqueness`, `text_scoring`, `high_level_semantic`, `micro_emotion`, `optical_flow`

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет кадры и `frames_dir/metadata.json` (contract `video_metadata_v1`)
- **ModelManager** (`dp_models`): управление моделями (CLIP, MiDaS, MediaPipe и др.), строго локальная загрузка без сети
- **Triton**: опциональный runtime для GPU-ускоренных моделей через HTTP API

### Sampling Policy

VisualProcessor использует **Sampling Policy** (аналог Segment Policy для AudioProcessor):
- Входные данные: кадры из `frames_dir` и `metadata.json` от Segmenter
- Сегменты определяются через `frame_indices` в `metadata.json` для каждого компонента:
  - `core_clip.frame_indices`: выборка кадров для CLIP embeddings
  - `core_depth_midas.frame_indices`: выборка кадров для depth maps
  - `core_face_landmarks.frame_indices`: выборка кадров для face landmarks
- Segmenter является единственным владельцем sampling — компоненты не генерируют выборку сами
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)

---

## core_clip

### Краткое описание

Вычисляет CLIP эмбеддинги для выборки кадров (union-domain) и сохраняет их в NPZ. Дополнительно сохраняет text embeddings для фиксированных prompt-наборов, чтобы downstream компоненты могли делать zero-shot scoring без загрузки CLIP весов.

**Версия**: 2.0.0  
**Категория**: embeddings  
**GPU**: preferred (может работать на CPU, но медленнее)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_embeddings`: CLIP эмбеддинги кадров, shape `[N, D]` float32 (D=512 для CLIP ViT-B/32)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32

**Text embeddings** (всегда включены, для zero-shot scoring):
- `shot_quality_text_embeddings`: эмбеддинги промптов для оценки качества кадров
- `scene_aesthetic_text_embeddings`: эмбеддинги промптов для эстетической оценки сцен
- `scene_luxury_text_embeddings`: эмбеддинги промптов для оценки роскоши сцен
- `scene_atmosphere_text_embeddings`: эмбеддинги промптов для атмосферы сцен
- `cut_detection_transition_text_embeddings`: эмбеддинги промптов для детекции переходов
- `popularity_topic_text_embeddings`: эмбеддинги промптов для популярных тем
- `places365_text_embeddings`: эмбеддинги 365 промптов Places365 для zero-shot классификации мест

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все text embeddings вычисляются один раз и кешируются (model-agnostic, одинаковы для всех размеров модели)
- `times_s` извлекается из `union_timestamps_sec[frame_indices]` в `metadata.json`

**Upstream зависимости**:
- Нет зависимостей от других компонентов (работает независимо)

**Downstream зависимости**:
- **shot_quality**: использует `frame_embeddings` и `shot_quality_text_embeddings` для zero-shot scoring
- **scene_classification**: использует `frame_embeddings` и `places365_text_embeddings` для классификации мест
- **cut_detection**: может использовать `cut_detection_transition_text_embeddings` для детекции переходов
- **video_pacing**: использует `frame_embeddings` для анализа темпа

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка CLIP моделей (`clip_vit_b32_inprocess`, `clip_vit_l14_inprocess`) или Triton клиентов (`clip_image`, `clip_text`) строго локально, без сетевых загрузок
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_clip.frame_indices`
- **Triton** (опционально): HTTP API для GPU-ускоренной инференции (runtime=triton)
- **Cache**: кеширование text embeddings по `(prompts_version, model_name, model_version, model_size)`

### Sampling Policy

- Использует `core_clip.frame_indices` из `frames_dir/metadata.json`
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Поддерживает batch processing с гибридным батчингом (сбор кадров из всех видео → батчинг → распределение результатов)

---

## core_depth_midas

### Краткое описание

Вычисляет depth maps (карты глубины) на primary выборке кадров (union-domain) с использованием MiDaS модели и сохраняет их в NPZ. Политика: Triton-only (локальные `torch.hub` / `engine=torch` запрещены).

**Версия**: 2.0.0  
**Категория**: depth_estimation  
**GPU**: preferred (Triton inference, требует GPU memory)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `depth_maps`: карты глубины, shape `[N, out_h, out_w]` float32 (по умолчанию 384×384)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32

**Статистики по кадрам** (всегда включены):
- `depth_mean`: среднее значение глубины для каждого кадра, shape `[N]` float32
- `depth_std`: стандартное отклонение глубины для каждого кадра, shape `[N]` float32
- `depth_p05`: 5-й перцентиль глубины (по finite значениям), shape `[N]` float32
- `depth_p95`: 95-й перцентиль глубины (по finite значениям), shape `[N]` float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий

**Зависимости между фичами**:
- Все статистики зависят от `depth_maps` (вычисляются как агрегаты по каждому кадру)
- Статистики вычисляются только по finite значениям (NaN/Inf игнорируются)

**Upstream зависимости**:
- Нет зависимостей от других компонентов (работает независимо)

**Downstream зависимости**:
- **shot_quality**: использует `depth_maps` и статистики для оценки качества кадров (глубина как индикатор композиции)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): управление Triton клиентами для MiDaS (`midas_256`, `midas_384`, `midas_512`) строго локально
- **Triton**: HTTP API для GPU-ускоренной инференции (обязательный runtime, no-fallback на локальные модели)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_depth_midas.frame_indices`
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8, автоматическое преобразование BGR→RGB если `--frames-bgr`)

### Sampling Policy

- Использует `core_depth_midas.frame_indices` из `frames_dir/metadata.json`
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Поддерживает batch processing с оптимизацией (векторизованные вычисления статистик, batch inference через Triton)
- Поддерживает 2-3 пресета размера входа: `midas_256`, `midas_384` (default), `midas_512`

---

## core_face_landmarks

### Краткое описание

Извлекает landmarks лица (MediaPipe FaceMesh) по выборке кадров (union-domain). Дополнительно (опционально) может извлекать `pose` и `hands`, но face_mesh обязателен для baseline, т.к. `shot_quality` зависит от face features.

**Версия**: 2.0.0  
**Категория**: face_detection  
**GPU**: не требуется (CPU-only, MediaPipe)

### Извлекаемые фичи

**Основные фичи** (всегда включены, если `face_mesh=True`):
- `face_landmarks`: landmarks лиц, shape `[N, FACES, 468, 3]` float32 (NaN если лицо не найдено)
- `face_present`: флаги наличия лиц, shape `[N, FACES]` bool
- `has_any_face`: флаг наличия хотя бы одного лица в видео, bool
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32

**Опциональные фичи** (feature-gated: `use_pose`, `use_hands`):
- `pose_landmarks`: landmarks позы, shape `[N, 33, 3]` float32 (NaN если поза не найдена)
- `pose_present`: флаги наличия позы, shape `[N]` bool
- `has_any_pose`: флаг наличия хотя бы одной позы, bool
- `hands_landmarks`: landmarks рук, shape `[N, HANDS, 21, 3]` float32 (NaN если рука не найдена)
- `hands_present`: флаги наличия рук, shape `[N, HANDS]` bool
- `has_any_hands`: флаг наличия хотя бы одной руки, bool

**Empty reasons** (privacy-safe):
- `empty_reason`: причина empty статуса (например, `"no_faces_in_video"`)
- `face_empty_reason`, `pose_empty_reason`, `hands_empty_reason`: причины empty для каждого типа landmarks

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все landmarks зависят от успешной детекции соответствующих объектов (лиц/поз/рук)
- `has_any_*` флаги зависят от наличия хотя бы одного соответствующего объекта в видео
- Empty reasons зависят от статуса детекции

**Upstream зависимости**:
- **core_object_detections** (опционально, baseline policy): читает `detections.npz` для фильтрации кадров (анализ лица только на кадрах с детектированным классом `person`)

**Downstream зависимости**:
- **shot_quality**: использует `face_landmarks` и `face_present` для оценки качества кадров (наличие лиц как индикатор качества)
- **emotion_face**: использует `face_landmarks` для анализа эмоций
- **detalize_face**: использует `face_landmarks` для детального анализа лиц
- **behavioral**: использует `pose_landmarks` и `hands_landmarks` для поведенческого анализа
- **micro_emotion**: использует `face_landmarks` для микро-эмоций

### Взаимосвязи с модулями системы

- **MediaPipe**: библиотека для детекции landmarks (FaceMesh, Pose, Hands) — CPU-only, inprocess
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_face_landmarks.frame_indices`
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8, MediaPipe ожидает RGB)
- **core_object_detections**: опциональная зависимость для фильтрации кадров по детекции `person`

### Sampling Policy

- Использует `core_face_landmarks.frame_indices` из `frames_dir/metadata.json`
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Baseline policy: компонент может фильтровать кадры по результатам `core_object_detections` (анализ только на кадрах с `person`)
- Если лиц нет → valid empty (`status="empty"`, `empty_reason="no_faces_in_video"`, `face_landmarks` остаются NaN)
- Если `pose/hands` не детектируются → не error, но записываются соответствующие `*_empty_reason`
- Поддерживает temporal filtering (опционально) для сглаживания landmarks по времени
- Профилирование включено по умолчанию (`--enable-profiling`) с детальными таймингами стадий

---

## core_object_detections

### Краткое описание

Вычисляет детекции объектов на primary выборке кадров (union-domain) с использованием YOLO (ultralytics или Triton). Сохраняет bounding boxes, scores, class_ids и valid_mask в NPZ, а также нормализованную геометрию и frame-level агрегаты (schema v2). Поддерживает 41 класс таксономии v1.0 (person, car, phone, logo_region, text_region и др.).

**Версия**: 2.2  
**Категория**: core provider (Tier-0)  
**GPU**: preferred (может работать на CPU, но медленнее)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `boxes`: bounding boxes в формате xyxy, shape `[N, MAX_DETECTIONS, 4]` float32
- `scores`: confidence scores детекций, shape `[N, MAX_DETECTIONS]` float32
- `class_ids`: ID классов детекций, shape `[N, MAX_DETECTIONS]` int32
- `valid_mask`: маска валидных детекций (выше порога), shape `[N, MAX_DETECTIONS]` bool
- `class_names`: маппинг class_id → class_name, shape `[M]` str (формат "id:name")

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- `valid_mask` зависит от `scores` и `box_threshold` (фильтрация по порогу)
- `times_s` извлекается из `union_timestamps_sec[frame_indices]` в `metadata.json`
- Все массивы выровнены по `frame_indices` (shared sampling group)

**Upstream зависимости**:
- Нет зависимостей от других компонентов (работает независимо)

**Downstream зависимости**:
- **shot_quality**: использует `boxes`, `valid_mask`, `class_ids` для оценки качества кадров
- **cut_detection**: использует для jump-cuts heuristics
- **core_car_semantics**, **core_brand_semantics**: используют bbox proposals; tracking удален, semantic heads строят surrogate `track_ids` per-detection
- **core_place_semantics**: использует `frame_indices` для выравнивания
- **ocr_extractor**: использует bbox класса `text_region` для OCR
- **action_recognition**: использует детекции класса `person` для распознавания действий

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка YOLO моделей (`yolo11x_640_triton` для Triton или `yolo11x_41_best.pt` для ultralytics) строго локально, без сетевых загрузок
- **Triton** (опционально): HTTP API для GPU-ускоренной инференции (runtime=triton, preset: `yolo11x_320/640/960`)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_object_detections.frame_indices` (shared sampling group)
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8)

### Sampling Policy

- Использует `core_object_detections.frame_indices` из `frames_dir/metadata.json` (shared sampling group с `core_clip`, `core_depth_midas`, `core_face_landmarks`)
- Кадры определяются Segmenter через union-domain выборку
- Требования к выборке: coverage (начало/середина/конец), непрерывная кривая, min_frames=50, max_frames=1500
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Требования к разрешению: min shorter side 320px, target 640px, max useful 1080px, апскейл запрещён
- Поддерживает batch processing с гибридным батчингом (сбор кадров из всех видео → batch inference → распределение результатов)
- Если нет детекций выше порога → valid empty (`status="empty"`, `empty_reason="no_detections_above_threshold"`)

---

## core_optical_flow

### Краткое описание

Вычисляет покадровую кривую движения (optical flow) на primary выборке кадров с использованием RAFT модели через Triton. Сохраняет motion norm per second для downstream модулей (например, `video_pacing`). Политика: Triton-only (локальный режим запрещён).

**Версия**: 2.2  
**schema_version**: `core_optical_flow_npz_v3`
**Категория**: core provider (Tier-0)  
**GPU**: required (Triton inference, требует GPU memory)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `motion_norm_per_sec_mean`: средняя норма движения на секунду, shape `[N]` float32 (0 для первого кадра)
- `dt_seconds`: временной интервал между кадрами, shape `[N]` float32 (NaN для первого кадра)

**Compact per-frame flow/camera stats (Audit v3, schema v3)** — все shape `[N]` float32, `NaN` на первом кадре:
- `flow_mag_std_per_sec_norm`, `flow_mag_p95_per_sec_norm`
- `flow_dx_mean_per_sec_norm`, `flow_dy_mean_per_sec_norm`
- `flow_dir_sin_mean`, `flow_dir_cos_mean`, `flow_dir_dispersion`
- `flow_div_abs_mean`, `flow_consistency`
- `cam_affine_scale`, `cam_affine_rotation`, `cam_tx_per_sec_norm`, `cam_ty_per_sec_norm`
- `cam_shake_std_norm`, `bg_ratio`

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- `motion_norm_per_sec_mean` вычисляется как `mean(√(dx²+dy²)) / dt / max(h,w)` для каждой пары кадров
- `dt_seconds` вычисляется из `times_s` (разница между соседними кадрами)
- `times_s` извлекается из `union_timestamps_sec[frame_indices]` в `metadata.json`

**Upstream зависимости**:
- Нет зависимостей от других компонентов (работает независимо)

**Downstream зависимости**:
- **video_pacing**: использует `motion_norm_per_sec_mean` для анализа темпа видео
- **cut_detection**: использует для детекции переходов (jump-cuts)
- **optical_flow** (module): использует для анализа оптического потока

### Взаимосвязи с модулями системы

- **Triton** (обязательно): HTTP API для GPU-ускоренной инференции RAFT (модели `raft_256/384/512`, ensemble с preprocessing)
- **ModelManager** (`dp_models`): управление Triton клиентами (`raft_256_triton`, `raft_384_triton`, `raft_512_triton`) строго локально
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_optical_flow.frame_indices` (shared sampling group с `video_pacing`)
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8, NHWC для Triton)

### Sampling Policy

- Использует `core_optical_flow.frame_indices` из `frames_dir/metadata.json` (shared sampling group с `video_pacing`)
- Кадры определяются Segmenter через union-domain выборку
- Требования: `len(frame_indices) >= 2` (нужны пары кадров для вычисления flow)
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой или `len(frame_indices) < 2` → error (no-fallback policy)
- Требования к разрешению: min shorter side 320px, target 640px, max useful 1080px, апскейл запрещён
- Поддерживает batch processing с оптимизацией (векторизованные вычисления magnitude, предвычисление dt)
- Поддерживает 2-3 пресета размера входа: `raft_256` (default, быстрее), `raft_384` (баланс), `raft_512` (высокое качество, требует больше VRAM)
- Empty недопустим (любая невозможность посчитать кривую → error)

---

## ocr_extractor

### Краткое описание

Выполняет OCR по bbox-кропам класса `text_region` из `core_object_detections`. Использует Tesseract CLI через subprocess для распознавания текста. Если tesseract не установлен → пишет valid empty artifact.

**Версия**: 0.1  
**Категория**: core provider  
**GPU**: не требуется (CPU-only, Tesseract CLI)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32 (строго = `core_object_detections.frame_indices`)
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `ocr_raw`: массив словарей с OCR результатами, object array (scalar) со значением `list[dict]`, где каждый dict содержит:
  - `frame`: int (union-domain индекс кадра)
  - `time_s`: float (временная метка)
  - `bbox`: `[x1, y1, x2, y2]` (bounding box из детекции)
  - `text_raw`: str (сырой распознанный текст)
  - `text_norm`: str (нормализованный текст)
  - `det_confidence`: float (score из `core_object_detections`)
  - `engine`: str (например, `"tesseract"`)
  - `lang`: str (язык движка, например `"eng+rus"`)

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все фичи зависят от успешной детекции `text_region` через `core_object_detections`
- `frame_indices` строго равен `core_object_detections.frame_indices` (shared sampling group)
- `times_s` извлекается из `union_timestamps_sec[frame_indices]` в `metadata.json`

**Upstream зависимости**:
- **core_object_detections** (обязательно): предоставляет bbox proposals для класса `text_region` и `frame_indices`

**Downstream зависимости**:
- **franchise_recognition**: использует OCR результаты для распознавания франшиз (опционально, через `--use-ocr-filtering`)
- **text_scoring**: использует OCR результаты для оценки текста

### Взаимосвязи с модулями системы

- **Tesseract**: системный бинарник для OCR (CLI через subprocess, не ML-модель)
- **core_object_detections**: источник bbox proposals для класса `text_region`
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_object_detections.frame_indices` (shared sampling group)
- **FrameManager**: загрузка кадров из `frames_dir` для извлечения crops

### Sampling Policy

- Использует `core_object_detections.frame_indices` из `frames_dir/metadata.json` (shared sampling group)
- Кадры определяются Segmenter через union-domain выборку
- Требования к выборке: coverage (начало/середина/конец), непрерывная кривая, min_frames=50, max_frames=1500
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `core_object_detections.frame_indices` пустой → error (no-fallback policy)
- Обрабатывает только детекции класса `text_region` из `core_object_detections`
- Cost controls: `--max-boxes-per-frame` (default: 5), `--max-total-boxes` (default: 5000), `--min-det-score` (default: 0.5)
- Если tesseract не установлен → valid empty (`status="empty"`, `empty_reason="dependency_missing"`)
- Если нет текста на всех обработанных bbox → valid empty (`status="empty"`, `empty_reason="no_text_available"`)

---

## brand_semantics

### Краткое описание

Распознает бренды и логотипы в видео через CLIP embeddings и Embedding Service. Использует bbox proposals из `core_object_detections` (класс `logo_region`/`text_region`), извлекает crops и сравнивает с базой известных брендов через Embedding Service.

**Версия**: 0.1.0  
**Категория**: semantic_head  
**GPU**: не требуется (работает через Embedding Service HTTP API)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `track_ids`: ID треков, shape `[T]` int32
- `track_topk_brand_ids`: Top-K брендов на трек, shape `[T, 5]` int32
- `track_topk_scores`: Similarity scores для треков, shape `[T, 5]` float32
- `frame_topk_brand_ids`: Top-K брендов на кадр, shape `[N, 5]` int32
- `frame_topk_scores`: Similarity scores для кадров, shape `[N, 5]` float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все фичи зависят от успешной детекции логотипов через `core_object_detections`
- Track-level фичи агрегируются из frame-level результатов по трекам
- Frame-level фичи дедуплицируются по `brand_name` (выбирается лучший similarity)

**Upstream зависимости**:
- **core_object_detections** (обязательно): предоставляет bbox proposals для классов `logo_region`/`text_region` (tracking удален; используются surrogate `track_ids`)
- **Embedding Service** (обязательно): база известных брендов (категория `brand`), модель `clip_336`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа присутствия брендов в видео)

### Взаимосвязи с модулями системы

- **Embedding Service**: HTTP API для поиска брендов (категория `brand`, модель `clip_336`), строго обязателен (fail-fast при недоступности)
- **core_object_detections**: источник bbox proposals для логотипов (tracking удален; используются surrogate `track_ids`)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_object_detections.frame_indices` (shared sampling group)
- **FrameManager**: загрузка кадров из `frames_dir` для извлечения crops

### Sampling Policy

- Использует `core_object_detections.frame_indices` из `frames_dir/metadata.json` (shared sampling group)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Обрабатывает только детекции класса `logo_region`/`text_region` из `core_object_detections`
- Выбирает лучший crop на трек по формуле: `score × area × (optional sharpness)`
- Поддерживает batch processing с гибридным батчингом (сбор детекций из всех видео → batch поиск → распределение результатов)

---

## car_semantics

### Краткое описание

Распознает автомобили в видео через retrieval в Embedding Service по bbox proposals из `core_object_detections` (default `proposal_classes="car"`). Возвращает top‑K кандидатов без threshold-gating (порог используется только для `*_is_confident_top1`), пишет deterministic label-space и `db_digest` для воспроизводимости.

**Версия**: 0.2.0  
**Категория**: semantic_head  
**GPU**: не требуется (работает через Embedding Service HTTP API)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `semantic_label_names`: label-space `"int:name"`, shape `[A]` str
- `semantic_object_ids`: UUID label-space из Embedding Service, shape `[A]` str
- `track_ids`: surrogate track ids (per-detection в baseline Audit v3), shape `[T]` int32
- `track_topk_ids`: Top-K label ids на track, shape `[T, 5]` int32
- `track_topk_scores`: similarity scores, shape `[T, 5]` float32 (NaN для missing)
- `frame_topk_ids`: Top-K label ids на кадр, shape `[N, 5]` int32
- `frame_topk_scores`: similarity scores, shape `[N, 5]` float32 (NaN для missing)
- `det_present_mask`: mask валидных det-slot’ов, shape `[N, M]` bool
- `det_topk_ids`: Top-K label ids на detection, shape `[N, M, 5]` int32
- `det_topk_scores`: similarity scores на detection, shape `[N, M, 5]` float32

**Метаданные**:
- `meta`: run identity + версии + status/empty_reason + `models_used/model_signature` + DB provenance (`db_digest`)
- `meta_json`: JSON строка meta (cross-venv safe)

**Зависимости между фичами**:
- Все фичи зависят от bbox proposals из `core_object_detections`
- Label-space строится детерминированно из Embedding Service labels (`db_digest`), id стабильны в пределах digest
- Frame-level агрегаты дедуплицируют label’ы по score

**Upstream зависимости**:
- **core_object_detections** (обязательно): предоставляет bbox proposals для класса `car`/`vehicle` (tracking удален; используются surrogate `track_ids`)
- **Embedding Service** (обязательно): база известных автомобилей (категория `car`), модель `clip_336`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа присутствия автомобилей в видео)

### Взаимосвязи с модулями системы

- **Embedding Service**: HTTP API для поиска автомобилей (категория `car`, модель `clip_336`), строго обязателен (fail-fast при недоступности)
- **core_object_detections**: источник bbox proposals для автомобилей (tracking удален; используются surrogate `track_ids`)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_object_detections.frame_indices` (shared sampling group)
- **FrameManager**: загрузка кадров из `frames_dir` для извлечения crops

### Sampling Policy

- Использует `core_object_detections.frame_indices` из `frames_dir/metadata.json` (shared sampling group)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Обрабатывает только детекции класса `car`/`vehicle` из `core_object_detections`
- Выбирает лучший crop на трек по формуле: `score × area × (optional sharpness)`
- Поддерживает batch processing с гибридным батчингом (сбор детекций из всех видео → batch поиск → распределение результатов)

---

## core_face_identity

### Краткое описание

Идентифицирует известных людей (celebrity retrieval) в видео через Embedding Service. Извлекает face crops из `core_face_landmarks` (bbox из landmarks) и сравнивает с базой известных лиц. Работает только с кадрами, где были найдены лица.

**Версия**: 0.1.0  
**Категория**: semantic_head  
**GPU**: не требуется (работает через Embedding Service HTTP API, модель ArcFace на сервере)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров (только кадры с лицами), shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `face_ids`: ID известных людей на каждом кадре, shape `[N, K]` int32 (-1 если нет результата)
- `face_names`: имена известных людей на каждом кадре, shape `[N, K]` str (пустая строка если нет)
- `face_similarities`: Similarity scores, shape `[N, K]` float32 (0.0 если нет результата)

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все фичи зависят от успешной детекции лиц через `core_face_landmarks`
- `frame_indices` фильтруется по `face_present` (только кадры с лицами)
- Результаты дедуплицируются по имени (выбирается лучший similarity для каждого имени)

**Upstream зависимости**:
- **core_face_landmarks** (обязательно): предоставляет `face_landmarks` и `face_present` для извлечения bbox и фильтрации кадров
- **Embedding Service** (обязательно): база известных людей (категория `face`), модель `arcface`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа присутствия известных людей в видео)

### Взаимосвязи с модулями системы

- **Embedding Service**: HTTP API для поиска лиц (категория `face`, модель `arcface`), строго обязателен (fail-fast при недоступности)
- **core_face_landmarks**: источник face landmarks и `face_present` для извлечения bbox и фильтрации кадров
- **Segmenter**: предоставляет кадры и `metadata.json` с `union_timestamps_sec`
- **FrameManager**: загрузка кадров из `frames_dir` для извлечения face crops

### Sampling Policy

- Читает `frame_indices` из `core_face_landmarks/landmarks.npz` (не из metadata)
- Фильтрует `frame_indices` по `face_present` — оставляет только кадры, где найдены лица
- Если лиц в видео нет → valid empty (`status="empty"`, `empty_reason="no_faces_in_video"`)
- Если `frame_indices` или `face_present` отсутствуют в landmarks.npz → error (no-fallback policy)
- Обрабатывает только кадры с лицами (выходной NPZ содержит только такие кадры)
- Извлекает bbox из landmarks с padding 5% для каждого лица
- Поддерживает batch processing с гибридным батчингом (сбор лиц из всех видео → batch поиск → распределение результатов)

---

## place_semantics

### Краткое описание

Распознает места и лэндмарки в видео через CLIP embeddings и Embedding Service. Группирует кадры по местам в tracks (временная сегментация) и возвращает per-track и per-frame top-K идентификаций мест.

**Версия**: 0.1.0  
**Категория**: semantic_head  
**GPU**: не требуется (работает через Embedding Service HTTP API)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `track_ids`: ID треков (отдельные tracks для разных мест), shape `[T]` int32
- `track_topk_ids`: Top-K мест на трек, shape `[T, K]` int32
- `track_topk_scores`: Similarity scores для треков, shape `[T, K]` float32
- `track_present_mask`: маска присутствия треков, shape `[T]` bool
- `track_is_confident_top1`: флаг уверенности для top-1 места на трек, shape `[T]` bool
- `frame_topk_ids`: Top-K мест на кадр, shape `[N, K]` int32
- `frame_topk_scores`: Similarity scores для кадров, shape `[N, K]` float32
- `frame_is_confident_top1`: флаг уверенности для top-1 места на кадр, shape `[N]` bool
- `semantic_label_names`: массив строк "id:name" для маппинга label_id → place_name, shape `[A]` str
- `threshold_per_label_arr`: пороги для каждого места, shape `[A]` float32 (NaN если нет)

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все фичи зависят от успешного поиска мест через Embedding Service
- Track-level фичи группируются из frame-level результатов по временной сегментации
- Confidence flags зависят от `similarity_threshold` (threshold-based, не гейтит top-K)

**Upstream зависимости**:
- **core_object_detections** (опционально, для shared sampling group): используется `core_object_detections.frame_indices` из metadata
- **Embedding Service** (обязательно): база известных мест (категория `place`), модель `clip_448`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа мест в видео)

### Взаимосвязи с модулями системы

- **Embedding Service**: HTTP API для поиска мест (категория `place`, модель `clip_448`), строго обязателен (fail-fast при недоступности)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_object_detections.frame_indices` (shared sampling group)
- **FrameManager**: загрузка кадров из `frames_dir` для отправки в Embedding Service

### Sampling Policy

- Использует `core_object_detections.frame_indices` из `frames_dir/metadata.json` (shared sampling group)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Требования к выборке: coverage (начало/середина/конец), непрерывная кривая, min_frames=50, max_frames=2000
- Группировка кадров в tracks: объединение кадров с одинаковым top-1 местом, объединение треков при разрыве ≤ `max_gap_sec`
- Поддерживает batch processing (оптимизация через batch API Embedding Service, если доступен)

---

## franchise_recognition

### Краткое описание

Распознает конкретные франшизы/тайтлы в видео (игры, аниме, мультфильмы) через CLIP frame embeddings и Embedding Service. Использует оптимизированный режим с локальным сравнением embeddings (10-50x ускорение) или fallback на image-based search.

**Версия**: 0.1.0  
**Категория**: semantic_head  
**GPU**: не требуется (работает через Embedding Service HTTP API, использует готовые embeddings из `core_clip`)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32 (строго = `core_clip.frame_indices`)
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `semantic_label_names`: массив строк "id:name" для маппинга label_id → franchise_name, shape `[A]` str
- `threshold_per_label_arr`: пороги для каждой франшизы, shape `[A]` float32 (NaN если нет)
- `track_ids`: ID трека (video-level aggregate), shape `[1]` int32 (=0)
- `track_present_mask`: маска присутствия трека, shape `[1]` bool
- `track_topk_ids`: Top-K франшиз на видео (max over time), shape `[1, 5]` int32
- `track_topk_scores`: Similarity scores для видео, shape `[1, 5]` float32
- `track_is_confident_top1`: флаг уверенности для top-1 франшизы, shape `[1]` bool
- `track_topk_evidence_frame_indices`: union frame index, где similarity максимальна, shape `[1, 5]` int32
- `frame_topk_ids`: Top-K франшиз на кадр, shape `[N, 5]` int32
- `frame_topk_scores`: Similarity scores для кадров, shape `[N, 5]` float32
- `frame_is_confident_top1`: флаг уверенности для top-1 франшизы на кадр, shape `[N]` bool

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), Embedding Service URL, OCR stats

**Зависимости между фичами**:
- Все фичи зависят от наличия `core_clip` frame embeddings
- Track-level фичи агрегируются из frame-level результатов (max over time)
- Confidence flags зависят от `threshold_global` (threshold-based, не гейтит top-K)
- OCR filtering (опционально) может использоваться для фильтрации кандидатов при больших базах (>500 франшиз)

**Upstream зависимости**:
- **core_clip** (обязательно): предоставляет `frame_embeddings` из `embeddings.npz`
- **Embedding Service** (обязательно): база франшиз (категория `franchise`), модель CLIP
- **ocr_extractor** (опционально): может использоваться для OCR filtering при `--use-ocr-filtering`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа франшиз в видео)

### Взаимосвязи с модулями системы

- **Embedding Service**: HTTP API для поиска франшиз (категория `franchise`), строго обязателен (fail-fast при недоступности)
- **core_clip**: источник frame embeddings (используются готовые embeddings, не загружает модель напрямую)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_clip.frame_indices`
- **Triton** (опционально, для OCR filtering): может использоваться для OCR, если включен `--use-ocr-filtering`

### Sampling Policy

- Использует `core_clip.frame_indices` из `frames_dir/metadata.json` (sampling group)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Требования к выборке: coverage (начало/середина/конец), непрерывная кривая, min_frames=10, max_frames=500
- Оптимизация: использует embeddings напрямую (получение всех franchise embeddings одним запросом, локальное сравнение через cosine similarity) — 10-50x ускорение
- Поддерживает batch processing с оптимизацией (использование embeddings напрямую для всех видео, batch search для fallback)

---

## content_domain

### Краткое описание

Определяет домен контента по кадрам (игра, аниме, мульт, live-action, screen-recording и др.) через **CLIP text-retrieval** поверх `core_clip` frame embeddings. Использует offline базу доменов (`domains.jsonl` + optional `thresholds.json`) и вычисляет text embeddings доменов через Triton (`clip_text`).

**Версия**: 0.2.0  
**Категория**: semantic_head  
**GPU**: preferred (использует CLIP text encoder через Triton)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32 (строго = `core_clip.frame_indices`)
- `times_s`: временные метки кадров в секундах, shape `[N]` float32
- `semantic_label_names`: массив строк "id:name" для маппинга label_id → domain_name, shape `[A]` str
- `threshold_per_label_arr`: пороги для каждого домена, shape `[A]` float32 (NaN если нет)
- `track_ids`: ID трека (video-level aggregate), shape `[1]` int32 (=0)
- `track_present_mask`: маска присутствия трека, shape `[1]` bool
- `track_topk_ids`: Top-K доменов на видео (max over time), shape `[1, 5]` int32
- `track_topk_scores`: Similarity scores для видео, shape `[1, 5]` float32
- `track_is_confident_top1`: флаг уверенности для top-1 домена, shape `[1]` bool
- `frame_topk_ids`: Top-K доменов на кадр, shape `[N, 5]` int32
- `frame_topk_scores`: Similarity scores для кадров, shape `[N, 5]` float32
- `frame_is_confident_top1`: флаг уверенности для top-1 домена на кадр, shape `[N]` bool

**Метаданные**:
- `meta`: run identity + версии + статус + тайминги стадий (`stage_timings_ms`) + `db_*` (информация о базе доменов) + thresholds
- `meta_json`: JSON строка meta (cross-venv safe)

**Зависимости между фичами**:
- Все фичи зависят от наличия `core_clip` frame embeddings
- Text embeddings для доменов вычисляются один раз через CLIP text encoder (Triton)
- Track-level фичи агрегируются из frame-level результатов (max over time)
- Confidence flags зависят от `confidence_threshold_top1` (и/или `thresholds.json`), **не гейтит top-K**

**Upstream зависимости**:
- **core_clip** (обязательно): предоставляет `frame_embeddings` из `embeddings.npz`
- **ModelManager** (`dp_models`): база доменов (`content_domain/v1` с `manifest.json`, `domains.jsonl`, optional `thresholds.json`)
- **Triton** (обязательно): CLIP text encoder для вычисления text embeddings доменов

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа домена контента видео)

### Взаимосвязи с модулями системы

- **core_clip**: источник frame embeddings (используются готовые embeddings, не загружает модель напрямую)
- **ModelManager** (`dp_models`): база доменов (offline asset, `content_domain/v1`)
- **Triton**: CLIP text encoder для вычисления text embeddings доменов (модель `clip_text`)
- **Segmenter**: предоставляет кадры и `metadata.json` с `core_clip.frame_indices`
- **Embedding Service** (опционально, потенциально): может быть расширен для хранения эталонных embeddings доменов

### Sampling Policy

- Использует `core_clip.frame_indices` из `frames_dir/metadata.json` (sampling group)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Алгоритм: CLIP text-retrieval через prompt ensemble (text embeddings для всех доменов вычисляются один раз, затем cosine similarity с frame embeddings)
- Поддерживает batch processing с оптимизацией (вычисление text embeddings один раз для всех видео, векторизованные вычисления cosine similarity)

---

## action_recognition

### Краткое описание

Распознавание действий людей в видео на основе архитектуры SlowFast R50 (Meta AI Research). Извлекает временные эмбеддинги и агрегированные метрики для анализа действий. Генерирует сегменты из детекций "person" (class_id=0) из `core_object_detections`, группируя последовательные кадры с person детекциями.

**Версия**: 2.0.0  
**Категория**: module  
**GPU**: preferred (может работать на CPU, но медленнее)

### Извлекаемые фичи

**Sequence Features** (для VisualTransformer, всегда включены):
- `embedding_normed_256d`: L2-нормализованные эмбеддинги для каждого клипа, shape `[num_clips, 256]` float32
- `clip_center_frame_indices`: индексы центров клипов (union-domain), shape `[num_clips]` int32
- `clip_center_times_s`: времена центров клипов, shape `[num_clips]` float32
- `temporal_jumps`: скачки между соседними клипами (L2 по normed), shape `[num_clips-1]` float32

**Aggregate Features** (для MLP/Tabular Head, всегда включены):
- `mean_embedding_norm_raw`: средняя норма raw-эмбеддингов (до проекции), float32
- `std_embedding_norm_raw`: стандартное отклонение норм raw-эмбеддингов, float32
- `max_temporal_jump`: максимальный скачок между соседними клипами, float32
- `mean_temporal_jump`: средний скачок между соседними клипами, float32
- `stability`: стабильность действий (через PCA+KMeans), float32
- `num_switches`: количество переключений между кластерами, int32
- `num_clips`: количество клипов для трека, int32
- `track_frame_count`: количество кадров в треке, int32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)

**Зависимости между фичами**:
- Все aggregate features зависят от sequence features (агрегируются из `embedding_normed_256d`)
- `temporal_jumps` вычисляются из `embedding_normed_256d` (L2 расстояние между соседними клипами)
- `stability` и `num_switches` зависят от кластеризации эмбеддингов (PCA+KMeans)

**Upstream зависимости**:
- **core_object_detections** (обязательно): предоставляет детекции класса `person` (class_id=0) для генерации сегментов

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа действий в видео)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка SlowFast R50 модели (`slowfast_r50_action_recognition`) строго локально, без сетевых загрузок
- **Segmenter**: предоставляет кадры и `metadata.json` с `action_recognition.frame_indices` (union-domain)
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8) для извлечения crops person детекций
- **core_object_detections**: источник детекций `person` для генерации сегментов (трекинг удален, модуль генерирует сегменты из последовательных кадров с person детекциями)

### Sampling Policy

- Использует `action_recognition.frame_indices` из `frames_dir/metadata.json` (union-domain)
- Кадры определяются Segmenter через union-domain выборку
- Рекомендуемая стратегия: `ease_out_power` (k=0.7, min_units=120, max_units=1600, linear_until_sec=60, cap_duration_sec=1200)
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Минимальная длительность видео: 5 сек, максимальная: 20 мин
- Генерация сегментов: модуль генерирует сегменты из детекций "person" (class_id=0), группируя последовательные кадры с person детекциями
- Если нет person детекций → valid empty (`status="empty"`, `empty_reason="no_person_detections"`)
- Поддерживает batch processing с оптимизацией (переиспользование конфигурации, параллельная обработка видео)

---

## behavioral

### Краткое описание

Комплексный анализ поведения людей в видео на основе MediaPipe landmarks. Извлекает детальные признаки жестов рук, языка тела, активности речи, вовлеченности, уверенности и признаков стресса. Работает с кадрами, где были найдены лица/тела через `core_face_landmarks`.

**Версия**: 2.0.1  
**Категория**: module  
**GPU**: не требуется (CPU-only, легковесные numpy-операции)

### Извлекаемые фичи

**Sequence Features** (для VisualTransformer, всегда включены):
- `seq_num_hands`: количество рук на кадр, shape `[N]` int32
- `seq_hands_visibility`: видимость рук (0/1), shape `[N]` int32
- `seq_hand_motion_energy`: энергия движения рук, shape `[N]` float32
- `seq_arm_openness`: открытость рук (wrist_distance / shoulder_width), shape `[N]` float32
- `seq_pose_expansion`: расширение позы (отношение площади человека к площади кадра), shape `[N]` float32
- `seq_body_lean_angle`: угол наклона тела (-1.0 до 1.0, назад → вперед), shape `[N]` float32
- `seq_balance_offset`: смещение центра масс (-1.0 до 1.0, влево → вправо), shape `[N]` float32
- `seq_shoulder_angle`: угол плеч в градусах, shape `[N]` float32
- `seq_shoulder_angle_velocity`: скорость изменения угла плеч, shape `[N]` float32
- `seq_head_position_x_norm`, `seq_head_position_y_norm`: нормализованная позиция головы (0.0-1.0), shape `[N]` float32
- `seq_head_motion_energy`: энергия движения головы, shape `[N]` float32
- `seq_head_stability`: стабильность головы (обратная к motion_energy), shape `[N]` float32
- `seq_mouth_width_norm`, `seq_mouth_height_norm`, `seq_mouth_area_norm`: нормализованные параметры рта, shape `[N]` float32
- `seq_mouth_velocity`: скорость изменения площади рта, shape `[N]` float32
- `seq_mouth_open_ratio`: соотношение открытия рта, shape `[N]` float32
- `seq_speech_activity_proxy`: прокси-метрика активности речи (0.0-1.0), shape `[N]` float32
- `seq_blink_flag`: флаг моргания (0/1), shape `[N]` int32
- `seq_blink_rate_short`: частота моргания за короткое окно, shape `[N]` float32
- `seq_self_touch_flag`: флаг self-touch жестов (0/1), shape `[N]` int32
- `seq_fidgeting_energy`: энергия ёрзания, shape `[N]` float32
- `seq_timestamp_norm`: нормализованное время (0.0-1.0), shape `[N]` float32
- `seq_gesture_prob_<gesture>`: вероятности жестов по классам (pointing, open_palm, hands_on_hips, self_touch, fist, thumbs_up, thumbs_down, victory, ok, rock, call_me, love), shape `[N]` float32

**Aggregate Features** (для MLP/Tabular Head, всегда включены):
- `avg_engagement`, `max_engagement`: средний и максимальный индекс вовлеченности, float32
- `avg_confidence`, `max_confidence`: средний и максимальный индекс уверенности, float32
- `avg_stress`, `max_stress`: средний и максимальный уровень стресса, float32
- `gesture_counts`: количество каждого типа жеста, dict[str, int]
- `gesture_rate_per_sec`: частота жестов в секунду, float32
- `gesture_entropy_mean`: средняя энтропия распределения жестов, float32
- `dominant_gesture_ratio`: доля доминирующего жеста, float32
- `gesture_switching_rate`: частота смены жестов, float32
- `avg_arm_openness`: средняя открытость рук, float32
- `avg_pose_expansion`: среднее расширение позы, float32
- `body_motion_energy_mean`, `body_motion_energy_var`: средняя энергия движения тела и её вариативность, float32
- `speech_activity_ratio`: доля времени с активной речью (>0.5), float32
- `speech_burstiness`: концентрация речевой активности, float32
- `mouth_rhythm_score`: ритмичность речи (стандартное отклонение), float32
- `hands_visibility_ratio`: доля кадров с видимыми руками, float32
- `face_visibility_ratio`: доля кадров с видимым лицом, float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)
- `frame_results`: per-frame результаты (словарь для каждого кадра)
- `aggregated`: агрегированные метрики по всему видео

**Зависимости между фичами**:
- Все sequence features зависят от успешной детекции landmarks через `core_face_landmarks`
- Aggregate features вычисляются из sequence features (агрегация по всему видео)
- Индексы вовлеченности, уверенности и стресса вычисляются на этапе агрегации из sequence features

**Upstream зависимости**:
- **core_face_landmarks** (обязательно): предоставляет `pose_landmarks`, `hands_landmarks`, `face_landmarks` и `face_present` для анализа поведения

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа поведения людей в видео)

### Взаимосвязи с модулями системы

- **core_face_landmarks**: источник landmarks (MediaPipe Pose, Hands, Face Mesh) — CPU-only, inprocess
- **Segmenter**: предоставляет кадры и `metadata.json` с `behavioral.frame_indices` (union-domain)
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8, MediaPipe ожидает RGB)

### Sampling Policy

- Использует `behavioral.frame_indices` из `frames_dir/metadata.json` (union-domain)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Обрабатывает только кадры с landmarks (кадры без landmarks заполняются NaN и отмечаются `landmarks_present=false`)
- Если landmarks отсутствуют для всех кадров → valid empty (`status="empty"`, `empty_reason="no_landmarks"`)
- Поддерживает batch processing (CPU processing, последовательная обработка каждого видео)

---

## cut_detection

### Краткое описание

Детектирует жесткие склейки (hard cuts) и мягкие переходы (soft transitions: fade/dissolve + motion transitions) на выборке кадров и создает timeline границ shots для downstream модулей. Извлекает богатый набор фичей монтажа и темпа (editing/pacing features). Модуль является частью baseline и обязателен.

**Версия**: 2.0.0  
**Категория**: module (Tier-0 baseline)  
**GPU**: опционально (для CLIP embeddings через Triton, по умолчанию CPU-only heuristics)

### Извлекаемые фичи

**Hard cuts (жесткие склейки)**:
- `hard_cuts_count`: количество жестких склеек, int32
- `hard_cuts_per_minute`: частота жестких склеек на минуту, float32
- `hard_cut_strength_mean`, `hard_cut_strength_p25/p50/p75`: средняя сила и перцентили, float32
- `hard_cuts`: список позиций склеек в sampled sequence, list[int]

**Soft transitions (мягкие переходы)**:
- `fade_in_count`, `fade_out_count`, `dissolve_count`: количество fade-in/out и dissolve, int32
- `avg_fade_duration`: средняя длительность fade переходов, float32
- `soft_events`: список событий с `{type, start, end, duration_s}`, list[dict]

**Motion-based cuts**:
- `whip_pan_transitions_count`, `zoom_transition_count`: количество whip pan и zoom переходов, int32
- `motion_cut_intensity_score`: интенсивность motion-based cuts, float32
- `motion_cuts`: список позиций motion cuts, list[int]

**Jump cuts**:
- `jump_cuts_count`: количество jump cuts (резкие склейки в одной сцене), int32
- `jump_cut_ratio_per_minute`: частота jump cuts на минуту, float32
- `jump_cut_intensity`: интенсивность jump cuts, float32
- `jump_cuts`: список позиций jump cuts (подмножество hard_cuts), list[int]

**Shot statistics**:
- `avg_shot_length`, `median_shot_length`: средняя и медианная длина shots, float32
- `short_shots_ratio`, `long_shots_ratio`: доля коротких/длинных shots, float32
- `very_long_shots_count`: количество очень длинных shots, int32
- `shots_count`: общее количество shots, int32

**Cut interval metrics**:
- `cuts_per_minute`: общая частота всех cuts на минуту, float32
- `median_cut_interval`, `cut_interval_std`, `cut_interval_cv`: статистики интервалов между cuts, float32
- `cut_interval_entropy`: энтропия распределения интервалов, float32

**Scene grouping**:
- `scene_count`: количество сцен (если включен semantic clustering), int32
- `avg_scene_length`: средняя длина сцены, float32
- `scene_to_shot_ratio`: отношение сцен к shots, float32

**Audio alignment** (опционально):
- `audio_cut_alignment_score`: оценка выравнивания аудио и визуальных cuts, float32
- `audio_spike_cut_ratio`: доля cuts, совпадающих с аудио spikes, float32

**Edit style probabilities** (если включен CLIP):
- `edit_style_*`: вероятности различных стилей монтажа через zero-shot CLIP scoring, float32

**Model-facing dense curves** (для FeatureEncoder):
- `hist_diff[t]`: различие гистограмм между кадрами, shape `[N-1]` float32
- `ssim_drop[t]`: падение SSIM между кадрами, shape `[N-1]` float32
- `flow_mag[t]`: величина оптического потока (из `core_optical_flow`), shape `[N-1]` float32
- `hard_score[t]`: комбинированный hard-cut score, shape `[N-1]` float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `models_used[]` (если включен CLIP)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров, shape `[N]` float32

**Зависимости между фичами**:
- `hard_cuts_per_minute` зависит от `hard_cuts_count` и длительности видео
- `jump_cuts` являются подмножеством `hard_cuts` (требуют `core_face_landmarks` и `core_object_detections`)
- Shot statistics зависят от `hard_cuts` (shot boundaries определяются через cuts)
- `scene_count` зависит от включения semantic clustering (требует CLIP)
- Model-facing curves (`hist_diff`, `ssim_drop`, `flow_mag`) используются для вычисления `hard_score`

**Upstream зависимости**:
- **core_optical_flow** (обязательно, no-fallback): предоставляет `flow.npz` с `flow_mag` для детекции motion-based cuts и soft transitions. Модуль переиспользует `core_optical_flow/flow.npz` и запрещает локальное вычисление flow.
- **core_face_landmarks** (soft / quality dep): используется для jump-cuts эвристик; если отсутствует/невалиден → warning и jump-cuts отключаются (качество хуже, но модуль продолжает работу)
- **core_object_detections** (soft / quality dep): используется для jump-cuts эвристик; если отсутствует/невалиден → warning и jump-cuts отключаются (качество хуже, но модуль продолжает работу)
- **core_clip** (опционально): используется для CLIP zero-shot классификации переходов/стиля (baseline: через Triton, без локальных весов)

**Downstream зависимости**:
- **shot_quality**: использует shot boundaries из `cut_detection` для оценки качества кадров
- **video_pacing**: использует shot boundaries для анализа темпа видео
- **high_level_semantic**: использует scene structure и emotion signals (зависит от `cut_detection`)
- **scene_classification**: использует hard cut boundaries для точности классификации сцен

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка CLIP моделей через Triton spec (`clip_image_triton`) для semantic clustering и edit style scoring (baseline: строго через Triton, локальная загрузка запрещена)
- **Triton** (опционально): HTTP API для GPU-ускоренной инференции CLIP (если `use_clip=True`)
- **Segmenter**: предоставляет кадры и `metadata.json` с `cut_detection.frame_indices` и `union_timestamps_sec`
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8)
- **Audio** (опционально): `audio/audio.wav` для audio-cut alignment (auto-resolved)

### Sampling Policy

- Использует `cut_detection.frame_indices` из `frames_dir/metadata.json` (Segmenter contract)
- **Coverage goal**: равномерное покрытие всего видео для надежной детекции cuts и вычисления pacing statistics
- **Min/target/max frames**: min=400, target=800, max=1500 (start values)
- **Time axis**: `union_timestamps_sec` из `metadata.json` является source-of-truth timeline (обязательно, no-fallback)
- **Max sampling gap**: если `max(diff(times_s)) > 6.0s` → error (sampling слишком разреженный для надежной детекции)
- **Resolution**: модуль робастен к умеренному downscaling; не требует высокого разрешения per-component
- Если `frame_indices` пустой или `len(frame_indices) < 2` → error (no-fallback policy)
- Поддерживает batch processing с гибридным батчингом (оптимизация конфигурации для всех видео)

---

## detalize_face

### Краткое описание

Модульная система для детального извлечения признаков лица из видео. Использует результаты `core_face_landmarks` (MediaPipe FaceMesh) и вычисляет производные фичи лица: геометрия, поза, качество, глаза, движение, структура, чтение по губам. В отличие от базового `core_face_landmarks`, который только извлекает координаты ключевых точек, этот модуль вычисляет производные метрики и признаки для различных аспектов лица.

**Версия**: 2.0.0  
**Категория**: module  
**GPU**: не требуется (CPU-only, геометрические вычисления на landmarks)

### Извлекаемые фичи

**Geometry Module** (геометрические фичи):
- `face_bbox_area`, `face_relative_size`, `face_box_ratio`: размеры и позиция лица, float32
- `face_center_x_norm`, `face_center_y_norm`: нормализованные координаты центра лица (0-1), float32
- `face_rotation_in_frame`, `aspect_ratio_stability`: поворот и стабильность соотношения сторон, float32
- `jaw_width`, `cheekbone_width`, `forehead_height`: размеры частей лица, float32
- `face_shape_vector`: вектор формы лица (16 dims), shape `[16]` float32

**Pose Module** (поза головы):
- `yaw`, `pitch`, `roll`: углы поворота головы в градусах, float32
- `yaw_norm`, `pitch_norm`, `roll_norm`: нормализованные углы (-1..1), float32
- `head_pose_variability`, `pose_stability_score`: вариативность и стабильность позы, float32
- `head_turn_frequency`, `attention_to_camera_ratio`: частота поворотов и внимание к камере, float32
- `looking_direction_vector`: вектор направления взгляда (3D unit vector), shape `[3]` float32

**Quality Module** (качество изображения):
- `face_sharpness`, `face_noise_level`, `face_exposure_score`: резкость, шум, экспозиция, float32
- `occlusion_proxy`, `quality_proxy_score`: оценка окклюзии и общего качества, float32

**Eyes Module** (глаза):
- `eye_opening_ratio`, `eye_opening_left`, `eye_opening_right`: открытие глаз, float32
- `blink_rate`, `blink_intensity`, `blink_flag`: частота и интенсивность моргания, float32
- `gaze_vector`: вектор взгляда, shape `[3]` float32
- `gaze_at_camera_prob`: вероятность взгляда в камеру (0-1), float32
- `attention_score`: оценка внимания (0-1), float32
- `iris_position`: позиция радужки, shape `[2]` float32

**Motion Module** (движение):
- Скорость и ускорение landmarks, микро-выражения, float32
- Временные метрики движения лица, float32

**Structure Module** (структура):
- `identity_shape_vector`: сжатый вектор идентичности (hash или PCA), shape `[8-16]` float32
- Mesh векторы, выражение лица, float32

**Lip Reading Module** (чтение по губам):
- Параметры рта, речевая активность, float32
- Вероятность речи (0-1), float32

**Model-facing time-series** (aligned to Segmenter axis `detalize_face.frame_indices`):
- `face_present`: наличие лица по `core_face_landmarks`, shape `[N]` bool
- `processed_mask`: флаг, что модуль реально считал фичи на кадре (face-gated + optional internal sampling), shape `[N]` bool
- `primary_valid`: флаг, что найден primary face, shape `[N]` bool
- `face_count`: количество лиц на кадр (из core_face_landmarks), shape `[N]` float32
- `primary_tracking_id`: tracking id primary лица (`-1` если нет), shape `[N]` int32
- `primary_compact_features`: compact embedding primary лица, shape `[N,40]` float32 (0 если нет)
- `aggregated`: агрегаты по видео для baseline/tabular head, object

**Optional heuristic curves** (`primary_*`, включаются флагом `write_primary_curves=true`):
- `primary_gaze_at_camera_prob`, `primary_blink_rate`, `primary_attention_score`, `primary_quality_proxy_score`,
  `primary_face_sharpness`, `primary_occlusion_proxy`, `primary_speech_activity_prob`: shape `[N]` float32 (NaN если `processed_mask=false`)

**Per-track aggregates** (faces_agg):
- Агрегированные фичи по каждому отслеженному лицу (tracking_id), dict

**Summary statistics**:
- `axis_frames`, `frames_with_faces_total`, `frames_with_faces_processed`, `processed_frames`: статистики по кадрам, int32
- `total_faces`, `primary_faces`, `avg_faces_per_processed_face_frame`: статистики по лицам, int32/float32
- `stage_timings_ms`: тайминги стадий обработки, dict

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий, `ui_payload` (для UI рендера)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров, shape `[N]` float32

**Зависимости между фичами**:
- Все модули зависят от `core_face_landmarks` (landmarks являются входными данными)
- `primary_*` фичи вычисляются для primary face (наибольшая bbox_area)
- Временные метрики (blink_rate, motion) требуют истории кадров (окно 30 кадров ≈ 1-1.5 сек)
- `attention_score` зависит от `gaze_at_camera_prob` и `head_pose`
- `quality_proxy_score` агрегирует `face_sharpness`, `face_noise_level`, `face_exposure_score`

**Upstream зависимости**:
- **core_face_landmarks** (обязательно, no-fallback): предоставляет `landmarks.npz` с `face_landmarks`, `face_present`, `frame_indices`. Модуль обрабатывает только кадры, где `core_face_landmarks` обнаружил лица.

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа лиц в видео)

### Взаимосвязи с модулями системы

- **MediaPipe**: косвенная зависимость через `core_face_landmarks` (FaceMesh для детекции landmarks)
- **Segmenter**: косвенная зависимость через `core_face_landmarks` (кадры определяются Segmenter)
- **FrameManager**: косвенная зависимость через `core_face_landmarks` (загрузка кадров для ROI extraction, если требуется)

### Sampling Policy

- **Axis**: `detalize_face.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, union-domain)
- **Compute gating**: модуль считает фичи только для кадров, где `core_face_landmarks` обнаружил лица (`face_present=true`)
- **Internal sampling (опционально)**: можно ограничить вычисления uniform‑выборкой среди face-кадров (`max_face_frames`)
- Источник истины по времени: `union_timestamps_sec[frame_indices]` (no-fallback)
- Если лиц нет на всём axis → `status="empty"`, `empty_reason="no_faces_in_video"`
- Поддерживает batch processing (CPU processing)

---

## emotion_face

### Краткое описание

Модуль для анализа эмоций на лицах в видео с использованием модели EmoNet. Извлекает базовые эмоции Ekman (8 классов), валентность/активацию (valence/arousal), ключевые кадры, метрики качества и расширенные фичи (микроэмоции, физиологические сигналы, асимметрия лица). Модуль по умолчанию отключен в `global_config.yaml` (`emotion_face: false`).

**Версия**: 2.0.0  
**Категория**: module  
**GPU**: preferred (EmoNet inference, может работать на CPU, но медленнее)

### Извлекаемые фичи

**Базовые эмоции** (8 классов Ekman; axis-aligned):
- `emotion_probs`: вероятности эмоций для каждого кадра, shape `[N, 8]` float32 (порядок: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- `dominant_emotion_id`: ID доминирующей эмоции (0-7), shape `[N]` int8 (`-1` если кадр не обработан)
- `emotion_confidence`: уверенность в предсказании эмоции (0-1), shape `[N]` float32

**Valence/Arousal** (непрерывные значения):
- `valence`: валентность эмоции (-1..1, отрицательная=негативная, положительная=позитивная), shape `[N]` float32
- `arousal`: активация эмоции (-1..1, низкая=спокойная, высокая=возбужденная), shape `[N]` float32
- `intensity`: интенсивность эмоции (`sqrt(valence² + arousal²)`), shape `[N]` float32

**Keyframes** (ключевые кадры):
- `keyframes`: список событий `emotion_peak` / `transition` с `global_index`, `local_index`, `time_s`, `score`, list[dict]

**Метрики качества**:
- `emotion_diversity`: разнообразие эмоций (unique_dom_emotions / 8), float32
- `transition_score`: доля кадров с значительными переходами эмоций, float32
- `monotonicity_score`: монотонность эмоций (1 - normalized_std), float32
- `variance_score`: вариативность максимальных вероятностей, float32
- `significant_transitions_count`: количество значительных переходов, int32

**Расширенные фичи** (feature-gated, off by default):
- `microexpressions`: микроэмоции (если `enable_microexpressions=True`), float32
- `emotional_individuality`: эмоциональная индивидуальность (если `enable_emotional_individuality=True`), float32
- `face_asymmetry`: асимметрия лица (если `enable_face_asymmetry=True`), float32

**Multi-face support** (per-frame):
- `face_count`: количество лиц на кадр, shape `[N]` int16
- `*_faces`: массивы с shape `(N, max_faces_per_frame, ...)` и `NaN` для отсутствующих лиц (если multi-face), float32

**Маски (model-facing)**:
- `face_present`: наличие хотя бы одного лица на кадре (из core_face_landmarks), shape `[N]` bool
- `processed_mask`: кадр реально прошёл inference (stride/cap sampling среди face-кадров), shape `[N]` bool

**Агрегаты** (per-face / per-video):
- `dominant_emotion`: доминирующая эмоция (categorical/one-hot), int8
- `neutral_percentage`: доля нейтральных кадров, float32
- `valence_avg`, `arousal_avg`: средние значения (weighted by confidence), float32
- `valence_std`, `arousal_std`: стандартные отклонения, float32
- `max_monotonic_streak`: максимальная монотонная последовательность, int32
- `overall_quality_flag`: флаг качества (based on quality thresholds & confidence), bool

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `ui_payload` (для UI рендера)
- `sequence_features`: словарь с `frame_indices`, `times_s`, `valence`, `arousal`, `intensity`, `emotion_confidence`, `emotion_probs`, `dominant_emotion_id`, `face_count`
- `summary`: словарь с `stage_timings_ms`

**Зависимости между фичами**:
- `intensity` зависит от `valence` и `arousal` (`sqrt(valence² + arousal²)`)
- `dominant_emotion_id` зависит от `emotion_probs` (argmax)
- `emotion_confidence` зависит от `emotion_probs` (max_softmax * face_confidence)
- `keyframes` зависят от `intensity` и `valence/arousal` (детекция пиков и переходов)
- Метрики качества зависят от временных рядов `valence`, `arousal`, `emotion_probs`
- Агрегаты зависят от покадровых фичей (mean/std/min/max по кадрам)

**Upstream зависимости**:
- **core_face_landmarks** (обязательно, no-fallback): предоставляет `landmarks.npz` с `face_landmarks`, `face_present`, `frame_indices`. Модуль обрабатывает только кадры, где `core_face_landmarks` обнаружил лица.

**Downstream зависимости**:
- **high_level_semantic**: использует emotion signals для высокоуровневой семантики (зависит от `emotion_face`)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка EmoNet модели (`emonet_8_inprocess`) строго локально, без сетевых загрузок. Weights: `DP_MODELS_ROOT/bundled_models/visual/emonet/emonet_8.pth`
- **Segmenter**: косвенная зависимость через `core_face_landmarks` (кадры определяются Segmenter)
- **FrameManager**: загрузка кадров для ROI extraction (face crops) из `frames_dir` (RGB uint8)

### Sampling Policy

- Baseline v1 sampling определяется так:
  1) берутся **все кадры**, где `core_face_landmarks.face_present` имеет хотя бы одно лицо
  2) применяется собственная выборка **по этим кадрам**: `face_frame_stride` (по умолчанию **каждый 4-й**, т.е. stride=4)
  3) применяется cap `max_frames` (по умолчанию **200** кадров)
- **Важно**: модуль **не** делает fallback на `fps` и не генерирует sampling по времени самостоятельно
- Источник истины по времени: `times_s` строго из `metadata.json["union_timestamps_sec"][frame_indices]` (no-fallback)
- Если `core_face_landmarks` не дал ни одного кадра с лицами → `status="empty"`, `empty_reason="no_faces_in_video"`
- Поддерживает batch processing с GPU batching (кадры из всех видео собираются в батчи и обрабатываются через EmoNet одновременно)

---

## failing_module

### Краткое описание

Утилитарный модуль для демонстрации обработки опциональных компонентов, которые могут завершаться с ошибкой без остановки всего пайплайна. Модуль всегда завершается с ошибкой (exit code 2) и не выполняет реальной обработки. Предназначен только для тестирования и демонстрации механизма обработки опциональных компонентов в VisualProcessor.

**Версия**: 1.0.0  
**Категория**: utility (testing)  
**GPU**: не требуется

### Извлекаемые фичи

Модуль не извлекает фичи (всегда завершается с ошибкой).

**Upstream зависимости**:
- Нет зависимостей от других компонентов

**Downstream зависимости**:
- Нет downstream зависимостей (модуль не создаёт артефактов)

### Взаимосвязи с модулями системы

- **VisualProcessor**: модуль интегрирован с оркестратором VisualProcessor для демонстрации обработки опциональных компонентов (PR-4 evidence)

### Sampling Policy

- Модуль принимает стандартные CLI аргументы (`--frames-dir`, `--rs-path`) для совместимости с оркестратором, но не выполняет обработку
- Всегда возвращает exit code 2 (FileNotFoundError code в VisualProcessor)

---

## frames_composition

### Краткое описание

Baseline-ready модуль для извлечения композиционных признаков по кадрам (union-domain) и их агрегации на уровне видео. Извлекает классические композиционные сигналы: якоря (thirds/golden/center), баланс, симметрия, негативное пространство, сложность, ведущие линии, глубина, объекты, лица, стиль. Модуль не загружает ML-модели напрямую и использует результаты core провайдеров.

**Версия**: 2.0.1  
**Категория**: module  
**GPU**: не требуется (CPU-only, геометрические вычисления)

### Извлекаемые фичи

**Per-frame фичи** (`frame_feature_values[N,D]`, `frame_feature_names[D]`):

**Faces group**:
- `face_present`: флаг наличия лица (0/1), float32
- `face_center_x`, `face_center_y`: нормализованные координаты центра лица (0-1), float32
- `face_area_ratio`: нормализованная bbox-area в unit-square, float32

**Objects group**:
- `object_count`: количество объектов на кадр, float32
- `object_max_area_ratio`: максимальная bbox area / frame area, float32
- `object_bbox_coverage_ratio`: сумма bbox areas (capped at 1.0), float32

**Anchors group**:
- `anchor_distance`: минимальное расстояние от лица до ближайшего эстетического якоря (thirds/golden/center), нормализовано (0-1), float32
- `anchor_type_id`: категориальный ID якоря (0=thirds, 1=golden, 2=center), float32
- `thirds_alignment`: heuristic alignment score (0-1), float32

**Balance group**:
- `saliency_center_offset`: смещение "центра внимания" (saliency proxy) от центра кадра (0-1), float32

**Symmetry group**:
- `symmetry_score`: среднее horizontal/vertical корреляции (≈[-1..1]), float32
- `symmetry_h`, `symmetry_v`: горизонтальная и вертикальная симметрия, float32

**Negative space group**:
- `negative_space_ratio`: доля негативного пространства (1 - bbox_coverage_ratio) (0-1), float32
- `neg_space_balance_lr`: баланс негативного пространства слева/справа (0-1), float32

**Complexity group**:
- `edge_density`: доля edge пикселей (Canny) (0-1), float32
- `texture_entropy`: local variance mean (cheap proxy), float32
- `hue_std`: стандартное отклонение hue / 180 (0-1), float32
- `saturation_mean`: средняя насыщенность (0-1), float32

**Leading lines group**:
- `line_strength`: суммарная длина линий / площадь кадра (0-1), float32
- `line_count`: количество линий, float32
- `convergence_score`: proxy "сходимости" линий (0-1), float32
- `dominant_line_id`: категориальный ID доминирующей линии (0=horizontal, 1=vertical, 2=diagonal, 3=none), float32

**Depth group** (из `core_depth_midas`):
- `depth_mean`, `depth_std`, `depth_p05`, `depth_p95`: статистики глубины, float32

**Style group** (UI explainability):
- `style_minimalist`, `style_cinematic`, `style_vlog`, `style_product_centered`: heuristic probabilities (сумма=1), float32

**Video-level агрегаты** (`feature_values[F]`, `feature_names[F]`):
- Статистики по per-frame фичам: `{feature}__mean`, `{feature}__std`, `{feature}__p10`, `{feature}__p50`, `{feature}__p90`, `{feature}__min`, `{feature}__max`, float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров, shape `[N]` float32

**Зависимости между фичами**:
- Video-level агрегаты зависят от per-frame фичей (статистики по времени)
- `negative_space_ratio` зависит от `object_bbox_coverage_ratio` (1 - coverage)
- `anchor_distance` зависит от `face_center_x`, `face_center_y` (расстояние до якорей)
- `depth_*` фичи зависят от `core_depth_midas` (no-fallback)

**Upstream зависимости**:
- **core_object_detections** (обязательно): предоставляет `detections.npz` с объектами для `objects` группы (может быть `status="empty"` внутри провайдера, это не ошибка для модуля)
- **core_face_landmarks** (обязательно): предоставляет `landmarks.npz` с лицами для `faces` группы. Валидная пустота допустима (если нет лиц → `status="empty"`, `empty_reason="no_faces_in_video"`)
- **core_depth_midas** (обязательно, no-fallback): предоставляет `depth_midas.npz` с depth maps для `depth` группы. Должен быть `status="ok"` (по требованиям аудита)

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа композиции кадров)

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет кадры и `metadata.json` с `frames_composition.frame_indices` (Segmenter-owned)
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8)
- **core_object_detections**, **core_face_landmarks**, **core_depth_midas**: источники данных для композиционных фичей

### Sampling Policy

- Использует `frames_composition.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Рекомендованная нелинейная кривая** (Segmenter-owned, `type="ease_out_power"`): `k=0.6`, `min_units=120`, `max_units=900`, `linear_until_sec=10`, `cap_duration_sec=600`
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- Если `frame_indices` пустой → error (no-fallback policy)
- Если во всём видео нет лиц → valid empty (`status="empty"`, `empty_reason="no_faces_in_video"`, фичи не "забиваются нулями", используется NaN)
- Все зависимости должны быть выровнены по одинаковому `frame_indices` (строгая проверка, no-fallback)
- Поддерживает batch processing (CPU processing, последовательная обработка каждого видео)
- Внутренний параллелизм по кадрам (`--num-workers`, по умолчанию `max(1, min(8, os.cpu_count() or 4))`)

---

## high_level_semantic

### Краткое описание

Baseline-ready модуль для производства высокоуровневых семантических сигналов, выровненных по временной оси Visual. Предназначен для ML schema/encoder input (dense per-frame vectors + scene embeddings), analytics (simple scalar stats) и UI explainability (sparse events stream + scene timeline). Модуль не загружает ML-модели напрямую и потребляет результаты upstream компонентов (core_clip, cut_detection, emotion_face, опционально text_processor, audio extractors).

**Версия**: 2.0.0  
**Категория**: module  
**GPU**: не требуется (CPU-only агрегатор, потребляет готовые embeddings)

### Извлекаемые фичи

**Dense per-frame features** (`frame_features[N,F]`, `frame_feature_names[F]`):
- `clip_sim_prev`: cosine similarity между соседними `core_clip` embeddings (NaN на первом кадре), float32
- `clip_novelty_prev`: `1 - clip_sim_prev` (новизна семантики), float32
- `scene_pos_norm`: позиция кадра внутри сцены (0-1), float32
- `loudness_dbfs`: интерполированный `dbfs` из `loudness_extractor` на `times_s` (NaN если недоступно), float32
- `tempo_bpm`: интерполированный `bpm` из `tempo_extractor` на `times_s` (NaN если недоступно), float32
- `emo_valence`: интерполированный `valence` из `emotion_face` на `times_s` (NaN если недоступно), float32
- `emo_arousal`: интерполированный `arousal` из `emotion_face` (NaN если недоступно), float32
- `emo_intensity`: `sqrt(valence² + arousal²)` (NaN если недоступно), float32

**Scenes** (для ML + UI):
- `scene_id`: идентификатор сцены для каждого sampled кадра, shape `[N]` int32
- `scene_embeddings`: scene embedding = mean по кадрам сцены от `core_clip.frame_embeddings`, затем L2-нормализация, shape `[S, D]` float32
- `scene_start_frame_idx`, `scene_end_frame_idx`: границы сцен в union-domain, shape `[S]` int32
- `scene_start_time_s`, `scene_end_time_s`, `scene_duration_s`: временные границы сцен, shape `[S]` float32
- `scene_representative_frame_idx`: репрезентативный кадр сцены, shape `[S]` int32

**Sparse events stream** (для encoder/UI):
- `event_times_s`: временные метки событий, shape `[E]` float32
- `event_type_id`: тип события (1=hard_cut, 200=semantic_jump, 210=emotion_keyframe), shape `[E]` int16
- `event_strength`: сила события, shape `[E]` float32
- `event_frame_pos`: позиция события в sampled sequence (0..N-1), shape `[E]` int32

**Text snapshot features** (опциональная копия, если включена группа `text`):
- `text_feature_names`, `text_feature_values`: privacy-safe табличные фичи TextProcessor (удобно для downstream "single read" сценариев, но source-of-truth остаётся `text_processor/text_features.npz`)

**Метаданные (Audit v3)**:
- `meta`: baseline meta contract + строгая schema validation (`high_level_semantic_npz_v2`), тайминги стадий (`stage_timings_ms`), config highlights, best-effort `models_used/model_signature` из upstream артефактов; offline render (без CDN). `ui.event_type_map` (таксономия событий)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров, shape `[N]` float32

**Зависимости между фичами**:
- `clip_novelty_prev` зависит от `clip_sim_prev` (`1 - clip_sim_prev`)
- `emo_intensity` зависит от `emo_valence` и `emo_arousal` (`sqrt(valence² + arousal²)`)
- `scene_embeddings` зависят от `core_clip.frame_embeddings` (mean по кадрам сцены, затем L2-нормализация)
- `scene_pos_norm` зависит от `scene_id` (позиция внутри сцены)
- События `semantic_jump` зависят от `clip_novelty_prev` (top-k peaks)
- События `emotion_keyframe` зависят от `emotion_face.keyframes` (best-effort mapping)

**Upstream зависимости**:
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings` (source-of-truth). Модуль не загружает CLIP веса, только потребляет готовые embeddings
- **cut_detection** (обязательно, no-fallback): предоставляет shot boundaries + scene grouping. `cut_detection.frame_indices` должен точно совпадать с `high_level_semantic.frame_indices`
- **emotion_face** (обязательно, no-fallback): предоставляет `emotion_face.npz` с emotion timeline (mapped to union time axis by time interpolation)
- **text_processor** (опционально, по умолчанию best-effort): предоставляет `text_features.npz` (privacy-safe табличные фичи). Может быть включено через `--require-text-processor`
- **loudness_extractor** (опционально, по умолчанию best-effort): предоставляет `loudness_extractor/*.npz` для `loudness_dbfs`. Может быть включено через `--require-audio-loudness`
- **tempo_extractor** (опционально, по умолчанию best-effort): предоставляет `tempo_extractor/*.npz` для `tempo_bpm`. Может быть включено через `--require-audio-tempo`
- **clap_extractor** (опционально, по умолчанию best-effort): предоставляет `clap_extractor/*.npz`. Может быть включено через `--require-audio-clap`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для ML schema/encoder input, analytics, UI explainability)

### Взаимосвязи с модулями системы

- **Segmenter**: предоставляет кадры и `metadata.json` с `high_level_semantic.frame_indices` и `union_timestamps_sec`. Segmenter должен обеспечить, что `high_level_semantic.frame_indices` равен `cut_detection.frame_indices` и покрыт `core_clip.frame_indices`
- **core_clip**: источник embeddings (Triton-backed в baseline, но модуль не загружает веса, только потребляет готовые embeddings)
- **cut_detection**: источник shot boundaries и scene grouping (no internal cut detection)
- **emotion_face**: источник emotion timeline (EmoNet in-process)
- **TextProcessor** (опционально): источник text features (privacy-safe)
- **AudioProcessor** (опционально): источники loudness, tempo, clap features

### Sampling Policy

- Модуль **не выбирает sampling**. Segmenter является единственным владельцем sampling policy
- **Unit**: `frame` в union-domain
- **Required inputs from Segmenter**: `metadata["high_level_semantic"]["frame_indices"]`, `union_timestamps_sec`
- **Alignment requirement**: Segmenter должен обеспечить, что `high_level_semantic.frame_indices` равен `cut_detection.frame_indices` и покрыт `core_clip.frame_indices`
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- Если `frame_indices` пустой → error (no-fallback policy)
- Поддерживает batch processing (`supports_batch = True`, использует дефолтную реализацию `BaseModule.process_batch()` - цикл по видео)

---

## micro_emotion

### Краткое описание

Модуль извлечения микроэмоций и Action Units (AU) из видео с использованием OpenFace через Docker. Извлекает оптимизированные признаки для анализа микроэмоций, детекции micro-expressions и генерации per-frame векторов для VisualTransformer. Модуль использует OpenFace (CMU) для анализа мимики лица и извлечения 45 Action Units, facial landmarks, head pose, gaze direction и micro-expressions.

**Версия**: 2.0.1  
**Категория**: module  
**GPU**: required (OpenFace через Docker требует GPU, device="cuda")

### Извлекаемые фичи

**Aggregate Features** (для MLP/Tabular Head; хранятся таблично как `feature_names/feature_values`):

**Ключевые Action Units** (10-14 AU: AU06, AU12, AU04, AU01, AU02, AU25, AU26, AU07, AU23, AU45, AU43, AU15, AU20, AU10):
- `{au}_intensity_mean`, `{au}_intensity_std`: средняя интенсивность и стандартное отклонение (0.0-5.0), float32
- `{au}_intensity_delta_mean`: средняя интенсивность относительно baseline, float32
- `{au}_presence_rate`: доля кадров с presence==1 (0.0-1.0), float32
- `{au}_peak_count`: количество пиков интенсивности, int32

**PCA для остальных AU**:
- `au_pca_1`, `au_pca_2`, `au_pca_3`: первые 3 PCA компоненты, float32
- `au_pca_var_explained_1..k`: доля объяснённой дисперсии для каждой компоненты, float32

**Head Pose**:
- `pose_Rx_mean`, `pose_Ry_mean`, `pose_Rz_mean`: средние значения поворотов (градусы), float32
- `pose_Rx_std`, `pose_Ry_std`, `pose_Rz_std`: стандартные отклонения поворотов, float32
- `pose_Rx_min`, `pose_Rx_max`, `pose_Ry_min`, `pose_Ry_max`: экстремальные значения, float32
- `pose_Tz_mean`, `pose_Tz_std`: приближение/удаление от камеры, float32
- `pose_stability_score`: оценка стабильности позы (0.0-1.0), float32

**Gaze Direction**:
- `gaze_x_mean`, `gaze_y_mean`: средние углы взгляда (градусы), float32
- `gaze_x_std`, `gaze_y_std`: стандартные отклонения углов, float32
- `gaze_centered_ratio`: доля кадров с взглядом в камеру (0.0-1.0), float32
- `blink_rate_per_min`: частота миганий в минуту, float32
- `eye_contact_score`: комбинированная оценка зрительного контакта (0.0-1.0), float32

**Facial Landmarks**:
- `mouth_opening_mean`, `mouth_opening_std`: открытие рта (нормализованное), float32
- `smile_width_mean`, `smile_width_std`: ширина улыбки, float32
- `face_asymmetry_score`: оценка асимметрии лица (0.0-1.0), float32
- `landmarks_pca_1..5`: первые 5 PCA компонент для landmarks, float32
- `head_depth_variation`: вариация глубины головы, float32

**Micro-expressions**:
- `microexpr_count`: количество обнаруженных micro-expressions, int32
- `microexpr_rate_per_min`: частота micro-expressions в минуту, float32
- `microexpr_max_intensity`: максимальная интенсивность, float32
- `microexpr_types_distribution`: распределение по типам (smile, surprise, frown, disgust), dict
- `microexpr_timestamps`: временные метки micro-expressions (секунды), list[float]
- `microexpr_types`: типы для каждого micro-expression, list[str]

**Видео-уровневые агрегаты**:
- `smile_ratio`: доля кадров с улыбкой, float32
- `eye_contact_ratio`: доля кадров с взглядом в камеру, float32
- `face_presence_ratio`: доля кадров с обнаруженным лицом, float32
- `avg_mouth_opening`: среднее открытие рта, float32

**Reliability Flags**:
- `au_quality_overall`: средняя уверенность AU (0.0-1.0), float32
- `au_quality_reliable`: флаг надёжности AU данных, bool
- `landmark_visibility_mean`: средняя доля видимых landmarks (0.0-1.0), float32
- `landmark_visibility_reliable`: флаг надёжности landmarks, bool
- `occlusion_flag`: флаг окклюзии лица, bool
- `lighting_flag`: флаг качества освещения, bool

**Per-Frame Features**:

**Wide Frame Features** (`frame_features[N,F]`, `frame_feature_names[F]`, F ~40-80):
- `time_norm`: нормализованное время кадра (0.0-1.0), float32
- `face_present_any`: флаг наличия лица (0.0/1.0), float32
- `{AU}_delta`: интенсивность ключевых AU относительно baseline, float32
- `pose_Rx`, `pose_Ry`, `pose_Rz`, `pose_Tz`: поза головы, float32
- `gaze_angle_x`, `gaze_angle_y`: углы взгляда, float32
- Значения `NaN` для кадров без лиц

**Compact22 Features** (`compact22[N,22]`, `compact22_feature_names[22]`):
- Компактный вектор для VisualTransformer (~22 числа): `time_norm`, `face_presence_flag`, `au12_intensity_delta`, `au6_intensity_delta`, `au4_intensity_delta`, `au25_intensity_delta`, `au25_presence_rate_short`, `blink_flag`, `pose_Ry_norm`, `pose_Rx_norm`, `gaze_centered_flag`, `gaze_x`, `gaze_y`, `mouth_opening_norm`, `face_asymmetry_score`, `microexpr_recent_count`, `au_pca_1`, `au_pca_2`, `au_pca_3`, `au_quality_flag`, float32

**Events** (sparse events stream):
- `event_times_s`: временные метки micro-expressions, shape `[E]` float32
- `event_type_id`: тип события (1=smile, 2=surprise, 3=frown, 4=disgust), shape `[E]` int16
- `event_strength`: сила события, shape `[E]` float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `ui_payload` (для UI рендера)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров, shape `[N]` float32
- `face_present_any`: флаги наличия лица на кадр, shape `[N]` bool

**Зависимости между фичами**:
- `{au}_intensity_delta` зависит от baseline subtraction (интенсивность - baseline для нейтральных кадров)
- `au_pca_*` зависят от PCA для остальных AU (не ключевых)
- `microexpr_*` зависят от детекции micro-expressions (сглаживание AU, поиск пиков)
- `gaze_centered_ratio` зависит от `gaze_x`, `gaze_y` (взгляд в камеру если `|gaze_x| < 10°` и `|gaze_y| < 10°`)
- `blink_rate_per_min` зависит от AU45 presence
- `compact22` зависит от wide frame features (компактное представление)
- Video-level агрегаты зависят от per-frame фичей (mean/std/min/max по времени)

**Upstream зависимости**:
- **core_face_landmarks** (обязательно, опционально для фильтрации): предоставляет `landmarks.npz` с `face_present` для фильтрации кадров перед запуском OpenFace (если `use_face_detection=True`). Модуль запускает OpenFace только на кадрах с лицами, но выход остаётся выровненным по primary indices

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа микроэмоций и мимики)

### Взаимосвязи с модулями системы

- **Docker**: модуль требует установленный и запущенный Docker daemon для запуска OpenFace контейнера
- **OpenFace Image**: требует загруженный образ `openface/openface:latest` (GPU-only)
- **core_face_landmarks**: опциональная зависимость для фильтрации кадров по `face_present` (оптимизация обработки)
- **Segmenter**: предоставляет кадры и `metadata.json` с `micro_emotion.frame_indices`

### Sampling Policy

- Использует `micro_emotion.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- **Опциональная фильтрация**: если `use_face_detection=True` (CLI default), модуль фильтрует кадры по `core_face_landmarks.face_present` перед запуском OpenFace (запускает OpenFace только на кадрах с лицами), но выход остаётся выровненным по primary indices
- Если `frame_indices` пустой → error (no-fallback policy)
- Поддерживает batch processing (GPU batching через Docker, обработка кадров батчами через OpenFace)

---

## color_light

### Краткое описание

Комплексный анализ цвета и освещения видео на трех уровнях: frame-level (компактные нормализованные признаки для VisualTransformer), scene-level (агрегированные признаки по сценам) и video-level (глобальные метрики стиля и эстетики). Извлекает цветовые характеристики в HSV и Lab, доминантные цвета, оценку освещения, стили цветокоррекции и эстетические оценки.

**Версия**: 2.0.2  
**Категория**: module  
**GPU**: не требуется (CPU-only, numpy/opencv операции; GPU используется только если подключены эстетические модели)

### Извлекаемые фичи

**Frame-level признаки** (компактный вектор для VisualTransformer, всегда включены):
- `hue_mean_norm`, `hue_std_norm`: нормализованные статистики hue (0-1), shape `[N]` float32
- `hue_entropy_weighted`: энтропия hue, взвешенная по насыщенности, shape `[N]` float32
- `sat_mean_norm`: нормализованная насыщенность (0-1), shape `[N]` float32
- `val_mean_norm`: нормализованная яркость Value (0-1), shape `[N]` float32
- `L_mean_norm`: нормализованная яркость Lab (0-1), shape `[N]` float32
- `global_contrast_norm`: нормализованный глобальный контраст (0-1), shape `[N]` float32
- `local_contrast_mean_norm`: нормализованный локальный контраст (0-1), shape `[N]` float32
- `colorfulness_norm`: нормализованный индекс цветности (0-1), shape `[N]` float32
- `skin_tone_ratio`: доля пикселей кожи (0-1), shape `[N]` float32
- `overexposed_ratio`, `underexposed_ratio`: доли пере/недоэкспонированных пикселей (0-1), shape `[N]` float32
- `vignetting_score_norm`: нормализованная оценка виньетирования (0-1), shape `[N]` float32
- `soft_light_prob`: вероятность мягкого света (0-1), shape `[N]` float32
- `dominant_lab_a_norm`, `dominant_lab_b_norm`: нормализованные координаты доминантного цвета в Lab (0-1), shape `[N]` float32

**Scene-level признаки** (всегда включены):
- Агрегированные покадровые признаки: `{feature}_mean`, `{feature}_std` для каждой числовой покадровой фичи
- `num_frames`: количество обработанных кадров, int32
- `num_frames_norm`: нормированная длина сцены (0-1), float32
- `brightness_change_speed`: средняя скорость изменения яркости, float32
- `scene_flicker_intensity`: интенсивность мерцания, float32
- `flash_events_count`: количество вспышек, int32
- `color_change_speed`: скорость изменения цвета, float32
- `color_stability`: стабильность цвета (1 / (1 + mean_color_diff)), float32
- `color_pattern_periodicity`: периодичность цветовых паттернов, float32
- `scene_contrast`: средний контраст по сцене, float32
- `dynamic_range`: динамический диапазон яркости в сцене, float32

**Video-level признаки** (всегда включены):
- Агрегаты по сценам: `{feature}_mean`, `{feature}_std`, `{feature}_min`, `{feature}_max` для каждой числовой сценовой фичи
- `color_distribution_entropy`: энтропия распределения hue по всему видео, float32
- `color_distribution_gini`: коэффициент Джини для распределения оттенков, float32
- `style_teal_orange_prob`: вероятность стиля Teal & Orange (0-1), float32
- `style_film_prob`: вероятность кинематографического стиля (0-1), float32
- `style_desaturated_prob`: вероятность десатурации (0-1), float32
- `style_hyper_saturated_prob`: вероятность гипернасыщенности (0-1), float32
- `style_vintage_prob`: вероятность винтажного стиля (0-1), float32
- `style_tiktok_prob`: вероятность стиля TikTok (0-1), float32
- `global_brightness_change_speed`: глобальная скорость изменения яркости, float32
- `global_color_change_speed`: глобальная скорость изменения цвета, float32
- `strobe_transition_frequency`: частота стробоскопических переходов, float32
- `global_color_periodicity`: глобальная периодичность цветовых паттернов, float32
- `nima_mean`, `nima_std`: оценки эстетики NIMA (NaN если модель не подключена), float32
- `laion_mean`, `laion_std`: оценки эстетики LAION (NaN если модель не подключена), float32
- `cinematic_lighting_score`: оценка кинематографического освещения (NaN если модель не подключена), float32
- `professional_look_score`: оценка профессиональности кадра (NaN если модель не подключена), float32

**Sequence inputs** (для VisualTransformer):
- `sequence_inputs["frames"]`: компактные нормализованные векторы кадров, shape `[N, D_frame]` float32 (D_frame ≈ 16-18)
- `sequence_inputs["scenes"]`: опционально, сценовые векторы, shape `[N_scenes, D_scene]` float32
- `sequence_inputs["global"]`: глобальный вектор, shape `[D_global]` float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`)
- `frames`: per-scene, per-frame результаты
- `scenes`: scene-level результаты
- `video_features`: video-level результаты

**Зависимости между фичами**:
- Scene-level признаки агрегируются из frame-level признаков (mean/std по кадрам сцены)
- Video-level признаки агрегируются из scene-level признаков (mean/std/min/max по сценам)
- Стили цветокоррекции вычисляются из цветовых характеристик (hue, saturation, Lab)
- Эстетические оценки требуют подключения моделей (NIMA, LAION) — в baseline возвращают NaN

**Upstream зависимости**:
- **scene_classification** (обязательно): предоставляет информацию о сценах (`scenes` с `indices` и `scene_label`) для группировки кадров

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа цвета и освещения в видео)

### Взаимосвязи с модулями системы

- **scene_classification**: источник информации о сценах (группировка кадров по сценам)
- **Segmenter**: предоставляет кадры и `metadata.json` с `color_light.frame_indices` (union-domain)
- **FrameManager**: загрузка кадров из `frames_dir` (RGB uint8, при `color_space=BGR` кадры конвертируются)
- **ModelManager** (`dp_models`): опционально, для эстетических моделей (NIMA, LAION) — в baseline не подключены

### Sampling Policy

- Использует `color_light.frame_indices` из `frames_dir/metadata.json` (union-domain)
- Кадры определяются Segmenter через union-domain выборку
- Параметры выборки сохраняются в `metadata.json` (Segmenter contract)
- Если `frame_indices` пустой → error (no-fallback policy)
- Модуль не пересэмплирует кадры — строго использует `frame_indices`, выданные Segmenter
- Группировка по сценам: кадры группируются по сценам из `scene_classification` (scene_key = "{scene_label}__{scene_id}")
- Поддерживает batch processing (CPU processing, последовательная обработка каждого видео)

---

## optical_flow

### Краткое описание

Модуль-потребитель (consumer), обрабатывающий данные оптического потока от компонента `core_optical_flow`. Извлекает агрегированные признаки движения из предварительно вычисленных данных RAFT. Модуль не вычисляет RAFT самостоятельно, а использует готовые результаты из `core_optical_flow/flow.npz`.

**Версия**: 2.0.2  
**schema_version**: `optical_flow_npz_v3`
**Категория**: module  
**GPU**: не требуется (CPU-only, consumer модуль)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `motion_norm_per_sec_mean`: покадровая кривая движения, нормализованная на секунду (px/sec), shape `[N]` float32

**Агрегированные признаки (tabular, всегда включены)**:
- `feature_names (F,) object[str]`
- `feature_values (F,) float32`

Фиксированный набор агрегатов:
- `motion_curve_mean`
- `motion_curve_median`
- `motion_curve_p90`
- `motion_curve_variance`
- `missing_frame_ratio`

**Per-frame compact (model-facing)**:
- `frame_feature_names (D,) object[str]`
- `frame_feature_values (N,D) float32` — содержит camera‑motion + flow direction/consistency признаки (NaN на первом кадре).

**Метаданные (Audit v3)**:
- `meta`: baseline meta contract, `stage_timings_ms` в meta, `ui_payload` (для UI рендера), best-effort `models_used/model_signature` из `core_optical_flow`

**Зависимости между фичами**:
- Все агрегированные признаки зависят от `motion_norm_per_sec_mean` (статистики по кривой)
- Статистики вычисляются с использованием `nan*` функций (NaN значения игнорируются)
- Первый элемент кривой обычно равен 0.0 (нет предыдущего кадра для сравнения)

**Upstream зависимости**:
- **core_optical_flow** (обязательно, no-fallback): предоставляет `flow.npz` с `frame_indices` и `motion_norm_per_sec_mean`. Модуль является consumer и не вычисляет RAFT самостоятельно.

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа движения в видео)

### Взаимосвязи с модулями системы

- **core_optical_flow**: источник данных оптического потока (RAFT через Triton)
- **Segmenter**: предоставляет кадры и `metadata.json` с `optical_flow.frame_indices` и `union_timestamps_sec`
- **FrameManager**: не требуется напрямую (модуль работает только с данными из NPZ)

### Sampling Policy

- Использует `optical_flow.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- **Требования к согласованности**: `frame_indices` модуля должны быть подмножеством `core_optical_flow.frame_indices` (Segmenter должен обеспечивать согласованную выборку)
- Если `frame_indices` пустой → error (no-fallback policy)
- Если часть индексов отсутствует в `core_optical_flow` → используется `NaN` для отсутствующих кадров (с предупреждением в логе), и это отражается в `missing_frame_ratio`
- Поддерживает batch processing (CPU processing, последовательная обработка каждого видео)

---

## scene_classification

### Краткое описание

Модуль сегментации и классификации сцен на Places365 с CLIP-based семантикой, вычисляемой строго из `core_clip`. Группирует кадры в сцены и классифицирует их по 365 категориям мест (Places365). Поддерживает два режима label fusion: Places365 (supervised) и CLIP zero-shot.

**Версия**: 2.0.1  
**Категория**: module  
**GPU**: preferred (Places365 ResNet через Triton, CLIP text embeddings из `core_clip`)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `label_fusion`: режим fusion (`places|clip`), str
- `scenes`: словарь сцен, mapping `scene_id -> scene_dict`, где `scene_id` = `s0000`, `s0001`, ...

**Scene-level фичи** (в `scene_dict`):
- `scene_label`: Places365 label, str
- `fusion_mode`: режим выбора label (`places|clip`), str
- `indices`: список union frame indices в сцене, list[int]
- `start_frame`, `end_frame`, `length_frames`, `length_seconds`: границы сцены, int32/float32
- `start_time_s`, `end_time_s`: временные границы сцены, float32
- `mean_score`, `class_entropy_mean`, `top1_prob_mean`, `top1_vs_top2_gap_mean`, `fraction_high_confidence_frames`: Places365 агрегаты, float32
- `mean_indoor`, `mean_outdoor`, `mean_nature`, `mean_urban`: онтологические агрегаты, float32
- `mean_aesthetic_score`, `aesthetic_std`, `aesthetic_frac_high`, `mean_luxury_score`, `mean_cozy`, `mean_scary`, `mean_epic`, `mean_neutral`, `atmosphere_entropy`: CLIP семантика из `core_clip`, float32
- `scene_change_score`, `label_stability`: стабильность сцены, float32
- `dominant_places_topk_ids`, `dominant_places_topk_probs`: топ-K мест, shape `[K]` int32/float32

**Flat arrays** (для NPZ-friendly tabular access):
- `scene_ids`, `scene_label`, `start_frame`, `end_frame`, `length_frames`, `length_seconds`, `start_time_s`, `end_time_s`: плоские массивы для всех сцен

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `models_used[]` (Places365 + upstream `core_clip`), `ui_payload` (для UI рендера)

**Зависимости между фичами**:
- Все scene-level фичи зависят от группировки кадров в сцены (последовательные кадры с одинаковым predicted label)
- CLIP семантика зависит от `core_clip` text embeddings (`scene_aesthetic_text_embeddings`, `scene_luxury_text_embeddings`, `scene_atmosphere_text_embeddings`, `places365_text_embeddings`)
- `label_stability` зависит от стабильности предсказаний Places365 по кадрам сцены
- Онтологические агрегаты зависят от маппинга Places365 labels на категории (indoor/outdoor/nature/urban)

**Upstream зависимости**:
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings`, `places365_text_embeddings` (если `label_fusion=clip`), `scene_aesthetic_text_embeddings`, `scene_luxury_text_embeddings`, `scene_atmosphere_text_embeddings`
- **cut_detection** (обязательно, no-fallback): предоставляет `shot_boundaries_frame_indices` для precision segmentation (cut-aware сегментация)

**Downstream зависимости**:
- **color_light**: использует `scenes` и обрабатывает каждую сцену как уникальный сегмент

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): загрузка Places365 моделей (`places365_resnet50_224/336/448_triton`) строго локально, без сетевых загрузок
- **Triton** (обязательно): HTTP API для GPU-ускоренной инференции Places365 (ensemble с preprocessing)
- **core_clip**: источник frame embeddings и text embeddings для Places365 (CLIP zero-shot mode)
- **cut_detection**: источник hard shot boundaries для precision segmentation
- **Segmenter**: предоставляет кадры и `metadata.json` с `scene_classification.frame_indices` и `union_timestamps_sec`

### Sampling Policy

- Использует `scene_classification.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Требования к выборке**: `frame_indices` должны быть подмножеством `core_clip.frame_indices` (иначе модуль fail-fast на missing embeddings)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- **Минимальное количество кадров**: `len(frame_indices) >= 2`, иначе error (no-fallback)
- **Cut-aware segmentation**: использует hard shot boundaries из `cut_detection` для повышения точности (сцены строятся накоплением последовательных shots до `min_scene_seconds` ≥ 2.0s)
- **Label fusion**: два режима — `places` (Places365 top-1, default) и `clip` (CLIP zero-shot над теми же 365 labels, требует `core_clip.places365_text_embeddings`)
- Если `frame_indices` пустой или `len(frame_indices) < 2` → error (no-fallback policy)
- Поддерживает batch processing с внутренним батчингом (при `runtime=triton` отправляет кадры батчами `--batch-size`)

---

## shot_quality

### Краткое описание

Модуль оценки технического качества видео на уровне кадров (frame-level) и шотов (shot-level). Извлекает признаки резкости, шума, экспозиции, контраста, цвета, компрессии, глубины, объектов и лиц. Использует zero-shot CLIP scoring для оценки качества кадров через промпты.

**Версия**: 2.0.2  
**schema_version**: `shot_quality_npz_v3`
**Категория**: module  
**GPU**: не требуется (CPU-only consumer; CLIP scoring использует готовые embeddings из `core_clip`)

### Извлекаемые фичи

**Frame-level фичи** (`frame_features[N,F]`, `feature_names[F]`, всегда включены):
- Точные имена/порядок колонок — в `feature_names` внутри NPZ (source-of-truth) и в `modules/shot_quality/SCHEMA.md`.
- Включает группы: sharpness/noise/exposure/contrast/color/compression/fog/temporal/depth/objects/face ROI.
  - Lens/rolling_shutter расширения включаются только в `preset=quality`.

**Zero-shot quality probabilities** (`quality_probs[N,P]`, всегда включены):
- Вероятности zero-shot классов качества через `core_clip` (P=10 промптов), shape `[N, 10]` float16

**Shot-level агрегаты** (всегда включены):
- `shot_ids`: принадлежность каждого кадра шоту, shape `[N]` int32
- `shot_start_frame`, `shot_end_frame`: границы шотов, shape `[S]` int32
- `shot_frame_count`: количество sampled кадров в шоте, shape `[S]` int32
- `shot_features_mean/std/min/max`: агрегаты по кадрам шота, shape `[S, F]` float32

**Метаданные**:
- `meta`: run identity + версии + `stage_timings_ms` + `meta.impl_meta` (debug: prompts hashes/mappings/config + `shot_boundaries_source`) + `meta.ui_payload` (для UI)
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров, shape `[N]` float32
- `frame_feature_present_ratio`: доля finite по каждой колонке `frame_features`, shape `[F]` float32
- `shot_frame_feature_present_ratio`: доля finite внутри каждого шота по каждой колонке `frame_features`, shape `[S,F]` float32
- `shot_quality_topk_ids/probs`, `shot_quality_conf_mean`, `shot_quality_entropy_mean`: shot-level агрегаты по `quality_probs`

**Зависимости между фичами**:
- Все frame-level фичи вычисляются независимо для каждого кадра
- Shot-level агрегаты зависят от frame-level фичей (mean/std/min/max по кадрам шота)
- `quality_probs` зависят от `core_clip` frame embeddings и text embeddings (`shot_quality_text_embeddings`)
- Depth фичи зависят от `core_depth_midas` (no-fallback)
- Face ROI quality зависит от `core_face_landmarks` (может быть NaN если лиц нет)
- Shot boundaries зависят от `cut_detection` (hard cuts)

**Upstream зависимости**:
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings`, `shot_quality_text_embeddings`, `frame_indices`
- **core_depth_midas** (обязательно, no-fallback): предоставляет `depth.npz` с `depth_maps`, `frame_indices`
- **core_object_detections** (обязательно, no-fallback): предоставляет `detections.npz` с `boxes`, `valid_mask`, `class_ids`, `frame_indices`
- **core_face_landmarks** (обязательно, no-fallback): предоставляет `landmarks.npz` с `face_landmarks`, `face_present`, `frame_indices`, `has_any_face`, `empty_reason`
- **cut_detection** (обязательно, no-fallback): предоставляет shot boundaries для shot-level агрегатов

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа качества видео)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): не требуется напрямую (использует готовые embeddings из `core_clip`)
- **core_clip**: источник frame embeddings и text embeddings для zero-shot scoring
- **core_depth_midas**: источник depth maps для depth фичей
- **core_object_detections**: источник детекций объектов для object summary фичей
- **core_face_landmarks**: источник face landmarks для face ROI quality фичей
- **cut_detection**: источник shot boundaries для shot-level агрегатов
- **Segmenter**: предоставляет кадры и `metadata.json` с `shot_quality.frame_indices` (shared sampling group с core providers)

### Sampling Policy

- Использует `shot_quality.frame_indices` из `frames_dir/metadata.json` (shared sampling group с `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`)
- **Требования к выборке**: `frame_indices` должны точно совпадать с `core_clip.frame_indices`, `core_depth_midas.frame_indices`, `core_object_detections.frame_indices`, `core_face_landmarks.frame_indices` (strict equality, no-fallback)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- **Рекомендуемая стратегия** (Segmenter-owned): target_N = 240-1200 кадров (например, 600 как центр), stratified uniform (равномерно по времени + обязательные кадры начала/середины/конца), per-shot sampling (минимум 1 кадр на шот)
- Если `frame_indices` пустой → error (no-fallback policy)
- Если нет лиц → **ok** результат: non-face метрики и `quality_probs` вычисляются, `face_*` признаки остаются `NaN`
- Поддерживает batch processing (CPU-only модуль, последовательная обработка каждого видео)
- **Presets**: `fast` (выключает entropy-heavy метрики и rolling_shutter/lens), `default` (включает entropy-heavy, выключает rolling_shutter/lens), `quality` (включает rolling_shutter + lens группу)

---

## similarity_metrics

### Краткое описание

Модуль вычисления метрик схожести для видео: intra-video coherence (покадровые графики на `core_clip`) и reference similarity (сравнение с reference set из `dp_models`) по нескольким модальностям (visual/audio/text/pacing/quality/emotion). Baseline версия фокусируется на intra-video coherence.

**Версия**: 2.0.2  
**schema_version**: `similarity_metrics_npz_v3`
**Категория**: module  
**GPU**: не требуется (CPU-only, numpy операции над embeddings)

### Извлекаемые фичи

**Intra-video coherence** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `centroid_sims`: покадровая схожесть каждого кадра к центроиду всех кадров (intra-video coherence), shape `[N]` float32 (cosine similarity, диапазон [-1.0, 1.0])
- `temporal_sim_next`: схожесть каждого кадра с предыдущим кадром (временная согласованность), shape `[N-1]` float32 (cosine similarity, диапазон [-1.0, 1.0])

**Reference similarity** (опционально, если задан `reference_set_id`):
- `reference_present`: флаг наличия reference set, bool
- `reference_similarity_mean_topn`: средняя схожесть с топ-N референсными видео, float32 (NaN если reference не предоставлен)
- `reference_similarity_max`: максимальная схожесть с референсными видео, float32 (NaN если reference не предоставлен)
- `reference_similarity_p10`: 10-й перцентиль схожести с референсными видео, float32 (NaN если reference не предоставлен)

**Агрегированные признаки** (tabular, всегда включены):
- `feature_names (F,) object[str]`
- `feature_values (F,) float32` — агрегаты coherence + reference similarity (NaN если модальность/референсы отсутствуют)

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `ui_payload` (для UI рендера с графиками coherence и top-K reference videos)

**Зависимости между фичами**:
- `centroid_sims` зависит от вычисления центроида всех кадров (среднее нормализованных embeddings)
- `temporal_sim_next` зависит от попарных сравнений соседних кадров
- Агрегированные признаки зависят от `centroid_sims` и `temporal_sim_next` (статистики по времени)
- Reference similarity метрики зависят от наличия `reference_set_id` и загрузки reference pack из `dp_models`

**Upstream зависимости**:
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings` и `frame_indices` (строгий match с axis модуля)

**Опциональные зависимости** (модальности для reference similarity):
- **clap_extractor** (опционально): audio embeddings (AudioProcessor); если отсутствует → аудио-схожесть маркируется как `NaN` (optional, отсутствие допустимо)
- **shot_quality** (опционально): quality/style фичи
- **video_pacing** (опционально): pacing фичи
- **text_processor** (опционально): text фичи (использует `primary_embedding` если `primary_embedding_present=true`)
- **micro_emotion** (опционально): emotion фичи (если нет лиц — это допустимо)

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа схожести и уникальности видео)

### Взаимосвязи с модулями системы

- **core_clip**: источник frame embeddings (используются готовые embeddings, не загружает модель напрямую)
- **ModelManager** (`dp_models`): опционально, для reference sets (`similarity/reference_sets/<reference_set_id>/` с `manifest.json` и `.npy` матрицами embeddings)
- **Segmenter**: предоставляет кадры и `metadata.json` с `similarity_metrics.frame_indices` и `union_timestamps_sec`
- **AudioProcessor** (опционально): источник `clap_extractor` embeddings для audio модальности
- **TextProcessor** (опционально): источник text embeddings для text модальности

### Sampling Policy

- Использует `similarity_metrics.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Требования к выборке (strict)**: `frame_indices` должны **строго совпадать** с `core_clip.frame_indices` (иначе error)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback, без “дропа”)
- Если `frame_indices` пустой → error (no-fallback policy)
- **Reference set**: если задан `reference_set_id`, загружается reference pack из `dp_models/bundled_models/similarity/reference_sets/<reference_set_id>/` (manifest.json с schema `similarity_reference_pack_v1` и `.npy` матрицы embeddings по модальностям)
- Поддерживает batch processing (CPU processing, последовательная обработка каждого видео)

---

## story_structure

### Краткое описание

Модуль анализа структуры истории видео, вычисляющий ключевые метрики повествования: hook (зацепка), climax (кульминация), energy (энергия) и coherence (связность). Tier-0 baseline модуль, работающий без локальных ML-моделей, используя только результаты core-провайдеров.

**Версия**: 3.0.2  
**schema_version**: `story_structure_npz_v3`
**Категория**: module (Tier-0 baseline)  
**GPU**: не требуется (CPU-only, агрегирует данные из core компонентов)

### Извлекаемые фичи

**Временные кривые** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `embedding_sim_next`: косинусное сходство между соседними кадрами, shape `[N-1]` float32
- `embedding_diff_next`: косинусное расстояние (1 - similarity), shape `[N-1]` float32
- `embedding_change_rate_per_sec`: скорость изменения embeddings на секунду, shape `[N]` float32
- `motion_norm_per_sec_mean`: кривая движения (нормализована на секунду), shape `[N]` float32
- `any_face_present`: наличие лиц в кадрах, shape `[N]` bool
- `story_energy_curve`: основная кривая энергии (z-score), shape `[N]` float32
- `story_energy_curve_downsampled_128`: downsampled версия (128 точек), shape `[128]` float32
- `story_energy_peaks_idx`: индексы пиков энергии, shape `[P]` int32
- `story_energy_peaks_times_s`: времена пиков энергии (сек), shape `[P]` float32
- `story_energy_peaks_values_z`: значения пиков энергии (z-score), shape `[P]` float32
- `topic_shift_curve`: topic shift (/s) из OCR→CLIP text (NaN там где текста нет), shape `[N]` float32
- `topic_shift_peaks_idx`: пики topic shift, shape `[Q]` int32

**Hook метрики** (model-facing, `feature_names/feature_values`, фиксированный список):
- `hook_visual_surprise_score`: среднее значение энергии на hook, float32
- `hook_visual_surprise_std`: стандартное отклонение, float32
- `hook_motion_intensity`: интенсивность движения, float32
- `hook_cut_rate`: частота резких кадров (кадров/сек), float32
- `hook_motion_spikes`: количество спайков движения, int32
- `hook_rhythm_score`: оценка ритма, float32
- `hook_face_presence`: доля кадров с лицами, float32

**Climax метрики** (model-facing, `feature_names/feature_values`):
- `climax_frame_index`: индекс кадра кульминации (union-domain frame index), int32
- `climax_time_sec`: время кульминации (секунды), float32
- `climax_position_normalized`: позиция в [0, 1], float32
- `climax_strength`: сила (raw), float32
- `climax_strength_normalized`: сила (z-score), float32
- `number_of_peaks`: количество пиков энергии, int32
- `time_from_hook_to_climax`: нормализованное время от hook до climax, float32
- `hook_to_avg_energy_ratio`: отношение энергии hook к средней, float32

**Character proxies** (model-facing, `feature_names/feature_values`):
- `main_character_screen_time`: доля кадров с лицами, float32
- `speaker_switch_rate`: частота переключений, float32
- `speaker_switches_per_minute`: переключения в минуту, float32

**Text (OCR → CLIP text)** (model-facing, `feature_names/feature_values` + `topic_shift_curve_present`):
- `topic_shift_curve_present`: наличие topic shift curve, bool
- `topic_shift_peaks_count`: количество пиков topic shift, int32
- OCR missing/empty → `topic_shift_curve_present=false`, `topic_shift_curve` = NaN (модуль остаётся `status="ok"`)

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `models_used[]` (из зависимостей), `ui_payload` (для UI рендера)

**Зависимости между фичами**:
- `story_energy_curve` зависит от комбинации `embedding_change_rate_per_sec` и `motion_norm_per_sec_mean` (Z-score нормализация и сглаживание)
- Hook метрики зависят от `story_energy_curve` в hook окне (min(5 секунд, 15% длины видео))
- Climax метрики зависят от `story_energy_curve` (максимальное значение)
- `topic_shift_curve` зависит от OCR и CLIP text embeddings (если `text_mode=ocr_clip_text`)

**Upstream зависимости**:
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings` и `frame_indices` для вычисления embedding change rate
- **core_optical_flow** (обязательно, no-fallback): предоставляет `flow.npz` с `motion_norm_per_sec_mean` и `frame_indices` для кривой движения
- **core_face_landmarks** (обязательно, no-fallback): предоставляет `landmarks.npz` с `face_present` и `frame_indices` для character proxies (может быть валидным empty `no_faces_in_video`)
- **ocr_extractor** (опционально, если `text_mode=ocr_clip_text`): предоставляет OCR для topic shift curve (если отсутствует → topic shift отключён, но visual часть остаётся валидной)

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа структуры истории видео)

### Взаимосвязи с модулями системы

- **ModelManager** (`dp_models`): опционально, для CLIP text encoder через Triton (если `text_mode=ocr_clip_text`)
- **Triton** (опционально): HTTP API для GPU-ускоренной инференции CLIP text encoder (если `text_mode=ocr_clip_text`)
- **core_clip**: источник CLIP embeddings для вычисления embedding change rate
- **core_optical_flow**: источник кривой движения для energy
- **core_face_landmarks**: источник информации о лицах для character proxies
- **ocr_extractor**: опциональный источник OCR для topic shift curve
- **Segmenter**: предоставляет кадры и `metadata.json` с `story_structure.frame_indices` и `union_timestamps_sec`

### Sampling Policy

- Использует `story_structure.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Требования к выборке**: минимум 30 кадров, целевое количество 120, максимум 200 кадров (превышение → fail-fast ошибка)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (обязательно, no-fallback, отсутствие или некорректность → error)
- **Требования к зависимостям**: все зависимости должны покрывать `frame_indices` из метаданных, `frame_indices` в зависимостях должны совпадать с запрошенными (через mapping)
- Если `frame_indices` пустой или `len(frame_indices) < 30` → error (no-fallback policy)
- Если `len(frame_indices) > 200` → error (fail-fast, ошибка sampling policy)
- Поддерживает batch processing (CPU-only модуль, последовательная обработка каждого видео)

---

## text_scoring

### Краткое описание

Модуль анализа взаимодействия текста с видео, извлекающий признаки синхронизации текста с движением, детекции call-to-action (CTA), непрерывности отображения текста и пиков акцента. Tier-0 baseline модуль, работающий как consumer OCR-артефакта от внешнего компонента. Модуль не выполняет OCR самостоятельно.

**Версия**: 2.0.1  
**schema_version**: `text_scoring_npz_v2`
**Категория**: module (Tier-0 baseline)  
**GPU**: не требуется (CPU-only, consumer модуль)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров, shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `text_present`: наличие текста в видео, bool
- `text_presence`: есть ли OCR детекции на кадре, shape `[N]` bool
- `text_count_per_frame`: количество OCR детекций на кадре, shape `[N]` int32
- `ocr_raw`: debug (по флагу `store_debug_objects=true`), shape `[M]` object
- `ocr_unique_elements`: debug (по флагу `store_debug_objects=true`), shape `[K]` object
- `feature_names`: имена скалярных model-facing фич, shape `[F]` object (фиксированный порядок)
- `feature_values`: значения скалярных фич, shape `[F]` float32

**Text → Action / Motion Correlation** (model-facing, `feature_names/feature_values`):
- `text_action_sync_score`: робастная оценка синхронизации текста с движением (trimmed-mean оконных z-score motion), float32
- `text_motion_alignment`: средняя оценка мультимодального выравнивания текста с моментами активности, float32
- `text_motion_alignment_windowed`: оконная версия alignment (максимум в окне [t-w, t+w]), float32
- `multimodal_attention_boost_score`: максимальная оценка мультимодального выравнивания (по всем элементам), float32
- `multimodal_attention_boost_position`: относительная позиция (0..1) текста с максимальным alignment, float32

**Text Duration and Continuity** (model-facing, `feature_names/feature_values`):
- `text_on_screen_continuity`: средняя длительность отображения уникального текста (секунды), float32
- `text_on_screen_continuity_median`: медианная длительность, float32
- `text_on_screen_continuity_max`: максимальная длительность, float32
- `text_on_screen_continuity_std`: стандартное отклонение длительности, float32
- `text_on_screen_continuity_normalized`: средняя длительность, нормализованная на длину видео, float32
- `text_switch_rate`: частота смены текста (уникальных элементов / секунда), float32
- `num_unique_texts`: количество уникальных текстовых элементов, int32
- `time_to_first_text_sec`: время до появления первого текста (секунды), float32 | None
- `time_to_first_text_position`: нормализованная позиция первого текста (0..1), float32 | None
- `text_area_fraction`: средняя доля площади кадра, занимаемая текстом, float32

**Call-to-Action (CTA) Detection** (model-facing, `feature_names/feature_values`):
- `cta_presence`: оценка вероятности наличия CTA (0..1), float32
- `cta_first_timestamp`: время первого CTA (секунды), float32 | None
- `cta_mean_timestamp`: среднее время CTA (секунды), float32 | None
- `cta_last_timestamp`: время последнего CTA (секунды), float32 | None
- `cta_first_position`: относительная позиция первого CTA (0..1), float32 | None
- `cta_mean_position`: относительная позиция среднего CTA (0..1), float32 | None
- `cta_last_position`: относительная позиция последнего CTA (0..1), float32 | None
- `cta_strength`: средняя сила CTA (усреднённое multimodal alignment для CTA-элементов, 0..1), float32
- `persistent_cta_flag`: флаг наличия "стойкого" CTA (удерживается > 3 секунд), bool

**Text Emphasis Peaks** (analytics-only; в model-facing хранится только `text_emphasis_peaks_count`):
- `text_emphasis_peak_flags`: список индексов текстовых элементов, где alignment образует пики, list[int]
- `text_emphasis_peak_prominence`: значения prominence для каждого пика, list[float]
- `text_emphasis_peak_positions`: относительные позиции пиков в видео (0..1), list[float]

**Дополнительные метрики** (model-facing scalar, но могут быть NaN если feature выключена):
- `text_readability_score`: средний скор читаемости текста (0..1), float32
- `ocr_language_entropy`: энтропия распределения языков по уникальным элементам, float32 (если `enable_language_entropy=True`)
- `text_movement_speed`: средняя скорость движения текстовых элементов (если `enable_text_movement_speed=True`), float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `ui_payload` (для UI рендера)

**Зависимости между фичами**:
- Все фичи зависят от наличия OCR-артефакта (если отсутствует → `status="empty"`, `empty_reason="dependency_missing"` или `no_text_available"`)
- Multimodal alignment зависит от motion signal (из `core_optical_flow`, опционально), face signal (из `core_face_landmarks`, если `use_face_data=True`), audio signal (placeholder)
- CTA detection зависит от лексического анализа (fuzzy match с CTA-ключевыми словами) и флагов `is_cta_candidate` из OCR
- Text continuity метрики зависят от группировки OCR-детекций в уникальные элементы (дедупликация по IoU и текстовой похожести)

**Upstream зависимости**:
- **ocr_extractor** или **TextProcessor/OCR service** (опционально, baseline policy): предоставляет OCR-артефакт (`text_ocr/ocr.npz`, `ocr/ocr.npz`, `text_scoring/ocr.npz`). Модуль не падает при отсутствии OCR, возвращает валидный empty результат
- **core_face_landmarks** (опционально, если `use_face_data=True`): предоставляет `landmarks.npz` с `face_present` и `frame_indices` для мультимодального alignment (face signal). Если запрошен, но не найден → `FileNotFoundError`

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа взаимодействия текста с видео)

### Взаимосвязи с модулями системы

- **TextProcessor/OCR service**: создаёт OCR-артефакт (`ocr.npz`) — модуль является consumer и не выполняет OCR самостоятельно
- **core_face_landmarks**: опциональный источник информации о лицах для мультимодального анализа (если `use_face_data=True`)
- **core_optical_flow**: может использоваться для motion signal (future enhancement, в baseline не используется)
- **Segmenter**: предоставляет кадры и `metadata.json` с `text_scoring.frame_indices` и `union_timestamps_sec`

### Sampling Policy

- Использует `text_scoring.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback)
- **Требования к согласованности**: если `use_face_data=True`, `frame_indices` модуля должны быть подмножеством `core_face_landmarks.frame_indices` (Segmenter должен обеспечивать согласованную выборку)
- **Graceful degradation**: модуль не падает при отсутствии OCR-артефакта, возвращает валидный empty результат (`status="empty"`, `empty_reason="dependency_missing"` или `no_text_available"`)
- Если `frame_indices` пустой → error (no-fallback policy)
- Поддерживает batch processing (CPU processing, последовательная обработка каждого видео)

---

## uniqueness

### Краткое описание

Baseline-компонент "уникальности" в MVP: считает intra-video метрики повторяемости/разнообразия по sampled кадрам, используя только `core_clip` embeddings. Вычисляет pairwise similarity между кадрами для оценки повторяемости и разнообразия контента.

**Версия**: 1.0.2  
**schema_version**: `uniqueness_npz_v4`
**Категория**: module (Tier-0 baseline)  
**GPU**: не требуется (CPU-only, numpy операции над embeddings)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров (union-domain), shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `max_sim_to_other`: для каждого кадра максимальная cosine similarity к любому другому кадру (diag исключена), shape `[N]` float32
- `cos_dist_next`: cosine distance между соседними кадрами (по времени/порядку sampling), shape `[N-1]` float32

- `feature_names`: имена агрегированных model-facing scalar фич, shape `[F]` object (фиксированный порядок)
- `feature_values`: значения scalar фич, shape `[F]` float32 (bool как 0/1)

**Repetition / similarity метрики** (model-facing scalars, `feature_names/feature_values`):
- `repeat_threshold_is_otsu`: флаг (0/1)
- `repeat_threshold_mode`: строка хранится в `meta.ui_payload.repeat_threshold_mode`
- `repeat_threshold_used`: итоговый порог (cosine similarity), выше которого кадр считается "повтором", float32
- `repeat_threshold_raw`: сырое значение порога из Otsu (до clamp), если `mode=otsu`, float32
- `repeat_threshold_min/max`: clamp-границы для auto режима, float32
- `repetition_ratio`: доля кадров, у которых `max_sim_to_other >= repeat_threshold_used`, float32
- `pairwise_sim_mean`: средняя попарная cosine similarity по верхнему треугольнику, float32
- `pairwise_sim_p95`: 95-й перцентиль попарной similarity, float32

**Temporal change метрики** (model-facing scalars, `feature_names/feature_values`):
- `temporal_change_mean`: средняя скорость изменения семантики (per-second), float32

**Diversity proxy** (model-facing scalars, `feature_names/feature_values`):
- `diversity_score`: `clip(1 - pairwise_sim_mean, 0..1)` (чем меньше средняя similarity, тем выше diversity), float32
- `n_frames`: число sampled кадров N, int32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`meta.stage_timings_ms`), `ui_payload` (top-K повторы)

**Зависимости между фичами**:
- `max_sim_to_other` зависит от pairwise similarity матрицы (N×N, вычисляется из `core_clip.frame_embeddings`)
- `cos_dist_next` зависит от соседних кадров в `frame_embeddings` (cosine distance = 1 - cosine similarity)
- `repetition_ratio` зависит от `max_sim_to_other` и `repeat_threshold_used` (порог определяется через Otsu или fixed)
- `temporal_change_mean` зависит от `cos_dist_next` и нормализуется на dt из `union_timestamps_sec`
- `diversity_score` зависит от `pairwise_sim_mean` (обратная зависимость)

**Upstream зависимости**:
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings` и `frame_indices`. `core_clip.frame_indices` обязан полностью покрывать `metadata["uniqueness"]["frame_indices"]`, иначе error

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа уникальности и повторяемости видео)

### Взаимосвязи с модулями системы

- **core_clip**: источник CLIP embeddings (используются готовые embeddings, не загружает модель напрямую)
- **Segmenter**: предоставляет кадры и `metadata.json` с `uniqueness.frame_indices` и `union_timestamps_sec`
- **ModelManager** (`dp_models`): не требуется напрямую (использует готовые embeddings из `core_clip`)

### Sampling Policy

- Использует `uniqueness.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Требования к выборке**: минимум 60 кадров (рекомендуется, не проверяется), целевое количество 120, максимум 200 кадров (проверяется в коде, fail-fast при превышении)
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (source-of-truth, no-fallback)
- **Требования к зависимостям**: `core_clip.frame_indices` обязан полностью покрывать `metadata["uniqueness"]["frame_indices"]`, иначе error (no-fallback)
- Если `frame_indices` пустой → error (no-fallback policy)
- Если `len(frame_indices) > 200` → error (fail-fast, ошибка sampling policy, из-за O(N²) сложности pairwise similarity)
- Поддерживает batch processing (CPU-only модуль, последовательная обработка каждого видео)

---

## video_pacing

### Краткое описание

Модуль вычисления признаков темпа/монтажа (shot pacing) и связанных метрик движения/семантических/цветовых изменений строго на sampled кадрах от Segmenter. Анализирует shot boundaries, motion, semantic change rate, color change rate и structural pacing.

**Версия**: 2.0.1  
**schema_version**: `video_pacing_npz_v3`
**Категория**: module (Tier-0 baseline)  
**GPU**: не требуется (CPU-only, агрегирует данные из core компонентов)

### Извлекаемые фичи

**Основные фичи** (всегда включены):
- `frame_indices`: индексы обработанных кадров (union-domain), shape `[N]` int32
- `times_s`: временные метки кадров в секундах, shape `[N]` float32 (`union_timestamps_sec[frame_indices]`)
- `shot_boundary_frame_indices`: union-domain индексы кадров, являющихся началом нового шота, shape `[S]` int32
- `motion_norm_per_sec_mean`: motion curve aligned to `frame_indices` (from `core_optical_flow`), shape `[N]` float32
- `semantic_change_rate_per_sec`: semantic change rate (/s) aligned to `frame_indices` (from `core_clip`), shape `[N]` float32
- `color_change_rate_per_sec`: color change rate (/s) aligned to `frame_indices` (cheap LAB proxy), shape `[N]` float32
- `feature_names`: имена агрегированных model-facing scalar фич (фиксированный порядок, включая flattened histogram bins), shape `[F]` object
- `feature_values`: значения scalar фич (bool как 0/1), shape `[F]` float32

**Shot statistics** (model-facing scalars, `feature_names/feature_values`, всегда включены):
- `shots_count`: число шотов, int32
- `shot_duration_mean/median/min/max/std`: статистики длительности шотов (секунды), float32
- `shot_duration_entropy`: энтропия распределения длительностей (20 бинов), float32
- `shot_duration_mean_normalized`: `mean / video_length_seconds`, float32
- `shot_length_gini`: Джини по длительностям шотов, float32
- `short_shot_fraction`: доля шотов короче 0.5s, float32
- `quick_cut_burst_count`: число "бурстов" ≥3 cut'ов в окне 1s, int32
- `shot_length_histogram_5bins`: 5-мерный вектор долей шотов по бинам длительности, shape `[5]` float32
- `tempo_entropy`: энтропия распределения длительностей по 5 бинам, float32
- `cuts_variance`: дисперсия длительностей шотов (sec²), float32
- `cuts_per_10s/max/median`: частота cut'ов (в окнах 10s; значения в 1/sec), float32
- `cut_density_map_8bins`: 8-мерный вектор плотности cut'ов по 8 равным временным сегментам (в 1/sec), shape `[8]` float32

**Pace curve** (model-facing scalars, часть feature-gated):
- `pace_curve_slope`: тренд по последовательности `log1p(shot_duration_sec)` (линейная регрессия), float32
- `pace_curve_slope_normalized`: `pace_curve_slope * mean(shot_duration_sec)`, float32
- `pace_curve_peaks`: пики pace curve, list[int]
- `pace_curve_peaks_mean_prominence`: средняя prominence пиков, float32
- `pace_curve_peak_positions`: позиции пиков, list[float]
- `pace_curve_dominant_period_sec`: периодичность по автокорреляции, float32 (если `enable_periodicity=True`)
- `pace_curve_power_at_period`: мощность на периоде, float32 (если `enable_periodicity=True`)

**Motion metrics** (model-facing scalars, из `core_optical_flow`):
- `mean_motion_speed_per_shot`: средняя скорость движения на шот, float32
- `motion_speed_median/variance/90perc`: статистики скорости движения, float32
- `share_of_high_motion_frames`: доля кадров выше 75-го перцентиля, float32
- `share_of_high_motion_shots`: доля шотов с высокой средней скоростью, float32
- `motion_shot_corr`: корреляция (Пирсон) между длительностью шота и его средней motion speed, float32

**Content change rate** (model-facing scalars, из `core_clip`):
- `frame_embedding_diff_mean/std`: средняя скорость изменения семантики (per-second), float32
- `high_change_frames_ratio`: доля переходов выше 75-го перцентиля, float32
- `scene_embedding_jumps`: число переходов выше `mean + 2σ`, int32
- `semantic_change_burst_count`: число "бурстов" ≥3 high-change переходов в окне 5s, int32

**Color pacing** (model-facing scalars):
- `color_change_rate_mean/std`: средняя скорость изменения цвета (DeltaE(LAB) per-second), float32
- `color_change_bursts`: пики detrended-скорости DeltaE, int32 (если `enable_bursts=True`)
- `saturation_change_rate`: std от ΔS/dt (HSV), float32
- `brightness_change_rate`: std от ΔV/dt (HSV), float32

**Lighting pacing** (model-facing scalars):
- `luminance_spikes_per_minute`: количество резких изменений яркости в минуту, float32

**Structural pacing** (model-facing scalars):
- `intro_speed`: медианная длительность шота в первой четверти видео, float32
- `main_speed`: медианная длительность шота в средней части, float32
- `climax_speed`: медианная длительность шота в последней четверти, float32
- `pacing_symmetry`: `(climax - intro) / median_overall`, float32

**Метаданные**:
- `meta`: словарь с run identity, версиями, статусом, таймингами стадий (`stage_timings_ms`), `ui_payload` (для UI рендера)

**Зависимости между фичами**:
- Shot statistics зависят от `shot_boundary_frame_indices` из `cut_detection` (shot boundaries как source-of-truth)
- Motion metrics зависят от `motion_norm_per_sec_mean` из `core_optical_flow` (уже нормализована на dt/max(H,W))
- Content change rate зависит от `frame_embeddings` из `core_clip` (CLIP cosine distance между соседними кадрами, нормализованная на dt)
- Color pacing зависит от cheap LAB proxy на downscaled frames (DeltaE между соседними кадрами, нормализованная на dt)
- Structural pacing зависит от группировки шотов по четвертям видео (по последовательности шотов)

**Upstream зависимости**:
- **cut_detection** (обязательно, no-fallback): предоставляет shot boundaries (`detections.shot_boundaries_frame_indices`) как source-of-truth
- **core_optical_flow** (обязательно, no-fallback): предоставляет `flow.npz` с `motion_norm_per_sec_mean` и `frame_indices`
- **core_clip** (обязательно, no-fallback): предоставляет `embeddings.npz` с `frame_embeddings` и `frame_indices` для semantic change rate

**Downstream зависимости**:
- Нет явных downstream зависимостей (результаты могут использоваться для анализа темпа и монтажа видео)

### Взаимосвязи с модулями системы

- **cut_detection**: источник shot boundaries (hard cuts) для shot statistics
- **core_optical_flow**: источник кривой движения (RAFT через Triton) для motion metrics
- **core_clip**: источник CLIP embeddings (через Triton) для semantic change rate
- **Segmenter**: предоставляет кадры и `metadata.json` с `video_pacing.frame_indices` и `union_timestamps_sec`
- **FrameManager**: загрузка кадров из `frames_dir` для cheap LAB proxy (color change rate)

### Sampling Policy

- Использует `video_pacing.frame_indices` из `frames_dir/metadata.json` (Segmenter contract, no-fallback)
- **Требования к выборке**: минимум 30 кадров (fail-fast), целевое количество ~200 кадров (на коротких видео — плотнее, см. Segmenter primary group policy), равномерное покрытие по времени
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (source-of-truth, no-fallback, отсутствие или немонотонность → error)
- **Alignment policy**: `video_pacing.frame_indices` должны быть подмножеством `core_clip.frame_indices` и `core_optical_flow.frame_indices`. Если зависимости не покрывают `frame_indices` → error (no-fallback)
- Если `frame_indices` пустой или `len(frame_indices) < 30` → error (no-fallback policy)
- Поддерживает batch processing с оптимизацией (гибридный батчинг, переиспользование конфигурации, параллельная обработка видео)
---

## Навигация

[Module README](../README.md) · [DataProcessor](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
