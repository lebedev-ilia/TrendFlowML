# `story_structure` — Модуль анализа структуры истории видео

## Описание

Модуль `story_structure` анализирует структуру истории видео, вычисляя ключевые метрики повествования: **hook** (зацепка), **climax** (кульминация), **energy** (энергия) и **coherence** (связность). Это **Tier-0 baseline** модуль, который работает без локальных ML-моделей, используя только результаты core-провайдеров.

### Версия
- **Версия модуля**: 3.0.2
- **Версия схемы**: `story_structure_npz_v3`

### Документы и проверка NPZ
- `docs/FEATURE_DESCRIPTION.md` — ключи NPZ, табличные фичи, meta → wide CSV, melt/QA/ RU
- `docs/FEATURES_DESCRIPTION.md` — назначение серий и скаляров
- `docs/SCHEMA.md` — формальный контракт
- Проверка: `utils/validate_story_structure.py` `<path>/story_structure.npz` `--struct` / `--qa` (нужен `DataProcessor/qa` на `PYTHONPATH`)

## Архитектура

Модуль наследуется от `BaseModule` и реализует следующий интерфейс:

```python
class StoryStructureBaselineModule(BaseModule):
    def required_dependencies(self) -> List[str]
    def process(self, frame_manager, frame_indices, config) -> Dict[str, Any]
    def get_models_used(self, config, metadata) -> List[Dict[str, Any]]
```

## Алгоритм работы

### 1. Загрузка зависимостей

Модуль загружает результаты трех core-провайдеров:

- **`core_clip`**: CLIP embeddings кадров (`embeddings.npz`)
  - Ключи: `frame_indices`, `frame_embeddings`
  
- **`core_optical_flow`**: Кривая движения (`flow.npz`)
  - Ключи: `frame_indices`, `motion_norm_per_sec_mean`
  
- **`core_face_landmarks`**: Наличие лиц (`landmarks.npz`)
  - Ключи: `frame_indices`, `face_present`

Все данные выравниваются по `frame_indices` через функцию `_align_by_frame_indices()`.

### 2. Вычисление временной оси

Модуль использует `union_timestamps_sec` из метаданных как **source-of-truth** для временной оси. Это обязательное поле — отсутствие или некорректность данных приводит к ошибке (no-fallback policy).

```python
times_s = union_timestamps_sec[frame_indices]
```

### 3. Вычисление кривых энергии

#### Embedding Change Rate
1. Нормализация CLIP embeddings (L2 норма)
2. Вычисление косинусного расстояния между соседними кадрами:
   ```python
   sim_next = dot(emb_n[1:], emb_n[:-1])
   diff_next = 1.0 - sim_next
   ```
3. Нормализация на секунду (per-second):
   ```python
   diff_rate = diff_next / dt
   ```

#### Motion Curve
Кривая движения уже нормализована на секунду в `core_optical_flow` (`motion_norm_per_sec_mean`).

#### Комбинированная кривая энергии
1. Сглаживание обеих кривых гауссовым фильтром (sigma из конфига)
2. Z-score нормализация каждой кривой
3. Комбинация: `combined = 0.5 * emb_z + 0.5 * mot_z`
4. Повторное сглаживание и Z-score → `story_energy_curve`

### 4. Анализ Hook (зацепки)

**Окно hook**: `min(5 секунд, 15% длины видео)`

Если в этом окне меньше 3 кадров, окно расширяется до первых 3 кадров.

Вычисляемые метрики:
- `hook_visual_surprise_score`: среднее значение `story_energy_curve` на hook
- `hook_visual_surprise_std`: стандартное отклонение
- `hook_motion_intensity`: средняя интенсивность движения
- `hook_cut_rate`: частота "резких" кадров (motion > 75-й перцентиль) на секунду
- `hook_motion_spikes`: количество "спайков" (motion > 90-й перцентиль)
- `hook_rhythm_score`: нормализованная интенсивность спайков
- `hook_face_presence`: доля кадров с лицами

### 5. Анализ Climax (кульминации)

- **Позиция**: кадр с максимальным значением `story_energy_curve`
- **Время**: `climax_time_sec`
- **Позиция нормализованная**: `climax_position_normalized` ∈ [0, 1]
- **Сила**: `climax_strength` (raw) и `climax_strength_normalized` (z-score)
- **Количество пиков**: пики выше 90-го перцентиля
- **Время от hook до climax**: нормализованное время

### 6. Character Proxies (прокси персонажей)

На основе наличия лиц:
- `main_character_screen_time`: доля кадров с лицами
- `speaker_switch_rate`: доля переходов "есть лицо/нет лица"
- `speaker_switches_per_minute`: переключения в минуту

## Входные данные

### Обязательные входы

1. **`frames_dir`**: Директория Segmenter с:
   - `metadata.json` с обязательными полями:
     - `union_timestamps_sec`: список временных меток (секунды) для каждого union-кадра
     - `story_structure.frame_indices`: индексы кадров для обработки (union-domain, 0..N-1)
     - `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash` (run identity keys)

2. **`rs_path`**: Путь к хранилищу результатов (result_store), содержащий:
   - `core_clip/embeddings.npz`
   - `core_optical_flow/flow.npz`
   - `core_face_landmarks/landmarks.npz`

### Конфигурация (config)

```python
{
    "min_frames": 30,               # Fail-fast минимум кадров (N)
    "max_frames": 200,              # Максимальное число кадров (fail-fast лимит)
    "energy_smoothing_sigma": 1.0,  # Sigma для гауссова сглаживания
    # Text (baseline): OCR -> CLIP text (triton). If OCR is missing/empty -> topic_shift_curve_present=false (module still status=ok).
    "text_mode": "none",             # none|ocr_clip_text
    "clip_text_model_spec": "clip_text_triton",
    "clip_text_batch_size": 64,
    "ocr_max_chars_per_frame": 256,
    "triton_http_url": None,        # Triton HTTP URL (или использовать TRITON_HTTP_URL env var)
    # legacy / compat
    "subtitles": None,               # legacy only (moved to legacy_story_structure.py)
    "clip_model": None,
    "sentence_model": None,
}
```

### Требования к sampling

- **Минимум кадров**: 30
- **Целевое количество**: 120
- **Максимум кадров**: 200 (превышение → fail-fast ошибка)

## Выходные данные

Результаты сохраняются в NPZ файл:
```
result_store/<platform_id>/<video_id>/<run_id>/story_structure/story_structure.npz
```

### Структура NPZ файла

#### Массивы (numpy)

| Ключ | Формат | Описание |
|------|--------|----------|
| `frame_indices` | `(N,) int32` | Индексы обработанных кадров (union-domain) |
| `times_s` | `(N,) float32` | Временные метки кадров (секунды) |
| `embedding_sim_next` | `(N-1,) float32` | Косинусное сходство между соседними кадрами |
| `embedding_diff_next` | `(N-1,) float32` | Косинусное расстояние (1 - similarity) |
| `embedding_change_rate_per_sec` | `(N,) float32` | Скорость изменения embeddings (на секунду) |
| `motion_norm_per_sec_mean` | `(N,) float32` | Кривая движения (нормализована на секунду) |
| `any_face_present` | `(N,) bool` | Наличие лиц в кадрах |
| `story_energy_curve` | `(N,) float32` | Основная кривая энергии (z-score) |
| `frame_feature_present_ratio` | `(N,) float32` | Доля finite среди model-facing float кривых (energy/motion/emb_rate/topic_shift) |
| `story_energy_curve_downsampled_128` | `(128,) float32` | Downsampled версия (128 точек) |
| `story_energy_peaks_idx` | `(P,) int32` | Индексы пиков `story_energy_curve` (в терминах N) |
| `story_energy_peaks_times_s` | `(P,) float32` | Времена пиков энергии (сек) |
| `story_energy_peaks_values_z` | `(P,) float32` | Значения пиков энергии (z-score) |
| `topic_shift_curve` | `(N,) float32` | Topic shift (/s) из OCR->CLIP text (NaN там где текста нет) |
| `topic_shift_curve_present` | `bool` | Есть ли текстовые точки (OCR) для topic_shift_curve |
| `topic_shift_peaks_idx` | `(Q,) int32` | Пики topic shift (индексы в терминах N) |

#### Табличные скалярные фичи (`feature_names` / `feature_values`)

Вместо object‑dict модуль сохраняет фиксированную таблицу:

- `feature_names (F,) object`
- `feature_values (F,) float32` (0/1 для bool)

**Hook метрики:**
- `hook_visual_surprise_score`: среднее энергии на hook
- `hook_visual_surprise_std`: стандартное отклонение
- `hook_motion_intensity`: интенсивность движения
- `hook_cut_rate`: частота резких кадров (кадров/сек)
- `hook_motion_spikes`: количество спайков движения
- `hook_rhythm_score`: оценка ритма
- `hook_face_presence`: доля кадров с лицами

**Climax метрики:**
- `climax_frame_index`: индекс кадра кульминации (union-domain frame index)
- `climax_time_sec`: время кульминации (секунды)
- `climax_position_normalized`: позиция в [0, 1]
- `climax_strength`: сила (raw, из combined_s)
- `climax_strength_normalized`: сила (z-score, из story_energy_curve)
- `number_of_peaks`: количество пиков энергии

**Text (OCR -> CLIP text):**
- `topic_shift_curve_present`: bool
- `topic_shift_peaks_count`: int
- `time_from_hook_to_climax`: нормализованное время от hook до climax
- `hook_to_avg_energy_ratio`: отношение энергии hook к средней

**Character proxies:**
- `main_character_screen_time`: доля кадров с лицами
- `speaker_switch_rate`: частота переключений
- `speaker_switches_per_minute`: переключения в минуту

**Trace/debugging:**
- `core_face_landmarks_empty_reason`: причина empty для core_face_landmarks (если доступна)
- `ocr_empty_reason`: причина отсутствия OCR (если доступна)

#### Метаданные (`meta`, object)

Стандартные поля BaseModule:
- `producer`: "story_structure"
- `producer_version`: "3.0.2"
- `schema_version`: "story_structure_npz_v3"
- `created_at`: ISO timestamp
- `status`: "ok"
- `empty_reason`: `null`
- `models_used`: список использованных моделей (из зависимостей)
- `model_signature`: хеш моделей
- `stage_timings_ms`: timings по стадиям (dict)
- Run identity keys: `platform_id`, `video_id`, `run_id`, и т.д.

## Зависимости

### Hard dependencies (обязательные)

Модуль требует наличия результатов следующих core-провайдеров:

1. **`core_clip`**
   - Файл: `rs_path/core_clip/embeddings.npz`
   - Ключи: `frame_indices`, `frame_embeddings`
   - Использование: вычисление embedding change rate

2. **`core_optical_flow`**
   - Файл: `rs_path/core_optical_flow/flow.npz`
   - Ключи: `frame_indices`, `motion_norm_per_sec_mean`
   - Использование: кривая движения для energy

3. **`core_face_landmarks`**
   - Файл: `rs_path/core_face_landmarks/landmarks.npz`
   - Ключи: `frame_indices`, `face_present`
   - Использование: character proxies, hook face presence
   - **Примечание**: Может быть валидным empty (`no_faces_in_video`) — это не ошибка

### Важные требования

- Все зависимости должны покрывать `frame_indices` из метаданных
- `frame_indices` в зависимостях должны совпадать с запрошенными (через mapping)
- Отсутствие обязательной зависимости → `FileNotFoundError`

## Использование

### CLI интерфейс

```bash
python -m modules.story_structure.main \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    [--min-frames 30] \
    [--max-frames 200] \
    [--energy-smoothing-sigma 1.0] \
    [--text-mode ocr_clip_text] \
    [--clip-text-model-spec clip_text_triton] \
    [--clip-text-batch-size 64] \
    [--ocr-max-chars-per-frame 256] \
    [--triton-http-url http://localhost:8000] \
    [--subtitles "text1,text2,text3"] \
    [--log-level INFO]
```

### Программный интерфейс

```python
from modules.story_structure.story_structure import StoryStructureBaselineModule

# Инициализация
module = StoryStructureBaselineModule(
    rs_path="/path/to/result_store",
    max_frames=200
)

# Конфигурация
config = {
    "min_frames": 30,
    "max_frames": 200,
    "energy_smoothing_sigma": 1.0,
    "text_mode": "ocr_clip_text",
    "clip_text_model_spec": "clip_text_triton",
    "clip_text_batch_size": 64,
    "ocr_max_chars_per_frame": 256,
}

# Запуск
saved_path = module.run(
    frames_dir="/path/to/frames",
    config=config
)
```

### Интеграция в pipeline

Модуль автоматически вызывается через VisualProcessor pipeline, если указан в конфигурации:

```yaml
visual_modules:
  - name: story_structure
    config:
      max_frames: 200
      energy_smoothing_sigma: 1.0
```

## Обработка ошибок

### Fail-fast ошибки

Модуль выбрасывает ошибки в следующих случаях:

1. **Отсутствие `union_timestamps_sec`**: `RuntimeError`
2. **Превышение `max_frames`**: `RuntimeError` с сообщением о необходимости исправить sampling
3. **Отсутствие обязательной зависимости**: `FileNotFoundError`
4. **Несовпадение `frame_indices`**: `RuntimeError` (Segmenter должен обеспечить консистентность)
5. **Немонотонность временных меток**: `RuntimeError`

### Валидные empty состояния (по продуктовой политике)

- Если `text_mode=ocr_clip_text` и OCR отсутствует/пустой → модуль **пишет валидный NPZ**, но `meta.status="empty"` (обычно `empty_reason="dependency_missing"` или `no_text_available`).
- `core_face_landmarks` должен быть доступен как hard dependency. “Нет лиц в видео” отражается булевым `any_face_present`, а не `status=empty`.

## UI payload

Модуль кладёт UI-метаданные в `meta.ui_payload` (JSON dict), без дублирования больших массивов:
- pointers на кривые (`npz_key`)
- markers: hook window + climax
- peaks list (time + strength)
- flags (topic_shift_present, faces_any_present)

## Legacy / Experimental

Файл `legacy_story_structure.py` содержит экспериментальные функции (не baseline) для:
- Topic features из субтитров/ASR через SentenceTransformer
- Clustering-based segmentation (acts 1/2/3 и т.п.)

## Roadmap (варианты для доработки topic shift)

- **A1 (TextProcessor transcript)**: topic shift по embeddings чанков транскрипта (нужно расширить `text_features.npz`, чтобы была time-axis curve).
- **A2 (title/description)**: не curve, а 1–2 скалярных “topic drift” индикатора.
- **B1 (baseline v1 сейчас)**: OCR по кадрам (`ocr_extractor/ocr.npz`) → CLIP text encoder (Triton, `clip_text_triton`) → topic shift curve.

**Важно**: Любые ML-модели должны загружаться через `dp_models.ModelManager` (не через прямые загрузки).

## Производительность

- **Сложность**: O(N), где N — количество кадров
- **Память**: O(N) для массивов кривых
- **Время выполнения**: ~0.1-1 секунда для типичного видео (120 кадров)

## Примеры использования результатов

### Загрузка результатов

```python
import numpy as np

# Загрузка NPZ
data = np.load("story_structure.npz", allow_pickle=True)

# Извлечение кривых
energy_curve = data["story_energy_curve"]
times = data["times_s"]
frame_indices = data["frame_indices"]

# Извлечение features
features = data["features"].item()  # object array → dict
hook_score = features["hook_visual_surprise_score"]
climax_time = features["climax_time_sec"]

# Метаданные
meta = data["meta"].item()
models_used = meta["models_used"]
```

### Визуализация кривой энергии

```python
import matplotlib.pyplot as plt

energy_curve = data["story_energy_curve"]
times = data["times_s"]

plt.figure(figsize=(12, 4))
plt.plot(times, energy_curve, label="Story Energy")
plt.axvline(features["climax_time_sec"], color="r", linestyle="--", label="Climax")
plt.axvspan(times[0], features.get("hook_end_time", times[0] + 5), 
            alpha=0.2, color="green", label="Hook")
plt.xlabel("Time (seconds)")
plt.ylabel("Energy (z-score)")
plt.legend()
plt.title("Story Structure Analysis")
plt.show()
```

## Связанные компоненты

- **Segmenter**: генерирует `frame_indices` и `union_timestamps_sec`
- **core_clip**: предоставляет CLIP embeddings
- **core_optical_flow**: предоставляет кривую движения
- **core_face_landmarks**: предоставляет информацию о лицах

## Примечания

1. **Baseline подход**: Модуль не использует локальные ML-модели, только комбинацию core-провайдеров
2. **Time axis**: Строго использует `union_timestamps_sec` — нет fallback на FPS
3. **Per-second normalization**: Все метрики изменения нормализованы на секунду для устойчивости к плотности sampling
4. **Z-score normalization**: Кривые нормализуются для комбинации разных сигналов
5. **Hook window**: Адаптивное окно с минимальным расширением до 3 кадров

## Примечание про OCR (topic shift)

В baseline v1 topic-shift curve считается только в режиме:
- `text_mode=ocr_clip_text`

Источником является OCR‑артефакт:
- `rs_path/ocr_extractor/ocr.npz` (или совместимые локации `text_ocr/ocr.npz`, `ocr/ocr.npz`, `text_scoring/ocr.npz`)

Если OCR отсутствует/пустой, модуль сохранит валидный `story_structure.npz`, но поставит:
- `meta.status="empty"`
- `meta.empty_reason` ∈ {`dependency_missing`, `no_text_available`}

### Поддержка разных форматов выхода Triton CLIP text encoder

Модуль автоматически обрабатывает различные форматы выхода Triton модели `clip_text`:
- `(B, 512)` — pooled embeddings (стандартный формат)
- `(B, 1, 512)` — single token embedding (legacy формат)
- `(B, 77, 512)` — per-token embeddings (извлекается эмбеддинг на позиции EOT токена)

Это обеспечивает совместимость с разными версиями Triton моделей без изменения конфигурации.

## Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео:

- **Batch-safe**: использует per-video rs_path (нет shared mutable state между видео).
- **Дефолтный process_batch()**: последовательная обработка каждого видео через BaseModule.
- **GPU batching**: не требуется (CPU-only модуль, агрегирует данные из core компонентов).

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): без изменений (компонент работает через subprocess)
- Для single video: без изменений

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными.

## Human-friendly визуализация (Render System)

`story_structure` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/story_structure/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по story structure метрикам (frames_count, video_length_seconds, hook_visual_surprise_score, climax_time_sec, number_of_peaks, main_character_screen_time, topic_shift_curve_present)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, story_energy, embedding_change_rate_per_sec, motion_norm_per_sec_mean, topic_shift_curve, any_face_present)
- **Distributions**: распределения метрик (story_energy_curve, embedding_change_rate_per_sec, motion_norm_per_sec_mean, topic_shift_curve) с min, max, mean, std, median, percentiles
- **Markers**: информация о hook window и climax (временные метки, позиции, силы)
- **Peaks**: список всех обнаруженных пиков энергии с временными метками и значениями

Render-context может быть использован:
- **LLM** для генерации текстовых описаний структуры истории видео
- **Frontend** для построения графиков и визуализаций (timeline charts с story energy, motion, embedding change rate, markers для hook и climax, peaks)
- **Debugging**: быстрая проверка качества story structure анализа без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../story_structure/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: story energy, motion, embedding change rate по времени с отмеченными hook window и climax
  - Markers: информация о hook window и climax
  - Peaks: список всех обнаруженных пиков энергии
  - Distributions: статистики по метрикам
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
story_structure:
  min_frames: 30
  max_frames: 200
  energy_smoothing_sigma: 1.0
  text_mode: "ocr_clip_text"
  clip_text_model_spec: "clip_text_triton"
  clip_text_batch_size: 64
  ocr_max_chars_per_frame: 256
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).
