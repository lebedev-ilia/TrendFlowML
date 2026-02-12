# Анализ достаточности бенчмарков для DynamicBatch

## Дата анализа
2024 (на основе бенчмарков `core_clip` из `DataProcessor/VisualProcessor/core/model_process/core_clip/README.md`)

## Ключевое понимание

**Для динамического батчинга самое главное:**
- **Summary Delta RAM** и **Summary Delta VRAM** для разных конфигураций
- Эти метрики позволяют scheduler'у заранее знать, сколько памяти съест компонент при определенных входных параметрах
- Scheduler может решить: можно ли запускать компонент параллельно с другими или в несколько потоков

---

## Требования DynamicBatch (из BENCHMARK_REGISTRY_CONTRACT.md)

### Обязательные метрики (§4):

1. **Unit-cost метрики:**
   - `latency_ms_mean_stable_per_unit` ✅ (можно вычислить)
   - `latency_ms_p95` ⚠️ (желательно, но не критично для MVP)

2. **Memory метрики:**
   - `cpu_rss_peak_mb` ✅ (есть)
   - `vram_triton_peak_mb` ✅ (есть)
   - `vram_triton_delta_run_mb` ✅ (есть, ключевая для batch sizing)
   - `vram_triton_drift_mb` + `restart_recommended` ⚠️ (для долгоживущих процессов)

3. **Constraints:**
   - `max_batch_size_component` ⚠️ (можно вывести из данных, но не явно)
   - `cross_video_batching` ❌ (не указано)
   - `spikes` (bool) ❌ (не указано)

---

## Доступные данные из бенчмарков core_clip

### Прямо доступные поля:
- `Duration (s)` - общее время выполнения
- `Frames cnt` - количество кадров (unit)
- `Triton Batch` - размер батча
- `Triton Delta RAM (MB)` - дельта RAM Triton
- `Triton Delta VRAM (MB)` - дельта VRAM Triton (ключевая для batch sizing)
- `Component Delta VRAM (MB)` - дельта VRAM компонента
- `Component Delta RAM (MB)` - дельта RAM компонента
- `Summary Delta RAM` - итоговая дельта RAM
- `Summary Delta VRAM` - итоговая дельта VRAM
- `Peak CPU %` - процент использования CPU
- `Peak GPU %` - процент использования GPU
- `Image Inf (s)` - время инференса изображений
- `Text Inf (s)` - время инференса текста

### Контекстные данные:
- `Model` - модель (ViT-B/32 vunknown)
- `Frame Shape` - разрешение входных кадров (1920x1080, 1024x576)
- `Triton model 1` - имя модели Triton (clip_image_224/336/448)
- `Triton Preprocess` - препроцессинг (preprocess_clip_image_224/336/448)
- `Triton model 2` - текстовая модель (clip_text)
- `Runs` - количество запусков (3)

---

## Маппинг данных на требования DynamicBatch

### Два уровня планирования:

1. **Batch sizing внутри одного видео** (использует `vram_triton_delta_run_mb`):
   - Scheduler вычисляет, сколько кадров можно обработать в одном батче
   - Формула: `batch_size = floor((free_vram - headroom) / vram_triton_delta_run_mb)`
   - Зависит от: model_branch, batch_size (Triton Batch)

2. **Планирование параллельного выполнения** (использует `Summary Delta RAM/VRAM`):
   - Scheduler решает, сколько видео можно обрабатывать параллельно
   - Формула: `max_parallel = floor((free_vram - headroom) / max(Summary Delta VRAM per video))`
   - Зависит от: model_branch, frames cnt (количество кадров в видео)
   - **Важно:** frame shape НЕ влияет на память, только на время

### ✅ Полностью покрыто:

1. **`latency_ms_mean_stable_per_unit`:**
   ```
   latency_ms_mean_stable_per_unit = (Duration (s) * 1000) / Frames cnt
   ```
   Пример: для строки с Duration=57s, Frames cnt=131:
   - `latency_ms_mean_stable_per_unit = (57 * 1000) / 131 ≈ 434.7 ms/frame`

2. **`cpu_rss_peak_mb` (для планирования параллельного выполнения):**
   - **Использовать `Summary Delta RAM`** - это полная память, которую съест компонент при обработке всего видео
   - Пример: 1876 MB, 1994 MB, 2025 MB, 2039 MB, 2105 MB, 2278 MB, 2292 MB, 2343 MB, 2359 MB, 2408 MB
   - **Важно:** это позволяет scheduler'у решать, можно ли запускать компонент параллельно с другими

3. **`vram_triton_peak_mb` (для планирования параллельного выполнения):**
   - **Использовать `Summary Delta VRAM`** - это полная VRAM, которую съест компонент при обработке всего видео
   - Пример: 668 MB, 684 MB, 685 MB, 687 MB, 688 MB, 689 MB, 690 MB, 1195 MB, 1198 MB, 1213 MB
   - **Важно:** это позволяет scheduler'у решать, можно ли запускать компонент параллельно с другими

4. **`vram_triton_delta_run_mb` (ключевая для batch sizing внутри одного видео):**
   - Использовать `Triton Delta VRAM (MB)` (это дельта на единицу работы - per frame)
   - Пример: 664-688 MB для разных конфигураций
   - **Важно:** это позволяет scheduler'у вычислять, сколько единиц (кадров) можно обработать в батче внутри одного видео
   - **Примечание:** для batch sizing внутри видео используется `vram_triton_delta_run_mb`, но для планирования параллельного выполнения нужен `Summary Delta VRAM`

### ⚠️ Частично покрыто:

1. **`latency_ms_p95`:**
   - Не указано в таблице, но можно вычислить, если есть raw данные с несколькими runs
   - Для MVP не критично (scheduler может использовать mean)

2. **`vram_triton_drift_mb` + `restart_recommended`:**
   - Не указано, но можно отслеживать через сравнение `Summary Delta VRAM` между разными batch sizes
   - Для MVP не критично (scheduler может использовать консервативные значения)

3. **`max_batch_size_component`:**
   - Можно вывести из данных: максимальный `Triton Batch`, при котором нет OOM
   - В таблице видно batch sizes до 16, но нет явного указания hard cap
   - **Рекомендация:** добавить явное поле или вывести из данных

### ❌ Не покрыто (но не критично для MVP):

1. **`spikes` (bool):**
   - Не указано, но можно вычислить из статистики по runs (если есть raw данные)
   - Для MVP можно использовать консервативное значение `false`

2. **`cross_video_batching`:**
   - Не указано, но для `core_clip` обычно `false` (per-video processing)
   - Можно задать по умолчанию

---

## Вывод: достаточность данных

### ✅ **Достаточно для MVP DynamicBatch**

**Обоснование:**

1. **Ключевые метрики присутствуют:**
   - ✅ `latency_ms_mean_stable_per_unit` - вычисляется из Duration/Frames cnt
   - ✅ `cpu_rss_peak_mb` - есть в Component/Summary Delta RAM
   - ✅ `vram_triton_peak_mb` - есть в Triton/Summary Delta VRAM
   - ✅ `vram_triton_delta_run_mb` - есть в Triton Delta VRAM (ключевая для batch sizing)

2. **Данные покрывают разные конфигурации:**
   - Разные batch sizes (1, 2, 4, 16)
   - Разные разрешения входных кадров (1920x1080, 1024x576)
   - Разные модели (clip_image_224/336/448)
   - Разные количества кадров (1, 2, 5, 50, 131, 304)

3. **Позволяет scheduler'у:**
   - Вычислять оптимальный batch_size на основе доступной VRAM
   - Оценивать latency на единицу работы
   - Планировать ресурсы для разных конфигураций

### ⚠️ Рекомендации для улучшения:

1. **Добавить явные поля:**
   - `max_batch_size_component` - hard cap из данных или явно указать
   - `spikes` (bool) - если есть статистика по runs
   - `cross_video_batching` - явно указать для каждого компонента

2. **Расширить метрики (опционально):**
   - `latency_ms_p95` - если есть raw данные по runs
   - `vram_triton_drift_mb` - для долгоживущих процессов

3. **Структурировать данные:**
   - Выделить `input_bucket` (Frame Shape, resolution)
   - Выделить `model_branch` (224/336/448)
   - Выделить `knobs` (batch_size, preset)

---

## Пример маппинга для resource_costs JSON

Для строки из бенчмарка:
```
clip_image_224 | preprocess_clip_image_224 | 16 | 131 | 3 | 56 | 5.189 | 34 | 100 | 5 | 1379 | 664 | 4 | 660 | 2039 | 668
```

**Ключевые поля:**
- `Triton Batch = 16` - размер батча
- `Frames cnt = 131` - количество кадров
- `Summary Delta RAM = 2039 MB` - **полная RAM для планирования параллельного выполнения**
- `Summary Delta VRAM = 668 MB` - **полная VRAM для планирования параллельного выполнения**
- `Triton Delta VRAM = 664 MB` - **дельта VRAM на единицу для batch sizing**

### Маппинг в `UnitCost` (для batch sizing внутри видео):

```json
{
  "component": "core_clip.clip_image",
  "unit": "frame",
  "model_branch": "224",
  "metrics": {
    "latency_ms_mean_stable_per_unit": 39.6,  // (5.189 * 1000) / 131
    "cpu_rss_peak_mb": 660,  // Component Delta RAM (для справки)
    "vram_triton_peak_mb": 664,  // Triton Delta VRAM (для справки)
    "vram_triton_delta_run_mb": 664  // Triton Delta VRAM (ключевая для batch sizing)
  }
}
```

### Расширенный маппинг (для планирования параллельного выполнения):

```json
{
  "component": "core_clip.clip_image",
  "unit": "frame",
  "model_branch": "224",
  "input_bucket": {
    "frame_shape": "1920x1080",
    "width": 1920,
    "height": 1080,
    "frames_cnt": 131  // количество кадров влияет на Summary Delta
  },
  "knobs": {
    "triton_batch": 16,
    "triton_preprocess": "preprocess_clip_image_224"
  },
  "metrics": {
    "latency_ms_mean_stable_per_unit": 39.6,
    "cpu_rss_peak_mb": 660,  // Component Delta RAM
    "vram_triton_peak_mb": 664,  // Triton Delta VRAM
    "vram_triton_delta_run_mb": 664,  // для batch sizing внутри видео
    
    // Ключевые метрики для планирования параллельного выполнения:
    "summary_delta_ram_mb": 2039,  // полная RAM на видео
    "summary_delta_vram_mb": 668,  // полная VRAM на видео
    
    "max_batch_size_component": 16  // выведено из данных
  }
}
```

### Важно для scheduler'а:

1. **Для batch sizing внутри видео:**
   - Использовать `vram_triton_delta_run_mb` (664 MB в примере)
   - Зависит от: model_branch, batch_size

2. **Для планирования параллельного выполнения:**
   - Использовать `summary_delta_vram_mb` (668 MB в примере)
   - Зависит от: model_branch, frames_cnt
   - **Frame shape НЕ влияет** (1920x1080 vs 1024x576 - одинаковое потребление памяти, разное время)

3. **Для планирования RAM:**
   - Использовать `summary_delta_ram_mb` (2039 MB в примере)
   - Аналогично VRAM: зависит от model_branch, frames_cnt

---

## Заключение

**Данные из бенчмарков `core_clip` достаточны для грамотного распределения нагрузки DynamicBatch'ем**, при условии:

1. ✅ Правильного маппинга полей:
   - Duration/Frames → `latency_ms_mean_stable_per_unit`
   - Summary Delta RAM/VRAM → для планирования параллельного выполнения
   - Triton Delta VRAM → для batch sizing внутри одного видео

2. ✅ Использования правильных полей для разных задач:
   - **Batch sizing внутри видео:** `vram_triton_delta_run_mb` (зависит от model_branch, batch_size)
   - **Планирование параллельного выполнения:** `summary_delta_vram_mb` (зависит от model_branch, frames_cnt)
   - **Frame shape:** НЕ влияет на память, только на время

3. ⚠️ Добавления явных полей при конвертации в resource_costs JSON:
   - `summary_delta_ram_mb` и `summary_delta_vram_mb` для планирования параллельного выполнения
   - `input_bucket.frames_cnt` для учета влияния количества кадров на память

**Для всех компонентов** аналогичный формат бенчмарков будет достаточен, если:
- Есть Duration и количество units (frames/segments/etc.)
- Есть **Summary Delta RAM/VRAM** (для планирования параллельного выполнения)
- Есть **Triton Delta VRAM** (для batch sizing внутри видео)
- Есть информация о batch sizes и параметрах конфигурации (model_branch, frames_cnt)

**Рекомендация:** создать скрипт конвертации бенчмарков в формат resource_costs JSON (аналогично `render_resource_costs_from_matrix.py` для cut_detection), который будет:
- Вычислять `latency_ms_mean_stable_per_unit` из Duration/Frames
- Маппить `Summary Delta RAM/VRAM` для планирования параллельного выполнения
- Маппить `Triton Delta VRAM` для batch sizing внутри видео
- Добавлять `input_bucket` с frames_cnt и model_branch для правильного выбора конфигурации

