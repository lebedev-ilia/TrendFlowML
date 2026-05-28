## Encoder contract (VisualEncoder + AudioEncoder)

Статус: v1.0 (legacy) + v2 (FINAL interface-aligned)  
См. также: `Models/docs/contracts/MODEL_INTERFACE_V2.md`

### Назначение

Encoder запускается **после всех компонентов DataProcessor** и решает проблему variable-length последовательностей (очень разные длительности видео/аудио) совместно с Segmenter:
- Segmenter: задаёт sampling policy и формирует time-axis.
- Encoder: приводит любые последовательности к **fixed-budget** представлению для моделей прогноза.

### Source-of-truth time axis

- **Visual**: `frames_dir/metadata.json.union_timestamps_sec`
  - компоненты пишут `frame_indices` (union-domain) и/или `times_s` согласованный с `union_timestamps_sec`.
- **Audio**: время в секундах (`times_sec` / `segment_centers_sec` / `events_times_sec`).

Encoder **не делает общий time-join** между модальностями. Каждая модальность сжимается отдельно, fusion учитывает время через time embeddings.

### Input types (что encoder читает)

Компоненты могут отдавать:
- **Dense time-series** (по времени/индексам)
- **Sparse events** (списки событий во времени)
- **Precomputed embeddings** (по кадрам/сегментам)

Важно: компонент может отдавать и seq, и агрегаты. Encoder читает только seq; агрегаты идут в meta/table view для baseline.

### Output contract (per modality)

Encoder выдаёт фиксированный интерфейс:
- `global_embedding (D,) float32`
- `summary_tokens (K, D) float32`
- `summary_times_s (K,) float32` — **якорные времена токенов** (v1: центры uniform bins; v2: learned/иерархические anchors)
- `summary_mask (K,) bool`

### Budgets (v1.0)

- `D = 768`
- `K` выбирается **адаптивно** по `duration_sec`:
  - duration_sec < 90 → K=64
  - 90 ≤ duration_sec < 600 → K=96
  - duration_sec ≥ 600 → K=128

Фактические значения `K_visual`, `K_audio`, `duration_sec` фиксируем в meta encoder output.

### Encoder v0 (deterministic baseline)

Базовый алгоритм:
- uniform time-binning на K бинов
- для каждого ряда в каждом бине считаем mean/max/quantiles (robust)
- линейная проекция в D → получаем `summary_tokens`
- `global_embedding` = pooled representation (например, mean over tokens)

### Encoder v1 (trainable)

Encoder обучается **end-to-end** вместе с v1 transformer. Разрешены auxiliary losses, но MVP — supervised end-to-end.

### Constraints

- сложность: ≤ O(N) по длине исходной последовательности
- missing values: используем NaN+masks, не “нули-заглушки”

---

## v2 (FINAL): Tokenizer + Learned Pooling (без требования uniform bins)

В v2 Encoder трактуется как **Tokenizer + Learned Pooling** (см. `MODEL_INTERFACE_V2.md`):

- Компоненты публикуют последовательности/события/эмбеддинги (и/или `TokenStreams` как артефакт).
- Encoder строит fixed-budget представление **контентно**, а не через uniform time-binning.

### v2 Input (рекомендуемый минимум)

Один или несколько stream’ов вида:

- `tokens (N,D_in)` + `times_s (N,)` + `mask (N,)`
- опционально: `token_type`, `spans_s`, `importance`

### v2 Output (совместим с v1 API)

Выход сохраняет те же ключи, но с новой семантикой:

- `summary_tokens (K,D)` — pooled tokens
- `summary_times_s (K,)` — anchors (например: центры событий/сцен/или learnable queries с присвоенными times)
- `summary_mask (K,)` — какие tokens валидны

**Важно**: uniform bins остаются допустимым fallback (v1 compatibility mode), но не являются обязательным/каноничным.


