## `comments_aggregator` (Comments Embedding Aggregator)

### Назначение

Агрегирует эмбеддинги комментариев (уже вычисленные `CommentsEmbedder`) в единые векторы-представители с использованием двух стратегий: взвешенное среднее и медиана по компонентам. Сохраняет агрегированные векторы в артефакты и возвращает метаданные.

**Версия**: 1.3.0  
**Категория**: embedding aggregation  
**GPU**: не требуется

**Описание фич, зеркал и диапазонов; валидатор среза в NPZ:** [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · `utils/validate_comments_aggregator_text_npz.py`

**Контракт (Audit v3)**: [`SCHEMA.md`](SCHEMA.md) · machine: [`schemas/comments_aggregator_output_v1.json`](../../schemas/comments_aggregator_output_v1.json)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/comments_aggregator_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/comments_aggregator_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/comments_aggregator_l2/`

### Входы

- **`VideoDocument`** с полями:
  - `comments`: список объектов комментариев с полем `text`
  - `comments_likes` (опционально): список лайков для каждого комментария
  - `comments_authority` (опционально): список значений авторитета авторов
  - `comments_recency` (опционально): список значений актуальности/новизны

- **Эмбеддинги комментариев** (должны быть созданы `CommentsEmbedder` и зарегистрированы в `doc.tp_artifacts`):
  - `doc.tp_artifacts["embeddings"]["comments"]["relpath"]` → `comments_embeddings.npy` (N, D, per-run)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Агрегированные эмбеддинги (per-run sub-artifacts)

Экстрактор сохраняет агрегаты в `text_processor/_artifacts/`:
- `comments_agg_mean.npy` (если `compute_mean=true`)
- `comments_agg_median.npy` (если `compute_median=true`)

Абсолютные пути в result/NPZ не возвращаются. Для downstream связывания:
- `doc.tp_artifacts["comments"]["agg_mean_relpath"]`
- `doc.tp_artifacts["comments"]["agg_median_relpath"]`

Возвращаемые скалярные признаки (`result.features_flat`):
- **Canonical (new)**: `tp_commentsagg_*` (стабильная схема, ключи всегда присутствуют)
  - `tp_commentsagg_present` (0/1) — агрегаты вычислены (не "файл существует")
  - `tp_commentsagg_count`, `tp_commentsagg_dim`
  - `tp_commentsagg_mean_std`, `tp_commentsagg_median_std`
  - `tp_commentsagg_artifact_mean_written`, `tp_commentsagg_artifact_median_written`
  - `tp_commentsagg_weights_applied` (0/1) — были ли применены веса
  - `tp_commentsagg_weights_mask_likes` (0/1) — использовались ли лайки как веса
  - `tp_commentsagg_weights_mask_authority` (0/1) — использовался ли авторитет как вес
  - `tp_commentsagg_weights_mask_recency` (0/1) — использовалась ли актуальность как вес
  - `tp_commentsagg_weights_align_present` (0/1) — были ли доступны индексы для выравнивания
  - `tp_commentsagg_weights_align_shape_ok` (0/1) — соответствует ли форма весов форме эмбеддингов
  - `tp_commentsagg_dim_mismatch_flag` (0/1) — см. [`SCHEMA.md`](SCHEMA.md) (ветка invalid/missing embeddings)
  - `tp_commentsagg_unsafe_relpath_flag` (0/1) — был ли обнаружен небезопасный путь
  - `tp_commentsagg_compute_mean_enabled`, `tp_commentsagg_compute_median_enabled`, `tp_commentsagg_compute_std_enabled` (0/1) — флаги включения вычислений
  - `tp_commentsagg_write_artifacts_enabled`, `tp_commentsagg_require_comment_embeddings_enabled` (0/1) — флаги настроек
  - `tp_commentsagg_agg_mean_ms`, `tp_commentsagg_agg_median_ms` — время агрегации в **мс** при `emit_extra_metrics=true` (иначе **NaN**; **NaN**, если соответствующий `compute_*` выключен)

- **Legacy aliases (back-compat)**: также всегда заполняются `tp_comments_agg_*` (включая веса и `compute_*` на пустой ветке) и `tp_cagg_*`.

### Алгоритмы

#### 1. Взвешенное среднее (Weighted Mean)

**Формула весов**:
```
w_i = likes_i × authority_i × recency_i
```

**Процесс**:
1. Загрузка индексов: чтение `selected_indices` из артефактов `CommentsEmbedder` (если доступны) для выравнивания весов с эмбеддингами
2. Инициализация весов: единичные веса для всех комментариев
3. Применение весов: если доступны `likes`, `authority`, `recency` и индексы выравнивания, они перемножаются (веса применяются к комментариям через `selected_indices`)
4. Клиппинг: веса ограничиваются снизу значением 0.1
5. Нормализация: веса нормализуются так, чтобы сумма = 1.0
6. Взвешенное среднее: `vec = Σ(w_i × emb_i) / Σ(w_i)`
7. L2 нормализация: `vec = vec / ||vec||`

**Std (если `compute_std=true`)**:
- считается как \( \text{mean}_d(\text{std}_n(embs)) \): std по комментариям для каждой координаты, затем среднее по координатам.

**Особенности**:
- Если веса недоступны, используется равномерное взвешивание
- Если сумма весов ≤ 0, используется равномерное взвешивание
- Результирующий вектор нормализован (L2 norm = 1.0)

#### 2. Медиана по компонентам (Component-wise Median)

**Процесс**:
1. Вычисление медианы: `vec[j] = median(emb[i][j] for all i)`
2. L2 нормализация: `vec = vec / ||vec||`

**Особенности**:
- Устойчива к выбросам
- Результирующий вектор нормализован (L2 norm = 1.0)

### Конфигурация

```python
{
    "artifacts_dir": None,              # Путь к директории артефактов (по умолчанию: default_artifacts_dir())
    "model_name": "intfloat/multilingual-e5-large",  # dp_models resolve (metadata); должен совпадать с CommentsEmbedder
    "compute_mean": True,
    "compute_median": True,
    "compute_std": False,
    "write_artifacts": True,
    "require_comment_embeddings": False,
    "emit_extra_metrics": False
}
```

**Параметры**:
- `artifacts_dir`: директория для сохранения агрегированных эмбеддингов
- `model_name`: идентификатор модели в **`dp_models`** (без inference в этом экстракторе); должен совпадать с **`CommentsEmbedder`**

### Архитектура

1. **Загрузка эмбеддингов**: детерминированно берём relpath из `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`
2. **Загрузка матрицы**: `np.load(...)` из per-run `text_processor/_artifacts/`
3. **Проверка наличия**: если эмбеддинги не найдены, возвращается пустой результат
4. **Загрузка индексов выравнивания**: чтение `selected_indices` из `doc.tp_artifacts["comments"]["selected_indices_relpath"]` (создаётся `CommentsEmbedder`)
5. **Извлечение весов**: опциональные веса из `comments_likes`, `comments_authority`, `comments_recency` с выравниванием через `selected_indices`
6. **Агрегация weighted mean**: вычисление взвешенного среднего (с весами, если доступны)
7. **Агрегация median**: вычисление медианы по компонентам
8. **Сохранение артефактов**: сохранение агрегированных векторов в `.npy` файлы (`comments_agg_mean.npy`, `comments_agg_median.npy`)
9. **Регистрация в tp_artifacts**: запись relpath в `doc.tp_artifacts["comments"]["agg_mean_relpath"]` и `agg_median_relpath`
10. **Возврат метаданных**: возврат путей к файлам и статистик

### Обработка ошибок

- **Отсутствие эмбеддингов**: valid empty (`tp_commentsagg_present=0`), агрегаты не создаются; fail-fast при `require_comment_embeddings=true`
- **Несоответствие размеров**: если веса не совпадают с количеством комментариев, они игнорируются (используется равномерное взвешивание)
- **Отсутствие индексов выравнивания**: если `selected_indices` недоступны, веса применяются напрямую (если их длина совпадает с количеством эмбеддингов)
- **Небезопасный путь**: если relpath пытается выйти за пределы `artifacts_dir`, устанавливается флаг `tp_commentsagg_unsafe_relpath_flag=1`
- **Ошибка сохранения**: ошибка логируется, но не прерывает выполнение

### Особенности

- **Зависимость от CommentsEmbedder**: требует предварительного создания эмбеддингов
- **Хеширование**: использует SHA256 для идентификации наборов комментариев
- **Взвешивание**: поддерживает множественные источники весов (лайки, авторитет, актуальность)
- **L2 нормализация**: все агрегированные векторы нормализованы
- **Атомарное сохранение**: использование временных файлов для безопасного сохранения
- **Две стратегии**: взвешенное среднее (учитывает важность) и медиана (устойчива к выбросам)

### Performance characteristics

**Resource costs**:
- **CPU**: низкие (numpy операции агрегации)
- **GPU**: не используется
- **Estimated duration**: ~0.01-0.1 секунд для типичного набора комментариев

**Complexity**:
- mean: \(O(N \cdot D)\)
- median: \(O(N \cdot D)\) (в практике обычно тяжелее mean)

**Параметры производительности**:
- Размер эмбеддингов: определяется автоматически из загруженного массива
- Размер по умолчанию: 384 (если размер не определен)

### Зависимости

- `numpy`: численные операции (агрегация, нормализация)
- `hashlib`: генерация хешей для идентификации наборов комментариев
- `pathlib`: работа с путями к файлам

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **CommentsEmbedder**: создает эмбеддинги комментариев (предварительное требование)
- **VideoDocument**: схема документа с комментариями
- **text_utils.normalize_whitespace**: нормализация текста
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию

### Примечания

1. **Порядок выполнения**: `CommentsEmbedder` должен быть запущен перед `CommentsAggregationExtractor`
2. **Хеширование**: хеш вычисляется на основе текстов комментариев и имени модели, поэтому изменения в текстах или модели приведут к другому хешу
3. **Веса**: если веса не указаны, используется равномерное взвешивание для weighted mean
4. **Клиппинг весов**: веса ограничиваются снизу значением 0.1 для избежания нулевых весов
5. **Нормализация**: все агрегированные векторы нормализованы (L2 norm = 1.0) для использования в косинусной метрике
6. **Размерность**: размерность эмбеддингов определяется автоматически из загруженного массива
7. **Пустые наборы**: если комментариев нет или эмбеддинги не найдены, возвращается пустой результат без ошибки

