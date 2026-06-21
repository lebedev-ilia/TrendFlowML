# Как создать semantic_clusters_v1

## Обзор

`semantic_clusters_v1` - это набор файлов для классификации эмбеддингов по семантическим кластерам:
- `pca.npy`: матрица PCA для снижения размерности
- `centroids.npy`: центроиды кластеров
- `clusters.jsonl`: словарь кластеров (id → name/group)

## Требования

1. **Эмбеддинги**: файл `.npy` с эмбеддингами формы `(N, D)`, где:
   - `N` - количество примеров
   - `D` - размерность эмбеддингов (например, 1024 для multilingual-e5-large)

2. **Python пакеты**:
   ```bash
   pip install numpy scikit-learn
   ```

## Шаг 1: Подготовка эмбеддингов

Соберите эмбеддинги из вашего датасета. Например, можно использовать эмбеддинги заголовков:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Загрузите модель
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Соберите тексты (например, заголовки видео)
titles = ["Заголовок 1", "Заголовок 2", ...]

# Вычислите эмбеддинги
embeddings = model.encode(titles, normalize_embeddings=True)

# Сохраните
np.save("embeddings.npy", embeddings)
```

Или используйте существующие эмбеддинги из TextProcessor результатов.

## Шаг 2: Запуск скрипта

```bash
cd /home/ilya/Рабочий\ стол/TrendFlowML/DataProcessor

python3 scripts/build_semantic_clusters_v1.py \
    --embeddings-path path/to/embeddings.npy \
    --output-dir dp_models/bundled_models/text/semantic_clusters_v1 \
    --n-clusters 32 \
    --reduced-dim 128 \
    --orig-dim 1024
```

### Параметры

- `--embeddings-path`: путь к файлу с эмбеддингами (`.npy`)
- `--output-dir`: директория для сохранения результатов (по умолчанию: `dp_models/bundled_models/text/semantic_clusters_v1`)
- `--n-clusters`: количество кластеров (по умолчанию: 32)
- `--reduced-dim`: размерность после PCA (по умолчанию: 128)
- `--orig-dim`: исходная размерность эмбеддингов (автоматически определяется, если не указана)
- `--clusters-names`: JSON файл с именами кластеров (опционально)
- `--random-state`: random state для воспроизводимости (по умолчанию: 42)

### Пример с именами кластеров

Создайте файл `clusters_meta.json`:
```json
[
    {"id": 0, "name": "технологии", "group": "IT"},
    {"id": 1, "name": "развлечения", "group": "media"},
    {"id": 2, "name": "образование", "group": "education"},
    ...
]
```

Запустите:
```bash
python3 scripts/build_semantic_clusters_v1.py \
    --embeddings-path embeddings.npy \
    --n-clusters 32 \
    --reduced-dim 128 \
    --clusters-names clusters_meta.json
```

## Шаг 3: Проверка результатов

Скрипт создаст три файла:
- `pca.npy`: матрица PCA `(orig_dim, reduced_dim)`
- `centroids.npy`: центроиды `(n_clusters, reduced_dim)`
- `clusters.jsonl`: словарь кластеров

Проверьте, что файлы созданы:
```bash
ls -lh dp_models/bundled_models/text/semantic_clusters_v1/
```

## Шаг 4: Проверка спецификации

Убедитесь, что spec файл существует и указывает на правильные пути:
```bash
cat dp_models/spec_catalog/text/semantic_clusters_v1.yaml
```

## Рекомендации

1. **Количество кластеров**: начните с 32-64 кластеров для небольшого датасета, увеличьте до 100-200 для большого
2. **Размерность PCA**: обычно 64-256, зависит от исходной размерности и количества кластеров
3. **Размер датасета**: рекомендуется минимум 1000 примеров для стабильных кластеров
4. **Имена кластеров**: после создания можно вручную отредактировать `clusters.jsonl` для более понятных имен

## Пример полного процесса

```bash
# 1. Соберите эмбеддинги (в Python)
python3 -c "
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')
titles = ['Пример заголовка 1', 'Пример заголовка 2', ...]  # ваши данные
embeddings = model.encode(titles, normalize_embeddings=True)
np.save('embeddings.npy', embeddings)
"

# 2. Создайте кластеры
python3 scripts/build_semantic_clusters_v1.py \
    --embeddings-path embeddings.npy \
    --n-clusters 32 \
    --reduced-dim 128

# 3. Проверьте результаты
python3 -c "
import numpy as np
pca = np.load('dp_models/bundled_models/text/semantic_clusters_v1/pca.npy')
centroids = np.load('dp_models/bundled_models/text/semantic_clusters_v1/centroids.npy')
print(f'PCA: {pca.shape}, Centroids: {centroids.shape}')
"
```

## Устранение проблем

**Ошибка: "PCA форма должна быть..."**
- Проверьте, что `--orig-dim` соответствует размерности эмбеддингов
- Или не указывайте `--orig-dim`, скрипт определит автоматически

**Ошибка: "Центроиды форма должна быть..."**
- Убедитесь, что `--n-clusters` и `--reduced-dim` согласованы
- Проверьте, что размерность PCA совпадает с размерностью центроидов

**Низкое качество кластеризации**
- Увеличьте количество примеров в датасете
- Попробуйте другое количество кластеров
- Проверьте, что эмбеддинги нормализованы

