# Руководство по заполнению базы брендов для `brand_semantics`

Это руководство описывает процесс заполнения базы данных брендов для компонента `brand_semantics`, который использует Embedding Service для распознавания логотипов и брендов в видео.

## 📋 Обзор процесса

Процесс состоит из двух этапов:

1. **Сбор данных** (локальная база): интерактивное добавление логотипов из видео/фото в локальную папку `known_brands/`
2. **Синхронизация** (Embedding Service): загрузка собранных брендов в Embedding Service для использования в production

## 🎯 Шаг 1: Сбор логотипов из видео/фото

### Подготовка

1. **Выберите источник данных**:
   - `--videos`: директория с видео (скрипт сам сэмплит кадры и ищет логотипы)
   - `--photos`: плоская директория с изображениями (скрипт ищет логотипы на фото)
   - `--photos-by-brand`: директория, где **каждая подпапка = имя бренда**, внутри — изображения этого бренда

2. **Убедитесь, что YOLO модель доступна**:
   - Установите переменную окружения `DP_MODELS_ROOT` или убедитесь, что модель находится по пути:
     `DataProcessor/dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt`

3. **Проверьте ID классов логотипов**:
   - В скрипте `add_brand.py` найдите `LOGO_CLASS_IDS`
   - Убедитесь, что ID соответствуют классам `logo_region` и `text_region` в вашей YOLO модели
   - Обычно это `[2, 3]`, но проверьте актуальные значения в вашей модели

### Запуск скрипта

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/brand_semantics/add_brand.py --help
```

#### Примеры

1) Интерактивно из видео (будет спрашивать brand name на каждый найденный логотип):

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/brand_semantics/add_brand.py \
  --videos "/path/to/videos"
```

2) Интерактивно из фото (плоская папка):

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/brand_semantics/add_brand.py \
  --photos "/path/to/photos"
```

3) Почти полностью автоматически из фото, разложенных по брендам:

Структура входа:
```
photos_by_brand/
├── ferrari/
│   ├── img1.jpg
│   └── img2.jpg
└── nike/
    ├── a.png
    └── b.jpg
```

Запуск (label берётся из имени подпапки):

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/brand_semantics/add_brand.py \
  --photos-by-brand "/path/to/photos_by_brand" \
  --auto-accept
```

Если вы работаете на headless машине/сервере, добавьте `--no-gui`.

### Вариант: автоскачивание логотипов из Wikimedia Commons (рекомендуется вместо Google)

Мы добавили скрипт `download_logos_wikimedia.py`, который:
- Не использует скрейпинг/браузер-автоматику
- Ищет файлы в Wikimedia Commons через API
- Скачивает **thumbnail** заданной ширины (в т.ч. для SVG → скачивается PNG thumburl)
- Складывает изображения прямо в `known_brands/<brand_path>/`

Пример (поддерживает вложенные пути `category/brand`):

```bash
cd DataProcessor
python3 VisualProcessor/core/model_process/core_identity/brand_semantics/download_logos_wikimedia.py \
  --brands "car/ferrari,car/lamborghini,wear/nike" \
  --per-brand 20 \
  --thumb-width 768
```

Пример через файл:

`brands.txt`:
```
car/ferrari|Ferrari logo
wear/nike|Nike logo
```

Запуск:

```bash
cd DataProcessor
python3 VisualProcessor/core/model_process/core_identity/brand_semantics/download_logos_wikimedia.py \
  --brands-file brands.txt \
  --per-brand 25
```

Важно про лицензии/трейдмарки: Wikimedia даёт доступ к файлам с разными лицензиями. Для production лучше сохранять (или дополнить) метаданные источника и при необходимости фильтровать по лицензии вручную.

### Процесс работы

1. **Скрипт автоматически**:
   - Загружает YOLO модель
   - Проходит по всем видео/фото в указанных директориях
   - Детектирует логотипы через YOLO (классы `logo_region`, `text_region`)
   - Для каждого найденного логотипа показывает кроп с padding

2. **Интерактивная разметка**:
   - Для каждого логотипа скрипт показывает превью (OpenCV GUI или matplotlib)
   - В консоли запрашивается название бренда
   - Введите название бренда (например: `coca_cola`, `nike`, `apple`)
   - Специальные команды:
     - `Enter` → пропустить этот логотип
     - `q` → выйти из скрипта

   **Режим `--photos-by-brand`**:
   - Бренд берётся из имени подпапки
   - Можно включить `--auto-accept`, чтобы сохранять без вопросов

3. **Автоматическое сохранение**:
   - Для каждого бренда создаётся папка `known_brands/<brand_name>/`
   - Логотипы сохраняются с числовыми именами: `1.jpg`, `2.jpg`, ...
   - Кропы сохраняются с padding (15% по умолчанию)

### Рекомендации по сбору данных

#### Качество изображений

- ✅ **Хорошо**: Чёткие, хорошо освещённые логотипы
- ✅ **Хорошо**: Логотипы с достаточным контекстом (padding помогает)
- ❌ **Избегайте**: Размытые, слишком маленькие логотипы
- ❌ **Избегайте**: Логотипы с сильными искажениями

#### Количество изображений на бренд

- **Минимум**: 3-5 изображений для базового распознавания
- **Рекомендуется**: 10-20 изображений для стабильного распознавания
- **Оптимально**: 20-50 изображений для высокого качества

#### Разнообразие

Старайтесь собирать логотипы с разными:
- Углами обзора
- Освещением
- Размерами
- Фонами
- Вариантами логотипа (если есть разные версии)

#### Названия брендов

Используйте единообразные названия:
- ✅ `coca_cola` (snake_case)
- ✅ `nike`
- ✅ `apple`
- ❌ Избегайте: `Coca-Cola`, `Nike Inc.`, `Apple Inc.`

**Важно**: Название бренда будет использоваться как `name` в Embedding Service, поэтому будьте последовательны.

## 🔄 Шаг 2: Синхронизация с Embedding Service

### Подготовка

1. **Запустите Embedding Service**:
   ```bash
   cd DataProcessor
   python embedding_service/run_server.py
   ```
   Сервис будет доступен на `http://localhost:8001` (или порт из `EMBEDDING_SERVICE_PORT`)

2. **Убедитесь, что Triton доступен**:
   - Embedding Service использует CLIP через Triton для категории `brand`
   - Модель: `clip_image_336` (CLIP 336x336)
   - Triton должен быть запущен и доступен

3. **Проверьте настройки PostgreSQL**:
   - Настройте переменные окружения или `.env` файл (см. `embedding_service/SETUP.md`)

### Запуск синхронизации

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/brand_semantics/sync_known_brands_to_embedding_service.py
```

### Процесс синхронизации

1. **Для каждого бренда** в `known_brands/`:
   - Поддерживаются структуры:
     - `known_brands/<brand>/...`
     - `known_brands/<category>/<brand>/...` (и глубже)
   - Брендом считается любая директория, где **есть изображения прямо внутри**.
   - Собираются все изображения
   - Для каждого изображения извлекается эмбеддинг через CLIP (clip_336)
   - Эмбеддинги усредняются: `avg_emb = mean(embeddings)`
   - Финальная L2-нормализация: `avg_emb = avg_emb / ||avg_emb||`

2. **Добавление в Embedding Service**:
   - Категория: `brand`
   - Название: `<brand_path>` (путь директории относительно `known_brands`, например `car/ferrari`)
   - Эмбеддинг: усреднённый и нормализованный
   - Метаданные: `{"source": "known_brands", "num_images": N, "brand_group": "...", "brand_leaf": "..."}`

3. **Результат**:
   - Каждый бренд добавляется как один объект в Embedding Service
   - Используется усреднённый эмбеддинг для стабильности распознавания

### Проверка результатов

#### Проверка количества брендов

```bash
curl http://localhost:8001/categories
curl "http://localhost:8001/categories/brand/count"
```

#### Тестовый поиск

```bash
curl -X POST "http://localhost:8001/search" \
  -F "category=brand" \
  -F "top_k=5" \
  -F "similarity_threshold=0.7" \
  -F "image=@path/to/logo.jpg"
```

Ожидаемый результат:
- В выдаче должны быть бренды из `known_brands/`
- Similarity scores должны быть разумными (обычно > 0.7 для хороших совпадений)

## 📊 Структура базы данных

### Локальная структура (`known_brands/`)

```
known_brands/
├── coca_cola/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
├── nike/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── apple/
    ├── 1.jpg
    └── ...
```

### В Embedding Service

После синхронизации каждый бренд представлен как один объект:
- `category`: `"brand"`
- `name`: название бренда (из имени папки)
- `embedding`: усреднённый эмбеддинг (512d, L2-нормализованный)
- `metadata`: `{"source": "known_brands", "num_images": N}`

## 🔧 Расширенные возможности

### Добавление метаданных

Вы можете расширить скрипт синхронизации для добавления дополнительных метаданных:

```python
metadata: Dict[str, object] = {
    "source": "known_brands",
    "num_images": len(embs),
    "aliases_en": ["Coca-Cola", "Coke"],  # Английские алиасы
    "aliases_ru": ["Кока-Кола", "Кока Кола"],  # Русские алиасы
    "category": "beverage",  # Категория бренда
}
```

### Прямое добавление через API

Вы также можете добавлять бренды напрямую через Embedding Service API:

```bash
curl -X POST "http://localhost:8001/objects/add" \
  -F "category=brand" \
  -F "name=coca_cola" \
  -F "metadata={\"aliases_en\":[\"Coke\"],\"category\":\"beverage\"}" \
  -F "image=@logo.jpg"
```

### Batch добавление

Для массового добавления используйте batch API:

```bash
curl -X POST "http://localhost:8001/objects/batch_add" \
  -F "category=brand" \
  -F "names=[\"brand1\",\"brand2\"]" \
  -F "images=@logo1.jpg" \
  -F "images=@logo2.jpg"
```

## 🐛 Решение проблем

### Проблема: YOLO не находит логотипы

**Решение**:
- Проверьте, что `LOGO_CLASS_IDS` соответствуют актуальным ID классов в вашей модели
- Убедитесь, что модель YOLO обучена на классах `logo_region` и `text_region`
- Попробуйте уменьшить порог уверенности в скрипте (по умолчанию 0.5)

### Проблема: Embedding Service недоступен

**Решение**:
- Убедитесь, что Embedding Service запущен: `python embedding_service/run_server.py`
- Проверьте порт (по умолчанию 8001)
- Проверьте логи Embedding Service для детальной информации

### Проблема: Triton недоступен

**Решение**:
- Убедитесь, что Triton Inference Server запущен
- Проверьте, что модель `clip_image_336` загружена в Triton
- Проверьте переменную окружения `TRITON_BASE_URL` (по умолчанию `http://localhost:8000`)

### Проблема: Низкое качество распознавания

**Решение**:
- Увеличьте количество изображений на бренд (рекомендуется 20+)
- Убедитесь, что изображения разнообразны (углы, освещение, размеры)
- Проверьте качество исходных изображений (чёткость, разрешение)
- Попробуйте добавить больше примеров для проблемных брендов

## 📝 Чеклист для начала работы

- [ ] Настроены пути к видео/фото в `add_brand.py`
- [ ] YOLO модель доступна и настроена
- [ ] Проверены `LOGO_CLASS_IDS` в скрипте
- [ ] Запущен `add_brand.py` и собраны логотипы
- [ ] Собрано минимум 3-5 изображений для каждого бренда
- [ ] Запущен Embedding Service
- [ ] Запущен Triton с моделью `clip_image_336`
- [ ] Запущен `sync_known_brands_to_embedding_service.py`
- [ ] Проверено количество брендов в Embedding Service
- [ ] Выполнен тестовый поиск для проверки качества

## 🚀 Следующие шаги

После заполнения базы брендов:

1. **Тестирование**: Запустите `brand_semantics` на тестовых видео и проверьте качество распознавания
2. **Итерация**: Добавьте больше примеров для брендов с низким качеством распознавания
3. **Мониторинг**: Отслеживайте качество распознавания в production и добавляйте новые бренды по мере необходимости

## 📚 Дополнительные ресурсы

- [README brand_semantics](./README.md) - Документация компонента
- [Embedding Service README](../../../../../../embedding_service/README.md) - Документация Embedding Service
- [Embedding Service SETUP](../../../../../../embedding_service/SETUP.md) - Инструкция по настройке

