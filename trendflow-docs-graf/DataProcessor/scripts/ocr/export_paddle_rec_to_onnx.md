## Экспорт PaddleOCR recognizer → ONNX (model.onnx + dict.txt) для `ppocr_rec_onnx_v1`

Цель: получить два файла и положить их в:

- `DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/model.onnx`
- `DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/dict.txt`

Далее `ocr_extractor` сможет работать с `engine=ppocr_rec_onnx` (no-network).

### 0) Что скачать (source)

Скачайте **recognition inference model** (не detection) из PaddleOCR (PP-OCR) model zoo.

Самый практичный способ “не блуждать по сайту” — идти по ссылкам прямо из их GitHub README:

- Канонический репозиторий PaddleOCR: `https://github.com/PaddlePaddle/PaddleOCR`
- **Где взять/как получить ONNX модели** (официальная дока):  
  `https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html`
- **PP-OCRv5 multilingual recognizer (список моделей/языков)**:  
  `https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html`

Вам нужен именно **rec** (recognizer) для языка(ов) RU/EN (или multilingual, если в кадре часто смешанные алфавиты).

Обычно архив содержит директорию с файлами:
- `inference.pdmodel`
- `inference.pdiparams`
- иногда `inference.pdiparams.info`

Также нужен словарь символов (**dict**) для этого rec‑моделя (обычно `.txt`, 1 символ на строку).

#### Вариант PP-OCRv5 (PIR format): `inference.json` + `inference.pdiparams`

Для некоторых новых моделей PaddleOCR (в т.ч. PP-OCRv5) inference может быть в формате **PIR**:

- `inference.json` (внутри `"magic":"pir"`)
- `inference.pdiparams`
- (опционально) `inference.yml`, где может быть встроенный `character_dict`

В этом случае конвертация в ONNX делается так же через `paddle2onnx`, но `--model_filename` указывает на `inference.json`.

### 1) Окружение для экспорта (отдельное, build-time)

Экспорт — это build-time шаг, можно сделать на любой машине, где разрешён интернет.
В runtime (DataProcessor) скачиваний не будет.

Пример (CPU):

```bash
python3 -m venv .venv_ocr_export
source .venv_ocr_export/bin/activate
pip install -U pip
pip install "paddlepaddle==2.6.*" paddle2onnx onnx
```

Если хотите GPU-экспорт — ставьте `paddlepaddle-gpu` под вашу CUDA, но это не обязательно.

### 2) Конвертация inference модели в ONNX

Пусть inference модель лежит в:
- `/path/to/rec_infer/` и внутри есть `inference.pdmodel` + `inference.pdiparams`

Команда:

```bash
paddle2onnx \
  --model_dir /path/to/rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file model.onnx \
  --opset_version 11
```

Если у вас PIR-формат (`inference.json`):

```bash
paddle2onnx \
  --model_dir /path/to/rec_infer \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file model.onnx \
  --opset_version 11
```

Если opset 11 не проходит — попробуйте 12 или 13.

### 3) Подготовка dict.txt

Возьмите dict, который соответствует выбранному recognizer’у (из PaddleOCR model zoo / репозитория).

Требование в нашем контракте:
- `dict.txt`: **1 символ на строку**, без blank; CTC blank считается классом `0`.

Скопируйте/переименуйте в `dict.txt`.

Если в вашей модели есть `inference.yml` и внутри есть `PostProcess -> character_dict` (YAML list),
dict можно получить оттуда (важно: в нашем загрузчике пустые строки игнорируются, поэтому **ASCII-пробел**
нужно сохранять как строку `" "` на отдельной строке, а не как пустую строку).

### 4) Положить файлы в нужное место

```bash
cp model.onnx DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/model.onnx
cp /path/to/dict.txt DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/dict.txt
```

### 5) Быстрая проверка (до запуска DataProcessor)

Проверить, что модель запускается и декодится:

```bash
DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/scripts/ocr/validate_ppocr_rec_pack.py \
  --onnx DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/model.onnx \
  --dict DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/dict.txt \
  --image /path/to/crop.png
```

### 6) Запуск `ocr_extractor`

В `DataProcessor/configs/global_config.yaml` уже выставлено:
- `ocr_extractor.engine="ppocr_rec_onnx"`
- `ocr_extractor.rec_model_spec="ppocr_rec_onnx_v1_inprocess"`

Если файлы на месте — компонент стартует, иначе будет fail-fast с `weights_missing`.


