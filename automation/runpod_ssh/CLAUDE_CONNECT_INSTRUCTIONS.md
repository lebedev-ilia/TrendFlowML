# Инструкция для Claude: подключение к RunPod-поду TrendFlow

Если ты читаешь этот файл в Cowork-сессии с подключённой папкой TrendFlowML —
у тебя есть всё нужное, чтобы подключиться к арендованному GPU-поду и гонять там тесты
компонентов DataProcessor. Ниже — что делать по порядку.

## Контекст

Проект TrendFlow (см. `CLAUDE.md` в корне репозитория) сейчас проходит доводку компонентов
DataProcessor (`VisualProcessor`/`AudioProcessor`/`TextProcessor`) до прод-состояния —
см. `docs/ANALYSIS_PLAN.md` и `docs/ANALYSIS_CHECKLIST.md`. Часть компонентов (например
`action_recognition` — see `DataProcessor/docs/component_reports/action_recognition/`)
требует GPU + PyTorch/CUDA для реального прогона тестов, а Cowork-песочница GPU не имеет
(torch не установлен, `nvidia-smi` недоступен). Поэтому для таких прогонов используется
арендованный RunPod-под, к которому нужно подключаться по SSH.

## Шаг 1 — прочитать актуальные данные подключения

Прочитай файл `automation/runpod_ssh/POD_CONNECTION.md` (та же папка, что и этот файл).
Там пользователь (Илья) вписывает актуальные Host/Port/статус после каждого деплоя пода —
это единственный источник истины, не полагайся на старые IP из истории чата.

Если в файле статус "not deployed" или поля не заполнены — под ещё не поднят, тестировать
не на чем. Сообщи об этом пользователю вместо того чтобы пытаться угадать IP.

## Шаг 2 — найти реальные пути в своей песочнице

Приватный ключ лежит в репозитории по пути (относительно корня TrendFlowML):
`automation/runpod_ssh/id_ed25519`

Через свой bash-тул узнай, под каким абсолютным путём смонтирована эта папка в твоей
песочнице (обычно что-то вроде `/sessions/<id>/mnt/TrendFlowML/...` — у каждой сессии
он свой, не переиспользуй путь из другого чата). Проверь права:

```bash
ls -la <путь-к-репо>/automation/runpod_ssh/
chmod 600 <путь-к-репо>/automation/runpod_ssh/id_ed25519
```

## Шаг 3 — проверить связь

```bash
ssh -i <путь>/automation/runpod_ssh/id_ed25519 -p <PORT> root@<HOST> \
  -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
  "echo ok && nvidia-smi --query-gpu=name,memory.total --format=csv"
```

Значения `<PORT>` и `<HOST>` — из `POD_CONNECTION.md`. Если получаешь `Permission denied
(publickey)` — скорее всего под пересоздан без подхвата ключа аккаунта, скажи пользователю
проверить Settings → SSH Public Keys на RunPod. Если connection timeout — под, вероятно,
остановлен (Stop), попроси пользователя нажать Start в консоли RunPod.

## Шаг 4 — подготовить окружение на поде

Каждый вызов ssh независим (как и твой локальный bash), поэтому либо оборачивай всё
в одну команду через `&&`, либо используй `tmux`/`nohup` на удалённой стороне для
долгих задач. Типовая последовательность первого запуска:

```bash
ssh -i <ключ> -p <PORT> root@<HOST> "\
  git clone <URL-репозитория TrendFlowML> /workspace/TrendFlowML || \
  (cd /workspace/TrendFlowML && git pull)"
```

(Если прямого доступа к git-репозиторию с пода нет — вместо clone используй `rsync -avz -e "ssh -i <ключ> -p <PORT>"` нужную подпапку, например `DataProcessor/`, с твоей локальной копии на под.)

Дальше — под конкретный компонент, например `action_recognition`:

```bash
ssh -i <ключ> -p <PORT> root@<HOST> "\
  cd /workspace/TrendFlowML/DataProcessor/VisualProcessor/modules/action_recognition && \
  pip install -r requirements.txt && \
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
```

(Версию CUDA под `torch` подбирай по выводу `nvidia-smi` с шага 3.)

Веса моделей — через `DataProcessor/scripts/download_models.py` (манифест
`configs/models_manifest.json`, HF dataset `Ilialebedev/trendflow_models`) — см. корневой
`CLAUDE.md` проекта.

## Шаг 5 — запустить тесты

Тестовые/валидационные скрипты компонента лежат в
`DataProcessor/docs/component_reports/<имя_компонента>/artifacts/run_*_validation*.py`
и/или `DataProcessor/VisualProcessor/modules/<имя_компонента>/scripts/run_tests.sh`.
Прогоняй их так же через ssh, вывод разбирай как обычно.

Для `action_recognition` компонент уже "заштампован" (см.
`REPORT_2026-07-05_FINAL.md` в том же component_reports — вердикт ✅ прод-готов).
Если тебя попросили именно про него — сверься с этим отчётом, возможно нужно не
перепроверять с нуля, а гнать конкретный регрессионный/остаточный пункт оттуда
(например OSNet ReID ветку или 200k/дисковый cleanup).

## Важное про стоимость

Под стоит денег, пока в статусе Running. Ты не можешь сам его остановить/запустить
без API-ключа (поле `RUNPOD_API_KEY` в `POD_CONNECTION.md` — если пользователь его
заполнил, можно дергать `https://rest.runpod.io/v1/pods/<id>/stop` через curl; если
пусто — по завершении тестов явно напомни пользователю остановить под вручную
в консоли RunPod).
