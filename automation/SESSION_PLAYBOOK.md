# Плейбук автономной часовой сессии (Claude работает без участия Ильи)

Цель: за ~1 час (= 5-часовой кредит-лимит) автономно довести следующие компоненты DataProcessor
в духе `action_recognition` (логика в приоритете + оптимизации + валидация + развёртывание).
Полный карт-бланш на изменения кода. Вопросы — только в конце сессии.

## 0. В начале сессии (первым делом)
1. Засечь старт: `date -u +%s > automation/.session_start` (буфер остановки — 55 мин).
2. Прочитать `runpod_ssh/POD_CONNECTION.md` (Host/Port/статус). Если под STOPPED — Илья
   его запустит; если нет данных — работать логикой/кодом/CPU-smoke без GPU.
3. SSH-нюанс (из песочницы): `cp automation/runpod_ssh/id_ed25519 /tmp/k && chmod 600 /tmp/k`,
   ssh с `-F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`.

## 1. Очередь компонентов (по порядку)
1. **scene_classification** → 2. **shot_quality** → далее по `DataProcessor/docs/COMPONENT_VALIDATION_CHECKLIST.md`.
Каждый — по протоколу `DataProcessor/docs/COMPONENT_VALIDATION_PROTOCOL.md`, отчёты в
`DataProcessor/docs/component_reports/<component>/`.

## 2. Что делать по каждому компоненту (охватываем ВСЕ области; логика — приоритет)
- **Логика:** static-review кода → RUN_SPEC → прогон на поде (GPU) через `scripts/run_ar_local.py`-
  подобный раннер (или прямой вызов компонента) → разбор NPZ → REPORT (4 оси: корректность,
  стабильность/golden, различимость, модель-fit). Правки кода при необходимости (карт-бланш).
- **Валидация:** input+output валидаторы (по образцу action_recognition `validate_*`), контракты
  в `COMPONENT_CONTRACTS.md`, метрики `metrics.{json,prom}`.
- **Оптимизации:** GPU/CPU, батчинг, fp16, стоимость/диск, капы; замерить и записать в
  `automation/runner/AGENT_CONTEXT.md` (раздел 6 — тайминги/ресурсы, бывший RESOURCE_TIMING_LEDGER.md).
- **Развёртывание:** веса в `models_manifest.json`/`provision_base_models.py`/`bootstrap.sh`
  (если новые); k8s/compose при необходимости.
- **Замер ресурсов:** время каждой стадии + пик VRAM (`nvidia-smi`) → в ledger.

## 3. Прогон на поде (быстрый повтор)
```
# rsync код+веса (см. runpod_ssh/POD_SETUP_LOG.md §2-3), затем:
ssh <pod> 'bash /workspace/TrendFlowML/automation/runpod_ssh/pod_setup.sh'
ssh <pod> 'cd /workspace/TrendFlowML/DataProcessor && DP_MODELS_ROOT=$PWD/dp_models \
  nohup python3 scripts/run_ar_local.py --video <fixture> --seconds 0 --fps 25 --device cuda \
  --workdir /workspace/<out> >/workspace/run.log 2>&1 &'
# опрашивать summary.json; забрать rs/*/*.npz + summary → анализ
```
> Для не-action компонентов: раннер `run_ar_local.py` заточен под action_recognition. Быстрый путь —
> сделать аналогичный тонкий раннер `run_component_local.py` (Segmenter + нужный компонент по его
> main.py/CLI + его валидатор) ИЛИ гонять компонент напрямую поверх готового Segmenter-выхода.

## 4. Тайминг и остановка (КАЖДЫЙ вызов проверяй время)
- Считай elapsed = `date -u +%s` − `automation/.session_start`.
- При elapsed ≥ **3300 c (55 мин)**: заверши текущий отчёт, обнови ledger+checklist, **останови под**
  `bash automation/runpod_ssh/stop_pod.sh`, затем задай накопленные вопросы Илье.
- Раньше 55 мин под НЕ останавливать (Илья вернётся через ~3ч и запустит следующую сессию).

## 5. В конце сессии
- Обновить `COMPONENT_VALIDATION_CHECKLIST.md` (статусы), `automation/runner/AGENT_CONTEXT.md` (раздел 6).
- Краткий статус: что заштамповано/в работе, что осталось, вопросы (если есть).
- Под остановлен (проверить).

## Принципы (как в action_recognition)
Логика прежде всего; не штамповать без прогона+сверки; чистая логика — юнит-тесты; модель-coupled
части — с guard'ами; честно про ограничения; каждый прод-компонент = source-of-truth NPZ + валидаторы
+ метрики + контракты + доки.
