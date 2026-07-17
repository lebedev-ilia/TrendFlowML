# Fetcher-инфраструктура — 3 постоянных CPU-пода RunPod (сбор датасета YouTube)

Отдельная система от `automation/runner/` (ML-валидация компонентов) — не связаны кодом, только
общий реестр `automation/runner/state/machines.json`, куда эти поды зарегистрированы с
`policy=persistent`, `kind=fetcher`, чтобы ML-раннер их никогда не гасил/не удалял (см.
`automation/runner/runpod_api.py`, шапка модуля, и `AGENT_CONTEXT.md` раздел 3.4).

## Инфраструктура (уже создана, 2026-07-16)

| Под | Роль | vCPU/RAM | Volume | ЦОД | $/ч |
|---|---|---|---|---|---|
| `fetcher-main` | discover (непрерывно, авто-восстановление после исчерпания квоты) + workers шард 0/3 | 2/8ГБ | 15ГБ | EU-CZ-1 | $0.08 |
| `fetcher-worker-b` | workers шард 1/3 | 2/8ГБ | 15ГБ | EU-CZ-1 | $0.08 |
| `fetcher-worker-c` | workers шард 2/3 | 2/8ГБ | 15ГБ | EU-CZ-1 | $0.08 |

Итого ~$0.24/ч compute + ~$0.09/ч эквивалент хранилища (3×15ГБ×$0.07/ГБ/мес) ≈ **$200/мес** при
непрерывной работе. ID подов/томов — `state/provision_result.json`.

Видео удаляются локально сразу после подтверждённого HF-аплоада (уже так в коде Fetcher) — 15ГБ на
под достаточно как буфер, не для хранения всего корпуса.

## Как запустить (после того, как есть секреты)

Нужно: `HF_TOKEN`, `keys.txt` (49 YouTube API-ключей), `cookies/*.txt` (для скачивания видео).

```python
# из automation/fetcher/
import deploy
HF_TOKEN = "hf_..."
for pod in ("fetcher-main", "fetcher-worker-b", "fetcher-worker-c"):
    deploy.deploy_secrets(pod, hf_token=HF_TOKEN,
                          youtube_keys_local="/path/to/keys.txt",   # только реально нужен на main (discover)
                          cookies_dir_local="/path/to/cookies")     # нужен на всех (download)
    print(pod, deploy.launch(pod, HF_TOKEN))
```

## Мониторинг — watchdog.py + hourly_report.py (Третий агент)

Работают НА ТВОЁМ ПК (не на подах — нужен доступ к твоей Claude-подписке через `claude login`,
которого на headless-поде нет), постоянно через systemd — переживают перезагрузку ПК так же, как
`automation/runner/runner.service` для Первого агента: systemd сам поднимает сервис при загрузке
(`WantedBy=multi-user.target`) и перезапускает при падении (`Restart=on-failure`). Внутреннего
состояния между запусками почти нет — процессы просто продолжают опрашивать поды по SSH раз в час,
рестарт/перезагрузка ПК не теряет прогресс (прогресс живёт на подах, не в watchdog).

### Установка (один раз)

```bash
cd ~/Рабочий\ стол/TrendFlowML/automation/fetcher
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env уже заполнен (RUNPOD_API_KEY, VK_TOKEN3, HF_TOKEN) — проверь cat .env, если пусто см. .env.example
```

### Автозапуск + автовосстановление после ребута (systemd)

```bash
sudo cp fetcher-watchdog.service /etc/systemd/system/
sudo cp fetcher-report.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fetcher-watchdog
sudo systemctl enable --now fetcher-report
journalctl -u fetcher-watchdog -f     # лог живьём
journalctl -u fetcher-report -f
```

`enable` — значит оба сервиса будут САМИ стартовать при каждой загрузке ПК, без ручного запуска.
`Restart=on-failure` — если процесс упадёт (обрыв сети, временная ошибка), systemd поднимет его
заново сам через 30с.

Остановить: `sudo systemctl stop fetcher-watchdog fetcher-report`.
Проверить статус: `systemctl status fetcher-watchdog`.

### Полная автономность watchdog.py (один раз, 2026-07-17)

Без этого watchdog может НАЙТИ и починить свой же баг локально, но не может ни запушить фикс на
GitHub, ни перезапустить себя, чтобы фикс применился — упирается и просит владельца сделать это
руками (было так 2026-07-17 с фиксом сброса квоты). Два одноразовых шага снимают оба ограничения:

**1. Git push без интерактивного ввода** — credential store на уровне ОС (решает то же самое и для
Первого агента/`deepdive_agent.py`, не только для watchdog):
```bash
git config --global credential.helper store
echo "https://lebedev-ilia:ЗАМЕНИ_НА_СВОЙ_ТОКЕН@github.com" >> ~/.git-credentials
chmod 600 ~/.git-credentials
```
Токен — GitHub Personal Access Token с правами `repo` + `workflow` (Settings → Developer settings →
Personal access tokens). После этого `git push` из ЛЮБОГО скрипта, запущенного под твоим пользователем
ОС (включая systemd-сервисы `User=ilya`), работает без запроса пароля.

**2. Passwordless sudo ТОЛЬКО на перезапуск двух сервисов** (см. `fetcher-watchdog.sudoers` — узкий
список команд, НЕ полный root-доступ):
```bash
cd ~/Рабочий\ стол/TrendFlowML/automation/fetcher
sudo visudo -cf fetcher-watchdog.sudoers   # синтаксис-чек — ОБЯЗАТЕЛЬНО перед установкой
sudo cp fetcher-watchdog.sudoers /etc/sudoers.d/fetcher-watchdog
sudo chmod 440 /etc/sudoers.d/fetcher-watchdog
```
Если `which systemctl` показывает путь, отличный от `/usr/bin/systemctl` — поправь путь в
`fetcher-watchdog.sudoers` ДО установки.

После этих двух шагов watchdog при обнаружении и починке своего бага сам: коммитит → пушит →
(если баг в его собственном коде, не в поде) перезапускает себя через
`sudo systemctl restart fetcher-watchdog` — без единого ручного действия владельца.

## Патч в самом Fetcher (2026-07-16, обновлено 2026-07-17)

`Fetcher/fetcher/dataset_collector/cli.py::command_discover` — при исчерпании квоты ВСЕГО пула
YouTube API-ключей (`QuotaExceededError`) процесс раньше падал необработанным исключением (см.
`FETCHER_DATASET_COLLECTOR_HANDOFF.md` §6). Теперь: сохраняет прогресс, спит чанками по ~1ч до
полуночи **Pacific Time** (America/Los_Angeles) — РЕАЛЬНАЯ граница сброса квоты Google (= 07:00 UTC
летом / 08:00 UTC зимой; полночь UTC — частая ошибочная догадка, включая мою первую версию патча
2026-07-16, исправлено вотчдогом 2026-07-17), продолжает В ТОМ ЖЕ процессе с чекпоинта. Внешний
`launch_role.sh` дополнительно оборачивает в `while true` на случай реальной (не квотной) ошибки.

## Файлы

- `config.py` — конфиг (свой .env, независимый от runner).
- `runpod_client.py` — минимальный REST/GraphQL клиент RunPod (CPU-поды, Network Volume). Специально
  НЕ переиспользует `automation/runner/runpod_api.py` — независимость систем.
- `deploy.py` — SSH-хелперы: деплой секретов, запуск/перезапуск ролей, чтение логов/summary.json.
- `launch_role.sh` — bash-скрипт, копируется на под и запускает discover+workers (main) или только
  workers (worker-b/c) через `nohup`.
- `watchdog.py` — Третий агент (Sonnet 4.6), свободный чат + почасовой анализ логов, чинит код,
  коммитит/пушит и перезапускает себя/поды полностью автономно (см. раздел выше).
- `hourly_report.py` — часовой отчёт метрик, чистый код без LLM.
- `fetcher-watchdog.sudoers` — узкий passwordless sudo (только restart/status этих 2 сервисов).
- `state/provision_result.json` — ID подов/томов созданной инфраструктуры.
- `ssh/id_ed25519` — тот же ключ, что у ML-раннера (RunPod автоматически прописывает один и тот же
  паблик-ключ аккаунта во все новые поды) — скопирован для независимости модулей.
