# Задачи владельца (человек) — батч 60+

Документ дополняет [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md): здесь перечислено то, что **не может** быть закрыто автоматически из репозитория.

**Журнал:** при выполнении отметьте пункты в чеклисте и кратко занесите строку в «Журнал выполнения» в шапке чеклиста.

---

## 1. Организация (фаза 0)

*Зафиксировано в чеклисте (2026-04-15): **solo** — **Илья** совмещает **0.1–0.3**; целевой размер батча **70** `video_id` (≥60 по плану).*

- [x] **0.1** Владелец реестра видео — **Илья**.
- [x] **0.2** Владелец наблюдаемости — **Илья** (Prometheus/Grafana: URL, доступы; не только `localhost`, если батч вне dev).
- [x] **0.3** Владелец пайплайна / релиза — **Илья** (заморозка конфига, `config_hash`, `RUN_LOG` после Go).
- [x] **0.4** Целевой размер — **60** (см. [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. 0.4).

---

## 2. Контент и реестр (фаза 3)

- [ ] **Найти и скачать / отобрать 70 видео** (цель по чеклисту п. **0.4**; не меньше **60** по плану) с учётом стратификации [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §3.1.
- [ ] Для каждого видео проставить **теги** в [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml) (сейчас `todo`).
- [ ] Периодически гонять проверку структуры реестра: `python DataProcessor/docs/audit_v4/scripts/validate_video_registry_60plus.py` (перед финальным Go — с флагом `--strict-count`, чтобы длина списка совпадала с `target_video_count`).
- [ ] Заполнить [COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md) и убедиться в квотах.
- [ ] Выполнить **hash** реестра и привязать к `config_hash` (чеклист 3.7).
- [ ] **Sign-off** реестра (чеклист 3.8).

---

## 3. Блокер TextProcessor 5/5 (фаза 2, п. 2.1) — приоритет

По данным L2 JSON (пример: `storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json`), на **3 из 5** путей набора B `text_processor` падает с ошибкой вида:

**`CUDA out of memory`** при загрузке **`intfloat/multilingual-e5-large`** (SentenceTransformer / TitleEmbedder), при занятой GPU **~5.6 GiB**.

**Что сделать вам (выбор стратегии):**

1. **Инфра:** увеличить VRAM, **сериализовать** text_processor на одной GPU, или выделить отдельную GPU под Text.
2. **Конфиг:** перевести проблемные прогоны на **CPU** для embedder (если поддерживается профилем) или уменьшить параллелизм воркеров.
3. **Оркестрация:** не запускать тяжёлые Visual+Text одновременно на одной карте без очереди.
4. **Waiver:** до исправения оформить письменное исключение из критерия «полный text на всех 60» (чеклист 2.1) — не рекомендуется без понимания downstream.

После правок — повторить прогон **5 видео** и убедиться `dataset_quality.n_ok == 5` в любом `*_audit_v4_stats.json` Text L2.

---

## 4. micro_emotion (фаза 2, п. 2.3)

- [x] Закрыто в коде + [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W1** (см. чеклист п. **2.3**, `RUN_LOG` follow-up 2026-04-15).

---

## 5. Инфраструктура наблюдаемости (фаза 4)

- [x] **Локальный E2E (2026-04-22):** `start_e2e_stack.sh --with-infra` поднимает Prometheus + Grafana; **два** scrape (API + worker с `DP_WORKER_METRICS_PORT`); дашборд **dataprocessor-overview** provisioned. Подробно: [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md), [METRICS_REFERENCE.md](../../monitoring/METRICS_REFERENCE.md).
- [ ] Поднять **тот же контур** в **среде, где будет батч 60+** (не только локальный хост) и **проверить** Targets / панели под нагрузкой.
- [ ] **Пилот (чеклист 4.5):** на 1–2 run подтвердить `processor`/`component` в Prometheus ([METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md)) — **не** закрыт только документацией стека.
- [x] **Воркер и метрики (путь API+worker):** HTTP `/metrics` на `DP_WORKER_METRICS_PORT` (см. `e2e_env.sh`, [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md)); **чисто CLI** без API — внешнее решение (чеклист 4.6).
- [ ] Записать **боевые URL** и retention в `monitoring/README.md` или внутренней wiki (чеклист 4.2, 4.7) — **прод/стенд TBD**; локальные URL см. [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md).

---

## 6. Пилот и Go (фазы 5–6)

- [ ] Сухой прогон **5–10** видео на **том же** пути, что батч.
- [ ] Заполнить **§7** чеклиста: подпись, дата старта UTC.

---

## 7. Конфиг батча (фаза 1)

- [x] Версии моделей / зависимостей (чеклист **1.2**): [BATCH_MODEL_VERSIONS_SNAPSHOT.md](BATCH_MODEL_VERSIONS_SNAPSHOT.md) — обновлять при смене `global_config` или образа Triton.
- [x] **config_hash** по шаблону `global_config` (чеклист **1.3**): см. таблицу «Config hash» в [BATCH_MODEL_VERSIONS_SNAPSHOT.md](BATCH_MODEL_VERSIONS_SNAPSHOT.md); при финальном YAML батча — пересчитать.
- [x] Text **1.5–1.6:** `emit_extra_metrics` / `compute_std` — **везде `true`**; векторный поиск — **только FAISS** — см. [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md) § Text.
- [ ] Политика **full max** (все процессоры и доступные фичи из шаблона): [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md), база [`DataProcessor/configs/global_config.yaml`](../configs/global_config.yaml).
- [ ] Указать **реальный путь** к **замороженному** YAML (снимок после патчей, аналог `global_config_e2e.yaml` из E2E), **версии моделей**, пересчитать **`config_hash`** и **commit** на момент Go (не полагаться на черновой hash из подготовки документов).
