# Эталон «полного» профиля для батча (п. 1.1 чеклиста)

**Цель:** один согласованный смысл «максимально полный прогон» — все три процессора (Audio / Text / Visual) и все переключаемые ветки в `global_config`, которые допустимы в вашей среде (GPU, Triton, токены).

## Базовый YAML

- **Источник правды по фичам и экстракторам:** [`DataProcessor/configs/global_config.yaml`](../../configs/global_config.yaml) В нём перечислены extractors/modules и их `feature_flags`; батчевый «full» **не** отключает компоненты, кроме осознанных исключений (см. ниже).

## Что считать «full max» (как в E2E)

Логика максимального включения Visual (все ключи `core_providers` и `modules` в `true`) и включения процессоров реализована в репозитории:

- [`backend/scripts/e2e_full_max_run.py`](../../../backend/scripts/e2e_full_max_run.py) — функции **`_enable_full_visual_inline_config`** и **`_patch_global_config_for_e2e`**.

Для батча **70** смысл тот же:

1. **`processors.audio.enabled: true`**, **`processors.text.enabled: true`**, **`processors.visual.enabled: true`** (и Segmenter по вашему оркестратору — обычно обязателен).
2. **Visual:** все булевы флаги в `processors.visual.inline_config.core_providers` и `.modules` → **`true`** (как в `_enable_full_visual_inline_config`), если не оговорён waiver на конкретный модуль.
3. **Text:** включить то, что в вашей версии `global_config` относится к «полному» табличному слою: в частности **`feature_flags.enable_embeddings: true`** и связанные embedder-ветки, если они были выключены в шаблоне.
4. **Triton / GPU:** для `core_depth_midas`, `core_optical_flow` и др. при `runtime: triton` нужен доступный **`triton_http_url`** (см. комментарии в `e2e_full_max_run.py` и `VisualProcessor/main.py`). Без Triton — осознанный режим (`local_visual_no_triton` в E2E), иначе «все фичи» физически недостижимы.
5. **Секреты / внешние сервисы:** что требует токенов (например HF) — должно быть в окружении, иначе компонент не «доступен» при всём `enabled: true` в YAML.

## Text — п. **1.5–1.6** чеклиста (зафиксировано 2026-04-15, Илья)

### **`emit_extra_metrics` и `compute_std` — везде `true`**

В **замороженном** YAML батча для всех релевантных экстракторов Text выставить **`emit_extra_metrics: true`** и, где поле есть, **`compute_std: true`**. В текущем шаблоне [`global_config.yaml`](../../configs/global_config.yaml) многие ветки имеют **`false`** — это нужно **переопределить** в снимке батча (не интерпретировать пустые диагностические поля на дашбордах как сбой пайплайна). См. план [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.4.

### **Векторный поиск — только FAISS**

Целевой backend батча: **FAISS** (`faiss-cpu` или `faiss-gpu` в окружении), без сознательного numpy-only fallback для сравнимости top‑K / корпусных метрик. Практика и ключи в `features_flat`: [FAISS_AND_NUMPY_BACKEND.md](../../TextProcessor/docs/FAISS_AND_NUMPY_BACKEND.md). Где в конфиге есть **`require_faiss`** / **`use_faiss_mode: always`** — настроить так, чтобы при отсутствии FAISS прогон **падал явно**, а не молча уходил в NumPy.

## Заморозка для Go (связь с п. 1.3–1.4)

Перед стартом батча сохраните **конкретный** снимок YAML (как делает E2E в `storage/e2e_full_max/<tag>/global_config_e2e.yaml`), зафиксируйте путь в чеклисте п. **1.1**, **`config_hash`** в п. **1.3**, **commit** в п. **1.4**.

## Тонкий profile (API)

Если оркестрация задаёт только «включить процессоры», как в кешированных профилях прогонов, типичный фрагмент:

```yaml
processors:
  segmenter: { enabled: true, required: true }
  audio:     { enabled: true, required: false }
  text:      { enabled: true, required: false }
  visual:    { enabled: true, required: false }
```

Детали merge с `--global-config` — в [`DataProcessor/configs/README.md`](../../configs/README.md).
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
