# Покрытие фич: доки, валидаторы, HTML-таблица (аудит)

Дата: 2026-04-23. Назначение: проверка требований (1) `docs/FEATURE_DESCRIPTION.md`, (2) валидатор выхода, (3) нормальные диапазоны в итоговой HTML, (4) пояснения к колонкам.

## 1. TextProcessor

| Требование | Статус |
|------------|--------|
| (1) FEATURE_DESCRIPTION | **22/22** экстракторов с `src/extractors/<name>/docs/FEATURE_DESCRIPTION.md` |
| (2) Валидатор | **22/22** — `utils/validate_*_text_npz.py` (или `validate_asr_text_proxy_text_npz.py` для asr_text_proxy) |
| (3) Диапазоны в HTML | Подсветка `--melt-qa`: для `text_processor/*` задан блок **`re:^text_processor(?:/.*)?$`** в `view_csv_feature_qa.json` (общие regex по `tp_*` + **уровень-2** для групп `tp_semclust_*`, `tp_cos_*`, `tp_embshift_*`, `tp_titlehashcos_*`, `tp_topktitles_*`, `tp_tragg_*`, `tp_tchunk_*`). Источник истины по контракту по-прежнему: `FEATURE_DESCRIPTION.md` + `validate_*_text_npz.py`. |
| (4) Пояснения RU | **`storage/result_store/view_csv_feature_descriptions_ru.json`** + fallback **`view_csv_melt_captions_ru.py`**; покрытие `tp_*` — по мере заполнения JSON (см. последние PR по экстракторам). |

Один merged NPZ: `text_processor/text_features.npz` — валидаторы проверяют срез по префиксу/схеме.

## 2. AudioProcessor

| Требование | Статус |
|------------|--------|
| (1) FEATURE_DESCRIPTION | **Все 18** экстракторов с `main.py` имеют `docs/FEATURE_DESCRIPTION.md` (вкл. ранее отсутствовавшие asr, band_energy, chroma, hpss, emotion_diarization; `clap_extractor` — референс в `docs/` + полный документ в корне). |
| (2) Валидатор | **18/18** — `utils/validate_*.py` |
| (3)(4) HTML + диапазоны | `view_csv.py --melt-qa` → **`view_csv_feature_qa.json`** с секциями `asr_extractor`, `spectral_extractor`, и т.д. Пояснения: merge **FULL_KEY_RU** / **SUFFIX** / `view_csv_feature_descriptions_ru.json` |

## 3. VisualProcessor (модули с `main.py`, без venv)

| Требование | Статус |
|------------|--------|
| (1) | Все **продуктовые** модули под `modules/` / `core/model_process/` (кроме тестовых `failing_module`) снабжены `docs/FEATURE_DESCRIPTION.md` или `FEATURE_DESCRIPTION.md` в корне. **`micro_emotion`** — добавлен `docs/FEATURE_DESCRIPTION.md`. |
| (2) | Валидаторы `utils/validate_*.py` / `validate_*_npz.py` присутствуют у основных модулей. |
| (3)(4) | Как и для VP: **`view_csv_feature_qa.json`** + melt captions; **Visual**-компоненты перечислены по `component` в JSON. |

## Как устроена итоговая HTML

1. `python3 storage/result_store/view_csv.py --melt` — таблица component | feature | **описание** | …видео.
2. Описания: `melt_feature_caption_ru(name)` ← overrides из **`view_csv_feature_descriptions_ru.json`**, иначе токен-glossary.
3. **Нормальные диапазоны** в HTML при **`--melt-qa`**: читается **`view_csv_feature_qa.json`**. Для **TextProcessor** используйте ключ компонента `text_processor/...` и правила с префиксом `re:^text_processor(?:/.*)?$` (см. выше).

## Чек-лист для нового компонента

1. `docs/FEATURE_DESCRIPTION.md` — все **выходные** скаляры/массивы, NaN-политика, ссылка на схему.
2. `utils/validate_*.py` — `--struct` / `--ranges` (+ `--timings` если есть), диапазоны = те же, что в доке.
3. Расширить **`view_csv_feature_descriptions_ru.json`** для RU-строки в melt.
4. (Опционально) Добавить/расширить **`view_csv_feature_qa.json`** для подсветки в wide-отчёте.
---

## Навигация

[Module README](../README.md) · [DataProcessor](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
