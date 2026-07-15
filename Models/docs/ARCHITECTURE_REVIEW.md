# Architecture review моделей TrendFlow + список изменений

Дата: 2026-07-02. Обзор архитектуры прогноза популярности (`Models/`) с учётом
актуального SOTA и контрактов DataProcessor. В конце — **пронумерованный список
изменений** по всей папке `Models/`, привязанный к файлам.

Опорные источники (см. «Источники» внизу): победитель SMP Challenge 2025 (MVP,
ACM MM'25) = pretrained-эмбеддинги + метадата → **градиентный бустинг**;
best practices мультимодального fusion (modality dropout, learned modality tokens);
Tweedie/NB/quantile для тяжёлохвостых счётчиков.

## 1. Вердикт

Дизайн грамотный и совпадает с трендами: двухпутёвость (tabular baseline + token
vNext), Encoder с fixed-budget (`global_embedding` + `summary_tokens (K,768)`,
K по `duration_sec`), cross-attention fusion, `log(1+Δ)` таргеты 7/14/21д,
snapshot_0 с ранней динамикой и статой канала, v2-контекст с TTL. Основной риск —
**переинвестиция в контентную башню до доказательства её вклада**; исправляется
baseline-first + ablation.

## 2. Углублённые рекомендации

### R1. Loss под тяжёлые хвосты (высокий приоритет)
Просмотры/лайки — сильно правоскошенные, «Tweedie-подобные» счётчики. Варианты:
- **Baseline (GBDT)**: `objective=tweedie` (LightGBM/XGBoost поддерживают) с
  подбором `tweedie_variance_power∈[1.1..1.9]`, либо оставить log1p+L2 и сравнить.
- **v1**: помимо log1p-регрессии добавить **quantile-головы** (напр. p10/p50/p90)
  для предсказательных интервалов — вирусность имеет большую aleatoric-неопределённость.
- Держать `log1p` перед нормализацией таргета (как сейчас), это ок.

### R2. Fusion-робастность (высокий)
- **Modality dropout** при обучении, фикс-ставка **0.2–0.5** — не даёт fusion
  переобучиться на самую предсказательную модальность и обеспечивает работу при
  пропущенной модальности (нет речи/лиц/комментов).
- Пропущенную модальность заменять **learned modality-token** (обучаемый вектор),
  а не занулением — заметно лучше по литературе.
- **Супервизировать несколько missing-конфигураций за батч** (не только одну),
  чтобы редкие dropout-случаи получали градиент.

### R3. Baseline-first + ablation (высокий)
- Собрать сильный **GBDT** на: pretrained-эмбеддинги (CLIP/CLAP/e5, агрегаты),
  метадата/канал/время, **ранняя динамика** — и зафиксировать планку.
- **Ablation по группам модальностей** (with/without visual|audio|text|semantic)
  → доказать вклад контента ДО end-to-end обучения трансформера. Это прямой мост к
  задаче валидации компонентов (полезность фичи = её вклад в метрику).
- GBDT остаётся прод-fallback (degraded-mode) — совпадает с выигравшим решением.

### R4. Ранняя динамика как отдельные фичи (высокий)
Помимо `views_0/likes_0` добавить **velocity/acceleration**: `dviews/dt`,
`dlikes/dt`, отношения `likes_0/views_0`, `comments_0/views_0`, возраст на момент
snapshot. Это обычно сильнейшие предикторы; сейчас в snapshot_0 только уровни.

### R5. Сплит и утечки (высокий)
- **Временной сплit по дате публикации** (не random) — иначе утечка и train/serve skew.
- Явные guard'ы: фичи `snapshot_0` не должны включать пост-snapshot информацию;
  зафиксировать t0 и «прогноз на любой момент» через явный `age_at_snapshot`.

### R6. Метрики (средний)
- Добавить **Spearman rank correlation** (продукту важнее ранжирование «какое видео
  зайдёт лучше», чем абсолютная ошибка) и **калиброванные интервалы** (coverage@p).
- Оставить per-horizon MAE/MAPE/log-MSE; следить за калибровкой uncertainty-весов.

### R7. Encoder/pooling (средний)
- `summary_tokens` через **learned attention-pooling / Perceiver-latent** (M→K) —
  уже заложено; зафиксировать как обязательное, не uniform-bins в v1.5.
- Time-embedding из `summary_times_s / duration` — ок; добавить нормировку возраста.

### R8. Связка sampling_policy ↔ dataset_version (средний, критично для воспроизводимости)
Любая переподгонка Segmenter под компонент **инвалидирует обученную модель**.
Правило: `dataset_version` пиньет `sampling_policy_version` + `feature_schema_version`;
смена любого → **обязательный ретрейн** и запись в model card.

### R9. Заморозка энкодеров (средний)
Сначала заморозить CLIP/CLAP/e5, учить только pooling+fusion+heads; end-to-end
разморозку — как отдельный этап после того, как замороженный вариант побьёт GBDT.

## 3. Список изменений по `Models/` (пронумерован, по файлам)

**Контракты (`Models/docs/contracts/`):**
1. `TARGETS_SPLITS_METRICS.md`: добавить (а) опцию **Tweedie/quantile** loss, (б)
   **временной split по дате**, (в) метрики **Spearman + interval coverage**, (г)
   явные правила анти-утечки snapshot_0 + `age_at_snapshot`.
2. `V1_TRANSFORMER_MODEL.md`: добавить **modality dropout (0.2–0.5)**, **learned
   modality-tokens** для пропусков, супервизию missing-конфигураций, **quantile-головы**.
3. `MODEL_INTERFACE_V2.md`: усилить требование — компоненты **экспонируют seq в
   стандартном NPZ** (не в debug `.npy`), с time-axis; добавить в snapshot **velocity**-фичи.
4. `BASELINE_MODEL.md`: расширить входы — pretrained-эмбеддинги + early-engagement
   velocity + все валидированные агрегаты; ввести **ablation-гейт** (доказательство вклада).
5. (новый) `FEATURE_TO_MODEL_CONTRACT.md`: пер-компонентный контракт «что отдавать
   модели» (тип token-stream, time-axis, dtype/range, seq-обязательность) — мост к
   `DataProcessor/docs/COMPONENT_VALIDATION_PROTOCOL.md`.

**Код (`Models/`):**
6. `baseline/Training/train_baseline.py`: `objective=tweedie` (+ подбор power),
   **SHAP/permutation importance**, ablation-режим по группам модальностей.
7. `v1/common/split.py`: реализовать **temporal split** по дате публикации (+ guard утечки).
8. `v1/model/v1_skeleton.py`: **modality dropout** + **modality-tokens** + **quantile-головы**.
9. `v1/encoder/encoder_v1.py`: зафиксировать **learned attention-pooling** (M→K),
   modality-token для отсутствующей модальности.
10. `v1/training/train_v1_skeleton.py` + `evaluate_v1.py`: Tweedie/quantile loss,
    Spearman + interval coverage, uncertainty-weighting с cap [0.2..5.0], фиксация
    `dataset_version`/`sampling_policy_version` в model card.
11. `v1/data/build_v1_dataset_index.py`: добавить **early-engagement velocity** и
    `age_at_snapshot`; пиньить `sampling_policy_version` в индекс датасета (R8).

**Приоритет исполнения:** сначала контрактные правки (1–5, дешёвые, задают правила),
затем baseline-путь (6,7,11) для планки+ablation, затем v1-код (8–10) как upside.
Всё это — **после** доказательства вклада контента ablation'ом (R3).

## Источники
- MVP: Winning Solution to SMP Challenge 2025 Video Track — https://arxiv.org/abs/2507.00950
- Multi-Modal Video Feature Extraction for Popularity Prediction — https://arxiv.org/abs/2501.01422
- Are Multimodal Transformers Robust to Missing Modality? — https://arxiv.org/pdf/2204.05454
- Gradient-Guided Modality Decoupling for Missing-Modality Robustness — https://arxiv.org/html/2402.16318v1
- Modality Dropout (обзор) — https://www.emergentmind.com/topics/modality-dropout-strategy
- Extended Poisson–Tweedie regression for count data — https://arxiv.org/pdf/1608.06888
- TweedieLoss (pytorch-forecasting) — https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.metrics.point.TweedieLoss.html
