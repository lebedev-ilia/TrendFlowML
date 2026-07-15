# Cursor: как прогнать action_recognition v3 (реализация Claude готова)

> ## ⟳ Итерация 5 (2026-07-05, v3.3) — плановые улучшения (backbone-выбор, ReID, сегменты)
> Всё в коде (компилируется; чистая логика тестами ✅). Прогоны нужны для сверки альтернатив.
>
> **Выбор backbone действия** — `action_recognition --backbone slowfast|videomae|videomaev2|hiera`
> (дефолт slowfast, заштампован). Не-slowfast делегируется `utils/backbones.py` (transformers,
> penultimate-hook, метки из id2label модели). Провижн весов: `./bootstrap.sh --with-action-backbones`
> (VideoMAE/VideoMAEv2/Hiera + torchreid). **Прогнать сравнение** на 4:35: slowfast vs videomae —
> зафиксировать embedding_dim, классы, тайминги; выбрать по baseline-ablation (Models).
> ⚠ для не-slowfast `clip_len=16` (backbone сам ставит); Segmenter cfg `clip_len` тоже задать 16,
> чтобы окна были согласованы (`window_frames=clip_len×window_len_mult`).
>
> **OSNet ReID** — `core_object_detections --track-embedder osnet` (+`--track-osnet-weights <path>`
> offline). Прогнать на 8:00, сравнить intra/inter cos с histogram.
>
> **num_action_segments** — теперь дробит по СМЕНЕ top-1 Kinetics-класса внутри трека (не 1/трек).
> Проверить, что `num_action_segments > num_tracks` на people-роликах.
>
> Заполнить `RUN_RESULT_v3_3.md` (сравнение backbone + OSNet + сегменты). SlowFast-путь не менялся —
> регрессий быть не должно.
>
> ---

> ## ⟳ Итерация 4 (2026-07-04) — фикс `mean_clips_per_track` (косто-нейтральный), нужен ещё 1 прогон
> Диагноз из v3.1: окно = ровно `clip_len` (32) → механически **1 клип на окно**; трекер между
> окнами назначает новый id → трек живёт в одном окне → `mean_clips_per_track=1.0`. Фикс: **окно
> теперь длиннее clip_len** (`window_len_mult=3` → 96 кадров), а число окон уменьшено
> пропорционально (48→16) → **бюджет кадров тот же** (16×96=1536 ≈ старые 48×32), но компонент
> скольжением (clip_len=32, stride=16) делает **~5 клипов** на трек-присутствие в окне →
> `mean_clips_per_track≈5`. Диск не растёт. Правка в `Segmenter/action_windows.py`+`segmenter.py`
> (дефолт `window_len_mult=3`; настраивается в cfg action_recognition).
>
> **Перепрогон:** те же фикстуры (можно только 4:35 + 8:00 + golden), профиль без изменений. Подтвердить:
> **`mean_clips_per_track > 1`** (ожидаем ~4–5), `num_action_segments` растёт, остальное (penultimate/
> tubelet/классы/valid-empty/golden/валидаторы/metrics) — как в v3.1. Заполнить `RUN_RESULT_v3_2.md`.
> Диск: убедись, что runner чистит `frames/` после NPZ (ты уже добавил rmtree).
>
> ---

> ## ⟳ Итерация 3 (2026-07-04, v3.1 — доработка по ASSESSMENT) — нужен перепрогон
> Реализованы правки по всем 4 областям оценки (компилируются; чистая логика тестами ✅). Прогон:
>
> **Качество эмбеддинга (§1.1, главное).** `clip_embeddings` теперь = **penultimate-фичи backbone**
> (forward-hook на голову SlowFast), а не случайная проекция логитов. Размерность **≈2304** (не 256),
> фактическая — в `meta.embedding_dim`, режим — `meta.embedding_mode`. Флаг `--embedding-mode
> penultimate` (дефолт). **Проверь `meta.embedding_mode`:** если `projection_fallback` — hook не
> встал (сообщи структуру головы, поправлю). Валидатор v3 уже dim-гибкий.
>
> **Оптимизация (§2).** `--precision fp16` (GPU, опц.); `min_clip_real_frames=16` (CLI/kwarg) —
> без padded-клипов; **адаптивные окна** `windows_per_min` в cfg action_recognition (длинные видео →
> больше окон, потолок 256); **батч-путь** теперь по умолчанию делегирует последовательному v3
> (не отдаёт v2 молча); диск — `DataProcessor/scripts/cleanup_frames_after_npz.py <rs> --apply`.
>
> **Инфра (§3).** Вход: `utils/validate_action_recognition_input.py <frames_dir> <detections.npz>`
> (гонять ДО компонента). Метрики: компонент пишет `<rs>/action_recognition/metrics.{json,prom}`.
> Доки SCHEMA/FEATURE_DESCRIPTION/контракты обновлены под v3.
>
> **Замена моделей (§4).** Реальная ветка `--track-embedder osnet` (torchreid; `--track-osnet-weights`
> для offline) в `core_object_detections`; при отсутствии — авто-fallback→histogram. VideoMAE-путь —
> провайдер `transformers_pretrained` (переключение по baseline-ablation).
>
> **Per-person tubelet + temporal localization (§1.2).** `--localization track_anchored` (клипы по
> интервалам присутствия трека — это и чинит `mean_clips_per_track`, теперь **>1**) + `--tubelet-crop
> true` (SlowFast видит КРОП по боксу трека, а не весь кадр → действие конкретного человека). Новое
> поле `clip_segment_id` (сегменты действия, change-point). Проверь на прогоне: `mean_clips_per_track>1`,
> `num_action_segments>0`, и что на групповой сцене разные треки дают разные top-действия.
>
> **Перепрогон:** те же 3 фикстуры + golden ×2 на 4:35, `--embedding-mode penultimate`. Заполнить
> `RUN_RESULT_v3_1.md`. Сверить: `embedding_mode=penultimate`, `embedding_dim≈2304`, эмбеддинги L2,
> классы осмысленны, контроль valid-empty, golden идентичен, validator (вход+выход) pass, есть
> `metrics.json`. Опц.: прогнать `--track-embedder osnet` на 8:00 и сравнить intra/inter cos с histogram.
>
> ---
> _Ниже — заметки итераций 1–2 (preflight по весам/CLI/меткам актуален)._

> ## ⟳ Итерация 2 (2026-07-04, по REPORT_2026-07-04) — доработка качества, нужен перепрогон
> Владелец делегировал решения; реализованы 3 правки (компилируются, чистая логика тестами ✅):
>
> **R1 — устранение padded-клипов (главное).** Первопричина `mean_clips_per_track=1.0` и паддинга:
> детекция сэмплировала разреженно, а action-окна плотные → внутри окна у трека мало кадров.
> Фикс в `Segmenter/segmenter.py` (шаг 2.55): **dense-кадры окон добавляются в выборку
> `core_object_detections`** → треки полные (≥clip_len), паддинга нет, multi-clip на трек возникает
> сам через re-ID между окнами. Плюс новый параметр `action_recognition.min_clip_real_frames=16`
> (kwarg модуля) — треки с <16 реальными кадрами не эмитят клип (не плодим статичные padded-клипы).
> Ожидание после перепрогона: `mean_clips_per_track > 1`, `det mean_track_len ≈ clip_len` на длинных.
>
> **R2 — метки Kinetics.** Запусти (нужна сеть):
> `DataProcessor/.data_venv/bin/python DataProcessor/scripts/provision_kinetics_labels.py`
> → пишет `dp_models/visual/action_recognition/kinetics400_labels.txt` в порядке индексов модели;
> `action_recognition` подхватит автоматически. После этого `class_names` = реальные действия
> (не `action_<id>`) — владелец сверит top-действия с роликами.
>
> **R3 — диск (для 200k, не блокер логики).** Dense-кадры ↑ кэш frames. Политика: удалять frames
> после NPZ (или не хранить). На валидации хватило ручной очистки; для прод — в LOAD_AND_SCALING_PLAN.
>
> **Перепрогон:** те же 3 фикстуры + golden ×2 на 4:35. Сверить, что: паддинг ушёл
> (`mean_clips_per_track>1`, доля коротких треков без клипов), классы осмысленны (с метками),
> контроль всё ещё valid-empty, golden идентичен. Заполнить `RUN_RESULT_v3.md`.
>
> ---
> _Ниже — исходная заметка (итерация 1). Preflight по весам/CLI актуален._

Все правки §D из `docs/design/ACTION_RECOGNITION_V3.md` **реализованы и юнит-протестированы**
(логика ассоциации трекера, планировщик окон, v3-builder, валидатор). Тебе — прогнать стек на
3 фикстурах и заполнить `RUN_RESULT.md`. Прогонять по `RUN_SPEC_v2.md`.

## Что уже сделано (не переписывать)
- **Segmenter** `Segmenter/action_windows.py` + шаг 2.55/окна в `segmenter.py`: для
  action_recognition — плотные окна ≥clip_len подряд идущих кадров, `windows` в metadata.
- **core_object_detections** `utils/appearance_tracker.py` + вкручен в `main.py`: appearance-трекер
  → `track_ids (N,M)` в `detections.npz`, schema **v3**. Дефолтный эмбеддер — `histogram` (zero-dep,
  cv2), OSNet/CLIP — опция (пока fallback→histogram, см. ниже).
- **action_recognition** `utils/action_recognition_v3.py` (builder) + `utils/validate_action_recognition_npz.py`
  (валидатор) + правки `action_recognition_slowfast.py`: `_prepare_tracks` читает реальные
  `track_ids`; `process()` стэшит плоский поток; `run()` пишет v3-npz. Классы Kinetics снимаются
  softmax'ом с `raw[:, :400]` (выход головы SlowFast).

## Preflight (блокеры)
1. **Веса SlowFast.** В репо лежит VideoMAE, а не `slowfast_r50.pyth`. Провизионь Kinetics-веса
   (см. `RUN_SPEC.md` v1 §Блокер, вариант A: `slowfast_r50(pretrained=True)` → сохранить под путь spec).
   Без этого `ModelManager.resolve` упадёт.
2. **(опц.) Метки Kinetics.** Положи `kinetics400_labels.txt` (400 строк) в
   `DataProcessor/dp_models/visual/action_recognition/` — иначе классы будут `action_i` (id корректны).
3. **(опц.) OSNet.** Если хочешь ReID вместо histogram — подключи веса OSNet и реализуй ветку
   `osnet` в `appearance_tracker`-эмбеддере (сейчас `--track-embedder osnet` делает fallback→histogram
   с пометкой в мете). Для первого прогона **histogram достаточно**.

## Прогон (профиль: Segmenter + core_object_detections + action_recognition)
Фикстуры: `DataProcessor/docs/component_reports/action_recognition/fixtures/`
- `ar_real_4m35_people.mp4` (275 s), `ar_real_8m00_people.mp4` (481 s),
  `ar_real_2m47_control_nopeople.mp4` (167 s, контроль без людей).

core_object_detections — трекинг включён по умолчанию; ключевые флаги (дефолты уже стоят):
```
--track-enabled true --track-embedder histogram
--track-sim-gate 0.5 --track-reid-sim-gate 0.6 --track-w-app 0.7 --track-w-mot 0.3
--track-max-age-steps 3 --track-max-lost-steps 10 --track-min-conf 0.3
```
action_recognition — `--clip-len 32`. Окна Segmenter: дефолт `clip_len=32, window_hop_s=2.0,
max_windows=48` (можно задать в cfg action_recognition: `window_hop_s`, `max_windows`).
**Обнови `sampling_policy_version`** (окна меняют выборку — иначе config_hash не отразит).

## Golden
Прогони `ar_real_4m35_people.mp4` **дважды** → сравни `detections.npz:track_ids` и
`action_recognition_features.npz:clip_embeddings` на идентичность (детерминизм трекера+модели fp32).

## Проверка контракта (после прогона)
```
DataProcessor/.data_venv/bin/python \
  DataProcessor/VisualProcessor/modules/action_recognition/utils/validate_action_recognition_npz.py \
  <rs_path>/action_recognition/action_recognition_features.npz
```
Ожидается `✅ ... соответствует schema v3` (или валидный empty на контрольном без людей).

## Что собрать в artifacts/ (для REPORT)
Per video: v3-npz (shapes всех `clip_*`, `clip_count`, `num_tracks`, `mean_clips_per_track`,
`classes_available`, top-действия через `class_names`); из `detections.npz` — `meta.tracking`
(`num_tracks`, `mean_track_len`, `frac_single_len`), распределение длин треков.
Прокси-метрики трекера (крит. приёмки, EMBEDDING_TRACKER §8): intra- vs inter-track cosine,
доля 1-клиповых треков (было ~100%), re-ID на 8:00 (id сохраняется после ухода из кадра —
рендер боксов с id). Тайминги стадий + **overhead трекинга** (`meta.stage_timings_ms.tracking`).

## На что смотреть (что подтвердит корректность реализации)
- `classes_available=true` (значит `raw[:,:400]` реально дал логиты; если false — голова вернула
  не 400-d, тогда логиты снимать иначе — сообщи, поправлю тап).
- Фрагментация ушла: `num_tracks` на 4:35/8:00 в разы меньше, чем 205/199 из прошлого прогона;
  `mean_clips_per_track > 1`.
- Контроль без людей → валидный empty.
