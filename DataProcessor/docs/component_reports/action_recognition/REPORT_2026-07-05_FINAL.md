# Финальный отчёт: action_recognition — штамп v3 (2026-07-05)

Автор: Claude. Вход: прогоны v1→v3.2 (`RUN_RESULT*.md` + `artifacts/`), реализация §D+ASSESSMENT.
Вердикт: **✅ ЗАШТАМПОВАН (action_recognition v3, прод-готов по логике).**

## Путь доработки (кратко)
v1 (SlowFast, per-track) → нашли: фрагментация треков, невалидный контроль, нет классов, эмбеддинг =
проекция логитов, padded-клипы, `mean_clips_per_track=1`. Закрыто итерациями v2→v3.2:

| Итерация | Что сделано | Итог |
|---|---|---|
| v2 | appearance-tracker (свой, эмбеддинг боксов) + Segmenter dense-окна + per-clip v3 + валидатор | фрагментация 205→26 треков, контракт v3 |
| v3 | penultimate-эмбеддинг (hook, 2304-d), классы Kinetics (softmax головы), tubelet, localization, input-validator, метрики, OSNet-ветка, fp16, очистка frames | классы осмысленны, tubelet различает действия по трекам |
| v3.1 | реальные метки Kinetics, detection на dense-кадрах | треки полные (mean_len 77), паддинг убран |
| v3.2 | окно = clip_len×3 (96), окон ×3 меньше → тот же бюджет | **`mean_clips_per_track`=4.0/4.06** |

## Финальные метрики (v3.2, проверено по NPZ)

| video | status | clip_count | tracks | mean_clips/track | clips/track dist | emb | classes | golden |
|---|---|---:|---:|---:|---|---|---|---|
| 4:35 | ok | 44 | 11 | **4.0** | [5×8, 2, 1] | (44,2304) L2 | ✅ | идентичен |
| 8:00 | ok | 65 | 16 | **4.06** | median 5 | (·,2304) L2 | ✅ | — |
| 2:47 control | empty (`no_person_detections`) | 0 | 0 | — | — | — | — | ✅ (v3.1) |

Бюджет кадров 1536 (16 окон×96) = как в v3.1; диск ~10.7 GB/видео (не вырос).

## Оценка по 4 осям (финал)
- **Корректность:** ✅ контракт v3 (input+output валидаторы pass), `clip_embeddings (C,2304)` L2/finite,
  `clip_times_s ⊆ union`, классы Kinetics снимаются (`classes_available=true`), контроль valid-empty.
- **Стабильность:** ✅ golden ×2 побитово идентичны (детерминизм трекера + fp32 backbone).
- **Различимость:** ✅ appearance-треки разделяются (intra≫inter); tubelet даёт разные действия
  разным трекам на групповой сцене; классы правдоподобны (clarinet/harmonica; yoga/marching).
- **Модель-fit:** ✅ плоский per-clip stream, эмбеддинг = **настоящие penultimate-фичи backbone**
  (не проекция логитов), плотность per-track ~4–5 клипов (траектория действия для Encoder).

## Что подтверждено как прод-готовое
1. Свой appearance-embedding трекер (`core_object_detections` schema v3, `track_ids`) — решил корень фрагментации.
2. Segmenter: dense-окна `clip_len×window_len_mult`, адаптивность, детекция выровнена под окна.
3. action_recognition v3: penultimate-эмбеддинг + Kinetics-классы + `clip_track_id`/`clip_segment_id`
   + агрегаты; per-person tubelet; temporal localization (track-anchored).
4. Инфра: input+output валидаторы, метрики `metrics.{json,prom}`, доки SCHEMA/FEATURE/контракты v3.
5. Оптимизация: fp16-опция, батч-путь v3-safe, очистка frames, косто-нейтральные окна.

## Остаточные (не блокеры штампа; плановые улучшения)
- `num_action_segments = num_tracks` (1 сегмент/трек): change-point внутри трека не срабатывает
  (эмбеддинги трека похожи). Если нужны более тонкие границы действий — поднять `seg_cos_threshold`
  или сегментировать по логит-классу, а не по эмбеддингу. **Аналитическая тонкость, не влияет на модель-выход.**
- OSNet ReID — ветка готова, но требует `torchreid` в env (сейчас histogram). Включить, если на
  толпном/длинном контенте потребуется лучшее разделение id (по метрике intra/inter).
- VideoMAEv2/Hiera как backbone — по результатам baseline-ablation (Models workstream).
- 200k/диск: `run_ar_local.py`/k8s-Job + `cleanup_frames_after_npz.py` — политика очистки кадров обязательна.

## Итог
**action_recognition v3 — штамп ✅.** Все оси зелёные, регрессий нет, главный критерий
(`mean_clips_per_track>1`) закрыт косто-нейтрально. Компонент готов как источник action-токенов и
Kinetics-распределений для Models/Encoder и аналитики. Остаточные пункты — плановые апгрейды,
не требуются для валидности логики.
