# FINAL REPORT — `story_structure`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `story_structure` (VisualProcessor `BaseModule`, Tier-3, CPU-only агрегатор) |
| Версия кода | `3.0.2` |
| Схема NPZ | `story_structure_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/story_structure/story_structure.npz` |
| Модель | **нет** — CPU (numpy/scipy), z-score-комбинация upstream-сигналов |
| Hard deps | `core_clip` + `core_optical_flow` + `core_face_landmarks` |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → story_structure ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L2 (min_frames=30 на коротких) |
| Код | `DataProcessor/VisualProcessor/modules/story_structure/utils/story_structure.py` (1102 строки) |

## 2. Резюме

`story_structure` — **анализатор нарративной структуры видео**: из семантической динамики (`core_clip`),
движения (`core_optical_flow`) и присутствия лиц (`core_face_landmarks`) строит **кривую «энергии истории»**
(`story_energy_curve`, z-score комбинация сигналов), находит **hook** (сила зацепа в начале), **climax**
(пик энергии), пики нарратива, экранное время главного героя — итого **22 скаляра** + кривая + поток пиков.
На реальном корпусе **ядро живо и различимо** (кривая энергии с пиками, climax_position 0.13–1.0, число пиков
0–4, main_character_screen_time 0.0–0.62). **Но** ровно как у соседних компонентов: (1) `hook_to_avg_energy_
ratio` в реальном батче **отравлен значениями ±10⁵–10⁶** (пре-фикс баг знаменателя, фикс закоммичен, но батч не
пере-прогнан) — это буквально фича, способная сломать модель; (2) `topic_shift_curve` **мёртв** на всём корпусе
(нет OCR/text deps в visual-standalone). golden Δ=0, min_frames=30 (L2, патч).

## 3. Функционал

Стоит в Tier-3 (агрегатор), после core_clip/core_optical_flow/core_face_landmarks. Логика:

1. **Компонентные кривые** — семантическое изменение (косинус соседних CLIP-эмбеддингов), motion (из
   core_optical_flow), присутствие лиц; каждая сглаживается и **z-score**-нормируется.
2. **story_energy_curve** = z-score комбинации сглаженных компонент — «энергия повествования» по времени.
3. **Hook** — сила/характеристики первых секунд (visual surprise, motion intensity, cut rate, rhythm, face).
4. **Climax/peaks** — пик энергии (`climax_frame_index` union-domain, time, position_norm, strength),
   `number_of_peaks`, `time_from_hook_to_climax`, `hook_to_avg_energy_ratio` (Sharpe-style).
5. **Персонаж/тема** — `main_character_screen_time` (доля кадров с лицом), `topic_shift_*` (из текста, обычно off).

**Зачем продукту:** структура повествования — **прямой драйвер удержания**: сильный хук в первые секунды
удерживает зрителя, наличие кульминации структурирует внимание. Это и model-сигнал (форма нарратива ↔
retention), и понятная креатору аналитика («слабый хук», «кульминация в конце», «N сюжетных пиков»).

## 4. Вход

- **`core_clip`** (hard) — эмбеддинги для семантической компоненты энергии.
- **`core_optical_flow`** (hard) — motion-кривая.
- **`core_face_landmarks`** (hard) — присутствие лиц (главный герой, hook_face).
- **`union_timestamps_sec`** + Segmenter `frame_indices` — ось.
- **`min_frames`** (дефолт 30) — N<min → **RuntimeError, нет empty-пути** (L2; в батче патч до 8, N=12 прошёл).
- OCR/text (soft) — для topic_shift; в visual-standalone off → topic_shift NaN.

## 5. Выход

- **Кривые:** `story_energy_curve (N,)` (z-score) + `story_energy_curve_downsampled_128`, `topic_shift_curve (N,)`
  (NaN by design без текста), `embedding_change_rate_per_sec`, `embedding_sim_next`/`diff_next`, `motion_norm_per_sec_mean`.
- **Пики:** `story_energy_peaks_idx`/`times_s`/`values_z`, `topic_shift_peaks_idx`.
- **22 скаляра:** `feature_names`/`feature_values` — hook_* (7), climax_* (5+peaks), `hook_to_avg_energy_ratio`,
  `main_character_screen_time`, `speaker_switch_*`, `topic_shift_*`, `n_frames`, `video_length_seconds`.
- **`any_face_present`**, `frame_feature_present_ratio`. Ось: `frame_indices`, `times_s`.

## 6. Фичи (важное/неочевидное)

- **`story_energy_curve` (z-score)** — несущая фича: единый «пульс» видео из motion+семантики+лиц. На реальных
  данных живая, с выраженными пиками. Z-score делает её сравнимой между видео.
- **`climax_position_normalized`** — где пик энергии (0=начало, 1=конец); на корпусе 0.13–1.0 (реально различает
  структуру: ранняя vs финальная кульминация).
- **`main_character_screen_time`** — доля кадров с лицом (0.0–0.62): прокси «человеко-центричности» контента.
- **`hook_to_avg_energy_ratio` — ОТРАВЛЕН в батче (±10⁶).** По задумке — Sharpe-style (hook_energy / std(z)),
  разумный диапазон ~±50. Но пре-фикс версия делила на `mean(z)≈0` → ±880k. В storage все 6 видео с этими
  экстремумами. Фича, буквально ломающая модель при сыром вводе (§9).
- **`topic_shift_curve` — NaN by design без текста** — в visual-standalone OCR/text не подключены → present=0.

## 7. Алгоритм / архитектура

- **Чистый CPU** (numpy/scipy): сглаживание, z-score, `find_peaks`, косинусы. Модели нет.
- **Сложность:** <1 c/видео (без загрузки deps); узкое место — I/O core_clip/optical_flow NPZ.
- **Детерминизм:** golden max|Δ|=0.0 (numpy+scipy полностью детерминированы).

## 8. Оптимизации

- **Reuse трёх core-провайдеров** — не считает эмбеддинги/motion/лица заново.
- **z-score-нормировка компонент** — делает энергию сравнимой между видео (осознанно).
- **downsampled_128 кривая** — компактная версия для UI/модели фикс-длины.
- **frame_feature_present_ratio** — трактовка NaN (тот же образцовый паттерн).

## 9. Слабые места

- **`hook_to_avg_energy_ratio` отравлен на 100% реального корпуса (главное).** Все 6 видео: ±10⁵–10⁶
  (−887k, +1071k, −123k, +43k, −256k, −788k). Пре-фикс баг (деление z-score на его же mean≈0). Фикс
  (знаменатель→std, ~±50) закоммичен, но **батч не пере-прогнан** → фича в данных катастрофична: без клиппинга
  сломает и baseline, и Encoder (взрыв градиентов/шкал). Хуже NaN — это «ядовитое» валидное число.
- **`topic_shift_curve` мёртв на всём корпусе** — present=0, curve NaN, `topic_shift_peaks_count` бесполезен
  (OCR/text не подключены в visual-standalone). Ещё 2 фичи неинформативны.
- **`speaker_switch_rate`/`speaker_switches_per_minute`** — аудио/диаризация-driven, в visual-standalone
  почти наверняка неактивны (наследует ту же проблему модальности, что audio в high_level_semantic).
- **min_frames=30 no-fallback, нет empty-пути** (L2) — N<30 → RuntimeError (не status=empty). В батче патч до 8,
  но «правильный» fix (по длительности / empty) не сделан — хрупко (как video_pacing).
- **Много репака** — как и high_level_semantic, компонент в основном комбинирует чужие сигналы; собственное
  новое = энергетическая крив021 + hook/climax-эвристики.
- **Эвристики hook/climax без калибровки** — веса/пороги подобраны вручную.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер данных] Пере-прогнать storage с фиксом hook_ratio** — сейчас в реальных артефактах фича
   ±10⁶; любой downstream, читающий `feature_values` без клиппинга, будет отравлен. Критично до обучения.
2. **[выс.] Довести L2-fix до правильного** — min_frames от длительности/policy или status=empty вместо
   RuntimeError/патча-8 (общий пункт с video_pacing).
3. **[сред.] Включить topic_shift** (OCR/text deps на этапе fusion) или убрать 3 мёртвые topic-фичи из контракта.
4. **[сред.] Защитный клиппинг/нормировка** экстремальных ratio-фич на выходе (guard против будущих div-by-~0).
5. **[низ.] Калибровать hook/climax-эвристики** на разметке «сильный/слабый хук».

## 11. Рекомендации по архитектуре / связям

- **Естественный fusion-компонент** (как high_level_semantic) — topic_shift/speaker требуют текст/аудио,
  доступных только над процессорами; логично считать story_structure на уровне fusion, а не в visual-standalone.
- **Reuse трёх core-провайдеров закреплён** — убедиться в shared sampling group (общий frame_indices).
- **story_energy_curve + peaks** — кандидат в прямой вход Encoder (нарративный таймлайн); согласовать с Models.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct | 23 batch + 3 pod | rc=0 | контракт ок |
| U2 ось времени | 23 | frame_indices↑ | ось корректна |
| U3 finite/health | 3 | nan_fv=0%, curve finite; topic NaN by design | health объясним |
| U4 expected-empty | N<30 | RuntimeError by design (нет empty-пути) | L2-хрупкость задекларирована |
| U5 golden | 2 прогона | max\|Δ\|=0.0 | детерминизм |
| U6 разные длины | N=43/65/119 | 22/22 finite, rc=0 | масштабируется |
| C1 curve finite | 100% | кривая надёжна |
| C2 feature finite/climax∈[0,1] | 22/22, climax 0.13–1.0 | скаляры валидны, различимы |
| C3 hook_ratio (после фикса) | -1.18/-0.16/+0.07 (**на pod, после фикса**) | разумно ПОСЛЕ фикса |
| **Реальный storage (мой прогон)** | 6 видео | energy/climax/peaks/faces живы; **hook_ratio ±10⁵–10⁶; topic_shift=0** | ядро живо, hook_ratio отравлен |

Вывод: **нарративное ядро живо и различимо**, но `hook_to_avg_energy_ratio` в реальном батче отравлен
(пре-фикс), а topic_shift/speaker мёртвы — до пере-прогона часть вектора опасна/бесполезна.

## 13. Интерпретируемость

- **Сильнейшая сторона:** нарратив интуитивен. `story_energy_curve` с пиками = «график напряжения истории»;
  hook/climax — прямые понятные креатору маркеры. `render.py` есть.
- **Добавить:** словесная оценка «сильный хук / кульминация в конце / 3 сюжетных пика»; overlay кривой энергии
  на таймлайн видео с отметками hook/climax; сравнение формы с успешными роликами ниши.

## 14. Польза для моделей

**Средняя (с оговоркой на отравление).** `story_energy_curve`+пики+climax_position — осмысленная нарративная
ось, правдоподобно связанная с удержанием (хук/кульминация ↔ retention), в удобной форме (кривая + downsampled_128
+ скаляры). **Но** на реальных данных `hook_to_avg_energy_ratio`=±10⁶ отравит модель без клиппинга, а topic/speaker
мертвы. До пере-прогона фактическая польза снижена риском; потенциал — 4 после фикса.

## 15. Польза для аналитиков

**Высокая.** «Структура истории» — понятнейшая креатору аналитика: сила хука, где кульминация, сколько пиков,
экранное время героя, график напряжения. Наглядно и сравнимо. Ограничения: topic_shift/speaker пусты,
hook_ratio в текущих данных нечитаем (±10⁶ до пере-прогона), эвристики без калибровки.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Нарративная энергия+hook/climax — ценная самостоятельная ось |
| 5. Выход (контракт) | 4 | Богатый curve/peaks/22-скаляра + present_ratio; часть фич мертва/отравлена |
| 6. Фичи | 3 | Энергия/climax/faces сильны; hook_ratio отравлен, topic/speaker мертвы |
| 8. Оптимизации | 4 | Reuse 3 deps, z-score, downsampled, present_ratio |
| 9. Слабые места (инверсно) | 2 | hook_ratio ±10⁶ в батче, topic/speaker мертвы, L2-хрупкость, репак |
| 12. Результаты тестов | 3 | Гейты PASS + ядро живо, но hook_ratio отравлен в реальных данных |
| 13. Интерпретируемость | 5 | Нарратив (хук/кульминация/график напряжения) — предельно понятен |
| 14. Польза для моделей | 3 | Энергия/climax ценны, но hook_ratio отравляет, topic мёртв |
| 15. Польза для аналитиков | 4 | Структура истории наглядна и сравнима; часть фич пуста/нечитаема |

### Итоговые оценки

- **Польза для моделей: 3/5.** Нарративная энергия + hook/climax — правдоподобно сильная ось для удержания в
  удобной форме (кривая + скаляры). Но в реальном батче `hook_to_avg_energy_ratio`=±10⁶ (пре-фикс) способен
  отравить модель без клиппинга, а topic_shift/speaker мертвы — фактическая польза ограничена до пере-прогона
  (потенциал 4).
- **Польза для аналитиков: 4/5.** «Структура истории» (сила хука, кульминация, сюжетные пики, график напряжения)
  — одна из самых понятных и наглядных креатору аналитик. Балл ниже 5 держат пустые topic/speaker, нечитаемый
  в текущих данных hook_ratio и некалиброванные эвристики.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/story_structure/utils/story_structure.py` (1102 строки), `legacy_story_structure.py`
- `.../story_structure/{main.py,utils/validate_story_structure.py,utils/render.py,docs/SCHEMA.md}`
- `DataProcessor/docs/component_reports/story_structure/{REPORT_2026-07-16.md,CRITERIA.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L2 min_frames)
- Cross-ref deps: `core_clip`, `core_optical_flow`, `core_face_landmarks` (FINAL_REPORTs)
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/story_structure/story_structure.npz`
  (ядро живо; **hook_to_avg_energy_ratio=±10⁵–10⁶ пре-фикс; topic_shift_present=0**)

## 18. Визуализации

![story_structure overview](story_structure_overview.png)

`story_structure_overview.png`: слева — `story_energy_curve` (z-score) с найденными пиками (нарративное ядро
живо и осмысленно); справа — |`hook_to_avg_energy_ratio`| в лог-шкале на 6 реальных видео: **±10⁵–10⁶** против
ожидаемого пост-фикс диапазона ≤~50 (зелёный пунктир) — пре-фикс баг знаменателя, не пере-прогнанный в батче.
Подтверждает раздельный вердикт: энергия/кульминация здоровы, но hook_ratio отравлен и требует пере-прогона.
