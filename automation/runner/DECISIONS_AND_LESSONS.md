# Решения и уроки (институциональная память)

> Append-only. Агент ЧИТАЕТ этот файл в начале каждой сессии и ДОПОЛНЯЕТ его в конце (новые уроки,
> подтверждённые решения, грабли). Цель — не переучиваться и не повторять ошибки между сессиями.
> Формат записи: дата · компонент/область · суть · почему.

## Окружение / прогоны
- **mediapipe `<0.10.15`** обязателен: 0.10.35 удалил `mp.solutions` → AttributeError (core_face_landmarks,
  shot_quality/face). Прошито в pod_setup.sh.
- **ffmpeg** на свежем/перезапущенном контейнере ставить `apt-get update && apt-get install -y ffmpeg`
  (без update — «Unable to locate package»).
- **Persistent venv** `/workspace/venv` (`python3 -m venv --system-site-packages`) на Network Volume —
  переживает рестарт пода; extras ~211 МБ. `git+CLIP` ставится медленно.
- **Не качать весь HF-датасет.** `snapshot_download(Ilialebedev/videos11)` без фильтра утащил 4930 файлов/
  85 ГБ и забил 100 ГБ volume. Всегда `allow_patterns=["<id>.mp4"]`. Держим ~30 видео разной длины в
  `/workspace/scene_videos`.

## Поды / RunPod
- **pod-id МЕНЯЕТСЯ при миграции.** Никогда не хардкодить id — останавливать/находить поды по списку из API
  (`stop_all_running` перебирает RUNNING). Из-за хардкода старого id под однажды «не остановился».
- **REST не отдаёт список GPU/цены** — только GraphQL (`gpuTypes`). Создание пода — REST POST /v1/pods.
- **Triton не обязателен:** core_clip inprocess (`clip.load ViT-B/32`); cut_detection farneback
  (`--no-require-core-optical-flow --no-use-clip`); depth — inprocess MiDaS-обход (`midas_depth_inprocess.py`).

## Логика компонентов
- **Строгое выравнивание кадров:** shot_quality требует, чтобы все 5 deps имели идентичные frame_indices
  (Segmenter aligned sampling). Ось времени всегда из source-of-truth (`union_timestamps_sec` / `times_s`).
- **Valid-empty — это фича, не ошибка:** на видео без лиц core_face_landmarks отдаёт `has_any_face=False` +
  массивы правильной формы; downstream (shot_quality face-ROI) считает это NaN by design. Валидаторы это учитывают.
- **NaN by design бывает штатным** (shot_quality face-ROI; similarity_metrics/uniqueness — политика NaN).
  Такие случаи фиксировать в CRITERIA.md компонента как явное исключение, а не «чинить».
- **appearance-tracker** (histogram HSV + Hungarian) даёт когерентные person-треки (mean 52–127, max ~295,
  frac_single≈0). schema v3, `track_ids (N,M)`. Каноничный вес детекции — `yolo11x_41_best.pt` (таксономия
  владельца), но на трекинг не влияет (person=class 0).
- **core_clip — это хаб:** CLIP-эмбеддинги считаются ОДИН раз и переиспользуются scene/shot/cut/similarity.
  Для 200k — не дублировать CLIP-инференс по компонентам.

## Процесс
- Отчёты — в `DataProcessor/docs/component_reports/<component>/REPORT_YYYY-MM-DD.md`; статусы — в
  `COMPONENT_VALIDATION_CHECKLIST.md`; тайминги/цены — в `automation/RESOURCE_TIMING_LEDGER.md`.
- Golden-детерминизм проверять там, где есть источники недетерминизма (CLIP/детекции — да; чистая numpy — нет).

---
## Новые записи (агент дополняет ниже)
<!-- дата · область · суть · почему -->
- **2026-07-06 · core_depth_midas · golden на GPU ПОБАЙТОВО детерминирован** (max|Δ|=0.0 ×7 видео).
  MiDaS_small через torch.hub + `F.interpolate(mode="bicubic")` + `inference_mode/eval` дают идентичные
  массивы между прогонами на RTX 2000 Ada. Т.е. для depth не нужен «порог GPU-стохастичности» — фиксируем 0.
- **2026-07-06 · core_depth_midas · порог различимости надо задавать масштабо-инвариантно.** Предложил в
  брифинге C2 = «std complexity между роликами > 0.01» ДО данных. Факт: complexity — средний |градиент| по
  норм.[0,1] карте 256², её абсолютная шкала ~0.004, std=0.00076 < 0.01, но CV=18.8% (самый большой из прокси).
  Урок: для незнакомой метрики порог различимости брать в CV/относительных единицах, не в абсолютных вслепую.
- **2026-07-06 · core_depth_midas · inprocess-инициализация ~20–27с/подпроцесс** — это torch.hub.load
  (импорт+кеш-чек+перенос на GPU), а не инференс (0.78 мс/кадр). В раннере каждый ролик = новый процесс, поэтому
  init платится каждый раз; в проде — разовая стоимость воркера. Не путать с per-video стоимостью.
- **2026-07-06 · инфра · QA-конфиг `storage/result_store/view_csv_feature_qa.json` ОТСУТСТВУЕТ во всём репо.**
  Валидаторы с `--qa` штатно печатают «QA: пропуск» и возвращают rc=0 (не падают). CSV-view/QA слой ещё не
  материализован — это ожидаемо, не дефект компонента. Учитывать при трактовке U1 с флагом --qa.
- **2026-07-06 · поды · pod_control action=status падал** («'str' object has no attribute 'get'»), но под,
  выданный владельцем (host/port в задании), был жив — работал по SSH напрямую. budget_status работает.
- **2026-07-11 · поды · лимит аккаунта RunPod = 2 пода одновременно.** «GPU недоступны во всём датацентре»
  при start/migrate в прошлой сессии — это НЕ отсутствие GPU на Network Volume, а упор в лимит: у владельца
  уже было создано 2 пода (пусть и EXITED/остановленных), и RunPod не давал поднять третий. Диагностика на
  будущее: если create/start упорно говорит «no instances», проверить число существующих подов (manager list /
  RunPod UI) — при 2 подах СНАЧАЛА forget/удалить лишний остановленный, затем создавать новый.
- **2026-07-06 · поды · репо на Network Volume может быть НЕПОЛНЫМ** (не git-репо): отсутствовали
  `scripts/run_depth_local.py` и `configs/audit_v3/visual/*.yaml`. Досинхронизировал scp с локального (источник
  истины). scp порт — `-P` (заглавная), а не `-p` как у ssh.

- **2026-07-11 · логика · ocr_extractor баг expected-empty (НАЙДЕН+ИСПРАВЛЕН).** В `main.py` при отсутствии
  класса `text_region` в таксономии детектора `proposal_ids` пуст, и условие отбора
  `if proposal_ids and int(class_ids[...]) not in proposal_ids: continue` схлопывается в False → фильтр по
  классу НЕ применяется и OCR-ятся ВСЕ боксы (противоречит warning «OCR will be empty», main.py:579). Фикс:
  после warning выставить `skip_ocr_processing=True; skip_ocr_reason="proposal_class_not_in_taxonomy"` (механизм
  skip уже был для tesseract_not_in_path). Результат: Dnobox → status=empty, rows=0, rc=0. Правка на поде
  (`main.py.bak_ocrfix` — бэкап), ТРЕБУЕТ переноса в git DataProcessor.
- **2026-07-11 · инфра · ocr_extractor валидируется В ИЗОЛЯЦИИ на синтетике** (`/workspace/ocr_synth_validate.py`):
  реального YOLO с классом text_region нет (зона владельца). Скрипт синтезирует frames_dir+detections.npz и
  гоняет настоящий main.py. Грабли: (1) venv на netVol НЕ содержал `onnxruntime` → ставить `pip install onnxruntime`
  (CPU-версии хватает для ppocr rec inference). (2) FrameManager требует в каждом batch metadata поля
  `start_frame`/`end_frame` (не только num_frames) — иначе KeyError в _load_batch_pid.
- **2026-07-11 · логика · rec_confidence на синтетике НЕ показателен.** ppocr_rec обучен на реальных кропах; на
  белом фоне с крупным DejaVu argmax верный (текст распознаётся правильно), но softmax размазан → max-prob ~0.005.
  Абсолютные conf на синтетике не калибровать, `min_rec_score` порог не настроить. Golden при этом идеален
  (max|Δconf|=0.0, ONNX детерминирован). Качество rec/калибровка — только на реальном детекторе.

- **2026-07-12 · behavioral · body_lean *5.0 насыщал в константу 1.0 (ФИКС подтверждён на B).** Убрал множитель →
  глобально std=0.239, frac==1.0=0.0, 546 уник/575. Урок: любой `clip(x*k,-1,1)` с большим k на признаке, чья сырая
  величина уже ~0.4, гарантированно даёт константу на границе — проверять насыщение ПЕРЕД клипом.
- **2026-07-12 · behavioral · «status=empty» ≠ «нет landmarks».** status=empty когда `not has_any_landmarks OR
  core_status=="empty"` (analyzer:1637). Видео с pose но без face-mesh → landmarks_present=True(все), но core пометил
  empty → behavioral проксирует empty_reason="no_faces_in_video". Это НЕ баг: seq pose-тира эмитится, engagement-агрегаты
  NaN. Для строгого U4 (lp=0, seq все NaN) нужен синтетик: сдвиг frame_indices в landmarks.npz на +999999 (0 пересечений).
- **2026-07-12 · behavioral · иерархия опор чистая и структурная.** 3 тира: ядро (23 поля, 0% NaN на present) ⊇ pose-тир
  (6 полей, co-NaN, 0 partial) ⊇ mouth/face-mesh-тир (6 полей, co-NaN). Проверка partial-NaN внутри тира=0 на всех видео —
  сильный признак корректной missing-policy (а не случайных дыр). Для Encoder: маска landmarks_present + per-feature isfinite.
- **2026-07-12 · behavioral · C4 порог NaN%≤20% строг для degraded-роликов.** Когда в ролике отсутствует целый тир
  landmarks (напр. нет pose — ItRcDFKFiSU status=ok но 35% агрегатов NaN: avg_arm_openness/pose/confidence/engagement),
  числовой порог превышается легитимно. Урок: пороги «доля NaN агрегатов» задавать условно на наличие тира, а не глобально.
- **2026-07-12 · behavioral · gesture-probs бимодальны, НЕ всегда сумма=1.** На present-кадрах: 47% =0.0 (num_hands=0,
  нет рук → нет жеста), 53% ≈1.0 (softmax, num_hands≥1). U3-критерий «сумма≈1» трактовать как «≈1 при наличии рук ИЛИ 0».

- **2026-07-12 · core_optical_flow · ШТАМП v3 (07-12), все гейты+критерии PASS.** inprocess raft_small (Triton-free,
  как depth), RANSAC-seed `cv2.setRNGSeed(0)` → cam_* детерминир., golden 3/3 diff=[]. Матрица 13 видео + пресеты 3×4.
  C1 CV=0.752, C2 динамика/статика 4.9×, C3 `consistency=1/(1+div)` err 3e-8. Дефолт raft_256_small.
- **2026-07-12 · core_optical_flow · пресет/модель почти не влияют на motion-АГРЕГАТЫ.** raft_{256,384,512}×{small,large}:
  mean/median motion расходятся ≤0.03, разделение статика/динамика стабильно. Кадр-к-кадру corr к 256_small: статика
  0.995+, динамика 0.70–0.89. Урок: для motion-фичи брать самый дешёвый пресет (256_small, 2× быстрее 512) — качество
  агрегатов не теряется; большие пресеты нужны только если downstream смотрит на per-frame форму, а не агрегаты.
- **2026-07-12 · core_optical_flow · bg_ratio ≈ 0.40 by design (НЕ баг).** Определён как mean(mag ≤ percentile40(mag)) →
  доля пикселей ниже 40-го перцентиля почти константна ~0.40. Валиден (∈[0,1]), но малоинформативен как фича — кандидат
  на исключение из seq. Урок: метрика вида «доля ниже фиксированного перцентиля» тавтологична — проверять определение фичи,
  прежде чем считать её сигналом.
- **2026-07-12 · core_optical_flow · ДЕФЕКТ batch-путь.** `VisualProcessor/utils/core_optical_flow_batch.py` пишет только
  motion_norm/dt/preview, НЕ заполняет audit-v3 per-frame фичи → batch-NPZ провалит структурный валидатор (_PER_FRAME).
  Валидация делалась per-video main.py. Синхронизировать batch перед прод-масштабом 200k.

- **2026-07-12 · color_light · БАГ video-level агрегации (НАЙДЕН+ФИКС подтверждён).** `color_distribution_gini=NaN`,
  `entropy≈0` в старых NPZ: в `processor.py::extract_video_features` hue_values читались `frame.get("hue_mean")`, но фичи в
  `frame["features"]["hue_mean"]` → всегда default 0.0. Все hue=0 → гистограмма в 1 бине (entropy≈0) и `_compute_gini`
  делит на sum=0 → NaN. Тот же дефолт занулял brightness/color change speed, strobe, periodicity, shift. Фикс: `getf(frame,key)`
  читает `frame["features"][key]`. После фикса gini=0.073/entropy=2.49 (совпало с ручным пересчётом). Уроки: (1) когда per-frame
  метрика валидна, а её video-агрегат=NaN — искать РАЗНЫЙ путь чтения (wrapper {"features":{...}} vs плоский dict), а не «чинить NaN»;
  (2) gini через `2Σ(i·x)/(nΣx)−(n+1)/n` = NaN при Σx=0 (все нули/пустой сигнал) — сигнал, что вход занулён.
- **2026-07-12 · color_light · golden CPU строг ТОЛЬКО при пиннинге потоков.** Multi-thread BLAS → 2/2128 compact-элем дрожат на
  1 ULP (max|Δ|=1.19e-7, frame_indices идентичны); `OMP_NUM_THREADS=1 OPENBLAS_/MKL_=1` → бит-идентично max|Δ|=0. Для строгого
  golden CPU-компонентов с numpy-редукциями пиннить потоки; ≤1 ULP дрожание качество фич не меняет, но ломает побайтовый diff.
- **2026-07-12 · color_light · expected-empty синтез = сдвиг индексов сцен.** color_light фильтрует `scenes[*].indices` по
  allowed frame_indices; сдвиг всех индексов сцен на +1e6 в scene_classification.npz → пересечение пусто → `after_filt_empty`:
  status=empty, compact (0,16) float32 NaN, все ключи, validator rc=0. Дешёвый детерминированный способ покрыть U4 без спец-видео.

- **2026-07-13 · браузер · клик по GPU-карточке RunPod.** Карточка — `<button data-testid="gpu-card-<ID>"
  data-ph-capture-attribute-gpu-card-selected="NVIDIA RTX 2000 Ada Generation">`. Кликать НАДЁЖНО по этой
  кнопке (селектор `button[data-ph-capture-attribute-gpu-card-selected*="<имя>"]`), а не по вложенному
  `<p>` (он невидим/виртуализирован, click/force не берут). Bonus: этот атрибут = точный gpuTypeId для API.
- **2026-07-13 · поды · нет GPU ≤ лимита → не простаивать.** Агент продолжает не-GPU работу по нескольким
  компонентам (анализ/правки/подготовка тестов) и часто (~2–3 мин) проверяет наличие GPU (`runpod_gpu_scraper.py`)
  — доступность мелькает, момент важно поймать.

- **2026-07-13 · micro_emotion · КОМПОНЕНТ БЫЛ МЁРТВ из-за leading-space OpenFace CSV (НАЙДЕН+ФИКС).**
  OpenFace FeatureExtraction пишет заголовки CSV с ВЕДУЩИМ ПРОБЕЛОМ (" AU12_r", " pose_Rx", " success", " x_0").
  pd.read_csv без skipinitialspace их сохраняет, а весь код читал голые имена (row.get('pose_Rx'),
  col.startswith('AU')) → ВСЕ обращения промахивались. Итог в 4 реальных ok-NPZ: compact22 при лице 20/22
  колонок КОНСТАНТА (std=0), 10 PCA-скаляров NaN, microexpr_count=0 везде, frame_features усечён до F=2.
  Фикс: df.columns=[c.strip()...] после read_csv (openface_analyzer + _load_openface_dataframe). Урок: для
  любого внешнего CSV-движка (OpenFace/и т.п.) СРАЗУ нормализовать имена колонок; симптом «фича-константа/весь
  вектор нули + PCA NaN» на реальных NPZ = искать сломанное ЧТЕНИЕ входа, а не «чинить NaN».
- **2026-07-13 · micro_emotion · перепутанные метки compact22 (2-й баг).** Порядок append() в
  compute_per_frame_vectors (pose_Rz@10,pose_Tz@11, потом gaze/mouth) НЕ совпадал с контрактом
  COMPACT22_FEATURE_NAMES (gaze@10-12,…,pose_Rz@20,pose_Tz@21) → Encoder читал не те столбцы. Проверять
  соответствие ФИЗИЧЕСКОГО порядка сборки вектора и списка имён-контракта эмпирически (сверка std по столбцам
  на синтетике со сдвинутыми метками ловит это мгновенно). Фикс: сборка строго по именам контракта.
- **2026-07-13 · micro_emotion/инфра · OpenFace только через Docker --gpus all → под RunPod (сам контейнер)
  docker-in-docker недоступен.** Логику постобработки валидировать СИНТЕТИЧЕСКИ: реалистичный OpenFace-df
  (с ведущими пробелами + варьируемые AU/pose/gaze/landmarks) → process_openface_dataframe напрямую (класс
  MicroEmotionProcessor лёгкий, не тянет docker). Тест до/после фикса + идеал (без пробелов) как эталон.
- **2026-07-13 · ридер · renderer._convert_numpy_to_python СХЛОПЫВАЕТ zero-size массив в [] (renderer.py:42
  `tolist() if size>0 else []`).** Массив (N,0) читается как 1D (0,) → строгие shape-валидаторы падают на
  штатном empty-пути (frame_features F=0). Чинить в валидаторе компонента (принять F=0 при status=empty),
  а не в общем ридере. Возможен тот же подвох у других компонентов с пустыми wide-фичами.
