# Waivers / известные исключения — батч 60+

**Правило:** любой waiver должен быть **письменным** (этот файл или тикет) с датой и ссылкой на пункт [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md).

| ID | Пункт чеклиста | Статус | Решение | Дата | Кто |
|----|----------------|--------|---------|------|-----|
| W1 | 2.3 micro_emotion | resolved (код) | В `compute_au_pca` число компонент ограничено `min(pca_components, n_samples, n_features)`; при `max_comp<1` PCA не строится; иначе проекция дополняется нулями до `pca_components` — см. `VisualProcessor/modules/micro_emotion/utils/micro_emotion_processor.py`. Smoke: `VisualProcessor/.vp_venv/bin/python` (2 кадра, `pca_components=3`) без `ValueError`. | 2026-04-15 | — |
| W2 | 2.1 Text 5/5 | pending | См. [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) §3 (CUDA OOM) | | |
| W3 | 2.4 action_recognition | accepted (waiver) | Пилот **A+B (5 run, 2026-04-13)** не проверяет многоклиповую временную ось: в агрегате L2 `tracks_with_multi_clips_total=0`, у каждого трека `metric__num_clips=1` → `metric__*temporal_jump*`, `metric__num_switches` и смысл «стабильности между клипами» на этом наборе **тривиальны**. Для батча 60+: либо включить в реестр видео, где person-треки дают **>1 клип** на трек (см. [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md), [COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md)), либо не трактовать эти метрики как валидированные пилотом. Эвиденс: `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json`, [RUN_LOG.md](RUN_LOG.md) §action_recognition. | 2026-04-15 | — |
| W4 | 2.0.4 place_semantics | accepted (valid empty) | `status="empty"` при **`empty_reason="no_places_detected"`** — штатно, если Embedding Service и БД мест доступны, но по кадрам/трекам не найдено совпадений с порогом (см. [place_semantics README](../../VisualProcessor/core/model_process/core_identity/place_semantics/README.md) §*Empty vs Error semantics*). На full-max E2E 2026-04-20 встречалось *empty* в сводке — **не** считать багом без другого `empty_reason` (например `embedding_service_unavailable_during_processing` при живом заранее health-check’е). | 2026-04-22 | — |
| W5 | 2.0.5 source_separation_extractor | accepted (valid empty) | `status="empty"` с **`audio_too_short`** (короткое аудио) или **`audio_silent`** — **контрактные** ветки ([source_separation AUDIT v3](../../AudioProcessor/docs/audit_v3/components/source_separation_extractor_AUDIT_V3_REPORT.md) §4; реализация: [`source_separation_extractor/main.py`](../../AudioProcessor/src/extractors/source_separation_extractor/main.py)). Повторяемое *empty* на mock/коротких сегментах в E2E — ожидаемо. | 2026-04-22 | — |
| W6 | 2.0.4 car_semantics (доп. к W4) | accepted (valid empty) | `status="empty"` при **`empty_reason="no_car_proposals"`** — нет валидных car-детекций/кропов для поиска в БД ([`car_semantics/main.py`](../../VisualProcessor/core/model_process/core_identity/car_semantics/main.py): `candidates` пуст). Не путать с error при отсутствии YOLO/ES при живых deps. | 2026-04-22 | — |

*Заполняйте таблицу при принятии решения. Пустые строки — черновик.*
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
