# Прогресс валидации: core_depth_midas

Сессия начата: 2026-07-06. Валидатор-агент.

## Статус: ✅ ЗАШТАМПОВАН (v3, 2026-07-06). Сессия закрыта, под погашен.

Владелец: «Штампуем». C2 переопределён на CV≥0.10 (в CRITERIA.md). Штамп ✅ в COMPONENT_VALIDATION_CHECKLIST.md.
Под остановлен (pod_control stop_all, бюджет $5 цел).

## Итог прогона (2026-07-06)
- 7 видео 5.7c…847.7c, все rc=0. Golden identical ×7, max|Δ|=0.0. raw/norm NaN=0 (983 кадра).
- Все критерии ✅ КРОМЕ C2: complexity различима (CV=18.8%), но абсолютный std=0.00076 < согласованного 0.01
  (порог мискалиброван до данных). Предложил владельцу переопределить C2 → CV≥0.10 (факт 0.188).
- REPORT: `component_reports/core_depth_midas/REPORT_2026-07-06.md`. Ledger + DECISIONS обновлены.
- Ждём ответ владельца → штамп в COMPONENT_VALIDATION_CHECKLIST.md + погасить под.

## Что сделано
- [x] Прочитан онбординг + DECISIONS_AND_LESSONS.
- [x] Брифинг владельцу отправлен, критерии одобрены → `CRITERIA.md` записан (согласован 2026-07-06).
- [x] Решения по открытым вопросам приняты (владелец: «реши сам»):
  - Seq для Encoder: валидируем ОБЕ ветки — per-frame агрегаты (dense time-series) + `depth_maps_norm`.
    В отчёте отметить, что практическая seq для трансформер-Encoder = per-frame агрегаты (карты N,256,256 тяжелы).
  - Expected-empty: by-design НЕТ (глубина есть для любого кадра). No-fallback падение при пустых frame_indices — корректно.
  - Golden: на GPU. При недетерминизме bicubic — фиксируем max|Δ|≤1e-3 как GPU-стохастичность (C4).
- [x] Изучен код: main.py (v2.2, schema v3), validate_core_depth_midas_npz.py, run_depth_local.py,
      audit_v4_npz_stats.py, конфиг `visual_core_depth_midas_only.yaml`, SCHEMA/FEATURE_DESCRIPTION.

## Ключевые факты по коду
- Раннер: `DataProcessor/scripts/run_depth_local.py` — Segmenter(depth-профиль) → core_depth_midas
  (--runtime inprocess MiDaS_small, out 256×256, preset midas_256, batch 8) → validate(--struct --ranges) → golden.
- Golden в раннере: 2-й прогон в rs2, сравнение sha256 ключевых массивов. При НЕидентичности раннер
  даёт только diff_keys — max|Δ| нужно считать вручную (доп. скрипт).
- Валидатор: --struct (ключи/N/формы), --ranges (norm∈[0,1], p05≤p95, std≥0, preview_k, times_s неубыв.),
  --qa (плоский meta по view_csv_feature_qa.json). U1 требует ВСЕ три флага.
- depth_maps инициализируются NaN, но main.py падает если карта невалидна (isfinite.any()==False) →
  nan_rate raw = 0 by-design при успехе.
- norm: robust p05/p95 scaling + clip[0,1]. complexity = mean|grad| по norm-карте. fg_bg = range/(std+eps).

## Осталось
- [ ] Поднять окружение на поде (venv, ffmpeg, torch.hub MiDaS_small).
- [ ] Прогон на матрице длин видео (~10с…8мин+): run_depth_local.py --golden по каждому.
- [ ] Валидатор --struct --ranges --qa (rc=0) на всех.
- [ ] audit_v4_npz_stats.py агрегат + feature_quality_audit при наличии.
- [ ] Golden max|Δ| если не побайтово идентично.
- [ ] Проверить Segmenter budget depth на коротких/длинных (плотность frame_indices).
- [ ] REPORT с сырыми числами по каждому критерию → ledger → урок в DECISIONS → ИТОГ владельцу.

## Под
- Выдан владельцем: 213.173.110.201:19112.
