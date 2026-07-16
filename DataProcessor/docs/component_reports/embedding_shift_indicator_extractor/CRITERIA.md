# Критерии приёмки: embedding_shift_indicator_extractor

**Версия компонента**: 1.3.0
**Дата согласования**: 2026-07-17
**Согласовано**: Второй агент (от имени владельца)

---

## Универсальные хард-гейты

| Гейт | Описание | Ожидание |
|------|----------|----------|
| U1 | Валидатор rc=0 | validate_embedding_shift_indicator_extractor_text_npz.py → 28/28 OK |
| U2 | Ось времени | N/A — flat-features компонент, нет temporal axis (frame_indices/timestamps) |
| U3 | finite/различимость | cosine_begin_end ∈ [-1,1] при present=1; разные эмбеддинги → разные косинусы (синтетически) |
| U4 | Expected-empty | Нет chunk_embeddings → present=0, нет падения; chunk_embed_missing_flag=1 при файл не найден |
| U5 | Golden-детерминизм | Семантические поля: max\|Δ\|=0.00 (CPU numpy детерминирован); load_ms/compute_ms excluded |
| U6 | Разные длины видео | n_chunks=1,4,5,6 — все отрабатывают без падений (синтетически) |

---

## Критерии компонента

**C1: load_ms/compute_ms excluded from golden (timing by design)**
- `tp_embshift_load_ms` и `tp_embshift_compute_ms` — CPU-стенное время (мс), меняются между вызовами.
- При `emit_extra_metrics=False` (production default) оба = NaN (28/28 NPZ в датасете).
- Семантические поля (cosine, margin, flags) → golden max|Δ|=0.0.
- Паттерн аналогичен semantic_cluster_extractor, comments_aggregator, embedding_stats_extractor.

**C2: emit_extra_metrics=False → load_ms=NaN, compute_ms=NaN**
- Production default: `emit_extra_metrics=False`.
- Проверено: все 28 NPZ в storage имеют load_ms=NaN, compute_ms=NaN ✅.

**C3: compute_extra_cosines=False → cosine_first_last=NaN, mean_cos_last_to_start_window=NaN**
- Production default: `compute_extra_cosines=False`.
- Валидатор проверяет: при enabled=0 эти поля должны быть NaN.

**C4: margin = cosine_begin_end − threshold (|Δ|<1e-3)**
- Инвариант: `tp_embshift_margin = tp_embshift_cosine_begin_end − tp_embshift_cosine_threshold`.
- Проверяется валидатором (validate_ranges). Подтверждено: |Δ|=0.00e+00 (синтетика).

**C5: present=0 при n_chunks < require_min_chunks (by design)**
- Если загружено эмбеддингов меньше `require_min_chunks` (default=2), cosine не вычисляется.
- `tp_embshift_present=0`, `n_chunks` и `dim` заполнены корректно.
- Это не дефект — защита от бессмысленного "косинуса" из одного чанка.

**C6: Разреженность (1/22 present=1 в датасете)**
- `present=1`: компонент вычислил косинус (chunk_embeddings найдены и загружены).
- `present=0`: отсутствуют эмбеддинги от transcript_chunk_embedder
  (опциональный компонент; аудиотранскрипт + эмбеддинги доступны не всегда).
- Обработка missing: компонент выдаёт `chunk_embed_missing_flag=1`,
  валидатор пропускает структурные проверки (expected-empty по дизайну, не баг).
- Интерпретация для Models: valid modality dropout — модель в Fusion
  получит `present=0` для этой модальности и корректно обработает.
- 1/22 present=1 в тестовом датасете — норма (transcript_chunk_embedder не запускался
  для большинства видео из-за отсутствия ASR-данных).
