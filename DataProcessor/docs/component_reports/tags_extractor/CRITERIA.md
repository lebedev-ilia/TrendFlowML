# Критерии приёмки: tags_extractor

**Компонент:** `tags_extractor` (TextProcessor)  
**Дата согласования:** 2026-07-17  
**Согласовано:** Второй агент от имени владельца (автоматический ответ)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Статус |
|------|----------|--------|
| U1 | `validate_schema` rc=0 на всех NPZ | ✅ PASS |
| U2 | Ось времени N/A — flat скаляры, нет seq/temporal оси | N/A |
| U3 | Различимость: синтетически (разные теги → разные hash01, max\|Δ\|>0) | ✅ PASS (синтетик) |
| U4 | Expected-empty: пустые title+description+hashtags → unique=0, NaN avg_len/max_len, top_i=NaN | ✅ PASS |
| U5 | Golden-детерминизм: SHA256 → max\|Δ\|=0.0 | ✅ PASS |
| U6 | Разные случаи: unicode, long title с truncation, json-only, many-tags, no-tags | ✅ PASS |

**Примечание к U3:** Все 22 ok-NPZ в storage содержат одинаковые 3 хэштега (тестовый Fetcher — аналог
comments_embedder). Различимость проверена СИНТЕТИЧЕСКИ: разные входные теги → разные hash01.
Это не дефект компонента.

---

## Критерии компонента (C1–C4)

### C1 — Структура схемы (числовой порог)
- `validate_schema` = OK для всех доступных NPZ (28/28 в storage)
- `validate_structure` + `validate_ranges` = 0 ошибок на всех NPZ со status=ok
- Ожидается: 28 базовых ключей + 3×K top-слотов (`tp_tags_top{i}_{present,hash01,len}` для i=1..K)

### C2 — Инвариант total_found (числовой)
- `tp_tags_hashtag_total_found_count` == `tp_tags_title_hashtag_found_count` + `tp_tags_description_hashtag_found_count`
- Инвариант строго соблюдается: JSON-merged теги НЕ входят в `total_found` (только inline из title+desc)
- Допуск: abs(total - title - desc) ≤ 0.01

### C3 — Top-K prefix-заполнение
- Слоты 1..min(K, unique_count): `present=1`, `hash01 ∈ [0,1)` (finite), `len ≥ 1`
- Слоты min(K,unique_count)+1..K: `present=0`, `hash01=NaN`, `len=NaN`
- Нет «дырок» — present=1 не может идти после present=0 (строгий префикс)

### C4 — NaN by design (явные исключения)
Следующие NaN-значения являются **ожидаемыми и корректными**:
- `tp_tags_hashtag_avg_len` = NaN при `unique_count=0` (нет тегов → нет средней длины)
- `tp_tags_hashtag_max_len` = NaN при `unique_count=0`
- `tp_tags_top{i}_hash01` = NaN при `tp_tags_top{i}_present=0` (пустой слот)
- `tp_tags_top{i}_len` = NaN при `tp_tags_top{i}_present=0`

Эти NaN не считаются дефектом и задокументированы в контракте схемы.

---

## Тайминги

Компонент CPU-only (строковый парсинг + SHA256), GPU не требуется.
Стоимость per-video: единицы мс (пренебрежима).
