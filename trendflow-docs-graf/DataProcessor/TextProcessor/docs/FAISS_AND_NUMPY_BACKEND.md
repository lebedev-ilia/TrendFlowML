# FAISS vs NumPy backend (TextProcessor)

Несколько экстракторов объявляют в конфиге «использовать FAISS», но при отсутствии пакета **`faiss`** или при политике **`require_faiss=false`** поиск идёт через **NumPy** (полный перебор / матричное умножение). Для сравнения прогонов на **наборе B** и для корреляций по табличному слою нужно различать **желаемую** настройку и **фактический** backend.

## Что смотреть в `features_flat`

| Идея | Типичные ключи (префиксы) | Заметка |
|------|---------------------------|---------|
| В конфиге хотели FAISS | `tp_semclust_use_faiss_enabled`, `tp_titleclent_use_faiss_enabled`, `tp_embpair_use_faiss_mode*`, флаги в `tp_topktitles_require_faiss_*` | **1** в конфиге ≠ фактический индекс |
| Фактически искали через FAISS | `tp_semclust_backend_faiss`, `tp_titleclent_backend_faiss`, `tp_topktitles_backend_faiss`, `tp_embpair_*` (комбинация режима и порога корпуса) | Сверяйте с **`tp_*_faiss_available`** / импортом |
| Доступность пакета | `tp_topktitles_faiss_available` и аналоги в meta | **0** → все «want faiss» ветки падают в NumPy (если не `require_faiss`) |

## Семантика скоров

- **`topk_similar_titles_extractor`**: при FAISS используется **HNSW + inner product** (`backend` в corpus meta — `faiss_hnsw_ip`); результаты **приближённые** и могут отличаться от **точной** сортировки по косинусу (**`numpy_cosine`**). На одном и том же корпусе нельзя без оговорки сравнивать ранги top‑K между HNSW и NumPy.
- **`embedding_pair_topk_extractor`**: режим **`use_faiss_mode`** (`auto` / `never` / `always`) и **`min_corpus_for_faiss`** определяют, когда поднимается FAISS; иначе — нормализованная матрица × запрос.
- **`semantic_cluster_extractor`**, **`title_embedding_cluster_entropy_extractor`**: при неудачном импорте `faiss` при **`use_faiss=true`** выполняется поиск ближайшего центроида в **NumPy** (`meta.backend` / `features_flat` отражают факт).

## Практика для L2 / набора B

1. Стратифицируйте или фильтруйте строки по **`tp_*_backend_faiss`** (и при необходимости по **`faiss_available`**).
2. Фиксируйте в `RUN_LOG` / конфиге: установлен ли **`faiss-cpu`** / **`faiss-gpu`** в окружении прогона.
3. Не интерпретируйте различия top‑K между машинами только как «дрейф модели», если backend мог переключиться.

## Связанные README

- `src/extractors/topk_similar_titles_extractor/README.md` — HNSW vs numpy, пороги корпуса.
- `src/extractors/embedding_pair_topk_extractor/README.md` — `use_faiss_mode`, `require_faiss`.
- `src/extractors/semantic_cluster_extractor/README.md`, `title_embedding_cluster_entropy_extractor/README.md` — зеркала конфига vs `backend_faiss`.
---

## Навигация

[TextProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
