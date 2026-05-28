# `asr_text_proxy_audio_features` — описание фич и артефактов

**Компонент:** `asr_text_proxy_audio_features` (`ASRTextProxyExtractor`, [`../main.py`](../main.py))  
**Вклад в NPZ:** плоские скаляры с префиксом `tp_asrproxy_*` в агрегированном `text_processor/text_features.npz` (`meta.schema_version`: `text_npz_v1`). Отдельного NPZ экстрактора нет.  
**Контракт:** [`../../../../schemas/asr_text_proxy_audio_features_output_v1.json`](../../../../schemas/asr_text_proxy_audio_features_output_v1.json) · схема процесса: [`../SCHEMA.md`](../SCHEMA.md) · пользовательская дока: [`../README.md`](../README.md).

**Версия реализации:** 1.2.0 (`ASRTextProxyExtractor.VERSION`).

---

## 1. Назначение

Прокси «аудиоподобных» метрик по **тексту ASR** (уверенность сегментов, эвристики шума, ритм, интонация по пунктуации) **без** анализа волны. Это **не** WER и не акустические признаки.

---

## 2. Полный перечень полей (`features_flat` → NPZ)

Ровно **37** ключей; набор совпадает с `asr_text_proxy_audio_features_output_v1.json` (`allow_extra_keys: false`).

| Группа | Ключ | Смысл |
|--------|------|--------|
| Конфиг (аудит) | `tp_asrproxy_enabled` | Экстрактор не отключён конструктором (0/1) |
| | `tp_asrproxy_basic_enabled` | Включены метрики confidence (`enable_basic`) |
| | `tp_asrproxy_noise_enabled` | Включены noise proxies (`enable_noise`) |
| | `tp_asrproxy_rhythm_enabled` | Включены ритм (`enable_rhythm`) |
| | `tp_asrproxy_intonation_enabled` | Включена интонация (`enable_intonation`) |
| | `tp_asrproxy_require_asr_text_enabled` | Режим `require_asr_text` |
| | `tp_asrproxy_strict_document_duration_enabled` | Режим `strict_document_duration` |
| | `tp_asrproxy_low_conf_threshold` | Порог низкой confidence (по умолчанию 0.5) |
| | `tp_asrproxy_words_per_minute_baseline` | Baseline WPM для отношения (по умолчанию 160, в ctor > 0) |
| | `tp_asrproxy_max_text_chars` | Лимит символов joined-текста (по умолчанию 200000) |
| Наличие / размер | `tp_asrproxy_present` | Непустой joined-транскрипт (0/1) |
| | `tp_asrproxy_has_confidence` | Хотя бы у одного сегмента есть confidence (0/1) |
| | `tp_asrproxy_segments_count` | Число dict-сегментов в payload |
| | `tp_asrproxy_text_chars`, `tp_asrproxy_word_count` | Длина текста и число токенов (после truncate) |
| | `tp_asrproxy_confidence_present_rate` | Доля сегментов, у которых поле confidence не `None` ([0..1] или NaN при 0 сегментов) |
| Длительность | `tp_asrproxy_audio_duration_sec` | Итоговая длительность (сек), > 0 при успешном extract |
| | `tp_asrproxy_duration_from_payload_flag` | Длительность взята из ASR payload (1), т.к. не было `audio_duration_sec` на документе |
| | `tp_asrproxy_duration_invalid_flag` | Обнаружена невалидная длительность (≤ 0); до NPZ обычно не доходит (RuntimeError) |
| Флаги валидации | `tp_asrproxy_text_truncated_flag` | Текст обрезан по `max_text_chars` |
| | `tp_asrproxy_asr_schema_invalid_flag` | Сегмент не dict |
| | `tp_asrproxy_conf_invalid_flag` | Confidence вне [0,1] или не число |
| | `tp_asrproxy_token_decode_failed_flag` | Ошибка token-id decode path (`shared_tokenizer_v1`) |
| Confidence | `tp_asrproxy_confidence_mean`, `tp_asrproxy_confidence_std` | По валидным confidence; NaN если нет данных / basic выкл. |
| | `tp_asrproxy_confidence_chunked_min` | Минимум **средних** confidence по блокам (~10 чанков по списку confidence), не «min по сегментам» |
| | `tp_asrproxy_low_conf_rate` | Доля сегментов с `confidence < low_conf_threshold` |
| Noise | `tp_asrproxy_text_noise_rare_ratio` | Доля «редких» токенов (эвристика длины/символов) |
| | `tp_asrproxy_text_noise_oov_ratio` | Доля токенов с малым числом букв (OOV-proxy) |
| | `tp_asrproxy_noise_proxy` | `min(1, mean)` по доступным из rare_ratio и low_conf_rate |
| | `tp_asrproxy_noise_proxy_present` | Удалось задать агрегат (0/1) |
| Rhythm | `tp_asrproxy_speech_rate_wpm` | Слова в минуту: `word_count / (duration_sec/60)` |
| | `tp_asrproxy_speech_rate_wpm_ratio_to_baseline` | `speech_rate_wpm / words_per_minute_baseline` |
| | `tp_asrproxy_speech_char_density` | Символов в секунду |
| | `tp_asrproxy_pause_density` | `(, + ; + :) / n_sent`, `n_sent` = число `. ? !` |
| | `tp_asrproxy_filler_ratio` | Доля слов из короткого filler-лексикона |
| Intonation | `tp_asrproxy_sentence_intonation` | `(count ! + count ?) / n_sent` |

**Тайминги:** в `extract()` возвращается `timings_s.total` (сек), в агрегированный `text_features.npz` **не** попадает; ориентир вручную: ~0.05–0.3 с на типичный транскрипт ([`README.md`](../README.md)).

---

## 3. Ожидаемые диапазоны (для `--ranges` валидатора)

Корректный **успешный** прогон (`meta.status=ok`, срез `tp_asrproxy_*` присутствует):

| Поле / группа | Ожидание |
|---------------|----------|
| Бинарные `*_flag`, `present`, `has_confidence`, `*_enabled`, `noise_proxy_present` | ∈ {0, 1} |
| `tp_asrproxy_low_conf_threshold` | [0, 1] (типично 0.3–0.7) |
| `tp_asrproxy_words_per_minute_baseline` | > 0 (типично 120–220) |
| `tp_asrproxy_max_text_chars` | ≥ 0 (типично 200000) |
| `tp_asrproxy_audio_duration_sec` | > 0 (float) |
| `tp_asrproxy_segments_count`, `tp_asrproxy_text_chars`, `tp_asrproxy_word_count` | ≥ 0 |
| Доли [0..1] или NaN | `confidence_present_rate`, `confidence_mean`, `chunked_min`, `low_conf_rate`, `text_noise_*`, `noise_proxy`, `filler_ratio`, `sentence_intonation` |
| `tp_asrproxy_confidence_std` | ≥ 0 при finite |
| `tp_asrproxy_pause_density` | ≥ 0 при finite |
| `tp_asrproxy_speech_rate_wpm`, `tp_asrproxy_speech_char_density` | ≥ 0 при finite (`present=1`, rhythm включён) |

**Пустой срез:** при `meta.status=error` пайплайн мог прерваться до заполнения таблицы — **0** имён `tp_asrproxy_*` (см. audit v4 отчёт). Валидатор сообщает предупреждение, не считая это нарушением контракта NPZ целиком.

---

## 4. Код / отчёты

- QA L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)
- Валидатор среза: [`../utils/validate_asr_text_proxy_text_npz.py`](../utils/validate_asr_text_proxy_text_npz.py)
- HTML: `text_processor/_render/asr_text_proxy_audio_features_report.html` (из [`../render.py`](../render.py))

---

## 5. Чеклист сверки с `text_features.npz`

1. `meta.status == ok` → в `feature_names` есть **все 37** ключей из JSON-схемы, без лишних `tp_asrproxy_*`.
2. Длина `feature_values` совпадает с `feature_names`.
3. При `tp_asrproxy_present=1` и включённых подсистемах конечные метрики не должны быть NaN (кроме случаев без confidence и т.д. — см. код).
