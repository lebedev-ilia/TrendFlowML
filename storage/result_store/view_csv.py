#!/usr/bin/env python3
"""
Просмотр wide-CSV (batch_features_report и др.): HTML с поиском и прокруткой, открытие в браузере.

Удобнее по колонкам: сгенерировать отчёт с
  DataProcessor/tools/batch_runs_feature_report.py --output-by-component-dir ./batch_by_component
и открыть в браузере ./batch_by_component/index.html (узкие csv в batch_by_component/csv/).

  cd storage/result_store
  python3 view_csv.py
  python3 view_csv.py --csv batch_features_report.csv
  # Длинный формат: component | feature | пояснение | value (одно видео = 4 колонки; N видео = 3+N)
  # По умолчанию --melt берёт ВСЕ колонки CSV (значения из NPZ в meta_*, feature_*).
  # Узко только «тех. поля»:  --melt --melt-compact
  # В melt скрыты run_path, npz и дублирующиеся meta_* (см. --melt-show-repeating-meta)
  python3 view_csv.py --melt
  python3 view_csv.py --melt --melt-compact
  # Только «полезные» фичи по компонентам (см. view_csv_melt_interesting.json, defaults.merge_into_each + components.include):
  python3 view_csv.py --melt --melt-interesting
  # Подсветка значений вне «нормальных» диапазонов (view_csv_feature_qa.json + DataProcessor/qa):
  python3 view_csv.py --melt --melt-interesting --melt-qa
  # алиас: --pivot-by-component (то же, что --melt)
"""

from __future__ import annotations

import argparse
import sys
import csv
import html
import json
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from view_csv_melt_captions_ru import (
    load_description_overrides,
    melt_feature_caption_ru,
)

# Компактный режим: иначе сотни meta_* не помещаются на экран.
DEFAULT_FOCUS_COLS: Sequence[str] = (
    "platform_id",
    "video_id",
    "run_id",
    "component",
    "component_type",
    "manifest_status",
    "manifest_empty_reason",
    "duration_ms",
    "device_used",
    "npz_error",
)

# Служебные поля: не «фичи» — фичи идут строками во 2-й колонке.
MELT_ID_COLS: frozenset[str] = frozenset(
    {
        "platform_id",
        "video_id",
        "run_id",
        "component",
        "component_type",
    }
)

# В --melt не показывать по умолчанию: одно и то же от видео к видео, засоряют таблицу.
MELT_SUPPRESS_REPEATING: frozenset[str] = frozenset(
    {
        "run_path",
        "npz",
        "meta_video_id",
        "meta_schema_version",
        "meta_sampling_policy_version",
        "meta_run_id",
        "meta_producer_version",
        "meta_producer",
        "meta_platform_id",
    }
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _pick_columns(
    header: List[str], focus: bool, extra_cols: List[str]
) -> List[str]:
    if not focus:
        return header
    want = [c for c in DEFAULT_FOCUS_COLS if c in header]
    for c in extra_cols:
        c = c.strip()
        if c and c in header and c not in want:
            want.append(c)
    if not want:
        return header[:20]
    return want


def _read_rows(
    path: Path,
    columns: List[str],
    max_rows: int,
) -> tuple[List[str], list[list[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return [], []
        use = [c for c in columns if c in r.fieldnames]
        rows_out: list[list[str]] = []
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            rows_out.append([row.get(c, "") or "" for c in use])
        return use, rows_out


def _read_dict_rows(
    path: Path,
    columns: List[str],
    max_rows: int,
) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return []
        use = [c for c in columns if c in r.fieldnames]
        out: List[Dict[str, str]] = []
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            d: Dict[str, str] = {}
            for c in use:
                v = row.get(c, "")
                d[c] = "" if v is None else str(v)
            out.append(d)
        return out


def _build_html(
    title: str,
    csv_path: str,
    headers: List[str],
    data: list[list[str]],
    focus_mode: bool,
) -> str:
    n = len(data)
    esc_headers = [html.escape(h) for h in headers]
    body_rows: List[str] = []
    for row in data:
        tds = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        body_rows.append(f"<tr>{tds}</tr>")
    mode_note = (
        "<p class=\"note\">Показаны не все колонки (компактный режим). Запустите с <code>--all-cols</code> для полной ширины.</p>"
        if focus_mode
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --fg: #e7e9ea;
      --muted: #8b98a5;
      --border: #38444d;
      --accent: #1d9bf0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
      font-size: 12px;
      background: var(--bg);
      color: var(--fg);
      margin: 0;
      padding: 16px 20px 40px;
    }}
    h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 8px; }}
    .meta {{ color: var(--muted); font-size: 11px; margin-bottom: 12px; }}
    .note {{ color: #ffad3d; font-size: 12px; margin: 8px 0 12px; }}
    .bar {{
      position: sticky; top: 0; z-index: 20;
      background: var(--bg);
      padding: 8px 0 12px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 8px;
    }}
    #q {{
      width: 100%; max-width: 560px;
      padding: 8px 12px; border-radius: 6px;
      border: 1px solid var(--border);
      background: #161b22; color: var(--fg);
      font: inherit;
    }}
    #q:focus {{ outline: 2px solid var(--accent); border-color: transparent; }}
    .wrap {{
      border: 1px solid var(--border);
      border-radius: 8px;
      max-height: calc(100vh - 140px);
      overflow: auto;
    }}
    table {{ border-collapse: collapse; min-width: 100%; }}
    th {{
      position: sticky; top: 0; z-index: 10;
      background: #1a2332;
      text-align: left; padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      font-weight: 600; white-space: nowrap;
      box-shadow: 0 1px 0 var(--border);
    }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #2f3943; vertical-align: top; max-width: 42rem; word-break: break-all; }}
    tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
    tr.hidden {{ display: none; }}
    .count {{ color: var(--muted); margin-left: 8px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Файл: <code>{html.escape(csv_path)}</code> · строк: <strong>{n}</strong> · колонок: <strong>{len(headers)}</strong></div>
  {mode_note}
  <div class="bar">
    <label>Фильтр по подстроке (строка целиком):</label>
    <input type="search" id="q" placeholder="например core_clip, empty, -Q6fn…" autocomplete="off" />
    <span class="count" id="cnt"></span>
  </div>
  <div class="wrap">
    <table>
      <thead><tr>{"".join(f"<th>{h}</th>" for h in esc_headers)}</tr></thead>
      <tbody id="tb">
        __INJECT_TBODY__
      </tbody>
    </table>
  </div>
  <script>
    const rows = () => Array.from(document.querySelectorAll("#tb tr"));
    const q = document.getElementById("q");
    const cnt = document.getElementById("cnt");
    function apply() {{
      const s = (q.value || "").trim().toLowerCase();
      let vis = 0;
      for (const tr of rows()) {{
        const t = tr.innerText.toLowerCase();
        const ok = !s || t.includes(s);
        tr.classList.toggle("hidden", !ok);
        if (ok) vis++;
      }}
      cnt.textContent = s ? "видимо: " + vis + " / " + rows().length : "";
    }}
    q.addEventListener("input", apply);
    apply();
  </script>
</body>
</html>
""".replace(
        "__INJECT_TBODY__", "\n".join(body_rows)
    )


def _melt_feature_columns(
    picked: List[str], *, show_repeating_meta: bool = False
) -> List[str]:
    """Порядок фич — как в выбранных колонках (без id-полей; без дублей run/npz/meta…)."""
    out = [c for c in picked if c not in MELT_ID_COLS]
    if not show_repeating_meta:
        out = [c for c in out if c not in MELT_SUPPRESS_REPEATING]
    return out


def _melt_video_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        str(row.get("video_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("platform_id") or ""),
    )


def _melt_build_index(
    rows: List[Dict[str, str]],
) -> tuple[
    Dict[Tuple[Tuple[str, str, str], str], Dict[str, str]],
    List[Tuple[str, str, str]],
    List[str],
]:
    """
    (video_key, component) -> строка CSV; порядок уникальных видео; компоненты.
    video_key = (video_id, run_id, platform_id).
    """
    v_order: List[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    index: Dict[Tuple[Tuple[str, str, str], str], Dict[str, str]] = {}
    for row in rows:
        vk = _melt_video_key(row)
        if vk not in seen:
            seen.add(vk)
            v_order.append(vk)
        comp = (row.get("component") or "").strip() or "—"
        index[(vk, comp)] = row
    components = sorted({(r.get("component") or "").strip() or "—" for r in rows})
    return index, v_order, components


def _melt_value_empty(s: str) -> bool:
    """Пусто для подавления строки: пусто, только пробелы, типичные «нет»."""
    t = (s or "").strip()
    if not t:
        return True
    low = t.lower()
    if low in (
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "-",
        "—",
    ):
        return True
    return False


def _melt_row_is_empty(
    comp: str,
    feat: str,
    v_cols: List[Tuple[str, str, str]],
    index: Dict[Tuple[Tuple[str, str, str], str], Dict[str, str]],
) -> bool:
    for vk in v_cols:
        r = index.get((vk, comp), {})
        v = (r or {}).get(feat, "")
        if not _melt_value_empty(str(v)):
            return False
    return True


def _melt_column_is_milliseconds(feat: str) -> bool:
    """
    Поля, где значение в мс. По суффиксу _ms (в т.ч. meta_timing_*_ms), не весь meta_timing_ —
    иначе счётчики вроде meta_timing_segments_count ошибочно станут «секундами».
    """
    if not feat:
        return False
    if feat == "duration_ms":
        return True
    return feat.endswith("_ms")


def _melt_feature_display_name(feat: str) -> str:
    """Убрать префикс meta_ в колонке «feature» (логика по полному имени колонки в CSV)."""
    if feat.startswith("meta_"):
        return feat[5:]
    return feat


def _melt_parse_float_string(s: str) -> Optional[float]:
    t = (s or "").strip()
    if not t:
        return None
    t = t.replace(" ", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _melt_ms_to_seconds_display_ru(ms_value: float) -> str:
    """
    Значение в мс → строка «с,с с» (запятая как разделитель).
    Для |сек| ≥ 0,01 — 2 знака после запятой; для очень малых — 4, 6 или 8.
    """
    sec = ms_value / 1000.0
    a = abs(sec)
    if a == 0.0:
        s = "0,00"
    elif a >= 0.01:
        s = f"{sec:.2f}".replace(".", ",")
    elif a >= 0.0001:
        s = f"{sec:.4f}".replace(".", ",")
    elif a >= 1e-6:
        s = f"{sec:.6f}".replace(".", ",")
    else:
        s = f"{sec:.8f}".replace(".", ",")
    return f"{s} с"


def _melt_format_value_cell_html(raw: str, feat: str) -> str:
    """
    Содержимое ячейки значения: для мс — сек., «с»; в title подсказка с исходом в мс.
    """
    t = str(raw)
    if not _melt_column_is_milliseconds(feat):
        return html.escape(t)
    v = _melt_parse_float_string(t)
    if v is None or v != v:  # nan
        return html.escape(t)
    disp = _melt_ms_to_seconds_display_ru(v)
    tip = f"{t.strip()} (мс, исх.)"
    return (
        f'<span class="melt-ms" title="{html.escape(tip)}">'
        f"{html.escape(disp)}</span>"
    )


def _melt_outer_max_width_px(n_video: int, with_desc: bool) -> int:
    """
    Ширина блока с таблицей: при 1–2 видео уже (по центру экрана), с ростом числа видео — шире, с потолком.
    """
    n = max(1, n_video)
    base = 400 + (300 if with_desc else 0)
    per = 120 if n <= 2 else (140 if n <= 4 else 160)
    return min(1780, base + n * per)


def _melt_thead_vcols(
    v_order: List[Tuple[str, str, str]], *, with_description: bool = False, with_range: bool = False
) -> str:
    """component | feature [| пояснение] [| норма] | value(и)."""
    h = [
        '<th class="th-comp">component</th>',
        '<th class="th-feat">feature</th>',
    ]
    if with_description:
        h.append('<th class="th-desc">пояснение</th>')
    if with_range:
        h.append('<th class="th-range">норма</th>')
    if not v_order:
        h.append('<th class="th-val">value</th>')
        return "<tr>" + "".join(h) + "</tr>"
    if len(v_order) == 1:
        vid, rid, _ = v_order[0]
        tip = f"video_id={vid}\nrun_id={rid}" if (vid or rid) else "value"
        h.append(f'<th class="th-val" title="{html.escape(tip)}">value</th>')
        return "<tr>" + "".join(h) + "</tr>"
    for j, vk in enumerate(v_order):
        vid, rid, _ = vk
        full = f"video_id={vid}\nrun_id={rid}\n# {j + 1}"
        if vid:
            short = vid if len(vid) <= 24 else (vid[:22] + "…")
        elif rid:
            short = rid if len(rid) <= 24 else (rid[:22] + "…")
        else:
            short = f"video {j + 1}"
        h.append(
            f'<th class="th-val" title="{html.escape(full)}">{html.escape(short)}</th>',
        )
    return "<tr>" + "".join(h) + "</tr>"


def _melt_interesting_config_path() -> Path:
    return _script_dir() / "view_csv_melt_interesting.json"


def _melt_qa_config_path() -> Path:
    return _script_dir() / "view_csv_feature_qa.json"


def _dataprocessor_dir_for_qa() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "DataProcessor"


def _load_qa_config_obj(path: Path) -> Any:
    dp = _dataprocessor_dir_for_qa()
    if str(dp) not in sys.path:
        sys.path.insert(0, str(dp))
    from qa.component_feature_qa import load_qa_config  # noqa: E402

    return load_qa_config(path)


def _melt_feature_desc_config_path() -> Path:
    return _script_dir() / "view_csv_feature_descriptions_ru.json"


def _load_melt_interesting_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _melt_interesting_columns_for_component(
    component: str, all_feas: List[str], cfg: dict[str, Any]
) -> List[str]:
    """
    defaults.merge_into_each (если задан) + include блока, без дублей;
    при add_all_meta_timing — все meta_timing_* из all_feas;
    при add_all_meta_asr_timing — все meta_asr_timing_* (стадии ASR из asr_stage_timings_ms).
    Нет блока в конфиге → весь all_feas.
    """
    comp_map = cfg.get("components")
    if isinstance(comp_map, dict) and component in comp_map and comp_map[component] is not None:
        block: dict[str, Any] = comp_map[component]  # type: ignore[assignment]
    else:
        block = cfg.get("fallback_unlisted")
        if not isinstance(block, dict):
            block = {}
    if not block:
        return list(all_feas)
    raw_include = list(block.get("include") or [])
    add_timing = bool(block.get("add_all_meta_timing", False))
    add_asr_timing = bool(block.get("add_all_meta_asr_timing", False))
    defaults = cfg.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    merge = defaults.get("merge_into_each")
    if not isinstance(merge, list):
        merge = []
    prefix = str(defaults.get("meta_timing_prefix", "meta_timing_"))
    asr_prefix = str(defaults.get("meta_asr_timing_prefix", "meta_asr_timing_"))
    out: List[str] = []
    seen: set[str] = set()
    for c in list(merge) + raw_include:
        if c in all_feas and c not in seen:
            out.append(c)
            seen.add(c)
    if add_timing:
        for c in sorted(all_feas):
            if c.startswith(prefix) and c not in seen:
                out.append(c)
                seen.add(c)
    if add_asr_timing:
        for c in sorted(all_feas):
            if c.startswith(asr_prefix) and c not in seen:
                out.append(c)
                seen.add(c)
    return out


def _melt_rule_range_ru(qa_config: Any, component: str, feature: str) -> str:
    """Короткий текст диапазона/перечня из QA-правил для колонки."""
    if qa_config is None:
        return "—"
    try:
        if hasattr(qa_config, "rule_for"):
            rule = qa_config.rule_for(component, feature)
        else:
            rmap = qa_config.rules_for_column(component)
            rule = rmap.get(feature) if isinstance(rmap, dict) else None
    except Exception:
        return "—"
    if not isinstance(rule, dict):
        return "—"
    optional = bool(rule.get("optional"))
    suffix = " (optional)" if optional else ""
    if "enum" in rule and isinstance(rule.get("enum"), (list, tuple)):
        vals = [str(x) for x in rule.get("enum", [])]
        if not vals:
            return "—"
        return "enum: " + ", ".join(vals[:8]) + (" ..." if len(vals) > 8 else "") + suffix
    has_min = "min" in rule and rule.get("min") is not None
    has_max = "max" in rule and rule.get("max") is not None
    if has_min and has_max:
        return f"[{rule.get('min')}, {rule.get('max')}]" + suffix
    if has_min:
        return f">= {rule.get('min')}" + suffix
    if has_max:
        return f"<= {rule.get('max')}" + suffix
    return "—"


def _build_html_melt(
    title: str,
    csv_path: str,
    feature_cols: List[str],
    rows: List[Dict[str, str]],
    focus_mode: bool,
    *,
    repeating_meta_suppressed: bool = True,
    interesting_cfg: Optional[dict[str, Any]] = None,
    interesting_config_path: str = "",
    with_feature_descriptions: bool = True,
    feature_desc_overrides: Optional[dict[str, str]] = None,
    feature_desc_config_path: str = "",
    meta_timing_prefix: str = "meta_timing_",
    qa_config: Any = None,
    qa_config_path: str = "",
) -> str:
    """
    1) component с rowspan = число фич; 2) feature (имя колонки);
    3) при with_feature_descriptions — краткое пояснение на русском;
    4..) по колонке на видео.
    """
    index, v_order, components = _melt_build_index(rows)
    v_cols = v_order if v_order else [("", "", "")]
    n_video = max(1, len(v_order)) if v_order else 1
    show_range_col = qa_config is not None
    base_cols = 2 + (1 if with_feature_descriptions else 0) + (1 if show_range_col else 0)
    ncols = base_cols + n_video
    if not v_order:
        n_video = 1
        ncols = base_cols + 1
    _over = feature_desc_overrides or {}

    body_rows: List[str] = []
    bidx = 0
    for comp in components:
        if interesting_cfg is not None:
            candidates = _melt_interesting_columns_for_component(
                comp, feature_cols, interesting_cfg
            )
        else:
            candidates = list(feature_cols)
        shown_feats = [
            f
            for f in candidates
            if not _melt_row_is_empty(comp, f, v_cols, index)
        ]
        if not shown_feats:
            continue
        Fc = len(shown_feats)
        for fi, feat in enumerate(shown_feats):
            tds: List[str] = []
            if fi == 0:
                tds.append(
                    f'<td class="comp" rowspan="{Fc}">{html.escape(comp)}</td>'
                )
            t_disp = _melt_feature_display_name(feat)
            tds.append(
                f'<td class="feat" title="{html.escape(feat)}">'
                f"{html.escape(t_disp)}</td>"
            )
            if with_feature_descriptions:
                cap = melt_feature_caption_ru(
                    feat,
                    _over,
                    timing_prefix=meta_timing_prefix,
                )
                tds.append(
                    f'<td class="feat-desc">{html.escape(cap)}</td>'
                )
            if show_range_col:
                rr = _melt_rule_range_ru(qa_config, comp, feat)
                tds.append(f'<td class="feat-range">{html.escape(rr)}</td>')
            for vk in v_cols:
                r = index.get((vk, comp), {})
                v = (r or {}).get(feat, "")
                inner = _melt_format_value_cell_html(str(v), feat)
                q_cls = ""
                q_title = ""
                if qa_config is not None:
                    warn = qa_config.warning_for(comp, feat, str(v))
                    if warn:
                        q_cls = " val--qa-warn"
                        q_title = f' title="{html.escape(warn)}"'
                tds.append(f'<td class="val{q_cls}"{q_title}>{inner}</td>')
            body_rows.append(
                f'<tr data-b="{bidx}">' + "".join(tds) + "</tr>"
            )
        bidx += 1
    n_body = len(body_rows)
    n_comp_shown = bidx
    n_vid = len(v_order) if v_order else 1
    melt_max_px = _melt_outer_max_width_px(n_vid, with_feature_descriptions)
    thead = _melt_thead_vcols(
        v_order, with_description=with_feature_descriptions, with_range=show_range_col
    )
    mode_parts: List[str] = []
    if focus_mode:
        mode_parts.append(
            '<p class="note">Сейчас только key-колонки. Чтобы в строках фич шли <strong>значения из NPZ</strong> (meta_*), уберите <code>--melt-compact</code> (по умолчанию melt уже берёт весь wide CSV) или укажите <code>--all-cols</code>.</p>'
        )
    else:
        mode_parts.append(
            '<p class="note">Список фич строится по <strong>всем</strong> колонкам отчёта (сотни meta_* = выжимка из NPZ). Кратко: <code>--melt --melt-compact</code>.</p>'
        )
    if repeating_meta_suppressed:
        mode_parts.append(
            '<p class="note">Из таблицы убраны дублирующиеся run_path, npz и meta_* (id схемы, продьюсер, platform…). Показать: <code>--melt-show-repeating-meta</code>.</p>'
        )
    if interesting_cfg is not None and interesting_config_path:
        mode_parts.append(
            f'<p class="note">Режим <strong>только полезных полей</strong>: {html.escape(interesting_config_path)} — курируйте <code>components</code> и <code>fallback_unlisted</code>.</p>'
        )
    if with_feature_descriptions:
        cap_p = feature_desc_config_path or str(_melt_feature_desc_config_path().resolve())
        mode_parts.append(
            f'<p class="note">Колонка «пояснение» — кратко по-русски: словарь токенов + переопределения в '
            f'<code>{html.escape(cap_p)}</code>. Скрыть: <code>--melt-no-descriptions</code>.</p>'
        )
    if show_range_col and qa_config_path:
        mode_parts.append(
            f'<p class="note">Режим <strong>QA-диапазонов</strong>: значения вне ожидаемых интервалов — '
            f"подсветка ячейки; подробности в <code>title</code>. Конфиг: <code>{html.escape(qa_config_path)}</code> "
            f"(см. <code>DataProcessor/qa/component_feature_qa.py</code>).</p>"
        )
    mode_parts.append(
        "<p class=\"note\">Суффикс <code>_ms</code> и колонка <code>duration_ms</code> (значения в мс) "
        "показаны в <strong>секундах</strong> с запятой; 2 знака, если |t| ≥ 0,01 с, иначе 4/6/8. "
        "Колонки <code>meta_timing_…</code> без <code>_ms</code> (счётчики и т.д.) — без пересчёта. "
        "Исход в мс — в подсказке к ячейке.</p>"
    )
    mode_note = "\n".join(mode_parts)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0a0e14;
      --bg-elev: #111820;
      --fg: #e8edf4;
      --muted: #8a9aac;
      --border: rgba(120, 144, 170, 0.18);
      --accent: #5eb0e8;
      --accent-soft: rgba(94, 176, 232, 0.12);
      --sticky-bg: #141c28;
      --sticky-comp: #0e1520;
      --desc-fg: #b4c2d4;
      --radius: 12px;
      --font-ui: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
      --font-mono: "IBM Plex Mono", ui-monospace, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: var(--font-ui);
      font-size: 13px;
      line-height: 1.45;
      background: linear-gradient(165deg, #070a0f 0%, #0c1219 45%, #080c12 100%);
      color: var(--fg);
      margin: 0;
      padding: 24px 20px 48px;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}
    .melt-outer {{
      max-width: {melt_max_px}px;
      width: 100%;
      margin-left: auto;
      margin-right: auto;
    }}
    h1 {{
      font-family: var(--font-ui);
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0 0 12px;
      color: #f2f6fb;
      text-shadow: 0 1px 20px rgba(94, 176, 232, 0.15);
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
      line-height: 1.5;
    }}
    .meta code {{ font-family: var(--font-mono); font-size: 11px; opacity: 0.95; }}
    .note {{
      color: #d4a84b;
      font-size: 12px;
      margin: 10px 0 12px;
      padding: 10px 14px;
      background: rgba(212, 168, 75, 0.06);
      border-left: 3px solid rgba(212, 168, 75, 0.45);
      border-radius: 0 var(--radius) var(--radius) 0;
      line-height: 1.5;
    }}
    .note code {{ font-family: var(--font-mono); font-size: 11px; color: #e8c97a; }}
    .bar {{
      position: sticky; top: 0; z-index: 20;
      background: rgba(7, 10, 15, 0.88);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      padding: 12px 0 14px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 14px;
    }}
    .bar label {{ display: block; font-size: 11px; color: var(--muted); margin-bottom: 6px; font-weight: 500; }}
    #q {{
      width: 100%; max-width: 520px;
      padding: 10px 14px; border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-elev); color: var(--fg);
      font: 500 13px var(--font-ui);
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.2);
    }}
    #q:focus {{ outline: none; border-color: rgba(94, 176, 232, 0.45); box-shadow: 0 0 0 3px var(--accent-soft); }}
    .wrap {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      max-height: calc(100vh - 170px);
      overflow: auto;
      background: var(--bg-elev);
      box-shadow: 0 8px 40px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255,255,255,0.03) inset;
    }}
    table.melt-tbl {{ border-collapse: collapse; width: 100%; min-width: 0; }}
    table.melt-tbl[data-nv="1"] td.val {{ max-width: 11rem; }}
    table.melt-tbl[data-nv="2"] td.val {{ max-width: 12rem; }}
    table.melt-tbl[data-nv="3"] td.val {{ max-width: 15rem; }}
    table.melt-tbl[data-nv="4"] td.val, table.melt-tbl[data-nv="5"] td.val {{ max-width: 18rem; }}
    table.melt-tbl[data-nv="6"] td.val, table.melt-tbl[data-nv="7"] td.val,
    table.melt-tbl[data-nv="8"] td.val {{ max-width: 20rem; }}
    table.melt-tbl td.val {{ max-width: 22rem; }}
    table.melt-tbl th {{
      position: sticky; top: 0; z-index: 12;
      background: linear-gradient(180deg, #182230 0%, #141c28 100%);
      text-align: left; padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      box-shadow: 0 1px 0 rgba(0,0,0,0.25);
    }}
    table.melt-tbl th.th-comp {{
      z-index: 14;
      box-shadow: 1px 1px 0 var(--border);
      font-family: var(--font-ui);
      font-weight: 600;
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #8fa0b8;
    }}
    table.melt-tbl th.th-feat {{
      font-family: var(--font-mono);
      font-weight: 500;
      font-size: 11px;
      letter-spacing: 0.03em;
      text-transform: none;
      color: #9eb6d4;
      min-width: 11rem;
    }}
    table.melt-tbl th.th-val {{
      font-family: var(--font-mono);
      font-weight: 500;
      font-size: 11px;
      text-transform: none;
      letter-spacing: 0.02em;
      color: #a8c4e6;
    }}
    table.melt-tbl th.th-desc, table.melt-tbl td.feat-desc {{
      min-width: 17rem;
      max-width: 32rem;
      width: 28%;
      color: var(--desc-fg);
      font-family: var(--font-ui);
      font-weight: 400;
      font-size: 13px;
      line-height: 1.48;
      letter-spacing: 0.01em;
      text-transform: none;
    }}
    table.melt-tbl th.th-desc {{ color: #8fa6c0; font-size: 11px; font-weight: 600; letter-spacing: 0.04em; }}
    table.melt-tbl th.th-range, table.melt-tbl td.feat-range {{
      min-width: 11rem;
      max-width: 16rem;
      width: 14%;
      font-family: var(--font-mono);
      font-size: 11px;
      color: #c1d1e5;
      letter-spacing: 0.01em;
    }}
    table.melt-tbl th.th-range {{ color: #91a9c6; font-weight: 600; font-size: 11px; letter-spacing: 0.04em; }}
    td.feat-desc {{ border-left: 1px solid rgba(80, 100, 130, 0.2); }}
    td.feat-range {{ border-left: 1px solid rgba(80, 100, 130, 0.2); }}
    td.val .melt-ms {{ cursor: help; border-bottom: 1px dotted rgba(138, 154, 172, 0.6); }}
    td.val.val--qa-warn {{ cursor: help; background: rgba(212, 168, 75, 0.1); box-shadow: inset 0 0 0 1px rgba(212, 168, 75, 0.45); border-radius: 4px; }}
    table.melt-tbl td {{
      font-family: var(--font-mono);
      font-size: 12px;
      padding: 8px 12px;
      border-bottom: 1px solid rgba(60, 76, 96, 0.25);
      vertical-align: top;
      word-break: break-word;
    }}
    td.comp {{
      vertical-align: middle;
      position: sticky;
      left: 0;
      z-index: 11;
      background: linear-gradient(90deg, #0e1520 0%, #101a28 100%);
      border-right: 1px solid var(--border);
      font-family: var(--font-ui);
      font-weight: 600;
      font-size: 12px;
      letter-spacing: 0.02em;
      color: #c5d6ec;
      white-space: nowrap;
      min-width: 16rem;
      max-width: 22rem;
      box-shadow: 4px 0 12px rgba(0,0,0,0.15);
    }}
    td.feat {{
      font-family: var(--font-mono);
      font-size: 12px;
      color: #a8c8ec;
      max-width: 22rem;
    }}
    tr:nth-child(even) td:not(.comp) {{ background: rgba(255,255,255,0.015); }}
    tr:hover td:not(.comp) {{ background: rgba(94, 176, 232, 0.04); }}
    tr.hidden {{ display: none; }}
    .count {{ color: var(--muted); margin-left: 10px; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="melt-outer">
  <h1>{html.escape(title)}</h1>
  <div class="meta">Файл: <code>{html.escape(csv_path)}</code> · компонентов (с непустыми фичами): <strong>{n_comp_shown}</strong> · видео (run): <strong>{len(v_order) or 1}</strong> · строк таблицы: <strong>{n_body}</strong> (пустые «фичи» по всем видео скрыты) · колонок: <strong>{ncols}</strong></div>
  {mode_note}
  <div class="bar">
    <label>Фильтр (весь блок component, если совпала строка):</label>
    <input type="search" id="q" placeholder="имя фичи, meta_, component…" autocomplete="off" />
    <span class="count" id="cnt"></span>
  </div>
  <div class="wrap">
    <table class="melt-tbl" data-nv="{n_vid}">
      <thead>{thead}</thead>
      <tbody id="tb">
        __INJECT_TBODY__
      </tbody>
    </table>
  </div>
  </div>
  <script>
    const rows = () => Array.from(document.querySelectorAll("#tb tr"));
    const q = document.getElementById("q");
    const cnt = document.getElementById("cnt");
    function apply() {{
      const s = (q.value || "").trim().toLowerCase();
      const byB = new Map();
      for (const tr of rows()) {{
        const b = tr.getAttribute("data-b");
        if (!byB.has(b)) byB.set(b, []);
        byB.get(b).push(tr);
      }}
      let vis = 0;
      for (const trs of byB.values()) {{
        const ok = !s || trs.some(tr => tr.innerText.toLowerCase().includes(s));
        for (const tr of trs) {{
          tr.classList.toggle("hidden", !ok);
        }}
        if (ok) vis += trs.length;
      }}
      cnt.textContent = s ? "видимо строк: " + vis + " / " + rows().length : "";
    }}
    q.addEventListener("input", apply);
    apply();
  </script>
</body>
</html>
""".replace(
        "__INJECT_TBODY__", "\n".join(body_rows)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="HTML-просмотр wide CSV (по умолчанию key-колонки).")
    ap.add_argument(
        "--csv",
        default="",
        help="Путь к CSV (по умолчанию: batch_features_report.csv рядом со скриптом)",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Куда записать HTML (по умолчанию: <имя.csv>.view.html рядом с CSV)",
    )
    ap.add_argument(
        "--all-cols",
        action="store_true",
        help="Включить все колонки (очень широко, осторожно в браузере).",
    )
    ap.add_argument(
        "--add-col",
        action="append",
        default=[],
        metavar="NAME",
        help="Добавить колонку в компактный режим (можно повторить).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        metavar="N",
        help="Максимум строк (0 = без ограничения).",
    )
    ap.add_argument(
        "--melt",
        action="store_true",
        help="Длинно: component | feature | value(и); фичи — по строкам, не по ширине.",
    )
    ap.add_argument(
        "--pivot-by-component",
        action="store_true",
        help="Синоним --melt (старый флаг).",
    )
    ap.add_argument(
        "--melt-compact",
        action="store_true",
        help="В melt взять только key-колонки (как в обычном просмотре). По умолчанию melt: все колонки CSV, чтобы видеть выходы NPZ (meta_*).",
    )
    ap.add_argument(
        "--melt-show-repeating-meta",
        action="store_true",
        help="В melt снова показать run_path, npz и дублирующиеся meta_* (по умолчанию в melt скрыты).",
    )
    ap.add_argument(
        "--melt-interesting",
        action="store_true",
        help="Сузить фичи по JSON (view_csv_melt_interesting.json): include + meta_timing_* на компонент; без секции — fallback_unlisted.",
    )
    ap.add_argument(
        "--melt-interesting-config",
        default="",
        metavar="PATH",
        help="Путь к JSON (по умолчанию view_csv_melt_interesting.json рядом с view_csv.py).",
    )
    ap.add_argument(
        "--melt-no-descriptions",
        action="store_true",
        help="В режиме melt не добавлять колонку с русскими пояснениями к фичам.",
    )
    ap.add_argument(
        "--melt-descriptions-config",
        default="",
        metavar="PATH",
        help="JSON с переопределениями пояснений: ключ = имя колонки, value = текст; "
        "также можно {\"descriptions\":{…}}. По умолчанию view_csv_feature_descriptions_ru.json "
        "рядом с view_csv.py (файл может отсутствовать — тогда только эвристика).",
    )
    ap.add_argument(
        "--melt-qa",
        action="store_true",
        help="Подсветка значений вне диапазонов из view_csv_feature_qa.json (melt-режим).",
    )
    ap.add_argument(
        "--melt-qa-config",
        default="",
        metavar="PATH",
        help="JSON с правилами QA (по умолчанию view_csv_feature_qa.json рядом с view_csv.py).",
    )
    ap.add_argument(
        "--no-open",
        action="store_true",
        help="Не открывать браузер, только записать HTML.",
    )
    args = ap.parse_args()

    base = _script_dir()
    csv_path = Path(args.csv).expanduser() if args.csv else (base / "batch_features_report.csv")
    if not csv_path.is_file():
        print(f"Нет файла: {csv_path}", flush=True)
        return 1

    use_melt = (
        bool(args.melt) or bool(args.pivot_by_component) or bool(args.melt_interesting)
    )
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Пустой CSV", flush=True)
            return 1

    if use_melt and "component" not in header:
        print("В CSV нет колонки component — режим melt невозможен.", flush=True)
        return 1

    if use_melt:
        # Melt: по умолчанию все колонки (реальные «выходы» в wide CSV — в meta_*, feature_*).
        # Узко: --melt-compact; --all-cols в паре с compact отменяет compact (показать всё).
        focus_for_melt = bool(args.melt_compact) and not bool(args.all_cols)
        focus_picked = focus_for_melt
    else:
        focus_picked = not bool(args.all_cols)
    cols = _pick_columns(
        header, focus=focus_picked, extra_cols=list(args.add_col or [])
    )
    if use_melt and "component" in header:
        cols = ["component"] + [c for c in cols if c != "component"]

    interesting_cfg: Optional[dict[str, Any]] = None
    interesting_path_resolved = ""
    if use_melt and bool(args.melt_interesting):
        ip = (
            Path(args.melt_interesting_config).expanduser()
            if args.melt_interesting_config
            else _melt_interesting_config_path()
        )
        if not ip.is_file():
            print(
                f"Нет файла интереса: {ip} (положите view_csv_melt_interesting.json или --melt-interesting-config ПУТЬ).",
                flush=True,
            )
            return 1
        interesting_cfg = _load_melt_interesting_config(ip)
        interesting_path_resolved = str(ip.resolve())

    desc_overrides: dict[str, str] = {}
    desc_resolved = ""
    if use_melt and not bool(args.melt_no_descriptions):
        p_load = (
            Path(args.melt_descriptions_config).expanduser()
            if args.melt_descriptions_config
            else _melt_feature_desc_config_path()
        )
        if p_load.is_file():
            desc_overrides = load_description_overrides(p_load)
            desc_resolved = str(p_load.resolve())
    timing_prefix = "meta_timing_"
    if interesting_cfg is not None:
        d0 = interesting_cfg.get("defaults")
        if isinstance(d0, dict) and d0.get("meta_timing_prefix"):
            timing_prefix = str(d0["meta_timing_prefix"])

    if bool(args.melt_qa) and not use_melt:
        print("Укажите --melt / --melt-interesting / --pivot-by-component вместе с --melt-qa.", flush=True)
        return 1

    qa_obj: Any = None
    qa_path_resolved = ""
    if use_melt and bool(args.melt_qa):
        qcp = (
            Path(args.melt_qa_config).expanduser()
            if args.melt_qa_config
            else _melt_qa_config_path()
        )
        if not qcp.is_file():
            print(f"Нет QA-конфига: {qcp}", flush=True)
            return 1
        qa_obj = _load_qa_config_obj(qcp)
        qa_path_resolved = str(qcp.resolve())

    if args.out:
        out_path = Path(args.out).expanduser()
    elif use_melt and bool(args.melt_interesting) and bool(args.melt_qa):
        out_path = csv_path.with_name(csv_path.stem + ".melt.interesting.qa.view.html")
    elif use_melt and bool(args.melt_interesting):
        out_path = csv_path.with_name(csv_path.stem + ".melt.interesting.view.html")
    elif use_melt and bool(args.melt_qa):
        out_path = csv_path.with_name(csv_path.stem + ".melt.qa.view.html")
    elif use_melt:
        out_path = csv_path.with_name(csv_path.stem + ".melt.view.html")
    else:
        out_path = csv_path.with_name(csv_path.stem + ".view.html")

    if use_melt:
        feats = _melt_feature_columns(
            cols,
            show_repeating_meta=bool(args.melt_show_repeating_meta),
        )
        if not feats:
            print(
                "Режим melt: после фильтров не осталось фич. Уберите --melt-compact, "
                "или добавьте колонки: --all-cols, --add-col …",
                flush=True,
            )
            return 1
        rows_d = _read_dict_rows(
            csv_path, cols, max_rows=args.max_rows or 0
        )
        page = _build_html_melt(
            title=csv_path.name,
            csv_path=str(csv_path),
            feature_cols=feats,
            rows=rows_d,
            focus_mode=bool(args.melt_compact) and not bool(args.all_cols),
            repeating_meta_suppressed=not bool(args.melt_show_repeating_meta),
            interesting_cfg=interesting_cfg,
            interesting_config_path=interesting_path_resolved,
            with_feature_descriptions=not bool(args.melt_no_descriptions),
            feature_desc_overrides=desc_overrides,
            feature_desc_config_path=desc_resolved,
            meta_timing_prefix=timing_prefix,
            qa_config=qa_obj,
            qa_config_path=qa_path_resolved,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as w:
            w.write(page)
        mode = "melt-interesting" if bool(args.melt_interesting) else "melt"
        if qa_obj is not None:
            mode += "+qa"
        src = "key-колонки" if (args.melt_compact and not args.all_cols) else "все колонки CSV"
        print(
            f"OK: {out_path} ({mode}: {len(feats)} кандидатов имён в CSV, {src})",
            flush=True,
        )
    else:
        headers, data = _read_rows(
            csv_path, cols, max_rows=args.max_rows or 0
        )
        page = _build_html(
            title=csv_path.name,
            csv_path=str(csv_path),
            headers=headers,
            data=data,
            focus_mode=not args.all_cols,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as w:
            w.write(page)
        print(
            f"OK: {out_path} ({len(data)} строк × {len(headers)} колонок)",
            flush=True,
        )
    if not args.no_open:
        # as_uri() требует абсолютный путь (относительный Path даёт ValueError)
        uri = out_path.resolve().as_uri()
        webbrowser.open(uri)
        print(f"Открыто в браузере: {uri}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
