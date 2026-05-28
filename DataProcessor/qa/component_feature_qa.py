"""
Плоский вывод meta (как в batch_runs_feature_report) и проверка значений
по rules из JSON (view_csv_feature_qa.json).

Схема rules (per component, см. any_component + components):
  column: str          — имя колонки как в wide CSV
  min / max: number    — для чисел (мс в *_ms, как в CSV)
  enum: [str, ...]     — допустимые строковые значения
  hint: str            — пояснение (показ в title / CLI)
  optional: true       — пустая строка не вызывает предупреждения
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_NOT_NUM = re.compile(r"^nan$|^none$|^\s*$", re.I)


def flatten_meta(meta: Any, prefix: str = "meta_") -> Dict[str, Any]:
    """Совпадает с batch_runs_feature_report._flatten_meta — один источник правды."""
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k == "stage_timings_ms" and isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)):
                    out[f"{prefix}timing_{sk}"] = sv
            continue
        if k == "asr_stage_timings_ms" and isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)):
                    out[f"{prefix}asr_timing_{sk}"] = sv
            continue
        if isinstance(v, bool):
            out[f"{prefix}{k}"] = int(v)
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = v
        elif isinstance(v, str) and len(v) < 200:
            out[f"{prefix}{k}"] = v
    return out


def _parse_floatish(raw: str) -> Optional[float]:
    t = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not t or _NOT_NUM.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _rule_msg(rule: dict, col: str, fact: str) -> str:
    parts = [f"column {col!r}"]
    if "min" in rule or "max" in rule:
        lo = rule.get("min")
        hi = rule.get("max")
        if lo is not None and hi is not None:
            parts.append(f"ожидается [{lo}, {hi}]")
        elif lo is not None:
            parts.append(f"ожидается >= {lo}")
        elif hi is not None:
            parts.append(f"ожидается <= {hi}")
    if "enum" in rule:
        parts.append(f"ожидается одно из: {rule['enum']}")
    if rule.get("hint"):
        parts.append(str(rule["hint"]))
    parts.append(f"факт: {fact!r}")
    return "; ".join(parts)


@dataclass
class QaConfig:
    """Правила any_component (база) + переопределения по component.

    Поддерживает:
    - exact component key: "text_processor"
    - regex component key: "re:^text_processor(?:/.*)?$"
    - exact column rule: {"column": "tp_xxx", ...}
    - regex column rule: {"column_regex": "^tp_.*_flag$", ...}
    """

    any_component: List[dict] = field(default_factory=list)
    by_component: Dict[str, List[dict]] = field(default_factory=dict)
    by_component_regex: List[Tuple[re.Pattern[str], List[dict]]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "QaConfig":
        any_c = data.get("any_component")
        if not isinstance(any_c, list):
            any_c = []
        by_c = data.get("components")
        if not isinstance(by_c, dict):
            by_c = {}
        exact: Dict[str, List[dict]] = {}
        regex_rules: List[Tuple[re.Pattern[str], List[dict]]] = []
        for k, v in by_c.items():
            if not isinstance(v, list):
                continue
            ks = str(k)
            if ks.startswith("re:"):
                pat = ks[3:]
                try:
                    rx = re.compile(pat)
                    regex_rules.append((rx, list(v)))
                except re.error:
                    # bad regex key: ignore to keep config robust
                    continue
            else:
                exact[ks] = list(v)
        return cls(
            any_component=list(any_c),
            by_component=exact,
            by_component_regex=regex_rules,
        )

    def rules_for_column(self, component: str) -> Dict[str, dict]:
        """column -> single merged rule; component overrides any_component on same column."""
        merged: Dict[str, dict] = {}
        for r in self.any_component:
            if isinstance(r, dict) and r.get("column"):
                merged[str(r["column"])] = r
        for r in self.by_component.get(component, []):
            if isinstance(r, dict) and r.get("column"):
                merged[str(r["column"])] = r
        for rx, arr in self.by_component_regex:
            if rx.search(component):
                for r in arr:
                    if isinstance(r, dict) and r.get("column"):
                        merged[str(r["column"])] = r
        return merged

    def _rules_for_component(self, component: str) -> List[dict]:
        arr: List[dict] = []
        arr.extend([r for r in self.any_component if isinstance(r, dict)])
        arr.extend([r for r in self.by_component.get(component, []) if isinstance(r, dict)])
        for rx, lst in self.by_component_regex:
            if rx.search(component):
                arr.extend([r for r in lst if isinstance(r, dict)])
        return arr

    def rule_for(self, component: str, column: str) -> Optional[dict]:
        """Вернуть наиболее релевантное правило для колонки.

        Приоритет:
        1) exact column match
        2) first matching column_regex (в порядке merge)
        """
        rules = self._rules_for_component(component)
        exact: Optional[dict] = None
        regex_hit: Optional[dict] = None
        for rule in rules:
            c = rule.get("column")
            if isinstance(c, str) and c == column:
                exact = rule
        if exact is not None:
            return exact
        for rule in rules:
            cr = rule.get("column_regex")
            if isinstance(cr, str):
                try:
                    if re.search(cr, column):
                        regex_hit = rule
                        break
                except re.error:
                    continue
        return regex_hit

    def warning_for(self, component: str, column: str, raw: str) -> Optional[str]:
        """
        None — ок или нет правила; иначе строка предупреждения (для title / CLI).
        """
        rule = self.rule_for(component, column)
        if not rule or not isinstance(rule, dict):
            return None
        s = (raw if raw is not None else "").strip()
        if rule.get("optional") and not s:
            return None
        if "enum" in rule:
            ev = rule["enum"]
            if not isinstance(ev, (list, tuple)):
                return None
            allowed = {str(x) for x in ev}
            if s in allowed or s.lower() in {x.lower() for x in allowed}:
                return None
            return _rule_msg(rule, column, s or "(пусто)")

        if "min" in rule or "max" in rule:
            v = _parse_floatish(s)
            if v is None or math.isnan(v):
                if rule.get("optional") and (not s or s.lower() in ("nan", "none")):
                    return None
                return _rule_msg(rule, column, s or "не число")
            lo = rule.get("min")
            hi = rule.get("max")
            if lo is not None and v < float(lo):
                return _rule_msg(rule, column, s)
            if hi is not None and v > float(hi):
                return _rule_msg(rule, column, s)
            return None
        return None


def load_qa_config(path: Path) -> QaConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return QaConfig()
    return QaConfig.from_dict(data)


def find_repo_root_from_path(start: Path) -> Optional[Path]:
    p = start.resolve()
    for d in [p, *p.parents]:
        if (d / "storage" / "result_store" / "view_csv_feature_qa.json").is_file():
            return d
    return None
