"""路由决策的规范化与选择逻辑(纯函数,便于单测与日志可复现)。

Coordinator(LLM)返回的 ``routing_decision_v1`` 必须先经过这里:
- 只保留目录中真实存在的 specialist 名字,防止模型自造角色(审计 3.4 节);
- 按 priority 排序、去重;
- 空选择时回退到默认集合,保证 Judge 至少能看到一次分析。
"""

from __future__ import annotations

from typing import Any

from orchestration.prompt_builder import SPECIALIST_SKILL_NAMES

# 当路由为空或解析失败时的兜底分析集合。
DEFAULT_FALLBACK_SKILLS = ["claim_decomposition"]


def normalize_routing_report(
    raw: dict[str, Any] | None,
    *,
    available_skills: set[str] | None = None,
    item_id: Any = None,
) -> dict[str, Any]:
    """把 Coordinator 原始输出规范成稳定的路由记录。

    返回结构:
    {
      "schema_version": "routing_decision_v1",
      "item_id": ...,
      "routing_features": {...},
      "routing_decision": {
        "selected_skills": [{"skill_name", "reason", "priority"}],
        "skipped_skills": [...],
        "stop_condition": ...,
        "routing_confidence": ...,
        "fallback_used": bool,
        "parse_error": bool
      }
    }
    """
    allowed = available_skills or set(SPECIALIST_SKILL_NAMES)
    if raw is None or not isinstance(raw, dict):
        return _fallback_record(item_id, parse_error=True)

    decision = raw.get("routing_decision")
    if not isinstance(decision, dict):
        return _fallback_record(item_id, parse_error=True)

    selected_raw = decision.get("selected_skills")
    if not isinstance(selected_raw, list):
        selected_raw = []

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in selected_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("skill_name", "")).strip()
        if name not in allowed or name in seen:
            continue
        seen.add(name)
        selected.append(
            {
                "skill_name": name,
                "reason": str(entry.get("reason", "")).strip(),
                "priority": int(entry.get("priority", 100)),
            }
        )
    selected.sort(key=lambda entry: (entry["priority"], entry["skill_name"]))

    skipped_raw = decision.get("skipped_skills")
    skipped: list[dict[str, Any]] = []
    if isinstance(skipped_raw, list):
        for entry in skipped_raw:
            if isinstance(entry, dict):
                name = str(entry.get("skill_name", "")).strip()
                if name in allowed:
                    skipped.append({"skill_name": name, "reason": str(entry.get("reason", "")).strip()})

    fallback_used = not selected
    if fallback_used:
        selected = [{"skill_name": name, "reason": "fallback: empty routing decision", "priority": 1}
                    for name in DEFAULT_FALLBACK_SKILLS]

    features = raw.get("routing_features")
    if not isinstance(features, dict):
        features = {}

    return {
        "schema_version": "routing_decision_v1",
        "item_id": raw.get("item_id", item_id),
        "dataset_format": str(raw.get("dataset_format", "")).strip() or None,
        "task_type": str(raw.get("task_type", "")).strip() or None,
        "language": str(raw.get("language", "")).strip() or None,
        "label_leakage_detected": bool(raw.get("label_leakage_detected", False)),
        "routing_features": features,
        "routing_decision": {
            "selected_skills": selected,
            "skipped_skills": skipped,
            "stop_condition": str(decision.get("stop_condition", "")).strip() or None,
            "routing_confidence": _to_float(decision.get("routing_confidence")),
            "fallback_used": fallback_used,
            "parse_error": False,
        },
    }


def selected_skill_names(routing: dict[str, Any]) -> list[str]:
    """按顺序取路由决策中的 specialist 名。"""
    decision = routing.get("routing_decision", {})
    entries = decision.get("selected_skills", [])
    if not isinstance(entries, list):
        return []
    return [str(entry.get("skill_name")) for entry in entries if isinstance(entry, dict)]


def _fallback_record(item_id: Any, *, parse_error: bool) -> dict[str, Any]:
    return {
        "schema_version": "routing_decision_v1",
        "item_id": item_id,
        "dataset_format": None,
        "task_type": None,
        "language": None,
        "label_leakage_detected": False,
        "routing_features": {},
        "routing_decision": {
            "selected_skills": [
                {"skill_name": name, "reason": "fallback: routing output unparseable", "priority": 1}
                for name in DEFAULT_FALLBACK_SKILLS
            ],
            "skipped_skills": [],
            "stop_condition": "routing output unparseable; ran default analysis",
            "routing_confidence": 0.0,
            "fallback_used": True,
            "parse_error": parse_error,
        },
    }


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
