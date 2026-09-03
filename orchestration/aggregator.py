"""结构化结果聚合器(审计 5 节: AGG)。

把多个 Specialist 的 ``specialist_report_v1`` 合并成一份供 Judge 消费的
``analysis_report_v2``:
- 相同 claim 文本(归一化后)跨报告合并,保留各自的 assessment 来源,以便
  Judge 看到分歧;
- claim / evidence 的局部 id(C1、E1)被重映射为全局稳定 id,保证可审计;
- 分歧与缺失信息被显式放进 conflicts / missing_information。
本模块是纯函数,不调用模型。
"""

from __future__ import annotations

import re
from typing import Any

IMPORTANCE_RANK = {"CENTRAL": 0, "SUPPORTING": 1, "MINOR": 2, "UNKNOWN": 3}
ASSESSMENTS = {
    "SUPPORTED",
    "REFUTED",
    "MIXED",
    "INSUFFICIENT_EVIDENCE",
    "NOT_CHECKABLE",
}

_TOKEN_PATTERN = re.compile(r"[\W_]+", re.UNICODE)


def normalize_claim_text(text: str) -> str:
    """claim 文本的轻量归一化(小写、去空白与标点)用于跨报告合并。"""
    if not isinstance(text, str):
        return ""
    return _TOKEN_PATTERN.sub("", text.casefold())


def _first_of(value: Any, default: Any = None) -> Any:
    return value if value is not None else default


def build_analysis_report(
    routing_report: dict[str, Any],
    specialist_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """聚合出 analysis_report_v2。

    specialist_reports 的每个元素是已解析或解析失败的报告包装:
    {"skill_name": str, "skill_version": str, "report": dict|None,
     "parse_error": bool, "error": str|None}
    """
    claims_by_key: dict[str, dict[str, Any]] = {}
    claim_global: dict[tuple[str, str], str] = {}
    evidence_by_key: dict[tuple[str, str], str] = {}
    evidence_entries: list[dict[str, Any]] = []
    specialists_summary: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    leakage_detected = bool(routing_report.get("label_leakage_detected", False))
    next_claim_id = 1
    next_evidence_id = 1

    def local_claim_id(skill: str, entry: dict[str, Any]) -> str:
        return str(entry.get("claim_id") or f"{skill}/C{len(claims_by_key) + 1}")

    for wrapped in specialist_reports:
        skill_name = str(wrapped.get("skill_name", "unknown_skill"))
        skill_version = str(wrapped.get("skill_version", ""))
        report = wrapped.get("report")
        if report is None or not isinstance(report, dict):
            specialists_summary.append(
                {
                    "name": skill_name,
                    "version": skill_version or "",
                    "status": "FAILED",
                    "contribution": "报告无法解析,没有可用的结构化结论",
                    "limitations": [str(wrapped.get("error", "parse_error"))],
                }
            )
            continue

        leakage_detected = leakage_detected or bool(report.get("label_leakage_detected", False))
        local_evidence: list[dict[str, Any]] = report.get("evidence")
        if not isinstance(local_evidence, list):
            local_evidence = []

        # 1) 先登记本报告内的 evidence,建立局部→全局映射。
        report_evidence_map: dict[str, str] = {}
        for entry in local_evidence:
            if not isinstance(entry, dict):
                continue
            local_id = str(entry.get("evidence_id") or f"E{next_evidence_id}")
            key = (skill_name, local_id)
            if key in evidence_by_key:
                global_id = evidence_by_key[key]
            else:
                global_id = f"E{next_evidence_id}"
                next_evidence_id += 1
                evidence_by_key[key] = global_id
                evidence_entries.append(
                    {
                        "evidence_id": global_id,
                        "origin": str(entry.get("origin", "SUPPLIED")),
                        "source": entry.get("source"),
                        "published_at": entry.get("published_at"),
                        "content_used": str(entry.get("content_used", "")),
                        "supports_claim_ids": [],
                        "refutes_claim_ids": [],
                        "temporal_fit": str(entry.get("temporal_fit", "UNKNOWN")),
                        "limitations": list(entry.get("limitations", []) or []),
                        "assessed_by": skill_name,
                    }
                )
            report_evidence_map[local_id] = global_id

        # 2) 再合并 claims,并解析它们引用的局部 evidence id。
        local_claims = report.get("claims")
        if not isinstance(local_claims, list):
            local_claims = []
        report_claim_map: dict[str, str] = {}
        for entry in local_claims:
            if not isinstance(entry, dict):
                continue
            local_id = local_claim_id(skill_name, entry)
            key = (skill_name, local_id)
            report_claim_map[local_id] = claim_global.get(
                key, claim_global.get((skill_name, local_id), "")
            )
            text = str(entry.get("text", "")).strip()
            norm = normalize_claim_text(text)
            assessment = str(entry.get("assessment", "INSUFFICIENT_EVIDENCE")).upper()
            if assessment not in ASSESSMENTS:
                assessment = "INSUFFICIENT_EVIDENCE"
            confidence = _clamp01(entry.get("confidence"))
            importance = str(entry.get("importance", "UNKNOWN")).upper()

            if norm in claims_by_key:
                merged = claims_by_key[norm]
                merged["assessment_sources"].append(
                    {"skill": skill_name, "assessment": assessment, "confidence": confidence}
                )
                if IMPORTANCE_RANK.get(importance, 3) < IMPORTANCE_RANK.get(merged["importance"], 3):
                    merged["importance"] = importance
                existing_reasons = set(merged["assessed_by"])
                if skill_name not in existing_reasons:
                    merged["assessed_by"] = merged["assessed_by"] + [skill_name]
                merged["reasons"][skill_name] = str(entry.get("reason", "")).strip()
                claim_global[key] = merged["claim_id"]
                report_claim_map[local_id] = merged["claim_id"]
                # evidence 引用延后到本报告 evidence 映射完成后统一补充。
                merged["_pending_evidence"].append(
                    (skill_name, list(entry.get("evidence_ids", []) or []))
                )
            else:
                claim_id = f"C{next_claim_id}"
                next_claim_id += 1
                record: dict[str, Any] = {
                    "claim_id": claim_id,
                    "text": text,
                    "importance": importance,
                    "assessment": assessment,
                    "confidence": confidence,
                    "evidence_ids": [],
                    "reason": str(entry.get("reason", "")).strip(),
                    "assessed_by": [skill_name],
                    "assessment_sources": [
                        {"skill": skill_name, "assessment": assessment, "confidence": confidence}
                    ],
                    "reasons": {skill_name: str(entry.get("reason", "")).strip()},
                    "_pending_evidence": [(skill_name, list(entry.get("evidence_ids", []) or []))],
                }
                claims_by_key[norm] = record
                claim_global[key] = claim_id
                report_claim_map[local_id] = claim_id

        # 3) 把本报告内 claim→evidence 的局部引用转成全局引用。
        for entry in local_claims:
            if not isinstance(entry, dict):
                continue
            local_id = local_claim_id(skill_name, entry)
            global_claim = report_claim_map.get(local_id)
            if not global_claim:
                continue
            record = next((c for c in claims_by_key.values() if c["claim_id"] == global_claim), None)
            if record is None:
                continue
            resolved = set(record["evidence_ids"])
            for local_evidence_id in list(entry.get("evidence_ids", []) or []):
                global_evidence = report_evidence_map.get(str(local_evidence_id))
                if global_evidence:
                    resolved.add(global_evidence)
            record["evidence_ids"] = sorted(resolved)

        # 4) 同步 evidence 条目里对 claim 的引用。
        for entry in local_evidence:
            if not isinstance(entry, dict):
                continue
            local_id = str(entry.get("evidence_id") or "")
            global_id = report_evidence_map.get(local_id)
            if not global_id:
                continue
            target = next((e for e in evidence_entries if e["evidence_id"] == global_id), None)
            if target is None:
                continue
            for ref, field in (("supports_claim_ids", "supports_claim_ids"),
                               ("refutes_claim_ids", "refutes_claim_ids")):
                for local_claim_ref in list(entry.get(ref, []) or []):
                    global_claim = report_claim_map.get(str(local_claim_ref))
                    if global_claim:
                        target[field].append(global_claim)

        # 5) specialists 汇总。
        claims_count = len(local_claims)
        note = str(report.get("note", "")).strip() if report.get("note") else ""
        contribution = note or (
            f"评估了 {claims_count} 条 claim,{len(local_evidence)} 条 evidence"
        )
        specialists_summary.append(
            {
                "name": skill_name,
                "version": skill_version or "",
                "status": str(report.get("status", "COMPLETED")),
                "contribution": contribution,
                "limitations": list(report.get("limitations", []) or []),
            }
        )

    # 6) 计算冲突(同一 claim 出现互相矛盾或含 MIXED 的评估)。
    for record in claims_by_key.values():
        sources = record["assessment_sources"]
        verdict_set = {s["assessment"] for s in sources}
        contradictory = ({"SUPPORTED", "REFUTED"} <= verdict_set) or ("MIXED" in verdict_set)
        if len(verdict_set) > 1 and contradictory:
            conflicts.append(
                {
                    "claim_id": record["claim_id"],
                    "text": record["text"],
                    "assessments": [
                        {"skill": s["skill"], "assessment": s["assessment"], "confidence": s["confidence"]}
                        for s in sources
                    ],
                    "reason": "不同 Specialist 对该 claim 给出了矛盾评估",
                }
            )

    # 7) 组装稳定输出字段。
    ordered_claims = sorted(
        claims_by_key.values(),
        key=lambda c: (IMPORTANCE_RANK.get(c["importance"], 3), c["claim_id"]),
    )
    for record in ordered_claims:
        record.pop("_pending_evidence", None)

    # evidence 全局条目只保留最终引用并排序。
    for entry in evidence_entries:
        entry["supports_claim_ids"] = sorted(set(entry["supports_claim_ids"]))
        entry["refutes_claim_ids"] = sorted(set(entry["refutes_claim_ids"]))

    decisive = [
        c for c in ordered_claims
        if c["importance"] == "CENTRAL" and c["assessment"] in {"SUPPORTED", "REFUTED"}
    ]
    unresolved = [
        c for c in ordered_claims
        if c["importance"] == "CENTRAL"
        and c["assessment"] in {"MIXED", "INSUFFICIENT_EVIDENCE", "NOT_CHECKABLE"}
    ]
    strongest_ids: set[str] = set()
    for claim in decisive:
        strongest_ids.update(claim["evidence_ids"])
    for entry in evidence_entries:
        if (set(entry["supports_claim_ids"]) | set(entry["refutes_claim_ids"])) & {
            c["claim_id"] for c in decisive
        }:
            strongest_ids.add(entry["evidence_id"])

    counts = {a: sum(1 for c in ordered_claims if c["assessment"] == a) for a in ASSESSMENTS}
    central_uncertain = any(c["assessment"] in {"INSUFFICIENT_EVIDENCE", "NOT_CHECKABLE"} for c in unresolved)

    if conflicts or central_uncertain:
        overall_uncertainty = "HIGH"
    elif any(c["evidence_ids"] for c in decisive):
        overall_uncertainty = "LOW"
    elif ordered_claims:
        overall_uncertainty = "MEDIUM"
    else:
        overall_uncertainty = "HIGH"

    supported_text = f"{counts['SUPPORTED']} 条支持、{counts['REFUTED']} 条反驳、{counts['MIXED']} 条混合"
    if unresolved:
        supported_text += f",{len(unresolved)} 条中心主张未决"
    if conflicts:
        supported_text += f",{len(conflicts)} 处跨 Specialist 分歧"
    summary = (
        f"聚合 {len(specialists_summary)} 个 Specialist 的 {len(ordered_claims)} 条主张:"
        f"{supported_text}。"
    )

    signals = {
        "internal_consistency": (
            "LOW" if any(c["assessment"] == "REFUTED" for c in decisive)
            else "HIGH" if decisive else "UNKNOWN"
        ),
        "source_transparency": "UNKNOWN",
        "temporal_consistency": "UNKNOWN",
        "context_completeness": "UNKNOWN",
        "sensational_language": "UNKNOWN",
        "unsupported_causality": "UNKNOWN",
    }

    return {
        "schema_version": "analysis_report_v2",
        "item_id": routing_report.get("item_id"),
        "dataset_format": routing_report.get("dataset_format"),
        "task_type": routing_report.get("task_type"),
        "language": routing_report.get("language"),
        "label_leakage_detected": leakage_detected,
        "input_quality": {"content_fields_used": [], "metadata_fields_used": []},
        "claims": ordered_claims,
        "evidence": evidence_entries,
        "paired_comparison": {"applicable": False},
        "signals": signals,
        "specialists": specialists_summary,
        "conflicts": conflicts,
        "missing_information": [
            {"claim_id": c["claim_id"], "text": c["text"]}
            for c in ordered_claims
            if c["assessment"] in {"INSUFFICIENT_EVIDENCE", "NOT_CHECKABLE"}
        ],
        "overall_uncertainty": overall_uncertainty,
        "judge_handoff": {
            "decisive_claim_ids": [c["claim_id"] for c in decisive],
            "strongest_evidence_ids": sorted(strongest_ids),
            "unresolved_claim_ids": [c["claim_id"] for c in unresolved],
            "summary": summary,
        },
        "routing": routing_report,
    }


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
