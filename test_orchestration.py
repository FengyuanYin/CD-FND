from orchestration.aggregator import build_analysis_report, normalize_claim_text
from orchestration.executor import dataset_format_label, visible_item
from orchestration.prompt_builder import (
    ROUTING_SKILL_NAME,
    SPECIALIST_SKILL_NAMES,
    coordinator_system_message,
    judge_system_message,
    specialist_system_message,
)
from orchestration.router import normalize_routing_report, selected_skill_names
from optimization.version_store import ActiveSkill

ROUTING = {
    "schema_version": "routing_decision_v1",
    "item_id": "i1",
    "dataset_format": "weibo21",
    "task_type": "single_item_classification",
    "language": "zh",
    "routing_features": {"sentence_count": 3, "has_date": True},
    "routing_decision": {
        "selected_skills": [
            {"skill_name": "temporal_reasoning", "reason": "有日期", "priority": 2},
            {"skill_name": "claim_decomposition", "reason": "多句", "priority": 1},
            {"skill_name": "made_up_skill", "reason": "目录外,应被剔除", "priority": 1},
        ],
        "skipped_skills": [{"skill_name": "evidence_assessment", "reason": "无证据字段"}],
        "stop_condition": "中心主张被覆盖",
        "routing_confidence": 0.8,
    },
}


def _wrap(skill, version, report):
    return {
        "skill_name": skill,
        "skill_version": version,
        "report": report,
        "parse_error": report is None,
        "error": None if report is not None else "boom",
    }


def test_normalize_routing_keeps_catalog_and_sorts_by_priority():
    normalized = normalize_routing_report(ROUTING)
    decision = normalized["routing_decision"]
    assert decision["parse_error"] is False
    assert decision["fallback_used"] is False
    # made_up_skill 被剔除,priority 升序。
    assert selected_skill_names(normalized) == ["claim_decomposition", "temporal_reasoning"]
    assert [e["skill_name"] for e in decision["skipped_skills"]] == ["evidence_assessment"]


def test_normalize_routing_unparseable_falls_back():
    normalized = normalize_routing_report(None)
    assert normalized["routing_decision"]["fallback_used"] is True
    assert selected_skill_names(normalized) == ["claim_decomposition"]


def test_normalize_routing_empty_selection_falls_back():
    raw = {"routing_decision": {"selected_skills": []}}
    normalized = normalize_routing_report(raw)
    assert normalized["routing_decision"]["fallback_used"] is True
    assert selected_skill_names(normalized) == ["claim_decomposition"]


def test_visible_item_removes_label_fields():
    item = {
        "content": "正文",
        "category": "科技",
        "label": 0,
        "expected_result": 1,
        "nested": {"content": "子内容", "label": 9},
    }
    cleaned = visible_item(item)
    assert "label" not in cleaned
    assert "expected_result" not in cleaned
    assert "label" not in cleaned["nested"]
    assert cleaned["content"] == "正文"


def test_dataset_format_label():
    assert dataset_format_label("weibo") == "weibo21"
    assert dataset_format_label("weibo21") == "weibo21"
    assert dataset_format_label("amtcele") == "amtcele"


def test_aggregator_merges_duplicate_claims_and_records_conflict():
    report_a = {
        "schema_version": "specialist_report_v1",
        "skill_name": "claim_decomposition",
        "skill_version": "1.0.0",
        "status": "COMPLETED",
        "claims": [
            {
                "claim_id": "C1",
                "text": "某地发生地震,伤亡过百。",
                "importance": "CENTRAL",
                "assessment": "SUPPORTED",
                "confidence": 0.9,
                "evidence_ids": ["E1"],
                "reason": "与 E1 一致",
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "origin": "SUPPLIED",
                "source": "报道A",
                "content_used": "伤亡过百",
                "supports_claim_ids": ["C1"],
                "refutes_claim_ids": [],
                "temporal_fit": "VALID",
                "limitations": [],
            }
        ],
        "limitations": [],
    }
    report_b = {
        "schema_version": "specialist_report_v1",
        "skill_name": "temporal_reasoning",
        "skill_version": "1.0.0",
        "status": "COMPLETED",
        "claims": [
            {
                "claim_id": "C1",
                "text": "某地发生地震,伤亡过百。",  # 归一化后与上面相同
                "importance": "CENTRAL",
                "assessment": "REFUTED",
                "confidence": 0.7,
                "evidence_ids": [],
                "reason": "时间线矛盾",
            }
        ],
        "evidence": [],
        "limitations": [],
    }
    analysis = build_analysis_report(
        normalize_routing_report(ROUTING),
        [_wrap("claim_decomposition", "1.0.0", report_a),
         _wrap("temporal_reasoning", "1.0.0", report_b)],
    )
    assert analysis["schema_version"] == "analysis_report_v2"
    assert len(analysis["claims"]) == 1
    claim = analysis["claims"][0]
    assert claim["assessed_by"] == ["claim_decomposition", "temporal_reasoning"]
    assert len(analysis["conflicts"]) == 1
    assert analysis["conflicts"][0]["claim_id"] == claim["claim_id"]
    # evidence 局部 id 已重映射为全局 E1,并被 claim 引用。
    assert analysis["evidence"][0]["evidence_id"] == "E1"
    assert claim["evidence_ids"] == ["E1"]
    assert len(analysis["specialists"]) == 2
    assert claim["claim_id"] in analysis["judge_handoff"]["decisive_claim_ids"]


def test_aggregator_handles_failed_report():
    analysis = build_analysis_report(
        normalize_routing_report(ROUTING),
        [_wrap("evidence_assessment", "1.0.0", None)],
    )
    assert len(analysis["claims"]) == 0
    assert analysis["specialists"][0]["status"] == "FAILED"
    assert analysis["overall_uncertainty"] == "HIGH"


def _skill(name):
    return ActiveSkill(
        name=name,
        description=f"{name} desc",
        version="1.0.0",
        instructions="Follow the instructions.",
    )


def test_prompt_builder_composes_role_skill_and_catalog():
    routing_skill = _skill(ROUTING_SKILL_NAME)
    catalog = [_skill(name) for name in SPECIALIST_SKILL_NAMES]

    coordinator_msg = coordinator_system_message(routing_skill, catalog)
    assert "routing_decision_v1" in coordinator_msg
    assert "<activated_skill>" in coordinator_msg
    assert "claim_decomposition" in coordinator_msg  # 目录可见

    specialist_msg = specialist_system_message(_skill("claim_decomposition"))
    assert "specialist_report_v1" in specialist_msg
    assert "Follow the instructions" in specialist_msg

    judge_msg = judge_system_message(_skill("judge_decision"))
    assert "judge_decision_v2" in judge_msg
    assert "Follow the instructions" in judge_msg


def test_normalize_claim_text():
    assert normalize_claim_text("地震,伤亡过百!") == normalize_claim_text("地震 伤亡过百")


def _skill_with_resources(name, resources: dict):
    return ActiveSkill(
        name=name,
        description=f"{name} desc",
        version="1.0.0",
        instructions="Follow the instructions.",
        resources=resources,
    )


def test_resources_injected_only_from_references_and_templates():
    skill = _skill_with_resources(
        "claim_decomposition",
        {
            "references/rules.md": "参考规则A",
            "templates/report.json": '{"template": 1}',
            "assets/data.txt": "资产不应注入",
            "scripts/helper.py": "def helper(): pass",
        },
    )
    message = specialist_system_message(skill)
    assert "<skill_resources>" in message
    assert "参考规则A" in message
    assert '"template": 1' in message
    assert "资产不应注入" not in message
    assert "def helper" not in message


def test_judge_and_coordinator_inject_their_own_skill_resources():
    judge_skill = _skill_with_resources("judge_decision", {"references/policy.md": "裁判规则"})
    assert "裁判规则" in judge_system_message(judge_skill)

    routing_skill = _skill_with_resources(ROUTING_SKILL_NAME, {"templates/plan.md": "路由模板"})
    coordinator_msg = coordinator_system_message(
        routing_skill, [_skill(name) for name in SPECIALIST_SKILL_NAMES]
    )
    assert "路由模板" in coordinator_msg
