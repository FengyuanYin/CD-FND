import pytest

import agents.coordinator as coordinator_mod
import agents.judge as judge_mod
import agents.optimizer as optimizer_mod
import agents.specialist as specialist_mod
from agents.coordinator import CoordinatorAgent
from agents.judge import JudgeAgent
from agents.optimizer import OptimizationAgent
from agents.specialist import SpecialistAgent
from config import Config
from optimization.version_store import ActiveSkill
from orchestration.prompt_builder import ROUTING_SKILL_NAME

_CONFIG = Config(
    model_name="test-model",
    base_url="https://example.invalid/v1",
    api_key="test-key",
    epoch=1,
    dataset_name="weibo21",
)


class FakeModelClient:
    """不发网络请求的模型客户端替身(仅用于构造期断言)。"""

    def __init__(self):
        self.model_info = {"function_calling": True}

    async def close(self):
        pass


def _skill(name, instructions="Skill instructions."):
    return ActiveSkill(
        name=name,
        description=f"{name} desc",
        version="1.0.0",
        instructions=instructions,
    )


def _prompt(assistant_agent) -> str:
    """读取 AssistantAgent 的系统提示词文本(0.7.x 内部结构是 _system_messages)。"""
    messages = getattr(assistant_agent, "_system_messages", None) or []
    parts = []
    for message in messages:
        content = getattr(message, "content", None)
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def _skills_all():
    from orchestration.prompt_builder import ALL_SKILL_NAMES

    return {name: _skill(name) for name in ALL_SKILL_NAMES}


@pytest.fixture(autouse=True)
def _fake_model_client(monkeypatch):
    for module in (coordinator_mod, specialist_mod, judge_mod, optimizer_mod):
        monkeypatch.setattr(module, "get_model", lambda config, role: FakeModelClient())
    yield


def test_coordinator_message_includes_routing_skill_and_catalog():
    coordinator = CoordinatorAgent(_CONFIG, _skills_all())
    message = _prompt(coordinator.agent)
    assert ROUTING_SKILL_NAME in message
    for name in ("claim_decomposition", "evidence_assessment", "temporal_reasoning"):
        assert name in message
    assert "routing_decision_v1" in message


def test_specialist_message_uses_skill_instructions():
    specialist = SpecialistAgent(_CONFIG, _skill("claim_decomposition", "Special text."))
    message = _prompt(specialist.agent)
    assert "Special text." in message
    assert "specialist_report_v1" in message
    assert specialist.agent.name == "claim_decomposition"


def test_judge_message_includes_judge_skill():
    judge = JudgeAgent(_CONFIG, _skill("judge_decision", "Judge rules."))
    assert "Judge rules." in _prompt(judge.agent)
    assert "judge_decision_v2" in _prompt(judge.agent)


def test_optimizer_uses_static_prompt():
    optimizer = OptimizationAgent(_CONFIG)
    assert "optimization_report_v2" in _prompt(optimizer.agent)


def test_rebuild_updates_system_message_after_skill_change():
    coordinator = CoordinatorAgent(_CONFIG, _skills_all())
    before = _prompt(coordinator.agent)
    new_skills = dict(_skills_all())
    new_skills[ROUTING_SKILL_NAME] = _skill(
        ROUTING_SKILL_NAME, "Completely new routing rules."
    )
    coordinator.rebuild(new_skills)
    after = _prompt(coordinator.agent)
    assert "Completely new routing rules" in after
    assert before != after


def test_agent_attaches_scripts_as_tools():
    """带 scripts/ 的 Skill 构建 Agent 时不应报错(tool 注册见 test_skill_tools)。"""
    from pathlib import Path
    from uuid import uuid4

    from optimization.version_store import scan_skill_resources

    root = Path("outputs") / "unit_tests"
    root.mkdir(parents=True, exist_ok=True)
    skill_dir = root / f"agent_tools_{uuid4().hex}"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "helper.py").write_text(
        'def count_words(text: str) -> int:\n    """Count words."""\n    return len(text.split())\n',
        encoding="utf-8",
    )
    scripted_skill = _skill("claim_decomposition", "With tools.")
    scripted_skill = ActiveSkill(
        name=scripted_skill.name,
        description=scripted_skill.description,
        version=scripted_skill.version,
        instructions=scripted_skill.instructions,
        resources=scan_skill_resources(skill_dir),
    )
    specialist = SpecialistAgent(_CONFIG, scripted_skill, tools_dir=skill_dir)
    assert "With tools." in _prompt(specialist.agent)
    assert specialist.agent._system_messages
