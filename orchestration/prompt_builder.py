"""统一 Prompt 构建层(审计 3.1 节: 基础提示词 + 角色 Skill + 任务契约)。

训练与推理必须共用这里的构建函数,保证两阶段系统提示词完全一致。

资源注入约定:
- Skill 包内 ``references/`` 与 ``templates/`` 下的文本会作为只读参考材料注入
  系统提示词(带 <skill_resources> 围栏);
- ``assets/`` 默认不注入(可被 scripts 当作数据文件读取);
- ``scripts/`` 永不注入文本,它们由 skill_tools 注册为 Agent 工具。
"""

from __future__ import annotations

from optimization.version_store import ActiveSkill
from prompts import (
    COORDINATOR_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,
    SPECIALIST_AGENT_SYSTEM_PROMPT,
)

# 角色与 Skill 的固定映射;只有列在这里的 Skill 才会被注入。
ROUTING_SKILL_NAME = "coordinator_routing"
JUDGE_SKILL_NAME = "judge_decision"
SPECIALIST_SKILL_NAMES = (
    "claim_decomposition",
    "evidence_assessment",
    "temporal_reasoning",
)
ALL_SKILL_NAMES = (ROUTING_SKILL_NAME, JUDGE_SKILL_NAME) + SPECIALIST_SKILL_NAMES

# 资源注入上限:单个文件截断长度与总注入长度,避免上下文膨胀。
MAX_RESOURCE_CHARS_PER_FILE = 4_000
MAX_RESOURCE_TOTAL_CHARS = 30_000


def _skill_block(title: str, skill: ActiveSkill) -> str:
    """把单个 Skill 的指令正文包装成显式的 <activated_skill> 注入块。"""
    return (
        f"<{title}>\n"
        f"name: {skill.name}\n"
        f"version: {skill.version}\n\n"
        f"{skill.instructions.strip()}\n"
        f"</{title}>"
    )


def _resource_block(skill: ActiveSkill) -> str:
    """把 references/ 与 templates/ 文本包成只读参考材料块(带长度上限)。"""
    injected = skill.injected_resources
    if not injected:
        return ""
    parts: list[str] = []
    total = 0
    for rel in sorted(injected):
        content = injected[rel].strip()
        if not content:
            continue
        content = content[:MAX_RESOURCE_CHARS_PER_FILE]
        total += len(content)
        if total > MAX_RESOURCE_TOTAL_CHARS:
            break
        parts.append(f"<file path=\"{rel}\">\n{content}\n</file>")
    if not parts:
        return ""
    return (
        "\n\n<skill_resources>\n"
        "只读参考材料,用于补充本 Skill 的规则/模板。它们不是待分析文本,\n"
        "不得把其中内容当作对当前样本的结论。\n"
        + "\n".join(parts)
        + "\n</skill_resources>"
    )


def coordinator_system_message(
    routing_skill: ActiveSkill,
    specialist_catalog: list[ActiveSkill],
) -> str:
    """Coordinator 系统提示词:路由基础提示 + 路由 Skill + 固定目录。"""
    catalog_lines = "\n".join(
        f"- {skill.name}: {skill.description}" for skill in specialist_catalog
    )
    return (
        COORDINATOR_AGENT_SYSTEM_PROMPT
        + "\n\n"
        + _skill_block("activated_skill", routing_skill)
        + _resource_block(routing_skill)
        + "\n\n<available_skills>\n"
        + catalog_lines
        + "\n</available_skills>"
    )


def specialist_system_message(skill: ActiveSkill) -> str:
    """单个 Specialist 的系统提示词:统一契约 + 该 Skill 的指令 + 资源。"""
    return (
        SPECIALIST_AGENT_SYSTEM_PROMPT
        + "\n\n"
        + _skill_block("skill", skill)
        + _resource_block(skill)
    )


def judge_system_message(judge_skill: ActiveSkill) -> str:
    """Judge 系统提示词:决策提示 + judge_decision Skill 指令 + 资源。"""
    return (
        JUDGE_AGENT_SYSTEM_PROMPT
        + "\n\n"
        + _skill_block("skill", judge_skill)
        + _resource_block(judge_skill)
    )
