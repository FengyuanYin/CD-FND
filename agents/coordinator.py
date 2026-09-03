"""路由 Coordinator(审计 4 节: 两阶段路由器)。

第一阶段抽取可观察的路由特征,第二阶段从固定 Specialist 目录中选择,
输出 ``routing_decision_v1``。本 Agent 不执行 Specialist 分析、不输出最终
类别;目录由 PromptBuilder 从 version store 注入,模型无法自造角色。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from agents.common import run_agent
from model import get_model
from optimization.version_store import ActiveSkill
from orchestration.prompt_builder import (
    ROUTING_SKILL_NAME,
    SPECIALIST_SKILL_NAMES,
    coordinator_system_message,
)
from orchestration.skill_tools import build_skill_tools


class CoordinatorAgent:
    """路由 Agent。构造后即可用 run(task) 推理;Skill 更新后调用 rebuild。"""

    def __init__(
        self,
        config: Any,
        skills: dict[str, ActiveSkill],
        tools_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.skills = skills
        self.tools_dir = Path(tools_dir) if tools_dir is not None else None
        self.model = get_model(config, role="coordinate")
        self.agent = self._build()

    def _build(self) -> AssistantAgent:
        routing_skill = self.skills[ROUTING_SKILL_NAME]
        catalog = [
            self.skills[name] for name in SPECIALIST_SKILL_NAMES if name in self.skills
        ]
        tools = build_skill_tools(routing_skill, self.tools_dir)
        return AssistantAgent(
            name="CoordinatorAgent",
            model_client=self.model,
            tools=tools,
            system_message=coordinator_system_message(routing_skill, catalog),
            max_tool_iterations=8 if tools else 1,
        )

    def rebuild(
        self,
        skills: dict[str, ActiveSkill],
        tools_dir: Path | None = None,
    ) -> None:
        """Skill 晋升/回滚后重建底层 Agent,让新版本立即生效。"""
        self.skills = skills
        if tools_dir is not None:
            self.tools_dir = Path(tools_dir)
        self.agent = self._build()

    async def run(self, task: str) -> str:
        return await run_agent(self.agent, task)

    async def close(self) -> None:
        await self.agent.close()
        await self.model.close()
