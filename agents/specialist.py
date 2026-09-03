"""固定 Skill 的 Specialist Agent(审计 3.4 节)。

一个 SpecialistAgent 与一个固定 Skill 绑定;实例 = 固定 Skill + 当前样本
任务。它只负责从自己 Skill 的视角产出 ``specialist_report_v1``,不决定最终
类别。同一个 SpecialistAgent 实例可跨样本复用,Skill 更新时用 rebuild
换一份系统提示词即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from agents.common import run_agent
from model import get_model
from optimization.version_store import ActiveSkill
from orchestration.prompt_builder import specialist_system_message
from orchestration.skill_tools import build_skill_tools


class SpecialistAgent:
    """与单个固定 Skill 绑定的分析 Agent;Skill 的 scripts/ 会注册为工具。"""

    def __init__(
        self,
        config: Any,
        skill: ActiveSkill,
        tools_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.skill = skill
        self.tools_dir = Path(tools_dir) if tools_dir is not None else None
        self.model = get_model(config, role="analysis")
        self.agent = self._build()

    def _build(self) -> AssistantAgent:
        tools = build_skill_tools(self.skill, self.tools_dir)
        return AssistantAgent(
            name=self.skill.name,
            model_client=self.model,
            tools=tools,
            system_message=specialist_system_message(self.skill),
            max_tool_iterations=8 if tools else 1,
        )

    def rebuild(self, skill: ActiveSkill, tools_dir: Path | None = None) -> None:
        self.skill = skill
        if tools_dir is not None:
            self.tools_dir = Path(tools_dir)
        self.agent = self._build()

    async def run(self, task: str) -> str:
        return await run_agent(self.agent, task)

    async def close(self) -> None:
        await self.agent.close()
        await self.model.close()
