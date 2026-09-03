"""Judge Agent(审计 5 节)。

消费聚合后的 ``analysis_report_v2`` + 源样本 + label_schema,输出
``judge_decision_v2``(最终预测/弃权/强制映射)。Skill(judge_decision)更新后
用 rebuild 重建底层 Agent。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from agents.common import run_agent
from model import get_model
from optimization.version_store import ActiveSkill
from orchestration.prompt_builder import judge_system_message
from orchestration.skill_tools import build_skill_tools


class JudgeAgent:
    """输出最终预测的 Judge;judge_decision Skill 的 scripts/ 会注册为工具。"""

    def __init__(
        self,
        config: Any,
        judge_skill: ActiveSkill,
        tools_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.judge_skill = judge_skill
        self.tools_dir = Path(tools_dir) if tools_dir is not None else None
        self.model = get_model(config, role="judge")
        self.agent = self._build()

    def _build(self) -> AssistantAgent:
        tools = build_skill_tools(self.judge_skill, self.tools_dir)
        return AssistantAgent(
            name="JudgeAgent",
            model_client=self.model,
            tools=tools,
            system_message=judge_system_message(self.judge_skill),
            max_tool_iterations=8 if tools else 1,
        )

    def rebuild(
        self,
        judge_skill: ActiveSkill,
        tools_dir: Path | None = None,
    ) -> None:
        self.judge_skill = judge_skill
        if tools_dir is not None:
            self.tools_dir = Path(tools_dir)
        self.agent = self._build()

    async def run(self, task: str) -> str:
        return await run_agent(self.agent, task)

    async def close(self) -> None:
        await self.agent.close()
        await self.model.close()
