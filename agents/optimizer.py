"""Optimization Agent(审计 6 节)。

在批次结束后接收:该批样本的完整 trace + Gold Label + 当前各 Skill 版本 +
聚合指标,输出 ``optimization_report_v2``。它一次只针对一个 Skill 提出
候选正文补丁,是否真的应用由调用方根据 allowed_actions 与独立验证集决定,
本 Agent 自身永不修改任何文件。
"""

from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent

from agents.common import run_agent
from model import get_model
from prompts import OPTIMIZATION_AGENT_SYSTEM_PROMPT


class OptimizationAgent:
    """只生成诊断与候选补丁、不直接改 Skill 的优化 Agent。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model = get_model(config, role="optimization")
        self.agent = AssistantAgent(
            name="OptimizationAgent",
            model_client=self.model,
            system_message=OPTIMIZATION_AGENT_SYSTEM_PROMPT,
            max_tool_iterations=1,
        )

    async def run(self, task: str) -> str:
        return await run_agent(self.agent, task)

    async def close(self) -> None:
        await self.agent.close()
        await self.model.close()
