"""AutoGen Agent 包(新的固定-Skill 多智能体结构)。

包内模块:
- coordinator.py  路由 Coordinator(输出 routing_decision_v1)
- specialist.py   固定 Skill 的 Specialist(输出 specialist_report_v1)
- judge.py        Judge(消费聚合后的 analysis_report_v2,输出 judge_decision_v2)
- optimizer.py    优化 Agent(针对单个 Skill 生成候选正文)

统一的约定:每个 Agent 的 ``run(task: str) -> str`` 直接返回模型最终文本
内容(由调用方用 output_parsing 解析),``close()`` 负责释放模型客户端;
Skill 晋升后调用 ``rebuild(skills)`` 重建底层 AssistantAgent。
"""

from __future__ import annotations

from agents.common import last_message_content, run_agent
from agents.coordinator import CoordinatorAgent
from agents.judge import JudgeAgent
from agents.optimizer import OptimizationAgent
from agents.specialist import SpecialistAgent

__all__ = [
    "CoordinatorAgent",
    "JudgeAgent",
    "OptimizationAgent",
    "SpecialistAgent",
    "last_message_content",
    "run_agent",
]
