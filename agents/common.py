"""Agent 公共小工具:统一的模型调用与结果文本抽取。"""

from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent


async def run_agent(agent: AssistantAgent, task: str) -> str:
    """运行 AssistantAgent 并返回其最终消息的文本内容。"""
    result = await agent.run(task=task)
    return last_message_content(result)


def last_message_content(result: Any) -> str:
    """从 AutoGen TaskResult(或其替身)里取最后一条消息的文本。"""
    messages = getattr(result, "messages", None)
    if not messages:
        return str(result)
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None:
        return str(last)
    if isinstance(content, str):
        return content
    # 内容可能是结构化对象(例如 ToolResult),退回其字符串表示。
    return str(content)
