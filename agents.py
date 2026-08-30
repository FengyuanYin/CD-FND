"""AutoGen agents used by the misinformation detection pipeline."""

from __future__ import annotations

import re
from typing import Annotated, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_core.tools import BaseTool, FunctionTool

from dynamic_workbench import DynamicWorkbench
from model import get_model
from prompts import COORDINATOR_AGENT_SYSTEM_PROMPT, JUDGE_AGENT_SYSTEM_PROMPT, OPTIMIZATION_AGENT_SYSTEM_PROMPT
from tools import AUTOGEN_FUNCTION_TOOLS, AUTOGEN_TOOL_FUNCTIONS


AGENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class ChildAgentManager:
    """Create bounded specialist agents and expose them through a workbench."""

    def __init__(
        self,
        *,
        model_client: Any,
        workbench: DynamicWorkbench,
        allowed_tools: dict[str, BaseTool[Any, Any]],
        max_children: int = 5,
    ) -> None:
        self.model_client = model_client
        self.workbench = workbench
        self.allowed_tools = allowed_tools
        self.max_children = max_children
        self.children: dict[str, AssistantAgent] = {}

    async def spawn_agent(
        self,
        name: Annotated[str, "Unique lowercase name such as evidence_reviewer."],
        description: Annotated[str, "Short description of the specialist capability."],
        system_message: Annotated[str, "Bounded role, task rules, and output contract."],
        tool_names: Annotated[list[str], "Approved tool names assigned to the specialist."],
    ) -> dict[str, Any]:
        """Create a specialist child agent and register it as a callable tool."""

        if not AGENT_NAME_PATTERN.fullmatch(name):
            raise ValueError("name must use lowercase letters, digits, or underscores")
        if name in self.children:
            raise ValueError(f"Child agent already exists: {name}")
        if len(self.children) >= self.max_children:
            raise ValueError(f"Child-agent limit reached: {self.max_children}")
        if not description.strip() or not system_message.strip():
            raise ValueError("description and system_message must not be empty")

        unique_tool_names = list(dict.fromkeys(tool_names))
        unknown = [item for item in unique_tool_names if item not in self.allowed_tools]
        if unknown:
            raise ValueError(f"Unknown or unauthorized tools: {unknown}")

        child = AssistantAgent(
            name=name,
            description=description,
            model_client=self.model_client,
            tools=[self.allowed_tools[item] for item in unique_tool_names],
            system_message=system_message,
            max_tool_iterations=10,
        )
        child_tool = AgentTool(child, return_value_as_last_message=True)
        await self.workbench.register(child_tool)
        self.children[name] = child

        return {
            "status": "success",
            "agent_name": name,
            "registered_tool": child_tool.name,
            "assigned_tools": unique_tool_names,
        }

    async def remove_agent(self, name: str) -> bool:
        """Remove a child agent and its callable tool."""

        child = self.children.get(name)
        if child is None:
            return False
        await self.workbench.unregister(name)
        await child.close()
        del self.children[name]
        return True

    async def close(self) -> None:
        for name in list(self.children):
            await self.remove_agent(name)


class CoordinatorAgent:
    def __init__(self, config: Any, max_children: int = 10):
        self.config = config
        self.model = get_model(config, role="coordinate")
        allowed_tools = {tool.name: tool for tool in AUTOGEN_FUNCTION_TOOLS}

        # The manager and workbench reference each other: the spawn tool creates
        # AgentTools and registers them back into this same workbench.
        self.workbench = DynamicWorkbench()
        self.child_manager = ChildAgentManager(
            model_client=self.model,
            workbench=self.workbench,
            allowed_tools=allowed_tools,
            max_children=max_children,
        )
        spawn_tool = FunctionTool(
            self.child_manager.spawn_agent,
            name="spawn_agent",
            description=(
                "Create a bounded specialist agent with approved tools. The new "
                "agent becomes callable by its name on the next tool iteration."
            ),
        )
        self.workbench = DynamicWorkbench([spawn_tool])
        self.child_manager.workbench = self.workbench
        self.agent = AssistantAgent(
            name="CoordinatorAgent",
            model_client=self.model,
            system_message=COORDINATOR_AGENT_SYSTEM_PROMPT,
            workbench=self.workbench,
            max_tool_iterations=100,
        )

    async def run(self, user_input: str):
        return await self.agent.run(task=user_input)

    async def close(self) -> None:
        await self.child_manager.close()
        await self.agent.close()
        await self.model.close()


class OptimizationAgent:
    def __init__(self, config: Any):
        self.config = config
        self.model = get_model(config, role="optimization")
        self.agent = AssistantAgent(
            name="OptimizationAgent",
            model_client=self.model,
            system_message=OPTIMIZATION_AGENT_SYSTEM_PROMPT,
            tools=AUTOGEN_TOOL_FUNCTIONS,
        )

    async def optimize(self, user_input: str):
        return await self.agent.run(task=user_input)

    async def close(self) -> None:
        await self.agent.close()
        await self.model.close()


class JudgeAgent:
    def __init__(self, config: Any):
        self.config = config
        self.model = get_model(config, role="judge")
        self.agent = AssistantAgent(
            name="JudgeAgent",
            model_client=self.model,
            system_message=JUDGE_AGENT_SYSTEM_PROMPT,
        )

    async def judge(self, user_input: str):
        return await self.agent.run(task=user_input)

    async def close(self) -> None:
        await self.agent.close()
        await self.model.close()
