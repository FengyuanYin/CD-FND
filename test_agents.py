import asyncio
from unittest.mock import AsyncMock, Mock

from autogen_core.tools import FunctionTool

from agents import ChildAgentManager, CoordinatorAgent
from dynamic_workbench import DynamicWorkbench


async def sample_tool(text: str) -> str:
    """Return the supplied text."""
    return text


def test_dynamic_workbench_register_and_call():
    async def scenario():
        workbench = DynamicWorkbench()
        await workbench.register(FunctionTool(sample_tool, description="Return text"))
        assert [item["name"] for item in await workbench.list_tools()] == ["sample_tool"]
        result = await workbench.call_tool("sample_tool", {"text": "ok"})
        assert result.is_error is False
        assert result.to_text() == "ok"
    asyncio.run(scenario())


def test_child_agent_is_registered_as_tool():
    async def scenario():
        model_client = Mock()
        model_client.model_info = {"function_calling": True}
        workbench = DynamicWorkbench()
        allowed = {"sample_tool": FunctionTool(sample_tool, description="Return text")}
        manager = ChildAgentManager(
            model_client=model_client,
            workbench=workbench,
            allowed_tools=allowed,
            max_children=1,
        )
        result = await manager.spawn_agent(
            name="reviewer",
            description="Reviews supplied material.",
            system_message="Review the task and return a concise report.",
            tool_names=["sample_tool"],
        )
        assert result["registered_tool"] == "reviewer"
        assert "reviewer" in manager.children
        assert [item["name"] for item in await workbench.list_tools()] == ["reviewer"]
        manager.children["reviewer"].close = AsyncMock()
        assert await manager.remove_agent("reviewer") is True
        assert await workbench.list_tools() == []
    asyncio.run(scenario())


def test_child_agent_rejects_unapproved_tools():
    async def scenario():
        manager = ChildAgentManager(
            model_client=Mock(model_info={"function_calling": True}),
            workbench=DynamicWorkbench(),
            allowed_tools={},
        )
        try:
            await manager.spawn_agent(
                name="reviewer",
                description="Reviews supplied material.",
                system_message="Return a concise review.",
                tool_names=["unapproved_tool"],
            )
        except ValueError as exc:
            assert "unauthorized" in str(exc)
        else:
            raise AssertionError("An unapproved tool was accepted")
    asyncio.run(scenario())


def test_coordinator_exposes_spawn_tool():
    config = Mock(
        model_name="test-model",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        vision_support=False,
        function_calling=True,
        json_output=True,
        structured_output=True,
    )
    coordinator = CoordinatorAgent(config)

    async def scenario():
        names = [item["name"] for item in await coordinator.workbench.list_tools()]
        assert names == ["spawn_agent"]
        await coordinator.close()
    asyncio.run(scenario())
