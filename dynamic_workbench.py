"""Runtime-mutable AutoGen workbench."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool, StaticWorkbench, ToolResult, ToolSchema, Workbench


class DynamicWorkbench(Workbench):
    """Workbench whose tools may be added or removed between model calls."""

    def __init__(self, tools: list[BaseTool[Any, Any]] | None = None) -> None:
        self._tools: dict[str, BaseTool[Any, Any]] = {}
        self._lock = asyncio.Lock()
        self._initial_tool_names: set[str] = set()
        for tool in tools or []:
            if tool.name in self._tools:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool
            self._initial_tool_names.add(tool.name)

    async def register(self, tool: BaseTool[Any, Any], *, replace: bool = False) -> str:
        async with self._lock:
            if tool.name in self._tools and not replace:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool
        return tool.name

    async def unregister(self, name: str) -> bool:
        async with self._lock:
            if name in self._initial_tool_names:
                raise ValueError(f"Initial tool cannot be removed: {name}")
            return self._tools.pop(name, None) is not None

    async def list_tools(self) -> list[ToolSchema]:
        async with self._lock:
            tools = list(self._tools.values())
        return await StaticWorkbench(tools).list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        async with self._lock:
            tools = list(self._tools.values())
        return await StaticWorkbench(tools).call_tool(
            name, arguments, cancellation_token, call_id
        )

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def reset(self) -> None:
        async with self._lock:
            self._tools = {
                name: tool
                for name, tool in self._tools.items()
                if name in self._initial_tool_names
            }

    async def save_state(self) -> Mapping[str, Any]:
        async with self._lock:
            return {"registered_tool_names": list(self._tools)}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        requested = set(state.get("registered_tool_names", []))
        async with self._lock:
            missing = requested.difference(self._tools)
        if missing:
            raise ValueError(
                "Register trusted tools before loading this state: "
                f"{sorted(missing)}"
            )
