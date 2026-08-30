"""工具注册、渐进披露和单次执行。"""
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field
from enum import StrEnum

SchemaLoader = Callable[[], Awaitable[dict[str, Any]]]
class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    DENIED = "denied"


class ToolResult(BaseModel):
    """可序列化、可审计的工具结果。"""

    tool_use_id: str
    tool_name: str
    content: str
    status: ToolStatus = ToolStatus.SUCCESS
    changed_paths: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.status is not ToolStatus.SUCCESS

class ToolExecutionContext(BaseModel):
    """一次工具执行的确定性环境。"""

    model_config = {"arbitrary_types_allowed": True}

    workspace: Path
    session_id: str
    agent_id: str
    # cancellation: CancellationToken

class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    progressive: bool = False

    def lightweight_definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description}

    def full_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    @abstractmethod
    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: dict[str, Any],
        tool_use_id: str,
    ) -> ToolResult:
        """执行工具并返回结构化结果。"""

class ToolRegistry:
    """注册表不负责并发策略，只负责发现、校验和权限。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_loaders: dict[str, SchemaLoader] = {}
        self._schema_cache: dict[str, dict[str, Any]] = {}
        self._activated: set[str] = set()

    def register(self, tool: Tool, schema_loader: SchemaLoader | None = None) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名称重复：{tool.name}")
        self._tools[tool.name] = tool
        if schema_loader:
            self._schema_loaders[tool.name] = schema_loader

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> set[str]:
        return set(self._tools)

    def lightweight_catalog(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            tool.lightweight_definition()
            for name, tool in self._tools.items()
            if allowed is None or name in allowed
        ]

    async def activate(self, name: str) -> str:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"工具不存在：{name}")
        if name not in self._schema_cache:
            loader = self._schema_loaders.get(name)
            self._schema_cache[name] = await loader() if loader else tool.input_schema
        self._activated.add(name)
        return f"工具 {name} 已加载，下一轮可直接调用。"

    async def definitions(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            if allowed is not None and name not in allowed:
                continue
            if tool.progressive and name not in self._activated:
                continue
            if name not in self._schema_cache:
                loader = self._schema_loaders.get(name)
                self._schema_cache[name] = await loader() if loader else tool.input_schema
            definition = tool.full_definition()
            definition["function"]["parameters"] = self._schema_cache[name]
            definitions.append(definition)
        return definitions

    async def execute_one(
        self,
        context: ToolExecutionContext,
        call: ToolCall,
        denied: set[str] | None = None,
    ) -> ToolResult:
        if call.name in (denied or set()):
            return ToolResult(
                tool_use_id=call.id,
                tool_name=call.name,
                content=f"权限拒绝：Agent 不允许使用工具 {call.name}",
                status=ToolStatus.DENIED,
            )
        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(
                tool_use_id=call.id,
                tool_name=call.name,
                content=f"未知工具：{call.name}",
                status=ToolStatus.ERROR,
            )
        schema = self._schema_cache.get(call.name, tool.input_schema)
        try:
            validate(call.arguments, schema)
        except ValidationError as exc:
            return ToolResult(
                tool_use_id=call.id,
                tool_name=call.name,
                content=f"工具参数校验失败：{exc.message}",
                status=ToolStatus.ERROR,
            )
        return await tool.execute(context, call.arguments, call.id)

