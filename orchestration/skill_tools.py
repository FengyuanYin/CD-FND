"""把 Skill 包内 ``scripts/*.py`` 注册为该 Skill 绑定 Agent 的工具。

约定(写入 README,供 Skill 作者遵循):
- ``scripts/`` 下的每个 ``.py`` 文件会被 ``importlib`` 按路径加载(隔离模块名);
- 工具 = 模块内公开、可调用、且带 docstring 的函数;若模块定义了 ``__all__``,
  则只注册 ``__all__`` 中列出的名字(最确定,推荐);
- 禁止以下划线开头的私有函数;跳过 ``__module__`` 不属于本模块的导入函数;
- 每个函数用 ``FunctionTool`` 包装:description 取自函数 docstring 首行,
  参数说明沿用函数签名中的 ``Annotated`` 或类型注解;
- 两个模块导出同名函数时显式报错,避免工具名冲突;
- ``scripts/`` 与工具在 Skill 晋升后随 ``rebuild`` 自动重载(新代码立即生效)。
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable

from autogen_core.tools import FunctionTool

from optimization.version_store import ActiveSkill


def _import_script(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本模块: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_tool_functions(module: Any) -> list[Callable[..., Any]]:
    """按约定收集模块里应注册为工具的函数。"""
    declared = getattr(module, "__all__", None)
    candidates: list[tuple[str, Any]] = []
    if isinstance(declared, list):
        for name in declared:
            if not isinstance(name, str):
                continue
            candidates.append((name, getattr(module, name, None)))
    else:
        candidates = [
            (name, value)
            for name, value in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_")
        ]
    tools: list[Callable[..., Any]] = []
    for name, value in candidates:
        if not callable(value):
            continue
        func = value
        if inspect.isfunction(func):
            # 只接受定义在本脚本模块中的函数,排除 import 进来的工具库函数。
            if getattr(func, "__module__", None) != module.__name__:
                continue
        doc = inspect.getdoc(func) or ""
        if not doc.strip():
            continue  # 无说明的函数不注册,保持目录可读
        tools.append(func)
    return tools


def _description(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else func.__name__


def build_skill_tools(
    skill: ActiveSkill,
    skill_dir: Path | None,
) -> list[FunctionTool]:
    """为该 Skill 绑定 Agent 构造工具列表。

    skill_dir 是当前 active 整包所在目录(基线在 skills/<name>,晋升版本在
    version store 物化目录);没有 scripts 或目录缺失时返回空列表。
    """
    if not skill.scripts:
        return []
    if skill_dir is None:
        raise ValueError(
            f"Skill {skill.name} 声明了 scripts/ 但未提供可加载目录(skill_dir)"
        )
    base_dir = Path(skill_dir).resolve()
    tools: list[FunctionTool] = []
    seen_names: dict[str, Path] = {}
    module_counter = 0
    for rel in skill.scripts:
        script_path = (base_dir / rel).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(
                f"Skill {skill.name} 声明脚本缺失: {rel}(目录: {base_dir})"
            )
        module_counter += 1
        module = _import_script(
            script_path, f"_skill_{skill.name}_{module_counter}"
        )
        for func in _public_tool_functions(module):
            tool_name = func.__name__
            if tool_name in seen_names:
                raise ValueError(
                    f"Skill {skill.name}: 脚本 {seen_names[tool_name]} 与 "
                    f"{rel} 都导出了工具函数 {tool_name!r},请改名或用 __all__ 消歧"
                )
            seen_names[tool_name] = Path(rel)
            try:
                tools.append(FunctionTool(func, description=_description(func)))
            except Exception as exc:
                raise ValueError(
                    f"Skill {skill.name}: 脚本 {rel} 的函数 {tool_name!r} 无法生成"
                    f"工具 Schema,请为参数补全类型注解/Annotated 说明"
                    f"({type(exc).__name__}: {exc})"
                ) from exc
    return tools
