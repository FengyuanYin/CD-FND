"""AutoGen 可直接使用的本地文件工具。

模块中的异步函数可以直接传给 ``AssistantAgent(tools=[...])``。AutoGen 会根据
函数签名、``Annotated`` 参数说明和文档字符串生成工具 Schema，并执行工具调用。

安全边界：所有路径都被限制在本项目 ``WORKSPACE`` 内；搜索会跳过常见缓存和
二进制目录；写入使用同目录临时文件和原子替换，避免产生不完整文件。
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool


# tools.py 位于项目根目录，因此将它所在目录作为工具能访问的工作区。
WORKSPACE = Path(__file__).resolve().parent

# 避免搜索 Git 数据、虚拟环境、依赖和缓存内容。
EXCLUDED_DIRECTORIES = {
    ".git",
    ".autocode",
    ".agents",
    ".codex",
    ".idea",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}

# 命令输出过长会占用大量 Agent 上下文，因此设置字符上限。
MAX_COMMAND_OUTPUT = 50_000


DANGEROUS_COMMANDS = re.compile(
    r"(?:"
    r"\brm\s+(?:-[^\s]*r[^\s]*f|-[^\s]*f[^\s]*r)\b|"
    r"\b(?:del|erase|rmdir|rd)\b|"
    r"\bRemove-Item\b[^\r\n]*(?:-Recurse|-Force)|"
    r"\b(?:format|diskpart|shutdown|reboot|poweroff)\b|"
    r"\breg\s+delete\b|"
    r"\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f)|"
    r"\b(?:Invoke-Expression|iex)\b"
    r")",
    flags=re.IGNORECASE,
)


def _resolve_inside_workspace(path: str) -> Path:
    """把用户路径解析成工作区内的绝对路径。

    这里只检查安全边界，不要求目标已存在。``Path.resolve`` 会消除 ``.`` 和
    ``..``，随后用 ``relative_to`` 阻止 ``../../secret.txt`` 一类路径穿越。
    """

    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    workspace = WORKSPACE.resolve()
    candidate = Path(path)
    target = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()

    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path is outside the workspace: {path}") from exc
    return target


def resolve_existing_workspace_path(path: str) -> Path:
    """解析只读路径，并确认目标已经存在。"""

    target = _resolve_inside_workspace(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return target


def resolve_writable_workspace_path(path: str) -> Path:
    """解析可写路径；不要求目标存在，因此允许创建新文件。"""

    return _resolve_inside_workspace(path)


def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
    """在耗时循环和产生副作用之前响应 AutoGen 的取消请求。"""

    if cancellation_token is not None and cancellation_token.is_cancelled():
        raise asyncio.CancelledError


async def search_text(
    query: Annotated[str, "Text to search for in UTF-8 project files."],
    path: Annotated[
        str, "File or directory path relative to the project workspace."
    ] = ".",
    limit: Annotated[
        int, "Maximum matches to return; must be between 1 and 10000."
    ] = 1000,
    case_sensitive: Annotated[
        bool, "Whether matching distinguishes uppercase and lowercase."
    ] = False,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """在项目内的 UTF-8 文件中搜索文本，返回文件路径、行号和原始行。

    ``cancellation_token`` 是框架参数，AutoGen 执行工具时会自动注入；模型不需要
    提供它。不能按 UTF-8 解码或无法读取的文件会被跳过。
    """

    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    if not 1 <= limit <= 10_000:
        raise ValueError("limit must be between 1 and 10000")

    target = resolve_existing_workspace_path(path)
    needle = query if case_sensitive else query.casefold()
    matches: list[dict[str, Any]] = []
    skipped_files = 0

    # 同时支持搜索单个文件和递归搜索目录。
    files = (target,) if target.is_file() else target.rglob("*")

    for file in files:
        _raise_if_cancelled(cancellation_token)
        if not file.is_file():
            continue

        relative_path = file.relative_to(WORKSPACE)
        if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
            continue

        try:
            # 逐行读取，避免把大文件一次性载入内存。
            with file.open("r", encoding="utf-8") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    _raise_if_cancelled(cancellation_token)
                    line = raw_line.rstrip("\r\n")
                    haystack = line if case_sensitive else line.casefold()
                    if needle not in haystack:
                        continue

                    matches.append(
                        {
                            "path": relative_path.as_posix(),
                            "line": line_number,
                            "text": line,
                        }
                    )
                    if len(matches) >= limit:
                        return {
                            "status": "success",
                            "query": query,
                            "search_path": path,
                            "matches": matches,
                            "count": len(matches),
                            "truncated": True,
                            "skipped_files": skipped_files,
                        }
        except (UnicodeDecodeError, OSError):
            # 二进制、非 UTF-8 和暂时不可访问的文件不终止整个搜索。
            skipped_files += 1

    return {
        "status": "success",
        "query": query,
        "search_path": path,
        "matches": matches,
        "count": len(matches),
        "truncated": False,
        "skipped_files": skipped_files,
    }


async def write_file(
    path: Annotated[str, "Target path relative to the project workspace."],
    content: Annotated[
        str, "Complete UTF-8 text; existing file content will be replaced."
    ],
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """将完整 UTF-8 内容原子写入工作区文件。

    这是完整覆盖工具，不是追加或局部编辑工具。它先写入目标同目录的唯一临时
    文件，再调用 ``os.replace`` 替换目标。两者位于同一文件系统时，替换通常是
    原子的，可以降低进程中断后留下半个文件的风险。
    """

    if not isinstance(content, str):
        raise TypeError("content must be a string")

    _raise_if_cancelled(cancellation_token)
    target = resolve_writable_workspace_path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"Target path is a directory: {path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

    try:
        temporary.write_text(content, encoding="utf-8")
        _raise_if_cancelled(cancellation_token)
        # 目标存在时覆盖，不存在时完成重命名。
        os.replace(temporary, target)
    finally:
        # 成功后临时文件已不存在；失败时尽力清理，不掩盖原始异常。
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

    relative_path = target.relative_to(WORKSPACE).as_posix()
    byte_count = len(content.encode("utf-8"))
    return {
        "status": "success",
        "message": (
            f"Atomically wrote {relative_path} "
            f"({len(content)} characters, {byte_count} UTF-8 bytes)."
        ),
        "path": relative_path,
        "changed_paths": [relative_path],
        "character_count": len(content),
        "byte_count": byte_count,
        "encoding": "utf-8",
    }


async def read_file(
    path: Annotated[str, "File path relative to the project workspace."],
    start_line: Annotated[int, "First line to read, using one-based numbering."] = 1,
    end_line: Annotated[
        int | None,
        "Last line to read, inclusive. Use null to read through the final line.",
    ] = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """读取工作区内 UTF-8 文本文件的指定行区间。

    返回内容中的每一行都带有真实的一基行号，便于 Agent 后续生成精确的修改请求。
    ``end_line`` 是闭区间；例如 ``start_line=2, end_line=4`` 会读取第 2～4 行。
    """

    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be at least 1 or null")
    if end_line is not None and start_line > end_line:
        raise ValueError("start_line must not be greater than end_line")

    _raise_if_cancelled(cancellation_token)
    target = resolve_existing_workspace_path(path)
    if not target.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8 text: {path}") from exc

    _raise_if_cancelled(cancellation_token)
    line_count = len(lines)

    # 空文件允许从第 1 行开始读取并返回空内容；非空文件不接受越过末行的起点。
    if line_count > 0 and start_line > line_count:
        raise ValueError(
            f"start_line {start_line} is greater than the file line count {line_count}"
        )
    if line_count == 0 and start_line != 1:
        raise ValueError("An empty file can only be read from start_line 1")

    actual_end = line_count if end_line is None else min(end_line, line_count)
    numbered_lines = [
        f"{index + 1}: {lines[index]}"
        for index in range(start_line - 1, actual_end)
    ]
    relative_path = target.relative_to(WORKSPACE).as_posix()

    return {
        "status": "success",
        "path": relative_path,
        "content": "\n".join(numbered_lines),
        "start_line": start_line,
        "end_line": actual_end,
        "line_count": line_count,
        "truncated": end_line is not None and end_line < line_count,
        "encoding": "utf-8",
    }


async def list_files(
    path: Annotated[
        str, "Directory or file path relative to the project workspace."
    ] = ".",
    pattern: Annotated[
        str, "Recursive glob pattern such as '*.py' or 'skills/**/*.md'."
    ] = "*",
    limit: Annotated[
        int, "Maximum number of paths to return; must be between 1 and 10000."
    ] = 2000,
    include_directories: Annotated[
        bool, "Whether directory paths should be included in the result."
    ] = False,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """递归列出工作区内匹配 glob 的文件，可选择同时返回目录。

    ``pattern`` 由 ``Path.rglob`` 解释。为保持路径边界清晰，这里拒绝绝对 glob 和
    包含 ``..`` 的 pattern。返回路径统一为相对于工作区的 POSIX 风格路径。
    """

    if not pattern or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError("pattern must be relative and must not contain '..'")
    if not 1 <= limit <= 10_000:
        raise ValueError("limit must be between 1 and 10000")

    _raise_if_cancelled(cancellation_token)
    target = resolve_existing_workspace_path(path)

    # 单文件目标不执行 glob；匹配文件名时直接返回它。
    if target.is_file():
        candidates = (target,) if target.match(pattern) else ()
    else:
        candidates = target.rglob(pattern)

    entries: list[str] = []
    matched_count = 0
    for item in candidates:
        _raise_if_cancelled(cancellation_token)
        relative_path = item.relative_to(WORKSPACE)
        if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
            continue
        if item.is_dir() and not include_directories:
            continue

        matched_count += 1
        # 只保存 limit 条，仍继续计数以返回真实的匹配总数。
        if len(entries) < limit:
            entries.append(relative_path.as_posix())

    entries.sort()
    return {
        "status": "success",
        "search_path": path,
        "pattern": pattern,
        "entries": entries,
        "count": matched_count,
        "returned_count": len(entries),
        "truncated": matched_count > limit,
        "include_directories": include_directories,
    }


async def apply_patch(
    path: Annotated[str, "Target UTF-8 file relative to the project workspace."],
    old_text: Annotated[str, "Exact existing text to find in the target file."],
    new_text: Annotated[str, "Replacement text."],
    replace_all: Annotated[
        bool, "Replace every occurrence instead of requiring one unique occurrence."
    ] = False,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """对工作区文本文件执行精确字符串替换，并原子写回。

    默认要求 ``old_text`` 在文件中恰好出现一次。如果出现多次，调用方必须提供更
    完整的上下文，或者显式设置 ``replace_all=True``。这能降低 Agent 意外修改错误
    位置的风险。此工具不是 unified-diff 解析器。
    """

    if not old_text:
        raise ValueError("old_text must not be empty")
    if not isinstance(new_text, str):
        raise TypeError("new_text must be a string")

    _raise_if_cancelled(cancellation_token)
    target = resolve_existing_workspace_path(path)
    if not target.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")

    try:
        original = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8 text: {path}") from exc

    occurrence_count = original.count(old_text)
    if occurrence_count == 0:
        raise ValueError("Patch context was not found; the file was not modified")
    if occurrence_count > 1 and not replace_all:
        raise ValueError(
            f"Patch context occurs {occurrence_count} times; provide unique context "
            "or enable replace_all"
        )

    replacement_count = occurrence_count if replace_all else 1
    updated = original.replace(old_text, new_text, -1 if replace_all else 1)
    _raise_if_cancelled(cancellation_token)

    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(updated, encoding="utf-8")
        _raise_if_cancelled(cancellation_token)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

    relative_path = target.relative_to(WORKSPACE).as_posix()
    return {
        "status": "success",
        "message": f"Updated {relative_path}; replaced {replacement_count} occurrence(s).",
        "path": relative_path,
        "changed_paths": [relative_path],
        "replacement_count": replacement_count,
        "encoding": "utf-8",
    }


async def run_command(
    command: Annotated[str, "Shell command to execute inside the project workspace."],
    timeout_seconds: Annotated[
        int, "Execution timeout in seconds; must be between 1 and 300."
    ] = 60,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """在项目工作区运行命令，返回退出码和合并后的标准输出/错误输出。

    命令通过 PowerShell（Windows）或 ``/bin/sh``（其他系统）执行。高风险模式会被
    拒绝，执行时间和输出长度也受到限制。正则拒绝规则不能替代容器或操作系统级
    沙箱，因此该工具只应授权给确实需要运行测试的受信任 Agent。
    """

    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be between 1 and 300")

    if DANGEROUS_COMMANDS.search(command):
        return {
            "status": "denied",
            "message": "Command matched a high-risk rule and was not executed.",
            "exit_code": None,
            "output": "",
            "truncated": False,
        }

    _raise_if_cancelled(cancellation_token)
    if os.name == "nt":
        program = "powershell.exe"
        program_args = ["-NoProfile", "-NonInteractive", "-Command", command]
    else:
        program = "/bin/sh"
        program_args = ["-lc", command]

    process = await asyncio.create_subprocess_exec(
        program,
        *program_args,
        cwd=WORKSPACE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "AUTOAGENT_WORKSPACE": str(WORKSPACE)},
    )
    communicate_task = asyncio.create_task(process.communicate())

    # AutoGen 取消时先取消等待任务；finally 中还会终止尚未退出的子进程。
    if cancellation_token is not None:
        cancellation_token.link_future(communicate_task)

    try:
        try:
            output, _ = await asyncio.wait_for(
                communicate_task,
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "status": "error",
                "message": f"Command exceeded {timeout_seconds} seconds and was terminated.",
                "exit_code": None,
                "output": "",
                "truncated": False,
            }
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise

        text = output.decode(errors="replace")
        truncated = len(text) > MAX_COMMAND_OUTPUT
        text = text[:MAX_COMMAND_OUTPUT]
        exit_code = process.returncode
        return {
            "status": "success" if exit_code == 0 else "error",
            "message": f"Command exited with code {exit_code}.",
            "exit_code": exit_code,
            "output": text,
            "truncated": truncated,
        }
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if not communicate_task.done():
            communicate_task.cancel()




# 简单用法：AssistantAgent 会自动把这些函数包装成 FunctionTool。
AUTOGEN_TOOL_FUNCTIONS = [
    read_file,
    list_files,
    search_text,
    write_file,
    apply_patch,
    run_command,
]

# 显式包装版本适合在多个 Agent 之间复用同一组工具对象。
AUTOGEN_FUNCTION_TOOLS = [
    FunctionTool(
        read_file,
        description=(
            "Read a selected one-based line range from a UTF-8 file inside the "
            "project workspace and return numbered lines."
        ),
    ),
    FunctionTool(
        list_files,
        description=(
            "Recursively list files matching a glob pattern inside the project "
            "workspace."
        ),
    ),
    FunctionTool(
        search_text,
        description=(
            "Search UTF-8 files inside the project workspace and return matching "
            "paths, line numbers, and source lines."
        ),
    ),
    FunctionTool(
        write_file,
        description=(
            "Atomically write complete UTF-8 text inside the project workspace. "
            "This replaces existing file content."
        ),
    ),
    FunctionTool(
        apply_patch,
        description=(
            "Safely replace exact text in a UTF-8 project file. The context must "
            "be unique unless replace_all is explicitly enabled."
        ),
    ),
    FunctionTool(
        run_command,
        description=(
            "Run a non-destructive shell command in the project workspace with "
            "timeout, cancellation, and output limits."
        ),
    ),
]


__all__ = [
    "AUTOGEN_FUNCTION_TOOLS",
    "AUTOGEN_TOOL_FUNCTIONS",
    "WORKSPACE",
    "apply_patch",
    "list_files",
    "read_file",
    "resolve_existing_workspace_path",
    "resolve_writable_workspace_path",
    "run_command",
    "search_text",
    "write_file",
]
