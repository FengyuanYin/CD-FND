"""从 AutoGen 输出文本中稳健地解析单个 JSON 对象。

模型可能返回裸 JSON、带 Markdown 围栏、或前后附加说明。本模块统一收口
解析逻辑:先尝试整段 JSON,再尝试摘取第一个平衡的 {...} 或 [...],失败返回
None 由调用方记录为解析失败(计入输出格式错误,而不是悄悄丢样本)。
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT_START = re.compile(r"\{")
_ARRAY_START = re.compile(r"\[")


def parse_json_content(text: str) -> dict[str, Any] | None:
    """尽量把模型输出解析为 dict;无法解析时返回 None。"""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    candidates: list[str] = []
    # 1) 去除可能的 markdown 围栏后整体尝试。
    fenced = _FENCE_PATTERN.findall(stripped)
    candidates.extend(fenced)

    # 2) 直接从 { 或 [ 到文本末尾截取。
    for start_match in (_OBJECT_START.search(stripped), _ARRAY_START.search(stripped)):
        if start_match:
            candidates.append(stripped[start_match.start():])

    # 3) 最后尝试原文本本身。
    candidates.append(stripped)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            return value[0]
    return None


def extract_json_array(text: str) -> list[Any] | None:
    """解析顶层为 JSON 数组的输出;失败返回 None。"""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    for start_match in (_ARRAY_START.search(stripped),):
        if start_match:
            candidate = stripped[start_match.start():]
            try:
                value = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, list):
                return value
    return None
