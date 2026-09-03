"""scripts/*.py → Agent 工具的注册测试。"""

from pathlib import Path
from uuid import uuid4

import pytest

from optimization.version_store import ActiveSkill
from orchestration.skill_tools import build_skill_tools

HELPER = '''"""计数与日期辅助工具。"""

from typing import Annotated


def count_sentences(text: Annotated[str, "待计数的文本"]) -> int:
    """按常见句末标点粗估句子数量。"""
    for sep in "。！？!?.":
        text = text.replace(sep, sep + "\\n")
    return len([p for p in text.splitlines() if p.strip()])


def _private_helper(x):
    return x
'''


def make_skill_dir(root: Path, *, with_second: bool = False) -> Path:
    skill_dir = root / "demo_skill"
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "scripts" / "helper.py").write_text(HELPER, encoding="utf-8")
    if with_second:
        (skill_dir / "scripts" / "extra.py").write_text(
            "def count_sentences(text: str) -> int:\n"
            "    \"\"\"Count sentences (dup).\"\"\"\n"
            "    return 0\n",
            encoding="utf-8",
        )
    return skill_dir


def skill_with_scripts(skill_dir: Path) -> ActiveSkill:
    from optimization.version_store import scan_skill_resources

    return ActiveSkill(
        name="demo_skill",
        description="demo",
        version="1.0.0",
        instructions="Do stuff.",
        resources=scan_skill_resources(skill_dir),
    )


@pytest.fixture()
def scratch_dir() -> Path:
    root = Path("outputs") / "unit_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    yield path
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            child.unlink() if child.is_file() else child.rmdir()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def test_registers_public_documented_functions(scratch_dir):
    skill_dir = make_skill_dir(scratch_dir)
    skill = skill_with_scripts(skill_dir)
    tools = build_skill_tools(skill, skill_dir)

    names = [tool.name for tool in tools]
    assert "count_sentences" in names
    assert "_private_helper" not in names
    tool = next(tool for tool in tools if tool.name == "count_sentences")
    assert "句子" in tool.description or "句子" in str(tool.description)


def test_no_scripts_returns_empty(scratch_dir):
    skill = ActiveSkill(name="x", description="x", version="1.0.0",
                        instructions="i", resources={})
    assert build_skill_tools(skill, scratch_dir) == []


def test_scripts_require_skill_dir(scratch_dir):
    skill = ActiveSkill(name="x", description="x", version="1.0.0",
                        instructions="i",
                        resources={"scripts/a.py": "def f():\\n    return 1\\n"})
    with pytest.raises(ValueError):
        build_skill_tools(skill, None)


def test_duplicate_tool_names_across_modules_raise(scratch_dir):
    skill_dir = make_skill_dir(scratch_dir, with_second=True)
    skill = skill_with_scripts(skill_dir)
    with pytest.raises(ValueError):
        build_skill_tools(skill, skill_dir)


def test_missing_declared_script_file_raises(scratch_dir):
    skill = ActiveSkill(name="x", description="x", version="1.0.0",
                        instructions="i",
                        resources={"scripts/nope.py": ""})
    with pytest.raises(FileNotFoundError):
        build_skill_tools(skill, scratch_dir)


def test_tool_registration_from_materialized_candidate_dir(scratch_dir):
    """验证候选包物化后,同一份脚本可作为工具加载(优化器新增脚本的场景)。"""
    from optimization.version_store import materialize_package

    target = scratch_dir / "candidate"
    materialize_package(
        {
            "metadata": {"name": "demo_skill", "version": "1.0.0"},
            "instructions": "Do stuff.",
            "resources": {"scripts/helper.py": HELPER},
        },
        target,
    )
    skill = ActiveSkill(
        name="demo_skill",
        description="demo",
        version="1.0.0-candidate",
        instructions="Do stuff.",
        resources={"scripts/helper.py": HELPER},
    )
    tools = build_skill_tools(skill, target)
    assert [tool.name for tool in tools] == ["count_sentences"]
