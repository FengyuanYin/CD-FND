import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import trainner
from config import Config
from optimization.validator import preflight_candidate_scripts
from optimization.version_store import SkillVersionStore
from trainner import Trainer

GOOD_SCRIPT = (
    "def count_sentences(text: str) -> int:\n"
    '    """Count sentences roughly."""\n'
    "    return 1\n"
)
BAD_SCRIPT = "def broken(:\n    return\n"


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


def _package(scripts: dict[str, str]) -> dict:
    return {
        "metadata": {"name": "claim_decomposition", "version": "1.0.0"},
        "instructions": "# Rules",
        "resources": dict(scripts),
    }


# ---- preflight_candidate_scripts ------------------------------------------


def test_preflight_passes_good_script(scratch_dir):
    errors = preflight_candidate_scripts(
        "claim_decomposition",
        _package({"scripts/helper.py": GOOD_SCRIPT}),
        scratch_dir,
    )
    assert errors is None


def test_preflight_skips_packages_without_scripts(scratch_dir):
    assert preflight_candidate_scripts("claim_decomposition", _package({}), scratch_dir) is None


def test_preflight_rejects_syntax_error(scratch_dir):
    errors = preflight_candidate_scripts(
        "claim_decomposition",
        _package({"scripts/bad.py": BAD_SCRIPT}),
        scratch_dir,
    )
    assert errors is not None
    assert any("语法错误" in error for error in errors)


def test_preflight_rejects_duplicate_tool_names(scratch_dir):
    second = (
        "def count_sentences(text: str) -> int:\n"
        '    """Count sentences (dup)."""\n'
        "    return 0\n"
    )
    errors = preflight_candidate_scripts(
        "claim_decomposition",
        _package({"scripts/a.py": GOOD_SCRIPT, "scripts/b.py": second}),
        scratch_dir,
    )
    assert errors is not None
    assert any("工具注册失败" in error for error in errors)


# ---- 门控优雅降级(trainner._run_promotion_gate) ---------------------------


def _make_skill(scratch_dir: Path) -> tuple[Path, Path]:
    skills_root = scratch_dir / "skills"
    (skills_root / "claim_decomposition").mkdir(parents=True, exist_ok=True)
    (skills_root / "claim_decomposition" / "SKILL.md").write_text(
        "---\nname: claim_decomposition\ndescription: d\nversion: \"1.0.0\"\n"
        "allowed_tools: []\n---\n\n# baseline\nOriginal rules.\n",
        encoding="utf-8",
    )
    return skills_root, scratch_dir / "skill_store"


def _make_trainer(skills_root: Path, store_root: Path) -> Trainer:
    config = Config(
        model_name="test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        epoch=1,
        dataset_name="weibo21",
        runs_dir=skills_root.parent / "outputs",
        skills_root=skills_root,
        skill_store_root=store_root,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.store = SkillVersionStore(skills_root, store_root)
    trainer.validation_items_by_domain = {"科技": [{"content": "x", "label": 0}]}
    trainer._runner_factory = lambda overrides=None: None
    trainer._runner = None
    return trainer


def test_gate_rejects_broken_script_before_evaluation(monkeypatch, scratch_dir):
    """坏脚本在 preflight 就被拒收,evaluate_candidate 不会被调用。"""
    async def _boom(*args, **kwargs):  # pragma: no cover - 不应被调用
        raise AssertionError("preflight 未拦截坏脚本")

    monkeypatch.setattr(trainner, "evaluate_candidate", _boom)
    skills_root, store_root = _make_skill(scratch_dir)
    trainer = _make_trainer(skills_root, store_root)

    result = asyncio.run(
        trainer._run_promotion_gate(
            target="claim_decomposition",
            patch_spec={"resources": {"scripts/bad.py": BAD_SCRIPT}},
            change_reason="x",
        )
    )
    assert result["accepted"] is False
    assert result["preflight_failed"] is True


def test_gate_tolerates_evaluation_crash(monkeypatch, scratch_dir):
    """端到端评测抛异常时优雅拒收,而不是让训练崩溃。"""
    async def _boom(*args, **kwargs):
        raise RuntimeError("candidate runner build failed")

    monkeypatch.setattr(trainner, "evaluate_candidate", _boom)
    skills_root, store_root = _make_skill(scratch_dir)
    trainer = _make_trainer(skills_root, store_root)

    result = asyncio.run(
        trainer._run_promotion_gate(
            target="claim_decomposition",
            patch_spec={"instructions": "# candidate\nNew rules."},
            change_reason="x",
        )
    )
    assert result["accepted"] is False
    assert "端到端评测失败" in result["reason"]
    assert "candidate runner build failed" in result["error"]
    # 仓库没有被污染:没有产生任何历史版本。
    assert trainer.store.list_history("claim_decomposition") == []
