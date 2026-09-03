from pathlib import Path
from uuid import uuid4

import pytest

from optimization.version_store import (
    SkillVersionStore,
    apply_package_patch,
    bump_patch_version,
    materialize_package,
    parse_skill_file,
    scan_skill_resources,
    validate_resource_relpath,
)

SKILL_MD = """---
name: claim_decomposition
description: Decompose claims.
version: "1.0.0"
allowed_tools: []
---

# v1 instructions
Original procedure text.
"""

REFERENCE_MD = "# Reference rules\nCheck twice.\n"


@pytest.fixture()
def scratch_dir() -> Path:
    """仓库内唯一目录,避免 pytest 系统临时目录在沙箱里不可枚举。"""
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


def make_skill(scratch: Path, name: str = "claim_decomposition") -> Path:
    skill_dir = scratch / "skills" / name
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (skill_dir / "references" / "rules.md").write_text(REFERENCE_MD, encoding="utf-8")
    return scratch / "skills"


@pytest.fixture()
def store(scratch_dir) -> SkillVersionStore:
    return SkillVersionStore(make_skill(scratch_dir), scratch_dir / "skill_store")


def _candidate_package(store, instructions="# v2\nNew rules.") -> dict:
    package = store.load_active("claim_decomposition").package
    package["instructions"] = instructions
    return package


# ---- 解析 / 基础 ---------------------------------------------------------


def test_parse_skill_file():
    metadata, instructions = parse_skill_file(SKILL_MD)
    assert metadata["name"] == "claim_decomposition"
    assert "Original procedure" in instructions


def test_bump_patch_version():
    assert bump_patch_version("1.0.0") == "1.0.1"
    assert bump_patch_version("1.0.1-candidate") == "1.0.2"
    with pytest.raises(ValueError):
        bump_patch_version("abc")


def test_scan_resources_includes_declared_text_only(scratch_dir):
    skills_root = make_skill(scratch_dir)
    resources = scan_skill_resources(skills_root / "claim_decomposition")
    assert "references/rules.md" in resources
    assert "SKILL.md" not in resources
    assert resources["references/rules.md"] == REFERENCE_MD


# ---- active / promote / rollback -----------------------------------------


def test_active_defaults_to_baseline(store):
    active = store.load_active("claim_decomposition")
    assert active.is_baseline and active.store_seq == 0
    assert active.version == "1.0.0"
    assert "Original procedure" in active.instructions
    assert active.package["resources"]["references/rules.md"] == REFERENCE_MD


def test_promote_stores_whole_package_and_materializes(store):
    candidate = _candidate_package(store, "# v2 instructions\nNew rule.")
    promoted = store.promote(
        "claim_decomposition",
        candidate,
        change_reason="重复遗漏中心主张",
        metrics_before={"macro_f1": 0.5},
        metrics_after={"macro_f1": 0.6},
    )
    assert promoted["seq"] == 1
    assert promoted["version"] == "1.0.1"
    assert promoted["parent_seq"] == 0
    assert promoted["package"]["instructions"] == "# v2 instructions\nNew rule."
    assert promoted["package"]["resources"]["references/rules.md"] == REFERENCE_MD

    active = store.load_active("claim_decomposition")
    assert not active.is_baseline
    assert active.store_seq == 1
    assert "New rule" in active.instructions
    # 物化目录存在且含 SKILL.md 与资源(供 scripts 加载)。
    active_dir = store.materialize_active("claim_decomposition")
    assert (active_dir / "references" / "rules.md").is_file()
    assert "New rule" in (active_dir / "SKILL.md").read_text(encoding="utf-8")
    assert active.version in (active_dir / "SKILL.md").read_text(encoding="utf-8")


def test_rollback_to_history_and_baseline(store):
    store.promote("claim_decomposition", _candidate_package(store, "# v2\nSecond."),
                  change_reason="a")
    store.promote("claim_decomposition", _candidate_package(store, "# v3\nThird."),
                  change_reason="b")
    assert store.load_active("claim_decomposition").version == "1.0.2"

    rolled = store.rollback("claim_decomposition", 1)
    assert rolled.store_seq == 1 and "Second" in rolled.instructions

    baseline = store.rollback("claim_decomposition", 0)
    assert baseline.is_baseline and "Original procedure" in baseline.instructions


def test_rollback_unknown_seq_raises(store):
    with pytest.raises(ValueError):
        store.rollback("claim_decomposition", 42)


def test_promote_rejects_empty_instructions(store):
    package = store.load_active("claim_decomposition").package
    package["instructions"] = "   "
    with pytest.raises(ValueError):
        store.promote("claim_decomposition", package, change_reason="x")


def test_load_catalog_override_does_not_touch_active(store):
    store.promote("claim_decomposition", _candidate_package(store, "# v2\nReal change."),
                  change_reason="x")
    package = store.load_active("claim_decomposition").package
    package["instructions"] = "# candidate\nHypothetical text."
    catalog = store.load_catalog(
        ["claim_decomposition"], overrides={"claim_decomposition": package}
    )
    candidate = catalog["claim_decomposition"]
    assert "Hypothetical text" in candidate.instructions
    assert candidate.version == "1.0.1-candidate"

    active = store.load_active("claim_decomposition")
    assert "Real change" in active.instructions
    assert len(store.list_history("claim_decomposition")) == 1


# ---- apply_package_patch ---------------------------------------------------


def _package(store):
    return store.load_active("claim_decomposition").package


def test_apply_patch_updates_instructions_and_resources(store):
    patched = apply_package_patch(
        _package(store),
        {
            "instructions": "# v2\nUpdated body.",
            "resources": {
                "references/rules.md": "New reference text.",
                "assets/extra.md": "extra asset",
            },
        },
    )
    assert patched["instructions"] == "# v2\nUpdated body."
    assert patched["resources"]["references/rules.md"] == "New reference text."
    assert patched["resources"]["assets/extra.md"] == "extra asset"
    # 未涉及的资源保持不变。
    assert patched["metadata"] == _package(store)["metadata"]


def test_apply_patch_deletes_resource_with_null(store):
    patched = apply_package_patch(
        _package(store),
        {"resources": {"references/rules.md": None}},
    )
    assert "references/rules.md" not in patched["resources"]
    assert patched["instructions"] == _package(store)["instructions"]


def test_apply_patch_rejects_bad_paths(store):
    for bad in ("SKILL.md", "../outside.md", "/etc/passwd", "assets/x.exe"):
        with pytest.raises(ValueError):
            apply_package_patch(_package(store), {"resources": {bad: "x"}})


def test_apply_patch_rejects_non_py_under_scripts(store):
    with pytest.raises(ValueError):
        apply_package_patch(_package(store), {"resources": {"scripts/note.md": "x"}})


def test_apply_patch_rejects_empty_instructions(store):
    with pytest.raises(ValueError):
        apply_package_patch(_package(store), {"instructions": "  "})


def test_materialize_package_writes_all_files(scratch_dir):
    target = scratch_dir / "materialized"
    package = {
        "metadata": {"name": "demo", "version": "0.0.1"},
        "instructions": "# Demo",
        "resources": {"assets/a.txt": "AAA", "scripts/h.py": "def f():\n    pass\n"},
    }
    materialize_package(package, target)
    assert (target / "SKILL.md").is_file()
    assert (target / "assets" / "a.txt").read_text(encoding="utf-8") == "AAA"
    assert (target / "scripts" / "h.py").is_file()


def test_validate_resource_relpath():
    assert validate_resource_relpath("scripts/a.py") == "scripts/a.py"
    with pytest.raises(ValueError):
        validate_resource_relpath("SKILL.md")


def test_missing_baseline_raises(scratch_dir):
    store = SkillVersionStore(scratch_dir / "skills", scratch_dir / "skill_store")
    with pytest.raises(FileNotFoundError):
        store.load_active("does_not_exist")
