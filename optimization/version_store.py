"""Skill 版本化管理(整包快照)。

设计(审计 §6 扩展):
- ``skills/`` 目录中的每个 Skill 是一个**包**:``SKILL.md``(frontmatter +
  正文指令)以及同目录下声明的文本资源(references/ assets/ templates/
  scripts/ 等)。目录内容即基线(seq=0),Optimizer 永不直接改写它;
- 被接受(晋升)的候选以**整包快照** {metadata, instructions, resources}
  追加到 ``skill_store_root/<name>/history.jsonl``,``active.json`` 指向当前
  active 版本;回滚 = 指回更早 seq(0 = 磁盘基线);
- 非基线 active 包可**物化**到 ``skill_store_root/<name>/active/``,供
  scripts 以真实文件形式被 importlib 加载(脚本即工具);
- frontmatter 元数据由代码维护(版本号自动 bump),优化器只能通过
  ``apply_package_patch`` 更新正文指令或包内文本资源,不能越权改元数据、
  不能写 SKILL.md、不能写二进制或越界路径。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)(-.*)?$")

# 允许纳入 Skill 包并支持版本化的文本资源后缀(scripts/ 下只允许 .py)。
TEXT_RESOURCE_SUFFIXES = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".py", ".csv", ".tsv",
    ".rst", ".html", ".css", ".js", ".toml", ".ini", ".cfg", ".conf",
}

MAX_RESOURCE_BYTES = 200_000  # 单个资源文件上限


@dataclass(frozen=True)
class ActiveSkill:
    """一次 prompt 注入/工具注册所需的 Skill 快照(不含文件系统句柄)。"""

    name: str
    description: str
    version: str
    instructions: str
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)
    is_baseline: bool = True
    store_seq: int = 0

    # ---- 派生物 ----------------------------------------------------------

    @property
    def package(self) -> dict[str, Any]:
        """整包快照(可直接 JSON 序列化与物化)。"""
        return {
            "metadata": dict(self.metadata or {}),
            "instructions": self.instructions,
            "resources": dict(self.resources or {}),
        }

    @property
    def scripts(self) -> list[str]:
        """包内 scripts/ 下的 .py 相对路径(升序,用于注册工具)。"""
        return sorted(
            rel for rel, _ in self.resources.items()
            if rel.startswith("scripts/") and rel.endswith(".py")
        )

    @property
    def injected_resources(self) -> dict[str, str]:
        """按约定会注入系统提示词的资源:references/ 与 templates/。"""
        return {
            rel: content for rel, content in self.resources.items()
            if rel.startswith("references/") or rel.startswith("templates/")
        }

    @classmethod
    def from_package(
        cls,
        name: str,
        package: dict[str, Any],
        *,
        version: str | None = None,
        is_baseline: bool = False,
        store_seq: int = 0,
    ) -> "ActiveSkill":
        metadata = dict(package.get("metadata") or {})
        return cls(
            name=name,
            description=str(metadata.get("description", "")).strip(),
            version=version or str(metadata.get("version", "0.0.1")).strip(),
            instructions=str(package.get("instructions", "")).strip(),
            allowed_tools=tuple(metadata.get("allowed_tools", []) or []),
            metadata=metadata,
            resources=dict(package.get("resources") or {}),
            is_baseline=is_baseline,
            store_seq=store_seq,
        )


# =========================================================================
# 解析 / 校验
# =========================================================================


def parse_skill_file(text: str) -> tuple[dict[str, Any], str]:
    """解析 SKILL.md:返回 (frontmatter 元数据, 正文指令)。

    与前端 SKILL.md 的字段口径保持一致;这里只负责解析,不做文件级路径校验。
    """
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    _, metadata_text, instructions = text.split("---", 2)
    metadata = yaml.safe_load(metadata_text)
    if not isinstance(metadata, dict):
        raise ValueError("Invalid Skill metadata")
    return metadata, instructions.strip()


def read_skill_file(path: Path) -> tuple[dict[str, Any], str]:
    """从磁盘读取并解析一个 SKILL.md 文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"Skill file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return parse_skill_file(text)


def bump_patch_version(version: str) -> str:
    """把 'x.y.z' 的 patch 位 +1(忽略候选后缀)。"""
    match = VERSION_PATTERN.match(version.strip())
    if not match:
        raise ValueError(f"无法识别的版本号: {version!r}")
    major, minor, patch = (int(part) for part in match.groups()[:3])
    return f"{major}.{minor}.{patch + 1}"


def validate_resource_relpath(rel: Any, *, allow_new: bool = True) -> str:
    """校验一个包内资源相对路径;返回规范化 posix 路径。"""
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("资源路径必须是非空字符串")
    rel = rel.strip().replace("\\", "/")
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"资源路径必须是包内相对路径: {rel!r}")
    if rel == "SKILL.md":
        raise ValueError("不允许通过资源更新 SKILL.md(元数据由代码维护)")
    if path.suffix.casefold() not in TEXT_RESOURCE_SUFFIXES:
        raise ValueError(f"不支持的资源文本后缀: {rel!r}")
    return rel


def validate_resource_content(rel: str, content: Any) -> str:
    """校验资源内容为不超过上限的 UTF-8 文本,返回规整后的字符串。"""
    if not isinstance(content, str):
        raise ValueError(f"资源内容必须是字符串: {rel!r}")
    if len(content.encode("utf-8")) > MAX_RESOURCE_BYTES:
        raise ValueError(
            f"资源超过大小上限({MAX_RESOURCE_BYTES} 字节): {rel!r}"
        )
    return content


def scan_skill_resources(skill_root: Path) -> dict[str, str]:
    """枚举 Skill 目录内除 SKILL.md 外的文本资源(SKILL.md 之外整包)。

    跳过隐藏路径与 __pycache__;遇到不可解码或超限文件直接报错,保证版本
    快照只包含干净的文本资源。
    """
    root = Path(skill_root).resolve()
    resources: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "SKILL.md":
            continue
        if path.suffix.casefold() not in TEXT_RESOURCE_SUFFIXES:
            raise ValueError(
                f"Skill 目录含不支持的非文本资源,无法版本化: {rel}"
            )
        if path.stat().st_size > MAX_RESOURCE_BYTES:
            raise ValueError(f"Skill 资源超过大小上限: {rel}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Skill 资源不是合法 UTF-8 文本: {rel}") from exc
        resources[rel] = content
    return resources


def apply_package_patch(
    package: dict[str, Any],
    patch_spec: dict[str, Any],
) -> dict[str, Any]:
    """把优化器的声明式补丁应用到整包快照上,返回新快照。

    patch_spec 形如::

        {"instructions": "完整新正文(可省略)",
         "resources": {"references/rules.md": "新文本",
                        "scripts/old.py": None}}   # None = 删除该文件

    校验规则:frontmatter 不可改;路径须包内相对、文本后缀、不超限;
    scripts/ 下只允许 .py。只做内存运算,不落盘。
    """
    if not isinstance(patch_spec, dict):
        raise ValueError("patch_spec 必须是 JSON 对象")

    new_instructions = package.get("instructions", "")
    if "instructions" in patch_spec:
        value = patch_spec["instructions"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("instructions 补丁必须是完整的非空正文")
        new_instructions = value.strip()

    resources = dict(package.get("resources") or {})
    resource_updates = patch_spec.get("resources")
    if resource_updates is not None:
        if not isinstance(resource_updates, dict):
            raise ValueError("patch_spec.resources 必须是 {路径: 内容|null}")
        for rel_raw, value in resource_updates.items():
            rel = validate_resource_relpath(rel_raw)
            if rel.startswith("scripts/") and not rel.endswith(".py"):
                raise ValueError(f"scripts/ 下只允许 .py 文件: {rel!r}")
            if value is None:
                resources.pop(rel, None)
                continue
            content = validate_resource_content(rel, value)
            resources[rel] = content

    return {
        "metadata": dict(package.get("metadata") or {}),
        "instructions": new_instructions,
        "resources": resources,
    }


# =========================================================================
# 版本仓库
# =========================================================================


class SkillVersionStore:
    """整包级版本化 Skill 仓库(见模块 docstring)。"""

    def __init__(self, skills_root: Path, store_root: Path) -> None:
        self.skills_root = Path(skills_root)
        self.store_root = Path(store_root)

    # ---- 内部磁盘访问 ---------------------------------------------------

    def _history_path(self, name: str) -> Path:
        return self.store_root / name / "history.jsonl"

    def _active_path(self, name: str) -> Path:
        return self.store_root / name / "active.json"

    def _active_dir(self, name: str) -> Path:
        return self.store_root / name / "active"

    def _read_history(self, name: str) -> list[dict[str, Any]]:
        path = self._history_path(name)
        if not path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _read_active(self, name: str) -> dict[str, Any] | None:
        path = self._active_path(name)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _write_active(self, name: str, record: dict[str, Any]) -> None:
        path = self._active_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def _append_history(self, name: str, record: dict[str, Any]) -> None:
        path = self._history_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str))
            stream.write("\n")

    # ---- 基线 / 整包读取 -------------------------------------------------

    def _baseline(self, name: str) -> ActiveSkill:
        """读取 skills/<name>/ 整包并返回基线快照(seq=0)。"""
        skill_dir = self.skills_root / name
        skill_file = skill_dir / "SKILL.md"
        metadata, instructions = read_skill_file(skill_file)
        if metadata.get("name") != name:
            raise ValueError(
                f"SKILL.md 中 name({metadata.get('name')!r}) 与目录名 {name!r} 不一致"
            )
        resources = scan_skill_resources(skill_dir)
        return ActiveSkill(
            name=name,
            description=str(metadata.get("description", "")).strip(),
            version=str(metadata.get("version", "0.0.1")).strip(),
            instructions=instructions,
            allowed_tools=tuple(metadata.get("allowed_tools", []) or []),
            metadata=metadata,
            resources=resources,
            is_baseline=True,
            store_seq=0,
        )

    def load_active(self, name: str) -> ActiveSkill:
        """返回当前 active 快照;未晋升过则等于磁盘基线。"""
        baseline = self._baseline(name)
        active = self._read_active(name)
        if active is None or int(active.get("seq", 0)) == 0:
            return baseline
        seq = int(active["seq"])
        for entry in self._read_history(name):
            if int(entry["seq"]) == seq:
                package = entry["package"]
                return ActiveSkill.from_package(
                    name,
                    package,
                    version=str(entry.get("version", baseline.version)),
                    is_baseline=False,
                    store_seq=seq,
                )
        raise ValueError(
            f"Skill {name}: active 指向的版本 {seq} 在 history 中不存在,仓库损坏"
        )

    def load_catalog(
        self,
        names: list[str],
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, ActiveSkill]:
        """一次取多个 Skill 的 active 快照,key 为 skill 名。

        ``overrides`` 用于“只读候选验证”:对给定 Skill 整体替换为候选整包
        (版本标为 ``<当前版本>-candidate``),不写入 active,验证失败即丢弃,
        不会污染正式版本。
        """
        catalog: dict[str, ActiveSkill] = {}
        for name in names:
            if overrides and name in overrides:
                active = self.load_active(name)
                candidate_package = overrides[name]
                if not isinstance(candidate_package, dict):
                    raise ValueError(f"override 必须是整包快照 dict: {name}")
                catalog[name] = ActiveSkill.from_package(
                    name,
                    candidate_package,
                    version=f"{active.version}-candidate",
                    is_baseline=False,
                    store_seq=active.store_seq,
                )
            else:
                catalog[name] = self.load_active(name)
        return catalog

    # ---- 晋升 / 回滚 -----------------------------------------------------

    def promote(
        self,
        name: str,
        package: dict[str, Any],
        *,
        change_reason: str,
        metrics_before: dict[str, Any] | None = None,
        metrics_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """接受一个候选整包:校验后追加 history、更新 active 并物化。

        返回写入的 history 记录。frontmatter 不允许在此被优化器改写:
        元数据保持基线/前代版本,版本号由代码自动 bump。
        """
        if not isinstance(package, dict):
            raise ValueError(f"promote 需要整包快照 dict: {name}")
        instructions = str(package.get("instructions", "")).strip()
        if not instructions:
            raise ValueError(f"promote 拒绝空指令: {name}")

        previous = self.load_active(name)
        history = self._read_history(name)
        next_seq = (history[-1]["seq"] + 1) if history else 1
        version = bump_patch_version(previous.version)

        # 保留前代元数据,仅替换正文与资源;版本号由代码维护并写回元数据,
        # 保证物化后的 SKILL.md 与 active.json 版本一致。
        metadata = dict(previous.metadata or {})
        metadata["version"] = version
        clean_package = {
            "metadata": metadata,
            "instructions": instructions,
            "resources": dict(package.get("resources") or {}),
        }
        record = {
            "name": name,
            "seq": next_seq,
            "version": version,
            "parent_seq": previous.store_seq,
            "parent_version": previous.version,
            "change_reason": change_reason,
            "metrics_before": metrics_before or {},
            "metrics_after": metrics_after or {},
            "accepted": True,
            "package": clean_package,
        }
        self._append_history(name, record)
        self._write_active(name, {"name": name, "seq": next_seq, "version": version})
        materialize_package(clean_package, self._active_dir(name))
        return record

    def rollback(self, name: str, seq: int) -> ActiveSkill:
        """把 active 指回历史版本 seq(0 = 磁盘基线)并物化/清理目录。"""
        if seq == 0:
            baseline = self._baseline(name)
            self._write_active(
                name, {"name": name, "seq": 0, "version": baseline.version}
            )
            # 物化目录里的旧文件不再使用,删除以免误导入(尽力而为)。
            active_dir = self._active_dir(name)
            if active_dir.is_dir():
                try:
                    shutil.rmtree(active_dir)
                except OSError:
                    pass
            return baseline
        history = self._read_history(name)
        match = next((e for e in history if int(e["seq"]) == seq), None)
        if match is None:
            raise ValueError(f"Skill {name}: 历史中不存在版本 seq={seq}")
        self._write_active(name, {"name": name, "seq": seq, "version": match["version"]})
        materialize_package(match["package"], self._active_dir(name))
        return self.load_active(name)

    def reset_to_baseline(self, name: str) -> ActiveSkill:
        """科研便利:丢弃该 Skill 的所有候选并回到基线。"""
        return self.rollback(name, 0)

    def list_history(self, name: str) -> list[dict[str, Any]]:
        """返回历史记录(含整包,可直接用于比对与回滚)。"""
        return self._read_history(name)

    # ---- 磁盘物化 -------------------------------------------------------

    def materialize_active(self, name: str, target_dir: Path | None = None) -> Path:
        """把当前 active 整包物化到磁盘并返回目录。

        基线包的文件本来就在 ``skills_root/<name>``;非基线包写入
        ``store_root/<name>/active``(或指定的 target_dir),供 scripts
        以真实文件加载。
        """
        active = self.load_active(name)
        if active.is_baseline and target_dir is None:
            return self.skills_root / name
        target = Path(target_dir) if target_dir is not None else self._active_dir(name)
        materialize_package(active.package, target)
        return target

    def clear_store(self) -> None:
        """删除整个 store(谨慎使用,用于重建实验)。"""
        if self.store_root.is_dir():
            shutil.rmtree(self.store_root)


def materialize_package(package: dict[str, Any], target_dir: Path) -> None:
    """把整包快照按文件写出到 target_dir(SKILL.md + 全部文本资源)。

    写入是幂等的;目标目录内的旧文件不会被清空,但同名文件会被覆盖。
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata = package.get("metadata") or {}
    instructions = str(package.get("instructions", "")).strip()
    frontmatter_text = yaml.safe_dump(
        metadata, allow_unicode=True, sort_keys=False
    ).rstrip()
    (target / "SKILL.md").write_text(
        f"---\n{frontmatter_text}\n---\n\n{instructions}\n",
        encoding="utf-8",
    )
    for rel, content in (package.get("resources") or {}).items():
        path = (target / rel).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
