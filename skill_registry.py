from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
import yaml
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: str
    allowed_tools: tuple[str, ...]
    instructions: str

    root_dir: Path
    source_path: Path

    references: tuple[Path, ...]
    scripts: tuple[Path, ...]
    assets: tuple[Path, ...]
    templates: tuple[Path, ...]

class SkillRegistry:
    def __init__(self, root: Path, available_tool_name: set[str]):
        self.root = root.resolve()
        self.available_tool_name = available_tool_name
        self._skills: dict[str, Skill] = {}

    def discover(self) -> None:
        self._skills.clear()

        for path in self.root.glob("*/SKILL.md"):
            print(f"Discovered skill: {path}")
            skill = self._read_skill(path)

            if skill.name in self._skills:
                raise ValueError(f"Duplicate skill name: {skill.name}")

            unknown_tools = (
                set(skill.allowed_tools) - self.available_tool_name
            )

            if unknown_tools:
                raise ValueError(
                    f"Skill {skill.name} requests unknown tools: "
                    f"{sorted(unknown_tools)}"
                )

            self._skills[skill.name] = skill

    def catalog(self) -> list[dict[str, Any]]:
        """Return a list of all registered skills in a brief way."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.version
            } for skill in self._skills.values()
        ]

    def load(self, name: str) -> Skill:
        """Load a skill by name."""
        try:
            return self._skills[name]
        except KeyError as exc:
            raise ValueError(f"Skill {name} not found") from exc

    def _read_skill(self, path: Path) -> Skill:
        """Read and validate one complete Skill package."""
        try:
            resolved = path.resolve(strict = True)
        except FileNotFoundError:
            raise ValueError(f"Skill file not found: {path}")

        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"Skill file is outside registry root: {path}"
            ) from exc

        if not resolved.is_file():
            raise ValueError(f"Skill file is not a file: {path}")

        if resolved.name != "SKILL.md":
            raise ValueError(f"Skill file must be named SKILL.md: {path}")

        max_skill_file_size = 256 * 1024  # 256 KB
        if resolved.stat().st_size > max_skill_file_size:
            raise ValueError(f"Skill file is too large: {path}")

        try:
            text = resolved.read_text(encoding="utf-8")

        except UnicodeDecodeError as exc:
            raise ValueError(f"Skill file is not valid UTF-8: {path}") from exc

        metadata, instruction = self._parse_frontmatter(text)

        skill_root = resolved.parent.resolve()# 每个 Skill 的根目录就是 SKILL.md 所在目录。

        name = metadata.get("name")
        if (
        not isinstance(name, str)
        
        ):
            raise ValueError(
                "Skill name must start with a lowercase letter and "
                "contain only lowercase letters, digits, or underscores: "
                f"{name!r}"
            )

        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Skill {name} must have a non-empty description"
            )

        version = metadata.get("version", "0.0.1")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(
                f"Skill {name} version must be a non-empty string"
            )

        allowed_tools = metadata.get("allowed_tools", [])
        if not isinstance(allowed_tools, list):
            raise ValueError(
                f"Skill {name} allowed_tools must be a list"
            )

        normalized_tools: list[str] = []

        for index, tool_name in enumerate(allowed_tools):
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(
                    f"Skill {name} allowed_tools[{index}] "
                    "must be a non-empty string"
                )
            normalized_tools.append(tool_name.strip())
        normalized_tools = list(dict.fromkeys(normalized_tools))

        instructions = instruction.strip()
        if not instructions:
            raise ValueError(
                f"Skill {name} must contain instructions after frontmatter"
            )

        references = self._resolve_resources(
        skill_root,
        metadata.get("references", []),
        field_name="references",
        )

        scripts = self._resolve_resources(
            skill_root,
            metadata.get("scripts", []),
            field_name="scripts",
        )

        assets = self._resolve_resources(
            skill_root,
            metadata.get("assets", []),
            field_name="assets",
        )

        templates = self._resolve_resources(
            skill_root,
            metadata.get("templates", []),
            field_name="templates",
        )

        return Skill(
            name=name,
            description=description.strip(),
            version=version.strip(),
            allowed_tools=tuple(normalized_tools),
            instructions=instructions,
            root_dir=skill_root,
            source_path=resolved,
            references=references,
            scripts=scripts,
            assets=assets,
            templates=templates,
        )


    def _resolve_resources(
        self,
        skill_root: Path,
        values: Any,
        *,
        field_name: str,
    ) -> tuple[Path, ...]:
        """Resolve and validate declared files inside one Skill package."""

        if values is None:
            return ()   

        if not isinstance(values, list):
            raise ValueError(
                f"Skill {field_name} must be a list of relative paths"
            )

        resources: list[Path] = []
        seen: set[Path] = set()

        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Skill {field_name}[{index}] must be a non-empty string"
                )

            relative_path = Path(value.strip())
            if relative_path.is_absolute():
                raise ValueError(
                    f"{field_name}[{index}] must be relative: {value}"
                )

            # resolved_path = (skill_root / relative_path).resolve()

            try:
                target = (skill_root / relative_path).resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Declared {field_name} resource does not exist: "
                    f"{value}"
                ) from exc

            try:
                target.relative_to(skill_root)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} resource is outside Skill directory: "
                    f"{value}"
                ) from exc

            if not target.is_file():
                raise ValueError(
                    f"{field_name} resource is not a file: {value}"
                )   
            if target in seen:
                raise ValueError(
                    f"Duplicate {field_name} resource: {value}"
                )

            seen.add(target)
            resources.append(target)
        return tuple(resources)


    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        """Parse the frontmatter from a SKILL.md file."""
        if not text.startswith("---\n"):
            raise ValueError("SKILL.md must start with YAML frontmatter")

        _, metadata_text, instructions = text.split("---", 2)
        metadata = yaml.safe_load(metadata_text)

        if not isinstance(metadata, dict):
            raise ValueError("Invalid Skill metadata")

        for field in ("name", "description"):
            if not metadata.get(field):
                raise ValueError(f"Missing required field: {field}")

        return metadata, instructions

    def list_resources(
        self,
        skill_name: str,
    ) -> dict[str, list[str]]:
        skill = self.load(skill_name)

        def relative(paths: tuple[Path, ...]) -> list[str]:
            return [
                path.relative_to(skill.root_dir).as_posix()
                for path in paths
            ]

        return {
            "references": relative(skill.references),
            "scripts": relative(skill.scripts),
            "assets": relative(skill.assets),
            "templates": relative(skill.templates),
        }


    def read_resource(self, skill_name: str, relative_path: str) -> str:
        skill = self.load(skill_name)
        target = (skill.root_dir / relative_path).resolve()

        target.relative_to(skill.root_dir)

        allowed = (
            *skill.references,
            *skill.scripts,
            *skill.assets,
            *skill.templates,
        )

        if target not in allowed:
            raise ValueError(f"Resource {relative_path} is not allowed for skill {skill_name}")

        if target.suffix not in {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh", ".html", ".css", ".js", ".csv", ".tsv", ".xml", ".ini", ".conf", ".cfg", ".toml", ".rst", ".log", ".bat", ".ps1", ".rb", ".pl", ".php", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".swift", ".kt", ".m", ".mm", ".r", ".jl", ".lua", ".dart", ".hs", ".erl", ".ex", ".exs", ".clj", ".cljs", ".groovy", ".scala", ".sql", ".mdown", ".markdown", ".adoc", ".asciidoc", ".tex", ".bib", ".sty", ".cls", ".dtx", ".ins", ".ltx", ".ltxdoc", ".ltxstyle", ".ltxclass", ".ltxpackage", ".ltxtemplate", ".ltxconfig", ".ltxoptions", ".ltxcommands", ".ltxmacros"}:
            raise ValueError("Resource is not a supported text format")

        return target.read_text(encoding="utf-8")



if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    passed = 0

    def check(condition: bool, label: str) -> None:
        """Record one successful assertion or fail with a useful label."""
        global passed
        if not condition:
            raise AssertionError(label)
        passed += 1
        print(f"[PASS] {label}")

    def expect_error(
        error_type: type[Exception],
        action: Any,
        label: str,
    ) -> None:
        """Assert that an operation raises the expected exception type."""
        global passed
        try:
            action()
        except error_type:
            passed += 1
            print(f"[PASS] {label}")
        else:
            raise AssertionError(
                f"{label}: expected {error_type.__name__}"
            )

    with TemporaryDirectory(prefix="skill_registry_test_") as temporary:
        registry_root = Path(temporary).resolve()
        skill_root = registry_root / "sample_skill"
        references_dir = skill_root / "references"
        scripts_dir = skill_root / "scripts"
        assets_dir = skill_root / "assets"
        templates_dir = skill_root / "templates"

        for directory in (
            references_dir,
            scripts_dir,
            assets_dir,
            templates_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        reference = references_dir / "rules.md"
        script = scripts_dir / "check.py"
        asset = assets_dir / "schema.json"
        template = templates_dir / "report.json"
        outside = registry_root / "outside.txt"

        reference.write_text("Evidence rules", encoding="utf-8")
        script.write_text("print('checked')", encoding="utf-8")
        asset.write_text('{"type": "object"}', encoding="utf-8")
        template.write_text('{"result": null}', encoding="utf-8")
        outside.write_text("outside skill", encoding="utf-8")

        skill_file = skill_root / "SKILL.md"
        skill_file.write_text(
            """---
name: sample_skill
description: A complete test skill.
version: "1.2.3"
allowed_tools:
  - read_file
  - read_file
references:
  - references/rules.md
scripts:
  - scripts/check.py
assets:
  - assets/schema.json
templates:
  - templates/report.json
---

# Sample Skill

Follow the declared test procedure.
""",
            encoding="utf-8",
        )

        registry = SkillRegistry(registry_root, {"read_file", "search_text"})

        # _parse_frontmatter
        metadata, instructions = registry._parse_frontmatter(
            skill_file.read_text(encoding="utf-8")
        )
        check(metadata["name"] == "sample_skill", "parse frontmatter metadata")
        check("Sample Skill" in instructions, "parse frontmatter instructions")

        # _resolve_resources
        resolved_references = registry._resolve_resources(
            skill_root,
            ["references/rules.md"],
            field_name="references",
        )
        check(resolved_references == (reference.resolve(),), "resolve declared resource")

        # _read_skill
        directly_read = registry._read_skill(skill_file)
        check(directly_read.name == "sample_skill", "read complete Skill package")
        check(directly_read.allowed_tools == ("read_file",), "deduplicate allowed tools")
        check(directly_read.references == (reference.resolve(),), "attach Skill references")
        check(directly_read.scripts == (script.resolve(),), "attach Skill scripts")
        check(directly_read.assets == (asset.resolve(),), "attach Skill assets")
        check(directly_read.templates == (template.resolve(),), "attach Skill templates")

        # discover, catalog, and load
        registry.discover()
        check(len(registry.catalog()) == 1, "discover and catalog one Skill")
        check(registry.catalog()[0]["version"] == "1.2.3", "catalog Skill version")
        check(registry.load("sample_skill").source_path == skill_file.resolve(), "load Skill by name")

        # list_resources
        listed = registry.list_resources("sample_skill")
        check(listed["references"] == ["references/rules.md"], "list reference paths")
        check(listed["scripts"] == ["scripts/check.py"], "list script paths")
        check(listed["assets"] == ["assets/schema.json"], "list asset paths")
        check(listed["templates"] == ["templates/report.json"], "list template paths")

        # read_resource
        check(
            registry.read_resource("sample_skill", "references/rules.md") == "Evidence rules",
            "read declared reference",
        )
        check(
            registry.read_resource("sample_skill", "assets/schema.json") == '{"type": "object"}',
            "read declared text asset",
        )
        check(
            registry.read_resource("sample_skill", "templates/report.json") == '{"result": null}',
            "read declared template",
        )

        # Expected validation failures.
        expect_error(
            ValueError,
            lambda: registry.load("missing_skill"),
            "reject unknown Skill",
        )
        expect_error(
            ValueError,
            lambda: registry.read_resource("sample_skill", "SKILL.md"),
            "reject undeclared resource",
        )
        check(
            registry.read_resource("sample_skill", "scripts/check.py")
            == "print('checked')",
            "read declared script as text resource",
        )
        expect_error(
            ValueError,
            lambda: registry._resolve_resources(
                skill_root,
                ["../outside.txt"],
                field_name="references",
            ),
            "reject resource outside Skill root",
        )
        expect_error(
            ValueError,
            lambda: registry._resolve_resources(
                skill_root,
                ["references/rules.md", "references/rules.md"],
                field_name="references",
            ),
            "reject duplicate resources",
        )
        expect_error(
            ValueError,
            lambda: registry._resolve_resources(
                skill_root,
                str(reference.resolve()),
                field_name="references",
            ),
            "reject non-list resource declaration",
        )
        expect_error(
            ValueError,
            lambda: registry._parse_frontmatter("# Missing frontmatter"),
            "reject missing frontmatter",
        )

        # discover must reject tools unavailable to this Registry.
        restricted_registry = SkillRegistry(registry_root, set())
        expect_error(
            ValueError,
            restricted_registry.discover,
            "reject unavailable allowed_tools during discovery",
        )

    print(f"SkillRegistry self-test passed: {passed} checks")
