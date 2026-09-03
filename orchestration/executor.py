"""逐样本执行器:Coordinator → Specialist → 聚合 → Judge 的完整推理路径。

InferenceRunner 持有 Skill 版本仓库与各 Agent 实例;一次 ``infer_one`` 完成
单样本的完整前向推理并把每层输出(含解析失败的原文)原样放进记录,供指标
计算与错误归因使用。训练、候选验证、最终测试三阶段必须共用本执行器。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.coordinator import CoordinatorAgent
from agents.judge import JudgeAgent
from agents.specialist import SpecialistAgent
from optimization.version_store import (
    ActiveSkill,
    SkillVersionStore,
    materialize_package,
)
from orchestration.aggregator import build_analysis_report
from orchestration.prompt_builder import (
    ALL_SKILL_NAMES,
    JUDGE_SKILL_NAME,
    ROUTING_SKILL_NAME,
    SPECIALIST_SKILL_NAMES,
)
from orchestration.router import normalize_routing_report, selected_skill_names
from output_parsing import parse_json_content

# 序列化给模型时永远剔除的标签字段(防标签泄漏)。
LABEL_KEYS = {"label", "gold_label", "gold", "target", "answer", "expected_result", "expected"}


def visible_item(item: dict[str, Any]) -> dict[str, Any]:
    """剔除样本中的标签字段后返回可见输入(递归处理嵌套的 item 包装)。"""
    cleaned: dict[str, Any] = {}
    for key, value in item.items():
        if key.casefold() in LABEL_KEYS:
            continue
        if isinstance(value, dict) and "content" in value:
            cleaned[key] = visible_item(value)
        else:
            cleaned[key] = value
    return cleaned


def dataset_format_label(config_name: str) -> str:
    """把数据集配置名映射为协议里的 dataset_format 值。"""
    name = str(config_name).casefold()
    return "weibo21" if name in {"weibo", "weibo21"} else name


class InferenceRunner:
    """一次构建、逐样本推理的执行器。Skill 变更后调用 refresh()。

    catalog_overrides: {skill_name: 候选整包快照 dict},用于“只读候选验证”。
    """

    def __init__(
        self,
        config: Any,
        catalog_overrides: dict[str, dict] | None = None,
    ) -> None:
        self.config = config
        self.label_schema = config.resolved_label_schema
        self.dataset_format = dataset_format_label(config.name)
        self.store = SkillVersionStore(
            Path(config.skills_root), Path(config.skill_store_root)
        )
        self.catalog_overrides = dict(catalog_overrides or {})
        self.coordinator: CoordinatorAgent | None = None
        self.specialists: dict[str, SpecialistAgent] = {}
        self.judge: JudgeAgent | None = None
        self._build_agents()

    # ---- Agent 生命周期 -------------------------------------------------

    def _catalog(self) -> dict[str, ActiveSkill]:
        required = [ROUTING_SKILL_NAME, JUDGE_SKILL_NAME] + list(SPECIALIST_SKILL_NAMES)
        missing = [name for name in required if not (Path(self.config.skills_root) / name / "SKILL.md").is_file()]
        if missing:
            raise FileNotFoundError(
                "缺少必需的 SKILL.md 文件(请先补齐 skills/ 目录): "
                + ", ".join(f"skills/{name}/SKILL.md" for name in missing)
            )
        return self.store.load_catalog(list(ALL_SKILL_NAMES), overrides=self.catalog_overrides)

    def _tools_dir_for(self, name: str, skill: ActiveSkill) -> Path | None:
        """Skill 声明了 scripts/ 时才物化/定位其目录(用于注册工具)。"""
        if not skill.scripts:
            return None
        if name in self.catalog_overrides:
            target = self.store.store_root / name / "candidate"
            materialize_package(self.catalog_overrides[name], target)
            return target
        return self.store.materialize_active(name)

    def _construct_agents(self) -> tuple[CoordinatorAgent, dict[str, SpecialistAgent], JudgeAgent]:
        """构建全套新 Agent;失败时抛出且不触碰当前实例的状态。"""
        catalog = self._catalog()
        coordinator = CoordinatorAgent(
            self.config,
            catalog,
            tools_dir=self._tools_dir_for(ROUTING_SKILL_NAME, catalog[ROUTING_SKILL_NAME]),
        )
        specialists = {
            name: SpecialistAgent(
                self.config,
                catalog[name],
                tools_dir=self._tools_dir_for(name, catalog[name]),
            )
            for name in SPECIALIST_SKILL_NAMES
        }
        judge = JudgeAgent(
            self.config,
            catalog[JUDGE_SKILL_NAME],
            tools_dir=self._tools_dir_for(JUDGE_SKILL_NAME, catalog[JUDGE_SKILL_NAME]),
        )
        return coordinator, specialists, judge

    def _build_agents(self) -> None:
        coordinator, specialists, judge = self._construct_agents()
        self.coordinator = coordinator
        self.specialists = specialists
        self.judge = judge

    async def refresh(self, catalog_overrides: dict[str, dict] | None = None) -> None:
        """按新覆盖重建 Agent;先构建成功再替换旧实例(失败保留旧链路可用)。"""
        if catalog_overrides is not None:
            self.catalog_overrides = dict(catalog_overrides)
        coordinator, specialists, judge = self._construct_agents()
        old = (self.coordinator, self.specialists, self.judge)
        self.coordinator, self.specialists, self.judge = coordinator, specialists, judge
        for agent in [old[0], old[2]] + list(old[1].values()):
            if agent is None:
                continue
            try:
                await agent.close()
            except Exception:  # 关闭旧 Agent 失败不应阻断(新 Agent 已就位)
                pass

    async def close(self) -> None:
        for agent in [self.coordinator, self.judge] + list(self.specialists.values()):
            if agent is not None:
                try:
                    await agent.close()
                except Exception:  # 关闭失败不应阻断后续清理
                    pass
        self.coordinator = None
        self.specialists = {}
        self.judge = None

    # ---- 单样本推理 -----------------------------------------------------

    async def infer_one(self, item: dict[str, Any]) -> dict[str, Any]:
        """推理一个样本并返回完整记录(可用于 trace 与指标)。"""
        item_id = item.get("id") or item.get("item_id")
        task_base = {
            "dataset_format": self.dataset_format,
            "item": visible_item(item),
        }
        errors: list[str] = []

        # 1) Coordinator 路由。
        assert self.coordinator is not None
        routing_raw_text = await self.coordinator.run(
            json.dumps(task_base, ensure_ascii=False, default=str)
        )
        routing_raw = parse_json_content(routing_raw_text)
        routing = normalize_routing_report(routing_raw, item_id=item_id)
        if routing["routing_decision"]["parse_error"]:
            errors.append("coordinator: routing output unparseable")

        # 2) 选定 Specialist 逐个分析(顺序执行,便于复现与成本核算)。
        wrapped_reports: list[dict[str, Any]] = []
        for skill_name in selected_skill_names(routing):
            specialist = self.specialists.get(skill_name)
            if specialist is None:
                continue
            specialist_task = dict(task_base)
            specialist_task["skill_name"] = skill_name
            try:
                text = await specialist.run(
                    json.dumps(specialist_task, ensure_ascii=False, default=str)
                )
            except Exception as exc:  # 单个 Specialist 失败不应中断整条链路
                wrapped_reports.append(
                    {
                        "skill_name": skill_name,
                        "skill_version": specialist.skill.version,
                        "report": None,
                        "parse_error": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                errors.append(f"specialist {skill_name}: run failed")
                continue
            parsed = parse_json_content(text)
            if parsed is None:
                wrapped_reports.append(
                    {
                        "skill_name": skill_name,
                        "skill_version": specialist.skill.version,
                        "report": None,
                        "parse_error": True,
                        "error": "specialist_report_v1 解析失败",
                    }
                )
                errors.append(f"specialist {skill_name}: output unparseable")
            else:
                wrapped_reports.append(
                    {
                        "skill_name": skill_name,
                        "skill_version": specialist.skill.version,
                        "report": parsed,
                        "parse_error": False,
                        "error": None,
                    }
                )

        # 3) 聚合为 analysis_report_v2。
        analysis_report = build_analysis_report(routing, wrapped_reports)

        # 4) Judge 输出最终决策。
        assert self.judge is not None
        judge_input = {
            "dataset_format": self.dataset_format,
            "source_item": visible_item(item),
            "analysis_report": analysis_report,
            "label_schema": self.label_schema,
        }
        judge_raw_text = await self.judge.run(
            json.dumps(judge_input, ensure_ascii=False, default=str)
        )
        judge_decision = parse_json_content(judge_raw_text)
        if judge_decision is None:
            errors.append("judge: judge_decision_v2 解析失败")

        return {
            "item": visible_item(item),
            "item_id": item_id,
            "dataset_format": self.dataset_format,
            "routing_report": routing,
            "routing_raw_text": routing_raw_text,
            "specialist_reports": wrapped_reports,
            "analysis_report": analysis_report,
            "judge_decision": judge_decision,
            "judge_raw_text": judge_raw_text,
            "errors": errors,
        }
