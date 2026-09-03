"""训练/验证闭环(审计 8 节阶段 1~3;第 1.2 节两条运行路径)。

数据流(训练路径):
    训练域样本
      → InferenceRunner(Coordinator 路由 → 固定 Specialist → 聚合 → Judge)
      → 预测记录(完整 trace,含每层原始输出)
      → Gold Label 只在预测完成后参与指标计算
      → 指标按批次聚合
      → optimization_mode:
           "off"     : 基线,不做任何优化调用;
           "suggest" : Optimizer 只出诊断与候选建议(不改文件);
           "apply"   : 候选经独立验证集门控,通过才 promote 到 version store,
                       下一批次自动加载新 Skill;失败 reject 保持旧版本。

推理路径(run_test_evaluation):冻结的测试域 + 当前 active Skill,全程不调用
Optimizer、不读取 Gold Label 之外的任何东西、不修改 Skill。
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm

from datasets import dataset as dataset_mod
from evaluation.metrics import (
    canonical_gold,
    classification_metrics,
    extract_pairs,
    reverse_label_mapping,
)
from optimization.validator import (
    evaluate_candidate,
    pick_validation_sample,
    preflight_candidate_scripts,
)
from optimization.version_store import SkillVersionStore, apply_package_patch
from orchestration.executor import InferenceRunner, visible_item
from orchestration.prompt_builder import ALL_SKILL_NAMES
from output_parsing import parse_json_content


def _trace_dir_path(config: Any) -> Path:
    run_id = getattr(config, "run_id", None) or datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.runs_dir) / f"{config.name}-{run_id}"


def _parse_patch_spec(patch: str) -> dict[str, Any]:
    """把 Optimizer 的 patch 解析为声明式补丁 spec。

    兼容两种形态:
    - 纯正文:整段视为对 instructions 的完整替换;
    - JSON 对象字符串:{"instructions": ...?, "resources": {路径: 内容|null}}。
    """
    text = patch.strip()
    parsed = parse_json_content(text)
    if isinstance(parsed, dict):
        return parsed
    return {"instructions": text}


def _optimizer_case(record: dict[str, Any]) -> dict[str, Any]:
    """把一条推理记录压成 Optimizer 诊断用的可序列化 case(控制 token 量)。"""
    routing = record.get("routing_report", {})
    decision = routing.get("routing_decision", {})
    analysis = record.get("analysis_report", {})
    judge = record.get("judge_decision") or {}
    prediction = judge.get("prediction") or {}
    safe_item = record.get("item")
    if isinstance(safe_item, dict):
        safe_item = visible_item(safe_item)  # 二次保险:Optimizer 也绝不接触标签字段
    return {
        "item": safe_item,
        "selected_skills": [e.get("skill_name") for e in decision.get("selected_skills", [])],
        "fallback_used": decision.get("fallback_used", False),
        "routing_features": routing.get("routing_features", {}),
        "specialist_statuses": [
            {"skill": s.get("name"), "status": s.get("status")}
            for s in analysis.get("specialists", [])
        ],
        "claims_summary": {
            "total": len(analysis.get("claims", [])),
            "supported": sum(1 for c in analysis.get("claims", []) if c.get("assessment") == "SUPPORTED"),
            "refuted": sum(1 for c in analysis.get("claims", []) if c.get("assessment") == "REFUTED"),
            "unresolved": len(analysis.get("judge_handoff", {}).get("unresolved_claim_ids", [])),
        },
        "conflicts": analysis.get("conflicts", []),
        "judge_decision": {
            "canonical_label": prediction.get("canonical_label"),
            "dataset_label": prediction.get("dataset_label"),
            "status": prediction.get("status"),
            "confidence": prediction.get("confidence"),
            "rationale": judge.get("rationale"),
            "uncertainty": judge.get("uncertainty"),
        },
        "gold_canonical_label": record.get("gold_canonical"),
        "errors": record.get("errors", []),
    }


class Trainer:
    """训练闭环入口:返回批次级日志列表(每条含指标与优化决策)。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.store = SkillVersionStore(
            Path(config.skills_root), Path(config.skill_store_root)
        )
        self.dataset = dataset_mod.load_dataset(config)
        self._ensure_domains(config.train_domains, role="train_domains")
        grouped = {domain: self.dataset.data[domain] for domain in config.train_domains}
        self.train_items_by_domain, self.validation_items_by_domain = dataset_mod.split_domains(
            grouped, seed=config.seed, validation_ratio=config.validation_ratio
        )

        self.run_dir = _trace_dir_path(config)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "predictions.jsonl"
        self.log_path = self.run_dir / "batches.jsonl"

        self._runner: InferenceRunner | None = None
        self._runner_factory: Callable[[dict | None], InferenceRunner] | None = None

    def _ensure_domains(self, domains: list[str], *, role: str) -> None:
        missing = [name for name in domains if name not in self.dataset.data]
        if missing:
            raise ValueError(
                f"{role} 中有数据集不存在的域: {missing};"
                f"数据集可用域: {sorted(self.dataset.data)}"
            )

    # ---- Agent 构造(便于测试替身) ---------------------------------------

    def make_runner(self, catalog_overrides: dict[str, str] | None = None) -> InferenceRunner:
        return InferenceRunner(self.config, catalog_overrides=catalog_overrides)

    # ---- 批次日志写入 ---------------------------------------------------

    def _write_log(self, entry: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str))
            stream.write("\n")

    def _write_trace(self, record: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str))
            stream.write("\n")

    # ---- 单批次收尾 -----------------------------------------------------

    async def _finish_batch(
        self,
        *,
        epoch: int,
        domain: str,
        batch_index: int,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        label_schema = self.config.resolved_label_schema
        metrics = classification_metrics(extract_pairs(batch, label_schema))
        entry: dict[str, Any] = {
            "epoch": epoch,
            "category": domain,
            "batch_index": batch_index,
            "batch_size": len(batch),
            "metrics": metrics,
            "optimization": {"mode": self.config.optimization_mode},
        }

        mode = self.config.optimization_mode
        if mode == "off":
            self._write_log(entry)
            return entry

        entry["optimization"] = await self._optimize_batch(
            epoch=epoch, domain=domain, batch=batch, mode=mode
        )
        self._write_log(entry)
        return entry

    async def _optimize_batch(
        self,
        *,
        epoch: int,
        domain: str,
        batch: list[dict[str, Any]],
        mode: str,
    ) -> dict[str, Any]:
        """suggest/apply 模式:调用 Optimizer 并(apply 时)走验证门控。"""
        reverse_map = reverse_label_mapping(self.config.resolved_label_schema)
        for record in batch:
            gold = record.get("gold_native_label")
            record["gold_canonical"] = canonical_gold(gold, reverse_map) if gold is not None else None

        cases = [_optimizer_case(record) for record in batch]
        versions = {
            name: skill.version
            for name, skill in self.store.load_catalog(list(ALL_SKILL_NAMES)).items()
        }
        allowed_actions = ["update_prompt"] if mode == "apply" else []
        payload = {
            "dataset": self.config.name,
            "epoch": epoch,
            "domain": domain,
            "allowed_actions": allowed_actions,
            "skill_versions": versions,
            "cases": cases,
        }
        optimizer_text = await self._optimizer().run(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
        report = parse_json_content(optimizer_text)
        result: dict[str, Any] = {
            "mode": mode,
            "raw_output": optimizer_text,
            "report": report,
            "promotion": None,
            "gate": None,
        }
        if report is None:
            result["error"] = "optimization_report_v2 解析失败"
            return result

        action = (report.get("decision") or {}).get("action")
        target = (report.get("target") or {}).get("agent")
        patch = (report.get("proposed_change") or {}).get("patch")
        if mode == "apply" and action == "UPDATE_PROMPT" and isinstance(patch, str) and patch.strip():
            result["gate"] = await self._run_promotion_gate(
                target=target,
                patch_spec=_parse_patch_spec(patch.strip()),
                change_reason=str((report.get("decision") or {}).get("reason", "")),
            )
        return result

    async def _run_promotion_gate(
        self, *, target: Any, patch_spec: dict[str, Any], change_reason: str
    ) -> dict[str, Any]:
        if target not in ALL_SKILL_NAMES:
            return {"error": f"Optimizer 目标不在固定 Skill 集合内: {target!r}", "accepted": False}
        try:
            active_package = self.store.load_active(target).package
            candidate_package = apply_package_patch(active_package, patch_spec)
        except Exception as exc:
            return {"error": f"候选补丁校验失败: {type(exc).__name__}: {exc}", "accepted": False}

        # 前置关卡 B:候选脚本静态自检(语法 + import + 工具注册冒烟)。
        preflight_errors = preflight_candidate_scripts(
            target,
            candidate_package,
            scratch_root=self.store.store_root / "_preflight",
        )
        if preflight_errors:
            return {
                "accepted": False,
                "reason": "scripts 静态自检未通过,候选被拒收",
                "script_errors": preflight_errors,
                "preflight_failed": True,
            }

        validation_items = pick_validation_sample(
            self.validation_items_by_domain,
            seed=self.config.seed,
            max_items=self.config.max_validation_items,
        )
        if not validation_items:
            return {"error": "验证集为空,无法执行晋升门控", "accepted": False}

        # 前置关卡 A:候选评测/晋升任何一步失败都优雅降级,不中断训练。
        assert self._runner_factory is not None
        try:
            comparison = await evaluate_candidate(
                self.config,
                self._runner_factory,
                validation_items,
                target,
                candidate_package,
            )
        except Exception as exc:
            return {
                "accepted": False,
                "reason": "候选端到端评测失败,候选被拒收",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not comparison["improved"]:
            return {
                "accepted": False,
                "reason": "候选指标未严格优于当前 active",
                **comparison,
            }
        try:
            record = self.store.promote(
                target,
                candidate_package,
                change_reason=change_reason,
                metrics_before=comparison["baseline_metrics"],
                metrics_after=comparison["candidate_metrics"],
            )
        except Exception as exc:
            return {
                "accepted": False,
                "reason": "候选指标通过但晋升写盘失败",
                "error": f"{type(exc).__name__}: {exc}",
                **comparison,
            }
        # 下一批次起加载新 Skill(含 scripts 工具的重新注册)。
        entry = {
            "accepted": True,
            "promoted_seq": record["seq"],
            "promoted_version": record["version"],
            **comparison,
        }
        if self._runner is not None:
            try:
                await self._runner.refresh(catalog_overrides=None)
            except Exception as exc:
                entry["warning"] = (
                    "Skill 已晋升但 Agent 重建失败(当前 runner 仍是旧 Skill),"
                    f"请检查日志: {type(exc).__name__}: {exc}"
                )
        return entry

    def _optimizer(self):
        if getattr(self, "_optimizer_agent", None) is None:
            from agents.optimizer import OptimizationAgent

            self._optimizer_agent = OptimizationAgent(self.config)
        return self._optimizer_agent

    # ---- 训练主循环 -----------------------------------------------------

    async def train(self) -> list[dict[str, Any]]:
        config = self.config
        self._runner_factory = self.make_runner
        self._runner = self._runner_factory(None)
        logs: list[dict[str, Any]] = []

        domain_items: dict[str, list[dict[str, Any]]] = {
            domain: list(items) for domain, items in self.train_items_by_domain.items()
        }
        total_items = sum(len(items) for items in domain_items.values())
        if total_items == 0:
            await self._runner.close()
            return logs

        try:
            for epoch_index in range(config.epoch):
                epoch = epoch_index + 1
                # 每轮内对每个训练域按固定派生种子洗牌。
                rng = random.Random(f"{config.seed}:epoch:{epoch}")
                queue: list[tuple[str, dict[str, Any]]] = []
                for domain in config.train_domains:
                    shuffled = list(domain_items[domain])
                    rng.shuffle(shuffled)
                    queue.extend((domain, item) for item in shuffled)

                batch: list[dict[str, Any]] = []
                batch_index = 0
                previous_domain: str | None = None
                assert self._runner is not None
                with tqdm(total=len(queue), desc=f"Epoch {epoch}/{config.epoch}",
                          unit="item", dynamic_ncols=True) as progress:
                    for domain, item in queue:
                        # 域切换时先清空未满 batch,保证日志里的 category 纯净。
                        if previous_domain is not None and domain != previous_domain and batch:
                            logs.append(
                                await self._finish_batch(
                                    epoch=epoch, domain=previous_domain,
                                    batch_index=batch_index, batch=batch,
                                )
                            )
                            batch_index += 1
                            batch = []
                        previous_domain = domain
                        record = await self._runner.infer_one(item)
                        record["gold_native_label"] = item.get("label")
                        record["domain"] = domain
                        self._write_trace(record)
                        batch.append(record)

                        if len(batch) >= config.batch_size:
                            logs.append(
                                await self._finish_batch(
                                    epoch=epoch, domain=domain,
                                    batch_index=batch_index, batch=batch,
                                )
                            )
                            batch_index += 1
                            batch = []
                        progress.update(1)

                if batch and previous_domain is not None:  # 不丢弃不足一批的尾部样本。
                    logs.append(
                        await self._finish_batch(
                            epoch=epoch, domain=previous_domain,
                            batch_index=batch_index, batch=batch,
                        )
                    )
        finally:
            if self._runner is not None:
                await self._runner.close()
                self._runner = None
            optimizer_agent = getattr(self, "_optimizer_agent", None)
            if optimizer_agent is not None:
                await optimizer_agent.close()
                self._optimizer_agent = None

        return logs


async def run_test_evaluation(config: Any) -> dict[str, Any]:
    """冻结测试域上的最终推理(推理路径):只读、不优化、不改 Skill。"""
    test_pool = dataset_mod.load_test_pool(config)
    missing = [name for name in config.test_domains if name not in test_pool.data]
    if missing:
        raise ValueError(f"test_domains 中有测试池不存在的域: {missing}")

    items: list[dict[str, Any]] = []
    for domain in config.test_domains:
        items.extend(test_pool.data[domain])

    run_dir = _trace_dir_path(config) / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = InferenceRunner(config)
    records: list[dict[str, Any]] = []
    try:
        for item in tqdm(items, desc="Test evaluation", unit="item", dynamic_ncols=True):
            record = await runner.infer_one(item)
            record["gold_native_label"] = item.get("label")
            records.append(record)
            with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str))
                stream.write("\n")
    finally:
        await runner.close()

    metrics = classification_metrics(
        extract_pairs(records, config.resolved_label_schema)
    )
    summary = {
        "dataset": config.name,
        "test_domains": config.test_domains,
        "n_items": len(records),
        "metrics": metrics,
        "run_dir": str(run_dir),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
