"""候选 Skill 的验证门控(审计 6 节: GATE)。

职责:在独立验证集上对比“当前 active Skill”与“候选整包”的端到端指标,
决定是否值得 promote。所有评测都走与训练相同的 InferenceRunner,避免引入
与线上不一致的第二条推理路径。候选整包由 Trainer 先经
``apply_package_patch`` 生成,再传入这里评测——验证阶段绝不写 version store。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from evaluation.metrics import classification_metrics, extract_pairs
from optimization.version_store import (
    ActiveSkill,
    materialize_package,
)
from orchestration.skill_tools import build_skill_tools

# 晋升比较使用的指标优先级:先看 macro_f1,再看 accuracy(仅在宏 F1 缺失时)。
_METRIC_ORDER = ("macro_f1", "accuracy")

RunnerFactory = Callable[[dict | None], Any]


def metric_value(metrics: dict[str, Any]) -> float:
    """从指标字典里取用于晋升比较的标量(0~1)。"""
    for key in _METRIC_ORDER:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    return 0.0


def is_improvement(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """候选指标严格优于当前 active 才算提升(论文阶段再引入显著性检验)。"""
    return metric_value(after) > metric_value(before)


def pick_validation_sample(
    validation_by_domain: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    max_items: int,
) -> list[dict[str, Any]]:
    """从各验证域里按 (seed, 域) 确定性抽样,拼成用于门控评测的小样本。"""
    sample: list[dict[str, Any]] = []
    for domain in sorted(validation_by_domain):
        items = validation_by_domain[domain]
        if not items:
            continue
        rng = random.Random(f"{seed}:validation:{domain}")
        shuffled = list(items)
        rng.shuffle(shuffled)
        sample.extend(shuffled[:max_items])
        if len(sample) >= max_items:
            break
    return sample[:max_items]


def preflight_candidate_scripts(
    skill_name: str,
    candidate_package: dict[str, Any],
    scratch_root: Path,
) -> list[str] | None:
    """候选脚本静态自检(晋升门控的前置关卡)。

    对候选整包里 scripts/ 下每个 .py 依次做:
    1. 语法编译(compile);
    2. 物化后按真实文件 import + FunctionTool 注册冒烟(可同时发现
       导入错误、工具重名、Schema 生成失败等运行时构建问题)。

    返回 None 表示通过;否则返回可读错误列表(调用方应拒收该候选,
    避免把坏脚本带进端到端评测并浪费 API 预算)。
    """
    resources = candidate_package.get("resources") or {}
    scripts = sorted(
        rel for rel in resources
        if rel.startswith("scripts/") and rel.endswith(".py")
    )
    if not scripts:
        return None

    errors: list[str] = []
    for rel in scripts:
        source = resources[rel]
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            errors.append(f"{rel}: 语法错误: {exc}")

    if errors:
        return errors

    scratch = Path(scratch_root) / skill_name
    try:
        materialize_package(candidate_package, scratch)
        skill = ActiveSkill.from_package(
            skill_name,
            candidate_package,
            version="preflight",
            is_baseline=False,
        )
        build_skill_tools(skill, scratch)
    except Exception as exc:
        errors.append(f"脚本 import/工具注册失败: {type(exc).__name__}: {exc}")
    return errors or None


async def evaluate_items(
    runner: Any,
    items: list[dict[str, Any]],
    label_schema: dict[str, Any],
) -> dict[str, Any]:
    """用给定 runner 逐样本推理并返回 (records, metrics)。"""
    records: list[dict[str, Any]] = []
    for item in items:
        record = await runner.infer_one(item)
        record["gold_native_label"] = item.get("label")
        records.append(record)
    metrics = classification_metrics(extract_pairs(records, label_schema))
    return {"records": records, "metrics": metrics}


async def evaluate_candidate(
    config: Any,
    runner_factory: RunnerFactory,
    validation_items: list[dict[str, Any]],
    skill_name: str,
    candidate_package: dict[str, Any],
) -> dict[str, Any]:
    """评测候选整包:分别跑当前 active 与候选覆盖,返回对比结果。

    candidate_package 是 {metadata, instructions, resources} 整包快照;
    评测只读,不写入 version store。
    """
    label_schema = config.resolved_label_schema

    baseline_runner = runner_factory(catalog_overrides=None)
    try:
        baseline = await evaluate_items(baseline_runner, validation_items, label_schema)
    finally:
        await baseline_runner.close()

    candidate_runner = runner_factory(
        catalog_overrides={skill_name: candidate_package}
    )
    try:
        candidate = await evaluate_items(candidate_runner, validation_items, label_schema)
    finally:
        await candidate_runner.close()

    improved = is_improvement(baseline["metrics"], candidate["metrics"])
    return {
        "skill_name": skill_name,
        "validation_items": len(validation_items),
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "improved": improved,
        "candidate_package": candidate_package,
        "acceptance_rule": "macro_f1 优先,其次 accuracy,需严格大于",
    }
