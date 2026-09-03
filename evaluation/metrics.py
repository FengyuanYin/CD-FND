"""评估指标模块(审计 5 节: 论文评估基础集)。

只做科研常用且无歧义的计算:Accuracy、Macro-F1、各类 Precision/Recall/F1、
混淆矩阵、弃权/覆盖率。ECE 等校准指标属于论文阶段扩展,暂不内置以免引入
不必要复杂度。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

DECIDED_LABELS = ("REAL", "FAKE")


def reverse_label_mapping(label_schema: dict[str, Any]) -> dict[Any, str]:
    """把 label_schema.mapping 反转为 {原生标签: 规范标签}。"""
    mapping = label_schema.get("mapping", {})
    reversed_map: dict[Any, str] = {}
    for canonical, native in mapping.items():
        if native is None:
            continue
        if native in reversed_map:
            raise ValueError(
                f"label_schema.mapping 不可逆: 多个规范标签映射到同一原生值 {native!r}"
            )
        reversed_map[native] = canonical
    return reversed_map


def canonical_gold(native_label: Any, reverse_map: dict[Any, str]) -> str | None:
    """把原生 Gold 标签转为规范标签;无法识别返回 None(该样本不参与指标)。"""
    return reverse_map.get(native_label)


def extract_pairs(
    records: list[dict[str, Any]],
    label_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """从推理记录中抽取 (gold_canonical, predicted_canonical) 对比行。

    records 是 InferenceRunner.infer_one 的输出,其中含 gold 原生标签。
    """
    reverse_map = reverse_label_mapping(label_schema)
    pairs: list[dict[str, Any]] = []
    for record in records:
        native = record.get("gold_native_label")
        gold = canonical_gold(native, reverse_map) if native is not None else None
        decision = record.get("judge_decision") or {}
        prediction = decision.get("prediction") or {}
        pairs.append(
            {
                "gold_canonical": gold,
                "predicted_canonical": prediction.get("canonical_label"),
                "prediction_status": prediction.get("status"),
                "dataset_label": prediction.get("dataset_label"),
                "gold_native": native,
            }
        )
    return pairs


def classification_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """对已决策(REAL/FAKE)对计算 Accuracy / Macro-F1 / 逐类 PRF / 混淆。

    返回的 key 稳定,便于多个 seed 的结果做均值±标准差。
    """
    decided = [
        (p["gold_canonical"], p["predicted_canonical"])
        for p in pairs
        if p["gold_canonical"] in DECIDED_LABELS
        and p["predicted_canonical"] in DECIDED_LABELS
    ]
    total = len(pairs)
    abstained = sum(
        1 for p in pairs if p["gold_canonical"] in DECIDED_LABELS
        and p["predicted_canonical"] not in DECIDED_LABELS
    )
    n_decided = len(decided)

    confusion = {
        "REAL": {"REAL": 0, "FAKE": 0},
        "FAKE": {"REAL": 0, "FAKE": 0},
    }
    for gold, predicted in decided:
        confusion[gold][predicted] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in DECIDED_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in DECIDED_LABELS if other != label)
        fn = sum(confusion[label][other] for other in DECIDED_LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
        f1_values.append(f1)

    accuracy = sum(1 for gold, predicted in decided if gold == predicted) / n_decided if n_decided else 0.0

    gold_counter = Counter(p["gold_canonical"] for p in pairs if p["gold_canonical"])
    return {
        "n_samples": total,
        "n_decided": n_decided,
        "n_abstained_or_unmapped": total - n_decided,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "gold_distribution": dict(gold_counter),
    }
