"""数据集加载与域内确定性切分(审计 3.5 节: 训练/验证/测试隔离)。

约定:
- 训练用 JSONL 视为“样本池”,按配置里的 train_domains 过滤,并在每个域内
  用固定随机种子切出 train 与 validation(meta-validation)两份;
- test_domains 对应的文件(test pool)只在最终评估阶段读取,训练与 Skill
  优化阶段一律不触碰;
- 未知类别(如 Weibo21 的“无法确定”)不再静默丢弃,加载时统计并告警。
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from typing import Any

WEIBO_KNOWN_DOMAINS = [
    "科技",
    "军事",
    "教育考试",
    "灾难事故",
    "政治",
    "医药健康",
    "财经商业",
    "文体娱乐",
    "社会生活",
]

AMTCELE_KNOWN_DOMAINS = ["biz", "celebrity", "edu", "entmt", "polit", "sports", "tech"]


def _read_jsonl(file_paths: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def load_weibo_data(file_paths: list[str]) -> list[dict[str, Any]]:
    """读取 Weibo21 记录;返回 dict,并打印被丢弃的未知类别统计。"""
    raw_items = _read_jsonl(file_paths)
    dropped: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for item in raw_items:
        category = item.get("category")
        if category in WEIBO_KNOWN_DOMAINS:
            kept.append(item)
        else:
            dropped[str(category)] += 1
    if dropped:
        detail = ", ".join(f"{name}: {count}" for name, count in dropped.most_common())
        print(f"[dataset] 丢弃未知类别样本 {sum(dropped.values())} 条: {detail}",
              file=sys.stderr)
    return kept


def load_AMTCele_data(file_paths: list[str]) -> list[dict[str, Any]]:
    """读取 AMTCele 记录;domain 字段去除数字后缀后按 7 域分组。"""
    items = _read_jsonl(file_paths)
    dropped = 0
    kept: list[dict[str, Any]] = []
    for item in items:
        domain = str(item.get("domain", "")).rstrip("0123456789")
        if domain in AMTCELE_KNOWN_DOMAINS:
            item = dict(item)
            item["domain"] = domain
            kept.append(item)
        else:
            dropped += 1
    if dropped:
        print(f"[dataset] AMTCele 丢弃 {dropped} 条未知域样本。", file=sys.stderr)
    return kept


def group_by_domain(items: list[dict[str, Any]], domains: list[str]) -> dict[str, list[dict[str, Any]]]:
    """按给定域列表分桶(保证每个 key 都存在)。"""
    grouped = {domain: [] for domain in domains}
    for item in items:
        domain = item.get("domain") if "domain" in item else item.get("category")
        if domain in grouped:
            grouped[domain].append(item)
    return grouped


def split_domains(
    items_by_domain: dict[str, list[dict[str, Any]]],
    seed: int,
    validation_ratio: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """每个域内按 (seed, 域名) 派生随机源,确定性切出训练与验证两份。

    返回 (train_by_domain, validation_by_domain)。验证集固定不变,训练集在
    Trainer 每轮内部再随机洗牌。
    """
    train: dict[str, list[dict[str, Any]]] = {}
    validation: dict[str, list[dict[str, Any]]] = {}
    for domain, items in items_by_domain.items():
        rng = random.Random(f"{seed}:{domain}")
        shuffled = list(items)
        rng.shuffle(shuffled)
        split_at = int(len(shuffled) * validation_ratio)
        if split_at <= 0 and len(shuffled) > 1:
            split_at = 1
        validation[domain] = shuffled[:split_at]
        train[domain] = shuffled[split_at:]
    return train, validation


def shuffle_items(items: list[dict[str, Any]], rng: random.Random) -> None:
    """原地洗牌(用于每轮训练集随机化)。"""
    rng.shuffle(items)


class Dataset_Weibo:
    """Weibo21 数据容器:data = {域: [样本]}。"""

    def __init__(self, file_path: list[str]):
        items = load_weibo_data(file_path)
        self.data = group_by_domain(items, WEIBO_KNOWN_DOMAINS)

    def verify_data(self) -> None:
        for category, items in self.data.items():
            print(f"Category: {category}, Number of items: {len(items)}")

    def shuffle_data(self, categorys: list[str]) -> None:
        """兼容旧接口:全量洗牌(新代码请改用 split_domains + shuffle_items)。"""
        rng = random.Random()
        for category in categorys:
            if category in self.data:
                rng.shuffle(self.data[category])


class Dataset_AMTCele:
    """AMTCele 数据容器:data = {域: [样本]}。"""

    def __init__(self, file_path: list[str]):
        items = load_AMTCele_data(file_path)
        self.data = group_by_domain(items, AMTCELE_KNOWN_DOMAINS)

    def verify_data(self) -> None:
        for category, items in self.data.items():
            print(f"Category: {category}, Number of items: {len(items)}")

    def shuffle_data(self, categorys: list[str]) -> None:
        rng = random.Random()
        for category in categorys:
            if category in self.data:
                rng.shuffle(self.data[category])


def load_dataset(config: Any):
    """按 config.dataset_name 加载训练池数据集(样本池文件)。"""
    name = config.name
    file_paths = getattr(config, "dataset_paths", None) or [
        f"./datasets/Weibo21/train.jsonl" if name in {"weibo", "weibo21"}
        else f"./datasets/AMTCele/AMTCele.jsonl"
    ]
    if name in {"weibo", "weibo21"}:
        return Dataset_Weibo(file_path=file_paths)
    if name == "amtcele":
        return Dataset_AMTCele(file_path=file_paths)
    raise ValueError(f"Unsupported dataset: {config.dataset_name}")


def load_test_pool(config: Any):
    """按 config.dataset_name 加载冻结测试池(仅最终评估阶段使用)。"""
    name = config.name
    if name in {"weibo", "weibo21"}:
        path = ["./datasets/Weibo21/test.jsonl"]
        return Dataset_Weibo(file_path=path)
    if name == "amtcele":
        # AMTCele 未提供独立 test 文件时退化为同一文件(域隔离仍保证)。
        return Dataset_AMTCele(file_path=["./datasets/AMTCele/AMTCele.jsonl"])
    raise ValueError(f"Unsupported dataset: {config.dataset_name}")
