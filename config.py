"""Experiment configuration for the CD-FND misinformation-detection system.

科研约定(遵循 LOCAL_PROJECT_AUDIT.md 第 1.2/3.5 节):
- 训练、验证、测试共用同一个 Config,禁止为某一阶段单独放宽约束。
- API Key 只从环境变量 ``DEEPSEEK_API_KEY`` 读取,代码中不允许出现明文密钥。
- ``train_domains`` 与 ``test_domains`` 默认互不重叠(跨域隔离);若要重叠,
  必须显式传入并自行承担评估可信度风险。
- 验证集从训练域内按固定随机种子切分(meta-validation),测试域样本在训练与
  Skill 优化阶段完全冻结。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

API_KEY_ENV = "DEEPSEEK_API_KEY"

# 默认数据集标签语义。来源说明:
#   Weibo21(MDFEND / MFND 基准,见 https://arxiv.org/abs/2112.10994):
#     惯例 0 = 真实(REAL), 1 = 虚假(FAKE)。已抽样核对样例与之一致,
#     论文写作前请再对照官方数据说明确认,必要时在 Config 中整体覆盖。
#   AMTCele(见 OFFICIAL_README.md): 原生标签为 "legit"(真实)与 "fake"(虚假)。
DEFAULT_LABEL_SCHEMAS: dict[str, dict] = {
    "weibo": {
        "allowed_labels": [0, 1],
        "mapping": {"REAL": 0, "FAKE": 1, "AMBIGUOUS": None},
        "abstention_allowed": True,
    },
    "weibo21": {
        "allowed_labels": [0, 1],
        "mapping": {"REAL": 0, "FAKE": 1, "AMBIGUOUS": None},
        "abstention_allowed": True,
    },
    "amtcele": {
        "allowed_labels": ["legit", "fake"],
        "mapping": {"REAL": "legit", "FAKE": "fake", "AMBIGUOUS": None},
        "abstention_allowed": True,
    },
}

# 每个数据集的训练/测试域默认划分(互不重叠)。
DEFAULT_DOMAINS: dict[str, tuple[list[str], list[str]]] = {
    "weibo": (
        ["科技", "军事", "教育考试", "灾难事故", "政治"],
        ["医药健康", "财经商业", "文体娱乐", "社会生活"],
    ),
    "weibo21": (
        ["科技", "军事", "教育考试", "灾难事故", "政治"],
        ["医药健康", "财经商业", "文体娱乐", "社会生活"],
    ),
    "amtcele": (
        ["tech", "biz", "celebrity", "edu"],
        ["polit", "entmt", "sports"],
    ),
}


def canonical_dataset_name(name: str) -> str:
    """把用户习惯写法(weibo / weibo21)规整为配置键。"""
    lowered = name.casefold().strip()
    return "weibo" if lowered == "weibo" else lowered


@dataclass
class Config:
    """实验配置。构造后直接传给各 Agent、Trainer 与评估流程。"""

    model_name: str
    base_url: str
    epoch: int = 1
    dataset_name: str = "weibo"
    batch_size: int = 8
    api_key: str | None = None

    # 域划分:验证集从 train_domains 内按 seed 切分,test_domains 全程冻结。
    train_domains: list[str] | None = None
    test_domains: list[str] | None = None
    seed: int = 42
    validation_ratio: float = 0.1

    # Skill 优化模式:
    #   "off"     - 不做任何优化调用,只跑 Coordinator/Specialist/Judge 基线;
    #   "suggest" - 每个批次结束后让 Optimizer 生成诊断与候选建议(不改文件);
    #   "apply"   - 生成候选后在独立验证集上评测,通过才 promote(写入 version
    #               store),失败自动 reject 并回滚。
    optimization_mode: str = "off"

    # candidate 验证阶段最多评测的样本数(控制 API 成本)。
    max_validation_items: int = 8

    # 输出目录:预测 trace、批次日志、指标都会写到这里。
    runs_dir: Path = field(default_factory=lambda: Path("outputs"))
    run_id: str | None = None

    # Skill 与版本仓库位置。
    skills_root: Path = field(default_factory=lambda: Path("skills"))
    skill_store_root: Path = field(default_factory=lambda: Path("skill_store"))

    # 由研究者显式提供时覆盖默认标签语义。
    label_schema: dict | None = None

    # ---- 派生属性(只读使用,不要修改) ---------------------------------

    @property
    def name(self) -> str:
        return canonical_dataset_name(self.dataset_name)

    @property
    def domain_name(self) -> list[str]:
        """兼容旧字段名的别名,等价于 train_domains。"""
        return self.train_domains or []

    @property
    def resolved_api_key(self) -> str:
        """优先取构造参数,否则读环境变量。"""
        if self.api_key:
            return self.api_key
        key = os.environ.get(API_KEY_ENV, "")
        if not key:
            raise ValueError(
                f"未找到 API Key:请设置环境变量 {API_KEY_ENV},不要在代码里写密钥。"
            )
        return key

    @property
    def resolved_label_schema(self) -> dict:
        if self.label_schema is not None:
            return self.label_schema
        schema = DEFAULT_LABEL_SCHEMAS.get(self.name)
        if schema is None:
            raise ValueError(f"未定义默认标签语义的数据集: {self.name}")
        return {key: list(value) if isinstance(value, list) else value
                for key, value in schema.items()}

    # ---- 初始化校验 ----------------------------------------------------

    def __post_init__(self) -> None:
        if self.epoch < 1 or self.batch_size < 1:
            raise ValueError("epoch 与 batch_size 必须为正整数")
        if not (0.0 < self.validation_ratio < 1.0):
            raise ValueError("validation_ratio 必须在 (0, 1) 之间")
        if self.optimization_mode not in {"off", "suggest", "apply"}:
            raise ValueError(
                "optimization_mode 必须是 off / suggest / apply 之一"
            )

        if self.train_domains is None or self.test_domains is None:
            defaults = DEFAULT_DOMAINS.get(self.name)
            if defaults is None:
                raise ValueError(
                    f"不支持的数据集: {self.dataset_name}。可用: "
                    f"{sorted(DEFAULT_DOMAINS)}"
                )
            train, test = defaults
            self.train_domains = list(train)
            self.test_domains = list(test)

        overlap = set(self.train_domains) & set(self.test_domains)
        if overlap:
            raise ValueError(
                "train_domains 与 test_domains 不能重叠(跨域实验隔离要求): "
                f"{sorted(overlap)}"
            )

        self.runs_dir = Path(self.runs_dir)
        self.skills_root = Path(self.skills_root)
        self.skill_store_root = Path(self.skill_store_root)
