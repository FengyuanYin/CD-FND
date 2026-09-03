import asyncio
from pathlib import Path
from uuid import uuid4

import trainner
from config import Config
from trainner import Trainer


def _scratch_runs_dir() -> Path:
    """仓库内唯一输出目录(沙箱不允许枚举系统临时目录)。"""
    root = Path("outputs") / "unit_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    return path


class FakeRunner:
    """替身 runner:按 gold 标签返回完全正确的 judge 决策,不发网络请求。"""

    calls: list = []

    def __init__(self, config, catalog_overrides=None):
        self.config = config

    async def infer_one(self, item):
        FakeRunner.calls.append(item)
        canonical = "REAL" if item.get("label") == 0 else "FAKE"
        return {
            "item": dict(item),
            "item_id": item.get("id"),
            "judge_decision": {
                "prediction": {
                    "canonical_label": canonical,
                    "dataset_label": item.get("label"),
                    "status": "DECIDED",
                    "confidence": 0.99,
                },
                "rationale": "fake",
            },
            "routing_report": {
                "routing_decision": {"selected_skills": [], "fallback_used": True}
            },
            "analysis_report": {"specialists": [], "claims": [], "conflicts": []},
            "errors": [],
        }

    async def close(self):
        pass

    def refresh(self, catalog_overrides=None):
        pass


class FakeDataset:
    data = {
        "科技": [
            {"content": f"科技{i}", "category": "科技", "label": i % 2}
            for i in range(4)
        ],
        "军事": [],
        "教育考试": [],
        "灾难事故": [],
        "政治": [],
    }


def _config(runs_dir: Path) -> Config:
    return Config(
        model_name="test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        epoch=1,
        dataset_name="weibo21",
        batch_size=2,
        runs_dir=runs_dir,
    )


def test_training_loop_batches_and_metrics(monkeypatch):
    FakeRunner.calls = []
    monkeypatch.setattr(trainner, "InferenceRunner", FakeRunner)
    monkeypatch.setattr(trainner.dataset_mod, "load_dataset", lambda config: FakeDataset())

    trainer = Trainer(_config(_scratch_runs_dir()))
    logs = asyncio.run(trainer.train())

    # 4 条训练样本,validation_ratio=0.1 -> 3 条进训练,分两个 batch: [2, 1]
    assert [entry["batch_size"] for entry in logs] == [2, 1]
    assert logs[0]["optimization"]["mode"] == "off"
    # 两批都预测正确:accuracy=1.0;宏 F1 视批次内类别是否齐全取 1.0 或 0.5。
    assert all(entry["metrics"]["accuracy"] == 1.0 for entry in logs)
    assert all(entry["metrics"]["macro_f1"] in (0.5, 1.0) for entry in logs)

    # trace 文件应包含全部训练样本记录。
    lines = [line for line in (trainer.trace_path).read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3

    # 训练样本经过 runner(每轮内会确定性洗牌,只校验标签集合)。
    labels_seen = sorted(item.get("label") for item in FakeRunner.calls)
    assert labels_seen == [0, 0, 1]


def test_training_requires_no_network_on_off_mode(monkeypatch):
    """off 模式不会构造 Optimizer,也不读取 skills 目录。"""
    monkeypatch.setattr(trainner, "InferenceRunner", FakeRunner)
    monkeypatch.setattr(trainner.dataset_mod, "load_dataset", lambda config: FakeDataset())
    trainer = Trainer(_config(_scratch_runs_dir()))
    asyncio.run(trainer.train())
    assert getattr(trainer, "_optimizer_agent", None) is None
