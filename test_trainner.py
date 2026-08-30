import asyncio
import json
from types import SimpleNamespace

import trainner
from trainner import Trainer


class FakeResult:
    def __init__(self, content):
        self.messages = [SimpleNamespace(content=content)]


class FakeCoordinator:
    inputs = []

    def __init__(self, config):
        pass

    async def run(self, value):
        self.inputs.append(json.loads(value))
        return FakeResult({"schema_version": "analysis_report_v2"})

    async def close(self):
        pass


class FakeJudge:
    inputs = []

    def __init__(self, config):
        pass

    async def judge(self, value):
        self.inputs.append(json.loads(value))
        return FakeResult({"prediction": {"dataset_label": 0}})

    async def close(self):
        pass


class FakeOptimizer:
    inputs = []

    def __init__(self, config):
        pass

    async def optimize(self, value):
        parsed = json.loads(value)
        self.inputs.append(parsed)
        return FakeResult({"case_count": len(parsed["cases"])})

    async def close(self):
        pass


class FakeDataset:
    def __init__(self):
        self.data = {
            "科技": [
                {"content": "a", "category": "科技", "label": 0},
                {"content": "b", "category": "科技", "label": 1},
                {"content": "c", "category": "科技", "label": 0},
            ]
        }

    def shuffle_data(self, domains):
        assert domains == ["科技"]


def test_training_batches_without_label_leakage(monkeypatch):
    FakeCoordinator.inputs.clear()
    FakeJudge.inputs.clear()
    FakeOptimizer.inputs.clear()
    monkeypatch.setattr(trainner, "CoordinatorAgent", FakeCoordinator)
    monkeypatch.setattr(trainner, "JudgeAgent", FakeJudge)
    monkeypatch.setattr(trainner, "OptimizationAgent", FakeOptimizer)

    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        dataset_name="weibo21",
        domain_name=["科技"],
        epoch=1,
        batch_size=2,
        label_schema={"mapping": {"REAL": 0, "FAKE": 1}},
    )
    trainer.dataset = FakeDataset()

    logs = asyncio.run(trainer.train())

    assert [entry["batch_size"] for entry in logs] == [2, 1]
    assert all("label" not in call["item"] for call in FakeCoordinator.inputs)
    assert all("label" not in call["source_item"] for call in FakeJudge.inputs)
    assert [case["expected_result"] for call in FakeOptimizer.inputs for case in call["cases"]] == [0, 1, 0]
