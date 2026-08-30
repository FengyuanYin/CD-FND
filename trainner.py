"""Training/evaluation loop for the misinformation-detection agents."""

from __future__ import annotations

import json
from typing import Any

from agents import CoordinatorAgent, JudgeAgent, OptimizationAgent
from datasets import dataset
from tqdm.auto import tqdm


def _result_content(result: Any) -> Any:
    """Extract the final message content from an AutoGen TaskResult."""

    messages = getattr(result, "messages", None)
    if messages:
        return getattr(messages[-1], "content", str(messages[-1]))
    return result


class Trainer:
    def __init__(self, config: Any):
        self.config = config
        dataset_name = config.dataset_name.casefold()
        file_paths = getattr(config, "dataset_paths", None)

        if dataset_name in {"weibo", "weibo21"}:
            self.dataset = dataset.Dataset_Weibo(
                file_path=file_paths or ["./datasets/Weibo21/train.jsonl"]
            )
        elif dataset_name == "amtcele":
            self.dataset = dataset.Dataset_AMTCele(
                file_path=file_paths or ["./datasets/AMTCele/AMTCele.jsonl"]
            )
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    @staticmethod
    def _model_input(item: dict[str, Any], dataset_name: str) -> str:
        """Serialize observable input fields without leaking the gold label."""

        visible_item = {
            key: value
            for key, value in item.items()
            if key.casefold() not in {"label", "gold_label", "target", "answer"}
        }
        return json.dumps(
            {
                "dataset_format": dataset_name,
                "item": visible_item,
            },
            ensure_ascii=False,
        )

    async def _optimize_batch(
        self,
        optimization_agent: OptimizationAgent,
        batch: list[dict[str, Any]],
        *,
        epoch: int,
        category: str,
    ) -> dict[str, Any]:
        """Optimize one completed batch and preserve the batch in the log."""

        batch_snapshot = list(batch)
        optimization_input = json.dumps(
            {
                "dataset": self.config.dataset_name,
                "cases": batch_snapshot,
                "allowed_actions": [],
            },
            ensure_ascii=False,
            default=str,
        )
        optimization_result = await optimization_agent.optimize(optimization_input)

        return {
            "epoch": epoch,
            "category": category,
            "batch_size": len(batch_snapshot),
            "optimization_response": _result_content(optimization_result),
            "batch": batch_snapshot,
        }

    async def train(self) -> list[dict[str, Any]]:
        """Run all configured epochs and return batch-level optimization logs."""

        print("Training the Misinformation Detection Agent...")
        coordinator_agent = CoordinatorAgent(self.config)
        optimization_agent = OptimizationAgent(self.config)
        judge_agent = JudgeAgent(self.config)
        logs: list[dict[str, Any]] = []

        domains = self.config.domain_name
        if isinstance(domains, str):
            domains = [domains]

        try:
            for epoch_index in range(self.config.epoch):
                epoch = epoch_index + 1
                self.dataset.shuffle_data(domains)
                total_items = sum(len(self.dataset.data.get(name, [])) for name in domains)

                with tqdm(
                    total=total_items,
                    desc=f"Epoch {epoch}/{self.config.epoch}",
                    unit="item",
                    dynamic_ncols=True,
                ) as progress:
                    for category in domains:
                        if category not in self.dataset.data:
                            raise ValueError(f"Unknown dataset category/domain: {category}")

                        progress.set_postfix(category=category, refresh=False)
                        batch: list[dict[str, Any]] = []
                        for item in self.dataset.data[category]:
                            user_input = self._model_input(item, self.config.dataset_name)
                            coordinator_result = await coordinator_agent.run(user_input)
                            coordinator_response = _result_content(coordinator_result)

                            judge_input = json.dumps(
                                {
                                    "dataset_format": self.config.dataset_name,
                                    "source_item": json.loads(user_input)["item"],
                                    "analysis_report": coordinator_response,
                                    "label_schema": getattr(self.config, "label_schema", None),
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                            judge_result = await judge_agent.judge(judge_input)
                            judge_response = _result_content(judge_result)

                            # The reference label is revealed only to OptimizationAgent
                            # after prediction, never to CoordinatorAgent or JudgeAgent.
                            batch.append(
                                {
                                    "input": json.loads(user_input),
                                    "coordinator_response": coordinator_response,
                                    "judge_response": judge_response,
                                    "expected_result": item.get("label"),
                                }
                            )

                            if len(batch) >= self.config.batch_size:
                                logs.append(
                                    await self._optimize_batch(
                                        optimization_agent,
                                        batch,
                                        epoch=epoch,
                                        category=category,
                                    )
                                )
                                tqdm.write(f"Processed a batch of {len(batch)} items.")
                                batch.clear()

                            progress.update(1)

                        # Do not silently discard the final incomplete batch.
                        if batch:
                            logs.append(
                                await self._optimize_batch(
                                    optimization_agent,
                                    batch,
                                    epoch=epoch,
                                    category=category,
                                )
                            )
                            tqdm.write(f"Processed a final batch of {len(batch)} items.")
        finally:
            await coordinator_agent.close()
            await optimization_agent.close()
            await judge_agent.close()

        return logs
