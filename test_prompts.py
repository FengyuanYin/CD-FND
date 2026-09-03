import json

from prompts import (
    COODINATOR_AGENT_SYSTEM_PROMPT,
    COORDINATOR_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,
    OPTIMIZATION_AGENT_SYSTEM_PROMPT,
    SPECIALIST_AGENT_SYSTEM_PROMPT,
)


def _output_example(prompt: str) -> dict:
    contract = prompt.split("<output_contract>", 1)[1].split("</output_contract>", 1)[0]
    start = contract.index("{")
    end = contract.rindex("}") + 1
    return json.loads(contract[start:end])


def test_prompt_output_examples_are_valid_json():
    assert _output_example(COORDINATOR_AGENT_SYSTEM_PROMPT)["schema_version"] == "routing_decision_v1"
    assert _output_example(SPECIALIST_AGENT_SYSTEM_PROMPT)["schema_version"] == "specialist_report_v1"
    assert _output_example(JUDGE_AGENT_SYSTEM_PROMPT)["schema_version"] == "judge_decision_v2"
    assert _output_example(OPTIMIZATION_AGENT_SYSTEM_PROMPT)["schema_version"] == "optimization_report_v2"


def test_prompts_cover_local_dataset_formats():
    for prompt in (
        COORDINATOR_AGENT_SYSTEM_PROMPT,
        SPECIALIST_AGENT_SYSTEM_PROMPT,
        JUDGE_AGENT_SYSTEM_PROMPT,
        OPTIMIZATION_AGENT_SYSTEM_PROMPT,
    ):
        for dataset_name in ("Weibo21", "AMTCele", "LiveFact", "AdvFake"):
            assert dataset_name in prompt


def test_legacy_coordinator_name_remains_compatible():
    assert COODINATOR_AGENT_SYSTEM_PROMPT == COORDINATOR_AGENT_SYSTEM_PROMPT
