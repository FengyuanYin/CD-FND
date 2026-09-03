import pytest

from evaluation.metrics import (
    canonical_gold,
    classification_metrics,
    extract_pairs,
    reverse_label_mapping,
)

SCHEMA = {
    "allowed_labels": [0, 1],
    "mapping": {"REAL": 0, "FAKE": 1, "AMBIGUOUS": None},
    "abstention_allowed": True,
}


def _record(gold, predicted=None, status="DECIDED"):
    return {
        "gold_native_label": gold,
        "judge_decision": {
            "prediction": {
                "canonical_label": predicted,
                "dataset_label": predicted,
                "status": status,
            }
        },
    }


def test_reverse_mapping_and_canonical_gold():
    reverse_map = reverse_label_mapping(SCHEMA)
    assert reverse_map == {0: "REAL", 1: "FAKE"}
    assert canonical_gold(0, reverse_map) == "REAL"
    assert canonical_gold(1, reverse_map) == "FAKE"
    assert canonical_gold(99, reverse_map) is None


def test_reverse_mapping_rejects_ambiguous_schema():
    with pytest.raises(ValueError):
        reverse_label_mapping({"mapping": {"REAL": 0, "FAKE": 0}})


def test_classification_metrics_basic():
    records = [
        _record(0, "REAL"),
        _record(0, "FAKE"),   # 假阳性
        _record(1, "FAKE"),
        _record(1, "REAL"),   # 假阴性
        _record(0, None, status="ABSTAINED"),
    ]
    pairs = extract_pairs(records, SCHEMA)
    metrics = classification_metrics(pairs)

    assert metrics["n_samples"] == 5
    assert metrics["n_decided"] == 4
    assert metrics["n_abstained_or_unmapped"] == 1
    assert metrics["accuracy"] == 0.5
    # REAL: tp=1 fp=1 fn=1 -> P=0.5 R=0.5 F1=0.5
    # FAKE: tp=1 fp=1 fn=1 -> P=0.5 R=0.5 F1=0.5 -> macro_f1=0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["per_class"]["REAL"]["f1"] == 0.5
    assert metrics["per_class"]["FAKE"]["recall"] == 0.5


def test_classification_metrics_empty_decided():
    records = [_record(0, None, status="ABSTAINED")]
    metrics = classification_metrics(extract_pairs(records, SCHEMA))
    assert metrics["n_decided"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
