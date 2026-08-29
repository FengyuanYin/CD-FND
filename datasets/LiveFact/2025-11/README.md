---
license: apache-2.0
language:
- en
configs:
- config_name: default
  data_files:
  - split: plus_3_cls
    path: "livefact_+3_cls.jsonl"
  - split: plus_3_inf
    path: "livefact_+3_inf.jsonl"
  - split: minus_3_cls
    path: "livefact_-3_cls.jsonl"
  - split: minus_3_inf
    path: "livefact_-3_inf.jsonl"
  - split: 0_cls
    path: "livefact_0_cls.jsonl"
  - split: 0_inf
    path: "livefact_0_inf.jsonl"
---