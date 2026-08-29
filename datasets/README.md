# Misinformation detection datasets

This directory intentionally contains only the four datasets selected for the
cross-domain, continual misinformation-detection experiments.

## Weibo21

- Files: `Weibo21/train.pkl`, `val.pkl`, `test.pkl`
- Fields: `content`, `label`, `category`
- Local processed split sizes: 5,751 / 1,918 / 1,923 (9,592 rows total)
- Source: https://github.com/kennqiang/MDFEND-Weibo21

These are the processed splits published in the authors' official repository.
The repository requires an application for access to the original, unsplit
Weibo21 data. The processed repository version has 9,592 rows, while the paper
reports 9,128 deduplicated examples; experiments must identify which version is
used and should check duplicates before constructing new splits.

## AMTCele

- File: `AMTCele/AMTCele.csv`
- Fields: `domain`, `label`, `text`
- Size: 980 rows
- Source: https://github.com/lzw108/RAEmoLLM

AMTCele combines FakeNewsAMT and Celebrity data into seven domains. The local
file is the processed version published by the RAEmoLLM authors.

## LiveFact

- Monthly releases: `LiveFact/2025-11`, `2025-12`, `2026-01`, `2026-02`
- Each release contains before (`-3`), during (`0`), and after (`+3`) files for
  classification (`cls`) and inference (`inf`).
- Source: https://huggingface.co/collections/bebxy/livefact

## AdvFake

- File: `AdvFake/final.csv`
- Size: 402 rows
- License: ODC-BY
- Source: https://huggingface.co/datasets/sanxing/advfake
