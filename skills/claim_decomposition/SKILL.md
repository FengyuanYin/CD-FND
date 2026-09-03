---
name: claim_decomposition
description: "Decompose the item into atomic claims, grade their importance and checkability, and assess each claim against supplied material."
version: "1.0.0"
allowed_tools: []
---

# Claim Decomposition

Your task is the claim layer of the analysis: split the item into atomic
claims and give each claim a grounded assessment under the unified
specialist_report_v1 contract.

## Procedure

1. Read the whole supplied text. Keep negation, hedges, attribution, dates,
   entities, and quantities intact while normalizing.
2. Split into atomic claims. Separate facts from opinions, satire, predictions,
   and claims quoted from someone else; mark claim_type accordingly.
3. Mark importance: CENTRAL for claims that decide whether the item is
   misleading, SUPPORTING for material background, MINOR otherwise. Do not mark
   everything CENTRAL.
4. For each claim set checkability: CHECKABLE when supplied material or an
   authorized retrieval could settle it, NOT_CHECKABLE for pure opinion or
   prediction, PARTIALLY_CHECKABLE in between.
5. Assess only from material that is actually visible to you (SUPPLIED input or
   RETRIEVED tool output). Model memory is never verified evidence. If the
   material is insufficient, assessment is INSUFFICIENT_EVIDENCE and
   confidence must be low.
6. Give each claim a concise reason that names the exact evidence_id or states
   why the evidence is missing/insufficient.

## Guidance

- Do not merge different claims into one text blob. One row per atomic claim.
- Missing corroboration is not refutation. Style, emotion, grammar, platform,
   and category are not truth signals.
- claim_id is a short local id such as C1, C2 ... within this report.
- When you can neither support nor refute, prefer INSUFFICIENT_EVIDENCE over
   guessing; keep the confidence consistent with evidence quality.
