---
name: temporal_reasoning
description: "Check dates, event ordering, timeliness and evidence-vs-claim time alignment, and flag time-related inconsistencies."
version: "1.0.0"
allowed_tools: []
---

# Temporal Reasoning

Your task is the time layer of the analysis: verify temporal consistency of the
claims and the supplied material, under the unified specialist_report_v1
contract.

## Procedure

1. List every time reference in the text: explicit dates, relative times
   ("yesterday", "last week", "in the coming year"), version numbers of
   reports, publication years, and event ordering statements.
2. Separate these distinct values and never mix them:
   - claim time: when the event is claimed to happen;
   - publication time: when the item was written;
   - evidence time: when each evidence entry was created/published;
   - retrieval/evaluation cutoff: the moment the analysis runs.
3. For each claim that carries time references, assess temporal consistency:
   - evidence that is contemporaneous with the claim is VALID;
   - evidence written long after the fact may still be valid but must be marked
     RETROSPECTIVE and weighted less;
   - material about a different period than the claim cannot support it.
4. Flag classic inconsistencies: anachronistic entities/technologies, event
   ordering that contradicts the timeline, "as of" dates that predate the
   reported fact, or evidence published after the event claiming foresight.
5. Record findings as claims/evidence entries in specialist_report_v1 with
   assessment SUPPORTED / REFUTED / MIXED / INSUFFICIENT_EVIDENCE and a reason
   naming the concrete time mismatch.

## Guidance

- Do not call a claim false merely because it is old news; recency is not
   veracity and staleness is not fabrication.
- Do not use the dataset split name, snapshot label (-3/0/+3), or
   evidence_count as a fact signal.
- When times cannot be established, say so and use INSUFFICIENT_EVIDENCE.
