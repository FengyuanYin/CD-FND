---
name: evidence_assessment
description: "Weigh each supplied piece of evidence against the claims: origin, independence, temporal fit, and support/refute mapping."
version: "1.0.0"
allowed_tools: []
---

# Evidence Assessment

Your task is the evidence layer of the analysis: decide what each visible piece
of evidence is worth, which claims it supports or refutes, and what its limits
are, under the unified specialist_report_v1 contract.

## Procedure

1. Inventory every evidence-like entry that is actually visible: full article
   text, snippets, titles, metadata blocks, retrieval results, or fields named
   evidence / evidence_count. Titles and publisher names alone are leads, not
   proof; record that in limitations when the body is absent.
2. For every evidence entry create an evidence object:
   - origin: SUPPLIED means it came inside the input; RETRIEVED means an
     authorized tool actually returned it. Model memory is never evidence.
   - source, published_at only when visible; otherwise null.
   - content_used: a short excerpt or exact description of what you read.
   - temporal_fit: VALID when the material is contemporaneous with the claim;
     RETROSPECTIVE when written later about the claim; UNKNOWN otherwise.
3. Map each claim to evidence: supports_claim_ids / refutes_claim_ids using the
   claim ids from the other reports when provided, otherwise state the claim
   text fragment. Repeated syndication of one original counts as one source
   line; say so in limitations.
4. Assess each claim that has material: SUPPORTED / REFUTED / MIXED /
   INSUFFICIENT_EVIDENCE with a confidence consistent with the actual
   material quality, not the number of hits.
5. Record independence limits (same-origin echoes) and temporal limits in the
   evidence limitations and in your report limitations.

## Guidance

- Missing corroboration is not refutation.
- A snippet may be truncated or misleading; never over-weight a headline.
- Do not read a verdict into the absence of evidence; set
   INSUFFICIENT_EVIDENCE and keep confidence low.
