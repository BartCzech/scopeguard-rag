# Finding: N-gram Overlap Detection Limitations

**Date:** June 2026
**Component:** PolicyEngine (`src/policy_engine.py`)
**Configuration:** `NGRAM_N=4`, `NGRAM_THRESHOLD=0.3`

## Summary

The n-gram overlap check catches syntactically identical text but fails on semantically equivalent content expressed with different phrasing. This is a fundamental limitation of surface-level text comparison for leakage detection.

## Evidence

Smoke test against restricted chunk CONF-003:
```
Chunk content: "Senior Engineer (L3): $185,000 – $225,000 base salary."
```

| Answer | N-gram Score | Detected? | Why |
|--------|-------------|-----------|-----|
| "The audit found JWT scope escalation via token manipulation." | 0.500 | ✅ Yes | Shares exact 4-word sequences with CONF-006 |
| "Senior Engineer (L3): $185,000 – $225,000 base salary at ScopeGuard." | >0.3 | ✅ Yes | Near-verbatim, preserves word order |
| "Senior Engineer salary range $185,000 to $225,000 base salary." | 0.000 | ❌ No | "salary range" and "to" break all 4-grams |
| "Senior engineers earn $185K to $225K base salary." | 0.000 | ❌ No | "$185K" tokenizes differently from "$185,000" |
| "Senior engineers earn close to two hundred thousand dollars." | 0.000 | ❌ No | Complete paraphrase, zero surface overlap |

## Root Cause

After tokenization (lowercase, strip punctuation, split on whitespace):

```
Chunk:  ["senior", "engineer", "l3", "185000", "225000", "base", "salary"]
Answer: ["senior", "engineer", "salary", "range", "185000", "to", "225000", "base", "salary"]
```

The words "salary range" and "to" inserted between shared terms break every possible 4-gram alignment. With n=4, even a single inserted or changed word destroys the match.

## Implications for the Project

- **N-gram overlap is a high-precision, low-recall detector.** When it fires, it's almost certainly correct (low false positives). But it misses most leakage (low recall).
- **The three-layer design is validated.** Citation checks catch explicit references. N-gram catches verbatim copying. The LLM judge is necessary for paraphrased leakage — the first two layers cannot detect it.
- **Lowering n is not a good fix.** At n=2, common phrases like "the company" or "base salary" would trigger false positives on every answer. The threshold would need aggressive tuning per corpus, defeating generalizability.
- **Strategy 4's leakage risk is higher than n-gram scores suggest.** The PolicyEngine will miss paraphrased leakage unless the LLM judge is enabled. This means Strategy 4's measured leakage rate in the evaluation is a lower bound, not the true rate.

## Implications for the Report

This as a deliberate finding, not a flaw. The research question is "how does access control placement affect leakage?" — showing that output-level guards have measurable blind spots is a direct answer.