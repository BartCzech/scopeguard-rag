"""Smoke test for PolicyEngine — visual verification of leakage detection."""

import sys

sys.path.insert(0, "src")  # noqa: E702

from models import RetrievalResult
from policy_engine import PolicyEngine


def _chunk(doc_id: str, content: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        access_level="confidential",
        score=0.9,
    )


def main():
    engine = PolicyEngine(use_judge=False)

    salary = _chunk(
        "CONF-003",
        "Senior Engineer (L3): $185,000 – $225,000 base salary. "
        "Staff Engineer (L4): $230,000 – $250,000 with 0.4% equity. "
        "Founders (Bart Check, Paul Kepovitz): $195,000 each.",
    )
    security = _chunk(
        "CONF-006",
        "Security audit SG-SEC-2024-07 found two critical vulnerabilities: "
        "JWT scope escalation via token manipulation and embedding cache poisoning "
        "allowing cross-tenant data access.",
    )

    allowed = ["PUB-001", "PUB-002", "PUB-003"]
    restricted = [salary, security]

    tests = [
        (
            "Direct citation leak",
            "The salary is $185K–$225K [CONF-003].",
            True,
        ),
        (
            "Verbatim copy without citation",
            "Senior Engineer salary range $185,000 to $225,000 base salary.",
            None,
        ),
        (
            "Paraphrased (may or may not catch)",
            "Senior engineers earn close to two hundred thousand dollars.",
            None,  # Unknown — depends on n-gram threshold
        ),
        (
            "Clean refusal",
            "I don't have enough information to answer this question.",
            False,
        ),
        (
            "Clean answer from public docs",
            "ScopeGuard integrates with Pinecone and Weaviate for vector search [PUB-001].",
            False,
        ),
        (
            "Security vocabulary overlap (false positive test)",
            "ScopeGuard offers enterprise security features including "
            "scope-based retrieval and identity verification.",
            False,
        ),
        (
            "Security audit verbatim",
            "The audit found JWT scope escalation via token manipulation.",
            True,
        ),
        (
            "Mixed: clean citation + leaked content",
            "Per [PUB-001], senior engineers earn $185K to $225K base salary.",
            None,
        ),
    ]

    print("PolicyEngine Smoke Tests")
    print("=" * 70)

    for name, answer, expected in tests:
        result = engine.check(answer, restricted, allowed)

        if expected is None:
            status = "⚠️  BORDERLINE"
        elif result.leaked == expected:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"\n{status} — {name}")
        print(f"  Answer: {answer[:80]}...")
        print(f"  Leaked: {result.leaked}")
        print(f"  N-gram score: {result.ngram_score:.3f}")
        print(f"  Citation violations: {result.citation_violations}")
        if result.reasons:
            for reason in result.reasons:
                print(f"  Reason: {reason}")

    print("\n" + "=" * 70)
    print("Review borderline cases manually. Adjust NGRAM_THRESHOLD if needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
