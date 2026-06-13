"""Smoke test for Strategy 3 — Pre-Retrieval Filtering. Requires FAISS index and OPENAI_API_KEY."""

import sys

from auth import make_test_tokens, validate_token
from generator import Generator
from retriever import Retriever
from strategies.strategy_3 import PreRetrievalFilteringStrategy

sys.path.insert(0, "src")  # noqa: E702

from dotenv import load_dotenv

load_dotenv()


def main():
    tokens = make_test_tokens()
    retriever = Retriever()
    generator = Generator()
    strategy = PreRetrievalFilteringStrategy(retriever, generator)

    # Test 1: Public user asks about salary
    print("=" * 60)
    print("TEST 1: Public user → salary question (pre-filtered)")
    print("=" * 60)
    ctx = validate_token(tokens["public_user"])
    result = strategy.answer("What is the salary range for senior engineers?", ctx)

    print(f"\nAnswer: {result.generation.answer}")
    print(f"Citations: {result.generation.cited_doc_ids}")

    print(f"\nRetrieved chunks ({len(result.retrieved_chunks)}):")
    for r in result.retrieved_chunks:
        print(f"  {r.doc_id} ({r.access_level}) score={r.score:.3f}")

    # Verify: no restricted chunks anywhere
    for r in result.retrieved_chunks:
        assert r.access_level == "public", f"LEAK: {r.doc_id} ({r.access_level}) in retrieved!"
    assert result.retrieved_chunks == result.context_chunks
    print("\n✅ Only public chunks retrieved. Retrieved == Context.")

    # Test 2: Confidential user asks the same question
    print("\n" + "=" * 60)
    print("TEST 2: Confidential user → same salary question")
    print("=" * 60)
    ctx = validate_token(tokens["confidential_user"])
    result = strategy.answer("What is the salary range for senior engineers?", ctx)

    print(f"\nAnswer: {result.generation.answer}")
    print(f"Citations: {result.generation.cited_doc_ids}")

    print(f"\nRetrieved chunks ({len(result.retrieved_chunks)}):")
    for r in result.retrieved_chunks:
        print(f"  {r.doc_id} ({r.access_level}) score={r.score:.3f}")

    context_ids = {r.doc_id for r in result.context_chunks}
    assert "CONF-003" in context_ids, "Expected CONF-003 for confidential user"
    print("\n✅ CONF-003 found in context")

    # Test 3: Compare retrieval scores with Strategy 2 output
    print("\n" + "=" * 60)
    print("TEST 3: Public user — compare with Strategy 2 ranking")
    print("=" * 60)
    ctx = validate_token(tokens["public_user"])
    result = strategy.answer("What did the security audit find?", ctx)

    print(f"\nAnswer: {result.generation.answer}")
    print(f"\nRetrieved chunks ({len(result.retrieved_chunks)}):")
    for r in result.retrieved_chunks:
        print(f"  {r.doc_id} ({r.access_level}) score={r.score:.3f}")

    for r in result.retrieved_chunks:
        assert r.access_level == "public"
    print("\n✅ All retrieved chunks are public")
    print("Compare these scores against Strategy 2's context scores for the same query.")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
