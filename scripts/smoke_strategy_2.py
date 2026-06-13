"""Smoke test for Strategy 2 — Post-Retrieval Filtering. Requires FAISS index and OPENAI_API_KEY."""

import sys

from auth import make_test_tokens, validate_token
from generator import Generator
from retriever import Retriever
from strategies.strategy_2 import PostRetrievalFilteringStrategy

sys.path.insert(0, "src")  # noqa: E702

from dotenv import load_dotenv

load_dotenv()


def main():
    tokens = make_test_tokens()
    retriever = Retriever()
    generator = Generator()
    strategy = PostRetrievalFilteringStrategy(retriever, generator)

    # Test 1: Public user asks about salary (confidential topic)
    print("=" * 60)
    print("TEST 1: Public user → salary question")
    print("=" * 60)
    ctx = validate_token(tokens["public_user"])
    result = strategy.answer("What is the salary range for senior engineers?", ctx)

    print(f"\nAnswer: {result.generation.answer}")
    print(f"Citations: {result.generation.cited_doc_ids}")
    print(f"Blocked by policy: {result.blocked_by_policy}")

    print(f"\nRetrieved chunks ({len(result.retrieved_chunks)}):")
    for r in result.retrieved_chunks:
        print(f"  {r.doc_id} ({r.access_level}) score={r.score:.3f}")

    print(f"\nContext chunks ({len(result.context_chunks)}):")
    for r in result.context_chunks:
        print(f"  {r.doc_id} ({r.access_level}) score={r.score:.3f}")

    context_levels = {r.access_level for r in result.context_chunks}
    assert "confidential" not in context_levels, "LEAK: confidential chunks in context!"
    print("\n✅ No confidential chunks in context")

    # Test 2: Confidential user asks the same question
    print("\n" + "=" * 60)
    print("TEST 2: Confidential user → same salary question")
    print("=" * 60)
    ctx = validate_token(tokens["confidential_user"])
    result = strategy.answer("What is the salary range for senior engineers?", ctx)

    print(f"\nAnswer: {result.generation.answer}")
    print(f"Citations: {result.generation.cited_doc_ids}")

    print(f"\nContext chunks ({len(result.context_chunks)}):")
    for r in result.context_chunks:
        print(f"  {r.doc_id} ({r.access_level}) score={r.score:.3f}")

    context_ids = {r.doc_id for r in result.context_chunks}
    assert "CONF-003" in context_ids, "Expected CONF-003 in context for confidential user"
    print("\n✅ CONF-003 found in context")

    # Test 3: Verify retrieved ≠ context for public user
    print("\n" + "=" * 60)
    print("TEST 3: Public user — retrieved vs context differ")
    print("=" * 60)
    ctx = validate_token(tokens["public_user"])
    result = strategy.answer("What did the security audit find?", ctx)

    retrieved_levels = {r.access_level for r in result.retrieved_chunks}
    context_levels = {r.access_level for r in result.context_chunks}

    print(f"Retrieved levels: {retrieved_levels}")
    print(f"Context levels: {context_levels}")

    if retrieved_levels != context_levels:
        print("✅ Filtering removed restricted chunks from context")
    else:
        print("⚠️  Same levels in both — no restricted chunks were retrieved")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
