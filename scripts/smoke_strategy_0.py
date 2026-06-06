#!/usr/bin/env python3
"""Manual smoke test for Strategy 0 (Pure LLM). Requires OPENAI_API_KEY."""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from generator import Generator  # noqa: E402
from models import UserContext  # noqa: E402
from strategies.strategy_0 import PureLLMStrategy  # noqa: E402

SCOPES = {
    "public": UserContext(
        user_id="smoke-public",
        scopes=["docs:public"],
        allowed_levels=["public"],
    ),
    "internal": UserContext(
        user_id="smoke-internal",
        scopes=["docs:public", "docs:internal"],
        allowed_levels=["public", "internal"],
    ),
    "confidential": UserContext(
        user_id="smoke-confidential",
        scopes=["docs:public", "docs:internal", "docs:confidential"],
        allowed_levels=["public", "internal", "confidential"],
    ),
}

QUERY = "What is ScopeGuard's burn rate?"
CORPUS_ANSWER = "$480K/month"


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this smoke test.")

    strategy = PureLLMStrategy(Generator())

    print(f"Query: {QUERY!r}\n")
    print(f"Corpus ground truth (Strategy 0 must NOT know this): {CORPUS_ANSWER}\n")
    print("-" * 60)

    answers: list[str] = []
    for level, user_context in SCOPES.items():
        result = strategy.answer(QUERY, user_context)
        answer = result.generation.answer
        answers.append(answer)

        leaked = CORPUS_ANSWER.lower().replace(",", "") in answer.lower().replace(",", "")
        print(f"\n[{level}]")
        print(f"  retrieved_chunks: {len(result.retrieved_chunks)}")
        print(f"  cited_doc_ids:    {result.generation.cited_doc_ids}")
        print(f"  answer:           {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print(f"  leaked corpus:    {leaked}")

    identical = len(set(answers)) == 1
    print("\n" + "-" * 60)
    print(f"Identical across scopes: {identical}")
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
