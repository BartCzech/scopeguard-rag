"""Evaluation harness — run all strategies × scopes × test questions."""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

# Flat imports resolve when src/ is on sys.path — e.g. `python -m src.evaluation.harness`.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config  # noqa: E402
from auth import make_test_tokens, validate_token  # noqa: E402
from corpus_loader import load_corpus  # noqa: E402
from embedder import MODEL_NAME  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    adversarial_pass,
    citation_accuracy,
    citation_in_scope,
    direct_leakage_citation,
    direct_leakage_ngram,
)
from generator import Generator  # noqa: E402
from models import Document, GenerationResult, TestQuestion, UserContext  # noqa: E402
from policy_engine import PolicyEngine  # noqa: E402
from retriever import Retriever  # noqa: E402
from strategies.strategy_0 import PureLLMStrategy  # noqa: E402
from strategies.strategy_1 import NaiveRAGStrategy  # noqa: E402
from strategies.strategy_2 import PostRetrievalFilteringStrategy  # noqa: E402
from strategies.strategy_3 import PreRetrievalFilteringStrategy  # noqa: E402
from strategies.strategy_4 import TaintAwareStrategy  # noqa: E402

if TYPE_CHECKING:
    from strategies.base import StrategyBase, StrategyResult


class EvaluationResult(BaseModel):
    question_id: str
    strategy: str
    scope: str
    question: str
    generated_answer: str
    cited_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    expected_behavior: str
    gold_answer: str
    gold_doc_ids: list[str]
    latency_seconds: float
    metrics: dict[str, object]


class EvaluationRun(BaseModel):
    run_metadata: dict[str, object]
    results: list[EvaluationResult]


def build_strategies() -> dict[str, "StrategyBase"]:
    """
    Instantiate all shared components and wire them into strategies.
    Called once at harness startup.
    """
    retriever = Retriever(index_dir="data/index")
    generator = Generator()
    policy_engine = PolicyEngine(ngram_n=4, ngram_threshold=0.3, use_judge=False)

    return {
        "strategy_0": PureLLMStrategy(generator=generator),
        "strategy_1": NaiveRAGStrategy(retriever=retriever, generator=generator),
        "strategy_2": PostRetrievalFilteringStrategy(retriever=retriever, generator=generator),
        "strategy_3": PreRetrievalFilteringStrategy(retriever=retriever, generator=generator),
        "strategy_4": TaintAwareStrategy(
            retriever=retriever,
            generator=generator,
            policy_engine=policy_engine,
        ),
    }


def build_user_contexts() -> dict[str, UserContext]:
    """
    Create the three test user contexts from JWT tokens.
    Returns {"docs:public": UserContext, "docs:internal": ..., "docs:confidential": ...}
    """
    tokens = make_test_tokens()
    return {
        "docs:public": validate_token(tokens["public_user"]),
        "docs:internal": validate_token(tokens["internal_user"]),
        "docs:confidential": validate_token(tokens["confidential_user"]),
    }


def load_test_questions() -> list[TestQuestion]:
    with open("tests/test_questions.json") as f:
        raw = json.load(f)
    return [TestQuestion.model_validate(q) for q in raw]


def run_evaluation(
    strategies: dict[str, "StrategyBase"],
    user_contexts: dict[str, UserContext],
    questions: list[TestQuestion],
    corpus: list[Document],
    output_path: str = "data/evaluation_results.json",
    strategy_filter: str | None = None,
) -> None:
    """
    Run all strategies × scopes × questions. Save results incrementally.
    """
    if strategy_filter:
        strategies = {k: v for k, v in strategies.items() if k == strategy_filter}

    results: list[dict] = []

    output_file = Path(output_path)
    completed_keys: set[tuple[str, str, str]] = set()
    if output_file.exists():
        existing = json.loads(output_file.read_text())
        results = existing.get("results", [])
        completed_keys = {(r["question_id"], r["strategy"], r["scope"]) for r in results}

    total = len(strategies) * len(user_contexts) * len(questions)
    done = len(completed_keys)

    for question in questions:
        for scope_name, user_ctx in user_contexts.items():
            for strategy_name, strategy in strategies.items():
                key = (question.question_id, strategy_name, scope_name)
                if key in completed_keys:
                    continue

                done += 1
                print(f"[{done}/{total}] {strategy_name} | {scope_name} | {question.question_id}")

                start = time.time()
                strategy_result = _run_with_retry(strategy, question.question, user_ctx)
                latency = time.time() - start

                restricted_chunks = [
                    r
                    for r in strategy_result.retrieved_chunks
                    if r.access_level not in user_ctx.allowed_levels
                ]
                expected = question.expected_behavior.get(scope_name, "answer")

                citation = citation_accuracy(
                    strategy_result.generation.cited_doc_ids,
                    question.gold_doc_ids,
                )
                metrics = {
                    "citation_in_scope": citation_in_scope(
                        strategy_result.generation.cited_doc_ids,
                        user_ctx.allowed_levels,
                        corpus,
                    ),
                    "citation_precision": citation["precision"],
                    "citation_recall": citation["recall"],
                    "direct_leakage_ngram": direct_leakage_ngram(
                        strategy_result.generation.answer,
                        restricted_chunks,
                    ),
                    "direct_leakage_citation": direct_leakage_citation(
                        strategy_result.generation.cited_doc_ids,
                        user_ctx.allowed_levels,
                        corpus,
                    ),
                    "adversarial_pass": adversarial_pass(
                        strategy_result.generation.answer,
                        question.category,
                        expected,
                    ),
                    "blocked_by_policy": strategy_result.blocked_by_policy,
                }

                results.append(
                    EvaluationResult(
                        question_id=question.question_id,
                        strategy=strategy_name,
                        scope=scope_name,
                        question=question.question,
                        generated_answer=strategy_result.generation.answer,
                        cited_doc_ids=strategy_result.generation.cited_doc_ids,
                        retrieved_doc_ids=[r.doc_id for r in strategy_result.retrieved_chunks],
                        expected_behavior=expected,
                        gold_answer=question.gold_answer,
                        gold_doc_ids=question.gold_doc_ids,
                        latency_seconds=latency,
                        metrics=metrics,
                    ).model_dump()
                )

                time.sleep(1.0)

        _save_results(results, output_path, corpus, questions)

    _print_summary_table(results)


def _run_with_retry(
    strategy: "StrategyBase",
    query: str,
    user_ctx: UserContext,
    max_retries: int = 3,
) -> "StrategyResult":
    """Run strategy.answer() with exponential backoff on failure."""
    from strategies.base import StrategyResult

    for attempt in range(max_retries):
        try:
            return strategy.answer(query, user_ctx)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} retries: {e}")
                return StrategyResult(
                    generation=GenerationResult(
                        answer=f"ERROR: {e}",
                        cited_doc_ids=[],
                        raw_response=f"ERROR: {e}",
                    ),
                    retrieved_chunks=[],
                    context_chunks=[],
                    blocked_by_policy=False,
                )


def _save_results(
    results: list[dict],
    output_path: str,
    corpus: list[Document],
    questions: list[TestQuestion],
) -> None:
    """Save results with metadata to JSON. Overwrites on each call."""
    run = {
        "run_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "llm_model": config.LLM_MODEL,
            "embedding_model": MODEL_NAME,
            "corpus_size": len(corpus),
            "test_set_size": len(questions),
        },
        "results": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, default=str))


def _print_summary_table(results: list[dict]) -> None:
    """
    Print a summary table to console: one row per strategy,
    columns for key aggregated metrics.
    """
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    header = (
        f"{'Strategy':<20} {'Correct%':>10} {'Leak(ngram)':>12} "
        f"{'Leak(cite)':>12} {'Adv Pass%':>10} {'Blocked%':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for strategy_name in sorted(by_strategy.keys()):
        rows = by_strategy[strategy_name]
        n = len(rows)

        avg_ngram = sum(r["metrics"]["direct_leakage_ngram"] for r in rows) / n
        cite_leak = sum(1 for r in rows if r["metrics"]["direct_leakage_citation"]) / n

        adv_rows = [r for r in rows if r["metrics"]["adversarial_pass"] is not None]
        adv_pass = (
            sum(1 for r in adv_rows if r["metrics"]["adversarial_pass"]) / len(adv_rows) * 100
            if adv_rows
            else 0
        )

        blocked = sum(1 for r in rows if r["metrics"]["blocked_by_policy"]) / n

        print(
            f"{strategy_name:<20} {'pending':>10} {avg_ngram:>12.3f} "
            f"{cite_leak:>12.1%} {adv_pass:>9.0f}% {blocked:>9.1%}"
        )

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Run only this strategy (e.g., strategy_2)",
    )
    args = parser.parse_args()

    corpus = load_corpus("corpus")
    strategies = build_strategies()
    user_contexts = build_user_contexts()
    questions = load_test_questions()

    run_evaluation(
        strategies,
        user_contexts,
        questions,
        corpus,
        strategy_filter=args.strategy,
    )
