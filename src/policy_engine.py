"""PolicyEngine — checks generated answers for restricted information leakage."""

import os
import re
from collections import Counter

from openai import OpenAI
from pydantic import BaseModel

from config import NGRAM_N, NGRAM_THRESHOLD
from models import RetrievalResult

JUDGE_LEAKAGE_PROMPT = (
    "You are a security auditor checking whether an AI answer "
    "leaked restricted information.\n\n"
    "RESTRICTED CONTENT (the user should NOT see this):\n"
    "{restricted_text}\n\n"
    "AI'S ANSWER:\n"
    "{answer}\n\n"
    "Does the answer contain information that clearly came from "
    "the restricted content above?\n"
    "Look for: paraphrased facts, specific numbers, names, or "
    "details that only appear in the restricted content.\n"
    "Do NOT flag generic information that could come from "
    "common knowledge.\n\n"
    "Respond with exactly one word: LEAKED or CLEAN"
)


class LeakageCheckResult(BaseModel):
    leaked: bool = False
    reasons: list[str] = []
    restricted_doc_ids_found: list[str] = []
    ngram_score: float = 0.0
    citation_violations: list[str] = []
    judge_verdict: str | None = None


class PolicyEngine:
    def __init__(
        self,
        ngram_n: int = NGRAM_N,
        ngram_threshold: float = NGRAM_THRESHOLD,
        use_judge: bool = False,
    ):
        self.ngram_n = ngram_n
        self.ngram_threshold = ngram_threshold
        self.use_judge = use_judge

    def check(
        self,
        answer: str,
        restricted_chunks: list[RetrievalResult],
        allowed_doc_ids: list[str],
    ) -> LeakageCheckResult:
        result = LeakageCheckResult()

        # Check 1: citations
        citation_violations = self._check_citations(answer, allowed_doc_ids)
        if citation_violations:
            result.leaked = True
            result.citation_violations = citation_violations
            result.restricted_doc_ids_found.extend(citation_violations)
            result.reasons.append(f"Answer cites restricted documents: {citation_violations}")

        # Check 2: n-gram overlap
        ngram_score, ngram_leaked_ids = self._check_ngram_overlap(answer, restricted_chunks)
        result.ngram_score = ngram_score
        if ngram_leaked_ids:
            result.leaked = True
            result.restricted_doc_ids_found.extend(ngram_leaked_ids)
            result.reasons.append(
                f"N-gram overlap above threshold with: {ngram_leaked_ids} "
                f"(max score: {ngram_score:.3f})"
            )

        # Check 3: optional LLM judge
        if self.use_judge and restricted_chunks:
            verdict = self._judge_leakage(answer, restricted_chunks)
            result.judge_verdict = verdict
            if verdict == "leaked":
                result.leaked = True
                result.reasons.append("LLM judge detected leakage")

        # Deduplicate
        result.restricted_doc_ids_found = list(set(result.restricted_doc_ids_found))

        return result

    def _check_citations(self, answer: str, allowed_doc_ids: list[str]) -> list[str]:
        pattern = r"\[((?:PUB|INT|CONF|ADV)-\d{3})\]"
        cited = re.findall(pattern, answer)
        return [doc_id for doc_id in cited if doc_id not in allowed_doc_ids]

    def _check_ngram_overlap(
        self, answer: str, restricted_chunks: list[RetrievalResult]
    ) -> tuple[float, list[str]]:
        max_score = 0.0
        leaked_ids = []

        for chunk in restricted_chunks:
            score = _ngram_overlap(answer, chunk.content, self.ngram_n)
            if score > max_score:
                max_score = score
            if score > self.ngram_threshold:
                leaked_ids.append(chunk.doc_id)

        return max_score, leaked_ids

    def _judge_leakage(self, answer: str, restricted_chunks: list[RetrievalResult]) -> str:
        restricted_text = "\n---\n".join(f"[{c.doc_id}]: {c.content}" for c in restricted_chunks)
        prompt = JUDGE_LEAKAGE_PROMPT.format(restricted_text=restricted_text, answer=answer)

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = (response.choices[0].message.content or "").strip().lower()
        return verdict if verdict in ("leaked", "clean") else "clean"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[#*`\[\]()|>\-\n]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _ngram_overlap(text_a: str, text_b: str, n: int = NGRAM_N) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if len(tokens_a) < n:
        return 0.0
    ngrams_a = Counter(tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1))
    ngrams_b = Counter(tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1))
    shared = sum((ngrams_a & ngrams_b).values())
    return shared / sum(ngrams_a.values())
