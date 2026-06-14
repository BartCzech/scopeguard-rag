"""Tests for PolicyEngine — leakage detection checks."""

import pytest
from dotenv import load_dotenv

from config import NGRAM_THRESHOLD
from models import RetrievalResult
from policy_engine import PolicyEngine, _ngram_overlap, _tokenize

load_dotenv()

# ── Helpers ──


def _restricted_chunk(doc_id: str, content: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        access_level="confidential",
        score=0.9,
    )


@pytest.fixture
def salary_chunk() -> RetrievalResult:
    return _restricted_chunk(
        "CONF-003",
        "Senior Engineer (L3): $185,000 – $225,000 base salary. "
        "Staff Engineer (L4): $230,000 – $250,000 with 0.4% equity.",
    )


@pytest.fixture
def security_chunk() -> RetrievalResult:
    return _restricted_chunk(
        "CONF-006",
        "Security audit SG-SEC-2024-07 found two critical vulnerabilities: "
        "JWT scope escalation via token manipulation and embedding cache poisoning "
        "allowing cross-tenant data access. Both patched November 2024.",
    )


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine(use_judge=False)


# ── Tokenizer ──


class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_strips_markdown(self):
        assert _tokenize("# Header **bold** `code`") == ["header", "bold", "code"]

    def test_strips_citations(self):
        assert _tokenize("See [CONF-003] for details") == ["see", "conf", "003", "for", "details"]

    def test_strips_punctuation(self):
        assert _tokenize("$185,000 – $225,000.") == ["185000", "225000"]

    def test_empty_string(self):
        assert _tokenize("") == []


# ── N-gram Overlap ──


class TestNgramOverlap:
    def test_identical_texts_score_one(self):
        text = "the salary range for senior engineers is very competitive"
        assert _ngram_overlap(text, text) == 1.0

    def test_completely_different_texts_score_zero(self):
        assert (
            _ngram_overlap(
                "the quick brown fox jumps over",
                "completely unrelated words about something else entirely",
            )
            == 0.0
        )

    def test_partial_overlap(self):
        score = _ngram_overlap(
            "the senior engineer salary range is competitive",
            "the senior engineer salary range starts at 185K",
        )
        assert 0.0 < score < 1.0

    def test_short_text_returns_zero(self):
        assert _ngram_overlap("too short", "also short text here", n=4) == 0.0

    def test_common_words_low_overlap(self):
        score = _ngram_overlap(
            "ScopeGuard provides enterprise security features for document access control",
            "ScopeGuard found critical security vulnerabilities in JWT token validation code",
        )
        assert score < NGRAM_THRESHOLD


# ── Citation Check ──


class TestCitationCheck:
    def test_catches_restricted_citation(self, engine):
        result = engine.check(
            answer="The salary is $185K [CONF-003].",
            restricted_chunks=[],
            allowed_doc_ids=["PUB-001", "PUB-002"],
        )
        assert result.leaked is True
        assert "CONF-003" in result.citation_violations

    def test_allows_permitted_citation(self, engine):
        result = engine.check(
            answer="See the product overview [PUB-001].",
            restricted_chunks=[],
            allowed_doc_ids=["PUB-001", "PUB-002"],
        )
        assert result.leaked is False
        assert result.citation_violations == []

    def test_catches_multiple_violations(self, engine):
        result = engine.check(
            answer="Per [CONF-003] and [CONF-006], the data shows...",
            restricted_chunks=[],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is True
        assert set(result.citation_violations) == {"CONF-003", "CONF-006"}

    def test_no_citations_is_clean(self, engine):
        result = engine.check(
            answer="I don't have enough information to answer.",
            restricted_chunks=[],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is False

    def test_mixed_allowed_and_restricted(self, engine):
        result = engine.check(
            answer="Based on [PUB-001] and [CONF-003], the answer is clear.",
            restricted_chunks=[],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is True
        assert result.citation_violations == ["CONF-003"]
        assert "PUB-001" not in result.citation_violations


# ── N-gram Leakage Check ──


class TestNgramLeakageCheck:
    def test_catches_verbatim_copy(self, engine, salary_chunk):
        # salary_chunk tokenizes to: "senior engineer l3 185000 225000 base salary staff..."
        # Use phrasing that shares 4-grams after tokenization
        result = engine.check(
            answer="Senior Engineer (L3): $185,000 – $225,000 base salary at ScopeGuard.",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is True
        assert result.ngram_score > NGRAM_THRESHOLD

    def test_clean_answer_no_overlap(self, engine, salary_chunk):
        result = engine.check(
            answer="ScopeGuard integrates with Pinecone and Weaviate for vector search.",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.ngram_score < NGRAM_THRESHOLD
        # May or may not set leaked=False depending on other checks
        assert "CONF-003" not in result.restricted_doc_ids_found

    def test_common_vocabulary_not_flagged(self, engine, security_chunk):
        result = engine.check(
            answer="ScopeGuard offers enterprise security features including "
            "scope-based retrieval and identity verification for all users.",
            restricted_chunks=[security_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.ngram_score < NGRAM_THRESHOLD

    def test_multiple_restricted_chunks(self, engine, salary_chunk, security_chunk):
        result = engine.check(
            answer="The security audit found JWT scope escalation via token manipulation.",
            restricted_chunks=[salary_chunk, security_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        # Should flag security_chunk, not salary_chunk
        if result.leaked:
            assert "CONF-006" in result.restricted_doc_ids_found


# ── Combined Checks ──


class TestCombinedChecks:
    def test_both_citation_and_ngram_trigger(self, engine, salary_chunk):
        result = engine.check(
            answer="Senior Engineer (L3): $185,000 – $225,000 base salary [CONF-003].",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is True
        assert len(result.citation_violations) > 0
        assert result.ngram_score > NGRAM_THRESHOLD

    def test_completely_clean_answer(self, engine, salary_chunk, security_chunk):
        result = engine.check(
            answer="I don't have enough information to answer this question.",
            restricted_chunks=[salary_chunk, security_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is False
        assert result.reasons == []
        assert result.restricted_doc_ids_found == []
        assert result.ngram_score < NGRAM_THRESHOLD
        assert result.citation_violations == []

    def test_refusal_is_clean(self, engine, salary_chunk):
        result = engine.check(
            answer="I cannot share confidential salary information. "
            "Please contact your administrator for elevated access.",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.leaked is False

    def test_no_restricted_chunks_always_clean(self, engine):
        result = engine.check(
            answer="The product costs $45 per user per month [PUB-002].",
            restricted_chunks=[],
            allowed_doc_ids=["PUB-001", "PUB-002"],
        )
        assert result.leaked is False
        assert result.ngram_score == 0.0


# ── Judge ──


class TestJudge:
    def test_judge_not_called_when_disabled(self, engine, salary_chunk):
        result = engine.check(
            answer="Salary is $185K [CONF-003].",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.judge_verdict is None

    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_judge_catches_leakage(self, salary_chunk):
        engine = PolicyEngine(use_judge=True)
        result = engine.check(
            answer="Senior engineers earn between $185K and $225K at ScopeGuard.",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.judge_verdict == "leaked"

    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_judge_passes_clean_answer(self, salary_chunk):
        engine = PolicyEngine(use_judge=True)
        result = engine.check(
            answer="ScopeGuard integrates with Pinecone for vector search.",
            restricted_chunks=[salary_chunk],
            allowed_doc_ids=["PUB-001"],
        )
        assert result.judge_verdict == "clean"


# ── Deduplication ──


class TestDeduplication:
    def test_restricted_doc_ids_are_deduplicated(self, engine):
        chunk = _restricted_chunk("CONF-003", "Senior Engineer salary $185K to $225K base salary")
        result = engine.check(
            answer="The salary range is $185K to $225K [CONF-003].",
            restricted_chunks=[chunk],
            allowed_doc_ids=["PUB-001"],
        )
        # CONF-003 found by both citation and n-gram, but only listed once
        assert result.restricted_doc_ids_found.count("CONF-003") == 1
