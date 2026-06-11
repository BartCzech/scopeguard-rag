"""Integration tests for MCP server tools. Requires built FAISS index."""

import pytest

from src.auth import make_test_tokens
from src.mcp_server import list_accessible_documents, read_document, search_documents


@pytest.fixture
def tokens() -> dict[str, str]:
    return make_test_tokens()


@pytest.fixture
def public_token(tokens) -> str:
    return tokens["public_user"]  # type: ignore[no-any-return]


@pytest.fixture
def confidential_token(tokens) -> str:
    return tokens["confidential_user"]  # type: ignore[no-any-return]


INVALID_TOKEN = "garbage-token"


class TestListAccessibleDocuments:
    @pytest.mark.parametrize(
        ("token_key", "expected_levels"),
        [
            ("public_user", {"public"}),
            ("internal_user", {"public", "internal"}),
            ("confidential_user", {"public", "internal", "confidential"}),
        ],
    )
    def test_returns_correct_access_levels(self, tokens, token_key, expected_levels):
        results = list_accessible_documents(tokens[token_key])
        levels = {r["access_level"] for r in results}
        assert levels == expected_levels

    def test_results_contain_required_keys(self, public_token):
        results = list_accessible_documents(public_token)
        assert len(results) > 0
        for r in results:
            assert "doc_id" in r
            assert "title" in r
            assert "access_level" in r

    def test_invalid_token_returns_error(self):
        results = list_accessible_documents(INVALID_TOKEN)
        assert results[0]["error"] == "invalid_token"


class TestReadDocument:
    @pytest.mark.parametrize(
        ("doc_id", "token_key", "expect_accessible"),
        [
            ("PUB-001", "public_user", True),
            ("PUB-001", "confidential_user", True),
            ("CONF-003", "confidential_user", True),
            ("CONF-003", "public_user", False),
            ("CONF-003", "internal_user", False),
            ("INT-001", "internal_user", True),
            ("INT-001", "public_user", False),
        ],
        ids=[
            "public_reads_public",
            "confidential_reads_public",
            "confidential_reads_confidential",
            "public_blocked_from_confidential",
            "internal_blocked_from_confidential",
            "internal_reads_internal",
            "public_blocked_from_internal",
        ],
    )
    def test_access_enforcement(self, tokens, doc_id, token_key, expect_accessible):
        result = read_document(doc_id, tokens[token_key])
        if expect_accessible:
            assert result["doc_id"] == doc_id
            assert "content" in result
        else:
            assert result == {"error": "not_accessible"}

    def test_nonexistent_document(self, confidential_token):
        result = read_document("FAKE-999", confidential_token)
        assert result == {"error": "not_accessible"}

    def test_concealment_same_error_for_missing_and_denied(self, public_token):
        """Security: 'not found' and 'access denied' return the same error."""
        missing = read_document("FAKE-999", public_token)
        denied = read_document("CONF-003", public_token)
        assert missing == denied

    def test_invalid_token_returns_error(self):
        result = read_document("PUB-001", INVALID_TOKEN)
        assert result["error"] == "invalid_token"


class TestSearchDocuments:
    @pytest.mark.parametrize("filtering_mode", ["pre", "post"])
    def test_filtered_modes_return_only_allowed(self, public_token, filtering_mode):
        results = search_documents(
            query="salary", top_k=5, token=public_token, filtering_mode=filtering_mode
        )
        for r in results:
            assert r["access_level"] == "public"

    def test_labeled_mode_may_return_restricted(self, public_token):
        results = search_documents(
            query="salary", top_k=5, token=public_token, filtering_mode="labeled"
        )
        assert len(results) > 0
        # Labeled mode returns everything — we just verify it doesn't crash

    def test_confidential_user_finds_conf003(self, confidential_token):
        results = search_documents(
            query="salary bands compensation",
            top_k=5,
            token=confidential_token,
            filtering_mode="pre",
        )
        doc_ids = {r["doc_id"] for r in results}
        assert "CONF-003" in doc_ids

    def test_results_contain_required_keys(self, public_token):
        results = search_documents(
            query="ScopeGuard", top_k=3, token=public_token, filtering_mode="pre"
        )
        assert len(results) > 0
        for r in results:
            assert "doc_id" in r
            assert "content" in r
            assert "access_level" in r
            assert "score" in r

    def test_invalid_filtering_mode_raises(self, public_token):
        with pytest.raises(ValueError, match="Invalid filtering mode"):
            search_documents(query="test", top_k=5, token=public_token, filtering_mode="invalid")

    def test_invalid_token_returns_error(self):
        results = search_documents(query="test", top_k=5, token=INVALID_TOKEN, filtering_mode="pre")
        assert results[0]["error"] == "invalid_token"
