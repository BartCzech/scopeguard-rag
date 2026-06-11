"""Tests for auth module — JWT creation, validation, and scope expansion."""

import jwt
import pytest

from src.auth import (
    ALGORITHM,
    SECRET,
    create_token,
    make_test_tokens,
    validate_token,
)


@pytest.fixture
def public_token() -> str:
    return create_token("user-1", ["docs:public"])


@pytest.fixture
def expired_token() -> str:
    payload = {"sub": "user-1", "scopes": ["docs:public"], "exp": 0}
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)  # type: ignore[no-any-return]


@pytest.fixture
def wrong_secret_token() -> str:
    payload = {"sub": "user-1", "scopes": ["docs:public"]}
    return jwt.encode(payload, "wrong-secret", algorithm=ALGORITHM)  # type: ignore[no-any-return]


class TestCreateToken:
    def test_returns_string(self, public_token):
        assert isinstance(public_token, str)

    def test_contains_sub_claim(self, public_token):
        payload = jwt.decode(public_token, SECRET, algorithms=[ALGORITHM])
        assert payload["sub"] == "user-1"

    def test_stores_raw_scopes_not_expanded(self):
        token = create_token("user-1", ["docs:confidential"])
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        assert payload["scopes"] == ["docs:confidential"]

    def test_contains_exp_claim(self, public_token):
        payload = jwt.decode(public_token, SECRET, algorithms=[ALGORITHM])
        assert "exp" in payload


class TestValidateToken:
    @pytest.mark.parametrize(
        ("scopes", "expected_levels"),
        [
            (["docs:public"], ["public"]),
            (["docs:internal"], ["internal", "public"]),
            (["docs:confidential"], ["confidential", "internal", "public"]),
            (["docs:admin"], []),
            ([], []),
        ],
        ids=["public", "internal", "confidential", "unknown_scope", "empty_scopes"],
    )
    def test_scope_expansion(self, scopes, expected_levels):
        token = create_token("user-1", scopes)
        ctx = validate_token(token)
        assert ctx.allowed_levels == expected_levels

    def test_returns_correct_user_id(self, public_token):
        ctx = validate_token(public_token)
        assert ctx.user_id == "user-1"

    def test_expired_token_raises(self, expired_token):
        with pytest.raises(jwt.ExpiredSignatureError):
            validate_token(expired_token)

    def test_invalid_token_raises(self):
        with pytest.raises(jwt.InvalidTokenError):
            validate_token("not.a.real.token")

    def test_wrong_secret_raises(self, wrong_secret_token):
        with pytest.raises(jwt.InvalidSignatureError):
            validate_token(wrong_secret_token)


class TestMakeTestTokens:
    def test_returns_three_tokens(self):
        tokens = make_test_tokens()
        assert set(tokens.keys()) == {"public_user", "internal_user", "confidential_user"}

    @pytest.mark.parametrize(
        ("token_key", "expected_user_id", "expected_levels"),
        [
            ("public_user", "user_public_01", ["public"]),
            ("internal_user", "user_internal_01", ["internal", "public"]),
            ("confidential_user", "user_confidential_01", ["confidential", "internal", "public"]),
        ],
    )
    def test_token_decodes_correctly(self, token_key, expected_user_id, expected_levels):
        tokens = make_test_tokens()
        ctx = validate_token(tokens[token_key])
        assert ctx.user_id == expected_user_id
        assert ctx.allowed_levels == expected_levels
