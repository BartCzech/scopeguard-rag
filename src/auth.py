"""JWT authentication for ScopeGuard retrieval server."""

from datetime import UTC, datetime, timedelta

import jwt

from models import UserContext

# In production this comes from env/secrets manager and settings.py or equivalent.
# Simplification for a research project.
SECRET = "scopeguard-dev-secret"
ALGORITHM = "HS256"
TOKEN_LIFETIME_HOURS = 24

# Hierarchy: each scope grants access to these document levels
SCOPE_TO_LEVELS: dict[str, list[str]] = {
    "docs:public": ["public"],
    "docs:internal": ["public", "internal"],
    "docs:confidential": ["public", "internal", "confidential"],
}


def create_token(user_id: str, scopes: list[str]) -> str:
    """Create a signed JWT. Stores raw scopes only - expansion happens at decode."""
    payload = {
        "sub": user_id,
        "scopes": scopes,
        "exp": datetime.now(UTC) + timedelta(hours=TOKEN_LIFETIME_HOURS),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)  # type: ignore[no-any-return]


def validate_token(token: str) -> UserContext:
    """Decode JWT, expand scopes to allowed_levels. Raises on expired/invalid (pyjwt)."""
    payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])

    scopes = payload.get("scopes", [])
    allowed: set[str] = set()
    for s in scopes:
        allowed.update(SCOPE_TO_LEVELS.get(s, []))

    return UserContext(
        user_id=payload["sub"],
        scopes=scopes,
        allowed_levels=sorted(allowed),
    )


def make_test_tokens() -> dict[str, str]:
    """
    Three fixed test tokens for evaluation. Same user IDs every time.
    Returns {"public_user": <token>, "internal_user": <token>, "confidential_user": <token>}
    """
    return {
        "public_user": create_token("user_public_01", ["docs:public"]),
        "internal_user": create_token("user_internal_01", ["docs:internal"]),
        "confidential_user": create_token("user_confidential_01", ["docs:confidential"]),
    }
