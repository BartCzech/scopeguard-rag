"""Shared fixtures for all tests. Auto-discovered by pytest."""

import pytest
from fakes import make_chunk

from models import UserContext


@pytest.fixture
def mixed_results():
    """Retrieval results with mixed access levels, sorted by score descending."""
    return [
        make_chunk("CONF-003", "confidential", 0.95),
        make_chunk("PUB-001", "public", 0.90),
        make_chunk("INT-001", "internal", 0.85),
        make_chunk("PUB-002", "public", 0.80),
        make_chunk("CONF-006", "confidential", 0.75),
        make_chunk("PUB-003", "public", 0.70),
        make_chunk("INT-002", "internal", 0.65),
    ]


@pytest.fixture
def public_user():
    return UserContext(
        user_id="user-1",
        scopes=["docs:public"],
        allowed_levels=["public"],
    )


@pytest.fixture
def internal_user():
    return UserContext(
        user_id="user-2",
        scopes=["docs:internal"],
        allowed_levels=["internal", "public"],
    )


@pytest.fixture
def confidential_user():
    return UserContext(
        user_id="user-3",
        scopes=["docs:confidential"],
        allowed_levels=["confidential", "internal", "public"],
    )
