from unittest.mock import MagicMock, patch

import config
from generator import Generator
from models import RetrievalResult


def _make_result(doc_id: str, access_level: str, content: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        access_level=access_level,
        score=0.9,
    )


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("generator.OpenAI")
def test_format_context_includes_doc_id_and_access_level(mock_openai_cls: MagicMock) -> None:
    mock_openai_cls.return_value = MagicMock()
    context = [
        _make_result("PUB-001", "public", "Product overview text."),
        _make_result("CONF-003", "confidential", "Salary band details."),
    ]
    formatted = Generator()._format_context(context)
    assert "[PUB-001] (public)" in formatted
    assert "Product overview text." in formatted
    assert "[CONF-003] (confidential)" in formatted
    assert "Salary band details." in formatted
    assert formatted.endswith("---")


def test_parse_citations_extracts_doc_ids() -> None:
    text = "ScopeGuard is middleware [PUB-001] and also [PUB-001] again [CONF-003]."
    cited = Generator._parse_citations(text)
    assert set(cited) == {"PUB-001", "CONF-003"}


def test_parse_citations_ignores_invalid_ids() -> None:
    text = "Valid [INT-002] but not [PUB-99] or [FOO-001]."
    cited = Generator._parse_citations(text)
    assert cited == ["INT-002"]


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("generator.OpenAI")
def test_generate_returns_result_with_citations(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Answer from [PUB-001] and [INT-004]."))]
    )

    generator = Generator()
    context = [_make_result("PUB-001", "public", "ScopeGuard overview.")]
    result = generator.generate(
        query="What is ScopeGuard?",
        context=context,
        system_prompt="You are a helpful assistant.",
    )

    assert result.answer
    assert result.raw_response == result.answer
    assert set(result.cited_doc_ids) == {"PUB-001", "INT-004"}

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == config.LLM_MODEL
    assert call_kwargs["temperature"] == config.LLM_TEMPERATURE
    assert call_kwargs["max_tokens"] == config.LLM_MAX_TOKENS
    user_message = call_kwargs["messages"][1]["content"]
    assert "[PUB-001] (public)" in user_message
    assert "What is ScopeGuard?" in user_message


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("generator.OpenAI")
def test_generate_no_context_returns_empty_citations(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="General knowledge answer."))]
    )

    generator = Generator()
    result = generator.generate_no_context(
        query="What is ScopeGuard?",
        system_prompt="Answer from general knowledge.",
    )

    assert result.answer == "General knowledge answer."
    assert result.cited_doc_ids == []
    user_message = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert user_message == "What is ScopeGuard?"
    assert "Context:" not in user_message
