import os
import re
import sys
from pathlib import Path

from openai import OpenAI

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config  # noqa: E402
from models import GenerationResult, RetrievalResult  # noqa: E402


class Generator:
    def __init__(self):
        """Initialize OpenAI client. API key read from env var."""
        self.client = OpenAI(api_key=os.environ[config.OPENAI_API_KEY_ENV])

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        system_prompt: str,
    ) -> GenerationResult:
        """
        Builds a full prompt, calls the LLM, parses citations.

        The context is formatted into the user message as:

        ---
        [PUB-001] (public)
        Content of the chunk...
        ---
        [CONF-003] (confidential)
        Content of the chunk...
        ---

        The system_prompt is provided by the calling strategy.
        The generator does NOT modify or extend it.
        """
        context_str = self._format_context(context)
        user_message = f"Context:\n{context_str}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        raw_response = response.choices[0].message.content or ""
        cited_ids = self._parse_citations(raw_response)

        return GenerationResult(
            answer=raw_response,
            cited_doc_ids=cited_ids,
            raw_response=raw_response,
        )

    def generate_no_context(
        self,
        query: str,
        system_prompt: str,
    ) -> GenerationResult:
        """
        For Strategy 0 (pure LLM). No retrieval context provided.
        """
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        raw_response = response.choices[0].message.content or ""
        return GenerationResult(
            answer=raw_response,
            cited_doc_ids=[],
            raw_response=raw_response,
        )

    def _format_context(self, context: list[RetrievalResult]) -> str:
        """Format retrieved chunks for injection into the prompt."""
        blocks = []
        for r in context:
            blocks.append(f"---\n[{r.doc_id}] ({r.access_level})\n{r.content}\n")
        blocks.append("---")
        return "\n".join(blocks)

    @staticmethod
    def _parse_citations(response: str) -> list[str]:
        """
        Extract all [DOC_ID] citations from the response text.
        Returns deduplicated list. Matches PUB-xxx, INT-xxx, CONF-xxx, ADV-xxx.
        """
        pattern = r"\[((?:PUB|INT|CONF|ADV)-\d{3})\]"
        return list(set(re.findall(pattern, response)))
