"""MCP tool server wrapping the retriever with scope enforcement."""

from fastmcp import FastMCP

from src.auth import validate_token
from src.corpus_loader import filter_by_level, get_document_by_id, load_corpus
from src.retriever import FILTERED_OVERFETCH, Retriever

corpus = load_corpus()
retriever = Retriever()
mcp = FastMCP("ScopeGuard Retrieval Server")


@mcp.tool
def list_accessible_documents(token: str) -> list[dict]:
    """List metadata for all documents the user can access."""
    try:
        allowed_levels = validate_token(token).allowed_levels
    except Exception as e:
        return [{"error": "invalid_token", "detail": str(e)}]
    documents = filter_by_level(corpus, allowed_levels)
    return [
        {
            "doc_id": document.doc_id,
            "title": document.title,
            "access_level": document.access_level,
        }
        for document in documents
    ]


@mcp.tool
def read_document(doc_id: str, token: str) -> dict:
    """Read a full document by ID. Access-checked against the token."""
    try:
        allowed_levels = validate_token(token).allowed_levels
    except Exception as e:
        return {"error": "invalid_token", "detail": str(e)}
    accessible = filter_by_level(corpus, allowed_levels)
    doc = get_document_by_id(accessible, doc_id)
    if doc is None:
        return {"error": "not_accessible"}

    return {"doc_id": doc.doc_id, "title": doc.title, "content": doc.content}


@mcp.tool
def search_documents(query: str, top_k: int, token: str, filtering_mode: str) -> list[dict]:
    """
    Search documents with scope enforcement.

    filtering_mode:
      "pre"     — search only within user's allowed levels
      "post"    — search everything, filter results after
      "labeled" — search everything, return all with access labels
    """
    try:
        allowed_levels = validate_token(token).allowed_levels
    except Exception as e:
        return [{"error": "invalid_token", "detail": str(e)}]

    results = []
    match filtering_mode:
        case "pre":
            results = retriever.retrieve_filtered(
                query=query, allowed_levels=allowed_levels, top_k=top_k
            )
        case "post":
            results = retriever.retrieve(query=query, top_k=FILTERED_OVERFETCH)
            results = [r for r in results if r.access_level in allowed_levels]
            results = results[:top_k]
        case "labeled":
            results = retriever.retrieve(query=query, top_k=top_k)
        case _:
            raise ValueError(f"Invalid filtering mode: {filtering_mode}")
    return [r.model_dump() for r in results]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
