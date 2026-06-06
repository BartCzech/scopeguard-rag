from pathlib import Path
from typing import cast

import frontmatter

from models import Document


def _parse_md_file(path: Path) -> Document:
    """Parse a single .md file with YAML frontmatter."""
    post = frontmatter.load(str(path))

    for key in ("doc_id", "title", "access_level", "adversarial"):
        if key not in post.metadata:
            raise ValueError(f"{path}: missing frontmatter field '{key}'")

    return Document(
        doc_id=cast(str, post["doc_id"]),
        title=cast(str, post["title"]),
        access_level=cast(str, post["access_level"]),
        adversarial=cast(bool, post["adversarial"]),
        content=post.content.strip(),
    )


def load_corpus(corpus_dir: str = "corpus") -> list[Document]:
    """
    Recursively loads all .md files under corpus_dir.
    Parses YAML frontmatter and returns Document objects.
    Raises ValueError if any file is missing required frontmatter fields.
    """
    root = Path(corpus_dir)
    documents: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        documents.append(_parse_md_file(path))
    return documents


def filter_by_level(
    documents: list[Document],
    allowed_levels: list[str],
) -> list[Document]:
    """
    Returns only documents whose access_level is in allowed_levels.
    Example: filter_by_level(docs, ["public", "internal"])
    """
    return [doc for doc in documents if doc.access_level in allowed_levels]


def get_document_by_id(
    documents: list[Document],
    doc_id: str,
) -> Document | None:
    """Lookup a single document by its doc_id."""
    for doc in documents:
        if doc.doc_id == doc_id:
            return doc
    return None
