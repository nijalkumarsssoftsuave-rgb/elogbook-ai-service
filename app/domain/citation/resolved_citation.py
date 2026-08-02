from typing import Any

from pydantic import BaseModel, Field


class ResolvedCitation(BaseModel):
    """A citation reference resolved against the evidence it points at.

    This is what a caller receives: enough to show where an answer came from and to follow
    it back to the document. `source_title` and `score` are explicit fields rather than
    entries in `metadata` because the API response already contracts on both, and demoting
    typed fields into an untyped dict would be a downgrade the caller pays for.

    `metadata` carries the retrieved chunk's own metadata unchanged, so anything the
    retriever recorded stays reachable without this model having to grow a field for it.
    """

    citation_id: str
    # Carried over from the reference rather than recomputed from list position: if
    # resolution ever drops a reference, positions shift but the surviving citations keep
    # the order they were cited in, and stay consistent with their citation_id.
    order: int
    chunk_id: str
    source_id: str
    document_id: str
    source_title: str
    page_number: int | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
