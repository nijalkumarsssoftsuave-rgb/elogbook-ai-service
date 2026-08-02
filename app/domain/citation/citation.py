from pydantic import BaseModel


class Citation(BaseModel):
    """A reference the model made: which chunk, and where in the answer it first appeared.

    Deliberately carries no source title, page number or score. Those belong to resolution,
    and keeping them out here is what stops generation reaching into retrieval's metadata
    to assemble presentation detail at the wrong layer.

    `order` and `citation_id` are different numbers and must not be confused. The id is the
    label the evidence was given *before* the model saw it; the order is the position the
    model first cited it. A model handed five pieces of evidence and citing [c3] before [c1]
    produces (c3, order 1) then (c1, order 2).
    """

    citation_id: str
    chunk_id: str
    order: int
