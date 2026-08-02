import re

from app.application.ports import ModelClientPort
from app.domain.citation import (
    CITATION_MARKER_PATTERN,
    Citation,
    evidence_citation_id,
)
from app.domain.models import (
    EvidenceItem,
    GeneratedAnswer,
    GenerationRequest,
    Question,
    RetrievedChunk,
)

# Composed from the domain pattern rather than respelling the syntax, so there is still one
# definition of what a citation marker looks like.
_MARKER_WITH_LEADING_SPACE = re.compile(rf"[ \t]*{CITATION_MARKER_PATTERN.pattern}")


class GenerationNode:
    """The generation step of the QA workflow: ask the model, and read its citations back.

    It labels each piece of evidence with a citation id before the model sees it, then maps
    the ids the model cited back to the chunks they stand for. Nothing else -- no lookup of
    source titles, no validation that the citations are grounded. Those are the resolution
    and validation steps, and keeping them out is what makes this one legible.

    Naming note: `retrieval.py` holds a node that *delegates* to a service, because
    retrieval has four collaborators worth hiding behind one. Generation has one, so the
    node is the step itself rather than a wrapper around it.
    """

    def __init__(self, model_client: ModelClientPort) -> None:
        self._model_client = model_client

    async def generate(
        self, question: Question, context_chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        chunk_by_citation_id = {
            evidence_citation_id(position): chunk
            for position, chunk in enumerate(context_chunks, start=1)
        }

        completion = await self._model_client.generate(
            GenerationRequest(
                question_text=question.text,
                language=question.language,
                evidence=[
                    EvidenceItem(citation_id=citation_id, text=chunk.text)
                    for citation_id, chunk in chunk_by_citation_id.items()
                ],
            )
        )
        return self._to_generated_answer(completion, chunk_by_citation_id)

    def _to_generated_answer(
        self, completion: str, chunk_by_citation_id: dict[str, RetrievedChunk]
    ) -> GeneratedAnswer:
        citations: list[Citation] = []
        seen: set[str] = set()

        for cited_id in CITATION_MARKER_PATTERN.findall(completion):
            chunk = chunk_by_citation_id.get(cited_id)
            if chunk is None or cited_id in seen:
                # An id we never issued cites nothing; a repeat is already recorded. Either
                # way there is no second Citation to add.
                continue
            seen.add(cited_id)
            # Order is where the model first cited this evidence, which is not its label:
            # a reply citing [c3] before [c1] gives (c3, 1) then (c1, 2).
            citations.append(
                Citation(
                    citation_id=cited_id, chunk_id=chunk.chunk_id, order=len(citations) + 1
                )
            )

        return GeneratedAnswer(
            answer_text=self._clean(completion, keep=seen),
            citations=citations,
            # No confidence signal until a real model client can report one.
            confidence=None,
        )

    @classmethod
    def _clean(cls, completion: str, keep: set[str]) -> str:
        """Keeps the markers that resolve and removes the ones that do not.

        Valid markers stay so a caller can render them as footnotes against the citation
        list -- that is the whole point of the id. An id we never issued is removed instead:
        leaving it would show the reader a footnote pointing at nothing, which is the text
        equivalent of a fabricated citation.

        The whitespace before a marker is consumed with it, so removing one from
        "something else [c9]." leaves "something else." rather than a space stranded before
        the full stop.
        """
        kept = _MARKER_WITH_LEADING_SPACE.sub(
            lambda match: match.group(0) if match.group(1) in keep else "", completion
        )
        return re.sub(r"[ \t]{2,}", " ", kept).strip()
