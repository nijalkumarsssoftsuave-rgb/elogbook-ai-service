import re

from app.domain.models import GroundedAnswer, Question, RetrievedChunk

# The one citation syntax the pipeline agrees on: the prompt instructs the model to emit
# [[chunk_id]] markers, and GenerationService parses the same pattern back out.
CITATION_MARKER_PATTERN = re.compile(r"\[\[([^\[\]\s]+)\]\]")

_INSTRUCTIONS = (
    "Answer the question using ONLY the evidence provided below.\n"
    "Every factual claim in your answer MUST be immediately followed by a citation "
    "marker in the exact form [[chunk_id]], where chunk_id is copied verbatim from one "
    "of the Evidence entries below.\n"
    "Never invent a chunk_id and never cite one that is not listed below.\n"
    "If the evidence does not contain enough information to answer, respond with "
    f'exactly this sentence and cite nothing: "{GroundedAnswer.REFUSAL_TEXT}"'
)


class PromptBuilder:
    """Builds the grounded prompt: instructions, labelled evidence, then the question.

    Pure string assembly with no model or infrastructure dependency, so it is real code
    rather than a stub even though the model client behind it is still stubbed.
    """

    def build(self, question: Question, context_chunks: list[RetrievedChunk]) -> str:
        evidence_block = "\n\n".join(
            f"[Evidence {index} | id={chunk.chunk_id}]\n{chunk.text}"
            for index, chunk in enumerate(context_chunks, start=1)
        )
        return (
            f"{_INSTRUCTIONS}\n\n"
            f"--- EVIDENCE ---\n{evidence_block}\n--- END EVIDENCE ---\n\n"
            f"Question ({question.language}): {question.text}\n\n"
            f"Answer:"
        )
