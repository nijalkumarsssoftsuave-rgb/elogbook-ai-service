from app.domain.citation import citation_marker
from app.domain.models import GenerationRequest, GroundedAnswer

# The instructions sent with every generation request. Model-specific wording, which is why
# it lives beside the client that talks to the model rather than a layer above it.
#
# The marker syntax in the examples is rendered from `citation_marker`, not typed out, so
# the instruction and the parser that reads the model's reply cannot describe different
# formats.
_INSTRUCTIONS = (
    "Answer using ONLY the evidence supplied below.\n"
    "After each claim, attach the citation id of the evidence it came from, written "
    f"exactly like {citation_marker('c1')} or {citation_marker('c2')}.\n"
    "Cite only ids that appear in the evidence below, and never invent one.\n"
    "If the evidence does not contain enough information to answer, reply with exactly "
    f'this sentence and cite nothing: "{GroundedAnswer.REFUSAL_TEXT}"'
)


def build_prompt(request: GenerationRequest) -> str:
    """Renders the prompt for one generation request.

    Evidence is labelled with its citation id and nothing else -- no chunk ids, no source
    titles, no scores. There is therefore nothing internal for the model to echo back or
    imitate, and the only citation it can make is one of the labels offered here.
    """
    evidence_block = "\n\n".join(
        f"{citation_marker(item.citation_id)} {item.text}" for item in request.evidence
    )
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"--- EVIDENCE ---\n{evidence_block}\n--- END EVIDENCE ---\n\n"
        f"Question ({request.language}): {request.question_text}\n\n"
        f"Answer:"
    )
