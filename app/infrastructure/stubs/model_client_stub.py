from app.domain.citation import citation_marker
from app.domain.models import GenerationRequest, GroundedAnswer
from app.infrastructure.model_serving.llm.prompt_template import build_prompt


class ModelClientStub:
    """Returns a canned completion that cites every piece of evidence it was given.
    Stands in for the Qwen model served via Cloudera AI Inference.

    It reads the citation ids straight off the request, which is what the structured port
    made possible -- the previous version scraped them back out of a prompt string with a
    regex, a development hack that only worked because it knew how the prompt was built.

    It still builds the prompt and discards it. That is not ceremony: it means a template
    that fails to render is caught by every development request, rather than on the day a
    real client is wired in behind it.
    """

    async def generate(self, request: GenerationRequest) -> str:
        build_prompt(request)

        if not request.evidence:
            return GroundedAnswer.REFUSAL_TEXT

        markers = " ".join(citation_marker(item.citation_id) for item in request.evidence)
        return f"This is a stub answer grounded in the provided evidence. {markers}"
