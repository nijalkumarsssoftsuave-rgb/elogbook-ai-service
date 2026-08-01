import re

from app.domain.models import GroundedAnswer

# Recovers the evidence ids the prompt actually offered, so the stub can cite them.
_EVIDENCE_ID_PATTERN = re.compile(r"\[Evidence \d+ \| id=([^\]\s]+)\]")


class ModelClientStub:
    """Returns a canned completion that cites every piece of evidence it was given.
    Stands in for the Qwen model served via Cloudera AI Inference.

    Reading ids back out of the prompt keeps the whole downstream path — citation
    parsing, validation, caching — genuinely exercisable in development. It is a
    development shortcut, not something the real client will do.
    """

    async def generate(self, prompt: str) -> str:
        chunk_ids = _EVIDENCE_ID_PATTERN.findall(prompt)
        if not chunk_ids:
            return GroundedAnswer.REFUSAL_TEXT

        markers = " ".join(f"[[{chunk_id}]]" for chunk_id in chunk_ids)
        return f"This is a stub answer grounded in the provided evidence. {markers}"
