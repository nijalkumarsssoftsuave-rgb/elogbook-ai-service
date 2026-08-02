import re

# The one citation syntax the pipeline agrees on.
#
# This lives in the domain because two layers have to agree on it exactly: the prompt
# template (infrastructure) tells the model to write these markers, and the generation node
# (application) reads them back. If those two ever drifted the failure would be silent --
# the model would cite diligently and every citation would vanish -- so both sides reference
# one definition rather than each spelling out the syntax.
CITATION_MARKER_PATTERN = re.compile(r"\[(c\d+)\]")


def citation_marker(citation_id: str) -> str:
    """Renders the marker a model is asked to write for a piece of evidence."""
    return f"[{citation_id}]"


def evidence_citation_id(position: int) -> str:
    """The label given to the evidence item at `position` (1-based).

    Assigned before the model sees the evidence, which is what stops it inventing a
    plausible-looking chunk id: the only thing it can cite is a label we issued.
    """
    return f"c{position}"
