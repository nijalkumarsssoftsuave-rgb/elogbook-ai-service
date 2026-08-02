from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceSignalName(StrEnum):
    """The measurable inputs to a confidence score.

    A closed set, because each name is a weight in configuration. Adding one without a
    weight to go with it would silently contribute nothing.
    """

    # How much of the evidence the caller asked for was actually found.
    RETRIEVAL = "retrieval"
    # How clearly the best-ranked evidence stands out from the rest of the ranking.
    RERANKER = "reranker"
    # How many of the answer's citations resolved to evidence that was retrieved.
    CITATION = "citation"


class ConfidenceSignal(BaseModel):
    """One measured input to a confidence score, on a common 0-1 scale.

    Normalising to 0-1 at the point of measurement is what makes the signals addable at
    all: retrieval counts, fusion scores and citation ratios have nothing in common until
    each is expressed as "how good is this, out of the best it could be".

    `detail` records what the number was read off, so a low score can be explained without
    re-running the pipeline -- "2 of 5 requested chunks retrieved" is an answer, "0.4" is
    not.
    """

    name: ConfidenceSignalName
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    detail: str = ""

    @property
    def contribution(self) -> float:
        """The signal's unnormalised share of the score, before weights are divided out."""
        return self.value * self.weight
