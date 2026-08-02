from pydantic import BaseModel, Field, model_validator

from app.domain.citation import CitationFailureReason, CitationValidationResult
from app.domain.confidence import (
    ConfidenceBand,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceSignalName,
)
from app.domain.models import GeneratedAnswer, RetrievedChunk

# Scores are rounded before they leave the service. Weighted means of ratios produce
# trailing float noise that differs by summation order, and "the same inputs always score
# the same" should hold for the number a caller compares and caches, not just in principle.
_SCORE_PRECISION = 6


class ConfidencePolicy(BaseModel):
    """How the signals are weighted and where the band boundaries sit.

    A value object rather than a read of Settings, so the scoring service stays in the
    application layer without reaching down into configuration. The API layer builds one
    from Settings and injects it.

    Weights are relative, not required to sum to one: the score divides by the weights that
    actually applied. That matters because a signal is omitted when it cannot be measured,
    and a fixed denominator would then quietly drag every score down.
    """

    retrieval_weight: float = Field(ge=0.0)
    reranker_weight: float = Field(ge=0.0)
    citation_weight: float = Field(ge=0.0)
    refusal_threshold: float = Field(ge=0.0, le=1.0)
    review_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_policy_is_usable(self) -> "ConfidencePolicy":
        if self.refusal_threshold > self.review_threshold:
            raise ValueError(
                "confidence refusal_threshold must not exceed review_threshold: "
                f"{self.refusal_threshold} > {self.review_threshold}"
            )
        if not (self.retrieval_weight or self.reranker_weight or self.citation_weight):
            raise ValueError("at least one confidence weight must be greater than zero")
        return self

    def weight_for(self, name: ConfidenceSignalName) -> float:
        return {
            ConfidenceSignalName.RETRIEVAL: self.retrieval_weight,
            ConfidenceSignalName.RERANKER: self.reranker_weight,
            ConfidenceSignalName.CITATION: self.citation_weight,
        }[name]

    def band_for(self, score: float) -> ConfidenceBand:
        if score < self.refusal_threshold:
            return ConfidenceBand.LOW
        if score < self.review_threshold:
            return ConfidenceBand.NEEDS_REVIEW
        return ConfidenceBand.HIGH


class ConfidenceScoringService:
    """Scores how well supported an answer is, from signals the pipeline already produced.

    It measures; it does not retrieve, generate, rerank or refuse. Everything it reads is
    an output of a step that has already run, which is what makes the score reproducible:
    the same chunks, the same answer and the same validation verdict always give the same
    number, with no clock, no randomness and no second call to a model.

    **Every signal is scale-free.** Nothing here divides by an assumed maximum score,
    because it cannot know one: BM25 scores are unbounded, cosine similarities are not, and
    after fusion the numbers are reciprocal ranks. So retrieval is measured as a proportion
    of what was asked for, ranking as a relative separation, and citations as a ratio of
    references. Swapping in a real reranker changes the numbers these read, not the way
    they are read.

    A signal that cannot be measured is **omitted rather than guessed** -- a single chunk
    has no runner-up to stand out from, and inventing a value for that would be putting an
    opinion into a measurement. The score is the weighted mean of the signals that were
    present.
    """

    def __init__(self, policy: ConfidencePolicy) -> None:
        self._policy = policy

    def score(
        self,
        chunks: list[RetrievedChunk],
        requested_top_k: int,
        generated: GeneratedAnswer,
        validation: CitationValidationResult,
    ) -> ConfidenceResult:
        signals = [
            signal
            for signal in (
                self._retrieval_signal(chunks, requested_top_k),
                self._reranker_signal(chunks),
                self._citation_signal(generated, validation),
            )
            if signal is not None
        ]

        applied_weight = sum(signal.weight for signal in signals)
        score = (
            round(sum(signal.contribution for signal in signals) / applied_weight, _SCORE_PRECISION)
            if applied_weight
            else 0.0
        )
        return ConfidenceResult(
            score=score, band=self._policy.band_for(score), signals=signals
        )

    def _retrieval_signal(
        self, chunks: list[RetrievedChunk], requested_top_k: int
    ) -> ConfidenceSignal | None:
        """How much of the evidence the caller asked for was actually found.

        A proportion rather than a score, because the scores themselves are not comparable
        across retrieval methods. Coming back with two chunks when five were asked for is a
        thin answer however well those two scored.
        """
        weight = self._policy.weight_for(ConfidenceSignalName.RETRIEVAL)
        if not weight or requested_top_k <= 0:
            return None
        found = min(len(chunks), requested_top_k)
        return ConfidenceSignal(
            name=ConfidenceSignalName.RETRIEVAL,
            value=found / requested_top_k,
            weight=weight,
            detail=f"{len(chunks)} of {requested_top_k} requested chunks retrieved",
        )

    def _reranker_signal(self, chunks: list[RetrievedChunk]) -> ConfidenceSignal | None:
        """How clearly the best evidence stands out from the rest of the ranking.

        Measured as the top chunk's relative lead over the mean of the others, which is a
        ratio and so survives any positive scoring scale. A ranking whose top result barely
        beats its runners-up is one where the ordering, and therefore the evidence the model
        was handed, could easily have come out differently.

        Omitted when there is nothing to compare against: a single chunk has no runner-up,
        and a non-positive top score means the scale is not one a ratio can be taken on.
        """
        weight = self._policy.weight_for(ConfidenceSignalName.RERANKER)
        if not weight or len(chunks) < 2:
            return None
        # Sorted here rather than trusting the incoming order: the chunks arrive ranked, but
        # a signal that silently reports nonsense if that ever stops being true is worse
        # than one that costs a sort. Duplicated top scores stay in `rest`, where they
        # belong -- two equally strong chunks means no separation, not a tie to discard.
        top, *rest = sorted((chunk.score for chunk in chunks), reverse=True)
        if top <= 0:
            return None
        lead = (top - sum(rest) / len(rest)) / top
        return ConfidenceSignal(
            name=ConfidenceSignalName.RERANKER,
            value=min(max(lead, 0.0), 1.0),
            weight=weight,
            detail=f"top-ranked chunk leads the rest by {lead:.0%} of its score",
        )

    def _citation_signal(
        self, generated: GeneratedAnswer, validation: CitationValidationResult
    ) -> ConfidenceSignal | None:
        """How many of the answer's citations resolved to evidence that was retrieved.

        An answer citing nothing scores zero: it may be perfectly fluent, but nothing in it
        is traceable to a source, which is the failure this whole pipeline is built to
        prevent.

        In today's pipeline this signal is 1.0 on every answer that reaches a caller,
        because generation can only reference labels it issued and the validator rejects
        anything else outright. It is not decoration: the ratio is what will move once a
        real model is answering, and it is measurable now on any answer that failed
        validation.
        """
        weight = self._policy.weight_for(ConfidenceSignalName.CITATION)
        if not weight:
            return None
        referenced = len(generated.citations)
        if not referenced or validation.reason is CitationFailureReason.NO_CITATIONS:
            return ConfidenceSignal(
                name=ConfidenceSignalName.CITATION,
                value=0.0,
                weight=weight,
                detail="the answer cited nothing",
            )
        resolved = referenced - len(validation.unresolved_citation_ids)
        return ConfidenceSignal(
            name=ConfidenceSignalName.CITATION,
            value=max(resolved, 0) / referenced,
            weight=weight,
            detail=f"{resolved} of {referenced} citations resolved to retrieved evidence",
        )
