from typing import NamedTuple

from app.application.audit_service import AuditService
from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validator import CitationValidator
from app.application.confidence.confidence_scoring_service import (
    ConfidenceScoringService,
)
from app.application.confidence.grounding_decision_service import (
    GroundingDecisionService,
)
from app.application.dto import QueryRequestDTO, QueryResultDTO
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.ports import FeatureFlagPort
from app.application.qa.nodes.generation import GenerationNode
from app.application.qa.nodes.retrieval import RetrievalNode
from app.application.review.human_review_service import HumanReviewService
from app.domain.confidence import ConfidenceResult
from app.domain.models import GroundedAnswer, Question, RetrievedChunk


class QAApplicationService:
    """Orchestrates a QA query end-to-end, one step per business capability:

    language detection -> cache lookup -> input guardrail -> retrieval ->
    evidence guardrail -> generate, validate & resolve citations (one retry, else refuse)
    -> output guardrail -> audit & cache.

    Each collaborator owns its own details; this service only sequences them.

    **Feature flags are read here and nowhere below.** A capability that can be switched
    off is a decision about the shape of the pipeline, not about how a step behaves, so
    the branch belongs to whatever sequences the steps. Pushing the check into
    ConfidenceScoringService or HumanReviewService would give each one a mode in which it
    does nothing, and every test of those services would then have to prove it was not in
    that mode.
    """

    # One initial attempt plus exactly one retry, then a safe refusal.
    _MAX_GENERATION_ATTEMPTS = 2

    def __init__(
        self,
        language_detection: LanguageDetectionService,
        guardrail_service: GuardrailService,
        retrieval_node: RetrievalNode,
        generation_node: GenerationNode,
        citation_resolver: CitationResolver,
        citation_validator: CitationValidator,
        confidence_scoring_service: ConfidenceScoringService,
        grounding_decision_service: GroundingDecisionService,
        human_review_service: HumanReviewService,
        audit_service: AuditService,
        features: FeatureFlagPort,
    ) -> None:
        self._language_detection = language_detection
        self._guardrail_service = guardrail_service
        self._retrieval_node = retrieval_node
        self._generation_node = generation_node
        self._citation_resolver = citation_resolver
        self._citation_validator = citation_validator
        self._confidence_scoring_service = confidence_scoring_service
        self._grounding_decision_service = grounding_decision_service
        self._human_review_service = human_review_service
        self._features = features
        self._audit_service = audit_service

    async def execute(self, request: QueryRequestDTO) -> QueryResultDTO:
        detected = await self._language_detection.detect_and_validate(request.query)

        question = Question(
            text=request.query,
            user_id=request.user_id,
            roles=request.roles,
            language=detected.code,
        )

        cached = await self._audit_service.get_cached(question)
        if cached is not None:
            await self._audit_service.record_cache_hit(
                question, cached, request.correlation_id, request.provenance
            )
            return QueryResultDTO.from_domain(cached, cache_hit=True)

        await self._guardrail_service.check_question(question)

        chunks = await self._retrieval_node.run(
            question, request.permission_scope, top_k=request.top_k, filters=request.filters
        )
        chunks = await self._guardrail_service.screen_retrieved_chunks(chunks)

        scored = await self._generate_and_validate(question, chunks, request.top_k)
        answer = await self._guardrail_service.check_answer(scored.delivered)

        if scored.confidence is not None and self._features.is_human_review_enabled():
            # Routing reads the answer and creates work; it returns nothing into the
            # flow, so there is no path by which queueing a review could change what
            # the caller receives. It is handed the *generated* answer rather than the
            # delivered one, because a withheld answer is exactly what a reviewer needs
            # to see.
            await self._human_review_service.route(
                question,
                scored.generated,
                scored.confidence,
                chunks,
                request.correlation_id,
            )

        # Provenance is handed to the audit trail and nowhere else -- this service does not
        # branch on how the question arrived, and must not start to.
        await self._audit_service.finalize(
            question, answer, request.correlation_id, request.provenance
        )
        return QueryResultDTO.from_domain(answer, cache_hit=False)

    async def _generate_and_validate(
        self, question: Question, chunks: list[RetrievedChunk], requested_top_k: int
    ) -> "_ScoredAnswer":
        """Generates an answer, resolves its citations, and keeps it only if they hold up.

                generate -> resolve -> validate -+- valid   -> return the answer
                                                 |
                                                 +- invalid -> retry, then refuse

        Acting on the verdict is this service's job, not the validator's: the validator
        reports, and the orchestrator decides. That split is what lets the same verdict drive
        a retry today and a conditional edge once the flow becomes a LangGraph graph, without
        the validator learning anything about either.

        A failed attempt is discarded whole. Dropping the offending citation and returning
        the rest would leave the claim it was supporting standing with nothing behind it, and
        substituting another citation would be fabricating a source -- so the only honest
        outcomes are a fresh attempt or a grounded refusal.

        The loop lives here rather than inside any one collaborator because it is the only
        place holding all three: having the generation node call the validator (or vice
        versa) would couple collaborators that are otherwise siblings.
        """
        for _ in range(self._MAX_GENERATION_ATTEMPTS):
            generated = await self._generation_node.generate(question, chunks)
            grounded = self._citation_resolver.resolve(generated, chunks)
            validation = self._citation_validator.validate(generated, grounded, chunks)
            if validation.is_valid:
                if not self._features.is_confidence_scoring_enabled():
                    # Nothing scored means nothing to decide on and nothing to route.
                    # The answer goes out exactly as it did before ES-334 -- grounded,
                    # cited, and carrying no confidence rather than a made-up one.
                    return _ScoredAnswer(
                        delivered=grounded, generated=grounded, confidence=None
                    )

                confidence = self._confidence_scoring_service.score(
                    chunks, requested_top_k, generated, validation
                )
                # The grounding decision is applied once, to the accepted answer, and
                # not from inside the retry: a thin score reflects thin evidence, and
                # retrying regenerates the answer without re-running retrieval, so a
                # second attempt would be asked to fix something it cannot reach.
                scored = grounded.model_copy(update={"confidence": confidence.score})
                return _ScoredAnswer(
                    delivered=self._grounding_decision_service.decide(scored, confidence),
                    generated=scored,
                    confidence=confidence,
                )
        # Nothing was ever scored, so there is nothing for a reviewer to weigh up: this
        # is an answer that could not be grounded at all, not a weak one.
        refusal = GroundedAnswer.refusal()
        return _ScoredAnswer(delivered=refusal, generated=refusal, confidence=None)


class _ScoredAnswer(NamedTuple):
    """What one pass of the answer pipeline produced.

    `delivered` and `generated` are the same object unless the grounding decision withheld
    the answer. Keeping both is what lets review routing see the answer that was refused --
    a review task holding only the canned refusal would be useless to the reviewer it was
    created for.
    """

    delivered: GroundedAnswer
    generated: GroundedAnswer
    confidence: ConfidenceResult | None
