"""Executes an evaluation dataset against the pipeline and aggregates the results.

Deterministic and fully offline: the corpus is in-memory, every model is a stub, and
nothing consults the clock except the report timestamp. Two runs over the same dataset
produce identical output apart from `generated_at`.
"""

from app.application.dto import QueryRequestDTO
from app.domain.models import PermissionScope, Question, RetrievalSearchContext
from tests.eval import metrics
from tests.eval.loader import LanguageThresholds
from tests.eval.schema import (
    AnswerQualityMetrics,
    CaseContractResult,
    CaseRetrievalResult,
    ContractAggregate,
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
    RetrievalAggregate,
    RetrievalMetrics,
    Surface,
)
from tests.eval.surfaces import EvaluationSurfaces, build_surfaces

# Surfaces that produce a ranked list and are therefore scored for retrieval accuracy.
# PIPELINE_CONTRACT is deliberately absent: its citation list is a function of the model
# stub citing everything it is handed, so scoring it would be a laundered copy of
# RETRIEVAL_RAW dressed up as an answer-level measurement.
_RANKED_SURFACES = (Surface.KEYWORD_ONLY, Surface.RETRIEVAL_RAW)


class EvaluationRunner:
    DEFAULT_K_VALUES = (1, 3, 5)

    def __init__(
        self,
        surfaces: EvaluationSurfaces | None = None,
        top_k: int = 5,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        user_id: str = "eval-harness",
        roles: tuple[str, ...] = ("viewer",),
    ) -> None:
        self._surfaces = surfaces or build_surfaces()
        self._top_k = top_k
        self._k_values = k_values
        self._user_id = user_id
        self._roles = list(roles)

    async def run(
        self, dataset: EvaluationDataset, thresholds: LanguageThresholds | None = None
    ) -> EvaluationReport:
        case_results: list[CaseRetrievalResult] = []
        contract_results: list[CaseContractResult] = []
        errors: list[str] = []

        for case in dataset.cases:
            for surface in _RANKED_SURFACES:
                case_results.append(await self._score_retrieval(case, surface, errors))
            contract_results.append(await self._score_contract(case, errors))

        report = self._aggregate(dataset, case_results, contract_results, errors)
        if thresholds is not None:
            report = report.model_copy(
                update={"threshold_outcomes": thresholds.evaluate(report)}
            )
        return report

    async def _retrieve(self, case: EvaluationCase, surface: Surface) -> list[str]:
        if surface is Surface.KEYWORD_ONLY:
            # The same candidate pool RetrievalService asks for, so the two surfaces are
            # comparable rather than differing merely by how much they fetched.
            chunks = await self._surfaces.keyword_retriever.search(
                case.query,
                top_k=self._surfaces.retrieval_service._CANDIDATE_POOL_SIZE,
                language=case.language,
            )
        else:
            question = Question(
                text=case.query,
                user_id=self._user_id,
                roles=self._roles,
                language=case.language,
            )
            chunks = await self._surfaces.retrieval_service.retrieve(
                RetrievalSearchContext(
                    question=question,
                    permission_scope=PermissionScope.from_roles(self._roles),
                    top_k=self._top_k,
                )
            )
        return [chunk.chunk_id for chunk in chunks]

    async def _score_retrieval(
        self, case: EvaluationCase, surface: Surface, errors: list[str]
    ) -> CaseRetrievalResult:
        """A failing case is recorded, never fatal: a report with fourteen results and
        one named error is far more useful than a traceback and no numbers.
        """
        try:
            retrieved = await self._retrieve(case, surface)
        except Exception as exc:  # noqa: BLE001 - any failure is recorded, never fatal
            message = f"{case.case_id}/{surface.value}: {type(exc).__name__}: {exc}"
            errors.append(message)
            return CaseRetrievalResult(
                case_id=case.case_id,
                surface=surface,
                retrieved_chunk_ids=[],
                relevant_chunk_ids=case.relevant_chunk_ids,
                error=message,
            )

        scored: list[RetrievalMetrics] = []
        if case.expects_answer:
            gold = set(case.relevant_chunk_ids)
            scored = [
                RetrievalMetrics(
                    k=k,
                    hit_rate=metrics.hit_rate_at_k(retrieved, gold, k),
                    recall=metrics.recall_at_k(retrieved, gold, k),
                    precision=metrics.precision_at_k(retrieved, gold, k),
                    reciprocal_rank=metrics.reciprocal_rank_at_k(retrieved, gold, k),
                    ndcg=metrics.ndcg_at_k(retrieved, gold, k),
                )
                for k in self._k_values
            ]
        return CaseRetrievalResult(
            case_id=case.case_id,
            surface=surface,
            retrieved_chunk_ids=retrieved,
            relevant_chunk_ids=case.relevant_chunk_ids,
            metrics=scored,
        )

    async def _score_contract(
        self, case: EvaluationCase, errors: list[str]
    ) -> CaseContractResult:
        try:
            question = Question(
                text=case.query,
                user_id=self._user_id,
                roles=self._roles,
                language=case.language,
            )
            retrieved = [
                chunk.chunk_id
                for chunk in await self._surfaces.retrieval_service.retrieve(
                    RetrievalSearchContext(
                        question=question,
                        permission_scope=PermissionScope.from_roles(self._roles),
                        top_k=self._top_k,
                    )
                )
            ]
            result = await self._surfaces.qa_service.execute(
                QueryRequestDTO(
                    query=case.query,
                    user_id=self._user_id,
                    roles=self._roles,
                    correlation_id=f"eval-{case.case_id}",
                    top_k=self._top_k,
                    permission_scope=PermissionScope.from_roles(self._roles),
                )
            )
        except Exception as exc:  # noqa: BLE001 - any failure is recorded, never fatal
            message = f"{case.case_id}/{Surface.PIPELINE_CONTRACT.value}: {type(exc).__name__}: {exc}"
            errors.append(message)
            return CaseContractResult(
                case_id=case.case_id,
                expects_answer=case.expects_answer,
                refused=False,
                is_grounded=False,
                retrieved_chunk_ids=[],
                citation_chunk_ids=[],
                citations_within_retrieved=False,
                citations_within_corpus=False,
                citations_resolving_to_corpus=0,
                cited_any_relevant=False,
                error=message,
            )

        citation_ids = [citation.chunk_id for citation in result.citations]
        corpus_ids = self._surfaces.corpus_chunk_ids
        gold = set(case.relevant_chunk_ids)
        resolving = sum(1 for cid in citation_ids if cid in corpus_ids)
        return CaseContractResult(
            case_id=case.case_id,
            expects_answer=case.expects_answer,
            refused=result.refused,
            is_grounded=result.is_grounded,
            retrieved_chunk_ids=retrieved,
            citation_chunk_ids=citation_ids,
            citations_within_retrieved=all(cid in set(retrieved) for cid in citation_ids),
            citations_within_corpus=resolving == len(citation_ids),
            citations_resolving_to_corpus=resolving,
            cited_any_relevant=bool(gold & set(citation_ids)),
        )

    def _aggregate(
        self,
        dataset: EvaluationDataset,
        case_results: list[CaseRetrievalResult],
        contract_results: list[CaseContractResult],
        errors: list[str],
    ) -> EvaluationReport:
        retrieval: dict[Surface, list[RetrievalAggregate]] = {}
        for surface in _RANKED_SURFACES:
            scored = [
                result
                for result in case_results
                if result.surface is surface and result.metrics
            ]
            retrieval[surface] = [
                self._aggregate_at_k(scored, k) for k in self._k_values
            ]

        return EvaluationReport(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            language=dataset.language,
            top_k=self._top_k,
            k_values=list(self._k_values),
            case_count=len(dataset.cases),
            corpus_size=len(self._surfaces.corpus),
            corpus_languages=self._surfaces.corpus_languages,
            retrieval=retrieval,
            contract=self._aggregate_contract(contract_results),
            answer_quality=AnswerQualityMetrics(),
            case_results=case_results,
            contract_results=contract_results,
            errors=errors,
            notes=list(_STANDING_NOTES),
        )

    @staticmethod
    def _aggregate_at_k(scored: list[CaseRetrievalResult], k: int) -> RetrievalAggregate:
        at_k = [
            metric
            for result in scored
            for metric in result.metrics
            if metric.k == k
        ]
        return RetrievalAggregate(
            k=k,
            case_count=len(at_k),
            hit_rate_at_k=metrics.mean([metric.hit_rate for metric in at_k]),
            recall_at_k=metrics.mean([metric.recall for metric in at_k]),
            precision_at_k=metrics.mean([metric.precision for metric in at_k]),
            mrr_at_k=metrics.mean([metric.reciprocal_rank for metric in at_k]),
            ndcg_at_k=metrics.mean([metric.ndcg for metric in at_k]),
        )

    @staticmethod
    def _aggregate_contract(results: list[CaseContractResult]) -> ContractAggregate:
        answerable = [result for result in results if result.expects_answer]
        out_of_corpus = [result for result in results if not result.expects_answer]
        all_citations = [cid for result in results for cid in result.citation_chunk_ids]
        return ContractAggregate(
            case_count=len(results),
            answerable_case_count=len(answerable),
            out_of_corpus_case_count=len(out_of_corpus),
            grounded_rate=metrics.rate(
                sum(1 for result in results if result.is_grounded), len(results)
            ),
            refusal_rate=metrics.rate(
                sum(1 for result in results if result.refused), len(results)
            ),
            citation_validity_rate=metrics.rate(
                sum(1 for result in results if result.citations_within_retrieved),
                len(results),
            ),
            citation_corpus_coverage=metrics.rate(
                sum(result.citations_resolving_to_corpus for result in results),
                len(all_citations),
            ),
            cited_relevant_rate=metrics.rate(
                sum(1 for result in answerable if result.cited_any_relevant), len(answerable)
            ),
            correct_refusal_rate=metrics.rate(
                sum(1 for result in out_of_corpus if result.refused), len(out_of_corpus)
            ),
            false_refusal_rate=metrics.rate(
                sum(1 for result in answerable if result.refused), len(answerable)
            ),
            empty_retrieval_rate=metrics.rate(
                sum(1 for result in results if not result.retrieved_chunk_ids), len(results)
            ),
        )


_STANDING_NOTES = (
    (
        "Answer quality is not measured: the LLM, embeddings, dense search and the "
        "reranker are stubs, so a score would measure the stub rather than the system."
    ),
    (
        "rank 1 of retrieval_raw is always the dense stub's chunk, which caps MRR at 0.5 "
        "and pins precision@1 at 0.0 there. Use keyword_only for the retrieval number."
    ),
    (
        "The service cannot currently refuse: the dense stub always supplies evidence, so "
        "citations always validate and correct_refusal_rate is structurally 0.0."
    ),
)
