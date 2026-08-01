"""Evaluation schema for ES-322.

What this measures, and just as importantly what it does not:

  MEASURED     Retrieval accuracy against known-correct chunk ids -- BM25 keyword search
               is the one real retrieval component in the service today -- plus the
               pipeline's structural contract: do citations resolve, does it refuse.

  NOT MEASURED Answer quality. The LLM, the embedding model, the dense vector store and
               the reranker are all stubs. `ModelClientStub` returns one fixed sentence
               citing every chunk it was handed, so any "answer accuracy" derived from it
               would measure the stub rather than the system. `AnswerQualityMetrics`
               reserves the fields; nothing populates them until a real model exists.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseTag(StrEnum):
    """Why a case is in the set, so the report can break accuracy down by question shape
    instead of reporting one number that hides which kinds of question fail.
    """

    SINGLE_SOURCE = "single_source"
    MULTI_SOURCE = "multi_source"
    PARAPHRASE = "paraphrase"
    MORPHOLOGY = "morphology"
    OUT_OF_CORPUS = "out_of_corpus"


class Surface(StrEnum):
    """The points results are read from. Each answers a different question, and reading
    only one of them would misattribute stub artifacts to retrieval quality.
    """

    KEYWORD_ONLY = "keyword_only"
    RETRIEVAL_RAW = "retrieval_raw"
    PIPELINE_CONTRACT = "pipeline_contract"


class EvaluationCase(BaseModel):
    """One question with its known-correct evidence.

    `extra="forbid"` matters: the datasets are hand-edited JSON, so a mistyped key must
    fail at load rather than be silently ignored, which would make the case score zero
    and look exactly like a regression.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    query: str
    language: str
    # Empty means nothing in the corpus answers this question. Such a case is scored on
    # refusal behaviour instead; retrieval metrics are undefined without a relevant set.
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    tags: list[CaseTag] = Field(default_factory=list)
    # Prose only, never scored. Records what a correct answer would say, so a reader can
    # judge the case and a future answer-quality judge has a ground truth to work from.
    expected_answer_note: str | None = None
    # Links an Arabic case to the English case it mirrors, so the two languages are
    # compared question-for-question rather than only in aggregate.
    parallel_case_id: str | None = None

    @property
    def expects_answer(self) -> bool:
        return bool(self.relevant_chunk_ids)


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    language: str
    version: str
    description: str
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> "EvaluationDataset":
        ids = [case.case_id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate case_id(s): {', '.join(duplicates)}")
        mismatched = [case.case_id for case in self.cases if case.language != self.language]
        if mismatched:
            raise ValueError(
                f"cases declare a language other than '{self.language}': "
                f"{', '.join(mismatched)}"
            )
        return self

    @property
    def answerable_cases(self) -> list[EvaluationCase]:
        return [case for case in self.cases if case.expects_answer]

    @property
    def out_of_corpus_cases(self) -> list[EvaluationCase]:
        return [case for case in self.cases if not case.expects_answer]


class RetrievalMetrics(BaseModel):
    """Per-case retrieval scores at one cut-off."""

    k: int
    hit_rate: float
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float


class CaseRetrievalResult(BaseModel):
    case_id: str
    surface: Surface
    # Verbatim, in rank order, exactly as the surface returned it. Nothing is filtered
    # out, so the numbers describe the real system rather than a tidied version of it.
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    metrics: list[RetrievalMetrics] = Field(default_factory=list)
    error: str | None = None


class CaseContractResult(BaseModel):
    """What the full pipeline did, structurally. Nothing here reads the answer prose."""

    case_id: str
    expects_answer: bool
    refused: bool
    is_grounded: bool
    retrieved_chunk_ids: list[str]
    citation_chunk_ids: list[str]
    # Every citation points at a chunk that was actually retrieved.
    citations_within_retrieved: bool
    # Every citation resolves to a real corpus chunk. Deliberately separate from the
    # above: today they differ, because the dense stub's chunk is in no corpus, and that
    # difference is itself the finding.
    citations_within_corpus: bool
    # How many of this case's citations resolve to a real corpus chunk, so the pooled
    # coverage rate can be computed without reconstructing it from the boolean.
    citations_resolving_to_corpus: int
    # At least one citation is a known-correct chunk -- the most meaningful end-to-end
    # signal available while the model is a stub.
    cited_any_relevant: bool
    error: str | None = None


class AnswerQualityMetrics(BaseModel):
    """RESERVED. Nothing computes these today.

    Field names mirror RAGAS metric names so enabling this is a mapping exercise rather
    than a schema redesign. All three of the following must hold first:

      1. `ModelClientPort` is backed by a real model, so the answer text is the system's
         own output rather than a canned sentence.
      2. A judge model is reachable inside the deployment boundary. RAGAS drives an LLM
         judge, and CLAUDE.md's air-gapped target rules out a hosted one.
      3. `ragas` is added to the dev dependencies and every case carries a ground-truth
         answer (add that field then, not now).

    Until then `computed` stays False and every score stays None: an absent number is
    honest, a fabricated one is not.
    """

    computed: bool = False
    judge_model: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RetrievalAggregate(BaseModel):
    k: int
    case_count: int
    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


class ContractAggregate(BaseModel):
    case_count: int
    answerable_case_count: int
    out_of_corpus_case_count: int

    grounded_rate: float
    refusal_rate: float
    citation_validity_rate: float
    citation_corpus_coverage: float
    cited_relevant_rate: float
    correct_refusal_rate: float
    false_refusal_rate: float
    empty_retrieval_rate: float


class ThresholdOutcome(BaseModel):
    metric_path: str
    floor: float | None = None
    ceiling: float | None = None
    actual: float
    passed: bool

    def describe(self) -> str:
        if self.floor is not None:
            bound = f">= {self.floor:.4f}"
            reference = self.floor
        else:
            assert self.ceiling is not None
            bound = f"<= {self.ceiling:.4f}"
            reference = self.ceiling
        status = "PASS" if self.passed else "FAIL"
        delta = "" if self.passed else f"  ({self.actual - reference:+.4f})"
        return f"{status}  {self.metric_path:<46} {bound:<14} actual {self.actual:.4f}{delta}"


class EvaluationReport(BaseModel):
    dataset_name: str
    dataset_version: str
    language: str
    top_k: int
    k_values: list[int]
    case_count: int
    corpus_size: int
    # Provenance: which corpus this scored against. If the Arabic documents are missing,
    # that is visible here rather than inferred from a surprising number.
    corpus_languages: list[str]
    generated_at: datetime | None = Field(default_factory=lambda: datetime.now(UTC))

    retrieval: dict[Surface, list[RetrievalAggregate]] = Field(default_factory=dict)
    contract: ContractAggregate
    answer_quality: AnswerQualityMetrics = Field(default_factory=AnswerQualityMetrics)

    case_results: list[CaseRetrievalResult] = Field(default_factory=list)
    contract_results: list[CaseContractResult] = Field(default_factory=list)
    threshold_outcomes: list[ThresholdOutcome] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def failed_thresholds(self) -> list[ThresholdOutcome]:
        return [outcome for outcome in self.threshold_outcomes if not outcome.passed]

    def stable(self) -> "EvaluationReport":
        """The same report without the timestamp, for the committed baseline: a baseline
        should diff only when a number changes.
        """
        return self.model_copy(update={"generated_at": None})
