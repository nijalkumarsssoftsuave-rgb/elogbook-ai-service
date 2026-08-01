"""Loading and cross-checking for evaluation datasets and thresholds.

Everything is read with an explicit UTF-8 encoding. On Windows the default encoding is
still cp1252, which raises UnicodeDecodeError the moment it meets the Arabic dataset.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from tests.eval.schema import EvaluationDataset, EvaluationReport, Surface, ThresholdOutcome

DATASETS_DIR = Path(__file__).parent / "datasets"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"
BASELINE_DIR = Path(__file__).parent / "baseline"
REPORTS_DIR = Path(__file__).parent / "reports"

ENGLISH_DATASET_PATH = DATASETS_DIR / "logbook_qa_en.v1.json"
ARABIC_DATASET_PATH = DATASETS_DIR / "logbook_qa_ar.v1.json"


def load_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def dataset_path_for(language: str) -> Path:
    paths = {"en": ENGLISH_DATASET_PATH, "ar": ARABIC_DATASET_PATH}
    if language not in paths:
        raise ValueError(f"no dataset for language '{language}'; expected one of {sorted(paths)}")
    return paths[language]


def validate_against_corpus(dataset: EvaluationDataset, corpus_chunk_ids: Iterable[str]) -> None:
    """Fails if a case cites a chunk id the corpus does not contain.

    Without this, a typo like 'log-06' scores 0.0 for that case and looks exactly like a
    retrieval regression rather than a dataset bug.
    """
    known = set(corpus_chunk_ids)
    unknown = sorted(
        {
            chunk_id
            for case in dataset.cases
            for chunk_id in case.relevant_chunk_ids
            if chunk_id not in known
        }
    )
    if unknown:
        raise ValueError(
            f"dataset '{dataset.name}' references chunk ids that are not in the corpus: "
            f"{', '.join(unknown)}"
        )


class ThresholdBound(BaseModel):
    """A single floor or ceiling. Exactly one direction must be given -- a bound with
    neither would silently never fail.
    """

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _exactly_one_direction(self) -> "ThresholdBound":
        if (self.min is None) == (self.max is None):
            raise ValueError("a threshold needs exactly one of 'min' or 'max'")
        return self

    def check(self, metric_path: str, actual: float) -> ThresholdOutcome:
        passed = actual >= self.min if self.min is not None else actual <= (self.max or 0.0)
        return ThresholdOutcome(
            metric_path=metric_path, floor=self.min, ceiling=self.max,
            actual=actual, passed=passed,
        )


class LanguageThresholds(BaseModel):
    """Floors for one language.

    `retrieval` is keyed by surface, then by k as a string (JSON object keys are always
    strings), then by metric name. `contract` is keyed by metric name directly.
    """

    model_config = ConfigDict(extra="forbid")

    retrieval: dict[str, dict[str, dict[str, ThresholdBound]]] = {}
    contract: dict[str, ThresholdBound] = {}

    def evaluate(self, report: EvaluationReport) -> list[ThresholdOutcome]:
        outcomes: list[ThresholdOutcome] = []
        for surface_name, by_k in self.retrieval.items():
            aggregates = {
                aggregate.k: aggregate
                for aggregate in report.retrieval.get(Surface(surface_name), [])
            }
            for k_text, by_metric in by_k.items():
                aggregate = aggregates.get(int(k_text))
                if aggregate is None:
                    continue
                for metric_name, bound in by_metric.items():
                    actual = getattr(aggregate, metric_name)
                    outcomes.append(
                        bound.check(f"{surface_name}.k={k_text}.{metric_name}", actual)
                    )
        for metric_name, bound in self.contract.items():
            actual = getattr(report.contract, metric_name)
            outcomes.append(bound.check(f"contract.{metric_name}", actual))
        return outcomes


class ThresholdFile(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerates the explanatory "_procedure" key

    version: int
    languages: dict[str, LanguageThresholds]


def load_thresholds(path: Path = THRESHOLDS_PATH) -> ThresholdFile:
    return ThresholdFile.model_validate_json(path.read_text(encoding="utf-8"))


def thresholds_for(language: str, path: Path = THRESHOLDS_PATH) -> LanguageThresholds:
    thresholds = load_thresholds(path)
    if language not in thresholds.languages:
        raise ValueError(f"no thresholds configured for language '{language}'")
    return thresholds.languages[language]
