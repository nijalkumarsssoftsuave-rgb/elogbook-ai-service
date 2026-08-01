"""Fixtures scoped to the evaluation suite.

Deliberately does not define `app`, `client` or `auth_headers`: shadowing a root fixture
from a subdirectory is legal but confusing to anyone debugging.
"""

import pytest

from tests.eval.loader import ARABIC_DATASET_PATH, ENGLISH_DATASET_PATH, load_dataset
from tests.eval.runner import EvaluationRunner
from tests.eval.schema import EvaluationDataset
from tests.eval.surfaces import EvaluationSurfaces, build_surfaces


@pytest.fixture(scope="session")
def evaluation_surfaces() -> EvaluationSurfaces:
    """Session-scoped: BM25 indexes the corpus at construction, and building it once is
    the point. Nothing in the pipeline is mutable, so sharing it is safe.
    """
    return build_surfaces()


@pytest.fixture(scope="session")
def evaluation_runner(evaluation_surfaces: EvaluationSurfaces) -> EvaluationRunner:
    return EvaluationRunner(surfaces=evaluation_surfaces)


@pytest.fixture(scope="session")
def english_dataset() -> EvaluationDataset:
    return load_dataset(ENGLISH_DATASET_PATH)


@pytest.fixture(scope="session")
def arabic_dataset() -> EvaluationDataset:
    return load_dataset(ARABIC_DATASET_PATH)
