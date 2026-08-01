import math
import re
from collections import Counter

from app.domain.models import RetrievedChunk
from app.infrastructure.retrieval.fixture_corpus import DEFAULT_FIXTURE_CORPUS, FixtureDocument

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25KeywordRetriever:
    """Real Okapi BM25 keyword retrieval over an in-memory corpus.

    The scoring is genuine BM25, not a stub — only the corpus is a fixture, standing in
    for a real ingested index until the document pipeline exists. The index is built once
    at construction and reused for every query.
    """

    # Standard Okapi BM25 parameters: k1 controls term-frequency saturation, b controls
    # how strongly document length normalises the score.
    _K1 = 1.5
    _B = 0.75

    def __init__(self, corpus: list[FixtureDocument] | None = None) -> None:
        self._corpus = list(DEFAULT_FIXTURE_CORPUS) if corpus is None else list(corpus)
        tokenized = [_tokenize(document.text) for document in self._corpus]
        self._term_frequencies = [Counter(tokens) for tokens in tokenized]
        self._doc_lengths = [len(tokens) for tokens in tokenized]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._idf = self._compute_idf(tokenized)

    def _compute_idf(self, tokenized: list[list[str]]) -> dict[str, float]:
        doc_count = len(tokenized)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        return {
            term: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in document_frequency.items()
        }

    async def search(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = _tokenize(query_text)
        scores = [self._score(index, query_terms) for index in range(len(self._corpus))]
        ranked = sorted(range(len(self._corpus)), key=lambda index: scores[index], reverse=True)

        results: list[RetrievedChunk] = []
        for index in ranked[:top_k]:
            if scores[index] <= 0:
                continue  # no query term matched this document
            document = self._corpus[index]
            results.append(
                RetrievedChunk(
                    chunk_id=document.chunk_id,
                    document_id=document.document_id,
                    text=document.text,
                    score=scores[index],
                    metadata={**document.metadata, "source": "bm25"},
                )
            )
        return results

    def _score(self, doc_index: int, query_terms: list[str]) -> float:
        if self._avg_doc_length == 0:
            return 0.0

        score = 0.0
        doc_length = self._doc_lengths[doc_index]
        frequencies = self._term_frequencies[doc_index]
        for term in query_terms:
            term_frequency = frequencies.get(term)
            if not term_frequency:
                continue
            length_norm = 1 - self._B + self._B * doc_length / self._avg_doc_length
            numerator = term_frequency * (self._K1 + 1)
            denominator = term_frequency + self._K1 * length_norm
            score += self._idf.get(term, 0.0) * numerator / denominator
        return score
