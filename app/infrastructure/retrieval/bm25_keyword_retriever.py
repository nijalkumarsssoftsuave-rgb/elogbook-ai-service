import math
from collections import Counter

from app.domain.models import RetrievedChunk
from app.infrastructure.retrieval.fixture_corpus import DEFAULT_FIXTURE_CORPUS, FixtureDocument
from app.infrastructure.retrieval.text_normalization import tokenize


class BM25KeywordRetriever:
    """Real Okapi BM25 keyword retrieval over an in-memory corpus.

    The scoring is genuine BM25, not a stub — only the corpus is a fixture, standing in
    for a real ingested index until the document pipeline exists. The index is built once
    at construction and reused for every query.

    All languages share one index. Arabic and English share no tokens, so cross-language
    matches score zero and drop out, but the Arabic documents do raise doc_count and
    shift the average document length, which scales every English score (measured:
    +20-35%, with rankings unchanged). If a future corpus ever moves an English *ranking*,
    the fix is per-language BM25 statistics rather than splitting the retriever.
    """

    # Standard Okapi BM25 parameters: k1 controls term-frequency saturation, b controls
    # how strongly document length normalises the score.
    _K1 = 1.5
    _B = 0.75

    def __init__(self, corpus: list[FixtureDocument] | None = None) -> None:
        self._corpus = list(DEFAULT_FIXTURE_CORPUS) if corpus is None else list(corpus)
        # Index time. `search` runs the query through the same `tokenize`; that shared
        # call is the only reason a normalized query term can match a document term.
        tokenized = [tokenize(document.text) for document in self._corpus]
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

    async def search(
        self, query_text: str, top_k: int = 5, language: str | None = None
    ) -> list[RetrievedChunk]:
        query_terms = tokenize(query_text)  # Query time. Same function as index time.
        scores = [self._score(index, query_terms) for index in range(len(self._corpus))]
        ranked = sorted(range(len(self._corpus)), key=lambda index: scores[index], reverse=True)

        results: list[RetrievedChunk] = []
        for index in ranked:
            if len(results) == top_k:
                break
            if scores[index] <= 0:
                continue  # no query term matched this document
            document = self._corpus[index]
            # Scoring stays global (one index, one set of corpus statistics); only the
            # results are restricted. The two corpora share tokens -- ASCII digits from
            # timestamps, Latin acronyms -- so without this an English question about
            # "aisle 7" surfaces the Arabic near-miss report.
            if language is not None and document.language != language:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=document.chunk_id,
                    document_id=document.document_id,
                    text=document.text,
                    score=scores[index],
                    metadata={
                        **document.metadata,
                        "source": "bm25",
                        "language": document.language,
                    },
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
