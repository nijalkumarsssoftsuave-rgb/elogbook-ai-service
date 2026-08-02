import asyncio

from app.application.ports import KeywordRetrieverPort, VectorStorePort
from app.domain.models import RetrievalCandidates, RetrievedChunk, SourceSearchRequest
from app.domain.retrieval import EffectiveSearchScope


class MultiSourceRetriever:
    """Searches every source it is handed, with both retrieval methods, and merges the
    candidates.

    It performs **no authorization**: the sources on the request are already the permitted
    ones, resolved upstream. Re-deciding access here would mean two places could disagree
    about who may read what. It performs **no reranking** either -- fusion and reranking
    stay downstream, untouched.

    Merging happens *per method*: all the dense candidates become one list, all the keyword
    candidates another, and Reciprocal Rank Fusion downstream reconciles the two exactly as
    it always has. Fusing the per-source lists directly would be the obvious alternative
    and is wrong here -- it compresses ranks, so a rank-1 hit in a three-document source
    outranks a globally stronger one from a larger source. Measured against the evaluation
    set, that reading moved 20 of 30 query orderings; this one leaves every ranking and
    every threshold exactly where it was.

    That merge is valid because one BM25 index backs all sources today, which makes scores
    comparable across them. When sources move to separate backends that stops being true,
    and the merge has to become score normalization or per-source fusion. That is a
    deliberate decision for the ticket which introduces the second backend, not something
    to let happen quietly.
    """

    def __init__(
        self, vector_store: VectorStorePort, keyword_retriever: KeywordRetrieverPort
    ) -> None:
        self._vector_store = vector_store
        self._keyword_retriever = keyword_retriever

    async def search(self, request: SourceSearchRequest) -> RetrievalCandidates:
        scope = request.search_scope
        if scope.is_empty:
            # Nothing permitted, so nothing to search. Returning empty candidates rather
            # than querying an unrestricted index is the whole point of the ticket.
            return RetrievalCandidates()

        # Every source is searched concurrently, which is what makes this worth doing at
        # all once the sources sit behind separate network-backed backends.
        per_source = await asyncio.gather(
            *(self._search_one(request, source.source_id) for source in scope.sources)
        )

        dense: list[RetrievedChunk] = []
        sparse: list[RetrievedChunk] = []
        for source_dense, source_sparse in per_source:
            dense.extend(source_dense)
            sparse.extend(source_sparse)

        return RetrievalCandidates(
            # Which sources were *queried*, not which returned a hit: a permitted source
            # that matched nothing is still part of the answer to "where did we look?".
            searched_source_ids=scope.source_ids,
            dense=self._merge(self._within_scope(dense, scope), request.limit_per_source),
            sparse=self._merge(self._within_scope(sparse, scope), request.limit_per_source),
        )

    @staticmethod
    def _within_scope(
        chunks: list[RetrievedChunk], scope: EffectiveSearchScope
    ) -> list[RetrievedChunk]:
        """Drops any candidate the scope does not permit.

        Since ES-338 the retrievers filter as they walk their own ranked lists, so in normal
        operation this removes nothing. It stays as a backstop, and deliberately: an adapter
        that ignores the scope it was handed -- a new backend, a stub, a driver that quietly
        drops an unsupported clause -- would otherwise leak restricted documents into an
        answer. Being handed the same EffectiveSearchScope means it cannot disagree with the
        retrievers about what the filter means, only about whether it was applied.
        """
        return [chunk for chunk in chunks if scope.permits(chunk.metadata)]

    async def _search_one(
        self, request: SourceSearchRequest, source_id: str
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        # Dense and keyword search are independent, so they overlap here as well. Both are
        # handed the scope so they can exclude while walking their own ranked list: a
        # filtered search then still fills top_k, instead of returning whatever survives
        # a post-hoc filter applied to an already-truncated list.
        #
        # Only the keyword leg is language-restricted, matching RetrievalService's original
        # reasoning: a shared-token match across languages is noise for BM25, but a genuine
        # semantic hit for a multilingual embedder like BGE-M3.
        dense, sparse = await asyncio.gather(
            self._vector_store.search(
                request.query_embedding,
                top_k=request.limit_per_source,
                source_id=source_id,
                scope=request.search_scope,
            ),
            self._keyword_retriever.search(
                request.query_text,
                top_k=request.limit_per_source,
                language=request.language,
                source_id=source_id,
                scope=request.search_scope,
            ),
        )
        return list(dense), list(sparse)

    @staticmethod
    def _merge(chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        """Orders one method's candidates from every source into a single ranked list.

        Sorted descending by score, which reconstructs exactly the ranking an unrestricted
        search over the permitted documents would have produced.
        """
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:limit]
