from app.application.fusion import reciprocal_rank_fusion
from app.application.ports import (
    EmbeddingPort,
    MultiSourceRetrieverPort,
    RerankerPort,
)
from app.domain.models import (
    Question,
    RetrievedChunk,
    SearchScope,
    SourceSearchRequest,
)


class RetrievalService:
    """Owns the retrieve -> rerank half of the pipeline:

    search every source in the caller's scope -> merge the candidates -> fuse dense and
    keyword results with RRF -> rerank -> return the top_k evidence.

    It is handed an already-resolved SearchScope rather than resolving one itself, so
    authorization is a decision made once, upstream, by the component named for it. That
    keeps this service about searching -- embeddings, the vector store, BM25, fusion and
    the reranker are its internals, and access rules are not among them.
    """

    # Each source is asked for more candidates than the caller wants, so fusion and
    # reranking have real material to work with before we narrow to top_k.
    _CANDIDATE_POOL_SIZE = 20

    def __init__(
        self,
        embedding: EmbeddingPort,
        multi_source_retriever: MultiSourceRetrieverPort,
        reranker: RerankerPort,
    ) -> None:
        self._embedding = embedding
        self._multi_source_retriever = multi_source_retriever
        self._reranker = reranker

    async def retrieve(
        self, question: Question, search_scope: SearchScope, top_k: int = 5
    ) -> list[RetrievedChunk]:
        if search_scope.is_empty:
            # Nothing permitted, nothing retrieved. Returning early rather than searching
            # an unrestricted index is the point of the whole arrangement; the pipeline
            # above then has no evidence and refuses, which is the correct answer to a
            # question the caller is not entitled to have answered.
            #
            # This guard stays here rather than at the caller on purpose. Since the scope
            # now arrives from outside, this is the last place that can fail closed.
            return []

        pool_size = max(top_k, self._CANDIDATE_POOL_SIZE)
        # Embedded once, however many sources end up being searched.
        query_embedding = await self._embedding.embed(question.text)

        candidates = await self._multi_source_retriever.search(
            SourceSearchRequest(
                query_text=question.text,
                query_embedding=query_embedding,
                language=question.language,
                search_scope=search_scope,
                limit_per_source=pool_size,
            )
        )

        # From here down nothing knows about sources or permissions: two ranked lists in,
        # fused, reranked, sliced -- exactly as before this service learned about scope.
        fused = reciprocal_rank_fusion([candidates.dense, candidates.sparse])
        reranked = await self._reranker.rerank(question, fused)
        return reranked[:top_k]
