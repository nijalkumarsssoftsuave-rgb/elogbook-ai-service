from app.application.fusion import reciprocal_rank_fusion
from app.application.ports import (
    EmbeddingPort,
    MultiSourceRetrieverPort,
    RerankerPort,
    SourceResolverPort,
)
from app.domain.models import (
    RetrievalSearchContext,
    RetrievedChunk,
    SourceSearchRequest,
)


class RetrievalService:
    """Owns the retrieve -> rerank half of the pipeline:

    resolve the sources the caller may read -> search all of them -> merge the candidates
    -> fuse dense and keyword results with RRF -> rerank -> return the top_k evidence.

    QAApplicationService sees only `retrieve()` and the chunks it returns. Source
    resolution, embeddings, the vector store, BM25, fusion and the reranker are all
    internal to this service.

    Authorization happens once, at the top: the resolver decides which sources are in
    play, and everything below it operates on that decision rather than re-checking access
    per result. A permission failure therefore looks like an empty source list, not like a
    filter someone downstream might forget to apply.
    """

    # Each source is asked for more candidates than the caller wants, so fusion and
    # reranking have real material to work with before we narrow to top_k.
    _CANDIDATE_POOL_SIZE = 20

    def __init__(
        self,
        embedding: EmbeddingPort,
        source_resolver: SourceResolverPort,
        multi_source_retriever: MultiSourceRetrieverPort,
        reranker: RerankerPort,
    ) -> None:
        self._embedding = embedding
        self._source_resolver = source_resolver
        self._multi_source_retriever = multi_source_retriever
        self._reranker = reranker

    async def retrieve(self, context: RetrievalSearchContext) -> list[RetrievedChunk]:
        question, top_k = context.question, context.top_k

        sources = await self._source_resolver.resolve(context.permission_scope)
        if not sources:
            # Nothing permitted, nothing retrieved. Returning early rather than searching
            # an unrestricted index is the point of the whole arrangement; the pipeline
            # above then has no evidence and refuses, which is the correct answer to a
            # question the caller is not entitled to have answered.
            return []

        pool_size = max(top_k, self._CANDIDATE_POOL_SIZE)
        # Embedded once, however many sources end up being searched.
        query_embedding = await self._embedding.embed(question.text)

        candidates = await self._multi_source_retriever.search(
            SourceSearchRequest(
                query_text=question.text,
                query_embedding=query_embedding,
                language=question.language,
                sources=sources,
                limit_per_source=pool_size,
            )
        )

        # From here down nothing knows about sources or permissions: two ranked lists in,
        # fused, reranked, sliced -- exactly as before this service learned about scope.
        fused = reciprocal_rank_fusion([candidates.dense, candidates.sparse])
        reranked = await self._reranker.rerank(question, fused)
        return reranked[:top_k]
