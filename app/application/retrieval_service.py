import asyncio

from app.application.fusion import reciprocal_rank_fusion
from app.application.ports import (
    EmbeddingPort,
    KeywordRetrieverPort,
    RerankerPort,
    VectorStorePort,
)
from app.domain.models import Question, RetrievedChunk


class RetrievalService:
    """Owns the retrieve -> rerank half of the pipeline: embed the query, run dense and
    keyword search in parallel, fuse the two ranked lists with RRF, rerank, and return
    the top_k evidence chunks.

    QAApplicationService sees only `retrieve()` and the chunks it returns — embeddings,
    the vector store, BM25, fusion, and the reranker are all internal to this service.
    """

    # Each retrieval method is asked for more candidates than the caller wants, so
    # fusion and reranking have real material to work with before we narrow to top_k.
    _CANDIDATE_POOL_SIZE = 20

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        keyword_retriever: KeywordRetrieverPort,
        reranker: RerankerPort,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._keyword_retriever = keyword_retriever
        self._reranker = reranker

    async def retrieve(self, question: Question, top_k: int = 5) -> list[RetrievedChunk]:
        pool_size = max(top_k, self._CANDIDATE_POOL_SIZE)
        query_embedding = await self._embedding.embed(question.text)

        # Dense and keyword search are independent; gather lets them overlap once the
        # vector store is a real network-backed service.
        #
        # Only the keyword leg is language-restricted. A shared-token match across
        # languages is noise for BM25 (an Arabic report matching an English question on
        # the digit "7"), but for a multilingual embedder like BGE-M3 a cross-language
        # match is a genuine semantic hit and should be kept.
        dense_results, sparse_results = await asyncio.gather(
            self._vector_store.search(query_embedding, top_k=pool_size),
            self._keyword_retriever.search(
                question.text, top_k=pool_size, language=question.language
            ),
        )

        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        reranked = await self._reranker.rerank(question, fused)
        return reranked[:top_k]
