from app.domain.models import RetrievedChunk

# Conventional RRF constant from Cormack, Clarke & Buettcher (SIGIR 2009). A larger k
# flattens the influence of top ranks; 60 is the widely-used default.
_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = _RRF_K
) -> list[RetrievedChunk]:
    """Merges several ranked chunk lists into one, using Reciprocal Rank Fusion.

    Each chunk's fused score is the sum of ``1 / (k + rank)`` over every list it appears
    in, so a chunk found by both dense and keyword retrieval outranks one found by only
    a single method. Results are deduplicated by ``chunk_id`` — the chunk, not the
    document, is the unit citations reference, so distinct chunks of one document are
    legitimately separate evidence.

    The returned chunks carry the fused RRF score in ``score``, replacing the original
    per-method score: cosine similarities and BM25 scores live on different, incomparable
    scales, whereas RRF scores are rank-based and comparable across sources.
    """
    rrf_scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            incumbent = best_chunk.get(chunk.chunk_id)
            if incumbent is None or chunk.score > incumbent.score:
                best_chunk[chunk.chunk_id] = chunk

    fused_ids = sorted(rrf_scores, key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
    return [
        best_chunk[chunk_id].model_copy(update={"score": rrf_scores[chunk_id]})
        for chunk_id in fused_ids
    ]
