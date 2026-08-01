from app.application.fusion import reciprocal_rank_fusion
from app.domain.models import RetrievedChunk


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id="d1", text="text", score=score)


def test_chunk_found_by_both_methods_outranks_chunks_found_by_one() -> None:
    dense = [_chunk("c1", 0.9), _chunk("c2", 0.5)]
    sparse = [_chunk("c2", 5.0), _chunk("c3", 3.0)]

    fused = reciprocal_rank_fusion([dense, sparse])

    # c2 scores 1/61 + 1/62; c1 scores 1/61 (rank 1 of one list); c3 scores 1/62.
    assert [chunk.chunk_id for chunk in fused] == ["c2", "c1", "c3"]


def test_duplicate_chunk_ids_are_merged_into_one_result() -> None:
    dense = [_chunk("c1", 0.9)]
    sparse = [_chunk("c1", 5.0)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert len(fused) == 1
    assert fused[0].chunk_id == "c1"


def test_fused_score_replaces_the_original_per_method_score() -> None:
    dense = [_chunk("c1", 0.9)]
    sparse = [_chunk("c1", 5.0)]

    fused = reciprocal_rank_fusion([dense, sparse])

    # Both lists rank it first, so the fused score is 2 * 1/(60 + 1).
    assert fused[0].score == 2 / 61


def test_empty_input_returns_empty_result() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
