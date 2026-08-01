from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.fixture_corpus import FixtureDocument


async def test_query_ranks_the_matching_logbook_entry_first() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search("fire alarm triggered during the night shift", top_k=3)

    assert results[0].chunk_id == "log-006"
    assert all(chunk.score > 0 for chunk in results)


async def test_query_with_no_matching_terms_returns_nothing() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search("zzzz qqqq nonexistent gibberish", top_k=5)

    assert results == []


async def test_results_carry_source_title_metadata_for_citations() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search("forklift near miss in aisle 7", top_k=1)

    assert results[0].metadata["source_title"] == "Near-Miss Report - Aisle 7"
    assert results[0].metadata["source"] == "bm25"


async def test_search_respects_top_k() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search("shift maintenance safety inspection report", top_k=2)

    assert len(results) <= 2


async def test_rarer_term_outranks_a_term_common_across_the_corpus() -> None:
    """IDF at work: a term appearing in one document should pull that document up."""
    corpus = [
        FixtureDocument("a", "doc-a", "shift report forklift collision", {}),
        FixtureDocument("b", "doc-b", "shift report routine check", {}),
        FixtureDocument("c", "doc-c", "shift report routine check", {}),
    ]
    retriever = BM25KeywordRetriever(corpus=corpus)

    results = await retriever.search("shift forklift", top_k=3)

    assert results[0].chunk_id == "a"


async def test_empty_corpus_returns_nothing_rather_than_dividing_by_zero() -> None:
    retriever = BM25KeywordRetriever(corpus=[])

    assert await retriever.search("anything at all", top_k=5) == []
