import pytest

from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.fixture_corpus import (
    ENGLISH_FIXTURE_CORPUS,
    FixtureDocument,
)

# Arabic appears here only as module-level constants so that assertion lines stay pure
# ASCII: a failing assertion prints its operands, and this project is developed on a
# cp1252 console where printing Arabic raises.
_ARABIC_ALARM_QUERY = "ما سبب إنذار الحريق في الوردية الليلية؟"
_ARABIC_ALARM_QUERY_DIACRITIZED = "ما سبب إنذار الحَرِيقُ في الوردية الليلية؟"
_ARABIC_FORKLIFT_QUERY = "صف حادث الرافعة الشوكية في الممر رقم 7"


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


_ENGLISH_BASELINE_QUERIES = [
    "fire alarm triggered during the night shift",
    "forklift near miss in aisle 7",
    "shift maintenance safety inspection report",
    "boiler pressure reading stable",
    "visitor contractor hvac inspection",
]


@pytest.mark.parametrize("query", _ENGLISH_BASELINE_QUERIES)
async def test_english_ranking_is_unchanged_by_the_arabic_documents(query: str) -> None:
    """Adding Arabic documents raises doc_count and shifts the average document length,
    so English *scores* move. This pins the thing that actually matters -- the ordering --
    against an English-only index.

    The language argument is what makes this hold: the two corpora share tokens (ASCII
    digits in timestamps, Latin acronyms), so an unrestricted search genuinely does
    return Arabic documents for some English queries.
    """
    english_only = BM25KeywordRetriever(corpus=ENGLISH_FIXTURE_CORPUS)
    bilingual = BM25KeywordRetriever()

    baseline = [chunk.chunk_id for chunk in await english_only.search(query, top_k=8)]
    actual = [
        chunk.chunk_id for chunk in await bilingual.search(query, top_k=8, language="en")
    ]

    assert actual == baseline


async def test_unrestricted_search_does_match_across_languages_on_shared_tokens() -> None:
    """Documents why the language argument exists. Timestamps and Latin acronyms are
    written identically in both corpora, so BM25 alone cannot keep them apart.
    """
    retriever = BM25KeywordRetriever()

    unrestricted = await retriever.search("forklift near miss in aisle 7", top_k=8)

    assert any(chunk.chunk_id.startswith("log-ar-") for chunk in unrestricted)


async def test_arabic_query_ranks_the_matching_arabic_entry_first() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search(_ARABIC_ALARM_QUERY, top_k=3)

    assert results[0].chunk_id == "log-ar-006"


async def test_diacritized_arabic_query_still_matches_the_plain_document() -> None:
    """Normalization proven end-to-end through BM25, not just in isolation: without
    diacritic stripping the vowelled word would tokenize differently and miss.
    """
    retriever = BM25KeywordRetriever()

    results = await retriever.search(_ARABIC_ALARM_QUERY_DIACRITIZED, top_k=3)

    assert results[0].chunk_id == "log-ar-006"


async def test_arabic_search_returns_no_english_documents() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search(_ARABIC_FORKLIFT_QUERY, top_k=8, language="ar")

    assert results
    assert all(chunk.metadata["language"] == "ar" for chunk in results)


async def test_english_search_returns_no_arabic_documents() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search("forklift near miss in aisle 7", top_k=8, language="en")

    assert results
    assert all(chunk.metadata["language"] == "en" for chunk in results)


async def test_language_restriction_still_fills_top_k() -> None:
    """Filtering happens while walking the ranked list, not after truncating it, so a
    restricted search still returns k results when k matching documents exist.
    """
    retriever = BM25KeywordRetriever()

    results = await retriever.search("shift report safety maintenance", top_k=3, language="en")

    assert len(results) == 3


async def test_results_carry_the_document_language_in_metadata() -> None:
    retriever = BM25KeywordRetriever()

    results = await retriever.search("fire alarm night shift", top_k=1)

    assert results[0].metadata["language"] == "en"
