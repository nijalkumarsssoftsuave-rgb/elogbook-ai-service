from datetime import date

import pytest

from app.domain.permission import SearchScope, Source
from app.domain.retrieval import EffectiveSearchScope, RetrievalFilter

SHIFT_LOGS = Source(source_id="shift-logs", display_name="Shift Logs")


def _combine(scope_kwargs: dict | None = None, **filter_kwargs) -> EffectiveSearchScope:
    return EffectiveSearchScope.combine(
        SearchScope(sources=[SHIFT_LOGS], **(scope_kwargs or {})),
        RetrievalFilter(**filter_kwargs),
    )


# --- empty filters preserve current behaviour ------------------------------------------------


def test_no_filters_leaves_the_entitlement_exactly_as_it_was() -> None:
    scope = _combine({"area_ids": ["north"], "department_ids": ["maintenance"]})

    assert scope.area_ids == ["north"]
    assert scope.department_ids == ["maintenance"]
    assert scope.is_satisfiable
    assert scope.source_ids == ["shift-logs"]


def test_an_unrestricted_entitlement_with_no_filters_stays_unrestricted() -> None:
    scope = _combine()

    assert scope.area_ids == []
    assert scope.date_range is None
    assert scope.permits({"area_id": "anywhere", "recorded_on": "2020-01-01"})


def test_an_unrestricted_scope_permits_a_document_declaring_nothing() -> None:
    """The property every pre-ES-338 caller depends on: with no filters, a document that
    declares no attributes is still retrievable.
    """
    assert _combine().permits({})


# --- filters narrow ---------------------------------------------------------------------------


def test_a_filter_narrows_an_unrestricted_entitlement() -> None:
    scope = _combine(area_ids=["north"])

    assert scope.area_ids == ["north"]
    assert scope.permits({"area_id": "north"})
    assert not scope.permits({"area_id": "south"})


def test_a_filter_inside_the_entitlement_narrows_to_itself() -> None:
    scope = _combine({"area_ids": ["north", "south"]}, area_ids=["north"])

    assert scope.area_ids == ["north"]


def test_a_filter_cannot_reach_outside_the_entitlement() -> None:
    """The rule the whole type exists for. Asking for two areas when entitled to one leaves
    the one -- never both.
    """
    scope = _combine({"area_ids": ["north"]}, area_ids=["north", "south"])

    assert scope.area_ids == ["north"]
    assert not scope.permits({"area_id": "south"})


def test_a_filter_disjoint_from_the_entitlement_is_unsatisfiable_not_unrestricted() -> None:
    """The trap this type was created to avoid. On a SearchScope an empty list means
    *unrestricted*, so returning the empty intersection would promote "nothing matches" into
    "everything matches" -- handing the caller precisely the documents they may not read.
    """
    scope = _combine({"area_ids": ["north"]}, area_ids=["south"])

    assert scope.is_satisfiable is False
    assert scope.is_empty
    assert not scope.permits({"area_id": "north"})
    assert not scope.permits({"area_id": "south"})
    assert not scope.permits({})


@pytest.mark.parametrize(
    "dimension", ["area_ids", "department_ids", "company_ids"]
)
def test_every_entitled_dimension_can_make_a_scope_unsatisfiable(dimension: str) -> None:
    """Walked rather than spot-checked: one dimension left out of the impossibility check
    would silently fall back to unrestricted on exactly that dimension.
    """
    scope = EffectiveSearchScope.combine(
        SearchScope(sources=[SHIFT_LOGS], **{dimension: ["permitted"]}),
        RetrievalFilter(**{dimension: ["requested"]}),
    )

    assert scope.is_satisfiable is False


# --- matching is fail-closed --------------------------------------------------------------------


def test_a_document_missing_the_filtered_attribute_is_excluded() -> None:
    """Treating a missing attribute as permitted would let any undeclared document slip past
    every filter, which is the failure this exists to prevent.
    """
    scope = _combine(department_ids=["maintenance"])

    assert not scope.permits({"area_id": "north"})


def test_filters_combine_as_and_not_or() -> None:
    scope = _combine(area_ids=["north"], department_ids=["maintenance"])

    assert scope.permits({"area_id": "north", "department_id": "maintenance"})
    assert not scope.permits({"area_id": "north", "department_id": "production"})
    assert not scope.permits({"area_id": "south", "department_id": "maintenance"})


# --- tags match on overlap, unlike the id dimensions ----------------------------------------------


def test_a_tag_filter_matches_a_document_carrying_any_of_the_tags() -> None:
    """A document carries several tags, so naming several means "any of these". Requiring
    all of them would make a two-tag filter almost unanswerable.
    """
    scope = _combine(tags=["safety", "alarm"])

    assert scope.permits({"tags": ["incident", "alarm"]})
    assert scope.permits({"tags": ["safety"]})
    assert not scope.permits({"tags": ["equipment", "handover"]})


def test_a_document_with_no_tags_cannot_satisfy_a_tag_filter() -> None:
    scope = _combine(tags=["safety"])

    assert not scope.permits({})
    assert not scope.permits({"tags": []})


def test_a_bare_string_where_a_tag_list_belongs_does_not_match() -> None:
    """Otherwise "safety" would match a filter for "s", "a", "f"... -- a substring match
    dressed up as a set intersection.
    """
    scope = _combine(tags=["s"])

    assert not scope.permits({"tags": "safety"})


# --- dates ------------------------------------------------------------------------------------


def test_a_date_range_includes_both_of_its_bounds() -> None:
    scope = _combine(date_from=date(2026, 7, 1), date_to=date(2026, 7, 31))

    assert scope.permits({"recorded_on": "2026-07-01"})
    assert scope.permits({"recorded_on": "2026-07-31"})
    assert not scope.permits({"recorded_on": "2026-06-30"})
    assert not scope.permits({"recorded_on": "2026-08-01"})


def test_an_open_ended_range_bounds_only_the_end_it_names() -> None:
    since = _combine(date_from=date(2026, 7, 1))
    until = _combine(date_to=date(2026, 7, 31))

    assert since.permits({"recorded_on": "2030-01-01"})
    assert not since.permits({"recorded_on": "2020-01-01"})
    assert until.permits({"recorded_on": "2020-01-01"})
    assert not until.permits({"recorded_on": "2030-01-01"})


def test_a_document_with_no_date_cannot_satisfy_a_date_filter() -> None:
    scope = _combine(date_from=date(2026, 7, 1))

    assert not scope.permits({"area_id": "north"})


def test_an_unparseable_date_is_excluded_rather_than_passed_through() -> None:
    """A filter that silently passes malformed data is not a filter."""
    scope = _combine(date_from=date(2026, 7, 1))

    assert not scope.permits({"recorded_on": "last Tuesday"})
    assert not scope.permits({"recorded_on": 20260702})


def test_a_real_date_object_matches_as_well_as_an_iso_string() -> None:
    """A future backend may hand back typed dates rather than the fixture's strings."""
    scope = _combine(date_from=date(2026, 7, 1), date_to=date(2026, 7, 31))

    assert scope.permits({"recorded_on": date(2026, 7, 15)})


# --- sources are not narrowed by filters -----------------------------------------------------


def test_filters_never_change_which_sources_are_searched() -> None:
    """Sources are an entitlement only. There is no request field for them, and adding one
    here by accident would let a caller aim at a source rather than merely narrow within it.
    """
    scope = _combine(area_ids=["north"], tags=["safety"])

    assert scope.source_ids == ["shift-logs"]


def test_an_entitlement_with_no_sources_is_empty_whatever_the_filters_say() -> None:
    scope = EffectiveSearchScope.combine(SearchScope(), RetrievalFilter(area_ids=["north"]))

    assert scope.is_empty
