from datetime import date

import pytest
from pydantic import ValidationError

from app.domain.retrieval import RetrievalFilter

# --- no filters ---------------------------------------------------------------------------


def test_filters_are_optional_and_default_to_asking_for_nothing() -> None:
    filters = RetrievalFilter()

    assert filters.is_empty
    assert filters.area_ids == []
    assert filters.date_from is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"area_ids": ["north"]},
        {"department_ids": ["maintenance"]},
        {"company_ids": ["acme-industrial"]},
        {"date_from": date(2026, 7, 1)},
        {"date_to": date(2026, 7, 31)},
    ],
)
def test_any_single_filter_makes_the_set_non_empty(kwargs: dict) -> None:
    """`is_empty` has to consider every field. One left out would make a filtered request
    look unfiltered to whatever reads it.
    """
    assert not RetrievalFilter(**kwargs).is_empty


# --- valid filters ------------------------------------------------------------------------


def test_the_documented_example_is_accepted() -> None:
    filters = RetrievalFilter.model_validate(
        {"area_ids": ["north"], "date_from": "2026-07-01", "date_to": "2026-07-31"}
    )

    assert filters.area_ids == ["north"]
    assert filters.date_from == date(2026, 7, 1)
    assert filters.date_to == date(2026, 7, 31)


def test_a_single_day_range_is_valid() -> None:
    """The bounds are inclusive, so from and to being equal asks for that one day rather
    than for nothing.
    """
    filters = RetrievalFilter(date_from=date(2026, 7, 1), date_to=date(2026, 7, 1))

    assert filters.date_from == filters.date_to


def test_an_open_ended_range_is_valid_in_either_direction() -> None:
    assert RetrievalFilter(date_from=date(2026, 7, 1)).date_to is None
    assert RetrievalFilter(date_to=date(2026, 7, 31)).date_from is None


# --- invalid filters ----------------------------------------------------------------------


def test_a_backwards_date_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="start must not be after end"):
        RetrievalFilter(date_from=date(2026, 7, 31), date_to=date(2026, 7, 1))


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_a_blank_filter_value_is_rejected(blank: str) -> None:
    """Accepting it would match nothing -- retrieval matches these fail-closed -- so the
    caller would get an empty answer and no reason for it.
    """
    with pytest.raises(ValidationError, match="must not be blank"):
        RetrievalFilter(area_ids=["north", blank])


def test_a_malformed_date_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"date_from": "the first of July"})


def test_an_unknown_filter_key_is_rejected_rather_than_ignored() -> None:
    """A caller sending `area_id` for `area_ids` would otherwise get an unfiltered answer
    that looks exactly like a correctly filtered one.
    """
    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"area_id": "north"})


def test_a_filter_value_that_is_not_a_list_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"area_ids": "north"})


def test_an_unbounded_filter_list_is_rejected() -> None:
    """A request naming thousands of areas is either a mistake or an attempt to make the
    service do work on the caller's behalf. Neither should reach retrieval.
    """
    with pytest.raises(ValidationError):
        RetrievalFilter(area_ids=[f"area-{n}" for n in range(51)])
