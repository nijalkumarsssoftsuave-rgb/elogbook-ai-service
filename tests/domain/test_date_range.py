from datetime import date

import pytest
from pydantic import ValidationError

from app.domain.retrieval import DateRange

JULY = DateRange(start=date(2026, 7, 1), end=date(2026, 7, 31))


def test_both_bounds_are_inclusive() -> None:
    """What a person asking for "July" means."""
    assert JULY.contains(date(2026, 7, 1))
    assert JULY.contains(date(2026, 7, 31))


def test_dates_outside_the_range_are_excluded() -> None:
    assert not JULY.contains(date(2026, 6, 30))
    assert not JULY.contains(date(2026, 8, 1))


def test_a_range_open_at_the_end_has_no_upper_bound() -> None:
    since = DateRange(start=date(2026, 7, 1))

    assert since.contains(date(2999, 1, 1))
    assert not since.contains(date(2026, 6, 30))


def test_a_range_open_at_the_start_has_no_lower_bound() -> None:
    until = DateRange(end=date(2026, 7, 31))

    assert until.contains(date(1970, 1, 1))
    assert not until.contains(date(2026, 8, 1))


def test_a_single_day_range_contains_only_that_day() -> None:
    one_day = DateRange(start=date(2026, 7, 15), end=date(2026, 7, 15))

    assert one_day.contains(date(2026, 7, 15))
    assert not one_day.contains(date(2026, 7, 14))
    assert not one_day.contains(date(2026, 7, 16))


def test_a_range_with_neither_bound_is_not_constructible() -> None:
    """That is not a range, it is the absence of one. Allowing it would give callers two
    ways to say "no date filter" -- `None` and a range that matches everything -- and only
    one of them would be checked at the places that ask whether a filter is set.
    """
    with pytest.raises(ValidationError, match="at least one bound"):
        DateRange()


def test_a_backwards_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="start must not be after end"):
        DateRange(start=date(2026, 7, 31), end=date(2026, 7, 1))
