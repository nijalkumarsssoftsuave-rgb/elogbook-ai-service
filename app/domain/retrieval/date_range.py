from datetime import date

from pydantic import BaseModel, model_validator


class DateRange(BaseModel):
    """An inclusive span of days, open at either end.

    Inclusive on both ends because that is what a person asking for "July" means, and open
    at either end because "everything since the first" and "everything up to the last" are
    both ordinary requests. A range with neither end is not constructible -- that is not a
    range, it is the absence of one, and callers say that with `None`.
    """

    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _reject_an_impossible_range(self) -> "DateRange":
        if self.start is None and self.end is None:
            raise ValueError("a date range needs at least one bound")
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start must not be after end: {self.start} > {self.end}")
        return self

    def contains(self, value: date) -> bool:
        """Whether a date falls inside the range, treating both bounds as inclusive."""
        if self.start and value < self.start:
            return False
        return not (self.end and value > self.end)
