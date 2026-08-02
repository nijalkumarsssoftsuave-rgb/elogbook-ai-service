from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Bounds what a single request can ask for. Not a business rule -- a request naming ten
# thousand areas is either a mistake or an attempt to make the service do work on the
# caller's behalf, and neither deserves to reach retrieval.
_MAX_FILTER_VALUES = 50


class QueryFilters(BaseModel):
    """What the caller asked to narrow the search to.

    **Not to be confused with the filters on SearchScope**, and the distinction is the whole
    reason this is a separate type. A SearchScope says what the caller is *entitled* to read;
    this says what they are *interested* in. They must never be merged by assignment: a
    request naming an area the caller has no grant for can only ever shrink the result set,
    never reach past the scope into it. Keeping them as two types means combining them has
    to be written deliberately, by the component that owns access decisions, rather than
    happening by a field with the same name being copied across.

    Unknown keys are rejected rather than ignored. A caller who sends `area_id` for
    `area_ids` would otherwise get an unfiltered answer that looks exactly like a correctly
    filtered one, which is the kind of silence that gets noticed months later.
    """

    model_config = ConfigDict(extra="forbid")

    area_ids: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    department_ids: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    company_ids: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    # Inclusive on both ends: a caller asking for July means the whole of July.
    date_from: date | None = None
    date_to: date | None = None

    @field_validator("area_ids", "department_ids", "company_ids", mode="after")
    @classmethod
    def _reject_blank_values(cls, values: list[str]) -> list[str]:
        """A blank id is a client bug, not a filter.

        Accepting it would silently match nothing -- retrieval matches these fail-closed --
        so the caller would see an empty answer and no reason for it.
        """
        if any(not value.strip() for value in values):
            raise ValueError("filter values must not be blank")
        return values

    @model_validator(mode="after")
    def _reject_an_impossible_date_range(self) -> "QueryFilters":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError(
                f"date_from must not be after date_to: {self.date_from} > {self.date_to}"
            )
        return self

    @property
    def is_empty(self) -> bool:
        """True when the caller asked for no narrowing at all."""
        return not (
            self.area_ids
            or self.department_ids
            or self.company_ids
            or self.date_from
            or self.date_to
        )
