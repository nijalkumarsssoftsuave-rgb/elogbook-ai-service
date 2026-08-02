from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.retrieval.date_range import DateRange

# Bounds what a single request can ask for. Not a business rule -- a request naming ten
# thousand areas is either a mistake or an attempt to make the service do work on the
# caller's behalf, and neither deserves to reach retrieval.
_MAX_FILTER_VALUES = 50


class RetrievalFilter(BaseModel):
    """What the caller asked to narrow the search to.

    **Not to be confused with the filters on SearchScope**, and the distinction is the whole
    reason this is a separate type. A SearchScope says what the caller is *entitled* to read;
    this says what they are *interested* in. They are never merged by assignment -- they are
    combined into an EffectiveSearchScope by a rule that can only ever narrow, which is what
    stops a request naming an unentitled area from reaching past its own scope.

    The date bounds stay flat on this model because that is the shape the request wire
    format uses, and are exposed through `date_range` for everything downstream: a value
    object that owns "is this date inside?" is worth having, and two spellings of the same
    span in the codebase is not.

    Unknown keys are rejected rather than ignored. A caller who sends `area_id` for
    `area_ids` would otherwise get an unfiltered answer that looks exactly like a correctly
    filtered one, which is the kind of silence that gets noticed months later.
    """

    model_config = ConfigDict(extra="forbid")

    area_ids: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    department_ids: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    company_ids: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    # A document carries several tags; a filter naming several means "any of these". That is
    # the opposite of the id fields, where a document has exactly one value to match.
    tags: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    # Plural, like every other list here. The ticket calls this dimension "status"; a field
    # holding a list of them reads better named for what it holds.
    statuses: list[str] = Field(default_factory=list, max_length=_MAX_FILTER_VALUES)
    # Inclusive on both ends: a caller asking for July means the whole of July.
    date_from: date | None = None
    date_to: date | None = None

    @field_validator("area_ids", "department_ids", "company_ids", "tags", "statuses")
    @classmethod
    def _reject_blank_values(cls, values: list[str]) -> list[str]:
        """A blank value is a client bug, not a filter.

        Accepting it would silently match nothing -- these are matched fail-closed -- so the
        caller would see an empty answer and no reason for it.
        """
        if any(not value.strip() for value in values):
            raise ValueError("filter values must not be blank")
        return values

    @model_validator(mode="after")
    def _reject_an_impossible_date_range(self) -> "RetrievalFilter":
        # Constructing the range is the validation: DateRange owns the rule, so there is no
        # second copy of it here to fall out of step.
        _ = self.date_range
        return self

    @property
    def date_range(self) -> DateRange | None:
        if self.date_from is None and self.date_to is None:
            return None
        return DateRange(start=self.date_from, end=self.date_to)

    @property
    def is_empty(self) -> bool:
        """True when the caller asked for no narrowing at all."""
        return not (
            self.area_ids
            or self.department_ids
            or self.company_ids
            or self.tags
            or self.statuses
            or self.date_from
            or self.date_to
        )
