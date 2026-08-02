from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from app.domain.permission import SearchScope, Source
from app.domain.retrieval.date_range import DateRange
from app.domain.retrieval.retrieval_filter import RetrievalFilter

# The metadata keys a document declares, and the single place they are spelled. A typo in
# one retriever would look exactly like a document that does not carry the attribute --
# which is to say, like a correct exclusion.
AREA_KEY = "area_id"
DEPARTMENT_KEY = "department_id"
COMPANY_KEY = "company_id"
TAGS_KEY = "tags"
STATUS_KEY = "status"
RECORDED_ON_KEY = "recorded_on"


class EffectiveSearchScope(BaseModel):
    """What retrieval is actually allowed and asked to search: an entitlement narrowed by a
    request.

    Built by `combine`, never by hand at a call site. The combination is the one place in
    the system where a permitted set and a requested set meet, and it has exactly one rule:
    **a filter can narrow what a scope permits and can never widen it.**

    That rule needs a third state, which is why this type exists rather than another
    SearchScope. On a SearchScope an empty filter list means *unrestricted*; but a caller
    entitled to the north area who asks for the south has an empty *intersection*, and
    returning an empty list there would silently promote "nothing matches" into "everything
    matches". Such a scope is marked unsatisfiable instead, and retrieval returns nothing --
    which travels up the pipeline as an ordinary grounded refusal.
    """

    sources: list[Source] = Field(default_factory=list)
    area_ids: list[str] = Field(default_factory=list)
    department_ids: list[str] = Field(default_factory=list)
    company_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    date_range: DateRange | None = None
    # False when some dimension narrowed to nothing. Distinct from "no filters", which is
    # the permissive case and the common one.
    is_satisfiable: bool = True

    @classmethod
    def combine(
        cls, search_scope: SearchScope, filters: RetrievalFilter | None = None
    ) -> "EffectiveSearchScope":
        """Narrows an entitlement by a request, or reports that the two cannot both hold."""
        filters = filters or RetrievalFilter()

        narrowed: dict[str, list[str] | None] = {
            "area_ids": cls._narrow(search_scope.area_ids, filters.area_ids),
            "department_ids": cls._narrow(
                search_scope.department_ids, filters.department_ids
            ),
            "company_ids": cls._narrow(search_scope.company_ids, filters.company_ids),
            # Tags and status carry no entitlement today -- no role grants by them -- so the
            # caller's request is the only constraint. Passing the scope's (always empty)
            # side through `_narrow` anyway means the day a grant does restrict by tag, this
            # already narrows instead of being overwritten.
            "tags": cls._narrow([], filters.tags),
            "statuses": cls._narrow([], filters.statuses),
        }

        if any(values is None for values in narrowed.values()):
            return cls(sources=list(search_scope.sources), is_satisfiable=False)

        return cls(
            sources=list(search_scope.sources),
            area_ids=narrowed["area_ids"] or [],
            department_ids=narrowed["department_ids"] or [],
            company_ids=narrowed["company_ids"] or [],
            tags=narrowed["tags"] or [],
            statuses=narrowed["statuses"] or [],
            date_range=filters.date_range,
        )

    @staticmethod
    def _narrow(permitted: list[str], requested: list[str]) -> list[str] | None:
        """Combines one dimension. `None` means the two cannot both be satisfied.

        Three cases, and the order matters:
        - the caller asked for nothing, so whatever the scope permits stands;
        - the scope is unrestricted here, so the caller's request stands alone;
        - both constrain, so only their overlap survives -- and an empty overlap is
          impossible rather than unrestricted.
        """
        if not requested:
            return list(permitted)
        if not permitted:
            return sorted(set(requested))
        overlap = sorted(set(permitted) & set(requested))
        return overlap or None

    @property
    def is_empty(self) -> bool:
        """Nothing to search: no permitted source, or a combination nothing can satisfy."""
        return not self.sources or not self.is_satisfiable

    @property
    def source_ids(self) -> list[str]:
        return [source.source_id for source in self.sources]

    def permits(self, metadata: Mapping[str, Any]) -> bool:
        """Whether a document declaring this metadata is inside the scope.

        The single definition of what a filter means, called by every retriever rather than
        reimplemented in each. Matching is **fail-closed throughout**: a non-empty filter is
        strict, and a document that does not declare the attribute at all cannot satisfy it.
        Treating a missing attribute as permitted would let any undated document slip past
        every date filter, which is the failure this exists to prevent.
        """
        if not self.is_satisfiable:
            return False
        return (
            self._matches_one_of(metadata.get(AREA_KEY), self.area_ids)
            and self._matches_one_of(metadata.get(DEPARTMENT_KEY), self.department_ids)
            and self._matches_one_of(metadata.get(COMPANY_KEY), self.company_ids)
            and self._matches_one_of(metadata.get(STATUS_KEY), self.statuses)
            and self._matches_any_tag(metadata.get(TAGS_KEY))
            and self._matches_date(metadata.get(RECORDED_ON_KEY))
        )

    @staticmethod
    def _matches_one_of(declared: Any, allowed: list[str]) -> bool:
        if not allowed:
            return True
        return declared in allowed

    def _matches_any_tag(self, declared: Any) -> bool:
        """A document carries several tags, so a filter naming several means "any of these".

        Deliberately different from the id fields, where a document has one value to match.
        Requiring *all* the requested tags would make a two-tag filter almost unanswerable,
        and is not what a person picking tags in a UI expects.
        """
        if not self.tags:
            return True
        if not isinstance(declared, Sequence) or isinstance(declared, str | bytes):
            # A document with no tag list, or with a single string where a list belongs,
            # cannot satisfy a tag filter.
            return False
        return bool(set(declared) & set(self.tags))

    def _matches_date(self, declared: Any) -> bool:
        if self.date_range is None:
            return True
        if isinstance(declared, date):
            return self.date_range.contains(declared)
        if isinstance(declared, str):
            try:
                return self.date_range.contains(date.fromisoformat(declared))
            except ValueError:
                # An unparseable date is not a date. Excluded, for the same reason a missing
                # one is: a filter that silently passes malformed data is not a filter.
                return False
        return False
