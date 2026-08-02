from pydantic import BaseModel, Field


class Source(BaseModel):
    """One logical corpus that retrieval can be pointed at."""

    source_id: str
    display_name: str


class SearchScope(BaseModel):
    """The concrete search set a caller is entitled to: which sources, narrowed by which
    organisational filters.

    The output of permission resolution and the only thing retrieval consults about access.
    Where `PermissionScope` says *who the caller is*, this says *what that entitles them to
    search* -- and separating the two is what lets entitlements grow richer without the
    caller's identity changing shape.

    An empty filter list means **no constraint**, not "nothing allowed": the default scope
    over a source is everything in it. A non-empty one is enforced strictly, so a document
    that does not declare the attribute cannot satisfy it.
    """

    sources: list[Source] = Field(default_factory=list)
    area_ids: list[str] = Field(default_factory=list)
    department_ids: list[str] = Field(default_factory=list)
    company_ids: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """No sources means nothing to search, whatever the filters say."""
        return not self.sources

    @property
    def source_ids(self) -> list[str]:
        return [source.source_id for source in self.sources]

    @property
    def is_unrestricted(self) -> bool:
        """True when no organisational filter narrows the permitted sources.

        Named because "every filter list is empty" is the shape of an unrestricted scope,
        and reading that off three separate lists at each call site invites one of them
        being forgotten.
        """
        return not (self.area_ids or self.department_ids or self.company_ids)
