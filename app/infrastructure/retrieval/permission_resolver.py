from typing import NamedTuple

from app.domain.models import PermissionScope, SearchScope, Source
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS

# The sources retrieval can be pointed at. A fixture catalogue, standing in for a real
# registry once the ingestion pipeline exists -- the same way fixture_corpus.py stands in
# for the ingested index.
SOURCE_CATALOGUE: dict[str, Source] = {
    SHIFT_LOGS: Source(source_id=SHIFT_LOGS, display_name="Shift Logs"),
    INCIDENTS: Source(source_id=INCIDENTS, display_name="Incident Reports"),
    SAFETY: Source(source_id=SAFETY, display_name="Safety and Visitor Records"),
}


class RoleGrant(NamedTuple):
    """What one role entitles its holder to.

    Every filter tuple is empty by default, and empty means *unrestricted* rather than
    *nothing* -- a role that says nothing about departments does not restrict by
    department.
    """

    sources: tuple[str, ...]
    area_ids: tuple[str, ...] = ()
    department_ids: tuple[str, ...] = ()
    company_ids: tuple[str, ...] = ()


# Which roles may read what. The shipped catalogue leaves every role unrestricted on area,
# department and company: no document carries those attributes yet, so granting one would
# mean granting access to nothing.
ROLE_GRANTS: dict[str, RoleGrant] = {
    "viewer": RoleGrant(sources=(SHIFT_LOGS, INCIDENTS, SAFETY)),
    "supervisor": RoleGrant(sources=(SHIFT_LOGS, INCIDENTS, SAFETY)),
    # Contractors see day-to-day operations but neither incident reports nor the safety
    # and visitor records. This is the role that makes exclusion observable.
    "contractor": RoleGrant(sources=(SHIFT_LOGS,)),
}


class PermissionResolver:
    """Turns a caller's roles into the search set they are entitled to.

    RBAC in, SearchScope out, and nothing else: this class never touches a document. Every
    access decision the system makes is made here, once, so the retrieval below it can
    operate on a settled answer rather than re-deciding per result -- which is the
    arrangement that stops a restricted document leaking through a code path that forgot to
    check.

    Roles are **additive**: a caller gets the union of what their roles grant, so holding an
    extra role can widen access but never narrow it. That rule has a subtlety in the
    filters, and getting it backwards would quietly shrink what a multi-role user can see:
    a role with no area restriction is *unrestricted*, so combining it with an
    area-restricted role must leave the caller unrestricted, not intersected down to the
    restricted role's areas.

    An unrecognised role contributes nothing rather than raising. A token minted by a newer
    version of the upstream backend should degrade to "sees less", never to a 500.
    """

    def __init__(
        self,
        catalogue: dict[str, Source] | None = None,
        role_grants: dict[str, RoleGrant] | None = None,
    ) -> None:
        self._catalogue = dict(SOURCE_CATALOGUE if catalogue is None else catalogue)
        self._role_grants = dict(ROLE_GRANTS if role_grants is None else role_grants)

    async def resolve(self, scope: PermissionScope) -> SearchScope:
        grants = [
            self._role_grants[role] for role in scope.roles if role in self._role_grants
        ]
        if not grants:
            # No recognised role: an empty search set, which retrieval treats as "nothing
            # to search" rather than "no restriction".
            return SearchScope()

        permitted_sources = {source_id for grant in grants for source_id in grant.sources}
        return SearchScope(
            # Ordered by the catalogue, not by role iteration, so the resolved scope is
            # stable regardless of the order roles arrive in the token.
            sources=[
                source
                for source_id, source in self._catalogue.items()
                if source_id in permitted_sources
            ],
            area_ids=self._union_filter(grants, "area_ids"),
            department_ids=self._union_filter(grants, "department_ids"),
            company_ids=self._union_filter(grants, "company_ids"),
        )

    @staticmethod
    def _union_filter(grants: list[RoleGrant], attribute: str) -> list[str]:
        """Combines one filter across roles, honouring "empty means unrestricted".

        If any role is unrestricted on this attribute the union is unrestricted, so the
        result is an empty list. Otherwise it is the union of every role's allowed values.
        """
        values: list[tuple[str, ...]] = [getattr(grant, attribute) for grant in grants]
        if any(not value for value in values):
            return []
        return sorted({item for value in values for item in value})
