from typing import NamedTuple

from app.domain.permission import PermissionScope, RoleType, SearchScope, Source, role_type


class RoleGrant(NamedTuple):
    """What one role entitles its holder to.

    Every filter tuple is empty by default, and empty means *unrestricted* rather than
    *nothing* -- a role that says nothing about departments does not restrict by
    department.

    Only a custom role should ever carry a filter here. A base role is unrestricted by
    definition, and the resolver enforces that regardless of what a grant claims.
    """

    sources: tuple[str, ...]
    area_ids: tuple[str, ...] = ()
    department_ids: tuple[str, ...] = ()
    company_ids: tuple[str, ...] = ()


class PermissionResolver:
    """Turns a caller's roles into the search set they are entitled to.

    RBAC in, SearchScope out, and nothing else: this class never touches a document. Every
    access decision the system makes is made here, once, so the retrieval below it can
    operate on a settled answer rather than re-deciding per result -- which is the
    arrangement that stops a restricted document leaking through a code path that forgot to
    check.

    It lives in the application layer because entitlement is policy, not I/O. The catalogue
    and the role map are the parts that will one day come from a store, so they are injected
    rather than reached for; the rules for combining them stay here.

    **The precedence rule, in full.** Roles combine in three steps, and this class is the
    only place any of them is decided:

    1. *Sources are unioned.* A caller may search every source any of their roles grants.
    2. *A base role overrides every custom restriction.* Holding one base operational role
       resolves to an unrestricted scope no matter what else the caller holds.
    3. *Otherwise custom restrictions union*, and "empty means unrestricted" applies -- one
       custom role that says nothing about areas leaves the combination unrestricted on
       areas.

    Every step points the same way: **holding an extra role can widen access but never
    narrow it.** That is the property worth protecting, because the alternatives break it in
    ways that are quiet rather than loud. Letting a custom restriction override a base role
    would mean granting someone an extra role *reduces* what they can see -- an admin who is
    also an area manager would lose sight of every other area. Intersecting restrictions is
    worse still: two custom roles covering different areas would cancel each other out and
    leave the caller with nothing.

    An unrecognised role contributes nothing rather than raising. A token minted by a newer
    version of the upstream backend should degrade to "sees less", never to a 500.
    """

    def __init__(
        self, catalogue: dict[str, Source], role_grants: dict[str, RoleGrant]
    ) -> None:
        self._catalogue = dict(catalogue)
        self._role_grants = dict(role_grants)

    async def resolve(self, scope: PermissionScope) -> SearchScope:
        granted = [
            (role, self._role_grants[role])
            for role in scope.roles
            if role in self._role_grants
        ]
        if not granted:
            # No recognised role: an empty search set, which retrieval treats as "nothing
            # to search" rather than "no restriction".
            return SearchScope()

        permitted_sources = {
            source_id for _, grant in granted for source_id in grant.sources
        }
        grants = [grant for _, grant in granted]
        unrestricted = self._base_role_overrides_custom_restrictions(granted)

        return SearchScope(
            # Ordered by the catalogue, not by role iteration, so the resolved scope is
            # stable regardless of the order roles arrive in the token.
            sources=[
                source
                for source_id, source in self._catalogue.items()
                if source_id in permitted_sources
            ],
            area_ids=self._filter(grants, "area_ids", unrestricted),
            department_ids=self._filter(grants, "department_ids", unrestricted),
            company_ids=self._filter(grants, "company_ids", unrestricted),
        )

    @staticmethod
    def _base_role_overrides_custom_restrictions(
        granted: list[tuple[str, RoleGrant]]
    ) -> bool:
        """Step 2 of the precedence rule: does any base role lift every custom restriction?

        A base role describes what someone does; a custom role describes which slice of the
        organisation they are confined to. Someone who is both a supervisor and an area
        manager is a supervisor, so no filter derived from the custom role may narrow them.

        Only roles that actually carry a grant count. A base role the catalogue does not
        recognise entitles its holder to nothing, and so must not be able to widen what
        another role allows -- otherwise naming an unknown base role in a token would be a
        way to strip filters.
        """
        return any(role_type(role) is RoleType.BASE for role, _ in granted)

    @staticmethod
    def _filter(grants: list[RoleGrant], attribute: str, unrestricted: bool) -> list[str]:
        """Combines one organisational filter across roles -- step 2 then step 3.

        Enforcing the base-role override here rather than by shipping empty grants means a
        filter accidentally added to a base role cannot quietly start restricting people.

        The union that follows honours "empty means unrestricted": if any custom role is
        unrestricted on this attribute the union is too, and the result is an empty list.
        """
        if unrestricted:
            return []
        values: list[tuple[str, ...]] = [getattr(grant, attribute) for grant in grants]
        if any(not value for value in values):
            return []
        return sorted({item for value in values for item in value})
