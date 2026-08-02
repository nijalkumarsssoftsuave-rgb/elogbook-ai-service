import pytest

from app.application.permission.permission_resolver import PermissionResolver, RoleGrant
from app.domain.models import PermissionScope, SearchScope
from app.domain.permission import BaseRole, CustomRole
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS
from app.infrastructure.retrieval.permission_catalogue import ROLE_GRANTS, SOURCE_CATALOGUE


def _resolver(role_grants: dict[str, RoleGrant] | None = None) -> PermissionResolver:
    return PermissionResolver(SOURCE_CATALOGUE, role_grants or ROLE_GRANTS)


async def _resolve(*roles: str) -> list[str]:
    scope = await _resolver().resolve(PermissionScope(roles=list(roles)))
    return scope.source_ids


# --- sources ------------------------------------------------------------------------------


async def test_a_viewer_may_read_every_source() -> None:
    assert await _resolve(BaseRole.VIEWER) == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_a_contractor_may_read_only_the_general_operations_source() -> None:
    assert await _resolve(CustomRole.CONTRACTOR) == [SHIFT_LOGS]


async def test_an_empty_scope_resolves_to_an_empty_search_set() -> None:
    """No roles means no sources, not "all sources". A scope that failed to populate must
    fail closed.
    """
    scope = await _resolver().resolve(PermissionScope())

    assert scope.sources == []
    assert scope.is_empty


async def test_an_unrecognised_role_contributes_nothing_rather_than_raising() -> None:
    """A token minted by a newer upstream backend should degrade to seeing less, never to
    a 500.
    """
    assert await _resolve("some-future-role") == []


async def test_roles_are_additive() -> None:
    assert await _resolve(CustomRole.CONTRACTOR, BaseRole.VIEWER) == [
        SHIFT_LOGS,
        INCIDENTS,
        SAFETY,
    ]


async def test_resolution_order_does_not_depend_on_the_order_roles_arrive() -> None:
    assert await _resolve(BaseRole.VIEWER, CustomRole.CONTRACTOR) == await _resolve(
        CustomRole.CONTRACTOR, BaseRole.VIEWER
    )


# --- base roles retrieve unrestricted -------------------------------------------------------


@pytest.mark.parametrize("role", [BaseRole.VIEWER, BaseRole.SUPERVISOR, BaseRole.ADMIN])
async def test_every_base_role_resolves_to_an_unrestricted_search_scope(
    role: BaseRole,
) -> None:
    scope = await _resolver().resolve(PermissionScope.from_roles([role]))

    assert scope.area_ids == []
    assert scope.department_ids == []
    assert scope.company_ids == []
    # Asserted alongside, so the empty filters cannot pass by the role resolving to nothing
    # at all -- which is how this test would go quietly vacuous.
    assert scope.source_ids == [SHIFT_LOGS, INCIDENTS, SAFETY]


@pytest.mark.parametrize("role", [BaseRole.VIEWER, BaseRole.SUPERVISOR, BaseRole.ADMIN])
async def test_a_filter_on_a_base_role_grant_does_not_restrict_its_holder(
    role: BaseRole,
) -> None:
    """The rule ES-332 makes real, and the test that stops it being an accident of the
    shipped table. A base role is unrestricted because of what it *is*, so a filter added
    to its grant -- by a future edit, or a registry that starts supplying them -- must not
    quietly start narrowing anyone.
    """
    tampered = {
        role: RoleGrant(
            sources=(SHIFT_LOGS, INCIDENTS, SAFETY),
            area_ids=("north",),
            department_ids=("maintenance",),
            company_ids=("acme",),
        )
    }

    scope = await _resolver(tampered).resolve(PermissionScope.from_roles([role]))

    assert scope.area_ids == []
    assert scope.department_ids == []
    assert scope.company_ids == []
    assert scope.source_ids == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_a_base_role_lifts_a_custom_roles_restriction() -> None:
    """Roles are additive: holding one more may widen access but never narrow it. Someone
    who is both an admin and an area manager is an admin.
    """
    grants = {
        BaseRole.ADMIN: RoleGrant(sources=(SHIFT_LOGS, INCIDENTS)),
        CustomRole.AREA_MANAGER: RoleGrant(sources=(SHIFT_LOGS,), area_ids=("north",)),
    }

    scope = await _resolver(grants).resolve(
        PermissionScope.from_roles([BaseRole.ADMIN, CustomRole.AREA_MANAGER])
    )

    assert scope.area_ids == []
    assert scope.source_ids == [SHIFT_LOGS, INCIDENTS]


async def test_a_base_role_that_grants_nothing_cannot_lift_a_restriction() -> None:
    """The inverse, and the one worth guarding. A base role absent from the catalogue
    entitles its holder to nothing, so it must not be able to widen what another role
    allows -- otherwise naming an unknown base role in a token would strip filters.
    """
    grants = {CustomRole.AREA_MANAGER: RoleGrant(sources=(SHIFT_LOGS,), area_ids=("north",))}

    scope = await _resolver(grants).resolve(
        PermissionScope.from_roles([BaseRole.ADMIN, CustomRole.AREA_MANAGER])
    )

    assert scope.area_ids == ["north"]
    assert scope.source_ids == [SHIFT_LOGS]


async def test_the_shipped_catalogue_applies_no_organisational_filters() -> None:
    """No document carries an area or department yet, so granting one would grant access to
    nothing. This pins that today's callers are unrestricted on all three.
    """
    for role in ROLE_GRANTS:
        scope = await _resolver().resolve(PermissionScope.from_roles([role]))

        assert scope.area_ids == []
        assert scope.department_ids == []
        assert scope.company_ids == []


# --- organisational filters, against an injected role map ------------------------------------

# Every role here is custom: base roles are unrestricted by definition, so a filter only
# ever means something on a custom role.
_GRANTS = {
    "north-lead": RoleGrant(sources=(SHIFT_LOGS,), area_ids=("north",)),
    "south-lead": RoleGrant(sources=(SHIFT_LOGS,), area_ids=("south",)),
    "maintenance": RoleGrant(sources=(SHIFT_LOGS,), department_ids=("maintenance",)),
    "auditor": RoleGrant(sources=(SHIFT_LOGS, INCIDENTS)),  # unrestricted on every filter
}


async def _resolve_with(*roles: str) -> SearchScope:
    return await _resolver(_GRANTS).resolve(PermissionScope(roles=list(roles)))


async def test_a_restricted_role_carries_its_filter_into_the_search_scope() -> None:
    scope = await _resolve_with("north-lead")

    assert scope.area_ids == ["north"]
    assert scope.source_ids == [SHIFT_LOGS]


async def test_two_restricted_roles_union_their_filters() -> None:
    scope = await _resolve_with("north-lead", "south-lead")

    assert scope.area_ids == ["north", "south"]


async def test_an_unrestricted_role_makes_the_combination_unrestricted() -> None:
    """The subtlety worth pinning. Roles are additive -- holding one more can widen access
    but never narrow it -- so combining a role that says nothing about areas with one
    restricted to "north" must leave the caller unrestricted, not pinned to north.
    Intersecting here would quietly shrink what a multi-role user can see.
    """
    scope = await _resolve_with("north-lead", "auditor")

    assert scope.area_ids == []
    assert scope.source_ids == [SHIFT_LOGS, INCIDENTS]


async def test_filters_on_different_attributes_do_not_bleed_into_each_other() -> None:
    scope = await _resolve_with("north-lead", "maintenance")

    # north-lead is unrestricted on department and maintenance is unrestricted on area, so
    # the union leaves both unrestricted.
    assert scope.area_ids == []
    assert scope.department_ids == []


async def test_a_single_role_restricted_on_one_attribute_leaves_the_others_open() -> None:
    scope = await _resolve_with("maintenance")

    assert scope.department_ids == ["maintenance"]
    assert scope.area_ids == []
    assert scope.company_ids == []


async def test_unrecognised_roles_are_ignored_when_combined_with_a_real_one() -> None:
    scope = await _resolve_with("north-lead", "not-a-role")

    assert scope.area_ids == ["north"]
    assert scope.source_ids == [SHIFT_LOGS]


async def test_the_resolver_reads_no_documents() -> None:
    """"RBAC -> SearchScope. Nothing else." made checkable: the resolver holds a catalogue
    and a role map, and no retriever it could call.
    """
    resolver = _resolver()

    assert not [
        attribute
        for attribute in vars(resolver)
        if "retriev" in attribute or "corpus" in attribute or "store" in attribute
    ]
