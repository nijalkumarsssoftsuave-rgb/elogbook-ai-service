from app.domain.models import PermissionScope
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS
from app.infrastructure.retrieval.permission_resolver import (
    PermissionResolver,
    RoleGrant,
)


async def _resolve(*roles: str) -> list[str]:
    scope = await PermissionResolver().resolve(PermissionScope(roles=list(roles)))
    return scope.source_ids


# --- sources ------------------------------------------------------------------------------


async def test_a_viewer_may_read_every_source() -> None:
    assert await _resolve("viewer") == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_a_contractor_may_read_only_the_general_operations_source() -> None:
    assert await _resolve("contractor") == [SHIFT_LOGS]


async def test_an_empty_scope_resolves_to_an_empty_search_set() -> None:
    """No roles means no sources, not "all sources". A scope that failed to populate must
    fail closed.
    """
    scope = await PermissionResolver().resolve(PermissionScope())

    assert scope.sources == []
    assert scope.is_empty


async def test_an_unrecognised_role_contributes_nothing_rather_than_raising() -> None:
    """A token minted by a newer upstream backend should degrade to seeing less, never to
    a 500.
    """
    assert await _resolve("some-future-role") == []


async def test_roles_are_additive() -> None:
    assert await _resolve("contractor", "viewer") == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_resolution_order_does_not_depend_on_the_order_roles_arrive() -> None:
    assert await _resolve("viewer", "contractor") == await _resolve("contractor", "viewer")


async def test_the_shipped_catalogue_applies_no_organisational_filters() -> None:
    """No document carries an area or department yet, so granting one would grant access to
    nothing. This pins that today's callers are unrestricted on all three.
    """
    scope = await PermissionResolver().resolve(PermissionScope.from_roles(["viewer"]))

    assert scope.area_ids == []
    assert scope.department_ids == []
    assert scope.company_ids == []


# --- organisational filters, against an injected role map ------------------------------------

_GRANTS = {
    "north-lead": RoleGrant(sources=(SHIFT_LOGS,), area_ids=("north",)),
    "south-lead": RoleGrant(sources=(SHIFT_LOGS,), area_ids=("south",)),
    "maintenance": RoleGrant(sources=(SHIFT_LOGS,), department_ids=("maintenance",)),
    "auditor": RoleGrant(sources=(SHIFT_LOGS, INCIDENTS)),  # unrestricted on every filter
}


async def _resolve_with(*roles: str):
    return await PermissionResolver(role_grants=_GRANTS).resolve(
        PermissionScope(roles=list(roles))
    )


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
    resolver = PermissionResolver()

    assert not [
        attribute
        for attribute in vars(resolver)
        if "retriev" in attribute or "corpus" in attribute or "store" in attribute
    ]
