from app.domain.models import PermissionScope
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS
from app.infrastructure.retrieval.source_resolver import RoleBasedSourceResolver


async def _resolve(*roles: str) -> list[str]:
    sources = await RoleBasedSourceResolver().resolve(PermissionScope(roles=list(roles)))
    return [source.source_id for source in sources]


async def test_a_viewer_may_read_every_source() -> None:
    assert await _resolve("viewer") == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_a_contractor_may_read_only_the_general_operations_source() -> None:
    assert await _resolve("contractor") == [SHIFT_LOGS]


async def test_an_empty_scope_resolves_to_nothing() -> None:
    """No roles means no sources, not "all sources". A scope that failed to populate must
    fail closed.
    """
    assert await _resolve() == []


async def test_an_unrecognised_role_contributes_nothing_rather_than_raising() -> None:
    """A token minted by a newer upstream backend should degrade to seeing less, never to
    a 500.
    """
    assert await _resolve("some-future-role") == []


async def test_roles_are_additive() -> None:
    """A caller sees the union of what their roles grant, so holding an extra role can
    widen access but never narrow it.
    """
    assert await _resolve("contractor", "viewer") == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_resolution_order_does_not_depend_on_the_order_roles_arrive() -> None:
    assert await _resolve("viewer", "contractor") == await _resolve("contractor", "viewer")


async def test_duplicate_grants_do_not_duplicate_sources() -> None:
    assert await _resolve("viewer", "supervisor") == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_the_resolver_reads_no_documents() -> None:
    """"No retrieval, only resolution" made checkable: the resolver is constructed with a
    catalogue and a role map and nothing else -- there is no retriever it could call.
    """
    resolver = RoleBasedSourceResolver()

    assert not [
        attribute
        for attribute in vars(resolver)
        if "retriev" in attribute or "corpus" in attribute or "store" in attribute
    ]


async def test_the_catalogue_and_role_map_can_be_overridden() -> None:
    """Deployments will not share one hardcoded role map; the defaults are a fixture."""
    resolver = RoleBasedSourceResolver(role_sources={"viewer": (INCIDENTS,)})

    resolved = await resolver.resolve(PermissionScope.from_roles(["viewer"]))

    assert [source.source_id for source in resolved] == [INCIDENTS]
