from app.domain.models import PermissionScope, Source
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS

# The sources retrieval can be pointed at. A fixture catalogue, standing in for a real
# registry once the ingestion pipeline exists -- the same way fixture_corpus.py stands in
# for the ingested index.
SOURCE_CATALOGUE: dict[str, Source] = {
    SHIFT_LOGS: Source(source_id=SHIFT_LOGS, display_name="Shift Logs"),
    INCIDENTS: Source(source_id=INCIDENTS, display_name="Incident Reports"),
    SAFETY: Source(source_id=SAFETY, display_name="Safety and Visitor Records"),
}

# Which roles may read which sources. Deliberately additive: a caller sees the union of
# everything their roles grant, so adding a role can widen access but never narrow it.
ROLE_SOURCES: dict[str, tuple[str, ...]] = {
    "viewer": (SHIFT_LOGS, INCIDENTS, SAFETY),
    "supervisor": (SHIFT_LOGS, INCIDENTS, SAFETY),
    # Contractors see day-to-day operations but neither incident reports nor the safety
    # and visitor records. This is the role that makes exclusion observable.
    "contractor": (SHIFT_LOGS,),
}


class RoleBasedSourceResolver:
    """Turns a permission scope into the sources it may read.

    Resolution only -- this class never touches a document. Keeping authorization in its
    own step is what lets the retriever below trust the list it is given rather than
    re-deciding access for every result it finds, which is the arrangement that stops a
    restricted document leaking through a code path that forgot to check.

    An unrecognised role contributes nothing rather than raising. A token minted by a
    newer version of the upstream backend should degrade to "sees less", never to a 500.
    """

    def __init__(
        self,
        catalogue: dict[str, Source] | None = None,
        role_sources: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._catalogue = dict(SOURCE_CATALOGUE if catalogue is None else catalogue)
        self._role_sources = dict(ROLE_SOURCES if role_sources is None else role_sources)

    async def resolve(self, scope: PermissionScope) -> list[Source]:
        permitted: set[str] = set()
        for role in scope.roles:
            permitted.update(self._role_sources.get(role, ()))

        # Ordered by the catalogue, not by role iteration, so the resolved list is stable
        # regardless of the order roles arrive in the token.
        return [
            source for source_id, source in self._catalogue.items() if source_id in permitted
        ]
