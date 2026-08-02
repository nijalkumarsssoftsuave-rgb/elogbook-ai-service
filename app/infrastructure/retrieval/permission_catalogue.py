from app.application.permission.permission_resolver import RoleGrant
from app.domain.permission import BaseRole, CustomRole, Source
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS

# The data PermissionResolver resolves against. It lives here rather than beside the
# resolver because it is what a real deployment will read out of a registry: the rules for
# combining grants are policy and stay in the application layer, the grants themselves are
# configuration and belong down here with the fixtures they name.

# The sources retrieval can be pointed at. A fixture catalogue, standing in for a real
# registry once the ingestion pipeline exists -- the same way fixture_corpus.py stands in
# for the ingested index.
SOURCE_CATALOGUE: dict[str, Source] = {
    SHIFT_LOGS: Source(source_id=SHIFT_LOGS, display_name="Shift Logs"),
    INCIDENTS: Source(source_id=INCIDENTS, display_name="Incident Reports"),
    SAFETY: Source(source_id=SAFETY, display_name="Safety and Visitor Records"),
}

_EVERY_SOURCE = (SHIFT_LOGS, INCIDENTS, SAFETY)

# The organisational values the corpus actually declares. Named here rather than spelled
# inline in a grant, because a filter naming a value no document carries silently entitles
# its holder to nothing -- fail-closed matching cannot tell a typo from a real exclusion.
NORTH = "north"
SOUTH = "south"
MAINTENANCE = "maintenance"

# Which roles may read what.
#
# A base role's grant never carries an organisational filter -- the resolver would override
# it anyway, so writing one here would only mislead. Filters belong to custom roles, which
# is what they are for.
ROLE_GRANTS: dict[str, RoleGrant] = {
    BaseRole.VIEWER: RoleGrant(sources=_EVERY_SOURCE),
    BaseRole.SUPERVISOR: RoleGrant(sources=_EVERY_SOURCE),
    BaseRole.ADMIN: RoleGrant(sources=_EVERY_SOURCE),
    # Contractors see day-to-day operations but neither incident reports nor the safety
    # and visitor records. Restricted by *source*, and unrestricted within it -- the role
    # that makes source exclusion observable.
    CustomRole.CONTRACTOR: RoleGrant(sources=(SHIFT_LOGS,)),
    # An area manager may read every source, but only for the area they run. Restricted by
    # *attribute* rather than by source, which is the other half of the same idea and the
    # role that makes organisational filtering observable.
    CustomRole.AREA_MANAGER: RoleGrant(sources=_EVERY_SOURCE, area_ids=(NORTH,)),
}
