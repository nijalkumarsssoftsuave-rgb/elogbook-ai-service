from app.application.permission.permission_resolver import RoleGrant
from app.domain.models import Source
from app.domain.permission import BaseRole, CustomRole
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

# Which roles may read what. No grant here carries an organisational filter: no document
# carries an area, department or company attribute yet, so shipping a filter would grant
# access to nothing. The base roles would be unrestricted regardless -- the resolver
# guarantees that from the role type rather than from this table happening to be empty.
ROLE_GRANTS: dict[str, RoleGrant] = {
    BaseRole.VIEWER: RoleGrant(sources=_EVERY_SOURCE),
    BaseRole.SUPERVISOR: RoleGrant(sources=_EVERY_SOURCE),
    BaseRole.ADMIN: RoleGrant(sources=_EVERY_SOURCE),
    # Contractors see day-to-day operations but neither incident reports nor the safety
    # and visitor records. This is the role that makes exclusion observable.
    CustomRole.CONTRACTOR: RoleGrant(sources=(SHIFT_LOGS,)),
    # CustomRole.AREA_MANAGER is deliberately absent. It is a custom role, so it is the
    # kind of role that *would* carry an area filter -- but no document records an area
    # yet, and retrieval fails closed on a missing attribute, so shipping that grant would
    # entitle its holders to nothing at all.
}
