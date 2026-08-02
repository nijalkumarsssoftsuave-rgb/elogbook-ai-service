from enum import StrEnum


class RoleType(StrEnum):
    """Which kind of role a caller holds, and therefore how it may restrict them."""

    # An operational role from the platform's own vocabulary. Base roles are unrestricted
    # on the organisational filters: they describe what someone does, not which slice of
    # the organisation they may look at.
    BASE = "base"
    # A role a deployment defines for itself. A custom role may narrow access to particular
    # areas, departments or companies -- that is what it exists for.
    CUSTOM = "custom"


class BaseRole(StrEnum):
    """The base operational roles, and a closed set on purpose.

    Adding one is a deliberate act, because a base role is by definition unrestricted on
    area, department and company. Anything that should be able to carry an organisational
    restriction is a CustomRole instead.
    """

    VIEWER = "viewer"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"


class CustomRole(StrEnum):
    """The custom roles this deployment ships today.

    Unlike BaseRole this set is *not* closed -- a deployment defines its own, and
    `role_type` treats anything outside BaseRole as custom whether or not it is named here.
    The enum exists so the roles we do know about have one spelling rather than a string
    literal per call site.
    """

    CONTRACTOR = "contractor"
    AREA_MANAGER = "area-manager"


_BASE_ROLE_NAMES = frozenset(role.value for role in BaseRole)


def role_type(role: str) -> RoleType:
    """Classifies a role name from the upstream token.

    Base is the closed set, so the test is membership of it; everything else is custom.
    That direction matters: an unfamiliar role must never be *promoted* to base and so
    granted unrestricted reach. A role nobody recognises is custom, holds no grant, and
    contributes nothing.
    """
    return RoleType.BASE if role in _BASE_ROLE_NAMES else RoleType.CUSTOM
