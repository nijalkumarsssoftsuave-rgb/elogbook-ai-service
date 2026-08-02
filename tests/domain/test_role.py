import pytest

from app.domain.permission import BaseRole, CustomRole, RoleType, role_type


@pytest.mark.parametrize("role", list(BaseRole))
def test_every_base_role_classifies_as_base(role: BaseRole) -> None:
    assert role_type(role) is RoleType.BASE


@pytest.mark.parametrize("role", list(CustomRole))
def test_every_known_custom_role_classifies_as_custom(role: CustomRole) -> None:
    assert role_type(role) is RoleType.CUSTOM


def test_a_role_nobody_has_heard_of_classifies_as_custom() -> None:
    """Base is the closed set and the test is membership of it, so an unfamiliar role
    cannot be promoted to base and granted unrestricted reach. It is custom, holds no
    grant, and contributes nothing.
    """
    assert role_type("some-future-role") is RoleType.CUSTOM


def test_classification_is_case_sensitive_and_does_not_guess() -> None:
    """Role names arrive from an upstream token. Matching "Admin" loosely would be a way to
    acquire base-role reach with a near-miss spelling.
    """
    assert role_type("Admin") is RoleType.CUSTOM
    assert role_type("ADMIN") is RoleType.CUSTOM
    assert role_type(" admin") is RoleType.CUSTOM


def test_the_two_role_sets_do_not_overlap() -> None:
    assert not {role.value for role in BaseRole} & {role.value for role in CustomRole}


def test_role_names_are_the_strings_the_token_carries() -> None:
    """StrEnum members compare equal to their wire spelling, which is what lets them be used
    directly as grant-table keys and in a PermissionScope.
    """
    assert BaseRole.VIEWER == "viewer"
    assert CustomRole.AREA_MANAGER == "area-manager"
