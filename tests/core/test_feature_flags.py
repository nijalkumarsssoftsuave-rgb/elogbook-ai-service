import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.feature_flags import Feature, FeatureFlags
from app.infrastructure.configuration.feature_flag_provider import FeatureFlagProvider

# --- default values -------------------------------------------------------------------------


def test_every_capability_is_on_by_default() -> None:
    """These are kill switches, not release gates. All four capabilities are shipped and
    tested, so adding a flag file must not silently switch any of them off.
    """
    flags = FeatureFlags()

    assert flags.voice_enabled
    assert flags.confidence_scoring_enabled
    assert flags.human_review_enabled
    assert flags.advanced_filters_enabled


def test_missing_configuration_is_not_an_error() -> None:
    """A deployment that says nothing about features gets all of them. Requiring the file
    would make the flags a deployment step rather than an override.
    """
    assert FeatureFlags().model_dump() == FeatureFlags(_env_file=None).model_dump()


# --- enabled and disabled values --------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["voice_enabled", "human_review_enabled", "advanced_filters_enabled"],
)
def test_any_single_capability_can_be_switched_off(field: str) -> None:
    flags = FeatureFlags(**{field: False})

    assert getattr(flags, field) is False
    # And only that one: switching one off must not disturb its neighbours.
    others = {"voice_enabled", "human_review_enabled", "advanced_filters_enabled"} - {field}
    assert all(getattr(flags, other) for other in others)


def test_switching_off_scoring_requires_switching_off_review_too() -> None:
    flags = FeatureFlags(confidence_scoring_enabled=False, human_review_enabled=False)

    assert flags.confidence_scoring_enabled is False
    assert flags.human_review_enabled is False


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
def test_a_flag_is_read_from_the_environment_as_a_boolean(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Operators write words, not Python literals. All the usual spellings of "off" have to
    mean off, because a flag that reads "no" as True fails in the least safe direction.
    """
    monkeypatch.setenv("FEATURE_VOICE_ENABLED", value)

    assert FeatureFlags().voice_enabled is False


# --- invalid configuration ---------------------------------------------------------------------


def test_review_without_scoring_is_rejected_at_startup() -> None:
    """Routing decides what to queue from the confidence band, so this combination would
    queue nothing while looking configured to queue everything. Startup is the only place
    that mistake is cheap to notice -- otherwise the queue stays quietly empty for a month.
    """
    with pytest.raises(ValidationError, match="requires FEATURE_CONFIDENCE_SCORING_ENABLED"):
        FeatureFlags(confidence_scoring_enabled=False, human_review_enabled=True)


def test_a_value_that_is_not_a_boolean_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATURE_VOICE_ENABLED", "sometimes")

    with pytest.raises(ValidationError):
        FeatureFlags()


def test_an_unknown_feature_key_is_ignored_rather_than_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike request filters, an unrecognised flag must not stop the service booting: it
    is usually a flag from a newer build, and refusing to start is a worse failure than
    running without it.
    """
    monkeypatch.setenv("FEATURE_TIME_TRAVEL_ENABLED", "true")

    assert FeatureFlags().voice_enabled is True


# --- the two settings classes share one .env ---------------------------------------------------


def test_a_feature_flag_in_the_environment_does_not_break_the_main_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap this nearly shipped with. Settings and FeatureFlags read the same .env, and
    if either rejected the other's keys, following the documented way to switch a capability
    off would stop the service starting.
    """
    monkeypatch.setenv("FEATURE_VOICE_ENABLED", "false")
    monkeypatch.setenv("SERVICE_JWT_SECRET", "a" * 40)

    assert Settings().service_jwt_algorithm == "HS256"
    assert FeatureFlags().voice_enabled is False


# --- the provider answers the port ---------------------------------------------------------------


def test_the_provider_reports_what_the_flags_say() -> None:
    provider = FeatureFlagProvider(
        FeatureFlags(
            voice_enabled=False,
            confidence_scoring_enabled=False,
            human_review_enabled=False,
            advanced_filters_enabled=True,
        )
    )

    assert provider.is_voice_enabled() is False
    assert provider.is_confidence_scoring_enabled() is False
    assert provider.is_human_review_enabled() is False
    assert provider.is_advanced_filter_enabled() is True


def test_every_feature_in_the_enum_has_a_flag_behind_it() -> None:
    """Walks the vocabulary rather than the four someone remembered: a Feature with no flag
    could be raised as disabled while nothing could ever disable it.
    """
    fields = set(FeatureFlags.model_fields)

    assert {f"{feature.value}_enabled" for feature in Feature} <= fields | {
        "advanced_filters_enabled"
    }
    assert len(list(Feature)) == len(fields)
