from enum import StrEnum
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Feature(StrEnum):
    """The capabilities that can be switched off, and the one place they are named.

    Shared by the flags, the port and the error a disabled capability raises, so a feature
    cannot be spelled one way where it is configured and another where it is enforced.
    """

    VOICE = "voice"
    CONFIDENCE_SCORING = "confidence_scoring"
    HUMAN_REVIEW = "human_review"
    ADVANCED_FILTERS = "advanced_filters"


class FeatureFlags(BaseSettings):
    """Which capabilities this deployment runs.

    **Every flag defaults to on**, and that is deliberate: all four capabilities are shipped
    and tested, so these are kill switches rather than release gates. Defaulting them off
    would mean adding a flag file silently disabled four working features -- a behaviour
    change dressed up as configuration. An operator turns something off on purpose, in
    `.env`, and the sample there shows how.

    Read from the environment with a FEATURE_ prefix, so `FEATURE_VOICE_ENABLED=false`
    switches off speech without touching anything else.
    """

    # The mirror of the note on Settings: this class ignores every .env key that is not
    # a feature flag, so the two can share one file without either rejecting the other.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FEATURE_",
        extra="ignore",
    )

    voice_enabled: bool = True
    confidence_scoring_enabled: bool = True
    human_review_enabled: bool = True
    advanced_filters_enabled: bool = True

    @model_validator(mode="after")
    def _reject_a_combination_that_cannot_work(self) -> "FeatureFlags":
        """Human review needs a confidence score to route on.

        Routing decides who gets reviewed by reading the confidence band, so review with
        scoring switched off would queue nothing while looking configured to queue
        everything. Failing at startup is the only place that mistake is cheap to notice --
        the alternative is a review queue that stays quietly empty for a month.
        """
        if self.human_review_enabled and not self.confidence_scoring_enabled:
            raise ValueError(
                "FEATURE_HUMAN_REVIEW_ENABLED requires FEATURE_CONFIDENCE_SCORING_ENABLED: "
                "review routing decides what to queue from the confidence band, so with "
                "scoring disabled it would never queue anything"
            )
        return self


@lru_cache
def get_feature_flags() -> FeatureFlags:
    return FeatureFlags()
