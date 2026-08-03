from app.core.feature_flags import FeatureFlags


class FeatureFlagProvider:
    """Answers the feature-flag port from configuration.

    Thin on purpose. Everything interesting about a flag -- its default, and the check that
    review is not enabled without scoring -- happens when FeatureFlags is built and
    validated, so by the time a value reaches here it has already been vetted.

    This is the file that changes when flags stop coming from `.env`. A Kubernetes
    ConfigMap, Azure App Configuration, AWS Parameter Store or LaunchDarkly each replaces
    this class and nothing above it: the application depends on the port, and the port asks
    four questions that any of those can answer.

    The flags are read once and held, which matches where they come from today -- a process
    reading its own environment. A provider backed by a service that can change a flag
    mid-flight would re-read per call instead, and that is a decision for the ticket that
    introduces one, not something to guess at now.
    """

    def __init__(self, flags: FeatureFlags) -> None:
        self._flags = flags

    def is_voice_enabled(self) -> bool:
        return self._flags.voice_enabled

    def is_confidence_scoring_enabled(self) -> bool:
        return self._flags.confidence_scoring_enabled

    def is_human_review_enabled(self) -> bool:
        return self._flags.human_review_enabled

    def is_advanced_filter_enabled(self) -> bool:
        return self._flags.advanced_filters_enabled
