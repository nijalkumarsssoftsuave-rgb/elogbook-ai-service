from typing import Protocol


class FeatureFlagPort(Protocol):
    """Which capabilities are switched on for this deployment.

    Four named questions rather than one `is_enabled(name)`, so a caller cannot ask about a
    feature that does not exist and get a plausible answer back. A misspelled string would
    read as "off", which is exactly the kind of silent disabling this port exists to make
    deliberate.

    The application depends on this and never on configuration. Today the answers come from
    `.env`; tomorrow they can come from Kubernetes, App Configuration, Parameter Store or
    LaunchDarkly, and nothing above this line changes -- which is the point of putting a
    port in front of four booleans that could otherwise have been read directly.
    """

    def is_voice_enabled(self) -> bool: ...

    def is_confidence_scoring_enabled(self) -> bool: ...

    def is_human_review_enabled(self) -> bool: ...

    def is_advanced_filter_enabled(self) -> bool: ...
