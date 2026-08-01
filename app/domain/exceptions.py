class UnsupportedLanguageError(Exception):
    """Raised when the detected query language is not in the configured allow-list."""

    def __init__(self, language_code: str, supported_languages: list[str]) -> None:
        self.language_code = language_code
        self.supported_languages = supported_languages
        super().__init__(
            f"Detected language '{language_code}' is not supported. "
            f"Supported languages: {', '.join(supported_languages)}."
        )
