from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# 25 MiB. Roughly 25 minutes of 16 kHz mono WAV, or a couple of hours of compressed
# audio -- far more than a spoken logbook question, while still bounding what a single
# request can pull into memory.
_DEFAULT_MAX_AUDIO_BYTES = 25 * 1024 * 1024

# The upload formats faster-whisper handles once it replaces the stub adapter.
_DEFAULT_AUDIO_CONTENT_TYPES = [
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
]


def _split_csv(value: Any) -> Any:
    """Turns `a, B ,c` into `["a", "b", "c"]`, leaving non-strings alone.

    Needed because pydantic-settings would otherwise try to JSON-decode any list-typed
    field before validation runs, which a plain comma-separated env var is not.
    """
    if isinstance(value, str):
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    service_jwt_secret: str
    service_jwt_algorithm: str = "HS256"
    # Both languages are genuinely supported, so a missing env var should not silently
    # disable a shipped capability.
    supported_languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["en", "ar"]
    )

    stt_max_audio_bytes: int = _DEFAULT_MAX_AUDIO_BYTES
    stt_allowed_content_types: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(_DEFAULT_AUDIO_CONTENT_TYPES)
    )

    # Which speech adapter serves /stt/transcribe. Defaults to the stub so a plain
    # checkout runs without the multi-gigabyte model stack; set to "faster_whisper" on a
    # machine that has it (`uv sync --extra stt`).
    stt_backend: str = "stub"
    # A model size name that faster-whisper resolves against HuggingFace, or a path to a
    # locally staged model directory.
    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "default"
    stt_timeout_seconds: float = 120.0
    # Air-gapped deployment (see CLAUDE.md): with local_files_only the model must already
    # be present under download_root, and faster-whisper never reaches for the network.
    stt_local_files_only: bool = False
    stt_download_root: str | None = None

    @field_validator("supported_languages", "stt_allowed_content_types", mode="before")
    @classmethod
    def _parse_csv_list(cls, value: Any) -> Any:
        return _split_csv(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()
