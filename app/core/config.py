from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    service_jwt_secret: str
    service_jwt_algorithm: str = "HS256"
    supported_languages: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["en"])

    @field_validator("supported_languages", mode="before")
    @classmethod
    def _parse_supported_languages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [lang.strip().lower() for lang in value.split(",") if lang.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
