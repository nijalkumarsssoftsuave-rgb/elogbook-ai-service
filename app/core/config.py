from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    service_jwt_secret: str
    service_jwt_algorithm: str = "HS256"


@lru_cache
def get_settings() -> Settings:
    return Settings()
