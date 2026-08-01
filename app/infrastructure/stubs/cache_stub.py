from app.domain.models import GroundedAnswer


class CacheStub:
    """No-op cache: never returns a hit, discards writes. Stands in for the Valkey session cache."""

    async def get(self, key: str) -> GroundedAnswer | None:
        return None

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None:
        return None
