from app.domain.models import Embedding


class EmbeddingStub:
    """Returns a deterministic dummy vector derived from the input text. Stands in for
    BGE-M3 embedding inference.

    The 8 dimensions are deliberately not BGE-M3's real 1024 — this vector carries no
    semantic meaning and exists only to give VectorStorePort a well-typed input.
    """

    _DIMENSION = 8

    async def embed(self, text: str) -> Embedding:
        seed = sum(text.encode("utf-8"))
        vector = [((seed + index) % 97) / 97 for index in range(self._DIMENSION)]
        return Embedding(vector=vector, model="stub-embedding")
