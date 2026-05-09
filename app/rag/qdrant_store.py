from __future__ import annotations

from collections.abc import Iterable
from uuid import NAMESPACE_URL, uuid5

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import Settings
from app.schemas import KnowledgeChunk


class QdrantStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.embedding_client = OpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or None,
        )

    def ensure_collection(self, collection_name: str) -> None:
        collections = self.client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dim,
                distance=getattr(models.Distance, self.settings.qdrant_distance.upper()),
            ),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Iterable[KnowledgeChunk],
        batch_size: int = 64,
    ) -> int:
        self.ensure_collection(collection_name)
        buffer: list[KnowledgeChunk] = []
        total = 0
        for chunk in chunks:
            buffer.append(chunk)
            if len(buffer) >= batch_size:
                total += self._upsert_batch(collection_name, buffer)
                buffer = []
        if buffer:
            total += self._upsert_batch(collection_name, buffer)
        return total

    def search(
        self,
        collection_name: str,
        query: str,
        filters: dict[str, object] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        vector = self.embed([query])[0]
        q_filter = self._build_filter(filters or {})
        try:
            points = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
        except AttributeError:
            result = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
            points = result.points

        return [{"score": float(p.score), "payload": p.payload or {}} for p in points]

    def _upsert_batch(self, collection_name: str, chunks: list[KnowledgeChunk]) -> int:
        vectors = self.embed([c.content for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        return len(points)

    @staticmethod
    def _build_filter(filters: dict[str, object]) -> models.Filter | None:
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=conditions) if conditions else None
