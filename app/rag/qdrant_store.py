# 封装了与 Qdrant 向量数据库的所有交互，提供三个核心能力：建库、存数据、搜数据。

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

    # 检查 Qdrant 中是否有指定名称的集合，没有就自动创建
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

    # 调用 OpenAI Embedding API，将文本转为向量
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # 将知识片段向量化后存入 Qdrant
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

    # 用语义搜索从 Qdrant 中找到最相关的知识片段。
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
        """
        批量内部实现:
        - 批量调用 `embed()` 将所有 chunk 转为向量
        - 为每个 chunk 生成稳定的 UUID__（用 `uuid5(NAMESPACE_URL, chunk.chunk_id)`）
        - 构造 `PointStruct` 调用 Qdrant 的 `upsert` 存入

        """
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

    # 将 Python 字典转换为 Qdrant 的 Filter 对象。
    @staticmethod
    def _build_filter(filters: dict[str, object]) -> models.Filter | None:
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=conditions) if conditions else None



"""
【导入时】
build_index.py
    → QdrantStore.upsert_chunks("constitution", chunks)
        → ensure_collection()     # 建集合
        → embed()                 # 转向量
        → client.upsert()         # 存 Qdrant

【运行时】
ChatService.chat()
    → KnowledgeRetriever.retrieve()
        → QdrantStore.search("advice", query, filters)
            → embed()             # 把用户问题转向量
            → _build_filter()     # 构建过滤条件
            → client.search()     # 在 Qdrant 中搜索

"""