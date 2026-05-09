from __future__ import annotations

from app.config import Settings
from app.rag.qdrant_store import QdrantStore


DIET_PRINCIPLE = "季节饮食原则"


class KnowledgeRetriever:
    def __init__(self, settings: Settings, store: QdrantStore):
        self.settings = settings
        self.store = store

    def retrieve(
        self,
        query: str,
        constitution: str,
        area: str | None,
        season: str | None,
        advice_type: str | None,
    ) -> list[dict]:
        results: list[dict] = []

        if advice_type is None or advice_type == DIET_PRINCIPLE:
            results.extend(
                self._search_with_fallback(
                    query=query,
                    base_filters={"type": "diet_principle", "constitution": constitution},
                    area=area,
                    season=season,
                    limit=2,
                )
            )

        if advice_type and advice_type != DIET_PRINCIPLE:
            filters = {"type": "suggestion", "constitution": constitution, "suggestion_name": advice_type}
            results.extend(self._search_with_fallback(query, filters, area, season, limit=3))
        elif advice_type is None:
            results.extend(
                self._search_with_fallback(
                    query=query,
                    base_filters={"type": "suggestion", "constitution": constitution},
                    area=area,
                    season=season,
                    limit=4,
                )
            )

        seen: set[str] = set()
        deduped: list[dict] = []
        for item in sorted(results, key=lambda x: x["score"], reverse=True):
            chunk_id = item["payload"].get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(item)
        return deduped[: self.settings.default_top_k]

    def _search_with_fallback(
        self,
        query: str,
        base_filters: dict[str, object],
        area: str | None,
        season: str | None,
        limit: int,
    ) -> list[dict]:
        attempts = [
            ("area_season_constitution", {**base_filters, "area": area, "season": season}),
            ("area_constitution", {**base_filters, "area": area}),
            ("constitution", base_filters),
        ]
        for fallback_level, filters in attempts:
            cleaned = {key: value for key, value in filters.items() if value is not None}
            found = self.store.search(
                self.settings.qdrant_advice_collection,
                query,
                filters=cleaned,
                limit=limit,
            )
            if found:
                for item in found:
                    item["fallback_level"] = fallback_level
                return found
        return []
