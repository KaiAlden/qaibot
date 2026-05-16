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
        constitution: str | None,
        area: str | None,
        season: str | None,
        advice_type: str | None,
        advice_types: list[str] | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        suggestion_types = list(dict.fromkeys(advice_types or []))
        include_diet = DIET_PRINCIPLE in suggestion_types
        suggestion_types = [item for item in suggestion_types if item != DIET_PRINCIPLE]
        if not suggestion_types and advice_type and advice_type != DIET_PRINCIPLE:
            suggestion_types = [advice_type]
        if advice_type == DIET_PRINCIPLE:
            include_diet = True

        if include_diet or (not suggestion_types and (advice_type is None or advice_type == DIET_PRINCIPLE)):
            results.extend(
                self._search_with_fallback(
                    query=query,
                    base_filters={"type": "diet_principle", "constitution": constitution},
                    area=area,
                    season=season,
                    limit=self.settings.diet_principle_top_k,
                )
            )

        if suggestion_types:
            per_type_limit = (
                self.settings.suggestion_per_type_top_k
                if len(suggestion_types) > 1
                else self.settings.suggestion_top_k
            )
            for suggestion_type in suggestion_types:
                filters = {
                    "type": "suggestion",
                    "constitution": constitution,
                    "suggestion_name": suggestion_type,
                }
                results.extend(self._search_with_fallback(query, filters, area, season, limit=per_type_limit))
        elif advice_type is None:
            results.extend(
                self._search_with_fallback(
                    query=query,
                    base_filters={"type": "suggestion", "constitution": constitution},
                    area=area,
                    season=season,
                    limit=self.settings.general_suggestion_top_k,
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

    def retrieve_for_targets(
        self,
        query: str,
        target_constitutions: list[str],
        area: str | None,
        season: str | None,
        advice_type: str | None,
        advice_types: list[str] | None = None,
        include_comparison_context: bool = False,
    ) -> list[dict]:
        targets = list(dict.fromkeys([item for item in target_constitutions if item]))
        if not targets:
            return self.retrieve(query, None, area, season, advice_type, advice_types)
        if len(targets) == 1 and not include_comparison_context:
            found = self.retrieve(query, targets[0], area, season, advice_type, advice_types)
            for item in found:
                item["target_constitution"] = targets[0]
                item["payload"] = {**item.get("payload", {}), "target_constitution": targets[0]}
            return found

        results: list[dict] = []
        per_target_limit = max(1, self.settings.default_top_k // len(targets))
        for target in targets:
            target_results = self.retrieve(query, target, area, season, advice_type, advice_types)
            for item in target_results[:per_target_limit]:
                item = {**item, "target_constitution": target}
                item["payload"] = {**item.get("payload", {}), "target_constitution": target}
                results.append(item)

        if include_comparison_context:
            general_results = self.retrieve(query, None, area, season, advice_type, advice_types)
            for item in general_results[: max(1, self.settings.default_top_k - len(results))]:
                item = {**item, "target_constitution": None}
                item["payload"] = {**item.get("payload", {}), "target_constitution": None}
                results.append(item)

        seen: set[tuple[str | None, str | None]] = set()
        deduped: list[dict] = []
        for item in sorted(results, key=lambda x: x["score"], reverse=True):
            payload = item.get("payload", {})
            key = (payload.get("chunk_id"), payload.get("target_constitution"))
            if key not in seen:
                seen.add(key)
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
            try:
                found = self.store.search(
                    self.settings.qdrant_advice_collection,
                    query,
                    filters=cleaned,
                    limit=limit,
                )
            except Exception:  # noqa: BLE001
                return []
            if found:
                for item in found:
                    item["fallback_level"] = fallback_level
                return found
        return []
