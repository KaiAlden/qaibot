from __future__ import annotations

from collections import defaultdict
from hashlib import md5
from pathlib import Path
import re

import pandas as pd

from app.domain import (
    ADVICE_TYPES,
    CONSTITUTIONS,
    clean_text,
    normalize_area,
    normalize_constitution,
    normalize_term,
)
from app.schemas import KnowledgeChunk


def stable_chunk_id(*parts: object) -> str:
    raw = "__".join(str(p) for p in parts if p not in (None, ""))
    digest = md5(raw.encode("utf-8")).hexdigest()[:10]
    prefix = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", raw)[:80].strip("_")
    return f"{prefix}__{digest}"


def build_constitution_chunks(txt_path: Path) -> list[KnowledgeChunk]:
    text = txt_path.read_text(encoding="utf-8")
    paragraphs = [clean_text(p) for p in re.split(r"\n\s*\n", text) if clean_text(p)]
    chunks: list[KnowledgeChunk] = []

    for paragraph in paragraphs:
        constitution = next((name for name in CONSTITUTIONS if paragraph.startswith(name)), None)
        constitution = constitution or normalize_constitution(paragraph[:8])
        content = f"【体质识别】{constitution}\n\n{paragraph}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=stable_chunk_id("constitution", constitution),
                type="constitution_identify",
                content=content,
                constitution=constitution,
            )
        )

    return chunks


def build_diet_chunks(df: pd.DataFrame) -> list[KnowledgeChunk]:
    required = {"area_name", "solar_terms_name", "constitution_name", "suggestion_name", "attribute_1", "attribute_2"}
    _ensure_columns(df, required, "季节饮食原则")

    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        area = normalize_area(row["area_name"])
        _, season = normalize_term(row["solar_terms_name"])
        constitution = normalize_constitution(row["constitution_name"])
        if not (area and season and constitution):
            continue

        title = clean_text(row["attribute_1"])
        detail = clean_text(row["attribute_2"])
        if title or detail:
            grouped[(area, season, constitution)].append(f"{title}：{detail}" if detail else title)

    chunks: list[KnowledgeChunk] = []
    for (area, season, constitution), items in grouped.items():
        content = (
            f"【季节饮食原则】{area} · {season} · {constitution}\n\n"
            + "\n".join(f"原则{i + 1}：{item}" for i, item in enumerate(items))
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=stable_chunk_id("diet", area, season, constitution),
                type="diet_principle",
                content=content,
                area=area,
                season=season,
                constitution=constitution,
                suggestion_name="季节饮食原则",
            )
        )
    return chunks


def build_suggestion_chunks(df: pd.DataFrame) -> list[KnowledgeChunk]:
    required = {"area_name", "solar_terms_name", "constitution_name", "suggestion_name", "attribute_1"}
    _ensure_columns(df, required, "suggestion")

    grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        area = normalize_area(row["area_name"])
        _, season = normalize_term(row["solar_terms_name"])
        constitution = normalize_constitution(row["constitution_name"])
        suggestion_name = clean_text(row["suggestion_name"])
        if suggestion_name not in ADVICE_TYPES:
            suggestion_name = suggestion_name or "调理建议"
        body = clean_text(row["attribute_1"])
        if not (area and season and constitution and body):
            continue
        grouped[(area, season, constitution, suggestion_name)].append(body)

    chunks: list[KnowledgeChunk] = []
    for (area, season, constitution, suggestion_name), items in grouped.items():
        content = (
            f"【调理建议·{suggestion_name}】{area} · {season} · {constitution}\n\n"
            + "\n\n".join(items)
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=stable_chunk_id("suggestion", area, season, constitution, suggestion_name),
                type="suggestion",
                content=content,
                area=area,
                season=season,
                constitution=constitution,
                suggestion_name=suggestion_name,
            )
        )
    return chunks


def build_all_chunks(data_dir: Path) -> list[KnowledgeChunk]:
    txt_path = next(data_dir.glob("*.txt"))
    xlsx_path = next(data_dir.glob("*.xlsx"))
    sheets = pd.read_excel(xlsx_path, sheet_name=None)

    chunks = build_constitution_chunks(txt_path)
    chunks.extend(build_diet_chunks(sheets["季节饮食原则"]))
    chunks.extend(build_suggestion_chunks(sheets["suggestion"]))
    return chunks


def _ensure_columns(df: pd.DataFrame, required: set[str], sheet_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{sheet_name} 缺少必要字段: {', '.join(sorted(missing))}")
