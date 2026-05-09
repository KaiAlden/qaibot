from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value == "" else float(value)


def _normalize_qdrant_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return "http://localhost:6333"

    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/")
    if path == "/dashboard":
        path = ""
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _normalize_openai_base_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return ""

    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float

    embedding_provider: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int

    qdrant_url: str
    qdrant_api_key: str
    qdrant_constitution_collection: str
    qdrant_advice_collection: str
    qdrant_distance: str

    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_charset: str
    session_ttl_days: int
    session_history_turns: int

    default_area: str
    default_top_k: int


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")

    return Settings(
        llm_provider=_env("LLM_PROVIDER", "openai"),
        llm_api_key=_env("LLM_API_KEY"),
        llm_base_url=_normalize_openai_base_url(_env("LLM_BASE_URL")),
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        embedding_provider=_env("EMBEDDING_PROVIDER", "openai"),
        embedding_api_key=_env("EMBEDDING_API_KEY") or _env("LLM_API_KEY"),
        embedding_base_url=_normalize_openai_base_url(_env("EMBEDDING_BASE_URL") or _env("LLM_BASE_URL")),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1536),
        qdrant_url=_normalize_qdrant_url(_env("QDRANT_URL", "http://localhost:6333")),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        qdrant_constitution_collection=_env(
            "QDRANT_CONSTITUTION_COLLECTION",
            "tcm_constitution_knowledge",
        ),
        qdrant_advice_collection=_env(
            "QDRANT_ADVICE_COLLECTION",
            "tcm_advice_knowledge",
        ),
        qdrant_distance=_env("QDRANT_DISTANCE", "Cosine"),
        mysql_host=_env("MYSQL_HOST", "127.0.0.1"),
        mysql_port=_env_int("MYSQL_PORT", 3306),
        mysql_user=_env("MYSQL_USER", "root"),
        mysql_password=_env("MYSQL_PASSWORD"),
        mysql_database=_env("MYSQL_DATABASE", "qaibot"),
        mysql_charset=_env("MYSQL_CHARSET", "utf8mb4"),
        session_ttl_days=_env_int("SESSION_TTL_DAYS", 30),
        session_history_turns=_env_int("SESSION_HISTORY_TURNS", 12),
        default_area=_env("DEFAULT_AREA", "华北"),
        default_top_k=_env_int("DEFAULT_TOP_K", 5),
    )
