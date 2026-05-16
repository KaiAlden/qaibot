from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any

import pymysql
from pymysql.connections import Connection

from app.config import Settings
from app.schemas import SessionState


class SessionStore:
    """MySQL-backed session state isolated by user_id + conversation_id."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ttl_days = settings.session_ttl_days
        self.history_turns = settings.session_history_turns
        self._init_db()

    def get(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT state_json FROM qaibot_sessions
                    WHERE user_id = %s AND conversation_id = %s
                    """,
                    (user_id, conversation_id),
                )
                row = cur.fetchone()
        if not row:
            return self._empty_state()
        state = {**self._empty_state(), **json.loads(row[0])}
        state.pop("constitution", None)
        return state

    def save(self, user_id: str, conversation_id: str, state: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        state.pop("constitution", None)
        state["history"] = self._trim_history(state.get("history", []))
        state_json = json.dumps(state, ensure_ascii=False)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qaibot_sessions(user_id, conversation_id, state_json, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      state_json = VALUES(state_json),
                      updated_at = VALUES(updated_at)
                    """,
                    (user_id, conversation_id, state_json, now),
                )

    def append_history(self, state: dict[str, Any], role: str, content: str) -> None:
        history = state.setdefault("history", [])
        history.append({"role": role, "content": content})
        state["history"] = self._trim_history(history)

    def to_public_state(self, state: dict[str, Any]) -> SessionState:
        return SessionState(
            user_constitution=state.get("user_constitution"),
            secondary_constitution=state.get("secondary_constitution"),
            target_constitutions=state.get("target_constitutions") or [],
            last_topic_constitutions=state.get("last_topic_constitutions") or [],
            last_topic_turn_index=state.get("last_topic_turn_index"),
            turn_index=state.get("turn_index") or 0,
            area=state.get("area"),
            season=state.get("season"),
            last_intent=state.get("last_intent"),
            last_advice_types=state.get("last_advice_types") or [],
            pending_clarification=state.get("pending_clarification"),
        )

    def cleanup_expired(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.ttl_days)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM qaibot_sessions WHERE updated_at < %s",
                    (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
                )
                return cur.rowcount

    def _init_db(self) -> None:
        database = self._safe_database_name()
        with self._connect(include_database=False) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{database}` "
                    f"CHARACTER SET {self.settings.mysql_charset} COLLATE utf8mb4_unicode_ci"
                )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS qaibot_sessions (
                        user_id VARCHAR(128) NOT NULL,
                        conversation_id VARCHAR(128) NOT NULL,
                        state_json LONGTEXT NOT NULL,
                        updated_at DATETIME NOT NULL,
                        PRIMARY KEY(user_id, conversation_id),
                        INDEX idx_updated_at(updated_at)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                    """
                )

    def _connect(self, include_database: bool = True) -> Connection:
        kwargs = {
            "host": self.settings.mysql_host,
            "port": self.settings.mysql_port,
            "user": self.settings.mysql_user,
            "password": self.settings.mysql_password,
            "charset": self.settings.mysql_charset,
            "autocommit": True,
        }
        if include_database:
            kwargs["database"] = self.settings.mysql_database
        return pymysql.connect(
            **kwargs,
        )

    def _safe_database_name(self) -> str:
        database = self.settings.mysql_database
        if not re.fullmatch(r"[A-Za-z0-9_]+", database):
            raise ValueError("MYSQL_DATABASE 只能包含字母、数字和下划线")
        return database

    def _trim_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        return history[-self.history_turns * 2 :]

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "user_constitution": None,
            "secondary_constitution": None,
            "target_constitutions": [],
            "last_topic_constitutions": [],
            "last_topic_turn_index": None,
            "turn_index": 0,
            "non_tcm_turns_since_topic": 0,
            "area": None,
            "season": None,
            "last_intent": None,
            "last_advice_types": [],
            "pending_clarification": None,
            "history": [],
        }
