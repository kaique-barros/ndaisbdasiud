# Memória flash: consulta o messages.db, pega as k últimas mensagens e formata para o prompt.
# Também guarda, por mensagem, a variação emocional causada (usada pelo notetaker) e
# marcadores (`meta`) de até onde cada processo em background já leu.
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.data.database import Database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    emotion_delta TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: str | None
    role: str
    content: str
    created_at: str
    emotion_delta: dict[str, float] = field(default_factory=dict)


def _message(r) -> Message:
    return Message(r["id"], r["conversation_id"], r["role"], r["content"], r["created_at"],
                   json.loads(r["emotion_delta"]) if r["emotion_delta"] else {})


class FlashMemory:
    def __init__(self, db: Database, k: int = 20) -> None:
        self.db = db
        self.k = k
        self.db.script(_SCHEMA)
        columns = {r["name"] for r in self.db.query("PRAGMA table_info(messages)")}
        if "emotion_delta" not in columns:  # bancos criados antes desta coluna
            self.db.execute("ALTER TABLE messages ADD COLUMN emotion_delta TEXT")

    def add(self, role: str, content: str, conversation_id: str | None = None,
            emotion_delta: dict[str, float] | None = None) -> int:
        return self.db.execute(
            "INSERT INTO messages (conversation_id, role, content, emotion_delta) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(emotion_delta) if emotion_delta else None),
        )

    def last(self, k: int | None = None, conversation_id: str | None = None) -> list[Message]:
        """As k últimas mensagens, em ordem cronológica."""
        sql, params = "SELECT * FROM messages", []
        if conversation_id is not None:
            sql += " WHERE conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(k if k is not None else self.k)
        return [_message(r) for r in reversed(self.db.query(sql, params))]

    def since(self, after_id: int, limit: int | None = None) -> list[Message]:
        """Mensagens com id > after_id, em ordem cronológica."""
        sql, params = "SELECT * FROM messages WHERE id > ? ORDER BY id", [after_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_message(r) for r in self.db.query(sql, params)]

    def count_since(self, after_id: int) -> int:
        return int(self.db.query_one("SELECT COUNT(*) AS n FROM messages WHERE id > ?",
                                     (after_id,))["n"])

    def delete(self, message_id: int) -> bool:
        exists = self.db.query_one("SELECT 1 FROM messages WHERE id = ?", (message_id,))
        self.db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        return exists is not None

    def get_meta(self, key: str, default: int = 0) -> int:
        row = self.db.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return int(row["value"]) if row else default

    def set_meta(self, key: str, value: int) -> None:
        self.db.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))

    def format(self, k: int | None = None, max_chars: int | None = None,
               conversation_id: str | None = None) -> str:
        """Texto para o prompt. Com max_chars, descarta as mais antigas até caber
        (o core/actor usa isso para ajustar a flash ao tamanho dos outros inputs)."""
        lines = [f"{m.role}: {m.content}" for m in self.last(k, conversation_id)]
        if max_chars is not None:
            while lines and sum(len(l) + 1 for l in lines) > max_chars:
                lines.pop(0)
        return "\n".join(lines)
