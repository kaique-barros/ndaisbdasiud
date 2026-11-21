# Memória flash: consulta o messages.db, pega as k últimas mensagens e formata para o prompt.
from __future__ import annotations

from dataclasses import dataclass

from src.data.database import Database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: str | None
    role: str
    content: str
    created_at: str


class FlashMemory:
    def __init__(self, db: Database, k: int = 20) -> None:
        self.db = db
        self.k = k
        self.db.script(_SCHEMA)

    def add(self, role: str, content: str, conversation_id: str | None = None) -> int:
        return self.db.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )

    def last(self, k: int | None = None, conversation_id: str | None = None) -> list[Message]:
        """As k últimas mensagens, em ordem cronológica."""
        sql, params = "SELECT * FROM messages", []
        if conversation_id is not None:
            sql += " WHERE conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(k if k is not None else self.k)
        rows = self.db.query(sql, params)
        return [Message(r["id"], r["conversation_id"], r["role"], r["content"], r["created_at"])
                for r in reversed(rows)]

    def format(self, k: int | None = None, max_chars: int | None = None,
               conversation_id: str | None = None) -> str:
        """Texto para o prompt. Com max_chars, descarta as mais antigas até caber
        (o core/actor usa isso para ajustar a flash ao tamanho dos outros inputs)."""
        lines = [f"{m.role}: {m.content}" for m in self.last(k, conversation_id)]
        if max_chars is not None:
            while lines and sum(len(l) + 1 for l in lines) > max_chars:
                lines.pop(0)
        return "\n".join(lines)
