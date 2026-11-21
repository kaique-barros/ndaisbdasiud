# Registro de personas (.data/global.db). Cada persona tem um uuid e metadados
# (status, última conversa...) para o menu listar/buscar sem abrir os arquivos de cada uma.
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.data.database import Database

ACTIVE = "active"
ARCHIVED = "archived"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    last_conversation_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_personas_status ON personas(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PersonaRecord:
    id: str
    name: str
    status: str
    created_at: str
    last_conversation_at: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_record(row) -> PersonaRecord:
    return PersonaRecord(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        last_conversation_at=row["last_conversation_at"],
        metadata=json.loads(row["metadata"]),
    )


class PersonaRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.db.script(_SCHEMA)

    def create(self, name: str, metadata: dict[str, Any] | None = None,
               persona_id: str | None = None) -> PersonaRecord:
        pid = persona_id or str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO personas (id, name, status, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (pid, name, ACTIVE, _now(), json.dumps(metadata or {}, ensure_ascii=False)),
        )
        return self.get(pid)  # type: ignore[return-value]

    def get(self, persona_id: str) -> PersonaRecord | None:
        row = self.db.query_one("SELECT * FROM personas WHERE id = ?", (persona_id,))
        return _to_record(row) if row else None

    def list(self, status: str | None = None, search: str | None = None) -> list[PersonaRecord]:
        """Mais recentemente conversadas primeiro; as nunca usadas ficam no fim."""
        sql, params = "SELECT * FROM personas WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if search:
            sql += " AND name LIKE ?"
            params.append(f"%{search}%")
        sql += " ORDER BY last_conversation_at IS NULL, last_conversation_at DESC, created_at DESC"
        return [_to_record(r) for r in self.db.query(sql, params)]

    def touch_conversation(self, persona_id: str) -> None:
        self.db.execute("UPDATE personas SET last_conversation_at = ? WHERE id = ?",
                        (_now(), persona_id))

    def set_status(self, persona_id: str, status: str) -> None:
        if status not in (ACTIVE, ARCHIVED):
            raise ValueError(f"status inválido: {status}")
        self.db.execute("UPDATE personas SET status = ? WHERE id = ?", (status, persona_id))

    def rename(self, persona_id: str, name: str) -> None:
        self.db.execute("UPDATE personas SET name = ? WHERE id = ?", (name, persona_id))

    def update_metadata(self, persona_id: str, **values: Any) -> None:
        record = self.get(persona_id)
        if record is None:
            raise KeyError(persona_id)
        merged = {**record.metadata, **values}
        self.db.execute("UPDATE personas SET metadata = ? WHERE id = ?",
                        (json.dumps(merged, ensure_ascii=False), persona_id))
