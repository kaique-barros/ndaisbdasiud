# Acesso aos bancos relacionais (SQLite). Uma conexão por arquivo, protegida por lock,
# com versões async (executadas em thread) para não bloquear o loop do orchestrator.
from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def script(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Executa uma escrita e devolve o lastrowid."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid or 0

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    async def aquery(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return await asyncio.to_thread(self.query, sql, params)

    async def aexecute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return await asyncio.to_thread(self.execute, sql, params)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
