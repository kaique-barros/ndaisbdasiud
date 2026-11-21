# Abstração do graph-RAG sobre SQLite: nós (entidades) e arestas (relações) com
# data de criação e de último acesso. A busca acha nós pelo nome (FTS5) e expande
# a vizinhança até `depth` saltos, priorizando relações mais próximas e mais reforçadas.
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.data.database import Database
from src.data.rag import fts_query, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    type TEXT,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name, content='nodes', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, name) VALUES (new.id, new.name);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, name) VALUES ('delete', old.id, old.name);
END;
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    UNIQUE (src, relation, dst)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""

_FACT_SQL = """
SELECT e.id, e.src, e.dst, s.name AS subject, e.relation, d.name AS object, e.weight,
       e.created_at, e.last_accessed_at
FROM edges e JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst
"""


@dataclass(frozen=True)
class Fact:
    id: int
    subject: str
    relation: str
    object: str
    weight: float = 1.0
    created_at: str = ""
    last_accessed_at: str | None = None


def _fact(row) -> Fact:
    return Fact(row["id"], row["subject"], row["relation"], row["object"], row["weight"],
                row["created_at"], row["last_accessed_at"])


class GraphRag:
    MAX_FRONTIER = 200

    def __init__(self, db: Database) -> None:
        self.db = db
        db.script(_SCHEMA)

    def add_fact(self, subject: str, relation: str, obj: str, subject_type: str | None = None,
                 object_type: str | None = None) -> Fact:
        """Insere a relação; se já existir, reforça (peso +1) e atualiza o último acesso."""
        relation = " ".join(relation.lower().split())
        src, dst = self._node(subject, subject_type), self._node(obj, object_type)
        if not relation:
            raise ValueError("relação vazia")
        now = now_iso()
        self.db.execute(
            "INSERT INTO edges (src, dst, relation, created_at, last_accessed_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(src, relation, dst) DO UPDATE SET "
            "weight = weight + 1, last_accessed_at = excluded.last_accessed_at",
            (src, dst, relation, now, now))
        row = self.db.query_one(_FACT_SQL + " WHERE e.src = ? AND e.relation = ? AND e.dst = ?",
                                (src, relation, dst))
        return _fact(row)

    def delete_fact(self, fact_id: int) -> bool:
        exists = self.db.query_one("SELECT 1 FROM edges WHERE id = ?", (fact_id,))
        self.db.execute("DELETE FROM edges WHERE id = ?", (fact_id,))
        return exists is not None

    def list_facts(self, limit: int = 50, offset: int = 0) -> list[Fact]:
        rows = self.db.query(_FACT_SQL + " ORDER BY e.id DESC LIMIT ? OFFSET ?", (limit, offset))
        return [_fact(r) for r in rows]

    def count(self) -> int:
        return int(self.db.query_one("SELECT COUNT(*) AS n FROM edges")["n"])

    def search(self, query: str, top_k: int = 8, depth: int = 2, touch: bool = True) -> list[Fact]:
        match = fts_query(query)
        if not match:
            return []
        seeds = [r["id"] for r in self.db.query(
            "SELECT rowid AS id FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY bm25(nodes_fts) "
            "LIMIT 5", (match,))]
        if not seeds:
            return []

        visited, frontier = set(seeds), set(seeds)
        found: dict[int, tuple[int, float, Fact]] = {}  # id -> (nível, peso, fato)
        for level in range(depth):
            marks = ",".join("?" * len(frontier))
            rows = self.db.query(
                _FACT_SQL + f" WHERE e.src IN ({marks}) OR e.dst IN ({marks}) "
                "ORDER BY e.weight DESC", [*frontier, *frontier])
            nxt: set[int] = set()
            for row in rows:
                if row["id"] in found:
                    continue
                found[row["id"]] = (level, row["weight"], _fact(row))
                for end in (row["src"], row["dst"]):
                    if end not in visited:
                        visited.add(end)
                        nxt.add(end)
            frontier = set(list(nxt)[: self.MAX_FRONTIER])
            if not frontier:
                break
        ranked = sorted(found.values(), key=lambda t: (t[0], -t[1], -t[2].id))[:top_k]
        facts = [f for _, _, f in ranked]
        if touch and facts:
            marks = ",".join("?" * len(facts))
            self.db.execute(f"UPDATE edges SET last_accessed_at = ? WHERE id IN ({marks})",
                            [now_iso(), *[f.id for f in facts]])
        return facts

    async def asearch(self, query: str, top_k: int = 8) -> list[Fact]:
        return await asyncio.to_thread(self.search, query, top_k)

    def _node(self, name: str, node_type: str | None) -> int:
        name = " ".join(name.split())
        if not name:
            raise ValueError("nome de entidade vazio")
        now = now_iso()
        self.db.execute("INSERT OR IGNORE INTO nodes (name, type, created_at) VALUES (?, ?, ?)",
                        (name, node_type, now))
        if node_type:
            self.db.execute("UPDATE nodes SET type = ? WHERE name = ? AND type IS NULL",
                            (node_type, name))
        return self.db.query_one("SELECT id FROM nodes WHERE name = ?", (name,))["id"]
