# Abstração do RAG sobre SQLite.
#   - busca lexical: FTS5 com bm25 (sem acentos), sempre disponível, zero dependências;
#   - busca vetorial: opcional, quando há um `embedder` (ex.: /embeddings do Ollama);
#     os dois rankings são combinados por Reciprocal Rank Fusion.
# Cada documento guarda texto, metadados (JSON), data de criação e do último acesso.
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from src.data.database import Database
from src.utils.logger import get_logger

Embedder = Callable[[Sequence[str]], "list[list[float]]"]

_TOKEN = re.compile(r"\w{2,}", re.UNICODE)
_NAME = re.compile(r"[a-z_]+")
RRF_K = 60


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fts_query(text: str, max_tokens: int = 32) -> str:
    """Consulta FTS5 segura: tokens entre aspas unidos por OR (o bm25 pesa os mais raros)."""
    seen: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok not in seen:
            seen.append(tok)
    return " OR ".join(f'"{t}"' for t in seen[:max_tokens])


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass(frozen=True)
class RagHit:
    id: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    created_at: str = ""
    last_accessed_at: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS {t} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{{}}',
    embedding TEXT,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS {t}_fts USING fts5(
    text, content='{t}', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS {t}_ai AFTER INSERT ON {t} BEGIN
    INSERT INTO {t}_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS {t}_ad AFTER DELETE ON {t} BEGIN
    INSERT INTO {t}_fts({t}_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


class RagIndex:
    def __init__(self, db: Database, name: str, embedder: Embedder | None = None,
                 min_similarity: float = 0.3) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f"nome de índice inválido: {name!r}")
        self.db, self.t = db, name
        self.embedder = embedder
        self.min_similarity = min_similarity
        self._embed_failed = False
        self.log = get_logger("rag", "data")
        db.script(_SCHEMA.format(t=name))

    # --- escrita ---

    def add(self, text: str, meta: dict[str, Any] | None = None) -> int:
        text = text.strip()
        if not text:
            raise ValueError("texto vazio")
        emb = self._embed_one(text)
        return self.db.execute(
            f"INSERT INTO {self.t} (text, meta, embedding, created_at) VALUES (?, ?, ?, ?)",
            (text, json.dumps(meta or {}, ensure_ascii=False),
             json.dumps(emb) if emb else None, now_iso()))

    def delete(self, doc_id: int) -> bool:
        exists = self.db.query_one(f"SELECT 1 FROM {self.t} WHERE id = ?", (doc_id,))
        self.db.execute(f"DELETE FROM {self.t} WHERE id = ?", (doc_id,))
        return exists is not None

    # --- leitura ---

    def get(self, doc_id: int) -> RagHit | None:
        row = self.db.query_one(f"SELECT * FROM {self.t} WHERE id = ?", (doc_id,))
        return self._hit(row) if row else None

    def list(self, limit: int = 50, offset: int = 0) -> list[RagHit]:
        rows = self.db.query(f"SELECT * FROM {self.t} ORDER BY id DESC LIMIT ? OFFSET ?",
                             (limit, offset))
        return [self._hit(r) for r in rows]

    def count(self) -> int:
        return int(self.db.query_one(f"SELECT COUNT(*) AS n FROM {self.t}")["n"])

    def search(self, query: str, top_k: int = 5, touch: bool = True) -> list[RagHit]:
        pool = max(top_k * 4, 20)
        lexical = self._lexical(query, pool)
        scores: dict[int, float] = {}
        vector = self._vector(query, pool)
        if vector is None:
            scores = {i: s for i, s in lexical}
        else:
            for ranking in ([i for i, _ in lexical], vector):
                for rank, doc_id in enumerate(ranking):
                    scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in self.db.query(
            f"SELECT * FROM {self.t} WHERE id IN ({marks})", ids)}
        if touch:
            self.db.execute(f"UPDATE {self.t} SET last_accessed_at = ? WHERE id IN ({marks})",
                            [now_iso(), *ids])
        return [self._hit(rows[i], scores[i]) for i in ids if i in rows]

    async def asearch(self, query: str, top_k: int = 5) -> list[RagHit]:
        return await asyncio.to_thread(self.search, query, top_k)

    # --- internos ---

    def _lexical(self, query: str, limit: int) -> list[tuple[int, float]]:
        match = fts_query(query)
        if not match:
            return []
        rows = self.db.query(
            f"SELECT rowid AS id, bm25({self.t}_fts) AS s FROM {self.t}_fts "
            f"WHERE {self.t}_fts MATCH ? ORDER BY s LIMIT ?", (match, limit))
        return [(r["id"], -float(r["s"])) for r in rows]  # bm25 menor = melhor

    def _vector(self, query: str, limit: int) -> list[int] | None:
        """Ids por similaridade de cosseno; None se não há embedder (ou ele falhou)."""
        if not self.embedder or self._embed_failed or not query.strip():
            return None
        try:
            qv = self.embedder([query])[0]
        except Exception as exc:
            self._disable_embeddings(exc)
            return None
        sims = []
        for row in self.db.query(f"SELECT id, embedding FROM {self.t} WHERE embedding IS NOT NULL"):
            sim = cosine(qv, json.loads(row["embedding"]))
            if sim >= self.min_similarity:
                sims.append((sim, row["id"]))
        sims.sort(reverse=True)
        return [i for _, i in sims[:limit]]

    def _embed_one(self, text: str) -> list[float] | None:
        if not self.embedder or self._embed_failed:
            return None
        try:
            return self.embedder([text])[0]
        except Exception as exc:
            self._disable_embeddings(exc)
            return None

    def _disable_embeddings(self, exc: Exception) -> None:
        self._embed_failed = True
        self.log.warning("embeddings desativados nesta sessão (%s); usando só busca lexical", exc)

    def _hit(self, row, score: float = 0.0) -> RagHit:
        return RagHit(row["id"], row["text"], json.loads(row["meta"]), score,
                      row["created_at"], row["last_accessed_at"])
