# Notas da persona: impressões pessoais sobre coisas, pessoas e eventos. Diferente de
# events/knowledge (factuais), aqui a persona diz o que PENSA e sente. Busca por RAG.
from __future__ import annotations

import asyncio
from typing import Any

from src.data.database import Database
from src.data.rag import Embedder, RagHit, RagIndex


class NotesMemory:
    name = "notes"

    def __init__(self, db: Database, embedder: Embedder | None = None) -> None:
        self.index = RagIndex(db, "notes", embedder)

    def add(self, text: str, subject: str | None = None, importance: float | None = None,
            emotions: dict[str, float] | None = None) -> int:
        meta: dict[str, Any] = {}
        if subject:
            meta["subject"] = subject
        if importance is not None:
            meta["importance"] = importance
        if emotions:
            meta["emotions"] = emotions
        return self.index.add(text, meta)

    def list(self, limit: int = 50, offset: int = 0) -> list[RagHit]:
        return self.index.list(limit, offset)

    def delete(self, note_id: int) -> bool:
        return self.index.delete(note_id)

    async def search(self, query: str, top_k: int) -> list[str]:
        hits = await asyncio.to_thread(self.index.search, query, top_k)
        return [self.format(h) for h in hits]

    @staticmethod
    def format(hit: RagHit) -> str:
        subject = hit.meta.get("subject")
        return f"(sobre {subject}) {hit.text}" if subject else hit.text
