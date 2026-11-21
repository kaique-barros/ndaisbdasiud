# Eventos: RAG factual sobre o que ACONTECEU (descrições detalhadas), sem impressões da persona.
# As impressões pessoais ficam em memory/notes.
from __future__ import annotations

import asyncio

from src.data.database import Database
from src.data.rag import Embedder, RagHit, RagIndex


class EventsMemory:
    name = "events"

    def __init__(self, db: Database, embedder: Embedder | None = None) -> None:
        self.index = RagIndex(db, "events", embedder)

    def add(self, text: str, when: str | None = None, conversation_id: str | None = None) -> int:
        meta = {k: v for k, v in (("when", when), ("conversation_id", conversation_id)) if v}
        return self.index.add(text, meta)

    def list(self, limit: int = 50, offset: int = 0) -> list[RagHit]:
        return self.index.list(limit, offset)

    def delete(self, event_id: int) -> bool:
        return self.index.delete(event_id)

    async def search(self, query: str, top_k: int) -> list[str]:
        hits = await asyncio.to_thread(self.index.search, query, top_k)
        return [self.format(h) for h in hits]

    @staticmethod
    def format(hit: RagHit) -> str:
        when = hit.meta.get("when") or hit.created_at[:10]
        return f"[{when}] {hit.text}"
