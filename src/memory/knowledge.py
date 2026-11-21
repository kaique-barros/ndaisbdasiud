# Conhecimento da persona: graph-RAG com tudo o que ela aprendeu (fatos como relações entre
# entidades), com data de criação e de último acesso. Esta classe só chama o GraphRag e
# personaliza a consulta e a formatação da resposta.
from __future__ import annotations

import asyncio

from src.data.database import Database
from src.data.graphRag import Fact, GraphRag


class KnowledgeMemory:
    name = "knowledge"

    def __init__(self, db: Database) -> None:
        self.graph = GraphRag(db)

    def add_fact(self, subject: str, relation: str, obj: str, subject_type: str | None = None,
                 object_type: str | None = None) -> Fact:
        return self.graph.add_fact(subject, relation, obj, subject_type, object_type)

    def list(self, limit: int = 50, offset: int = 0) -> list[Fact]:
        return self.graph.list_facts(limit, offset)

    def delete(self, fact_id: int) -> bool:
        return self.graph.delete_fact(fact_id)

    async def search(self, query: str, top_k: int) -> list[str]:
        facts = await asyncio.to_thread(self.graph.search, query, top_k)
        return [self.format(f) for f in facts]

    @staticmethod
    def format(fact: Fact) -> str:
        return f"{fact.subject} — {fact.relation} — {fact.object}"
